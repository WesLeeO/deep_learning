# Multi-addition transfer learning with last-layer finetuning (we use interpolated relative position biases).
import os
from typing import List, Tuple, Optional, Dict

import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

from src.data.addition_algo import BoardConfig
from src.data.multi_addition_algo import generate_multi_trajectory
from src.data.problems import generate_multi_addition_problems
from src.data.board_dataset import BlackboardMultiAdditionStepDataset
from src.models.transformers import BlackboardTransformer
from src.models.positional_encodings import (
    SinusoidalPositionalEncoding,
    LearnedPositionalEncoding1D,
    SinusoidalPositionalEncoding2D,
    LearnedPositionalEncoding2D,
    RelativePositionBias2D,
    Abs2DPlusRelBias2D,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BASE_DIR = os.path.dirname(__file__)
CHECKPOINT_DIR = os.path.join(BASE_DIR, "trained_weights")
OUTPUT_DIR = os.path.join(BASE_DIR, "attn_viz_multi_addition_transfer")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


def masked_cross_entropy(logits: torch.Tensor, target_ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    vocab_size = logits.size(-1)
    logits_flat = logits.reshape(-1, vocab_size)
    targets_flat = target_ids.reshape(-1)
    mask_flat = mask.reshape(-1)

    logits_sel = logits_flat[mask_flat]
    targets_sel = targets_flat[mask_flat]

    return F.cross_entropy(logits_sel, targets_sel)


def accuracy_with_splits(
    logits: torch.Tensor,
    target_ids: torch.Tensor,
    mask: torch.Tensor,
    cfg: BoardConfig,
) -> Tuple[int, int, int, int, int, int]:
    B, L, V = logits.shape
    device = logits.device

    preds = logits.argmax(dim=-1)
    correct = (preds == target_ids) & mask

    total_tokens = mask.sum().item()
    total_correct = correct.sum().item()

    positions = torch.arange(L, device=device)
    row_idx = positions // cfg.W

    carry_pos = (row_idx == cfg.carry_row).unsqueeze(0).expand_as(mask)
    digit_pos = (row_idx == cfg.result_row).unsqueeze(0).expand_as(mask)

    carry_tokens = (mask & carry_pos).sum().item()
    digit_tokens = (mask & digit_pos).sum().item()

    carry_correct = (correct & carry_pos).sum().item()
    digit_correct = (correct & digit_pos).sum().item()

    return (
        total_correct,
        total_tokens,
        carry_correct,
        carry_tokens,
        digit_correct,
        digit_tokens,
    )


def build_blackboard_model(pe_key: str, cfg: BoardConfig) -> BlackboardTransformer:
    d_model = 128
    n_heads = 4
    num_layers = 4
    dim_feedforward = 512
    max_len = cfg.H * cfg.W
    vocab_size = 12

    if pe_key == "relative_pe":
        pos_enc = RelativePositionBias2D(n_heads=n_heads, H=cfg.H, W=cfg.W)

    elif pe_key == "abs_1d_sinusoidal":
        pos_enc = SinusoidalPositionalEncoding(d_model=D_MODEL, max_len=max_len)

    elif pe_key == "abs_2d_sinusoidal":
        pos_enc = SinusoidalPositionalEncoding2D(d_model=D_MODEL, H=cfg.H, W=cfg.W)

    elif pe_key == "abs_1d_learned":
        pos_enc = LearnedPositionalEncoding1D(d_model=D_MODEL, max_len=max_len)

    elif pe_key == "abs_2d_learned":
        pos_enc = LearnedPositionalEncoding2D(d_model=D_MODEL, H=cfg.H, W=cfg.W)

    elif pe_key == "abs_2d_sin+rel_2d_bias":
        pos_enc = Abs2DPlusRelBias2D(
            abs_pe=SinusoidalPositionalEncoding2D(d_model=D_MODEL, H=cfg.H, W=cfg.W),
            rel_bias=RelativePositionBias2D(n_heads=n_heads, H=cfg.H, W=cfg.W),
        )
    else:
        raise ValueError(f"Unknown PE key: {pe_key}")

    model = BlackboardTransformer(
        vocab_size=vocab_size,
        d_model=d_model,
        nhead=n_heads,
        num_layers=num_layers,
        dim_feedforward=dim_feedforward,
        max_len=max_len,
        dropout=0.1,
        pos_enc=pos_enc,
    ).to(DEVICE)

    return model


def interpolate_rel_axis(
    old_weight: torch.Tensor,
    H_old: int,
    H_new: int,
) -> torch.Tensor:
    n_old, n_heads = old_weight.shape
    assert n_old == 2 * H_old - 1
    n_new = 2 * H_new - 1

    new_weight = torch.empty(n_new, n_heads, device=old_weight.device, dtype=old_weight.dtype)

    if H_old == H_new:
        new_weight.copy_(old_weight)
        return new_weight

    for i_new in range(n_new):
        dx_new = i_new - (H_new - 1)
        if H_new > 1:
            dx_old_cont = dx_new * (H_old - 1) / float(H_new - 1)
        else:
            dx_old_cont = 0.0

        x = dx_old_cont + (H_old - 1)
        x = max(0.0, min(float(n_old - 1), x))

        i0 = int(torch.floor(torch.tensor(x)).item())
        i1 = min(i0 + 1, n_old - 1)
        alpha = x - i0

        new_weight[i_new] = (1.0 - alpha) * old_weight[i0] + alpha * old_weight[i1]

    return new_weight


def load_multi_addition_from_addition(
    pe_key: str,
    cfg_add: BoardConfig,
    cfg_multi: BoardConfig,
) -> BlackboardTransformer:
    model_add = build_blackboard_model(pe_key, cfg_add)
    ckpt_add = os.path.join(CHECKPOINT_DIR, f"blackboard_{pe_key}.pt")
    state_add = torch.load(ckpt_add, map_location=DEVICE)
    model_add.load_state_dict(state_add)

    model_multi = build_blackboard_model(pe_key, cfg_multi)
    state_multi = model_multi.state_dict()

    for name, param in model_add.state_dict().items():
        if name.startswith("pos_enc.rel_height.weight") or name.startswith("pos_enc.rel_width.weight"):
            continue
        if name in state_multi and state_multi[name].shape == param.shape:
            state_multi[name] = param

    if isinstance(model_add.pos_enc, RelativePositionBias2D) and isinstance(model_multi.pos_enc, RelativePositionBias2D):
        H_old = model_add.pos_enc.H
        H_new = model_multi.pos_enc.H
        W_old = model_add.pos_enc.W
        W_new = model_multi.pos_enc.W

        old_rel_height = model_add.pos_enc.rel_height.weight.data
        old_rel_width = model_add.pos_enc.rel_width.weight.data

        new_rel_height = interpolate_rel_axis(old_rel_height, H_old, H_new)
        new_rel_width = interpolate_rel_axis(old_rel_width, W_old, W_new)

        state_multi["pos_enc.rel_height.weight"] = new_rel_height
        state_multi["pos_enc.rel_width.weight"] = new_rel_width

    model_multi.load_state_dict(state_multi)
    return model_multi


def freeze_all_but_last_layer(model: BlackboardTransformer) -> None:
    for param in model.parameters():
        param.requires_grad = False
    for param in model.layers[-1].parameters():
        param.requires_grad = True
    for param in model.output_proj.parameters():
        param.requires_grad = True


def train_or_load_multi_addition_models(
    FINETUNE: bool,
    cfg_add: BoardConfig,
    cfg_multi: BoardConfig,
) -> Dict[str, BlackboardTransformer]:
    pe_keys = ["relative_pe"]
    models: Dict[str, BlackboardTransformer] = {}

    n_train_problems = 200_000
    n_val_problems = 5_000
    batch_size = 64
    num_epochs = 2
    lr = 3e-4

    if FINETUNE:
        train_problems = generate_multi_addition_problems(cfg_multi, n_train_problems, seed=20)
        val_problems = generate_multi_addition_problems(cfg_multi, n_val_problems, seed=21)

        train_ds = BlackboardMultiAdditionStepDataset(train_problems)
        val_ds = BlackboardMultiAdditionStepDataset(val_problems)

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    else:
        train_loader = None
        val_loader = None

    for pe_key in pe_keys:
        print(f"==== {pe_key} (multi-addition transfer) ====")
        ckpt_multi = os.path.join(CHECKPOINT_DIR, f"blackboard_{pe_key}_multiadd_lastlayer.pt")

        if FINETUNE:
            model = load_multi_addition_from_addition(pe_key, cfg_add, cfg_multi)
            freeze_all_but_last_layer(model)

            optimizer = torch.optim.Adam(
                [p for p in model.parameters() if p.requires_grad],
                lr=lr,
            )

            for epoch in range(1, num_epochs + 1):
                model.train()
                total_loss = 0.0
                total_tokens = 0
                total_correct = 0
                total_carry_correct = 0
                total_carry_tokens = 0
                total_digit_correct = 0
                total_digit_tokens = 0

                pbar = tqdm(train_loader, desc=f"{pe_key} Epoch {epoch}/{num_epochs} [train multi-add]")
                for batch in pbar:
                    input_ids = batch["input_ids"].to(DEVICE)
                    target_ids = batch["target_ids"].to(DEVICE)
                    mask = batch["mask"].to(DEVICE)

                    optimizer.zero_grad()
                    logits, _ = model(input_ids)

                    loss = masked_cross_entropy(logits, target_ids, mask)
                    loss.backward()
                    optimizer.step()

                    batch_tokens = mask.sum().item()
                    batch_loss = loss.item()

                    (
                        b_total_correct,
                        b_total_tokens,
                        b_carry_correct,
                        b_carry_tokens,
                        b_digit_correct,
                        b_digit_tokens,
                    ) = accuracy_with_splits(logits, target_ids, mask, cfg_multi)

                    total_loss += batch_loss * batch_tokens
                    total_tokens += batch_tokens
                    total_correct += b_total_correct
                    total_carry_correct += b_carry_correct
                    total_carry_tokens += b_carry_tokens
                    total_digit_correct += b_digit_correct
                    total_digit_tokens += b_digit_tokens

                    batch_acc = b_total_correct / max(b_total_tokens, 1)
                    pbar.set_postfix(loss=batch_loss, acc=batch_acc)

                avg_loss = total_loss / max(total_tokens, 1)
                avg_acc = total_correct / max(total_tokens, 1)
                carry_acc = (
                    total_carry_correct / total_carry_tokens
                    if total_carry_tokens > 0
                    else 0.0
                )
                digit_acc = (
                    total_digit_correct / total_digit_tokens
                    if total_digit_tokens > 0
                    else 0.0
                )

                print(
                    f"{pe_key} Epoch {epoch}/{num_epochs} "
                    f"| train loss/token: {avg_loss:.4f} "
                    f"| train acc(masked): {avg_acc:.4f} "
                    f"| train carry acc: {carry_acc:.4f} "
                    f"| train digit acc: {digit_acc:.4f}"
                )

                model.eval()
                val_loss = 0.0
                val_tokens = 0
                val_correct = 0
                val_carry_correct = 0
                val_carry_tokens = 0
                val_digit_correct = 0
                val_digit_tokens = 0

                pbar_val = tqdm(val_loader, desc=f"{pe_key} Epoch {epoch}/{num_epochs} [val multi-add]")
                with torch.no_grad():
                    for batch in pbar_val:
                        input_ids = batch["input_ids"].to(DEVICE)
                        target_ids = batch["target_ids"].to(DEVICE)
                        mask = batch["mask"].to(DEVICE)

                        logits, _ = model(input_ids)
                        loss = masked_cross_entropy(logits, target_ids, mask)

                        batch_tokens = mask.sum().item()
                        batch_loss = loss.item()

                        (
                            b_total_correct,
                            b_total_tokens,
                            b_carry_correct,
                            b_carry_tokens,
                            b_digit_correct,
                            b_digit_tokens,
                        ) = accuracy_with_splits(logits, target_ids, mask, cfg_multi)

                        val_loss += batch_loss * batch_tokens
                        val_tokens += batch_tokens
                        val_correct += b_total_correct
                        val_carry_correct += b_carry_correct
                        val_carry_tokens += b_carry_tokens
                        val_digit_correct += b_digit_correct
                        val_digit_tokens += b_digit_tokens

                        batch_acc = b_total_correct / max(b_total_tokens, 1)
                        pbar_val.set_postfix(loss=batch_loss, acc=batch_acc)

                val_avg_loss = val_loss / max(val_tokens, 1)
                val_avg_acc = val_correct / max(val_tokens, 1)
                val_carry_acc = (
                    val_carry_correct / val_carry_tokens
                    if val_carry_tokens > 0
                    else 0.0
                )
                val_digit_acc = (
                    val_digit_correct / val_digit_tokens
                    if val_digit_tokens > 0
                    else 0.0
                )

                print(
                    f"{pe_key} Epoch {epoch}/{num_epochs} "
                    f"| val loss/token: {val_avg_loss:.4f} "
                    f"| val acc(masked): {val_avg_acc:.4f} "
                    f"| val carry acc: {val_carry_acc:.4f} "
                    f"| val digit acc: {val_digit_acc:.4f}"
                )

            torch.save(model.state_dict(), ckpt_multi)
            print(f"Saved multi-addition-transfer checkpoint to {ckpt_multi}")
        else:
            if not os.path.isfile(ckpt_multi):
                raise FileNotFoundError(
                    f"Multi-addition-transfer checkpoint not found for {pe_key}: {ckpt_multi}. "
                    f"Set FINETUNE=True once to create it."
                )
            model = build_blackboard_model(pe_key, cfg_multi)
            state = torch.load(ckpt_multi, map_location=DEVICE)
            model.load_state_dict(state)
            print(f"Loaded multi-addition-transfer checkpoint from {ckpt_multi}")

        models[pe_key] = model

    return models


def build_multi_addition_examples(cfg: BoardConfig) -> List[Tuple[str, np.ndarray]]:
    examples = [
        ("no_carry_3add", np.array([111, 222, 333], dtype=np.int64)),
        ("single_carry_3add", np.array([129, 232, 139], dtype=np.int64)),
        ("carry_chain_3add", np.array([199, 299, 502], dtype=np.int64)),
        ("full_carry_chain_3add", np.array([999, 1, 1], dtype=np.int64)),
    ]
    return examples


def board_to_input_tensor(board: np.ndarray) -> torch.Tensor:
    x = torch.from_numpy(board.astype(np.int64)).view(-1)
    return x.unsqueeze(0).to(DEVICE)


def query_indices_for_step(cfg: BoardConfig, step: int) -> Tuple[int, Optional[int]]:
    col_end = cfg.W - 1
    col = col_end - step

    result_row = cfg.result_row
    result_idx = result_row * cfg.W + col

    carry_idx: Optional[int] = None
    next_col = col - 1
    if next_col >= 0:
        carry_row = cfg.carry_row
        carry_idx = carry_row * cfg.W + next_col

    return result_idx, carry_idx


def plot_attention_grid(
    attn_layers: List[torch.Tensor],
    cfg: BoardConfig,
    q_idx: int,
    title: str,
    out_path: str,
) -> None:
    num_layers = len(attn_layers)
    if num_layers == 0:
        raise ValueError("No attention layers provided.")

    B, n_heads, L, _ = attn_layers[0].shape
    assert B == 1
    assert L == cfg.H * cfg.W

    fig, axes = plt.subplots(
        num_layers,
        n_heads,
        figsize=(3 * n_heads, 3 * num_layers),
        squeeze=False,
    )

    vmin, vmax = 0.0, 1.0

    for layer_idx, attn in enumerate(attn_layers):
        attn_layer = attn[0]
        for head_idx in range(n_heads):
            A = attn_layer[head_idx]
            a_q = A[q_idx].detach().cpu().numpy()
            heatmap = a_q.reshape(cfg.H, cfg.W)

            ax = axes[layer_idx][head_idx]
            im = ax.imshow(heatmap, origin="upper", vmin=vmin, vmax=vmax)
            q_row = q_idx // cfg.W
            q_col = q_idx % cfg.W
            ax.scatter(
                q_col,
                q_row,
                marker="s",
                edgecolor="black",
                facecolor="none",
                s=60,
            )

            ax.set_xticks(range(cfg.W))
            ax.set_yticks(range(cfg.H))
            ax.set_xlabel("col")
            ax.set_ylabel("row")
            ax.set_title(f"L{layer_idx} H{head_idx}")

    fig.suptitle(title)
    fig.tight_layout()
    fig.subplots_adjust(top=0.92)

    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cbar_ax)

    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def visualize_attention(models: Dict[str, BlackboardTransformer], cfg: BoardConfig) -> None:
    examples = build_multi_addition_examples(cfg)

    for pe_key, model in models.items():
        model.eval()
        for ex_name, xs in examples:
            S_seq, M_seq = generate_multi_trajectory(cfg, xs)

            step = cfg.n_digits - 1
            S_t = S_seq[step]
            input_ids = board_to_input_tensor(S_t)

            with torch.no_grad():
                logits, attn_layers = model(input_ids, return_attn=True)

            result_idx, carry_idx = query_indices_for_step(cfg, step)

            if result_idx is not None:
                title = f"{pe_key} | multi-addition | {ex_name} | step={step} | query=result"
                out_path = os.path.join(
                    OUTPUT_DIR, f"attn_multi_{pe_key}_{ex_name}_step{step}_result.png"
                )
                plot_attention_grid(attn_layers, cfg, result_idx, title, out_path)

            if carry_idx is not None:
                title = f"{pe_key} | multi-addition | {ex_name} | step={step} | query=carry"
                out_path = os.path.join(
                    OUTPUT_DIR, f"attn_multi_{pe_key}_{ex_name}_step{step}_carry.png"
                )
                plot_attention_grid(attn_layers, cfg, carry_idx, title, out_path)


def main():
    cfg_add = BoardConfig(H=4, W=5, n_digits=3)
    cfg_multi = BoardConfig(
        H=5,
        W=5,
        n_digits=3,
        n_addends=3,
        carry_row=0,
        top_row=1,
        bottom_row=3,
        result_row=4,
    )

    FINETUNE = True

    models = train_or_load_multi_addition_models(FINETUNE, cfg_add, cfg_multi)
    visualize_attention(models, cfg_multi)


if __name__ == "__main__":
    main()
