import os
from typing import List, Tuple, Optional, Dict

import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

from src.data.addition_algo import BoardConfig, generate_trajectory_variant_A
from src.data.problems import generate_diversified_problems
from src.data.board_dataset import BlackboardAdditionStepDataset
from src.models.transformers import BlackboardTransformer
from src.models.positional_encodings import (
    SinusoidalPositionalEncoding,
    LearnedPositionalEncoding1D,
    SinusoidalPositionalEncoding2D,
    LearnedPositionalEncoding2D,
    RelativePositionBias2D,
    Abs2DPlusRelBias2D,
    RotaryPositionalEmbedding1D

)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CHECKPOINT_DIR = "src/training/trained_weights"
OUTPUT_DIR = "attn_viz_1head_3layers_all_PEs"
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
    n_heads = 3
    num_layers = 3
    dim_feedforward = 512
    max_len = cfg.H * cfg.W
    vocab_size = 12




    if pe_key == "relative_pe":
        pos_enc = RelativePositionBias2D(n_heads, cfg.H, cfg.W)
    elif pe_key == "abs_1d_sinusoidal":
        pos_enc = SinusoidalPositionalEncoding(d_model, max_len=max_len)
    elif pe_key == "abs_2d_sinusoidal":
        pos_enc = SinusoidalPositionalEncoding2D(d_model, cfg.H, cfg.W)

    elif pe_key == "abs_1d_learned":
        pos_enc = LearnedPositionalEncoding1D(d_model, cfg.H * cfg.W)
    elif pe_key == "abs_2d_learned":
        pos_enc = LearnedPositionalEncoding2D(d_model, cfg.H , cfg.W)

    elif pe_key == "abs_2d_sin+rel_2d_bias":
        pos_enc = Abs2DPlusRelBias2D(
            abs_pe=SinusoidalPositionalEncoding2D(d_model, cfg.H, cfg.W),
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




def train_or_load_models(TRAIN: bool, cfg: BoardConfig) -> Dict[str, BlackboardTransformer]:
    pe_keys = [
            "relative_pe",
            "abs_1d_sinusoidal",
            "abs_2d_sinusoidal",
            "abs_1d_learned",
            "abs_2d_learned",
            "abs_2d_sin+rel_2d_bias",
            ]

    models: Dict[str, BlackboardTransformer] = {}

    n_train_problems = 500_000
    n_val_problems = 5_000
    batch_size = 64
    num_epochs = 3
    lr = 3e-4

    if TRAIN:
        train_problems = generate_diversified_problems(cfg, n_train_problems, seed=0)
        val_problems = generate_diversified_problems(cfg, n_val_problems, seed=1)

        train_ds = BlackboardAdditionStepDataset(train_problems)
        val_ds = BlackboardAdditionStepDataset(val_problems)

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    else:
        train_loader = None
        val_loader = None

    for pe_key in pe_keys:
        print(f"==== {pe_key} ====")
        model = build_blackboard_model(pe_key, cfg)
        ckpt_path = os.path.join(CHECKPOINT_DIR, f"blackboard_{pe_key}.pt")

        if TRAIN:
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)

            for epoch in range(1, num_epochs + 1):
                model.train()
                total_loss = 0.0
                total_tokens = 0
                total_correct = 0
                total_carry_correct = 0
                total_carry_tokens = 0
                total_digit_correct = 0
                total_digit_tokens = 0

                pbar = tqdm(train_loader, desc=f"{pe_key} Epoch {epoch}/{num_epochs} [train]")
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
                    ) = accuracy_with_splits(logits, target_ids, mask, cfg)

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

                pbar_val = tqdm(val_loader, desc=f"{pe_key} Epoch {epoch}/{num_epochs} [val]")
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
                        ) = accuracy_with_splits(logits, target_ids, mask, cfg)

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

            torch.save(model.state_dict(), ckpt_path)
            print(f"Saved checkpoint to {ckpt_path}")
        else:
            if not os.path.isfile(ckpt_path):
                raise FileNotFoundError(
                    f"Checkpoint not found for {pe_key}: {ckpt_path}. "
                    f"Set TRAIN=True once to create it."
                )
            state = torch.load(ckpt_path, map_location=DEVICE)
            model.load_state_dict(state)
            print(f"Loaded checkpoint from {ckpt_path}")

        models[pe_key] = model

    return models


def build_examples(cfg: BoardConfig) -> List[Tuple[str, np.ndarray]]:
    examples = [
        ("no_carry", np.array([123, 456], dtype=np.int64)),
        ("single_carry_units", np.array([129, 432], dtype=np.int64)),
        ("carry_chain", np.array([199, 812], dtype=np.int64)),
        ("full_carry_chain", np.array([999, 1], dtype=np.int64)),
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
        figsize=(12, 3.2 * num_layers),
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

    fig.tight_layout(rect=[0.0, 0.0, 0.86, 0.92])
    fig.subplots_adjust(top=0.92)


    cbar_ax = fig.add_axes([0.89, 0.15, 0.02, 0.7]) 
    fig.colorbar(im, cax=cbar_ax)

    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def visualize_attention(models: Dict[str, BlackboardTransformer], cfg: BoardConfig) -> None:
    examples = build_examples(cfg)

    for pe_key, model in models.items():
        model.eval()
        for ex_name, xs in examples:
            S_seq, M_seq = generate_trajectory_variant_A(cfg, xs)

            step = cfg.n_digits - 1
            S_t = S_seq[step]
            input_ids = board_to_input_tensor(S_t)

            with torch.no_grad():
                logits, attn_layers = model(input_ids, return_attn=True)

            result_idx, carry_idx = query_indices_for_step(cfg, step)

            if result_idx is not None:
                title = f"{pe_key} | {ex_name} | step={step} | query=result"
                out_path = os.path.join(
                    OUTPUT_DIR, f"attn_{pe_key}_{ex_name}_step{step}_result.png"
                )
                plot_attention_grid(attn_layers, cfg, result_idx, title, out_path)

            if carry_idx is not None:
                title = f"{pe_key} | {ex_name} | step={step} | query=carry"
                out_path = os.path.join(
                    OUTPUT_DIR, f"attn_{pe_key}_{ex_name}_step{step}_carry.png"
                )
                plot_attention_grid(attn_layers, cfg, carry_idx, title, out_path)


def main():
    cfg = BoardConfig(H=4, W=5, n_digits=3)

    TRAIN = True  # Set to True to train models, False to load existing checkpoints

    models = train_or_load_models(TRAIN, cfg)
    visualize_attention(models, cfg)


if __name__ == "__main__":
    main()
