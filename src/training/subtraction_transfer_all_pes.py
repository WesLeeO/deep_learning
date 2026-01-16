# Multi-subtraction transfer learning with last-layer finetuning across various positional encodings.
import os
from typing import List, Tuple, Optional, Dict

import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

from src.data.addition_algo import BoardConfig
from src.data.subtraction_algo import generate_trajectory_variant_A as generate_subtraction_trajectory
from src.data.problems import generate_subtraction_problems
from src.data.board_dataset import BlackboardSubtractionStepDataset
from src.models.transformers import BlackboardTransformer
from src.models.positional_encodings import (
    RelativePositionBias2D,
    SinusoidalPositionalEncoding,
    LearnedPositionalEncoding1D,
    SinusoidalPositionalEncoding2D,
    LearnedPositionalEncoding2D,
    Abs2DPlusRelBias2D,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CHECKPOINT_DIR = "src/training/trained_weights"
OUTPUT_DIR = "attn_viz_subtraction_transfer"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


N_HEADS = 4
D_MODEL = 128
NUM_LAYERS = 3
DIM_FEEDFORWARD = 512
DROPOUT = 0.1
VOCAB_SIZE = 12

PE_KEYS: List[str] = [
    "relative_pe",
    "abs_1d_sinusoidal",
    "abs_2d_sinusoidal",
    "abs_1d_learned",
    "abs_2d_learned",
    "abs_2d_sin+rel_2d_bias",
]

PE_LABELS: Dict[str, str] = {
    "relative_pe": "Relative PE (2D bias)",
    "abs_1d_sinusoidal": "Sinusoidal PE (1D abs)",
    "abs_2d_sinusoidal": "Sinusoidal PE (2D abs)",
    "abs_1d_learned": "Learned PE (1D abs)",
    "abs_2d_learned": "Learned PE (2D abs)",
    "abs_2d_sin+rel_2d_bias": "Sin2D + RelBias2D",
}


def masked_cross_entropy(logits: torch.Tensor, target_ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    vocab_size = logits.size(-1)
    logits_flat = logits.reshape(-1, vocab_size)
    targets_flat = target_ids.reshape(-1)
    mask_flat = mask.reshape(-1)

    logits_sel = logits_flat[mask_flat]
    targets_sel = targets_flat[mask_flat]
    return F.cross_entropy(logits_sel, targets_sel)


def overall_masked_accuracy(logits: torch.Tensor, target_ids: torch.Tensor, mask: torch.Tensor) -> Tuple[int, int]:
    preds = logits.argmax(dim=-1)
    correct = (preds == target_ids) & mask
    return correct.sum().item(), mask.sum().item()


def build_blackboard_model(pe_key: str, cfg: BoardConfig) -> BlackboardTransformer:
    max_len = cfg.H * cfg.W

    if pe_key == "relative_pe":
        pos_enc = RelativePositionBias2D(n_heads=N_HEADS, H=cfg.H, W=cfg.W)

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
            rel_bias=RelativePositionBias2D(n_heads=N_HEADS, H=cfg.H, W=cfg.W),
        )

    else:
        raise ValueError(f"Unknown PE key: {pe_key}")

    model = BlackboardTransformer(
        vocab_size=VOCAB_SIZE,
        d_model=D_MODEL,
        nhead=N_HEADS,
        num_layers=NUM_LAYERS,
        dim_feedforward=DIM_FEEDFORWARD,
        max_len=max_len,
        dropout=DROPOUT,
        pos_enc=pos_enc,
    ).to(DEVICE)

    return model


def load_addition_checkpoint(pe_key: str, cfg: BoardConfig) -> BlackboardTransformer:
    """
    Loads the addition-trained checkpoint created by your first module:
    src/training/trained_weights/blackboard_{pe_key}.pt
    """
    ckpt_path = os.path.join(CHECKPOINT_DIR, f"blackboard_{pe_key}.pt")
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Addition checkpoint not found: {ckpt_path}")

    model = build_blackboard_model(pe_key, cfg)
    state = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(state, strict=True)
    print(f"Loaded addition base checkpoint <- {ckpt_path}")
    return model


def freeze_all_but_last_layer(model: BlackboardTransformer) -> None:
    for p in model.parameters():
        p.requires_grad = False
    # last transformer block + output head
    for p in model.layers[-1].parameters():
        p.requires_grad = True
    for p in model.output_proj.parameters():
        p.requires_grad = True


@torch.no_grad()
def evaluate(model: BlackboardTransformer, loader: DataLoader, desc: str) -> float:
    model.eval()
    total_correct, total_tokens = 0, 0
    pbar = tqdm(loader, desc=desc)
    for batch in pbar:
        input_ids = batch["input_ids"].to(DEVICE)
        target_ids = batch["target_ids"].to(DEVICE)
        mask = batch["mask"].to(DEVICE)

        logits, _ = model(input_ids)
        c, t = overall_masked_accuracy(logits, target_ids, mask)
        total_correct += c
        total_tokens += t
        pbar.set_postfix(acc=total_correct / max(total_tokens, 1))
    return total_correct / max(total_tokens, 1)


def finetune_or_load_subtraction(
    FINETUNE: bool,
    cfg: BoardConfig,
) -> Tuple[Dict[str, BlackboardTransformer], Dict[str, float]]:
    models: Dict[str, BlackboardTransformer] = {}
    val_acc_per_pe: Dict[str, float] = {}

    n_train_problems = 200_000
    n_val_problems = 5_000
    batch_size = 64
    num_epochs = 2
    lr = 3e-4

    if FINETUNE:
        train_problems = generate_subtraction_problems(cfg, n_train_problems, seed=10)
        val_problems = generate_subtraction_problems(cfg, n_val_problems, seed=11)

        train_ds = BlackboardSubtractionStepDataset(train_problems)
        val_ds = BlackboardSubtractionStepDataset(val_problems)

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    else:
        train_loader = None
        val_problems = generate_subtraction_problems(cfg, n_val_problems, seed=11)
        val_ds = BlackboardSubtractionStepDataset(val_problems)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    for pe_key in PE_KEYS:
        label = PE_LABELS.get(pe_key, pe_key)
        ckpt_sub_path = os.path.join(CHECKPOINT_DIR, f"blackboard_{pe_key}_subtraction_lastlayer.pt")

        print(f"\n==== {label} | transfer addition -> subtraction (4 heads) ====")

        if FINETUNE:
            model = load_addition_checkpoint(pe_key, cfg)
            freeze_all_but_last_layer(model)

            optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)

            for epoch in range(1, num_epochs + 1):
                model.train()
                total_correct, total_tokens = 0, 0

                pbar = tqdm(train_loader, desc=f"{label} Epoch {epoch}/{num_epochs} [train subtraction]")
                for batch in pbar:
                    input_ids = batch["input_ids"].to(DEVICE)
                    target_ids = batch["target_ids"].to(DEVICE)
                    mask = batch["mask"].to(DEVICE)

                    optimizer.zero_grad()
                    logits, _ = model(input_ids)
                    loss = masked_cross_entropy(logits, target_ids, mask)
                    loss.backward()
                    optimizer.step()

                    c, t = overall_masked_accuracy(logits, target_ids, mask)
                    total_correct += c
                    total_tokens += t
                    pbar.set_postfix(loss=loss.item(), acc=total_correct / max(total_tokens, 1))

                val_acc = evaluate(model, val_loader, desc=f"{label} Epoch {epoch}/{num_epochs} [val subtraction]")
                print(f"{label} Epoch {epoch}/{num_epochs} | val acc(masked): {val_acc:.4f}")
                print("-" * 80)

            torch.save(model.state_dict(), ckpt_sub_path)
            print(f"Saved subtraction-transfer checkpoint -> {ckpt_sub_path}")

        else:
            if not os.path.isfile(ckpt_sub_path):
                raise FileNotFoundError(
                    f"Subtraction-transfer checkpoint not found: {ckpt_sub_path}\n"
                    f"-> Set FINETUNE_SUBTRACTION=True once to create it."
                )
            model = build_blackboard_model(pe_key, cfg)
            state = torch.load(ckpt_sub_path, map_location=DEVICE)
            model.load_state_dict(state, strict=True)
            print(f"Loaded subtraction-transfer checkpoint <- {ckpt_sub_path}")

        final_val_acc = evaluate(model, val_loader, desc=f"{label} [final eval]")
        val_acc_per_pe[pe_key] = final_val_acc
        models[pe_key] = model

    return models, val_acc_per_pe


def plot_pe_barplot(val_acc_per_pe: Dict[str, float]) -> None:
    pe_keys = list(val_acc_per_pe.keys())
    labels = [PE_LABELS.get(k, k) for k in pe_keys]
    accs = [val_acc_per_pe[k] for k in pe_keys]

    plt.figure(figsize=(8, 4))
    plt.bar(labels, accs)
    plt.ylabel("Validation accuracy (subtraction transfer)")
    plt.title("Subtraction transfer (4 heads) | accuracy after fine-tuning")
    plt.ylim(0.0, 1.0)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()

    out_path = os.path.join(CHECKPOINT_DIR, "subtraction_transfer_barplot_all_pes.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved barplot -> {out_path}")



def build_subtraction_examples(cfg: BoardConfig) -> List[Tuple[str, np.ndarray]]:
    return [
        ("no_borrow", np.array([765, 123], dtype=np.int64)),
        ("single_borrow_units", np.array([302, 129], dtype=np.int64)),
        ("borrow_chain", np.array([400, 199], dtype=np.int64)),
        ("full_borrow_chain", np.array([1000 - 1, 1], dtype=np.int64)),
    ]


def board_to_input_tensor(board: np.ndarray) -> torch.Tensor:
    x = torch.from_numpy(board.astype(np.int64)).view(-1)
    return x.unsqueeze(0).to(DEVICE)


def query_indices_for_step(cfg: BoardConfig, step: int) -> Tuple[int, Optional[int]]:
    col_end = cfg.W - 1
    col = col_end - step
    result_idx = cfg.result_row * cfg.W + col

    carry_idx: Optional[int] = None
    next_col = col - 1
    if next_col >= 0:
        carry_idx = cfg.carry_row * cfg.W + next_col

    return result_idx, carry_idx


def plot_attention_grid(attn_layers, cfg, q_idx, title, out_path):
    num_layers = len(attn_layers)
    B, n_heads, L, _ = attn_layers[0].shape
    assert B == 1 and n_heads == N_HEADS and L == cfg.H * cfg.W

    fig, axes = plt.subplots(num_layers, n_heads, figsize=(3 * n_heads, 3 * num_layers), squeeze=False)
    vmin, vmax = 0.0, 1.0

    for layer_idx, attn in enumerate(attn_layers):
        attn_layer = attn[0]
        for head_idx in range(n_heads):
            A = attn_layer[head_idx]
            a_q = A[q_idx].detach().cpu().numpy().reshape(cfg.H, cfg.W)

            ax = axes[layer_idx][head_idx]
            im = ax.imshow(a_q, origin="upper", vmin=vmin, vmax=vmax)

            q_row, q_col = q_idx // cfg.W, q_idx % cfg.W
            ax.scatter(q_col, q_row, marker="s", edgecolor="black", facecolor="none", s=60)
            ax.set_xticks(range(cfg.W))
            ax.set_yticks(range(cfg.H))
            ax.set_title(f"L{layer_idx} H{head_idx}")

    fig.suptitle(title)
    fig.tight_layout()
    fig.subplots_adjust(top=0.92)
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cbar_ax)

    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def visualize_attention(models: Dict[str, BlackboardTransformer], cfg: BoardConfig) -> None:
    examples = build_subtraction_examples(cfg)

    for pe_key, model in models.items():
        model.eval()
        for ex_name, xs in examples:
            S_seq, _ = generate_subtraction_trajectory(cfg, xs)
            step = cfg.n_digits - 1
            input_ids = board_to_input_tensor(S_seq[step])

            with torch.no_grad():
                _, attn_layers = model(input_ids, return_attn=True)

            result_idx, carry_idx = query_indices_for_step(cfg, step)

            label = pe_key
            if result_idx is not None:
                title = f"{label} | subtraction | {ex_name} | heads={N_HEADS} | step={step} | query=result"
                out_path = os.path.join(OUTPUT_DIR, f"attn_sub_{label}_{ex_name}_step{step}_result.png")
                plot_attention_grid(attn_layers, cfg, result_idx, title, out_path)

            if carry_idx is not None:
                title = f"{label} | subtraction | {ex_name} | heads={N_HEADS} | step={step} | query=carry"
                out_path = os.path.join(OUTPUT_DIR, f"attn_sub_{label}_{ex_name}_step{step}_carry.png")
                plot_attention_grid(attn_layers, cfg, carry_idx, title, out_path)


def main():
    cfg = BoardConfig(H=4, W=5, n_digits=3)

    FINETUNE_SUBTRACTION = True  # True: run finetune and save; False: load finetuned ckpts and plot
    VIS_ATTENTION = True       

    models, val_acc_per_pe = finetune_or_load_subtraction(FINETUNE_SUBTRACTION, cfg)
    plot_pe_barplot(val_acc_per_pe)

    if VIS_ATTENTION:
        visualize_attention(models, cfg)


if __name__ == "__main__":
    main()
