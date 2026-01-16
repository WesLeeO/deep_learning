import os
from typing import Dict, List, Tuple

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

from src.data.addition_algo import BoardConfig
from src.data.problems import generate_diversified_problems
from src.data.board_dataset import BlackboardAdditionStepDataset
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
os.makedirs(CHECKPOINT_DIR, exist_ok=True)


TRAIN_3_4_DIGITS = True  #set False to train on 3 digits only


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


@torch.no_grad()
def evaluate_board_model(
    model: BlackboardTransformer,
    data_loader: DataLoader,
    device: torch.device,
    desc: str,
) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    total_correct = 0

    pbar = tqdm(data_loader, desc=desc)
    for batch in pbar:
        input_ids = batch["input_ids"].to(device)
        target_ids = batch["target_ids"].to(device)
        mask = batch["mask"].to(device)

        logits, _ = model(input_ids)
        loss = masked_cross_entropy(logits, target_ids, mask)

        b_correct, b_tokens = overall_masked_accuracy(logits, target_ids, mask)

        total_loss += loss.item() * b_tokens
        total_correct += b_correct
        total_tokens += b_tokens

        pbar.set_postfix(loss=loss.item(), acc=b_correct / max(b_tokens, 1))

    avg_loss = total_loss / max(total_tokens, 1)
    avg_acc = total_correct / max(total_tokens, 1)
    return avg_loss, avg_acc


def build_blackboard_model(pe_key: str, cfg: BoardConfig) -> BlackboardTransformer:
    d_model = 128
    n_heads = 4
    num_layers = 3
    dim_feedforward = 512
    max_len = cfg.H * cfg.W
    vocab_size = 12

    if pe_key == "relative_pe":
        pos_enc = RelativePositionBias2D(n_heads=n_heads, H=cfg.H, W=cfg.W)

    elif pe_key == "abs_1d_sinusoidal":
        pos_enc = SinusoidalPositionalEncoding(d_model=d_model, max_len=max_len)

    elif pe_key == "abs_1d_learned":
        pos_enc = LearnedPositionalEncoding1D(d_model=d_model, max_len=max_len)

    elif pe_key == "abs_2d_sinusoidal":
        pos_enc = SinusoidalPositionalEncoding2D(d_model=d_model, H=cfg.H, W=cfg.W)

    elif pe_key == "abs_2d_learned":
        pos_enc = LearnedPositionalEncoding2D(d_model=d_model, H=cfg.H, W=cfg.W)

    elif pe_key == "abs_2d_sin+rel_2d_bias":
        pos_enc = Abs2DPlusRelBias2D(
            abs_pe=SinusoidalPositionalEncoding2D(d_model=d_model, H=cfg.H, W=cfg.W),
            rel_bias=RelativePositionBias2D(n_heads=n_heads, H=cfg.H, W=cfg.W),
        )

    elif pe_key == "abs_2d_learn+rel_2d_bias":
        pos_enc = Abs2DPlusRelBias2D(
            abs_pe=LearnedPositionalEncoding2D(d_model=d_model, H=cfg.H, W=cfg.W),
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


def checkpoint_path(pe_key: str, cfg_train: BoardConfig, train_offset: int, train_mix_34: bool) -> str:
    mix_tag = "train3and4" if train_mix_34 else "train3only"
    return os.path.join(
        CHECKPOINT_DIR,
        f"localgen_blackboard_{mix_tag}_{pe_key}_H{cfg_train.H}_W{cfg_train.W}_trainoff{train_offset}.pt",
    )


def train_or_load_one(
    pe_key: str,
    cfg_train: BoardConfig,
    train_loader: DataLoader,
    val_loader: DataLoader,
    train_offset: int,
    TRAIN: bool,
    train_mix_34: bool,
    lr: float = 3e-4,
    num_epochs: int = 3,
) -> BlackboardTransformer:
    model = build_blackboard_model(pe_key, cfg_train)
    ckpt = checkpoint_path(pe_key, cfg_train, train_offset, train_mix_34=train_mix_34)

    if not TRAIN:
        if not os.path.isfile(ckpt):
            raise FileNotFoundError(
                f"Checkpoint not found: {ckpt}\n"
                f"-> Lance TRAIN=True une fois pour le créer."
            )
        state = torch.load(ckpt, map_location=DEVICE)
        model.load_state_dict(state)
        print(f"[LOAD] {pe_key} <- {ckpt}")
        return model

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss = 0.0
        total_tokens = 0
        total_correct = 0

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

            b_correct, b_tokens = overall_masked_accuracy(logits, target_ids, mask)
            total_loss += loss.item() * b_tokens
            total_correct += b_correct
            total_tokens += b_tokens

            pbar.set_postfix(loss=loss.item(), acc=b_correct / max(b_tokens, 1))

        train_loss = total_loss / max(total_tokens, 1)
        train_acc = total_correct / max(total_tokens, 1)

        val_loss, val_acc = evaluate_board_model(
            model, val_loader, DEVICE, desc=f"{pe_key} Epoch {epoch}/{num_epochs} [val]"
        )
        print(
            f"{pe_key} Epoch {epoch}/{num_epochs} | "
            f"train loss/token: {train_loss:.4f} | train acc: {train_acc:.4f} | "
            f"val loss/token: {val_loss:.4f} | val acc: {val_acc:.4f}"
        )
        print("-" * 80)

    torch.save(model.state_dict(), ckpt)
    print(f"[SAVE] {pe_key} -> {ckpt}")
    return model


def main():
    TRAIN = True
    print("Using device:", DEVICE)


    H_total = 8
    train_offset = 1

    # We want to optionally train on 3+4 digits; need a single fixed W for the model.
    max_train_digits = 4 if TRAIN_3_4_DIGITS else 3
    W = max_train_digits + 2  # important to keep one fixed sequence length

    # Local-generalization eval stays on 3 digits
    n_digits_eval = 3


    cfg_train = BoardConfig(
        H=H_total,
        W=W,
        n_digits=max_train_digits,
        carry_row=train_offset,
        top_row=train_offset + 1,
        bottom_row=train_offset + 2,
        result_row=train_offset + 3,
    )

    n_train_problems = 200_000
    n_val_problems = 2_000
    batch_size = 64
    num_epochs = 2
    lr = 3e-4


    if TRAIN:
        if not TRAIN_3_4_DIGITS:
            cfg3 = BoardConfig(
                H=H_total, W=W, n_digits=3,
                carry_row=train_offset, top_row=train_offset + 1,
                bottom_row=train_offset + 2, result_row=train_offset + 3,
            )
            train_problems = generate_diversified_problems(cfg3, n_train_problems, seed=0)
            val_problems = generate_diversified_problems(cfg3, n_val_problems, seed=1)

            train_ds = BlackboardAdditionStepDataset(train_problems)
            val_ds = BlackboardAdditionStepDataset(val_problems)

        else:
            n_each_train = n_train_problems // 2
            n_each_val = n_val_problems // 2

            cfg3 = BoardConfig(
                H=H_total, W=W, n_digits=3,
                carry_row=train_offset, top_row=train_offset + 1,
                bottom_row=train_offset + 2, result_row=train_offset + 3,
            )
            cfg4 = BoardConfig(
                H=H_total, W=W, n_digits=4,
                carry_row=train_offset, top_row=train_offset + 1,
                bottom_row=train_offset + 2, result_row=train_offset + 3,
            )

            train3 = generate_diversified_problems(cfg3, n_each_train, seed=0)
            train4 = generate_diversified_problems(cfg4, n_each_train, seed=1)

            val3 = generate_diversified_problems(cfg3, n_each_val, seed=10)
            val4 = generate_diversified_problems(cfg4, n_each_val, seed=11)

            train_ds3 = BlackboardAdditionStepDataset(train3)
            train_ds4 = BlackboardAdditionStepDataset(train4)
            val_ds3 = BlackboardAdditionStepDataset(val3)
            val_ds4 = BlackboardAdditionStepDataset(val4)

            train_ds = torch.utils.data.ConcatDataset([train_ds3, train_ds4])
            val_ds = torch.utils.data.ConcatDataset([val_ds3, val_ds4])

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    else:
        train_loader = None
        val_loader = None

    pe_variants: List[Tuple[str, str]] = [
        ("relative_pe", "Relative PE (2D bias)"),
        ("abs_1d_sinusoidal", "Sinusoidal PE (1D abs)"),
        ("abs_1d_learned", "Learned PE (1D abs)"),
        ("abs_2d_sinusoidal", "Sinusoidal PE (2D abs)"),
        ("abs_2d_learned", "Learned PE (2D abs)"),
        ("abs_2d_sin+rel_2d_bias", "Sin2D + RelBias2D"),
        ("abs_2d_learn+rel_2d_bias", "Learn2D + RelBias2D"),
    ]

    max_offset = H_total - 4
    offsets = list(range(0, max_offset + 1))

    all_accs: Dict[str, List[float]] = {}

    for pe_key, legend_name in pe_variants:
        print(f"\n==== {legend_name} ({pe_key}) ====")

        model = train_or_load_one(
            pe_key=pe_key,
            cfg_train=cfg_train,
            train_loader=train_loader,
            val_loader=val_loader,
            train_offset=train_offset,
            TRAIN=TRAIN,
            train_mix_34=TRAIN_3_4_DIGITS,
            lr=lr,
            num_epochs=num_epochs,
        )

        accs = []
        for offset in offsets:
            cfg_eval = BoardConfig(
                H=H_total,
                W=W,
                n_digits=n_digits_eval,
                carry_row=offset,
                top_row=offset + 1,
                bottom_row=offset + 2,
                result_row=offset + 3,
            )
            eval_problems = generate_diversified_problems(cfg_eval, n_val_problems, seed=100 + offset)
            eval_ds = BlackboardAdditionStepDataset(eval_problems)
            eval_loader = DataLoader(eval_ds, batch_size=batch_size, shuffle=False)

            _, eval_acc = evaluate_board_model(
                model, eval_loader, DEVICE, desc=f"{legend_name} offset={offset}"
            )
            accs.append(eval_acc)

        all_accs[legend_name] = accs

    plt.figure()
    for legend_name, accs in all_accs.items():
        plt.plot(offsets, accs, marker="o", label=legend_name)

    title = (
        "Local generalization after training on mixed 3+4 digits"
        if TRAIN_3_4_DIGITS
        else "Local generalization"
    )
    plt.title(title)
    plt.xlabel("Global block vertical offset (carry row index)")
    plt.ylabel("Overall masked accuracy")
    plt.ylim(0.0, 1.05)
    plt.grid(True)
    plt.legend()

    out_path = "local_generalization_all_pes_train3and4.png" if TRAIN_3_4_DIGITS else "local_generalization_all_pes.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved figure -> {out_path}")


if __name__ == "__main__":
    main()
