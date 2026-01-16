
# Multi-addition transfer learning from large-grid addition (with different #heads).
import os
from typing import Dict, Tuple, List

import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

from src.data.addition_algo import BoardConfig
from src.data.problems import (
    generate_diversified_problems,
    generate_multi_addition_problems,
)
from src.data.board_dataset import (
    BlackboardAdditionStepDataset,
    BlackboardMultiAdditionStepDataset,
)
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

CHECKPOINT_DIR = "src/training/trained_weights"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

HEAD_COUNTS: List[int] = [1, 2, 8] 


def masked_cross_entropy(
    logits: torch.Tensor,
    target_ids: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
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


def build_blackboard_model(pe_key: str, cfg: BoardConfig, n_heads: int) -> BlackboardTransformer:
    d_model = 128
    num_layers = 3
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


def freeze_all_but_last_layer(model: BlackboardTransformer) -> None:
    for param in model.parameters():
        param.requires_grad = False
    for param in model.layers[-1].parameters():
        param.requires_grad = True
    for param in model.output_proj.parameters():
        param.requires_grad = True


def train_or_load_largegrid_addition_models(
    TRAIN_BASE: bool,
    cfg_add_large: BoardConfig,
) -> Dict[int, BlackboardTransformer]:
    pe_key = "relative_pe"
    models: Dict[int, BlackboardTransformer] = {}

    n_train_problems = 200_000
    n_val_problems = 5_000
    batch_size = 64
    num_epochs = 3 #2
    lr = 3e-4

    if TRAIN_BASE:
        train_problems = generate_diversified_problems(cfg_add_large, n_train_problems, seed=0)
        val_problems = generate_diversified_problems(cfg_add_large, n_val_problems, seed=1)

        train_ds = BlackboardAdditionStepDataset(train_problems)
        val_ds = BlackboardAdditionStepDataset(val_problems)

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    else:
        train_loader = None
        val_loader = None

    for n_heads in HEAD_COUNTS:
        print(f"==== {pe_key}, {n_heads} heads (large grid addition, 3 digits) ====")
        ckpt_path = os.path.join(
            CHECKPOINT_DIR,
            f"blackboard_{pe_key}_{n_heads}heads_3digits_largegrid.pt",
        )

        model = build_blackboard_model(pe_key, cfg_add_large, n_heads)

        if TRAIN_BASE:
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

                pbar = tqdm(
                    train_loader,
                    desc=f"{pe_key} {n_heads}h Epoch {epoch}/{num_epochs} [train largegrid add]",
                )
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
                    ) = accuracy_with_splits(logits, target_ids, mask, cfg_add_large)

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
                    f"{pe_key} {n_heads}h Epoch {epoch}/{num_epochs} "
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

                pbar_val = tqdm(
                    val_loader,
                    desc=f"{pe_key} {n_heads}h Epoch {epoch}/{num_epochs} [val largegrid add]",
                )
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
                        ) = accuracy_with_splits(logits, target_ids, mask, cfg_add_large)

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
                    f"{pe_key} {n_heads}h Epoch {epoch}/{num_epochs} "
                    f"| val loss/token: {val_avg_loss:.4f} "
                    f"| val acc(masked): {val_avg_acc:.4f} "
                    f"| val carry acc: {val_carry_acc:.4f} "
                    f"| val digit acc: {val_digit_acc:.4f}"
                )

            torch.save(model.state_dict(), ckpt_path)
            print(f"Saved large-grid addition checkpoint to {ckpt_path}")
        else:
            if not os.path.isfile(ckpt_path):
                raise FileNotFoundError(
                    f"Large-grid addition checkpoint not found for {pe_key}, {n_heads} heads: {ckpt_path}. "
                    f"Set TRAIN_BASE=True once to create it."
                )
            state = torch.load(ckpt_path, map_location=DEVICE)
            model.load_state_dict(state)
            print(f"Loaded large-grid addition checkpoint from {ckpt_path}")

        models[n_heads] = model

    return models


def train_or_load_multiaddition_from_largegrid(
    FINETUNE_MULTI: bool,
    cfg_add_large: BoardConfig,
    cfg_multi: BoardConfig,
) -> Tuple[Dict[int, BlackboardTransformer], Dict[int, float]]:
    pe_key = "relative_pe"
    models: Dict[int, BlackboardTransformer] = {}
    val_acc_per_heads: Dict[int, float] = {}

    n_train_problems = 200_000
    n_val_problems = 5_000
    batch_size = 64
    num_epochs = 2
    lr = 3e-4

    if FINETUNE_MULTI:
        train_problems = generate_multi_addition_problems(cfg_multi, n_train_problems, seed=20)
        val_problems = generate_multi_addition_problems(cfg_multi, n_val_problems, seed=21)

        train_ds = BlackboardMultiAdditionStepDataset(train_problems)
        val_ds = BlackboardMultiAdditionStepDataset(val_problems)

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    else:
        train_loader = None
        val_problems = generate_multi_addition_problems(cfg_multi, n_val_problems, seed=21)
        val_ds = BlackboardMultiAdditionStepDataset(val_problems)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    for n_heads in HEAD_COUNTS:
        print(f"==== {pe_key}, {n_heads} heads (multi-addition, >3 digits, from largegrid) ====")
        ckpt_base = os.path.join(
            CHECKPOINT_DIR,
            f"blackboard_{pe_key}_{n_heads}heads_3digits_largegrid.pt",
        )
        ckpt_multi = os.path.join(
            CHECKPOINT_DIR,
            f"blackboard_{pe_key}_{n_heads}heads_multiadd_5digits_largegrid_lastlayer.pt",
        )

        model_multi = build_blackboard_model(pe_key, cfg_multi, n_heads)

        if FINETUNE_MULTI:
            if not os.path.isfile(ckpt_base):
                raise FileNotFoundError(
                    f"Base large-grid checkpoint not found for {pe_key}, {n_heads} heads: {ckpt_base}. "
                    f"Train base first."
                )
            state_base = torch.load(ckpt_base, map_location=DEVICE)
            model_multi.load_state_dict(state_base)

            freeze_all_but_last_layer(model_multi)

            optimizer = torch.optim.Adam(
                [p for p in model_multi.parameters() if p.requires_grad],
                lr=lr,
            )

            for epoch in range(1, num_epochs + 1):
                model_multi.train()
                total_loss = 0.0
                total_tokens = 0
                total_correct = 0
                total_carry_correct = 0
                total_carry_tokens = 0
                total_digit_correct = 0
                total_digit_tokens = 0

                pbar = tqdm(
                    train_loader,
                    desc=f"{pe_key} {n_heads}h Epoch {epoch}/{num_epochs} [train multi-add >3d]",
                )
                for batch in pbar:
                    input_ids = batch["input_ids"].to(DEVICE)
                    target_ids = batch["target_ids"].to(DEVICE)
                    mask = batch["mask"].to(DEVICE)

                    optimizer.zero_grad()
                    logits, _ = model_multi(input_ids)

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
                    f"{pe_key} {n_heads}h Epoch {epoch}/{num_epochs} "
                    f"| train loss/token: {avg_loss:.4f} "
                    f"| train acc(masked): {avg_acc:.4f} "
                    f"| train carry acc: {carry_acc:.4f} "
                    f"| train digit acc: {digit_acc:.4f}"
                )

                model_multi.eval()
                val_loss = 0.0
                val_tokens = 0
                val_correct = 0
                val_carry_correct = 0
                val_carry_tokens = 0
                val_digit_correct = 0
                val_digit_tokens = 0

                pbar_val = tqdm(
                    val_loader,
                    desc=f"{pe_key} {n_heads}h Epoch {epoch}/{num_epochs} [val multi-add >3d]",
                )
                with torch.no_grad():
                    for batch in pbar_val:
                        input_ids = batch["input_ids"].to(DEVICE)
                        target_ids = batch["target_ids"].to(DEVICE)
                        mask = batch["mask"].to(DEVICE)

                        logits, _ = model_multi(input_ids)
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
                    f"{pe_key} {n_heads}h Epoch {epoch}/{num_epochs} "
                    f"| val loss/token: {val_avg_loss:.4f} "
                    f"| val acc(masked): {val_avg_acc:.4f} "
                    f"| val carry acc: {val_carry_acc:.4f} "
                    f"| val digit acc: {val_digit_acc:.4f}"
                )

                val_acc_per_heads[n_heads] = val_avg_acc

            torch.save(model_multi.state_dict(), ckpt_multi)
            print(f"Saved multi-addition (>3 digits) checkpoint to {ckpt_multi}")
        else:
            if not os.path.isfile(ckpt_multi):
                raise FileNotFoundError(
                    f"Multi-addition checkpoint not found for {pe_key}, {n_heads} heads: {ckpt_multi}. "
                    f"Set FINETUNE_MULTI=True once to create it."
                )
            state = torch.load(ckpt_multi, map_location=DEVICE)
            model_multi.load_state_dict(state)
            print(f"Loaded multi-addition checkpoint from {ckpt_multi}")

            model_multi.eval()
            val_loss = 0.0
            val_tokens = 0
            val_correct = 0
            val_carry_correct = 0
            val_carry_tokens = 0
            val_digit_correct = 0
            val_digit_tokens = 0

            pbar_val = tqdm(
                val_loader,
                desc=f"{pe_key} {n_heads}h [eval multi-add >3d]",
            )
            with torch.no_grad():
                for batch in pbar_val:
                    input_ids = batch["input_ids"].to(DEVICE)
                    target_ids = batch["target_ids"].to(DEVICE)
                    mask = batch["mask"].to(DEVICE)

                    logits, _ = model_multi(input_ids)
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
            val_acc_per_heads[n_heads] = val_avg_acc

        models[n_heads] = model_multi

    return models, val_acc_per_heads


def plot_heads_vs_accuracy(val_acc_per_heads: Dict[int, float]) -> None:
    heads = sorted(val_acc_per_heads.keys())
    accs = [val_acc_per_heads[h] for h in heads]

    plt.figure(figsize=(5, 4))
    plt.plot(heads, accs, marker="o")
    plt.xlabel("Number of heads")
    plt.ylabel("Validation accuracy (multi-addition)")
    plt.title("Effect of number of heads on multi-addition accuracy")
    plt.xticks(heads)
    plt.ylim(0.0, 1.0)
    out_path = os.path.join(
        CHECKPOINT_DIR,
        "multi_add_heads_vs_val_acc.png",
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved heads vs accuracy plot to {out_path}")


def main():
    cfg_add_large = BoardConfig(
        H=5,
        W=7,
        n_digits=3,
        n_addends=2,
        carry_row=0,
        top_row=1,
        bottom_row=2,
        result_row=4,
    )

    cfg_multi = BoardConfig(
        H=5,
        W=7,
        n_digits=5,
        n_addends=3,
        carry_row=0,
        top_row=1,
        bottom_row=3,
        result_row=4,
    )

    TRAIN_BASE = True
    FINETUNE_MULTI = True

    _ = train_or_load_largegrid_addition_models(TRAIN_BASE, cfg_add_large)
    _, val_acc_per_heads = train_or_load_multiaddition_from_largegrid(
        FINETUNE_MULTI,
        cfg_add_large,
        cfg_multi,
    )
    plot_heads_vs_accuracy(val_acc_per_heads)


if __name__ == "__main__":
    main()
