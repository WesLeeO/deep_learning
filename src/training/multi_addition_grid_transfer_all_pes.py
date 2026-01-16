# Multi-addition transfer learning across various positional encodings.
import os
from typing import Dict, Tuple, List, Any

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


N_HEADS: int = 4


PE_KEYS: List[str] = [
    "relative_pe",
    "abs_1d_sinusoidal",
    "abs_2d_sinusoidal",
    "abs_1d_learned",
    "abs_2d_learned",
    "abs_2d_sin+rel_2d_bias",
]

PE_LABELS: Dict[str, str] = {
    "relative_pe": "RelBias 2D",
    "abs_1d_sinusoidal": "Abs 1D Sin",
    "abs_2d_sinusoidal": "Abs 2D Sin",
    "abs_1d_learned": "Abs 1D Learned",
    "abs_2d_learned": "Abs 2D Learned",
    "abs_2d_sin+rel_2d_bias": "Abs2D Sin + RelBias2D",
}


DIGIT_LIST: List[int] = [5, 7, 9, 11]


SAVE_ATTENTION: bool = False
ATTN_OUT_DIR = os.path.join(CHECKPOINT_DIR, "attn_viz_multiadd_digits")
os.makedirs(ATTN_OUT_DIR, exist_ok=True)




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


def masked_accuracy(
    logits: torch.Tensor,
    target_ids: torch.Tensor,
    mask: torch.Tensor,
) -> Tuple[int, int]:
    preds = logits.argmax(dim=-1)
    correct = (preds == target_ids) & mask
    return correct.sum().item(), mask.sum().item()



def build_blackboard_model(pe_key: str, cfg: BoardConfig, n_heads: int) -> BlackboardTransformer:
    D_MODEL = 128
    NUM_LAYERS = 3
    DIM_FF = 512
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
        d_model=D_MODEL,
        nhead=n_heads,
        num_layers=NUM_LAYERS,
        dim_feedforward=DIM_FF,
        max_len=max_len,
        dropout=0.1,
        pos_enc=pos_enc,
    ).to(DEVICE)

    return model


def freeze_all_but_last_layer(model: BlackboardTransformer) -> None:
    for p in model.parameters():
        p.requires_grad = False
    for p in model.layers[-1].parameters():
        p.requires_grad = True
    for p in model.output_proj.parameters():
        p.requires_grad = True


def maybe_save_attention(attn_obj: Any, pe_key: str, n_digits: int, cfg: BoardConfig) -> None:
    """
    Save simple attention heatmaps if available.
    Expected common format: list[tensor] where each tensor is (B, heads, L, L).
    """
    if not SAVE_ATTENTION or attn_obj is None:
        return

    try:
        layers = attn_obj if isinstance(attn_obj, (list, tuple)) else [attn_obj]
        for li, A in enumerate(layers):
            if A is None or not hasattr(A, "dim") or A.dim() != 4:
                continue
            A_mean = A.mean(dim=0)  # (heads, L, L)
            for hi in range(min(A_mean.size(0), 8)):
                plt.figure(figsize=(5, 4))
                plt.imshow(A_mean[hi].detach().cpu().numpy(), aspect="auto")
                plt.title(f"{PE_LABELS.get(pe_key, pe_key)} | digits={n_digits} | layer={li} | head={hi}")
                plt.xlabel("Key pos")
                plt.ylabel("Query pos")
                out_path = os.path.join(ATTN_OUT_DIR, f"attn_{pe_key}_d{n_digits}_layer{li}_head{hi}.png")
                plt.tight_layout()
                plt.savefig(out_path, dpi=150)
                plt.close()
    except Exception as e:
        print(f"[WARN] Could not save attention maps: {e}")


def train_or_load_base_addition(pe_key: str, TRAIN_BASE: bool, cfg_add_base: BoardConfig) -> str:
    """
    Train (or load) base model on 2-addend addition (n_digits=3) on a large grid (H,W fixed),
    return checkpoint path.
    """
    ckpt_path = os.path.join(
        CHECKPOINT_DIR,
        f"blackboard_{pe_key}_{N_HEADS}heads_add_base_3digits_largegrid.pt",
    )

    if not TRAIN_BASE:
        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(f"Base checkpoint not found: {ckpt_path}. Set TRAIN_BASE=True once.")
        print(f"Base checkpoint found: {ckpt_path}")
        return ckpt_path

    n_train_problems = 100_000
    n_val_problems = 1000
    batch_size = 64
    num_epochs = 2
    lr = 3e-4

    model = build_blackboard_model(pe_key, cfg_add_base, N_HEADS)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_problems = generate_diversified_problems(cfg_add_base, n_train_problems, seed=0)
    val_problems = generate_diversified_problems(cfg_add_base, n_val_problems, seed=1)

    train_loader = DataLoader(BlackboardAdditionStepDataset(train_problems), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(BlackboardAdditionStepDataset(val_problems), batch_size=batch_size, shuffle=False)

    for epoch in range(1, num_epochs + 1):
        model.train()
        tot_loss, tot_tok, tot_cor = 0.0, 0, 0

        pbar = tqdm(train_loader, desc=f"[BASE] {PE_LABELS.get(pe_key, pe_key)} ep {epoch}/{num_epochs}")
        for batch in pbar:
            input_ids = batch["input_ids"].to(DEVICE)
            target_ids = batch["target_ids"].to(DEVICE)
            mask = batch["mask"].to(DEVICE)

            optimizer.zero_grad()
            logits, _ = model(input_ids)
            loss = masked_cross_entropy(logits, target_ids, mask)
            loss.backward()
            optimizer.step()

            cor, tok = masked_accuracy(logits, target_ids, mask)
            tot_loss += loss.item() * tok
            tot_tok += tok
            tot_cor += cor

            pbar.set_postfix(loss=loss.item(), acc=(cor / max(tok, 1)))

        train_acc = tot_cor / max(tot_tok, 1)
        train_loss = tot_loss / max(tot_tok, 1)

        model.eval()
        v_loss, v_tok, v_cor = 0.0, 0, 0
        with torch.no_grad():
            pbarv = tqdm(val_loader, desc=f"[BASE-VAL] {PE_LABELS.get(pe_key, pe_key)} ep {epoch}")
            for batch in pbarv:
                input_ids = batch["input_ids"].to(DEVICE)
                target_ids = batch["target_ids"].to(DEVICE)
                mask = batch["mask"].to(DEVICE)

                logits, _ = model(input_ids)
                loss = masked_cross_entropy(logits, target_ids, mask)
                cor, tok = masked_accuracy(logits, target_ids, mask)

                v_loss += loss.item() * tok
                v_tok += tok
                v_cor += cor
                pbarv.set_postfix(loss=loss.item(), acc=(cor / max(tok, 1)))

        val_acc = v_cor / max(v_tok, 1)
        val_loss = v_loss / max(v_tok, 1)

        print(
            f"[BASE] {pe_key} ep{epoch}: "
            f"train loss/token={train_loss:.4f} acc={train_acc:.4f} | "
            f"val loss/token={val_loss:.4f} acc={val_acc:.4f}"
        )

    torch.save(model.state_dict(), ckpt_path)
    print(f"Saved base checkpoint to {ckpt_path}")
    return ckpt_path


def finetune_or_eval_multiadd_for_digits(
    pe_key: str,
    FINETUNE_MULTI: bool,
    base_ckpt_path: str,
    cfg_multi: BoardConfig,
) -> float:
    """
    Last-layer fine-tuning from base checkpoint for a given n_digits multi-add setting.
    Returns val overall masked accuracy.
    """
    n_digits = cfg_multi.n_digits
    ckpt_multi = os.path.join(
        CHECKPOINT_DIR,
        f"blackboard_{pe_key}_{N_HEADS}heads_multiadd_{n_digits}digits_largegrid_lastlayer.pt",
    )

    model = build_blackboard_model(pe_key, cfg_multi, N_HEADS)

    n_train_problems = 100_000
    n_val_problems = 1000
    batch_size = 64
    num_epochs = 2
    lr = 3e-4

    # Validation data is needed in both modes
    val_problems = generate_multi_addition_problems(cfg_multi, n_val_problems, seed=21 + n_digits)
    val_loader = DataLoader(BlackboardMultiAdditionStepDataset(val_problems), batch_size=batch_size, shuffle=False)

    if FINETUNE_MULTI:
        if not os.path.isfile(base_ckpt_path):
            raise FileNotFoundError(f"Base checkpoint missing: {base_ckpt_path}")

        state_base = torch.load(base_ckpt_path, map_location=DEVICE)
        model.load_state_dict(state_base)

        freeze_all_but_last_layer(model)

        train_problems = generate_multi_addition_problems(cfg_multi, n_train_problems, seed=20 + n_digits)
        train_loader = DataLoader(BlackboardMultiAdditionStepDataset(train_problems), batch_size=batch_size, shuffle=True)

        optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)

        for epoch in range(1, num_epochs + 1):
            model.train()
            tot_loss, tot_tok, tot_cor = 0.0, 0, 0

            pbar = tqdm(
                train_loader,
                desc=f"[FT] {PE_LABELS.get(pe_key, pe_key)} d={n_digits} ep {epoch}/{num_epochs}",
            )
            for bi, batch in enumerate(pbar):
                input_ids = batch["input_ids"].to(DEVICE)
                target_ids = batch["target_ids"].to(DEVICE)
                mask = batch["mask"].to(DEVICE)

                optimizer.zero_grad()
                logits, attn = model(input_ids)
                loss = masked_cross_entropy(logits, target_ids, mask)
                loss.backward()
                optimizer.step()

                cor, tok = masked_accuracy(logits, target_ids, mask)
                tot_loss += loss.item() * tok
                tot_tok += tok
                tot_cor += cor

                pbar.set_postfix(loss=loss.item(), acc=(cor / max(tok, 1)))

                # Save attention once (last epoch, first batch) if enabled
                if SAVE_ATTENTION and bi == 0 and epoch == num_epochs:
                    maybe_save_attention(attn, pe_key, n_digits, cfg_multi)

            train_acc = tot_cor / max(tot_tok, 1)
            train_loss = tot_loss / max(tot_tok, 1)

            # Validation
            model.eval()
            v_loss, v_tok, v_cor = 0.0, 0, 0
            with torch.no_grad():
                pbarv = tqdm(val_loader, desc=f"[FT-VAL] {PE_LABELS.get(pe_key, pe_key)} d={n_digits} ep {epoch}")
                for batch in pbarv:
                    input_ids = batch["input_ids"].to(DEVICE)
                    target_ids = batch["target_ids"].to(DEVICE)
                    mask = batch["mask"].to(DEVICE)

                    logits, _ = model(input_ids)
                    loss = masked_cross_entropy(logits, target_ids, mask)
                    cor, tok = masked_accuracy(logits, target_ids, mask)

                    v_loss += loss.item() * tok
                    v_tok += tok
                    v_cor += cor
                    pbarv.set_postfix(loss=loss.item(), acc=(cor / max(tok, 1)))

            val_acc = v_cor / max(v_tok, 1)
            val_loss = v_loss / max(v_tok, 1)

            print(
                f"[FT] {pe_key} d={n_digits} ep{epoch}: "
                f"train loss/token={train_loss:.4f} acc={train_acc:.4f} | "
                f"val loss/token={val_loss:.4f} acc={val_acc:.4f}"
            )

        torch.save(model.state_dict(), ckpt_multi)
        print(f"Saved multi-add checkpoint to {ckpt_multi}")
        return val_acc


    if not os.path.isfile(ckpt_multi):
        raise FileNotFoundError(f"Multi-add checkpoint not found: {ckpt_multi}. Set FINETUNE_MULTI=True once.")

    state = torch.load(ckpt_multi, map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()

    v_tok, v_cor = 0, 0
    last_attn = None
    with torch.no_grad():
        pbarv = tqdm(val_loader, desc=f"[EVAL] {PE_LABELS.get(pe_key, pe_key)} d={n_digits}")
        for batch in pbarv:
            input_ids = batch["input_ids"].to(DEVICE)
            target_ids = batch["target_ids"].to(DEVICE)
            mask = batch["mask"].to(DEVICE)

            logits, attn = model(input_ids)
            last_attn = attn
            cor, tok = masked_accuracy(logits, target_ids, mask)

            v_tok += tok
            v_cor += cor
            pbarv.set_postfix(acc=(cor / max(tok, 1)))

    if SAVE_ATTENTION:
        maybe_save_attention(last_attn, pe_key, n_digits, cfg_multi)

    return v_cor / max(v_tok, 1)



def plot_digits_vs_accuracy(acc_by_pe: Dict[str, Dict[int, float]]) -> None:
    plt.figure(figsize=(7.2, 4.6))

    for pe_key in PE_KEYS:
        digits = sorted(acc_by_pe[pe_key].keys())
        accs = [acc_by_pe[pe_key][d] for d in digits]
        plt.plot(digits, accs, marker="o", label=PE_LABELS.get(pe_key, pe_key))

    plt.xlabel("Number of digits (multi-addition)")
    plt.ylabel("Validation overall masked accuracy")
    plt.title(f"Transfer learning multi-addition: accuracy vs digits (heads={N_HEADS})")
    plt.xticks(DIGIT_LIST)
    plt.ylim(0.0, 1.0)
    plt.grid(True, alpha=0.25)
    plt.legend()

    out_path = os.path.join(CHECKPOINT_DIR, f"multi_add_digits_vs_val_acc_all_PEs_{N_HEADS}heads.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved digits-vs-accuracy plot to {out_path}")



def main():

    H, W = 5, 13

    cfg_add_base = BoardConfig(
        H=H,
        W=W,
        n_digits=3,
        n_addends=2,
        carry_row=0,
        top_row=1,
        bottom_row=2,
        result_row=4,
    )

    TRAIN_BASE = True
    FINETUNE_MULTI = True

    acc_by_pe: Dict[str, Dict[int, float]] = {k: {} for k in PE_KEYS}

    for pe_key in PE_KEYS:
        print(f"\n==================== PE = {PE_LABELS.get(pe_key, pe_key)} | heads={N_HEADS} ====================\n")
        base_ckpt = train_or_load_base_addition(pe_key, TRAIN_BASE, cfg_add_base)

        for n_digits in DIGIT_LIST:
            cfg_multi = BoardConfig(
                H=H,
                W=W,
                n_digits=n_digits,
                n_addends=3,
                carry_row=0,
                top_row=1,
                bottom_row=3,
                result_row=4,
            )

            print(f"\n---- Multi-add transfer: {PE_LABELS.get(pe_key, pe_key)} | digits={n_digits} ----")
            val_acc = finetune_or_eval_multiadd_for_digits(
                pe_key=pe_key,
                FINETUNE_MULTI=FINETUNE_MULTI,
                base_ckpt_path=base_ckpt,
                cfg_multi=cfg_multi,
            )
            acc_by_pe[pe_key][n_digits] = val_acc

    plot_digits_vs_accuracy(acc_by_pe)


if __name__ == "__main__":
    main()
