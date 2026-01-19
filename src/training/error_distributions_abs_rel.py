import os
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.data.addition_algo import BoardConfig
from src.data.board_dataset import BlackboardAdditionStepDataset
from src.data.sample_efficiency import generate_setting1_random_fraction
from src.models.transformers import BlackboardTransformer
from src.models.positional_encodings import (
    AbsolutePositionalEncoding2D,
    RelativePositionBias2D,
)


ROW_TOP = 0
ROW_BOT = 1
ROW_CARRY = 2
ROW_OUT = 3

VOCAB_SIZE = 12  


# -----------------------------
# Loss / eval helpers
# -----------------------------
def masked_cross_entropy(logits, target_ids, mask):
    vocab_size = logits.size(-1)
    logits_flat = logits.reshape(-1, vocab_size)
    targets_flat = target_ids.reshape(-1)
    mask_flat = mask.reshape(-1)

    logits_sel = logits_flat[mask_flat]
    targets_sel = targets_flat[mask_flat]
    return F.cross_entropy(logits_sel, targets_sel)


def evaluate_accuracy(model, data_loader, device) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    total_correct = 0

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch["input_ids"].to(device)
            target_ids = batch["target_ids"].to(device)
            mask = batch["mask"].to(device)

            logits, _ = model(input_ids)
            loss = masked_cross_entropy(logits, target_ids, mask)

            batch_tokens = int(mask.sum().item())
            total_loss += float(loss.item()) * batch_tokens
            total_tokens += batch_tokens

            preds = logits.argmax(dim=-1)
            total_correct += int(((preds == target_ids) & mask).sum().item())

    avg_loss = total_loss / max(total_tokens, 1)
    avg_acc = total_correct / max(total_tokens, 1)
    return avg_loss, avg_acc


# -----------------------------
# Board indexing helpers
# -----------------------------
def _digit_col_from_digit_idx(cfg: BoardConfig, digit_idx_lsd0: int) -> int:
    """
    Map digit index (0 = LSD) to board column index.
    We assume digits occupy the last cfg.n_digits columns.
    """
    return (cfg.W - 1) - digit_idx_lsd0


def _read_operand_digits_lsd_first(input_ids_1d: torch.Tensor, cfg: BoardConfig, row: int) -> List[int]:
    """
    Read digits from a given operand row as a list [LSD,...,MSD].
    Assumes digit tokens are 0..9.
    """
    digits = []
    for k in range(cfg.n_digits):  # k = digit_idx (LSD=0)
        col = _digit_col_from_digit_idx(cfg, k)
        tok = int(input_ids_1d[row * cfg.W + col].item())
        digits.append(tok)
    return digits


def _compute_carry_in_per_digit(top_digits_lsd: List[int], bot_digits_lsd: List[int], base: int) -> List[int]:
    carry = 0
    cin_list = []
    for a, b in zip(top_digits_lsd, bot_digits_lsd):
        cin_list.append(carry)
        s = a + b + carry
        carry = s // base
    return cin_list  # length = n_digits


# -----------------------------
# Error histogram collection: carry-row (UNMASKED)
# -----------------------------
@torch.no_grad()
def collect_carry_error_histograms_unmasked(
    model: torch.nn.Module,
    data_loader: DataLoader,
    cfg: BoardConfig,
    device: torch.device,
    vocab_size: int = VOCAB_SIZE,
) -> Dict[int, torch.Tensor]:
    """
    Does NOT require dataset mask to include carry row.

    We compute the true carry_in per digit from operands in input_ids,
    then look at model prediction on the carry row at that column.

    Errors where: predicted token != true carry (0/1).

    Returns:
        hists[cin] = counts over predicted token IDs at carry-row positions
                     where true carry_in == cin AND model was wrong.
        cin in {0,1}.
    """
    model.eval()
    hists = {
        0: torch.zeros(vocab_size, dtype=torch.long),
        1: torch.zeros(vocab_size, dtype=torch.long),
    }

    total_seen = {0: 0, 1: 0}
    total_wrong = {0: 0, 1: 0}

    for batch in tqdm(data_loader, desc="Collecting carry error histograms (UNMASKED)"):
        input_ids = batch["input_ids"].to(device)    # (B, L)
        logits, _ = model(input_ids)
        preds = logits.argmax(dim=-1)  # (B, L)

        B, L = input_ids.shape
        W = cfg.W
        assert L == cfg.H * cfg.W, f"Expected L=H*W={cfg.H*cfg.W}, got {L}"

        for i in range(B):
            x = input_ids[i]   # (L,)
            p = preds[i]       # (L,)

            # operands -> true carry_in per digit
            top_digits = _read_operand_digits_lsd_first(x, cfg, ROW_TOP)
            bot_digits = _read_operand_digits_lsd_first(x, cfg, ROW_BOT)
            cin_list = _compute_carry_in_per_digit(top_digits, bot_digits, base=cfg.base)

            for k in range(cfg.n_digits):
                col = _digit_col_from_digit_idx(cfg, k)
                pos = ROW_CARRY * W + col

                cin_true = int(cin_list[k])
                if cin_true not in hists:
                    continue

                total_seen[cin_true] += 1
                pred_tok = int(p[pos].item())

                # True carry token assumed to be 0/1
                if pred_tok != cin_true:
                    total_wrong[cin_true] += 1
                    if 0 <= pred_tok < vocab_size:
                        hists[cin_true][pred_tok] += 1

    print("[carry-hist] UNMASKED carry-row evaluation")
    for cin in [0, 1]:
        seen = total_seen[cin]
        wrong = total_wrong[cin]
        rate = (wrong / seen) if seen > 0 else 0.0
        print(f"  cin={cin}: seen={seen}, wrong={wrong}, wrong_rate={rate:.4f}")

    if total_seen[0] + total_seen[1] == 0:
        print(
            "[carry-hist] WARNING: Saw zero carry positions. "
            "This usually means the board layout assumptions (rows/cols) are wrong."
        )

    return hists


# -----------------------------
#Error histogram collection: output-digit row (UNMASKED)
# -----------------------------
@torch.no_grad()
def collect_output_digit_error_hist_unmasked(
    model: torch.nn.Module,
    data_loader: DataLoader,
    cfg: BoardConfig,
    device: torch.device,
    vocab_size: int = VOCAB_SIZE,
) -> torch.Tensor:
    """
    Collect a histogram over predicted token IDs at OUTPUT row digit positions
    *when the model is wrong*.

    We use the dataset's target_ids as the ground-truth output tokens
    (no need to reconstruct sum), and we evaluate UNMASKED (ignore batch["mask"]).

    Returns:
        counts[v] = number of wrong predictions that predicted token v
                    at OUTPUT row digit positions.
    """
    model.eval()
    counts = torch.zeros(vocab_size, dtype=torch.long)

    total_seen = 0
    total_wrong = 0

    for batch in tqdm(data_loader, desc="Collecting output-digit error histogram (UNMASKED)"):
        input_ids = batch["input_ids"].to(device)
        target_ids = batch["target_ids"].to(device)
        logits, _ = model(input_ids)
        preds = logits.argmax(dim=-1)

        B, L = input_ids.shape
        W = cfg.W
        assert L == cfg.H * cfg.W, f"Expected L=H*W={cfg.H*cfg.W}, got {L}"

        for i in range(B):
            y = target_ids[i]
            p = preds[i]

            # only digit columns (ignore the +2 non-digit columns)
            for k in range(cfg.n_digits):
                col = _digit_col_from_digit_idx(cfg, k)
                pos = ROW_OUT * W + col

                true_tok = int(y[pos].item())
                pred_tok = int(p[pos].item())

                total_seen += 1
                if pred_tok != true_tok:
                    total_wrong += 1
                    if 0 <= pred_tok < vocab_size:
                        counts[pred_tok] += 1

    wrong_rate = (total_wrong / total_seen) if total_seen > 0 else 0.0
    print("[out-hist] UNMASKED output-row digit evaluation")
    print(f"  seen={total_seen}, wrong={total_wrong}, wrong_rate={wrong_rate:.4f}")

    if total_seen == 0:
        print(
            "[out-hist] WARNING: Saw zero output digit positions. "
            "Check board layout assumptions."
        )

    return counts


# -----------------------------
# Plotting helper
# -----------------------------
def plot_hist_pair(counts: torch.Tensor, title: str, outpath: str):
    """
    Two subplots side-by-side:
      - left: linear frequency
      - right: log frequency
    """
    counts_np = counts.cpu().numpy()
    xs = list(range(len(counts_np)))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].bar(xs, counts_np)
    axes[0].set_title(title + " (linear)")
    axes[0].set_xlabel("Predicted token id")
    axes[0].set_ylabel("Count")
    axes[0].set_xticks(xs)

    axes[1].bar(xs, counts_np)
    if counts_np.sum() > 0:
        axes[1].set_yscale("log")
    axes[1].set_title(title + " (log scale)")
    axes[1].set_xlabel("Predicted token id")
    axes[1].set_ylabel("Count (log)")
    axes[1].set_xticks(xs)

    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()


# -----------------------------
# Train + analyze (Rel vs Abs2D)
# -----------------------------
def train_one_model(
    cfg: BoardConfig,
    train_problems,
    test_problems,
    pos_enc_name: str,
    pos_enc_module,
    device: torch.device,
    d_model: int = 128,
    n_heads: int = 2,
    num_layers: int = 2,
    dim_feedforward: int = 512,
    dropout: float = 0.1,
    batch_size: int = 128,
    num_epochs: int = 8,
    lr: float = 3e-4,
):
    train_ds = BlackboardAdditionStepDataset(train_problems)
    test_ds = BlackboardAdditionStepDataset(test_problems)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    vocab_size = VOCAB_SIZE
    max_len = cfg.H * cfg.W

    model = BlackboardTransformer(
        vocab_size=vocab_size,
        d_model=d_model,
        nhead=n_heads,
        num_layers=num_layers,
        dim_feedforward=dim_feedforward,
        max_len=max_len,
        dropout=dropout,
        pos_enc=pos_enc_module,
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss = 0.0
        total_tokens = 0
        total_correct = 0

        pbar = tqdm(train_loader, desc=f"{pos_enc_name} | epoch {epoch}/{num_epochs}")
        for batch in pbar:
            input_ids = batch["input_ids"].to(device)
            target_ids = batch["target_ids"].to(device)
            mask = batch["mask"].to(device)

            opt.zero_grad()
            logits, _ = model(input_ids)
            loss = masked_cross_entropy(logits, target_ids, mask)
            loss.backward()
            opt.step()

            batch_tokens = int(mask.sum().item())
            total_loss += float(loss.item()) * batch_tokens
            total_tokens += batch_tokens

            preds = logits.argmax(dim=-1)
            correct = int(((preds == target_ids) & mask).sum().item())
            total_correct += correct

            pbar.set_postfix(loss=float(loss.item()), acc=correct / max(batch_tokens, 1))

        avg_loss = total_loss / max(total_tokens, 1)
        avg_acc = total_correct / max(total_tokens, 1)
        print(f"[{pos_enc_name}] Epoch {epoch}/{num_epochs} | loss/token={avg_loss:.4f} | acc={avg_acc:.4f}")

    test_loss, test_acc = evaluate_accuracy(model, test_loader, device)
    print(f"[{pos_enc_name}] FINAL test loss/token={test_loss:.4f} | test acc={test_acc:.4f}")

    return model, test_loader


def main():
    os.makedirs("carry_error_plots", exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    torch.manual_seed(0)

    # 3-digit addition
    cfg = BoardConfig(H=4, W=3 + 2, n_digits=3)

    # fixed training sizes
    n_train = 40_000
    n_test = 50_000
    print(f"Generating Setting1 data: n_train={n_train}, n_test={n_test}")
    train_problems, test_problems = generate_setting1_random_fraction(
        cfg, n_train=n_train, n_test=n_test, seed=0
    )

    # fixed model hyperparams
    d_model = 128
    n_heads = 2
    num_layers = 4
    d_ff = 512
    dropout = 0.1
    batch_size = 128
    num_epochs = 4
    lr = 3e-4

    experiments = [
        ("Relative PE", RelativePositionBias2D(n_heads, cfg.H, cfg.W)),
        ("Absolute PE", AbsolutePositionalEncoding2D(d_model, cfg.H, cfg.W)),
    ]

    for name, pe in experiments:
        print("\n" + "=" * 90)
        print(f"Training: {name}")
        model, test_loader = train_one_model(
            cfg=cfg,
            train_problems=train_problems,
            test_problems=test_problems,
            pos_enc_name=name,
            pos_enc_module=pe,
            device=device,
            d_model=d_model,
            n_heads=n_heads,
            num_layers=num_layers,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_size=batch_size,
            num_epochs=num_epochs,
            lr=lr,
        )

        # (1) Carry-row wrong-prediction histograms (true cin=0/1)
        hists = collect_carry_error_histograms_unmasked(
            model=model,
            data_loader=test_loader,
            cfg=cfg,
            device=device,
            vocab_size=VOCAB_SIZE,
        )

        for cin in [0, 1]:
            outpath = os.path.join(
                "carry_error_plots",
                f"{name.replace(' ', '_').lower()}_true_cin_{cin}_wrong_pred_hist2.png",
            )
            title = f"{name}: wrong carry-row predictions | true carry_in={cin}"
            plot_hist_pair(hists[cin], title=title, outpath=outpath)
            print(f"Saved: {outpath}")

        # (2) Output-row digit wrong-prediction histogram
        out_counts = collect_output_digit_error_hist_unmasked(
            model=model,
            data_loader=test_loader,
            cfg=cfg,
            device=device,
            vocab_size=VOCAB_SIZE,
        )
        outpath = os.path.join(
            "carry_error_plots",
            f"{name.replace(' ', '_').lower()}_output_digit_wrong_pred_hist.png",
        )
        title = f"{name}: wrong OUTPUT-digit predictions (all digit cols)"
        plot_hist_pair(out_counts, title=title, outpath=outpath)
        print(f"Saved: {outpath}")


if __name__ == "__main__":
    main()
