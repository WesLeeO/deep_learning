# Training script to evaluate length generalization of various positional encodings
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm.auto import tqdm  

from src.data.addition_algo import BoardConfig
from src.data.problems import generate_problems, generate_diversified_problems
from src.data.board_dataset import BlackboardAdditionStepDataset
from src.models.transformers import BlackboardTransformer

from src.data.cot_dataset import CoTAdditionDataset, COT_VOCAB_TOKENS, ID2TOK
from src.data.sampler import BucketBatchSampler
from src.models.transformers import COTTransformer
from src.models.positional_encodings import *


def masked_cross_entropy(logits, target_ids, mask):
    """
    logits: (B, L, V)
    target_ids: (B, L)
    mask: (B, L) bool, True where we train

    Returns CE averaged over masked positions only.
    """
    vocab_size = logits.size(-1)

    logits_flat = logits.reshape(-1, vocab_size)   # (B*L, V)
    targets_flat = target_ids.reshape(-1)          # (B*L,)
    mask_flat = mask.reshape(-1)                   # (B*L,)

    logits_sel = logits_flat[mask_flat]
    targets_sel = targets_flat[mask_flat]

    return F.cross_entropy(logits_sel, targets_sel)


def accuracy_with_splits(
    logits: torch.Tensor,
    target_ids: torch.Tensor,
    mask: torch.Tensor,
    H: int,
    W: int,
    carry_row: int = 0,
    digit_row: int | None = None,
):
    """
    Compute overall accuracy, plus accuracies restricted to:
      - carry row (smallest row index, by default 0)
      - 'next digit' row (by default last row, H-1)

    logits: (B, L, V)
    target_ids: (B, L)
    mask: (B, L) bool
    """
    if digit_row is None:
        digit_row = H - 1

    B, L, V = logits.shape
    device = logits.device

    preds = logits.argmax(dim=-1)          # (B, L)
    correct = (preds == target_ids) & mask  # (B, L) bool

    total_tokens = mask.sum().item()
    total_correct = correct.sum().item()

    # Row indices for each sequence position (same for all batch elements)
    positions = torch.arange(L, device=device)
    row_idx = positions // W  # (L,)

    # Masks for carry row (row 0) and 'next digit' row (row H-1 by default)
    carry_pos = (row_idx == carry_row).unsqueeze(0).expand_as(mask)   # (B, L)
    digit_pos = (row_idx == digit_row).unsqueeze(0).expand_as(mask)   # (B, L)

    # Effective masks (still intersected with the training mask via 'correct')
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

def evaluate_board_model(model, data_loader, cfg, device, desc):
    model.eval()
    val_loss = 0.0
    val_tokens = 0
    val_correct = 0
    val_carry_correct = 0
    val_carry_tokens = 0
    val_digit_correct = 0
    val_digit_tokens = 0

    pbar_val = tqdm(data_loader, desc=desc)
    with torch.no_grad():
        for batch in pbar_val:
            input_ids  = batch["input_ids"].to(device)
            target_ids = batch["target_ids"].to(device)
            mask       = batch["mask"].to(device)

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
            ) = accuracy_with_splits(
                logits, target_ids, mask, H=cfg.H, W=cfg.W
            )

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
    val_avg_acc  = val_correct / max(val_tokens, 1)
    val_carry_acc = (
        val_carry_correct / val_carry_tokens
        if val_carry_tokens > 0 else 0.0
    )
    val_digit_acc = (
        val_digit_correct / val_digit_tokens
        if val_digit_tokens > 0 else 0.0
    )

    return val_avg_loss, val_avg_acc, val_carry_acc, val_digit_acc


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # -----------------------------
    # Fixed experiment settings
    # -----------------------------
    max_eval_digits = 11
    vocab_size = 12

    n_train_problems_total = 500_000
    n_val_problems = 20_000
 
    batch_size = 1024
    num_epochs = 8
    lr = 3e-4

    d_model = 128
    n_heads = 4
    num_layers = 3
    dim_feedforward = 512
    dropout = 0.1

    # We keep W sized for max_eval_digits so model can run on 9-digit boards.
    # Training configs will differ only by n_digits, but W stays max_eval_digits+2.
    base_cfg = BoardConfig(H=4, W=max_eval_digits + 2, n_digits=3)
    max_len = base_cfg.H * base_cfg.W

    pe_specs = [
    # -----------------------------
    # Relative-only (2D bias)
    # -----------------------------
    ("Relative PE (2D bias)", lambda: RelativePositionBias2D(n_heads, base_cfg.H, base_cfg.W)),

    # -----------------------------
    # Absolute-only (1D)
    # -----------------------------
    ("Sinusoidal PE (1D abs)", lambda: SinusoidalPositionalEncoding(d_model, max_len=max_len)),
    ("Learned PE (1D abs)",    lambda: LearnedPositionalEncoding1D(d_model, max_len=max_len)),

    # -----------------------------
    # Absolute-only (2D)
    # -----------------------------
    ("Sinusoidal PE (2D abs)", lambda: SinusoidalPositionalEncoding2D(d_model, base_cfg.H, base_cfg.W)),
    ("Learned PE (2D abs)",    lambda: LearnedPositionalEncoding2D(d_model, base_cfg.H, base_cfg.W)),

    # -----------------------------
    # Absolute + Relative (2D)
    # -----------------------------
    ("Sin2D + RelBias2D", lambda: Abs2DPlusRelBias2D(
        abs_pe=SinusoidalPositionalEncoding2D(d_model, base_cfg.H, base_cfg.W),
        rel_bias=RelativePositionBias2D(n_heads, base_cfg.H, base_cfg.W),
    )),
    ("Learn2D + RelBias2D", lambda: Abs2DPlusRelBias2D(
        abs_pe=LearnedPositionalEncoding2D(d_model, base_cfg.H, base_cfg.W),
        rel_bias=RelativePositionBias2D(n_heads, base_cfg.H, base_cfg.W),
    )),
]



    eval_digits = [5, 7, 9, 11]

    # -----------------------------
    # Helper: train + eval one PE for a given training dataset
    # -----------------------------
    def train_and_eval_one_pe(train_loader, pe_name: str, pe_module):
        model = BlackboardTransformer(
            vocab_size=vocab_size,
            d_model=d_model,
            nhead=n_heads,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            max_len=max_len,
            dropout=dropout,
            pos_enc=pe_module,
        ).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        # Train
        for epoch in range(1, num_epochs + 1):
            model.train()
            total_loss = 0.0
            total_tokens = 0
            total_correct = 0

            pbar = tqdm(train_loader, desc=f"{pe_name} | epoch {epoch}/{num_epochs} [train]")
            for batch in pbar:
                input_ids  = batch["input_ids"].to(device)
                target_ids = batch["target_ids"].to(device)
                mask       = batch["mask"].to(device)

                optimizer.zero_grad()
                logits, _ = model(input_ids)

                loss = masked_cross_entropy(logits, target_ids, mask)
                loss.backward()
                optimizer.step()

                batch_tokens = mask.sum().item()
                total_loss += loss.item() * batch_tokens
                total_tokens += batch_tokens

                preds = logits.argmax(dim=-1)
                total_correct += ((preds == target_ids) & mask).sum().item()

                pbar.set_postfix(loss=loss.item(), acc=(total_correct / max(total_tokens, 1)))

            avg_loss = total_loss / max(total_tokens, 1)
            avg_acc = total_correct / max(total_tokens, 1)
            print(f"[{pe_name}] Epoch {epoch}/{num_epochs} | loss/token={avg_loss:.4f} | acc={avg_acc:.4f}")

        # Evaluate generalization (overall acc only)
        gen_accs = {}
        for nd in eval_digits:
            cfg_eval = BoardConfig(H=base_cfg.H, W=base_cfg.W, n_digits=nd)
            eval_problems = generate_diversified_problems(cfg_eval, n_val_problems, seed=10 + nd)
            eval_ds = BlackboardAdditionStepDataset(eval_problems)
            eval_loader = DataLoader(eval_ds, batch_size=batch_size, shuffle=False)

            eval_loss, eval_acc, _carry_acc, _digit_acc = evaluate_board_model(
                model,
                eval_loader,
                cfg_eval,
                device,
                desc=f"{pe_name} | {nd}-digit eval",
            )
            gen_accs[nd] = eval_acc
            print(f"[{pe_name}] {nd}-digit | loss={eval_loss:.4f} | acc={eval_acc:.4f}")

        return gen_accs

    # -----------------------------
    # Helper: make a training loader for a given training regime
    # -----------------------------
    def make_train_loader_train_3_only():
        cfg_train = BoardConfig(H=base_cfg.H, W=base_cfg.W, n_digits=3)
        train_problems = generate_diversified_problems(cfg_train, n_train_problems_total, seed=0)
        train_ds = BlackboardAdditionStepDataset(train_problems)
        return DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    def make_train_loader_train_3_and_4():
        n_each = n_train_problems_total // 2  # 250k + 250k
        cfg3 = BoardConfig(H=base_cfg.H, W=base_cfg.W, n_digits=3)
        cfg4 = BoardConfig(H=base_cfg.H, W=base_cfg.W, n_digits=4)

        train3 = generate_diversified_problems(cfg3, n_each, seed=0)
        train4 = generate_diversified_problems(cfg4, n_each, seed=1)

        train_ds3 = BlackboardAdditionStepDataset(train3)
        train_ds4 = BlackboardAdditionStepDataset(train4)

        merged_ds = torch.utils.data.ConcatDataset([train_ds3, train_ds4])
        return DataLoader(merged_ds, batch_size=batch_size, shuffle=True)

    # -----------------------------
    # EXP A: train on 3-digit only
    # -----------------------------
    print("\n" + "=" * 90)
    print("EXPERIMENT A: Train on 3-digit only; eval on 5/7/9 digits (overall acc)")
    train_loader_A = make_train_loader_train_3_only()

    results_A = {name: [] for name, _ in pe_specs}
    for pe_name, pe_factory in pe_specs:
        pe = pe_factory()
        gen_accs = train_and_eval_one_pe(train_loader_A, pe_name, pe)
        results_A[pe_name] = [gen_accs[d] for d in eval_digits]

    plt.figure()
    for pe_name in results_A:
        plt.plot(eval_digits, results_A[pe_name], marker="o", label=pe_name)
    plt.xlabel("Number of digits (test)")
    plt.ylabel("Overall masked accuracy")
    plt.ylim(0.0, 1.05)
    plt.title("Generalization after training on 3-digit only")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("gen_overall_train3_test5_7_9.png")
    plt.close()
    print("Saved: gen_overall_train3_test5_7_9.png")

    # -----------------------------
    # EXP B: train on mixed 3+4 digits
    # -----------------------------
    print("\n" + "=" * 90)
    print("EXPERIMENT B: Train on 3+4 digits (250k each); eval on 5/7/9 digits (overall acc)")
    train_loader_B = make_train_loader_train_3_and_4()

    results_B = {name: [] for name, _ in pe_specs}
    for pe_name, pe_factory in pe_specs:
        pe = pe_factory()
        gen_accs = train_and_eval_one_pe(train_loader_B, pe_name, pe)
        results_B[pe_name] = [gen_accs[d] for d in eval_digits]

    plt.figure()
    for pe_name in results_B:
        plt.plot(eval_digits, results_B[pe_name], marker="o", label=pe_name)
    plt.xlabel("Number of digits (test)")
    plt.ylabel("Overall masked accuracy")
    plt.ylim(0.0, 1.05)
    plt.title("Generalization after training on mixed 3+4 digits")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("gen_overall_train3and4_test5_7_9.png")
    plt.close()
    print("Saved: gen_overall_train3and4_test5_7_9.png")


if __name__ == "__main__":
    main()


