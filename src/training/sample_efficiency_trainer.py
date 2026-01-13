# src/training/sample_efficiency_trainer.py

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from typing import List, Tuple, Dict

from src.data.addition_algo import BoardConfig
from src.data.board_dataset import BlackboardAdditionStepDataset
from src.models.transformers import BlackboardTransformer
from src.models.positional_encodings import *


from src.data.sample_efficiency import (
    Triplet,
    generate_setting1_random_fraction,
    generate_setting2_position_split,
    generate_setting3_order_constraint,
    generate_setting4_triplet_holdout,
    extract_triplets,
)


# ---------------------------------------------------------------------------
# Loss + accuracy utilities (same spirit as training_board_phase_0)
# ---------------------------------------------------------------------------

def masked_cross_entropy(logits, target_ids, mask):
    """
    logits: (B, L, V)
    target_ids: (B, L)
    mask: (B, L) bool

    Cross-entropy averaged over masked positions.
    """
    vocab_size = logits.size(-1)

    logits_flat = logits.reshape(-1, vocab_size)   # (B*L, V)
    targets_flat = target_ids.reshape(-1)          # (B*L,)
    mask_flat = mask.reshape(-1)                   # (B*L,)

    logits_sel = logits_flat[mask_flat]
    targets_sel = targets_flat[mask_flat]

    return F.cross_entropy(logits_sel, targets_sel)


def compute_accuracy(logits, target_ids, mask):
    """
    Overall masked accuracy.

    logits: (B, L, V)
    target_ids: (B, L)
    mask: (B, L) bool
    """
    preds = logits.argmax(dim=-1)          # (B, L)
    correct = (preds == target_ids) & mask  # (B, L)
    total = mask.sum().item()
    if total == 0:
        return 0.0
    return correct.sum().item() / total


def evaluate_model(model, data_loader, device):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    total_correct = 0

    with torch.no_grad():
        for batch in data_loader:
            input_ids  = batch["input_ids"].to(device)
            target_ids = batch["target_ids"].to(device)
            mask       = batch["mask"].to(device)

            logits, _ = model(input_ids)

            loss = masked_cross_entropy(logits, target_ids, mask)
            batch_tokens = mask.sum().item()
            total_loss += loss.item() * batch_tokens
            total_tokens += batch_tokens

            preds = logits.argmax(dim=-1)
            correct = ((preds == target_ids) & mask).sum().item()
            total_correct += correct

    avg_loss = total_loss / max(total_tokens, 1)
    avg_acc  = total_correct / max(total_tokens, 1)
    return avg_loss, avg_acc


def train_one_model(
    cfg: BoardConfig,
    train_problems,
    test_problems,
    pos_enc_name: str,
    pos_enc_module,
    device,
    d_model: int = 128,
    n_heads: int = 2,
    num_layers: int = 2,
    dim_feedforward: int = 512,
    dropout: float = 0.1,
    batch_size: int = 128,
    num_epochs: int = 3,
    lr: float = 3e-4,
):
    """
    Train a BlackboardTransformer with the given positional encoding on
    train_problems, evaluate on test_problems, and return test accuracy.
    """
    # Datasets / loaders
    train_ds = BlackboardAdditionStepDataset(train_problems)
    test_ds  = BlackboardAdditionStepDataset(test_problems)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False)

    vocab_size = 12
    max_len = cfg.H * cfg.W

    # Some PEs are absolute (no args), some depend on H,W. We assume the
    # module passed is already instantiated with correct shape if needed.
    # BlackboardTransformer takes a pos_enc module as argument.
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

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss = 0.0
        total_tokens = 0
        total_correct = 0

        pbar = tqdm(train_loader, desc=f"{pos_enc_name} | epoch {epoch}/{num_epochs}")
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
            correct = ((preds == target_ids) & mask).sum().item()
            total_correct += correct

            batch_acc = correct / max(batch_tokens, 1)
            pbar.set_postfix(loss=loss.item(), acc=batch_acc)

        avg_train_loss = total_loss / max(total_tokens, 1)
        avg_train_acc  = total_correct / max(total_tokens, 1)
        print(
            f"[{pos_enc_name}] Epoch {epoch}/{num_epochs} "
            f"| train loss/token: {avg_train_loss:.4f} "
            f"| train acc(masked): {avg_train_acc:.4f}"
        )

    test_loss, test_acc = evaluate_model(model, test_loader, device)
    print(
        f"[{pos_enc_name}] Final test loss/token: {test_loss:.4f} "
        f"| test acc(masked): {test_acc:.4f}"
    )
    return test_acc


# ---------------------------------------------------------------------------
# Main experiment: 4 settings
# ---------------------------------------------------------------------------

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

#prev was with 2 H, 2 L (26 dec)

    # -------------------------------
    # Shared model hyperparameters
    # -------------------------------
    d_model = 128
    n_heads = 2
    num_layers = 3
    dim_feedforward = 512
    dropout = 0.1
    batch_size = 128
    num_epochs = 10            
    lr = 3e-4

    # -------------------------------
    # Base config: 3-digit addition
    # used for settings 1, 3, 4
    # -------------------------------
    cfg_3 = BoardConfig(H=4, W=3 + 2, n_digits=3)
    max_len_3 = cfg_3.H * cfg_3.W

    # Positional encodings for 3-digit experiments (includes NEW Abs+Rel)
    pes_3 = [
    # -----------------------------
    # Relative-only (2D bias)
    # -----------------------------
    ("Relative PE (2D bias)", RelativePositionBias2D(n_heads, cfg_3.H, cfg_3.W)),

    # -----------------------------
    # Absolute-only (1D)
    # -----------------------------
    ("Sinusoidal PE (1D abs)", SinusoidalPositionalEncoding(d_model, max_len=max_len_3)),
    ("Learned PE (1D abs)",    LearnedPositionalEncoding1D(d_model, max_len=max_len_3)),

    # -----------------------------
    # Absolute-only (2D)
    # -----------------------------
    ("Sinusoidal PE (2D abs)", SinusoidalPositionalEncoding2D(d_model, cfg_3.H, cfg_3.W)),
    ("Learned PE (2D abs)",    LearnedPositionalEncoding2D(d_model, cfg_3.H, cfg_3.W)),

    # -----------------------------
    # Absolute + Relative (2D)
    # -----------------------------
    ("Sin2D + RelBias2D", Abs2DPlusRelBias2D(
        abs_pe=SinusoidalPositionalEncoding2D(d_model, cfg_3.H, cfg_3.W),
        rel_bias=RelativePositionBias2D(n_heads, cfg_3.H, cfg_3.W),
    )),
    ("Learn2D + RelBias2D", Abs2DPlusRelBias2D(
        abs_pe=LearnedPositionalEncoding2D(d_model, cfg_3.H, cfg_3.W),
        rel_bias=RelativePositionBias2D(n_heads, cfg_3.H, cfg_3.W),
    )),
]

    # ----------------------------------------------------------------------
    # Setting 1: random subset size sweep (fraction of a fixed max_train)
    # ----------------------------------------------------------------------
    print("\n==================== SETTING 1: Random fraction sweep ====================")

    max_train_setting1 = 40000
    n_test_setting1 = 40000
    """
    max_train_setting1 = 10
    n_test_setting1 = 10
    """
    seed_base = 0

    frac_values = [0.025, 0.05, 0.075, 0.1, 0.2, 0.3, 0.4, 0.6, 0.8, 1.0]

    setting1_results: Dict[str, List[float]] = {name: [] for name, _ in pes_3}

    for frac in frac_values:
        n_train = max(1, int(frac * max_train_setting1))
        print(f"\n[Setting 1] frac={frac:.3f}, n_train={n_train}, n_test={n_test_setting1}")

        train_problems, test_problems = generate_setting1_random_fraction(
            cfg_3,
            n_train=n_train,
            n_test=n_test_setting1,
            seed=seed_base + int(frac * 1000),
        )

        for name, pe_module in pes_3:
            print(f"\n--- Setting 1 | frac={frac:.3f} | PE={name} ---")
            acc = train_one_model(
                cfg_3,
                train_problems,
                test_problems,
                pos_enc_name=name,
                pos_enc_module=pe_module,
                device=device,
                d_model=d_model,
                n_heads=n_heads,
                num_layers=num_layers,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_size=batch_size,
                num_epochs=num_epochs,
                lr=lr,
            )
            setting1_results[name].append(acc)

    # Plot Setting 1: accuracy vs fraction
    plt.figure()
    for name in setting1_results:
        plt.plot(frac_values, setting1_results[name], marker="o", label=name)
    plt.xlabel("Training fraction (of max_train_setting1)")
    plt.ylabel("Test masked accuracy")
    plt.ylim(0.0, 1.05)
    plt.title("Setting 1: sample size vs accuracy (3-digit)")
    plt.grid(True)
    plt.legend()
    #plt.savefig("/cluster/project/infk/krause/wnanadavies/deep_learning/plots/setting1_sample_efficiency.png")
    plt.savefig("s1.png")
    plt.close()

    # ----------------------------------------------------------------------
    # Shared pool for settings 2/3/4
    #  - We ensure BOTH carry=0 and carry=1 exist (important for your new samplers).
    #  - We'll use 16 triplets for T/patterns, and 16 disjoint triplets for background B.
    # ----------------------------------------------------------------------

    # Background B (16 total, 8 with cin=0, 8 with cin=1)
    BACKGROUND_B: List[Triplet] = [
        # cin=0 (8)
        (0, 0, 1), (0, 0, 2), (0, 1, 4), (0, 2, 5),
        (0, 3, 6), (0, 4, 7), (0, 5, 8), (0, 6, 9),
        # cin=1 (8)
        (1, 0, 0), (1, 0, 1), (1, 1, 2), (1, 2, 3),
        (1, 3, 4), (1, 4, 5), (1, 5, 6), (1, 6, 7),
    ]

    # Triplets pool for Setting 2 / Setting 4 "forbidden" etc. (16 total, disjoint from BACKGROUND_B)
    # (8 with cin=0, 8 with cin=1)
    TRIPLETS_POOL: List[Triplet] = [
        # cin=0 (8) - chosen to avoid overlap with BACKGROUND_B cin=0 entries
        (0, 7, 0), (0, 7, 2), (0, 8, 1), (0, 8, 3),
        (0, 9, 2), (0, 9, 4), (0, 6, 3), (0, 5, 4),
        # cin=1 (8) - chosen to avoid overlap with BACKGROUND_B cin=1 entries
        (1, 7, 8), (1, 8, 0), (1, 9, 6), (1, 5, 9),
        (1, 8, 9), (1, 4, 8), (1, 7, 9), (1, 9, 2),
    ]

    # ----------------------------------------------------------------------
    # Setting 2: position split with triplets_of_interest; 10-digit addition
    # ----------------------------------------------------------------------
    print("\n==================== SETTING 2: Position split (10 digits) ====================")

    cfg_10 = BoardConfig(H=4, W=15 + 2, n_digits=15)
    max_len_10 = cfg_10.H * cfg_10.W

    pes_10 = [
    # -----------------------------
    # Relative-only (2D bias)
    # -----------------------------
    ("Relative PE (2D bias)", RelativePositionBias2D(n_heads, cfg_10.H, cfg_10.W)),

    # -----------------------------
    # Absolute-only (1D)
    # -----------------------------
    ("Sinusoidal PE (1D abs)", SinusoidalPositionalEncoding(d_model, max_len=max_len_10)),
    ("Learned PE (1D abs)",    LearnedPositionalEncoding1D(d_model, max_len=max_len_10)),

    # -----------------------------
    # Absolute-only (2D)
    # -----------------------------
    ("Sinusoidal PE (2D abs)", SinusoidalPositionalEncoding2D(d_model, cfg_10.H, cfg_10.W)),
    ("Learned PE (2D abs)",    LearnedPositionalEncoding2D(d_model, cfg_10.H, cfg_10.W)),

    # -----------------------------
    # Absolute + Relative (2D)
    # -----------------------------
    ("Sin2D + RelBias2D", Abs2DPlusRelBias2D(
        abs_pe=SinusoidalPositionalEncoding2D(d_model, cfg_10.H, cfg_10.W),
        rel_bias=RelativePositionBias2D(n_heads, cfg_10.H, cfg_10.W),
    )),
    ("Learn2D + RelBias2D", Abs2DPlusRelBias2D(
        abs_pe=LearnedPositionalEncoding2D(d_model, cfg_10.H, cfg_10.W),
        rel_bias=RelativePositionBias2D(n_heads, cfg_10.H, cfg_10.W),
    )),
]


    n_train_setting2 = 30000
    n_test_setting2 = 30000
  
    seed_setting2 = 42

    frac_pos_values = [i / 10.0 for i in range(1, 10)]  # 0.1..0.9

    setting2_results: Dict[str, List[float]] = {name: [] for name, _ in pes_10}

    for frac_pos in frac_pos_values:
        print(f"\n[Setting 2] frac_positions={frac_pos:.1f}, n_train={n_train_setting2}, n_test={n_test_setting2}")

        # UPDATED generate_setting2_position_split returns (train, test, allowed_cols, B_by_cin)
        train_problems, test_problems, allowed_cols, _B2 = generate_setting2_position_split(
            cfg_10,
            n_train=n_train_setting2,
            n_test=n_test_setting2,
            seed=seed_setting2 + int(frac_pos * 100),
            triplets_of_interest=TRIPLETS_POOL,
            frac_positions=frac_pos,
            background_triplets=BACKGROUND_B,
            background_size_per_cin=8,  # unused since background_triplets is provided
            p_use_special=0.5,
            p_use_forbidden=0.5,
        )
        print(f"Allowed training columns per triplet: {allowed_cols}")

        for name, pe_module in pes_10:
            print(f"\n--- Setting 2 | frac_positions={frac_pos:.1f} | PE={name} ---")
            acc = train_one_model(
                cfg_10,
                train_problems,
                test_problems,
                pos_enc_name=name,
                pos_enc_module=pe_module,
                device=device,
                d_model=d_model,
                n_heads=n_heads,
                num_layers=num_layers,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_size=batch_size,
                num_epochs=num_epochs,
                lr=lr,
            )
            setting2_results[name].append(acc)

    # Plot Setting 2
    plt.figure()
    for name in setting2_results:
        plt.plot(frac_pos_values, setting2_results[name], marker="o", label=name)
    plt.xlabel("frac_positions (train columns allowed for each triplet)")
    plt.ylabel("Test masked accuracy")
    plt.ylim(0.0, 1.05)
    plt.title("Setting 2: position split vs accuracy (10-digit)")
    plt.grid(True)
    plt.legend()
    #plt.savefig("/cluster/project/infk/krause/wnanadavies/deep_learning/plots/setting2_position_split_10digit.png")
    plt.savefig("s2.png")
    plt.close()

    # ----------------------------------------------------------------------
    # Setting 3: permutation generalization (a,b)->(b,a), 3 digits
    #   IMPORTANT: train pool has (cin,a,b), test pool uses (cin,b,a)
    #   Ensure both cin=0 and cin=1 appear in the pattern pools.
    # ----------------------------------------------------------------------
    print("\n==================== SETTING 3: Digit-permutation generalization (3 digits) ====================")

    # Choose 16 pattern triplets (8 cin=0, 8 cin=1) from TRIPLETS_POOL
    pattern_train: List[Triplet] = TRIPLETS_POOL[:]  # already 16 with both carries
    pattern_test: List[Triplet] = [(cin, b, a) for (cin, a, b) in pattern_train]

    n_train_setting3 = 40000
    n_test_setting3 = 40000

    seed_setting3 = 123

    # UPDATED generate_setting3_order_constraint returns (train, test, B_by_cin, Ptrain_by_cin, Ptest_by_cin)
    train3, test3, _B3, _Ptr3, _Pte3 = generate_setting3_order_constraint(
        cfg_3,
        n_train=n_train_setting3,
        n_test=n_test_setting3,
        seed=seed_setting3,
        pattern_train=pattern_train,
        pattern_test=pattern_test,
        background_triplets=BACKGROUND_B,
        background_size_per_cin=8,
        p_use_pattern=0.5,
    )

    print("Setting 3 pools (train/test are digit-swaps):")
    print("  pattern_train (first 5):", pattern_train[:5])
    print("  pattern_test  (first 5):", pattern_test[:5])

    setting3_results: Dict[str, float] = {}
    for name, pe_module in pes_3:
        print(f"\n--- Setting 3 | PE={name} ---")
        acc = train_one_model(
            cfg_3,
            train3,
            test3,
            pos_enc_name=name,
            pos_enc_module=pe_module,
            device=device,
            d_model=d_model,
            n_heads=n_heads,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_size=batch_size,
            num_epochs=num_epochs,
            lr=lr,
        )
        print(f"[Setting 3] {name} test accuracy: {acc:.4f}")
        setting3_results[name] = acc

    with open("setting3_results.txt", "w") as f:
        f.write("=== Setting 3: Digit-permutation generalization (3 digits) ===\n")
        f.write(f"pattern_train (len={len(pattern_train)}): {pattern_train}\n")
        f.write(f"pattern_test  (len={len(pattern_test)}): {pattern_test}\n\n")
        for name, acc in setting3_results.items():
            f.write(f"{name}: test_accuracy = {acc:.6f}\n")

    # ----------------------------------------------------------------------
    # Setting 4: triplet hold-out (forbidden never in train, required in test), 3 digits
    #   UPDATED generator returns (train, test, B_by_cin, H_by_cin)
    # ----------------------------------------------------------------------
    print("\n==================== SETTING 4: Triplet hold-out (3 digits) ====================")

    forbidden_triplets: List[Triplet] = TRIPLETS_POOL[:]  # all 16 are forbidden (both carries)

    n_train_setting4 = 40000
    n_test_setting4 = 40000
 
    seed_setting4 = 999

    train4, test4, _B4, _H4 = generate_setting4_triplet_holdout(
        cfg_3,
        n_train=n_train_setting4,
        n_test=n_test_setting4,
        seed=seed_setting4,
        forbidden_triplets=forbidden_triplets,
        background_triplets=BACKGROUND_B,
        background_size_per_cin=8,
        p_use_forbidden=0.5,
    )

    print("Setting 4 forbidden triplets (len={}):".format(len(forbidden_triplets)))
    print("  (first 10):", forbidden_triplets[:10])

    setting4_results: Dict[str, float] = {}
    for name, pe_module in pes_3:
        print(f"\n--- Setting 4 | PE={name} ---")
        acc = train_one_model(
            cfg_3,
            train4,
            test4,
            pos_enc_name=name,
            pos_enc_module=pe_module,
            device=device,
            d_model=d_model,
            n_heads=n_heads,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_size=batch_size,
            num_epochs=num_epochs,
            lr=lr,
        )
        print(f"[Setting 4] {name} test accuracy: {acc:.4f}")
        setting4_results[name] = acc

    with open("setting4_results.txt", "w") as f:
        f.write("=== Setting 4: Triplet hold-out (3 digits) ===\n")
        f.write(f"forbidden_triplets (len={len(forbidden_triplets)}): {forbidden_triplets}\n\n")
        for name, acc in setting4_results.items():
            f.write(f"{name}: test_accuracy = {acc:.6f}\n")




if __name__ == "__main__":
    main()

