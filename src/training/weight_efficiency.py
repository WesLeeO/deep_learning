# src/training/pe_scaling_trainer.py

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from typing import List, Tuple, Dict

from src.data.addition_algo import BoardConfig
from src.data.problems import generate_diversified_problems
from src.data.board_dataset import BlackboardAdditionStepDataset
from src.models.transformers import BlackboardTransformer
from src.models.positional_encodings import *


# ---------------------------------------------------------------------------
# Loss + accuracy utilities (same spirit as your other trainer)
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
    d_model: int,
    n_heads: int,
    num_layers: int,
    dim_feedforward: int,
    dropout: float,
    batch_size: int,
    num_epochs: int,
    lr: float,
) -> Tuple[float, int]:
    """
    Train a BlackboardTransformer with the given hyperparameters + positional encoding
    on train_problems, evaluate on test_problems, and return:

        (test_accuracy, total_trainable_parameters)
    """
    # Datasets / loaders
    train_ds = BlackboardAdditionStepDataset(train_problems)
    test_ds  = BlackboardAdditionStepDataset(test_problems)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False)

    vocab_size = 12
    max_len = cfg.H * cfg.W

    # Instantiate model
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

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(
        f"[{pos_enc_name}] d_model={d_model}, n_heads={n_heads}, "
        f"layers={num_layers}, d_ff={dim_feedforward} "
        f"→ trainable params: {total_params:,}"
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss = 0.0
        total_tokens = 0
        total_correct = 0

        pbar = tqdm(
            train_loader,
            desc=(f"{pos_enc_name} | dm={d_model}, nh={n_heads}, "
                  f"L={num_layers}, dff={dim_feedforward} | epoch {epoch}/{num_epochs}")
        )
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
        f"[{pos_enc_name}] FINAL test loss/token: {test_loss:.4f} "
        f"| test acc(masked): {test_acc:.4f}"
    )
    return test_acc, total_params


# ---------------------------------------------------------------------------
# Main scaling experiments (all on 5-digit addition)
# ---------------------------------------------------------------------------

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    torch.manual_seed(0)

    # -------------------------
    # Data: 5-digit addition
    # -------------------------
    n_digits = 7
    cfg = BoardConfig(H=4, W=n_digits + 2, n_digits=n_digits)
    max_len = cfg.H * cfg.W

    n_train = 150000
    n_test  = 150000

    print(f"Generating data: {n_train} train problems, {n_test} test problems (5-digit)")
    train_problems = generate_diversified_problems(cfg, n_train, seed=0)
    test_problems  = generate_diversified_problems(cfg, n_test,  seed=1)

    dropout = 0.1
    batch_size = 256
    num_epochs = 8
    lr = 3e-4

    # Include your mixed variant here
    # Include all variants here
    pe_names = [
        "Relative PE (2D bias)",
        "Sinusoidal PE (1D abs)",
        "Learned PE (1D abs)",
        "Sinusoidal PE (2D abs)",
        "Learned PE (2D abs)",
        "Sin2D + RelBias2D",
        "Learn2D + RelBias2D",
    ]

    def build_pos_enc(pe_name: str, d_model: int, n_heads: int):
        """Factory: build the PE module consistent with d_model / n_heads."""
        if pe_name == "Relative PE (2D bias)":
            return RelativePositionBias2D(n_heads, cfg.H, cfg.W)

        # 1D absolute
        if pe_name == "Sinusoidal PE (1D abs)":
            return SinusoidalPositionalEncoding(d_model, max_len=max_len)
        if pe_name == "Learned PE (1D abs)":
            return LearnedPositionalEncoding1D(d_model, max_len=max_len)

        # 2D absolute
        if pe_name == "Sinusoidal PE (2D abs)":
            return SinusoidalPositionalEncoding2D(d_model, cfg.H, cfg.W)
        if pe_name == "Learned PE (2D abs)":
            return LearnedPositionalEncoding2D(d_model, cfg.H, cfg.W)

        # 2D absolute + 2D relative bias
        if pe_name == "Sin2D + RelBias2D":
            return Abs2DPlusRelBias2D(
                abs_pe=SinusoidalPositionalEncoding2D(d_model, cfg.H, cfg.W),
                rel_bias=RelativePositionBias2D(n_heads, cfg.H, cfg.W),
            )
        if pe_name == "Learn2D + RelBias2D":
            return Abs2DPlusRelBias2D(
                abs_pe=LearnedPositionalEncoding2D(d_model, cfg.H, cfg.W),
                rel_bias=RelativePositionBias2D(n_heads, cfg.H, cfg.W),
            )

        raise ValueError(f"Unknown PE name: {pe_name}")


    # ----------------------------------------------------------------------
    # Experiment 1: width sweep (vary d_model, d_ff), fixed layers & heads
    # ----------------------------------------------------------------------

    """
    print("\n==================== Experiment 1: Width sweep ====================")

    width_configs: List[Tuple[int, int]] = [
        (64, 256),
        (96, 384),
        (128, 512),
        (192, 768),
        (256, 1024),
    ]
    fixed_heads_exp1 = 2
    fixed_layers_exp1 = 2

    exp1_results: Dict[str, Dict[str, List[float]]] = {
        name: {"params": [], "acc": []} for name in pe_names
    }

    for (d_model, d_ff) in width_configs:
        if d_model % fixed_heads_exp1 != 0:
            print(f"Skipping d_model={d_model} because not divisible by heads={fixed_heads_exp1}")
            continue

        print(f"\n[Exp1] d_model={d_model}, d_ff={d_ff}, layers={fixed_layers_exp1}, heads={fixed_heads_exp1}")

        for pe_name in pe_names:
            pos_enc = build_pos_enc(pe_name, d_model=d_model, n_heads=fixed_heads_exp1)

            acc, n_params = train_one_model(
                cfg=cfg,
                train_problems=train_problems,
                test_problems=test_problems,
                pos_enc_name=f"{pe_name} (width sweep)",
                pos_enc_module=pos_enc,
                device=device,
                d_model=d_model,
                n_heads=fixed_heads_exp1,
                num_layers=fixed_layers_exp1,
                dim_feedforward=d_ff,
                dropout=dropout,
                batch_size=batch_size,
                num_epochs=num_epochs,
                lr=lr,
            )

            exp1_results[pe_name]["params"].append(n_params)
            exp1_results[pe_name]["acc"].append(acc)

    plt.figure()
    for pe_name in pe_names:
        params = exp1_results[pe_name]["params"]
        accs   = exp1_results[pe_name]["acc"]
        pairs = sorted(zip(params, accs), key=lambda x: x[0])
        if not pairs:
            continue
        p_sorted, a_sorted = zip(*pairs)
        plt.plot(p_sorted, a_sorted, marker="o", label=pe_name)

    plt.xlabel("Total trainable parameters")
    plt.ylabel("Test masked accuracy")
    plt.ylim(0.0, 1.05)
    plt.title("Scaling with width (5-digit addition)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("pe_scaling_exp1_width_vs_params.png")
    plt.close()
    """
    # ----------------------------------------------------------------------
    # Experiment 2: depth sweep (vary num_layers), fixed width & heads
    # ----------------------------------------------------------------------
    print("\n==================== Experiment 2: Depth sweep ====================")

    d_model_exp2 = 128
    d_ff_exp2    = 512

    # test both 1 head and 2 heads (keep everything else the same)
    heads_list = [1, 2]

    layers_list = [1, 2, 3, 4, 5, 6]

    # results indexed by n_heads, then pe_name
    exp2_results: Dict[int, Dict[str, Dict[str, List[float]]]] = {
        nh: {name: {"layers": [], "acc": []} for name in pe_names}
        for nh in heads_list
    }

    for n_heads_exp2 in heads_list:
        if d_model_exp2 % n_heads_exp2 != 0:
            raise ValueError(f"d_model_exp2 must be divisible by n_heads_exp2 (got {d_model_exp2} % {n_heads_exp2})")

        for L in layers_list:
            print(f"\n[Exp2] num_layers={L}, d_model={d_model_exp2}, d_ff={d_ff_exp2}, heads={n_heads_exp2}")

            for pe_name in pe_names:
                pos_enc = build_pos_enc(pe_name, d_model=d_model_exp2, n_heads=n_heads_exp2)

                acc, _ = train_one_model(
                    cfg=cfg,
                    train_problems=train_problems,
                    test_problems=test_problems,
                    pos_enc_name=f"{pe_name} (depth sweep, heads={n_heads_exp2})",
                    pos_enc_module=pos_enc,
                    device=device,
                    d_model=d_model_exp2,
                    n_heads=n_heads_exp2,
                    num_layers=L,
                    dim_feedforward=d_ff_exp2,
                    dropout=dropout,
                    batch_size=batch_size,
                    num_epochs=num_epochs,
                    lr=lr,
                )

                exp2_results[n_heads_exp2][pe_name]["layers"].append(L)
                exp2_results[n_heads_exp2][pe_name]["acc"].append(acc)

    # plot: one figure per head count
    for n_heads_exp2 in heads_list:
        plt.figure()
        for pe_name in pe_names:
            layers = exp2_results[n_heads_exp2][pe_name]["layers"]
            accs   = exp2_results[n_heads_exp2][pe_name]["acc"]
            plt.plot(layers, accs, marker="o", label=pe_name)

        plt.xlabel("Number of Transformer layers")
        plt.ylabel("Test masked accuracy")
        plt.ylim(0.0, 1.05)
        plt.title(f"Scaling with depth (7-digit addition) - heads={n_heads_exp2}")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"pe_scaling_exp2_depth_vs_acc_heads{n_heads_exp2}.png")
        plt.close()


    # ----------------------------------------------------------------------
    # Experiment 3: head sweep (vary n_heads), fixed width & depth
    # ----------------------------------------------------------------------
    """
    print("\n==================== Experiment 3: Head sweep ====================")

    d_model_exp3 = 132
    d_ff_exp3    = 512
    num_layers_exp3 = 2

    heads_list = [1, 2, 3, 4]  # valid because 120 divisible by all of these

    exp3_results: Dict[str, Dict[str, List[float]]] = {
        name: {"heads": [], "acc": []} for name in pe_names
    }

    for nh in heads_list:
        if d_model_exp3 % nh != 0:
            print(f"Skipping nh={nh} because d_model={d_model_exp3} not divisible by nh")
            continue

        print(f"\n[Exp3] n_heads={nh}, d_model={d_model_exp3}, d_ff={d_ff_exp3}, layers={num_layers_exp3}")

        for pe_name in pe_names:
            pos_enc = build_pos_enc(pe_name, d_model=d_model_exp3, n_heads=nh)

            acc, _ = train_one_model(
                cfg=cfg,
                train_problems=train_problems,
                test_problems=test_problems,
                pos_enc_name=f"{pe_name} (head sweep)",
                pos_enc_module=pos_enc,
                device=device,
                d_model=d_model_exp3,
                n_heads=nh,
                num_layers=num_layers_exp3,
                dim_feedforward=d_ff_exp3,
                dropout=dropout,
                batch_size=batch_size,
                num_epochs=num_epochs,
                lr=lr,
            )

            exp3_results[pe_name]["heads"].append(nh)
            exp3_results[pe_name]["acc"].append(acc)

    plt.figure()
    for pe_name in pe_names:
        heads = exp3_results[pe_name]["heads"]
        accs  = exp3_results[pe_name]["acc"]
        plt.plot(heads, accs, marker="o", label=pe_name)

    plt.xlabel("Number of attention heads")
    plt.ylabel("Test masked accuracy")
    plt.ylim(0.0, 1.05)
    plt.title("Scaling with heads (5-digit addition)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("pe_scaling_exp3_heads_vs_acc.png")
    plt.close()
    """

if __name__ == "__main__":
    main()


