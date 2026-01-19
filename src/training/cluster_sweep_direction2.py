import os, sys
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
import json
import csv
import time
import argparse
from dataclasses import asdict
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from src.data.addition_algo import (
    BoardConfig,
    generate_trajectory_variant_A,
    BLANK_TOKEN,
    VOID_TOKEN,
)

from src.data.problems import generate_diversified_problems

from src.data.board_dataset import (
    BlackboardAdditionStepDataset,                # classic
    BlackboardAdditionDenoisingStepDataset,       # local/global
)

from src.models.transformers import BlackboardTransformer
from src.models.positional_encodings import (
    LearnedPositionalEncoding1D,
    SinusoidalPositionalEncoding,       # 1D sinus
    LearnedPositionalEncoding2D,
    SinusoidalPositionalEncoding2D,     # 2D sinus
    RelativePositionBias2D,
    Abs2DPlusRelBias2D,
)

from src.training.configs import ModelConfig


# -------------------------
# Helpers
# -------------------------
def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def model_cfg_name(cfg: ModelConfig) -> str:
    return f"d{cfg.d_model}_h{cfg.nhead}_L{cfg.num_layers}_ff{cfg.dim_feedforward}"


def parse_int_list(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]

def write_history_csv(path: str, train_loss: List[float], val_acc: List[float]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "train_loss", "val_masked_acc"])
        for i, (tl, va) in enumerate(zip(train_loss, val_acc), start=1):
            w.writerow([i, tl, va])


def plot_curve(out_path: str, ys: List[float], title: str, ylabel: str):
    plt.figure(figsize=(6, 4))
    plt.plot(range(1, len(ys) + 1), ys)
    plt.xlabel("epoch")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

# -------------------------
# New PE list
# -------------------------
def make_pes(model_cfg: ModelConfig, board_cfg: BoardConfig):
    return [
        (
            "abs_1d_learned",
            LearnedPositionalEncoding1D(model_cfg.d_model, board_cfg.H * board_cfg.W),
        ),
        ("abs_1d_sinusoidal",
         SinusoidalPositionalEncoding(model_cfg.d_model, model_cfg.max_len)),
        (
            "abs_2d_learned",
            LearnedPositionalEncoding2D(model_cfg.d_model, board_cfg.H, board_cfg.W),
        ),
        ("abs_2d_sinusoidal",
         SinusoidalPositionalEncoding2D(model_cfg.d_model, board_cfg.H, board_cfg.W)),
        (
            "rel_2d_bias",
            RelativePositionBias2D(model_cfg.nhead, board_cfg.H, board_cfg.W),
        ),
        

    ]


# -------------------------
# Loss / metrics
# -------------------------
def masked_cross_entropy(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    logits:  (B, L, V)
    targets: (B, L)
    mask:    (B, L) bool
    """
    V = logits.size(-1)
    logits_f = logits.reshape(-1, V)
    targets_f = targets.reshape(-1)
    mask_f = mask.reshape(-1).bool()

    if mask_f.sum().item() == 0:
        return torch.tensor(0.0, device=logits.device)

    return F.cross_entropy(logits_f[mask_f], targets_f[mask_f])


@torch.no_grad()
def masked_accuracy(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> float:
    preds = logits.argmax(dim=-1)
    correct = ((preds == targets) & mask).sum().item()
    total = mask.sum().item()
    return float(correct) / float(max(total, 1))


# -------------------------
# Rollout + noise (flip digit)
# -------------------------
def stepwrite_mask(cfg: BoardConfig, step_idx: int) -> torch.Tensor:
    H, W = cfg.H, cfg.W
    L = H * W
    m = torch.zeros(L, dtype=torch.bool)
    col_end = W - 1
    col = col_end - step_idx
    if 0 <= col < W:
        m[cfg.result_row * W + col] = True
    if 0 <= (col - 1) < W:
        m[cfg.carry_row * W + (col - 1)] = True
    return m


def prev_step_mask(cfg: BoardConfig, step_idx: int) -> torch.Tensor:
    H, W = cfg.H, cfg.W
    L = H * W
    m = torch.zeros(L, dtype=torch.bool)
    if step_idx <= 0:
        return m
    col_end = W - 1
    col_prev = col_end - (step_idx - 1)
    if 0 <= col_prev < W:
        m[cfg.result_row * W + col_prev] = True
    if 0 <= (col_prev - 1) < W:
        m[cfg.carry_row * W + (col_prev - 1)] = True
    return m


def editable_mask_global(cfg: BoardConfig) -> torch.Tensor:
    H, W = cfg.H, cfg.W
    L = H * W
    m = torch.zeros(L, dtype=torch.bool)
    for r in [cfg.carry_row, cfg.result_row]:
        m[r * W : r * W + W] = True
    return m


def inject_rollout_noise_inplace(
    board: torch.Tensor,
    cfg: BoardConfig,
    step_idx: int,
    rng: torch.Generator,
    p_noise: float,
    n_noise: int = 1,
):
    """
    Flip-digit noise on already-written result/carry cells for steps < step_idx.
    Triggered with prob p_noise per rollout iteration.
    Returns: list[int] positions that were corrupted (may be empty).
    """

    corrupted_positions = []
    if p_noise <= 0.0 or step_idx <= 0:
        return corrupted_positions

    u = torch.rand((), generator=rng, device=board.device).item()
    if u > p_noise:
        return corrupted_positions

    W = cfg.W
    cand = []

    for s in range(step_idx):
        c_res = (W - 1) - s
        if 0 <= c_res < W:
            cand.append(cfg.result_row * W + c_res)

        c_car = (W - 2) - s
        if 0 <= c_car < W:
            cand.append(cfg.carry_row * W + c_car)

    if not cand:
        return corrupted_positions

    for _ in range(n_noise):
        j = int(torch.randint(0, len(cand), (1,), generator=rng, device=board.device).item())
        pos = cand[j]
        old = int(board[pos].item())
        if 0 <= old <= 9:
            d = int(torch.randint(0, 9, (1,), generator=rng, device=board.device).item())
            new = d if d < old else d + 1
        else:
            new = int(torch.randint(0, 10, (1,), generator=rng, device=board.device).item())
        board[pos] = new
        corrupted_positions.append(pos)

    return corrupted_positions


def result_positions(cfg: BoardConfig) -> torch.Tensor:
    # result digits are the last n_digits columns in the result row
    cols = list(range(cfg.W - cfg.n_digits, cfg.W))
    idxs = [cfg.result_row * cfg.W + c for c in cols]
    return torch.tensor(idxs, dtype=torch.long)


@torch.no_grad()
def rollout_one(
    model: torch.nn.Module,
    cfg: BoardConfig,
    xs: np.ndarray,
    setting: str,         # "classic" | "local" | "global"
    p_noise: float,
    seed: int,
    max_iters: int = 400,
):
    device = next(model.parameters()).device
    rng_torch = torch.Generator(device=device)
    rng_torch.manual_seed(seed)

    S_seq, _ = generate_trajectory_variant_A(cfg, xs)  # teacher trajectory
    board = torch.from_numpy(S_seq[0]).view(-1).long().to(device)

    recovery_events = []  # [(pos, correct_token), ...]  <-- NEW

    t = 0
    iters = 0
    W = cfg.W
    col_end = W - 1

    while t < cfg.n_digits and iters < max_iters:
        iters += 1

        # --- NEW: inject noise AND record "what should it be"
        corrupted_positions = inject_rollout_noise_inplace(
            board, cfg, t, rng_torch, p_noise=p_noise, n_noise=1
        )
        if corrupted_positions:
            teacher_flat_t = S_seq[t].reshape(-1)  # correct board at step pointer t
            for pos in corrupted_positions:
                recovery_events.append((pos, int(teacher_flat_t[pos])))

        out = model(board.unsqueeze(0))
        logits = out[0] if isinstance(out, (tuple, list)) else out
        pred = logits.argmax(dim=-1).squeeze(0)

        if setting == "classic":
            sm = stepwrite_mask(cfg, t).to(device)
            board[sm] = pred[sm]
            t += 1
            continue

        if setting == "local":
            pm = prev_step_mask(cfg, t).to(device)
            if pm.any() and (pred[pm] == VOID_TOKEN).any():
                board[pm] = VOID_TOKEN
                t = max(t - 1, 0)
                continue

            sm = stepwrite_mask(cfg, t).to(device)
            board[sm] = pred[sm]
            t += 1
            continue

        if setting == "global":
            em = editable_mask_global(cfg).to(device)
            void_pos = em & (pred == VOID_TOKEN)
            if void_pos.any():
                board[void_pos] = VOID_TOKEN

                erased_cols = []
                idxs = void_pos.nonzero(as_tuple=False).view(-1).tolist()
                for idx in idxs:
                    r = idx // W
                    c = idx % W
                    if r == cfg.result_row:
                        erased_cols.append(c)
                    elif r == cfg.carry_row:
                        erased_cols.append(c + 1)

                erased_cols = [c for c in erased_cols if 0 <= c < W]
                if erased_cols:
                    cmax = max(erased_cols)
                    t = max(0, col_end - cmax)
                else:
                    t = max(t - 1, 0)
                continue

            sm = stepwrite_mask(cfg, t).to(device)
            board[sm] = pred[sm]
            t += 1
            continue

        raise ValueError(f"Unknown setting: {setting}")

    finished = (t >= cfg.n_digits)  # <-- NEW
    return board.cpu(), recovery_events, finished



@torch.no_grad()
def rollout_metrics(
    model: torch.nn.Module,
    cfg: BoardConfig,
    problems,
    setting: str,
    p_noise: float,
    seed: int,
):
    rp = result_positions(cfg)

    n_ok_exact = 0
    digit_correct = 0
    digit_total = 0

    n_inj = 0
    n_fixed = 0

    n_finished = 0

    for i, pr in enumerate(problems):
        xs = pr.operands
        S_seq, _ = generate_trajectory_variant_A(cfg, xs)
        target_final = torch.from_numpy(S_seq[-1]).view(-1).long()

        pred_final, recovery_events, finished = rollout_one(
            model=model,
            cfg=cfg,
            xs=xs,
            setting=setting,
            p_noise=p_noise,
            seed=seed + i,
            max_iters=400,
        )

        if finished:
            n_finished += 1

        # exact full-board
        if torch.equal(pred_final, target_final):
            n_ok_exact += 1

        # digit accuracy (result row digits)
        digit_correct += (pred_final[rp] == target_final[rp]).sum().item()
        digit_total += rp.numel()

        # recovery rate
        for (pos, correct_tok) in recovery_events:
            n_inj += 1
            if int(pred_final[pos].item()) == int(correct_tok):
                n_fixed += 1

    exact_acc = n_ok_exact / len(problems)
    digit_acc = digit_correct / max(digit_total, 1)
    recovery_rate = (n_fixed / n_inj) if n_inj > 0 else float("nan")
    finish_rate = n_finished / len(problems)

    return {
        "exact_acc": exact_acc,
        "digit_acc": digit_acc,
        "recovery_rate": recovery_rate,
        "finish_rate": finish_rate,
        "n_injected": n_inj,
    }




# -------------------------
# Training (simple loop, saves best-by-val)
# -------------------------
def train_one(
    *,
    device: torch.device,
    board_cfg: BoardConfig,
    model_cfg: ModelConfig,
    pe_name: str,
    pe_module: torch.nn.Module,
    setting: str,
    vocab_size: int,
    train_problems,
    val_problems,
    batch_size: int,
    epochs: int,
    lr: float,
    denoise_rate: float,
    p_revert: float,
    seed: int,
) -> Tuple[Dict[str, torch.Tensor], Dict]:
    set_seed(seed)

    if setting == "classic":
        train_ds = BlackboardAdditionStepDataset(train_problems)
        val_ds   = BlackboardAdditionStepDataset(val_problems)
    else:
        train_ds = BlackboardAdditionDenoisingStepDataset(
            train_problems,
            setting=setting,
            denoise_rate=denoise_rate,
            p_revert=p_revert,
            seed=seed,
        )
        val_ds   = BlackboardAdditionDenoisingStepDataset(
            val_problems,
            setting=setting,
            denoise_rate=denoise_rate,
            p_revert=p_revert,
            seed=seed + 1,
        )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0)

    model = BlackboardTransformer(
        vocab_size=vocab_size,
        pos_enc=pe_module,
        **asdict(model_cfg),
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=lr)

    best_val = -1.0
    best_epoch = -1
    best_state = None

    history_train_loss = []
    history_val_acc = []

    for ep in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        n_batches = 0

        for batch in train_loader:
            x = batch["input_ids"].to(device)
            y = batch["target_ids"].to(device)
            m = batch["mask"].to(device).bool()

            opt.zero_grad(set_to_none=True)
            out = model(x)
            logits = out[0] if isinstance(out, (tuple, list)) else out
            loss = masked_cross_entropy(logits, y, m)

            loss.backward()
            opt.step()

            total_loss += float(loss.item())
            n_batches += 1

        train_loss = total_loss / max(n_batches, 1)
        history_train_loss.append(train_loss)

        # val masked acc
        model.eval()
        vals = []
        for batch in val_loader:
            x = batch["input_ids"].to(device)
            y = batch["target_ids"].to(device)
            m = batch["mask"].to(device).bool()
            out = model(x)
            logits = out[0] if isinstance(out, (tuple, list)) else out
            vals.append(masked_accuracy(logits, y, m))
        val_acc = float(np.mean(vals)) if vals else 0.0
        history_val_acc.append(val_acc)

        if val_acc > best_val:
            best_val = val_acc
            best_epoch = ep
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    info = {
        "best_val_masked_acc": float(best_val),
        "best_epoch": int(best_epoch),
        "train_loss": history_train_loss,
        "val_masked_acc": history_val_acc,
    }
    return best_state, info



def build_model_with_state(
    device: torch.device,
    model_cfg: ModelConfig,
    board_cfg: BoardConfig,
    pe_name: str,
    vocab_size: int,
    state_dict: Dict[str, torch.Tensor],
) -> torch.nn.Module:
    pe_dict = dict(make_pes(model_cfg, board_cfg))
    pe_module = pe_dict[pe_name]
    model = BlackboardTransformer(
        vocab_size=vocab_size,
        pos_enc=pe_module,
        **asdict(model_cfg),
    ).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


# -------------------------
# Plotting + saving
# -------------------------
def plot_grouped_bars(out_path: str, pe_names: List[str],
                      classic_vals: List[float], local_vals: List[float], global_vals: List[float],
                      title: str,
                      ylabel: str):
    x = np.arange(len(pe_names))
    width = 0.25

    plt.figure(figsize=(max(10, len(pe_names) * 1.4), 5))
    plt.bar(x - width, classic_vals, width, label="classic")
    plt.bar(x,         local_vals,   width, label="local")
    plt.bar(x + width, global_vals,  width, label="global")

    plt.xticks(x, pe_names, rotation=20, ha="right")
    plt.ylim(0, 1.0)
    plt.ylabel("Rollout exact accuracy")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def write_csv(path: str, rows: List[Dict]):
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


# -------------------------
# Main: folder structure you requested
# -------------------------
def main():
    import hashlib

    p = argparse.ArgumentParser()

    p.add_argument("--out-dir", type=str, default="cluster_direction2_outputs2")
    p.add_argument("--seed", type=int, default=42)

    # sweep
    p.add_argument(
        "--train-sizes",
        type=str,
        default="100000",
    )
    p.add_argument("--n-val", type=int, default=10000)

    # task (5 digits)
    p.add_argument("--n-digits", type=int, default=5)
    p.add_argument("--H", type=int, default=4)
    p.add_argument("--W", type=int, default=7)  # must satisfy W >= n_digits + 2

    # train hyperparams
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=3e-4)

    # direction2 hyperparams
    p.add_argument("--denoise-rate", type=float, default=0.15)
    p.add_argument("--p-revert", type=float, default=0.25)

    # eval
    p.add_argument("--n-test", type=int, default=5000)
    p.add_argument("--noise-p", type=float, default=0.1)

    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    if args.W < args.n_digits + 2:
        raise ValueError(f"W must be >= n_digits + 2. Got W={args.W}, n_digits={args.n_digits}.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    set_seed(args.seed)

    board_cfg = BoardConfig(H=args.H, W=args.W, n_digits=args.n_digits)
    train_sizes = parse_int_list(args.train_sizes)

    # fixed test set for fair comparisons
    test_problems = generate_diversified_problems(board_cfg, args.n_test, seed=999)

    # model configs (same as blackboard.py)
    model_cfgs = [
        ModelConfig(d_model=64,  nhead=1, num_layers=2, dim_feedforward=256, dropout=0.1, max_len=200),
        ModelConfig(d_model=128, nhead=2, num_layers=3, dim_feedforward=512, dropout=0.1, max_len=200),
        ModelConfig(d_model=256, nhead=4, num_layers=4, dim_feedforward=512, dropout=0.1, max_len=200),
        ModelConfig(d_model=256, nhead=8, num_layers=6, dim_feedforward=1024, dropout=0.1, max_len=200),
    ]
        # ModelConfig(d_model=128, nhead=2, num_layers=3, dim_feedforward=512, dropout=0.1, max_len=200),
        # ModelConfig(d_model=256, nhead=4, num_layers=4, dim_feedforward=512, dropout=0.1, max_len=200),
        # ModelConfig(d_model=256, nhead=8, num_layers=6, dim_feedforward=1024, dropout=0.1, max_len=200),

    def stable_seed(tag: str) -> int:
        # stable across machines/runs (unlike Python's hash())
        h = hashlib.md5(tag.encode("utf-8")).hexdigest()
        return (args.seed + int(h[:8], 16)) % (2**31 - 1)

    for model_cfg in model_cfgs:
        cfg_name = model_cfg_name(model_cfg)
        model_dir = os.path.join(args.out_dir, cfg_name)
        os.makedirs(model_dir, exist_ok=True)

        print(f"\n==============================")
        print(f"MODEL CFG: {cfg_name}")
        print(f"==============================")

        for n_train in train_sizes:
            run_dir = os.path.join(model_dir, f"train_{n_train}")
            os.makedirs(run_dir, exist_ok=True)

            print(f"\n--- Training size: {n_train} ---")
            t0 = time.time()

            # data
            train_problems = generate_diversified_problems(board_cfg, n_train, seed=1000 + n_train)
            val_problems   = generate_diversified_problems(board_cfg, args.n_val, seed=2000 + n_train)

            pe_list = make_pes(model_cfg, board_cfg)
            pe_names = [name for name, _ in pe_list]

            # collect rollout accuracies for bar plots
            exact0 = {pe: {"classic": 0.0, "local": 0.0, "global": 0.0} for pe in pe_names}
            digit0 = {pe: {"classic": 0.0, "local": 0.0, "global": 0.0} for pe in pe_names}

            exact1 = {pe: {"classic": 0.0, "local": 0.0, "global": 0.0} for pe in pe_names}
            digit1 = {pe: {"classic": 0.0, "local": 0.0, "global": 0.0} for pe in pe_names}
            reco1  = {pe: {"classic": 0.0, "local": 0.0, "global": 0.0} for pe in pe_names}

            # per-PE summary table
            rows = []

            for pe_name, _ in pe_list:
                print(f"\n>>> PE: {pe_name}")
                # we'll re-instantiate fresh PE modules each time (safer than sharing modules across trainings)
                # ---- CLASSIC (vocab=12) ----
                pe_c = dict(make_pes(model_cfg, board_cfg))[pe_name]
                state_c, info_c = train_one(
                    device=device,
                    board_cfg=board_cfg,
                    model_cfg=model_cfg,
                    pe_name=pe_name,
                    pe_module=pe_c,
                    setting="classic",
                    vocab_size=12,
                    train_problems=train_problems,
                    val_problems=val_problems,
                    batch_size=args.batch_size,
                    epochs=args.epochs,
                    lr=args.lr,
                    denoise_rate=args.denoise_rate,
                    p_revert=args.p_revert,
                    seed=stable_seed(f"{cfg_name}|{n_train}|{pe_name}|classic"),
                )

                subdir_c = os.path.join(run_dir, pe_name, "classic")
                os.makedirs(subdir_c, exist_ok=True)

                torch.save(
                    {
                        "model_state_dict": state_c,
                        "model_cfg": asdict(model_cfg),
                        "board_cfg": {"H": board_cfg.H, "W": board_cfg.W, "n_digits": board_cfg.n_digits},
                        "pe": pe_name,
                        "setting": "classic",
                        "vocab_size": 12,
                        "best_val_masked_acc": info_c["best_val_masked_acc"],
                        "best_epoch": info_c["best_epoch"],
                    },
                    os.path.join(subdir_c, "best_model.pt"),
                )
                write_history_csv(os.path.join(subdir_c, "history.csv"), info_c["train_loss"], info_c["val_masked_acc"])
                plot_curve(
                    os.path.join(subdir_c, "train_loss.png"),
                    info_c["train_loss"],
                    title=f"{cfg_name} | train={n_train} | {pe_name} | classic",
                    ylabel="train loss",
                )
                plot_curve(
                    os.path.join(subdir_c, "val_masked_acc.png"),
                    info_c["val_masked_acc"],
                    title=f"{cfg_name} | train={n_train} | {pe_name} | classic",
                    ylabel="val masked acc",
                )

                model_c = build_model_with_state(device, model_cfg, board_cfg, pe_name, 12, state_c)
                m0_c = rollout_metrics(model_c, board_cfg, test_problems, "classic", p_noise=0.0,         seed=args.seed + 12345)
                m1_c = rollout_metrics(model_c, board_cfg, test_problems, "classic", p_noise=args.noise_p, seed=args.seed + 54321)

                exact0[pe_name]["classic"] = m0_c["exact_acc"]
                digit0[pe_name]["classic"] = m0_c["digit_acc"]

                exact1[pe_name]["classic"] = m1_c["exact_acc"]
                digit1[pe_name]["classic"] = m1_c["digit_acc"]
                reco1[pe_name]["classic"]  = m1_c["recovery_rate"]

                # ---- LOCAL (vocab=13) ----
                pe_l = dict(make_pes(model_cfg, board_cfg))[pe_name]
                state_l, info_l = train_one(
                    device=device,
                    board_cfg=board_cfg,
                    model_cfg=model_cfg,
                    pe_name=pe_name,
                    pe_module=pe_l,
                    setting="local",
                    vocab_size=13,
                    train_problems=train_problems,
                    val_problems=val_problems,
                    batch_size=args.batch_size,
                    epochs=args.epochs,
                    lr=args.lr,
                    denoise_rate=args.denoise_rate,
                    p_revert=args.p_revert,
                    seed=stable_seed(f"{cfg_name}|{n_train}|{pe_name}|local"),
                )

                subdir_l = os.path.join(run_dir, pe_name, "local")
                os.makedirs(subdir_l, exist_ok=True)

                torch.save(
                    {
                        "model_state_dict": state_l,
                        "model_cfg": asdict(model_cfg),
                        "board_cfg": {"H": board_cfg.H, "W": board_cfg.W, "n_digits": board_cfg.n_digits},
                        "pe": pe_name,
                        "setting": "local",
                        "vocab_size": 13,
                        "best_val_masked_acc": info_l["best_val_masked_acc"],
                        "best_epoch": info_l["best_epoch"],
                    },
                    os.path.join(subdir_l, "best_model.pt"),
                )
                write_history_csv(os.path.join(subdir_l, "history.csv"), info_l["train_loss"], info_l["val_masked_acc"])
                plot_curve(
                    os.path.join(subdir_l, "train_loss.png"),
                    info_l["train_loss"],
                    title=f"{cfg_name} | train={n_train} | {pe_name} | local",
                    ylabel="train loss",
                )
                plot_curve(
                    os.path.join(subdir_l, "val_masked_acc.png"),
                    info_l["val_masked_acc"],
                    title=f"{cfg_name} | train={n_train} | {pe_name} | local",
                    ylabel="val masked acc",
                )

                model_l = build_model_with_state(device, model_cfg, board_cfg, pe_name, 13, state_l)
                m0_l = rollout_metrics(model_l, board_cfg, test_problems, "local", p_noise=0.0,          seed=args.seed + 12345)
                m1_l = rollout_metrics(model_l, board_cfg, test_problems, "local", p_noise=args.noise_p, seed=args.seed + 54321)

                exact0[pe_name]["local"] = m0_l["exact_acc"]
                digit0[pe_name]["local"] = m0_l["digit_acc"]

                exact1[pe_name]["local"] = m1_l["exact_acc"]
                digit1[pe_name]["local"] = m1_l["digit_acc"]
                reco1[pe_name]["local"]  = m1_l["recovery_rate"]

                # ---- GLOBAL (vocab=13) ----
                pe_g = dict(make_pes(model_cfg, board_cfg))[pe_name]
                state_g, info_g = train_one(
                    device=device,
                    board_cfg=board_cfg,
                    model_cfg=model_cfg,
                    pe_name=pe_name,
                    pe_module=pe_g,
                    setting="global",
                    vocab_size=13,
                    train_problems=train_problems,
                    val_problems=val_problems,
                    batch_size=args.batch_size,
                    epochs=args.epochs,
                    lr=args.lr,
                    denoise_rate=args.denoise_rate,
                    p_revert=args.p_revert,
                    seed=stable_seed(f"{cfg_name}|{n_train}|{pe_name}|global"),
                )

                subdir_g = os.path.join(run_dir, pe_name, "global")
                os.makedirs(subdir_g, exist_ok=True)

                torch.save(
                    {
                        "model_state_dict": state_g,
                        "model_cfg": asdict(model_cfg),
                        "board_cfg": {"H": board_cfg.H, "W": board_cfg.W, "n_digits": board_cfg.n_digits},
                        "pe": pe_name,
                        "setting": "global",
                        "vocab_size": 13,
                        "best_val_masked_acc": info_g["best_val_masked_acc"],
                        "best_epoch": info_g["best_epoch"],
                    },
                    os.path.join(subdir_g, "best_model.pt"),
                )
                write_history_csv(os.path.join(subdir_g, "history.csv"), info_g["train_loss"], info_g["val_masked_acc"])
                plot_curve(
                    os.path.join(subdir_g, "train_loss.png"),
                    info_g["train_loss"],
                    title=f"{cfg_name} | train={n_train} | {pe_name} | global",
                    ylabel="train loss",
                )
                plot_curve(
                    os.path.join(subdir_g, "val_masked_acc.png"),
                    info_g["val_masked_acc"],
                    title=f"{cfg_name} | train={n_train} | {pe_name} | global",
                    ylabel="val masked acc",
                )

                model_g = build_model_with_state(device, model_cfg, board_cfg, pe_name, 13, state_g)
                m0_g = rollout_metrics(model_g, board_cfg, test_problems, "global", p_noise=0.0,          seed=args.seed + 12345)
                m1_g = rollout_metrics(model_g, board_cfg, test_problems, "global", p_noise=args.noise_p, seed=args.seed + 54321)

                exact0[pe_name]["global"] = m0_g["exact_acc"]
                digit0[pe_name]["global"] = m0_g["digit_acc"]

                exact1[pe_name]["global"] = m1_g["exact_acc"]
                digit1[pe_name]["global"] = m1_g["digit_acc"]
                reco1[pe_name]["global"]  = m1_g["recovery_rate"]

                # one row per PE containing the 3 settings (makes plotting / reading easy)
                rows.append(
                    {
                        "model_cfg": cfg_name,
                        "n_train": n_train,
                        "pe": pe_name,

                        "classic_exact_noise0": m0_c["exact_acc"],
                        f"classic_exact_noise{args.noise_p}": m1_c["exact_acc"],
                        "classic_digit_noise0": m0_c["digit_acc"],
                        f"classic_digit_noise{args.noise_p}": m1_c["digit_acc"],
                        f"classic_recovery_noise{args.noise_p}": m1_c["recovery_rate"],

                        "local_exact_noise0": m0_l["exact_acc"],
                        f"local_exact_noise{args.noise_p}": m1_l["exact_acc"],
                        "local_digit_noise0": m0_l["digit_acc"],
                        f"local_digit_noise{args.noise_p}": m1_l["digit_acc"],
                        f"local_recovery_noise{args.noise_p}": m1_l["recovery_rate"],

                        "global_exact_noise0": m0_g["exact_acc"],
                        f"global_exact_noise{args.noise_p}": m1_g["exact_acc"],
                        "global_digit_noise0": m0_g["digit_acc"],
                        f"global_digit_noise{args.noise_p}": m1_g["digit_acc"],
                        f"global_recovery_noise{args.noise_p}": m1_g["recovery_rate"],
                    }
                )

                # incremental save so you can kill job and keep partial progress
                write_csv(os.path.join(run_dir, "results.csv"), rows)

            # bar plots for THIS (model_cfg, n_train) folder
            def extract(metric_dict, key):
                return [metric_dict[pe][key] for pe in pe_names]

            # --- EXACT (noise=0.0)
            plot_grouped_bars(
                out_path=os.path.join(run_dir, "barplot_exact_noise0.png"),
                pe_names=pe_names,
                classic_vals=extract(exact0, "classic"),
                local_vals=extract(exact0, "local"),
                global_vals=extract(exact0, "global"),
                title=f"{cfg_name} | n_train={n_train} | exact final-board acc (noise=0.0) | n_test={args.n_test}",
                ylabel="Rollout exact accuracy",
            )

            # --- DIGIT ACC (noise=0.0)
            plot_grouped_bars(
                out_path=os.path.join(run_dir, "barplot_digit_noise0.png"),
                pe_names=pe_names,
                classic_vals=extract(digit0, "classic"),
                local_vals=extract(digit0, "local"),
                global_vals=extract(digit0, "global"),
                title=f"{cfg_name} | n_train={n_train} | result digit accuracy (noise=0.0) | n_test={args.n_test}",
                ylabel="Rollout digit accuracy",
            )

            # --- EXACT (noise=args.noise_p)
            plot_grouped_bars(
                out_path=os.path.join(run_dir, f"barplot_exact_noise{args.noise_p}.png"),
                pe_names=pe_names,
                classic_vals=extract(exact1, "classic"),
                local_vals=extract(exact1, "local"),
                global_vals=extract(exact1, "global"),
                title=f"{cfg_name} | n_train={n_train} | exact final-board acc (noise={args.noise_p}) | n_test={args.n_test}",
                ylabel="Rollout exact accuracy",
            )

            # --- DIGIT ACC (noise=args.noise_p)
            plot_grouped_bars(
                out_path=os.path.join(run_dir, f"barplot_digit_noise{args.noise_p}.png"),
                pe_names=pe_names,
                classic_vals=extract(digit1, "classic"),
                local_vals=extract(digit1, "local"),
                global_vals=extract(digit1, "global"),
                title=f"{cfg_name} | n_train={n_train} | result digit accuracy (noise={args.noise_p}) | n_test={args.n_test}",
                ylabel="Rollout digit accuracy",)

            # --- RECOVERY RATE (only meaningful with noise)
            plot_grouped_bars(
                out_path=os.path.join(run_dir, f"barplot_recovery_noise{args.noise_p}.png"),
                pe_names=pe_names,
                classic_vals=extract(reco1, "classic"),
                local_vals=extract(reco1, "local"),
                global_vals=extract(reco1, "global"),
                title=f"{cfg_name} | n_train={n_train} | recovery rate (noise={args.noise_p}) | n_test={args.n_test}",
                ylabel="Rollout recovery rate",
                )


            dt = time.time() - t0
            with open(os.path.join(run_dir, "done.json"), "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "model_cfg": cfg_name,
                        "n_train": n_train,
                        "n_digits": args.n_digits,
                        "n_test": args.n_test,
                        "noise_p": args.noise_p,
                        "seconds": dt,
                    },
                    f,
                    indent=2,
                )

            print(f"Saved plots + models + histories to: {run_dir}")



if __name__ == "__main__":
    main()
