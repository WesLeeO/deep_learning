import os, sys
import json
import argparse
from dataclasses import asdict
from typing import Dict, List, Tuple

import numpy as np
import torch
import matplotlib.pyplot as plt

# -------------------------
# Ensure imports work when script is at repo root
# -------------------------
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data.addition_algo import BoardConfig, generate_trajectory_variant_A, VOID_TOKEN
from src.data.problems import generate_diversified_problems
from src.models.transformers import BlackboardTransformer
from src.models.positional_encodings import (
    LearnedPositionalEncoding1D,
    SinusoidalPositionalEncoding,       # 1D sinus
    LearnedPositionalEncoding2D,
    SinusoidalPositionalEncoding2D,     # 2D sinus
)
from src.training.configs import ModelConfig


# -------------------------
# Positional encodings (match your PE names)
# -------------------------
def make_pe(pe_name: str, model_cfg: ModelConfig, board_cfg: BoardConfig) -> torch.nn.Module:
    if pe_name == "abs_1d_learned":
        return LearnedPositionalEncoding1D(model_cfg.d_model, board_cfg.H * board_cfg.W)
    if pe_name == "abs_1d_sinusoidal":
        return SinusoidalPositionalEncoding(model_cfg.d_model, model_cfg.max_len)
    if pe_name == "abs_2d_learned":
        return LearnedPositionalEncoding2D(model_cfg.d_model, board_cfg.H, board_cfg.W)
    if pe_name == "abs_2d_sinusoidal":
        return SinusoidalPositionalEncoding2D(model_cfg.d_model, board_cfg.H, board_cfg.W)
    raise ValueError(f"Unknown pe_name: {pe_name}")


# -------------------------
# Masks + helpers
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


def result_positions(cfg: BoardConfig) -> torch.Tensor:
    # result digits are last n_digits columns of result row
    cols = list(range(cfg.W - cfg.n_digits, cfg.W))
    idxs = [cfg.result_row * cfg.W + c for c in cols]
    return torch.tensor(idxs, dtype=torch.long)


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
    """
    if p_noise <= 0.0 or step_idx <= 0:
        return

    u = torch.rand((), generator=rng, device=board.device).item()
    if u > p_noise:
        return

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
        return

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


@torch.no_grad()
def rollout_one(
    model: torch.nn.Module,
    cfg: BoardConfig,
    xs: np.ndarray,
    setting: str,         # "classic" | "local" | "global"
    p_noise: float,
    seed: int,
    max_iters: int,
) -> Tuple[torch.Tensor, bool]:
    """
    Returns (final_board_flat_cpu, finished_flag)
    """
    device = next(model.parameters()).device
    rng_torch = torch.Generator(device=device)
    rng_torch.manual_seed(seed)

    S_seq, _ = generate_trajectory_variant_A(cfg, xs)
    board = torch.from_numpy(S_seq[0]).view(-1).long().to(device)

    t = 0
    iters = 0
    W = cfg.W
    col_end = W - 1

    while t < cfg.n_digits and iters < max_iters:
        iters += 1

        inject_rollout_noise_inplace(board, cfg, t, rng_torch, p_noise=p_noise, n_noise=1)

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
                # jump back based on erased cols
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

    finished = (t >= cfg.n_digits)
    return board.detach().cpu(), finished


@torch.no_grad()
def eval_metrics_finished_vs_all(
    model: torch.nn.Module,
    cfg: BoardConfig,
    problems,
    setting: str,
    p_noise: float,
    seed: int,
    max_iters: int,
) -> Dict[str, float]:
    """
    Computes:
      - exact_acc_all
      - exact_acc_finished
      - digit_acc_all
      - digit_acc_finished
      - finish_rate
    """
    rp = result_positions(cfg)

    n = len(problems)
    n_finished = 0

    # "all" totals
    ok_exact_all = 0
    digit_correct_all = 0
    digit_total_all = 0

    # "finished-only" totals
    ok_exact_fin = 0
    digit_correct_fin = 0
    digit_total_fin = 0

    for i, pr in enumerate(problems):
        xs = pr.operands
        S_seq, _ = generate_trajectory_variant_A(cfg, xs)
        target_final = torch.from_numpy(S_seq[-1]).view(-1).long()

        pred_final, finished = rollout_one(
            model=model,
            cfg=cfg,
            xs=xs,
            setting=setting,
            p_noise=p_noise,
            seed=seed + i,
            max_iters=max_iters,
        )

        # all
        if torch.equal(pred_final, target_final):
            ok_exact_all += 1
        digit_correct_all += (pred_final[rp] == target_final[rp]).sum().item()
        digit_total_all += rp.numel()

        # finished-only
        if finished:
            n_finished += 1
            if torch.equal(pred_final, target_final):
                ok_exact_fin += 1
            digit_correct_fin += (pred_final[rp] == target_final[rp]).sum().item()
            digit_total_fin += rp.numel()

    exact_acc_all = ok_exact_all / max(n, 1)
    digit_acc_all = digit_correct_all / max(digit_total_all, 1)
    finish_rate = n_finished / max(n, 1)

    if n_finished > 0:
        exact_acc_finished = ok_exact_fin / n_finished
        digit_acc_finished = digit_correct_fin / max(digit_total_fin, 1)
    else:
        exact_acc_finished = float("nan")
        digit_acc_finished = float("nan")

    return {
        "exact_acc_all": exact_acc_all,
        "exact_acc_finished": exact_acc_finished,
        "digit_acc_all": digit_acc_all,
        "digit_acc_finished": digit_acc_finished,
        "finish_rate": finish_rate,
        "n_finished": float(n_finished),
        "n_total": float(n),
    }


# -------------------------
# Loading trained models from your folder layout
# -------------------------
def load_checkpoint(pt_path: str, map_location="cpu") -> Dict:
    return torch.load(pt_path, map_location=map_location)


def build_model_from_ckpt(ckpt: Dict, device: torch.device) -> torch.nn.Module:
    model_cfg = ModelConfig(**ckpt["model_cfg"])
    bc = ckpt["board_cfg"]
    board_cfg = BoardConfig(H=bc["H"], W=bc["W"], n_digits=bc["n_digits"])
    pe_name = ckpt["pe"]
    vocab_size = int(ckpt.get("vocab_size", 12))

    pe = make_pe(pe_name, model_cfg, board_cfg)

    model = BlackboardTransformer(
        vocab_size=vocab_size,
        pos_enc=pe,
        **asdict(model_cfg),
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, model_cfg, board_cfg, pe_name, vocab_size


# -------------------------
# Plotting
# -------------------------
def plot_grouped_bars(out_path: str, pe_names: List[str],
                      classic_vals: List[float], local_vals: List[float], global_vals: List[float],
                      ylabel: str,
                      title: str):
    x = np.arange(len(pe_names))
    width = 0.25

    plt.figure(figsize=(max(10, len(pe_names) * 1.5), 5))
    plt.bar(x - width, classic_vals, width, label="classic")
    plt.bar(x,         local_vals,   width, label="local")
    plt.bar(x + width, global_vals,  width, label="global")

    plt.xticks(x, pe_names, rotation=20, ha="right")
    plt.ylim(0, 1.0)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def find_model_cfg_dirs(root: str) -> List[str]:
    # expects folders like root/d64_h1_L2_ff256/...
    subdirs = []
    for name in sorted(os.listdir(root)):
        p = os.path.join(root, name)
        if os.path.isdir(p) and name.startswith("d") and "_h" in name and "_L" in name and "_ff" in name:
            subdirs.append(p)
    return subdirs


def find_train_dirs(model_cfg_dir: str) -> List[str]:
    # expects model_cfg_dir/train_50000/...
    subdirs = []
    for name in sorted(os.listdir(model_cfg_dir)):
        p = os.path.join(model_cfg_dir, name)
        if os.path.isdir(p) and name.startswith("train_"):
            subdirs.append(p)
    return subdirs


def list_pe_names(train_dir: str) -> List[str]:
    # each PE is a folder inside train_dir
    pes = []
    for name in sorted(os.listdir(train_dir)):
        p = os.path.join(train_dir, name)
        if os.path.isdir(p):
            # must contain classic/local/global subdirs (at least classic)
            if os.path.isdir(os.path.join(p, "classic")):
                pes.append(name)
    return pes


def ckpt_path(train_dir: str, pe: str, setting: str) -> str:
    return os.path.join(train_dir, pe, setting, "best_model.pt")


# -------------------------
# Main
# -------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=str, required=True,
                   help="Path to run folder, e.g. .../run_50k")
    p.add_argument("--n-test", type=int, default=5000)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--noise-p", type=float, default=0.1)
    p.add_argument("--max-iters", type=int, default=800)
    p.add_argument("--out-dirname", type=str, default="reval_finished_vs_all",
                   help="Output folder name created inside each train_* directory")
    args = p.parse_args()

    root = os.path.abspath(args.root)
    if not os.path.isdir(root):
        raise FileNotFoundError(f"--root not found: {root}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # We'll generate a fixed test set PER (board_cfg) from the checkpoints
    # So we load one ckpt per train_dir to infer board_cfg then create problems.

    model_cfg_dirs = find_model_cfg_dirs(root)
    if not model_cfg_dirs:
        raise RuntimeError(f"No model_cfg dirs found in {root}. Expected dirs like d64_h1_L2_ff256/")

    for mdir in model_cfg_dirs:
        cfg_name = os.path.basename(mdir)
        train_dirs = find_train_dirs(mdir)
        if not train_dirs:
            print(f"[skip] No train_* dirs in {mdir}")
            continue

        for tdir in train_dirs:
            train_tag = os.path.basename(tdir)  # e.g. train_50000
            print(f"\n=== Re-evaluating: {cfg_name} / {train_tag} ===")

            # create output folder
            out_dir = os.path.join(tdir, args.out_dirname)
            os.makedirs(out_dir, exist_ok=True)

            pe_names = list_pe_names(tdir)
            if not pe_names:
                print(f"[skip] No PE folders found in {tdir}")
                continue

            # Load one checkpoint to infer board_cfg for test set creation
            # Prefer classic if exists
            sample_ckpt_path = ckpt_path(tdir, pe_names[0], "classic")
            ckpt0 = load_checkpoint(sample_ckpt_path, map_location="cpu")
            bc0 = ckpt0["board_cfg"]
            board_cfg = BoardConfig(H=bc0["H"], W=bc0["W"], n_digits=bc0["n_digits"])

            test_problems = generate_diversified_problems(board_cfg, args.n_test, seed=999)

            # collect metrics dicts per PE per setting, for two noise levels
            def init_metrics():
                return {pe: {"classic": {}, "local": {}, "global": {}} for pe in pe_names}

            met0 = init_metrics()  # noise=0
            met1 = init_metrics()  # noise=noise_p

            for pe in pe_names:
                for setting in ["classic", "local", "global"]:
                    pt = ckpt_path(tdir, pe, setting)
                    if not os.path.isfile(pt):
                        print(f"[warn] missing: {pt} (skip)")
                        continue

                    ckpt = load_checkpoint(pt, map_location="cpu")
                    model, model_cfg, bc, pe_name, vocab_size = build_model_from_ckpt(ckpt, device)

                    # sanity: ensure consistent board_cfg
                    if (bc.H != board_cfg.H) or (bc.W != board_cfg.W) or (bc.n_digits != board_cfg.n_digits):
                        raise RuntimeError(f"Board cfg mismatch in {pt}: got {bc} vs {board_cfg}")

                    m0 = eval_metrics_finished_vs_all(
                        model, board_cfg, test_problems, setting,
                        p_noise=0.0, seed=args.seed + 1000, max_iters=args.max_iters
                    )
                    m1 = eval_metrics_finished_vs_all(
                        model, board_cfg, test_problems, setting,
                        p_noise=args.noise_p, seed=args.seed + 2000, max_iters=args.max_iters
                    )

                    met0[pe][setting] = m0
                    met1[pe][setting] = m1

                    print(f"  {pe:18s} {setting:7s} | "
                          f"finish={m0['finish_rate']:.3f} exact_all={m0['exact_acc_all']:.3f} "
                          f"| noise exact_all={m1['exact_acc_all']:.3f} finish={m1['finish_rate']:.3f}")

            # helper to extract values (default nan if missing)
            def get(metric_store, pe, setting, key):
                if (pe in metric_store) and (setting in metric_store[pe]) and (key in metric_store[pe][setting]):
                    return float(metric_store[pe][setting][key])
                return float("nan")

            def extract(metric_store, key, setting):
                return [get(metric_store, pe, setting, key) for pe in pe_names]

            # ---- PLOTS (noise=0)
            plot_grouped_bars(
                out_path=os.path.join(out_dir, "barplot_exact_all_noise0.png"),
                pe_names=pe_names,
                classic_vals=extract(met0, "exact_acc_all", "classic"),
                local_vals=extract(met0, "exact_acc_all", "local"),
                global_vals=extract(met0, "exact_acc_all", "global"),
                ylabel="Exact final-board acc (ALL)",
                title=f"{cfg_name} | {train_tag} | exact acc (ALL) | noise=0.0 | n_test={args.n_test} | max_iters={args.max_iters}",
            )
            plot_grouped_bars(
                out_path=os.path.join(out_dir, "barplot_exact_finished_noise0.png"),
                pe_names=pe_names,
                classic_vals=extract(met0, "exact_acc_finished", "classic"),
                local_vals=extract(met0, "exact_acc_finished", "local"),
                global_vals=extract(met0, "exact_acc_finished", "global"),
                ylabel="Exact final-board acc (FINISHED)",
                title=f"{cfg_name} | {train_tag} | exact acc (FINISHED) | noise=0.0 | n_test={args.n_test} | max_iters={args.max_iters}",
            )
            plot_grouped_bars(
                out_path=os.path.join(out_dir, "barplot_digit_all_noise0.png"),
                pe_names=pe_names,
                classic_vals=extract(met0, "digit_acc_all", "classic"),
                local_vals=extract(met0, "digit_acc_all", "local"),
                global_vals=extract(met0, "digit_acc_all", "global"),
                ylabel="Digit acc (ALL)",
                title=f"{cfg_name} | {train_tag} | digit acc (ALL) | noise=0.0 | n_test={args.n_test} | max_iters={args.max_iters}",
            )
            plot_grouped_bars(
                out_path=os.path.join(out_dir, "barplot_digit_finished_noise0.png"),
                pe_names=pe_names,
                classic_vals=extract(met0, "digit_acc_finished", "classic"),
                local_vals=extract(met0, "digit_acc_finished", "local"),
                global_vals=extract(met0, "digit_acc_finished", "global"),
                ylabel="Digit acc (FINISHED)",
                title=f"{cfg_name} | {train_tag} | digit acc (FINISHED) | noise=0.0 | n_test={args.n_test} | max_iters={args.max_iters}",
            )

            # ---- PLOTS (noise=noise_p)
            np_str = str(args.noise_p)
            plot_grouped_bars(
                out_path=os.path.join(out_dir, f"barplot_exact_all_noise{np_str}.png"),
                pe_names=pe_names,
                classic_vals=extract(met1, "exact_acc_all", "classic"),
                local_vals=extract(met1, "exact_acc_all", "local"),
                global_vals=extract(met1, "exact_acc_all", "global"),
                ylabel="Exact final-board acc (ALL)",
                title=f"{cfg_name} | {train_tag} | exact acc (ALL) | noise={args.noise_p} | n_test={args.n_test} | max_iters={args.max_iters}",
            )
            plot_grouped_bars(
                out_path=os.path.join(out_dir, f"barplot_exact_finished_noise{np_str}.png"),
                pe_names=pe_names,
                classic_vals=extract(met1, "exact_acc_finished", "classic"),
                local_vals=extract(met1, "exact_acc_finished", "local"),
                global_vals=extract(met1, "exact_acc_finished", "global"),
                ylabel="Exact final-board acc (FINISHED)",
                title=f"{cfg_name} | {train_tag} | exact acc (FINISHED) | noise={args.noise_p} | n_test={args.n_test} | max_iters={args.max_iters}",
            )
            plot_grouped_bars(
                out_path=os.path.join(out_dir, f"barplot_digit_all_noise{np_str}.png"),
                pe_names=pe_names,
                classic_vals=extract(met1, "digit_acc_all", "classic"),
                local_vals=extract(met1, "digit_acc_all", "local"),
                global_vals=extract(met1, "digit_acc_all", "global"),
                ylabel="Digit acc (ALL)",
                title=f"{cfg_name} | {train_tag} | digit acc (ALL) | noise={args.noise_p} | n_test={args.n_test} | max_iters={args.max_iters}",
            )
            plot_grouped_bars(
                out_path=os.path.join(out_dir, f"barplot_digit_finished_noise{np_str}.png"),
                pe_names=pe_names,
                classic_vals=extract(met1, "digit_acc_finished", "classic"),
                local_vals=extract(met1, "digit_acc_finished", "local"),
                global_vals=extract(met1, "digit_acc_finished", "global"),
                ylabel="Digit acc (FINISHED)",
                title=f"{cfg_name} | {train_tag} | digit acc (FINISHED) | noise={args.noise_p} | n_test={args.n_test} | max_iters={args.max_iters}",
            )

            # save raw metrics
            with open(os.path.join(out_dir, "metrics_noise0.json"), "w", encoding="utf-8") as f:
                json.dump(met0, f, indent=2)
            with open(os.path.join(out_dir, f"metrics_noise{np_str}.json"), "w", encoding="utf-8") as f:
                json.dump(met1, f, indent=2)

            print(f"Saved plots + metrics to: {out_dir}")


if __name__ == "__main__":
    main()
