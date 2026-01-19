from src.training.configs import ModelConfig, TrainConfig
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm.auto import tqdm  

from src.training.trainers import BlackboardTrainer
from src.data.addition_algo import BoardConfig
from src.data.problems import generate_problems, generate_diversified_problems
from src.models.positional_encodings import *
from src.data.board_dataset import BlackboardAdditionStepDataset
from src.models.transformers import BlackboardTransformer
import os
from dataclasses import asdict


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

def model_cfg_name(cfg: ModelConfig) -> str:
    return (
        f"d{cfg.d_model}"
        f"_h{cfg.nhead}"
        f"_L{cfg.num_layers}"
        f"_ff{cfg.dim_feedforward}"
    )


def make_pes(model_cfg, board_cfg):
    return [ 
        (
            "abs_1d_learned",
            LearnedPositionalEncoding1D(
                model_cfg.d_model, board_cfg.H * board_cfg.W
            )
        ),

        (
            "abs_1d_sinusoidal",
            SinusoidalPositionalEncoding(
                model_cfg.d_model,
                model_cfg.max_len,
            )
        ),

        (
            "abs_2d_learned",
            LearnedPositionalEncoding2D(
                model_cfg.d_model,
                board_cfg.H,
                board_cfg.W
            )
        ),

        (
            "abs_2d_sin+rel_2d_bias",  Abs2DPlusRelBias2D(
            abs_pe=SinusoidalPositionalEncoding2D(model_cfg.d_model, board_cfg.H, board_cfg.W),
            rel_bias=RelativePositionBias2D(model_cfg.nhead, board_cfg.H, board_cfg.W),
            )
        ),

        (
            "abs_2d_sinusoidal",
            SinusoidalPositionalEncoding2D(
                model_cfg.d_model,
                board_cfg.H,
                board_cfg.W,
            )
        ),

        (
            "rel_2d_bias",
            RelativePositionBias2D(
                model_cfg.nhead,
                board_cfg.H,
                board_cfg.W,
            )
        ),
    
    ]


if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    board_cfg = BoardConfig(H=4, W=5, n_digits=3)
    vocab_size = 12
    max_len = board_cfg.H * board_cfg.W

    n_train = 40_000
    n_val = 10_000

    model_cfgs = [
        ModelConfig(
            d_model=64,
            nhead=1,
            num_layers=2,
            dim_feedforward=256,
            dropout=0.1,
            max_len=200,
        ),

        ModelConfig(
            d_model=64,
            nhead=2,
            num_layers=3,
            dim_feedforward=256,
            dropout=0.1,
            max_len=200,
        ),

        ModelConfig(
            d_model=128,
            nhead=2,
            num_layers=3,
            dim_feedforward=512,
            dropout=0.1,
            max_len=200,
        ),

        ModelConfig(
            d_model=128,
            nhead=4,
            num_layers=4,
            dim_feedforward=512,
            dropout=0.1,
            max_len=200,
        ),
        ModelConfig(
            d_model=256,
            nhead=4,
            num_layers=4,
            dim_feedforward=512,
            dropout=0.1,
            max_len=200,
        )
    ]

    train_cfg = TrainConfig(
        batch_size = 512,
        num_epochs = 10,
        lr = 3e-4,
        log_interval = 0.1, 
        enable_docs=True,
        save_model = True,
        seed = 42,
        out_dir="models",
        exp_name="",
    )

  
    train_problems = generate_diversified_problems(board_cfg, n_train, seed=0)
    val_problems   = generate_diversified_problems(board_cfg, n_val,   seed=1)

    train_ds = BlackboardAdditionStepDataset(train_problems)
    val_ds   = BlackboardAdditionStepDataset(val_problems)

    train_loader = DataLoader(
        train_ds,
        batch_size=train_cfg.batch_size,
        shuffle=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=train_cfg.batch_size,
        shuffle=False,
    )

    trainer = BlackboardTrainer(
        model=None,
        optimizer=None,
        device=device,
        train_loader=train_loader,
        val_loader=val_loader,
        train_cfg=None,
        board_cfg=board_cfg,
        accuracy_with_splits=accuracy_with_splits,
        masked_cross_entropy_fn=masked_cross_entropy
    )


    for model_cfg in model_cfgs:
        pes = make_pes(model_cfg, board_cfg)
        for pe_name, pe in pes:

            cfg_name = model_cfg_name(model_cfg)
            train_cfg.exp_name = os.path.join(pe_name, cfg_name)

            print(f"\n===== {pe_name} | {cfg_name} =====")

            model = BlackboardTransformer(
                vocab_size=12,
                pos_enc=pe,
                **asdict(model_cfg)
            ).to(device)

            optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg.lr)

            trainer.model = model
            trainer.optimizer = optimizer
            trainer.train_cfg = train_cfg

            trainer.fit()