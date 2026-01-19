from dataclasses import dataclass
from typing import List, Tuple
import numpy as np

# Token ids
DIGIT_TOKENS = {str(i): i for i in range(10)}
PLUS_TOKEN   = 10
BLANK_TOKEN  = 11
VOID_TOKEN   = 12

VOCAB_SIZE   = 13  # digits (0-9) + PLUS + BLANK + VOID

@dataclass
class BoardConfig:
    H: int             # grid height
    W: int             # grid width
    n_digits: int      # number of digits per addend (e.g. 3 for Phase 0)
    base: int = 10
    n_addends: int = 2

    # row indices for convenience (can be changed later if needed)
    carry_row: int = 0
    top_row: int = 1
    bottom_row: int = 2
    result_row: int = 3


def sample_operands(cfg: BoardConfig, rng: np.random.Generator) -> np.ndarray:
    """
    Sample n_addends integers with cfg.n_digits digits (allowing leading zeros).
    """
    low  = 0
    high = cfg.base ** cfg.n_digits
    xs = rng.integers(low, high, size=cfg.n_addends, endpoint=False)
    return xs


def number_to_digits(x: int, n_digits: int, base: int = 10) -> List[int]:
    """
    Least-significant digit first, padded with zeros to length n_digits.
    """
    digits = []
    for _ in range(n_digits):
        digits.append(x % base)
        x //= base
    return digits  # [d0, d1, ..., d_{n_digits-1}]


def build_initial_board(cfg: BoardConfig, xs: np.ndarray) -> np.ndarray:
    """
    Build S0 for 2-addend addition.

    Layout:
      - row top_row:    first addend
      - row bottom_row: second addend
      - row carry_row:  all blanks initially
      - row result_row: all blanks initially

    Digits are right-aligned with the units digit at the rightmost column.
    One extra column to the left is reserved for a possible overflow digit,
    and, if there is space, an additional column to the left for the '+' sign.
    """
    board = np.full((cfg.H, cfg.W), BLANK_TOKEN, dtype=np.int64)

    assert cfg.n_addends == 2, "build_initial_board currently assumes 2 addends."
    assert cfg.W >= cfg.n_digits + 1, (
        "Board width W should be at least n_digits + 1 to allow an overflow column."
    )

    col_end = cfg.W - 1                         # rightmost column (units)
    col_start = col_end - (cfg.n_digits - 1)    # leftmost digit column

    # Place the two operands.
    top, bottom = xs[0], xs[1]
    top_digits    = number_to_digits(top, cfg.n_digits, cfg.base)
    bottom_digits = number_to_digits(bottom, cfg.n_digits, cfg.base)

    # Digits are placed right-to-left: units at col_end.
    for j, d in enumerate(top_digits):
        col = col_end - j
        board[cfg.top_row, col] = d

    for j, d in enumerate(bottom_digits):
        col = col_end - j
        board[cfg.bottom_row, col] = d

    # Reserve an overflow column immediately to the left of the most significant digit.
    overflow_col = col_start - 1

    # Optional: put a '+' sign one more column to the left, if there is space.
    plus_col = overflow_col - 1
    if plus_col >= 0:
        board[cfg.bottom_row, plus_col] = PLUS_TOKEN

    return board


def generate_trajectory_variant_A(
    cfg: BoardConfig,
    xs: np.ndarray
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Generate a trajectory (S_seq, M_seq) for masked-write training (variant A).

    S_seq: list of boards S_t, length T+1 (including S_0 and S_T)
    M_seq: list of boolean masks M_t, length T (one per transition S_t -> S_{t+1})

    One step per digit column, from least significant (rightmost) to most significant
    (moving left). At each step:
      - write the result digit in cfg.result_row at the current column
      - write the carry symbol in cfg.carry_row one column to the left:
          * BLANK_TOKEN if there is no carry
          * a digit token if there is a carry.
    """
    S_seq: List[np.ndarray] = []
    M_seq: List[np.ndarray] = []

    S_t = build_initial_board(cfg, xs)
    S_seq.append(S_t.copy())

    carry = 0
    col_end = cfg.W - 1
    # leftmost digit column (not strictly needed below but kept for clarity)
    col_start = col_end - (cfg.n_digits - 1)

    # One step per digit column, starting from the units on the right.
    for step in range(cfg.n_digits):
        col = col_end - step  # current digit column, moving left

        # Read current column digits.
        column_sum = carry
        for row in [cfg.top_row, cfg.bottom_row]:
            d = S_t[row, col]
            if d != BLANK_TOKEN:
                column_sum += int(d)

        new_digit = column_sum % cfg.base
        carry     = column_sum // cfg.base

        # Prepare next board and mask.
        S_next = S_t.copy()
        M_t = np.zeros_like(S_t, dtype=bool)

        # Write result digit in result row, current column.
        S_next[cfg.result_row, col] = new_digit
        M_t[cfg.result_row, col] = True

        # Always supervise the carry cell one column to the left:
        # - BLANK_TOKEN if there is no carry
        # - a digit token if there is a carry.
        next_col = col - 1
        if next_col >= 0:
            if carry == 0:
                S_next[cfg.carry_row, next_col] = 0
            else:
                S_next[cfg.carry_row, next_col] = carry
            M_t[cfg.carry_row, next_col] = True

        S_seq.append(S_next)
        M_seq.append(M_t)

        S_t = S_next

    return S_seq, M_seq


def generate_trajectory_variant_B(
    cfg: BoardConfig,
    xs: np.ndarray
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
        Generate a trajectory (S_seq, M_seq) for no-masking training (variant B).
    """
    S_seq: List[np.ndarray] = []
    M_seq: List[np.ndarray] = []

    S_t = build_initial_board(cfg, xs)
    S_seq.append(S_t.copy())

    carry = 0
    col_end = cfg.W - 1
    # leftmost digit column (not strictly needed below but kept for clarity)
    col_start = col_end - (cfg.n_digits - 1)

    # One step per digit column, starting from the units on the right.
    for step in range(cfg.n_digits):
        col = col_end - step  # current digit column, moving left

        # Read current column digits.
        column_sum = carry
        for row in [cfg.top_row, cfg.bottom_row]:
            d = S_t[row, col]
            if d != BLANK_TOKEN:
                column_sum += int(d)

        new_digit = column_sum % cfg.base
        carry     = column_sum // cfg.base

        # Prepare next board and mask.
        S_next = S_t.copy()
        M_t = np.ones_like(S_t, dtype=bool)

        # Write result digit in result row, current column.
        S_next[cfg.result_row, col] = new_digit

        # Always supervise the carry cell one column to the left:
        # - BLANK_TOKEN if there is no carry
        # - a digit token if there is a carry.
        next_col = col - 1
        if next_col >= 0:
            if carry == 0:
                S_next[cfg.carry_row, next_col] = 0 # previously BLANK_TOKEN
            else:
                S_next[cfg.carry_row, next_col] = carry

        S_seq.append(S_next)
        M_seq.append(M_t)

        S_t = S_next

    return S_seq, M_seq

