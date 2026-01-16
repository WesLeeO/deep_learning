from typing import List, Tuple, Iterable, Dict
import warnings
import numpy as np

from src.data.addition_algo import BoardConfig, number_to_digits
from src.data.problems import AdditionProblem, digits_to_number


Triplet = Tuple[int, int, int]  # (carry_in, top_digit, bottom_digit)


# ---------------------------------------------------------------------------
# Helper: extract per-column (carry_in, a, b) triplets for ONE problem
# ---------------------------------------------------------------------------

def extract_triplets(cfg: BoardConfig, xs: np.ndarray) -> List[Triplet]:
    """
    For a 2-addend problem, compute per-column (carry_in, a, b) triplets.
    Least-significant column is index 0 (consistent with number_to_digits).
    """
    assert cfg.n_addends == 2, "extract_triplets assumes 2 addends."
    base = cfg.base
    top_val, bot_val = int(xs[0]), int(xs[1])

    top_digits = number_to_digits(top_val, cfg.n_digits, base=base)
    bot_digits = number_to_digits(bot_val, cfg.n_digits, base=base)

    triplets: List[Triplet] = []
    carry = 0
    for col in range(cfg.n_digits):
        a = top_digits[col]
        b = bot_digits[col]
        cin = carry
        s = a + b + cin
        carry = s // base
        triplets.append((cin, a, b))
    return triplets


def _check_budget(target: int, got: int, what: str):
    if got < target:
        warnings.warn(
            f"[sample_efficiency] Only generated {got} {what} "
            f"(requested {target}). Perhaps constraints are too strict."
        )


# ---------------------------------------------------------------------------
# Exhaustive enumeration over ALL top,bottom pairs (small n_digits only)
# ---------------------------------------------------------------------------

def enumerate_all_problems(cfg: BoardConfig) -> List[AdditionProblem]:
    """
    Enumerate ALL possible problems (top, bottom) for given n_digits and base.

    Number of pairs = (base^n_digits)^2, so only safe for small n_digits
    (e.g. 3).
    """
    base = cfg.base
    n_digits = cfg.n_digits
    assert n_digits <= 3, (
        "enumerate_all_problems is only intended for small n_digits (<=3)."
    )

    max_val = base ** n_digits
    problems: List[AdditionProblem] = []
    for top in range(max_val):
        for bottom in range(max_val):
            xs = np.array([top, bottom], dtype=np.int64)
            problems.append(AdditionProblem(xs, cfg))
    return problems


# ======================================================================
# Setting 1: simple random subset from all possible 3-digit problems
# ======================================================================

def generate_setting1_random_fraction(
    cfg: BoardConfig,
    n_train: int,
    n_test: int,
    seed: int,
) -> Tuple[List[AdditionProblem], List[AdditionProblem]]:
    """
    Setting 1 (baseline sample efficiency):

      - Enumerate all possible problems (top, bottom) for this cfg.
      - Shuffle them.
      - Take first n_train for train, next n_test for test.

    Intended for cfg.n_digits <= 3.
    """
    assert cfg.n_digits <= 3, "Setting 1 is only meant for small n_digits (<=3)."

    all_problems = enumerate_all_problems(cfg)
    total = len(all_problems)

    rng = np.random.default_rng(seed)
    indices = np.arange(total)
    rng.shuffle(indices)

    n_total_needed = n_train + n_test
    if n_total_needed > total:
        warnings.warn(
            f"[setting1] Requested {n_total_needed} total problems but only "
            f"{total} exist. Using all {total} instead."
        )
        n_total_needed = total

    train_end = min(n_train, n_total_needed)
    test_end = min(n_total_needed - train_end, n_test)

    train_idx = indices[:train_end]
    test_idx = indices[train_end:train_end + test_end]

    train_problems = [all_problems[i] for i in train_idx]
    test_problems  = [all_problems[i] for i in test_idx]

    _check_budget(n_train, len(train_problems), "train problems (setting1)")
    _check_budget(n_test,  len(test_problems),  "test problems  (setting1)")

    return train_problems, test_problems


# ======================================================================
# Setting 2: position split with triplets_of_interest, generative version
#  (UPDATED: replace uniform-random digits with sampling from a finite
#   background set B to reduce "digit generalization" confound)
# ======================================================================

def _build_background_by_cin(
    cfg,
    rng: np.random.Generator,
    T: List[Triplet],
    background_triplets: Iterable[Triplet] | None,
    background_size_per_cin: int,
) -> Dict[int, List[Triplet]]:
    """
    Build a finite background set B, grouped by carry_in, and disjoint from T.
    """
    base = cfg.base
    T_set = set(T)

    if background_triplets is not None:
        B_by_cin: Dict[int, List[Triplet]] = {}
        for (cin, a, b) in background_triplets:
            if (cin, a, b) in T_set:
                continue
            B_by_cin.setdefault(cin, []).append((cin, a, b))
        return B_by_cin

    B_by_cin: Dict[int, List[Triplet]] = {cin: [] for cin in range(base)}
    B_set = set()

    def _add_if_ok(t: Triplet):
        if t in T_set:
            return
        if t in B_set:
            return
        cin, _, _ = t
        if len(B_by_cin[cin]) >= background_size_per_cin:
            return
        B_by_cin[cin].append(t)
        B_set.add(t)

    max_tries = 1_000_000
    tries = 0
    while tries < max_tries:
        tries += 1
        done = all(len(B_by_cin[cin]) >= background_size_per_cin for cin in range(base))
        if done:
            break
        cin = int(rng.integers(0, base))
        a = int(rng.integers(0, base))
        b = int(rng.integers(0, base))
        _add_if_ok((cin, a, b))

    return {cin: lst for cin, lst in B_by_cin.items() if lst}


def _sample_background_triplet(
    rng: np.random.Generator,
    B_by_cin: Dict[int, List[Triplet]],
    carry_in: int,
) -> Triplet:
    candidates = B_by_cin.get(carry_in, [])
    if not candidates:
        raise ValueError(f"No background triplets available for carry_in={carry_in}")
    return candidates[int(rng.integers(0, len(candidates)))]


# ---------------------------------------------------------------------
# NEW: precompute maps for setting 2
# ---------------------------------------------------------------------

def _precompute_train_candidates(
    cfg,
    col_to_triplets: Dict[int, List[Triplet]],
) -> Dict[int, Dict[int, List[Triplet]]]:
    """
    train_cands[col][cin] = list of triplets allowed at column col with carry_in=cin
    """
    train_cands: Dict[int, Dict[int, List[Triplet]]] = {col: {} for col in range(cfg.n_digits)}
    for col in range(cfg.n_digits):
        by_cin: Dict[int, List[Triplet]] = {}
        for (cin, a, b) in col_to_triplets.get(col, []):
            by_cin.setdefault(cin, []).append((cin, a, b))
        train_cands[col] = by_cin
    return train_cands


def _precompute_forbidden_candidates(
    cfg,
    T: List[Triplet],
    col_to_triplets: Dict[int, List[Triplet]],
) -> Dict[int, Dict[int, List[Triplet]]]:
    """
    forbidden_cands[col][cin] = list of triplets in T with carry_in=cin
                               that are NOT allowed at column col in train.
    """
    # Split T by cin once
    T_by_cin: Dict[int, List[Triplet]] = {}
    for t in T:
        T_by_cin.setdefault(t[0], []).append(t)

    # Allowed sets per column (for membership tests)
    allowed_set_by_col: Dict[int, set] = {
        col: set(trips) for col, trips in col_to_triplets.items()
    }

    forbidden_cands: Dict[int, Dict[int, List[Triplet]]] = {col: {} for col in range(cfg.n_digits)}
    for col in range(cfg.n_digits):
        allowed_here = allowed_set_by_col.get(col, set())
        by_cin: Dict[int, List[Triplet]] = {}
        for cin, trips in T_by_cin.items():
            # keep only trips not allowed here
            by_cin[cin] = [t for t in trips if t not in allowed_here]
        forbidden_cands[col] = by_cin

    return forbidden_cands


# ---------------------------------------------------------------------
# UPDATED sampling using precomputed maps
# ---------------------------------------------------------------------

def _sample_train_problem_setting2(
    cfg,
    rng: np.random.Generator,
    train_cands: Dict[int, Dict[int, List[Triplet]]],
    B_by_cin: Dict[int, List[Triplet]],
    p_use_special: float = 0.5,
):
    base = cfg.base
    n_digits = cfg.n_digits

    top_digits: List[int] = []
    bot_digits: List[int] = []
    carry_in = 0

    for col in range(n_digits):
        candidates = train_cands[col].get(carry_in, [])
        use_special = bool(candidates) and (rng.random() < p_use_special)

        if use_special:
            cin, a, b = candidates[int(rng.integers(0, len(candidates)))]
        else:
            cin, a, b = _sample_background_triplet(rng, B_by_cin, carry_in)

        s = a + b + cin
        carry_in = s // base

        top_digits.append(a)
        bot_digits.append(b)

    top_val = digits_to_number(np.array(top_digits, dtype=np.int64), base=base)
    bot_val = digits_to_number(np.array(bot_digits, dtype=np.int64), base=base)
    xs = np.array([top_val, bot_val], dtype=np.int64)
    return AdditionProblem(xs, cfg)


def _sample_test_problem_setting2(
    cfg,
    rng: np.random.Generator,
    forbidden_cands: Dict[int, Dict[int, List[Triplet]]],
    B_by_cin: Dict[int, List[Triplet]],
    p_use_forbidden: float = 0.5,
):
    base = cfg.base
    n_digits = cfg.n_digits

    while True:
        top_digits: List[int] = []
        bot_digits: List[int] = []
        carry_in = 0
        used_forbidden = False

        for col in range(n_digits):
            forb = forbidden_cands[col].get(carry_in, [])
            use_forb = bool(forb) and (rng.random() < p_use_forbidden)

            if use_forb:
                cin, a, b = forb[int(rng.integers(0, len(forb)))]
                used_forbidden = True
            else:
                cin, a, b = _sample_background_triplet(rng, B_by_cin, carry_in)

            s = a + b + cin
            carry_in = s // base

            top_digits.append(a)
            bot_digits.append(b)

        if not used_forbidden:
            continue

        top_val = digits_to_number(np.array(top_digits, dtype=np.int64), base=base)
        bot_val = digits_to_number(np.array(bot_digits, dtype=np.int64), base=base)
        xs = np.array([top_val, bot_val], dtype=np.int64)
        return AdditionProblem(xs, cfg)


def generate_setting2_position_split(
    cfg,
    n_train: int,
    n_test: int,
    seed: int,
    triplets_of_interest: Iterable[Triplet],
    frac_positions: float,
    background_triplets: Iterable[Triplet] | None = None,
    background_size_per_cin: int = 32,
    p_use_special: float = 0.5,
    p_use_forbidden: float = 0.5,
):
    assert cfg.n_addends == 2
    rng = np.random.default_rng(seed)
    T = list(triplets_of_interest)

    # 0) Background
    B_by_cin = _build_background_by_cin(
        cfg=cfg,
        rng=rng,
        T=T,
        background_triplets=background_triplets,
        background_size_per_cin=background_size_per_cin,
    )

    for cin_needed in [0, 1]:
        if cin_needed in range(cfg.base) and cin_needed not in B_by_cin:
            warnings.warn(
                f"[setting2] Background has no triplets for carry_in={cin_needed}. "
                "You may hit errors if this carry occurs."
            )

    # 1) Allowed columns for each triplet
    cols = np.arange(cfg.n_digits)
    allowed_cols: Dict[Triplet, set] = {}
    for t in T:
        n_allowed = max(1, int(cfg.n_digits * frac_positions))
        choice = rng.choice(cols, size=n_allowed, replace=False)
        allowed_cols[t] = set(int(c) for c in choice)

    col_to_triplets: Dict[int, List[Triplet]] = {col: [] for col in range(cfg.n_digits)}
    for t, cols_set in allowed_cols.items():
        for col in cols_set:
            col_to_triplets[col].append(t)

    # 2) NEW: precompute lookup tables once
    train_cands = _precompute_train_candidates(cfg, col_to_triplets)
    forbidden_cands = _precompute_forbidden_candidates(cfg, T, col_to_triplets)

    # 3) Generate train
    train_problems = [
        _sample_train_problem_setting2(cfg, rng, train_cands, B_by_cin, p_use_special=p_use_special)
        for _ in range(n_train)
    ]

    # 4) Generate test
    test_problems: List[AdditionProblem] = []
    max_tries = 50 * n_test if n_test > 0 else 0
    tries = 0
    while len(test_problems) < n_test and tries < max_tries:
        tries += 1
        test_problems.append(
            _sample_test_problem_setting2(cfg, rng, forbidden_cands, B_by_cin, p_use_forbidden=p_use_forbidden)
        )

    _check_budget(n_train, len(train_problems), "train problems (setting2)")
    _check_budget(n_test,  len(test_problems),  "test problems  (setting2)")

    return train_problems, test_problems, allowed_cols, B_by_cin


# ======================================================================
# Setting 3: order-of-triplets constraint (train pattern vs test pattern)
# ======================================================================

# ======================================================================
# Setting 3: order-of-triplets pattern vs background (generative version)
#   UPDATED: replace uniform-random digits with sampling from a finite
#   background set B (disjoint from pattern triplets) to reduce confounding.
# ======================================================================

# ======================================================================
# Setting 3: carry-conditioned pattern-vs-background shift (generative)
#   UPDATED v2: no dependence on column index / len(pattern).
#   For each column:
#       with prob p_use_pattern -> sample from PATTERN triplets matching carry_in
#       else                  -> sample from BACKGROUND triplets matching carry_in
# ======================================================================

def generate_setting3_order_constraint(
    cfg: BoardConfig,
    n_train: int,
    n_test: int,
    seed: int,
    pattern_train: List[Triplet],
    pattern_test: List[Triplet],
    # background controls
    background_triplets: Iterable[Triplet] | None = None,
    background_size_per_cin: int = 32,
    p_use_pattern: float = 0.5,
) -> Tuple[List[AdditionProblem], List[AdditionProblem], Dict[int, List[Triplet]], Dict[int, List[Triplet]], Dict[int, List[Triplet]]]:
    """
    Setting 3 (carry-conditioned mixture) UPDATED:

    Train and test differ only in which "pattern triplet pool" is preferred.

    For each column:
      - carry_in is determined by previous columns.
      - With probability p_use_pattern:
            sample a triplet from PATTERN pool that matches carry_in
        Else:
            sample a triplet from BACKGROUND pool that matches carry_in
      - Update carry_out from a+b+carry_in.

    Important details:
      - We no longer use col < len(pattern). Patterns are treated as sets/pools.
      - Pattern pools and background pools are both split by carry_in to ensure
        sampling always matches the current carry.
      - Background B is made disjoint from BOTH pattern_train and pattern_test.

    Returns:
      train_problems, test_problems, B_by_cin, Ptr_by_cin, Pte_by_cin
    """
    assert cfg.n_addends == 2
    base = cfg.base
    n_digits = cfg.n_digits
    rng = np.random.default_rng(seed)

    # --- Build pattern pools, split by carry_in ---
    def _split_by_cin(trips: Iterable[Triplet]) -> Dict[int, List[Triplet]]:
        out: Dict[int, List[Triplet]] = {}
        for t in trips:
            cin, a, b = t
            out.setdefault(cin, []).append((cin, a, b))
        return out

    P_train_by_cin = _split_by_cin(pattern_train)
    P_test_by_cin  = _split_by_cin(pattern_test)

    # --- Build background B disjoint from both patterns ---
    pattern_union = set(pattern_train) | set(pattern_test)
    B_by_cin = _build_background_by_cin(
        cfg=cfg,
        rng=rng,
        T=list(pattern_union),                 # treat union as "T" to exclude from B
        background_triplets=background_triplets,
        background_size_per_cin=background_size_per_cin,
    )

    # Safety: ensure we have *some* candidates for common carries
    # (If a carry appears and the pool is empty, we'll fall back to background if possible.)
    def _sample_from_pool_by_cin(pool_by_cin: Dict[int, List[Triplet]], cin: int) -> Triplet | None:
        cands = pool_by_cin.get(cin, [])
        if not cands:
            return None
        return cands[int(rng.integers(0, len(cands)))]

    def _sample_background(cin: int) -> Triplet:
        return _sample_background_triplet(rng, B_by_cin, cin)

    def _sample_one(pattern_by_cin: Dict[int, List[Triplet]]) -> AdditionProblem:
        top_digits: List[int] = []
        bot_digits: List[int] = []
        carry_in = 0

        for _col in range(n_digits):
            use_pat = (rng.random() < p_use_pattern)

            t = None
            if use_pat:
                # try pattern pool first
                t = _sample_from_pool_by_cin(pattern_by_cin, carry_in)

            if t is None:
                # fallback to background (either because use_pat=False, or pattern empty for this cin)
                t = _sample_background(carry_in)

            cin, a, b = t  # cin should match carry_in (pattern pools are split by cin)

            s = a + b + cin
            carry_out = s // base

            top_digits.append(a)
            bot_digits.append(b)
            carry_in = carry_out

        top_val = digits_to_number(np.array(top_digits, dtype=np.int64), base=base)
        bot_val = digits_to_number(np.array(bot_digits, dtype=np.int64), base=base)
        xs = np.array([top_val, bot_val], dtype=np.int64)
        return AdditionProblem(xs, cfg)

    train_problems = [_sample_one(P_train_by_cin) for _ in range(n_train)]
    test_problems  = [_sample_one(P_test_by_cin)  for _ in range(n_test)]

    return train_problems, test_problems, B_by_cin, P_train_by_cin, P_test_by_cin

# ======================================================================
# Setting 4: full triplet hold-out (never seen in train, required in test)
# ======================================================================

# ======================================================================
# Setting 4: full triplet hold-out (generative version)
# ======================================================================

# ======================================================================
# Setting 4: full triplet hold-out (generative version)
#   UPDATED: replace uniform random digits (not in H) with sampling from
#   a finite background set B (disjoint from H), grouped by carry_in.
# ======================================================================

def generate_setting4_triplet_holdout(
    cfg: BoardConfig,
    n_train: int,
    n_test: int,
    seed: int,
    forbidden_triplets: Iterable[Triplet],
    # background controls
    background_triplets: Iterable[Triplet] | None = None,
    background_size_per_cin: int = 64,
    p_use_forbidden: float = 0.5,
) -> Tuple[List[AdditionProblem], List[AdditionProblem], Dict[int, List[Triplet]], Dict[int, List[Triplet]]]:
    """
    Setting 4 (carry-conditioned holdout), aligned with Setting 3 sampling style.

    Pools (split by carry_in):
      - H_by_cin[cin]: forbidden triplets with that cin
      - B_by_cin[cin]: background triplets with that cin, disjoint from H

    TRAIN (per column):
      - always sample from B_by_cin[carry_in]
      - update carry

    TEST (per column):
      - with probability p_use_forbidden:
          sample from H_by_cin[carry_in] if non-empty, else fallback to background
        else:
          sample from background
      - update carry
      - enforce: at least one forbidden used somewhere in the whole test problem
    """
    assert cfg.n_addends == 2
    base = cfg.base
    n_digits = cfg.n_digits
    rng = np.random.default_rng(seed)

    # --- Split forbidden pool by carry ---
    H = list(forbidden_triplets)
    H_set = set(H)
    H_by_cin: Dict[int, List[Triplet]] = {}
    for (cin, a, b) in H:
        H_by_cin.setdefault(cin, []).append((cin, a, b))

    # --- Build background disjoint from forbidden, split by carry ---
    B_by_cin = _build_background_by_cin(
        cfg=cfg,
        rng=rng,
        T=list(H_set),  # exclude forbidden from background
        background_triplets=background_triplets,
        background_size_per_cin=background_size_per_cin,
    )

    def _sample_from_pool_by_cin(pool_by_cin: Dict[int, List[Triplet]], cin: int) -> Triplet | None:
        cands = pool_by_cin.get(cin, [])
        if not cands:
            return None
        return cands[int(rng.integers(0, len(cands)))]

    def _sample_background(cin: int) -> Triplet:
        # uses your helper; raises if missing cin bucket
        return _sample_background_triplet(rng, B_by_cin, cin)

    # --- TRAIN: only background ---
    def _sample_train_problem() -> AdditionProblem:
        top_digits: List[int] = []
        bot_digits: List[int] = []
        carry_in = 0

        for _col in range(n_digits):
            cin, a, b = _sample_background(carry_in)

            s = a + b + cin
            carry_out = s // base

            top_digits.append(a)
            bot_digits.append(b)
            carry_in = carry_out

        top_val = digits_to_number(np.array(top_digits, dtype=np.int64), base=base)
        bot_val = digits_to_number(np.array(bot_digits, dtype=np.int64), base=base)
        xs = np.array([top_val, bot_val], dtype=np.int64)
        return AdditionProblem(xs, cfg)

    # --- TEST: per-column mixture forbidden/background, ensure at least one forbidden ---
    def _sample_test_problem() -> AdditionProblem:
        while True:
            top_digits: List[int] = []
            bot_digits: List[int] = []
            carry_in = 0
            used_forbidden = False

            for _col in range(n_digits):
                use_forb = (rng.random() < p_use_forbidden)

                t: Triplet | None = None
                if use_forb:
                    t = _sample_from_pool_by_cin(H_by_cin, carry_in)

                if t is None:
                    # either we didn't choose forbidden, or no forbidden exists for this carry
                    t = _sample_background(carry_in)
                else:
                    used_forbidden = True

                cin, a, b = t
                s = a + b + cin
                carry_out = s // base

                top_digits.append(a)
                bot_digits.append(b)
                carry_in = carry_out

            if not used_forbidden:
                continue

            top_val = digits_to_number(np.array(top_digits, dtype=np.int64), base=base)
            bot_val = digits_to_number(np.array(bot_digits, dtype=np.int64), base=base)
            xs = np.array([top_val, bot_val], dtype=np.int64)
            return AdditionProblem(xs, cfg)

    # Build train / test
    train_problems: List[AdditionProblem] = [_sample_train_problem() for _ in range(n_train)]

    test_problems: List[AdditionProblem] = []
    max_tries = 50 * n_test if n_test > 0 else 0
    tries = 0
    while len(test_problems) < n_test and tries < max_tries:
        tries += 1
        test_problems.append(_sample_test_problem())

    _check_budget(n_train, len(train_problems), "train problems (setting4)")
    _check_budget(n_test,  len(test_problems),  "test problems  (setting4)")

    # returning both pools can help debug (optional)
    return train_problems, test_problems, B_by_cin, H_by_cin




# ======================================================================
# Small debug main
# ======================================================================
def _debug_print_problems(label: str, cfg: BoardConfig, problems: List[AdditionProblem], max_n: int = 5):
    print(f"\n=== {label} (showing up to {max_n}) ===")
    for i, prob in enumerate(problems[:max_n]):
        xs = prob.operands
        trips = extract_triplets(cfg, xs)
        print(f"  [{i}] {xs[0]} + {xs[1]}  -> triplets: {trips}")


def _debug_print_background(label: str, B_by_cin: Dict[int, List[Triplet]], max_per_cin: int = 10):
    print(f"\n=== {label} (showing up to {max_per_cin} per cin) ===")
    for cin in sorted(B_by_cin.keys()):
        shown = B_by_cin[cin][:max_per_cin]
        print(f"  cin={cin}: {shown} (total={len(B_by_cin[cin])})")


def main():
    # Simple config: 3-digit addition, base 10
    cfg = BoardConfig(H=4, W=5, n_digits=5)

    print("######## Setting 1: random fraction ########")
    # train1, test1 = generate_setting1_random_fraction(cfg, n_train=10, n_test=5, seed=0)
    # _debug_print_problems("Setting 1 - train", cfg, train1)
    # _debug_print_problems("Setting 1 - test",  cfg, test1)

    # -----------------------------
    # Choose a tiny background B so you can inspect prints easily
    # (Make sure it is disjoint from T2 / patterns / forbidden where relevant)
    # -----------------------------
    B_small: List[Triplet] = [
        # cin=0 background
        (0, 0, 1),
        (0, 2, 2),
        (0, 7, 5),
        (0, 8, 9),
        # cin=1 background
        (1, 0, 0),
        (1, 1, 0),
        (1, 2, 8),
        (1, 3, 9),
    ]

    print("\n######## Setting 2: position split with limited triplets ########")
    T2: List[Triplet] = [
        (0, 6, 5),
        (0, 3, 4),
        (1, 6, 5),
        (1, 3, 4),
    ]

    train2, test2, allowed_cols2, B2 = generate_setting2_position_split(
        cfg,
        n_train=10,
        n_test=5,
        seed=16,
        triplets_of_interest=T2,
        frac_positions=0.5,
        background_triplets=B_small,          # <-- force tiny B
        background_size_per_cin=4,            # unused if background_triplets provided
        p_use_special=0.5,
        p_use_forbidden=0.5,
    )
    print(f"Allowed training columns per triplet (setting 2): {allowed_cols2}")
    _debug_print_background("Setting 2 - background B", B2, max_per_cin=10)
    _debug_print_problems("Setting 2 - train", cfg, train2)
    _debug_print_problems("Setting 2 - test",  cfg, test2)

    print("\n######## Setting 3: order constraint ########")
    # IMPORTANT: if you want *order* shift (not triplet-set shift),
    # keep the same triplets and only permute them.
    tA: Triplet = (1, 1, 2)
    tB: Triplet = (0, 3, 4)
    tC: Triplet = (0, 5, 6)
    pattern_train = [tA, tB, tC]
    pattern_test  = [(1,2,1), (0,4,3), (0,6,5)]  # same triplets, different order

    train3, test3, B3, Ptrain3, Ptest3 = generate_setting3_order_constraint(
    cfg,
    n_train=10,
    n_test=10,
    seed=12,
    pattern_train=pattern_train,
    pattern_test=pattern_test,
    background_triplets=B_small,          # <-- tiny B again
    background_size_per_cin=4,
    p_use_pattern=0.5,
)

    #_debug_print_background("Setting 3 - background B", B3, max_per_cin=10)
    #print("\nSetting 3 - pattern pools by carry:")
    #print("  P_train_by_cin:", Ptrain3)
    #print("  P_test_by_cin: ", Ptest3)

    #_debug_print_problems("Setting 3 - train", cfg, train3)
    #_debug_print_problems("Setting 3 - test",  cfg, test3)
    print("\n######## Setting 4: triplet hold-out ########")
    forbidden: List[Triplet] = [(0, 9, 4), (1, 9, 4)]

    # Updated generate_setting4_triplet_holdout returns:
    #   (train4, test4, B_by_cin, H_by_cin)
    train4, test4, B4, H4 = generate_setting4_triplet_holdout(
        cfg,
        n_train=10,
        n_test=10,
        seed=11,
        forbidden_triplets=forbidden,
        background_triplets=B_small,          # <-- tiny B again
        background_size_per_cin=4,
        p_use_forbidden=0.5,
    )

    #_debug_print_background("Setting 4 - background B", B4, max_per_cin=10)
    #print("\nSetting 4 - forbidden pool by carry:")
    #print("  H_by_cin:", H4)

    #_debug_print_problems("Setting 4 - train (no forbidden triplets)", cfg, train4)
    #_debug_print_problems("Setting 4 - test  (contains forbidden triplets)", cfg, test4)



if __name__ == "__main__":
    main()

#setting 1 tests nb of samples needed to generaliye
#setting 2 tests how able to generalise to tripplets appearing in different cols from train/test
#settign 3 tests if can generalise from permutation of cols (ie carry,a ,b becomes carr,b,a)
#setting 4 tests generalisation to unseen tripplets (probably the hardest, none should generalise)