from typing import Dict, List
import torch
from torch.utils.data import Dataset

from src.data.addition_algo import BoardConfig, generate_trajectory_variant_A as generate_addition_trajectory, BLANK_TOKEN, VOID_TOKEN
from src.data.subtraction_algo import generate_trajectory_variant_A as generate_subtraction_trajectory
from src.data.multi_addition_algo import generate_multi_trajectory
from src.data.problems import AdditionProblem, SubtractionProblem, MultiAdditionProblem


class BlackboardAdditionStepDataset(Dataset):
    """
    Dataset for masked-write training (variant A) on 2D blackboard addition.

    Each item corresponds to one step t in one trajectory:
      - input_ids:  S_t  flattened to shape (L,)
      - target_ids: S_{t+1} flattened to shape (L,)
      - mask:       M_t flattened to shape (L,), boolean

    Where L = H * W.
    """

    def __init__(self, problems: List[AdditionProblem]) -> None:
        super().__init__()
        assert len(problems) > 0, "Problems list must not be empty."
        self.problems = problems
        self.cfg: BoardConfig = problems[0].cfg
        self.n_steps = self.cfg.n_digits  # one step per digit

    def __len__(self) -> int:
        # one sample per (example, step)
        return len(self.problems) * self.n_steps

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        traj_idx = idx // self.n_steps
        step_idx = idx % self.n_steps

        problem = self.problems[traj_idx]
        xs = problem.operands

        S_seq, M_seq = generate_addition_trajectory(self.cfg, xs)

        S_t   = torch.from_numpy(S_seq[step_idx]).view(-1).long()
        S_tp1 = torch.from_numpy(S_seq[step_idx + 1]).view(-1).long()
        M_t   = torch.from_numpy(M_seq[step_idx]).view(-1)

        return {
            "input_ids": S_t,      # (L,)
            "target_ids": S_tp1,   # (L,)
            "mask": M_t,           # (L,) bool
        }
def _sample_bert_style_mask(eligible: torch.Tensor, rate: float, rng: torch.Generator) -> torch.Tensor:
    """Sample a boolean mask (same shape as `eligible`) from eligible positions
    eligible: bool tensor, True where masking is allowed
    rate: probability of selecting each eligible position
    Returns a bool tensor M with M[p]=True meaning "predict/denoise this position"
    """
    if rate <= 0.0:
        return torch.zeros_like(eligible, dtype=torch.bool)
    u = torch.rand(eligible.shape, generator=rng)
    return eligible & (u < rate)


def _apply_bert_corruption(input_ids: torch.Tensor, denoise_mask: torch.Tensor, rng: torch.Generator, random_token_ids: torch.Tensor, void_token_id: int = VOID_TOKEN, p_void: float = 0.8, p_random: float = 0.1) -> torch.Tensor:
    """Apply BERT-style corruption in-place on a copy of input_ids
    80% -> replace with VOID
    10% -> replace with random valid token
    10% -> keep unchanged
    """
    if denoise_mask.sum().item() == 0:
        return input_ids

    x = input_ids.clone()
    idx = denoise_mask.nonzero(as_tuple=False).view(-1)
    # sample decision per masked position
    r = torch.rand(idx.shape[0], generator=rng)

    # 80% VOID
    void_sel = idx[r < p_void]
    if void_sel.numel() > 0:
        x[void_sel] = int(void_token_id)

    # 10% random token, choose random tokens from provided set
    rand_sel = idx[(r >= p_void) & (r < (p_void + p_random))]
    if rand_sel.numel() > 0:
        choice = torch.randint(
            low=0,
            high=random_token_ids.numel(),
            size=(rand_sel.numel(),),
            generator=rng,
        )
        x[rand_sel] = random_token_ids[choice]

    # remaining 10%: keep unchanged
    return x

class BlackboardSubtractionStepDataset(Dataset):
    def __init__(self, problems: List[SubtractionProblem]) -> None:
        super().__init__()
        assert len(problems) > 0
        self.problems = problems
        self.cfg: BoardConfig = problems[0].cfg
        self.n_steps = self.cfg.n_digits

    def __len__(self) -> int:
        return len(self.problems) * self.n_steps

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        traj_idx = idx // self.n_steps
        step_idx = idx % self.n_steps

        problem = self.problems[traj_idx]
        xs = problem.operands

        S_seq, M_seq = generate_subtraction_trajectory(self.cfg, xs)

        S_t = torch.from_numpy(S_seq[step_idx]).view(-1).long()
        S_tp1 = torch.from_numpy(S_seq[step_idx + 1]).view(-1).long()
        M_t = torch.from_numpy(M_seq[step_idx]).view(-1)

        return {
            "input_ids": S_t,
            "target_ids": S_tp1,
            "mask": M_t,
        }
        

class BlackboardMultiAdditionStepDataset(Dataset):
    def __init__(self, problems: List[MultiAdditionProblem]) -> None:
        super().__init__()
        assert len(problems) > 0
        self.problems = problems
        self.cfg: BoardConfig = problems[0].cfg
        self.n_steps = self.cfg.n_digits

    def __len__(self) -> int:
        return len(self.problems) * self.n_steps

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        traj_idx = idx // self.n_steps
        step_idx = idx % self.n_steps

        problem = self.problems[traj_idx]
        xs = problem.operands

        S_seq, M_seq = generate_multi_trajectory(self.cfg, xs)

        S_t = torch.from_numpy(S_seq[step_idx]).view(-1).long()
        S_tp1 = torch.from_numpy(S_seq[step_idx + 1]).view(-1).long()
        M_t = torch.from_numpy(M_seq[step_idx]).view(-1)

        return {
            "input_ids": S_t,
            "target_ids": S_tp1,
            "mask": M_t,
        }
    
class BlackboardAdditionDenoisingStepDataset(Dataset):
    """Masked-write + BERT-style denoising dataset (Direction 2).

    Compared to BlackboardAdditionStepDataset, this dataset:
      - corrupts a subset of *eligible* cells in S_t using BERT-style corruption
        (80% VOID / 10% random / 10% unchanged, modulable)
      - expands the training mask to include denoising positions:
            M_loss = M_stepwrite ∪ M_denoise

    Two eligibility regimes:
      - setting="local"  (Setting 2): eligible = stepwrite cells + last-step cells
      - setting="global" (Setting 3): eligible = all editable cells (carry/result rows)
    """

    def __init__(self, problems: List[AdditionProblem], setting: str = "local", denoise_rate: float = 0.15, p_revert: float = 0.0, revert_token_id: int = VOID_TOKEN, disable_stepwrite_on_revert: bool = True, seed: int = 42) -> None:
        super().__init__()
        assert len(problems) > 0
        assert setting in {"local", "global"}
        # - explicit revert/backtracking
        # With probability p_revert (and when step_idx > 0), we create a sample where
        # the target asks the model to write `revert_token_id` (typically VOID) into
        # the last written cells (and possibly more for global), instead of advancing, to train reverting
        self.problems = problems
        self.cfg: BoardConfig = problems[0].cfg
        self.n_steps = self.cfg.n_digits
        self.setting = setting
        self.denoise_rate = float(denoise_rate)
        self.p_revert = float(p_revert)
        self.revert_token_id = int(revert_token_id)
        self.disable_stepwrite_on_revert = bool(disable_stepwrite_on_revert)
        self.seed = int(seed)

        # For random replacements we restrict to non-operator tokens.
        # (digits + BLANK + VOID)
        self._random_token_ids = torch.tensor(
            list(range(10)) + [BLANK_TOKEN, VOID_TOKEN], dtype=torch.long
        )

    def __len__(self) -> int:
        return len(self.problems) * self.n_steps


    def _positions_prev_step(self, *, step_idx: int) -> torch.Tensor:
        """Return a bool mask (flattened) for the previous step cells (result+carry).

        Used for Setting 2-style local revert (erase last-step result+carry).
        If step_idx == 0, returns all-False (as nothing to revert yet).
        """
        H, W = self.cfg.H, self.cfg.W
        L = H * W
        m = torch.zeros(L, dtype=torch.bool)
        if step_idx <= 0:
            return m

        col_end = W - 1
        col_prev = col_end - (step_idx - 1)  # one column to the right of current
        if 0 <= col_prev < W:
            m[self.cfg.result_row * W + col_prev] = True
        if 0 <= (col_prev - 1) < W:
            m[self.cfg.carry_row * W + (col_prev - 1)] = True
        return m


    def _positions_last_k_steps(self, *, step_idx: int, k: int) -> torch.Tensor:
        """Return a bool mask (flattened) for the last k written steps (result+carry).

        Used for Setting 3-style global revert (erase multiple recent columns at once).
        For k=1, this is equivalent to previous step when step_idx > 0.
        """
        H, W = self.cfg.H, self.cfg.W
        L = H * W
        m = torch.zeros(L, dtype=torch.bool)
        if step_idx <= 0 or k <= 0:
            return m

        col_end = W - 1
        # steps already written at time t are: (step_idx) digits to the right
        k = min(k, step_idx)  # cannot erase more than already written
        for i in range(k):
            col = col_end - i
            if 0 <= col < W:
                m[self.cfg.result_row * W + col] = True
            if 0 <= (col - 1) < W:
                m[self.cfg.carry_row * W + (col - 1)] = True
        return m
    

    def _eligible_positions(self, *, step_idx: int) -> torch.Tensor:
        """Return a bool mask of eligible denoising positions (flattened)."""
        H, W = self.cfg.H, self.cfg.W
        L = H * W

        eligible = torch.zeros(L, dtype=torch.bool)

        if self.setting == "global":
            # Setting 3: allow edits anywhere on editable rows (carry + result).
            for r in [self.cfg.carry_row, self.cfg.result_row]:
                start = r * W
                eligible[start : start + W] = True
            return eligible

        # Setting 2: local revert.
        # Eligible = last-step written cells (result+carry of t-1) plus current step write mask.
        col_end = W - 1
        col_cur = col_end - step_idx
        # current step cells: result at col_cur and carry at col_cur-1
        if 0 <= col_cur < W:
            eligible[self.cfg.result_row * W + col_cur] = True
        if 0 <= (col_cur - 1) < W:
            eligible[self.cfg.carry_row * W + (col_cur - 1)] = True

        # previous step cells (if step_idx > 0)
        if step_idx > 0:
            col_prev = col_end - (step_idx - 1)  # one column to the right
            if 0 <= col_prev < W:
                eligible[self.cfg.result_row * W + col_prev] = True
            if 0 <= (col_prev - 1) < W:
                eligible[self.cfg.carry_row * W + (col_prev - 1)] = True

        return eligible


    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        traj_idx = idx // self.n_steps
        step_idx = idx % self.n_steps

        problem = self.problems[traj_idx]
        xs = problem.operands

        S_seq, M_seq = generate_addition_trajectory(self.cfg, xs)

        S_t = torch.from_numpy(S_seq[step_idx]).view(-1).long()
        S_tp1 = torch.from_numpy(S_seq[step_idx + 1]).view(-1).long()
        M_stepwrite = torch.from_numpy(M_seq[step_idx]).view(-1).bool()

        # Per-sample RNG (deterministic given seed + idx)
        rng = torch.Generator()
        rng.manual_seed(self.seed + idx) # deterministic per sample

        eligible = self._eligible_positions(step_idx=step_idx)
        M_denoise = _sample_bert_style_mask(eligible=eligible, rate=self.denoise_rate, rng=rng)

        # - explicit revert/backtracking supervision
        # If enabled, sometimes ask the model to output `revert_token_id` (VOID)
        # in the last written cells, instead of advancing
        do_revert = (self.p_revert > 0.0) and (step_idx > 0)
        if do_revert: 
            u_revert = torch.rand((), generator=rng).item()
            do_revert = u_revert < self.p_revert # probabilistic decision

        target_ids = S_tp1.clone()

        # Build a revert mask (positions we want to erase) + inject an inconsistency
        # in the input so the model has a cue to revert
        revert_mask = torch.zeros_like(M_stepwrite)
        input_corrupt = S_t.clone()

        if do_revert:
            if self.setting == "local":
                revert_mask = self._positions_prev_step(step_idx=step_idx)
            else:
                # For global revert, erase a random number of the last written columns
                # (1..step_idx). This approximates "jump back to last valid state"
                k = int(torch.randint(1, step_idx + 1, (1,), generator=rng).item())
                revert_mask = self._positions_last_k_steps(step_idx=step_idx, k=k)

            # Inject a plausible wrong digit/carry in revert positions
            # We use digits 0-9 and ensure it's different from the clean token
            idxs = revert_mask.nonzero(as_tuple=False).view(-1)
            if idxs.numel() > 0:
                wrong = torch.randint(0, 10, (idxs.numel(),), generator=rng, dtype=torch.long)
                clean = input_corrupt[idxs]
                # ensure wrong != clean when clean is a digit; if clean is not a digit, just use wrong
                is_digit = clean < 10
                if is_digit.any():
                    same = is_digit & (wrong == clean)
                    # bump by 1 mod 10 to ensure different digit
                    wrong[same] = (wrong[same] + 1) % 10
                input_corrupt[idxs] = torch.where(is_digit, wrong, wrong)

            # Target asks to erase these cells
            target_ids[revert_mask] = int(self.revert_token_id)

        # Apply BERT-style corruption (denoising) on top.
        input_noisy = _apply_bert_corruption(
            input_ids=input_corrupt,
            denoise_mask=M_denoise,
            rng=rng,
            random_token_ids=self._random_token_ids,
            void_token_id=VOID_TOKEN,
        )

        # Loss mask logic:
        # - Normal samples: supervise stepwrite ∪ denoise
        # - Revert samples: supervise revert (and optionally denoise); usually NOT stepwrite
        # (revert sample = the board is inconsistent, erase the last written stuff before continuing)
        if do_revert and self.disable_stepwrite_on_revert:
            M_loss = (revert_mask | M_denoise).bool()
        else:
            M_loss = (M_stepwrite | M_denoise | revert_mask).bool()

        return {
            "input_ids": input_noisy,       # (L,)
            "target_ids": target_ids,       # (L,)
            "mask": M_loss,                 # (L,) bool
            "denoise_mask": M_denoise,      # (L,) bool
            "revert_mask": revert_mask,     # (L,) bool
            "do_revert": torch.tensor(int(do_revert)),
        }