# src/models/positional_encodings.py
import math
from typing import Optional, Tuple

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# 1D absolute sinusoidal PE 
# ---------------------------------------------------------------------------

class SinusoidalPositionalEncoding(nn.Module):
    """
    Standard 1D absolute sinusoidal PE (as in "Attention Is All You Need").
    """

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)  # (L, D)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, L, D) for broadcasting over batch
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, L, D)
        """
        L = x.size(1)
        return x + self.pe[:, :L, :]
    
class LearnedPositionalEncoding1D(nn.Module):
    """
    Learned 1D absolute positional encoding (trainable).
    Like GPT/BERT learned position embeddings.

    Adds a trainable embedding for each position 0..max_len-1.
    """
    def __init__(self, d_model: int, max_len: int = 500):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.pos_emb = nn.Embedding(max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, L, D)
        """
        B, L, D = x.shape
        if D != self.d_model:
            raise ValueError(f"Mismatch in d_model: x has D={D}, expected {self.d_model}")
        if L > self.max_len:
            raise ValueError(f"Sequence length L={L} exceeds max_len={self.max_len}")

        positions = torch.arange(L, device=x.device, dtype=torch.long)  # (L,)
        pe = self.pos_emb(positions).unsqueeze(0)  # (1, L, D)
        return x + pe


def _build_sinusoid_table(length: int, d_model: int, device=None, dtype=torch.float32) -> torch.Tensor:
    """
    Returns: (length, d_model) fixed sinusoidal table (no learnable params).
    Same formula as standard 1D sinusoidal PE.
    """
    if length <= 0:
        raise ValueError(f"length must be > 0, got {length}")
    if d_model <= 0:
        raise ValueError(f"d_model must be > 0, got {d_model}")

    device = device if device is not None else torch.device("cpu")
    pe = torch.zeros(length, d_model, device=device, dtype=dtype)

    position = torch.arange(0, length, device=device, dtype=dtype).unsqueeze(1)  # (L, 1)
    div_term = torch.exp(
        torch.arange(0, d_model, 2, device=device, dtype=dtype) * (-math.log(10000.0) / d_model)
    )  # (d_model/2,)

    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe


class SinusoidalPositionalEncoding2D(nn.Module):
    """
    Fixed 2D absolute positional encoding for an H x W grid (NO learnable params).

    For flattened position p (0..H*W-1):
        row = p // W
        col = p % W
        PE[p] = row_sin[row] + col_sin[col]

    where row_sin is a 1D sinusoidal table of shape (H, d_model),
    and col_sin is a 1D sinusoidal table of shape (W, d_model).

    This matches your previous learned Abs2D behavior (row_emb + col_emb),
    but with fixed sin/cos features instead of embeddings.
    """

    def __init__(self, d_model: int, H: int, W: int, scale: float = 1.0):
        super().__init__()
        self.d_model = d_model
        self.H = H
        self.W = W
        self.scale = float(scale)

        # Build fixed sin/cos tables on CPU; they’ll move with .to(device)
        row_pe = _build_sinusoid_table(H, d_model, device=torch.device("cpu"))
        col_pe = _build_sinusoid_table(W, d_model, device=torch.device("cpu"))

        # Store as buffers (not trainable)
        self.register_buffer("row_pe", row_pe, persistent=False)  # (H, D)
        self.register_buffer("col_pe", col_pe, persistent=False)  # (W, D)

        # Precompute row/col indices for flattened positions
        positions = torch.arange(H * W, dtype=torch.long)
        row_idx = positions // W
        col_idx = positions % W
        self.register_buffer("row_idx", row_idx, persistent=False)  # (L,)
        self.register_buffer("col_idx", col_idx, persistent=False)  # (L,)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, L, D) where L == H*W and D == d_model
        returns x + scale * PE
        """
        B, L, D = x.shape
        if D != self.d_model:
            raise ValueError(f"Mismatch in d_model: x has D={D}, expected {self.d_model}")
        if L != self.H * self.W:
            raise ValueError(f"Sequence length must be H*W={self.H*self.W}, got L={L}")

        # row_pe[row_idx] -> (L, D), col_pe[col_idx] -> (L, D)
        pe = self.row_pe[self.row_idx] + self.col_pe[self.col_idx]  # (L, D)
        pe = pe.unsqueeze(0)  # (1, L, D)
        return x + self.scale * pe

# ---------------------------------------------------------------------------
# 2D absolute PE: row + column embeddings (DETR-style)
# ---------------------------------------------------------------------------
class LearnedPositionalEncoding2D(nn.Module):
    """
    2D absolute positional encoding for an H x W grid.

    For each flattened position p (0..H*W-1), we compute:
        row = p // W
        col = p % W
        PE[p] = row_emb[row] + col_emb[col]

    This is the "row+column embeddings" style often used in vision Transformers.
    """


    def __init__(self, d_model: int, H: int, W: int):
        super().__init__()
        self.d_model = d_model
        self.H = H
        self.W = W

        self.row_emb = nn.Embedding(H, d_model)
        self.col_emb = nn.Embedding(W, d_model)

        # Precompute row/col indices for flattened positions as buffers
        positions = torch.arange(H * W, dtype=torch.long)
        row_idx = positions // W
        col_idx = positions % W
        self.register_buffer("row_idx", row_idx, persistent=False)
        self.register_buffer("col_idx", col_idx, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, L, D) where L == H * W
        returns x + PE
        """
        B, L, D = x.shape
        assert D == self.d_model, "Mismatch in d_model."
        assert L == self.H * self.W, "Sequence length must be H*W for 2D PE."

        row_pe = self.row_emb(self.row_idx)  # (L, D)
        col_pe = self.col_emb(self.col_idx)  # (L, D)
        pe = row_pe + col_pe                 # (L, D)
        pe = pe.unsqueeze(0)                 # (1, L, D) for broadcast

        return x + pe

# ---------------------------------------------------------------------------
# 2D relative position bias: B(q,k) = b_x(Δx) + b_y(Δy)
# ---------------------------------------------------------------------------

class RelativePositionBias2D(nn.Module):
    """
    2D relative position bias for an H x W grid.

    For flattened positions i,j with coordinates (x_i, y_i), (x_j, y_j),
    we define:

        Δx = x_i - x_j        in [-(H-1), ..., +(H-1)]
        Δy = y_i - y_j        in [-(W-1), ..., +(W-1)]

    and learn per-head biases:

        B_x(Δx) in R^{n_heads},  B_y(Δy) in R^{n_heads}

    The final bias is:
        B(h, i, j) = B_x(h, Δx) + B_y(h, Δy),

    returned as a tensor of shape (n_heads, L, L) where L = H * W.

    You add this to the attention logits for each layer.
    """

    def __init__(self, n_heads: int, H: int, W: int):
        super().__init__()
        self.n_heads = n_heads
        self.H = H
        self.W = W

        self.rel_height = nn.Embedding(2 * H - 1, n_heads)  # Δx
        self.rel_width = nn.Embedding(2 * W - 1, n_heads)   # Δy

        # Precompute row/col indices and relative offsets as buffers
        positions = torch.arange(H * W, dtype=torch.long)
        row = positions // W    # (L,)
        col = positions % W     # (L,)
        self.register_buffer("row", row, persistent=False)
        self.register_buffer("col", col, persistent=False)

        # Δ indices will be built on the fly in forward (depends on device)

    def forward(self) -> torch.Tensor:
        """
        Returns:
            bias: (n_heads, L, L) tensor suitable to add to attention logits.
        """
        device = self.row.device
        H, W = self.H, self.W
        n_heads = self.n_heads

        row = self.row.to(device)  # (L,)
        col = self.col.to(device)  # (L,)
        L = row.numel()

        # Compute Δx and Δy matrices: (L, L)
        delta_x = row.unsqueeze(1) - row.unsqueeze(0)   # (L, L)
        delta_y = col.unsqueeze(1) - col.unsqueeze(0)   # (L, L)

        #result[i, j] = row[i, 0] - row[0, j]

        # Shift to be >= 0 for embedding lookup
        delta_x_index = delta_x + (H - 1)  # in [0, 2H-2]
        delta_y_index = delta_y + (W - 1)  # in [0, 2W-2]

        # Look up biases for each offset; result: (L, L, n_heads)
        bias_x = self.rel_height(delta_x_index)  # (L, L, n_heads)
        bias_y = self.rel_width(delta_y_index)   # (L, L, n_heads)

        bias = bias_x + bias_y                   # (L, L, n_heads)
        bias = bias.permute(2, 0, 1)             # (n_heads, L, L)

        return bias
        # In attention: expand to (B*n_heads, L, L) and add to attn logits.

#mixed absolute + relative positional encoding
#eg to call : 
"""
pos_enc = Abs2DPlusRelBias2D(
    abs_pe=AbsolutePositionalEncoding2D(d_model, H, W),
    rel_bias=RelativePositionBias2D(n_heads=nhead, H=H, W=W),
)
model = BlackboardTransformer(vocab_size=vocab, d_model=d_model, nhead=nhead, pos_enc=pos_enc)
"""

# ---------------------------------------------------------------------------
# Combined absolute 2D PE + relative 2D position bias
# ---------------------------------------------------------------------------
class Abs2DPlusRelBias2D(nn.Module):
    """
    Combine absolute 2D positional encoding (added to x)
    + relative 2D position bias (added to attention logits).

    - forward(x) returns x + abs_pe(x)
    - get_pos_bias() returns (n_heads, L, L) like RelativePositionBias2D
    """
    def __init__(
        self,
        abs_pe: nn.Module ,
        rel_bias: RelativePositionBias2D ,
    ):
        super().__init__()
        self.abs_pe = abs_pe
        self.rel_bias = rel_bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.abs_pe is None:
            return x
        return self.abs_pe(x)

    def get_pos_bias(self) -> torch.Tensor:
        return self.rel_bias()



# ---------------------------------------------------------------------------
# Rotary positional embedding helper (1D, usable on flattened grid)
# ---------------------------------------------------------------------------

class RotaryPositionalEmbedding1D(nn.Module):
    """
    Rotary positional embedding (RoPE) helper.

    This module does NOT modify your token embeddings directly; instead it
    provides a function to apply RoPE to query/key tensors inside attention.

    Typical usage once you implement a custom attention:

        rope = RotaryPositionalEmbedding1D(dim=head_dim, max_len=H*W)
        q_rot, k_rot = rope(q, k)  # q,k: (B, n_heads, L, head_dim)

    For now this is a helper; integrating it into attention will require
    slight changes to your InspectableEncoderLayer (computing Q,K explicitly).
    """

    def __init__(self, dim: int, max_len: int):
        """
        dim: head dimension (must be even)
        max_len: maximum sequence length (e.g. H*W)
        """
        super().__init__()
        assert dim % 2 == 0, "RoPE dimension must be even."
        self.dim = dim

        # Precompute angles for positions 0..max_len-1
        inv_freq = 1.0 / (
            10000
            ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim)
        )  # (dim/2,)
        t = torch.arange(max_len, dtype=torch.float32)  # (L,)
        freqs = torch.einsum("i,j->ij", t, inv_freq)  # (L, dim/2)
        # Build cos/sin for interleaved pairs
        emb = torch.cat([freqs, freqs], dim=-1)  # (L, dim)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def _apply_rotary(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        """
        x:   (B, n_heads, L, D)
        cos: (L, D)
        sin: (L, D)
        """
        # rotate every (2i, 2i+1) pair
        x1, x2 = x[..., ::2], x[..., 1::2]
        cos = cos[None, None, :, ::2]
        sin = sin[None, None, :, ::2]
        # standard RoPE formula
        x_rotated_first = x1 * cos - x2 * sin
        x_rotated_second = x1 * sin + x2 * cos
        x_out = torch.stack([x_rotated_first, x_rotated_second], dim=-1)
        # merge last two dims back to D
        x_out = x_out.flatten(-2)  # (B, n_heads, L, D)
        return x_out

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply RoPE to queries and keys.

        q, k: (B, n_heads, L, D)
        Returns:
            q_rot, k_rot: same shape as inputs
        """
        B, n_heads, L, D = q.shape
        assert D == self.dim, "Head dimension mismatch in RoPE."

        cos = self.cos_cached[:L, :]  # (L, D)
        sin = self.sin_cached[:L, :]  # (L, D)

        q_rot = self._apply_rotary(q, cos, sin)
        k_rot = self._apply_rotary(k, cos, sin)
        return q_rot, k_rot