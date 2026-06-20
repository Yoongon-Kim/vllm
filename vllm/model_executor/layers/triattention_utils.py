# SPDX-License-Identifier: Apache-2.0
"""TriAttention frequency-domain scoring helpers (vLLM-side, decode path).

Self-contained copy of the three scoring primitives from the reference
TriAttention impl (LRoSA-dev/triattention/methods/pruning_utils.py) needed by
``TriAttentionMLAIndexer``. Kept byte-faithful to the reference so the MLA port
reproduces the original score exactly on the RoPE'd complex pairs.

Conventions (must match the calibration side that produced the stats):
  * ``style="half"``  — RoPE pairs the front/back halves of the rotated dim
    (real = ``x[..., :d/2]``, imag = ``x[..., d/2:]``). This is GLM-4.7-Flash's
    ``apply_rotary_pos_emb`` convention (``rope_interleave`` off).
  * ``omega``         — the rope ``inv_freq`` (one per complex pair).
  * ``freq_scale_sq`` — square of ``compute_frequency_scaling`` (= 1 for
    attention_scaling == 1, i.e. default rope).
  * geometric ``offsets`` = [1, 2, 4, 8, ...] up to ``offset_max_length``.
"""
from __future__ import annotations

import torch


def to_complex_pairs(tensor: torch.Tensor, *, style: str = "half") -> torch.Tensor:
    """Pack the last (even) dim into complex pairs. ``half`` = front/back halves."""
    if tensor.size(-1) % 2 != 0:
        raise ValueError("Head dimension must be even to form complex pairs")
    real_dtype = (torch.float32
                  if tensor.dtype in (torch.bfloat16, torch.float16)
                  else tensor.dtype)
    t = tensor.to(dtype=real_dtype)
    if style == "interleaved":
        return torch.complex(t[..., ::2].contiguous(), t[..., 1::2].contiguous())
    fc = tensor.shape[-1] // 2
    return torch.complex(t[..., :fc].contiguous(), t[..., fc:].contiguous())


def build_geometric_offsets(max_length: int, device: torch.device) -> torch.Tensor:
    """Geometric offset grid [1, 2, 4, ...] <= max_length (RoPE aliasing probe)."""
    if max_length < 1:
        raise ValueError("offset_max_length must be >= 1")
    offsets = []
    value = 1
    while value <= max_length:
        offsets.append(float(value))
        value *= 2
    return torch.tensor(offsets, device=device, dtype=torch.float32)


def compute_frequency_statistics_from_means(
    q_mean_complex: torch.Tensor,
    q_abs_mean: torch.Tensor,
    k_unrot: torch.Tensor,
    *,
    style: str = "half",
    disable_mlr: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """amp/phi/extra terms from the calibrated query stat + PRE-RoPE keys.

    q_mean_complex [nFC] (complex), q_abs_mean [nFC] (real), k_unrot [L, rope]
    (pre-rope key) -> amp [L, nFC], phi [L, nFC], extra [L, nFC].
    """
    k_complex = to_complex_pairs(k_unrot, style=style)          # [L, nFC]
    q_mean_abs = torch.abs(q_mean_complex)                      # [nFC]
    k_abs = torch.abs(k_complex)                                # [L, nFC]
    relative = q_mean_complex.unsqueeze(0) * torch.conj(k_complex)  # [L, nFC]
    phi = torch.atan2(relative.imag, relative.real)            # [L, nFC]
    amp = q_mean_abs.unsqueeze(0) * k_abs                       # [L, nFC]
    if disable_mlr:
        extra = q_abs_mean.unsqueeze(0) * k_abs
    else:
        extra = (q_abs_mean - q_mean_abs).unsqueeze(0) * k_abs  # [L, nFC]
    return amp, phi, extra


def score_keys_for_round(
    key_indices: torch.Tensor,
    round_start: int,
    amp: torch.Tensor,
    phi: torch.Tensor,
    omega: torch.Tensor,
    extra: torch.Tensor,
    offsets: torch.Tensor,
    aggregation: str,
    freq_scale_sq: torch.Tensor,
    disable_trig: bool = False,
) -> torch.Tensor:
    """TriAttention frequency score of each key for a given decode round.

    delta = round_start - key_idx; delta_grid = delta + offsets;
    phase = delta_grid * omega + phi; base = Σ_FC amp*freq_scale_sq*cos(phase);
    additive = Σ_FC extra*freq_scale_sq; combined = base + additive; aggregate
    over the offset grid (mean | max). Returns [L].
    """
    if key_indices.numel() == 0:
        return torch.empty(0, device=amp.device, dtype=torch.float32)

    base_delta = round_start - key_indices.to(device=amp.device, dtype=torch.float32)
    delta_grid = base_delta.unsqueeze(1) + offsets.unsqueeze(0)         # [L, n_off]

    freq_scale_sq = freq_scale_sq.to(device=amp.device, dtype=torch.float32)
    phase = delta_grid.unsqueeze(2) * omega.view(1, 1, -1) + phi.unsqueeze(1)  # [L,n_off,nFC]
    cos_phase = torch.cos(phase)
    scale = freq_scale_sq.view(1, 1, -1)
    base_scores = (amp.unsqueeze(1) * scale * cos_phase).sum(dim=2)     # [L, n_off]
    # additive uses the original freq_scale_sq (offset-independent, broadcast).
    additive = (extra * freq_scale_sq.view(1, -1)).sum(dim=1, keepdim=True)  # [L, 1]
    combined = additive if disable_trig else (base_scores + additive)  # [L, n_off]

    if aggregation == "mean":
        return combined.mean(dim=1)
    return combined.max(dim=1).values
