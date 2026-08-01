"""Non-orthogonal subspace projection and coefficient swaps."""

from __future__ import annotations

import torch


def coefficients(
    hidden: torch.Tensor, directions: torch.Tensor, rcond: float = 1e-6
) -> torch.Tensor:
    """Least-squares coefficients for columns of ``directions`` (never assumes orthogonality)."""
    if directions.ndim != 2 or hidden.shape[-1] != directions.shape[0]:
        raise ValueError("directions must be [width, rank] and match hidden width")
    return hidden @ torch.linalg.pinv(directions, rtol=rcond).T


def replace_coefficients(
    hidden: torch.Tensor, directions: torch.Tensor, replacement: torch.Tensor, strength: float = 1.0
) -> tuple[torch.Tensor, torch.Tensor]:
    """Replace span coordinates and return (intervened hidden, applied delta)."""
    old = coefficients(hidden, directions)
    if replacement.shape != old.shape:
        replacement = torch.broadcast_to(replacement, old.shape)
    delta = (replacement - old) @ directions.T * strength
    return hidden + delta, delta


def swap_pair(
    hidden: torch.Tensor, directions: torch.Tensor, strength: float = 1.0
) -> tuple[torch.Tensor, torch.Tensor]:
    """Exchange coefficients of a two-direction, potentially non-orthogonal basis."""
    if directions.shape[1] != 2:
        raise ValueError("pair swap requires exactly two directions")
    old = coefficients(hidden, directions)
    return replace_coefficients(hidden, directions, old.flip(-1), strength)


def matched_random_delta(delta: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    """Isotropic control with exactly the semantic delta norm at every token."""
    random = torch.randn(delta.shape, dtype=delta.dtype, device=delta.device, generator=generator)
    random = random / random.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    return random * delta.norm(dim=-1, keepdim=True)
