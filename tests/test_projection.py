import pytest
import torch
from torch import nn

from jspace_study.intervention import SwapHooks
from jspace_study.projection import (
    coefficients,
    matched_random_delta,
    replace_coefficients,
    swap_pair,
)


def test_nonorthogonal_coefficients_recovered() -> None:
    directions = torch.tensor([[1.0, 1.0], [0.0, 1.0], [0.0, 0.0]])
    expected = torch.tensor([[2.0, -3.0]])
    hidden = expected @ directions.T + torch.tensor([[0.0, 0.0, 7.0]])
    torch.testing.assert_close(coefficients(hidden, directions), expected)


def test_swap_exchanges_nonorthogonal_coordinates_and_preserves_complement() -> None:
    directions = torch.tensor([[1.0, 0.8], [0.0, 0.6], [0.0, 0.0]])
    hidden = torch.tensor([[2.0, 3.0, 11.0]])
    before = coefficients(hidden, directions)
    changed, delta = swap_pair(hidden, directions)
    torch.testing.assert_close(
        coefficients(changed, directions), before.flip(-1), atol=1e-5, rtol=1e-5
    )
    assert changed[0, 2] == 11 and torch.allclose(changed, hidden + delta)


def test_swap_twice_restores_activation_and_is_symmetric() -> None:
    directions = torch.tensor([[1.0, 0.6], [0.2, 1.0], [0.0, 0.0]], dtype=torch.float64)
    hidden = torch.tensor([[3.0, -2.0, 7.5]], dtype=torch.float64)
    first, forward_delta = swap_pair(hidden, directions)
    restored, reverse_delta = swap_pair(first, directions)
    torch.testing.assert_close(restored, hidden, atol=1e-10, rtol=1e-10)
    torch.testing.assert_close(reverse_delta, -forward_delta, atol=1e-10, rtol=1e-10)


def test_zero_equal_coefficients_are_noop() -> None:
    directions = torch.tensor([[1.0, 0.5], [0.0, 1.0], [0.0, 0.0]])
    orthogonal_only = torch.tensor([[0.0, 0.0, 4.0]])
    changed, delta = swap_pair(orthogonal_only, directions)
    torch.testing.assert_close(changed, orthogonal_only, atol=1e-7, rtol=1e-7)
    torch.testing.assert_close(delta, torch.zeros_like(delta), atol=1e-7, rtol=1e-7)


def test_orthogonal_residual_is_preserved_exactly() -> None:
    directions = torch.tensor([[1.0, 0.4], [0.0, 1.0], [0.0, 0.0], [0.0, 0.0]])
    hidden = torch.tensor([[2.0, 3.0, -5.0, 11.0]])
    changed, _ = swap_pair(hidden, directions)
    torch.testing.assert_close(changed[:, 2:], hidden[:, 2:], atol=0, rtol=0)


def test_replace_shape_validation() -> None:
    with pytest.raises(ValueError):
        replace_coefficients(torch.ones(2), torch.ones(3, 1), torch.zeros(1))


def test_random_control_is_norm_matched_and_deterministic() -> None:
    delta = torch.tensor([[[3.0, 4.0, 0.0], [0.0, 0.0, 2.0]]])
    first = matched_random_delta(delta, torch.Generator().manual_seed(4))
    second = matched_random_delta(delta, torch.Generator().manual_seed(4))
    torch.testing.assert_close(first.norm(dim=-1), delta.norm(dim=-1))
    torch.testing.assert_close(first.square().sum(dim=-1), delta.square().sum(dim=-1))
    torch.testing.assert_close(first, second)


def test_random_hook_records_per_invocation_per_token_energy() -> None:
    block = nn.Identity()
    directions = {0: torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])}
    hidden = torch.tensor([[[3.0, 1.0, 2.0], [1.0, 4.0, 5.0]]])
    hook = SwapHooks([block], directions, kind="random", seed=7)
    with hook:
        block(hidden)
        block(hidden * 2)
    records = hook.measurements[0]
    assert len(records) == 2
    for record in records:
        assert record["shape"] == [1, 2]
        torch.testing.assert_close(
            torch.tensor(record["semantic_energy"]),
            torch.tensor(record["applied_energy"]),
            atol=1e-5,
            rtol=1e-5,
        )
