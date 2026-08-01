import pytest
import torch

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


def test_replace_shape_validation() -> None:
    with pytest.raises(ValueError):
        replace_coefficients(torch.ones(2), torch.ones(3, 1), torch.zeros(1))


def test_random_control_is_norm_matched_and_deterministic() -> None:
    delta = torch.tensor([[[3.0, 4.0, 0.0], [0.0, 0.0, 2.0]]])
    first = matched_random_delta(delta, torch.Generator().manual_seed(4))
    second = matched_random_delta(delta, torch.Generator().manual_seed(4))
    torch.testing.assert_close(first.norm(dim=-1), delta.norm(dim=-1))
    torch.testing.assert_close(first, second)
