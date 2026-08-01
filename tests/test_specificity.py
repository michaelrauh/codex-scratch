import pandas as pd
import pytest
import torch

from jspace_study.specificity import choose_probability_matched_control, summarize_specificity


def test_probability_control_chooses_closest_eligible_token() -> None:
    log_probs = torch.tensor([-5.0, -2.0, -2.2, -3.0])
    assert choose_probability_matched_control(log_probs, 1, [("near", 2), ("far", 3)]) == (
        "near",
        2,
    )


def test_specificity_summary_is_paired() -> None:
    frame = pd.DataFrame(
        {
            "semantic_target_delta": [4.0, 6.0],
            "control_target_delta": [1.0, 2.0],
            "semantic_control_delta": [0.0, 1.0],
            "clean_log_probability_mismatch": [0.1, -0.2],
            "relative_energy_mismatch": [1e-7, 2e-7],
        }
    )
    result = summarize_specificity(frame, 7)
    assert result["semantic_minus_control_nominated_gain_mean"] == pytest.approx(3.5)
    assert result["semantic_target_minus_off_target_gain_mean"] == pytest.approx(4.5)
    assert result["maximum_relative_energy_mismatch"] == pytest.approx(2e-7)
