import json
from pathlib import Path

import pandas as pd

from jspace_study.benchmark import classify, summarize


def test_classify_has_three_exclusive_outcomes() -> None:
    assert classify(" Beijing", "Beijing", "Paris") == "expected_target"
    assert classify("Paris", "Beijing", "Paris") == "original_source"
    assert classify("London", "Beijing", "Paris") == "other_wrong"


def test_summary_writes_requested_metrics(tmp_path: Path) -> None:
    trials = pd.DataFrame(
        [
            {
                "model": model,
                "parameters": index + 1,
                "clean_correct": clean,
                "swap_success": success,
                "source_retention": not success,
                "other_wrong": False,
            }
            for index, (model, clean, success) in enumerate(
                [("small", False, False), ("medium", True, False), ("large", True, True)]
            )
        ]
    )
    summary = summarize(trials, tmp_path, seed=7, bootstraps=20)
    assert set(summary.columns) >= {
        "clean_accuracy_pct",
        "swap_success_pct",
        "source_retention_pct",
        "other_wrong_pct",
    }
    assert (
        json.loads((tmp_path / "exploratory_correlations.json").read_text())["label"]
        == "exploratory"
    )
    assert (tmp_path / "swap_success_by_model.svg").is_file()
    assert (tmp_path / "clean_vs_swap_success.svg").is_file()
