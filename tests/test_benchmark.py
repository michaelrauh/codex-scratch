from pathlib import Path

import pandas as pd
import yaml

from jspace_study.benchmark import (
    aggregate_results,
    classify,
    model_directory,
    relative_windows,
    summarize,
)


def test_classify_has_three_exclusive_outcomes() -> None:
    assert classify(" Beijing", "Beijing", "Paris") == "expected_target"
    assert classify("Paris", "Beijing", "Paris") == "original_source"
    assert classify("London", "Beijing", "Paris") == "other_wrong"


def test_relative_windows_are_contiguous_and_middle() -> None:
    windows = relative_windows(36, [[0.25, 0.45], [0.55, 0.75]])
    assert all(window == list(range(window[0], window[-1] + 1)) for window in windows)
    assert all(0 not in window and 35 not in window for window in windows)


def test_tuning_and_evaluation_are_separate_and_evaluation_has_headroom() -> None:
    tuning = yaml.safe_load(Path("data/country_tuning.yaml").read_text())
    evaluation = yaml.safe_load(Path("data/country_eval.yaml").read_text())
    tuning_entities = {row[key] for row in tuning for key in ("source_entity", "target_entity")}
    evaluation_entities = {
        row[key] for row in evaluation for key in ("source_entity", "target_entity")
    }
    assert tuning_entities.isdisjoint(evaluation_entities)
    assert len(evaluation) >= 60


def test_summary_writes_requested_metrics(tmp_path: Path) -> None:
    trials = pd.DataFrame(
        [
            {
                "model": model,
                "parameters": index + 1,
                "eligible": clean,
                "swap_success": success,
                "source_retention": not success,
                "other_wrong": False,
            }
            for index, (model, clean, success) in enumerate(
                [("small", True, False), ("medium", True, False), ("large", True, True)]
            )
        ]
    )
    summary = summarize(trials, tmp_path, seed=7, bootstraps=20)
    assert set(summary.columns) >= {
        "clean_capability_pct",
        "clean_eligible_trials",
        "swap_success_pct",
        "source_retention_pct",
        "other_wrong_pct",
    }
    assert (tmp_path / "swap_success_by_model.svg").is_file()
    assert (tmp_path / "clean_vs_swap_success.svg").is_file()


def test_model_directory_is_stable() -> None:
    assert model_directory(Path("results"), "Qwen/Qwen3-8B") == Path("results/Qwen--Qwen3-8B")


def test_aggregate_combines_only_completed_non_smoke_models(tmp_path: Path) -> None:
    for index, (name, smoke) in enumerate((("one", False), ("two", False), ("smoke", True))):
        directory = tmp_path / name
        directory.mkdir()
        (directory / "completed.json").write_text(f'{{"smoke": {str(smoke).lower()}}}')
        pd.DataFrame(
            [
                {
                    "model": name,
                    "parameters": index + 1,
                    "clean_capability_pct": 100,
                    "swap_success_pct": 50,
                    "swap_success_ci_low": 25,
                    "swap_success_ci_high": 75,
                }
            ]
        ).to_csv(directory / "model_summary.csv", index=False)
    result = aggregate_results(tmp_path)
    assert pd.read_csv(result).model.tolist() == ["one", "two"]
