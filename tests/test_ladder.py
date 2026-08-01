import csv
import json
from pathlib import Path

import pytest

from jspace_study import ladder


def test_ladder_preserves_success_and_continues_after_failure(
    monkeypatch, tmp_path: Path
) -> None:
    config = {
        "models": [
            {"label": "works", "id": "test/works", "cache_name": "works", "parameters": 1},
            {"label": "fails", "id": "test/fails", "cache_name": "fails", "parameters": 2},
        ]
    }

    def fake_run(_config, output, max_new_tokens=4, model_id=None):
        if model_id == "test/fails":
            raise RuntimeError("measured failure")
        output.mkdir(parents=True)
        result = {
            "conditions": {
                "clean": {"selected_probabilities": {"target": 0.1}},
                "semantic": {"selected_probabilities": {"target": 0.4}},
                "matched_random": {"selected_probabilities": {"target": 0.2}},
            }
        }
        path = output / "milestone.json"
        path.write_text(json.dumps(result))
        return path

    monkeypatch.setattr(ladder, "run_milestone", fake_run)
    result_path = ladder.run_ladder(config, tmp_path)
    rows = json.loads(result_path.read_text())
    assert rows[0]["success"] is True
    assert rows[0]["semantic_probability_change"] == pytest.approx(0.3)
    assert rows[0]["matched_random_probability_change"] == pytest.approx(0.1)
    assert rows[0]["effect_size"] == pytest.approx(0.2)
    assert rows[1]["success"] is False
    assert rows[1]["error"] == "RuntimeError: measured failure"
    with (tmp_path / "ladder_results.csv").open() as handle:
        assert len(list(csv.DictReader(handle))) == 2
