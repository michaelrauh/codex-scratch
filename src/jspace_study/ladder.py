"""Run every configured model, preserving measured successes and explicit failures."""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Any

from .milestone import run_milestone


def run_ladder(config: dict[str, Any], output: Path) -> Path:
    output.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for spec in config["models"]:
        started = time.perf_counter()
        row: dict[str, Any] = {
            "model": spec["label"],
            "model_id": spec["id"],
            "parameter_count": spec["parameters"],
        }
        try:
            model_output = output / spec["cache_name"]
            result_path = run_milestone(config, model_output, model_id=spec["id"])
            result = json.loads(result_path.read_text())
            clean = result["conditions"]["clean"]["selected_probabilities"]["target"]
            semantic = result["conditions"]["semantic"]["selected_probabilities"]["target"]
            random = result["conditions"]["matched_random"]["selected_probabilities"]["target"]
            row.update(
                semantic_probability_change=semantic - clean,
                matched_random_probability_change=random - clean,
                effect_size=semantic - random,
                success=True,
                error="",
            )
        except Exception as exc:  # noqa: BLE001 - the ladder must record and continue all failures
            row.update(
                semantic_probability_change=None,
                matched_random_probability_change=None,
                effect_size=None,
                success=False,
                error=f"{type(exc).__name__}: {exc}",
            )
        row["runtime_seconds"] = time.perf_counter() - started
        rows.append(row)

    json_path = output / "ladder_results.json"
    json_path.write_text(json.dumps(rows, indent=2) + "\n")
    with (output / "ladder_results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return json_path
