"""Qwen3 capability ladder using prefitted averaged Jacobian lenses."""

from __future__ import annotations

import gc
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml

from .intervention import SwapHooks


def cleanup_memory() -> None:
    """Release per-trial tensors on CPU, MPS, and CUDA without changing computation."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


def normalize(text: str) -> str:
    return " ".join(text.strip().lower().split()).rstrip(".!,;:")


def answers_fact(answer: str, fact: str) -> bool:
    """Score a short greedy answer by an exact, normalized leading fact string."""
    return normalize(answer).startswith(normalize(fact))


def classify(answer: str, target: str, source: str) -> str:
    if answers_fact(answer, target):
        return "expected_target"
    if answers_fact(answer, source):
        return "original_source"
    return "other_wrong"


def relative_windows(n_layers: int, definitions: list[list[float]]) -> list[list[int]]:
    """Convert preregistered relative bounds to distinct contiguous middle windows."""
    windows = []
    for low, high in definitions:
        start = max(1, min(n_layers - 2, round(low * (n_layers - 1))))
        stop = max(start + 1, min(n_layers - 1, round(high * (n_layers - 1)) + 1))
        window = list(range(start, stop))
        if window not in windows:
            windows.append(window)
    return windows


def _token_direction(lens: Any, weight: torch.Tensor, token: int, layer: int) -> torch.Tensor:
    direction = lens.jacobians[layer].to(weight.device).float().T @ weight[token].float()
    return direction / direction.norm().clamp_min(1e-12)


def _generate(model: Any, tokenizer: Any, prompt: str, max_new_tokens: int) -> str:
    encoded = tokenizer(prompt, return_tensors="pt")
    device = next(model.parameters()).device
    encoded = {key: value.to(device) for key, value in encoded.items()}
    with torch.inference_mode():
        output = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(output[0, encoded["input_ids"].shape[1] :], skip_special_tokens=True)


def _directions(
    lens: Any, model: Any, tokenizer: Any, source: str, target: str, layers: list[int]
) -> dict[int, torch.Tensor]:
    source_ids = tokenizer.encode(f" {source}", add_special_tokens=False)
    target_ids = tokenizer.encode(f" {target}", add_special_tokens=False)
    if not source_ids or not target_ids:
        raise ValueError("facts must tokenize to at least one token")
    weight = model.get_output_embeddings().weight
    return {
        layer: torch.stack(
            [
                _token_direction(lens, weight, source_ids[0], layer),
                _token_direction(lens, weight, target_ids[0], layer),
            ],
            dim=1,
        )
        for layer in layers
    }


def _eligible(
    model: Any, tokenizer: Any, trial: dict[str, Any], max_new_tokens: int
) -> dict[str, Any]:
    source_answer = _generate(model, tokenizer, trial["source_prompt"], max_new_tokens)
    target_answer = _generate(model, tokenizer, trial["target_prompt"], max_new_tokens)
    source_ok = answers_fact(source_answer, trial["source_fact"])
    target_ok = answers_fact(target_answer, trial["target_fact"])
    return {
        "source_clean_answer": source_answer,
        "target_clean_answer": target_answer,
        "source_clean_correct": source_ok,
        "target_clean_correct": target_ok,
        "eligible": source_ok and target_ok,
        "exclusion_reason": ""
        if source_ok and target_ok
        else ";".join(
            name
            for name, ok in (
                ("source_clean_incorrect", source_ok),
                ("target_clean_incorrect", target_ok),
            )
            if not ok
        ),
    }


def run_trial(
    model: Any,
    wrapped: Any,
    lens: Any,
    tokenizer: Any,
    trial: dict[str, Any],
    layers: list[int],
    max_new_tokens: int,
) -> str:
    directions = _directions(
        lens, model, tokenizer, trial["source_fact"], trial["target_fact"], layers
    )
    with SwapHooks(wrapped.layers, directions):
        return _generate(model, tokenizer, trial["source_prompt"], max_new_tokens)


def _load_model(spec: dict[str, Any]) -> tuple[Any, Any]:
    """Load one model with the native accelerator's memory-safe inference dtype."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(spec["id"], revision=spec.get("revision", "main"))
    factory: Any = AutoModelForCausalLM
    common = {
        "revision": spec.get("revision", "main"),
        "low_cpu_mem_usage": True,
    }
    if torch.backends.mps.is_available():
        model = factory.from_pretrained(
            spec["id"], torch_dtype=torch.float16, **common
        ).to("mps")
    elif torch.cuda.is_available():
        model = factory.from_pretrained(
            spec["id"], torch_dtype=torch.bfloat16, device_map="auto", **common
        )
    else:
        model = factory.from_pretrained(
            spec["id"], torch_dtype=torch.float32, **common
        )
    return tokenizer, model.eval()


def _write_trials(rows: list[dict[str, Any]], path: Path) -> None:
    """Atomically checkpoint evaluation progress after every trial."""
    temporary = path.with_suffix(".tmp")
    pd.DataFrame(rows).to_csv(temporary, index=False)
    temporary.replace(path)


def run_model(
    spec: dict[str, Any], config: dict[str, Any], output: Path, *, smoke: bool = False
) -> Path:
    """Tune and run one configured model, resuming its per-model evaluation rows."""
    import jlens

    output.mkdir(parents=True, exist_ok=True)
    tokenizer, model = _load_model(spec)
    wrapped = jlens.from_hf(model, tokenizer)
    lens = jlens.JacobianLens.from_pretrained(config["lens_repository"], filename=spec["lens"])
    max_tokens = int(config["benchmark"]["max_new_tokens"])
    tuning = yaml.safe_load(Path(config["data"]["tuning"]).read_text())
    evaluation = yaml.safe_load(Path(config["data"]["evaluation"]).read_text())
    windows = relative_windows(len(wrapped.layers), config["benchmark"]["relative_windows"])
    tuning_path = output / "window_tuning.json"
    if tuning_path.exists():
        tuning_record = json.loads(tuning_path.read_text())
        selected = tuning_record["selected_window"]
    else:
        tuning_eligible = []
        for trial in tuning:
            screening = _eligible(model, tokenizer, trial, max_tokens)
            if screening["eligible"]:
                tuning_eligible.append(trial)
            cleanup_memory()
        scores = []
        for window in windows:
            successes = 0
            for trial in tuning_eligible:
                successes += (
                    classify(
                        run_trial(model, wrapped, lens, tokenizer, trial, window, max_tokens),
                        trial["target_fact"],
                        trial["source_fact"],
                    )
                    == "expected_target"
                )
                cleanup_memory()
            scores.append(successes / len(tuning_eligible) if tuning_eligible else 0.0)
        selected = windows[int(np.argmax(scores))]
        tuning_record = {
            "model": spec["label"],
            "tuning_trials": len(tuning),
            "tuning_clean_eligible": len(tuning_eligible),
            "candidate_windows": windows,
            "candidate_success_rates": scores,
            "selected_window": selected,
        }
        tuning_path.write_text(json.dumps(tuning_record, indent=2) + "\n")

    trials_path = output / "trials.csv"
    rows = pd.read_csv(trials_path).to_dict("records") if trials_path.exists() else []
    completed_ids = {row["trial_id"] for row in rows}
    eligible_count = sum(bool(row["eligible"]) for row in rows)
    for trial in evaluation:
        if trial["trial_id"] in completed_ids or (smoke and eligible_count >= 5):
            continue
        screening = _eligible(model, tokenizer, trial, max_tokens)
        row = {
            "model": spec["label"],
            "model_id": spec["id"],
            "parameters": spec["parameters"],
            **trial,
            **screening,
            "window_start": selected[0],
            "window_end": selected[-1],
        }
        if screening["eligible"]:
            answer = run_trial(model, wrapped, lens, tokenizer, trial, selected, max_tokens)
            category = classify(answer, trial["target_fact"], trial["source_fact"])
            row.update(
                swap_answer=answer,
                classification=category,
                swap_success=category == "expected_target",
                source_retention=category == "original_source",
                other_wrong=category == "other_wrong",
            )
        else:
            row.update(
                swap_answer="",
                classification="excluded",
                swap_success=False,
                source_retention=False,
                other_wrong=False,
            )
        rows.append(row)
        eligible_count += int(screening["eligible"])
        _write_trials(rows, trials_path)
        cleanup_memory()
    minimum = 5 if smoke else int(config["benchmark"]["minimum_clean_eligible"])
    if eligible_count < minimum:
        raise RuntimeError(
            f"{spec['label']} requires {minimum} clean-eligible trials; got {eligible_count}"
        )
    frame = pd.DataFrame(rows)
    frame.to_parquet(output / "trials.parquet", index=False)
    summarize(frame, output, int(config["seed"]), int(config["benchmark"]["bootstraps"]))
    (output / "completed.json").write_text(
        json.dumps(
            {
                "model": spec["label"],
                "model_id": spec["id"],
                "smoke": smoke,
                "clean_eligible_trials": eligible_count,
            },
            indent=2,
        )
        + "\n"
    )
    del wrapped, lens, model
    cleanup_memory()
    return output / "model_summary.csv"


def _bootstrap(values: np.ndarray, rng: np.random.Generator, count: int) -> tuple[float, float]:
    samples = np.array([rng.choice(values, len(values), replace=True).mean() for _ in range(count)])
    return tuple(100 * np.quantile(samples, [0.025, 0.975]))  # type: ignore[return-value]


def summarize(
    trials: pd.DataFrame, output: Path, seed: int, bootstraps: int = 2000
) -> pd.DataFrame:
    """Summarize clean-eligible evaluation rows and emit both requested charts."""
    output.mkdir(parents=True, exist_ok=True)
    rng, rows = np.random.default_rng(seed), []
    for model, all_rows in trials.groupby("model", sort=False):
        frame = all_rows[all_rows.eligible]
        if frame.empty:
            raise RuntimeError(f"{model} has no clean-eligible evaluation trials")
        row = {
            "model": model,
            "parameters": int(frame.parameters.iloc[0]),
            "evaluation_trials": len(all_rows),
            "clean_eligible_trials": len(frame),
            "clean_capability_pct": 100 * len(frame) / len(all_rows),
        }
        for name in ("swap_success", "source_retention", "other_wrong"):
            values = frame[name].astype(float).to_numpy()
            low, high = _bootstrap(values, rng, bootstraps)
            row.update(
                {f"{name}_pct": 100 * values.mean(), f"{name}_ci_low": low, f"{name}_ci_high": high}
            )
        rows.append(row)
    summary = pd.DataFrame(rows)
    summary.to_csv(output / "model_summary.csv", index=False)
    write_charts(summary, output)
    return summary


def write_charts(summary: pd.DataFrame, output: Path) -> None:
    """Render the same preregistered charts for one model or an aggregate."""
    error = np.vstack(
        (
            summary.swap_success_pct - summary.swap_success_ci_low,
            summary.swap_success_ci_high - summary.swap_success_pct,
        )
    )
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(summary.model, summary.swap_success_pct, yerr=error, capsize=4)
    ax.set(ylabel="Swap success (%)", title="Qwen3 J-space capability ladder", ylim=(0, 100))
    fig.tight_layout()
    fig.savefig(output / "swap_success_by_model.svg")
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(summary.clean_capability_pct, summary.swap_success_pct)
    for row in summary.itertuples():
        ax.annotate(
            row.model,
            (row.clean_capability_pct, row.swap_success_pct),
            xytext=(4, 4),
            textcoords="offset points",
        )
    ax.set(
        xlabel="Clean capability (% evaluation trials eligible)",
        ylabel="Swap success (%)",
        title="Clean capability vs. swap success",
    )
    fig.tight_layout()
    fig.savefig(output / "clean_vs_swap_success.svg")
    plt.close(fig)


def model_directory(root: Path, model_id: str) -> Path:
    return root / model_id.replace("/", "--")


def run_benchmark(
    config: dict[str, Any], output: Path, model_id: str, *, smoke: bool = False
) -> Path:
    """Run exactly one configured checkpoint in its resumable output directory."""
    spec = next((item for item in config["models"] if item["id"] == model_id), None)
    if spec is None:
        configured = ", ".join(item["id"] for item in config["models"])
        raise ValueError(f"model is not configured: {model_id}; choose one of: {configured}")
    return run_model(spec, config, model_directory(output, model_id), smoke=smoke)


def aggregate_results(output: Path) -> Path:
    """Combine independently completed non-smoke models without recomputing scores."""
    rows = []
    for marker in sorted(output.glob("*/completed.json")):
        completion = json.loads(marker.read_text())
        if completion.get("smoke"):
            continue
        summary_path = marker.parent / "model_summary.csv"
        if summary_path.is_file():
            rows.append(pd.read_csv(summary_path))
    if not rows:
        raise RuntimeError(f"no completed non-smoke model outputs found below {output}")
    summary = pd.concat(rows, ignore_index=True).sort_values("parameters")
    aggregate = output / "aggregate"
    aggregate.mkdir(parents=True, exist_ok=True)
    summary.to_csv(aggregate / "model_summary.csv", index=False)
    write_charts(summary, aggregate)
    return aggregate / "model_summary.csv"
