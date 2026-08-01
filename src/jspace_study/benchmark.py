"""The fixed cross-model semantic-swap benchmark."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from scipy import stats

from .projection import swap_pair

PROMPTS = {
    "capital": "The capital of {entity} is",
    "currency": "The currency of {entity} is",
    "language": "The main language of {entity} is",
    "continent": "{entity} is in the continent of",
}


def classify(answer: str, target: str, source: str) -> str:
    """Classify a generated answer, preferring the target if strings overlap."""
    normalized = " ".join(answer.strip().lower().split())
    if normalized.startswith(target.lower()):
        return "expected_target"
    if normalized.startswith(source.lower()):
        return "original_source"
    return "other_wrong"


def _directions(
    model: Any,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    answer_ids: list[tuple[int, int]],
) -> torch.Tensor:
    """Compute batched exact last-block Jacobian-lens directions for two answer logits."""
    captured: list[torch.Tensor] = []

    def capture(_module: Any, _inputs: Any, output: Any) -> Any:
        hidden = output if torch.is_tensor(output) else output[0]
        hidden.retain_grad()
        captured.append(hidden)
        return output

    handle = model.transformer.h[-1].register_forward_hook(capture)
    try:
        logits = model(input_ids, attention_mask=attention_mask).logits[:, -1]
        vectors = []
        for index in range(2):
            model.zero_grad(set_to_none=True)
            selected = logits[
                torch.arange(len(answer_ids)),
                torch.tensor([tokens[index] for tokens in answer_ids]),
            ]
            selected.sum().backward(retain_graph=index == 0)
            vectors.append(captured[0].grad[:, -1].detach().float().clone())
            captured[0].grad.zero_()
    finally:
        handle.remove()
    return torch.stack(
        [vector / vector.norm(dim=-1, keepdim=True).clamp_min(1e-12) for vector in vectors],
        dim=-1,
    )


def _answer_token(tokenizer: Any, answer: str) -> int:
    tokens = tokenizer.encode(f" {answer}", add_special_tokens=False)
    if len(tokens) != 1:
        raise ValueError(f"benchmark answer must be one token for this tokenizer: {answer!r}")
    return int(tokens[0])


def run_model(spec: dict[str, Any], pairs: list[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    """Run all fixed trials for one model, loading one checkpoint at a time."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(seed)
    tokenizer = AutoTokenizer.from_pretrained(spec["id"], revision=spec.get("revision", "main"))
    model = AutoModelForCausalLM.from_pretrained(spec["id"], revision=spec.get("revision", "main"))
    model.eval()
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    items = []
    for pair_index, pair in enumerate(pairs):
        for relation, facts in pair["facts"].items():
            prompt = PROMPTS[relation].format(entity=pair["source"])
            source_token = _answer_token(tokenizer, facts["source"])
            target_token = _answer_token(tokenizer, facts["target"])
            items.append((pair_index, pair, relation, facts, prompt, source_token, target_token))

    encoded = tokenizer([item[4] for item in items], padding=True, return_tensors="pt")
    ids, mask = encoded.input_ids, encoded.attention_mask
    with torch.inference_mode():
        clean_tokens = model(ids, attention_mask=mask).logits[:, -1].argmax(dim=-1)
    directions = _directions(model, ids, mask, [(item[5], item[6]) for item in items])

    def edit(_module: Any, _inputs: Any, output: Any) -> Any:
        hidden = output if torch.is_tensor(output) else output[0]
        changed = hidden.clone()
        for row in range(len(items)):
            swapped, _ = swap_pair(
                hidden[row, -1].float(), directions[row].to(hidden.device), strength=1.0
            )
            changed[row, -1] = swapped.to(hidden.dtype)
        return changed if torch.is_tensor(output) else (changed, *output[1:])

    handle = model.transformer.h[-1].register_forward_hook(edit)
    try:
        with torch.inference_mode():
            swap_tokens = model(ids, attention_mask=mask).logits[:, -1].argmax(dim=-1)
    finally:
        handle.remove()

    rows = []
    for row, item in enumerate(items):
        pair_index, pair, relation, facts, prompt, _, _ = item
        clean_answer = tokenizer.decode([int(clean_tokens[row])]).strip()
        swap_answer = tokenizer.decode([int(swap_tokens[row])]).strip()
        category = classify(swap_answer, facts["target"], facts["source"])
        rows.append(
            {
                "model": spec["label"],
                "model_id": spec["id"],
                "parameters": spec["parameters"],
                "trial_id": f"pair-{pair_index}-{relation}",
                "source_entity": pair["source"],
                "target_entity": pair["target"],
                "relation": relation,
                "prompt": prompt,
                "expected_source": facts["source"],
                "expected_target": facts["target"],
                "clean_answer": clean_answer,
                "swap_answer": swap_answer,
                "classification": category,
                "clean_correct": classify(clean_answer, facts["source"], facts["target"])
                == "expected_target",
                "swap_success": category == "expected_target",
                "source_retention": category == "original_source",
                "other_wrong": category == "other_wrong",
                "intervention_layer": len(model.transformer.h) - 1,
                "intervention_strength": 1.0,
            }
        )
    del model
    return rows


def summarize(
    trials: pd.DataFrame, output: Path, seed: int, bootstraps: int = 2000
) -> pd.DataFrame:
    """Write model metrics, trial bootstrap intervals, exploratory correlations, and charts."""
    rng = np.random.default_rng(seed)
    rows = []
    for model, frame in trials.groupby("model", sort=False):
        success = frame.swap_success.astype(float).to_numpy()
        samples = np.array(
            [rng.choice(success, len(success), replace=True).mean() for _ in range(bootstraps)]
        )
        rows.append(
            {
                "model": model,
                "parameters": int(frame.parameters.iloc[0]),
                "trials": len(frame),
                "clean_accuracy_pct": 100 * frame.clean_correct.mean(),
                "swap_success_pct": 100 * success.mean(),
                "swap_success_ci_low": 100 * np.quantile(samples, 0.025),
                "swap_success_ci_high": 100 * np.quantile(samples, 0.975),
                "source_retention_pct": 100 * frame.source_retention.mean(),
                "other_wrong_pct": 100 * frame.other_wrong.mean(),
            }
        )
    summary = pd.DataFrame(rows)
    summary.to_csv(output / "model_summary.csv", index=False)
    pearson = stats.pearsonr(summary.clean_accuracy_pct, summary.swap_success_pct)
    spearman = stats.spearmanr(summary.clean_accuracy_pct, summary.swap_success_pct)
    correlations = {
        "label": "exploratory",
        "n_models": len(summary),
        "pearson": {"r": float(pearson.statistic), "p": float(pearson.pvalue)},
        "spearman": {"rho": float(spearman.statistic), "p": float(spearman.pvalue)},
    }
    (output / "exploratory_correlations.json").write_text(json.dumps(correlations, indent=2) + "\n")

    error = np.vstack(
        (
            summary.swap_success_pct - summary.swap_success_ci_low,
            summary.swap_success_ci_high - summary.swap_success_pct,
        )
    )
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(summary.model, summary.swap_success_pct, yerr=error, capsize=4)
    ax.set(ylabel="Swap success (%)", title="Fixed J-space semantic-swap benchmark", ylim=(0, 100))
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(output / "swap_success_by_model.svg")
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(summary.clean_accuracy_pct, summary.swap_success_pct)
    for row in summary.itertuples():
        ax.annotate(
            row.model,
            (row.clean_accuracy_pct, row.swap_success_pct),
            xytext=(4, 4),
            textcoords="offset points",
        )
    ax.set(
        xlabel="Clean accuracy (%)",
        ylabel="Swap success (%)",
        title="Clean capability vs. swap success (exploratory)",
    )
    fig.tight_layout()
    fig.savefig(output / "clean_vs_swap_success.svg")
    plt.close(fig)
    return summary


def run_benchmark(config: dict[str, Any], output: Path) -> Path:
    """Run the identical benchmark and require three completed checkpoints."""
    output.mkdir(parents=True, exist_ok=True)
    pairs = yaml.safe_load(Path(config["benchmark"]["pairs"]).read_text())
    rows: list[dict[str, Any]] = []
    for spec in config["models"]:
        rows.extend(run_model(spec, pairs, int(config["seed"])))
    trials = pd.DataFrame(rows)
    if trials.model.nunique() < 3:
        raise RuntimeError("the comparison requires at least three completed models")
    trials.to_csv(output / "trials.csv", index=False)
    trials.to_parquet(output / "trials.parquet", index=False)
    summarize(trials, output, int(config["seed"]), int(config["benchmark"]["bootstraps"]))
    return output / "model_summary.csv"
