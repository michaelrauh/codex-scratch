"""Held-out test of whether the GPT-2 effect is specific to semantic targets."""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any, cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy import stats

from .one_step import (
    _bootstrap_ci,
    _capture_clean,
    _direction,
    _forward_with_deltas,
    _rank,
    _split,
    screen_country_pairs,
)
from .projection import swap_pair


def choose_probability_matched_control(
    clean_log_probs: torch.Tensor,
    target_token: int,
    candidates: list[tuple[str, int]],
) -> tuple[str, int]:
    """Choose the preregistered non-capital token nearest the target in clean log probability."""
    eligible = [(word, token) for word, token in candidates if token != target_token]
    if not eligible:
        raise ValueError("no eligible specificity control token")
    target = float(clean_log_probs[target_token])
    return min(eligible, key=lambda item: abs(float(clean_log_probs[item[1]]) - target))


def summarize_specificity(frame: pd.DataFrame, seed: int) -> dict[str, Any]:
    """Compute paired, prompt-bootstrap specificity estimates."""
    nominated = frame.semantic_target_delta.to_numpy() - frame.control_target_delta.to_numpy()
    off_target = frame.semantic_target_delta.to_numpy() - frame.semantic_control_delta.to_numpy()
    return {
        "n_evaluation_pairs": len(frame),
        "semantic_target_log_probability_delta_mean": float(frame.semantic_target_delta.mean()),
        "probability_matched_control_target_delta_mean": float(frame.control_target_delta.mean()),
        "semantic_minus_control_nominated_gain_mean": float(nominated.mean()),
        "semantic_minus_control_nominated_gain_ci95": _bootstrap_ci(nominated, seed),
        "semantic_minus_control_nominated_gain_paired_ttest_p": float(
            stats.ttest_rel(frame.semantic_target_delta, frame.control_target_delta).pvalue
        ),
        "semantic_target_minus_off_target_gain_mean": float(off_target.mean()),
        "semantic_target_minus_off_target_gain_ci95": _bootstrap_ci(off_target, seed + 1),
        "semantic_target_minus_off_target_gain_paired_ttest_p": float(
            stats.ttest_rel(frame.semantic_target_delta, frame.semantic_control_delta).pvalue
        ),
        "semantic_control_off_target_delta_mean": float(frame.semantic_control_delta.mean()),
        "mean_absolute_clean_log_probability_mismatch": float(
            frame.clean_log_probability_mismatch.abs().mean()
        ),
        "maximum_relative_energy_mismatch": float(frame.relative_energy_mismatch.max()),
    }


def run_specificity_study(config: dict[str, Any], output: Path) -> Path:
    """Run a frozen-layer comparison with energy- and clean-probability-matched noun targets."""
    import jlens
    from transformers import AutoModelForCausalLM, AutoTokenizer

    spec = next(item for item in config["models"] if item["id"] == "openai-community/gpt2")
    model_path = Path(config["cache_dir"]) / "models" / spec["cache_name"]
    lens_root = Path(config["cache_dir"]) / "lenses" / config["lens_repository"].replace("/", "--")
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True).eval()
    wrapped = jlens.from_hf(model, tokenizer)
    lens = jlens.JacobianLens.from_pretrained(str(lens_root), filename=spec["lens"])
    included, excluded = screen_country_pairs(model, tokenizer, Path("data/country_facts.yaml"))
    evaluation = [pair for pair in included if _split(pair) == "evaluation"]
    evaluation = evaluation[: int(config["specificity"]["evaluation_pairs"])]
    if len(evaluation) != int(config["specificity"]["evaluation_pairs"]):
        raise RuntimeError("insufficient held-out pairs for specificity experiment")
    capital_tokens = {int(pair["source_token"]) for pair in included} | {
        int(pair["target_token"]) for pair in included
    }
    candidates = [
        (decoded.strip(), token)
        for token in range(len(tokenizer))
        if token not in capital_tokens
        and re.fullmatch(r" [A-Za-z]{3,}", decoded := cast(str, tokenizer.decode([token])))
    ]
    if len(candidates) < 1000:
        raise RuntimeError("fewer than 1,000 alphabetic non-capital control tokens are available")

    layers = tuple(int(x) for x in config["specificity"]["selected_layers"])
    weight = model.get_output_embeddings().weight.detach().cpu().float()
    rows = []
    started = time.perf_counter()
    for pair in evaluation:
        input_ids = tokenizer(pair["prompt"], return_tensors="pt").input_ids
        clean_logits, activations = _capture_clean(model, wrapped.layers, input_ids)
        clean_lp = clean_logits.log_softmax(-1)
        control_word, control_token = choose_probability_matched_control(
            clean_lp, int(pair["target_token"]), candidates
        )
        semantic_deltas = {}
        control_deltas = {}
        semantic_energy = control_energy = 0.0
        for layer in layers:
            hidden = activations[layer][:, -1, :].cpu()
            source_direction = _direction(lens, weight, int(pair["source_token"]), layer)
            semantic_directions = torch.stack(
                [source_direction, _direction(lens, weight, int(pair["target_token"]), layer)], dim=1
            )
            control_directions = torch.stack(
                [source_direction, _direction(lens, weight, control_token, layer)], dim=1
            )
            _, semantic = swap_pair(hidden, semantic_directions)
            _, control = swap_pair(hidden, control_directions)
            control = control * (semantic.norm() / control.norm().clamp_min(1e-12))
            semantic_deltas[layer] = semantic
            control_deltas[layer] = control
            semantic_energy += float(semantic.square().sum())
            control_energy += float(control.square().sum())
        semantic_logits = _forward_with_deltas(model, wrapped.layers, input_ids, semantic_deltas)
        control_logits = _forward_with_deltas(model, wrapped.layers, input_ids, control_deltas)
        semantic_lp = semantic_logits.log_softmax(-1)
        control_lp = control_logits.log_softmax(-1)
        target = int(pair["target_token"])
        rows.append(
            {
                "source_entity": pair["source_entity"],
                "target_entity": pair["target_entity"],
                "target_answer": pair["target_answer"],
                "control_word": control_word,
                "target_token": target,
                "control_token": control_token,
                "target_clean_rank": _rank(clean_logits, target),
                "control_clean_rank": _rank(clean_logits, control_token),
                "target_clean_log_probability": float(clean_lp[target]),
                "control_clean_log_probability": float(clean_lp[control_token]),
                "clean_log_probability_mismatch": float(clean_lp[target] - clean_lp[control_token]),
                "semantic_target_delta": float(semantic_lp[target] - clean_lp[target]),
                "semantic_control_delta": float(semantic_lp[control_token] - clean_lp[control_token]),
                "control_target_delta": float(control_lp[control_token] - clean_lp[control_token]),
                "control_semantic_target_delta": float(control_lp[target] - clean_lp[target]),
                "semantic_energy": semantic_energy,
                "control_energy": control_energy,
                "relative_energy_mismatch": abs(semantic_energy - control_energy)
                / max(semantic_energy, 1e-12),
            }
        )
    output.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "pair_results.csv", index=False)
    summary = summarize_specificity(frame, int(config["seed"]))
    summary.update(
        {
            "model": spec["id"],
            "model_resolved_revision": "607a30d783dfa663caf39e06633721c8d4cfcd7e",
            "lens_repository_resolved_revision": "a4114d7752d11eb546e6cf372213d7e75526d3a1",
            "selected_layers_frozen_from_one_step": list(layers),
            "control": "nearest-clean-log-probability alphabetic non-capital token; per-layer energy matched",
            "n_screened_pairs": len(included) + len(excluded),
            "n_eligible_pairs": len(included),
            "runtime_seconds": time.perf_counter() - started,
        }
    )
    path = output / "summary.json"
    path.write_text(json.dumps(summary, indent=2) + "\n")
    fig, axis = plt.subplots(figsize=(6.4, 4.5))
    values = [frame.semantic_target_delta.mean(), frame.control_target_delta.mean()]
    cis = [
        _bootstrap_ci(frame.semantic_target_delta.to_numpy(), int(config["seed"]) + 2),
        _bootstrap_ci(frame.control_target_delta.to_numpy(), int(config["seed"]) + 3),
    ]
    errors = np.array([[v - ci[0] for v, ci in zip(values, cis, strict=True)], [ci[1] - v for v, ci in zip(values, cis, strict=True)]])
    axis.bar(["Capital target", "Matched noun target"], values, yerr=errors, capsize=5)
    axis.axhline(0, color="black", linewidth=0.8)
    axis.set_ylabel("Nominated-token log-probability change")
    axis.set_title("GPT-2 124M semantic-specificity test (95% bootstrap CI)")
    fig.tight_layout()
    chart_path = output / "specificity.svg"
    fig.savefig(chart_path)
    plt.close(fig)
    chart_path.write_text("\n".join(line.rstrip() for line in chart_path.read_text().splitlines()) + "\n")
    return path
