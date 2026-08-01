"""Teacher-forced one-step causal validation from shared clean activations."""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from scipy import stats
from torch import nn

from .projection import matched_random_delta, swap_pair

CONDITIONS = ("noop", "semantic", "matched_random", "shuffled", "positive_control")


def _prompt(entity: str) -> str:
    return f"Country: {entity}\nCapital:"


def _rank(logits: torch.Tensor, token: int) -> int:
    return int((logits > logits[token]).sum()) + 1


def _direction(lens: Any, weight: torch.Tensor, token: int, layer: int) -> torch.Tensor:
    value = lens.jacobians[layer].float().T @ weight[token].float()
    return value / value.norm().clamp_min(1e-12)


def screen_country_pairs(
    model: Any,
    tokenizer: Any,
    facts_path: Path,
    *,
    clean_rank_max: int = 5,
    minimum_target_probability: float = 1e-6,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Screen every ordered fact pair and retain all model-specific exclusions."""
    facts = yaml.safe_load(facts_path.read_text())
    fact_metrics: dict[str, dict[str, Any]] = {}
    for fact in facts:
        token_ids = tokenizer.encode(f" {fact['answer']}", add_special_tokens=False)
        metric: dict[str, Any] = {**fact, "token_count": len(token_ids)}
        if len(token_ids) == 1:
            inputs = tokenizer(_prompt(fact["entity"]), return_tensors="pt").input_ids
            with torch.inference_mode():
                logits = model(inputs).logits[0, -1].float().cpu()
            token = token_ids[0]
            metric.update(
                token=token,
                rank=_rank(logits, token),
                probability=float(logits.softmax(-1)[token]),
            )
        fact_metrics[fact["entity"]] = metric

    included: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    for source in facts:
        for target in facts:
            if source == target:
                continue
            source_metric = fact_metrics[source["entity"]]
            target_metric = fact_metrics[target["entity"]]
            reasons: list[str] = []
            if source_metric["token_count"] != 1:
                reasons.append(f"source_token_count={source_metric['token_count']}")
            if target_metric["token_count"] != 1:
                reasons.append(f"target_token_count={target_metric['token_count']}")
            if source_metric.get("rank", clean_rank_max + 1) > clean_rank_max:
                reasons.append(f"source_clean_rank={source_metric.get('rank')}>{clean_rank_max}")
            if target_metric.get("rank", clean_rank_max + 1) > clean_rank_max:
                reasons.append(f"target_prompt_rank={target_metric.get('rank')}>{clean_rank_max}")
            target_probability = float(source_metric.get("distribution", {}).get("target", 0.0))
            if not reasons:
                inputs = tokenizer(_prompt(source["entity"]), return_tensors="pt").input_ids
                with torch.inference_mode():
                    logits = model(inputs).logits[0, -1].float().cpu()
                target_probability = float(logits.softmax(-1)[target_metric["token"]])
                if target_probability < minimum_target_probability:
                    reasons.append(
                        f"target_probability={target_probability}<{minimum_target_probability}"
                    )
            record = {
                "source_entity": source["entity"],
                "target_entity": target["entity"],
                "prompt": _prompt(source["entity"]),
                "source_answer": source["answer"],
                "target_answer": target["answer"],
                "source_token": source_metric.get("token"),
                "target_token": target_metric.get("token"),
                "source_clean_rank": source_metric.get("rank"),
                "source_clean_probability": source_metric.get("probability"),
                "target_prompt_rank": target_metric.get("rank"),
                "target_prompt_probability": target_metric.get("probability"),
                "target_clean_probability": target_probability,
            }
            if reasons:
                record["exclusion_reasons"] = reasons
                excluded.append(record)
            else:
                included.append(record)
    return included, excluded


def _split(pair: dict[str, Any]) -> str:
    key = f"{pair['source_entity']}->{pair['target_entity']}".encode()
    return "tuning" if int(hashlib.sha256(key).hexdigest(), 16) % 2 == 0 else "evaluation"


def _shuffled_token(pairs: list[dict[str, Any]], index: int) -> int:
    pair = pairs[index]
    for offset in range(1, len(pairs)):
        candidate = pairs[(index + offset) % len(pairs)]["target_token"]
        if candidate not in (pair["source_token"], pair["target_token"]):
            return int(candidate)
    raise RuntimeError("no distinct shuffled target token is available")


def _capture_clean(model: Any, layers: Sequence[nn.Module], input_ids: torch.Tensor):
    activations: dict[int, torch.Tensor] = {}
    handles = []
    for index, block in enumerate(layers):

        def capture(_module, _inputs, output, layer=index):
            tensor = output if torch.is_tensor(output) else output[0]
            activations[layer] = tensor.detach().float().clone()

        handles.append(block.register_forward_hook(capture))
    with torch.inference_mode():
        logits = model(input_ids).logits[0, -1].float().cpu()
    for handle in handles:
        handle.remove()
    return logits, activations


def _forward_with_deltas(
    model: Any,
    blocks: Sequence[nn.Module],
    input_ids: torch.Tensor,
    deltas: dict[int, torch.Tensor],
) -> torch.Tensor:
    handles = []
    for layer, delta in deltas.items():

        def apply(_module, _inputs, output, value=delta):
            tensor = output if torch.is_tensor(output) else output[0]
            changed = tensor.clone()
            changed[:, -1, :] = changed[:, -1, :] + value.to(changed)
            return changed if torch.is_tensor(output) else (changed, *output[1:])

        handles.append(blocks[layer].register_forward_hook(apply))
    with torch.inference_mode():
        logits = model(input_ids).logits[0, -1].float().cpu()
    for handle in handles:
        handle.remove()
    return logits


def evaluate_pair_window(
    model: Any,
    wrapped: Any,
    tokenizer: Any,
    lens: Any,
    pair: dict[str, Any],
    layers: tuple[int, ...],
    shuffled_token: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Evaluate all controls using deltas calculated from identical clean activations."""
    input_ids = tokenizer(pair["prompt"], return_tensors="pt").input_ids
    clean_logits, activations = _capture_clean(model, wrapped.layers, input_ids)
    weight = model.get_output_embeddings().weight.detach().cpu().float()
    source, target = pair["source_token"], pair["target_token"]
    deltas: dict[str, dict[int, torch.Tensor]] = {name: {} for name in CONDITIONS}
    for layer in layers:
        hidden = activations[layer][:, -1, :].cpu()
        directions = torch.stack(
            [_direction(lens, weight, source, layer), _direction(lens, weight, target, layer)],
            dim=1,
        )
        _, semantic = swap_pair(hidden, directions)
        generator = torch.Generator().manual_seed(seed + layer)
        random = matched_random_delta(semantic, generator)
        shuffled_directions = torch.stack(
            [
                _direction(lens, weight, source, layer),
                _direction(lens, weight, shuffled_token, layer),
            ],
            dim=1,
        )
        _, shuffled = swap_pair(hidden, shuffled_directions)
        shuffled = shuffled * (semantic.norm() / shuffled.norm().clamp_min(1e-12))
        direct = (weight[target] - weight[source]).unsqueeze(0)
        direct = direct * (semantic.norm() / direct.norm().clamp_min(1e-12))
        deltas["noop"][layer] = torch.zeros_like(semantic)
        deltas["semantic"][layer] = semantic
        deltas["matched_random"][layer] = random
        deltas["shuffled"][layer] = shuffled
        deltas["positive_control"][layer] = direct

    clean_log_probs = clean_logits.log_softmax(-1)
    clean_probs = clean_logits.softmax(-1)
    rows = []
    semantic_flat = torch.cat([deltas["semantic"][layer].flatten() for layer in layers])
    for condition in CONDITIONS:
        logits = _forward_with_deltas(model, wrapped.layers, input_ids, deltas[condition])
        log_probs = logits.log_softmax(-1)
        probs = logits.softmax(-1)
        flat = torch.cat([deltas[condition][layer].flatten() for layer in layers])
        cosine = (
            float(torch.nn.functional.cosine_similarity(flat, semantic_flat, dim=0))
            if flat.norm()
            else 0.0
        )
        rows.append(
            {
                "condition": condition,
                "layers": "-".join(map(str, layers)),
                "source_logit_clean": float(clean_logits[source]),
                "target_logit_clean": float(clean_logits[target]),
                "source_logit": float(logits[source]),
                "target_logit": float(logits[target]),
                "source_logit_delta": float(logits[source] - clean_logits[source]),
                "target_logit_delta": float(logits[target] - clean_logits[target]),
                "source_log_probability_delta": float(log_probs[source] - clean_log_probs[source]),
                "target_log_probability_delta": float(log_probs[target] - clean_log_probs[target]),
                "source_probability_clean": float(clean_probs[source]),
                "target_probability_clean": float(clean_probs[target]),
                "source_probability": float(probs[source]),
                "target_probability": float(probs[target]),
                "target_rank_clean": _rank(clean_logits, target),
                "target_rank": _rank(logits, target),
                "target_rank_change": _rank(logits, target) - _rank(clean_logits, target),
                "kl_clean_to_intervened": float(
                    (clean_probs * (clean_log_probs - log_probs)).sum()
                ),
                "intervention_norm": float(flat.norm()),
                "intervention_cosine_with_semantic": cosine,
                "removed_energy": float(flat.square().sum()),
                "per_layer_energy": json.dumps(
                    {str(layer): float(deltas[condition][layer].square().sum()) for layer in layers}
                ),
            }
        )
    return rows


def _bootstrap_ci(values: np.ndarray, seed: int, repetitions: int = 5000) -> list[float]:
    rng = np.random.default_rng(seed)
    means = np.array(
        [rng.choice(values, len(values), replace=True).mean() for _ in range(repetitions)]
    )
    return [float(x) for x in np.quantile(means, [0.025, 0.975])]


def run_one_step_study(config: dict[str, Any], output: Path) -> Path:
    import jlens
    from transformers import AutoModelForCausalLM, AutoTokenizer

    spec = next(item for item in config["models"] if item["id"] == "openai-community/gpt2")
    model_path = Path(config["cache_dir"]) / "models" / spec["cache_name"]
    lens_root = Path(config["cache_dir"]) / "lenses" / config["lens_repository"].replace("/", "--")
    tokenizer: Any = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    factory: Any = AutoModelForCausalLM
    model: Any = factory.from_pretrained(model_path, local_files_only=True).eval()
    wrapped = jlens.from_hf(model, tokenizer)
    lens = jlens.JacobianLens.from_pretrained(str(lens_root), filename=spec["lens"])
    included, excluded = screen_country_pairs(model, tokenizer, Path("data/country_facts.yaml"))
    for pair in included:
        pair["split"] = _split(pair)
    output.mkdir(parents=True, exist_ok=True)
    (output / "screening.json").write_text(
        json.dumps({"included": included, "excluded": excluded}, indent=2) + "\n"
    )
    if len(included) < 50:
        raise RuntimeError(f"only {len(included)} pairs passed screening; at least 50 required")
    tuning = [pair for pair in included if pair["split"] == "tuning"][:30]
    evaluation = [pair for pair in included if pair["split"] == "evaluation"][:60]
    candidate_windows: list[tuple[int, ...]] = [(layer,) for layer in lens.source_layers]
    candidate_windows += [
        tuple(range(start, start + width)) for width in (2, 3) for start in range(11 - width + 1)
    ]
    tuning_rows = []
    started = time.perf_counter()
    for pair_index, pair in enumerate(tuning):
        shuffled = _shuffled_token(tuning, pair_index)
        for layers in candidate_windows:
            for row in evaluate_pair_window(
                model, wrapped, tokenizer, lens, pair, layers, shuffled, config["seed"] + pair_index
            ):
                tuning_rows.append({**pair, **row, "split": "tuning"})
    tuning_frame = pd.DataFrame(tuning_rows)
    pivot = tuning_frame.pivot_table(
        index=["source_entity", "target_entity", "layers"],
        columns="condition",
        values="target_log_probability_delta",
    ).reset_index()
    pivot["selective_effect"] = pivot.semantic - (pivot.matched_random + pivot.shuffled) / 2
    window_summary = pivot.groupby("layers", as_index=False).selective_effect.mean()
    selected = str(window_summary.sort_values("selective_effect", ascending=False).iloc[0].layers)
    selected_layers = tuple(int(x) for x in selected.split("-"))

    evaluation_rows = []
    for pair_index, pair in enumerate(evaluation):
        shuffled = _shuffled_token(evaluation, pair_index)
        for row in evaluate_pair_window(
            model,
            wrapped,
            tokenizer,
            lens,
            pair,
            selected_layers,
            shuffled,
            config["seed"] + pair_index,
        ):
            evaluation_rows.append({**pair, **row, "split": "evaluation"})
    raw = pd.concat([tuning_frame, pd.DataFrame(evaluation_rows)], ignore_index=True)
    raw.to_csv(output / "one_step_raw.csv", index=False)
    pd.DataFrame(evaluation_rows).groupby("condition", as_index=False).mean(
        numeric_only=True
    ).to_csv(output / "heldout_condition_summary.csv", index=False)
    eval_pivot = pd.DataFrame(evaluation_rows).pivot_table(
        index=["source_entity", "target_entity"],
        columns="condition",
        values="target_log_probability_delta",
    )
    contrast = (
        eval_pivot.semantic - (eval_pivot.matched_random + eval_pivot.shuffled) / 2
    ).to_numpy()
    semantic_vs_random = (eval_pivot.semantic - eval_pivot.matched_random).to_numpy()
    semantic_vs_shuffled = (eval_pivot.semantic - eval_pivot.shuffled).to_numpy()
    positive = eval_pivot.positive_control.to_numpy()
    condition_summary = {
        condition: {
            "mean_target_log_probability_delta": float(values.mean()),
            "ci95": _bootstrap_ci(values, config["seed"] + index + 10),
        }
        for index, (condition, values) in enumerate(
            {
                "semantic": eval_pivot.semantic.to_numpy(),
                "matched_random": eval_pivot.matched_random.to_numpy(),
                "shuffled": eval_pivot.shuffled.to_numpy(),
                "positive_control": positive,
            }.items()
        )
    }
    summary = {
        "n_screened_pairs": len(included) + len(excluded),
        "n_eligible_pairs": len(included),
        "n_tuning_pairs": len(tuning),
        "n_evaluation_pairs": len(evaluation),
        "selected_layers": list(selected_layers),
        "selection_metric": "tuning mean semantic target log-probability delta minus mean matched controls",
        "heldout_semantic_selective_effect_mean": float(contrast.mean()),
        "heldout_semantic_selective_effect_ci95": _bootstrap_ci(contrast, config["seed"]),
        "heldout_semantic_vs_zero_ttest_p": float(stats.ttest_1samp(contrast, 0).pvalue),
        "heldout_semantic_minus_random_mean": float(semantic_vs_random.mean()),
        "heldout_semantic_minus_random_ci95": _bootstrap_ci(semantic_vs_random, config["seed"] + 2),
        "heldout_semantic_minus_random_paired_ttest_p": float(
            stats.ttest_rel(eval_pivot.semantic, eval_pivot.matched_random).pvalue
        ),
        "heldout_semantic_minus_shuffled_mean": float(semantic_vs_shuffled.mean()),
        "heldout_semantic_minus_shuffled_ci95": _bootstrap_ci(
            semantic_vs_shuffled, config["seed"] + 3
        ),
        "heldout_semantic_minus_shuffled_paired_ttest_p": float(
            stats.ttest_rel(eval_pivot.semantic, eval_pivot.shuffled).pvalue
        ),
        "heldout_positive_control_mean": float(positive.mean()),
        "heldout_positive_control_ci95": _bootstrap_ci(positive, config["seed"] + 1),
        "heldout_positive_control_vs_zero_ttest_p": float(stats.ttest_1samp(positive, 0).pvalue),
        "heldout_condition_summary": condition_summary,
        "runtime_seconds": time.perf_counter() - started,
    }
    path = output / "summary.json"
    path.write_text(json.dumps(summary, indent=2) + "\n")
    heat = tuning_frame[tuning_frame.condition == "semantic"].pivot_table(
        index="layers", values=["target_logit_delta", "kl_clean_to_intervened"], aggfunc="mean"
    )
    heat.to_csv(output / "layer_window_tuning.csv")
    fig, axes = plt.subplots(1, 2, figsize=(12, max(5, len(heat) * 0.25)))
    for axis, column in zip(axes, heat.columns, strict=True):
        image = axis.imshow(heat[[column]].to_numpy(), aspect="auto", cmap="coolwarm")
        axis.set(yticks=range(len(heat)), yticklabels=heat.index, xticks=[0], xticklabels=[column])
        fig.colorbar(image, ax=axis)
    fig.tight_layout()
    fig.savefig(output / "layer_window_heatmap.png", dpi=180)
    plt.close(fig)
    return path
