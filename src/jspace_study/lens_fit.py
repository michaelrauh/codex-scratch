"""Provenance-preserving wrapper around Anthropic's ``jlens.fit`` estimator."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def corpus_prompts(dataset_path: Path, tokenizer: Any, count: int, tokens: int) -> list[str]:
    """Deterministically pack non-empty WikiText rows into fixed-length prompt chunks."""
    rows = pd.read_parquet(dataset_path, columns=["text"])["text"]
    prompts: list[str] = []
    buffer = ""
    for text in rows:
        if str(text).strip():
            buffer += str(text)
        if len(tokenizer.encode(buffer, add_special_tokens=False)) >= tokens:
            prompts.append(buffer)
            buffer = ""
            if len(prompts) == count:
                break
    if len(prompts) != count:
        raise ValueError(f"dataset supplied only {len(prompts)} of {count} requested prompts")
    return prompts


def agreement_at_k(prediction: torch.Tensor, target: torch.Tensor, k: int) -> float:
    return float((prediction.topk(k, dim=-1).indices == target[:, None]).any(dim=-1).float().mean())


def validate_lens(
    lens: Any,
    model: Any,
    prompts: list[str],
    output_dir: Path,
    *,
    minimum_top10: float,
) -> dict[str, Any]:
    """Compare J-lens and ordinary-logit-lens agreement with final model predictions."""
    layers = lens.source_layers
    totals: dict[int, dict[str, list[float]]] = {
        layer: {f"{kind}_top{k}": [] for kind in ("jacobian", "ordinary") for k in (1, 5, 10)}
        for layer in layers
    }
    with torch.no_grad():
        for prompt in prompts:
            jacobian, final, _ = lens.apply(model, prompt, layers=layers, positions=[-8, -4, -1])
            ordinary, _, _ = lens.apply(
                model, prompt, layers=layers, positions=[-8, -4, -1], use_jacobian=False
            )
            target = final.argmax(dim=-1)
            for layer in layers:
                for k in (1, 5, 10):
                    totals[layer][f"jacobian_top{k}"].append(
                        agreement_at_k(jacobian[layer], target, k)
                    )
                    totals[layer][f"ordinary_top{k}"].append(
                        agreement_at_k(ordinary[layer], target, k)
                    )
    layerwise = {
        str(layer): {key: sum(values) / len(values) for key, values in metrics.items()}
        for layer, metrics in totals.items()
    }
    mean_agreement = {
        kind: {
            f"top{k}": sum(row[f"{kind}_top{k}"] for row in layerwise.values()) / len(layerwise)
            for k in (1, 5, 10)
        }
        for kind in ("jacobian", "ordinary")
    }
    jacobian_top10 = mean_agreement["jacobian"]["top10"]
    ordinary_top10 = mean_agreement["ordinary"]["top10"]
    passed = jacobian_top10 >= minimum_top10
    result = {
        "definition": "fraction of positions whose final-model top-1 token is in lens top-k",
        "positions_per_prompt": [-8, -4, -1],
        "n_validation_prompts": len(prompts),
        "minimum_mean_jacobian_top10": minimum_top10,
        "requires_not_worse_than_ordinary_top10": False,
        "mean_jacobian_top10": jacobian_top10,
        "mean_ordinary_top10": ordinary_top10,
        "mean_agreement": mean_agreement,
        "passed": passed,
        "layerwise": layerwise,
    }
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
    for axis, k in zip(axes, (1, 5, 10), strict=True):
        axis.plot(layers, [layerwise[str(x)][f"jacobian_top{k}"] for x in layers], label="J-lens")
        axis.plot(
            layers, [layerwise[str(x)][f"ordinary_top{k}"] for x in layers], label="logit lens"
        )
        axis.set(title=f"Top-{k}", xlabel="Layer", ylabel="Agreement")
        axis.set_ylim(0, 1)
    axes[-1].legend()
    fig.tight_layout()
    fig.savefig(output_dir / "layerwise_validation.png", dpi=180)
    plt.close(fig)
    return result


def validate_lens_with_baselines(
    lens: Any,
    model: Any,
    prompts: list[str],
    output_dir: Path,
    *,
    seed: int,
) -> Path:
    """Validate on >=100 positions against ordinary, random, and frequency baselines."""
    layers = lens.source_layers
    positions = [-8, -4, -1]
    targets: list[int] = []
    predictions: dict[str, dict[int, list[torch.Tensor]]] = {
        "jacobian": {layer: [] for layer in layers},
        "ordinary": {layer: [] for layer in layers},
    }
    for prompt in prompts:
        jacobian, final, _ = lens.apply(model, prompt, layers=layers, positions=positions)
        ordinary, _, _ = lens.apply(
            model, prompt, layers=layers, positions=positions, use_jacobian=False
        )
        targets.extend(final.argmax(dim=-1).tolist())
        for layer in layers:
            predictions["jacobian"][layer].append(jacobian[layer])
            predictions["ordinary"][layer].append(ordinary[layer])
    target = torch.tensor(targets)
    if len(target) < 100:
        raise ValueError(f"validation requires >=100 positions, got {len(target)}")
    example_logits = predictions["jacobian"][layers[0]][0]
    vocab_size = example_logits.shape[-1]
    counts = torch.bincount(target, minlength=vocab_size)
    generator = torch.Generator().manual_seed(seed)
    random_scores = torch.rand(len(target), vocab_size, generator=generator)

    def ci(values: np.ndarray) -> list[float]:
        rng = np.random.default_rng(seed)
        means = np.array(
            [rng.choice(values, len(values), replace=True).mean() for _ in range(5000)]
        )
        return [float(x) for x in np.quantile(means, [0.025, 0.975])]

    metrics: dict[str, dict[str, Any]] = {}
    layerwise: dict[str, dict[str, float]] = {}
    for kind in ("jacobian", "ordinary"):
        metrics[kind] = {}
        for k in (1, 5, 10):
            by_layer = []
            for layer in layers:
                logits = torch.cat(predictions[kind][layer])
                hits = (logits.topk(k, dim=-1).indices == target[:, None]).any(dim=-1).float()
                by_layer.append(hits)
                layerwise.setdefault(str(layer), {})[f"{kind}_top{k}"] = float(hits.mean())
            per_position = torch.stack(by_layer).float().mean(dim=0).numpy()
            metrics[kind][f"top{k}"] = {
                "mean": float(per_position.mean()),
                "ci95": ci(per_position),
            }
    for kind, scores in (
        ("frequency", counts.float().expand(len(target), -1)),
        ("random", random_scores),
    ):
        metrics[kind] = {}
        for k in (1, 5, 10):
            baseline_hits = (
                (scores.topk(k, dim=-1).indices == target[:, None])
                .any(dim=-1)
                .float()
                .numpy()
            )
            metrics[kind][f"top{k}"] = {
                "mean": float(baseline_hits.mean()),
                "ci95": ci(baseline_hits),
            }
    result = {
        "definition": "fraction of positions whose final-model top-1 is present in baseline top-k",
        "n_prompts": len(prompts),
        "n_positions": len(target),
        "positions_per_prompt": positions,
        "metrics": metrics,
        "layerwise": layerwise,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "validation_with_baselines.json"
    path.write_text(json.dumps(result, indent=2) + "\n")
    frame = pd.DataFrame.from_dict(layerwise, orient="index")
    frame.index.name = "layer"
    frame.to_csv(output_dir / "validation_layerwise.csv")
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
    for axis, k in zip(axes, (1, 5, 10), strict=True):
        axis.plot(layers, frame[f"jacobian_top{k}"], label="J-lens")
        axis.plot(layers, frame[f"ordinary_top{k}"], label="logit lens")
        axis.axhline(metrics["frequency"][f"top{k}"]["mean"], linestyle="--", label="frequency")
        axis.axhline(metrics["random"][f"top{k}"]["mean"], linestyle=":", label="random")
        axis.set(title=f"Top-{k}", xlabel="Layer", ylabel="Agreement", ylim=(0, 1))
    axes[-1].legend()
    fig.tight_layout()
    fig.savefig(output_dir / "validation_with_baselines.png", dpi=180)
    plt.close(fig)
    return path


def fit_and_validate(config: dict[str, Any], model_id: str, output_dir: Path) -> Path:
    """Fit the exact checkpoint with upstream code and retain full provenance."""
    import jlens
    from transformers import AutoModelForCausalLM, AutoTokenizer

    spec = next((item for item in config["models"] if item["id"] == model_id), None)
    if spec is None:
        raise ValueError(f"model is not configured: {model_id}")
    settings = config["lens_fit"]
    model_path = Path(config["cache_dir"]) / "models" / spec["cache_name"]
    dataset_path = Path(settings["dataset_path"])
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer: Any = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    factory: Any = AutoModelForCausalLM
    hf: Any = factory.from_pretrained(model_path, local_files_only=True).eval()
    model = jlens.from_hf(hf, tokenizer)
    all_prompts = corpus_prompts(
        dataset_path,
        tokenizer,
        settings["fit_prompts"] + settings["validation_prompts"],
        settings["prompt_tokens"],
    )
    fit_prompts = all_prompts[: settings["fit_prompts"]]
    validation_prompts = all_prompts[settings["fit_prompts"] :]
    provenance_path = output_dir / "provenance.json"
    prior_runtime = 0.0
    if provenance_path.exists():
        prior_runtime = float(json.loads(provenance_path.read_text()).get("runtime_seconds", 0.0))
    started = time.perf_counter()
    lens = jlens.fit(
        model,
        fit_prompts,
        source_layers=settings["source_layers"],
        dim_batch=settings["dim_batch"],
        max_seq_len=settings["max_seq_len"],
        skip_first=settings["skip_first"],
        checkpoint_path=str(output_dir / "fit.checkpoint.pt"),
    )
    lens_path = output_dir / "jacobian_lens.pt"
    lens.save(str(lens_path))
    validation = validate_lens(
        lens,
        model,
        validation_prompts,
        output_dir,
        minimum_top10=settings["minimum_mean_jacobian_top10"],
    )
    validation_path = output_dir / "validation.json"
    validation_path.write_text(json.dumps(validation, indent=2) + "\n")
    provenance = {
        "procedure": "jlens.fit running-mean Jacobian estimator",
        "upstream_commit": "581d398613e5602a5af361e1c34d3a92ea82ba8e",
        "model_id": spec["id"],
        "model_revision": spec["resolved_revision"],
        "dataset_id": settings["dataset_id"],
        "dataset_revision": settings["dataset_revision"],
        "dataset_file": str(dataset_path),
        "dataset_sha256": sha256(dataset_path),
        "fit_config": settings,
        "n_prompts_fitted": lens.n_prompts,
        "runtime_seconds": prior_runtime + time.perf_counter() - started,
        "losses": None,
        "loss_note": "The upstream estimator is a direct Jacobian mean and has no optimized loss.",
        "validation": validation,
        "artifacts": {
            "lens_sha256": sha256(lens_path),
            "validation_sha256": sha256(validation_path),
            "checkpoint_sha256": sha256(output_dir / "fit.checkpoint.pt"),
            "validation_plot_sha256": sha256(output_dir / "layerwise_validation.png"),
        },
    }
    provenance_path.write_text(json.dumps(provenance, indent=2) + "\n")
    return provenance_path
