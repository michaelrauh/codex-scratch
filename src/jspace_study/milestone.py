"""Real GPT-2 clean/semantic/random milestone; no synthetic lenses or outcomes."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import torch

from .intervention import SwapHooks
from .prompts import screen_prompts


@dataclass
class Condition:
    answer: str
    top_token: str
    selected_probabilities: dict[str, float]
    logits_file: str
    intervention: dict[int, list[dict[str, object]]]


def _token_direction(lens: Any, weight: torch.Tensor, token: int, layer: int) -> torch.Tensor:
    direction = lens.jacobians[layer].to(weight.device).float().T @ weight[token].float()
    return direction / direction.norm().clamp_min(1e-12)


def run_milestone(
    config: dict[str, Any], output: Path, max_new_tokens: int = 4, model_id: str | None = None
) -> Path:
    """Load pre-cached real artifacts, execute three conditions, and persist evidence.

    Validation is intentionally network-free. Run ``scripts/cache_assets.py`` during
    environment setup; a missing cache raises before any output is created.
    """
    import jlens
    from transformers import AutoModelForCausalLM, AutoTokenizer

    spec = next(
        (item for item in config["models"] if model_id is None or item["id"] == model_id), None
    )
    if spec is None:
        raise ValueError(f"model is not configured: {model_id}")
    if not spec.get("lens"):
        raise FileNotFoundError(
            f"no real Jacobian lens is configured for {spec['id']}; fit and configure one first"
        )
    started = time.perf_counter()
    torch.manual_seed(config["seed"])
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    cache_root = Path(config["cache_dir"])
    model_path = cache_root / "models" / spec["cache_name"]
    lens_path = cache_root / "lenses" / config["lens_repository"].replace("/", "--")
    lens_file = lens_path / spec["lens"]
    validation_file = lens_path / spec["validation"] if spec.get("validation") else None
    model_files = [model_path / "config.json", model_path / "tokenizer.json"]
    has_weights = any(
        (model_path / filename).exists() for filename in ("model.safetensors", "pytorch_model.bin")
    )
    missing = [str(path) for path in model_files if not path.is_file()]
    if not has_weights:
        missing.append(f"{model_path}/{{model.safetensors,pytorch_model.bin}}")
    if not lens_file.is_file():
        missing.append(str(lens_file))
    if validation_file is not None and not validation_file.is_file():
        missing.append(str(validation_file))
    if missing:
        raise FileNotFoundError(
            f"real {spec['id']} milestone cache is incomplete ("
            + ", ".join(missing)
            + "); run `uv run python scripts/cache_assets.py` in environment setup"
        )
    if validation_file is not None:
        validation = json.loads(validation_file.read_text())
        required_top10 = float(config["lens_fit"]["minimum_mean_jacobian_top10"])
        observed_top10 = float(validation.get("mean_jacobian_top10", 0.0))
        if not validation.get("passed", False) or observed_top10 < required_top10:
            raise RuntimeError(
                f"refusing causal run: {spec['id']} lens top-10 agreement {observed_top10} "
                f"is below required {required_top10} at {validation_file}"
            )
    tokenizer: Any = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model_factory: Any = AutoModelForCausalLM
    hf: Any = model_factory.from_pretrained(model_path, local_files_only=True).to(device).eval()
    wrapped = jlens.from_hf(hf, tokenizer)
    lens = jlens.JacobianLens.from_pretrained(str(lens_path), filename=spec["lens"])
    included, excluded = screen_prompts(hf, tokenizer, Path(config["data"]["evaluation"]))
    if not included:
        raise RuntimeError(f"no held-out prompts passed model-specific screening for {spec['id']}")
    output.mkdir(parents=True, exist_ok=True)
    screening_path = output / "prompt_screening.json"
    screening_path.write_text(
        json.dumps({"included": included, "excluded": excluded}, indent=2) + "\n"
    )
    selected_prompt = included[0]
    prompt = selected_prompt["prompt"]
    source, target = f" {selected_prompt['original']}", f" {selected_prompt['desired']}"
    source_ids = tokenizer.encode(source, add_special_tokens=False)
    target_ids = tokenizer.encode(target, add_special_tokens=False)
    if len(source_ids) != 1 or len(target_ids) != 1:
        raise RuntimeError("milestone country names must each be one tokenizer token")
    layers = [layer for layer in lens.source_layers if 3 <= layer <= 8]
    if not layers:
        raise RuntimeError("lens does not cover preregistered GPT-2 layer window 3..8")
    weight = hf.get_output_embeddings().weight
    directions = {
        layer: torch.stack(
            [
                _token_direction(lens, weight, source_ids[0], layer),
                _token_direction(lens, weight, target_ids[0], layer),
            ],
            dim=1,
        )
        for layer in layers
    }
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    selected = {
        name: ids[0]
        for name, ids in {
            "original": tokenizer.encode(source, add_special_tokens=False),
            "target": tokenizer.encode(target, add_special_tokens=False),
        }.items()
        if len(ids) == 1
    }

    def condition(name: str, kind: str | None) -> Condition:
        hook = (
            SwapHooks(wrapped.layers, directions, kind=kind, seed=config["seed"]) if kind else None
        )
        with torch.inference_mode(), hook if hook else torch.no_grad():
            logits = hf(input_ids).logits[0, -1].float().cpu()
            generated = hf.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        logits_path = output / f"{name}_logits.pt"
        torch.save(logits, logits_path)
        probs = logits.softmax(-1)
        return Condition(
            cast(str, tokenizer.decode(generated[0, input_ids.shape[1] :])),
            cast(str, tokenizer.decode([int(logits.argmax())])),
            {key: float(probs[token]) for key, token in selected.items()},
            str(logits_path),
            {} if hook is None else hook.measurements,
        )

    conditions = {
        "clean": condition("clean", None),
        "semantic": condition("semantic", "semantic"),
        "matched_random": condition("matched_random", "random"),
    }
    clean_logits = torch.load(output / "clean_logits.pt", weights_only=True)
    clean_probs = clean_logits.softmax(-1)
    condition_metrics: dict[str, dict[str, float]] = {}
    for name in ("semantic", "matched_random"):
        other_logits = torch.load(output / f"{name}_logits.pt", weights_only=True)
        other_probs = other_logits.softmax(-1)
        output_kl = float(
            (clean_probs * (clean_probs.clamp_min(1e-12).log() - other_probs.clamp_min(1e-12).log())).sum()
        )
        clean_answer = tokenizer.encode(conditions["clean"].answer, add_special_tokens=False)
        other_answer = tokenizer.encode(conditions[name].answer, add_special_tokens=False)
        compared = max(len(clean_answer), len(other_answer), 1)
        generation_divergence = sum(
            (clean_answer[i] if i < len(clean_answer) else None)
            != (other_answer[i] if i < len(other_answer) else None)
            for i in range(compared)
        ) / compared
        records = conditions[name].intervention
        condition_metrics[name] = {
            "cumulative_removed_energy": sum(
                sum(cast(list[float], record["applied_energy"]))
                for layer_records in records.values()
                for record in layer_records
            ),
            "output_kl_from_clean": output_kl,
            "generation_token_divergence": generation_divergence,
        }

    result = {
        "status": "measured",
        "model": spec,
        "lens": {
            "repository": config["lens_repository"],
            "file": spec["lens"],
            "n_prompts": lens.n_prompts,
        },
        "device": device,
        "runtime_seconds": time.perf_counter() - started,
        "prompt": prompt,
        "source": selected_prompt["source"],
        "target": selected_prompt["target"],
        "original_answer": selected_prompt["original"],
        "desired_answer": selected_prompt["desired"],
        "clean_answer_rank": selected_prompt["clean_answer_rank"],
        "clean_answer_probability": selected_prompt["clean_answer_probability"],
        "screening_file": str(screening_path),
        "layers": layers,
        "conditions": {name: asdict(value) for name, value in conditions.items()},
        "condition_metrics": condition_metrics,
    }
    path = output / "milestone.json"
    path.write_text(json.dumps(result, indent=2))
    return path
