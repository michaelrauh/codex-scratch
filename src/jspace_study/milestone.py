"""Real GPT-2 clean/semantic/random milestone; no synthetic lenses or outcomes."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import torch

from .intervention import SwapHooks


@dataclass
class Condition:
    answer: str
    top_token: str
    selected_probabilities: dict[str, float]
    logits_file: str
    intervention: dict[int, dict[str, float]]


def _token_direction(lens: Any, weight: torch.Tensor, token: int, layer: int) -> torch.Tensor:
    direction = lens.jacobians[layer].to(weight.device).float().T @ weight[token].float()
    return direction / direction.norm().clamp_min(1e-12)


def run_milestone(config: dict[str, Any], output: Path, max_new_tokens: int = 4) -> Path:
    """Load pre-cached real artifacts, execute three conditions, and persist evidence.

    Validation is intentionally network-free. Run ``scripts/cache_assets.py`` during
    environment setup; a missing cache raises before any output is created.
    """
    import jlens
    from transformers import AutoModelForCausalLM, AutoTokenizer

    spec = config["models"][0]
    if spec["id"] != "openai-community/gpt2" or not spec.get("lens"):
        raise ValueError("milestone requires configured gpt2 and a real lens artifact")
    torch.manual_seed(config["seed"])
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    cache_root = Path(config["cache_dir"])
    model_path = cache_root / "models" / spec["cache_name"]
    lens_path = cache_root / "lenses" / config["lens_repository"].replace("/", "--")
    lens_file = lens_path / spec["lens"]
    model_files = [model_path / "config.json", model_path / "tokenizer.json"]
    has_weights = any(
        (model_path / filename).exists() for filename in ("model.safetensors", "pytorch_model.bin")
    )
    missing = [str(path) for path in model_files if not path.is_file()]
    if not has_weights:
        missing.append(f"{model_path}/{{model.safetensors,pytorch_model.bin}}")
    if not lens_file.is_file():
        missing.append(str(lens_file))
    if missing:
        raise FileNotFoundError(
            "real GPT-2 milestone cache is incomplete ("
            + ", ".join(missing)
            + "); run `uv run python scripts/cache_assets.py` in environment setup"
        )
    tokenizer: Any = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    hf: Any = (
        AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True).to(device).eval()
    )
    wrapped = jlens.from_hf(hf, tokenizer)
    lens = jlens.JacobianLens.from_pretrained(str(lens_path), filename=spec["lens"])
    prompt, source, target = "The capital of France is", " France", " China"
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
    output.mkdir(parents=True, exist_ok=True)
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    selected = {
        name: ids[0]
        for name, ids in {
            "original": tokenizer.encode(" Paris", add_special_tokens=False),
            "target": tokenizer.encode(" Beijing", add_special_tokens=False),
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

    result = {
        "status": "measured",
        "model": spec,
        "lens": {
            "repository": config["lens_repository"],
            "file": spec["lens"],
            "n_prompts": lens.n_prompts,
        },
        "device": device,
        "prompt": prompt,
        "source": source.strip(),
        "target": target.strip(),
        "layers": layers,
        "conditions": {
            name: asdict(value)
            for name, value in {
                "clean": condition("clean", None),
                "semantic": condition("semantic", "semantic"),
                "matched_random": condition("matched_random", "random"),
            }.items()
        },
    }
    path = output / "milestone.json"
    path.write_text(json.dumps(result, indent=2))
    return path
