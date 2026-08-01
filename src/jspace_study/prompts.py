"""Model-specific prompt eligibility with auditable exclusion reasons."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import yaml


def screen_prompts(
    model: Any, tokenizer: Any, path: Path, *, top_k: int = 20
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    included: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    for index, item in enumerate(yaml.safe_load(path.read_text())):
        record = {"prompt_index": index, **item}
        original = tokenizer.encode(f" {item['original']}", add_special_tokens=False)
        desired = tokenizer.encode(f" {item['desired']}", add_special_tokens=False)
        reasons = []
        if len(original) != 1:
            reasons.append(f"original_answer_token_count={len(original)}")
        if len(desired) != 1:
            reasons.append(f"target_answer_token_count={len(desired)}")
        if not reasons:
            inputs = tokenizer(item["prompt"], return_tensors="pt").input_ids
            with torch.inference_mode():
                logits = model(inputs).logits[0, -1].float().cpu()
            rank = int((logits > logits[original[0]]).sum()) + 1
            probability = float(logits.softmax(-1)[original[0]])
            record.update(clean_answer_rank=rank, clean_answer_probability=probability)
            if rank > top_k:
                reasons.append(f"clean_answer_rank={rank}>top_k={top_k}")
        if reasons:
            record["exclusion_reasons"] = reasons
            excluded.append(record)
        else:
            record.update(original_token=original[0], desired_token=desired[0])
            included.append(record)
    return included, excluded
