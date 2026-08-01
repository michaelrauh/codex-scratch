"""Hooks extending the replication's direction-swap convention with correct projections."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextlib import AbstractContextManager
from typing import Self

import torch
from torch import nn

from .projection import matched_random_delta, swap_pair


class SwapHooks(AbstractContextManager["SwapHooks"]):
    def __init__(
        self,
        blocks: Sequence[nn.Module],
        directions: Mapping[int, torch.Tensor],
        *,
        kind: str = "semantic",
        strength: float = 1.0,
        seed: int = 0,
    ) -> None:
        if kind not in {"semantic", "random"}:
            raise ValueError(f"unsupported control: {kind}")
        self.blocks, self.directions, self.kind, self.strength = blocks, directions, kind, strength
        self.generator = torch.Generator().manual_seed(seed)
        self.handles: list[torch.utils.hooks.RemovableHandle] = []
        self.measurements: dict[int, dict[str, float]] = {}

    def _hook(self, layer: int):
        directions = self.directions[layer]

        def apply(_module: nn.Module, _inputs: object, output: object) -> object:
            tensor = output if torch.is_tensor(output) else output[0]  # type: ignore[index]
            h = tensor.float()
            _, semantic_delta = swap_pair(h, directions.to(h.device), self.strength)
            delta = (
                semantic_delta
                if self.kind == "semantic"
                else matched_random_delta(semantic_delta, self.generator)
            )
            self.measurements[layer] = {
                "mean_norm": float(delta.norm(dim=-1).mean()),
                "removed_energy": float(delta.square().sum(dim=-1).mean()),
            }
            changed = (h + delta).to(tensor.dtype)
            return changed if torch.is_tensor(output) else (changed, *output[1:])  # type: ignore[index]

        return apply

    def __enter__(self) -> Self:
        for layer in self.directions:
            self.handles.append(self.blocks[layer].register_forward_hook(self._hook(layer)))
        return self

    def __exit__(self, *args: object) -> None:
        for handle in self.handles:
            handle.remove()
