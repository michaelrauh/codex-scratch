"""Environment setup: download immutable inputs before network-free validation.

Run once with network access::

    uv run python scripts/cache_assets.py

The runtime never falls back to the Hub, so an unavailable cache cannot silently
change a model revision or produce a result from a different artifact.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml
from huggingface_hub import HfApi, snapshot_download

MODEL_PATTERNS = [
    "*.json",
    "*.model",
    "*.tiktoken",
    "*.txt",
    "*.safetensors",
]


def main() -> None:
    config: dict[str, Any] = yaml.safe_load(Path("configs/study.yaml").read_text())
    root = Path(config["cache_dir"])
    lens_dir = root / "lenses" / config["lens_repository"].replace("/", "--")
    lens_dir.mkdir(parents=True, exist_ok=True)
    available_lenses = set(HfApi().list_repo_files(config["lens_repository"]))
    manifest: dict[str, Any] = {"models": [], "lens_repository": config["lens_repository"]}
    for spec in config["models"]:
        model_dir = root / "models" / spec["cache_name"]
        model_dir.mkdir(parents=True, exist_ok=True)
        entry = {
            "label": spec["label"],
            "repo_id": spec["id"],
            "requested_revision": spec["revision"],
            "lens": spec.get("lens"),
        }
        try:
            entry["resolved_revision"] = HfApi().model_info(
                spec["id"], revision=spec["revision"]
            ).sha
            entry["path"] = snapshot_download(
                repo_id=spec["id"],
                revision=spec["revision"],
                allow_patterns=MODEL_PATTERNS,
                local_dir=model_dir,
            )
            entry["model_status"] = "cached"
        except Exception as exc:  # noqa: BLE001 - every model failure belongs in the manifest
            entry["model_status"] = "failed"
            entry["model_error"] = f"{type(exc).__name__}: {exc}"
        lens = spec.get("lens")
        if lens and lens in available_lenses:
            snapshot_download(
                repo_id=config["lens_repository"], allow_patterns=[lens], local_dir=lens_dir
            )
            entry["lens_status"] = "cached"
        else:
            entry["lens_status"] = "unavailable"
        manifest["models"].append(entry)
    (root / "asset_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
