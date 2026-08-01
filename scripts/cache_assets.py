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
from huggingface_hub import snapshot_download


def main() -> None:
    config: dict[str, Any] = yaml.safe_load(Path("configs/study.yaml").read_text())
    spec = config["models"][0]
    root = Path(config["cache_dir"])
    model_dir = root / "models" / spec["cache_name"]
    lens_dir = root / "lenses" / config["lens_repository"].replace("/", "--")
    model_dir.mkdir(parents=True, exist_ok=True)
    lens_dir.mkdir(parents=True, exist_ok=True)

    resolved_model = snapshot_download(
        repo_id=spec["id"],
        revision=spec["revision"],
        local_dir=model_dir,
    )
    resolved_lens = snapshot_download(
        repo_id=config["lens_repository"],
        allow_patterns=[spec["lens"]],
        local_dir=lens_dir,
    )
    manifest = {
        "model": {
            "repo_id": spec["id"],
            "requested_revision": spec["revision"],
            "path": resolved_model,
        },
        "lens": {
            "repo_id": config["lens_repository"],
            "filename": spec["lens"],
            "path": resolved_lens,
        },
    }
    (root / "asset_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
