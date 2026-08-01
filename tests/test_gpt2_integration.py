import os
from pathlib import Path

import pytest
import yaml

from jspace_study.milestone import run_milestone


@pytest.mark.integration
@pytest.mark.skipif(os.getenv("RUN_GPT2_INTEGRATION") != "1", reason="requires model/lens download")
def test_real_gpt2_milestone(tmp_path: Path) -> None:
    config = yaml.safe_load(Path("configs/study.yaml").read_text())
    result = run_milestone(config, tmp_path, max_new_tokens=1)
    assert result.exists()
    assert all(
        (tmp_path / f"{name}_logits.pt").exists()
        for name in ("clean", "semantic", "matched_random")
    )


def test_missing_cache_fails_without_results(tmp_path: Path) -> None:
    config = yaml.safe_load(Path("configs/study.yaml").read_text())
    config["cache_dir"] = str(tmp_path / "empty-cache")
    output = tmp_path / "results"
    with pytest.raises(FileNotFoundError, match="scripts/cache_assets.py"):
        run_milestone(config, output, max_new_tokens=1)
    assert not output.exists()
