from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import structlog
import yaml

from .analysis import aggregate, chart
from .milestone import run_milestone

log = structlog.get_logger()


def load_config(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text())


def main() -> None:
    parser = argparse.ArgumentParser(prog="jspace-study")
    parser.add_argument("--config", type=Path, default=Path("configs/study.yaml"))
    commands = parser.add_subparsers(dest="command", required=True)
    fit = commands.add_parser("fit-lens", help="fit a lens; never creates a fake artifact")
    fit.add_argument("--model", required=True)
    fit.add_argument("--prompts", type=Path, required=True)
    fit.add_argument("--output", type=Path, required=True)
    validate = commands.add_parser("validate-swap")
    validate.add_argument("--output", type=Path, default=Path("results/milestone"))
    run = commands.add_parser("run-benchmark")
    run.add_argument("--output", type=Path, default=Path("results/milestone"))
    agg = commands.add_parser("aggregate")
    agg.add_argument("--raw", type=Path, default=Path("results/raw_trials.parquet"))
    charts = commands.add_parser("chart")
    charts.add_argument("--summary", type=Path, default=Path("results/model_summary.csv"))
    args = parser.parse_args()
    config = load_config(args.config)
    if args.command == "fit-lens":
        import jlens
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model_spec = next((m for m in config["models"] if m["id"] == args.model), None)
        if model_spec is None:
            raise ValueError(
                "model must be declared in YAML (unsupported architectures fail here or in jlens.from_hf)"
            )
        hf = AutoModelForCausalLM.from_pretrained(args.model, revision=model_spec["revision"])
        wrapped = jlens.from_hf(
            hf, AutoTokenizer.from_pretrained(args.model, revision=model_spec["revision"])
        )
        prompts = [line.strip() for line in args.prompts.read_text().splitlines() if line.strip()]
        args.output.parent.mkdir(parents=True, exist_ok=True)
        jlens.fit(wrapped, prompts=prompts, checkpoint_path=str(args.output) + ".checkpoint").save(
            args.output
        )
        log.info("lens_fitted", model=args.model, prompts=len(prompts), output=str(args.output))
    elif args.command in {"validate-swap", "run-benchmark"}:
        path = run_milestone(config, args.output)
        log.info("real_milestone_complete", path=str(path))
        if args.command == "run-benchmark":
            log.warning(
                "ladder_requires_lenses",
                message="Fit/configure remaining real lenses, then execute trial matrix; no results fabricated.",
            )
    elif args.command == "aggregate":
        summary = aggregate(args.raw, Path(config["results_dir"]), config["seed"])
        log.info("aggregated", models=len(summary))
    else:
        import pandas as pd

        chart(
            pd.read_csv(args.summary),
            Path("charts"),
            Path(config["results_dir"]) / "correlation_summary.json",
        )
        log.info("charts_written")


if __name__ == "__main__":
    main()
