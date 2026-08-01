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
    fit.add_argument("--output", type=Path, required=True)
    validate = commands.add_parser("validate-swap")
    validate.add_argument("--output", type=Path, default=Path("results/milestone"))
    validate.add_argument("--model")
    one_step = commands.add_parser("validate-one-step")
    one_step.add_argument("--output", type=Path, default=Path("results/one-step-gpt2"))
    run = commands.add_parser("run-benchmark")
    run.add_argument("--output", type=Path, default=Path("results/ladder"))
    agg = commands.add_parser("aggregate")
    agg.add_argument("--raw", type=Path, default=Path("results/raw_trials.parquet"))
    charts = commands.add_parser("chart")
    charts.add_argument("--summary", type=Path, default=Path("results/model_summary.csv"))
    args = parser.parse_args()
    config = load_config(args.config)
    if args.command == "fit-lens":
        from .lens_fit import fit_and_validate

        path = fit_and_validate(config, args.model, args.output)
        log.info("lens_fitted_and_validated", model=args.model, provenance=str(path))
    elif args.command == "validate-swap":
        path = run_milestone(config, args.output, model_id=args.model)
        log.info("real_milestone_complete", path=str(path))
    elif args.command == "validate-one-step":
        from .one_step import run_one_step_study

        path = run_one_step_study(config, args.output)
        log.info("one_step_validation_complete", path=str(path))
    elif args.command == "run-benchmark":
        from .ladder import run_ladder

        path = run_ladder(config, args.output)
        log.info("real_ladder_complete", path=str(path))
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
