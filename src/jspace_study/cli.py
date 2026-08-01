from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import structlog
import yaml

log = structlog.get_logger()


def load_config(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text())


def main() -> None:
    parser = argparse.ArgumentParser(prog="jspace-study")
    parser.add_argument("--config", type=Path, default=Path("configs/study.yaml"))
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run-benchmark")
    run.add_argument("--output", type=Path, default=Path("results/benchmark"))
    run.add_argument("--model", required=True, metavar="MODEL_ID")
    run.add_argument("--smoke-test", action="store_true", help="stop after five eligible trials")
    aggregate = commands.add_parser("aggregate")
    aggregate.add_argument("--output", type=Path, default=Path("results/benchmark"))
    args = parser.parse_args()
    config = load_config(args.config)
    from .benchmark import aggregate_results, run_benchmark

    if args.command == "aggregate":
        path = aggregate_results(args.output)
    else:
        path = run_benchmark(config, args.output, args.model, smoke=args.smoke_test)
    log.info("benchmark_complete", path=str(path))


if __name__ == "__main__":
    main()
