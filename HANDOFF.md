# Benchmark handoff

The project now implements only the fixed semantic-swap hypothesis described in `REPORT.md`.
GPT-2 124M, GPT-2 Medium, and GPT-2 Large completed the same 12 trials on 2026-08-01. The retained
CSV, SVG, and JSON outputs are real results. The Parquet table is generated alongside the CSV but is
ignored by Git because generated binary artifacts are not committed.

Run `uv run jspace-study run-benchmark --output results/benchmark` to reproduce all outputs.
