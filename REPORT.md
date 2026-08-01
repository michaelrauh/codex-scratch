# Qwen3 J-space capability ladder

## Hypothesis

More capable models will produce the expected semantic result from a J-space swap on a greater
percentage of eligible trials.

## Method

The initial milestone contains Qwen3 1.7B, 4B, and 8B. Each uses its matching prefitted averaged
Jacobian lens from `neuronpedia/jacobian-lens`; no lens is fit in this repository. For a source and
target fact, the lens transports their output-token directions to every layer in a selected
contiguous middle-layer window. At each layer, the intervention exchanges their least-squares
J-space coefficients without assuming the directions are orthogonal.

Window selection is model-specific but uses only `data/country_tuning.yaml`. Evaluation rows in
`data/country_eval.yaml` never influence selection. A trial is eligible only when the unedited model
greedily produces the source fact for the source prompt and the target fact for the target prompt.
Every exclusion and both clean answers are retained. Eligible swaps are scored in exactly three
classes: expected target fact, original source fact, and other/wrong.

## Results status

The corrected ladder has not been executed in this CPU-only development environment. The previous
GPT-2 measurements were methodologically inapplicable and have been removed. Run each checkpoint
independently on a suitable machine. Each command writes a resumable model-specific directory below
the shared output root. MPS uses FP16 and CUDA uses BF16 without changing prompts, interventions, or
scoring. A full model run requires 60 clean-eligible trials. Qwen3 14B and 32B remain staged later.

```bash
uv sync --extra dev
uv run jspace-study run-benchmark --model Qwen/Qwen3-1.7B --output results/qwen3-initial
uv run jspace-study run-benchmark --model Qwen/Qwen3-4B --output results/qwen3-initial
uv run jspace-study run-benchmark --model Qwen/Qwen3-8B --output results/qwen3-initial
uv run jspace-study aggregate --output results/qwen3-initial
```

Append `--smoke-test` to a model command to stop after five clean-eligible evaluation trials.
Smoke-test summaries are marked and deliberately excluded from aggregation.
