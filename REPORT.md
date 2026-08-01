# Fixed cross-model J-space benchmark

## Hypothesis

More capable models will respond correctly to a J-space semantic swap on a greater percentage of
trials than less capable models.

## Protocol

Every checkpoint receives the 12 trials in `data/benchmark.yaml`: three fixed entity swaps crossed
with capital, currency, language, and continent. The intervention uses the exact Jacobian of the two
answer logits at the final transformer block. Both directions are unit-normalized and their
least-squares coefficients are exchanged at strength 1.0. This is the same intervention and the same
last-block layer-selection rule for every checkpoint.

The generated next token is classified as the expected target fact, the original source fact, or
other/wrong. Clean accuracy is scored against the source fact. A swap succeeds only for the expected
target fact.

## Results

The identical benchmark completed on GPT-2 124M, GPT-2 Medium, and GPT-2 Large. Each model had 50%
clean accuracy and 0% swap success (bootstrap 95% CI 0%–0%); source retention and other/wrong were
both 50%. These are real checkpoint outputs, not simulated values. The fixed intervention therefore
did not support the hypothesis in this small benchmark.

Pearson and Spearman correlations are reported as exploratory. Both are undefined because clean
accuracy and swap success are constant across the three models.

Artifacts are in `results/benchmark/`: one CSV and Parquet row per trial, one CSV row per model, both
requested charts, bootstrap intervals, and exploratory correlations.

## Reproduce

```bash
uv sync --extra dev
uv run jspace-study run-benchmark --output results/benchmark
```
