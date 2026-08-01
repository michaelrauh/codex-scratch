# Capability and causal J-space responsiveness

**Status: protocol and real GPT-2 milestone implementation complete; full experiment not yet run.**
No empirical result, chart, CSV, or correlation is presented because doing so before running the
models would fabricate evidence. The hypothesis is currently **inconclusive**.

## Provenance and models

Exact upstream commits and reuse boundaries are in `upstream-lock.yaml`. Model IDs, requested
revisions, approximate parameter covariates, and lens locations are frozen in `configs/study.yaml`.
For a completed run, immutable Hugging Face resolved commit hashes must be copied from cache metadata
into the report; the mutable `main` request is not falsely represented as an exact resolved revision.
Model and lens assets are downloaded only by the documented environment setup command. Experimental
runtime uses the local cache exclusively and fails without producing results if it is incomplete.

## Preregistered methods and metrics

The primary capability metric is the mean normalized clean score across country completion, analogy,
one-hop QA, easy two-hop composition, and next-token loss components for which every included model
exceeds the 0.05 floor. Components must also be reported separately. Parameter count is a covariate,
not capability.

The primary intervention uses matched removed activation energy. Within the non-orthogonal span of
the source and target transported unembedding directions, coefficients are obtained by pseudoinverse
and exchanged. The primary response is semantic target-capture gain minus matched-random capture gain
minus unrelated-task degradation. Clean accuracy, desired capture, original retention, random and
shuffled capture, incoherent/other answers, norms, rank, layers, width, and parameters remain separate.

Layer windows and strengths are selected using only `data/country_tuning.yaml`; the disjoint
`data/country_eval.yaml` is opened after selection. Fixed-rank, width-proportional rank, matched-energy,
and matched-KL analyses are specified; matched energy is primary because it equates activation damage
directly and admits an exact per-token random control.

## Outputs, uncertainty, and limitations

After real trials, aggregation produces the requested Parquet-derived model CSV, JSON, PNG/SVG chart,
OLS line, Pearson/Spearman estimates, prompt-bootstrap within-model intervals, and leave-one-model-out
results. Prompt resampling quantifies prompt uncertainty only and never increases model-level *n*.
With four models all correlations are exploratory and severely underpowered. Architecture and family,
width, rank, layers, intervention norm, and parameters are descriptive confounders; a five-covariate
regression with four observations is unidentified and will not be dressed up as adjustment.

Runtime and machine information, exclusions, resolved artifact revisions, sensitivity charts, negative
findings, and the final support category remain intentionally blank until a completed benchmark run.
J-space writability would not establish consciousness or a global workspace.
