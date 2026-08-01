# Capability and causal J-space responsiveness

**Status: positive one-step causal validation established for GPT-2 124M; the requested three-model
capability pilot has not completed.** Missing-lens cases remain setup failures, not model experiments.
No cross-model capability conclusion is made.

## Positive teacher-forced causal validation (2026-08-01)

The corrected experiment uses one teacher-forced next-token step. For every prompt it caches the clean
residuals, calculates semantic, random, shuffled, no-op, and direct-logit positive-control deltas from
those identical activations, and runs separate forwards. It never generates an earlier token. All 210
ordered pairs built from 15 country/capital facts passed model-specific screening: both answers were
single tokens, both corresponding clean prompts ranked their correct answers in the top five, and the
target had clean probability at least 1e-6 on the source prompt. Thirty pairs were reserved for layer
selection and 60 disjoint pairs for the single held-out evaluation.

Every individual lens layer and every contiguous two- and three-layer window was evaluated on tuning
data. Layers 8--10 maximized the prespecified tuning contrast (semantic target log-probability change
minus the mean of matched-random and shuffled changes) and were then evaluated once on held-out data.
The layer/window CSV and heatmap are retained; no held-out result influenced selection.

On the 60 held-out pairs, semantic intervention increased target log probability by a mean 7.409760
(95% prompt-bootstrap CI [6.680161, 8.140408]). Matched-random changed it by -0.085437
([-0.276841, 0.109730]) and shuffled by -2.507272 ([-3.613129, -1.454204]). The paired semantic-minus-
random mean was 7.495197 (CI [6.755520, 8.221258], paired t-test p=2.02e-27); semantic-minus-shuffled
was 9.917031 (CI [8.486633, 11.416360], p=3.59e-19). The semantic-minus-mean-controls estimate was
8.706114 (CI [7.684331, 9.746264]). This meets the required positive-validation outcome.

The positive control independently raised target log probability by 7.960910 (CI [7.324494,
8.575760], p=6.14e-33), showing that the hook/continuation machinery can causally move the target.
No-op reproduced clean logits and KL exactly. Semantic and matched-random intervention norms averaged
57.8477 and removed energies averaged 3890.8459; their maximum held-out relative energy mismatch was
2.37e-7. Mean clean-to-intervened KL was 5.938929 for semantic, 0.193277 for random, and 5.798354 for
shuffled. The semantic target *logit* itself decreased by 1.630849 on average while source logit
decreased by 20.913299, so the positive result is specifically a large relative/log-probability and
rank effect, not direct excitation of the target logit.

The 277-prompt pretrained GPT-2 lens was also checked at 120 disjoint WikiText positions. Mean
J-lens top-1/5/10 agreement was 0.034091/0.081818/0.103788, compared with ordinary-logit-lens
0.146970/0.303030/0.475000, frequency 0.283333/0.566667/0.691667, and random
0/0/0.008333. Thus the causal result is positive under energy-matched controls, but this particular
future-token agreement metric does not show that J-lens is the best observational decoder. The raw
metrics and confidence intervals are preserved rather than hidden.

## Measured GPT-2 swap milestone (2026-08-01)

Hugging Face access succeeded and cached `openai-community/gpt2` plus the 277-prompt Neuronpedia
Jacobian lens. The model cache resolved `main` to commit
`607a30d783dfa663caf39e06633721c8d4cfcd7e`; the lens cache resolved to
`a4114d7752d11eb546e6cf372213d7e75526d3a1`. Validation then ran network-free on CPU over lens layers
3--8. Complete next-token logits, measurements, and SHA-256 hashes are retained in
`results/milestone/`.

For the prompt `The capital of France is`, all three conditions greedily generated
` now home to the`. The clean, semantic-swap, and matched-random probabilities for the selected
original token (` Paris`) were respectively 0.001629043, 0.001580215, and 0.001438739. Probabilities
for the selected target token (` Beijing`) were respectively 0.0000424950, 0.00000295880, and
0.000210946. Relative to clean, the semantic intervention changed those probabilities by
-0.0000488280 and -0.0000395362; the random intervention changed them by -0.000190303 and
+0.000168451.

Recorded per-layer removed energies for the semantic condition ranged from 0.461762 to 34.360355
(sum 119.144202); matched-random recorded energies ranged from 0.009473 to 16.808817 (sum
36.348811). That historical artifact used the earlier overwrite-only measurement format, so its
cross-condition sums do **not** demonstrate end-to-end energy equality.
The random delta is norm-matched locally to the semantic delta calculated on the random condition's
current activation; after earlier-layer interventions, the two conditions follow different activation
trajectories. This single prompt also moved the nominated target probability in the opposite direction
from the intended semantic capture. It is consequently a negative validation result, not evidence for
the study hypothesis, and motivates checking measurement timing/control semantics before the full run.

## Historical autoregressive setup results (superseded; not causal evidence)

All seven requested model repositories were successfully downloaded before execution; their resolved
revisions and download statuses are preserved in `results/ladder/asset_manifest.json`. The initial run
performed setup checks for every configured model and continued after failures. Runtime measures each attempt,
not asset-download time. Probability changes are relative to the clean target-token probability, and
effect size is semantic change minus matched-random change.

| Model | Parameter count | Semantic probability change | Matched-random probability change | Effect size | Runtime (s) | Success/failure |
|---|---:|---:|---:|---:|---:|---|
| GPT-2 124M | 124,000,000 | -0.0000395362 | +0.000168451 | -0.000207987 | 8.178770 | Success |
| GPT-2 Medium 355M | 355,000,000 | -0.0000371898 | +0.0000499791 | -0.0000871689 | 10.167501 | Pipeline validation success |
| Qwen2.5 0.5B | 500,000,000 | — | — | — | 0.000014 | Setup failure: no configured real lens |
| GPT-2 Large 774M | 774,000,000 | — | — | — | 0.000021 | Setup failure: no configured real lens |
| GPT-2 XL 1.5B | 1,500,000,000 | — | — | — | 0.000016 | Setup failure: no configured real lens |
| Qwen2.5 1.5B | 1,500,000,000 | — | — | — | 0.000012 | Setup failure: no configured real lens |
| Qwen2.5 3B | 3,000,000,000 | — | — | — | 0.000012 | Setup failure: no configured real lens |

The Neuronpedia repository has no pretrained lens for GPT-2 Medium, so a checkpoint-specific lens was
fitted locally with the pinned upstream `jlens.fit` running-mean Jacobian estimator. The fit used four
32-token WikiText sequences, layers 5--8, and the immutable model/dataset revisions recorded in the
provenance artifact. Fitting and validation consumed 2,196.868 seconds on CPU. This is much smaller than the
upstream recommendation of roughly 100 usable prompts and must be treated as a milestone lens, not a
publication-quality lens.

On four disjoint validation prompts at three positions each, mean top-10 agreement with the final
model's top-1 token was 0.041667 for J-lens and 0.208333 for ordinary logit lens. The earlier
resource-constrained 0.04 gate was not credible and has been retired. The enforced threshold is now
0.20, so this lens is rejected for causal conclusions and its existing intervention is retained only as
an intentional weak-lens control. It must be refitted near the upstream-recommended corpus size and
revalidated on at least 100 positions before GPT-2 Medium can enter the pilot.

The held-out prompt screen admitted five of nine candidates for GPT-2 Medium and retained every
exclusion reason. The first admitted prompt, `The capital of Japan is`, had clean ` Tokyo` rank 1 and
probability 0.0707906. Clean, semantic, and random target (` Rome`) probabilities were 0.000138031,
0.000100841, and 0.000188010. Semantic generation was ` Tokyo, and the`; matched-random generation was
` Tokyo, home to`. The semantic target change was -0.0000371898, the matched-random change was
+0.0000499791, and their difference was -0.0000871689.

Every semantic/random hook invocation now retains per-token, per-layer reference and applied energies.
Within each random invocation those arrays agree to floating-point tolerance. Across the whole generated
trajectory they need not have equal cumulative energy because earlier interventions produce different
later activations: observed semantic/random cumulative energies were 43,431.6163 and 17,256.3777.
Output KL divergences from clean were 0.00609956 and 0.0410099; generation-token divergences were 0 and
0.5. GPT-2 Large, GPT-2 XL, and all three Qwen checkpoints remain **setup failures**, not completed
experiments. No capability chart, fitted trend, or correlation is justified until at least three models
complete validated held-out evaluations.

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

Sensitivity charts and the final support category remain intentionally blank until a completed benchmark
with multiple lens-compatible models.
J-space writability would not establish consciousness or a global workspace.
