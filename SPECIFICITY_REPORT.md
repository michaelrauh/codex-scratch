# GPT-2 124M semantic-specificity experiment

## Question and frozen evaluation

The earlier held-out experiment showed a large causal gain for a nominated target capital. This
follow-up asks whether that gain is specific to the capital relation, rather than a general ability to
write a nominated token through the same J-space operation.

The experiment reused the already selected GPT-2 124M lens layers 8--10 and the same first 60
hash-assigned evaluation pairs. It did not tune a layer, threshold, or control vocabulary on these
outcomes. For each source prompt, the control was the single-token alphabetic vocabulary item that:

1. was not any screened country-capital answer token; and
2. had clean log probability closest to the nominated target capital.

At each layer, the source-capital direction was swapped with either the semantic target-capital
direction or the matched non-capital direction. The latter delta was rescaled separately per layer to
equal the semantic delta's removed activation energy. Both conditions started from the same cached
clean activation. The primary specificity contrast is the paired difference between the two
conditions' gain for their own nominated token. A second, within-intervention contrast compares the
semantic target with the matched off-target token.

## Measured results

All 210 ordered country pairs remained eligible; 60 frozen evaluation pairs were run.

| Measure | Estimate |
|---|---:|
| Semantic capital nominated-token log-probability gain | 7.409757 |
| Matched non-capital nominated-token gain | 7.253299 |
| Semantic minus non-capital nominated gain | 0.156458 |
| Bootstrap 95% CI | [-0.226416, 0.567615] |
| Paired t-test | p = 0.445959 |
| Semantic target minus matched off-target gain | 10.296925 |
| Bootstrap 95% CI | [8.874132, 11.750713] |
| Paired t-test | p = 2.98e-20 |
| Mean absolute clean log-probability mismatch | 0.007851 |
| Maximum relative removed-energy mismatch | 2.07e-7 |

The machine-readable summary is `results/specificity-gpt2/summary.json`; all 60 paired measurements
are in `results/specificity-gpt2/pair_results.csv`, and `results/specificity-gpt2/specificity.svg`
plots the two nominated-target means with prompt-bootstrap intervals.

## Numerical conclusion

The operation is strongly target-token-specific: under a semantic swap, its nominated capital gained
10.30 log-probability units more than the matched off-target. It is **not measurably
capital-semantic-specific** under the more diagnostic paired control. An equally probable alphabetic
non-capital target gained almost as much, and the semantic advantage was only 0.156 log-probability
units with a confidence interval spanning zero and p=0.446.

Accordingly, the prior positive result supports causal, selective token writability through these
J-space directions, but this experiment does not support interpreting the magnitude as specificity
for the country-to-capital semantic relation. This conclusion is limited to GPT-2 124M, this lens,
layers 8--10, the screened country prompts, and alphabetic token controls; it is not a cross-model
capability claim.
