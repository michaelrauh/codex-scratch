# Fixed benchmark implementation

1. Load each GPT-2 checkpoint in order so memory use remains bounded.
2. Apply the 12 prompts and swaps from `data/benchmark.yaml` without model-specific changes.
3. Use the exact final-block Jacobian directions, unit normalization, coefficient exchange at
   strength 1.0, and the final block for every checkpoint.
4. Score clean answers and the three mutually exclusive swap outcomes.
5. Save trial and model tables, bootstrap intervals, charts, and exploratory correlations.
6. Require at least three successfully completed models before publishing any comparison.
