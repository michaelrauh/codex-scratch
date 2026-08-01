# Qwen3 capability-ladder protocol

## Hypothesis

More capable models will produce the expected semantic result from a J-space swap on a greater
percentage of eligible trials.

## Staged execution

1. Run Qwen3 1.7B, 4B, and 8B independently with `--model`, using their exact prefitted averaged
   lenses from `neuronpedia/jacobian-lens`. Do not fit or substitute lenses. Each model-specific
   directory checkpoints evaluation rows for resumption on the same or another machine.
2. For each model, screen the separate tuning set by requiring correct greedy answers to both the
   source and target prompts. Select the contiguous relative middle-layer window with the highest
   tuning swap-success rate (ties choose the earliest preregistered candidate).
3. Freeze that model's selected window, independently screen the evaluation set, and record every
   eligible and excluded row. Require at least 60 clean-eligible evaluation trials per model.
4. Classify eligible interventions as expected target fact, original source fact, or other/wrong.
5. Use `aggregate` to combine completed non-smoke model summaries and publish the success bar chart
   and clean-capability scatter chart. No invocation requires all three models to be present.
6. Report those three models' measured percentages before changing the `stage` of Qwen3 14B and
   32B from `later` to `initial`.
