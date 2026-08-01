# J-space experiment handoff

## State of the branch

This cleanup starts from source/result commit `6ea8c3c`. It does not rerun an experiment or change any
measured value. The experiment code uses Anthropic `jacobian-lens` commit
`581d398613e5602a5af361e1c34d3a92ea82ba8e`. The measured GPT-2 inputs resolved to model commit
`607a30d783dfa663caf39e06633721c8d4cfcd7e` and Neuronpedia lens-repository commit
`a4114d7752d11eb546e6cf372213d7e75526d3a1`. The local GPT-2 Medium fit records model commit
`6dcaa7a952f72f9298047fd5137cd6e4f05f41da` and Salesforce/WikiText dataset commit
`b08601e04326c79dfdd32d625aee71d232d685c3`.

Implemented and actually run:

- cache-only configured model/lens loading and explicit setup-failure recording;
- pseudoinverse-based swaps in non-orthogonal bases and exact same-activation energy-matched controls;
- a provenance-recorded, four-prompt GPT-2 Medium lens fit (now rejected by the strengthened 0.20
  observational threshold and retained only as a weak-lens control);
- model-specific prompt screening with explicit exclusions;
- teacher-forced one-step GPT-2 124M evaluation from shared clean activations, including no-op,
  semantic, matched-random, shuffled-token, and direct-logit positive controls;
- tuning over every individual pretrained-lens layer and contiguous two-/three-layer window, followed
  by one evaluation of selected layers 8--10 on 60 held-out pairs;
- a 120-position GPT-2 lens comparison against ordinary-logit-lens, frequency, and random baselines.

## Exact validated results

No values below were recomputed during PR cleanup. The machine-readable source is
`results/one-step-gpt2/summary.json`, with condition means in
`results/one-step-gpt2/heldout_condition_summary.csv`.

- screened/eligible pairs: 210/210; tuning/evaluation pairs: 30/60; selected layers: 8--10;
- semantic target log-probability delta: `7.409759628772735`, bootstrap 95% CI
  `[6.680161013106505, 8.14040847837925]`;
- matched-random delta: `-0.08543691635131836`, CI
  `[-0.2768407060702642, 0.10973023792107899]`;
- shuffled delta: `-2.507271639506022`, CI
  `[-3.6131291244427364, -1.4542037689685825]`;
- semantic minus random: `7.495196545124054`, CI
  `[6.755520100394884, 8.221257936656475]`, paired t-test
  `p=2.0217575704737054e-27`;
- semantic minus shuffled: `9.917031268278757`, CI
  `[8.486633374790351, 11.416359972258407]`, paired t-test
  `p=3.5914588601741415e-19`;
- semantic minus mean matched controls: `8.706113906701406`, CI
  `[7.684331009685994, 9.746263875663281]`;
- positive-control delta: `7.960909716288248`, CI
  `[7.324494065443674, 8.575759610732396]`, `p=6.138695777552244e-33`;
- no-op logit delta, KL, and energy: exactly zero;
- maximum held-out semantic/random relative energy mismatch: `2.3723329047318552e-07`;
- pretrained GPT-2 J-lens top-1/5/10 agreement on 120 positions:
  `0.034090910106897354 / 0.08181818574666977 / 0.10378788411617279`;
- ordinary-logit-lens top-1/5/10 agreement:
  `0.14696970582008362 / 0.3030303120613098 / 0.4749999940395355`.

## Reproduction commands

These commands intentionally are documented but were **not** rerun during binary cleanup:

```bash
uv sync --extra dev
uv run python scripts/cache_assets.py
uv run jspace-study validate-one-step --output results/one-step-gpt2
RUN_GPT2_INTEGRATION=1 uv run pytest -q
uv run ruff check .
uv run mypy src
```

The 120-position baseline comparison additionally needs the pinned WikiText validation parquet and can
be reproduced with the retained helper:

```bash
uv run python - <<'PY'
from pathlib import Path
import jlens
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM, AutoTokenizer
from jspace_study.lens_fit import corpus_prompts, validate_lens_with_baselines

dataset = hf_hub_download(
    "Salesforce/wikitext",
    "wikitext-103-raw-v1/validation-00000-of-00001.parquet",
    repo_type="dataset",
    revision="b08601e04326c79dfdd32d625aee71d232d685c3",
    local_dir="artifacts/datasets/Salesforce--wikitext",
)
model_path = Path("artifacts/models/openai-community--gpt2")
tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
hf = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True).eval()
model = jlens.from_hf(hf, tokenizer)
lens = jlens.JacobianLens.load(
    "artifacts/lenses/neuronpedia--jacobian-lens/"
    "gpt2-small/jlens/Salesforce-wikitext/gpt2_jacobian_lens.pt"
)
prompts = corpus_prompts(Path(dataset), tokenizer, 40, 40)
validate_lens_with_baselines(
    lens, model, prompts, Path("results/one-step-gpt2/lens-validation"), seed=20260801
)
PY
```

To reproduce the intentionally weak GPT-2 Medium fit, download the same pinned dataset first and run:

```bash
uv run jspace-study fit-lens \
  --model openai-community/gpt2-medium \
  --output artifacts/lenses/neuronpedia--jacobian-lens/fitted/openai-community--gpt2-medium
```

## Next unfinished milestone

Do not start GPT-2 Large or a capability chart yet. Refit GPT-2 Medium on approximately the upstream-
recommended corpus size (about 100 prompts), validate on at least 100 disjoint positions against all
three baselines, and require the configured 0.20 top-10 threshold before running the same one-step pilot.
Only after GPT-2 Medium passes should the third model be fitted and a cross-model pilot considered.

## Local untracked binary/raw outputs

The following paths were removed from Git tracking but were left in the working directory. They are
ignored and will not exist in a fresh checkout; the retained `SHA256SUMS` files preserve their hashes.

```text
artifacts/lenses/neuronpedia--jacobian-lens/fitted/openai-community--gpt2-medium/fit.checkpoint.pt
artifacts/lenses/neuronpedia--jacobian-lens/fitted/openai-community--gpt2-medium/jacobian_lens.pt
artifacts/lenses/neuronpedia--jacobian-lens/fitted/openai-community--gpt2-medium/layerwise_validation.png
results/ladder/openai-community--gpt2/clean_logits.pt
results/ladder/openai-community--gpt2/matched_random_logits.pt
results/ladder/openai-community--gpt2/semantic_logits.pt
results/medium-milestone/clean_logits.pt
results/medium-milestone/matched_random_logits.pt
results/medium-milestone/semantic_logits.pt
results/milestone/clean_logits.pt
results/milestone/matched_random_logits.pt
results/milestone/semantic_logits.pt
results/one-step-gpt2/layer_window_heatmap.png
results/one-step-gpt2/lens-validation/validation_with_baselines.png
results/one-step-gpt2/one_step_raw.csv
```

Model caches, downloaded pretrained lenses, and dataset caches also belong under ignored `artifacts/`
(`artifacts/models/`, `artifacts/lenses/`, and `artifacts/datasets/`) and must remain untracked.
