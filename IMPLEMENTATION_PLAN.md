# Concrete implementation plan and upstream inspection

## Reuse decision

Pinned revisions are recorded in `upstream-lock.yaml`. Anthropic's package is installed directly,
not copied: `jlens.from_hf`, `JacobianLens.from_pretrained`, `jlens.fit`, the GPT-2/Qwen layout
adapter, and Jacobian storage remain authoritative. The replication supplies the useful convention
`J_l.T @ unembedding[token]` and block-output hook pattern. Its two-vector edit assumes unit vectors
are orthogonal; this project deliberately wraps rather than copies that edit and uses a pseudoinverse.

## Stages

1. **Frozen milestone:** run the real GPT-2 model and downloaded Neuronpedia lens, persist complete
   next-token logits, generation, selected probabilities, layer list, and norm/energy measurements
   for clean, semantic, and exactly norm-matched random conditions.
2. **Lens completion:** fit missing lenses from one frozen generic-text corpus; store provenance,
   prompt hashes, upstream/model revisions, fit count, and checksums beside every artifact.
3. **Tune without leakage:** choose layer window and strength only on `country_tuning.yaml`; freeze
   them before evaluating `country_eval.yaml`. Run semantic, random, token-shuffled, and residual
   controls with resumable trial keys. Add the capability tasks and preserve logits/probabilities.
4. **Normalization matrix:** primary matched removed activation energy; repeat at fixed rank,
   width-proportional rank, and matched KL. KL matching uses monotone strength search on tuning data.
5. **Inference:** emit the raw Parquet, model CSV, correlations, prompt bootstrap intervals, LOMO,
   descriptive covariate residualization, primary/sensitivity charts, and final report. Four models
   remain the statistical unit; family/architecture are inseparable confounders in this ladder.

No claim is made until the real runs finish; missing lenses stop execution rather than being imputed.

## Environment asset setup

Pre-cache every configured model and any available model-specific lens while the setup environment has
network access. The manifest records unavailable lenses rather than substituting incompatible ones:

```bash
uv sync --extra dev
uv run python scripts/cache_assets.py
```

`validate-swap` then loads only the resulting `artifacts/models/` and `artifacts/lenses/` paths with
Transformers' `local_files_only=True`. It raises an actionable `FileNotFoundError` before creating a
result when either artifact is absent; runtime validation never attempts an implicit network download.
