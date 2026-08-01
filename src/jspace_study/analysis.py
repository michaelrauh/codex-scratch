"""Aggregation, model-level correlations, prompt bootstrap, and publication charts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

REQUIRED = {
    "model",
    "prompt_id",
    "split",
    "capability_score",
    "semantic_target_capture",
    "random_target_capture",
    "unrelated_degradation",
}


def aggregate(
    raw_path: Path, results_dir: Path, seed: int = 20260801, bootstraps: int = 2000
) -> pd.DataFrame:
    raw = pd.read_parquet(raw_path)
    missing = REQUIRED - set(raw.columns)
    if missing:
        raise ValueError(f"raw trials missing columns: {sorted(missing)}")
    evaluation = raw.loc[raw.split.eq("evaluation")].copy()
    evaluation["selective_effect"] = (
        evaluation.semantic_target_capture
        - evaluation.random_target_capture
        - evaluation.unrelated_degradation
    )
    summary = evaluation.groupby("model", as_index=False).agg(
        capability=("capability_score", "mean"),
        selective_effect=("selective_effect", "mean"),
        parameter_count=("parameter_count", "first"),
        residual_width=("residual_width", "first"),
        intervention_rank=("intervention_rank", "first"),
        layers_intervened=("layers_intervened", "first"),
        intervention_norm=("intervention_norm", "mean"),
        clean_accuracy=("clean_accuracy", "mean"),
        semantic_capture=("semantic_target_capture", "mean"),
        random_capture=("random_target_capture", "mean"),
        shuffled_capture=("shuffled_target_capture", "mean"),
        original_retention=("original_retention", "mean"),
        unrelated_degradation=("unrelated_degradation", "mean"),
    )
    rng = np.random.default_rng(seed)
    cis: dict[str, tuple[float, float]] = {}
    for model, frame in evaluation.groupby("model"):
        values = frame.selective_effect.to_numpy()
        means = np.array(
            [rng.choice(values, len(values), replace=True).mean() for _ in range(bootstraps)]
        )
        cis[model] = tuple(np.quantile(means, [0.025, 0.975]))
    summary[["selective_ci_low", "selective_ci_high"]] = summary.model.map(cis).apply(pd.Series)
    results_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(results_dir / "model_summary.csv", index=False)
    pearson = stats.pearsonr(summary.capability, summary.selective_effect)
    spearman = stats.spearmanr(summary.capability, summary.selective_effect)
    loo = []
    for held in summary.model:
        sub = summary[summary.model.ne(held)]
        loo.append(
            {
                "held_out": held,
                "pearson_r": float(stats.pearsonr(sub.capability, sub.selective_effect).statistic),
                "spearman_rho": float(
                    stats.spearmanr(sub.capability, sub.selective_effect).statistic
                ),
            }
        )
    correlations: dict[str, Any] = {
        "n_models": len(summary),
        "warning": "Exploratory and underpowered; prompt bootstrap does not increase model-level n.",
        "pearson": {"r": float(pearson.statistic), "p": float(pearson.pvalue)},
        "spearman": {"rho": float(spearman.statistic), "p": float(spearman.pvalue)},
        "leave_one_out": loo,
    }
    (results_dir / "correlation_summary.json").write_text(json.dumps(correlations, indent=2))
    return summary


def chart(summary: pd.DataFrame, charts_dir: Path, correlations_path: Path) -> None:
    charts_dir.mkdir(parents=True, exist_ok=True)
    correlations = json.loads(correlations_path.read_text())
    fig, ax = plt.subplots(figsize=(8, 5.5))
    yerr = np.vstack(
        [
            summary.selective_effect - summary.selective_ci_low,
            summary.selective_ci_high - summary.selective_effect,
        ]
    )
    ax.errorbar(summary.capability, summary.selective_effect, yerr=yerr, fmt="o", capsize=4)
    if len(summary) >= 2:
        x = np.linspace(summary.capability.min(), summary.capability.max(), 100)
        ax.plot(
            x,
            np.polyval(np.polyfit(summary.capability, summary.selective_effect, 1), x),
            "--",
            label="OLS",
        )
    for row in summary.itertuples():
        ax.annotate(
            row.model,
            (row.capability, row.selective_effect),
            xytext=(5, 5),
            textcoords="offset points",
        )
    ax.set(
        xlabel="Clean capability (preregistered normalized mean)", ylabel="Selective J-space effect"
    )
    ax.set_title(
        f"Pearson r={correlations['pearson']['r']:.2f}; Spearman ρ={correlations['spearman']['rho']:.2f}; model n={len(summary)}"
    )
    ax.legend()
    fig.tight_layout()
    for suffix in ("png", "svg"):
        fig.savefig(charts_dir / f"capability_vs_selective_effect.{suffix}", dpi=180)
    plt.close(fig)
