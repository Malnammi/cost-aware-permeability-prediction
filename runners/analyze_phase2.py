#!/usr/bin/env python
"""
Phase 2 Model Selection - Mean/Boxplot Analysis

This runner promotes the Phase 2 winner by ranking configurations on the
cross-fold mean of the diagnostic metric (RMSE for original scope, RMSE_log for
log/both scope). The script keeps LOWO fold-level context visible through
boxplots and per-well heatmaps.

Outputs
-------
  Tables  -> results/phase2_model_selection/analysis/tables/
  Figures -> results/phase2_model_selection/analysis/figures/{pdf,svg,png}/
  Report  -> results/phase2_model_selection/analysis/phase2_report.md
  HTML    -> results/phase2_model_selection/analysis/phase2_report.html  (--html)

Usage
-----
    python runners/analyze_phase2.py
    python runners/analyze_phase2.py --html
    python runners/analyze_phase2.py --metric-scope log
    python runners/analyze_phase2.py --alpha 0.10
    python runners/analyze_phase2.py --no-figures
    python runners/analyze_phase2.py --top-n 7
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

# - Constants -----------------------------------------------------------------

METRICS = ["RMSE", "MAE", "R2", "RMSE_log", "MAE_log", "R2_log"]
ORIGINAL_METRICS = ["RMSE", "MAE", "R2"]
LOG_METRICS = ["RMSE_log", "MAE_log", "R2_log"]

METRIC_SCOPE_CHOICES = ("original", "log", "both")
DEFAULT_METRIC_SCOPE = "both"

HIGHER_IS_BETTER = {"R2", "R2_log"}
LOWER_IS_BETTER = {"RMSE", "MAE", "RMSE_log", "MAE_log"}
BOXPLOT_MEAN_PROPS = {
    "color": "red",
    "alpha": 0.65,
    "linewidth": 1.6,
    "linestyle": "-",
}

N_FOLDS = 7
FOLD_IDS = list(range(N_FOLDS))
WELL_LABELS = [chr(65 + i) for i in range(N_FOLDS)]  # A-G


def resolve_metric_scope(metric_scope: str) -> tuple[list[str], list[str], str]:
    scope = metric_scope.lower()
    if scope == "original":
        return ORIGINAL_METRICS.copy(), ["RMSE", "R2"], "RMSE"
    if scope == "log":
        return LOG_METRICS.copy(), ["RMSE_log", "R2_log"], "RMSE_log"
    return METRICS.copy(), ["RMSE_log", "R2_log"], "RMSE_log"


def scoped_stem(stem: str, metric_scope: str) -> str:
    return stem if metric_scope == DEFAULT_METRIC_SCOPE else f"{stem}_{metric_scope}"


# - Data loading ---------------------------------------------------------------


def load_experiment_config(config_path: Optional[Path] = None) -> dict:
    if config_path is None:
        config_path = (
            Path(__file__).parent.parent / "configs" / "experiment_config.json"
        )
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_best_trials(results_dir: Path) -> pd.DataFrame:
    """For each (variant, model), load the best CV trial by mean_score."""
    cv_dir = results_dir / "cv_results"
    if not cv_dir.exists():
        print(f"Warning: CV results directory not found: {cv_dir}")
        return pd.DataFrame()

    csv_files = sorted(cv_dir.glob("*.csv"))
    if not csv_files:
        print(f"Warning: No CSV files found in {cv_dir}")
        return pd.DataFrame()

    rows: list[dict] = []
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
        except Exception as exc:
            print(f"Warning: Could not read {csv_file}: {exc}")
            continue

        if df.empty:
            continue

        for (variant, model), group in df.groupby(["variant", "model"]):
            best_idx = group["mean_score"].idxmax()
            best = group.loc[best_idx]

            row: dict = {"variant": variant, "model": model}
            for fold in FOLD_IDS:
                for metric in METRICS:
                    col = f"fold_{fold}_{metric}"
                    if col in best.index:
                        row[col] = best[col]

            for metric in METRICS:
                if metric in best.index:
                    row[f"mean_{metric}"] = best[metric]
                std_col = f"{metric}_std"
                if std_col in best.index:
                    row[f"std_{metric}"] = best[std_col]

            row["mean_score"] = best.get("mean_score", np.nan)
            row["best_trial_id"] = int(best.get("trial_id", -1))
            row["search_type"] = best.get("search_type", "unknown")
            rows.append(row)

    return pd.DataFrame(rows) if rows else pd.DataFrame()


def load_timing_info(results_dir: Path) -> pd.DataFrame:
    """Load runtime information from *_best_params.json files."""
    hp_dir = results_dir / "hp_search"
    if not hp_dir.exists():
        return pd.DataFrame()

    rows: list[dict] = []
    for json_file in sorted(hp_dir.glob("*_best_params.json")):
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as exc:
            print(f"Warning: Could not read {json_file}: {exc}")
            continue

        timing = data.get("timing", {})
        rows.append(
            {
                "variant": data.get("variant", "unknown"),
                "model": data.get("model", "unknown"),
                "total_search_seconds": timing.get("total_search_seconds", np.nan),
                "random_search_seconds": timing.get(
                    "random_search_seconds", np.nan
                ),
                "bayesian_search_seconds": timing.get(
                    "bayesian_search_seconds", np.nan
                ),
                "n_total_trials": timing.get("n_total_trials", np.nan),
                "avg_trial_seconds": timing.get(
                    "avg_trial_seconds_overall", np.nan
                ),
            }
        )

    return pd.DataFrame(rows) if rows else pd.DataFrame()


# - Metrics summary ------------------------------------------------------------


def compute_ci_bounds(
    best_trials_df: pd.DataFrame,
    alpha: float = 0.05,
    metrics: Optional[list[str]] = None,
) -> pd.DataFrame:
    """Compute mean/std/CI columns for each configuration and metric.

    CI columns are retained for descriptive reporting only (not ranking).
    """
    t_crit = stats.t.ppf(1 - alpha / 2, df=N_FOLDS - 1)
    active_metrics = metrics or METRICS

    rows: list[dict] = []
    for _, trial in best_trials_df.iterrows():
        row: dict = {"variant": trial["variant"], "model": trial["model"]}

        for metric in active_metrics:
            fold_vals = [
                trial[f"fold_{fold}_{metric}"]
                for fold in FOLD_IDS
                if f"fold_{fold}_{metric}" in trial.index
                and pd.notna(trial[f"fold_{fold}_{metric}"])
            ]

            if len(fold_vals) < 2:
                row[f"{metric}_mean"] = np.nan
                row[f"{metric}_std"] = np.nan
                row[f"{metric}_ci_lower"] = np.nan
                row[f"{metric}_ci_upper"] = np.nan
                continue

            mean = float(np.mean(fold_vals))
            std = float(np.std(fold_vals, ddof=1))
            sem = std / np.sqrt(len(fold_vals))
            ci_lower = mean - t_crit * sem
            ci_upper = mean + t_crit * sem

            row[f"{metric}_mean"] = mean
            row[f"{metric}_std"] = std
            row[f"{metric}_ci_lower"] = ci_lower
            row[f"{metric}_ci_upper"] = ci_upper

        rows.append(row)

    return pd.DataFrame(rows)


def build_summary(
    ci_df: pd.DataFrame,
    metrics: list[str],
    diagnostic_metric: str = "RMSE_log",
) -> pd.DataFrame:
    """Build a wide configuration summary sorted by diagnostic mean."""
    rows: list[dict] = []
    for _, ci_row in ci_df.iterrows():
        row: dict = {
            "variant": ci_row["variant"],
            "model": ci_row["model"],
            "label": f"{ci_row['model']} | {ci_row['variant']}",
        }
        for metric in metrics:
            row[f"{metric}_mean"] = ci_row.get(f"{metric}_mean")
            row[f"{metric}_std"] = ci_row.get(f"{metric}_std")
            row[f"{metric}_ci_lower"] = ci_row.get(f"{metric}_ci_lower")
            row[f"{metric}_ci_upper"] = ci_row.get(f"{metric}_ci_upper")
        rows.append(row)

    summary = pd.DataFrame(rows)
    if summary.empty:
        return summary

    sort_col = f"{diagnostic_metric}_mean"
    ascending = diagnostic_metric in LOWER_IS_BETTER
    summary = summary.sort_values(sort_col, ascending=ascending).reset_index(drop=True)
    summary.insert(0, "overall_rank", range(1, len(summary) + 1))
    return summary


def three_level_summary(
    summary: pd.DataFrame,
    diagnostic_metric: str = "RMSE_log",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Three-level summary based on mean performance."""
    mean_col = f"{diagnostic_metric}_mean"
    ascending = diagnostic_metric in LOWER_IS_BETTER

    if summary.empty or mean_col not in summary.columns:
        empty_level = pd.DataFrame()
        return empty_level, empty_level, summary.copy()

    level1 = (
        summary.groupby("variant")[mean_col]
        .agg(["mean", "std", "min", "max", "count"])
        .rename(
            columns={
                "mean": "avg_mean",
                "std": "std_mean",
                "min": "best_mean",
                "max": "worst_mean",
                "count": "n_models",
            }
        )
        .sort_values("avg_mean", ascending=ascending)
        .reset_index()
    )
    level1["std_mean"] = level1["std_mean"].fillna(0)
    level1["variant_rank"] = range(1, len(level1) + 1)

    level2 = (
        summary.groupby("model")[mean_col]
        .agg(["mean", "std", "min", "max", "count"])
        .rename(
            columns={
                "mean": "avg_mean",
                "std": "std_mean",
                "min": "best_mean",
                "max": "worst_mean",
                "count": "n_variants",
            }
        )
        .sort_values("avg_mean", ascending=ascending)
        .reset_index()
    )
    level2["std_mean"] = level2["std_mean"].fillna(0)
    level2["model_rank"] = range(1, len(level2) + 1)

    level3 = summary.copy()
    return level1, level2, level3


# - Tables --------------------------------------------------------------------


def save_tables(
    summary: pd.DataFrame,
    level1: pd.DataFrame,
    level2: pd.DataFrame,
    best_trials_df: pd.DataFrame,
    timing_df: pd.DataFrame,
    output_dir: Path,
    metrics: list[str],
    metric_scope: str = DEFAULT_METRIC_SCOPE,
    top_n: int = 10,
) -> Dict[str, Path]:
    tables_dir = output_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    paths: Dict[str, Path] = {}

    t1_cols = [
        "variant_rank",
        "variant",
        "avg_mean",
        "std_mean",
        "best_mean",
        "worst_mean",
        "n_models",
    ]
    t1_path = tables_dir / f"{scoped_stem('variant_mean_summary', metric_scope)}.csv"
    level1[[c for c in t1_cols if c in level1.columns]].to_csv(t1_path, index=False)
    paths["T1"] = t1_path

    t2_cols = [
        "model_rank",
        "model",
        "avg_mean",
        "std_mean",
        "best_mean",
        "worst_mean",
        "n_variants",
    ]
    t2_path = tables_dir / f"{scoped_stem('model_mean_summary', metric_scope)}.csv"
    level2[[c for c in t2_cols if c in level2.columns]].to_csv(t2_path, index=False)
    paths["T2"] = t2_path

    t3_cols: list[str] = ["overall_rank", "variant", "model"]
    for metric in metrics:
        t3_cols.extend(
            [
                f"{metric}_mean",
                f"{metric}_std",
                f"{metric}_ci_lower",
                f"{metric}_ci_upper",
            ]
        )
    t3_path = tables_dir / f"{scoped_stem('overall_mean_ranking', metric_scope)}.csv"
    summary[[c for c in t3_cols if c in summary.columns]].to_csv(t3_path, index=False)
    paths["T3"] = t3_path

    top_configs = (
        summary.head(top_n)[["variant", "model"]].values.tolist()
        if not summary.empty
        else []
    )
    for metric in metrics:
        rows: list[dict] = []
        for variant, model in top_configs:
            match = best_trials_df[
                (best_trials_df["variant"] == variant)
                & (best_trials_df["model"] == model)
            ]
            if match.empty:
                continue
            trial = match.iloc[0]
            row: dict = {"variant": variant, "model": model}
            for fold, well in zip(FOLD_IDS, WELL_LABELS):
                row[f"Well_{well}"] = trial.get(f"fold_{fold}_{metric}", np.nan)
            rows.append(row)

        if rows:
            t5_df = pd.DataFrame(rows)
            metric_tag = metric.replace("_", "")
            t5_path = tables_dir / f"{scoped_stem(f'per_well_{metric_tag}', metric_scope)}.csv"
            t5_df.to_csv(t5_path, index=False)
            paths[f"T5_{metric}"] = t5_path

    if not timing_df.empty:
        t6_path = tables_dir / f"{scoped_stem('runtime_summary', metric_scope)}.csv"
        timing_df.sort_values(["variant", "model"]).reset_index(drop=True).to_csv(
            t6_path, index=False
        )
        paths["T6"] = t6_path

    for tag, path in paths.items():
        print(f"  [{tag}] {path}")
    return paths


# - Figure helpers -------------------------------------------------------------


def _short_label(variant: str) -> str:
    return (
        variant.replace("without_outlier_", "WO-")
        .replace("with_outlier_default_", "W-")
        .replace("adaptive", "A")
        .replace("linear", "L")
    )


def _set_style() -> None:
    sns.set_theme(style="whitegrid", font_scale=1.05)
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "font.family": "serif",
        }
    )


def _save_fig(fig: plt.Figure, fig_dir: Path, stem: str) -> None:
    """Save fig as PDF, SVG, and PNG, then close it."""
    for subdir in ("pdf", "svg", "png"):
        (fig_dir / subdir).mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_dir / "pdf" / f"{stem}.pdf")
    fig.savefig(fig_dir / "svg" / f"{stem}.svg")
    fig.savefig(fig_dir / "png" / f"{stem}.png")
    plt.close(fig)


def _fold_long_df(
    best_trials_df: pd.DataFrame,
    configs: list[list[str]],
    metric: str,
) -> pd.DataFrame:
    rows: list[dict] = []
    for variant, model in configs:
        match = best_trials_df[
            (best_trials_df["variant"] == variant)
            & (best_trials_df["model"] == model)
        ]
        if match.empty:
            continue
        trial = match.iloc[0]
        label = f"{model} ({_short_label(variant)})"
        for fold, well in zip(FOLD_IDS, WELL_LABELS):
            value = trial.get(f"fold_{fold}_{metric}", np.nan)
            if pd.notna(value):
                rows.append(
                    {
                        "variant": variant,
                        "model": model,
                        "label": label,
                        "well": well,
                        "value": value,
                    }
                )
    return pd.DataFrame(rows)


# - Figures -------------------------------------------------------------------


def save_figures(
    ci_df: pd.DataFrame,
    summary: pd.DataFrame,
    level1: pd.DataFrame,
    level2: pd.DataFrame,
    best_trials_df: pd.DataFrame,
    output_dir: Path,
    metrics: list[str],
    diagnostic_metric: str = "RMSE_log",
    metric_scope: str = DEFAULT_METRIC_SCOPE,
    top_n: int = 10,
) -> Dict[str, Path]:
    _set_style()
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    paths: Dict[str, Path] = {}

    n_show = min(top_n, len(summary))
    top_overall = (
        summary.head(n_show)[["variant", "model"]].values.tolist()
        if not summary.empty
        else []
    )

    for metric in metrics:
        metric_tag = metric.replace("_", "")
        mean_col = f"{metric}_mean"
        std_col = f"{metric}_std"
        if mean_col not in summary.columns:
            continue

        ascending = metric in LOWER_IS_BETTER
        metric_sorted = summary.sort_values(mean_col, ascending=ascending).head(n_show)
        metric_sorted = metric_sorted.copy()
        metric_sorted["label"] = (
            metric_sorted["model"] + " (" + metric_sorted["variant"].map(_short_label) + ")"
        )

        # Per-metric boxplot (Configuration Selection; same style as other levels)
        try:
            metric_configs = metric_sorted[["variant", "model"]].values.tolist()
            long_df = _fold_long_df(best_trials_df, metric_configs, metric)
            if not long_df.empty:
                # Left-to-right order follows mean ranking (best -> worst).
                label_order = metric_sorted["label"].tolist()
                fig, ax = plt.subplots(
                    figsize=(max(9, len(label_order) * 0.8), 6)
                )
                sns.boxplot(
                    data=long_df,
                    x="label",
                    y="value",
                    order=label_order,
                    palette=sns.color_palette("Set3", len(label_order)),
                    width=0.55,
                    linewidth=0.8,
                    fliersize=0,
                    showmeans=True,
                    meanline=True,
                    meanprops=BOXPLOT_MEAN_PROPS,
                    ax=ax,
                )
                sns.stripplot(
                    data=long_df,
                    x="label",
                    y="value",
                    order=label_order,
                    color="black",
                    dodge=False,
                    size=5,
                    alpha=0.75,
                    ax=ax,
                )
                ax.set_xlabel("")
                ax.set_ylabel(metric)
                ax.set_title(
                    f"Top {len(label_order)} - {metric} "
                    f"(sorted by mean, best at left; red line = mean)"
                )
                ax.set_xticks(range(len(label_order)))
                ax.set_xticklabels(label_order, rotation=45, ha="right")
                stem = scoped_stem(f"boxplot_{metric_tag}", metric_scope)
                _save_fig(fig, fig_dir, stem)
                paths[f"boxplot_{metric}"] = fig_dir / "pdf" / f"{stem}.pdf"
        except Exception as exc:
            print(f"  Warning [boxplot {metric}]: {exc}")

        # Per-well heatmap (for top overall configs)
        try:
            hm_rows: list[dict] = []
            hm_labels: list[str] = []
            for variant, model in top_overall:
                match = best_trials_df[
                    (best_trials_df["variant"] == variant)
                    & (best_trials_df["model"] == model)
                ]
                if match.empty:
                    continue
                trial = match.iloc[0]
                hm_rows.append(
                    {
                        f"Well {well}": trial.get(f"fold_{fold}_{metric}", np.nan)
                        for fold, well in zip(FOLD_IDS, WELL_LABELS)
                    }
                )
                hm_labels.append(f"{model} ({_short_label(variant)})")

            if hm_rows:
                hm_df = pd.DataFrame(hm_rows, index=hm_labels)
                avg_row = hm_df.mean(axis=0)
                avg_row.name = "Well Average"
                hm_df = pd.concat([hm_df, avg_row.to_frame().T])
                cmap = "RdYlGn_r" if metric in LOWER_IS_BETTER else "RdYlGn"

                fig, ax = plt.subplots(figsize=(9, max(3.5, len(hm_df) * 0.55)))
                sns.heatmap(
                    hm_df,
                    annot=True,
                    fmt=".3f",
                    cmap=cmap,
                    linewidths=0.5,
                    cbar_kws={"label": metric},
                    ax=ax,
                )
                ax.set_title(f"Per-Well {metric} - Top Configurations")
                ax.set_ylabel("")
                ax.xaxis.tick_top()
                ax.xaxis.set_label_position("top")
                ax.tick_params(
                    axis="x",
                    rotation=0,
                    bottom=False,
                    top=True,
                    labelbottom=False,
                    labeltop=True,
                )
                stem = scoped_stem(f"per_well_heatmap_{metric_tag}", metric_scope)
                _save_fig(fig, fig_dir, stem)
                paths[f"heatmap_{metric}"] = fig_dir / "pdf" / f"{stem}.pdf"
        except Exception as exc:
            print(f"  Warning [heatmap {metric}]: {exc}")

        # Mean vs std scatter
        try:
            if mean_col not in ci_df.columns or std_col not in ci_df.columns:
                continue

            fig, ax = plt.subplots(figsize=(9, 7))
            variants = sorted(ci_df["variant"].unique())
            palette = sns.color_palette("Set2", len(variants))
            for idx, variant in enumerate(variants):
                subset = ci_df[ci_df["variant"] == variant]
                ax.scatter(
                    subset[mean_col],
                    subset[std_col],
                    label=variant,
                    color=palette[idx],
                    s=60,
                    edgecolors="black",
                    linewidth=0.5,
                    alpha=0.8,
                )

            for _, row in metric_sorted.head(5).iterrows():
                std_values = ci_df.loc[
                    (ci_df["variant"] == row["variant"])
                    & (ci_df["model"] == row["model"]),
                    std_col,
                ]
                if std_values.empty:
                    continue
                ax.annotate(
                    row["model"],
                    (row[mean_col], std_values.iloc[0]),
                    fontsize=8,
                    ha="left",
                    va="bottom",
                    xytext=(4, 4),
                    textcoords="offset points",
                )

            ax.set_xlabel(f"Mean {metric}")
            ax.set_ylabel(f"Std {metric}")
            ax.set_title(f"Performance vs Consistency ({metric})")
            ax.legend(fontsize=8, title="Variant")

            stem = scoped_stem(f"mean_vs_std_{metric_tag}", metric_scope)
            _save_fig(fig, fig_dir, stem)
            paths[f"scatter_{metric}"] = fig_dir / "pdf" / f"{stem}.pdf"
        except Exception as exc:
            print(f"  Warning [scatter {metric}]: {exc}")

    # Diagnostic model robustness boxplot (mean only)
    try:
        diag_mean_col = f"{diagnostic_metric}_mean"
        model_order = level2.sort_values(
            "avg_mean",
            ascending=(diagnostic_metric in LOWER_IS_BETTER),
        )["model"].tolist()
        if model_order and diag_mean_col in summary.columns:
            fig, ax = plt.subplots(figsize=(max(8, len(model_order) * 0.7), 6))
            sns.boxplot(
                data=summary,
                x="model",
                y=diag_mean_col,
                order=model_order,
                palette=sns.color_palette("colorblind", len(model_order)),
                width=0.55,
                linewidth=0.8,
                fliersize=0,
                showmeans=True,
                meanline=True,
                meanprops=BOXPLOT_MEAN_PROPS,
                ax=ax,
            )
            sns.stripplot(
                data=summary,
                x="model",
                y=diag_mean_col,
                order=model_order,
                color="black",
                size=5,
                alpha=0.75,
                ax=ax,
            )
            direction = "lower = better" if diagnostic_metric in LOWER_IS_BETTER else "higher = better"
            ax.set_xlabel("")
            ax.set_ylabel(f"{diagnostic_metric} mean ({direction})")
            ax.set_title("Cross-Variant Model Robustness (red line = mean)")
            ax.set_xticks(range(len(model_order)))
            ax.set_xticklabels(model_order, rotation=45, ha="right")

            stem = scoped_stem("model_robustness", metric_scope)
            _save_fig(fig, fig_dir, stem)
            paths["model_boxplot"] = fig_dir / "pdf" / f"{stem}.pdf"
    except Exception as exc:
        print(f"  Warning [model_boxplot]: {exc}")

    # Diagnostic variant boxplot — all models
    try:
        diag_mean_col = f"{diagnostic_metric}_mean"
        variant_order = level1.sort_values(
            "avg_mean",
            ascending=(diagnostic_metric in LOWER_IS_BETTER),
        )["variant"].tolist()
        if variant_order and diag_mean_col in summary.columns:
            fig, ax = plt.subplots(figsize=(9, 5))
            sns.boxplot(
                data=summary,
                x="variant",
                y=diag_mean_col,
                order=variant_order,
                palette="Set3",
                width=0.55,
                linewidth=0.8,
                fliersize=0,
                showmeans=True,
                meanline=True,
                meanprops=BOXPLOT_MEAN_PROPS,
                ax=ax,
            )
            sns.stripplot(
                data=summary,
                x="variant",
                y=diag_mean_col,
                order=variant_order,
                color="black",
                size=5,
                alpha=0.75,
                ax=ax,
            )
            direction = "lower = better" if diagnostic_metric in LOWER_IS_BETTER else "higher = better"
            ax.set_xlabel("Dataset Variant")
            ax.set_ylabel(f"{diagnostic_metric} mean ({direction})")
            ax.set_title(
                f"Cross Dataset Variant — All Models — {diagnostic_metric} "
                f"(red line = mean)"
            )
            ax.tick_params(axis="x", rotation=30)

            stem = scoped_stem("variant_boxplot_all", metric_scope)
            _save_fig(fig, fig_dir, stem)
            paths["variant_boxplot_all"] = fig_dir / "pdf" / f"{stem}.pdf"
    except Exception as exc:
        print(f"  Warning [variant_boxplot_all]: {exc}")

    # Diagnostic variant boxplot — top 8 models per variant
    try:
        diag_mean_col = f"{diagnostic_metric}_mean"
        ascending = diagnostic_metric in LOWER_IS_BETTER
        top_k = 8
        top_per_variant = (
            summary.sort_values(diag_mean_col, ascending=ascending)
            .groupby("variant")
            .head(top_k)
        )
        variant_order_top = (
            top_per_variant.groupby("variant")[diag_mean_col]
            .mean()
            .sort_values(ascending=ascending)
            .index.tolist()
        )
        if variant_order_top and diag_mean_col in top_per_variant.columns:
            fig, ax = plt.subplots(figsize=(9, 5))
            sns.boxplot(
                data=top_per_variant,
                x="variant",
                y=diag_mean_col,
                order=variant_order_top,
                palette="Set3",
                width=0.55,
                linewidth=0.8,
                fliersize=0,
                showmeans=True,
                meanline=True,
                meanprops=BOXPLOT_MEAN_PROPS,
                ax=ax,
            )
            sns.stripplot(
                data=top_per_variant,
                x="variant",
                y=diag_mean_col,
                order=variant_order_top,
                color="black",
                size=5,
                alpha=0.75,
                ax=ax,
            )
            direction = "lower = better" if diagnostic_metric in LOWER_IS_BETTER else "higher = better"
            ax.set_xlabel("Dataset Variant")
            ax.set_ylabel(f"{diagnostic_metric} mean ({direction})")
            ax.set_title(
                f"Cross Dataset Variant — Top {top_k} Models — {diagnostic_metric} "
                f"(red line = mean)"
            )
            ax.tick_params(axis="x", rotation=30)

            stem = scoped_stem("variant_boxplot_top8", metric_scope)
            _save_fig(fig, fig_dir, stem)
            paths["variant_boxplot_top8"] = fig_dir / "pdf" / f"{stem}.pdf"
    except Exception as exc:
        print(f"  Warning [variant_boxplot_top8]: {exc}")

    for tag, path in paths.items():
        print(f"  [{tag}] {path}")
    return paths


# - Markdown report ------------------------------------------------------------


def _md_table(df: pd.DataFrame, columns: list[str], fmt: str = ".3f") -> str:
    """Render a DataFrame as a markdown table."""
    if df.empty:
        return "_No data available._"

    available = [col for col in columns if col in df.columns]
    if not available:
        return "_No columns available._"

    header = "| " + " | ".join(available) + " |"
    sep = "| " + " | ".join("---" for _ in available) + " |"
    rows = [header, sep]

    for _, record in df.iterrows():
        cells: list[str] = []
        for col in available:
            value = record[col]
            if pd.isna(value):
                cells.append("—")
            elif isinstance(value, float):
                cells.append(f"{value:{fmt}}")
            else:
                cells.append(str(value))
        rows.append("| " + " | ".join(cells) + " |")

    return "\n".join(rows)


def generate_report(
    summary: pd.DataFrame,
    level1: pd.DataFrame,
    level2: pd.DataFrame,
    ci_df: pd.DataFrame,
    config: dict,
    alpha: float,
    table_paths: Dict[str, Path],
    figure_paths: Dict[str, Path],
    metrics: list[str],
    diagnostic_metric: str = "RMSE_log",
    metric_scope: str = DEFAULT_METRIC_SCOPE,
) -> str:
    t_crit = stats.t.ppf(1 - alpha / 2, df=N_FOLDS - 1)
    best_model = summary.iloc[0]["model"] if not summary.empty else "N/A"
    best_variant = summary.iloc[0]["variant"] if not summary.empty else "N/A"
    l1_winner = level1.iloc[0]["variant"] if not level1.empty else "N/A"

    lines = [
        "# Phase 2 Model Selection - Mean Performance Analysis",
        "",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Methodology",
        "",
        f"- **Selection criterion**: mean cross-fold `{diagnostic_metric}`",
        "- **Primary visual**: boxplot + fold points (LOWO wells) per metric",
        f"- **Metric scope**: `{metric_scope}`",
        f"- **Confidence level (descriptive)**: {100 * (1 - alpha):.0f}%",
        f"- **t-critical (df={N_FOLDS - 1})**: {t_crit:.3f}",
        "- **CI role**: CI columns are descriptive only and do not drive ranking",
        "",
        "**Selection rule**: configurations are sorted by diagnostic mean, and the",
        "top-ranked configuration is promoted to Phase 3.",
        "",
        "## Dataset Overview",
        "",
        f"- **Configurations**: {len(summary)} "
        f"({summary['variant'].nunique() if not summary.empty else 0} variants x "
        f"{summary['model'].nunique() if not summary.empty else 0} models)",
        f"- **Folds (LOWO CV)**: {N_FOLDS}",
        "",
        "---",
        "",
        "## Level 1: Dataset Variant View",
        "",
        f"Best variant (best average mean): **{l1_winner}**",
        "",
        _md_table(
            level1,
            ["variant_rank", "variant", "avg_mean", "std_mean", "n_models"],
        ),
        "",
        f"**All Models:** see `{scoped_stem('variant_boxplot_all', metric_scope)}.png`",
        "",
        f"**Top 8 Models Per Variant:** see `{scoped_stem('variant_boxplot_top8', metric_scope)}.png`",
        "",
        "---",
        "",
        "## Level 2: Model Robustness",
        "",
        _md_table(
            level2,
            ["model_rank", "model", "avg_mean", "std_mean", "n_variants"],
        ),
        "",
        "---",
        "",
        "## Level 3: Configuration Selection",
        "",
        f"Best configuration: **{best_model}** on `{best_variant}`",
        "",
    ]

    l3_cols: list[str] = ["overall_rank", "variant", "model"]
    l3_cols.extend([f"{metric}_mean" for metric in metrics])
    l3_cols.extend([f"{metric}_ci_lower" for metric in metrics])
    l3_cols.extend([f"{metric}_ci_upper" for metric in metrics])
    lines.append(_md_table(summary.head(15), l3_cols, fmt=".4f"))

    lines.extend(
        [
            "",
            "---",
            "",
            "## Output Files",
            "",
            "### Tables",
            "",
        ]
    )
    for tag, path in sorted(table_paths.items()):
        lines.append(f"- **{tag}**: `{path}`")

    lines.extend(["", "### Figures", ""])
    for tag, path in sorted(figure_paths.items()):
        lines.append(f"- **{tag}**: `{path}`")

    lines.append("")
    return "\n".join(lines)


# - HTML report ----------------------------------------------------------------


def _img_base64_tag(fig_dir: Path, stem: str) -> str:
    """Read a saved PNG and return an HTML <img> tag with base64 data URI."""
    import base64

    png_path = fig_dir / "png" / f"{stem}.png"
    if not png_path.exists():
        return (
            '<p class="text-muted fst-italic">'
            "Figure not available (run without --no-figures)."
            "</p>"
        )
    encoded = base64.b64encode(png_path.read_bytes()).decode()
    return (
        f'<img src="data:image/png;base64,{encoded}" '
        f'class="img-fluid rounded shadow-sm" alt="{stem}">'
    )


def _html_table(
    df: pd.DataFrame,
    columns: Optional[list[str]] = None,
    fmt: str = ".3f",
) -> str:
    """Render a DataFrame as a Bootstrap-styled HTML table."""
    if df.empty:
        return '<p class="text-muted">No data available.</p>'

    selected = [col for col in (columns or list(df.columns)) if col in df.columns]
    if not selected:
        return '<p class="text-muted">No columns available.</p>'

    lines = [
        '<div class="table-responsive">',
        '<table class="table table-striped table-hover table-sm align-middle">',
        '<thead class="table-dark"><tr>',
    ]
    for col in selected:
        lines.append(f"<th>{col}</th>")
    lines.append("</tr></thead><tbody>")

    for _, record in df.iterrows():
        lines.append("<tr>")
        for col in selected:
            value = record[col]
            if pd.isna(value):
                lines.append('<td class="text-muted">&mdash;</td>')
            elif isinstance(value, float):
                lines.append(f"<td>{value:{fmt}}</td>")
            else:
                lines.append(f"<td>{value}</td>")
        lines.append("</tr>")

    lines.append("</tbody></table></div>")
    return "\n".join(lines)


def generate_html_report(
    summary: pd.DataFrame,
    level1: pd.DataFrame,
    level2: pd.DataFrame,
    ci_df: pd.DataFrame,
    best_trials_df: pd.DataFrame,
    timing_df: pd.DataFrame,
    alpha: float,
    output_dir: Path,
    metrics: list[str],
    diagnostic_metric: str = "RMSE_log",
    metric_scope: str = DEFAULT_METRIC_SCOPE,
    top_n: int = 10,
) -> Path:
    """Generate a self-contained HTML report with embedded figures."""
    fig_dir = output_dir / "figures"
    t_crit = stats.t.ppf(1 - alpha / 2, df=N_FOLDS - 1)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    best_model = summary.iloc[0]["model"] if not summary.empty else "N/A"
    best_variant = summary.iloc[0]["variant"] if not summary.empty else "N/A"
    l1_winner = level1.iloc[0]["variant"] if not level1.empty else "N/A"

    winner_cards = ""
    if not summary.empty:
        winner = summary.iloc[0]
        cards: list[str] = []
        for metric in metrics:
            mean_col = f"{metric}_mean"
            if mean_col in winner.index and pd.notna(winner[mean_col]):
                cards.append(
                    f'<div class="col"><div class="card text-center h-100">'
                    f'<div class="card-body py-2">'
                    f'<h6 class="card-subtitle text-muted mb-1">{metric}</h6>'
                    f'<span class="fs-5 fw-bold">{winner[mean_col]:.4f}</span>'
                    "</div></div></div>"
                )
        winner_cards = (
            '<div class="row row-cols-2 row-cols-md-4 g-2 mt-2">'
            + "".join(cards)
            + "</div>"
        )

    l1_table = _html_table(
        level1,
        ["variant_rank", "variant", "avg_mean", "std_mean", "best_mean", "worst_mean", "n_models"],
    )
    l2_table = _html_table(
        level2,
        ["model_rank", "model", "avg_mean", "std_mean", "best_mean", "worst_mean", "n_variants"],
    )
    l3_cols = (
        ["overall_rank", "variant", "model"]
        + [f"{metric}_mean" for metric in metrics]
        + [f"{metric}_ci_lower" for metric in metrics]
        + [f"{metric}_ci_upper" for metric in metrics]
    )
    l3_table = _html_table(summary.head(top_n), l3_cols, fmt=".4f")

    runtime_html = (
        _html_table(
            timing_df.sort_values(["variant", "model"]).reset_index(drop=True),
            fmt=".1f",
        )
        if not timing_df.empty
        else '<p class="text-muted">No runtime data available.</p>'
    )

    boxplot_sections = ""
    perwell_sections = ""
    scatter_sections = ""
    for metric in metrics:
        metric_tag = metric.replace("_", "")
        boxplot_img = _img_base64_tag(fig_dir, scoped_stem(f"boxplot_{metric_tag}", metric_scope))
        perwell_img = _img_base64_tag(
            fig_dir, scoped_stem(f"per_well_heatmap_{metric_tag}", metric_scope)
        )
        scatter_img = _img_base64_tag(
            fig_dir, scoped_stem(f"mean_vs_std_{metric_tag}", metric_scope)
        )
        boxplot_sections += (
            f'<h5 class="mt-4">{metric}</h5>\n'
            f'<div class="fig">{boxplot_img}</div>\n'
        )
        perwell_sections += (
            f'<h5 class="mt-4">{metric}</h5>\n'
            f'<div class="fig">{perwell_img}</div>\n'
        )
        scatter_sections += (
            f'<h5 class="mt-4">{metric}</h5>\n'
            f'<div class="fig">{scatter_img}</div>\n'
        )

    fig_model = _img_base64_tag(fig_dir, scoped_stem("model_robustness", metric_scope))
    fig_variant_all = _img_base64_tag(fig_dir, scoped_stem("variant_boxplot_all", metric_scope))
    fig_variant_top8 = _img_base64_tag(fig_dir, scoped_stem("variant_boxplot_top8", metric_scope))

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Phase 2 - Mean/Boxplot Model Selection</title>
<link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }}
  .nav-tabs .nav-link {{ color: #495057; }}
  .nav-tabs .nav-link.active {{ font-weight: 600; }}
  .card-winner {{ border-left: 4px solid #198754; background: #f8f9fa; }}
  .fig {{ text-align: center; margin: 1.5rem 0; }}
  .fig img {{ max-width: 100%; height: auto; }}
  .sec {{ border-bottom: 2px solid #dee2e6; padding-bottom: .5rem; margin-bottom: 1rem; }}
  .tab-pane {{ padding-top: 1.5rem; }}
  th {{ white-space: nowrap; }}
  .table {{ font-size: .88rem; }}
</style>
</head>
<body>
<nav class="navbar navbar-dark bg-dark mb-4">
  <div class="container-fluid">
    <span class="navbar-brand mb-0 h1">Phase 2 - Mean/Boxplot Selection</span>
    <span class="navbar-text text-light">{timestamp}</span>
  </div>
</nav>
<div class="container-fluid px-4">

<ul class="nav nav-tabs" id="tabs" role="tablist">
  <li class="nav-item" role="presentation"><button class="nav-link active" data-bs-toggle="tab" data-bs-target="#overview" type="button">Overview</button></li>
  <li class="nav-item" role="presentation"><button class="nav-link" data-bs-toggle="tab" data-bs-target="#variants" type="button">Dataset Variant View</button></li>
  <li class="nav-item" role="presentation"><button class="nav-link" data-bs-toggle="tab" data-bs-target="#robust" type="button">Model Robustness</button></li>
  <li class="nav-item" role="presentation"><button class="nav-link" data-bs-toggle="tab" data-bs-target="#select" type="button">Selection</button></li>
  <li class="nav-item" role="presentation"><button class="nav-link" data-bs-toggle="tab" data-bs-target="#perwell" type="button">Per-Well</button></li>
  <li class="nav-item" role="presentation"><button class="nav-link" data-bs-toggle="tab" data-bs-target="#diag" type="button">Diagnostics</button></li>
</ul>

<div class="tab-content" id="tabContent">
  <div class="tab-pane fade show active" id="overview" role="tabpanel">
    <h4 class="sec">Overview</h4>
    <div class="row mb-3">
      <div class="col-md-6">
        <div class="card card-winner p-3">
          <h5 class="mb-1">Winner</h5>
          <p class="mb-1"><strong>Model:</strong> {best_model}</p>
          <p class="mb-0"><strong>Variant:</strong> <code>{best_variant}</code></p>
        </div>
      </div>
      <div class="col-md-6">
        <div class="card p-3 h-100">
          <h6>Methodology</h6>
          <ul class="mb-0 small">
            <li>Selection: mean {diagnostic_metric} across LOWO folds</li>
            <li>CI values are descriptive (not ranking criteria)</li>
            <li>Confidence: {100 * (1 - alpha):.0f}% (t = {t_crit:.3f}, df = {N_FOLDS - 1})</li>
            <li>Configs: {len(summary)} ({summary['variant'].nunique() if not summary.empty else 0} variants × {summary['model'].nunique() if not summary.empty else 0} models)</li>
            <li>Metric scope: {metric_scope}</li>
          </ul>
        </div>
      </div>
    </div>
    <h5>Winner Metrics (mean across folds)</h5>
    {winner_cards}
  </div>

  <div class="tab-pane fade" id="variants" role="tabpanel">
    <h4 class="sec">Dataset Variant View</h4>
    <p>Dataset variants by average diagnostic mean across models. Best: <strong>{l1_winner}</strong></p>
    {l1_table}
    <h5 class="mt-4">All Models</h5>
    <div class="fig">{fig_variant_all}</div>
    <h5 class="mt-4">Top 8 Models Per Variant</h5>
    <div class="fig">{fig_variant_top8}</div>
  </div>

  <div class="tab-pane fade" id="robust" role="tabpanel">
    <h4 class="sec">Model Robustness</h4>
    {l2_table}
    <div class="fig">{fig_model}</div>
  </div>

  <div class="tab-pane fade" id="select" role="tabpanel">
    <h4 class="sec">Configuration Selection</h4>
    <p>Best: <strong>{best_model}</strong> on <code>{best_variant}</code></p>
    {l3_table}
    <h4 class="sec mt-4">Per-Metric Boxplots (top ranked configurations)</h4>
{boxplot_sections}
  </div>

  <div class="tab-pane fade" id="perwell" role="tabpanel">
    <h4 class="sec">Per-Well Performance (all metrics)</h4>
{perwell_sections}
  </div>

  <div class="tab-pane fade" id="diag" role="tabpanel">
    <h4 class="sec">Diagnostics</h4>
    <h5 class="mt-3">Performance vs Consistency (mean vs std)</h5>
{scatter_sections}
    <h5 class="mt-4">Runtime Summary</h5>
    {runtime_html}
  </div>
</div>
</div>
<footer class="text-center text-muted py-3 mt-4 border-top">
  <small>Generated by analyze_phase2.py - {timestamp}</small>
</footer>
<script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
</body>
</html>"""

    html_path = output_dir / f"{scoped_stem('phase2_report', metric_scope)}.html"
    html_path.write_text(html, encoding="utf-8")
    return html_path


# - Main pipeline --------------------------------------------------------------


def run_analysis(
    results_dir: str | Path | None = None,
    alpha: float = 0.05,
    generate_figs: bool = True,
    generate_html: bool = False,
    top_n_models: int = 10,
    metric_scope: str = DEFAULT_METRIC_SCOPE,
    verbose: bool = True,
) -> dict:
    """Run the full Phase 2 analysis pipeline."""
    if results_dir is None:
        results_dir = (
            Path(__file__).parent.parent / "results" / "phase2_model_selection"
        )
    else:
        results_dir = Path(results_dir)

    output_dir = results_dir / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    active_metrics, _primary_metrics, diagnostic_metric = resolve_metric_scope(
        metric_scope
    )

    config: dict = {}
    try:
        config = load_experiment_config()
    except FileNotFoundError:
        if verbose:
            print("Warning: Could not load experiment_config.json")

    if verbose:
        print("Loading best trials...")
    best_trials_df = load_best_trials(results_dir)
    if best_trials_df.empty:
        print("No results found. Aborting.")
        return {}

    if verbose:
        n_variants = best_trials_df["variant"].nunique()
        n_models = best_trials_df["model"].nunique()
        print(f"  {len(best_trials_df)} configs ({n_variants} variant(s) x {n_models} model(s))")
        print(f"  Metric scope: {metric_scope}")

    if verbose:
        print(f"Computing {100 * (1 - alpha):.0f}% CI statistics (descriptive only)...")
    ci_df = compute_ci_bounds(best_trials_df, alpha=alpha, metrics=active_metrics)
    timing_df = load_timing_info(results_dir)

    if verbose:
        print("Building mean-based summaries...")
    summary = build_summary(ci_df, metrics=active_metrics, diagnostic_metric=diagnostic_metric)
    level1, level2, level3 = three_level_summary(summary, diagnostic_metric=diagnostic_metric)

    best_variant = summary.iloc[0]["variant"] if not summary.empty else "N/A"
    best_model = summary.iloc[0]["model"] if not summary.empty else "N/A"
    if verbose:
        l1_winner = level1.iloc[0]["variant"] if not level1.empty else "N/A"
        l2_top = level2.iloc[0]["model"] if not level2.empty else "N/A"
        print(f"  Level 1 winner: {l1_winner}")
        print(f"  Level 2 top:    {l2_top}")
        print(f"  Level 3 winner: {best_model} on {best_variant}")

    if verbose:
        print("\nSaving tables...")
    table_paths = save_tables(
        summary,
        level1,
        level2,
        best_trials_df,
        timing_df,
        output_dir,
        metrics=active_metrics,
        metric_scope=metric_scope,
        top_n=top_n_models,
    )

    figure_paths: Dict[str, Path] = {}
    if generate_figs:
        if verbose:
            print("\nGenerating figures...")
        figure_paths = save_figures(
            ci_df,
            summary,
            level1,
            level2,
            best_trials_df,
            output_dir,
            metrics=active_metrics,
            diagnostic_metric=diagnostic_metric,
            metric_scope=metric_scope,
            top_n=top_n_models,
        )

    if verbose:
        print("\nGenerating markdown report...")
    report_text = generate_report(
        summary,
        level1,
        level2,
        ci_df,
        config,
        alpha,
        table_paths,
        figure_paths,
        metrics=active_metrics,
        diagnostic_metric=diagnostic_metric,
        metric_scope=metric_scope,
    )
    report_path = output_dir / f"{scoped_stem('phase2_report', metric_scope)}.md"
    report_path.write_text(report_text, encoding="utf-8")
    if verbose:
        print(f"  Report: {report_path}")

    html_path = None
    if generate_html:
        if verbose:
            print("\nGenerating HTML report...")
        html_path = generate_html_report(
            summary,
            level1,
            level2,
            ci_df,
            best_trials_df,
            timing_df,
            alpha,
            output_dir,
            metrics=active_metrics,
            diagnostic_metric=diagnostic_metric,
            metric_scope=metric_scope,
            top_n=top_n_models,
        )
        if verbose:
            print(f"  HTML:   {html_path}")

    if verbose:
        print("\nDone.")

    return {
        "best_trials_df": best_trials_df,
        "ci_df": ci_df,
        "summary": summary,
        "level1": level1,
        "level2": level2,
        "level3": level3,
        "table_paths": table_paths,
        "figure_paths": figure_paths,
        "report_path": report_path,
        "html_path": html_path,
    }


# - CLI -----------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 2 mean-based analysis with boxplot reporting.",
    )
    parser.add_argument(
        "--results_dir",
        "-r",
        type=str,
        default=None,
        help="Results directory. Default: results/phase2_model_selection",
    )
    parser.add_argument(
        "--alpha",
        "-a",
        type=float,
        default=0.05,
        help="Significance level for descriptive CI columns (default: 0.05).",
    )
    parser.add_argument(
        "--no-figures",
        action="store_true",
        help="Skip figure generation.",
    )
    parser.add_argument(
        "--html",
        action="store_true",
        help="Generate a self-contained HTML report with embedded figures.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=10,
        help="Number of top configurations used in plots/tables (default: 10).",
    )
    parser.add_argument(
        "--metric-scope",
        choices=METRIC_SCOPE_CHOICES,
        default=DEFAULT_METRIC_SCOPE,
        help="Metric family for analysis/reporting: original, log, or both (default: both).",
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Suppress verbose output.",
    )
    args = parser.parse_args()

    run_analysis(
        results_dir=args.results_dir,
        alpha=args.alpha,
        generate_figs=not args.no_figures,
        generate_html=args.html,
        top_n_models=args.top_n,
        metric_scope=args.metric_scope,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()

