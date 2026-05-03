#!/usr/bin/env python
"""
Phase 3 Feature Selection — Analysis and Visualization

Post-run analysis of SHAP importance and Pareto sweep results.
Generates figures, summary tables, and a markdown report.

Inputs (from results/phase3_feature_selection/):
  shap/                  : SHAP analysis outputs  (run_phase3_shap.py)
  pareto_sweep/          : Brute-force sweep       (run_phase3_pareto.py --sweep)
  pareto_retune/         : HP re-tuning            (run_phase3_pareto.py --retune)  [optional]
  pareto_validation/     : Validation model check  (run_phase3_pareto.py --validate)[optional]

Outputs (to results/phase3_feature_selection/analysis/):
  figures/{pdf,svg,png}/
    - shap_global_bar.*
    - shap_beeswarm.*
    - shap_per_well_heatmap.*
    - pareto_frontier_{metric}.*   (one per ranking metric)
    - pareto_frontier_retune.*
    - pareto_validation.*
    - feature_frequency_pareto.*
  tables/
    - pareto_optimal_subsets.csv
    - shap_grouped_feature_importance.csv
    - shap_vs_cost.csv
  phase3_report.md

Usage:
    python runners/analyze_phase3.py
    python runners/analyze_phase3.py --results_dir results/phase3_feature_selection
    python runners/analyze_phase3.py --no-figures
    python runners/analyze_phase3.py -q
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.feature_selection import (
    aggregate_shap_importance,
    collapse_shap_columns,
    collapse_shap_feature_name,
    load_phase3_config,
)

RANKING_METRICS = ["RMSE_log", "MAE_log", "R2_log", "RMSE", "MAE", "R2"]
ORIGINAL_RANKING_METRICS = ["RMSE", "MAE", "R2"]
LOG_RANKING_METRICS = ["RMSE_log", "MAE_log", "R2_log"]
METRIC_SCOPE_CHOICES = ("original", "log", "both")
DEFAULT_METRIC_SCOPE = "both"

METRIC_DIRECTIONS = {
    "RMSE_log": "minimize",
    "MAE_log": "minimize",
    "R2_log": "maximize",
    "RMSE": "minimize",
    "MAE": "minimize",
    "R2": "maximize",
}


def resolve_metric_scope(metric_scope: str) -> list[str]:
    scope = metric_scope.lower()
    if scope == "original":
        return ORIGINAL_RANKING_METRICS.copy()
    if scope == "log":
        return LOG_RANKING_METRICS.copy()
    return RANKING_METRICS.copy()


# ── Style / save helpers ─────────────────────────────────────────────────────


def _set_style():
    sns.set_theme(style="whitegrid", font_scale=1.1)
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "font.family": "serif",
    })


def _save_fig(fig: plt.Figure, fig_dir: Path, stem: str) -> None:
    for sub in ("pdf", "svg", "png"):
        (fig_dir / sub).mkdir(exist_ok=True)
    fig.savefig(fig_dir / "pdf" / f"{stem}.pdf")
    fig.savefig(fig_dir / "svg" / f"{stem}.svg")
    fig.savefig(fig_dir / "png" / f"{stem}.png")
    plt.close(fig)


# ── Data loading ─────────────────────────────────────────────────────────────


def load_shap_outputs(results_dir: Path) -> dict:
    shap_dir = results_dir / "shap"
    outputs: dict = {}

    path = shap_dir / "global_importance.csv"
    if path.exists():
        global_df = pd.read_csv(path)
        if {"feature", "mean_abs_shap"}.issubset(global_df.columns):
            global_df = (
                global_df.assign(
                    feature=global_df["feature"].map(collapse_shap_feature_name)
                )
                .groupby("feature", as_index=False)["mean_abs_shap"]
                .sum()
                .sort_values("mean_abs_shap", ascending=False)
                .reset_index(drop=True)
            )
        outputs["global_importance"] = global_df

    path = shap_dir / "per_well_importance.csv"
    if path.exists():
        per_well_df = pd.read_csv(path)
        if {"well", "feature", "mean_abs_shap"}.issubset(per_well_df.columns):
            per_well_df = (
                per_well_df.assign(
                    feature=per_well_df["feature"].map(collapse_shap_feature_name)
                )
                .groupby(["well", "feature"], as_index=False)["mean_abs_shap"]
                .sum()
            )
        outputs["per_well_importance"] = per_well_df

    path = shap_dir / "shap_values.npz"
    if path.exists():
        data = np.load(path, allow_pickle=True)
        raw_shap_values = data["shap_values"]
        raw_feature_names = [str(x) for x in data["feature_names"]]
        raw_fold_labels = data["fold_labels"]
        fold_labels = np.array([
            v.decode("utf-8") if isinstance(v, (bytes, bytearray)) else str(v)
            for v in raw_fold_labels
        ])

        # Collapse transformed columns (e.g., one-hot Zone_*) back to parent
        # features for more interpretable SHAP reporting and figures.
        collapsed_shap, collapsed_feature_names = collapse_shap_columns(
            raw_shap_values, raw_feature_names
        )
        outputs["shap_values"] = collapsed_shap
        outputs["feature_names"] = collapsed_feature_names
        outputs["sample_indices"] = data["sample_indices"]
        outputs["fold_labels"] = fold_labels
        outputs["expected_values"] = data["expected_values"]

        # Recompute global and per-well importance from raw SHAP arrays so
        # analysis reflects collapsed parent features even if legacy CSVs exist.
        shap_entries = []
        for fold_name in sorted(np.unique(fold_labels)):
            fold_mask = fold_labels == fold_name
            if np.any(fold_mask):
                shap_entries.append({
                    "shap_values": raw_shap_values[fold_mask],
                    "fold_name": fold_name,
                })
        if shap_entries:
            global_imp, per_well_imp = aggregate_shap_importance(
                shap_entries, raw_feature_names
            )
            outputs["global_importance"] = global_imp
            outputs["per_well_importance"] = per_well_imp

    path = shap_dir / "shap_run_meta.json"
    if path.exists():
        with open(path, "r") as f:
            outputs["meta"] = json.load(f)

    return outputs


def load_sweep_outputs(results_dir: Path) -> dict:
    sweep_dir = results_dir / "pareto_sweep"
    outputs: dict = {}

    path = sweep_dir / "sweep_results.csv"
    if path.exists():
        outputs["sweep_results"] = pd.read_csv(path)

    outputs["frontiers"] = {}
    for metric in RANKING_METRICS:
        path = sweep_dir / f"pareto_frontier_{metric}.csv"
        if path.exists():
            outputs["frontiers"][metric] = pd.read_csv(path)

    path = sweep_dir / "sweep_meta.json"
    if path.exists():
        with open(path, "r") as f:
            outputs["meta"] = json.load(f)

    return outputs


def load_retune_outputs(results_dir: Path) -> dict:
    retune_dir = results_dir / "pareto_retune"
    outputs: dict = {}

    path = retune_dir / "retune_results.csv"
    if path.exists():
        outputs["retune_results"] = pd.read_csv(path)

    path = retune_dir / "pareto_frontier_retune.csv"
    if path.exists():
        outputs["frontier_retune"] = pd.read_csv(path)

    return outputs


def load_validation_outputs(results_dir: Path) -> dict:
    val_dir = results_dir / "pareto_validation"
    outputs: dict = {}

    path = val_dir / "validation_results.csv"
    if path.exists():
        outputs["validation_results"] = pd.read_csv(path)

    path = val_dir / "validation_meta.json"
    if path.exists():
        with open(path, "r") as f:
            outputs["meta"] = json.load(f)

    return outputs


# ── SHAP figures ─────────────────────────────────────────────────────────────


def fig_shap_global_bar(global_importance: pd.DataFrame, fig_dir: Path) -> None:
    df = global_importance.copy()

    fig, ax = plt.subplots(figsize=(8, max(4, len(df) * 0.35)))

    colors = sns.color_palette("viridis_r", len(df))
    ax.barh(
        range(len(df)),
        df["mean_abs_shap"].values,
        color=colors,
        edgecolor="black",
        linewidth=0.3,
    )
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(df["feature"].values, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Mean |SHAP Value|")
    ax.set_title("Global Feature Importance (Mean |SHAP|)")

    _save_fig(fig, fig_dir, "shap_global_bar")


def fig_shap_beeswarm(
    shap_values: np.ndarray,
    feature_names: list[str],
    fig_dir: Path,
) -> None:
    try:
        import shap

        fig = plt.figure(figsize=(10, max(5, len(feature_names) * 0.35)))
        shap.summary_plot(
            shap_values,
            feature_names=feature_names,
            show=False,
            max_display=min(20, len(feature_names)),
        )
        _save_fig(plt.gcf(), fig_dir, "shap_beeswarm")
    except Exception as e:
        print(f"  Warning [shap_beeswarm]: {e}")
        plt.close("all")


def fig_shap_per_well_heatmap(
    per_well_importance: pd.DataFrame,
    fig_dir: Path,
) -> None:
    pivot = per_well_importance.pivot_table(
        index="feature", columns="well", values="mean_abs_shap",
    )

    feat_order = pivot.mean(axis=1).sort_values(ascending=False).index
    pivot = pivot.loc[feat_order]
    well_order = sorted(pivot.columns)
    pivot = pivot[well_order]

    fig, ax = plt.subplots(
        figsize=(max(7, len(well_order) * 1.2),
                 max(5, len(feat_order) * 0.4)),
    )
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".4f",
        cmap="YlOrRd",
        linewidths=0.5,
        ax=ax,
        cbar_kws={"label": "Mean |SHAP|"},
    )
    ax.set_title("Per-Well Feature Importance (Mean |SHAP|)")
    ax.set_ylabel("")
    ax.set_xlabel("Well (held-out)")
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position("top")
    ax.tick_params(
        axis="x", rotation=0, bottom=False, top=True,
        labelbottom=False, labeltop=True,
    )

    _save_fig(fig, fig_dir, "shap_per_well_heatmap")


# ── Pareto figures ───────────────────────────────────────────────────────────


def _label_frontier_points(ax, frontier_sorted, score_col, max_labels=15):
    """Annotate Pareto frontier points with feature count, avoiding clutter."""
    step = max(1, len(frontier_sorted) // max_labels)
    for i, (_, row) in enumerate(frontier_sorted.iterrows()):
        if i % step != 0 and i != len(frontier_sorted) - 1:
            continue
        n_feat = int(row.get("n_features", 0))
        ax.annotate(
            f"n={n_feat}",
            (row["cost"], row[score_col]),
            fontsize=7,
            ha="left",
            va="bottom",
            xytext=(5, 5),
            textcoords="offset points",
        )


def fig_pareto_frontier(
    sweep_results: pd.DataFrame,
    frontier: pd.DataFrame,
    metric: str,
    fig_dir: Path,
) -> None:
    score_col = f"{metric}_mean"
    direction = METRIC_DIRECTIONS[metric]

    fig, ax = plt.subplots(figsize=(10, 7))

    ax.scatter(
        sweep_results["cost"],
        sweep_results[score_col],
        s=8,
        alpha=0.15,
        color="gray",
        label=f"All subsets (n={len(sweep_results)})",
        rasterized=True,
    )

    frontier_sorted = frontier.sort_values("cost")
    ax.plot(
        frontier_sorted["cost"],
        frontier_sorted[score_col],
        "o-",
        color="crimson",
        linewidth=2,
        markersize=7,
        markeredgecolor="black",
        markeredgewidth=0.5,
        label=f"Pareto frontier (n={len(frontier)})",
        zorder=5,
    )

    _label_frontier_points(ax, frontier_sorted, score_col)

    full_n = sweep_results["n_features"].max()
    full_mask = sweep_results["n_features"] == full_n
    if full_mask.any():
        full_row = sweep_results[full_mask].iloc[0]
        ax.scatter(
            [full_row["cost"]],
            [full_row[score_col]],
            s=120,
            marker="*",
            color="blue",
            edgecolors="black",
            linewidth=0.5,
            zorder=6,
            label="Full feature set",
        )

    better = "lower is better" if direction == "minimize" else "higher is better"
    ax.set_xlabel("Acquisition Cost (surrogate score)")
    ax.set_ylabel(f"{metric} Mean ({better})")
    ax.set_title(f"Pareto Frontier: Cost vs {metric}")
    ax.legend(loc="best", fontsize=9)

    _save_fig(fig, fig_dir, f"pareto_frontier_{metric}")


def fig_pareto_frontier_retune(
    sweep_results: pd.DataFrame,
    sweep_frontier: pd.DataFrame,
    retune_frontier: pd.DataFrame,
    fig_dir: Path,
) -> None:
    score_col = "RMSE_log_mean"

    fig, ax = plt.subplots(figsize=(10, 7))

    ax.scatter(
        sweep_results["cost"],
        sweep_results[score_col],
        s=8,
        alpha=0.15,
        color="gray",
        label="All subsets (sweep)",
        rasterized=True,
    )

    sf = sweep_frontier.sort_values("cost")
    ax.plot(
        sf["cost"], sf[score_col],
        "s--",
        color="steelblue",
        linewidth=1.5,
        markersize=6,
        markeredgecolor="black",
        markeredgewidth=0.3,
        alpha=0.7,
        label="Sweep frontier (fixed HPs)",
        zorder=4,
    )

    rf = retune_frontier.sort_values("cost")
    ax.plot(
        rf["cost"], rf[score_col],
        "o-",
        color="crimson",
        linewidth=2,
        markersize=7,
        markeredgecolor="black",
        markeredgewidth=0.5,
        label="Retune frontier (optimized HPs)",
        zorder=5,
    )

    _label_frontier_points(ax, rf, score_col)

    full_n = sweep_results["n_features"].max()
    full_mask = sweep_results["n_features"] == full_n
    if full_mask.any():
        full_row = sweep_results[full_mask].iloc[0]
        ax.scatter(
            [full_row["cost"]],
            [full_row[score_col]],
            s=120,
            marker="*",
            color="blue",
            edgecolors="black",
            linewidth=0.5,
            zorder=6,
            label="Full feature set",
        )

    ax.set_xlabel("Acquisition Cost")
    ax.set_ylabel("RMSE_log Mean (lower is better)")
    ax.set_title("Pareto Frontier: Sweep vs Re-tuned (RMSE_log)")
    ax.legend(loc="best", fontsize=9)

    _save_fig(fig, fig_dir, "pareto_frontier_retune")


def fig_pareto_validation(
    sweep_results: Optional[pd.DataFrame],
    primary_frontier: pd.DataFrame,
    validation_results: pd.DataFrame,
    config: dict,
    fig_dir: Path,
) -> None:
    score_col = "RMSE_log_mean"
    primary_model = config.get("best_model", "ExtraTrees")
    val_model = config.get("validation_model", "RandomForest")

    fig, ax = plt.subplots(figsize=(10, 7))

    if sweep_results is not None and score_col in sweep_results.columns:
        ax.scatter(
            sweep_results["cost"],
            sweep_results[score_col],
            s=8, alpha=0.1, color="gray",
            label="All subsets (sweep)", rasterized=True,
        )

    pf = primary_frontier.sort_values("cost")
    ax.plot(
        pf["cost"], pf[score_col],
        "o-", color="crimson", linewidth=2, markersize=7,
        markeredgecolor="black", markeredgewidth=0.5,
        label=f"{primary_model} (primary)", zorder=5,
    )

    vr = validation_results.sort_values("cost")
    ax.plot(
        vr["cost"], vr[score_col],
        "s--", color="teal", linewidth=1.5, markersize=6,
        markeredgecolor="black", markeredgewidth=0.3,
        label=f"{val_model} (validation)", zorder=4,
    )

    ax.set_xlabel("Acquisition Cost")
    ax.set_ylabel("RMSE_log Mean (lower is better)")
    ax.set_title(f"Pareto Validation: {primary_model} vs {val_model}")
    ax.legend(loc="best", fontsize=9)

    _save_fig(fig, fig_dir, "pareto_validation")


def fig_feature_frequency_pareto(
    frontiers: Dict[str, pd.DataFrame],
    reporting_metrics: list[str],
    sweep_features: list[str],
    fig_dir: Path,
) -> None:
    all_freq = {f: 0 for f in sweep_features}
    primary_freq = {f: 0 for f in sweep_features}
    focus_metric = (
        "RMSE_log" if "RMSE_log" in reporting_metrics else reporting_metrics[0]
    )

    for metric in reporting_metrics:
        frontier = frontiers.get(metric)
        if frontier is None or frontier.empty:
            continue
        for _, row in frontier.iterrows():
            feats_str = row.get("features", "")
            if not feats_str or feats_str == "(DEPTH-only)":
                continue
            for f in feats_str.split(","):
                f = f.strip()
                if f in all_freq:
                    all_freq[f] += 1
                    if metric == focus_metric:
                        primary_freq[f] += 1

    sorted_feats = sorted(sweep_features, key=lambda f: all_freq[f], reverse=True)

    fig, ax = plt.subplots(figsize=(10, max(4, len(sorted_feats) * 0.4)))

    y = np.arange(len(sorted_feats))
    bar_w = 0.35
    ax.barh(
        y - bar_w / 2,
        [all_freq[f] for f in sorted_feats],
        bar_w,
        color="steelblue",
        edgecolor="black",
        linewidth=0.3,
        label="Scoped metrics",
    )
    ax.barh(
        y + bar_w / 2,
        [primary_freq[f] for f in sorted_feats],
        bar_w,
        color="crimson",
        edgecolor="black",
        linewidth=0.3,
        label=f"{focus_metric} only",
    )

    ax.set_yticks(y)
    ax.set_yticklabels(sorted_feats, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Frequency in Pareto-Optimal Subsets")
    ax.set_title("Feature Frequency in Pareto-Optimal Subsets")
    ax.legend(loc="lower right", fontsize=9)

    _save_fig(fig, fig_dir, "feature_frequency_pareto")


# ── Tables ───────────────────────────────────────────────────────────────────


def save_tables(
    shap_data: dict,
    sweep_data: dict,
    config: dict,
    tables_dir: Path,
    reporting_metrics: Optional[list[str]] = None,
    verbose: bool = True,
) -> Dict[str, Path]:
    tables_dir.mkdir(parents=True, exist_ok=True)
    paths: Dict[str, Path] = {}

    active_metrics = reporting_metrics or RANKING_METRICS
    primary_frontier = sweep_data.get("frontiers", {}).get("RMSE_log")
    if primary_frontier is not None and not primary_frontier.empty:
        pareto = primary_frontier.copy().sort_values("cost").reset_index(drop=True)
        pareto.insert(0, "rank", range(1, len(pareto) + 1))

        keep = ["rank", "subset_id", "features", "n_features", "cost"]
        metrics_for_columns = ["RMSE_log"] + [
            m for m in active_metrics if m != "RMSE_log"
        ]
        for m in metrics_for_columns:
            for s in ("_mean", "_std", "_ci"):
                col = f"{m}{s}"
                if col in pareto.columns:
                    keep.append(col)
        avail = [c for c in keep if c in pareto.columns]
        out = tables_dir / "pareto_optimal_subsets.csv"
        pareto[avail].to_csv(out, index=False)
        paths["T1"] = out
        if verbose:
            print(f"  [T1] {out}")

    global_imp = shap_data.get("global_importance")
    feature_costs = config.get("feature_costs", {})
    if global_imp is not None:
        grouped_global = (
            global_imp.copy()
            .sort_values("mean_abs_shap", ascending=False)
            .reset_index(drop=True)
        )
        grouped_global.insert(0, "rank", range(1, len(grouped_global) + 1))
        out_grouped = tables_dir / "shap_grouped_feature_importance.csv"
        grouped_global.to_csv(out_grouped, index=False)
        paths["T2"] = out_grouped
        if verbose:
            print(f"  [T2] {out_grouped}")

        rows = []
        for _, row in grouped_global.iterrows():
            feat = row["feature"]
            rows.append({
                "feature": feat,
                "mean_abs_shap": row["mean_abs_shap"],
                "acquisition_cost": feature_costs.get(feat),
            })
        out = tables_dir / "shap_vs_cost.csv"
        pd.DataFrame(rows).to_csv(out, index=False)
        paths["T3"] = out
        if verbose:
            print(f"  [T3] {out}")

    return paths


# ── Markdown helpers ─────────────────────────────────────────────────────────


def _md_table(df: pd.DataFrame, columns: list[str], fmt: str = ".4f") -> str:
    available = [c for c in columns if c in df.columns]
    if not available:
        return "(no data)\n"
    header = "| " + " | ".join(available) + " |"
    sep = "| " + " | ".join("---" for _ in available) + " |"
    lines = [header, sep]
    for _, r in df.iterrows():
        cells = []
        for c in available:
            v = r[c]
            if pd.isna(v):
                cells.append("\u2014")
            elif isinstance(v, float):
                cells.append(f"{v:{fmt}}")
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


# ── Report generation ────────────────────────────────────────────────────────


def generate_report(
    shap_data: dict,
    sweep_data: dict,
    retune_data: dict,
    validation_data: dict,
    config: dict,
    table_paths: Dict[str, Path],
    figure_paths: Dict[str, Path],
    reporting_metrics: Optional[list[str]] = None,
    metric_scope: str = DEFAULT_METRIC_SCOPE,
) -> str:
    active_metrics = reporting_metrics or RANKING_METRICS
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    sweep_features = config.get("sweep_features", [])
    n_sweep = len(sweep_features)

    lines = [
        "# Phase 3 Feature Selection \u2014 Analysis Report",
        "",
        f"Generated: {timestamp}",
        "",
        "## Configuration",
        "",
        f"- **Primary model**: {config.get('best_model', 'N/A')}",
        f"- **Primary variant**: `{config.get('best_variant', 'N/A')}`",
        f"- **Metric scope**: `{metric_scope}`",
        f"- **Reporting metrics**: {', '.join(active_metrics)}",
        f"- **Sweep features**: {n_sweep} ({', '.join(sweep_features)})",
        f"- **Total subsets**: 2^{n_sweep} = {1 << n_sweep}",
        f"- **Always included**: DEPTH (continuous), "
        f"Source & Zone (categorical)",
        "",
    ]

    # ── SHAP section ──────────────────────────────────────────────────────
    global_imp = shap_data.get("global_importance")
    if global_imp is not None:
        lines.extend([
            "---",
            "",
            "## SHAP Feature Importance",
            "",
            "SHAP values computed across all LOWO folds using TreeExplainer.",
            "",
            "### Global Importance (Top 15 by mean |SHAP|)",
            "",
            _md_table(global_imp.head(15), ["feature", "mean_abs_shap"]),
            "",
        ])

        per_well = shap_data.get("per_well_importance")
        if per_well is not None:
            wells = sorted(per_well["well"].unique())
            lines.extend([
                f"### Per-Well Breakdown",
                "",
                f"Wells analysed: {', '.join(str(w) for w in wells)}",
                "",
            ])

    # ── Sweep section ─────────────────────────────────────────────────────
    sweep_results = sweep_data.get("sweep_results")
    frontiers = sweep_data.get("frontiers", {})
    primary_frontier = frontiers.get("RMSE_log")

    if sweep_results is not None:
        lines.extend([
            "---",
            "",
            "## Pareto Sweep Results",
            "",
            f"- **Subsets evaluated**: {len(sweep_results)}",
            f"- **Cost range**: "
            f"{sweep_results['cost'].min():.0f} \u2013 "
            f"{sweep_results['cost'].max():.0f}",
        ])
        for metric in active_metrics:
            frontier = frontiers.get(metric)
            if frontier is not None:
                lines.append(
                    f"- **Pareto frontier ({metric})**: "
                    f"{len(frontier)} subsets"
                )
        lines.append("")

    if primary_frontier is not None and not primary_frontier.empty:
        pf_sorted = primary_frontier.sort_values("cost")
        lines.extend([
            "### Primary Pareto Frontier (RMSE_log)",
            "",
            _md_table(
                pf_sorted,
                ["subset_id", "features", "n_features", "cost",
                 "RMSE_log_mean", "RMSE_log_std", "RMSE_log_ci"],
            ),
            "",
            "**Key operating points:**",
            "",
        ])

        cheapest = pf_sorted.iloc[0]
        best_perf = pf_sorted.iloc[-1]
        mean_val = cheapest.get("RMSE_log_mean")
        mean_str = (
            f", RMSE_log_mean={mean_val:.4f}" if pd.notna(mean_val) else ""
        )
        lines.append(
            f"- Cheapest on frontier: cost={cheapest['cost']:.0f}, "
            f"n_features={int(cheapest.get('n_features', 0))}"
            f"{mean_str}"
        )
        mean_val = best_perf.get("RMSE_log_mean")
        mean_str = (
            f", RMSE_log_mean={mean_val:.4f}" if pd.notna(mean_val) else ""
        )
        lines.append(
            f"- Best performance on frontier: cost={best_perf['cost']:.0f}, "
            f"n_features={int(best_perf.get('n_features', 0))}"
            f"{mean_str}"
        )
        lines.append("")

    # ── Retune section ────────────────────────────────────────────────────
    retune_results = retune_data.get("retune_results")
    if retune_results is not None:
        sort_col = (
            "RMSE_log_mean" if "RMSE_log_mean" in retune_results.columns
            else "cost"
        )
        lines.extend([
            "---",
            "",
            "## HP Re-Tuning Results",
            "",
            f"- **Subsets re-tuned**: {len(retune_results)}",
            "",
            _md_table(
                retune_results.sort_values(sort_col),
                ["subset_id", "features", "n_features", "cost",
                 "RMSE_log_mean", "RMSE_log_std", "RMSE_log_ci"],
            ),
            "",
        ])

    # ── Validation section ────────────────────────────────────────────────
    validation_results = validation_data.get("validation_results")
    if validation_results is not None:
        val_model = config.get("validation_model", "N/A")
        val_variant = config.get("validation_variant", "N/A")
        lines.extend([
            "---",
            "",
            f"## Validation ({val_model})",
            "",
            f"Re-evaluated Pareto subsets with {val_model} on "
            f"`{val_variant}`.",
            "",
            _md_table(
                validation_results.sort_values("cost"),
                ["subset_id", "features", "n_features", "cost",
                 "RMSE_log_mean", "RMSE_log_std", "RMSE_log_ci"],
            ),
            "",
        ])

    # ── Output files ──────────────────────────────────────────────────────
    lines.extend([
        "---",
        "",
        "## Output Files",
        "",
        "### Tables",
        "",
    ])
    for tag, p in sorted(table_paths.items()):
        lines.append(f"- **{tag}**: `{p}`")

    lines.extend(["", "### Figures", ""])
    for tag, p in sorted(figure_paths.items()):
        lines.append(f"- **{tag}**: `{p}`")

    lines.append("")
    return "\n".join(lines)


# ── HTML report ───────────────────────────────────────────────────────────────


def _img_base64_tag(fig_dir: Path, stem: str) -> str:
    """Read a saved PNG and return an <img> tag with base64 data URI."""
    import base64

    png_path = fig_dir / "png" / f"{stem}.png"
    if not png_path.exists():
        return '<p class="text-muted fst-italic">Figure not available (run without --no-figures).</p>'
    data = base64.b64encode(png_path.read_bytes()).decode()
    return (
        f'<img src="data:image/png;base64,{data}" '
        f'class="img-fluid rounded shadow-sm" alt="{stem}">'
    )


def _html_table(
    df: pd.DataFrame,
    columns: Optional[list[str]] = None,
    fmt: str = ".4f",
) -> str:
    """Render a DataFrame as a Bootstrap-styled HTML table."""
    if df.empty:
        return '<p class="text-muted">No data available.</p>'
    cols = [c for c in (columns or df.columns) if c in df.columns]
    if not cols:
        return '<p class="text-muted">No columns available.</p>'

    lines = [
        '<div class="table-responsive">',
        '<table class="table table-striped table-hover table-sm align-middle">',
        '<thead class="table-dark"><tr>',
    ]
    for c in cols:
        lines.append(f"<th>{c}</th>")
    lines.append("</tr></thead><tbody>")

    for _, r in df.iterrows():
        lines.append("<tr>")
        for c in cols:
            v = r[c]
            if pd.isna(v):
                lines.append('<td class="text-muted">&mdash;</td>')
            elif isinstance(v, float):
                lines.append(f"<td>{v:{fmt}}</td>")
            else:
                lines.append(f"<td>{v}</td>")
        lines.append("</tr>")

    lines.append("</tbody></table></div>")
    return "\n".join(lines)


def generate_html_report(
    shap_data: dict,
    sweep_data: dict,
    retune_data: dict,
    validation_data: dict,
    config: dict,
    table_paths: Dict[str, Path],
    figure_paths: Dict[str, Path],
    output_dir: Path,
    reporting_metrics: Optional[list[str]] = None,
    metric_scope: str = DEFAULT_METRIC_SCOPE,
) -> Path:
    """Generate a self-contained HTML report with embedded figures and tables."""
    fig_dir = output_dir / "figures"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    global_imp = shap_data.get("global_importance", pd.DataFrame())
    per_well = shap_data.get("per_well_importance", pd.DataFrame())
    sweep_results = sweep_data.get("sweep_results", pd.DataFrame())
    frontiers = sweep_data.get("frontiers", {})
    primary_frontier = frontiers.get("RMSE_log", pd.DataFrame())
    retune_results = retune_data.get("retune_results", pd.DataFrame())
    validation_results = validation_data.get("validation_results", pd.DataFrame())

    cfg_table = pd.DataFrame(
        [
            {"field": "Primary model", "value": config.get("best_model", "N/A")},
            {"field": "Primary variant", "value": config.get("best_variant", "N/A")},
            {"field": "Metric scope", "value": metric_scope},
            {
                "field": "Reporting metrics",
                "value": ", ".join(reporting_metrics or RANKING_METRICS),
            },
            {"field": "Sweep features", "value": len(config.get("sweep_features", []))},
            {"field": "Cost mode", "value": config.get("cost_mode", "N/A")},
            {"field": "Validation model", "value": config.get("validation_model", "N/A")},
            {"field": "Validation variant", "value": config.get("validation_variant", "N/A")},
        ]
    )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Phase 3 - Feature Selection Report</title>
<link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }}
  .fig {{ text-align: center; margin: 1.25rem 0; }}
  .fig img {{ max-width: 100%; height: auto; }}
  .sec {{ border-bottom: 2px solid #dee2e6; padding-bottom: .5rem; margin-bottom: 1rem; }}
  .table {{ font-size: .9rem; }}
</style>
</head>
<body>
<nav class="navbar navbar-dark bg-dark mb-4">
  <div class="container-fluid">
    <span class="navbar-brand mb-0 h1">Phase 3 - Feature Selection Report</span>
    <span class="navbar-text text-light">{timestamp}</span>
  </div>
</nav>
<div class="container-fluid px-4">

<h4 class="sec">Configuration</h4>
{_html_table(cfg_table)}

<h4 class="sec mt-4">SHAP Global Importance</h4>
{_html_table(global_imp.head(20), ["feature", "mean_abs_shap"])}
<div class="fig">{_img_base64_tag(fig_dir, "shap_global_bar")}</div>
<div class="fig">{_img_base64_tag(fig_dir, "shap_beeswarm")}</div>
<div class="fig">{_img_base64_tag(fig_dir, "shap_per_well_heatmap")}</div>

<h4 class="sec mt-4">Pareto Sweep</h4>
{_html_table(primary_frontier, ["subset_id", "features", "n_features", "cost", "RMSE_log_mean", "RMSE_log_std", "RMSE_log_ci"])}
<div class="fig">{_img_base64_tag(fig_dir, "pareto_frontier_RMSE_log")}</div>
<div class="fig">{_img_base64_tag(fig_dir, "feature_frequency_pareto")}</div>

<h4 class="sec mt-4">Re-tune and Validation</h4>
<h6>Retune</h6>
{_html_table(retune_results, ["subset_id", "features", "n_features", "cost", "RMSE_log_mean", "RMSE_log_std", "RMSE_log_ci"])}
<div class="fig">{_img_base64_tag(fig_dir, "pareto_frontier_retune")}</div>
<h6 class="mt-4">Validation</h6>
{_html_table(validation_results, ["subset_id", "features", "n_features", "cost", "RMSE_log_mean", "RMSE_log_std", "RMSE_log_ci"])}
<div class="fig">{_img_base64_tag(fig_dir, "pareto_validation")}</div>

<h4 class="sec mt-4">Output Files</h4>
<p><strong>Tables:</strong></p>
<ul>
{''.join(f'<li>{k}: <code>{p}</code></li>' for k, p in sorted(table_paths.items()))}
</ul>
<p><strong>Figures:</strong></p>
<ul>
{''.join(f'<li>{k}: <code>{p}</code></li>' for k, p in sorted(figure_paths.items()))}
</ul>

</div>
<footer class="text-center text-muted py-3 mt-4 border-top">
  <small>Generated by analyze_phase3.py - {timestamp}</small>
</footer>
</body>
</html>"""

    html_path = output_dir / "phase3_report.html"
    html_path.write_text(html, encoding="utf-8")
    return html_path


# ── Main pipeline ────────────────────────────────────────────────────────────


def run_analysis(
    results_dir: str | Path | None = None,
    config_path: str | None = None,
    generate_figs: bool = True,
    generate_html: bool = False,
    metric_scope: str = DEFAULT_METRIC_SCOPE,
    verbose: bool = True,
) -> dict:
    """Run the full Phase 3 analysis pipeline."""
    project_root = Path(__file__).parent.parent

    if results_dir is None:
        results_dir = project_root / "results" / "phase3_feature_selection"
    else:
        results_dir = Path(results_dir)

    analysis_dir_name = (
        "analysis"
        if metric_scope == DEFAULT_METRIC_SCOPE
        else f"analysis_{metric_scope}"
    )
    output_dir = results_dir / analysis_dir_name
    output_dir.mkdir(parents=True, exist_ok=True)

    config = load_phase3_config(config_path)
    reporting_metrics = resolve_metric_scope(metric_scope)

    if verbose:
        print("Phase 3 Feature Selection \u2014 Analysis")
        print(f"  Results dir: {results_dir}")
        print(f"  Output dir:  {output_dir}")
        print(f"  Metric scope: {metric_scope}")
        print(f"  Reporting metrics: {', '.join(reporting_metrics)}")

    # ── Load all prior outputs ────────────────────────────────────────────
    if verbose:
        print("\nLoading data...")

    shap_data = load_shap_outputs(results_dir)
    sweep_data = load_sweep_outputs(results_dir)
    retune_data = load_retune_outputs(results_dir)
    validation_data = load_validation_outputs(results_dir)

    has_shap = "global_importance" in shap_data
    has_sweep = "sweep_results" in sweep_data
    has_retune = "retune_results" in retune_data
    has_validation = "validation_results" in validation_data

    if verbose:
        print(f"  SHAP data:       {'yes' if has_shap else 'no'}")
        print(f"  Sweep data:      {'yes' if has_sweep else 'no'}")
        print(f"  Retune data:     {'yes' if has_retune else 'no'}")
        print(f"  Validation data: {'yes' if has_validation else 'no'}")

    if not has_shap and not has_sweep:
        print(
            "No Phase 3 results found.  Run run_phase3_shap.py and/or "
            "run_phase3_pareto.py --sweep first."
        )
        return {}

    # ── Tables ────────────────────────────────────────────────────────────
    if verbose:
        print("\nSaving tables...")
    tables_dir = output_dir / "tables"
    table_paths = save_tables(
        shap_data,
        sweep_data,
        config,
        tables_dir,
        reporting_metrics=reporting_metrics,
        verbose=verbose,
    )

    # ── Figures ───────────────────────────────────────────────────────────
    figure_paths: Dict[str, Path] = {}

    if generate_figs:
        if verbose:
            print("\nGenerating figures...")
        _set_style()
        fig_dir = output_dir / "figures"
        fig_dir.mkdir(parents=True, exist_ok=True)

        # SHAP figures
        if has_shap:
            try:
                if verbose:
                    print("  shap_global_bar")
                fig_shap_global_bar(shap_data["global_importance"], fig_dir)
                figure_paths["F1"] = fig_dir / "pdf" / "shap_global_bar.pdf"
            except Exception as e:
                print(f"  Warning [shap_global_bar]: {e}")

            if "shap_values" in shap_data:
                try:
                    if verbose:
                        print("  shap_beeswarm")
                    fig_shap_beeswarm(
                        shap_data["shap_values"],
                        shap_data["feature_names"],
                        fig_dir,
                    )
                    figure_paths["F2"] = (
                        fig_dir / "pdf" / "shap_beeswarm.pdf"
                    )
                except Exception as e:
                    print(f"  Warning [shap_beeswarm]: {e}")

            if "per_well_importance" in shap_data:
                try:
                    if verbose:
                        print("  shap_per_well_heatmap")
                    fig_shap_per_well_heatmap(
                        shap_data["per_well_importance"], fig_dir,
                    )
                    figure_paths["F3"] = (
                        fig_dir / "pdf" / "shap_per_well_heatmap.pdf"
                    )
                except Exception as e:
                    print(f"  Warning [shap_per_well_heatmap]: {e}")

        # Pareto figures
        if has_sweep:
            sweep_results = sweep_data["sweep_results"]
            frontiers = sweep_data.get("frontiers", {})

            for metric in reporting_metrics:
                frontier = frontiers.get(metric)
                if frontier is None or frontier.empty:
                    continue
                try:
                    if verbose:
                        print(f"  pareto_frontier_{metric}")
                    fig_pareto_frontier(
                        sweep_results, frontier, metric, fig_dir,
                    )
                    tag = f"F4_{metric}"
                    figure_paths[tag] = (
                        fig_dir / "pdf" / f"pareto_frontier_{metric}.pdf"
                    )
                except Exception as e:
                    print(f"  Warning [pareto_frontier_{metric}]: {e}")

            if "RMSE_log" not in reporting_metrics:
                primary_frontier = frontiers.get("RMSE_log")
                if primary_frontier is not None and not primary_frontier.empty:
                    try:
                        if verbose:
                            print("  pareto_frontier_RMSE_log (lineage)")
                        fig_pareto_frontier(
                            sweep_results, primary_frontier, "RMSE_log", fig_dir,
                        )
                        figure_paths["F4_primary_RMSE_log"] = (
                            fig_dir / "pdf" / "pareto_frontier_RMSE_log.pdf"
                        )
                    except Exception as e:
                        print(f"  Warning [pareto_frontier_RMSE_log]: {e}")

            if has_retune and "frontier_retune" in retune_data:
                sweep_frontier = frontiers.get("RMSE_log", pd.DataFrame())
                if not sweep_frontier.empty:
                    try:
                        if verbose:
                            print("  pareto_frontier_retune")
                        fig_pareto_frontier_retune(
                            sweep_results,
                            sweep_frontier,
                            retune_data["frontier_retune"],
                            fig_dir,
                        )
                        figure_paths["F5"] = (
                            fig_dir / "pdf" / "pareto_frontier_retune.pdf"
                        )
                    except Exception as e:
                        print(f"  Warning [pareto_frontier_retune]: {e}")

            if has_validation:
                try:
                    if verbose:
                        print("  pareto_validation")
                    best_frontier = (
                        retune_data["frontier_retune"]
                        if has_retune and "frontier_retune" in retune_data
                        else frontiers.get("RMSE_log", pd.DataFrame())
                    )
                    if not best_frontier.empty:
                        fig_pareto_validation(
                            sweep_results,
                            best_frontier,
                            validation_data["validation_results"],
                            config,
                            fig_dir,
                        )
                        figure_paths["F6"] = (
                            fig_dir / "pdf" / "pareto_validation.pdf"
                        )
                except Exception as e:
                    print(f"  Warning [pareto_validation]: {e}")

            if frontiers:
                try:
                    if verbose:
                        print("  feature_frequency_pareto")
                    fig_feature_frequency_pareto(
                        frontiers,
                        reporting_metrics,
                        config["sweep_features"],
                        fig_dir,
                    )
                    figure_paths["F7"] = (
                        fig_dir / "pdf" / "feature_frequency_pareto.pdf"
                    )
                except Exception as e:
                    print(f"  Warning [feature_frequency_pareto]: {e}")

    # ── Report ────────────────────────────────────────────────────────────
    if verbose:
        print("\nGenerating report...")

    report_text = generate_report(
        shap_data, sweep_data, retune_data, validation_data,
        config, table_paths, figure_paths,
        reporting_metrics=reporting_metrics,
        metric_scope=metric_scope,
    )
    report_path = output_dir / "phase3_report.md"
    report_path.write_text(report_text, encoding="utf-8")
    if verbose:
        print(f"  {report_path}")

    html_path = None
    if generate_html:
        if verbose:
            print("Generating HTML report...")
        html_path = generate_html_report(
            shap_data=shap_data,
            sweep_data=sweep_data,
            retune_data=retune_data,
            validation_data=validation_data,
            config=config,
            table_paths=table_paths,
            figure_paths=figure_paths,
            output_dir=output_dir,
            reporting_metrics=reporting_metrics,
            metric_scope=metric_scope,
        )
        if verbose:
            print(f"  {html_path}")

    if verbose:
        print(f"\n{'=' * 60}")
        print("Phase 3 analysis complete.")
        print(f"  Tables:  {len(table_paths)}")
        print(f"  Figures: {len(figure_paths)}")
        print(f"  Report:  {report_path}")
        if html_path is not None:
            print(f"  HTML:    {html_path}")
        print(f"{'=' * 60}")

    return {
        "table_paths": table_paths,
        "figure_paths": figure_paths,
        "report_path": report_path,
        "html_path": html_path,
    }


# ── CLI ───────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Phase 3 Feature Selection \u2014 analysis, visualization, "
                    "and reporting.",
    )
    parser.add_argument(
        "--results_dir", "-r",
        type=str, default=None,
        help="Phase 3 results directory "
             "(default: results/phase3_feature_selection).",
    )
    parser.add_argument(
        "--config",
        type=str, default=None,
        help="Path to phase3 config file (standalone or bundled).",
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
        "--metric-scope",
        choices=METRIC_SCOPE_CHOICES,
        default=DEFAULT_METRIC_SCOPE,
        help="Metric family for sweep-frontier reporting: original, log, or both (default: both).",
    )
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Suppress verbose output.",
    )

    args = parser.parse_args()

    run_analysis(
        results_dir=args.results_dir,
        config_path=args.config,
        generate_figs=not args.no_figures,
        generate_html=args.html,
        metric_scope=args.metric_scope,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
