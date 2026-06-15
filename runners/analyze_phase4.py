#!/usr/bin/env python
"""
Phase 4 Nested LOWO - Analysis and Reporting.

Consumes run artifacts from results/phase4_generalization/run and produces:
- nested_summary.csv / nested_summary.json
- bias_comparison.csv
- baseline_comparison.csv (nested vs petrophysical/regression baselines, when
  baseline outputs from run_phase4_baselines.py are available)
- analysis/tables/per_zone_metrics_*.csv (per-zone error breakdown from nested
  predictions joined to Zone via row_index)
- ablation_comparison.csv (delta-vs-full per subset for every metric, when
  structural-covariate ablation outputs from run_phase4_ablation.py exist)
- analysis tables/figures/report
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.nested_cv import METRIC_NAMES, load_phase4_config
from src.cv_utils import compute_metrics

MINIMIZE_METRICS = {"RMSE", "MAE", "MedAE", "RMSLE", "RMSE_log", "MAE_log"}
MAXIMIZE_METRICS = {"R2", "R2_log"}
METRIC_SCOPE_CHOICES = ("original", "log", "both")
DEFAULT_METRIC_SCOPE = "log"

# Mean-line styling for per-well boxplot, mirrors analyze_phase2.BOXPLOT_MEAN_PROPS.
BOXPLOT_MEAN_PROPS = {
    "color": "red",
    "alpha": 0.65,
    "linewidth": 1.6,
    "linestyle": "-",
}

# Held-out wells in this study; explicit so the same color is used across
# Option-A and Option-B prediction figures.
WELL_NAMES_DEFAULT = ["A", "B", "C", "D", "E", "F", "G"]
PANEL_LETTERS = "abcdefghijklmnop"

# Compact per-subset display labels for box/heatmap ticks. Falls back to the raw
# subset_label when an entry is missing.
SUBSET_DISPLAY_LABELS = {
    "best_performer": "Best-performer",
    "cpor_sm_only_baseline": "CPOR_SM-only",
    "budget_wireline_only": "Budget wireline",
    "full_feature_baseline": "Full-feature",
}


def _format_subset_label(subset_id: int, subset_label: str) -> str:
    """Render `<subset_id> (<short label>)` for axis ticks/heatmap rows."""
    short = SUBSET_DISPLAY_LABELS.get(subset_label, subset_label)
    return f"{subset_id} ({short})"


# Compact petrophysical-baseline tick labels for the combined boxplot. Falls
# back to a trimmed version of baseline_label when a slug is missing.
BASELINE_DISPLAY_LABELS = {
    "loglinear_cpor_sm": "Log-linear (CPOR_SM)",
    "loglinear_phit": "Log-linear (PHIT)",
    "timur_cpor_sm_swt": "Timur (CPOR_SM)",
    "timur_phit_swt": "Timur (PHIT)",
}


def _short_baseline_label(slug, fallback_label) -> str:
    """Short, baseline-model tick label for axis display."""
    if slug in BASELINE_DISPLAY_LABELS:
        return BASELINE_DISPLAY_LABELS[slug]
    text = str(fallback_label or slug or "baseline")
    return text.replace("porosity-permeability ", "").replace(" baseline", "")


def resolve_metric_scope(metric_scope: str) -> list[str]:
    scope = metric_scope.lower()
    if scope == "original":
        return ["original"]
    if scope == "both":
        return ["log", "original"]
    return ["log"]


def primary_metric_for_family(family: str) -> str:
    return "RMSE_log" if family == "log" else "RMSE"


def secondary_metric_for_family(family: str) -> str:
    return "R2_log" if family == "log" else "R2"


def mae_metric_for_family(family: str) -> str:
    return "MAE_log" if family == "log" else "MAE"


def scoped_stem(stem: str, metric_scope: str) -> str:
    return stem if metric_scope == DEFAULT_METRIC_SCOPE else f"{stem}_{metric_scope}"


def _set_style():
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
    for sub in ("pdf", "svg", "png"):
        (fig_dir / sub).mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_dir / "pdf" / f"{stem}.pdf")
    fig.savefig(fig_dir / "svg" / f"{stem}.svg")
    fig.savefig(fig_dir / "png" / f"{stem}.png")
    plt.close(fig)


def load_run_results(run_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    nested_path = run_dir / "nested_outer_results.csv"
    trace_path = run_dir / "selection_trace.csv"
    if not nested_path.exists():
        raise FileNotFoundError(f"Missing nested results: {nested_path}")
    nested_df = pd.read_csv(nested_path)
    trace_df = pd.read_csv(trace_path) if trace_path.exists() else pd.DataFrame()
    return nested_df, trace_df


def aggregate_nested_summary(
    nested_df: pd.DataFrame, sort_metric: str = "RMSE_log"
) -> pd.DataFrame:
    group_cols = ["subset_id", "subset_label", "features", "n_features", "cost", "model", "variant"]
    summary = (
        nested_df.groupby(group_cols, dropna=False)
        .size()
        .reset_index(name="n_outer_folds")
    )

    for metric in METRIC_NAMES:
        col = f"outer_{metric}"
        agg = nested_df.groupby(group_cols, dropna=False)[col].agg(["mean", "std"]).reset_index()
        agg = agg.rename(columns={"mean": f"{metric}_mean", "std": f"{metric}_std"})
        summary = summary.merge(agg, on=group_cols, how="left")

        n = summary["n_outer_folds"]
        t_crit = stats.t.ppf(1 - 0.05 / 2, df=n - 1)
        sem = summary[f"{metric}_std"].fillna(0.0) / np.sqrt(n)
        if metric in MINIMIZE_METRICS:
            summary[f"{metric}_ci"] = summary[f"{metric}_mean"] + t_crit * sem
        else:
            summary[f"{metric}_ci"] = summary[f"{metric}_mean"] - t_crit * sem

    sort_col = f"{sort_metric}_ci"
    if sort_col not in summary.columns:
        sort_col = "RMSE_log_ci" if "RMSE_log_ci" in summary.columns else "cost"
    return summary.sort_values([sort_col, "cost"], ascending=[True, True]).reset_index(
        drop=True
    )


def build_selection_summary(trace_df: pd.DataFrame) -> pd.DataFrame:
    if trace_df.empty:
        return pd.DataFrame()
    counts = (
        trace_df.groupby(["selected_subset_id", "selected_subset_label"], dropna=False)
        .size()
        .reset_index(name="selected_outer_folds")
        .sort_values("selected_outer_folds", ascending=False)
        .reset_index(drop=True)
    )
    return counts


def load_phase3_reference(config: dict, project_root: Path) -> pd.DataFrame:
    path = config.get("phase3_retune_results_path")
    if not path:
        return pd.DataFrame()
    full = project_root / path
    if not full.exists():
        return pd.DataFrame()
    return pd.read_csv(full)


def load_phase2_reference(config: dict, project_root: Path) -> dict:
    path = config.get("phase2_best_params_path")
    if not path:
        return {}
    full = project_root / path
    if not full.exists():
        return {}
    with open(full, "r") as f:
        payload = json.load(f)
    return payload.get("best_metrics", {})


def load_baseline_outputs(baseline_dir: Path) -> dict:
    """Load Phase 4 petrophysical/regression baseline outputs.

    Returns a dict with optional keys ``baseline_summary`` (per-baseline
    aggregated metrics) and ``baseline_outer`` (per-fold metrics), each a
    DataFrame when the corresponding CSV exists.
    """
    outputs: dict = {}

    summary_path = baseline_dir / "baseline_summary.csv"
    if summary_path.exists():
        outputs["baseline_summary"] = pd.read_csv(summary_path)

    outer_path = baseline_dir / "baseline_outer_results.csv"
    if outer_path.exists():
        outputs["baseline_outer"] = pd.read_csv(outer_path)

    meta_path = baseline_dir / "baseline_meta.json"
    if meta_path.exists():
        with open(meta_path, "r") as f:
            outputs["meta"] = json.load(f)

    return outputs


def load_ablation_outputs(ablation_dir: Path) -> dict:
    """Load Phase 4 structural-covariate ablation outputs (nested layout).

    Each ablated config is run under the full nested LOWO protocol; the
    full-covariate reference is reused from the canonical nested run rather than
    recomputed here. ``reduce_ablation`` writes ``nested_summary.csv``
    (per subset x ablated config) and ``nested_results_all.csv`` (every outer
    fold row). Returns a dict with optional keys ``ablation_summary``,
    ``ablation_results`` (all fold rows), and ``meta``; each present only when
    the corresponding file exists.
    """
    outputs: dict = {}

    summary_path = ablation_dir / "nested_summary.csv"
    if summary_path.exists():
        outputs["ablation_summary"] = pd.read_csv(summary_path)

    results_path = ablation_dir / "nested_results_all.csv"
    if results_path.exists():
        outputs["ablation_results"] = pd.read_csv(results_path)

    meta_path = ablation_dir / "ablation_meta.json"
    if meta_path.exists():
        with open(meta_path, "r") as f:
            outputs["meta"] = json.load(f)

    return outputs


def build_ablation_comparison(
    ablation_summary_df: pd.DataFrame,
    full_summary_df: pd.DataFrame,
) -> pd.DataFrame:
    """Per-subset deltas of each ablated config vs the canonical full nested run.

    Each ablated config (``-Source``, ``-Zone``, ``-DEPTH``, ``-all-three``) is
    run under the full nested LOWO protocol. The full-covariate reference is the
    canonical Phase 4 nested summary (``full_summary_df``, the same per-subset
    means reported in the headline generalization table), looked up by
    ``subset_id``. For each config and every metric we emit the config mean, the
    full mean, and ``delta_vs_full = config_mean - full_mean``.

    Sign convention (same as the other comparison tables): for minimize metrics
    (e.g. RMSE_log) a positive delta means the ablated config is worse than full
    (dropping the covariate hurt); for maximize metrics (R2_log) a negative
    delta means worse.
    """
    if (
        ablation_summary_df is None
        or ablation_summary_df.empty
        or full_summary_df is None
        or full_summary_df.empty
    ):
        return pd.DataFrame()

    full_lookup = {
        int(r["subset_id"]): r.to_dict() for _, r in full_summary_df.iterrows()
    }
    # Canonical config ordering for deterministic, readable tables.
    config_order = {
        "no_source": 1,
        "no_zone": 2,
        "no_depth": 3,
        "no_all_three": 4,
    }

    rows = []
    for _, crow in ablation_summary_df.iterrows():
        subset_id = int(crow["subset_id"])
        if subset_id not in full_lookup:
            continue
        full = full_lookup[subset_id]
        for metric in METRIC_NAMES:
            config_mean = crow.get(f"{metric}_mean")
            full_mean = full.get(f"{metric}_mean")
            if pd.isna(config_mean) or pd.isna(full_mean):
                continue
            direction = "minimize" if metric in MINIMIZE_METRICS else "maximize"
            rows.append(
                {
                    "subset_id": subset_id,
                    "subset_label": crow.get("subset_label"),
                    "config_slug": crow.get("config_slug"),
                    "config_label": crow.get("config_label"),
                    "dropped": crow.get("dropped"),
                    "metric": metric,
                    "direction": direction,
                    "reference_source": "phase4_nested_full",
                    "config_mean": config_mean,
                    "full_mean": full_mean,
                    "delta_vs_full": config_mean - full_mean,
                }
            )

    if not rows:
        return pd.DataFrame()

    out = pd.DataFrame(rows)
    out["_config_order"] = out["config_slug"].map(config_order).fillna(99)
    out["_metric_order"] = out["metric"].map(
        {m: i for i, m in enumerate(METRIC_NAMES)}
    )
    out = out.sort_values(
        ["subset_id", "_config_order", "_metric_order"]
    ).drop(columns=["_config_order", "_metric_order"]).reset_index(drop=True)
    return out


def _is_all_targets(raw) -> bool:
    """Return True when ``comparison_targets`` requests every nested subset."""
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return False
    return str(raw).strip().lower() in {"all", "*"}


def _parse_comparison_targets(raw) -> list[int]:
    """Parse a ``comparison_targets`` cell into a list of subset ids."""
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return []
    text = str(raw).strip()
    if not text:
        return []
    ids = []
    for token in text.replace(";", ",").split(","):
        token = token.strip()
        if token.lstrip("-").isdigit():
            ids.append(int(token))
    return ids


def build_baseline_comparison(
    summary_df: pd.DataFrame,
    baseline_summary_df: pd.DataFrame,
) -> pd.DataFrame:
    """Compare each petrophysical baseline against its target nested subset(s).

    For every baseline row, the ``comparison_targets`` column lists the nested
    operating-point subset ids it should be measured against. A value of
    ``"all"`` (or ``"*"``) compares the baseline against every nested subset.
    For each matched subset and metric, we emit nested vs baseline means plus
    the delta that quantifies how much the ML pipeline improves on the
    classical baseline.

    ``delta_nested_minus_baseline`` follows the same sign convention as the
    bias comparison: for minimize metrics a negative delta means the nested
    model beats the baseline; for maximize metrics a positive delta does.
    """
    if summary_df.empty or baseline_summary_df.empty:
        return pd.DataFrame()

    nested_lookup = {int(r["subset_id"]): r.to_dict() for _, r in summary_df.iterrows()}
    all_subset_ids = sorted(nested_lookup.keys())

    rows = []
    for _, brow in baseline_summary_df.iterrows():
        raw_targets = brow.get("comparison_targets")
        if _is_all_targets(raw_targets):
            target_ids = all_subset_ids
        else:
            target_ids = _parse_comparison_targets(raw_targets)
        for sid in target_ids:
            if sid not in nested_lookup:
                continue
            nested = nested_lookup[sid]
            for metric in METRIC_NAMES:
                nested_mean = nested.get(f"{metric}_mean")
                baseline_mean = brow.get(f"{metric}_mean")
                if pd.isna(nested_mean) or pd.isna(baseline_mean):
                    continue
                direction = "minimize" if metric in MINIMIZE_METRICS else "maximize"
                rows.append(
                    {
                        "baseline_slug": brow.get("baseline_slug"),
                        "baseline_label": brow.get("baseline_label"),
                        "equation": brow.get("equation"),
                        "swirr_proxy_note": brow.get("swirr_proxy_note", ""),
                        "subset_id": sid,
                        "subset_label": nested.get("subset_label"),
                        "metric": metric,
                        "direction": direction,
                        "nested_mean": nested_mean,
                        "baseline_mean": baseline_mean,
                        "delta_nested_minus_baseline": nested_mean - baseline_mean,
                    }
                )

    return pd.DataFrame(rows)


def build_bias_comparison(
    summary_df: pd.DataFrame,
    phase3_ref_df: pd.DataFrame,
    phase2_ref_metrics: dict,
) -> pd.DataFrame:
    rows = []
    phase3_lookup = {}
    if not phase3_ref_df.empty and "subset_id" in phase3_ref_df.columns:
        for _, row in phase3_ref_df.iterrows():
            phase3_lookup[int(row["subset_id"])] = row.to_dict()

    for _, srow in summary_df.iterrows():
        sid = int(srow["subset_id"])
        for metric in METRIC_NAMES:
            nested_mean = srow.get(f"{metric}_mean")
            direction = "minimize" if metric in MINIMIZE_METRICS else "maximize"

            if sid in phase3_lookup:
                ref_val = phase3_lookup[sid].get(f"{metric}_mean")
                if pd.notna(ref_val):
                    rows.append(
                        {
                            "subset_id": sid,
                            "subset_label": srow["subset_label"],
                            "metric": metric,
                            "direction": direction,
                            "reference_source": "phase3_single_lowo_retune",
                            "nested_mean": nested_mean,
                            "reference_mean": ref_val,
                            "delta_nested_minus_reference": nested_mean - ref_val,
                        }
                    )

            if phase2_ref_metrics and metric in phase2_ref_metrics:
                ref_val = phase2_ref_metrics[metric]
                rows.append(
                    {
                        "subset_id": sid,
                        "subset_label": srow["subset_label"],
                        "metric": metric,
                        "direction": direction,
                        "reference_source": "phase2_best_single_lowo",
                        "nested_mean": nested_mean,
                        "reference_mean": ref_val,
                        "delta_nested_minus_reference": nested_mean - ref_val,
                    }
                )

    return pd.DataFrame(rows)


def fig_nested_frontier(
    summary_df: pd.DataFrame,
    fig_dir: Path,
    metric: str = "RMSE_log",
    stem: Optional[str] = None,
) -> None:
    ci_col = f"{metric}_ci"
    if ci_col not in summary_df.columns:
        return
    fig_stem = stem or f"phase4_nested_frontier_{metric.lower()}"
    fig, ax = plt.subplots(figsize=(9, 6))
    view = summary_df.sort_values("cost")
    ax.plot(
        view["cost"],
        view[ci_col],
        "o-",
        color="crimson",
        linewidth=2,
        markersize=7,
        markeredgecolor="black",
        markeredgewidth=0.4,
    )
    for _, row in view.iterrows():
        ax.annotate(
            str(int(row["subset_id"])),
            (row["cost"], row[ci_col]),
            textcoords="offset points",
            xytext=(5, 5),
            fontsize=8,
        )
    ax.set_xlabel("Acquisition cost")
    ax.set_ylabel(f"{metric} CI (95% bound)")
    ax.set_title(f"Nested LOWO: Cost vs {metric}")
    _save_fig(fig, fig_dir, fig_stem)


def fig_per_well_boxplot(
    nested_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    fig_dir: Path,
    metric: str = "RMSE_log",
    stem: Optional[str] = None,
    baseline_outer_df: Optional[pd.DataFrame] = None,
) -> None:
    """Boxplot + strip plot of outer-fold metric by subset, sorted by mean (best at left).

    Styling mirrors ``analyze_phase2.save_figures`` for visual consistency across phases:
    Set3 palette, mean line overlay (red), strip points, and 45-degree rotated labels.

    When ``baseline_outer_df`` is supplied (per-fold petrophysical baseline
    metrics), the baselines are overlaid as additional boxes on the same axis so
    the ML subsets and the classical correlations are directly comparable. The
    baseline boxes use a distinct color and a hatch so they stand apart from the
    subset boxes.
    """
    mean_col = f"{metric}_mean"
    outer_col = f"outer_{metric}"
    if mean_col not in summary_df.columns or outer_col not in nested_df.columns:
        return
    fig_stem = stem or f"phase4_outer_{metric.lower()}_boxplot"

    ascending = metric in MINIMIZE_METRICS
    label_lookup = {
        int(r["subset_id"]): _format_subset_label(int(r["subset_id"]), str(r["subset_label"]))
        for _, r in summary_df.iterrows()
    }

    plot_df = nested_df.copy()
    plot_df["display_label"] = plot_df["subset_id"].astype(int).map(label_lookup)
    plot_df = plot_df.dropna(subset=["display_label"])
    if plot_df.empty:
        return

    # Unify subsets and (optional) baselines into one long frame with a common
    # value column so they can share a single mean-sorted axis.
    combined = plot_df[["display_label", outer_col]].rename(columns={outer_col: "value"})
    combined["is_baseline"] = False

    has_baseline = (
        baseline_outer_df is not None
        and not baseline_outer_df.empty
        and metric in baseline_outer_df.columns
    )
    if has_baseline:
        bdf = baseline_outer_df.copy()
        label_col = "baseline_label" if "baseline_label" in bdf.columns else "baseline_slug"
        bdf["display_label"] = bdf.apply(
            lambda r: _short_baseline_label(r.get("baseline_slug"), r.get(label_col)),
            axis=1,
        )
        b_combined = bdf[["display_label", metric]].rename(columns={metric: "value"})
        b_combined["is_baseline"] = True
        combined = pd.concat([combined, b_combined], ignore_index=True)

    # Order every box by its mean (best at left) on the same metric.
    order_means = (
        combined.groupby("display_label")["value"].mean().sort_values(ascending=ascending)
    )
    label_order = order_means.index.tolist()
    baseline_labels = set(combined.loc[combined["is_baseline"], "display_label"].unique())

    n_boxes = len(label_order)
    # Neutral gray for baselines so they never collide with the Set3 subset
    # colors (notably the salmon assigned to 640); the hatch carries the cue.
    baseline_color = "#9e9e9e"
    set3 = sns.color_palette("Set3", max(1, n_boxes - len(baseline_labels)))
    palette = []
    si = 0
    for lab in label_order:
        if lab in baseline_labels:
            palette.append(baseline_color)
        else:
            palette.append(set3[si % len(set3)])
            si += 1

    fig, ax = plt.subplots(figsize=(max(9, n_boxes * 0.8), 6))
    sns.boxplot(
        data=combined,
        x="display_label",
        y="value",
        order=label_order,
        palette=palette,
        width=0.55,
        linewidth=0.8,
        fliersize=0,
        showmeans=True,
        meanline=True,
        meanprops=BOXPLOT_MEAN_PROPS,
        ax=ax,
    )
    sns.stripplot(
        data=combined,
        x="display_label",
        y="value",
        order=label_order,
        color="black",
        dodge=False,
        size=5,
        alpha=0.75,
        ax=ax,
    )

    if baseline_labels:
        for patch, lab in zip(ax.patches, label_order):
            if lab in baseline_labels:
                patch.set_hatch("//")

    ax.set_xlabel("")
    ax.set_ylabel(metric)
    if baseline_labels:
        title_main = f"Nested LOWO per-well {metric}:\nsubsets vs petrophysical baselines"
    else:
        title_main = f"Nested LOWO per-well {metric} by subset"
    ax.set_title(
        f"{title_main}\nSorted by mean, best at left; red line = mean",
        fontsize=11,
    )
    ax.set_xticks(range(n_boxes))
    ax.set_xticklabels(label_order, rotation=45, ha="right")

    if baseline_labels:
        from matplotlib.patches import Patch

        handles = [
            Patch(facecolor="white", edgecolor="black", label="ML subset"),
            Patch(
                facecolor=baseline_color,
                edgecolor="black",
                hatch="//",
                label="Petrophysical baseline",
            ),
        ]
        #ax.legend(handles=handles, fontsize=8, loc="best", framealpha=0.9)

    _save_fig(fig, fig_dir, fig_stem)


def fig_baseline_boxplot(
    baseline_outer_df: pd.DataFrame,
    fig_dir: Path,
    metric: str = "RMSE_log",
    stem: str = "phase4_baseline_rmse_log_boxplot",
) -> Optional[Path]:
    """Boxplot + strip plot of per-fold baseline metric by baseline model.

    Uses a distinct stem from the nested per-well boxplot so the two figures
    never clobber each other. Returns the SVG path, or ``None`` when the
    requested metric is unavailable.
    """
    if baseline_outer_df.empty or metric not in baseline_outer_df.columns:
        return None

    label_col = (
        "baseline_label"
        if "baseline_label" in baseline_outer_df.columns
        else "baseline_slug"
    )
    if label_col not in baseline_outer_df.columns:
        return None

    ascending = metric in MINIMIZE_METRICS
    order = (
        baseline_outer_df.groupby(label_col)[metric]
        .mean()
        .sort_values(ascending=ascending)
        .index.tolist()
    )

    n_baselines = len(order)
    fig, ax = plt.subplots(figsize=(max(8, n_baselines * 1.6), 6))
    sns.boxplot(
        data=baseline_outer_df,
        x=label_col,
        y=metric,
        order=order,
        palette=sns.color_palette("Set2", n_baselines),
        width=0.55,
        linewidth=0.8,
        fliersize=0,
        showmeans=True,
        meanline=True,
        meanprops=BOXPLOT_MEAN_PROPS,
        ax=ax,
    )
    sns.stripplot(
        data=baseline_outer_df,
        x=label_col,
        y=metric,
        order=order,
        color="black",
        dodge=False,
        size=5,
        alpha=0.75,
        ax=ax,
    )
    ax.set_xlabel("")
    ax.set_ylabel(metric)
    ax.set_title(
        f"Nested LOWO per-well {metric}: petrophysical baselines "
        "(red line = mean)"
    )
    ax.set_xticks(range(n_baselines))
    ax.set_xticklabels(order, rotation=20, ha="right")
    _save_fig(fig, fig_dir, stem)
    return fig_dir / "svg" / f"{stem}.svg"


def fig_per_well_heatmap(
    nested_df: pd.DataFrame,
    fig_dir: Path,
    metric: str = "RMSE_log",
    stem: Optional[str] = None,
    summary_df: Optional[pd.DataFrame] = None,
    row_order: Optional[list[int]] = None,
) -> None:
    """Per-well outer-fold metric heatmap, sorted by mean (best at top) with Well Average row.

    Styling mirrors ``analyze_phase2.save_figures``: RdYlGn (or RdYlGn_r for minimize
    metrics), x-axis ticks moved to the top, ``Well Average`` summary row appended at
    the bottom, and 3-decimal annotations.
    """
    outer_col = f"outer_{metric}"
    if outer_col not in nested_df.columns:
        return
    fig_stem = stem or f"phase4_outer_{metric.lower()}_heatmap"

    pivot = nested_df.pivot_table(
        index="subset_id",
        columns="outer_fold",
        values=outer_col,
        aggfunc="mean",
    )
    if pivot.empty:
        return
    pivot = pivot.loc[:, sorted(pivot.columns)]
    pivot.columns = [f"Well {c}" for c in pivot.columns]

    # Decide row order. A caller-supplied ``row_order`` (a fixed canonical order,
    # e.g. by RMSE_log mean) takes priority so every heatmap lists subsets in the
    # same order regardless of the displayed metric. Otherwise fall back to the
    # per-metric mean ordering.
    if row_order is not None:
        ordered_ids = [int(sid) for sid in row_order if int(sid) in pivot.index]
        ordered_ids += [sid for sid in pivot.index if sid not in ordered_ids]
    elif summary_df is not None and f"{metric}_mean" in summary_df.columns:
        ascending = metric in MINIMIZE_METRICS
        ordered_ids = (
            summary_df.sort_values(f"{metric}_mean", ascending=ascending)["subset_id"]
            .astype(int)
            .tolist()
        )
        ordered_ids = [sid for sid in ordered_ids if sid in pivot.index]
    else:
        row_means = pivot.mean(axis=1)
        ascending = metric in MINIMIZE_METRICS
        ordered_ids = row_means.sort_values(ascending=ascending).index.tolist()
    pivot = pivot.loc[ordered_ids]

    label_lookup: dict[int, str] = {}
    if summary_df is not None and "subset_label" in summary_df.columns:
        label_lookup = {
            int(r["subset_id"]): _format_subset_label(
                int(r["subset_id"]), str(r["subset_label"])
            )
            for _, r in summary_df.iterrows()
        }
    pivot.index = [label_lookup.get(int(sid), str(sid)) for sid in pivot.index]

    avg_row = pivot.mean(axis=0)
    avg_row.name = "Well Average"
    hm_df = pd.concat([pivot, avg_row.to_frame().T])

    cmap = "RdYlGn_r" if metric in MINIMIZE_METRICS else "RdYlGn"
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
    ax.set_title(f"Nested LOWO per-well {metric} by subset")
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
    _save_fig(fig, fig_dir, fig_stem)


def _build_well_color_map(wells: list[str]) -> dict[str, tuple]:
    """Return a stable color mapping from well name to RGB tuple."""
    palette = sns.color_palette("tab10", n_colors=max(len(wells), 7))
    return {w: palette[i % len(palette)] for i, w in enumerate(wells)}


def load_predictions_for_subset(run_dir: Path, subset_id: int) -> pd.DataFrame:
    """Load all per-well prediction CSVs for a given subset id and concatenate."""
    pred_dir = run_dir / "predictions" / str(subset_id)
    if not pred_dir.exists():
        return pd.DataFrame()
    frames = []
    for csv_path in sorted(pred_dir.glob("*.csv")):
        try:
            frames.append(pd.read_csv(csv_path))
        except Exception:
            continue
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def discover_prediction_subset_ids(run_dir: Path) -> list[int]:
    """Return subset ids that have a populated predictions directory, sorted ascending."""
    pred_root = run_dir / "predictions"
    if not pred_root.exists():
        return []
    found = []
    for child in pred_root.iterdir():
        if not child.is_dir():
            continue
        if not child.name.lstrip("-").isdigit():
            continue
        if any(child.glob("*.csv")):
            found.append(int(child.name))
    return sorted(found)


def load_zone_lookup(config: dict, project_root: Path) -> Optional[pd.DataFrame]:
    """Return a ``row_index -> {Zone, DEPTH}`` lookup from the variant file.

    Predictions persist a ``row_index`` that is the positional index into the
    variant dataframe (``pd.read_csv`` default RangeIndex), exactly matching
    how the nested runner records ``outer_test_df.index``. We reload the same
    variant and expose its ``Zone`` column (and ``DEPTH`` when present) under
    that index so per-row predictions can be attributed to geological zones,
    and so the positional join can be sanity-checked against ``DEPTH``.

    Returns ``None`` when the variant file or ``Zone`` column is unavailable.
    """
    best_variant = config.get("best_variant")
    if not best_variant:
        return None
    variant_path = project_root / best_variant
    if not variant_path.exists():
        return None
    df = pd.read_csv(variant_path)
    if "Zone" not in df.columns:
        return None
    cols = ["Zone"] + (["DEPTH"] if "DEPTH" in df.columns else [])
    lookup = df[cols].copy()
    lookup.index = range(len(lookup))
    return lookup


def build_per_zone_metrics(
    run_dir: Path,
    subset_ids: list[int],
    zone_lookup: pd.DataFrame,
    summary_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Aggregate held-out predictions into per-zone metrics for each subset.

    Predictions are joined to zones via ``row_index`` and grouped by ``Zone``;
    each group is scored with the standard :func:`compute_metrics` path so the
    numbers are directly comparable to the pooled nested metrics. The output is
    a long-form table (one row per subset x zone) carrying all METRIC_NAMES
    plus a ``n`` sample count.

    As a sanity check on the positional ``row_index`` join, when both the
    predictions and the variant lookup carry ``DEPTH`` we assert the depths
    agree row-for-row; a mismatch would indicate the predictions and variant
    are misaligned and any zone attribution would be wrong.
    """
    if zone_lookup is None or len(subset_ids) == 0:
        return pd.DataFrame()

    zone_series = zone_lookup["Zone"]
    depth_series = zone_lookup["DEPTH"] if "DEPTH" in zone_lookup.columns else None

    label_lookup: dict[int, str] = {}
    if summary_df is not None and not summary_df.empty:
        label_lookup = {
            int(r["subset_id"]): r.get("subset_label")
            for _, r in summary_df.iterrows()
        }

    rows = []
    for subset_id in subset_ids:
        preds = load_predictions_for_subset(run_dir, subset_id)
        if preds.empty or "row_index" not in preds.columns:
            continue
        preds = preds.copy()

        # Sanity check: the positional row_index join must line up with the
        # variant. DEPTH is an independent column carried in both, so equal
        # depths confirm the join is valid before we trust the Zone mapping.
        if depth_series is not None and "DEPTH" in preds.columns:
            expected_depth = preds["row_index"].map(depth_series).to_numpy(dtype=float)
            actual_depth = preds["DEPTH"].to_numpy(dtype=float)
            mismatch = ~np.isclose(
                expected_depth, actual_depth, rtol=0.0, atol=1e-14, equal_nan=True
            )
            if mismatch.any():
                bad = preds.loc[mismatch, "row_index"].to_numpy()[:5]
                raise AssertionError(
                    f"DEPTH mismatch between predictions and variant for "
                    f"subset {subset_id} ({int(mismatch.sum())} rows); the "
                    f"row_index->Zone join is misaligned. First offending "
                    f"row_index values: {bad.tolist()}"
                )

        preds["Zone"] = preds["row_index"].map(zone_series)
        preds = preds.dropna(subset=["Zone"])
        if preds.empty:
            continue

        for zone, grp in preds.groupby("Zone"):
            metrics = compute_metrics(
                y_true=grp["y_true_raw"].to_numpy(),
                y_pred=grp["y_pred_raw"].to_numpy(),
                y_true_log=grp["y_true_log"].to_numpy(),
                y_pred_log=grp["y_pred_log"].to_numpy(),
            )
            try:
                zone_float = float(zone)
                zone_label = int(zone_float) if zone_float.is_integer() else zone_float
            except (TypeError, ValueError):
                zone_label = zone
            row = {
                "subset_id": int(subset_id),
                "subset_label": label_lookup.get(int(subset_id), ""),
                "Zone": zone_label,
                "n": int(len(grp)),
                "n_wells": int(grp["outer_fold"].nunique())
                if "outer_fold" in grp.columns
                else np.nan,
            }
            for metric in METRIC_NAMES:
                row[metric] = metrics.get(metric, np.nan)
            rows.append(row)

    if not rows:
        return pd.DataFrame()

    out = pd.DataFrame(rows)
    return out.sort_values(["subset_id", "Zone"]).reset_index(drop=True)


def fig_per_zone_metric(
    per_zone_df: pd.DataFrame,
    fig_dir: Path,
    metric: str = "RMSE_log",
    stem: str = "phase4_per_zone_rmse_log",
    row_order: Optional[list[int]] = None,
    min_samples: int = 2,
) -> Optional[Path]:
    """Per-zone metric heatmap (subsets x zones), styled like the per-well heatmap.

    Rows are operating-point subsets (best mean across zones at top), columns are
    geological zones, with a ``Zone Average`` summary row appended at the bottom.
    Mirrors ``fig_per_well_heatmap`` (RdYlGn_r for minimize metrics, x-ticks on
    top, 3-decimal annotations) for cross-figure consistency. Returns the SVG
    path, or ``None`` when the metric/columns are unavailable.
    """
    if per_zone_df.empty or metric not in per_zone_df.columns:
        return None
    if "Zone" not in per_zone_df.columns:
        return None

    df = per_zone_df.copy()

    # Drop zones with fewer than ``min_samples`` pooled samples (e.g. the single
    # sample in the M2-M3 transition zone). Such cells are statistically
    # unreliable and R2 is undefined for them. The full per-zone counts remain in
    # per_zone_metrics_all.csv and the supplementary core-count table.
    if min_samples > 1 and "n" in df.columns:
        zone_counts = df.groupby("Zone")["n"].max()
        keep_zones = set(zone_counts[zone_counts >= min_samples].index)
        df = df[df["Zone"].isin(keep_zones)]
    if df.empty:
        return None

    pivot = df.pivot_table(
        index="subset_id",
        columns="Zone",
        values=metric,
        aggfunc="mean",
    )
    if pivot.empty:
        return None

    # Numeric-aware zone ordering, then pretty column labels.
    def _zone_key(z):
        s = str(z)
        return (not s.lstrip("-").isdigit(), float(s) if s.lstrip("-").isdigit() else s)

    pivot = pivot.loc[:, sorted(pivot.columns, key=_zone_key)]
    pivot.columns = [f"Zone {c}" for c in pivot.columns]

    # Row order. A caller-supplied canonical ``row_order`` takes priority so the
    # per-zone heatmaps list subsets identically to the per-well heatmaps;
    # otherwise fall back to best mean across zones at top.
    if row_order is not None:
        ordered_ids = [int(sid) for sid in row_order if int(sid) in pivot.index]
        ordered_ids += [sid for sid in pivot.index if sid not in ordered_ids]
    else:
        ascending = metric in MINIMIZE_METRICS
        row_means = pivot.mean(axis=1)
        ordered_ids = row_means.sort_values(ascending=ascending).index.tolist()
    pivot = pivot.loc[ordered_ids]

    # Row labels consistent with the other Phase 4 figures.
    label_lookup: dict[int, str] = {}
    if "subset_label" in df.columns:
        label_lookup = {
            int(r["subset_id"]): _format_subset_label(
                int(r["subset_id"]), str(r["subset_label"])
            )
            for _, r in df.iterrows()
        }
    pivot.index = [label_lookup.get(int(sid), str(sid)) for sid in pivot.index]

    avg_row = pivot.mean(axis=0)
    avg_row.name = "Zone Average"
    hm_df = pd.concat([pivot, avg_row.to_frame().T])

    cmap = "RdYlGn_r" if metric in MINIMIZE_METRICS else "RdYlGn"
    fig, ax = plt.subplots(
        figsize=(max(9, hm_df.shape[1] * 1.1), max(3.5, len(hm_df) * 0.55))
    )
    sns.heatmap(
        hm_df,
        annot=True,
        fmt=".3f",
        cmap=cmap,
        linewidths=0.5,
        cbar_kws={"label": metric},
        ax=ax,
    )
    ax.set_title(f"Nested LOWO per-zone {metric} by subset")
    ax.set_xlabel("")
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
    _save_fig(fig, fig_dir, stem)
    return fig_dir / "svg" / f"{stem}.svg"


def fig_ablation_delta(
    ablation_comparison_df: pd.DataFrame,
    fig_dir: Path,
    metric: str = "RMSE_log",
    stem: str = "phase4_ablation_delta_rmse_log",
) -> Optional[Path]:
    """Grouped bar chart of ``delta_vs_full`` for one metric, by config.

    One bar group per ablation config (covariate dropped), hue = subset. A
    positive RMSE_log delta means dropping that covariate degraded the model
    relative to the full feature set. Returns the SVG path, or ``None`` when
    the requested metric has no rows.
    """
    if ablation_comparison_df.empty:
        return None
    plot_df = ablation_comparison_df[ablation_comparison_df["metric"] == metric].copy()
    if plot_df.empty:
        return None

    config_order = ["no_source", "no_zone", "no_depth", "no_all_three"]
    present = [c for c in config_order if c in set(plot_df["config_slug"])]
    if not present:
        present = sorted(plot_df["config_slug"].unique())

    # Legend uses the same "<id> (<short>)" labels as the heatmaps (e.g.
    # "640 (Budget wireline)"), so the figure stays consistent and never shows
    # the word "baseline" (reserved for the petrophysical baseline models) or
    # raw underscored slugs.
    hue_col = "subset_display"
    if plot_df.get("subset_label", pd.Series(dtype=object)).notna().any():
        plot_df[hue_col] = plot_df.apply(
            lambda r: _format_subset_label(int(r["subset_id"]), str(r["subset_label"])),
            axis=1,
        )
    else:
        plot_df[hue_col] = plot_df["subset_id"].astype(str)
    hue_order = [
        disp for _, disp in sorted(
            {int(r["subset_id"]): r[hue_col] for _, r in plot_df.iterrows()}.items()
        )
    ]
    n_hue = len(hue_order)

    fig, ax = plt.subplots(figsize=(max(8, len(present) * 1.6), 6))
    sns.barplot(
        data=plot_df,
        x="config_slug",
        y="delta_vs_full",
        hue=hue_col,
        order=present,
        hue_order=hue_order,
        palette=sns.color_palette("Set2", n_hue),
        edgecolor="black",
        linewidth=0.5,
        ax=ax,
    )
    ax.axhline(0.0, color="black", lw=1.0, zorder=1)
    worse = "higher" if metric in MINIMIZE_METRICS else "lower"
    config_display = {
        "no_source": "Drop source",
        "no_zone": "Drop zone",
        "no_depth": "Drop depth",
        "no_all_three": "Drop source,\nzone, and depth",
    }
    ax.set_xticks(range(len(present)))
    ax.set_xticklabels([config_display.get(c, c) for c in present])
    ax.set_xlabel("Ablation configuration")
    ax.set_ylabel(f"{metric} change vs all-covariates model")
    ax.set_title(
        f"Nested LOWO structural-covariate ablation:\n{metric} change vs "
        f"all-covariates model ({worse} = worse)"
    )
    ax.legend(title="Subset", fontsize=8, title_fontsize=9, loc="best", framealpha=0.9)
    _save_fig(fig, fig_dir, stem)
    return fig_dir / "svg" / f"{stem}.svg"


def _compute_log_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float, int]:
    """Return (RMSE_log, R2_log, n) for an array of log-space residuals."""
    n = int(len(y_true))
    if n == 0:
        return float("nan"), float("nan"), 0
    diff = y_true - y_pred
    rmse = float(np.sqrt(np.mean(diff * diff)))
    ss_res = float(np.sum(diff * diff))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return rmse, r2, n


def _draw_pred_vs_actual_panel(
    ax: plt.Axes,
    df: pd.DataFrame,
    title: str,
    color_map: Optional[dict[str, tuple]],
    axis_min: float,
    axis_max: float,
    show_well_legend: bool = False,
    display_metrics: Optional[tuple[float, float]] = None,
) -> None:
    """Render a single predicted vs measured log-permeability scatter with reference bands.

    The scatter always shows every pooled held-out point. The annotated
    ``RMSE_log`` / ``R2_log``, however, default to the equal-weighted mean over
    the outer LOWO folds when ``display_metrics`` is supplied as
    ``(rmse_mean, r2_mean)``. This keeps the panel numbers identical to the
    headline nested summary (and the boxplot/heatmap), rather than the
    sample-weighted pooled metric computed over the concatenated points, which
    down-weights the small, hard wells and would read lower. When
    ``display_metrics`` is ``None`` (e.g. the per-well grid, where one panel is a
    single fold and pooled == per-fold), the pooled metric is used.
    """
    if df.empty:
        ax.set_visible(False)
        return

    if color_map is None:
        ax.scatter(
            df["y_true_log"],
            df["y_pred_log"],
            s=14,
            alpha=0.6,
            color="steelblue",
            edgecolors="white",
            linewidths=0.3,
        )
    else:
        for well, sub in df.groupby("outer_fold"):
            ax.scatter(
                sub["y_true_log"],
                sub["y_pred_log"],
                s=14,
                alpha=0.7,
                color=color_map.get(well, "gray"),
                label=str(well),
                edgecolors="white",
                linewidths=0.3,
            )
        if show_well_legend:
            ax.legend(
                title="Held-out well",
                fontsize=8,
                title_fontsize=8,
                loc="lower right",
                framealpha=0.85,
                ncol=2,
            )

    pad = 0.08 * (axis_max - axis_min) if axis_max > axis_min else 0.5
    lo = axis_min - pad
    hi = axis_max + pad
    line_x = np.array([lo, hi])

    ax.plot(line_x, line_x, color="black", linestyle="-", lw=1.0, zorder=1)
    ax.plot(line_x, line_x + 0.5, color="black", linestyle="--", lw=0.7, alpha=0.55, zorder=1)
    ax.plot(line_x, line_x - 0.5, color="black", linestyle="--", lw=0.7, alpha=0.55, zorder=1)
    ax.plot(line_x, line_x + 1.0, color="black", linestyle=":", lw=0.7, alpha=0.45, zorder=1)
    ax.plot(line_x, line_x - 1.0, color="black", linestyle=":", lw=0.7, alpha=0.45, zorder=1)

    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(r"Measured $\log_{10}(k)$")
    ax.set_ylabel(r"Predicted $\log_{10}(k)$")
    ax.set_title(title, fontsize=10)

    rmse_pooled, r2_pooled, n = _compute_log_metrics(
        df["y_true_log"].to_numpy(),
        df["y_pred_log"].to_numpy(),
    )
    if display_metrics is not None:
        rmse, r2 = display_metrics
    else:
        rmse, r2 = rmse_pooled, r2_pooled
    annotation = (
        f"RMSE$_{{\\log}}$ = {rmse:.3f}\n"
        f"$R^2_{{\\log}}$ = {r2:.3f}\n"
        f"$n$ = {n}"
    )
    ax.text(
        0.04,
        0.96,
        annotation,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=8.5,
        bbox=dict(
            boxstyle="round,pad=0.35",
            facecolor="white",
            alpha=0.85,
            edgecolor="lightgray",
        ),
    )


def _grid_layout_for(n: int) -> tuple[int, int]:
    """Pick a reasonable (nrows, ncols) grid for n panels."""
    if n <= 1:
        return 1, max(n, 1)
    if n == 2:
        return 1, 2
    if n in (3, 4):
        return 2, 2
    if n in (5, 6):
        return 2, 3
    if n in (7, 8):
        return 2, 4
    ncols = 4
    nrows = (n + ncols - 1) // ncols
    return nrows, ncols


def fig_pred_vs_actual_grid(
    run_dir: Path,
    summary_df: pd.DataFrame,
    fig_dir: Path,
    subset_ids: Optional[list[int]] = None,
    stem: str = "phase4_pred_vs_actual_grid",
) -> Optional[Path]:
    """Option A: one panel per feature subset; points colored by held-out well.

    Args:
        run_dir: Phase 4 run directory containing ``predictions/{subset_id}/`` folders.
        summary_df: Aggregated nested summary used for cost / label / panel ordering.
        fig_dir: Figures output directory.
        subset_ids: Restrict panels to this list (default: all available).
        stem: Output filename stem.

    Returns:
        Path to the SVG output, or ``None`` if no predictions were available.
    """
    available = discover_prediction_subset_ids(run_dir)
    if not available:
        return None

    if subset_ids is not None:
        wanted = [int(s) for s in subset_ids]
        available = [s for s in available if s in wanted]
    if not available:
        return None

    cost_lookup = {
        int(r["subset_id"]): float(r.get("cost", 0))
        for _, r in summary_df.iterrows()
    }
    label_lookup = {
        int(r["subset_id"]): str(r.get("subset_label", f"subset_{int(r['subset_id'])}"))
        for _, r in summary_df.iterrows()
    }
    # Headline (equal-weighted over outer folds) metrics, so each panel annotation
    # matches the nested summary / boxplot / heatmap rather than the pooled,
    # sample-weighted value recomputed from the concatenated points.
    metric_lookup: dict[int, tuple[float, float]] = {}
    for _, r in summary_df.iterrows():
        rmse_mean = r.get("RMSE_log_mean")
        r2_mean = r.get("R2_log_mean")
        if pd.notna(rmse_mean) and pd.notna(r2_mean):
            metric_lookup[int(r["subset_id"])] = (float(rmse_mean), float(r2_mean))
    available.sort(key=lambda s: cost_lookup.get(s, float("inf")))

    panel_data: dict[int, pd.DataFrame] = {}
    for sid in available:
        df = load_predictions_for_subset(run_dir, sid)
        if not df.empty:
            panel_data[sid] = df
    if not panel_data:
        return None

    all_vals = np.concatenate(
        [df["y_true_log"].to_numpy() for df in panel_data.values()]
        + [df["y_pred_log"].to_numpy() for df in panel_data.values()]
    )
    axis_min = float(np.nanmin(all_vals))
    axis_max = float(np.nanmax(all_vals))

    wells_present = sorted(
        {w for df in panel_data.values() for w in df["outer_fold"].astype(str).unique()}
    )
    color_map = _build_well_color_map(wells_present or WELL_NAMES_DEFAULT)

    nrows, ncols = _grid_layout_for(len(panel_data))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(5.4 * ncols, 5.4 * nrows),
        squeeze=False,
    )
    axes_flat = axes.flatten()

    for idx, (sid, df) in enumerate(panel_data.items()):
        ax = axes_flat[idx]
        label = label_lookup.get(sid, f"subset_{sid}")
        cost = cost_lookup.get(sid, float("nan"))
        cost_text = f"cost {int(cost)}" if not np.isnan(cost) else ""
        title = (
            f"({PANEL_LETTERS[idx]}) {_format_subset_label(sid, label)}; {cost_text}"
        ).strip("; ")
        _draw_pred_vs_actual_panel(
            ax,
            df,
            title,
            color_map=color_map,
            axis_min=axis_min,
            axis_max=axis_max,
            show_well_legend=(idx == 0),
            display_metrics=metric_lookup.get(sid),
        )

    for idx in range(len(panel_data), len(axes_flat)):
        axes_flat[idx].set_visible(False)

    fig.suptitle(
        r"Nested LOWO: Predicted vs Measured $\log_{10}(k)$",
        fontsize=12,
        y=1.0,
    )
    fig.tight_layout()
    _save_fig(fig, fig_dir, stem)
    return fig_dir / "svg" / f"{stem}.svg"


def fig_pred_vs_actual_per_well(
    run_dir: Path,
    summary_df: pd.DataFrame,
    fig_dir: Path,
    detail_subset_id: int,
    stem: Optional[str] = None,
) -> Optional[Path]:
    """Option B: one panel per held-out well for a single headline subset.

    Args:
        run_dir: Phase 4 run directory containing ``predictions/{subset_id}/`` folders.
        summary_df: Aggregated nested summary used for the suptitle / labels.
        fig_dir: Figures output directory.
        detail_subset_id: Subset id whose per-well predictions are panelized.
        stem: Optional output filename stem; defaults to per-well name with subset id.

    Returns:
        Path to the SVG output, or ``None`` if predictions for ``detail_subset_id``
        were not found.
    """
    df = load_predictions_for_subset(run_dir, detail_subset_id)
    if df.empty:
        return None

    fig_stem = stem or f"phase4_pred_vs_actual_per_well_{detail_subset_id}"
    label_lookup = {
        int(r["subset_id"]): str(r.get("subset_label", f"subset_{int(r['subset_id'])}"))
        for _, r in summary_df.iterrows()
    }
    cost_lookup = {
        int(r["subset_id"]): float(r.get("cost", 0))
        for _, r in summary_df.iterrows()
    }

    wells = sorted(df["outer_fold"].astype(str).unique())
    color_map = _build_well_color_map(wells or WELL_NAMES_DEFAULT)

    axis_min = float(min(df["y_true_log"].min(), df["y_pred_log"].min()))
    axis_max = float(max(df["y_true_log"].max(), df["y_pred_log"].max()))

    nrows, ncols = _grid_layout_for(len(wells))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(4.2 * ncols, 4.2 * nrows),
        squeeze=False,
    )
    axes_flat = axes.flatten()

    for idx, well in enumerate(wells):
        ax = axes_flat[idx]
        sub = df[df["outer_fold"].astype(str) == well]
        title = f"({PANEL_LETTERS[idx]}) Well {well}"
        _draw_pred_vs_actual_panel(
            ax,
            sub,
            title,
            color_map={well: color_map.get(well, "steelblue")},
            axis_min=axis_min,
            axis_max=axis_max,
            show_well_legend=False,
        )

    for idx in range(len(wells), len(axes_flat)):
        axes_flat[idx].set_visible(False)

    sub_label = label_lookup.get(detail_subset_id, f"subset_{detail_subset_id}")
    cost = cost_lookup.get(detail_subset_id, float("nan"))
    cost_text = f"cost {int(cost)}" if not np.isnan(cost) else ""
    fig.suptitle(
        r"Nested LOWO: Per-well Predicted vs Measured $\log_{10}(k)$"
        + "\n"
        + f"{_format_subset_label(detail_subset_id, sub_label)}"
        + (f"; {cost_text}" if cost_text else ""),
        fontsize=12,
        y=1.01,
    )
    fig.tight_layout()
    _save_fig(fig, fig_dir, fig_stem)
    return fig_dir / "svg" / f"{fig_stem}.svg"


def generate_report(
    config: dict,
    summary_df: pd.DataFrame,
    selection_summary: pd.DataFrame,
    bias_df: pd.DataFrame,
    table_paths: dict[str, Path],
    figure_paths: dict[str, Path],
    display_families: list[str],
    metric_scope: str,
    baseline_comparison_df: Optional[pd.DataFrame] = None,
    per_zone_df: Optional[pd.DataFrame] = None,
    ablation_comparison_df: Optional[pd.DataFrame] = None,
    ablation_nested_summary_df: Optional[pd.DataFrame] = None,
) -> str:
    lines = [
        "# Phase 4 Generalization Validation Report",
        "",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Configuration",
        "",
        f"- **Model**: {config.get('best_model', 'N/A')}",
        f"- **Variant**: `{config.get('best_variant', 'N/A')}`",
        f"- **HP budget**: {config.get('hp_budget', 'N/A')}",
        f"- **Candidate subsets**: {len(config.get('candidate_subsets', []))}",
        f"- **Metric scope**: `{metric_scope}`",
        "",
    ]

    for family in display_families:
        primary_metric = primary_metric_for_family(family)
        mae_metric = mae_metric_for_family(family)
        secondary_metric = secondary_metric_for_family(family)
        ci_col = f"{primary_metric}_ci"
        section_title = (
            "Log-space" if family == "log" else "Original-space"
        )

        lines.extend(
            [
                f"## Nested Summary ({section_title} view)",
                "",
            ]
        )

        show_cols = [
            "subset_id",
            "subset_label",
            "cost",
            "n_outer_folds",
            f"{primary_metric}_mean",
            f"{primary_metric}_std",
            ci_col,
            f"{mae_metric}_mean",
            f"{secondary_metric}_mean",
        ]
        avail = [c for c in show_cols if c in summary_df.columns]
        view_df = (
            summary_df.sort_values([ci_col, "cost"])
            if ci_col in summary_df.columns
            else summary_df
        )
        if avail:
            lines.append("| " + " | ".join(avail) + " |")
            lines.append("| " + " | ".join("---" for _ in avail) + " |")
            for _, row in view_df.iterrows():
                cells = []
                for c in avail:
                    val = row[c]
                    if isinstance(val, float):
                        cells.append(f"{val:.4f}")
                    else:
                        cells.append(str(val))
                lines.append("| " + " | ".join(cells) + " |")
        else:
            lines.append("(no summary rows)")
        lines.append("")

    if not selection_summary.empty:
        lines.extend(
            [
                "## Outer-fold Winner Frequency",
                "",
                "| selected_subset_id | selected_subset_label | selected_outer_folds |",
                "| --- | --- | --- |",
            ]
        )
        for _, row in selection_summary.iterrows():
            lines.append(
                "| "
                f"{int(row['selected_subset_id'])} | {row['selected_subset_label']} | "
                f"{int(row['selected_outer_folds'])} |"
            )
        lines.append("")

    if not bias_df.empty:
        top_bias = bias_df[bias_df["metric"] == "RMSE_log"].copy()
        top_bias = top_bias.sort_values(["reference_source", "subset_id"])
        lines.extend(
            [
                "## Bias Comparison (Nested vs Selection-stage references)",
                "",
                "| subset_id | subset_label | reference_source | nested_mean | reference_mean | delta_nested_minus_reference |",
                "| --- | --- | --- | --- | --- | --- |",
            ]
        )
        for _, row in top_bias.iterrows():
            lines.append(
                "| "
                f"{int(row['subset_id'])} | {row['subset_label']} | {row['reference_source']} | "
                f"{row['nested_mean']:.4f} | {row['reference_mean']:.4f} | {row['delta_nested_minus_reference']:.4f} |"
            )
        lines.append("")

    if baseline_comparison_df is not None and not baseline_comparison_df.empty:
        base_view = baseline_comparison_df[
            baseline_comparison_df["metric"] == "RMSE_log"
        ].copy()
        base_view = base_view.sort_values(["baseline_slug", "subset_id"])
        lines.extend(
            [
                "## Petrophysical / Regression Baseline Comparison "
                "(Nested vs classical baselines)",
                "",
                "Negative `delta_nested_minus_baseline` (RMSE_log) means the "
                "nested ML pipeline beats the classical baseline at that "
                "operating point.",
                "",
                "| baseline_label | subset_id | subset_label | nested_mean | baseline_mean | delta_nested_minus_baseline |",
                "| --- | --- | --- | --- | --- | --- |",
            ]
        )
        for _, row in base_view.iterrows():
            lines.append(
                "| "
                f"{row['baseline_label']} | {int(row['subset_id'])} | {row['subset_label']} | "
                f"{row['nested_mean']:.4f} | {row['baseline_mean']:.4f} | "
                f"{row['delta_nested_minus_baseline']:.4f} |"
            )
        lines.append("")

    if per_zone_df is not None and not per_zone_df.empty:
        lines.extend(
            [
                "## Per-Zone Error Breakdown (held-out nested predictions)",
                "",
                "RMSE_log / R2_log / MAE_log per geological zone, by operating "
                "point. `n` is the number of held-out samples in that zone.",
                "",
                "| subset_id | subset_label | Zone | n | RMSE_log | R2_log | MAE_log |",
                "| --- | --- | --- | --- | --- | --- | --- |",
            ]
        )
        for _, row in per_zone_df.iterrows():
            lines.append(
                "| "
                f"{int(row['subset_id'])} | {row['subset_label']} | {row['Zone']} | "
                f"{int(row['n'])} | {row['RMSE_log']:.4f} | {row['R2_log']:.4f} | "
                f"{row['MAE_log']:.4f} |"
            )
        lines.append("")

    if ablation_comparison_df is not None and not ablation_comparison_df.empty:
        abl_view = ablation_comparison_df[
            ablation_comparison_df["metric"] == "RMSE_log"
        ].copy()
        lines.extend(
            [
                "## Structural-Covariate Ablation (nested, delta vs full)",
                "",
                "Nested-LOWO RMSE_log change when each structural covariate is "
                "dropped, relative to the canonical full-covariate nested run "
                "(reused, not recomputed). Positive `delta_vs_full` means "
                "dropping the covariate **hurt** generalization. The CSV "
                "`ablation_comparison.csv` carries deltas for all metrics, not "
                "just RMSE_log.",
                "",
                "| subset_id | subset_label | config | dropped | full_mean | config_mean | delta_vs_full |",
                "| --- | --- | --- | --- | --- | --- | --- |",
            ]
        )
        for _, row in abl_view.iterrows():
            lines.append(
                "| "
                f"{int(row['subset_id'])} | {row['subset_label']} | {row['config_slug']} | "
                f"{row['dropped']} | {row['full_mean']:.4f} | {row['config_mean']:.4f} | "
                f"{row['delta_vs_full']:+.4f} |"
            )
        lines.append("")
        # Cross-link the aggregate -Zone effect to the spatial per-zone view.
        if per_zone_df is not None and not per_zone_df.empty:
            lines.extend(
                [
                    "The `no_zone` rows quantify the *aggregate* effect of "
                    "removing Zone as a feature; the Per-Zone Error Breakdown "
                    "above (and `per_zone_metrics_*.csv`) shows *where* that "
                    "signal concentrates across zones.",
                    "",
                ]
            )

        if (
            ablation_nested_summary_df is not None
            and not ablation_nested_summary_df.empty
        ):
            lines.extend(
                [
                    "### Ablated nested configurations (per subset x config)",
                    "",
                    "| subset_id | config | dropped | RMSE_log_mean | RMSE_log_std | R2_log_mean |",
                    "| --- | --- | --- | --- | --- | --- |",
                ]
            )
            for _, row in ablation_nested_summary_df.iterrows():
                lines.append(
                    "| "
                    f"{int(row.get('subset_id'))} | {row.get('config_slug')} | "
                    f"{row.get('dropped')} | "
                    f"{row.get('RMSE_log_mean', float('nan')):.4f} | "
                    f"{row.get('RMSE_log_std', float('nan')):.4f} | "
                    f"{row.get('R2_log_mean', float('nan')):.4f} |"
                )
            lines.append("")

    lines.extend(["## Output Files", "", "### Tables", ""])
    for key, path in sorted(table_paths.items()):
        lines.append(f"- **{key}**: `{path}`")
    lines.extend(["", "### Figures", ""])
    for key, path in sorted(figure_paths.items()):
        lines.append(f"- **{key}**: `{path}`")
    lines.append("")
    return "\n".join(lines)


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


def generate_html_predictions_subpage(
    summary_df: pd.DataFrame,
    output_dir: Path,
    metric_scope: str,
    pred_subset_ids: Optional[list[int]] = None,
    main_report_filename: Optional[str] = None,
) -> Optional[Path]:
    """Generate a self-contained HTML sub-page for the predicted-vs-actual figures.

    The sub-page includes:
      - The Option-A 4-panel grid (one panel per subset, points colored by held-out well).
      - Per-subset Option-B grids (one figure per subset; one panel per held-out well).
      - A short interpretation / "what is a low RMSE for this problem?" note.

    Returns the sub-page path, or ``None`` if no predicted-vs-actual figures were found.
    """
    fig_dir = output_dir / "figures"
    grid_stem = "phase4_pred_vs_actual_grid"
    grid_present = (fig_dir / "png" / f"{grid_stem}.png").exists()

    if pred_subset_ids is not None:
        candidate_ids = list(pred_subset_ids)
    else:
        candidate_ids = sorted(int(s) for s in summary_df["subset_id"].astype(int).tolist())

    cost_lookup = {
        int(r["subset_id"]): float(r.get("cost", 0))
        for _, r in summary_df.iterrows()
    }
    label_lookup = {
        int(r["subset_id"]): str(r.get("subset_label", f"subset_{int(r['subset_id'])}"))
        for _, r in summary_df.iterrows()
    }
    candidate_ids.sort(key=lambda s: cost_lookup.get(s, float("inf")))

    per_well_entries: list[tuple[int, str, float, str]] = []
    for sid in candidate_ids:
        stem = f"phase4_pred_vs_actual_per_well_{sid}"
        if (fig_dir / "png" / f"{stem}.png").exists():
            per_well_entries.append(
                (
                    sid,
                    label_lookup.get(sid, f"subset_{sid}"),
                    cost_lookup.get(sid, float("nan")),
                    stem,
                )
            )

    if not grid_present and not per_well_entries:
        return None

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Per-subset summary table (RMSE_log / R2_log / cost), surfacing the same
    # numerics that get annotated on the per-well plots.
    summary_cols = [
        "subset_id",
        "subset_label",
        "cost",
        "n_outer_folds",
        "RMSE_log_mean",
        "R2_log_mean",
    ]
    if not summary_df.empty:
        ordered = summary_df.copy()
        ordered["subset_id"] = ordered["subset_id"].astype(int)
        ordered = ordered[ordered["subset_id"].isin(candidate_ids)]
        if "RMSE_log_mean" in ordered.columns:
            ordered = ordered.sort_values("RMSE_log_mean")
    else:
        ordered = summary_df
    summary_block = _html_table(ordered, summary_cols)

    interp_block = """
<div class="alert alert-secondary mb-4" role="alert">
  <h6 class="alert-heading mb-2">How to read these plots</h6>
  <ul class="mb-0 small">
    <li>Solid diagonal: perfect prediction (<code>y = x</code>).</li>
    <li>Dashed band: <code>&plusmn; 0.5</code> in <code>log<sub>10</sub></code>, a factor-3 residual envelope (commonly considered <em>good</em> for carbonate permeability).</li>
    <li>Dotted band: <code>&plusmn; 1.0</code> in <code>log<sub>10</sub></code>, one full decade (commonly considered <em>borderline</em>).</li>
    <li>Each point is one held-out core measurement aggregated across the seven outer LOWO folds; colors identify the held-out well so per-well bias and dispersion are visible alongside the aggregate fit.</li>
    <li>Annotations report <code>RMSE<sub>log</sub></code>, <code>R<sup>2</sup><sub>log</sub></code>, and <code>n</code>; an <code>RMSE<sub>log</sub></code> of <code>r</code> corresponds to a typical residual factor of <code>10<sup>r</sup></code> on the original mD scale.</li>
  </ul>
</div>
""".strip()

    grid_block = ""
    if grid_present:
        grid_block = f"""
<h4 class="sec mt-4">Option A &mdash; All Subsets (one panel per subset, colored by well)</h4>
<p class="text-muted small mb-2">
  Aggregated outer-fold predictions across all available feature subsets. Use this
  view to compare the residual cloud across cost regimes; the contraction from low-cost
  to high-cost panels is the visual analogue of the cost-10 discontinuity.
</p>
<div class="fig">{_img_base64_tag(fig_dir, grid_stem)}</div>
""".strip()

    per_well_block = ""
    if per_well_entries:
        per_well_sections = []
        for sid, raw_label, cost, stem in per_well_entries:
            short = SUBSET_DISPLAY_LABELS.get(raw_label, raw_label)
            cost_str = f"cost {int(cost)}" if not np.isnan(cost) else ""
            heading = f"Subset {sid} &mdash; {short}" + (f" ({cost_str})" if cost_str else "")
            per_well_sections.append(
                f"""
<h5 class="mt-4 mb-2">{heading}</h5>
<div class="fig">{_img_base64_tag(fig_dir, stem)}</div>
""".strip()
            )
        per_well_block = (
            '<h4 class="sec mt-4">Option B &mdash; Per-Well Detail (one panel per held-out well)</h4>\n'
            '<p class="text-muted small mb-2">'
            'For each subset, predictions are decomposed by the held-out well. '
            'Use this view to identify per-well biases or dispersion that the aggregate plots may hide.'
            '</p>\n'
            + "\n".join(per_well_sections)
        )

    back_link_html = ""
    if main_report_filename:
        back_link_html = (
            f'<a class="btn btn-sm btn-outline-light" href="{main_report_filename}">'
            "&larr; Main report</a>"
        )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Phase 4 - Predicted vs Measured</title>
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
    <span class="navbar-brand mb-0 h1">Phase 4 - Predicted vs Measured</span>
    <div class="d-flex align-items-center gap-3">
      {back_link_html}
      <span class="navbar-text text-light">{timestamp}</span>
    </div>
  </div>
</nav>
<div class="container-fluid px-4">

{interp_block}

<h4 class="sec mt-4">Per-Subset Summary</h4>
{summary_block}

{grid_block}

{per_well_block}

</div>
<footer class="text-center text-muted py-3 mt-4 border-top">
  <small>Generated by analyze_phase4.py - {timestamp}</small>
</footer>
</body>
</html>"""

    subpage_path = output_dir / f"{scoped_stem('phase4_predictions_report', metric_scope)}.html"
    subpage_path.write_text(html, encoding="utf-8")
    return subpage_path


def generate_html_report(
    config: dict,
    summary_df: pd.DataFrame,
    selection_summary: pd.DataFrame,
    bias_df: pd.DataFrame,
    table_paths: dict[str, Path],
    figure_paths: dict[str, Path],
    output_dir: Path,
    display_families: list[str],
    metric_scope: str,
    predictions_subpage_path: Optional[Path] = None,
    baseline_comparison_df: Optional[pd.DataFrame] = None,
    per_zone_df: Optional[pd.DataFrame] = None,
    ablation_comparison_df: Optional[pd.DataFrame] = None,
    ablation_nested_summary_df: Optional[pd.DataFrame] = None,
) -> Path:
    """Generate a self-contained HTML report with embedded figures and tables."""
    fig_dir = output_dir / "figures"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    bias_view = bias_df[bias_df["metric"] == "RMSE_log"].sort_values(
        ["reference_source", "subset_id"]
    )
    bias_cols = [
        "subset_id",
        "subset_label",
        "reference_source",
        "nested_mean",
        "reference_mean",
        "delta_nested_minus_reference",
    ]

    baseline_cols = [
        "baseline_label",
        "subset_id",
        "subset_label",
        "nested_mean",
        "baseline_mean",
        "delta_nested_minus_baseline",
    ]
    if baseline_comparison_df is not None and not baseline_comparison_df.empty:
        baseline_view = baseline_comparison_df[
            baseline_comparison_df["metric"] == "RMSE_log"
        ].sort_values(["baseline_slug", "subset_id"])
    else:
        baseline_view = pd.DataFrame()
    baseline_section = ""
    if not baseline_view.empty:
        baseline_section = (
            '<h4 class="sec mt-4">Petrophysical / Regression Baselines (RMSE_log)</h4>'
            '<p class="text-muted">Negative <code>delta_nested_minus_baseline</code> '
            "means the nested ML pipeline beats the classical baseline at that "
            "operating point.</p>"
            f"{_html_table(baseline_view, baseline_cols)}"
        )

    per_zone_cols = [
        "subset_id",
        "subset_label",
        "Zone",
        "n",
        "RMSE_log",
        "R2_log",
        "MAE_log",
    ]
    per_zone_section = ""
    if per_zone_df is not None and not per_zone_df.empty:
        per_zone_section = (
            '<h4 class="sec mt-4">Per-Zone Error Breakdown</h4>'
            '<p class="text-muted">Held-out nested predictions attributed to '
            "geological zones via <code>row_index</code>, scored per zone.</p>"
            f"{_html_table(per_zone_df, per_zone_cols)}"
        )

    ablation_cols = [
        "subset_id",
        "subset_label",
        "config_slug",
        "dropped",
        "full_mean",
        "config_mean",
        "delta_vs_full",
    ]
    ablation_section = ""
    if ablation_comparison_df is not None and not ablation_comparison_df.empty:
        abl_view = ablation_comparison_df[
            ablation_comparison_df["metric"] == "RMSE_log"
        ].copy()
        crosslink = ""
        if per_zone_df is not None and not per_zone_df.empty:
            crosslink = (
                " The <code>no_zone</code> rows are the aggregate effect of "
                "dropping Zone; see the Per-Zone Error Breakdown for where it "
                "concentrates."
            )
        nested_block = ""
        if (
            ablation_nested_summary_df is not None
            and not ablation_nested_summary_df.empty
        ):
            nested_cols = [
                "subset_id",
                "config_slug",
                "dropped",
                "RMSE_log_mean",
                "RMSE_log_std",
                "R2_log_mean",
            ]
            nested_block = (
                "<h5 class=\"mt-3\">Ablated nested configurations (per subset x config)</h5>"
                f"{_html_table(ablation_nested_summary_df, nested_cols)}"
            )
        ablation_section = (
            '<h4 class="sec mt-4">Structural-Covariate Ablation (nested, RMSE_log delta vs full)</h4>'
            '<p class="text-muted">Positive <code>delta_vs_full</code> means '
            "dropping the covariate hurt generalization (nested LOWO; full "
            "reused from the canonical run). The CSV carries deltas for all "
            "metrics." + crosslink + "</p>"
            f"{_html_table(abl_view, ablation_cols)}"
            f"{nested_block}"
        )

    cfg_table = pd.DataFrame(
        [
            {"field": "Model", "value": config.get("best_model", "N/A")},
            {"field": "Variant", "value": config.get("best_variant", "N/A")},
            {"field": "HP budget", "value": config.get("hp_budget", "N/A")},
            {"field": "Cost mode", "value": config.get("cost_mode", "N/A")},
            {"field": "Metric scope", "value": metric_scope},
            {"field": "Candidate subsets", "value": len(config.get("candidate_subsets", []))},
        ]
    )

    predictions_link_btn = ""
    predictions_link_card = ""
    if predictions_subpage_path is not None:
        sub_filename = Path(predictions_subpage_path).name
        predictions_link_btn = (
            f'<a class="btn btn-sm btn-light" href="{sub_filename}">'
            'Predicted vs Measured &rarr;</a>'
        )
        predictions_link_card = f"""
<div class="card border-primary mb-4">
  <div class="card-body d-flex flex-wrap align-items-center justify-content-between gap-3">
    <div>
      <h5 class="card-title mb-1">Predicted vs Measured plots</h5>
      <p class="card-text text-muted mb-0 small">
        Aggregated outer-fold predictions for each candidate subset, plus per-well detail grids.
      </p>
    </div>
    <a class="btn btn-primary" href="{sub_filename}">Open sub-page &rarr;</a>
  </div>
</div>
""".strip()

    family_sections: list[str] = []
    for family in display_families:
        primary_metric = primary_metric_for_family(family)
        mae_metric = mae_metric_for_family(family)
        secondary_metric = secondary_metric_for_family(family)
        ci_col = f"{primary_metric}_ci"
        table_cols = [
            "subset_id",
            "subset_label",
            "cost",
            "n_outer_folds",
            f"{primary_metric}_mean",
            f"{primary_metric}_std",
            ci_col,
            f"{mae_metric}_mean",
            f"{secondary_metric}_mean",
        ]
        view_df = (
            summary_df.sort_values([ci_col, "cost"])
            if ci_col in summary_df.columns
            else summary_df
        )
        metric_token = primary_metric.lower()
        secondary_token = secondary_metric.lower()
        frontier_stem = scoped_stem(
            f"phase4_nested_frontier_{metric_token}", metric_scope
        )
        heatmap_stem = scoped_stem(
            f"phase4_outer_{metric_token}_heatmap", metric_scope
        )
        secondary_heatmap_stem = scoped_stem(
            f"phase4_outer_{secondary_token}_heatmap", metric_scope
        )
        boxplot_stem = scoped_stem(
            f"phase4_outer_{metric_token}_boxplot", metric_scope
        )
        section_title = "Log-space" if family == "log" else "Original-space"
        secondary_block = ""
        if (fig_dir / "png" / f"{secondary_heatmap_stem}.png").exists():
            secondary_block = (
                f'<div class="fig">{_img_base64_tag(fig_dir, secondary_heatmap_stem)}</div>'
            )
        family_sections.append(
            f"""
<h4 class="sec mt-4">Nested Summary ({section_title} view)</h4>
{_html_table(view_df, table_cols)}
<div class="fig">{_img_base64_tag(fig_dir, frontier_stem)}</div>
<div class="fig">{_img_base64_tag(fig_dir, heatmap_stem)}</div>
{secondary_block}
<div class="fig">{_img_base64_tag(fig_dir, boxplot_stem)}</div>
"""
        )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Phase 4 - Nested LOWO Report</title>
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
    <span class="navbar-brand mb-0 h1">Phase 4 - Nested LOWO Report</span>
    <div class="d-flex align-items-center gap-3">
      {predictions_link_btn}
      <span class="navbar-text text-light">{timestamp}</span>
    </div>
  </div>
</nav>
<div class="container-fluid px-4">

{predictions_link_card}

<h4 class="sec">Configuration</h4>
{_html_table(cfg_table)}

{"".join(family_sections)}

<h4 class="sec mt-4">Outer-fold Winner Frequency</h4>
{_html_table(selection_summary, ["selected_subset_id", "selected_subset_label", "selected_outer_folds"])}

<h4 class="sec mt-4">Bias Comparison (RMSE_log)</h4>
{_html_table(bias_view, bias_cols)}

{baseline_section}

{per_zone_section}

{ablation_section}

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
  <small>Generated by analyze_phase4.py - {timestamp}</small>
</footer>
</body>
</html>"""

    html_path = output_dir / f"{scoped_stem('phase4_report', metric_scope)}.html"
    html_path.write_text(html, encoding="utf-8")
    return html_path


def run_analysis(
    run_dir: str | Path | None = None,
    config_path: str | None = None,
    generate_figs: bool = True,
    generate_html: bool = False,
    metric_scope: str = DEFAULT_METRIC_SCOPE,
    pred_subset_ids: Optional[list[int]] = None,
    detail_subset_id: Optional[int] = None,
    baseline_dir: str | Path | None = None,
    ablation_dir: str | Path | None = None,
    verbose: bool = True,
) -> dict:
    project_root = Path(__file__).parent.parent
    config = load_phase4_config(config_path)
    display_families = resolve_metric_scope(metric_scope)
    sort_metric = primary_metric_for_family(display_families[0])

    if run_dir is None:
        run_dir = project_root / "results" / "phase4_generalization" / "run"
    else:
        run_dir = Path(run_dir)

    if baseline_dir is None:
        baseline_dir = run_dir.parent / "baselines"
    else:
        baseline_dir = Path(baseline_dir)

    if ablation_dir is None:
        ablation_dir = run_dir.parent / "ablation_nested"
    else:
        ablation_dir = Path(ablation_dir)

    out_dir = run_dir.parent / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    tables_dir = out_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = out_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    nested_df, trace_df = load_run_results(run_dir)
    summary_df = aggregate_nested_summary(nested_df, sort_metric=sort_metric)
    selection_summary = build_selection_summary(trace_df)
    phase3_ref = load_phase3_reference(config, project_root)
    phase2_ref = load_phase2_reference(config, project_root)
    bias_df = build_bias_comparison(summary_df, phase3_ref, phase2_ref)

    baseline_outputs = load_baseline_outputs(baseline_dir)
    baseline_summary_df = baseline_outputs.get("baseline_summary", pd.DataFrame())
    baseline_outer_df = baseline_outputs.get("baseline_outer", pd.DataFrame())
    baseline_comparison_df = build_baseline_comparison(summary_df, baseline_summary_df)

    ablation_outputs = load_ablation_outputs(ablation_dir)
    ablation_summary_df = ablation_outputs.get("ablation_summary", pd.DataFrame())
    ablation_results_df = ablation_outputs.get("ablation_results", pd.DataFrame())
    ablation_nested_summary_df = ablation_summary_df
    ablation_comparison_df = build_ablation_comparison(ablation_summary_df, summary_df)

    nested_summary_csv = run_dir / f"{scoped_stem('nested_summary', metric_scope)}.csv"
    nested_summary_json = run_dir / f"{scoped_stem('nested_summary', metric_scope)}.json"
    bias_csv = run_dir / f"{scoped_stem('bias_comparison', metric_scope)}.csv"
    summary_df.to_csv(nested_summary_csv, index=False)
    with open(nested_summary_json, "w") as f:
        json.dump(
            summary_df.to_dict(orient="records"),
            f,
            indent=2,
            default=lambda o: float(o) if isinstance(o, np.floating) else str(o),
        )
    bias_df.to_csv(bias_csv, index=False)
    selection_summary_csv = run_dir / f"{scoped_stem('selection_summary', metric_scope)}.csv"
    if not selection_summary.empty:
        selection_summary.to_csv(selection_summary_csv, index=False)

    table_paths = {
        "T1_nested_summary": nested_summary_csv,
        "T2_bias_comparison": bias_csv,
    }
    if not selection_summary.empty:
        table_paths["T3_selection_summary"] = selection_summary_csv

    if not baseline_comparison_df.empty:
        baseline_comparison_csv = (
            run_dir / f"{scoped_stem('baseline_comparison', metric_scope)}.csv"
        )
        baseline_comparison_df.to_csv(baseline_comparison_csv, index=False)
        table_paths["T4_baseline_comparison"] = baseline_comparison_csv
    elif verbose:
        print(
            f"[analyze_phase4] No baseline comparison produced "
            f"(baseline_dir={baseline_dir}); skipping baseline table."
        )

    # Per-zone error breakdown from held-out nested predictions (row_index -> Zone).
    zone_lookup = load_zone_lookup(config, project_root)
    per_zone_subset_ids = discover_prediction_subset_ids(run_dir)
    if pred_subset_ids is not None:
        keep = set(pred_subset_ids)
        per_zone_subset_ids = [s for s in per_zone_subset_ids if s in keep]
    per_zone_df = build_per_zone_metrics(
        run_dir, per_zone_subset_ids, zone_lookup, summary_df=summary_df
    )
    if not per_zone_df.empty:
        per_zone_all_csv = tables_dir / "per_zone_metrics_all.csv"
        per_zone_df.to_csv(per_zone_all_csv, index=False)
        table_paths["T5_per_zone_metrics"] = per_zone_all_csv
        for subset_id, grp in per_zone_df.groupby("subset_id"):
            sub_csv = tables_dir / f"per_zone_metrics_{int(subset_id)}.csv"
            grp.to_csv(sub_csv, index=False)
            table_paths[f"T5_per_zone_metrics_{int(subset_id)}"] = sub_csv
    elif verbose:
        if zone_lookup is None:
            print(
                "[analyze_phase4] Zone lookup unavailable "
                "(variant/Zone column missing); skipping per-zone breakdown."
            )
        else:
            print(
                "[analyze_phase4] No nested predictions found; "
                "skipping per-zone breakdown."
            )

    # Structural-covariate ablation: nested delta-vs-(canonical full) per subset.
    if not ablation_comparison_df.empty:
        ablation_comparison_csv = (
            run_dir / f"{scoped_stem('ablation_comparison', metric_scope)}.csv"
        )
        ablation_comparison_df.to_csv(ablation_comparison_csv, index=False)
        table_paths["T6_ablation_comparison"] = ablation_comparison_csv
        # Also surface the raw per subset x config nested summary for convenience.
        if not ablation_summary_df.empty:
            ablation_summary_csv = tables_dir / "ablation_nested_summary.csv"
            ablation_summary_df.to_csv(ablation_summary_csv, index=False)
            table_paths["T6_ablation_nested_summary"] = ablation_summary_csv
        if not ablation_results_df.empty:
            ablation_grid_csv = tables_dir / "ablation_nested_results_all.csv"
            ablation_results_df.to_csv(ablation_grid_csv, index=False)
            table_paths["T6_ablation_results"] = ablation_grid_csv
    elif verbose:
        print(
            f"[analyze_phase4] No ablation comparison produced "
            f"(ablation_dir={ablation_dir}); skipping ablation table."
        )

    # Canonical subset row order (by RMSE_log mean, best first) shared by every
    # heatmap so the per-well and per-zone heatmaps list subsets identically
    # regardless of the metric being displayed.
    if not summary_df.empty and "RMSE_log_mean" in summary_df.columns:
        canonical_row_order = (
            summary_df.sort_values("RMSE_log_mean")["subset_id"].astype(int).tolist()
        )
    else:
        canonical_row_order = None

    figure_paths: dict[str, Path] = {}
    if generate_figs:
        _set_style()
        for idx, family in enumerate(display_families, start=1):
            primary_metric = primary_metric_for_family(family)
            metric_token = primary_metric.lower()
            frontier_stem = scoped_stem(
                f"phase4_nested_frontier_{metric_token}", metric_scope
            )
            heatmap_stem = scoped_stem(
                f"phase4_outer_{metric_token}_heatmap", metric_scope
            )
            boxplot_stem = scoped_stem(
                f"phase4_outer_{metric_token}_boxplot", metric_scope
            )

            fig_nested_frontier(
                summary_df,
                figures_dir,
                metric=primary_metric,
                stem=frontier_stem,
            )
            figure_paths[f"F{idx}_nested_frontier_{primary_metric}"] = (
                figures_dir / "pdf" / f"{frontier_stem}.pdf"
            )
            fig_per_well_heatmap(
                nested_df,
                figures_dir,
                metric=primary_metric,
                stem=heatmap_stem,
                summary_df=summary_df,
                row_order=canonical_row_order,
            )
            figure_paths[f"F{idx}_per_well_heatmap_{primary_metric}"] = (
                figures_dir / "pdf" / f"{heatmap_stem}.pdf"
            )

            # Companion heatmap on the secondary metric (e.g., R2_log) so the user
            # sees both error magnitude and explained-variance per well at a glance.
            secondary_metric = secondary_metric_for_family(family)
            secondary_outer_col = f"outer_{secondary_metric}"
            if secondary_outer_col in nested_df.columns:
                secondary_token = secondary_metric.lower()
                secondary_heatmap_stem = scoped_stem(
                    f"phase4_outer_{secondary_token}_heatmap", metric_scope
                )
                fig_per_well_heatmap(
                    nested_df,
                    figures_dir,
                    metric=secondary_metric,
                    stem=secondary_heatmap_stem,
                    summary_df=summary_df,
                    row_order=canonical_row_order,
                )
                figure_paths[f"F{idx}_per_well_heatmap_{secondary_metric}"] = (
                    figures_dir / "pdf" / f"{secondary_heatmap_stem}.pdf"
                )
            fig_per_well_boxplot(
                nested_df,
                summary_df,
                figures_dir,
                metric=primary_metric,
                stem=boxplot_stem,
                baseline_outer_df=baseline_outer_df,
            )
            figure_paths[f"F{idx}_per_well_boxplot_{primary_metric}"] = (
                figures_dir / "pdf" / f"{boxplot_stem}.pdf"
            )

        # Petrophysical-baseline per-well boxplot (distinct stem so it never
        # collides with the nested per-well boxplots above).
        if not baseline_outer_df.empty:
            baseline_box_stem = "phase4_baseline_rmse_log_boxplot"
            baseline_box_path = fig_baseline_boxplot(
                baseline_outer_df,
                figures_dir,
                metric="RMSE_log",
                stem=baseline_box_stem,
            )
            if baseline_box_path is not None:
                figure_paths["B1_baseline_rmse_log_boxplot"] = (
                    figures_dir / "pdf" / f"{baseline_box_stem}.pdf"
                )

        # Per-zone error breakdown heatmap (subsets x zones). Generate both the
        # RMSE_log and R2_log variants so either is available downstream.
        if not per_zone_df.empty:
            for zi, zone_metric in enumerate(("RMSE_log", "R2_log"), start=1):
                per_zone_stem = f"phase4_per_zone_{zone_metric.lower()}"
                per_zone_path = fig_per_zone_metric(
                    per_zone_df,
                    figures_dir,
                    metric=zone_metric,
                    stem=per_zone_stem,
                    row_order=canonical_row_order,
                )
                if per_zone_path is not None:
                    figure_paths[f"Z{zi}_per_zone_{zone_metric.lower()}"] = (
                        figures_dir / "pdf" / f"{per_zone_stem}.pdf"
                    )

        # Structural-covariate ablation delta bar chart (RMSE_log vs full).
        if not ablation_comparison_df.empty:
            ablation_stem = "phase4_ablation_delta_rmse_log"
            ablation_path = fig_ablation_delta(
                ablation_comparison_df,
                figures_dir,
                metric="RMSE_log",
                stem=ablation_stem,
            )
            if ablation_path is not None:
                figure_paths["A2_ablation_delta_rmse_log"] = (
                    figures_dir / "pdf" / f"{ablation_stem}.pdf"
                )

        # Predicted vs measured plots (independent of metric scope; always log space).
        # Option A: one figure with all selected subsets as panels.
        grid_stem = "phase4_pred_vs_actual_grid"
        grid_path = fig_pred_vs_actual_grid(
            run_dir=run_dir,
            summary_df=summary_df,
            fig_dir=figures_dir,
            subset_ids=pred_subset_ids,
            stem=grid_stem,
        )
        if grid_path is not None:
            figure_paths["G1_pred_vs_actual_grid"] = (
                figures_dir / "pdf" / f"{grid_stem}.pdf"
            )

        # Option B: one per-well figure per subset by default. Narrow with --detail-subset
        # (single id) or --pred-subsets (intersection with discovered prediction folders).
        available_pred_ids = discover_prediction_subset_ids(run_dir)
        if pred_subset_ids is not None:
            available_pred_ids = [
                s for s in available_pred_ids if s in set(pred_subset_ids)
            ]
        if detail_subset_id is not None:
            per_well_targets = (
                [detail_subset_id] if detail_subset_id in available_pred_ids else []
            )
        else:
            per_well_targets = available_pred_ids

        for sid in per_well_targets:
            per_well_stem = f"phase4_pred_vs_actual_per_well_{sid}"
            per_well_path = fig_pred_vs_actual_per_well(
                run_dir=run_dir,
                summary_df=summary_df,
                fig_dir=figures_dir,
                detail_subset_id=sid,
                stem=per_well_stem,
            )
            if per_well_path is not None:
                figure_paths[f"G2_pred_vs_actual_per_well_{sid}"] = (
                    figures_dir / "pdf" / f"{per_well_stem}.pdf"
                )

    report = generate_report(
        config=config,
        summary_df=summary_df,
        selection_summary=selection_summary,
        bias_df=bias_df,
        table_paths=table_paths,
        figure_paths=figure_paths,
        display_families=display_families,
        metric_scope=metric_scope,
        baseline_comparison_df=baseline_comparison_df,
        per_zone_df=per_zone_df,
        ablation_comparison_df=ablation_comparison_df,
        ablation_nested_summary_df=ablation_nested_summary_df,
    )
    report_path = out_dir / f"{scoped_stem('phase4_report', metric_scope)}.md"
    report_path.write_text(report, encoding="utf-8")

    html_path = None
    predictions_subpage_path = None
    if generate_html:
        # Build the predictions sub-page first so the main report can link to it.
        main_report_filename = f"{scoped_stem('phase4_report', metric_scope)}.html"
        predictions_subpage_path = generate_html_predictions_subpage(
            summary_df=summary_df,
            output_dir=out_dir,
            metric_scope=metric_scope,
            pred_subset_ids=pred_subset_ids,
            main_report_filename=main_report_filename,
        )
        html_path = generate_html_report(
            config=config,
            summary_df=summary_df,
            selection_summary=selection_summary,
            bias_df=bias_df,
            table_paths=table_paths,
            figure_paths=figure_paths,
            output_dir=out_dir,
            display_families=display_families,
            metric_scope=metric_scope,
            predictions_subpage_path=predictions_subpage_path,
            baseline_comparison_df=baseline_comparison_df,
            per_zone_df=per_zone_df,
            ablation_comparison_df=ablation_comparison_df,
            ablation_nested_summary_df=ablation_nested_summary_df,
        )

    if verbose:
        print("Phase 4 analysis complete.")
        print(f"  Metric scope:   {metric_scope}")
        print(f"  Nested summary: {nested_summary_csv}")
        print(f"  Bias table:     {bias_csv}")
        if "T4_baseline_comparison" in table_paths:
            print(f"  Baseline table: {table_paths['T4_baseline_comparison']}")
        if "T5_per_zone_metrics" in table_paths:
            print(f"  Per-zone table: {table_paths['T5_per_zone_metrics']}")
        if "T6_ablation_comparison" in table_paths:
            print(f"  Ablation table: {table_paths['T6_ablation_comparison']}")
        print(f"  Report:         {report_path}")
        if html_path is not None:
            print(f"  HTML:           {html_path}")
        if predictions_subpage_path is not None:
            print(f"  HTML sub-page:  {predictions_subpage_path}")

    return {
        "nested_summary_csv": nested_summary_csv,
        "nested_summary_json": nested_summary_json,
        "bias_csv": bias_csv,
        "baseline_comparison_csv": table_paths.get("T4_baseline_comparison"),
        "per_zone_metrics_csv": table_paths.get("T5_per_zone_metrics"),
        "ablation_comparison_csv": table_paths.get("T6_ablation_comparison"),
        "report_path": report_path,
        "html_path": html_path,
        "predictions_subpage_path": predictions_subpage_path,
        "table_paths": table_paths,
        "figure_paths": figure_paths,
    }


def main():
    parser = argparse.ArgumentParser(description="Analyze Phase 4 nested LOWO outputs.")
    parser.add_argument(
        "--run_dir",
        "-r",
        type=str,
        default=None,
        help="Directory containing nested run outputs (default: results/phase4_generalization/run).",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to phase4 config (default: configs/phase4_config_bundled.json).",
    )
    parser.add_argument(
        "--baseline_dir",
        type=str,
        default=None,
        help=(
            "Directory with petrophysical/regression baseline outputs from "
            "run_phase4_baselines.py (default: <run_dir>/../baselines)."
        ),
    )
    parser.add_argument(
        "--ablation_dir",
        type=str,
        default=None,
        help=(
            "Directory with structural-covariate ablation outputs from "
            "run_phase4_ablation.py (default: <run_dir>/../ablation_nested)."
        ),
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
        help="Metric family for reporting views: original, log, or both (default: log).",
    )
    parser.add_argument(
        "--pred-subsets",
        type=str,
        default=None,
        help=(
            "Comma-separated subset IDs to include in the predicted-vs-actual figures "
            "(applies to both Option-A panels and Option-B per-well grids; "
            "default: auto-detect all subsets with predictions/ folders)."
        ),
    )
    parser.add_argument(
        "--detail-subset",
        type=int,
        default=None,
        help=(
            "Restrict the Option-B per-well grid to a single subset ID. "
            "By default, a per-well figure is generated for every available subset."
        ),
    )
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Suppress verbose output.",
    )
    args = parser.parse_args()

    pred_subset_ids: Optional[list[int]] = None
    if args.pred_subsets:
        raw = args.pred_subsets.strip().lower()
        if raw not in ("", "all"):
            pred_subset_ids = [
                int(v.strip()) for v in args.pred_subsets.split(",") if v.strip()
            ]

    run_analysis(
        run_dir=args.run_dir,
        config_path=args.config,
        generate_figs=not args.no_figures,
        generate_html=args.html,
        metric_scope=args.metric_scope,
        pred_subset_ids=pred_subset_ids,
        detail_subset_id=args.detail_subset,
        baseline_dir=args.baseline_dir,
        ablation_dir=args.ablation_dir,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
