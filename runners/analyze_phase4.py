#!/usr/bin/env python
"""
Phase 4 Nested LOWO - Analysis and Reporting.

Consumes run artifacts from results/phase4_generalization/run and produces:
- nested_summary.csv / nested_summary.json
- bias_comparison.csv
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

MINIMIZE_METRICS = {"RMSE", "MAE", "MedAE", "RMSLE", "RMSE_log", "MAE_log"}
MAXIMIZE_METRICS = {"R2", "R2_log"}
METRIC_SCOPE_CHOICES = ("original", "log", "both")
DEFAULT_METRIC_SCOPE = "log"


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
    ax.set_title(f"Phase 4 Nested LOWO: Cost vs {metric}")
    _save_fig(fig, fig_dir, fig_stem)


def fig_per_well_boxplot(
    nested_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    fig_dir: Path,
    metric: str = "RMSE_log",
    stem: Optional[str] = None,
) -> None:
    """Boxplot + strip plot of outer-fold metric by subset."""
    mean_col = f"{metric}_mean"
    outer_col = f"outer_{metric}"
    if mean_col not in summary_df.columns or outer_col not in nested_df.columns:
        return
    fig_stem = stem or f"phase4_outer_{metric.lower()}_boxplot"
    subset_order = summary_df.sort_values(mean_col)["subset_label"].tolist()

    LABEL_MAP = {
        "best_performer": "4550\nBest-performer",
        "cpor_sm_only_baseline": "4096\nCPOR_SM-only",
        "budget_wireline_only": "640\nBudget wireline",
    }
    plot_df = nested_df.copy()
    plot_df["display_label"] = plot_df["subset_label"].map(LABEL_MAP).fillna(plot_df["subset_label"])
    display_order = [LABEL_MAP.get(s, s) for s in subset_order]

    n_subsets = len(display_order)
    box_colors = sns.color_palette("colorblind", n_subsets)

    fig, ax = plt.subplots(figsize=(max(5, n_subsets * 2), 5))
    sns.boxplot(
        data=plot_df,
        x="display_label",
        y=outer_col,
        order=display_order,
        palette=box_colors,
        width=0.5,
        linewidth=0.8,
        fliersize=0,
        ax=ax,
    )
    sns.stripplot(
        data=plot_df,
        x="display_label",
        y=outer_col,
        order=display_order,
        color="black",
        dodge=False,
        size=6,
        alpha=0.7,
        ax=ax,
    )
    ax.set_xlabel("")
    ax.set_ylabel(f"{metric} (outer fold)")
    ax.set_title(f"Nested LOWO: Per-Well {metric} by Feature Configuration")
    ax.tick_params(axis="x", which="major", length=5, direction="out")
    _save_fig(fig, fig_dir, fig_stem)


def fig_per_well_heatmap(
    nested_df: pd.DataFrame,
    fig_dir: Path,
    metric: str = "RMSE_log",
    stem: Optional[str] = None,
) -> None:
    outer_col = f"outer_{metric}"
    if outer_col not in nested_df.columns:
        return
    fig_stem = stem or f"phase4_outer_{metric.lower()}_heatmap"
    pivot = nested_df.pivot_table(
        index="subset_label",
        columns="outer_fold",
        values=outer_col,
        aggfunc="mean",
    )
    if pivot.empty:
        return
    pivot = pivot.loc[sorted(pivot.index), sorted(pivot.columns)]
    fig, ax = plt.subplots(figsize=(max(8, pivot.shape[1] * 1.1), max(4, pivot.shape[0] * 0.9)))
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".4f",
        cmap="YlOrRd",
        linewidths=0.5,
        cbar_kws={"label": f"Outer {metric}"},
        ax=ax,
    )
    ax.set_title(f"Outer-fold {metric} by subset")
    ax.set_xlabel("Held-out well")
    ax.set_ylabel("Subset")
    _save_fig(fig, fig_dir, fig_stem)


def generate_report(
    config: dict,
    summary_df: pd.DataFrame,
    selection_summary: pd.DataFrame,
    bias_df: pd.DataFrame,
    table_paths: dict[str, Path],
    figure_paths: dict[str, Path],
    display_families: list[str],
    metric_scope: str,
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
        frontier_stem = scoped_stem(
            f"phase4_nested_frontier_{metric_token}", metric_scope
        )
        heatmap_stem = scoped_stem(
            f"phase4_outer_{metric_token}_heatmap", metric_scope
        )
        boxplot_stem = scoped_stem(
            f"phase4_outer_{metric_token}_boxplot", metric_scope
        )
        section_title = "Log-space" if family == "log" else "Original-space"
        family_sections.append(
            f"""
<h4 class="sec mt-4">Nested Summary ({section_title} view)</h4>
{_html_table(view_df, table_cols)}
<div class="fig">{_img_base64_tag(fig_dir, frontier_stem)}</div>
<div class="fig">{_img_base64_tag(fig_dir, heatmap_stem)}</div>
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
    <span class="navbar-text text-light">{timestamp}</span>
  </div>
</nav>
<div class="container-fluid px-4">

<h4 class="sec">Configuration</h4>
{_html_table(cfg_table)}

{"".join(family_sections)}

<h4 class="sec mt-4">Outer-fold Winner Frequency</h4>
{_html_table(selection_summary, ["selected_subset_id", "selected_subset_label", "selected_outer_folds"])}

<h4 class="sec mt-4">Bias Comparison (RMSE_log)</h4>
{_html_table(bias_view, bias_cols)}

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
            )
            figure_paths[f"F{idx}_per_well_heatmap_{primary_metric}"] = (
                figures_dir / "pdf" / f"{heatmap_stem}.pdf"
            )
            fig_per_well_boxplot(
                nested_df,
                summary_df,
                figures_dir,
                metric=primary_metric,
                stem=boxplot_stem,
            )
            figure_paths[f"F{idx}_per_well_boxplot_{primary_metric}"] = (
                figures_dir / "pdf" / f"{boxplot_stem}.pdf"
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
    )
    report_path = out_dir / f"{scoped_stem('phase4_report', metric_scope)}.md"
    report_path.write_text(report, encoding="utf-8")

    html_path = None
    if generate_html:
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
        )

    if verbose:
        print("Phase 4 analysis complete.")
        print(f"  Metric scope:   {metric_scope}")
        print(f"  Nested summary: {nested_summary_csv}")
        print(f"  Bias table:     {bias_csv}")
        print(f"  Report:         {report_path}")
        if html_path is not None:
            print(f"  HTML:           {html_path}")

    return {
        "nested_summary_csv": nested_summary_csv,
        "nested_summary_json": nested_summary_json,
        "bias_csv": bias_csv,
        "report_path": report_path,
        "html_path": html_path,
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
        "-q",
        "--quiet",
        action="store_true",
        help="Suppress verbose output.",
    )
    args = parser.parse_args()

    run_analysis(
        run_dir=args.run_dir,
        config_path=args.config,
        generate_figs=not args.no_figures,
        generate_html=args.html,
        metric_scope=args.metric_scope,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
