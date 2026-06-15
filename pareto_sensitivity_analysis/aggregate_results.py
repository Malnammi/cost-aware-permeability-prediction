#!/usr/bin/env python
"""
Aggregate the per-cell Monte Carlo outputs into the master table and figure.

The four Monte Carlo cells (``{primary, secondary} x {rank_preserving, unconstrained}``)
are run as independent ``montecarlo_sensitivity.py`` jobs, each writing its own files.
This join step runs AFTER they finish: it concatenates the per-cell summaries into the
master ``montecarlo_results.csv`` keyed by ``(parameter_set, scenario, metric)``, writes a
wide pivot for the manuscript table, and renders ``fig:app-cost-sensitivity`` - the
baseline frontier with each recommended operating point's perturbed cost envelope marked.

To run:

    python pareto_sensitivity_analysis/aggregate_results.py
    python pareto_sensitivity_analysis/aggregate_results.py --no-figure
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import sensitivity_utils as su  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--out-dir", default=None,
        help="MC results directory holding the per-cell files (default: "
             "results/phase3_feature_selection/pareto_sensitivity/montecarlo_sensitivity/).",
    )
    p.add_argument("--no-figure", action="store_true", help="Skip the envelope figure.")
    return p.parse_args()


def _cell_stem(parameter_set: str, scenario: str) -> str:
    return f"montecarlo_{parameter_set}_{scenario}"


def collect_summaries(out_dir: Path) -> pd.DataFrame:
    """Concatenate every present per-cell summary into one long table."""
    frames = []
    for parameter_set in su.PARAMETER_SETS:
        for scenario in su.MC_SCENARIOS:
            path = out_dir / f"{_cell_stem(parameter_set, scenario)}.csv"
            if path.exists():
                frames.append(pd.read_csv(path))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def collect_draws(out_dir: Path) -> pd.DataFrame:
    """Concatenate every present per-cell draws file (for the envelope figure)."""
    frames = []
    for parameter_set in su.PARAMETER_SETS:
        for scenario in su.MC_SCENARIOS:
            path = out_dir / f"{_cell_stem(parameter_set, scenario)}_draws.csv"
            if path.exists():
                frames.append(pd.read_csv(path))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def plot_cost_envelope(draws: pd.DataFrame, out_dir: Path) -> None:
    """Baseline frontier with each recommended point's perturbed-cost envelope (p5-p95)."""
    sweep = su.load_sweep()
    base_cost, base_bundles, _ = su.load_baseline_cost_config()
    scored, frontier = su.recompute_frontier(sweep, base_cost, base_bundles)
    fr = frontier.sort_values("cost")

    fig, ax = plt.subplots(figsize=(10, 7))

    ax.scatter(
        scored["cost"], scored[su.PERF_COL],
        s=8, alpha=0.15, color="gray", rasterized=True,
        label=f"All subsets (n={len(scored)})",
    )
    ax.plot(
        fr["cost"], fr[su.PERF_COL], "o-",
        color="crimson", linewidth=2, markersize=6,
        markeredgecolor="black", markeredgewidth=0.5,
        label="Baseline Pareto frontier", zorder=4,
    )

    palette = ["#1f77b4", "#2ca02c", "#9467bd"]
    for color, sid in zip(palette, su.RECOMMENDED_IDS):
        cost_col = f"cost_{sid}"
        srow = scored[scored["subset_id"] == sid]
        if srow.empty:
            continue
        y = float(srow[su.PERF_COL].iloc[0])
        base_x = float(srow["cost"].iloc[0])

        if cost_col in draws.columns and draws[cost_col].notna().any():
            costs = draws[cost_col].to_numpy(dtype=float)
            lo, hi = np.percentile(costs, [5, 95])
            ax.hlines(y, lo, hi, color=color, linewidth=3, alpha=0.5, zorder=5)
            ax.plot([lo, hi], [y, y], "|", color=color, markersize=10, zorder=5)
        ax.scatter(
            [base_x], [y], s=160, marker="*", color=color,
            edgecolors="black", linewidth=0.8, zorder=6,
            label=f"{su.RECOMMENDED_LABELS.get(sid, sid)}",
        )

    ax.set_xlabel("Acquisition cost (surrogate score)")
    ax.set_ylabel(r"$\mathrm{RMSE}_{\log}$ mean (lower is better)")
    ax.set_title(
        "Recommended operating points under cost perturbation\n"
        "(stars = baseline cost; bars = 5th-95th pct cost across MC draws)"
    )
    ax.legend(loc="best", fontsize=8)

    su.save_fig(fig, su.get_fig_dir(out_dir), "app_cost_sensitivity")


def main() -> int:
    args = parse_args()
    out_dir = su.get_results_dir(args.out_dir or (su.RESULTS_DIR / "montecarlo_sensitivity"))

    summary = collect_summaries(out_dir)
    if summary.empty:
        print(f"No per-cell summaries found in {out_dir}. Run montecarlo_sensitivity.py first.")
        return 0

    su.write_csv(summary, out_dir, "montecarlo_results.csv")

    wide = summary.pivot_table(
        index="metric", columns=["parameter_set", "scenario"], values="value",
    )
    wide.columns = [f"{ps}_{sc}" for ps, sc in wide.columns]
    wide = wide.reset_index()
    su.write_csv(wide, out_dir, "montecarlo_summary_wide.csv")

    cells = sorted(
        summary[["parameter_set", "scenario"]]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    )
    print(f"Aggregated {len(cells)} MC cell(s): {cells}")
    print(f"  wrote montecarlo_results.csv ({len(summary)} rows) and montecarlo_summary_wide.csv")

    if not args.no_figure:
        draws = collect_draws(out_dir)
        if draws.empty:
            print("  no per-cell draws files found; skipping envelope figure.")
        else:
            su.set_style()
            plot_cost_envelope(draws, out_dir)
            print("  wrote figure app_cost_sensitivity.{pdf,svg,png}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
