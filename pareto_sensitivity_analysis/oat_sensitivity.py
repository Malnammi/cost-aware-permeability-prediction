#!/usr/bin/env python
"""
One-factor-at-a-time (OFAT/OAT, local) cost sensitivity.

Sweeps each of the seven primary cost parameters across the perturbation levels
``{-0.3, -0.2, -0.1, 0, +0.1, +0.2, +0.3}`` while holding all other costs at baseline,
recomputes the Pareto frontier at every step, and
records whether each recommended operating point (640, 4096, 4550) remains
non-dominated and whether the two-regime split persists.

It additionally reports the CPOR_SM breakpoint: the standalone cost level at/below
which the two-regime split ceases to hold, obtained from a fine scan of the CPOR_SM
standalone cost. 

To run:

    python pareto_sensitivity_analysis/oat_sensitivity.py
    python pareto_sensitivity_analysis/oat_sensitivity.py --no-figure --breakpoint-step 0.05
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
        help="Results directory (default: "
             "results/phase3_feature_selection/pareto_sensitivity/oat_sensitivity/).",
    )
    p.add_argument(
        "--breakpoint-step", type=float, default=0.1,
        help="Cost-grid step for the CPOR_SM breakpoint fine scan (default 0.1).",
    )
    p.add_argument(
        "--no-figure", action="store_true",
        help="Skip rendering the tornado figure.",
    )
    return p.parse_args()


def _param_label(param) -> str:
    feat, kind = param
    return f"{feat}_{kind}"


def _baseline_value(param, cost_map, bundles) -> float:
    feat, kind = param
    if kind == "standalone":
        return float(cost_map[feat])
    return float(bundles[feat]["marginal_cost"])


def _perturbed_value(param, cost_map, bundles) -> float:
    return _baseline_value(param, cost_map, bundles)


def run_oat(sweep, base_cost, base_bundles) -> pd.DataFrame:
    """Sweep each primary parameter over the OAT levels; one row per (parameter, level)."""
    _, baseline_frontier = su.recompute_frontier(sweep, base_cost, base_bundles)
    baseline_ids = su.frontier_ids(baseline_frontier)

    rows = []
    for param in su.OAT_PARAMETERS:
        for delta in su.OAT_LEVELS:
            multipliers = {param: 1.0 + float(delta)}
            cost_map, bundles = su.perturb_costs(base_cost, base_bundles, multipliers)
            scored, frontier = su.recompute_frontier(sweep, cost_map, bundles)

            fr_ids = su.frontier_ids(frontier)
            split = su.two_regime_split(frontier)

            row = {
                "parameter": _param_label(param),
                "feature": param[0],
                "kind": param[1],
                "delta": float(delta),
                "perturbed_value": _perturbed_value(param, cost_map, bundles),
                "frontier_size": len(fr_ids),
                "jaccard_vs_baseline": su.jaccard(fr_ids, baseline_ids),
                "split_holds": bool(split["holds"]),
                "boundary_cost": split["boundary_cost"],
                "wireline_max_cost": split["wireline_max_cost"],
                "n_wireline": split["n_wireline"],
                "n_cpor": split["n_cpor"],
                "cpor_entry_cost": su.cpor_entry_cost(frontier),
            }
            for sid in su.RECOMMENDED_IDS:
                row[f"on_frontier_{sid}"] = sid in fr_ids
            rows.append(row)

    return pd.DataFrame(rows)


def compute_cpor_breakpoint(sweep, base_cost, base_bundles, step: float):
    """
    Fine scan of the CPOR_SM standalone cost to locate the two-regime breakpoint.

    Returns ``(breakpoint_cost, scan_df)`` where ``breakpoint_cost`` is the highest
    CPOR_SM standalone cost at which the split fails (NaN if it never fails in range);
    the frontier is recomputed at each grid point so removed/added wireline points are
    handled correctly.
    """
    base_val = float(base_cost["CPOR_SM"])
    grid = np.round(np.arange(1.0, base_val + 1e-9, step), 6)

    records = []
    for c in grid:
        cm = dict(base_cost)
        cm["CPOR_SM"] = float(c)
        _, frontier = su.recompute_frontier(sweep, cm, base_bundles)
        split = su.two_regime_split(frontier)
        records.append({
            "cpor_standalone_cost": float(c),
            "split_holds": bool(split["holds"]),
            "boundary_cost": split["boundary_cost"],
            "wireline_max_cost": split["wireline_max_cost"],
        })

    scan_df = pd.DataFrame(records)
    failing = scan_df.loc[~scan_df["split_holds"], "cpor_standalone_cost"]
    breakpoint_cost = float(failing.max()) if not failing.empty else float("nan")
    return breakpoint_cost, scan_df


def plot_tornado(oat_df: pd.DataFrame, fig_dir: Path) -> None:
    """Tornado ranking parameters by their maximum frontier disturbance over the sweep."""
    rec_cols = [f"on_frontier_{sid}" for sid in su.RECOMMENDED_IDS]

    summary = []
    for label, g in oat_df.groupby("parameter"):
        max_dissim = float(1.0 - g["jaccard_vs_baseline"].min())
        stable = bool(g[rec_cols].all(axis=None) and g["split_holds"].all())
        summary.append({"parameter": label, "max_dissim": max_dissim, "stable": stable})

    sdf = pd.DataFrame(summary).sort_values("max_dissim", ascending=True)

    fig, ax = plt.subplots(figsize=(9, 6))
    colors = ["gray" if s else "crimson" for s in sdf["stable"]]
    ax.barh(range(len(sdf)), sdf["max_dissim"].values,
            color=colors, edgecolor="black", linewidth=0.4)
    ax.set_yticks(range(len(sdf)))
    ax.set_yticklabels(sdf["parameter"].values, fontsize=9)
    ax.set_xlabel(r"Max frontier dissimilarity $1 - J$ over $\delta \in [-0.3, +0.3]$")
    ax.set_title("OAT cost sensitivity (gray = all recommendations stable)")

    handles = [
        plt.Rectangle((0, 0), 1, 1, color="gray", ec="black"),
        plt.Rectangle((0, 0), 1, 1, color="crimson", ec="black"),
    ]
    ax.legend(handles, ["recommendations stable", "a recommendation flips / split breaks"],
              loc="lower right", fontsize=8)

    su.save_fig(fig, fig_dir, "oat_tornado")


def main() -> int:
    args = parse_args()
    out_dir = su.get_results_dir(args.out_dir or (su.RESULTS_DIR / "oat_sensitivity"))

    sweep = su.load_sweep()
    base_cost, base_bundles, _ = su.load_baseline_cost_config()

    oat_df = run_oat(sweep, base_cost, base_bundles)
    su.write_csv(oat_df, out_dir, "oat_results.csv")

    breakpoint_cost, scan_df = compute_cpor_breakpoint(
        sweep, base_cost, base_bundles, args.breakpoint_step
    )
    su.write_csv(scan_df, out_dir, "cpor_breakpoint_scan.csv")

    rec_cols = [f"on_frontier_{sid}" for sid in su.RECOMMENDED_IDS]
    summary = {
        "n_configurations": int(len(oat_df)),
        "baseline_cpor_cost": float(base_cost["CPOR_SM"]),
        "cpor_breakpoint_cost": breakpoint_cost,
        "cpor_breakpoint_fraction_of_baseline": (
            breakpoint_cost / float(base_cost["CPOR_SM"])
            if breakpoint_cost == breakpoint_cost else float("nan")  # NaN-safe
        ),
        "recommendations_always_on_frontier": {
            str(sid): bool(oat_df[f"on_frontier_{sid}"].all())
            for sid in su.RECOMMENDED_IDS
        },
        "split_always_holds": bool(oat_df["split_holds"].all()),
        "parameters_that_ever_disturb_a_recommendation": sorted(
            oat_df.loc[
                ~(oat_df[rec_cols].all(axis=1) & oat_df["split_holds"]),
                "parameter",
            ].unique().tolist()
        ),
    }
    su.write_json(summary, out_dir, "oat_summary.json")

    if not args.no_figure:
        su.set_style()
        plot_tornado(oat_df, su.get_fig_dir(out_dir))

    _print_summary(summary)
    return 0


def _print_summary(summary: dict) -> None:
    print("OAT cost sensitivity complete.")
    print(f"  configurations: {summary['n_configurations']}")
    print(f"  CPOR_SM breakpoint cost: {summary['cpor_breakpoint_cost']} "
          f"(baseline {summary['baseline_cpor_cost']})")
    print(f"  split always holds: {summary['split_always_holds']}")
    for sid, ok in summary["recommendations_always_on_frontier"].items():
        print(f"  subset {sid} always on frontier: {ok}")
    disturbers = summary["parameters_that_ever_disturb_a_recommendation"]
    if disturbers:
        print(f"  parameters that ever disturb a recommendation/split: {disturbers}")


if __name__ == "__main__":
    sys.exit(main())
