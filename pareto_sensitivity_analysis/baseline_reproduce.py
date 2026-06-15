#!/usr/bin/env python
"""
Baseline reproduction check for the Phase 3 cost-sensitivity analysis.

This is the engine sanity check: before any perturbation is trusted, confirm that
``recompute_frontier`` at the *baseline* cost vector reproduces the published Pareto
frontier exactly, and that the three recommended operating points (640, 4096, 4550)
lie on it with their documented costs.

To run:

    python pareto_sensitivity_analysis/baseline_reproduce.py
    python pareto_sensitivity_analysis/baseline_reproduce.py --no-figure
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import sensitivity_utils as su  # noqa: E402

PUBLISHED_FRONTIER = (
    su.ROOT / "results" / "phase3_feature_selection" / "pareto_sweep"
    / "pareto_frontier_RMSE_log.csv"
)

# Documented bundle-aware costs of the recommended operating points.
EXPECTED_COSTS = {640: 7, 4096: 10, 4550: 21}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--out-dir", default=None,
        help="Results directory (default: "
             "results/phase3_feature_selection/pareto_sensitivity/baseline_reproduce/).",
    )
    p.add_argument(
        "--published-frontier", default=str(PUBLISHED_FRONTIER),
        help="CSV of the published RMSE_log Pareto frontier to compare against.",
    )
    p.add_argument(
        "--no-figure", action="store_true",
        help="Skip rendering the baseline frontier figure.",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = su.get_results_dir(args.out_dir or (su.RESULTS_DIR / "baseline_reproduce"))

    sweep = su.load_sweep()
    cost_map, bundles, _ = su.load_baseline_cost_config()

    # Identity perturbation: recompute cost + frontier at the baseline cost vector.
    scored, frontier = su.recompute_frontier(sweep, cost_map, bundles)

    recomputed_ids = su.frontier_ids(frontier)
    published = pd.read_csv(args.published_frontier)
    published_ids = set(int(s) for s in published["subset_id"])

    checks = []

    # 1) Exact frontier-membership reproduction.
    missing = sorted(published_ids - recomputed_ids)   # published but not recomputed
    extra = sorted(recomputed_ids - published_ids)      # recomputed but not published
    frontier_match = (not missing) and (not extra)
    checks.append(("frontier_membership_exact", frontier_match, {
        "n_published": len(published_ids),
        "n_recomputed": len(recomputed_ids),
        "missing_from_recomputed": missing,
        "extra_in_recomputed": extra,
    }))

    # 2) Recommended operating points on the frontier.
    for sid in su.RECOMMENDED_IDS:
        on = su.is_on_frontier(frontier, sid)
        checks.append((f"recommended_{sid}_on_frontier", on, {
            "label": su.RECOMMENDED_LABELS.get(sid, ""),
        }))

    # 3) Recommended operating-point costs match the documented values.
    for sid, expected in EXPECTED_COSTS.items():
        row = scored[scored["subset_id"] == sid]
        actual = float(row["cost"].iloc[0]) if not row.empty else float("nan")
        ok = (not row.empty) and (abs(actual - expected) < 1e-9)
        checks.append((f"cost_{sid}_equals_{expected}", ok, {
            "expected": expected, "actual": actual,
        }))

    # 4) Two-regime split holds at baseline.
    split = su.two_regime_split(frontier)
    checks.append(("two_regime_split_holds", bool(split["holds"]), split))

    all_pass = all(ok for _, ok, _ in checks)

    report = {
        "status": "PASS" if all_pass else "FAIL",
        "published_frontier": str(args.published_frontier),
        "cpor_entry_cost": su.cpor_entry_cost(frontier),
        "checks": [
            {"name": name, "pass": bool(ok), "detail": detail}
            for name, ok, detail in checks
        ],
    }

    su.write_json(report, out_dir, "baseline_reproduce_report.json")
    _write_text_report(report, out_dir)

    if not args.no_figure:
        su.set_style()
        su.plot_frontier(
            scored, frontier, su.get_fig_dir(out_dir),
            stem="baseline_frontier",
            title="Baseline Pareto frontier (reproduction check)",
        )

    _print_summary(report)
    return 0 if all_pass else 1


def _write_text_report(report: dict, out_dir: Path) -> None:
    lines = [
        "Phase 3 cost-sensitivity - baseline reproduction check",
        "=" * 56,
        f"Status: {report['status']}",
        f"Published frontier: {report['published_frontier']}",
        f"CPOR_SM frontier entry cost: {report['cpor_entry_cost']}",
        "",
    ]
    for c in report["checks"]:
        flag = "PASS" if c["pass"] else "FAIL"
        lines.append(f"[{flag}] {c['name']}")
        for k, v in c["detail"].items():
            lines.append(f"        {k}: {v}")
    su.get_results_dir(out_dir)
    (Path(out_dir) / "baseline_reproduce_report.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def _print_summary(report: dict) -> None:
    print(f"Baseline reproduction: {report['status']}")
    for c in report["checks"]:
        flag = "PASS" if c["pass"] else "FAIL"
        print(f"  [{flag}] {c['name']}")
    if report["status"] == "FAIL":
        print("  -> see baseline_reproduce_report.txt for details")


if __name__ == "__main__":
    sys.exit(main())
