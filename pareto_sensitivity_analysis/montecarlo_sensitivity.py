#!/usr/bin/env python
"""
Monte Carlo (global) cost sensitivity.

Perturbs all targeted costs simultaneously with independent multipliers drawn from
``Uniform(0.7, 1.3)`` and recomputes the Pareto frontier for every draw (performance
is never recomputed). 

Scenarios
---------
- ``rank_preserving``: rejection-sample draws so the perturbed standalone costs keep the
  documented weak (non-strict) tier ordering of tab:bundle_surrogate_costs (ties allowed,
  no cross-tier inversions).  This is the primary analysis.
- ``unconstrained``: accept every draw (rank inversions allowed); a harsher stress test.

To run:

    python pareto_sensitivity_analysis/montecarlo_sensitivity.py --parameter-set primary --scenario rank_preserving
    python pareto_sensitivity_analysis/montecarlo_sensitivity.py --parameter-set secondary --scenario unconstrained --n-draws 10000
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import sensitivity_utils as su  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--parameter-set", required=True, choices=tuple(su.PARAMETER_SETS),
        help="Which perturbation surface to sample (primary 7-param or secondary 13+3).",
    )
    p.add_argument(
        "--scenario", required=True, choices=su.MC_SCENARIOS,
        help="rank_preserving (tier order enforced) or unconstrained (inversions allowed).",
    )
    p.add_argument(
        "--n-draws", type=int, default=su.MC_N_DRAWS,
        help=f"Accepted draws to collect (default {su.MC_N_DRAWS}).",
    )
    p.add_argument(
        "--seed", type=int, default=None,
        help="RNG seed (default: deterministic function of parameter-set and scenario).",
    )
    p.add_argument(
        "--max-reject-factor", type=int, default=2000,
        help="Cap on total attempts = n_draws * factor for rank_preserving rejection "
             "sampling (default 2000); prevents an unbounded loop if acceptance is low.",
    )
    p.add_argument(
        "--out-dir", default=None,
        help="Results directory (default: "
             "results/phase3_feature_selection/pareto_sensitivity/montecarlo_sensitivity/).",
    )
    p.add_argument("--tag", default="", help="Optional label recorded in the outputs.")
    return p.parse_args()


def derive_seed(parameter_set: str, scenario: str) -> int:
    """Deterministic, independent seed per (parameter_set, scenario) cell."""
    key = f"pareto-mc|{parameter_set}|{scenario}".encode("utf-8")
    return int(hashlib.md5(key).hexdigest(), 16) % (2 ** 32)


def run_cell(
    sweep: pd.DataFrame,
    base_cost,
    base_bundles,
    parameters,
    scenario: str,
    n_draws: int,
    rng: np.random.Generator,
    max_reject_factor: int,
):
    """Run one MC cell; return (per-draw DataFrame, attempts, n_accepted)."""
    _, baseline_frontier = su.recompute_frontier(sweep, base_cost, base_bundles)
    baseline_ids = su.frontier_ids(baseline_frontier)

    # Feature lists of the recommended subsets, so their (moving) cost can be recorded
    # per draw without scanning all 8,192 rows each time.
    rec_features = {
        sid: sweep.loc[sweep["subset_id"] == sid, "_feature_list"].iloc[0]
        for sid in su.RECOMMENDED_IDS
    }

    enforce_order = scenario == "rank_preserving"
    max_attempts = n_draws * max_reject_factor if enforce_order else n_draws

    records = []
    attempts = 0
    while len(records) < n_draws and attempts < max_attempts:
        attempts += 1
        multipliers = {
            param: float(rng.uniform(su.MC_LOW, su.MC_HIGH)) for param in parameters
        }
        cost_map, bundles = su.perturb_costs(base_cost, base_bundles, multipliers)

        if enforce_order and not su.respects_tier_order(
            cost_map, base_cost, su.SECONDARY_STANDALONE_FEATURES
        ):
            continue

        _, frontier = su.recompute_frontier(sweep, cost_map, bundles)
        fr_ids = su.frontier_ids(frontier)
        split = su.two_regime_split(frontier)

        rec = {
            "draw": len(records),
            "frontier_size": len(fr_ids),
            "jaccard_vs_baseline": su.jaccard(fr_ids, baseline_ids),
            "split_holds": bool(split["holds"]),
            "boundary_cost": split["boundary_cost"],
            "wireline_max_cost": split["wireline_max_cost"],
            "cpor_entry_cost": su.cpor_entry_cost(frontier),
        }
        for sid in su.RECOMMENDED_IDS:
            rec[f"on_frontier_{sid}"] = sid in fr_ids
            rec[f"cost_{sid}"] = float(
                su.compute_subset_cost_bundled(rec_features[sid], cost_map, bundles)
            )
        records.append(rec)

    return pd.DataFrame(records), attempts, len(records)


def summarize(draws: pd.DataFrame, parameter_set: str, scenario: str,
              attempts: int, n_accepted: int, n_requested: int, tag: str) -> pd.DataFrame:
    """Collapse per-draw records to a tidy long table keyed by metric."""
    def _q(col, q):
        return float(np.percentile(draws[col], q)) if len(draws) else float("nan")

    metrics = {
        "p_on_frontier_640": float(draws["on_frontier_640"].mean()),
        "p_on_frontier_4096": float(draws["on_frontier_4096"].mean()),
        "p_on_frontier_4550": float(draws["on_frontier_4550"].mean()),
        "p_split_holds": float(draws["split_holds"].mean()),
        "jaccard_median": _q("jaccard_vs_baseline", 50),
        "jaccard_p5": _q("jaccard_vs_baseline", 5),
        "jaccard_min": float(draws["jaccard_vs_baseline"].min()) if len(draws) else float("nan"),
        "cpor_entry_cost_median": _q("cpor_entry_cost", 50),
        "cpor_entry_cost_p5": _q("cpor_entry_cost", 5),
        "cpor_entry_cost_p95": _q("cpor_entry_cost", 95),
        "n_requested": float(n_requested),
        "n_accepted": float(n_accepted),
        "attempts": float(attempts),
        "acceptance_rate": float(n_accepted / attempts) if attempts else float("nan"),
    }
    return pd.DataFrame({
        "parameter_set": parameter_set,
        "scenario": scenario,
        "tag": tag,
        "metric": list(metrics),
        "value": list(metrics.values()),
    })


def main() -> int:
    args = parse_args()
    out_dir = su.get_results_dir(args.out_dir or (su.RESULTS_DIR / "montecarlo_sensitivity"))

    seed = args.seed if args.seed is not None else derive_seed(args.parameter_set, args.scenario)
    rng = np.random.default_rng(seed)

    sweep = su.load_sweep()
    base_cost, base_bundles, _ = su.load_baseline_cost_config()
    parameters = su.PARAMETER_SETS[args.parameter_set]

    draws, attempts, n_accepted = run_cell(
        sweep, base_cost, base_bundles, parameters,
        args.scenario, args.n_draws, rng, args.max_reject_factor,
    )

    stem = f"montecarlo_{args.parameter_set}_{args.scenario}"
    draws.insert(0, "scenario", args.scenario)
    draws.insert(0, "parameter_set", args.parameter_set)
    su.write_csv(draws, out_dir, f"{stem}_draws.csv")

    summary = summarize(
        draws, args.parameter_set, args.scenario,
        attempts, n_accepted, args.n_draws, args.tag,
    )
    su.write_csv(summary, out_dir, f"{stem}.csv")

    _print_summary(args, seed, attempts, n_accepted, summary)
    if n_accepted < args.n_draws:
        print(f"  WARNING: only {n_accepted}/{args.n_draws} draws accepted within "
              f"{attempts} attempts (raise --max-reject-factor if needed).")
    return 0


def _print_summary(args, seed, attempts, n_accepted, summary) -> None:
    print(f"MC cell complete: parameter_set={args.parameter_set} scenario={args.scenario}")
    print(f"  seed={seed}  attempts={attempts}  accepted={n_accepted}")
    keyed = dict(zip(summary["metric"], summary["value"]))
    for k in ("p_on_frontier_640", "p_on_frontier_4096", "p_on_frontier_4550",
              "p_split_holds", "jaccard_median", "acceptance_rate"):
        print(f"  {k}: {keyed[k]:.4f}")


if __name__ == "__main__":
    sys.exit(main())
