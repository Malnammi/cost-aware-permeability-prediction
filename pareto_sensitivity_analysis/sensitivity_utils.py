#!/usr/bin/env python
"""
Shared helpers for the Phase 3 cost-sensitivity analysis.

"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Make ``src`` importable regardless of the current working directory.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.feature_selection import (  # noqa: E402
    compute_subset_cost_bundled,
    identify_pareto_frontier,
    load_phase3_config,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

SWEEP_CSV = ROOT / "results" / "phase3_feature_selection" / "pareto_sweep" / "sweep_results.csv"
CONFIG_PATH = ROOT / "configs" / "phase3_config_bundled.json"
RESULTS_DIR = ROOT / "results" / "phase3_feature_selection" / "pareto_sensitivity"


def get_results_dir(base: Path | str | None = None) -> Path:
    """Return (and create) the sensitivity-analysis results directory."""
    out = Path(base) if base is not None else RESULTS_DIR
    out.mkdir(parents=True, exist_ok=True)
    return out


def get_fig_dir(base: Path | str | None = None) -> Path:
    """Return (and create) the ``figures`` subfolder under the results dir."""
    fig_dir = get_results_dir(base) / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    return fig_dir


# ---------------------------------------------------------------------------
# Recommended operating points (manuscript anchors)
# ---------------------------------------------------------------------------

RECOMMENDED_IDS: Tuple[int, ...] = (640, 4096, 4550)

RECOMMENDED_LABELS: Dict[int, str] = {
    640: "PEF+PHIT (cost 7)",
    4096: "CPOR_SM-only (cost 10)",
    4550: "best-performance (cost 21)",
}

CPOR_FEATURE = "CPOR_SM"
PERF_COL = "RMSE_log_mean"
PERF_DIRECTION = "minimize"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_baseline_cost_config(config_path: Path | str | None = None):
    """
    Load the baseline standalone cost vector and bundle definitions.

    Returns
    -------
    (feature_costs, feature_bundles, sweep_features) : (dict, dict, list)
        ``feature_costs`` maps feature -> integer standalone cost; ``feature_bundles``
        maps derived feature -> ``{"parents": [...], "marginal_cost": int}``;
        ``sweep_features`` is the ordered 13-feature list used for bitmask decoding.
    """
    config = load_phase3_config(config_path or CONFIG_PATH)
    feature_costs = dict(config["feature_costs"])
    feature_bundles = copy.deepcopy(config["feature_bundles"])
    sweep_features = list(config["sweep_features"])
    return feature_costs, feature_bundles, sweep_features


def _decode_subset(subset_id: int, sweep_features: Sequence[str]) -> List[str]:
    """Reconstruct the feature list for a bitmask ``subset_id`` (DEPTH excluded)."""
    return [sweep_features[i] for i in range(len(sweep_features)) if subset_id & (1 << i)]


def load_sweep(
    sweep_csv: Path | str | None = None,
    config_path: Path | str | None = None,
) -> pd.DataFrame:
    """
    Load the cached 8,192-subset sweep with reconstructed feature lists.

    The returned frame carries the fixed performance column (``RMSE_log_mean``),
    a reconstructed ``_feature_list`` decoded from ``subset_id`` (cross-checked
    against the stored ``features`` column), and a ``_has_cpor`` flag.  The
    ``cost`` column is intentionally left to ``recompute_frontier`` so every cost
    is produced by the same bundle-aware function under whatever cost vector is in
    effect.
    """
    sweep_csv = Path(sweep_csv) if sweep_csv is not None else SWEEP_CSV
    _, _, sweep_features = load_baseline_cost_config(config_path)

    df = pd.read_csv(sweep_csv)
    df["_feature_list"] = [_decode_subset(int(sid), sweep_features) for sid in df["subset_id"]]
    df["_has_cpor"] = [CPOR_FEATURE in feats for feats in df["_feature_list"]]

    _crosscheck_feature_lists(df)
    return df


def _crosscheck_feature_lists(df: pd.DataFrame) -> None:
    """Validate bitmask-decoded feature lists against the stored ``features`` column."""
    mismatches = []
    for _, row in df.iterrows():
        stored = str(row["features"]).strip()
        if stored in ("(DEPTH-only)", "", "nan"):
            stored_set = set()
        else:
            stored_set = {f for f in stored.split(",") if f}
        if stored_set != set(row["_feature_list"]):
            mismatches.append(int(row["subset_id"]))
    if mismatches:
        raise ValueError(
            "subset_id bitmask decode disagrees with the stored 'features' column "
            f"for {len(mismatches)} subsets (e.g. {mismatches[:5]}). "
            "Refusing to proceed: the cost engine would operate on the wrong feature sets."
        )


# ---------------------------------------------------------------------------
# Perturbation model 
# ---------------------------------------------------------------------------
#
# A "parameter" is a (feature, kind) pair where kind is "standalone" or "marginal".
#   standalone:  c'(f) = max(1, c(f) * mult)
#   marginal:    m'(f) = min(c'(f), max(0, m(f) * mult))   for baseline m(f) > 0
# Byproduct marginals (DRHO, CT; baseline 0) are never perturbed and stay at 0.

Parameter = Tuple[str, str]  # (feature, kind)

# Primary surface: 7 independently varied parameters.
PRIMARY_PARAMETERS: List[Parameter] = [
    ("CPOR_SM", "standalone"),
    ("PHIT", "standalone"),
    ("PHIT", "marginal"),
    ("SWT", "standalone"),
    ("SWT", "marginal"),
    ("PEF", "standalone"),
    ("PEF", "marginal"),
]

# Secondary surface: all 13 standalone costs + the 3 derived-feature marginals.
SECONDARY_STANDALONE_FEATURES: List[str] = [
    "GR", "CALI", "DRHO", "RHOB", "NPHI", "RT", "CT",
    "PEF", "MSFL", "PHIT", "DT", "SWT", "CPOR_SM",
]
SECONDARY_MARGINAL_FEATURES: List[str] = ["PEF", "PHIT", "SWT"]
SECONDARY_PARAMETERS: List[Parameter] = (
    [(f, "standalone") for f in SECONDARY_STANDALONE_FEATURES]
    + [(f, "marginal") for f in SECONDARY_MARGINAL_FEATURES]
)

PARAMETER_SETS: Dict[str, List[Parameter]] = {
    "primary": PRIMARY_PARAMETERS,
    "secondary": SECONDARY_PARAMETERS,
}

# OAT (one-factor-at-a-time): 9 levels per primary parameter (9 x 7 = 63 incl. baseline).
OAT_LEVELS: List[float] = [-0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4]
OAT_PARAMETERS: List[Parameter] = PRIMARY_PARAMETERS

# Monte Carlo: independent multiplier ~ Uniform(MC_LOW, MC_HIGH) per perturbed parameter.
MC_LOW: float = 0.6
MC_HIGH: float = 1.4
MC_N_DRAWS: int = 20_000
MC_SCENARIOS: Tuple[str, ...] = ("rank_preserving", "unconstrained")


def perturb_costs(
    cost_map: Mapping[str, float],
    bundles: Mapping[str, dict],
    multipliers: Mapping[Parameter, float],
) -> Tuple[Dict[str, float], Dict[str, dict]]:
    """
    Apply multiplicative cost perturbations and return a new ``(cost_map, bundles)``.

    Parameters
    ----------
    cost_map : mapping
        Baseline standalone costs (feature -> cost).
    bundles : mapping
        Baseline bundle definitions (feature -> {"parents", "marginal_cost"}).
    multipliers : mapping
        ``{(feature, kind): multiplier}``.  A multiplier of ``1 + delta`` is applied
        as in the manuscript.  Features/kinds absent from the mapping keep their
        baseline value (the coherence clip ``m' <= c'`` is still enforced).

    Notes
    -----
    Standalone costs are perturbed first so the marginal coherence clip uses the
    perturbed standalone cost ``c'(f)``.  The standalone floor is 1; the marginal
    floor is 0.  No upper cap is imposed.  Byproduct marginals (baseline 0) remain 0.
    """
    new_cost: Dict[str, float] = dict(cost_map)
    for (feat, kind), mult in multipliers.items():
        if kind != "standalone":
            continue
        new_cost[feat] = max(1.0, float(cost_map[feat]) * float(mult))

    new_bundles: Dict[str, dict] = copy.deepcopy(dict(bundles))
    for feat, spec in new_bundles.items():
        base_marginal = float(bundles[feat]["marginal_cost"])
        c_prime = float(new_cost.get(feat, cost_map.get(feat, 0.0)))
        key = (feat, "marginal")
        if key in multipliers and base_marginal > 0:
            raw = max(0.0, base_marginal * float(multipliers[key]))
        else:
            raw = base_marginal
        spec["marginal_cost"] = min(c_prime, raw)

    return new_cost, new_bundles


def deltas_to_multipliers(deltas: Mapping[Parameter, float]) -> Dict[Parameter, float]:
    """Convert perturbation fractions ``delta`` to multipliers ``1 + delta``."""
    return {param: 1.0 + float(d) for param, d in deltas.items()}


# ---------------------------------------------------------------------------
# Frontier recomputation (the consistency choke point)
# ---------------------------------------------------------------------------

def recompute_frontier(
    sweep_df: pd.DataFrame,
    cost_map: Mapping[str, float],
    bundles: Mapping[str, dict],
    perf_col: str = PERF_COL,
    direction: str = PERF_DIRECTION,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Recompute the per-subset cost under ``cost_map``/``bundles`` and the Pareto frontier.

    Returns ``(scored_df, frontier_df)`` where ``scored_df`` is ``sweep_df`` with a
    freshly computed ``cost`` column and ``frontier_df`` is the non-dominated subset
    (ascending cost).  Performance is taken untouched from ``perf_col``.
    """
    df = sweep_df.copy()
    df["cost"] = [
        compute_subset_cost_bundled(feats, cost_map, bundles)
        for feats in df["_feature_list"]
    ]
    frontier = identify_pareto_frontier(df, "cost", perf_col, direction=direction)
    return df, frontier


# ---------------------------------------------------------------------------
# Structural tests
# ---------------------------------------------------------------------------

def frontier_ids(frontier: pd.DataFrame) -> set:
    """Return the set of ``subset_id`` values on a frontier."""
    return set(int(s) for s in frontier["subset_id"])


def is_on_frontier(frontier: pd.DataFrame, subset_id: int) -> bool:
    """Whether ``subset_id`` is non-dominated on this frontier."""
    return int(subset_id) in frontier_ids(frontier)


def jaccard(set_a: Iterable[int], set_b: Iterable[int]) -> float:
    """Jaccard similarity between two subset-id sets (1.0 if both empty)."""
    a, b = set(set_a), set(set_b)
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def two_regime_split(frontier: pd.DataFrame) -> Dict[str, float]:
    """
    Test whether the frontier preserves the wireline / CPOR_SM two-regime split.

    The split is a *cost discontinuity* that cleanly separates the cheap wireline-only
    regime from the expensive CPOR_SM-inclusive regime.  On a Pareto frontier higher
    cost implies lower RMSE_log, so a clean separation by cost automatically places the
    high-accuracy regime at the CPOR_SM end; no separate accuracy-gap clause is needed.

    The split holds iff all three conditions are met:
      1. at least one *substantive* wireline-only point is on the frontier;
      2. at least one CPOR_SM-inclusive point is on the frontier;
      3. the two cost ranges do not overlap, i.e. the most expensive substantive
         wireline-only frontier point is strictly cheaper than the cheapest
         CPOR_SM-inclusive one (no CPOR_SM subset appears at or below a wireline
         subset's cost).

    A "substantive" wireline point carries at least one wireline log (``n_features >=
    1``).  The DEPTH-only null baseline (``n_features == 0``, no CPOR_SM) is excluded
    because it is not a real wireline-acquisition operating point; counting it would
    anchor the wireline side at cost 0 and make the split hold trivially no matter how
    cheap CPOR_SM becomes, masking the genuine breakpoint.

    Returns a dict with ``holds`` (bool), ``boundary_cost`` (cheapest CPOR_SM frontier
    cost), ``wireline_max_cost`` (most expensive substantive wireline-only frontier
    cost), ``n_wireline`` and ``n_cpor`` (frontier counts in each regime).
    """
    result = {
        "holds": False,
        "boundary_cost": float("nan"),
        "wireline_max_cost": float("nan"),
        "n_wireline": 0,
        "n_cpor": 0,
    }

    has_cpor = frontier["_has_cpor"].to_numpy(dtype=bool)
    cost = frontier["cost"].to_numpy(dtype=float)
    n_features = frontier["n_features"].to_numpy(dtype=float)

    is_wireline = (~has_cpor) & (n_features >= 1)
    wireline_costs = cost[is_wireline]
    cpor_costs = cost[has_cpor]

    result["n_wireline"] = int(wireline_costs.size)
    result["n_cpor"] = int(cpor_costs.size)

    if wireline_costs.size == 0 or cpor_costs.size == 0:
        return result

    wireline_max = float(wireline_costs.max())
    cpor_min = float(cpor_costs.min())
    result["wireline_max_cost"] = wireline_max
    result["boundary_cost"] = cpor_min
    result["holds"] = bool(wireline_max < cpor_min)
    return result


def cpor_entry_cost(frontier: pd.DataFrame) -> float:
    """Minimum frontier cost among CPOR_SM-inclusive subsets (NaN if none)."""
    cpor_rows = frontier[frontier["_has_cpor"]]
    if cpor_rows.empty:
        return float("nan")
    return float(cpor_rows["cost"].min())


def respects_tier_order(
    cost_map: Mapping[str, float],
    base_cost_map: Mapping[str, float],
    features: Sequence[str],
) -> bool:
    """
    Whether perturbed standalone costs preserve the baseline weak (non-strict) tier order.

    Features sharing a baseline cost form a tier; ties within a tier are allowed.
    Across distinct baseline tiers, every feature in a lower tier must have a perturbed
    cost no greater than every feature in the next-higher tier (no rank inversions).
    """
    tiers: Dict[float, List[str]] = {}
    for f in features:
        tiers.setdefault(float(base_cost_map[f]), []).append(f)

    ordered_levels = sorted(tiers)
    prev_max = -float("inf")
    for level in ordered_levels:
        perturbed = [float(cost_map[f]) for f in tiers[level]]
        lo, hi = min(perturbed), max(perturbed)
        if lo < prev_max:
            return False
        prev_max = hi
    return True


# ---------------------------------------------------------------------------
# Plotting (mirrors runners/analyze_phase3.py styling)
# ---------------------------------------------------------------------------

def set_style() -> None:
    """Match ``analyze_phase3._set_style`` so figures look native to the paper."""
    sns.set_theme(style="whitegrid", font_scale=1.1)
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "font.family": "serif",
    })


def save_fig(fig: plt.Figure, fig_dir: Path, stem: str) -> None:
    """Write a figure to ``pdf/``, ``svg/``, ``png/`` subfolders (as in analyze_phase3)."""
    fig_dir = Path(fig_dir)
    for sub in ("pdf", "svg", "png"):
        (fig_dir / sub).mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_dir / "pdf" / f"{stem}.pdf")
    fig.savefig(fig_dir / "svg" / f"{stem}.svg")
    fig.savefig(fig_dir / "png" / f"{stem}.png")
    plt.close(fig)


def plot_frontier(
    scored_df: pd.DataFrame,
    frontier: pd.DataFrame,
    fig_dir: Path,
    stem: str,
    perf_col: str = PERF_COL,
    title: str | None = None,
    recommended_ids: Sequence[int] = RECOMMENDED_IDS,
) -> None:
    """
    Plot a single cost-vs-RMSE_log frontier in the analyze_phase3 palette.

    Gray scatter for all subsets, a crimson ``o-`` Pareto line with black marker
    edges, and the recommended operating points marked.
    """
    fig, ax = plt.subplots(figsize=(10, 7))

    ax.scatter(
        scored_df["cost"], scored_df[perf_col],
        s=8, alpha=0.15, color="gray", rasterized=True,
        label=f"All subsets (n={len(scored_df)})",
    )

    fr = frontier.sort_values("cost")
    ax.plot(
        fr["cost"], fr[perf_col], "o-",
        color="crimson", linewidth=2, markersize=7,
        markeredgecolor="black", markeredgewidth=0.5,
        label=f"Pareto frontier (n={len(fr)})", zorder=5,
    )

    fr_ids = frontier_ids(frontier)
    for sid in recommended_ids:
        row = scored_df[scored_df["subset_id"] == sid]
        if row.empty:
            continue
        row = row.iloc[0]
        on = sid in fr_ids
        ax.scatter(
            [row["cost"]], [row[perf_col]],
            s=140, marker="*",
            color="gold" if on else "white",
            edgecolors="black", linewidth=0.8, zorder=6,
            label=f"{RECOMMENDED_LABELS.get(sid, sid)}{'' if on else ' (off)'}",
        )

    ax.set_xlabel("Acquisition cost (surrogate score)")
    ax.set_ylabel(r"$\mathrm{RMSE}_{\log}$ mean (lower is better)")
    ax.set_title(title or "Pareto frontier under perturbed costs")
    ax.legend(loc="best", fontsize=8)

    save_fig(fig, fig_dir, stem)


# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------

def write_csv(df: pd.DataFrame, out_dir: Path | str, name: str) -> Path:
    """Write a DataFrame to ``out_dir/name`` and return the path."""
    out_dir = get_results_dir(out_dir)
    path = out_dir / name
    df.to_csv(path, index=False)
    return path


def write_json(payload: dict, out_dir: Path | str, name: str) -> Path:
    """Write a JSON report to ``out_dir/name`` and return the path."""
    out_dir = get_results_dir(out_dir)
    path = out_dir / name
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, default=_json_default)
    return path


def _json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (set, np.ndarray)):
        return list(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
