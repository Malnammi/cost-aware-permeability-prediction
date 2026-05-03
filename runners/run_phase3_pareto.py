#!/usr/bin/env python
"""
Phase 3 Pareto Sweep — Cost-aware Feature Selection

Three modes:
  --sweep    : Brute-force evaluate all 2^13 feature subsets (fixed HPs)
  --retune   : Re-tune HPs for Pareto-optimal subsets from the sweep
  --validate : Re-evaluate Pareto subsets with the validation model

Outputs (to results/phase3_feature_selection/):
  pareto_sweep/
    - sweep_results.csv              : all 8192 subsets with metrics
    - pareto_frontier_{metric}.csv   : Pareto-optimal subsets per metric
    - sweep_meta.json
  pareto_retune/
    - {subset_id}_best_params.json   : tuned HPs per Pareto subset
    - retune_results.csv             : re-evaluated metrics after tuning
    - pareto_frontier_retune.csv     : refined Pareto frontier (RMSE_log)
    - retune_meta.json
  pareto_validation/
    - validation_results.csv         : validation model metrics per subset
    - validation_meta.json

Usage:
    python runners/run_phase3_pareto.py --sweep [-o OUTPUT_DIR] [--quiet]
    python runners/run_phase3_pareto.py --retune [-o OUTPUT_DIR] [--hp_budget 1000] [--quiet]
    python runners/run_phase3_pareto.py --validate [-o OUTPUT_DIR] [--quiet]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.preprocessing import (
    load_features_info,
    get_preprocessor,
    detect_variant_type,
    transform_target,
)
from src.models import get_model
from src.cv_utils import get_group_kfold_splits, compute_fold_summary
from src.feature_selection import (
    load_phase3_config,
    enumerate_feature_subsets,
    compute_subset_cost,
    compute_subset_cost_bundled,
    resolve_binary_columns,
    evaluate_subset,
    identify_pareto_frontier,
)
from src.hp_search import run_hybrid_search

RANKING_METRICS = ["RMSE_log", "MAE_log", "R2_log", "RMSE", "MAE", "R2"]
PARETO_SELECTION_SUFFIX = "_mean"

METRIC_DIRECTIONS = {
    "RMSE_log": "minimize",
    "MAE_log": "minimize",
    "R2_log": "maximize",
    "RMSE": "minimize",
    "MAE": "minimize",
    "R2": "maximize",
}


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------

def _numpy_serializer(obj):
    """JSON serializer for numpy types."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


def _pareto_selection_col(metric: str) -> str:
    """Return the metric column used for Pareto selection."""
    return f"{metric}{PARETO_SELECTION_SUFFIX}"


def _build_result_row(bitmask, subset, cost, fold_metrics):
    """Build one CSV row from a subset evaluation's per-fold metrics."""
    summary = compute_fold_summary(fold_metrics)
    n_folds = len(fold_metrics)
    t_crit = stats.t.ppf(1 - 0.05 / 2, df=n_folds - 1)
    row = {
        "subset_id": bitmask,
        "features": ",".join(subset) if subset else "(DEPTH-only)",
        "n_features": len(subset),
        "cost": cost,
    }
    row.update(summary)
    for metric in RANKING_METRICS:
        direction = METRIC_DIRECTIONS[metric]
        mean_val = summary[f"{metric}_mean"]
        std_val = summary[f"{metric}_std"]
        sem = std_val / np.sqrt(n_folds)
        if direction == "minimize":
            row[f"{metric}_ci"] = mean_val + t_crit * sem
        else:
            row[f"{metric}_ci"] = mean_val - t_crit * sem
    return row


def _save_pareto_frontiers(results_df, output_dir, verbose=True):
    """Identify and save Pareto frontiers using mean metric columns."""
    frontiers = {}
    for metric in RANKING_METRICS:
        score_col = _pareto_selection_col(metric)
        direction = METRIC_DIRECTIONS[metric]
        frontier = identify_pareto_frontier(
            results_df, "cost", score_col, direction,
        )
        path = output_dir / f"pareto_frontier_{metric}.csv"
        frontier.to_csv(path, index=False)
        frontiers[metric] = frontier
        if verbose:
            print(
                f"  Pareto frontier ({metric:8s}): "
                f"{len(frontier):3d} subsets -> {path.name}"
            )
    return frontiers


def _parse_feature_list(feat_str):
    """Parse the comma-separated feature string back into a list."""
    if not feat_str or feat_str == "(DEPTH-only)":
        return []
    return feat_str.split(",")


# -------------------------------------------------------------------
# Mode 1: Sweep
# -------------------------------------------------------------------

def run_sweep(
    output_dir: str = None,
    config_path: str = None,
    verbose: bool = True,
) -> dict:
    """
    Brute-force evaluate all 2^13 feature subsets with fixed HPs.

    Parameters
    ----------
    output_dir : str, optional
        Override output directory.
    config_path : str, optional
        Path to a phase3 config file (standalone or bundled).
    verbose : bool
        Print progress.

    Returns
    -------
    dict
        Run metadata including paths and timing.
    """
    project_root = Path(__file__).parent.parent
    config = load_phase3_config(config_path)
    features_info = load_features_info()

    model_name = config["best_model"]
    variant_path = project_root / config["best_variant"]
    params_path = project_root / config["best_params_path"]
    random_state = config["random_state"]
    sweep_features = config["sweep_features"]
    always_include = config["always_include"]
    feature_costs = config["feature_costs"]
    binary_map = config["feature_binary_map"]
    cost_mode = config.get("cost_mode", "standalone")
    bundles = config.get("feature_bundles", {})

    def _cost(subset):
        if cost_mode == "bundled" and bundles:
            return compute_subset_cost_bundled(subset, feature_costs, bundles)
        return compute_subset_cost(subset, feature_costs)

    n_total = 1 << len(sweep_features)

    if verbose:
        print("Phase 3 Pareto Sweep  |  Mode: --sweep")
        print(f"  Model:          {model_name}")
        print(f"  Variant:        {variant_path.name}")
        print(f"  Cost mode:      {cost_mode}")
        print(f"  Sweep features: {len(sweep_features)}")
        print(f"  Total subsets:  {n_total}")

    # ------------------------------------------------------------------
    # Load data and hyperparameters
    # ------------------------------------------------------------------
    df = pd.read_csv(variant_path)
    has_outlier = detect_variant_type(df)

    with open(params_path, "r") as f:
        best_params = json.load(f)["best_params"]

    splits = get_group_kfold_splits(df)

    if verbose:
        print(f"  Samples:        {len(df)}")
        print(f"  Folds:          {len(splits)}")
        print(f"  Best HPs:       {best_params}")

    # ------------------------------------------------------------------
    # Factories
    # ------------------------------------------------------------------
    np.random.seed(random_state)
    model_class, _, default_kwargs = get_model(model_name)

    def model_factory():
        return model_class(
            **default_kwargs, random_state=random_state, **best_params,
        )

    def preprocessor_factory(continuous_cols, binary_cols):
        return get_preprocessor(
            features_info=features_info,
            has_outlier_columns=has_outlier,
            continuous_subset=continuous_cols,
            binary_subset=binary_cols,
        )

    # ------------------------------------------------------------------
    # Sweep all 2^13 subsets
    # ------------------------------------------------------------------
    results_rows = []
    total_start = time.time()

    for idx, (bitmask, subset) in enumerate(
        enumerate_feature_subsets(sweep_features),
    ):
        cost = _cost(subset)

        fold_metrics = evaluate_subset(
            subset_features=subset,
            df=df,
            preprocessor_factory=preprocessor_factory,
            model_factory=model_factory,
            splits=splits,
            always_include=always_include,
            binary_map=binary_map,
        )

        row = _build_result_row(bitmask, subset, cost, fold_metrics)
        results_rows.append(row)

        if verbose and (idx + 1) % 100 == 0:
            elapsed = time.time() - total_start
            rate = (idx + 1) / elapsed
            eta = (n_total - idx - 1) / rate if rate > 0 else 0
            print(
                f"  [{idx + 1:5d}/{n_total}]  "
                f"cost={cost:2d}  n_feat={len(subset):2d}  "
                f"RMSE_log={row['RMSE_log_mean']:.4f}  "
                f"rate={rate:.1f} sub/s  ETA={eta / 60:.1f}m"
            )

    total_elapsed = time.time() - total_start

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    if output_dir is None:
        output_dir = (
            project_root / "results" / "phase3_feature_selection" / "pareto_sweep"
        )
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results_df = pd.DataFrame(results_rows)
    sweep_path = output_dir / "sweep_results.csv"
    results_df.to_csv(sweep_path, index=False)

    if verbose:
        print(f"\n  Sweep results saved: {sweep_path}")

    frontiers = _save_pareto_frontiers(results_df, output_dir, verbose)

    meta = {
        "mode": "sweep",
        "model": model_name,
        "variant": config["best_variant"],
        "cost_mode": cost_mode,
        "best_params": best_params,
        "random_state": random_state,
        "n_subsets": n_total,
        "n_sweep_features": len(sweep_features),
        "sweep_features": sweep_features,
        "elapsed_seconds": total_elapsed,
        "timestamp": datetime.now().isoformat(),
    }
    meta_path = output_dir / "sweep_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2, default=_numpy_serializer)

    if verbose:
        print(f"\n{'=' * 60}")
        print(
            f"Sweep complete.  {n_total} subsets in {total_elapsed:.1f}s "
            f"({total_elapsed / 60:.1f} min)"
        )
        primary = frontiers.get("RMSE_log", pd.DataFrame())
        if not primary.empty:
            print(
                f"\nPrimary Pareto frontier (RMSE_log): "
                f"{len(primary)} subsets"
            )
            for _, r in primary.iterrows():
                print(
                    f"  cost={r['cost']:2.0f}  "
                    f"RMSE_log_mean={r['RMSE_log_mean']:.4f}  "
                    f"n_feat={r['n_features']:.0f}  "
                    f"features={r['features']}"
                )
        print(f"{'=' * 60}")

    return meta


# -------------------------------------------------------------------
# Mode 2: Retune
# -------------------------------------------------------------------

def run_retune(
    output_dir: str = None,
    config_path: str = None,
    hp_budget: int = None,
    sweep_dir: str = None,
    verbose: bool = True,
) -> dict:
    """
    Re-tune hyperparameters for Pareto-optimal subsets.

    Loads the primary (RMSE_log) Pareto frontier from the sweep, runs
    a full hybrid HP search for each subset, then re-evaluates with
    per-fold CV to produce a refined Pareto frontier.

    Parameters
    ----------
    output_dir : str, optional
        Override output directory.
    config_path : str, optional
        Path to a phase3 config file (standalone or bundled).
    hp_budget : int, optional
        HP search budget per subset.  Falls back to config's
        ``retune_hp_budget`` (default 1000).
    sweep_dir : str, optional
        Directory containing sweep results.
    verbose : bool
        Print progress.

    Returns
    -------
    dict
        Run metadata.
    """
    project_root = Path(__file__).parent.parent
    config = load_phase3_config(config_path)
    features_info = load_features_info()

    model_name = config["best_model"]
    variant_path = project_root / config["best_variant"]
    random_state = config["random_state"]
    sweep_features = config["sweep_features"]
    always_include = config["always_include"]
    feature_costs = config["feature_costs"]
    binary_map = config["feature_binary_map"]
    cost_mode = config.get("cost_mode", "standalone")
    bundles = config.get("feature_bundles", {})
    budget = hp_budget or config.get("retune_hp_budget", 1000)

    def _cost(subset):
        if cost_mode == "bundled" and bundles:
            return compute_subset_cost_bundled(subset, feature_costs, bundles)
        return compute_subset_cost(subset, feature_costs)

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    df = pd.read_csv(variant_path)
    has_outlier = detect_variant_type(df)
    target_col = features_info.get("target_label", "CKHL_SM")
    y_raw = df[target_col].values
    y_log = transform_target(y_raw)
    groups = df["Source"].values
    splits = get_group_kfold_splits(df)

    np.random.seed(random_state)
    model_class, param_distributions, default_kwargs = get_model(model_name)

    # ------------------------------------------------------------------
    # Load primary Pareto frontier from sweep
    # ------------------------------------------------------------------
    if sweep_dir is None:
        sweep_dir = (
            project_root / "results" / "phase3_feature_selection" / "pareto_sweep"
        )
    else:
        sweep_dir = Path(sweep_dir)

    frontier_path = sweep_dir / "pareto_frontier_RMSE_log.csv"
    if not frontier_path.exists():
        raise FileNotFoundError(
            f"Primary Pareto frontier not found: {frontier_path}\n"
            "Run --sweep first."
        )

    frontier_df = pd.read_csv(frontier_path)

    if verbose:
        print("Phase 3 Pareto Sweep  |  Mode: --retune")
        print(f"  Model:          {model_name}")
        print(f"  Variant:        {variant_path.name}")
        print(f"  Cost mode:      {cost_mode}")
        print(f"  HP budget:      {budget}")
        print(f"  Pareto subsets: {len(frontier_df)}")
        print(f"  Samples:        {len(df)}")
        print(f"  Folds:          {len(splits)}")

    # ------------------------------------------------------------------
    # Output directory
    # ------------------------------------------------------------------
    if output_dir is None:
        output_dir = (
            project_root / "results" / "phase3_feature_selection" / "pareto_retune"
        )
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Preprocessor factory for per-fold evaluation after tuning
    def preprocessor_factory(continuous_cols, binary_cols):
        return get_preprocessor(
            features_info=features_info,
            has_outlier_columns=has_outlier,
            continuous_subset=continuous_cols,
            binary_subset=binary_cols,
        )

    # ------------------------------------------------------------------
    # Re-tune each Pareto-optimal subset
    # ------------------------------------------------------------------
    total_start = time.time()
    retune_rows = []

    for row_idx, frow in frontier_df.iterrows():
        subset_id = int(frow["subset_id"])
        subset = _parse_feature_list(frow["features"])
        cost = _cost(subset)

        if verbose:
            print(
                f"\n  [{row_idx + 1}/{len(frontier_df)}]  "
                f"subset_id={subset_id}  cost={cost}  "
                f"features={frow['features']}"
            )

        # Build reduced column set for this subset
        continuous_cols = list(always_include["continuous"]) + list(subset)
        binary_cols = (
            resolve_binary_columns(subset, binary_map)
            + list(always_include.get("binary", []))
        )
        categorical_cols = list(always_include["categorical"])
        all_cols = continuous_cols + categorical_cols + binary_cols

        X_raw = df[all_cols]
        preprocessor = get_preprocessor(
            features_info=features_info,
            has_outlier_columns=has_outlier,
            continuous_subset=continuous_cols,
            binary_subset=binary_cols,
        )

        # Hybrid HP search (preprocessor fitted per fold to avoid leakage)
        search_results = run_hybrid_search(
            model_name=model_name,
            model_class=model_class,
            param_distributions=param_distributions,
            X=X_raw,
            y=y_log,
            groups=groups,
            budget=budget,
            random_state=random_state,
            default_kwargs=default_kwargs,
            verbose=verbose,
            preprocessor=preprocessor,
        )

        # Save best params for this subset
        params_out = {
            "subset_id": subset_id,
            "features": subset,
            "cost": cost,
            "best_params": search_results["best_params"],
            "best_score": search_results["best_score"],
            "best_metrics": search_results["best_metrics"],
            "timing": search_results["timing"],
        }
        params_path = output_dir / f"{subset_id}_best_params.json"
        with open(params_path, "w") as f:
            json.dump(params_out, f, indent=2, default=_numpy_serializer)

        # Re-evaluate with per-fold preprocessing for consistent metrics
        tuned_params = search_results["best_params"]

        def model_factory(_bp=tuned_params):
            return model_class(
                **default_kwargs, random_state=random_state, **_bp,
            )

        fold_metrics = evaluate_subset(
            subset_features=subset,
            df=df,
            preprocessor_factory=preprocessor_factory,
            model_factory=model_factory,
            splits=splits,
            always_include=always_include,
            binary_map=binary_map,
        )

        row = _build_result_row(subset_id, subset, cost, fold_metrics)
        retune_rows.append(row)

        if verbose:
            sweep_rmse = frow.get("RMSE_log_mean", None)
            sweep_str = f"{sweep_rmse:.4f}" if sweep_rmse is not None else "N/A"
            print(
                f"    Tuned RMSE_log: {row['RMSE_log_mean']:.4f} +/- "
                f"{row['RMSE_log_std']:.4f}  "
                f"(sweep was {sweep_str})"
            )

    total_elapsed = time.time() - total_start

    # ------------------------------------------------------------------
    # Save retune results and refined frontier
    # ------------------------------------------------------------------
    retune_df = pd.DataFrame(retune_rows)
    retune_path = output_dir / "retune_results.csv"
    retune_df.to_csv(retune_path, index=False)

    if verbose:
        print(f"\n  Retune results saved: {retune_path}")

    # Refined Pareto frontiers from tuned metrics
    for metric in RANKING_METRICS:
        score_col = _pareto_selection_col(metric)
        direction = METRIC_DIRECTIONS[metric]
        if score_col in retune_df.columns:
            frontier = identify_pareto_frontier(
                retune_df, "cost", score_col, direction,
            )
            fp = output_dir / f"pareto_frontier_retune_{metric}.csv"
            frontier.to_csv(fp, index=False)
            if verbose:
                print(
                    f"  Refined frontier ({metric:8s}): "
                    f"{len(frontier):3d} subsets -> {fp.name}"
                )

    # Primary refined frontier (RMSE_log) at the expected path
    if "RMSE_log_mean" in retune_df.columns:
        primary_refined = identify_pareto_frontier(
            retune_df, "cost", "RMSE_log_mean", "minimize",
        )
        primary_path = output_dir / "pareto_frontier_retune.csv"
        primary_refined.to_csv(primary_path, index=False)

    meta = {
        "mode": "retune",
        "model": model_name,
        "variant": config["best_variant"],
        "cost_mode": cost_mode,
        "random_state": random_state,
        "hp_budget": budget,
        "n_pareto_subsets": len(frontier_df),
        "elapsed_seconds": total_elapsed,
        "timestamp": datetime.now().isoformat(),
    }
    meta_path = output_dir / "retune_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2, default=_numpy_serializer)

    if verbose:
        print(f"\n{'=' * 60}")
        print(
            f"Retune complete.  {len(frontier_df)} subsets in "
            f"{total_elapsed:.1f}s ({total_elapsed / 60:.1f} min)"
        )
        print(f"Results: {retune_path}")
        print(f"{'=' * 60}")

    return meta


# -------------------------------------------------------------------
# Mode 3: Validate
# -------------------------------------------------------------------

def run_validate(
    output_dir: str = None,
    config_path: str = None,
    retune_dir: str = None,
    sweep_dir: str = None,
    verbose: bool = True,
) -> dict:
    """
    Re-evaluate Pareto subsets with the validation model for robustness.

    Uses the refined Pareto frontier (from retune) if available,
    otherwise falls back to the sweep frontier.

    Parameters
    ----------
    output_dir : str, optional
        Override output directory.
    config_path : str, optional
        Path to a phase3 config file (standalone or bundled).
    retune_dir : str, optional
        Directory containing retune results.
    sweep_dir : str, optional
        Directory containing sweep results.
    verbose : bool
        Print progress.

    Returns
    -------
    dict
        Run metadata.
    """
    project_root = Path(__file__).parent.parent
    config = load_phase3_config(config_path)
    features_info = load_features_info()

    random_state = config["random_state"]
    always_include = config["always_include"]
    feature_costs = config["feature_costs"]
    binary_map = config["feature_binary_map"]
    cost_mode = config.get("cost_mode", "standalone")
    bundles = config.get("feature_bundles", {})

    def _cost(subset):
        if cost_mode == "bundled" and bundles:
            return compute_subset_cost_bundled(subset, feature_costs, bundles)
        return compute_subset_cost(subset, feature_costs)

    # ------------------------------------------------------------------
    # Validation model / variant from config
    # ------------------------------------------------------------------
    val_model_name = config.get("validation_model")
    val_variant_rel = config.get("validation_variant")
    val_params_rel = config.get("validation_params_path")

    if not all([val_model_name, val_variant_rel, val_params_rel]):
        raise ValueError(
            "Config missing validation fields: validation_model, "
            "validation_variant, validation_params_path.  "
            "Add them to the phase3 config file."
        )

    val_variant_path = project_root / val_variant_rel
    val_params_path = project_root / val_params_rel

    if verbose:
        print("Phase 3 Pareto Sweep  |  Mode: --validate")
        print(f"  Validation model:   {val_model_name}")
        print(f"  Validation variant: {val_variant_path.name}")
        print(f"  Cost mode:          {cost_mode}")

    # ------------------------------------------------------------------
    # Load validation data
    # ------------------------------------------------------------------
    val_df = pd.read_csv(val_variant_path)
    val_has_outlier = detect_variant_type(val_df)

    with open(val_params_path, "r") as f:
        val_best_params = json.load(f)["best_params"]

    val_splits = get_group_kfold_splits(val_df)

    if verbose:
        print(f"  Samples:            {len(val_df)}")
        print(f"  Folds:              {len(val_splits)}")
        print(f"  Validation HPs:     {val_best_params}")

    np.random.seed(random_state)
    val_model_class, _, val_default_kwargs = get_model(val_model_name)

    def val_model_factory():
        return val_model_class(
            **val_default_kwargs, random_state=random_state,
            **val_best_params,
        )

    def val_preprocessor_factory(continuous_cols, binary_cols):
        return get_preprocessor(
            features_info=features_info,
            has_outlier_columns=val_has_outlier,
            continuous_subset=continuous_cols,
            binary_subset=binary_cols,
        )

    # ------------------------------------------------------------------
    # Load Pareto frontier: prefer retune, fall back to sweep
    # ------------------------------------------------------------------
    base_dir = project_root / "results" / "phase3_feature_selection"
    retune_dir_path = Path(retune_dir) if retune_dir else base_dir / "pareto_retune"
    sweep_dir_path = Path(sweep_dir) if sweep_dir else base_dir / "pareto_sweep"

    retune_frontier = retune_dir_path / "pareto_frontier_retune.csv"
    sweep_frontier = sweep_dir_path / "pareto_frontier_RMSE_log.csv"

    if retune_frontier.exists():
        frontier_path = retune_frontier
        frontier_source = "retune"
    elif sweep_frontier.exists():
        frontier_path = sweep_frontier
        frontier_source = "sweep"
    else:
        raise FileNotFoundError(
            f"No Pareto frontier found.  Checked:\n"
            f"  {retune_frontier}\n"
            f"  {sweep_frontier}\n"
            "Run --sweep (and optionally --retune) first."
        )

    frontier_df = pd.read_csv(frontier_path)

    if verbose:
        print(f"  Frontier source:    {frontier_source} ({frontier_path.name})")
        print(f"  Pareto subsets:     {len(frontier_df)}")

    # ------------------------------------------------------------------
    # Output directory
    # ------------------------------------------------------------------
    if output_dir is None:
        output_dir = base_dir / "pareto_validation"
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Evaluate each Pareto subset with the validation model
    # ------------------------------------------------------------------
    total_start = time.time()
    val_rows = []

    for row_idx, frow in frontier_df.iterrows():
        subset_id = int(frow["subset_id"])
        subset = _parse_feature_list(frow["features"])
        cost = _cost(subset)

        if verbose:
            print(
                f"  [{row_idx + 1}/{len(frontier_df)}]  "
                f"subset_id={subset_id}  cost={cost}  "
                f"features={frow['features']}"
            )

        fold_metrics = evaluate_subset(
            subset_features=subset,
            df=val_df,
            preprocessor_factory=val_preprocessor_factory,
            model_factory=val_model_factory,
            splits=val_splits,
            always_include=always_include,
            binary_map=binary_map,
        )

        row = _build_result_row(subset_id, subset, cost, fold_metrics)
        val_rows.append(row)

        if verbose:
            print(
                f"    val RMSE_log={row['RMSE_log_mean']:.4f} +/- "
                f"{row['RMSE_log_std']:.4f}"
            )

    total_elapsed = time.time() - total_start

    # ------------------------------------------------------------------
    # Save validation results
    # ------------------------------------------------------------------
    val_results_df = pd.DataFrame(val_rows)
    val_path = output_dir / "validation_results.csv"
    val_results_df.to_csv(val_path, index=False)

    meta = {
        "mode": "validate",
        "validation_model": val_model_name,
        "validation_variant": val_variant_rel,
        "cost_mode": cost_mode,
        "frontier_source": frontier_source,
        "random_state": random_state,
        "n_pareto_subsets": len(frontier_df),
        "elapsed_seconds": total_elapsed,
        "timestamp": datetime.now().isoformat(),
    }
    meta_path = output_dir / "validation_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2, default=_numpy_serializer)

    if verbose:
        print(f"\n{'=' * 60}")
        print(
            f"Validation complete.  {len(frontier_df)} subsets in "
            f"{total_elapsed:.1f}s ({total_elapsed / 60:.1f} min)"
        )
        print(f"Results: {val_path}")
        print(f"{'=' * 60}")

    return meta


# -------------------------------------------------------------------
# CLI
# -------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Phase 3 Pareto Sweep: cost-aware feature selection via "
                    "brute-force subset evaluation.",
    )

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--sweep", action="store_true",
        help="Evaluate all 2^13 feature subsets with fixed HPs.",
    )
    mode.add_argument(
        "--retune", action="store_true",
        help="Re-tune HPs for Pareto-optimal subsets from the sweep.",
    )
    mode.add_argument(
        "--validate", action="store_true",
        help="Re-evaluate Pareto subsets with the validation model.",
    )

    parser.add_argument(
        "-o", "--output_dir",
        type=str, default=None,
        help="Output directory override.",
    )
    parser.add_argument(
        "--config",
        type=str, default=None,
        help="Path to phase3 config file (standalone or bundled).",
    )
    parser.add_argument(
        "--hp_budget",
        type=int, default=None,
        help="HP search budget per subset (retune mode; default from config).",
    )
    parser.add_argument(
        "--sweep_dir",
        type=str, default=None,
        help="Sweep results directory (for retune / validate).",
    )
    parser.add_argument(
        "--retune_dir",
        type=str, default=None,
        help="Retune results directory (for validate).",
    )
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Suppress verbose output.",
    )

    args = parser.parse_args()

    if args.sweep:
        run_sweep(
            output_dir=args.output_dir,
            config_path=args.config,
            verbose=not args.quiet,
        )
    elif args.retune:
        run_retune(
            output_dir=args.output_dir,
            config_path=args.config,
            hp_budget=args.hp_budget,
            sweep_dir=args.sweep_dir,
            verbose=not args.quiet,
        )
    elif args.validate:
        run_validate(
            output_dir=args.output_dir,
            config_path=args.config,
            retune_dir=args.retune_dir,
            sweep_dir=args.sweep_dir,
            verbose=not args.quiet,
        )


if __name__ == "__main__":
    main()
