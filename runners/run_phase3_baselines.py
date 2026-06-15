#!/usr/bin/env python
"""
Phase 3 Selector Baselines — SHAP top-k, correlation filter, and SBS.

Builds baseline feature subsets, then applies the same retune+evaluate protocol
used for Pareto subset retuning so results are directly comparable.

Outputs (to results/phase3_feature_selection/baseline_selectors/):
  - selected_subsets.json
  - shap_topk_results.csv
  - corr_filter_results.csv
  - sbs_results.csv
  - baseline_meta.json
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
    detect_variant_type,
    get_preprocessor,
    load_features_info,
    transform_target,
)
from src.models import get_model
from src.cv_utils import get_group_kfold_splits, compute_fold_summary
from src.feature_selection import (
    load_phase3_config,
    compute_subset_cost,
    compute_subset_cost_bundled,
    resolve_binary_columns,
    evaluate_subset,
)
from src.hp_search import run_hybrid_search
from src.selector_baselines import (
    shap_topk_subsets,
    correlation_filter_subset,
    sbs_subsets,
    DEFAULT_SHAP_K_VALUES,
    DEFAULT_SBS_TARGET_SIZES,
)

RANKING_METRICS = ["RMSE_log", "MAE_log", "R2_log", "RMSE", "MAE", "R2"]
METRIC_DIRECTIONS = {
    "RMSE_log": "minimize",
    "MAE_log": "minimize",
    "R2_log": "maximize",
    "RMSE": "minimize",
    "MAE": "minimize",
    "R2": "maximize",
}


def _numpy_serializer(obj):
    """JSON serializer for numpy types."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


def _parse_feature_list(feat_str):
    """Parse the comma-separated feature string back into a list."""
    if not feat_str or feat_str == "(DEPTH-only)":
        return []
    return feat_str.split(",")


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


def retune_and_evaluate_subset(
    *,
    subset_id: int,
    subset: list[str],
    cost: float,
    df: pd.DataFrame,
    model_name: str,
    model_class,
    param_distributions: dict,
    default_kwargs: dict,
    random_state: int,
    budget: int,
    y_log: np.ndarray,
    groups: np.ndarray,
    features_info: dict,
    has_outlier: bool,
    preprocessor_factory,
    splits: list,
    always_include: dict,
    binary_map: dict,
    output_dir: Path,
    verbose: bool = True,
) -> tuple[dict, dict]:
    """
    Retune HPs for one subset and re-evaluate via LOWO folds.

    This is intentionally identical to the per-subset retune protocol used in
    Phase 3 Pareto retuning so baseline comparisons remain apples-to-apples.
    """
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
    return row, search_results


def _parse_int_list(raw: str) -> list[int]:
    """Parse comma-separated integers, preserving order and uniqueness."""
    values = []
    for token in str(raw).split(","):
        token = token.strip()
        if not token:
            continue
        value = int(token)
        if value not in values:
            values.append(value)
    return values


ALL_SELECTORS = ("shap_topk", "corr_filter", "sbs")


def run_baselines(
    output_dir: str = None,
    config_path: str = None,
    hp_budget: int = None,
    shap_global_path: str = None,
    shap_k: list[int] | None = None,
    corr_method: str = "spearman",
    corr_threshold: float = 0.10,
    corr_top_m: int | None = 6,
    sbs_sizes: list[int] | None = None,
    selectors: list[str] | None = None,
    verbose: bool = True,
) -> dict:
    """
    Build selector baselines and evaluate each via subset-specific retuning.

    ``selectors`` restricts which selector families are built, retuned, and
    written (subset of {shap_topk, corr_filter, sbs}); defaults to all three.
    Running a single selector writes only its ``{selector}_results.csv`` so
    the families can be parallelized across SLURM nodes without clobbering one
    another. Shared metadata files (``selected_subsets``/``baseline_meta``) are
    suffixed with the selector set when not running the full trio.
    """
    if selectors is None:
        selectors = list(ALL_SELECTORS)
    else:
        unknown = [s for s in selectors if s not in ALL_SELECTORS]
        if unknown:
            raise ValueError(
                f"Unknown selector(s): {unknown}. Valid: {list(ALL_SELECTORS)}"
            )
        # Preserve canonical ordering regardless of CLI order.
        selectors = [s for s in ALL_SELECTORS if s in set(selectors)]
    selector_set = set(selectors)
    project_root = Path(__file__).parent.parent
    config = load_phase3_config(config_path)
    features_info = load_features_info()

    model_name = config["best_model"]
    variant_path = project_root / config["best_variant"]
    params_path = project_root / config["best_params_path"]
    random_state = config["random_state"]
    budget = hp_budget or config.get("retune_hp_budget", 1000)
    sweep_features = config["sweep_features"]
    always_include = config["always_include"]
    binary_map = config["feature_binary_map"]
    feature_costs = config["feature_costs"]
    cost_mode = config.get("cost_mode", "standalone")
    bundles = config.get("feature_bundles", {})

    shap_k_values = shap_k or list(DEFAULT_SHAP_K_VALUES)
    sbs_target_sizes = sbs_sizes or list(DEFAULT_SBS_TARGET_SIZES)
    corr_top_m = corr_top_m if (corr_top_m is not None and corr_top_m > 0) else None

    def _cost(subset):
        if cost_mode == "bundled" and bundles:
            return compute_subset_cost_bundled(subset, feature_costs, bundles)
        return compute_subset_cost(subset, feature_costs)

    # ------------------------------------------------------------------
    # Load data and model definitions
    # ------------------------------------------------------------------
    df = pd.read_csv(variant_path)
    has_outlier = detect_variant_type(df)
    target_col = features_info.get("target_label", "CKHL_SM")
    y_log = transform_target(df[target_col].values)
    groups = df["Source"].values
    splits = get_group_kfold_splits(df)

    with open(params_path, "r") as f:
        best_params = json.load(f)["best_params"]

    np.random.seed(random_state)
    model_class, param_distributions, default_kwargs = get_model(model_name)

    def preprocessor_factory(continuous_cols, binary_cols):
        return get_preprocessor(
            features_info=features_info,
            has_outlier_columns=has_outlier,
            continuous_subset=continuous_cols,
            binary_subset=binary_cols,
        )

    def sbs_model_factory():
        return model_class(
            **default_kwargs, random_state=random_state, **best_params,
        )

    # ------------------------------------------------------------------
    # Build baseline subsets
    # ------------------------------------------------------------------
    shap_rows = []
    if "shap_topk" in selector_set:
        shap_rows = shap_topk_subsets(
            sweep_features=sweep_features,
            feature_costs=feature_costs,
            shap_global_path=shap_global_path,
            k_values=shap_k_values,
            cost_mode=cost_mode,
            feature_bundles=bundles,
        )
    corr_rows = []
    if "corr_filter" in selector_set:
        corr_rows = correlation_filter_subset(
            df=df,
            sweep_features=sweep_features,
            feature_costs=feature_costs,
            target_col=target_col,
            corr_method=corr_method,
            threshold=corr_threshold,
            top_m=corr_top_m,
            cost_mode=cost_mode,
            feature_bundles=bundles,
        )
    sbs_rows = []
    if "sbs" in selector_set:
        sbs_rows = sbs_subsets(
            df=df,
            sweep_features=sweep_features,
            preprocessor_factory=preprocessor_factory,
            model_factory=sbs_model_factory,
            splits=splits,
            always_include=always_include,
            binary_map=binary_map,
            feature_costs=feature_costs,
            feature_bundles=bundles,
            cost_mode=cost_mode,
            target_col=target_col,
            target_sizes=sbs_target_sizes,
        )

    selected_by_selector = {
        sel: rows
        for sel, rows in (
            ("shap_topk", shap_rows),
            ("corr_filter", corr_rows),
            ("sbs", sbs_rows),
        )
        if sel in selector_set
    }
    selected_rows = shap_rows + corr_rows + sbs_rows

    if verbose:
        print("Phase 3 Selector Baselines")
        print(f"  Model:          {model_name}")
        print(f"  Variant:        {variant_path.name}")
        print(f"  Cost mode:      {cost_mode}")
        print(f"  HP budget:      {budget}")
        print(f"  Samples:        {len(df)}")
        print(f"  Folds:          {len(splits)}")
        print(f"  Selectors:      {selectors}")
        print(
            "  Selected sets:  "
            f"shap_topk={len(shap_rows)}, "
            f"corr_filter={len(corr_rows)}, "
            f"sbs={len(sbs_rows)}"
        )

    # ------------------------------------------------------------------
    # Output directory
    # ------------------------------------------------------------------
    if output_dir is None:
        output_dir = (
            project_root / "results" / "phase3_feature_selection" / "baseline_selectors"
        )
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    params_root = output_dir / "best_params"
    params_root.mkdir(parents=True, exist_ok=True)

    # Suffix shared metadata files when running a partial selector set so that
    # parallel per-selector jobs do not overwrite each other's metadata. The
    # per-selector {selector}_results.csv files are always disjoint.
    is_full_run = selector_set == set(ALL_SELECTORS)
    meta_suffix = "" if is_full_run else "_" + "_".join(selectors)

    selected_path = output_dir / f"selected_subsets{meta_suffix}.json"
    with open(selected_path, "w") as f:
        json.dump(selected_by_selector, f, indent=2, default=_numpy_serializer)

    # ------------------------------------------------------------------
    # Retune + evaluate each selected subset
    # ------------------------------------------------------------------
    total_start = time.time()
    results_by_selector = {sel: [] for sel in selectors}

    base_keys = {
        "selector", "selector_param", "subset_id", "features", "n_features", "cost",
    }
    for idx, sel_row in enumerate(selected_rows):
        selector = str(sel_row["selector"])
        selector_param = str(sel_row["selector_param"])
        subset_id = int(sel_row["subset_id"])
        subset = _parse_feature_list(str(sel_row["features"]))
        cost = _cost(subset)

        if verbose:
            print(
                f"\n  [{idx + 1}/{len(selected_rows)}]  "
                f"{selector} ({selector_param})  subset_id={subset_id}  "
                f"cost={cost}  features={sel_row['features']}"
            )

        params_dir = params_root / selector
        params_dir.mkdir(parents=True, exist_ok=True)

        result_row, _search_results = retune_and_evaluate_subset(
            subset_id=subset_id,
            subset=subset,
            cost=cost,
            df=df,
            model_name=model_name,
            model_class=model_class,
            param_distributions=param_distributions,
            default_kwargs=default_kwargs,
            random_state=random_state,
            budget=budget,
            y_log=y_log,
            groups=groups,
            features_info=features_info,
            has_outlier=has_outlier,
            preprocessor_factory=preprocessor_factory,
            splits=splits,
            always_include=always_include,
            binary_map=binary_map,
            output_dir=params_dir,
            verbose=verbose,
        )
        result_row["selector"] = selector
        result_row["selector_param"] = selector_param
        # Copy selector-specific metadata (e.g. corr_method, sbs_selection_*)
        # without ever clobbering a retuned metric already in result_row.
        for k, v in sel_row.items():
            if k not in base_keys and k not in result_row:
                result_row[k] = v
        results_by_selector[selector].append(result_row)

        if verbose:
            print(
                f"    Retuned RMSE_log: {result_row['RMSE_log_mean']:.4f} +/- "
                f"{result_row['RMSE_log_std']:.4f}"
            )

    total_elapsed = time.time() - total_start

    # ------------------------------------------------------------------
    # Save per-selector CSVs and metadata
    # ------------------------------------------------------------------
    csv_paths = {}
    for selector, rows in results_by_selector.items():
        out_path = output_dir / f"{selector}_results.csv"
        pd.DataFrame(rows).to_csv(out_path, index=False)
        csv_paths[selector] = str(out_path)

    meta = {
        "mode": "phase3_baselines",
        "model": model_name,
        "variant": config["best_variant"],
        "cost_mode": cost_mode,
        "random_state": random_state,
        "hp_budget": budget,
        "selectors": selectors,
        "selector_defaults": {
            "shap_k": shap_k_values,
            "corr_method": corr_method,
            "corr_threshold": corr_threshold,
            "corr_top_m": corr_top_m,
            "sbs_sizes": sbs_target_sizes,
        },
        "counts": {
            "shap_topk": len(shap_rows),
            "corr_filter": len(corr_rows),
            "sbs": len(sbs_rows),
            "total": len(selected_rows),
        },
        "outputs": {
            "selected_subsets": str(selected_path),
            "csvs": csv_paths,
            "best_params_dir": str(params_root),
        },
        "elapsed_seconds": total_elapsed,
        "timestamp": datetime.now().isoformat(),
    }
    meta_path = output_dir / f"baseline_meta{meta_suffix}.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2, default=_numpy_serializer)

    if verbose:
        print(f"\n{'=' * 60}")
        print(
            f"Baseline selector retune complete in {total_elapsed:.1f}s "
            f"({total_elapsed / 60:.1f} min)"
        )
        for selector, path in csv_paths.items():
            print(f"  {selector:11s} -> {path}")
        print(f"  selected_subsets -> {selected_path}")
        print(f"  baseline_meta    -> {meta_path}")
        print(f"{'=' * 60}")

    return meta


def main():
    parser = argparse.ArgumentParser(
        description="Phase 3 selector baselines with subset-specific retuning.",
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
        help="HP search budget per selected subset (default from config).",
    )
    parser.add_argument(
        "--shap_global_path",
        type=str, default=None,
        help="Optional path to global_importance.csv (default phase3 artifact).",
    )
    parser.add_argument(
        "--shap_k",
        type=str, default=",".join(str(v) for v in DEFAULT_SHAP_K_VALUES),
        help="Comma-separated k values for SHAP top-k subsets.",
    )
    parser.add_argument(
        "--corr_method",
        type=str, default="spearman", choices=["spearman", "pearson"],
        help="Correlation method for correlation-filter subsets.",
    )
    parser.add_argument(
        "--corr_threshold",
        type=float, default=0.10,
        help="Absolute-correlation threshold for correlation-filter subset.",
    )
    parser.add_argument(
        "--corr_top_m",
        type=int, default=6,
        help="Top-m ranked-correlation subset size (set <=0 to disable).",
    )
    parser.add_argument(
        "--sbs_sizes",
        type=str, default=",".join(str(v) for v in DEFAULT_SBS_TARGET_SIZES),
        help="Comma-separated target subset sizes to record along SBS path.",
    )
    parser.add_argument(
        "--selectors",
        type=str, default=",".join(ALL_SELECTORS),
        help=(
            "Comma-separated selector families to run "
            "(subset of shap_topk,corr_filter,sbs). Each writes its own "
            "{selector}_results.csv, enabling one SLURM job per selector."
        ),
    )
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Suppress verbose output.",
    )
    args = parser.parse_args()

    selectors = [s.strip() for s in str(args.selectors).split(",") if s.strip()]

    run_baselines(
        output_dir=args.output_dir,
        config_path=args.config,
        hp_budget=args.hp_budget,
        shap_global_path=args.shap_global_path,
        shap_k=_parse_int_list(args.shap_k),
        corr_method=args.corr_method,
        corr_threshold=args.corr_threshold,
        corr_top_m=args.corr_top_m,
        sbs_sizes=_parse_int_list(args.sbs_sizes),
        selectors=selectors,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
