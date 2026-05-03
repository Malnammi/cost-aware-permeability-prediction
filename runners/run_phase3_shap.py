#!/usr/bin/env python
"""
Phase 3 SHAP Analysis — Feature Importance via TreeSHAP

Computes SHAP values for the Phase 2 winning model (ExtraTrees on
without-outlier adaptive) across all 7 LOWO folds, then aggregates
into global and per-well importance rankings.

Outputs (to results/phase3_feature_selection/shap/):
  - shap_values.npz       : raw SHAP arrays + sample metadata
  - global_importance.csv  : features ranked by mean |SHAP|
  - per_well_importance.csv: per-well mean |SHAP| per feature
  - fold_metrics.csv       : sanity-check CV metrics (should match Phase 2)
  - shap_run_meta.json     : run metadata and timing

Usage:
    python runners/run_phase3_shap.py [-o OUTPUT_DIR] [--quiet]
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
from sklearn.base import clone

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.preprocessing import (
    load_features_info,
    prepare_data,
    transform_target,
    inverse_transform_target,
)
from src.models import get_model
from src.cv_utils import get_group_kfold_splits, compute_metrics
from src.feature_selection import (
    load_phase3_config,
    compute_shap_fold,
    aggregate_shap_importance,
)


def run_shap_analysis(
    output_dir: str = None,
    config_path: str = None,
    verbose: bool = True,
) -> dict:
    """
    Run SHAP analysis across all LOWO folds for the Phase 2 winner.

    Parameters
    ----------
    output_dir : str, optional
        Output directory. Defaults to results/phase3_feature_selection/shap.
    config_path : str, optional
        Path to a phase3 config file (standalone or bundled).
    verbose : bool
        Print progress.

    Returns
    -------
    dict
        Run metadata including paths to saved outputs and timing.
    """
    project_root = Path(__file__).parent.parent

    # ------------------------------------------------------------------
    # Load configuration
    # ------------------------------------------------------------------
    config = load_phase3_config(config_path)
    features_info = load_features_info()

    model_name = config["best_model"]
    variant_path = project_root / config["best_variant"]
    params_path = project_root / config["best_params_path"]
    random_state = config["random_state"]

    if verbose:
        print("Phase 3 SHAP Analysis")
        print(f"  Model:        {model_name}")
        print(f"  Variant:      {variant_path.name}")
        print(f"  Params file:  {params_path.name}")
        print(f"  Random state: {random_state}")

    # ------------------------------------------------------------------
    # Load dataset and best hyperparameters
    # ------------------------------------------------------------------
    df = pd.read_csv(variant_path)

    with open(params_path, "r") as f:
        best_params = json.load(f)["best_params"]

    if verbose:
        print(f"  Samples:      {len(df)}")
        print(f"  Best HPs:     {best_params}")

    # ------------------------------------------------------------------
    # Prepare data (full feature set, identical to Phase 2)
    # ------------------------------------------------------------------
    X, y_raw_series, preprocessor = prepare_data(df, features_info)
    y_raw = y_raw_series.values
    y_log = transform_target(y_raw)
    splits = get_group_kfold_splits(df)

    if verbose:
        print(f"  X shape:      {X.shape}")
        print(f"  Folds:        {len(splits)}")

    # ------------------------------------------------------------------
    # Model factory
    # ------------------------------------------------------------------
    model_class, _, default_kwargs = get_model(model_name)

    np.random.seed(random_state)

    def make_model():
        return model_class(
            **default_kwargs,
            random_state=random_state,
            **best_params,
        )

    # ------------------------------------------------------------------
    # SHAP computation across LOWO folds
    # ------------------------------------------------------------------
    shap_entries = []
    fold_expected_values = []
    fold_metrics_list = []
    feature_names = None

    total_start = time.time()

    for fold_idx, (train_idx, test_idx, fold_name) in enumerate(splits):
        fold_start = time.time()
        if verbose:
            print(f"\n  Fold {fold_idx + 1}/{len(splits)}  "
                  f"(held-out well: {fold_name}, "
                  f"n_test={len(test_idx)})")

        pre = clone(preprocessor)
        X_train = pre.fit_transform(X.iloc[train_idx])
        X_test = pre.transform(X.iloc[test_idx])

        if feature_names is None:
            feature_names = list(pre.get_feature_names_out())

        model = make_model()
        model.fit(X_train, y_log[train_idx])

        # Predictions (sanity-check: should reproduce Phase 2 CV scores)
        y_pred_log = np.clip(model.predict(X_test), -15, 15)
        y_pred = inverse_transform_target(y_pred_log)
        metrics = compute_metrics(
            y_raw[test_idx], y_pred, y_log[test_idx], y_pred_log,
        )
        fold_metrics_list.append({"fold": fold_name, **metrics})

        # SHAP values via TreeExplainer
        shap_vals, expected_val = compute_shap_fold(
            model, X_train, X_test, feature_names,
        )

        shap_entries.append({
            "shap_values": shap_vals,
            "fold_name": fold_name,
            "test_indices": test_idx,
        })
        fold_expected_values.append(expected_val)

        fold_elapsed = time.time() - fold_start
        if verbose:
            print(f"    RMSE_log={metrics['RMSE_log']:.4f}  "
                  f"R2_log={metrics['R2_log']:.4f}  "
                  f"SHAP shape={shap_vals.shape}  "
                  f"({fold_elapsed:.1f}s)")

    total_elapsed = time.time() - total_start

    # ------------------------------------------------------------------
    # Aggregate SHAP importance
    # ------------------------------------------------------------------
    global_imp, per_well_imp = aggregate_shap_importance(
        shap_entries, feature_names,
    )

    # ------------------------------------------------------------------
    # Save outputs
    # ------------------------------------------------------------------
    if output_dir is None:
        output_dir = (
            project_root / "results" / "phase3_feature_selection" / "shap"
        )
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Raw SHAP values + metadata
    all_shap = np.concatenate(
        [e["shap_values"] for e in shap_entries], axis=0,
    )
    all_indices = np.concatenate(
        [e["test_indices"] for e in shap_entries],
    )
    fold_labels = np.concatenate([
        np.full(len(e["test_indices"]), e["fold_name"])
        for e in shap_entries
    ])

    npz_path = output_dir / "shap_values.npz"
    np.savez(
        npz_path,
        shap_values=all_shap,
        feature_names=np.array(feature_names),
        sample_indices=all_indices,
        fold_labels=fold_labels,
        expected_values=np.array(fold_expected_values),
    )

    global_path = output_dir / "global_importance.csv"
    global_imp.to_csv(global_path, index=False)

    per_well_path = output_dir / "per_well_importance.csv"
    per_well_imp.to_csv(per_well_path, index=False)

    metrics_df = pd.DataFrame(fold_metrics_list)
    metrics_path = output_dir / "fold_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)

    meta = {
        "model": model_name,
        "variant": config["best_variant"],
        "best_params": best_params,
        "random_state": random_state,
        "n_folds": len(splits),
        "n_samples": len(df),
        "n_features_raw": X.shape[1],
        "n_features_transformed": len(feature_names),
        "feature_names": feature_names,
        "elapsed_seconds": total_elapsed,
        "outputs": {
            "shap_values": str(npz_path),
            "global_importance": str(global_path),
            "per_well_importance": str(per_well_path),
            "fold_metrics": str(metrics_path),
        },
        "timestamp": datetime.now().isoformat(),
    }
    meta_path = output_dir / "shap_run_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2, default=str)

    if verbose:
        print(f"\n{'=' * 60}")
        print(f"SHAP analysis complete.  Wall time: {total_elapsed:.1f}s "
              f"({total_elapsed / 60:.1f} min)")
        print(f"\nOutputs saved to: {output_dir}")
        print(f"  shap_values.npz         ({all_shap.shape})")
        print(f"  global_importance.csv    ({len(global_imp)} features)")
        print(f"  per_well_importance.csv  ({len(per_well_imp)} rows)")
        print(f"  fold_metrics.csv         ({len(metrics_df)} folds)")
        print(f"  shap_run_meta.json")
        print(f"\nTop-5 features by mean |SHAP|:")
        for _, row in global_imp.head(5).iterrows():
            print(f"  {row['feature']:35s}  {row['mean_abs_shap']:.6f}")
        print(f"{'=' * 60}")

    return meta


def main():
    parser = argparse.ArgumentParser(
        description="Phase 3 SHAP Analysis: compute SHAP values for the "
                    "Phase 2 winning model across all LOWO folds.",
    )
    parser.add_argument(
        "-o", "--output_dir",
        type=str,
        default=None,
        help="Output directory (default: results/phase3_feature_selection/shap)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to phase3 config file (default: configs/phase3_config_standalone.json)",
    )
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Suppress verbose output.",
    )

    args = parser.parse_args()
    run_shap_analysis(
        output_dir=args.output_dir,
        config_path=args.config,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
