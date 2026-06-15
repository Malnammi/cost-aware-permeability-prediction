#!/usr/bin/env python
"""
Phase 4 Petrophysical/Regression Baselines - Outer LOWO runner.

Baselines:
  - Log-linear porosity-permeability regression (CPOR_SM)
  - Log-linear porosity-permeability regression (PHIT)
  - Timur-style baseline (CPOR_SM, SWT as Swirr proxy)
  - Timur-style baseline (PHIT, SWT as Swirr proxy)

Outputs (to results/phase4_generalization/baselines/):
  - baseline_outer_results.csv
  - baseline_summary.csv
  - predictions/{slug}/{well}.csv
  - baseline_meta.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.cv_utils import compute_fold_summary, compute_metrics, get_group_kfold_splits
from src.feature_selection import compute_subset_cost, compute_subset_cost_bundled
from src.nested_cv import load_phase4_config
from src.permeability_baseline_models import (
    fit_log_linear_baseline,
    fit_timur_baseline,
    predict_log_linear_baseline,
    predict_timur_baseline,
)
from src.preprocessing import inverse_transform_target, transform_target

METRIC_NAMES = ["RMSE", "MAE", "MedAE", "R2", "RMSLE", "RMSE_log", "MAE_log", "R2_log"]
MINIMIZE_METRICS = {"RMSE", "MAE", "MedAE", "RMSLE", "RMSE_log", "MAE_log"}


def _numpy_serializer(obj: Any) -> Any:
    """JSON serializer for numpy types."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


def _compute_subset_cost(config: dict, subset_features: list[str]) -> int:
    """Compute subset cost in standalone or bundled mode."""
    mode = config.get("cost_mode", "standalone")
    cost_map = config["feature_costs"]
    bundles = config.get("feature_bundles", {})
    if mode == "bundled" and bundles:
        return int(compute_subset_cost_bundled(subset_features, cost_map, bundles))
    return int(compute_subset_cost(subset_features, cost_map))


def _build_summary_row(
    baseline_spec: dict,
    fold_rows: list[dict],
) -> dict:
    """Aggregate per-fold baseline metrics into mean/std/CI summary row."""
    if not fold_rows:
        return {}

    fold_metrics = []
    for row in fold_rows:
        fold_metrics.append({m: row[m] for m in METRIC_NAMES})
    summary = compute_fold_summary(fold_metrics)

    n_folds = len(fold_rows)
    if n_folds > 1:
        t_crit = stats.t.ppf(1 - 0.05 / 2, df=n_folds - 1)
    else:
        t_crit = np.nan

    out = {
        "baseline_slug": baseline_spec["slug"],
        "baseline_label": baseline_spec["label"],
        "equation": baseline_spec["equation"],
        "phi_column": baseline_spec.get("phi_column"),
        "swirr_column": baseline_spec.get("swirr_column"),
        "swirr_proxy_note": baseline_spec.get("swirr_proxy_note", ""),
        "comparison_targets": baseline_spec.get("comparison_targets", ""),
        "cost": baseline_spec["cost"],
        "n_outer_folds": n_folds,
    }
    out.update(summary)

    for metric in METRIC_NAMES:
        mean_col = f"{metric}_mean"
        std_col = f"{metric}_std"
        if mean_col not in out or std_col not in out:
            continue
        if n_folds <= 1 or pd.isna(out[std_col]):
            out[f"{metric}_ci"] = out[mean_col]
            continue
        sem = out[std_col] / np.sqrt(n_folds)
        if metric in MINIMIZE_METRICS:
            out[f"{metric}_ci"] = out[mean_col] + t_crit * sem
        else:
            out[f"{metric}_ci"] = out[mean_col] - t_crit * sem

    return out


def _baseline_specs(config: dict) -> list[dict]:
    """Define all baseline models and metadata in one place."""
    return [
        {
            "slug": "loglinear_cpor_sm",
            "label": "Log-linear porosity-permeability (CPOR_SM)",
            "kind": "log_linear",
            "phi_column": "CPOR_SM",
            "swirr_column": None,
            "equation": "log10(k) = a + b*CPOR_SM",
            "swirr_proxy_note": "",
            "comparison_targets": "4550,4096",
            "cost": _compute_subset_cost(config, ["CPOR_SM"]),
        },
        {
            "slug": "loglinear_phit",
            "label": "Log-linear porosity-permeability (PHIT)",
            "kind": "log_linear",
            "phi_column": "PHIT",
            "swirr_column": None,
            "equation": "log10(k) = a + b*PHIT",
            "swirr_proxy_note": "",
            "comparison_targets": "640",
            "cost": _compute_subset_cost(config, ["PHIT"]),
        },
        {
            "slug": "timur_cpor_sm_swt",
            "label": "Timur baseline (CPOR_SM, SWT as Swirr)",
            "kind": "timur",
            "phi_column": "CPOR_SM",
            "swirr_column": "SWT",
            "equation": "log10(k) = log(a) + b*log10(CPOR_SM) - c*log10(SWT)",
            "swirr_proxy_note": "SWT used as Swirr proxy",
            "comparison_targets": "all",
            "cost": _compute_subset_cost(config, ["CPOR_SM", "SWT"]),
        },
        {
            "slug": "timur_phit_swt",
            "label": "Timur baseline (PHIT, SWT as Swirr)",
            "kind": "timur",
            "phi_column": "PHIT",
            "swirr_column": "SWT",
            "equation": "log10(k) = log(a) + b*log10(PHIT) - c*log10(SWT)",
            "swirr_proxy_note": "SWT used as Swirr proxy",
            "comparison_targets": "640",
            "cost": _compute_subset_cost(config, ["PHIT", "SWT"]),
        },
    ]


def run_phase4_baselines(
    *,
    config_path: str | None = None,
    output_dir: str | Path | None = None,
    verbose: bool = True,
) -> dict:
    """Run outer-LOWO petrophysical baseline evaluations."""
    project_root = Path(__file__).parent.parent
    config = load_phase4_config(config_path)

    variant_path = project_root / config["best_variant"]
    if not variant_path.exists():
        raise FileNotFoundError(f"Variant file not found: {variant_path}")

    df = pd.read_csv(variant_path)
    group_col = config.get("group_column", "Source")
    target_col = config.get("target_column", "CKHL_SM")
    unique_groups = sorted(df[group_col].dropna().unique())
    splits = get_group_kfold_splits(
        df,
        group_col=group_col,
        n_splits=len(unique_groups),
    )

    if output_dir is None:
        output_dir = (
            project_root / "results" / "phase4_generalization" / "baselines"
        )
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pred_root = output_dir / "predictions"
    pred_root.mkdir(parents=True, exist_ok=True)

    specs = _baseline_specs(config)

    if verbose:
        print("Phase 4 Baselines (outer LOWO)")
        print(f"  Variant:       {variant_path.name}")
        print(f"  Group column:  {group_col}")
        print(f"  Target column: {target_col}")
        print(f"  Folds:         {len(splits)}")
        print(f"  Output dir:    {output_dir}")
        print(f"  Baselines:     {len(specs)}")

    total_start = time.time()
    outer_rows: list[dict] = []
    summary_rows: list[dict] = []

    for spec in specs:
        slug = spec["slug"]
        pred_dir = pred_root / slug
        pred_dir.mkdir(parents=True, exist_ok=True)
        fold_rows: list[dict] = []

        if verbose:
            print(f"\n[{slug}] {spec['label']}")

        for fold_idx, (train_idx, test_idx, fold_name) in enumerate(splits):
            train_df = df.iloc[train_idx].copy()
            test_df = df.iloc[test_idx].copy()

            y_train_raw = train_df[target_col].to_numpy()
            y_train_log = transform_target(y_train_raw)
            y_test_raw = test_df[target_col].to_numpy()
            y_test_log = transform_target(y_test_raw)

            if spec["kind"] == "log_linear":
                params = fit_log_linear_baseline(
                    train_df[spec["phi_column"]].to_numpy(),
                    y_train_log,
                    phi_name=spec["phi_column"],
                )
                y_pred_log = predict_log_linear_baseline(
                    test_df[spec["phi_column"]].to_numpy(),
                    params,
                )
            elif spec["kind"] == "timur":
                params = fit_timur_baseline(
                    train_df[spec["phi_column"]].to_numpy(),
                    train_df[spec["swirr_column"]].to_numpy(),
                    y_train_log,
                    phi_name=spec["phi_column"],
                    swirr_name=spec["swirr_column"],
                )
                y_pred_log = predict_timur_baseline(
                    test_df[spec["phi_column"]].to_numpy(),
                    test_df[spec["swirr_column"]].to_numpy(),
                    params,
                )
            else:
                raise ValueError(f"Unsupported baseline kind: {spec['kind']}")

            # Match Phase 2/Phase 4 metric path:
            # clip only for inverse-transform to raw units.
            y_pred_log_safe = np.clip(y_pred_log, -15, 15)
            y_pred_raw = inverse_transform_target(y_pred_log_safe)
            metrics = compute_metrics(
                y_true=y_test_raw,
                y_pred=y_pred_raw,
                y_true_log=y_test_log,
                y_pred_log=y_pred_log,
            )

            pred_record = {
                "baseline_slug": [slug] * len(y_test_raw),
                "baseline_label": [spec["label"]] * len(y_test_raw),
                "outer_fold": [fold_name] * len(y_test_raw),
                "row_index": test_df.index.to_numpy(),
                "Zone": test_df["Zone"].to_numpy() if "Zone" in test_df.columns else np.nan,
                "y_true_raw": y_test_raw,
                "y_pred_raw": y_pred_raw,
                "y_true_log": y_test_log,
                "y_pred_log": y_pred_log,
            }
            if "DEPTH" in test_df.columns:
                pred_record["DEPTH"] = test_df["DEPTH"].to_numpy()
            pd.DataFrame(pred_record).to_csv(
                pred_dir / f"{fold_name}.csv", index=False
            )

            row = {
                "baseline_slug": slug,
                "baseline_label": spec["label"],
                "equation": spec["equation"],
                "phi_column": spec.get("phi_column"),
                "swirr_column": spec.get("swirr_column"),
                "swirr_proxy_note": spec.get("swirr_proxy_note", ""),
                "comparison_targets": spec.get("comparison_targets", ""),
                "cost": spec["cost"],
                "outer_fold": fold_name,
                "outer_fold_index": fold_idx,
                "n_train": int(len(train_idx)),
                "n_test": int(len(test_idx)),
                "params_json": json.dumps(params, default=_numpy_serializer),
                "timestamp": datetime.now().isoformat(),
            }
            row.update(metrics)
            outer_rows.append(row)
            fold_rows.append(row)

            if verbose:
                print(
                    f"  - fold {fold_name}: "
                    f"RMSE_log={metrics['RMSE_log']:.4f}, "
                    f"R2_log={metrics['R2_log']:.4f}"
                )

        summary_row = _build_summary_row(spec, fold_rows)
        if summary_row:
            summary_rows.append(summary_row)

    baseline_outer_path = output_dir / "baseline_outer_results.csv"
    baseline_summary_path = output_dir / "baseline_summary.csv"
    pd.DataFrame(outer_rows).to_csv(baseline_outer_path, index=False)
    pd.DataFrame(summary_rows).sort_values(
        "RMSE_log_mean", ascending=True
    ).to_csv(baseline_summary_path, index=False)

    elapsed = time.time() - total_start
    meta = {
        "mode": "phase4_baselines_outer_lowo",
        "variant": config["best_variant"],
        "group_column": group_col,
        "target_column": target_col,
        "n_rows": int(len(df)),
        "n_outer_folds": int(len(splits)),
        "baselines": [s["slug"] for s in specs],
        "baseline_outer_results_csv": str(baseline_outer_path),
        "baseline_summary_csv": str(baseline_summary_path),
        "predictions_dir": str(pred_root),
        "elapsed_seconds": elapsed,
        "timestamp": datetime.now().isoformat(),
    }
    meta_path = output_dir / "baseline_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2, default=_numpy_serializer)

    if verbose:
        print(f"\n{'=' * 60}")
        print("Phase 4 baseline run complete")
        print(f"  Outer rows: {len(outer_rows)}")
        print(f"  Summary:    {baseline_summary_path}")
        print(f"  Predictions:{pred_root}")
        print(f"  Meta:       {meta_path}")
        print(f"{'=' * 60}")

    return meta


def main():
    parser = argparse.ArgumentParser(
        description="Run Phase 4 petrophysical/regression baselines under outer LOWO."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to phase4 config (default: configs/phase4_config_bundled.json).",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        default=None,
        help="Output directory (default: results/phase4_generalization/baselines).",
    )
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Suppress verbose output.",
    )
    args = parser.parse_args()

    run_phase4_baselines(
        config_path=args.config,
        output_dir=args.output_dir,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
