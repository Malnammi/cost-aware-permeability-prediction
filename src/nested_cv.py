"""
Nested LOWO utilities for Phase 4 generalization validation.

Implements a lightweight nested leave-one-well-out workflow:
- Outer LOWO: hold out one well for unbiased evaluation
- Inner LOWO: tune hyperparameters on remaining wells
- Candidate subsets from Phase 3 are evaluated per outer fold
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.cv_utils import compute_metrics, get_group_kfold_splits
from src.feature_selection import (
    compute_subset_cost,
    compute_subset_cost_bundled,
    resolve_binary_columns,
)
from src.hp_search import run_hybrid_search
from src.models import get_model
from src.preprocessing import (
    detect_variant_type,
    get_preprocessor,
    inverse_transform_target,
    load_features_info,
    transform_target,
)

METRIC_NAMES = ["RMSE", "MAE", "MedAE", "R2", "RMSLE", "RMSE_log", "MAE_log", "R2_log"]


def _numpy_serializer(obj: Any) -> Any:
    """Serialize numpy values for JSON outputs."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


def load_phase4_config(path: str | None = None) -> dict:
    """Load and validate the Phase 4 configuration file."""
    if path is None:
        path = Path(__file__).parent.parent / "configs" / "phase4_config_bundled.json"
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Phase 4 config not found: {path}")

    with open(path, "r") as f:
        config = json.load(f)

    required = [
        "best_variant",
        "best_model",
        "candidate_subsets",
        "always_include",
        "feature_binary_map",
        "feature_costs",
        "random_state",
        "hp_budget",
    ]
    missing = [k for k in required if k not in config]
    if missing:
        raise ValueError(f"Phase 4 config missing required keys: {missing}")

    if not config["candidate_subsets"]:
        raise ValueError("Phase 4 config has no candidate_subsets.")

    return config


def _compute_cost(config: dict, subset_features: list[str]) -> int:
    """Compute subset cost using standalone or bundled mode."""
    mode = config.get("cost_mode", "standalone")
    cost_map = config["feature_costs"]
    bundles = config.get("feature_bundles", {})
    if mode == "bundled" and bundles:
        return int(compute_subset_cost_bundled(subset_features, cost_map, bundles))
    return int(compute_subset_cost(subset_features, cost_map))


def _instantiate_model(model_class: type, default_kwargs: dict, params: dict, random_state: int):
    """Instantiate a model with defaults, tuned params, and reproducible seed."""
    kwargs = dict(default_kwargs or {})
    kwargs.update(params or {})

    try:
        probe_params = model_class().get_params()
        if "random_state" in probe_params:
            kwargs.setdefault("random_state", random_state)
    except Exception:
        # Fall back silently if probing model params fails.
        pass

    return model_class(**kwargs)


def _append_rows(csv_path: Path, rows: list[dict]) -> None:
    """Append row dictionaries to a CSV file."""
    if not rows:
        return
    df = pd.DataFrame(rows)
    header = not csv_path.exists()
    df.to_csv(csv_path, mode="a", header=header, index=False)


def run_nested_lowo(
    config: dict,
    output_dir: str | Path,
    subset_ids: list[int] | None = None,
    outer_folds: list[str] | None = None,
    hp_budget: int | None = None,
    resume: bool = True,
    verbose: bool = True,
) -> dict:
    """
    Run nested LOWO for the configured model and candidate feature subsets.

    Writes incremental artifacts:
    - nested_outer_results.csv
    - selection_trace.csv
    - fold_artifacts/{fold}/{subset_id}.json
    - predictions/{subset_id}/{fold}.csv (per-well actual-vs-predicted)

    Predictions CSVs are written only for fold/subset pairs processed in the
    current invocation. To backfill predictions for already-completed pairs,
    re-run with resume=False.
    """
    project_root = Path(__file__).parent.parent
    features_info = load_features_info()

    variant_path = project_root / config["best_variant"]
    if not variant_path.exists():
        raise FileNotFoundError(f"Variant file not found: {variant_path}")

    df = pd.read_csv(variant_path)
    has_outlier = detect_variant_type(df)

    group_col = config.get("group_column", "Source")
    target_col = config.get("target_column", features_info.get("target_label", "CKHL_SM"))
    model_name = config["best_model"]
    random_state = int(config["random_state"])
    random_fraction = float(config.get("random_search_fraction", 0.5))
    budget = int(hp_budget if hp_budget is not None else config["hp_budget"])
    always_include = config["always_include"]
    binary_map = config["feature_binary_map"]

    model_class, param_distributions, default_kwargs = get_model(model_name)

    unique_wells = sorted(df[group_col].unique())
    fold_seed_index_map = {well: idx for idx, well in enumerate(unique_wells)}
    outer_splits = get_group_kfold_splits(
        df,
        group_col=group_col,
        n_splits=len(unique_wells),
    )
    if outer_folds:
        keep = set(outer_folds)
        outer_splits = [s for s in outer_splits if s[2] in keep]
        if not outer_splits:
            raise ValueError(f"No outer folds matched selection: {sorted(keep)}")

    candidates = config["candidate_subsets"]
    if subset_ids:
        allow = set(int(v) for v in subset_ids)
        candidates = [s for s in candidates if int(s["subset_id"]) in allow]
        if not candidates:
            raise ValueError(f"No candidate_subsets matched IDs: {sorted(allow)}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fold_dir = output_dir / "fold_artifacts"
    fold_dir.mkdir(parents=True, exist_ok=True)
    predictions_dir = output_dir / "predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)
    nested_results_csv = output_dir / "nested_outer_results.csv"
    selection_trace_csv = output_dir / "selection_trace.csv"

    completed_pairs: set[tuple[str, int]] = set()
    if resume and nested_results_csv.exists():
        existing = pd.read_csv(nested_results_csv)
        if {"outer_fold", "subset_id"}.issubset(existing.columns):
            completed_pairs = {
                (str(r["outer_fold"]), int(r["subset_id"]))
                for _, r in existing.iterrows()
            }

    if verbose:
        print("Phase 4 Nested LOWO")
        print(f"  Model:            {model_name}")
        print(f"  Variant:          {variant_path.name}")
        print(f"  Candidate subsets:{len(candidates)}")
        print(f"  Outer folds:      {len(outer_splits)}")
        print(f"  HP budget:        {budget}")
        print(f"  Resume mode:      {resume}")
        print(f"  Output dir:       {output_dir}")

    total_start = datetime.now().timestamp()
    selection_rows = []
    processed = 0
    skipped = 0

    for outer_idx, (train_idx, test_idx, fold_name) in enumerate(outer_splits):
        outer_train_df = df.iloc[train_idx].copy()
        outer_test_df = df.iloc[test_idx].copy()
        fold_seed_index = fold_seed_index_map[fold_name]

        if verbose:
            print(
                f"\n[Outer {outer_idx + 1}/{len(outer_splits)}] "
                f"Held-out well: {fold_name}"
            )

        fold_rows = []
        for subset in candidates:
            subset_id = int(subset["subset_id"])
            subset_label = str(subset.get("label", f"subset_{subset_id}"))
            subset_features = list(subset.get("features", []))
            key = (fold_name, subset_id)

            if key in completed_pairs:
                skipped += 1
                if verbose:
                    print(f"  - skip subset {subset_id} ({subset_label}) [already complete]")
                continue

            continuous_cols = list(always_include["continuous"]) + subset_features
            binary_cols = (
                resolve_binary_columns(subset_features, binary_map)
                + list(always_include.get("binary", []))
            )
            categorical_cols = list(always_include["categorical"])
            all_cols = continuous_cols + categorical_cols + binary_cols
            subset_cost = _compute_cost(config, subset_features)

            inner_X_raw = outer_train_df[all_cols]
            inner_y_raw = outer_train_df[target_col].values
            inner_y_log = transform_target(inner_y_raw)
            inner_groups = outer_train_df[group_col].values
            n_inner_splits = len(np.unique(inner_groups))

            pre_inner = get_preprocessor(
                features_info=features_info,
                has_outlier_columns=has_outlier,
                continuous_subset=continuous_cols,
                binary_subset=binary_cols,
            )

            seed = random_state + (fold_seed_index * 10000) + subset_id
            if verbose:
                print(
                    f"  - subset {subset_id:4d} ({subset_label}), "
                    f"cost={subset_cost}, n_inner_splits={n_inner_splits}"
                )

            search = run_hybrid_search(
                model_name=model_name,
                model_class=model_class,
                param_distributions=param_distributions,
                X=inner_X_raw,
                y=inner_y_log,
                groups=inner_groups,
                budget=budget,
                n_splits=n_inner_splits,
                random_state=seed,
                random_fraction=random_fraction,
                default_kwargs=default_kwargs,
                verbose=verbose,
                preprocessor=pre_inner,
            )

            best_params = search["best_params"]
            best_inner_metrics = search.get("best_metrics", {})

            pre_outer = get_preprocessor(
                features_info=features_info,
                has_outlier_columns=has_outlier,
                continuous_subset=continuous_cols,
                binary_subset=binary_cols,
            )
            outer_X_train = pre_outer.fit_transform(inner_X_raw)
            outer_X_test = pre_outer.transform(outer_test_df[all_cols])

            outer_y_train_log = transform_target(outer_train_df[target_col].values)
            outer_y_test_raw = outer_test_df[target_col].values
            outer_y_test_log = transform_target(outer_y_test_raw)

            model = _instantiate_model(
                model_class=model_class,
                default_kwargs=default_kwargs,
                params=best_params,
                random_state=seed,
            )
            model.fit(outer_X_train, outer_y_train_log)

            outer_pred_log = model.predict(outer_X_test)
            # Match Phase 2: clip only for inverse-transforming to original units.
            outer_pred_log_safe = np.clip(outer_pred_log, -15, 15)
            outer_pred_raw = inverse_transform_target(outer_pred_log_safe)
            outer_metrics = compute_metrics(
                y_true=outer_y_test_raw,
                y_pred=outer_pred_raw,
                y_true_log=outer_y_test_log,
                y_pred_log=outer_pred_log,
            )

            # Persist per-well actual-vs-predicted records for downstream visualization.
            # Layout: predictions/{subset_id}/{fold_name}.csv. One CSV per held-out well so
            # users can re-plot without re-running the nested LOWO sweep.
            subset_pred_dir = predictions_dir / str(subset_id)
            subset_pred_dir.mkdir(parents=True, exist_ok=True)
            pred_record = {
                "outer_fold": [fold_name] * len(outer_y_test_raw),
                "row_index": outer_test_df.index.to_numpy(),
                "y_true_raw": outer_y_test_raw,
                "y_pred_raw": outer_pred_raw,
                "y_true_log": outer_y_test_log,
                "y_pred_log": outer_pred_log,
            }
            if "DEPTH" in outer_test_df.columns:
                pred_record["DEPTH"] = outer_test_df["DEPTH"].to_numpy()
            pd.DataFrame(pred_record).to_csv(
                subset_pred_dir / f"{fold_name}.csv", index=False
            )

            row = {
                "outer_fold": fold_name,
                "outer_fold_index": outer_idx,
                "subset_id": subset_id,
                "subset_label": subset_label,
                "features": ",".join(subset_features) if subset_features else "(DEPTH-only)",
                "n_features": len(subset_features),
                "cost": subset_cost,
                "model": model_name,
                "variant": config["best_variant"],
                "hp_budget": budget,
                "inner_best_score": search.get("best_score"),
                "inner_best_search_type": search.get("best_search_type"),
                "best_params_json": json.dumps(best_params, default=_numpy_serializer),
                "timestamp": datetime.now().isoformat(),
            }
            for metric in METRIC_NAMES:
                row[f"outer_{metric}"] = outer_metrics.get(metric)
                row[f"inner_{metric}"] = best_inner_metrics.get(metric)
            fold_rows.append(row)

            artifact_dir = fold_dir / fold_name
            artifact_dir.mkdir(parents=True, exist_ok=True)
            artifact_path = artifact_dir / f"{subset_id}.json"
            with open(artifact_path, "w") as f:
                json.dump(
                    {
                        "outer_fold": fold_name,
                        "subset_id": subset_id,
                        "subset_label": subset_label,
                        "features": subset_features,
                        "cost": subset_cost,
                        "model": model_name,
                        "variant": config["best_variant"],
                        "hp_budget": budget,
                        "search_timing": search.get("timing", {}),
                        "inner_best_score": search.get("best_score"),
                        "inner_best_metrics": best_inner_metrics,
                        "outer_metrics": outer_metrics,
                        "best_params": best_params,
                        "timestamp": datetime.now().isoformat(),
                    },
                    f,
                    indent=2,
                    default=_numpy_serializer,
                )

            processed += 1

        _append_rows(nested_results_csv, fold_rows)

        if fold_rows:
            winner = max(fold_rows, key=lambda r: float(r["inner_best_score"]))
            selection_rows.append(
                {
                    "outer_fold": fold_name,
                    "selected_subset_id": winner["subset_id"],
                    "selected_subset_label": winner["subset_label"],
                    "selected_features": winner["features"],
                    "selected_cost": winner["cost"],
                    "selected_inner_best_score": winner["inner_best_score"],
                    "selected_outer_RMSE_log": winner["outer_RMSE_log"],
                    "selected_outer_MAE_log": winner["outer_MAE_log"],
                    "selected_outer_R2_log": winner["outer_R2_log"],
                    "timestamp": datetime.now().isoformat(),
                }
            )

    _append_rows(selection_trace_csv, selection_rows)

    elapsed = datetime.now().timestamp() - total_start
    meta = {
        "mode": "nested_lowo",
        "model": model_name,
        "variant": config["best_variant"],
        "n_outer_folds": len(outer_splits),
        "n_candidate_subsets": len(candidates),
        "hp_budget": budget,
        "processed_fold_subset_pairs": processed,
        "skipped_fold_subset_pairs": skipped,
        "elapsed_seconds": elapsed,
        "nested_results_csv": str(nested_results_csv),
        "selection_trace_csv": str(selection_trace_csv),
        "timestamp": datetime.now().isoformat(),
    }
    with open(output_dir / "run_meta.json", "w") as f:
        json.dump(meta, f, indent=2, default=_numpy_serializer)

    if verbose:
        print(f"\nRun complete. Processed={processed}, skipped={skipped}")
        print(f"Results: {nested_results_csv}")
        print(f"Trace:   {selection_trace_csv}")

    return meta
