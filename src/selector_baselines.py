"""
Baseline selector utilities for Phase 3 comparisons.

This module adds three selector alternatives on top of the existing
LOWO evaluation machinery:
  - SHAP top-k subsets from cached global importance ranking
  - Correlation-filter subsets from the training frame
  - Sequential backward selection (SBS) under LOWO
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import pandas as pd

from src.cv_utils import compute_fold_summary
from src.feature_selection import (
    compute_subset_cost,
    compute_subset_cost_bundled,
    evaluate_subset,
)
from src.preprocessing import transform_target


DEFAULT_SHAP_K_VALUES = (1, 2, 3, 6, 9, 12)
DEFAULT_SBS_TARGET_SIZES = (1, 2, 3, 6, 9, 12)


def _subset_id(subset: Sequence[str], sweep_features: Sequence[str]) -> int:
    """Encode a subset as a sweep bitmask using sweep_features order."""
    sweep_index = {feat: idx for idx, feat in enumerate(sweep_features)}
    subset_set = set(subset)
    unknown = sorted(subset_set.difference(sweep_index))
    if unknown:
        raise ValueError(f"Subset contains unknown sweep features: {unknown}")

    bitmask = 0
    for feat in subset_set:
        bitmask |= 1 << sweep_index[feat]
    return bitmask


def _canonical_subset(
    subset: Sequence[str],
    sweep_features: Sequence[str],
) -> list[str]:
    """Return subset deduplicated and ordered by sweep feature order."""
    subset_set = set(subset)
    return [feat for feat in sweep_features if feat in subset_set]


def _subset_cost(
    subset: Sequence[str],
    feature_costs: Mapping[str, float],
    cost_mode: str = "standalone",
    feature_bundles: Mapping[str, Mapping[str, object]] | None = None,
) -> float:
    """Compute subset acquisition cost with optional bundle mode."""
    if cost_mode == "bundled" and feature_bundles:
        return float(
            compute_subset_cost_bundled(
                list(subset),
                dict(feature_costs),
                dict(feature_bundles),
            )
        )
    return float(compute_subset_cost(list(subset), dict(feature_costs)))


def _format_features(subset: Sequence[str]) -> str:
    """Match existing Phase 3 text formatting for subset feature lists."""
    if not subset:
        return "(DEPTH-only)"
    return ",".join(subset)


def _build_selector_row(
    *,
    selector: str,
    selector_param: str,
    subset: Sequence[str],
    sweep_features: Sequence[str],
    feature_costs: Mapping[str, float],
    cost_mode: str = "standalone",
    feature_bundles: Mapping[str, Mapping[str, object]] | None = None,
    extra: Mapping[str, object] | None = None,
) -> dict:
    """Build a normalized subset metadata row for selector outputs."""
    canonical = _canonical_subset(subset, sweep_features)
    row = {
        "selector": selector,
        "selector_param": selector_param,
        "subset_id": _subset_id(canonical, sweep_features),
        "features": _format_features(canonical),
        "n_features": len(canonical),
        "cost": _subset_cost(
            canonical,
            feature_costs=feature_costs,
            cost_mode=cost_mode,
            feature_bundles=feature_bundles,
        ),
    }
    if extra:
        row.update(dict(extra))
    return row


def _normalize_sizes(
    sizes: Sequence[int],
    *,
    max_size: int,
) -> list[int]:
    """Deduplicate/validate subset sizes while preserving input order."""
    normalized: list[int] = []
    for raw in sizes:
        size = int(raw)
        if size < 1 or size > max_size:
            continue
        if size not in normalized:
            normalized.append(size)
    return normalized


def _metric_direction(metric: str) -> str:
    """Infer optimization direction from metric name."""
    return "maximize" if metric.startswith("R2") else "minimize"


def _is_better(
    *,
    candidate_score: float,
    best_score: float | None,
    direction: str,
    candidate_cost: float,
    best_cost: float | None,
    tol: float = 1e-12,
) -> bool:
    """Compare two candidate scores with deterministic tie-breaking."""
    if best_score is None:
        return True

    if direction == "minimize":
        if candidate_score < best_score - tol:
            return True
        if abs(candidate_score - best_score) <= tol:
            return best_cost is None or candidate_cost < (best_cost - tol)
        return False

    if candidate_score > best_score + tol:
        return True
    if abs(candidate_score - best_score) <= tol:
        return best_cost is None or candidate_cost < (best_cost - tol)
    return False


def _default_shap_global_path() -> Path:
    """Return the canonical SHAP global-importance CSV path."""
    return (
        Path(__file__).resolve().parent.parent
        / "results"
        / "phase3_feature_selection"
        / "shap"
        / "global_importance.csv"
    )


def shap_topk_subsets(
    sweep_features: Sequence[str],
    feature_costs: Mapping[str, float],
    *,
    shap_global_path: str | Path | None = None,
    k_values: Sequence[int] = DEFAULT_SHAP_K_VALUES,
    cost_mode: str = "standalone",
    feature_bundles: Mapping[str, Mapping[str, object]] | None = None,
) -> list[dict]:
    """
    Build SHAP top-k subsets from cached global feature ranking.

    Notes
    -----
    The ranking is filtered to ``sweep_features`` only and any sweep feature
    absent from the SHAP file is appended in sweep order to keep deterministic
    behavior for all requested k values.
    """
    shap_path = Path(shap_global_path) if shap_global_path else _default_shap_global_path()
    if not shap_path.exists():
        raise FileNotFoundError(f"SHAP global importance not found: {shap_path}")

    shap_df = pd.read_csv(shap_path)
    required_cols = {"feature", "mean_abs_shap"}
    if not required_cols.issubset(shap_df.columns):
        raise ValueError(
            f"SHAP file missing required columns {sorted(required_cols)}: {shap_path}"
        )

    sweep_set = set(sweep_features)
    ranked: list[str] = []
    sorted_df = shap_df.sort_values("mean_abs_shap", ascending=False)
    for feat in sorted_df["feature"].astype(str):
        if feat in sweep_set and feat not in ranked:
            ranked.append(feat)
    for feat in sweep_features:
        if feat not in ranked:
            ranked.append(feat)

    normalized_k = _normalize_sizes(k_values, max_size=len(sweep_features))
    rows = []
    for k in normalized_k:
        chosen = ranked[:k]
        rows.append(
            _build_selector_row(
                selector="shap_topk",
                selector_param=f"k={k}",
                subset=chosen,
                sweep_features=sweep_features,
                feature_costs=feature_costs,
                cost_mode=cost_mode,
                feature_bundles=feature_bundles,
            )
        )
    return rows


def correlation_filter_subset(
    df: pd.DataFrame,
    sweep_features: Sequence[str],
    feature_costs: Mapping[str, float],
    *,
    target_col: str = "CKHL_SM",
    corr_method: str = "spearman",
    threshold: float = 0.10,
    top_m: int | None = 6,
    cost_mode: str = "standalone",
    feature_bundles: Mapping[str, Mapping[str, object]] | None = None,
) -> list[dict]:
    """
    Select subsets from absolute feature-target correlation on log10 target.

    Returns one or two subsets:
      - threshold-based: all features with ``abs(corr) >= threshold``
      - top-m fallback/companion: strongest ``top_m`` correlated features
    """
    if corr_method not in {"pearson", "spearman"}:
        raise ValueError("corr_method must be either 'pearson' or 'spearman'")
    if target_col not in df.columns:
        raise ValueError(f"Target column not found in DataFrame: {target_col}")

    missing = [feat for feat in sweep_features if feat not in df.columns]
    if missing:
        raise ValueError(f"Sweep features missing in DataFrame: {missing}")

    y_log = transform_target(df[target_col].to_numpy())
    y_series = pd.Series(y_log)

    corr_rows: list[dict] = []
    for idx, feat in enumerate(sweep_features):
        x = pd.to_numeric(df[feat], errors="coerce")
        corr = x.corr(y_series, method=corr_method)
        if pd.isna(corr):
            corr = 0.0
        corr_rows.append(
            {
                "feature": feat,
                "abs_corr": float(abs(corr)),
                "corr": float(corr),
                "sweep_idx": idx,
            }
        )

    corr_df = pd.DataFrame(corr_rows).sort_values(
        ["abs_corr", "sweep_idx"],
        ascending=[False, True],
    )
    ranked = corr_df["feature"].tolist()

    threshold_feats = corr_df.loc[corr_df["abs_corr"] >= float(threshold), "feature"].tolist()
    if not threshold_feats and ranked:
        threshold_feats = [ranked[0]]

    rows = [
        _build_selector_row(
            selector="corr_filter",
            selector_param=f"{corr_method}|abs>={threshold:g}",
            subset=threshold_feats,
            sweep_features=sweep_features,
            feature_costs=feature_costs,
            cost_mode=cost_mode,
            feature_bundles=feature_bundles,
            extra={"corr_method": corr_method},
        )
    ]

    if top_m is not None and int(top_m) > 0:
        top_m_feats = ranked[: int(top_m)]
        top_m_row = _build_selector_row(
            selector="corr_filter",
            selector_param=f"{corr_method}|top_m={int(top_m)}",
            subset=top_m_feats,
            sweep_features=sweep_features,
            feature_costs=feature_costs,
            cost_mode=cost_mode,
            feature_bundles=feature_bundles,
            extra={"corr_method": corr_method},
        )
        if top_m_row["subset_id"] != rows[0]["subset_id"]:
            rows.append(top_m_row)

    return rows


def sbs_subsets(
    df: pd.DataFrame,
    sweep_features: Sequence[str],
    *,
    preprocessor_factory,
    model_factory,
    splits,
    always_include: Mapping[str, Sequence[str]],
    binary_map: Mapping[str, Sequence[str]],
    feature_costs: Mapping[str, float],
    feature_bundles: Mapping[str, Mapping[str, object]] | None = None,
    cost_mode: str = "standalone",
    target_col: str = "CKHL_SM",
    target_sizes: Sequence[int] = DEFAULT_SBS_TARGET_SIZES,
    score_metric: str = "RMSE_log_mean",
) -> list[dict]:
    """
    Sequential backward selection under LOWO, recording requested subset sizes.

    Starting from the full sweep-feature set, each SBS step removes one
    feature whose removal gives the best LOWO score on ``score_metric``.
    """
    n_total = len(sweep_features)
    normalized_sizes = _normalize_sizes(target_sizes, max_size=n_total)
    if not normalized_sizes:
        raise ValueError("No valid target_sizes provided for SBS")

    direction = _metric_direction(score_metric)
    min_target = min(normalized_sizes)
    size_targets = set(normalized_sizes)

    current = list(sweep_features)
    selected_by_size: dict[int, dict] = {}

    while len(current) >= min_target:
        if len(current) in size_targets and len(current) not in selected_by_size:
            fold_metrics = evaluate_subset(
                subset_features=current,
                df=df,
                preprocessor_factory=preprocessor_factory,
                model_factory=model_factory,
                splits=splits,
                always_include=always_include,
                binary_map=binary_map,
                target_col=target_col,
            )
            summary = compute_fold_summary(fold_metrics)
            if score_metric not in summary:
                raise ValueError(
                    f"score_metric '{score_metric}' not found in fold summary: "
                    f"{sorted(summary.keys())}"
                )
            selected_by_size[len(current)] = _build_selector_row(
                selector="sbs",
                selector_param=f"n_features={len(current)}",
                subset=current,
                sweep_features=sweep_features,
                feature_costs=feature_costs,
                cost_mode=cost_mode,
                feature_bundles=feature_bundles,
                extra={f"sbs_selection_{score_metric}": float(summary[score_metric])},
            )

        if len(current) == min_target:
            break

        best_score = None
        best_cost = None
        best_subset = None
        for remove_idx, _removed_feature in enumerate(current):
            candidate = current[:remove_idx] + current[remove_idx + 1:]
            fold_metrics = evaluate_subset(
                subset_features=candidate,
                df=df,
                preprocessor_factory=preprocessor_factory,
                model_factory=model_factory,
                splits=splits,
                always_include=always_include,
                binary_map=binary_map,
                target_col=target_col,
            )
            summary = compute_fold_summary(fold_metrics)
            if score_metric not in summary:
                raise ValueError(
                    f"score_metric '{score_metric}' not found in fold summary: "
                    f"{sorted(summary.keys())}"
                )

            candidate_score = float(summary[score_metric])
            candidate_cost = _subset_cost(
                candidate,
                feature_costs=feature_costs,
                cost_mode=cost_mode,
                feature_bundles=feature_bundles,
            )
            better = _is_better(
                candidate_score=candidate_score,
                best_score=best_score,
                direction=direction,
                candidate_cost=candidate_cost,
                best_cost=best_cost,
            )
            if better:
                best_score = candidate_score
                best_cost = candidate_cost
                best_subset = candidate

        if best_subset is None:
            raise RuntimeError("SBS failed to select the next subset")

        current = best_subset

    return [selected_by_size[size] for size in normalized_sizes if size in selected_by_size]
