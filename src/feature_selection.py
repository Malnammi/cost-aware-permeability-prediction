"""
Feature selection utilities for Phase 3: cost-aware Pareto optimization.

Provides functions for:
- Loading and validating Phase 3 configuration
- Enumerating all 2^13 feature subsets with bitmask IDs
- Computing acquisition cost for each subset
- Resolving binary indicator columns tied to parent features
- Evaluating a feature subset via LOWO cross-validation
- Identifying Pareto-optimal subsets (cost vs performance)
- Computing SHAP values per fold and aggregating importance

Imports from src.preprocessing, src.cv_utils.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone

from src.preprocessing import transform_target, inverse_transform_target
from src.cv_utils import compute_metrics


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def load_phase3_config(path=None):
    """
    Load and validate the Phase 3 configuration file.

    Parameters
    ----------
    path : str or Path, optional
        Path to a phase3 config file (standalone or bundled).  Falls back
        to ``configs/phase3_config_standalone.json`` relative to the
        project root.

    Returns
    -------
    dict
        Validated configuration dictionary.

    Raises
    ------
    FileNotFoundError
        If the config file does not exist.
    ValueError
        If required keys are missing.
    """
    if path is None:
        path = Path(__file__).parent.parent / "configs" / "phase3_config_bundled.json"
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Phase 3 config not found: {path}")

    with open(path, "r") as f:
        config = json.load(f)

    required_keys = [
        "best_variant", "best_model", "best_params_path",
        "feature_costs", "sweep_features", "always_include",
        "feature_binary_map", "random_state",
    ]
    missing = [k for k in required_keys if k not in config]
    if missing:
        raise ValueError(f"Phase 3 config missing required keys: {missing}")

    return config


# ---------------------------------------------------------------------------
# Subset enumeration and cost
# ---------------------------------------------------------------------------

def enumerate_feature_subsets(features):
    """
    Yield all 2^n subsets of *features* with integer bitmask IDs.

    Bitmask 0 corresponds to the empty subset (DEPTH-only baseline when
    DEPTH is in ``always_include``).

    Parameters
    ----------
    features : list of str
        Ordered list of sweep feature names (length *n*).

    Yields
    ------
    (int, list of str)
        ``(bitmask, subset_list)`` for each of the 2^n subsets.
    """
    n = len(features)
    for bitmask in range(1 << n):
        subset = [features[i] for i in range(n) if bitmask & (1 << i)]
        yield bitmask, subset


def compute_subset_cost(subset, cost_map):
    """
    Sum surrogate acquisition costs for a feature subset (standalone mode).

    DEPTH cost (always-include baseline) is added automatically.

    Parameters
    ----------
    subset : list of str
        Sweep feature names in this subset (DEPTH excluded).
    cost_map : dict
        Mapping of feature name -> integer cost score.

    Returns
    -------
    int
        Total cost for the subset (including DEPTH).
    """
    cost = cost_map.get("DEPTH", 0)
    for feat in subset:
        cost += cost_map.get(feat, 0)
    return cost


def compute_subset_cost_bundled(subset, cost_map, bundles):
    """
    Bundle-aware surrogate acquisition cost for a feature subset.

    For features with parent dependencies (defined in *bundles*), the
    marginal cost is used when **all** parents are present in the subset;
    otherwise the standalone cost from *cost_map* is used.

    DEPTH cost (always-include baseline) is added automatically.

    Parameters
    ----------
    subset : list of str
        Sweep feature names in this subset (DEPTH excluded).
    cost_map : dict
        Mapping of feature name -> integer standalone cost score.
    bundles : dict
        Mapping of feature name -> ``{"parents": [...], "marginal_cost": int}``
        for features with parent dependencies.  Features absent from this
        dict always use their standalone cost.

    Returns
    -------
    int
        Total bundle-aware cost for the subset (including DEPTH).
    """
    cost = cost_map.get("DEPTH", 0)
    subset_set = set(subset)
    for feat in subset:
        if feat in bundles:
            parents = bundles[feat]["parents"]
            if all(p in subset_set for p in parents):
                cost += bundles[feat]["marginal_cost"]
            else:
                cost += cost_map.get(feat, 0)
        else:
            cost += cost_map.get(feat, 0)
    return cost


def resolve_binary_columns(subset, binary_map):
    """
    Determine which binary indicator columns accompany a sweep-feature subset.

    When a parent continuous feature is excluded, its associated binary
    columns (e.g. ``CT`` -> ``CT_missing``) are also excluded.

    Parameters
    ----------
    subset : list of str
        Sweep feature names included in the current subset.
    binary_map : dict
        Mapping from feature name -> list of associated binary column names.

    Returns
    -------
    list of str
        Binary column names to include.
    """
    binary_cols = []
    for feat in subset:
        binary_cols.extend(binary_map.get(feat, []))
    return binary_cols


# ---------------------------------------------------------------------------
# Subset evaluation
# ---------------------------------------------------------------------------

def evaluate_subset(
    subset_features,
    df,
    preprocessor_factory,
    model_factory,
    splits,
    always_include,
    binary_map,
    target_col="CKHL_SM",
):
    """
    Train and evaluate a feature subset via LOWO cross-validation.

    Follows the Phase 2 data flow: per-fold preprocessor fit, log-target
    training, clipped predictions, and full metrics computation.

    Parameters
    ----------
    subset_features : list of str
        Sweep feature names to include (excluding always-include features).
    df : pd.DataFrame
        Full dataset with all feature and target columns.
    preprocessor_factory : callable
        ``preprocessor_factory(continuous_cols, binary_cols)`` ->
        unfitted ``ColumnTransformer``.
    model_factory : callable
        ``model_factory()`` -> unfitted scikit-learn estimator.
    splits : list of (ndarray, ndarray, str)
        LOWO splits from ``get_group_kfold_splits``.
    always_include : dict
        ``{"continuous": [...], "categorical": [...], "binary": [...]}``.
    binary_map : dict
        Feature -> binary column mapping (``feature_binary_map``).
    target_col : str
        Name of the target column.

    Returns
    -------
    list of dict
        Per-fold metric dictionaries (compatible with
        ``cv_utils.compute_fold_summary``).
    """
    continuous_cols = list(always_include["continuous"]) + list(subset_features)
    binary_cols = (
        resolve_binary_columns(subset_features, binary_map)
        + list(always_include.get("binary", []))
    )
    categorical_cols = list(always_include["categorical"])

    all_feature_cols = continuous_cols + categorical_cols + binary_cols
    X = df[all_feature_cols]
    y_raw = df[target_col].values
    y_log = transform_target(y_raw)

    preprocessor = preprocessor_factory(continuous_cols, binary_cols)

    fold_metrics = []
    for train_idx, test_idx, _fold_name in splits:
        pre = clone(preprocessor)
        X_train = pre.fit_transform(X.iloc[train_idx])
        X_test = pre.transform(X.iloc[test_idx])

        model = model_factory()
        model.fit(X_train, y_log[train_idx])

        y_pred_log = model.predict(X_test)
        # Match Phase 2: clip only for inverse-transforming to original units.
        y_pred_log_safe = np.clip(y_pred_log, -15, 15)
        y_pred = inverse_transform_target(y_pred_log_safe)

        metrics = compute_metrics(
            y_true=y_raw[test_idx],
            y_pred=y_pred,
            y_true_log=y_log[test_idx],
            y_pred_log=y_pred_log,
        )
        fold_metrics.append(metrics)

    return fold_metrics


# ---------------------------------------------------------------------------
# Pareto frontier
# ---------------------------------------------------------------------------

def identify_pareto_frontier(df, cost_col, perf_col, direction="minimize"):
    """
    Return Pareto-optimal rows from a cost-performance DataFrame.

    A row is Pareto-optimal if no other row achieves both lower (or equal)
    cost **and** better (or equal) performance with at least one strict
    improvement.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain *cost_col* and *perf_col*.
    cost_col : str
        Column name for acquisition cost (always minimised).
    perf_col : str
        Column name for the performance metric.
    direction : {"minimize", "maximize"}
        Whether lower (*minimize*) or higher (*maximize*) *perf_col* values
        are better.

    Returns
    -------
    pd.DataFrame
        Subset of *df* containing only the Pareto-optimal rows, sorted by
        ascending cost.
    """
    if direction not in ("minimize", "maximize"):
        raise ValueError(
            f"direction must be 'minimize' or 'maximize', got '{direction}'"
        )

    ascending_perf = direction == "minimize"
    sorted_df = df.sort_values(
        [cost_col, perf_col], ascending=[True, ascending_perf]
    )

    if direction == "minimize":
        best = float("inf")
        frontier_idx = []
        for idx in sorted_df.index:
            val = sorted_df.at[idx, perf_col]
            if val < best:
                frontier_idx.append(idx)
                best = val
    else:
        best = float("-inf")
        frontier_idx = []
        for idx in sorted_df.index:
            val = sorted_df.at[idx, perf_col]
            if val > best:
                frontier_idx.append(idx)
                best = val

    return df.loc[frontier_idx].sort_values(cost_col).copy()


# ---------------------------------------------------------------------------
# SHAP helpers
# ---------------------------------------------------------------------------

def compute_shap_fold(model, X_train, X_test, feature_names):
    """
    Compute SHAP values for one cross-validation fold.

    Uses ``shap.TreeExplainer`` (exact Tree SHAP algorithm).

    Parameters
    ----------
    model : fitted tree-based estimator
        Must be supported by ``shap.TreeExplainer``.
    X_train : array-like of shape (n_train, n_features)
        Training data (used as background dataset).
    X_test : array-like of shape (n_test, n_features)
        Test data to explain.
    feature_names : list of str
        Feature names after preprocessing (length = n_features).

    Returns
    -------
    shap_values : np.ndarray of shape (n_test, n_features)
        SHAP values for each test sample and feature.
    expected_value : float
        Base value (expected model output over the background set).
    """
    import shap

    explainer = shap.TreeExplainer(model, X_train)
    shap_values = explainer.shap_values(X_test)

    if isinstance(shap_values, list):
        shap_values = shap_values[0]

    expected_value = explainer.expected_value
    if isinstance(expected_value, (list, np.ndarray)):
        expected_value = float(expected_value[0])

    return shap_values, expected_value


def collapse_shap_feature_name(feature_name):
    """
    Map a transformed feature name back to its parent feature name.

    Examples
    --------
    ``continuous__CPOR_SM`` -> ``CPOR_SM``
    ``binary__RT_missing`` -> ``RT_missing``
    ``categorical__Zone_6`` -> ``Zone``
    """
    if "__" in feature_name:
        transformer, remainder = feature_name.split("__", 1)
    else:
        transformer, remainder = "", feature_name

    if transformer == "categorical":
        return remainder.rsplit("_", 1)[0] if "_" in remainder else remainder
    return remainder


def collapse_shap_columns(shap_values, feature_names):
    """
    Collapse transformed SHAP columns back to parent feature names.

    This is primarily used to merge one-hot encoded categorical columns such
    as ``categorical__Zone_6`` and ``categorical__Zone_7`` back into a single
    ``Zone`` contribution by summing SHAP values across those dummy columns.

    Parameters
    ----------
    shap_values : np.ndarray of shape (n_samples, n_transformed_features)
        SHAP values in transformed feature space.
    feature_names : list of str
        Transformed feature names (e.g., from ``get_feature_names_out``).

    Returns
    -------
    collapsed_shap : np.ndarray of shape (n_samples, n_parent_features)
        SHAP values aggregated at parent-feature level.
    collapsed_names : list of str
        Parent feature names aligned with columns in ``collapsed_shap``.
    """
    parent_names = [
        collapse_shap_feature_name(name) for name in feature_names
    ]

    parent_to_idx = {}
    collapsed_names = []
    for parent in parent_names:
        if parent not in parent_to_idx:
            parent_to_idx[parent] = len(collapsed_names)
            collapsed_names.append(parent)

    collapsed_shap = np.zeros((shap_values.shape[0], len(collapsed_names)))
    for col_idx, parent in enumerate(parent_names):
        collapsed_shap[:, parent_to_idx[parent]] += shap_values[:, col_idx]

    return collapsed_shap, collapsed_names


def aggregate_shap_importance(shap_values_list, feature_names):
    """
    Aggregate per-fold SHAP values into global and per-well importance.

    Parameters
    ----------
    shap_values_list : list of dict
        One entry per CV fold, each with keys:

        - ``"shap_values"`` : ndarray of shape (n_test, n_features)
        - ``"fold_name"``   : str, held-out well identifier (e.g. ``"A"``)

    feature_names : list of str
        Feature names matching columns of the SHAP arrays.
        If these are transformed names (e.g., one-hot columns), they are
        collapsed back to parent names before aggregation.

    Returns
    -------
    global_importance : pd.DataFrame
        Columns ``["feature", "mean_abs_shap"]``, sorted descending, at parent
        feature level.
    per_well_importance : pd.DataFrame
        Columns ``["well", "feature", "mean_abs_shap"]``, at parent feature
        level.
    """
    all_shap = np.concatenate(
        [entry["shap_values"] for entry in shap_values_list], axis=0
    )
    collapsed_all_shap, collapsed_names = collapse_shap_columns(
        all_shap, feature_names
    )
    global_mean = np.mean(np.abs(collapsed_all_shap), axis=0)

    global_importance = (
        pd.DataFrame({"feature": collapsed_names, "mean_abs_shap": global_mean})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )

    per_well_records = []
    for entry in shap_values_list:
        fold_shap, fold_names = collapse_shap_columns(
            entry["shap_values"], feature_names
        )
        fold_mean = np.mean(np.abs(fold_shap), axis=0)
        for i, feat in enumerate(fold_names):
            per_well_records.append({
                "well": entry["fold_name"],
                "feature": feat,
                "mean_abs_shap": fold_mean[i],
            })

    per_well_importance = pd.DataFrame(per_well_records)

    return global_importance, per_well_importance
