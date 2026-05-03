"""
Cross-validation utilities for permeability prediction.

Implements GroupKFold (Leave-One-Well-Out) cross-validation and metrics computation.
With 7 wells (A-G), this creates 7 folds where each fold holds out one well for testing.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    r2_score,
)


def get_group_kfold_splits(
    df: pd.DataFrame,
    group_col: str = "Source",
    n_splits: int = 7
) -> list:
    """
    Generate GroupKFold cross-validation splits based on well/source grouping.
    
    This implements Leave-One-Well-Out (LOWO) cross-validation where each fold
    holds out all samples from one well for testing. With 7 wells (A-G), this
    creates 7 folds.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing the group column.
    group_col : str, optional
        Column name to group by. Default is "Source" (well identifier A-G).
    n_splits : int, optional
        Number of folds. Default is 7 (one per well).
        
    Returns
    -------
    list
        List of tuples (train_idx, test_idx, fold_name) where:
        - train_idx: numpy array of training indices
        - test_idx: numpy array of test indices  
        - fold_name: string identifier for the held-out group (e.g., "A", "B", ...)
        
    Raises
    ------
    ValueError
        If group_col not in DataFrame or n_splits exceeds number of unique groups.
        
    Examples
    --------
    >>> splits = get_group_kfold_splits(df, group_col="Source", n_splits=7)
    >>> for train_idx, test_idx, fold_name in splits:
    ...     print(f"Fold {fold_name}: train={len(train_idx)}, test={len(test_idx)}")
    Fold A: train=1950, test=336
    Fold B: train=1800, test=486
    ...
    """
    if group_col not in df.columns:
        raise ValueError(f"Group column '{group_col}' not found in DataFrame. "
                        f"Available columns: {list(df.columns)}")
    
    groups = df[group_col].values
    unique_groups = df[group_col].unique()
    
    if n_splits > len(unique_groups):
        raise ValueError(f"n_splits={n_splits} exceeds number of unique groups "
                        f"({len(unique_groups)}). Groups: {list(unique_groups)}")
    
    # Initialize GroupKFold
    group_kfold = GroupKFold(n_splits=n_splits)
    
    # Generate splits with fold names
    splits = []
    for train_idx, test_idx in group_kfold.split(df, groups=groups):
        # Get the held-out group name(s) for this fold
        test_groups = df.iloc[test_idx][group_col].unique()
        fold_name = "_".join(sorted(test_groups))
        
        splits.append((train_idx, test_idx, fold_name))
    
    # Sort by fold name for reproducibility (A, B, C, D, E, F, G)
    splits.sort(key=lambda x: x[2])
    
    return splits


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_true_log: np.ndarray = None,
    y_pred_log: np.ndarray = None
) -> dict:
    """
    Compute regression metrics for model evaluation.
    
    Computes metrics in both original and log-transformed scales:
    - Original scale metrics are useful for interpretability
    - Log scale metrics are appropriate since model is trained on log-transformed target
    
    Parameters
    ----------
    y_true : array-like
        True target values in original scale (permeability in mD).
    y_pred : array-like
        Predicted values in original scale.
    y_true_log : array-like, optional
        True target values in log10 scale. If None, computed from y_true.
    y_pred_log : array-like, optional
        Predicted values in log10 scale. If None, computed from y_pred.
        
    Returns
    -------
    dict
        Dictionary containing:
        - RMSE: Root Mean Squared Error (original scale)
        - MAE: Mean Absolute Error (original scale)
        - MedAE: Median Absolute Error (original scale)
        - R2: R-squared coefficient (original scale)
        - RMSLE: Root Mean Squared Log Error (uses log10 values)
        - RMSE_log: RMSE in log10 scale
        - MAE_log: MAE in log10 scale
        - R2_log: R2 in log10 scale
        
    Notes
    -----
    RMSLE is computed as RMSE of log10-transformed values, which is appropriate
    for permeability prediction where values span several orders of magnitude.
    This metric penalizes under-predictions and over-predictions equally in
    relative terms.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    # Compute log values if not provided
    if y_true_log is None:
        # Handle zero/negative values by adding small epsilon
        y_true_safe = np.maximum(y_true, 1e-10)
        y_true_log = np.log10(y_true_safe)
    else:
        y_true_log = np.asarray(y_true_log)
        
    if y_pred_log is None:
        # Handle zero/negative predictions
        y_pred_safe = np.maximum(y_pred, 1e-10)
        y_pred_log = np.log10(y_pred_safe)
    else:
        y_pred_log = np.asarray(y_pred_log)
    
    # Original scale metrics
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    medae = median_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    # Log scale metrics
    rmse_log = np.sqrt(mean_squared_error(y_true_log, y_pred_log))
    mae_log = mean_absolute_error(y_true_log, y_pred_log)
    r2_log = r2_score(y_true_log, y_pred_log)
    
    # RMSLE is the RMSE in log space (standard definition)
    rmsle = rmse_log
    
    return {
        # Original scale metrics
        "RMSE": rmse,
        "MAE": mae,
        "MedAE": medae,
        "R2": r2,
        # Log scale metrics
        "RMSLE": rmsle,
        "RMSE_log": rmse_log,
        "MAE_log": mae_log,
        "R2_log": r2_log,
    }


def compute_fold_summary(fold_metrics: list) -> dict:
    """
    Aggregate metrics across all cross-validation folds.
    
    Parameters
    ----------
    fold_metrics : list
        List of metric dictionaries, one per fold (from compute_metrics).
        
    Returns
    -------
    dict
        Dictionary with mean and std for each metric across folds.
        Keys are formatted as "{metric}_mean" and "{metric}_std".
        
    Examples
    --------
    >>> fold_results = [compute_metrics(y_true, y_pred) for ...]
    >>> summary = compute_fold_summary(fold_results)
    >>> print(f"RMSE: {summary['RMSE_mean']:.3f} +/- {summary['RMSE_std']:.3f}")
    """
    if not fold_metrics:
        return {}
    
    # Get all metric names from first fold
    metric_names = list(fold_metrics[0].keys())
    
    summary = {}
    for metric in metric_names:
        values = [fold[metric] for fold in fold_metrics]
        summary[f"{metric}_mean"] = np.mean(values)
        summary[f"{metric}_std"] = np.std(values, ddof=1)
    
    return summary


def get_well_names() -> list:
    """
    Return the list of well identifiers in standard order.
    
    Returns
    -------
    list
        Well names ["A", "B", "C", "D", "E", "F", "G"].
    """
    return ["A", "B", "C", "D", "E", "F", "G"]


def print_fold_info(splits: list, df: pd.DataFrame, group_col: str = "Source") -> None:
    """
    Print summary information about cross-validation splits.
    
    Parameters
    ----------
    splits : list
        List of (train_idx, test_idx, fold_name) tuples from get_group_kfold_splits.
    df : pd.DataFrame
        Original DataFrame used to generate splits.
    group_col : str, optional
        Column name used for grouping. Default is "Source".
    """
    print(f"Cross-validation splits (GroupKFold by {group_col}):")
    print("-" * 60)
    
    total_train = 0
    total_test = 0
    
    for train_idx, test_idx, fold_name in splits:
        train_groups = sorted(df.iloc[train_idx][group_col].unique())
        print(f"Fold {fold_name}:")
        print(f"  Train: {len(train_idx):5d} samples from wells {train_groups}")
        print(f"  Test:  {len(test_idx):5d} samples from well  {fold_name}")
        total_train += len(train_idx)
        total_test += len(test_idx)
    
    print("-" * 60)
    print(f"Average train size: {total_train / len(splits):.0f}")
    print(f"Average test size:  {total_test / len(splits):.0f}")