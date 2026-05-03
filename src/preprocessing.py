"""
Preprocessing module for permeability prediction.

Creates sklearn Pipeline with ColumnTransformer:
- Continuous features: RobustScaler -> KNNImputer
- Categorical features (Source, Zone): OneHotEncoder
- Binary features: Passthrough
- Target (CKHL_SM): log10 transform
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, OneHotEncoder


def load_features_info(config_path: str = None) -> dict:
    """
    Load feature configuration from JSON file.
    
    Parameters
    ----------
    config_path : str, optional
        Path to features_info.json. If None, uses default location.
        
    Returns
    -------
    dict
        Feature configuration dictionary.
    """
    if config_path is None:
        # Default path relative to project root
        config_path = Path(__file__).parent.parent / "configs" / "features_info.json"
    
    with open(config_path, "r") as f:
        return json.load(f)


def get_feature_columns(features_info: dict, has_outlier_columns: bool = True) -> dict:
    """
    Extract feature column names by type from features_info.
    
    Parameters
    ----------
    features_info : dict
        Feature configuration dictionary.
    has_outlier_columns : bool
        Whether the dataset variant has outlier indicator columns.
        True for 'with_outlier' variants, False for 'without_outlier' variants.
        
    Returns
    -------
    dict
        Dictionary with keys 'continuous', 'categorical', 'binary'.
    """
    continuous = features_info["continuous"]
    categorical = features_info["categorical"]
    
    # Binary columns depend on dataset variant
    binary = list(features_info["binary_common"])
    if has_outlier_columns:
        binary.extend(features_info["binary_with_outlier_only"])
    
    return {
        "continuous": continuous,
        "categorical": categorical,
        "binary": binary
    }


def get_preprocessor(
    features_info: dict = None,
    has_outlier_columns: bool = True,
    knn_neighbors: int = 5,
    continuous_subset: list = None,
    binary_subset: list = None
) -> ColumnTransformer:
    """
    Create a ColumnTransformer for preprocessing features.
    
    Pipeline structure:
    - Continuous: RobustScaler -> KNNImputer (handles missing DT, PEF)
    - Categorical: OneHotEncoder (Source, Zone)
    - Binary: Passthrough (no transformation needed)
    
    Parameters
    ----------
    features_info : dict, optional
        Feature configuration dictionary. If None, loads from default path.
    has_outlier_columns : bool
        Whether the dataset has outlier indicator columns.
    knn_neighbors : int
        Number of neighbors for KNNImputer. Default is 5.
    continuous_subset : list, optional
        Explicit list of continuous feature names to include. When None,
        all continuous features from features_info are used (Phase 2 behavior).
    binary_subset : list, optional
        Explicit list of binary column names to include. When None,
        all binary columns for the variant type are used (Phase 2 behavior).
        
    Returns
    -------
    ColumnTransformer
        Unfitted preprocessing transformer.
    """
    if features_info is None:
        features_info = load_features_info()
    
    feature_cols = get_feature_columns(features_info, has_outlier_columns)
    
    continuous_cols = continuous_subset if continuous_subset is not None else feature_cols["continuous"]
    binary_cols = binary_subset if binary_subset is not None else feature_cols["binary"]
    
    # Continuous pipeline: RobustScaler -> KNNImputer
    # Note: KNNImputer works on scaled data, which is recommended for distance-based imputation
    continuous_pipeline = Pipeline([
        ("scaler", RobustScaler()),
        ("imputer", KNNImputer(n_neighbors=knn_neighbors, weights="distance"))
    ])
    
    # Get known categories for categorical features (ensures consistent encoding across CV folds)
    categorical_cols = feature_cols["categorical"]
    categorical_values = features_info.get("categorical_values", {})
    categories = [
        categorical_values.get(col, "auto") for col in categorical_cols
    ]
    
    # Categorical pipeline: OneHotEncoder with explicit categories
    # Using explicit categories ensures consistent feature dimensionality across all CV folds
    # handle_unknown='error' since all valid categories are known upfront
    categorical_pipeline = OneHotEncoder(
        categories=categories,
        drop="first",  # Drop first category to avoid multicollinearity
        sparse_output=False,
        handle_unknown="error"
    )
    
    transformers = [("categorical", categorical_pipeline, categorical_cols)]
    if continuous_cols:
        transformers.insert(0, ("continuous", continuous_pipeline, continuous_cols))
    if binary_cols:
        transformers.append(("binary", "passthrough", binary_cols))
    
    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=True
    )
    
    return preprocessor


def transform_target(y: np.ndarray) -> np.ndarray:
    """
    Apply log10 transformation to target values.
    
    Permeability values span several orders of magnitude, so log10 transform
    helps normalize the distribution for better model performance.
    
    Parameters
    ----------
    y : array-like
        Raw permeability values (CKHL_SM).
        
    Returns
    -------
    np.ndarray
        Log10-transformed permeability values.
        
    Raises
    ------
    ValueError
        If any values are <= 0 (cannot take log of non-positive numbers).
    """
    y = np.asarray(y)
    if np.any(y <= 0):
        raise ValueError("Target values must be positive for log10 transformation. "
                        f"Found {np.sum(y <= 0)} non-positive values.")
    return np.log10(y)


def inverse_transform_target(y_log: np.ndarray) -> np.ndarray:
    """
    Reverse log10 transformation to get original scale predictions.
    
    Parameters
    ----------
    y_log : array-like
        Log10-transformed permeability values.
        
    Returns
    -------
    np.ndarray
        Permeability values in original scale.
    """
    return np.power(10, np.asarray(y_log))


def detect_variant_type(df: pd.DataFrame) -> bool:
    """
    Detect whether a DataFrame is from a 'with_outlier' variant.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame to check.
        
    Returns
    -------
    bool
        True if any outlier indicator columns are present.
    """
    outlier_cols = [
        "CALI_outlier", "CT_outlier", "DRHO_outlier", "GR_outlier",
        "MSFL_outlier", "NPHI_outlier", "PHIT_outlier", "RHOB_outlier",
        "RT_outlier", "SWT_outlier", "DT_outlier", "PEF_outlier",
        "CPOR_SM_outlier"
    ]
    return any(col in df.columns for col in outlier_cols)


def prepare_data(
    df: pd.DataFrame,
    features_info: dict = None,
    target_col: str = None,
    continuous_subset: list = None,
    binary_subset: list = None
) -> tuple:
    """
    Prepare data for model training/prediction.
    
    Separates features and target, applies appropriate preprocessing setup.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with features and target.
    features_info : dict, optional
        Feature configuration. If None, loads from default path.
    target_col : str, optional
        Name of target column. If None, uses value from features_info.
    continuous_subset : list, optional
        Explicit list of continuous feature names to include. When None,
        all continuous features from features_info are used (Phase 2 behavior).
    binary_subset : list, optional
        Explicit list of binary column names to include. When None,
        all binary columns for the variant type are used (Phase 2 behavior).
        
    Returns
    -------
    tuple
        (X, y, preprocessor) where:
        - X: DataFrame with feature columns only
        - y: Series with target values (raw, not transformed)
        - preprocessor: Unfitted ColumnTransformer
    """
    if features_info is None:
        features_info = load_features_info()
    
    if target_col is None:
        target_col = features_info["target_label"]
    
    # Detect variant type automatically
    has_outlier = detect_variant_type(df)
    
    # Get feature columns for this variant
    feature_cols = get_feature_columns(features_info, has_outlier)
    
    continuous_cols = continuous_subset if continuous_subset is not None else feature_cols["continuous"]
    categorical_cols = feature_cols["categorical"]
    binary_cols = binary_subset if binary_subset is not None else feature_cols["binary"]
    
    all_features = continuous_cols + categorical_cols + binary_cols
    
    # Filter to only columns that exist in the DataFrame
    available_features = [col for col in all_features if col in df.columns]
    
    X = df[available_features].copy()
    y = df[target_col].copy()
    
    preprocessor = get_preprocessor(
        features_info, has_outlier,
        continuous_subset=continuous_subset,
        binary_subset=binary_subset
    )
    
    return X, y, preprocessor


def get_feature_names_after_transform(
    preprocessor: ColumnTransformer,
    features_info: dict = None,
    has_outlier_columns: bool = True
) -> list:
    """
    Get feature names after preprocessing transformation.
    
    Must be called after the preprocessor has been fitted.
    
    Parameters
    ----------
    preprocessor : ColumnTransformer
        Fitted ColumnTransformer.
    features_info : dict, optional
        Feature configuration dictionary.
    has_outlier_columns : bool
        Whether the dataset has outlier indicator columns.
        
    Returns
    -------
    list
        List of feature names after transformation.
    """
    try:
        return list(preprocessor.get_feature_names_out())
    except AttributeError:
        # Fallback for unfitted transformer or older sklearn versions
        if features_info is None:
            features_info = load_features_info()
        
        feature_cols = get_feature_columns(features_info, has_outlier_columns)
        
        # Continuous features keep their names
        names = [f"continuous__{col}" for col in feature_cols["continuous"]]
        
        # Categorical features get one-hot encoded names
        # (approximate - exact names depend on fit data)
        for cat_col in feature_cols["categorical"]:
            names.append(f"categorical__{cat_col}_encoded")
        
        # Binary features keep their names
        names.extend([f"binary__{col}" for col in feature_cols["binary"]])
        
        return names
