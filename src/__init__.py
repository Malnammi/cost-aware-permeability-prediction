"""
src - Permeability Prediction Library

This package provides reusable components for the permeability prediction
model selection and feature selection pipeline.

Modules:
    preprocessing: ColumnTransformer pipeline with RobustScaler, KNNImputer,
                   and OneHotEncoder for categorical features
    models: Registry of 16 model classes with hyperparameter distributions
    cv_utils: GroupKFold cross-validation and metrics computation
    hp_search: Hybrid Random + Bayesian hyperparameter search
    feature_selection: Phase 3 subset enumeration, cost computation,
                       Pareto frontier, and SHAP helpers

Example usage:
    from src.preprocessing import get_preprocessor, transform_target
    from src.models import MODEL_REGISTRY, get_model
    from src.cv_utils import get_group_kfold_splits, compute_metrics
    from src.hp_search import run_hybrid_search
    from src.feature_selection import load_phase3_config, enumerate_feature_subsets
"""

__version__ = "0.1.0"

# Enable imports from submodules
from .preprocessing import (
    get_preprocessor, 
    transform_target, 
    inverse_transform_target,
    load_features_info,
    prepare_data,
    detect_variant_type
)
from .models import MODEL_REGISTRY, get_model, list_models
from .cv_utils import (
    get_group_kfold_splits, 
    compute_metrics, 
    compute_fold_summary,
    get_well_names
)
from .hp_search import (
    random_search, 
    bayesian_search, 
    run_hybrid_search,
    results_to_dataframe
)
from .feature_selection import (
    load_phase3_config,
    enumerate_feature_subsets,
    compute_subset_cost,
    compute_subset_cost_bundled,
    resolve_binary_columns,
    evaluate_subset,
    identify_pareto_frontier,
    compute_shap_fold,
    aggregate_shap_importance,
)
