"""
Model registry for Phase 2 model selection.

Provides a MODEL_REGISTRY mapping model names to (estimator_class,
param_distributions, default_kwargs) for use in randomized and Bayesian
hyperparameter search.

16 model classes across 5 families:
  - Linear: Ridge, Lasso, ElasticNet
  - Tree-based: DecisionTree, RandomForest, ExtraTrees, Bagging
  - Boosting: GradientBoosting, XGBoost, LightGBM, CatBoost, AdaBoost,
              HistGradientBoosting
  - Kernel / Instance-based: SVR, KNeighbors
  - Neural Network: MLP
"""

from __future__ import annotations

from typing import Dict, Tuple, Any

from scipy.stats import loguniform, randint, uniform
from sklearn.ensemble import (
    AdaBoostRegressor,
    BaggingRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

try:
    from xgboost import XGBRegressor
except ImportError:  # pragma: no cover - optional dependency
    XGBRegressor = None

try:
    from lightgbm import LGBMRegressor
except ImportError:  # pragma: no cover - optional dependency
    LGBMRegressor = None

try:
    from catboost import CatBoostRegressor
except ImportError:  # pragma: no cover - optional dependency
    CatBoostRegressor = None


# MODEL_REGISTRY maps model names to (estimator_class, param_distributions, default_kwargs).
# default_kwargs are applied at instantiation time (e.g. n_jobs, verbosity) and are NOT
# part of the hyperparameter search space.
MODEL_REGISTRY: Dict[str, Tuple[Any, Dict[str, Any], Dict[str, Any]]] = {
    # -------------------------------------------------------------------------
    # Linear models
    # -------------------------------------------------------------------------
    "Ridge": (
        Ridge,
        {"alpha": loguniform(1e-4, 100)},
        {},
    ),
    "Lasso": (
        Lasso,
        {"alpha": loguniform(1e-4, 100)},
        {},
    ),
    "ElasticNet": (
        ElasticNet,
        {"alpha": loguniform(1e-4, 100), "l1_ratio": uniform(0.0, 1.0)},
        {},
    ),
    # -------------------------------------------------------------------------
    # Tree-based models
    # -------------------------------------------------------------------------
    "DecisionTree": (
        DecisionTreeRegressor,
        {
            "max_depth": [None, 3, 5, 10, 20, 30],
            "min_samples_split": randint(2, 11),
            "min_samples_leaf": randint(1, 6),
            "max_features": ["sqrt", "log2", None],
        },
        {},
    ),
    "RandomForest": (
        RandomForestRegressor,
        {
            "n_estimators": [100, 200, 300, 500, 800, 1000],
            "max_depth": [None, 5, 10, 20, 30],
            "max_features": ["sqrt", "log2", 0.5],
            "min_samples_split": randint(2, 11),
            "min_samples_leaf": randint(1, 6),
        },
        {"n_jobs": -1},
    ),
    "ExtraTrees": (
        ExtraTreesRegressor,
        {
            "n_estimators": [100, 200, 300, 500, 800, 1000],
            "max_depth": [None, 5, 10, 20, 30],
            "max_features": ["sqrt", "log2", 0.5],
            "min_samples_split": randint(2, 11),
            "min_samples_leaf": randint(1, 6),
        },
        {"n_jobs": -1},
    ),
    "Bagging": (
        BaggingRegressor,
        {
            "n_estimators": [50, 100, 200, 500],
            "max_samples": uniform(0.5, 0.5),       # [0.5, 1.0]
            "max_features": uniform(0.5, 0.5),       # [0.5, 1.0]
            "bootstrap": [True, False],
            "bootstrap_features": [True, False],
        },
        {"n_jobs": -1},
    ),
    # -------------------------------------------------------------------------
    # Boosting models
    # -------------------------------------------------------------------------
    "GradientBoosting": (
        GradientBoostingRegressor,
        {
            "n_estimators": [100, 200, 400],
            "learning_rate": loguniform(1e-3, 0.3),
            "max_depth": randint(2, 8),
            "subsample": uniform(0.6, 0.4),
            "min_samples_split": randint(2, 11),
            "min_samples_leaf": randint(1, 6),
        },
        {},  # GBM is inherently sequential
    ),
    "XGBoost": (
        XGBRegressor,
        {
            "n_estimators": [200, 500, 800],
            "max_depth": randint(3, 10),
            "learning_rate": loguniform(1e-3, 0.3),
            "subsample": uniform(0.6, 0.4),
            "colsample_bytree": uniform(0.6, 0.4),
            "reg_alpha": loguniform(1e-6, 10),
            "reg_lambda": loguniform(1e-6, 10),
            "min_child_weight": randint(1, 10),
            "gamma": loguniform(1e-8, 5),
        },
        {"nthread": -1, "verbosity": 0},
    ),
    "LightGBM": (
        LGBMRegressor,
        {
            "n_estimators": [200, 500, 800],
            "max_depth": randint(3, 10),
            "learning_rate": loguniform(1e-3, 0.3),
            "subsample": uniform(0.6, 0.4),
            "colsample_bytree": uniform(0.6, 0.4),
            "reg_alpha": loguniform(1e-6, 10),
            "reg_lambda": loguniform(1e-6, 10),
            "num_leaves": randint(15, 128),
            "min_child_samples": randint(5, 50),
        },
        {"n_jobs": -1, "verbose": -1},
    ),
    "CatBoost": (
        CatBoostRegressor,
        {
            "iterations": [200, 500, 800],
            "depth": randint(4, 10),
            "learning_rate": loguniform(1e-3, 0.3),
            "l2_leaf_reg": loguniform(1e-3, 10),
            "bagging_temperature": uniform(0.0, 1.0),
            "border_count": [32, 64, 128, 254],
        },
        {"thread_count": -1, "verbose": False},
    ),
    "AdaBoost": (
        AdaBoostRegressor,
        {
            "n_estimators": [50, 100, 200, 500],
            "learning_rate": loguniform(1e-2, 2.0),
            "loss": ["linear", "square", "exponential"],
        },
        {},
    ),
    "HistGradientBoosting": (
        HistGradientBoostingRegressor,
        {
            "max_iter": [100, 200, 500, 800],
            "learning_rate": loguniform(1e-3, 0.3),
            "max_depth": [None, 5, 10, 20],
            "min_samples_leaf": randint(5, 50),
            "max_leaf_nodes": [None, 31, 63, 127, 255],
            "l2_regularization": loguniform(1e-6, 10),
        },
        {},  # Uses all cores by default
    ),
    # -------------------------------------------------------------------------
    # Kernel / Instance-based models
    # -------------------------------------------------------------------------
    "SVR": (
        SVR,
        {
            "kernel": ["rbf", "poly", "sigmoid", "linear"],
            "C": loguniform(1e-2, 1e3),
            "epsilon": loguniform(1e-4, 1.0),
            "gamma": ["scale", "auto"],
            "degree": [2, 3, 4],
        },
        {},
    ),
    "KNeighbors": (
        KNeighborsRegressor,
        {
            "n_neighbors": randint(3, 31),
            "weights": ["uniform", "distance"],
            "p": [1, 2],
        },
        {"n_jobs": -1},
    ),
    # -------------------------------------------------------------------------
    # Neural Network
    # -------------------------------------------------------------------------
    "MLP": (
        MLPRegressor,
        {
            "hidden_layer_sizes": [
                (64,), (128,), (256,),
                (64, 32), (128, 64), (256, 128),
                (128, 64, 32),
            ],
            "learning_rate_init": loguniform(1e-4, 1e-2),
            "alpha": loguniform(1e-6, 1e-2),
            "activation": ["relu", "tanh"],
            "solver": ["adam", "lbfgs"],
            "batch_size": [32, 64, 128, 256],
        },
        {},
    ),
}


def get_model(name: str) -> Tuple[Any, Dict[str, Any], Dict[str, Any]]:
    """
    Return (estimator_class, param_distributions, default_kwargs) for the given model name.

    default_kwargs contains fixed settings (e.g. n_jobs=-1, verbosity) that should
    be applied at instantiation time but are NOT part of the HP search space.
    """
    if name not in MODEL_REGISTRY:
        available = ", ".join(sorted(MODEL_REGISTRY))
        raise KeyError(f"Unknown model '{name}'. Available: {available}")

    model_class, param_distributions, default_kwargs = MODEL_REGISTRY[name]
    if model_class is None:
        raise ImportError(
            f"Model '{name}' requires an optional dependency that is not installed."
        )

    return model_class, param_distributions, default_kwargs


def list_models() -> list[str]:
    """
    Return sorted list of available model names.
    """
    return sorted(MODEL_REGISTRY.keys())
