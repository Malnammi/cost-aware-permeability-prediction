"""
Hyperparameter search module for Phase 2 model selection.

Implements a hybrid search strategy:
1. Step 1 - Random search (N/2 iterations): Broad exploration of HP space
2. Step 2 - Bayesian search (N/2 trials): Focused optimization using Optuna

The hybrid approach combines the benefits of random search (good coverage,
embarrassingly parallel) with Bayesian optimization (efficient exploitation
of promising regions).
"""

from __future__ import annotations

import time
import warnings
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import pandas as pd
from scipy.stats import loguniform, randint, uniform
from sklearn.model_selection import GroupKFold, ParameterSampler
from sklearn.base import clone

# Import from sibling modules for metrics computation
from src.cv_utils import compute_metrics
from src.preprocessing import inverse_transform_target

try:
    import optuna
    from optuna.distributions import (
        CategoricalDistribution,
        FloatDistribution,
        IntDistribution,
    )
    from optuna.samplers import TPESampler
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False


def _sample_from_distribution(dist: Any, random_state: np.random.RandomState) -> Any:
    """
    Sample a single value from a scipy.stats distribution or list.
    
    Parameters
    ----------
    dist : distribution or list
        Either a scipy.stats distribution (loguniform, uniform, randint)
        or a list of discrete choices.
    random_state : np.random.RandomState
        Random state for reproducibility.
        
    Returns
    -------
    Any
        Sampled value.
    """
    if isinstance(dist, list):
        return random_state.choice(dist)
    elif hasattr(dist, 'rvs'):
        # scipy.stats distribution
        return dist.rvs(random_state=random_state)
    else:
        return dist


def _param_dist_to_optuna(
    trial: "optuna.Trial",
    param_name: str,
    dist: Any
) -> Any:
    """
    Convert a scipy.stats distribution to an Optuna suggestion.
    
    Parameters
    ----------
    trial : optuna.Trial
        Optuna trial object.
    param_name : str
        Name of the hyperparameter.
    dist : distribution or list
        scipy.stats distribution or list of choices.
        
    Returns
    -------
    Any
        Suggested value from Optuna.
    """
    if isinstance(dist, list):
        # Discrete choices - need to handle None and mixed types
        # Convert all to strings for Optuna, then convert back
        str_choices = [str(v) for v in dist]
        selected = trial.suggest_categorical(param_name, str_choices)
        # Find original value
        idx = str_choices.index(selected)
        return dist[idx]
    
    # Handle scipy.stats distributions
    dist_name = type(dist).__name__
    
    if hasattr(dist, 'args') and hasattr(dist, 'kwds'):
        # scipy.stats frozen distribution
        args = dist.args
        kwds = dist.kwds
        
        # loguniform (scipy calls it loguniform_gen)
        if 'loguniform' in dist_name.lower() or (
            hasattr(dist, 'dist') and 'loguniform' in type(dist.dist).__name__.lower()
        ):
            # loguniform(a, b) samples from [a, a*scale] where scale = b-a+1
            # For loguniform(1e-4, 100), low=1e-4, high=100
            low = args[0] if args else kwds.get('loc', 1e-4)
            # Scale is args[1] or kwds['scale'], actual high = low + scale
            scale = args[1] if len(args) > 1 else kwds.get('scale', 1)
            high = low * scale
            return trial.suggest_float(param_name, low, high, log=True)
        
        # uniform
        elif 'uniform' in dist_name.lower() or (
            hasattr(dist, 'dist') and 'uniform' in type(dist.dist).__name__.lower()
        ):
            loc = args[0] if args else kwds.get('loc', 0)
            scale = args[1] if len(args) > 1 else kwds.get('scale', 1)
            return trial.suggest_float(param_name, loc, loc + scale)
        
        # randint
        elif 'randint' in dist_name.lower() or (
            hasattr(dist, 'dist') and 'randint' in type(dist.dist).__name__.lower()
        ):
            low = args[0] if args else kwds.get('low', 0)
            high = args[1] if len(args) > 1 else kwds.get('high', 10)
            return trial.suggest_int(param_name, low, high - 1)  # randint is [low, high)
    
    # Fallback: try to extract bounds from the distribution
    if hasattr(dist, 'a') and hasattr(dist, 'b'):
        # Has bounds
        if hasattr(dist, 'interval'):
            low, high = dist.interval(1.0)
            return trial.suggest_float(param_name, low, high)
    
    # Last resort: sample from distribution
    warnings.warn(f"Could not convert distribution for {param_name}, using random sample")
    return dist.rvs()


def _scipy_dist_to_optuna_dist(param_name: str, dist: Any) -> Any:
    """
    Convert a scipy.stats distribution or list to an Optuna Distribution object.
    
    This is needed for creating completed trials via optuna.trial.create_trial(),
    which requires explicit Optuna distribution objects.
    
    Parameters
    ----------
    param_name : str
        Name of the hyperparameter (for error messages).
    dist : distribution or list
        scipy.stats distribution or list of discrete choices.
        
    Returns
    -------
    optuna Distribution
        The corresponding Optuna distribution.
    """
    if not OPTUNA_AVAILABLE:
        raise ImportError("Optuna is required")
    
    if isinstance(dist, list):
        # Categorical: convert all values to strings for Optuna
        return CategoricalDistribution([str(v) for v in dist])
    
    dist_name = type(dist).__name__
    
    if hasattr(dist, 'args') and hasattr(dist, 'kwds'):
        args = dist.args
        kwds = dist.kwds
        
        # loguniform
        if 'loguniform' in dist_name.lower() or (
            hasattr(dist, 'dist') and 'loguniform' in type(dist.dist).__name__.lower()
        ):
            low = args[0] if args else kwds.get('loc', 1e-4)
            scale = args[1] if len(args) > 1 else kwds.get('scale', 1)
            high = low * scale
            return FloatDistribution(low, high, log=True)
        
        # uniform
        elif 'uniform' in dist_name.lower() or (
            hasattr(dist, 'dist') and 'uniform' in type(dist.dist).__name__.lower()
        ):
            loc = args[0] if args else kwds.get('loc', 0)
            scale = args[1] if len(args) > 1 else kwds.get('scale', 1)
            return FloatDistribution(loc, loc + scale)
        
        # randint
        elif 'randint' in dist_name.lower() or (
            hasattr(dist, 'dist') and 'randint' in type(dist.dist).__name__.lower()
        ):
            low = args[0] if args else kwds.get('low', 0)
            high = args[1] if len(args) > 1 else kwds.get('high', 10)
            return IntDistribution(low, high - 1)  # randint is [low, high)
    
    # Fallback
    warnings.warn(f"Could not convert distribution for {param_name} to Optuna, "
                  f"using wide float range")
    return FloatDistribution(1e-10, 1e10, log=True)


def _evaluate_cv_with_metrics(
    model: Any,
    X,
    y_log: np.ndarray,
    groups: np.ndarray,
    cv: GroupKFold,
    preprocessor=None,
) -> Dict[str, Any]:
    """
    Evaluate a model using GroupKFold CV and compute all metrics.
    
    This performs a manual CV loop to get predictions and compute the full
    suite of metrics from cv_utils.compute_metrics().
    
    Parameters
    ----------
    model : estimator
        Unfitted scikit-learn compatible estimator.
    X : array-like or DataFrame
        Training features. If *preprocessor* is provided, X should be the
        raw (untransformed) feature matrix; preprocessing is applied
        per-fold to avoid data leakage. Otherwise X is assumed to be
        already preprocessed.
    y_log : array-like of shape (n_samples,)
        Target values (log-transformed).
    groups : array-like of shape (n_samples,)
        Group labels for GroupKFold (well identifiers A-G).
    cv : GroupKFold
        Cross-validation splitter.
    preprocessor : ColumnTransformer, optional
        Unfitted preprocessor blueprint. When provided, it is cloned and
        fitted on each training fold independently, preventing leakage of
        test-fold statistics into the imputer and scaler.
        
    Returns
    -------
    dict
        Dictionary containing:
        - 'fold_metrics': list of metric dicts per fold
        - 'mean_metrics': dict of mean metrics across folds
        - 'std_metrics': dict of std metrics across folds
        - 'optimization_score': float, negative RMSE_log (for HP optimization)
    """
    fold_metrics_list = []
    
    for train_idx, test_idx in cv.split(X, y_log, groups):
        if preprocessor is not None:
            pre = clone(preprocessor)
            X_train = pre.fit_transform(X.iloc[train_idx])
            X_test = pre.transform(X.iloc[test_idx])
        else:
            X_train, X_test = X[train_idx], X[test_idx]
        y_train_log, y_test_log = y_log[train_idx], y_log[test_idx]
        
        # Clone and fit model
        model_clone = clone(model)
        model_clone.fit(X_train, y_train_log)
        
        # Predict in log space
        y_pred_log = model_clone.predict(X_test)
        
        # Clip predictions to prevent float64 overflow in 10**y_pred_log.
        # Permeability realistically spans ~10^-3 to ~10^4 mD (log10 in [-3, 4]);
        # anything outside [-15, 15] is a garbage prediction from a
        # poorly-configured estimator (e.g. SVR sigmoid with extreme C).
        y_pred_log_safe = np.clip(y_pred_log, -15, 15)
        
        # Convert to original scale
        y_test_orig = inverse_transform_target(y_test_log)
        y_pred_orig = inverse_transform_target(y_pred_log_safe)
        
        # Compute all metrics
        fold_metrics = compute_metrics(
            y_true=y_test_orig,
            y_pred=y_pred_orig,
            y_true_log=y_test_log,
            y_pred_log=y_pred_log
        )
        fold_metrics_list.append(fold_metrics)
    
    # Aggregate metrics across folds
    metric_names = list(fold_metrics_list[0].keys())
    mean_metrics = {}
    std_metrics = {}
    
    for metric in metric_names:
        values = [fm[metric] for fm in fold_metrics_list]
        mean_metrics[metric] = float(np.mean(values))
        std_metrics[metric] = float(np.std(values))
    
    # Primary optimization score: negative RMSE_log (higher is better, sklearn convention)
    optimization_score = -mean_metrics['RMSE_log']
    
    return {
        'fold_metrics': fold_metrics_list,
        'mean_metrics': mean_metrics,
        'std_metrics': std_metrics,
        'optimization_score': optimization_score
    }


def random_search(
    model_class: type,
    param_distributions: Dict[str, Any],
    X,
    y: np.ndarray,
    groups: np.ndarray,
    n_iter: int = 500,
    n_splits: int = 7,
    random_state: int = 42,
    default_kwargs: Dict[str, Any] = None,
    verbose: bool = True,
    preprocessor=None,
) -> List[Dict[str, Any]]:
    """
    Step 1: Random search for hyperparameter optimization.
    
    Performs random sampling from the parameter distributions and evaluates
    each configuration using GroupKFold cross-validation with full metrics.
    
    Parameters
    ----------
    model_class : type
        Scikit-learn compatible estimator class.
    param_distributions : dict
        Dictionary mapping parameter names to distributions (scipy.stats
        distributions or lists of discrete choices).
    X : array-like or DataFrame
        Training features. Raw (untransformed) if *preprocessor* is
        provided; already preprocessed otherwise.
    y : array-like of shape (n_samples,)
        Target values (log-transformed).
    groups : array-like of shape (n_samples,)
        Group labels for GroupKFold (well identifiers A-G).
    n_iter : int, default=500
        Number of random configurations to evaluate.
    n_splits : int, default=7
        Number of CV folds.
    random_state : int, default=42
        Random seed for reproducibility.
    verbose : bool, default=True
        Whether to print progress.
    preprocessor : ColumnTransformer, optional
        Unfitted preprocessor; cloned and fitted per fold when provided.
        
    Returns
    -------
    list
        List of dictionaries, each containing:
        - 'params': dict of hyperparameter values
        - 'mean_score': float, optimization score (negative RMSE_log)
        - 'std_score': float, std of optimization score across folds
        - 'mean_metrics': dict of mean metrics across folds
        - 'std_metrics': dict of std metrics across folds
        - 'fold_metrics': list of metric dicts per fold
        - 'search_type': 'random'
        - 'trial_id': int
    """
    cv = GroupKFold(n_splits=n_splits)
    
    results = []
    
    # Use ParameterSampler for efficient random sampling
    param_sampler = ParameterSampler(
        param_distributions,
        n_iter=n_iter,
        random_state=random_state
    )
    
    if verbose:
        print(f"Starting random search with {n_iter} iterations...")
    
    phase_start = time.time()
    
    _default_kwargs = default_kwargs or {}
    supports_random_state = False
    try:
        supports_random_state = "random_state" in model_class().get_params()
    except Exception:
        supports_random_state = False
    
    for trial_id, params in enumerate(param_sampler):
        try:
            # Instantiate model: default_kwargs first, then sampled HP params on top
            model_kwargs = dict(_default_kwargs)
            model_kwargs.update(params)
            
            # Add random_state and max_iter where applicable
            model_name = model_class.__name__
            if 'MLP' in model_name:
                model_kwargs.setdefault('random_state', random_state)
                model_kwargs.setdefault('max_iter', 500)
            elif supports_random_state:
                model_kwargs.setdefault('random_state', random_state)
            
            model = model_class(**model_kwargs)
            
            # Run cross-validation with full metrics
            trial_start = time.time()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                cv_results = _evaluate_cv_with_metrics(
                    model=model,
                    X=X,
                    y_log=y,
                    groups=groups,
                    cv=cv,
                    preprocessor=preprocessor,
                )
            trial_elapsed = time.time() - trial_start
            
            # Extract optimization score and std from RMSE_log fold values
            rmse_log_values = [fm['RMSE_log'] for fm in cv_results['fold_metrics']]
            optimization_scores = [-v for v in rmse_log_values]  # Negative for sklearn convention
            
            results.append({
                'params': params,
                'mean_score': cv_results['optimization_score'],
                'std_score': float(np.std(optimization_scores)),
                'mean_metrics': cv_results['mean_metrics'],
                'std_metrics': cv_results['std_metrics'],
                'fold_metrics': cv_results['fold_metrics'],
                'search_type': 'random',
                'trial_id': trial_id,
                'trial_elapsed_seconds': trial_elapsed
            })
            
            if verbose and (trial_id + 1) % 50 == 0:
                best_so_far = max(r['mean_score'] for r in results)
                phase_elapsed = time.time() - phase_start
                print(f"  Random search: {trial_id + 1}/{n_iter} trials, "
                      f"best score: {best_so_far:.4f}, "
                      f"elapsed: {phase_elapsed:.1f}s")
                
        except Exception as e:
            if verbose:
                print(f"  Trial {trial_id} failed: {e}")
            continue
    
    phase_elapsed = time.time() - phase_start
    
    if verbose:
        if results:
            best = max(results, key=lambda r: r['mean_score'])
            print(f"Random search complete. Best score: {best['mean_score']:.4f}, "
                  f"wall time: {phase_elapsed:.1f}s")
        else:
            print(f"Random search complete. No successful trials. "
                  f"Wall time: {phase_elapsed:.1f}s")
    
    return results


def bayesian_search(
    model_class: type,
    param_distributions: Dict[str, Any],
    X,
    y: np.ndarray,
    groups: np.ndarray,
    n_trials: int = 500,
    n_splits: int = 7,
    random_state: int = 42,
    warm_start_params: List[Dict[str, Any]] = None,
    default_kwargs: Dict[str, Any] = None,
    verbose: bool = True,
    preprocessor=None,
) -> List[Dict[str, Any]]:
    """
    Step 2: Bayesian search using Optuna's TPE sampler.
    
    Uses Tree-structured Parzen Estimator (TPE) for efficient exploration
    of the hyperparameter space. When warm_start_params is provided, ALL
    prior observations are injected into the Optuna study as completed
    trials (via study.add_trials). This seeds TPE's surrogate model with
    the full random search history, so the Bayesian phase makes informed
    proposals from trial 1 without re-evaluating any prior configuration.
    
    Parameters
    ----------
    model_class : type
        Scikit-learn compatible estimator class.
    param_distributions : dict
        Dictionary mapping parameter names to distributions.
    X : array-like or DataFrame
        Training features. Raw if *preprocessor* is provided.
    y : array-like of shape (n_samples,)
        Target values (log-transformed).
    groups : array-like of shape (n_samples,)
        Group labels for GroupKFold (well identifiers A-G).
    n_trials : int, default=500
        Number of NEW Optuna trials to run (none spent on re-evaluations).
    n_splits : int, default=7
        Number of CV folds.
    random_state : int, default=42
        Random seed for reproducibility.
    warm_start_params : list, optional
        List of result dicts from random search. All observations are
        injected as completed trials into the study history, giving TPE
        a fully informed surrogate model from the start.
    verbose : bool, default=True
        Whether to print progress.
        
    Returns
    -------
    list
        List of dictionaries (same format as random_search results).
        
    Raises
    ------
    ImportError
        If Optuna is not installed.
    """
    if not OPTUNA_AVAILABLE:
        raise ImportError(
            "Optuna is required for Bayesian search. "
            "Install it with: pip install optuna"
        )
    
    cv = GroupKFold(n_splits=n_splits)
    results = []
    
    # Suppress Optuna logging if not verbose
    if not verbose:
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    else:
        optuna.logging.set_verbosity(optuna.logging.INFO)
    
    # Build Optuna distribution objects for create_trial
    optuna_distributions = {}
    for param_name, dist in param_distributions.items():
        optuna_distributions[param_name] = _scipy_dist_to_optuna_dist(param_name, dist)
    
    # Create sampler with seed; set n_startup_trials=0 when seeding with RS
    # history so TPE starts model-based sampling immediately
    n_startup = 0 if warm_start_params else 10
    sampler = TPESampler(seed=random_state, n_startup_trials=n_startup)
    
    # Create study (direction is 'maximize' since sklearn uses neg scores)
    study = optuna.create_study(
        direction='maximize',
        sampler=sampler,
        study_name=f"{model_class.__name__}_bayesian"
    )
    
    # Seed study with ALL random search results as completed trials.
    # This injects the full RS history into TPE's surrogate model without
    # re-evaluating any configuration, so every Bayesian trial budget is
    # spent on new configurations proposed by the informed TPE sampler.
    n_seeded = 0
    if warm_start_params:
        if verbose:
            print(f"Seeding Bayesian search with {len(warm_start_params)} "
                  f"random search observations...")
        
        seed_trials = []
        for config in warm_start_params:
            try:
                # Convert params to Optuna format (strings for categoricals)
                trial_params = {}
                for param_name, value in config['params'].items():
                    dist = param_distributions.get(param_name)
                    if isinstance(dist, list):
                        trial_params[param_name] = str(value)
                    else:
                        trial_params[param_name] = value
                
                completed_trial = optuna.trial.create_trial(
                    params=trial_params,
                    distributions=optuna_distributions,
                    values=[config['mean_score']],
                    state=optuna.trial.TrialState.COMPLETE,
                )
                seed_trials.append(completed_trial)
            except Exception as e:
                if verbose:
                    print(f"  Could not create seed trial: {e}")
                continue
        
        if seed_trials:
            study.add_trials(seed_trials)
            n_seeded = len(seed_trials)
            if verbose:
                print(f"  Successfully seeded {n_seeded} trials into TPE history")
    
    _default_kwargs = default_kwargs or {}
    supports_random_state = False
    try:
        supports_random_state = "random_state" in model_class().get_params()
    except Exception:
        supports_random_state = False
    
    def objective(trial: optuna.Trial) -> float:
        """Objective function for Optuna."""
        # Sample parameters
        params = {}
        for param_name, dist in param_distributions.items():
            params[param_name] = _param_dist_to_optuna(trial, param_name, dist)
        
        try:
            # Instantiate model: default_kwargs first, then sampled HP params on top
            model_kwargs = dict(_default_kwargs)
            model_kwargs.update(params)
            model_name = model_class.__name__
            
            # Add random_state and max_iter where applicable
            if 'MLP' in model_name:
                model_kwargs.setdefault('random_state', random_state)
                model_kwargs.setdefault('max_iter', 500)
            elif supports_random_state:
                model_kwargs.setdefault('random_state', random_state)
            
            model = model_class(**model_kwargs)
            
            # Run cross-validation with full metrics
            trial_start = time.time()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                cv_results = _evaluate_cv_with_metrics(
                    model=model,
                    X=X,
                    y_log=y,
                    groups=groups,
                    cv=cv,
                    preprocessor=preprocessor,
                )
            trial_elapsed = time.time() - trial_start
            
            # Extract optimization score and std from RMSE_log fold values
            rmse_log_values = [fm['RMSE_log'] for fm in cv_results['fold_metrics']]
            optimization_scores = [-v for v in rmse_log_values]
            
            # Store results
            results.append({
                'params': params,
                'mean_score': cv_results['optimization_score'],
                'std_score': float(np.std(optimization_scores)),
                'mean_metrics': cv_results['mean_metrics'],
                'std_metrics': cv_results['std_metrics'],
                'fold_metrics': cv_results['fold_metrics'],
                'search_type': 'bayesian',
                'trial_id': trial.number,
                'trial_elapsed_seconds': trial_elapsed
            })
            
            return cv_results['optimization_score']
            
        except Exception as e:
            if verbose:
                print(f"  Bayesian trial {trial.number} failed: {e}")
            return float('-inf')
    
    if verbose:
        print(f"Starting Bayesian search with {n_trials} trials...")
    
    phase_start = time.time()
    
    # Define callback for progress reporting
    def progress_callback(study, trial):
        if verbose and (trial.number + 1) % 50 == 0:
            phase_elapsed = time.time() - phase_start
            print(f"  Bayesian search: {trial.number + 1}/{n_trials} trials, "
                  f"best score: {study.best_value:.4f}, "
                  f"elapsed: {phase_elapsed:.1f}s")
    
    # Run optimization
    study.optimize(
        objective,
        n_trials=n_trials,
        callbacks=[progress_callback] if verbose else None,
        show_progress_bar=False
    )
    
    phase_elapsed = time.time() - phase_start
    
    if verbose:
        print(f"Bayesian search complete. Best score: {study.best_value:.4f}, "
              f"wall time: {phase_elapsed:.1f}s")
        print(f"Seeded trials: {n_seeded}, new trials: {len(results)}, "
              f"total study history: {len(study.trials)}")
        print(f"Best params: {study.best_params}")
    
    return results


def run_hybrid_search(
    model_name: str,
    model_class: type,
    param_distributions: Dict[str, Any],
    X,
    y: np.ndarray,
    groups: np.ndarray,
    budget: int = 1000,
    n_splits: int = 7,
    random_state: int = 42,
    random_fraction: float = 0.5,
    default_kwargs: Dict[str, Any] = None,
    verbose: bool = True,
    preprocessor=None,
) -> Dict[str, Any]:
    """
    Orchestrate hybrid random + Bayesian HP search.
    
    First runs random search to broadly explore the space, then runs
    Bayesian search (warm-started from best random configs) for focused
    optimization. Uses RMSE_log as the optimization metric.
    
    Parameters
    ----------
    model_name : str
        Name of the model (for logging).
    model_class : type
        Scikit-learn compatible estimator class.
    param_distributions : dict
        Dictionary mapping parameter names to distributions.
    X : array-like or DataFrame
        Training features. Raw if *preprocessor* is provided.
    y : array-like of shape (n_samples,)
        Target values (log-transformed).
    groups : array-like of shape (n_samples,)
        Group labels for GroupKFold (well identifiers A-G).
    budget : int, default=1000
        Total number of trials (split between random and Bayesian).
    n_splits : int, default=7
        Number of CV folds.
    random_state : int, default=42
        Random seed for reproducibility.
    random_fraction : float, default=0.5
        Fraction of budget allocated to random search (remainder to Bayesian).
    verbose : bool, default=True
        Whether to print progress.
    preprocessor : ColumnTransformer, optional
        Unfitted preprocessor; cloned and fitted per CV fold when provided.
        
    Returns
    -------
    dict
        Dictionary containing:
        - 'all_results': list of all trial results from both searches
        - 'random_results': list of random search results
        - 'bayesian_results': list of Bayesian search results
        - 'best_params': dict of best hyperparameters
        - 'best_score': float, best optimization score (negative RMSE_log)
        - 'best_metrics': dict of all mean metrics for best configuration
        - 'best_search_type': str, 'random' or 'bayesian'
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"Hybrid HP Search for {model_name}")
        print(f"Budget: {budget} trials ({random_fraction:.0%} random, "
              f"{1-random_fraction:.0%} Bayesian)")
        print(f"{'='*60}")
    
    hybrid_start = time.time()
    
    # Split budget
    n_random = int(budget * random_fraction)
    n_bayesian = budget - n_random
    
    if default_kwargs is None:
        default_kwargs = {}
    
    # Step 1: Random search
    random_start = time.time()
    random_results = random_search(
        model_class=model_class,
        param_distributions=param_distributions,
        X=X,
        y=y,
        groups=groups,
        n_iter=n_random,
        n_splits=n_splits,
        random_state=random_state,
        default_kwargs=default_kwargs,
        verbose=verbose,
        preprocessor=preprocessor,
    )
    random_elapsed = time.time() - random_start
    
    # Step 2: Bayesian search (warm-started from random search)
    bayesian_results = []
    bayesian_elapsed = 0.0
    if n_bayesian > 0 and OPTUNA_AVAILABLE:
        bayesian_start = time.time()
        bayesian_results = bayesian_search(
            model_class=model_class,
            param_distributions=param_distributions,
            X=X,
            y=y,
            groups=groups,
            n_trials=n_bayesian,
            n_splits=n_splits,
            random_state=random_state + 1,  # Different seed for Bayesian
            warm_start_params=random_results,
            default_kwargs=default_kwargs,
            verbose=verbose,
            preprocessor=preprocessor,
        )
        bayesian_elapsed = time.time() - bayesian_start
    elif n_bayesian > 0:
        if verbose:
            print("Optuna not available, skipping Bayesian search")
    
    hybrid_elapsed = time.time() - hybrid_start
    
    # Combine results
    all_results = random_results + bayesian_results
    
    # Find best configuration
    if all_results:
        best_result = max(all_results, key=lambda r: r['mean_score'])
        best_params = best_result['params']
        best_score = best_result['mean_score']
        best_metrics = best_result['mean_metrics']
        best_search_type = best_result['search_type']
    else:
        best_params = {}
        best_score = float('-inf')
        best_metrics = {}
        best_search_type = 'none'
    
    # Compute timing summary
    timing = {
        'random_search_seconds': random_elapsed,
        'bayesian_search_seconds': bayesian_elapsed,
        'total_search_seconds': hybrid_elapsed,
        'n_random_trials': len(random_results),
        'n_bayesian_trials': len(bayesian_results),
        'n_total_trials': len(all_results),
    }
    if random_results:
        timing['avg_trial_seconds_random'] = random_elapsed / len(random_results)
    if bayesian_results:
        timing['avg_trial_seconds_bayesian'] = bayesian_elapsed / len(bayesian_results)
    if all_results:
        timing['avg_trial_seconds_overall'] = hybrid_elapsed / len(all_results)
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"Hybrid search complete for {model_name}")
        print(f"Best score (neg RMSE_log): {best_score:.4f}")
        if best_metrics:
            print(f"Best metrics: RMSE={best_metrics.get('RMSE', 'N/A'):.4f}, "
                  f"R2={best_metrics.get('R2', 'N/A'):.4f}, "
                  f"RMSE_log={best_metrics.get('RMSE_log', 'N/A'):.4f}")
        print(f"Best params: {best_params}")
        print(f"Timing: random={random_elapsed:.1f}s, "
              f"bayesian={bayesian_elapsed:.1f}s, "
              f"total={hybrid_elapsed:.1f}s")
        print(f"{'='*60}\n")
    
    return {
        'all_results': all_results,
        'random_results': random_results,
        'bayesian_results': bayesian_results,
        'best_params': best_params,
        'best_score': best_score,
        'best_metrics': best_metrics,
        'best_search_type': best_search_type,
        'timing': timing
    }


def results_to_dataframe(
    results: List[Dict[str, Any]],
    model_name: str = None,
    variant_name: str = None
) -> pd.DataFrame:
    """
    Convert search results to a pandas DataFrame.
    
    Parameters
    ----------
    results : list
        List of result dictionaries from random_search or bayesian_search.
    model_name : str, optional
        Model name to include as column.
    variant_name : str, optional
        Dataset variant name to include as column.
        
    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        - trial_id, search_type, mean_score, std_score
        - Mean metrics: RMSE, MAE, MedAE, R2, RMSLE, RMSE_log, MAE_log, R2_log
        - Std metrics: RMSE_std, MAE_std, etc.
        - Per-fold RMSE_log: fold_0_RMSE_log, fold_1_RMSE_log, ...
        - hp_* columns for each hyperparameter
        - model (if model_name provided)
        - variant (if variant_name provided)
    """
    if not results:
        return pd.DataFrame()
    
    rows = []
    for result in results:
        row = {
            'trial_id': result['trial_id'],
            'search_type': result['search_type'],
            'mean_score': result['mean_score'],
            'std_score': result['std_score'],
            'trial_elapsed_seconds': result.get('trial_elapsed_seconds', None),
        }
        
        # Add mean metrics
        if 'mean_metrics' in result:
            for metric_name, value in result['mean_metrics'].items():
                row[metric_name] = value
        
        # Add std metrics
        if 'std_metrics' in result:
            for metric_name, value in result['std_metrics'].items():
                row[f'{metric_name}_std'] = value
        
        # Add per-fold values for ALL metrics
        if 'fold_metrics' in result:
            for i, fold_metric in enumerate(result['fold_metrics']):
                for metric_name, value in fold_metric.items():
                    row[f'fold_{i}_{metric_name}'] = value
        
        # Add hyperparameters with hp_ prefix
        for param_name, param_value in result['params'].items():
            row[f'hp_{param_name}'] = param_value
        
        rows.append(row)
    
    df = pd.DataFrame(rows)
    
    # Add metadata columns
    if model_name:
        df.insert(0, 'model', model_name)
    if variant_name:
        df.insert(0, 'variant', variant_name)
    
    # Sort by mean score descending
    df = df.sort_values('mean_score', ascending=False).reset_index(drop=True)
    
    return df


def get_top_configs(
    results: List[Dict[str, Any]],
    n_top: int = 10,
    search_type: str = None
) -> List[Dict[str, Any]]:
    """
    Get top N configurations from search results.
    
    Parameters
    ----------
    results : list
        List of result dictionaries.
    n_top : int, default=10
        Number of top configurations to return.
    search_type : str, optional
        Filter by search type ('random' or 'bayesian').
        If None, include all.
        
    Returns
    -------
    list
        Top N result dictionaries sorted by mean_score descending.
    """
    filtered = results
    if search_type:
        filtered = [r for r in results if r['search_type'] == search_type]
    
    sorted_results = sorted(
        filtered,
        key=lambda r: r['mean_score'],
        reverse=True
    )
    
    return sorted_results[:n_top]
