#!/usr/bin/env python
"""
Phase 2 Model Selection - Main Experiment Runner

Orchestrates the model selection experiment:
1. Load dataset variant(s)
2. Preprocess features
3. Run hybrid HP search for each model
4. Save results to CSV/JSON

Configuration is loaded from configs/experiment_config.json which defines:
- n_folds, hp_budget, random_state, random_search_fraction

Usage:
    # Run all models on all variants
    python runners/run_phase2.py --variants all

    # Run single model (for cluster job submission)
    python runners/run_phase2.py --model RandomForest

    # Run specific variant only
    python runners/run_phase2.py --variant with_outlier_default_adaptive
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import random

import numpy as np
import pandas as pd

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.preprocessing import (
    load_features_info, 
    prepare_data, 
    transform_target,
    inverse_transform_target,
    detect_variant_type
)
from src.models import MODEL_REGISTRY, get_model, list_models
from src.cv_utils import get_group_kfold_splits, compute_metrics, get_well_names
from src.hp_search import run_hybrid_search, results_to_dataframe


def load_experiment_config(config_path: str = None) -> dict:
    """Load experiment configuration from JSON file."""
    if config_path is None:
        config_path = Path(__file__).parent.parent / "configs" / "experiment_config.json"
    
    with open(config_path, "r") as f:
        return json.load(f)


def get_variant_paths(experiment_config: dict) -> dict:
    """Get dataset variant paths from experiment_config."""
    return experiment_config.get("dataset_variants", {})


def run_single_model(
    model_name: str,
    df: pd.DataFrame,
    variant_name: str,
    budget: int,
    n_splits: int,
    random_state: int,
    output_dir: Path,
    verbose: bool = True
) -> dict:
    """
    Run HP search for a single model on a single variant.
    
    Returns dict with results and saves CSV.
    """
    # Get model class, param distributions, and default kwargs
    model_class, param_distributions, default_kwargs = get_model(model_name)
    
    # Prepare data (raw features + unfitted preprocessor blueprint)
    features_info = load_features_info()
    X, y_raw, preprocessor = prepare_data(df, features_info)
    
    # Transform target
    y_log = transform_target(y_raw)
    
    # Get groups for CV
    groups = df["Source"].values
    
    if verbose:
        print(f"\nRunning HP search for {model_name} on {variant_name}")
        print(f"  X shape: {X.shape}")
        print(f"  y shape: {y_log.shape}")
        print(f"  Groups: {np.unique(groups)}")
    
    # Run hybrid search (preprocessor fitted per fold to avoid leakage)
    start_time = time.time()
    results = run_hybrid_search(
        model_name=model_name,
        model_class=model_class,
        param_distributions=param_distributions,
        X=X,
        y=y_log,
        groups=groups,
        budget=budget,
        n_splits=n_splits,
        random_state=random_state,
        default_kwargs=default_kwargs,
        verbose=verbose,
        preprocessor=preprocessor,
    )
    elapsed_time = time.time() - start_time
    
    # Convert to DataFrame and save
    results_df = results_to_dataframe(
        results['all_results'],
        model_name=model_name,
        variant_name=variant_name
    )
    
    # Add well-level fold scores with proper names
    well_names = get_well_names()
    
    # Save results CSV
    csv_path = output_dir / "cv_results" / f"{variant_name}_{model_name}_results.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(csv_path, index=False)
    
    # Extract timing info from hybrid search
    timing = results.get('timing', {})
    
    # Save best params JSON
    best_params_path = output_dir / "hp_search" / f"{variant_name}_{model_name}_best_params.json"
    best_params_path.parent.mkdir(parents=True, exist_ok=True)
    with open(best_params_path, 'w') as f:
        json.dump({
            'model': model_name,
            'variant': variant_name,
            'best_params': results['best_params'],
            'best_score': results['best_score'],
            'best_metrics': results.get('best_metrics', {}),
            'best_search_type': results['best_search_type'],
            'elapsed_seconds': elapsed_time,
            'timing': timing,
            'timestamp': datetime.now().isoformat()
        }, f, indent=2, default=str)
    
    if verbose:
        print(f"\nSaved results to: {csv_path}")
        print(f"Saved best params to: {best_params_path}")
        print(f"Elapsed time: {elapsed_time:.1f} seconds")
    
    return {
        'model': model_name,
        'variant': variant_name,
        'best_params': results['best_params'],
        'best_score': results['best_score'],
        'best_metrics': results.get('best_metrics', {}),
        'elapsed_seconds': elapsed_time,
        'timing': timing
    }


def run_experiment(
    models: list = None,
    variants: list = None,
    output_dir: str = None,
    verbose: bool = True,
    config_path: str = None
) -> list:
    """
    Run the full experiment for specified models and variants.
    
    Experiment parameters (n_folds, hp_budget, random_state) are loaded from
    config_path when provided, otherwise from configs/experiment_config.json.
    
    Parameters
    ----------
    models : list, optional
        List of model names. If None, runs all models.
    variants : list, optional
        List of variant names. If None, runs all variants.
    output_dir : str, optional
        Output directory. Defaults to results/phase2_model_selection.
    verbose : bool
        Whether to print progress.
    config_path : str, optional
        Path to experiment config JSON. Defaults to configs/experiment_config.json.
        
    Returns
    -------
    list
        List of result dictionaries.
    """
    # Load configs
    experiment_config = load_experiment_config(config_path)
    features_info = load_features_info()
    variant_paths = get_variant_paths(experiment_config)
    
    # Extract experiment parameters from config
    budget = experiment_config["hp_budget"]
    n_splits = experiment_config["n_folds"]
    random_state = experiment_config["random_state"]
    
    # Set global seeds for full reproducibility
    np.random.seed(random_state)
    random.seed(random_state)
    
    if verbose:
        print(f"Loaded experiment config:")
        print(f"  HP budget: {budget}")
        print(f"  CV folds: {n_splits}")
        print(f"  Random state: {random_state}")
    
    # Default to all models/variants
    if models is None:
        models = list_models()
    if variants is None:
        variants = list(variant_paths.keys())
    
    # Setup output directory
    if output_dir is None:
        output_dir = Path(__file__).parent.parent / "results" / "phase2_model_selection"
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_results = []
    variant_timings = {}
    experiment_start = time.time()
    
    for variant_name in variants:
        if variant_name not in variant_paths:
            print(f"Warning: Unknown variant '{variant_name}', skipping")
            continue
        
        # Load data
        variant_path = Path(__file__).parent.parent / variant_paths[variant_name]
        if not variant_path.exists():
            print(f"Warning: Variant file not found: {variant_path}, skipping")
            continue
        
        if verbose:
            print(f"\n{'#'*70}")
            print(f"# Loading variant: {variant_name}")
            print(f"# Path: {variant_path}")
            print(f"{'#'*70}")
        
        df = pd.read_csv(variant_path)
        
        if verbose:
            print(f"Loaded {len(df)} rows, {len(df.columns)} columns")
            print(f"Has outlier columns: {detect_variant_type(df)}")
        
        variant_start = time.time()
        
        for model_name in models:
            if model_name not in MODEL_REGISTRY:
                print(f"Warning: Unknown model '{model_name}', skipping")
                continue
            
            try:
                result = run_single_model(
                    model_name=model_name,
                    df=df,
                    variant_name=variant_name,
                    budget=budget,
                    n_splits=n_splits,
                    random_state=random_state,
                    output_dir=output_dir,
                    verbose=verbose
                )
                all_results.append(result)
                
            except Exception as e:
                print(f"Error running {model_name} on {variant_name}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        variant_elapsed = time.time() - variant_start
        variant_timings[variant_name] = variant_elapsed
        if verbose:
            print(f"\nVariant '{variant_name}' complete. Wall time: {variant_elapsed:.1f}s "
                  f"({variant_elapsed / 60:.1f} min)")
    
    experiment_elapsed = time.time() - experiment_start
    
    # Build runtime summary
    runtime_summary = {
        'total_experiment_seconds': experiment_elapsed,
        'total_experiment_minutes': experiment_elapsed / 60,
        'per_variant_seconds': variant_timings,
        'per_model_seconds': {
            r['model']: r.get('elapsed_seconds', 0) for r in all_results
        },
    }
    
    # Save summary
    if all_results:
        summary_path = output_dir / "summary" / "experiment_summary.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with open(summary_path, 'w') as f:
            json.dump({
                'models': models,
                'variants': variants,
                'budget': budget,
                'n_splits': n_splits,
                'random_state': random_state,
                'runtime': runtime_summary,
                'results': all_results,
                'timestamp': datetime.now().isoformat()
            }, f, indent=2, default=str)
        
        if verbose:
            print(f"\n{'='*70}")
            print(f"Experiment complete.")
            print(f"Total wall time: {experiment_elapsed:.1f}s "
                  f"({experiment_elapsed / 60:.1f} min)")
            if variant_timings:
                print(f"Per-variant wall times:")
                for vname, vtime in variant_timings.items():
                    print(f"  {vname}: {vtime:.1f}s ({vtime / 60:.1f} min)")
            print(f"Summary saved to: {summary_path}")
            print(f"{'='*70}")
    
    return all_results


def main():
    parser = argparse.ArgumentParser(
        description="Phase 2 Model Selection - HP Search Experiment. "
                    "Experiment parameters are loaded from --config if provided, "
                    "otherwise from configs/experiment_config.json."
    )
    
    parser.add_argument(
        "--model", "-m",
        type=str,
        default=None,
        help="Single model to run (e.g., RandomForest). Default: all models."
    )
    parser.add_argument(
        "--variant", "-v",
        type=str,
        default=None,
        help="Single variant to run. Default: all variants."
    )
    parser.add_argument(
        "--variants",
        type=str,
        default=None,
        help="Comma-separated list of variants, or 'all'. Default: all."
    )
    parser.add_argument(
        "--output_dir", "-o",
        type=str,
        default=None,
        help="Output directory. Default: results/phase2_model_selection."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to experiment config JSON. Default: configs/experiment_config.json."
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress verbose output."
    )
    parser.add_argument(
        "--list_models",
        action="store_true",
        help="List available models and exit."
    )
    parser.add_argument(
        "--list_variants",
        action="store_true",
        help="List available variants and exit."
    )
    parser.add_argument(
        "--show_config",
        action="store_true",
        help="Show experiment config and exit."
    )
    
    args = parser.parse_args()
    
    # Handle info commands
    if args.list_models:
        print("Available models:")
        for model in list_models():
            print(f"  - {model}")
        return
    
    if args.list_variants:
        config = load_experiment_config(args.config)
        variant_paths = get_variant_paths(config)
        print("Available variants:")
        for name, path in variant_paths.items():
            print(f"  - {name}: {path}")
        return
    
    if args.show_config:
        config = load_experiment_config(args.config)
        if args.config:
            print(f"Experiment configuration ({args.config}):")
        else:
            print("Experiment configuration (configs/experiment_config.json):")
        for key, value in config.items():
            if not key.startswith("_"):
                print(f"  {key}: {value}")
        return
    
    # Parse model(s)
    models = None
    if args.model:
        models = [args.model]
    
    # Parse variant(s)
    variants = None
    if args.variant:
        variants = [args.variant]
    elif args.variants:
        if args.variants.lower() == 'all':
            variants = None  # Will use all
        else:
            variants = [v.strip() for v in args.variants.split(',')]
    
    # Run experiment
    run_experiment(
        models=models,
        variants=variants,
        output_dir=args.output_dir,
        verbose=not args.quiet,
        config_path=args.config
    )


if __name__ == "__main__":
    main()
