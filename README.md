# cost-aware-permeability-prediction

Cost-aware machine learning framework for predicting horizontal permeability in a heterogeneous carbonate reservoir from wireline logs and core data. Implements the four-phase pipeline reported in the thesis: data preparation, model selection, cost-aware feature selection (Pareto frontier + SHAP), and nested generalization.

## Setup

The project uses a conda environment defined in `perm_env.yml`:

```bash
conda env create -f perm_env.yml
conda activate perm_env
```

## Repository layout

- `configs/` — JSON configuration: feature catalogue (`features_info.json`), per-model hyperparameter search spaces (`model_hyperparams.json`), experiment-level settings (`experiment_config.json`), per-phase cost / feature-bundle settings (`phase3_config_*.json`, `phase4_config_*.json`), and HPC job definitions (`cluster_jobs.json`).
- `src/` — Reusable library code: cross-validation utilities (`cv_utils.py`), preprocessing pipeline (`preprocessing.py`), model registry (`models.py`), hyperparameter search (`hp_search.py`), feature-subset evaluation (`feature_selection.py`), and nested CV (`nested_cv.py`).
- `runners/` — Phase entry points (`run_phase2.py`, `run_phase3_pareto.py`, `run_phase3_shap.py`, `run_phase4.py`) and matching `analyze_phase{2,3,4}.py` post-processing scripts.
- `datasets/` — Synthetic surrogate generator, synthetic configs, and optional local data files.
- `results/` — Per-phase outputs, organized as `phase2_model_selection/`, `phase3_feature_selection/`, and `phase4_generalization/`. 

## Synthetic surrogate workflow

Use `datasets/README.md` for full generator details. Quick start:

```bash
python datasets/generate_synthetic_surrogate.py
```

Synthetic configs available:

- `datasets/experiment_config_synthetic.json` (Phase 2)
- `datasets/phase3_config_synthetic.json` (Phase 3)
- `datasets/phase4_config_synthetic.json` (Phase 4)

Recommended smoke-test order:

```bash
# 1) Phase 2 synthetic run (produces best params JSON used by later phases)
python runners/run_phase2.py --config datasets/experiment_config_synthetic.json --variant synthetic_without_outlier_adaptive --model ExtraTrees --output_dir results/phase2_model_selection_synthetic

# 2) Phase 3 synthetic sweep + retune + validate + shap
python runners/run_phase3_pareto.py --sweep --config datasets/phase3_config_synthetic.json --output_dir results/phase3_feature_selection_synthetic
python runners/run_phase3_pareto.py --retune --config datasets/phase3_config_synthetic.json --output_dir results/phase3_feature_selection_synthetic
python runners/run_phase3_pareto.py --validate --config datasets/phase3_config_synthetic.json --output_dir results/phase3_feature_selection_synthetic
python runners/run_phase3_shap.py --config datasets/phase3_config_synthetic.json --output_dir results/phase3_feature_selection_synthetic

# 3) Phase 4 synthetic run
python runners/run_phase4.py --config datasets/phase4_config_synthetic.json --output_dir results/phase4_generalization_synthetic/run
```

## Analyze-based figure regeneration

Run these after the corresponding phase outputs exist:

```bash
python runners/analyze_phase2.py --results_dir results/phase2_model_selection_synthetic --metric-scope both
python runners/analyze_phase3.py --results_dir results/phase3_feature_selection_synthetic --config datasets/phase3_config_synthetic.json --metric-scope both
python runners/analyze_phase4.py --run_dir results/phase4_generalization_synthetic/run --config datasets/phase4_config_synthetic.json --metric-scope log
```

If you see `Warning: CV results directory not found .../cv_results`, run Phase 2 first or point `--results_dir` to a directory that already contains `cv_results/`.

## Figure mapping (script -> output folder)

- Phase 2 variant/model comparison figures: `runners/analyze_phase2.py` -> `results/<phase2_dir>/analysis/figures/`
- Phase 3 Pareto frontier and feature-frequency figures: `runners/analyze_phase3.py` -> `results/<phase3_dir>/analysis/figures/`
- Phase 3 SHAP bar/beeswarm/per-well heatmap: `runners/analyze_phase3.py` (consumes `shap/`) -> `results/<phase3_dir>/analysis/figures/`
- Phase 4 nested LOWO frontier/boxplot/heatmap and pred-vs-actual figures: `runners/analyze_phase4.py` -> `results/<phase4_run_dir_parent>/analysis/figures/`
