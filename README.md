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
- `datasets/` — Input data (well-log curves and core measurements). Not committed; place anonymized files here before running any phase.
- `results/` — Per-phase outputs, organized as `phase2_model_selection/`, `phase3_feature_selection/`, and `phase4_generalization/`.
