# Synthetic Surrogate Data

This folder contains the synthetic surrogate data generator used for reproducibility checks.

The generator creates a schema-matched CSV for the `without-outlier-adaptive` variant used in the pipeline.

## Files

- `generate_synthetic_surrogate.py`: synthetic dataset generator.
- `synthetic_without_outlier_adaptive.csv`: default output file (created after running the generator).
- `experiment_config_synthetic.json`: Phase 2 config for quick synthetic smoke tests.
- `phase3_config_synthetic.json`: synthetic config for `run_phase3_pareto.py` and `run_phase3_shap.py`.
- `phase4_config_synthetic.json`: synthetic config for `run_phase4.py`.

## Generate Data

Run from repository root:

```bash
python datasets/generate_synthetic_surrogate.py
```

Custom output path, sample count, and seed:

```bash
python datasets/generate_synthetic_surrogate.py --output datasets/synthetic_custom.csv --n-rows 3000 --seed 42
```

Custom synthetic depth range:

```bash
python datasets/generate_synthetic_surrogate.py --depth-min 0 --depth-max 2000
```

### CLI Arguments

- `--output`: CSV path to write (default: `datasets/synthetic_without_outlier_adaptive.csv`)
- `--n-rows`: number of rows (default: `2284`)
- `--seed`: RNG seed (default: `42`)
- `--depth-min`: minimum synthetic relative depth (default: `0.0`)
- `--depth-max`: maximum synthetic relative depth (default: `2000.0`)

## Synthetic Config And Phase Runs

The file `datasets/experiment_config_synthetic.json` is a minimal Phase 2
config that points to:

- `datasets/synthetic_without_outlier_adaptive.csv`

It uses a lower default `hp_budget` (`200`) so quick checks finish faster.

### Phase 2 smoke test

Run from repository root:

```bash
python runners/run_phase2.py --config datasets/experiment_config_synthetic.json --variant synthetic_without_outlier_adaptive --model ExtraTrees --output_dir results/phase2_model_selection_synthetic
```

### Phase 2 analysis on synthetic output

```bash
python runners/analyze_phase2.py --results_dir results/phase2_model_selection_synthetic --metric-scope both
```

### Phase 3 and Phase 4 with synthetic data

Synthetic configs are provided in this folder:

- `datasets/phase3_config_synthetic.json`
- `datasets/phase4_config_synthetic.json`

Phase 3 sweep:

```bash
python runners/run_phase3_pareto.py --sweep --config datasets/phase3_config_synthetic.json --output_dir results/phase3_feature_selection_synthetic
```

Phase 3 retune:

```bash
python runners/run_phase3_pareto.py --retune --config datasets/phase3_config_synthetic.json --output_dir results/phase3_feature_selection_synthetic
```

Phase 3 validate:

```bash
python runners/run_phase3_pareto.py --validate --config datasets/phase3_config_synthetic.json --output_dir results/phase3_feature_selection_synthetic
```

Phase 3 SHAP:

```bash
python runners/run_phase3_shap.py --config datasets/phase3_config_synthetic.json --output_dir results/phase3_feature_selection_synthetic
```

Phase 4 nested LOWO:

```bash
python runners/run_phase4.py --config datasets/phase4_config_synthetic.json --output_dir results/phase4_generalization_synthetic/run
```

## Output Schema

Columns:

`DEPTH, CKHL_SM, CPOR_SM, CALI, CT, DRHO, GR, MSFL, NPHI, PHIT, RHOB, RT, SWT, CT_missing, RT_missing, SWT_missing, Source, Zone, DT, PEF, MSFL_missing_source, DT_missing_source, PEF_missing_source`

## How Data Is Generated

### 1) Physical ranges

The following channels are sampled from physical reference ranges:

- `RT`: `[0.2, 2000]`
- `CT`: `[0.0005, 5.0]`
- `SWT`: `[0.0, 1.0]`
- `GR`: `[0, 150]`
- `NPHI`: `[-0.05, 0.45]`
- `RHOB`: `[1.95, 2.95]`
- `DRHO`: `[-0.25, 0.25]`
- `DT`: `[40, 140]`
- `PEF`: `[1.0, 6.0]`
- `CALI`: `[4, 22]`
- `MSFL`: `[0.2, 2000]`
- `PHIT`: `[0.0, 0.40]`

### 2) Base continuous variables

- `DEPTH`: synthetic relative depth coordinate, generated separately per `Source`.
  - For each source, positive increments are sampled (`LogNormal(0, 0.35)`), cumulatively summed, and rescaled to `[depth_min, depth_max]`.
  - Default range is `[0, 2000]`.
  - These are not measured physical depths.
- `CPOR_SM ~ Uniform(0.03, 0.35)`

Other channels are sampled independently to keep the generator simple:

- `RT ~ LogUniform(0.2, 2000)`
- `CT ~ LogUniform(0.0005, 5.0)`
- `MSFL ~ LogUniform(0.2, 2000)`
- `SWT ~ Uniform(0.0, 1.0)`
- `GR ~ Uniform(0, 150)`
- `NPHI ~ Uniform(-0.05, 0.45)`
- `RHOB ~ Uniform(1.95, 2.95)`
- `DRHO ~ Uniform(-0.25, 0.25)`
- `DT ~ Uniform(40, 140)`
- `PEF ~ Uniform(1.0, 6.0)`
- `CALI ~ Uniform(4, 22)`
- `PHIT ~ Uniform(0.0, 0.40)`

No cross-feature physical equations are used.

### 3) Categorical fields

- `Source`: balanced over `A..G` by repeating and shuffling labels.
- `Zone`: derived from `DEPTH` by uniformly splitting `[depth_min, depth_max]` into 11 equal-width bins, mapped to `{0,1,2,3,4,5,6,7,8,9,10}`.

### 4) Missingness flags and NaN injection

Per-sample missing flags:

- `CT_missing ~ Bernoulli(0.02)`
- `RT_missing ~ Bernoulli(0.02)`
- `SWT_missing ~ Bernoulli(0.02)`

When a flag is `1`, the corresponding feature value is set to `NaN`.

Source-level missing-source flags:

- `MSFL_missing_source = 1` for `Source == G`
- `DT_missing_source = 1` for `Source in {F, G}`
- `PEF_missing_source = 1` for `Source in {E, F, G}`

When a missing-source flag is `1`, the corresponding channel value is set to `NaN`.

### 5) Synthetic target generation (`CKHL_SM`)

The target is generated in log space from a simple linear combination of
normalized continuous features:

- Build feature vector from:
  - `DEPTH, CPOR_SM, CALI, CT, DRHO, GR, MSFL, NPHI, PHIT, RHOB, RT, SWT, DT, PEF`
- Min-max normalize each feature to `[0, 1]` within the generated dataset.
- Sample random weights per run:
  - `w_i ~ Uniform(-1, 1)`
- Compute:
  - `log_perm = 1.0 + (1/p) * sum_i(w_i * x_i_norm) + Normal(0, 0.20)` where `p = 14`
- `CKHL_SM = 10^(log_perm)`, clipped to `[0.01, 5000]`

This formulation is synthetic and not physically calibrated. It exists to
produce a stable positive target with simple feature dependence. Clipping
guarantees strictly positive values required for downstream `log10`
transformation.

## Notes

- This dataset is synthetic and intended for code-path validation and reproducibility demonstrations.
- It is not designed to reproduce manuscript metrics.
- It is not a physics-based forward model of reservoir properties.
