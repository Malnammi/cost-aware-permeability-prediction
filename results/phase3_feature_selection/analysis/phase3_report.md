# Phase 3 Feature Selection — Analysis Report

Generated: 2026-06-06 16:06:54

## Configuration

- **Primary model**: ExtraTrees
- **Primary variant**: `dataset/prepared_variants/all_sources_without_outlier_adaptive.csv`
- **Metric scope**: `both`
- **Reporting metrics**: RMSE_log, MAE_log, R2_log, RMSE, MAE, R2
- **Sweep features**: 13 (GR, CALI, DRHO, RHOB, NPHI, RT, CT, PEF, MSFL, PHIT, DT, SWT, CPOR_SM)
- **Total subsets**: 2^13 = 8192
- **Always included**: DEPTH (continuous), Source & Zone (categorical)

---

## SHAP Feature Importance

SHAP values computed across all LOWO folds using TreeExplainer.

### Global Importance (Top 15 by mean |SHAP|)

| feature | mean_abs_shap |
| --- | --- |
| CPOR_SM | 0.5834 |
| PHIT | 0.1231 |
| Zone | 0.1079 |
| RHOB | 0.0811 |
| PEF_missing_source | 0.0631 |
| NPHI | 0.0626 |
| SWT | 0.0521 |
| DT | 0.0363 |
| PEF | 0.0270 |
| MSFL | 0.0182 |
| CALI | 0.0154 |
| CT | 0.0148 |
| GR | 0.0129 |
| Source | 0.0125 |
| RT | 0.0120 |

### Per-Well Breakdown

Wells analysed: A, B, C, D, E, F, G

---

## Pareto Sweep Results

- **Subsets evaluated**: 8192
- **Cost range**: 0 – 39
- **Pareto frontier (RMSE_log)**: 15 subsets
- **Pareto frontier (MAE_log)**: 19 subsets
- **Pareto frontier (R2_log)**: 16 subsets
- **Pareto frontier (RMSE)**: 18 subsets
- **Pareto frontier (MAE)**: 13 subsets
- **Pareto frontier (R2)**: 9 subsets

### Primary Pareto Frontier (RMSE_log)

| subset_id | features | n_features | cost | RMSE_log_mean | RMSE_log_std | RMSE_log_ci |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | (DEPTH-only) | 0 | 0 | 1.1174 | 0.1131 | 1.2221 |
| 1 | GR | 1 | 1 | 1.0768 | 0.0400 | 1.1138 |
| 12 | DRHO,RHOB | 2 | 2 | 0.8792 | 0.0799 | 0.9531 |
| 140 | DRHO,RHOB,PEF | 3 | 3 | 0.8404 | 0.0788 | 0.9133 |
| 152 | RHOB,NPHI,PEF | 3 | 5 | 0.8193 | 0.1006 | 0.9124 |
| 640 | PEF,PHIT | 2 | 7 | 0.8134 | 0.0867 | 0.8935 |
| 704 | CT,PEF,PHIT | 3 | 9 | 0.8069 | 0.0940 | 0.8938 |
| 4096 | CPOR_SM | 1 | 10 | 0.7394 | 0.0366 | 0.7733 |
| 4098 | CALI,CPOR_SM | 2 | 11 | 0.6959 | 0.0354 | 0.7286 |
| 4236 | DRHO,RHOB,PEF,CPOR_SM | 4 | 13 | 0.6788 | 0.0445 | 0.7200 |
| 4238 | CALI,DRHO,RHOB,PEF,CPOR_SM | 5 | 14 | 0.6751 | 0.0507 | 0.7220 |
| 4300 | DRHO,RHOB,CT,PEF,CPOR_SM | 5 | 15 | 0.6727 | 0.0483 | 0.7174 |
| 4290 | CALI,CT,PEF,CPOR_SM | 4 | 16 | 0.6623 | 0.0585 | 0.7164 |
| 4326 | CALI,DRHO,RT,CT,PEF,CPOR_SM | 6 | 18 | 0.6609 | 0.0638 | 0.7199 |
| 4550 | CALI,DRHO,CT,PEF,MSFL,CPOR_SM | 6 | 21 | 0.6585 | 0.0561 | 0.7103 |

**Key operating points:**

- Cheapest on frontier: cost=0, n_features=0, RMSE_log_mean=1.1174
- Best performance on frontier: cost=21, n_features=6, RMSE_log_mean=0.6585

---

## HP Re-Tuning Results

- **Subsets re-tuned**: 15

| subset_id | features | n_features | cost | RMSE_log_mean | RMSE_log_std | RMSE_log_ci |
| --- | --- | --- | --- | --- | --- | --- |
| 4550 | CALI,DRHO,CT,PEF,MSFL,CPOR_SM | 6 | 21 | 0.6523 | 0.0563 | 0.7043 |
| 4326 | CALI,DRHO,RT,CT,PEF,CPOR_SM | 6 | 18 | 0.6587 | 0.0594 | 0.7136 |
| 4290 | CALI,CT,PEF,CPOR_SM | 4 | 16 | 0.6602 | 0.0637 | 0.7191 |
| 4238 | CALI,DRHO,RHOB,PEF,CPOR_SM | 5 | 14 | 0.6731 | 0.0476 | 0.7172 |
| 4300 | DRHO,RHOB,CT,PEF,CPOR_SM | 5 | 15 | 0.6732 | 0.0481 | 0.7178 |
| 4236 | DRHO,RHOB,PEF,CPOR_SM | 4 | 13 | 0.6766 | 0.0443 | 0.7176 |
| 4098 | CALI,CPOR_SM | 2 | 11 | 0.6928 | 0.0335 | 0.7238 |
| 4096 | CPOR_SM | 1 | 10 | 0.7171 | 0.0403 | 0.7544 |
| 640 | PEF,PHIT | 2 | 7 | 0.8042 | 0.0880 | 0.8856 |
| 704 | CT,PEF,PHIT | 3 | 9 | 0.8060 | 0.0926 | 0.8917 |
| 152 | RHOB,NPHI,PEF | 3 | 5 | 0.8156 | 0.0919 | 0.9006 |
| 140 | DRHO,RHOB,PEF | 3 | 3 | 0.8345 | 0.0709 | 0.9001 |
| 12 | DRHO,RHOB | 2 | 2 | 0.8569 | 0.0664 | 0.9183 |
| 1 | GR | 1 | 1 | 1.0349 | 0.0478 | 1.0791 |
| 0 | (DEPTH-only) | 0 | 0 | 1.0494 | 0.0408 | 1.0872 |

---

## Validation (XGBoost)

Re-evaluated Pareto subsets with XGBoost on `dataset/prepared_variants/all_sources_with_outlier_default_linear.csv`.

| subset_id | features | n_features | cost | RMSE_log_mean | RMSE_log_std | RMSE_log_ci |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | (DEPTH-only) | 0 | 0 | 1.0965 | 0.0972 | 1.1864 |
| 1 | GR | 1 | 1 | 1.0475 | 0.0846 | 1.1257 |
| 12 | DRHO,RHOB | 2 | 2 | 0.8665 | 0.0730 | 0.9340 |
| 140 | DRHO,RHOB,PEF | 3 | 3 | 0.8359 | 0.0667 | 0.8976 |
| 152 | RHOB,NPHI,PEF | 3 | 5 | 0.8413 | 0.1120 | 0.9449 |
| 640 | PEF,PHIT | 2 | 7 | 0.8349 | 0.0926 | 0.9205 |
| 4096 | CPOR_SM | 1 | 10 | 0.7296 | 0.0514 | 0.7771 |
| 4098 | CALI,CPOR_SM | 2 | 11 | 0.7330 | 0.0538 | 0.7828 |
| 4236 | DRHO,RHOB,PEF,CPOR_SM | 4 | 13 | 0.6814 | 0.0552 | 0.7325 |
| 4238 | CALI,DRHO,RHOB,PEF,CPOR_SM | 5 | 14 | 0.7176 | 0.0755 | 0.7874 |
| 4290 | CALI,CT,PEF,CPOR_SM | 4 | 16 | 0.6961 | 0.0347 | 0.7281 |
| 4326 | CALI,DRHO,RT,CT,PEF,CPOR_SM | 6 | 18 | 0.7025 | 0.0573 | 0.7555 |
| 4550 | CALI,DRHO,CT,PEF,MSFL,CPOR_SM | 6 | 21 | 0.6939 | 0.0438 | 0.7344 |

---

## Output Files

### Tables

- **T1**: `/home/moayad/predict-permeability/results/phase3_feature_selection/analysis/tables/pareto_optimal_subsets.csv`
- **T2**: `/home/moayad/predict-permeability/results/phase3_feature_selection/analysis/tables/shap_grouped_feature_importance.csv`
- **T3**: `/home/moayad/predict-permeability/results/phase3_feature_selection/analysis/tables/shap_vs_cost.csv`

### Figures

- **F1**: `/home/moayad/predict-permeability/results/phase3_feature_selection/analysis/figures/pdf/shap_global_bar.pdf`
- **F2**: `/home/moayad/predict-permeability/results/phase3_feature_selection/analysis/figures/pdf/shap_beeswarm.pdf`
- **F3**: `/home/moayad/predict-permeability/results/phase3_feature_selection/analysis/figures/pdf/shap_per_well_heatmap.pdf`
- **F4_MAE**: `/home/moayad/predict-permeability/results/phase3_feature_selection/analysis/figures/pdf/pareto_frontier_MAE.pdf`
- **F4_MAE_log**: `/home/moayad/predict-permeability/results/phase3_feature_selection/analysis/figures/pdf/pareto_frontier_MAE_log.pdf`
- **F4_R2**: `/home/moayad/predict-permeability/results/phase3_feature_selection/analysis/figures/pdf/pareto_frontier_R2.pdf`
- **F4_R2_log**: `/home/moayad/predict-permeability/results/phase3_feature_selection/analysis/figures/pdf/pareto_frontier_R2_log.pdf`
- **F4_RMSE**: `/home/moayad/predict-permeability/results/phase3_feature_selection/analysis/figures/pdf/pareto_frontier_RMSE.pdf`
- **F4_RMSE_log**: `/home/moayad/predict-permeability/results/phase3_feature_selection/analysis/figures/pdf/pareto_frontier_RMSE_log.pdf`
- **F5**: `/home/moayad/predict-permeability/results/phase3_feature_selection/analysis/figures/pdf/pareto_frontier_retune.pdf`
- **F6**: `/home/moayad/predict-permeability/results/phase3_feature_selection/analysis/figures/pdf/pareto_validation.pdf`
- **F7**: `/home/moayad/predict-permeability/results/phase3_feature_selection/analysis/figures/pdf/feature_frequency_pareto.pdf`
