# Phase 2 Model Selection - Mean Performance Analysis

Generated: 2026-04-14 14:26:59

## Methodology

- **Selection criterion**: mean cross-fold `RMSE_log`
- **Primary visual**: boxplot + fold points (LOWO wells) per metric
- **Metric scope**: `log`
- **Confidence level (descriptive)**: 95%
- **t-critical (df=6)**: 2.447
- **CI role**: CI columns are descriptive only and do not drive ranking

**Selection rule**: configurations are sorted by diagnostic mean, and the
top-ranked configuration is promoted to Phase 3.

## Dataset Overview

- **Configurations**: 60 (4 variants x 15 models)
- **Folds (LOWO CV)**: 7

---

## Level 1: Dataset Variant View

Best variant (best average mean): **with_outlier_default_adaptive**

| variant_rank | variant | avg_mean | std_mean | n_models |
| --- | --- | --- | --- | --- |
| 1 | with_outlier_default_adaptive | 0.712 | 0.053 | 15 |
| 2 | with_outlier_default_linear | 0.712 | 0.052 | 15 |
| 3 | without_outlier_adaptive | 0.713 | 0.058 | 15 |
| 4 | without_outlier_linear | 0.717 | 0.058 | 15 |

**All Models:** see `variant_boxplot_all_log.png`

**Top 8 Models Per Variant:** see `variant_boxplot_top8_log.png`

---

## Level 2: Model Robustness

| model_rank | model | avg_mean | std_mean | n_variants |
| --- | --- | --- | --- | --- |
| 1 | ExtraTrees | 0.674 | 0.001 | 4 |
| 2 | LightGBM | 0.675 | 0.003 | 4 |
| 3 | CatBoost | 0.677 | 0.002 | 4 |
| 4 | XGBoost | 0.677 | 0.003 | 4 |
| 5 | RandomForest | 0.679 | 0.001 | 4 |
| 6 | HistGradientBoosting | 0.683 | 0.004 | 4 |
| 7 | Bagging | 0.684 | 0.001 | 4 |
| 8 | AdaBoost | 0.687 | 0.001 | 4 |
| 9 | GradientBoosting | 0.695 | 0.003 | 4 |
| 10 | Lasso | 0.695 | 0.003 | 4 |
| 11 | ElasticNet | 0.699 | 0.001 | 4 |
| 12 | DecisionTree | 0.739 | 0.007 | 4 |
| 13 | MLP | 0.796 | 0.012 | 4 |
| 14 | KNeighbors | 0.797 | 0.006 | 4 |
| 15 | Ridge | 0.846 | 0.008 | 4 |

---

## Level 3: Configuration Selection

Best configuration: **ExtraTrees** on `without_outlier_adaptive`

| overall_rank | variant | model | RMSE_log_mean | MAE_log_mean | R2_log_mean | RMSE_log_ci_lower | MAE_log_ci_lower | R2_log_ci_lower | RMSE_log_ci_upper | MAE_log_ci_upper | R2_log_ci_upper |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | without_outlier_adaptive | ExtraTrees | 0.6722 | 0.5427 | 0.6703 | 0.6302 | 0.5000 | 0.5880 | 0.7142 | 0.5853 | 0.7527 |
| 2 | without_outlier_adaptive | LightGBM | 0.6727 | 0.5462 | 0.6688 | 0.6231 | 0.4916 | 0.5804 | 0.7223 | 0.6008 | 0.7571 |
| 3 | with_outlier_default_adaptive | ExtraTrees | 0.6735 | 0.5427 | 0.6692 | 0.6266 | 0.4972 | 0.5865 | 0.7205 | 0.5883 | 0.7519 |
| 4 | with_outlier_default_adaptive | LightGBM | 0.6736 | 0.5446 | 0.6691 | 0.6263 | 0.4906 | 0.5866 | 0.7209 | 0.5987 | 0.7515 |
| 5 | with_outlier_default_linear | XGBoost | 0.6739 | 0.5486 | 0.6697 | 0.6394 | 0.5122 | 0.5947 | 0.7085 | 0.5850 | 0.7447 |
| 6 | without_outlier_linear | ExtraTrees | 0.6746 | 0.5455 | 0.6675 | 0.6286 | 0.4993 | 0.5819 | 0.7207 | 0.5917 | 0.7531 |
| 7 | without_outlier_adaptive | CatBoost | 0.6747 | 0.5500 | 0.6665 | 0.6272 | 0.4941 | 0.5765 | 0.7223 | 0.6059 | 0.7565 |
| 8 | with_outlier_default_linear | LightGBM | 0.6748 | 0.5490 | 0.6678 | 0.6307 | 0.4965 | 0.5855 | 0.7188 | 0.6016 | 0.7501 |
| 9 | with_outlier_default_linear | ExtraTrees | 0.6748 | 0.5437 | 0.6678 | 0.6313 | 0.5011 | 0.5845 | 0.7183 | 0.5864 | 0.7512 |
| 10 | without_outlier_adaptive | XGBoost | 0.6763 | 0.5475 | 0.6675 | 0.6358 | 0.5074 | 0.5928 | 0.7169 | 0.5877 | 0.7423 |
| 11 | with_outlier_default_linear | CatBoost | 0.6771 | 0.5501 | 0.6669 | 0.6327 | 0.4998 | 0.5908 | 0.7215 | 0.6005 | 0.7430 |
| 12 | with_outlier_default_adaptive | CatBoost | 0.6775 | 0.5526 | 0.6644 | 0.6386 | 0.5077 | 0.5780 | 0.7164 | 0.5975 | 0.7508 |
| 13 | without_outlier_adaptive | RandomForest | 0.6781 | 0.5553 | 0.6650 | 0.6390 | 0.5150 | 0.5846 | 0.7172 | 0.5957 | 0.7453 |
| 14 | without_outlier_linear | XGBoost | 0.6783 | 0.5549 | 0.6642 | 0.6360 | 0.5109 | 0.5808 | 0.7205 | 0.5990 | 0.7476 |
| 15 | with_outlier_default_linear | RandomForest | 0.6788 | 0.5542 | 0.6646 | 0.6302 | 0.5032 | 0.5842 | 0.7273 | 0.6052 | 0.7449 |

---

## Output Files

### Tables

- **T1**: `D:\moeman\research\bayan\predict-permeability\github-codebase\results\phase2_model_selection\analysis\tables\variant_mean_summary_log.csv`
- **T2**: `D:\moeman\research\bayan\predict-permeability\github-codebase\results\phase2_model_selection\analysis\tables\model_mean_summary_log.csv`
- **T3**: `D:\moeman\research\bayan\predict-permeability\github-codebase\results\phase2_model_selection\analysis\tables\overall_mean_ranking_log.csv`
- **T5_MAE_log**: `D:\moeman\research\bayan\predict-permeability\github-codebase\results\phase2_model_selection\analysis\tables\per_well_MAElog_log.csv`
- **T5_R2_log**: `D:\moeman\research\bayan\predict-permeability\github-codebase\results\phase2_model_selection\analysis\tables\per_well_R2log_log.csv`
- **T5_RMSE_log**: `D:\moeman\research\bayan\predict-permeability\github-codebase\results\phase2_model_selection\analysis\tables\per_well_RMSElog_log.csv`
- **T6**: `D:\moeman\research\bayan\predict-permeability\github-codebase\results\phase2_model_selection\analysis\tables\runtime_summary_log.csv`

### Figures

- **boxplot_MAE_log**: `D:\moeman\research\bayan\predict-permeability\github-codebase\results\phase2_model_selection\analysis\figures\pdf\boxplot_MAElog_log.pdf`
- **boxplot_R2_log**: `D:\moeman\research\bayan\predict-permeability\github-codebase\results\phase2_model_selection\analysis\figures\pdf\boxplot_R2log_log.pdf`
- **boxplot_RMSE_log**: `D:\moeman\research\bayan\predict-permeability\github-codebase\results\phase2_model_selection\analysis\figures\pdf\boxplot_RMSElog_log.pdf`
- **heatmap_MAE_log**: `D:\moeman\research\bayan\predict-permeability\github-codebase\results\phase2_model_selection\analysis\figures\pdf\per_well_heatmap_MAElog_log.pdf`
- **heatmap_R2_log**: `D:\moeman\research\bayan\predict-permeability\github-codebase\results\phase2_model_selection\analysis\figures\pdf\per_well_heatmap_R2log_log.pdf`
- **heatmap_RMSE_log**: `D:\moeman\research\bayan\predict-permeability\github-codebase\results\phase2_model_selection\analysis\figures\pdf\per_well_heatmap_RMSElog_log.pdf`
- **model_boxplot**: `D:\moeman\research\bayan\predict-permeability\github-codebase\results\phase2_model_selection\analysis\figures\pdf\model_robustness_log.pdf`
- **scatter_MAE_log**: `D:\moeman\research\bayan\predict-permeability\github-codebase\results\phase2_model_selection\analysis\figures\pdf\mean_vs_std_MAElog_log.pdf`
- **scatter_R2_log**: `D:\moeman\research\bayan\predict-permeability\github-codebase\results\phase2_model_selection\analysis\figures\pdf\mean_vs_std_R2log_log.pdf`
- **scatter_RMSE_log**: `D:\moeman\research\bayan\predict-permeability\github-codebase\results\phase2_model_selection\analysis\figures\pdf\mean_vs_std_RMSElog_log.pdf`
- **variant_boxplot_all**: `D:\moeman\research\bayan\predict-permeability\github-codebase\results\phase2_model_selection\analysis\figures\pdf\variant_boxplot_all_log.pdf`
- **variant_boxplot_top8**: `D:\moeman\research\bayan\predict-permeability\github-codebase\results\phase2_model_selection\analysis\figures\pdf\variant_boxplot_top8_log.pdf`
