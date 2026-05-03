# Phase 4 Generalization Validation Report

Generated: 2026-04-16 02:43:36

## Configuration

- **Model**: ExtraTrees
- **Variant**: `dataset/prepared_variants/all_sources_without_outlier_adaptive.csv`
- **HP budget**: 1000
- **Candidate subsets**: 3
- **Metric scope**: `log`

## Nested Summary (Log-space view)

| subset_id | subset_label | cost | n_outer_folds | RMSE_log_mean | RMSE_log_std | RMSE_log_ci | MAE_log_mean | R2_log_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 4550 | best_performer | 21 | 7 | 0.6591 | 0.0573 | 0.7122 | 0.5320 | 0.6831 |
| 4096 | cpor_sm_only_baseline | 10 | 7 | 0.7205 | 0.0431 | 0.7604 | 0.5785 | 0.6170 |
| 640 | budget_wireline_only | 7 | 7 | 0.8160 | 0.0819 | 0.8917 | 0.6614 | 0.5053 |

## Outer-fold Winner Frequency

| selected_subset_id | selected_subset_label | selected_outer_folds |
| --- | --- | --- |
| 4550 | best_performer | 7 |

## Bias Comparison (Nested vs Selection-stage references)

| subset_id | subset_label | reference_source | nested_mean | reference_mean | delta_nested_minus_reference |
| --- | --- | --- | --- | --- | --- |
| 640 | budget_wireline_only | phase2_best_single_lowo | 0.8160 | 0.6722 | 0.1438 |
| 4096 | cpor_sm_only_baseline | phase2_best_single_lowo | 0.7205 | 0.6722 | 0.0483 |
| 4550 | best_performer | phase2_best_single_lowo | 0.6591 | 0.6722 | -0.0131 |
| 640 | budget_wireline_only | phase3_single_lowo_retune | 0.8160 | 0.8065 | 0.0095 |
| 4096 | cpor_sm_only_baseline | phase3_single_lowo_retune | 0.7205 | 0.7204 | 0.0001 |
| 4550 | best_performer | phase3_single_lowo_retune | 0.6591 | 0.6552 | 0.0040 |

## Output Files

### Tables

- **T1_nested_summary**: `results/phase4_generalization/run/nested_summary.csv`
- **T2_bias_comparison**: `results/phase4_generalization/run/bias_comparison.csv`
- **T3_selection_summary**: `results/phase4_generalization/run/selection_summary.csv`

### Figures

- **F1_nested_frontier_RMSE_log**: `results/phase4_generalization/analysis/figures/pdf/phase4_nested_frontier_rmse_log.pdf`
- **F1_per_well_boxplot_RMSE_log**: `results/phase4_generalization/analysis/figures/pdf/phase4_outer_rmse_log_boxplot.pdf`
- **F1_per_well_heatmap_RMSE_log**: `results/phase4_generalization/analysis/figures/pdf/phase4_outer_rmse_log_heatmap.pdf`
