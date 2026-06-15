# Phase 4 Generalization Validation Report

Generated: 2026-06-08 13:54:19

## Configuration

- **Model**: ExtraTrees
- **Variant**: `dataset/prepared_variants/all_sources_without_outlier_adaptive.csv`
- **HP budget**: 1000
- **Candidate subsets**: 4
- **Metric scope**: `log`

## Nested Summary (Log-space view)

| subset_id | subset_label | cost | n_outer_folds | RMSE_log_mean | RMSE_log_std | RMSE_log_ci | MAE_log_mean | R2_log_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 4550 | best_performer | 21 | 7 | 0.6610 | 0.0608 | 0.7172 | 0.5329 | 0.6812 |
| 8191 | full_feature_baseline | 31 | 7 | 0.6773 | 0.0496 | 0.7232 | 0.5467 | 0.6654 |
| 4096 | cpor_sm_only_baseline | 10 | 7 | 0.7230 | 0.0394 | 0.7595 | 0.5791 | 0.6155 |
| 640 | budget_wireline_only | 7 | 7 | 0.8103 | 0.0820 | 0.8862 | 0.6571 | 0.5123 |

## Outer-fold Winner Frequency

| selected_subset_id | selected_subset_label | selected_outer_folds |
| --- | --- | --- |
| 4550 | best_performer | 7 |

## Bias Comparison (Nested vs Selection-stage references)

| subset_id | subset_label | reference_source | nested_mean | reference_mean | delta_nested_minus_reference |
| --- | --- | --- | --- | --- | --- |
| 640 | budget_wireline_only | phase2_best_single_lowo | 0.8103 | 0.6722 | 0.1381 |
| 4096 | cpor_sm_only_baseline | phase2_best_single_lowo | 0.7230 | 0.6722 | 0.0508 |
| 4550 | best_performer | phase2_best_single_lowo | 0.6610 | 0.6722 | -0.0113 |
| 8191 | full_feature_baseline | phase2_best_single_lowo | 0.6773 | 0.6722 | 0.0051 |
| 640 | budget_wireline_only | phase3_single_lowo_retune | 0.8103 | 0.8042 | 0.0061 |
| 4096 | cpor_sm_only_baseline | phase3_single_lowo_retune | 0.7230 | 0.7171 | 0.0060 |
| 4550 | best_performer | phase3_single_lowo_retune | 0.6610 | 0.6523 | 0.0087 |

## Petrophysical / Regression Baseline Comparison (Nested vs classical baselines)

Negative `delta_nested_minus_baseline` (RMSE_log) means the nested ML pipeline beats the classical baseline at that operating point.

| baseline_label | subset_id | subset_label | nested_mean | baseline_mean | delta_nested_minus_baseline |
| --- | --- | --- | --- | --- | --- |
| Log-linear porosity-permeability (CPOR_SM) | 4096 | cpor_sm_only_baseline | 0.7230 | 0.7244 | -0.0013 |
| Log-linear porosity-permeability (CPOR_SM) | 4550 | best_performer | 0.6610 | 0.7244 | -0.0634 |
| Log-linear porosity-permeability (PHIT) | 640 | budget_wireline_only | 0.8103 | 0.8536 | -0.0433 |
| Timur baseline (CPOR_SM, SWT as Swirr) | 640 | budget_wireline_only | 0.8103 | 0.7918 | 0.0185 |
| Timur baseline (CPOR_SM, SWT as Swirr) | 4096 | cpor_sm_only_baseline | 0.7230 | 0.7918 | -0.0688 |
| Timur baseline (CPOR_SM, SWT as Swirr) | 4550 | best_performer | 0.6610 | 0.7918 | -0.1308 |
| Timur baseline (CPOR_SM, SWT as Swirr) | 8191 | full_feature_baseline | 0.6773 | 0.7918 | -0.1145 |
| Timur baseline (PHIT, SWT as Swirr) | 640 | budget_wireline_only | 0.8103 | 0.8856 | -0.0752 |

## Per-Zone Error Breakdown (held-out nested predictions)

RMSE_log / R2_log / MAE_log per geological zone, by operating point. `n` is the number of held-out samples in that zone.

| subset_id | subset_label | Zone | n | RMSE_log | R2_log | MAE_log |
| --- | --- | --- | --- | --- | --- | --- |
| 640 | budget_wireline_only | 2 | 1 | 0.8607 | nan | 0.8607 |
| 640 | budget_wireline_only | 3 | 44 | 0.9993 | -0.3472 | 0.7389 |
| 640 | budget_wireline_only | 4 | 507 | 0.7243 | 0.4740 | 0.5922 |
| 640 | budget_wireline_only | 5 | 806 | 0.6886 | 0.4224 | 0.5266 |
| 640 | budget_wireline_only | 6 | 531 | 0.9411 | 0.4395 | 0.7836 |
| 640 | budget_wireline_only | 7 | 281 | 0.8597 | 0.1352 | 0.7045 |
| 640 | budget_wireline_only | 8 | 102 | 0.9497 | 0.3527 | 0.8252 |
| 640 | budget_wireline_only | 9 | 12 | 0.4415 | -0.4118 | 0.3918 |
| 4096 | cpor_sm_only_baseline | 2 | 1 | 0.0281 | nan | 0.0281 |
| 4096 | cpor_sm_only_baseline | 3 | 44 | 0.7250 | 0.2908 | 0.6264 |
| 4096 | cpor_sm_only_baseline | 4 | 507 | 0.6046 | 0.6335 | 0.5041 |
| 4096 | cpor_sm_only_baseline | 5 | 806 | 0.6914 | 0.4178 | 0.5293 |
| 4096 | cpor_sm_only_baseline | 6 | 531 | 0.8411 | 0.5523 | 0.6915 |
| 4096 | cpor_sm_only_baseline | 7 | 281 | 0.7087 | 0.4124 | 0.5803 |
| 4096 | cpor_sm_only_baseline | 8 | 102 | 0.7356 | 0.6116 | 0.5548 |
| 4096 | cpor_sm_only_baseline | 9 | 12 | 0.3871 | -0.0854 | 0.3326 |
| 4550 | best_performer | 2 | 1 | 0.1100 | nan | 0.1100 |
| 4550 | best_performer | 3 | 44 | 0.8046 | 0.1265 | 0.7002 |
| 4550 | best_performer | 4 | 507 | 0.5690 | 0.6754 | 0.4769 |
| 4550 | best_performer | 5 | 806 | 0.5988 | 0.5633 | 0.4738 |
| 4550 | best_performer | 6 | 531 | 0.7051 | 0.6854 | 0.5658 |
| 4550 | best_performer | 7 | 281 | 0.7645 | 0.3162 | 0.6382 |
| 4550 | best_performer | 8 | 102 | 0.6963 | 0.6519 | 0.5574 |
| 4550 | best_performer | 9 | 12 | 0.3887 | -0.0939 | 0.3399 |
| 8191 | full_feature_baseline | 2 | 1 | 0.1911 | nan | 0.1911 |
| 8191 | full_feature_baseline | 3 | 44 | 0.8031 | 0.1298 | 0.6836 |
| 8191 | full_feature_baseline | 4 | 507 | 0.5869 | 0.6546 | 0.4890 |
| 8191 | full_feature_baseline | 5 | 806 | 0.5984 | 0.5639 | 0.4765 |
| 8191 | full_feature_baseline | 6 | 531 | 0.7432 | 0.6504 | 0.5994 |
| 8191 | full_feature_baseline | 7 | 281 | 0.8069 | 0.2382 | 0.6640 |
| 8191 | full_feature_baseline | 8 | 102 | 0.6763 | 0.6717 | 0.5171 |
| 8191 | full_feature_baseline | 9 | 12 | 0.3599 | 0.0617 | 0.3211 |

## Structural-Covariate Ablation (nested, delta vs full)

Nested-LOWO RMSE_log change when each structural covariate is dropped, relative to the canonical full-covariate nested run (reused, not recomputed). Positive `delta_vs_full` means dropping the covariate **hurt** generalization. The CSV `ablation_comparison.csv` carries deltas for all metrics, not just RMSE_log.

| subset_id | subset_label | config | dropped | full_mean | config_mean | delta_vs_full |
| --- | --- | --- | --- | --- | --- | --- |
| 640 | budget_wireline_only | no_source | Source | 0.8103 | 0.8183 | +0.0079 |
| 640 | budget_wireline_only | no_zone | Zone | 0.8103 | 0.8052 | -0.0051 |
| 640 | budget_wireline_only | no_depth | DEPTH | 0.8103 | 0.8149 | +0.0046 |
| 640 | budget_wireline_only | no_all_three | DEPTH,Source,Zone | 0.8103 | 0.8165 | +0.0061 |
| 4096 | cpor_sm_only_baseline | no_source | Source | 0.7230 | 0.7027 | -0.0204 |
| 4096 | cpor_sm_only_baseline | no_zone | Zone | 0.7230 | 0.7388 | +0.0158 |
| 4096 | cpor_sm_only_baseline | no_depth | DEPTH | 0.7230 | 0.7332 | +0.0102 |
| 4096 | cpor_sm_only_baseline | no_all_three | DEPTH,Source,Zone | 0.7230 | 0.7230 | -0.0000 |
| 4550 | best_performer | no_source | Source | 0.6610 | 0.6606 | -0.0003 |
| 4550 | best_performer | no_zone | Zone | 0.6610 | 0.6872 | +0.0263 |
| 4550 | best_performer | no_depth | DEPTH | 0.6610 | 0.6683 | +0.0073 |
| 4550 | best_performer | no_all_three | DEPTH,Source,Zone | 0.6610 | 0.7169 | +0.0560 |

The `no_zone` rows quantify the *aggregate* effect of removing Zone as a feature; the Per-Zone Error Breakdown above (and `per_zone_metrics_*.csv`) shows *where* that signal concentrates across zones.

### Ablated nested configurations (per subset x config)

| subset_id | config | dropped | RMSE_log_mean | RMSE_log_std | R2_log_mean |
| --- | --- | --- | --- | --- | --- |
| 640 | no_source | Source | 0.8183 | 0.0865 | 0.5037 |
| 640 | no_zone | Zone | 0.8052 | 0.0949 | 0.5147 |
| 640 | no_depth | DEPTH | 0.8149 | 0.0739 | 0.5088 |
| 640 | no_all_three | DEPTH,Source,Zone | 0.8165 | 0.1035 | 0.5017 |
| 4096 | no_source | Source | 0.7027 | 0.0442 | 0.6398 |
| 4096 | no_zone | Zone | 0.7388 | 0.0449 | 0.6027 |
| 4096 | no_depth | DEPTH | 0.7332 | 0.0498 | 0.6022 |
| 4096 | no_all_three | DEPTH,Source,Zone | 0.7230 | 0.0559 | 0.6160 |
| 4550 | no_source | Source | 0.6606 | 0.0616 | 0.6824 |
| 4550 | no_zone | Zone | 0.6872 | 0.0611 | 0.6567 |
| 4550 | no_depth | DEPTH | 0.6683 | 0.0597 | 0.6739 |
| 4550 | no_all_three | DEPTH,Source,Zone | 0.7169 | 0.0751 | 0.6304 |

## Output Files

### Tables

- **T1_nested_summary**: `results/phase4_generalization/run/nested_summary.csv`
- **T2_bias_comparison**: `results/phase4_generalization/run/bias_comparison.csv`
- **T3_selection_summary**: `results/phase4_generalization/run/selection_summary.csv`
- **T4_baseline_comparison**: `results/phase4_generalization/run/baseline_comparison.csv`
- **T5_per_zone_metrics**: `results/phase4_generalization/analysis/tables/per_zone_metrics_all.csv`
- **T5_per_zone_metrics_4096**: `results/phase4_generalization/analysis/tables/per_zone_metrics_4096.csv`
- **T5_per_zone_metrics_4550**: `results/phase4_generalization/analysis/tables/per_zone_metrics_4550.csv`
- **T5_per_zone_metrics_640**: `results/phase4_generalization/analysis/tables/per_zone_metrics_640.csv`
- **T5_per_zone_metrics_8191**: `results/phase4_generalization/analysis/tables/per_zone_metrics_8191.csv`
- **T6_ablation_comparison**: `results/phase4_generalization/run/ablation_comparison.csv`
- **T6_ablation_nested_summary**: `results/phase4_generalization/analysis/tables/ablation_nested_summary.csv`
- **T6_ablation_results**: `results/phase4_generalization/analysis/tables/ablation_nested_results_all.csv`

### Figures

- **A2_ablation_delta_rmse_log**: `results/phase4_generalization/analysis/figures/pdf/phase4_ablation_delta_rmse_log.pdf`
- **B1_baseline_rmse_log_boxplot**: `results/phase4_generalization/analysis/figures/pdf/phase4_baseline_rmse_log_boxplot.pdf`
- **F1_nested_frontier_RMSE_log**: `results/phase4_generalization/analysis/figures/pdf/phase4_nested_frontier_rmse_log.pdf`
- **F1_per_well_boxplot_RMSE_log**: `results/phase4_generalization/analysis/figures/pdf/phase4_outer_rmse_log_boxplot.pdf`
- **F1_per_well_heatmap_R2_log**: `results/phase4_generalization/analysis/figures/pdf/phase4_outer_r2_log_heatmap.pdf`
- **F1_per_well_heatmap_RMSE_log**: `results/phase4_generalization/analysis/figures/pdf/phase4_outer_rmse_log_heatmap.pdf`
- **G1_pred_vs_actual_grid**: `results/phase4_generalization/analysis/figures/pdf/phase4_pred_vs_actual_grid.pdf`
- **G2_pred_vs_actual_per_well_4096**: `results/phase4_generalization/analysis/figures/pdf/phase4_pred_vs_actual_per_well_4096.pdf`
- **G2_pred_vs_actual_per_well_4550**: `results/phase4_generalization/analysis/figures/pdf/phase4_pred_vs_actual_per_well_4550.pdf`
- **G2_pred_vs_actual_per_well_640**: `results/phase4_generalization/analysis/figures/pdf/phase4_pred_vs_actual_per_well_640.pdf`
- **G2_pred_vs_actual_per_well_8191**: `results/phase4_generalization/analysis/figures/pdf/phase4_pred_vs_actual_per_well_8191.pdf`
- **Z1_per_zone_rmse_log**: `results/phase4_generalization/analysis/figures/pdf/phase4_per_zone_rmse_log.pdf`
- **Z2_per_zone_r2_log**: `results/phase4_generalization/analysis/figures/pdf/phase4_per_zone_r2_log.pdf`
