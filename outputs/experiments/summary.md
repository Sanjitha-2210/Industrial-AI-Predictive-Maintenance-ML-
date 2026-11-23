# DATF-Net experiments (FD001)

Seeds [42, 43, 44, 45, 46]; lambdas selected on validation: {'lambda_hi': 0.1, 'lambda_mono': 0.1, 'run': 'datf_lh0.1_lm0.1'}; test used only for final evaluation.

| Run | Params | Val MAE | Val RMSE | Test MAE | Test RMSE | Test NASA | Test RMSE (clipped truth) |
|---|---|---|---|---|---|---|---|
| abl_concat_fusion | 54,626 | 8.58 ± 0.22 | 11.44 ± 0.29 | 10.79 ± 0.71 | 14.92 ± 0.85 | 289 ± 38 | 13.52 ± 0.83 |
| abl_last_step_pooling | 47,714 | 8.28 ± 0.22 | 11.18 ± 0.17 | 10.55 ± 0.32 | 14.41 ± 0.61 | 268 ± 22 | 13.05 ± 0.59 |
| abl_lstm_encoder | 48,146 | 8.31 ± 0.31 | 11.50 ± 0.44 | 10.66 ± 0.43 | 14.51 ± 0.51 | 281 ± 19 | 13.22 ± 0.44 |
| abl_no_baseline | 54,626 | 10.09 ± 0.17 | 13.40 ± 0.23 | 11.28 ± 0.23 | 15.00 ± 0.45 | 312 ± 28 | 13.47 ± 0.49 |
| abl_no_degradation | 39,761 | 12.13 ± 0.24 | 16.05 ± 0.44 | 12.30 ± 0.51 | 16.62 ± 0.74 | 423 ± 67 | 15.18 ± 0.76 |
| abl_no_hi_supervision | 54,626 | 8.50 ± 0.30 | 11.30 ± 0.29 | 10.74 ± 0.50 | 14.95 ± 0.54 | 290 ± 25 | 13.49 ± 0.53 |
| abl_no_slope | 53,570 | 8.86 ± 0.18 | 11.86 ± 0.44 | 11.55 ± 0.38 | 15.50 ± 0.54 | 318 ± 22 | 14.20 ± 0.53 |
| datf_full | 54,626 | 8.61 ± 0.18 | 11.43 ± 0.27 | 10.66 ± 0.41 | 14.89 ± 0.59 | 286 ± 23 | 13.40 ± 0.60 |
| lstm_baseline | 60,801 | 10.58 ± 0.37 | 14.39 ± 0.39 | 11.65 ± 0.23 | 16.02 ± 0.35 | 427 ± 72 | 14.52 ± 0.44 |
| lstm_info_matched | 65,153 | 8.68 ± 0.41 | 11.59 ± 0.55 | 12.02 ± 0.72 | 16.34 ± 0.92 | 377 ± 41 | 15.03 ± 0.90 |

Protected single-seed baseline (outputs/metrics.json, unchanged): test MAE 11.82, RMSE 16.07, NASA 504.6.

## Paired bootstrap on the 100 test engines (RMSE difference, mean over seeds; negative = first is better)

| Comparison | ΔRMSE | 95% CI |
|---|---|---|
| datf_full_minus_lstm_baseline | -1.13 | [-3.25, +0.90] |
| lstm_info_matched_minus_lstm_baseline | +0.32 | [-1.80, +2.30] |
| datf_full_minus_lstm_info_matched | -1.45 | [-3.02, +0.01] |
| abl_no_baseline_minus_lstm_baseline | -1.01 | [-2.55, +0.51] |
| abl_concat_fusion_minus_datf_full | +0.03 | [-0.42, +0.50] |
| abl_last_step_pooling_minus_datf_full | -0.48 | [-1.19, +0.23] |
| abl_lstm_encoder_minus_datf_full | -0.38 | [-1.33, +0.58] |
| abl_no_baseline_minus_datf_full | +0.11 | [-1.50, +1.76] |
| abl_no_degradation_minus_datf_full | +1.73 | [-0.20, +3.70] |
| abl_no_hi_supervision_minus_datf_full | +0.06 | [-0.21, +0.33] |
| abl_no_slope_minus_datf_full | +0.61 | [-0.05, +1.24] |
