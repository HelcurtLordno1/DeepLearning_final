# Measured test benchmarks

These tables were exported from the completed local experiments in `run/summary/`. Publishing them did not retrain the models or change the test predictions.

| Task | Final result | Completed seeds |
| --- | --- | --- |
| 1: OT at t+2 | Test RMSE 0.846787 ± 0.007900; 10.38% lower than persistence | 42, 123, 2026 |
| 2: OT at t+1 to t+5 | Test macro RMSE 0.982309 ± 0.017116; 13.55% lower than persistence | 42, 123, 2026 |
| 3: seven-character plate recognition | Test full-plate 95.322%; character 99.148%; CER 0.846% | One fine-tuning seed: 42 |

- [All test metrics](all_test_metrics.csv): locked model comparisons, forecasting baselines, individual seeds and horizons.
- [Selected-model seed summary](selected_models_seed_summary.csv): actual counts, means and sample standard deviations. A single seed has no seed standard deviation.
- [Three-seed forecasting summary](selected_models_three_seeds.csv): Task 1–2 only.
- [Task 3 comparison](../task3/test_benchmark.csv): attention baseline 62.764% → aligned 80.540% → refined 95.322% full-plate accuracy.
- [Result audit](../verification/results_audit.json) and [notebook checks](../verification/presentation_checks.json).

All configuration/checkpoint choices used validation only. Task 3 uses a custom plate-string-grouped 50,000-image CCPD-Base subset with annotation-derived crops. It is recognition with known plate location, rather than full-image detection or the official CCPD challenge protocol. The final model is warm-started through baseline → aligned → refined; the comparison combines architecture changes with additional fine-tuning.

R² and NSE use the same residual-based definition. Task 2 macro averages differ from pooled metrics. Task 3's decoded one-hot R²/NSE/MAE/RMSE are auxiliary categorical diagnostics; its primary metrics are full-plate accuracy, character accuracy and Levenshtein CER.

Checkpoints, source snapshots, prediction CSVs, training histories and machine-monitor logs remain in local `run/`, which is excluded from Git. Paths in the exported tables refer to those local artifacts. The repository includes saved notebook outputs and report figures for reviewing the completed results without those files.
