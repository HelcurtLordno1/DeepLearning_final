# Measured test benchmarks — Tasks 2 and 3

The active submission follows the lecturer's revised scope. Task numbers are retained. These exports come from completed local experiments; presentation changes do not retrain or reselect any model.

| Task | Final test result | Completed seeds |
| --- | --- | --- |
| 2: OT at t+1 through t+5 | Macro RMSE 0.982309 ± 0.017116; 13.55% below persistence; macro R²/NSE 0.874910 | 42, 123, 2026 |
| 3: seven-character plate recognition | Full-plate 95.322%; character 99.148%; CER 0.846% | One fine-tuning seed: 42 |

- [All active test metrics](all_test_metrics.csv): models, baselines, seeds and horizons.
- [Actual seed counts/means/sample SD](selected_models_seed_summary.csv).
- [Task 2 three-seed summary](selected_models_three_seeds.csv).
- [Task 3 stages](../task3/test_benchmark.csv): 62.764% → 80.540% → 95.322% full-plate accuracy.
- [Result audit](../verification/results_audit.json) and [notebook checks](../verification/presentation_checks.json).

Task 2 macro metrics average individual horizon scores; pooled metrics combine predictions before scoring. Its t+1 R² is about 0.9481, distinct from its five-horizon macro R². R²/NSE use the same residual-based definition and are not classification accuracy.

Task 3 uses a custom plate-grouped CCPD-Base cohort and annotation-derived crops, rather than full-image detection or the official challenge protocol. The baseline → aligned → refined comparison combines architecture changes and additional warm-start training. No multi-seed standard deviation is asserted for the final OCR model. One-hot regression-style metrics are auxiliary categorical diagnostics.

Full checkpoints, prediction CSVs, source snapshots and logs remain in local `run/`. Historical Task 1 artifacts remain locally preserved and are outside the current submission. Paths in the tables refer to local experiment directories.
