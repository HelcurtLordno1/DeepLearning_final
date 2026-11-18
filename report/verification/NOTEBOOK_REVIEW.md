# Task 2 and Task 3 presentation review

Active scope follows the lecturer’s updated instruction. Task numbers are retained.

| Notebook | Code cells | Markdown cells | Inline figures | Errors |
| --- | ---: | ---: | ---: | ---: |
| task2 | 19 | 28 | 7 | 0 |
| task3 | 19 | 25 | 5 | 0 |

Task 2 includes a 48-hour/five-target training example, training-only persistence errors, and a validation-based decision table. Figure-specific explanations identify the axes, panels, target markers and interpretation limits. Its model, optimizer, training function and frozen results are unchanged.

Both notebooks display detailed architecture diagrams with tensor shapes, recurrent state initialization, attention memory connections and the output path. The diagrams use the same drawing functions as the English report.

Task 3 explains spatial information and the warm-start chain. It remains a one-seed fine-tuned recognition model with annotation-derived crops.

Checks use fresh native Windows kernels with TRAIN=False, saved results and small CPU forward passes. No training or CUDA initialization is performed. Model/attention/training-step definitions are checked against the actual source.

Task 2 macro RMSE: 0.982309 ± 0.017116 (3 seeds), 13.55% below persistence; macro R²/NSE 0.874910. Task 3 full-plate accuracy: 95.322%; character accuracy: 99.148%; CER: 0.846% (one fine-tuning seed).

Evidence: presentation_checks.json, source_agreement.json, and ../verification/results_audit.json.
