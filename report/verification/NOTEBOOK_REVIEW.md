# Task 2 and Task 3 presentation review

Active scope follows the lecturer’s updated instruction. Task numbers are retained.

| Notebook | Code cells | Markdown cells | Inline figures | Errors |
| --- | ---: | ---: | ---: | ---: |
| task2 | 18 | 23 | 6 | 0 |
| task3 | 18 | 24 | 4 | 0 |

Task 2 adds a 48-hour/five-target training example, training-only persistence errors, and a validation-based decision table. Its model, optimizer, training function and frozen results are unchanged.

Task 3 adds concise explanations of spatial information and the warm-start chain. It remains a one-seed fine-tuned recognition model with annotation-derived crops.

Checks use fresh native Windows kernels with TRAIN=False, saved results and small CPU forward passes. No training or CUDA initialization is performed. Model/attention/training-step definitions are checked against the actual source.

Task 2 macro RMSE: 0.982309 ± 0.017116 (3 seeds), 13.55% below persistence; macro R²/NSE 0.874910. Task 3 full-plate accuracy: 95.322%; character accuracy: 99.148%; CER: 0.846% (one fine-tuning seed).

Evidence: presentation_checks.json, source_agreement.json, and ../verification/results_audit.json.
