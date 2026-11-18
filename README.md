# Attention-Based Multi-Step Forecasting and License Plate Recognition

Deep Learning final project — **Group 6, DSEB 65A**. Instructor: **Nguyễn Kim Ngân**.

The lecturer’s updated scope requires **Task 2 and Task 3 only**. Original task numbers are retained. Historical Task 1 files remain local and are excluded from Git.

## Tasks and measured results

| Task | Problem and model | Locked test result |
| --- | --- | --- |
| 2 | ETTh1 oil temperature at `t+1 … t+5`, using an LSTM Encoder–Decoder with attention and residual forecasting | Macro RMSE **0.982309 ± 0.017116**, macro R²/NSE **0.874910**, 3 seeds |
| 3 | Seven-character license plate recognition with a spatial CNN encoder, GRU decoder and visual attention | Full-plate accuracy **95.322%**, character accuracy **99.148%**, CER **0.846%**, one fine-tuning seed |

Task 2 reduces macro RMSE by **13.55%** against persistence. Its t+1 R² is approximately **0.9481**, while the five-horizon macro R² is **0.8749**. These are regression scores, distinct from Task 3’s recognition accuracy.

Task 3 correctly reads **7,132 of 7,482** test plates, improving full-plate accuracy by **32.56 percentage points** over the attention baseline. Its comparison combines architectural changes and additional warm-start training; it is not an isolated component ablation or a three-seed result.

## Notebooks

- [Task 2](task2/task2.ipynb): data checks, training-only EDA, exact target timestamps, visible model layers, validation selection, three-seed benchmarks and per-horizon errors.
- [Task 3](task3/task3.ipynb): image/label inspection, grouped splits, visible CNN–GRU layers, warm-start explanations, locked test results and error examples.

Choose **Final Project (Windows .venv)** and Run All. `TRAIN=False` displays saved results, loads checkpoints on CPU and performs small demonstration forwards. It does not retrain the models.

Task 2 retains its original simple model and training code. Two short analysis cells illustrate the 48-hour input/five-target window and persistence errors on training data. EDA motivates hypotheses; saved validation comparisons provide the evidence for selecting the model. A 48-hour lookback is a design choice, not a demonstrated optimum over unrun alternatives.

**On a fresh clone:** notebook outputs and reports can be viewed immediately. Running all cells also requires raw data and the local checkpoints/result tables under `run/`. The repository excludes `.venv`, raw images, processed caches and full experiment directories. Downloading data alone does not restore checkpoint-viewing mode.

## Reports

- [Complete English Word report](report/full_project/Group6_Final_Major_Assignment_Report_2026_EN.docx) and [PDF preview](report/full_project/Group6_Final_Major_Assignment_Report_2026_EN.pdf): Tasks 2 and 3, following the lecturer’s template structure.
- [Public benchmark tables](report/benchmark/README.md) and [independent result audit](report/verification/results_audit.json).
- [Task 3 detailed report](report/task3/task3_report.docx).
- [Three-column registration information](report/project_information.csv).

The revised report preserves the group cover and theoretical/practical/conclusion/reference structure. ETTh1 dataset and EDA material is included within Task 2 so it is self-contained. Earlier three-task reports are preserved in local `run/task23_revision/original/`.

## Download data

| Task | Dataset | Download |
| --- | --- | --- |
| 2 | ETTh1, hourly oil temperature and six electrical-load variables | [CSV](https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv) · [Authors](https://github.com/zhouhaoyi/ETDataset) |
| 3 | Fixed 50,000-image CCPD-Base cohort | [Official source](https://github.com/detectRecog/CCPD) · [Pinned train.zip mirror](https://huggingface.co/datasets/okita-souji/ccpd2019train/resolve/230a8d1092b7c8b0113541efe9861527275b5c04/train.zip?download=true) |

Use the checked selection and manifest in `data/splits/`; the source archive’s name `train.zip` does not define our partitions. ETTh1 uses chronological 70/15/15 splits with training-fitted scalers. CCPD has **35,043 / 7,475 / 7,482** images, grouped by complete plate text. Recognition uses annotation-derived crops and is not full-image detection. See [data documentation](data/raw/README.md).

## Layout

```text
task2/ task3/         Presentation notebooks and task READMEs
  code/              Models and training code
src/                 Data, metric and notebook helpers
scripts/             Preparation, guarded execution, verification and report tools
environment/         Windows dependency lists
data/splits/         Fixed manifests and vocabulary
report/              English report, figures, benchmark and verification exports
run/                 Local checkpoints, predictions, histories and safety logs
```

See [scripts/README.md](scripts/README.md) for the remaining tools. `task1/`, `assignments/`, `lessons_slides/`, `AGENTS.md` and `project_replan.md` are local-only. Ignoring Task 1 does not delete its local code or past results.

## Windows PowerShell

Use the project’s native Windows Python 3.12 `.venv`; no Linux/WSL Python environment is used for model work. On a new machine, first run `scripts/setup_environment.ps1`.

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\.venv\Scripts\Activate.ps1
$env:PYTHONUTF8 = "1"
python .\scripts\check_environment.py
```

When no training is active, verify the two notebooks and stored measurements:

```powershell
.\.venv\Scripts\python.exe .\scripts\verify_presentation.py
.\.venv\Scripts\python.exe .\scripts\verify_results.py
```

Read-only resource check:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\run_experiments.ps1
```

New training uses finite monitored sessions, batch ≤16, zero loader workers, two CPU threads, synchronized pauses, and RAM/commit/disk/GPU/hardware-event checks. Safety stops are not automatically retried. Frozen models are protected from continued development under the same run name. This revision adds presentation and analysis only; benchmark weights and splits are unchanged.
