# Downloaded datasets

Updated: 16 November 2026. The project data is stored on the D: drive.

## Storage check

Before downloading, D: had approximately **80.5 GiB free**. ETTh1 takes 2,589,657 bytes. Our 50,000 selected original CCPD images total **3,680,307,496 bytes (3.43 GiB)** before filesystem overhead. There is ample room for these datasets and later crops/checkpoints.

After downloading, auditing, and saving the split, D: had **76.88 GiB free**. The measured snapshot is in `data/storage_check.json`. C: was also checked and had about 108 GiB free; project datasets remain on D:.

For comparison, the complete official CCPD2019 archive is 12.3 GiB and its XZ index reports 24.1 GiB of uncompressed tar data. We checked that it would fit, then found a smaller Base mirror and fetched selected originals directly using HTTP byte ranges. The full archive is not retained. Preparation preserves at least 20 GiB free; future training should check storage again before making large copies.

## ETTh1 — downloaded and checked

- File: `etth1/ETTh1.csv`.
- Source: [authors' repository](https://github.com/zhouhaoyi/ETDataset), [direct CSV](https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv).
- Local source documentation: `etth1/SOURCE_README.md` and `etth1/LICENSE`.
- Observations: 17,420, one per hour, 2016-07-01 00:00 to 2018-06-26 19:00.
- Columns: date, six load measurements, and `OT` oil temperature.
- Checked: finite numeric values, unique timestamps, and uninterrupted hourly spacing.
- Details and SHA-256: `etth1/metadata.json`.

Use this CSV for Task 2 in the revised two-task submission. Fit scalers on training rows only. The plan specifies chronological raw-row counts of 12,194 / 2,613 / 2,613 before windowing.

For a concrete first sample, 48 input rows cover 2016-07-01 00:00 through 2016-07-02 23:00. Task 2 targets 2016-07-03 00:00 through 04:00 under the plan's working timing convention. The actual target values are recorded in `etth1/metadata.json`.

## CCPD-Base — downloaded and checked

- Original dataset and filename specification: [CCPD authors](https://github.com/detectRecog/CCPD).
- Image mirror: [okita-souji/ccpd2019train](https://huggingface.co/datasets/okita-souji/ccpd2019train/tree/230a8d1092b7c8b0113541efe9861527275b5c04), pinned revision `230a8d1092b7c8b0113541efe9861527275b5c04`. The official Google Drive download returned a quota-exceeded page.
- The mirror's `train.zip` contains 100,000 JPEGs. All 100,000 filenames were matched to the authors' official training list. Its full size is 5,476,210,226 bytes; we only fetch ranges containing our selection.
- Whole-ZIP SHA-256 published by the mirror: `26407b80254917256d7254d9df54dbe783bf1b5a7bf0f19311263cfea8b5a845`. We do not claim to have verified that whole-archive hash because we do not download the entire ZIP. Each selected image is checked against the ZIP member's CRC32, then receives its own local SHA-256 during preparation.
- Local source documentation: `ccpd/SOURCE_README.md`, `ccpd/LICENSE`, and `ccpd/official_splits/`.

`ccpd/ccpd_base/` contains the selected original vehicle JPEGs; filenames preserve the full text and crop coordinates. These are original-resolution images, not previously resized OCR crops. `ccpd/download_metadata.json` records the actual acquisition status and source details. Temporary full-archive attempts were stopped and their partial files removed after selecting this smaller download route.

To reduce transfer and storage, the selection samples **500 blocks of 100 consecutive JPEG entries** from the pinned ZIP, using Python `random.Random(42).sample(range(1000), 500)`. Selection happened before training, without using model results. It samples blocks rather than individual images; describe that method when documenting the subset. `data/splits/ccpd_download_selection.txt` fixes the exact 50,000 filenames.

This replaces the initial plan's sample from the combined 199,996-image Base pool with a sample from the 100,000-image official-training pool available in this mirror. The required task, complete seven-character labels, 50,000-image target, and independent grouped holdouts remain the same. Our custom train/validation/test partitions are created from this pool; the source word “train” does not mean all these images become our training set.

`scripts/prepare_data.py` reads the saved selection, decodes JPEG pixels, checks boxes and duplicate image content, and saves the custom split grouped by plate text. This is a coursework recognition subset, not the official CCPD challenge protocol. The checked split is included in Git; preserve it when reproducing the experiments.

To reproduce preparation after the verified archive is present, run from the project root in an environment with Pillow:

```powershell
.\.venv\Scripts\python.exe .\scripts\prepare_data.py
```

Preparation saves `data/splits/ccpd_subset.csv`, `data/splits/ccpd_vocabulary.json`, and `ccpd/preparation_report.json`. These supply the actual split counts; estimates in the original plan are not measured counts.

The completed audit found:

| Partition | Images | Distinct plate strings |
| --- | --- | --- |
| Train | 35,043 | 31,481 |
| Validation | 7,475 | 6,746 |
| Test | 7,482 | 6,747 |
| Total | 50,000 | 44,974 |

- All 50,000 files passed ZIP member CRC32 checks and full JPEG pixel decoding.
- Every image is width 720 × height 1160; all stored bounding boxes are inside the image.
- All labels have seven characters and use the saved 67-class vocabulary.
- No identical image bytes were found; plate strings have zero overlap between partitions.
- Every character appearing in validation/test labels appears in training labels.
- The longest image path at the original Windows project location was 215 characters; no path exceeded 259 characters. This check was recorded before training. The completed experiments subsequently used native Windows Python.
- Thirty labeled crops were manually reviewed. Readable characters looked consistent; some blur prevents complete visual checking. No difficult examples were excluded. Preview: `data/processed/plate_preview.jpg`.

The saved manifest's SHA-256 is `afa052053851c3006e32921e8979733eb32f0567850de2d81eb16cdff3e43b30`. Model training should reuse these splits. The source's official split is different; do not present our scores as the official benchmark.

For a future fresh download, obtain `train.zip` from the pinned mirror, place it in `data/raw/ccpd/`, and run the same preparation command with the saved selection. The helper can extract just those images from the ZIP. Keeping the full ZIP is optional and requires another 5.1 GiB.

The data audit above preceded model training. Current test results are published in [report/benchmark/](../../report/benchmark/README.md) and [report/task3/](../../report/task3/task3_report.md); full experiment artifacts remain in local `run/`. Leave original downloads unchanged and keep raw data out of Git.
