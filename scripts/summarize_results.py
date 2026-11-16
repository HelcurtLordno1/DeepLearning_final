"""Read the locked run lists; never choose a model using its test score."""

import bootstrap  # Resolve src/ and the project root.

from pathlib import Path
import json
import pandas as pd
from utils import ROOT


def main():
    rows, forecast_predictions, plate_predictions = [], [], []
    for task in ["task1", "task2", "task3"]:
        lock = ROOT/"run"/task/"locked_test_runs.json"
        if not lock.exists(): continue
        for relative in json.loads(lock.read_text(encoding="utf-8")):
            path = ROOT/relative
            config = json.loads((path/"config.json").read_text(encoding="utf-8"))
            metadata = json.loads((path/"metadata.json").read_text(encoding="utf-8"))
            base = {"task": task, "name": config["name"], "seed": config["seed"],
                    "run": relative, "parameters": metadata["parameters"],
                    "training_seconds": metadata["training_seconds"], "best_epoch": metadata["best_epoch"]}
            predictions = pd.read_csv(path/"test_predictions.csv")
            predictions.insert(0, "seed", config["seed"])
            predictions.insert(0, "model", config["name"])
            predictions.insert(0, "task", task)
            if task == "task3":
                predictions["image_id"] = predictions["image_path"].map(lambda value: Path(value).name)
                plate_predictions.append(predictions)
                scores = json.loads((path/"test_metrics.json").read_text(encoding="utf-8"))
                rows.append({**base, "model": "model", "horizon": "plate", **scores})
            else:
                forecast_predictions.append(predictions)
                table = pd.read_csv(path/"test_metrics.csv")
                rows.extend({**base, **r} for r in table.to_dict("records"))
    frame = pd.DataFrame(rows)
    output = ROOT/"run"/"summary"
    output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output/"all_test_metrics.csv", index=False)
    if forecast_predictions:
        pd.concat(forecast_predictions, ignore_index=True).to_csv(output/"forecast_test_predictions.csv", index=False)
    if plate_predictions:
        pd.concat(plate_predictions, ignore_index=True).to_csv(output/"plate_test_predictions.csv", index=False, encoding="utf-8")
    if frame.empty: return
    selections = json.loads((ROOT/"run"/"selected_configs.json").read_text(encoding="utf-8"))
    selected = pd.concat([frame[(frame["task"] == task) & (frame["name"] == item["config"]["name"]) & (frame["model"] == "model")]
                          for task, item in selections.items()], ignore_index=True)
    metrics = [m for m in ["R2", "NSE", "MAE", "RMSE", "plate_accuracy", "character_accuracy", "CER"] if m in frame]
    summary = selected.groupby(["task", "name", "horizon"])[metrics].agg(["count", "mean", "std"])
    summary.to_csv(output/"selected_models_seed_summary.csv")
    # Keep the old three-seed table scoped to the forecasting experiments.
    summary.loc[summary.index.get_level_values('task').isin(['task1','task2'])].to_csv(output/"selected_models_three_seeds.csv")
    lines = ["# Actual experiment results", "", "All model choices used validation only. Runtime: Windows Python 3.12 and CUDA.",
             "", "See all_test_metrics.csv for baselines and every horizon; selected_models_seed_summary.csv contains actual count/mean/sample-standard-deviation. A single seed has no estimated seed standard deviation.", ""]
    pending = [task for task in ["task1", "task2", "task3"] if task not in selections]
    if pending:
        lines.extend(["Unfinished final experiments: "+", ".join(pending)+". Completed validation trials are not final test/three-seed results.", ""])
    if (ROOT/"run/safety/INCIDENT.md").exists():
        lines.extend(["Long training was stopped after the laptop shutdown. See [incident and recovery limits](../safety/INCIDENT.md). Original run settings/source snapshots remain unchanged.", ""])
    for task, item in selections.items():
        subset = selected[(selected["task"] == task) & selected["horizon"].isin(["macro", "plate"])]
        lines.extend(["## "+task+" — "+item["config"]["name"], "", "Completed seeds: "+", ".join(map(str, sorted(subset["seed"].unique()))), ""])
        for metric in metrics:
            values = subset[metric].dropna()
            if len(values)>1:
                lines.append(f"- {metric}: {values.mean():.6f} ± {values.std(ddof=1):.6f} (n={len(values)})")
            elif len(values)==1:
                lines.append(f"- {metric}: {values.iloc[0]:.6f} (n=1; no seed standard deviation)")
        lines.append("")
    lines.extend(["Task 3 auxiliary one-hot metrics are categorical diagnostics, not numeric plate regression.",
                  "Task 3 uses annotation-derived crops and a custom grouped Base subset, not end-to-end plate detection.",
                  "Forecast R2 and NSE use the same residual-based formula. Task 2 macro and pooled rows are distinct."])
    (output/"RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    print("Saved actual comparisons to run/summary/", flush=True)


if __name__ == "__main__":
    main()
