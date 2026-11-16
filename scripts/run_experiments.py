"""Run validation comparisons, lock choices, then evaluate the held-out tests."""

import bootstrap  # Resolve src/ and the project root.

from pathlib import Path
import gc
import json
import importlib
import pandas as pd
import torch
from utils import ROOT, save_json, file_hash, safe_training_config

# One GPU experiment at a time. Every run uses the full fixed data split.
TASKS = ["task1", "task2", "task3"]
FINAL_SEEDS = [42, 123, 2026]


def completed_run(task, config):
    for path in sorted((ROOT / "run" / task / config["name"] / str(config["seed"])).glob("*/config.json"), reverse=True):
        folder = path.parent
        if not (folder / "validation_metrics.csv").exists() and not (folder / "validation_metrics.json").exists():
            continue
        if json.loads(path.read_text(encoding="utf-8")) != config:
            continue
        if all(file_hash(folder/"source"/name) == file_hash(current) for name, current in
               [("utils.py", ROOT/"src/utils.py"), (task+".py", ROOT/task/"code"/(task+".py"))]):
            return folder
    return None


def run_one(task, settings):
    module = importlib.import_module(task + ".code." + task)
    config = safe_training_config({**module.BASE_CONFIG, **settings})
    path = completed_run(task, config)
    if path is None:
        path = module.run_experiment(config, evaluate_test=False)
    else:
        print("Reusing completed validation run:", path.relative_to(ROOT), flush=True)
    gc.collect(); torch.cuda.empty_cache()
    return path


def candidates(task):
    if task == "task1":
        return [
            {"name": "rnn64", "cell": "RNN", "attention": False, "residual": False},
            {"name": "lstm64", "attention": False, "residual": False},
            {"name": "lstm64_attention", "attention": True, "residual": False},
            {"name": "lstm64_residual", "attention": False, "residual": True},
            {"name": "lstm64_attention_residual", "attention": True, "residual": True},
            {"name": "lstm128_attention_residual", "hidden_size": 128, "attention": True, "residual": True},
            {"name": "lstm64_attention_residual_OT", "univariate": True, "attention": True, "residual": True}]
    if task == "task2":
        return [
            {"name": "seq64", "attention": False, "residual": False},
            {"name": "seq64_attention", "attention": True, "residual": False},
            {"name": "seq64_residual", "attention": False, "residual": True},
            {"name": "seq64_attention_residual", "attention": True, "residual": True},
            {"name": "seq128_attention_residual", "hidden_size": 128, "attention": True, "residual": True},
            {"name": "seq64_attention_residual_lr3e4", "learning_rate": .0003},
            {"name": "seq64_attention_residual_OT", "univariate": True}]
    return [
        {"name": "cnn_gru", "attention": False, "position": False, "augment": False},
        {"name": "cnn_gru_attention", "attention": True, "position": False, "augment": False},
        {"name": "cnn_gru_attention_position", "attention": True, "position": True, "augment": False},
        {"name": "cnn_gru_attention_position_aug", "attention": True, "position": True, "augment": True},
        {"name": "cnn_gru_attention_position_aug_rectified", "attention": True, "position": True, "augment": True, "crop_mode": "perspective"}]


def main():
    raise RuntimeError("Long automatic sweeps are disabled after the laptop shutdown. Use run_experiments.ps1 for one monitored recovery run.")


def full_sweep_reference():
    # Retain the original comparison procedure for later review; not auto-executed.
    selections, records = {}, []
    for task in TASKS:
        print("\nVALIDATION EXPERIMENTS:", task, flush=True)
        paths = [run_one(task, {**config, "seed": 42}) for config in candidates(task)]
        if task == "task3":
            # Optional small dense context preserves image-column order.
            scores = [json.loads((p/"metadata.json").read_text(encoding="utf-8"))["validation_score"] for p in paths]
            if max(scores) < .90:
                paths.append(run_one(task, {"name": "cnn_gru_attention_position_flat",
                    "seed": 42, "position": True, "flat_context": True, "augment": False}))
        metadata = [json.loads((p/"metadata.json").read_text(encoding="utf-8")) for p in paths]
        if task == "task3":
            best_index = max(range(len(paths)), key=lambda i: (metadata[i]["validation_score"], -metadata[i]["validation_CER"]))
        else:
            best_index = min(range(len(paths)), key=lambda i: metadata[i]["validation_score"])
        best_path = paths[best_index]
        best_config = json.loads((best_path/"config.json").read_text(encoding="utf-8"))
        save_json(ROOT/task/"selected_config.json", best_config)
        selections[task] = {"config": best_config, "seed42_run": str(best_path.relative_to(ROOT)),
                             "selection_metric": "plate_accuracy_then_CER" if task == "task3" else "macro_RMSE",
                             "selection_uses": "validation only"}
        save_json(ROOT/"run"/"selected_configs.json", selections)
        for p, info in zip(paths, metadata):
            records.append({"task": task, "run": str(p.relative_to(ROOT)),
                "name": json.loads((p/"config.json").read_text(encoding="utf-8"))["name"],
                "seed": 42, "validation_score": info["validation_score"],
                "selected": p == best_path})
        pd.DataFrame(records).to_csv(ROOT/"run"/"validation_comparisons.csv", index=False)
        # Additional seeds use the same locked configuration.
        for seed in FINAL_SEEDS[1:]:
            p = run_one(task, {**best_config, "seed": seed})
            paths.append(p)
        # All model settings for this task are now locked before any test metrics.
        save_json(ROOT/"run"/task/"locked_test_runs.json", [str(p.relative_to(ROOT)) for p in paths])
        module = importlib.import_module(task + ".code." + task)
        for path in paths:
            test_file = path / ("test_metrics.json" if task == "task3" else "test_metrics.csv")
            if not test_file.exists():
                module.evaluate_run(path)
        gc.collect(); torch.cuda.empty_cache()
    from summarize_results import main as summarize
    summarize()


if __name__ == "__main__":
    main()
