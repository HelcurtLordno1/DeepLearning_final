"""Recalculate reported scores from saved predictions and original labels."""

import bootstrap  # Resolve src/ and the project root.

from pathlib import Path
import ast
import json
import sys
import numpy as np
import pandas as pd
from utils import ROOT, save_json, file_hash


def numeric_scores(actual, predicted):
    a, p = np.asarray(actual, dtype="float64"), np.asarray(predicted, dtype="float64")
    squared_error = np.dot((a-p).ravel(), (a-p).ravel())
    variation = np.square(a-a.mean()).sum()
    return {"R2": 1-squared_error/variation, "NSE": 1-squared_error/variation,
            "MAE": np.abs(a-p).mean(), "RMSE": np.sqrt(squared_error/a.size)}


def assert_scores(record, recalculated, prefix=""):
    for key, value in recalculated.items():
        assert np.isclose(record[prefix+key], value, atol=1e-5, rtol=1e-5), (prefix+key, record[prefix+key], value)


def audit_forecast(path, task, raw):
    predictions = pd.read_csv(path/"test_predictions.csv", parse_dates=["origin_time", "target_time"])
    scores = pd.read_csv(path/"test_metrics.csv")
    horizons, expected_count = ([2], 2564) if task == "task1" else ([1, 2, 3, 4, 5], 2561)
    assert len(predictions) == expected_count*len(horizons)
    assert set(predictions["horizon"]) == set(horizons)
    delta_hours = (predictions["target_time"]-predictions["origin_time"]).dt.total_seconds()/3600
    np.testing.assert_array_equal(delta_hours, predictions["horizon"])
    original_targets = raw.reindex(predictions["target_time"])["OT"].to_numpy()
    np.testing.assert_allclose(original_targets, predictions["actual"], atol=1e-5)
    assert predictions["target_time"].min() >= raw.index[int(len(raw)*.85)]
    for model, column in [("model", "predicted"), ("persistence", "persistence"), ("daily_seasonal", "daily_seasonal")]:
        by_horizon = []
        for horizon in horizons:
            values = predictions[predictions["horizon"].eq(horizon)]
            assert len(values) == expected_count
            calculated = numeric_scores(values["actual"], values[column])
            row = scores[scores["model"].eq(model) & scores["horizon"].eq(f"t+{horizon}")].iloc[0]
            assert_scores(row, calculated)
            by_horizon.append(calculated)
        macro = pd.DataFrame(by_horizon).mean().to_dict()
        assert_scores(scores[scores["model"].eq(model) & scores["horizon"].eq("macro")].iloc[0], macro)
        pooled = numeric_scores(predictions["actual"], predictions[column])
        assert_scores(scores[scores["model"].eq(model) & scores["horizon"].eq("pooled")].iloc[0], pooled)


def string_distance(a, b):
    matrix = np.zeros((len(a)+1, len(b)+1), dtype=int)
    matrix[:, 0], matrix[0] = np.arange(len(a)+1), np.arange(len(b)+1)
    for i in range(1, len(a)+1):
        for j in range(1, len(b)+1):
            matrix[i, j] = min(matrix[i-1, j]+1, matrix[i, j-1]+1, matrix[i-1, j-1]+(a[i-1] != b[j-1]))
    return int(matrix[-1, -1])


def audit_plates(path, manifest):
    predictions = pd.read_csv(path/"test_predictions.csv")
    scores = json.loads((path/"test_metrics.json").read_text(encoding="utf-8"))
    test = manifest[manifest["split"].eq("test")].set_index("image_path")
    assert len(predictions) == 7482 and predictions["image_path"].is_unique
    assert set(predictions["image_path"]) == set(test.index)
    truth = test.loc[predictions["image_path"], "text"].tolist()
    assert truth == predictions["true_text"].tolist()
    predicted = predictions["predicted_text"].tolist()
    vocabulary = json.loads((path/"vocabulary.json").read_text(encoding="utf-8"))
    lookup = {c: i for i, c in enumerate(vocabulary)}
    a = np.array([[lookup[c] for c in text] for text in truth])
    p = np.array([[lookup[c] for c in text] for text in predicted])
    assert a.shape == p.shape == (7482, 7)
    assert np.isclose(scores["plate_accuracy"], np.mean([x == y for x, y in zip(truth, predicted)]))
    assert np.isclose(scores["character_accuracy"], (a == p).mean())
    distances = [string_distance(x, y) for x, y in zip(truth, predicted)]
    np.testing.assert_array_equal(distances, predictions["edit_distance"])
    assert np.isclose(scores["CER"], sum(distances)/(7482*7))
    # Explicit binary arrays independently check the compact training-report formula.
    one_hot_a, one_hot_p = np.eye(len(vocabulary))[a], np.eye(len(vocabulary))[p]
    assert_scores(scores, numeric_scores(one_hot_a.ravel(), one_hot_p.ravel()), "aux_one_hot_")
    confusion = pd.read_csv(path/"test_confusion.csv", index_col=0).to_numpy()
    assert confusion.sum() == 7482*7


def audit_notebooks():
    records = json.loads((ROOT/"run/notebook_execution.json").read_text(encoding="utf-8"))
    assert {record["task"] for record in records} == {"task1", "task2", "task3"}
    for record in records:
        task = record["task"]
        path = ROOT/task/(task+".ipynb")
        assert record["notebook_sha256"] == file_hash(path)
        notebook = json.loads(path.read_text(encoding="utf-8"))
        assert notebook["metadata"]["kernelspec"]["name"] == "final-project-windows"
        cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
        assert all(cell["execution_count"] is not None for cell in cells)
        assert not any(output["output_type"] == "error" for cell in cells for output in cell["outputs"])
        # Presentation cells omit CLI/file management but retain the actual model,
        # attention and training step, rather than a different illustrative network.
        roles = {cell["metadata"].get("presentation_role"): "".join(cell["source"]) for cell in cells}
        model_name = notebook["metadata"].get("report_model_class", {"task1": "ForecastRNN", "task2": "ForecastSeq2Seq", "task3": "PlateSeq2Seq"}[task])
        step_name = "plate_train_step" if task == "task3" else "forecast_train_step"
        model_file = ROOT/notebook["metadata"].get("report_model_source",task+"/code/"+task+".py")
        definitions = [("model", model_file, model_name),
                                     ("attention", ROOT/"src/utils.py", "dot_attention"),
                                     ("training_step", ROOT/"src/utils.py", step_name)]
        if 'optimizer_factory' in roles:
            definitions.append(('optimizer_factory', ROOT/'task3/code/train_advanced.py','make_optimizer'))
        for role, filename, name in definitions:
            original = ast.parse(filename.read_text(encoding="utf-8"))
            definition = next(node for node in original.body if getattr(node, "name", None) == name)
            shown = ast.parse(roles[role]).body[0]
            assert ast.dump(definition) == ast.dump(shown), (task, role)
    return {"fresh_windows_kernels": 3, "executed_without_errors": True,
            "paired_model_attention_training_step_match": True}


def main():
    if sys.platform != "win32":
        raise RuntimeError("Run result evaluation with the project's Windows Python.")
    raw = pd.read_csv(ROOT/"data/raw/etth1/ETTh1.csv", parse_dates=["date"]).set_index("date")
    manifest = pd.read_csv(ROOT/"data/splits/ccpd_subset.csv")
    checked = []
    for task in ["task1", "task2", "task3"]:
        lock = ROOT/"run"/task/"locked_test_runs.json"
        if not lock.exists(): continue
        paths = json.loads(lock.read_text(encoding="utf-8"))
        for relative in paths:
            path = ROOT/relative
            metadata = json.loads((path/"metadata.json").read_text(encoding="utf-8"))
            assert metadata["platform"].startswith("Windows") and metadata["test_evaluated"]
            assert Path(metadata["executable"]).resolve() == (ROOT/".venv/Scripts/python.exe").resolve()
            for filename in ["best.pt", "history.csv", "learning_curves.png", "source/utils.py", "source/"+task+".py"]:
                assert (path/filename).is_file(), path/filename
            audit_plates(path, manifest) if task == "task3" else audit_forecast(path, task, raw)
        checked.append({"task": task, "checked_test_runs": len(paths), "passed": True})
        print(task, "saved predictions, original labels, horizons, and metrics: PASS", flush=True)
    report = {"checked": checked, "all_three_tasks_finished": len(checked) == 3}
    selection_path = ROOT/'run/task3/final_selection.json'
    if selection_path.exists():
        selection = json.loads(selection_path.read_text(encoding='utf-8'))
        locked = json.loads((ROOT/'run/task3/locked_test_runs.json').read_text(encoding='utf-8'))
        assert locked == [item['run'] for item in selection['checkpoints']]
        assert selection['selected_on'] == 'validation only'
        for item in selection['checkpoints']:
            folder = ROOT/item['run']
            assert file_hash(folder/'best.pt') == item['checkpoint_sha256']
            assert file_hash(folder/'config.json') == item['config_sha256']
        chosen = ROOT/selection['selected_run']
        final_config = json.loads((chosen/'config.json').read_text(encoding='utf-8'))
        assert selection['final_seeds'] == [final_config['seed']]
        report['task3_frozen_checkpoint_hashes_match'] = True
        report['task3_completed_fine_tuning_seeds'] = selection['final_seeds']
    if (ROOT/"run/notebook_execution.json").exists():
        report["notebooks"] = audit_notebooks()
        print("Three fresh Windows notebooks and their paired scripts: PASS", flush=True)
    save_json(ROOT/"run/verification/results_audit.json", report)


if __name__ == "__main__":
    main()
