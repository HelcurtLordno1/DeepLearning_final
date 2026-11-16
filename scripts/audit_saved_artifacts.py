"""Read small run artifacts after the crash; no raw-image scan or CUDA work."""

import bootstrap  # Resolve src/ and the project root.

from pathlib import Path
import json
import zipfile
import numpy as np
import pandas as pd
from PIL import Image
import torch
from utils import ROOT, file_hash, save_json, plate_metrics, edit_distance


def main():
    torch.set_num_threads(2)
    assert not torch.cuda.is_initialized()
    problems, checked = [], {"json": 0, "checkpoints": 0, "png": 0}
    recovery = ROOT/"run/recovery"
    recovery.mkdir(parents=True, exist_ok=True)
    for path in (ROOT/"run").rglob("*.json"):
        if "recovery" in path.relative_to(ROOT/"run").parts:
            continue
        checked["json"] += 1
        try:
            json.loads(path.read_text(encoding="utf-8-sig"))
        except (ValueError, UnicodeError) as error:
            problems.append({"file": str(path.relative_to(ROOT)), "kind": "json", "error": str(error)})

    # A ZIP CRC check catches damaged tensor bytes that torch.load could accept.
    for task in ["task1", "task2", "task3"]:
        for path in (ROOT/"run"/task).rglob("best.pt"):
            checked["checkpoints"] += 1
            try:
                with zipfile.ZipFile(path) as archive:
                    assert archive.testzip() is None, "Checkpoint ZIP CRC mismatch"
                checkpoint = torch.load(path, map_location="cpu", weights_only=True)
                assert all(torch.isfinite(value).all() for value in checkpoint["model_state"].values())
            except Exception as error:
                problems.append({"file": str(path.relative_to(ROOT)), "kind": "checkpoint", "error": str(error)})
        for path in (ROOT/"run"/task).rglob("*.png"):
            checked["png"] += 1
            try:
                with Image.open(path) as image:
                    image.verify()
            except Exception as error:
                problems.append({"file": str(path.relative_to(ROOT)), "kind": "png", "error": str(error)})

    mappings = {}
    manifest = pd.read_csv(ROOT/"data/splits/ccpd_subset.csv")
    validation = manifest[manifest["split"].eq("validation")].set_index("image_path")
    for problem in problems:
        path = ROOT/problem["file"]
        if path.name != "validation_metrics.json" or "task3" not in path.parts:
            continue
        predictions = pd.read_csv(path.parent/"validation_predictions.csv")
        assert len(predictions) == len(validation) == 7475
        assert predictions["image_path"].is_unique and set(predictions["image_path"]) == set(validation.index)
        assert predictions["true_text"].tolist() == validation.loc[predictions["image_path"], "text"].tolist()
        vocabulary = json.loads((path.parent/"vocabulary.json").read_text(encoding="utf-8"))
        lookup = {char: index for index, char in enumerate(vocabulary)}
        actual = np.array([[lookup[c] for c in text] for text in predictions["true_text"]])
        predicted = np.array([[lookup[c] for c in text] for text in predictions["predicted_text"]])
        assert actual.shape == predicted.shape == (7475, 7)
        distances = [edit_distance(a, b) for a, b in zip(predictions["true_text"], predictions["predicted_text"])]
        np.testing.assert_array_equal(distances, predictions["edit_distance"])
        np.testing.assert_array_equal(predictions["true_text"].eq(predictions["predicted_text"]), predictions["exact_match"])
        scores = plate_metrics(actual, predicted, vocabulary)
        metadata = json.loads((path.parent/"metadata.json").read_text(encoding="utf-8"))
        assert np.isclose(scores["plate_accuracy"], metadata["validation_score"])
        assert np.isclose(scores["CER"], metadata["validation_CER"])
        output = recovery/"task3_attention_validation_metrics.json"
        save_json(output, scores)
        save_json(recovery/"task3_attention_provenance.json", {
            "original_damaged_file": problem["file"], "original_preserved": True,
            "prediction_source": str((path.parent/"validation_predictions.csv").relative_to(ROOT)),
            "predictions_sha256": file_hash(path.parent/"validation_predictions.csv"),
            "manifest_sha256": file_hash(ROOT/"data/splits/ccpd_subset.csv"),
            "rows": len(predictions), "split": "validation",
            "checks": "All labels match fixed validation manifest; text lengths, exact matches, edit distances, metadata score and CER match.",
            "cross_entropy": "Not recovered: hard predictions do not determine logit cross-entropy.",
            "cuda_initialized": False})
        mappings[problem["file"]] = str(output.relative_to(ROOT))
    save_json(recovery/"recovered_files.json", mappings)
    assert not torch.cuda.is_initialized()
    save_json(ROOT/"run/safety/artifact_audit.json", {"checked": checked, "problems": problems,
        "recovered_metrics": mappings, "original_files_modified": False, "cuda_initialized": False})
    print(json.dumps({"checked": checked, "problems": problems, "recovered": mappings}, indent=2))


if __name__ == "__main__":
    main()
