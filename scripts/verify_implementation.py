"""Meaningful data/shape/metric checks before the full Windows CUDA runs."""

import bootstrap  # Resolve src/ and the project root.

import json
import time
import numpy as np
import torch
from torch import nn
from PIL import Image
from utils import (ROOT, windows_device, prepare_forecast, regression_metrics,
                   plate_metrics, edit_distance, prepare_plate_cache, PlateDataset, save_json, perspective_crop)
from task1.code.task1 import ForecastRNN, BASE_CONFIG as CONFIG1
from task2.code.task2 import ForecastSeq2Seq, BASE_CONFIG as CONFIG2
from task3.code.task3 import PlateSeq2Seq, BASE_CONFIG as CONFIG3


def main():
    device = windows_device(42)
    for task, config, model_class, count, width in [
            ("task1", CONFIG1, ForecastRNN, 12145, 1),
            ("task2", CONFIG2, ForecastSeq2Seq, 12142, 5)]:
        parts, scalers, horizons, _ = prepare_forecast(task, config)
        assert parts["train"]["X"].shape == (count, 48, 7)
        first = parts["train"]
        origin = first["origins"][0]
        for h, target in zip(horizons, first["targets"][0]):
            assert (np.datetime64(target)-np.datetime64(origin))/np.timedelta64(1, "h") == h
        model = model_class(7, config).to(device)
        X = torch.tensor(first["X"][:4], device=device)
        last = torch.tensor(first["last"][:4], device=device)
        prediction = model(X, last)
        assert prediction.shape == (4, width)
        # The zero-initialized residual head starts at persistence, in target scale.
        assert torch.allclose(prediction, last[:, None].expand_as(prediction))
        nn.functional.mse_loss(prediction, torch.tensor(first["y"][:4], device=device)).backward()
        assert any(p.grad is not None and bool(p.grad.abs().sum()) for p in model.parameters())
        print(task, "real timestamps, window counts, output shape, and gradients: PASS", flush=True)
    example = regression_metrics([1., 2., 3.], [1., 2., 2.])
    assert np.isclose(example["R2"], .5) and np.isclose(example["MAE"], 1/3)
    assert example["R2"] == example["NSE"]
    assert regression_metrics([2., 2.], [1., 2.])["R2"] is None
    assert edit_distance("AB", "BA") == 2 and edit_distance("ABC", "AC") == 1
    truth = np.array([[0, 1], [1, 2]])
    pred = np.array([[0, 1], [2, 2]])
    metrics = plate_metrics(truth, pred, list("ABC"))
    one_hot_true, one_hot_pred = np.eye(3)[truth].ravel(), np.eye(3)[pred].ravel()
    direct = regression_metrics(one_hot_true, one_hot_pred)
    assert np.isclose(metrics["aux_one_hot_R2"], direct["R2"])
    assert np.isclose(metrics["aux_one_hot_RMSE"], direct["RMSE"])
    assert metrics["plate_accuracy"] == .5 and metrics["character_accuracy"] == .75
    print("Metric hand calculations and one-hot equivalence: PASS", flush=True)

    pixels = np.zeros((64, 192, 3), dtype="uint8")
    pixels[:, :96, 0] = 255; pixels[:, 96:, 2] = 255
    rectangle = {"corners": json.dumps([191, 63, 0, 63, 0, 0, 191, 0])}
    rectified = np.asarray(perspective_crop(Image.fromarray(pixels), rectangle))
    assert rectified[32, 20, 0] > 240 and rectified[32, 170, 2] > 240
    print("CCPD corner order and rectification direction: PASS", flush=True)

    frame, vocabulary, cache, _ = prepare_plate_cache("bbox")
    dataset = PlateDataset(frame, vocabulary, cache, "train")
    # Temporary training-only diagnostic, not a benchmark or a test-set score.
    boxes = [json.loads(value) for value in frame.loc[dataset.rows, "bbox"]]
    areas = [(b[2]-b[0])*(b[3]-b[1]) for b in boxes]
    clear_indices = np.argsort(areas)[-64:]
    X, y = zip(*(dataset[int(i)] for i in clear_indices))
    X, y = torch.stack(X).to(device), torch.stack(y).to(device)
    config = {**CONFIG3, "position": True}
    model = PlateSeq2Seq(len(vocabulary), config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=.003)
    history, start = [], time.perf_counter()
    for step in range(1, 1001):
        if step == 400:
            optimizer.param_groups[0]["lr"] = .0003
        if step == 700:
            optimizer.param_groups[0]["lr"] = .0001
        model.train(); optimizer.zero_grad(set_to_none=True)
        logits = model(X)
        loss = nn.functional.cross_entropy(logits.transpose(1, 2), y)
        loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.); optimizer.step()
        if step % 50 == 0:
            accuracy = float((logits.argmax(-1) == y).float().mean().item())
            history.append({"step": step, "loss": loss.item(), "training_character_accuracy": accuracy,
                            "learning_rate": optimizer.param_groups[0]["lr"]})
            print("Training-only 64-image diagnostic:", history[-1], flush=True)
            if accuracy >= .995:
                break
    if history[-1]["training_character_accuracy"] < .98:
        raise RuntimeError("Small training-set fit failed; inspect recognition model before full training")
    output = ROOT / "run" / "verification"
    output.mkdir(parents=True, exist_ok=True)
    save_json(output / "checks.json", {"platform": "Windows CUDA", "forecast_checks": "passed",
              "metric_checks": "passed", "small_training_only_fit": history,
              "diagnostic_seconds": time.perf_counter()-start,
              "training_manifest_rows": dataset.rows[clear_indices].tolist(),
              "note": "64 training images only; this is not validation/test performance."})
    print("All implementation checks passed.", flush=True)


if __name__ == "__main__":
    main()
