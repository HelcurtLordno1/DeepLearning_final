"""Small shared helpers. Models and optimizer choices stay in each task file."""

from pathlib import Path
from datetime import datetime
import hashlib
import json
import platform
import random
import shutil
import sys
import time
import os
import ctypes
from ctypes import wintypes

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance, ImageFilter
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
FEATURES = ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"]
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
_training_lock = None
_monitor_handle = None
_kernel32 = None


def check_monitor():
    if _monitor_handle is None or _kernel32.WaitForSingleObject(_monitor_handle, 0) != 258:
        raise RuntimeError("The PowerShell resource monitor stopped; GPU work is stopping too.")


def connect_monitor():
    global _monitor_handle, _kernel32
    monitor = os.environ.get("FINAL_PROJECT_MONITOR_PID", "")
    if not monitor.isdecimal():
        raise RuntimeError("GPU work requires the monitored PowerShell launcher: run_experiments.ps1 -Mode Train -Task taskX")
    if _monitor_handle is None:
        _kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        _kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        _kernel32.OpenProcess.restype = wintypes.HANDLE
        _kernel32.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        _kernel32.WaitForSingleObject.restype = wintypes.DWORD
        _monitor_handle = _kernel32.OpenProcess(0x00100000, False, int(monitor))
    check_monitor()


def claim_training_slot():
    """Windows releases this byte lock even if Python crashes or is stopped."""
    global _training_lock
    if _training_lock is not None:
        return
    import msvcrt
    folder = ROOT / "run" / "safety"
    folder.mkdir(parents=True, exist_ok=True)
    handle = (folder / "python_training.lock").open("a+b")
    handle.seek(0, 2)
    if handle.tell() == 0:
        handle.write(b"0"); handle.flush()
    handle.seek(0)
    try:
        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
    except OSError:
        handle.close()
        raise RuntimeError("Another project training/evaluation process already holds the GPU slot.") from None
    _training_lock = handle


def safe_training_config(config):
    # Record the actual reduced settings, including when a saved config was loaded.
    return {**config, "batch_size": min(config["batch_size"], 16), "workers": 0,
            "safety_profile": "recovery-v1", "batch_pause_seconds": .05}


def batch_pause(device):
    # Synchronize first: the sleep must give the GPU a real break between batches.
    torch.cuda.synchronize(device)
    check_monitor()
    time.sleep(.05)


def save_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name+".tmp")
    with temporary.open("w", encoding="utf-8") as output:
        output.write(json.dumps(value, ensure_ascii=False, indent=2))
        output.flush(); os.fsync(output.fileno())
    temporary.replace(path)


def save_checkpoint(path, checkpoint):
    # An interrupted write should leave the previous best checkpoint intact.
    temporary = path.with_suffix(".partial.pt")
    with temporary.open("wb") as output:
        torch.save(checkpoint, output)
        output.flush(); os.fsync(output.fileno())
    temporary.replace(path)


def save_history(path, history):
    temporary = path.with_name("history.partial.csv")
    with temporary.open("w", encoding="utf-8", newline="") as output:
        pd.DataFrame(history).to_csv(output, index=False)
        output.flush(); os.fsync(output.fileno())
    temporary.replace(path)


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def windows_device(seed=42):
    if sys.platform != "win32":
        raise RuntimeError("Training must use Windows .venv/Scripts/python.exe from PowerShell.")
    if Path(sys.executable).resolve() != (ROOT / ".venv" / "Scripts" / "python.exe").resolve():
        raise RuntimeError("Use this project's own .venv/Scripts/python.exe.")
    connect_monitor()
    claim_training_slot()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Run check_environment.py in the Windows environment.")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = False
    torch.cuda.set_per_process_memory_fraction(.25)
    torch.set_float32_matmul_precision("high")
    return torch.device("cuda")


def new_run(task, config, data_id, output_dir=None):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    path = Path(output_dir) if output_dir else ROOT / "run" / task / config["name"] / str(config["seed"]) / timestamp
    path.mkdir(parents=True, exist_ok=False)
    save_json(path / "config.json", config)
    info = {"task": task, "data_id": data_id, "platform": platform.platform(),
            "python": sys.version, "executable": sys.executable,
            "torch": torch.__version__, "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0), "matmul_precision": "high",
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "safety_profile": "recovery-v1", "cuda_allocator_fraction_limit": .25,
            "supervisor_pid": int(os.environ["FINAL_PROJECT_MONITOR_PID"]),
            "cpu_threads": torch.get_num_threads(), "test_evaluated": False}
    save_json(path / "metadata.json", info)
    source = path / "source"
    source.mkdir()
    files = [ROOT / "src" / "utils.py", ROOT / task / "code" / (task + ".py")]
    if task == "task3" and (ROOT/task/"code/refined_model.py").exists():
        files.append(ROOT/task/"code/refined_model.py")
    for file in files:
        shutil.copy2(file, source / file.name)
    print("Run:", path.relative_to(ROOT) if path.is_relative_to(ROOT) else path, flush=True)
    return path


def update_metadata(path, **values):
    file = Path(path) / "metadata.json"
    info = json.loads(file.read_text(encoding="utf-8"))
    info.update(values)
    save_json(file, info)


def dot_attention(query, memory):
    # query [B, outputs, H]; memory [B, input_positions, H].
    scores = torch.bmm(query, memory.transpose(1, 2)) / np.sqrt(memory.shape[-1])
    weights = scores.softmax(dim=-1)
    return torch.bmm(weights, memory), weights


def regression_metrics(actual, predicted):
    actual, predicted = np.asarray(actual, dtype=np.float64), np.asarray(predicted, dtype=np.float64)
    error = actual - predicted
    denominator = np.square(actual - actual.mean()).sum()
    score = float(1 - np.square(error).sum() / denominator) if denominator > 0 else None
    return {"R2": score, "NSE": score, "MAE": float(np.abs(error).mean()),
            "RMSE": float(np.sqrt(np.square(error).mean()))}


def forecast_metric_table(actual, predicted, horizons):
    rows = [{"horizon": f"t+{h}", **regression_metrics(actual[:, i], predicted[:, i])}
            for i, h in enumerate(horizons)]
    frame = pd.DataFrame(rows)
    rows.append({"horizon": "macro", **frame[["R2", "NSE", "MAE", "RMSE"]].mean().to_dict()})
    rows.append({"horizon": "pooled", **regression_metrics(actual.ravel(), predicted.ravel())})
    return pd.DataFrame(rows)


def prepare_forecast(task, config):
    path = ROOT / "data" / "raw" / "etth1" / "ETTh1.csv"
    frame = pd.read_csv(path, parse_dates=["date"])
    assert len(frame) == 17420 and frame["date"].is_unique
    assert frame["date"].diff().dropna().eq(pd.Timedelta(hours=1)).all()
    columns = ["OT"] if config.get("univariate", False) else FEATURES
    a, b = int(len(frame) * .70), int(len(frame) * .85)
    x_scaler = StandardScaler().fit(frame.iloc[:a][columns])
    y_scaler = StandardScaler().fit(frame.iloc[:a][["OT"]])
    horizons = [2] if task == "task1" else [1, 2, 3, 4, 5]
    lookback, parts = config["lookback"], {}
    for name, part in [("train", frame.iloc[:a]), ("validation", frame.iloc[a:b]), ("test", frame.iloc[b:])]:
        features = x_scaler.transform(part[columns]).astype("float32")
        target = part["OT"].to_numpy(dtype="float32")
        scaled_target = y_scaler.transform(part[["OT"]]).astype("float32").ravel()
        X, y, last, seasonal, origins, targets = [], [], [], [], [], []
        for end in range(lookback, len(part) - max(horizons) + 1):
            X.append(features[end-lookback:end])
            y.append([scaled_target[end-1+h] for h in horizons])
            last.append(scaled_target[end-1])
            seasonal.append([target[end-1+h-24] for h in horizons])
            origins.append(part["date"].iloc[end-1])
            targets.append([part["date"].iloc[end-1+h] for h in horizons])
        parts[name] = {"X": np.asarray(X, dtype="float32"), "y": np.asarray(y, dtype="float32"),
                       "last": np.asarray(last, dtype="float32"), "seasonal": np.asarray(seasonal),
                       "origins": np.asarray(origins), "targets": np.asarray(targets), "raw_rows": len(part)}
    scalers = {"features": columns, "x_mean": x_scaler.mean_.tolist(), "x_scale": x_scaler.scale_.tolist(),
               "y_mean": float(y_scaler.mean_[0]), "y_scale": float(y_scaler.scale_[0])}
    identity = file_hash(path) + f":chronological70-15-15:L{lookback}:H{horizons}"
    return parts, scalers, horizons, identity


def forecast_loader(part, batch_size, shuffle=False):
    dataset = TensorDataset(torch.from_numpy(part["X"]), torch.from_numpy(part["y"]), torch.from_numpy(part["last"]))
    return DataLoader(dataset, batch_size=min(batch_size, 16), shuffle=shuffle, num_workers=0, pin_memory=False)


@torch.no_grad()
def predict_forecast(model, loader, device):
    model.eval()
    outputs = []
    for X, _, last in loader:
        outputs.append(model(X.to(device), last.to(device)).cpu().numpy())
        batch_pause(device)
    return np.concatenate(outputs)


def forecast_train_step(model, optimizer, X, y, last):
    optimizer.zero_grad(set_to_none=True)
    prediction = model(X, last)
    loss = nn.functional.mse_loss(prediction, y)
    if not torch.isfinite(loss):
        raise RuntimeError("Non-finite forecasting training loss")
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), 1.)
    optimizer.step()
    return loss.item()


def plate_train_step(model, optimizer, X, y):
    optimizer.zero_grad(set_to_none=True)
    logits = model(X)                           # [B, 7, 67]
    loss = nn.functional.cross_entropy(logits.transpose(1, 2), y)
    if not torch.isfinite(loss):
        raise RuntimeError("Non-finite recognition training loss")
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), 1.)
    optimizer.step()
    return loss.item()


def fit_forecast(model, optimizer, parts, config, scalers, path, device):
    train = forecast_loader(parts["train"], config["batch_size"], shuffle=True)
    validation = forecast_loader(parts["validation"], config["batch_size"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=.5, min_lr=1e-6)
    best, wait, history = float("inf"), 0, []
    start = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    for epoch in range(1, config["epochs"] + 1):
        epoch_start, total = time.perf_counter(), 0.
        model.train()
        for X, y, last in train:
            X, y, last = X.to(device), y.to(device), last.to(device)
            loss = forecast_train_step(model, optimizer, X, y, last)
            batch_pause(device)
            total += loss * len(X)
        predictions = predict_forecast(model, validation, device)
        error = (predictions - parts["validation"]["y"]) * scalers["y_scale"]
        score = float(np.sqrt(np.mean(error.astype("float64")**2, axis=0)).mean())
        history.append({"epoch": epoch, "train_loss": total / len(train.dataset),
                        "val_RMSE": score, "lr": optimizer.param_groups[0]["lr"],
                        "seconds": time.perf_counter()-epoch_start})
        print(f"Epoch {epoch:02d}: train_MSE={history[-1]['train_loss']:.5f}, val_RMSE={score:.5f}", flush=True)
        if score < best - 1e-5:
            best, wait = score, 0
            save_checkpoint(path / "best.pt", {"model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
                        "config": config, "epoch": epoch, "validation_score": score})
        else:
            wait += 1
        scheduler.step(score)
        save_history(path / "history.csv", history)
        if wait >= config["patience"]:
            break
    checkpoint = torch.load(path / "best.pt", map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state"])
    update_metadata(path, training_seconds=time.perf_counter()-start, best_epoch=checkpoint["epoch"],
                    validation_score=best, parameters=sum(p.numel() for p in model.parameters()),
                    peak_vram_gib=torch.cuda.max_memory_allocated()/2**30)
    plot_history(path, "val_RMSE")
    return best


def evaluate_forecast(model, parts, scalers, horizons, path, device, split="test"):
    part = parts[split]
    predicted = predict_forecast(model, forecast_loader(part, 256), device) * scalers["y_scale"] + scalers["y_mean"]
    actual = part["y"] * scalers["y_scale"] + scalers["y_mean"]
    persistence = np.repeat((part["last"] * scalers["y_scale"] + scalers["y_mean"])[:, None], len(horizons), axis=1)
    tables = []
    for name, values in [("model", predicted), ("persistence", persistence), ("daily_seasonal", part["seasonal"])]:
        table = forecast_metric_table(actual, values, horizons)
        table.insert(0, "model", name)
        tables.append(table)
    table = pd.concat(tables, ignore_index=True)
    table.to_csv(path / f"{split}_metrics.csv", index=False)
    records = []
    for i in range(len(actual)):
        for j, horizon in enumerate(horizons):
            records.append({"origin_time": part["origins"][i], "target_time": part["targets"][i, j],
                            "horizon": horizon, "actual": actual[i, j], "predicted": predicted[i, j],
                            "persistence": persistence[i, j], "daily_seasonal": part["seasonal"][i, j]})
    pd.DataFrame(records).to_csv(path / f"{split}_predictions.csv", index=False)
    fig, axes = plt.subplots(len(horizons), 1, figsize=(11, 3*len(horizons)), squeeze=False)
    for j, horizon in enumerate(horizons):
        ax = axes[j, 0]
        ax.plot(actual[:240, j], label="Actual")
        ax.plot(predicted[:240, j], label="Model")
        ax.plot(persistence[:240, j], label="Persistence", alpha=.5)
        ax.set_title(f"{split}: t+{horizon}")
        ax.legend()
    fig.tight_layout(); fig.savefig(path / f"{split}_forecasts.png", dpi=130); plt.close(fig)
    print(table.to_string(index=False), flush=True)
    update_metadata(path, **{f"{split}_evaluated": True})
    return table


def plot_history(path, metric):
    history = pd.read_csv(path / "history.csv")
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    axes[0].plot(history["epoch"], history["train_loss"], label="Training loss")
    if "val_loss" in history:
        axes[0].plot(history["epoch"], history["val_loss"], label="Validation loss")
    axes[0].legend(); axes[0].set_xlabel("Epoch")
    axes[1].plot(history["epoch"], history[metric]); axes[1].set_title(metric); axes[1].set_xlabel("Epoch")
    fig.tight_layout(); fig.savefig(path / "learning_curves.png", dpi=130); plt.close(fig)


def perspective_crop(image, row):
    # CCPD corners: bottom-right, bottom-left, top-left, top-right.
    corners = np.asarray(json.loads(row["corners"]), dtype=float).reshape(4, 2)[[2, 3, 0, 1]]
    destination = [(0, 0), (191, 0), (191, 63), (0, 63)]
    A, values = [], []
    for (x, y), (u, v) in zip(destination, corners):
        A.extend([[x, y, 1, 0, 0, 0, -u*x, -u*y], [0, 0, 0, x, y, 1, -v*x, -v*y]])
        values.extend([u, v])
    coefficients = np.linalg.solve(np.asarray(A), np.asarray(values))
    return image.transform((192, 64), Image.Transform.PERSPECTIVE, tuple(coefficients.tolist()),
                           resample=Image.Resampling.BICUBIC)


def prepare_plate_cache(mode="bbox"):
    manifest_path = ROOT / "data" / "splits" / "ccpd_subset.csv"
    identity = file_hash(manifest_path)
    frame = pd.read_csv(manifest_path)
    assert len(frame) == 50000 and frame["text"].str.len().eq(7).all()
    groups = {split: set(frame.loc[frame["split"].eq(split), "text"]) for split in ["train", "validation", "test"]}
    assert not (groups["train"] & groups["validation"] or groups["train"] & groups["test"] or groups["validation"] & groups["test"])
    vocabulary = json.loads((ROOT / "data/splits/ccpd_vocabulary.json").read_text(encoding="utf-8"))
    cache = ROOT / "data" / "processed" / f"ccpd_{mode}_64x192_{identity[:8]}.npy"
    cache.parent.mkdir(parents=True, exist_ok=True)
    metadata = cache.with_suffix(".json")
    if cache.exists() and metadata.exists():
        info = json.loads(metadata.read_text(encoding="utf-8"))
        if info["manifest_sha256"] != identity or info["shape"] != [50000, 64, 192, 3]:
            raise RuntimeError("Processed cache metadata mismatch")
        return frame, vocabulary, cache, identity
    if shutil.disk_usage(ROOT).free < 20*2**30 + 50000*64*192*3:
        raise RuntimeError("Not enough disk space while preserving 20 GiB free")
    temporary = cache.with_suffix(".partial.npy")
    images = np.lib.format.open_memmap(temporary, mode="w+", dtype="uint8", shape=(len(frame), 64, 192, 3))

    def read_crop(row):
        with Image.open(ROOT / row["image_path"]) as source:
            image = source.convert("RGB")
            if mode == "perspective":
                crop = perspective_crop(image, row)
            else:
                crop = image.crop(json.loads(row["bbox"])).resize((192, 64), Image.Resampling.BILINEAR)
            return np.asarray(crop, dtype="uint8")

    print("Preparing", mode, "crop cache from original data/raw images...", flush=True)
    # One image at a time: avoid a queue of 50,000 future crop results in RAM.
    for index, row in enumerate(frame.to_dict("records")):
        images[index] = read_crop(row)
        if (index+1) % 5000 == 0:
            images.flush()
            print("Cropped", index+1, "/", len(frame), flush=True)
        if (index+1) % 32 == 0:
            check_monitor()
            time.sleep(.02)
    images.flush(); del images
    temporary.replace(cache)
    save_json(metadata, {"manifest_sha256": identity, "mode": mode, "shape": [50000, 64, 192, 3],
                         "dtype": "uint8", "source": "Original images listed in the fixed manifest"})
    return frame, vocabulary, cache, identity


class PlateDataset(Dataset):
    def __init__(self, frame, vocabulary, cache, split, augment=False):
        self.rows = frame.index[frame["split"].eq(split)].to_numpy()
        lookup = {char: i for i, char in enumerate(vocabulary)}
        self.labels = np.asarray([[lookup[c] for c in text] for text in frame.loc[self.rows, "text"]], dtype="int64")
        self.cache, self.images, self.augment = str(cache), None, augment

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        # Open the memory map in the worker, avoiding copies when Windows spawns it.
        if self.images is None:
            self.images = np.load(self.cache, mmap_mode="r")
        pixels = np.array(self.images[self.rows[index]], copy=True)
        if self.augment:
            image = Image.fromarray(pixels)
            image = ImageEnhance.Brightness(image).enhance(random.uniform(.8, 1.2))
            image = ImageEnhance.Contrast(image).enhance(random.uniform(.85, 1.15))
            image = image.rotate(random.uniform(-5, 5), resample=Image.Resampling.BILINEAR,
                                 expand=True).resize((192, 64), Image.Resampling.BILINEAR)
            if random.random() < .1:
                image = image.filter(ImageFilter.GaussianBlur(random.uniform(.1, .5)))
            pixels = np.asarray(image).copy()
        X = torch.from_numpy(pixels).permute(2, 0, 1).float() / 255.
        return X, torch.from_numpy(self.labels[index])


def seed_worker(worker_id):
    seed = torch.initial_seed() % 2**32
    random.seed(seed); np.random.seed(seed)


def plate_loader(dataset, config, shuffle=False):
    return DataLoader(dataset, batch_size=min(config["batch_size"], 16), shuffle=shuffle,
                      num_workers=0, pin_memory=False,
                      worker_init_fn=seed_worker,
                      generator=torch.Generator().manual_seed(config["seed"]))


def edit_distance(actual, predicted):
    previous = list(range(len(predicted)+1))
    for i, left in enumerate(actual, 1):
        current = [i]
        for j, right in enumerate(predicted, 1):
            current.append(min(current[-1]+1, previous[j]+1, previous[j-1]+(left != right)))
        previous = current
    return previous[-1]


def decode_plates(ids, vocabulary):
    return ["".join(vocabulary[int(value)] for value in row) for row in ids]


def plate_metrics(actual, predicted, vocabulary):
    true_text, pred_text = decode_plates(actual, vocabulary), decode_plates(predicted, vocabulary)
    errors = [edit_distance(a, p) for a, p in zip(true_text, pred_text)]
    correct = actual == predicted
    n, length = actual.shape
    # Equivalent to flattening the decoded one-hot arrays: SSE = 2 * wrong positions.
    wrong, classes = int((~correct).sum()), len(vocabulary)
    one_hot_mse = 2*wrong/(n*length*classes)
    # Decoded binary one-hot differences have equal absolute and squared errors.
    one_hot_r2 = 1 - 2*wrong/(n*length*(1-1/classes))
    return {"plate_accuracy": float(correct.all(axis=1).mean()),
            "character_accuracy": float(correct.mean()), "CER": float(sum(errors)/(n*length)),
            "aux_one_hot_R2": one_hot_r2, "aux_one_hot_NSE": one_hot_r2,
            "aux_one_hot_MAE": one_hot_mse, "aux_one_hot_RMSE": float(np.sqrt(one_hot_mse))}


@torch.no_grad()
def predict_plates(model, loader, device):
    model.eval()
    predictions, labels, total_loss = [], [], 0.
    for X, y in loader:
        logits = model(X.to(device, non_blocking=True))
        total_loss += nn.functional.cross_entropy(logits.transpose(1, 2), y.to(device)).item()*len(X)
        predictions.append(logits.argmax(-1).cpu().numpy()); labels.append(y.numpy())
        batch_pause(device)
    return np.concatenate(labels), np.concatenate(predictions), total_loss/len(loader.dataset)


def fit_plates(model, optimizer, loaders, config, vocabulary, path, device):
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=.5, min_lr=1e-5)
    best_accuracy, best_cer, wait, history = -1., float("inf"), 0, []
    start = time.perf_counter(); torch.cuda.reset_peak_memory_stats()
    for epoch in range(1, config["epochs"]+1):
        model.train(); total_loss = 0.; epoch_start = time.perf_counter()
        for X, y in loaders["train"]:
            X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
            loss = plate_train_step(model, optimizer, X, y)
            batch_pause(device)
            total_loss += loss*len(X)
        actual, predicted, val_loss = predict_plates(model, loaders["validation"], device)
        metrics = plate_metrics(actual, predicted, vocabulary)
        accuracy, cer = metrics["plate_accuracy"], metrics["CER"]
        history.append({"epoch": epoch, "train_loss": total_loss/len(loaders["train"].dataset),
                        "val_loss": val_loss, "val_plate_accuracy": accuracy, "val_CER": cer,
                        "val_character_accuracy": metrics["character_accuracy"],
                        "lr": optimizer.param_groups[0]["lr"], "seconds": time.perf_counter()-epoch_start})
        print(f"Epoch {epoch:02d}: train_CE={history[-1]['train_loss']:.4f}, val_plate={accuracy:.4f}, val_CER={cer:.4f}", flush=True)
        better = accuracy > best_accuracy + 1e-8 or (abs(accuracy-best_accuracy) < 1e-8 and cer < best_cer-1e-8)
        if better:
            best_accuracy, best_cer, wait = accuracy, cer, 0
            save_checkpoint(path / "best.pt", {"model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
                        "config": config, "epoch": epoch, "validation_score": accuracy,
                        "validation_CER": cer})
        else:
            wait += 1
        scheduler.step(val_loss)
        save_history(path / "history.csv", history)
        if wait >= config["patience"]:
            break
    checkpoint = torch.load(path / "best.pt", map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state"])
    update_metadata(path, training_seconds=time.perf_counter()-start, best_epoch=checkpoint["epoch"],
                    validation_score=best_accuracy, validation_CER=best_cer,
                    parameters=sum(p.numel() for p in model.parameters()),
                    peak_vram_gib=torch.cuda.max_memory_allocated()/2**30)
    plot_history(path, "val_plate_accuracy")
    return best_accuracy


def evaluate_plates(model, loader, frame, vocabulary, path, device, split="test", predictions=None):
    actual, predicted, loss = predict_plates(model, loader, device) if predictions is None else predictions
    scores = {"cross_entropy": loss, **plate_metrics(actual, predicted, vocabulary)}
    save_json(path / f"{split}_metrics.json", scores)
    true_text, pred_text = decode_plates(actual, vocabulary), decode_plates(predicted, vocabulary)
    rows = frame.loc[loader.dataset.rows, ["image_path"]].reset_index(drop=True)
    rows["true_text"], rows["predicted_text"] = true_text, pred_text
    rows["exact_match"] = [a == p for a, p in zip(true_text, pred_text)]
    rows["edit_distance"] = [edit_distance(a, p) for a, p in zip(true_text, pred_text)]
    rows.to_csv(path / f"{split}_predictions.csv", index=False, encoding="utf-8")
    pd.DataFrame({"position": range(1, 8), "accuracy": (actual == predicted).mean(axis=0)}).to_csv(path / f"{split}_positions.csv", index=False)
    confusion = np.zeros((len(vocabulary), len(vocabulary)), dtype=int)
    np.add.at(confusion, (actual.ravel(), predicted.ravel()), 1)
    pd.DataFrame(confusion, index=vocabulary, columns=vocabulary).to_csv(path / f"{split}_confusion.csv", encoding="utf-8")
    # First examples from each outcome, so displayed cases are not picked by hand.
    selected = list(rows.index[rows["exact_match"]][:6]) + list(rows.index[~rows["exact_match"]][:6])
    fig, axes = plt.subplots(4, 3, figsize=(12, 8))
    for ax in axes.ravel(): ax.axis("off")
    for ax, index in zip(axes.ravel(), selected):
        X, _ = loader.dataset[index]
        ax.imshow(X.permute(1, 2, 0).numpy()); ax.set_title(f"{true_text[index]} → {pred_text[index]}", fontsize=10)
    fig.tight_layout(); fig.savefig(path / f"{split}_examples.png", dpi=130); plt.close(fig)
    if model.attention and len(selected):
        X, _ = loader.dataset[selected[0]]
        with torch.no_grad(): _, weights = model(X[None].to(device), return_attention=True)
        fig, ax = plt.subplots(figsize=(11, 3))
        ax.imshow(weights[0].cpu().numpy(), aspect="auto", cmap="viridis")
        ax.set_yticks(range(7), list(true_text[selected[0]])); ax.set_xlabel("Visual column"); ax.set_ylabel("Output character")
        fig.tight_layout(); fig.savefig(path / f"{split}_attention.png", dpi=130); plt.close(fig)
    print(json.dumps(scores, indent=2), flush=True)
    update_metadata(path, **{f"{split}_evaluated": True})
    return scores
