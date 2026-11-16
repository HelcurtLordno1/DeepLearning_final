# %% 1. Imports and settings
from pathlib import Path
import argparse
import json
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2] if "__file__" in globals() else (Path.cwd() if (Path.cwd()/"src/utils.py").exists() else Path.cwd().parent)
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT/"src"))
import torch
from torch import nn
from utils import (windows_device, prepare_forecast, dot_attention, new_run,
                   save_json, fit_forecast, evaluate_forecast, safe_training_config)

TRAIN = False  # Opening a notebook must not automatically start GPU training.
BASE_CONFIG = {"name": "seq2seq_attention_residual", "seed": 42, "lookback": 48,
          "hidden_size": 64, "attention": True, "residual": True, "dropout": 0.,
          "univariate": False, "learning_rate": .001,
          "batch_size": 64, "epochs": 60, "patience": 8}
CONFIG = BASE_CONFIG.copy()
selected_file = PROJECT_ROOT / "task2" / "selected_config.json"
if selected_file.exists():
    CONFIG.update(json.loads(selected_file.read_text(encoding="utf-8")))


# %% 2. LSTM Encoder–Decoder with five recurrent output steps
class ForecastSeq2Seq(nn.Module):
    def __init__(self, features, config):
        super().__init__()
        h = config["hidden_size"]
        self.attention = config["attention"]
        self.residual = config["residual"]
        self.encoder = nn.LSTM(features, h, batch_first=True)
        self.decoder = nn.LSTM(h, h, batch_first=True)
        self.dropout = nn.Dropout(config["dropout"])
        self.output = nn.Linear(2*h if self.attention else h, 1)
        if self.residual:
            nn.init.zeros_(self.output.weight)
            nn.init.zeros_(self.output.bias)

    def forward(self, X, last):
        memory, (hidden, cell) = self.encoder(X)
        repeated_context = hidden[-1][:, None].repeat(1, 5, 1)
        decoded, _ = self.decoder(repeated_context, (hidden, cell))
        if self.attention:
            context, _ = dot_attention(decoded, memory)
            decoded = torch.cat([decoded, context], dim=-1)
        prediction = self.output(self.dropout(decoded)).squeeze(-1)  # [batch, 5]
        if self.residual:
            prediction = prediction + last[:, None]
        return prediction


# %% 3. Prepare data, train, and save a run
def run_experiment(settings=None, evaluate_test=True, output_dir=None):
    config = safe_training_config({**CONFIG, **(settings or {})})
    device = windows_device(config["seed"])
    parts, scalers, horizons, identity = prepare_forecast("task2", config)
    for split, part in parts.items():
        print(split, "X:", part["X"].shape, "y:", part["y"].shape, flush=True)
    print("First training origin:", parts["train"]["origins"][0], "targets:", parts["train"]["targets"][0], flush=True)
    model = ForecastSeq2Seq(len(scalers["features"]), config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    path = new_run("task2", config, identity, output_dir)
    save_json(path / "scalers.json", scalers)
    save_json(path / "data_shapes.json", {s: {"X": list(p["X"].shape), "y": list(p["y"].shape), "raw_rows": p["raw_rows"]} for s, p in parts.items()})
    print(model, flush=True)
    fit_forecast(model, optimizer, parts, config, scalers, path, device)
    evaluate_forecast(model, parts, scalers, horizons, path, device, "validation")
    if evaluate_test:
        evaluate_forecast(model, parts, scalers, horizons, path, device, "test")
    return path


# %% 4. Evaluate a saved checkpoint
def evaluate_run(path, split="test"):
    path = Path(path)
    config = json.loads((path / "config.json").read_text(encoding="utf-8"))
    device = windows_device(config["seed"])
    parts, scalers, horizons, identity = prepare_forecast("task2", config)
    assert identity == json.loads((path / "metadata.json").read_text(encoding="utf-8"))["data_id"]
    model = ForecastSeq2Seq(len(scalers["features"]), config).to(device)
    model.load_state_dict(torch.load(path / "best.pt", map_location=device, weights_only=True)["model_state"])
    return evaluate_forecast(model, parts, scalers, horizons, path, device, split)


# %% 5. PowerShell entry point
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--seed", type=int, choices=[42, 123, 2026], default=42)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if args.test:
        if args.checkpoint is None: parser.error("--test requires --checkpoint RUN_DIRECTORY")
        evaluate_run(args.checkpoint)
    elif args.train:
        if not 1 <= args.epochs <= 10: parser.error("Recovery runs allow 1 to 10 epochs.")
        run_experiment({"seed": args.seed, "epochs": args.epochs,
            "name": CONFIG["name"]+f"_recovery_{args.epochs}epochs", "recovery_check": True},
            evaluate_test=False, output_dir=args.output_dir)
    else:
        parser.print_help()


if __name__ == "__main__":
    if "__file__" in globals():
        main()
    elif TRAIN:
        notebook_run = run_experiment(CONFIG)

