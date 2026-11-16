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
from utils import (windows_device, prepare_plate_cache, PlateDataset, plate_loader,
                   dot_attention, new_run, save_json, fit_plates, evaluate_plates, safe_training_config)

TRAIN = False  # Opening a notebook must not automatically start GPU training.
BASE_CONFIG = {"name": "cnn_gru_attention", "seed": 42, "hidden_size": 128,
          "attention": True, "position": False, "flat_context": False,
          "augment": False, "crop_mode": "bbox", "learning_rate": .001,
          "batch_size": 16, "epochs": 80, "patience": 10, "workers": 0}
CONFIG = BASE_CONFIG.copy()
selected_file = PROJECT_ROOT / "task3" / "selected_config.json"
if selected_file.exists():
    CONFIG.update(json.loads(selected_file.read_text(encoding="utf-8")))


# %% 2. CNN encoder and a seven-step GRU decoder
class PlateSeq2Seq(nn.Module):
    def __init__(self, classes, config):
        super().__init__()
        h = config["hidden_size"]
        self.attention = config["attention"]
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, h, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d((2, 1))
        )
        # Hai tùy chọn để so sánh sau baseline; cấu hình đang báo cáo tắt cả hai.
        self.position_embedding = None
        if config["position"]:
            self.position_embedding = nn.Parameter(torch.randn(1, 48, h) * .02)
        self.context_layer = None
        if config["flat_context"]:
            self.context_layer = nn.Linear(48 * h, h)
        self.decoder = nn.GRU(h, h, batch_first=True)
        self.output = nn.Linear(2*h if self.attention else h, classes)

    def forward(self, X, return_attention=False):
        features = self.cnn(X)                       # [B, H, 8, 48]
        memory = features.mean(dim=2).transpose(1, 2) # [B, 48, H]
        if self.position_embedding is not None:
            memory = memory + self.position_embedding
        context = memory.mean(dim=1)                 # [B, H]
        if self.context_layer is not None:
            context = torch.tanh(self.context_layer(memory.flatten(1)))
        repeated_context = context[:, None].repeat(1, 7, 1)
        decoded, _ = self.decoder(repeated_context, context[None])
        weights = None
        if self.attention:
            attended, weights = dot_attention(decoded, memory)
            decoded = torch.cat([decoded, attended], dim=-1)
        logits = self.output(decoded)                # [B, 7, vocabulary_size]
        if return_attention:
            return logits, weights
        return logits


# %% 2.1. Keep vertical detail and give the GRU seven ordered image regions
class AlignedPlateSeq2Seq(nn.Module):
    def __init__(self, classes, config):
        super().__init__()
        h = config["hidden_size"]
        self.attention = True
        self.alignment = config.get("alignment", .5)
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, h, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d((2, 1))
        )
        self.height_projection = nn.Linear(8 * h, h)
        # Start as the old height average; training can learn different weights
        # for the top and bottom of a character instead of losing that detail.
        with torch.no_grad():
            self.height_projection.weight.copy_(torch.eye(h).repeat(1, 8) / 8)
            self.height_projection.bias.zero_()
        self.character_regions = nn.AdaptiveAvgPool1d(7)
        self.decoder = nn.GRU(h, h, batch_first=True)
        self.output = nn.Linear(2 * h, classes)

    def forward(self, X, return_attention=False):
        features = self.cnn(X)                       # [B, H, 8, 48]
        columns = features.permute(0, 3, 2, 1).flatten(2)
        memory = self.height_projection(columns)    # [B, 48, H]
        context = memory.mean(dim=1)                # [B, H]
        regions = self.character_regions(memory.transpose(1, 2)).transpose(1, 2)
        # Seven left-to-right regions + global image information.
        decoder_input = (1-self.alignment) * context[:, None] + self.alignment * regions
        decoded, _ = self.decoder(decoder_input, context[None])
        attended, weights = dot_attention(decoded, memory)
        logits = self.output(torch.cat([decoded, attended], dim=-1))
        if return_attention:
            return logits, weights
        return logits


def recognizer_class(config):
    if config.get("architecture") == "refined":
        from task3.code.refined_model import RefinedPlateSeq2Seq
        return RefinedPlateSeq2Seq
    if config.get("architecture") == "aligned":
        return AlignedPlateSeq2Seq
    return PlateSeq2Seq


# %% 3. Read the fixed manifest and train
def run_experiment(settings=None, evaluate_test=True, output_dir=None):
    config = safe_training_config({**CONFIG, **(settings or {})})
    device = windows_device(config["seed"])
    frame, vocabulary, cache, identity = prepare_plate_cache(config["crop_mode"])
    datasets = {split: PlateDataset(frame, vocabulary, cache, split,
                 augment=config["augment"] and split == "train") for split in ["train", "validation", "test"]}
    loaders = {split: plate_loader(dataset, config, shuffle=split == "train") for split, dataset in datasets.items()}
    print("Image counts:", {s: len(d) for s, d in datasets.items()}, flush=True)
    print("Input: [batch, 3, 64, 192]; output: [batch, 7,", len(vocabulary), "]", flush=True)
    model = recognizer_class(config)(len(vocabulary), config).to(device)
    if config.get("initial_checkpoint"):
        initial_path = PROJECT_ROOT/config["initial_checkpoint"]
        initial_metadata = json.loads((initial_path.parent/"metadata.json").read_text(encoding="utf-8"))
        initial_vocabulary = json.loads((initial_path.parent/"vocabulary.json").read_text(encoding="utf-8"))
        assert initial_metadata["data_id"] == identity + ":" + config["crop_mode"]
        assert initial_vocabulary == vocabulary
        checkpoint = torch.load(initial_path, map_location=device, weights_only=True)
        missing, unexpected = model.load_state_dict(checkpoint["model_state"], strict=False)
        new_layers = ("position_embedding", "column_positions", "height_projection.", "refinement.")
        assert not unexpected and all(name.startswith(new_layers) for name in missing)
        print("Fine-tune attention model; learn the added column positions.", flush=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    path = new_run("task3", config, identity + ":" + config["crop_mode"], output_dir)
    save_json(path / "vocabulary.json", vocabulary)
    print(model, flush=True)
    fit_plates(model, optimizer, loaders, config, vocabulary, path, device)
    evaluate_plates(model, loaders["validation"], frame, vocabulary, path, device, "validation")
    if evaluate_test:
        evaluate_plates(model, loaders["test"], frame, vocabulary, path, device, "test")
    return path


# %% 4. Evaluate a saved recognizer
def evaluate_run(path, split="test"):
    path = Path(path)
    config = json.loads((path / "config.json").read_text(encoding="utf-8"))
    device = windows_device(config["seed"])
    frame, vocabulary, cache, identity = prepare_plate_cache(config["crop_mode"])
    assert identity + ":" + config["crop_mode"] == json.loads((path / "metadata.json").read_text(encoding="utf-8"))["data_id"]
    assert vocabulary == json.loads((path / "vocabulary.json").read_text(encoding="utf-8"))
    dataset = PlateDataset(frame, vocabulary, cache, split)
    loader = plate_loader(dataset, config)
    model = recognizer_class(config)(len(vocabulary), config).to(device)
    model.load_state_dict(torch.load(path / "best.pt", map_location=device, weights_only=True)["model_state"])
    return evaluate_plates(model, loader, frame, vocabulary, path, device, split)


# %% 5. PowerShell entry point
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--improve", action="store_true", help="Fine-tune attention with column positions; validation only")
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
        trial = json.loads((PROJECT_ROOT/"task3/improvement_config.json").read_text(encoding="utf-8")) if args.improve else CONFIG.copy()
        trial.update({"seed": args.seed, "epochs": args.epochs,
            "name": trial["name"]+f"_recovery_{args.epochs}epochs", "recovery_check": True})
        run_experiment(trial,
            evaluate_test=False, output_dir=args.output_dir)
    else:
        parser.print_help()


if __name__ == "__main__":
    if "__file__" in globals():
        main()
    elif TRAIN:
        notebook_run = run_experiment(CONFIG)
