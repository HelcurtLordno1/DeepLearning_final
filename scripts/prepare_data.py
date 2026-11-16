"""Check downloaded CCPD images and prepare their fixed grouped split.

Run from the project root: python prepare_data.py
Requires Pillow. This prepares data only; it does not train a model.
The saved selection also works with the source train.zip or CCPD2019.tar.xz.
"""

import bootstrap  # Resolve src/ and the project root.

from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
import csv
import hashlib
import json
import random
import shutil
import tarfile
import zipfile

from PIL import Image


# 1. Settings and the character tables from the official CCPD README.
ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "ccpd"
IMAGE_DIR = RAW / "ccpd_base"
SPLIT_DIR = ROOT / "data" / "splits"
SEED = 42
NUMBER_OF_IMAGES = 50_000
MINIMUM_FREE_GIB = 20

PROVINCES = list("皖沪津渝冀晋蒙辽吉黑苏浙京闽赣鲁豫鄂湘粤桂琼川贵云藏陕甘青宁新警学")
LETTERS = list("ABCDEFGHJKLMNPQRSTUVWXYZ")
CHARACTERS = LETTERS + list("0123456789")


def read_annotation(filename):
    """Convert the encoded filename into the full plate and its coordinates."""
    fields = Path(filename).stem.split("-")
    indices = [int(value) for value in fields[4].split("_")]
    if len(indices) != 7 or any(value < 0 for value in indices):
        raise ValueError("Expected seven valid character indices")

    # The source's final O entry means 'no character'; omit it from our tables.
    text = PROVINCES[indices[0]] + LETTERS[indices[1]]
    text += "".join(CHARACTERS[value] for value in indices[2:])
    box = [int(value) for pair in fields[2].split("_") for value in pair.split("&")]
    corners = [int(value) for pair in fields[3].split("_") for value in pair.split("&")]
    if len(box) != 4 or len(corners) != 8:
        raise ValueError("Invalid box or corner coordinates")
    if not (0 <= box[0] < box[2] <= 720 and 0 <= box[1] < box[3] <= 1160):
        raise ValueError("Box outside the documented image bounds")
    return {"filename": filename, "text": text, "box": box, "corners": corners}


def select_images():
    """Choose a repeatable random subset, independently of any model results."""
    selection = SPLIT_DIR / "ccpd_download_selection.txt"
    if selection.exists():
        filenames = selection.read_text().splitlines()
        if len(filenames) != NUMBER_OF_IMAGES or len(set(filenames)) != NUMBER_OF_IMAGES:
            raise ValueError("The saved download selection must contain 50,000 distinct images")
        chosen = [read_annotation(name) for name in filenames]
        print(f"Using the saved selection of {len(chosen):,} images.", flush=True)
        return chosen, {}

    paths = set()
    for name in ["train.txt", "val.txt"]:
        paths.update((RAW / "official_splits" / name).read_text().splitlines())

    candidates, excluded = [], Counter()
    for path in sorted(paths):
        if not path.startswith("ccpd_base/"):
            continue
        try:
            candidates.append(read_annotation(Path(path).name))
        except (ValueError, IndexError):
            excluded["invalid_filename_label_or_box"] += 1

    chosen = random.Random(SEED).sample(candidates, NUMBER_OF_IMAGES)
    print(f"Valid filename candidates: {len(candidates):,}; selected: {len(chosen):,}", flush=True)
    return chosen, dict(excluded)


def extract_images(chosen):
    """Extract only selected original JPEGs; never unpack the entire archive."""
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    wanted = {row["filename"] for row in chosen}
    existing = {path.name for path in IMAGE_DIR.iterdir()}
    missing = wanted - existing
    if not missing:
        print("Selected images are already extracted.", flush=True)
        return

    print(f"Extracting {len(missing):,} selected images...", flush=True)
    zip_path = RAW / "train.zip"
    if zip_path.exists():
        with zipfile.ZipFile(zip_path) as source:
            members = {Path(item.filename).name: item for item in source.infolist()}
            for name in sorted(missing):
                member = members[name]
                if shutil.disk_usage(ROOT).free < MINIMUM_FREE_GIB * 2**30 + member.file_size:
                    raise RuntimeError("Stopped extraction to preserve 20 GiB of free storage")
                with source.open(member) as image_file:
                    with (IMAGE_DIR / name).open("wb") as destination:
                        shutil.copyfileobj(image_file, destination)
        return

    archive = RAW / "CCPD2019.tar.xz"
    if not archive.exists():
        raise FileNotFoundError("Download the selected images or place the source archive in data/raw/ccpd")
    # Streaming avoids loading an index of every archive member into memory.
    with tarfile.open(archive, "r|xz") as source:
        for member in source:
            name = Path(member.name).name
            if name not in missing or "ccpd_base" not in Path(member.name).parts:
                continue
            if not member.isfile():
                continue
            if shutil.disk_usage(ROOT).free < MINIMUM_FREE_GIB * 2**30 + member.size:
                raise RuntimeError("Stopped extraction to preserve 20 GiB of free storage")
            # Write only a basename we selected; do not extract arbitrary paths.
            with source.extractfile(member) as image_file:
                with (IMAGE_DIR / name).open("wb") as destination:
                    shutil.copyfileobj(image_file, destination)
            missing.remove(name)
            completed = len(wanted) - len(missing)
            if completed % 5000 == 0:
                print(f"Extracted {completed:,}/{len(wanted):,}", flush=True)
            if not missing:
                break
    if missing:
        raise RuntimeError(f"Archive is missing {len(missing)} selected images")


def inspect_image(row):
    """Read one JPEG and check that its pixels and crop box are usable."""
    path = IMAGE_DIR / row["filename"]
    payload = path.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    # Reuse the bytes already read for hashing, keeping the disk work small.
    with Image.open(BytesIO(payload)) as image:
        width, height = image.size
        if image.format != "JPEG":
            raise ValueError(f"Expected JPEG: {path.name}")
        image.load()  # Decode pixels too; checking the JPEG header alone is insufficient.
    x1, y1, x2, y2 = row["box"]
    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        raise ValueError(f"Invalid crop: {path.name}")
    return width, height, digest, len(payload)


def check_images(chosen):
    """Check images in four reading threads, then collect results in order."""
    dimensions, image_hashes = Counter(), set()
    image_bytes = 0
    # This is only to speed up the one-time disk audit; model code stays sequential.
    with ThreadPoolExecutor(max_workers=4) as readers:
        for index, (row, info) in enumerate(zip(chosen, readers.map(inspect_image, chosen)), 1):
            width, height, digest, size = info
            if digest in image_hashes:
                raise ValueError("Duplicate image content found; review before freezing the subset")
            image_hashes.add(digest)
            row["width"], row["height"] = width, height
            row["sha256"] = digest
            dimensions[f"{width}x{height}"] += 1
            image_bytes += size
            if index % 10000 == 0:
                print(f"Verified {index:,}/{len(chosen):,} JPEGs", flush=True)
    return dict(dimensions), image_bytes


def save_splits(chosen):
    """Assign whole plate strings to train/validation/test, then save one CSV."""
    groups = sorted({row["text"] for row in chosen})
    random.Random(SEED).shuffle(groups)
    train_end, val_end = int(len(groups) * 0.70), int(len(groups) * 0.85)
    partition = {}
    for name, values in [("train", groups[:train_end]),
                         ("validation", groups[train_end:val_end]),
                         ("test", groups[val_end:])]:
        partition.update({text: name for text in values})

    SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = SPLIT_DIR / "ccpd_subset.csv"
    columns = ["image_path", "text", "split", "width", "height", "bbox", "corners", "sha256"]
    with csv_path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=columns)
        writer.writeheader()
        for row in sorted(chosen, key=lambda item: item["filename"]):
            writer.writerow({
                "image_path": "data/raw/ccpd/ccpd_base/" + row["filename"],
                "text": row["text"], "split": partition[row["text"]],
                "width": row["width"], "height": row["height"],
                "bbox": json.dumps(row["box"]), "corners": json.dumps(row["corners"]),
                "sha256": row["sha256"],
            })
    vocabulary = PROVINCES + LETTERS + list("0123456789")
    (SPLIT_DIR / "ccpd_vocabulary.json").write_text(
        json.dumps(vocabulary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    counts = Counter(partition[row["text"]] for row in chosen)
    group_counts = Counter(partition.values())
    return dict(counts), dict(group_counts), hashlib.sha256(csv_path.read_bytes()).hexdigest()


def main():
    if shutil.disk_usage(ROOT).free < MINIMUM_FREE_GIB * 2**30:
        raise RuntimeError("Need at least 20 GiB free before preparing plates")
    chosen, excluded = select_images()
    extract_images(chosen)
    dimensions, image_bytes = check_images(chosen)
    counts, group_counts, manifest_hash = save_splits(chosen)
    report = {
        "dataset": "CCPD2019-Base, fixed 50,000-image subset",
        "prepared_at_utc": datetime.now(timezone.utc).isoformat(),
        "selection_seed": SEED, "split_seed": SEED,
        "selection_file": "data/splits/ccpd_download_selection.txt",
        "selection_details": "See data/raw/ccpd/download_metadata.json for the pinned mirror and block sampling",
        "images": len(chosen), "image_bytes": image_bytes,
        "dimensions_width_x_height": dimensions,
        "filename_candidate_exclusions": excluded,
        "image_counts": counts, "plate_group_counts": group_counts,
        "plate_overlap_between_splits": 0, "exact_duplicate_images": 0,
        "manifest_sha256": manifest_hash,
        "protocol": "Custom plate-string-grouped split; annotation-based crops; not the official benchmark",
        "training_status": "not started",
    }
    (RAW / "preparation_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
