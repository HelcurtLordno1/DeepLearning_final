"""Run with Windows Python to record the environment used for this project."""

import bootstrap  # Resolve src/ and the project root.

from pathlib import Path
from datetime import datetime
import importlib.metadata as metadata
import importlib
import json
import platform
import subprocess
import sys


def main():
    if sys.platform != "win32":
        raise RuntimeError("Use .venv/Scripts/python.exe from Windows PowerShell.")
    packages = {}
    for name in ["torch", "torchvision", "numpy", "pandas", "matplotlib",
                 "scikit-learn", "Pillow", "nbformat", "nbclient", "ipykernel"]:
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = None
    info = {"platform": platform.platform(), "python": sys.version, "prefix": sys.prefix,
            "executable": sys.executable, "packages": packages}
    if packages["torch"]:
        import torch
        info["torch_import_path"] = torch.__file__
        info["cuda_available"] = torch.cuda.is_available()
        info["torch_cuda_version"] = torch.version.cuda
        if info["cuda_available"]:
            # Read device information without allocating CUDA tensors/context.
            inventory = subprocess.run(["nvidia-smi.exe", "--id=0", "--query-gpu=name,memory.total",
                                        "--format=csv,noheader"], capture_output=True, text=True, timeout=10)
            if inventory.returncode:
                raise RuntimeError("Could not read GPU inventory.")
            info["gpu_inventory"] = inventory.stdout.strip()
        info["cuda_initialized"] = torch.cuda.is_initialized()
        info["cuda_tensor_probe_performed"] = False
    missing = [name for name, version in packages.items() if version is None]
    if not missing:
        info["module_paths"] = {name: importlib.import_module(name).__file__ for name in
                                ["torch", "numpy", "pandas", "sklearn", "PIL", "matplotlib"]}
        site = Path(__file__).resolve().parents[1] / ".venv" / "Lib" / "site-packages"
        info["imports_inside_project_venv"] = all(Path(p).is_relative_to(site) for p in info["module_paths"].values())
    output = Path(__file__).resolve().parents[1] / "run" / "environment"
    output.mkdir(parents=True, exist_ok=True)
    target = output / (datetime.now().strftime("%Y%m%d_%H%M%S") + ".json")
    target.write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(json.dumps(info, ensure_ascii=True, indent=2))
    if missing:
        raise RuntimeError("Missing required packages: " + ", ".join(missing))
    if not info.get("cuda_available") or not info.get("imports_inside_project_venv"):
        raise RuntimeError("Use the complete project Windows .venv with CUDA support.")


if __name__ == "__main__":
    main()
