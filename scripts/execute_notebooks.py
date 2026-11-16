"""Execute each notebook in a fresh Windows kernel and retain its real output."""

import bootstrap  # Resolve src/ and the project root.

from datetime import datetime
from pathlib import Path
import asyncio
import json
import sys
import nbformat
from nbclient import NotebookClient
from utils import ROOT, save_json, file_hash


def main(tasks=("task1", "task2", "task3")):
    if sys.platform != "win32":
        raise RuntimeError("Use Windows Python for notebook execution.")
    if Path(sys.executable).resolve() != (ROOT/".venv/Scripts/python.exe").resolve():
        raise RuntimeError("Use the project's Windows .venv.")
    # The selector loop is supported by the installed Windows Jupyter/ZMQ stack.
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    record_file = ROOT/"run/notebook_execution.json"
    old_records = json.loads(record_file.read_text(encoding="utf-8")) if record_file.exists() else []
    records = [record for record in old_records if record["task"] not in tasks]
    for task in tasks:
        path = ROOT/task/(task+".ipynb")
        notebook = nbformat.read(path, as_version=4)
        settings = next(c.source for c in notebook.cells if c.cell_type == "code")
        if "TRAIN = False" not in settings:
            raise RuntimeError("Notebook verification must display saved results without retraining.")
        backup = ROOT/"run"/"notebook_backups"/datetime.now().strftime("%Y%m%d_%H%M%S")
        backup.mkdir(parents=True, exist_ok=True)
        nbformat.write(notebook, backup/path.name)
        print("Fresh Windows notebook kernel:", task, flush=True)
        client = NotebookClient(notebook, timeout=None, kernel_name="final-project-windows",
                                resources={"metadata": {"path": str(ROOT/task)}})
        try:
            client.execute()
        except Exception:
            nbformat.write(notebook, backup/(task+"_failed.ipynb"))
            raise
        nbformat.write(notebook, path)
        records.append({"task": task, "notebook_sha256": file_hash(path),
                        "executed_code_cells": sum(c.cell_type == "code" and c.execution_count is not None for c in notebook.cells),
                        "kernel": "final-project-windows", "full_train_default": False,
                        "cuda_initialized": False,
                        "note": "Fresh kernel, saved results only. No GPU training was launched."})
        save_json(ROOT/"run"/"notebook_execution.json", records)
        print("Saved real notebook output:", task, flush=True)


if __name__ == "__main__":
    main()
