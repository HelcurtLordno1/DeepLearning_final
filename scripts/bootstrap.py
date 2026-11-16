"""Project paths for command-line tools; no model or training logic."""
from pathlib import Path
import os
import sys

ROOT = Path(__file__).resolve().parents[1]
for folder in [ROOT, ROOT/"src", ROOT/"scripts"]:
    sys.path.insert(0, str(folder))
os.environ["PYTHONPATH"] = os.pathsep.join([str(ROOT/"src"), str(ROOT), str(ROOT/"scripts"), os.environ.get("PYTHONPATH", "")])
