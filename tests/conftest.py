"""Puts the repo root on sys.path so tests can `import src...` under `python -I -m pytest`."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
