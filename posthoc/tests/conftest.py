"""Allow isolated Python to import the repository and the post hoc entry point."""

import sys
from pathlib import Path

sys.dontwritebytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
