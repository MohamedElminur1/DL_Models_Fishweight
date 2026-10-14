"""Config and file helpers shared by the Phase 4 data scripts (events.py, qc.py, features.py).

`load_config()` returns configs/phase4_data.yaml as a dict; every threshold of the data
pipeline lives there, and `grid_dates()` derives the sampling dates from it. `write_csv()`
writes through a temporary file in the same directory and renames it, so a run that dies
halfway never leaves a truncated output and a re-run replaces the file. `log_run()` appends
the runtime and hardware of one script run to the CSV named in the config.
"""

from __future__ import annotations

import datetime
import os
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "phase4_data.yaml"


def load_config(path: Path | None = None) -> dict:
    """The data-pipeline configuration (configs/phase4_data.yaml unless `path` is given)."""
    with (path or DEFAULT_CONFIG).open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def repo_path(relative: str) -> Path:
    """Absolute path of a repo-relative path string from the config."""
    return REPO_ROOT / relative


def grid_dates(cfg: dict) -> pd.DatetimeIndex:
    """The sampling dates day0 + period_days * k for k = k_first..k_last."""
    day0 = pd.Timestamp(cfg["cohort"]["day0"])
    grid = cfg["cohort"]["grid"]
    return pd.DatetimeIndex(
        [
            day0 + pd.Timedelta(days=grid["period_days"] * k)
            for k in range(grid["k_first"], grid["k_last"] + 1)
        ]
    )


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    """Write `frame` without index to `path` (gzip when the name ends in .gz), atomically.

    Creates missing directories. Dates must already be strings; floats are written with
    their shortest round-trip representation, so reading back with
    `float_precision="round_trip"` returns the same values.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    compression = {"method": "gzip", "mtime": 0} if path.suffix == ".gz" else None
    frame.to_csv(tmp, index=False, compression=compression)
    os.replace(tmp, path)


def _cpu_model() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def log_run(cfg: dict, script: str, started: datetime.datetime, seconds: float) -> None:
    """Append one row (script, start time, seconds, hardware, library versions) to the run log."""
    row = {
        "script": script,
        "started_utc": started.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "seconds": round(seconds, 1),
        "cpu_model": _cpu_model(),
        "n_cpu": os.cpu_count(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
    }
    path = repo_path(cfg["paths"]["run_log"])
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([row]).to_csv(path, mode="a", header=not path.exists(), index=False)
