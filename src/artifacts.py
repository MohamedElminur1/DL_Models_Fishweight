"""Where Phase 4 results go, and how they are written.

`Layout` maps (task, scheme, model) to the paths of predictions, tuning records, the runtime
log and the files of later steps, under `results/` or, for --dev, `results/dev/`. All writes
go through a temporary file in the same directory and `os.replace`, so a run that dies halfway
never leaves a truncated file, and gzip output has a fixed timestamp so that the same content
gives the same bytes. `append_runtime_rows` replaces the rows of one (task, scheme, model) in
the shared runtime log under a file lock, so re-running a model never duplicates its rows and
two models can finish at the same time.
"""

from __future__ import annotations

import fcntl
import functools
import hashlib
import json
import os
import platform
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]

RUNTIME_COLUMNS = [
    "task", "scheme", "model", "fold", "stage", "trial_or_seed", "epochs", "seconds",
    "cpu", "python", "torch", "notes",
]


def repo_path(relative: str | Path) -> Path:
    """Absolute path of a repo-relative path (an absolute path is returned unchanged)."""
    return REPO_ROOT / relative


def model_key(model: str, sample_set: str = "eligible") -> str:
    """The name under which a run is stored: the model, with `__all` for the all-events set."""
    return model if sample_set == "eligible" else f"{model}__{sample_set}"


class Layout:
    """Result paths under `results/` (or `results/dev/` when `dev` is true)."""

    def __init__(self, cfg: dict, dev: bool = False):
        self.dev = dev
        self.root = repo_path(cfg["paths"]["dev_results" if dev else "results"])

    def predictions(self, task: str, scheme: str, key: str) -> Path:
        return self.root / "predictions" / task / scheme / f"{key}.csv"

    def tuning(self, task: str, scheme: str, key: str) -> Path:
        return self.root / "tuning" / task / scheme / key

    def runtime_log(self) -> Path:
        return self.root / "logs" / "runtime.csv"

    def selection(self, task: str) -> Path:
        return self.root / "selection" / f"primary_{task}.json"

    def permutation(self, task: str, model: str) -> Path:
        return self.root / "permutation" / task / f"{model}.csv"

    def metrics(self, name: str) -> Path:
        return self.root / "metrics" / name

    def stats(self, name: str) -> Path:
        return self.root / "stats" / name


def write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def write_json_atomic(path: Path, payload: dict) -> None:
    write_text_atomic(path, json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")


def write_csv_atomic(frame: pd.DataFrame, path: Path, float_format: str | None = None) -> None:
    """Write `frame` without index (gzip with a fixed timestamp when the name ends in .gz)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    compression = {"method": "gzip", "mtime": 0} if path.suffix == ".gz" else None
    frame.to_csv(tmp, index=False, compression=compression, float_format=float_format)
    os.replace(tmp, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def cpu_model() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


@functools.lru_cache(maxsize=1)
def _hardware() -> tuple[tuple[str, str], ...]:
    import torch  # imported here so that scripts that never train do not load it

    return (("cpu", cpu_model()), ("python", platform.python_version()), ("torch", torch.__version__))


def hardware() -> dict[str, str]:
    """CPU model, Python and torch versions, as written to every runtime row."""
    return dict(_hardware())


def runtime_row(task: str, scheme: str, model: str, fold, stage: str, trial_or_seed="",
                epochs="", seconds=0.0, notes: str = "") -> dict:
    """One row of the runtime log with the hardware columns filled in."""
    return {"task": task, "scheme": scheme, "model": model, "fold": fold, "stage": stage,
            "trial_or_seed": trial_or_seed, "epochs": epochs, "seconds": round(float(seconds), 4),
            **hardware(), "notes": notes}


def append_runtime_rows(path: Path, rows: list[dict], task: str, scheme: str, model: str) -> None:
    """Replace the rows of (task, scheme, model) in the runtime log by `rows`, under a file lock."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lock = path.with_name(path.name + ".lock")
    with lock.open("w") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            new = pd.DataFrame(rows, columns=RUNTIME_COLUMNS)
            if path.exists():
                old = pd.read_csv(path, dtype=str, keep_default_na=False)
                same = (old["task"] == task) & (old["scheme"] == scheme) & (old["model"] == model)
                new = pd.concat([old[~same], new.astype(str)], ignore_index=True)
            write_csv_atomic(new[RUNTIME_COLUMNS], path)
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)
