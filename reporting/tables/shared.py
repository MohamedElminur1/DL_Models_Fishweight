"""Formatting and provenance for manuscript CSV tables, without analysis imports."""

import csv
import io
import json
import math
from pathlib import Path
import statistics
from collections.abc import Iterable, Sequence
import os
import tempfile

import yaml


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "manuscript/src/tables"
MISSING = "–"


def literal(value: object) -> str:
    """Escape source text that the manuscript builder could interpret as markup."""
    return "".join("\\" + c if c in "*_^[{@" else c for c in str(value))


def number(value: float | str | None, kind: str, decimals: int | None = None) -> str:
    """Format a saved metric; empty values stay unavailable rather than becoming zero."""
    if value is None or value == "":
        return MISSING
    numeric = float(value)
    if not math.isfinite(numeric):
        if numeric == math.inf and kind in {"s", "log"}:
            return "∞"
        raise ValueError(f"non-finite {kind}: {value}")
    if kind == "p":
        if not 0 <= numeric <= 1:
            raise ValueError(f"invalid p-value: {value}")
        return "< 0.001" if numeric < 0.001 else f"{numeric:.3f}"
    if kind in {"grams", "percent", "r2"} and abs(numeric) >= 10000:
        coefficient, exponent = f"{numeric:.1e}".split("e")
        return f"{coefficient.replace('-', '−')} × 10^{{{int(exponent)}}}"
    places = {"s": 3, "log": 3, "grams": 1 if abs(numeric) < 100 else 0,
              "percent": 1, "r2": 3}[kind] if decimals is None else decimals
    return f"{numeric:.{places}f}"


def mean_sd(values: Sequence[float], kind: str) -> str:
    """Report seed metrics as mean and sample SD without fixing the seed count here."""
    if len(values) < 2:
        raise ValueError("mean ± SD requires at least two seeds")
    mean = statistics.mean(values)
    places = len(number(mean, kind).split(" × ")[0].partition(".")[2])
    return f"{number(mean, kind)} ± {number(statistics.stdev(values), kind, places)}"


class Sources:
    """Record table inputs while excluding independent verification files."""

    def __init__(self) -> None:
        self.files: set[str] = set()

    def path(self, relative: str) -> Path:
        path = (ROOT / relative).resolve()
        name = path.relative_to(ROOT).as_posix()
        allowed = ("configs/", "data/processed/", "results/")
        if not name.startswith(allowed) and name not in {"ANALYSIS_PLAN.md", "DECISIONS.md", "reporting/names.py"}:
            raise ValueError(f"not a table input: {name}")
        self.files.add(name)
        return path

    def csv(self, relative: str) -> list[dict[str, str]]:
        with self.path(relative).open(encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))

    def config(self, relative: str) -> dict:
        with self.path(relative).open(encoding="utf-8") as handle:
            return yaml.safe_load(handle)

    def json(self, relative: str) -> dict:
        """Read saved selection records with provenance."""
        return json.loads(self.text(relative))

    def text(self, relative: str) -> str:
        """Read frozen plan text when a setting is not represented in the YAML."""
        return self.path(relative).read_text(encoding="utf-8")


def _replace(path: Path, content: str) -> None:
    """Replace one complete output atomically; unchanged outputs retain their timestamps."""
    if path.exists() and path.read_text(encoding="utf-8") == content:
        return
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="",
                                     dir=path.parent, prefix=path.name + ".", delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(content)
    os.replace(temporary, path)


def write_table(name: str, header: Sequence[str], rows: Iterable[Sequence[str]],
                sources: Sources, notes: Sequence[str]) -> None:
    """Write a rectangular CSV and its source list, with table footnotes in the sidecar.

    Re-running converges to the same bytes. An interrupted pair of replacements can leave
    an old sidecar beside a new CSV; rerun the generator and checker to finish the pair.
    """
    if not name.isidentifier() or not sources.files:
        raise ValueError("a table needs a simple name and recorded sources")
    data = list(rows)
    if len(set(header)) != len(header) or any(len(row) != len(header) for row in data):
        raise ValueError(f"{name}: duplicate headers or ragged rows")
    if any("\n" in cell or "\r" in cell for row in [header, *data] for cell in row):
        raise ValueError(f"{name}: table cells must be single-line")
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(header)
    writer.writerows(data)
    sidecar = "Files read (repository-relative):\n" + "".join(f"{p}\n" for p in sorted(sources.files))
    if any(" × 10^{" in cell for row in data for cell in row):
        notes = [*notes, "R2, gram and percentage values with absolute size 10,000 or more use two significant digits in builder markup, for example −4.3 × 10^{5}."]
    sidecar += "\nFooter / definitions / limitations:\n" + "".join(f"{note}\n" for note in notes)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    _replace(OUTPUT / f"{name}.csv", buffer.getvalue())
    _replace(OUTPUT / f"{name}.source.txt", sidecar)
    print(f"{name}: {len(data)} rows x {len(header)} columns")
