"""Step 2 of the Phase 4 data pipeline: quality control of the timestamped raw sensor rows.

Run from the repo root:  .venv/bin/python -I src/qc.py

For every cohort pond: keep the rows with a clock time (ts_kind in the config), drop exact
duplicate lines, sort by ts with ties by file row, then test each sensor's raw values in that
order. A value that fails becomes missing (never clipped). A value is counted under the first
rule it fails, in this order: non-finite, out of range, flat-line. Ammonia is tested on the
raw value and log10-transformed afterwards. Writes data/interim/qc_rows_pondNN.csv.gz
(row, ts, date, hour, one QC'd column per sensor) and results/qc/qc_counts.csv.
Specification: ANALYSIS_PLAN.md section 2.5, DECISIONS.md Gate 1, tasks/phase4_experiments.md
step 2. Thresholds: configs/phase4_data.yaml.
"""

from __future__ import annotations

import datetime
import sys
import time
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.load import load_pond  # noqa: E402
from src.pipeline_io import load_config, log_run, repo_path, write_csv  # noqa: E402

# Outcome per raw value. The numeric order is only a label; precedence is set in `classify`.
VALID, MISSING, NONFINITE, OUT_OF_RANGE, FLATLINE = 0, 1, 2, 3, 4

# The five outcomes of a value; they add up to n_values.
OUTCOME_COLUMNS = ["n_missing_raw", "n_nonfinite", "n_out_of_range", "n_flatline", "n_valid"]
COUNT_COLUMNS = ["pond", "sensor", "n_values", *OUTCOME_COLUMNS, "n_duplicate_rows_dropped"]


@dataclass(frozen=True)
class SensorSpec:
    """One sensor of the config: output name, loader column, range bounds, log10 flag."""

    name: str
    raw: str
    lo: float | None
    lo_inclusive: bool
    hi: float | None
    hi_inclusive: bool
    log10: bool


def sensor_specs(cfg: dict) -> list[SensorSpec]:
    """The sensors of the config in output order."""
    return [SensorSpec(**entry) for entry in cfg["sensors"]]


def in_range(values: np.ndarray, spec: SensorSpec) -> np.ndarray:
    """True where a value is finite and inside the sensor's range (bounds as configured)."""
    ok = np.isfinite(values)
    if spec.lo is not None:
        ok &= (values >= spec.lo) if spec.lo_inclusive else (values > spec.lo)
    if spec.hi is not None:
        ok &= (values <= spec.hi) if spec.hi_inclusive else (values < spec.hi)
    return ok


def flatline_mask(
    values: np.ndarray, ts: np.ndarray, limit: np.timedelta64, max_gap: np.timedelta64
) -> np.ndarray:
    """True for every value that belongs to a flat-line run of at least `limit`.

    `values` (float) and `ts` (datetime64) are one sensor's rows in timestamp order. A run is
    a maximal sequence of consecutive rows with the identical finite value. It ends at a
    missing or non-finite value, and where two consecutive rows are more than `max_gap`
    apart. A run whose last ts minus first ts is at least `limit` is flagged as a whole.
    Raises ValueError when `ts` is not sorted.
    """
    n = values.size
    if n == 0:
        return np.zeros(0, dtype=bool)
    if np.any(ts[1:] < ts[:-1]):
        raise ValueError("rows must be in timestamp order")
    finite = np.isfinite(values)
    extends_previous = np.zeros(n, dtype=bool)
    extends_previous[1:] = (
        finite[1:] & finite[:-1] & (values[1:] == values[:-1]) & ((ts[1:] - ts[:-1]) <= max_gap)
    )
    run_starts = np.flatnonzero(finite & ~extends_previous)
    ends_run = finite.copy()
    ends_run[:-1] &= ~extends_previous[1:]
    run_ends = np.flatnonzero(ends_run)
    long_run = (ts[run_ends] - ts[run_starts]) >= limit
    change = np.zeros(n + 1, dtype=np.int64)
    change[run_starts[long_run]] += 1
    change[run_ends[long_run] + 1] -= 1
    return np.cumsum(change[:-1]) > 0


def classify(
    values: np.ndarray,
    ts: np.ndarray,
    spec: SensorSpec,
    limit: np.timedelta64,
    max_gap: np.timedelta64,
) -> np.ndarray:
    """Outcome code per value: VALID, or the first rule it fails.

    MISSING is a NaN raw value (an empty cell), not a rule. Among the rules the order is
    non-finite (+-inf), out of range, flat-line. The flat-line test sees the raw values of
    the whole series, whatever their range.
    """
    codes = np.full(values.shape, VALID, dtype=np.int8)
    codes[flatline_mask(values, ts, limit, max_gap)] = FLATLINE
    codes[np.isfinite(values) & ~in_range(values, spec)] = OUT_OF_RANGE
    codes[np.isinf(values)] = NONFINITE
    codes[np.isnan(values)] = MISSING
    return codes


def qc_values(values: np.ndarray, codes: np.ndarray, spec: SensorSpec) -> np.ndarray:
    """The raw value where it passed QC (log10 of it for ammonia), NaN everywhere else."""
    out = np.full(values.shape, np.nan)
    valid = codes == VALID
    out[valid] = np.log10(values[valid]) if spec.log10 else values[valid]
    return out


def check_valid_in_range(values: np.ndarray, codes: np.ndarray, spec: SensorSpec) -> None:
    """Raise RuntimeError when a value kept as valid is outside the range or cannot be logged."""
    kept = values[codes == VALID]
    if not in_range(kept, spec).all():
        raise RuntimeError(f"{spec.name}: a valid value lies outside its range")
    if spec.log10 and not (kept > 0).all():
        raise RuntimeError(f"{spec.name}: a valid value cannot be log10-transformed")


def drop_exact_duplicates(frame: pd.DataFrame, key_columns: list[str]) -> tuple[pd.DataFrame, int]:
    """Drop rows that repeat an earlier row (lower file row) in every key column.

    Missing values compare equal to each other. Returns the kept rows in file order and the
    number dropped.
    """
    ordered = frame.sort_values("row")
    repeated = ordered.duplicated(subset=key_columns, keep="first")
    return ordered.loc[~repeated], int(repeated.sum())


def timed_rows(frame: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, int]:
    """The rows that enter QC in QC order, and the number of duplicate rows dropped.

    Keeps rows whose ts_kind is in the config, drops exact duplicate lines, then sorts by ts
    with ties by file row.
    """
    timed = frame.loc[frame["ts_kind"].isin(cfg["qc"]["ts_kinds"])]
    kept, n_dropped = drop_exact_duplicates(timed, cfg["qc"]["duplicate_key_columns"])
    return kept.sort_values(["ts", "row"]).reset_index(drop=True), n_dropped


def qc_pond(frame: pd.DataFrame, pond: int, cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """QC one pond's `load_pond` frame.

    Returns (rows, counts). `rows` has one line per row that entered QC: row, ts, date, hour
    and the QC'd value of every sensor (NaN when invalid, ammonia as log10). `counts` has one
    line per sensor with the columns of COUNT_COLUMNS; n_values counts the rows after the
    duplicate drop and equals the sum of the five outcome columns.
    """
    limit = pd.Timedelta(hours=cfg["qc"]["flatline_hours"]).to_timedelta64()
    max_gap = pd.Timedelta(hours=cfg["qc"]["flatline_max_gap_hours"]).to_timedelta64()
    timed, n_dropped = timed_rows(frame, cfg)
    ts = timed["ts"].to_numpy()
    rows = pd.DataFrame(
        {
            "row": timed["row"].to_numpy(),
            "ts": timed["ts"].to_numpy(),
            "date": timed["date"].to_numpy(),
            "hour": timed["ts"].dt.hour.to_numpy(),
        }
    )
    counts = []
    for spec in sensor_specs(cfg):
        raw = timed[spec.raw].to_numpy(dtype="float64")
        codes = classify(raw, ts, spec, limit, max_gap)
        check_valid_in_range(raw, codes, spec)
        rows[spec.name] = qc_values(raw, codes, spec)
        entry = {
            "pond": pond,
            "sensor": spec.name,
            "n_values": len(timed),
            "n_missing_raw": int((codes == MISSING).sum()),
            "n_nonfinite": int((codes == NONFINITE).sum()),
            "n_out_of_range": int((codes == OUT_OF_RANGE).sum()),
            "n_flatline": int((codes == FLATLINE).sum()),
            "n_valid": int((codes == VALID).sum()),
            "n_duplicate_rows_dropped": n_dropped,
        }
        if sum(entry[c] for c in OUTCOME_COLUMNS) != entry["n_values"]:
            raise RuntimeError(f"pond {pond} {spec.name}: outcome counts do not add up")
        counts.append(entry)
    return rows, pd.DataFrame(counts, columns=COUNT_COLUMNS)


def percent_table(counts: pd.DataFrame) -> pd.DataFrame:
    """Each outcome count as a percentage of n_values, for the console summary."""
    out = counts[["pond", "sensor", "n_values"]].copy()
    for column in OUTCOME_COLUMNS:
        out["pct_" + column[2:]] = (100.0 * counts[column] / counts["n_values"]).round(2)
    out["n_dup_dropped"] = counts["n_duplicate_rows_dropped"]
    return out


def main() -> None:
    started = datetime.datetime.now(datetime.timezone.utc)
    clock = time.perf_counter()
    cfg = load_config()

    count_parts = []
    for pond in cfg["cohort"]["ponds"]:
        rows, counts = qc_pond(load_pond(pond), pond, cfg)
        written = rows.assign(
            ts=rows["ts"].dt.strftime("%Y-%m-%d %H:%M:%S"),
            date=rows["date"].dt.strftime("%Y-%m-%d"),
        )
        write_csv(written, repo_path(cfg["paths"]["qc_rows"].format(pond=pond)))
        count_parts.append(counts)
        dropped = counts["n_duplicate_rows_dropped"].iloc[0]
        print(f"pond {pond}: {len(rows)} rows after QC, {dropped} duplicate rows dropped")
    counts = pd.concat(count_parts, ignore_index=True)
    write_csv(counts, repo_path(cfg["paths"]["qc_counts"]))

    print(percent_table(counts).to_string(index=False))
    seconds = time.perf_counter() - clock
    log_run(cfg, "qc.py", started, seconds)
    print(f"done in {seconds:.1f} s")


if __name__ == "__main__":
    main()
