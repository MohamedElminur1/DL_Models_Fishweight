"""Canonical loader for the raw IoTPond CSV files (aquacj-4561230, Phase 1).

Use `load_pond(pond)` for analysis. It returns one row per data row of
`data/raw/pondNN.csv`, in file order, with canonical columns. It does not clean, drop,
impute, sort or convert time zones. Faults stay in the frame: text that does not parse as a
number becomes NaN, and a stamp without a clock time has `ts` NaT. `src/audit.py` counts
every such case. `read_raw(pond)` returns every cell as text, including the trailing
unnamed columns that `load_pond` drops.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = REPO_ROOT / "data" / "raw"

POND_IDS = (1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12)
SENSOR_VARS = ("temperature", "turbidity", "do", "ph", "ammonia", "nitrate")
MEASURE_VARS = ("population", "length", "weight")
NUMERIC_VARS = SENSOR_VARS + MEASURE_VARS

# Header text as written in each file, mapped to the canonical name. A file whose header
# differs from its entry in any way raises ValueError.
_COLS_1 = {
    "created_at": "ts",
    "entry_id": "entry_id",
    "Temperature (C)": "temperature",
    "Turbidity(NTU)": "turbidity",
    "Dissolved Oxygen(g/ml)": "do",
    "PH": "ph",
    "Ammonia(g/ml)": "ammonia",
    "Nitrate(g/ml)": "nitrate",
    "Population": "population",
    "Fish_Length(cm)": "length",
    "Fish_Weight(g)": "weight",
}
_COLS_2 = {
    "created_at": "ts",
    "entry_id": "entry_id",
    "Temperature (C)": "temperature",
    "Turbidity (NTU)": "turbidity",
    "Dissolved Oxygen(g/ml)": "do",
    "PH": "ph",
    "Ammonia(g/ml)": "ammonia",
    "Nitrate(g/ml)": "nitrate",
    "Population": "population",
    "Fish_Length (cm)": "length",
    "Fish_Weight (g)": "weight",
}
_COLS_3 = {
    "created_at": "ts",
    "entry_id": "entry_id",
    "Temperature(C)": "temperature",
    "Turbidity(NTU)": "turbidity",
    "Dissolved Oxygen(g/ml)": "do",
    "PH": "ph",
    "Ammonia(g/ml)": "ammonia",
    "Nitrate(g/ml)": "nitrate",
    "Population": "population",
    "Fish_Length(cm)": "length",
    "Fish_Weight(g)": "weight",
}
_COLS_7 = {
    "created_at": "ts",
    "Date": "date_col",
    "entry_id": "entry_id",
    "temperature(C)": "temperature",
    "turbidity (NTU)": "turbidity",
    "Dissolved Oxygen (g/ml)": "do",
    "PH": "ph",
    "ammonia(g/ml)": "ammonia",
    "nitrate(g/ml)": "nitrate",
    "Fish_length(cm)": "length",
    "Fish_weight(g)": "weight",
}
_COLS_8 = {
    "Unnamed: 0": "ts",
    "Date": "date_col",
    "entry_id": "entry_id",
    "Temperature (C)": "temperature",
    "Turbidity(NTU)": "turbidity",
    "Dissolved Oxygen (mg/L)": "do",
    "PH": "ph",
    "Ammonia (mg/L)": "ammonia",
    "Nitrate (mg/L)": "nitrate",
    "Total_length (cm)": "length",
    "Weight (g)": "weight",
}
_COLS_9 = {
    "created_at": "ts",
    "DATE": "date_col",
    "entry_id": "entry_id",
    "Temperature(C)": "temperature",
    "Turbidity(NTU)": "turbidity",
    "Dissolved Oxygen(g/ml)": "do",
    "PH": "ph",
    "Ammonia(g/ml)": "ammonia",
    "Nitrate(g/ml)": "nitrate",
    "Fish_length(cm)": "length",
    "Fish_weight(g)": "weight",
}
_COLS_10 = {
    "created_at": "ts",
    "entry_id": "entry_id",
    "TEMPERATURE": "temperature",
    "TURBIDITY": "turbidity",
    "DISOLVED OXYGEN": "do",
    "pH": "ph",
    "AMMONIA": "ammonia",
    "NITRATE": "nitrate",
    "Population": "population",
    "Length": "length",
    "Weight": "weight",
}
_COLS_11 = {
    "created_at": "ts",
    "entry_id": "entry_id",
    "TEMPERATURE": "temperature",
    "TURBIDITY": "turbidity",
    "DISOLVED OXYGEN": "do",
    "pH": "ph",
    "AMMONIA": "ammonia",
    "NITRATE": "nitrate",
    "Population": "population",
    "Lenght": "length",
    "Weight": "weight",
}

HEADER_MAP: dict[int, dict[str, str]] = {
    1: _COLS_1,
    2: _COLS_2,
    3: _COLS_3,
    4: _COLS_3,
    6: _COLS_3,
    7: _COLS_7,
    8: _COLS_8,
    9: _COLS_9,
    10: _COLS_10,
    11: _COLS_11,
    12: _COLS_10,
}

# Empty headers that pandas names "Unnamed: <index>". These columns are dropped from
# `load_pond` output; `audit.py` reports every non-empty cell in them.
TRAILING_UNNAMED: dict[int, tuple[str, ...]] = {
    7: ("Unnamed: 11", "Unnamed: 12", "Unnamed: 13"),
    8: ("Unnamed: 11",),
}

_UNIT_RX = re.compile(r"\(([^()]*)\)\s*$")
_DATETIME_LABEL_RX = re.compile(
    r"^(?P<y>\d{4})-(?P<mo>\d{2})-(?P<d>\d{2}) (?P<h>\d{2}):(?P<mi>\d{2}):(?P<s>\d{2}) "
    r"(?P<tz>[A-Z]{2,5})$"
)
_ISO_OFFSET_RX = re.compile(
    r"^(?P<y>\d{4})-(?P<mo>\d{2})-(?P<d>\d{2})T(?P<h>\d{2}):(?P<mi>\d{2}):(?P<s>\d{2})"
    r"(?P<tz>[+-]\d{2}:\d{2})$"
)
_DATE_ONLY_RX = re.compile(r"^(?P<d>\d{1,2})/(?P<mo>\d{1,2})/(?P<y>\d{4})$")


def raw_path(pond: int) -> Path:
    """Path to the read-only symlink for pond N (ponds are numbered as in the archive)."""
    _check_pond(pond)
    return RAW_DIR / f"pond{pond:02d}.csv"


def unit_as_labelled(header: str) -> str:
    """Unit text in parentheses at the end of a header, or "" when there is none.

    Used for both the `UNITS_AS_LABELLED` table and the schema audit. The unit is reported
    as written in the source (for example "g/ml" for ammonia in ponds 1 to 4), not
    converted.
    """
    match = _UNIT_RX.search(header)
    return match.group(1) if match else ""


def _units_for(pond: int) -> dict[str, str]:
    """Unit text for each numeric variable of one pond; "" when the variable is absent."""
    by_canonical = {canon: header for header, canon in HEADER_MAP[pond].items()}
    return {
        var: unit_as_labelled(by_canonical[var]) if var in by_canonical else ""
        for var in NUMERIC_VARS
    }


UNITS_AS_LABELLED: dict[int, dict[str, str]] = {pond: _units_for(pond) for pond in POND_IDS}


def _check_pond(pond: int) -> None:
    if pond not in HEADER_MAP:
        raise ValueError(f"unknown pond {pond!r}; pond ids are {POND_IDS} (there is no pond 5)")


def read_raw(pond: int) -> pd.DataFrame:
    """Every cell of pond N as text, every column, in file order. Empty cells are "".

    Raises ValueError when the header is not exactly the one in `HEADER_MAP` and
    `TRAILING_UNNAMED`. Used by `load_pond` and by the trailing-column audit.
    """
    path = raw_path(pond)
    expected = set(HEADER_MAP[pond]) | set(TRAILING_UNNAMED.get(pond, ()))
    raw = pd.read_csv(path, dtype=str, keep_default_na=False, na_filter=False, low_memory=False)
    found = list(raw.columns)
    unknown = [name for name in found if name not in expected]
    absent = sorted(expected - set(found))
    if unknown or absent or len(found) != len(expected):
        raise ValueError(
            f"pond {pond:02d}: header does not match the column table; "
            f"unknown={unknown} absent={absent}"
        )
    return raw


def _blank_to_na(text: pd.Series) -> pd.Series:
    """Replace empty strings with NA, keeping the text otherwise unchanged."""
    return text.mask(text.fillna("") == "")


def _to_float(text: pd.Series) -> pd.Series:
    """Parse as float64. Text that does not parse becomes NaN. "inf" parses to +/-inf."""
    return pd.to_numeric(_blank_to_na(text), errors="coerce").astype("float64")


def _to_integer(text: pd.Series) -> pd.Series:
    """Parse as nullable integer. Non-integral or non-numeric text becomes NA."""
    number = pd.to_numeric(_blank_to_na(text), errors="coerce").astype("float64")
    return number.where(number % 1 == 0).astype("Int64")


def _parse_stamp(text: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """Classify the timestamp text and return (ts_kind, tz_label, ts, date).

    `ts` is the clock time as written with any zone label or offset dropped (no conversion).
    `date` is the calendar date of `ts`, or of a day-first date-only string.
    """
    missing = text.isna()
    label = text.str.extract(_DATETIME_LABEL_RX)
    iso = text.str.extract(_ISO_OFFSET_RX)
    date_only = text.str.extract(_DATE_ONLY_RX)
    has_label = label["y"].notna()
    has_iso = iso["y"].notna()
    has_date = date_only["y"].notna()

    timed = label.combine_first(iso)
    clock = (
        timed["y"] + "-" + timed["mo"] + "-" + timed["d"]
        + " " + timed["h"] + ":" + timed["mi"] + ":" + timed["s"]
    )
    ts = pd.to_datetime(clock, format="%Y-%m-%d %H:%M:%S", errors="coerce")
    day_str = (
        date_only["y"] + "-" + date_only["mo"].str.zfill(2) + "-" + date_only["d"].str.zfill(2)
    )
    day_only = pd.to_datetime(day_str, format="%Y-%m-%d", errors="coerce")

    timed_ok = (has_label | has_iso) & ts.notna()
    date_ok = has_date & day_only.notna()
    kind = pd.Series(
        np.select(
            [missing, has_label & timed_ok, has_iso & timed_ok, date_ok],
            ["missing", "datetime_label", "iso_offset", "date_only"],
            default="unparsed",
        ),
        index=text.index,
        dtype="str",
    )
    tz = pd.Series(
        np.select(
            [has_label & timed_ok, has_iso & timed_ok],
            [label["tz"].fillna(""), iso["tz"].fillna("")],
            default="",
        ),
        index=text.index,
        dtype="str",
    )
    ts = ts.where(timed_ok)
    day = ts.dt.normalize().where(timed_ok, day_only.where(date_ok))
    return kind, tz, ts, day


def load_pond(pond: int) -> pd.DataFrame:
    """Canonical frame for pond N: one row per data row of the file, in file order.

    Columns: pond, row (0-based data row, header excluded), ts_raw (raw stamp, NA when
    empty), ts_kind, tz_label, ts (naive clock time as written), date (calendar date),
    date_col_raw (separate Date/DATE column, ponds 7, 8, 9), entry_id (Int64), the six
    sensor variables, population (NaN when the file has no population column), length,
    weight. Trailing unnamed columns are not returned; see `read_raw`.
    """
    raw = read_raw(pond)
    by_canonical = {canon: header for header, canon in HEADER_MAP[pond].items()}
    n = len(raw)

    ts_text = _blank_to_na(raw[by_canonical["ts"]])
    kind, tz, ts, day = _parse_stamp(ts_text)
    if "date_col" in by_canonical:
        date_col_raw = _blank_to_na(raw[by_canonical["date_col"]])
    else:
        date_col_raw = pd.Series(pd.NA, index=raw.index, dtype="str")

    out = pd.DataFrame(
        {
            "pond": np.full(n, pond, dtype="int64"),
            "row": np.arange(n, dtype="int64"),
            "ts_raw": ts_text,
            "ts_kind": kind,
            "tz_label": tz,
            "ts": ts,
            "date": day,
            "date_col_raw": date_col_raw,
            "entry_id": _to_integer(raw[by_canonical["entry_id"]]),
        },
        index=raw.index,
    )
    for var in SENSOR_VARS + MEASURE_VARS:
        if var in by_canonical:
            out[var] = _to_float(raw[by_canonical[var]])
        else:
            out[var] = np.nan
    return out
