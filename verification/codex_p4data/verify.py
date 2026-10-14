"""Independent implementation of SPEC.md; run with the specified Python -I."""
import csv
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
RAW = Path('/home/dev/development/publications/data/raw')
PONDS = ('01', '02', '03', '04', '06', '07', '08', '09')
SENSORS = ('temperature', 'turbidity', 'do', 'ph', 'ammonia_log10', 'nitrate')
START = pd.Timestamp('2021-06-19')
GRID = pd.date_range(START, periods=17, freq='14D')
CALENDAR = pd.date_range('2021-05-22', '2022-01-29')
STATS = ('mean', 'sd', 'min', 'max')


def parse_timestamp(raw: str) -> tuple[pd.Timestamp, bool]:
    value = raw.strip()
    if not value:
        return pd.NaT, False
    if re.search(r'\d{2}:\d{2}', value):
        # Retain the written clock, discarding both named and numeric zones.
        match = re.fullmatch(r'(\d{4}-\d{2}-\d{2})[ T](\d{2}:\d{2}:\d{2})(?:\s*[A-Za-z]+|[+-]\d{2}:\d{2}|Z)?', value)
        if not match:
            raise ValueError(f'Unrecognized timestamp: {raw!r}')
        return pd.Timestamp(f'{match[1]} {match[2]}'), True
    return pd.Timestamp(pd.to_datetime(value, format='%d/%m/%Y')), False


def load_pond(pond: str) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    with (RAW / f'pond{pond}.csv').open(newline='') as stream:
        reader = csv.reader(stream)
        header = next(reader)
        rows = list(reader)
    if any(len(row) != len(header) for row in rows):
        raise ValueError(f'Unexpected row width in pond {pond}')
    normalized = [re.sub(r'[^a-z]', '', name.lower()) for name in header]
    prefixes = {'temperature': 'temperature', 'turbidity': 'turbidity', 'do': 'dissolvedoxygen', 'ph': 'ph', 'ammonia_log10': 'ammonia', 'nitrate': 'nitrate', 'length': 'fishlength', 'weight': 'fishweight'}
    indices = {}
    for name, prefix in prefixes.items():
        matches = [i for i, column in enumerate(normalized) if column.startswith(prefix)]
        if not matches and name in ('length', 'weight'):
            prefix = 'totallength' if name == 'length' else 'weight'
            matches = [i for i, column in enumerate(normalized) if column.startswith(prefix)]
        if len(matches) != 1:
            raise ValueError(f'Ambiguous column {name} in pond {pond}: {header}')
        indices[name] = matches[0]
    parsed = [parse_timestamp(row[0]) for row in rows]
    frame = pd.DataFrame({'time': [item[0] for item in parsed], 'clock': [item[1] for item in parsed]})
    frame['date'] = frame.time.dt.normalize()
    for name, index in indices.items():
        frame[name] = pd.to_numeric(pd.Series([row[index] for row in rows]), errors='coerce')
    keep = []
    seen = set()
    duplicates = 0
    for index, row in enumerate(rows):
        if not parsed[index][1]:
            continue
        key = tuple(row)
        if key in seen:
            duplicates += 1
        else:
            seen.add(key)
            keep.append(index)
    sensor_frame = frame.loc[keep].sort_values('time', kind='stable').reset_index(drop=True)
    audit = {'pond': pond, 'n_raw_rows': len(rows), 'n_timestamped_before': int(frame.clock.sum()), 'n_duplicate_drops': duplicates, 'n_timestamped_after': len(sensor_frame), 'n_date_only': int((~frame.clock & frame.time.notna()).sum()), 'n_undated': int(frame.time.isna().sum())}
    return frame, sensor_frame, audit


def fish_events(frame: pd.DataFrame, pond: str) -> pd.DataFrame:
    records = []
    for date in GRID:
        day = frame.loc[frame.date.eq(date)]
        if day.weight.notna().any():
            records.append({'pond': pond, 'date': date, 't': (date - START).days, 'weight': day.weight.mode().iloc[0], 'length': day.length.mode().iloc[0] if day.length.notna().any() else np.nan, 'n_rows': len(day)})
    return pd.DataFrame(records, columns=['pond', 'date', 't', 'weight', 'length', 'n_rows'])


def sensor_qc(times: pd.DatetimeIndex, values: np.ndarray, sensor: str) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    finite = np.isfinite(values)
    if sensor == 'temperature':
        in_range = (values >= 10) & (values <= 40)
    elif sensor == 'turbidity':
        in_range = (values >= 0) & (values <= 100)
    elif sensor == 'do':
        in_range = (values > 0) & (values <= 20)
    elif sensor == 'ph':
        in_range = (values >= 0) & (values <= 14)
    elif sensor == 'ammonia_log10':
        in_range = (values > 0) & (values <= 500)
    elif sensor == 'nitrate':
        in_range = values > 0
    else:
        raise ValueError(sensor)
    flat = np.zeros(len(values), dtype=bool)
    if len(values):
        clock_ns = times.as_unit('ns').asi8
        gaps = np.diff(clock_ns)
        breaks = (~finite[1:]) | (~finite[:-1]) | (values[1:] != values[:-1]) | (gaps > pd.Timedelta(hours=1).value)
        starts = np.r_[0, np.flatnonzero(breaks) + 1]
        ends = np.r_[starts[1:] - 1, len(values) - 1]
        long = finite[starts] & ((clock_ns[ends] - clock_ns[starts]) >= pd.Timedelta(hours=24).value)
        for start, end in zip(starts[long], ends[long]):
            flat[start:end + 1] = True
    bad_range = finite & ~in_range
    bad_flat = finite & in_range & flat
    valid = finite & in_range & ~flat
    counts = (int((~finite).sum()), int(bad_range.sum()), int(bad_flat.sum()), int(valid.sum()))
    assert sum(counts) == len(values)
    return valid, counts


def daily_table(times: pd.DatetimeIndex, values: np.ndarray) -> pd.DataFrame:
    valid = np.isfinite(values)
    frame = pd.DataFrame({'date': times[valid].normalize(), 'hour': times[valid].hour, 'value': values[valid]})
    grouped = frame.groupby('date')
    stats = grouped.value.agg(['mean', 'min', 'max'])
    stats['sd'] = grouped.value.std(ddof=0)
    stats['n_hours'] = grouped.hour.nunique()
    result = stats.reindex(CALENDAR)
    result['n_hours'] = result.n_hours.fillna(0).astype(int)
    observed = result.n_hours >= 12
    result.loc[~observed, list(STATS)] = np.nan
    result['status'] = np.where(observed, 'observed', 'missing')
    result['left_anchor'] = pd.NaT
    result['right_anchor'] = pd.NaT
    observed_positions = np.flatnonzero(observed.to_numpy())
    for left, right in zip(observed_positions[:-1], observed_positions[1:]):
        if 1 <= right - left - 1 <= 2:
            for position in range(left + 1, right):
                fraction = (position - left) / (right - left)
                result.loc[CALENDAR[position], list(STATS)] = result.loc[CALENDAR[left], list(STATS)] * (1 - fraction) + result.loc[CALENDAR[right], list(STATS)] * fraction
                result.loc[CALENDAR[position], ['status', 'left_anchor', 'right_anchor']] = ['interpolated', CALENDAR[left], CALENDAR[right]]
    result.index.name = 'date'
    return result


def window_features(tables: dict[str, pd.DataFrame], reference: pd.Timestamp) -> dict:
    features = {}
    availability = {}
    for sensor, table in tables.items():
        availability[sensor] = table.status.eq('observed') | (table.status.eq('interpolated') & table.right_anchor.lt(reference))
    available = pd.DataFrame(availability)
    for prefix, length in [('t1', 14), ('t2', 28)]:
        dates = pd.date_range(reference - pd.Timedelta(days=length), reference - pd.Timedelta(days=1))
        for sensor in SENSORS:
            selected = tables[sensor].loc[dates].loc[available.loc[dates, sensor]]
            for stat in STATS:
                value = np.nan
                if len(selected) >= 3:
                    value = selected[stat].min() if stat == 'min' else selected[stat].max() if stat == 'max' else selected[stat].mean()
                features[f'{prefix}_{sensor}_{stat}_w'] = value
        if prefix == 't1':
            cumulative_dates = pd.date_range(START, reference - pd.Timedelta(days=1))
            for sensor in SENSORS:
                selected = tables[sensor].loc[cumulative_dates].loc[available.loc[cumulative_dates, sensor]]
                features[f't1_{sensor}_mean_cum'] = selected['mean'].mean() if len(selected) >= 3 else np.nan
        features[f'{prefix}_n_covered_window'] = int((available.loc[dates].sum(axis=1) >= 3).sum())
    return features


def eligibility(events: pd.DataFrame, features: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    t1 = events.loc[events.t > 0, ['pond', 'date', 't']].merge(features[['pond', 'date', 't1_n_covered_window']], on=['pond', 'date'], validate='one_to_one')
    t1['eligible'] = t1.t1_n_covered_window >= 7
    lookup = features.set_index(['pond', 'date'])
    records = []
    for pond, pond_events in events.groupby('pond', sort=False):
        ordered = pond_events.sort_values('date').reset_index(drop=True)
        event_dates = set(ordered.date)
        for position in range(1, len(ordered)):
            origin = ordered.iloc[position]
            coverage = int(lookup.loc[(pond, origin.date), 't2_n_covered_window'])
            for horizon in (1, 2):
                target = origin.date + pd.Timedelta(days=14 * horizon)
                if target in event_dates:
                    records.append({'pond': pond, 'origin_date': origin.date, 'target_date': target, 'horizon': horizon, 'dt_prev': int(origin.t - ordered.iloc[position - 1].t), 't2_n_covered_window': coverage, 'eligible': coverage >= 14})
    return t1, pd.DataFrame(records, columns=['pond', 'origin_date', 'target_date', 'horizon', 'dt_prev', 't2_n_covered_window', 'eligible'])


def write_csv(frame: pd.DataFrame, name: str) -> None:
    frame.to_csv(ROOT / name, index=False, float_format='%.17g', date_format='%Y-%m-%d')


def main() -> None:
    all_events, all_daily, all_features, qc_records, audits = [], [], [], [], []
    for pond in PONDS:
        raw, sensors, audit = load_pond(pond)
        audits.append(audit)
        events = fish_events(raw, pond)
        all_events.append(events)
        times = pd.DatetimeIndex(sensors.time)
        tables = {}
        for sensor in SENSORS:
            values = sensors[sensor].to_numpy(dtype=float)
            valid, counts = sensor_qc(times, values, sensor)
            qc_records.append(dict(zip(['pond', 'sensor', 'n_values', 'n_invalid_missing_or_nonfinite', 'n_out_of_range', 'n_flatline', 'n_valid'], [pond, sensor, len(values), *counts])))
            cleaned = np.full(len(values), np.nan)
            cleaned[valid] = np.log10(values[valid]) if sensor == 'ammonia_log10' else values[valid]
            table = daily_table(times, cleaned)
            tables[sensor] = table
            daily = table.reset_index()
            daily.insert(0, 'pond', pond)
            daily.insert(2, 'sensor', sensor)
            all_daily.append(daily[['pond', 'date', 'sensor', *STATS, 'n_hours', 'status', 'left_anchor', 'right_anchor']])
        for date in GRID[1:]:
            all_features.append({'pond': pond, 'date': date, **window_features(tables, date)})
        print(f'pond {pond}: {len(events)} events, {audit["n_duplicate_drops"]} exact duplicate timestamped rows dropped', flush=True)
    events = pd.concat(all_events, ignore_index=True)
    daily = pd.concat(all_daily, ignore_index=True).sort_values(['pond', 'date', 'sensor'], kind='stable')
    features = pd.DataFrame(all_features)
    t1, t2 = eligibility(events, features)
    outputs = {'codex_events.csv': events, 'codex_qc_counts.csv': pd.DataFrame(qc_records), 'codex_duplicate_counts.csv': pd.DataFrame(audits), 'codex_daily.csv': daily, 'codex_wq_by_date.csv': features, 'codex_t1.csv': t1, 'codex_t2.csv': t2}
    assert len(daily) == len(PONDS) * len(SENSORS) * len(CALENDAR)
    assert len(features) == len(PONDS) * 16
    assert not daily.duplicated(['pond', 'date', 'sensor']).any()
    assert not events.duplicated(['pond', 'date']).any()
    for name, frame in outputs.items():
        write_csv(frame, name)
    lines = ['# Completed independent verification', '', 'Implemented from SPEC.md and the eight named raw CSVs only. No other project source or results were read.', '', '## Output files', '']
    lines += [f'- `{name}`: {len(frame)} rows.' for name, frame in outputs.items()]
    lines += ['- `verify.py`: rerunnable implementation.', '- `test_verify.py`: focused synthetic checks.', '- `DONE.md`: this report.', '', '## Counts', '', '| Pond | Events | T1 eligible | T2 horizon 1 eligible | T2 horizon 2 eligible |', '|---|---:|---:|---:|---:|']
    for pond in PONDS:
        lines.append(f'| {pond} | {int(events.pond.eq(pond).sum())} | {int((t1.pond.eq(pond) & t1.eligible).sum())} | {int((t2.pond.eq(pond) & t2.horizon.eq(1) & t2.eligible).sum())} | {int((t2.pond.eq(pond) & t2.horizon.eq(2) & t2.eligible).sum())} |')
    lines += ['', '## Interpretations and assumptions', '', '- Pond identifiers are zero-padded NN strings. Numeric outputs use 17 significant digits; missing values are empty CSV cells.', '- Column labels are mapped by their meaning; values use their written numeric scales, with no unit conversion.', '- Event modes use all rows dated that day, before sensor deduplication. The smallest numeric mode wins a tie. n_rows counts all rows dated that day, including rows lacking weight.', '- Non-numeric fish values are missing. The specification gives no fish range or finiteness filter, so numeric infinities would remain fish values.', '- The single pond 06 row with an empty timestamp is excluded from dated events and sensor processing; no date is inferred.', '- Exact duplicate detection compares every parsed CSV cell as raw text, including auxiliary and unnamed columns. CSV quoting itself is not a column value.', '- Flat-line equality compares numeric values before QC and before ammonia log10; textual representations of the same number compare equal. Range failures still participate in raw runs, but are counted under range first.', '- Missing and interpolated days retain the actual number of distinct hours with valid measurements in n_hours. Anchors are populated only for interpolated days.', '- Daily interpolation uses only original observed anchors. Reference-date features reject interpolation with right_anchor equal to or later than the reference date.', '- T2 emits only existing exact +14/+28-day targets and requires an earlier origin event. Eligibility depends on origin coverage only.', '', '## Verification', '', 'Synthetic checks cover zone stripping and day-first dates, flat-line duration and gaps, QC precedence, daily coverage and short-gap interpolation, reference-date anchor exclusion, the three-day feature minimum, event modes, and T2 previous-event gaps. Run with:', '', '```sh', '/home/dev/development/publications/.venv/bin/python -I test_verify.py', '/home/dev/development/publications/.venv/bin/python -I verify.py', '```', '', 'The script checks output dimensions, unique event and daily keys, and QC count reconciliation. Re-running overwrites only its named outputs and report; the raw data remain read-only. If interrupted, rerun to regenerate the complete set.']
    (ROOT / 'DONE.md').write_text('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
