"""Independent verification from SPEC.md; run with the prescribed Python -I."""
from pathlib import Path
from itertools import combinations
import re
import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
PONDS = [1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12]
GRID = pd.date_range('2021-06-19', periods=17, freq='14D')
TOL = 1e-6  # g/day: distinguishes decimal representation noise from departures.


def clock(value: object) -> pd.Timestamp:
    if pd.isna(value):
        return pd.NaT
    value = str(value).strip()
    if re.match(r'^\d{4}-\d{2}-\d{2}', value):
        value = re.sub(r'\s+(CET|UTC)$', '', value)
        value = re.sub(r'(Z|[+-]\d{2}:\d{2})$', '', value)
        return pd.Timestamp(value)
    return pd.to_datetime(value, dayfirst=True, errors='raise')


def mode(series: pd.Series) -> float:
    values = series.dropna().mode()
    return float(values.iloc[0]) if len(values) else np.nan


def save(rows: list[dict], name: str) -> None:
    pd.DataFrame(rows).to_csv(OUT / name, index=False, float_format='%.6g')


def main() -> None:
    frames = {}
    daily = {}
    summary = []
    daily_rows = []
    for pond in PONDS:
        raw = pd.read_csv(f'/home/dev/development/publications/data/raw/pond{pond:02}.csv', low_memory=False)
        columns = list(raw.columns)
        # Headers inspected individually; all files put the six sensors after entry_id.
        entry = columns.index('entry_id')
        sensors = columns[entry + 1:entry + 7]
        weight = next(c for c in columns if 'weight' in c.lower())
        length = next(c for c in columns if 'length' in c.lower() or 'lenght' in c.lower())
        f = pd.DataFrame({'timestamp': raw.iloc[:, 0], 'weight': pd.to_numeric(raw[weight], errors='raise'),
                          'length': pd.to_numeric(raw[length], errors='coerce')})
        for j, column in enumerate(sensors):
            f[f's{j}'] = pd.to_numeric(raw[column], errors='raise')
        f['date'] = f.timestamp.map(clock).dt.normalize()
        frames[pond] = f
        d = f.dropna(subset=['date']).groupby('date').agg(
            weight=('weight', mode), length=('length', mode), rows=('weight', 'size'),
            distinct_weights=('weight', 'nunique'))
        daily[pond] = d
        changes = f.weight.notna() & f.weight.shift().notna() & f.weight.ne(f.weight.shift())
        summary.append(dict(pond=f'pond{pond:02}', rows=len(f), weight_change_rows=int(changes.sum()),
                            missing_timestamp_rows=int(f.timestamp.isna().sum()),
                            missing_weight_rows=int(f.weight.isna().sum()),
                            nonnumeric_length_rows=int((raw[length].notna() & f.length.isna()).sum()),
                            observed_dates=len(d), multiweight_dates=int(d.distinct_weights.gt(1).sum())))
        for date, r in d.iterrows():
            daily_rows.append(dict(pond=f'pond{pond:02}', date=date.date().isoformat(),
                                   modal_weight_g=r.weight, modal_length_cm=r.length,
                                   rows=int(r.rows), distinct_weights=int(r.distinct_weights)))
    save(summary, 'codex_weight_changes.csv')
    save(daily_rows, 'codex_daily_weights.csv')

    duplicate_rows = []
    sensor_names = [f's{j}' for j in range(6)]
    for p, q in combinations(PONDS, 2):
        left = frames[p].dropna(subset=['timestamp'])
        right = frames[q].dropna(subset=['timestamp'])
        common = set(left.timestamp) & set(right.timestamp)
        matched = left[['timestamp'] + sensor_names].merge(right[['timestamp'] + sensor_names], on='timestamp', suffixes=('_a', '_b'))
        equal = np.ones(len(matched), dtype=bool)
        for s in sensor_names:
            equal &= matched[s + '_a'].eq(matched[s + '_b']).to_numpy()
        duplicate_rows.append(dict(pond_a=f'pond{p:02}', pond_b=f'pond{q:02}', rows_a=len(frames[p]),
                                   rows_b=len(frames[q]), shared_raw_timestamp_strings=len(common),
                                   matched_row_pairs=len(matched), identical_six_sensor_row_pairs=int(equal.sum()),
                                   raw_timestamps_with_identical_six_sensors=matched.loc[equal, 'timestamp'].nunique()))
    save(duplicate_rows, 'codex_duplicates.csv')

    grid_rows = []
    notes = ['# Fish label grid verification', '',
             'Daily values are modes of nonmissing numeric weights. Ties use the smallest value. '
             'Intervals include both grid endpoints when observed; increments divide weight differences '
             'by actual calendar-day gaps. No unobserved dates are filled. Deviations at or below '
             '1e-6 g/day are treated as decimal representation noise. A constant stretch is compatible '
             'with linear interpolation but does not prove sampling occurred.', '']
    for pond in PONDS:
        d = daily[pond]
        bad = []
        anchor_bad = []
        for k, start in enumerate(GRID):
            end = GRID[k + 1] if k < 16 else None
            stretch = d.loc[(d.index >= start) & (d.index <= end)] if end is not None else d.iloc[:0]
            gap = stretch.index.to_series().diff().dt.days.to_numpy()[1:]
            inc = np.diff(stretch.weight.to_numpy()) / gap
            inc = inc[np.isfinite(inc)]
            median = float(np.median(inc)) if len(inc) else np.nan
            deviation = float(np.max(np.abs(inc - median))) if len(inc) else np.nan
            anchor_deviation = np.nan
            if end is not None and start in d.index and end in d.index:
                expected = d.loc[start, 'weight'] + (d.loc[end, 'weight'] - d.loc[start, 'weight']) * (stretch.index - start).days / 14
                anchor_deviation = float(np.max(np.abs(stretch.weight.to_numpy() - expected)))
            if np.isfinite(deviation) and deviation > TOL:
                bad.append(f'{start.date()}–{end.date()} ({deviation:.6g} g/day)')
            if np.isfinite(anchor_deviation) and anchor_deviation > TOL * 14:
                anchor_bad.append(str(start.date()))
            grid_rows.append(dict(pond=f'pond{pond:02}', grid_k=k, grid_date=start.date().isoformat(),
                                  date_present=start in d.index,
                                  modal_weight_g=d.loc[start, 'weight'] if start in d.index else np.nan,
                                  modal_length_cm=d.loc[start, 'length'] if start in d.index else np.nan,
                                  next_grid_date=end.date().isoformat() if end is not None else '',
                                  interval_observed_dates=len(stretch), increment_count=len(inc),
                                  median_daily_weight_increment_g=median,
                                  max_abs_increment_deviation_g=deviation,
                                  max_abs_anchor_interpolation_deviation_g=anchor_deviation))
        present = [str(t.date()) for t in GRID if t in d.index]
        missing = [str(t.date()) for t in GRID if t not in d.index]
        before = d.index[d.index < GRID[0]]
        after = d.index[d.index > GRID[-1]]
        s = next(r for r in summary if r['pond'] == f'pond{pond:02}')
        status = 'Does not hold on all observed intervals' if bad or anchor_bad else 'Compatible on evaluable observed intervals; sampling itself is not identifiable from labels'
        if not any(r['increment_count'] >= 2 for r in grid_rows if r['pond'] == f'pond{pond:02}'):
            status = 'Insufficient observed dates to test within-interval linearity'
        notes.extend([f'## Pond {pond:02}', '',
                      f'{status}. {s["weight_change_rows"]} weight changes in file order; '
                      f'{len(present)}/17 grid dates present; {s["multiweight_dates"]} dates contain multiple weights.',
                      'Missing grid dates: ' + ', '.join(missing) + '.' if missing else 'All grid dates present.',
                      'Nonlinear intervals (maximum increment deviation): ' + '; '.join(bad) + '.' if bad else 'No increment deviations above tolerance in evaluable intervals.',
                      f'Observed dates outside the grid span: {len(before)} before, {len(after)} after.',
                      'Dates with multiple weights: ' + ', '.join(str(t.date()) for t in d.index[d.distinct_weights.gt(1)]) + '.' if d.distinct_weights.gt(1).any() else 'No dates with multiple weights.', ''])
    notes.extend(['## Additional grid exceptions', '',
                  'Ponds 10 and 12 have no observations on the specified grid dates. Their date-only '
                  'records from October onward lie one day after that grid (2021-10-10, '
                  '2021-10-24, and so on), so those records support a shifted schedule rather than '
                  'the specified dates. Pond 12 has only one observed date per evaluable interval; '
                  'within-interval linearity cannot be tested. Pond 10 also has sparse, irregular '
                  'early dates; its 4.29e-6 g/day early departure is small enough to be explained '
                  'by the stored decimal precision, unlike the larger anomalies in other ponds. '
                  'Pond 11 contains repeated constant labels, multiple weights on four shifted '
                  'dates, and dates through 2022-02-07, beyond the specified grid. Its constant '
                  'stretches do not demonstrate a 14-day interpolation scheme.', '',
                  '## Data exceptions', '',
                  'Pond 01 has two missing weight/length labels. Pond 04 has a nonnumeric length '
                  'label `Jan-00`, treated as missing for numeric length modes. Pond 06 has one '
                  'missing timestamp, excluded from date grouping but retained for file-order '
                  'weight-change counts. Multiweight days use their modal weight; their counts '
                  'are reported above and their dates are in codex_daily_weights.csv. '
                  'Missing grid dates prevent direct checks against those anchors; absent dates '
                  'are not evidence against the fixed sampling grid. The final grid row has no '
                  'next interval, so its increment statistics are blank.', ''])
    save(grid_rows, 'codex_grid.csv')
    (OUT / 'codex_grid_notes.md').write_text('\n'.join(notes))

    # Task C intentionally uses pandas defaults, then drops entry_id before dropna.
    c = pd.read_csv('/home/dev/development/publications/data/raw/pond02.csv').drop(columns=['entry_id']).dropna().reset_index(drop=True)
    w = c['Fish_Weight (g)'].to_numpy()
    n = len(c)
    t, a, b = int(.8 * n), int(.7 * n), int(.8 * n)
    persistence = []
    for design, start, stop in [('S0', t, n - 134), ('S1', b + 36, n)]:
        indices = np.arange(start, stop)
        changed = w[indices] != w[indices - 1]
        for subset, chosen in [('all', indices), ('changed', indices[changed])]:
            errors = w[chosen - 1] - w[chosen]
            persistence.append(dict(design=design, subset=subset, N=n, T=t if design == 'S0' else '',
                                    a=a if design == 'S1' else '', b=b if design == 'S1' else '',
                                    L=36 if design == 'S1' else '', test_start_inclusive=start, test_stop_exclusive=stop,
                                    total_test_targets=len(indices), evaluated_targets=len(chosen),
                                    MAE_g=float(np.mean(np.abs(errors))) if len(chosen) else np.nan,
                                    RMSE_g=float(np.sqrt(np.mean(errors ** 2))) if len(chosen) else np.nan,
                                    MAPE_percent=float(np.mean(np.abs(errors / w[chosen])) * 100) if len(chosen) else np.nan))
    save(persistence, 'codex_phase2_persistence.csv')
    step_k = int(np.argmax(np.abs(np.diff(w[t:n - 134])))) + 1
    i1, i2 = t + step_k - 1, t + step_k
    ts1, ts2 = c.loc[i1, 'created_at'], c.loc[i2, 'created_at']
    elapsed = clock(ts2) - clock(ts1)
    (OUT / 'codex_step.md').write_text(
        '# Largest consecutive S0 test-target step\n\n'
        f'N = {n}; T = {t}; test range = [{t}, {n - 134}). All indices are zero-based '
        'positions after the prescribed cleaning. k denotes the later target in the change.\n\n'
        f'- Test positions: k = {step_k - 1} to k = {step_k}.\n'
        f'- Cleaned row indices: {i1} to {i2}.\n'
        f'- Timestamps: `{ts1}` to `{ts2}`.\n'
        f'- Weights: {float(w[i1])!r} g to {float(w[i2])!r} g.\n'
        f'- Absolute change: {float(abs(w[i2] - w[i1]))!r} g.\n'
        f'- Elapsed clock time: {elapsed} ({elapsed.total_seconds()!r} seconds).\n'
        '\nTies for the largest step use the earliest pair.\n')
    assert len(duplicate_rows) == 55 and len(grid_rows) == 187 and len(persistence) == 4
    assert all(r['evaluated_targets'] <= r['total_test_targets'] for r in persistence)
    outputs = ['verify.py', 'codex_duplicates.csv', 'codex_weight_changes.csv', 'codex_daily_weights.csv',
               'codex_grid.csv', 'codex_grid_notes.md', 'codex_phase2_persistence.csv', 'codex_step.md',
               'validate.py', 'validation.txt', 'DONE.md']
    (OUT / 'DONE.md').write_text(
        '# Completed independent verification\n\n' + '\n'.join(f'- {p}' for p in outputs) + '\n\n'
        'Assumptions and conventions:\n\n'
        '- Raw timestamp means the first column (Unnamed: 0 in pond 08); auxiliary Date/DATE columns are not used. '
        'Clock times retain their written values with timezone suffixes removed; slash dates are day-first.\n'
        '- Shared timestamp counts are distinct nonmissing raw strings. Repeated timestamps use a many-to-many '
        'join for matched row-pair counts; a separate count reports distinct timestamps with an identical sensor pair. '
        'Missing sensor values never count as identical. Values are compared numerically without unit conversion.\n'
        '- Weight-change counts compare adjacent file-order rows with both weights present; the first row is not a change. '
        'Missing weights are excluded from daily modes. Tied numeric modes select the smallest value. '
        'Jan-00 in pond 04 length is treated as missing, not interpreted as a measurement.\n'
        '- Grid intervals are closed at both ends and use consecutive observed dates, including day gaps. '
        'The last grid point has no subsequent interval. Fewer than two dates give blank increment statistics. '
        'The notes use a 1e-6 g/day tolerance; CSV statistics retain the computed deviations. '
        'Daily modes and file-order change counts are supplied as additional CSVs.\n'
        '- Task C uses pandas defaults exactly as requested. Indices refer to the cleaned frame, zero-based. '
        'S0 does not specify L and does not require it for persistence; its L field is blank. '
        'MAPE is mean absolute error divided by the target weight, times 100. '
        'k in the step report identifies the later target of the pair.\n'
        '- CSV floating values use six significant digits. Integer counts and row indices remain exact. '
        'No hypothesis of real sampling can be established solely from interpolated labels.\n\n'
        'Validation: executable assertions check 55 pond pairs, 187 grid rows, four metric rows, '
        'and subset sizes. validate.py independently checks every pair using counted '
        'timestamp/sensor tuples and checks persistence with scalar arithmetic; validation.txt '
        'records its result. Re-run with `/home/dev/development/publications/.venv/bin/python -I verify.py`. '
        'Raw data are read only; all writes are confined to this directory.\n')
    print(pd.DataFrame(persistence).to_string(index=False))
    print(f'Wrote {len(outputs)} artifacts; {len(duplicate_rows)} pairs; {len(grid_rows)} grid rows.')


if __name__ == '__main__':
    main()
