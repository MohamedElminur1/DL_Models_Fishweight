"""Cross-check outputs with counted tuples and scalar persistence arithmetic."""
from collections import Counter
from itertools import combinations
from pathlib import Path
import math
import pandas as pd

ROOT = Path(__file__).resolve().parent
PONDS = [1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12]


def main() -> None:
    counts, stamps, sizes = {}, {}, {}
    for pond in PONDS:
        d = pd.read_csv(f'/home/dev/development/publications/data/raw/pond{pond:02}.csv', low_memory=False)
        start = d.columns.get_loc('entry_id') + 1
        selected = d.iloc[:, [0] + list(range(start, start + 6))].dropna()
        counts[pond] = Counter(selected.itertuples(index=False, name=None))
        stamps[pond] = set(d.iloc[:, 0].dropna())
        sizes[pond] = len(d)
    results = pd.read_csv(ROOT / 'codex_duplicates.csv')
    for (p, q), r in zip(combinations(PONDS, 2), results.itertuples(index=False)):
        assert (r.pond_a, r.pond_b) == (f'pond{p:02}', f'pond{q:02}')
        common_keys = counts[p].keys() & counts[q].keys()
        assert r.rows_a == sizes[p] and r.rows_b == sizes[q]
        assert r.shared_raw_timestamp_strings == len(stamps[p] & stamps[q])
        assert r.identical_six_sensor_row_pairs == sum(counts[p][key] * counts[q][key] for key in common_keys)
        assert r.raw_timestamps_with_identical_six_sensors == len({key[0] for key in common_keys})
    c = pd.read_csv('/home/dev/development/publications/data/raw/pond02.csv')
    c = c.drop(columns=['entry_id']).dropna()
    weights = c['Fish_Weight (g)'].tolist()
    n = len(weights)
    metrics = pd.read_csv(ROOT / 'codex_phase2_persistence.csv')
    for r in metrics.itertuples(index=False):
        start = int(.8 * n) + (36 if r.design == 'S1' else 0)
        stop = n if r.design == 'S1' else n - 134
        pairs = [(weights[i - 1], weights[i]) for i in range(start, stop)]
        if r.subset == 'changed':
            pairs = [(old, new) for old, new in pairs if old != new]
        errors = [abs(old - new) for old, new in pairs]
        expected = [math.fsum(errors) / len(pairs),
                    math.sqrt(math.fsum(e * e for e in errors) / len(pairs)),
                    100 * math.fsum(abs(old - new) / new for old, new in pairs) / len(pairs)]
        assert r.N == n and r.evaluated_targets == len(pairs)
        assert r.test_start_inclusive == start and r.test_stop_exclusive == stop
        for value, written in zip(expected, [r.MAE_g, r.RMSE_g, r.MAPE_percent]):
            assert float(format(value, '.6g')) == written
    grid = pd.read_csv(ROOT / 'codex_grid.csv')
    daily = pd.read_csv(ROOT / 'codex_daily_weights.csv')
    assert len(grid) == 187
    for r in grid.itertuples(index=False):
        on_date = daily[daily.pond.eq(r.pond) & daily.date.eq(r.grid_date)]
        assert bool(len(on_date)) == r.date_present
        if len(on_date):
            assert r.modal_weight_g == on_date.modal_weight_g.iloc[0]
            assert (pd.isna(r.modal_length_cm) and pd.isna(on_date.modal_length_cm.iloc[0])) or r.modal_length_cm == on_date.modal_length_cm.iloc[0]
    report = ('PASS: all 55 pond-pair shared timestamps and identical sensor counts independently checked.\n'
              'PASS: all four persistence rows checked using scalar arithmetic, including target counts and boundaries.\n'
              'PASS: all 187 grid-date presence flags and modes checked against the daily output.\n')
    (ROOT / 'validation.txt').write_text(report)
    print(report, end='')


if __name__ == '__main__':
    main()
