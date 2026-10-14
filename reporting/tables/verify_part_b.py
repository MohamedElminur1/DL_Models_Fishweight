"""Exercise the full checker and revision mismatches on disposable artifact copies.

Run from the root with env PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -I
reporting/tables/verify_part_b.py. Add --regenerate to rerun Part B generators and
require byte-identical CSVs and sidecars; otherwise manuscript outputs are read-only.
"""

import argparse
import contextlib
import csv
import io
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_tables import main as check_main

ROOT = Path(__file__).resolve().parents[2]
PROBES = {
    'main_results': ['S', 'MAE (g)', 'Ponds below the primary baseline (of 7)'],
    'main_results_full': ['S', 'MAE (g)', 'Ponds with lower S'],
    'confirmatory': ['T+', 'Exact one-sided p', 'Exact two-sided p'],
    'permutation': ['Observed S', 'Permuted S ≤ observed', 'Permutation p'],
    'inner_selection': ['Outer folds', 'Mean inner-validation S', 'Selection'],
    'chosen_hyperparameters': ['Chosen trial', 'Held-out pond', 'Chosen value'],
    'calendar_holdout': ['S', 'MAE (g)', 'Ponds below the primary baseline (of N)'],
    'all_events': ['S', 'MAE (g)', 'Ponds below the primary baseline (of 8)'],
    'sensitivity': ['S', 'MAE (g)', 'Ponds below the primary baseline (of 7)'],
    'sensitivity_merged_units': ['S model', 'T+', 'Exact one-sided p'],
    'sensitivity_influence': ['S model', 'S comparator', 'S difference'],
    'per_pond': ['S', 'MAE (g)', 'MAPE (%)'],
    'runtimes': ['Tuning time (s)', 'Final-fit time (s)', 'CPU'],
    'posthoc_ridge': ['Post hoc S', 'Frozen S', 'Post hoc permutation p'],
    'posthoc_ridge_alphas': ['Outer fold', 'Held-out pond', 'Chosen alpha'],
    'posthoc_ridge_agreement': ['Re-implementation alpha', 'Max absolute log-prediction difference', 'Max absolute weight difference (g)'],
    'sensor_qc': ['Pond', 'Sensor', 'Unit as labelled', 'Readings after duplicate removal',
                  'Missing raw readings', 'Non-finite readings', 'Out-of-range readings',
                  'Flat-line readings', 'Valid readings', 'Duplicate rows dropped',
                  'Sensor-days', 'Observed sensor-days', 'Interpolated sensor-days', 'Missing sensor-days'],
}
GENERATORS = ('main_results', 'confirmatory', 'inner_selection', 'chosen_hyperparameters',
              'calendar_holdout', 'all_events', 'sensitivity', 'per_pond', 'runtimes', 'posthoc_ridge', 'sensor_qc')


def run_check(directory: Path) -> tuple[int, str]:
    """Run the real entry point on a copied directory, capturing expected failures."""
    previous = sys.argv
    sys.argv = ['check_tables.py', '--table-dir', str(directory)]
    output = io.StringIO()
    try:
        with contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
            status = check_main()
    finally:
        sys.argv = previous
    return status, output.getvalue()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--regenerate', action='store_true')
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix='tables-part-b-') as temporary:
        directory = Path(temporary) / 'tables'
        shutil.copytree(ROOT / 'manuscript/src/tables', directory)
        status, output = run_check(directory)
        if status:
            raise ValueError(output)
        print('PASS: intact copies pass the full checker', flush=True)
        failures = 0
        for name, columns in PROBES.items():
            path = directory / f'{name}.csv'
            original = path.read_text(encoding='utf-8')
            for column in columns:
                reader = csv.DictReader(io.StringIO(original))
                header = reader.fieldnames
                rows = list(reader)
                row = next((r for r in rows if r[column] not in ('–', 'stopped (plan 9.0)')), rows[0])
                row[column] = 'CORRUPTED'
                with path.open('w', encoding='utf-8', newline='') as handle:
                    writer = csv.DictWriter(handle, fieldnames=header, lineterminator='\n')
                    writer.writeheader()
                    writer.writerows(rows)
                status, output = run_check(directory)
                path.write_text(original, encoding='utf-8')
                if status == 0 or 'FAIL:' not in output:
                    raise ValueError(f'checker accepted {name} {column}')
                failures += 1
            print(f'PASS: {name}, all {len(columns)} column corruptions rejected', flush=True)
        status, output = run_check(directory)
        if status:
            raise ValueError(output)
        print(f'PASS: {failures} mismatches rejected; restored copies pass', flush=True)
        for name, old, new in (
            ('calendar_holdout', '(of N)', '(of 7)'),
            ('all_events', '(of 8)', '(of 7)'),
            ('sensitivity', '(of 7)', '(of 8)'),
            ('sensor_qc', 'Sensor-days,', 'Days,',),
        ):
            path = directory / f'{name}.csv'
            original = path.read_text(encoding='utf-8')
            if old not in original:
                raise ValueError(f'missing header probe: {name} {old}')
            path.write_text(original.replace(old, new, 1), encoding='utf-8')
            status, output = run_check(directory)
            path.write_text(original, encoding='utf-8')
            if status == 0 or 'FAIL:' not in output:
                raise ValueError(f'checker accepted {name} incorrect header')
            failures += 1
        revisions = [
            ('permutation', 'Status', 'Descriptive permutation test'),
            ('main_results', 'R^{2}', 'CORRUPTED large-value markup'),
            ('main_results_full', 'R^{2}', 'CORRUPTED large-value markup'),
            ('calendar_holdout', 'R^{2}', 'CORRUPTED large-value markup'),
            ('per_pond', 'MAE (g)', 'CORRUPTED large-value markup'),
        ]
        for name, column, replacement in revisions:
            path = directory / f'{name}.csv'
            original = path.read_text(encoding='utf-8')
            reader = csv.DictReader(io.StringIO(original))
            header, rows = reader.fieldnames, list(reader)
            row = next(r for r in rows if (r[column] == '–' if column == 'Status' else ' × 10^{' in r[column]))
            row[column] = replacement
            with path.open('w', encoding='utf-8', newline='') as handle:
                writer = csv.DictWriter(handle, fieldnames=header, lineterminator='\n')
                writer.writeheader()
                writer.writerows(rows)
            status, output = run_check(directory)
            path.write_text(original, encoding='utf-8')
            if status == 0 or 'FAIL:' not in output:
                raise ValueError(f'checker accepted revision corruption: {name} {column}')
            failures += 1
        footers = [
            ('main_results', 'primary baseline', 'CORRUPTED baseline'),
            ('main_results', '7 ponds', '6 ponds'),
            ('main_results', 'samples.', 'CORRUPTED samples.'),
            ('main_results', 'over five seeds.', 'over four seeds.'),
            ('main_results', 'two significant digits in builder markup', 'CORRUPTED precision'),
            ('main_results_full', 'listed without predictions in results/metrics/completeness.csv', 'absent in results/metrics/completeness.csv'),
            ('permutation', 'outside the confirmatory sequence', 'inside the confirmatory sequence'),
        ]
        for name, ponds in (('calendar_holdout', '4 ponds'), ('all_events', '8 ponds'), ('sensitivity', '7 ponds')):
            footers.extend((name, old, new) for old, new in (
                ('primary baseline', 'CORRUPTED baseline'), (ponds, '99 ponds'),
                ('samples.', 'CORRUPTED samples.'), ('over five seeds.', 'over four seeds.')))
        footers.extend([
            ('calendar_holdout', 'N = 7 scored ponds', 'N = 4 scored ponds'),
            ('calendar_holdout', 'N = 4 scored ponds', 'N = 7 scored ponds'),
            ('sensor_qc', 'temperature 10 to 40', 'temperature 0 to 45'),
            ('sensor_qc', 'turbidity 0 to 100', 'turbidity 0 to 1000'),
            ('sensor_qc', 'pH 0 to 14', 'pH 1 to 14'),
            ('sensor_qc', 'dissolved oxygen > 0 and ≤ 20', 'dissolved oxygen ≥ 0 and ≤ 20'),
            ('sensor_qc', 'ammonia > 0 and ≤ 500', 'ammonia > 0 and ≤ 1000'),
            ('sensor_qc', 'nitrate > 0', 'nitrate ≥ 0'),
            ('sensor_qc', 'at least 24 h', 'at least 12 h'),
            ('sensor_qc', 'more than 1 h', 'more than 2 h'),
            ('sensor_qc', 'at least 12 distinct clock hours', 'at least 6 distinct clock hours'),
            ('sensor_qc', 'at most 2 missing days', 'at most 3 missing days'),
            ('sensor_qc', 'with observed days on both sides', 'with an observed day on one side'),
            ('sensor_qc', 'Every interpolated sensor-day is counted here', 'Only some interpolated sensor-days are counted here'),
            ('sensor_qc', 'Total interpolated sensor-days:', 'CORRUPTED total:'),
            ('sensor_qc', 'without conversion', 'after conversion'),
            ('sensor_qc', '2021-05-22 to 2022-01-29', '2021-06-19 to 2022-01-29'),
        ])
        for name in ('main_results', 'main_results_full', 'calendar_holdout', 'sensitivity', 'per_pond'):
            footers.append((name, 'example -4.3 × 10^{5}', 'example −4.3 × 10^{5}'))
        for name, old, new in footers:
            path = directory / f'{name}.source.txt'
            original = path.read_text(encoding='utf-8')
            if old not in original:
                raise ValueError(f'missing footer probe: {name} {old}')
            path.write_text(original.replace(old, new), encoding='utf-8')
            status, output = run_check(directory)
            path.write_text(original, encoding='utf-8')
            if status == 0 or 'FAIL:' not in output:
                raise ValueError(f'checker accepted revision footer corruption: {name} {old}')
            failures += 1
        stop_columns = {
            'main_results': ('S', 'MAE (g)'),
            'main_results_full': ('S', 'Ponds'),
            'sensitivity': ('S', 'RMSE (g)'),
            'permutation': ('Observed S', 'Status'),
            'inner_selection': ('Mean inner-validation S', 'Selection'),
            'chosen_hyperparameters': ('Outer fold', 'Chosen value'),
            'runtimes': ('Tuning time (s)', 'Final-fit time (s)'),
        }
        for name, (first, other) in stop_columns.items():
            path = directory / f'{name}.csv'
            original = path.read_text(encoding='utf-8')
            for mode in ('missing', 'repeated', 'misplaced'):
                reader = csv.DictReader(io.StringIO(original))
                header, rows = reader.fieldnames, list(reader)
                row = next(r for r in rows if r[first] == 'stopped (plan 9.0)')
                if mode in ('missing', 'misplaced'):
                    row[first] = '–'
                if mode in ('repeated', 'misplaced'):
                    row[other] = 'stopped (plan 9.0)'
                with path.open('w', encoding='utf-8', newline='') as handle:
                    writer = csv.DictWriter(handle, fieldnames=header, lineterminator='\n')
                    writer.writeheader()
                    writer.writerows(rows)
                status, output = run_check(directory)
                path.write_text(original, encoding='utf-8')
                if status == 0 or 'FAIL:' not in output:
                    raise ValueError(f'checker accepted {name} {mode} stop marker')
                failures += 1
            print(f'PASS: {name}, missing, repeated and misplaced stop markers rejected', flush=True)
        stop_note = next(line for line in (directory / 'main_results_full.source.txt').read_text(encoding='utf-8').splitlines()
                         if '28-day window: stopped (plan 9.0)' in line)
        for name in ('permutation', 'all_events', 'per_pond'):
            path = directory / f'{name}.source.txt'
            original = path.read_text(encoding='utf-8')
            path.write_text(original + stop_note + '\n', encoding='utf-8')
            status, output = run_check(directory)
            path.write_text(original, encoding='utf-8')
            if status == 0 or 'FAIL:' not in output:
                raise ValueError(f'checker accepted {name} unrepresented stop note')
            failures += 1
        for name in stop_columns:
            path = directory / f'{name}.source.txt'
            original = path.read_text(encoding='utf-8')
            line = next(line for line in original.splitlines() if ': stopped (plan 9.0)' in line)
            path.write_text(original.replace(line + '\n', ''), encoding='utf-8')
            status, output = run_check(directory)
            path.write_text(original, encoding='utf-8')
            if status == 0 or 'FAIL:' not in output:
                raise ValueError(f'checker accepted {name} missing stop note')
            failures += 1
        for name, token in (('main_results', '-4.3 × 10^{5}'), ('main_results', '-4.718'),
                            ('search_spaces_tabular', '10^{-3}')):
            path = directory / f'{name}.csv'
            original = path.read_text(encoding='utf-8')
            if token not in original:
                raise ValueError(f'missing minus probe: {name} {token}')
            path.write_text(original.replace(token, token.replace('-', '−'), 1), encoding='utf-8')
            status, output = run_check(directory)
            path.write_text(original, encoding='utf-8')
            if status == 0 or 'FAIL:' not in output:
                raise ValueError(f'checker accepted {name} Unicode minus')
            failures += 1
        status, output = run_check(directory)
        if status:
            raise ValueError(output)
        print(f'PASS: {failures} total mismatches rejected, including revision cells and footers; restored copies pass', flush=True)
    if args.regenerate:
        directory = ROOT / 'manuscript/src/tables'
        before = {p.name: p.read_bytes() for pattern in ('*.csv', '*.source.txt') for p in directory.glob(pattern)}
        for name in GENERATORS:
            subprocess.run(['env', 'PYTHONDONTWRITEBYTECODE=1', '.venv/bin/python', '-I',
                            f'reporting/tables/{name}.py'], cwd=ROOT, check=True)
        after = {p.name: p.read_bytes() for pattern in ('*.csv', '*.source.txt') for p in directory.glob(pattern)}
        if before != after:
            changed = sorted(name for name in before.keys() | after.keys() if before.get(name) != after.get(name))
            raise ValueError(f'regeneration changed artifacts: {changed}')
        print(f'PASS: byte-identical regeneration; all {len(before)} CSVs and sidecars unchanged', flush=True)


if __name__ == '__main__':
    main()
