"""Independently check accepted Part A and generated Part B against saved sources.

No generator or shared formatter is imported. --table-dir checks copied artifacts, so
intentional corruption can exercise the failure path without changing manuscript outputs.
"""

import argparse
from collections import Counter
import csv
import math
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
import sys

sys.dont_write_bytecode = True
import yaml


ROOT = Path(__file__).resolve().parents[2]
MAIN_HEADER = ["Design", "Inputs", "Model", "Seeds", "MAE (g)", "RMSE (g)", "MAPE (%)", "R^{2}"]
DETAIL_HEADER = ["Design", "Inputs", "Model", "Evaluation set", "Seeds", "n min", "n max", *MAIN_HEADER[4:]]
INPUT_LABELS = {"none": "none", "notebook": "notebook inputs (water quality and the lagged weight)",
                "manuscript": "manuscript inputs (water quality, population, length)", "wq_only": "water quality only"}
DESIGN_LABELS = {"S0_notebook": "Notebook design (scaler fitted on all rows; last 20 % of rows as test)",
                 "S1_chrono": "Chronological split (scaler fitted on training rows)", "S2_random": "Random split of windows"}
EVAL_LABELS = {"all": "All test windows", "change_rows": "Test rows where weight changes",
               "grid_first_rows": "First test row on each 14-day measurement-grid date"}
MODEL_LABELS = {"persistence": "Persistence", "cnn_lstm": "CNN-LSTM"}
DEEP_ROWS = ["Hidden size", "Recurrent layers", "Convolution filters", "Kernel", "Pooling", "Attention heads",
             "Dropout", "Learning rate", "Weight decay", "Batch size", "Trials per outer fold and sampler",
             "Inner folds", "Maximum epochs", "Early-stopping patience", "Final-fit epochs", "Final-fit seeds", "Optimizer", "Loss"]


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    """Read UTF-8 cells and reject missing, duplicate, multiline or ragged headings/rows."""
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        header = reader.fieldnames
        if not header or len(header) != len(set(header)):
            raise ValueError(f"{path}: missing or duplicate headings")
        rows = list(reader)
    if any(None in row or None in row.values() for row in rows):
        raise ValueError(f"{path}: ragged CSV")
    if any("\n" in cell or "\r" in cell for row in [header, *[list(row.values()) for row in rows]] for cell in row):
        raise ValueError(f"{path}: multiline cell")
    return header, rows


def escape(value: object) -> str:
    """Escape source identifiers independently of the manuscript table helper."""
    result = str(value)
    for token in "*_^[{@":
        result = result.replace(token, "\\" + token)
    return result


def match(actual: str, expected: str, label: str) -> None:
    """Report the mismatched cell and independently expected text."""
    if actual != expected:
        raise ValueError(f"{label}: got {actual!r}, expected {expected!r}")


def only(rows: list[dict[str, str]], **criteria: str) -> dict[str, str]:
    """Require exactly one row for a source or output identity."""
    selected = [row for row in rows if all(row[key] == value for key, value in criteria.items())]
    if len(selected) != 1:
        raise ValueError(f"expected one row for {criteria}, found {len(selected)}")
    return selected[0]


def provenance(directory: Path, name: str, expected: set[str], notes: tuple[str, ...] = ()) -> None:
    """Check the complete file list and any required limitations or constant citations."""
    text = (directory / f"{name}.source.txt").read_text(encoding="utf-8")
    listing, footer = text.split("\nFooter / definitions / limitations:\n", 1)
    files = listing.splitlines()[1:]
    if set(files) != expected or len(files) != len(expected) or not footer.strip():
        raise ValueError(f"{name}: missing, duplicate or unexpected provenance/footnotes")
    if any(note not in footer for note in notes):
        raise ValueError(f"{name}: missing required source note")


def check_dataset(directory: Path, cfg: dict) -> None:
    header, table = read_csv(directory / "dataset.csv")
    expected_header = ["Pond", "Sensor rows after QC", "Logging period", "Longest logging gap (days)", "Weighings",
                       "Eligible T1 events", "Eligible T2 samples (h = 1 / h = 2)", "Sensor-days observed / interpolated (%)"]
    if header != expected_header or [row["Pond"] for row in table] != list(map(str, cfg["design"]["cohort"])):
        raise ValueError("dataset: incorrect columns or cohort order")
    _, counts = read_csv(ROOT / "results/qc/qc_counts.csv")
    _, audit = read_csv(ROOT / "results/audit/summary.csv")
    _, gaps = read_csv(ROOT / "results/audit/gaps.csv")
    _, daily = read_csv(ROOT / cfg["paths"]["daily_wq"])
    _, t1 = read_csv(ROOT / cfg["paths"]["t1_table"])
    _, t2 = read_csv(ROOT / cfg["paths"]["t2_table"])
    _, events = read_csv(ROOT / "data/processed/events.csv")
    _, shared = read_csv(ROOT / "results/audit/cross_pond_duplicates.csv")
    duplicate_total = 0
    for output in table:
        pond = output["Pond"]
        saved = only(audit, pond=pond)
        first, last = saved["min_ts"].split()[0], saved["max_ts"].split()[0]
        sensor_counts = [row for row in counts if row["pond"] == pond]
        if len(sensor_counts) != 6 or {row['sensor'] for row in sensor_counts} != set(cfg['design']['sensors']):
            raise ValueError(f"dataset: pond {pond} invalid QC sensors")
        qc_rows = {row["n_values"] for row in sensor_counts}
        duplicates = {int(row["n_duplicate_rows_dropped"]) for row in sensor_counts}
        if len(qc_rows) != 1 or len(duplicates) != 1:
            raise ValueError(f"dataset: pond {pond} inconsistent QC counts")
        duplicate_total += duplicates.pop()
        duration = max((datetime.fromisoformat(row['ts_after']) - datetime.fromisoformat(row['ts_before'])).total_seconds()
                       for row in gaps if row['pond'] == pond) / 86400
        period = (date.fromisoformat(last) - date.fromisoformat(first)).days + 1
        selected_days = [row for row in daily if row['pond'] == pond and first <= row['date'] <= last]
        status = Counter(row['status'] for row in selected_days)
        if len({(row['date'], row['sensor']) for row in selected_days}) != period * 6 or len(selected_days) != period * 6:
            raise ValueError(f"dataset: pond {pond} incomplete daily grid")
        expected = [pond, qc_rows.pop(), f"{first} to {last}", f"{duration:.1f}",
                    str(sum(row['pond'] == pond for row in events)),
                    str(sum(row['pond'] == pond and row['eligible_eval'] == 'True' for row in t1)),
                    " / ".join(str(sum(row['pond'] == pond and row['horizon'] == str(h) and row['eligible_eval'] == 'True'
                                       for row in t2)) for h in (1, 2)),
                    " / ".join(f"{100 * status[state] / (period * 6):.1f}" for state in ('observed', 'interpolated'))]
        for column, value in zip(header, expected):
            match(output[column], value, f"dataset: pond {pond} {column}")
    common = int(only(shared, pond_a="1", pond_b="2")["n_common_ts_raw"])
    provenance(directory, "dataset", {"configs/phase4.yaml", "results/qc/qc_counts.csv", "results/audit/summary.csv",
               "results/audit/gaps.csv", "results/audit/cross_pond_duplicates.csv", "data/processed/events.csv",
               cfg["paths"]["t1_table"], cfg["paths"]["t2_table"], cfg["paths"]["daily_wq"]},
               (f"Exact duplicate rows dropped in QC: {duplicate_total}", f"{common:,} common timestamps", "one sensor stream"))


def diagnostic_number(value: str | None, kind: str) -> str:
    """Recompute printed precision using decimal exponents rather than the generator."""
    if value is None or value == "":
        return "NA"
    numeric = Decimal(str(value))
    if not numeric.is_finite():
        raise ValueError("nonfinite diagnostic source")
    if kind == "r2":
        places = 5 if numeric.quantize(Decimal('0.001')) == Decimal('1.000') else 3
    elif 0 < abs(numeric) < 1:
        places = 1 - abs(numeric).adjusted()
    else:
        places = 1 if kind == "percent" or abs(numeric) < 100 else 0
    return format(numeric, f".{places}f")


def check_metrics(output: dict[str, str], source: dict[str, str], name: str) -> None:
    """Compare all saved means and seed SDs, including unavailable R2 cells."""
    for key, column, kind in [('mae', 'MAE (g)', 'grams'), ('rmse', 'RMSE (g)', 'grams'),
                              ('mape', 'MAPE (%)', 'percent'), ('r2', 'R^{2}', 'r2')]:
        expected = diagnostic_number(source.get(key + '_mean'), kind)
        if int(source['n_seeds']) > 1 and expected != 'NA':
            if not source.get(key + '_sd'):
                raise ValueError(f"{name}: missing saved seed SD")
            expected += ' ± ' + diagnostic_number(source[key + '_sd'], kind)
        match(output[column], expected, f"{name}: {column}")


def check_original_design(directory: Path) -> None:
    header, detail = read_csv(directory / "original_design_detail.csv")
    _, source = read_csv(ROOT / "results/phase2/metrics_summary.csv")
    if header != DETAIL_HEADER or len(detail) != 30 or len(detail) != len(source):
        raise ValueError("original_design_detail: incorrect columns or diagnostic count")
    for i, (output, saved) in enumerate(zip(detail, source)):
        identity = [DESIGN_LABELS[saved['split']], INPUT_LABELS[saved['input_set']], MODEL_LABELS[saved['model']],
                    EVAL_LABELS[saved['eval_set']], saved['n_seeds'], saved['n_min'], saved['n_max']]
        if [output[column] for column in header[:7]] != identity:
            raise ValueError(f"original_design_detail: row {i + 1} identity or order")
        check_metrics(output, saved, f"original_design_detail: row {i + 1}")
    provenance(directory, "original_design_detail", {"results/phase2/metrics_summary.csv"})
    header, table = read_csv(directory / "original_design.csv")
    if header != MAIN_HEADER or len(table) != 15:
        raise ValueError("original_design: incorrect columns or compact comparison count")
    all_rows = [row for row in source if row['eval_set'] == 'all']
    for i, saved in enumerate(all_rows, 1):
        expected_identity = [DESIGN_LABELS[saved['split']], INPUT_LABELS[saved['input_set']], MODEL_LABELS[saved['model']], saved['n_seeds']]
        if [table[i][column] for column in header[:4]] != expected_identity:
            raise ValueError(f"original_design: row {i + 1} identity or order")
        check_metrics(table[i], saved, f"original_design: row {i + 1}")
    reported_rows = [(0, 'Reported single-step (submitted manuscript, Table 3)', 'CNN-LSTM', ('0.55', '1.95', '0.20')),
                     (12, 'Reported 18-step (submitted manuscript, Table 4)', 'TFT', ('1.78', '2.25', '3.70')),
                     (13, 'Reported 18-step (submitted manuscript, Table 4)', 'CNN-LSTM', ('12.75', '18.41', '26.60'))]
    for index, design, model, values in reported_rows:
        output = table[index]
        expected = [design, INPUT_LABELS['manuscript'], model, 'NA', *values, 'NA']
        if list(output.values()) != expected:
            raise ValueError(f"original_design: reported row {index + 1} differs from submitted constants")
    _, posthoc = read_csv(ROOT / "results/phase2/posthoc_s2_final_period_summary.csv")
    saved_posthoc = only(posthoc, input_set="notebook", model="cnn_lstm")
    if list(table[11].values())[:4] != ['Random split, original test period only (post hoc)', INPUT_LABELS['notebook'], 'CNN-LSTM', saved_posthoc['n_seeds']]:
        raise ValueError("original_design: post hoc identity")
    check_metrics(table[11], saved_posthoc, "original_design: post hoc")
    _, forecast = read_csv(ROOT / "results/phase2/forecast18_persistence.csv")
    saved_forecast = only(forecast, split="S0_notebook", horizon="18")
    expected = ['Notebook design, 18-step persistence', 'none', 'Persistence', '1',
                diagnostic_number(saved_forecast['mae'], 'grams'), diagnostic_number(saved_forecast['rmse'], 'grams'),
                diagnostic_number(saved_forecast['mape'], 'percent'), 'NA']
    if list(table[14].values()) != expected:
        raise ValueError("original_design: 18-step persistence identity or metrics")
    provenance(directory, "original_design", {"results/phase2/metrics_summary.csv",
               "results/phase2/posthoc_s2_final_period_summary.csv", "results/phase2/forecast18_persistence.csv"},
               ("submitted manuscript, Table 3", "submitted manuscript, Table 4", "S0_notebook", "does not supply R2"))


def expected_range(spec: dict) -> str:
    """Read choices or endpoints directly from the configured parameter specification."""
    def text(value):
        exponent = math.log10(abs(value)) if value else 0.0
        return f"10^{{{round(exponent)}}}" if value and abs(value) < 1e-3 and abs(exponent - round(exponent)) < 1e-12 else format(value, 'g')
    if 'choices' in spec:
        return ', '.join(text(value) for value in spec['choices'])
    distribution = 'integer' if spec['type'] == 'int' else 'log-uniform' if spec.get('log', False) else 'uniform'
    return '{} to {} ({})'.format(text(spec['low']), text(spec['high']), distribution)


def check_search_spaces(directory: Path, cfg: dict) -> None:
    header, deep = read_csv(directory / "search_spaces_deep.csv")
    if header != ['Parameter / setting', 'LSTM', 'GRU', 'CNN-LSTM', 'TFT', 'Static MLP'] or [row[header[0]] for row in deep] != DEEP_ROWS:
        raise ValueError("search_spaces_deep: incorrect columns, settings or order")
    parameters = ['hidden', 'layers', 'filters', 'kernel', 'pool', 'heads', 'dropout', 'lr', 'weight_decay', 'batch_size']
    for column, model in zip(header[1:], ['lstm', 'gru', 'cnn_lstm', 'tft', 'mlp_static']):
        for row, parameter in zip(deep, parameters):
            spec = cfg['deep']['common_space'].get(parameter, cfg['deep']['arch_space'][model].get(parameter))
            expected = expected_range(spec) if spec is not None else ''
            if model == 'tft' and parameter == 'layers':
                expected = '1 (encoder)'
            match(row[column], expected, f"search_spaces_deep: {column} {parameter}")
        settings = [f"{cfg['budget']['n_trials']} trials; Optuna TPE, seed {cfg['seeds']['sampler']}",
                    f"{cfg['design']['n_inner_folds']} (grouped by sensor stream)", str(cfg['budget']['max_epochs']),
                    f"{cfg['budget']['patience']} epochs", "Median best epoch of the chosen trial's inner folds",
                    ', '.join(str(seed) for seed in cfg['seeds']['final']), 'AdamW', 'Mean squared error on the standardized target']
        for row, expected in zip(deep[10:], settings):
            match(row[column], expected, f"search_spaces_deep: {column} {row[header[0]]}")
    plan = ' '.join((ROOT / 'ANALYSIS_PLAN.md').read_text(encoding='utf-8').split())
    for phrase in ['Optuna TPE', 'median best epoch', 'AdamW', 'mean squared error on the standardized target', '1 (encoder)']:
        if phrase not in plan:
            raise ValueError(f"search_spaces_deep: plan constant changed: {phrase}")
    provenance(directory, 'search_spaces_deep', {'configs/phase4.yaml', 'ANALYSIS_PLAN.md', 'reporting/names.py'}, ('section 7.2', 'section 6.3', 'section 13'))
    header, tabular = read_csv(directory / 'search_spaces_tabular.csv')
    expected_parameters = [escape(key) for key in cfg['gbm']['space']] + ['Trials per outer fold', escape('random_state'), 'Alpha grid', 'Tie rule']
    if header != ['Model', 'Parameter / setting', 'Range / value'] or [row[header[1]] for row in tabular] != expected_parameters or [row['Model'] for row in tabular] != ['GBM'] * 7 + ['Ridge'] * 2:
        raise ValueError('search_spaces_tabular: incorrect columns, parameters or order')
    expected_values = [expected_range(spec) for spec in cfg['gbm']['space'].values()]
    expected_values += [str(cfg['budget']['n_trials']), str(cfg['seeds']['gbm'])]
    grid = cfg['ridge']['alpha_log10']
    start, stop = Decimal(str(grid['start'])), Decimal(str(grid['stop']))
    step = (stop - start) / (grid['num'] - 1)
    powers = [format(start + step * i, 'f').rstrip('0').rstrip('.') for i in range(grid['num'])]
    powers = ['0' if power in {'', '-0'} else power for power in powers]
    expected_values += [', '.join('10^{' + power + '}' for power in powers), 'Choose the larger alpha']
    for row, expected in zip(tabular, expected_values):
        match(row['Range / value'], expected, f"search_spaces_tabular: {row[header[1]]}")
    if 'ties go to the larger alpha' not in (ROOT / 'configs/phase4.yaml').read_text(encoding='utf-8'):
        raise ValueError('search_spaces_tabular: YAML tie rule changed')
    provenance(directory, 'search_spaces_tabular', {'configs/phase4.yaml'}, ('tie rule',))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--table-dir', type=Path, default=ROOT / 'manuscript/src/tables')
    args = parser.parse_args()
    try:
        with (ROOT / 'configs/phase4.yaml').open(encoding='utf-8') as handle:
            cfg = yaml.safe_load(handle)
        check_dataset(args.table_dir, cfg)
        check_original_design(args.table_dir)
        check_search_spaces(args.table_dir, cfg)
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from check_part_b import check
        check(args.table_dir, cfg)
        if (args.table_dir / 'search_spaces.csv').exists() or (args.table_dir / 'search_spaces.source.txt').exists():
            raise ValueError('obsolete search_spaces artifacts remain')
    except (OSError, ValueError, KeyError, ZeroDivisionError, ArithmeticError) as error:
        print(f'FAIL: {error}', file=sys.stderr)
        return 1
    print('PASS: all data cells, structure, row order and provenance for all five revised Part A tables')
    return 0


if __name__ == '__main__':
    sys.exit(main())
