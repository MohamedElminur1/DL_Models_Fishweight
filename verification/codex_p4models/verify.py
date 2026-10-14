"""Independent implementation of SPEC.md; run with the specified Python -I."""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from sklearn.linear_model import LinearRegression, Ridge

ROOT = Path(__file__).resolve().parent
INPUT_ROOT = Path('/home/dev/development/publications/data/processed')
COHORT = [1, 2, 3, 4, 6, 7, 8, 9]
SENSORS = ['temperature', 'turbidity', 'do', 'ph', 'ammonia_log10', 'nitrate']
WINDOW = [f'{s}_{stat}_w' for s in SENSORS for stat in ['mean', 'sd', 'min', 'max']]
CUMULATIVE = [f'{s}_mean_cum' for s in SENSORS]
ALPHAS = 10. ** np.arange(-3, 3.01, .5)
MODELS = {'T1': ['gc', 'cohort_mean', 'ridge_time', 'ridge_time_wq', 'allometric'],
          'T2': ['persistence', 'linear', 'sgr', 'anchored_gc', 'cohort_growth',
                 'ridge_pw', 'ridge_pw_wq']}
CURVE_CACHE = {}


def outer_folds(rows: pd.DataFrame) -> list[dict]:
    eligible = rows.loc[rows.eligible_eval.eq(True) & rows.pond.isin(COHORT)].copy()
    folds = []
    for pond in sorted(eligible.pond.unique()):
        dropped = [3 - int(pond)] if pond in (1, 2) else []
        train = eligible.loc[~eligible.pond.isin([pond] + dropped)].copy()
        test = eligible.loc[eligible.pond.eq(pond)].copy()
        folds.append({'held_out': int(pond), 'dropped': dropped, 'train': train, 'test': test})
    return folds


def inner_ponds(train: pd.DataFrame) -> list[list[int]]:
    ponds = sorted(int(p) for p in train.pond.unique())
    units = []
    for p in ponds:
        if p == 2 and 1 in ponds:
            continue
        units.append([1, 2] if p == 1 and 2 in ponds else [p])
    return [[p for i, unit in enumerate(units) if i % 3 == j for p in unit]
            for j in range(3)]


def statistic_s(ponds, errors) -> float:
    # Aggregate explicitly: pandas groupby means can mishandle infinities.
    ponds, errors = np.asarray(ponds), np.asarray(errors, dtype=float)
    return float(np.mean([np.mean(errors[ponds == p]) for p in np.unique(ponds)]))


def score(task: str, rows: pd.DataFrame, predictions: np.ndarray) -> tuple:
    predictions = np.asarray(predictions, dtype=float)
    target = rows.log_w.to_numpy() if task == 'T1' else rows.y.to_numpy()
    errors = np.abs(predictions - target)
    errors[np.isnan(predictions)] = np.inf
    if task == 'T1':
        return statistic_s(rows.pond, errors), np.nan, np.nan
    per_horizon = [statistic_s(rows.loc[rows.horizon.eq(h), 'pond'],
                              errors[rows.horizon.to_numpy() == h]) for h in [1, 2]]
    return float(np.mean(per_horizon)), *per_horizon


def gompertz(t, log_a, b, c):
    return log_a - b * np.exp(-c * t)


def von_bertalanffy(t, log_winf, k, t0):
    return log_winf + 3 * np.log(1 - np.exp(-k * (t - t0)))


def logistic(t, log_a, k, tm):
    return log_a - np.logaddexp(0, -k * (t - tm))


CURVES = [
    ('gompertz', gompertz, (2.3, .01, 1e-4), (11.5, 30, 1),
     [(7, 6, .02), (6, 5, .04), (9, 8, .01)]),
    ('von_bertalanffy', von_bertalanffy, (2.3, 1e-4, -300), (11.5, 1, -.001),
     [(7, .02, -10), (6.5, .01, -30), (8, .005, -50)]),
    ('logistic', logistic, (2.3, 1e-4, -100), (11.5, 1, 600),
     [(7, .03, 100), (6, .05, 60), (8, .02, 150)]),
]


def fit_growth(t: np.ndarray, log_w: np.ndarray) -> dict:
    t, log_w = np.asarray(t, dtype=float), np.asarray(log_w, dtype=float)
    key = (t.tobytes(), log_w.tobytes())
    if key in CURVE_CACHE:
        return CURVE_CACHE[key]
    best, failed, failed_starts = None, [], []
    for name, function, lower, upper, starts in CURVES:
        curve_best = None
        for i, start in enumerate(starts):
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    parameters, _ = curve_fit(function, t, log_w, p0=start,
                                              bounds=(lower, upper), max_nfev=20000)
                rss = float(np.sum((function(t, *parameters) - log_w) ** 2))
                if not np.isfinite(rss):
                    raise ValueError('nonfinite RSS')
            except Exception as exc:
                failed_starts.append(f'{name}:start{i}:{type(exc).__name__}')
                continue
            if curve_best is None or rss < curve_best['RSS']:
                with np.errstate(divide='ignore'):
                    aic = float(len(t) * np.log(rss / len(t)) + 6)
                curve_best = {'curve': name, 'function': function, 'parameters': parameters,
                              'RSS': rss, 'AIC': aic}
        if curve_best is None:
            failed.append(name)
        elif best is None or curve_best['AIC'] < best['AIC']:
            best = curve_best
    result = {'best': best, 'failed': failed, 'failed_starts': failed_starts}
    CURVE_CACHE[key] = result
    return result


def anchored_points(train: pd.DataFrame) -> pd.DataFrame:
    origins = pd.DataFrame({'pond': train.pond, 'date': train.origin_date,
                            't': train.t_k, 'log_w': train.log_w_k})
    targets = pd.DataFrame({'pond': train.pond, 'date': train.target_date,
                            't': train.t_k + train.dt_target, 'log_w': np.log(train.w_target)})
    points = pd.concat([origins, targets], ignore_index=True)
    for _, event in points.groupby(['pond', 'date'], sort=False):
        if event.t.nunique() != 1 or not np.allclose(event.log_w, event.log_w.iloc[0],
                                                   rtol=0, atol=1e-12):
            raise ValueError('Conflicting observations for one anchored event')
    return points.drop_duplicates(['pond', 'date'], keep='first').reset_index(drop=True)


class RidgePreprocessor:
    """Fit medians and population scaling only on the supplied training rows."""

    def __init__(self, features: list[str], water_quality: bool):
        self.features = features
        self.water_quality = water_quality

    def fit(self, train: pd.DataFrame):
        self.columns = [c for c in self.features if train[c].notna().any()]
        self.medians = train[self.columns].median()
        matrix = self._matrix(train)
        self.mean = matrix.mean(axis=0)
        self.sd = matrix.std(axis=0, ddof=0)
        self.sd[self.sd == 0] = 1
        return self

    def _matrix(self, rows: pd.DataFrame) -> np.ndarray:
        matrix = rows[self.columns].fillna(self.medians).to_numpy(dtype=float)
        if self.water_quality:
            indicators = rows[[f'{s}_mean_w' for s in SENSORS]].isna().to_numpy(dtype=float)
            matrix = np.column_stack([matrix, indicators])
        return matrix

    def transform(self, rows: pd.DataFrame) -> np.ndarray:
        return (self._matrix(rows) - self.mean) / self.sd


def ridge_features(task: str, model: str) -> list[str]:
    base = ['t', 't2'] if task == 'T1' else ['log_w_k', 'log_w_prev', 't_k', 'dt_prev', 'dt_target']
    return base + (WINDOW + (CUMULATIVE if task == 'T1' else []) if model.endswith('_wq') else [])


def predict_model(task: str, model: str, train: pd.DataFrame, test: pd.DataFrame,
                  alpha: float | None = None) -> tuple:
    info = {'fallbacks': {}, 'growth': None}

    def record(name, count):
        if count:
            info['fallbacks'][name] = info['fallbacks'].get(name, 0) + int(count)

    def growth_prediction(rows):
        if info['growth'] is None:
            points = train if task == 'T1' else anchored_points(train)
            info['growth'] = fit_growth(points.t.to_numpy(), points.log_w.to_numpy())
        best = info['growth']['best']
        if best is not None:
            f, p = best['function'], best['parameters']
            if task == 'T1':
                return f(rows.t.to_numpy(), *p)
            return f((rows.t_k + rows.dt_target).to_numpy(), *p) - f(rows.t_k.to_numpy(), *p)
        if task == 'T2':
            record('anchored_gc -> sgr', len(rows))
            return ((rows.log_w_k - rows.log_w_prev) / rows.dt_prev * rows.dt_target).to_numpy()
        record('gc -> cohort_mean_without_fallback', len(rows))
        predictions = []
        for day in rows.t:
            matched = train.loc[train.t.eq(day)]
            if matched.pond.nunique() < 3:
                raise ValueError('All curves failed and cohort_mean has fewer than three ponds')
            predictions.append(matched.log_w.mean())
        return np.asarray(predictions)

    if model.startswith('ridge_'):
        if alpha is None:
            raise ValueError('Ridge requires a chosen alpha')
        prep = RidgePreprocessor(ridge_features(task, model), model.endswith('_wq')).fit(train)
        target = train.log_w if task == 'T1' else train.y
        fitted = Ridge(alpha=alpha).fit(prep.transform(train), target)
        predictions = fitted.predict(prep.transform(test))
    elif model == 'allometric':
        fitted = LinearRegression().fit(train[['log_l']], train.log_w)
        predictions = fitted.predict(test[['log_l']])
    elif model in ('gc', 'anchored_gc'):
        predictions = growth_prediction(test)
    elif model in ('cohort_mean', 'cohort_growth'):
        keys = ['t'] if task == 'T1' else ['origin_date', 'target_date']
        target = 'log_w' if task == 'T1' else 'y'
        lookup = {}
        for key, group in train.groupby(keys, sort=False):
            if group.pond.nunique() >= 3:
                lookup[key if isinstance(key, tuple) else (key,)] = float(group[target].mean())
        predictions = np.array([lookup.get(tuple(row), np.nan)
                                for row in test[keys].itertuples(index=False, name=None)])
        missing = np.isnan(predictions)
        if missing.any():
            record(f'{model} -> {"gc" if task == "T1" else "anchored_gc"}', missing.sum())
            predictions[missing] = growth_prediction(test.loc[missing])
    elif model == 'persistence':
        predictions = np.zeros(len(test))
    elif model == 'sgr':
        predictions = ((test.log_w_k - test.log_w_prev) / test.dt_prev * test.dt_target).to_numpy()
    elif model == 'linear':
        weights = (test.w_origin + (test.w_origin - np.exp(test.log_w_prev)) /
                   test.dt_prev * test.dt_target).to_numpy()
        predictions = np.full(len(test), np.nan)
        positive = weights > 0
        predictions[positive] = np.log(weights[positive] / test.w_origin.to_numpy()[positive])
        return predictions, weights, info
    else:
        raise ValueError(model)
    weights = np.exp(predictions)
    if task == 'T2':
        weights *= test.w_origin.to_numpy()
    predictions = np.array(predictions, dtype=float, copy=True)
    predictions[weights <= 0] = np.nan
    return predictions, weights, info


def choose_alpha(task: str, model: str, train: pd.DataFrame) -> tuple:
    groups = inner_ponds(train)
    best_alpha, best_mean, best_scores = None, np.inf, None
    for alpha in ALPHAS:
        scores = []
        for ponds in groups:
            validation = train.loc[train.pond.isin(ponds)]
            fitting = train.loc[~train.pond.isin(ponds)]
            prediction, _, _ = predict_model(task, model, fitting, validation, float(alpha))
            scores.append(score(task, validation, prediction))
        mean = float(np.mean([s[0] for s in scores]))
        if best_alpha is None or mean <= best_mean:
            best_alpha, best_mean, best_scores = float(alpha), mean, scores
    return best_alpha, best_scores


def write_csv(name: str, rows: list[dict], columns: list[str]) -> pd.DataFrame:
    result = pd.DataFrame(rows, columns=columns)
    result.to_csv(ROOT / name, index=False, float_format='%.17g')
    return result


def main() -> None:
    folds_output, inner_output, predictions_output, curves_output = [], [], [], []
    fallback_output, summary = [], {}
    linear_inner_count = 0
    for task, filename in [('T1', 't1_table.csv'), ('T2', 't2_table.csv')]:
        data = pd.read_csv(INPUT_ROOT / filename)
        if data.eligible_eval.dtype != bool:
            raise ValueError('eligible_eval must parse as boolean')
        folds = outer_folds(data)
        summary[task] = (len(folds), sum(len(f['test']) for f in folds))
        for fold in folds:
            held, train, test = fold['held_out'], fold['train'], fold['test']
            groups = inner_ponds(train)
            folds_output.append({'task': task, 'held_out_pond': held,
                                 'train_ponds': ' '.join(map(str, sorted(train.pond.unique()))),
                                 'dropped_ponds': ' '.join(map(str, fold['dropped'])),
                                 'n_train': len(train), 'n_test': len(test),
                                 **{f'inner{j}_val_ponds': ' '.join(map(str, g))
                                    for j, g in enumerate(groups)}})
            for model in MODELS[task]:
                alpha, selected_scores = choose_alpha(task, model, train) if model.startswith('ridge_') else (None, None)
                for j, ponds in enumerate(groups):
                    validation = train.loc[train.pond.isin(ponds)]
                    if selected_scores is None:
                        fitting = train.loc[~train.pond.isin(ponds)]
                        prediction, inner_weights, info = predict_model(task, model, fitting, validation)
                        if model == 'linear':
                            linear_inner_count += int((inner_weights <= 0).sum())
                        values = score(task, validation, prediction)
                        for fallback, count in info['fallbacks'].items():
                            fallback_output.append((task, model, held, f'inner{j}', fallback, count))
                    else:
                        values = selected_scores[j]
                    inner_output.append({'task': task, 'model': model, 'held_out_pond': held,
                                         'inner_fold': j, 'n_val': len(validation), 'S': values[0],
                                         'S_h1': values[1], 'S_h2': values[2], 'alpha': alpha})
                prediction, weights, info = predict_model(task, model, train, test, alpha)
                for fallback, count in info['fallbacks'].items():
                    fallback_output.append((task, model, held, 'outer', fallback, count))
                if model in ('gc', 'anchored_gc'):
                    growth, best = info['growth'], info['growth']['best']
                    curves_output.append({'task': task, 'model': model, 'held_out_pond': held,
                                          'chosen_curve': best['curve'] if best else '',
                                          **{f'parameter{i+1}': best['parameters'][i] if best else np.nan
                                             for i in range(3)},
                                          'RSS': best['RSS'] if best else np.nan,
                                          'AIC': best['AIC'] if best else np.nan,
                                          'failed_curves': ' '.join(growth['failed'])})
                for i, (_, row) in enumerate(test.iterrows()):
                    predictions_output.append({'task': task, 'model': model, 'held_out_pond': held,
                                               'pond': int(row.pond),
                                               'date': row.date if task == 'T1' else row.target_date,
                                               'origin_date': '' if task == 'T1' else row.origin_date,
                                               'horizon': 0 if task == 'T1' else int(row.horizon),
                                               'y_true_log': row.log_w if task == 'T1' else row.y,
                                               'y_pred_log': prediction[i],
                                               'w_true': row.w if task == 'T1' else row.w_target,
                                               'w_pred': weights[i]})
            print(f'{task} held out pond {held}: complete', flush=True)
    write_csv('codex_folds.csv', folds_output,
              ['task', 'held_out_pond', 'train_ponds', 'dropped_ponds', 'n_train', 'n_test',
               'inner0_val_ponds', 'inner1_val_ponds', 'inner2_val_ponds'])
    write_csv('codex_inner.csv', inner_output,
              ['task', 'model', 'held_out_pond', 'inner_fold', 'n_val', 'S', 'S_h1', 'S_h2', 'alpha'])
    predictions = write_csv('codex_predictions.csv', predictions_output,
                            ['task', 'model', 'held_out_pond', 'pond', 'date', 'origin_date',
                             'horizon', 'y_true_log', 'y_pred_log', 'w_true', 'w_pred'])
    write_csv('codex_curves.csv', curves_output,
              ['task', 'model', 'held_out_pond', 'chosen_curve', 'parameter1', 'parameter2',
               'parameter3', 'RSS', 'AIC', 'failed_curves'])
    scores_output = []
    for (task, model, horizon), group in predictions.groupby(['task', 'model', 'horizon'], sort=False):
        errors = np.abs(group.y_pred_log.to_numpy() - group.y_true_log.to_numpy())
        errors[group.w_pred.to_numpy() <= 0] = np.inf
        per_pond = {int(p): float(np.mean(errors[group.pond.to_numpy() == p]))
                    for p in sorted(group.pond.unique())}
        scores_output.append({'task': task, 'model': model, 'horizon': horizon,
                              'S': float(np.mean(list(per_pond.values()))), 'n_ponds': len(per_pond),
                              **{f'pond_{p}': per_pond.get(p, np.nan) for p in COHORT}})
    write_csv('codex_scores.csv', scores_output,
              ['task', 'model', 'horizon', 'S', 'n_ponds'] + [f'pond_{p}' for p in COHORT])
    linear_count = int(((predictions.model == 'linear') & (predictions.w_pred <= 0)).sum())
    lines = ['# Verification complete', '', 'Files:', '',
             '- verify.py', '- test_verify.py', '- audit_outputs.py', '- pytest.ini', '- codex_folds.csv',
             '- codex_inner.csv', '- codex_predictions.csv', '- codex_curves.csv',
             '- codex_scores.csv', '- DONE.md', '',
             *[f'{task}: {n} outer folds, {rows} test rows per model.'
               for task, (n, rows) in summary.items()], '',
             f'Linear outer-test predictions at or below 0 g: {linear_count}.', '',
             f'Linear inner-validation predictions at or below 0 g: {linear_inner_count}.', '',
             'Fallbacks (count means predicted rows; outer is test, inner0–2 are validation):', '',
             '| Task | Model | Held-out pond | Stage | Fallback | Count |',
             '| --- | --- | --- | --- | --- | --- |']
    lines += ['| ' + ' | '.join(map(str, row)) + ' |' for row in fallback_output]
    if not fallback_output:
        lines.append('No fallbacks used.')
    lines += ['', 'Assumptions and conventions:', '',
              '- dropped_ponds lists the shared-stream exclusion only; the held-out pond is already named. Pond 8 has no eligible rows and is absent from all fits and tests.',
              '- codex_curves.csv records final outer-training curve fits only, one per gc or anchored_gc model per fold. Parameters follow the selected formula’s table order. A failed curve means all three starts failed.',
              '- Curve and start ties retain the first in the specified order. Ridge ties use exact computed equality and select the larger alpha.',
              '- For deduplicated anchored events, origins precede targets and the first occurrence is retained after checking consistent times and log weights (absolute tolerance 1e-12 for floating-point roundoff).',
              '- If no T1 curve fits and cohort_mean without fallback lacks three ponds for a requested day, execution raises rather than inventing an unspecified prediction. This case did not occur.',
              '- T1 has blank origin_date, S_h1 and S_h2; non-ridge alpha and absent pond scores are blank. Nonpositive linear weights remain in w_pred, have blank y_pred_log, and score as infinity.',
              '- The logistic formula uses logaddexp for numerical stability. The growth cache reuses only fits with identical ordered training times and log weights.',
              '', 'Verification:', '',
              'Run /home/dev/development/publications/.venv/bin/python -I -m pytest -c pytest.ini --rootdir=. --confcutdir=. test_verify.py -q.',
              'Run /home/dev/development/publications/.venv/bin/python -I verify.py to regenerate every CSV and this report.',
              'Run /home/dev/development/publications/.venv/bin/python -I audit_outputs.py to independently audit the saved CSVs.',
              'CSV floats use 17 significant digits.']
    (ROOT / 'DONE.md').write_text('\n'.join(lines) + '\n')
    print(f'Outputs complete; nonpositive linear outer predictions: {linear_count}', flush=True)


if __name__ == '__main__':
    main()
