"""Independently check the CSV artifacts without importing verify.py."""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
COHORT = [1, 2, 3, 4, 6, 7, 8, 9]


def main() -> None:
    predictions = pd.read_csv(ROOT / 'codex_predictions.csv', float_precision='round_trip')
    pond_columns = ['train_ponds', 'dropped_ponds'] + [f'inner{j}_val_ponds' for j in range(3)]
    folds = pd.read_csv(ROOT / 'codex_folds.csv', dtype={c: str for c in pond_columns}).fillna('')
    inner = pd.read_csv(ROOT / 'codex_inner.csv', float_precision='round_trip')
    curves = pd.read_csv(ROOT / 'codex_curves.csv', float_precision='round_trip')
    scores = pd.read_csv(ROOT / 'codex_scores.csv', float_precision='round_trip')
    assert (len(folds), len(inner), len(predictions), len(curves), len(scores)) == (14, 252, 546, 14, 19)
    for task, table in [('T1', 't1_table.csv'), ('T2', 't2_table.csv')]:
        source = pd.read_csv(Path('/home/dev/development/publications/data/processed') / table)
        source = source[source.eligible_eval & source.pond.isin(COHORT)]
        assert len(source) == (35 if task == 'T1' else 53)
        for _, group in predictions[predictions.task == task].groupby('model'):
            assert len(group) == len(source)
            keys = ['pond', 'date'] if task == 'T1' else ['pond', 'origin_date', 'horizon']
            joined = group.merge(source, on=keys, suffixes=('_out', '_in'), validate='one_to_one')
            assert len(joined) == len(source)
            assert (joined.held_out_pond == joined.pond).all()
            np.testing.assert_allclose(joined.y_true_log, joined.log_w if task == 'T1' else joined.y, rtol=1e-14)
            np.testing.assert_allclose(joined.w_true, joined.w if task == 'T1' else joined.w_target, rtol=1e-14)
            if task == 'T2':
                assert (joined.date == joined.target_date).all()
            positive = joined.w_pred > 0
            expected = np.exp(joined.loc[positive, 'y_pred_log'])
            if task == 'T2':
                expected *= joined.loc[positive, 'w_origin']
            np.testing.assert_allclose(joined.loc[positive, 'w_pred'], expected, rtol=1e-13)
            assert joined.loc[~positive, 'y_pred_log'].isna().all()
        for _, fold in folds[folds.task == task].iterrows():
            held = int(fold.held_out_pond)
            dropped = [3 - held] if held in [1, 2] else []
            training = source[~source.pond.isin([held] + dropped)]
            assert fold.n_train == len(training)
            assert fold.n_test == (source.pond == held).sum()
            assert fold.train_ponds == ' '.join(str(p) for p in sorted(training.pond.unique()))
            assert fold.dropped_ponds == ' '.join(map(str, dropped))
            units = []
            for pond in sorted(training.pond.unique()):
                if pond == 2 and 1 in training.pond.values:
                    continue
                units.append([1, 2] if pond == 1 and 2 in training.pond.values else [pond])
            for j in range(3):
                ponds = [p for i, unit in enumerate(units) if i % 3 == j for p in unit]
                assert fold[f'inner{j}_val_ponds'] == ' '.join(map(str, ponds))
                rows = inner[(inner.task == task) & (inner.held_out_pond == held) & (inner.inner_fold == j)]
                assert rows.n_val.eq(training.pond.isin(ponds).sum()).all()
    for _, row in scores.iterrows():
        group = predictions[(predictions.task == row.task) & (predictions.model == row.model) &
                            (predictions.horizon == row.horizon)].copy()
        group['error'] = abs(group.y_pred_log - group.y_true_log)
        group.loc[group.w_pred <= 0, 'error'] = np.inf
        means = [np.mean(g.error.to_numpy()) for _, g in group.groupby('pond')]
        np.testing.assert_allclose(row.S, np.mean(means), rtol=1e-13)
        assert row.n_ponds == group.pond.nunique()
        for pond, pond_rows in group.groupby('pond'):
            np.testing.assert_allclose(row[f'pond_{pond}'], np.mean(pond_rows.error.to_numpy()), rtol=1e-13)
    assert np.isfinite(predictions.w_pred).all()
    assert np.isfinite(predictions.y_pred_log).all()
    assert curves.chosen_curve.notna().all()
    assert curves.failed_curves.isna().all()
    assert np.isfinite(inner.S).all()
    for _, group in inner[inner.model.str.startswith('ridge_')].groupby(['task', 'model', 'held_out_pond']):
        assert group.alpha.nunique() == 1
        assert np.isclose(group.alpha.iloc[0], 10. ** np.arange(-3, 3.01, .5)).any()
    for _, row in inner[inner.task == 'T2'].iterrows():
        np.testing.assert_allclose(row.S, (row.S_h1 + row.S_h2) / 2, rtol=1e-13)
    print('PASS: 14 folds, 252 inner scores, 546 predictions, 14 curves, 19 aggregate scores.')
    print(scores[['task', 'model', 'horizon', 'S']].to_string(index=False))


if __name__ == '__main__':
    main()
