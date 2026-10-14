import importlib.util
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

spec = importlib.util.spec_from_file_location('verify', Path(__file__).with_name('verify.py'))
verify = importlib.util.module_from_spec(spec)
sys.modules['verify'] = verify
spec.loader.exec_module(verify)


def test_outer_folds_filter_eligibility_and_shared_stream():
    rows = pd.DataFrame({'pond': [1, 2, 3, 4, 6, 7, 8, 9, 10],
                         'eligible_eval': [True] * 6 + [False, True, True]})
    folds = verify.outer_folds(rows)
    assert [f['held_out'] for f in folds] == [1, 2, 3, 4, 6, 7, 9]
    assert folds[0]['train'].pond.tolist() == [3, 4, 6, 7, 9]
    assert folds[0]['dropped'] == [2]
    assert folds[1]['train'].pond.tolist() == [3, 4, 6, 7, 9]
    assert folds[1]['dropped'] == [1]
    assert folds[2]['train'].pond.tolist() == [1, 2, 4, 6, 7, 9]
    assert folds[2]['dropped'] == []


def test_inner_units_keep_shared_stream_together():
    rows = pd.DataFrame({'pond': [1, 2, 4, 6, 7, 9]})
    assert verify.inner_ponds(rows) == [[1, 2, 7], [4, 9], [6]]
    assert verify.inner_ponds(rows[rows.pond != 1]) == [[2, 7], [4, 9], [6]]


def test_S_weights_ponds_equally():
    assert verify.statistic_s([1, 1, 2], [0, 2, 5]) == 3
    assert np.isinf(verify.statistic_s([1, 2], [0, np.inf]))


def test_T2_scores_average_horizons_after_ponds():
    rows = pd.DataFrame({'pond': [1, 1, 2, 1, 2, 2],
                         'horizon': [1, 1, 1, 2, 2, 2], 'y': [0] * 6})
    s, h1, h2 = verify.score('T2', rows, np.array([0, 2, 5, 2, 4, 6]))
    assert (s, h1, h2) == (3.25, 3, 3.5)


def baseline_rows():
    return pd.DataFrame({'pond': [1, 2], 'origin_date': ['a', 'a'],
                         'horizon': [1, 2],
                         'target_date': ['b', 'c'], 't_k': [20, 20],
                         'dt_prev': [10, 10], 'dt_target': [14, 28],
                         'log_w_k': np.log([20, 10]), 'log_w_prev': np.log([10, 20]),
                         'w_origin': [20, 10], 'w_target': [40, 5],
                         'y': np.log([2, .5])})


def test_T2_baselines_and_nonpositive_linear():
    rows = baseline_rows()
    for model in ['persistence', 'linear', 'sgr']:
        prediction, weights, _ = verify.predict_model('T2', model, rows, rows)
        if model == 'persistence':
            np.testing.assert_array_equal(prediction, [0, 0])
            np.testing.assert_array_equal(weights, [20, 10])
        elif model == 'linear':
            np.testing.assert_allclose(weights, [34, -18])
            assert prediction[0] == pytest.approx(np.log(1.7))
            assert np.isnan(prediction[1])
            assert np.isinf(verify.score('T2', rows, prediction)[0])
        else:
            np.testing.assert_allclose(prediction, [1.4 * np.log(2), -2.8 * np.log(2)])
            np.testing.assert_allclose(weights, rows.w_origin * np.exp(prediction))


def test_cohort_growth_three_distinct_ponds():
    train = pd.concat([baseline_rows().iloc[[0]].assign(pond=p, y=y)
                       for p, y in [(1, 1), (2, 2), (3, 3)]], ignore_index=True)
    prediction, _, info = verify.predict_model('T2', 'cohort_growth', train, train.iloc[[0]])
    assert prediction[0] == 2
    assert info['fallbacks'] == {}


def test_ridge_preprocessing_uses_training_only():
    train = pd.DataFrame({'t': [1., 3., np.nan], 't2': [np.nan] * 3,
                          **{s + '_mean_w': [np.nan, 2., 3.] for s in verify.SENSORS}})
    prep = verify.RidgePreprocessor(['t', 't2'], True).fit(train)
    assert prep.columns == ['t']
    assert prep.medians['t'] == 2
    transformed = prep.transform(train.iloc[[0]].assign(t=100))
    assert transformed[0, 0] == pytest.approx(98 / np.sqrt(2 / 3))
    assert transformed.shape == (1, 7)


def test_alpha_ties_choose_larger():
    train = pd.DataFrame({'pond': [3, 4, 6], 't': [1., 1., 1.],
                          't2': [1., 1., 1.], 'log_w': [2., 2., 2.]})
    alpha, scores = verify.choose_alpha('T1', 'ridge_time', train)
    assert alpha == 1000
    assert len(scores) == 3


def test_anchored_points_deduplicate_event_dates():
    rows = baseline_rows().iloc[[0]].copy()
    rows = pd.concat([rows, rows.assign(horizon=2)], ignore_index=True)
    points = verify.anchored_points(rows)
    assert len(points) == 2
    assert points.t.tolist() == [20, 34]
    np.testing.assert_allclose(points.log_w, np.log([20, 40]))


def test_curve_fit_order_parameters_and_aic(monkeypatch):
    calls = []

    def fake_fit(function, t, y, p0, bounds, max_nfev):
        calls.append((function.__name__, p0, bounds, max_nfev))
        return np.array(p0), None

    monkeypatch.setattr(verify, 'curve_fit', fake_fit)
    monkeypatch.setattr(verify, 'CURVE_CACHE', {})
    t = np.array([14., 28., 42., 56.])
    target = np.array([2., 3., 4., 5.])
    result = verify.fit_growth(t, target)
    candidates = []
    for name, function, lower, upper, starts in verify.CURVES:
        rss = min(float(np.sum((function(t, *p) - target) ** 2)) for p in starts)
        candidates.append((len(t) * np.log(rss / len(t)) + 6, name, rss))
    aic, name, rss = min(candidates)
    assert result['best']['curve'] == name
    assert result['best']['RSS'] == pytest.approx(rss)
    assert result['best']['AIC'] == pytest.approx(aic)
    assert calls == [(function.__name__, start, (lower, upper), 20000)
                     for _, function, lower, upper, starts in verify.CURVES for start in starts]


def test_growth_failure_falls_back_to_sgr(monkeypatch):
    monkeypatch.setattr(verify, 'fit_growth', lambda *args: {'best': None})
    rows = baseline_rows()
    predicted, _, info = verify.predict_model('T2', 'anchored_gc', rows, rows)
    expected, _, _ = verify.predict_model('T2', 'sgr', rows, rows)
    np.testing.assert_allclose(predicted, expected)
    assert info['fallbacks'] == {'anchored_gc -> sgr': 2}


def test_T1_cohort_mean_threshold_and_curve_fallback(monkeypatch):
    monkeypatch.setattr(verify, 'fit_growth', lambda *args: {
        'best': {'function': lambda t, *p: np.full(len(t), 9.), 'parameters': []}})
    train = pd.DataFrame({'pond': [1, 2, 3, 1, 1, 2], 't': [14, 14, 14, 28, 28, 28],
                          'log_w': [1., 2., 3., 4., 5., 6.]})
    test = pd.DataFrame({'t': [14, 28]})
    prediction, _, info = verify.predict_model('T1', 'cohort_mean', train, test)
    np.testing.assert_array_equal(prediction, [2, 9])
    assert info['fallbacks'] == {'cohort_mean -> gc': 1}


def test_failed_growth_curves_are_recorded(monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError('synthetic failure')

    monkeypatch.setattr(verify, 'curve_fit', fail)
    monkeypatch.setattr(verify, 'CURVE_CACHE', {})
    result = verify.fit_growth(np.array([1., 2.]), np.array([3., 4.]))
    assert result['best'] is None
    assert result['failed'] == ['gompertz', 'von_bertalanffy', 'logistic']
    assert len(result['failed_starts']) == 9
