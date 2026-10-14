"""Unit tests for src/scoring.py: the primary statistic S, hand-computed on small cases."""

import numpy as np
import pandas as pd
import pytest

from src import scoring


def test_abs_log_error_is_the_absolute_difference_and_nan_predictions_are_infinite():
    err = scoring.abs_log_error([1.0, 2.0, np.nan, -1.0], [1.5, 1.0, 0.0, -1.0])
    np.testing.assert_allclose(err, [0.5, 1.0, np.inf, 0.0])


def test_abs_log_error_rejects_bad_targets_and_shapes():
    with pytest.raises(ValueError, match="not finite"):
        scoring.abs_log_error([1.0], [np.nan])
    with pytest.raises(ValueError, match="shapes"):
        scoring.abs_log_error([1.0, 2.0], [1.0])


def test_s_is_the_mean_over_ponds_of_the_per_pond_means_not_the_pooled_mean():
    ponds = [1, 1, 1, 2]
    errors = [0.1, 0.2, 0.3, 0.8]
    per_pond = scoring.pond_scores(ponds, errors)
    np.testing.assert_allclose(per_pond.to_numpy(), [0.2, 0.8])
    assert list(per_pond.index) == [1, 2]
    assert scoring.s_value(per_pond) == pytest.approx(0.5)            # (0.2 + 0.8) / 2
    assert np.mean(errors) == pytest.approx(0.35)                      # the pooled mean differs
    assert scoring.s_statistic(ponds, np.log([2, 2, 2, 2]) + np.array(errors), np.log([2, 2, 2, 2])) \
        == pytest.approx(0.5)


def test_one_infinite_error_makes_the_pond_and_the_set_infinite():
    pond = scoring.pond_scores([1, 1, 2], [0.1, np.inf, 0.2])
    assert pond[1] == np.inf and pond[2] == pytest.approx(0.2)
    assert scoring.s_value(pond) == np.inf


def test_s_needs_samples():
    with pytest.raises(ValueError):
        scoring.pond_scores([], [])
    with pytest.raises(ValueError):
        scoring.s_value([])


def test_horizon_s_scores_each_horizon_separately():
    ponds = [1, 1, 2, 2, 2]
    horizons = [1, 2, 1, 1, 2]
    errors = [0.1, 0.5, 0.3, 0.5, 0.9]
    result = scoring.horizon_s(ponds, horizons, errors)
    # h=1: pond 1 -> 0.1, pond 2 -> 0.4 ; h=2: pond 1 -> 0.5, pond 2 -> 0.9
    assert result == {1: pytest.approx(0.25), 2: pytest.approx(0.7)}


def test_t2_tuning_s_is_the_mean_of_the_two_horizon_scores_not_the_pooled_score():
    ponds = [1, 1, 1, 1, 2]
    horizons = [1, 1, 1, 2, 2]          # horizon 1 has three times the samples of horizon 2 in pond 1
    errors = [0.1, 0.1, 0.1, 0.9, 0.7]
    s_h1 = 0.1                          # only pond 1 has horizon-1 samples
    s_h2 = (0.9 + 0.7) / 2              # ponds 1 and 2 at horizon 2
    assert scoring.tuning_s("T2", ponds, errors, horizons) == pytest.approx((s_h1 + s_h2) / 2)
    pooled = scoring.s_value(scoring.pond_scores(ponds, errors))   # pond 1: 0.3, pond 2: 0.7
    assert pooled == pytest.approx(0.5) and pooled != pytest.approx((s_h1 + s_h2) / 2)


def test_t2_tuning_s_needs_both_horizons_and_the_horizons():
    with pytest.raises(ValueError, match="horizon"):
        scoring.tuning_s("T2", [1, 2], [0.1, 0.2], [1, 1])
    with pytest.raises(ValueError, match="horizon of every sample"):
        scoring.tuning_s("T2", [1, 2], [0.1, 0.2])


def test_t1_tuning_s_is_plain_s():
    assert scoring.tuning_s("T1", [1, 1, 2], [0.1, 0.3, 0.6]) == pytest.approx((0.2 + 0.6) / 2)


def test_seed_averaged_pond_scores_average_each_ponds_values_over_seeds():
    frame = pd.DataFrame({
        "pond": [1, 1, 2, 1, 1, 2],
        "seed": [0, 0, 0, 1, 1, 1],
        "abs_err": [0.2, 0.4, 0.6, 0.0, 0.2, 1.0],
    })
    per_pond = scoring.seed_averaged_pond_scores(frame)
    # pond 1: seed 0 -> 0.3, seed 1 -> 0.1, mean 0.2 ; pond 2: (0.6 + 1.0) / 2 = 0.8
    np.testing.assert_allclose(per_pond.to_numpy(), [0.2, 0.8])
    assert scoring.s_value(per_pond) == pytest.approx(0.5)


def test_seed_averaging_refuses_seeds_that_cover_different_ponds():
    frame = pd.DataFrame({"pond": [1, 2, 1], "seed": [0, 0, 1], "abs_err": [0.1, 0.2, 0.3]})
    with pytest.raises(ValueError, match="same ponds"):
        scoring.seed_averaged_pond_scores(frame)
