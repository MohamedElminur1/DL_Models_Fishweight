from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import g4p2


def daily_row(date, value, *, sensor="temperature", status="observed", anchor=None, pond=3):
    return dict(pond=pond, date=date, sensor=sensor, status=status,
                right_anchor_date=anchor, mean=value, sd=value + 1,
                min=value + 2, max=value + 3)


def t1_row(date, t, target=0.0, pond=3):
    return dict(pond=pond, date=date, t=t, log_w=target)


def test_validity_is_strict_and_sample_specific():
    reference = pd.Timestamp("2021-06-22")
    assert g4p2.cell_valid("observed", pd.NaT, reference)
    assert g4p2.cell_valid("interpolated", pd.Timestamp("2021-06-21"), reference)
    assert not g4p2.cell_valid("interpolated", reference, reference)
    assert not g4p2.cell_valid("interpolated", pd.Timestamp("2021-06-23"), reference)
    assert not g4p2.cell_valid("interpolated", pd.NaT, reference)
    assert not g4p2.cell_valid("missing", pd.Timestamp("2021-06-21"), reference)


def test_t1_window_excludes_reference_and_left_pads():
    store = g4p2.DailyStore(pd.DataFrame([
        daily_row("2021-06-18", 900),
        daily_row("2021-06-19", 1),
        daily_row("2021-06-20", 2),
        daily_row("2021-06-21", 800),
    ]))
    window = g4p2.make_window(store, t1_row("2021-06-21", 2), "T1", 4)
    np.testing.assert_array_equal(window.valid[:, 0], [False, False, True, True])
    np.testing.assert_array_equal(window.values[:, 0, 0], [0, 0, 1, 2])
    assert np.all(window.indices[:2] == -1)


def test_t2_window_uses_origin_and_exactly_28_days():
    store = g4p2.DailyStore(pd.DataFrame([
        daily_row("2021-06-01", 999),
        daily_row("2021-06-02", 1),
        daily_row("2021-06-29", 2),
        daily_row("2021-06-30", 888),
        daily_row("2021-06-03", 777, status="interpolated", anchor="2021-07-01"),
    ]))
    row = dict(pond=3, origin_date="2021-06-30", target_date="2021-07-14")
    window = g4p2.make_window(store, row, "T2", None)
    assert window.values.shape == (28, 6, 4)
    assert window.values[0, 0, 0] == 1
    assert window.values[-1, 0, 0] == 2
    assert window.valid[:, 0].sum() == 2
    assert not window.valid[1, 0]


def test_channel_layout_masks_and_invalid_zero():
    store = g4p2.DailyStore(pd.DataFrame([
        daily_row("2021-06-19", 10),
        daily_row("2021-06-19", 20, sensor="nitrate"),
        daily_row("2021-06-19", 100, sensor="do", status="missing"),
    ]))
    window = g4p2.make_window(store, t1_row("2021-06-20", 1), "T1", 2)
    scalers = dict(channel_mean=np.arange(24, dtype=np.float64),
                   channel_sd=np.full(24, 2.0))
    sequence = g4p2.scale_window(window, scalers)
    assert sequence.shape == (2, 30)
    assert sequence.dtype == np.float32
    np.testing.assert_array_equal(sequence[0], np.zeros(30))
    np.testing.assert_array_equal(sequence[1, :4], [5, 5, 5, 5])
    np.testing.assert_array_equal(sequence[1, 20:24], [0, 0, 0, 0])
    np.testing.assert_array_equal(sequence[1, 4:20], np.zeros(16))
    np.testing.assert_array_equal(sequence[1, 24:], [1, 0, 0, 0, 0, 1])


def test_scalers_use_unique_training_cells_valid_in_any_own_window():
    store = g4p2.DailyStore(pd.DataFrame([
        daily_row("2021-06-19", 0),
        daily_row("2021-06-20", 2),
        daily_row("2021-06-21", 10, status="interpolated", anchor="2021-06-22"),
        daily_row("2021-06-22", 999, status="interpolated", anchor="2021-06-23"),
        daily_row("2021-06-23", 888),
        daily_row("2021-06-19", 777, pond=2),
        daily_row("2021-06-18", 666),
    ]))
    rows = pd.DataFrame([t1_row("2021-06-22", 3, 1), t1_row("2021-06-23", 4, 3)])
    windows = [g4p2.make_window(store, row, "T1", 4) for row in rows.to_dict("records")]
    assert not windows[0].valid[-1, 0]
    assert windows[1].valid[-2, 0]
    scalers = g4p2.fit_scalers(store, windows, rows, "T1", 4)
    np.testing.assert_allclose(scalers["channel_mean"][:4], [4, 5, 6, 7])
    np.testing.assert_allclose(scalers["channel_sd"][:4], np.sqrt(56 / 3))
    np.testing.assert_array_equal(scalers["channel_mean"][4:], np.zeros(20))
    np.testing.assert_array_equal(scalers["channel_sd"][4:], np.ones(20))
    np.testing.assert_array_equal(scalers["static_mean"], [3.5])
    np.testing.assert_array_equal(scalers["static_sd"], [0.5])
    assert scalers["target_mean"] == 2
    assert scalers["target_sd"] == 1
    assert scalers["L"] == 4


def test_constant_scales_and_t2_static_order():
    store = g4p2.DailyStore(pd.DataFrame([daily_row("2021-06-19", 4)]))
    rows = pd.DataFrame([dict(pond=3, origin_date="2021-06-20", log_w_k=5,
                              log_w_prev=4, t_k=1, dt_prev=14, dt_target=28, y=2)])
    window = g4p2.make_window(store, rows.iloc[0], "T2", None)
    scalers = g4p2.fit_scalers(store, [window], rows, "T2", None)
    np.testing.assert_array_equal(scalers["static_mean"], [5, 4, 1, 14, 28])
    np.testing.assert_array_equal(scalers["static_sd"], np.ones(5))
    np.testing.assert_array_equal(scalers["channel_sd"], np.ones(24))
    assert scalers["target_sd"] == 1
