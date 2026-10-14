import importlib.util
import unittest
from pathlib import Path
import numpy as np
import pandas as pd

spec = importlib.util.spec_from_file_location('verify', Path(__file__).with_name('verify.py'))
v = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v)


class VerificationTests(unittest.TestCase):
    def test_clock_and_date_parsing(self):
        self.assertEqual(v.parse_timestamp('2021-12-06T11:34:43+01:00'), (pd.Timestamp('2021-12-06 11:34:43'), True))
        self.assertEqual(v.parse_timestamp('5/12/2021'), (pd.Timestamp('2021-12-05'), False))
        self.assertEqual(v.parse_timestamp('2021-06-19 00:00:05 CET')[0], pd.Timestamp('2021-06-19 00:00:05'))

    def test_flatline_gap_and_rule_precedence(self):
        times = pd.date_range('2021-06-01', periods=26, freq='h')
        values = np.full(26, 25.0)
        valid, counts = v.sensor_qc(times, values, 'temperature')
        self.assertEqual(counts, (0, 0, 26, 0))
        self.assertFalse(valid.any())
        times = times.to_series().reset_index(drop=True)
        times.iloc[13:] += pd.Timedelta(hours=2)
        valid, counts = v.sensor_qc(pd.DatetimeIndex(times), values, 'temperature')
        self.assertEqual(counts, (0, 0, 0, 26))
        _, counts = v.sensor_qc(pd.date_range('2021-06-01', periods=26, freq='h'), np.full(26, 41.0), 'temperature')
        self.assertEqual(counts, (0, 26, 0, 0))
        _, counts = v.sensor_qc(pd.date_range('2021-06-01', periods=5, freq='h'), np.array([np.nan, np.inf, 9, 10, 40]), 'temperature')
        self.assertEqual(counts, (2, 1, 0, 2))

    def test_daily_coverage_and_interpolation(self):
        times = pd.date_range('2021-05-22', periods=12, freq='h').append(pd.date_range('2021-05-25', periods=12, freq='h')).append(pd.date_range('2021-05-29', periods=11, freq='h'))
        values = np.r_[np.full(12, 2.), np.full(12, 8.), np.full(11, 10.)]
        daily = v.daily_table(times, values)
        self.assertEqual(daily.loc['2021-05-22', 'status'], 'observed')
        self.assertEqual(daily.loc['2021-05-23', 'mean'], 4.)
        self.assertEqual(daily.loc['2021-05-24', 'mean'], 6.)
        self.assertEqual(daily.loc['2021-05-23', 'right_anchor'], pd.Timestamp('2021-05-25'))
        self.assertEqual(daily.loc['2021-05-29', 'status'], 'missing')
        self.assertEqual(daily.loc['2021-05-29', 'n_hours'], 11)
        self.assertTrue(pd.isna(daily.loc['2021-05-29', 'mean']))

    def test_reference_date_blocks_future_anchor_and_three_day_minimum(self):
        times = pd.date_range('2021-06-19', periods=12, freq='h').append(pd.date_range('2021-06-22', periods=12, freq='h'))
        daily = v.daily_table(times, np.r_[np.full(12, 2.), np.full(12, 8.)])
        tables = {sensor: daily for sensor in v.SENSORS}
        before = v.window_features(tables, pd.Timestamp('2021-06-22'))
        self.assertEqual(before['t1_n_covered_window'], 1)
        self.assertTrue(np.isnan(before['t1_temperature_mean_w']))
        after = v.window_features(tables, pd.Timestamp('2021-06-23'))
        self.assertEqual(after['t1_n_covered_window'], 4)
        self.assertEqual(after['t1_temperature_mean_w'], 5.)

    def test_events_and_t2_previous_event_gap(self):
        rows = [('2021-06-19', 1., 2.), ('2021-06-19', 1., 3.), ('2021-06-19', 2., 3.), ('2021-07-17', 4., 5.), ('2021-07-31', 6., 7.)]
        frame = pd.DataFrame(rows, columns=['date', 'weight', 'length'])
        frame['date'] = pd.to_datetime(frame['date'])
        events = v.fish_events(frame, '01')
        self.assertEqual(events.iloc[0]['n_rows'], 3)
        self.assertEqual(events.iloc[0]['weight'], 1.)
        self.assertEqual(events.iloc[0]['length'], 3.)
        features = pd.DataFrame([{'pond': '01', 'date': date, 't1_n_covered_window': 7, 't2_n_covered_window': 14} for date in events.date])
        t1, t2 = v.eligibility(events, features)
        self.assertEqual(len(t1), 2)
        self.assertEqual(len(t2), 1)
        self.assertEqual(t2.iloc[0]['dt_prev'], 28)
        self.assertEqual(t2.iloc[0]['horizon'], 1)
        self.assertTrue(t2.iloc[0]['eligible'])


if __name__ == '__main__':
    unittest.main()
