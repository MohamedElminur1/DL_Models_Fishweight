"""Generate supplementary sensor QC counts without rerunning the data pipeline."""

from datetime import date, timedelta
from pathlib import Path
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
from shared import MISSING, Sources, literal, write_table

HEADER = ['Pond', 'Sensor', 'Unit as labeled', 'Readings after duplicate removal',
          'Missing raw readings', 'Non-finite readings', 'Out-of-range readings',
          'Flat-line readings', 'Valid readings', 'Duplicate rows dropped',
          'Sensor-days', 'Observed sensor-days', 'Interpolated sensor-days', 'Missing sensor-days']
READING_KEYS = ('n_values', 'n_missing_raw', 'n_nonfinite', 'n_out_of_range',
                'n_flatline', 'n_valid', 'n_duplicate_rows_dropped')
DAY_KEYS = ('n_days', 'n_observed', 'n_interpolated', 'n_missing')


def main() -> None:
    """Join the two saved count files by pond and sensor, using audited source labels."""
    sources = Sources()
    cfg = sources.config('configs/phase4_data.yaml')
    sources.text('ANALYSIS_PLAN.md')
    qc = sources.csv('results/qc/qc_counts.csv')
    daily = sources.csv('results/qc/interpolation_counts.csv')
    schema = sources.csv('results/audit/schema.csv')
    identities = [(str(p), s['name']) for p in cfg['cohort']['ponds'] for s in cfg['sensors']]
    readings = {(r['pond'], r['sensor']): r for r in qc}
    days = {(r['pond'], r['sensor']): r for r in daily}
    if len(readings) != len(qc) or len(days) != len(daily) or set(readings) != set(identities) or set(days) != set(identities):
        raise ValueError('QC sources have duplicate, missing or unexpected pond/sensor rows')
    labels = {(r['pond'], r['canonical_name']): r for r in schema}
    rows, ranges = [], []
    for pond in cfg['cohort']['ponds']:
        for sensor in cfg['sensors']:
            key = str(pond), sensor['name']
            label = labels[str(pond), sensor['raw']]
            name = label['original_name'].split('(', 1)[0].strip()
            name = 'pH' if name.upper() == 'PH' else name.capitalize()
            values = [readings[key][k] for k in READING_KEYS] + [days[key][k] for k in DAY_KEYS]
            counts = [int(v) for v in values]
            if any(v < 0 for v in counts) or counts[0] != sum(counts[1:6]) or counts[7] != sum(counts[8:]):
                raise ValueError(f'counts do not balance: {key}')
            rows.append([str(pond), literal(name), literal(label['unit_as_labelled']) or MISSING, *values])
            if pond == cfg['cohort']['ponds'][0]:
                lo, hi = sensor['lo'], sensor['hi']
                if sensor['lo_inclusive'] and sensor['hi_inclusive'] and hi is not None:
                    limits = f'{lo:g} to {hi:g}'
                else:
                    limits = ('≥' if sensor['lo_inclusive'] else '>') + f' {lo:g}'
                    if hi is not None:
                        limits += (' and ≤' if sensor['hi_inclusive'] else ' and <') + f' {hi:g}'
                ranges.append(f'{name.lower() if name != "pH" else name} {limits}')
    end = date.fromisoformat(cfg['cohort']['day0']) + timedelta(days=cfg['cohort']['grid']['period_days'] * cfg['cohort']['grid']['k_last'])
    write_table('sensor_qc', HEADER, rows, sources, [
        'Title: Supplementary sensor quality control, one row per cohort pond and sensor.',
        'Definitions follow ANALYSIS_PLAN.md sections 2.5 to 2.7; thresholds and daily grid are read from configs/phase4_data.yaml.',
        'Values count timestamped raw readings after exact duplicate rows are dropped, not hourly averages. Date-only fish rows do not enter QC. Missing raw readings are empty or NaN. Non-finite readings are positive or negative infinity. Invalid readings are assigned to the first failed rule: non-finite, out of range, then flat-line. Valid readings pass all rules; these five outcomes sum to the readings count.',
        'Duplicate rows dropped counts exact repeated timestamped rows before QC. The duplicate count is repeated for each sensor of a pond and must not be summed across sensors.',
        'Allowed raw ranges, plan 2.5: ' + '; '.join(ranges) + '. Out-of-range readings fail these limits and become missing, never clipped. Ammonia is checked on its raw scale and valid values are log10-transformed before daily aggregation; the counts refer to the raw ammonia readings.',
        f"Flat-line readings belong to runs of identical consecutive finite raw values spanning at least {cfg['qc']['flatline_hours']} h; a missing or non-finite value or a logging gap of more than {cfg['qc']['flatline_max_gap_hours']} h ends a run. The whole run is invalid, including when it crosses an event date (plan 2.5).",
        f"Sensor-days count the full padded daily grid, {cfg['daily']['first_date']} to {end.isoformat()}, inclusive, for each pond and sensor. Observed sensor-days have valid readings in at least {cfg['daily']['min_hours']} distinct clock hours (plan 2.6).",
        f"Interpolated sensor-days are linearly interpolated only inside runs of at most {cfg['daily']['max_gap_days']} missing days with observed days on both sides (plan 2.7). Missing sensor-days have neither an observed nor an interpolated value. These three daily statuses sum to Sensor-days. Every interpolated sensor-day is counted here, including days unavailable to an event because its later observed anchor is on or after the event date.",
        f"Total interpolated sensor-days: {sum(int(r['n_interpolated']) for r in daily)}.",
        'Units are reproduced as labelled in the source, without conversion, from results/audit/schema.csv. – means no unit is labelled, as for pH. The source uses g/ml for dissolved oxygen, ammonia and nitrate in most cohort ponds, and mg/L in pond 8; these labels do not change the frozen QC limits.',
    ])


if __name__ == '__main__':
    main()
