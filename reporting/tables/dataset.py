"""Generate the cohort table using each pond's own inclusive sensor-logging period."""

from datetime import date
from pathlib import Path
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))

from shared import Sources, number, write_table


def main() -> None:
    sources = Sources()
    cfg = sources.config("configs/phase4.yaml")
    qc = sources.csv("results/qc/qc_counts.csv")
    audit = {int(row["pond"]): row for row in sources.csv("results/audit/summary.csv")}
    gaps = sources.csv("results/audit/gaps.csv")
    shared = sources.csv("results/audit/cross_pond_duplicates.csv")
    events = sources.csv("data/processed/events.csv")
    t1 = sources.csv(cfg["paths"]["t1_table"])
    t2 = sources.csv(cfg["paths"]["t2_table"])
    daily = sources.csv(cfg["paths"]["daily_wq"])
    sensors = set(cfg["design"]["sensors"])
    if len(sensors) != 6 or cfg["design"]["horizons"] != [1, 2]:
        raise ValueError("expected six sensors and horizons 1 and 2")
    rows = []
    duplicates = {}
    for pond in cfg["design"]["cohort"]:
        counts = [row for row in qc if int(row["pond"]) == pond]
        if len(counts) != 6 or {row["sensor"] for row in counts} != sensors:
            raise ValueError(f"pond {pond}: unexpected QC sensors")
        row_counts = {int(row["n_values"]) for row in counts}
        dropped = {int(row["n_duplicate_rows_dropped"]) for row in counts}
        if len(row_counts) != 1 or len(dropped) != 1:
            raise ValueError(f"pond {pond}: inconsistent QC row counts")
        duplicates[pond] = dropped.pop()
        first, last = (audit[pond][key][:10] for key in ("min_ts", "max_ts"))
        denominator = ((date.fromisoformat(last) - date.fromisoformat(first)).days + 1) * 6
        days = [row for row in daily if int(row["pond"]) == pond and first <= row["date"] <= last]
        identities = {(row["date"], row["sensor"]) for row in days}
        if len(days) != denominator or len(identities) != denominator or {row["sensor"] for row in days} != sensors:
            raise ValueError(f"pond {pond}: incomplete or duplicated daily grid")
        if any(row["status"] not in {"observed", "interpolated", "missing"} for row in days):
            raise ValueError(f"pond {pond}: unknown daily status")
        shares = [number(100 * sum(row["status"] == status for row in days) / denominator, "percent")
                  for status in ("observed", "interpolated")]
        eligible_t1 = sum(int(row["pond"]) == pond and row["eligible_eval"] == "True" for row in t1)
        horizons = [sum(int(row["pond"]) == pond and int(row["horizon"]) == horizon
                        and row["eligible_eval"] == "True" for row in t2) for horizon in (1, 2)]
        longest = max(float(row["duration_h"]) for row in gaps if int(row["pond"]) == pond) / 24
        rows.append([str(pond), str(row_counts.pop()), f"{first} to {last}", f"{longest:.1f}",
                     str(sum(int(row["pond"]) == pond for row in events)), str(eligible_t1),
                     " / ".join(map(str, horizons)), " / ".join(shares)])
    pair = [row for row in shared if {int(row["pond_a"]), int(row["pond_b"])} == {1, 2}]
    if len(pair) != 1:
        raise ValueError("expected one pond 1/2 shared-stream audit row")
    write_table("dataset", ["Pond", "Sensor rows after QC", "Logging period", "Longest logging gap (days)",
                            "Weighings", "Eligible T1 events", "Eligible T2 samples (h = 1 / h = 2)",
                            "Sensor-days observed / interpolated (%)"], rows, sources, [
        "QC retains timestamped rows after exact duplicate removal; invalid individual sensor values become missing. Rows are not complete-case filtered.",
        f"Exact duplicate rows dropped in QC: {sum(duplicates.values())} across the cohort; " + "; ".join(f"pond {pond}: {count}" for pond, count in duplicates.items()) + ". Counts repeated per sensor are counted once per pond.",
        f"Ponds 1 and 2 are one sensor stream (DECISIONS.md Gate 1); {int(pair[0]['n_common_ts_raw']):,} common timestamps, quoted from results/audit/cross_pond_duplicates.csv.",
        "Logging dates are the saved first and last sensor timestamps' dates in each file's clock, without timezone conversion; fish-only rows after logging ends do not extend the period.",
        "Longest logging gap is the maximum duration_h in results/audit/gaps.csv divided by 24, rounded to one decimal day.",
        "Weighings include day 0 and fish-only events. Eligibility uses eligible_eval=True. T2 cells list eligible origin-horizon pairs for h = 1 / h = 2.",
        "Sensor-day shares use saved daily statuses restricted to each pond's first through last sensor date, inclusive. Denominator is the number of days in that period times six sensors, including missing days and excluding grid padding. Cells list observed / interpolated percentages.",
        "Shares are marginal daily coverage; event-specific causal interpolation restrictions remain in the saved eligibility flags.",
    ])


if __name__ == "__main__":
    main()
