"""Read-only check of two Gate 1 statements about ponds 10 to 12 (row counts, pond 11 weights)."""
import sys
from pathlib import Path

import pandas as pd

raw = Path(sys.argv[1])
for pond in (10, 11, 12):
    df = pd.read_csv(raw / f"pond{pond:02d}.csv", dtype=str)
    stamp = df["created_at"].fillna("")
    timed = stamp.str.contains(r"\d{1,2}:\d{2}", regex=True)
    weight_col = [c for c in df.columns if c.lower().startswith("weight")][0]
    w = pd.to_numeric(df[weight_col], errors="coerce")
    print(f"pond {pond}: rows {len(df)}, rows with clock time {int(timed.sum())}, "
          f"weight non-missing {int(w.notna().sum())}, zeros {int((w == 0).sum())}, "
          f"min {w.min()}, max {w.max()}, values >= 345 {int((w >= 345).sum())}")
