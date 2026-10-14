"""Rebuild the eight part 2 figures; append their source-complete captions with --captions.

Run from the repository root:
  env PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/tmp/mplconfig_figs .venv/bin/python -I reporting/figs/run_part2.py --captions
Each figure overwrites only its own PNG/PDF. Caption sections append once, preserving the
existing part 1 text. A failed run leaves completed figure pairs and can be rerun safely.
"""

import argparse
import importlib
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from reporting.figs import _part2 as p

FIGURES = (
    ("fig_pond_scores", "main"),
    ("fig_confirmatory", "main"),
    ("fig_permutation", "main"),
    ("fig_trajectories", "main"),
    ("fig_t2_scatter", "main"),
    ("fig_calendar", "supp"),
    ("fig_learning_curves", "supp"),
    ("fig_ablation", "supp"),
)


def append_captions(records: list[dict]) -> None:
    """Append only absent sections; refuse to silently retain a stale part 2 caption."""
    path = p.ROOT / "figures/CAPTIONS.md"
    current = path.read_text()
    additions = []
    for record in records:
        stem, group = record["stem"], record["group"]
        sources = ",\n".join(f"`{source}`" for source in record["sources"])
        section = (f"## {stem}\n\n{record['caption'].strip()}\n\n"
                   f"Script `reporting/figs/{stem}.py`. Saved data sources:\n{sources}.\n"
                   "Configuration `configs/phase4.yaml`; style `reporting/figstyle.py`; "
                   "display labels `reporting/names.py`.\n"
                   f"Writes `figures/{group}/{stem}.png` and `figures/{group}/{stem}.pdf`.\n"
                   f"Rebuild from the repository root with "
                   f"`env PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/tmp/mplconfig_figs "
                   f".venv/bin/python -I reporting/figs/{stem}.py`.\n")
        if f"## {stem}\n" in current:
            if section.strip() not in current:
                raise RuntimeError(f"Caption differs for {stem}; inspect before appending")
        else:
            additions.append(section)
    if additions:
        with path.open("a") as handle:
            handle.write("\n\n" + "\n\n".join(additions))
        print(f"Appended {len(additions)} captions to figures/CAPTIONS.md")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--captions", action="store_true")
    args = parser.parse_args()
    records = []
    for stem, _ in FIGURES:
        p.SOURCES.clear()
        record = importlib.import_module(f"reporting.figs.{stem}").main()
        records.append(record)
    if args.captions:
        append_captions(records)


if __name__ == "__main__":
    main()
