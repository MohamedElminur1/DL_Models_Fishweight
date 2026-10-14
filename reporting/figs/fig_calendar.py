"""Descriptive T1 calendar hold-out predictions after the frozen cut-off."""

import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from reporting.figs import _part2 as p
from reporting.figs._part2_trajectories import draw

CAPTION = """Descriptive T1 calendar hold-out, one panel per pond with a test event.
The dashed line marks the cut-off, 28 August 2021. Filled circles are real weighings from
the displayed period, including unscored events for context. Hollow markers show saved
predictions only on eligible test dates on or after the cut-off; no later weighing enters
training. Line styles distinguish the primary pair, the configured permutation-test tabular
models and Static MLP. Neural predictions are arithmetic means in grams over the five seeds,
with light bands and vertical ranges spanning the minimum to maximum seed prediction.
Panels share one logarithmic gram range that includes every real weighing shown. A
prediction outside the range is an open triangle at the panel edge pointing outward, with
its value labelled in grams. The common calendar axis ends on 31 October 2021; later
weighings, which were not scored, are not shown. Connecting
segments only join saved test predictions; they are not daily estimates. A pond with one
test event has markers and ranges but no prediction line segment. The empty eighth grid
position corresponds to no additional test pond. This hold-out is descriptive, with no
p-value or confidence interval; seed ranges describe fitted-seed variation only."""


def main() -> dict:
    return p.finish(draw("chrono"), "fig_calendar", "supp", CAPTION)


if __name__ == "__main__":
    main()
