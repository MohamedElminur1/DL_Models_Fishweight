"""T1 event-date trajectories for the prespecified shown models under LOPO."""

import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from reporting.figs import _part2 as p
from reporting.figs._part2_trajectories import draw

CAPTION = """T1 weight estimation under leave-one-pond-out validation, one panel per pond
with eligible scored events. The eighth position of the 2 by 4 grid is unused because pond 8
has no eligible event. Filled circles are real weighings, including unscored weighings for
context. Hollow markers and line styles distinguish the fixed shown models: the primary
baseline and deep model, the configured permutation-test tabular models, and Static MLP.
Predictions occur only at eligible event dates; connecting segments are visual guides and
do not add observations or daily predictions. LSTM and Static MLP predictions are arithmetic
means of predicted grams across the five seeds. Light bands span the seed minimum and
maximum; vertical ranges make the spread visible at individual dates. Every panel uses a
logarithmic gram axis with one shared range that includes every real weighing shown. A
prediction outside that range is an open triangle at the panel edge pointing outward, with
its value labelled in grams. The calendar axis ends on 31 October 2021; later weighings,
which were not scored, are not shown. Pond colours are fixed across figures. LOPO transfers between ponds over an
overlapping calendar period and does not test forecasting into an unseen future."""


def main() -> dict:
    return p.finish(draw("lopo"), "fig_trajectories", "main", CAPTION)


if __name__ == "__main__":
    main()
