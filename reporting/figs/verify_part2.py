"""Check the figure artifacts, source alignment, neural seeds and chosen learning curves.

Run from the repository root with:
env PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/tmp/mplconfig_figs .venv/bin/python -I reporting/figs/verify_part2.py.
This verifies saved-data alignment and output formats; it does not replace visual inspection.
"""

import ast
import itertools
import subprocess
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from PIL import Image
import numpy as np
from scipy.stats import rankdata

from reporting.figs import _part2 as p
from reporting.figs.fig_confirmatory import pond_differences
from reporting.figs.fig_learning_curves import selected_curves
from reporting.figs.run_part2 import FIGURES, FIGURE_SCRIPTS


def main() -> None:
    scores = p.csv("results/metrics/scores.csv")
    ponds = p.csv("results/metrics/per_pond.csv")
    tests = p.csv("results/stats/confirmatory.csv")
    for row in tests.itertuples():
        differences = pond_differences(ponds, row.task, int(row.horizon)).to_numpy()
        assert len(differences) == row.n_ponds == 7
        assert int(np.sum(differences < 0)) == row.n_lower
        nonzero = differences[differences != 0]
        ranks = rankdata(np.abs(nonzero), method="average")
        observed = ranks[nonzero > 0].sum()
        sums = [np.dot(signs, ranks) for signs in itertools.product((0, 1), repeat=len(ranks))]
        p_lower = float(np.mean(np.asarray(sums) <= observed))
        assert p_lower == row.p_one_sided_lower
        print(f"PASS {p.horizon_name(row.task, int(row.horizon))}: confirmatory pond dots reproduce "
              f"n_lower = {row.n_lower} of 7, exact one-sided p = {p_lower:g}")
    for task, horizon in p.STRATA:
        expected = {model for group in p.GROUPS for model in p.CFG["models"][task][group]}
        saved = p.part(scores, task, horizon)
        plotted = saved[saved.model.isin(expected)]
        missing = expected - set(plotted.model)
        assert missing == ({"ridge_pw_wq"} if task == "T2" else set())
        perpond = p.part(ponds, task, horizon)
        for model in plotted.model:
            own = perpond[perpond.model == model]
            assert set(own.pond) == {1, 2, 3, 4, 6, 7, 9}
            assert not own.pond.duplicated().any()
        print(f"PASS {p.horizon_name(task, horizon)}: all required saved model rows, seven ponds each")
    for task in ("T1", "T2"):
        for scheme in ("lopo", "chrono"):
            reference = None
            for model in p.shown(task):
                frame = p.prediction(task, scheme, model)
                if frame is None:
                    continue
                summary = p.seed_predictions(frame, task, model)
                assert summary.w_pred.gt(0).all()
                ids = set(summary.row_id)
                assert reference is None or reference == ids
                reference = ids
        for model in p.CFG["models"][task]["deep"] + ["mlp_static"]:
            inner, final, chosen = selected_curves(task, model)
            assert len(chosen) == 7
            assert inner.groupby(["outer_fold", "inner_fold", "seed"]).ngroups == 21
            assert final.groupby(["outer_fold", "seed"]).ngroups == 35
        print(f"PASS {task}: shown models share sample IDs and expected seeds; all five neural models have complete selected curves")
    captions = (p.ROOT / "figures/CAPTIONS.md").read_text()
    for stem, group in FIGURES:
        script = p.ROOT / f"reporting/figs/{FIGURE_SCRIPTS.get(stem, stem)}.py"
        ast.parse(script.read_text())
        png = p.ROOT / f"figures/{group}/{stem}.png"
        pdf = png.with_suffix(".pdf")
        with Image.open(png) as image:
            assert all(abs(dpi - p.fs.DPI) < 0.1 for dpi in image.info["dpi"])
            assert abs(image.width / p.fs.DPI * 2.54 - p.fs.FULL_CM) < 0.01
            if stem in ("fig_pond_scores", "fig_pond_scores_t2"):
                height_cm = 13 if stem == "fig_pond_scores" else 11
                assert abs(image.height / p.fs.DPI * 2.54 - height_cm) < 0.01
            image.verify()
        info = subprocess.run(["pdfinfo", str(pdf)], check=True, text=True, capture_output=True).stdout
        assert "Pages:           1" in info
        text = subprocess.run(["pdftotext", str(pdf), "-"], check=True, text=True, capture_output=True).stdout
        assert text.strip() and "\ufffd" not in text
        if stem == "fig_pond_scores":
            assert "T2" not in text
            assert "Linear below 0.5; logarithmic above (line at 0.5)" in text
            assert "Model S" in text
        elif stem == "fig_pond_scores_t2":
            assert "(a) T2, 14 days ahead" in text and "(b) T2, 28 days ahead" in text
            assert text.count("stopped (plan 9.0)") == 2
            for group_name in (p.group_name(group) for group in p.GROUPS
                               if p.CFG["models"]["T2"][group]):
                assert text.count(group_name) == 1
            for model in (model for group in p.GROUPS for model in p.CFG["models"]["T2"][group]):
                assert sum(line.startswith(p.model_name(model)) for line in text.splitlines()) == 1
            assert text.count("Pond 1") == 1 and text.count("Model S") == 1
        assert captions.count(f"## {stem}\n") == 1, f"Missing or duplicated caption: {stem}"
        print(f"PASS {stem}: parseable script, 16 cm / 600 dpi PNG, one-page PDF with text, caption")
    print("LIMITATION T2 Ridge (past weights + WQ): LOPO and permutation outputs absent after plan 9.0 stop")


if __name__ == "__main__":
    main()
