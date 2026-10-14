"""Saved-data readers and drawing conventions shared by the phase 6 part 2 figures."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
import yaml
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, NullFormatter, NullLocator

from reporting import figstyle as fs
from reporting.names import group_name, horizon_name, model_name

ROOT = fs.REPO
CONFIG_PATH = ROOT / "configs/phase4.yaml"
CFG = yaml.safe_load(CONFIG_PATH.read_text())
STRATA = (("T1", 0), ("T2", 1), ("T2", 2))
GROUPS = ("time_only", "water_quality", "deep", "static_only", "reference")
MARKERS = ("o", "s", "^", "D", "v", "P")
STYLES = ("-", "--", "-.", ":", (0, (5, 1, 1, 1)))
SOURCES: set[str] = set()


def csv(relative: str, **kwargs) -> pd.DataFrame:
    """Read a saved source and register its exact path for the caption."""
    if not relative.startswith(("results/", "data/processed/")) or "posthoc" in relative:
        raise ValueError(f"Not a permitted plotting source: {relative}")
    SOURCES.add(relative)
    return pd.read_csv(ROOT / relative, **kwargs)


def selection(task: str) -> dict:
    """Read the committed inner-validation selection, never choose by plotted scores."""
    relative = f"results/selection/primary_{task}.json"
    SOURCES.add(relative)
    return json.loads((ROOT / relative).read_text())


def shown(task: str) -> list[str]:
    """The fixed shown-model list, retaining a stopped model's position."""
    selected = selection(task)
    models = [selected["primary_baseline"], selected["primary_deep"],
              *CFG["permutation"]["tabular_models"][task], "mlp_static"]
    if task == "T2":
        models.append("persistence")
    return list(dict.fromkeys(models))


def part(frame: pd.DataFrame, task: str, horizon: int, scheme: str = "lopo") -> pd.DataFrame:
    """Select exactly the eligible task-horizon, excluding all-event and pooled horizons."""
    return frame[(frame.task == task) & (frame.scheme == scheme)
                 & (frame.sample_set == "eligible") & (frame.horizon.astype(str) == str(horizon))]


def prediction(task: str, scheme: str, model: str) -> pd.DataFrame | None:
    """Return eligible predictions, or None only for the documented stopped T2 ridge."""
    relative = f"results/predictions/{task}/{scheme}/{model}.csv"
    if not (ROOT / relative).exists():
        if (task, scheme, model) != ("T2", "lopo", "ridge_pw_wq"):
            raise FileNotFoundError(relative)
        complete = csv("results/metrics/completeness.csv")
        row = complete[(complete.task == task) & (complete.scheme == scheme)
                       & (complete.model == model) & (complete.sample_set == "eligible")]
        assert len(row) == 1 and not bool(row.iloc[0].has_predictions)
        return None
    frame = csv(relative, parse_dates=["date"])
    assert set(frame.sample_set) == {"eligible"}
    assert set(frame.model) == {model}
    assert np.isfinite(frame[["w_true", "w_pred"]]).all().all()
    assert (frame.w_true > 0).all()
    return frame


def seed_predictions(frame: pd.DataFrame, task: str, model: str) -> pd.DataFrame:
    """Summarize saved predictions per sample; neural means and ranges are captioned."""
    neural = model in CFG["models"][task]["deep"] + CFG["models"][task]["static_only"]
    expected = CFG["seeds"]["final"] if neural else [0]
    assert sorted(frame.seed.unique()) == expected
    assert not frame.duplicated(["row_id", "seed"]).any()
    assert (frame.groupby("row_id").seed.nunique() == len(expected)).all()
    assert (frame.groupby("row_id").w_true.nunique() == 1).all()
    return frame.groupby(["row_id", "pond", "horizon", "date"], as_index=False).agg(
        w_true=("w_true", "first"), w_pred=("w_pred", "mean"),
        low=("w_pred", "min"), high=("w_pred", "max"))


def pond_handles() -> list[Line2D]:
    return [Line2D([], [], marker="o", ls="none", ms=4, mfc=color, mec=fs.EDGE,
                   mew=0.4, label=f"Pond {pond}")
            for pond, color in fs.POND_COLORS.items() if pond != 8]


def model_handles(task: str, stopped: bool = False) -> list[Line2D]:
    handles = []
    for i, model in enumerate(shown(task)):
        label = model_name(model)
        if stopped and model == "ridge_pw_wq":
            handles.append(Line2D([], [], color="none", label=label + ": stopped (plan 9.0)"))
        else:
            handles.append(Line2D([], [], color=fs.INK, marker=MARKERS[i], ls="none",
                                  mfc="white", mew=0.7, ms=4, label=label))
    return handles


def log_weight_axis(ax, which: str = "y") -> None:
    """Label gram values directly, rather than as powers of ten."""
    getattr(ax, f"set_{which}scale")("log")
    axis = getattr(ax, f"{which}axis")
    axis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
    axis.set_minor_formatter(NullFormatter())


def score_axis(ax, values: list[float]) -> None:
    """Keep small errors legible while preserving every large saved S value."""
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    high = float(finite.max()) * 1.12 if finite.size else 1.0
    if high > 2:
        ax.set_xscale("symlog", linthresh=0.5, linscale=1.0)
        ticks = [v for v in (0, 0.25, 0.5, 1, 2, 5, 10, 20, 50, 100) if v < high]
        ax.set_xticks(ticks, [f"{v:g}" for v in ticks])
        ax.xaxis.set_minor_locator(NullLocator())
        ax.axvline(0.5, color=fs.GAP_COLOR, lw=0.6, zorder=1)
    ax.set_xlim(0, high)
    ax.grid(axis="x", color=fs.GRID_COLOR, lw=0.45, zorder=0)
    ax.set_xlabel("S (mean absolute log error)" +
                  ("\nLinear below 0.5; logarithmic above (line at 0.5)" if high > 2 else ""))


def finish(fig, stem: str, group: str, caption: str) -> dict:
    """Save the two publication formats and return the caption's source manifest."""
    sources = sorted(SOURCES)
    paths = fs.save(fig, stem, group)
    for path in paths:
        print(path.relative_to(ROOT), flush=True)
    return {"stem": stem, "group": group, "caption": caption, "sources": sources}
