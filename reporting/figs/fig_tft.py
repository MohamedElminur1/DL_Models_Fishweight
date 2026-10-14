"""fig_tft: the temporal fusion transformer as implemented in src/models/tft.py (own redraw).

Run from the repo root: `.venv/bin/python -I reporting/figs/fig_tft.py`

Replaces Figure 8 of the original paper; adapted from Lim et al. (2021) but drawn from our own
code, in the order of TFTRegressor.forward: static variable selection and four static covariate
encoders, temporal variable selection over the six sensors, LSTM encoder started from the static
contexts, gated skip, static enrichment, interpretable multi-head attention with the last real
day as the only query and padded days masked, gated skip, position-wise GRN, gated skip, linear
output. The counts and dates that appear (six sensors, four aggregates, day 0, the target gaps)
are read from configs/phase4.yaml.
Output: figures/main/fig_tft.{png,pdf,svg}
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import yaml  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

from reporting import figstyle as fs  # noqa: E402
from reporting.figs._diagram import Canvas, tint  # noqa: E402

WIDTH = 16.0
MARGIN = 0.15
STATIC_X, STATIC_W = MARGIN, 5.0
MAIN_X, MAIN_W = 6.4, 7.7
MAIN_CX = MAIN_X + MAIN_W / 2
ENCODER_X, ENCODER_W = 1.4, STATIC_X + STATIC_W - 1.4
TRUNK_X = 0.66
SKIP_X = (14.45, 14.8, 15.15)   # loops of the three gated skips, left to right
GAP = 0.32
ATTENTION_GAP = 0.8
SKIP_GAP = 0.46                 # gap above the blocks where a skip loop branches off
INPUTS_TEXT_X = 2.5             # offset of the text in the input block, right of the sequence picture

BLUE, GREEN, ORANGE = fs.OKABE_ITO["blue"], fs.OKABE_ITO["bluish_green"], fs.OKABE_ITO["orange"]
PURPLE, VERMILLION = fs.OKABE_ITO["reddish_purple"], fs.OKABE_ITO["vermillion"]
FILL = {
    "static": tint(BLUE, 0.13),
    "temporal": tint(GREEN, 0.14),
    "sequence": tint(ORANGE, 0.16),
    "attention": tint(PURPLE, 0.16),
    "gate": "#EDEDED",
    "output": tint(VERMILLION, 0.13),
}


def read_facts() -> dict:
    """Counts shown in the figure, from the model configuration."""
    config = yaml.safe_load((fs.REPO / "configs/phase4.yaml").read_text())
    design = config["design"]
    return {"sensors": len(design["sensors"]), "stats": len(design["stats"]), "day0": design["day0"],
            "gaps": " or ".join(str(h * design["horizon_days"]) for h in design["horizons"])}


def main_blocks(f: dict) -> list[dict]:
    """The temporal path from the inputs (first) to the output layer (last)."""
    sensors = {6: "six"}.get(f["sensors"], str(f["sensors"]))
    return [
        {"key": "inputs", "fill": "temporal", "title": "Temporal inputs",
         "body": f"Per sensor and day: {f['stats']} standardized aggregates (mean, SD, minimum, maximum) and a "
                 f"validity mask. Sequences are left-padded; attention masks the padded days.", "min_h": 1.75},
        {"key": "embed", "fill": "temporal", "title": "Sensor embeddings", "body": "one linear layer per sensor"},
        {"key": "vsn", "fill": "temporal", "title": "Temporal variable selection",
         "body": f"Weights the {sensors} sensors on every day; context $c_s$"},
        {"key": "lstm", "fill": "sequence", "title": "LSTM encoder",
         "body": "Initial hidden and cell state from $c_h$ and $c_c$", "min_h": 0.98,
         "extra_gap_before": SKIP_GAP - GAP},
        {"key": "gate1", "fill": "gate", "title": "Gated skip connection",
         "body": "GLU, add and norm; skip from the selected inputs"},
        {"key": "enrich", "fill": "sequence", "title": "Static enrichment", "body": "GRN with context $c_e$",
         "extra_gap_before": SKIP_GAP - GAP},
        {"key": "attn", "fill": "attention", "title": "Interpretable multi-head attention",
         "body": "Query: the last real day. Keys and values: all days, padded days masked. Heads share the values "
                 "and are averaged.", "extra_gap_before": ATTENTION_GAP - GAP},
        {"key": "gate2", "fill": "gate", "title": "Gated skip connection",
         "body": "GLU, add and norm; skip from the attention query"},
        {"key": "ffn", "fill": "sequence", "title": "Position-wise GRN", "body": ""},
        {"key": "gate3", "fill": "gate", "title": "Gated skip connection",
         "body": "GLU, add and norm; skip from the first gated skip, last day"},
        {"key": "out", "fill": "output", "title": "Output layer", "body": "Linear, one value per sequence"},
    ]


def draw_grid(canvas: Canvas, x: float, y: float, rows: int = 6, columns: int = 10, padded: int = 4,
              cell_w: float = 0.2, cell_h: float = 0.16) -> None:
    """Small picture of one input sequence: rows are sensors, columns are days, left cells are padding."""
    for column in range(columns):
        for row in range(rows):
            is_pad = column < padded
            canvas.ax.add_patch(Rectangle(
                (x + column * cell_w, y + row * cell_h), cell_w, cell_h,
                fc="white" if is_pad else tint(GREEN, 0.5), ec="#9A9A9A" if is_pad else "white",
                lw=0.35 if is_pad else 0.5, hatch="//////" if is_pad else None, zorder=3))
    last = x + (columns - 1) * cell_w
    canvas.ax.add_patch(Rectangle((last, y), cell_w, rows * cell_h, fc="none", ec=fs.INK, lw=0.9, zorder=4))
    pad_w, y_rule = padded * cell_w, y - 0.07
    canvas.arrow([(x + 0.02, y_rule), (x + pad_w - 0.03, y_rule)], head=False, color="#9A9A9A", zorder=3.5)
    canvas.arrow([(x + pad_w + 0.03, y_rule), (x + columns * cell_w - 0.02, y_rule)], head=False, zorder=3.5)
    canvas.text(x + pad_w / 2, y - 0.28, "padding", va="center")
    canvas.text(x + pad_w + (columns - padded) * cell_w / 2, y - 0.28, "days", va="center")


def draw(canvas: Canvas, f: dict) -> float:
    """Draw the diagram from the bottom up; return the y of its top edge."""
    blocks = main_blocks(f)
    inputs_w_text = MAIN_W - INPUTS_TEXT_X
    y = MARGIN
    placed: dict[str, dict] = {}
    for block in blocks:
        width = inputs_w_text if block["key"] == "inputs" else MAIN_W
        height = max(canvas.measure(block["title"], block["body"], width), block.get("min_h", 0.0), 0.56)
        y += block.get("extra_gap_before", 0.0)
        block["h"] = height
        block["y"] = y
        placed[block["key"]] = block
        y += height + GAP

    # Main column boxes and the arrows between them.
    boxes = {}
    for block in blocks:
        if block["key"] == "inputs":
            frame = canvas.box(MAIN_X, block["y"], MAIN_W, block["h"], fc=FILL[block["fill"]])
            canvas.box(MAIN_X + INPUTS_TEXT_X, block["y"], inputs_w_text, block["h"], block["title"], block["body"],
                       fc="none", ec="none", align="left", valign="center")
            boxes[block["key"]] = frame
            draw_grid(canvas, MAIN_X + 0.34, block["y"] + 0.62)
        else:
            boxes[block["key"]] = canvas.box(MAIN_X, block["y"], MAIN_W, block["h"], block["title"], block["body"],
                                             fc=FILL[block["fill"]])
    order = [b["key"] for b in blocks]
    for lower, upper in zip(order, order[1:]):
        if upper == "attn":
            continue
        canvas.arrow([(MAIN_CX, boxes[lower].top), (MAIN_CX, boxes[upper].bottom)])

    # Attention inputs: keys and values from all days, the query from the last real day.
    enrich, attn = boxes["enrich"], boxes["attn"]
    y_bus = enrich.top + 0.2
    x_kv, x_q = MAIN_CX - 1.6, MAIN_CX + 1.6
    canvas.arrow([(MAIN_CX, enrich.top), (MAIN_CX, y_bus)], head=False)
    canvas.arrow([(x_kv, y_bus), (x_q, y_bus)], head=False)
    canvas.arrow([(x_kv, y_bus), (x_kv, attn.bottom)])
    canvas.arrow([(x_q, y_bus), (x_q, attn.bottom)])
    canvas.text(x_kv - 0.17, y_bus + 0.2, "keys, values", ha="right", va="baseline")
    canvas.text(x_q + 0.17, y_bus + 0.2, "query", ha="left", va="baseline")

    # Skip connections of the three gated blocks.
    vsn, gate1, lstm = boxes["vsn"], boxes["gate1"], boxes["lstm"]
    y1 = (vsn.top + lstm.bottom) / 2
    canvas.arrow([(MAIN_CX, y1), (SKIP_X[0], y1), (SKIP_X[0], gate1.cy), (gate1.right, gate1.cy)])
    gate2 = boxes["gate2"]
    canvas.arrow([(x_q, y_bus), (SKIP_X[1], y_bus), (SKIP_X[1], gate2.cy), (gate2.right, gate2.cy)])
    gate3 = boxes["gate3"]
    y3 = (gate1.top + enrich.bottom) / 2
    canvas.arrow([(MAIN_CX, y3), (SKIP_X[2], y3), (SKIP_X[2], gate3.cy), (gate3.right, gate3.cy)])

    # Static path: inputs, variable selection, trunk with four encoders on the level of their targets.
    inputs = boxes["inputs"]
    static_inputs = canvas.box(STATIC_X, inputs.bottom, STATIC_W, inputs.h, "Static inputs",
                               f"T1: $t$, days since {f['day0']}. T2: $\\log W_k$, $\\log W_{{k-1}}$, $t_k$, "
                               f"$t_k-t_{{k-1}}$ and the target gap ({f['gaps']} days).",
                               fc=FILL["static"])
    embed = boxes["embed"]
    svsn_h = max(canvas.measure("Static variable selection",
                                "Each scalar embedded by its own linear layer, then weighted by a VSN", STATIC_W), 0.9)
    svsn = canvas.box(STATIC_X, embed.bottom - 0.02, STATIC_W, svsn_h, "Static variable selection",
                      "Each scalar embedded by its own linear layer, then weighted by a VSN", fc=FILL["static"])
    canvas.arrow([(static_inputs.cx, static_inputs.top), (static_inputs.cx, svsn.bottom)])

    enc_h = 0.5
    encoders = {
        "c_s": canvas.box(ENCODER_X, vsn.cy - enc_h / 2, ENCODER_W, enc_h, "", "$c_s$: variable selection",
                          fc=FILL["static"]),
        "c_h": canvas.box(ENCODER_X, lstm.cy + 0.04, ENCODER_W, enc_h - 0.06, "", "$c_h$: LSTM hidden state",
                          fc=FILL["static"]),
        "c_c": canvas.box(ENCODER_X, lstm.cy - 0.04 - (enc_h - 0.06), ENCODER_W, enc_h - 0.06, "",
                          "$c_c$: LSTM cell state", fc=FILL["static"]),
        "c_e": canvas.box(ENCODER_X, enrich.cy - enc_h / 2, ENCODER_W, enc_h, "", "$c_e$: static enrichment",
                          fc=FILL["static"]),
    }
    trunk_top = encoders["c_e"].cy
    canvas.arrow([(TRUNK_X, svsn.top), (TRUNK_X, trunk_top)], head=False)
    for box in encoders.values():
        canvas.arrow([(TRUNK_X, box.cy), (box.left, box.cy)])
    canvas.text(TRUNK_X - 0.3, svsn.top + 0.25, r"$\zeta$", ha="center")
    canvas.text(encoders["c_e"].cx + 0.35, encoders["c_e"].top + 0.28, "Static covariate encoders (four GRNs)",
                bold=True)

    # The contexts enter their blocks from the left.
    canvas.arrow([(encoders["c_s"].right, encoders["c_s"].cy), (vsn.left, encoders["c_s"].cy)])
    canvas.arrow([(encoders["c_h"].right, encoders["c_h"].cy), (lstm.left, encoders["c_h"].cy)])
    canvas.arrow([(encoders["c_c"].right, encoders["c_c"].cy), (lstm.left, encoders["c_c"].cy)])
    canvas.arrow([(encoders["c_e"].right, encoders["c_e"].cy), (enrich.left, encoders["c_e"].cy)])

    # Prediction label above the output layer.
    out = boxes["out"]
    canvas.arrow([(MAIN_CX, out.top), (MAIN_CX, out.top + 0.32)])
    canvas.text(MAIN_CX, out.top + 0.5, "Prediction (standardized target)", bold=True, va="center")
    top = out.top + 0.5 + 0.2

    # Key and scope note in the free corner above the encoders.
    key_h = canvas.measure("Key", "GRN, gated residual network\nVSN, variable selection network\n"
                                  "GLU, gated linear unit\nAdd and norm: layer norm of skip plus gated output",
                           STATIC_W)
    note_h = canvas.measure("Scope", "Encoder only: no decoder and no known future inputs; one point prediction "
                                     "per sequence, trained on mean squared error.", STATIC_W)
    key_y = top - key_h
    canvas.box(STATIC_X, key_y, STATIC_W, key_h, "Key", "GRN, gated residual network\nVSN, variable selection network\n"
               "GLU, gated linear unit\nAdd and norm: layer norm of skip plus gated output", fc="white",
               align="left", valign="top", dashed=True)
    note_y = key_y - 0.3 - note_h
    canvas.box(STATIC_X, note_y, STATIC_W, note_h, "Scope", "Encoder only: no decoder and no known future inputs; "
               "one point prediction per sequence, trained on mean squared error.", fc="white", align="left",
               valign="top", dashed=True)
    return top


def main() -> None:
    facts = read_facts()
    probe = Canvas(WIDTH, 40.0)
    top = draw(probe, facts)
    fs.plt.close(probe.fig)
    canvas = Canvas(WIDTH, top + MARGIN)
    draw(canvas, facts)
    print(f"canvas {WIDTH} x {top + MARGIN:.2f} cm")
    for path in canvas.save("fig_tft", "main", svg=True):
        print(path)


if __name__ == "__main__":
    main()
