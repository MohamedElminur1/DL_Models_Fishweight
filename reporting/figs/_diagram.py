"""Drawing primitives for the flow diagrams (fig_workflow, fig_tft).

A Canvas is one full-bleed axes whose data units are centimetres of the final figure, origin at
the bottom left, so a box of width 3 is 3 cm wide on paper and a line of 8 pt text is about
0.34 cm tall. Text is wrapped to the width of its box by measuring it, and `check_fit` stops the
script when any text leaves its box, so overflow is caught at run time and not by eye alone.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.path import Path
from matplotlib.text import Text

from reporting import figstyle as fs

LINE_PT = 0.7
ARROW_STYLE = "-|>,head_length=0.5,head_width=0.22"
FIT_TOLERANCE_CM = 0.02


def tint(hex_color: str, amount: float) -> str:
    """`hex_color` mixed with white; amount 1 keeps the colour, 0 gives white."""
    channels = [int(hex_color[i:i + 2], 16) for i in (1, 3, 5)]
    mixed = [round(255 * (1 - amount) + c * amount) for c in channels]
    return "#" + "".join(f"{c:02X}" for c in mixed)


def no_break(text: str) -> str:
    """Bind a number to the word, unit or percent sign after it with a no-break space."""
    return re.sub(r"(\d) (?=[%A-Za-z(])", "\\1\u00a0", text)


def _protect_math(text: str) -> str:
    """Hide the spaces inside $...$ so that a formula is never split across two lines."""
    return re.sub(r"\$[^$]*\$", lambda m: m.group(0).replace(" ", "\u0001"), text)


@dataclass
class Box:
    """A drawn box in cm; helpers return anchor points for arrows."""

    x: float
    y: float
    w: float
    h: float
    texts: list[Text] = field(default_factory=list)

    @property
    def cx(self) -> float:
        return self.x + self.w / 2

    @property
    def cy(self) -> float:
        return self.y + self.h / 2

    @property
    def left(self) -> float:
        return self.x

    @property
    def right(self) -> float:
        return self.x + self.w

    @property
    def top(self) -> float:
        return self.y + self.h

    @property
    def bottom(self) -> float:
        return self.y


class Canvas:
    """One figure-sized drawing surface; call `save` when done."""

    def __init__(self, width_cm: float, height_cm: float):
        self.width, self.height = width_cm, height_cm
        self.fig = fs.figure(width_cm, height_cm)
        self.fig.set_dpi(fs.DPI)  # text is measured at the resolution of the PNG
        self.ax = self.fig.add_axes((0, 0, 1, 1))
        self.ax.set_xlim(0, width_cm)
        self.ax.set_ylim(0, height_cm)
        self.ax.axis("off")
        self._boxes: list[Box] = []

    # -- measuring ------------------------------------------------------------------------
    def _renderer(self):
        return self.fig.canvas.get_renderer()

    def _size_cm(self, text: Text) -> tuple[float, float]:
        box = text.get_window_extent(self._renderer())
        return box.width / self.fig.dpi * 2.54, box.height / self.fig.dpi * 2.54

    def _width_cm(self, string: str, weight: str, size: float) -> float:
        probe = self.ax.text(0, 0, string, fontsize=size, fontweight=weight)
        width, _ = self._size_cm(probe)
        probe.remove()
        return width

    def wrap(self, string: str, max_width: float, weight: str = "normal", size: float = fs.TEXT_PT) -> str:
        """Break `string` at spaces so that no line is wider than `max_width` cm; keeps "\\n"."""
        lines: list[str] = []
        for paragraph in no_break(string).split("\n"):
            current = ""
            for word in _protect_math(paragraph).split(" "):
                word = word.replace("\u0001", " ")
                trial = f"{current} {word}".strip()
                if current and self._width_cm(trial, weight, size) > max_width:
                    lines.append(current)
                    current = word
                else:
                    current = trial
            lines.append(current)
        return "\n".join(lines)

    def measure(self, title: str, body: str, w: float, *, size: float = fs.TEXT_PT, pad: float = 0.12,
                gap: float = 0.07, pad_v: float = 0.1) -> float:
        """Height in cm a box of width `w` needs for `title` and `body` with `pad_v` above and below."""
        heights = []
        for string, weight in ((title, "bold"), (body, "normal")):
            if string:
                probe = self.ax.text(0, 0, self.wrap(string, w - 2 * pad, weight, size), fontsize=size,
                                     fontweight=weight)
                heights.append(self._size_cm(probe)[1])
                probe.remove()
        return sum(heights) + gap * (len(heights) - 1) + 2 * pad_v

    # -- drawing --------------------------------------------------------------------------
    def box(self, x: float, y: float, w: float, h: float, title: str = "", body: str = "", *,
            fc: str = "white", ec: str = fs.INK, lw: float = 0.6, rounding: float = 0.1, pad: float = 0.12,
            align: str = "center", valign: str = "center", size: float = fs.TEXT_PT, dashed: bool = False,
            zorder: float = 2.0, gap: float = 0.07, pad_v: float = 0.1) -> Box:
        """Rounded box with a bold title and a regular body, both wrapped to the box width."""
        patch = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={rounding}", fc=fc, ec=ec,
                               lw=lw, ls=(0, (3, 2)) if dashed else "-", zorder=zorder)
        self.ax.add_patch(patch)
        box = Box(x, y, w, h)
        inner = w - 2 * pad
        anchor_x = {"center": x + w / 2, "left": x + pad, "right": x + w - pad}[align]
        pieces = []
        if title:
            pieces.append((self.wrap(title, inner, "bold", size), "bold"))
        if body:
            pieces.append((self.wrap(body, inner, "normal", size), "normal"))
        objects = [self.ax.text(anchor_x, 0, s, fontsize=size, fontweight=weight, ha=align, va="top",
                                multialignment=align, zorder=zorder + 0.1)
                   for s, weight in pieces]
        heights = [self._size_cm(t)[1] for t in objects]
        total = sum(heights) + gap * (len(objects) - 1)
        top = y + h / 2 + total / 2 if valign == "center" else y + h - pad_v
        for text, height in zip(objects, heights):
            text.set_position((anchor_x, top))
            top -= height + gap
        box.texts = objects
        self._boxes.append(box)
        return box

    def text(self, x: float, y: float, string: str, *, ha: str = "center", va: str = "center",
             bold: bool = False, size: float = fs.TEXT_PT, color: str = fs.INK, rotation: float = 0,
             zorder: float = 3.0) -> Text:
        """Free text at (x, y) in cm."""
        return self.ax.text(x, y, string, ha=ha, va=va, fontsize=size, fontweight="bold" if bold else "normal",
                            color=color, rotation=rotation, zorder=zorder, multialignment=ha)

    def arrow(self, points: list[tuple[float, float]], *, head: bool = True, lw: float = LINE_PT,
              color: str = fs.INK, dashed: bool = False, zorder: float = 1.5) -> None:
        """Polyline through `points` (cm) with an arrowhead at the last point when `head`."""
        codes = [Path.MOVETO] + [Path.LINETO] * (len(points) - 1)
        patch = FancyArrowPatch(path=Path(points, codes), arrowstyle=ARROW_STYLE if head else "-",
                                mutation_scale=7, lw=lw, color=color, shrinkA=0, shrinkB=0,
                                ls=(0, (3, 2)) if dashed else "-", joinstyle="miter", capstyle="butt",
                                zorder=zorder)
        self.ax.add_patch(patch)

    def bus(self, sources: list[tuple[float, float]], targets: list[tuple[float, float]], y_mid: float,
            *, lw: float = LINE_PT, color: str = fs.INK) -> None:
        """Join `sources` (bottom edge points) to `targets` (top edge points) through a horizontal bus.

        With one source and one target at the same x this is a plain arrow.
        """
        xs = [x for x, _ in sources] + [x for x, _ in targets]
        for x, y in sources:
            self.arrow([(x, y), (x, y_mid)], head=False, lw=lw, color=color)
        if max(xs) - min(xs) > 1e-9:
            self.arrow([(min(xs), y_mid), (max(xs), y_mid)], head=False, lw=lw, color=color)
        for x, y in targets:
            self.arrow([(x, y_mid), (x, y)], lw=lw, color=color)

    # -- checks and output ------------------------------------------------------------------
    def check_fit(self) -> None:
        """Raise when any box text extends past its box, or when two texts of one box overlap."""
        renderer = self._renderer()
        problems = []
        to_cm = 2.54 / self.fig.dpi
        origin = self.ax.transData.transform((0, 0))
        for box in self._boxes:
            for text in box.texts:
                extent = text.get_window_extent(renderer)
                left, right = (extent.x0 - origin[0]) * to_cm, (extent.x1 - origin[0]) * to_cm
                bottom, top = (extent.y0 - origin[1]) * to_cm, (extent.y1 - origin[1]) * to_cm
                if (left < box.left - FIT_TOLERANCE_CM or right > box.right + FIT_TOLERANCE_CM
                        or bottom < box.bottom - FIT_TOLERANCE_CM or top > box.top + FIT_TOLERANCE_CM):
                    problems.append(f"text {text.get_text()[:40]!r} leaves its box at ({box.x:.2f}, {box.y:.2f}): "
                                    f"text x {left:.2f} to {right:.2f}, y {bottom:.2f} to {top:.2f}; "
                                    f"box x {box.left:.2f} to {box.right:.2f}, y {box.bottom:.2f} to {box.top:.2f}")
        if problems:
            raise ValueError("\n".join(problems))

    def save(self, stem: str, group: str = "main", svg: bool = True):
        """Check the fit, then write PNG, PDF and (by default) SVG through figstyle.save."""
        self.check_fit()
        return fs.save(self.fig, stem, group, svg=svg)
