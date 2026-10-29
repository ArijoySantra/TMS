"""Matplotlib canvases styled to match the app."""
from __future__ import annotations

import matplotlib

matplotlib.use("QtAgg")
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from traffic_ai.gui.common import BORDER, INK, MUTED  # noqa: E402

matplotlib.rcParams.update({
    "font.size": 9, "axes.edgecolor": BORDER, "axes.labelcolor": MUTED, "xtick.color": MUTED,
    "ytick.color": MUTED, "axes.titlecolor": INK, "axes.titlesize": 10, "axes.titleweight": "semibold",
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": "#edf0f4",
    "grid.linewidth": 0.8, "axes.axisbelow": True, "legend.frameon": False, "figure.facecolor": "white",
})


class Chart(FigureCanvasQTAgg):
    def __init__(self, height: float = 3.0, parent=None):
        self.fig = Figure(figsize=(6, height), dpi=100, layout="constrained")
        super().__init__(self.fig)
        self.setParent(parent)
        self.setMinimumHeight(int(height * 70))

    def axes(self, n: int = 1, **kw):
        self.fig.clear()
        ax = self.fig.subplots(1, n, **kw)
        return ax

    def message(self, text: str):
        self.fig.clear()
        ax = self.fig.add_subplot()
        ax.axis("off")
        ax.text(0.5, 0.5, text, ha="center", va="center", color=MUTED, wrap=True)
        self.draw_idle()
