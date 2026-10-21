"""Shared look of the figures: the reference palette of the dataviz method, light surface."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Categorical slots 1 to 3, in their fixed order.
SERIES = ("#2a78d6", "#eb6834", "#1baf7a")
INK, INK_2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df", "#fcfcfb"


def apply() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": MUTED,
        "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK_2, "text.color": INK,
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    })  # fmt: skip


def clean(ax, grid_axis: str | None = None) -> None:
    """Recessive axes: no ticks marks, no top or right spine, a light grid if asked."""
    ax.tick_params(length=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    if grid_axis:
        ax.grid(axis=grid_axis, color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
