"""Shared style for every figure in the dissertation.

Imported by each figure script so that all figures share one typeface,
one line weight and one page width. Figures are drawn at the width they
occupy on the page rather than drawn large and scaled down in Word,
which is what makes axis labels unreadable in print.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Text block of the Stirling A4 template is about 6.3 inches wide.
WIDTH_IN = 8.0
HEIGHT_IN = 4.0

# Colour-blind safe. Distinguishable in greyscale by linestyle, since the
# examiner may print the dissertation.
BLUE = "#0072B2"
VERMILLION = "#D55E00"
GREEN = "#009E73"
GREY = "#6E6E6E"
LIGHT = "#BFBFBF"

RC = {
    "figure.figsize": (WIDTH_IN, HEIGHT_IN),
    "figure.dpi": 200,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "font.size": 9,
    "axes.titlesize": 9,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "legend.frameon": False,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": "#444444",
    "axes.linewidth": 0.8,
    "xtick.color": "#444444",
    "ytick.color": "#444444",
    "xtick.direction": "out",
    "ytick.direction": "out",
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.color": "#DDDDDD",
    "grid.linewidth": 0.6,
    "lines.linewidth": 1.6,
    "lines.solid_capstyle": "round",
}


def apply() -> None:
    plt.rcParams.update(RC)


def save(fig, stem: str, outdir: str = "data/figures") -> None:
    """Write PNG for Word and PDF for anything vector."""
    from pathlib import Path

    root = Path(outdir)
    root.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        path = root / f"{stem}.{suffix}"
        fig.savefig(path)
        print(f"written to {path}")