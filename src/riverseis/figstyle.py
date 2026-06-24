"""Shared paper-grade Matplotlib defaults.

Call `paper_style()` at the top of a figure script (after importing pyplot) to
get the group's figure standard: legible fonts, tight bounding boxes (minimal
white space), and publication DPI. Per-figure layout (y-limits that emphasize
the signal, legend placement that avoids the data) stays in each script — this
only sets the global defaults so every figure shares one consistent look.
"""
from __future__ import annotations

import matplotlib as mpl


def paper_style() -> None:
    mpl.rcParams.update({
        "font.size": 12,
        "axes.titlesize": 12,
        "axes.labelsize": 12,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 10,
        "legend.framealpha": 0.9,
        "figure.dpi": 110,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",      # trim surrounding white space
        "savefig.pad_inches": 0.03,
        "axes.grid": False,
    })
