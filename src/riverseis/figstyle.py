"""Shared paper-grade Matplotlib defaults.

Call `paper_style()` at the top of a figure script (after importing pyplot) to
get the group's figure standard: a modern typeface (Space Grotesk, vendored under
`assets/fonts/` so figures rebuild offline), tight bounding boxes (minimal white
space), and publication DPI. Per-figure layout (y-limits that emphasize the
signal, legend placement that avoids the data) stays in each script — this only
sets the global defaults so every figure shares one consistent look.

Typeface: **Source Sans 3** (SIL OFL), a clean humanist sans widely used in
scientific publishing — neutral and legible rather than techy. It has native Greek
and superscripts; DejaVu Sans is kept as a per-glyph fallback for anything it lacks.
We use the *regular* weight with *semibold* for emphasis, never heavy bold.

Base font sizes are set **large** on purpose: these multi-panel composites are
shrunk to journal page width, so a generous in-figure size is what keeps axis
labels and annotations readable on the printed page.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
from matplotlib import font_manager as fm

_FONTDIR = Path(__file__).resolve().parents[2] / "assets" / "fonts"
_FALLBACK = "DejaVu Sans"


def _register_fonts() -> str:
    """Register the vendored Source Sans 3 TTFs; return the family name to use."""
    if not _FONTDIR.exists():
        return _FALLBACK
    fam = _FALLBACK
    for ttf in sorted(_FONTDIR.glob("SourceSans3-*.ttf")):
        try:
            fm.fontManager.addfont(str(ttf))
            fam = fm.FontProperties(fname=str(ttf)).get_name()
        except Exception:                       # noqa: BLE001 — fall back silently
            pass
    return fam


def paper_style() -> None:
    family = _register_fonts()
    mpl.rcParams.update({
        # explicit LIST as font.family enables matplotlib's per-glyph fallback:
        # Source Sans 3 for text, DejaVu Sans for any glyph it lacks, so scientific
        # units/symbols (β, τ, σ, superscript-minus, …) always render.
        "font.family": [family, "DejaVu Sans"],
        "font.sans-serif": [family, "DejaVu Sans"],
        "font.weight": "regular",
        "axes.titleweight": "semibold",
        "axes.labelweight": "regular",
        "figure.titleweight": "semibold",
        # large base sizes -> stay legible after the figure is scaled to page width
        "font.size": 15,
        "axes.titlesize": 16,
        "axes.labelsize": 15,
        "xtick.labelsize": 13.5,
        "ytick.labelsize": 13.5,
        "legend.fontsize": 13,
        "legend.framealpha": 0.9,
        "axes.edgecolor": "#3a3a3a",
        "axes.linewidth": 0.9,
        "figure.dpi": 110,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",      # trim surrounding white space
        "savefig.pad_inches": 0.04,
        "axes.grid": False,
    })
