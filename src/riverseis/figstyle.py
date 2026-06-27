"""Shared paper-grade Matplotlib defaults.

Call `paper_style()` at the top of a figure script (after importing pyplot) to
get the group's figure standard: a modern typeface (Space Grotesk, vendored under
`assets/fonts/` so figures rebuild offline), tight bounding boxes (minimal white
space), and publication DPI. Per-figure layout (y-limits that emphasize the
signal, legend placement that avoids the data) stays in each script — this only
sets the global defaults so every figure shares one consistent look.

Typeface: **Space Grotesk** (SIL OFL), a slick geometric sans. We deliberately
lean on the *regular* weight and use *medium* for emphasis instead of heavy bold,
which reads cleaner and more contemporary in dense scientific panels.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
from matplotlib import font_manager as fm

_FONTDIR = Path(__file__).resolve().parents[2] / "assets" / "fonts"
_FALLBACK = "DejaVu Sans"


def _register_fonts() -> str:
    """Register the vendored Space Grotesk TTFs; return the family name to use."""
    if not _FONTDIR.exists():
        return _FALLBACK
    fam = _FALLBACK
    for ttf in sorted(_FONTDIR.glob("SpaceGrotesk-*.ttf")):
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
        # Space Grotesk for text, DejaVu Sans for the glyphs it lacks (β, τ, σ,
        # superscript-minus, …) so scientific units/symbols render correctly.
        "font.family": [family, "DejaVu Sans"],
        "font.sans-serif": [family, "DejaVu Sans"],
        # lean on regular weight; medium for emphasis, never heavy bold
        "font.weight": "regular",
        "axes.titleweight": "medium",
        "axes.labelweight": "regular",
        "figure.titleweight": "medium",
        "font.size": 12,
        "axes.titlesize": 12,
        "axes.labelsize": 12,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 10,
        "legend.framealpha": 0.9,
        "axes.edgecolor": "#3a3a3a",
        "axes.linewidth": 0.9,
        "figure.dpi": 110,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",      # trim surrounding white space
        "savefig.pad_inches": 0.03,
        "axes.grid": False,
    })
