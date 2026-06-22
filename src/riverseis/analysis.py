"""Core analysis for seismic power vs discharge scaling and hysteresis.

Paper-grade, reusable functions used by the figure workflow. All scaling fits
are done in log10-log10 space on robustly-cleaned, time-aligned proxy/discharge
pairs. The key diagnostic is the exponent b in P ∝ Q^b and its dependence on
frequency band; b ≈ 0.9–1.4 is the turbulent-flow baseline (Gimbert et al. 2014),
and b rising above it (especially in higher-frequency bands) is the bedload
signature (Tsai et al. 2012; Bakker et al. 2020).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

# Turbulent-flow (water) discharge-scaling baseline from Gimbert et al. (2014):
# P_water ∝ H^(7/3); with at-a-station hydraulic geometry H ∝ Q^(0.3..0.6) this
# gives b ≈ 0.9–1.4. Use the band as the "no-bedload" reference.
WATER_BASELINE = (0.9, 1.4)


@dataclass
class ScalingFit:
    station: str
    band: tuple[float, float]
    n: int
    r: float
    b_ols: float
    b_theilsen: float
    b_lo: float          # 2.5th pct bootstrap
    b_hi: float          # 97.5th pct bootstrap
    intercept: float
    excess: float = field(default=np.nan)  # b - water baseline upper (bedload excess)


def load_timeseries(path: str | Path) -> pd.DataFrame:
    """Load a *_timeseries.csv (proxy, gauge[, gauge_shifted]) -> aligned frame."""
    df = pd.read_csv(path, parse_dates=["time_utc"]).set_index("time_utc")
    df.index = pd.to_datetime(df.index, utc=True, errors="coerce")
    df = df[df.index.notna()]
    p = pd.to_numeric(df["proxy"], errors="coerce")
    g = pd.to_numeric(df["gauge"], errors="coerce")
    j = pd.concat([p.rename("P"), g.rename("Q")], axis=1).sort_index()
    j["Q"] = j["Q"].interpolate(method="linear", limit=12)
    return j.dropna()


def clean_loglog(j: pd.DataFrame, n_mad: float = 6.0) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Return (logQ, logP, kept_frame) after robust MAD outlier rejection in logP."""
    lp = np.log10(j["P"].clip(lower=1e-30))
    lq = np.log10(j["Q"].clip(lower=1e-6))
    med = lp.median()
    mad = 1.4826 * (lp - med).abs().median()
    keep = (lp - med).abs() < n_mad * max(mad, 1e-12)
    return lq[keep].values, lp[keep].values, j[keep]


def _theilsen(x: np.ndarray, y: np.ndarray, n_sub: int = 400, seed: int = 0) -> float:
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(x), size=min(len(x), n_sub), replace=False)
    xs, ys = x[idx], y[idx]
    slopes = []
    for i in range(len(xs)):
        dx = xs[i + 1:] - xs[i]
        ok = dx != 0
        slopes.extend(((ys[i + 1:] - ys[i])[ok] / dx[ok]).tolist())
    return float(np.median(slopes)) if slopes else float("nan")


def fit_scaling(j: pd.DataFrame, station: str, band: tuple[float, float],
                n_boot: int = 500, seed: int = 0) -> ScalingFit:
    """Robust log-log power-law fit P ∝ Q^b with bootstrap CI on the slope."""
    x, y, _ = clean_loglog(j)
    if len(x) < 30:
        return ScalingFit(station, band, len(x), np.nan, np.nan, np.nan, np.nan, np.nan, np.nan)
    b_ols, a = np.polyfit(x, y, 1)
    r = float(np.corrcoef(x, y)[0, 1])
    b_ts = _theilsen(x, y, seed=seed)
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        s = rng.integers(0, len(x), len(x))
        boots.append(np.polyfit(x[s], y[s], 1)[0])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return ScalingFit(station, band, len(x), r, float(b_ols), b_ts,
                      float(lo), float(hi), float(a),
                      excess=float(b_ols - WATER_BASELINE[1]))


def lawler_hysteresis_index(q: np.ndarray, p_log: np.ndarray, q_norm: float = 0.5) -> float:
    """Lawler (1989) hysteresis index at normalized discharge q_norm.

    Splits the series at peak Q into rising/falling limbs, interpolates the
    (log) seismic power on each at the same normalized discharge, and returns
    (P_rise - P_fall)/max. Positive => clockwise (more power on the rising limb).
    """
    q = np.asarray(q, float)
    p_log = np.asarray(p_log, float)
    if len(q) < 8:
        return float("nan")
    qn = (q - q.min()) / (q.max() - q.min() + 1e-12)
    pk = int(np.argmax(q))
    def _interp(sl):
        xq, yp = qn[sl], p_log[sl]
        o = np.argsort(xq); xq, yp = xq[o], yp[o]
        if len(np.unique(xq)) < 2:
            return np.nan
        return float(np.interp(q_norm, xq, yp))
    pr, pf = _interp(slice(0, pk + 1)), _interp(slice(pk, len(q)))
    if not (np.isfinite(pr) and np.isfinite(pf)):
        return float("nan")
    return (pr - pf) / max(abs(pr), abs(pf), 1e-12)


def event_window(j: pd.DataFrame, days: float = 3.0) -> pd.DataFrame:
    """Slice ±`days` around the discharge peak (for per-event hysteresis)."""
    qpk = j["Q"].idxmax()
    return j.loc[qpk - pd.Timedelta(days=days): qpk + pd.Timedelta(days=days)]
