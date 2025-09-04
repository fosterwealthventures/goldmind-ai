
from __future__ import annotations

"""
v2 structural_insights.py
-------------------------
Free "research spine" built on FRED to inform recommendations (NOT a price source).

Metrics:
- inflation_yoy          : CPI YoY (%)
- real_10y_pct           : DGS10 - T10YIE (percentage points)
- dollar_change_3m       : DTWEXBGS 65d % change (%)
- yield_curve_slope      : T10Y2Y (percentage points)
- hy_oas                 : BAMLH0A0HYM2 (bps)
- hy_oas_change_3m       : 65d change in HY OAS (bps)
- wti_change_3m          : DCOILWTICO 65d % change (%)
- stress_level           : STLFSI2 (index level)
- growth_nowcast         : composite of INDPRO YoY and inverted UNRATE z-score (unitless)

Derived advisory:
- macro_tilt.score in [-1, 1] from three pillars:
  * rates  = -z(real_10y_pct)
  * dollar = -z(dollar_change_3m)
  * stress =  z(hy_oas) + z(stress_level)
  Optional add-ons (0.5 weight each): z(wti_change_3m) and 0.5 if curve inverted.

Environment:
- FRED_API_KEY (required)
- WINDOWS (optional): ROLLING_WINDOW_DAYS (default 1095), CHANGE_WINDOW_DAYS (default 65)

This module is intentionally sync; server.py wraps it.
"""

import os
import math
from datetime import date, datetime, timedelta, timezone
from typing import Dict, List, Tuple, Optional

import requests


FRED_API = "https://api.stlouisfed.org/fred/series/observations"

# --- general utils ---

def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _pct_change(new: float, old: float) -> Optional[float]:
    try:
        if old == 0 or not math.isfinite(new) or not math.isfinite(old):
            return None
        return 100.0 * (new / old - 1.0)
    except Exception:
        return None


def _zscore_latest(values: List[Optional[float]], window: int) -> Optional[float]:
    """Z-score of the latest finite value versus last `window` finite observations."""
    xs = [x for x in values[-window:] if x is not None and math.isfinite(x)]
    if len(xs) < 10:
        return None
    mean = sum(xs) / len(xs)
    var = sum((x - mean) ** 2 for x in xs) / max(1, len(xs) - 1)
    sd = math.sqrt(var) if var > 0 else 0.0
    if sd == 0.0:
        return None
    latest = next((x for x in reversed(values) if x is not None and math.isfinite(x)), None)
    if latest is None:
        return None
    return (latest - mean) / sd


def _fred_series(series_id: str, api_key: str, start: Optional[str]) -> List[Tuple[str, float]]:
    params = {"series_id": series_id, "api_key": api_key, "file_type": "json"}
    if start:
        params["observation_start"] = start
    r = requests.get(FRED_API, params=params, timeout=20)
    r.raise_for_status()
    obs = r.json().get("observations", [])
    out: List[Tuple[str, float]] = []
    for o in obs:
        v = o.get("value")
        if v not in (None, "", "."):
            try:
                out.append((o["date"], float(v)))
            except Exception:
                pass
    return out


def _daily_forward_fill(obs: List[Tuple[str, float]]) -> Tuple[List[str], List[Optional[float]]]:
    """Expand (date,value) to daily calendar with forward-fill."""
    if not obs:
        return [], []
    try:
        start = date.fromisoformat(obs[0][0])
        end = date.fromisoformat(obs[-1][0])
    except Exception:
        return [d for d, _ in obs], [v for _, v in obs]
    dmap = {d: v for d, v in obs}
    cur = start
    val = dmap.get(start)
    labels: List[str] = []
    values: List[Optional[float]] = []
    while cur <= end:
        ds = cur.isoformat()
        if ds in dmap:
            val = dmap[ds]
        labels.append(ds)
        values.append(val)
        cur += timedelta(days=1)
    return labels, values


def _percent_change_over(values: List[Optional[float]], lag_days: int) -> Optional[float]:
    """Compute % change between last value and value lag_days before within the same list."""
    if not values or len(values) <= lag_days:
        return None
    latest = values[-1]
    past = values[-(lag_days + 1)]
    if latest is None or past is None:
        return None
    return _pct_change(latest, past)


# --- main snapshot ---

def snapshot() -> Dict[str, object]:
    api_key = os.getenv("FRED_API_KEY")
    if not api_key:
        return {
            "source": "FRED",
            "ok": False,
            "errors": ["Missing FRED_API_KEY"],
            "headline": "Macro snapshot unavailable",
            "as_of": _now_iso(),
            # backward-compat fields
            "inflation_yoy": None,
            "real_10y_pct": None,
            "dollar_change_3m": None,
            "gold_trend_6m_slope": None,
            "gold_change_6m": None,
        }

    # windows
    ROLL = int(os.getenv("ROLLING_WINDOW_DAYS", "1095"))  # ~3y
    LAG = int(os.getenv("CHANGE_WINDOW_DAYS", "65"))      # ~3 months of trading days

    today = date.today()
    start_3y = today.replace(year=today.year - 3).isoformat()
    start_5y = today.replace(year=today.year - 5).isoformat()
    start_2y = today.replace(year=today.year - 2).isoformat()
    errors: List[str] = []

    # Fetch series
    try:
        cpi = _fred_series("CPIAUCSL", api_key, start_5y)           # monthly
        dgs10 = _fred_series("DGS10", api_key, start_5y)            # daily
        t10yie = _fred_series("T10YIE", api_key, start_5y)          # daily
        t10y2y = _fred_series("T10Y2Y", api_key, start_5y)          # daily
        dxy = _fred_series("DTWEXBGS", api_key, start_5y)           # daily
        hy = _fred_series("BAMLH0A0HYM2", api_key, start_5y)        # daily (bps)
        wti = _fred_series("DCOILWTICO", api_key, start_5y)         # daily (USD/bbl)
        stl = _fred_series("STLFSI2", api_key, start_5y)            # weekly (index)
        indpro = _fred_series("INDPRO", api_key, start_5y)          # monthly index
        unrate = _fred_series("UNRATE", api_key, start_5y)          # monthly %
    except Exception as e:
        return {
            "source": "FRED",
            "ok": False,
            "errors": [f"{type(e).__name__}: {e}"],
            "headline": "Macro snapshot unavailable",
            "as_of": _now_iso(),
            "inflation_yoy": None,
            "real_10y_pct": None,
            "dollar_change_3m": None,
            "gold_trend_6m_slope": None,
            "gold_change_6m": None,
        }

    # --- Derived metrics ---

    # Inflation YoY (%)
    inflation_yoy = None
    try:
        if len(cpi) >= 13:
            latest = cpi[-1][1]
            prev12 = cpi[-13][1]
            inflation_yoy = _pct_change(latest, prev12)
    except Exception:
        errors.append("cpi_yoy_failed")

    # Real 10y (pp) from daily
    real_10y_pct = None
    try:
        # align by last available reading (just take last of each; daily series are near-synchronous)
        if dgs10 and t10yie:
            real_10y_pct = dgs10[-1][1] - t10yie[-1][1]
    except Exception:
        errors.append("real10y_failed")

    # Dollar 3m change (%)
    dollar_change_3m = None
    try:
        dxy_labels, dxy_vals = _daily_forward_fill(dxy)
        dollar_change_3m = _percent_change_over(dxy_vals, LAG)
    except Exception:
        errors.append("dollar3m_failed")

    # Yield curve slope (pp)
    yield_curve_slope = None
    try:
        if t10y2y:
            yield_curve_slope = t10y2y[-1][1]
    except Exception:
        errors.append("curve_failed")

    # HY OAS level (bps) and 3m change
    hy_oas = None
    hy_oas_change_3m = None
    try:
        hy_labels, hy_vals = _daily_forward_fill(hy)
        hy_oas = hy_vals[-1] if hy_vals else None
        if isinstance(hy_oas, float):
            # 3m change in bps (level difference)
            past = hy_vals[-(LAG + 1)] if len(hy_vals) > LAG else None
            if past is not None:
                hy_oas_change_3m = hy_oas - past
    except Exception:
        errors.append("hy_failed")

    # WTI 3m change (%)
    wti_change_3m = None
    try:
        wti_labels, wti_vals = _daily_forward_fill(wti)
        wti_change_3m = _percent_change_over(wti_vals, LAG)
    except Exception:
        errors.append("wti_failed")

    # Stress level
    stress_level = None
    try:
        stl_labels, stl_vals = _daily_forward_fill(stl)
        stress_level = stl_vals[-1] if stl_vals else None
    except Exception:
        errors.append("stress_failed")

    # Growth nowcast: z(INDPRO YoY) - z(UNRATE)
    growth_nowcast = None
    try:
        indpro_yoy = None
        if len(indpro) >= 13:
            indpro_yoy = _pct_change(indpro[-1][1], indpro[-13][1])
        # Build comparable daily-ish vectors by forward fill on monthly
        ind_labels, ind_vals = _daily_forward_fill(indpro)
        un_labels, un_vals = _daily_forward_fill(unrate)
        # Compute last-3y z-scores for latest values
        z_ind = _zscore_latest(ind_vals, ROLL) if ind_vals else None
        z_un = _zscore_latest(un_vals, ROLL) if un_vals else None
        if z_ind is not None and z_un is not None:
            growth_nowcast = z_ind - z_un  # higher indpro, lower unemployment -> stronger growth
    except Exception:
        errors.append("growth_failed")

    # --- Build z-scores for advisory pillars ---
    z_real = None
    z_dollar3m = None
    z_hy = None
    z_stress = None
    z_wti3m = None

    try:
        # Build histories for z calculations
        # Real rate history
        if dgs10 and t10yie:
            # daily align by calendar
            d10_lab, d10_val = _daily_forward_fill(dgs10)
            e10_lab, e10_val = _daily_forward_fill(t10yie)
            rr_vals: List[Optional[float]] = []
            for x, y in zip(d10_val, e10_val):
                if x is None or y is None:
                    rr_vals.append(None)
                else:
                    rr_vals.append(x - y)
            z_real = _zscore_latest(rr_vals, ROLL)

        # Dollar 3m change history
        if dxy:
            dxy_lab, dxy_val = _daily_forward_fill(dxy)
            # construct 3m change vector
            chg_vec: List[Optional[float]] = []
            for i in range(len(dxy_val)):
                if i < LAG or dxy_val[i] is None or dxy_val[i - LAG] is None:
                    chg_vec.append(None)
                else:
                    chg_vec.append(_pct_change(dxy_val[i], dxy_val[i - LAG]))
            z_dollar3m = _zscore_latest(chg_vec, ROLL)

        # HY OAS z
        if hy:
            _, hy_val = _daily_forward_fill(hy)
            z_hy = _zscore_latest(hy_val, ROLL)

        # Stress z
        if stl:
            _, stl_val = _daily_forward_fill(stl)
            z_stress = _zscore_latest(stl_val, ROLL)

        # WTI 3m change z
        if wti:
            _, wti_val = _daily_forward_fill(wti)
            chg_vec2: List[Optional[float]] = []
            for i in range(len(wti_val)):
                if i < LAG or wti_val[i] is None or wti_val[i - LAG] is None:
                    chg_vec2.append(None)
                else:
                    chg_vec2.append(_pct_change(wti_val[i], wti_val[i - LAG]))
            z_wti3m = _zscore_latest(chg_vec2, ROLL)
    except Exception:
        errors.append("zscore_failed")

    # --- Macro tilt score ---
    # Pillars: rates, dollar, stress. Optional add-ons: wti pulse (+0.5 * z_wti3m) and curve inversion flag (+0.5)
    pillars: Dict[str, Optional[float]] = {}
    extras: Dict[str, Optional[float]] = {}

    rates = (-z_real) if (z_real is not None) else None
    dollar = (-z_dollar3m) if (z_dollar3m is not None) else None
    stress = None
    if z_hy is not None and z_stress is not None:
        stress = z_hy + z_stress
    elif z_hy is not None:
        stress = z_hy
    elif z_stress is not None:
        stress = z_stress

    pillars["rates"] = rates
    pillars["dollar"] = dollar
    pillars["stress"] = stress

    add = 0.0
    add_count = 0
    if z_wti3m is not None:
        extras["wti"] = 0.5 * z_wti3m
        add += extras["wti"]
        add_count += 1
    else:
        extras["wti"] = None

    curve_bonus = 0.5 if (yield_curve_slope is not None and yield_curve_slope < 0.0) else 0.0
    extras["curve_inversion_flag"] = curve_bonus
    add += curve_bonus
    add_count += 1 if curve_bonus != 0.0 else 0

    # Average the available pillars
    base_vals = [x for x in (rates, dollar, stress) if x is not None]
    base = sum(base_vals) / len(base_vals) if base_vals else None
    score = None
    if base is not None:
        score = base + add  # extras are small tilts
        # Bound it softly to [-1, 1] for UI
        score = max(-1.0, min(1.0, score))

    # Headline
    parts = []
    if inflation_yoy is not None:
        parts.append(f"Inflation {inflation_yoy:+.1f}% YoY")
    if real_10y_pct is not None:
        parts.append(f"real 10y {real_10y_pct:+.2f}%")
    if dollar_change_3m is not None:
        parts.append(f"dollar 3m {dollar_change_3m:+.1f}%")
    headline = "; ".join(parts) if parts else "Macro snapshot"

    return {
        "source": "FRED",
        "ok": True,
        "errors": errors,
        "inflation_yoy": inflation_yoy,
        "real_10y_pct": real_10y_pct,
        "dollar_change_3m": dollar_change_3m,
        "yield_curve_slope": yield_curve_slope,
        "hy_oas": hy_oas,
        "hy_oas_change_3m": hy_oas_change_3m,
        "wti_change_3m": wti_change_3m,
        "stress_level": stress_level,
        "growth_nowcast": growth_nowcast,
        # Back-compat placeholders for API shape
        "gold_trend_6m_slope": None,
        "gold_change_6m": None,
        # Advisory
        "scores": {
            "z_real_10y": z_real,
            "z_dollar_3m": z_dollar3m,
            "z_hy_oas": z_hy,
            "z_stress": z_stress,
            "z_wti_3m": z_wti3m,
        },
        "macro_tilt": {
            "score": score,
            "pillars": pillars,
            "extras": extras,
        },
        "headline": headline,
        "as_of": _now_iso(),
    }
