# api/app/macro_insights.py
from __future__ import annotations
import os, asyncio, datetime as dt
from typing import Dict, Any, List, Tuple, Optional
import aiohttp
import pandas as pd
import numpy as np
from cachetools import TTLCache

FRED_BASE = "https://api.stlouisfed.org/fred/series/observations"
FRED_KEY  = os.getenv("FRED_API_KEY", "").strip()

# TTL: insights okay to refresh every 3 hours (adjust as you like)
_CACHE = TTLCache(maxsize=32, ttl=3 * 60 * 60)

SERIES = {
    "USD_Broad": "DTWEXBGS",   # Broad Dollar Index
    "UST10Y":    "DGS10",      # 10y nominal yield
    "INF_EXP":   "T10YIE",     # 10y breakeven inflation expectations
    "CPI_Core":  "CPILFESL",   # Core CPI (index)
    "TIPS10Y":   "DFII10",     # 10y TIPS yield (real)
}

def _fred_url(series_id: str, start: str) -> str:
    return (
        f"{FRED_BASE}?series_id={series_id}"
        f"&api_key={FRED_KEY}&file_type=json&observation_start={start}"
    )

async def _fetch_series(session: aiohttp.ClientSession, series_id: str, start: str) -> pd.Series:
    url = _fred_url(series_id, start)
    async with session.get(url, timeout=30) as r:
        js = await r.json()
    obs = js.get("observations", [])
    if not obs:
        return pd.Series(dtype=float, name=series_id)
    # FRED returns strings; convert
    df = pd.DataFrame(obs)
    df = df[df["value"] != "."]
    s = pd.to_numeric(df["value"], errors="coerce")
    s.index = pd.to_datetime(df["date"])
    s.name = series_id
    s = s.asfreq("D").ffill()   # daily fill-forward
    return s

async def fetch_macro() -> Dict[str, pd.Series]:
    """
    Pulls the macro series in parallel. Caches results for the TTL above.
    """
    cache_key = "macro:raw"
    if cache_key in _CACHE:
        return _CACHE[cache_key]

    if not FRED_KEY or len(FRED_KEY) < 16:
        raise RuntimeError("FRED_API_KEY missing or invalid")

    start = (dt.date.today() - dt.timedelta(days=5 * 365)).isoformat()  # ~5y history
    async with aiohttp.ClientSession() as session:
        tasks = [ _fetch_series(session, sid, start) for sid in SERIES.values() ]
        results = await asyncio.gather(*tasks)

    data = { name: ser for name, ser in zip(SERIES.keys(), results) }
    _CACHE[cache_key] = data
    return data

def _zscore(s: pd.Series, win: int = 252) -> float:
    s = s.dropna()
    if len(s) < win:
        win = max(30, min(len(s), win))
    if len(s) < 10:
        return float("nan")
    last = s.iloc[-1]
    mu = s.iloc[-win:].mean()
    sd = s.iloc[-win:].std(ddof=0)
    return float((last - mu) / (sd if sd > 0 else 1e-9))

def _mom_yoy(s: pd.Series) -> float:
    s = s.dropna()
    if len(s) < 365:
        return float("nan")
    return float((s.iloc[-1] / s.iloc[-365] - 1.0) * 100.0)

def compute_features(raw: Dict[str, pd.Series]) -> Dict[str, Any]:
    # Prefer real yield from TIPS; fallback = nominal - infl expectations
    tips = raw.get("TIPS10Y")
    real_yield: Optional[pd.Series] = None
    if tips is not None and len(tips) > 0 and tips.dropna().shape[0] > 10:
        real_yield = tips.rename("REAL10Y")
    else:
        if raw.get("UST10Y") is not None and raw.get("INF_EXP") is not None:
            real_yield = (raw["UST10Y"] - raw["INF_EXP"]).rename("REAL10Y")

    features: Dict[str, Any] = {
        "now": dt.datetime.utcnow().isoformat() + "Z",
        "latest": {},
        "zscore": {},
        "yoy": {},
    }

    for label, s in raw.items():
        if s is None or s.empty: 
            continue
        features["latest"][label] = float(s.iloc[-1])
        features["zscore"][label] = _zscore(s, win=252)
        # yoy is meaningful for level series; it’s fine to compute broadly
        features["yoy"][label] = _mom_yoy(s)

    if real_yield is not None and not real_yield.empty:
        features["latest"]["REAL10Y"] = float(real_yield.iloc[-1])
        features["zscore"]["REAL10Y"] = _zscore(real_yield, win=252)
        features["yoy"]["REAL10Y"] = _mom_yoy(real_yield)

    return features

def score_from_features(F: Dict[str, Any]) -> Dict[str, Any]:
    """
    Turn features into a single 'Gold Macro Score' in [-1, +1].
    Heuristic weighting (easy to tweak):
      - REAL10Y (most important): bearish when high → weight -0.45
      - USD_Broad: stronger USD bearish → weight -0.25
      - INF_EXP: higher expected inflation bullish → weight +0.20
      - UST10Y: higher nominal yields bearish, but partly in REAL → small -0.10
    We use z-scores, then squash with tanh to keep in [-1,1].
    """
    Z = F.get("zscore", {})
    def getz(key): 
        z = Z.get(key)
        return 0.0 if z is None or np.isnan(z) else float(z)

    # Collect signals
    z_real = getz("REAL10Y")
    z_usd  = getz("USD_Broad")
    z_inf  = getz("INF_EXP")
    z_nom  = getz("UST10Y")

    raw_score = (
        -0.45 * z_real +
        -0.25 * z_usd  +
        +0.20 * z_inf  +
        -0.10 * z_nom
    )

    score = float(np.tanh(raw_score / 2.0))  # gentle squashing
    stance = "bullish" if score > 0.25 else "bearish" if score < -0.25 else "neutral"

    why: List[str] = []
    if z_real > 0.5:  why.append("Real yields are elevated (bearish).")
    if z_real < -0.5: why.append("Real yields are depressed (bullish).")
    if z_usd > 0.5:   why.append("USD is strong (bearish).")
    if z_usd < -0.5:  why.append("USD is weak (bullish).")
    if z_inf > 0.5:   why.append("Inflation expectations are elevated (bullish).")
    if z_nom > 0.5:   why.append("Nominal yields are elevated (bearish).")

    return {
        "score": score,             # -1..+1
        "stance": stance,           # bullish / neutral / bearish
        "drivers": {
            "REAL10Y_z": z_real,
            "USD_Broad_z": z_usd,
            "INF_EXP_z": z_inf,
            "UST10Y_z": z_nom,
        },
        "explanation": why,
    }

async def build_insights() -> Dict[str, Any]:
    raw = await fetch_macro()
    feats = compute_features(raw)
    macro = score_from_features(feats)
    return {
        "ok": True,
        "features": feats,
        "macro_score": macro,
        "source": "FRED",
    }
