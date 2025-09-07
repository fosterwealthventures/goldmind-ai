# aggregator.py
import os
import math
import datetime as dt
from typing import Any, Dict, List, Optional, Tuple

import httpx
import numpy as np
import pandas as pd

TD_BASE = os.getenv("TWELVE_DATA_BASE", os.getenv("TWELVEDATA_BASE", "https://api.twelvedata.com"))
TD_KEY  = os.getenv("TWELVE_DATA_API_KEY", os.getenv("TWELVEDATA_API_KEY", ""))
FRED_BASE = "https://api.stlouisfed.org/fred/series/observations"
FRED_KEY  = os.getenv("FRED_API_KEY", "")

# --- Public: UI tile catalog ---------------------------------------------------
def list_indicators() -> List[Dict[str, Any]]:
    return [
        {"id": "SMA",        "label": "Simple Moving Average (SMA)",        "params": {"periods":[20,50,200]}},
        {"id": "EMA",        "label": "Exponential Moving Average (EMA)",   "params": {"periods":[9,21,55]}},
        {"id": "RSI",        "label": "Relative Strength Index (RSI)",      "params": {"period":14}},
        {"id": "MACD",       "label": "MACD",                               "params": {"fast":12, "slow":26, "signal":9}},
        {"id": "BOLL",       "label": "Bollinger Bands",                    "params": {"period":20, "stdev":2}},
        {"id": "STOCH",      "label": "Stochastic Oscillator",              "params": {"k":14, "d":3}},
        {"id": "DXY_CORR",   "label": "DXY Correlation",                    "params": {"window":90}},
        {"id": "REAL_YIELDS","label": "Real Yields Impact (10y TIPS)",      "params": {}},
        {"id": "COT",        "label": "COT Positioning",                    "params": {}},
        {"id": "GDX_GLD",    "label": "Gold Miners Ratio (GDX/GLD)",        "params": {}},
        {"id": "VIX_CORR",   "label": "VIX Correlation",                    "params": {"window":90}},
        {"id": "CB_POLICY",  "label": "Central Bank Policy",                "params": {}},
    ]

# --- Public: run all requested insights ---------------------------------------
async def run_insights(
    symbol: str,
    timeframe: str,
    range: str,
    indicators: List[str],
    params: Dict[str, Any],
    context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:

    out: Dict[str, Any] = {
        "asof": dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
        "summary": {},
        "indicators": {},
        "explanations": [],
        "sources": [],
        "errors": [],
    }

    try:
        px = await _fetch_ohlc(symbol, _interval(timeframe), range)
        if px.empty:
            raise RuntimeError("No price data")
    except Exception as e:
        out["errors"].append({"stage": "prices", "detail": str(e)})
        return out

    # --- TECHNICALS ---
    if "SMA" in indicators:
        per = params.get("SMA", {}).get("periods", [20,50,200])
        sma_vals = {str(p): float(px["close"].rolling(p).mean().iloc[-1]) for p in per if len(px)>=p}
        signal = _sma_signal(px["close"], per)
        out["indicators"]["SMA"] = {"values": sma_vals, "signal": signal}
    if "EMA" in indicators:
        per = params.get("EMA", {}).get("periods", [9,21,55])
        ema_vals = {str(p): float(px["close"].ewm(span=p, adjust=False).mean().iloc[-1]) for p in per if len(px)>=p}
        out["indicators"]["EMA"] = {"values": ema_vals, "signal": _ema_signal(px["close"], per)}
    if "RSI" in indicators:
        period = int(params.get("RSI", {}).get("period", 14))
        rsi_val = float(_rsi(px["close"], period)[-1])
        out["indicators"]["RSI"] = {"value": rsi_val, "signal": _rsi_sig(rsi_val)}
    if "MACD" in indicators:
        fast = int(params.get("MACD", {}).get("fast", 12))
        slow = int(params.get("MACD", {}).get("slow", 26))
        sig  = int(params.get("MACD", {}).get("signal", 9))
        macd_line, sig_line, hist = _macd(px["close"], fast, slow, sig)
        out["indicators"]["MACD"] = {
            "line": float(macd_line[-1]), "signal": float(sig_line[-1]), "hist": float(hist[-1]),
            "signal_hint": "bullish" if hist[-1] > 0 else "bearish" if hist[-1] < 0 else "neutral"
        }
    if "BOLL" in indicators:
        period = int(params.get("BOLL", {}).get("period", 20))
        stdev  = float(params.get("BOLL", {}).get("stdev", 2))
        mid = px["close"].rolling(period).mean()
        sd  = px["close"].rolling(period).std()
        upper = float((mid + stdev*sd).iloc[-1]); middle = float(mid.iloc[-1]); lower = float((mid - stdev*sd).iloc[-1])
        width = (upper-lower)/middle if middle else None
        out["indicators"]["BOLL"] = {"upper": upper, "middle": middle, "lower": lower, "width": float(width) if width else None}
    if "STOCH" in indicators:
        kperiod = int(params.get("STOCH", {}).get("k", 14))
        dperiod = int(params.get("STOCH", {}).get("d", 3))
        k, d = _stoch(px, kperiod, dperiod)
        out["indicators"]["STOCH"] = {"k": float(k[-1]), "d": float(d[-1]), "signal": _stoch_sig(k[-1], d[-1])}

    # --- CROSS / MACRO ---
    if "DXY_CORR" in indicators:
        try:
            dxy = await _fetch_ohlc("DXY", _interval(timeframe), range)
            rho = _rolling_corr(px["close"], dxy["close"], window=int(params.get("DXY_CORR", {}).get("window", 90)))
            out["indicators"]["DXY_CORR"] = {"rho": float(rho), "window": "90d", "signal": "inverse OK" if rho < 0 else "check"}
            out["sources"].append("TwelveData")
        except Exception as e:
            out["errors"].append({"stage": "DXY", "detail": str(e)})

    if "VIX_CORR" in indicators:
        try:
            vix = await _fetch_ohlc("^VIX", _interval(timeframe), range)
            rho = _rolling_corr(px["close"], vix["close"], window=int(params.get("VIX_CORR", {}).get("window", 90)))
            out["indicators"]["VIX_CORR"] = {"rho": float(rho), "window": "90d", "signal": "normal" }
            out["sources"].append("CBOE/TwelveData")
        except Exception as e:
            out["errors"].append({"stage": "VIX", "detail": str(e)})

    if "GDX_GLD" in indicators:
        try:
            gdx = await _fetch_ohlc("GDX", _interval(timeframe), range)
            gld = await _fetch_ohlc("GLD", _interval(timeframe), range)
            ratio = float((gdx["close"].iloc[-1] / gld["close"].iloc[-1]))
            trend = "improving" if ratio > (gdx["close"]/gld["close"]).rolling(50).mean().iloc[-1] else "deteriorating"
            out["indicators"]["GDX_GLD"] = {"ratio": ratio, "trend": trend, "signal": "risk-on miners" if trend=="improving" else "risk-off miners"}
            out["sources"].append("TwelveData")
        except Exception as e:
            out["errors"].append({"stage": "GDX/GLD", "detail": str(e)})

    if "REAL_YIELDS" in indicators:
        try:
            tips10 = await _fetch_fred_series("DFII10")  # 10-Year TIPS
            val = float(tips10.dropna().iloc[-1])
            out["indicators"]["REAL_YIELDS"] = {"tips10": val, "signal": "headwind" if val>1.5 else "neutral/tailwind"}
            out["sources"].append("FRED")
        except Exception as e:
            out["errors"].append({"stage": "REAL_YIELDS", "detail": str(e)})

    if "COT" in indicators:
        try:
            # Easiest maintained proxy: net noncommercial futures from FRED series for gold futures:
            # e.g., GC Net Positions (try FRED series: "CFTC_GC_F_L_ALL" variants vary; keep flexible at runtime)
            cot = await _try_fred_any(["CFTC_GC_F_L_ALL", "CFTC_GC_F_NOCHG_ALL", "CFTC_COMMFUT_NETPOS"])
            if cot is not None:
                val = float(cot.dropna().iloc[-1])
                out["indicators"]["COT"] = {"net_noncommercial": val, "signal": "supportive" if val>0 else "headwind"}
                out["sources"].append("CFTC/FRED")
            else:
                out["errors"].append({"stage":"COT","detail":"no known FRED series matched"})
        except Exception as e:
            out["errors"].append({"stage": "COT", "detail": str(e)})

    if "CB_POLICY" in indicators:
        # Lightweight placeholder; your macro module can enrich this
        out["indicators"]["CB_POLICY"] = {"stance":"on-hold","path":"data-dependent","signal":"neutral"}

    # --- simple, readable summary
    out["summary"] = _build_summary(out["indicators"])
    return out

# --- helpers -------------------------------------------------------------------

def _interval(tf: str) -> str:
    return {"1h":"1h", "4h":"4h", "1d":"1day", "1wk":"1week"}.get(tf, "1day")

async def _fetch_ohlc(symbol: str, interval: str, range_: str) -> pd.DataFrame:
    if not TD_KEY:
        raise RuntimeError("TWELVE_DATA_API_KEY missing")
    url = f"{TD_BASE}/time_series"
    params = {"symbol": symbol, "interval": interval, "outputsize": _outputsize(range_, interval), "apikey": TD_KEY, "format":"JSON"}
    async with httpx.AsyncClient(timeout=20.0) as client:
        r = await client.get(url, params=params)
        r.raise_for_status()
        js = r.json()
    if "values" not in js:
        raise RuntimeError(f"TwelveData error: {js}")
    df = pd.DataFrame(js["values"])
    for c in ["open","high","low","close"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").set_index("datetime")
    return df

def _outputsize(range_: str, interval: str) -> int:
    # coarse but good enough
    if "y" in range_:
        yrs = int(range_.replace("y",""))
        return 260*yrs if interval=="1day" else 1000
    if "m" in range_:
        mos = int(range_.replace("m",""))
        return 22*mos if interval=="1day" else 1000
    return 1000

async def _fetch_fred_series(series_id: str) -> pd.Series:
    if not FRED_KEY:
        raise RuntimeError("FRED_API_KEY missing")
    params = {"series_id": series_id, "api_key": FRED_KEY, "file_type": "json", "observation_start": "2000-01-01"}
    async with httpx.AsyncClient(timeout=20.0) as client:
        r = await client.get(FRED_BASE, params=params)
        r.raise_for_status()
        js = r.json()
    if "observations" not in js:
        raise RuntimeError(f"FRED error: {js}")
    df = pd.DataFrame(js["observations"])
    df["date"] = pd.to_datetime(df["date"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.set_index("date")["value"]
    return df

async def _try_fred_any(series_ids: List[str]) -> Optional[pd.Series]:
    for s in series_ids:
        try:
            ser = await _fetch_fred_series(s)
            if ser.dropna().shape[0] > 0:
                return ser
        except Exception:
            continue
    return None

def _sma_signal(close: pd.Series, periods: List[int]) -> str:
    vals = [close.rolling(p).mean().iloc[-1] for p in periods if len(close)>=p]
    if len(vals) >= 2 and vals[0] > vals[-1]:
        return "bullish"
    if len(vals) >= 2 and vals[0] < vals[-1]:
        return "bearish"
    return "neutral"

def _ema_signal(close: pd.Series, periods: List[int]) -> str:
    e = [close.ewm(span=p, adjust=False).mean().iloc[-1] for p in periods if len(close)>=p]
    return "bullish" if e and e[0] > e[-1] else "bearish" if e else "neutral"

def _rsi(close: pd.Series, period: int) -> pd.Series:
    delta = close.diff()
    up = np.where(delta>0, delta, 0.0); down = np.where(delta<0, -delta, 0.0)
    roll_up = pd.Series(up, index=close.index).rolling(period).mean()
    roll_dn = pd.Series(down, index=close.index).rolling(period).mean()
    rs = roll_up/roll_dn
    rsi = 100 - (100/(1+rs))
    return rsi

def _rsi_sig(v: float) -> str:
    if v >= 70: return "overbought"
    if v <= 30: return "oversold"
    return "neutral-bullish" if v>50 else "neutral-bearish"

def _macd(close: pd.Series, fast: int, slow: int, signal: int) -> Tuple[pd.Series,pd.Series,pd.Series]:
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    sig = macd.ewm(span=signal, adjust=False).mean()
    hist = macd - sig
    return macd, sig, hist

def _stoch(px: pd.DataFrame, k: int, d: int) -> Tuple[pd.Series,pd.Series]:
    low_k  = px["low"].rolling(k).min()
    high_k = px["high"].rolling(k).max()
    percent_k = 100*(px["close"]-low_k)/(high_k-low_k)
    percent_d = percent_k.rolling(d).mean()
    return percent_k, percent_d

def _stoch_sig(k: float, d: float) -> str:
    if k>80 and k>d: return "overbought-ish"
    if k<20 and k<d: return "oversold-ish"
    return "neutral"

def _rolling_corr(a: pd.Series, b: pd.Series, window: int) -> float:
    join = pd.concat([a, b], axis=1).dropna()
    if len(join) < window:
        window = max(10, len(join)//2)
    return float(join.iloc[-window: , 0].corr(join.iloc[-window:, 1]))

def _build_summary(ind: Dict[str, Any]) -> Dict[str, Any]:
    tech = []
    if "SMA" in ind and "EMA" in ind: tech.append(ind["SMA"].get("signal","neutral"))
    if "RSI" in ind: tech.append(ind["RSI"].get("signal","neutral"))
    tech_sig = " • ".join([t for t in tech if t and t!="neutral"])
    macro = []
    if "REAL_YIELDS" in ind:
        y = ind["REAL_YIELDS"].get("tips10")
        if y is not None: macro.append("headwind" if y>1.5 else "neutral/tailwind")
    risk = []
    if "VIX_CORR" in ind: risk.append("normal")
    return {"technical": tech_sig or "mixed", "macro": ", ".join(macro) or "mixed", "risk": ", ".join(risk) or "balanced"}
