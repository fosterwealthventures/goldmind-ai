"""
financial_data_framework.py  — GoldMIND AI
Unified, production‑oriented data access with:
- Provider fallback chain (TwelveData → yfinance → Alpha Vantage → Metals API → FRED)
- SQLite cache (WAL) + optional in‑memory TTL cache
- Resilient aiohttp session management (single shared client)
- Retries with exponential backoff on network errors
- Clean OHLCV normalization to a consistent schema
- Drop‑in compatibility with server.py and other managers

Public surface:
    FinancialDataFramework(api_keys: dict, config: dict|None)
        .get_processed_data(ticker, days_back, interval="1day") -> pd.DataFrame
        .get_realtime_price(ticker) -> float|None
        .get_connection() -> sqlite3.Connection context manager
        .close() -> close aiohttp session
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sqlite3
import time
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, Optional, Tuple

import aiohttp
import pandas as pd
import yfinance as yf

# --------------------------- Logging ---------------------------

log = logging.getLogger("goldmind.fdf")
if not log.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")


# --------------------------- Helpers ---------------------------

def _now() -> datetime:
    return datetime.now()


def _df_normalize(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize columns to: Open, High, Low, Close, Volume, Price
    Index must be DatetimeIndex in ascending order.
    """
    if df is None or df.empty:
        return pd.DataFrame()

    # Standardize column names first
    renamed = {}
    for c in df.columns:
        lc = c.lower()
        if lc in ("open",): renamed[c] = "Open"
        elif lc in ("high",): renamed[c] = "High"
        elif lc in ("low",): renamed[c] = "Low"
        elif lc in ("close", "adj close", "adjusted close"): renamed[c] = "Close"
        elif lc in ("volume",): renamed[c] = "Volume"
        elif lc in ("price",): renamed[c] = "Price"

    df = df.rename(columns=renamed)
    # Fill derived columns
    if "Price" not in df and "Close" in df:
        df["Price"] = df["Close"]

    # Ensure correct column order
    cols = [c for c in ["Open", "High", "Low", "Close", "Volume", "Price"] if c in df.columns]
    df = df[cols]

    # Ensure datetime index ascending
    if not isinstance(df.index, pd.DatetimeIndex):
        # try to parse index
        try:
            df.index = pd.to_datetime(df.index, utc=False)
        except Exception:
            pass
    df = df.sort_index()
    return df


def _interval_to_yf(interval: str) -> str:
    # map "1day" → "1d", "1week" → "1wk", etc.
    m = {
        "1min": "1m", "5min": "5m", "15min": "15m", "30min": "30m", "60min": "60m",
        "90min": "90m", "1hour": "60m",
        "1day": "1d", "1wk": "1wk", "1week": "1wk", "1mo": "1mo", "1month": "1mo",
    }
    return m.get(interval, "1d")


# --------------------------- In‑memory cache ---------------------------

@dataclass
class _MemEntry:
    expires_at: float
    df: Optional[pd.DataFrame] = None
    value: Optional[float] = None


class _TTLCache:
    def __init__(self, ttl_sec: int = 300, max_items: int = 64) -> None:
        self.ttl = ttl_sec
        self.max = max_items
        self._store: Dict[str, _MemEntry] = {}

    def _evict_if_needed(self) -> None:
        if len(self._store) <= self.max:
            return
        # evict oldest expiry
        oldest_key = min(self._store, key=lambda k: self._store[k].expires_at)
        self._store.pop(oldest_key, None)

    def get_df(self, key: str) -> Optional[pd.DataFrame]:
        e = self._store.get(key)
        if not e or e.expires_at < time.time():
            self._store.pop(key, None)
            return None
        return e.df

    def put_df(self, key: str, df: pd.DataFrame, ttl: Optional[int] = None) -> None:
        exp = time.time() + (ttl if ttl is not None else self.ttl)
        self._store[key] = _MemEntry(expires_at=exp, df=df)
        self._evict_if_needed()

    def get_val(self, key: str) -> Optional[float]:
        e = self._store.get(key)
        if not e or e.expires_at < time.time():
            self._store.pop(key, None)
            return None
        return e.value

    def put_val(self, key: str, val: float, ttl: Optional[int] = None) -> None:
        exp = time.time() + (ttl if ttl is not None else self.ttl)
        self._store[key] = _MemEntry(expires_at=exp, value=val)
        self._evict_if_needed()


# --------------------------- Main class ---------------------------

class FinancialDataFramework:
    """
    A unified data access layer for GoldMIND AI with multi‑provider fallback and caching.
    """

    def __init__(self, api_keys: Dict[str, str] | None = None, config: Optional[Dict[str, Any]] = None) -> None:
        # config merging
        self.api_keys = api_keys or {}
        cfg_path = Path(__file__).parent / "config.json"
        global_cfg: Dict[str, Any] = {}
        if cfg_path.exists():
            try:
                global_cfg = json.loads(cfg_path.read_text())
                log.info("Loaded global config from %s", cfg_path)
            except Exception as e:
                log.warning("Failed to load global config: %s", e)
        self.config: Dict[str, Any] = {**global_cfg, **(config or {})}

        # cache
        self.mem_cache = _TTLCache(ttl_sec=int(self.config.get("cache", {}).get("ttl_sec", 300)))
        self.db_path = self.config.get("database", {}).get("path", "./data/financial_data_cache.db")
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._init_database()

        # provider config
        fd_cfg = self.config.get("financial_data_api", {})
        self.twelvedata_symbol = fd_cfg.get("twelvedata_gold_symbol", "XAU/USD")
        self.yfinance_symbol = fd_cfg.get("yfinance_symbol", "GC=F")
        self.alpha_vantage_symbol = fd_cfg.get("alpha_vantage_symbol", "XAU")
        self.fred_series_id = fd_cfg.get("fred_series_id", "GOLDAMGBD228NLBM")
        self.metals_api_base = fd_cfg.get("metals_api_base_url", "https://metals-api.com/api")
        self.goldpricez_base = fd_cfg.get("goldpricez_base_url", "https://goldpricez.com/api")
        self.twelvedata_base = "https://api.twelvedata.com/time_series"
        self.twelvedata_price = "https://api.twelvedata.com/price"
        self.alpha_vantage_base = "https://www.alphavantage.co/query"

        # aiohttp client
        self._session: Optional[aiohttp.ClientSession] = None
        self._timeout = aiohttp.ClientTimeout(total=float(self.config.get("http", {}).get("timeout_total_sec", 12.0)))
        self._retries = int(self.config.get("http", {}).get("retries", 2))
        self._backoff = float(self.config.get("http", {}).get("backoff_sec", 0.5))
        log.info("✅ FinancialDataFramework initialized with DB=%s", self.db_path)

    # --------------------- Session management ---------------------

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            headers = {"User-Agent": "GoldMIND-FDF/1.0"}
            self._session = aiohttp.ClientSession(timeout=self._timeout, headers=headers)
        return self._session

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()
            log.info("AIOHTTP client session closed.")

    # --------------------- SQLite helpers ---------------------

    @contextmanager
    def get_connection(self):
        conn = None
        try:
            conn = sqlite3.connect(self.db_path)
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA synchronous=NORMAL;")
            yield conn
        finally:
            if conn:
                conn.close()

    def _init_database(self) -> None:
        with self.get_connection() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS historical_data_cache (
                    ticker TEXT NOT NULL,
                    datetime TEXT NOT NULL,
                    interval TEXT NOT NULL,
                    open REAL, high REAL, low REAL, close REAL, volume INTEGER,
                    last_updated TIMESTAMP,
                    PRIMARY KEY (ticker, datetime, interval)
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_hist_ticker_interval ON historical_data_cache (ticker, interval)"
            )
            conn.commit()

    # --------------------- Public APIs ---------------------

    async def get_processed_data(self, ticker: str, days_back: int, interval: str = "1day") -> pd.DataFrame:
        """
        Return OHLCV (+Price) DataFrame for the last `days_back` days at `interval`.
        Order: cache → TwelveData → yfinance → Alpha Vantage → Metals API → FRED.
        """
        end_dt = _now()
        start_dt = end_dt - timedelta(days=int(days_back))

        # Memory cache
        mem_key = f"hist:{ticker}:{interval}:{days_back}"
        cached_df = self.mem_cache.get_df(mem_key)
        if cached_df is not None:
            return cached_df

        # SQLite cache
        df = self._get_from_cache_sqlite(ticker, start_dt, end_dt, interval)
        if df is not None:
            self.mem_cache.put_df(mem_key, df)
            return df

        # Providers
        providers = [
            self._fetch_twelvedata,
            self._fetch_yfinance,
            self._fetch_alpha_vantage,
            self._fetch_metals_api,
            self._fetch_fred,
        ]
        for fetch in providers:
            symbol = self._map_ticker_for_provider(fetch.__name__, ticker)
            s, e = start_dt.strftime("%Y-%m-%d"), end_dt.strftime("%Y-%m-%d")
            df = await fetch(symbol, s, e, interval)
            if df is not None and not df.empty:
                df = _df_normalize(df)
                self._cache_to_sqlite(ticker, df, interval)
                self.mem_cache.put_df(mem_key, df)
                return df

        log.error("All providers failed for %s", ticker)
        return pd.DataFrame()

    async def get_realtime_price(self, ticker: str) -> Optional[float]:
        """
        Return latest price from first available real-time provider.
        """
        mem_key = f"rt:{ticker}"
        cached = self.mem_cache.get_val(mem_key)
        if cached is not None:
            return cached

        providers = [
            self._fetch_latest_twelvedata,
            self._fetch_latest_yfinance,
            self._fetch_latest_alpha_vantage,
            self._fetch_latest_goldpricez,
        ]
        for fetch in providers:
            symbol = self._map_ticker_for_provider(fetch.__name__, ticker)
            price = await fetch(symbol)
            if price is not None:
                self.mem_cache.put_val(mem_key, float(price), ttl=30)  # short TTL for spot price
                return float(price)
        log.error("All realtime providers failed for %s", ticker)
        return None

    # --------------------- Cache (SQLite) ---------------------

    def _get_from_cache_sqlite(self, ticker: str, start: datetime, end: datetime, interval: str) -> Optional[pd.DataFrame]:
        with self.get_connection() as conn:
            df = pd.read_sql_query(
                "SELECT datetime, open, high, low, close, volume FROM historical_data_cache "
                "WHERE ticker=? AND interval=? AND datetime BETWEEN ? AND ? ORDER BY datetime",
                conn,
                params=(ticker, interval, start.strftime("%Y-%m-%d %H:%M:%S"), end.strftime("%Y-%m-%d %H:%M:%S")),
                parse_dates=["datetime"],
                index_col="datetime",
            )
        if df.empty:
            return None
        df = _df_normalize(df)
        return df

    def _cache_to_sqlite(self, ticker: str, df: pd.DataFrame, interval: str) -> None:
        if df is None or df.empty:
            return
        payload = []
        for ts, row in df.iterrows():
            payload.append(
                (
                    ticker,
                    ts.strftime("%Y-%m-%d %H:%M:%S"),
                    interval,
                    float(row["Open"]) if "Open" in row and pd.notna(row["Open"]) else None,
                    float(row["High"]) if "High" in row and pd.notna(row["High"]) else None,
                    float(row["Low"]) if "Low" in row and pd.notna(row["Low"]) else None,
                    float(row["Close"]) if "Close" in row and pd.notna(row["Close"]) else None,
                    int(row["Volume"]) if "Volume" in row and pd.notna(row["Volume"]) else None,
                    _now().strftime("%Y-%m-%d %H:%M:%S"),
                )
            )
        with self.get_connection() as conn:
            conn.executemany(
                "INSERT OR REPLACE INTO historical_data_cache "
                "(ticker, datetime, interval, open, high, low, close, volume, last_updated) "
                "VALUES (?,?,?,?,?,?,?,?,?)",
                payload,
            )
            conn.commit()
        log.info("Cached %d rows for %s@%s", len(payload), ticker, interval)

    # --------------------- Provider plumbing ---------------------

    async def _request_json(self, url: str, params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        GET JSON with retries/backoff.
        """
        last_err: Optional[Exception] = None
        for attempt in range(self._retries + 1):
            try:
                session = await self._get_session()
                async with session.get(url, params=params) as resp:
                    resp.raise_for_status()
                    return await resp.json()
            except Exception as e:
                last_err = e
                await asyncio.sleep(self._backoff * (2 ** attempt))
        log.error("Request failed for %s params=%s error=%s", url, params, last_err)
        return None

    def _map_ticker_for_provider(self, func_name: str, ticker: str) -> str:
        t = ticker.upper()
        if "yfinance" in func_name and t in ("XAU/USD", "XAUUSD"):
            return self.yfinance_symbol
        if "twelvedata" in func_name and t in ("XAU/USD", "XAUUSD"):
            return self.twelvedata_symbol
        if "alpha_vantage" in func_name and t in ("XAU/USD", "XAUUSD"):
            return self.alpha_vantage_symbol
        if "fred" in func_name and t in ("XAU/USD", "XAUUSD"):
            return self.fred_series_id
        if "metals_api" in func_name and t in ("XAU/USD", "XAUUSD"):
            return "XAU"
        return ticker

    # ---- Historical providers ----

    async def _fetch_twelvedata(self, ticker: str, start: str, end: str, interval: str) -> Optional[pd.DataFrame]:
        key = self.api_keys.get("twelvedata_api_key")
        if not key:
            log.info("TwelveData: no API key; skipping.")
            return None
        params = dict(symbol=ticker, interval=interval, start_date=start, end_date=end, apikey=key, order="ASC")
        data = await self._request_json(self.twelvedata_base, params)
        if not data or data.get("status") == "error" or "values" not in data:
            return None
        df = pd.DataFrame(data["values"])
        if df.empty:
            return None
        df["datetime"] = pd.to_datetime(df["datetime"])
        df = df.set_index("datetime")
        for c in ("open", "high", "low", "close", "volume"):
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        df.dropna(subset=["open", "high", "low", "close"], inplace=True)
        return _df_normalize(df)

    async def _fetch_yfinance(self, ticker: str, start: str, end: str, interval: str) -> Optional[pd.DataFrame]:
        yf_interval = _interval_to_yf(interval)

        def sync_dl():
            df = yf.download(ticker, start=start, end=end, interval=yf_interval, progress=False)
            if df is None or df.empty:
                return None
            # yfinance returns columns like Open, High, Low, Close, Adj Close, Volume
            df = df.rename(columns=str.capitalize)
            return _df_normalize(df)

        try:
            return await asyncio.to_thread(sync_dl)
        except Exception as e:
            log.warning("yfinance error for %s: %s", ticker, e)
            return None

    async def _fetch_alpha_vantage(self, ticker: str, start: str, end: str, interval: str) -> Optional[pd.DataFrame]:
        key = self.api_keys.get("alpha_vantage_api_key")
        if not key or interval != "1day":
            return None
        params = dict(function="TIME_SERIES_DAILY_ADJUSTED", symbol=ticker, outputsize="full", apikey=key)
        data = await self._request_json(self.alpha_vantage_base, params)
        ts = (data or {}).get("Time Series (Daily)")
        if not ts:
            return None
        df = pd.DataFrame.from_dict(ts, orient="index")
        df.index = pd.to_datetime(df.index)
        df.sort_index(inplace=True)
        rename = {
            "1. open": "Open", "2. high": "High", "3. low": "Low", "4. close": "Close",
            "6. volume": "Volume"
        }
        df = df.rename(columns=rename)
        for c in ["Open", "High", "Low", "Close", "Volume"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        df.dropna(inplace=True)
        return _df_normalize(df)

    async def _fetch_metals_api(self, ticker: str, start: str, end: str, interval: str) -> Optional[pd.DataFrame]:
        key = self.api_keys.get("metalprice_api_key")
        if not key or interval != "1day":
            return None
        url = f"{self.metals_api_base}/timeseries"
        symbol = "XAU" if ticker.upper() in ("XAU/USD", "XAUUSD") else ticker
        params = dict(access_key=key, start_date=start, end_date=end, symbols=symbol, base="USD")
        data = await self._request_json(url, params)
        if not data or not data.get("rates"):
            return None
        df = pd.DataFrame.from_dict(data["rates"], orient="index")
        df.index = pd.to_datetime(df.index)
        df = df.rename(columns={symbol: "Close"})
        df["Open"] = df["High"] = df["Low"] = df["Close"]
        df["Volume"] = 0
        return _df_normalize(df)

    async def _fetch_fred(self, ticker: str, start: str, end: str, interval: str) -> Optional[pd.DataFrame]:
        key = self.api_keys.get("fred_api_key")
        if not key or interval != "1day":
            return None
        url = "https://api.stlouisfed.org/fred/series/observations"
        params = dict(series_id=self.fred_series_id, api_key=key, file_type="json",
                      observation_start=start, observation_end=end)
        data = await self._request_json(url, params)
        obs = (data or {}).get("observations")
        if not obs:
            return None
        df = pd.DataFrame(obs)
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df.dropna(inplace=True)
        df = df.rename(columns={"value": "Close"})
        df["Open"] = df["High"] = df["Low"] = df["Price"] = df["Close"]
        df["Volume"] = 0
        return _df_normalize(df)

    # ---- Realtime providers ----

    async def _fetch_latest_twelvedata(self, ticker: str) -> Optional[float]:
        key = self.api_keys.get("twelvedata_api_key")
        if not key:
            return None
        params = dict(symbol=ticker, apikey=key)
        data = await self._request_json(self.twelvedata_price, params)
        try:
            return float(data.get("price")) if data and "price" in data else None
        except Exception:
            return None

    async def _fetch_latest_yfinance(self, ticker: str) -> Optional[float]:
        def sync_price():
            t = yf.Ticker(ticker)
            hist = t.history(period="1d")
            if hist is None or hist.empty:
                return None
            return float(hist["Close"].iloc[-1])

        try:
            return await asyncio.to_thread(sync_price)
        except Exception:
            return None

    async def _fetch_latest_alpha_vantage(self, ticker: str) -> Optional[float]:
        key = self.api_keys.get("alpha_vantage_api_key")
        if not key:
            return None
        av_ticker = "XAU" if ticker.upper() in ("XAU/USD", "XAUUSD") else ticker
        params = dict(function="CURRENCY_EXCHANGE_RATE", from_currency=av_ticker, to_currency="USD", apikey=key)
        data = await self._request_json(self.alpha_vantage_base, params)
        rate = (data or {}).get("Realtime Currency Exchange Rate", {})
        try:
            return float(rate.get("5. Exchange Rate")) if "5. Exchange Rate" in rate else None
        except Exception:
            return None

    async def _fetch_latest_goldpricez(self, ticker: str) -> Optional[float]:
        key = self.api_keys.get("goldpricez_api_key")
        if not key:
            return None
        gpz_ticker = "XAU" if ticker.upper() in ("XAU/USD", "XAUUSD") else ticker
        url = f"{self.goldpricez_base}/rates/currency/usd/{key}"
        data = await self._request_json(url, {})
        try:
            val = (data or {}).get(gpz_ticker.lower())
            return float(val) if val is not None else None
        except Exception:
            return None


# --------------------------- Standalone test ---------------------------

if __name__ == "__main__":
    async def _test():
        logging.basicConfig(level=logging.INFO)
        api_keys = {
            "twelvedata_api_key": "",
            "alpha_vantage_api_key": "",
            "metalprice_api_key": "",
            "fred_api_key": "",
            "goldpricez_api_key": "",
        }
        fdf = FinancialDataFramework(api_keys)
        df = await fdf.get_processed_data("XAU/USD", 60)
        print("--- df.head() ---")
        print(df.head())
        price = await fdf.get_realtime_price("XAU/USD")
        print("spot:", price)
        await fdf.close()

    asyncio.run(_test())
