"""
FinancialDataFramework.py

A unified data access layer for GoldMIND AI.
This module fetches and processes historical data for various financial instruments,
with a focus on gold (XAU/USD), using a multi-provider fallback system and caching.
"""

import asyncio
import aiohttp
import sqlite3
import pandas as pd
import yfinance as yf
import os
import json
from datetime import datetime, timedelta
from contextlib import contextmanager
import logging
from pathlib import Path
from typing import Optional, Dict, Any


class FinancialDataFramework:
    """
    A unified data access layer for GoldMIND AI.
    Fetches and processes historical and real-time data for gold using multiple providers
    with fallback logic and SQLite caching.
    """

    def __init__(self, api_keys: Dict[str, str], config: Optional[Dict[str, Any]] = None):
        # Use an instance-specific logger
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.INFO)

        # Merge provided config with any global config.json
        global_cfg = {}
        cfg_path = Path(__file__).parent / "config.json"
        if cfg_path.exists():
            try:
                global_cfg = json.loads(cfg_path.read_text())
                self.logger.info(f"Loaded global config from {cfg_path}")
            except Exception as e:
                self.logger.warning(f"Failed to load global config: {e}")
        self.config = {**global_cfg, **(config or {})}
        self.api_keys = api_keys

        # Database/cache settings
        self.db_path = self.config.get("database", {}).get("path", "./data/financial_data_cache.db")
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._init_database()

        # Provider symbols
        fd_cfg = self.config.get("financial_data_api", {})
        self.twelvedata_symbol = fd_cfg.get("twelvedata_gold_symbol", "XAU/USD")
        self.yfinance_symbol = fd_cfg.get("yfinance_symbol", "GC=F")
        self.alpha_vantage_symbol = fd_cfg.get("alpha_vantage_symbol", "XAU")
        self.fred_series_id = fd_cfg.get("fred_series_id", "GOLDAMGBD228NLBM")

        # Base URLs
        self.twelvedata_base = "https://api.twelvedata.com/time_series"
        self.twelvedata_price = "https://api.twelvedata.com/price"
        self.alpha_vantage_base = "https://www.alphavantage.co/query"
        self.metals_api_base = fd_cfg.get("metals_api_base_url", "https://metals-api.com/api")
        self.goldpricez_base = fd_cfg.get("goldpricez_base_url", "https://goldpricez.com/api")
        
        self.session = None
        self.logger.info("✅ FinancialDataFramework initialized.")

    async def _get_session(self):
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession()
        return self.session

    async def close(self):
        if self.session and not self.session.closed:
            await self.session.close()
            self.logger.info("AIOHTTP client session closed.")

    @contextmanager
    def get_connection(self):
        conn = None
        try:
            conn = sqlite3.connect(self.db_path)
            conn.execute("PRAGMA journal_mode=WAL;")
            yield conn
        except sqlite3.Error as e:
            self.logger.error(f"DB connection error: {e}")
            raise
        finally:
            if conn:
                conn.close()

    def _init_database(self):
        with self.get_connection() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS historical_data_cache (
                    ticker TEXT NOT NULL,
                    datetime TEXT NOT NULL,
                    interval TEXT NOT NULL,
                    open REAL,
                    high REAL,
                    low REAL,
                    close REAL,
                    volume INTEGER,
                    last_updated TIMESTAMP,
                    PRIMARY KEY (ticker, datetime, interval)
                )"""
            )
            conn.commit()

    async def get_processed_data(self, ticker: str, days_back: int, interval: str = "1day") -> pd.DataFrame:
        """
        Public method: returns historical OHLCV+Price for the past `days_back` days.
        Checks SQLite cache first, then falls back through providers.
        """
        end_dt = datetime.now()
        start_dt = end_dt - timedelta(days=days_back)

        # 1) Cache
        df = self._get_from_cache(ticker, start_dt, end_dt, interval)
        if df is not None:
            return df

        # 2) Providers in order
        providers = [
            self._fetch_twelvedata,
            self._fetch_yfinance,
            self._fetch_alpha_vantage,
            self._fetch_metals_api,
            self._fetch_fred
        ]
        for fetch in providers:
            symbol = self._map_ticker_for_provider(fetch.__name__, ticker)
            start_str, end_str = start_dt.strftime('%Y-%m-%d'), end_dt.strftime('%Y-%m-%d')
            df = await fetch(symbol, start_str, end_str, interval)
            if df is not None and not df.empty:
                self._cache_data(ticker, df, interval)
                return df

        self.logger.error(f"All providers failed for {ticker}")
        return pd.DataFrame()

    async def get_realtime_price(self, ticker: str) -> Optional[float]:
        """
        Public method: returns latest price from first available real-time provider.
        """
        providers = [
            self._fetch_latest_twelvedata,
            self._fetch_latest_yfinance,
            self._fetch_latest_alpha_vantage,
            self._fetch_latest_goldpricez
        ]
        for fetch in providers:
            symbol = self._map_ticker_for_provider(fetch.__name__, ticker)
            price = await fetch(symbol)
            if price is not None:
                return price
        self.logger.error(f"All realtime providers failed for {ticker}")
        return None

    def _map_ticker_for_provider(self, func_name: str, ticker: str) -> str:
        key = ticker.upper()
        if "yfinance" in func_name and key == "XAU/USD":
            return self.yfinance_symbol
        if "twelvedata" in func_name and key == "XAU/USD":
            return self.twelvedata_symbol
        if "alpha_vantage" in func_name and key == "XAU/USD":
            return self.alpha_vantage_symbol
        if "fred" in func_name and key == "XAU/USD":
            return self.fred_series_id
        # metals API uses 'XAU'
        if "metals_api" in func_name and key == "XAU/USD":
            return "XAU"
        return ticker

    def _get_from_cache(self, ticker: str, start: datetime, end: datetime, interval: str) -> Optional[pd.DataFrame]:
        with self.get_connection() as conn:
            df = pd.read_sql_query(
                "SELECT datetime, open, high, low, close, volume FROM historical_data_cache "
                "WHERE ticker=? AND interval=? AND datetime BETWEEN ? AND ? ORDER BY datetime",
                conn, params=(ticker, interval, start.strftime('%Y-%m-%d %H:%M:%S'), end.strftime('%Y-%m-%d %H:%M:%S')),
                parse_dates=['datetime'], index_col='datetime'
            )
        if df.empty:
            self.logger.info("No data found in SQLite cache.")
            return None
        df.rename(columns={'open':'Open','high':'High','low':'Low','close':'Close','volume':'Volume'}, inplace=True)
        df['Price'] = df['Close']
        return df

    def _cache_data(self, ticker: str, df: pd.DataFrame, interval: str):
        records = []
        for ts, row in df.iterrows():
            records.append((
                ticker,
                ts.strftime('%Y-%m-%d %H:%M:%S'),
                interval,
                float(row['Open']) if pd.notna(row['Open']) else None,
                float(row['High']) if pd.notna(row['High']) else None,
                float(row['Low']) if pd.notna(row['Low']) else None,
                float(row['Close']) if pd.notna(row['Close']) else None,
                int(row['Volume']) if pd.notna(row['Volume']) else None,
                datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            ))
        with self.get_connection() as conn:
            conn.executemany(
                "INSERT OR REPLACE INTO historical_data_cache "
                "(ticker, datetime, interval, open, high, low, close, volume, last_updated) "
                "VALUES (?,?,?,?,?,?,?,?,?);", records
            )
            conn.commit()
            self.logger.info(f"Cached {len(records)} rows for {ticker} at {interval} to SQLite.")

    # ---------------- Provider implementations ----------------

    async def _fetch_twelvedata(self, ticker, start, end, interval):
        key = self.api_keys.get('twelvedata_api_key')
        if not key:
            self.logger.warning("Twelve Data API key not found.")
            return None
        params = dict(symbol=ticker, interval=interval, start_date=start, end_date=end, apikey=key)
        try:
            session = await self._get_session()
            resp = await session.get(self.twelvedata_base, params=params, timeout=10)
            resp.raise_for_status()
            data = await resp.json()
            return self._process_twelve_data(data)
        except Exception as e:
            self.logger.error(f"TwelveData historical fetch error for {ticker}: {e}")
            return None

    async def _fetch_yfinance(self, ticker, start, end, interval):
        yf_interval = interval.replace('day','d')
        def sync():
            df = yf.download(ticker, start=start, end=end, interval=yf_interval, progress=False)
            if df.empty: return None
            df.rename(columns=str.capitalize, inplace=True)
            df['Price'] = df['Close']
            return df[['Open','High','Low','Close','Volume','Price']]
        try:
            return await asyncio.to_thread(sync)
        except Exception as e:
            self.logger.error(f"YFinance historical fetch error for {ticker}: {e}")
            return None

    async def _fetch_alpha_vantage(self, ticker, start, end, interval):
        key = self.api_keys.get('alpha_vantage_api_key')
        if not key or interval!='1day':
            self.logger.warning("Alpha Vantage API key not found or unsupported interval.")
            return None
        params = dict(function='TIME_SERIES_DAILY_ADJUSTED', symbol=ticker, outputsize='full', apikey=key)
        try:
            session = await self._get_session()
            resp = await session.get(self.alpha_vantage_base, params=params, timeout=10)
            resp.raise_for_status()
            data = await resp.json()
            return self._process_alpha_vantage(data)
        except Exception as e:
            self.logger.error(f"Alpha Vantage historical fetch error for {ticker}: {e}")
            return None

    async def _fetch_metals_api(self, ticker, start, end, interval):
        key = self.api_keys.get('metalprice_api_key')
        if not key or interval!='1day':
            self.logger.warning("Metalprice API key not found or unsupported interval.")
            return None
        symbol = 'XAU' if ticker.upper()=='XAU/USD' else ticker
        url = f"{self.metals_api_base}/timeseries"
        params = dict(access_key=key, start_date=start, end_date=end, symbols=symbol, base='USD')
        try:
            session = await self._get_session()
            resp = await session.get(url, params=params, timeout=10)
            resp.raise_for_status()
            data = await resp.json()
            rates = data.get('rates', {})
            if not rates:
                self.logger.warning("Metals-API returned no rates.")
                return None
            df = pd.DataFrame.from_dict(rates, orient='index')
            df.index = pd.to_datetime(df.index)
            df.rename(columns={symbol:'Close'}, inplace=True)
            df['Open']=df['High']=df['Low']=df['Close']
            df['Volume']=0; df['Price']=df['Close']
            return df[['Open','High','Low','Close','Volume','Price']]
        except Exception as e:
            self.logger.error(f"Metals-API historical fetch error for {ticker}: {e}")
            return None

    async def _fetch_fred(self, ticker, start, end, interval):
        key = self.api_keys.get('fred_api_key')
        if not key or interval!='1day':
            self.logger.warning("FRED API key not found or unsupported interval.")
            return None
        params = dict(series_id=self.fred_series_id, api_key=key, file_type='json', observation_start=start, observation_end=end)
        try:
            session = await self._get_session()
            resp = await session.get('https://api.stlouisfed.org/fred/series/observations', params=params, timeout=10)
            resp.raise_for_status()
            data = await resp.json()
            return self._process_fred_data(data)
        except Exception as e:
            self.logger.error(f"FRED historical fetch error for {ticker}: {e}")
            return None

    async def _fetch_latest_twelvedata(self, ticker):
        key = self.api_keys.get('twelvedata_api_key')
        if not key: return None
        params = dict(symbol=ticker, apikey=key)
        try:
            session = await self._get_session()
            resp = await session.get(self.twelvedata_price, params=params, timeout=5)
            resp.raise_for_status()
            data = await resp.json()
            return float(data.get('price')) if 'price' in data else None
        except Exception as e:
            self.logger.error(f"TwelveData latest price error for {ticker}: {e}")
            return None

    async def _fetch_latest_yfinance(self, ticker):
        def sync():
            t = yf.Ticker(ticker)
            hist = t.history(period="1d")
            return float(hist["Close"].iloc[-1]) if not hist.empty else None
        try:
            return await asyncio.to_thread(sync)
        except Exception as e:
            self.logger.error(f"YFinance latest price error for {ticker}: {e}")
            return None

    async def _fetch_latest_alpha_vantage(self, ticker):
        key = self.api_keys.get('alpha_vantage_api_key')
        if not key: return None
        av_ticker = "XAU" if ticker.upper() == "XAU/USD" else ticker
        params = dict(function='CURRENCY_EXCHANGE_RATE', from_currency=av_ticker, to_currency='USD', apikey=key)
        try:
            session = await self._get_session()
            resp = await session.get(self.alpha_vantage_base, params=params, timeout=5)
            resp.raise_for_status()
            data = await resp.json()
            rate_info = data.get('Realtime Currency Exchange Rate', {})
            return float(rate_info.get('5. Exchange Rate', 0)) if '5. Exchange Rate' in rate_info else None
        except Exception as e:
            self.logger.error(f"Alpha Vantage realtime error for {ticker}: {e}")
            return None

    async def _fetch_latest_goldpricez(self, ticker):
        key = self.api_keys.get('goldpricez_api_key')
        if not key: return None
        gpz_ticker = "XAU" if ticker.upper() == "XAU/USD" else ticker
        url = f"{self.goldpricez_base}/rates/currency/usd/{key}"
        try:
            session = await self._get_session()
            resp = await session.get(url, timeout=5)
            resp.raise_for_status()
            data = await resp.json()
            price = data.get(gpz_ticker.lower())
            return float(price) if price is not None else None
        except Exception as e:
            self.logger.error(f"Goldpricez latest price error for {ticker}: {e}")
            return None

    # ---------------- Data processors ----------------

    def _process_twelve_data(self, data):
        if not data or data.get('status')=='error' or 'values' not in data: return None
        df = pd.DataFrame(data['values']); df['datetime']=pd.to_datetime(df['datetime']); df.set_index('datetime', inplace=True)
        for c in ['open','high','low','close']: df[c]=pd.to_numeric(df[c],errors='coerce')
        df['volume']=pd.to_numeric(df.get('volume',0),errors='coerce').fillna(0)
        df.dropna(subset=['open','high','low','close'], inplace=True)
        df.rename(columns={'open':'Open','high':'High','low':'Low','close':'Close','volume':'Volume'}, inplace=True)
        df['Price']=df['Close']; return df[['Open','High','Low','Close','Volume','Price']]

    def _process_alpha_vantage(self, data):
        ts = data.get('Time Series (Daily)');
        if not ts: return None
        df = pd.DataFrame.from_dict(ts,orient='index'); df.index=pd.to_datetime(df.index); df.sort_index(inplace=True)
        df.rename(columns={'1. open':'Open','2. high':'High','3. low':'Low','4. close':'Close','6. volume':'Volume'}, inplace=True)
        for c in ['Open','High','Low','Close','Volume']: df[c]=pd.to_numeric(df[c],errors='coerce')
        df.dropna(inplace=True); df['Price']=df['Close']; return df[['Open','High','Low','Close','Volume','Price']]

    def _process_fred_data(self, data):
        obs = data.get('observations');
        if not obs: return None
        df = pd.DataFrame(obs); df['date']=pd.to_datetime(df['date']); df.set_index('date',inplace=True)
        df['value']=pd.to_numeric(df['value'],errors='coerce'); df.dropna(inplace=True)
        df.rename(columns={'value':'Close'}, inplace=True)
        df['Open']=df['High']=df['Low']=df['Price']=df['Close']; df['Volume']=0
        return df[['Open','High','Low','Close','Volume','Price']]

# Sample standalone test
if __name__ == '__main__':
    async def _test():
        logging.basicConfig(level=logging.INFO)
        api_keys = {
            'twelvedata_api_key':'', 'alpha_vantage_api_key':'', 'metalprice_api_key':'',
            'fred_api_key':'', 'goldpricez_api_key':''
        }
        fdf = FinancialDataFramework(api_keys)
        df = await fdf.get_processed_data('XAU/USD',90)
        print(df.head())
        price = await fdf.get_realtime_price('XAU/USD')
        print('Price:',price)
        await fdf.close() # Ensure the session is closed
    asyncio.run(_test())