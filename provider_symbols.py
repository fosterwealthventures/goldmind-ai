# provider_symbols.py
PROVIDER_SYMBOLS = {
    "twelvedata": {"XAUUSD": "XAU/USD", "GLD": "GLD"},
    "alphavantage": {"XAUUSD": "XAUUSD", "GLD": "GLD"},   # FX_DIGITAL or CURRENCY_EXCHANGE
    "yahoo": {"XAUUSD": "XAUUSD=X", "GLD": "GLD"},        # Yahoo tickers
    "yfinance": {"XAUUSD": "XAUUSD=X", "GLD": "GLD"},
    "metalprice": {"XAUUSD": "XAUUSD"},
    "goldpricez": {"XAUUSD": "XAUUSD"},
}
