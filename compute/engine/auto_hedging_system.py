"""
auto_hedging_system.py — GoldMIND AI (hardened)
-----------------------------------------------
Automated hedging & risk management aligned with the new dashboard and
backend modules (FinancialDataFramework, NotificationManager).

Key upgrades vs earlier draft:
- Missing imports fixed (threading, sqlite3); safer async usage in thread loop
- Clear dashboard helpers (to_recommendation_card / to_summary_block)
- Defensive DB writes (SQLite-friendly) with schema bootstrap helper
- Stable enums & dataclasses, consistent JSON-serializable outputs
- Broker interface remains mock-friendly; quantity sizing and execution guards
- Circuit breaker + effectiveness tracking preserved
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import json
import logging
import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger("goldmind.auto_hedging")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")

# ---------- Optional deps & soft imports ----------

try:
    from financial_data_framework import FinancialDataFramework
except Exception:  # pragma: no cover
    class FinancialDataFramework:  # type: ignore
        def get_connection(self):
            return sqlite3.connect(":memory:", check_same_thread=False)
        async def get_multiple_stocks(self, symbols: List[str], outputsize: int) -> Dict[str, pd.DataFrame]:
            idx = pd.date_range(end=datetime.utcnow(), periods=outputsize, freq="D")
            out = {}
            for s in symbols:
                prices = np.cumsum(np.random.normal(0, 1, len(idx))) + 100
                out[s] = pd.DataFrame({"close": prices, "high": prices*1.003, "low": prices*0.997}, index=idx)
            return out
        async def get_economic_data(self, series_id: str) -> pd.DataFrame:
            idx = pd.date_range(end=datetime.utcnow(), periods=30, freq="D")
            return pd.DataFrame({"value": np.random.uniform(15, 30, len(idx))}, index=idx)

try:
    from notification_system import NotificationManager
except Exception:  # pragma: no cover
    class NotificationManager:  # type: ignore
        def send_email_notification(self, *a, **k): logger.info("Mock email sent: %s", k)
        def send_system_alert(self, *a, **k): logger.info("Mock system alert: %s", k)
        def send_alert(self, *a, **k): logger.info("Mock alert: %s", k)
        def send_report(self, *a, **k): logger.info("Mock report: %s", k)


# ---------- Types ----------

class HedgeType(Enum):
    DIRECT_OFFSET = "direct_offset"
    VOLATILITY_HEDGE = "volatility_hedge"
    CORRELATION_HEDGE = "correlation_hedge"
    SECTOR_DIVERSIFICATION = "sector_diversification"
    TAIL_RISK_HEDGE = "tail_risk_hedge"
    LIQUIDITY_HEDGE = "liquidity_hedge"


class RiskLevel(Enum):
    LOW = 1
    MODERATE = 2
    HIGH = 3
    CRITICAL = 4
    EXTREME = 5


class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"


@dataclass
class Position:
    symbol: str
    quantity: float
    entry_price: float
    current_price: float
    entry_time: datetime
    position_type: str  # 'long' or 'short'
    market_value: float  # current_price * quantity


@dataclass
class HedgeRecommendation:
    hedge_id: str
    hedge_type: HedgeType
    hedge_symbol: str
    hedge_action: str  # BUY or SELL
    hedge_size: float  # fraction of portfolio value (e.g., 0.05 = 5%)
    hedge_ratio: float
    risk_reduction: float
    cost_estimate: float  # bps
    confidence: float  # 0..1
    reasoning: str
    urgency: RiskLevel
    expiry_date: Optional[datetime] = None
    order_type: OrderType = OrderType.MARKET


@dataclass
class RiskMetrics:
    portfolio_var: float
    max_drawdown: float
    sharpe_ratio: float
    volatility: float
    concentration_risk: float
    correlation_risk: float
    tail_risk: float
    liquidity_risk: float
    overall_risk_score: float


# ---------- Broker abstraction (mock-friendly) ----------

class BrokerAPI:
    async def place_order(self, symbol: str, action: str, quantity: float, order_type: OrderType) -> Optional[str]: ...
    async def cancel_order(self, order_id: str) -> bool: ...
    async def get_position(self, symbol: str) -> Optional[Position]: ...
    async def get_account_info(self) -> Dict[str, Any]: ...
    async def get_current_price(self, symbol: str) -> Optional[float]: ...


class AlpacaBroker(BrokerAPI):
    def __init__(self, api_key: str, secret_key: str):
        self.api_key = api_key
        self.secret_key = secret_key
        self.mock_positions: Dict[str, Position] = {}
        self.mock_orders: Dict[str, Dict[str, Any]] = {}
        logger.info("AlpacaBroker (mock) initialized.")

    async def place_order(self, symbol: str, action: str, quantity: float, order_type: OrderType) -> Optional[str]:
        oid = str(uuid.uuid4())
        px = await self.get_current_price(symbol) or 100.0
        self.mock_orders[oid] = {
            "id": oid, "symbol": symbol, "qty": quantity, "side": action.lower(),
            "type": order_type.value, "status": "filled", "filled_avg_price": px,
            "submitted_at": datetime.utcnow().isoformat()
        }
        pos = self.mock_positions.get(symbol)
        if not pos:
            pos = Position(symbol, 0.0, 0.0, px, datetime.utcnow(), "long", 0.0)
            self.mock_positions[symbol] = pos
        if action.upper() == "BUY":
            new_q = pos.quantity + quantity
            pos.entry_price = ((pos.entry_price * pos.quantity) + (px * quantity)) / max(new_q, 1e-9)
            pos.quantity = new_q
            pos.position_type = "long" if pos.quantity >= 0 else "short"
        else:
            pos.quantity -= quantity
            pos.position_type = "short" if pos.quantity < 0 else "long"
            pos.entry_price = px
        pos.current_price = px
        pos.market_value = pos.quantity * pos.current_price
        return oid

    async def cancel_order(self, order_id: str) -> bool:
        if order_id in self.mock_orders and self.mock_orders[order_id]["status"] != "filled":
            self.mock_orders[order_id]["status"] = "canceled"
            return True
        return False

    async def get_position(self, symbol: str) -> Optional[Position]:
        pos = self.mock_positions.get(symbol)
        if pos:
            px = await self.get_current_price(symbol) or pos.current_price
            pos.current_price = px
            pos.market_value = px * pos.quantity
        return pos

    async def get_account_info(self) -> Dict[str, Any]:
        pv = sum(p.market_value for p in self.mock_positions.values())
        return {"cash": 50_000.0, "portfolio_value": pv, "buying_power": 100_000.0, "equity": 50_000.0 + pv}

    async def get_current_price(self, symbol: str) -> Optional[float]:
        base = {"GLD": 185.5, "IAU": 40.0, "TLT": 95.0, "SPY": 500.0, "DUST": 15.0, "VIXY": 22.0, "GC=F": 1900.0}.get(symbol, 100.0)
        return float(base + np.random.uniform(-1, 1))


class TradingInterface:
    def __init__(self, brokers: Dict[str, BrokerAPI]):
        if not brokers:
            raise ValueError("TradingInterface requires at least one broker.")
        self.brokers = brokers
        self.default_broker = next(iter(brokers.values()))

    async def execute_hedge(self, recommendation: HedgeRecommendation, portfolio_value: float) -> Optional[str]:
        broker = self.default_broker
        px = await broker.get_current_price(recommendation.hedge_symbol)
        if not px or px <= 0:
            logger.error("Price unavailable for %s", recommendation.hedge_symbol)
            return None
        qty = round((portfolio_value * recommendation.hedge_size) / px, 4)
        if qty <= 0:
            logger.warning("Computed zero quantity for hedge %s", recommendation.hedge_id)
            return None
        return await broker.place_order(recommendation.hedge_symbol, recommendation.hedge_action, qty, recommendation.order_type)

    async def get_current_portfolio_positions(self) -> List[Position]:
        out: List[Position] = []
        for b in self.brokers.values():
            if hasattr(b, "mock_positions"):
                out.extend(list(getattr(b, "mock_positions").values()))
        return out


# ---------- Core Auto Hedging System ----------

class AutoHedgingSystem:
    def __init__(
        self,
        config: Dict[str, Any],
        db_manager: FinancialDataFramework,
        trading_interface: TradingInterface,
        notification_manager: NotificationManager,
    ):
        self.config = config or {}
        self.hedging_config = self.config.get("auto_hedging", {}) or {}
        self.db_manager = db_manager
        self.trading_interface = trading_interface
        self.notification_manager = notification_manager

        # thresholds
        self.risk_trigger_level = float(self.hedging_config.get("risk_trigger", 7.0))
        self.circuit_breaker_level = float(self.hedging_config.get("circuit_breaker", 9.5))
        self.max_portfolio_exposure = float(self.hedging_config.get("max_exposure", 0.15))
        self.hedge_cost_limit = float(self.hedging_config.get("cost_limit", 50))
        self.min_hedge_size = float(self.hedging_config.get("min_hedge_size", 0.02))
        self.max_hedge_size = float(self.hedging_config.get("max_hedge_size", 0.20))
        self.hedge_effectiveness_threshold = float(self.hedging_config.get("effectiveness_threshold", 0.6))
        self.max_hedge_duration = timedelta(days=int(self.hedging_config.get("max_hedge_duration", 30)))

        # monitoring
        self.monitoring_interval = int(self.hedging_config.get("monitoring_interval", 300))
        self._running = False
        self._stop = threading.Event()
        self.active_hedges: Dict[str, Dict[str, Any]] = {}
        self.circuit_breaker_active = False

        self.hedge_instruments = self._load_hedge_instruments()
        self._ensure_schema()
        logger.info("AutoHedgingSystem initialized.")

    # ----- schema -----
    def _ensure_schema(self) -> None:
        try:
            conn = self.db_manager.get_connection()
            cur = conn.cursor()
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS hedge_executions (
                    hedge_id TEXT PRIMARY KEY,
                    user_id INTEGER,
                    hedge_type TEXT,
                    hedge_symbol TEXT,
                    hedge_action TEXT,
                    hedge_size REAL,
                    risk_reduction REAL,
                    cost_estimate REAL,
                    executed_at TIMESTAMP,
                    portfolio_value REAL,
                    status TEXT,
                    closed_at TIMESTAMP,
                    close_reason TEXT
                )
                """
            )
            conn.commit()
            conn.close()
        except Exception as e:
            logger.warning("Failed to ensure hedge_executions schema: %s", e)

    # ----- lifecycle -----
    def start_risk_monitoring(self) -> None:
        if self._running:
            logger.warning("Risk monitoring already running.")
            return
        self._running = True
        self._stop.clear()
        threading.Thread(target=self._monitor_loop, name="hedge-monitor", daemon=True).start()
        logger.info("Auto-hedging risk monitoring started.")

    def stop_risk_monitoring(self) -> None:
        if not self._running:
            return
        self._running = False
        self._stop.set()
        logger.info("Auto-hedging risk monitoring stop signaled.")

    def _monitor_loop(self) -> None:
        while not self._stop.is_set():
            try:
                self._run_async(self.check_all_portfolios())
                self._run_async(self.update_active_hedges())
                self.cleanup_expired_hedges()
            except Exception as e:
                logger.error("Monitoring error: %s", e, exc_info=True)
            self._stop.wait(self.monitoring_interval)
        logger.info("Monitoring loop exited.")

    def _run_async(self, coro):
        """Run an async coroutine safely from a thread (fresh event loop)."""
        try:
            asyncio.run(coro)
        except RuntimeError:
            # Fallback: create and run custom loop
            loop = asyncio.new_event_loop()
            try:
                asyncio.set_event_loop(loop)
                loop.run_until_complete(coro)
            finally:
                loop.close()

    # ----- hedging instruments -----
    def _load_hedge_instruments(self) -> Dict[str, Dict[str, Any]]:
        return {
            "gold_inverse_etf": {"symbol": "DUST", "correlation": -0.8, "volatility_multiplier": 1.5, "cost_bps": 10},
            "gold_futures": {"symbol": "GC=F", "correlation": -1.0, "volatility_multiplier": 1.0, "cost_bps": 5},
            "treasuries": {"symbol": "TLT", "correlation": -0.3, "volatility_multiplier": 0.5, "cost_bps": 2},
            "vix_etf": {"symbol": "VIXY", "correlation": 0.6, "volatility_multiplier": 2.0, "cost_bps": 20},
            "gold_options": {"symbol": "GLD_PUT", "correlation": -0.9, "volatility_multiplier": 3.0, "cost_bps": 30},
        }

    # ----- portfolio checks -----
    async def check_all_portfolios(self) -> None:
        user_ids = [1]  # replace with DB-driven list
        await asyncio.gather(*(self.check_user_portfolio_risk(uid) for uid in user_ids))

    async def check_user_portfolio_risk(self, user_id: int) -> Optional[List[HedgeRecommendation]]:
        portfolio = await self.get_user_portfolio_data(user_id)
        if not portfolio:
            return None
        metrics = await self.calculate_portfolio_risk(user_id, portfolio)
        if metrics.overall_risk_score >= self.risk_trigger_level:
            recs = await self.generate_hedge_recommendations(user_id, metrics, portfolio)
            if recs:
                if self.hedging_config.get("auto_execute", False):
                    await self.execute_hedge_recommendations(user_id, recs, portfolio.get("total_value", 0.0))
                self.notification_manager.send_email_notification(
                    user_id=user_id,
                    subject=f"High Risk Alert — Score {metrics.overall_risk_score:.2f}/10",
                    message=json.dumps([self.to_recommendation_card(r) for r in recs], indent=2),
                )
                return recs
        return None

    async def get_user_portfolio_data(self, user_id: int) -> Optional[Dict[str, Any]]:
        try:
            acct = await self.trading_interface.default_broker.get_account_info()
            pv = float(acct.get("portfolio_value", 0.0))
            cash = float(acct.get("cash", 0.0))
            # example positions
            pos_gld_px = await self.trading_interface.default_broker.get_current_price("GLD") or 185.0
            pos_iau_px = await self.trading_interface.default_broker.get_current_price("IAU") or 40.0
            positions = [
                asdict(Position("GLD", 100.0, 180.0, pos_gld_px, datetime.utcnow() - timedelta(days=30), "long", pos_gld_px * 100)),
                asdict(Position("IAU", 200.0, 38.0, pos_iau_px, datetime.utcnow() - timedelta(days=60), "long", pos_iau_px * 200)),
            ]
            vix_df = await self.db_manager.get_economic_data("VIXCLS")
            vix = float(vix_df.iloc[-1]["value"]) if isinstance(vix_df, pd.DataFrame) and not vix_df.empty else 20.0
            return {
                "user_id": user_id,
                "total_value": pv,
                "cash": cash,
                "positions": positions,
                "market_volatility_index": vix,
                "overall_liquidity_score": 0.8,
            }
        except Exception as e:
            logger.error("Portfolio fetch failed: %s", e, exc_info=True)
            return None

    # ----- risk metrics -----
    async def calculate_portfolio_risk(self, user_id: int, portfolio_data: Dict[str, Any]) -> RiskMetrics:
        pos = portfolio_data.get("positions", [])
        pv = float(portfolio_data.get("total_value", 0.0))
        if not pos or pv <= 0:
            return RiskMetrics(0, 0, 0, 0, 0, 0, 0, 0, 0)

        symbols = [p["symbol"] for p in pos]
        hist = await self.db_manager.get_multiple_stocks(symbols, outputsize=252)
        rets = self._portfolio_returns(pos, hist)
        if rets.empty or len(rets) < 30:
            return RiskMetrics(0, 0, 0, 0, 0, 0, 0, 0, 5.0)

        with concurrent.futures.ThreadPoolExecutor() as ex:
            var = ex.submit(self._value_at_risk, rets, pv).result()
            mdd = ex.submit(self._max_drawdown, rets).result()
            sharpe = ex.submit(self._sharpe_ratio, rets).result()
            vol = ex.submit(self._annual_vol, rets).result()
            conc = ex.submit(self._concentration_risk, pos).result()
            corr = ex.submit(self._correlation_risk, pos, hist).result()
            tail = ex.submit(self._tail_risk, rets, pv).result()
            liq = ex.submit(self._liquidity_risk, pos).result()

        scores = {
            "var_score": min(10.0, var / (pv * 0.02)),
            "drawdown_score": min(10.0, mdd * 100 / 5),
            "volatility_score": min(10.0, vol * 100 / 10),
            "concentration_score": conc,
            "correlation_score": corr,
            "tail_risk_score": min(10.0, tail / (pv * 0.03)),
            "liquidity_score": liq,
        }
        weights = self._risk_weights(portfolio_data)
        overall = sum(weights[k] * scores[k] for k in scores)
        overall = float(max(0.0, min(10.0, overall)))

        if overall >= self.circuit_breaker_level:
            self.activate_circuit_breaker(f"Score {overall:.2f} >= circuit breaker {self.circuit_breaker_level:.2f}")

        return RiskMetrics(float(var), float(mdd), float(sharpe), float(vol), float(conc), float(corr), float(tail), float(liq), overall)

    def _portfolio_returns(self, positions: List[Dict[str, Any]], hist: Dict[str, pd.DataFrame]) -> pd.Series:
        dates = set()
        for s, df in hist.items():
            if isinstance(df, pd.DataFrame) and not df.empty:
                dates.update(df.index.tolist())
        if not dates:
            return pd.Series(dtype=float)
        idx = pd.DatetimeIndex(sorted(dates))
        total = pd.Series(0.0, index=idx)
        for p in positions:
            s = p["symbol"]
            q = float(p["quantity"])
            if s in hist and not hist[s].empty:
                px = hist[s]["close"].reindex(idx).ffill().fillna(0.0)
                total = total.add(px * q, fill_value=0.0)
        return total.pct_change().dropna()

    def _value_at_risk(self, returns: pd.Series, pv: float, cl: float = 0.95) -> float:
        if returns.empty:
            return 0.0
        q = returns.quantile(1 - cl)
        return float(pv * abs(q))

    def _max_drawdown(self, returns: pd.Series) -> float:
        if returns.empty:
            return 0.0
        cum = (1 + returns).cumprod()
        peak = cum.cummax()
        dd = (cum - peak) / peak
        return float(abs(dd.min()))

    def _sharpe_ratio(self, returns: pd.Series, rf: float = 0.0001) -> float:
        if returns.empty or returns.std() == 0:
            return 0.0
        mu = returns.mean() * 252
        sigma = returns.std() * np.sqrt(252)
        if sigma == 0:
            return 0.0
        return float((mu - rf) / sigma)

    def _annual_vol(self, returns: pd.Series) -> float:
        return float(returns.std() * np.sqrt(252)) if not returns.empty else 0.0

    def _concentration_risk(self, positions: List[Dict[str, Any]]) -> float:
        tv = sum(p["market_value"] for p in positions) or 1.0
        hhi = sum((p["market_value"] / tv) ** 2 for p in positions)
        return float(min(10.0, hhi * 10.0))

    def _correlation_risk(self, positions: List[Dict[str, Any]], hist: Dict[str, pd.DataFrame]) -> float:
        syms = [p["symbol"] for p in positions]
        df = pd.DataFrame({s: hist[s]["close"] for s in syms if s in hist and not hist[s].empty})
        if df.empty or df.shape[1] < 2:
            return 0.0
        ret = df.pct_change().dropna()
        if ret.empty or ret.shape[1] < 2:
            return 0.0
        cm = ret.corr().abs()
        vals = cm.values
        n = vals.shape[0]
        avg = (np.sum(vals) - n) / max(1, (n * (n - 1)))
        return float(min(10.0, avg * 10.0))

    def _tail_risk(self, returns: pd.Series, pv: float, alpha: float = 0.01) -> float:
        if returns.empty:
            return 0.0
        var = returns.quantile(alpha)
        tail = returns[returns <= var]
        if tail.empty:
            return 0.0
        return float(pv * abs(tail.mean()))

    def _liquidity_risk(self, positions: List[Dict[str, Any]]) -> float:
        table = {"GLD": 0.9, "IAU": 0.85, "GC=F": 0.95, "DUST": 0.6, "TLT": 0.9, "SPY": 0.98}
        tv = sum(p["market_value"] for p in positions) or 1.0
        w_liq = sum((p["market_value"] / tv) * table.get(p["symbol"], 0.7) for p in positions)
        return float(min(10.0, max(0.0, (1.0 - w_liq) * 10.0)))

    def _risk_weights(self, portfolio_data: Dict[str, Any]) -> Dict[str, float]:
        w = {"var_score": 0.15, "drawdown_score": 0.15, "volatility_score": 0.15, "concentration_score": 0.15, "correlation_score": 0.15, "tail_risk_score": 0.15, "liquidity_score": 0.10}
        if portfolio_data.get("market_volatility_index", 0) > 25:
            w.update({"volatility_score": 0.20, "tail_risk_score": 0.20, "var_score": 0.20, "concentration_score": 0.10})
        if portfolio_data.get("overall_liquidity_score", 1.0) < 0.5:
            w.update({"liquidity_score": 0.20, "concentration_score": 0.20})
        total = sum(w.values())
        return {k: v / total for k, v in w.items()}

    # ----- hedge generation -----
    async def generate_hedge_recommendations(self, user_id: int, metrics: RiskMetrics, portfolio: Dict[str, Any]) -> List[HedgeRecommendation]:
        recs: List[HedgeRecommendation] = []
        urgency = self._urgency(metrics.overall_risk_score)
        gold_longs = [p for p in portfolio.get("positions", []) if p["symbol"] in {"GLD", "IAU"} and p["position_type"] == "long"]
        if gold_longs:
            main = max(gold_longs, key=lambda p: p["market_value"])
            inv = self.hedge_instruments["gold_inverse_etf"]
            size = min(self.max_hedge_size, (main["market_value"] / max(1.0, portfolio["total_value"])) * 0.5)
            recs.append(HedgeRecommendation(str(uuid.uuid4()), HedgeType.DIRECT_OFFSET, inv["symbol"], "SELL", size, abs(inv["correlation"]), metrics.concentration_risk * 2, inv["cost_bps"], 0.85, f"Offset {main['symbol']} concentration.", urgency))
        if metrics.volatility > 0.02:
            vix = self.hedge_instruments["vix_etf"]
            size = min(self.max_hedge_size, metrics.volatility * 2.0)
            recs.append(HedgeRecommendation(str(uuid.uuid4()), HedgeType.VOLATILITY_HEDGE, vix["symbol"], "BUY", size, 0.7, metrics.volatility * 10, vix["cost_bps"], 0.70, "Hedge volatility spikes.", urgency))
        if metrics.correlation_risk > 6.0 and gold_longs:
            gf = self.hedge_instruments["gold_futures"]
            main = max(gold_longs, key=lambda p: p["market_value"])
            size = min(self.max_hedge_size, main["market_value"] / max(1.0, portfolio["total_value"]) * metrics.correlation_risk / 10.0)
            recs.append(HedgeRecommendation(str(uuid.uuid4()), HedgeType.CORRELATION_HEDGE, gf["symbol"], "SELL", size, abs(gf["correlation"]), metrics.correlation_risk * 1.5, gf["cost_bps"], 0.80, "Short correlated gold futures.", urgency))
        if metrics.overall_risk_score > 8.0 and not gold_longs:
            tr = self.hedge_instruments["treasuries"]
            size = min(self.max_hedge_size, metrics.overall_risk_score / 10.0 * 0.05)
            recs.append(HedgeRecommendation(str(uuid.uuid4()), HedgeType.SECTOR_DIVERSIFICATION, tr["symbol"], "BUY", size, 0.2, metrics.overall_risk_score * 0.5, tr["cost_bps"], 0.60, "Diversify with treasuries.", urgency))
            opt = self.hedge_instruments["gold_options"]
            size = min(self.max_hedge_size, metrics.tail_risk / max(1.0, portfolio["total_value"]) * 0.5)
            recs.append(HedgeRecommendation(str(uuid.uuid4()), HedgeType.TAIL_RISK_HEDGE, opt["symbol"], "BUY", size, 0.8, metrics.tail_risk * 0.05, opt["cost_bps"], 0.75, "Buy GLD puts for tail protection.", urgency))
        if metrics.liquidity_risk > 6.0:
            tr = self.hedge_instruments["treasuries"]
            size = min(self.max_hedge_size, metrics.liquidity_risk / 10.0 * 0.1)
            recs.append(HedgeRecommendation(str(uuid.uuid4()), HedgeType.LIQUIDITY_HEDGE, tr["symbol"], "BUY", size, 0.9, metrics.liquidity_risk * 1.0, tr["cost_bps"], 0.85, "Shift to highly liquid assets.", urgency))

        # filter bounds & costs
        recs = [r for r in recs if self.min_hedge_size <= r.hedge_size <= self.max_hedge_size and r.cost_estimate <= self.hedge_cost_limit]
        # sort by expected benefit per cost
        recs.sort(key=lambda r: (r.risk_reduction * r.confidence) / max(r.cost_estimate / 100.0, 0.001), reverse=True)
        return recs[:3]

    def _urgency(self, score: float) -> RiskLevel:
        if score >= self.circuit_breaker_level:
            return RiskLevel.EXTREME
        if score >= self.risk_trigger_level:
            return RiskLevel.CRITICAL
        if score >= self.risk_trigger_level - 2.0:
            return RiskLevel.HIGH
        if score >= self.risk_trigger_level - 4.0:
            return RiskLevel.MODERATE
        return RiskLevel.LOW

    # ----- execution & tracking -----
    async def execute_hedge_recommendations(self, user_id: int, recs: List[HedgeRecommendation], portfolio_value: float) -> None:
        executed: List[Dict[str, Any]] = []
        for r in recs:
            if not self._should_execute(r):
                continue
            oid = await self.trading_interface.execute_hedge(r, portfolio_value)
            if oid:
                record = {
                    "hedge_id": r.hedge_id,
                    "user_id": user_id,
                    "recommendation": asdict(r),
                    "executed_order_id": oid,
                    "executed_at": datetime.utcnow().isoformat(),
                    "portfolio_value_at_execution": portfolio_value,
                    "status": "active",
                }
                executed.append(record)
                self.active_hedges[r.hedge_id] = {
                    "user_id": user_id,
                    "recommendation": r,
                    "status": "active",
                    "executed_at": datetime.utcnow(),
                    "initial_risk": 10.0,  # optionally store snapshot
                }
                self.notification_manager.send_email_notification(user_id=user_id, subject=f"Hedge Executed: {r.hedge_type.value}", message=json.dumps(record))
        if executed:
            await self._store_executions(executed)

    def _should_execute(self, r: HedgeRecommendation) -> bool:
        return (r.confidence >= self.hedge_effectiveness_threshold and r.cost_estimate <= self.hedge_cost_limit and self.min_hedge_size <= r.hedge_size <= self.max_hedge_size and not self.circuit_breaker_active)

    async def update_active_hedges(self) -> None:
        now = datetime.utcnow()
        remove: List[str] = []
        for hid, h in list(self.active_hedges.items()):
            r: HedgeRecommendation = h["recommendation"]
            if r.expiry_date and now > r.expiry_date:
                await self._close_hedge(hid, reason="Expired")
                remove.append(hid)
                continue
            perf = await self._hedge_performance(h)
            h["performance"] = perf
            h["last_updated"] = now
            if perf.get("effectiveness", 0.0) < 0.3:
                await self._adjust_hedge(hid, h)
                remove.append(hid)
        for hid in remove:
            self.active_hedges.pop(hid, None)

    async def _adjust_hedge(self, hedge_id: str, h: Dict[str, Any]) -> None:
        await self._close_hedge(hedge_id, partial=True, reason="Underperforming")
        uid = h["user_id"]
        port = await self.get_user_portfolio_data(uid)
        if not port:
            return
        metrics = await self.calculate_portfolio_risk(uid, port)
        recs = await self.generate_hedge_recommendations(uid, metrics, port)
        if recs:
            await self.execute_hedge_recommendations(uid, recs[:1], port.get("total_value", 0.0))

    async def _close_hedge(self, hedge_id: str, partial: bool = False, reason: str = "System") -> None:
        if hedge_id not in self.active_hedges:
            return
        h = self.active_hedges[hedge_id]
        r: HedgeRecommendation = h["recommendation"]
        close_action = "SELL" if r.hedge_action == "BUY" else "BUY"
        pv = h.get("portfolio_value_at_execution", 100_000.0)
        close_rec = HedgeRecommendation(str(uuid.uuid4()), r.hedge_type, r.hedge_symbol, close_action, r.hedge_size, r.hedge_ratio, 0.0, r.cost_estimate, 1.0, f"Close {r.hedge_id}: {reason}", RiskLevel.MODERATE)
        oid = await self.trading_interface.execute_hedge(close_rec, pv)
        h["status"] = "adjusted" if partial else "closed"
        h["closed_at"] = datetime.utcnow()
        h["close_order_id"] = oid
        h["close_reason"] = reason
        await self._store_executions([h])
        if not partial:
            self.active_hedges.pop(hedge_id, None)

    async def _hedge_performance(self, h: Dict[str, Any]) -> Dict[str, float]:
        try:
            uid = h["user_id"]
            port = await self.get_user_portfolio_data(uid)
            metrics = await self.calculate_portfolio_risk(uid, port) if port else RiskMetrics(0,0,0,0,0,0,0,0,0)
            init = float(h.get("initial_risk", 0.0))
            cur = float(metrics.overall_risk_score)
            reduction = max(0.0, init - cur)
            eff = min(1.0, reduction / max(1e-6, h["recommendation"].risk_reduction))
            return {"effectiveness": float(eff), "risk_reduction_achieved": float(reduction), "current_portfolio_risk": float(cur)}
        except Exception as e:
            logger.error("Hedge performance failed: %s", e)
            return {"effectiveness": 0.0, "risk_reduction_achieved": 0.0, "current_portfolio_risk": 0.0}

    # ----- persistence -----
    async def _store_executions(self, items: List[Dict[str, Any]]) -> None:
        try:
            conn = self.db_manager.get_connection()
            cur = conn.cursor()
            cur.executemany(
                """
                INSERT OR REPLACE INTO hedge_executions
                (hedge_id, user_id, hedge_type, hedge_symbol, hedge_action, hedge_size, risk_reduction, cost_estimate, executed_at, portfolio_value, status, closed_at, close_reason)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        d.get("hedge_id") or d["recommendation"]["hedge_id"],
                        d.get("user_id"),
                        (d.get("recommendation") or {}).get("hedge_type", {}).get("value") if isinstance(d.get("recommendation"), dict) else d["recommendation"].hedge_type.value,  # type: ignore
                        (d.get("recommendation") or {}).get("hedge_symbol") if isinstance(d.get("recommendation"), dict) else d["recommendation"].hedge_symbol,  # type: ignore
                        (d.get("recommendation") or {}).get("hedge_action") if isinstance(d.get("recommendation"), dict) else d["recommendation"].hedge_action,  # type: ignore
                        (d.get("recommendation") or {}).get("hedge_size") if isinstance(d.get("recommendation"), dict) else d["recommendation"].hedge_size,  # type: ignore
                        (d.get("recommendation") or {}).get("risk_reduction") if isinstance(d.get("recommendation"), dict) else d["recommendation"].risk_reduction,  # type: ignore
                        (d.get("recommendation") or {}).get("cost_estimate") if isinstance(d.get("recommendation"), dict) else d["recommendation"].cost_estimate,  # type: ignore
                        d.get("executed_at") or datetime.utcnow().isoformat(),
                        d.get("portfolio_value_at_execution", 0.0),
                        d.get("status", "active"),
                        d.get("closed_at"),
                        d.get("close_reason"),
                    )
                    for d in items
                ],
            )
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error("Persist executions failed: %s", e)

    def cleanup_expired_hedges(self) -> None:
        now = datetime.utcnow()
        for hid, h in list(self.active_hedges.items()):
            rec: HedgeRecommendation = h["recommendation"]
            if rec.expiry_date and now > rec.expiry_date:
                self.active_hedges.pop(hid, None)

    # ----- circuit breaker -----
    def activate_circuit_breaker(self, reason: str) -> None:
        if self.circuit_breaker_active:
            return
        self.circuit_breaker_active = True
        logger.critical("CIRCUIT BREAKER ACTIVATED — %s", reason)
        self.notification_manager.send_system_alert("Circuit Breaker", f"Reason: {reason}")

    # ----- dashboard helpers -----
    @staticmethod
    def to_recommendation_card(r: HedgeRecommendation) -> Dict[str, Any]:
        return {
            "hedge_id": r.hedge_id,
            "type": r.hedge_type.value,
            "symbol": r.hedge_symbol,
            "action": r.hedge_action,
            "size_pct": round(r.hedge_size * 100, 2),
            "expected_reduction": round(r.risk_reduction, 2),
            "cost_bps": r.cost_estimate,
            "confidence": round(r.confidence, 2),
            "urgency": r.urgency.name,
            "expires": r.expiry_date.isoformat() if r.expiry_date else None,
        }

    @staticmethod
    def to_summary_block(metrics: RiskMetrics) -> Dict[str, Any]:
        return {
            "overall_score": round(metrics.overall_risk_score, 2),
            "volatility": round(metrics.volatility * 100, 2),
            "var": round(metrics.portfolio_var, 2),
            "drawdown": round(metrics.max_drawdown * 100, 2),
            "liquidity_risk": round(metrics.liquidity_risk, 2),
        }


# ---------- Demo ----------
if __name__ == "__main__":
    async def _demo():
        class MockFDF(FinancialDataFramework):
            pass

        fdf = MockFDF()
        broker = AlpacaBroker("key", "secret")
        ti = TradingInterface({"alpaca": broker})
        notifier = NotificationManager()
        cfg = {"auto_hedging": {"auto_execute": True, "monitoring_interval": 3}}

        ahs = AutoHedgingSystem(cfg, fdf, ti, notifier)
        # seed a couple positions
        broker.mock_positions["GLD"] = Position("GLD", 100.0, 180.0, 185.0, datetime.utcnow(), "long", 18_500.0)
        broker.mock_positions["IAU"] = Position("IAU", 200.0, 38.0, 40.0, datetime.utcnow(), "long", 8_000.0)

        # single cycle
        port = await ahs.get_user_portfolio_data(1)
        metrics = await ahs.calculate_portfolio_risk(1, port or {})
        recs = await ahs.generate_hedge_recommendations(1, metrics, port or {})
        print("Summary:", AutoHedgingSystem.to_summary_block(metrics))
        print("Cards:", [AutoHedgingSystem.to_recommendation_card(r) for r in recs])

        # start monitor briefly
        ahs.start_risk_monitoring()
        await asyncio.sleep(6)
        ahs.stop_risk_monitoring()

    asyncio.run(_demo())
