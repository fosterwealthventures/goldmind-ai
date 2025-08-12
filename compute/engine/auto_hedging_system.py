""" 
Enhanced Automated Hedging and Risk Management System for GoldMIND AI
- Real-time position monitoring with circuit breakers
- Dynamic correlation-based hedging
- Multi-broker API integration
- Hedge effectiveness tracking
- Asynchronous architecture
- Integrates with FinancialDataFramework for data and persistence
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any, Callable
import logging
import json
from dataclasses import dataclass, asdict
from enum import Enum
import asyncio
import aiohttp
import concurrent.futures
import time
from scipy.stats import pearsonr, norm # Added norm for VaR calculation
from abc import ABC, abstractmethod
import uuid # For generating unique IDs

logger = logging.getLogger(__name__)

# Import FinancialDataFramework for database interaction and data fetching
try:
    from financial_data_framework import FinancialDataFramework
except ImportError:
    logger.critical("❌ Could not import FinancialDataFramework. Please ensure 'financial_data_framework.py' is accessible.")
    class FinancialDataFramework: # Mock for parsing
        def __init__(self, *args, **kwargs): pass
        def get_connection(self): return None
        async def get_stock_data(self, *args, **kwargs): return pd.DataFrame()
        async def get_economic_data(self, *args, **kwargs): return pd.DataFrame()

# Import NotificationManager (placeholder if not yet created)
try:
    # from notification_system import NotificationManager # Uncomment if you have this module
    class NotificationManager: # Mock for parsing
        def __init__(self, *args, **kwargs): pass
        def send_email_notification(self, *args, **kwargs): logger.info("Mock Email Notification Sent.")
        def send_system_alert(self, *args, **kwargs): logger.info("Mock System Alert Sent.")
except ImportError:
    logger.warning("NotificationManager not found. Using a mock for alerts.")
    class NotificationManager: # Mock for parsing
        def __init__(self, *args, **kwargs): pass
        def send_email_notification(self, *args, **kwargs): logger.info("Mock Email Notification Sent.")
        def send_system_alert(self, *args, **kwargs): logger.info("Mock System Alert Sent.")


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
    market_value: float # current_price * quantity

@dataclass
class HedgeRecommendation:
    hedge_id: str # Unique ID for this specific hedge recommendation
    hedge_type: HedgeType
    hedge_symbol: str
    hedge_action: str  # BUY, SELL
    hedge_size: float  # Percentage of portfolio value to hedge, e.g., 0.05 for 5%
    hedge_ratio: float  # Hedge ratio (0-1), e.g., 0.8 for 80% effectiveness
    risk_reduction: float  # Expected risk reduction % (e.g., 0.1 for 10%)
    cost_estimate: float  # Estimated cost in basis points (e.g., 50 for 0.5%)
    confidence: float  # Confidence in hedge effectiveness (0.0-1.0)
    reasoning: str
    urgency: RiskLevel
    expiry_date: Optional[datetime] = None
    order_type: OrderType = OrderType.MARKET

@dataclass
class RiskMetrics:
    portfolio_var: float  # Value at Risk ($)
    max_drawdown: float  # Maximum drawdown (%)
    sharpe_ratio: float  # Risk-adjusted returns
    volatility: float  # Portfolio volatility (%)
    concentration_risk: float  # Concentration risk score (0-10)
    correlation_risk: float  # Correlation risk score (0-10)
    tail_risk: float  # Tail risk measure (e.g., Conditional VaR) ($)
    liquidity_risk: float  # Liquidity risk score (0-10)
    overall_risk_score: float  # Combined risk score (0-10)

class BrokerAPI(ABC):
    """Abstract base class for broker integrations."""
    
    @abstractmethod
    async def place_order(self, symbol: str, action: str, quantity: float, order_type: OrderType) -> Optional[str]:
        """Places a trade order. Returns order ID or None on failure."""
        pass
    
    @abstractmethod
    async def cancel_order(self, order_id: str) -> bool:
        """Cancels an existing order. Returns True on success."""
        pass
    
    @abstractmethod
    async def get_position(self, symbol: str) -> Optional[Position]:
        """Gets current position for a symbol."""
        pass

    @abstractmethod
    async def get_account_info(self) -> Dict[str, Any]:
        """Gets overall account information (e.g., buying power, equity)."""
        pass

    @abstractmethod
    async def get_current_price(self, symbol: str) -> Optional[float]:
        """Gets the current market price for a symbol."""
        pass

class AlpacaBroker(BrokerAPI):
    """Mock implementation for Alpaca Markets API."""
    
    def __init__(self, api_key: str, secret_key: str):
        self.api_key = api_key
        self.secret_key = secret_key
        self.base_url = "https://paper-api.alpaca.markets/v2" # Using paper trading URL for mock
        self.headers = {
            "APCA-API-KEY-ID": self.api_key,
            "APCA-API-SECRET-KEY": self.secret_key,
            "Content-Type": "application/json"
        }
        self.mock_positions: Dict[str, Position] = {}
        self.mock_orders: Dict[str, Dict] = {}
        logger.info("Mock AlpacaBroker initialized.")
    
    async def place_order(self, symbol: str, action: str, quantity: float, order_type: OrderType) -> Optional[str]:
        """Simulates placing a trade order."""
        order_id = str(uuid.uuid4())
        logger.info(f"Mock Alpaca: Placing {action} {quantity} of {symbol} ({order_type.value}) - Order ID: {order_id}")
        
        # Simulate execution
        current_price = await self.get_current_price(symbol) or 100.0 # Fallback price
        
        self.mock_orders[order_id] = {
            "id": order_id,
            "symbol": symbol,
            "qty": quantity,
            "side": action.lower(),
            "type": order_type.value,
            "status": "filled", # Assume immediate fill for mock
            "filled_avg_price": current_price,
            "submitted_at": datetime.utcnow().isoformat()
        }
        
        # Update mock position
        if symbol not in self.mock_positions:
            self.mock_positions[symbol] = Position(
                symbol=symbol, quantity=0.0, entry_price=0.0, current_price=current_price,
                entry_time=datetime.utcnow(), position_type='long', market_value=0.0
            )
        
        pos = self.mock_positions[symbol]
        if action.lower() == 'buy':
            new_quantity = pos.quantity + quantity
            new_entry_price = ((pos.entry_price * pos.quantity) + (current_price * quantity)) / new_quantity if new_quantity > 0 else current_price
            pos.quantity = new_quantity
            pos.entry_price = new_entry_price
            pos.position_type = 'long'
        elif action.lower() == 'sell':
            new_quantity = pos.quantity - quantity
            if new_quantity < 0: # Selling more than held, assume shorting
                pos.position_type = 'short'
            else:
                pos.position_type = 'long' # Still long or closed
            pos.quantity = new_quantity
            # Entry price logic for shorts or closing is more complex, simplified for mock
            pos.entry_price = current_price # Simplified
        pos.current_price = current_price
        pos.market_value = pos.current_price * pos.quantity
        
        return order_id
    
    async def cancel_order(self, order_id: str) -> bool:
        """Simulates canceling an order."""
        if order_id in self.mock_orders and self.mock_orders[order_id]['status'] != 'filled':
            self.mock_orders[order_id]['status'] = 'canceled'
            logger.info(f"Mock Alpaca: Canceled order {order_id}.")
            return True
        logger.warning(f"Mock Alpaca: Order {order_id} not found or already filled/canceled.")
        return False
    
    async def get_position(self, symbol: str) -> Optional[Position]:
        """Simulates getting a position."""
        pos = self.mock_positions.get(symbol)
        if pos:
            # Update current price for mock position
            pos.current_price = await self.get_current_price(symbol) or pos.current_price
            pos.market_value = pos.current_price * pos.quantity
        return pos

    async def get_account_info(self) -> Dict[str, Any]:
        """Simulates getting account info."""
        total_portfolio_value = sum(pos.market_value for pos in self.mock_positions.values())
        return {
            "cash": 50000.0, # Mock cash
            "portfolio_value": total_portfolio_value,
            "buying_power": 100000.0, # Mock buying power
            "equity": 50000.0 + total_portfolio_value,
            "last_updated": datetime.utcnow().isoformat()
        }

    async def get_current_price(self, symbol: str) -> Optional[float]:
        """Simulates getting current market price."""
        # This would ideally call the FinancialDataFramework
        if symbol == 'GLD': return 185.50 + np.random.uniform(-1, 1)
        if symbol == 'IAU': return 40.00 + np.random.uniform(-0.5, 0.5)
        if symbol == 'DUST': return 15.00 + np.random.uniform(-0.2, 0.2)
        if symbol == 'GDXV': return 25.00 + np.random.uniform(-0.3, 0.3)
        if symbol == 'TLT': return 95.00 + np.random.uniform(-0.5, 0.5)
        if symbol == 'SPY': return 500.00 + np.random.uniform(-2, 2)
        return 100.0 + np.random.uniform(-1, 1) # Default mock price

class TradingInterface:
    """Unified trading interface for multi-broker support."""
    
    def __init__(self, brokers: Dict[str, BrokerAPI]):
        self.brokers = brokers
        if not self.brokers:
            raise ValueError("TradingInterface requires at least one broker API.")
        self.default_broker = next(iter(brokers.values()))
        logger.info(f"TradingInterface initialized with brokers: {list(brokers.keys())}")
    
    async def execute_hedge(self, recommendation: HedgeRecommendation, portfolio_value: float) -> Optional[str]:
        """Execute hedge trade using appropriate broker."""
        broker = self.select_broker(recommendation.hedge_symbol)
        if not broker:
            logger.error(f"No suitable broker found for {recommendation.hedge_symbol}.")
            return None

        current_price = await broker.get_current_price(recommendation.hedge_symbol)
        if not current_price or current_price <= 0:
            logger.error(f"Could not get current price for {recommendation.hedge_symbol}. Cannot execute hedge.")
            return None

        quantity = self.calculate_quantity(recommendation, portfolio_value, current_price)
        if quantity <= 0:
            logger.warning(f"Calculated quantity for hedge {recommendation.hedge_id} is zero or negative. Skipping execution.")
            return None

        try:
            order_id = await broker.place_order(
                symbol=recommendation.hedge_symbol,
                action=recommendation.hedge_action,
                quantity=quantity,
                order_type=recommendation.order_type
            )
            return order_id
        except Exception as e:
            logger.error(f"Error executing hedge {recommendation.hedge_id} via broker: {e}", exc_info=True)
            return None
    
    def select_broker(self, symbol: str) -> BrokerAPI:
        """Select broker based on instrument type or other rules."""
        # This can be expanded with more sophisticated routing logic
        # For now, just use the default broker
        return self.default_broker
    
    def calculate_quantity(self, recommendation: HedgeRecommendation, portfolio_value: float, current_price: float) -> float:
        """Calculate position size based on dollar value and current price."""
        if current_price <= 0:
            return 0.0
        dollar_amount = portfolio_value * recommendation.hedge_size # hedge_size is already a fraction (e.g., 0.05)
        quantity = dollar_amount / current_price
        return round(quantity, 4) # Round to avoid fractional shares if not supported

    async def get_current_portfolio_positions(self) -> List[Position]:
        """Aggregates positions from all connected brokers."""
        all_positions = []
        for broker_name, broker in self.brokers.items():
            try:
                account_info = await broker.get_account_info()
                # This is a mock, real brokers have specific position endpoints
                # For mock, we'll just use the mock_positions from AlpacaBroker
                if hasattr(broker, 'mock_positions'):
                    for symbol, pos in broker.mock_positions.items():
                        all_positions.append(pos)
            except Exception as e:
                logger.error(f"Failed to get positions from {broker_name}: {e}")
        return all_positions


class AutoHedgingSystem:
    """
    Automated Hedging and Risk Management System.
    Monitors portfolio risk, generates, and executes hedge recommendations.
    """
    def __init__(self, config: Dict, db_manager: FinancialDataFramework, 
                 trading_interface: TradingInterface, notification_manager: NotificationManager):
        self.config = config
        self.hedging_config = config.get('auto_hedging', {})
        self.db_manager = db_manager
        self.trading_interface = trading_interface
        self.notification_manager = notification_manager
        
        # Risk thresholds
        self.risk_trigger_level = self.hedging_config.get('risk_trigger', 7.0) # Score 0-10
        self.max_portfolio_exposure = self.hedging_config.get('max_exposure', 0.15) # 15%
        self.hedge_cost_limit = self.hedging_config.get('cost_limit', 50)  # basis points (0.5%)
        self.circuit_breaker_level = self.hedging_config.get('circuit_breaker', 9.5) # Score 0-10
        
        # Hedging parameters
        self.min_hedge_size = self.hedging_config.get('min_hedge_size', 0.02)  # 2% of portfolio value
        self.max_hedge_size = self.hedging_config.get('max_hedge_size', 0.20)  # 20% of portfolio value
        self.hedge_effectiveness_threshold = 0.6 # 60% expected risk reduction
        self.max_hedge_duration = timedelta(days=self.hedging_config.get('max_hedge_duration', 30))
        
        # Monitoring
        self.is_monitoring = False
        self.shutdown_event = threading.Event()
        self.monitoring_interval = self.hedging_config.get('monitoring_interval', 300)  # 5 minutes
        self.active_hedges: Dict[str, Dict] = {} # Stores currently active hedges
        self.circuit_breaker_active = False
        
        # Hedge instruments (simplified examples, would be dynamic in real system)
        self.hedge_instruments = self.load_hedge_instruments()
        
        logger.info("AutoHedgingSystem initialized.")
    
    def load_hedge_instruments(self) -> Dict[str, Dict]:
        """Loads available hedge instruments and their characteristics."""
        # In a real system, this would come from a configuration or a dedicated instrument service.
        return {
            'gold_inverse_etf': {
                'symbol': 'DUST', # Direxion Daily Gold Miners Index Bear 2X Shares
                'correlation': -0.8, # Negative correlation to gold miners
                'volatility_multiplier': 1.5, # How much more volatile than underlying
                'cost_bps': 10 # 0.1% cost
            },
            'gold_futures': {
                'symbol': 'GC=F', # Gold Futures
                'correlation': -1.0, # Direct inverse for shorting futures
                'volatility_multiplier': 1.0,
                'cost_bps': 5
            },
            'treasuries': {
                'symbol': 'TLT', # iShares 20+ Year Treasury Bond ETF
                'correlation': -0.3, # Weak negative correlation to equities/commodities
                'volatility_multiplier': 0.5,
                'cost_bps': 2
            },
            'vix_etf': {
                'symbol': 'VIXY', # ProShares VIX Short-Term Futures ETF
                'correlation': 0.6, # Positive correlation to market fear/volatility
                'volatility_multiplier': 2.0,
                'cost_bps': 20
            },
            'gold_options': {
                'symbol': 'GLD_PUT', # Placeholder for GLD Put options
                'correlation': -0.9,
                'volatility_multiplier': 3.0,
                'cost_bps': 30 # Higher cost for options
            }
        }
    
    def start_risk_monitoring(self):
        """Start continuous risk monitoring in a separate thread."""
        if self.is_monitoring:
            logger.warning("Risk monitoring already running.")
            return
        
        self.is_monitoring = True
        self.shutdown_event.clear()
        
        monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True, name="HedgingMonitor")
        monitoring_thread.start()
        logger.info("Started auto-hedging risk monitoring.")
    
    def stop_risk_monitoring(self):
        """Stop risk monitoring gracefully."""
        if not self.is_monitoring:
            return
        
        self.is_monitoring = False
        self.shutdown_event.set()
        logger.info("Signaled auto-hedging risk monitoring to stop.")
    
    def _monitoring_loop(self):
        """Asynchronous monitoring loop running in a separate thread."""
        # This loop needs to run an async function, so we use asyncio.run
        # or manage the event loop explicitly. For a simple thread,
        # we can just run the async function within the loop.
        while self.is_monitoring and not self.shutdown_event.is_set():
            try:
                start_time = time.time()
                
                # Check if circuit breaker is active
                if self.circuit_breaker_active:
                    logger.warning("Circuit breaker active - skipping monitoring cycle.")
                    self.shutdown_event.wait(60) # Wait a bit longer
                    continue
                
                # Run async checks
                asyncio.run(self.check_all_portfolios())
                asyncio.run(self.update_active_hedges())
                
                self.cleanup_expired_hedges()
                
                # Calculate processing time and sleep
                processing_time = time.time() - start_time
                sleep_time = max(1, self.monitoring_interval - processing_time)
                self.shutdown_event.wait(sleep_time)
                
            except Exception as e:
                logger.error(f"Risk monitoring loop error: {e}", exc_info=True)
                self.shutdown_event.wait(60) # Wait on error
        logger.info("Auto-hedging monitoring thread stopped.")

    async def check_all_portfolios(self):
        """Check risk levels for all user portfolios asynchronously."""
        try:
            # In a real system, you'd get a list of user_ids from your user management.
            # For now, we'll assume a single user or get from a mock user list.
            # A more robust approach would query the DB for active users with positions.
            mock_user_ids = [1, 2] # Example user IDs
            
            tasks = [self.check_user_portfolio_risk(user_id) for user_id in mock_user_ids]
            await asyncio.gather(*tasks)
            
        except Exception as e:
            logger.error(f"Check all portfolios error: {e}", exc_info=True)
    
    async def check_user_portfolio_risk(self, user_id: int) -> Optional[List[HedgeRecommendation]]:
        """Check risk level for a specific user portfolio and trigger hedging if needed."""
        try:
            # Calculate current risk metrics
            portfolio_data = await self.get_user_portfolio_data(user_id)
            if not portfolio_data:
                logger.info(f"No portfolio data found for user {user_id}. Skipping risk check.")
                return None

            risk_metrics = await self.calculate_portfolio_risk(user_id, portfolio_data)
            
            if risk_metrics.overall_risk_score >= self.risk_trigger_level:
                logger.warning(f"High risk detected for user {user_id}: {risk_metrics.overall_risk_score:.2f}/{self.risk_trigger_level}")
                
                # Generate hedge recommendations
                hedge_recommendations = await self.generate_hedge_recommendations(user_id, risk_metrics, portfolio_data)
                
                if hedge_recommendations:
                    # Execute automatic hedges if configured
                    if self.hedging_config.get('auto_execute', False):
                        await self.execute_hedge_recommendations(user_id, hedge_recommendations, portfolio_data.get('total_value', 0))
                    
                    # Send notifications
                    self.notification_manager.send_email_notification(
                        user_id=user_id,
                        subject=f"High Risk Alert for Portfolio (Score: {risk_metrics.overall_risk_score:.2f})",
                        message=f"Your portfolio risk is high. Recommended hedges: {json.dumps([asdict(h) for h in hedge_recommendations], indent=2)}"
                    )
                    
                    return hedge_recommendations
            
            return None
            
        except Exception as e:
            logger.error(f"User portfolio risk check error for user {user_id}: {e}", exc_info=True)
            return None
    
    async def calculate_portfolio_risk(self, user_id: int, portfolio_data: Dict) -> RiskMetrics:
        """Calculate comprehensive portfolio risk metrics asynchronously."""
        try:
            positions = portfolio_data.get('positions', [])
            total_value = portfolio_data.get('total_value', 0.0)

            if not positions or total_value <= 0:
                return RiskMetrics(0, 0, 0, 0, 0, 0, 0, 0, 0) # Return zero risk for empty portfolio
            
            # Fetch historical data for all symbols in portfolio for calculations
            symbols = [pos['symbol'] for pos in positions]
            historical_data_dict = await self.db_manager.get_multiple_stocks(symbols, outputsize=252) # 1 year of daily data

            # Prepare data for risk calculations
            portfolio_returns = self._calculate_portfolio_returns(positions, historical_data_dict)
            
            if portfolio_returns.empty or len(portfolio_returns) < 30: # Need sufficient data for stats
                logger.warning(f"Insufficient historical data for user {user_id} portfolio risk calculation.")
                return RiskMetrics(0, 0, 0, 0, 0, 0, 0, 0, 5) # Default moderate risk

            # Calculate risk components in parallel
            with concurrent.futures.ThreadPoolExecutor() as executor:
                # Value at Risk (VaR) - 95% confidence, 1-day
                var = executor.submit(self.calculate_value_at_risk, portfolio_returns, total_value).result()
                
                # Max Drawdown
                max_drawdown = executor.submit(self.calculate_max_drawdown, portfolio_returns).result()
                
                # Sharpe Ratio (requires risk-free rate, simplified for now)
                sharpe_ratio = executor.submit(self.calculate_sharpe_ratio, portfolio_returns).result()
                
                # Volatility (annualized standard deviation of returns)
                volatility = executor.submit(self.calculate_portfolio_volatility, portfolio_returns).result()
                
                # Concentration Risk
                concentration_risk = executor.submit(self.calculate_concentration_risk, positions).result()
                
                # Correlation Risk
                correlation_risk = executor.submit(self.calculate_correlation_risk, positions, historical_data_dict).result()
                
                # Tail Risk (Conditional VaR / Expected Shortfall)
                tail_risk = executor.submit(self.calculate_tail_risk, portfolio_returns, total_value).result()
                
                # Liquidity Risk
                liquidity_risk = executor.submit(self.calculate_liquidity_risk, positions).result()
            
            # Calculate overall risk score (0-10 scale)
            risk_components_scores = {
                'var_score': min(10.0, var / (total_value * 0.02)), # VaR as % of portfolio, scaled
                'drawdown_score': min(10.0, max_drawdown * 100 / 5), # 5% drawdown = 1 score point
                'volatility_score': min(10.0, volatility * 100 / 10), # 10% volatility = 1 score point
                'concentration_score': concentration_risk,
                'correlation_score': correlation_risk,
                'tail_risk_score': min(10.0, tail_risk / (total_value * 0.03)), # Tail VaR as % of portfolio, scaled
                'liquidity_score': liquidity_risk
            }
            
            # Dynamic weights for overall risk score
            weights = self.get_risk_weights(portfolio_data)
            
            overall_risk_score = sum(
                weights.get(component_key, 0) * score 
                for component_key, score in risk_components_scores.items()
            )
            overall_risk_score = min(10.0, max(0.0, overall_risk_score)) # Clamp 0-10

            # Apply circuit breaker for extreme risk
            if overall_risk_score >= self.circuit_breaker_level:
                self.activate_circuit_breaker(f"Portfolio risk score {overall_risk_score:.2f} exceeded circuit breaker level.")
            
            return RiskMetrics(
                portfolio_var=float(var),
                max_drawdown=float(max_drawdown),
                sharpe_ratio=float(sharpe_ratio),
                volatility=float(volatility),
                concentration_risk=float(concentration_risk),
                correlation_risk=float(correlation_risk),
                tail_risk=float(tail_risk),
                liquidity_risk=float(liquidity_risk),
                overall_risk_score=float(overall_risk_score)
            )
            
        except Exception as e:
            logger.error(f"Portfolio risk calculation error for user {user_id}: {e}", exc_info=True)
            return RiskMetrics(0, 0, 0, 0, 0, 0, 0, 0, 5.0)  # Default moderate risk on error

    def _calculate_portfolio_returns(self, positions: List[Dict], historical_data_dict: Dict[str, pd.DataFrame]) -> pd.Series:
        """Calculates historical daily returns of the portfolio."""
        if not positions:
            return pd.Series([])

        # Get a common date range for all assets
        all_dates = []
        for symbol in historical_data_dict:
            if not historical_data_dict[symbol].empty:
                all_dates.extend(historical_data_dict[symbol].index.tolist())
        
        if not all_dates:
            return pd.Series([])

        common_dates = pd.to_datetime(sorted(list(set(all_dates))))
        common_dates = common_dates[common_dates <= datetime.now()] # Only past dates
        
        if len(common_dates) < 2:
            return pd.Series([])

        portfolio_daily_values = pd.Series(0.0, index=common_dates)

        for pos in positions:
            symbol = pos['symbol']
            quantity = pos['quantity']
            
            if symbol in historical_data_dict and not historical_data_dict[symbol].empty:
                asset_prices = historical_data_dict[symbol]['close'].reindex(common_dates, method='ffill').fillna(0)
                portfolio_daily_values += asset_prices * quantity
        
        # Calculate daily returns from portfolio value
        portfolio_returns = portfolio_daily_values.pct_change().dropna()
        return portfolio_returns

    def calculate_value_at_risk(self, portfolio_returns: pd.Series, portfolio_value: float, confidence_level: float = 0.95) -> float:
        """Calculate Value at Risk (VaR) using historical simulation."""
        if portfolio_returns.empty: return 0.0
        
        # Sort returns and find the percentile
        sorted_returns = portfolio_returns.sort_values()
        var_return = sorted_returns.quantile(1 - confidence_level)
        
        # VaR in dollar terms
        var_dollar = portfolio_value * abs(var_return)
        return var_dollar

    def calculate_max_drawdown(self, portfolio_returns: pd.Series) -> float:
        """Calculate Maximum Drawdown."""
        if portfolio_returns.empty: return 0.0
        
        cumulative_returns = (1 + portfolio_returns).cumprod()
        peak = cumulative_returns.expanding(min_periods=1).max()
        drawdown = (cumulative_returns - peak) / peak
        return abs(drawdown.min())

    def calculate_sharpe_ratio(self, portfolio_returns: pd.Series, risk_free_rate: float = 0.0001) -> float:
        """Calculate Sharpe Ratio (annualized)."""
        if portfolio_returns.empty or portfolio_returns.std() == 0: return 0.0
        
        # Assuming daily returns, annualize
        annualized_return = portfolio_returns.mean() * 252 # 252 trading days
        annualized_volatility = portfolio_returns.std() * np.sqrt(252)
        
        if annualized_volatility == 0: return 0.0
        
        sharpe = (annualized_return - risk_free_rate) / annualized_volatility
        return sharpe

    def calculate_portfolio_volatility(self, portfolio_returns: pd.Series) -> float:
        """Calculate annualized portfolio volatility."""
        if portfolio_returns.empty: return 0.0
        return portfolio_returns.std() * np.sqrt(252) # Annualized

    def calculate_concentration_risk(self, positions: List[Dict]) -> float:
        """Calculate concentration risk score (0-10) using Herfindahl-Hirschman Index (HHI)."""
        if not positions: return 0.0
        
        total_value = sum(pos['market_value'] for pos in positions)
        if total_value == 0: return 0.0

        hhi = 0.0
        for pos in positions:
            weight = pos['market_value'] / total_value
            hhi += (weight ** 2)
        
        # Scale HHI (max 1 for single asset, min near 0 for many small assets) to 0-10 score
        # A simple linear scaling: HHI of 1.0 (100% in one asset) maps to 10.
        concentration_score = hhi * 10.0
        return min(10.0, concentration_score)

    def calculate_correlation_risk(self, positions: List[Dict], historical_data_dict: Dict[str, pd.DataFrame]) -> float:
        """Calculate correlation risk score (0-10)."""
        if len(positions) < 2: return 0.0 # Need at least two positions to calculate correlation
        
        # Create a DataFrame of historical prices for all relevant symbols
        symbols = [pos['symbol'] for pos in positions]
        price_data = pd.DataFrame()
        for symbol in symbols:
            if symbol in historical_data_dict and not historical_data_dict[symbol].empty:
                price_data[symbol] = historical_data_dict[symbol]['close']
        
        if price_data.empty or len(price_data.columns) < 2: return 0.0
        
        # Calculate daily returns
        returns_df = price_data.pct_change().dropna()
        
        if returns_df.empty or len(returns_df.columns) < 2: return 0.0

        # Calculate correlation matrix
        correlation_matrix = returns_df.corr()
        
        # Sum of off-diagonal absolute correlations (simplified metric)
        sum_abs_correlations = 0.0
        count = 0
        for i in range(len(correlation_matrix.columns)):
            for j in range(i + 1, len(correlation_matrix.columns)):
                sum_abs_correlations += abs(correlation_matrix.iloc[i, j])
                count += 1
        
        avg_abs_correlation = sum_abs_correlations / count if count > 0 else 0.0
        
        # Scale to 0-10 score (e.g., avg_abs_correlation of 0.5 maps to 5.0)
        correlation_score = avg_abs_correlation * 10.0
        return min(10.0, correlation_score)

    def calculate_tail_risk(self, portfolio_returns: pd.Series, portfolio_value: float, alpha: float = 0.01) -> float:
        """Calculate Conditional Value at Risk (CVaR) / Expected Shortfall."""
        if portfolio_returns.empty: return 0.0
        
        # Calculate VaR
        var_return = portfolio_returns.quantile(alpha)
        
        # Filter returns less than VaR
        tail_returns = portfolio_returns[portfolio_returns <= var_return]
        
        if tail_returns.empty: return 0.0

        # CVaR is the average of these tail returns
        cvar_return = tail_returns.mean()
        
        # CVaR in dollar terms
        cvar_dollar = portfolio_value * abs(cvar_return)
        return cvar_dollar

    def calculate_liquidity_risk(self, positions: List[Dict]) -> float:
        """Calculate liquidity risk score (0-10)."""
        if not positions: return 0.0
        
        # Simplified: Assume higher volume means higher liquidity.
        # In a real system, would use average daily volume, bid-ask spread, market depth.
        # For mock, we'll assign a mock liquidity score to each symbol.
        
        mock_liquidity_scores = {
            'GLD': 0.9, 'IAU': 0.85, 'GC=F': 0.95, 'DUST': 0.6, 'TLT': 0.9, 'SPY': 0.98
        }
        
        total_market_value = sum(pos['market_value'] for pos in positions)
        if total_market_value == 0: return 0.0

        weighted_liquidity = 0.0
        for pos in positions:
            symbol_liquidity = mock_liquidity_scores.get(pos['symbol'], 0.7) # Default to 0.7
            weighted_liquidity += (pos['market_value'] / total_market_value) * symbol_liquidity
        
        # Convert to risk score (lower liquidity = higher risk)
        # Score of 0.9 (high liquidity) -> 1.0 risk, 0.5 (medium) -> 5.0 risk, 0.1 (low) -> 9.0 risk
        liquidity_risk_score = (1.0 - weighted_liquidity) * 10.0
        return min(10.0, max(0.0, liquidity_risk_score))

    def get_risk_weights(self, portfolio_data: Dict) -> Dict[str, float]:
        """Dynamic risk weighting based on market conditions (mock)."""
        # Base weights
        weights = {
            'var_score': 0.15,
            'drawdown_score': 0.15,
            'volatility_score': 0.15,
            'concentration_score': 0.15,
            'correlation_score': 0.15,
            'tail_risk_score': 0.15,
            'liquidity_score': 0.10
        }
        
        # Adjust weights during high volatility (mock condition)
        if portfolio_data.get('market_volatility_index', 0) > 25: # Example VIX > 25
            weights['volatility_score'] = 0.20
            weights['tail_risk_score'] = 0.20
            weights['var_score'] = 0.20
            weights['concentration_score'] = 0.10 # Reduce other weights to compensate
        
        # Adjust during low liquidity (mock condition)
        if portfolio_data.get('overall_liquidity_score', 0.0) < 0.5:
            weights['liquidity_score'] = 0.20
            weights['concentration_score'] = 0.20
        
        # Normalize weights
        total = sum(weights.values())
        return {k: v/total for k, v in weights.items()}
    
    async def get_user_portfolio_data(self, user_id: int) -> Optional[Dict]:
        """
        Retrieves user's current portfolio data from the database.
        This would typically involve querying a 'positions' table.
        For demo, returns simulated data.
        """
        try:
            # In a real system, this would query your database for actual positions
            # For now, simulate fetching from DB or a live broker connection
            
            # Mock account info
            mock_account_info = await self.trading_interface.default_broker.get_account_info()
            total_value = mock_account_info['portfolio_value'] # Use mock broker's portfolio value

            # Mock positions (these would come from broker/DB)
            mock_positions = [
                Position(symbol='GLD', quantity=100.0, entry_price=180.0, current_price=await self.trading_interface.default_broker.get_current_price('GLD') or 185.0, entry_time=datetime.utcnow() - timedelta(days=30), position_type='long', market_value=0.0),
                Position(symbol='IAU', quantity=200.0, entry_price=38.0, current_price=await self.trading_interface.default_broker.get_current_price('IAU') or 40.0, entry_time=datetime.utcnow() - timedelta(days=60), position_type='long', market_value=0.0)
            ]
            # Update market values for mock positions
            for pos in mock_positions:
                pos.market_value = pos.current_price * pos.quantity

            # Add some mock market context
            mock_market_volatility_index = await self.db_manager.get_economic_data('VIXCLS')
            market_volatility_index = float(mock_market_volatility_index.iloc[-1]['value']) if not mock_market_volatility_index.empty else 20.0

            return {
                'user_id': user_id,
                'total_value': total_value,
                'cash': mock_account_info['cash'],
                'positions': [asdict(p) for p in mock_positions], # Convert dataclasses to dicts
                'market_volatility_index': market_volatility_index, # Example market context
                'overall_liquidity_score': 0.8 # Example overall liquidity
            }
        except Exception as e:
            logger.error(f"Error getting user portfolio data for user {user_id}: {e}", exc_info=True)
            return None
    
    async def generate_hedge_recommendations(self, user_id: int, risk_metrics: RiskMetrics, portfolio_data: Dict) -> List[HedgeRecommendation]:
        """Generate appropriate hedge recommendations based on risk profile."""
        try:
            recommendations = []
            
            if not portfolio_data:
                logger.warning(f"No portfolio data for user {user_id}. Cannot generate hedge recommendations.")
                return recommendations
            
            # Determine urgency level
            urgency = self.determine_urgency_level(risk_metrics.overall_risk_score)
            
            # Generate different types of hedges based on risk factors
            # Prioritize hedges based on highest risk components
            
            # Example: If concentration risk is high, suggest direct offset or diversification
            if risk_metrics.concentration_risk > 7.0:
                rec = self.create_direct_offset_hedge(portfolio_data, risk_metrics, urgency)
                if rec: recommendations.append(rec)
            
            # Example: If volatility is high, suggest volatility hedge
            if risk_metrics.volatility > 0.02: # 2% daily volatility
                rec = self.create_volatility_hedge(portfolio_data, risk_metrics, urgency)
                if rec: recommendations.append(rec)
            
            # Example: If correlation risk is high, suggest correlation hedge
            if risk_metrics.correlation_risk > 6.0:
                rec = self.create_correlation_hedge(portfolio_data, risk_metrics, urgency)
                if rec: recommendations.append(rec)
            
            # Example: If overall risk is very high, suggest diversification or tail risk hedge
            if risk_metrics.overall_risk_score > 8.0:
                rec = self.create_diversification_hedge(portfolio_data, risk_metrics, urgency)
                if rec: recommendations.append(rec)
                rec = self.create_tail_risk_hedge(portfolio_data, risk_metrics, urgency)
                if rec: recommendations.append(rec)
            
            # Example: If liquidity risk is high
            if risk_metrics.liquidity_risk > 6.0:
                rec = self.create_liquidity_hedge(portfolio_data, risk_metrics, urgency)
                if rec: recommendations.append(rec)
            
            # Filter out hedges that exceed cost limit or are too small/large
            recommendations = [
                r for r in recommendations 
                if r.cost_estimate <= self.hedge_cost_limit and
                   self.min_hedge_size <= r.hedge_size <= self.max_hedge_size
            ]

            # Sort by expected risk reduction and confidence, penalizing cost
            recommendations.sort(
                key=lambda h: (h.risk_reduction * h.confidence) / max(h.cost_estimate / 100, 0.001), # Cost in %
                reverse=True
            )
            
            return recommendations[:3] # Return top 3 recommendations
            
        except Exception as e:
            logger.error(f"Generate hedge recommendations error: {e}", exc_info=True)
            return []

    def determine_urgency_level(self, risk_score: float) -> RiskLevel:
        """Determines the urgency level of hedging based on overall risk score."""
        if risk_score >= self.circuit_breaker_level:
            return RiskLevel.EXTREME
        elif risk_score >= self.risk_trigger_level:
            return RiskLevel.CRITICAL
        elif risk_score >= (self.risk_trigger_level - 2.0): # E.g., if trigger is 7, high from 5
            return RiskLevel.HIGH
        elif risk_score >= (self.risk_trigger_level - 4.0):
            return RiskLevel.MODERATE
        else:
            return RiskLevel.LOW

    def create_direct_offset_hedge(self, portfolio_data: Dict, risk_metrics: RiskMetrics, urgency: RiskLevel) -> Optional[HedgeRecommendation]:
        """Create a direct offset hedge (e.g., shorting a highly correlated inverse ETF)."""
        # Find the largest long position in gold-related assets
        gold_positions = [p for p in portfolio_data.get('positions', []) if p['symbol'] in ['GLD', 'IAU'] and p['position_type'] == 'long']
        if not gold_positions: return None
        
        main_gold_pos = max(gold_positions, key=lambda p: p['market_value'])
        
        hedge_instrument = self.hedge_instruments.get('gold_inverse_etf')
        if not hedge_instrument: return None

        # Calculate hedge size based on market value of concentrated position
        hedge_size_value = main_gold_pos['market_value'] * 0.5 # Hedge 50% of the concentrated position
        hedge_size_percent = hedge_size_value / portfolio_data['total_value'] # Convert to % of total portfolio
        hedge_size_percent = min(hedge_size_percent, self.max_hedge_size) # Cap at max allowed hedge size

        return HedgeRecommendation(
            hedge_id=str(uuid.uuid4()),
            hedge_type=HedgeType.DIRECT_OFFSET,
            hedge_symbol=hedge_instrument['symbol'],
            hedge_action='SELL', # Short the inverse ETF
            hedge_size=hedge_size_percent,
            hedge_ratio=abs(hedge_instrument['correlation']),
            risk_reduction=risk_metrics.concentration_risk * 2, # Higher reduction for direct
            cost_estimate=hedge_instrument['cost_bps'],
            confidence=0.85,
            reasoning=f"Directly offsetting concentrated {main_gold_pos['symbol']} position with inverse ETF.",
            urgency=urgency
        )

    def create_volatility_hedge(self, portfolio_data: Dict, risk_metrics: RiskMetrics, urgency: RiskLevel) -> Optional[HedgeRecommendation]:
        """Create a volatility hedge (e.g., buying VIX futures ETF)."""
        hedge_instrument = self.hedge_instruments.get('vix_etf')
        if not hedge_instrument: return None

        # Hedge size scales with current volatility and risk score
        hedge_size_percent = min(self.max_hedge_size, risk_metrics.volatility * 2.0) # Scale volatility to hedge size
        
        return HedgeRecommendation(
            hedge_id=str(uuid.uuid4()),
            hedge_type=HedgeType.VOLATILITY_HEDGE,
            hedge_symbol=hedge_instrument['symbol'],
            hedge_action='BUY',
            hedge_size=hedge_size_percent,
            hedge_ratio=0.7, # VIX correlation is not 1.0
            risk_reduction=risk_metrics.volatility * 10, # Higher reduction for direct
            cost_estimate=hedge_instrument['cost_bps'],
            confidence=0.70,
            reasoning="Hedging against expected market volatility spikes using VIX-related ETF.",
            urgency=urgency
        )

    def create_correlation_hedge(self, portfolio_data: Dict, risk_metrics: RiskMetrics, urgency: RiskLevel) -> Optional[HedgeRecommendation]:
        """Create a correlation-based hedge (e.g., shorting gold futures if long gold ETFs)."""
        # Find a gold long position
        gold_long_pos = next((p for p in portfolio_data.get('positions', []) if p['symbol'] in ['GLD', 'IAU'] and p['position_type'] == 'long'), None)
        if not gold_long_pos: return None

        hedge_instrument = self.hedge_instruments.get('gold_futures')
        if not hedge_instrument: return None

        # Hedge size based on correlation risk and position size
        hedge_size_percent = min(self.max_hedge_size, gold_long_pos['market_value'] / portfolio_data['total_value'] * risk_metrics.correlation_risk / 10.0)
        
        return HedgeRecommendation(
            hedge_id=str(uuid.uuid4()),
            hedge_type=HedgeType.CORRELATION_HEDGE,
            hedge_symbol=hedge_instrument['symbol'],
            hedge_action='SELL', # Short gold futures
            hedge_size=hedge_size_percent,
            hedge_ratio=abs(hedge_instrument['correlation']),
            risk_reduction=risk_metrics.correlation_risk * 1.5,
            cost_estimate=hedge_instrument['cost_bps'],
            confidence=0.80,
            reasoning=f"Hedging {gold_long_pos['symbol']} exposure by shorting highly correlated gold futures.",
            urgency=urgency
        )

    def create_diversification_hedge(self, portfolio_data: Dict, risk_metrics: RiskMetrics, urgency: RiskLevel) -> Optional[HedgeRecommendation]:
        """Create a diversification hedge (e.g., buying treasuries for equity-heavy portfolio)."""
        # This hedge is more about adding uncorrelated assets than directly offsetting.
        # Assume portfolio is equity-heavy if gold positions are small.
        if portfolio_data.get('total_value', 0) > 0 and any(p['symbol'] in ['GLD', 'IAU'] for p in portfolio_data.get('positions', [])):
            # If there are gold positions, this might not be the right hedge
            return None

        hedge_instrument = self.hedge_instruments.get('treasuries')
        if not hedge_instrument: return None

        # Hedge size based on overall risk, but small for diversification
        hedge_size_percent = min(self.max_hedge_size, risk_metrics.overall_risk_score / 10.0 * 0.05) # Max 5% of portfolio
        
        return HedgeRecommendation(
            hedge_id=str(uuid.uuid4()),
            hedge_type=HedgeType.SECTOR_DIVERSIFICATION,
            hedge_symbol=hedge_instrument['symbol'],
            hedge_action='BUY',
            hedge_size=hedge_size_percent,
            hedge_ratio=0.2, # Low correlation, so low hedge ratio
            risk_reduction=risk_metrics.overall_risk_score * 0.5,
            cost_estimate=hedge_instrument['cost_bps'],
            confidence=0.60,
            reasoning="Diversifying portfolio with low-correlated treasuries to reduce overall risk.",
            urgency=urgency
        )

    def create_tail_risk_hedge(self, portfolio_data: Dict, risk_metrics: RiskMetrics, urgency: RiskLevel) -> Optional[HedgeRecommendation]:
        """Create tail risk hedge using options strategies (e.g., buying GLD put options)."""
        hedge_instrument = self.hedge_instruments.get('gold_options')
        if not hedge_instrument: return None

        # Calculate hedge size based on tail risk metric
        # Example: if tail_risk is $5000, hedge 1% of portfolio per $1000 tail risk
        hedge_size_percent = min(self.max_hedge_size, risk_metrics.tail_risk / portfolio_data['total_value'] * 0.5)
        
        return HedgeRecommendation(
            hedge_id=str(uuid.uuid4()),
            hedge_type=HedgeType.TAIL_RISK_HEDGE,
            hedge_symbol=hedge_instrument['symbol'],
            hedge_action='BUY', # Buy put options
            hedge_size=hedge_size_percent,
            hedge_ratio=0.8, # Options can be highly effective
            risk_reduction=risk_metrics.tail_risk * 0.05, # 5% of tail risk value
            cost_estimate=hedge_instrument['cost_bps'],
            confidence=0.75,
            reasoning="Buying put options to protect against extreme downside movements in gold.",
            urgency=urgency,
            order_type=OrderType.LIMIT, # Often limit orders for options
            expiry_date=datetime.now() + timedelta(days=30) # Short-term options
        )
    
    def create_liquidity_hedge(self, portfolio_data: Dict, risk_metrics: RiskMetrics, urgency: RiskLevel) -> Optional[HedgeRecommendation]:
        """Create liquidity hedge using highly liquid instruments."""
        hedge_instrument = self.hedge_instruments.get('treasuries') # Treasuries are highly liquid
        if not hedge_instrument: return None

        # Hedge size scales with liquidity risk
        hedge_size_percent = min(self.max_hedge_size, risk_metrics.liquidity_risk / 10.0 * 0.1) # Max 10% of portfolio
        
        return HedgeRecommendation(
            hedge_id=str(uuid.uuid4()),
            hedge_type=HedgeType.LIQUIDITY_HEDGE,
            hedge_symbol=hedge_instrument['symbol'],
            hedge_action='BUY',
            hedge_size=hedge_size_percent,
            hedge_ratio=0.9, # High liquidity instruments are effective
            risk_reduction=risk_metrics.liquidity_risk * 1.0,
            cost_estimate=hedge_instrument['cost_bps'],
            confidence=0.85,
            reasoning="Increasing allocation to highly liquid assets (e.g., treasuries) to improve portfolio liquidity.",
            urgency=urgency
        )

    async def execute_hedge_recommendations(self, user_id: int, recommendations: List[HedgeRecommendation], portfolio_value: float):
        """Execute hedge recommendations asynchronously."""
        try:
            executed_hedges = []
            
            for recommendation in recommendations:
                if self.should_execute_hedge(recommendation):
                    try:
                        hedge_order_id = await self.trading_interface.execute_hedge(
                            recommendation, 
                            portfolio_value
                        )
                        
                        if hedge_order_id:
                            executed_hedge_record = {
                                'hedge_id': recommendation.hedge_id,
                                'user_id': user_id,
                                'recommendation': asdict(recommendation), # Store dataclass as dict
                                'executed_order_id': hedge_order_id,
                                'executed_at': datetime.now().isoformat(),
                                'portfolio_value_at_execution': portfolio_value,
                                'initial_risk_score': self.active_hedges.get(recommendation.hedge_id, {}).get('initial_risk', 0.0) # Capture initial risk
                            }
                            executed_hedges.append(executed_hedge_record)
                            
                            # Add to active hedges tracking
                            self.active_hedges[recommendation.hedge_id] = {
                                'user_id': user_id,
                                'recommendation': recommendation,
                                'status': 'active',
                                'executed_at': datetime.now(),
                                'initial_risk': self.active_hedges.get(recommendation.hedge_id, {}).get('initial_risk', 0.0) # Use initial risk from check
                            }
                            
                            logger.info(f"Executed hedge {recommendation.hedge_id} for user {user_id}.")
                            self.notification_manager.send_email_notification(
                                user_id=user_id,
                                subject=f"Hedge Executed: {recommendation.hedge_type.value}",
                                message=f"Executed {recommendation.hedge_action} {recommendation.hedge_symbol} to reduce risk. Order ID: {hedge_order_id}"
                            )
                        
                    except Exception as e:
                        logger.error(f"Hedge execution failed for {recommendation.hedge_id}: {e}", exc_info=True)
            
            # Store execution history in DB
            if executed_hedges:
                await self.store_hedge_executions(executed_hedges)
            
        except Exception as e:
            logger.error(f"Error in execute_hedge_recommendations: {e}", exc_info=True)
    
    def should_execute_hedge(self, recommendation: HedgeRecommendation) -> bool:
        """Determine if hedge should be executed based on cost, confidence, and size limits."""
        return (
            recommendation.confidence >= self.hedge_effectiveness_threshold and
            recommendation.cost_estimate <= self.hedge_cost_limit and
            self.min_hedge_size <= recommendation.hedge_size <= self.max_hedge_size and
            not self.circuit_breaker_active # Ensure circuit breaker is not active
        )
    
    async def update_active_hedges(self):
        """Update status and performance of active hedges asynchronously."""
        try:
            current_time = datetime.now()
            hedges_to_remove = []
            
            for hedge_id, hedge_data in list(self.active_hedges.items()):
                try:
                    rec = hedge_data['recommendation']
                    
                    # Check if hedge has expired
                    if rec.expiry_date and current_time > rec.expiry_date:
                        logger.info(f"Hedge {hedge_id} expired. Closing.")
                        await self.close_hedge(hedge_id, reason="Expired")
                        hedges_to_remove.append(hedge_id)
                        continue

                    # Check if hedge should be closed based on effectiveness or risk reduction
                    performance = await self.calculate_hedge_performance(hedge_data)
                    hedge_data['performance'] = performance
                    hedge_data['last_updated'] = current_time
                    
                    if performance.get('effectiveness', 0.0) < 0.3: # If effectiveness drops below 30%
                        logger.warning(f"Hedge {hedge_id} effectiveness too low ({performance['effectiveness']:.2f}). Adjusting/Closing.")
                        await self.adjust_hedge(hedge_id, hedge_data)
                        # Assume adjustment leads to new hedge or closure, so mark for removal
                        hedges_to_remove.append(hedge_id)
                        continue
                    
                    # Check if risk is sufficiently mitigated by this hedge
                    user_id = hedge_data['user_id']
                    current_portfolio_data = await self.get_user_portfolio_data(user_id)
                    current_risk = await self.calculate_portfolio_risk(user_id, current_portfolio_data)
                    
                    if current_risk.overall_risk_score < (self.risk_trigger_level - 1.0) and hedge_data.get('initial_risk', 10.0) >= self.risk_trigger_level:
                        # If risk has dropped significantly below trigger, consider closing
                        logger.info(f"Portfolio risk for user {user_id} has significantly reduced. Considering closing hedge {hedge_id}.")
                        await self.close_hedge(hedge_id, reason="Risk mitigated")
                        hedges_to_remove.append(hedge_id)
                        continue

                except Exception as e:
                    logger.error(f"Error updating active hedge {hedge_id}: {e}", exc_info=True)
            
            for hedge_id in hedges_to_remove:
                self.active_hedges.pop(hedge_id, None)
                
        except Exception as e:
            logger.error(f"Error in update_active_hedges loop: {e}", exc_info=True)
    
    async def adjust_hedge(self, hedge_id: str, hedge_data: Dict):
        """Adjust or re-evaluate an underperforming hedge position."""
        try:
            logger.info(f"Adjusting/re-evaluating underperforming hedge: {hedge_id}")
            
            # Option 1: Close partial position
            # Option 2: Close entire position and generate new recommendation
            await self.close_hedge(hedge_id, partial=True, reason="Underperforming")
            
            # Generate new recommendation based on current risk
            user_id = hedge_data['user_id']
            current_portfolio_data = await self.get_user_portfolio_data(user_id)
            current_risk = await self.calculate_portfolio_risk(user_id, current_portfolio_data)
            new_recommendations = await self.generate_hedge_recommendations(
                user_id, current_risk, current_portfolio_data
            )
            
            if new_recommendations:
                logger.info(f"Generated new hedge recommendations after adjustment for {hedge_id}.")
                await self.execute_hedge_recommendations(user_id, new_recommendations[:1], current_portfolio_data.get('total_value', 0)) # Execute top new hedge
            
        except Exception as e:
            logger.error(f"Hedge adjustment failed for {hedge_id}: {e}", exc_info=True)
    
    async def close_hedge(self, hedge_id: str, partial: bool = False, reason: str = "System initiated"):
        """Close an active hedge asynchronously."""
        try:
            if hedge_id not in self.active_hedges:
                logger.warning(f"Attempted to close non-existent hedge: {hedge_id}")
                return
            
            hedge_data = self.active_hedges[hedge_id]
            rec = hedge_data['recommendation']
            
            # Determine close action (opposite of original)
            close_action = 'SELL' if rec.hedge_action == 'BUY' else 'BUY'
            
            # Place closing order
            # For partial close, calculate quantity based on `rec.hedge_size * 0.5` or similar logic
            # For full close, use original quantity
            
            # For simplicity in mock, assume full close
            portfolio_value_at_execution = hedge_data.get('portfolio_value_at_execution', 100_000)
            closing_recommendation = HedgeRecommendation(
                hedge_id=str(uuid.uuid4()), # New ID for closing order
                hedge_type=rec.hedge_type,
                hedge_symbol=rec.hedge_symbol,
                hedge_action=close_action,
                hedge_size=rec.hedge_size, # Use original size for calculation
                hedge_ratio=rec.hedge_ratio,
                risk_reduction=0.0, # Not applicable for closing
                cost_estimate=rec.cost_estimate,
                confidence=1.0, # High confidence in closing
                reasoning=f"Closing hedge {rec.hedge_id} due to: {reason}",
                urgency=RiskLevel.MODERATE
            )
            
            close_order_id = await self.trading_interface.execute_hedge(
                closing_recommendation, 
                portfolio_value_at_execution # Use value from execution time for quantity
            )
            
            if close_order_id:
                # Update hedge status
                hedge_data['status'] = 'closed' if not partial else 'adjusted'
                hedge_data['closed_at'] = datetime.now()
                hedge_data['close_order_id'] = close_order_id
                hedge_data['close_reason'] = reason
                
                # Store in history (before deleting from active)
                await self.store_hedge_executions([hedge_data]) # Store the updated record
                
                logger.info(f"Closed hedge {hedge_id} with order {close_order_id}. Reason: {reason}")
                
                # Notify user
                self.notification_manager.send_email_notification(
                    user_id=hedge_data['user_id'],
                    subject=f"Hedge Closed: {rec.hedge_type.value}",
                    message=f"Your hedge {rec.hedge_symbol} has been closed. Reason: {reason}"
                )
                
                # Remove from active hedges if fully closed
                if not partial:
                    self.active_hedges.pop(hedge_id, None)
            
        except Exception as e:
            logger.error(f"Close hedge error for {hedge_id}: {e}", exc_info=True)
    
    async def calculate_hedge_performance(self, hedge_data: Dict) -> Dict:
        """Calculate hedge performance metrics asynchronously."""
        try:
            rec = hedge_data['recommendation']
            user_id = hedge_data['user_id']
            executed_at = hedge_data['executed_at']
            
            # Get current risk
            current_portfolio_data = await self.get_user_portfolio_data(user_id)
            current_risk_metrics = await self.calculate_portfolio_risk(user_id, current_portfolio_data)
            
            initial_risk_score = hedge_data.get('initial_risk', 0.0)
            current_risk_score = current_risk_metrics.overall_risk_score
            
            # Calculate risk reduction achieved
            risk_reduction_achieved = max(0.0, initial_risk_score - current_risk_score)
            
            # Effectiveness: how much of the *expected* reduction was achieved
            effectiveness = min(1.0, risk_reduction_achieved / rec.risk_reduction) if rec.risk_reduction > 0 else 0.0
            
            # Duration
            duration_seconds = (datetime.now() - executed_at).total_seconds()
            duration_days = duration_seconds / (24 * 3600)
            
            return {
                'effectiveness': float(effectiveness),
                'risk_reduction_achieved': float(risk_reduction_achieved),
                'duration_days': float(duration_days),
                'current_portfolio_risk': float(current_risk_score),
                'cost_to_benefit_ratio': float(rec.cost_estimate / max(risk_reduction_achieved, 0.001)) # Avoid div by zero
            }
            
        except Exception as e:
            logger.error(f"Calculate hedge performance error for {hedge_data.get('hedge_id')}: {e}", exc_info=True)
            return {'effectiveness': 0.0, 'risk_reduction_achieved': 0.0, 'duration_days': 0.0, 'current_portfolio_risk': 0.0, 'cost_to_benefit_ratio': 0.0}
    
    def activate_circuit_breaker(self, reason: str):
        """Activate circuit breaker for extreme market conditions."""
        if not self.circuit_breaker_active:
            self.circuit_breaker_active = True
            logger.critical(f"🚨 MARKET CIRCUIT BREAKER ACTIVATED - ALL HEDGING SUSPENDED. Reason: {reason}")
            
            # Close all active hedges (asynchronously)
            asyncio.create_task(self.close_all_hedges(reason="Circuit Breaker Activated"))
            
            # Send emergency notifications
            self.notification_manager.send_system_alert("Circuit Breaker Activated", f"Reason: {reason}", "emergency")
        else:
            logger.warning("Circuit breaker already active.")
    
    async def close_all_hedges(self, reason: str = "System initiated closure"):
        """Close all active hedge positions."""
        logger.info(f"Closing all active hedges. Reason: {reason}")
        tasks = [self.close_hedge(hedge_id, reason=reason) for hedge_id in list(self.active_hedges.keys())]
        await asyncio.gather(*tasks, return_exceptions=True)
        logger.info("All active hedges closure initiated.")
    
    async def store_hedge_executions(self, executed_hedges_records: List[Dict]):
        """Store hedge execution records in database asynchronously."""
        try:
            if not self.db_manager:
                logger.warning("DB manager not set for AutoHedgingSystem. Cannot store hedge executions.")
                return
            
            # Prepare data for batch insert/update
            records = []
            for hedge_record in executed_hedges_records:
                rec = hedge_record['recommendation'] # This is the HedgeRecommendation dataclass
                records.append((
                    rec.hedge_id, # Use hedge_id as primary key for INSERT OR REPLACE
                    hedge_record['user_id'],
                    rec.hedge_type.value,
                    rec.hedge_symbol,
                    rec.hedge_action,
                    rec.hedge_size,
                    rec.risk_reduction,
                    rec.cost_estimate,
                    hedge_record['executed_at'],
                    hedge_record['portfolio_value_at_execution'],
                    hedge_record.get('status', 'active'), # Status like 'active', 'closed', 'adjusted'
                    hedge_record.get('closed_at'),
                    hedge_record.get('close_reason')
                ))
            
            conn = self.db_manager.get_connection()
            cursor = conn.cursor()
            
            # Use INSERT OR REPLACE to handle updates to existing hedge_id records
            cursor.executemany('''
                INSERT OR REPLACE INTO hedge_executions 
                (hedge_id, user_id, hedge_type, hedge_symbol, hedge_action, 
                 hedge_size, risk_reduction, cost_estimate, executed_at, 
                 portfolio_value, status, closed_at, close_reason)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', records)
            conn.commit()
            conn.close()
            
            logger.info(f"Stored/updated {len(executed_hedges_records)} hedge execution records.")
            
        except Exception as e:
            logger.error(f"Store hedge executions error: {e}", exc_info=True)

    def cleanup_expired_hedges(self):
        """Removes expired hedges from active_hedges that were not closed by other means."""
        hedges_to_remove = []
        for hedge_id, hedge_data in self.active_hedges.items():
            rec = hedge_data['recommendation']
            if rec.expiry_date and datetime.now() > rec.expiry_date:
                logger.info(f"Cleaning up expired hedge {hedge_id} from active list (was not explicitly closed).")
                hedges_to_remove.append(hedge_id)
        
        for hedge_id in hedges_to_remove:
            self.active_hedges.pop(hedge_id, None)


# Integration example (for standalone testing)
async def main():
    # Configure logging for standalone test
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Mock config
    config = {
        'auto_hedging': {
            'risk_trigger': 7.0,
            'max_exposure': 0.15,
            'cost_limit': 50,
            'auto_execute': True, # Set to True for testing auto-execution
            'monitoring_interval': 5, # Short interval for testing
            'circuit_breaker': 9.5,
            'max_hedge_duration': 45
        }
    }
    
    # Mock FinancialDataFramework for DB and data fetching
    class MockDataFrameworkForHedging:
        def get_connection(self):
            conn = sqlite3.connect(':memory:')
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS hedge_executions (
                    hedge_id TEXT PRIMARY KEY, user_id INTEGER, hedge_type TEXT, 
                    hedge_symbol TEXT, hedge_action TEXT, hedge_size REAL, 
                    risk_reduction REAL, cost_estimate REAL, executed_at TIMESTAMP, 
                    portfolio_value REAL, status TEXT, closed_at TIMESTAMP, close_reason TEXT
                )
            ''')
            conn.commit()
            return conn
        
        async def get_multiple_stocks(self, symbols: List[str], outputsize: int) -> Dict[str, pd.DataFrame]:
            logger.info(f"MockDataFramework: Generating synthetic historical data for {symbols}.")
            mock_dfs = {}
            for symbol in symbols:
                dates = pd.date_range(end=datetime.now(), periods=outputsize, freq='D')
                prices = np.cumsum(np.random.randn(outputsize)) + (1800 if symbol in ['GLD', 'IAU', 'GC=F'] else 100)
                df = pd.DataFrame({'close': prices, 'high': prices*1.005, 'low': prices*0.995}, index=dates)
                mock_dfs[symbol] = df
            return mock_dfs
        
        async def get_economic_data(self, series_id: str) -> pd.DataFrame:
            logger.info(f"MockDataFramework: Generating synthetic economic data for {series_id}.")
            dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
            values = np.random.uniform(15.0, 30.0, len(dates)) # Mock VIX
            df = pd.DataFrame({'value': values}, index=dates)
            return df

    mock_db_manager = MockDataFrameworkForHedging()
    
    # Initialize Mock Notification Manager
    mock_notification_manager = NotificationManager()

    # Initialize Mock Broker and Trading Interface
    mock_alpaca_broker = AlpacaBroker(api_key="MOCK_ALPACA_KEY", secret_key="MOCK_ALPACA_SECRET")
    trading_interface = TradingInterface(brokers={'alpaca': mock_alpaca_broker})

    # Initialize AutoHedgingSystem
    hedging_system = AutoHedgingSystem(
        config=config,
        db_manager=mock_db_manager,
        trading_interface=trading_interface,
        notification_manager=mock_notification_manager
    )
    
    # --- Test 1: Calculate Portfolio Risk ---
    print("\n--- Test 1: Calculating Portfolio Risk ---")
    demo_user_id = 1
    # Simulate an initial position for risk calculation
    mock_alpaca_broker.mock_positions['GLD'] = Position(
        symbol='GLD', quantity=100.0, entry_price=180.0, current_price=185.0,
        entry_time=datetime.utcnow() - timedelta(days=10), position_type='long', market_value=18500.0
    )
    mock_alpaca_broker.mock_positions['IAU'] = Position(
        symbol='IAU', quantity=500.0, entry_price=39.0, current_price=40.0,
        entry_time=datetime.utcnow() - timedelta(days=5), position_type='long', market_value=20000.0
    )

    portfolio_data_initial = await hedging_system.get_user_portfolio_data(demo_user_id)
    if portfolio_data_initial:
        risk_metrics = await hedging_system.calculate_portfolio_risk(demo_user_id, portfolio_data_initial)
        print(f"Portfolio Risk Score: {risk_metrics.overall_risk_score:.2f}/10")
        print(f"VaR: ${risk_metrics.portfolio_var:.2f}, Max Drawdown: {risk_metrics.max_drawdown:.2%}")
        print(f"Concentration Risk: {risk_metrics.concentration_risk:.2f}")
    else:
        print("Failed to get initial portfolio data.")

    # --- Test 2: Generate and Execute Hedge Recommendations (if risk is high) ---
    print("\n--- Test 2: Generating and Executing Hedge Recommendations ---")
    if risk_metrics.overall_risk_score >= hedging_system.risk_trigger_level:
        print("Risk is high, generating hedge recommendations...")
        recommendations = await hedging_system.generate_hedge_recommendations(demo_user_id, risk_metrics, portfolio_data_initial)
        print(f"Generated {len(recommendations)} hedge recommendations.")
        
        for rec in recommendations:
            print(f"- {rec.hedge_type.value}: {rec.hedge_symbol} ({rec.risk_reduction:.1%}) - Cost: {rec.cost_estimate}bps")
        
        if recommendations and hedging_system.hedging_config.get('auto_execute', False):
            print("Auto-execute is enabled. Attempting to execute top recommendation...")
            await hedging_system.execute_hedge_recommendations(demo_user_id, recommendations[:1], portfolio_data_initial.get('total_value', 0))
            print(f"Active Hedges: {list(hedging_system.active_hedges.keys())}")
        else:
            print("No recommendations or auto-execute is off.")
    else:
        print("Portfolio risk is below trigger level. No hedges generated.")

    # --- Test 3: Start Continuous Monitoring (and simulate changes) ---
    print("\n--- Test 3: Starting Continuous Monitoring (will run for 15 seconds) ---")
    hedging_system.start_risk_monitoring()
    await asyncio.sleep(15) # Let it run for a bit
    hedging_system.stop_risk_monitoring()
    print("Monitoring stopped.")

    # --- Test 4: Check Active Hedges and History ---
    print(f"\nFinal Active Hedges: {list(hedging_system.active_hedges.keys())}")
    conn = mock_db_manager.get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM hedge_executions")
    hedge_history_db = cursor.fetchall()
    conn.close()
    print(f"Hedge Execution History in DB ({len(hedge_history_db)} records):")
    for row in hedge_history_db:
        print(dict(row))

    print("\nAuto-Hedging System testing complete.")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Auto-Hedging System example interrupted by user.")
    except Exception as e:
        logger.critical(f"Auto-Hedging System example failed: {e}", exc_info=True)
        exit(1)
