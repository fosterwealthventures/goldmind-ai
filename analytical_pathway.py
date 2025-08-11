import logging
import random
import pandas as pd
import numpy as np

# A class to simulate a rule-based or heuristic analytical pathway.
class AnalyticalPathway:
    def __init__(self):
        """
        Initializes the AnalyticalPathway component.
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info("AnalyticalPathway initialized. (Using mock rule-based strategy)")

    def _generate_price_targets(self, last_price, action, volatility_factor=0.01):
        """
        Generates mock buy and sell targets based on the last price and a simulated action.
        
        Args:
            last_price (float): The last known closing price of the asset.
            action (str): The simulated action ("BUY", "SELL", or "HOLD").
            volatility_factor (float): A factor to simulate market volatility for target prices.

        Returns:
            tuple: A tuple containing (buy_target, sell_target).
        """
        if action == "BUY":
            # For a BUY action, the buy target should be below the last price,
            # and the sell target should be above it.
            buy_target = last_price * (1 - random.uniform(volatility_factor, volatility_factor * 2))
            sell_target = last_price * (1 + random.uniform(volatility_factor * 3, volatility_factor * 5))
        elif action == "SELL":
            # For a SELL action, the buy target should be above the last price (to cover),
            # and the sell target should be below it.
            buy_target = last_price * (1 + random.uniform(volatility_factor, volatility_factor * 2))
            sell_target = last_price * (1 - random.uniform(volatility_factor * 3, volatility_factor * 5))
        else: # HOLD
            buy_target = None
            sell_target = None
            
        return round(buy_target, 2) if buy_target else None, round(sell_target, 2) if sell_target else None

    def get_recommendation(self, processed_data):
        """
        Generates a recommendation based on a simple Moving Average crossover strategy.

        Args:
            processed_data (pd.DataFrame): The time-series data for the asset.

        Returns:
            dict: A recommendation containing action, confidence, and price targets.
        """
        self.logger.info("Running analytical and heuristic pathway (Moving Average Crossover).")
        
        # Check if there is enough data for the moving averages
        if processed_data.empty or len(processed_data) < 50:
            return {
                "action": "HOLD",
                "confidence": 0.0,
                "reasoning": "Insufficient data for technical analysis.",
                "buy_target": None,
                "sell_target": None
            }

        # Calculate a short-term (20-period) and a long-term (50-period) Simple Moving Average
        processed_data['SMA_20'] = processed_data['Close'].rolling(window=20).mean()
        processed_data['SMA_50'] = processed_data['Close'].rolling(window=50).mean()

        # Get the latest values
        current_sma_20 = processed_data['SMA_20'].iloc[-1]
        current_sma_50 = processed_data['SMA_50'].iloc[-1]
        previous_sma_20 = processed_data['SMA_20'].iloc[-2]
        previous_sma_50 = processed_data['SMA_50'].iloc[-2]

        action = "HOLD"
        confidence = 0.5
        reasoning = "Moving average lines are not in a clear crossover pattern."

        # Check for a "Golden Cross" (20 SMA crosses above 50 SMA)
        if current_sma_20 > current_sma_50 and previous_sma_20 <= previous_sma_50:
            action = "BUY"
            confidence = 0.8
            reasoning = "A 'Golden Cross' has occurred: the 20-period SMA has crossed above the 50-period SMA, indicating a potential bullish trend."
        # Check for a "Death Cross" (20 SMA crosses below 50 SMA)
        elif current_sma_20 < current_sma_50 and previous_sma_20 >= previous_sma_50:
            action = "SELL"
            confidence = 0.8
            reasoning = "A 'Death Cross' has occurred: the 20-period SMA has crossed below the 50-period SMA, indicating a potential bearish trend."
        
        last_price = processed_data['Close'].iloc[-1]
        buy_target, sell_target = self._generate_price_targets(last_price, action)

        return {
            "action": action,
            "confidence": confidence,
            "reasoning": reasoning,
            "buy_target": buy_target,
            "sell_target": sell_target
        }