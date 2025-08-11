"""Technical analysis module for calculating financial indicators."""

from typing import Dict, Any
import math
from langchain_core.runnables import RunnableConfig


class TechnicalAnalysis:
    """Class for performing technical analysis on financial data."""
    
    def __init__(self, config: RunnableConfig):
        """Initialize TechnicalAnalysis."""
        self.config = config
    
    def calculate_sma(self, prices: list, period: int) -> float:
        """Calculate Simple Moving Average."""
        if len(prices) < period:
            return 0
        
        closes = [price.get('close', 0) for price in prices[-period:] if price.get('close')]
        if len(closes) < period:
            return 0
            
        return sum(closes) / len(closes)
    
    def calculate_ema(self, prices: list, period: int) -> float:
        """Calculate Exponential Moving Average."""
        if len(prices) < period:
            return 0
        
        closes = [price.get('close', 0) for price in prices[-period:] if price.get('close')]
        if len(closes) < period:
            return 0
            
        # Simple EMA calculation
        multiplier = 2 / (period + 1)
        ema = closes[0]
        for i in range(1, len(closes)):
            ema = (closes[i] * multiplier) + (ema * (1 - multiplier))
        return ema
    
    def calculate_rsi(self, prices: list, period: int = 14) -> float:
        """Calculate Relative Strength Index."""
        if len(prices) < period + 1:
            return 50  # Neutral RSI
        
        closes = [price.get('close', 0) for price in prices[-(period+1):] if price.get('close')]
        if len(closes) < period + 1:
            return 50
        
        # Calculate price changes
        gains = []
        losses = []
        
        for i in range(1, len(closes)):
            change = closes[i] - closes[i-1]
            if change > 0:
                gains.append(change)
                losses.append(0)
            else:
                gains.append(0)
                losses.append(abs(change))
        
        # Calculate average gains and losses
        avg_gain = sum(gains) / len(gains)
        avg_loss = sum(losses) / len(losses)
        
        if avg_loss == 0:
            return 100  # No losses, perfect RSI
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def calculate_bollinger_bands(self, prices: list, period: int = 20) -> Dict[str, float]:
        """Calculate Bollinger Bands."""
        if len(prices) < period:
            return {"upper": 0, "middle": 0, "lower": 0}
        
        closes = [price.get('close', 0) for price in prices[-period:] if price.get('close')]
        if len(closes) < period:
            return {"upper": 0, "middle": 0, "lower": 0}
            
        # Calculate moving average
        ma = sum(closes) / len(closes)
        
        # Calculate standard deviation
        variance = sum((x - ma) ** 2 for x in closes) / len(closes)
        std_dev = math.sqrt(variance)
        
        # Calculate bands
        upper_band = ma + (2 * std_dev)
        lower_band = ma - (2 * std_dev)
        
        return {
            "upper": upper_band,
            "middle": ma,
            "lower": lower_band
        }
    
    def calculate_macd(self, prices: list) -> Dict[str, float]:
        """Calculate MACD (12-day EMA - 26-day EMA)."""
        if len(prices) < 26:
            return {"macd": 0, "signal": 0, "histogram": 0}
        
        # Calculate 12-day and 26-day EMAs
        ema_12 = self.calculate_ema(prices, 12)
        ema_26 = self.calculate_ema(prices, 26)
        
        # Calculate MACD line
        macd_line = ema_12 - ema_26
        
        # Calculate signal line (9-day EMA of MACD line)
        # For simplicity, we'll approximate this
        signal_line = macd_line * 0.8  # Simplified approximation
        
        # Calculate histogram
        histogram = macd_line - signal_line
        
        return {
            "macd": macd_line,
            "signal": signal_line,
            "histogram": histogram
        }
    
    def analyze_trend(self, prices: list) -> Dict[str, Any]:
        """Analyze trend using multiple indicators."""
        if not prices or len(prices) < 50:
            return {"trend": "unknown", "strength": 0, "confidence": 0}
        
        # Calculate moving averages
        sma_20 = self.calculate_sma(prices, 20)
        sma_50 = self.calculate_sma(prices, 50)
        ema_12 = self.calculate_ema(prices, 12)
        ema_26 = self.calculate_ema(prices, 26)
        
        current_price = prices[-1].get('close', 0)
        
        # Determine trend direction
        trend = "neutral"
        if current_price > sma_20 > sma_50:
            trend = "bullish"
        elif current_price < sma_20 < sma_50:
            trend = "bearish"
        
        # Calculate trend strength based on MA separation
        strength = 0
        if sma_20 > 0 and sma_50 > 0:
            strength = abs(sma_20 - sma_50) / sma_50 * 100
        
        # Calculate confidence based on MACD
        macd_data = self.calculate_macd(prices)
        macd = macd_data["macd"]
        signal = macd_data["signal"]
        
        confidence = 50  # Neutral confidence
        if (trend == "bullish" and macd > signal) or (trend == "bearish" and macd < signal):
            confidence = 70
        elif (trend == "bullish" and macd < signal) or (trend == "bearish" and macd > signal):
            confidence = 30
        
        return {
            "trend": trend,
            "strength": strength,
            "confidence": confidence,
            "indicators": {
                "sma_20": sma_20,
                "sma_50": sma_50,
                "ema_12": ema_12,
                "ema_26": ema_26,
                "macd": macd,
                "signal": signal
            }
        }
    
    def analyze_momentum(self, prices: list) -> Dict[str, Any]:
        """Analyze momentum using RSI and price changes."""
        if not prices or len(prices) < 14:
            return {"momentum": "unknown", "rsi": 50, "price_change": 0}
        
        # Calculate RSI
        rsi = self.calculate_rsi(prices, 14)
        
        # Calculate price change over last 14 days
        price_change = 0
        if len(prices) >= 14:
            current_price = prices[-1].get('close', 0)
            past_price = prices[-14].get('close', current_price)
            if past_price > 0:
                price_change = ((current_price / past_price) - 1) * 100
        
        # Determine momentum direction
        momentum = "neutral"
        if rsi > 70 or price_change > 5:
            momentum = "bullish"
        elif rsi < 30 or price_change < -5:
            momentum = "bearish"
        
        return {
            "momentum": momentum,
            "rsi": rsi,
            "price_change": price_change
        }
    
    def analyze_volatility(self, prices: list) -> Dict[str, Any]:
        """Analyze volatility using Bollinger Bands."""
        if not prices or len(prices) < 20:
            return {"volatility": "unknown", "bandwidth": 0, "position": 0}
        
        # Calculate Bollinger Bands
        bb = self.calculate_bollinger_bands(prices, 20)
        upper = bb["upper"]
        middle = bb["middle"]
        lower = bb["lower"]
        
        current_price = prices[-1].get('close', middle)
        
        # Calculate bandwidth (volatility measure)
        bandwidth = 0
        if middle > 0:
            bandwidth = (upper - lower) / middle * 100
        
        # Calculate position within bands
        position = 50  # Middle position
        if upper > lower:
            position = (current_price - lower) / (upper - lower) * 100
        
        # Determine volatility level
        volatility = "medium"
        if bandwidth > 10:
            volatility = "high"
        elif bandwidth < 5:
            volatility = "low"
        
        return {
            "volatility": volatility,
            "bandwidth": bandwidth,
            "position": position,
            "bands": bb
        }