from common.agent_state import AgentState
from langchain.schema import AIMessage
from langchain_core.callbacks import dispatch_custom_event
from langchain_core.runnables import RunnableConfig
from typing import Dict, Any
import time
from common import markdown
from langgraph.types import StreamWriter
import math


class MeanReversionAnalysis():
    def __init__(self, options: Dict[str, Any]):
        self.options = options

    
    def calculate_sma(self, prices: list, period: int) -> float:
        """Calculate Simple Moving Average"""
        if len(prices) < period:
            return 0
        
        closes = [price.get('close', 0) for price in prices[-period:] if price.get('close')]
        if len(closes) < period:
            return 0
            
        return sum(closes) / len(closes)
    
    def calculate_std(self, prices: list, period: int) -> float:
        """Calculate Standard Deviation"""
        if len(prices) < period:
            return 0
        
        closes = [price.get('close', 0) for price in prices[-period:] if price.get('close')]
        if len(closes) < period:
            return 0
            
        mean = sum(closes) / len(closes)
        variance = sum((x - mean) ** 2 for x in closes) / len(closes)
        return math.sqrt(variance)
    
    def calculate_rsi(self, prices: list, period: int = 14) -> float:
        """Calculate Relative Strength Index"""
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
    
    def analyze(self, prices: list) -> dict[str, any]:
        """Analyze mean reversion strategy using statistical measures and Bollinger Bands."""
        result = {"score": 0, "max_score": 10, "details": [], "indicators": {}}
        if not prices or len(prices) < 50:
            result["details"].append('Insufficient price data for mean reversion analysis (need at least 50 days)')
            return result

        # Calculate moving averages and standard deviation
        ma_50 = self.calculate_sma(prices, 50)
        std_50 = self.calculate_std(prices, 50)
        
        # Calculate Bollinger Bands
        bb_upper = ma_50 + (2 * std_50)
        bb_lower = ma_50 - (2 * std_50)
        
        # Calculate z-score of price relative to moving average
        current_price = prices[-1].get('close', 0) if prices else 0
        z_score = (current_price - ma_50) / std_50 if std_50 > 0 else 0
        
        # Calculate price position within Bollinger Bands
        price_vs_bb = (current_price - bb_lower) / (bb_upper - bb_lower) if (bb_upper - bb_lower) > 0 else 0.5
        
        # Calculate RSI with multiple timeframes
        rsi_14 = self.calculate_rsi(prices, 14)
        rsi_28 = self.calculate_rsi(prices, 28)
        
        # Store indicators
        result["indicators"] = {
            "ma_50": ma_50,
            "bb_upper": bb_upper,
            "bb_lower": bb_lower,
            "z_score": z_score,
            "price_vs_bb": price_vs_bb,
            "rsi_14": rsi_14,
            "rsi_28": rsi_28
        }
        
        score = 5  # Start with neutral score
        reasoning = []
        
        # Mean reversion signals
        if z_score < -2 and price_vs_bb < 0.2:
            score += 2  # Strong bullish mean reversion signal
            reasoning.append(f"Strong bullish mean reversion signal (Z-score: {z_score:.2f}, Price vs BB: {price_vs_bb:.2f})")
        elif z_score > 2 and price_vs_bb > 0.8:
            score -= 2  # Strong bearish mean reversion signal
            reasoning.append(f"Strong bearish mean reversion signal (Z-score: {z_score:.2f}, Price vs BB: {price_vs_bb:.2f})")
        elif z_score < -1 and price_vs_bb < 0.3:
            score += 1  # Moderate bullish mean reversion signal
            reasoning.append(f"Moderate bullish mean reversion signal (Z-score: {z_score:.2f}, Price vs BB: {price_vs_bb:.2f})")
        elif z_score > 1 and price_vs_bb > 0.7:
            score -= 1  # Moderate bearish mean reversion signal
            reasoning.append(f"Moderate bearish mean reversion signal (Z-score: {z_score:.2f}, Price vs BB: {price_vs_bb:.2f})")
        else:
            reasoning.append(f"No strong mean reversion signal (Z-score: {z_score:.2f})")
        
        # RSI signals
        if rsi_14 < 30:
            score += 1  # Oversold condition
            reasoning.append(f"Oversold condition (RSI-14: {rsi_14:.1f})")
        elif rsi_14 > 70:
            score -= 1  # Overbought condition
            reasoning.append(f"Overbought condition (RSI-14: {rsi_14:.1f})")
        elif 40 <= rsi_14 <= 60:
            reasoning.append(f"Neutral RSI reading (RSI-14: {rsi_14:.1f})")
        
        # RSI divergence
        if abs(rsi_14 - rsi_28) > 10:
            if rsi_14 < rsi_28:
                score += 0.5  # Bullish divergence
                reasoning.append(f"Bullish RSI divergence")
            else:
                score -= 0.5  # Bearish divergence
                reasoning.append(f"Bearish RSI divergence")
        
        result["score"] = max(0, min(10, score))  # Clamp between 0-10
        result["details"] = reasoning
        return result

    def get_markdown(self, analysis:dict):
        """
        Convert analysis to markdown.
        """
        markdown_content = markdown.analysis_data(analysis)
        return markdown_content

    def __call__(self, state: AgentState, config: RunnableConfig, writer: StreamWriter) -> Dict[str, Any]:
        context = state.get('context')
        analysis_data = context.get('analysis_data')
        if analysis_data is None:
            analysis_data = {}
            context['analysis_data'] = analysis_data
        prices = context.get('prices')
        analysis = self.analyze(prices)
        analysis['type'] = 'mean_reversion_analysis'
        analysis['title'] = f'Mean Reversion Analysis'

        analysis_data['mean_reversion_analysis'] = analysis
        ai_message = AIMessage(content=self.get_markdown(analysis))
        return {
            "context": context,
            "messages": [
                ai_message
            ]
        }