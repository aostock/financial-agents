from common.agent_state import AgentState
from langchain.schema import AIMessage
from langchain_core.callbacks import dispatch_custom_event
from langchain_core.runnables import RunnableConfig
from typing import Dict, Any
import time
from common import markdown
from langgraph.types import StreamWriter
import math


class TrendAnalysis():
    def __init__(self, options: Dict[str, Any]):
        self.options = options

    
    def calculate_ema(self, prices: list, period: int) -> float:
        """Calculate Exponential Moving Average"""
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
    
    def calculate_adx(self, prices: list, period: int = 14) -> dict:
        """Calculate Average Directional Index (simplified version)"""
        if len(prices) < period + 1:
            return {"adx": 0, "+di": 0, "-di": 0}
        
        # Simplified ADX calculation
        closes = [price.get('close', 0) for price in prices[-(period+1):] if price.get('close')]
        highs = [price.get('high', 0) for price in prices[-(period+1):] if price.get('high')]
        lows = [price.get('low', 0) for price in prices[-(period+1):] if price.get('low')]
        
        if len(closes) < period + 1:
            return {"adx": 0, "+di": 0, "-di": 0}
        
        # Calculate True Range and Directional Movement (simplified)
        tr_values = []
        plus_dm_values = []
        minus_dm_values = []
        
        for i in range(1, len(closes)):
            high = highs[i]
            low = lows[i]
            prev_close = closes[i-1]
            prev_high = highs[i-1]
            prev_low = lows[i-1]
            
            # True Range
            tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
            tr_values.append(tr)
            
            # Directional Movement
            up_move = high - prev_high
            down_move = prev_low - low
            
            if up_move > down_move and up_move > 0:
                plus_dm = up_move
                minus_dm = 0
            elif down_move > up_move and down_move > 0:
                plus_dm = 0
                minus_dm = down_move
            else:
                plus_dm = 0
                minus_dm = 0
                
            plus_dm_values.append(plus_dm)
            minus_dm_values.append(minus_dm)
        
        if len(tr_values) == 0:
            return {"adx": 0, "+di": 0, "-di": 0}
        
        # Calculate averages
        avg_tr = sum(tr_values) / len(tr_values)
        avg_plus_dm = sum(plus_dm_values) / len(plus_dm_values)
        avg_minus_dm = sum(minus_dm_values) / len(minus_dm_values)
        
        if avg_tr == 0:
            return {"adx": 0, "+di": 0, "-di": 0}
        
        # Calculate DI+/DI-
        plus_di = 100 * (avg_plus_dm / avg_tr)
        minus_di = 100 * (avg_minus_dm / avg_tr)
        
        # Calculate DX
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di) if (plus_di + minus_di) != 0 else 0
        
        # ADX is smoothed DX
        adx = dx  # Simplified - in reality, this would be smoothed over multiple periods
        
        return {"adx": adx, "+di": plus_di, "-di": minus_di}
    
    def analyze(self, prices: list) -> dict[str, any]:
        """Analyze trend following strategy using multiple timeframes and indicators."""
        result = {"score": 0, "max_score": 10, "details": [], "indicators": {}}
        if not prices or len(prices) < 55:
            result["details"].append('Insufficient price data for trend analysis (need at least 55 days)')
            return result

        # Calculate EMAs for multiple timeframes
        ema_8 = self.calculate_ema(prices, 8)
        ema_21 = self.calculate_ema(prices, 21)
        ema_55 = self.calculate_ema(prices, 55)
        
        # Calculate ADX for trend strength
        adx_data = self.calculate_adx(prices, 14)
        adx = adx_data["adx"]
        
        # Store indicators
        result["indicators"] = {
            "ema_8": ema_8,
            "ema_21": ema_21,
            "ema_55": ema_55,
            "adx": adx,
            "plus_di": adx_data["+di"],
            "minus_di": adx_data["-di"]
        }
        
        # Determine trend direction and strength
        current_price = prices[-1].get('close', 0) if prices else 0
        short_trend = ema_8 > ema_21 if ema_8 > 0 and ema_21 > 0 else False
        medium_trend = ema_21 > ema_55 if ema_21 > 0 and ema_55 > 0 else False
        
        # Combine signals with confidence weighting
        trend_strength = adx / 100.0 if adx > 0 else 0
        
        score = 5  # Start with neutral score
        reasoning = []
        
        # Trend direction scoring
        if short_trend and medium_trend:
            score += 2  # Strong bullish trend
            reasoning.append(f"Strong bullish trend confirmed across multiple timeframes")
        elif not short_trend and not medium_trend:
            score -= 2  # Strong bearish trend
            reasoning.append(f"Strong bearish trend confirmed across multiple timeframes")
        elif short_trend and not medium_trend:
            score += 1  # Weak bullish trend
            reasoning.append(f"Weak bullish trend on short-term timeframe")
        elif not short_trend and medium_trend:
            score += 0.5  # Mixed trend signals
            reasoning.append(f"Mixed trend signals - short-term bearish, long-term bullish")
        else:
            reasoning.append(f"No clear trend direction")
        
        # Trend strength scoring
        if adx > 25:
            if short_trend and medium_trend:
                score += 1  # Strong trend with strong direction
                reasoning.append(f"Strong trend strength (ADX: {adx:.1f})")
            elif not short_trend and not medium_trend:
                score -= 1  # Strong trend with strong direction (bearish)
                reasoning.append(f"Strong trend strength (ADX: {adx:.1f})")
        elif adx < 20:
            reasoning.append(f"Weak trend strength (ADX: {adx:.1f})")
        
        # Price vs EMA positioning
        if current_price > 0 and ema_8 > 0 and ema_21 > 0:
            if current_price > ema_8 > ema_21:
                score += 0.5
                reasoning.append(f"Price above key moving averages")
            elif current_price < ema_8 < ema_21:
                score -= 0.5
                reasoning.append(f"Price below key moving averages")
        
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
        analysis['type'] = 'trend_analysis'
        analysis['title'] = f'Trend Analysis'

        analysis_data['trend_analysis'] = analysis
        ai_message = AIMessage(content=self.get_markdown(analysis))
        return {
            "context": context,
            "messages": [
                ai_message
            ]
        }