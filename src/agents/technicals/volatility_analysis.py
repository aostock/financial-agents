from common.agent_state import AgentState
from langchain.schema import AIMessage
from langchain_core.callbacks import dispatch_custom_event
from langchain_core.runnables import RunnableConfig
from typing import Dict, Any
import time
from common import markdown
from langgraph.types import StreamWriter
import math


class VolatilityAnalysis():
    def __init__(self, options: Dict[str, Any]):
        self.options = options

    
    def calculate_returns(self, prices: list) -> list:
        """Calculate daily percentage returns"""
        if len(prices) < 2:
            return []
        
        returns = []
        for i in range(1, len(prices)):
            prev_close = prices[i-1].get('close', 0)
            curr_close = prices[i].get('close', 0)
            if prev_close > 0:
                ret = (curr_close - prev_close) / prev_close * 100
                returns.append(ret)
            else:
                returns.append(0)
        return returns
    
    def calculate_std(self, data: list) -> float:
        """Calculate Standard Deviation"""
        if len(data) < 2:
            return 0
            
        mean = sum(data) / len(data)
        variance = sum((x - mean) ** 2 for x in data) / (len(data) - 1)
        return math.sqrt(variance)
    
    def calculate_atr(self, prices: list, period: int = 14) -> float:
        """Calculate Average True Range"""
        if len(prices) < period + 1:
            return 0
        
        tr_values = []
        for i in range(1, len(prices)):
            high = prices[i].get('high', 0)
            low = prices[i].get('low', 0)
            prev_close = prices[i-1].get('close', 0)
            
            # True Range
            tr1 = high - low
            tr2 = abs(high - prev_close)
            tr3 = abs(low - prev_close)
            tr = max(tr1, tr2, tr3)
            tr_values.append(tr)
        
        if len(tr_values) < period:
            return 0
            
        # Simple moving average of TR values
        atr = sum(tr_values[-period:]) / period
        return atr
    
    def analyze(self, prices: list) -> dict[str, any]:
        """Analyze volatility-based trading strategy."""
        result = {"score": 0, "max_score": 10, "details": [], "indicators": {}}
        if not prices or len(prices) < 63:  # Need at least 63 days for volatility analysis
            result["details"].append('Insufficient price data for volatility analysis (need at least 63 days)')
            return result

        # Calculate returns
        returns = self.calculate_returns(prices)
        if len(returns) < 21:
            result["details"].append('Insufficient return data for volatility analysis')
            return result
        
        # Calculate various volatility metrics
        # Historical volatility (21-day)
        hist_vol_21 = self.calculate_std(returns[-21:]) if len(returns) >= 21 else 0
        # Historical volatility (63-day)
        hist_vol_63 = self.calculate_std(returns[-63:]) if len(returns) >= 63 else 0
        
        # Annualized volatility
        annualized_vol = hist_vol_21 * math.sqrt(252) if hist_vol_21 > 0 else 0
        
        # Volatility regime detection
        if len(returns) >= 84:  # Need 84 days for 63-day MA + 21-day current
            vol_ma_63 = self.calculate_std(returns[-84:-21]) if len(returns) >= 84 else 0
            vol_regime = hist_vol_21 / vol_ma_63 if vol_ma_63 > 0 else 1
        else:
            vol_regime = 1
        
        # Volatility mean reversion (z-score)
        if len(returns) >= 84:
            vol_std_63 = self.calculate_std([self.calculate_std(returns[i:i+21]) for i in range(len(returns)-84, len(returns)-21, 1)]) if len(returns) >= 84 else 0
            vol_z_score = (hist_vol_21 - (sum([self.calculate_std(returns[i:i+21]) for i in range(len(returns)-84, len(returns)-21, 1)]) / 63)) / vol_std_63 if vol_std_63 > 0 else 0
        else:
            vol_z_score = 0
        
        # ATR ratio
        atr = self.calculate_atr(prices, 14)
        current_price = prices[-1].get('close', 0) if prices else 0
        atr_ratio = atr / current_price if current_price > 0 and atr > 0 else 0
        
        # Store indicators
        result["indicators"] = {
            "historical_volatility_21d": hist_vol_21,
            "historical_volatility_63d": hist_vol_63,
            "annualized_volatility": annualized_vol,
            "volatility_regime": vol_regime,
            "volatility_z_score": vol_z_score,
            "atr": atr,
            "atr_ratio": atr_ratio
        }
        
        score = 5  # Start with neutral score
        reasoning = []
        
        # Generate signal based on volatility regime
        if vol_regime < 0.8 and vol_z_score < -1:
            score += 2  # Low vol regime, potential for expansion
            reasoning.append(f"Low volatility regime with mean reversion signal - potential for expansion")
        elif vol_regime > 1.2 and vol_z_score > 1:
            score -= 2  # High vol regime, potential for contraction
            reasoning.append(f"High volatility regime with mean reversion signal - potential for contraction")
        elif vol_regime < 0.9:
            score += 1  # Moderately low volatility
            reasoning.append(f"Moderately low volatility regime - cautious bullish")
        elif vol_regime > 1.1:
            score -= 1  # Moderately high volatility
            reasoning.append(f"Moderately high volatility regime - cautious bearish")
        else:
            reasoning.append(f"Normal volatility regime")
        
        # ATR analysis
        if atr_ratio > 0.02:  # 2% ATR ratio is high
            if score > 5:  # Already bullish
                score += 0.5  # High volatility confirming bullish signal
                reasoning.append(f"High volatility confirming trend (ATR ratio: {atr_ratio:.3f})")
            else:
                score -= 0.5  # High volatility increasing risk
                reasoning.append(f"High volatility increasing risk (ATR ratio: {atr_ratio:.3f})")
        elif atr_ratio < 0.005:  # 0.5% ATR ratio is low
            reasoning.append(f"Low volatility environment (ATR ratio: {atr_ratio:.3f})")
        else:
            reasoning.append(f"Moderate volatility environment (ATR ratio: {atr_ratio:.3f})")
        
        # Volatility trend
        if len(returns) >= 42:
            recent_vol = self.calculate_std(returns[-21:])  # Last 21 days
            prior_vol = self.calculate_std(returns[-42:-21])  # Prior 21 days
            if recent_vol > prior_vol * 1.2:
                score -= 0.5  # Volatility increasing
                reasoning.append(f"Volatility increasing rapidly")
            elif recent_vol < prior_vol * 0.8:
                score += 0.5  # Volatility decreasing
                reasoning.append(f"Volatility decreasing")
        
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
        analysis['type'] = 'volatility_analysis'
        analysis['title'] = f'Volatility Analysis'

        analysis_data['volatility_analysis'] = analysis
        ai_message = AIMessage(content=self.get_markdown(analysis))
        return {
            "context": context,
            "messages": [
                ai_message
            ]
        }