from common.agent_state import AgentState
from langchain.schema import AIMessage
from langchain_core.callbacks import dispatch_custom_event
from langchain_core.runnables import RunnableConfig
from typing import Dict, Any
import time
from common import markdown
from langgraph.types import StreamWriter
import math


class MomentumAnalysis():
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
    
    def calculate_momentum(self, returns: list, period: int) -> float:
        """Calculate momentum over a given period"""
        if len(returns) < period:
            return 0
        
        # Sum returns over the period
        momentum = sum(returns[-period:])
        return momentum
    
    def calculate_volume_momentum(self, prices: list, period: int = 21) -> float:
        """Calculate volume momentum relative to moving average"""
        if len(prices) < period:
            return 1  # Neutral
        
        volumes = [price.get('volume', 0) for price in prices[-period:] if price.get('volume')]
        if len(volumes) < period:
            return 1  # Neutral
        
        current_volume = volumes[-1]
        avg_volume = sum(volumes) / len(volumes)
        
        if avg_volume > 0:
            return current_volume / avg_volume
        return 1  # Neutral
    
    def analyze(self, prices: list) -> dict[str, any]:
        """Analyze multi-factor momentum strategy."""
        result = {"score": 0, "max_score": 10, "details": [], "indicators": {}}
        if not prices or len(prices) < 63:  # Need at least 63 days for 3M momentum
            result["details"].append('Insufficient price data for momentum analysis (need at least 63 days)')
            return result

        # Calculate returns
        returns = self.calculate_returns(prices)
        if len(returns) < 63:
            result["details"].append('Insufficient return data for momentum analysis')
            return result
        
        # Price momentum for different periods
        mom_1m = self.calculate_momentum(returns, 21)  # 1 month
        mom_3m = self.calculate_momentum(returns, 63)  # 3 months
        mom_6m = self.calculate_momentum(returns, 126) # 6 months
        
        # Volume momentum
        volume_momentum = self.calculate_volume_momentum(prices, 21)
        
        # Store indicators
        result["indicators"] = {
            "momentum_1m": mom_1m,
            "momentum_3m": mom_3m,
            "momentum_6m": mom_6m,
            "volume_momentum": volume_momentum
        }
        
        score = 5  # Start with neutral score
        reasoning = []
        
        # Calculate momentum score (weighted average)
        momentum_score = (0.4 * mom_1m + 0.3 * mom_3m + 0.3 * mom_6m)
        
        # Volume confirmation
        volume_confirmation = volume_momentum > 1.0
        
        # Momentum scoring
        if momentum_score > 5 and volume_confirmation:
            score += 2  # Strong bullish momentum
            reasoning.append(f"Strong bullish momentum with volume confirmation (1M: {mom_1m:+.2f}%, 3M: {mom_3m:+.2f}%, 6M: {mom_6m:+.2f}%)")
        elif momentum_score > 2:
            score += 1  # Moderate bullish momentum
            reasoning.append(f"Moderate bullish momentum (1M: {mom_1m:+.2f}%, 3M: {mom_3m:+.2f}%, 6M: {mom_6m:+.2f}%)")
        elif momentum_score < -5 and volume_confirmation:
            score -= 2  # Strong bearish momentum
            reasoning.append(f"Strong bearish momentum with volume confirmation (1M: {mom_1m:+.2f}%, 3M: {mom_3m:+.2f}%, 6M: {mom_6m:+.2f}%)")
        elif momentum_score < -2:
            score -= 1  # Moderate bearish momentum
            reasoning.append(f"Moderate bearish momentum (1M: {mom_1m:+.2f}%, 3M: {mom_3m:+.2f}%, 6M: {mom_6m:+.2f}%)")
        else:
            reasoning.append(f"Neutral momentum (1M: {mom_1m:+.2f}%, 3M: {mom_3m:+.2f}%, 6M: {mom_6m:+.2f}%)")
        
        # Volume analysis
        if volume_momentum > 1.5:
            if momentum_score > 0:
                score += 0.5  # Strong volume confirmation
                reasoning.append(f"Strong volume confirmation (Volume/MA: {volume_momentum:.2f}x)")
            else:
                score -= 0.5  # Bearish volume divergence
                reasoning.append(f"Bearish volume divergence (Volume/MA: {volume_momentum:.2f}x)")
        elif volume_momentum < 0.8:
            reasoning.append(f"Below average volume (Volume/MA: {volume_momentum:.2f}x)")
        else:
            reasoning.append(f"Normal volume levels (Volume/MA: {volume_momentum:.2f}x)")
        
        # Recent momentum acceleration
        if len(returns) >= 42:  # Need at least 42 days for comparison
            recent_mom = sum(returns[-21:])  # Last 1 month
            prior_mom = sum(returns[-42:-21])  # Prior 1 month
            if recent_mom > prior_mom and recent_mom > 0:
                score += 0.5  # Momentum acceleration
                reasoning.append(f"Momentum accelerating in recent period")
            elif recent_mom < prior_mom and recent_mom < 0:
                score -= 0.5  # Momentum deceleration
                reasoning.append(f"Momentum decelerating in recent period")
        
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
        analysis['type'] = 'momentum_analysis'
        analysis['title'] = f'Momentum Analysis'

        analysis_data['momentum_analysis'] = analysis
        ai_message = AIMessage(content=self.get_markdown(analysis))
        return {
            "context": context,
            "messages": [
                ai_message
            ]
        }