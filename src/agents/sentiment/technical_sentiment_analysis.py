from common.agent_state import AgentState
from langchain.schema import AIMessage
from langchain_core.callbacks import dispatch_custom_event
from langchain_core.runnables import RunnableConfig
from typing import Dict, Any

import time
from common import markdown
from langgraph.types import StreamWriter
import re


class TechnicalSentimentAnalysis():
    def __init__(self, options: Dict[str, Any]):
        self.options = options

    
    def analyze(self, prices: list) -> dict[str, any]:
        """Analyze technical sentiment based on price action and momentum indicators."""
        result = {"score": 0, "max_score": 10, "details": [], "technical_indicators": {}}
        if not prices or len(prices) < 10:
            result["details"].append('Insufficient price data for technical analysis (need at least 10 days)')
            return result

        # Calculate simple moving averages
        closes = [price.get('close', 0) for price in prices if price.get('close')]
        if len(closes) < 10:
            result["details"].append('Insufficient closing price data')
            return result

        # Calculate 5-day and 20-day moving averages
        if len(closes) >= 5:
            ma5 = sum(closes[-5:]) / 5
        else:
            ma5 = closes[-1] if closes else 0
            
        if len(closes) >= 20:
            ma20 = sum(closes[-20:]) / 20
        else:
            ma20 = sum(closes) / len(closes) if closes else 0

        current_price = closes[-1] if closes else 0

        # Calculate price momentum (percent change over last 5 days)
        momentum = 0
        if len(closes) >= 5 and closes[-5] != 0:
            momentum = (current_price - closes[-5]) / closes[-5] * 100

        # Calculate volume trend (if volume data available)
        volumes = [price.get('volume', 0) for price in prices if price.get('volume')]
        avg_volume = 0
        volume_trend = 0
        
        if volumes and len(volumes) >= 5:
            avg_volume = sum(volumes[-10:]) / 10 if len(volumes) >= 10 else sum(volumes) / len(volumes)
            if avg_volume > 0 and volumes[-1] > 0:
                volume_trend = (volumes[-1] - avg_volume) / avg_volume * 100

        result["technical_indicators"] = {
            "current_price": current_price,
            "moving_average_5": ma5,
            "moving_average_20": ma20,
            "momentum_5d": momentum,
            "current_volume": volumes[-1] if volumes else 0,
            "average_volume": avg_volume,
            "volume_trend": volume_trend
        }

        # Calculate technical sentiment score based on indicators
        score = 5  # Start with neutral score
        
        # Price vs moving averages
        if ma5 > 0 and ma20 > 0:
            # Short-term vs long-term MA
            if ma5 > ma20:
                score += 1  # Bullish crossover
            else:
                score -= 1  # Bearish crossover
                
            # Current price vs moving averages
            if current_price > ma5:
                score += 1  # Price above short-term MA
            else:
                score -= 1  # Price below short-term MA
                
            if current_price > ma20:
                score += 1  # Price above long-term MA
            else:
                score -= 1  # Price below long-term MA

        # Momentum factor
        if momentum > 5:
            score += 1  # Strong positive momentum
        elif momentum > 2:
            score += 0.5  # Moderate positive momentum
        elif momentum < -5:
            score -= 1  # Strong negative momentum
        elif momentum < -2:
            score -= 0.5  # Moderate negative momentum

        # Volume trend
        if volume_trend > 20:
            score += 1  # Above average volume confirming trend
        elif volume_trend < -20:
            score -= 1  # Below average volume, trend may be weakening

        # Clamp score between 0-10
        score = max(0, min(10, score))
        result["score"] = score

        # Add descriptive details
        if score >= 7:
            result["details"].append(f"Positive technical sentiment with bullish price action and momentum ({momentum:+.1f}% 5-day momentum)")
        elif score >= 3:
            result["details"].append(f"Neutral technical sentiment with mixed signals ({momentum:+.1f}% 5-day momentum)")
        else:
            result["details"].append(f"Negative technical sentiment with bearish price action ({momentum:+.1f}% 5-day momentum)")

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
        analysis['type'] = 'technical_sentiment_analysis'
        analysis['title'] = f'Technical Sentiment Analysis'

        analysis_data['technical_sentiment_analysis'] = analysis
        ai_message = AIMessage(content=self.get_markdown(analysis))
        return {
            "context": context,
            "messages": [
                ai_message
            ]
        }