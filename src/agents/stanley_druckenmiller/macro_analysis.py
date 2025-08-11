from common.agent_state import AgentState
from langchain.schema import AIMessage
from langchain_core.callbacks import dispatch_custom_event
from langchain_core.runnables import RunnableConfig
from typing import Dict, Any

import time
from common import markdown
from langgraph.types import StreamWriter


class MacroAnalysis():
    def __init__(self, options: Dict[str, Any]):
        self.options = options

    
    def analyze(self, prices: list) -> dict[str, any]:
        """Analyze macroeconomic factors affecting the investment."""
        result = {"score": 0, "max_score": 10, "details": []}
        if not prices or len(prices) < 20:
            result["details"].append('Insufficient price data for macro analysis')
            return result

        # Calculate price trends over different time periods
        closes = [price.get('close', 0) for price in prices if price.get('close')]
        if len(closes) < 20:
            result["details"].append('Insufficient closing price data')
            return result

        score = 0
        reasoning = []
        
        # Short-term trend (20-day)
        if len(closes) >= 20:
            short_term_trend = (closes[-1] - closes[-20]) / closes[-20] * 100
            if short_term_trend > 10:
                score += 2
                reasoning.append(f"Strong short-term momentum: {short_term_trend:+.1f}% (20-day)")
            elif short_term_trend > 5:
                score += 1
                reasoning.append(f"Positive short-term momentum: {short_term_trend:+.1f}% (20-day)")
            elif short_term_trend < -10:
                score -= 2
                reasoning.append(f"Severe short-term weakness: {short_term_trend:+.1f}% (20-day)")
            elif short_term_trend < -5:
                score -= 1
                reasoning.append(f"Negative short-term momentum: {short_term_trend:+.1f}% (20-day)")
            else:
                reasoning.append(f"Neutral short-term trend: {short_term_trend:+.1f}% (20-day)")

        # Medium-term trend (60-day)
        if len(closes) >= 60:
            medium_term_trend = (closes[-1] - closes[-60]) / closes[-60] * 100
            if medium_term_trend > 15:
                score += 2
                reasoning.append(f"Strong medium-term trend: {medium_term_trend:+.1f}% (60-day)")
            elif medium_term_trend > 5:
                score += 1
                reasoning.append(f"Positive medium-term trend: {medium_term_trend:+.1f}% (60-day)")
            elif medium_term_trend < -15:
                score -= 2
                reasoning.append(f"Severe medium-term weakness: {medium_term_trend:+.1f}% (60-day)")
            elif medium_term_trend < -5:
                score -= 1
                reasoning.append(f"Negative medium-term trend: {medium_term_trend:+.1f}% (60-day)")
            else:
                reasoning.append(f"Neutral medium-term trend: {medium_term_trend:+.1f}% (60-day)")

        # Long-term trend (120-day)
        if len(closes) >= 120:
            long_term_trend = (closes[-1] - closes[-120]) / closes[-120] * 100
            if long_term_trend > 20:
                score += 2
                reasoning.append(f"Strong long-term trend: {long_term_trend:+.1f}% (120-day)")
            elif long_term_trend > 10:
                score += 1
                reasoning.append(f"Positive long-term trend: {long_term_trend:+.1f}% (120-day)")
            elif long_term_trend < -20:
                score -= 2
                reasoning.append(f"Severe long-term weakness: {long_term_trend:+.1f}% (120-day)")
            elif long_term_trend < -10:
                score -= 1
                reasoning.append(f"Negative long-term trend: {long_term_trend:+.1f}% (120-day)")
            else:
                reasoning.append(f"Neutral long-term trend: {long_term_trend:+.1f}% (120-day)")

        # Volatility analysis
        if len(closes) >= 20:
            avg_price = sum(closes[-20:]) / 20
            volatility = (sum((p - avg_price) ** 2 for p in closes[-20:]) / 20) ** 0.5 / avg_price * 100
            if volatility < 2:
                score += 1
                reasoning.append(f"Low volatility environment: {volatility:.1f}% (20-day)")
            elif volatility > 5:
                score -= 1
                reasoning.append(f"High volatility environment: {volatility:.1f}% (20-day)")
            else:
                reasoning.append(f"Moderate volatility: {volatility:.1f}% (20-day)")

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
        analysis['type'] = 'macro_analysis'
        analysis['title'] = f'Macro Analysis'

        analysis_data['macro_analysis'] = analysis
        ai_message = AIMessage(content=self.get_markdown(analysis))
        return {
            "context": context,
            "messages": [
                ai_message
            ]
        }