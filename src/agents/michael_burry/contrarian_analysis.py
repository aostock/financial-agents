from common.agent_state import AgentState
from langchain.schema import AIMessage
from langchain_core.callbacks import dispatch_custom_event
from langchain_core.runnables import RunnableConfig
from typing import Dict, Any

import time
from common import markdown
from langgraph.types import StreamWriter


class ContrarianAnalysis():
    def __init__(self, options: Dict[str, Any]):
        self.options = options

    
    def analyze(self, metrics: list) -> dict[str, any]:
        """Analyze contrarian investment opportunities and market sentiment."""
        result = {"score": 0, "max_score": 10, "details": []}
        if not metrics or len(metrics) < 3:
            result["details"].append('Insufficient historical data (need at least 3 years)')
            return result

        latest_metrics = metrics[0]
        previous_metrics = metrics[1] if len(metrics) > 1 else None

        score = 0
        reasoning = []

        # 1. Contrarian valuation signals
        # Look for stocks that are out of favor but fundamentally sound
        if latest_metrics.get('price_to_earnings_ratio'):
            pe_ratio = latest_metrics['price_to_earnings_ratio']
            
            # Very low P/E might indicate market pessimism on a good business
            if pe_ratio < 8 and pe_ratio > 0:
                score += 2
                reasoning.append(f"Extreme value P/E ratio ({pe_ratio:.1f}x) - potential market pessimism")
            elif pe_ratio < 12 and pe_ratio > 0:
                score += 1
                reasoning.append(f"Low P/E ratio ({pe_ratio:.1f}x) - possible market pessimism")

        # 2. Sentiment contrarian indicators
        # Compare current performance to recent trends
        if len(metrics) >= 3:
            # Check if earnings have been improving while market may not have recognized it
            earnings = [m.get('net_income') for m in metrics[:3] if m.get('net_income') is not None]
            if len(earnings) >= 3:
                recent_trend = (earnings[0] - earnings[2]) / earnings[2] if earnings[2] != 0 else 0
                if recent_trend > 0.2:  # 20%+ earnings improvement
                    score += 2
                    reasoning.append(f"Significant earnings improvement ({recent_trend:.1%}) not yet reflected in valuation")
                elif recent_trend > 0.1:  # 10%+ earnings improvement
                    score += 1
                    reasoning.append(f"Moderate earnings improvement ({recent_trend:.1%})")

        # 3. Market positioning contrarian analysis
        # Look for companies in out-of-favor sectors or industries
        if latest_metrics.get('return_on_equity') and latest_metrics.get('price_to_earnings_ratio'):
            roe = latest_metrics['return_on_equity']
            pe_ratio = latest_metrics['price_to_earnings_ratio']
            
            # High ROE with low P/E suggests market may be ignoring quality
            if roe > 0.15 and pe_ratio < 15:
                score += 3
                reasoning.append(f"High quality business (ROE: {roe:.1%}) trading at low multiple (P/E: {pe_ratio:.1f}x)")
            elif roe > 0.12 and pe_ratio < 12:
                score += 2
                reasoning.append(f"Good quality business (ROE: {roe:.1%}) trading at attractive multiple (P/E: {pe_ratio:.1f}x)")

        # 4. Institutional sentiment contrarian signals
        # This would require institutional ownership data, but we can infer from valuation metrics
        if latest_metrics.get('price_to_book_ratio'):
            pb_ratio = latest_metrics['price_to_book_ratio']
            
            # Very low P/B might indicate institutional abandonment
            if pb_ratio < 1.0:
                score += 2
                reasoning.append(f"Below book value (P/B: {pb_ratio:.2f}x) - potential institutional abandonment")
            elif pb_ratio < 1.5:
                score += 1
                reasoning.append(f"Discount to book value (P/B: {pb_ratio:.2f}x)")

        # 5. Contrarian momentum analysis
        # Look for stocks that have been declining but show fundamental strength
        if (len(metrics) >= 3 and latest_metrics.get('market_cap') and 
            metrics[2].get('market_cap') and metrics[2].get('market_cap') > 0):
            
            market_cap_current = latest_metrics['market_cap']
            market_cap_past = metrics[2]['market_cap']
            price_change = (market_cap_current - market_cap_past) / market_cap_past
            
            if price_change < -0.2:  # 20%+ decline
                # Check if fundamentals are still strong
                if latest_metrics.get('return_on_equity') and latest_metrics['return_on_equity'] > 0.12:
                    score += 3
                    reasoning.append(f"Significant price decline ({price_change:.1%}) with strong fundamentals (ROE: {latest_metrics['return_on_equity']:.1%})")
                elif latest_metrics.get('return_on_equity') and latest_metrics['return_on_equity'] > 0.08:
                    score += 2
                    reasoning.append(f"Price decline ({price_change:.1%}) with decent fundamentals (ROE: {latest_metrics['return_on_equity']:.1%})")
            elif price_change < -0.1:  # 10%+ decline
                if latest_metrics.get('return_on_equity') and latest_metrics['return_on_equity'] > 0.15:
                    score += 2
                    reasoning.append(f"Moderate price decline ({price_change:.1%}) with strong fundamentals (ROE: {latest_metrics['return_on_equity']:.1%})")

        # 6. Contrarian quality analysis
        # Look for high-quality businesses trading at bargain prices
        quality_score = 0
        quality_reasons = []
        
        # High ROE
        if latest_metrics.get('return_on_equity') and latest_metrics['return_on_equity'] > 0.15:
            quality_score += 1
            quality_reasons.append("High ROE")
            
        # Low debt
        if latest_metrics.get('debt_to_equity') and latest_metrics['debt_to_equity'] < 0.3:
            quality_score += 1
            quality_reasons.append("Low debt")
            
        # Strong margins
        if latest_metrics.get('operating_margin') and latest_metrics['operating_margin'] > 0.15:
            quality_score += 1
            quality_reasons.append("Strong operating margins")
            
        # Good liquidity
        if latest_metrics.get('current_ratio') and latest_metrics['current_ratio'] > 2.0:
            quality_score += 1
            quality_reasons.append("Strong liquidity")
            
        # If high quality but potentially undervalued
        if quality_score >= 3:  # Strong quality metrics
            if latest_metrics.get('price_to_earnings_ratio') and latest_metrics['price_to_earnings_ratio'] < 15:
                score += 2
                reasoning.append(f"High-quality business ({', '.join(quality_reasons)}) trading at bargain multiple (P/E: {latest_metrics['price_to_earnings_ratio']:.1f}x)")

        result["score"] = score
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
        metrics = context.get('metrics')
        analysis = self.analyze(metrics)
        analysis['type'] = 'contrarian_analysis'
        analysis['title'] = f'Contrarian Analysis'

        analysis_data['contrarian_analysis'] = analysis
        ai_message = AIMessage(content=self.get_markdown(analysis))
        return {
            "context": context,
            "messages": [
                ai_message
            ]
        }