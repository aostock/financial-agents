from common.agent_state import AgentState
from langchain.schema import AIMessage
from langchain_core.callbacks import dispatch_custom_event
from langchain_core.runnables import RunnableConfig
from typing import Dict, Any

import time
from common import markdown
from langgraph.types import StreamWriter


class ValuationAnalysis():
    def __init__(self, options: Dict[str, Any]):
        self.options = options

    
    def analyze(self, metrics: list) -> dict[str, any]:
        """Analyze valuation from a macro-oriented perspective."""
        result = {"score": 0, "max_score": 10, "details": []}
        if not metrics:
            result["details"].append('No metrics available for valuation analysis')
            return result

        latest_metrics = metrics[0]
        score = 0
        reasoning = []
        
        # P/E ratio analysis
        if latest_metrics.get('price_to_earnings_ratio'):
            pe_ratio = latest_metrics['price_to_earnings_ratio']
            if pe_ratio < 10:
                score += 2
                reasoning.append(f"Attractive valuation: P/E ratio of {pe_ratio:.1f}")
            elif pe_ratio < 15:
                score += 1
                reasoning.append(f"Reasonable valuation: P/E ratio of {pe_ratio:.1f}")
            elif pe_ratio > 30:
                score -= 2
                reasoning.append(f"Rich valuation: P/E ratio of {pe_ratio:.1f}")
            elif pe_ratio > 20:
                score -= 1
                reasoning.append(f"Elevated valuation: P/E ratio of {pe_ratio:.1f}")
            else:
                reasoning.append(f"Fair valuation: P/E ratio of {pe_ratio:.1f}")
        
        # EV/EBIT ratio (enterprise value to earnings)
        if latest_metrics.get('enterprise_value') and latest_metrics.get('ebit'):
            if latest_metrics['ebit'] > 0:  # Avoid division by zero or negative EBIT
                ev_ebit = latest_metrics['enterprise_value'] / latest_metrics['ebit']
                if ev_ebit < 10:
                    score += 2
                    reasoning.append(f"Compelling EV/EBIT: {ev_ebit:.1f}")
                elif ev_ebit < 15:
                    score += 1
                    reasoning.append(f"Reasonable EV/EBIT: {ev_ebit:.1f}")
                elif ev_ebit > 25:
                    score -= 2
                    reasoning.append(f"Expensive EV/EBIT: {ev_ebit:.1f}")
                elif ev_ebit > 20:
                    score -= 1
                    reasoning.append(f"Elevated EV/EBIT: {ev_ebit:.1f}")
                else:
                    reasoning.append(f"Fair EV/EBIT: {ev_ebit:.1f}")
            else:
                reasoning.append("Negative or zero EBIT - EV/EBIT not meaningful")
        
        # Price to book ratio
        if latest_metrics.get('market_cap') and latest_metrics.get('stockholders_equity'):
            if latest_metrics['stockholders_equity'] > 0:  # Avoid division by zero
                price_to_book = latest_metrics['market_cap'] / latest_metrics['stockholders_equity']
                if price_to_book < 1.5:
                    score += 2
                    reasoning.append(f"Undervalued on P/B basis: {price_to_book:.2f}")
                elif price_to_book < 3:
                    score += 1
                    reasoning.append(f"Reasonable P/B ratio: {price_to_book:.2f}")
                elif price_to_book > 5:
                    score -= 2
                    reasoning.append(f"Overvalued on P/B basis: {price_to_book:.2f}")
                elif price_to_book > 4:
                    score -= 1
                    reasoning.append(f"Elevated P/B ratio: {price_to_book:.2f}")
                else:
                    reasoning.append(f"Fair P/B ratio: {price_to_book:.2f}")
            else:
                reasoning.append("Negative book value - P/B not meaningful")
        
        # Free cash flow yield
        if latest_metrics.get('free_cash_flow') and latest_metrics.get('market_cap'):
            if latest_metrics['market_cap'] > 0:  # Avoid division by zero
                fcf_yield = latest_metrics['free_cash_flow'] / latest_metrics['market_cap'] * 100
                if fcf_yield > 8:
                    score += 2
                    reasoning.append(f"High FCF yield: {fcf_yield:.1f}%")
                elif fcf_yield > 5:
                    score += 1
                    reasoning.append(f"Decent FCF yield: {fcf_yield:.1f}%")
                elif fcf_yield < 2:
                    score -= 2
                    reasoning.append(f"Low FCF yield: {fcf_yield:.1f}%")
                elif fcf_yield < 3:
                    score -= 1
                    reasoning.append(f"Modest FCF yield: {fcf_yield:.1f}%")
                else:
                    reasoning.append(f"Moderate FCF yield: {fcf_yield:.1f}%")
            else:
                reasoning.append("Zero market cap - FCF yield not meaningful")
        
        # Compare with sector/market (using beta as a proxy)
        if latest_metrics.get('price_to_earnings_ratio') and latest_metrics.get('beta'):
            pe_ratio = latest_metrics['price_to_earnings_ratio']
            beta = latest_metrics['beta']
            # Higher beta companies should trade at lower P/E ratios for equivalent risk
            expected_pe = 15 - (beta - 1) * 5  # Simple model: base P/E of 15, adjusted for beta
            if pe_ratio < expected_pe * 0.8:
                score += 1
                reasoning.append(f"P/E discount to risk-adjusted benchmark")
            elif pe_ratio > expected_pe * 1.2:
                score -= 1
                reasoning.append(f"P/E premium to risk-adjusted benchmark")
            else:
                reasoning.append(f"P/E in line with risk-adjusted benchmark")

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
        metrics = context.get('metrics')
        analysis = self.analyze(metrics)
        analysis['type'] = 'valuation_analysis'
        analysis['title'] = f'Valuation Analysis'

        analysis_data['valuation_analysis'] = analysis
        ai_message = AIMessage(content=self.get_markdown(analysis))
        return {
            "context": context,
            "messages": [
                ai_message
            ]
        }