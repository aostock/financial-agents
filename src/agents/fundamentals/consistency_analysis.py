from common.agent_state import AgentState
from langchain.schema import AIMessage
from langchain_core.callbacks import dispatch_custom_event
from langchain_core.runnables import RunnableConfig
from typing import Dict, Any

import time
from common import markdown
from langgraph.types import StreamWriter


class ConsistencyAnalysis():
    def __init__(self, options: Dict[str, Any]):
        self.options = options

    
    def analyze(self, metrics: list) -> dict[str, any]:
        """Analyze financial consistency and stability over time."""
        result = {"score": 0, "max_score": 10, "details": []}
        if not metrics or len(metrics) < 5:
            result["details"].append('Insufficient historical data (need at least 5 years)')
            return result

        score = 0
        reasoning = []

        # Get key metrics over time
        roes = [m.get('return_on_equity') for m in metrics if m.get('return_on_equity') is not None]
        margins = [m.get('operating_margin') for m in metrics if m.get('operating_margin') is not None]
        revenues = [m.get('revenue') for m in metrics if m.get('revenue') is not None]
        net_incomes = [m.get('net_income') for m in metrics if m.get('net_income') is not None]
        free_cash_flows = [m.get('free_cash_flow') for m in metrics if m.get('free_cash_flow') is not None]

        # Check ROE consistency (at least 70% of years > 10%)
        if len(roes) >= 3:
            strong_roe_years = sum(1 for roe in roes[:min(5, len(roes))] if roe > 0.10)
            if strong_roe_years >= len(roes[:min(5, len(roes))]) * 0.7:
                score += 3
                reasoning.append(f"Consistent strong ROE ({strong_roe_years}/{min(5, len(roes))} years > 10%)")
            elif strong_roe_years >= len(roes[:min(5, len(roes))]) * 0.5:
                score += 2
                reasoning.append(f"Moderately consistent ROE ({strong_roe_years}/{min(5, len(roes))} years > 10%)")
            else:
                reasoning.append(f"Inconsistent ROE ({strong_roe_years}/{min(5, len(roes))} years > 10%)")

        # Check margin stability
        if len(margins) >= 3:
            avg_margin = sum(margins[:min(5, len(margins))]) / len(margins[:min(5, len(margins))])
            margin_volatility = sum(abs(m - avg_margin) for m in margins[:min(5, len(margins))]) / len(margins[:min(5, len(margins))])
            
            if margin_volatility < 0.03:  # Less than 3% volatility
                score += 2
                reasoning.append(f"Stable operating margins (volatility: {margin_volatility:.1%} < 3%)")
            elif margin_volatility < 0.05:  # Less than 5% volatility
                score += 1
                reasoning.append(f"Moderately stable margins (volatility: {margin_volatility:.1%} < 5%)")
            else:
                reasoning.append(f"Unstable margins (volatility: {margin_volatility:.1%} > 5%)")

        # Check revenue growth consistency
        if len(revenues) >= 3:
            # Calculate year-over-year growth rates
            growth_rates = []
            for i in range(len(revenues[:min(5, len(revenues))])-1):
                if revenues[i+1] != 0:
                    growth_rate = (revenues[i] - revenues[i+1]) / revenues[i+1]
                    growth_rates.append(growth_rate)
            
            if len(growth_rates) > 0:
                positive_growth_years = sum(1 for g in growth_rates if g > 0)
                if positive_growth_years >= len(growth_rates) * 0.8:
                    score += 2
                    reasoning.append(f"Consistent revenue growth ({positive_growth_years}/{len(growth_rates)} years positive)")
                elif positive_growth_years >= len(growth_rates) * 0.6:
                    score += 1
                    reasoning.append(f"Mostly consistent revenue growth ({positive_growth_years}/{len(growth_rates)} years positive)")
                else:
                    reasoning.append(f"Erratic revenue growth ({positive_growth_years}/{len(growth_rates)} years positive)")

        # Check earnings consistency
        if len(net_incomes) >= 3:
            positive_earnings_years = sum(1 for ni in net_incomes[:min(5, len(net_incomes))] if ni > 0)
            if positive_earnings_years == min(5, len(net_incomes)):
                score += 2
                reasoning.append(f"Consistently profitable ({positive_earnings_years}/{min(5, len(net_incomes))} years)")
            elif positive_earnings_years >= min(5, len(net_incomes)) * 0.8:
                score += 1
                reasoning.append(f"Mostly profitable ({positive_earnings_years}/{min(5, len(net_incomes))} years)")
            else:
                reasoning.append(f"Erratic profitability ({positive_earnings_years}/{min(5, len(net_incomes))} years)")

        # Check cash flow consistency
        if len(free_cash_flows) >= 3:
            positive_fcf_years = sum(1 for fcf in free_cash_flows[:min(5, len(free_cash_flows))] if fcf > 0)
            if positive_fcf_years == min(5, len(free_cash_flows)):
                score += 1
                reasoning.append(f"Consistently positive free cash flow ({positive_fcf_years}/{min(5, len(free_cash_flows))} years)")
            elif positive_fcf_years >= min(5, len(free_cash_flows)) * 0.8:
                reasoning.append(f"Mostly positive free cash flow ({positive_fcf_years}/{min(5, len(free_cash_flows))} years)")

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
        analysis['type'] = 'consistency_analysis'
        analysis['title'] = f'Consistency Analysis'

        analysis_data['consistency_analysis'] = analysis
        ai_message = AIMessage(content=self.get_markdown(analysis))
        return {
            "context": context,
            "messages": [
                ai_message
            ]
        }