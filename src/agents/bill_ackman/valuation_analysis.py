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
        """
        Ackman invests in companies trading at a discount to intrinsic value.
        Uses a simplified DCF with FCF as a proxy, plus margin of safety analysis.
        """
        result = {"score": 0, "max_score": 5, "details": []}
        if not metrics:
            result["details"].append("No metrics available")
            return result

        # Since metrics are in descending order (newest first),
        # the most recent period is the first element
        latest = metrics[0] if metrics else None
        fcf = latest.get('free_cash_flow') if latest and latest.get('free_cash_flow') else 0
        market_cap = latest.get('market_cap') if latest and latest.get('market_cap') else 0

        if fcf <= 0:
            result["details"].append(f"No positive FCF for valuation; FCF = {fcf}")
            result["intrinsic_value"] = None
            return result

        # Basic DCF assumptions
        growth_rate = 0.06
        discount_rate = 0.10
        terminal_multiple = 15
        projection_years = 5

        present_value = 0
        for year in range(1, projection_years + 1):
            future_fcf = fcf * (1 + growth_rate) ** year
            pv = future_fcf / ((1 + discount_rate) ** year)
            present_value += pv

        # Terminal Value
        terminal_value = (
            fcf * (1 + growth_rate) ** projection_years * terminal_multiple
        ) / ((1 + discount_rate) ** projection_years)

        intrinsic_value = present_value + terminal_value
        margin_of_safety = (intrinsic_value - market_cap) / market_cap if market_cap > 0 else 0

        score = 0
        # Simple scoring
        if margin_of_safety > 0.3:
            score += 3
        elif margin_of_safety > 0.1:
            score += 1

        details = [
            f"Calculated intrinsic value: ~{intrinsic_value:,.2f}",
            f"Market cap: ~{market_cap:,.2f}",
            f"Margin of safety: {margin_of_safety:.2%}"
        ]

        result["score"] = score
        result["details"] = details
        result["intrinsic_value"] = intrinsic_value
        result["margin_of_safety"] = margin_of_safety
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
        analysis['title'] = f'Valuation analysis'

        analysis_data['valuation_analysis'] = analysis
        ai_message = AIMessage(content=self.get_markdown(analysis))
        return {
            "context": context,
            "messages": [
                ai_message
            ]
        }