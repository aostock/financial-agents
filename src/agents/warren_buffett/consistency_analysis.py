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

    def analyze(self, financial_line_items: list) -> dict[str, any]:
        """Analyze earnings consistency and growth."""
        result = {"score": 0, "max_score": 3, "details": []}
        if len(financial_line_items) < 4:  # Need at least 4 periods for trend analysis
            result["details"].append("Insufficient historical data")
            return result

        score = 0
        reasoning = []

        # Check earnings growth trend
        earnings_values = [item.get('net_income') for item in financial_line_items if item.get('net_income')]
        if len(earnings_values) >= 4:
            # Simple check: is each period's earnings bigger than the next?
            earnings_growth = all(earnings_values[i] > earnings_values[i + 1] for i in range(len(earnings_values) - 1))

            if earnings_growth:
                score += 3
                reasoning.append("Consistent earnings growth over past periods")
            else:
                reasoning.append("Inconsistent earnings growth pattern")

            # Calculate total growth rate from oldest to latest
            if len(earnings_values) >= 2 and earnings_values[-1] != 0:
                growth_rate = (earnings_values[0] - earnings_values[-1]) / abs(earnings_values[-1])
                reasoning.append(f"Total earnings growth of {growth_rate:.1%} over past {len(earnings_values)} periods")
        else:
            reasoning.append("Insufficient earnings data for trend analysis")

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
        analysis['title'] = 'Consistency analysis'

        analysis_data['consistency_analysis'] = analysis
        ai_message = AIMessage(content=self.get_markdown(analysis))
        return {
            "context": context,
            "messages": [
                ai_message
            ]
        }