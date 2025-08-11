from common.agent_state import AgentState
from langchain.schema import AIMessage
from langchain_core.callbacks import dispatch_custom_event
from langchain_core.runnables import RunnableConfig
from typing import Dict, Any

import time
from common import markdown
from langgraph.types import StreamWriter


class CashFlowAnalysis():
    def __init__(self, options: Dict[str, Any]):
        self.options = options

    
    def analyze(self, metrics: list) -> dict[str, any]:
        """
        Evaluate free cash flow and dividend behavior.
        Jhunjhunwala appreciated companies generating strong free cash flow and rewarding shareholders.
        """
        result = {"score": 0, "max_score": 3, "details": []}
        if not metrics:
            result["details"].append('No cash flow data')
            return result

        latest_metrics = metrics[0]
        score = 0
        reasoning = []

        # Free cash flow analysis
        if latest_metrics.get('free_cash_flow') and latest_metrics['free_cash_flow']:
            if latest_metrics['free_cash_flow'] > 0:
                score += 2
                reasoning.append(f"Positive free cash flow: {latest_metrics['free_cash_flow']}")
            else:
                reasoning.append(f"Negative free cash flow: {latest_metrics['free_cash_flow']}")
        else:
            reasoning.append("Free cash flow data not available")

        # Dividend analysis
        if latest_metrics.get('dividends_and_other_cash_distributions') and latest_metrics['dividends_and_other_cash_distributions']:
            if latest_metrics['dividends_and_other_cash_distributions'] < 0:  # Negative indicates cash outflow for dividends
                score += 1
                reasoning.append("Company pays dividends to shareholders")
            else:
                reasoning.append("No significant dividend payments")
        else:
            reasoning.append("No dividend payment data available")

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
        analysis['type'] = 'cash_flow_analysis'
        analysis['title'] = f'Cash flow analysis'

        analysis_data['cash_flow_analysis'] = analysis
        ai_message = AIMessage(content=self.get_markdown(analysis))
        return {
            "context": context,
            "messages": [
                ai_message
            ]
        }