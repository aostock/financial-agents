from common.agent_state import AgentState
from langchain.schema import AIMessage
from langchain_core.callbacks import dispatch_custom_event
from langchain_core.runnables import RunnableConfig
from typing import Dict, Any

import time
from common import markdown
from langgraph.types import StreamWriter


class EarningsStabilityAnalysis():
    def __init__(self, options: Dict[str, Any]):
        self.options = options

    
    def analyze(self, metrics: list) -> dict[str, any]:
        """
        Graham wants at least several years of consistently positive earnings (ideally 5+).
        We'll check:
        1. Number of years with positive EPS.
        2. Growth in EPS from first to last period.
        """
        result = {"score": 0, "max_score": 5, "details": []}
        if not metrics:
            result["details"].append('No metrics available')
            return result

        eps_vals = []
        for item in metrics:
            if item.get('earnings_per_share') is not None:
                eps_vals.append(item.get('earnings_per_share'))

        if len(eps_vals) < 2:
            result["details"].append("Not enough multi-year EPS data.")
            return result

        score = 0
        details = []

        # 1. Consistently positive EPS
        positive_eps_years = sum(1 for e in eps_vals if e > 0)
        total_eps_years = len(eps_vals)
        if positive_eps_years == total_eps_years:
            score += 3
            details.append("EPS was positive in all available periods.")
        elif positive_eps_years >= (total_eps_years * 0.8):
            score += 2
            details.append("EPS was positive in most periods.")
        else:
            details.append("EPS was negative in multiple periods.")

        # 2. EPS growth from earliest to latest
        if len(eps_vals) > 1 and eps_vals[-1] > eps_vals[0]:  # Latest is first in reversed list
            score += 2
            details.append("EPS grew from earliest to latest period.")
        elif len(eps_vals) > 1:
            details.append("EPS did not grow from earliest to latest period.")

        result["score"] = score
        result["details"] = details
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
        analysis['type'] = 'earnings_stability_analysis'
        analysis['title'] = f'Earnings stability analysis'

        analysis_data['earnings_stability_analysis'] = analysis
        ai_message = AIMessage(content=self.get_markdown(analysis))
        return {
            "context": context,
            "messages": [
                ai_message
            ]
        }