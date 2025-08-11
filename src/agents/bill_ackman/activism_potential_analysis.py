from common.agent_state import AgentState
from langchain.schema import AIMessage
from langchain_core.callbacks import dispatch_custom_event
from langchain_core.runnables import RunnableConfig
from typing import Dict, Any

import time
from common import markdown
from langgraph.types import StreamWriter


class ActivismPotentialAnalysis():
    def __init__(self, options: Dict[str, Any]):
        self.options = options

    
    def analyze(self, metrics: list) -> dict[str, any]:
        """
        Bill Ackman often engages in activism if a company has a decent brand or moat
        but is underperforming operationally.

        We'll do a simplified approach:
        - Look for positive revenue trends but subpar margins
        - That may indicate 'activism upside' if operational improvements could unlock value.
        """
        result = {"score": 0, "max_score": 5, "details": []}
        if not metrics:
            result["details"].append('No metrics available')
            return result

        # Check revenue growth vs. operating margin
        revenues = [item.get('revenue') for item in metrics if item.get('revenue') is not None]
        op_margins = [item.get('operating_margin') for item in metrics if item.get('operating_margin') is not None]

        if len(revenues) < 2 or not op_margins:
            result["details"].append("Not enough data to assess activism potential (need multi-year revenue + margins).")
            return result

        initial, final = revenues[-1], revenues[0]
        revenue_growth = (final - initial) / abs(initial) if initial else 0
        avg_margin = sum(op_margins) / len(op_margins)

        score = 0
        details = []

        # Suppose if there's decent revenue growth but margins are below 10%, Ackman might see activism potential.
        if revenue_growth > 0.15 and avg_margin < 0.10:
            score += 2
            details.append(
                f"Revenue growth is healthy (~{revenue_growth*100:.1f}%), but margins are low (avg {avg_margin*100:.1f}%). "
                "Activism could unlock margin improvements."
            )
        else:
            details.append("No clear sign of activism opportunity (either margins are already decent or growth is weak).")

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
        analysis['type'] = 'activism_potential_analysis'
        analysis['title'] = f'Activism potential analysis'

        analysis_data['activism_potential_analysis'] = analysis
        ai_message = AIMessage(content=self.get_markdown(analysis))
        return {
            "context": context,
            "messages": [
                ai_message
            ]
        }