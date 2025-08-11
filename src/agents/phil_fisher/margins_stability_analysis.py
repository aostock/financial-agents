from common.agent_state import AgentState
from langchain.schema import AIMessage
from langchain_core.callbacks import dispatch_custom_event
from langchain_core.runnables import RunnableConfig
from typing import Dict, Any

import time
from common import markdown
import statistics
from langgraph.types import StreamWriter


class MarginsStabilityAnalysis():
    def __init__(self, options: Dict[str, Any]):
        self.options = options

    
    def analyze(self, metrics: list) -> dict[str, any]:
        """Analyze margins stability based on Phil Fisher's criteria."""
        result = {"score": 0, "max_score": 10, "details": []}
        if not metrics or len(metrics) < 2:
            result["details"].append('Insufficient historical data (need at least 2 years)')
            return result

        latest_metrics = metrics[0]
        previous_metrics = metrics[1]

        score = 0
        reasoning = []

        # 1. Operating Margin Consistency
        op_margins = [m.get('operating_margin') for m in metrics if m.get('operating_margin') is not None]
        if len(op_margins) >= 2:
            # Check if margins are stable or improving (comparing oldest to newest)
            oldest_op_margin = op_margins[-1]
            newest_op_margin = op_margins[0]
            if newest_op_margin and oldest_op_margin and newest_op_margin >= oldest_op_margin > 0:
                score += 3
                reasoning.append(f"Operating margin stable or improving ({oldest_op_margin:.1%} -> {newest_op_margin:.1%})")
            elif newest_op_margin and newest_op_margin > 0:
                score += 2
                reasoning.append(f"Operating margin positive but slightly declined")
            else:
                reasoning.append(f"Operating margin may be negative or uncertain")
        else:
            reasoning.append("Not enough operating margin data points")

        # 2. Gross Margin Level
        gm_values = [m.get('gross_margin') for m in metrics if m.get('gross_margin') is not None]
        if gm_values and gm_values[0]:
            # We'll just take the most recent
            recent_gm = gm_values[0]
            if recent_gm > 0.5:
                score += 3
                reasoning.append(f"Strong gross margin: {recent_gm:.1%}")
            elif recent_gm > 0.3:
                score += 2
                reasoning.append(f"Moderate gross margin: {recent_gm:.1%}")
            else:
                reasoning.append(f"Low gross margin: {recent_gm:.1%}")
        else:
            reasoning.append("No gross margin data available")

        # 3. Multi-year Margin Stability
        #   e.g. if we have at least 3 data points, see if standard deviation is low.
        if len(op_margins) >= 3:
            stdev = statistics.pstdev(op_margins)
            if stdev < 0.02:
                score += 4
                reasoning.append("Operating margin extremely stable over multiple years")
            elif stdev < 0.05:
                score += 3
                reasoning.append("Operating margin reasonably stable")
            else:
                reasoning.append("Operating margin volatility is high")
        else:
            reasoning.append("Not enough margin data points for volatility check")

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
        analysis['type'] = 'margins_stability_analysis'
        analysis['title'] = f'Margins & Stability Analysis'

        analysis_data['margins_stability_analysis'] = analysis
        ai_message = AIMessage(content=self.get_markdown(analysis))
        return {
            "context": context,
            "messages": [
                ai_message
            ]
        }