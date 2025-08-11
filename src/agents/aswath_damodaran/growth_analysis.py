from common.agent_state import AgentState
from langchain.schema import AIMessage
from langchain_core.callbacks import dispatch_custom_event
from langchain_core.runnables import RunnableConfig
from typing import Dict, Any

import time
from common import markdown
from langgraph.types import StreamWriter


class GrowthAnalysis():
    def __init__(self, options: Dict[str, Any]):
        self.options = options

    
    def analyze(self, metrics: list) -> dict[str, any]:
        """
        Growth score (0-4):
          +2  5-yr CAGR of revenue > 8 %
          +1  5-yr CAGR of revenue > 3 %
          +1  Positive FCFF growth over 5 yr
        Reinvestment efficiency (ROIC > WACC proxy) adds +1
        """
        result = {"score": 0, "max_score": 4, "details": []}
        if not metrics or len(metrics) < 2:
            result["details"].append('Insufficient history')
            return result

        # Revenue CAGR (oldest to latest)
        revs = [m.get('revenue') for m in reversed(metrics) if m.get('revenue')]
        if len(revs) >= 2 and revs[0] and revs[0] > 0:
            cagr = (revs[-1] / revs[0]) ** (1 / (len(revs) - 1)) - 1
        else:
            cagr = None

        score = 0
        details = []

        if cagr is not None:
            if cagr > 0.08:
                score += 2
                details.append(f"Revenue CAGR {cagr:.1%} (> 8%)")
            elif cagr > 0.03:
                score += 1
                details.append(f"Revenue CAGR {cagr:.1%} (> 3%)")
            else:
                details.append(f"Sluggish revenue CAGR {cagr:.1%}")
        else:
            details.append("Revenue data incomplete")

        # FCFF growth (proxy: free_cash_flow trend)
        fcfs = [m.get('free_cash_flow') for m in reversed(metrics) if m.get('free_cash_flow')]
        if len(fcfs) >= 2 and fcfs[-1] and fcfs[0] and fcfs[-1] > fcfs[0]:
            score += 1
            details.append("Positive FCFF growth")
        else:
            details.append("Flat or declining FCFF")

        # Reinvestment efficiency (ROIC vs. 10% hurdle)
        latest = metrics[0]
        if latest.get('return_on_invested_capital') and latest['return_on_invested_capital'] > 0.10:
            score += 1
            details.append(f"ROIC {latest['return_on_invested_capital']:.1%} (> 10%)")

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
        analysis['type'] = 'growth_analysis'
        analysis['title'] = f'Growth analysis'

        analysis_data['growth_analysis'] = analysis
        ai_message = AIMessage(content=self.get_markdown(analysis))
        return {
            "context": context,
            "messages": [
                ai_message
            ]
        }