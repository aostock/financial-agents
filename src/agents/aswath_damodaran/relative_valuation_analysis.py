from common.agent_state import AgentState
from langchain.schema import AIMessage
from langchain_core.callbacks import dispatch_custom_event
from langchain_core.runnables import RunnableConfig
from typing import Dict, Any

import time
from common import markdown
from langgraph.types import StreamWriter


class RelativeValuationAnalysis():
    def __init__(self, options: Dict[str, Any]):
        self.options = options

    
    def analyze(self, metrics: list) -> dict[str, any]:
        """
        Simple PE check vs. historical median (proxy since sector comps unavailable):
          +1 if TTM P/E < 70 % of 5-yr median
          +0 if between 70 %-130 %
          ‑1 if >130 %
        """
        result = {"score": 0, "max_score": 1, "details": []}
        if not metrics or len(metrics) < 5:
            result["details"].append("Insufficient P/E history")
            return result

        pes = [m.get('price_to_earnings_ratio') for m in metrics if m.get('price_to_earnings_ratio')]
        if len(pes) < 5:
            result["details"].append("P/E data sparse")
            return result

        ttm_pe = pes[0]
        median_pe = sorted(pes)[len(pes) // 2]

        if ttm_pe and median_pe and ttm_pe < 0.7 * median_pe:
            score, desc = 1, f"P/E {ttm_pe:.1f} vs. median {median_pe:.1f} (cheap)"
        elif ttm_pe and median_pe and ttm_pe > 1.3 * median_pe:
            score, desc = -1, f"P/E {ttm_pe:.1f} vs. median {median_pe:.1f} (expensive)"
        else:
            score, desc = 0, f"P/E inline with history"

        result["score"] = score
        result["details"] = [desc]
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
        analysis['type'] = 'relative_valuation_analysis'
        analysis['title'] = f'Relative valuation analysis'

        analysis_data['relative_valuation_analysis'] = analysis
        ai_message = AIMessage(content=self.get_markdown(analysis))
        return {
            "context": context,
            "messages": [
                ai_message
            ]
        }