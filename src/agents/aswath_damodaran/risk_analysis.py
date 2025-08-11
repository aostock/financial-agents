from common.agent_state import AgentState
from langchain.schema import AIMessage
from langchain_core.callbacks import dispatch_custom_event
from langchain_core.runnables import RunnableConfig
from typing import Dict, Any

import time
from common import markdown
from langgraph.types import StreamWriter


class RiskAnalysis():
    def __init__(self, options: Dict[str, Any]):
        self.options = options

    
    def analyze(self, metrics: list) -> dict[str, any]:
        """
        Risk score (0-3):
          +1  Beta < 1.3
          +1  Debt/Equity < 1
          +1  Interest Coverage > 3×
        """
        result = {"score": 0, "max_score": 3, "details": []}
        if not metrics:
            result["details"].append('No metrics available')
            return result

        latest = metrics[0]
        score = 0
        details = []

        # Beta
        beta = latest.get('beta')
        if beta is not None:
            if beta < 1.3:
                score += 1
                details.append(f"Beta {beta:.2f}")
            else:
                details.append(f"High beta {beta:.2f}")
        else:
            details.append("Beta NA")

        # Debt / Equity
        dte = latest.get('debt_to_equity')
        if dte is not None:
            if dte < 1:
                score += 1
                details.append(f"D/E {dte:.1f}")
            else:
                details.append(f"High D/E {dte:.1f}")
        else:
            details.append("D/E NA")

        # Interest coverage
        ebit = latest.get('ebit')
        interest = latest.get('interest_expense')
        if ebit and interest and interest != 0:
            coverage = ebit / abs(interest)
            if coverage > 3:
                score += 1
                details.append(f"Interest coverage × {coverage:.1f}")
            else:
                details.append(f"Weak coverage × {coverage:.1f}")
        else:
            details.append("Interest coverage NA")

        # Compute cost of equity for later use
        cost_of_equity = self.estimate_cost_of_equity(beta)

        result["score"] = score
        result["details"] = details
        result["cost_of_equity"] = cost_of_equity
        result["beta"] = beta
        return result

    def estimate_cost_of_equity(self, beta: float | None) -> float:
        """CAPM: r_e = r_f + β × ERP (use Damodaran's long-term averages)."""
        risk_free = 0.04          # 10-yr US Treasury proxy
        erp = 0.05                # long-run US equity risk premium
        beta = beta if beta is not None else 1.0
        return risk_free + beta * erp

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
        analysis['type'] = 'risk_analysis'
        analysis['title'] = f'Risk analysis'

        analysis_data['risk_analysis'] = analysis
        ai_message = AIMessage(content=self.get_markdown(analysis))
        return {
            "context": context,
            "messages": [
                ai_message
            ]
        }