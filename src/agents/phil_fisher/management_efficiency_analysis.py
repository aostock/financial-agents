from common.agent_state import AgentState
from langchain.schema import AIMessage
from langchain_core.callbacks import dispatch_custom_event
from langchain_core.runnables import RunnableConfig
from typing import Dict, Any

import time
from common import markdown
from langgraph.types import StreamWriter


class ManagementEfficiencyAnalysis():
    def __init__(self, options: Dict[str, Any]):
        self.options = options

    
    def analyze(self, metrics: list) -> dict[str, any]:
        """Analyze management efficiency & leverage based on Phil Fisher's criteria."""
        result = {"score": 0, "max_score": 10, "details": []}
        if not metrics:
            result["details"].append('No financial data for management efficiency analysis')
            return result

        latest_metrics = metrics[0]

        score = 0
        reasoning = []

        # 1. Return on Equity (ROE)
        ni_values = [m.get('net_income') for m in metrics if m.get('net_income') is not None]
        eq_values = [m.get('stockholders_equity') for m in metrics if m.get('stockholders_equity') is not None]
        if ni_values and eq_values and len(ni_values) == len(eq_values) and ni_values[0] and eq_values[0]:
            recent_ni = ni_values[0]
            recent_eq = eq_values[0] if eq_values[0] else 1e-9
            if recent_ni > 0:
                roe = recent_ni / recent_eq
                if roe > 0.2:
                    score += 3
                    reasoning.append(f"High ROE: {roe:.1%}")
                elif roe > 0.1:
                    score += 2
                    reasoning.append(f"Moderate ROE: {roe:.1%}")
                elif roe > 0:
                    score += 1
                    reasoning.append(f"Positive but low ROE: {roe:.1%}")
                else:
                    reasoning.append(f"ROE is near zero or negative: {roe:.1%}")
            else:
                reasoning.append("Recent net income is zero or negative, hurting ROE")
        else:
            reasoning.append("Insufficient data for ROE calculation")

        # 2. Debt-to-Equity
        debt_values = [m.get('total_liabilities') for m in metrics if m.get('total_liabilities') is not None]
        if debt_values and eq_values and len(debt_values) == len(eq_values) and debt_values[0] and eq_values[0]:
            recent_debt = debt_values[0]
            recent_equity = eq_values[0] if eq_values[0] else 1e-9
            dte = recent_debt / recent_equity
            if dte < 0.3:
                score += 3
                reasoning.append(f"Low debt-to-equity: {dte:.2f}")
            elif dte < 1.0:
                score += 2
                reasoning.append(f"Manageable debt-to-equity: {dte:.2f}")
            else:
                reasoning.append(f"High debt-to-equity: {dte:.2f}")
        else:
            reasoning.append("Insufficient data for debt/equity analysis")

        # 3. FCF Consistency
        fcf_values = [m.get('free_cash_flow') for m in metrics if m.get('free_cash_flow') is not None]
        if fcf_values and len(fcf_values) >= 2:
            # Check if FCF is positive in recent years
            positive_fcf_count = sum(1 for x in fcf_values if x and x > 0)
            # We'll be simplistic: if most are positive, reward
            ratio = positive_fcf_count / len(fcf_values)
            if ratio > 0.8:
                score += 4
                reasoning.append(f"Majority of periods have positive FCF ({positive_fcf_count}/{len(fcf_values)})")
            else:
                reasoning.append(f"Free cash flow is inconsistent or often negative")
        else:
            reasoning.append("Insufficient or no FCF data to check consistency")

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
        analysis['type'] = 'management_efficiency_analysis'
        analysis['title'] = f'Management Efficiency Analysis'

        analysis_data['management_efficiency_analysis'] = analysis
        ai_message = AIMessage(content=self.get_markdown(analysis))
        return {
            "context": context,
            "messages": [
                ai_message
            ]
        }