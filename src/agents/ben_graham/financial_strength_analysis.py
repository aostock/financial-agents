from common.agent_state import AgentState
from langchain.schema import AIMessage
from langchain_core.callbacks import dispatch_custom_event
from langchain_core.runnables import RunnableConfig
from typing import Dict, Any

import time
from common import markdown
from langgraph.types import StreamWriter


class FinancialStrengthAnalysis():
    def __init__(self, options: Dict[str, Any]):
        self.options = options

    
    def analyze(self, metrics: list) -> dict[str, any]:
        """
        Graham checks liquidity (current ratio >= 2), manageable debt,
        and dividend record (preferably some history of dividends).
        """
        result = {"score": 0, "max_score": 5, "details": []}
        if not metrics:
            result["details"].append('No metrics available')
            return result

        latest_item = metrics[0]
        total_assets = latest_item.get('total_assets') or 0
        total_liabilities = latest_item.get('total_liabilities') or 0
        current_assets = latest_item.get('current_assets') or 0
        current_liabilities = latest_item.get('current_liabilities') or 0

        score = 0
        details = []

        # 1. Current ratio
        if current_liabilities and current_liabilities > 0:
            current_ratio = current_assets / current_liabilities
            if current_ratio >= 2.0:
                score += 2
                details.append(f"Current ratio = {current_ratio:.2f} (>=2.0: solid).")
            elif current_ratio >= 1.5:
                score += 1
                details.append(f"Current ratio = {current_ratio:.2f} (moderately strong).")
            else:
                details.append(f"Current ratio = {current_ratio:.2f} (<1.5: weaker liquidity).")
        else:
            details.append("Cannot compute current ratio (missing or zero current_liabilities).")

        # 2. Debt vs. Assets
        if total_assets and total_assets > 0:
            debt_ratio = total_liabilities / total_assets if total_liabilities else 0
            if debt_ratio < 0.5:
                score += 2
                details.append(f"Debt ratio = {debt_ratio:.2f}, under 0.50 (conservative).")
            elif debt_ratio < 0.8:
                score += 1
                details.append(f"Debt ratio = {debt_ratio:.2f}, somewhat high but could be acceptable.")
            else:
                details.append(f"Debt ratio = {debt_ratio:.2f}, quite high by Graham standards.")
        else:
            details.append("Cannot compute debt ratio (missing total_assets).")

        # 3. Dividend track record
        div_periods = [item.get('dividends_and_other_cash_distributions') for item in metrics if item.get('dividends_and_other_cash_distributions') is not None]
        if div_periods:
            # In many data feeds, dividend outflow is shown as a negative number
            # (money going out to shareholders). We'll consider any negative as 'paid a dividend'.
            div_paid_years = sum(1 for d in div_periods if d < 0)
            if div_paid_years > 0:
                # e.g. if at least half the periods had dividends
                if div_paid_years >= (len(div_periods) // 2 + 1):
                    score += 1
                    details.append("Company paid dividends in the majority of the reported years.")
                else:
                    details.append("Company has some dividend payments, but not most years.")
            else:
                details.append("Company did not pay dividends in these periods.")
        else:
            details.append("No dividend data available to assess payout consistency.")

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
        analysis['type'] = 'financial_strength_analysis'
        analysis['title'] = f'Financial strength analysis'

        analysis_data['financial_strength_analysis'] = analysis
        ai_message = AIMessage(content=self.get_markdown(analysis))
        return {
            "context": context,
            "messages": [
                ai_message
            ]
        }