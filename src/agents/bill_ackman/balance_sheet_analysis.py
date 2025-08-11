from common.agent_state import AgentState
from langchain.schema import AIMessage
from langchain_core.callbacks import dispatch_custom_event
from langchain_core.runnables import RunnableConfig
from typing import Dict, Any

import time
from common import markdown
from langgraph.types import StreamWriter


class BalanceSheetAnalysis():
    def __init__(self, options: Dict[str, Any]):
        self.options = options

    
    def analyze(self, metrics: list) -> dict[str, any]:
        """
        Evaluate the company's balance sheet over multiple periods:
        - Debt ratio trends
        - Capital returns to shareholders over time (dividends, buybacks)
        """
        result = {"score": 0, "max_score": 5, "details": []}
        if not metrics:
            result["details"].append('No metrics available')
            return result

        score = 0
        details = []

        # 1. Multi-period debt ratio or debt_to_equity
        debt_to_equity_vals = [item.get('debt_to_equity') for item in metrics if item.get('debt_to_equity') is not None]
        if debt_to_equity_vals:
            below_one_count = sum(1 for d in debt_to_equity_vals if d < 1.0)
            if below_one_count >= (len(debt_to_equity_vals) // 2 + 1):
                score += 2
                details.append("Debt-to-equity < 1.0 for the majority of periods (reasonable leverage).")
            else:
                details.append("Debt-to-equity >= 1.0 in many periods (could be high leverage).")
        else:
            # Fallback to total_liabilities / total_assets
            liab_to_assets = []
            for item in metrics:
                if item.get('total_liabilities') and item.get('total_assets') and item['total_assets'] > 0:
                    liab_to_assets.append(item['total_liabilities'] / item['total_assets'])

            if liab_to_assets:
                below_50pct_count = sum(1 for ratio in liab_to_assets if ratio < 0.5)
                if below_50pct_count >= (len(liab_to_assets) // 2 + 1):
                    score += 2
                    details.append("Liabilities-to-assets < 50% for majority of periods.")
                else:
                    details.append("Liabilities-to-assets >= 50% in many periods.")
            else:
                details.append("No consistent leverage ratio data available.")

        # 2. Capital allocation approach (dividends + share counts)
        dividends_list = [
            item.get('dividends_and_other_cash_distributions')
            for item in metrics
            if item.get('dividends_and_other_cash_distributions') is not None
        ]
        if dividends_list:
            paying_dividends_count = sum(1 for d in dividends_list if d < 0)
            if paying_dividends_count >= (len(dividends_list) // 2 + 1):
                score += 1
                details.append("Company has a history of returning capital to shareholders (dividends).")
            else:
                details.append("Dividends not consistently paid or no data on distributions.")
        else:
            details.append("No dividend data found across periods.")

        # Check for decreasing share count (simple approach)
        shares = [item.get('outstanding_shares') for item in metrics if item.get('outstanding_shares') is not None]
        if len(shares) >= 2:
            # For buybacks, the newest count should be less than the oldest count
            if shares[0] and shares[-1] and shares[0] < shares[-1]:
                score += 1
                details.append("Outstanding shares have decreased over time (possible buybacks).")
            else:
                details.append("Outstanding shares have not decreased over the available periods.")
        else:
            details.append("No multi-period share count data to assess buybacks.")

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
        analysis['type'] = 'balance_sheet_analysis'
        analysis['title'] = f'Balance sheet analysis'

        analysis_data['balance_sheet_analysis'] = analysis
        ai_message = AIMessage(content=self.get_markdown(analysis))
        return {
            "context": context,
            "messages": [
                ai_message
            ]
        }