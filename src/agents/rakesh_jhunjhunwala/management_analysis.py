from common.agent_state import AgentState
from langchain.schema import AIMessage
from langchain_core.callbacks import dispatch_custom_event
from langchain_core.runnables import RunnableConfig
from typing import Dict, Any

import time
from common import markdown
from langgraph.types import StreamWriter


class ManagementAnalysis():
    def __init__(self, options: Dict[str, Any]):
        self.options = options

    
    def analyze(self, metrics: list) -> dict[str, any]:
        """Analyze management quality based on Rakesh Jhunjhunwala's criteria."""
        result = {"score": 0, "max_score": 10, "details": []}
        if not metrics:
            result["details"].append('No metrics available')
            return result

        latest_metrics = metrics[0]

        score = 0
        reasoning = []

        # Check shareholder friendliness through dividend payments
        if latest_metrics.get('dividends_and_other_cash_distributions') and latest_metrics.get('net_income'):
            dividends = latest_metrics['dividends_and_other_cash_distributions']
            net_income = latest_metrics['net_income']
            if net_income > 0:  # Avoid division by zero or negative income
                dividend_payout = abs(dividends) / net_income
                if dividend_payout > 0.3:  # High dividend payout
                    score += 2
                    reasoning.append(f"Shareholder friendly with {dividend_payout:.0%} dividend payout ratio")
                elif dividend_payout > 0.1:  # Moderate dividend payout
                    score += 1
                    reasoning.append(f"Moderate dividend payout of {dividend_payout:.0%}")
                else:
                    reasoning.append(f"Low dividend payout of {dividend_payout:.0%}")
            else:
                reasoning.append("Cannot calculate dividend payout (negative net income)")
        else:
            reasoning.append("Dividend data not available")

        # Analyze share buybacks vs. issuance to determine capital allocation quality
        if latest_metrics.get('issuance_or_purchase_of_equity_shares'):
            equity_activity = latest_metrics['issuance_or_purchase_of_equity_shares']
            if equity_activity < 0:  # Negative means buybacks
                buyback_amount = abs(equity_activity)
                if latest_metrics.get('market_cap'):
                    buyback_ratio = buyback_amount / latest_metrics['market_cap']
                    if buyback_ratio > 0.02:  # >2% of market cap
                        score += 3
                        reasoning.append(f"Significant share buybacks ({buyback_ratio:.1%} of market cap)")
                    elif buyback_ratio > 0.01:  # >1% of market cap
                        score += 2
                        reasoning.append(f"Meaningful share buybacks ({buyback_ratio:.1%} of market cap)")
                    else:
                        score += 1
                        reasoning.append(f"Minor share buybacks ({buyback_ratio:.1%} of market cap)")
                else:
                    score += 1
                    reasoning.append("Share buybacks activity detected")
            elif equity_activity > 0:  # Positive means issuance
                issuance_amount = equity_activity
                if latest_metrics.get('market_cap'):
                    issuance_ratio = issuance_amount / latest_metrics['market_cap']
                    reasoning.append(f"Equity issuance ({issuance_ratio:.1%} of market cap) - potential dilution")
                else:
                    reasoning.append("Equity issuance activity detected - potential dilution")
            else:
                reasoning.append("No significant equity activity")
        else:
            reasoning.append("Equity activity data not available")

        # Check capital expenditure vs. depreciation to assess growth investment
        if latest_metrics.get('capital_expenditure') and latest_metrics.get('depreciation_and_amortization'):
            capex = abs(latest_metrics['capital_expenditure'])
            depreciation = abs(latest_metrics['depreciation_and_amortization'])
            if depreciation > 0:
                capex_to_depreciation = capex / depreciation
                if capex_to_depreciation > 1.5:  # High reinvestment
                    score += 2
                    reasoning.append(f"High reinvestment with capex/depreciation ratio of {capex_to_depreciation:.2f}")
                elif capex_to_depreciation > 1.0:  # Moderate reinvestment
                    score += 1
                    reasoning.append(f"Moderate reinvestment with capex/depreciation ratio of {capex_to_depreciation:.2f}")
                else:
                    reasoning.append(f"Low reinvestment with capex/depreciation ratio of {capex_to_depreciation:.2f}")
            else:
                reasoning.append("Cannot calculate capex/depreciation ratio")
        else:
            reasoning.append("Capex or depreciation data not available")

        # Check earnings quality through net income vs. operating cash flow (if available in dataset)
        # This would require cash flow data which may not be available in the current dataset structure

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
        analysis['type'] = 'management_analysis'
        analysis['title'] = f'Management analysis'

        analysis_data['management_analysis'] = analysis
        ai_message = AIMessage(content=self.get_markdown(analysis))
        return {
            "context": context,
            "messages": [
                ai_message
            ]
        }