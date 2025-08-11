from common.agent_state import AgentState
from langchain.schema import AIMessage
from langchain_core.callbacks import dispatch_custom_event
from langchain_core.runnables import RunnableConfig
from typing import Dict, Any

import time
from common import markdown
from langgraph.types import StreamWriter



class ManagementQualityAnalysis():
    def __init__(self, options: Dict[str, Any]):
        self.options = options

    
    def analyze(self, financial_line_items: list) -> dict[str, any]:
        """
        Checks for share dilution or consistent buybacks, and some dividend track record.
        A simplified approach:
        - if there's net share repurchase or stable share count, it suggests management
            might be shareholder-friendly.
        - if there's a big new issuance, it might be a negative sign (dilution).
        """
        result = {"score": 0, "max_score": 2, "details": []}
        if not financial_line_items:
            result["details"].append("Insufficient data for management analysis")
            return result

        reasoning = []
        mgmt_score = 0

        latest = financial_line_items[0]
        if latest.get('issuance_or_purchase_of_equity_shares') and latest.get('issuance_or_purchase_of_equity_shares') < 0:
            # Negative means the company spent money on buybacks
            mgmt_score += 1
            reasoning.append("Company has been repurchasing shares (shareholder-friendly)")

        if latest.get('issuance_or_purchase_of_equity_shares') and latest.get('issuance_or_purchase_of_equity_shares') > 0:
            # Positive issuance means new shares => possible dilution
            reasoning.append("Recent common stock issuance (potential dilution)")
        else:
            reasoning.append("No significant new stock issuance detected")

        # Check for any dividends
        if latest.get('dividends_and_other_cash_distributions') and latest.get('dividends_and_other_cash_distributions') < 0:
            mgmt_score += 1
            reasoning.append("Company has a track record of paying dividends")
        else:
            reasoning.append("No or minimal dividends paid")

        result["score"] = mgmt_score
        result["details"].append(reasoning)
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
        analysis['type'] = 'management_quality_analysis'
        analysis['title'] = 'Management quality analysis'

        analysis_data['management_quality_analysis'] = analysis
        ai_message = AIMessage(content=self.get_markdown(analysis))
        return {
            "context": context,
            "messages": [
                ai_message
            ]
        }