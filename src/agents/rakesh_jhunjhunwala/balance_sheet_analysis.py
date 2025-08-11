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
        Check financial strength - healthy asset/liability structure, liquidity.
        Jhunjhunwala favored companies with clean balance sheets and manageable debt.
        """
        result = {"score": 0, "max_score": 4, "details": []}
        if not metrics:
            result["details"].append('No balance sheet data')
            return result

        latest_metrics = metrics[0]
        score = 0
        reasoning = []

        # Debt to asset ratio
        if (latest_metrics.get('total_assets') and latest_metrics.get('total_liabilities') 
            and latest_metrics['total_assets'] and latest_metrics['total_liabilities'] 
            and latest_metrics['total_assets'] > 0):
            debt_ratio = latest_metrics['total_liabilities'] / latest_metrics['total_assets']
            if debt_ratio < 0.5:
                score += 2
                reasoning.append(f"Low debt ratio: {debt_ratio:.2f}")
            elif debt_ratio < 0.7:
                score += 1
                reasoning.append(f"Moderate debt ratio: {debt_ratio:.2f}")
            else:
                reasoning.append(f"High debt ratio: {debt_ratio:.2f}")
        else:
            reasoning.append("Insufficient data to calculate debt ratio")

        # Current ratio (liquidity)
        if (latest_metrics.get('current_assets') and latest_metrics.get('current_liabilities') 
            and latest_metrics['current_assets'] and latest_metrics['current_liabilities'] 
            and latest_metrics['current_liabilities'] > 0):
            current_ratio = latest_metrics['current_assets'] / latest_metrics['current_liabilities']
            if current_ratio > 2.0:
                score += 2
                reasoning.append(f"Excellent liquidity with current ratio: {current_ratio:.2f}")
            elif current_ratio > 1.5:
                score += 1
                reasoning.append(f"Good liquidity with current ratio: {current_ratio:.2f}")
            else:
                reasoning.append(f"Weak liquidity with current ratio: {current_ratio:.2f}")
        else:
            reasoning.append("Insufficient data to calculate current ratio")

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