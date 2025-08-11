from common.agent_state import AgentState
from langchain.schema import AIMessage
from langchain_core.callbacks import dispatch_custom_event
from langchain_core.runnables import RunnableConfig
from typing import Dict, Any

import time
from common import markdown
from langgraph.types import StreamWriter


class InsiderActivityAnalysis():
    def __init__(self, options: Dict[str, Any]):
        self.options = options

    
    def analyze(self, insider_transactions: list) -> dict[str, any]:
        """Analyze insider activity based on Phil Fisher's criteria."""
        # Default is neutral (5/10).
        result = {"score": 5, "max_score": 10, "details": []}
        score = 5
        reasoning = []

        if not insider_transactions:
            reasoning.append("No insider trades data; defaulting to neutral")
            result["details"] = reasoning
            return result

        buys, sells = 0, 0
        for transaction in insider_transactions:
            transaction_shares = transaction.get('transaction_shares')
            if transaction_shares is not None:
                if transaction_shares > 0:
                    buys += 1
                elif transaction_shares < 0:
                    sells += 1

        total = buys + sells
        if total == 0:
            reasoning.append("No buy/sell transactions found; neutral")
            result["details"] = reasoning
            return result

        buy_ratio = buys / total
        if buy_ratio > 0.7:
            score = 9
            reasoning.append(f"Heavy insider buying: {buys} buys vs. {sells} sells")
        elif buy_ratio > 0.6:
            score = 8
            reasoning.append(f"Significant insider buying: {buys} buys vs. {sells} sells")
        elif buy_ratio > 0.5:
            score = 7
            reasoning.append(f"Moderate insider buying: {buys} buys vs. {sells} sells")
        elif buy_ratio > 0.4:
            score = 6
            reasoning.append(f"Slight insider buying: {buys} buys vs. {sells} sells")
        elif buy_ratio > 0.3:
            score = 5
            reasoning.append(f"Balanced insider activity: {buys} buys vs. {sells} sells")
        else:
            score = 4
            reasoning.append(f"Mostly insider selling: {buys} buys vs. {sells} sells")

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
        insider_transactions = context.get('insider_transactions')
        analysis = self.analyze(insider_transactions)
        analysis['type'] = 'insider_activity_analysis'
        analysis['title'] = f'Insider Activity Analysis'

        analysis_data['insider_activity_analysis'] = analysis
        ai_message = AIMessage(content=self.get_markdown(analysis))
        return {
            "context": context,
            "messages": [
                ai_message
            ]
        }