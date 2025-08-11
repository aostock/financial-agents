from common.agent_state import AgentState
from langchain.schema import AIMessage
from langchain_core.callbacks import dispatch_custom_event
from langchain_core.runnables import RunnableConfig
from typing import Dict, Any

import time
from common import markdown
from langgraph.types import StreamWriter
import re


class InsiderSentimentAnalysis():
    def __init__(self, options: Dict[str, Any]):
        self.options = options

    
    def analyze(self, insider_transactions: list) -> dict[str, any]:
        """Analyze insider trading sentiment for management confidence."""
        result = {"score": 0, "max_score": 10, "details": [], "transaction_summary": {}}
        if not insider_transactions:
            result["details"].append('No insider transaction data available')
            return result

        # Get recent insider transactions (last 90 days)
        recent_transactions = []
        cutoff_date = time.time() - 90*24*60*60  # 90 days ago
        
        for transaction in insider_transactions:
            if transaction.get('date'):
                try:
                    # Parse date string to timestamp
                    trans_date = time.mktime(time.strptime(transaction['date'], "%Y-%m-%d"))
                    if trans_date >= cutoff_date:
                        recent_transactions.append(transaction)
                except:
                    continue

        if not recent_transactions:
            result["details"].append('No recent insider transaction data available')
            return result

        # Analyze insider trading patterns
        total_buy_value = 0
        total_sell_value = 0
        buy_transactions = 0
        sell_transactions = 0
        
        for transaction in recent_transactions:
            transaction_type = transaction.get('transaction_type', '').lower()
            value = transaction.get('value', 0) or 0
            
            if 'buy' in transaction_type or 'purchase' in transaction_type:
                total_buy_value += value
                buy_transactions += 1
            elif 'sell' in transaction_type or 'sale' in transaction_type:
                total_sell_value += value
                sell_transactions += 1

        result["transaction_summary"] = {
            "total_buy_value": total_buy_value,
            "total_sell_value": total_sell_value,
            "buy_transactions": buy_transactions,
            "sell_transactions": sell_transactions
        }

        # Calculate insider sentiment score based on buying vs selling activity
        total_activity = total_buy_value + total_sell_value
        
        if total_activity > 0:
            buy_ratio = total_buy_value / total_activity
            sell_ratio = total_sell_value / total_activity
            
            # Score calculation: 0-10 scale where 5 is neutral
            # Higher buying activity = higher score (bullish sentiment)
            # Higher selling activity = lower score (bearish sentiment)
            score = 5 + (buy_ratio - sell_ratio) * 5
            score = max(0, min(10, score))  # Clamp between 0-10
            
            result["score"] = score
            
            if score >= 7:
                result["details"].append(f"Positive insider sentiment - significant buying activity (${total_buy_value:,.0f} in buys vs ${total_sell_value:,.0f} in sells)")
            elif score >= 3:
                result["details"].append(f"Neutral insider sentiment - balanced buying and selling (${total_buy_value:,.0f} in buys vs ${total_sell_value:,.0f} in sells)")
            else:
                result["details"].append(f"Negative insider sentiment - significant selling activity (${total_sell_value:,.0f} in sells vs ${total_buy_value:,.0f} in buys)")
        else:
            result["details"].append('No significant insider trading activity detected')

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
        analysis['type'] = 'insider_sentiment_analysis'
        analysis['title'] = f'Insider Sentiment Analysis'

        analysis_data['insider_sentiment_analysis'] = analysis
        ai_message = AIMessage(content=self.get_markdown(analysis))
        return {
            "context": context,
            "messages": [
                ai_message
            ]
        }