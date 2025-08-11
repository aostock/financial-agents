from common.agent_state import AgentState
from langchain.schema import AIMessage
from langchain_core.callbacks import dispatch_custom_event
from langchain_core.runnables import RunnableConfig
from typing import Dict, Any

import time
from common import markdown
from langgraph.types import StreamWriter


class ValuationAnalysis():
    def __init__(self, options: Dict[str, Any]):
        self.options = options

    
    def analyze(self, metrics: list) -> dict[str, any]:
        """Analyze valuation based on Phil Fisher's criteria."""
        result = {"score": 0, "max_score": 10, "details": []}
        if not metrics:
            result["details"].append('Insufficient data to perform valuation')
            return result

        latest_metrics = metrics[0]

        score = 0
        reasoning = []

        # Gather needed data
        net_income = latest_metrics.get('net_income')
        fcf = latest_metrics.get('free_cash_flow')
        market_cap = latest_metrics.get('market_cap')

        # 1) P/E
        if net_income and net_income > 0 and market_cap:
            pe = market_cap / net_income
            if pe < 20:
                score += 3
                reasoning.append(f"Reasonably attractive P/E: {pe:.2f}")
            elif pe < 30:
                score += 2
                reasoning.append(f"Somewhat high but possibly justifiable P/E: {pe:.2f}")
            else:
                reasoning.append(f"Very high P/E: {pe:.2f}")
        else:
            reasoning.append("No positive net income for P/E calculation")

        # 2) P/FCF
        if fcf and fcf > 0 and market_cap:
            pfcf = market_cap / fcf
            if pfcf < 20:
                score += 3
                reasoning.append(f"Reasonable P/FCF: {pfcf:.2f}")
            elif pfcf < 30:
                score += 2
                reasoning.append(f"Somewhat high P/FCF: {pfcf:.2f}")
            else:
                reasoning.append(f"Excessively high P/FCF: {pfcf:.2f}")
        else:
            reasoning.append("No positive free cash flow for P/FCF calculation")

        # 3) Price to Book Value
        total_assets = latest_metrics.get('total_assets')
        total_liabilities = latest_metrics.get('total_liabilities')
        shares = latest_metrics.get('ordinary_shares_number')
        
        if total_assets and total_liabilities and shares and shares > 0 and market_cap:
            book_value = total_assets - total_liabilities
            book_value_per_share = book_value / shares
            price_per_share = market_cap / shares if market_cap > 0 else 0
            
            if price_per_share > 0 and book_value_per_share > 0:
                price_to_book = price_per_share / book_value_per_share
                
                if price_to_book < 1.5:
                    score += 4
                    reasoning.append(f"Discount to book value (P/B: {price_to_book:.2f})")
                elif price_to_book < 3.0:
                    score += 2
                    reasoning.append(f"Fair valuation relative to book (P/B: {price_to_book:.2f})")
                else:
                    reasoning.append(f"Premium to book value (P/B: {price_to_book:.2f})")
            else:
                reasoning.append("Unable to calculate price-to-book ratio")
        else:
            reasoning.append("Insufficient data for book value analysis")

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
        analysis['type'] = 'valuation_analysis'
        analysis['title'] = f'Valuation Analysis'

        analysis_data['valuation_analysis'] = analysis
        ai_message = AIMessage(content=self.get_markdown(analysis))
        return {
            "context": context,
            "messages": [
                ai_message
            ]
        }