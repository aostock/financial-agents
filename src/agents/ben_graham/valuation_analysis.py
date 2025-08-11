from common.agent_state import AgentState
from langchain.schema import AIMessage
from langchain_core.callbacks import dispatch_custom_event
from langchain_core.runnables import RunnableConfig
from typing import Dict, Any

import time
from common import markdown
import math
from langgraph.types import StreamWriter


class ValuationAnalysis():
    def __init__(self, options: Dict[str, Any]):
        self.options = options

    
    def analyze(self, metrics: list) -> dict[str, any]:
        """
        Core Graham approach to valuation:
        1. Net-Net Check: (Current Assets - Total Liabilities) vs. Market Cap
        2. Graham Number: sqrt(22.5 * EPS * Book Value per Share)
        3. Compare per-share price to Graham Number => margin of safety
        """
        result = {"score": 0, "max_score": 5, "details": []}
        if not metrics:
            result["details"].append("No metrics available")
            return result

        latest = metrics[0]
        current_assets = latest.get('current_assets') or 0
        total_liabilities = latest.get('total_liabilities') or 0
        book_value_ps = latest.get('book_value_per_share') or 0
        eps = latest.get('earnings_per_share') or 0
        shares_outstanding = latest.get('outstanding_shares') or 0
        market_cap = latest.get('market_cap') or 0

        details = []
        score = 0

        # 1. Net-Net Check
        #   NCAV = Current Assets - Total Liabilities
        #   If NCAV > Market Cap => historically a strong buy signal
        net_current_asset_value = current_assets - total_liabilities
        if net_current_asset_value > 0 and shares_outstanding and shares_outstanding > 0:
            net_current_asset_value_per_share = net_current_asset_value / shares_outstanding
            price_per_share = market_cap / shares_outstanding if market_cap and market_cap > 0 else 0

            details.append(f"Net Current Asset Value = {net_current_asset_value:,.2f}")
            details.append(f"NCAV Per Share = {net_current_asset_value_per_share:,.2f}")
            details.append(f"Price Per Share = {price_per_share:,.2f}")

            if market_cap and net_current_asset_value > market_cap:
                score += 4  # Very strong Graham signal
                details.append("Net-Net: NCAV > Market Cap (classic Graham deep value).")
            else:
                # For partial net-net discount
                if price_per_share > 0 and net_current_asset_value_per_share >= (price_per_share * 0.67):
                    score += 2
                    details.append("NCAV Per Share >= 2/3 of Price Per Share (moderate net-net discount).")
        else:
            details.append("NCAV not exceeding market cap or insufficient data for net-net approach.")

        # 2. Graham Number
        #   GrahamNumber = sqrt(22.5 * EPS * BVPS).
        #   Compare the result to the current price_per_share
        #   If GrahamNumber >> price, indicates undervaluation
        graham_number = None
        if eps > 0 and book_value_ps > 0:
            graham_number = math.sqrt(22.5 * eps * book_value_ps)
            details.append(f"Graham Number = {graham_number:.2f}")
        else:
            details.append("Unable to compute Graham Number (EPS or Book Value missing/<=0).")

        # 3. Margin of Safety relative to Graham Number
        if graham_number and shares_outstanding and shares_outstanding > 0:
            current_price = market_cap / shares_outstanding if market_cap and market_cap > 0 else 0
            if current_price > 0:
                margin_of_safety = (graham_number - current_price) / current_price
                details.append(f"Margin of Safety (Graham Number) = {margin_of_safety:.2%}")
                if margin_of_safety > 0.5:
                    score += 1  # Additional point for strong margin of safety
                    details.append("Price is well below Graham Number (>=50% margin).")
                elif margin_of_safety > 0.2:
                    details.append("Some margin of safety relative to Graham Number.")
                else:
                    details.append("Price close to or above Graham Number, low margin of safety.")
            else:
                details.append("Current price is zero or invalid; can't compute margin of safety.")
        # else: already appended details for missing graham_number

        result["score"] = score
        result["details"] = details
        result["graham_number"] = graham_number
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
        analysis['title'] = f'Valuation analysis'

        analysis_data['valuation_analysis'] = analysis
        ai_message = AIMessage(content=self.get_markdown(analysis))
        return {
            "context": context,
            "messages": [
                ai_message
            ]
        }