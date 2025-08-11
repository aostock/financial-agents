from common.agent_state import AgentState
from langchain.schema import AIMessage
from langchain_core.callbacks import dispatch_custom_event
from langchain_core.runnables import RunnableConfig
from typing import Dict, Any

import time
from common import markdown
from langgraph.types import StreamWriter


class MarketInefficiencyAnalysis():
    def __init__(self, options: Dict[str, Any]):
        self.options = options

    
    def analyze(self, metrics: list) -> dict[str, any]:
        """Identify market inefficiencies and mispricings."""
        result = {"score": 0, "max_score": 10, "details": []}
        if not metrics or len(metrics) < 3:
            result["details"].append('Insufficient historical data (need at least 3 years)')
            return result

        latest_metrics = metrics[0]
        previous_metrics = metrics[1] if len(metrics) > 1 else None

        score = 0
        reasoning = []

        # Check for valuation disconnects
        # P/E ratio vs. earnings quality
        if latest_metrics.get('price_to_earnings_ratio') and latest_metrics.get('return_on_equity'):
            pe_ratio = latest_metrics['price_to_earnings_ratio']
            roe = latest_metrics['return_on_equity']
            
            # High P/E with low ROE suggests overvaluation
            if pe_ratio > 25 and roe < 0.10:
                reasoning.append(f"Potential overvaluation: High P/E ({pe_ratio:.1f}x) with low ROE ({roe:.1%})")
            # Low P/E with high ROE suggests undervaluation
            elif pe_ratio < 15 and roe > 0.15:
                score += 3
                reasoning.append(f"Potential undervaluation: Low P/E ({pe_ratio:.1f}x) with high ROE ({roe:.1%})")
            # Reasonable alignment
            elif pe_ratio > 15 and pe_ratio < 25 and roe > 0.10 and roe < 0.20:
                score += 1
                reasoning.append(f"Reasonable P/E to ROE alignment (P/E: {pe_ratio:.1f}x, ROE: {roe:.1%})")

        # P/B ratio vs. ROE
        if latest_metrics.get('price_to_book_ratio') and latest_metrics.get('return_on_equity'):
            pb_ratio = latest_metrics['price_to_book_ratio']
            roe = latest_metrics['return_on_equity']
            
            # Low P/B with high ROE suggests undervaluation
            if pb_ratio < 1.5 and roe > 0.15:
                score += 2
                reasoning.append(f"Attractive P/B vs. ROE: Low P/B ({pb_ratio:.1f}x) with high ROE ({roe:.1%})")
            # High P/B with low ROE suggests overvaluation
            elif pb_ratio > 3.0 and roe < 0.10:
                reasoning.append(f"Concerning P/B vs. ROE: High P/B ({pb_ratio:.1f}x) with low ROE ({roe:.1%})")

        # Check for market sentiment disconnects
        # Compare current metrics to historical averages
        if len(metrics) >= 3:
            historical_roes = [m.get('return_on_equity') for m in metrics[:3] if m.get('return_on_equity') is not None]
            if len(historical_roes) >= 2:
                avg_roe = sum(historical_roes) / len(historical_roes)
                current_roe = latest_metrics.get('return_on_equity', 0)
                
                if current_roe > avg_roe * 1.2:  # 20% above average
                    score += 1
                    reasoning.append(f"Improving ROE: Current {current_roe:.1%} vs. historical average {avg_roe:.1%}")
                elif current_roe < avg_roe * 0.8:  # 20% below average
                    reasoning.append(f"Deteriorating ROE: Current {current_roe:.1%} vs. historical average {avg_roe:.1%}")

        # Look for sector/industry mispricing opportunities
        # This would require sector data, but we can make some basic comparisons
        if latest_metrics.get('return_on_invested_capital') and latest_metrics.get('return_on_equity'):
            roic = latest_metrics['return_on_invested_capital']
            roe = latest_metrics['return_on_equity']
            
            # High ROE with low ROIC might indicate excessive leverage
            if roe > 0.20 and (roic < 0.10 or roic < roe * 0.7):
                reasoning.append(f"Potential leverage-driven ROE: ROE {roe:.1%} vs. ROIC {roic:.1%}")
            # High ROIC with reasonable ROE suggests good capital efficiency
            elif roic > 0.15 and roe > 0.12:
                score += 2
                reasoning.append(f"Strong capital efficiency: ROIC {roic:.1%} and ROE {roe:.1%}")

        # Check for potential hidden value
        if (latest_metrics.get('cash_and_equivalents') and latest_metrics.get('total_liabilities') and 
            latest_metrics.get('market_cap') and latest_metrics.get('ordinary_shares_number')):
            
            net_cash = latest_metrics['cash_and_equivalents'] - latest_metrics['total_liabilities']
            shares = latest_metrics['ordinary_shares_number']
            market_cap = latest_metrics['market_cap']
            
            if shares > 0 and market_cap > 0:
                net_cash_per_share = net_cash / shares
                current_price = market_cap / shares
                
                # If net cash per share is significant relative to stock price
                if net_cash_per_share > current_price * 0.2:  # Net cash > 20% of market cap per share
                    score += 2
                    reasoning.append(f"Significant net cash position: ${net_cash_per_share:.2f} per share vs. ${current_price:.2f} stock price")

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
        analysis['type'] = 'market_inefficiency_analysis'
        analysis['title'] = f'Market Inefficiency Analysis'

        analysis_data['market_inefficiency_analysis'] = analysis
        ai_message = AIMessage(content=self.get_markdown(analysis))
        return {
            "context": context,
            "messages": [
                ai_message
            ]
        }