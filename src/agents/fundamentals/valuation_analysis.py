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
        """Analyze company valuation using multiple approaches."""
        result = {"score": 0, "max_score": 10, "details": [], "intrinsic_value": None}
        if not metrics:
            result["details"].append('No metrics available')
            return result

        latest_metrics = metrics[0]

        score = 0
        reasoning = []
        intrinsic_value = None

        # 1. Simple DCF Valuation based on Free Cash Flow
        if latest_metrics.get('free_cash_flow') and latest_metrics.get('ordinary_shares_number'):
            fcf = latest_metrics['free_cash_flow']
            shares = latest_metrics['ordinary_shares_number']
            
            # Assume 10% discount rate and 3% perpetual growth (simple model)
            discount_rate = 0.10
            growth_rate = 0.03
            
            # Project FCF for 10 years
            projected_fcfs = [fcf * (1 + growth_rate) ** i for i in range(1, 11)]
            
            # Present value of projected FCFs
            pv_fcfs = sum(fcf_val / ((1 + discount_rate) ** i) for i, fcf_val in enumerate(projected_fcfs, 1))
            
            # Terminal value
            terminal_value = (projected_fcfs[-1] * (1 + growth_rate)) / (discount_rate - growth_rate)
            pv_terminal = terminal_value / ((1 + discount_rate) ** 10)
            
            # Total enterprise value
            enterprise_value = pv_fcfs + pv_terminal
            
            # Approximate equity value (simplified)
            net_debt = 0  # Simplification - would need debt data for accurate calculation
            equity_value = enterprise_value - net_debt
            
            # Intrinsic value per share
            intrinsic_value = equity_value / shares if shares > 0 else None
            
            # Compare to current market cap
            market_cap = latest_metrics.get('market_cap')
            if intrinsic_value and market_cap and shares > 0:
                current_price = market_cap / shares
                iv_price_ratio = intrinsic_value / current_price if current_price > 0 else 0
                
                if iv_price_ratio > 1.5:  # 50%+ margin of safety
                    score += 4
                    reasoning.append(f"Significant undervaluation (IV/Price: {iv_price_ratio:.2f}x)")
                elif iv_price_ratio > 1.2:  # 20%+ margin of safety
                    score += 3
                    reasoning.append(f"Undervalued (IV/Price: {iv_price_ratio:.2f}x)")
                elif iv_price_ratio > 0.9:  # Near fair value
                    score += 2
                    reasoning.append(f"Fairly valued (IV/Price: {iv_price_ratio:.2f}x)")
                elif iv_price_ratio > 0.7:  # 20-30% overvalued
                    score += 1
                    reasoning.append(f"Somewhat overvalued (IV/Price: {iv_price_ratio:.2f}x)")
                else:
                    reasoning.append(f"Significantly overvalued (IV/Price: {iv_price_ratio:.2f}x)")

        # 2. P/E Ratio Analysis
        if latest_metrics.get('price_to_earnings_ratio'):
            pe_ratio = latest_metrics['price_to_earnings_ratio']
            
            # Compare to historical or sector average (using simplified thresholds)
            if pe_ratio < 15:
                score += 2
                reasoning.append(f"Attractive P/E ratio ({pe_ratio:.1f}x < 15x)")
            elif pe_ratio < 20:
                score += 1
                reasoning.append(f"Reasonable P/E ratio ({pe_ratio:.1f}x < 20x)")
            else:
                reasoning.append(f"High P/E ratio ({pe_ratio:.1f}x > 20x)")

        # 3. P/B Ratio Analysis
        if latest_metrics.get('price_to_book_ratio'):
            pb_ratio = latest_metrics['price_to_book_ratio']
            
            # Compare to thresholds
            if pb_ratio < 1.5:
                score += 2
                reasoning.append(f"Low P/B ratio ({pb_ratio:.1f}x < 1.5x)")
            elif pb_ratio < 3.0:
                score += 1
                reasoning.append(f"Reasonable P/B ratio ({pb_ratio:.1f}x < 3.0x)")
            else:
                reasoning.append(f"High P/B ratio ({pb_ratio:.1f}x > 3.0x)")

        # 4. EV/EBIT Ratio Analysis (if data available)
        if latest_metrics.get('enterprise_value') and latest_metrics.get('ebit'):
            ebit = latest_metrics['ebit']
            if ebit > 0:
                ev_ebit = latest_metrics['enterprise_value'] / ebit
                
                if ev_ebit < 10:
                    score += 1
                    reasoning.append(f"Attractive EV/EBIT ({ev_ebit:.1f}x < 10x)")
                elif ev_ebit < 15:
                    reasoning.append(f"Reasonable EV/EBIT ({ev_ebit:.1f}x < 15x)")
                else:
                    reasoning.append(f"High EV/EBIT ({ev_ebit:.1f}x > 15x)")

        result["score"] = score
        result["details"] = reasoning
        result["intrinsic_value"] = intrinsic_value
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