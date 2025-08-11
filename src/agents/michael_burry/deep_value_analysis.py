from common.agent_state import AgentState
from langchain.schema import AIMessage
from langchain_core.callbacks import dispatch_custom_event
from langchain_core.runnables import RunnableConfig
from typing import Dict, Any

import time
from common import markdown
from langgraph.types import StreamWriter


class DeepValueAnalysis():
    def __init__(self, options: Dict[str, Any]):
        self.options = options

    
    def analyze(self, metrics: list) -> dict[str, any]:
        """Analyze deep value opportunities and calculate intrinsic value."""
        result = {"score": 0, "max_score": 10, "details": [], "intrinsic_value": None}
        if not metrics:
            result["details"].append('No metrics available')
            return result

        latest_metrics = metrics[0]

        score = 0
        reasoning = []
        intrinsic_value = None

        # 1. Asset-based valuation (Benjamin Graham approach)
        if (latest_metrics.get('total_assets') and latest_metrics.get('total_liabilities') and 
            latest_metrics.get('ordinary_shares_number')):
            
            book_value = latest_metrics['total_assets'] - latest_metrics['total_liabilities']
            shares = latest_metrics['ordinary_shares_number']
            
            if shares > 0:
                book_value_per_share = book_value / shares
                
                # Net current asset value (NCAV) - more conservative approach
                current_assets = 0
                if latest_metrics.get('cash_and_equivalents'):
                    current_assets += latest_metrics['cash_and_equivalents']
                if latest_metrics.get('accounts_receivable'):
                    current_assets += latest_metrics['accounts_receivable']
                if latest_metrics.get('inventory'):
                    current_assets += latest_metrics['inventory']
                
                ncav = current_assets - latest_metrics.get('total_liabilities', 0)
                ncav_per_share = ncav / shares if shares > 0 else 0
                
                market_cap = latest_metrics.get('market_cap', 0)
                current_price = market_cap / shares if shares > 0 and market_cap else 0
                
                # Check for deep value opportunities (Graham's criteria)
                if current_price > 0:
                    if current_price < book_value_per_share * 0.67:  # Two-thirds of book value
                        score += 3
                        reasoning.append(f"Significant discount to book value: ${current_price:.2f} vs. ${book_value_per_share:.2f} BV")
                    elif current_price < book_value_per_share * 0.85:  # 15%+ discount
                        score += 2
                        reasoning.append(f"Discount to book value: ${current_price:.2f} vs. ${book_value_per_share:.2f} BV")
                    
                    if current_price < ncav_per_share * 1.5:  # Near NCAV
                        score += 2
                        reasoning.append(f"Near net current asset value: ${current_price:.2f} vs. ${ncav_per_share:.2f} NCAV")

        # 2. Earnings-based valuation (P/E approach)
        if latest_metrics.get('net_income') and latest_metrics.get('ordinary_shares_number') and latest_metrics.get('market_cap'):
            shares = latest_metrics['ordinary_shares_number']
            market_cap = latest_metrics['market_cap']
            
            if shares > 0 and market_cap > 0:
                current_price = market_cap / shares
                eps = latest_metrics['net_income'] / shares if shares > 0 else 0
                
                if eps > 0:
                    pe_ratio = current_price / eps
                    
                    # Look for low P/E stocks (value opportunities)
                    if pe_ratio < 10:
                        score += 2
                        reasoning.append(f"Low P/E ratio suggesting value: {pe_ratio:.1f}x")
                    elif pe_ratio < 15:
                        score += 1
                        reasoning.append(f"Reasonable P/E ratio: {pe_ratio:.1f}x")

        # 3. Free cash flow yield analysis
        if latest_metrics.get('free_cash_flow') and latest_metrics.get('market_cap'):
            fcf = latest_metrics['free_cash_flow']
            market_cap = latest_metrics['market_cap']
            
            if market_cap > 0:
                fcf_yield = fcf / market_cap if fcf > 0 else 0
                
                if fcf_yield > 0.08:  # 8%+ FCF yield
                    score += 2
                    reasoning.append(f"Attractive FCF yield: {fcf_yield:.1%}")
                elif fcf_yield > 0.05:  # 5%+ FCF yield
                    score += 1
                    reasoning.append(f"Good FCF yield: {fcf_yield:.1%}")

        # 4. Simple DCF valuation (conservative approach)
        if latest_metrics.get('free_cash_flow') and latest_metrics.get('ordinary_shares_number'):
            fcf = latest_metrics['free_cash_flow']
            shares = latest_metrics['ordinary_shares_number']
            
            if fcf > 0 and shares > 0:
                # Conservative assumptions: 12% discount rate, 2% perpetual growth
                discount_rate = 0.12
                growth_rate = 0.02
                
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
                intrinsic_value = equity_value / shares
                market_cap = latest_metrics.get('market_cap', 0)
                current_price = market_cap / shares if shares > 0 and market_cap else 0
                
                if current_price > 0 and intrinsic_value > 0:
                    margin_of_safety = (intrinsic_value - current_price) / current_price
                    
                    if margin_of_safety > 0.5:  # 50%+ margin of safety
                        score += 3
                        reasoning.append(f"Significant margin of safety: IV ${intrinsic_value:.2f} vs. current ${current_price:.2f} ({margin_of_safety:.1%})")
                    elif margin_of_safety > 0.25:  # 25%+ margin of safety
                        score += 2
                        reasoning.append(f"Good margin of safety: IV ${intrinsic_value:.2f} vs. current ${current_price:.2f} ({margin_of_safety:.1%})")
                    elif margin_of_safety > 0.1:  # 10%+ margin of safety
                        score += 1
                        reasoning.append(f"Modest margin of safety: IV ${intrinsic_value:.2f} vs. current ${current_price:.2f} ({margin_of_safety:.1%})")

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
        analysis['type'] = 'deep_value_analysis'
        analysis['title'] = f'Deep Value Analysis'

        analysis_data['deep_value_analysis'] = analysis
        ai_message = AIMessage(content=self.get_markdown(analysis))
        return {
            "context": context,
            "messages": [
                ai_message
            ]
        }