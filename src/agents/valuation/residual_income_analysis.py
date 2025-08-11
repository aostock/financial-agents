from common.agent_state import AgentState
from langchain.schema import AIMessage
from langchain_core.callbacks import dispatch_custom_event
from langchain_core.runnables import RunnableConfig
from typing import Dict, Any
import time
from common import markdown
from langgraph.types import StreamWriter
import math

class ResidualIncomeAnalysis():
    def __init__(self, options: Dict[str, Any]):
        self.options = options

    def calculate_residual_income_value(
        self,
        market_cap: float,
        net_income: float,
        price_to_book_ratio: float,
        book_value_growth: float = 0.03,
        cost_of_equity: float = 0.10,
        terminal_growth_rate: float = 0.03,
        num_years: int = 5,
    ):
        """Residual Income Model (Edwards-Bell-Ohlson)."""
        if not (market_cap and net_income and price_to_book_ratio and price_to_book_ratio > 0):
            return 0

        book_val = market_cap / price_to_book_ratio
        ri0 = net_income - cost_of_equity * book_val
        if ri0 <= 0:
            return 0

        pv_ri = 0.0
        for yr in range(1, num_years + 1):
            ri_t = ri0 * (1 + book_value_growth) ** yr
            pv_ri += ri_t / (1 + cost_of_equity) ** yr

        term_ri = ri0 * (1 + book_value_growth) ** (num_years + 1) / (
            cost_of_equity - terminal_growth_rate
        )
        pv_term = term_ri / (1 + cost_of_equity) ** num_years

        intrinsic = book_val + pv_ri + pv_term
        return intrinsic * 0.8  # 20% margin of safety

    def analyze(self, metrics: list) -> dict[str, any]:
        """Analyze residual income valuation."""
        result = {"score": 0, "max_score": 10, "details": [], "indicators": {}}
        if not metrics or len(metrics) == 0:
            result["details"].append('Insufficient financial data for residual income analysis')
            return result

        # Get the most recent metrics
        m0 = metrics[0] if metrics else None
        if not m0:
            result["details"].append('No financial metrics available')
            return result

        # Extract required data
        market_cap = m0.get('market_cap')
        net_income = m0.get('net_income')
        price_to_book_ratio = m0.get('price_to_book_ratio')
        book_value_growth = m0.get('book_value_growth', 0.03)  # Default to 3%
        return_on_equity = m0.get('return_on_equity')
        
        if not all(v is not None for v in [market_cap, net_income, price_to_book_ratio]) or price_to_book_ratio <= 0:
            result["details"].append('Insufficient data for residual income analysis')
            return result

        # Calculate book value
        book_val = market_cap / price_to_book_ratio
        
        # Calculate residual income
        cost_of_equity = 0.10  # 10% cost of equity
        residual_income = net_income - cost_of_equity * book_val
        
        # Calculate intrinsic value using residual income model
        intrinsic_value = self.calculate_residual_income_value(
            market_cap, net_income, price_to_book_ratio, 
            book_value_growth, cost_of_equity
        )

        # Store indicators
        result["indicators"] = {
            "market_cap": market_cap,
            "net_income": net_income,
            "book_value": book_val,
            "price_to_book_ratio": price_to_book_ratio,
            "return_on_equity": return_on_equity,
            "book_value_growth": book_value_growth,
            "cost_of_equity": cost_of_equity,
            "residual_income": residual_income,
            "intrinsic_value": intrinsic_value
        }
        
        score = 5  # Start with neutral score
        reasoning = []
        
        # Calculate margin of safety
        if market_cap and intrinsic_value > 0:
            margin_of_safety = (intrinsic_value - market_cap) / market_cap
            
            result["indicators"]["margin_of_safety"] = margin_of_safety
            
            if margin_of_safety > 0.5:  # 50%+ margin of safety
                score += 3
                reasoning.append(f"Strong margin of safety ({margin_of_safety:.1%}) - RIM value: ${intrinsic_value:,.0f}M vs Market Cap: ${market_cap:,.0f}M")
            elif margin_of_safety > 0.25:  # 25%+ margin of safety
                score += 2
                reasoning.append(f"Good margin of safety ({margin_of_safety:.1%}) - RIM value: ${intrinsic_value:,.0f}M vs Market Cap: ${market_cap:,.0f}M")
            elif margin_of_safety > 0.10:  # 10%+ margin of safety
                score += 1
                reasoning.append(f"Modest margin of safety ({margin_of_safety:.1%}) - RIM value: ${intrinsic_value:,.0f}M vs Market Cap: ${market_cap:,.0f}M")
            elif margin_of_safety > 0:  # Positive but small margin
                reasoning.append(f"Fair valuation (small margin of safety: {margin_of_safety:.1%}) - RIM value: ${intrinsic_value:,.0f}M vs Market Cap: ${market_cap:,.0f}M")
            elif margin_of_safety > -0.10:  # Slightly overvalued
                score -= 1
                reasoning.append(f"Slight overvaluation ({margin_of_safety:.1%}) - RIM value: ${intrinsic_value:,.0f}M vs Market Cap: ${market_cap:,.0f}M")
            elif margin_of_safety > -0.25:  # Moderately overvalued
                score -= 2
                reasoning.append(f"Moderate overvaluation ({margin_of_safety:.1%}) - RIM value: ${intrinsic_value:,.0f}M vs Market Cap: ${market_cap:,.0f}M")
            else:  # Severely overvalued
                score -= 3
                reasoning.append(f"Significant overvaluation ({margin_of_safety:.1%}) - RIM value: ${intrinsic_value:,.0f}M vs Market Cap: ${market_cap:,.0f}M")
        else:
            reasoning.append("Unable to calculate margin of safety - missing market cap data")
        
        # ROE assessment
        if return_on_equity is not None:
            if return_on_equity > 0.20:  # 20%+ ROE
                reasoning.append(f"Exceptional ROE ({return_on_equity:.1%}) - strong competitive positioning")
            elif return_on_equity > 0.15:  # 15%+ ROE
                reasoning.append(f"Strong ROE ({return_on_equity:.1%}) - good capital efficiency")
            elif return_on_equity > 0.10:  # 10%+ ROE
                reasoning.append(f"Decent ROE ({return_on_equity:.1%}) - acceptable returns")
            elif return_on_equity > 0.05:  # 5%+ ROE
                reasoning.append(f"Low ROE ({return_on_equity:.1%}) - concerning returns")
            else:
                reasoning.append(f"Poor ROE ({return_on_equity:.1%}) - value destruction")
        
        # Residual income assessment
        if residual_income > 0:
            ri_to_net_income = residual_income / net_income if net_income > 0 else 0
            if ri_to_net_income > 0.5:
                reasoning.append(f"High quality residual income ({ri_to_net_income:.1%} of net income)")
            elif ri_to_net_income > 0.3:
                reasoning.append(f"Good quality residual income ({ri_to_net_income:.1%} of net income)")
            elif ri_to_net_income > 0.1:
                reasoning.append(f"Moderate residual income ({ri_to_net_income:.1%} of net income)")
            else:
                reasoning.append(f"Low residual income ({ri_to_net_income:.1%} of net income)")
        else:
            reasoning.append("Negative residual income - not covering cost of equity")
        
        # P/B ratio assessment
        if price_to_book_ratio < 1:
            reasoning.append(f"Attractive P/B ratio ({price_to_book_ratio:.1f}) - potential value")
        elif price_to_book_ratio < 2:
            reasoning.append(f"Reasonable P/B ratio ({price_to_book_ratio:.1f})")
        elif price_to_book_ratio < 3:
            reasoning.append(f"Elevated P/B ratio ({price_to_book_ratio:.1f})")
        else:
            reasoning.append(f"Rich P/B ratio ({price_to_book_ratio:.1f}) - expensive valuation")
        
        result["score"] = max(0, min(10, score))  # Clamp between 0-10
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
        analysis['type'] = 'residual_income_analysis'
        analysis['title'] = f'Residual Income Analysis'

        analysis_data['residual_income_analysis'] = analysis
        ai_message = AIMessage(content=self.get_markdown(analysis))
        return {
            "context": context,
            "messages": [
                ai_message
            ]
        }