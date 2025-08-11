from common.agent_state import AgentState
from langchain.schema import AIMessage
from langchain_core.callbacks import dispatch_custom_event
from langchain_core.runnables import RunnableConfig
from typing import Dict, Any
import time
from common import markdown
from langgraph.types import StreamWriter
import math

class DCFAnalysis():
    def __init__(self, options: Dict[str, Any]):
        self.options = options

    def calculate_intrinsic_value(
        self,
        free_cash_flow: float,
        growth_rate: float = 0.05,
        discount_rate: float = 0.10,
        terminal_growth_rate: float = 0.02,
        num_years: int = 5,
    ) -> float:
        """Classic DCF on FCF with constant growth and terminal value."""
        if free_cash_flow is None or free_cash_flow <= 0:
            return 0

        pv = 0.0
        for yr in range(1, num_years + 1):
            fcft = free_cash_flow * (1 + growth_rate) ** yr
            pv += fcft / (1 + discount_rate) ** yr

        term_val = (
            free_cash_flow * (1 + growth_rate) ** num_years * (1 + terminal_growth_rate)
        ) / (discount_rate - terminal_growth_rate)
        pv_term = term_val / (1 + discount_rate) ** num_years

        return pv + pv_term

    def analyze(self, metrics: list) -> dict[str, any]:
        """Analyze DCF valuation."""
        result = {"score": 0, "max_score": 10, "details": [], "indicators": {}}
        if not metrics or len(metrics) == 0:
            result["details"].append('Insufficient financial data for DCF analysis')
            return result

        # Get the most recent metrics
        m0 = metrics[0] if metrics else None
        if not m0:
            result["details"].append('No financial metrics available')
            return result

        # Extract required data
        free_cash_flow = m0.get('free_cash_flow')
        market_cap = m0.get('market_cap')
        earnings_growth = m0.get('earnings_growth', 0.05)  # Default to 5%
        
        if not free_cash_flow or free_cash_flow <= 0:
            result["details"].append('Insufficient free cash flow data for DCF analysis')
            return result

        # Calculate intrinsic value using DCF
        # Use conservative assumptions
        growth_rate = min(earnings_growth, 0.10)  # Cap at 10%
        discount_rate = 0.10  # 10% cost of capital
        terminal_growth_rate = 0.02  # 2% terminal growth
        intrinsic_value = self.calculate_intrinsic_value(
            free_cash_flow, growth_rate, discount_rate, terminal_growth_rate
        )
        
        # Apply margin of safety (20%)
        intrinsic_value_safety = intrinsic_value * 0.8

        # Store indicators
        result["indicators"] = {
            "free_cash_flow": free_cash_flow,
            "growth_rate": growth_rate,
            "discount_rate": discount_rate,
            "terminal_growth_rate": terminal_growth_rate,
            "intrinsic_value": intrinsic_value,
            "intrinsic_value_safety": intrinsic_value_safety,
            "market_cap": market_cap
        }
        
        score = 5  # Start with neutral score
        reasoning = []
        
        # Calculate margin of safety
        if market_cap and intrinsic_value_safety > 0:
            margin_of_safety = (intrinsic_value_safety - market_cap) / market_cap
            
            result["indicators"]["margin_of_safety"] = margin_of_safety
            
            if margin_of_safety > 0.5:  # 50%+ margin of safety
                score += 3
                reasoning.append(f"Strong margin of safety ({margin_of_safety:.1%}) - DCF value: ${intrinsic_value_safety:,.0f}M vs Market Cap: ${market_cap:,.0f}M")
            elif margin_of_safety > 0.25:  # 25%+ margin of safety
                score += 2
                reasoning.append(f"Good margin of safety ({margin_of_safety:.1%}) - DCF value: ${intrinsic_value_safety:,.0f}M vs Market Cap: ${market_cap:,.0f}M")
            elif margin_of_safety > 0.10:  # 10%+ margin of safety
                score += 1
                reasoning.append(f"Modest margin of safety ({margin_of_safety:.1%}) - DCF value: ${intrinsic_value_safety:,.0f}M vs Market Cap: ${market_cap:,.0f}M")
            elif margin_of_safety > 0:  # Positive but small margin
                reasoning.append(f"Fair valuation (small margin of safety: {margin_of_safety:.1%}) - DCF value: ${intrinsic_value_safety:,.0f}M vs Market Cap: ${market_cap:,.0f}M")
            elif margin_of_safety > -0.10:  # Slightly overvalued
                score -= 1
                reasoning.append(f"Slight overvaluation ({margin_of_safety:.1%}) - DCF value: ${intrinsic_value_safety:,.0f}M vs Market Cap: ${market_cap:,.0f}M")
            elif margin_of_safety > -0.25:  # Moderately overvalued
                score -= 2
                reasoning.append(f"Moderate overvaluation ({margin_of_safety:.1%}) - DCF value: ${intrinsic_value_safety:,.0f}M vs Market Cap: ${market_cap:,.0f}M")
            else:  # Severely overvalued
                score -= 3
                reasoning.append(f"Significant overvaluation ({margin_of_safety:.1%}) - DCF value: ${intrinsic_value_safety:,.0f}M vs Market Cap: ${market_cap:,.0f}M")
        else:
            reasoning.append("Unable to calculate margin of safety - missing market cap data")
        
        # Growth rate assessment
        if growth_rate > 0.10:
            reasoning.append(f"Conservative growth assumption applied (capped at 10%)")
        elif growth_rate > 0.07:
            reasoning.append(f"Healthy growth assumption ({growth_rate:.1%})")
        elif growth_rate > 0.04:
            reasoning.append(f"Moderate growth assumption ({growth_rate:.1%})")
        elif growth_rate > 0.02:
            reasoning.append(f"Conservative growth assumption ({growth_rate:.1%})")
        else:
            reasoning.append(f"Low growth assumption ({growth_rate:.1%}) - limiting valuation")
        
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
        analysis['type'] = 'dcf_analysis'
        analysis['title'] = f'Discounted Cash Flow Analysis'

        analysis_data['dcf_analysis'] = analysis
        ai_message = AIMessage(content=self.get_markdown(analysis))
        return {
            "context": context,
            "messages": [
                ai_message
            ]
        }