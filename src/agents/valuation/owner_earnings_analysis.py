from common.agent_state import AgentState
from langchain.schema import AIMessage
from langchain_core.callbacks import dispatch_custom_event
from langchain_core.runnables import RunnableConfig
from typing import Dict, Any
import time
from common import markdown
from langgraph.types import StreamWriter
import math

class OwnerEarningsAnalysis():
    def __init__(self, options: Dict[str, Any]):
        self.options = options

    def calculate_owner_earnings_value(
        self,
        net_income: float,
        depreciation: float,
        capex: float,
        working_capital_change: float,
        growth_rate: float = 0.05,
        required_return: float = 0.15,
        margin_of_safety: float = 0.25,
        num_years: int = 5,
    ) -> float:
        """Buffett owner-earnings valuation with margin-of-safety."""
        if not all(isinstance(x, (int, float)) for x in [net_income, depreciation, capex, working_capital_change]):
            return 0

        owner_earnings = net_income + depreciation - capex - working_capital_change
        if owner_earnings <= 0:
            return 0

        pv = 0.0
        for yr in range(1, num_years + 1):
            future = owner_earnings * (1 + growth_rate) ** yr
            pv += future / (1 + required_return) ** yr

        terminal_growth = min(growth_rate, 0.03)
        term_val = (owner_earnings * (1 + growth_rate) ** num_years * (1 + terminal_growth)) / (
            required_return - terminal_growth
        )
        pv_term = term_val / (1 + required_return) ** num_years

        intrinsic = pv + pv_term
        return intrinsic * (1 - margin_of_safety)

    def analyze(self, metrics: list, historical_metrics: list) -> dict[str, any]:
        """Analyze owner earnings valuation."""
        result = {"score": 0, "max_score": 10, "details": [], "indicators": {}}
        if not metrics or len(metrics) == 0:
            result["details"].append('Insufficient financial data for owner earnings analysis')
            return result

        # Get the most recent metrics
        m0 = metrics[0] if metrics else None
        if not m0:
            result["details"].append('No financial metrics available')
            return result

        # Extract required data
        net_income = m0.get('net_income')
        depreciation = m0.get('depreciation_and_amortization')
        capex = m0.get('capital_expenditure')
        working_capital = m0.get('working_capital')
        market_cap = m0.get('market_cap')
        earnings_growth = m0.get('earnings_growth', 0.05)  # Default to 5%
        
        # Calculate working capital change (need at least 2 periods)
        working_capital_change = 0
        if len(metrics) >= 2 and working_capital is not None:
            working_capital_prev = metrics[1].get('working_capital')
            if working_capital_prev is not None:
                working_capital_change = working_capital - working_capital_prev

        # Check if we have all required data
        if not all(v is not None for v in [net_income, depreciation, capex, working_capital]):
            result["details"].append('Insufficient financial data for owner earnings calculation')
            return result

        # Calculate owner earnings
        owner_earnings = net_income + depreciation - capex - working_capital_change
        
        # Calculate intrinsic value using owner earnings
        # Use conservative assumptions
        growth_rate = min(earnings_growth, 0.10)  # Cap at 10%
        required_return = 0.15  # 15% required return
        intrinsic_value = self.calculate_owner_earnings_value(
            net_income, depreciation, capex, working_capital_change,
            growth_rate, required_return
        )

        # Store indicators
        result["indicators"] = {
            "net_income": net_income,
            "depreciation": depreciation,
            "capex": capex,
            "working_capital_change": working_capital_change,
            "owner_earnings": owner_earnings,
            "growth_rate": growth_rate,
            "required_return": required_return,
            "intrinsic_value": intrinsic_value,
            "market_cap": market_cap
        }
        
        score = 5  # Start with neutral score
        reasoning = []
        
        # Calculate margin of safety
        if market_cap and intrinsic_value > 0:
            margin_of_safety = (intrinsic_value - market_cap) / market_cap
            
            result["indicators"]["margin_of_safety"] = margin_of_safety
            
            if margin_of_safety > 0.5:  # 50%+ margin of safety
                score += 3
                reasoning.append(f"Strong margin of safety ({margin_of_safety:.1%}) - Owner earnings value: ${intrinsic_value:,.0f}M vs Market Cap: ${market_cap:,.0f}M")
            elif margin_of_safety > 0.25:  # 25%+ margin of safety
                score += 2
                reasoning.append(f"Good margin of safety ({margin_of_safety:.1%}) - Owner earnings value: ${intrinsic_value:,.0f}M vs Market Cap: ${market_cap:,.0f}M")
            elif margin_of_safety > 0.10:  # 10%+ margin of safety
                score += 1
                reasoning.append(f"Modest margin of safety ({margin_of_safety:.1%}) - Owner earnings value: ${intrinsic_value:,.0f}M vs Market Cap: ${market_cap:,.0f}M")
            elif margin_of_safety > 0:  # Positive but small margin
                reasoning.append(f"Fair valuation (small margin of safety: {margin_of_safety:.1%}) - Owner earnings value: ${intrinsic_value:,.0f}M vs Market Cap: ${market_cap:,.0f}M")
            elif margin_of_safety > -0.10:  # Slightly overvalued
                score -= 1
                reasoning.append(f"Slight overvaluation ({margin_of_safety:.1%}) - Owner earnings value: ${intrinsic_value:,.0f}M vs Market Cap: ${market_cap:,.0f}M")
            elif margin_of_safety > -0.25:  # Moderately overvalued
                score -= 2
                reasoning.append(f"Moderate overvaluation ({margin_of_safety:.1%}) - Owner earnings value: ${intrinsic_value:,.0f}M vs Market Cap: ${market_cap:,.0f}M")
            else:  # Severely overvalued
                score -= 3
                reasoning.append(f"Significant overvaluation ({margin_of_safety:.1%}) - Owner earnings value: ${intrinsic_value:,.0f}M vs Market Cap: ${market_cap:,.0f}M")
        else:
            reasoning.append("Unable to calculate margin of safety - missing market cap data")
        
        # Owner earnings quality assessment
        if owner_earnings > 0:
            oe_to_net_income = owner_earnings / net_income if net_income > 0 else 0
            if oe_to_net_income > 0.8:
                reasoning.append(f"High quality owner earnings ({oe_to_net_income:.1%} of net income)")
            elif oe_to_net_income > 0.6:
                reasoning.append(f"Good quality owner earnings ({oe_to_net_income:.1%} of net income)")
            elif oe_to_net_income > 0.4:
                reasoning.append(f"Moderate quality owner earnings ({oe_to_net_income:.1%} of net income)")
            else:
                reasoning.append(f"Lower quality owner earnings ({oe_to_net_income:.1%} of net income) - high capex or working capital needs")
        else:
            reasoning.append("Negative owner earnings - business consuming cash")
        
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
        historical_metrics = context.get('historical_metrics')
        analysis = self.analyze(metrics, historical_metrics)
        analysis['type'] = 'owner_earnings_analysis'
        analysis['title'] = f'Owner Earnings Analysis'

        analysis_data['owner_earnings_analysis'] = analysis
        ai_message = AIMessage(content=self.get_markdown(analysis))
        return {
            "context": context,
            "messages": [
                ai_message
            ]
        }