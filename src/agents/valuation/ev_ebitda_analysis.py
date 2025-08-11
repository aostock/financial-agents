from common.agent_state import AgentState
from langchain.schema import AIMessage
from langchain_core.callbacks import dispatch_custom_event
from langchain_core.runnables import RunnableConfig
from typing import Dict, Any
import time
from common import markdown
from langgraph.types import StreamWriter
import math
from statistics import median

class EVEBITDAAnalysis():
    def __init__(self, options: Dict[str, Any]):
        self.options = options

    def calculate_ev_ebitda_value(self, financial_metrics: list):
        """Implied equity value via median EV/EBITDA multiple."""
        if not financial_metrics:
            return 0
        m0 = financial_metrics[0]
        if not (m0.get('enterprise_value') and m0.get('enterprise_value_to_ebitda_ratio')):
            return 0
        if m0.get('enterprise_value_to_ebitda_ratio') == 0:
            return 0

        ebitda_now = m0.get('enterprise_value') / m0.get('enterprise_value_to_ebitda_ratio')
        # Get median multiple from historical data
        ev_ebitda_ratios = [m.get('enterprise_value_to_ebitda_ratio') for m in financial_metrics if m.get('enterprise_value_to_ebitda_ratio')]
        if not ev_ebitda_ratios:
            return 0
            
        med_mult = median([r for r in ev_ebitda_ratios if r > 0])
        ev_implied = med_mult * ebitda_now
        net_debt = (m0.get('enterprise_value') or 0) - (m0.get('market_cap') or 0)
        return max(ev_implied - net_debt, 0)

    def analyze(self, metrics: list, historical_metrics: list) -> dict[str, any]:
        """Analyze EV/EBITDA valuation."""
        result = {"score": 0, "max_score": 10, "details": [], "indicators": {}}
        if not metrics or len(metrics) == 0:
            result["details"].append('Insufficient financial data for EV/EBITDA analysis')
            return result

        # Get the most recent metrics
        m0 = metrics[0] if metrics else None
        if not m0:
            result["details"].append('No financial metrics available')
            return result

        # Extract required data
        enterprise_value = m0.get('enterprise_value')
        ev_ebitda_ratio = m0.get('enterprise_value_to_ebitda_ratio')
        market_cap = m0.get('market_cap')
        
        if not enterprise_value or not ev_ebitda_ratio or ev_ebitda_ratio <= 0:
            result["details"].append('Insufficient EV/EBITDA data for analysis')
            return result

        # Calculate EBITDA
        ebitda = enterprise_value / ev_ebitda_ratio
        
        # Get historical EV/EBITDA ratios for comparison
        historical_ratios = [m.get('enterprise_value_to_ebitda_ratio') for m in historical_metrics if m.get('enterprise_value_to_ebitda_ratio') and m.get('enterprise_value_to_ebitda_ratio') > 0]
        
        if not historical_ratios:
            result["details"].append('Insufficient historical data for EV/EBITDA analysis')
            return result
            
        # Calculate median and compare
        median_ratio = median(historical_ratios)
        current_ratio = ev_ebitda_ratio
        
        # Calculate implied equity value using median multiple
        implied_equity_value = self.calculate_ev_ebitda_value(metrics + historical_metrics)
        
        # Store indicators
        result["indicators"] = {
            "enterprise_value": enterprise_value,
            "ebitda": ebitda,
            "current_ev_ebitda_ratio": current_ratio,
            "median_ev_ebitda_ratio": median_ratio,
            "implied_equity_value": implied_equity_value,
            "market_cap": market_cap
        }
        
        score = 5  # Start with neutral score
        reasoning = []
        
        # Compare current ratio to median
        ratio_comparison = (current_ratio - median_ratio) / median_ratio
        
        if ratio_comparison < -0.3:  # 30%+ below median
            score += 3
            reasoning.append(f"Significant undervaluation (EV/EBITDA: {current_ratio:.1f} vs median: {median_ratio:.1f})")
        elif ratio_comparison < -0.15:  # 15%+ below median
            score += 2
            reasoning.append(f"Undervaluation (EV/EBITDA: {current_ratio:.1f} vs median: {median_ratio:.1f})")
        elif ratio_comparison < -0.05:  # 5%+ below median
            score += 1
            reasoning.append(f"Slight undervaluation (EV/EBITDA: {current_ratio:.1f} vs median: {median_ratio:.1f})")
        elif ratio_comparison < 0.05:  # Within 5% of median
            reasoning.append(f"Fair valuation (EV/EBITDA: {current_ratio:.1f} vs median: {median_ratio:.1f})")
        elif ratio_comparison < 0.15:  # 5-15% above median
            score -= 1
            reasoning.append(f"Slight overvaluation (EV/EBITDA: {current_ratio:.1f} vs median: {median_ratio:.1f})")
        elif ratio_comparison < 0.30:  # 15-30% above median
            score -= 2
            reasoning.append(f"Overvaluation (EV/EBITDA: {current_ratio:.1f} vs median: {median_ratio:.1f})")
        else:  # 30%+ above median
            score -= 3
            reasoning.append(f"Significant overvaluation (EV/EBITDA: {current_ratio:.1f} vs median: {median_ratio:.1f})")
        
        # Absolute level assessment
        if current_ratio < 8:
            reasoning.append(f"Attractive EV/EBITDA multiple ({current_ratio:.1f})")
        elif current_ratio < 12:
            reasoning.append(f"Reasonable EV/EBITDA multiple ({current_ratio:.1f})")
        elif current_ratio < 15:
            reasoning.append(f"Elevated EV/EBITDA multiple ({current_ratio:.1f})")
        else:
            reasoning.append(f"Rich EV/EBITDA multiple ({current_ratio:.1f})")
        
        # Implied value vs market cap
        if market_cap and implied_equity_value > 0:
            value_gap = (implied_equity_value - market_cap) / market_cap
            result["indicators"]["implied_value_gap"] = value_gap
            
            if value_gap > 0.30:  # 30%+ above market cap
                reasoning.append(f"Strong value indication (implied value: ${implied_equity_value:,.0f}M vs Market Cap: ${market_cap:,.0f}M)")
            elif value_gap > 0.15:  # 15%+ above market cap
                reasoning.append(f"Value indication (implied value: ${implied_equity_value:,.0f}M vs Market Cap: ${market_cap:,.0f}M)")
            elif value_gap > 0.05:  # 5%+ above market cap
                reasoning.append(f"Modest value indication (implied value: ${implied_equity_value:,.0f}M vs Market Cap: ${market_cap:,.0f}M)")
            elif value_gap > -0.05:  # Within 5% of market cap
                reasoning.append(f"Fair valuation (implied value: ${implied_equity_value:,.0f}M vs Market Cap: ${market_cap:,.0f}M)")
            else:  # Below market cap
                reasoning.append(f"Valuation concern (implied value: ${implied_equity_value:,.0f}M vs Market Cap: ${market_cap:,.0f}M)")
        
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
        analysis['type'] = 'ev_ebitda_analysis'
        analysis['title'] = f'EV/EBITDA Analysis'

        analysis_data['ev_ebitda_analysis'] = analysis
        ai_message = AIMessage(content=self.get_markdown(analysis))
        return {
            "context": context,
            "messages": [
                ai_message
            ]
        }