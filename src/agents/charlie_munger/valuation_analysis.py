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
        """
        Calculate intrinsic value using Munger's approach:
        - Focus on owner earnings (approximated by FCF)
        - Simple multiple on normalized earnings
        - Prefer paying a fair price for a wonderful business
        """
        result = {"score": 0, "max_score": 10, "details": []}
        if not metrics:
            result["details"].append('No metrics available')
            return result

        # Get FCF values (Munger's preferred "owner earnings" metric)
        fcf_values = [item.get('free_cash_flow') for item in metrics 
                     if item.get('free_cash_flow') is not None]
        
        if not fcf_values or len(fcf_values) < 3:
            result["details"].append('Insufficient free cash flow data for valuation')
            return result
        
        score = 0
        details = []
        
        # 1. Normalize earnings by taking average of last 3-5 years
        # (Munger prefers to normalize earnings to avoid over/under-valuation based on cyclical factors)
        normalized_fcf = sum(fcf_values[:min(5, len(fcf_values))]) / min(5, len(fcf_values))
        
        if normalized_fcf <= 0:
            result["details"].append(f"Negative or zero normalized FCF ({normalized_fcf}), cannot value")
            result["intrinsic_value"] = None
            return result
        
        # 2. Get market cap for comparison
        market_caps = [item.get('market_cap') for item in metrics 
                      if item.get('market_cap') is not None]
        
        if not market_caps or len(market_caps) == 0:
            result["details"].append("No market cap data available for valuation")
            return result
            
        market_cap = market_caps[0]  # Most recent market cap
        
        if market_cap <= 0:
            result["details"].append(f"Invalid market cap ({market_cap}), cannot value")
            return result
        
        # 3. Calculate FCF yield (inverse of P/FCF multiple)
        fcf_yield = normalized_fcf / market_cap
        
        # 4. Apply Munger's FCF multiple based on business quality
        # Munger would pay higher multiples for wonderful businesses
        # Let's use a sliding scale where higher FCF yields are more attractive
        if fcf_yield > 0.08:  # >8% FCF yield (P/FCF < 12.5x)
            score += 4
            details.append(f"Excellent value: {fcf_yield:.1%} FCF yield")
        elif fcf_yield > 0.05:  # >5% FCF yield (P/FCF < 20x)
            score += 3
            details.append(f"Good value: {fcf_yield:.1%} FCF yield")
        elif fcf_yield > 0.03:  # >3% FCF yield (P/FCF < 33x)
            score += 1
            details.append(f"Fair value: {fcf_yield:.1%} FCF yield")
        else:
            details.append(f"Expensive: Only {fcf_yield:.1%} FCF yield")
        
        # 5. Calculate simple intrinsic value range
        # Munger tends to use straightforward valuations, avoiding complex DCF models
        conservative_value = normalized_fcf * 10  # 10x FCF = 10% yield
        reasonable_value = normalized_fcf * 15    # 15x FCF ≈ 6.7% yield
        optimistic_value = normalized_fcf * 20    # 20x FCF = 5% yield
        
        # 6. Calculate margins of safety
        current_to_reasonable = (reasonable_value - market_cap) / market_cap
        
        if current_to_reasonable > 0.3:  # >30% upside
            score += 3
            details.append(f"Large margin of safety: {current_to_reasonable:.1%} upside to reasonable value")
        elif current_to_reasonable > 0.1:  # >10% upside
            score += 2
            details.append(f"Moderate margin of safety: {current_to_reasonable:.1%} upside to reasonable value")
        elif current_to_reasonable > -0.1:  # Within 10% of reasonable value
            score += 1
            details.append(f"Fair price: Within 10% of reasonable value ({current_to_reasonable:.1%})")
        else:
            details.append(f"Expensive: {-current_to_reasonable:.1%} premium to reasonable value")
        
        # 7. Check earnings trajectory for additional context
        # Munger likes growing owner earnings
        if len(fcf_values) >= 3:
            recent_avg = sum(fcf_values[:3]) / 3
            older_avg = sum(fcf_values[-3:]) / 3 if len(fcf_values) >= 6 else fcf_values[-1]
            
            if recent_avg > older_avg * 1.2:  # >20% growth in FCF
                score += 3
                details.append("Growing FCF trend adds to intrinsic value")
            elif recent_avg > older_avg:
                score += 2
                details.append("Stable to growing FCF supports valuation")
            else:
                details.append("Declining FCF trend is concerning")
        
        # Scale score to 0-10 range
        # Maximum possible raw score would be 10 (4+3+3)
        final_score = min(10, score * 10 / 10) 
        
        result["score"] = final_score
        result["details"] = details
        result["intrinsic_value_range"] = {
            "conservative": conservative_value,
            "reasonable": reasonable_value,
            "optimistic": optimistic_value
        }
        result["fcf_yield"] = fcf_yield
        result["normalized_fcf"] = normalized_fcf
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