from common.agent_state import AgentState
from langchain.schema import AIMessage
from langchain_core.callbacks import dispatch_custom_event
from langchain_core.runnables import RunnableConfig
from typing import Dict, Any

import time
from common import markdown
from langgraph.types import StreamWriter


class MoatStrengthAnalysis():
    def __init__(self, options: Dict[str, Any]):
        self.options = options

    
    def analyze(self, metrics: list) -> dict[str, any]:
        """
        Analyze the business's competitive advantage using Munger's approach:
        - Consistent high returns on capital (ROIC)
        - Pricing power (stable/improving gross margins)
        - Low capital requirements
        - Network effects and intangible assets (R&D investments, goodwill)
        """
        result = {"score": 0, "max_score": 10, "details": []}
        if not metrics:
            result["details"].append('No metrics available')
            return result

        score = 0
        details = []
        
        # 1. Return on Invested Capital (ROIC) analysis - Munger's favorite metric
        roic_values = [item.get('return_on_invested_capital') for item in metrics 
                       if item.get('return_on_invested_capital') is not None]
        
        if roic_values:
            # Check if ROIC consistently above 15% (Munger's threshold)
            high_roic_count = sum(1 for r in roic_values if r > 0.15)
            if high_roic_count >= len(roic_values) * 0.8:  # 80% of periods show high ROIC
                score += 3
                details.append(f"Excellent ROIC: >15% in {high_roic_count}/{len(roic_values)} periods")
            elif high_roic_count >= len(roic_values) * 0.5:  # 50% of periods
                score += 2
                details.append(f"Good ROIC: >15% in {high_roic_count}/{len(roic_values)} periods")
            elif high_roic_count > 0:
                score += 1
                details.append(f"Mixed ROIC: >15% in only {high_roic_count}/{len(roic_values)} periods")
            else:
                details.append("Poor ROIC: Never exceeds 15% threshold")
        else:
            details.append("No ROIC data available")
        
        # 2. Pricing power - check gross margin stability and trends
        gross_margins = [item.get('gross_margin') for item in metrics 
                        if item.get('gross_margin') is not None]
        
        if gross_margins and len(gross_margins) >= 3:
            # Munger likes stable or improving gross margins
            margin_trend = sum(1 for i in range(1, len(gross_margins)) if gross_margins[i] >= gross_margins[i-1])
            if margin_trend >= len(gross_margins) * 0.7:  # Improving in 70% of periods
                score += 2
                details.append("Strong pricing power: Gross margins consistently improving")
            elif sum(gross_margins) / len(gross_margins) > 0.3:  # Average margin > 30%
                score += 1
                details.append(f"Good pricing power: Average gross margin {sum(gross_margins)/len(gross_margins):.1%}")
            else:
                details.append("Limited pricing power: Low or declining gross margins")
        else:
            details.append("Insufficient gross margin data")
        
        # 3. Capital intensity - Munger prefers low capex businesses
        if len(metrics) >= 3:
            capex_to_revenue = []
            for item in metrics:
                if (item.get('capital_expenditure') is not None and 
                    item.get('revenue') is not None and item.get('revenue') > 0):
                    # Note: capital_expenditure is typically negative in financial statements
                    capex_ratio = abs(item.get('capital_expenditure')) / item.get('revenue')
                    capex_to_revenue.append(capex_ratio)
            
            if capex_to_revenue:
                avg_capex_ratio = sum(capex_to_revenue) / len(capex_to_revenue)
                if avg_capex_ratio < 0.05:  # Less than 5% of revenue
                    score += 2
                    details.append(f"Low capital requirements: Avg capex {avg_capex_ratio:.1%} of revenue")
                elif avg_capex_ratio < 0.10:  # Less than 10% of revenue
                    score += 1
                    details.append(f"Moderate capital requirements: Avg capex {avg_capex_ratio:.1%} of revenue")
                else:
                    details.append(f"High capital requirements: Avg capex {avg_capex_ratio:.1%} of revenue")
            else:
                details.append("No capital expenditure data available")
        else:
            details.append("Insufficient data for capital intensity analysis")
        
        # 4. Intangible assets - Munger values R&D and intellectual property
        r_and_d = [item.get('research_and_development') for item in metrics
                  if item.get('research_and_development') is not None]
        
        goodwill_and_intangible_assets = [item.get('goodwill_and_intangible_assets') for item in metrics
                   if item.get('goodwill_and_intangible_assets') is not None]

        if r_and_d and len(r_and_d) > 0:
            if sum(r_and_d) > 0:  # If company is investing in R&D
                score += 1
                details.append("Invests in R&D, building intellectual property")
        
        if (goodwill_and_intangible_assets and len(goodwill_and_intangible_assets) > 0):
            score += 1
            details.append("Significant goodwill/intangible assets, suggesting brand value or IP")
        
        # Scale score to 0-10 range
        final_score = min(10, score * 10 / 9)  # Max possible raw score is 9
        
        result["score"] = final_score
        result["details"] = details
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
        analysis['type'] = 'moat_strength_analysis'
        analysis['title'] = f'Moat strength analysis'

        analysis_data['moat_strength_analysis'] = analysis
        ai_message = AIMessage(content=self.get_markdown(analysis))
        return {
            "context": context,
            "messages": [
                ai_message
            ]
        }