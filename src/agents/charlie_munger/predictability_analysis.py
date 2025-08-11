from common.agent_state import AgentState
from langchain.schema import AIMessage
from langchain_core.callbacks import dispatch_custom_event
from langchain_core.runnables import RunnableConfig
from typing import Dict, Any

import time
from common import markdown
from langgraph.types import StreamWriter


class PredictabilityAnalysis():
    def __init__(self, options: Dict[str, Any]):
        self.options = options

    
    def analyze(self, metrics: list) -> dict[str, any]:
        """
        Assess the predictability of the business - Munger strongly prefers businesses
        whose future operations and cashflows are relatively easy to predict.
        """
        result = {"score": 0, "max_score": 10, "details": []}
        if not metrics or len(metrics) < 5:
            result["details"].append('Insufficient data to analyze business predictability (need 5+ years)')
            return result

        score = 0
        details = []
        
        # 1. Revenue stability and growth
        revenues = [item.get('revenue') for item in metrics 
                   if item.get('revenue') is not None]
        
        if revenues and len(revenues) >= 5:
            # Calculate year-over-year growth rates, handling zero division
            growth_rates = []
            for i in range(len(revenues)-1):
                if revenues[i+1] != 0:  # Avoid division by zero
                    growth_rate = (revenues[i] / revenues[i+1] - 1)
                    growth_rates.append(growth_rate)
            
            if not growth_rates:
                details.append("Cannot calculate revenue growth: zero revenue values found")
            else:
                avg_growth = sum(growth_rates) / len(growth_rates)
                growth_volatility = sum(abs(r - avg_growth) for r in growth_rates) / len(growth_rates)
                
                if avg_growth > 0.05 and growth_volatility < 0.1:
                    # Steady, consistent growth (Munger loves this)
                    score += 3
                    details.append(f"Highly predictable revenue: {avg_growth:.1%} avg growth with low volatility")
                elif avg_growth > 0 and growth_volatility < 0.2:
                    # Positive but somewhat volatile growth
                    score += 2
                    details.append(f"Moderately predictable revenue: {avg_growth:.1%} avg growth with some volatility")
                elif avg_growth > 0:
                    # Growing but unpredictable
                    score += 1
                    details.append(f"Growing but less predictable revenue: {avg_growth:.1%} avg growth with high volatility")
                else:
                    details.append(f"Declining or highly unpredictable revenue: {avg_growth:.1%} avg growth")
        else:
            details.append("Insufficient revenue history for predictability analysis")
        
        # 2. Operating income stability
        op_income = [item.get('operating_income') for item in metrics 
                    if item.get('operating_income') is not None]
        
        if op_income and len(op_income) >= 5:
            # Count positive operating income periods
            positive_periods = sum(1 for income in op_income if income > 0)
            
            if positive_periods == len(op_income):
                # Consistently profitable operations
                score += 3
                details.append("Highly predictable operations: Operating income positive in all periods")
            elif positive_periods >= len(op_income) * 0.8:
                # Mostly profitable operations
                score += 2
                details.append(f"Predictable operations: Operating income positive in {positive_periods}/{len(op_income)} periods")
            elif positive_periods >= len(op_income) * 0.6:
                # Somewhat profitable operations
                score += 1
                details.append(f"Somewhat predictable operations: Operating income positive in {positive_periods}/{len(op_income)} periods")
            else:
                details.append(f"Unpredictable operations: Operating income positive in only {positive_periods}/{len(op_income)} periods")
        else:
            details.append("Insufficient operating income history")
        
        # 3. Margin consistency - Munger values stable margins
        op_margins = [item.get('operating_margin') for item in metrics 
                     if item.get('operating_margin') is not None]
        
        if op_margins and len(op_margins) >= 5:
            # Calculate margin volatility
            avg_margin = sum(op_margins) / len(op_margins)
            margin_volatility = sum(abs(m - avg_margin) for m in op_margins) / len(op_margins)
            
            if margin_volatility < 0.03:  # Very stable margins
                score += 2
                details.append(f"Highly predictable margins: {avg_margin:.1%} avg with minimal volatility")
            elif margin_volatility < 0.07:  # Moderately stable margins
                score += 1
                details.append(f"Moderately predictable margins: {avg_margin:.1%} avg with some volatility")
            else:
                details.append(f"Unpredictable margins: {avg_margin:.1%} avg with high volatility ({margin_volatility:.1%})")
        else:
            details.append("Insufficient margin history")
        
        # 4. Cash generation reliability
        fcf_values = [item.get('free_cash_flow') for item in metrics 
                     if item.get('free_cash_flow') is not None]
        
        if fcf_values and len(fcf_values) >= 5:
            # Count positive FCF periods
            positive_fcf_periods = sum(1 for fcf in fcf_values if fcf > 0)
            
            if positive_fcf_periods == len(fcf_values):
                # Consistently positive FCF
                score += 2
                details.append("Highly predictable cash generation: Positive FCF in all periods")
            elif positive_fcf_periods >= len(fcf_values) * 0.8:
                # Mostly positive FCF
                score += 1
                details.append(f"Predictable cash generation: Positive FCF in {positive_fcf_periods}/{len(fcf_values)} periods")
            else:
                details.append(f"Unpredictable cash generation: Positive FCF in only {positive_fcf_periods}/{len(fcf_values)} periods")
        else:
            details.append("Insufficient free cash flow history")
        
        # Scale score to 0-10 range
        # Maximum possible raw score would be 10 (3+3+2+2)
        final_score = min(10, score * 10 / 10)
        
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
        analysis['type'] = 'predictability_analysis'
        analysis['title'] = f'Predictability analysis'

        analysis_data['predictability_analysis'] = analysis
        ai_message = AIMessage(content=self.get_markdown(analysis))
        return {
            "context": context,
            "messages": [
                ai_message
            ]
        }