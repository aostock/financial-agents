from common.agent_state import AgentState
from langchain.schema import AIMessage
from langchain_core.callbacks import dispatch_custom_event
from langchain_core.runnables import RunnableConfig
from typing import Dict, Any

import time
from common import markdown
from langgraph.types import StreamWriter
import statistics


class GrowthAnalysis():
    def __init__(self, options: Dict[str, Any]):
        self.options = options

    
    def analyze(self, metrics: list) -> dict[str, any]:
        """Analyze company growth based on Rakesh Jhunjhunwala's criteria."""
        result = {"score": 0, "max_score": 10, "details": []}
        if not metrics or len(metrics) < 3:
            result["details"].append('Insufficient metrics for growth analysis (need at least 3 years)')
            return result

        # Get last 5 years of data if available, otherwise use what we have
        years_to_analyze = min(5, len(metrics))
        recent_metrics = metrics[:years_to_analyze]
        
        # Extract revenue and net income data
        revenues = []
        net_incomes = []
        years = []
        
        for metric in recent_metrics:
            if metric.get('revenue') and metric.get('net_income'):
                revenues.append(metric['revenue'])
                net_incomes.append(metric['net_income'])
                years.append(metric.get('date', 'N/A'))

        if len(revenues) < 3:
            result["details"].append('Insufficient revenue/net income data for growth analysis')
            return result

        score = 0
        reasoning = []

        # Calculate CAGR for revenue
        try:
            revenue_cagr = self.calculate_cagr(revenues[0], revenues[-1], len(revenues)-1)
            if revenue_cagr > 0.20:  # >20% excellent
                score += 3
                reasoning.append(f"Excellent revenue CAGR of {revenue_cagr:.1%} (target: >20%)")
            elif revenue_cagr > 0.15:  # >15% good
                score += 2
                reasoning.append(f"Good revenue CAGR of {revenue_cagr:.1%} (target: >15%)")
            elif revenue_cagr > 0.10:  # >10% moderate
                score += 1
                reasoning.append(f"Moderate revenue CAGR of {revenue_cagr:.1%} (target: >10%)")
            else:
                reasoning.append(f"Low revenue CAGR of {revenue_cagr:.1%}")
        except:
            reasoning.append("Could not calculate revenue CAGR")

        # Calculate CAGR for net income
        try:
            net_income_cagr = self.calculate_cagr(net_incomes[0], net_incomes[-1], len(net_incomes)-1)
            if net_income_cagr > 0.25:  # >25% very high
                score += 3
                reasoning.append(f"Very high net income CAGR of {net_income_cagr:.1%} (target: >25%)")
            elif net_income_cagr > 0.20:  # >20% high
                score += 2
                reasoning.append(f"High net income CAGR of {net_income_cagr:.1%} (target: >20%)")
            elif net_income_cagr > 0.15:  # >15% good
                score += 1
                reasoning.append(f"Good net income CAGR of {net_income_cagr:.1%} (target: >15%)")
            else:
                reasoning.append(f"Low net income CAGR of {net_income_cagr:.1%}")
        except:
            reasoning.append("Could not calculate net income CAGR")

        # Check growth consistency - percentage of years with positive growth
        try:
            revenue_growth_rates = []
            net_income_growth_rates = []
            
            for i in range(len(revenues)-1):
                if revenues[i+1] != 0:
                    revenue_growth = (revenues[i] - revenues[i+1]) / revenues[i+1]
                    revenue_growth_rates.append(revenue_growth)
                
                if net_incomes[i+1] != 0:
                    net_income_growth = (net_incomes[i] - net_incomes[i+1]) / net_incomes[i+1]
                    net_income_growth_rates.append(net_income_growth)
            
            # Check consistency (at least 80% of years with positive growth)
            if revenue_growth_rates:
                positive_revenue_years = sum(1 for g in revenue_growth_rates if g > 0)
                revenue_consistency = positive_revenue_years / len(revenue_growth_rates)
                if revenue_consistency >= 0.8:
                    score += 2
                    reasoning.append(f"Strong revenue growth consistency ({revenue_consistency:.0%} positive years)")
                elif revenue_consistency >= 0.6:
                    score += 1
                    reasoning.append(f"Moderate revenue growth consistency ({revenue_consistency:.0%} positive years)")
                else:
                    reasoning.append(f"Low revenue growth consistency ({revenue_consistency:.0%} positive years)")
            
            if net_income_growth_rates:
                positive_net_income_years = sum(1 for g in net_income_growth_rates if g > 0)
                net_income_consistency = positive_net_income_years / len(net_income_growth_rates)
                if net_income_consistency >= 0.8:
                    score += 2
                    reasoning.append(f"Strong net income growth consistency ({net_income_consistency:.0%} positive years)")
                elif net_income_consistency >= 0.6:
                    score += 1
                    reasoning.append(f"Moderate net income growth consistency ({net_income_consistency:.0%} positive years)")
                else:
                    reasoning.append(f"Low net income growth consistency ({net_income_consistency:.0%} positive years)")
        except:
            reasoning.append("Could not calculate growth consistency")

        result["score"] = score
        result["details"] = reasoning
        return result

    def calculate_cagr(self, ending_value, beginning_value, years):
        """Calculate Compound Annual Growth Rate"""
        if beginning_value <= 0 or years <= 0:
            return 0
        return (ending_value / beginning_value) ** (1 / years) - 1

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
        analysis['type'] = 'growth_analysis'
        analysis['title'] = f'Growth analysis'

        analysis_data['growth_analysis'] = analysis
        ai_message = AIMessage(content=self.get_markdown(analysis))
        return {
            "context": context,
            "messages": [
                ai_message
            ]
        }