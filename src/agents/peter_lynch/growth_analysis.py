from common.agent_state import AgentState
from langchain.schema import AIMessage
from langchain_core.callbacks import dispatch_custom_event
from langchain_core.runnables import RunnableConfig
from typing import Dict, Any

import time
from common import markdown
from langgraph.types import StreamWriter


class GrowthAnalysis():
    def __init__(self, options: Dict[str, Any]):
        self.options = options

    
    def analyze(self, metrics: list) -> dict[str, any]:
        """Analyze company growth based on Peter Lynch's criteria."""
        result = {"score": 0, "max_score": 10, "details": []}
        if not metrics or len(metrics) < 3:
            result["details"].append('Insufficient historical data (need at least 3 years)')
            return result

        latest_metrics = metrics[0]
        previous_metrics = metrics[1] if len(metrics) > 1 else None
        older_metrics = metrics[2] if len(metrics) > 2 else None

        score = 0
        reasoning = []

        # Calculate earnings growth rate from net income data
        earnings_growth = None
        if (latest_metrics.get('net_income') and previous_metrics and 
            previous_metrics.get('net_income') and previous_metrics.get('net_income') > 0):
            earnings_growth = (latest_metrics.get('net_income') - previous_metrics.get('net_income')) / previous_metrics.get('net_income')
        
        if earnings_growth and earnings_growth > 0.25:
            score += 3
            reasoning.append(f"Exceptional earnings growth ({earnings_growth:.1%}) - Fast grower")
        elif earnings_growth and earnings_growth > 0.15:
            score += 2
            reasoning.append(f"Strong earnings growth ({earnings_growth:.1%}) - Stalwart")
        elif earnings_growth and earnings_growth > 0.10:
            score += 1
            reasoning.append(f"Good earnings growth ({earnings_growth:.1%})")
        elif earnings_growth is not None:
            reasoning.append(f"Slow earnings growth ({earnings_growth:.1%}) - Slow grower")
        else:
            reasoning.append(f"Earnings growth data not available")

        # Check revenue growth consistency over multiple years
        revenues = [m.get('revenue') for m in metrics[:5] if m.get('revenue') is not None]
        if len(revenues) >= 3:
            growth_rates = []
            for i in range(1, len(revenues)):
                if revenues[i-1] and revenues[i-1] > 0:
                    growth_rate = (revenues[i] - revenues[i-1]) / revenues[i-1]
                    growth_rates.append(growth_rate)
            
            if growth_rates:
                avg_growth = sum(growth_rates) / len(growth_rates)
                if avg_growth > 0.15:
                    score += 2
                    reasoning.append(f"Consistent revenue growth ({avg_growth:.1%} avg)")
                elif avg_growth > 0.10:
                    score += 1
                    reasoning.append(f"Moderate revenue growth ({avg_growth:.1%} avg)")
                else:
                    reasoning.append(f"Slow revenue growth ({avg_growth:.1%} avg)")
            else:
                reasoning.append("Unable to calculate revenue growth rates")
        else:
            reasoning.append("Insufficient revenue data for growth analysis")

        # Check net income growth consistency
        incomes = [m.get('net_income') for m in metrics[:5] if m.get('net_income') is not None]
        if len(incomes) >= 3:
            growth_rates = []
            for i in range(1, len(incomes)):
                if incomes[i-1] and incomes[i-1] > 0:
                    growth_rate = (incomes[i] - incomes[i-1]) / incomes[i-1]
                    growth_rates.append(growth_rate)
            
            if growth_rates:
                avg_growth = sum(growth_rates) / len(growth_rates)
                if avg_growth > 0.15:
                    score += 2
                    reasoning.append(f"Strong earnings consistency ({avg_growth:.1%} avg)")
                elif avg_growth > 0.10:
                    score += 1
                    reasoning.append(f"Good earnings consistency ({avg_growth:.1%} avg)")
                else:
                    reasoning.append(f"Poor earnings consistency ({avg_growth:.1%} avg)")
            else:
                reasoning.append("Unable to calculate earnings growth rates")
        else:
            reasoning.append("Insufficient earnings data for growth analysis")

        # Check if growth is accelerating or decelerating
        if len(metrics) >= 3:
            recent_growth = None
            older_growth = None
            
            # Calculate recent growth (year over year)
            if (latest_metrics.get('net_income') and previous_metrics and 
                previous_metrics.get('net_income') and previous_metrics.get('net_income') > 0):
                recent_growth = (latest_metrics.get('net_income') - previous_metrics.get('net_income')) / previous_metrics.get('net_income')
            
            # Calculate older growth (year over year)
            if (previous_metrics and previous_metrics.get('net_income') and older_metrics and 
                older_metrics.get('net_income') and older_metrics.get('net_income') > 0):
                older_growth = (previous_metrics.get('net_income') - older_metrics.get('net_income')) / older_metrics.get('net_income')
            
            if recent_growth is not None and older_growth is not None:
                if recent_growth > older_growth and recent_growth > 0.10:
                    score += 1
                    reasoning.append("Earnings growth accelerating")
                elif recent_growth < older_growth:
                    reasoning.append("Earnings growth decelerating")
                else:
                    reasoning.append("Stable earnings growth trend")
            else:
                reasoning.append("Insufficient data for growth trend analysis")

        # Check free cash flow growth
        fcf_values = [m.get('free_cash_flow') for m in metrics[:3] if m.get('free_cash_flow') is not None]
        if len(fcf_values) >= 2:
            if fcf_values[0] and fcf_values[0] > 0 and len(fcf_values) > 1 and fcf_values[1] and fcf_values[1] > 0:
                fcf_growth = (fcf_values[0] - fcf_values[1]) / fcf_values[1]
                if fcf_growth > 0.20:
                    score += 1
                    reasoning.append(f"Strong FCF growth ({fcf_growth:.1%})")
                elif fcf_growth > 0.10:
                    score += 1
                    reasoning.append(f"Good FCF growth ({fcf_growth:.1%})")
                elif fcf_growth > 0:
                    reasoning.append(f"Modest FCF growth ({fcf_growth:.1%})")
                else:
                    reasoning.append(f"Declining FCF ({fcf_growth:.1%})")
            else:
                reasoning.append("Negative or insufficient FCF data")
        else:
            reasoning.append("Insufficient FCF data for growth analysis")

        result["score"] = score
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
        analysis['type'] = 'growth_analysis'
        analysis['title'] = f'Growth Analysis'

        analysis_data['growth_analysis'] = analysis
        ai_message = AIMessage(content=self.get_markdown(analysis))
        return {
            "context": context,
            "messages": [
                ai_message
            ]
        }