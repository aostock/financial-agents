from common.agent_state import AgentState
from langchain.schema import AIMessage
from langchain_core.callbacks import dispatch_custom_event
from langchain_core.runnables import RunnableConfig
from typing import Dict, Any

import time
from common import markdown
from langgraph.types import StreamWriter


class DisruptivePotentialAnalysis():
    def __init__(self, options: Dict[str, Any]):
        self.options = options

    
    def analyze(self, metrics: list) -> dict[str, any]:
        """
        Analyze whether the company has disruptive products, technology, or business model.
        Evaluates multiple dimensions of disruptive potential:
        1. Revenue Growth Acceleration - indicates market adoption
        2. R&D Intensity - shows innovation investment
        3. Gross Margin Trends - suggests pricing power and scalability
        4. Operating Leverage - demonstrates business model efficiency
        """
        result = {"score": 0, "max_score": 5, "details": []}
        if not metrics:
            result["details"].append('No metrics available')
            return result

        score = 0
        details = []

        # 1. Revenue Growth Analysis - Check for accelerating growth
        revenues = [item.get('revenue') for item in metrics if item.get('revenue')]
        if len(revenues) >= 3:  # Need at least 3 periods to check acceleration
            growth_rates = []
            for i in range(len(revenues) - 1):
                if revenues[i] and revenues[i + 1]:
                    growth_rate = (revenues[i] - revenues[i + 1]) / abs(revenues[i + 1]) if revenues[i + 1] != 0 else 0
                    growth_rates.append(growth_rate)

            # Check if growth is accelerating (first growth rate higher than last, since they're in reverse order)
            if len(growth_rates) >= 2 and growth_rates[0] > growth_rates[-1]:
                score += 2
                details.append(f"Revenue growth is accelerating: {(growth_rates[0]*100):.1f}% vs {(growth_rates[-1]*100):.1f}%")

            # Check absolute growth rate (most recent growth rate is at index 0)
            latest_growth = growth_rates[0] if growth_rates else 0
            if latest_growth > 1.0:
                score += 3
                details.append(f"Exceptional revenue growth: {(latest_growth*100):.1f}%")
            elif latest_growth > 0.5:
                score += 2
                details.append(f"Strong revenue growth: {(latest_growth*100):.1f}%")
            elif latest_growth > 0.2:
                score += 1
                details.append(f"Moderate revenue growth: {(latest_growth*100):.1f}%")
        else:
            details.append("Insufficient revenue data for growth analysis")

        # 2. Gross Margin Analysis - Check for expanding margins
        gross_margins = [item.get('gross_margin') for item in metrics if item.get('gross_margin') is not None]
        if len(gross_margins) >= 2:
            margin_trend = gross_margins[0] - gross_margins[-1]
            if margin_trend > 0.05:  # 5% improvement
                score += 2
                details.append(f"Expanding gross margins: +{(margin_trend*100):.1f}%")
            elif margin_trend > 0:
                score += 1
                details.append(f"Slightly improving gross margins: +{(margin_trend*100):.1f}%")

            # Check absolute margin level (most recent margin is at index 0)
            if gross_margins[0] > 0.50:  # High margin business
                score += 2
                details.append(f"High gross margin: {(gross_margins[0]*100):.1f}%")
        else:
            details.append("Insufficient gross margin data")

        # 3. Operating Leverage Analysis
        revenues = [item.get('revenue') for item in metrics if item.get('revenue')]
        operating_expenses = [item.get('operating_expense') for item in metrics if item.get('operating_expense')]

        if len(revenues) >= 2 and len(operating_expenses) >= 2:
            rev_growth = (revenues[0] - revenues[-1]) / abs(revenues[-1])
            opex_growth = (operating_expenses[0] - operating_expenses[-1]) / abs(operating_expenses[-1])

            if rev_growth > opex_growth:
                score += 2
                details.append("Positive operating leverage: Revenue growing faster than expenses")
        else:
            details.append("Insufficient data for operating leverage analysis")

        # 4. R&D Investment Analysis
        rd_expenses = [item.get('research_and_development') for item in metrics if item.get('research_and_development') is not None]
        if rd_expenses and revenues:
            rd_intensity = rd_expenses[0] / revenues[0]
            if rd_intensity > 0.15:  # High R&D intensity
                score += 3
                details.append(f"High R&D investment: {(rd_intensity*100):.1f}% of revenue")
            elif rd_intensity > 0.08:
                score += 2
                details.append(f"Moderate R&D investment: {(rd_intensity*100):.1f}% of revenue")
            elif rd_intensity > 0.05:
                score += 1
                details.append(f"Some R&D investment: {(rd_intensity*100):.1f}% of revenue")
        else:
            details.append("No R&D data available")

        result["score"] = score
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
        analysis['type'] = 'disruptive_potential_analysis'
        analysis['title'] = f'Disruptive potential analysis'

        analysis_data['disruptive_potential_analysis'] = analysis
        ai_message = AIMessage(content=self.get_markdown(analysis))
        return {
            "context": context,
            "messages": [
                ai_message
            ]
        }