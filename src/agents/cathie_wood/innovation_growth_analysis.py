from common.agent_state import AgentState
from langchain.schema import AIMessage
from langchain_core.callbacks import dispatch_custom_event
from langchain_core.runnables import RunnableConfig
from typing import Dict, Any

import time
from common import markdown
from langgraph.types import StreamWriter


class InnovationGrowthAnalysis():
    def __init__(self, options: Dict[str, Any]):
        self.options = options

    
    def analyze(self, metrics: list) -> dict[str, any]:
        """
        Evaluate the company's commitment to innovation and potential for exponential growth.
        Analyzes multiple dimensions:
        1. R&D Investment Trends - measures commitment to innovation
        2. Free Cash Flow Generation - indicates ability to fund innovation
        3. Operating Efficiency - shows scalability of innovation
        4. Capital Allocation - reveals innovation-focused management
        5. Growth Reinvestment - demonstrates commitment to future growth
        """
        result = {"score": 0, "max_score": 5, "details": []}
        if not metrics:
            result["details"].append('No metrics available')
            return result

        score = 0
        details = []

        # 1. R&D Investment Trends
        rd_expenses = [item.get('research_and_development') for item in metrics if item.get('research_and_development')]
        revenues = [item.get('revenue') for item in metrics if item.get('revenue')]

        if rd_expenses and revenues and len(rd_expenses) >= 2:
            rd_growth = (rd_expenses[0] - rd_expenses[-1]) / abs(rd_expenses[-1]) if rd_expenses[-1] != 0 else 0
            if rd_growth > 0.5:  # 50% growth in R&D
                score += 3
                details.append(f"Strong R&D investment growth: +{(rd_growth*100):.1f}%")
            elif rd_growth > 0.2:
                score += 2
                details.append(f"Moderate R&D investment growth: +{(rd_growth*100):.1f}%")

            # Check R&D intensity trend (corrected for reverse chronological order)
            rd_intensity_start = rd_expenses[-1] / revenues[-1]
            rd_intensity_end = rd_expenses[0] / revenues[0]
            if rd_intensity_end > rd_intensity_start:
                score += 2
                details.append(f"Increasing R&D intensity: {(rd_intensity_end*100):.1f}% vs {(rd_intensity_start*100):.1f}%")
        else:
            details.append("Insufficient R&D data for trend analysis")

        # 2. Free Cash Flow Analysis
        fcf_vals = [item.get('free_cash_flow') for item in metrics if item.get('free_cash_flow')]
        if fcf_vals and len(fcf_vals) >= 2:
            fcf_growth = (fcf_vals[0] - fcf_vals[-1]) / abs(fcf_vals[-1])
            positive_fcf_count = sum(1 for f in fcf_vals if f > 0)

            if fcf_growth > 0.3 and positive_fcf_count == len(fcf_vals):
                score += 3
                details.append("Strong and consistent FCF growth, excellent innovation funding capacity")
            elif positive_fcf_count >= len(fcf_vals) * 0.75:
                score += 2
                details.append("Consistent positive FCF, good innovation funding capacity")
            elif positive_fcf_count > len(fcf_vals) * 0.5:
                score += 1
                details.append("Moderately consistent FCF, adequate innovation funding capacity")
        else:
            details.append("Insufficient FCF data for analysis")

        # 3. Operating Efficiency Analysis
        op_margin_vals = [item.get('operating_margin') for item in metrics if item.get('operating_margin')]
        if op_margin_vals and len(op_margin_vals) >= 2:
            margin_trend = op_margin_vals[0] - op_margin_vals[-1]

            if op_margin_vals[0] > 0.15 and margin_trend > 0:
                score += 3
                details.append(f"Strong and improving operating margin: {(op_margin_vals[0]*100):.1f}%")
            elif op_margin_vals[0] > 0.10:
                score += 2
                details.append(f"Healthy operating margin: {(op_margin_vals[0]*100):.1f}%")
            elif margin_trend > 0:
                score += 1
                details.append("Improving operating efficiency")
        else:
            details.append("Insufficient operating margin data")

        # 4. Capital Allocation Analysis
        capex = [item.get('capital_expenditure') for item in metrics if item.get('capital_expenditure')]
        if capex and revenues and len(capex) >= 2:
            capex_intensity = abs(capex[0]) / revenues[0]
            capex_growth = (abs(capex[0]) - abs(capex[-1])) / abs(capex[-1]) if capex[-1] != 0 else 0

            if capex_intensity > 0.10 and capex_growth > 0.2:
                score += 2
                details.append("Strong investment in growth infrastructure")
            elif capex_intensity > 0.05:
                score += 1
                details.append("Moderate investment in growth infrastructure")
        else:
            details.append("Insufficient CAPEX data")

        # 5. Growth Reinvestment Analysis
        dividends = [item.get('dividends_and_other_cash_distributions') for item in metrics if item.get('dividends_and_other_cash_distributions')]
        if dividends and fcf_vals:
            latest_payout_ratio = dividends[0] / fcf_vals[0] if fcf_vals[0] != 0 else 1
            if latest_payout_ratio < 0.2:  # Low dividend payout ratio suggests reinvestment focus
                score += 2
                details.append("Strong focus on reinvestment over dividends")
            elif latest_payout_ratio < 0.4:
                score += 1
                details.append("Moderate focus on reinvestment over dividends")
        else:
            details.append("Insufficient dividend data")

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
        analysis['type'] = 'innovation_growth_analysis'
        analysis['title'] = f'Innovation growth analysis'

        analysis_data['innovation_growth_analysis'] = analysis
        ai_message = AIMessage(content=self.get_markdown(analysis))
        return {
            "context": context,
            "messages": [
                ai_message
            ]
        }