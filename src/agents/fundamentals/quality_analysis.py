from common.agent_state import AgentState
from langchain.schema import AIMessage
from langchain_core.callbacks import dispatch_custom_event
from langchain_core.runnables import RunnableConfig
from typing import Dict, Any

import time
from common import markdown
from langgraph.types import StreamWriter


class QualityAnalysis():
    def __init__(self, options: Dict[str, Any]):
        self.options = options

    
    def analyze(self, metrics: list) -> dict[str, any]:
        """Analyze business quality based on returns on capital and efficiency metrics."""
        result = {"score": 0, "max_score": 10, "details": []}
        if not metrics:
            result["details"].append('No metrics available')
            return result

        latest_metrics = metrics[0]

        score = 0
        reasoning = []

        # Check ROIC (Return on Invested Capital)
        if latest_metrics.get('return_on_invested_capital') and latest_metrics['return_on_invested_capital'] > 0.12:  # 12% ROIC threshold
            score += 4
            reasoning.append(f"Exceptional ROIC of {latest_metrics['return_on_invested_capital']:.1%} (>12%)")
        elif latest_metrics.get('return_on_invested_capital') and latest_metrics['return_on_invested_capital'] > 0.10:
            score += 3
            reasoning.append(f"Strong ROIC of {latest_metrics['return_on_invested_capital']:.1%} (>10%)")
        elif latest_metrics.get('return_on_invested_capital') and latest_metrics['return_on_invested_capital'] > 0.08:
            score += 2
            reasoning.append(f"Good ROIC of {latest_metrics['return_on_invested_capital']:.1%} (>8%)")
        elif latest_metrics.get('return_on_invested_capital'):
            reasoning.append(f"Weak ROIC of {latest_metrics['return_on_invested_capital']:.1%} (<8%)")
        else:
            reasoning.append(f"ROIC data not available")

        # Check Asset Turnover
        if latest_metrics.get('asset_turnover') and latest_metrics['asset_turnover'] > 0.8:
            score += 2
            reasoning.append(f"Efficient asset utilization ({latest_metrics['asset_turnover']:.2f} > 0.8)")
        elif latest_metrics.get('asset_turnover') and latest_metrics['asset_turnover'] > 0.5:
            score += 1
            reasoning.append(f"Reasonable asset turnover ({latest_metrics['asset_turnover']:.2f} > 0.5)")
        elif latest_metrics.get('asset_turnover'):
            reasoning.append(f"Low asset turnover ({latest_metrics['asset_turnover']:.2f} < 0.5)")
        else:
            reasoning.append(f"Asset turnover data not available")

        # Check Free Cash Flow Generation
        if latest_metrics.get('free_cash_flow') and latest_metrics.get('net_income'):
            fcf_ratio = latest_metrics['free_cash_flow'] / latest_metrics['net_income'] if latest_metrics['net_income'] != 0 else 0
            if fcf_ratio > 0.9:
                score += 2
                reasoning.append(f"Excellent FCF conversion ({fcf_ratio:.1%} of net income)")
            elif fcf_ratio > 0.7:
                score += 1
                reasoning.append(f"Good FCF conversion ({fcf_ratio:.1%} of net income)")
            elif latest_metrics['free_cash_flow'] > 0:
                reasoning.append(f"Positive but limited FCF conversion ({fcf_ratio:.1%} of net income)")
            else:
                reasoning.append(f"Negative free cash flow")

        # Check Capital Efficiency (ROIC vs. Cost of Capital proxy)
        if latest_metrics.get('return_on_invested_capital') and latest_metrics.get('beta'):
            # Simple cost of capital proxy: 2% risk-free + beta * 5% equity risk premium
            cost_of_capital = 0.02 + latest_metrics['beta'] * 0.05 if latest_metrics['beta'] < 2 else 0.12
            if latest_metrics['return_on_invested_capital'] > cost_of_capital + 0.03:  # 3% margin
                score += 2
                reasoning.append(f"Strong economic profit generation (ROIC {latest_metrics['return_on_invested_capital']:.1%} vs. cost of capital ~{cost_of_capital:.1%})")
            elif latest_metrics['return_on_invested_capital'] > cost_of_capital:
                score += 1
                reasoning.append(f"Positive economic profit (ROIC {latest_metrics['return_on_invested_capital']:.1%} vs. cost of capital ~{cost_of_capital:.1%})")
            else:
                reasoning.append(f"Destroying economic value (ROIC {latest_metrics['return_on_invested_capital']:.1%} vs. cost of capital ~{cost_of_capital:.1%})")

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
        analysis['type'] = 'quality_analysis'
        analysis['title'] = f'Business Quality Analysis'

        analysis_data['quality_analysis'] = analysis
        ai_message = AIMessage(content=self.get_markdown(analysis))
        return {
            "context": context,
            "messages": [
                ai_message
            ]
        }