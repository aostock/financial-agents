from common.agent_state import AgentState
from langchain.schema import AIMessage
from langchain_core.callbacks import dispatch_custom_event
from langchain_core.runnables import RunnableConfig
from typing import Dict, Any

import time
from common import markdown
from langgraph.types import StreamWriter


class BusinessQualityAnalysis():
    def __init__(self, options: Dict[str, Any]):
        self.options = options

    
    def analyze(self, metrics: list) -> dict[str, any]:
        """
        Analyze whether the company has a high-quality business with stable or growing cash flows,
        durable competitive advantages (moats), and potential for long-term growth.
        """
        result = {"score": 0, "max_score": 5, "details": []}
        if not metrics:
            result["details"].append('No metrics available')
            return result

        score = 0
        details = []

        # 1. Multi-period revenue growth analysis
        revenues = [item.get('revenue') for item in metrics if item.get('revenue') is not None]
        if len(revenues) >= 2:
            initial, final = revenues[-1], revenues[0]
            if initial and final and final > initial:
                growth_rate = (final - initial) / abs(initial)
                if growth_rate > 0.5:  # e.g., 50% cumulative growth
                    score += 2
                    details.append(f"Revenue grew by {(growth_rate*100):.1f}% over the full period (strong growth).")
                else:
                    score += 1
                    details.append(f"Revenue growth is positive but under 50% cumulatively ({(growth_rate*100):.1f}%).")
            else:
                details.append("Revenue did not grow significantly or data insufficient.")
        else:
            details.append("Not enough revenue data for multi-period trend.")

        # 2. Operating margin and free cash flow consistency
        fcf_vals = [item.get('free_cash_flow') for item in metrics if item.get('free_cash_flow') is not None]
        op_margin_vals = [item.get('operating_margin') for item in metrics if item.get('operating_margin') is not None]

        if op_margin_vals:
            above_15 = sum(1 for m in op_margin_vals if m > 0.15)
            if above_15 >= (len(op_margin_vals) // 2 + 1):
                score += 1
                details.append("Operating margins have often exceeded 15% (indicates good profitability).")
            else:
                details.append("Operating margin not consistently above 15%.")
        else:
            details.append("No operating margin data across periods.")

        if fcf_vals:
            positive_fcf_count = sum(1 for f in fcf_vals if f > 0)
            if positive_fcf_count >= (len(fcf_vals) // 2 + 1):
                score += 1
                details.append("Majority of periods show positive free cash flow.")
            else:
                details.append("Free cash flow not consistently positive.")
        else:
            details.append("No free cash flow data across periods.")

        # 3. Return on Equity (ROE) check from the latest metrics
        latest_metrics = metrics[0] if metrics else None
        if latest_metrics and latest_metrics.get('return_on_equity') and latest_metrics['return_on_equity'] > 0.15:
            score += 1
            details.append(f"High ROE of {latest_metrics['return_on_equity']:.1%}, indicating a competitive advantage.")
        elif latest_metrics and latest_metrics.get('return_on_equity'):
            details.append(f"ROE of {latest_metrics['return_on_equity']:.1%} is moderate.")
        else:
            details.append("ROE data not available.")

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
        analysis['type'] = 'business_quality_analysis'
        analysis['title'] = f'Business quality analysis'

        analysis_data['business_quality_analysis'] = analysis
        ai_message = AIMessage(content=self.get_markdown(analysis))
        return {
            "context": context,
            "messages": [
                ai_message
            ]
        }