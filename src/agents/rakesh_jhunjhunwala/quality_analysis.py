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
        """Analyze company quality based on Rakesh Jhunjhunwala's criteria."""
        result = {"score": 0, "max_score": 10, "details": []}
        if not metrics:
            result["details"].append('No metrics available')
            return result

        latest_metrics = metrics[0]

        score = 0
        reasoning = []

        # Check ROE consistency over time (at least 3 years)
        if len(metrics) >= 3:
            roe_values = []
            for metric in metrics[:min(5, len(metrics))]:  # Check up to 5 years
                if metric.get('return_on_equity'):
                    roe_values.append(metric['return_on_equity'])
            
            if len(roe_values) >= 3:
                # Check if ROE has been consistently high
                high_roe_years = sum(1 for roe in roe_values if roe > 0.15)
                roe_consistency = high_roe_years / len(roe_values)
                
                if roe_consistency >= 0.8:  # 80%+ years with high ROE
                    score += 3
                    reasoning.append(f"Excellent ROE consistency ({roe_consistency:.0%} years with ROE > 15%)")
                elif roe_consistency >= 0.6:  # 60%+ years with high ROE
                    score += 2
                    reasoning.append(f"Good ROE consistency ({roe_consistency:.0%} years with ROE > 15%)")
                else:
                    reasoning.append(f"Moderate ROE consistency ({roe_consistency:.0%} years with ROE > 15%)")
            else:
                reasoning.append("Insufficient ROE data for consistency analysis")
        else:
            reasoning.append("Need at least 3 years of data for quality analysis")

        # Check debt management - debt to asset ratio
        if latest_metrics.get('total_liabilities') and latest_metrics.get('total_assets'):
            debt_to_asset = latest_metrics['total_liabilities'] / latest_metrics['total_assets']
            if debt_to_asset < 0.3:  # Low debt to asset ratio
                score += 2
                reasoning.append(f"Excellent debt management with debt-to-asset ratio of {debt_to_asset:.2f}")
            elif debt_to_asset < 0.5:  # Moderate debt to asset ratio
                score += 1
                reasoning.append(f"Good debt management with debt-to-asset ratio of {debt_to_asset:.2f}")
            else:
                reasoning.append(f"High debt burden with debt-to-asset ratio of {debt_to_asset:.2f}")
        else:
            reasoning.append("Debt to asset data not available")

        # Check asset turnover - efficiency metric
        if latest_metrics.get('asset_turnover'):
            asset_turnover = latest_metrics['asset_turnover']
            if asset_turnover > 1.0:  # High asset turnover
                score += 2
                reasoning.append(f"High asset turnover of {asset_turnover:.2f} (efficient asset utilization)")
            elif asset_turnover > 0.7:  # Moderate asset turnover
                score += 1
                reasoning.append(f"Moderate asset turnover of {asset_turnover:.2f}")
            else:
                reasoning.append(f"Low asset turnover of {asset_turnover:.2f}")
        else:
            reasoning.append("Asset turnover data not available")

        # Check free cash flow generation
        if latest_metrics.get('free_cash_flow') and latest_metrics.get('net_income'):
            fcf = latest_metrics['free_cash_flow']
            net_income = latest_metrics['net_income']
            if net_income > 0:  # Avoid division by zero or negative income
                fcf_ratio = fcf / net_income
                if fcf_ratio > 0.8:  # High free cash flow conversion
                    score += 2
                    reasoning.append(f"Excellent free cash flow generation ({fcf_ratio:.0%} of net income)")
                elif fcf_ratio > 0.5:  # Good free cash flow conversion
                    score += 1
                    reasoning.append(f"Good free cash flow generation ({fcf_ratio:.0%} of net income)")
                else:
                    reasoning.append(f"Low free cash flow generation ({fcf_ratio:.0%} of net income)")
            else:
                reasoning.append("Cannot calculate free cash flow ratio (negative net income)")
        else:
            reasoning.append("Free cash flow data not available")

        # Check gross margin stability
        if len(metrics) >= 3:
            gross_margins = []
            for metric in metrics[:min(3, len(metrics))]:  # Check up to 3 years
                if metric.get('gross_margin'):
                    gross_margins.append(metric['gross_margin'])
            
            if len(gross_margins) >= 3:
                # Check margin stability (low standard deviation)
                avg_margin = sum(gross_margins) / len(gross_margins)
                if avg_margin > 0:
                    # Simple stability check - all margins within 10% of average
                    stable = all(abs(margin - avg_margin) / avg_margin < 0.10 for margin in gross_margins if avg_margin > 0)
                    if stable:
                        score += 1
                        reasoning.append(f"Stable gross margins (avg: {avg_margin:.1%})")
                    else:
                        reasoning.append(f"Volatile gross margins (avg: {avg_margin:.1%})")
            else:
                reasoning.append("Insufficient gross margin data for stability analysis")

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
        analysis['title'] = f'Quality analysis'

        analysis_data['quality_analysis'] = analysis
        ai_message = AIMessage(content=self.get_markdown(analysis))
        return {
            "context": context,
            "messages": [
                ai_message
            ]
        }