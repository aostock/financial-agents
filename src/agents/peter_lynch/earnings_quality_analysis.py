from common.agent_state import AgentState
from langchain.schema import AIMessage
from langchain_core.callbacks import dispatch_custom_event
from langchain_core.runnables import RunnableConfig
from typing import Dict, Any

import time
from common import markdown
from langgraph.types import StreamWriter


class EarningsQualityAnalysis():
    def __init__(self, options: Dict[str, Any]):
        self.options = options

    
    def analyze(self, metrics: list) -> dict[str, any]:
        """Analyze earnings quality based on Peter Lynch's criteria."""
        result = {"score": 0, "max_score": 10, "details": []}
        if not metrics or len(metrics) < 3:
            result["details"].append('Insufficient historical data (need at least 3 years)')
            return result

        latest_metrics = metrics[0]
        previous_metrics = metrics[1] if len(metrics) > 1 else None

        score = 0
        reasoning = []

        # Check earnings consistency - Lynch looks for companies with consistent earnings
        earnings = [m.get('net_income') for m in metrics[:5] if m.get('net_income') is not None]
        if len(earnings) >= 3:
            positive_earnings = sum(1 for e in earnings if e and e > 0)
            if positive_earnings == len(earnings):
                score += 2
                reasoning.append("Consistently profitable over last 5 years")
            elif positive_earnings >= len(earnings) * 0.8:
                score += 1
                reasoning.append("Mostly profitable with occasional losses")
            else:
                reasoning.append("Frequently unprofitable")
        else:
            reasoning.append("Insufficient earnings data for consistency analysis")

        # Check earnings quality - compare net income to cash flow
        net_income = latest_metrics.get('net_income')
        free_cash_flow = latest_metrics.get('free_cash_flow')
        
        if net_income and free_cash_flow:
            if net_income > 0 and free_cash_flow > 0:
                cash_flow_coverage = free_cash_flow / net_income
                if cash_flow_coverage > 0.8:
                    score += 2
                    reasoning.append(f"High quality earnings (FCF coverage: {cash_flow_coverage:.1%})")
                elif cash_flow_coverage > 0.5:
                    score += 1
                    reasoning.append(f"Good earnings quality (FCF coverage: {cash_flow_coverage:.1%})")
                else:
                    reasoning.append(f"Poor earnings quality (FCF coverage: {cash_flow_coverage:.1%})")
            elif net_income > 0 and free_cash_flow < 0:
                reasoning.append("Reported profits but negative free cash flow - concerning")
            elif net_income < 0 and free_cash_flow < 0:
                reasoning.append("Both earnings and cash flow negative")
            else:
                reasoning.append("Mixed earnings and cash flow signals")
        else:
            reasoning.append("Insufficient data for earnings quality assessment")

        # Check for earnings manipulation - look for unusual patterns
        if (latest_metrics.get('net_income') and previous_metrics and previous_metrics.get('net_income') and
            latest_metrics.get('revenue') and previous_metrics.get('revenue')):
            
            earnings_change = (latest_metrics['net_income'] - previous_metrics['net_income']) / abs(previous_metrics['net_income']) if previous_metrics['net_income'] != 0 else 0
            revenue_change = (latest_metrics['revenue'] - previous_metrics['revenue']) / previous_metrics['revenue'] if previous_metrics['revenue'] != 0 else 0
            
            if revenue_change > 0.1 and earnings_change > revenue_change * 2:
                reasoning.append("Earnings growth significantly exceeds revenue growth - potential accounting issues")
            elif revenue_change > 0.1 and earnings_change > 0:
                score += 1
                reasoning.append("Earnings growth aligned with revenue growth")
            elif revenue_change <= 0.1 and earnings_change > 0.2:
                reasoning.append("Earnings growth without revenue support - investigate further")
            else:
                reasoning.append("Earnings and revenue growth patterns appear normal")

        # Check for one-time items or extraordinary charges
        # This would require more detailed financial statement data
        reasoning.append("Earnings quality analysis complete - look for consistent, high-quality earnings")

        # Check for dividend policy consistency
        if latest_metrics.get('dividends_and_other_cash_distributions'):
            score += 1
            reasoning.append("Company pays dividends - indicates financial stability")
        else:
            reasoning.append("No dividend payments - may retain earnings for growth")

        # Check for share count changes (dilution/concentration)
        shares = latest_metrics.get('ordinary_shares_number')
        if len(metrics) >= 2 and metrics[1].get('ordinary_shares_number') and shares:
            share_change = (shares - metrics[1]['ordinary_shares_number']) / metrics[1]['ordinary_shares_number']
            if share_change < -0.05:  # Significant share buybacks
                score += 1
                reasoning.append(f"Share buybacks indicate management confidence ({share_change:.1%})")
            elif share_change > 0.05:  # Significant dilution
                reasoning.append(f"Share dilution may reduce per-share value ({share_change:.1%})")
            else:
                reasoning.append("Stable share count")

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
        analysis['type'] = 'earnings_quality_analysis'
        analysis['title'] = f'Earnings Quality Analysis'

        analysis_data['earnings_quality_analysis'] = analysis
        ai_message = AIMessage(content=self.get_markdown(analysis))
        return {
            "context": context,
            "messages": [
                ai_message
            ]
        }