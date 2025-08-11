from common.agent_state import AgentState
from langchain.schema import AIMessage
from langchain_core.callbacks import dispatch_custom_event
from langchain_core.runnables import RunnableConfig
from typing import Dict, Any

import time
from common import markdown
from langgraph.types import StreamWriter


class AdaptiveStrategyAnalysis():
    def __init__(self, options: Dict[str, Any]):
        self.options = options

    
    def analyze(self, metrics: list, prices: list) -> dict[str, any]:
        """Analyze the company's adaptive strategy and business model flexibility."""
        result = {"score": 0, "max_score": 10, "details": []}
        if not metrics:
            result["details"].append('No metrics available for adaptive strategy analysis')
            return result

        latest_metrics = metrics[0]
        score = 0
        reasoning = []
        
        # Revenue growth consistency (ability to adapt to market changes)
        if latest_metrics.get('revenue'):
            # Calculate revenue growth rate over multiple years
            revenue_growth_scores = []
            for i in range(min(5, len(metrics))):  # Check up to 5 years
                if metrics[i].get('revenue') and i+1 < len(metrics) and metrics[i+1].get('revenue'):
                    growth = (metrics[i]['revenue'] - metrics[i+1]['revenue']) / metrics[i+1]['revenue'] * 100
                    revenue_growth_scores.append(growth)
            
            if len(revenue_growth_scores) >= 3:
                avg_growth = sum(revenue_growth_scores) / len(revenue_growth_scores)
                if avg_growth > 10:
                    score += 2
                    reasoning.append(f"Strong consistent revenue growth: {avg_growth:.1f}% annually")
                elif avg_growth > 5:
                    score += 1
                    reasoning.append(f"Moderate revenue growth: {avg_growth:.1f}% annually")
                elif avg_growth < 0:
                    score -= 1
                    reasoning.append(f"Revenue decline: {avg_growth:.1f}% annually")
                else:
                    reasoning.append(f"Stable revenue: {avg_growth:.1f}% annually")
            else:
                reasoning.append("Insufficient revenue data for growth analysis")
        
        # Profitability consistency (adaptive cost management)
        if latest_metrics.get('net_income') and latest_metrics.get('revenue'):
            profit_margin = latest_metrics['net_income'] / latest_metrics['revenue'] * 100
            if profit_margin > 15:
                score += 2
                reasoning.append(f"Strong profitability margin: {profit_margin:.1f}%")
            elif profit_margin > 5:
                score += 1
                reasoning.append(f"Healthy profitability margin: {profit_margin:.1f}%")
            elif profit_margin < 0:
                score -= 2
                reasoning.append(f"Unprofitable: {profit_margin:.1f}% margin")
            else:
                reasoning.append(f"Low profitability margin: {profit_margin:.1f}%")
        
        # Free cash flow generation (financial flexibility)
        if latest_metrics.get('free_cash_flow'):
            fcf = latest_metrics['free_cash_flow']
            if fcf > 0:
                score += 1
                reasoning.append("Positive free cash flow generation - financial flexibility")
            else:
                score -= 1
                reasoning.append("Negative free cash flow - potential liquidity constraints")
        
        # Debt management (adaptive capital structure)
        if latest_metrics.get('debt_to_equity'):
            debt_ratio = latest_metrics['debt_to_equity']
            if debt_ratio < 0.3:
                score += 1
                reasoning.append(f"Conservative capital structure: {debt_ratio:.2f} debt-to-equity")
            elif debt_ratio > 1.0:
                score -= 1
                reasoning.append(f"High leverage: {debt_ratio:.2f} debt-to-equity")
            else:
                reasoning.append(f"Moderate leverage: {debt_ratio:.2f} debt-to-equity")
        
        # Asset turnover (operational efficiency and adaptability)
        if latest_metrics.get('asset_turnover'):
            asset_turnover = latest_metrics['asset_turnover']
            if asset_turnover > 1.0:
                score += 1
                reasoning.append(f"High asset efficiency: {asset_turnover:.2f} asset turnover")
            elif asset_turnover < 0.5:
                score -= 1
                reasoning.append(f"Low asset efficiency: {asset_turnover:.2f} asset turnover")
            else:
                reasoning.append(f"Moderate asset efficiency: {asset_turnover:.2f} asset turnover")

        result["score"] = max(0, min(10, score))  # Clamp between 0-10
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
        prices = context.get('prices')
        analysis = self.analyze(metrics, prices)
        analysis['type'] = 'adaptive_strategy_analysis'
        analysis['title'] = f'Adaptive Strategy Analysis'

        analysis_data['adaptive_strategy_analysis'] = analysis
        ai_message = AIMessage(content=self.get_markdown(analysis))
        return {
            "context": context,
            "messages": [
                ai_message
            ]
        }