from common.agent_state import AgentState
from langchain.schema import AIMessage
from langchain_core.callbacks import dispatch_custom_event
from langchain_core.runnables import RunnableConfig
from typing import Dict, Any

import time
from common import markdown
from langgraph.types import StreamWriter


class RiskAnalysis():
    def __init__(self, options: Dict[str, Any]):
        self.options = options

    
    def analyze(self, metrics: list) -> dict[str, any]:
        """Analyze risk factors based on Druckenmiller's risk management principles."""
        result = {"score": 0, "max_score": 10, "details": []}
        if not metrics:
            result["details"].append('No metrics available for risk analysis')
            return result

        latest_metrics = metrics[0]
        score = 10  # Start with maximum score and subtract for risks
        reasoning = []
        
        # Financial risk - Debt levels
        if latest_metrics.get('debt_to_equity'):
            debt_ratio = latest_metrics['debt_to_equity']
            if debt_ratio > 1.0:
                score -= 3
                reasoning.append(f"High financial risk: Debt-to-equity ratio of {debt_ratio:.2f}")
            elif debt_ratio > 0.5:
                score -= 1
                reasoning.append(f"Moderate financial risk: Debt-to-equity ratio of {debt_ratio:.2f}")
            else:
                reasoning.append(f"Low financial risk: Debt-to-equity ratio of {debt_ratio:.2f}")
        
        # Liquidity risk - Current ratio
        if latest_metrics.get('current_ratio'):
            current_ratio = latest_metrics['current_ratio']
            if current_ratio < 1.0:
                score -= 2
                reasoning.append(f"Liquidity risk: Current ratio of {current_ratio:.2f}")
            elif current_ratio < 1.5:
                score -= 1
                reasoning.append(f"Moderate liquidity: Current ratio of {current_ratio:.2f}")
            else:
                reasoning.append(f"Strong liquidity: Current ratio of {current_ratio:.2f}")
        
        # Profitability risk - Consistent losses
        if latest_metrics.get('net_income'):
            net_income = latest_metrics['net_income']
            if net_income < 0:
                score -= 3
                reasoning.append("Loss-making company - fundamental business risk")
            else:
                reasoning.append("Profitable company - lower fundamental risk")
        
        # Market risk - Beta
        if latest_metrics.get('beta'):
            beta = latest_metrics['beta']
            if beta > 1.5:
                score -= 2
                reasoning.append(f"High market risk: Beta of {beta:.2f}")
            elif beta > 1.2:
                score -= 1
                reasoning.append(f"Elevated market risk: Beta of {beta:.2f}")
            else:
                reasoning.append(f"Moderate market risk: Beta of {beta:.2f}")
        
        # Business risk - Volatility in earnings
        if len(metrics) >= 3:
            earnings = [m.get('net_income', 0) for m in metrics[:3] if m.get('net_income') is not None]
            if len(earnings) >= 2:
                avg_earnings = sum(earnings) / len(earnings)
                if avg_earnings > 0:
                    volatility = sum(abs(e - avg_earnings) for e in earnings) / len(earnings) / avg_earnings * 100
                    if volatility > 50:
                        score -= 2
                        reasoning.append(f"High earnings volatility: {volatility:.1f}%")
                    elif volatility > 25:
                        score -= 1
                        reasoning.append(f"Moderate earnings volatility: {volatility:.1f}%")
                    else:
                        reasoning.append(f"Stable earnings: {volatility:.1f}% volatility")
        
        # Size risk - Small companies have higher risk
        if latest_metrics.get('market_cap'):
            market_cap = latest_metrics['market_cap']
            if market_cap < 1000000000:  # < $1B
                score -= 1
                reasoning.append("Small-cap company - higher idiosyncratic risk")
            elif market_cap < 5000000000:  # < $5B
                reasoning.append("Mid-cap company - moderate idiosyncratic risk")
            else:
                reasoning.append("Large-cap company - lower idiosyncratic risk")

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
        analysis = self.analyze(metrics)
        analysis['type'] = 'risk_analysis'
        analysis['title'] = f'Risk Analysis'

        analysis_data['risk_analysis'] = analysis
        ai_message = AIMessage(content=self.get_markdown(analysis))
        return {
            "context": context,
            "messages": [
                ai_message
            ]
        }