from common.agent_state import AgentState
from langchain.schema import AIMessage
from langchain_core.callbacks import dispatch_custom_event
from langchain_core.runnables import RunnableConfig
from typing import Dict, Any

import time
from common import markdown
from langgraph.types import StreamWriter


class FundamentalAnalysis():
    def __init__(self, options: Dict[str, Any]):
        self.options = options

    
    def analyze(self, metrics: list) -> dict[str, any]:
        """Analyze company fundamentals based on Rakesh Jhunjhunwala's criteria."""
        result = {"score": 0, "max_score": 10, "details": []}
        if not metrics:
            result["details"].append('No metrics available')
            return result

        latest_metrics = metrics[0]

        score = 0
        reasoning = []

        # Check ROE (Return on Equity) - Jhunjhunwala looks for high ROE
        if latest_metrics.get('return_on_equity'):
            roe = latest_metrics['return_on_equity']
            if roe > 0.20:  # >20% excellent
                score += 3
                reasoning.append(f"Excellent ROE of {roe:.1%} (target: >20%)")
            elif roe > 0.15:  # >15% strong
                score += 2
                reasoning.append(f"Strong ROE of {roe:.1%} (target: >15%)")
            elif roe > 0.10:  # >10% moderate
                score += 1
                reasoning.append(f"Moderate ROE of {roe:.1%} (target: >10%)")
            else:
                reasoning.append(f"Weak ROE of {roe:.1%}")
        else:
            reasoning.append(f"ROE data not available")

        # Check Debt to Equity - Jhunjhunwala prefers low debt companies
        if latest_metrics.get('debt_to_equity'):
            debt_ratio = latest_metrics['debt_to_equity']
            if debt_ratio < 0.5:  # Low debt
                score += 2
                reasoning.append(f"Low debt company with debt-to-equity ratio of {debt_ratio:.2f}")
            else:
                reasoning.append(f"High debt company with debt-to-equity ratio of {debt_ratio:.2f}")
        else:
            reasoning.append(f"Debt to equity data not available")

        # Check Operating Margin - Jhunjhunwala looks for strong operating efficiency
        if latest_metrics.get('operating_margin'):
            op_margin = latest_metrics['operating_margin']
            if op_margin > 0.15:  # >15% strong
                score += 2
                reasoning.append(f"Strong operating margin of {op_margin:.1%} (target: >15%)")
            elif op_margin > 0.10:  # >10% acceptable
                score += 1
                reasoning.append(f"Acceptable operating margin of {op_margin:.1%}")
            else:
                reasoning.append(f"Weak operating margin of {op_margin:.1%}")
        else:
            reasoning.append(f"Operating margin data not available")

        # Check Current Ratio - Jhunjhunwala looks for strong liquidity
        if latest_metrics.get('current_ratio'):
            current_ratio = latest_metrics['current_ratio']
            if current_ratio > 2.0:  # Excellent liquidity
                score += 2
                reasoning.append(f"Excellent liquidity with current ratio of {current_ratio:.2f}")
            elif current_ratio > 1.5:  # Good liquidity
                score += 1
                reasoning.append(f"Good liquidity with current ratio of {current_ratio:.2f}")
            else:
                reasoning.append(f"Weak liquidity with current ratio of {current_ratio:.2f}")
        else:
            reasoning.append(f"Current ratio data not available")

        # Check Return on Invested Capital (ROIC) - Additional quality metric
        if latest_metrics.get('return_on_invested_capital'):
            roic = latest_metrics['return_on_invested_capital']
            if roic > 0.15:  # >15% excellent
                score += 1
                reasoning.append(f"Excellent ROIC of {roic:.1%}")
            elif roic > 0.10:  # >10% good
                score += 0.5
                reasoning.append(f"Good ROIC of {roic:.1%}")
            else:
                reasoning.append(f"Low ROIC of {roic:.1%}")
        else:
            reasoning.append(f"ROIC data not available")

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
        analysis['type'] = 'fundamental_analysis'
        analysis['title'] = f'Fundamental analysis'

        analysis_data['fundamental_analysis'] = analysis
        ai_message = AIMessage(content=self.get_markdown(analysis))
        return {
            "context": context,
            "messages": [
                ai_message
            ]
        }