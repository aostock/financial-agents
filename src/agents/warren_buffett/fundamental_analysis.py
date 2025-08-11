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
        """Analyze company fundamentals based on Buffett's criteria."""
        result = {"score": 0, "max_score": 7, "details": []}
        if not metrics:
            result["details"].append('No metrics available')
            return result

        latest_metrics = metrics[0]

        score = 0
        reasoning = []

        # Check ROE (Return on Equity)
        if latest_metrics.get('return_on_equity') and latest_metrics['return_on_equity'] > 0.15:  # 15% ROE threshold
            score += 2
            reasoning.append(f"Strong ROE of {latest_metrics['return_on_equity']:.1%}")
        elif latest_metrics.get('return_on_equity'):
            reasoning.append(f"Weak ROE of {latest_metrics['return_on_equity']:.1%}")
        else:
            reasoning.append(f"ROE data not available")

        # Check Debt to Equity
        if latest_metrics.get('debt_to_equity') and latest_metrics['debt_to_equity'] < 0.5:
            score += 2
            reasoning.append(f"Conservative debt levels")
        elif latest_metrics.get('debt_to_equity'):
            reasoning.append(f"High debt to equity ratio of {latest_metrics['debt_to_equity']:.1f}")
        else:
            reasoning.append(f"Debt to equity data not available")

        # Check Operating Margin
        if latest_metrics.get('operating_margin') and latest_metrics['operating_margin'] > 0.15:
            score += 2
            reasoning.append(f"Strong operating margins")
        elif latest_metrics.get('operating_margin'):
            reasoning.append(f"Weak operating margin of {latest_metrics['operating_margin']:.1%}")
        else:
            reasoning.append(f"Operating margin data not available")

        # Check Current Ratio
        if latest_metrics.get('current_ratio') and latest_metrics['current_ratio'] > 1.5:
            score += 1
            reasoning.append(f"Good liquidity position")
        elif latest_metrics.get('current_ratio'):
            reasoning.append(f"Weak liquidity with current ratio of {latest_metrics['current_ratio']:.1f}")
        else:
            reasoning.append(f"Current ratio data not available")
        # types.SimpleNamespace to json

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