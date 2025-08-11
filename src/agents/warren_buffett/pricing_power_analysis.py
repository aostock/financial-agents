from common.agent_state import AgentState
from langchain.schema import AIMessage
from langchain_core.callbacks import dispatch_custom_event
from langchain_core.runnables import RunnableConfig
from typing import Dict, Any

import time
from common import markdown
from langgraph.types import StreamWriter



class PricingPowerAnalysis():
    def __init__(self, options: Dict[str, Any]):
        self.options = options

    
    def analyze(self, financial_line_items: list) -> dict[str, any]:
        """
        Analyze pricing power - Buffett's key indicator of a business moat.
        Looks at ability to raise prices without losing customers (margin expansion during inflation).
        """
        result = {"score": 0, "max_score": 5, "details": []}
        if not financial_line_items:
            result["details"].append("Insufficient data for pricing power analysis")
            return result
        
        score = 0
        reasoning = []
        
        # Check gross margin trends (ability to maintain/expand margins)
        gross_margins = []
        for item in financial_line_items:
            if item.get('gross_margin') is not None:
                gross_margins.append(item.get('gross_margin'))
        
        if len(gross_margins) >= 3:
            # Check margin stability/improvement
            recent_avg = sum(gross_margins[:2]) / 2 if len(gross_margins) >= 2 else gross_margins[0]
            older_avg = sum(gross_margins[-2:]) / 2 if len(gross_margins) >= 2 else gross_margins[-1]
            
            if recent_avg > older_avg + 0.02:  # 2%+ improvement
                score += 3
                reasoning.append("Expanding gross margins indicate strong pricing power")
            elif recent_avg > older_avg:
                score += 2
                reasoning.append("Improving gross margins suggest good pricing power")
            elif abs(recent_avg - older_avg) < 0.01:  # Stable within 1%
                score += 1
                reasoning.append("Stable gross margins during economic uncertainty")
            else:
                reasoning.append("Declining gross margins may indicate pricing pressure")
        
        # Check if company has been able to maintain high margins consistently
        if gross_margins:
            avg_margin = sum(gross_margins) / len(gross_margins)
            if avg_margin > 0.5:  # 50%+ gross margins
                score += 2
                reasoning.append(f"Consistently high gross margins ({avg_margin:.1%}) indicate strong pricing power")
            elif avg_margin > 0.3:  # 30%+ gross margins
                score += 1
                reasoning.append(f"Good gross margins ({avg_margin:.1%}) suggest decent pricing power")
        
        result["score"] = score
        result["details"] = reasoning if reasoning else ["Limited pricing power analysis available"]
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
        analysis['type'] = 'pricing_power_analysis'
        analysis['title'] = 'Pricing power analysis'

        analysis_data['pricing_power_analysis'] = analysis
        ai_message = AIMessage(content=self.get_markdown(analysis))
        return {
            "context": context,
            "messages": [
                ai_message
            ]
        }