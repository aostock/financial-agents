from common.agent_state import AgentState
from langchain.schema import AIMessage
from langchain_core.callbacks import dispatch_custom_event
from langchain_core.runnables import RunnableConfig
from typing import Dict, Any

import time
from common import markdown
from langgraph.types import StreamWriter


class GlobalMarketAnalysis():
    def __init__(self, options: Dict[str, Any]):
        self.options = options

    
    def analyze(self, metrics: list) -> dict[str, any]:
        """Analyze global market factors and intermarket relationships."""
        result = {"score": 0, "max_score": 10, "details": []}
        if not metrics:
            result["details"].append('No metrics available for global market analysis')
            return result

        latest_metrics = metrics[0]
        score = 0
        reasoning = []
        
        # Beta analysis (market sensitivity)
        if latest_metrics.get('beta'):
            beta = latest_metrics['beta']
            if beta > 1.2:
                score += 1
                reasoning.append(f"High market sensitivity (beta: {beta:.2f}) - leveraged to market moves")
            elif beta < 0.8:
                score += 1
                reasoning.append(f"Low market sensitivity (beta: {beta:.2f}) - defensive characteristics")
            else:
                reasoning.append(f"Average market sensitivity (beta: {beta:.2f})")
        
        # Market cap analysis (size effect)
        if latest_metrics.get('market_cap'):
            market_cap = latest_metrics['market_cap']
            if market_cap > 100000000000:  # > $100B
                score += 1
                reasoning.append(f"Large-cap company - typically more stable in global markets")
            elif market_cap < 2000000000:  # < $2B
                score -= 1
                reasoning.append(f"Small-cap company - higher volatility in global market stress")
            else:
                reasoning.append(f"Mid-cap company - moderate global market exposure")
        
        # International exposure (inferred from market cap and sector)
        # This would ideally come from specific data, but we'll make inferences
        sector_exposure_score = 0
        # Companies with international operations typically have higher market caps
        if latest_metrics.get('market_cap') and latest_metrics.get('revenue'):
            # Simple heuristic: large companies are more likely to have international exposure
            if latest_metrics['market_cap'] > 50000000000:  # > $50B
                sector_exposure_score += 1
                reasoning.append("Likely has significant international operations based on size")
            else:
                reasoning.append("Limited international exposure inferred from size")
        
        score += sector_exposure_score
        
        # Diversification benefit analysis
        if latest_metrics.get('beta') and latest_metrics.get('market_cap'):
            # Companies with low beta and large market cap provide diversification
            if latest_metrics['beta'] < 1.0 and latest_metrics['market_cap'] > 50000000000:
                score += 1
                reasoning.append("Provides portfolio diversification benefits")
            elif latest_metrics['beta'] > 1.2 and latest_metrics['market_cap'] < 5000000000:
                score -= 1
                reasoning.append("High correlation risk in stressed global markets")
        
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
        analysis['type'] = 'global_market_analysis'
        analysis['title'] = f'Global Market Analysis'

        analysis_data['global_market_analysis'] = analysis
        ai_message = AIMessage(content=self.get_markdown(analysis))
        return {
            "context": context,
            "messages": [
                ai_message
            ]
        }