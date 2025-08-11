from common.agent_state import AgentState
from langchain.schema import AIMessage
from langchain_core.callbacks import dispatch_custom_event
from langchain_core.runnables import RunnableConfig
from typing import Dict, Any

import time
from common import markdown
from langgraph.types import StreamWriter


class GrowthQualityAnalysis():
    def __init__(self, options: Dict[str, Any]):
        self.options = options

    
    def analyze(self, metrics: list) -> dict[str, any]:
        """Analyze growth & quality based on Phil Fisher's criteria."""
        result = {"score": 0, "max_score": 10, "details": []}
        if not metrics or len(metrics) < 2:
            result["details"].append('Insufficient historical data (need at least 2 years)')
            return result

        latest_metrics = metrics[0]
        previous_metrics = metrics[1]

        score = 0
        reasoning = []

        # Calculate revenue growth (multi-year if possible)
        revenues = [m.get('revenue') for m in metrics if m.get('revenue') is not None]
        if len(revenues) >= 2:
            latest_rev = revenues[0]
            oldest_rev = revenues[-1]
            if oldest_rev and oldest_rev > 0:
                rev_growth = (latest_rev - oldest_rev) / abs(oldest_rev)
                if rev_growth > 0.80:
                    score += 3
                    reasoning.append(f"Very strong multi-period revenue growth: {rev_growth:.1%}")
                elif rev_growth > 0.40:
                    score += 2
                    reasoning.append(f"Moderate multi-period revenue growth: {rev_growth:.1%}")
                elif rev_growth > 0.10:
                    score += 1
                    reasoning.append(f"Slight multi-period revenue growth: {rev_growth:.1%}")
                else:
                    reasoning.append(f"Minimal or negative multi-period revenue growth: {rev_growth:.1%}")
            else:
                reasoning.append("Oldest revenue is zero/negative; cannot compute growth.")
        else:
            reasoning.append("Not enough revenue data points for growth calculation.")

        # Calculate EPS growth (multi-year if possible)
        eps_values = [m.get('earnings_per_share') for m in metrics if m.get('earnings_per_share') is not None]
        if len(eps_values) >= 2:
            latest_eps = eps_values[0]
            oldest_eps = eps_values[-1]
            if oldest_eps and abs(oldest_eps) > 1e-9:
                eps_growth = (latest_eps - oldest_eps) / abs(oldest_eps)
                if eps_growth > 0.80:
                    score += 3
                    reasoning.append(f"Very strong multi-period EPS growth: {eps_growth:.1%}")
                elif eps_growth > 0.40:
                    score += 2
                    reasoning.append(f"Moderate multi-period EPS growth: {eps_growth:.1%}")
                elif eps_growth > 0.10:
                    score += 1
                    reasoning.append(f"Slight multi-period EPS growth: {eps_growth:.1%}")
                else:
                    reasoning.append(f"Minimal or negative multi-period EPS growth: {eps_growth:.1%}")
            else:
                reasoning.append("Oldest EPS near zero; skipping EPS growth calculation.")
        else:
            reasoning.append("Not enough EPS data points for growth calculation.")

        # R&D as % of Revenue (if we have R&D data)
        rnd_values = [m.get('research_and_development') for m in metrics if m.get('research_and_development') is not None]
        if rnd_values and revenues and len(rnd_values) == len(revenues) and rnd_values[0] and revenues[0]:
            recent_rnd = rnd_values[0]
            recent_rev = revenues[0] if revenues[0] else 1e-9
            rnd_ratio = recent_rnd / recent_rev
            # Generally, Fisher admired companies that invest aggressively in R&D,
            # but it must be appropriate. We'll assume "3%-15%" is healthy, just as an example.
            if 0.03 <= rnd_ratio <= 0.15:
                score += 4
                reasoning.append(f"R&D ratio {rnd_ratio:.1%} indicates significant investment in future growth")
            elif rnd_ratio > 0.15:
                score += 3
                reasoning.append(f"R&D ratio {rnd_ratio:.1%} is very high (could be good if well-managed)")
            elif rnd_ratio > 0.0:
                score += 2
                reasoning.append(f"R&D ratio {rnd_ratio:.1%} is somewhat low but still positive")
            else:
                reasoning.append("No meaningful R&D expense ratio")
        else:
            reasoning.append("Insufficient R&D data to evaluate")

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
        analysis['type'] = 'growth_quality_analysis'
        analysis['title'] = f'Growth & Quality Analysis'

        analysis_data['growth_quality_analysis'] = analysis
        ai_message = AIMessage(content=self.get_markdown(analysis))
        return {
            "context": context,
            "messages": [
                ai_message
            ]
        }