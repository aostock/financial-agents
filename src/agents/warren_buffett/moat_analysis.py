from common.agent_state import AgentState
from langchain.schema import AIMessage
from langchain_core.callbacks import dispatch_custom_event
from langchain_core.runnables import RunnableConfig
from typing import Dict, Any

import time
from common import markdown
from langgraph.types import StreamWriter



class MoatAnalysis():
    def __init__(self, options: Dict[str, Any]):
        self.options = options

    
    def analyze(self, metrics: list) -> dict[str, any]:
        """
        Evaluate whether the company likely has a durable competitive advantage (moat).
        Enhanced to include multiple moat indicators that Buffett actually looks for:
        1. Consistent high returns on capital
        2. Pricing power (stable/growing margins)
        3. Scale advantages (improving metrics with size)
        4. Brand strength (inferred from margins and consistency)
        5. Switching costs (inferred from customer retention)
        """
        result = {"score": 0, "max_score": 5, "details": []}
        if not metrics or len(metrics) < 5:  # Need more data for proper moat analysis
            result["details"].append("Insufficient data for comprehensive moat analysis")
            return result

        reasoning = []
        moat_score = 0
        max_score = 5

        # 1. Return on Capital Consistency (Buffett's favorite moat indicator)
        historical_roes = [m.get('return_on_equity') for m in metrics if m.get('return_on_equity') is not None]
        historical_roics = [m.get('return_on_invested_capital') for m in metrics if m.get('return_on_invested_capital') is not None]
        
        if len(historical_roes) >= 5:
            # Check for consistently high ROE (>15% for most periods)
            high_roe_periods = sum(1 for roe in historical_roes if roe > 0.15)
            roe_consistency = high_roe_periods / len(historical_roes)
            
            if roe_consistency >= 0.8:  # 80%+ of periods with ROE > 15%
                moat_score += 2
                avg_roe = sum(historical_roes) / len(historical_roes)
                reasoning.append(f"Excellent ROE consistency: {high_roe_periods}/{len(historical_roes)} periods >15% (avg: {avg_roe:.1%}) - indicates durable competitive advantage")
            elif roe_consistency >= 0.6:
                moat_score += 1
                reasoning.append(f"Good ROE performance: {high_roe_periods}/{len(historical_roes)} periods >15%")
            else:
                reasoning.append(f"Inconsistent ROE: only {high_roe_periods}/{len(historical_roes)} periods >15%")
        else:
            reasoning.append("Insufficient ROE history for moat analysis")

        # 2. Operating Margin Stability (Pricing Power Indicator)
        historical_margins = [m.get('operating_margin') for m in metrics if m.get('operating_margin') is not None]
        if len(historical_margins) >= 5:
            # Check for stable or improving margins (sign of pricing power)
            avg_margin = sum(historical_margins) / len(historical_margins)
            recent_margins = historical_margins[:3]  # Last 3 periods
            older_margins = historical_margins[-3:]  # First 3 periods
            
            recent_avg = sum(recent_margins) / len(recent_margins)
            older_avg = sum(older_margins) / len(older_margins)
            
            if avg_margin > 0.2 and recent_avg >= older_avg:  # 20%+ margins and stable/improving
                moat_score += 1
                reasoning.append(f"Strong and stable operating margins (avg: {avg_margin:.1%}) indicate pricing power moat")
            elif avg_margin > 0.15:  # At least decent margins
                reasoning.append(f"Decent operating margins (avg: {avg_margin:.1%}) suggest some competitive advantage")
            else:
                reasoning.append(f"Low operating margins (avg: {avg_margin:.1%}) suggest limited pricing power")
        
        # 3. Asset Efficiency and Scale Advantages
        if len(metrics) >= 5:
            # Check asset turnover trends (revenue efficiency)
            asset_turnovers = []
            for m in metrics:
                if m.get('asset_turnover') is not None:
                    asset_turnovers.append(m.get('asset_turnover'))
            
            if len(asset_turnovers) >= 3:
                if any(turnover > 1.0 for turnover in asset_turnovers):  # Efficient asset use
                    moat_score += 1
                    reasoning.append("Efficient asset utilization suggests operational moat")
        
        # 4. Competitive Position Strength (inferred from trend stability)
        if len(historical_roes) >= 5 and len(historical_margins) >= 5:
            # Calculate coefficient of variation (stability measure)
            roe_avg = sum(historical_roes) / len(historical_roes)
            roe_variance = sum((roe - roe_avg) ** 2 for roe in historical_roes) / len(historical_roes)
            roe_stability = 1 - (roe_variance ** 0.5) / roe_avg if roe_avg > 0 else 0
            
            margin_avg = sum(historical_margins) / len(historical_margins)
            margin_variance = sum((margin - margin_avg) ** 2 for margin in historical_margins) / len(historical_margins)
            margin_stability = 1 - (margin_variance ** 0.5) / margin_avg if margin_avg > 0 else 0
            
            overall_stability = (roe_stability + margin_stability) / 2
            
            if overall_stability > 0.7:  # High stability indicates strong competitive position
                moat_score += 1
                reasoning.append(f"High performance stability ({overall_stability:.1%}) suggests strong competitive moat")
        
        # Cap the score at max_score
        moat_score = min(moat_score, max_score)

        result["score"] = moat_score
        result["details"] = reasoning if reasoning else ["Limited moat analysis available"]
        result["max_score"] = max_score

        return  result

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
        analysis['type'] = 'moat_analysis'
        analysis['title'] = f'MOAT analysis'

        analysis_data['moat_analysis'] = analysis
        ai_message = AIMessage(content=self.get_markdown(analysis))
        return {
            "context": context,
            "messages": [
                ai_message
            ]
        }