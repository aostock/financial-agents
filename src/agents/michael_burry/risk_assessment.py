from common.agent_state import AgentState
from langchain.schema import AIMessage
from langchain_core.callbacks import dispatch_custom_event
from langchain_core.runnables import RunnableConfig
from typing import Dict, Any

import time
from common import markdown
from langgraph.types import StreamWriter


class RiskAssessment():
    def __init__(self, options: Dict[str, Any]):
        self.options = options

    
    def analyze(self, metrics: list) -> dict[str, any]:
        """Assess investment risks and evaluate asymmetric risk-reward profiles."""
        result = {"score": 0, "max_score": 10, "details": []}
        if not metrics or len(metrics) < 3:
            result["details"].append('Insufficient historical data (need at least 3 years)')
            return result

        latest_metrics = metrics[0]
        previous_metrics = metrics[1] if len(metrics) > 1 else None

        score = 0
        reasoning = []

        # 1. Financial risk assessment
        # Debt levels and coverage
        if latest_metrics.get('total_liabilities') and latest_metrics.get('total_assets'):
            debt_ratio = latest_metrics['total_liabilities'] / latest_metrics['total_assets'] if latest_metrics['total_assets'] > 0 else 0
            if debt_ratio < 0.3:  # Conservative leverage
                score += 2
                reasoning.append(f"Low financial risk: Debt-to-assets ratio {debt_ratio:.1%}")
            elif debt_ratio < 0.5:  # Moderate leverage
                score += 1
                reasoning.append(f"Moderate financial risk: Debt-to-assets ratio {debt_ratio:.1%}")
            else:
                reasoning.append(f"High financial risk: Debt-to-assets ratio {debt_ratio:.1%}")

        # Interest coverage
        if latest_metrics.get('ebit') and latest_metrics.get('interest_expense'):
            interest_coverage = latest_metrics['ebit'] / latest_metrics['interest_expense'] if latest_metrics['interest_expense'] > 0 else 0
            if interest_coverage > 5:  # Strong coverage
                score += 2
                reasoning.append(f"Strong interest coverage: {interest_coverage:.1f}x")
            elif interest_coverage > 3:  # Adequate coverage
                score += 1
                reasoning.append(f"Adequate interest coverage: {interest_coverage:.1f}x")
            elif interest_coverage > 0:
                reasoning.append(f"Weak interest coverage: {interest_coverage:.1f}x")
            else:
                reasoning.append(f"Interest coverage deficit")

        # 2. Business risk assessment
        # Revenue stability
        revenues = [m.get('revenue') for m in metrics[:5] if m.get('revenue') is not None]
        if len(revenues) >= 3:
            avg_revenue = sum(revenues) / len(revenues)
            volatility = sum(abs(r - avg_revenue) for r in revenues) / len(revenues) if avg_revenue != 0 else 0
            normalized_volatility = volatility / avg_revenue if avg_revenue != 0 else 0
            
            if normalized_volatility < 0.1:  # Low revenue volatility
                score += 2
                reasoning.append(f"Stable revenue stream (volatility: {normalized_volatility:.1%})")
            elif normalized_volatility < 0.2:  # Moderate revenue volatility
                score += 1
                reasoning.append(f"Moderately stable revenue (volatility: {normalized_volatility:.1%})")
            else:
                reasoning.append(f"High revenue volatility (volatility: {normalized_volatility:.1%})")

        # Earnings stability
        earnings = [m.get('net_income') for m in metrics[:5] if m.get('net_income') is not None]
        if len(earnings) >= 3:
            avg_earnings = sum(earnings) / len(earnings)
            volatility = sum(abs(e - avg_earnings) for e in earnings) / len(earnings) if avg_earnings != 0 else 0
            normalized_volatility = volatility / abs(avg_earnings) if avg_earnings != 0 else 0
            
            if normalized_volatility < 0.2:  # Low earnings volatility
                score += 2
                reasoning.append(f"Stable earnings pattern (volatility: {normalized_volatility:.1%})")
            elif normalized_volatility < 0.4:  # Moderate earnings volatility
                score += 1
                reasoning.append(f"Moderately stable earnings (volatility: {normalized_volatility:.1%})")
            else:
                reasoning.append(f"High earnings volatility (volatility: {normalized_volatility:.1%})")

        # 3. Market risk assessment
        # Beta analysis
        if latest_metrics.get('beta'):
            beta = latest_metrics['beta']
            if beta < 0.8:  # Low market sensitivity
                score += 2
                reasoning.append(f"Low market risk (Beta: {beta:.2f})")
            elif beta < 1.2:  # Market-like sensitivity
                score += 1
                reasoning.append(f"Moderate market risk (Beta: {beta:.2f})")
            else:
                reasoning.append(f"High market risk (Beta: {beta:.2f})")

        # 4. Liquidity risk assessment
        # Current ratio
        if latest_metrics.get('current_ratio'):
            current_ratio = latest_metrics['current_ratio']
            if current_ratio > 2.5:  # Strong liquidity
                score += 1
                reasoning.append(f"Strong liquidity position (Current Ratio: {current_ratio:.2f})")
            elif current_ratio > 1.5:  # Adequate liquidity
                reasoning.append(f"Adequate liquidity (Current Ratio: {current_ratio:.2f})")
            else:
                reasoning.append(f"Liquidity concerns (Current Ratio: {current_ratio:.2f})")

        # 5. Asymmetric risk-reward assessment
        # Compare potential upside to downside
        market_cap = latest_metrics.get('market_cap', 0)
        if market_cap > 0 and latest_metrics.get('ordinary_shares_number') and latest_metrics.get('free_cash_flow'):
            shares = latest_metrics['ordinary_shares_number']
            fcf = latest_metrics['free_cash_flow']
            
            if shares > 0:
                current_price = market_cap / shares
                
                # Simple intrinsic value calculation
                # Using FCF yield approach for quick assessment
                if fcf > 0:
                    fcf_yield = fcf / market_cap
                    # Intrinsic value estimate based on target yield (e.g., 8%)
                    target_yield = 0.08
                    intrinsic_value = (fcf / target_yield) / shares if target_yield > 0 else current_price
                    
                    if intrinsic_value > current_price:
                        upside = (intrinsic_value - current_price) / current_price
                        # If upside is significant (>50%), it suggests asymmetric opportunity
                        if upside > 0.5:
                            score += 2
                            reasoning.append(f"Significant upside potential: {upside:.1%}")
                        elif upside > 0.2:
                            score += 1
                            reasoning.append(f"Good upside potential: {upside:.1%}")

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
        analysis['type'] = 'risk_assessment'
        analysis['title'] = f'Risk Assessment'

        analysis_data['risk_assessment'] = analysis
        ai_message = AIMessage(content=self.get_markdown(analysis))
        return {
            "context": context,
            "messages": [
                ai_message
            ]
        }