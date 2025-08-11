from common.agent_state import AgentState
from langchain.schema import AIMessage
from langchain_core.callbacks import dispatch_custom_event
from langchain_core.runnables import RunnableConfig
from typing import Dict, Any

import time
from common import markdown
from langgraph.types import StreamWriter


class ValuationAnalysis():
    def __init__(self, options: Dict[str, Any]):
        self.options = options

    
    def analyze(self, metrics: list) -> dict[str, any]:
        """Analyze company valuation based on Rakesh Jhunjhunwala's criteria."""
        result = {"score": 0, "max_score": 10, "details": [], "intrinsic_value": 0, "margin_of_safety": 0}
        if not metrics:
            result["details"].append('No metrics available')
            return result

        latest_metrics = metrics[0]

        score = 0
        reasoning = []

        # Calculate intrinsic value using simplified DCF approach
        intrinsic_value = self.calculate_intrinsic_value(metrics)
        if intrinsic_value <= 0:
            reasoning.append("Could not calculate intrinsic value")
            result["details"] = reasoning
            return result

        result["intrinsic_value"] = intrinsic_value

        # Get current market price
        if latest_metrics.get('market_cap') and latest_metrics.get('ordinary_shares_number'):
            shares_outstanding = latest_metrics['ordinary_shares_number']
            if shares_outstanding > 0:
                current_price = latest_metrics['market_cap'] / shares_outstanding
                
                # Calculate margin of safety
                if intrinsic_value > 0:
                    margin_of_safety = (intrinsic_value - current_price) / intrinsic_value
                    result["margin_of_safety"] = margin_of_safety
                    
                    if margin_of_safety > 0.30:  # >30% margin of safety (Jhunjhunwala's preference)
                        score += 4
                        reasoning.append(f"Excellent margin of safety of {margin_of_safety:.1%} (intrinsic: ${intrinsic_value:.2f}, current: ${current_price:.2f})")
                    elif margin_of_safety > 0.20:  # >20% margin of safety
                        score += 3
                        reasoning.append(f"Good margin of safety of {margin_of_safety:.1%} (intrinsic: ${intrinsic_value:.2f}, current: ${current_price:.2f})")
                    elif margin_of_safety > 0.10:  # >10% margin of safety
                        score += 2
                        reasoning.append(f"Moderate margin of safety of {margin_of_safety:.1%} (intrinsic: ${intrinsic_value:.2f}, current: ${current_price:.2f})")
                    elif margin_of_safety > 0:  # Some margin of safety
                        score += 1
                        reasoning.append(f"Minor margin of safety of {margin_of_safety:.1%} (intrinsic: ${intrinsic_value:.2f}, current: ${current_price:.2f})")
                    else:
                        reasoning.append(f"Overvalued by {abs(margin_of_safety):.1%} (intrinsic: ${intrinsic_value:.2f}, current: ${current_price:.2f})")
                else:
                    reasoning.append(f"Current price: ${current_price:.2f}, intrinsic value calculation failed")
            else:
                reasoning.append("Could not calculate current price (no shares outstanding data)")
        else:
            reasoning.append("Market price data not available")

        # Check P/E ratio
        if latest_metrics.get('price_to_earnings_ratio'):
            pe_ratio = latest_metrics['price_to_earnings_ratio']
            if pe_ratio > 0:  # Only for positive P/E ratios
                if pe_ratio < 15:  # Low P/E
                    score += 2
                    reasoning.append(f"Attractive P/E ratio of {pe_ratio:.1f}")
                elif pe_ratio < 25:  # Moderate P/E
                    score += 1
                    reasoning.append(f"Reasonable P/E ratio of {pe_ratio:.1f}")
                else:
                    reasoning.append(f"High P/E ratio of {pe_ratio:.1f}")
            else:
                reasoning.append(f"Unusual P/E ratio of {pe_ratio:.1f}")
        else:
            reasoning.append("P/E ratio data not available")

        result["score"] = score
        result["details"] = reasoning
        return result

    def calculate_intrinsic_value(self, metrics: list) -> float:
        """Calculate intrinsic value using a simplified DCF model based on Jhunjhunwala's approach."""
        if not metrics or len(metrics) < 3:
            return 0

        # Get last 3 years of free cash flow data
        fcfs = []
        for metric in metrics[:min(3, len(metrics))]:
            if metric.get('free_cash_flow'):
                fcfs.append(metric['free_cash_flow'])

        if len(fcfs) < 3:
            return 0

        # Calculate average free cash flow
        avg_fcf = sum(fcfs) / len(fcfs)
        
        # Determine quality-based discount rate
        # This is a simplified approach - in reality, this would be more complex
        discount_rate = 0.15  # Default 15% discount rate
        
        # Calculate average ROE to determine quality
        roes = []
        for metric in metrics[:min(3, len(metrics))]:
            if metric.get('return_on_equity'):
                roes.append(metric['return_on_equity'])
        
        if roes:
            avg_roe = sum(roes) / len(roes)
            # Adjust discount rate based on quality
            if avg_roe > 0.20:  # High quality
                discount_rate = 0.12
            elif avg_roe > 0.15:  # Medium quality
                discount_rate = 0.15
            else:  # Lower quality
                discount_rate = 0.18

        # Simplified DCF with 10-year projection and terminal value
        # Project FCF growth for next 10 years (conservative assumption)
        growth_rate = 0.10  # 10% growth rate (conservative)
        
        # Calculate present value of projected FCFs
        intrinsic_value = 0
        projected_fcf = avg_fcf
        
        for i in range(10):
            projected_fcf *= (1 + growth_rate)
            present_value = projected_fcf / ((1 + discount_rate) ** (i + 1))
            intrinsic_value += present_value

        # Add terminal value (perpetuity growth model)
        terminal_growth_rate = 0.03  # 3% perpetual growth
        if discount_rate > terminal_growth_rate and projected_fcf > 0:
            terminal_value = (projected_fcf * (1 + terminal_growth_rate)) / (discount_rate - terminal_growth_rate)
            present_terminal_value = terminal_value / ((1 + discount_rate) ** 10)
            intrinsic_value += present_terminal_value

        # Get shares outstanding to calculate per-share value
        latest_metrics = metrics[0]
        if latest_metrics.get('ordinary_shares_number') and latest_metrics['ordinary_shares_number'] > 0:
            shares_outstanding = latest_metrics['ordinary_shares_number']
            intrinsic_value_per_share = intrinsic_value / shares_outstanding
            return intrinsic_value_per_share

        return 0

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
        analysis['type'] = 'valuation_analysis'
        analysis['title'] = f'Valuation analysis'

        analysis_data['valuation_analysis'] = analysis
        ai_message = AIMessage(content=self.get_markdown(analysis))
        return {
            "context": context,
            "messages": [
                ai_message
            ]
        }