from common.agent_state import AgentState
from langchain.schema import AIMessage
from langchain_core.callbacks import dispatch_custom_event
from langchain_core.runnables import RunnableConfig
from typing import Dict, Any

import time
from common import markdown
from langgraph.types import StreamWriter


class IntrinsicValueAnalysis():
    def __init__(self, options: Dict[str, Any]):
        self.options = options

    
    def analyze(self, metrics: list) -> dict[str, any]:
        """Calculate intrinsic value using a simplified approach aligned with Phil Fisher's methodology."""
        result = {"intrinsic_value": None, "details": []}
        if not metrics or len(metrics) < 2:
            result["details"].append('Insufficient historical data (need at least 2 years)')
            return result

        latest_metrics = metrics[0]
        previous_metrics = metrics[1]

        intrinsic_value = None
        reasoning = []

        # Calculate earnings growth rate from net income data
        earnings_growth = None
        if (latest_metrics.get('net_income') and previous_metrics and 
            previous_metrics.get('net_income') and previous_metrics.get('net_income') > 0):
            earnings_growth = (latest_metrics.get('net_income') - previous_metrics.get('net_income')) / previous_metrics.get('net_income')

        # Phil Fisher's approach to valuation considers growth potential and quality
        # But we can still calculate a simple intrinsic value for reference
        
        # Simple dividend discount model (for stable companies) or earnings model
        earnings_per_share = None
        shares = latest_metrics.get('ordinary_shares_number')
        net_income = latest_metrics.get('net_income')
        
        if shares and shares > 0 and net_income:
            earnings_per_share = net_income / shares
        
        # Calculate a simple intrinsic value based on earnings and growth
        if earnings_per_share and earnings_growth:
            # Simple approach: Project earnings for 5 years and apply a P/E ratio
            projected_eps = earnings_per_share * (1 + earnings_growth) ** 5
            # Apply a reasonable P/E ratio (could be based on growth rate)
            target_pe = min(30, max(15, earnings_growth * 100 * 0.8))  # Cap P/E at 30, floor at 15
            projected_price = projected_eps * target_pe
            # Discount back to present value (assume 10% discount rate)
            intrinsic_value = projected_price / (1.10 ** 5)
            
            reasoning.append(f"EPS: ${earnings_per_share:.2f}, Growth: {earnings_growth:.1%}")
            reasoning.append(f"Projected EPS in 5 years: ${projected_eps:.2f}")
            reasoning.append(f"Target P/E: {target_pe:.1f}")
            reasoning.append(f"Projected price: ${projected_price:.2f}")
            reasoning.append(f"Intrinsic value: ${intrinsic_value:.2f}")
        else:
            reasoning.append("Insufficient data for intrinsic value calculation")

        # Alternative approach: Simple P/E based valuation
        pe_ratio = latest_metrics.get('price_to_earnings_ratio')
        if earnings_per_share and pe_ratio:
            current_price = earnings_per_share * pe_ratio
            reasoning.append(f"Current price based on TTM EPS: ${current_price:.2f}")
        else:
            reasoning.append("Unable to determine current price from EPS and P/E")

        # Check if we can calculate a book value-based intrinsic value
        if (latest_metrics.get('total_assets') and latest_metrics.get('total_liabilities') and 
            latest_metrics.get('ordinary_shares_number')):
            
            book_value = latest_metrics['total_assets'] - latest_metrics['total_liabilities']
            shares = latest_metrics['ordinary_shares_number']
            if shares > 0:
                book_value_per_share = book_value / shares
                reasoning.append(f"Book value per share: ${book_value_per_share:.2f}")
                
                # If no other intrinsic value calculated, use book value as floor
                if intrinsic_value is None:
                    intrinsic_value = book_value_per_share
                    reasoning.append("Using book value as intrinsic value floor")

        # Add margin of safety calculation
        market_cap = latest_metrics.get('market_cap')
        if intrinsic_value and market_cap and shares and shares > 0:
            current_price = market_cap / shares
            margin_of_safety = (intrinsic_value - current_price) / current_price if current_price > 0 else 0
            reasoning.append(f"Current price: ${current_price:.2f}")
            reasoning.append(f"Margin of safety: {margin_of_safety:.1%}")
        else:
            reasoning.append("Unable to calculate margin of safety")

        result["intrinsic_value"] = intrinsic_value
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
        analysis['type'] = 'intrinsic_value_analysis'
        analysis['title'] = f'Intrinsic Value Analysis'

        analysis_data['intrinsic_value_analysis'] = analysis
        ai_message = AIMessage(content=self.get_markdown(analysis))
        return {
            "context": context,
            "messages": [
                ai_message
            ]
        }