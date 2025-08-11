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
        """Analyze company valuation based on Peter Lynch's PEG ratio approach."""
        result = {"score": 0, "max_score": 10, "details": []}
        if not metrics or len(metrics) < 2:
            result["details"].append('Insufficient historical data (need at least 2 years)')
            return result

        latest_metrics = metrics[0]
        previous_metrics = metrics[1]

        score = 0
        reasoning = []

        # Calculate earnings growth rate from net income data for PEG ratio
        earnings_growth = None
        if (latest_metrics.get('net_income') and previous_metrics and 
            previous_metrics.get('net_income') and previous_metrics.get('net_income') > 0):
            earnings_growth = (latest_metrics.get('net_income') - previous_metrics.get('net_income')) / previous_metrics.get('net_income')
        
        # Calculate PEG ratio - Lynch's key metric: PEG < 1.0 is attractive
        pe_ratio = latest_metrics.get('price_to_earnings_ratio')
        
        if pe_ratio and earnings_growth and earnings_growth > 0:
            peg_ratio = pe_ratio / (earnings_growth * 100)  # Convert growth to percentage
            
            if peg_ratio < 0.5:
                score += 3
                reasoning.append(f"Exceptional value: PEG ratio of {peg_ratio:.2f} (<0.5)")
            elif peg_ratio < 1.0:
                score += 2
                reasoning.append(f"Attractive value: PEG ratio of {peg_ratio:.2f} (<1.0)")
            elif peg_ratio < 1.5:
                score += 1
                reasoning.append(f"Fair value: PEG ratio of {peg_ratio:.2f} (<1.5)")
            else:
                reasoning.append(f"Overvalued: PEG ratio of {peg_ratio:.2f} (>1.5)")
        else:
            reasoning.append("Insufficient data to calculate PEG ratio")

        # Check P/E ratio relative to growth
        if pe_ratio and earnings_growth:
            if pe_ratio < (earnings_growth * 100):  # P/E should roughly equal growth rate in percentage terms
                score += 2
                reasoning.append(f"P/E aligns with growth: P/E {pe_ratio:.1f} vs. growth {earnings_growth:.1%}")
            else:
                reasoning.append(f"P/E exceeds growth: P/E {pe_ratio:.1f} vs. growth {earnings_growth:.1%}")
        else:
            reasoning.append("Insufficient data for P/E to growth comparison")

        # Check market capitalization relative to growth potential
        market_cap = latest_metrics.get('market_cap')
        if market_cap and earnings_growth:
            if market_cap < 10000000000 and earnings_growth > 0.20:  # Small cap with high growth
                score += 2
                reasoning.append("Small-cap high-growth opportunity")
            elif market_cap < 50000000000 and earnings_growth > 0.15:  # Mid cap with good growth
                score += 1
                reasoning.append("Mid-cap growth opportunity")
            elif market_cap > 100000000000 and earnings_growth < 0.10:  # Large cap with slow growth
                reasoning.append("Large-cap slow growth - may be overvalued")
            else:
                reasoning.append("Market cap appropriate for growth profile")
        else:
            reasoning.append("Insufficient data for market cap analysis")

        # Check price relative to book value
        if (latest_metrics.get('total_assets') and latest_metrics.get('total_liabilities') and 
            latest_metrics.get('ordinary_shares_number') and market_cap):
            
            book_value = latest_metrics['total_assets'] - latest_metrics['total_liabilities']
            shares = latest_metrics['ordinary_shares_number']
            if shares > 0:
                book_value_per_share = book_value / shares
                price_per_share = market_cap / shares if market_cap > 0 else 0
                
                if price_per_share > 0 and book_value_per_share > 0:
                    price_to_book = price_per_share / book_value_per_share
                    
                    if price_to_book < 1.5:
                        score += 1
                        reasoning.append(f"Discount to book value (P/B: {price_to_book:.2f})")
                    elif price_to_book < 3.0:
                        reasoning.append(f"Fair valuation relative to book (P/B: {price_to_book:.2f})")
                    else:
                        reasoning.append(f"Premium to book value (P/B: {price_to_book:.2f})")
                else:
                    reasoning.append("Unable to calculate price-to-book ratio")
            else:
                reasoning.append("Invalid share count for book value calculation")
        else:
            reasoning.append("Insufficient data for book value analysis")

        # Check if the stock is trading at a discount to intrinsic value (if available from other analysis)
        reasoning.append("Valuation analysis complete - PEG ratio is the primary Lynch metric")

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
        analysis['type'] = 'valuation_analysis'
        analysis['title'] = f'Valuation Analysis'

        analysis_data['valuation_analysis'] = analysis
        ai_message = AIMessage(content=self.get_markdown(analysis))
        return {
            "context": context,
            "messages": [
                ai_message
            ]
        }