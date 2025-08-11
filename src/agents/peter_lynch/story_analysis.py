from common.agent_state import AgentState
from langchain.schema import AIMessage
from langchain_core.callbacks import dispatch_custom_event
from langchain_core.runnables import RunnableConfig
from typing import Dict, Any

import time
from common import markdown
from langgraph.types import StreamWriter


class StoryAnalysis():
    def __init__(self, options: Dict[str, Any]):
        self.options = options

    
    def analyze(self, metrics: list, ticker_data: dict = None) -> dict[str, any]:
        """Analyze the business story based on Peter Lynch's approach."""
        result = {"score": 0, "max_score": 10, "details": [], "business_category": "Unknown"}
        if not metrics or len(metrics) < 2:
            result["details"].append('Insufficient historical data (need at least 2 years)')
            return result

        latest_metrics = metrics[0]
        previous_metrics = metrics[1]
        ticker_info = ticker_data or {}

        score = 0
        reasoning = []
        business_category = "Unknown"

        # Calculate earnings growth rate from net income data
        earnings_growth = None
        if (latest_metrics.get('net_income') and previous_metrics and 
            previous_metrics.get('net_income') and previous_metrics.get('net_income') > 0):
            earnings_growth = (latest_metrics.get('net_income') - previous_metrics.get('net_income')) / previous_metrics.get('net_income')

        # Determine business category based on growth and size - Lynch's classification system
        market_cap = latest_metrics.get('market_cap')
        
        if market_cap and earnings_growth:
            if market_cap < 5000000000 and earnings_growth > 0.20:
                business_category = "Fast Grower"
                score += 3
                reasoning.append("Small-cap high-growth company - Fast Grower category")
            elif market_cap < 50000000000 and earnings_growth > 0.15:
                business_category = "Fast Grower"
                score += 2
                reasoning.append("Mid-cap growth company - Fast Grower category")
            elif market_cap > 100000000000 and earnings_growth < 0.10:
                business_category = "Slow Grower"
                score += 1
                reasoning.append("Large-cap stable company - Slow Grower category")
            elif market_cap > 50000000000 and earnings_growth > 0.10:
                business_category = "Stalwart"
                score += 2
                reasoning.append("Large-cap growth company - Stalwart category")
            else:
                business_category = "Cyclical"
                reasoning.append("Company with cyclical characteristics")
        else:
            reasoning.append("Insufficient data to categorize business")

        # Check for turnaround potential
        net_income = latest_metrics.get('net_income', 0)
        if len(metrics) >= 3:
            previous_income = metrics[2].get('net_income', 0)
            if net_income > 0 and previous_income < 0:
                business_category = "Turnaround"
                score += 2
                reasoning.append("Recent profitability turnaround - Turnaround category")
            elif net_income < 0 and previous_income < 0:
                business_category = "Asset Play"
                score += 1
                reasoning.append("Unprofitable but may have asset value - Asset Play category")

        # Check for asset play potential (trading below book value)
        if (latest_metrics.get('total_assets') and latest_metrics.get('total_liabilities') and 
            latest_metrics.get('ordinary_shares_number') and market_cap):
            
            book_value = latest_metrics['total_assets'] - latest_metrics['total_liabilities']
            shares = latest_metrics['ordinary_shares_number']
            if shares > 0:
                book_value_per_share = book_value / shares
                price_per_share = market_cap / shares if market_cap > 0 else 0
                
                if price_per_share > 0 and book_value_per_share > 0 and price_per_share < book_value_per_share:
                    business_category = "Asset Play"
                    score += 2
                    reasoning.append("Trading below book value - Asset Play category")
                elif price_per_share > 0 and book_value_per_share > 0:
                    reasoning.append("Trading above book value")

        # Check for strong brand or consumer product potential
        sector = ticker_info.get('sector', '').lower() if ticker_info else ''
        industry = ticker_info.get('industry', '').lower() if ticker_info else ''
        
        consumer_sectors = ['consumer defensive', 'consumer cyclical', 'communication services']
        if any(s in sector for s in consumer_sectors):
            score += 1
            reasoning.append("Consumer-focused business - potential for strong brand loyalty")
        
        # Check for simplicity of business model
        # Lynch prefers businesses that are easy to understand
        if latest_metrics.get('operating_margin') and latest_metrics['operating_margin'] > 0.10:
            score += 1
            reasoning.append("Business with clear profitability - easy to understand model")
        
        # Check for insider ownership (proxy for management alignment)
        reasoning.append("Business story analysis complete - Lynch focuses on understandable businesses")

        result["score"] = score
        result["details"] = reasoning
        result["business_category"] = business_category
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
        ticker = context.get('current_task', {}).get('ticker', {})
        analysis = self.analyze(metrics, ticker)
        analysis['type'] = 'story_analysis'
        analysis['title'] = f'Story Analysis'

        analysis_data['story_analysis'] = analysis
        ai_message = AIMessage(content=self.get_markdown(analysis))
        return {
            "context": context,
            "messages": [
                ai_message
            ]
        }