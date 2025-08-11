from common.agent_state import AgentState
from langchain.schema import AIMessage
from langchain_core.callbacks import dispatch_custom_event
from langchain_core.runnables import RunnableConfig
from typing import Dict, Any

import time
from common import markdown
from langgraph.types import StreamWriter


class BusinessUnderstandingAnalysis():
    def __init__(self, options: Dict[str, Any]):
        self.options = options

    
    def analyze(self, metrics: list, ticker_data: dict = None) -> dict[str, any]:
        """Analyze business understanding based on Peter Lynch's 'invest in what you know' principle."""
        result = {"score": 0, "max_score": 10, "details": []}
        if not metrics or len(metrics) < 2:
            result["details"].append('Insufficient historical data (need at least 2 years)')
            return result

        latest_metrics = metrics[0]
        previous_metrics = metrics[1]
        ticker_info = ticker_data or {}

        score = 0
        reasoning = []

        # Calculate earnings growth to understand business growth pattern
        earnings_growth = None
        if (latest_metrics.get('net_income') and previous_metrics and 
            previous_metrics.get('net_income') and previous_metrics.get('net_income') > 0):
            earnings_growth = (latest_metrics.get('net_income') - previous_metrics.get('net_income')) / previous_metrics.get('net_income')
        
        # Lynch prefers businesses that are simple and understandable
        simple_business_sectors = [
            'consumer defensive', 'consumer cyclical', 'utilities', 'financial services',
            'healthcare', 'communication services', 'industrials'
        ]
        
        complex_business_sectors = [
            'technology', 'semiconductors', 'biotechnology', 'pharmaceuticals'
        ]
        
        if any(s in sector for s in simple_business_sectors):
            score += 2
            reasoning.append(f"Simple, understandable business model ({sector.title()})")
        elif any(s in sector for s in complex_business_sectors):
            reasoning.append(f"Complex business model ({sector.title()}) - requires specialized knowledge")
        else:
            score += 1
            reasoning.append(f"Moderately complex business ({sector.title()})")

        # Check if this is a product or service the average person uses
        consumer_products_keywords = [
            'food', 'beverage', 'retail', 'restaurant', 'consumer', 'entertainment',
            'media', 'apparel', 'household', 'personal', 'cosmetics', 'pharmacy'
        ]
        
        if any(keyword in industry for keyword in consumer_products_keywords) or any(keyword in short_name for keyword in consumer_products_keywords):
            score += 2
            reasoning.append("Consumer product/service that people use regularly")
        else:
            reasoning.append("Business may not be part of daily consumer experience")

        # Check business model simplicity based on financial metrics
        # Simple businesses tend to have stable margins and straightforward operations
        operating_margin = latest_metrics.get('operating_margin')
        if operating_margin:
            if operating_margin > 0.15:
                score += 1
                reasoning.append("High, stable operating margins suggest simple business model")
            elif operating_margin > 0.05:
                reasoning.append("Moderate operating margins")
            else:
                reasoning.append("Low operating margins may indicate complex operations")

        # Check for recurring revenue or subscription models
        revenue = latest_metrics.get('revenue')
        if revenue and revenue > 0:
            reasoning.append("Business generates revenue - fundamental requirement")

        # Check if the company has a strong brand or competitive position
        # Lynch looks for companies with some form of competitive advantage
        market_cap = latest_metrics.get('market_cap')
        if market_cap and market_cap > 10000000000:  # Large market cap suggests brand strength
            score += 1
            reasoning.append("Large market capitalization suggests established brand/company")
        elif market_cap and market_cap < 1000000000:  # Small market cap but high growth could be emerging brand
            growth = latest_metrics.get('earnings_growth')
            if growth and growth > 0.25:
                score += 1
                reasoning.append("Small company with high growth - potential emerging brand")
        
        # Check for business stability
        if len(metrics) >= 5:
            revenues = [m.get('revenue') for m in metrics[:5] if m.get('revenue') is not None]
            if len(revenues) >= 5:
                # Check revenue stability
                avg_revenue = sum(revenues) / len(revenues)
                volatility = sum(abs(r - avg_revenue) for r in revenues) / len(revenues) if avg_revenue != 0 else 0
                normalized_volatility = volatility / avg_revenue if avg_revenue != 0 else 0
                
                if normalized_volatility < 0.1:
                    score += 1
                    reasoning.append("Stable revenue base - predictable business")
                elif normalized_volatility < 0.2:
                    reasoning.append("Moderately stable revenue")
                else:
                    reasoning.append("Highly volatile revenue - unpredictable business")

        # Check for management alignment with shareholders
        reasoning.append("Business understanding analysis complete - Lynch prefers simple, understandable businesses")

        # Check for 'boring' but profitable businesses
        boring_but_profitable = [
            'utilities', 'consumer staples', 'banking', 'insurance'
        ]
        if any(s in sector for s in boring_but_profitable):
            score += 1
            reasoning.append("Boring but profitable business - Lynch's sweet spot")

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
        ticker = context.get('current_task', {}).get('ticker', {})
        analysis = self.analyze(metrics, ticker)
        analysis['type'] = 'business_understanding_analysis'
        analysis['title'] = f'Business Understanding Analysis'

        analysis_data['business_understanding_analysis'] = analysis
        ai_message = AIMessage(content=self.get_markdown(analysis))
        return {
            "context": context,
            "messages": [
                ai_message
            ]
        }