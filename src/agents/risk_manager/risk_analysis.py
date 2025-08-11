from common.agent_state import AgentState
from langchain.schema import AIMessage
from langchain_core.callbacks import dispatch_custom_event
from langchain_core.runnables import RunnableConfig
from typing import Dict, Any
import time
from common import markdown
from common.dataset import Dataset
from langgraph.types import StreamWriter


class RiskAnalysis():
    def __init__(self, options: Dict[str, Any]):
        self.options = options

    
    def analyze(self, prices: list, portfolio: dict, current_ticker: str, config: RunnableConfig) -> dict[str, any]:
        """
        Controls position sizing based on real-world risk factors for the current ticker.
        """
        result = {"score": 0, "max_score": 10, "details": [], "position_limit": 0, "remaining_position_limit": 0}
        
        if not prices:
            result["details"].append('No price data available for risk analysis')
            return result

        # Get current price
        current_price = prices[0].get('close') if prices else 0
        if not current_price:
            result["details"].append('No current price data available')
            return result

        # Calculate total portfolio value based on current market prices (Net Liquidation Value)
        total_portfolio_value = portfolio.get("cash", 0.0)
        dataset_client = Dataset(config)
        # Add market value of existing positions
        for ticker, position in portfolio.get("positions", {}).items():
            # Get price for this ticker


            ticker_prices = dataset_client.get_prices(ticker, time.strftime("%Y-%m-%d"), time.strftime("%Y-%m-%d"))
            ticker_price = ticker_prices[0].get('close') if ticker_prices else 0
            
            if ticker_price:
                # Add market value of long positions
                total_portfolio_value += position.get("long", 0) * ticker_price
                # Subtract market value of short positions (since they represent borrowed shares we need to buy back)
                total_portfolio_value -= position.get("short", 0) * ticker_price
        
        # Calculate position limit (20% of total portfolio)
        position_limit = total_portfolio_value * 0.20
        
        # Calculate current market value of this position
        position = portfolio.get("positions", {}).get(current_ticker, {})
        long_value = position.get("long", 0) * current_price
        short_value = position.get("short", 0) * current_price
        current_position_value = abs(long_value - short_value)  # Use absolute exposure
        
        # Calculate remaining limit for this position
        remaining_position_limit = position_limit - current_position_value
        
        # Ensure we don't exceed available cash for long positions
        max_position_size = min(remaining_position_limit, portfolio.get("cash", 0))
        
        # For short positions, we also need to consider margin requirements
        # Assuming 50% margin requirement for short positions
        margin_requirement = 0.5
        available_margin = portfolio.get("cash", 0) / margin_requirement
        
        # For short positions, the limit is the minimum of:
        # 1. Remaining position limit
        # 2. Available margin
        max_short_position_size = min(remaining_position_limit, available_margin)
        
        # Final position limit is the maximum we can take based on position type
        final_position_limit = max(max_position_size, max_short_position_size)
        
        # Risk score based on position sizing
        if final_position_limit >= position_limit * 0.8:
            score = 10
        elif final_position_limit >= position_limit * 0.6:
            score = 8
        elif final_position_limit >= position_limit * 0.4:
            score = 6
        elif final_position_limit >= position_limit * 0.2:
            score = 4
        else:
            score = 2
            
        result["score"] = score
        result["max_score"] = 10
        result["position_limit"] = position_limit
        result["remaining_position_limit"] = final_position_limit
        result["details"] = [
            f"Total portfolio value: ${total_portfolio_value:,.2f}",
            f"Current position value: ${current_position_value:,.2f}",
            f"Position limit (20% of portfolio): ${position_limit:,.2f}",
            f"Remaining position limit: ${final_position_limit:,.2f}",
            f"Available cash: ${portfolio.get('cash', 0):,.2f}"
        ]
        
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
            
        prices = context.get('prices')
        portfolio = context.get('portfolio', {})
        ticker = context.get('current_task').get('ticker').get('symbol')
        
        analysis = self.analyze(prices, portfolio, ticker, config)
        analysis['type'] = 'risk_analysis'
        analysis['title'] = f'Risk analysis'

        analysis_data['risk_analysis'] = analysis
        ai_message = AIMessage(content=self.get_markdown(analysis))
        return {
            "context": context,
            "messages": [
                ai_message
            ]
        }