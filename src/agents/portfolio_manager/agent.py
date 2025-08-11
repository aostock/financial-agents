"""
This is the main entry point for the Portfolio Manager agent.
It defines the workflow graph, state, tools, nodes and edges.
"""

import time
from common.agent_state import AgentState
from common.util import get_dict_json
from langchain.tools import tool
from langchain_core.messages import AIMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.types import Command
from llm.llm_model import ainvoke

# Import analysis modules
from agents.portfolio_manager.portfolio_analysis import PortfolioAnalysis

from nodes.next_step_suggestions import NextStepSuggestions

from nodes.ticker_search import TickerSearch
from typing_extensions import Literal
from common import markdown
from common.dataset import Dataset

next_step_suggestions_node = NextStepSuggestions({})
portfolio_analysis_node = PortfolioAnalysis({})

async def start_analysis(state: AgentState, config: RunnableConfig):
    
    end_date = state.get('action').get('parameters').get('end_date')
    end_date = end_date if end_date else time.strftime("%Y-%m-%d")

    context = state.get('context')
    ticker = context.get('current_task').get('ticker')
    
    # Create dataset client
    dataset_client = Dataset(config)
    
    # Get required financial metrics and items for portfolio analysis
    metrics = dataset_client.get_financial_items(ticker.get('symbol'), [
        "return_on_equity", "debt_to_equity", "operating_margin", "current_ratio", 
        "return_on_invested_capital", "asset_turnover", "market_cap", "beta",
        "price_to_earnings_ratio", "enterprise_value", "free_cash_flow", "ebit",
        "interest_expense", "capital_expenditure", "depreciation_and_amortization",
        "ordinary_shares_number", "total_assets", "total_liabilities", "stockholders_equity",
        "net_income", "revenue", "gross_profit", "gross_margin"
    ], end_date, period="yearly")
    
    # Get additional data for portfolio analysis
    prices = dataset_client.get_prices(ticker.get('symbol'), end_date, end_date)
    info = dataset_client.get_info(ticker.get('symbol'))
    
    context['metrics'] = metrics
    context['prices'] = prices
    context['info'] = info
    
    return {
        'context': context,
        'messages':[AIMessage(content=markdown.to_h2('Portfolio Analysis for '+ ticker.get('symbol')))]
    }

async def end_analysis(state: AgentState, config: RunnableConfig):
    context = state.get('context')
    ticker = context.get('current_task').get('ticker')
    analysis_data = context.get('analysis_data')

    messages = [
            (
                "system",
                """You are a professional portfolio manager making final trading decisions based on comprehensive analysis.
                
                YOUR ROLE:
                You are responsible for making final trading decisions based on multiple analytical inputs. You manage an existing portfolio with current positions and must consider:
                1. Current portfolio positions (both long and short)
                2. Available cash for new investments
                3. Margin requirements for short positions
                4. Risk management constraints
                5. Signals from multiple analytical agents
                
                TRADING RULES:
                - For long positions:
                  * Only buy if you have available cash
                  * Only sell if you currently hold long shares of that ticker
                  * Sell quantity must be ≤ current long position shares
                  * Buy quantity must respect position limits
                
                - For short positions:
                  * Only short if you have available margin (position value × margin requirement)
                  * Only cover if you currently have short shares of that ticker
                  * Cover quantity must be ≤ current short position shares
                  * Short quantity must respect margin requirements
                
                AVAILABLE ACTIONS:
                - "buy": Open or add to long position
                - "sell": Close or reduce long position (only if you currently hold long shares)
                - "short": Open or add to short position
                - "cover": Close or reduce short position (only if you currently hold short shares)
                - "hold": Maintain current position without any changes (quantity should be 0 for hold)
                
                DECISION FRAMEWORK:
                1. Position Management:
                   - If you currently hold LONG shares of a ticker (long > 0), you can:
                     * HOLD: Keep your current position (quantity = 0)
                     * SELL: Reduce/close your long position (quantity = shares to sell)
                     * BUY: Add to your long position (quantity = additional shares to buy)
                     
                   - If you currently hold SHORT shares of a ticker (short > 0), you can:
                     * HOLD: Keep your current position (quantity = 0)
                     * COVER: Reduce/close your short position (quantity = shares to cover)
                     * SHORT: Add to your short position (quantity = additional shares to short)
                     
                   - If you currently hold NO shares of a ticker (long = 0, short = 0), you can:
                     * HOLD: Stay out of the position (quantity = 0)
                     * BUY: Open a new long position (quantity = shares to buy)
                     * SHORT: Open a new short position (quantity = shares to short)
                
                2. Risk Management:
                   - Consider both long and short opportunities based on signals
                   - Maintain appropriate risk management with both long and short exposure
                   - Respect position limits and margin requirements
                   - Diversify portfolio across different sectors and risk levels
                
                3. Signal Integration:
                   - Weight bullish signals more heavily for buy decisions
                   - Weight bearish signals more heavily for short/sell decisions
                   - Consider confidence levels in decision making
                   - Balance aggressive opportunities with conservative positions
                
                CONFIDENCE LEVELS:
                - 90-100%: High conviction trade with strong fundamentals and favorable technicals
                - 70-89%: Solid opportunity with good risk/reward profile
                - 50-69%: Neutral position or small position size for uncertain opportunities
                - 30-49%: Consider reducing or closing existing positions
                - 10-29%: Strong signal to exit or avoid position
                
                Remember: "The goal of portfolio management is not to maximize returns, but to optimize risk-adjusted returns while preserving capital." Focus on making prudent decisions that balance opportunity with risk management.
                """,
            ),
            (
                "human",
                f"""Based on the comprehensive analysis, make your trading decision for {ticker.get('symbol')} ({ticker.get('short_name')}):
                
                COMPREHENSIVE ANALYSIS DATA:
                {analysis_data}
                
                Current Portfolio Data:
                - Cash: $100,000.00
                - Current Positions: {{}}
                - Margin Requirement: 0.50
                - Total Margin Used: $0.00
                
                Current Price: ${{current_price}}
                Maximum Shares Allowed: {{max_shares}}
                
                Please provide your trading decision in exactly this JSON format, notice to use 'PortfolioDecision' before json:
                ```PortfolioDecision
                {{
                  "action": "buy" | "sell" | "short" | "cover" | "hold",
                  "quantity": integer,
                  "confidence": float between 0 and 100,
                  "reasoning": "string explaining your decision considering current position and risk management"
                }}
                ```
                
                In your reasoning, be specific about:
                1. Your assessment of the current position (if any) and whether to hold, add, or reduce
                2. The risk/reward profile of this opportunity
                3. How this decision fits within the overall portfolio strategy
                4. Any risk management considerations (position sizing, stop-loss levels, etc.)
                5. The confidence level in your decision and key factors driving it
                
                Make a prudent decision that balances opportunity with risk management.
                """,
            ),
        ]
    response = await ainvoke(messages, config, analyzer=True)
    
    return {
        "messages": response,
        "action": None,
    }



# Define the workflow graph
workflow = StateGraph(AgentState)
workflow.add_node("start_analysis", start_analysis)

workflow.add_node("portfolio_analysis", portfolio_analysis_node)

workflow.add_node("end_analysis", end_analysis)

workflow.add_edge("start_analysis", "portfolio_analysis")
workflow.add_edge("portfolio_analysis", "end_analysis")

workflow.set_entry_point("start_analysis")
workflow.set_finish_point("end_analysis")
# Compile the workflow graph
agent = workflow.compile()