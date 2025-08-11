"""
This is the main entry point for the Stanley Druckenmiller agent.
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
from agents.stanley_druckenmiller.macro_analysis import MacroAnalysis
from agents.stanley_druckenmiller.global_market_analysis import GlobalMarketAnalysis
from agents.stanley_druckenmiller.adaptive_strategy_analysis import AdaptiveStrategyAnalysis
from agents.stanley_druckenmiller.risk_analysis import RiskAnalysis
from agents.stanley_druckenmiller.valuation_analysis import ValuationAnalysis
from agents.stanley_druckenmiller.flexibility_analysis import FlexibilityAnalysis

from nodes.next_step_suggestions import NextStepSuggestions

from nodes.ticker_search import TickerSearch
from typing_extensions import Literal
from common import markdown
from common.dataset import Dataset

next_step_suggestions_node = NextStepSuggestions({})
macro_analysis_node = MacroAnalysis({})
global_market_analysis_node = GlobalMarketAnalysis({})
adaptive_strategy_analysis_node = AdaptiveStrategyAnalysis({})
risk_analysis_node = RiskAnalysis({})
valuation_analysis_node = ValuationAnalysis({})
flexibility_analysis_node = FlexibilityAnalysis({})

async def start_analysis(state: AgentState, config: RunnableConfig):
    
    end_date = state.get('action').get('parameters').get('end_date')
    end_date = end_date if end_date else time.strftime("%Y-%m-%d")

    context = state.get('context')
    ticker = context.get('current_task').get('ticker')
    
    # Create dataset client
    dataset_client = Dataset(config)
    
    # Get required financial metrics and items for Stanley Druckenmiller analysis
    metrics = dataset_client.get_financial_items(ticker.get('symbol'), [
        "return_on_equity", "debt_to_equity", "operating_margin", "current_ratio", 
        "return_on_invested_capital", "asset_turnover", "market_cap", "beta",
        "price_to_earnings_ratio", "enterprise_value", "free_cash_flow", "ebit",
        "interest_expense", "capital_expenditure", "depreciation_and_amortization",
        "ordinary_shares_number", "total_assets", "total_liabilities", "stockholders_equity",
        "net_income", "revenue", "gross_profit", "gross_margin",
        "dividends_and_other_cash_distributions", "issuance_or_purchase_of_equity_shares"
    ], end_date, period="yearly")
    
    # Get additional data for macro analysis
    prices = dataset_client.get_prices(ticker.get('symbol'), 
                                      time.strftime("%Y-%m-%d", time.localtime(time.time() - 365*24*60*60)), 
                                      end_date)
    
    context['metrics'] = metrics
    context['prices'] = prices
    return {
        'context': context,
        'messages':[AIMessage(content=markdown.to_h2('Analysis for '+ ticker.get('symbol')))]
    }

async def end_analysis(state: AgentState, config: RunnableConfig):
    context = state.get('context')
    ticker = context.get('current_task').get('ticker')
    analysis_data = context.get('analysis_data')

    # Calculate total score
    total_score = (
        analysis_data.get('macro_analysis').get("score") + 
        analysis_data.get('global_market_analysis').get("score") + 
        analysis_data.get('adaptive_strategy_analysis').get("score") + 
        analysis_data.get('risk_analysis').get("score") +
        analysis_data.get('valuation_analysis').get("score") +
        analysis_data.get('flexibility_analysis').get("score")
    )
    
    # Update max possible score calculation
    max_possible_score = (
        analysis_data.get('macro_analysis').get("max_score") + 
        analysis_data.get('global_market_analysis').get("max_score") + 
        analysis_data.get('adaptive_strategy_analysis').get("max_score") + 
        analysis_data.get('risk_analysis').get("max_score") +
        analysis_data.get('valuation_analysis').get("max_score") +
        analysis_data.get('flexibility_analysis').get("max_score")
    )

    analysis_data['total_score'] = total_score
    analysis_data['max_possible_score'] = max_possible_score

    messages = [
            (
                "system",
                """You are Stanley Druckenmiller, one of the most successful macro investors in history. Analyze investment opportunities using my proven methodology developed through decades of managing the Soros Fund and other investments:

                MY CORE PRINCIPLES:
                1. Macro Analysis First: "The big money is made in the big moves." Understand the macroeconomic environment before analyzing individual companies.
                2. Global Market Perspective: "The world is one big market." Look for opportunities across all asset classes and geographies.
                3. Adaptive Strategy: "The markets are constantly changing." Flexibility and adaptability are more important than being right.
                4. Risk Management: "Don't worry about being right. Worry about the size of your losses." Position sizing and risk control are paramount.
                5. Conviction and Size: "You've got to put size behind your best ideas." When you have high conviction, commit significant capital.
                6. Market Timing: "It's not whether you're right or wrong, but how much money you make when you're right and how much you lose when you're wrong."
                7. Information Edge: "The key is to be early, but not too early." Wait for confirming data before making major moves.

                MY INVESTMENT APPROACH:
                - Start with macroeconomic analysis (interest rates, inflation, GDP growth, currency movements)
                - Identify global market trends and intermarket relationships
                - Adapt investment strategy based on changing market conditions
                - Focus on risk management and position sizing
                - Look for asymmetric opportunities with limited downside
                - Consider valuation but don't let it prevent you from participating in major trends
                - Be willing to change positions quickly when the thesis changes

                MY LANGUAGE & STYLE:
                - Use confident, analytical language with a focus on market dynamics
                - Reference macroeconomic factors and global market trends
                - Show understanding of market psychology and intermarket relationships
                - Be decisive but acknowledge uncertainty and risk
                - Express conviction when appropriate but remain flexible
                - Use specific examples of market moves and economic factors

                CONFIDENCE LEVELS:
                - 90-100%: Clear macro trend with strong conviction and asymmetric risk/reward
                - 70-89%: Favorable macro environment with good risk/reward profile
                - 50-69%: Mixed signals or moderate conviction, position accordingly
                - 30-49%: Unfavorable macro environment or high uncertainty
                - 10-29%: Strong headwinds or significant risks outweigh potential rewards

                Remember: The goal is not to be right all the time, but to make money consistently over time by focusing on the big moves and managing risk effectively.
                """,
            ),
            (
                "human",
                f"""Analyze this investment opportunity for {ticker.get('symbol')} ({ticker.get('short_name')}):

                COMPREHENSIVE ANALYSIS DATA:
                {analysis_data}

                Please provide your investment decision in exactly this JSON format, notice to use 'AnalysisResult' before json:
                ```AnalysisResult
                {{
                  "signal": "bullish" | "bearish" | "neutral",
                  "confidence": float between 0 and 100
                }}
                ```
                then provide a detailed reasoning for your decision.

                In your reasoning, be specific about:
                1. Your macroeconomic assessment and how it affects this investment
                2. Global market trends and intermarket relationships that impact this opportunity
                3. Your adaptive strategy approach for this specific situation
                4. Risk assessment and position sizing considerations
                5. Valuation relative to the macro backdrop
                6. Flexibility factors and potential changes in thesis
                7. How this fits within your overall portfolio strategy

                Write as Stanley Druckenmiller would speak - with analytical precision, market awareness, and specific references to the data provided.
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

workflow.add_node("macro_analysis", macro_analysis_node)
workflow.add_node("global_market_analysis", global_market_analysis_node)
workflow.add_node("adaptive_strategy_analysis", adaptive_strategy_analysis_node)
workflow.add_node("risk_analysis", risk_analysis_node)
workflow.add_node("valuation_analysis", valuation_analysis_node)
workflow.add_node("flexibility_analysis", flexibility_analysis_node)

workflow.add_node("end_analysis", end_analysis)

workflow.add_edge("start_analysis", "macro_analysis")
workflow.add_edge("macro_analysis", "global_market_analysis")
workflow.add_edge("global_market_analysis", "adaptive_strategy_analysis")
workflow.add_edge("adaptive_strategy_analysis", "risk_analysis")
workflow.add_edge("risk_analysis", "valuation_analysis")
workflow.add_edge("valuation_analysis", "flexibility_analysis")
workflow.add_edge("flexibility_analysis", "end_analysis")

workflow.set_entry_point("start_analysis")
workflow.set_finish_point("end_analysis")
# Compile the workflow graph
agent = workflow.compile()