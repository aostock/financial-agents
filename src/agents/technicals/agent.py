"""
This is the main entry point for the technicals agent.
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
from agents.technicals.trend_analysis import TrendAnalysis
from agents.technicals.mean_reversion_analysis import MeanReversionAnalysis
from agents.technicals.momentum_analysis import MomentumAnalysis
from agents.technicals.volatility_analysis import VolatilityAnalysis
from agents.technicals.statistical_arbitrage_analysis import StatisticalArbitrageAnalysis

from nodes.next_step_suggestions import NextStepSuggestions

from nodes.ticker_search import TickerSearch
from typing_extensions import Literal
from common import markdown
from common.dataset import Dataset

next_step_suggestions_node = NextStepSuggestions({})
trend_analysis_node = TrendAnalysis({})
mean_reversion_analysis_node = MeanReversionAnalysis({})
momentum_analysis_node = MomentumAnalysis({})
volatility_analysis_node = VolatilityAnalysis({})
statistical_arbitrage_analysis_node = StatisticalArbitrageAnalysis({})

async def start_analysis(state: AgentState, config: RunnableConfig):
    
    end_date = state.get('action').get('parameters').get('end_date')
    end_date = end_date if end_date else time.strftime("%Y-%m-%d")

    context = state.get('context')
    ticker = context.get('current_task').get('ticker')
    
    # Create dataset client
    dataset_client = Dataset(config)
    
    # Get price data for technical analysis
    start_date = time.strftime("%Y-%m-%d", time.localtime(time.time() - 365*24*60*60))  # 1 year of data
    prices = dataset_client.get_prices(ticker.get('symbol'), start_date, end_date)
    
    context['prices'] = prices
    return {
        'context': context,
        'messages':[AIMessage(content=markdown.to_h2('Technical Analysis for '+ ticker.get('symbol')))]
    }

async def end_analysis(state: AgentState, config: RunnableConfig):
    context = state.get('context')
    ticker = context.get('current_task').get('ticker')
    analysis_data = context.get('analysis_data')

    # Calculate total score
    total_score = (
        analysis_data.get('trend_analysis').get("score") + 
        analysis_data.get('mean_reversion_analysis').get("score") + 
        analysis_data.get('momentum_analysis').get("score") + 
        analysis_data.get('volatility_analysis').get("score") +
        analysis_data.get('statistical_arbitrage_analysis').get("score")
    )
    
    # Update max possible score calculation
    max_possible_score = (
        analysis_data.get('trend_analysis').get("max_score") + 
        analysis_data.get('mean_reversion_analysis').get("max_score") + 
        analysis_data.get('momentum_analysis').get("max_score") + 
        analysis_data.get('volatility_analysis').get("max_score") +
        analysis_data.get('statistical_arbitrage_analysis').get("max_score")
    )

    analysis_data['total_score'] = total_score
    analysis_data['max_possible_score'] = max_possible_score

    messages = [
            (
                "system",
                """You are a world-class technical analyst. Analyze market opportunities using proven technical analysis methodologies developed over decades of trading:

                YOUR CORE PRINCIPLES:
                1. Trend Following: "The trend is your friend." Identify and follow the dominant market trends across multiple timeframes.
                2. Mean Reversion: "Prices tend to revert to the mean." Identify overbought and oversold conditions for contrarian opportunities.
                3. Momentum: "Ride the strongest moves." Follow assets with strong price momentum and volume confirmation.
                4. Volatility Analysis: "Volatility precedes opportunity." Understand volatility regimes and how they create trading opportunities.
                5. Statistical Arbitrage: "Markets exhibit predictable patterns." Use statistical measures to identify trading edges.

                YOUR TECHNICAL APPROACH:
                - Multi-timeframe analysis for trend confirmation
                - Multiple indicators for signal validation
                - Risk management through position sizing and stop-losses
                - Adaptive strategies that change with market conditions
                - Focus on high-probability setups with favorable risk/reward

                YOUR LANGUAGE & STYLE:
                - Use precise, technical language with specific indicator values
                - Reference concrete price levels, support/resistance zones
                - Show understanding of market structure and price action
                - Be decisive but acknowledge uncertainty and risk
                - Express conviction when appropriate but remain flexible
                - Use specific examples of technical patterns and setups

                CONFIDENCE LEVELS:
                - 90-100%: Clear technical setup with multiple confirming factors
                - 70-89%: Strong technical signals with good risk/reward profile
                - 50-69%: Mixed signals or moderate conviction, position accordingly
                - 30-49%: Weak technical signals or high uncertainty
                - 10-29%: Poor technical setup or significant risks outweigh potential rewards

                Remember: Technical analysis is about identifying high-probability setups and managing risk effectively. Always consider the broader market context and adapt to changing conditions.
                """,
            ),
            (
                "human",
                f"""Analyze this technical opportunity for {ticker.get('symbol')} ({ticker.get('short_name')}):

                COMPREHENSIVE TECHNICAL ANALYSIS DATA:
                {analysis_data}

                Please provide your trading decision in exactly this JSON format, notice to use 'AnalysisResult' before json:
                ```AnalysisResult
                {{
                  "signal": "bullish" | "bearish" | "neutral",
                  "confidence": float between 0 and 100
                }}
                ```
                then provide a detailed reasoning for your decision.

                In your reasoning, be specific about:
                1. Trend analysis and multiple timeframe confirmation
                2. Mean reversion opportunities and overbought/oversold conditions
                3. Momentum signals and volume confirmation
                4. Volatility regime and how it affects trading opportunities
                5. Statistical arbitrage signals and market patterns
                6. Key support/resistance levels and price targets
                7. Risk management considerations and stop-loss levels

                Write as a professional technical analyst would speak - with precision, technical knowledge, and specific references to the data provided.
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

workflow.add_node("trend_analysis", trend_analysis_node)
workflow.add_node("mean_reversion_analysis", mean_reversion_analysis_node)
workflow.add_node("momentum_analysis", momentum_analysis_node)
workflow.add_node("volatility_analysis", volatility_analysis_node)
workflow.add_node("statistical_arbitrage_analysis", statistical_arbitrage_analysis_node)

workflow.add_node("end_analysis", end_analysis)

workflow.add_edge("start_analysis", "trend_analysis")
workflow.add_edge("trend_analysis", "mean_reversion_analysis")
workflow.add_edge("mean_reversion_analysis", "momentum_analysis")
workflow.add_edge("momentum_analysis", "volatility_analysis")
workflow.add_edge("volatility_analysis", "statistical_arbitrage_analysis")
workflow.add_edge("statistical_arbitrage_analysis", "end_analysis")

workflow.set_entry_point("start_analysis")
workflow.set_finish_point("end_analysis")
# Compile the workflow graph
agent = workflow.compile()