"""
This is the main entry point for the Benjamin Graham agent.
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
from agents.ben_graham.earnings_stability_analysis import EarningsStabilityAnalysis
from agents.ben_graham.financial_strength_analysis import FinancialStrengthAnalysis
from agents.ben_graham.valuation_analysis import ValuationAnalysis

from nodes.next_step_suggestions import NextStepSuggestions
from nodes.ticker_search import TickerSearch
from typing_extensions import Literal
from common import markdown
from common.dataset import Dataset

next_step_suggestions_node = NextStepSuggestions({})
earnings_stability_analysis_node = EarningsStabilityAnalysis({})
financial_strength_analysis_node = FinancialStrengthAnalysis({})
valuation_analysis_node = ValuationAnalysis({})

async def start_analysis(state: AgentState, config: RunnableConfig):
    end_date = state.get('action').get('parameters').get('end_date')
    end_date = end_date if end_date else time.strftime("%Y-%m-%d")

    context = state.get('context')
    ticker = context.get('current_task').get('ticker')
    
    # Create dataset client
    dataset_client = Dataset(config)
    
    # Get required financial metrics and items for Graham analysis
    metrics = dataset_client.get_financial_items(ticker.get('symbol'), [
        "earnings_per_share", "revenue", "net_income", "book_value_per_share", 
        "total_assets", "total_liabilities", "current_assets", "current_liabilities",
        "dividends_and_other_cash_distributions", "outstanding_shares", "market_cap",
        "price_to_earnings_ratio"
    ], end_date, period="yearly")
    
    context['metrics'] = metrics
    return {
        'context': context,
        'messages': [AIMessage(content=markdown.to_h2('Benjamin Graham Analysis for ' + ticker.get('symbol')))]
    }

async def end_analysis(state: AgentState, config: RunnableConfig):
    context = state.get('context')
    ticker = context.get('current_task').get('ticker')
    analysis_data = context.get('analysis_data')

    # Calculate total score
    total_score = (
        analysis_data.get('earnings_stability_analysis').get("score") + 
        analysis_data.get('financial_strength_analysis').get("score") + 
        analysis_data.get('valuation_analysis').get("score")
    )
    
    # Update max possible score calculation
    max_possible_score = 15  # Total possible from the three analysis functions

    # Map total_score to signal
    signal = "neutral"
    confidence = 50.0
    
    if total_score >= 0.7 * max_possible_score:
        signal = "bullish"
        confidence = min(90.0, 70.0 + (total_score / max_possible_score) * 30)
    elif total_score <= 0.3 * max_possible_score:
        signal = "bearish"
        confidence = max(10.0, 30.0 - (total_score / max_possible_score) * 30)
    else:
        signal = "neutral"
        confidence = 40.0 + (total_score / max_possible_score) * 20

    analysis_data['total_score'] = total_score
    analysis_data['max_possible_score'] = max_possible_score
    analysis_data['signal'] = signal

    messages = [
        (
            "system",
            """You are Benjamin Graham, the father of value investing. Analyze investment opportunities using the principles you developed:

            YOUR CORE PRINCIPLES:
            1. Margin of Safety: Never pay more than the intrinsic value. Look for investments with a substantial margin of safety.
            2. Intrinsic Value: Estimate intrinsic value based on proven fundamentals - earnings, assets, dividends.
            3. Conservative Investing: Focus on financial strength (low debt, adequate liquidity) and stable earnings.
            4. Mr. Market: Treat market volatility as an opportunity, not a threat. Be fearful when others are greedy, and greedy when others are fearful.
            5. Diversification: Spread risk across multiple securities to protect against permanent capital loss.
            6. Investor vs. Speculator: Be an investor who analyzes facts and demands a margin of safety. Avoid speculation.

            YOUR VALUATION METHODOLOGY:
            1. Earnings Stability: Look for companies with consistently positive earnings over multiple years (ideally 5+).
            2. Financial Strength: Evaluate balance sheet strength (current ratio >= 2, low debt-to-equity).
            3. Net-Net Approach: (Current Assets - Total Liabilities) vs. Market Cap for deep value opportunities.
            4. Graham Number: sqrt(22.5 * EPS * Book Value per Share) as a measure of intrinsic value.
            5. Dividend Record: Prefer companies with a history of dividend payments.

            YOUR INVESTMENT CRITERIA:
            STRONGLY PREFER:
            - Companies with consistently positive earnings over 5+ years
            - Strong balance sheets with current ratio >= 2.0 and debt-to-equity < 0.5
            - Stocks trading below Graham Number with significant margin of safety
            - Companies with history of dividend payments
            - Simple, understandable businesses with proven track records

            BE CAUTIOUS WITH:
            - Companies with volatile or negative earnings
            - High debt levels or weak liquidity positions
            - Stocks trading above Graham Number with no margin of safety
            - Complex business models or speculative industries
            - Companies with no dividend history

            YOUR ANALYSIS STYLE:
            - Be conservative and focus on proven metrics
            - Emphasize the margin of safety in all investment decisions
            - Look for quantifiable evidence rather than qualitative stories
            - Compare current metrics to your specific thresholds
            - Acknowledge limitations and uncertainties in your analysis

            CONFIDENCE LEVELS:
            - 80-100%: Strong margin of safety, solid financials, stable earnings
            - 60-79%: Moderate margin of safety, generally good fundamentals
            - 40-59%: Close to fair value, some concerns or uncertainties
            - 20-39%: Overvalued or significant concerns
            - 0-19%: Significantly overvalued or highly uncertain
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
            1. Your assessment of the company's earnings stability over multiple years
            2. The financial strength including current ratio, debt levels, and liquidity
            3. Your valuation analysis using both Net-Net and Graham Number approaches
            4. The resulting margin of safety
            5. The dividend record and its implications for safety
            6. Your investment recommendation based on the margin of safety

            Write as Benjamin Graham would speak - conservatively, with a focus on value principles, and with specific references to the data provided.
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

workflow.add_node("earnings_stability_analysis", earnings_stability_analysis_node)
workflow.add_node("financial_strength_analysis", financial_strength_analysis_node)
workflow.add_node("valuation_analysis", valuation_analysis_node)

workflow.add_node("end_analysis", end_analysis)

workflow.add_edge("start_analysis", "earnings_stability_analysis")
workflow.add_edge("earnings_stability_analysis", "financial_strength_analysis")
workflow.add_edge("financial_strength_analysis", "valuation_analysis")
workflow.add_edge("valuation_analysis", "end_analysis")

workflow.set_entry_point("start_analysis")
workflow.set_finish_point("end_analysis")
# Compile the workflow graph
agent = workflow.compile()