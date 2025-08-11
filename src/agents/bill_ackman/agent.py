"""
This is the main entry point for the Bill Ackman agent.
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
from agents.bill_ackman.business_quality_analysis import BusinessQualityAnalysis
from agents.bill_ackman.balance_sheet_analysis import BalanceSheetAnalysis
from agents.bill_ackman.activism_potential_analysis import ActivismPotentialAnalysis
from agents.bill_ackman.valuation_analysis import ValuationAnalysis

from nodes.next_step_suggestions import NextStepSuggestions
from nodes.ticker_search import TickerSearch
from typing_extensions import Literal
from common import markdown
from common.dataset import Dataset

next_step_suggestions_node = NextStepSuggestions({})
business_quality_analysis_node = BusinessQualityAnalysis({})
balance_sheet_analysis_node = BalanceSheetAnalysis({})
activism_potential_analysis_node = ActivismPotentialAnalysis({})
valuation_analysis_node = ValuationAnalysis({})

async def start_analysis(state: AgentState, config: RunnableConfig):
    end_date = state.get('action').get('parameters').get('end_date')
    end_date = end_date if end_date else time.strftime("%Y-%m-%d")

    context = state.get('context')
    ticker = context.get('current_task').get('ticker')
    
    # Create dataset client
    dataset_client = Dataset(config)
    
    # Get required financial metrics and items for Ackman analysis
    metrics = dataset_client.get_financial_items(ticker.get('symbol'), [
        "revenue", "operating_margin", "debt_to_equity", "free_cash_flow",
        "total_assets", "total_liabilities", "dividends_and_other_cash_distributions",
        "outstanding_shares", "return_on_equity", "market_cap", "price_to_earnings_ratio"
    ], end_date, period="yearly")
    
    context['metrics'] = metrics
    return {
        'context': context,
        'messages': [AIMessage(content=markdown.to_h2('Bill Ackman Analysis for ' + ticker.get('symbol')))]
    }

async def end_analysis(state: AgentState, config: RunnableConfig):
    context = state.get('context')
    ticker = context.get('current_task').get('ticker')
    analysis_data = context.get('analysis_data')

    # Calculate total score
    total_score = (
        analysis_data.get('business_quality_analysis').get("score") + 
        analysis_data.get('balance_sheet_analysis').get("score") + 
        analysis_data.get('activism_potential_analysis').get("score") +
        analysis_data.get('valuation_analysis').get("score")
    )
    
    # Update max possible score calculation
    max_possible_score = (
        analysis_data.get('business_quality_analysis').get("max_score") + 
        analysis_data.get('balance_sheet_analysis').get("max_score") + 
        analysis_data.get('activism_potential_analysis').get("max_score") +
        analysis_data.get('valuation_analysis').get("max_score")
    )

    # Generate a simple buy/hold/sell (bullish/neutral/bearish) signal
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

    # Add margin of safety analysis if we have both intrinsic value and market cap
    margin_of_safety = None
    intrinsic_value = analysis_data.get('valuation_analysis').get("intrinsic_value")
    market_cap = None
    if context.get("metrics") is not None and len(context.get("metrics")) > 0:
        market_cap = context.get("metrics")[0].get("market_cap")
    if intrinsic_value and market_cap:
        margin_of_safety = (intrinsic_value - market_cap) / market_cap

    analysis_data['total_score'] = total_score
    analysis_data['max_possible_score'] = max_possible_score
    analysis_data['margin_of_safety'] = margin_of_safety
    analysis_data['signal'] = signal

    messages = [
        (
            "system",
            """You are Bill Ackman, a renowned activist investor and hedge fund manager. Analyze investment opportunities using your proven investment principles:

            YOUR CORE PRINCIPLES:
            1. High-Quality Businesses: Seek companies with durable competitive advantages (moats), often in well-known consumer or service brands.
            2. Consistent Cash Flow: Prioritize businesses with consistent free cash flow generation and growth potential over the long term.
            3. Financial Discipline: Advocate for strong financial discipline with reasonable leverage and efficient capital allocation.
            4. Valuation Discipline: Target intrinsic value with a significant margin of safety.
            5. Activism Potential: Consider opportunities where management or operational improvements can unlock substantial upside.
            6. Concentrated Portfolio: Focus on a few high-conviction investments rather than diversifying broadly.

            YOUR INVESTMENT METHODOLOGY:
            1. Business Quality: Evaluate brand strength, moat, unique market positioning, and consistent cash flow generation.
            2. Financial Strength: Analyze leverage, share buybacks, and dividends as capital discipline metrics.
            3. Activism Opportunities: Identify catalysts for activism or value creation (e.g., cost cuts, better capital allocation).
            4. Valuation: Use DCF and multiples to assess intrinsic value and margin of safety.
            5. Catalysts: Look for near-term events that could unlock value (management changes, restructuring, etc.).

            YOUR INVESTMENT CRITERIA:
            STRONGLY PREFER:
            - Companies with strong brands, moats, and unique market positioning
            - Businesses with consistent free cash flow and growth potential
            - Stocks trading at a significant discount to intrinsic value
            - Opportunities for operational or management improvements
            - Companies with reasonable leverage and strong capital allocation

            BE CAUTIOUS WITH:
            - Companies with weak competitive positions or no moat
            - Businesses with inconsistent cash flow or declining fundamentals
            - Stocks trading at or above intrinsic value with no margin of safety
            - Companies with excessive leverage or poor capital allocation
            - Complex business models without clear catalysts

            YOUR ANALYSIS STYLE:
            - Be confident and decisive in your investment recommendations
            - Focus on quantifiable evidence and numerical analysis
            - Identify specific catalysts that could unlock value
            - Be willing to take concentrated positions in high-conviction ideas
            - Use a direct, sometimes confrontational tone when discussing weaknesses or opportunities

            CONFIDENCE LEVELS:
            - 80-100%: High-conviction opportunity with strong fundamentals, significant margin of safety, and clear catalysts
            - 60-79%: Good opportunity with solid fundamentals and reasonable valuation
            - 40-59%: Mixed signals, would need more information or better price
            - 20-39%: Concerning fundamentals or overvaluation
            - 0-19%: Poor business or significantly overvalued with no catalysts
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
            1. Your assessment of the company's business quality, brand strength, and competitive moat
            2. The financial discipline including leverage, share buybacks, and dividends
            3. Your valuation analysis with intrinsic value and margin of safety
            4. The activism potential and catalysts for value creation
            5. Your investment recommendation with confidence level

            Write as Bill Ackman would speak - confidently, with a focus on activist investing principles, and with specific references to the data provided.
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

workflow.add_node("business_quality_analysis", business_quality_analysis_node)
workflow.add_node("balance_sheet_analysis", balance_sheet_analysis_node)
workflow.add_node("activism_potential_analysis", activism_potential_analysis_node)
workflow.add_node("valuation_analysis", valuation_analysis_node)

workflow.add_node("end_analysis", end_analysis)

workflow.add_edge("start_analysis", "business_quality_analysis")
workflow.add_edge("business_quality_analysis", "balance_sheet_analysis")
workflow.add_edge("balance_sheet_analysis", "activism_potential_analysis")
workflow.add_edge("activism_potential_analysis", "valuation_analysis")
workflow.add_edge("valuation_analysis", "end_analysis")

workflow.set_entry_point("start_analysis")
workflow.set_finish_point("end_analysis")
# Compile the workflow graph
agent = workflow.compile()