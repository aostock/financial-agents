"""
This is the main entry point for the fundamentals agent.
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

from agents.fundamentals.fundamental_analysis import FundamentalAnalysis
from agents.fundamentals.consistency_analysis import ConsistencyAnalysis
from agents.fundamentals.valuation_analysis import ValuationAnalysis
from agents.fundamentals.quality_analysis import QualityAnalysis
from agents.fundamentals.growth_analysis import GrowthAnalysis

from nodes.next_step_suggestions import NextStepSuggestions

from nodes.ticker_search import TickerSearch
from typing_extensions import Literal
from common import markdown
from common.dataset import Dataset

next_step_suggestions_node = NextStepSuggestions({})
fundamental_analysis_node = FundamentalAnalysis({})
consistency_analysis_node = ConsistencyAnalysis({})
valuation_analysis_node = ValuationAnalysis({})
quality_analysis_node = QualityAnalysis({})
growth_analysis_node = GrowthAnalysis({})

async def start_analysis(state: AgentState, config: RunnableConfig):
    
    end_date = state.get('action').get('parameters').get('end_date')
    end_date = end_date if end_date else time.strftime("%Y-%m-%d")

    context = state.get('context')
    
    ticker = context.get('current_task').get('ticker')
    
    dataset_client = Dataset(config)
    metrics = dataset_client.get_financial_items(ticker.get('symbol'), [
        "return_on_equity","debt_to_equity","operating_margin","current_ratio","return_on_invested_capital","asset_turnover","market_cap",
            "capital_expenditure",
            "depreciation_and_amortization",
            "net_income",
            "ordinary_shares_number",
            "total_assets",
            "total_liabilities",
            "stockholders_equity",
            "dividends_and_other_cash_distributions",
            "issuance_or_purchase_of_equity_shares",
            "gross_profit",
            "revenue",
            "free_cash_flow",
            "gross_margin",
            "ebit",
            "interest_expense",
            "price_to_earnings_ratio",
            "price_to_book_ratio",
            "enterprise_value",
            "beta"
        ], end_date, period="yearly")
    
    context['metrics'] = metrics
    return {
        'context': context,
        'messages':[AIMessage(content=markdown.to_h2('Fundamental Analysis for '+ ticker.get('symbol')))]
    }

async def end_analysis(state: AgentState, config: RunnableConfig):
    context = state.get('context')
    ticker = context.get('current_task').get('ticker')
    analysis_data = context.get('analysis_data')

    # Calculate total score
    total_score = (
        analysis_data.get('fundamental_analysis').get("score") + 
        analysis_data.get('consistency_analysis').get("score") + 
        analysis_data.get('quality_analysis').get("score") + 
        analysis_data.get('growth_analysis').get("score") +
        analysis_data.get('valuation_analysis').get("score")
    )
    
    # Update max possible score calculation
    max_possible_score = (
        analysis_data.get('fundamental_analysis').get("max_score") + 
        analysis_data.get('consistency_analysis').get("max_score") + 
        analysis_data.get('quality_analysis').get("max_score") + 
        analysis_data.get('growth_analysis').get("max_score") +
        analysis_data.get('valuation_analysis').get("max_score")
    )

    # Add margin of safety analysis if we have both intrinsic value and current price
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

    messages = [
            (
                "system",
                """You are a fundamentals-focused investment analyst. Analyze investment opportunities using a comprehensive fundamental analysis approach:

                CORE ANALYSIS PRINCIPLES:
                1. Business Fundamentals: Strong returns on capital, conservative debt levels, healthy margins, and solid liquidity
                2. Financial Consistency: Consistent earnings, stable cash flows, and reliable financial performance over time
                3. Business Quality: High returns on invested capital, efficient asset utilization, and sustainable competitive advantages
                4. Growth Potential: Sustainable revenue growth, earnings growth, and book value growth
                5. Valuation Discipline: Paying reasonable prices relative to intrinsic value and market multiples

                ANALYSIS FRAMEWORK:
                1. Fundamental Strength: Evaluate ROE, debt levels, operating margins, and liquidity
                2. Consistency Metrics: Analyze earnings stability, cash flow reliability, and financial track record
                3. Quality Indicators: Assess ROIC, asset turnover, and overall business efficiency
                4. Growth Assessment: Review revenue growth, earnings growth, and expansion potential
                5. Valuation Analysis: Compare intrinsic value to market price and assess margin of safety

                INVESTMENT CRITERIA:
                PREFER:
                - Companies with consistent high returns on capital (ROE > 15%, ROIC > 12%)
                - Businesses with conservative debt levels (debt/equity < 0.5)
                - Organizations with strong operating margins and gross margins
                - Companies with stable, predictable earnings and cash flows
                - Businesses trading at reasonable valuations relative to intrinsic value

                BE CAUTIOUS WITH:
                - Companies with volatile earnings or unpredictable cash flows
                - Businesses with excessive leverage or deteriorating financial metrics
                - Organizations with declining returns on capital over time
                - Companies trading at significant premiums to intrinsic value
                - Businesses with poor capital allocation track records

                ANALYSIS STYLE:
                - Be data-driven and quantitative, focusing on measurable financial metrics
                - Acknowledge the limitations and uncertainties in the analysis
                - Show your work - assumptions should be clearly stated and justifiable
                - Compare valuations to market prices to determine margin of safety
                - Be willing to say "I don't know" when the data is insufficient or unclear

                CONFIDENCE LEVELS:
                - 80-100%: Strong fundamentals, consistent performance, attractive valuation
                - 60-79%: Generally good fundamentals with some concerns or fair valuation
                - 40-59%: Mixed signals, would need more information or better price
                - 20-39%: Weak fundamentals or significantly overvalued
                - 0-19%: Poor business quality or highly uncertain prospects

                Remember: "In investing, what is comfortable is rarely profitable." Focus on the numbers and fundamentals rather than stories or market sentiment.
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
                1. Your assessment of the company's fundamental strength (ROE, debt, margins, liquidity)
                2. Financial consistency and stability over time
                3. Business quality and efficiency (ROIC, asset turnover)
                4. Growth prospects and sustainability
                5. Valuation relative to intrinsic value and market multiples
                6. The resulting margin of safety
                7. Key risks and concerns
                8. Overall investment recommendation

                Write as a fundamentals-focused analyst would speak - analytically, with a focus on financial metrics, and with specific references to the data provided.
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

workflow.add_node("fundamental_analysis", fundamental_analysis_node)
workflow.add_node("consistency_analysis", consistency_analysis_node)
workflow.add_node("quality_analysis", quality_analysis_node)
workflow.add_node("growth_analysis", growth_analysis_node)
workflow.add_node("valuation_analysis", valuation_analysis_node)

workflow.add_node("end_analysis", end_analysis)

workflow.add_edge("start_analysis", "fundamental_analysis")
workflow.add_edge("fundamental_analysis", "consistency_analysis")
workflow.add_edge("consistency_analysis", "quality_analysis")
workflow.add_edge("quality_analysis", "growth_analysis")
workflow.add_edge("growth_analysis", "valuation_analysis")
workflow.add_edge("valuation_analysis", "end_analysis")

workflow.set_entry_point("start_analysis")
workflow.set_finish_point("end_analysis")
# Compile the workflow graph
agent = workflow.compile()