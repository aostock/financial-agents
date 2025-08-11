"""
This is the main entry point for the Aswath Damodaran agent.
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
from agents.aswath_damodaran.growth_analysis import GrowthAnalysis
from agents.aswath_damodaran.risk_analysis import RiskAnalysis
from agents.aswath_damodaran.intrinsic_value_analysis import IntrinsicValueAnalysis
from agents.aswath_damodaran.relative_valuation_analysis import RelativeValuationAnalysis
from agents.aswath_damodaran.story_narrative_analysis import StoryNarrativeAnalysis

from nodes.next_step_suggestions import NextStepSuggestions

from nodes.ticker_search import TickerSearch
from typing_extensions import Literal
from common import markdown
from common.dataset import Dataset

next_step_suggestions_node = NextStepSuggestions({})
growth_analysis_node = GrowthAnalysis({})
risk_analysis_node = RiskAnalysis({})
intrinsic_value_analysis_node = IntrinsicValueAnalysis({})
relative_valuation_analysis_node = RelativeValuationAnalysis({})
story_narrative_analysis_node = StoryNarrativeAnalysis({})

async def start_analysis(state: AgentState, config: RunnableConfig):
    
    end_date = state.get('action').get('parameters').get('end_date')
    end_date = end_date if end_date else time.strftime("%Y-%m-%d")

    context = state.get('context')
    ticker = context.get('current_task').get('ticker')
    dataset_client = Dataset(config)
    # Get required financial metrics and items for Damodaran analysis
    metrics = dataset_client.get_financial_items(ticker.get('symbol'), [
        "return_on_equity", "debt_to_equity", "operating_margin", "current_ratio", 
        "return_on_invested_capital", "asset_turnover", "market_cap", "beta",
        "price_to_earnings_ratio", "enterprise_value", "free_cash_flow", "ebit",
        "interest_expense", "capital_expenditure", "depreciation_and_amortization",
        "ordinary_shares_number", "total_assets", "total_liabilities", "stockholders_equity",
        "net_income", "revenue", "gross_profit", "gross_margin"
    ], end_date, period="yearly")
    
    context['metrics'] = metrics
    return {
        'context': context,
        'messages':[AIMessage(content=markdown.to_h2('Aswath Damodaran Analysis for '+ ticker.get('symbol')))]
    }

async def end_analysis(state: AgentState, config: RunnableConfig):
    context = state.get('context')
    ticker = context.get('current_task').get('ticker')
    analysis_data = context.get('analysis_data')

    # Calculate total score
    total_score = (
        analysis_data.get('story_narrative_analysis').get("score") +
        analysis_data.get('growth_analysis').get("score") + 
        analysis_data.get('risk_analysis').get("score") + 
        analysis_data.get('relative_valuation_analysis').get("score")
    )
    
    # Update max possible score calculation
    max_possible_score = (
        analysis_data.get('story_narrative_analysis').get("max_score") +
        analysis_data.get('growth_analysis').get("max_score") + 
        analysis_data.get('risk_analysis').get("max_score") + 
        analysis_data.get('relative_valuation_analysis').get("max_score")
    )

    # Add margin of safety analysis if we have both intrinsic value and market cap
    margin_of_safety = None
    intrinsic_value = analysis_data.get('intrinsic_value_analysis').get("intrinsic_value")
    market_cap = None
    if context.get("metrics") is not None and len(context.get("metrics")) > 0:
        market_cap = context.get("metrics")[0].get("market_cap")
    if intrinsic_value and market_cap:
        margin_of_safety = (intrinsic_value - market_cap) / market_cap

    analysis_data['total_score'] = total_score
    analysis_data['max_possible_score'] = max_possible_score
    analysis_data['margin_of_safety'] = margin_of_safety

    # Determine signal based on margin of safety (Damodaran tends to act with ~20-25% MOS)
    signal = "neutral"
    confidence = 50.0
    
    if margin_of_safety is not None:
        if margin_of_safety >= 0.25:
            signal = "bullish"
            confidence = min(90.0, 70.0 + (margin_of_safety * 100) / 3)  # Scale confidence
        elif margin_of_safety <= -0.25:
            signal = "bearish"
            confidence = max(10.0, 30.0 - (abs(margin_of_safety) * 100) / 3)  # Scale confidence

    messages = [
            (
                "system",
                """You are Aswath Damodaran, Professor of Finance at NYU Stern School of Business. Analyze investment opportunities using valuation techniques developed over your career:

                YOUR CORE VALUATION PRINCIPLES:
                1. Intrinsic Value: All assets have an intrinsic value that can be estimated based on fundamentals - expected cash flows, growth, and risk.
                2. Margin of Safety: Never pay more than the intrinsic value. Look for investments with at least a 20-25% margin of safety.
                3. Growth and Risk: Higher growth and lower risk lead to higher valuations. But growth without returns above the cost of capital destroys value.
                4. First Principles: Base your analysis on first principles rather than conventional wisdom or market prices.
                5. Story to Numbers to Value: Start with the business story, convert it to numbers (growth, margins, risk), then derive value.
                6. Embrace Uncertainty: Uncertainty is not the same as risk. Build it into your valuations through probability distributions.
                7. Cross-Check Valuations: Always cross-check with relative valuation (PE ratios, etc.) to ensure your DCF is reasonable.

                YOUR VALUATION METHODOLOGY:
                1. Business Narrative: Understand the company's story, competitive positioning, and management quality
                2. Cost of Equity: Estimated via CAPM (Risk-free rate + Beta * Equity Risk Premium)
                3. Growth Analysis: Examine historical revenue and FCFF growth, reinvestment efficiency
                4. Risk Profile: Evaluate beta, debt levels, and interest coverage
                5. DCF Valuation: Discount FCFF using cost of capital to estimate intrinsic value
                6. Relative Valuation: Compare PE ratios to historical medians for sanity checks

                YOUR INVESTMENT CRITERIA:
                STRONGLY PREFER:
                - Companies with predictable cash flows and understandable business models
                - Businesses with positive returns on capital that exceed their cost of capital
                - Management teams that are good capital allocators and are aligned with shareholders
                - Transparent financial reporting and disclosure
                - Coherent business narratives with strong competitive advantages

                BE CAUTIOUS WITH:
                - Companies with high growth but low or negative returns on capital
                - Businesses with excessive leverage or volatile earnings
                - Complex financial instruments or business models you don't understand
                - Companies where management incentives are misaligned with shareholders
                - Companies with weak or inconsistent business narratives

                YOUR ANALYSIS STYLE:
                - Be data-driven and quantitative, but don't ignore qualitative factors
                - Acknowledge the limitations and uncertainties in your analysis
                - Show your work - assumptions should be clearly stated and justifiable
                - Compare your intrinsic value to market price to determine margin of safety
                - Be willing to say "I don't know" when the data is insufficient or unclear

                CONFIDENCE LEVELS:
                - 80-100%: Strong margin of safety, solid fundamentals, low uncertainty
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
                1. Your assessment of the company's business narrative and competitive positioning
                2. Your assessment of the company's growth prospects and reinvestment efficiency
                3. The risk profile including beta, leverage, and interest coverage
                4. Your DCF-based intrinsic value calculation and key assumptions
                5. Relative valuation checks using PE ratios
                6. The resulting margin of safety
                7. Major uncertainties and how they affect your valuation
                8. Your investment recommendation based on the margin of safety

                Write as Aswath Damodaran would speak - analytically, with a focus on valuation principles, and with specific references to the data provided.
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

workflow.add_node("growth_analysis", growth_analysis_node)
workflow.add_node("risk_analysis", risk_analysis_node)
workflow.add_node("intrinsic_value_analysis", intrinsic_value_analysis_node)
workflow.add_node("relative_valuation_analysis", relative_valuation_analysis_node)
workflow.add_node("story_narrative_analysis", story_narrative_analysis_node)

workflow.add_node("end_analysis", end_analysis)

workflow.add_edge("start_analysis", "story_narrative_analysis")
workflow.add_edge("story_narrative_analysis", "growth_analysis")
workflow.add_edge("growth_analysis", "risk_analysis")
workflow.add_edge("risk_analysis", "intrinsic_value_analysis")
workflow.add_edge("intrinsic_value_analysis", "relative_valuation_analysis")
workflow.add_edge("relative_valuation_analysis", "end_analysis")

workflow.set_entry_point("start_analysis")
workflow.set_finish_point("end_analysis")
# Compile the workflow graph
agent = workflow.compile()