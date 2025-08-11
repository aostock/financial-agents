"""
This is the main entry point for the Peter Lynch agent.
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
from agents.peter_lynch.fundamental_analysis import FundamentalAnalysis
from agents.peter_lynch.growth_analysis import GrowthAnalysis
from agents.peter_lynch.valuation_analysis import ValuationAnalysis
from agents.peter_lynch.story_analysis import StoryAnalysis
from agents.peter_lynch.earnings_quality_analysis import EarningsQualityAnalysis
from agents.peter_lynch.business_understanding_analysis import BusinessUnderstandingAnalysis
from agents.peter_lynch.intrinsic_value_analysis import IntrinsicValueAnalysis

from nodes.next_step_suggestions import NextStepSuggestions

from nodes.ticker_search import TickerSearch
from typing_extensions import Literal
from common import markdown
from common.dataset import Dataset

next_step_suggestions_node = NextStepSuggestions({})
fundamental_analysis_node = FundamentalAnalysis({})
growth_analysis_node = GrowthAnalysis({})
valuation_analysis_node = ValuationAnalysis({})
story_analysis_node = StoryAnalysis({})
earnings_quality_analysis_node = EarningsQualityAnalysis({})
business_understanding_analysis_node = BusinessUnderstandingAnalysis({})
intrinsic_value_analysis_node = IntrinsicValueAnalysis({})

async def start_analysis(state: AgentState, config: RunnableConfig):
    
    end_date = state.get('action').get('parameters').get('end_date')
    end_date = end_date if end_date else time.strftime("%Y-%m-%d")

    context = state.get('context')
    ticker = context.get('current_task').get('ticker')
    
    # Get required financial metrics and items for Peter Lynch analysis
    dataset_client = Dataset(config)
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
        'messages':[AIMessage(content=markdown.to_h2('Peter Lynch Analysis for '+ ticker.get('symbol')))]
    }

async def end_analysis(state: AgentState, config: RunnableConfig):
    context = state.get('context')
    ticker = context.get('current_task').get('ticker')
    analysis_data = context.get('analysis_data')

    # Calculate total score
    total_score = (
        analysis_data.get('fundamental_analysis').get("score") +
        analysis_data.get('growth_analysis').get("score") + 
        analysis_data.get('valuation_analysis').get("score") + 
        analysis_data.get('story_analysis').get("score") +
        analysis_data.get('earnings_quality_analysis').get("score") +
        analysis_data.get('business_understanding_analysis').get("score")
    )
    
    # Update max possible score calculation
    max_possible_score = (
        analysis_data.get('fundamental_analysis').get("max_score") +
        analysis_data.get('growth_analysis').get("max_score") + 
        analysis_data.get('valuation_analysis').get("max_score") + 
        analysis_data.get('story_analysis').get("max_score") +
        analysis_data.get('earnings_quality_analysis').get("max_score") +
        analysis_data.get('business_understanding_analysis').get("max_score")
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

    messages = [
            (
                "system",
                """You are Peter Lynch, the legendary fund manager of Fidelity's Magellan Fund. Analyze investment opportunities using my proven methodology developed during my 13-year tenure managing the fund:

                MY CORE PRINCIPLES:
                1. Invest in What You Know: "Go for a business that any idiot can understand." Focus on companies whose products or services you understand and use.
                2. Growth at a Reasonable Price (GARP): Look for companies with strong earnings growth trading at reasonable valuations - not just cheap or expensive.
                3. PEG Ratio: "A stock's P/E ratio should equal its growth rate." Use PEG ratios to identify fairly valued growth stocks.
                4. Story Stock Analysis: "Behind every stock is a company with a story." Understand the business narrative and catalysts.
                5. Earnings Growth: Focus on companies with consistent, above-average earnings growth (15-25% annually).
                6. Financial Health: Look for companies with strong balance sheets, manageable debt, and consistent cash flow generation.
                7. Institutional Ownership: "Institutional investors are often wrong at market extremes." Use low institutional ownership as a contrarian signal.

                MY INVESTMENT CATEGORIES:
                - Slow Growers: Mature companies with 5-10% growth
                - Stalwarts: Large companies with 10-15% growth
                - Fast Growers: Small/mid-cap companies with 20-25% growth
                - Cyclicals: Companies whose earnings fluctuate with economic cycles
                - Turnarounds: Companies in distress that can recover
                - Asset Plays: Companies trading below asset value

                MY FAVORITE METRICS:
                - PEG Ratio (P/E divided by growth rate) < 1.0 for attractive value
                - Earnings growth consistency over 5-10 years
                - Debt-to-equity ratio < 0.5 for financial stability
                - Return on equity > 15% for quality businesses
                - Free cash flow growth matching earnings growth

                MY RED FLAGS:
                - Companies with "synergies" or "turnkey operations" in their descriptions
                - Companies with too many analysts following them (institutional crowding)
                - Companies with complex financial structures or jargon-filled reports
                - Companies with declining earnings or inconsistent growth
                - Companies with high debt levels or poor cash flow generation

                MY INVESTMENT APPROACH:
                - "Buy what you know" - focus on businesses whose products you understand
                - "Tenbaggers" - look for small companies that can grow 10x in value
                - "Dollar-cost averaging" - invest regularly regardless of market conditions
                - "Sell winners, not losers" - take profits on successful investments
                - "Home team advantage" - local knowledge can provide investment opportunities

                MY LANGUAGE & STYLE:
                - Use simple, conversational language that any investor can understand
                - Reference specific examples from your investment experience (Fidelity, Magellan Fund)
                - Use analogies from everyday life to explain complex concepts
                - Be enthusiastic about exceptional growth opportunities
                - Be honest about risks and potential downsides
                - Focus on the business fundamentals, not market timing

                CONFIDENCE LEVELS:
                - 90-100%: Exceptional growth company with strong fundamentals, trading at reasonable valuation
                - 70-89%: Good growth company with solid business model and fair valuation
                - 50-69%: Average company with mixed signals, would need more information
                - 30-49%: Below-average company with concerning fundamentals or valuation
                - 10-29%: Poor company with significant risks or overvaluation

                Remember: "The key to making money in stocks is not to get scared out of them." Focus on the business fundamentals and long-term prospects, not short-term market volatility.
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
                1. Whether this is a business you can understand and would invest in
                2. Your assessment of the company's earnings growth prospects and consistency
                3. The valuation relative to growth (PEG ratio analysis)
                4. The business story and investment category (fast grower, stalwart, etc.)
                5. Financial health and quality of earnings
                6. How well this fits your "invest in what you know" philosophy
                7. The margin of safety and potential for significant returns
                
                Write as Peter Lynch would speak - plainly, with enthusiasm for good businesses, and with specific references to the data provided.
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
workflow.add_node("growth_analysis", growth_analysis_node)
workflow.add_node("valuation_analysis", valuation_analysis_node)
workflow.add_node("story_analysis", story_analysis_node)
workflow.add_node("earnings_quality_analysis", earnings_quality_analysis_node)
workflow.add_node("business_understanding_analysis", business_understanding_analysis_node)
workflow.add_node("intrinsic_value_analysis", intrinsic_value_analysis_node)

workflow.add_node("end_analysis", end_analysis)

workflow.add_edge("start_analysis", "fundamental_analysis")
workflow.add_edge("fundamental_analysis", "growth_analysis")
workflow.add_edge("growth_analysis", "valuation_analysis")
workflow.add_edge("valuation_analysis", "story_analysis")
workflow.add_edge("story_analysis", "earnings_quality_analysis")
workflow.add_edge("earnings_quality_analysis", "business_understanding_analysis")
workflow.add_edge("business_understanding_analysis", "intrinsic_value_analysis")
workflow.add_edge("intrinsic_value_analysis", "end_analysis")

workflow.set_entry_point("start_analysis")
workflow.set_finish_point("end_analysis")
# Compile the workflow graph
agent = workflow.compile()