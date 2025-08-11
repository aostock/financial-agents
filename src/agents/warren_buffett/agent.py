"""
This is the main entry point for the agent.
It defines the workflow graph, state, tools, nodes and edges.
"""

from pkgutil import resolve_name
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

from agents.warren_buffett.fundamental_analysis import FundamentalAnalysis
from agents.warren_buffett.consistency_analysis import ConsistencyAnalysis
from agents.warren_buffett.pricing_power_analysis import PricingPowerAnalysis
from agents.warren_buffett.book_value_growth_analysis import BookValueGrowthAnalysis
from agents.warren_buffett.intrinsic_value_analysis import IntrinsicValueAnalysis
from agents.warren_buffett.moat_analysis import MoatAnalysis
from agents.warren_buffett.management_quality_analysis import ManagementQualityAnalysis


from nodes.next_step_suggestions import NextStepSuggestions

from nodes.ticker_search import TickerSearch
from typing_extensions import Literal
from common import markdown
from common.dataset import Dataset

next_step_suggestions_node = NextStepSuggestions({})
fundamental_analysis_node = FundamentalAnalysis({})
consistency_analysis_node = ConsistencyAnalysis({})
pricing_power_analysis_node = PricingPowerAnalysis({})
book_value_growth_analysis_node = BookValueGrowthAnalysis({})
intrinsic_value_analysis_node = IntrinsicValueAnalysis({})
moat_analysis_node = MoatAnalysis({})
management_quality_analysis_node = ManagementQualityAnalysis({})

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
        ], end_date, period="yearly")
    
    context['metrics'] = metrics
    return {
        'context': context,
        'messages':[AIMessage(content=markdown.to_h2('Analysis for '+ ticker.get('symbol')))]
    }

async def end_analysis(state: AgentState, config: RunnableConfig):
    context = state.get('context')
    ticker = context.get('current_task').get('ticker')
    analysis_data = context.get('analysis_data')

    # Calculate total score without circle of competence (LLM will handle that)
    total_score = (
        analysis_data.get('fundamental_analysis').get("score") + 
        analysis_data.get('consistency_analysis').get("score") + 
        analysis_data.get('moat_analysis').get("score") + 
        analysis_data.get('management_quality_analysis').get("score") +
        analysis_data.get('pricing_power_analysis').get("score") + 
        analysis_data.get('book_value_growth_analysis').get("score")
    )
    
    # Update max possible score calculation
    max_possible_score = (
        10 +  # fundamental_analysis (ROE, debt, margins, current ratio)
        analysis_data.get('moat_analysis').get("max_score") + 
        analysis_data.get('management_quality_analysis').get("max_score") +
        5 +   # pricing_power (0-5)
        5     # book_value_growth (0-5)
    )

    # Add margin of safety analysis if we have both intrinsic value and current price
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
                """You are Warren Buffett, the Oracle of Omaha. Analyze investment opportunities using my proven methodology developed over 60+ years of investing:

                MY CORE PRINCIPLES:
                1. Circle of Competence: "Risk comes from not knowing what you're doing." Only invest in businesses I thoroughly understand.
                2. Economic Moats: Seek companies with durable competitive advantages - pricing power, brand strength, scale advantages, switching costs.
                3. Quality Management: Look for honest, competent managers who think like owners and allocate capital wisely.
                4. Financial Fortress: Prefer companies with strong balance sheets, consistent earnings, and minimal debt.
                5. Intrinsic Value & Margin of Safety: Pay significantly less than what the business is worth - "Price is what you pay, value is what you get."
                6. Long-term Perspective: "Our favorite holding period is forever." Look for businesses that will prosper for decades.
                7. Pricing Power: The best businesses can raise prices without losing customers.

                MY CIRCLE OF COMPETENCE PREFERENCES:
                STRONGLY PREFER:
                - Consumer staples with strong brands (Coca-Cola, P&G, Walmart, Costco)
                - Commercial banking (Bank of America, Wells Fargo) - NOT investment banking
                - Insurance (GEICO, property & casualty)
                - Railways and utilities (BNSF, simple infrastructure)
                - Simple industrials with moats (UPS, FedEx, Caterpillar)
                - Energy companies with reserves and pipelines (Chevron, not exploration)

                GENERALLY AVOID:
                - Complex technology (semiconductors, software, except Apple due to consumer ecosystem)
                - Biotechnology and pharmaceuticals (too complex, regulatory risk)
                - Airlines (commodity business, poor economics)
                - Cryptocurrency and fintech speculation
                - Complex derivatives or financial instruments
                - Rapid technology change industries
                - Capital-intensive businesses without pricing power

                APPLE EXCEPTION: I own Apple not as a tech stock, but as a consumer products company with an ecosystem that creates switching costs.

                MY INVESTMENT CRITERIA HIERARCHY:
                First: Circle of Competence - If I don't understand the business model or industry dynamics, I don't invest, regardless of potential returns.
                Second: Business Quality - Does it have a moat? Will it still be thriving in 20 years?
                Third: Management - Do they act in shareholders' interests? Smart capital allocation?
                Fourth: Financial Strength - Consistent earnings, low debt, strong returns on capital?
                Fifth: Valuation - Am I paying a reasonable price for this wonderful business?

                MY LANGUAGE & STYLE:
                - Use folksy wisdom and simple analogies ("It's like...")
                - Reference specific past investments when relevant (Coca-Cola, Apple, GEICO, See's Candies, etc.)
                - Quote my own sayings when appropriate
                - Be candid about what I don't understand
                - Show patience - most opportunities don't meet my criteria
                - Express genuine enthusiasm for truly exceptional businesses
                - Be skeptical of complexity and Wall Street jargon

                CONFIDENCE LEVELS:
                - 90-100%: Exceptional business within my circle, trading at attractive price
                - 70-89%: Good business with decent moat, fair valuation
                - 50-69%: Mixed signals, would need more information or better price
                - 30-49%: Outside my expertise or concerning fundamentals
                - 10-29%: Poor business or significantly overvalued

                Remember: I'd rather own a wonderful business at a fair price than a fair business at a wonderful price. And when in doubt, the answer is usually "no" - there's no penalty for missed opportunities, only for permanent capital loss.
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
                1. Whether this falls within your circle of competence and why (CRITICAL FIRST STEP)
                2. Your assessment of the business's competitive moat
                3. Management quality and capital allocation
                4. Financial health and consistency
                5. Valuation relative to intrinsic value
                6. Long-term prospects and any red flags
                7. How this compares to opportunities in your portfolio

                Write as Warren Buffett would speak - plainly, with conviction, and with specific references to the data provided.
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
workflow.add_node("moat_analysis", moat_analysis_node)
workflow.add_node("pricing_power_analysis", pricing_power_analysis_node)
workflow.add_node("book_value_growth_analysis", book_value_growth_analysis_node)
workflow.add_node("management_quality_analysis", management_quality_analysis_node)
workflow.add_node("intrinsic_value_analysis", intrinsic_value_analysis_node)

workflow.add_node("end_analysis", end_analysis)

workflow.add_edge("start_analysis", "fundamental_analysis")
workflow.add_edge("fundamental_analysis", "consistency_analysis")
workflow.add_edge("consistency_analysis", "moat_analysis")
workflow.add_edge("moat_analysis", "pricing_power_analysis")
workflow.add_edge("pricing_power_analysis", "book_value_growth_analysis")
workflow.add_edge("book_value_growth_analysis", "management_quality_analysis")
workflow.add_edge("management_quality_analysis", "intrinsic_value_analysis")
workflow.add_edge("intrinsic_value_analysis", "end_analysis")

workflow.set_entry_point("start_analysis")
workflow.set_finish_point("end_analysis")
# Compile the workflow graph
agent = workflow.compile()
