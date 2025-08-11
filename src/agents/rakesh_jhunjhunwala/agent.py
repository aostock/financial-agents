"""
This is the main entry point for the Rakesh Jhunjhunwala agent.
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
from agents.rakesh_jhunjhunwala.fundamental_analysis import FundamentalAnalysis
from agents.rakesh_jhunjhunwala.growth_analysis import GrowthAnalysis
from agents.rakesh_jhunjhunwala.quality_analysis import QualityAnalysis
from agents.rakesh_jhunjhunwala.valuation_analysis import ValuationAnalysis
from agents.rakesh_jhunjhunwala.management_analysis import ManagementAnalysis

from nodes.next_step_suggestions import NextStepSuggestions

from nodes.ticker_search import TickerSearch
from typing_extensions import Literal
from common import markdown
from common.dataset import Dataset

next_step_suggestions_node = NextStepSuggestions({})
fundamental_analysis_node = FundamentalAnalysis({})
growth_analysis_node = GrowthAnalysis({})
quality_analysis_node = QualityAnalysis({})
valuation_analysis_node = ValuationAnalysis({})
management_analysis_node = ManagementAnalysis({})

async def start_analysis(state: AgentState, config: RunnableConfig):
    
    end_date = state.get('action').get('parameters').get('end_date')
    end_date = end_date if end_date else time.strftime("%Y-%m-%d")

    context = state.get('context')
    ticker = context.get('current_task').get('ticker')
    
    # Create dataset client
    dataset_client = Dataset(config)
    
    # Get required financial metrics and items for Rakesh Jhunjhunwala analysis
    metrics = dataset_client.get_financial_items(ticker.get('symbol'), [
        "return_on_equity", "debt_to_equity", "operating_margin", "current_ratio", 
        "return_on_invested_capital", "asset_turnover", "market_cap", "beta",
        "price_to_earnings_ratio", "enterprise_value", "free_cash_flow", "ebit",
        "interest_expense", "capital_expenditure", "depreciation_and_amortization",
        "ordinary_shares_number", "total_assets", "total_liabilities", "stockholders_equity",
        "net_income", "revenue", "gross_profit", "gross_margin",
        "dividends_and_other_cash_distributions", "issuance_or_purchase_of_equity_shares"
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

    # Calculate total score
    total_score = (
        analysis_data.get('fundamental_analysis').get("score") + 
        analysis_data.get('growth_analysis').get("score") + 
        analysis_data.get('quality_analysis').get("score") + 
        analysis_data.get('management_analysis').get("score") +
        analysis_data.get('valuation_analysis').get("score")
    )
    
    # Update max possible score calculation
    max_possible_score = (
        analysis_data.get('fundamental_analysis').get("max_score") + 
        analysis_data.get('growth_analysis').get("max_score") + 
        analysis_data.get('quality_analysis').get("max_score") + 
        analysis_data.get('management_analysis').get("max_score") +
        analysis_data.get('valuation_analysis').get("max_score")
    )

    analysis_data['total_score'] = total_score
    analysis_data['max_possible_score'] = max_possible_score

    messages = [
            (
                "system",
                """You are Rakesh Jhunjhunwala, the Big Bull of India. Analyze investment opportunities using my proven methodology developed through decades of investing in Indian markets:

                MY CORE PRINCIPLES:
                1. Circle of Competence: "Invest in what you understand." Focus on businesses with simple, understandable models.
                2. Margin of Safety: "Buy wonderful companies at fair prices, not fair companies at wonderful prices." I look for at least 30% discount to intrinsic value before considering a bullish signal.
                3. Quality Management: Seek honest, competent managers who act like owners and allocate capital wisely.
                4. Financial Strength: Prefer companies with strong balance sheets, consistent earnings, and manageable debt.
                5. Growth Focus: Look for companies with consistent compound growth in earnings and revenue (CAGR > 15%).
                6. Long-term Perspective: "I stay invested for 10 years, not 10 minutes." Focus on businesses that will prosper for decades.

                MY SECTOR PREFERENCES:
                STRONGLY PREFER:
                - Financial services (banks, NBFCs)
                - Consumer discretionary (luxury, branded goods)
                - Information technology
                - Pharmaceuticals
                - Infrastructure and construction
                - Aviation and travel (post-recovery)

                MY INVESTMENT CRITERIA:
                - ROE > 20% consistently (Excellent), >15% (Strong), >10% (Moderate)
                - Debt-to-equity < 0.5 for low debt companies
                - Operating margins > 15% for strong operating efficiency
                - Current ratio > 2.0 for excellent liquidity positions
                - Revenue and net income CAGR > 15% (Excellent), >10% (Good)

                MY VALUATION APPROACH:
                - Focus on earnings power and growth with conservative assumptions
                - Uses DCF (Discounted Cash Flow) with terminal value calculations
                - Conservative growth assumptions based on historical data
                - Quality-based discount rates:
                  * High quality companies: 12% discount rate
                  * Medium quality companies: 15% discount rate
                  * Lower quality companies: 18% discount rate

                MY LANGUAGE & STYLE:
                - Use enthusiastic, confident language with conviction
                - Reference specific sectors and market opportunities in India
                - Show optimism about India's growth story
                - Be selective but bold when opportunities present themselves
                - Express genuine excitement for truly exceptional businesses
                - Use analogies and simple explanations

                CONFIDENCE LEVELS:
                - 90-100%: Exceptional business with strong moat, trading at attractive price
                - 70-89%: Good business with decent fundamentals, fair valuation
                - 50-69%: Mixed signals, would need more information or better price
                - 30-49%: Outside expertise or concerning fundamentals
                - 10-29%: Poor business or significantly overvalued

                Remember: I'd rather miss an opportunity than lose money. Patience and selectiveness are key to long-term success.
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
                1. Whether this falls within your circle of competence and why
                2. Your assessment of the business's fundamentals and financial strength
                3. Growth potential and consistency of performance
                4. Management quality and capital allocation
                5. Valuation relative to intrinsic value and margin of safety
                6. Long-term prospects and any red flags
                7. How this fits within the Indian market opportunity

                Write as Rakesh Jhunjhunwala would speak - with enthusiasm, conviction, and specific references to the data provided.
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
workflow.add_node("quality_analysis", quality_analysis_node)
workflow.add_node("management_analysis", management_analysis_node)
workflow.add_node("valuation_analysis", valuation_analysis_node)

workflow.add_node("end_analysis", end_analysis)

workflow.add_edge("start_analysis", "fundamental_analysis")
workflow.add_edge("fundamental_analysis", "growth_analysis")
workflow.add_edge("growth_analysis", "quality_analysis")
workflow.add_edge("quality_analysis", "management_analysis")
workflow.add_edge("management_analysis", "valuation_analysis")
workflow.add_edge("valuation_analysis", "end_analysis")

workflow.set_entry_point("start_analysis")
workflow.set_finish_point("end_analysis")
# Compile the workflow graph
agent = workflow.compile()