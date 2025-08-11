"""
This is the main entry point for the valuation agent.
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
from agents.valuation.dcf_analysis import DCFAnalysis
from agents.valuation.owner_earnings_analysis import OwnerEarningsAnalysis
from agents.valuation.ev_ebitda_analysis import EVEBITDAAnalysis
from agents.valuation.residual_income_analysis import ResidualIncomeAnalysis

from nodes.next_step_suggestions import NextStepSuggestions

from nodes.ticker_search import TickerSearch
from typing_extensions import Literal
from common import markdown
from common.dataset import Dataset

next_step_suggestions_node = NextStepSuggestions({})
dcf_analysis_node = DCFAnalysis({})
owner_earnings_analysis_node = OwnerEarningsAnalysis({})
ev_ebitda_analysis_node = EVEBITDAAnalysis({})
residual_income_analysis_node = ResidualIncomeAnalysis({})

async def start_analysis(state: AgentState, config: RunnableConfig):
    
    end_date = state.get('action').get('parameters').get('end_date')
    end_date = end_date if end_date else time.strftime("%Y-%m-%d")

    context = state.get('context')
    ticker = context.get('current_task').get('ticker')
    
    # Get financial metrics for valuation analysis
    dataset_client = Dataset(config)
    metrics = dataset_client.get_financial_items(ticker.get('symbol'), [
        "free_cash_flow", "net_income", "depreciation_and_amortization", 
        "capital_expenditure", "working_capital", "enterprise_value",
        "enterprise_value_to_ebitda_ratio", "market_cap", "book_value",
        "earnings_growth", "price_to_book_ratio", "return_on_equity"
    ], end_date, period="ttm")
    
    context['metrics'] = metrics
    
    # Get additional historical data for median calculations
    historical_metrics = dataset_client.get_financial_items(ticker.get('symbol'), [
        "enterprise_value_to_ebitda_ratio", "price_to_book_ratio", "return_on_equity"
    ], end_date, period="yearly")
    
    context['historical_metrics'] = historical_metrics
    
    return {
        'context': context,
        'messages':[AIMessage(content=markdown.to_h2('Valuation Analysis for '+ ticker.get('symbol')))]
    }

async def end_analysis(state: AgentState, config: RunnableConfig):
    context = state.get('context')
    ticker = context.get('current_task').get('ticker')
    analysis_data = context.get('analysis_data')

    # Calculate total score
    total_score = (
        analysis_data.get('dcf_analysis').get("score") + 
        analysis_data.get('owner_earnings_analysis').get("score") + 
        analysis_data.get('ev_ebitda_analysis').get("score") + 
        analysis_data.get('residual_income_analysis').get("score")
    )
    
    # Update max possible score calculation
    max_possible_score = (
        analysis_data.get('dcf_analysis').get("max_score") + 
        analysis_data.get('owner_earnings_analysis').get("max_score") + 
        analysis_data.get('ev_ebitda_analysis').get("max_score") + 
        analysis_data.get('residual_income_analysis').get("max_score")
    )

    analysis_data['total_score'] = total_score
    analysis_data['max_possible_score'] = max_possible_score

    messages = [
            (
                "system",
                """You are a world-class valuation analyst. Analyze investment opportunities using proven valuation methodologies:

                VALUATION METHODOLOGIES:
                1. Discounted Cash Flow (DCF): Intrinsic value based on projected free cash flows
                2. Owner Earnings: Buffett's approach focusing on cash generation after capital expenditures
                3. EV/EBITDA: Relative valuation using enterprise value multiples
                4. Residual Income: Value based on excess returns above cost of equity

                YOUR ANALYTICAL APPROACH:
                - Use multiple valuation methods for cross-validation
                - Apply appropriate margins of safety
                - Consider growth sustainability and business quality
                - Factor in market conditions and sector dynamics
                - Weight different methods based on their reliability for the specific business

                YOUR LANGUAGE & STYLE:
                - Use precise financial language with specific numbers
                - Reference concrete valuation metrics and their implications
                - Show understanding of business economics and competitive positioning
                - Be decisive but acknowledge uncertainty and limitations
                - Express conviction when appropriate but remain flexible
                - Use specific examples of valuation ranges and key drivers

                CONFIDENCE LEVELS:
                - 90-100%: Strong convergence across multiple methods with significant margin of safety
                - 70-89%: Good alignment with reasonable margin of safety
                - 50-69%: Mixed signals or moderate margin of safety
                - 30-49%: Weak signals or limited data reliability
                - 10-29%: Poor valuation setup or significant overvaluation

                Remember: Price is what you pay, value is what you get. Always consider the margin of safety and the quality of the underlying business.
                """,
            ),
            (
                "human",
                f"""Analyze this valuation opportunity for {ticker.get('symbol')} ({ticker.get('short_name')}):

                COMPREHENSIVE VALUATION ANALYSIS DATA:
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
                1. DCF valuation and key assumptions
                2. Owner earnings analysis and cash generation quality
                3. Relative valuation using EV/EBITDA multiples
                4. Residual income model and return on equity analysis
                5. Convergence or divergence across methods
                6. Margin of safety assessment
                7. Key risks and limitations in the analysis

                Write as a professional valuation analyst would speak - with precision, financial knowledge, and specific references to the data provided.
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

workflow.add_node("dcf_analysis", dcf_analysis_node)
workflow.add_node("owner_earnings_analysis", owner_earnings_analysis_node)
workflow.add_node("ev_ebitda_analysis", ev_ebitda_analysis_node)
workflow.add_node("residual_income_analysis", residual_income_analysis_node)

workflow.add_node("end_analysis", end_analysis)

workflow.add_edge("start_analysis", "dcf_analysis")
workflow.add_edge("dcf_analysis", "owner_earnings_analysis")
workflow.add_edge("owner_earnings_analysis", "ev_ebitda_analysis")
workflow.add_edge("ev_ebitda_analysis", "residual_income_analysis")
workflow.add_edge("residual_income_analysis", "end_analysis")

workflow.set_entry_point("start_analysis")
workflow.set_finish_point("end_analysis")
# Compile the workflow graph
agent = workflow.compile()