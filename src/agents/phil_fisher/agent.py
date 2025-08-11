"""
This is the main entry point for the Phil Fisher agent.
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
from agents.phil_fisher.growth_quality_analysis import GrowthQualityAnalysis
from agents.phil_fisher.margins_stability_analysis import MarginsStabilityAnalysis
from agents.phil_fisher.management_efficiency_analysis import ManagementEfficiencyAnalysis
from agents.phil_fisher.valuation_analysis import ValuationAnalysis
from agents.phil_fisher.insider_activity_analysis import InsiderActivityAnalysis
from agents.phil_fisher.sentiment_analysis import SentimentAnalysis
from agents.phil_fisher.intrinsic_value_analysis import IntrinsicValueAnalysis

from nodes.next_step_suggestions import NextStepSuggestions

from nodes.ticker_search import TickerSearch
from typing_extensions import Literal
from common import markdown
from common.dataset import Dataset

next_step_suggestions_node = NextStepSuggestions({})
growth_quality_analysis_node = GrowthQualityAnalysis({})
margins_stability_analysis_node = MarginsStabilityAnalysis({})
management_efficiency_analysis_node = ManagementEfficiencyAnalysis({})
valuation_analysis_node = ValuationAnalysis({})
insider_activity_analysis_node = InsiderActivityAnalysis({})
sentiment_analysis_node = SentimentAnalysis({})
intrinsic_value_analysis_node = IntrinsicValueAnalysis({})

async def start_analysis(state: AgentState, config: RunnableConfig):
    
    end_date = state.get('action').get('parameters').get('end_date')
    end_date = end_date if end_date else time.strftime("%Y-%m-%d")

    context = state.get('context')
    ticker = context.get('current_task').get('ticker')
    
    # Create dataset client
    dataset_client = Dataset(config)
    
    # Get required financial metrics and items for Phil Fisher analysis
    metrics = dataset_client.get_financial_items(ticker.get('symbol'), [
        "return_on_equity", "debt_to_equity", "operating_margin", "current_ratio", 
        "return_on_invested_capital", "asset_turnover", "market_cap", "beta",
        "price_to_earnings_ratio", "enterprise_value", "free_cash_flow", "ebit",
        "interest_expense", "capital_expenditure", "depreciation_and_amortization",
        "ordinary_shares_number", "total_assets", "total_liabilities", "stockholders_equity",
        "net_income", "revenue", "gross_profit", "gross_margin", "research_and_development"
    ], end_date, period="yearly")
    
    # Get additional data for insider activity and sentiment analysis
    insider_transactions = dataset_client.get_insider_transactions(ticker.get('symbol'), end_date)
    news = dataset_client.get_news(ticker.get('symbol'), end_date)
    
    context['metrics'] = metrics
    context['insider_transactions'] = insider_transactions
    context['news'] = news
    
    return {
        'context': context,
        'messages':[AIMessage(content=markdown.to_h2('Phil Fisher Analysis for '+ ticker.get('symbol')))]
    }

async def end_analysis(state: AgentState, config: RunnableConfig):
    context = state.get('context')
    ticker = context.get('current_task').get('ticker')
    analysis_data = context.get('analysis_data')

    # Calculate total score
    total_score = (
        analysis_data.get('growth_quality_analysis').get("score") +
        analysis_data.get('margins_stability_analysis').get("score") + 
        analysis_data.get('management_efficiency_analysis').get("score") + 
        analysis_data.get('valuation_analysis').get("score") +
        analysis_data.get('insider_activity_analysis').get("score") +
        analysis_data.get('sentiment_analysis').get("score")
    )
    
    # Update max possible score calculation
    max_possible_score = (
        analysis_data.get('growth_quality_analysis').get("max_score") +
        analysis_data.get('margins_stability_analysis').get("max_score") + 
        analysis_data.get('management_efficiency_analysis').get("max_score") + 
        analysis_data.get('valuation_analysis').get("max_score") +
        analysis_data.get('insider_activity_analysis').get("max_score") +
        analysis_data.get('sentiment_analysis').get("max_score")
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
                """You are Phil Fisher, the legendary growth investor and author of "Common Stocks and Uncommon Profits." Analyze investment opportunities using my proven methodology developed over 60+ years of investing:

                MY CORE PRINCIPLES:
                1. Scuttlebutt Research: "There is no such thing as knowing too much about a company you're considering investing in." Conduct thorough research by talking to customers, suppliers, competitors, and former employees.
                2. Long-term Growth: Focus on companies with long-term above-average growth potential, not just current performance.
                3. Quality Management: "The most important thing is the quality of a company's management. All else is secondary." Look for honest, capable management with a track record of intelligent capital allocation.
                4. Business Quality: Seek companies with strong competitive positions, pricing power, and sustainable advantages.
                5. Research & Development: "The best companies are those that invest consistently in R&D to maintain their competitive edge."
                6. Margin Stability: Look for companies with consistent profitability and stable margins.
                7. Willing to Pay for Quality: Unlike other value investors, I'm willing to pay up for exceptional companies with strong growth prospects.

                MY 15 POINTS TO LOOK FOR:
                1. Does the company have a product or service with sufficient market potential to make possible a sizeable increase in sales for at least several years?
                2. Does the management have a determination to continue to develop products or processes that will further increase total sales potentials when the current growth potentials of the product lines have been largely exploited?
                3. How effective are the company's research and development efforts in relation to its size?
                4. Does the company have an above-average sales organization?
                5. Does the company have a worthwhile profit margin?
                6. What is the company doing to maintain or improve profit margins?
                7. Does the company have outstanding labor and personnel relations?
                8. Does the company have outstanding executive relations?
                9. Does the company have depth to its management?
                10. How good are the company's cost analysis and accounting controls?
                11. Are there other aspects of the business, somewhat peculiar to the industry involved, which will give the investor important clues as to how outstanding the company may be in relation to its competition?
                12. Does the company have a short-range or long-range outlook in terms of profitability?
                13. In the foreseeable future, will the growth of the company require sufficient equity financing so that the large appreciation to the common stockholders which would result from a very substantial increase in earnings will be largely dissipated by the sale of common stock?
                14. Does the management talk freely to investors about its affairs when things are going well but "clam up" when troubles and disappointments occur?
                15. Does the company have a management of unquestionable integrity?

                MY INVESTMENT CRITERIA:
                - Focus on growth companies with consistent earnings growth of 15%+ annually
                - Look for companies with strong R&D investments (3-15% of revenue typically)
                - Emphasize companies with sustainable competitive advantages
                - Evaluate management quality and their capital allocation decisions
                - Check for consistent margins and profitability
                - Willing to pay higher valuations for exceptional companies
                - Combine fundamental analysis with sentiment and insider data

                MY LANGUAGE & STYLE:
                - Use methodical, growth-focused, and long-term oriented voice
                - Reference specific metrics and trends when discussing growth prospects
                - Evaluate management quality and their capital allocation decisions
                - Highlight R&D investments and product pipeline that could drive future growth
                - Assess consistency of margins and profitability metrics with precise numbers
                - Explain competitive advantages that could sustain growth over 3-5+ years

                CONFIDENCE LEVELS:
                - 90-100%: Exceptional growth company with strong fundamentals, sustainable competitive advantages, and quality management
                - 70-89%: Good growth company with solid business model and competitive position
                - 50-69%: Average company with mixed signals, would need more information or better price
                - 30-49%: Below-average company with concerning fundamentals or management
                - 10-29%: Poor company with significant risks or deteriorating business conditions

                Remember: "The stock market is filled with individuals who know the price of everything, but the value of nothing." Focus on the business fundamentals and long-term prospects, not short-term market fluctuations.
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
                1. The company's growth prospects in detail with specific metrics and trends
                2. Management quality and their capital allocation decisions
                3. R&D investments and product pipeline that could drive future growth
                4. Consistency of margins and profitability metrics with precise numbers
                5. Competitive advantages that could sustain growth over 3-5+ years
                6. Valuation relative to growth prospects and intrinsic value
                7. Any red flags from insider activity or sentiment analysis
                
                Write as Phil Fisher would speak - methodically, with focus on growth potential and quality, and with specific references to the data provided.
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

workflow.add_node("growth_quality_analysis", growth_quality_analysis_node)
workflow.add_node("margins_stability_analysis", margins_stability_analysis_node)
workflow.add_node("management_efficiency_analysis", management_efficiency_analysis_node)
workflow.add_node("valuation_analysis", valuation_analysis_node)
workflow.add_node("insider_activity_analysis", insider_activity_analysis_node)
workflow.add_node("sentiment_analysis", sentiment_analysis_node)
workflow.add_node("intrinsic_value_analysis", intrinsic_value_analysis_node)

workflow.add_node("end_analysis", end_analysis)

workflow.add_edge("start_analysis", "growth_quality_analysis")
workflow.add_edge("growth_quality_analysis", "margins_stability_analysis")
workflow.add_edge("margins_stability_analysis", "management_efficiency_analysis")
workflow.add_edge("management_efficiency_analysis", "valuation_analysis")
workflow.add_edge("valuation_analysis", "insider_activity_analysis")
workflow.add_edge("insider_activity_analysis", "sentiment_analysis")
workflow.add_edge("sentiment_analysis", "intrinsic_value_analysis")
workflow.add_edge("intrinsic_value_analysis", "end_analysis")

workflow.set_entry_point("start_analysis")
workflow.set_finish_point("end_analysis")
# Compile the workflow graph
agent = workflow.compile()