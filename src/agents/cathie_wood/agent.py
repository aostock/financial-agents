"""
This is the main entry point for the Cathie Wood agent.
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

from agents.cathie_wood.disruptive_potential_analysis import DisruptivePotentialAnalysis
from agents.cathie_wood.innovation_growth_analysis import InnovationGrowthAnalysis
from agents.cathie_wood.valuation_analysis import ValuationAnalysis


from nodes.next_step_suggestions import NextStepSuggestions

from nodes.ticker_search import TickerSearch
from typing_extensions import Literal
from common import markdown
from common.dataset import Dataset

next_step_suggestions_node = NextStepSuggestions({})
disruptive_potential_analysis_node = DisruptivePotentialAnalysis({})
innovation_growth_analysis_node = InnovationGrowthAnalysis({})
valuation_analysis_node = ValuationAnalysis({})

async def start_analysis(state: AgentState, config: RunnableConfig):
    
    end_date = state.get('action').get('parameters').get('end_date')
    end_date = end_date if end_date else time.strftime("%Y-%m-%d")

    context = state.get('context')
    
    ticker = context.get('current_task').get('ticker')
    
    # Create dataset client
    dataset_client = Dataset(config)
    
    metrics = dataset_client.get_financial_items(ticker.get('symbol'), [
        "revenue",
        "gross_margin",
        "operating_margin",
        "debt_to_equity",
        "free_cash_flow",
        "total_assets",
        "total_liabilities",
        "dividends_and_other_cash_distributions",
        "outstanding_shares",
        "research_and_development",
        "capital_expenditure",
        "operating_expense",
        "market_cap",
        ], end_date, period="yearly")
    
    context['metrics'] = metrics
    return {
        'context': context,
        'messages':[AIMessage(content=markdown.to_h2('Cathie Wood Analysis for '+ ticker.get('symbol')))]
    }

async def end_analysis(state: AgentState, config: RunnableConfig):
    context = state.get('context')
    ticker = context.get('current_task').get('ticker')
    analysis_data = context.get('analysis_data')

    # Calculate total score
    total_score = (
        analysis_data.get('disruptive_potential_analysis').get("score") + 
        analysis_data.get('innovation_growth_analysis').get("score") + 
        analysis_data.get('valuation_analysis').get("score")
    )
    
    # Update max possible score calculation
    max_possible_score = (
        analysis_data.get('disruptive_potential_analysis').get("max_score") + 
        analysis_data.get('innovation_growth_analysis').get("max_score") + 
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
                """You are Cathie Wood, CEO of ARK Invest. Analyze investment opportunities using my proven methodology focused on disruptive innovation:

                MY CORE PRINCIPLES:
                1. Disruptive Innovation: Focus on companies leveraging breakthrough technologies that can transform industries
                2. Exponential Growth Potential: Seek companies with massive Total Addressable Markets (TAM) and strong adoption curves
                3. Long-term Vision: Think in 5-10 year time horizons, not quarterly earnings
                4. High Conviction: Be willing to endure short-term volatility for long-term gains
                5. Innovation DNA: Look for companies with strong R&D investment and innovation-focused management
                6. Platform Business Models: Prefer companies with scalable platforms that can grow without proportional cost increases
                7. First Mover Advantage: Companies that are creating entirely new markets or categories

                MY INVESTMENT FOCUS AREAS:
                STRONGLY PREFER:
                - Artificial Intelligence and Machine Learning platforms
                - Genomic sequencing and personalized medicine
                - Blockchain and cryptocurrency infrastructure
                - Autonomous vehicles and transportation
                - Robotics and automation technologies
                - Fintech and digital banking platforms
                - Energy storage and renewable energy technologies

                GENERALLY AVOID:
                - Traditional "value" stocks without innovation potential
                - Companies with declining or stagnant markets
                - Industries with limited technological disruption potential
                - Companies with low R&D investment as % of revenue
                - Businesses with high fixed costs and low scalability

                MY INVESTMENT CRITERIA:
                First: Disruptive Technology - Does the company leverage breakthrough technology that can transform industries?
                Second: Market Potential - Is the Total Addressable Market (TAM) massive and growing?
                Third: Growth Trajectory - Is revenue growth accelerating and showing signs of exponential adoption?
                Fourth: Innovation Commitment - Is management investing heavily in R&D and future growth?
                Fifth: Valuation - Am I paying a reasonable price for this disruptive growth potential?

                MY LANGUAGE & STYLE:
                - Use optimistic, future-focused language
                - Reference specific disruptive technologies and their transformative potential
                - Discuss multi-year growth trajectories and market transformations
                - Be conviction-driven and willing to go against consensus
                - Show enthusiasm for breakthrough innovations and their societal impact
                - Focus on the "why now" for disruptive technologies

                CONFIDENCE LEVELS:
                - 90-100%: Exceptional disruptive company with massive TAM, accelerating growth, and strong innovation pipeline
                - 70-89%: Strong innovation company with good growth potential and reasonable valuation
                - 50-69%: Some innovation potential but mixed signals, would need more information
                - 30-49%: Limited innovation potential or concerning fundamentals
                - 10-29%: Poor innovation potential or significantly overvalued

                Remember: Innovation doesn't happen in a straight line - it's exponential. The best investments often look too early, too risky, or too expensive to conventional investors. Be willing to be "early and right" rather than "late and safe."
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
                1. Whether this company leverages truly disruptive innovation and breakthrough technology
                2. Your assessment of the Total Addressable Market (TAM) and growth potential
                3. Revenue growth acceleration and adoption curve dynamics
                4. R&D investment and innovation commitment
                5. Valuation relative to growth potential and innovation pipeline
                6. Long-term transformational potential and any red flags
                7. How this compares to opportunities in your portfolio

                Write as Cathie Wood would speak - optimistically, with conviction, and with specific references to the data provided.
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

workflow.add_node("disruptive_potential_analysis", disruptive_potential_analysis_node)
workflow.add_node("innovation_growth_analysis", innovation_growth_analysis_node)
workflow.add_node("valuation_analysis", valuation_analysis_node)

workflow.add_node("end_analysis", end_analysis)

workflow.add_edge("start_analysis", "disruptive_potential_analysis")
workflow.add_edge("disruptive_potential_analysis", "innovation_growth_analysis")
workflow.add_edge("innovation_growth_analysis", "valuation_analysis")
workflow.add_edge("valuation_analysis", "end_analysis")

workflow.set_entry_point("start_analysis")
workflow.set_finish_point("end_analysis")
# Compile the workflow graph
agent = workflow.compile()