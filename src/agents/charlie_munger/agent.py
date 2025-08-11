"""
This is the main entry point for the Charlie Munger agent.
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

from agents.charlie_munger.moat_strength_analysis import MoatStrengthAnalysis
from agents.charlie_munger.management_quality_analysis import ManagementQualityAnalysis
from agents.charlie_munger.predictability_analysis import PredictabilityAnalysis
from agents.charlie_munger.valuation_analysis import ValuationAnalysis


from nodes.next_step_suggestions import NextStepSuggestions

from nodes.ticker_search import TickerSearch
from typing_extensions import Literal
from common import markdown
from common.dataset import Dataset

next_step_suggestions_node = NextStepSuggestions({})
moat_strength_analysis_node = MoatStrengthAnalysis({})
management_quality_analysis_node = ManagementQualityAnalysis({})
predictability_analysis_node = PredictabilityAnalysis({})
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
        "net_income",
        "operating_income",
        "return_on_invested_capital",
        "gross_margin",
        "operating_margin",
        "free_cash_flow",
        "capital_expenditure",
        "cash_and_equivalents",
        "total_debt",
        "shareholders_equity",
        "outstanding_shares",
        "research_and_development",
        "goodwill_and_intangible_assets",
        "market_cap",
        ], end_date, period="yearly")
    
    context['metrics'] = metrics
    return {
        'context': context,
        'messages':[AIMessage(content=markdown.to_h2('Charlie Munger Analysis for '+ ticker.get('symbol')))]
    }

async def end_analysis(state: AgentState, config: RunnableConfig):
    context = state.get('context')
    ticker = context.get('current_task').get('ticker')
    analysis_data = context.get('analysis_data')

    # Calculate total score with Munger's weighting preferences
    # Munger weights quality and predictability higher than current valuation
    total_score = (
        analysis_data.get('moat_strength_analysis').get("score") * 0.35 +
        analysis_data.get('management_quality_analysis').get("score") * 0.25 +
        analysis_data.get('predictability_analysis').get("score") * 0.25 +
        analysis_data.get('valuation_analysis').get("score") * 0.15
    )
    
    # Update max possible score calculation
    max_possible_score = 10  # Scale to 0-10

    analysis_data['total_score'] = total_score
    analysis_data['max_possible_score'] = max_possible_score

    messages = [
            (
                "system",
                """You are Charlie Munger, Vice Chairman of Berkshire Hathaway. Analyze investment opportunities using my proven methodology focused on business quality and mental models:

                MY CORE PRINCIPLES:
                1. Focus on Quality: Seek businesses with strong, durable competitive advantages (moats)
                2. Mental Models: Apply principles from multiple disciplines (mathematics, physics, psychology, economics) to analyze investments
                3. Predictability: Prefer businesses with consistent, understandable operations and cash flows
                4. Management Quality: Value integrity, competence, and shareholder-friendly capital allocation
                5. Long-term Thinking: Think in decades, not quarters
                6. Margin of Safety: Never overpay, always demand a buffer against errors
                7. Invert: "All I want to know is where I'm going to die, so I'll never go there." Focus on avoiding stupidity rather than seeking brilliance.
                8. Simplicity: Avoid complexity and businesses you don't understand

                MY INVESTMENT FOCUS AREAS:
                STRONGLY PREFER:
                - Businesses with consistent high returns on invested capital (ROIC > 15%)
                - Companies with pricing power and stable gross margins
                - Operations with predictable revenue and cash flow patterns
                - Management with skin in the game and proven capital allocation skills
                - Low capital intensity businesses that don't require constant reinvestment

                GENERALLY AVOID:
                - Businesses with volatile earnings or unpredictable cash flows
                - Companies with excessive leverage or financial engineering
                - Industries with rapid technological change that I can't understand
                - Management teams with poor capital allocation or self-serving behavior
                - Complex financial instruments or business models

                MY INVESTMENT CRITERIA:
                First: Business Quality - Does it have a strong, durable competitive advantage?
                Second: Predictability - Are the operations and cash flows reasonably predictable?
                Third: Management - Do they act with integrity and allocate capital wisely?
                Fourth: Valuation - Am I paying a fair price for this quality business?

                MY LANGUAGE & STYLE:
                - Use pithy, direct wisdom and mental model references
                - Reference specific disciplines when explaining analysis (e.g., "Basic microeconomics shows...")
                - Apply the principle of inversion when discussing risks ("I avoid businesses where...")
                - Show intellectual humility - acknowledge limits of knowledge
                - Focus on avoiding mistakes rather than making brilliant moves
                - Use concrete examples and specific numbers when discussing quality metrics

                CONFIDENCE LEVELS:
                - 90-100%: Exceptional business quality with strong predictability and reasonable valuation
                - 70-89%: Good quality business with some concerns or fair valuation
                - 50-69%: Mixed signals, would need more information or better price
                - 30-49%: Poor business quality or concerning fundamentals
                - 10-29%: Significant red flags or overvalued

                Remember: "It's not supposed to be easy. No wise person ever expected it to be easy. It's not supposed to be easy to build a great company or make good investments. That's what's so great about it - the difficulty filters out the competition."
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
                1. Whether this business has a strong, durable competitive advantage (moat)
                2. Your assessment of the predictability of operations and cash flows
                3. Management quality and capital allocation track record
                4. Valuation relative to business quality and predictability
                5. Key risks and what you would "avoid" in your analysis (apply inversion)
                6. Which mental models or disciplines you applied in your analysis
                7. How this compares to opportunities in Berkshire's portfolio

                Write as Charlie Munger would speak - directly, with wisdom from multiple disciplines, and with specific references to the data provided.
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

workflow.add_node("moat_strength_analysis", moat_strength_analysis_node)
workflow.add_node("management_quality_analysis", management_quality_analysis_node)
workflow.add_node("predictability_analysis", predictability_analysis_node)
workflow.add_node("valuation_analysis", valuation_analysis_node)

workflow.add_node("end_analysis", end_analysis)

workflow.add_edge("start_analysis", "moat_strength_analysis")
workflow.add_edge("moat_strength_analysis", "management_quality_analysis")
workflow.add_edge("management_quality_analysis", "predictability_analysis")
workflow.add_edge("predictability_analysis", "valuation_analysis")
workflow.add_edge("valuation_analysis", "end_analysis")

workflow.set_entry_point("start_analysis")
workflow.set_finish_point("end_analysis")
# Compile the workflow graph
agent = workflow.compile()