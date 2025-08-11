"""
This is the main entry point for the Risk Manager agent.
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
from agents.risk_manager.risk_analysis import RiskAnalysis

from nodes.next_step_suggestions import NextStepSuggestions

from nodes.ticker_search import TickerSearch
from typing_extensions import Literal
from common import markdown
from common.dataset import Dataset

next_step_suggestions_node = NextStepSuggestions({})
risk_analysis_node = RiskAnalysis({})

async def start_analysis(state: AgentState, config: RunnableConfig):
    
    end_date = state.get('action').get('parameters').get('end_date')
    end_date = end_date if end_date else time.strftime("%Y-%m-%d")

    context = state.get('context')
    ticker = context.get('current_task').get('ticker')
    
    # Get price data for risk analysis
    dataset_client = Dataset(config)
    prices = dataset_client.get_prices(ticker.get('symbol'), end_date, end_date)
    
    # Get portfolio data from context
    portfolio = context.get('portfolio', {
        "cash": 100000.00,
        "positions": {}
    })
    
    context['prices'] = prices
    context['portfolio'] = portfolio
    
    return {
        'context': context,
        'messages':[AIMessage(content=markdown.to_h2('Risk Analysis for '+ ticker.get('symbol')))]
    }

async def end_analysis(state: AgentState, config: RunnableConfig):
    context = state.get('context')
    ticker = context.get('current_task').get('ticker')
    analysis_data = context.get('analysis_data')
    
    # Get risk analysis data
    risk_analysis = analysis_data.get('risk_analysis', {})
    position_limit = risk_analysis.get('position_limit', 0)
    remaining_position_limit = risk_analysis.get('remaining_position_limit', 0)
    
    # Calculate risk score
    total_score = risk_analysis.get("score", 0)
    max_possible_score = risk_analysis.get("max_score", 10)

    messages = [
            (
                "system",
                """You are a professional risk manager responsible for controlling position sizing and portfolio risk exposure. Your role is to ensure that investment decisions align with prudent risk management principles.

                YOUR ROLE:
                You are responsible for evaluating the risk profile of investment opportunities and determining appropriate position sizing limits. You must consider:
                1. Total portfolio value and diversification
                2. Position concentration limits (maximum 20% of portfolio per position)
                3. Available cash for new long positions
                4. Available margin for new short positions
                5. Current portfolio exposures and risk concentration

                RISK MANAGEMENT PRINCIPLES:
                - Position Sizing: No single position should exceed 20% of total portfolio value
                - Diversification: Spread risk across multiple positions and sectors
                - Cash Management: Ensure sufficient liquidity for opportunities
                - Margin Control: Monitor and limit margin usage for short positions
                - Risk Concentration: Avoid overexposure to correlated positions

                POSITION LIMITS:
                - Maximum position size: 20% of total portfolio value
                - For long positions: Limited by available cash
                - For short positions: Limited by available margin (50% margin requirement)
                - Final position limit: Minimum of position limit and capital constraints

                RISK SCORE INTERPRETATION:
                - 90-100%: Excellent risk profile with significant room for additional investment
                - 70-89%: Good risk profile with moderate room for additional investment
                - 50-69%: Acceptable risk profile with limited room for additional investment
                - 30-49%: Elevated risk requiring careful monitoring
                - 10-29%: High risk requiring position reduction

                Your analysis should provide clear guidance on:
                1. Maximum position size allowed for this ticker
                2. Current portfolio risk exposures
                3. Capital constraints affecting position sizing
                4. Recommendations for risk-adjusted position sizing
                """,
            ),
            (
                "human",
                f"""Evaluate the risk profile and position sizing for {ticker.get('symbol')} ({ticker.get('short_name')}):

                RISK ANALYSIS DATA:
                {analysis_data}

                Portfolio Data:
                - Cash: $100,000.00
                - Current Positions: {{}}
                
                Please provide your risk assessment in exactly this JSON format, notice to use 'RiskAssessment' before json:
                ```RiskAssessment
                {{
                  "max_position_size": float,
                  "risk_level": "low" | "medium" | "high",
                  "confidence": float between 0 and 100,
                  "recommendation": "string explaining the risk-based position sizing recommendation"
                }}
                ```
                then provide a detailed reasoning for your assessment.

                In your reasoning, be specific about:
                1. The maximum position size allowed based on portfolio constraints
                2. Current portfolio risk exposures and diversification
                3. Capital constraints affecting position sizing (cash, margin)
                4. Risk concentration concerns
                5. Recommendations for risk-adjusted position sizing
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
workflow.add_node("risk_analysis", risk_analysis_node)
workflow.add_node("end_analysis", end_analysis)

workflow.add_edge("start_analysis", "risk_analysis")
workflow.add_edge("risk_analysis", "end_analysis")

workflow.set_entry_point("start_analysis")
workflow.set_finish_point("end_analysis")
# Compile the workflow graph
agent = workflow.compile()