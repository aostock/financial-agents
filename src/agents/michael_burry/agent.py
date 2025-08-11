"""
This is the main entry point for the Michael Burry agent.
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

from agents.michael_burry.financial_statement_analysis import FinancialStatementAnalysis
from agents.michael_burry.market_inefficiency_analysis import MarketInefficiencyAnalysis
from agents.michael_burry.deep_value_analysis import DeepValueAnalysis
from agents.michael_burry.risk_assessment import RiskAssessment
from agents.michael_burry.contrarian_analysis import ContrarianAnalysis

from nodes.next_step_suggestions import NextStepSuggestions

from nodes.ticker_search import TickerSearch
from typing_extensions import Literal
from common import markdown
from common.dataset import Dataset

next_step_suggestions_node = NextStepSuggestions({})
financial_statement_analysis_node = FinancialStatementAnalysis({})
market_inefficiency_analysis_node = MarketInefficiencyAnalysis({})
deep_value_analysis_node = DeepValueAnalysis({})
risk_assessment_node = RiskAssessment({})
contrarian_analysis_node = ContrarianAnalysis({})

async def start_analysis(state: AgentState, config: RunnableConfig):
    
    end_date = state.get('action').get('parameters').get('end_date')
    end_date = end_date if end_date else time.strftime("%Y-%m-%d")

    context = state.get('context')
    
    ticker = context.get('current_task').get('ticker')
    
    # Create dataset client
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
            "beta",
            "cash_and_equivalents",
            "inventory",
            "accounts_receivable",
            "accounts_payable",
            "short_term_debt",
            "long_term_debt",
            "operating_income"
        ], end_date, period="yearly")
    
    context['metrics'] = metrics
    return {
        'context': context,
        'messages':[AIMessage(content=markdown.to_h2('Michael Burry Analysis for '+ ticker.get('symbol')))]
    }

async def end_analysis(state: AgentState, config: RunnableConfig):
    context = state.get('context')
    ticker = context.get('current_task').get('ticker')
    analysis_data = context.get('analysis_data')

    # Calculate total score
    total_score = (
        analysis_data.get('financial_statement_analysis').get("score") + 
        analysis_data.get('market_inefficiency_analysis').get("score") + 
        analysis_data.get('deep_value_analysis').get("score") + 
        analysis_data.get('risk_assessment').get("score") +
        analysis_data.get('contrarian_analysis').get("score")
    )
    
    # Update max possible score calculation
    max_possible_score = (
        analysis_data.get('financial_statement_analysis').get("max_score") + 
        analysis_data.get('market_inefficiency_analysis').get("max_score") + 
        analysis_data.get('deep_value_analysis').get("max_score") + 
        analysis_data.get('risk_assessment').get("max_score") +
        analysis_data.get('contrarian_analysis').get("max_score")
    )

    # Add margin of safety analysis if we have both intrinsic value and current price
    margin_of_safety = None
    intrinsic_value = analysis_data.get('deep_value_analysis').get("intrinsic_value")
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
                """You are Michael Burry, the investor famous for predicting the 2008 housing market crash as depicted in "The Big Short". Analyze investment opportunities using your proven methodology:

                YOUR CORE PRINCIPLES:
                1. Deep Financial Statement Analysis: Scrutinize every line item in financial statements for anomalies, inconsistencies, and hidden risks.
                2. Market Inefficiencies: Identify mispriced securities where the market has misunderstood or ignored key risks.
                3. Contrarian Thinking: Go against popular opinion when data supports a different conclusion.
                4. Risk Assessment: Focus on asymmetric risk-reward profiles with limited downside and significant upside.
                5. Deep Value Investing: Look for securities trading at significant discounts to intrinsic value with strong catalysts.

                YOUR INVESTMENT APPROACH:
                1. Financial Forensics: Examine footnotes, accounting practices, and off-balance sheet items for red flags.
                2. Market Mispricing Identification: Look for securities where market pricing doesn't reflect underlying fundamentals.
                3. Contrarian Opportunities: Identify situations where crowd psychology has created mispricings.
                4. Risk/Reward Analysis: Evaluate potential losses vs. potential gains with a focus on asymmetric payoffs.
                5. Deep Value Opportunities: Find securities with strong fundamentals trading at significant discounts.

                YOUR INVESTMENT CRITERIA:
                STRONGLY PREFER:
                - Companies with clean, transparent financial statements and conservative accounting
                - Securities where market sentiment doesn't reflect underlying fundamentals
                - Situations with clear catalysts for value realization
                - Investments with asymmetric risk-reward profiles (limited downside, significant upside)
                - Businesses with strong balance sheets and low debt levels

                GENERALLY AVOID:
                - Companies with complex, opaque financial statements
                - Securities where market pricing seems to ignore key risks
                - Investments without clear catalysts for value realization
                - Situations with significant downside risk and limited upside
                - Businesses with excessive leverage or accounting irregularities

                YOUR ANALYSIS STYLE:
                - Be meticulous and detail-oriented, focusing on forensic accounting
                - Challenge conventional wisdom and market sentiment
                - Look for data that contradicts popular narratives
                - Focus on asymmetric risk-reward opportunities
                - Be patient but ready to act decisively when opportunities arise

                CONFIDENCE LEVELS:
                - 90-100%: Exceptional opportunity with strong fundamentals, significant undervaluation, and clear catalysts
                - 70-89%: Good opportunity with favorable risk-reward profile and reasonable valuation
                - 50-69%: Mixed signals, would need more information or better risk-reward profile
                - 30-49%: Unfavorable risk-reward profile or significant concerns
                - 10-29%: Poor opportunity with significant risks or overvaluation

                Remember: "The market can stay irrational longer than you can stay solvent." But when you find a real mispricing with asymmetric risk-reward, be decisive.
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
                1. Your assessment of the company's financial statement quality and any red flags
                2. Market inefficiencies or mispricings you've identified
                3. Deep value opportunities and intrinsic value assessment
                4. Risk assessment including downside protection and catalysts
                5. Contrarian aspects of this investment opportunity
                6. The resulting margin of safety
                7. Overall investment recommendation

                Write as Michael Burry would speak - analytically, with a focus on forensic accounting, and with specific references to the data provided.
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

workflow.add_node("financial_statement_analysis", financial_statement_analysis_node)
workflow.add_node("market_inefficiency_analysis", market_inefficiency_analysis_node)
workflow.add_node("deep_value_analysis", deep_value_analysis_node)
workflow.add_node("risk_assessment", risk_assessment_node)
workflow.add_node("contrarian_analysis", contrarian_analysis_node)

workflow.add_node("end_analysis", end_analysis)

workflow.add_edge("start_analysis", "financial_statement_analysis")
workflow.add_edge("financial_statement_analysis", "market_inefficiency_analysis")
workflow.add_edge("market_inefficiency_analysis", "deep_value_analysis")
workflow.add_edge("deep_value_analysis", "risk_assessment")
workflow.add_edge("risk_assessment", "contrarian_analysis")
workflow.add_edge("contrarian_analysis", "end_analysis")

workflow.set_entry_point("start_analysis")
workflow.set_finish_point("end_analysis")
# Compile the workflow graph
agent = workflow.compile()