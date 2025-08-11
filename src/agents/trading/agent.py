"""
This is the main entry point for the trading agent.
It defines the workflow graph, state, tools, nodes and edges.
"""

import time
from common.agent_state import AgentState
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig
from llm.llm_model import ainvoke
from langgraph.graph import END, StateGraph

from nodes.next_step_suggestions import NextStepSuggestions

from common import markdown
from common.dataset import Dataset

# Import technical analysis module
from agents.trading.technical_analysis import TechnicalAnalysis

next_step_suggestions_node = NextStepSuggestions({})


async def start_analysis(state: AgentState, config: RunnableConfig):
    end_date = state.get('action').get('parameters').get('end_date')
    end_date = end_date if end_date else time.strftime("%Y-%m-%d")

    context = state.get('context')
    
    ticker = context.get('current_task').get('ticker')
    
    # Create dataset client
    dataset_client = Dataset(config)
    
    # Get market data for technical analysis (last 200 days)
    start_date = time.strftime("%Y-%m-%d", time.localtime(time.time() - 200*24*60*60))
    prices = dataset_client.get_prices(ticker.get('symbol'), start_date, end_date)
    
    # Get financial metrics
    metrics = dataset_client.get_financial_items(ticker.get('symbol'), [
        "return_on_equity", "debt_to_equity", "operating_margin", "current_ratio", 
        "return_on_invested_capital", "asset_turnover", "market_cap",
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
        "gross_margin"
    ], end_date, period="yearly")
    
    # Get news data
    news = dataset_client.get_news(ticker.get('symbol'), end_date)
    
    # Get insider transactions
    insider_transactions = dataset_client.get_insider_transactions(ticker.get('symbol'), end_date)
    
    context['prices'] = prices
    context['metrics'] = metrics
    context['news'] = news
    context['insider_transactions'] = insider_transactions
    
    return {
        'context': context,
        'messages': [AIMessage(content=markdown.to_h2('Trading Analysis for '+ ticker.get('symbol')))]
    }


async def market_analysis(state: AgentState, config: RunnableConfig):
    context = state.get('context')
    ticker = context.get('current_task').get('ticker')
    prices = context.get('prices', [])
    metrics = context.get('metrics', [])
    
    # Perform technical analysis
    technical_analyzer = TechnicalAnalysis(config)
    trend_analysis = technical_analyzer.analyze_trend(prices)
    momentum_analysis = technical_analyzer.analyze_momentum(prices)
    volatility_analysis = technical_analyzer.analyze_volatility(prices)
    
    # Prepare data for analysis
    price_data = ""
    if prices and len(prices) > 0:
        price_data = f"Current price: {prices[-1].get('close', 'N/A')}\n"
        if len(prices) > 5:
            price_data += f"5-day change: {((prices[-1].get('close', 0) / prices[-5].get('close', 1)) - 1) * 100:.2f}%\n"
        if len(prices) > 20:
            price_data += f"20-day change: {((prices[-1].get('close', 0) / prices[-20].get('close', 1)) - 1) * 100:.2f}%\n"
    
    metrics_data = ""
    if metrics and len(metrics) > 0:
        latest_metrics = metrics[0]
        metrics_data = f"ROE: {latest_metrics.get('return_on_equity', 'N/A')}\n"
        metrics_data += f"Debt/Equity: {latest_metrics.get('debt_to_equity', 'N/A')}\n"
        metrics_data += f"Operating Margin: {latest_metrics.get('operating_margin', 'N/A')}\n"
        metrics_data += f"Current Ratio: {latest_metrics.get('current_ratio', 'N/A')}\n"
    
    # Prepare technical analysis data
    technical_data = f"""
Trend Analysis:
- Trend: {trend_analysis.get('trend', 'unknown')}
- Strength: {trend_analysis.get('strength', 0):.2f}%
- Confidence: {trend_analysis.get('confidence', 0):.0f}%

Momentum Analysis:
- Momentum: {momentum_analysis.get('momentum', 'unknown')}
- RSI: {momentum_analysis.get('rsi', 50):.2f}
- Price Change: {momentum_analysis.get('price_change', 0):.2f}%

Volatility Analysis:
- Volatility: {volatility_analysis.get('volatility', 'unknown')}
- Bandwidth: {volatility_analysis.get('bandwidth', 0):.2f}%
- Position: {volatility_analysis.get('position', 50):.2f}%
"""
    
    messages = [
        (
            "system",
            """You are a professional trading analyst. Analyze the market data and provide insights on the current market conditions for this stock.
            
            Focus on:
            1. Price trends and momentum
            2. Key technical indicators
            3. Volume patterns
            4. Market sentiment
            5. Relative strength vs market
            
            Be concise but thorough in your analysis.""",
        ),
        (
            "human",
            f"""Analyze the market conditions for {ticker.get('symbol')} ({ticker.get('short_name')}):
            
            PRICE DATA:
            {price_data}
            
            FINANCIAL METRICS:
            {metrics_data}
            
            TECHNICAL ANALYSIS:
            {technical_data}
            
            Provide a comprehensive market analysis focusing on technical indicators, price trends, and momentum.
            """,
        ),
    ]
    
    response = await ainvoke(messages, config)
    
    # Store analysis in context
    analysis_data = context.get('analysis_data', {})
    analysis_data['market_analysis'] = response.content
    analysis_data['technical_analysis'] = {
        'trend': trend_analysis,
        'momentum': momentum_analysis,
        'volatility': volatility_analysis
    }
    context['analysis_data'] = analysis_data
    
    return {
        "messages": [response],
        "context": context
    }


async def sentiment_analysis(state: AgentState, config: RunnableConfig):
    context = state.get('context')
    ticker = context.get('current_task').get('ticker')
    news = context.get('news', [])
    insider_transactions = context.get('insider_transactions', [])
    
    # Prepare news data
    news_data = ""
    if news and len(news) > 0:
        for item in news[:10]:  # Last 10 news items
            news_data += f"{item.get('pub_date', '')}: {item.get('title', '')}\n"
    
    # Prepare insider transaction data
    insider_data = ""
    if insider_transactions and len(insider_transactions) > 0:
        for item in insider_transactions[:10]:  # Last 10 transactions
            # Using the correct attribute names based on the finance-data model
            date = item.get('start_date', item.get('date', ''))
            name = item.get('insider', item.get('name', ''))
            transaction_code = item.get('transaction', item.get('transactionCode', ''))
            change = item.get('shares', item.get('change', ''))
            insider_data += f"{date}: {name} {transaction_code} {change} shares\n"
    
    messages = [
        (
            "system",
            """You are a sentiment analysis expert. Analyze the news and insider transaction data to assess market sentiment for this stock.
            
            Focus on:
            1. Overall sentiment from news (positive, negative, neutral)
            2. Key themes in recent news
            3. Insider transaction patterns
            4. Sentiment trends over time
            5. Potential catalysts or risks
            
            Be concise but thorough in your analysis.""",
        ),
        (
            "human",
            f"""Analyze the market sentiment for {ticker.get('symbol')} ({ticker.get('short_name')}):
            
            RECENT NEWS:
            {news_data}
            
            INSIDER TRANSACTIONS:
            {insider_data}
            
            Provide a comprehensive sentiment analysis focusing on news sentiment and insider activity.
            """,
        ),
    ]
    
    response = await ainvoke(messages, config)
    
    # Store analysis in context
    analysis_data = context.get('analysis_data', {})
    analysis_data['sentiment_analysis'] = response.content
    context['analysis_data'] = analysis_data
    
    return {
        "messages": [response],
        "context": context
    }


async def risk_analysis(state: AgentState, config: RunnableConfig):
    context = state.get('context')
    ticker = context.get('current_task').get('ticker')
    metrics = context.get('metrics', [])
    
    metrics_data = ""
    if metrics and len(metrics) > 0:
        latest_metrics = metrics[0]
        metrics_data = f"Debt/Equity: {latest_metrics.get('debt_to_equity', 'N/A')}\n"
        metrics_data += f"Current Ratio: {latest_metrics.get('current_ratio', 'N/A')}\n"
        metrics_data += f"ROE: {latest_metrics.get('return_on_equity', 'N/A')}\n"
        metrics_data += f"Operating Margin: {latest_metrics.get('operating_margin', 'N/A')}\n"
    
    messages = [
        (
            "system",
            """You are a risk analysis expert. Analyze the financial metrics to assess the risk profile of this stock.
            
            Focus on:
            1. Financial risk (debt levels, liquidity)
            2. Operational risk (profitability, efficiency)
            3. Market risk (volatility, beta)
            4. Sector/industry risks
            5. Overall risk rating
            
            Be concise but thorough in your analysis.""",
        ),
        (
            "human",
            f"""Analyze the risk profile for {ticker.get('symbol')} ({ticker.get('short_name')}):
            
            FINANCIAL METRICS:
            {metrics_data}
            
            Provide a comprehensive risk analysis focusing on financial, operational, and market risks.
            """,
        ),
    ]
    
    response = await ainvoke(messages, config)
    
    # Store analysis in context
    analysis_data = context.get('analysis_data', {})
    analysis_data['risk_analysis'] = response.content
    context['analysis_data'] = analysis_data
    
    return {
        "messages": [response],
        "context": context
    }


async def generate_trading_signal(state: AgentState, config: RunnableConfig):
    context = state.get('context')
    ticker = context.get('current_task').get('ticker')
    analysis_data = context.get('analysis_data', {})
    
    market_analysis = analysis_data.get('market_analysis', '')
    sentiment_analysis = analysis_data.get('sentiment_analysis', '')
    risk_analysis = analysis_data.get('risk_analysis', '')
    technical_analysis = analysis_data.get('technical_analysis', {})
    
    # Prepare technical analysis summary
    technical_summary = ""
    if technical_analysis:
        trend = technical_analysis.get('trend', {})
        momentum = technical_analysis.get('momentum', {})
        volatility = technical_analysis.get('volatility', {})
        
        technical_summary = f"""
TECHNICAL ANALYSIS SUMMARY:
Trend: {trend.get('trend', 'unknown')} (Strength: {trend.get('strength', 0):.2f}%, Confidence: {trend.get('confidence', 0):.0f}%)
Momentum: {momentum.get('momentum', 'unknown')} (RSI: {momentum.get('rsi', 50):.2f})
Volatility: {volatility.get('volatility', 'unknown')} (Bandwidth: {volatility.get('bandwidth', 0):.2f}%)
"""
    
    messages = [
        (
            "system",
            """You are a professional trading strategist. Based on the comprehensive analysis, generate a clear trading signal.
            
            Your response must be in exactly this JSON format:
            ```AnalysisResult
            {
              "signal": "bullish" | "bearish" | "neutral",
              "confidence": float between 0 and 100,
              "position_size": float between 0 and 100,
              "stop_loss": float,
              "take_profit": float
            }
            ```
            
            Then provide a detailed explanation of your recommendation.
            
            Consider:
            1. Market trends and momentum
            2. Sentiment indicators
            3. Risk factors
            4. Technical analysis signals
            5. Position sizing based on risk
            6. Entry/exit points
            """,
        ),
        (
            "human",
            f"""Generate a trading signal for {ticker.get('symbol')} ({ticker.get('short_name')}):
            
            MARKET ANALYSIS:
            {market_analysis}
            
            SENTIMENT ANALYSIS:
            {sentiment_analysis}
            
            RISK ANALYSIS:
            {risk_analysis}
            
            TECHNICAL ANALYSIS:
            {technical_summary}
            
            Provide your trading recommendation with a clear signal, confidence level, and risk management parameters.
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
workflow.add_node("market_analysis", market_analysis)
workflow.add_node("sentiment_analysis", sentiment_analysis)
workflow.add_node("risk_analysis", risk_analysis)
workflow.add_node("generate_trading_signal", generate_trading_signal)

workflow.add_edge("start_analysis", "market_analysis")
workflow.add_edge("market_analysis", "sentiment_analysis")
workflow.add_edge("sentiment_analysis", "risk_analysis")
workflow.add_edge("risk_analysis", "generate_trading_signal")

workflow.set_entry_point("start_analysis")
workflow.set_finish_point("generate_trading_signal")

# Compile the workflow graph
agent = workflow.compile()