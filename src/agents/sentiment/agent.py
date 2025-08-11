"""
This is the main entry point for the Sentiment agent.
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
from agents.sentiment.news_sentiment_analysis import NewsSentimentAnalysis
from agents.sentiment.social_sentiment_analysis import SocialSentimentAnalysis
from agents.sentiment.insider_sentiment_analysis import InsiderSentimentAnalysis
from agents.sentiment.technical_sentiment_analysis import TechnicalSentimentAnalysis
from agents.sentiment.composite_sentiment_analysis import CompositeSentimentAnalysis

from nodes.next_step_suggestions import NextStepSuggestions

from nodes.ticker_search import TickerSearch
from typing_extensions import Literal
from common import markdown
from common.dataset import Dataset

next_step_suggestions_node = NextStepSuggestions({})
news_sentiment_analysis_node = NewsSentimentAnalysis({})
social_sentiment_analysis_node = SocialSentimentAnalysis({})
insider_sentiment_analysis_node = InsiderSentimentAnalysis({})
technical_sentiment_analysis_node = TechnicalSentimentAnalysis({})
composite_sentiment_analysis_node = CompositeSentimentAnalysis({})

async def start_analysis(state: AgentState, config: RunnableConfig):
    
    end_date = state.get('action').get('parameters').get('end_date')
    end_date = end_date if end_date else time.strftime("%Y-%m-%d")

    context = state.get('context')
    ticker = context.get('current_task').get('ticker')
    
    # Create dataset client
    dataset_client = Dataset(config)
    
    # Get news data for sentiment analysis
    news = dataset_client.get_news(ticker.get('symbol'), end_date)
    
    # Get insider transactions data
    insider_transactions = dataset_client.get_insider_transactions(ticker.get('symbol'), end_date)
    
    # Get price data for technical analysis
    # Get 30 days of price data for technical indicators
    start_date = time.strftime("%Y-%m-%d", time.localtime(time.time() - 30*24*60*60))
    prices = dataset_client.get_prices(ticker.get('symbol'), start_date, end_date)
    
    context['news'] = news
    context['insider_transactions'] = insider_transactions
    context['prices'] = prices
    
    return {
        'context': context,
        'messages':[AIMessage(content=markdown.to_h2('Sentiment Analysis for '+ ticker.get('symbol')))]
    }

async def end_analysis(state: AgentState, config: RunnableConfig):
    context = state.get('context')
    ticker = context.get('current_task').get('ticker')
    analysis_data = context.get('analysis_data')

    # Calculate total sentiment score
    total_score = 0
    max_possible_score = 0
    
    if analysis_data.get('news_sentiment_analysis'):
        total_score += analysis_data.get('news_sentiment_analysis').get("score", 0)
        max_possible_score += analysis_data.get('news_sentiment_analysis').get("max_score", 10)
    
    if analysis_data.get('social_sentiment_analysis'):
        total_score += analysis_data.get('social_sentiment_analysis').get("score", 0)
        max_possible_score += analysis_data.get('social_sentiment_analysis').get("max_score", 10)
    
    if analysis_data.get('insider_sentiment_analysis'):
        total_score += analysis_data.get('insider_sentiment_analysis').get("score", 0)
        max_possible_score += analysis_data.get('insider_sentiment_analysis').get("max_score", 10)
    
    if analysis_data.get('technical_sentiment_analysis'):
        total_score += analysis_data.get('technical_sentiment_analysis').get("score", 0)
        max_possible_score += analysis_data.get('technical_sentiment_analysis').get("max_score", 10)

    analysis_data['total_score'] = total_score
    analysis_data['max_possible_score'] = max_possible_score

    messages = [
            (
                "system",
                """You are a professional market sentiment analyst. Analyze market sentiment using multiple data sources to provide a comprehensive view of market perception for investment decisions:

                YOUR ROLE:
                You analyze market sentiment from multiple angles to provide a holistic view of how the market perceives a company. Your analysis helps investors understand the emotional and psychological factors driving market movements.

                SENTIMENT ANALYSIS SOURCES:
                1. News Sentiment: Analyze tone and sentiment in recent news articles
                2. Social Media Sentiment: Evaluate public perception from social media discussions
                3. Insider Activity: Interpret buying/selling patterns by company insiders
                4. Technical Indicators: Assess market momentum and price action signals
                5. Composite Analysis: Combine all factors for overall sentiment assessment

                SENTIMENT INTERPRETATION:
                - News Sentiment: Positive headlines and content indicate bullish sentiment
                - Social Sentiment: High engagement and positive discussions suggest market enthusiasm
                - Insider Activity: Insider buying suggests confidence; insider selling may indicate concerns
                - Technical Indicators: Price momentum and volume patterns reflect market psychology
                - Composite View: Overall market perception combining all factors

                SIGNAL INTERPRETATION:
                - Bullish: Strong positive sentiment across multiple sources
                - Bearish: Predominantly negative sentiment with concerning patterns
                - Neutral: Mixed signals or moderate sentiment levels

                CONFIDENCE LEVELS:
                - 90-100%: Clear, consistent sentiment signals across all sources
                - 70-89%: Strong sentiment with good supporting evidence
                - 50-69%: Mixed signals or moderate sentiment levels
                - 30-49%: Weak sentiment signals or conflicting indicators
                - 10-29%: Unclear sentiment or insufficient data

                Remember: Sentiment analysis provides insights into market psychology but should be combined with fundamental analysis for comprehensive investment decisions.
                """,
            ),
            (
                "human",
                f"""Analyze market sentiment for {ticker.get('symbol')} ({ticker.get('short_name')}):

                COMPREHENSIVE SENTIMENT ANALYSIS DATA:
                {analysis_data}

                Please provide your sentiment assessment in exactly this JSON format, notice to use 'SentimentResult' before json:
                ```SentimentResult
                {{
                  "signal": "bullish" | "bearish" | "neutral",
                  "confidence": float between 0 and 100
                }}
                ```
                then provide a detailed reasoning for your assessment.

                In your reasoning, be specific about:
                1. News sentiment trends and key themes in recent coverage
                2. Social media perception and public engagement levels
                3. Insider trading activity and what it suggests about management confidence
                4. Technical price patterns and market momentum
                5. How these factors combine to form the overall sentiment view
                6. Any key risks or opportunities highlighted by the sentiment analysis

                Provide a clear, concise assessment of market sentiment that would be valuable for investment decision-making.
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

workflow.add_node("news_sentiment_analysis", news_sentiment_analysis_node)
workflow.add_node("social_sentiment_analysis", social_sentiment_analysis_node)
workflow.add_node("insider_sentiment_analysis", insider_sentiment_analysis_node)
workflow.add_node("technical_sentiment_analysis", technical_sentiment_analysis_node)
workflow.add_node("composite_sentiment_analysis", composite_sentiment_analysis_node)

workflow.add_node("end_analysis", end_analysis)

workflow.add_edge("start_analysis", "news_sentiment_analysis")
workflow.add_edge("news_sentiment_analysis", "social_sentiment_analysis")
workflow.add_edge("social_sentiment_analysis", "insider_sentiment_analysis")
workflow.add_edge("insider_sentiment_analysis", "technical_sentiment_analysis")
workflow.add_edge("technical_sentiment_analysis", "composite_sentiment_analysis")
workflow.add_edge("composite_sentiment_analysis", "end_analysis")

workflow.set_entry_point("start_analysis")
workflow.set_finish_point("end_analysis")
# Compile the workflow graph
agent = workflow.compile()