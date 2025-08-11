from common.agent_state import AgentState
from langchain.schema import AIMessage
from langchain_core.callbacks import dispatch_custom_event
from langchain_core.runnables import RunnableConfig
from typing import Dict, Any

import time
from common import markdown
from langgraph.types import StreamWriter
import re


class NewsSentimentAnalysis():
    def __init__(self, options: Dict[str, Any]):
        self.options = options

    
    def analyze(self, news: list) -> dict[str, any]:
        """Analyze news sentiment for market perception."""
        result = {"score": 0, "max_score": 10, "details": [], "sentiment_breakdown": {}}
        if not news:
            result["details"].append('No news data available')
            return result

        # Get recent news (last 30 days)
        recent_news = []
        cutoff_date = time.time() - 30*24*60*60  # 30 days ago
        
        for article in news:
            if article.get('pub_date'):
                try:
                    # Parse date string to timestamp
                    pub_date = time.mktime(time.strptime(article['pub_date'], "%Y-%m-%d"))
                    if pub_date >= cutoff_date:
                        recent_news.append(article)
                except:
                    continue

        if not recent_news:
            result["details"].append('No recent news data available')
            return result

        # Analyze sentiment of recent news
        positive_count = 0
        negative_count = 0
        neutral_count = 0
        total_articles = len(recent_news)
        
        # Simple keyword-based sentiment analysis
        positive_keywords = [
            'strong', 'excellent', 'outstanding', 'beats', 'surpasses', 'exceeds', 'record', 
            'growth', 'increase', 'rise', 'gain', 'boost', 'success', 'win', 'positive',
            'bullish', 'upbeat', 'optimistic', 'promising', 'encouraging', 'improves',
            'advantage', 'benefit', 'profit', 'revenue', 'earnings', 'upgrade', 'buy'
        ]
        
        negative_keywords = [
            'weak', 'poor', 'disappointing', 'misses', 'falls short', 'decline', 'drop',
            'fall', 'loss', 'hurt', 'failure', 'negative', 'bearish', 'pessimistic',
            'concern', 'worry', 'problem', 'issue', 'challenge', 'risk', 'threat',
            'downgrade', 'sell', 'lawsuit', 'scandal', 'investigation', 'layoff'
        ]

        for article in recent_news:
            title = article.get('title', '').lower()
            summary = article.get('summary', '').lower()
            
            # Combine title and summary for analysis
            text = title + ' ' + summary
            
            positive_matches = sum(1 for keyword in positive_keywords if keyword in text)
            negative_matches = sum(1 for keyword in negative_keywords if keyword in text)
            
            if positive_matches > negative_matches:
                positive_count += 1
            elif negative_matches > positive_matches:
                negative_count += 1
            else:
                neutral_count += 1

        # Calculate sentiment score (0-10 scale)
        if total_articles > 0:
            positive_ratio = positive_count / total_articles
            negative_ratio = negative_count / total_articles
            
            # Score calculation: 
            # 0-10 scale where 5 is neutral, >5 is positive, <5 is negative
            score = 5 + (positive_ratio - negative_ratio) * 5
            score = max(0, min(10, score))  # Clamp between 0-10
            
            result["sentiment_breakdown"] = {
                "positive": positive_count,
                "negative": negative_count,
                "neutral": neutral_count,
                "total": total_articles
            }
            
            if score >= 7:
                result["score"] = score
                result["details"].append(f"Positive news sentiment ({positive_count}/{total_articles} positive articles)")
            elif score >= 3:
                result["score"] = score
                result["details"].append(f"Neutral news sentiment ({neutral_count}/{total_articles} neutral articles)")
            else:
                result["score"] = score
                result["details"].append(f"Negative news sentiment ({negative_count}/{total_articles} negative articles)")
        else:
            result["details"].append('Could not analyze news sentiment')

        return result

    def get_markdown(self, analysis:dict):
        """
        Convert analysis to markdown.
        """
        markdown_content = markdown.analysis_data(analysis)
        return markdown_content

    def __call__(self, state: AgentState, config: RunnableConfig, writer: StreamWriter) -> Dict[str, Any]:
        context = state.get('context')
        analysis_data = context.get('analysis_data')
        if analysis_data is None:
            analysis_data = {}
            context['analysis_data'] = analysis_data
        news = context.get('news')
        analysis = self.analyze(news)
        analysis['type'] = 'news_sentiment_analysis'
        analysis['title'] = f'News Sentiment Analysis'

        analysis_data['news_sentiment_analysis'] = analysis
        ai_message = AIMessage(content=self.get_markdown(analysis))
        return {
            "context": context,
            "messages": [
                ai_message
            ]
        }