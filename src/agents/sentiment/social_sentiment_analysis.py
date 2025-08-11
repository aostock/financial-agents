from common.agent_state import AgentState
from langchain.schema import AIMessage
from langchain_core.callbacks import dispatch_custom_event
from langchain_core.runnables import RunnableConfig
from typing import Dict, Any

import time
from common import markdown
from langgraph.types import StreamWriter
import re


class SocialSentimentAnalysis():
    def __init__(self, options: Dict[str, Any]):
        self.options = options

    
    def analyze(self, news: list) -> dict[str, any]:
        """Analyze social sentiment based on news engagement and discussion patterns."""
        result = {"score": 0, "max_score": 10, "details": [], "engagement_metrics": {}}
        if not news:
            result["details"].append('No news data available for social sentiment analysis')
            return result

        # Get recent news (last 30 days) with engagement data
        recent_news = []
        cutoff_date = time.time() - 30*24*60*60  # 30 days ago
        
        for article in news:
            if article.get('pub_date') and (article.get('share_count') or article.get('comment_count')):
                try:
                    # Parse date string to timestamp
                    pub_date = time.mktime(time.strptime(article['pub_date'], "%Y-%m-%d"))
                    if pub_date >= cutoff_date:
                        recent_news.append(article)
                except:
                    continue

        if not recent_news:
            result["details"].append('No recent news with engagement data available')
            return result

        # Analyze engagement patterns
        total_shares = 0
        total_comments = 0
        total_articles = len(recent_news)
        
        positive_engagement = 0
        neutral_engagement = 0
        negative_engagement = 0

        # Simple engagement-based sentiment analysis
        for article in recent_news:
            shares = article.get('share_count', 0) or 0
            comments = article.get('comment_count', 0) or 0
            
            total_shares += shares
            total_comments += comments
            
            # Articles with high engagement relative to others may indicate strong sentiment
            # This is a simplified approach - in reality, would need actual social media data
            
            # For now, we'll use a proxy based on the news sentiment analysis
            # In a real implementation, this would connect to social media APIs
            
            title = article.get('title', '').lower()
            summary = article.get('summary', '').lower()
            
            # Combine title and summary for analysis
            text = title + ' ' + summary
            
            # Simple keyword-based sentiment for social perception
            positive_keywords = [
                'popular', 'trending', 'viral', 'buzz', 'hype', 'excitement', 'enthusiasm',
                'love', 'support', 'endorsement', 'recommendation', 'praise', 'acclaim'
            ]
            
            negative_keywords = [
                'controversy', 'backlash', 'criticism', 'complaint', 'disappointment',
                'anger', 'frustration', 'concern', 'worry', 'panic', 'fear', 'doubt'
            ]
            
            positive_matches = sum(1 for keyword in positive_keywords if keyword in text)
            negative_matches = sum(1 for keyword in negative_keywords if keyword in text)
            
            if positive_matches > negative_matches:
                positive_engagement += 1
            elif negative_matches > positive_matches:
                negative_engagement += 1
            else:
                neutral_engagement += 1

        # Calculate engagement-based sentiment score
        if total_articles > 0:
            avg_shares = total_shares / total_articles
            avg_comments = total_comments / total_articles
            
            result["engagement_metrics"] = {
                "average_shares": avg_shares,
                "average_comments": avg_comments,
                "total_articles": total_articles
            }
            
            # Score based on sentiment of highly engaged content
            positive_ratio = positive_engagement / total_articles
            negative_ratio = negative_engagement / total_articles
            
            # Score calculation: 0-10 scale
            score = 5 + (positive_ratio - negative_ratio) * 5
            score = max(0, min(10, score))  # Clamp between 0-10
            
            result["score"] = score
            
            if score >= 7:
                result["details"].append(f"Positive social sentiment with good engagement (avg {avg_shares:.0f} shares, {avg_comments:.0f} comments per article)")
            elif score >= 3:
                result["details"].append(f"Neutral social sentiment with moderate engagement (avg {avg_shares:.0f} shares, {avg_comments:.0f} comments per article)")
            else:
                result["details"].append(f"Negative social sentiment with concerning engagement patterns (avg {avg_shares:.0f} shares, {avg_comments:.0f} comments per article)")
        else:
            result["details"].append('Could not analyze social sentiment')

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
        analysis['type'] = 'social_sentiment_analysis'
        analysis['title'] = f'Social Sentiment Analysis'

        analysis_data['social_sentiment_analysis'] = analysis
        ai_message = AIMessage(content=self.get_markdown(analysis))
        return {
            "context": context,
            "messages": [
                ai_message
            ]
        }