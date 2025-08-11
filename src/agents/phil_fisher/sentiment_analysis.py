from common.agent_state import AgentState
from langchain.schema import AIMessage
from langchain_core.callbacks import dispatch_custom_event
from langchain_core.runnables import RunnableConfig
from typing import Dict, Any

import time
from common import markdown
from langgraph.types import StreamWriter


class SentimentAnalysis():
    def __init__(self, options: Dict[str, Any]):
        self.options = options

    
    def analyze(self, news_items: list) -> dict[str, any]:
        """Analyze sentiment based on Phil Fisher's criteria."""
        if not news_items:
            return {"score": 5, "max_score": 10, "details": ["No news data; defaulting to neutral sentiment"]}

        negative_keywords = ["lawsuit", "fraud", "negative", "downturn", "decline", "investigation", "recall", "bankruptcy", "loss"]
        positive_keywords = ["innovation", "partnership", "breakthrough", "expansion", "acquisition", "growth", "profit", "success"]
        
        negative_count = 0
        positive_count = 0
        
        for news in news_items:
            title = news.get('title', '') if news else ''
            title_lower = title.lower()
            
            if any(word in title_lower for word in negative_keywords):
                negative_count += 1
            elif any(word in title_lower for word in positive_keywords):
                positive_count += 1

        reasoning = []
        total_news = len(news_items)
        
        if negative_count > total_news * 0.3:
            score = 3
            reasoning.append(f"High proportion of negative headlines: {negative_count}/{total_news}")
        elif negative_count > total_news * 0.1:
            score = 4
            reasoning.append(f"Moderate proportion of negative headlines: {negative_count}/{total_news}")
        elif positive_count > total_news * 0.3:
            score = 8
            reasoning.append(f"High proportion of positive headlines: {positive_count}/{total_news}")
        elif positive_count > total_news * 0.1:
            score = 7
            reasoning.append(f"Moderate proportion of positive headlines: {positive_count}/{total_news}")
        else:
            score = 6
            reasoning.append(f"Mostly neutral headlines: {positive_count} positive, {negative_count} negative, {total_news-positive_count-negative_count} neutral")

        return {"score": score, "max_score": 10, "details": reasoning}

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
        analysis['type'] = 'sentiment_analysis'
        analysis['title'] = f'Sentiment Analysis'

        analysis_data['sentiment_analysis'] = analysis
        ai_message = AIMessage(content=self.get_markdown(analysis))
        return {
            "context": context,
            "messages": [
                ai_message
            ]
        }