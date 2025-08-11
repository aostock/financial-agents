from common.agent_state import AgentState
from langchain.schema import AIMessage
from langchain_core.callbacks import dispatch_custom_event
from langchain_core.runnables import RunnableConfig
from typing import Dict, Any

import time
from common import markdown
from langgraph.types import StreamWriter
import re


class CompositeSentimentAnalysis():
    def __init__(self, options: Dict[str, Any]):
        self.options = options

    
    def analyze(self, analysis_data: dict) -> dict[str, any]:
        """Combine all sentiment factors for a composite view."""
        result = {"score": 0, "max_score": 10, "details": [], "component_scores": {}}
        
        # Extract scores from individual analyses
        component_scores = {}
        
        if analysis_data.get('news_sentiment_analysis'):
            news_score = analysis_data['news_sentiment_analysis'].get('score', 0)
            component_scores['news'] = news_score
            
        if analysis_data.get('social_sentiment_analysis'):
            social_score = analysis_data['social_sentiment_analysis'].get('score', 0)
            component_scores['social'] = social_score
            
        if analysis_data.get('insider_sentiment_analysis'):
            insider_score = analysis_data['insider_sentiment_analysis'].get('score', 0)
            component_scores['insider'] = insider_score
            
        if analysis_data.get('technical_sentiment_analysis'):
            technical_score = analysis_data['technical_sentiment_analysis'].get('score', 0)
            component_scores['technical'] = technical_score

        result["component_scores"] = component_scores

        # Calculate weighted composite score
        # All components are equally weighted in this implementation
        total_score = 0
        valid_components = 0
        
        for score in component_scores.values():
            total_score += score
            valid_components += 1
            
        if valid_components > 0:
            composite_score = total_score / valid_components
            result["score"] = composite_score
            
            # Add descriptive details about component alignment
            positive_components = sum(1 for score in component_scores.values() if score >= 5)
            negative_components = sum(1 for score in component_scores.values() if score < 5)
            
            if composite_score >= 7:
                result["details"].append(f"Strongly positive composite sentiment ({positive_components}/{valid_components} components positive)")
            elif composite_score >= 5:
                result["details"].append(f"Moderately positive composite sentiment ({positive_components}/{valid_components} components positive)")
            elif composite_score >= 3:
                result["details"].append(f"Neutral composite sentiment with mixed signals")
            else:
                result["details"].append(f"Negative composite sentiment ({negative_components}/{valid_components} components negative)")
        else:
            result["details"].append('Could not calculate composite sentiment - no valid component scores')

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
        analysis = self.analyze(analysis_data)
        analysis['type'] = 'composite_sentiment_analysis'
        analysis['title'] = f'Composite Sentiment Analysis'

        analysis_data['composite_sentiment_analysis'] = analysis
        ai_message = AIMessage(content=self.get_markdown(analysis))
        return {
            "context": context,
            "messages": [
                ai_message
            ]
        }