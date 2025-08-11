from common.agent_state import AgentState
from langchain.schema import AIMessage
from langchain_core.callbacks import dispatch_custom_event
from langchain_core.runnables import RunnableConfig
from typing import Dict, Any

import time
from common import markdown
from langgraph.types import StreamWriter



class BookValueGrowthAnalysis():
    def __init__(self, options: Dict[str, Any]):
        self.options = options

    def _calculate_book_value_cagr(self,book_values: list) -> tuple[int, str]:
        """Helper function to safely calculate book value CAGR and return score + reasoning."""
        if len(book_values) < 2:
            return 0, "Insufficient data for CAGR calculation"
        
        oldest_bv, latest_bv = book_values[-1], book_values[0]
        years = len(book_values) - 1
        
        # Handle different scenarios
        if oldest_bv > 0 and latest_bv > 0:
            cagr = ((latest_bv / oldest_bv) ** (1/years)) - 1
            if cagr > 0.15:
                return 2, f"Excellent book value CAGR: {cagr:.1%}"
            elif cagr > 0.1:
                return 1, f"Good book value CAGR: {cagr:.1%}"
            else:
                return 0, f"Book value CAGR: {cagr:.1%}"
        elif oldest_bv < 0 < latest_bv:
            return 3, "Excellent: Company improved from negative to positive book value"
        elif oldest_bv > 0 > latest_bv:
            return 0, "Warning: Company declined from positive to negative book value"
        else:
            return 0, "Unable to calculate meaningful book value CAGR due to negative values"

    
    def analyze(self, financial_line_items: list) -> dict[str, any]:
        """Analyze book value per share growth - a key Buffett metric."""
        result = {"score": 0, "max_score": 6, "details": []}
        if len(financial_line_items) < 3:
            result["details"].append("Insufficient data for book value analysis")
            return result
        
        # Extract book values per share
        book_values = [
            item.get('stockholders_equity') / item.get('ordinary_shares_number')
            for item in financial_line_items
            if hasattr(item, 'stockholders_equity') and hasattr(item, 'ordinary_shares_number')
            and item.get('stockholders_equity') and item.get('ordinary_shares_number')
        ]
        
        if len(book_values) < 3:
            result["details"].append("Insufficient book value data for growth analysis")
            return result
        
        score = 0
        reasoning = []
        
        # Analyze growth consistency
        growth_periods = sum(1 for i in range(len(book_values) - 1) if book_values[i] > book_values[i + 1])
        growth_rate = growth_periods / (len(book_values) - 1)
        
        # Score based on consistency
        if growth_rate >= 0.8:
            score += 3
            reasoning.append("Consistent book value per share growth (Buffett's favorite metric)")
        elif growth_rate >= 0.6:
            score += 2
            reasoning.append("Good book value per share growth pattern")
        elif growth_rate >= 0.4:
            score += 1
            reasoning.append("Moderate book value per share growth")
        else:
            reasoning.append("Inconsistent book value per share growth")
        
        # Calculate and score CAGR
        cagr_score, cagr_reason = self._calculate_book_value_cagr(book_values)
        score += cagr_score
        reasoning.append(cagr_reason)
        
        result["score"] = score
        result["details"] = reasoning
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
        metrics = context.get('metrics')
        analysis = self.analyze(metrics)
        analysis['type'] = 'book_value_growth_analysis'
        analysis['title'] = 'Book value growth analysis'

        analysis_data['book_value_growth_analysis'] = analysis
        ai_message = AIMessage(content=self.get_markdown(analysis))
        return {
            "context": context,
            "messages": [
                ai_message
            ]
        }