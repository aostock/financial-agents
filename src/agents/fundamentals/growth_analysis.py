from common.agent_state import AgentState
from langchain.schema import AIMessage
from langchain_core.callbacks import dispatch_custom_event
from langchain_core.runnables import RunnableConfig
from typing import Dict, Any

import time
from common import markdown
from langgraph.types import StreamWriter


class GrowthAnalysis():
    def __init__(self, options: Dict[str, Any]):
        self.options = options

    
    def analyze(self, metrics: list) -> dict[str, any]:
        """Analyze company growth potential based on historical growth and reinvestment metrics."""
        result = {"score": 0, "max_score": 10, "details": []}
        if not metrics or len(metrics) < 3:
            result["details"].append('Insufficient historical data (need at least 3 years)')
            return result

        score = 0
        reasoning = []

        # Calculate revenue CAGR (oldest to latest)
        revs = [m.get('revenue') for m in reversed(metrics) if m.get('revenue')]
        if len(revs) >= 2 and revs[0] and revs[0] > 0:
            rev_cagr = (revs[-1] / revs[0]) ** (1 / (len(revs) - 1)) - 1
        else:
            rev_cagr = None

        # Calculate earnings CAGR
        earnings = [m.get('net_income') for m in reversed(metrics) if m.get('net_income')]
        if len(earnings) >= 2 and earnings[0] and earnings[0] > 0:
            earnings_cagr = (earnings[-1] / earnings[0]) ** (1 / (len(earnings) - 1)) - 1
        else:
            earnings_cagr = None

        # Calculate book value growth
        book_values = [m.get('stockholders_equity') for m in reversed(metrics) if m.get('stockholders_equity')]
        if len(book_values) >= 2 and book_values[0] and book_values[0] > 0:
            book_cagr = (book_values[-1] / book_values[0]) ** (1 / (len(book_values) - 1)) - 1
        else:
            book_cagr = None

        # Evaluate revenue growth
        if rev_cagr is not None:
            if rev_cagr > 0.10:  # 10%+ CAGR
                score += 3
                reasoning.append(f"Strong revenue growth ({rev_cagr:.1%} CAGR)")
            elif rev_cagr > 0.05:  # 5%+ CAGR
                score += 2
                reasoning.append(f"Good revenue growth ({rev_cagr:.1%} CAGR)")
            elif rev_cagr > 0.02:  # 2%+ CAGR
                score += 1
                reasoning.append(f"Modest revenue growth ({rev_cagr:.1%} CAGR)")
            else:
                reasoning.append(f"Slow/negative revenue growth ({rev_cagr:.1%} CAGR)")

        # Evaluate earnings growth
        if earnings_cagr is not None:
            if earnings_cagr > 0.10:  # 10%+ CAGR
                score += 3
                reasoning.append(f"Strong earnings growth ({earnings_cagr:.1%} CAGR)")
            elif earnings_cagr > 0.05:  # 5%+ CAGR
                score += 2
                reasoning.append(f"Good earnings growth ({earnings_cagr:.1%} CAGR)")
            elif earnings_cagr > 0.02:  # 2%+ CAGR
                score += 1
                reasoning.append(f"Modest earnings growth ({earnings_cagr:.1%} CAGR)")
            else:
                reasoning.append(f"Slow/negative earnings growth ({earnings_cagr:.1%} CAGR)")

        # Evaluate book value growth
        if book_cagr is not None:
            if book_cagr > 0.10:  # 10%+ CAGR
                score += 2
                reasoning.append(f"Strong book value growth ({book_cagr:.1%} CAGR)")
            elif book_cagr > 0.05:  # 5%+ CAGR
                score += 1
                reasoning.append(f"Good book value growth ({book_cagr:.1%} CAGR)")
            else:
                reasoning.append(f"Slow/negative book value growth ({book_cagr:.1%} CAGR)")

        # Check reinvestment efficiency (if we have data)
        latest = metrics[0]
        if (latest.get('net_income') and latest.get('dividends_and_other_cash_distributions') and 
            latest.get('capital_expenditure') and latest.get('stockholders_equity')):
            
            # Calculate retained earnings
            retained_earnings = latest['net_income'] - latest['dividends_and_other_cash_distributions']
            
            # Calculate reinvestment rate
            if latest['net_income'] != 0:
                reinvestment_rate = retained_earnings / latest['net_income']
                
                # Calculate return on retained earnings (proxied by book value growth / reinvestment rate)
                if reinvestment_rate > 0 and book_cagr is not None:
                    rore = book_cagr / reinvestment_rate if reinvestment_rate != 0 else 0
                    if rore > 0.15:  # 15%+ return on retained earnings
                        score += 2
                        reasoning.append(f"High return on retained earnings ({rore:.1%})")
                    elif rore > 0.10:  # 10%+ return
                        score += 1
                        reasoning.append(f"Good return on retained earnings ({rore:.1%})")

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
        analysis['type'] = 'growth_analysis'
        analysis['title'] = f'Growth Analysis'

        analysis_data['growth_analysis'] = analysis
        ai_message = AIMessage(content=self.get_markdown(analysis))
        return {
            "context": context,
            "messages": [
                ai_message
            ]
        }