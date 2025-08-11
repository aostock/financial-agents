from common.agent_state import AgentState
from langchain.schema import AIMessage
from langchain_core.callbacks import dispatch_custom_event
from langchain_core.runnables import RunnableConfig
from typing import Dict, Any

import time
from common import markdown
from langgraph.types import StreamWriter


class PortfolioAnalysis():
    def __init__(self, options: Dict[str, Any]):
        self.options = options

    
    def analyze(self, metrics: list, prices: list, info: dict) -> dict[str, any]:
        """Analyze portfolio positioning based on comprehensive data."""
        result = {"score": 0, "max_score": 10, "details": []}
        if not metrics:
            result["details"].append('No metrics available for portfolio analysis')
            return result

        latest_metrics = metrics[0] if metrics else {}
        current_price = prices[0].get('close') if prices else 0

        score = 0
        reasoning = []

        # 1. Valuation Analysis
        pe_ratio = latest_metrics.get('price_to_earnings_ratio')
        if pe_ratio:
            if pe_ratio < 15:
                score += 2
                reasoning.append(f"Attractive valuation (P/E: {pe_ratio:.1f})")
            elif pe_ratio < 25:
                score += 1
                reasoning.append(f"Fair valuation (P/E: {pe_ratio:.1f})")
            else:
                reasoning.append(f"High valuation (P/E: {pe_ratio:.1f})")
        else:
            reasoning.append("P/E ratio data not available")

        # 2. Financial Health
        debt_to_equity = latest_metrics.get('debt_to_equity')
        current_ratio = latest_metrics.get('current_ratio')
        
        if debt_to_equity is not None and current_ratio:
            if debt_to_equity < 0.5 and current_ratio > 1.5:
                score += 2
                reasoning.append(f"Strong financial health (D/E: {debt_to_equity:.2f}, Current: {current_ratio:.1f})")
            elif debt_to_equity < 1.0 and current_ratio > 1.0:
                score += 1
                reasoning.append(f"Good financial health (D/E: {debt_to_equity:.2f}, Current: {current_ratio:.1f})")
            else:
                reasoning.append(f"Weak financial health (D/E: {debt_to_equity:.2f}, Current: {current_ratio:.1f})")
        else:
            reasoning.append("Financial health data not available")

        # 3. Profitability
        roe = latest_metrics.get('return_on_equity')
        operating_margin = latest_metrics.get('operating_margin')
        
        if roe and operating_margin:
            if roe > 0.15 and operating_margin > 0.15:
                score += 2
                reasoning.append(f"Exceptional profitability (ROE: {roe:.1%}, Op. Margin: {operating_margin:.1%})")
            elif roe > 0.10 and operating_margin > 0.10:
                score += 1
                reasoning.append(f"Good profitability (ROE: {roe:.1%}, Op. Margin: {operating_margin:.1%})")
            else:
                reasoning.append(f"Weak profitability (ROE: {roe:.1%}, Op. Margin: {operating_margin:.1%})")
        else:
            reasoning.append("Profitability data not available")

        # 4. Growth Potential
        revenue = latest_metrics.get('revenue')
        net_income = latest_metrics.get('net_income')
        
        if revenue and net_income and len(metrics) > 1:
            prev_revenue = metrics[1].get('revenue')
            prev_net_income = metrics[1].get('net_income')
            
            if prev_revenue and prev_net_income:
                revenue_growth = (revenue - prev_revenue) / prev_revenue
                income_growth = (net_income - prev_net_income) / prev_net_income
                
                if revenue_growth > 0.10 and income_growth > 0.10:
                    score += 2
                    reasoning.append(f"Strong growth (Revenue: {revenue_growth:.1%}, Income: {income_growth:.1%})")
                elif revenue_growth > 0.05 and income_growth > 0.05:
                    score += 1
                    reasoning.append(f"Moderate growth (Revenue: {revenue_growth:.1%}, Income: {income_growth:.1%})")
                else:
                    reasoning.append(f"Slow growth (Revenue: {revenue_growth:.1%}, Income: {income_growth:.1%})")
            else:
                reasoning.append("Growth data incomplete")
        else:
            reasoning.append("Growth data not available")

        # 5. Market Position
        market_cap = latest_metrics.get('market_cap')
        beta = latest_metrics.get('beta')
        
        if market_cap and beta:
            if market_cap > 10000000000 and beta < 1.2:
                score += 1
                reasoning.append(f"Large-cap with stable volatility (Market Cap: ${market_cap:,.0f}, Beta: {beta:.2f})")
            elif market_cap > 10000000000:
                score += 1
                reasoning.append(f"Large-cap company (Market Cap: ${market_cap:,.0f})")
            else:
                reasoning.append(f"Small/mid-cap company (Market Cap: ${market_cap:,.0f})")
        else:
            reasoning.append("Market position data not available")

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
        prices = context.get('prices')
        info = context.get('info')
        analysis = self.analyze(metrics, prices, info)
        analysis['type'] = 'portfolio_analysis'
        analysis['title'] = f'Portfolio Analysis'

        analysis_data['portfolio_analysis'] = analysis
        ai_message = AIMessage(content=self.get_markdown(analysis))
        return {
            "context": context,
            "messages": [
                ai_message
            ]
        }