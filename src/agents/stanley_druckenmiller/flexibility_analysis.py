from common.agent_state import AgentState
from langchain.schema import AIMessage
from langchain_core.callbacks import dispatch_custom_event
from langchain_core.runnables import RunnableConfig
from typing import Dict, Any

import time
from common import markdown
from langgraph.types import StreamWriter


class FlexibilityAnalysis():
    def __init__(self, options: Dict[str, Any]):
        self.options = options

    
    def analyze(self, metrics: list) -> dict[str, any]:
        """Analyze business model flexibility and strategic adaptability."""
        result = {"score": 0, "max_score": 10, "details": []}
        if not metrics:
            result["details"].append('No metrics available for flexibility analysis')
            return result

        latest_metrics = metrics[0]
        score = 0
        reasoning = []
        
        # Revenue diversification (inferred from business model)
        if latest_metrics.get('revenue') and latest_metrics.get('gross_profit'):
            gross_margin = latest_metrics.get('gross_margin', 0)
            # Higher gross margins often indicate more flexible business models
            if gross_margin > 0.5:
                score += 2
                reasoning.append(f"High gross margin business ({gross_margin:.1%}) - typically more flexible")
            elif gross_margin > 0.3:
                score += 1
                reasoning.append(f"Healthy gross margin ({gross_margin:.1%}) - moderate flexibility")
            elif gross_margin < 0.1:
                score -= 1
                reasoning.append(f"Low gross margin ({gross_margin:.1%}) - less flexible business model")
            else:
                reasoning.append(f"Standard gross margin ({gross_margin:.1%})")
        
        # Capital expenditure requirements (operational flexibility)
        if latest_metrics.get('capital_expenditure') and latest_metrics.get('net_income'):
            capex_to_income = abs(latest_metrics['capital_expenditure']) / abs(latest_metrics['net_income']) if latest_metrics['net_income'] != 0 else 0
            if capex_to_income < 0.5:
                score += 2
                reasoning.append(f"Low capex intensity ({capex_to_income:.1f}) - high operational flexibility")
            elif capex_to_income < 1.0:
                score += 1
                reasoning.append(f"Moderate capex intensity ({capex_to_income:.1f}) - reasonable flexibility")
            elif capex_to_income > 2.0:
                score -= 2
                reasoning.append(f"High capex intensity ({capex_to_income:.1f}) - lower operational flexibility")
            else:
                reasoning.append(f"Standard capex intensity ({capex_to_income:.1f})")
        
        # Working capital management (financial flexibility)
        if latest_metrics.get('current_ratio') and latest_metrics.get('asset_turnover'):
            # Companies with good current ratios and asset turnover are typically more flexible
            if latest_metrics['current_ratio'] > 1.5 and latest_metrics['asset_turnover'] > 0.8:
                score += 2
                reasoning.append("Strong working capital management - financial flexibility")
            elif latest_metrics['current_ratio'] < 1.0 or latest_metrics['asset_turnover'] < 0.5:
                score -= 1
                reasoning.append("Working capital challenges - reduced flexibility")
            else:
                reasoning.append("Adequate working capital management")
        
        # Debt flexibility (ability to take on more debt if needed)
        if latest_metrics.get('debt_to_equity'):
            debt_ratio = latest_metrics['debt_to_equity']
            if debt_ratio < 0.3:
                score += 2
                reasoning.append(f"Low debt levels provide financial flexibility (D/E: {debt_ratio:.2f})")
            elif debt_ratio > 1.0:
                score -= 2
                reasoning.append(f"High debt constrains financial flexibility (D/E: {debt_ratio:.2f})")
            else:
                reasoning.append(f"Moderate debt levels (D/E: {debt_ratio:.2f})")
        
        # Cash generation ability (strategic flexibility)
        if latest_metrics.get('free_cash_flow') and latest_metrics.get('net_income'):
            if latest_metrics['net_income'] > 0:  # Avoid division by zero
                fcf_conversion = latest_metrics['free_cash_flow'] / latest_metrics['net_income'] * 100
                if fcf_conversion > 80:
                    score += 2
                    reasoning.append(f"High cash conversion ({fcf_conversion:.1f}%) - strategic flexibility")
                elif fcf_conversion > 50:
                    score += 1
                    reasoning.append(f"Good cash conversion ({fcf_conversion:.1f}%) - adequate flexibility")
                elif fcf_conversion < 20:
                    score -= 2
                    reasoning.append(f"Low cash conversion ({fcf_conversion:.1f}%) - limited flexibility")
                else:
                    reasoning.append(f"Moderate cash conversion ({fcf_conversion:.1f}%)")
            else:
                reasoning.append("Not profitable - cash generation analysis not meaningful")
        
        # Market position flexibility (size and market cap)
        if latest_metrics.get('market_cap'):
            market_cap = latest_metrics['market_cap']
            if market_cap > 100000000000:  # > $100B
                score += 1
                reasoning.append("Large market cap provides strategic options and flexibility")
            elif market_cap < 2000000000:  # < $2B
                score -= 1
                reasoning.append("Small market cap may limit strategic flexibility")
            else:
                reasoning.append("Mid-cap size provides reasonable strategic flexibility")

        result["score"] = max(0, min(10, score))  # Clamp between 0-10
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
        analysis['type'] = 'flexibility_analysis'
        analysis['title'] = f'Flexibility Analysis'

        analysis_data['flexibility_analysis'] = analysis
        ai_message = AIMessage(content=self.get_markdown(analysis))
        return {
            "context": context,
            "messages": [
                ai_message
            ]
        }