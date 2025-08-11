from common.agent_state import AgentState
from langchain.schema import AIMessage
from langchain_core.callbacks import dispatch_custom_event
from langchain_core.runnables import RunnableConfig
from typing import Dict, Any

import time
from common import markdown
from langgraph.types import StreamWriter


class ManagementQualityAnalysis():
    def __init__(self, options: Dict[str, Any]):
        self.options = options

    
    def analyze(self, metrics: list) -> dict[str, any]:
        """
        Evaluate management quality using Munger's criteria:
        - Capital allocation wisdom
        - Insider ownership and transactions
        - Cash management efficiency
        - Candor and transparency
        - Long-term focus
        """
        result = {"score": 0, "max_score": 10, "details": []}
        if not metrics:
            result["details"].append('No metrics available')
            return result

        score = 0
        details = []
        
        # 1. Capital allocation - Check FCF to net income ratio
        # Munger values companies that convert earnings to cash
        fcf_values = [item.get('free_cash_flow') for item in metrics 
                     if item.get('free_cash_flow') is not None]
        
        net_income_values = [item.get('net_income') for item in metrics 
                            if item.get('net_income') is not None]
        
        if fcf_values and net_income_values and len(fcf_values) == len(net_income_values):
            # Calculate FCF to Net Income ratio for each period
            fcf_to_ni_ratios = []
            for i in range(len(fcf_values)):
                if net_income_values[i] and net_income_values[i] > 0:
                    fcf_to_ni_ratios.append(fcf_values[i] / net_income_values[i])
            
            if fcf_to_ni_ratios:
                avg_ratio = sum(fcf_to_ni_ratios) / len(fcf_to_ni_ratios)
                if avg_ratio > 1.1:  # FCF > net income suggests good accounting
                    score += 3
                    details.append(f"Excellent cash conversion: FCF/NI ratio of {avg_ratio:.2f}")
                elif avg_ratio > 0.9:  # FCF roughly equals net income
                    score += 2
                    details.append(f"Good cash conversion: FCF/NI ratio of {avg_ratio:.2f}")
                elif avg_ratio > 0.7:  # FCF somewhat lower than net income
                    score += 1
                    details.append(f"Moderate cash conversion: FCF/NI ratio of {avg_ratio:.2f}")
                else:
                    details.append(f"Poor cash conversion: FCF/NI ratio of only {avg_ratio:.2f}")
            else:
                details.append("Could not calculate FCF to Net Income ratios")
        else:
            details.append("Missing FCF or Net Income data")
        
        # 2. Debt management - Munger is cautious about debt
        debt_values = [item.get('total_debt') for item in metrics 
                      if item.get('total_debt') is not None]
        
        equity_values = [item.get('shareholders_equity') for item in metrics 
                        if item.get('shareholders_equity') is not None]
        
        if debt_values and equity_values and len(debt_values) == len(equity_values):
            # Calculate D/E ratio for most recent period
            recent_de_ratio = debt_values[0] / equity_values[0] if equity_values[0] > 0 else float('inf')
            
            if recent_de_ratio < 0.3:  # Very low debt
                score += 3
                details.append(f"Conservative debt management: D/E ratio of {recent_de_ratio:.2f}")
            elif recent_de_ratio < 0.7:  # Moderate debt
                score += 2
                details.append(f"Prudent debt management: D/E ratio of {recent_de_ratio:.2f}")
            elif recent_de_ratio < 1.5:  # Higher but still reasonable debt
                score += 1
                details.append(f"Moderate debt level: D/E ratio of {recent_de_ratio:.2f}")
            else:
                details.append(f"High debt level: D/E ratio of {recent_de_ratio:.2f}")
        else:
            details.append("Missing debt or equity data")
        
        # 3. Cash management efficiency - Munger values appropriate cash levels
        cash_values = [item.get('cash_and_equivalents') for item in metrics
                      if item.get('cash_and_equivalents') is not None]
        revenue_values = [item.get('revenue') for item in metrics
                         if item.get('revenue') is not None]
        
        if cash_values and revenue_values and len(cash_values) > 0 and len(revenue_values) > 0:
            # Calculate cash to revenue ratio (Munger likes 10-20% for most businesses)
            cash_to_revenue = cash_values[0] / revenue_values[0] if revenue_values[0] > 0 else 0
            
            if 0.1 <= cash_to_revenue <= 0.25:
                # Goldilocks zone - not too much, not too little
                score += 2
                details.append(f"Prudent cash management: Cash/Revenue ratio of {cash_to_revenue:.2f}")
            elif 0.05 <= cash_to_revenue < 0.1 or 0.25 < cash_to_revenue <= 0.4:
                # Reasonable but not ideal
                score += 1
                details.append(f"Acceptable cash position: Cash/Revenue ratio of {cash_to_revenue:.2f}")
            elif cash_to_revenue > 0.4:
                # Too much cash - potentially inefficient capital allocation
                details.append(f"Excess cash reserves: Cash/Revenue ratio of {cash_to_revenue:.2f}")
            else:
                # Too little cash - potentially risky
                details.append(f"Low cash reserves: Cash/Revenue ratio of {cash_to_revenue:.2f}")
        else:
            details.append("Insufficient cash or revenue data")
        
        # 4. Consistency in share count - Munger prefers stable/decreasing shares
        share_counts = [item.get('outstanding_shares') for item in metrics
                       if item.get('outstanding_shares') is not None]
        
        if share_counts and len(share_counts) >= 3:
            if share_counts[0] < share_counts[-1] * 0.95:  # 5%+ reduction in shares
                score += 2
                details.append("Shareholder-friendly: Reducing share count over time")
            elif share_counts[0] < share_counts[-1] * 1.05:  # Stable share count
                score += 1
                details.append("Stable share count: Limited dilution")
            elif share_counts[0] > share_counts[-1] * 1.2:  # >20% dilution
                score -= 1  # Penalty for excessive dilution
                details.append("Concerning dilution: Share count increased significantly")
            else:
                details.append("Moderate share count increase over time")
        else:
            details.append("Insufficient share count data")
        
        # Scale score to 0-10 range
        # Maximum possible raw score would be 12 (3+3+2+2+2), but we'll cap it at 10
        final_score = max(0, min(10, score * 10 / 12))
        
        result["score"] = final_score
        result["details"] = details
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
        analysis['type'] = 'management_quality_analysis'
        analysis['title'] = f'Management quality analysis'

        analysis_data['management_quality_analysis'] = analysis
        ai_message = AIMessage(content=self.get_markdown(analysis))
        return {
            "context": context,
            "messages": [
                ai_message
            ]
        }