from common.agent_state import AgentState
from langchain.schema import AIMessage
from langchain_core.callbacks import dispatch_custom_event
from langchain_core.runnables import RunnableConfig
from typing import Dict, Any

import time
from common import markdown
from langgraph.types import StreamWriter


class FinancialStatementAnalysis():
    def __init__(self, options: Dict[str, Any]):
        self.options = options

    
    def analyze(self, metrics: list) -> dict[str, any]:
        """Analyze financial statements for quality, red flags, and accounting practices."""
        result = {"score": 0, "max_score": 10, "details": []}
        if not metrics or len(metrics) < 3:
            result["details"].append('Insufficient historical data (need at least 3 years)')
            return result

        latest_metrics = metrics[0]
        previous_metrics = metrics[1] if len(metrics) > 1 else None

        score = 0
        reasoning = []

        # Check for accounting quality - look for consistency and transparency
        # Revenue recognition quality
        if latest_metrics.get('revenue') and latest_metrics.get('accounts_receivable'):
            receivables_ratio = latest_metrics['accounts_receivable'] / latest_metrics['revenue'] if latest_metrics['revenue'] > 0 else 0
            if receivables_ratio < 0.2:  # Receivables less than 20% of revenue suggests good collection
                score += 2
                reasoning.append(f"Healthy receivables ratio ({receivables_ratio:.1%} of revenue)")
            elif receivables_ratio < 0.3:
                score += 1
                reasoning.append(f"Acceptable receivables ratio ({receivables_ratio:.1%} of revenue)")
            else:
                reasoning.append(f"High receivables ratio ({receivables_ratio:.1%} of revenue) - potential revenue recognition issues")

        # Inventory quality
        if latest_metrics.get('inventory') and latest_metrics.get('revenue'):
            inventory_ratio = latest_metrics['inventory'] / latest_metrics['revenue'] if latest_metrics['revenue'] > 0 else 0
            if inventory_ratio < 0.3:  # Inventory less than 30% of revenue suggests efficient management
                score += 2
                reasoning.append(f"Efficient inventory management ({inventory_ratio:.1%} of revenue)")
            elif inventory_ratio < 0.5:
                score += 1
                reasoning.append(f"Reasonable inventory levels ({inventory_ratio:.1%} of revenue)")
            else:
                reasoning.append(f"High inventory levels ({inventory_ratio:.1%} of revenue) - potential obsolescence risk")

        # Debt structure analysis
        if latest_metrics.get('total_liabilities') and latest_metrics.get('total_assets'):
            debt_ratio = latest_metrics['total_liabilities'] / latest_metrics['total_assets'] if latest_metrics['total_assets'] > 0 else 0
            if debt_ratio < 0.4:  # Conservative leverage
                score += 2
                reasoning.append(f"Conservative leverage ({debt_ratio:.1%} of assets)")
            elif debt_ratio < 0.6:
                score += 1
                reasoning.append(f"Moderate leverage ({debt_ratio:.1%} of assets)")
            else:
                reasoning.append(f"High leverage ({debt_ratio:.1%} of assets) - financial risk")

        # Current ratio analysis
        if latest_metrics.get('current_ratio'):
            if latest_metrics['current_ratio'] > 2.0:
                score += 2
                reasoning.append(f"Strong liquidity position (Current Ratio: {latest_metrics['current_ratio']:.2f})")
            elif latest_metrics['current_ratio'] > 1.5:
                score += 1
                reasoning.append(f"Adequate liquidity (Current Ratio: {latest_metrics['current_ratio']:.2f})")
            else:
                reasoning.append(f"Weak liquidity (Current Ratio: {latest_metrics['current_ratio']:.2f})")

        # Cash flow quality
        if latest_metrics.get('free_cash_flow') and latest_metrics.get('net_income'):
            fcf_ratio = latest_metrics['free_cash_flow'] / latest_metrics['net_income'] if latest_metrics['net_income'] != 0 else 0
            if fcf_ratio > 0.8:  # High FCF conversion
                score += 2
                reasoning.append(f"High quality cash flow generation ({fcf_ratio:.1%} of net income)")
            elif fcf_ratio > 0.5:
                score += 1
                reasoning.append(f"Good cash flow generation ({fcf_ratio:.1%} of net income)")
            else:
                reasoning.append(f"Poor cash flow conversion ({fcf_ratio:.1%} of net income)")

        # Check for earnings quality - consistency over time
        net_incomes = [m.get('net_income') for m in metrics[:3] if m.get('net_income') is not None]
        if len(net_incomes) >= 3:
            avg_income = sum(net_incomes) / len(net_incomes)
            volatility = sum(abs(ni - avg_income) for ni in net_incomes) / len(net_incomes) if avg_income != 0 else 0
            normalized_volatility = volatility / abs(avg_income) if avg_income != 0 else 0
            
            if normalized_volatility < 0.2:  # Low earnings volatility
                score += 1
                reasoning.append(f"Stable earnings pattern (volatility: {normalized_volatility:.1%})")
            elif normalized_volatility < 0.4:
                reasoning.append(f"Moderate earnings volatility (volatility: {normalized_volatility:.1%})")
            else:
                reasoning.append(f"High earnings volatility (volatility: {normalized_volatility:.1%})")

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
        analysis['type'] = 'financial_statement_analysis'
        analysis['title'] = f'Financial Statement Analysis'

        analysis_data['financial_statement_analysis'] = analysis
        ai_message = AIMessage(content=self.get_markdown(analysis))
        return {
            "context": context,
            "messages": [
                ai_message
            ]
        }