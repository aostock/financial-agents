from common.agent_state import AgentState
from langchain.schema import AIMessage
from langchain_core.callbacks import dispatch_custom_event
from langchain_core.runnables import RunnableConfig
from typing import Dict, Any

import time
from common import markdown
from langgraph.types import StreamWriter



class IntrinsicValueAnalysis():
    def __init__(self, options: Dict[str, Any]):
        self.options = options

    def estimate_maintenance_capex(self, financial_line_items: list) -> float:
        """
        Estimate maintenance capital expenditure using multiple approaches.
        Buffett considers this crucial for understanding true owner earnings.
        """
        if not financial_line_items:
            return 0
        
        # Approach 1: Historical average as % of revenue
        capex_ratios = []
        depreciation_values = []
        
        for item in financial_line_items[:5]:  # Last 5 periods
            if item.get('capital_expenditure') and item.get('revenue'):
                if item.get('capital_expenditure') and item.get('revenue') and item.get('revenue') > 0:
                    capex_ratio = abs(item.get('capital_expenditure')) / item.get('revenue')
                    capex_ratios.append(capex_ratio)
            
            if item.get('depreciation_and_amortization') and item.get('depreciation_and_amortization'):
                depreciation_values.append(item.get('depreciation_and_amortization'))
        
        # Approach 2: Percentage of depreciation (typically 80-120% for maintenance)
        latest_depreciation = financial_line_items[0].get('depreciation_and_amortization') if financial_line_items[0].get('depreciation_and_amortization') else 0
        
        # Approach 3: Industry-specific heuristics
        latest_capex = abs(financial_line_items[0].get('capital_expenditure')) if financial_line_items[0].get('capital_expenditure') else 0
        
        # Conservative estimate: Use the higher of:
        # 1. 85% of total capex (assuming 15% is growth capex)
        # 2. 100% of depreciation (replacement of worn-out assets)
        # 3. Historical average if stable
        
        method_1 = latest_capex * 0.85  # 85% of total capex
        method_2 = latest_depreciation  # 100% of depreciation
        
        # If we have historical data, use average capex ratio
        if len(capex_ratios) >= 3:
            avg_capex_ratio = sum(capex_ratios) / len(capex_ratios)
            latest_revenue = financial_line_items[0].get('revenue') if financial_line_items[0].get('revenue') else 0
            method_3 = avg_capex_ratio * latest_revenue if latest_revenue else 0
            
            # Use the median of the three approaches for conservatism
            estimates = sorted([method_1, method_2, method_3])
            return estimates[1]  # Median
        else:
            # Use the higher of method 1 and 2
            return max(method_1, method_2)


    def calculate_owner_earnings(self, financial_line_items: list) -> dict[str, any]:
        """
        Calculate owner earnings (Buffett's preferred measure of true earnings power).
        Enhanced methodology: Net Income + Depreciation/Amortization - Maintenance CapEx - Working Capital Changes
        Uses multi-period analysis for better maintenance capex estimation.
        """
        if not financial_line_items or len(financial_line_items) < 2:
            return {"owner_earnings": None, "details": ["Insufficient data for owner earnings calculation"]}

        latest = financial_line_items[0]
        details = []

        # Core components
        net_income = latest.get('net_income')
        depreciation = latest.get('depreciation_and_amortization')
        capex = latest.get('capital_expenditure')

        if not all([net_income is not None, depreciation is not None, capex is not None]):
            missing = []
            if net_income is None: missing.append("net income")
            if depreciation is None: missing.append("depreciation")
            if capex is None: missing.append("capital expenditure")
            return {"owner_earnings": None, "details": [f"Missing components: {', '.join(missing)}"]}

        # Enhanced maintenance capex estimation using historical analysis
        maintenance_capex = self.estimate_maintenance_capex(financial_line_items)
        
        # Working capital change analysis (if data available)
        working_capital_change = 0
        if len(financial_line_items) >= 2:
            try:
                current_assets_current = latest.get('current_assets')
                current_liab_current = latest.get('current_liabilities')
                
                previous = financial_line_items[1]
                current_assets_previous = previous.get('current_assets')
                current_liab_previous = previous.get('current_liabilities')
                
                if all([current_assets_current, current_liab_current, current_assets_previous, current_liab_previous]):
                    wc_current = current_assets_current - current_liab_current
                    wc_previous = current_assets_previous - current_liab_previous
                    working_capital_change = wc_current - wc_previous
                    details.append(f"Working capital change: ${working_capital_change:,.0f}")
            except:
                pass  # Skip working capital adjustment if data unavailable

        # Calculate owner earnings
        owner_earnings = net_income + depreciation - maintenance_capex - working_capital_change

        # Sanity checks
        if owner_earnings < net_income * 0.3:  # Owner earnings shouldn't be less than 30% of net income typically
            details.append("Warning: Owner earnings significantly below net income - high capex intensity")
        
        if maintenance_capex > depreciation * 2:  # Maintenance capex shouldn't typically exceed 2x depreciation
            details.append("Warning: Estimated maintenance capex seems high relative to depreciation")

        details.extend([
            f"Net income: ${net_income:,.0f}",
            f"Depreciation: ${depreciation:,.0f}",
            f"Estimated maintenance capex: ${maintenance_capex:,.0f}",
            f"Owner earnings: ${owner_earnings:,.0f}"
        ])

        return {
            "owner_earnings": owner_earnings,
            "components": {
                "net_income": net_income,
                "depreciation": depreciation,
                "maintenance_capex": maintenance_capex,
                "working_capital_change": working_capital_change,
                "total_capex": abs(capex) if capex else 0
            },
            "details": details,
        }

    def analyze(self, financial_line_items: list) -> dict[str, any]:
        """
        Calculate intrinsic value using enhanced DCF with owner earnings.
        Uses more sophisticated assumptions and conservative approach like Buffett.
        """
        result = {"intrinsic_value": None, "score": 0, "max_score": 6, "details": []}
        if not financial_line_items or len(financial_line_items) < 3:
            result["details"].append("Insufficient data for reliable valuation")
            return result

        # Calculate owner earnings with better methodology
        earnings_data = self.calculate_owner_earnings(financial_line_items)
        if not earnings_data["owner_earnings"]:
            result["details"].append(earnings_data["details"])
            return result

        owner_earnings = earnings_data["owner_earnings"]
        latest_financial_line_items = financial_line_items[0]
        shares_outstanding = latest_financial_line_items.get('ordinary_shares_number')

        if not shares_outstanding or shares_outstanding <= 0:
            result["details"].append("Missing or invalid shares outstanding data")
            return result

        # Enhanced DCF with more realistic assumptions
        details = []
        
        # Estimate growth rate based on historical performance (more conservative)
        historical_earnings = []
        for item in financial_line_items[:5]:  # Last 5 years
            if hasattr(item, 'net_income') and item.net_income:
                historical_earnings.append(item.net_income)
        
        # Calculate historical growth rate
        if len(historical_earnings) >= 3:
            oldest_earnings = historical_earnings[-1]
            latest_earnings = historical_earnings[0]
            years = len(historical_earnings) - 1
            
            if oldest_earnings > 0:
                historical_growth = ((latest_earnings / oldest_earnings) ** (1/years)) - 1
                # Conservative adjustment - cap growth and apply haircut
                historical_growth = max(-0.05, min(historical_growth, 0.15))  # Cap between -5% and 15%
                conservative_growth = historical_growth * 0.7  # Apply 30% haircut for conservatism
            else:
                conservative_growth = 0.03  # Default 3% if negative base
        else:
            conservative_growth = 0.03  # Default conservative growth
        
        # Buffett's conservative assumptions
        stage1_growth = min(conservative_growth, 0.08)  # Stage 1: cap at 8%
        stage2_growth = min(conservative_growth * 0.5, 0.04)  # Stage 2: half of stage 1, cap at 4%
        terminal_growth = 0.025  # Long-term GDP growth rate
        
        # Risk-adjusted discount rate based on business quality
        base_discount_rate = 0.09  # Base 9%
        
        # Adjust based on analysis scores (if available in calling context)
        # For now, use conservative 10%
        discount_rate = 0.10
        
        # Three-stage DCF model
        stage1_years = 5   # High growth phase
        stage2_years = 5   # Transition phase
        
        present_value = 0
        details.append(f"Using three-stage DCF: Stage 1 ({stage1_growth:.1%}, {stage1_years}y), Stage 2 ({stage2_growth:.1%}, {stage2_years}y), Terminal ({terminal_growth:.1%})")
        
        # Stage 1: Higher growth
        stage1_pv = 0
        for year in range(1, stage1_years + 1):
            future_earnings = owner_earnings * (1 + stage1_growth) ** year
            pv = future_earnings / (1 + discount_rate) ** year
            stage1_pv += pv
        
        # Stage 2: Transition growth
        stage2_pv = 0
        stage1_final_earnings = owner_earnings * (1 + stage1_growth) ** stage1_years
        for year in range(1, stage2_years + 1):
            future_earnings = stage1_final_earnings * (1 + stage2_growth) ** year
            pv = future_earnings / (1 + discount_rate) ** (stage1_years + year)
            stage2_pv += pv
        
        # Terminal value using Gordon Growth Model
        final_earnings = stage1_final_earnings * (1 + stage2_growth) ** stage2_years
        terminal_earnings = final_earnings * (1 + terminal_growth)
        terminal_value = terminal_earnings / (discount_rate - terminal_growth)
        terminal_pv = terminal_value / (1 + discount_rate) ** (stage1_years + stage2_years)
        
        # Total intrinsic value
        intrinsic_value = stage1_pv + stage2_pv + terminal_pv
        
        # Apply additional margin of safety (Buffett's conservatism)
        conservative_intrinsic_value = intrinsic_value * 0.85  # 15% additional haircut
        
        details.extend([
            f"Stage 1 PV: ${stage1_pv:,.0f}",
            f"Stage 2 PV: ${stage2_pv:,.0f}",
            f"Terminal PV: ${terminal_pv:,.0f}",
            f"Total IV: ${intrinsic_value:,.0f}",
            f"Conservative IV (15% haircut): ${conservative_intrinsic_value:,.0f}",
            f"Owner earnings: ${owner_earnings:,.0f}",
            f"Discount rate: {discount_rate:.1%}"
        ])

        score = 0
        if conservative_intrinsic_value > 0:
            score = 6
        elif intrinsic_value > 0:
            score = 3

        return {
            "intrinsic_value": conservative_intrinsic_value,
            "raw_intrinsic_value": intrinsic_value,
            "owner_earnings": owner_earnings,
            "score": score, "max_score": result["max_score"],
            "assumptions": {
                "stage1_growth": stage1_growth,
                "stage2_growth": stage2_growth,
                "terminal_growth": terminal_growth,
                "discount_rate": discount_rate,
                "stage1_years": stage1_years,
                "stage2_years": stage2_years,
                "historical_growth": conservative_growth if 'conservative_growth' in locals() else None,
            },
            "details": details,
        }

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
        analysis['type'] = 'intrinsic_value_analysis'
        analysis['title'] = 'Intrinsic value analysis'

        analysis_data['intrinsic_value_analysis'] = analysis
        ai_message = AIMessage(content=self.get_markdown(analysis))
        return {
            "context": context,
            "messages": [
                ai_message
            ]
        }