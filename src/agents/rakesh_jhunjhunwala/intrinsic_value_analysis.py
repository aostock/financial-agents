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

    
    def analyze(self, metrics: list, prices: list) -> dict[str, any]:
        """
        Calculate intrinsic value using Rakesh Jhunjhunwala's approach:
        - Focus on earnings power and growth
        - Conservative discount rates
        - Quality premium for consistent performers
        """
        result = {"score": 0, "max_score": 10, "details": [], "intrinsic_value": None, "margin_of_safety": None}
        if not metrics:
            result["details"].append('No metrics available for intrinsic value calculation')
            return result
        
        try:
            latest_metrics = metrics[0]
            
            # Need positive earnings as base
            if not latest_metrics.get('net_income') or latest_metrics['net_income'] <= 0:
                result["details"].append('No positive earnings for intrinsic value calculation')
                return result
            
            # Get historical earnings for growth calculation
            net_incomes = [m.get('net_income') for m in metrics[:5] 
                           if m.get('net_income') is not None and m.get('net_income') > 0]
            
            if len(net_incomes) < 2:
                # Use current earnings with conservative multiple for stable companies
                intrinsic_value = latest_metrics['net_income'] * 12  # Conservative P/E of 12
                result["intrinsic_value"] = intrinsic_value
                result["details"].append(f"Simple intrinsic value calculation: {intrinsic_value}")
            else:
                # Calculate sustainable growth rate using historical data
                initial_income = net_incomes[-1]  # Oldest
                final_income = net_incomes[0]     # Latest
                years = len(net_incomes) - 1
                
                # Calculate historical CAGR
                if initial_income > 0:
                    historical_growth = ((final_income / initial_income) ** (1/years) - 1)
                else:
                    historical_growth = 0.05  # Default to 5%
                
                # Conservative growth assumptions (Jhunjhunwala style)
                if historical_growth > 0.25:  # Cap at 25% for sustainability
                    sustainable_growth = 0.20  # Conservative 20%
                elif historical_growth > 0.15:
                    sustainable_growth = historical_growth * 0.8  # 80% of historical
                elif historical_growth > 0.05:
                    sustainable_growth = historical_growth * 0.9  # 90% of historical
                else:
                    sustainable_growth = 0.05  # Minimum 5% for inflation
                
                # Quality assessment affects discount rate
                quality_score = self._assess_quality_metrics(metrics)
                
                # Discount rate based on quality (Jhunjhunwala preferred quality)
                if quality_score >= 8:  # High quality
                    discount_rate = 0.12  # 12% for high quality companies
                    terminal_multiple = 18
                elif quality_score >= 6:  # Medium quality
                    discount_rate = 0.15  # 15% for medium quality
                    terminal_multiple = 15
                else:  # Lower quality
                    discount_rate = 0.18  # 18% for riskier companies
                    terminal_multiple = 12
                
                # Simple DCF with terminal value
                current_earnings = latest_metrics['net_income']
                terminal_value = 0
                dcf_value = 0
                
                # Project 5 years of earnings
                for year in range(1, 6):
                    projected_earnings = current_earnings * ((1 + sustainable_growth) ** year)
                    present_value = projected_earnings / ((1 + discount_rate) ** year)
                    dcf_value += present_value
                
                # Terminal value (year 5 earnings * terminal multiple)
                year_5_earnings = current_earnings * ((1 + sustainable_growth) ** 5)
                terminal_value = (year_5_earnings * terminal_multiple) / ((1 + discount_rate) ** 5)
                
                total_intrinsic_value = dcf_value + terminal_value
                result["intrinsic_value"] = total_intrinsic_value
                
                # Calculate margin of safety
                current_price = prices[0].get('close') if prices else 0
                if current_price and total_intrinsic_value:
                    margin_of_safety = (total_intrinsic_value - current_price) / current_price if current_price > 0 else 0
                    result["margin_of_safety"] = margin_of_safety
                    
                    # Score based on margin of safety
                    if margin_of_safety >= 0.50:  # 50%+ margin of safety
                        result["score"] = 10
                        result["details"].append(f"Excellent margin of safety: {margin_of_safety:.1%}")
                    elif margin_of_safety >= 0.30:  # 30%+ margin of safety
                        result["score"] = 8
                        result["details"].append(f"Good margin of safety: {margin_of_safety:.1%}")
                    elif margin_of_safety >= 0.15:  # 15%+ margin of safety
                        result["score"] = 6
                        result["details"].append(f"Moderate margin of safety: {margin_of_safety:.1%}")
                    elif margin_of_safety >= 0:  # Any positive margin
                        result["score"] = 4
                        result["details"].append(f"Small margin of safety: {margin_of_safety:.1%}")
                    else:  # Overvalued
                        result["score"] = 2
                        result["details"].append(f"Overvalued: {abs(margin_of_safety):.1%} overpriced")
                else:
                    result["details"].append(f"Intrinsic value calculated: {total_intrinsic_value}")
        except Exception as e:
            result["details"].append(f"Error in intrinsic value calculation: {str(e)}")
            # Fallback to simple earnings multiple
            if latest_metrics.get('net_income') and latest_metrics['net_income'] > 0:
                result["intrinsic_value"] = latest_metrics['net_income'] * 15
                result["details"].append(f"Fallback intrinsic value: {result['intrinsic_value']}")
        
        return result

    def _assess_quality_metrics(self, metrics: list) -> float:
        """
        Assess company quality based on Jhunjhunwala's criteria.
        Returns a score between 0 and 10.
        """
        if not metrics:
            return 5.0  # Neutral score
        
        latest_metrics = metrics[0]
        quality_factors = []
        
        # ROE consistency and level
        if (latest_metrics.get('net_income') and latest_metrics.get('total_assets') and 
            latest_metrics.get('total_liabilities') and latest_metrics['total_assets'] and latest_metrics['total_liabilities']):
            
            shareholders_equity = latest_metrics['total_assets'] - latest_metrics['total_liabilities']
            if shareholders_equity > 0 and latest_metrics['net_income']:
                roe = latest_metrics['net_income'] / shareholders_equity
                if roe > 0.20:  # ROE > 20%
                    quality_factors.append(10)
                elif roe > 0.15:  # ROE > 15%
                    quality_factors.append(8)
                elif roe > 0.10:  # ROE > 10%
                    quality_factors.append(6)
                else:
                    quality_factors.append(3)
            else:
                quality_factors.append(0)
        else:
            quality_factors.append(5)
        
        # Debt levels (lower is better)
        if (latest_metrics.get('total_assets') and latest_metrics.get('total_liabilities') and 
            latest_metrics['total_assets'] and latest_metrics['total_liabilities']):
            debt_ratio = latest_metrics['total_liabilities'] / latest_metrics['total_assets']
            if debt_ratio < 0.3:  # Low debt
                quality_factors.append(10)
            elif debt_ratio < 0.5:  # Moderate debt
                quality_factors.append(7)
            elif debt_ratio < 0.7:  # High debt
                quality_factors.append(4)
            else:  # Very high debt
                quality_factors.append(1)
        else:
            quality_factors.append(5)
        
        # Growth consistency
        net_incomes = [m.get('net_income') for m in metrics[:4] 
                       if m.get('net_income') is not None and m.get('net_income') > 0]
        
        if len(net_incomes) >= 3:
            declining_years = sum(1 for i in range(1, len(net_incomes)) if net_incomes[i-1] > net_incomes[i])
            consistency = 1 - (declining_years / (len(net_incomes) - 1))
            quality_score = consistency * 10
            quality_factors.append(quality_score)
        
        # Return average quality score
        return sum(quality_factors) / len(quality_factors) if quality_factors else 5.0

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
        analysis = self.analyze(metrics, prices)
        analysis['type'] = 'intrinsic_value_analysis'
        analysis['title'] = f'Intrinsic value analysis'

        analysis_data['intrinsic_value_analysis'] = analysis
        ai_message = AIMessage(content=self.get_markdown(analysis))
        return {
            "context": context,
            "messages": [
                ai_message
            ]
        }