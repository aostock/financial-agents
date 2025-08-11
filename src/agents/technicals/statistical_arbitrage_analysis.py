from common.agent_state import AgentState
from langchain.schema import AIMessage
from langchain_core.callbacks import dispatch_custom_event
from langchain_core.runnables import RunnableConfig
from typing import Dict, Any
import time
from common import markdown
from langgraph.types import StreamWriter
import math


class StatisticalArbitrageAnalysis():
    def __init__(self, options: Dict[str, Any]):
        self.options = options

    
    def calculate_returns(self, prices: list) -> list:
        """Calculate daily percentage returns"""
        if len(prices) < 2:
            return []
        
        returns = []
        for i in range(1, len(prices)):
            prev_close = prices[i-1].get('close', 0)
            curr_close = prices[i].get('close', 0)
            if prev_close > 0:
                ret = (curr_close - prev_close) / prev_close * 100
                returns.append(ret)
            else:
                returns.append(0)
        return returns
    
    def calculate_skewness(self, data: list) -> float:
        """Calculate skewness of a dataset"""
        if len(data) < 3:
            return 0
            
        n = len(data)
        mean = sum(data) / n
        std = math.sqrt(sum((x - mean) ** 2 for x in data) / (n - 1)) if n > 1 else 0
        
        if std == 0:
            return 0
            
        # Calculate skewness
        skew = sum(((x - mean) / std) ** 3 for x in data) * n / ((n - 1) * (n - 2))
        return skew
    
    def calculate_kurtosis(self, data: list) -> float:
        """Calculate kurtosis of a dataset"""
        if len(data) < 4:
            return 3  # Normal distribution kurtosis
            
        n = len(data)
        mean = sum(data) / n
        std = math.sqrt(sum((x - mean) ** 2 for x in data) / (n - 1)) if n > 1 else 0
        
        if std == 0:
            return 3
            
        # Calculate kurtosis
        kurt = sum(((x - mean) / std) ** 4 for x in data) * n * (n + 1) / ((n - 1) * (n - 2) * (n - 3)) - 3 * (n - 1) ** 2 / ((n - 2) * (n - 3))
        return kurt
    
    def calculate_hurst_exponent(self, prices: list, max_lag: int = 20) -> float:
        """Calculate Hurst Exponent to determine long-term memory of time series
        H < 0.5: Mean reverting series
        H = 0.5: Random walk
        H > 0.5: Trending series
        """
        if len(prices) < max_lag * 2:
            return 0.5  # Default to random walk
        
        closes = [price.get('close', 0) for price in prices if price.get('close')]
        if len(closes) < max_lag * 2:
            return 0.5  # Default to random walk
        
        # Simplified Hurst exponent calculation
        lags = range(2, min(max_lag, len(closes) // 4))
        if len(lags) < 2:
            return 0.5
            
        # Calculate tau values
        tau = []
        for lag in lags:
            if lag < len(closes):
                diffs = [abs(closes[i] - closes[i-lag]) for i in range(lag, len(closes))]
                if diffs:
                    std_dev = math.sqrt(sum((x - sum(diffs)/len(diffs))**2 for x in diffs) / (len(diffs) - 1)) if len(diffs) > 1 else 0
                    if std_dev > 0:
                        tau.append(max(1e-8, std_dev))
                    else:
                        tau.append(1e-8)
                else:
                    tau.append(1e-8)
            else:
                tau.append(1e-8)
        
        # Estimate Hurst exponent from log-log regression
        if len(lags) >= 2 and len(tau) >= 2:
            log_lags = [math.log(lag) for lag in lags if lag > 0]
            log_tau = [math.log(t) for t in tau if t > 0]
            
            if len(log_lags) >= 2 and len(log_tau) >= 2:
                # Simple linear regression to estimate slope (Hurst exponent)
                n = min(len(log_lags), len(log_tau))
                if n >= 2:
                    sum_x = sum(log_lags[:n])
                    sum_y = sum(log_tau[:n])
                    sum_xy = sum(log_lags[i] * log_tau[i] for i in range(n))
                    sum_xx = sum(log_lags[i] ** 2 for i in range(n))
                    
                    denominator = n * sum_xx - sum_x ** 2
                    if abs(denominator) > 1e-10:  # Avoid division by zero
                        hurst = (n * sum_xy - sum_x * sum_y) / denominator
                        return max(0, min(1, hurst))  # Clamp between 0 and 1
        
        return 0.5  # Default to random walk
    
    def analyze(self, prices: list) -> dict[str, any]:
        """Analyze statistical arbitrage signals based on price action analysis."""
        result = {"score": 0, "max_score": 10, "details": [], "indicators": {}}
        if not prices or len(prices) < 63:  # Need at least 63 days for statistical analysis
            result["details"].append('Insufficient price data for statistical arbitrage analysis (need at least 63 days)')
            return result

        # Calculate returns
        returns = self.calculate_returns(prices)
        if len(returns) < 63:
            result["details"].append('Insufficient return data for statistical arbitrage analysis')
            return result
        
        # Calculate price distribution statistics
        # Skewness and kurtosis (63-day window)
        skew = self.calculate_skewness(returns[-63:]) if len(returns) >= 63 else 0
        kurt = self.calculate_kurtosis(returns[-63:]) if len(returns) >= 63 else 3
        
        # Test for mean reversion using Hurst exponent
        hurst = self.calculate_hurst_exponent(prices)
        
        # Store indicators
        result["indicators"] = {
            "hurst_exponent": hurst,
            "skewness": skew,
            "kurtosis": kurt
        }
        
        score = 5  # Start with neutral score
        reasoning = []
        
        # Generate signal based on statistical properties
        if hurst < 0.4 and skew > 1:
            score += 2  # Strong bullish mean reversion signal
            reasoning.append(f"Strong mean reversion signal (Hurst: {hurst:.3f}, Skewness: {skew:.2f})")
        elif hurst < 0.4 and skew < -1:
            score -= 2  # Strong bearish mean reversion signal
            reasoning.append(f"Strong mean reversion signal (Hurst: {hurst:.3f}, Skewness: {skew:.2f})")
        elif hurst < 0.45:
            score += 1  # Moderate mean reversion tendency
            reasoning.append(f"Mean reversion tendency (Hurst: {hurst:.3f})")
        elif hurst > 0.55:
            score -= 1  # Trending tendency
            reasoning.append(f"Trending tendency (Hurst: {hurst:.3f})")
        else:
            reasoning.append(f"Random walk characteristics (Hurst: {hurst:.3f})")
        
        # Skewness analysis
        if skew > 2:
            score += 1  # Positive skew - more positive outliers
            reasoning.append(f"Positive return skewness ({skew:.2f}) - favorable for long positions")
        elif skew < -2:
            score -= 1  # Negative skew - more negative outliers
            reasoning.append(f"Negative return skewness ({skew:.2f}) - unfavorable for long positions")
        
        # Kurtosis analysis (fat tails)
        if kurt > 6:
            score -= 0.5  # High kurtosis - fat tails, higher risk
            reasoning.append(f"High kurtosis ({kurt:.2f}) - increased tail risk")
        elif kurt < 2:
            score += 0.5  # Low kurtosis - thin tails, more predictable
            reasoning.append(f"Low kurtosis ({kurt:.2f}) - more predictable returns")
        
        # Combined statistical signal
        if hurst < 0.3 and abs(skew) > 2:
            if skew > 0:
                score += 0.5  # Very strong bullish statistical setup
                reasoning.append(f"Very strong statistical arbitrage setup - mean reversion with positive skew")
            else:
                score -= 0.5  # Very strong bearish statistical setup
                reasoning.append(f"Very strong statistical arbitrage setup - mean reversion with negative skew")
        
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
        prices = context.get('prices')
        analysis = self.analyze(prices)
        analysis['type'] = 'statistical_arbitrage_analysis'
        analysis['title'] = f'Statistical Arbitrage Analysis'

        analysis_data['statistical_arbitrage_analysis'] = analysis
        ai_message = AIMessage(content=self.get_markdown(analysis))
        return {
            "context": context,
            "messages": [
                ai_message
            ]
        }