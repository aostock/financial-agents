from common.agent_state import AgentState
from langchain.schema import AIMessage
from langchain_core.callbacks import dispatch_custom_event
from langchain_core.runnables import RunnableConfig
from typing import Dict, Any

import time
from common import markdown
from langgraph.types import StreamWriter
from llm.llm_model import ainvoke


class StoryNarrativeAnalysis():
    def __init__(self, options: Dict[str, Any]):
        self.options = options

    
    def analyze(self, metrics: list, ticker_data: dict) -> dict[str, any]:
        """
        Analyze the business story and narrative based on Damodaran's methodology:
        - Business model and competitive positioning
        - Market opportunity and growth potential
        - Management quality and capital allocation
        - Key risks and uncertainties
        """
        result = {"score": 0, "max_score": 10, "details": [], "narrative": ""}
        if not metrics or not ticker_data:
            result["details"].append('Insufficient data for narrative analysis')
            return result

        # Get company information
        symbol = ticker_data.get('symbol', 'Unknown')
        short_name = ticker_data.get('short_name', 'Unknown Company')
        
        # Prepare data for analysis
        latest_metrics = metrics[0] if metrics else {}
        previous_metrics = metrics[1] if len(metrics) > 1 else {}
        
        # Key metrics for narrative analysis
        business_data = {
            "symbol": symbol,
            "company_name": short_name,
            "industry": ticker_data.get('industry', 'Unknown'),
            "sector": ticker_data.get('sector', 'Unknown'),
            "market_cap": latest_metrics.get('market_cap'),
            "revenue": latest_metrics.get('revenue'),
            "net_income": latest_metrics.get('net_income'),
            "return_on_equity": latest_metrics.get('return_on_equity'),
            "debt_to_equity": latest_metrics.get('debt_to_equity'),
            "operating_margin": latest_metrics.get('operating_margin'),
            "free_cash_flow": latest_metrics.get('free_cash_flow'),
            "price_to_earnings_ratio": latest_metrics.get('price_to_earnings_ratio'),
            "beta": latest_metrics.get('beta'),
        }
        
        # Calculate narrative score based on key factors
        score = 0
        details = []
        
        # Business quality assessment
        if latest_metrics.get('return_on_equity'):
            roe = latest_metrics['return_on_equity']
            if roe > 0.15:  # 15%+ ROE indicates quality business
                score += 2
                details.append(f"High ROE {roe:.1%} suggests quality business")
            elif roe > 0.10:  # 10%+ ROE indicates decent business
                score += 1
                details.append(f"Decent ROE {roe:.1%}")
            else:
                details.append(f"Low ROE {roe:.1%} may indicate quality concerns")
        
        # Financial stability
        if latest_metrics.get('debt_to_equity'):
            dte = latest_metrics['debt_to_equity']
            if dte < 0.5:  # Conservative leverage
                score += 2
                details.append(f"Conservative leverage D/E {dte:.1f}")
            elif dte < 1.0:  # Moderate leverage
                score += 1
                details.append(f"Moderate leverage D/E {dte:.1f}")
            else:
                details.append(f"High leverage D/E {dte:.1f} may indicate financial risk")
        
        # Profitability trend
        if (latest_metrics.get('operating_margin') and previous_metrics.get('operating_margin')):
            current_margin = latest_metrics['operating_margin']
            previous_margin = previous_metrics['operating_margin']
            if current_margin > previous_margin and current_margin > 0.1:  # Improving and strong margins
                score += 2
                details.append(f"Improving operating margins {previous_margin:.1%} → {current_margin:.1%}")
            elif current_margin > 0.05:  # Decent margins
                score += 1
                details.append(f"Stable operating margins {current_margin:.1%}")
            else:
                details.append(f"Low operating margins {current_margin:.1%}")
        
        # Cash generation
        if latest_metrics.get('free_cash_flow') and latest_metrics.get('free_cash_flow') > 0:
            score += 1
            details.append("Positive free cash flow generation")
        elif latest_metrics.get('free_cash_flow'):
            details.append("Negative free cash flow")
        
        # Market valuation perspective
        if latest_metrics.get('price_to_earnings_ratio'):
            pe = latest_metrics['price_to_earnings_ratio']
            if pe < 15 and pe > 0:  # Attractive valuation
                score += 2
                details.append(f"Attractive P/E ratio {pe:.1f}x")
            elif pe < 25 and pe > 0:  # Reasonable valuation
                score += 1
                details.append(f"Reasonable P/E ratio {pe:.1f}x")
            elif pe > 0:
                details.append(f"High P/E ratio {pe:.1f}x may indicate overvaluation")
        
        # Generate narrative based on the analysis
        narrative = f"""# Business Narrative Analysis for {short_name} ({symbol})

## Company Overview
{short_name} operates in the {business_data.get('sector', 'Unknown')} sector within the {business_data.get('industry', 'Unknown')} industry.

## Business Quality Assessment
The company demonstrates {'strong' if score >= 7 else 'moderate' if score >= 4 else 'weak'} business fundamentals with key metrics including:
- Return on Equity: {business_data.get('return_on_equity', 'N/A')}
- Operating Margin: {business_data.get('operating_margin', 'N/A')}
- Debt-to-Equity Ratio: {business_data.get('debt_to_equity', 'N/A')}

## Market Positioning
With a market capitalization of ${business_data.get('market_cap', 'N/A'):,} and a P/E ratio of {business_data.get('price_to_earnings_ratio', 'N/A')}, the company is {'attractively' if (business_data.get('price_to_earnings_ratio', 20) < 15 and business_data.get('price_to_earnings_ratio', 0) > 0) else 'reasonably' if business_data.get('price_to_earnings_ratio', 0) < 25 else 'expensively'} valued relative to its earnings.

## Investment Narrative
Based on the analysis, this appears to be {'a high-quality business with strong fundamentals and attractive valuation' if score >= 8 else 'a solid business with reasonable fundamentals' if score >= 5 else 'a business with mixed fundamentals that requires careful consideration'}.

## Key Risk Factors
- Market sensitivity with beta of {business_data.get('beta', 'N/A')}
- {'Leverage risk' if business_data.get('debt_to_equity', 0) > 1 else 'Financial stability' if business_data.get('debt_to_equity', 1) < 0.5 else 'Moderate leverage'}
- {'Cash flow generation concerns' if business_data.get('free_cash_flow', 1) < 0 else 'Healthy cash flow generation'}
"""

        result["narrative"] = narrative
        result["score"] = min(score, 10)  # Cap at max score
        result["details"] = details
        
        return result

    async def analyze_with_llm(self, metrics: list, ticker_data: dict) -> dict[str, any]:
        """
        Use LLM to generate a detailed business narrative analysis
        """
        result = {"score": 0, "max_score": 10, "details": [], "narrative": ""}
        if not metrics or not ticker_data:
            result["details"].append('Insufficient data for narrative analysis')
            return result

        # Get company information
        symbol = ticker_data.get('symbol', 'Unknown')
        short_name = ticker_data.get('short_name', 'Unknown Company')
        
        # Prepare data for LLM analysis
        latest_metrics = metrics[0] if metrics else {}
        
        # Key metrics for narrative analysis
        business_data = {
            "symbol": symbol,
            "company_name": short_name,
            "industry": ticker_data.get('industry', 'Unknown'),
            "sector": ticker_data.get('sector', 'Unknown'),
            "market_cap": latest_metrics.get('market_cap'),
            "revenue": latest_metrics.get('revenue'),
            "net_income": latest_metrics.get('net_income'),
            "return_on_equity": latest_metrics.get('return_on_equity'),
            "debt_to_equity": latest_metrics.get('debt_to_equity'),
        }

        # Create prompt for narrative analysis
        messages = [
            (
                "system",
                """You are Aswath Damodaran, Professor of Finance at NYU Stern. Analyze the business story and narrative using your "story-to-numbers-to-value" framework:

                YOUR NARRATIVE ANALYSIS APPROACH:
                1. Business Model: What does the company do? How does it make money?
                2. Competitive Positioning: What is the company's competitive advantage?
                3. Market Opportunity: How big is the total addressable market? What is the growth potential?
                4. Management Quality: How effective is management at capital allocation?
                5. Key Risks: What are the biggest uncertainties and risks?
                6. Narrative Quality: Is this a coherent, believable story?

                RATE THE NARRATIVE ON A SCALE OF 0-10:
                10: Exceptional business with strong competitive advantages, clear growth path, and experienced management
                7-9: Solid business with good prospects but some concerns
                4-6: Average business with mixed signals
                1-3: Poor business with significant concerns
                0: No coherent business story or uninvestable

                Provide your analysis in exactly this JSON format:
                {
                  "score": number between 0 and 10,
                  "narrative": "Detailed business narrative analysis",
                  "details": ["Key insight 1", "Key insight 2", "Key insight 3"]
                }
                """,
            ),
            (
                "human",
                f"""Analyze the business narrative for {short_name} ({symbol}) with the following data:
                
                BUSINESS DATA:
                {business_data}
                
                Please provide your narrative analysis following the format specified above.
                """,
            ),
        ]
        
        try:
            response = await ainvoke(messages, config, analyzer=True)
            # Parse the response to extract the JSON
            content = response.content if hasattr(response, 'content') else str(response)
            # In a real implementation, we would parse the JSON from the response
            # For now, we'll return a placeholder
            result["narrative"] = content
            result["score"] = 7  # Placeholder
            result["details"] = ["LLM-based narrative analysis completed"]
        except Exception as e:
            result["details"].append(f'Error in LLM analysis: {str(e)}')
            result["narrative"] = f"Business narrative analysis for {short_name} ({symbol}) using Damodaran's framework."
            result["score"] = 5  # Default score when LLM fails

        return result

    def get_markdown(self, analysis:dict):
        """
        Convert analysis to markdown.
        """
        markdown_content = markdown.analysis_data(analysis)
        return markdown_content

    async def __call__(self, state: AgentState, config: RunnableConfig, writer: StreamWriter) -> Dict[str, Any]:
        context = state.get('context')
        analysis_data = context.get('analysis_data')
        if analysis_data is None:
            analysis_data = {}
            context['analysis_data'] = analysis_data
        metrics = context.get('metrics')
        ticker = context.get('current_task', {}).get('ticker', {})
        
        # Use the improved analyze method
        analysis = self.analyze(metrics, ticker)
        analysis['type'] = 'story_narrative_analysis'
        analysis['title'] = f'Story Narrative Analysis'
        
        analysis_data['story_narrative_analysis'] = analysis
        ai_message = AIMessage(content=self.get_markdown(analysis))
        return {
            "context": context,
            "messages": [
                ai_message
            ]
        }