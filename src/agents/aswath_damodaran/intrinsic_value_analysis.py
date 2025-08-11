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

    
    def analyze(self, metrics: list, risk_analysis: dict) -> dict[str, any]:
        """
        FCFF DCF with:
          • Base FCFF = latest free cash flow
          • Growth = 5-yr revenue CAGR (capped 12 %)
          • Fade linearly to terminal growth 2.5 % by year 10
          • Discount @ cost of equity (no debt split given data limitations)
        """
        result = {"intrinsic_value": None, "details": []}
        if not metrics or len(metrics) < 2:
            result["details"].append("Insufficient data")
            return result

        latest_m = metrics[0]
        fcff0 = latest_m.get('free_cash_flow')
        shares = latest_m.get('ordinary_shares_number')
        if not fcff0 or not shares:
            result["details"].append("Missing FCFF or share count")
            return result

        # Growth assumptions
        revs = [m.get('revenue') for m in reversed(metrics) if m.get('revenue')]
        if len(revs) >= 2 and revs[0] and revs[0] > 0:
            base_growth = min((revs[-1] / revs[0]) ** (1 / (len(revs) - 1)) - 1, 0.12)
        else:
            base_growth = 0.04  # fallback

        terminal_growth = 0.025
        years = 10

        # Discount rate
        discount = risk_analysis.get("cost_of_equity") or 0.09

        # Project FCFF and discount
        pv_sum = 0.0
        g = base_growth
        g_step = (terminal_growth - base_growth) / (years - 1)
        for yr in range(1, years + 1):
            fcff_t = fcff0 * (1 + g)
            pv = fcff_t / (1 + discount) ** yr
            pv_sum += pv
            g += g_step

        # Terminal value (perpetuity with terminal growth)
        tv = (
            fcff0
            * (1 + terminal_growth)
            / (discount - terminal_growth)
            / (1 + discount) ** years
        )

        equity_value = pv_sum + tv
        intrinsic_per_share = equity_value / shares

        result["intrinsic_value"] = equity_value
        result["intrinsic_per_share"] = intrinsic_per_share
        result["assumptions"] = {
            "base_fcff": fcff0,
            "base_growth": base_growth,
            "terminal_growth": terminal_growth,
            "discount_rate": discount,
            "projection_years": years,
        }
        result["details"] = ["FCFF DCF completed"]
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
        risk_analysis = analysis_data.get('risk_analysis', {})
        analysis = self.analyze(metrics, risk_analysis)
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