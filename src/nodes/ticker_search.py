from typing import Any, Dict, Generic, TypeVar
from langchain.schema import AIMessage
from langchain_core.runnables import RunnableConfig
from langgraph.types import StreamWriter

from common.agent_state import AgentState

import common.markdown as markdown
from llm.llm_model import ainvoke
from common.dataset import Dataset


T = TypeVar('T')
class TickerSearch(Generic[T]):
    """
    This class is used to handle stock ticker search logic. It utilizes an intent recognition assistant 
    to determine if the user input contains a stock name or symbol. Based on the result, it looks up 
    the corresponding stock information and finally returns the search results or a prompt indicating 
    that no ticker was found.
    """
    
    def __init__(self, options: Dict[str, Any]):
        self.options = options

    async def __call__(self, state: AgentState, writer: StreamWriter, config: RunnableConfig) -> T:
        tickers = state.get('action').get('parameters').get('tickers')
        json_markdown = ''
        for ticker in tickers:
            query = ticker.get('symbol', '')
            if query == '':
                query = ticker.get('en_name', '')
            if query == '':
                query = ticker.get('short_name', '')
            lookup_result = []
            if query != '':
                dataset = Dataset(config)
                lookup_result = dataset.lookup_ticker(query)
            if len(lookup_result) == 0:
                json_markdown += f'* {query} not found\n'
            else:
                json_markdown += markdown.ticker_select({'list': lookup_result, 'selected': lookup_result[0]}) + '\n'
        
        if len(tickers) == 0:
            return {
                    "messages": [AIMessage(content="can not find ticker, please check your input and try again")],
                    "action": None
                }
        return {
                "messages": [AIMessage(content=json_markdown)],
                "action": None
            }
        