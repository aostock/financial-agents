from langgraph.graph import MessagesState
from dataclasses import dataclass
from typing import Optional, TypedDict

@dataclass
class StateAction(TypedDict):
    type: str | int
    parameters: dict

@dataclass
class StateContext(TypedDict):
    analysis_data: dict[str, any]
    metrics: list[dict[str, any]]
    tasks: list[dict[str, any]]
    current_task: dict[str, any]
    task_index: int


@dataclass
class StateTicker(TypedDict):
    symbol: str
    exchange: str
    industry_link: str
    industry_name: str
    quote_type: str
    rank: float
    regular_market_change: float
    regular_market_percent_change: float
    regular_market_price: float
    short_name: str
    time: str

@dataclass
class AgentState(MessagesState):
    locale: str = "en-US"
    tickers: Optional[list[StateTicker]] = None
    agents: Optional[list[str]] = None
    # params from front-end, should be clear when end
    action: Optional[StateAction] = None
    
    suggestions: Optional[list[str]] = None
    # only used in back-end, clear when end
    context: Optional[StateContext] = None
    ui: Optional[list] = None
    settings: Optional[dict[str, any]] = None


