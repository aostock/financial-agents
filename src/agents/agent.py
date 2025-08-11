"""
This is the main entry point for the agent.
It defines the workflow graph, state, tools, nodes and edges.
"""

from pkgutil import resolve_name
from common import markdown
from langchain_core.runnables import RunnableConfig
from typing_extensions import Literal
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, AIMessage
from langchain.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.types import Command
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from common.agent_state import AgentState, StateContext
from langgraph.types import StreamWriter
import asyncio
import uuid
from common.util import get_dict_json, get_at_items, get_latest_message_content, get_array_json
from agents.warren_buffett.agent import agent as warren_buffett_agent
from agents.aswath_damodaran.agent import agent as aswath_damodaran_agent
from agents.ben_graham.agent import agent as ben_graham_agent
from agents.bill_ackman.agent import agent as bill_ackman_agent
from agents.cathie_wood.agent import agent as cathie_wood_agent
from agents.charlie_munger.agent import agent as charlie_munger_agent
from agents.fundamentals.agent import agent as fundamentals_agent
from agents.michael_burry.agent import agent as michael_burry_agent
from agents.peter_lynch.agent import agent as peter_lynch_agent
from agents.phil_fisher.agent import agent as phil_fisher_agent
from agents.portfolio_manager.agent import agent as portfolio_manager_agent
from agents.rakesh_jhunjhunwala.agent import agent as rakesh_jhunjhunwala_agent
from agents.risk_manager.agent import agent as risk_manager_agent
from agents.sentiment.agent import agent as sentiment_agent
from agents.stanley_druckenmiller.agent import agent as stanley_druckenmiller_agent
from agents.technicals.agent import agent as technicals_agent
from agents.trading.agent import agent as trading_agent
from agents.valuation.agent import agent as valuation_agent
from agents.information_query.agent import agent as information_query_agent
from llm.llm_model import ainvoke
from nodes.ticker_search import TickerSearch
from nodes.next_step_suggestions import NextStepSuggestions



ticker_search = TickerSearch[Command[Literal['clear_cache']]]({})
next_step_suggestions = NextStepSuggestions({})

analysis_agents = {
    'information_query': information_query_agent,
    'warren_buffett': warren_buffett_agent,
    'aswath_damodaran': aswath_damodaran_agent,
    'ben_graham': ben_graham_agent,
    'bill_ackman': bill_ackman_agent,
    'cathie_wood': cathie_wood_agent,
    'charlie_munger': charlie_munger_agent,
    'fundamentals': fundamentals_agent,
    'michael_burry': michael_burry_agent,
    'peter_lynch': peter_lynch_agent,
    'phil_fisher': phil_fisher_agent,
    'portfolio_manager': portfolio_manager_agent,
    'rakesh_jhunjhunwala': rakesh_jhunjhunwala_agent,
    'risk_manager': risk_manager_agent,
    'sentiment': sentiment_agent,
    'stanley_druckenmiller': stanley_druckenmiller_agent,
    'technicals': technicals_agent,
    'trading': trading_agent,
    'valuation': valuation_agent
}

async def planner_node(state: AgentState, config: RunnableConfig) -> Command[Literal["ticker_switch", "ticker_analysis", "ticker_search", "next_step_suggestions"]]:
    """
    Initialize the planner node by setting up the context in the agent state.
    
    Args:
        state (AgentState): The current state of the agent containing context and messages
        config (RunnableConfig): Configuration for the runnable
        
    Returns:
        dict: Updated context dictionary
    """
    context = state.get('context')
    if context is None:
        context = {}
    
    if state.get('action') is not None:
        if state.get('action')['type'] == "ticker_switch":
        # The switch has been completed on the front-end, no back-end processing is required
            return Command(goto='ticker_switch', update={'context': context})
        elif state.get('action')['type'] == "ticker_analysis":
            return Command(goto='ticker_analysis', update={'context': context})
        else:
            return Command(goto='next_step_suggestions', update={'context': context})
    else:
        tickers = await get_tickers_from_content(state, config)
        if tickers is None or len(tickers) == 0:
            return Command(goto='next_step_suggestions', update={'context': context})
        content = get_latest_message_content(state)
        items = get_at_items(content)
        agents = []
        for item in items:
            if item in analysis_agents:
                agents.append(item)
        
        action = {'type': 'ticker_search', 'parameters': {'agents': agents, 'tickers': tickers}}
        if len(agents) > 0:
            action['type'] = 'ticker_analysis'
        return Command(goto=action['type'], update={'context': context, 'action': action})


def ticker_analysis(state: AgentState, config: RunnableConfig):
    """
    Prepare tasks for ticker analysis by creating analysis tasks for each agent and ticker combination.
    This function manages a loop through multiple analysis tasks.
    
    Args:
        state (AgentState): The current state of the agent containing context, messages, and action
        config (RunnableConfig): Configuration for the runnable
        
    Returns:
        dict: Updated context with tasks and current task information
    """
    # is a loop, so we need to get the task_index
    context = state.get('context')
    if context.get('tasks') is None:
        # in last messages content, there some @agent_name content, get the @ list
        agents = state.get('action').get('parameters').get('agents')
        tickers = state.get('action').get('parameters').get('tickers')
        tasks = []
        for agent_name in agents:
            for ticker in tickers:
                tasks.append({
                    'agent': agent_name,
                    'ticker': ticker
                })
        context['tasks'] = tasks
        context['task_index'] = 0
    else:
        context['task_index'] += 1
    if context['task_index'] < len(context['tasks']):
        context['current_task'] = context['tasks'][context['task_index']]
    else:
         context['current_task'] = None
    return {'context': context}

def agent_conditional(state: AgentState, config: RunnableConfig):
    """
    Determine which analysis agent to route to based on the current task.
    
    Args:
        state (AgentState): The current state of the agent containing context with current task information
        config (RunnableConfig): Configuration for the runnable
        
    Returns:
        str: The name of the analysis agent to route to, or 'clear_cache' if all tasks are completed
    """
    context = state.get('context')
    if context.get('current_task') is not None and context.get('task_index') < len(context.get('tasks')):
        return context.get('current_task')['agent']
    return 'clear_cache'

def ticker_switch(state: AgentState, config: RunnableConfig) -> Command[Literal['clear_cache']]:
    """
    Handle ticker switching by generating a ticker selection message and clearing the action.
    
    Args:
        state (AgentState): The current state of the agent containing action parameters
        config (RunnableConfig): Configuration for the runnable
        
    Returns:
        Command: A command to goto 'clear_cache' with updated action and messages
    """
    message = AIMessage(content=markdown.ticker_select(state.get('action').get('parameters')))
    return Command(goto='clear_cache', update={'action': None, 'messages': [message]})


async def get_tickers_from_content(state: AgentState, config: RunnableConfig):
    """
    Extract tickers from the content.
    
    Args:
        content (str): The content to extract tickers from
        
    Returns:
        list: A list of tickers
    """
    prompt = """Please extract the most recent one or more stock information entries currently being discussed from the above conversation history. The extracted information must include:  

1. **short_name**: The name of the stock or company  
2. **symbol**: The stock symbol, derived from user input or analyzed from short_name. Include the stock market suffix when necessary (e.g., `AAPL`, `601398.SS`)  
3. **en_name**: The English abbreviation of the stock, analyzed from short_name and symbol  

**Output ONLY the JSON—validate structure before responding.**  
Example output:  
`[{"short_name": "Apple", "en_name": "Apple Inc.", "symbol": "AAPL"}]`
"""
    messages = state["messages"] + [SystemMessage(content=prompt)]
        
    # not show in ui and not save in db
    output = await ainvoke(messages, config, stream=False)
    tickers = get_array_json(output.content)
    return tickers



def clear_cache(state: AgentState, config: RunnableConfig):
    """
    Clear the action and context from the agent state, effectively resetting the state.
    
    Args:
        state (AgentState): The current state of the agent
        config (RunnableConfig): Configuration for the runnable
        
    Returns:
        dict: A dictionary with cleared action and context
    """
    return {'action': None, 'context': {}}

# Define the workflow graph
# This creates a state graph that orchestrates the agent's behavior through various nodes
workflow = StateGraph(AgentState)
workflow.add_node("planner", planner_node)
workflow.add_node("ticker_switch", ticker_switch)
workflow.add_node("ticker_search", ticker_search)
workflow.add_node("next_step_suggestions", next_step_suggestions)
workflow.add_node("ticker_analysis", ticker_analysis)

workflow.add_node("clear_cache", clear_cache)

# Set entry point and define edges between nodes
workflow.set_entry_point("planner")
workflow.add_edge("ticker_search", 'clear_cache')
workflow.add_edge("next_step_suggestions", 'clear_cache')
workflow.add_edge("ticker_switch", 'clear_cache')
workflow.set_finish_point('clear_cache')

# ticker_analysis node, use conditional edge to switch to different analysis agent
# Add nodes for each analysis agent and connect them to the workflow

for name, node in analysis_agents.items():
    workflow.add_node(name, node)
    workflow.add_edge(name, 'ticker_analysis')
workflow.add_conditional_edges("ticker_analysis", agent_conditional, list(analysis_agents.keys()) + ['clear_cache'])

# Compile the workflow graph into a runnable agent
agent = workflow.compile()
