"""
This is the main entry point for the information query agent.
It defines the workflow graph, state, tools, nodes and edges.
"""

from ast import arg
import time
import json
from typing import Any, Dict, List

from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph

from common.agent_state import AgentState
from common.util import get_latest_message_content
from langchain_mcp_adapters.client import MultiServerMCPClient
from llm.llm_model import ainvoke_with_tools, ainvoke
from pydantic_core import ArgsKwargs

client = MultiServerMCPClient(
    {
        "financial_data": {
            # Ensure you start your financial data server on port 8000
            "url": "http://127.0.0.1:8000/mcp",
            "headers": {
                "Authorization":"Bearer secret-token"
            },
            "transport": "streamable_http",
        }
    }
)

async def query(state: AgentState, config: RunnableConfig) -> Dict[str, Any]:
    """Prepare the query for the MCP call.
    
    Args:
        state: The current agent state
        config: The runnable configuration
        
    Returns:
        A dictionary with messages and action
    """
    end_date = state.get('action', {}).get('parameters', {}).get('end_date')
    end_date = end_date if end_date else time.strftime("%Y-%m-%d")

    context = state.get('context', {})
    
    ticker = context.get('current_task', {}).get('ticker', {})

    last_content = get_latest_message_content(state)
    
    return {
        "messages": [],
        "action": None,
    }


async def call_mcp(state: AgentState, config: RunnableConfig) -> Dict[str, Any]:
    """Call the financial_data MCP to get financial information
    
    Args:
        state: The current agent state
        config: The runnable configuration
        
    Returns:
        A dictionary with messages and action
    """
    try:
        # Get tools from the financial_data MCP
        tools = await client.get_tools()
        
        # Get the latest message content
        last_content = get_latest_message_content(state)
        
        # Get ticker information
        context = state.get('context', {})
        ticker = context.get('current_task', {}).get('ticker', {})
        
        # Create messages for the LLM with tools
        messages = [{"role": "user", 'content':f"""Given the current context—stock symbol: {ticker.get('symbol', '')}, company name: {ticker.get('short_name', '')}.

—and the user's latest input: {last_content}.

select the single most appropriate tool from the available toolset for invocation. 
Derive all required tool parameters exclusively from the user input to ensure relevance (e.g., for 'query Apple Inc.', parameters should resolve to AAPL/Apple Inc.). 
If the input provides sufficient clarity for parameter determination (as in this case), invoke exactly one tool. 
If the input lacks specificity or fails to define parameters (e.g., ambiguous or incomplete queries), skip tool invocation entirely. 
Prioritize accuracy over tool usage; never assume parameters beyond the input.

"""}]
        
        # Invoke the LLM with tools
        response = await ainvoke_with_tools(messages, config, tools, analyzer=True)
        
        # Process tool calls if they exist
        tool_messages = []
        if hasattr(response, 'tool_calls') and response.tool_calls:
            for tool_call in response.tool_calls:
                tool_name = tool_call["name"]
                tool_args = tool_call.get('args', {})
                
                # Find the tool in our tools list
                target_tool = None
                for tool in tools:
                    if tool.name == tool_name:
                        target_tool = tool
                        break
                
                # Execute the tool if found
                if target_tool:
                    try:
                        tool_response = await target_tool.ainvoke(tool_args)
                        # if tool_response is not None and isinstance(tool_response, str):
                        #     tool_response = tool_response.strip("```json").strip("```")
                        #     try:
                        #         tool_response = json.loads(tool_response)
                        #     except:
                        #         pass
                        tool_messages.append(ToolMessage(content=str(tool_response), name=tool_name, args=tool_args, tool_call_id=tool_call['id']))
                    except Exception as tool_error:
                        tool_messages.append(ToolMessage(content=f"Error executing tool {tool_name}: {str(tool_error)}", name=tool_name, args=tool_args, tool_call_id=tool_call['id']))
                        # tool_call['result'] = f"Error executing tool {tool_name}: {str(tool_error)}"
    

        summary_prompt = f"""Tool messages: 
```json
{tool_messages}
```
The user's latest input: {last_content}.

Based on the user's input, extract the required information from the tool's response and provide a concise summary in the final answer.
"""
        summary_response = await ainvoke([{"role": "user", 'content':summary_prompt}], config, analyzer=True)

        return {
            "messages": [response] + tool_messages + [summary_response],
            "action": None,
        }
    except Exception as e:
        # Handle any errors that occur during tool execution
        error_message = f"Error calling financial_data MCP: {str(e)}"
        return {
            "messages": [AIMessage(content=error_message)],
            "action": None,
        }

# Define the workflow graph
workflow = StateGraph(AgentState)
workflow.add_node("query", query)
workflow.add_node("call_mcp", call_mcp)
workflow.set_entry_point("query")
workflow.add_edge("query", "call_mcp")

# Compile the workflow graph
agent = workflow.compile()
