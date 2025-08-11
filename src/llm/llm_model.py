from langchain.schema import AIMessage
from langchain_litellm import ChatLiteLLMRouter
from litellm import Router
from langchain_core.runnables import RunnableConfig
from common.settings import Settings
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

from langchain_mcp_adapters.tools import load_mcp_tools, BaseTool
from langgraph.prebuilt import create_react_agent
from langchain_mcp_adapters.client import MultiServerMCPClient



async def get_tools(server_params) -> list[BaseTool]:
    async with streamablehttp_client(**server_params) as (read, write, _):
        async with ClientSession(read, write) as session:
            # Initialize the connection
            await session.initialize()
            # Load the remote graph as if it was a tool
            tools = await load_mcp_tools(session)
            return tools

def get_llm(settings: Settings)->ChatLiteLLMRouter:
    litellm_router = Router(model_list=settings.get_model_list())
    llm = ChatLiteLLMRouter(router=litellm_router, model_name=settings.get_intent_recognition_model().get("model", ""))
    return llm

def get_analyzer(settings: Settings)->ChatLiteLLMRouter:
    litellm_router = Router(model_list=settings.get_model_list())
    llm = ChatLiteLLMRouter(router=litellm_router, model_name=settings.get_analysis_model().get("model", ""))
    return llm


async def ainvoke(messages, config: RunnableConfig,  stream=True, analyzer=False):
    # create a new UUID
    # if hidden_stream:
    #     run_id = uuid4()
    #     if config is None:
    #         config = {}
    #     config["run_id"] = run_id
    #     message_id = f'{_LC_ID_PREFIX}-{run_id}'
    #     writer({
    #         "type": "hidden_stream",
    #         "message_id": message_id
    #     })
    settings = Settings(config)
    if analyzer:
        return await get_analyzer(settings).ainvoke(messages, config, stream=stream)
    return await get_llm(settings).ainvoke(messages, config, stream=stream)


async def ainvoke_with_tools(messages, config: RunnableConfig, tools: list[BaseTool], stream=True, analyzer=False):
    settings = Settings(config)
    llm = None
    if analyzer:
        llm = get_analyzer(settings)
    else:
        llm = get_llm(settings)
    agent = llm.bind_tools(tools, tool_choice="auto")
    return await agent.ainvoke(messages, config, stream=stream)



async def ainvoke2(messages, config: RunnableConfig, response_metadata = None):
    full_content = ""
    message_id = None
    final_response_metadata = {}
    if response_metadata is None:
        response_metadata = {}
    final_response_metadata.update(response_metadata)
    settings = Settings(config)
    async for event in get_llm(settings).astream_events(messages, config=config, version="v2"):
        if event["event"] == "on_chat_model_stream":
            chunk = event["data"]["chunk"]
            if chunk.content:
                full_content += chunk.content
                if message_id is None:
                    message_id = chunk.id
                    # 只在第一次 chunk 中更新 response_metadata， chunk的操作是追加， 第一次更新后， 后续前端都会有这个值
                    chunk.response_metadata.update(response_metadata)
                else:
                    final_response_metadata.update(chunk.response_metadata)


    return AIMessage(content=full_content, id=message_id, response_metadata=final_response_metadata)

