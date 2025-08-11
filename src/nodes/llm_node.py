from common.agent_state import AgentState
from langchain_core.runnables import RunnableConfig
from typing import Dict, Any
from llm.llm_model import ainvoke


class LLMNode():
    """
    This class represents a large language model node, responsible for invoking the language model and processing the output.
    It receives configuration options during initialization and performs model inference based on the incoming agent state and run configuration when called.
    """
    def __init__(self, options: Dict[str, Any]):
        self.options = options

    async def __call__(self, state: AgentState, config: RunnableConfig) -> Dict[str, Any]:
        output = await ainvoke(self.options["messages"], config)
        return {"messages": [output]}