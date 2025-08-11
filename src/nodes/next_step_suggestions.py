from common.agent_state import AgentState
from langchain_core.runnables import RunnableConfig
from typing import Dict, Any
from llm.llm_model import ainvoke
from common.util import get_array_json
from langchain_core.messages import AIMessage, SystemMessage
from common import markdown



class NextStepSuggestions():
    def __init__(self, options: Dict[str, Any]):
        self.options = options

    async def __call__(self, state: AgentState, config: RunnableConfig) -> Dict[str, Any]:
        action = state.get("action")
        suggestions = []
        if action is not None and action.get("parameters") is not None and action.get("parameters").get("suggestions") is not None:
            suggestions = action.get("parameters").get("suggestions")
        else:
            messages = state["messages"]
            prompt = f"""Based on current conversation, predict user intent and generate 2 intelligent question suggestions:
    1. Questions should be specific and valuable
    2. Avoid overly broad or repetitive queries
    3. Consider users' actual scenario needs
    4. Uniform format for easy selection

    Please output in the following JSON format:
    [ "Question 1","Question 2"]
    """
            messages = state["messages"] + [SystemMessage(content=prompt)]

            # not show in ui and not save in db
            output = await ainvoke(messages, config, stream=False)
            suggestions = get_array_json(output.content)
        content = f"""## 🔍 Next Steps Suggestions
{markdown.list_str_to_sequence(suggestions)}
"""
        return {"suggestions": suggestions, "messages": AIMessage(content=content)}



