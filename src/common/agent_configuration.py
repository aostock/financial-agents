from typing import Any, Dict, TypedDict, Optional

class AgentConfiguration(TypedDict):
    """Configurable parameters for the agent.

    Set these when creating assistants OR when invoking the graph.
    See: https://langchain-ai.github.io/langgraph/cloud/how-tos/configuration_cloud/
    """

    settings: Optional[Dict[str, Any]] = None
