import os
from fastapi import FastAPI
import uvicorn
from copilotkit.integrations.fastapi import add_fastapi_endpoint 
from copilotkit import CopilotKitRemoteEndpoint, LangGraphAgent 
from agents.warren_buffett.agent import agent as warren_buffett_agent
from agents.information_query.agent import agent as information_query_agent
from agents.agent import agent
 
from dotenv import load_dotenv
load_dotenv()
 
app = FastAPI()

sdk = CopilotKitRemoteEndpoint(
    agents=[
        LangGraphAgent(
            name="agent", # the name of your agent defined in langgraph.json
            description="agent",
            graph=agent, # the graph object from your langgraph import
        ),
        LangGraphAgent(
            name="warren_buffett", # the name of your agent defined in langgraph.json
            description="Describe your agent here, will be used for multi-agent orchestration",
            graph=warren_buffett_agent, # the graph object from your langgraph import
        ),
        LangGraphAgent(
            name="information_query", # the name of your agent defined in langgraph.json
            description="Provides detailed company information and key financial metrics analysis",
            graph=information_query_agent, # the graph object from your langgraph import
        )
    ],
)
 
# Use CopilotKit's FastAPI integration to add a new endpoint for your LangGraph agents #
add_fastapi_endpoint(app, sdk, "/api/copilotkit", use_thread_pool=False)
 
# add new route for health check
@app.get("/health")
def health():
    """Health check."""
    return {"status": "ok"}
 
if __name__ == "__main__":
    """Run the uvicorn server."""
    port = int(os.getenv("PORT", "2024"))
    uvicorn.run(
        "main:app", # the path to your FastAPI file, replace this if its different
        host="0.0.0.0",
        port=port,
        reload=True,
    )