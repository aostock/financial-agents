# export all agent graph flow picture
import json
from langgraph.graph.state import CompiledStateGraph

def export_agent_graph_picture(graph: CompiledStateGraph, name: str):
    graph.get_graph().draw_mermaid_png(output_file_path=f"images/agents/{name}.png")

def get_agents_from_langgraph_json():
    with open("langgraph.json", "r") as f:
        data = json.load(f)
        agents = data["graphs"]
        return agents

def auto_import_agent_from_path(agent_path:str):
    import importlib
    import sys
    import os
    # Add current directory and src directory to sys.path if not already there
    current_dir = os.path.dirname(os.path.abspath(__file__))
    src_dir = os.path.join(current_dir, 'src')
    
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)
    
    # Split the agent_path to get module path and attribute name
    if ':' in agent_path:
        module_path, attr_name = agent_path.split(':', 1)
    else:
        module_path, attr_name = agent_path, 'agent'
    
    # Remove .py extension if present
    if module_path.endswith('.py'):
        module_path = module_path[:-3]
    
    # Convert path to module notation
    module_name = module_path.replace('/', '.').replace('\\', '.')
    # Remove leading dots
    module_name = module_name.lstrip('.')
    
    try:
        agent_module = importlib.import_module(module_name)
        agent = getattr(agent_module, attr_name)
        return agent
    except ImportError as e:
        print(f"Error importing {module_name}: {e}")
        raise

def auto_export_agent_graph_picture():
    agents = get_agents_from_langgraph_json()
    for name, agent_path in agents.items():
        agent = auto_import_agent_from_path(agent_path)
        export_agent_graph_picture(agent, name)

if __name__ == "__main__":
    auto_export_agent_graph_picture()
    