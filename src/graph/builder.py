from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from .state import AgentState
from agents.ricardo_garcia import ricardo_garcia_agent

# Define your agents and tools
analyst_agents = {
    "ricardo_garcia_agent": ricardo_garcia_agent,
}
tools = [] # Add tools here if you have any
tool_node = ToolNode(tools)

def master_agent_router(state: AgentState):
    # This simple router will always yield the ricardo_garcia_agent
    # for processing.
    yield "ricardo_garcia_agent"

def create_graph():
    """
    Builds the execution graph for the agents.
    """
    workflow = StateGraph(AgentState)

    workflow.add_node("master_agent", master_agent_router)
    
    for name, agent in analyst_agents.items():
        workflow.add_node(name, agent)
    
    workflow.add_node("tools", tool_node)

    workflow.set_entry_point("master_agent")

    # The router will now determine the next step
    workflow.add_edge("master_agent", "ricardo_garcia_agent")
    workflow.add_edge("ricardo_garcia_agent", END)
    workflow.add_edge("tools", END)
    
    return workflow.compile() 