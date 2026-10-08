# geoai_agent.py
# Purpose: LangGraph-based GeoAI Agent that chains MCP tool calls
# to answer plain English questions

import os
import httpx

from typing import TypedDict

from dotenv import load_dotenv

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from langchain_anthropic import ChatAnthropic


# ---------------------------------------------------------
# Load environment variables
# ---------------------------------------------------------

load_dotenv()


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

MCP_SERVER_URL = os.getenv(
    "MCP_SERVER_URL",
    "http://localhost:8000"
)

MODEL = os.getenv(
    "AGENT_MODEL",
    "claude-sonnet-4-6"
)


# ---------------------------------------------------------
# Agent state
# ---------------------------------------------------------

class AgentState(TypedDict):
    question: str
    tool_calls: list
    tool_results: list
    final_answer: str


# ---------------------------------------------------------
# Claude LLM
# ---------------------------------------------------------

llm = ChatAnthropic(
    model=MODEL,
    api_key=os.getenv("ANTHROPIC_API_KEY")
)


# ---------------------------------------------------------
# Available MCP tools
# ---------------------------------------------------------

TOOL_CATALOG = [
    "score_location",
    "classify_imagery",
    "lidar_profile",
    "streetview_scene",
    "get_climate_risk",
    "get_demographics",
    "find_similar_sites",
]


# ---------------------------------------------------------
# Node 1 — Understand the user's question
# ---------------------------------------------------------

def understand_query(state: AgentState) -> AgentState:
    """
    Node 1:
    Ask Claude which MCP tools are relevant to the question.
    """

    prompt = (
        f"Given this user question: '{state['question']}'\n"
        f"and this list of available tools: {TOOL_CATALOG}\n"
        f"Return a comma-separated list of the tool names "
        f"that should be called."
    )

    response = llm.invoke(prompt)

    # ChatAnthropic normally returns text in response.content.
    # Keep only valid tool names.
    chosen = [
        tool.strip()
        for tool in response.content.split(",")
        if tool.strip() in TOOL_CATALOG
    ]

    state["tool_calls"] = chosen

    return state


# ---------------------------------------------------------
# Node 2 — Execute selected MCP tools
# ---------------------------------------------------------

def execute_tools(state: AgentState) -> AgentState:
    """
    Node 2:
    Call each selected MCP tool.

    The beginner version uses the documented Chennai
    demonstration coordinate:
        latitude  = 13.0827
        longitude = 80.2707

    LiDAR requires a bounding box, so it receives the
    documented Sicily LiDAR study-area bbox.
    """

    results = []

    with httpx.Client() as client:

        for tool_name in state["tool_calls"]:

            # ---------------------------------------------
            # Standard location-based tools
            # ---------------------------------------------

            if tool_name != "lidar_profile":

                payload = {
                    "latitude": 13.0827,
                    "longitude": 80.2707
                }

            # ---------------------------------------------
            # LiDAR requires a bounding box
            # ---------------------------------------------

            else:

                payload = {
                    "min_lat": 38.413989,
                    "min_lon": 14.957943,
                    "max_lat": 38.416862,
                    "max_lon": 14.961042
                }

            # ---------------------------------------------
            # Call MCP endpoint
            # ---------------------------------------------

            response = client.post(
                f"{MCP_SERVER_URL}/tools/{tool_name}",
                json=payload,
                timeout=120
            )

            response.raise_for_status()

            results.append({
                "tool": tool_name,
                "result": response.json()
            })

    state["tool_results"] = results

    return state


# ---------------------------------------------------------
# Node 3 — Synthesize final response
# ---------------------------------------------------------

def synthesize_response(state: AgentState) -> AgentState:
    """
    Node 3:
    Ask Claude to turn the raw MCP tool results into
    a clear final answer.
    """

    prompt = (
        f"User asked: '{state['question']}'\n\n"
        f"Tool results: {state['tool_results']}\n\n"
        f"Write a clear, concise answer for the user "
        f"based only on this evidence."
    )

    response = llm.invoke(prompt)

    state["final_answer"] = response.content

    return state


# ---------------------------------------------------------
# Build LangGraph agent
# ---------------------------------------------------------

def build_agent_graph():

    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node(
        "understand_query",
        understand_query
    )

    graph.add_node(
        "execute_tools",
        execute_tools
    )

    graph.add_node(
        "synthesize_response",
        synthesize_response
    )

    # Entry point
    graph.set_entry_point(
        "understand_query"
    )

    # Graph flow
    graph.add_edge(
        "understand_query",
        "execute_tools"
    )

    graph.add_edge(
        "execute_tools",
        "synthesize_response"
    )

    graph.add_edge(
        "synthesize_response",
        END
    )

    # Short-term conversation memory
    memory = MemorySaver()

    return graph.compile(
        checkpointer=memory
    )


# ---------------------------------------------------------
# Public agent function
# ---------------------------------------------------------

def ask_agent(
    question,
    thread_id="default-session"
):

    app = build_agent_graph()

    config = {
        "configurable": {
            "thread_id": thread_id
        }
    }

    result = app.invoke(
        {
            "question": question,
            "tool_calls": [],
            "tool_results": [],
            "final_answer": ""
        },
        config
    )

    return result["final_answer"]


# ---------------------------------------------------------
# Command-line test
# ---------------------------------------------------------

if __name__ == "__main__":

    answer = ask_agent(
        "Is this location in Chennai suitable for a new clinic?"
    )

    print("\n" + "=" * 70)
    print("GEOSENSE GEOAI AGENT")
    print("=" * 70)

    print("\nQuestion:")
    print("Is this location in Chennai suitable for a new clinic?")

    print("\nAgent Answer:")
    print(answer)

    print("\n" + "=" * 70)