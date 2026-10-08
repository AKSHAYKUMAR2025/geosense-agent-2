# test_phase4_agent.py
# Purpose: Automated tests to verify Phase 4 modules work correctly
# Run with: python -m pytest tests/ -v
# NOTE: requires the MCP server running locally on port 8000

import sys
import httpx

sys.path.append("src/phase4_agent")

from geoai_agent import ask_agent


MCP_URL = "http://localhost:8000"


# ---------------------------------------------------------
# Test 1 — MCP server documentation
# ---------------------------------------------------------

def test_mcp_server_docs_reachable():

    resp = httpx.get(
        f"{MCP_URL}/docs"
    )

    assert resp.status_code == 200, (
        "MCP server docs page should be reachable"
    )


# ---------------------------------------------------------
# Test 2 — Site scoring tool
# ---------------------------------------------------------

def test_score_location_tool():

    resp = httpx.post(
        f"{MCP_URL}/tools/score_location",
        json={
            "latitude": 13.0827,
            "longitude": 80.2707
        }
    )

    assert resp.status_code == 200

    assert "opportunity_score" in resp.json()


# ---------------------------------------------------------
# Test 3 — Agent returns an answer
# ---------------------------------------------------------

def test_agent_returns_nonempty_answer():

    answer = ask_agent(
        "Is this location suitable for a clinic?"
    )

    assert isinstance(answer, str)

    assert len(answer) > 0, (
        "Agent should return a non-empty answer"
    )


# ---------------------------------------------------------
# Test 4 — Agent selects at least one tool
# ---------------------------------------------------------

def test_agent_selects_at_least_one_tool():

    from geoai_agent import build_agent_graph

    app = build_agent_graph()

    config = {
        "configurable": {
            "thread_id": "test-thread"
        }
    }

    result = app.invoke(
        {
            "question": "Score this location for flood risk",
            "tool_calls": [],
            "tool_results": [],
            "final_answer": ""
        },
        config
    )

    assert len(result["tool_calls"]) > 0, (
        "Agent should select at least one tool"
    )