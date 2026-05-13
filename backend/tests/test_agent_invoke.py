"""Smoke tests for agent invoke (mock graph; no real OPENAI_API_KEY).

Run: PYTHONPATH=backend pytest backend/tests/test_agent_invoke.py -v
"""

from __future__ import annotations

from collections import defaultdict
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage


def test_invoke_agent_runs_list_catalog_tool():
    pytest.importorskip("langchain_core")

    from app.agent.agent_service import invoke_agent, reset_agent_graph

    g = MagicMock()
    g.threads = {}

    def invoke_once(state, config=None):
        tid = config["configurable"]["thread_id"]
        human = state["messages"][-1]
        g.threads[tid] = [
            human,
            AIMessage(
                content="",
                tool_calls=[
                    {"name": "list_simulator_catalog", "args": {}, "id": "call_mock_1"},
                ],
            ),
            ToolMessage(
                content='{"gpus":["A"]}',
                tool_call_id="call_mock_1",
                name="list_simulator_catalog",
            ),
            AIMessage(content="已列出可用 GPU / 模型。"),
        ]
        return {"messages": list(g.threads[tid])}

    g.invoke.side_effect = invoke_once
    g.get_state.side_effect = lambda cfg: SimpleNamespace(
        values={"messages": list(g.threads.get(cfg["configurable"]["thread_id"], []))}
    )

    with patch("app.agent.agent_service.get_agent_graph", return_value=g):
        reset_agent_graph()
        reply, steps = invoke_agent("thread-smoke-1", "有哪些 GPU？")

    tools_used = [s.get("tool") for s in steps]
    assert "list_simulator_catalog" in tools_used
    assert reply


def test_invoke_agent_same_thread_preserves_history():
    pytest.importorskip("langchain_core")

    from app.agent.agent_service import get_thread_messages_for_test, invoke_agent, reset_agent_graph

    g = MagicMock()
    g.threads = defaultdict(list)
    calls = defaultdict(int)

    def invoke_multi(state, config=None):
        tid = config["configurable"]["thread_id"]
        human = state["messages"][-1]
        g.threads[tid].append(human)
        calls[tid] += 1
        if calls[tid] % 2 == 1:
            cid = f"call_{calls[tid]}"
            g.threads[tid].append(
                AIMessage(
                    content="",
                    tool_calls=[{"name": "list_simulator_catalog", "args": {}, "id": cid}],
                )
            )
            g.threads[tid].append(ToolMessage(content="{}", tool_call_id=cid, name="list_simulator_catalog"))
        g.threads[tid].append(AIMessage(content="ok"))
        return {"messages": list(g.threads[tid])}

    g.invoke.side_effect = invoke_multi
    g.get_state.side_effect = lambda cfg: SimpleNamespace(
        values={"messages": list(g.threads.get(cfg["configurable"]["thread_id"], []))}
    )

    with patch("app.agent.agent_service.get_agent_graph", return_value=g):
        reset_agent_graph()
        tid = "thread-smoke-2"
        invoke_agent(tid, "第一次：列出目录")
        invoke_agent(tid, "第二次：再列一次")
        msgs = get_thread_messages_for_test(tid)
    assert len(msgs) >= 2


@pytest.fixture(autouse=True)
def fake_openai_key(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key-for-mock")
