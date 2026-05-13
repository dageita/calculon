"""Unit tests for reasoning_content round-trip."""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from app.agent.reasoning_chat_openai import _merge_reasoning_into_payload


def test_merge_reasoning_into_payload_for_tool_turn():
    messages = [
        HumanMessage(content="hi"),
        AIMessage(
            content="",
            tool_calls=[{"name": "list_simulator_catalog", "args": {}, "id": "c1"}],
            additional_kwargs={"reasoning_content": "step-by-step plan"},
        ),
        ToolMessage(content="{}", tool_call_id="c1", name="list_simulator_catalog"),
    ]
    payload = {
        "messages": [
            {"role": "user", "content": "hi"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "list_simulator_catalog", "arguments": "{}"}}],
            },
            {"role": "tool", "content": "{}", "tool_call_id": "c1"},
        ]
    }
    _merge_reasoning_into_payload(messages, payload)
    assistant = payload["messages"][1]
    assert assistant.get("reasoning_content") == "step-by-step plan"
