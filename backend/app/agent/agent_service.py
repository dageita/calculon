"""LangGraph react agent + checkpoint (Memory or SQLite) wrapping simulator tools."""

from __future__ import annotations

import json
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent

from app.agent.reasoning_chat_openai import ReasoningAwareChatOpenAI
from app.agent.tools import build_simulator_tools
from app.config import settings

try:
    from langgraph.checkpoint.sqlite import SqliteSaver
except ImportError:  # pragma: no cover
    SqliteSaver = None  # type: ignore[misc, assignment]

SYSTEM_PROMPT = """You are an assistant for an LLM training time simulator (Calculon-based).
Rules:
- Never invent batch times, communication times, or GPU timings. Always obtain them by calling run_calculate or run_optimal.
- First call list_simulator_catalog when you need valid GPU names, model names, strategies, or topologies.
- Use get_gpu_datatypes when the user asks which dtypes a GPU profile supports (JSON keys are float16/float32/bfloat16).
- For run_calculate: datatype must be float16, float32, or bfloat16; optional num_procs must equal tensor_par*pipeline_par*data_par; batch_size must be divisible by data_par*microbatch_size.
- Use parse_benchmark_csv only when the user provides raw benchmark CSV text (iteration start/end format), similar to uploading benchmark.csv.
- If the user is vague, ask a short clarifying question or propose defaults and run tools.
- Answer in the same language as the user when possible (e.g. Chinese for Chinese users).
"""


def _make_checkpointer():
    path = (settings.AGENT_CHECKPOINT_SQLITE or "").strip()
    if path:
        if SqliteSaver is None:
            raise RuntimeError(
                "AGENT_CHECKPOINT_SQLITE is set but langgraph-checkpoint-sqlite is not installed."
            )
        parent = path.rsplit("/", 1)[0] if "/" in path else ""
        if parent:
            import os

            os.makedirs(parent, exist_ok=True)
        return SqliteSaver.from_conn_string(path)
    return MemorySaver()


_graph = None
_checkpointer = None


def get_checkpointer():
    """Expose checkpointer for tests and session reset."""
    global _checkpointer
    if _checkpointer is None:
        _checkpointer = _make_checkpointer()
    return _checkpointer


def get_agent_graph():
    global _graph
    if _graph is None:
        llm = ReasoningAwareChatOpenAI(
            model=settings.AGENT_MODEL,
            api_key=settings.OPENAI_API_KEY,
            base_url=settings.OPENAI_BASE_URL or None,
            temperature=0.0,
            timeout=settings.AGENT_LLM_TIMEOUT_SEC,
            max_retries=settings.AGENT_LLM_MAX_RETRIES,
        )
        tools = build_simulator_tools()
        cp = get_checkpointer()
        _graph = create_react_agent(
            llm,
            tools,
            prompt=SystemMessage(content=SYSTEM_PROMPT),
            checkpointer=cp,
        )
    return _graph


def reset_agent_graph() -> None:
    """Test hook: force rebuild after settings change."""
    global _graph, _checkpointer
    _graph = None
    _checkpointer = None


def delete_thread_checkpoint(thread_id: str) -> None:
    """Remove all checkpoints for a thread (new session on client should use a new thread_id)."""
    try:
        get_checkpointer().delete_thread(thread_id)
    except Exception:
        pass


def get_thread_messages_for_test(thread_id: str) -> List[BaseMessage]:
    """Test hook: messages restored from checkpointer for thread."""
    graph = get_agent_graph()
    snap = graph.get_state({"configurable": {"thread_id": thread_id}})
    return list(snap.values.get("messages") or [])


def _simulator_fields_from_tool_output(tool_name: Optional[str], full_text: str) -> dict[str, Any]:
    """Parse full (untruncated) tool JSON so large fields can be returned separately for the UI."""
    extra: dict[str, Any] = {}
    if tool_name not in ("run_calculate", "run_optimal") or not (full_text or "").strip():
        return extra
    try:
        obj = json.loads(full_text)
    except json.JSONDecodeError:
        return extra
    if not isinstance(obj, dict) or obj.get("status") == "error":
        return extra
    te = obj.get("timeline_events")
    if isinstance(te, list) and te:
        extra["timeline_events"] = te
    summary = obj.get("summary")
    if isinstance(summary, dict):
        extra["simulator_summary"] = summary
    return extra


def _extract_reply_and_steps(messages: List[BaseMessage]) -> Tuple[str, List[dict[str, Any]]]:
    reply = ""
    for m in reversed(messages):
        if isinstance(m, AIMessage):
            c = m.content
            if c:
                reply = c if isinstance(c, str) else str(c)
                break

    steps_out: List[dict[str, Any]] = []
    for i, m in enumerate(messages):
        if not isinstance(m, AIMessage) or not m.tool_calls:
            continue
        for tc in m.tool_calls:
            tid = tc.get("id") or ""
            name = tc.get("name")
            args = tc.get("args") if isinstance(tc.get("args"), dict) else tc.get("args")
            full_obs = ""
            for m2 in messages[i + 1 :]:
                if isinstance(m2, ToolMessage) and (m2.tool_call_id == tid or not tid):
                    full_obs = str(m2.content)
                    break
            extra = _simulator_fields_from_tool_output(name, full_obs)
            obs_cap = 16_000
            obs_display = full_obs[:obs_cap]
            if len(full_obs) > obs_cap:
                obs_display += f"\n... [truncated, total {len(full_obs)} chars]"
            step: dict[str, Any] = {
                "tool": name,
                "tool_input": args,
                "observation": obs_display,
            }
            if "timeline_events" in extra:
                step["timeline_events"] = extra["timeline_events"]
            if "simulator_summary" in extra:
                step["simulator_summary"] = extra["simulator_summary"]
            steps_out.append(step)
    return reply, steps_out


def _agent_invoke_config(thread_id: str) -> dict[str, Any]:
    return {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": max(25, settings.AGENT_MAX_ITERATIONS * 4),
    }


def invoke_agent(thread_id: str, message: str) -> Tuple[str, List[dict[str, Any]]]:
    graph = get_agent_graph()
    result = graph.invoke(
        {"messages": [HumanMessage(content=message)]},
        config=_agent_invoke_config(thread_id),
    )
    msgs = list(result.get("messages") or [])
    reply, steps_out = _extract_reply_and_steps(msgs)
    return reply, steps_out


def _sse_data(obj: dict) -> str:
    return f"data: {json.dumps(obj, ensure_ascii=False, default=str)}\n\n"


async def stream_agent_sse(thread_id: str, message: str) -> AsyncIterator[str]:
    """SSE lines (`data: {...}\\n\\n`) from LangGraph event stream (v2)."""
    graph = get_agent_graph()
    last_messages: List[BaseMessage] = []

    async for ev in graph.astream_events(
        {"messages": [HumanMessage(content=message)]},
        config=_agent_invoke_config(thread_id),
        version="v2",
    ):
        et = ev.get("event")
        if et == "on_tool_start":
            data = ev.get("data") or {}
            tool_name = data.get("name") or ev.get("name")
            yield _sse_data({"type": "tool_start", "tool": tool_name, "input": data.get("input")})
        elif et == "on_tool_end":
            data = ev.get("data") or {}
            out = data.get("output")
            obs = str(out)[:16_000] if out is not None else ""
            yield _sse_data({"type": "tool_end", "observation": obs})
        elif et == "on_chat_model_stream":
            chunk = (ev.get("data") or {}).get("chunk")
            text = getattr(chunk, "content", None) if chunk is not None else None
            if text:
                yield _sse_data({"type": "token", "text": text})
        elif et == "on_chain_end":
            data = ev.get("data") or {}
            out = data.get("output")
            if isinstance(out, dict) and "messages" in out:
                last_messages = list(out.get("messages") or [])

    if not last_messages:
        snap = graph.get_state({"configurable": {"thread_id": thread_id}})
        last_messages = list(snap.values.get("messages") or [])

    if last_messages:
        reply, formatted_steps = _extract_reply_and_steps(last_messages)
        yield _sse_data({"type": "result", "reply": reply, "intermediate_steps": formatted_steps})
    yield _sse_data({"type": "done"})
