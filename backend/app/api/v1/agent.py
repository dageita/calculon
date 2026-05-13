import json
import logging
import uuid

import fastapi
from fastapi import HTTPException
from fastapi.responses import StreamingResponse

from app.agent.tool_catalog import tool_catalog_entries
from app.agent.tools import build_simulator_tools
from app.agent.agent_service import (
    delete_thread_checkpoint,
    invoke_agent,
    stream_agent_sse,
)
from app.config import settings
from app.models.agent_chat import AgentChatRequest, AgentChatResponse, AgentIntermediateStep, AgentSessionResponse

logger = logging.getLogger(__name__)
router = fastapi.APIRouter()


@router.post("/sessions", response_model=AgentSessionResponse)
def create_session():
    return AgentSessionResponse(thread_id=str(uuid.uuid4()))


@router.post("/sessions/{thread_id}/reset")
def reset_session(thread_id: str):
    """Clear LangGraph checkpoint history for this thread (same id will start fresh)."""
    delete_thread_checkpoint(thread_id)
    return {"ok": True}


@router.get("/tools")
def list_agent_tools():
    """OpenAI-style tool name/description/parameters for simulator Agent tools."""
    return {"tools": tool_catalog_entries(build_simulator_tools())}


@router.post("/chat", response_model=AgentChatResponse)
def chat(body: AgentChatRequest):
    if not settings.OPENAI_API_KEY:
        raise HTTPException(
            status_code=503,
            detail="LLM is not configured: set OPENAI_API_KEY in the environment or .env.",
        )
    try:
        reply, steps = invoke_agent(body.thread_id, body.message)
    except Exception as e:
        logger.exception("agent chat failed")
        return AgentChatResponse(reply="", error=str(e))
    return AgentChatResponse(
        reply=reply,
        intermediate_steps=[
            AgentIntermediateStep(tool=s.get("tool"), tool_input=s.get("tool_input"), observation=s.get("observation", ""))
            for s in steps
        ],
    )


@router.post("/chat/stream")
async def chat_stream(body: AgentChatRequest):
    if not settings.OPENAI_API_KEY:
        raise HTTPException(
            status_code=503,
            detail="LLM is not configured: set OPENAI_API_KEY in the environment or .env.",
        )

    async def event_gen():
        try:
            async for line in stream_agent_sse(body.thread_id, body.message):
                yield line
        except Exception as e:
            logger.exception("agent stream failed")
            yield f"data: {json.dumps({'type': 'error', 'error': str(e)}, ensure_ascii=False)}\n\n"

    return StreamingResponse(event_gen(), media_type="text/event-stream")
