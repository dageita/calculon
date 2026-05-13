from typing import Any, List, Optional

from pydantic import BaseModel, Field


class AgentSessionResponse(BaseModel):
    thread_id: str = Field(..., description="Use this id for subsequent /agent/chat calls.")


class AgentChatRequest(BaseModel):
    thread_id: str
    message: str = Field(..., min_length=1)


class AgentIntermediateStep(BaseModel):
    tool: Optional[str] = None
    tool_input: Optional[Any] = None
    observation: str = ""


class AgentChatResponse(BaseModel):
    reply: str = ""
    intermediate_steps: List[AgentIntermediateStep] = Field(default_factory=list)
    error: Optional[str] = None
