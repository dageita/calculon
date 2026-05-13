"""ChatOpenAI subclass that preserves provider `reasoning_content` (DeepSeek thinking mode).

OpenAI-compatible `ChatOpenAI` does not round-trip `reasoning_content` on assistant
messages; DeepSeek returns 400 if a prior thinking-mode assistant turn is replayed
without it. This wrapper stores it on parse and injects it into the request payload.
"""

from __future__ import annotations

from typing import Any

import openai
from langchain_core.messages import AIMessage, AIMessageChunk
from langchain_core.outputs import ChatResult
from langchain_openai import ChatOpenAI


def _merge_reasoning_into_payload(messages: list, payload: dict[str, Any]) -> None:
    api_msgs = payload.get("messages")
    if not isinstance(api_msgs, list) or len(api_msgs) != len(messages):
        return
    for lc_msg, api_msg in zip(messages, api_msgs):
        if not isinstance(lc_msg, AIMessage) or api_msg.get("role") != "assistant":
            continue
        if "reasoning_content" in lc_msg.additional_kwargs:
            api_msg["reasoning_content"] = lc_msg.additional_kwargs["reasoning_content"]


def _attach_reasoning_from_response(response: Any, result: ChatResult) -> None:
    if not result.generations:
        return
    out_msg = result.generations[0].message
    if not isinstance(out_msg, AIMessage):
        return
    if isinstance(response, openai.BaseModel) and getattr(response, "choices", None):
        raw = response.choices[0].message
        if hasattr(raw, "reasoning_content"):
            rc = getattr(raw, "reasoning_content", None)
            if rc is not None:
                out_msg.additional_kwargs["reasoning_content"] = rc
                return
        model_extra = getattr(raw, "model_extra", None)
        if isinstance(model_extra, dict) and (r := model_extra.get("reasoning")) is not None:
            out_msg.additional_kwargs["reasoning_content"] = r
        return
    if isinstance(response, dict):
        choices = response.get("choices") or []
        if not choices:
            return
        m = choices[0].get("message") or {}
        if m.get("reasoning_content") is not None:
            out_msg.additional_kwargs["reasoning_content"] = m["reasoning_content"]


class ReasoningAwareChatOpenAI(ChatOpenAI):
    """Preserves `reasoning_content` for APIs that require it on follow-up turns (e.g. DeepSeek thinking)."""

    def _convert_chunk_to_generation_chunk(
        self,
        chunk: dict,
        default_chunk_class: type,
        base_generation_info: dict | None,
    ):
        generation_chunk = super()._convert_chunk_to_generation_chunk(
            chunk,
            default_chunk_class,
            base_generation_info,
        )
        if (choices := chunk.get("choices")) and generation_chunk:
            top = choices[0]
            if isinstance(generation_chunk.message, AIMessageChunk):
                if (rc := top.get("delta", {}).get("reasoning_content")) is not None:
                    generation_chunk.message.additional_kwargs["reasoning_content"] = rc
                elif (r := top.get("delta", {}).get("reasoning")) is not None:
                    generation_chunk.message.additional_kwargs["reasoning_content"] = r
        return generation_chunk

    def _get_request_payload(
        self,
        input_,
        *,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        messages = self._convert_input(input_).to_messages()
        payload = super()._get_request_payload(input_, stop=stop, **kwargs)
        _merge_reasoning_into_payload(messages, payload)
        return payload

    def _create_chat_result(
        self,
        response: dict | openai.BaseModel,
        generation_info: dict | None = None,
    ) -> ChatResult:
        result = super()._create_chat_result(response, generation_info)
        _attach_reasoning_from_response(response, result)
        return result
