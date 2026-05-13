"""Serializable LangChain tool specs for clients (OpenAI-style name/description/parameters)."""

from __future__ import annotations

from typing import Any, Dict, List

from langchain_core.tools import BaseTool


def tool_catalog_entries(tools: List[BaseTool]) -> List[Dict[str, Any]]:
    """Return one dict per tool: name, description, parameters (JSON Schema when available)."""
    out: List[Dict[str, Any]] = []
    for t in tools:
        entry: Dict[str, Any] = {
            "name": t.name,
            "description": (t.description or "").strip(),
        }
        schema = getattr(t, "args_schema", None)
        if schema is not None:
            try:
                entry["parameters"] = schema.model_json_schema()
            except Exception:
                entry["parameters"] = {"type": "object", "properties": {}}
        else:
            entry["parameters"] = {"type": "object", "properties": {}}
        out.append(entry)
    return out
