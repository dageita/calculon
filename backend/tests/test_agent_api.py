"""HTTP surface for agent tools catalog (no OPENAI_API_KEY required)."""

from __future__ import annotations

import asyncio

import httpx
import pytest


def test_agent_tools_list_endpoint():
    try:
        from main import get_application
    except Exception as exc:
        pytest.skip(str(exc))

    app = get_application()

    async def _call():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            return await client.get("/llm_training_calculator/agent/tools")

    r = asyncio.run(_call())
    assert r.status_code == 200
    body = r.json()
    assert "tools" in body
    names = {t["name"] for t in body["tools"]}
    assert "run_calculate" in names
    assert "list_simulator_catalog" in names
    assert "get_gpu_datatypes" in names
