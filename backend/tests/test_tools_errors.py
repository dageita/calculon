"""Structured tool errors return JSON strings (matches CalculateRepository error shape)."""

from __future__ import annotations

import json

import pytest


def _require_agent_deps():
    pytest.importorskip("langchain_core")
    try:
        import calculon.llm.runner  # noqa: F401
    except Exception as exc:
        pytest.skip(str(exc))


def test_run_calculate_unknown_gpu_returns_error_dict_json():
    _require_agent_deps()
    from app.config import settings
    from app.agent.tools import run_calculate

    model_name = next(m.name for m in settings.MODEL_LIST if m.name)
    out = run_calculate.invoke(
        {
            "gpu_name": "__invalid_gpu_xyz__",
            "model_name": model_name,
            "batch_size": 8,
            "microbatch_size": 4,
        }
    )
    data = json.loads(out)
    assert data.get("status") == "error"
    assert "gpu_name" in data.get("error", "").lower() or "unknown" in data.get("error", "").lower()


def test_run_calculate_num_procs_mismatch_returns_error_json():
    _require_agent_deps()
    from app.config import settings
    from app.agent.tools import run_calculate

    gpu_name = next(g.name for g in settings.GPU_LIST if g.name)
    model_name = next(m.name for m in settings.MODEL_LIST if m.name)
    out = run_calculate.invoke(
        {
            "gpu_name": gpu_name,
            "model_name": model_name,
            "batch_size": 8,
            "microbatch_size": 4,
            "tensor_par": 2,
            "pipeline_par": 1,
            "data_par": 1,
            "num_procs": 99,
        }
    )
    data = json.loads(out)
    assert data.get("status") == "error"
    assert "num_procs" in data.get("error", "").lower()


def test_run_calculate_batch_not_divisible_by_dp_times_micro():
    _require_agent_deps()
    from app.config import settings
    from app.agent.tools import run_calculate

    gpu_name = next(g.name for g in settings.GPU_LIST if g.name)
    model_name = next(m.name for m in settings.MODEL_LIST if m.name)
    out = run_calculate.invoke(
        {
            "gpu_name": gpu_name,
            "model_name": model_name,
            "batch_size": 10,
            "microbatch_size": 4,
            "tensor_par": 1,
            "pipeline_par": 1,
            "data_par": 2,
        }
    )
    data = json.loads(out)
    assert data.get("status") == "error"
    assert "batch_size" in data.get("error", "").lower()
