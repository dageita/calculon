"""LangChain tools wrapping SimulatorFacade and calculator/benchmark REST-aligned helpers."""

from __future__ import annotations

import json
import os
from typing import Any, List, Optional

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from app.agent.simulator_facade import SimulatorFacade
from app.config import settings
from app.core.benchmark_repository import BenchmarkRepository
from app.core.calculate_repository import OptimizationStrategyType, NetworkTopologyType
from app.models.calculator_input import Gpu, Model, Network, OptimalConfig, TrainningConfig

_facade = SimulatorFacade()


def _simulator_root() -> str:
    """Workspace root (contains systems/). Same depth as CalculateRepository project_root."""
    agent_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(agent_dir, "..", "..", ".."))


def _gpu_datatypes_dict(gpu_name: str) -> dict[str, Any]:
    systems_dir = os.path.join(_simulator_root(), "systems")
    gpu_file_path = os.path.join(systems_dir, f"{gpu_name}.json")
    if not os.path.exists(gpu_file_path):
        alt = os.path.join(systems_dir, f"{gpu_name.lower()}.json")
        gpu_file_path = alt if os.path.exists(alt) else gpu_file_path
    if not os.path.exists(gpu_file_path):
        return {"status": "error", "error": f"GPU config not found: {gpu_file_path}"}
    with open(gpu_file_path, "r", encoding="utf-8") as f:
        gpu_config = json.load(f)
    if "matrix" not in gpu_config or "vector" not in gpu_config:
        return {"status": "error", "error": "GPU JSON missing matrix or vector"}
    matrix_datatypes = set(gpu_config["matrix"].keys())
    vector_datatypes = set(gpu_config["vector"].keys())
    common = sorted(matrix_datatypes.intersection(vector_datatypes))
    if not common:
        return {"status": "error", "error": "No common datatypes in matrix/vector"}
    return {"gpu_name": gpu_name, "datatypes": common}


def _catalog_dict() -> dict[str, Any]:
    return {
        "gpus": [g.name for g in settings.GPU_LIST if g.name],
        "models": [m.name for m in settings.MODEL_LIST if m.name],
        "optimization_strategies": [s.value for s in OptimizationStrategyType],
        "network_topologies": [t.value for t in NetworkTopologyType],
    }


def _resolve_gpu(gpu_name: str) -> Gpu:
    for g in settings.GPU_LIST:
        if g.name == gpu_name:
            return g
    raise ValueError(f"Unknown gpu_name={gpu_name!r}. Use list_simulator_catalog.")


def _resolve_model(model_name: str) -> Model:
    for m in settings.MODEL_LIST:
        if m.name == model_name:
            return m
    raise ValueError(f"Unknown model_name={model_name!r}. Use list_simulator_catalog.")


def _normalize_datatype(datatype: str) -> str:
    """Map short names to Calculon System.TypeSizes keys (float16, float32, bfloat16)."""
    key = (datatype or "").strip().lower()
    aliases = {
        "fp16": "float16",
        "float16": "float16",
        "fp32": "float32",
        "float32": "float32",
        "bf16": "bfloat16",
        "bp16": "bfloat16",
        "bfloat16": "bfloat16",
    }
    return aliases.get(key, key)


def _dump_result(result: Any, max_chars: int = 120_000) -> str:
    """Serialize simulator output for the model; truncate very large payloads."""

    if isinstance(result, dict) and result.get("status") == "error":
        return json.dumps(result, ensure_ascii=False)

    try:
        text = json.dumps(result, ensure_ascii=False, default=str)
    except TypeError:
        text = str(result)
    if len(text) > max_chars:
        return text[:max_chars] + f"\n... [truncated, total {len(text)} chars]"
    return text


@tool
def list_simulator_catalog() -> str:
    """Return JSON with valid gpu names, model names, optimization strategies, and network topologies."""
    return json.dumps(_catalog_dict(), ensure_ascii=False)


class RunCalculateInput(BaseModel):
    gpu_name: str = Field(description="Must match a GPU name from list_simulator_catalog.")
    model_name: str = Field(description="Must match a model name from list_simulator_catalog.")
    batch_size: int = Field(ge=1)
    microbatch_size: int = Field(ge=1)
    network_bandwidth_gbps: float = Field(
        0.0, description="Inter-node network bandwidth (Gbps); use 0 for single-machine style setups."
    )
    network_topology: str = Field(
        "Single machine",
        description="Must be one of network_topologies from list_simulator_catalog.",
    )
    optimization_strategy: str = Field(
        "Full recomputation",
        description="Must be one of optimization_strategies from list_simulator_catalog.",
    )
    tensor_par: int = Field(1, ge=1)
    pipeline_par: int = Field(1, ge=1)
    data_par: int = Field(1, ge=1)
    num_procs: Optional[int] = Field(
        default=None,
        description="Total GPU count; if set must equal tensor_par * pipeline_par * data_par.",
    )
    datatype: str = Field(
        "float16",
        description="Training datatype: float16, float32, or bfloat16 (short aliases fp16/fp32/bf16 accepted).",
    )


@tool(args_schema=RunCalculateInput)
def run_calculate(
    gpu_name: str,
    model_name: str,
    batch_size: int,
    microbatch_size: int,
    network_bandwidth_gbps: float = 0.0,
    network_topology: str = "Single machine",
    optimization_strategy: str = "Full recomputation",
    tensor_par: int = 1,
    pipeline_par: int = 1,
    data_par: int = 1,
    num_procs: Optional[int] = None,
    datatype: str = "float16",
) -> str:
    """Run the full training-time simulator (HybridLlm / Calculon). Do not guess timings — call this for batch times."""
    prod = tensor_par * pipeline_par * data_par
    if num_procs is not None and num_procs != prod:
        return json.dumps(
            {
                "status": "error",
                "error": f"num_procs({num_procs}) must equal tensor_par*pipeline_par*data_par ({prod}).",
            },
            ensure_ascii=False,
        )
    dp_micro = data_par * microbatch_size
    if dp_micro < 1 or batch_size % dp_micro != 0:
        return json.dumps(
            {
                "status": "error",
                "error": (
                    f"batch_size({batch_size}) must be divisible by data_par*microbatch_size "
                    f"({data_par}*{microbatch_size}={dp_micro})."
                ),
            },
            ensure_ascii=False,
        )
    dtype = _normalize_datatype(datatype)
    try:
        gpu = _resolve_gpu(gpu_name).model_copy(update={"num_procs": prod})
        model = _resolve_model(model_name)
    except ValueError as e:
        return json.dumps({"status": "error", "error": str(e)}, ensure_ascii=False)
    network = Network(
        network_bandwidth=network_bandwidth_gbps,
        network_topology=network_topology,
    )
    trainning_config = TrainningConfig(
        optimization_strategy=optimization_strategy,
        tensor_par=tensor_par,
        pipeline_par=pipeline_par,
        data_par=data_par,
        batch_size=batch_size,
        microbatch_size=microbatch_size,
        datatype=dtype,
    )
    result = _facade.calculate(gpu, network, model, trainning_config)
    return _dump_result(result)


class RunOptimalInput(BaseModel):
    gpu_name: str = Field(description="Must match a GPU name from list_simulator_catalog.")
    model_name: str = Field(description="Must match a model name from list_simulator_catalog.")
    num_procs: int = Field(ge=1, description="Total GPU count for the job.")
    max_batch_size: int = Field(ge=1)
    network_bandwidth_gbps: float = Field(0.0)
    network_topology: str = Field("Single machine")
    datatype: str = Field("float16", description="float16, float32, or bfloat16 (fp16/fp32/bf16 aliases ok).")


@tool(args_schema=RunOptimalInput)
def run_optimal(
    gpu_name: str,
    model_name: str,
    num_procs: int,
    max_batch_size: int,
    network_bandwidth_gbps: float = 0.0,
    network_topology: str = "Single machine",
    datatype: str = "float16",
) -> str:
    """Search for a good parallel configuration (can be slow; uses multiprocessing). Returns JSON string."""
    dtype = _normalize_datatype(datatype)
    try:
        gpu = _resolve_gpu(gpu_name).model_copy(update={"num_procs": num_procs})
        model = _resolve_model(model_name)
    except ValueError as e:
        return json.dumps({"status": "error", "error": str(e)}, ensure_ascii=False)
    network = Network(
        network_bandwidth=network_bandwidth_gbps,
        network_topology=network_topology,
    )
    optimal_config = OptimalConfig(
        num_procs=num_procs,
        max_batch_size=max_batch_size,
        datatype=dtype,
    )
    result = _facade.optimal(gpu, network, model, optimal_config)
    return _dump_result(result)


class GetGpuDatatypesInput(BaseModel):
    gpu_name: str = Field(description="GPU name as in list_simulator_catalog / systems/*.json")


@tool(args_schema=GetGpuDatatypesInput)
def get_gpu_datatypes(gpu_name: str) -> str:
    """Return JSON listing datatypes supported by the GPU system profile (matrix/vector intersection). Mirrors GET /calculator/datatype."""
    return json.dumps(_gpu_datatypes_dict(gpu_name), ensure_ascii=False)


class ParseBenchmarkCsvInput(BaseModel):
    content: str = Field(description="Raw benchmark CSV text (lines), same format as POST /benchmark/upload.")


@tool(args_schema=ParseBenchmarkCsvInput)
def parse_benchmark_csv(content: str) -> str:
    """Parse benchmark CSV with iteration start/end blocks; returns JSON list of row tuples. Mirrors POST /benchmark/upload."""
    lines = content.strip().splitlines()
    br = BenchmarkRepository()
    data = br.read_benchmark_file(lines)
    return _dump_result(data, max_chars=60_000)


def build_simulator_tools() -> List:
    return [
        list_simulator_catalog,
        get_gpu_datatypes,
        run_calculate,
        run_optimal,
        parse_benchmark_csv,
    ]
