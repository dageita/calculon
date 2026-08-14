"""Shared helpers for BW1100 operator calibration.

The scripts deliberately derive the device ISA at runtime.  BW1100 machines in
the field expose gfx938, so hard-coding the similarly named gfx936 library
directory would create binaries that cannot execute on this accelerator.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import torch

DTYPE_NBYTES = {"float8": 1, "int8": 1, "float16": 2, "bfloat16": 2, "float32": 4}
SYSTEM_JSON = Path(__file__).resolve().parents[2] / "systems" / "BW1100.json"


def require_bw1100() -> str:
    """Verify the DTK/HIP runtime and return the actual GPU ISA."""
    if not torch.cuda.is_available():
        raise RuntimeError(
            "No BW1100 HIP device is visible. Start the container with "
            "--device=/dev/kfd --device=/dev/dri --group-add video "
            "-v /opt/hyhal:/opt/hyhal:ro."
        )
    props = torch.cuda.get_device_properties(0)
    isa = getattr(props, "gcnArchName", "").split(":", 1)[0]
    if isa != "gfx938":
        raise RuntimeError(f"Expected BW1100 gfx938, found {props.name!r} ({isa or 'unknown ISA'}).")
    return isa


def merge_efficiency_bins(points: Iterable[tuple[float, float]], rel_tol: float = 0.08) -> list[list[float]]:
    """Merge near-identical work-size bins, retaining the fastest result."""
    merged: list[list[float]] = []
    for gflops, efficiency in sorted(points, reverse=True):
        if merged and abs(merged[-1][0] - gflops) <= rel_tol * max(merged[-1][0], gflops, 1e-12):
            merged[-1][1] = max(merged[-1][1], float(efficiency))
        else:
            merged.append([float(gflops), float(efficiency)])
    return merged


def enforce_monotonic_efficiency(points: Iterable[tuple[float, float]], floor_eff: float) -> list[list[float]]:
    """Make a descending GFLOP/s efficiency curve safe for Calculon."""
    out: list[list[float]] = []
    ceiling = 1.0
    for gflops, efficiency in sorted(points, reverse=True):
        ceiling = min(ceiling, min(1.0, max(floor_eff, float(efficiency))))
        out.append([round(float(gflops), 9), round(ceiling, 6)])
    out.append([0.0, min(floor_eff, out[-1][1]) if out else floor_eff])
    return out


def write_curve(path: Path, curve: list[list[float]], metadata: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"metadata": metadata, "curve": curve}, indent=2) + "\n", encoding="utf-8")


def dtype_from_name(name: str) -> torch.dtype:
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[name]


def configured_peak(section: str, dtype: str, system_json: Path = SYSTEM_JSON) -> float:
    """Read the physical peak already configured for this BW1100 system."""
    try:
        peak = json.loads(system_json.read_text(encoding="utf-8"))[section][dtype]["tflops"]
    except (FileNotFoundError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Cannot read {section}.{dtype}.tflops from {system_json}") from exc
    if not isinstance(peak, (int, float)) or peak <= 0:
        raise RuntimeError(f"Invalid {section}.{dtype}.tflops in {system_json}: {peak!r}")
    return float(peak)


def update_system_curve(system_json: Path, section: str, dtype: str, peak: float, curve: list[list[float]]) -> None:
    """Atomically replace only one Calculon efficiency curve."""
    document = json.loads(system_json.read_text(encoding="utf-8"))
    target = document.setdefault(section, {}).setdefault(dtype, {})
    target["tflops"] = peak
    target["gflops_efficiency"] = curve
    temporary = system_json.with_suffix(system_json.suffix + ".tmp")
    temporary.write_text(_format_system_json(document) + "\n", encoding="utf-8")
    temporary.replace(system_json)


def update_vector_launch_floor(system_json: Path, latency_s: float) -> None:
    """Update the vector-kernel launch floor without touching efficiency data."""
    if latency_s <= 0:
        raise ValueError("vector launch floor must be positive")
    document = json.loads(system_json.read_text(encoding="utf-8"))
    document["vector_launch_s"] = float(latency_s)
    temporary = system_json.with_suffix(system_json.suffix + ".tmp")
    temporary.write_text(_format_system_json(document) + "\n", encoding="utf-8")
    temporary.replace(system_json)


def update_linear_shape_model(system_json: Path, dtype: str, reference_k: int,
                              samples: Iterable[tuple[tuple[int, int, int], float, float, bool]]) -> bool:
    """Store measured K=4096 Linear latency floors outside generic curves.

    Calculon's generic matrix curve is indexed only by FLOPs, while BW1100
    GEMM latency also changes materially with output width N.  This model
    records only exact, explicitly calibrated N buckets, so it improves common
    Linear projections without extrapolating to unmeasured aspect ratios.
    """
    buckets: dict[str, list[list[float | int]]] = {}
    for (m, n, k), latency, _tflops, contributes_to_curve in samples:
        if k != reference_k or n not in (16, 64, 256, 1024, 4096):
            continue
        buckets.setdefault(str(n), []).append([int(m), round(float(latency), 12)])
    if not buckets:
        return False
    for points in buckets.values():
        points.sort(key=lambda point: point[0], reverse=True)
    document = json.loads(system_json.read_text(encoding="utf-8"))
    model = document.setdefault("linear_shape", {})
    model["reference_k"] = int(reference_k)
    model["exact_n_buckets"] = sorted(int(bucket) for bucket in buckets)
    model.setdefault("latency_s", {})[dtype] = buckets
    temporary = system_json.with_suffix(system_json.suffix + ".tmp")
    temporary.write_text(_format_system_json(document) + "\n", encoding="utf-8")
    temporary.replace(system_json)
    return True


def update_memory_curve(system_json: Path, section: str, capacity_gib: float, peak_gbps: float,
                        curve: list[list[float]]) -> None:
    """Atomically replace one Calculon memory-tier calibration."""
    document = json.loads(system_json.read_text(encoding="utf-8"))
    target = document.setdefault(section, {})
    target["GiB"] = capacity_gib
    target["GBps"] = peak_gbps
    target["MB_efficiency"] = curve
    temporary = system_json.with_suffix(system_json.suffix + ".tmp")
    temporary.write_text(_format_system_json(document) + "\n", encoding="utf-8")
    temporary.replace(system_json)


def _format_system_json(value: object, level: int = 0, compact_points: bool = False) -> str:
    """Pretty-print system JSON while keeping numerical lookup points on one line."""
    indent = "  " * level
    child_indent = "  " * (level + 1)
    if isinstance(value, dict):
        if not value:
            return "{}"
        fields = []
        for key, item in value.items():
            rendered = _format_system_json(
                item, level + 1,
                compact_points or key in ("gflops_efficiency", "MB_efficiency", "latency_s"),
            )
            fields.append(f"{child_indent}{json.dumps(key)}: {rendered}")
        return "{\n" + ",\n".join(fields) + f"\n{indent}}}"
    if isinstance(value, list):
        if compact_points and all(isinstance(item, (int, float)) for item in value):
            return json.dumps(value, ensure_ascii=False, separators=(", ", ": "))
        if not value:
            return "[]"
        return "[\n" + ",\n".join(
            f"{child_indent}{_format_system_json(item, level + 1, compact_points)}" for item in value
        ) + f"\n{indent}]"
    return json.dumps(value, ensure_ascii=False)
