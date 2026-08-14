#!/usr/bin/env python3
"""Shared helpers for H20 efficiency calibration (DeepSeek-V3 / Calculon).

Calculon system JSON keys match System.TypeSizes / Processor datatypes:
  float8 | float16 | bfloat16 | float32

DeepSeek-V3 training on H20 typically uses FP8 Tensor-Core GEMM (matrix.float8)
with BF16-heavy vector math; default --dtype=float8 targets that matrix path.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

# Collapse indent=2 pair arrays onto one line: [gflops, eff] / [MB, eff].
_PAIR_ARRAY_RE = re.compile(
    r'\[\s*\n\s*(-?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][-+]?\d+)?),'
    r'\s*\n\s*(-?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][-+]?\d+)?)\s*\n\s*\]'
)


def write_system_json(path: str, cfg: Any) -> None:
    """Write system JSON with indent=2, but keep 2-number arrays on one line.

    Produces entries like ``[2.14, 0.34]`` instead of a 4-line expanded array.
    """
    text = json.dumps(cfg, indent=2)
    text = _PAIR_ARRAY_RE.sub(r'[\1, \2]', text)
    with open(path, 'w') as f:
        f.write(text)
        f.write('\n')

# CLI alias -> canonical calculon dtype name (JSON key under matrix/vector).
DTYPE_ALIASES: Dict[str, str] = {
    'float8': 'float8',
    'fp8': 'float8',
    'e4m3': 'float8',
    'float16': 'float16',
    'fp16': 'float16',
    'half': 'float16',
    'bfloat16': 'bfloat16',
    'bf16': 'bfloat16',
    'float32': 'float32',
    'fp32': 'float32',
}

# H20 Tensor-Core / CUDA peaks used as matrix defaults (TFLOPS).
# Vector peaks are usually --peak-tflops auto (mem-bound).
H20_MATRIX_PEAK_TFLOPS: Dict[str, float] = {
    'float8': 296.0,      # fp8_tc_flops
    'float16': 148.0,     # float16_tc_flops
    'bfloat16': 148.0,    # same TC peak class as FP16 on H20
    'float32': 74.0,
}

H20_MEM_PEAK_GBPS = 4022.0
H20_MEM_CAPACITY_GIB = 96.0

# mem2 = host / CPU DRAM via PCIe (offload path). Peak is unidirectional.
H20_MEM2_PEAK_GBPS = 64.0
H20_MEM2_CAPACITY_GIB = 512.0

# Network fabric peaks (GB/s, unidirectional) — systems/H20.json networks[].bandwidth.
# Flow simulator effective BW = bandwidth * efficiency.
H20_INTRA_PEAK_GBPS = 450.0   # NVLink / scale-up per GPU
H20_INTER_PEAK_GBPS = 25.0    # NIC / scale-out per GPU (25e9 B/s)

# Bytes per element (matches calculon System.TypeSizes).
DTYPE_NBYTES: Dict[str, int] = {
    'float8': 1,
    'float16': 2,
    'bfloat16': 2,
    'float32': 4,
}


def normalize_dtype(name: str) -> str:
    key = name.strip().lower()
    if key not in DTYPE_ALIASES:
        allowed = ', '.join(sorted(set(DTYPE_ALIASES.values())))
        raise SystemExit(f'Unsupported dtype={name!r}. Use one of: {allowed} '
                         f'(aliases: fp8, fp16, bf16, fp32, ...)')
    return DTYPE_ALIASES[key]


def torch_storage_dtype(dtype: str):
    """Torch dtype used to store operands for this calculon datatype."""
    return {
        'float8': torch.float8_e4m3fn,
        'float16': torch.float16,
        'bfloat16': torch.bfloat16,
        'float32': torch.float32,
    }[dtype]


def torch_compute_dtype(dtype: str):
    """Compute / output dtype (FP8 GEMM accumulates to BF16)."""
    if dtype == 'float8':
        return torch.bfloat16
    return torch_storage_dtype(dtype)


def add_dtype_arg(parser, default: str = 'float8') -> None:
    parser.add_argument(
        '--dtype', type=str, default=default,
        help='Precision for calibration / JSON key: float8|float16|bfloat16|float32 '
             '(aliases: fp8, fp16, bf16, fp32). Default float8 for DeepSeek-V3 FP8 training.',
    )


def dtype_banner(dtype: str, role: str) -> None:
    print(f'Target: DeepSeek-V3 on H20 | role={role} | dtype={dtype} '
          f'(storage={torch_storage_dtype(dtype)}, '
          f'compute={torch_compute_dtype(dtype)}, '
          f'{DTYPE_NBYTES[dtype]} byte/elem)')


def enforce_monotonic_efficiency(
    curve: List[List[float]], floor_eff: float,
) -> List[List[float]]:
    """Make gflops_efficiency lookup-safe.

    Processor scans descending gflops bins; as op size grows, efficiency must
    be non-decreasing. Equivalently, walking large→small gflops, eff must be
    non-increasing. Fixes Phase1 sawtooth / pred_f drops when m increases.
    """
    if not curve:
        return [[0, floor_eff]]
    ordered = sorted(
        ([float(g), float(e)] for g, e in curve if float(g) > 0),
        key=lambda p: -p[0],
    )
    out: List[List[float]] = []
    last_eff = 1.0
    for g, e in ordered:
        e = min(max(floor_eff, e), last_eff, 1.0)
        out.append([g, e])
        last_eff = e
    out.append([0.0, min(floor_eff, last_eff)])
    return out


def estimate_launch_s(
    latencies_s: Sequence[float],
    gflops_list: Sequence[float],
    max_gflops: float = 1.0,
    min_samples: int = 3,
) -> Optional[float]:
    """Estimate kernel launch floor from tiny GEMM latencies (seconds)."""
    tiny = [lat for lat, g in zip(latencies_s, gflops_list) if g <= max_gflops]
    if len(tiny) < min_samples:
        # Fall back to the smallest few latencies overall.
        tiny = sorted(latencies_s)[:max(min_samples, min(5, len(latencies_s)))]
    if not tiny:
        return None
    tiny = sorted(tiny)
    mid = len(tiny) // 2
    return float(tiny[mid])


def merge_efficiency_bins(
    points: Sequence[Tuple[float, float]],
    rel_tol: float = 0.08,
) -> List[List[float]]:
    """Bin nearby gflops keys; keep max eff in each bin (pre-monotonic)."""
    if not points:
        return []
    ordered = sorted(((float(g), float(e)) for g, e in points), key=lambda p: -p[0])
    bins: List[List[float]] = []
    for g, e in ordered:
        if bins and abs(bins[-1][0] - g) <= rel_tol * max(bins[-1][0], g, 1e-12):
            bins[-1][1] = max(bins[-1][1], e)
            # Representative gflops: keep the larger key (first in desc order).
        else:
            bins.append([g, e])
    return bins
