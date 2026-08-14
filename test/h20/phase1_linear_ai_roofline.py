#!/usr/bin/env python3
"""Phase1: validate Calculon roofline max() on calibrated H20 via Linear AI scan.

Goal (H3 / Exp-1 + E4): after Phase0 efficiency calibration, check whether
    time = max(flops/T_eff, bytes/BW_eff)
matches isolated GEMM latency across memory-bound → ridge → compute-bound.

Uses only parsable Linear/GEMM (no MoE block, no .so / communication).

For each (m, k, n) = (batch_seq, c_in, c_out):
  AI = flops / bytes = 2·m·k·n / ((m·k + k·n + m·n)·bpe)
  pred_f, pred_m from Calculon Linear.compute_*_time("fw")
  pred_max = max(pred_f, pred_m)   # roofline
  pred_sum = pred_f + pred_m       # no_overlap ablation
  meas     = CUDA-event GEMM latency

Judgement:
  - compute-bound region: meas ≈ pred_f
  - memory-bound region:  meas ≈ pred_m
  - full sweep: MAPE(max) < MAPE(sum) → keep processing_mode=roofline

Example:
  python test/phase1_linear_ai_roofline.py
  python test/phase1_linear_ai_roofline.py --dtype float8 --min-ms 500 \\
      --csv test/phase1_linear_ai.csv
  python test/phase1_linear_ai_roofline.py --predict-only   # no CUDA meas
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from dataclasses import asdict, dataclass
from typing import List, Optional, Sequence, Tuple

_TEST_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_TEST_DIR)
if _TEST_DIR not in sys.path:
    sys.path.insert(0, _TEST_DIR)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from calibrate_h20_common import (  # noqa: E402
    DTYPE_NBYTES,
    add_dtype_arg,
    normalize_dtype,
)
from calibrate_h20_matrix_efficiency import (  # noqa: E402
    _align,
    benchmark_shape,
)

from calculon.system import System  # noqa: E402
from calculon.llm.layers import Linear  # noqa: E402

# DeepSeek-V3-oriented Linear families (k=c_in, n=c_out). No full MoE.
# m = microbatch * seq; default sweep crosses ridge on H20 (~4 TB/s HBM).
DEFAULT_M_SWEEP: List[int] = [
    64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384,
]

# (name, k, n, m_list_or_None)  — None → use DEFAULT_M_SWEEP
SHAPE_FAMILIES: List[Tuple[str, int, int, Optional[List[int]]]] = [
    # High AI — dense SwiGLU w1 (compute-bound as m grows)
    ('dense_w1', 7168, 18432, None),
    # Mid / transition — MoE expert w1 (single expert GEMM, not full MoE)
    ('expert_w1', 7168, 2048, None),
    # Low AI — WKR-class skinny projection
    ('wkr', 7168, 64, None),
    # Low AI — Router
    ('router', 7168, 256, None),
    # Extreme skinny: on H20+FP8, bpe=1 pushes ridge high; this family
    # is sized to populate the memory-bound side of local ridge_ai.
    ('skinny_ext', 7168, 16, [2048, 4096, 8192, 16384, 32768, 65536]),
    # Fixed DS-V3 reference points at m=4096 (S=4096, mbs=1)
    ('ref_wdq', 7168, 1536, [4096]),
    ('ref_dense_w2', 18432, 7168, [4096]),
    ('ref_expert_w2', 2048, 7168, [4096]),
]


@dataclass
class Row:
    family: str
    m: int
    k: int
    n: int
    dtype: str
    flops: float
    bytes: float
    ai: float
    gflops: float
    pred_f_s: float
    pred_m_s: float
    pred_max_s: float
    pred_sum_s: float
    bound: str
    ridge_ai: float
    meas_s: Optional[float]
    err_max_pct: Optional[float]
    err_sum_pct: Optional[float]
    err_f_pct: Optional[float]
    err_m_pct: Optional[float]
    meas_tflops: Optional[float]
    skipped: str


def _rel_err_pct(pred: float, meas: float) -> float:
    if meas <= 0:
        return float('nan')
    return 100.0 * (pred - meas) / meas


def _mape(errs: Sequence[float]) -> Optional[float]:
    vals = [abs(e) for e in errs if e is not None and not math.isnan(e)]
    if not vals:
        return None
    return sum(vals) / len(vals)


def load_system(path: str) -> System:
    with open(path) as f:
        cfg = json.load(f)
    if cfg.get('processing_mode') != 'roofline':
        print(f'WARNING: {path} processing_mode={cfg.get("processing_mode")!r}; '
              f'forcing analysis as roofline max() vs sum ablation.',
              file=sys.stderr)
    return System(cfg)


def make_linear(sys: System, m: int, k: int, n: int, dtype: str) -> Linear:
    """Linear(batch_seq=m, c_in=k, c_out=n); GEMM A[m,k] @ W[k,n]."""
    sys.set_datatypes(dtype, dtype)
    layer = Linear('phase1_linear', sys, batch_seq=m, c_in=k, c_out=n)
    layer.set_bytes_per_element(System.TypeSizes[dtype])
    return layer


def predict(sys: System, m: int, k: int, n: int, dtype: str
            ) -> Tuple[float, float, float, float, float, float, float, float]:
    """Return flops, bytes, ai, pred_f, pred_m, pred_max, pred_sum, ridge_ai."""
    layer = make_linear(sys, m, k, n, dtype)
    flops = layer.get_fw_flops()
    nbytes = layer.get_fw_mem_accessed()
    ai = flops / nbytes if nbytes > 0 else float('inf')
    pred_f = layer.compute_flops_time('fw')
    pred_m = layer.compute_mem_time('fw')
    pred_max = max(pred_f, pred_m)
    pred_sum = pred_f + pred_m
    # Local ridge using size-dependent T_eff / BW_eff implied by times.
    t_eff = flops / pred_f if pred_f > 0 else float('inf')
    bw_eff = nbytes / pred_m if pred_m > 0 else float('inf')
    ridge_ai = t_eff / bw_eff if bw_eff > 0 else float('inf')
    return flops, nbytes, ai, pred_f, pred_m, pred_max, pred_sum, ridge_ai


def iter_shapes(
    families: Sequence[Tuple[str, int, int, Optional[List[int]]]],
    m_sweep: Sequence[int],
) -> List[Tuple[str, int, int, int]]:
    out = []
    for name, k, n, ms in families:
        for m in (ms if ms is not None else m_sweep):
            out.append((name, int(m), int(k), int(n)))
    return out


def run_case(
    sys: System,
    family: str,
    m: int,
    k: int,
    n: int,
    dtype: str,
    measure: bool,
    warmup: int,
    iters: int,
    min_ms: float,
) -> Row:
    m_a, n_a, k_a = _align(m), _align(n), _align(k)
    flops, nbytes, ai, pred_f, pred_m, pred_max, pred_sum, ridge_ai = predict(
        sys, m_a, k_a, n_a, dtype)
    bound = 'compute' if pred_f >= pred_m else 'memory'
    gflops = flops / 1e9

    meas_s = err_max = err_sum = err_f = err_m = meas_tflops = None
    skipped = ''
    if measure:
        import torch
        try:
            _, tflops, lat, _ = benchmark_shape(
                m_a, n_a, k_a, dtype, warmup, iters, min_ms)
            meas_s = lat
            meas_tflops = tflops
            err_max = _rel_err_pct(pred_max, meas_s)
            err_sum = _rel_err_pct(pred_sum, meas_s)
            err_f = _rel_err_pct(pred_f, meas_s)
            err_m = _rel_err_pct(pred_m, meas_s)
        except (torch.cuda.OutOfMemoryError, RuntimeError, MemoryError) as e:
            torch.cuda.empty_cache()
            skipped = f'{type(e).__name__}: {e}'[:160]
    else:
        skipped = 'predict-only'

    return Row(
        family=family, m=m_a, k=k_a, n=n_a, dtype=dtype,
        flops=flops, bytes=nbytes, ai=ai, gflops=gflops,
        pred_f_s=pred_f, pred_m_s=pred_m, pred_max_s=pred_max,
        pred_sum_s=pred_sum, bound=bound, ridge_ai=ridge_ai,
        meas_s=meas_s, err_max_pct=err_max, err_sum_pct=err_sum,
        err_f_pct=err_f, err_m_pct=err_m, meas_tflops=meas_tflops,
        skipped=skipped,
    )


def write_csv(path: str, rows: Sequence[Row]) -> None:
    if not rows:
        return
    fieldnames = list(asdict(rows[0]).keys())
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(asdict(r))
    print(f'\nWrote CSV: {path}')


def print_table(rows: Sequence[Row]) -> None:
    hdr = (f'{"family":14s} {"m":>6} {"k":>6} {"n":>6} {"AI":>8} '
           f'{"bound":8s} {"pred_f_us":>10} {"pred_m_us":>10} '
           f'{"pred_max":>10} {"meas_us":>10} {"err_max%":>9} {"err_sum%":>9}')
    print(hdr)
    print('-' * len(hdr))
    for r in rows:
        meas_us = f'{r.meas_s*1e6:.1f}' if r.meas_s is not None else '—'
        emax = f'{r.err_max_pct:+.1f}' if r.err_max_pct is not None else '—'
        esum = f'{r.err_sum_pct:+.1f}' if r.err_sum_pct is not None else '—'
        flag = f'  SKIP {r.skipped}' if r.skipped and r.meas_s is None else ''
        print(
            f'{r.family:14s} {r.m:6d} {r.k:6d} {r.n:6d} {r.ai:8.2f} '
            f'{r.bound:8s} {r.pred_f_s*1e6:10.1f} {r.pred_m_s*1e6:10.1f} '
            f'{r.pred_max_s*1e6:10.1f} {meas_us:>10s} {emax:>9s} {esum:>9s}'
            f'{flag}'
        )


def summarize(rows: Sequence[Row]) -> None:
    measured = [r for r in rows if r.meas_s is not None]
    print('\n=== Phase1 summary (H3 / roofline max vs sum) ===')
    if not measured:
        n_comp = sum(1 for r in rows if r.bound == 'compute')
        n_mem = sum(1 for r in rows if r.bound == 'memory')
        print(f'predict-only: {len(rows)} shapes '
              f'(bound compute={n_comp}, memory={n_mem})')
        print('Re-run without --predict-only on H20 to get meas / MAPE.')
        return

    mape_max = _mape([r.err_max_pct for r in measured])
    mape_sum = _mape([r.err_sum_pct for r in measured])
    print(f'N measured: {len(measured)} / {len(rows)}')
    print(f'MAPE(pred_max vs meas): {mape_max:.2f}%' if mape_max is not None
          else 'MAPE(max): n/a')
    print(f'MAPE(pred_sum vs meas): {mape_sum:.2f}%' if mape_sum is not None
          else 'MAPE(sum): n/a')
    if mape_max is not None and mape_sum is not None:
        if mape_max < mape_sum:
            print('→ MAPE(max) < MAPE(sum): keep processing_mode=roofline')
        else:
            print('→ MAPE(sum) <= MAPE(max): consider no_overlap or partial overlap')

    for region in ('compute', 'memory'):
        sub = [r for r in measured if r.bound == region]
        if not sub:
            hint = ''
            if region == 'memory':
                hint = ('  (try --families skinny_ext wkr, larger --m-sweep, '
                        'or --dtype float16; FP8 bpe=1 raises AI)')
            print(f'[{region}] no points{hint}')
            continue
        if region == 'compute':
            mape_dom = _mape([r.err_f_pct for r in sub])
            label = 'MAPE(pred_f)'
        else:
            mape_dom = _mape([r.err_m_pct for r in sub])
            label = 'MAPE(pred_m)'
        mape_r = _mape([r.err_max_pct for r in sub])
        print(f'[{region}] N={len(sub)}  MAPE(max)={mape_r:.2f}%  '
              f'{label}={mape_dom:.2f}%' if mape_r is not None and mape_dom is not None
              else f'[{region}] N={len(sub)}')

    # Per-family quick view
    print('\nPer-family MAPE(max):')
    fams = sorted({r.family for r in measured})
    for fam in fams:
        sub = [r for r in measured if r.family == fam]
        m = _mape([r.err_max_pct for r in sub])
        bounds = ','.join(sorted({r.bound[0] for r in sub}))
        print(f'  {fam:14s}  N={len(sub):2d}  MAPE={m:6.2f}%  bounds={bounds}'
              if m is not None else f'  {fam}: n/a')


def rows_from_csv(path: str) -> List[Row]:
    """Reload Phase1 CSV for --replot-csv (no GPU needed)."""
    def _f(v: str, cast=float):
        if v is None or v == '':
            return None
        return cast(v)

    rows: List[Row] = []
    with open(path, newline='') as f:
        for d in csv.DictReader(f):
            rows.append(Row(
                family=d['family'],
                m=int(float(d['m'])),
                k=int(float(d['k'])),
                n=int(float(d['n'])),
                dtype=d['dtype'],
                flops=float(d['flops']),
                bytes=float(d['bytes']),
                ai=float(d['ai']),
                gflops=float(d['gflops']),
                pred_f_s=float(d['pred_f_s']),
                pred_m_s=float(d['pred_m_s']),
                pred_max_s=float(d['pred_max_s']),
                pred_sum_s=float(d['pred_sum_s']),
                bound=d['bound'],
                ridge_ai=float(d['ridge_ai']),
                meas_s=_f(d.get('meas_s')),
                err_max_pct=_f(d.get('err_max_pct')),
                err_sum_pct=_f(d.get('err_sum_pct')),
                err_f_pct=_f(d.get('err_f_pct')),
                err_m_pct=_f(d.get('err_m_pct')),
                meas_tflops=_f(d.get('meas_tflops')),
                skipped=d.get('skipped') or '',
            ))
    return rows


def _plot_svg(rows: Sequence[Row], path: str) -> None:
    """Dependency-free SVG fallback (log-x AI vs TFLOPS)."""
    measured = [r for r in rows if r.meas_s is not None and r.ai > 0]
    w, h, pad = 960, 560, 70
    xs_all, ys_all = [], []
    series = []
    colors = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
        '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
    ]
    for i, fam in enumerate(sorted({r.family for r in measured})):
        sub = sorted([r for r in measured if r.family == fam], key=lambda r: r.ai)
        xs = [r.ai for r in sub]
        y_m = [r.flops / r.meas_s / 1e12 for r in sub]
        y_p = [r.flops / r.pred_max_s / 1e12 for r in sub]
        series.append((fam, xs, y_m, y_p, colors[i % len(colors)]))
        xs_all.extend(xs)
        ys_all.extend(y_m + y_p)
    if not xs_all:
        raise ValueError('no points')
    xmin, xmax = min(xs_all), max(xs_all)
    ymin, ymax = 0.0, max(ys_all) * 1.1
    if xmin <= 0:
        xmin = min(x for x in xs_all if x > 0)
    lx0, lx1 = math.log10(xmin), math.log10(xmax)
    if lx1 <= lx0:
        lx1 = lx0 + 1.0

    def sx(x: float) -> float:
        return pad + (math.log10(x) - lx0) / (lx1 - lx0) * (w - 2 * pad)

    def sy(y: float) -> float:
        return h - pad - (y - ymin) / (ymax - ymin + 1e-12) * (h - 2 * pad)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{w/2}" y="28" text-anchor="middle" font-size="16" '
        f'font-family="sans-serif">Phase1 Linear AI scan — H20 roofline</text>',
        f'<text x="{w/2}" y="{h-18}" text-anchor="middle" font-size="12" '
        f'font-family="sans-serif">Arithmetic intensity (FLOPs/byte, log)</text>',
        f'<text x="18" y="{h/2}" text-anchor="middle" font-size="12" '
        f'font-family="sans-serif" transform="rotate(-90 18 {h/2})">'
        f'Achieved TFLOPS</text>',
        f'<rect x="{pad}" y="{pad}" width="{w-2*pad}" height="{h-2*pad}" '
        f'fill="none" stroke="#333"/>',
    ]
    # grid
    for t in range(0, 6):
        yv = ymin + (ymax - ymin) * t / 5
        y = sy(yv)
        parts.append(
            f'<line x1="{pad}" y1="{y:.1f}" x2="{w-pad}" y2="{y:.1f}" '
            f'stroke="#ddd"/>'
            f'<text x="{pad-8}" y="{y+4:.1f}" text-anchor="end" '
            f'font-size="10" font-family="sans-serif">{yv:.1f}</text>'
        )
    legend_y = pad + 12
    for fam, xs, y_m, y_p, col in series:
        if len(xs) >= 2:
            pts_m = ' '.join(f'{sx(x):.1f},{sy(y):.1f}' for x, y in zip(xs, y_m))
            pts_p = ' '.join(f'{sx(x):.1f},{sy(y):.1f}' for x, y in zip(xs, y_p))
            parts.append(
                f'<polyline fill="none" stroke="{col}" stroke-width="2" '
                f'points="{pts_m}"/>'
            )
            parts.append(
                f'<polyline fill="none" stroke="{col}" stroke-width="1.5" '
                f'stroke-dasharray="6,4" points="{pts_p}"/>'
            )
        for x, y in zip(xs, y_m):
            parts.append(
                f'<circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="3.5" fill="{col}"/>'
            )
        parts.append(
            f'<text x="{w-pad-10}" y="{legend_y}" text-anchor="end" '
            f'font-size="11" font-family="sans-serif" fill="{col}">'
            f'{fam} solid=meas dashed=pred</text>'
        )
        legend_y += 14
    parts.append('</svg>')

    # If user asked for .png but matplotlib missing, write .svg alongside.
    svg_path = path
    if path.lower().endswith('.png'):
        svg_path = path[:-4] + '.svg'
    with open(svg_path, 'w') as f:
        f.write('\n'.join(parts))
    print(f'Wrote plot (SVG fallback): {os.path.abspath(svg_path)}', flush=True)


def try_plot(rows: Sequence[Row], path: str) -> None:
    measured = [r for r in rows if r.meas_s is not None and r.ai > 0]
    if not measured:
        print('No measured points to plot.', flush=True)
        return

    out_dir = os.path.dirname(os.path.abspath(path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print('matplotlib not installed; using SVG fallback '
              '(pip install matplotlib for PNG).', flush=True)
        _plot_svg(rows, path)
        return

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for fam in sorted({r.family for r in measured}):
        sub = sorted([r for r in measured if r.family == fam], key=lambda r: r.ai)
        xs = [r.ai for r in sub]
        y_meas = [r.flops / r.meas_s / 1e12 for r in sub]
        y_pred = [r.flops / r.pred_max_s / 1e12 for r in sub]
        if len(sub) >= 2:
            ax.plot(xs, y_meas, 'o-', label=f'{fam} meas', alpha=0.85)
            ax.plot(xs, y_pred, '--', label=f'{fam} pred_max', alpha=0.75)
        else:
            ax.scatter(xs, y_meas, marker='o', label=f'{fam} meas', zorder=3)
            ax.scatter(xs, y_pred, marker='x', label=f'{fam} pred_max', zorder=3)
    ax.set_xscale('log')
    ax.set_xlabel('Arithmetic intensity (FLOPs/byte)')
    ax.set_ylabel('Achieved TFLOPS')
    ax.set_title('Phase1 Linear AI scan — H20 roofline')
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, which='both', alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f'Wrote plot: {os.path.abspath(path)}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_dtype_arg(parser, default='float8')
    parser.add_argument(
        '--system-json', type=str,
        default=os.path.join(_ROOT, 'systems', 'H20.json'),
        help='Calibrated system JSON (default systems/H20.json)',
    )
    parser.add_argument('--m-sweep', type=int, nargs='+', default=None,
                        help='Override m values for family sweeps')
    parser.add_argument('--families', type=str, nargs='+', default=None,
                        help='Subset of family names (default: all)')
    parser.add_argument('--warmup', type=int, default=20)
    parser.add_argument('--iters', type=int, default=50)
    parser.add_argument('--min-ms', type=float, default=500.0)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--predict-only', action='store_true',
                        help='Skip CUDA measurement; print Calculon preds only')
    parser.add_argument('--csv', type=str, default=None,
                        help='Write results CSV path')
    parser.add_argument('--plot', type=str, default=None,
                        help='PNG path (requires: pip install matplotlib)')
    parser.add_argument(
        '--replot-csv', type=str, default=None,
        help='Only load an existing Phase1 CSV and write --plot (no GPU)',
    )
    args = parser.parse_args()

    if args.replot_csv:
        if not args.plot:
            raise SystemExit('--replot-csv requires --plot <path.png>')
        rows = rows_from_csv(args.replot_csv)
        print(f'Replot from {args.replot_csv}: {len(rows)} rows')
        print_table(rows)
        summarize(rows)
        try_plot(rows, args.plot)
        return

    dtype = normalize_dtype(args.dtype)
    m_sweep = args.m_sweep or DEFAULT_M_SWEEP
    families = SHAPE_FAMILIES
    if args.families:
        want = set(args.families)
        families = [f for f in SHAPE_FAMILIES if f[0] in want]
        missing = want - {f[0] for f in families}
        if missing:
            raise SystemExit(f'Unknown families: {sorted(missing)}. '
                             f'Known: {[f[0] for f in SHAPE_FAMILIES]}')

    sys_obj = load_system(args.system_json)
    print(f'System: {args.system_json}')
    print(f'processing_mode={sys_obj.proc_mode}  dtype={dtype}  '
          f'bpe={DTYPE_NBYTES[dtype]}')
    print(f'matrix peak={sys_obj.matrix.flops(dtype)/1e12:.1f} TF  '
          f'mem1={sys_obj.mem1.bandwidth/1e9:.0f} GB/s')
    launch_us = (sys_obj.matrix_launch_s or 0.0) * 1e6
    print(f'matrix_launch_s={launch_us:.2f} us'
          + ('  (disabled)' if launch_us <= 0 else ''))
    print('Phase1 Linear AI scan (no MoE / no comm)\n')

    measure = not args.predict_only
    if measure:
        import torch
        if not torch.cuda.is_available():
            print('CUDA required for measurement; use --predict-only',
                  file=sys.stderr)
            sys.exit(1)
        torch.cuda.set_device(args.device)
        print(f'Device: {torch.cuda.get_device_name(args.device)}\n')

    shapes = iter_shapes(families, m_sweep)
    rows: List[Row] = []
    for family, m, k, n in shapes:
        # Rough footprint gate before allocating
        bpe = DTYPE_NBYTES[dtype]
        est_gib = (m * k + k * n + m * n) * bpe / (1024 ** 3)
        print(f'>> {family}  m={m} k={k} n={n}  ~{est_gib:.2f} GiB tensors',
              flush=True)
        row = run_case(
            sys_obj, family, m, k, n, dtype, measure,
            args.warmup, args.iters, args.min_ms)
        rows.append(row)
        if measure:
            import torch
            torch.cuda.empty_cache()

    print()
    print_table(rows)
    summarize(rows)

    csv_path = args.csv
    if csv_path is None and measure:
        csv_path = os.path.join(_TEST_DIR, f'phase1_linear_ai_{dtype}.csv')
    if csv_path:
        write_csv(csv_path, rows)
    if args.plot:
        try_plot(rows, args.plot)


if __name__ == '__main__':
    main()
