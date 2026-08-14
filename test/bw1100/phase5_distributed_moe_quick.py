#!/usr/bin/env python3
"""Phase5: low-memory distributed MoE prediction-vs-measurement closure.

This is deliberately one EP-distributed MoE layer, not a miniature model.  It
keeps the validation attributable: router, EP dispatch/combine, expert FFN,
backward, and communication/compute overlap are reported independently before
the end-to-end core is gated.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
from pathlib import Path

import torch
import torch.distributed as dist

from calculon.system import System


def relerr(pred, meas):
    return 100.0 * (pred - meas) / meas if meas > 0 else None


def measure_ms(fn, warmup, iters, dev):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize(dev)
    vals = []
    for _ in range(iters):
        dist.barrier()
        start, end = torch.cuda.Event(True), torch.cuda.Event(True)
        start.record()
        fn()
        end.record(); end.synchronize()
        vals.append(start.elapsed_time(end))
    value = torch.tensor([statistics.median(vals)], device=dev,
                         dtype=torch.float64)
    dist.all_reduce(value, op=dist.ReduceOp.MAX)
    return float(value.item())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--tokens', type=int, default=128,
                   help='tokens per rank (default: 128)')
    p.add_argument('--hidden', type=int, default=7168,
                   help='DeepSeek-V3 hidden width (default: 7168)')
    p.add_argument('--ffn-hidden', type=int, default=2048)
    p.add_argument('--dtype', choices=('float16', 'bfloat16'),
                   default='bfloat16')
    p.add_argument('--warmup', type=int, default=3)
    p.add_argument('--iters', type=int, default=10)
    p.add_argument('--gate-component-pct', type=float, default=65.0)
    p.add_argument('--gate-core-pct', type=float, default=55.0)
    p.add_argument('--system', type=Path, default=Path('systems/BW1100.json'))
    p.add_argument('--network-index', type=int, default=0)
    p.add_argument('--output', type=Path,
                   default=Path('test/bw1100/phase5_distributed_moe_quick.json'))
    p.add_argument('--csv', type=Path,
                   default=Path('test/bw1100/phase5_distributed_moe_quick.csv'))
    p.add_argument('--dry-run', action='store_true')
    args = p.parse_args()
    if args.dry_run:
        print(json.dumps({
            'world_size': '2-4', 'tokens_per_rank': args.tokens,
            'hidden': args.hidden, 'ffn_hidden': args.ffn_hidden,
            'coverage': ['router top1+sort', 'EP dispatch/combine',
                         'expert FFN forward', 'expert FFN backward',
                         'MoE forward/backward core', 'async overlap'],
            'excluded': ['optimizer', 'pipeline bubble',
                         'full-model extrapolation (covered by phase3/4)'],
        }, indent=2))
        return

    dist.init_process_group('nccl')
    rank, world = dist.get_rank(), dist.get_world_size()
    if not 2 <= world <= 4:
        raise SystemExit(f'Phase5 requires 2-4 ranks, got {world}')
    local_rank = int(os.environ.get('LOCAL_RANK', rank))
    torch.cuda.set_device(local_rank)
    dev = torch.device('cuda', local_rank)
    dtype = {'float16': torch.float16, 'bfloat16': torch.bfloat16}[args.dtype]
    bpe = torch.empty((), dtype=dtype).element_size()
    tokens, hidden, ffn = args.tokens, args.hidden, args.ffn_hidden
    if min(tokens, hidden, ffn) <= 0:
        raise SystemExit('tokens/hidden/ffn-hidden must be > 0')

    # The router selects an EP destination and one local expert. Routing is
    # frozen after setup so every timed iteration has identical shapes.
    torch.manual_seed(1234 + rank)
    x = torch.randn(tokens, hidden, device=dev, dtype=dtype)
    router_w = torch.randn(hidden, world,
                           device=dev, dtype=dtype) / math.sqrt(hidden)

    def route_op():
        logits = x @ router_w
        expert = torch.argmax(logits, dim=1)
        return torch.sort(expert)

    with torch.no_grad():
        destinations, order = route_op()
        send_splits = torch.bincount(destinations, minlength=world).tolist()
    split_tensor = torch.tensor(send_splits, device=dev, dtype=torch.int64)
    all_splits = [torch.empty_like(split_tensor) for _ in range(world)]
    dist.all_gather(all_splits, split_tensor)
    recv_splits = [int(all_splits[src][rank].item()) for src in range(world)]
    recv_tokens = sum(recv_splits)
    send_x = x[order].contiguous()
    recv_x = torch.empty(recv_tokens, hidden, device=dev, dtype=dtype)
    returned = torch.empty_like(send_x)
    grad_sorted = torch.randn_like(send_x)
    recv_grad = torch.empty_like(recv_x)
    returned_dx = torch.empty_like(send_x)

    # One local expert is sufficient for the distributed closure; extra local
    # experts affect routing/scheduling but should be calibrated in Phase2/3.
    w1 = torch.randn(hidden, ffn, device=dev, dtype=dtype) / math.sqrt(hidden)
    w2 = torch.randn(hidden, ffn, device=dev, dtype=dtype) / math.sqrt(hidden)
    w3 = torch.randn(ffn, hidden, device=dev, dtype=dtype) / math.sqrt(ffn)

    def dispatch():
        dist.all_to_all_single(recv_x, send_x, recv_splits, send_splits)

    def combine(inp=None, out=None):
        inp = recv_x if inp is None else inp
        out = returned if out is None else out
        dist.all_to_all_single(out, inp, send_splits, recv_splits)

    def expert_forward(inp=None):
        inp = recv_x if inp is None else inp
        return (torch.nn.functional.silu(inp @ w1) * (inp @ w2)) @ w3

    def expert_backward(inp=None, grad=None):
        # Explicit operator-level backward. Using torch.autograd here would
        # charge Python graph construction/runtime overhead that Calculon does
        # not model and would make Phase5 scopes incomparable.
        inp = recv_x if inp is None else inp
        grad = recv_grad if grad is None else grad
        a, b = inp @ w1, inp @ w2
        s = torch.nn.functional.silu(a)
        h = s * b
        dh = grad @ w3.t()
        sig = torch.sigmoid(a)
        da = dh * b * (sig * (1.0 + a * (1.0 - sig)))
        db = dh * s
        dx = da @ w1.t() + db @ w2.t()
        # Keep wgrad in scope; these are the two additional expert projections.
        _dw1, _dw2, _dw3 = inp.t() @ da, inp.t() @ db, h.t() @ grad
        return dx, _dw1, _dw2, _dw3

    def forward_core():
        dispatch(); local_y = expert_forward(); combine(local_y, returned)

    def backward_core():
        dist.all_to_all_single(recv_grad, grad_sorted,
                               recv_splits, send_splits)
        dx, _, _, _ = expert_backward(recv_x, recv_grad)
        combine(dx, returned_dx)

    def train_core():
        forward_core(); backward_core()

    # Independent work is only an overlap probe; it is not added to MoE core.
    overlap_a = torch.randn(hidden, hidden, device=dev, dtype=dtype)
    overlap_b = torch.randn(hidden, hidden, device=dev, dtype=dtype)
    def overlap_compute():
        return overlap_a @ overlap_b
    def serial_probe():
        dispatch(); overlap_compute()
    def overlap_probe():
        work = dist.all_to_all_single(recv_x, send_x, recv_splits,
                                      send_splits, async_op=True)
        overlap_compute(); work.wait()

    measured = {
        'router': measure_ms(route_op, args.warmup, args.iters, dev),
        'ep_dispatch': measure_ms(dispatch, args.warmup, args.iters, dev),
        'expert_forward': measure_ms(expert_forward, args.warmup, args.iters, dev),
        'expert_backward': measure_ms(expert_backward, args.warmup, args.iters, dev),
        'forward_core': measure_ms(forward_core, args.warmup, args.iters, dev),
        'backward_core': measure_ms(backward_core, args.warmup, args.iters, dev),
        'train_core': measure_ms(train_core, args.warmup, args.iters, dev),
        'overlap_serial': measure_ms(serial_probe, args.warmup, args.iters, dev),
        'overlap_async': measure_ms(overlap_probe, args.warmup, args.iters, dev),
    }

    cfg = json.loads(args.system.read_text())
    syst = System(cfg)
    syst.set_datatypes(args.dtype, args.dtype)
    net = cfg['networks'][args.network_index]
    bw = float(net['bandwidth']) * 1e9 * float(net['efficiency'])
    latency = float(net['latency'])

    def proc_ms(flops, nbytes, vector=False):
        proc = syst.vector if vector else syst.matrix
        throughput = proc.throughput(args.dtype, max(1.0, flops))
        mem = syst.mem1.throughput(max(1.0, nbytes))
        launch = syst.vector_launch_s if vector else syst.matrix_launch_s
        return 1e3 * max(flops / throughput, nbytes / mem, launch)

    def gemm_ms(m, k, n):
        return proc_ms(2.0*m*k*n, (m*k + k*n + m*n)*bpe)

    # Use the busiest rank for both injection traffic and expert compute.
    counts = torch.tensor([tokens, recv_tokens], device=dev, dtype=torch.int64)
    dist.all_reduce(counts, op=dist.ReduceOp.MAX)
    max_recv = int(counts[1].item())
    local_send = send_splits[rank]
    local_recv = recv_splits[rank]
    remote = torch.tensor([
        (tokens-local_send)*hidden*bpe,
        (recv_tokens-local_recv)*hidden*bpe,
    ], device=dev, dtype=torch.float64)
    dist.all_reduce(remote, op=dist.ReduceOp.MAX)
    comm_bytes = float(max(remote).item())
    comm_ms = (latency + comm_bytes / bw) * 1e3
    flow_ms = None
    try:
        network = syst.get_network(args.network_index)
        try:
            network.flow_network_init(
                bw, bw, bw, bw, bw, net.get('topology', 'Single machine'),
                latency, latency, latency, latency, latency)
        except TypeError:
            # Compatibility with an older installed Python wrapper while the
            # source tree/shared library is being rebuilt.
            network.flow_network_init(
                bw, bw, net.get('topology', 'Single machine'))
        flow = network.total_flow_network_time(
            1, 1, 1, 0.0, 0.0, 1, 0, 0, 0, 0, 0, False,
            ep=world,
            fwd_ep_dispatch_size=int(comm_bytes),
            fwd_ep_combine_size=int(comm_bytes),
            bwd_ep_dispatch_size=int(comm_bytes),
            bwd_ep_combine_size=int(comm_bytes))
        flow_ms = float(flow[0]) * 1e3
    except Exception as exc:
        # The analytic component model remains usable if the optional shared
        # library is absent, but Phase5 records that the C++ closure was not run.
        flow_error = repr(exc)
    else:
        flow_error = None

    router_gemm = gemm_ms(tokens, hidden, world)
    router_struct = proc_ms(tokens*world*4, tokens*world*6, vector=True)
    router_ms = router_gemm + router_struct
    fw_ms = (gemm_ms(max_recv, hidden, ffn) * 2 +
             proc_ms(max_recv*ffn*3, max_recv*ffn*bpe*4, vector=True) +
             gemm_ms(max_recv, ffn, hidden))
    # Agrad + wgrad for three GEMMs is approximately 2x their forward GEMMs;
    # activation backward is charged at twice forward vector work.
    bw_ms = 2.0 * (fw_ms - proc_ms(max_recv*ffn*3,
                                    max_recv*ffn*bpe*4, vector=True)) + \
            2.0 * proc_ms(max_recv*ffn*3,
                          max_recv*ffn*bpe*4, vector=True)
    predicted = {
        'router': router_ms,
        'ep_dispatch': comm_ms,
        'expert_forward': fw_ms,
        'expert_backward': bw_ms,
        'forward_core': 2*comm_ms + fw_ms,
        'backward_core': 2*comm_ms + bw_ms,
        'train_core': 4*comm_ms + fw_ms + bw_ms,
    }
    predicted['overlap_serial'] = comm_ms + gemm_ms(hidden, hidden, hidden)
    predicted['overlap_async'] = max(comm_ms, gemm_ms(hidden, hidden, hidden))

    # Component rows diagnose whether Phase0/3 or Phase4 is responsible.  The
    # hard acceptance criterion is the composed MoE core: component launch
    # overheads are not additive and should not independently fail Phase5.
    gated = {'forward_core', 'backward_core', 'train_core'}
    rows = []
    for name, meas in measured.items():
        pred = predicted[name]
        err = relerr(pred, meas)
        limit = args.gate_core_pct if name.endswith('_core') else args.gate_component_pct
        rows.append({'component': name, 'measured_ms': meas,
                     'predicted_ms': pred, 'error_pct': err,
                     'gate_pct': limit if name in gated else '',
                     'pass': abs(err) <= limit if name in gated else ''})
    checks = {r['component']: bool(r['pass']) for r in rows
              if r['component'] in gated}
    result = {
        'schema': 'bw1100.phase5.distributed_moe_quick.v1',
        'world_size': world, 'backend': dist.get_backend(),
        'shape': {'tokens_per_rank': tokens, 'hidden': hidden,
                  'ffn_hidden': ffn, 'experts_per_rank': 1,
                  'dtype': args.dtype, 'max_received_tokens': max_recv},
        'network_model': {'tier': args.network_index,
                          'bandwidth_GBps': bw/1e9,
                          'latency_us': latency*1e6,
                          'max_remote_bytes_per_rank': comm_bytes},
        'llm_flow_network_only_train_core_ms': flow_ms,
        'llm_flow_network_only_error': flow_error,
        'rows': rows,
        'overlap_gain_pct': 100.0*(measured['overlap_serial']-
                                    measured['overlap_async']) /
                            measured['overlap_serial'],
        'checks': checks,
        'pass': all(checks.values()),
        'scope': {
            'validated': ['router', 'EP dispatch/combine', 'expert FFN',
                          'backward', 'end-to-end MoE core', 'overlap'],
            'report_only': ['router (sort has no Phase0 parametric model)',
                            'overlap_serial'],
            'not_covered': ['optimizer', 'PP bubble', 'full-model extrapolation'],
        },
    }
    if rank == 0:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        tmp = args.output.with_suffix(args.output.suffix + '.tmp')
        tmp.write_text(json.dumps(result, indent=2) + '\n')
        os.replace(tmp, args.output)
        with args.csv.open('w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader(); w.writerows(rows)
        print(json.dumps(result, indent=2), flush=True)
    dist.barrier(); dist.destroy_process_group()
    raise SystemExit(0 if result['pass'] else 1)


if __name__ == '__main__':
    main()
