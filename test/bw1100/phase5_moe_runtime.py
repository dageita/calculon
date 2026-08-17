"""Runtime helpers for the BW1100 Phase5 distributed MoE closure."""
from __future__ import annotations

import math
import os
import re
import statistics
import subprocess
import threading
import time

import torch
import torch.distributed as dist


def _percentile(values, q):
    values = sorted(float(v) for v in values)
    if not values:
        return None
    pos = (len(values) - 1) * float(q)
    lo, hi = int(math.floor(pos)), int(math.ceil(pos))
    if lo == hi:
        return values[lo]
    return values[lo] * (hi - pos) + values[hi] * (pos - lo)


def summarize(values):
    values = [float(v) for v in values]
    mean = statistics.mean(values)
    stdev = statistics.stdev(values) if len(values) > 1 else 0.0
    return {
        'median_ms': statistics.median(values),
        'mean_ms': mean,
        'p10_ms': _percentile(values, .10),
        'p90_ms': _percentile(values, .90),
        'stdev_ms': stdev,
        'cv_pct': 100.0 * stdev / mean if mean > 0 else 0.0,
        'samples_ms': values,
    }


def measure_distributed(fn, warmup, iters, dev):
    for _ in range(max(0, warmup)):
        fn()
    torch.cuda.synchronize(dev)
    values = []
    for _ in range(max(1, iters)):
        dist.barrier()
        start, end = torch.cuda.Event(True), torch.cuda.Event(True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        values.append(float(start.elapsed_time(end)))
    local = summarize(values)
    per_rank = [None] * dist.get_world_size()
    dist.all_gather_object(per_rank, local)
    worst = max(per_rank, key=lambda x: x['median_ms'])
    return {
        'measured_ms': float(worst['median_ms']),
        'max_rank_cv_pct': max(float(x['cv_pct']) for x in per_rank),
        'global_p10_ms': min(float(x['p10_ms']) for x in per_rank),
        'global_p90_ms': max(float(x['p90_ms']) for x in per_rank),
        'per_rank': per_rank,
    }


class EventDAG:
    """Small deterministic resource-constrained event scheduler."""
    def __init__(self):
        self.nodes = []

    def add(self, name, duration_s, resource, deps=()):
        self.nodes.append({
            'name': name, 'duration_s': max(0.0, float(duration_s)),
            'resource': resource, 'deps': tuple(deps),
        })
        return name

    def schedule(self):
        end = {}
        resources = {}
        timeline = []
        pending = list(self.nodes)
        while pending:
            progressed = False
            for node in list(pending):
                if all(dep in end for dep in node['deps']):
                    start = max([end[d] for d in node['deps']] +
                                [resources.get(node['resource'], 0.0)])
                    finish = start + node['duration_s']
                    end[node['name']] = finish
                    resources[node['resource']] = finish
                    timeline.append({**node, 'start_s': start, 'end_s': finish})
                    pending.remove(node)
                    progressed = True
            if not progressed:
                raise RuntimeError(f'event DAG cycle/missing dependency: {pending}')
        return max(end.values(), default=0.0), timeline


def read_hcu_utilization(device_ids):
    """Return physical HCU utilization; unavailable counters are explicit."""
    try:
        text = subprocess.check_output(
            ['/opt/hyhal/bin/hy-smi', '--showuse'], text=True,
            stderr=subprocess.DEVNULL, timeout=4)
    except (OSError, subprocess.SubprocessError):
        return {str(i): None for i in device_ids}
    found = {int(i): float(v) for i, v in
             re.findall(r'HCU\[(\d+)\].*?HCU use \(%\):\s*([0-9.]+)', text)}
    return {str(i): found.get(int(i)) for i in device_ids}


class UtilizationSampler:
    def __init__(self, device_ids, interval_s=.25):
        self.device_ids = list(device_ids)
        self.interval_s = interval_s
        self.samples = []
        self.stop_event = threading.Event()
        self.thread = None

    def __enter__(self):
        def sample():
            while not self.stop_event.is_set():
                self.samples.append({
                    'time_s': time.time(),
                    'utilization_pct': read_hcu_utilization(self.device_ids),
                })
                self.stop_event.wait(self.interval_s)
        self.thread = threading.Thread(target=sample, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *_):
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=2)

    def summary(self):
        result = {'sample_count': len(self.samples), 'per_device': {}}
        for dev in self.device_ids:
            vals = [s['utilization_pct'].get(str(dev)) for s in self.samples]
            vals = [float(v) for v in vals if v is not None]
            result['per_device'][str(dev)] = {
                'mean_pct': statistics.mean(vals) if vals else None,
                'max_pct': max(vals) if vals else None,
            }
        return result


def _grouped_call(a, w, out, counts):
    import triton
    from phase3_fused_block import _grouped_expert_fp8_kernel

    experts, m, k = a.shape
    n = out.shape[2]
    bm, bn, bk = 32, 64, 64
    gm, gn = triton.cdiv(m, bm), triton.cdiv(n, bn)
    grid = (experts * gm * gn,)
    _grouped_expert_fp8_kernel[grid](
        a, w, out, counts, m, n, k,
        a.stride(0), a.stride(1), a.stride(2),
        w.stride(0), w.stride(1), w.stride(2),
        out.stride(0), out.stride(1), out.stride(2),
        BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk,
        GRID_M=gm, GRID_N=gn, num_warps=4, num_stages=1)


def _dense_call(a, w, out):
    import triton
    from triton_fp8_gemm import _fp8_gemm

    m, k = a.shape
    n = out.shape[1]
    bm = 64 if m < 128 else 128
    bn = 64 if n < 128 else 128
    bk = 32 if k <= 32 else 64 if k <= 64 else 128
    grid = (triton.cdiv(m, bm) * triton.cdiv(n, bn),)
    _fp8_gemm[grid](
        a, w, out, m, n, k,
        a.stride(0), a.stride(1), w.stride(0), w.stride(1),
        out.stride(0), out.stride(1), BLOCK_M=bm, BLOCK_N=bn,
        BLOCK_K=bk, num_warps=4, num_stages=1)


class DeepSeekGroupedMoeRuntime:
    """Routing-faithful EP workload using BW1100's Triton grouped FP8 path."""
    backend = 'triton_grouped_fp8_one_launch'

    def __init__(self, tokens, hidden, ffn, experts, topk, shared_experts,
                 dev, measure_wgrad=True):
        self.rank, self.world = dist.get_rank(), dist.get_world_size()
        self.dev = dev
        self.tokens, self.hidden, self.ffn = tokens, hidden, ffn
        self.experts, self.topk = experts, topk
        self.shared_experts = shared_experts
        if experts % self.world:
            raise ValueError('num experts must be divisible by EP world size')
        self.local_experts = experts // self.world
        self.measure_wgrad = measure_wgrad
        storage = torch.float8_e4m3fn

        torch.manual_seed(1234 + self.rank)
        self.x = torch.randn(tokens, hidden, device=dev, dtype=torch.bfloat16)
        self.router_w = (torch.randn(hidden, experts, device=dev,
                                     dtype=torch.bfloat16) /
                         math.sqrt(hidden))
        self.router_grad = torch.randn(tokens, experts, device=dev,
                                       dtype=torch.bfloat16)
        assignments = tokens * topk
        self.send_x = torch.empty(assignments, hidden, device=dev,
                                  dtype=torch.bfloat16)
        self.send_expert = torch.empty(assignments, device=dev,
                                       dtype=torch.int64)
        self._route_and_pack_send()
        self.send_splits = self._splits_from_destinations()
        split_tensor = torch.tensor(self.send_splits, device=dev,
                                    dtype=torch.int64)
        gathered = [torch.empty_like(split_tensor) for _ in range(self.world)]
        dist.all_gather(gathered, split_tensor)
        self.recv_splits = [int(gathered[src][self.rank].item())
                            for src in range(self.world)]
        recv_rows = sum(self.recv_splits)
        self.recv_x = torch.empty(recv_rows, hidden, device=dev,
                                  dtype=torch.bfloat16)
        self.recv_expert = torch.empty(recv_rows, device=dev,
                                       dtype=torch.int64)
        self.returned = torch.empty_like(self.send_x)
        self.returned_dx = torch.empty_like(self.send_x)
        self.recv_grad = torch.empty_like(self.recv_x)
        self.grad_sorted = torch.randn_like(self.send_x)
        dist.all_to_all_single(self.recv_expert, self.send_expert,
                               self.recv_splits, self.send_splits)
        self.dispatch()

        local_ids = (self.recv_expert - self.rank * self.local_experts)
        if bool(((local_ids < 0) | (local_ids >= self.local_experts)).any()):
            raise RuntimeError('received expert does not belong to this EP rank')
        host_ids = [int(v) for v in local_ids.cpu().tolist()]
        seen = [0] * self.local_experts
        slots = []
        for expert in host_ids:
            slots.append(seen[expert]); seen[expert] += 1
        self.local_ids = local_ids
        self.slots = torch.tensor(slots, device=dev, dtype=torch.int64)
        self.counts_host = seen
        self.max_m = max(1, max(seen, default=0))
        self.counts = torch.tensor(seen, device=dev, dtype=torch.int32)

        self.packed = torch.empty(self.local_experts, self.max_m, hidden,
                                  device=dev, dtype=storage)
        self.gate_w = torch.empty(self.local_experts, hidden, ffn,
                                  device=dev, dtype=storage).fill_(1)
        self.up_w = torch.empty_like(self.gate_w).fill_(1)
        self.down_w = torch.empty(self.local_experts, ffn, hidden,
                                  device=dev, dtype=storage).fill_(1)
        self.gate_out = torch.empty(self.local_experts, self.max_m, ffn,
                                    device=dev, dtype=torch.bfloat16)
        self.up_out = torch.empty_like(self.gate_out)
        self.hidden_fp8 = torch.empty(self.local_experts, self.max_m, ffn,
                                      device=dev, dtype=storage)
        self.down_out = torch.empty(self.local_experts, self.max_m, hidden,
                                    device=dev, dtype=torch.bfloat16)
        self.routed_rows = torch.empty(recv_rows, hidden, device=dev,
                                       dtype=torch.bfloat16)

        # Shared expert is replicated and runs as ordinary FP8 GEMMs.
        self.shared_x = torch.empty(tokens, hidden, device=dev, dtype=storage)
        self.shared_gate_w = torch.empty(ffn, hidden, device=dev,
                                         dtype=storage).fill_(1).t()
        self.shared_up_w = torch.empty(ffn, hidden, device=dev,
                                       dtype=storage).fill_(1).t()
        self.shared_down_w = torch.empty(hidden, ffn, device=dev,
                                         dtype=storage).fill_(1).t()
        self.shared_gate = torch.empty(tokens, ffn, device=dev,
                                       dtype=torch.bfloat16)
        self.shared_up = torch.empty_like(self.shared_gate)
        self.shared_hidden = torch.empty(tokens, ffn, device=dev, dtype=storage)
        self.shared_out = torch.empty(tokens, hidden, device=dev,
                                      dtype=torch.bfloat16)
        self.shared_grad_h = torch.empty(tokens, hidden, device=dev,
                                         dtype=storage).fill_(1)
        self.shared_grad_f = torch.empty(tokens, ffn, device=dev,
                                         dtype=storage).fill_(1)
        self.shared_agrad_h = torch.empty(tokens, hidden, device=dev,
                                          dtype=torch.bfloat16)
        self.shared_agrad_f = torch.empty(tokens, ffn, device=dev,
                                          dtype=torch.bfloat16)
        self.shared_wgrad_a_h = torch.empty(hidden, tokens, device=dev,
                                             dtype=storage).fill_(1)
        self.shared_wgrad_a_f = torch.empty(ffn, tokens, device=dev,
                                             dtype=storage).fill_(1)
        self.shared_wgrad_b_h = torch.empty(tokens, hidden, device=dev,
                                             dtype=storage).fill_(1)
        self.shared_wgrad_b_f = torch.empty(tokens, ffn, device=dev,
                                             dtype=storage).fill_(1)
        self.shared_wgrad_out = torch.empty(hidden, ffn, device=dev,
                                            dtype=torch.bfloat16)

        # Explicit agrad shapes reuse transposed forward weights.  One large
        # wgrad output buffer is reused across Gate/Up/Down to bound memory.
        self.grad_h = torch.empty(self.local_experts, self.max_m, hidden,
                                  device=dev, dtype=storage).fill_(1)
        self.grad_f = torch.empty(self.local_experts, self.max_m, ffn,
                                  device=dev, dtype=storage).fill_(1)
        self.agrad_h = torch.empty(self.local_experts, self.max_m, hidden,
                                   device=dev, dtype=torch.bfloat16)
        self.agrad_f = torch.empty(self.local_experts, self.max_m, ffn,
                                   device=dev, dtype=torch.bfloat16)
        if measure_wgrad:
            self.wgrad_a_h = torch.empty(self.local_experts, hidden, self.max_m,
                                         device=dev, dtype=storage).fill_(1)
            self.wgrad_a_f = torch.empty(self.local_experts, ffn, self.max_m,
                                         device=dev, dtype=storage).fill_(1)
            self.wgrad_b_f = torch.empty(self.local_experts, self.max_m, ffn,
                                         device=dev, dtype=storage).fill_(1)
            self.wgrad_b_h = torch.empty(self.local_experts, self.max_m, hidden,
                                         device=dev, dtype=storage).fill_(1)
            self.wgrad_out = torch.empty(self.local_experts, hidden, ffn,
                                         device=dev, dtype=torch.bfloat16)
            self.full_h = torch.full((self.local_experts,), hidden, device=dev,
                                     dtype=torch.int32)
            self.full_f = torch.full((self.local_experts,), ffn, device=dev,
                                     dtype=torch.int32)
        self.pack()

    def _route_tensors(self):
        scores = torch.sigmoid(self.x @ self.router_w)
        _, experts = torch.topk(scores, self.topk, dim=1, sorted=False)
        token_ids = torch.arange(self.tokens, device=self.dev).repeat_interleave(
            self.topk)
        flat_experts = experts.reshape(-1)
        destinations = torch.div(flat_experts, self.local_experts,
                                 rounding_mode='floor')
        order = torch.argsort(destinations)
        return token_ids, flat_experts, destinations, order

    def _route_and_pack_send(self):
        token_ids, experts, destinations, order = self._route_tensors()
        self.send_x.copy_(self.x.index_select(0, token_ids.index_select(0, order)))
        self.send_expert.copy_(experts.index_select(0, order))
        self.destinations = destinations.index_select(0, order)
        inverse = torch.empty_like(order)
        inverse[order] = torch.arange(order.numel(), device=self.dev)
        self.inverse_order = inverse

    def _splits_from_destinations(self):
        return torch.bincount(self.destinations, minlength=self.world).tolist()

    def router(self):
        self._route_and_pack_send()

    def router_backward(self):
        # Explicit router dX/dW plus score derivative/top-k structural work.
        logits = self.x @ self.router_w
        sig = torch.sigmoid(logits)
        grad = self.router_grad * sig * (1.0 - sig)
        self.router_dx = grad @ self.router_w.t()
        self.router_dw = self.x.t() @ grad
        torch.topk(sig, self.topk, dim=1, sorted=False)

    def dispatch(self):
        # Expert ids/splits are frozen for this controlled closure and were
        # exchanged once during setup.  Re-sending them as a second RCCL
        # collective would charge an extra startup absent from the C++ EP
        # phase and from production fused dispatch metadata.
        dist.all_to_all_single(self.recv_x, self.send_x,
                               self.recv_splits, self.send_splits)

    def dispatch_async(self):
        work = dist.all_to_all_single(self.recv_x, self.send_x,
                                   self.recv_splits, self.send_splits,
                                   async_op=True)
        return (work,)

    def pack(self):
        self.packed[self.local_ids, self.slots] = self.recv_x.to(
            torch.float8_e4m3fn)

    def routed_forward(self):
        _grouped_call(self.packed, self.gate_w, self.gate_out, self.counts)
        _grouped_call(self.packed, self.up_w, self.up_out, self.counts)
        self.hidden_fp8.copy_((torch.nn.functional.silu(self.gate_out) *
                              self.up_out).to(torch.float8_e4m3fn))
        _grouped_call(self.hidden_fp8, self.down_w, self.down_out, self.counts)

    def shared_forward(self):
        self.shared_x.copy_(self.x.to(torch.float8_e4m3fn))
        _dense_call(self.shared_x, self.shared_gate_w, self.shared_gate)
        _dense_call(self.shared_x, self.shared_up_w, self.shared_up)
        self.shared_hidden.copy_((torch.nn.functional.silu(self.shared_gate) *
                                  self.shared_up).to(torch.float8_e4m3fn))
        _dense_call(self.shared_hidden, self.shared_down_w, self.shared_out)

    def unpack(self):
        self.routed_rows.copy_(self.down_out[self.local_ids, self.slots])

    def combine(self, inp=None, out=None):
        dist.all_to_all_single(self.returned if out is None else out,
                               self.routed_rows if inp is None else inp,
                               self.send_splits, self.recv_splits)

    def unpermute(self):
        routed = self.returned.index_select(0, self.inverse_order)
        return routed.view(self.tokens, self.topk, self.hidden).sum(1).add_(
            self.shared_out)

    def expert_forward(self):
        self.routed_forward(); self.shared_forward()

    def expert_recompute(self):
        self.pack(); self.routed_forward(); self.shared_forward()

    def expert_agrad(self):
        _grouped_call(self.grad_h, self.down_w.transpose(1, 2),
                      self.agrad_f, self.counts)
        # DTK PyTorch intentionally has no implicit BF16/FP8 promotion.
        self.agrad_f.mul_(self.hidden_fp8.to(torch.bfloat16))
        _grouped_call(self.grad_f, self.gate_w.transpose(1, 2),
                      self.agrad_h, self.counts)
        _grouped_call(self.grad_f, self.up_w.transpose(1, 2),
                      self.agrad_h, self.counts)
        # Replicated shared expert gradients use the same FP8 Triton backend.
        _dense_call(self.shared_grad_h, self.shared_down_w.transpose(0, 1),
                    self.shared_agrad_f)
        _dense_call(self.shared_grad_f, self.shared_gate_w.transpose(0, 1),
                    self.shared_agrad_h)
        _dense_call(self.shared_grad_f, self.shared_up_w.transpose(0, 1),
                    self.shared_agrad_h)

    def expert_wgrad(self):
        _dense_call(self.shared_wgrad_a_h, self.shared_wgrad_b_f,
                    self.shared_wgrad_out)
        _dense_call(self.shared_wgrad_a_h, self.shared_wgrad_b_f,
                    self.shared_wgrad_out)
        _dense_call(self.shared_wgrad_a_f, self.shared_wgrad_b_h,
                    self.shared_wgrad_out.view(self.ffn, self.hidden))
        if self.measure_wgrad:
            _grouped_call(self.wgrad_a_h, self.wgrad_b_f,
                          self.wgrad_out, self.full_h)
            _grouped_call(self.wgrad_a_h, self.wgrad_b_f,
                          self.wgrad_out, self.full_h)
            _grouped_call(self.wgrad_a_f, self.wgrad_b_h,
                          self.wgrad_out.view(self.local_experts,
                                              self.ffn, self.hidden), self.full_f)

    def expert_backward(self):
        # Recompute, agrad and wgrad are separate models/measurements; this
        # composed callable validates their runtime interaction.
        self.expert_recompute(); self.expert_agrad(); self.expert_wgrad()

    def forward_core(self):
        self.router()
        works = self.dispatch_async()
        self.shared_forward()
        for work in works:
            work.wait()
        self.pack(); self.routed_forward(); self.unpack(); self.combine()
        self.unpermute()

    def forward_core_serial(self):
        self.router(); self.dispatch(); self.shared_forward()
        self.pack(); self.routed_forward(); self.unpack(); self.combine()
        self.unpermute()

    def backward_core(self):
        dist.all_to_all_single(self.recv_grad, self.grad_sorted,
                               self.recv_splits, self.send_splits)
        self.expert_backward()
        self.routed_rows.copy_(self.agrad_h[self.local_ids, self.slots])
        self.combine(self.routed_rows, self.returned_dx)
        self.router_backward()

    def train_core(self):
        self.forward_core(); self.backward_core()

    def imbalance(self):
        counts = torch.tensor(self.counts_host, device=self.dev,
                              dtype=torch.float64)
        total = counts.clone()
        gathered = [None] * self.world
        dist.all_gather_object(gathered, self.counts_host)
        flat = [float(v) for part in gathered for v in part]
        mean = statistics.mean(flat) if flat else 0.0
        cv = (100.0 * statistics.pstdev(flat) / mean
              if mean > 0 and len(flat) > 1 else 0.0)
        return {'mean_tokens_per_expert': mean,
                'max_tokens_per_expert': max(flat, default=0.0),
                'cv_pct': cv, 'counts': flat}

    def communication_bytes(self):
        local_send = int(self.send_splits[self.rank])
        local_recv = int(self.recv_splits[self.rank])
        dispatch = (self.send_x.shape[0] - local_send) * self.hidden * 2
        combine = (self.recv_x.shape[0] - local_recv) * self.hidden * 2
        values = torch.tensor([dispatch, combine], device=self.dev,
                              dtype=torch.float64)
        dist.all_reduce(values, op=dist.ReduceOp.MAX)
        return int(values[0].item()), int(values[1].item())


class DenseDebugRuntime:
    """Small BF16 dense-expert unit test; not a production MoE validation."""
    backend = 'torch_bf16_dense_debug'

    def __init__(self, tokens, hidden, ffn, dev, dtype=torch.bfloat16):
        self.rank, self.world = dist.get_rank(), dist.get_world_size()
        self.dev, self.tokens = dev, tokens
        self.hidden, self.ffn, self.dtype = hidden, ffn, dtype
        torch.manual_seed(1234 + self.rank)
        self.x = torch.randn(tokens, hidden, device=dev, dtype=dtype)
        self.router_w = torch.randn(hidden, self.world, device=dev,
                                    dtype=dtype) / math.sqrt(hidden)
        self.router_grad = torch.randn(tokens, self.world, device=dev,
                                       dtype=dtype)
        self.send_x = torch.empty_like(self.x)
        self.router()
        self.send_splits = torch.bincount(
            self.destinations, minlength=self.world).tolist()
        local = torch.tensor(self.send_splits, device=dev, dtype=torch.int64)
        gathered = [torch.empty_like(local) for _ in range(self.world)]
        dist.all_gather(gathered, local)
        self.recv_splits = [int(gathered[src][self.rank].item())
                            for src in range(self.world)]
        self.recv_x = torch.empty(sum(self.recv_splits), hidden, device=dev,
                                  dtype=dtype)
        self.returned = torch.empty_like(self.send_x)
        self.recv_grad = torch.empty_like(self.recv_x)
        self.returned_dx = torch.empty_like(self.send_x)
        self.grad_sorted = torch.randn_like(self.send_x)
        self.w1 = torch.randn(hidden, ffn, device=dev, dtype=dtype)
        self.w2 = torch.randn(hidden, ffn, device=dev, dtype=dtype)
        self.w3 = torch.randn(ffn, hidden, device=dev, dtype=dtype)

    def router(self):
        self.destinations, self.order = torch.sort(torch.argmax(
            self.x @ self.router_w, dim=1))
        self.send_x.copy_(self.x.index_select(0, self.order))

    def dispatch(self):
        dist.all_to_all_single(self.recv_x, self.send_x,
                               self.recv_splits, self.send_splits)

    def router_backward(self):
        self.router_dx = self.router_grad @ self.router_w.t()
        self.router_dw = self.x.t() @ self.router_grad

    def expert_forward(self):
        self.local_y = (torch.nn.functional.silu(self.recv_x @ self.w1) *
                        (self.recv_x @ self.w2)) @ self.w3

    def expert_backward(self):
        a, b = self.recv_x @ self.w1, self.recv_x @ self.w2
        s = torch.nn.functional.silu(a)
        h = s * b
        dh = self.recv_grad @ self.w3.t()
        sig = torch.sigmoid(a)
        da = dh * b * (sig * (1.0 + a * (1.0 - sig)))
        db = dh * s
        self.local_dx = da @ self.w1.t() + db @ self.w2.t()
        self.dw1, self.dw2, self.dw3 = (
            self.recv_x.t() @ da, self.recv_x.t() @ db, h.t() @ self.recv_grad)

    def combine(self, inp=None, out=None):
        dist.all_to_all_single(self.returned if out is None else out,
                               self.local_y if inp is None else inp,
                               self.send_splits, self.recv_splits)

    def forward_core(self):
        self.router(); self.dispatch(); self.expert_forward(); self.combine()

    forward_core_serial = forward_core

    def backward_core(self):
        dist.all_to_all_single(self.recv_grad, self.grad_sorted,
                               self.recv_splits, self.send_splits)
        self.expert_backward(); self.combine(self.local_dx, self.returned_dx)
        self.router_backward()

    def train_core(self):
        self.forward_core(); self.backward_core()

    def imbalance(self):
        gathered = [None] * self.world
        dist.all_gather_object(gathered, self.recv_splits)
        flat = [float(v) for part in gathered for v in part]
        mean = statistics.mean(flat) if flat else 0.0
        return {'mean_tokens_per_expert': mean,
                'max_tokens_per_expert': max(flat, default=0.0),
                'cv_pct': (100.0 * statistics.pstdev(flat) / mean
                           if mean > 0 else 0.0), 'counts': flat}

    def communication_bytes(self):
        local_send = self.send_splits[self.rank]
        local_recv = self.recv_splits[self.rank]
        bpe = torch.empty((), dtype=self.dtype).element_size()
        values = torch.tensor([
            (self.tokens - local_send) * self.hidden * bpe,
            (self.recv_x.shape[0] - local_recv) * self.hidden * bpe,
        ], device=self.dev, dtype=torch.float64)
        dist.all_reduce(values, op=dist.ReduceOp.MAX)
        return int(values[0].item()), int(values[1].item())
