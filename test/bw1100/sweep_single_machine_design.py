#!/usr/bin/env python3
"""Sweep a fixed DeepSeek-V3 request over defensible 5D scale-out points.

Each candidate has TP*PP*DP*EP*CP == num_procs.  DP alone expands global
batch size (batch_size=DP and microbatch_size=1), keeping per-DP-replica work
constant so the final throughput trend is meaningful.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


# The first four rows grow scale-up parallelism.  From 512 onward, DP doubles
# while batch=DP, which is the least ambiguous linear-scaling experiment.
CASES = (
    (32,   4, 2, 1, 4,  1),
    (64,   4, 2, 1, 8,  1),
    (128,  8, 2, 1, 8,  1),
    (256,  8, 2, 1, 16, 1),
    (512,  8, 4, 1, 16, 1),
    (1024, 8, 4, 2, 16, 1),
    (2048, 8, 4, 4, 16, 1),
    (4096, 8, 4, 8, 16, 1),
)

HEADERS = [
    'Model', 'Network', 'datatype', 'TP', 'PP', 'DP', 'EP', 'CP',
    'Batch Size', 'Microbatch Size', 'Activation recompute',
    'Optimizer sharding', 'Batch Time(s)', 'Comm Time(s)',
    'Per-batch EP communication time(s)', 'Comm Ratio', 'Memory(GiB)',
    'MFU', 'Linear Scaling Throughput (samples/s)',
]


def payload(nproc, tp, pp, dp, ep, cp):
    assert tp * pp * dp * ep * cp == nproc
    return {
        'gpu': {
            'name': 'BW1100',
            'sparse_tensor_fp16_processing_power': 354,
            'sparse_tensor_fp32_processing_power': 44,
            'memory': 144,
            'memory_bandwidth': 2400,
            'bus_bandwidth': 158.51514702896458,
            'network_bandwidth': 25,
            'support_p2p': True,
            'num_procs': nproc,
        },
        'network': {'network_bandwidth': 25, 'network_topology': 'Single machine'},
        'model': {
            'name': 'DeepSeek-V3 671B', 'seq_size': 4096, 'hidden': 7168,
            'feedforward': 18432, 'attn_heads': 128, 'kv_heads': None,
            'attn_size': 128, 'rope_theta': None, 'rms_norm': None,
            'qk_norm': None, 'ffn_type': None, 'untied_embeddings': None,
            'num_blocks': 61, 'vocab_size': 129280, 'num_experts': 256,
            'moe_topk': 8, 'norm_topk_prob': None, 'router_aux_loss_coef': None,
            'num_shared_experts': 1, 'moe_feedforward': 2048,
            'first_k_dense': 3, 'moe_layer_freq': 1, 'kv_size': 576,
            'q_lora_rank': 1536, 'kv_lora_rank': 512,
            'qk_nope_head_dim': 128, 'qk_rope_head_dim': 64, 'v_head_dim': 128,
        },
        'trainning_config': {
            'optimization_strategy': 'Full recomputation',
            'activation_recompute': 'full', 'optimizer_sharding': False,
            'tensor_par': tp, 'pipeline_par': pp, 'data_par': dp,
            'expert_par': ep, 'context_par': cp,
            'batch_size': dp, 'microbatch_size': 1,
            'matrix_dtype': 'float8', 'vector_dtype': 'float8',
        },
    }


def get_path(obj, *keys):
    cur = obj
    for key in keys:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    return cur


def number(obj, *paths):
    for path in paths:
        value = get_path(obj, *path)
        if value is not None:
            if isinstance(value, str):
                # The current API renders memory as e.g. "90.830 GiB".
                # Keep the CSV/Excel value numeric while the column header
                # retains the unit.
                match = re.search(r'[-+]?\d+(?:\.\d+)?', value)
                if match:
                    return float(match.group(0))
            return value
    return None


def row_from_result(result, nproc, tp, pp, dp, ep, cp):
    summary = result.get('summary', {})
    mem = result.get('memory_usage', {})
    comm = result.get('communication', {})
    batch_time = number(result, ('summary', 'batch_total_time'))
    comm_time = number(result, ('communication', 'total_comm_time'))
    ep_time = number(result, ('communication', 'batch_ep_comm_time'))
    return {
        'Model': 'DeepSeek-V3 671B',
        'Network': f'Single machine ({nproc} DCUs)',
        'datatype': 'FP8 matrix / FP8 vector',
        'TP': tp, 'PP': pp, 'DP': dp, 'EP': ep, 'CP': cp,
        'Batch Size': dp, 'Microbatch Size': 1,
        'Activation recompute': 'full', 'Optimizer sharding': False,
        'Batch Time(s)': batch_time,
        'Comm Time(s)': comm_time,
        'Per-batch EP communication time(s)': ep_time,
        'Comm Ratio': (comm_time / batch_time if batch_time and comm_time is not None else None),
        'Memory(GiB)': number(result,
                              ('memory_usage', 'overall_usage'),
                              ('memory_usage', 'overall_usage_gib')),
        'MFU': number(result, ('summary', 'total_efficiency')),
        'Linear Scaling Throughput (samples/s)': number(
            result, ('summary', 'linear_scaling_throughput')),
    }


def post_json(url, body, timeout):
    data = json.dumps(body).encode('utf-8')
    request = urllib.request.Request(
        url, data=data, method='POST',
        headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', default=(
        'http://10.103.240.21:3001/llm_training_calculator/calculator/calculate'))
    parser.add_argument('--output-dir', type=Path,
                        default=Path('test/bw1100/single_machine_design'))
    parser.add_argument('--timeout', type=float, default=180.0)
    parser.add_argument('--pause-s', type=float, default=0.2)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records, raw = [], []
    for nproc, tp, pp, dp, ep, cp in CASES:
        body = payload(nproc, tp, pp, dp, ep, cp)
        print(f'case: n={nproc} TP={tp} PP={pp} DP={dp} EP={ep} CP={cp}', flush=True)
        if args.dry_run:
            records.append(row_from_result({}, nproc, tp, pp, dp, ep, cp))
            continue
        try:
            result = post_json(args.url, body, args.timeout)
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode('utf-8', 'replace')
            raise RuntimeError(f'API HTTP {exc.code} for n={nproc}: {detail}') from exc
        records.append(row_from_result(result, nproc, tp, pp, dp, ep, cp))
        raw.append({'request': body, 'response': result})
        time.sleep(args.pause_s)
    (args.output_dir / 'raw_responses.json').write_text(
        json.dumps(raw, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    with (args.output_dir / 'results.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=HEADERS)
        writer.writeheader(); writer.writerows(records)
    assumptions = {
        'selection_rule': 'TP×PP×DP×EP×CP must equal num_procs',
        'batch_rule': 'microbatch_size=1; batch_size=DP',
        'scale_strategy': '32→512 grows scale-up dimensions; 1024→4096 doubles DP and batch together',
        'threshold_story': '128 DCUs is the first balanced TP8×PP2×EP8 point; compare its latency/MFU/throughput to 64.',
        'cases': [dict(num_procs=n, tp=t, pp=p, dp=d, ep=e, cp=c)
                  for n, t, p, d, e, c in CASES],
    }
    (args.output_dir / 'assumptions.json').write_text(
        json.dumps(assumptions, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(json.dumps({'csv': str(args.output_dir / 'results.csv'),
                      'raw': str(args.output_dir / 'raw_responses.json'),
                      'rows': len(records)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
