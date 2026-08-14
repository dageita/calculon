#!/usr/bin/env python3
"""Phase4: DeepSeek-V3 + H20 下 EP/CP（分层 MLA/FFN + 两阶段 A2A）验证。

分层验收：
  L0  单元：LLMFlowSimulator test_ep.py（legacy EP/CP）
  L1  体积：Calculon 算出的 dispatch/combine / CP 字节与公式一致
  L2  事件：timeline 出现 EP_DISPATCH/COMBINE、COMPUTE_FFN、CP_COMM
  L3  消融：同规模下 EP↑ / 开 CP 时 epComm、cpComm、globalTime 单调合理

推荐并行（H20 96GB, FP8 GEMM + BF16 vector, seq=4096）：
  EP8 CP1 : 256 GPU = TP1 × PP16 × DP2  × EP8 × CP1
  EP8 CP2 : 512 GPU = TP1 × PP16 × DP2  × EP8 × CP2

用法：
  cd /src/Simulator/calculon
  python3 test/phase4_dsv3_ep_cp_validate.py              # L1+L2+L3，默认关 timeline 加速
  python3 test/phase4_dsv3_ep_cp_validate.py --timeline   # 打开 timeline 做事件检查
  python3 test/phase4_dsv3_ep_cp_validate.py --case ep8
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

_TEST = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_TEST)
sys.path.insert(0, _ROOT)

from calculon.llm.llm import Llm  # noqa: E402
from calculon.system import System  # noqa: E402

APP_PATH = os.path.join(_ROOT, 'models', 'deepseek-v3-671b.json')
SYS_PATH = os.path.join(_ROOT, 'systems', 'H20.json')

BASE_EXE = dict(
    tensor_par=1,
    tensor_par_net=0,
    pipeline_par_net=1,
    data_par_net=1,
    expert_par_net=1,
    context_par_net=0,
    datatype='float16',
    matrix_dtype='float8',
    vector_dtype='bfloat16',
    fused_activation=True,
    attention_type='multihead',
    activation_recompute='full',
    pipeline_interleaving=1,
    optimizer_sharding=True,
    tensor_par_comm_type='ar',
    tensor_par_overlap='none',
    seq_par_ag_redo=False,
    data_par_overlap=False,
    weight_offload=False,
    activations_offload=False,
    optimizer_offload=False,
    training=True,
    microbatch_size=1,
)

CASES = {
    # name -> parallel knobs (num_procs must equal product)
    'ep4': dict(num_procs=256, pipeline_par=16, data_par=4, expert_par=4,
                context_par=1, batch_size=16),
    'ep8': dict(num_procs=256, pipeline_par=16, data_par=2, expert_par=8,
                context_par=1, batch_size=16),
    'ep8_cp2': dict(num_procs=512, pipeline_par=16, data_par=2, expert_par=8,
                    context_par=2, batch_size=16),
}


def _decode(t) -> str:
    if t is None:
        return ''
    if isinstance(t, bytes):
        return t.decode('utf-8', errors='ignore').strip()
    return str(t).strip()


def expected_ep_phase_bytes(app: Llm.Application, exe: Llm.Execution,
                            bpe: int) -> int:
    """tokens * topk * hidden * bpe * locality * moe_blocks_per_proc."""
    tokens = exe.microbatch_size * app.seq_size
    locality = (exe.expert_par - 1) / exe.expert_par
    moe_bpp = app.num_moe_blocks / exe.pipeline_par
    return int(tokens * app.moe_topk * app.hidden * bpe * locality * moe_bpp)


def expected_cp_fw_bytes(app: Llm.Application, exe: Llm.Execution,
                         bpe: int, blocks_per_proc: float) -> int:
    chunk = 2 * exe.microbatch_size * (app.seq_size / exe.context_par) * \
        app.kv_size * bpe
    return int(blocks_per_proc * chunk)


def build_model(par: Dict[str, Any], log: logging.Logger) -> Llm:
    app = Llm.Application(json.load(open(APP_PATH)))
    syst = System(json.load(open(SYS_PATH)), log)
    cfg = {**BASE_EXE, **par}
    exe = Llm.Execution.from_json(cfg)
    model = Llm(app, log)
    model.compile(syst, exe)
    model.run(syst)
    return model


def check_sizes(model: Llm) -> List[str]:
    errs = []
    app, exe = model.app, model.exe
    bpe = model._bytes_per_element
    if exe.expert_par > 1 and app.is_moe:
        exp = expected_ep_phase_bytes(app, exe, bpe)
        for name, got in (
            ('dispatch', model._ep_fw_dispatch_size),
            ('combine', model._ep_fw_combine_size),
        ):
            if got != exp:
                errs.append(f'EP {name} size {got} != formula {exp}')
        if model._ep_fw_comm_size != (
                model._ep_fw_dispatch_size + model._ep_fw_combine_size):
            errs.append('EP fw total != dispatch+combine')
    if exe.context_par > 1:
        exp = expected_cp_fw_bytes(app, exe, bpe, model._blocks_per_proc)
        if model._cp_fw_comm_size != exp:
            errs.append(f'CP fw size {model._cp_fw_comm_size} != formula {exp}')
    return errs


def run_flow(model: Llm, timeline: bool) -> Tuple[Dict[str, Any], Counter]:
    model._flow_network_cache = None
    t0 = time.time()
    if timeline:
        r = model.get_total_flow_network_time()
    else:
        r = model._flow_net.total_flow_network_time(
            **model._flow_network_kwargs(enable_timeline=False))
    dt = time.time() - t0
    out = {
        'global_time': r[0],
        'total_comm': r[12],
        'ep_fw': r[19],
        'ep_bw': r[20],
        'ep_comm': r[21],
        'cp_fw': r[22],
        'cp_bw': r[23],
        'cp_comm': r[24],
        'flow_s': dt,
    }
    c: Counter = Counter()
    if timeline and len(r) > 15:
        n = r[13]
        for t in r[15][:n]:
            s = _decode(t)
            if s:
                c[s] += 1
    return out, c


def check_events(exe: Llm.Execution, counts: Counter) -> List[str]:
    errs = []
    if not counts:
        return ['timeline empty (pass --timeline)']
    if exe.expert_par > 1:
        for k in ('EP_DISPATCH_FWD', 'EP_COMBINE_FWD',
                  'EP_DISPATCH_BWD', 'EP_COMBINE_BWD'):
            if counts.get(k, 0) <= 0:
                errs.append(f'missing timeline event {k}')
        if counts.get('COMPUTE_FFN_FWD', 0) <= 0:
            errs.append('missing COMPUTE_FFN_FWD (layered MoE path)')
        if counts.get('COMPUTE_MLA_BWD', 0) <= 0:
            errs.append('missing COMPUTE_MLA_BWD (layered MoE path)')
    if exe.context_par > 1:
        for k in ('CP_COMM_FWD', 'CP_COMM_BWD'):
            if counts.get(k, 0) <= 0:
                errs.append(f'missing timeline event {k}')
    return errs


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--case', choices=list(CASES) + ['all'], default='all')
    p.add_argument('--timeline', action='store_true',
                   help='Enable C++ timeline return (L2 event checks). '
                        'Without this, C++ cout is muted (enableTimeline=false).')
    p.add_argument('-v', '--verbose', action='store_true',
                   help='Keep C++ simulator logs even without --timeline '
                        '(sets SIM_VERBOSE=1)')
    args = p.parse_args()

    if args.verbose:
        os.environ['SIM_VERBOSE'] = '1'

    log = logging.getLogger('phase4')
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format='%(levelname)s %(message)s')

    names = list(CASES) if args.case == 'all' else [args.case]
    rows = []
    failed = 0

    print('=== Phase4 DSV3 EP/CP validation ===')
    print(f'app={APP_PATH}')
    print(f'sys={SYS_PATH}')
    print(f'timeline={args.timeline}')

    for name in names:
        par = CASES[name]
        print(f'\n--- case {name}: {par} ---')
        t0 = time.time()
        try:
            model = build_model(par, log)
        except Exception as e:
            print(f'COMPILE/RUN FAIL: {e}')
            failed += 1
            continue
        print(f'compile+analyt={time.time()-t0:.2f}s  '
              f'mem={model.get_mem_tier1_cap_req()/1e9:.1f}GB  '
              f'attn_fw={model._block_attn_fw_time:.4e}s  '
              f'ffn_fw={model._block_ffn_fw_time:.4e}s  '
              f'mbs={model.exe._num_microbatches}')

        size_errs = check_sizes(model)
        for e in size_errs:
            print('SIZE FAIL:', e)
        if size_errs:
            failed += 1

        flow, counts = run_flow(model, args.timeline)
        print(f"flow={flow['flow_s']:.2f}s  global={flow['global_time']:.4f}s  "
              f"epComm={flow['ep_comm']:.4f}s  cpComm={flow['cp_comm']:.4f}s  "
              f"disp={model._ep_fw_dispatch_size}  cp_fw={model._cp_fw_comm_size}")
        if args.timeline:
            interesting = {k: v for k, v in counts.items()
                           if any(x in k for x in (
                               'EP_', 'CP_', 'COMPUTE_FFN', 'COMPUTE_MLA',
                               'COMPUTE_FWD', 'COMPUTE_BWD'))}
            print('events:', interesting)
            ev_errs = check_events(model.exe, counts)
            for e in ev_errs:
                print('EVENT FAIL:', e)
            if ev_errs:
                failed += 1
        else:
            # L2 without timeline: at least EP/CP times must be >0 when enabled
            if model.exe.expert_par > 1 and flow['ep_comm'] <= 0:
                print('EVENT FAIL: ep_comm==0 with EP>1')
                failed += 1
            if model.exe.context_par > 1 and flow['cp_comm'] <= 0:
                print('EVENT FAIL: cp_comm==0 with CP>1')
                failed += 1

        rows.append({'name': name, **flow,
                     'disp': model._ep_fw_dispatch_size,
                     'ep': model.exe.expert_par,
                     'cp': model.exe.context_par})

    # L3: EP4 → EP8 epComm should generally rise (topk volume × locality↑)
    by = {r['name']: r for r in rows}
    if 'ep4' in by and 'ep8' in by:
        if by['ep8']['ep_comm'] < by['ep4']['ep_comm'] * 0.5:
            print('\nABLATION WARN: EP8 epComm unexpectedly << EP4 '
                  f"({by['ep8']['ep_comm']} vs {by['ep4']['ep_comm']})")
        else:
            print('\nABLATION OK: EP8 epComm >= ~0.5× EP4 '
                  f"({by['ep8']['ep_comm']:.4f} vs {by['ep4']['ep_comm']:.4f})")
    if 'ep8' in by and 'ep8_cp2' in by:
        if by['ep8_cp2']['cp_comm'] <= 0:
            print('ABLATION FAIL: CP2 cpComm==0')
            failed += 1
        else:
            print('ABLATION OK: EP8+CP2 cpComm>0 '
                  f"({by['ep8_cp2']['cp_comm']:.4f})")

    print('\n=== DONE ===', 'FAIL' if failed else 'PASS', f'({failed} failures)')
    sys.exit(1 if failed else 0)


if __name__ == '__main__':
    main()
