"""End-to-end verification: DeepSeek V3 (MoE) + EP/CP through calculon -> C++ LLMFlowSimulator."""
import json
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from calculon.llm.llm import Llm
from calculon.llm.runner import Runner
from calculon import System

logging.basicConfig(level=logging.WARNING)
log = logging.getLogger('moe_e2e')

ROOT = os.path.dirname(os.path.abspath(__file__))


def make_system(num_procs):
    with open(os.path.join(ROOT, 'systems', 'a100_80g.json')) as f:
        cfg = json.load(f)
    cfg['networks'][0]['size'] = num_procs
    for net in cfg['networks']:
        net.setdefault('topology', 'fully connected')
    # 本测试聚焦 EP/CP 通信链路；calculon 的逐层显存模型尚未按 EP 切分专家权重，
    # 671B 在 16 卡上会触发显存上限检查，故放大 HBM 绕过（不影响通信仿真）。
    cfg['mem1']['GiB'] = 16384
    return System(cfg, log)


def make_app():
    with open(os.path.join(ROOT, 'models', 'deepseek-v3-671b.json')) as f:
        return Llm.Application(json.load(f))


def make_exe(num_procs, tp, pp, dp, ep, cp):
    return Llm.Execution.from_json({
        'num_procs': num_procs,
        'tensor_par': tp,
        'pipeline_par': pp,
        'data_par': dp,
        'expert_par': ep,
        'context_par': cp,
        'tensor_par_net': 0,
        'pipeline_par_net': 0,
        'data_par_net': 0,
        'expert_par_net': 0,
        'context_par_net': 0,
        'batch_size': 32,
        'microbatch_size': 1,
        'datatype': 'float16',
        'fused_activation': False,
        'attention_type': 'multihead',
        'activation_recompute': 'none',
        'pipeline_interleaving': 1,
        'optimizer_sharding': False,
        'tensor_par_comm_type': 'ar',
        'tensor_par_overlap': 'none',
        'seq_par_ag_redo': False,
        'data_par_overlap': False,
        'weight_offload': False,
        'activations_offload': False,
        'optimizer_offload': False,
        'training': True,
    })


def run_case(name, num_procs, tp, pp, dp, ep, cp):
    app = make_app()
    syst = make_system(num_procs)
    exe = make_exe(num_procs, tp, pp, dp, ep, cp)
    model = Llm(app, log)
    model.compile(syst, exe)
    model.run(syst)
    res = Runner.get_simulator_res_json(model)
    comm = res['communication']
    summary = res['summary']
    print(f"\n=== {name}: tp={tp} pp={pp} dp={dp} ep={ep} cp={cp} ===")
    print(f"  ep_comm_fw_size={comm['ep_comm_fw_size']}  ep_comm_bw_size={comm['ep_comm_bw_size']}")
    print(f"  cp_comm_fw_size={comm['cp_comm_fw_size']}  cp_comm_bw_size={comm['cp_comm_bw_size']}")
    print(f"  batch_ep_comm_time={comm['batch_ep_comm_time']:.6f}s "
          f"(fw={comm['batch_ep_fw_comm_time']:.6f} bw={comm['batch_ep_bw_comm_time']:.6f})")
    print(f"  batch_cp_comm_time={comm['batch_cp_comm_time']:.6f}s "
          f"(fw={comm['batch_cp_fw_comm_time']:.6f} bw={comm['batch_cp_bw_comm_time']:.6f})")
    print(f"  total_comm_time={comm['total_comm_time']:.6f}s  "
          f"batch_total_time={summary['batch_total_time']:.6f}s")
    return comm


def main():
    app = make_app()
    total_p = app.num_parameters()
    act_p = app.num_activated_parameters()
    print(f"DeepSeek-V3 params: total={total_p/1e9:.1f}B activated={act_p/1e9:.1f}B "
          f"moe_blocks={app.num_moe_blocks}")
    # calculon 的 4h^2 attention 惯例对 MLA 有高估，故给区间断言
    assert 600e9 < total_p < 750e9, 'total params should be ~671B'
    assert 35e9 < act_p < 60e9, 'activated params should be ~37B (calculon attn convention overestimates MLA)'

    # 16 procs baseline: no EP/CP -> EP/CP times must be zero
    base = run_case('baseline', 16, tp=2, pp=1, dp=8, ep=1, cp=1)
    assert base['batch_ep_comm_time'] == 0, 'baseline EP must be 0'
    assert base['batch_cp_comm_time'] == 0, 'baseline CP must be 0'

    # 16 procs with EP=4 (MoE) -> EP comm > 0
    ep_case = run_case('ep4', 16, tp=2, pp=1, dp=2, ep=4, cp=1)
    assert ep_case['batch_ep_comm_time'] > 0, 'EP comm time must be positive'
    assert ep_case['batch_ep_bw_comm_time'] >= ep_case['batch_ep_fw_comm_time']

    # 16 procs with CP=2 -> CP comm > 0
    cp_case = run_case('cp2', 16, tp=2, pp=1, dp=4, ep=1, cp=2)
    assert cp_case['batch_cp_comm_time'] > 0, 'CP comm time must be positive'

    # full combo TP+EP+CP
    combo = run_case('tp2_ep4_cp2', 16, tp=2, pp=1, dp=1, ep=4, cp=2)
    assert combo['batch_ep_comm_time'] > 0 and combo['batch_cp_comm_time'] > 0

    # EP=1 on MoE model must be zero (no a2a without expert parallelism)
    moe_no_ep = run_case('moe_ep1', 16, tp=2, pp=1, dp=8, ep=1, cp=1)
    assert moe_no_ep['batch_ep_comm_time'] == 0

    print("\nALL E2E CHECKS PASSED")


if __name__ == '__main__':
    main()
