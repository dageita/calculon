from typing import Optional

from pydantic import BaseModel


class Gpu(BaseModel):
    name: Optional[str] = None
    sparse_tensor_fp16_processing_power: Optional[float] = None
    sparse_tensor_fp32_processing_power: Optional[float] = None
    memory: Optional[int] = None
    memory_bandwidth: Optional[int] = None
    # Bandwidth fields are unidirectional GB/s. Calculon converts with `* 1e9` → Byte/s.
    bus_bandwidth: Optional[float] = None  # intra-node (NVLink / scale-up)
    network_bandwidth: Optional[float] = None  # inter-node (NIC / scale-out)
    pcie_bandwidth: Optional[float] = None  # PCIe / mem2 (offload path), not used as NVLink
    support_p2p: Optional[bool] = None
    num_procs: Optional[int] = None  # GPU数量


class Network(BaseModel):
    # Optional override; prefer Gpu.network_bandwidth from systems JSON (GB/s).
    # Kept for API compatibility / Single-machine callers that pass 0.
    network_bandwidth: Optional[float] = None
    network_topology: Optional[str] = None  # 网络拓扑类型


class Model(BaseModel):
    name: Optional[str] = None
    seq_size: Optional[int] = None
    hidden: Optional[int] = None
    feedforward: Optional[int] = None
    attn_heads: Optional[int] = None
    kv_heads: Optional[int] = None          # GQA KV heads; defaults to attn_heads
    attn_size: Optional[int] = None
    rope_theta: Optional[float] = None      # RoPE frequency base
    rms_norm: Optional[bool] = None
    qk_norm: Optional[bool] = None
    ffn_type: Optional[str] = None            # gelu | swiglu
    untied_embeddings: Optional[bool] = None
    num_blocks: Optional[int] = None
    vocab_size: Optional[int] = None
    # MoE 架构字段（缺省表示 dense 模型；前端对 dense 模型会传 null）
    num_experts: Optional[int] = None        # n_routed_experts
    moe_topk: Optional[int] = None           # num_experts_per_tok
    norm_topk_prob: Optional[bool] = None    # Qwen top-k score renormalization
    router_aux_loss_coef: Optional[float] = None
    num_shared_experts: Optional[int] = None
    moe_feedforward: Optional[int] = None    # moe_intermediate_size
    first_k_dense: Optional[int] = None      # first_k_dense_replace
    moe_layer_freq: Optional[int] = None
    kv_size: Optional[int] = None            # CP KV 维度（MLA: kv_lora_rank + qk_rope_head_dim）
    # MLA 字段
    q_lora_rank: Optional[int] = None
    kv_lora_rank: Optional[int] = None
    qk_nope_head_dim: Optional[int] = None
    qk_rope_head_dim: Optional[int] = None
    v_head_dim: Optional[int] = None


class TrainningConfig(BaseModel):
    optimization_strategy: Optional[str] = None  # 兼容旧前端；优先用 activation_recompute
    activation_recompute: Optional[str] = None  # full | attn_only | none
    optimizer_sharding: Optional[bool] = None  # ZeRO-1；仅 DP>1 时有效
    tensor_par: Optional[int] = None
    pipeline_par: Optional[int] = None
    data_par: Optional[int] = None
    expert_par: Optional[int] = None   # 专家并行度，缺省 1
    context_par: Optional[int] = None  # 上下文并行度，缺省 1
    batch_size: Optional[int] = None
    microbatch_size: Optional[int] = None
    matrix_dtype: Optional[str] = None  # GEMM 精度（systems JSON matrix.*）
    vector_dtype: Optional[str] = None  # Norm / Softmax / Act 精度（systems JSON vector.*）
    datatype: Optional[str] = None  # 兼容旧前端；若提供则作为 matrix_dtype 回退


class OptimalConfig(BaseModel):
    num_procs: Optional[int] = None  # 优化策略
    max_batch_size: Optional[int] = None
    matrix_dtype: Optional[str] = None
    vector_dtype: Optional[str] = None
    datatype: Optional[str] = None  # 兼容旧前端；若提供则作为 matrix_dtype 回退


class OtherConfig(BaseModel):
    tensor_parallel_degree: Optional[int] = None
    pipeline_parallel_degree: Optional[int] = None
    microbatch_size: Optional[int] = None
    optimization_strategy: Optional[str] = None


class InputConfig(BaseModel):
    data_parallel_degree: Optional[int] = None
    number_of_input_tokens: Optional[int] = None  # 单位为M
    epochs: Optional[int] = None
