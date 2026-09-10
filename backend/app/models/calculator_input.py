from typing import List, Optional, Literal

from pydantic import BaseModel, ConfigDict


class Gpu(BaseModel):
    name: Optional[str] = None
    sparse_tensor_fp16_processing_power: Optional[float] = None
    sparse_tensor_fp32_processing_power: Optional[float] = None
    memory: Optional[float] = None
    memory_bandwidth: Optional[float] = None
    # Bandwidth fields are unidirectional GB/s. Calculon converts with `* 1e9` → Byte/s.
    bus_bandwidth: Optional[float] = None  # intra-node (NVLink / scale-up)
    intra_latency: Optional[float] = None  # seconds, exposed for design-search defaults
    inter_latency: Optional[float] = None  # seconds, exposed for design-search defaults
    network_bandwidth: Optional[float] = None  # per-GPU inter-node injection BW
    pcie_bandwidth: Optional[float] = None  # PCIe / mem2 (offload path), not used as NVLink
    support_p2p: Optional[bool] = None
    num_procs: Optional[int] = None  # GPU数量


class Network(BaseModel):
    # Optional override; prefer Gpu.network_bandwidth from systems JSON (GB/s).
    # Kept for API compatibility / Single-machine callers that pass 0.
    network_bandwidth: Optional[float] = None
    network_topology: Optional[str] = None  # 网络拓扑类型
    scale_up_size: Optional[int] = None
    intra_latency: Optional[float] = None
    inter_latency: Optional[float] = None


class Model(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    name: Optional[str] = None
    model_family: Optional[str] = None
    hf_model_id: Optional[str] = None
    config_source: Optional[str] = None
    seq_size: Optional[int] = None
    max_position_embeddings: Optional[int] = None
    hidden: Optional[int] = None
    feedforward: Optional[int] = None
    attn_heads: Optional[int] = None
    kv_heads: Optional[int] = None          # GQA KV heads; defaults to attn_heads
    attn_size: Optional[int] = None
    rope_theta: Optional[float] = None      # RoPE frequency base
    position_embedding_type: Optional[str] = None
    rms_norm: Optional[bool] = None
    norm_epsilon: Optional[float] = None
    qk_norm: Optional[bool] = None
    ffn_type: Optional[str] = None          # gelu | relu | swiglu | geglu
    untied_embeddings: Optional[bool] = None
    attention_bias: Optional[bool] = None
    mlp_bias: Optional[bool] = None
    parallel_block: Optional[bool] = None
    architecture_approximation: Optional[bool] = None
    architecture_notes: Optional[str] = None
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
    moe_layer_offset: Optional[int] = None
    mlp_only_layers: Optional[List[int]] = None
    router_score_func: Optional[str] = None
    router_topk_method: Optional[str] = None
    router_n_groups: Optional[int] = None
    router_topk_groups: Optional[int] = None
    routed_scaling_factor: Optional[float] = None
    router_has_bias: Optional[bool] = None
    kv_size: Optional[int] = None            # CP KV 维度（MLA: kv_lora_rank + qk_rope_head_dim）
    # MLA 字段
    q_lora_rank: Optional[int] = None
    kv_lora_rank: Optional[int] = None
    qk_nope_head_dim: Optional[int] = None
    qk_rope_head_dim: Optional[int] = None
    v_head_dim: Optional[int] = None
    mla_attn_impl: Optional[str] = None
    num_nextn_predict_layers: Optional[int] = None
    include_mtp: Optional[bool] = None


class TrainningConfig(BaseModel):
    attention_kernel: Literal["flash", "unfused"] = "flash"
    optimization_strategy: Optional[str] = None  # 兼容旧前端；优先用 activation_recompute
    activation_recompute: Optional[str] = None  # full | attn_only | none
    optimizer_sharding: bool = False  # Megatron --use-distributed-optimizer
    use_precision_aware_optimizer: bool = False
    # Precision-aware optimizer is a single preset, not independently tunable
    # state dtypes: BF16 main grads/moments, FP16 main params, and BF16 reduce.
    main_grads_dtype: Literal["fp32", "bf16"] = "fp32"
    main_params_dtype: Literal["fp32", "fp16"] = "fp32"
    exp_avg_dtype: Literal["fp32", "fp16", "bf16", "fp8"] = "fp32"
    exp_avg_sq_dtype: Literal["fp32", "fp16", "bf16", "fp8"] = "fp32"
    grad_reduce_in_bf16: bool = False
    sequence_parallel: bool = False
    optimizer_offload: bool = False
    optimizer_offload_fraction: float = 1.0
    use_torch_optimizer_for_cpu_offload: bool = False
    overlap_cpu_optimizer_d2h_h2d: bool = False
    pin_cpu_grads: bool = True
    pin_cpu_params: bool = True
    tensor_par: Optional[int] = None
    pipeline_par: Optional[int] = None
    data_par: Optional[int] = None
    expert_par: Optional[int] = None   # EP，MoE only
    expert_tensor_par: Optional[int] = None  # ETP; Megatron defaults to TP
    expert_data_par: Optional[int] = None  # EDP; derived from world/(ETP*EP*PP)
    context_par: Optional[int] = None  # 上下文并行度，缺省 1
    batch_size: Optional[int] = None
    microbatch_size: Optional[int] = None
    matrix_dtype: Optional[str] = None  # GEMM 精度（systems JSON matrix.*）
    vector_dtype: Optional[str] = None  # Norm / Softmax / Act 精度（systems JSON vector.*）
    datatype: Optional[str] = None  # 兼容旧前端；若提供则作为 matrix_dtype 回退


class OptimalConfig(BaseModel):
    num_procs: Optional[int] = None  # 优化策略
    max_batch_size: Optional[int] = None
    global_batch_size: Optional[int] = None
    matrix_dtype: Optional[str] = None
    vector_dtype: Optional[str] = None
    datatype: Optional[str] = None  # 兼容旧前端；若提供则作为 matrix_dtype 回退
    objective: Optional[str] = "throughput"  # legacy single-objective input
    objectives: Optional[List[str]] = None  # throughput | batch_time | mfu | time_to_train
    training_samples: Optional[float] = None
    scale_up_size: Optional[int] = None
    placement_policies: Optional[List[str]] = None
    top_n: Optional[int] = 1
    max_candidates: Optional[int] = 512
    max_global_batch_size: Optional[int] = None
    activation_recompute_options: Optional[List[str]] = None
    optimizer_sharding_options: Optional[List[bool]] = None
    tensor_par_comm_types: Optional[List[str]] = None
    tensor_par_overlap_options: Optional[List[str]] = None
    data_par_overlap_options: Optional[List[bool]] = None
    weight_offload_options: Optional[List[bool]] = None
    activations_offload_options: Optional[List[bool]] = None
    optimizer_offload_options: Optional[List[bool]] = None


class HardwareDesignConfig(OptimalConfig):
    """Hardware/software co-design search space for Superpod Mode.

    Lists are explicit design points. The frontend can construct them from a
    start/stop/step form without coupling the backend to UI conventions.
    """
    gpu_numbers: Optional[List[int]] = None  # legacy: must equal [num_procs]
    max_num_procs: Optional[int] = None  # deprecated: count is fixed by num_procs
    scale_up_sizes: Optional[List[int]] = None
    max_scale_up_size: Optional[int] = None
    intra_bandwidths: Optional[List[float]] = None
    inter_pod_bandwidths: Optional[List[float]] = None  # aggregate GB/s per scale-up pod
    inter_bandwidths: Optional[List[float]] = None  # legacy per-GPU GB/s input
    intra_latencies: Optional[List[float]] = None
    inter_latencies: Optional[List[float]] = None
    network_topologies: Optional[List[str]] = None
    hardware_sampling: Optional[str] = "cartesian"  # cartesian | one_at_a_time | balanced
    hardware_top_n: Optional[int] = 10
    software_top_k: Optional[int] = 4
    activations_offload_options: Optional[List[bool]] = [False, True]
    coarse_top_k: Optional[int] = 128
    adaptive_refinement_points: Optional[int] = 8
    oom_fallback_candidates: Optional[int] = 256
    global_hardware_samples: Optional[int] = 12
    gpu_compute_factors: Optional[List[float]] = None
    gpu_memory_capacity_factors: Optional[List[float]] = None
    gpu_memory_bandwidth_factors: Optional[List[float]] = None


class OtherConfig(BaseModel):
    tensor_parallel_degree: Optional[int] = None
    pipeline_parallel_degree: Optional[int] = None
    microbatch_size: Optional[int] = None
    optimization_strategy: Optional[str] = None


class InputConfig(BaseModel):
    data_parallel_degree: Optional[int] = None
    number_of_input_tokens: Optional[int] = None  # 单位为M
    epochs: Optional[int] = None
