from pydantic import BaseModel


class Gpu(BaseModel):
    name: str = None
    sparse_tensor_fp16_processing_power: float = None
    sparse_tensor_fp32_processing_power: float = None
    memory: int = None
    memory_bandwidth: int = None
    bus_bandwidth: int = None  # 单向
    support_p2p: bool = None
    num_procs: int = None  # GPU数量

class Network(BaseModel):
    network_bandwidth: float = None
    network_topology: str = None  # 网络拓扑类型列表

class Model(BaseModel):
    name: str = None
    seq_size: int = None
    hidden: int = None
    feedforward: int = None
    attn_heads: int = None
    attn_size: int = None
    num_blocks: int = None
    vocab_size: int = None

class TrainningConfig(BaseModel):
    optimization_strategy: str = None  # 优化策略
    tensor_par: int = None
    pipeline_par: int = None
    data_par: int = None
    batch_size: int = None
    microbatch_size: int = None
    datatype: str = None  # 数据类型

class OptimalConfig(BaseModel):
    num_procs: int = None  # 优化策略
    max_batch_size: int = None
    datatype: str = None

class OtherConfig(BaseModel):
    tensor_parallel_degree: int = None
    pipeline_parallel_degree: int = None
    microbatch_size: int = None
    optimization_strategy: str = None

class InputConfig(BaseModel):
    data_parallel_degree: int = None
    number_of_input_tokens: int = None  # 单位为M
    epochs: int = None
