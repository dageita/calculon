from pydantic import BaseModel


class Cluster(BaseModel):
    name: str = None
    sparse_tensor_fp16_processing_power: int = None
    fp32_processing_power: int = None
    memory: int = None
    memory_bandwidth: int = None
    bus_bandwidth: int = None
    delay: int = None
    launch_msrp: int = None
    network_bandwidth: int = None


class Model(BaseModel):
    name: str = None
    token_length: int = None
    num_attention_heads: int = None
    hidden_layer_size: int = None
    num_layers: int = None
    vocab_size: int = None
    minibatch_size: int = None


class OtherConfig(BaseModel):
    tensor_parallel_degree: int = None
    pipeline_parallel_degree: int = None
    microbatch_size: int = None
    optimization_strategy: str = None


class InputConfig(BaseModel):
    data_parallel_degree: int = None
    number_of_input_tokens: int = None  # 单位为M
    epochs: int = None
