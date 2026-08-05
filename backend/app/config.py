import os
from typing import List
from pydantic import AnyHttpUrl
from pydantic_settings import BaseSettings

from app.models.calculator_input import Gpu, Model


class Settings(BaseSettings):
    PROJECT_NAME: str = "llm-training-calculator"
    BACKEND_CORS_ORIGINS: List[AnyHttpUrl] = ["http://localhost:8080", "https://localhost:8080", "http://localhost",
                                              "https://localhost"]
    API_V1_STR: str = "/api/v1"

    CALCULATOR_RESULT_TEMPLATE: str = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                                   "template.xlsx")

    # Bandwidth units: GB/s unidirectional (must match systems/<name>.json
    # networks[0]=intra, networks[1]=inter). H20: 450 GB/s NVLink, 25 GB/s NIC
    # (= 25e9 B/s); do NOT use mem2/PCIe (64 GB/s) as intra.
    GPU_LIST: List[Gpu] = [
        Gpu(
            name="A800_80GB_SXM",
            sparse_tensor_fp16_processing_power=624,
            sparse_tensor_fp32_processing_power=312,
            memory=80,
            memory_bandwidth=2039,
            bus_bandwidth=450,
            network_bandwidth=32,
            pcie_bandwidth=32,
            support_p2p=True,
        ),
        Gpu(
            name="H100_80G_SXM",
            sparse_tensor_fp16_processing_power=1979,
            sparse_tensor_fp32_processing_power=989,
            memory=80,
            memory_bandwidth=3350,
            bus_bandwidth=450,
            network_bandwidth=50,
            pcie_bandwidth=450,
            support_p2p=True,
        ),
        Gpu(
            name="A100_80G_NVLINK",
            sparse_tensor_fp16_processing_power=624,
            sparse_tensor_fp32_processing_power=312,
            memory=80,
            memory_bandwidth=2039,
            bus_bandwidth=300,
            network_bandwidth=25,
            pcie_bandwidth=32,
            support_p2p=True,
        ),
        Gpu(
            name="H20",
            sparse_tensor_fp16_processing_power=148,
            sparse_tensor_fp32_processing_power=74,
            memory=96,
            memory_bandwidth=4000,
            bus_bandwidth=450,
            network_bandwidth=25,
            pcie_bandwidth=64,
            support_p2p=True,
        ),
        Gpu(
            name="L20",
            sparse_tensor_fp16_processing_power=119.5,
            sparse_tensor_fp32_processing_power=59.8,
            memory=48,
            memory_bandwidth=864,
            bus_bandwidth=32,
            network_bandwidth=32,
            pcie_bandwidth=32,
            support_p2p=True,
        ),
    ]

    MODEL_LIST: List[Model] = [
        Model(
            name="Megatron-GPT2 345M",
            seq_size=1024,
            hidden=1024,
            feedforward=4096,
            attn_heads=16,
            attn_size=64,
            num_blocks=24,
            vocab_size=51200,
        ),
        # https://huggingface.co/minhtoan/gpt3-small-finetune-cnndaily-news/blob/main/config.json
        Model(
            name="GPT-3 Small",
            seq_size=2048,
            hidden=768,
            feedforward=3072,
            attn_heads=12,
            attn_size=64,
            num_blocks=12,
            vocab_size=50257,
        ),
        # https://huggingface.co/TurkuNLP/gpt3-finnish-medium/blob/main/config.json
        Model(
            name="GPT-3 Medium",
            seq_size=2048,
            hidden=1024,
            feedforward=4096,
            attn_heads=16,
            attn_size=64,
            num_blocks=24,
            vocab_size=131072,
        ),
        # https://huggingface.co/TurkuNLP/gpt3-finnish-large/blob/main/config.json
        Model(
            name="GPT-3 Large",
            seq_size=2048,
            hidden=1536,
            feedforward=4096,
            attn_heads=16,
            attn_size=96,
            num_blocks=24,
            vocab_size=131072,
        ),
        # https://huggingface.co/TurkuNLP/gpt3-finnish-xl/blob/main/config.json
        Model(
            name="GPT-3 XL",
            seq_size=2048,
            hidden=2064,
            feedforward=8256,
            attn_heads=24,
            attn_size=86,
            num_blocks=24,
            vocab_size=131072,
        ),
        # https://huggingface.co/cerebras/Cerebras-GPT-2.7B/blob/main/config.json
        Model(
            name="GPT-3 2.7B",
            seq_size=2048,
            hidden=2560,
            feedforward=10240,
            attn_heads=32,
            attn_size=80,
            num_blocks=32,
            vocab_size=50257,
        ),
        # https://huggingface.co/AI-Sweden-Models/gpt-sw3-6.7b/blob/main/config.json
        Model(
            name="GPT-3 6.7B",
            seq_size=2048,
            hidden=4096,
            feedforward=16384,
            attn_heads=32,
            attn_size=128,
            num_blocks=32,
            vocab_size=640000,
        ),
        # https://huggingface.co/TurkuNLP/gpt3-finnish-13B/blob/main/config.json
        Model(
            name="GPT-3 13B",
            seq_size=2048,
            hidden=5120,
            feedforward=20560,
            attn_heads=40,
            attn_size=128,
            num_blocks=40,
            vocab_size=131072,
        ),
        # https://huggingface.co/PKUFlyingPig/gpt175b-config/blob/main/config.json
        Model(
            name="GPT-3 175B",
            seq_size=2048,
            hidden=12288,
            feedforward=49152,
            attn_heads=96,
            attn_size=128,
            num_blocks=96,
            vocab_size=50257,
        ),

        # https://huggingface.co/huggyllama/llama-7b/blob/main/config.json
        Model(
            name="LLAMA-7B",
            seq_size=2048,
            hidden=4096,
            feedforward=11008,
            attn_heads=32,
            attn_size=128,
            num_blocks=32,
            vocab_size=32000,
        ),
        # https://huggingface.co/huggyllama/llama-13b/blob/main/config.json
        Model(
            name="LLAMA-13B",
            seq_size=2048,
            hidden=5120,
            feedforward=13824,
            attn_heads=40,
            attn_size=128,
            num_blocks=40,
            vocab_size=32000,
        ),
        # https://huggingface.co/huggyllama/llama-65b/blob/main/config.json
        Model(
            name="LLAMA-65B",
            seq_size=2048,
            hidden=8192,
            feedforward=22016,
            attn_heads=64,
            attn_size=128,
            num_blocks=80,
            vocab_size=32000,
        ),
        # https://www.modelscope.cn/models/modelscope/Llama-2-70b-ms/file/view/master/config.json?status=1
        Model(
            name="LLaMA2 70B",
            seq_size=4096,
            hidden=8192,
            feedforward=28672,
            attn_heads=64,
            attn_size=128,
            num_blocks=80,
            vocab_size=32000 
        ),
        # https://huggingface.co/deepseek-ai/DeepSeek-V3/blob/main/config.json
        # MoE: 61 层 = 3 dense + 58 MoE；256 路由专家 + 1 共享专家，topk=8；
        # kv_size = kv_lora_rank(512) + qk_rope_head_dim(64)（MLA 压缩 KV）。
        Model(
            name="DeepSeek-V3 671B",
            seq_size=4096,
            hidden=7168,
            feedforward=18432,
            attn_heads=128,
            attn_size=128,
            num_blocks=61,
            vocab_size=129280,
            num_experts=256,
            moe_topk=8,
            num_shared_experts=1,
            moe_feedforward=2048,
            first_k_dense=3,
            moe_layer_freq=1,
            kv_size=576,
            q_lora_rank=1536,
            kv_lora_rank=512,
            qk_nope_head_dim=128,
            qk_rope_head_dim=64,
            v_head_dim=128,
        ),

    ]

    class Config:
        case_sensitive = False
        env_file = ".env"


settings = Settings()
