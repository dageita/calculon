import os
from typing import List
from pydantic import AnyHttpUrl
from pydantic_settings import BaseSettings

from app.models.calculator_input import Cluster, Model


class Settings(BaseSettings):
    PROJECT_NAME: str = "llm-training-calculator"
    BACKEND_CORS_ORIGINS: List[AnyHttpUrl] = ["http://localhost:8080", "https://localhost:8080", "http://localhost",
                                              "https://localhost"]
    API_V1_STR: str = "/api/v1"

    CALCULATOR_RESULT_TEMPLATE: str = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                                   "template.xlsx")

    GPU_LIST: List[Cluster] = [
        Cluster(
            name="A800_80GB_SXM",
            sparse_tensor_fp16_processing_power=624,
            sparse_tensor_fp32_processing_power=312,
            memory=80,
            memory_bandwidth=2039,
            bus_bandwidth=200,
            support_p2p=True,
        ),
        Cluster(
            name="H100_80G_SXM",
            sparse_tensor_fp16_processing_power=1979,
            sparse_tensor_fp32_processing_power=989,
            memory=80,
            memory_bandwidth=3350,
            bus_bandwidth=400,
            support_p2p=True,
        ),
        Cluster(
            name="A100_80G_NVLINK",
            sparse_tensor_fp16_processing_power=624,
            sparse_tensor_fp32_processing_power=312,
            memory=80,
            memory_bandwidth=2039,
            bus_bandwidth=300,
            support_p2p=True,
        ),
        Cluster(
            name="H20",
            sparse_tensor_fp16_processing_power=148,
            sparse_tensor_fp32_processing_power=74,
            memory=96,
            memory_bandwidth=4000,
            bus_bandwidth=64,
            support_p2p=True,
        ),
        Cluster(
            name="L20",
            sparse_tensor_fp16_processing_power=119.5,
            sparse_tensor_fp32_processing_power=59.8,
            memory=48,
            memory_bandwidth=864,
            bus_bandwidth=32,
            support_p2p=True,
        ),
    ]

    MODEL_LIST: List[Model] = [
        Model(
            name="LLAMA-7B",
            token_length=4096,
            num_attention_heads=32,
            hidden_layer_size=4096,
            num_layers=32,
            vocab_size=32000,
            minibatch_size=0,
        ),
        Model(
            name="LLAMA-13B",
            token_length=4096,
            num_attention_heads=40,
            hidden_layer_size=5120,
            num_layers=40,
            vocab_size=32000,
            minibatch_size=0,
        ),
        Model(
            name="LLAMA-32B",
            token_length=4096,
            num_attention_heads=52,
            hidden_layer_size=6656,
            num_layers=60,
            vocab_size=32000,
            minibatch_size=0,
        ),
        Model(
            name="LLAMA-65B",
            token_length=4096,
            num_attention_heads=64,
            hidden_layer_size=8192,
            num_layers=80,
            vocab_size=32000,
            minibatch_size=0,
        ),
        Model(
            name="LLaMA2 70B",
            token_length=4096,
            num_attention_heads=64,
            hidden_layer_size=8192,
            num_layers=80,
            vocab_size=32000,
            minibatch_size=0,
        ),
        Model(
            name="Gpt3",
            token_length=2048,
            num_attention_heads=96,
            hidden_layer_size=12288,
            num_layers=96,
            vocab_size=50257,
            minibatch_size=0,
        ),
        Model(
            name="GPT-3 Small",
            token_length=2048,
            num_attention_heads=12,
            hidden_layer_size=768,
            num_layers=12,
            vocab_size=50257,
            minibatch_size=0,
        ),
        Model(
            name="GPT-3 Medium",
            token_length=2048,
            num_attention_heads=16,
            hidden_layer_size=1024,
            num_layers=16,
            vocab_size=50257,
            minibatch_size=0,
        ),
        Model(
            name="GPT-3 Large",
            token_length=2048,
            num_attention_heads=16,
            hidden_layer_size=1536,
            num_layers=16,
            vocab_size=50257,
            minibatch_size=0,
        ),
        Model(
            name="GPT-3 XL",
            token_length=2048,
            num_attention_heads=24,
            hidden_layer_size=2048,
            num_layers=24,
            vocab_size=50257,
            minibatch_size=0,
        ),
        Model(
            name="GPT-32.7B",
            token_length=2048,
            num_attention_heads=32,
            hidden_layer_size=2560,
            num_layers=32,
            vocab_size=50257,
            minibatch_size=0,
        ),
        Model(
            name="GPT-3 6.7B",
            token_length=2048,
            num_attention_heads=32,
            hidden_layer_size=4096,
            num_layers=32,
            vocab_size=50257,
            minibatch_size=0,
        ),
        Model(
            name="GPT-3 13B",
            token_length=2048,
            num_attention_heads=40,
            hidden_layer_size=5140,
            num_layers=40,
            vocab_size=50257,
            minibatch_size=0,
        ),
        Model(
            name="GPT-3 175B",
            token_length=2048,
            num_attention_heads=96,
            hidden_layer_size=12288,
            num_layers=96,
            vocab_size=50257,
            minibatch_size=0,
        ),
    ]

    class Config:
        case_sensitive = False
        env_file = ".env"


settings = Settings()
