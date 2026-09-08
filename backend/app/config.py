import json
import os
from pathlib import Path
from typing import List
from pydantic import AnyHttpUrl
from pydantic_settings import BaseSettings

from app.models.calculator_input import Gpu, Model



_MODEL_DIR = Path(__file__).resolve().parents[2] / "models"
_MODEL_CATALOG = (
    ("GPT-2 124M", "gpt2-124m.json"),
    ("Qwen3-0.6B", "qwen3-0.6b.json"),
    ("Qwen3-1.7B", "qwen3-1.7b.json"),
    ("Qwen3-4B", "qwen3-4b.json"),
    ("Qwen3-8B", "qwen3-8b.json"),
    ("Qwen3-14B", "qwen3-14b.json"),
    ("Qwen3-32B", "qwen3-32b.json"),
    ("Qwen3-30B-A3B", "qwen3-30b-a3b.json"),
    ("DeepSeek-V3 671B", "deepseek-v3-671b.json"),
    ("DeepSeek-V4-tiny 2.7B", "deepseek-v4-tiny-2.7b.json"),
    ("DeepSeek-V2-Lite", "deepseek-v2-lite.json"),
    ("DeepSeek-Coder-V2-Lite", "deepseek-coder-v2-lite.json"),
    ("Megatron-GPT2 345M", "gpt2-345M.json"),
    ("GPT-3 Small", "gpt3-small.json"),
    ("GPT-3 Medium", "gpt3-medium.json"),
    ("GPT-3 Large", "gpt3-large.json"),
    ("GPT-3 XL", "gpt3-xl.json"),
    ("GPT-3 2.7B", "gpt3-2.7b.json"),
    ("GPT-3 6.7B", "gpt3-6.7b.json"),
    ("GPT-3 13B", "gpt3-13B.json"),
    ("GPT-3 175B", "gpt3-175B.json"),
    ("LLaMA-7B", "llama-7b.json"),
    ("LLaMA-13B", "llama-13b.json"),
    ("LLaMA-65B", "llama-65b.json"),
    ("LLaMA2 70B", "llama2-70b.json"),
    ("Anthropic 52B preset", "anthropic-52B.json"),
    ("BERT 6.7B preset", "bert-6.7B.json"),
    ("BERT 15B preset", "bert-15B.json"),
    ("Chinchilla preset", "chinchilla.json"),
    ("Gopher 280B preset", "gopher-280B.json"),
    ("LaMDA preset", "lamda.json"),
    ("PaLM 540B preset", "palm-540B.json"),
    ("Turing-NLG 530B preset", "turing-530B.json"),
    ("Megatron 126M preset", "megatron-126M.json"),
    ("Megatron 5B preset", "megatron-5B.json"),
    ("Megatron 22B preset", "megatron-22B.json"),
    ("Megatron 40B preset", "megatron-40B.json"),
    ("Megatron 1T preset", "megatron-1T.json"),
)

def _load_model_catalog() -> List[Model]:
    models = []
    for display_name, filename in _MODEL_CATALOG:
        with (_MODEL_DIR / filename).open(encoding="utf-8") as stream:
            models.append(Model(name=display_name, **json.load(stream)))
    return models

_SYSTEMS_DIR = Path(__file__).resolve().parents[2] / "systems"

def _load_gpu_catalog() -> List[Gpu]:
    """Build the UI GPU catalog directly from Calculon system definitions."""
    catalog = []
    for system_path in sorted(_SYSTEMS_DIR.glob("*.json"), key=lambda path: path.name.lower()):
        with system_path.open(encoding="utf-8") as stream:
            system = json.load(stream)

        matrix = system.get("matrix") or {}
        mem1 = system.get("mem1") or {}
        mem2 = system.get("mem2") or {}
        networks = system.get("networks") or []
        if not isinstance(networks, list) or len(networks) < 2:
            raise ValueError(f"{system_path} must define intra- and inter-node networks")

        intra_network, inter_network = networks[:2]
        catalog.append(Gpu(
            name=system_path.stem,
            sparse_tensor_fp16_processing_power=(matrix.get("float16") or {}).get("tflops"),
            sparse_tensor_fp32_processing_power=(matrix.get("float32") or {}).get("tflops"),
            memory=mem1.get("GiB"),
            memory_bandwidth=mem1.get("GBps"),
            bus_bandwidth=intra_network.get("bandwidth"),
            intra_latency=intra_network.get("latency"),
            network_bandwidth=inter_network.get("bandwidth"),
            inter_latency=inter_network.get("latency"),
            pcie_bandwidth=mem2.get("GBps"),
            support_p2p=bool((intra_network.get("ops") or {}).get("p2p")),
        ))
    return catalog

class Settings(BaseSettings):
    PROJECT_NAME: str = "llm-training-calculator"
    BACKEND_CORS_ORIGINS: List[AnyHttpUrl] = ["http://localhost:8080", "https://localhost:8080", "http://localhost",
                                              "https://localhost"]
    API_V1_STR: str = "/api/v1"

    CALCULATOR_RESULT_TEMPLATE: str = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                                   "template.xlsx")

    # Single source of truth: the UI catalog is derived from systems/*.json.
    GPU_LIST: List[Gpu] = _load_gpu_catalog()

    # Single source of truth: UI and simulator consume calculon/models/*.json.
    MODEL_LIST: List[Model] = _load_model_catalog()

    class Config:
        case_sensitive = False
        env_file = ".env"


settings = Settings()
