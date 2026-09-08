"""Print the shared Hugging Face model contract as Megatron CLI flags."""
import os
import sys
from model_catalog import application_from_hf, default_model_path, megatron_flags

if __name__ == "__main__":
    case = os.environ.get("MODEL_CASE") or (sys.argv[1] if len(sys.argv) > 1 else "qwen3_06b")
    path = os.environ.get("MODEL_PATH") or default_model_path(case)
    sequence = int(os.environ.get("SEQ_LEN", 1024))
    print("\n".join(megatron_flags(case, application_from_hf(case, path, sequence))))
