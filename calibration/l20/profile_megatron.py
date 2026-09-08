"""Diagnostic entry point: CUDA/CPU events without expensive Python stack tracing.

This does not change training operators, precision, optimizer, or timing results.
Use only with Megatron --profile --use-pytorch-profiler; never for reference timing.
"""
import runpy
import sys
import torch

_original_profile = torch.profiler.profile


def lightweight_profile(*args, **kwargs):
    kwargs.update(with_stack=False, record_shapes=False, profile_memory=False)
    return _original_profile(*args, **kwargs)


if __name__ == "__main__":
    torch.profiler.profile = lightweight_profile
    sys.path.insert(0, "/app/Megatron-LM")
    runpy.run_path("/app/Megatron-LM/pretrain_gpt.py", run_name="__main__")
