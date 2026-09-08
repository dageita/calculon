# L20 calibration and Megatron validation

Canonical calibration files live here. The legacy entry point
`megatron_validation/run_l20_auto_validation.py` remains as a compatibility
wrapper.

## Full one-command run

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python /src/Simulator/megatron_validation/run_l20_auto_validation.py \
  --gpus 8 \
  --system /src/Simulator/calculon/systems/L20.json \
  --model-path /models/Qwen3-0.6B \
  --moe-model-path /models/DeepSeek-V4-2.7B-tiny \
  --dataset-path /datasets/ShareGPT_V3_unfiltered_cleaned_split.json
```

This runs matrix and vector calibration for float8, float16, bfloat16 and
float32, model-aware small-N Linear calibration, memory/network calibration,
Megatron Dense/MoE training, Calculon simulation, comparison and residual
runtime fitting when the configured error gate is exceeded.

Use `--quick-calibration` for a smoke test. Use
`--skip-hardware-calibration` only when the selected `--system` already
contains valid measurements.

Raw JSON/JSONL datasets are recorded as provenance but cannot be passed directly
to Megatron. The validation uses mock-data unless `--dataset-path` denotes a
preprocessed prefix having both `.bin` and `.idx`; this keeps iteration timing
reproducible. `prepare_data.py` normalizes raw input before Megatron's own
`tools/preprocess_data.py` step.

## Calibration only

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
python /src/Simulator/calculon/calibration/l20/calibrate_l20.py \
  --gpus 4 --execute \
  --system /src/Simulator/calculon/systems/L20.json \
  --model-path /models/Qwen3-0.6B \
  --model-path /models/DeepSeek-V4-2.7B-tiny \
  --dataset-path /datasets/ShareGPT_V3_unfiltered_cleaned_split.json
```

Every phase receives the same explicit output JSON. Numeric lookup points are
formatted on one line, for example `[8192, 5.7633518e-05]`.

Small-N calibration is separate from the generic FLOPs curve. It only applies
on exact measured K/N buckets; other shapes safely fall back to the matrix
roofline. This avoids contaminating square-GEMM efficiency with router/GQA and
expert projection aspect-ratio effects.

## One-command Megatron vs Calculon comparison

Use run_megatron_sim_compare.py for one workload. It performs Megatron measurement, Calculon prediction and writes CASE.log, simulator-baseline.json, prediction-error.json, and summary.json under --output-dir.

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \\
python /src/Simulator/calculon/calibration/l20/run_megatron_sim_compare.py \\
  --case qwen3_06b \\
  --model-path /models/Qwen3-0.6B \\
  --dataset-path /datasets/ShareGPT_V3_unfiltered_cleaned_split.json \\
  --system /src/Simulator/calculon/systems/L20.json \\
  --tp 1 --pp 1 --cp 1 --dp 4 --ep 1 \\
  --iterations 30 --seq-length 1024 --global-batch 8 --micro-batch 1 \\
  --precision bf16 --lr 1e-4 --min-lr 1e-5 \\
  --output-dir /src/Simulator/calculon/calibration/l20/results/qwen-4gpu
```

5D semantics are world_size = TP * PP * CP * DP. EP partitions a DP group, so EP must divide DP but is not multiplied into world size. Supported architecture cases are `gpt2_124m`, `qwen3_06b`, `qwen3_17b`, `qwen3_4b`, `qwen3_8b`, `qwen3_14b`, `deepseek_v4_tiny`, `deepseek_v2_lite`, and `deepseek_coder_v2_lite`. The legacy aliases `gpt2` and `moe_v4_tiny` remain accepted.

This Megatron build does not support FP16 together with expert parallelism greater than one. MoE runs with `--ep > 1` must use `--precision bf16`; FP16 remains available with `--ep 1`, subject to model memory capacity. The comparison driver rejects the unsupported combination before launching `torchrun` and never changes the requested precision silently.

A raw JSON/JSONL dataset is accepted as provenance but the timed run uses mock-data. Pass a Megatron indexed prefix (with matching .bin and .idx) to measure the real input pipeline. Additional Megatron flags can be repeated via --extra-megatron-arg=...
