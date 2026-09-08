#!/usr/bin/env bash
set -euo pipefail

MODEL_CASE="${1:-gpt2}"
NUM_GPUS="${NUM_GPUS:-4}"
TP="${TP:-1}"
PP="${PP:-1}"
DP="${DP:-}"
EP="${EP:-1}"
ETP="${ETP:-$TP}"
EDP="${EDP:-}"
CP="${CP:-1}"
SEQUENCE_PARALLEL="${SEQUENCE_PARALLEL:-0}"
PRECISION="${PRECISION:-bf16}"
OPTIMIZATION_STRATEGY="${OPTIMIZATION_STRATEGY:-none}"
LR="${LR:-1e-4}"
MIN_LR="${MIN_LR:-1e-5}"
WARMUP_FRACTION="${WARMUP_FRACTION:-.01}"
WEIGHT_DECAY="${WEIGHT_DECAY:-.1}"
CLIP_GRAD="${CLIP_GRAD:-1.0}"
EXTRA_MEGATRON_ARGS="${EXTRA_MEGATRON_ARGS:-}"
ITERATIONS="${ITERATIONS:-30}"
SEQ_LEN="${SEQ_LEN:-1024}"
MICRO_BATCH="${MICRO_BATCH:-1}"
GLOBAL_BATCH="${GLOBAL_BATCH:-8}"
DATA_PATH="${DATA_PATH:-}"
OUT_DIR="${OUT_DIR:-/src/Simulator/megatron_validation/results}"
mkdir -p "$OUT_DIR"

if (( TP > 1 || CP > 1 )); then
  export CUDA_DEVICE_MAX_CONNECTIONS=1
fi

COMMON=(
  --use-mcore-models --transformer-impl transformer_engine
  --tensor-model-parallel-size "$TP" --pipeline-model-parallel-size "$PP"
  --context-parallel-size "$CP"
  --micro-batch-size "$MICRO_BATCH" --global-batch-size "$GLOBAL_BATCH"
  --seq-length "$SEQ_LEN" --max-position-embeddings "$SEQ_LEN"
  --train-iters "$ITERATIONS" --lr "$LR" --min-lr "$MIN_LR"
  --lr-decay-style cosine --lr-warmup-fraction "$WARMUP_FRACTION"
  --weight-decay "$WEIGHT_DECAY" --clip-grad "$CLIP_GRAD"
  --distributed-backend nccl --no-gradient-accumulation-fusion
  --no-rope-fusion
  --log-interval 1 --log-throughput --timing-log-level "${TIMING_LOG_LEVEL:-0}" --eval-iters 0
  --eval-interval "${EVAL_INTERVAL:-1000}"
  --no-save-optim --no-save-rng --seed 1234
)
case "$PRECISION" in
  bf16) COMMON+=(--bf16) ;;
  fp16) COMMON+=(--fp16 --initial-loss-scale "${INITIAL_LOSS_SCALE:-16384}") ;;
  *) echo "unsupported PRECISION=$PRECISION (use bf16 or fp16)" >&2; exit 2 ;;
esac
case "$OPTIMIZATION_STRATEGY" in
  none) ;;
  attention-only) COMMON+=(--recompute-granularity selective --recompute-modules core_attn) ;;
  full) COMMON+=(--recompute-granularity full --recompute-method uniform --recompute-num-layers 1) ;;
  *) echo "unsupported OPTIMIZATION_STRATEGY=$OPTIMIZATION_STRATEGY (use none, attention-only, or full)" >&2; exit 2 ;;
esac
if [[ "$SEQUENCE_PARALLEL" == "1" ]]; then
  COMMON+=(--sequence-parallel)
fi
printf "Optimization strategy: %s\n" "$OPTIMIZATION_STRATEGY"
if [[ -n "$EXTRA_MEGATRON_ARGS" ]]; then
  read -r -a EXTRA_ARGS <<< "$EXTRA_MEGATRON_ARGS"
  COMMON+=("${EXTRA_ARGS[@]}")
fi
if [[ -n "$DATA_PATH" ]]; then
  COMMON+=(--data-path "$DATA_PATH" --split 949,50,1)
else
  COMMON+=(--mock-data --tokenizer-type NullTokenizer)
fi

MODEL_FLAGS=$(MODEL_CASE="$MODEL_CASE" python /src/Simulator/calculon/calibration/l20/model_contract.py "$MODEL_CASE")
mapfile -t MODEL <<< "$MODEL_FLAGS"
if printf "%s\n" "${MODEL[@]}" | grep -qx -- "--num-experts"; then
  MODEL+=(--expert-model-parallel-size "$EP" --expert-tensor-parallel-size "$ETP" --moe-grouped-gemm)
  MOE_KERNEL_PATH="grouped-gemm"
  printf "MoE kernel path: %s (%s)\n" "$MOE_KERNEL_PATH" "$PRECISION"
fi

LOG="$OUT_DIR/${MODEL_CASE}.log"
torchrun --standalone --nproc_per_node="$NUM_GPUS" "${MEGATRON_ENTRYPOINT:-/app/Megatron-LM/pretrain_gpt.py}" "${COMMON[@]}" "${MODEL[@]}" 2>&1 | tee "$LOG"
python /src/Simulator/calculon/calibration/l20/compare_iteration_logs.py \
  --model "$MODEL_CASE" --log "$LOG" --output "$OUT_DIR/${MODEL_CASE}.json"
