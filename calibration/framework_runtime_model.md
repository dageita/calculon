# Framework runtime model

This model represents host-side Megatron/PyTorch/Transformer Engine execution
work as graph events. It is not a multiplier on an iteration or a model-specific
calibration profile.

## Event sources

- Every layer stage with device work produces a logical CUDA-kernel dispatch and
  an autograd node. Tensor-parallel collectives add a dispatch event.
- Unfused vocabulary-parallel cross entropy exposes six forward and two backward
  kernels, matching its stable-max/subtract/exp/sum/normalize/loss sequence.
- Recomputed operators add their forward graph events; one explicit checkpoint synchronization is emitted for each transformer-block checkpoint boundary.
- Decoder endpoint work is counted per microbatch, independent of transformer
  depth.
- Optimizer host work is represented by the number of generic fused-Adam buckets
  derived from parameter-gradient bytes; scalar synchronization counts model
  Megatron's found-inf/loss-scale and grad-norm reads.

The event counts scale with blocks, microbatches, PP partitioning, TP collectives,
activation-checkpoint scope, vocabulary size and optimizer sharding topology. They
are therefore applicable to decoder models beyond Qwen3 and not specific to DP=4.

## System capability calibration

`calculon/calibration/calibrate_framework_runtime.py` measures five values using
standalone tensors and Transformer Engine operators only: CUDA submission, autograd-node scheduling, checkpoint
bookkeeping, scalar host synchronization, Adam bucket dispatch, and physical Linear/RMSNorm CUDA-kernel multiplicities and asynchronous host-call durations. It reads no
model, sequence length, parallelism configuration, profile trace or training
timing. The resulting values belong in a system JSON under `framework_runtime`.

The L20 values in `systems/L20.json` were measured with PyTorch 2.7.0+cu128,
CUDA 12.8 and an NVIDIA L20. Re-run the utility after changing this runtime stack.
Other systems retain a disabled zero-cost runtime model until independently calibrated.

`operator_events` distinguishes physical CUDA launches from autograd nodes. For example, on the validated L20 stack a TE Linear backward has four physical kernels but remains one logical autograd function; Calculon assigns the extra launches only to dispatch cost.

## Dual-resource scheduling

`RuntimeEventDAG` maintains an ordered Host tail and GPU-stream tail. Each Layer operator advances Host submission and enqueues its device work without serially adding the two durations. Checkpoint, `.item()`, found-inf and grad-norm boundaries advance Host to the current GPU queue tail, then charge only the independently measured synchronization-API intrinsic overhead. Full recompute replays the complete block forward subgraph behind one checkpoint boundary; selective recompute replays only its selected subgraph. The attention/FFN durations handed to `LLMFlowSimulator` are conserved to the whole-block DAG so layered PP/EP scheduling cannot duplicate checkpoint cost.

The C++ flow graph remains responsible for cross-rank PP/TP/DP/EP/CP dependencies. Its local compute nodes now carry the completion time of the generated Host/GPU sub-DAG rather than `device stage total + runtime constant`. JSON output exposes the block and iteration-boundary DAG summaries, including Host work, GPU work, queue-drain wait, intrinsic sync overhead and sync count.

## Current L20 validation

For the clean Qwen3-0.6B steady-state run (FP16, 4 GPUs, TP=1/PP=1/DP=4, selective attention recompute), Megatron measured 0.818616s and the dual-resource model predicts 0.848261s: +3.62% signed error. The former additive model predicted 0.6053s (-26.06%). No Qwen timing, path, dataset, model-name factor or parallel-strategy factor is stored in `L20.json`.

The independent MoE smoke run (DeepSeek-V4-tiny, BF16, DP=4/EP=4, sequence 128) measured 0.467567s after warmup and predicts 0.507643s: +8.57%. Regression coverage also exercises dense and MoE graphs, no/selective/full recompute, TP=2/PP=2, DP=2/CP=2 and PP=2/DP=2/EP=2 layouts.
