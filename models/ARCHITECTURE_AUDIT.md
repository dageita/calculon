# Model architecture audit

Audited against local `/models/*/config.json`, official Hugging Face `config.json`
where available, and the original paper for Calculon scaling presets.

## Field semantics

- `seq_size`: sequence length of the simulated training workload. It is not
  `hidden_size` and it is not the checkpoint maximum context length.
- `max_position_embeddings`: checkpoint/model maximum context length.
- `hidden`, `feedforward`, `attn_heads`, `kv_heads`, `attn_size`, `num_blocks`
  map to HF `hidden_size`, `intermediate_size`, `num_attention_heads`,
  `num_key_value_heads`, `head_dim`, and `num_hidden_layers`.
- `untied_embeddings` is the inverse of HF `tie_word_embeddings`.

## Source classification

| Models | Source status |
| --- | --- |
| Qwen3 0.6B / 32B / 30B-A3B | Exact architecture fields from official Qwen HF configs |
| DeepSeek-V3 | Exact base, MLA, MoE and router fields from official DeepSeek config |
| GPT-2 Medium, Cerebras-GPT 2.7B, GPT-SW3 6.7B, TurkuNLP Finnish GPT family, LLaMA/LLaMA2 | Checkpoint or model-card architecture fields |
| GPT-3 175B | Published/config-only architecture preset |
| Anthropic, scaled BERT, GPT-3 Small, Megatron size presets, Turing-NLG | Synthetic/paper size presets; no canonical HF checkpoint config |
| Chinchilla, Gopher, LaMDA, PaLM | Paper architecture presets; marked `architecture_approximation=true` |
| DeepSeek-V4-tiny 2.7B | Local experimental config; compressed/sparse/sliding attention remains an explicit approximation |

## Dense versus MoE

Dense blocks select 2-matrix GeLU/ReLU or 3-matrix SwiGLU/GeGLU, GQA/MQA,
position encoding, norm type, projection biases, tied embeddings and optional
parallel attention/FFN blocks. MoE additionally specifies routed/shared expert
counts and widths, top-k, sparse-layer schedule, score function, grouped top-k,
correction bias and routed-score scaling. DeepSeek MLA fields are independent of
EP and are included in attention parameter/FLOP accounting.

The UI catalog is loaded from these JSON files, so the UI and simulation no
longer maintain separate architecture values.
