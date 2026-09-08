"""Shared local-HF model contracts for Megatron, Calculon, and the UI."""
import json
from pathlib import Path

CASES = {
 "gpt2_124m": ("GPT-2 124M", "gpt2", "gpt2-124m.json"),
 "qwen3_06b": ("Qwen3-0.6B", "Qwen3-0.6B", "qwen3-0.6b.json"),
 "qwen3_17b": ("Qwen3-1.7B", "Qwen3-1.7B", "qwen3-1.7b.json"),
 "qwen3_4b": ("Qwen3-4B", "Qwen3-4B", "qwen3-4b.json"),
 "qwen3_8b": ("Qwen3-8B", "Qwen3-8B", "qwen3-8b.json"),
 "qwen3_14b": ("Qwen3-14B", "Qwen3-14B", "qwen3-14b.json"),
 "deepseek_v4_tiny": ("DeepSeek-V4-tiny 2.7B", "DeepSeek-V4-2.7B-tiny",
                       "deepseek-v4-tiny-2.7b.json"),
 "deepseek_v2_lite": ("DeepSeek-V2-Lite", "DeepSeek-V2-Lite",
                       "deepseek-v2-lite.json"),
 "deepseek_coder_v2_lite": ("DeepSeek-Coder-V2-Lite",
                             "DeepSeek-Coder-V2-Lite-Instruct",
                             "deepseek-coder-v2-lite.json"),
}
ALIASES = {"gpt2": "gpt2_124m", "moe_v4_tiny": "deepseek_v4_tiny"}
CASE_CHOICES = tuple(CASES) + tuple(ALIASES)

def canonical_case(case): return ALIASES.get(case, case)
def case_info(case): return CASES[canonical_case(case)]
def default_model_path(case): return Path("/models") / case_info(case)[1]

def _common(hf, sequence):
 return {
  "hidden": hf["hidden_size"],
  "feedforward": hf.get("intermediate_size", 4 * hf["hidden_size"]),
  "seq_size": sequence, "attn_heads": hf["num_attention_heads"],
  "num_blocks": hf["num_hidden_layers"], "vocab_size": hf["vocab_size"],
  "model_family": "decoder",
  "max_position_embeddings": hf.get("max_position_embeddings", sequence),
  "rope_theta": float(hf.get("rope_theta", 10000.0)),
  "position_embedding_type": "rope", "rms_norm": True,
  "norm_epsilon": hf.get("rms_norm_eps", 1e-6), "ffn_type": "swiglu",
  "untied_embeddings": not hf.get("tie_word_embeddings", False),
  "attention_bias": bool(hf.get("attention_bias", False)),
  "mlp_bias": bool(hf.get("mlp_bias", False)),
 }

def application_from_hf(case, model_path, sequence):
 case = canonical_case(case)
 hf = json.loads((Path(model_path) / "config.json").read_text())
 kind = hf.get("model_type")
 if case == "gpt2_124m":
  if kind != "gpt2": raise ValueError(f"{case} requires gpt2, got {kind}")
  hidden, heads = hf["n_embd"], hf["n_head"]
  return {"hidden": hidden, "feedforward": hf.get("n_inner") or 4*hidden,
   "seq_size": sequence, "attn_heads": heads, "kv_heads": heads,
   "attn_size": hidden//heads, "num_blocks": hf["n_layer"],
   "vocab_size": hf["vocab_size"], "model_family": "decoder",
   "max_position_embeddings": hf.get("n_positions", sequence),
   "position_embedding_type": "learned_absolute", "rms_norm": False,
   "norm_epsilon": hf.get("layer_norm_epsilon", 1e-5), "ffn_type": "gelu",
   "untied_embeddings": False, "attention_bias": True, "mlp_bias": True}
 if case.startswith("qwen3_"):
  if kind != "qwen3": raise ValueError(f"{case} requires qwen3, got {kind}")
  app = _common(hf, sequence)
  app.update(kv_heads=hf["num_key_value_heads"],
             attn_size=hf.get("head_dim") or
                       hf["hidden_size"]//hf["num_attention_heads"],
             qk_norm=True)
  return app
 if case in ("deepseek_v2_lite", "deepseek_coder_v2_lite"):
  if kind != "deepseek_v2":
   raise ValueError(f"{case} requires deepseek_v2, got {kind}")
  app = _common(hf, sequence)
  nope, rope = hf["qk_nope_head_dim"], hf["qk_rope_head_dim"]
  app.update(kv_heads=hf.get("num_key_value_heads", hf["num_attention_heads"]),
   attn_size=nope+rope, q_lora_rank=hf.get("q_lora_rank") or 0,
   kv_lora_rank=hf["kv_lora_rank"], qk_nope_head_dim=nope,
   qk_rope_head_dim=rope, v_head_dim=hf["v_head_dim"],
   kv_size=hf["kv_lora_rank"]+rope, mla_attn_impl="naive", qk_norm=True,
   num_experts=hf["n_routed_experts"], moe_topk=hf["num_experts_per_tok"],
   num_shared_experts=hf.get("n_shared_experts", 0),
   moe_feedforward=hf["moe_intermediate_size"],
   first_k_dense=hf.get("first_k_dense_replace", 0),
   moe_layer_freq=hf.get("moe_layer_freq", 1),
   norm_topk_prob=hf.get("norm_topk_prob", False),
   router_aux_loss_coef=hf.get("aux_loss_alpha", 0.0),
   router_score_func=hf.get("scoring_func", "softmax"),
   router_topk_method="greedy", router_n_groups=hf.get("n_group", 1),
   router_topk_groups=hf.get("topk_group", 1),
   routed_scaling_factor=hf.get("routed_scaling_factor", 1.0),
   router_has_bias=False)
  return app
 if case == "deepseek_v4_tiny":
  if kind != "deepseek_v4":
   raise ValueError(f"{case} requires deepseek_v4, got {kind}")
  app = _common(hf, sequence)
  app.update(feedforward=hf.get("intermediate_size", 10944),
   kv_heads=hf.get("num_key_value_heads", 1),
   attn_size=hf["hidden_size"]//hf["num_attention_heads"],
   num_experts=hf["n_routed_experts"], moe_topk=hf["num_experts_per_tok"],
   num_shared_experts=hf.get("n_shared_experts", 0),
   moe_feedforward=hf["moe_intermediate_size"],
   first_k_dense=hf.get("first_k_dense_replace", 0),
   moe_layer_freq=hf.get("moe_layer_freq", 1),
   norm_topk_prob=hf.get("norm_topk_prob", False),
   router_aux_loss_coef=hf.get("router_aux_loss_coef", 0.0),
   router_score_func=hf.get("scoring_func", "sqrtsoftplus"),
   router_topk_method=hf.get("topk_method", "noaux_tc"),
   router_n_groups=1, router_topk_groups=1,
   routed_scaling_factor=hf.get("routed_scaling_factor", 1.0),
   router_has_bias=True, architecture_approximation=True,
   architecture_notes="Experimental V4 compressed/sparse/sliding attention "
    "and sqrtsoftplus routing use a consistent standard GQA+MoE approximation.")
  return app
 raise ValueError(f"unsupported case: {case}")

def megatron_flags(case, app):
 case = canonical_case(case)
 flags = ["--num-layers", str(app["num_blocks"]), "--hidden-size",
  str(app["hidden"]), "--num-attention-heads", str(app["attn_heads"]),
  "--ffn-hidden-size", str(app["feedforward"]), "--vocab-size",
  str(app["vocab_size"]), "--position-embedding-type",
  app["position_embedding_type"]]
 if app["position_embedding_type"] == "learned_absolute": return flags
 flags += ["--rotary-base", str(int(app["rope_theta"])), "--swiglu",
  "--disable-bias-linear", "--normalization", "RMSNorm", "--norm-epsilon",
  str(app.get("norm_epsilon", 1e-6))]
 if app.get("kv_lora_rank"):
  flags += ["--multi-latent-attention", "--kv-lora-rank",
   str(app["kv_lora_rank"]), "--qk-head-dim", str(app["qk_nope_head_dim"]),
   "--qk-pos-emb-head-dim", str(app["qk_rope_head_dim"]), "--v-head-dim",
   str(app["v_head_dim"]), "--qk-layernorm"]
  if app.get("q_lora_rank"):
   flags += ["--q-lora-rank", str(app["q_lora_rank"])]
 else:
  flags += ["--kv-channels", str(app["attn_size"])]
  if app.get("kv_heads", app["attn_heads"]) != app["attn_heads"]:
   flags += ["--group-query-attention", "--num-query-groups",
             str(app["kv_heads"])]
  if app.get("qk_norm"): flags += ["--qk-layernorm"]
 if app.get("untied_embeddings"):
  flags += ["--untie-embeddings-and-output-weights"]
 if app.get("num_experts"):
  dense = app.get("first_k_dense", 0)
  freq = f"[0]*{dense}+[1]*{app['num_blocks']-dense}" if dense else str(
      app.get("moe_layer_freq", 1))
  score = app["router_score_func"]
  flags += ["--num-experts", str(app["num_experts"]), "--moe-layer-freq",
   freq, "--moe-router-topk", str(app["moe_topk"]),
   "--moe-ffn-hidden-size", str(app["moe_feedforward"]),
   "--moe-token-dispatcher-type", "alltoall", "--moe-router-score-function",
   "softmax" if score == "sqrtsoftplus" else score,
   "--moe-aux-loss-coeff", str(app.get("router_aux_loss_coef", 0.0))]
  shared = app.get("num_shared_experts", 0)*app["moe_feedforward"]
  if shared:
   flags += ["--moe-shared-expert-intermediate-size", str(shared),
             "--moe-shared-expert-overlap"]
  if not app.get("norm_topk_prob", False):
   flags += ["--moe-router-pre-softmax", "--moe-router-topk-scaling-factor",
             str(app.get("routed_scaling_factor", 1.0))]
  if case in ("deepseek_v2_lite", "deepseek_coder_v2_lite"):
   flags += ["--moe-router-load-balancing-type", "seq_aux_loss"]
 return flags

def sync_model_jsons(destination, sequence=1024):
 destination = Path(destination); destination.mkdir(parents=True, exist_ok=True)
 for case, (_, _, filename) in CASES.items():
  app = application_from_hf(case, default_model_path(case), sequence)
  (destination/filename).write_text(json.dumps(app, indent=2)+"\n")
