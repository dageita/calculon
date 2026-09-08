"""Causal tiled attention with online softmax and recomputed backward scores."""
from .layers import Layer


class FlashAttention(Layer):
  def __init__(self, name, sys, batch, sequence, heads, kv_heads, head_dim,
               needs_recompute=False, context_partitions=1):
    self.batch, self.sequence = batch, sequence
    self.context_partitions = context_partitions
    self.heads, self.kv_heads, self.head_dim = heads, kv_heads, head_dim
    # Balanced causal CP assigns equal work; communication is modeled separately.
    pairs = batch * heads * sequence * (sequence + 1) / (2*context_partitions)
    self.pairs = pairs
    q = batch * heads * sequence * head_dim / context_partitions
    kv = batch * kv_heads * sequence * head_dim / context_partitions
    super().__init__(
      name, sys, fw_flops=4*pairs*head_dim,
      # dV, dP, dQ, dK plus QK recomputation: five GEMMs.
      agrad_flops=10*pairs*head_dim,
      inputs_size=q+2*kv, output_size=q,
      activation_space=q+2*kv, activation_grads=q,
      needs_recompute=needs_recompute)

  def use_matrix_engine(self):
    return True

  def compute_flops_time(self, stage):
    flops = self.get_fw_flops() if stage == 'fw' else (
      self.get_agrad_flops() if stage == 'agrad' else 0)
    if not flops:
      return 0.0
    return flops / self.sys.get_bmm_throughput(flops)

  def get_fw_mem_accessed(self):
    # Q/O are streamed once. A query tile rereads the preceding causal KV
    # tiles. Account for these HBM reads without materializing S*S scores.
    tiles = (self.sequence + 127)//128
    kv_reads = self.batch*self.heads*self.sequence*self.head_dim*(tiles+1)
    q_io = 2*self.batch*self.heads*self.sequence*self.head_dim
    return ((q_io+kv_reads)*self.bytes_per_element +
            4*self.batch*self.heads*self.sequence) / self.context_partitions

  def get_agrad_mem_accessed(self):
    # Two score-gradient sweeps and dQ/dK/dV outputs.
    return 2*self.get_fw_mem_accessed() + (
      self.inputs_size+self.output_size)*self.bytes_per_element

  def compute_processing_time(self, stage):
    if stage not in ('fw', 'agrad'):
      return 0.0
    return max(self.sys.matrix_launch_s,
               self.sys.get_processing_time(
                 self.compute_flops_time(stage), self.compute_mem_time(stage)))
