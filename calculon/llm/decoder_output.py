"""Decoder output operations, including Megatron's unfused FP32 vocabulary loss."""
from .layers import Layer, Linear, RMSNorm, LayerNorm


class VocabParallelCrossEntropy(Layer):
  """Stable softmax loss: max, subtract, exp, sum, normalize, backward scale.

  Each pass is a separate kernel in Megatron's unfused implementation. FLOPs
  and traffic scale with local tokens and the TP-sharded vocabulary. Tensor
  reductions exchange one FP32 scalar per token, three times in forward.
  """
  def __init__(self, sys, tokens, vocabulary, tp, net):
    self.tokens = tokens
    self.elements = tokens * vocabulary
    self.tp = tp
    self.net = net
    super().__init__('Output_CrossEntropy', sys,
                     fw_flops=5*self.elements + 2*tokens,
                     agrad_flops=self.elements + tokens,
                     activation_space=self.elements)
    self.set_bytes_per_element(4)

  def passes(self, stage):
    n, t = self.elements, self.tokens
    if stage == 'fw':
      return [(n, 4*n+4*t), (n, 8*n), (n, 8*n),
              (n, 4*n+4*t), (n, 8*n), (2*t, 12*t)]
    if stage == 'agrad':
      return [(t, 12*t), (n, 8*n)]
    return []

  def get_framework_events(self, stage):
    # Megatron's vocabulary-parallel implementation intentionally materializes
    # each stable-softmax pass rather than one opaque operator.
    count = len(self.passes(stage))
    return {'kernels': count, 'autograd_nodes': count}

  def compute_processing_time(self, stage):
    duration = 0.0
    for flops, traffic in self.passes(stage):
      dtype = ('float32' if 'float32' in self.sys.vector.supported_datatypes()
               else self.sys.vector_dtype)
      throughput = self.sys.vector.throughput(dtype, flops)
      duration += max(self.sys.vector_launch_s,
                      self.sys.get_processing_time(
                        flops/throughput,
                        traffic/self.sys.get_mem1_throughput(traffic)))
    if stage == 'fw' and self.tp > 1:
      duration += 3*self.net.collective_time(
        'all_reduce', 4*self.tokens, self.tp)
    return duration


def decoder_output_stats(model):
  """Once-per-microbatch decoder head; never multiply by transformer depth."""
  app, exe, sys = model.app, model.exe, model.sys
  tokens = exe.microbatch_size * app.seq_size // exe.context_par
  vocabulary = (app.vocab_size + exe.tensor_par - 1) // exe.tensor_par
  head = Linear('Output_Projection', sys, tokens, app.hidden, vocabulary)
  norm = (RMSNorm if app.rms_norm else LayerNorm)(
    'Output_FinalNorm', sys, tokens*app.hidden, app.hidden)
  for layer in (head, norm):
    layer.set_bytes_per_element(model._matrix_bytes_per_element)
    grad_dtype = 'bf16' if exe.grad_reduce_in_bf16 else exe.main_grads_dtype
    layer.configure_optimizer(
      grad_dtype, exe.main_params_dtype,
      exe.exp_avg_dtype, exe.exp_avg_sq_dtype)
    if exe.optimizer_sharding:
      layer.shard_optimizer(exe.data_par)
  loss = VocabParallelCrossEntropy(
    sys, tokens, vocabulary, exe.tensor_par, model._tp_net)
  bpe = model._matrix_bytes_per_element
  elements = tokens*vocabulary
  # FP16/BF16 logits -> FP32 loss, and loss gradient -> model precision.
  cast_bytes = (bpe+4)*elements if bpe != 4 else 0
  cast_time = (max(sys.vector_launch_s,
                   cast_bytes/sys.get_mem1_throughput(cast_bytes))
               if cast_bytes else 0.0)
  embedding_bytes = 2*tokens*app.hidden*bpe
  lookup = embedding_bytes/sys.get_mem1_throughput(embedding_bytes)
  fw = head.compute_processing_time('fw') + norm.compute_processing_time('fw')
  fw += loss.compute_processing_time('fw') + cast_time + lookup
  bw = sum(x.compute_processing_time(s) for x in (head, norm)
           for s in ('agrad', 'wgrad'))
  bw += loss.compute_processing_time('agrad') + cast_time
  if exe.tensor_par > 1:
    bw += model._tp_net.collective_time(
      'all_reduce', tokens*app.hidden*bpe, exe.tensor_par)
  # Untied embeddings share one rank only when PP=1. With PP>1 the input
  # embedding and output projection live on different pipeline stages, so the
  # rank-critical optimizer time and storage contain one vocabulary matrix.
  endpoint_copies = 2 if app.untied_embeddings and exe.pipeline_par == 1 else 1
  params = endpoint_copies*head.weight_space + norm.weight_space
  # Embedding lookup accumulates into a dense gradient buffer.
  grad_bytes = tokens*app.hidden*bpe*3
  bw += grad_bytes/sys.get_mem1_throughput(grad_bytes)
  optim = (endpoint_copies * head.compute_processing_time('optim') +
           norm.compute_processing_time('optim'))
  framework = {
    # Projection, norm, six/two stable-loss passes, cast/lookup/scatter, and
    # the same autograd nodes.  Optimizer buckets are handled globally.
    'forward_kernels': 1 + 1 + 6 + 2,
    'forward_autograd_nodes': 1 + 1 + 6 + 2,
    'backward_kernels': 2 + 2 + 2 + 2 + int(exe.tensor_par > 1),
    'backward_autograd_nodes': 2 + 2 + 2 + 2,
  }
  copies = endpoint_copies
  resident_gradient_bytes = (
    copies*head.get_weight_grad() + norm.get_weight_grad())
  optimizer_bytes = copies*head.get_optimizer() + norm.get_optimizer()
  optimizer_mem_accessed = (
    copies*head.get_optim_step_mem_accessed() +
    norm.get_optim_step_mem_accessed())
  optimizer_flops = (copies*head.get_optim_step_flops() +
                     norm.get_optim_step_flops())
  communication_gradient_bytes = (
    copies*head.get_weight_grad(sharded=False) +
    norm.get_weight_grad(sharded=False))
  return dict(forward=fw, backward=bw, optimizer=optim, framework=framework,
              parameters=params, weight_bytes=params*bpe,
              gradient_bytes=communication_gradient_bytes,
              resident_gradient_bytes=resident_gradient_bytes,
              optimizer_bytes=optimizer_bytes,
              optimizer_mem_accessed=optimizer_mem_accessed,
              optimizer_flops=optimizer_flops,
              activation_bytes=(4+bpe)*elements,
              projection_forward=head.compute_processing_time('fw'),
              loss_forward=loss.compute_processing_time('fw')+cast_time,
              loss_backward=loss.compute_processing_time('agrad')+cast_time)
