"""
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *  https://www.apache.org/licenses/LICENSE-2.0
 *
 * See the NOTICE file distributed with this work for additional information
 * regarding copyright ownership.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
"""

from calculon import *


class Layer:
  """
  A single layer of a neural network. Has weights, activation space,
  gradients, and optimizer state associated with it. May invoke compute,
  memory access, or network operation.
  """

  def __init__(self, name, sys, fw_flops=0, agrad_flops=0, wgrad_flops=0,
               inputs_size=0, output_size=0, activation_space=0,
               activation_grads=0, weight_space=0, weight_grads=0,
               optim_space=0, needs_recompute=False, needs_recomm=False,
               activation_reused=False, activation_stored=True,
               output_stored=True):
    self.name = name
    self.sys = sys
    self.fw_flops = fw_flops
    self.agrad_flops = agrad_flops
    self.wgrad_flops = wgrad_flops
    self.inputs_size = inputs_size
    self.output_size = output_size
    # activations equal input size, we store them to compute Wgrad during BW
    self.activation_space = activation_space
    # activation grads equal output size and correspond grads w.r.t. the output
    self.activation_grads = activation_grads
    self.weight_space = weight_space
    self.weight_grads = weight_grads
    self.optim_space = optim_space
    self.optim_sharding_num_proc = 1

    # Add optimizations and parallelization split
    self.needs_recompute = needs_recompute
    self.needs_recomm = needs_recomm
    self.activation_reused=activation_reused
    self.activation_stored = activation_stored
    self.output_stored = output_stored
    # Before bytes_per_element set by SW config, we operate with just
    # parameter count, setting bytes_per_element to 1
    self.bytes_per_element = 1
    self.processing_time = None
    self.net_exposed_time = None

  def get_stats_json(self):
    return {
      'name': self.name,
      'inputs_size': self.inputs_size,
      'outputs_size': self.output_size,
      'fw_flops': self.get_fw_flops(),
      'fw_mem_accessed': self.get_fw_mem_accessed(),
      'fw_arithmetic_intensity': self.get_fw_arithmetic_intensity(),
      'fw_processing_time': self.compute_processing_time('fw'),
      'baseblock_fw_tp_comm_tile': self.get_comm_tile('fw', baseblock=True),
      'edgeblock_fw_tp_comm_tile': self.get_comm_tile('fw', baseblock=False),
      'baseblock_fw_tp_comm_size': self.get_comm_bytes('fw', baseblock=True),
      'edgeblock_fw_tp_comm_size': self.get_comm_bytes('fw', baseblock=False),
      'baseblock_fw_tp_comm_time': self.compute_net_time('fw', baseblock=True),
      'edgeblock_fw_tp_comm_time': self.compute_net_time('fw',baseblock=False),
      'baseblock_fw_tp_comm_time_exposed': self.get_exposed_net_time(
        'fw', baseblock=True),
      'edgeblock_fw_tp_comm_time_exposed': self.get_exposed_net_time(
        'fw', baseblock=False),
      'agrad_flops': self.get_agrad_flops(),
      'agrad_mem_accessed': self.get_agrad_mem_accessed(),
      'agrad_arithmetic_intensity': self.get_agrad_arithmetic_intensity(),
      'agrad_processing_time': self.compute_processing_time('agrad'),
      'baseblock_bw_tp_comm_tile': self.get_comm_tile('agrad', baseblock=True),
      'edgeblock_bw_tp_comm_tile': self.get_comm_tile('agrad', baseblock=False),
      'baseblock_bw_tp_comm_size': self.get_comm_bytes('agrad', baseblock=True),
      'edgeblock_bw_tp_comm_size': self.get_comm_bytes('agrad', baseblock=False),
      'baseblock_bw_tp_comm_time': self.compute_net_time('agrad', baseblock=True),
      'edgeblock_bw_tp_comm_time': self.compute_net_time('agrad', baseblock=False),
      'baseblock_bw_tp_comm_time_exposed': self.get_exposed_net_time(
        'agrad', baseblock=True),
      'edgeblock_bw_tp_comm_time_exposed': self.get_exposed_net_time(
        'agrad', baseblock=False),
      'wgrad_flops': self.get_wgrad_flops(),
      'wgrad_mem_accessed': self.get_wgrad_mem_accessed(),
      'wgrad_arithmetic_intensity': self.get_wgrad_arithmetic_intensity(),
      'wgrad_processing_time': self.compute_processing_time('wgrad'),
      'baseblock_recomm_tile': self.get_comm_tile('wgrad', baseblock=True),
      'edgeblock_recomm_tile': self.get_comm_tile('wgrad', baseblock=False),
      'baseblock_recomm_size': self.get_comm_bytes('wgrad', baseblock=True),
      'edgeblock_recomm_size': self.get_comm_bytes('wgrad', baseblock=False),
      'baseblock_recomm_time': self.compute_net_time('wgrad', baseblock=True),
      'edgeblock_recomm_time': self.compute_net_time('wgrad', baseblock=False),
      'baseblock_recomm_time_exposed': self.get_exposed_net_time(
        'wgrad', baseblock=True),
      'edgeblock_recomm_time_exposed': self.get_exposed_net_time(
        'wgrad', baseblock=False),
      'optim_flops': self.get_optim_step_flops(),
      'optim_mem_accessed': self.get_optim_step_mem_accessed(),
      'optim_arithmetic_intensity': self.get_optim_step_arithmetic_intensity(),
      'optim_processing_time': self.compute_processing_time('optim'),
      'weight': self.get_weight(),
      'activation': self.get_activation(),
      'weight_grad': self.get_weight_grad(),
      'activation_grad': self.get_activation_grad(),
      'optimizer': self.get_optimizer()
    }

  def get_stats_str(self):
    stats = "Operation {0}:\n{1} FW flops, {2} FW bytes accessed,".format(
      self.name,
      human_format(self.get_fw_flops(), 'flops'),
      human_format(self.get_fw_mem_accessed(), 'bytes'))
    stats += " FW AI: {0:.3f}\n".format(self.get_fw_arithmetic_intensity())
    stats += "{0} BW Adrad flops, {1} BW Agrad bytes accessed,".format(
      human_format(self.get_agrard_flops(), 'flops'),
      human_format(self.get_agrad_mem_accessed(), 'bytes'))
    stats += " BW Agrad AI: {0:.3f}\n".format(
      self.get_agrad_arithmetic_intensity())
    stats += "{0} BW Wdrad flops, {1} BW Wgrad bytes accessed,".format(
      human_format(self.get_wgrard_flops(), 'flops'),
      human_format(self.get_wgrad_mem_accessed(), 'bytes'))
    stats += " BW Wgrad AI: {0:.3f}\n".format(
      self.get_wgrad_arithmetic_intensity())
    stats += "{0} Optim flops, {1} Optim bytes accessed,".format(
      human_format(self.get_optim_step_flops(), 'flops'),
      human_format(self.get_optim_step_mem_accessed(), 'bytes'))
    stats += " Optim AI: {0:.3f}\n".format(
      self.get_optim_step_arithmetic_intensity())
    stats += "W: {0}, Act: {1}, WGrad: {2}, AGrad: {3}, Optim: {4}".format(
      human_format(self.get_weight(), 'bytes'),
      human_format(self.get_activation(), 'bytes'),
      human_format(self.get_weight_grad(), 'bytes'),
      human_format(self.get_activation_grad(), 'bytes'),
      human_format(self.get_optimizer(), 'bytes'))
    return stats

  def set_bytes_per_element(self, bytes_per_element):
    self.bytes_per_element = bytes_per_element

  # Shard (distribute) optimizer and weight grads between data parallel nodes
  def shard_optimizer(self, num_procs):
    self.optim_sharding_num_proc = num_procs

  # getters that will be called from Llm model class, can be rewritten
  def get_fw_flops(self):
    return self.fw_flops

  def get_fw_mem_accessed(self):
    mem_accessed = self.inputs_size + self.output_size + self.weight_space
    mem_accessed *= self.bytes_per_element
    return mem_accessed
  
  def get_extra_and_embedding_mem_accessed(self):
    mem_accessed = self.inputs_size
    return mem_accessed

  def get_fw_arithmetic_intensity(self):
    if self.fw_flops == 0:
      return 0
    if self.get_fw_mem_accessed() == 0:
      return float('inf')
    return self.fw_flops / self.get_fw_mem_accessed()

  def get_recompute_flag(self):
    return self.needs_recompute

  def get_recomm_flag(self):
    return self.needs_recomm

  def reuses_activation(self):
    return self.activation_reused

  def stores_activation(self):
    return self.activation_stored

  def stores_output(self):
    return self.output_stored

  def get_agrad_flops(self):
    return self.agrad_flops

  def get_agrad_mem_accessed(self):
    # activation grads equal output size and correspond grads w.r.t.
    # layer output; activations are equal to input size
    grad_mem = self.weight_space + (
      self.activation_space + self.activation_grads)
    grad_mem *= self.bytes_per_element
    return grad_mem

  def get_agrad_arithmetic_intensity(self):
    if self.agrad_flops == 0:
      return 0
    if self.get_agrad_mem_accessed() == 0:
      return float('inf')
    return self.agrad_flops / self.get_agrad_mem_accessed()

  def get_wgrad_flops(self):
    return self.wgrad_flops

  def get_wgrad_mem_accessed(self):
    if self.weight_space == 0:
      assert self.wgrad_flops == 0, \
        f"Haven't expected to see wgrad flops in layer {self.name}"
      return 0
    # activation grads equal output size and correspond grads w.r.t.
    # layer output; activations are equal to input size
    grad_mem = self.weight_grads + (
      self.activation_space + self.activation_grads)
    grad_mem *= self.bytes_per_element
    return grad_mem

  def get_wgrad_arithmetic_intensity(self):
    if self.wgrad_flops == 0:
      return 0
    if self.get_wgrad_mem_accessed() == 0:
      return float('inf')
    return self.wgrad_flops / self.get_wgrad_mem_accessed()

  # We use Adam optimizer. The amount of flops is based on the number of
  # weight grads to accommodate for possible weight_grad sharding
  # among data parallel nodes
  def get_optim_step_flops(self):
    optim_flops = self.weight_grads / self.optim_sharding_num_proc * 11
    return optim_flops

  def get_optim_step_mem_accessed(self):
    return self.get_optimizer()

  def get_optim_step_arithmetic_intensity(self):
    if self.get_optim_step_flops() == 0:
      return 0
    if self.get_optim_step_mem_accessed() == 0:
      return float('inf')
    return self.get_optim_step_flops() / self.get_optim_step_mem_accessed()

  def get_weight(self):
    return self.weight_space * self.bytes_per_element

  def get_activation(self):
    return self.activation_space * self.bytes_per_element

  def get_output(self):
    return self.output_size * self.bytes_per_element

  def get_weight_grad(self, sharded=True):
    # Keep lower precision copy of grads for mem and net transfers
    grads = self.weight_grads
    if sharded:
      # We keep grads in lower precision for communication
      grads *= self.bytes_per_element
      grads /= self.optim_sharding_num_proc
    else:
      # otherwise keep grads in 32 bit for accumulation
      grads *= 4
    return grads

  def get_activation_grad(self):
    return self.activation_grads * self.bytes_per_element

  def get_optimizer(self):
    # Keep 32-bits master copy of weights, plus both moments (m,v)
    # master copy for grads is accounted for in get_weight_grad()
    moments_size = self.optim_space * 4
    if self.bytes_per_element < 4:
      master_copy_size = self.weight_space * 4
    else:
      master_copy_size = 0
    return (master_copy_size + moments_size) / self.optim_sharding_num_proc

  def set_processing_time(self, processing_time):
    self.processing_time = processing_time

  def get_processing_time(self):
    return self.processing_time

  def use_matrix_engine(self):
    return False

  def get_comm_bytes(self, stage, baseblock=True):
    return 0

  def get_comm_tile(self, stage, baseblock=True):
    return self.get_comm_bytes(stage, baseblock)

  def compute_flops_time(self, stage):
    if stage == "fw":
      flops = self.get_fw_flops()
    elif stage == "agrad":
      flops = self.get_agrad_flops()
    elif stage == "wgrad":
      flops = self.get_wgrad_flops()
    elif stage == "optim":
      flops = self.get_optim_step_flops()
    elif stage == "extra":
      flops = 0
    else:
      raise Exception(f'Bad compute stage : {stage}')
    if flops <= 0:
      return 0
    if self.use_matrix_engine() and stage != "optim":
      throughput = self.sys.get_matrix_throughput(flops)
      t = flops / throughput if throughput > 0 else 0
      launch = getattr(self.sys, 'matrix_launch_s', 0.0) or 0.0
      return max(t, launch)
    else:
      throughput = self.sys.get_vector_throughput(flops)
      t = flops / throughput if throughput > 0 else 0
      launch = getattr(self.sys, 'vector_launch_s', 0.0) or 0.0
      return max(t, launch)

  def compute_mem_time(self, stage):
    if stage == "fw":
      mem = self.get_fw_mem_accessed()
    elif stage == "agrad":
      mem = self.get_agrad_mem_accessed()
    elif stage == "wgrad":
      mem = self.get_wgrad_mem_accessed()
    elif stage == "optim":
      mem = self.get_optim_step_mem_accessed()
    elif stage == "extra":
      mem = self.get_extra_and_embedding_mem_accessed()
    else:
      raise Exception(f'Bad compute stage : {stage}')
    if mem <= 0:
      return 0
    return mem / self.sys.get_mem1_throughput(mem)

  def compute_net_time(self, stage, baseblock=True):
    return 0

  def get_exposed_net_time(self, stage, baseblock=True):
    return 0

  def get_required_bandwidth(self, stage, baseblock=True):
    return 0

  def compute_processing_time(self, stage):
    self.processing_time =  self.sys.get_processing_time(
      self.compute_flops_time(stage),
      self.compute_mem_time(stage)
    )
    return self.processing_time

# We can factor all layers peculiarities and layer-wise optimizations by
# rewriting parent class member functions when needed
class Linear(Layer):
  def __init__(self, name, sys, batch_seq, c_in, c_out,
               needs_recompute=False, activation_reused=False,
               activation_stored=True, output_stored=True,
               weight_multiplier=1.0, flop_multiplier=1.0):
    """GEMM layer.

    weight_multiplier / flop_multiplier scale stored weights and compute
    independently — used by MoE to store all local experts while only
    charging FLOPs for the activated (topk/EP + shared) equivalent.

    flop_multiplier=0 (MLA absorb WUK/WUV): weights still reside for
    capacity/optimizer, but fw/agrad/wgrad *processing* is not charged —
    traffic is accounted in the absorb BatchMatMuls instead.
    """
    m, n, k = batch_seq, c_in, c_out
    wm, fm = float(weight_multiplier), float(flop_multiplier)
    self.weight_multiplier = wm
    self.flop_multiplier = fm
    self.batch_seq = m
    self.c_in = n
    self.c_out = k
    super().__init__(name,
                     sys,
                     fw_flops=2*m*n*k*fm,
                     agrad_flops=2*m*n*k*fm,
                     wgrad_flops=2*m*n*k*fm,
                     inputs_size=m*n,
                     output_size=m*k,
                     weight_space=n*k*wm,
                     weight_grads=n*k*wm,
                     activation_space=m*n,
                     activation_grads=m*k,
                     optim_space=2*n*k*wm,
                     needs_recompute=needs_recompute,
                     activation_reused=activation_reused,
                     activation_stored=activation_stored,
                     output_stored=output_stored)

  def use_matrix_engine(self):
    return True

  def compute_flops_time(self, stage):
    # The generic matrix_launch_s is intentionally replaced by the measured
    # shape compute time for a matching forward Linear.  Applying the global launch
    # floor here would mask the shape-specific calibration; gradients have
    # different GEMM orientations and retain the normal generic model.
    grouped_moe_time = self.sys.get_grouped_moe_time(
      self.name, self.batch_seq, self.c_in, self.c_out,
      self.weight_multiplier, self.flop_multiplier,
      self.bytes_per_element) if stage == 'fw' else 0.0
    if grouped_moe_time > 0:
      return grouped_moe_time
    workload_time = self.sys.get_parametric_operator_linear_time(
      self.batch_seq, self.c_in, self.c_out) if stage == 'fw' else 0.0
    if workload_time > 0 and self.name.startswith('AttnBlock_MLA_'):
      return workload_time
    param_time = self.sys.get_parametric_linear_time(
      self.batch_seq, self.c_in, self.c_out) if stage == 'fw' else 0.0
    if param_time > 0:
      return param_time
    op_time = self.sys.get_linear_op_time(
      self.name, self.batch_seq, self.c_in, self.c_out) if stage == 'fw' else 0.0
    if op_time > 0:
      return op_time
    shape_time = self.sys.get_linear_shape_time(
      self.batch_seq, self.c_in, self.c_out) if stage == 'fw' else 0.0
    if shape_time <= 0:
      t = super().compute_flops_time(stage)
    else:
      t = shape_time
    if stage == 'fw' and self.name == 'MlpBlock_Router':
      t *= self.sys.router_linear_time_scale
    return t

  def get_fw_mem_accessed(self):
    if self.flop_multiplier == 0:
      return 0
    return super().get_fw_mem_accessed()

  def get_agrad_mem_accessed(self):
    if self.flop_multiplier == 0:
      return 0
    return super().get_agrad_mem_accessed()

  def get_wgrad_mem_accessed(self):
    if self.flop_multiplier == 0:
      return 0
    return super().get_wgrad_mem_accessed()


class LinearOverlapped(Layer):
  def __init__(self, name, sys, batch_seq, c_in, c_out, tensor_par_comm_type,
               num_tiles, net_id, num_peers, conjugate=False,
               in_network_reduction=False, tp_overlap='pipe',
               needs_recompute=False, needs_recomm=False,
               activation_reused=False, activation_stored=True,
               output_stored=True):
    m, n, k = batch_seq, c_in, c_out
    self.tensor_par_comm_type = tensor_par_comm_type
    self.num_tiles = num_tiles
    self.net = sys.get_network(net_id)
    self.num_peers = num_peers
    self.conjugate = conjugate
    self.in_network_reduction = in_network_reduction
    self.tp_overlap = tp_overlap
    self._processed_flag = False
    if self.tensor_par_comm_type == 'rs_ag':
      if not conjugate:
        #AllGather case
        assert k % self.num_peers == 0
        # assert m % self.num_peers == 0         # this should be true for seq_par
        k = k // self.num_peers
        act_space = m * n // num_tiles
        act_grad_space = m * k
        act_net_buffer = m * n // num_tiles
        act_grad_net_buffer = 0
      else:
        # ReduceScatter case
        assert n % self.num_peers == 0
        # assert m % self.num_peers == 0         # this should be true for seq_par
        n = n // self.num_peers
        act_space = m * n
        act_grad_space = m * k // num_tiles
        act_net_buffer = 0
        act_grad_net_buffer = m * k // num_tiles
        #act_net_buffer = m * k // num_tiles
    else:
      if not conjugate:
        # AllReduce case
        assert k % self.num_peers == 0
        k = k // self.num_peers
        act_space = m * n
        act_grad_space = 0
        act_net_buffer = m * n // num_tiles
        act_grad_net_buffer = 0
      else:
        # Identityy case
        assert n % self.num_peers == 0
        n = n // self.num_peers
        act_space = 0
        act_grad_space = m * k
        act_net_buffer = 0
        act_grad_net_buffer = m * k

    super().__init__(name,
                     sys,
                     fw_flops=2*m*n*k,
                     agrad_flops=2*m*n*k,
                     wgrad_flops=2*m*n*k,
                     inputs_size=m*n,
                     output_size=m*k,
                     weight_space=n*k,
                     weight_grads=n*k,
                     activation_space=act_space, # + act_net_buffer,
                     activation_grads=act_grad_space + act_grad_net_buffer,
                     optim_space=2*n*k,
                     needs_recompute=needs_recompute,
                     needs_recomm=needs_recomm,
                     activation_reused=activation_reused,
                     activation_stored=activation_stored,
                     output_stored=output_stored)

  def use_matrix_engine(self):
    return True

  def get_comm_bytes(self, stage, baseblock=True):
    if self.num_peers == 1:
      return 0
    split_comm = (self.tensor_par_comm_type == 'rs_ag') or (
      (self.tensor_par_comm_type == 'p2p_rs_ag') and not baseblock)
    ag_comm_size = self.inputs_size * self.bytes_per_element
    ar_rs_comm_size = self.output_size * self.bytes_per_element
    if stage == 'fw':
      if self.conjugate:
        # ReduceScatter or AllReduce on FW
        return ar_rs_comm_size
      else:
        if split_comm:
          # AllGather on FW
          return ag_comm_size
        else:
          # Identity on FW
          return 0
    if stage == 'agrad':
      # Comm sizes during FW and BW pass are the same
      if not self.conjugate:
        # ReduceScatter or AllReduce on BW
        return ag_comm_size
      else:
        if split_comm:
          # AllGather on BW
          return ar_rs_comm_size
        else:
          # Identity on BW
          return 0
    if stage == 'wgrad':
      if self.needs_recomm:
        return self.get_comm_bytes('fw', baseblock)
      else:
        return 0
    if stage == 'optim':
      return 0

  def get_comm_flops(self, stage, baseblock=True):
    return self.get_comm_bytes(stage, baseblock) / self.bytes_per_element

  def get_num_tiles(self):
    return self.num_tiles

  def get_comm_tile(self, stage, baseblock=True):
    return self.get_comm_bytes(stage, baseblock) / self.get_num_tiles()

  def compute_net_time(self, stage, baseblock=True):
    if self.num_peers == 1:
      return 0
    split_comm = (self.tensor_par_comm_type == 'rs_ag') or (
      (self.tensor_par_comm_type == 'p2p_rs_ag') and not baseblock)
    if self.conjugate:
      if split_comm:
        # ReduceScatter case
        fw_comm_type = 'reduce_scatter'
        bw_comm_type = 'all_gather'
      else:
        #AllReduce case
        fw_comm_type = 'all_reduce'
        bw_comm_type = None
      if not self.in_network_reduction:
        fw_flops = self.get_comm_flops(stage, baseblock) * (
          self.num_peers - 1) / self.num_peers
        fw_flop_time = fw_flops / self.sys.get_vector_throughput(fw_flops)
      else:
        fw_flop_time = 0
      bw_flop_time = 0
    else:
      if split_comm:
        #AllGather case
        fw_comm_type = 'all_gather'
        bw_comm_type = 'reduce_scatter'
      else:
        # Identity case
        fw_comm_type = None
        bw_comm_type = 'all_reduce'
      fw_flop_time = 0
      if not self.in_network_reduction:
        bw_flops = self.get_comm_flops(stage, baseblock) * (
          self.num_peers - 1) / self.num_peers
        bw_flop_time = bw_flops / self.sys.get_vector_throughput(bw_flops)
      else:
        bw_flop_time = 0
    if stage == 'fw':
      if fw_comm_type == None:
        return 0
      else:
        fw_net_time = self.net.time(
          fw_comm_type, self.get_comm_bytes(stage, baseblock), self.num_peers)
        return fw_net_time + fw_flop_time
    if stage == 'agrad':
      if bw_comm_type == None:
        return 0
      else:
        bw_net_time = self.net.time(
          bw_comm_type, self.get_comm_bytes(stage, baseblock), self.num_peers)
        return bw_net_time + bw_flop_time
    if stage == 'wgrad':
      if self.needs_recomm and fw_comm_type:
        # AllGather Redo (RS_AG only) or full recompute
        return self.net.time(
          fw_comm_type, self.get_comm_bytes(stage, baseblock), self.num_peers)
      else:
        return 0
    if stage == 'optim':
      return 0

  def compute_processing_time(self, stage):
    flop_time = self.compute_flops_time(stage)
    flop_time_slowed = flop_time / (1 - self.net.processor_usage)
    mem_time = self.compute_mem_time(stage)
    net_time = self.compute_net_time(stage)
    compute_time = self.sys.get_processing_time(flop_time, mem_time)
    if net_time == 0:
      time = compute_time
      net_exposed_time = 0
    else:
      compute_time_slowed = self.sys.get_processing_time(
        flop_time_slowed, mem_time)
      # Tiled time computed as fraction of full time, to model high effective
      # throughput when processing many consequitive tiles
      flop_tile = flop_time / self.num_tiles
      flop_tile_slowed = flop_time_slowed / self.num_tiles
      net_tile = net_time / self.num_tiles
      compute_tile = compute_time / self.num_tiles
      compute_tile_slowed = compute_time_slowed / self.num_tiles
      overlap_inflection = net_tile - flop_tile_slowed
      # we have one exposed comm tile if tp_comm is not ring,
      # one exposed compute tile, and
      # (Proc - 1) overlapped tiles, where either compute or comm is exposed
      if overlap_inflection > 0:
        # Tcomm is larger than compute, excess is exposed
        # compute time itself is the compute + mem
        time = compute_tile + (self.num_tiles - 1) * compute_tile_slowed
        net_exposed_time = (self.num_tiles - 1) * overlap_inflection
      else:
        # Tcomm is smaller than compute and hidden, but it contributes to
        # compute slowdown due part of compute resources orchestrating comm
        time = compute_tile + (self.num_tiles - 1) * compute_tile + (
          self.num_tiles - 1) * net_tile * self.net.processor_usage
        net_exposed_time = 0
      if self.tp_overlap == 'pipe':
        # If overlap type is pipe, we need to add an exposed comm tile
        # with ring-based overlap, we have a special schedule for comm and avoid
        # sending an extra tile we have in the beginning
        net_exposed_time += net_tile
        time += net_tile
    self.processing_time = time
    self.net_exposed_time = net_exposed_time
    self._processed_flag = True
    return self.processing_time

  def get_exposed_net_time(self, stage, baseblock=True):
    # only use after calling compute_processing_time(), otherwise it's set with None
    assert self._processed_flag
    return self.net_exposed_time

  def get_required_bandwidth(self, stage, baseblock=True):
    assert self._processed_flag
    net_tile_size = self.get_comm_tile(stage, baseblock)
    flop_time = self.compute_flops_time(stage)
    flop_time_slowed = flop_time / (1 - self.net.processor_usage)
    flop_tile_slowed = flop_time_slowed / self.num_tiles
    return net_tile_size / flop_tile_slowed

class BatchMatMul(Layer):
  def __init__(self, name, sys, batch, size_a, contraction_size, size_b,
               needs_recompute=False, activation_reused=False,
               activation_stored=True, output_stored=True,
               time_scale=1.0):
    """Batched GEMM.

    Compute uses ``sys.bmm_dtype`` (default BF16 on H20) — matching DeepSeek
    MLA absorb / score ``torch.bmm``, not FP8 Linear GEMM peaks.
    time_scale multiplies flops_time only (not mem). Used to correct
    attention Score/Attn BMMs that achieve lower efficiency than Linear GEMMs
    of the same FLOP count (Phase2 G2 on H20); Absorb projections stay ~1.0.
    """
    m, n, k = size_a, contraction_size, size_b
    self.time_scale = float(time_scale)
    self.batch = int(batch)
    self.m = int(m)
    self.n = int(n)
    self.k = int(k)
    super().__init__(name,
                     sys,
                     fw_flops=batch*2*m*n*k,
                     agrad_flops=batch*2*2*m*n*k,
                     inputs_size=batch*(m*n+n*k),
                     output_size=batch*m*k,
                     activation_space=batch*(m*n+n*k),
                     activation_grads=batch*m*k,
                     needs_recompute=needs_recompute,
                     activation_reused=activation_reused,
                     activation_stored=activation_stored,
                     output_stored=output_stored)

  def use_matrix_engine(self):
    return True

  def compute_flops_time(self, stage):
    if stage == 'fw':
      param_time = self.sys.get_parametric_bmm_time(
        self.batch, self.m, self.n, self.k)
      if param_time > 0:
        return param_time
      op_time = self.sys.get_bmm_op_time(
        self.name, self.batch, self.m, self.n, self.k)
      if op_time > 0:
        return op_time
    if stage == "fw":
      flops = self.get_fw_flops()
    elif stage == "agrad":
      flops = self.get_agrad_flops()
    elif stage == "wgrad":
      flops = self.get_wgrad_flops()
    elif stage == "optim":
      flops = self.get_optim_step_flops()
    elif stage == "extra":
      flops = 0
    else:
      raise Exception(f'Bad compute stage : {stage}')
    if flops <= 0:
      return 0
    throughput = self.sys.get_bmm_throughput(flops)
    t = flops / throughput if throughput > 0 else 0
    launch = getattr(self.sys, 'matrix_launch_s', 0.0) or 0.0
    return max(t, launch) * self.time_scale

# https://kratzert.github.io/2016/02/12/understanding-the-gradient-flow-through-the-batch-normalization-layer.html
# https://cthorey.github.io./blog/2016/backpropagation/
class LayerNorm(Layer):
  """Classic LayerNorm (mean+var); keep for legacy dense GPT-style models."""

  def __init__(self, name, sys, act_size, hidden,
               needs_recompute=False, activation_reused=False,
               activation_stored=True, output_stored=True):
    super().__init__(name,
                     sys,
                     fw_flops=9*act_size,
                     agrad_flops=14*act_size,
                     wgrad_flops=7*act_size,
                     inputs_size=act_size,
                     output_size=act_size,
                     activation_space=act_size,
                     activation_grads=act_size,
                     weight_space=2*hidden,
                     weight_grads=2*hidden,
                     optim_space=2*2*hidden,
                     needs_recompute=needs_recompute,
                     activation_reused=activation_reused,
                     activation_stored=activation_stored,
                     output_stored=output_stored)


class RMSNorm(Layer):
  """RMSNorm (DeepSeek-V3 / Llama-style): single scale, ~5 ops/elem fw.

  Cheaper than classic LayerNorm (9 ops/elem). Used for MLA/MLP pre-norms.
  """
  FW_FLOPS_PER_ACT = 5
  AGRAD_FLOPS_PER_ACT = 8
  WGRAD_FLOPS_PER_ACT = 4

  def __init__(self, name, sys, act_size, hidden,
               needs_recompute=False, activation_reused=False,
               activation_stored=True, output_stored=True):
    self.hidden = int(hidden)
    super().__init__(name,
                     sys,
                     fw_flops=self.FW_FLOPS_PER_ACT * act_size,
                     agrad_flops=self.AGRAD_FLOPS_PER_ACT * act_size,
                     wgrad_flops=self.WGRAD_FLOPS_PER_ACT * act_size,
                     inputs_size=act_size,
                     output_size=act_size,
                     activation_space=act_size,
                     activation_grads=act_size,
                     weight_space=hidden,       # single scale (no bias)
                     weight_grads=hidden,
                     optim_space=2 * hidden,
                     needs_recompute=needs_recompute,
                     activation_reused=activation_reused,
                     activation_stored=activation_stored,
                     output_stored=output_stored)

  def compute_flops_time(self, stage):
    t = super().compute_flops_time(stage)
    if stage == 'fw':
      t *= self.sys.get_rmsnorm_time_scale(self.hidden)
    return t


class DropOut(Layer):
  def __init__(self, name, sys, act_size,
               needs_recompute=False, activation_reused=False,
               activation_stored=True, output_stored=True):
    super().__init__(name,
                     sys,
                     fw_flops=act_size,
                     agrad_flops=act_size,
                     inputs_size=act_size,
                     output_size=act_size,
                     activation_space=act_size,
                     activation_grads=act_size,
                     needs_recompute=needs_recompute,
                     activation_reused=activation_reused,
                     activation_stored=activation_stored,
                     output_stored=output_stored)


  # need to account for DropOut mask of bool type that takes 1 B per element
  # mask is the only DropOut activation
  def get_activation(self):
    return self.activation_space

  def get_activation_grad(self):
    return self.activation_grads

  def get_fw_mem_accessed(self):
    mask_size = self.activation_space
    mem_accessed = self.inputs_size + self.output_size
    mem_accessed *= self.bytes_per_element
    mem_accessed += mask_size
    return mem_accessed

  def get_agrad_mem_accessed(self):
    return self.get_fw_mem_accessed()


# https://mlfromscratch.com/activation-functions-explained/#/
class GeLU(Layer):
  # Unfused flop constants (approx elementwise ops / activation).
  FW_FLOPS_PER_ACT = 8
  AGRAD_FLOPS_PER_ACT = 13

  def __init__(self, name, sys, act_size,
               needs_recompute=False, activation_reused=False,
               activation_stored=True, output_stored=True,
               fused=False):
    # Fused into previous Linear epilogue: no standalone compute / traffic.
    self._fused = fused
    if fused:
      fw_flops = 0
      agrad_flops = 0
      io = 0
      eff_act_space = 0
      eff_act_grads = 0
    else:
      fw_flops = self.FW_FLOPS_PER_ACT * act_size
      agrad_flops = self.AGRAD_FLOPS_PER_ACT * act_size
      io = act_size
      eff_act_space = act_size
      eff_act_grads = act_size
    super().__init__(name, sys, fw_flops=fw_flops, agrad_flops=agrad_flops,
                     inputs_size=io, output_size=io,
                     activation_space=eff_act_space,
                     activation_grads=eff_act_grads,
                     needs_recompute=needs_recompute,
                     activation_reused=activation_reused,
                     activation_stored=activation_stored,
                     output_stored=output_stored)

  def get_agrad_mem_accessed(self):
    return self.get_fw_mem_accessed()


class SiLU(GeLU):
  """SwiGLU gate activation; cheaper than GeLU when unfused.

  With fused_activation=True (DeepSeek-V3 default path), compute is folded
  into the preceding Gate GEMM epilogue and charged as 0 here.
  """
  FW_FLOPS_PER_ACT = 4
  AGRAD_FLOPS_PER_ACT = 6


# https://automata88.medium.com/how-to-implement-the-softmax-derivative-independently-from-any-loss-function-ae6d44363a9d
class SoftMax(Layer):
  def __init__(self, name, sys, act_size,
               needs_recompute=False, activation_reused=False,
               activation_stored=True, output_stored=True,
               fused=False, time_scale=1.0):
    """Attention softmax.

    fused=True: folded into flash-attn / fused attention (no standalone time).
    time_scale: optional discount when partially fused (ignored if fused).
    """
    self._fused = bool(fused)
    self.time_scale = 0.0 if self._fused else float(time_scale)
    if self._fused:
      super().__init__(name, sys,
                       fw_flops=0, agrad_flops=0,
                       inputs_size=0, output_size=0,
                       activation_space=0, activation_grads=0,
                       needs_recompute=needs_recompute,
                       activation_reused=activation_reused,
                       activation_stored=activation_stored,
                       output_stored=output_stored)
    else:
      super().__init__(name,
                       sys,
                       fw_flops=5*act_size,
                       agrad_flops=8*act_size,
                       inputs_size=act_size,
                       output_size=act_size,
                       activation_space=act_size,
                       activation_grads=act_size,
                       needs_recompute=needs_recompute,
                       activation_reused=activation_reused,
                       activation_stored=activation_stored,
                       output_stored=output_stored)

  def get_agrad_mem_accessed(self):
    return self.get_fw_mem_accessed()

  def compute_flops_time(self, stage):
    if self._fused or self.time_scale == 0:
      return 0
    return super().compute_flops_time(stage) * self.time_scale

  def compute_mem_time(self, stage):
    if self._fused or self.time_scale == 0:
      return 0
    return super().compute_mem_time(stage) * self.time_scale


# https://explained.ai/matrix-calculus/#sec:1.4.2
class ElementWise(Layer):
  def __init__(self, name, sys, operand1, operand2,
               needs_recompute=False, activation_reused=False,
               activation_stored=True, output_stored=True,
               fused=False):
    # fused=True: SwiGLU GateUp (silu(g)*up) folded into GEMM epilogue.
    self._fused = fused
    act_size = max(operand1, operand2)
    if fused:
      super().__init__(name, sys,
                       fw_flops=0, agrad_flops=0,
                       inputs_size=0, output_size=0,
                       activation_space=0, activation_grads=0,
                       needs_recompute=needs_recompute,
                       activation_reused=activation_reused,
                       activation_stored=activation_stored,
                       output_stored=output_stored)
    else:
      super().__init__(name,
                       sys,
                       fw_flops=act_size,
                       agrad_flops=(operand1+operand2),
                       inputs_size=(operand1+operand2),
                       output_size=act_size,
                       activation_space=(operand1+operand2),
                       activation_grads=act_size,
                       needs_recompute=needs_recompute,
                       activation_reused=activation_reused,
                       activation_stored=activation_stored,
                       output_stored=output_stored)


# Splits activation on the forward pass, sums gradients on the backward
class Fork(Layer):
  def __init__(self, name, sys, act_size, num_users,
               needs_recompute=False, activation_reused=False,
               activation_stored=True, output_stored=True):
    self.num_users = num_users
    super().__init__(name,
                     sys,
                     inputs_size=act_size,
                     agrad_flops=num_users*act_size,
                     activation_space=act_size,
                     # Gradients from num_users accumulated in a single storage
                     # that's accounted in the other layers
                     # use 0 here to avoid double accounting
                     activation_grads=0,
                     needs_recompute=needs_recompute,
                     activation_reused=activation_reused,
                     activation_stored=activation_stored,
                     output_stored=output_stored)

  def get_fw_mem_accessed(self):
    return 0

  def get_agrad_mem_accessed(self):
    return self.activation_space * self.bytes_per_element * (
      self.num_users + 1)


class TPComm(Layer):

  def __init__(self, name, sys, act_size, net_id, num_peers, tensor_par_comm_type,
               conjugate=False, in_network_reduction=False,
               needs_recomm=False, activation_reused=False,
               activation_stored=True, output_stored=True):
    self.net = sys.get_network(net_id)
    self.num_peers = num_peers
    self.tensor_par_comm_type = tensor_par_comm_type
    self.comm_size = act_size
    self.conjugate = conjugate
    if self.num_peers == 1:
      fw_flops = 0
      bw_flops = 0
      in_size = 0
      out_size = 0
    else:
      if not self.conjugate:
        # FW pass Identity/AllGather, BW pass AllReduce/ReduceScatter
        fw_flops = 0
        if not in_network_reduction:
          bw_flops = act_size * (self.num_peers - 1) / self.num_peers
        else:
          bw_flops = 0
        in_size = act_size
        out_size = act_size
      else:
        # Conjugate function is opposite
        if not in_network_reduction:
          fw_flops = act_size * (self.num_peers - 1) / self.num_peers
        else:
          fw_flops = 0
        bw_flops = 0
        in_size = act_size
        out_size = act_size
    super().__init__(name,
                     sys,
                     fw_flops=fw_flops,
                     agrad_flops=bw_flops,
                     inputs_size=in_size,
                     output_size=out_size,
                     activation_space=in_size,
                     activation_grads=out_size,
                     needs_recomm=needs_recomm,
                     activation_reused=activation_reused,
                     activation_stored=activation_stored,
                     output_stored=output_stored)

  def get_activation(self):
    if self.tensor_par_comm_type == 'rs_ag':
      return self.activation_space * self.bytes_per_element / self.num_peers
    else:
      if self.conjugate:
        return self.activation_space * self.bytes_per_element
      else:
        # Identity
        return 0

  def get_fw_mem_accessed(self):
    if not self.tensor_par_comm_type == 'rs_ag' and not self.conjugate:
      # Identity
      return 0
    else:
      return super().get_fw_mem_accessed()

  def get_activation_grad(self):
    if self.tensor_par_comm_type == 'rs_ag':
      return self.activation_space * self.bytes_per_element / self.num_peers
    else:
      if not self.conjugate:
        return self.activation_grads * self.bytes_per_element
      else:
        # Identity
        return 0

  def get_agrad_mem_accessed(self):
    if not self.tensor_par_comm_type == 'rs_ag' and self.conjugate:
      # Identity
      return 0
    else:
      return super().get_agrad_mem_accessed()

  def get_comm_bytes(self, stage, baseblock=True):
    if self.num_peers == 1:
      return 0
    split_comm = (self.tensor_par_comm_type == 'rs_ag') or (
      (self.tensor_par_comm_type == 'p2p_rs_ag') and not baseblock)
    if (not split_comm and (self.conjugate and stage == 'agrad' or
        not self.conjugate and stage == 'fw')):
      # Identity FW or AllReduce BW
      return 0
    else:
      if stage == 'fw' or stage == 'agrad':
        return self.comm_size * self.bytes_per_element
      if stage == 'wgrad' and self.needs_recomm and (
          split_comm or self.conjugate):
        # with AG Redo, we need recomm both on FW pass (not self.conjugate)
        # and BW pass (self.conjugate)
        return self.comm_size * self.bytes_per_element
      else:
        # optim and wgrad stage has no comm if no ag_redo flag for RS_AG
        return 0

  def compute_net_time(self, stage, baseblock=True):
    if self.num_peers == 1:
      return 0
    split_comm = (self.tensor_par_comm_type == 'rs_ag') or (
      (self.tensor_par_comm_type == 'p2p_rs_ag') and not baseblock)
    net_compute_time = super().compute_processing_time(stage)
    if split_comm:
      if self.conjugate:
        # ReduceScatter case
        fw_net_time = self.net.time('reduce_scatter',
          self.get_comm_bytes(stage, baseblock), self.num_peers)
        bw_net_time = self.net.time('all_gather',
          self.get_comm_bytes(stage, baseblock), self.num_peers)
      else:
        #AllGather case
        fw_net_time = self.net.time('all_gather',
          self.get_comm_bytes(stage, baseblock), self.num_peers)
        bw_net_time = self.net.time('reduce_scatter',
          self.get_comm_bytes(stage, baseblock), self.num_peers)
    else:
      if self.conjugate:
        fw_net_time = self.net.time('all_reduce',
          self.get_comm_bytes(stage, baseblock), self.num_peers)
        bw_net_time = 0
      else:
        fw_net_time = 0
        bw_net_time = self.net.time('all_reduce',
          self.get_comm_bytes(stage, baseblock), self.num_peers)
    if stage == 'fw':
      return fw_net_time + net_compute_time
    elif stage == 'agrad':
      return bw_net_time + net_compute_time
    elif stage == 'wgrad':
      # with AG Redo, we need recomm both on FW pass (not self.conjugate)
      # and BW pass (self.conjugate)
      if self.needs_recomm:
        return fw_net_time + net_compute_time
      else:
        return 0
    elif stage == 'optim':
      return 0
    else:
      raise Exception(f'Bad compute stage : {stage}')
    return 0

  def get_exposed_net_time(self, stage, baseblock=True):
    # only use after calling compute_processing_time(), otherwise it's set witth None
    return self.compute_net_time(stage, baseblock)

  def compute_processing_time(self, stage):
    return 0


class RotaryEmbedding(Layer):
  """Per-element rotary position embedding for Q/K; theta sets frequencies."""
  def __init__(self, name, sys, act_size, theta, **kwargs):
    assert theta > 0
    self.theta = float(theta)
    super().__init__(
      name, sys, fw_flops=6 * act_size, agrad_flops=8 * act_size,
      inputs_size=act_size, output_size=act_size,
      activation_space=act_size, activation_grads=act_size, **kwargs)

  def get_agrad_mem_accessed(self):
    return self.get_fw_mem_accessed()


class RouterSigmoid(Layer):
  """Qwen MoE router score activation over all expert logits."""
  def __init__(self, name, sys, act_size, **kwargs):
    super().__init__(
      name, sys, fw_flops=4 * act_size, agrad_flops=6 * act_size,
      inputs_size=act_size, output_size=act_size,
      activation_space=act_size, activation_grads=act_size, **kwargs)

  def get_agrad_mem_accessed(self):
    return self.get_fw_mem_accessed()

  def compute_flops_time(self, stage):
    t = super().compute_flops_time(stage)
    if stage == 'fw':
      t *= self.sys.router_sigmoid_time_scale
    return t


class RouterTopKNormalize(Layer):
  """Top-k selection plus Qwen norm_topk_prob score renormalization."""
  def __init__(self, name, sys, tokens, topk, experts, **kwargs):
    assert 0 < topk <= experts
    self.topk = topk
    self.experts = experts
    # Read all scores to select top-k; normalize only selected scores.
    score_count = tokens * experts
    selected_count = tokens * topk
    super().__init__(
      name, sys, fw_flops=score_count + 3 * selected_count,
      agrad_flops=2 * score_count + 5 * selected_count,
      inputs_size=score_count, output_size=selected_count,
      activation_space=selected_count, activation_grads=selected_count,
      **kwargs)


class RouterAuxiliaryLoss(Layer):
  """Training-time router load-balance auxiliary loss and its gradient."""
  def __init__(self, name, sys, score_count, coefficient, **kwargs):
    assert coefficient > 0
    self.coefficient = float(coefficient)
    super().__init__(
      name, sys, fw_flops=7 * score_count, agrad_flops=10 * score_count,
      inputs_size=score_count, output_size=1,
      activation_space=0, activation_grads=score_count, **kwargs)
