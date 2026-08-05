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

from .memory import *
from .network import *
from .processor import *

class System:
  """Configuration for a system."""

  TypeSizes = {
    'float8'   : 1,
    'float16'  : 2,
    'float32'  : 4,
    'bfloat16' : 2
  }

  @staticmethod
  def supported_datatypes():
    return list(System.TypeSizes.keys())

  def __init__(self, cfg, log=None):
    self.cfg = cfg
    self.matrix = Processor(cfg['matrix'])
    self.vector = Processor(cfg['vector'])
    # Dual compute dtypes (DeepSeek-V3 style: matrix=FP8, vector=BF16).
    self.matrix_dtype = None
    self.vector_dtype = None
    # Legacy alias: equals matrix_dtype after set_datatypes / set_datatype.
    self.datatype = None

    self.mem1 = Memory(cfg['mem1'])
    self.mem2 = Memory(cfg['mem2'])

    # Optional kernel launch floor (seconds) for matrix-engine ops.
    # Calibrated by test/calibrate_h20_matrix_efficiency.py; 0 disables.
    self.matrix_launch_s = float(cfg.get('matrix_launch_s', 0.0) or 0.0)
    self.vector_launch_s = float(cfg.get('vector_launch_s', 0.0) or 0.0)

    # BMM compute dtype (DeepSeek MLA absorb / score path uses BF16 bmm in
    # training; Linear GEMMs stay on matrix_dtype=FP8). Independent of
    # set_datatypes(); default bfloat16 when present in matrix processor.
    bmm_dtype_cfg = cfg.get('bmm_dtype') or 'bfloat16'
    self.bmm_dtype = str(bmm_dtype_cfg)

    # BMM time scales vs Linear GEMM efficiency (Phase2 G2 on H20).
    # attn_score: ScoreKV/ScorePE/AttnKV (and naive QK/AttnV); absorb: Q/V absorb.
    bmm_scales = cfg.get('bmm_time_scale') or {}
    if isinstance(bmm_scales, (int, float)):
      bmm_scales = {'default': float(bmm_scales)}
    self.bmm_time_scale = {
      'default': float(bmm_scales.get('default', 1.0)),
      'absorb': float(bmm_scales.get('absorb', bmm_scales.get('default', 1.0))),
      'attn_score': float(bmm_scales.get(
        'attn_score', bmm_scales.get('default', 1.0))),
    }

    # Attention SoftMax folded into flash-attn (DeepSeek / modern stacks).
    # When True, SoftMax fw/agrad processing is charged as 0 (known gap vs
    # isolated torch.softmax microbench).
    self.attn_softmax_fused = bool(cfg.get('attn_softmax_fused', False))
    self.attn_softmax_time_scale = float(
      cfg.get('attn_softmax_time_scale', 1.0) or 1.0)

    self.proc_mode = cfg['processing_mode']
    assert self.proc_mode in ['roofline', 'no_overlap']

    self.networks = [Network(n, log) for n in cfg['networks']]

  def get_bmm_dtype(self):
    """Dtype used for BatchMatMul matrix throughput (and BMM mem width).

    Falls back to matrix_dtype if bmm_dtype is missing from the processor.
    ``_bmm_dtype_override`` (set by Phase2 microbench) temporarily forces
    a sensitivity dtype without mutating ``bmm_dtype``.
    """
    override = getattr(self, '_bmm_dtype_override', None)
    if override and override in self.matrix.supported_datatypes():
      return override
    dt = self.bmm_dtype or self.matrix_dtype or 'bfloat16'
    if dt in self.matrix.supported_datatypes():
      return dt
    if self.matrix_dtype and self.matrix_dtype in self.matrix.supported_datatypes():
      return self.matrix_dtype
    return self.matrix.supported_datatypes()[0]

  def get_bmm_throughput(self, flops):
    """Matrix-engine throughput for BMM ops under ``bmm_dtype``."""
    return self.matrix.throughput(self.get_bmm_dtype(), flops)

  def get_bmm_time_scale(self, kind='default'):
    """Return BMM flops_time multiplier for kind in {default, absorb, attn_score}."""
    return self.bmm_time_scale.get(kind, self.bmm_time_scale['default'])

  @staticmethod
  def bmm_scale_kind(layer_name):
    """Classify MLA/MHA BatchMatMul names for bmm_time_scale lookup."""
    # Attention score / context BMMs (lower achieved eff than Linear).
    attn_markers = (
      'ScoreKV', 'ScorePE', 'AttnKV',
      'Key_Query', 'Multihead_Attn',
    )
    if any(m in layer_name for m in attn_markers):
      return 'attn_score'
    # QAbsorb / VAbsorb and anything else default to absorb/default.
    if 'Absorb' in layer_name:
      return 'absorb'
    return 'default'

  @property
  def num_networks(self):
    return len(self.networks)

  def get_network(self, tier):
    assert tier < len(self.networks), f'Bad network tier ID: {tier}'
    return self.networks[tier]

  def set_datatype(self, datatype):
    """Backward-compatible: use the same dtype for matrix and vector engines. """
    self.set_datatypes(datatype, datatype)

  def set_datatypes(self, matrix_dtype, vector_dtype):
    """Select independent compute dtypes for matrix vs vector engines."""
    assert matrix_dtype in System.TypeSizes, \
      f'Unsupported matrix data type: {matrix_dtype}'
    assert vector_dtype in System.TypeSizes, \
      f'Unsupported vector data type: {vector_dtype}'
    assert matrix_dtype in self.matrix.supported_datatypes(), \
      (f'matrix dtype {matrix_dtype} not in system JSON; '
       f'supported={self.matrix.supported_datatypes()}')
    assert vector_dtype in self.vector.supported_datatypes(), \
      (f'vector dtype {vector_dtype} not in system JSON; '
       f'supported={self.vector.supported_datatypes()}')
    self.matrix_dtype = matrix_dtype
    self.vector_dtype = vector_dtype
    self.datatype = matrix_dtype

  def get_matrix_throughput(self, flops):
    return self.matrix.throughput(self.matrix_dtype, flops)

  def get_vector_throughput(self, flops):
    return self.vector.throughput(self.vector_dtype, flops)

  def get_mem1_throughput(self, size):
    return self.mem1.throughput(size)

  def get_mem2_throughput(self, size):
    return self.mem2.throughput(size)

  def compute_offload_time(self, size):
    return size / self.mem2.throughput(size)

  def get_processing_time(self, flops_time, mem_time):
    if self.proc_mode == 'roofline':
      return max(flops_time, mem_time)
    elif self.proc_mode == 'no_overlap':
      return flops_time + mem_time
