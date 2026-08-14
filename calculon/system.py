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
    # Phase2 operator correction for reduction-shaped RMSNorm.  Keys are the
    # exact reduction width; values scale the generic vector compute time.
    # This stays separate from vector.gflops_efficiency because ordinary
    # elementwise kernels do not have RMSNorm's cross-width reduction cost.
    norm_scales = cfg.get('rmsnorm_time_scale') or {}
    self.rmsnorm_time_scale = {
      str(width): float(scale) for width, scale in norm_scales.items()
      if str(width) != 'default'
    }
    self.rmsnorm_default_scale = float(norm_scales.get('default', 1.0))
    # Phase2 G4 operator-specific corrections.  Router is a very narrow
    # FP8 Linear and sigmoid is a standalone transcendental kernel; neither
    # is represented faithfully by the generic matrix/vector curves.
    self.router_linear_time_scale = float(
      cfg.get('router_linear_time_scale', 1.0) or 1.0)
    self.router_sigmoid_time_scale = float(
      cfg.get('router_sigmoid_time_scale', 1.0) or 1.0)
    # Phase2 exact-shape operator tables.  These are intentionally keyed by
    # both layer name and full shape: they correct MLA's unusual aspect ratios
    # without leaking workload-specific timings into generic efficiency curves.
    op_times = cfg.get('operator_shape_latency_s') or {}
    self.linear_op_times = op_times.get('linear', {})
    self.bmm_op_times = op_times.get('bmm', {})
    # Phase3 grouped-MoE timings are deliberately separate from ordinary
    # Linear/operator calibration. A grouped expert projection has many tiny-M
    # groups, distinct weights and scheduler costs that a dense GEMM roofline
    # cannot represent.
    self.grouped_moe_times = cfg.get('grouped_moe_shape_latency_s') or {}
    # Optional shape-aware compute times for K=4096 Linear GEMMs.  These complement
    # the FLOPs-only matrix curve for explicitly calibrated output widths.
    self.linear_shape = cfg.get('linear_shape') or cfg.get('linear_small_n') or {}
    # Compact hardware-level model fitted from legacy per-shape tables. It is
    # activated only when the fitter's MAPE gate passes; exact tables remain a
    # compatibility fallback.
    self.parametric_shapes = cfg.get('parametric_shape_models') or {}

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

  def get_linear_shape_time(self, m, k, n):
    """Return an exact-N calibrated forward Linear compute time, or 0.

    No nearest-N interpolation is used: unmeasured output widths retain the
    generic matrix model, preserving safe behaviour for arbitrary networks.
    """
    model = self.linear_shape
    if not model or int(k) != int(model.get('reference_k', -1)):
      return 0.0
    curves = model.get('latency_s', {}).get(self.matrix_dtype, {})
    if not isinstance(curves, dict):
      return 0.0
    bucket = str(int(n))
    if bucket not in curves:
      return 0.0
    points = curves[bucket]
    for min_m, latency_s in points:
      if m >= int(min_m):
        return float(latency_s)
    return float(points[-1][1]) if points else 0.0

  def get_parametric_linear_time(self, m, k, n):
    model = (self.parametric_shapes.get('linear', {})
             .get(self.matrix_dtype, {}))
    if not model.get('enabled', False):
      return 0.0
    if int(k) != int(model.get('reference_k', -1)):
      return 0.0
    c = model.get('coefficients_s') or []
    if len(c) != 4:
      return 0.0
    flops = 2.0 * m * k * n / 1e15
    nbytes = (m*k + k*n + m*n) * System.TypeSizes[self.matrix_dtype] / 1e12
    tiles = ((m + 63)//64) * ((n + 63)//64) / 1e6
    return max(0.0, c[0] + c[1]*flops + c[2]*nbytes + c[3]*tiles)

  def get_parametric_operator_linear_time(self, m, k, n):
    model = (self.parametric_shapes.get('operator_linear', {})
             .get(self.matrix_dtype, {}))
    if not model.get('enabled', False):
      return 0.0
    c = model.get('coefficients_s') or []
    if len(c) != 4:
      return 0.0
    bpe = System.TypeSizes[self.matrix_dtype]
    x = [1., 2*m*k*n/1e15, (m*k+k*n+m*n)*bpe/1e12,
         ((m+63)//64)*((n+63)//64)/1e6]
    return max(0.0, sum(ci*xi for ci,xi in zip(c,x)))

  def get_parametric_bmm_time(self, batch, m, n, k):
    model = (self.parametric_shapes.get('bmm', {})
             .get(self.get_bmm_dtype(), {}))
    if not model.get('enabled', False):
      return 0.0
    c = model.get('coefficients_s') or []
    if len(c) != 4:
      return 0.0
    flops = batch*2.0*m*n*k/1e15
    bpe = System.TypeSizes[self.get_bmm_dtype()]
    nbytes = batch*(m*n+n*k+m*k)*bpe/1e12
    tiles = batch*((m+63)//64)*((k+63)//64)/1e6
    return max(0.0, c[0]+c[1]*flops+c[2]*nbytes+c[3]*tiles)

  def get_vector_throughput(self, flops):
    return self.vector.throughput(self.vector_dtype, flops)

  def get_rmsnorm_time_scale(self, width):
    return self.rmsnorm_time_scale.get(
      str(int(width)), self.rmsnorm_default_scale)

  @staticmethod
  def _exact_operator_time(table, dtype, name, shape):
    entry = (table.get(dtype) or {}).get(name)
    if not isinstance(entry, dict):
      return 0.0
    configured = entry.get('shape')
    if not isinstance(configured, list) or [int(x) for x in configured] != [int(x) for x in shape]:
      return 0.0
    return float(entry.get('latency_s', 0.0) or 0.0)

  def get_linear_op_time(self, name, m, k, n):
    return self._exact_operator_time(
      self.linear_op_times, self.matrix_dtype, name, [m, k, n])

  def get_grouped_moe_time(self, name, m, k, n, weight_mult, flop_mult,
                           bytes_per_element=1):
    entry = (self.grouped_moe_times.get(self.matrix_dtype) or {}).get(name)
    if not isinstance(entry, dict):
      return 0.0
    shape = [int(x) for x in entry.get('shape', [])]
    if len(shape) != 3 or shape[1:] != [int(k), int(n)]:
      return 0.0
    if float(entry.get('weight_multiplier', -1)) != float(weight_mult):
      return 0.0
    if float(entry.get('flop_multiplier', -1)) != float(flop_mult):
      return 0.0
    # Exact anchor remains authoritative. For another seq/M, extrapolate with
    # two physical resources: fixed distinct expert-weight traffic plus
    # M-scaled activation traffic, and M-scaled useful FLOPs. This avoids a
    # separate table entry for every seq_len while retaining shape isolation.
    anchors = entry.get('anchors') or []
    if anchors:
      anchors = sorted(anchors, key=lambda x: int(x['m']))
      nearest = min(anchors, key=lambda x: abs(int(x['m']) - int(m)))
      if int(nearest['m']) == int(m):
        return float(nearest['latency_s'])
      # Interpolate hardware rates in log-M; clamp beyond measured regimes.
      lo = max((x for x in anchors if int(x['m']) <= int(m)),
               key=lambda x: int(x['m']), default=anchors[0])
      hi = min((x for x in anchors if int(x['m']) >= int(m)),
               key=lambda x: int(x['m']), default=anchors[-1])
      if int(lo['m']) == int(hi['m']):
        eff_bw = float(lo['effective_bandwidth_Bps'])
        eff_flops = float(lo['effective_flops_per_s'])
      else:
        import math
        w = ((math.log(float(m))-math.log(float(lo['m']))) /
             (math.log(float(hi['m']))-math.log(float(lo['m']))))
        eff_bw = math.exp((1-w)*math.log(float(lo['effective_bandwidth_Bps']))
                          + w*math.log(float(hi['effective_bandwidth_Bps'])))
        eff_flops = math.exp((1-w)*math.log(float(lo['effective_flops_per_s']))
                             + w*math.log(float(hi['effective_flops_per_s'])))
    elif shape[0] == int(m):
      return float(entry.get('latency_s', 0.0) or 0.0)
    else:
      eff_bw = float(entry.get('effective_bandwidth_Bps', 0.0) or 0.0)
      eff_flops = float(entry.get('effective_flops_per_s', 0.0) or 0.0)
    if eff_bw <= 0 or eff_flops <= 0:
      return 0.0
    flops = 2.0 * float(m) * float(k) * float(n) * float(flop_mult)
    nbytes = (float(m) * float(k) + float(m) * float(n)
              + float(k) * float(n) * float(weight_mult)) \
             * float(bytes_per_element)
    return max(flops / eff_flops, nbytes / eff_bw)

  def get_bmm_op_time(self, name, batch, m, n, k):
    return self._exact_operator_time(
      self.bmm_op_times, self.get_bmm_dtype(), name, [batch, m, n, k])

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
