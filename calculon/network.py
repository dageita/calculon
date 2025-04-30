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
from ctypes import CDLL, c_int, c_float

lib = CDLL("./libpycallclass.so")
pycall_main = lib.pycall_main
pycall_main.argtypes = [
    c_int,   # pp
    c_int,   # dp
    c_int,   # tp
    c_float, # inter
    c_float, # intra
    c_int,   # microbatches
    c_int,   # fwdTPSize
    c_int,   # bwdTPSize
    c_int,   # fwdPPSize
    c_int,   # bwdPPSize
    c_int    # dpSize
]
pycall_main.restype = c_float  # 返回值类型


class Network:
  """Configuration for a network."""

  kKeys = set(['bandwidth', 'efficiency', 'size', 'latency', 'ops',
               'must_be_filled', 'processor_usage'])
  kNetOps = set(['p2p', 'reduce_scatter', 'all_gather', 'all_reduce'])
  kCollectives = set(['reduce_scatter', 'all_gather', 'all_reduce'])

  class Op:
    def __init__(self, scalar, offset):
      self.scalar = scalar
      self.offset = offset

  @staticmethod
  def _parse_op(op, scalar, offset):
    assert op in Network.kNetOps, f'Invalid network op: {op}'
    assert scalar > 0.0, f'Invalid network scalar for {op}: {scalar}'
    if op in Network.kCollectives:
      assert offset is not None, f'Must give offset for {op}'
      return Network.Op(scalar, offset)
    else:
      assert offset is None, f'Can\'t give offset for {op}'
      return Network.Op(scalar, 0)
  
  # def get_parallel_groups(TP, PP, DP, gpus_per_server):
  #   # 调用 C++ 生成数据
  #   lib.generate_groups(TP, PP, DP, gpus_per_server)

  #   # 解析 TP 组
  #   tp_group_count = lib.get_tp_group_count()
  #   tp_sizes = np.ctypeslib.as_array(lib.get_tp_group_sizes(), shape=(tp_group_count,))
  #   tp_data = np.ctypeslib.as_array(lib.get_tp_data(), shape=(sum(tp_sizes),))
  #   tp_groups = np.split(tp_data, np.cumsum(tp_sizes)[:-1])

  #   # 解析 PP 组
  #   pp_group_count = lib.get_pp_group_count()
  #   pp_sizes = np.ctypeslib.as_array(lib.get_pp_group_sizes(), shape=(pp_group_count,))
  #   pp_data = np.ctypeslib.as_array(lib.get_pp_data(), shape=(sum(pp_sizes),))
  #   pp_groups = np.split(pp_data, np.cumsum(pp_sizes)[:-1])

  #   # 解析 DP 组
  #   dp_group_count = lib.get_dp_group_count()
  #   dp_sizes = np.ctypeslib.as_array(lib.get_dp_group_sizes(), shape=(dp_group_count,))
  #   dp_data = np.ctypeslib.as_array(lib.get_dp_data(), shape=(sum(dp_sizes),))
  #   dp_groups = np.split(dp_data, np.cumsum(dp_sizes)[:-1])

  #   return tp_groups, pp_groups, dp_groups

  def __init__(self, cfg):
    assert Network.kKeys == set(cfg.keys())
    self._bw = cfg['bandwidth'] * 1e9  # Specified in GB/s
    assert self._bw > 0
    self._eff = cfg['efficiency']
    assert 0 < self._eff <= 1.0
    self._size = cfg['size']
    assert self._size >= 0
    self._latency = cfg['latency']
    self._ops = {}
    for op in cfg['ops']:
      self._ops[op] = Network._parse_op(
        op, cfg['ops'][op][0], cfg['ops'][op][1])
    assert set(self._ops.keys()) == Network.kNetOps
    self._must_be_filled = cfg['must_be_filled']
    self._proc_usage = cfg['processor_usage']
    assert self._proc_usage >= 0.0 and self._proc_usage < 1.0

  @property
  def size(self):
    return self._size

  @property
  def must_be_filled(self):
    return self._must_be_filled

  @property
  def processor_usage(self):
    return self._proc_usage

  def time(self, op, op_size, comm_size):
    """ Computes the time taken for a network operation.

    Args:
      op (str)        : operation name
      op_size (int)   : operation size in bytes
      comm_size (int) : number of participants in operation

    Returns:
      time (float)    : time needed for operation
    """
    if op not in Network.kCollectives:
      assert comm_size == 2
    else:
      assert comm_size >= 2
    assert op in Network.kNetOps
    assert op_size >= 0

    # Scales the op_size by the scalar
    op_size *= self._ops[op].scalar

    # Scales the op_size by the op offset
    chunk_size = 1 / comm_size * op_size
    op_size += chunk_size * self._ops[op].offse

    # Calculates time based on raw bandwidth,  bandwidth efficiency, and latency
    return self._latency + op_size / (self._bw * self._eff)
    
  def total_flow_network_time(self, pp, dp, tp, inter, intra, microbatches, fwdTPSize, bwdTPSize, fwdPPSize, bwdPPSize, dpSize):
    time = pycall_main(
    pp=pp, dp=dp, tp=tp,
    inter=inter, intra=intra,
    microbatches=microbatches,
    fwdTPSize=fwdTPSize, bwdTPSize=bwdTPSize,
    fwdPPSize=fwdPPSize, bwdPPSize=bwdPPSize,
    dpSize=dpSize
    )
    return time






    
# float pycall_main(int pp, int dp, int tp, float inter, float intra, int microbatches, int fwdTPSize, int bwdTPSize, int fwdPPSize, int bwdPPSize, int dpSize) {


# so = ctypes.cdll.LoadLibrary
# lib = so("./libpycallclass.so")
# lib.simulator_main.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_float, ctypes.c_int]
# lib.simulator_main.restype = ctypes.c_int
# lib.test(1)
# lib.simulator_main(8, 8, ctypes.c_float(102.4), ctypes.c_float(10240000), 4)