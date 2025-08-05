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
from ctypes import CDLL, c_int, c_float, c_char_p, c_double, c_uint64, POINTER, byref

lib = CDLL("./libpycallclass.so")
pycall_main = lib.pycall_main
pycall_main.argtypes = [
    c_int,   # pp
    c_int,   # dp
    c_int,   # tp
    c_double, # inter
    c_double, # intra
    c_double, # fwdCompTime
    c_double, # bwdCompTime
    c_int,   # microbatches
    c_char_p, # topology_type
    c_uint64,   # fwdTPSize
    c_uint64,   # bwdTPSize
    c_uint64,   # fwdPPSize
    c_uint64,   # bwdPPSize
    c_uint64,   # dpSize 
    POINTER(c_double),  # globalTime
    POINTER(c_double),  # tpComm
    POINTER(c_double),  # tpFwComm
    POINTER(c_double),  # tpBwComm
    POINTER(c_double),  # ppComm
    POINTER(c_double),  # ppFwComm
    POINTER(c_double),  # ppBwComm
    POINTER(c_double),  # dpComm
    POINTER(c_double)   # totalComm
]
pycall_main.restype = None  # 返回值类型



class Network:
  """Configuration for a network."""

  kKeys = set(['bandwidth', 'topology', 'efficiency', 'size', 'latency', 'ops',
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

  def __init__(self, cfg):
    assert Network.kKeys == set(cfg.keys())
    self._bw = cfg['bandwidth'] * 1e9  # Specified in GB/s
    # assert self._bw > 0
    self._eff = cfg['efficiency']
    assert 0 < self._eff <= 1.0
    self._size = cfg['size']
    assert self._size >= 0
    self._latency = cfg['latency']
    self._topology = cfg.get('topology', 'default')  # Default to 'default' if not specified
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
    op_size += chunk_size * self._ops[op].offset

    # Calculates time based on raw bandwidth,  bandwidth efficiency, and latency
    return self._latency + op_size / (self._bw * self._eff)

   # 显式转换函数
  def cast_uint64(self, value):
      return c_uint64(value & 0xFFFFFFFFFFFFFFFF)  # 强制64位掩码
  
# unit: Bps
  def flow_network_init(self, inter, intra, topology):
    self._inter = inter # 机间网络带宽，单位Bps
    self._intra = intra # 机内网络带宽，单位Bps
    self._topology = topology # 网络拓扑类型
    print("wxftest flow network init", self._inter, self._intra, self._topology)

  def total_flow_network_time(self, pp, dp, tp, fwdCompTime, bwdCompTime, microbatches, fwdTPSize, bwdTPSize, fwdPPSize, bwdPPSize, dpSize):
    topology_bytes = self._topology.encode("utf-8") if isinstance(self._topology, str) else self._topology
    # parameters = locals()
    globalTime = c_double()
    tpComm = c_double()
    tpFwComm = c_double()
    tpBwComm = c_double()
    ppComm = c_double()
    ppFwComm = c_double()
    ppBwComm = c_double()
    dpComm = c_double()
    totalComm = c_double()

    pycall_main(
      pp, dp, tp,
      self._inter, self._intra, 
      fwdCompTime, bwdCompTime, microbatches,
      topology_bytes,
      self.cast_uint64(fwdTPSize), self.cast_uint64(bwdTPSize),
      self.cast_uint64(fwdPPSize), self.cast_uint64(bwdPPSize),
      self.cast_uint64(dpSize),
      byref(globalTime), 
      byref(tpComm), byref(tpFwComm), byref(tpBwComm), 
      byref(ppComm), byref(ppFwComm), byref(ppBwComm), 
      byref(dpComm), byref(totalComm)
    )

    print("wxftest", globalTime.value, tpComm.value, tpFwComm.value, tpBwComm.value, ppComm.value, ppFwComm.value, ppBwComm.value, dpComm.value, totalComm.value)
    return globalTime.value, tpComm.value, tpFwComm.value, tpBwComm.value, ppComm.value, ppFwComm.value, ppBwComm.value, dpComm.value, totalComm.value
