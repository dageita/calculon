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
# from ctypes import CDLL, c_int, c_float, c_char_p, c_double, c_uint64, POINTER, byref, create_string_buffer
import ctypes
import logging
from ctypes import *
from ctypes import addressof

lib = CDLL("./libpycallclass.so")
pycall_main = lib.pycall_main
# 使用动态函数签名，避免固定数组大小限制
def create_dynamic_pycall_main(max_events, enable_timeline=True):
    """动态创建pycall_main函数签名 - 始终使用完整的函数签名以匹配C++端"""
    # C++端的函数声明是固定的，包含所有timeline参数
    # 无论enable_timeline为true还是false，都需要传递所有参数
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
        c_bool,     # enableTimeline
        POINTER(c_int),     # timelineEventCount
        POINTER(c_int),     # timelineRanks
        (c_char_p * max_events),  # timelineEventTypes - 动态大小
        POINTER(c_int),     # timelineMicrobatches
        POINTER(c_double),  # timelineStartTimes
        POINTER(c_double),  # timelineEndTimes
        POINTER(c_double),  # globalTime
        POINTER(c_double),  # batchTpFwComm
        POINTER(c_double),  # batchTpBwComm
        POINTER(c_double),  # batchPpFwComm
        POINTER(c_double),  # batchPpBwComm
        POINTER(c_double),  # batchDpComm
        POINTER(c_double),  # batchTpComm
        POINTER(c_double),  # batchPpComm
        POINTER(c_double),  # microbatchTpFwComm
        POINTER(c_double),  # microbatchTpBwComm
        POINTER(c_double),  # microbatchPpFwComm
        POINTER(c_double),  # microbatchPpBwComm
        POINTER(c_double)   # totalCommTime
    ]
pycall_main.restype = None



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

  def __init__(self, cfg, log=None):
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
    
    # 初始化logger - 如果传入了logger则使用，否则创建新的
    if log is not None:
      self.log = log
    else:
      self.log = logging.getLogger(f'{self.__class__.__name__}')
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
    self.log.info("wxftest flow network init: inter=%f, intra=%f, topology=%s", self._inter, self._intra, self._topology)

  def total_flow_network_time(self, pp, dp, tp, fwdCompTime, bwdCompTime, microbatches, fwdTPSize, bwdTPSize, fwdPPSize, bwdPPSize, dpSize, enable_timeline):
    topology_bytes = self._topology.encode("utf-8") if isinstance(self._topology, str) else self._topology
    self.log.info("wxftest total flow network time: pp=%d, dp=%d, tp=%d, fwdCompTime=%f, bwdCompTime=%f, microbatches=%d, fwdTPSize=%d, bwdTPSize=%d, fwdPPSize=%d, bwdPPSize=%d, dpSize=%d, enable_timeline=%s", pp, dp, tp, fwdCompTime, bwdCompTime, microbatches, fwdTPSize, bwdTPSize, fwdPPSize, bwdPPSize, dpSize, enable_timeline)
    
    # 始终分配timeline相关内存，以匹配C++端的函数签名
    # 无论enable_timeline为true还是false，都需要传递所有参数
    timelineEventCount = c_int(0)
    
    # 预分配一个较大的缓冲区
    initial_max_events = 1000
    timelineRanks = (c_int * initial_max_events)()
    timelineMicrobatches = (c_int * initial_max_events)()
    timelineStartTimes = (c_double * initial_max_events)()
    timelineEndTimes = (c_double * initial_max_events)()
    timelineEventTypes = (c_char_p * initial_max_events)()
    
    # 预分配字符串缓冲区
    string_buffers = []
    for i in range(initial_max_events):
        buffer = create_string_buffer(64)
        string_buffers.append(buffer)
        timelineEventTypes[i] = cast(buffer, c_char_p)

    # 新的返回值变量
    globalTime = c_double()
    batchTpFwComm = c_double()
    batchTpBwComm = c_double()
    batchPpFwComm = c_double()
    batchPpBwComm = c_double()
    batchDpComm = c_double()
    batchTpComm = c_double()
    batchPpComm = c_double()
    microbatchTpFwComm = c_double()
    microbatchTpBwComm = c_double()
    microbatchPpFwComm = c_double()
    microbatchPpBwComm = c_double()
    totalCommTime = c_double()

    try:
        # 设置函数签名 - 始终使用完整的函数签名
        create_dynamic_pycall_main(initial_max_events, enable_timeline)
        
        # 始终使用相同的调用方式，传递所有参数
        # C++端会根据enableTimeline参数决定是否使用timeline相关参数
        pycall_main(
            pp, dp, tp,
            self._inter, self._intra, 
            fwdCompTime, bwdCompTime, microbatches,
            topology_bytes,
            self.cast_uint64(fwdTPSize), self.cast_uint64(bwdTPSize),
            self.cast_uint64(fwdPPSize), self.cast_uint64(bwdPPSize),
            self.cast_uint64(dpSize),
            c_bool(enable_timeline),  # enableTimeline参数
            byref(timelineEventCount), timelineRanks, timelineEventTypes,
            timelineMicrobatches, timelineStartTimes, timelineEndTimes,
            byref(globalTime), 
            byref(batchTpFwComm), byref(batchTpBwComm), 
            byref(batchPpFwComm), byref(batchPpBwComm), 
            byref(batchDpComm), byref(batchTpComm), byref(batchPpComm),
            byref(microbatchTpFwComm), byref(microbatchTpBwComm), 
            byref(microbatchPpFwComm), byref(microbatchPpBwComm), 
            byref(totalCommTime)
        )
        
        # 检查事件数量是否超过缓冲区大小（仅在enable_timeline=True时有效）
        if enable_timeline:
            actual_events = timelineEventCount.value
            if actual_events > initial_max_events:
                self.log.warning("Event count (%d) exceeds buffer size (%d). Some timeline events may be truncated. Consider increasing max_events parameter.", actual_events, initial_max_events)
        
    except Exception as e:
        self.log.error("Error in pycall main: %s", e)
        # 返回默认值
        return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, [], [], [], [], [])

    self.log.debug("wxftest - New return values:")
    self.log.debug("  globalTime: %f", globalTime.value)
    self.log.debug("  batchTpFwComm: %f", batchTpFwComm.value)
    self.log.debug("  batchTpBwComm: %f", batchTpBwComm.value)
    self.log.debug("  batchPpFwComm: %f", batchPpFwComm.value)
    self.log.debug("  batchPpBwComm: %f", batchPpBwComm.value)
    self.log.debug("  batchDpComm: %f", batchDpComm.value)
    self.log.debug("  batchTpComm: %f", batchTpComm.value)
    self.log.debug("  batchPpComm: %f", batchPpComm.value)
    self.log.debug("  microbatchTpFwComm: %f", microbatchTpFwComm.value)
    self.log.debug("  microbatchTpBwComm: %f", microbatchTpBwComm.value)
    self.log.debug("  microbatchPpFwComm: %f", microbatchPpFwComm.value)
    self.log.debug("  microbatchPpBwComm: %f", microbatchPpBwComm.value)
    self.log.debug("  totalCommTime: %f", totalCommTime.value)
    
    # 只在启用timeline时打印timeline相关数据
    if enable_timeline:
        self.log.debug("Timeline data:")
        self.log.debug("  Event count: %d", timelineEventCount.value)
        self.log.debug("  First 5 ranks: %s", list(timelineRanks[:5]))
        self.log.debug("  First 5 microbatches: %s", list(timelineMicrobatches[:5]))
        self.log.debug("  First 5 start times: %s", list(timelineStartTimes[:5]))
        self.log.debug("  First 5 end times: %s", list(timelineEndTimes[:5]))
        
        # 打印字符串缓冲区状态
        self.log.debug("String buffers status:")
        for i in range(min(5, timelineEventCount.value)):
            if i < len(timelineEventTypes) and timelineEventTypes[i]:
                try:
                    # 直接使用timelineEventTypes[i]，它已经是bytes对象
                    buffer_content = timelineEventTypes[i]
                    self.log.debug("  Buffer %d: %s", i, buffer_content)
                    if buffer_content:
                        try:
                            decoded = buffer_content.decode('utf-8')
                        except UnicodeDecodeError as e:
                            self.log.debug("    Decode error: %s", e)
                    else:
                        self.log.debug("    Empty buffer")
                except Exception as e:
                    self.log.debug("    Error accessing buffer %d: %s", i, e)
            else:
                self.log.debug("  Buffer %d: None or invalid", i)
    
    # 始终返回相同格式的结果，包含timeline数据
    # 当enable_timeline=False时，timeline相关数据为空或默认值
    return (globalTime.value, batchTpFwComm.value, batchTpBwComm.value, 
            batchPpFwComm.value, batchPpBwComm.value, batchDpComm.value,
            batchTpComm.value, batchPpComm.value,
            microbatchTpFwComm.value, microbatchTpBwComm.value, 
            microbatchPpFwComm.value, microbatchPpBwComm.value, 
            totalCommTime.value,
            timelineEventCount.value, timelineRanks, timelineEventTypes, 
            timelineMicrobatches, timelineStartTimes, timelineEndTimes)
