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
from ctypes import *
from ctypes import addressof

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
    POINTER(c_double),  # totalComm
    POINTER(c_int),     # timelineEventCount
    POINTER(c_int),     # timelineRanks
    (c_char_p * 100),   # timelineEventTypes - 修改为数组类型
    POINTER(c_int),     # timelineMicrobatches
    POINTER(c_double),  # timelineStartTimes
    POINTER(c_double)   # timelineEndTimes
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

    # 新增时间线相关参数 - 预分配内存避免C++端coredump
    max_events = 100  # 确保足够大

    # 为每个数组预分配内存
    timelineEventCount = c_int(0)  # 单个int值，用于返回事件数量
    timelineRanks = (c_int * max_events)()
    timelineMicrobatches = (c_int * max_events)()
    timelineStartTimes = (c_double * max_events)()
    timelineEndTimes = (c_double * max_events)()

    # 创建字符串指针数组
    timelineEventTypes = (c_char_p * max_events)()
    string_buffers = []  # 保存引用，防止垃圾回收

    # 预分配所有字符串缓冲区，确保内存稳定
    string_buffers = []
    for i in range(max_events):
        buffer = create_string_buffer(64)
        string_buffers.append(buffer)  # 保存引用
        
        # 使用cast进行正确的类型转换
        timelineEventTypes[i] = cast(buffer, c_char_p)
        
        # 验证转换是否成功
        if timelineEventTypes[i]:
            print(f"Successfully created buffer {i}: {timelineEventTypes[i]}")
        else:
            print(f"Failed to create buffer {i}")

    # # 创建指向字符串指针数组的指针，符合C++端char**的期望
    # timelineEventTypes_ptr = cast(timelineEventTypes, POINTER(c_char_p))
    
    # # 验证指针创建是否成功
    # print(f"Created timelineEventTypes_ptr: {timelineEventTypes_ptr}")
    # print(f"Pointer address: {addressof(timelineEventTypes_ptr.contents)}")

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
        byref(dpComm), byref(totalComm),
        byref(timelineEventCount), timelineRanks, timelineEventTypes,
        timelineMicrobatches, timelineStartTimes, timelineEndTimes
    )

    print("wxftest", globalTime.value, tpComm.value, tpFwComm.value, tpBwComm.value, ppComm.value, ppFwComm.value, ppBwComm.value, dpComm.value, totalComm.value)
    
    # 打印timeline相关数据用于调试
    print("Timeline data:")
    print(f"  Event count: {timelineEventCount.value}")
    print(f"  First 5 ranks: {list(timelineRanks[:5])}")
    print(f"  First 5 microbatches: {list(timelineMicrobatches[:5])}")
    print(f"  First 5 start times: {list(timelineStartTimes[:5])}")
    print(f"  First 5 end times: {list(timelineEndTimes[:5])}")
    
    # 打印字符串缓冲区状态
    print("String buffers status:")
    for i in range(min(5, timelineEventCount.value)):
        if i < len(timelineEventTypes) and timelineEventTypes[i]:
            try:
                # 直接使用timelineEventTypes[i]，它已经是bytes对象
                buffer_content = timelineEventTypes[i]
                print(f"  Buffer {i}: {buffer_content}")
                if buffer_content:
                    try:
                        decoded = buffer_content.decode('utf-8')
                    except UnicodeDecodeError as e:
                        print(f"    Decode error: {e}")
                else:
                    print(f"    Empty buffer")
            except Exception as e:
                print(f"    Error accessing buffer {i}: {e}")
        else:
            print(f"  Buffer {i}: None or invalid")
    
    return (globalTime.value, tpComm.value, tpFwComm.value, tpBwComm.value, 
            ppComm.value, ppFwComm.value, ppBwComm.value, dpComm.value, totalComm.value,
            timelineEventCount.value, timelineRanks, timelineEventTypes, timelineMicrobatches, timelineStartTimes, timelineEndTimes)
