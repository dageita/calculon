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

import calculon
from calculon.llm import *

class Runner(calculon.CommandLine):
  NAME = 'llm'
  ALIASES = []

  @staticmethod
  def create_parser(subparser):
    sp = subparser.add_parser(Runner.NAME, aliases=Runner.ALIASES,
                              help='run a single llm calculation')
    sp.set_defaults(func=Runner.run_command)
    sp.add_argument('application', type=str,
                    help='File path to application configuration')
    sp.add_argument('execution', type=str,
                    help='File path to execution configuration')
    sp.add_argument('system', type=str,
                    help='File path to system configuration')
    sp.add_argument('stats', type=str,
                    help='File path to stats output ("-" for stdout")')
    sp.add_argument('-p', '--peers', type=str, default=None,
                    help='File path to write out peers file')
    sp.add_argument('-l', '--layers', action='store_true',
                    help='Include layers information in output stats file')

  @staticmethod
  def run_command(logger, args):
    app_json = calculon.io.read_json_file(args.application)
    exe_json = calculon.io.read_json_file(args.execution)
    sys_json = calculon.io.read_json_file(args.system)

    app = Llm.Application(app_json)
    exe = Llm.Execution.from_json(exe_json)
    syst = System(sys_json, logger)

    try:
      model = Llm(app, logger)
      model.compile(syst, exe)
      model.run(syst)
    except Llm.Error as error:
      print(f'ERROR: {error}')
      return -1

    if args.stats == '-':
      model.display_stats()
    elif calculon.is_json_extension(args.stats):
      calculon.write_json_file(model.get_stats_json(args.layers), args.stats)
    else:
      assert False, f'unknown stats extension: {args.stats}'

    if args.peers:
      calculon.write_json_file(exe.get_peers_json(), args.peers)

    return 0

  @staticmethod
  def isinstance_run_command(logger, app, exe, syst, stats='-', peers=None, layers=False):
      try:
          model = Llm(app, logger)
          model.compile(syst, exe)
          model.run(syst)
      except Llm.Error as error:
        logger.error(f'LLM Error: {error}')  # 记录日志
        return {"status": "error", "error": str(error)} 
      except Exception as e:
        logger.error(f'Unexpected error: {e}') 
        return {"status": "error", "error": f"Internal error: {str(e)}"}

      # if stats == '-':
      #     model.display_stats()
      # elif calculon.is_json_extension(stats):
      #     calculon.write_json_file(model.get_stats_json(layers), stats)
      # else:
      #     assert False, f'unknown stats extension: {stats}'

      # if peers:
      #     calculon.write_json_file(exe.get_peers_json(), peers)

      res = Runner.get_simulator_res_json(model)

      return res

  @staticmethod
  def get_simulator_res_json(model: Llm):
    # 获取网络时间数据，包括timeline信息
    network_result = model.get_total_flow_network_time()
    
    # 解包返回值 - 新版本有20个返回值
    if len(network_result) == 9:
        # 兼容旧版本，只有9个返回值
        global_time, tp_comm, tp_fw_comm, tp_bw_comm, pp_comm, pp_fw_comm, pp_bw_comm, dp_comm, total_comm = network_result
        timeline_events = []
        # 设置默认值
        batch_tp_fw_comm = tp_fw_comm
        batch_tp_bw_comm = tp_bw_comm
        batch_pp_fw_comm = pp_fw_comm
        batch_pp_bw_comm = pp_bw_comm
        batch_dp_comm = dp_comm
        batch_tp_comm = tp_comm
        batch_pp_comm = pp_comm
        microbatch_tp_fw_comm = tp_fw_comm / model.exe._num_microbatches if model.exe._num_microbatches > 0 else 0
        microbatch_tp_bw_comm = tp_bw_comm / model.exe._num_microbatches if model.exe._num_microbatches > 0 else 0
        microbatch_pp_fw_comm = pp_fw_comm / model.exe._num_microbatches if model.exe._num_microbatches > 0 else 0
        microbatch_pp_bw_comm = pp_bw_comm / model.exe._num_microbatches if model.exe._num_microbatches > 0 else 0
        total_comm_time = total_comm
    else:
        # 新版本，包含timeline数据和新的通信时间数据
        (global_time, batch_tp_fw_comm, batch_tp_bw_comm, 
         batch_pp_fw_comm, batch_pp_bw_comm, batch_dp_comm,
         batch_tp_comm, batch_pp_comm,
         microbatch_tp_fw_comm, microbatch_tp_bw_comm, 
         microbatch_pp_fw_comm, microbatch_pp_bw_comm, 
         total_comm_time,
         timeline_event_count, timeline_ranks, timeline_event_types, 
         timeline_microbatches, timeline_start_times, timeline_end_times) = network_result
        
        # 构建timeline事件列表
        timeline_events = []
        for i in range(timeline_event_count):
            try:
                # 获取事件数据
                rank = timeline_ranks[i] if i < len(timeline_ranks) else 0
                
                # 获取事件类型字符串
                if i < len(timeline_event_types) and timeline_event_types[i]:
                    try:
                        buffer_value = timeline_event_types[i]
                        if buffer_value:
                            # 直接解码bytes对象
                            event_type = buffer_value.decode('utf-8').strip()
                            if not event_type:  # 如果字符串为空
                                event_type = "unknown"
                        else:
                            event_type = "unknown"
                    except (UnicodeDecodeError, AttributeError) as e:
                        event_type = "unknown"
                else:
                    event_type = "unknown"
                
                microbatch = timeline_microbatches[i] if i < len(timeline_microbatches) else 0
                start_time = float(timeline_start_times[i]) if i < len(timeline_start_times) else 0.0
                end_time = float(timeline_end_times[i]) if i < len(timeline_end_times) else 0.0
                
                # 添加到事件列表
                timeline_events.append({
                    "rank": int(rank),           # rank
                    "event_type": event_type,          # event_type
                    "microbatch": int(microbatch),     # microbatch
                    "start_time": round(start_time, 6), # start_time，保留6位小数
                    "end_time": round(end_time, 6)    # end_time，保留6位小数
                })
            except (IndexError, ValueError, TypeError) as e:
                print(f"Warning: Failed to process timeline event {i}: {e}")
                continue
        
        # 打印timeline事件统计信息
        print(f"Successfully processed {len(timeline_events)} timeline events out of {timeline_event_count} total events")
        if timeline_events:
            print(f"Sample event: {timeline_events[0]}")
    return {
        "memory_usage": {
            "optimizer": human_format(model.get_optimizer_space(), 'bytes'),
            "weights": human_format(model.get_weight_space(), 'bytes'),
            "weight_gradients": human_format(model.get_weight_grad_space(), 'bytes'),
            "activation": human_format(model.get_act_space(), 'bytes'),
            "activation_gradients": human_format(model.get_act_grad_space(), 'bytes'),
            "overall_usage": human_format(model.get_mem_tier1_cap_req(), 'bytes'),
        },
        "computation": {
            "per_device_blocks": model._blocks_per_proc,
            "num_microbatches": model.exe._num_microbatches,
            "batch_forward_computation_time": model.get_fw_time(),
            "microbatch_forward_computation_time": model._block_fw_time * model._blocks_per_proc,
            "batch_backward_computation_time": model.get_bw_time(),
            "microbatch_backward_computation_time": (model._block_agrad_time + model._block_wgrad_time) * model._blocks_per_proc,
        },
        "communication": {
            "dp_comm_size": human_format(model._dp_comm_size, 'bytes'),
            "tp_comm_fw_size": human_format(model._tp_fw_comm_size, 'bytes'),
            "tp_comm_bw_size": human_format(model._tp_bw_comm_size, 'bytes'),
            "pp_comm_fw_size": human_format(model._pp_fw_comm_size, 'bytes'),
            "pp_comm_bw_size": human_format(model._pp_bw_comm_size, 'bytes'),
            "batch_dp_comm_time": batch_dp_comm,
            "batch_tp_fw_comm_time": batch_tp_fw_comm,
            "batch_tp_bw_comm_time": batch_tp_bw_comm,
            "batch_pp_fw_comm_time": batch_pp_fw_comm,
            "batch_pp_bw_comm_time": batch_pp_bw_comm,
            "batch_tp_comm_time": batch_tp_comm,
            "batch_pp_comm_time": batch_pp_comm,
            "microbatch_tp_fw_comm_time": microbatch_tp_fw_comm,
            "microbatch_tp_bw_comm_time": microbatch_tp_bw_comm,
            "microbatch_pp_fw_comm_time": microbatch_pp_fw_comm,
            "microbatch_pp_bw_comm_time": microbatch_pp_bw_comm,
            "total_comm_time": total_comm_time,
        },
        "timeline_events": timeline_events,
        "summary": {
            "global_batch_size": model.exe.global_batch_size,
            "local_batch_size": model.exe._local_batch_size,
            "batch_total_time": global_time,
            "totoal_number_of_gpus": model.exe.num_procs,
            "total_efficiency": round(model.get_total_efficiency(), 4)
        }
    }


calculon.CommandLine.register(Runner)
