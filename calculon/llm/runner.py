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
    syst = System(sys_json)

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
          print(f'ERROR: {error}')
          return -1

      if stats == '-':
          model.display_stats()
      elif calculon.is_json_extension(stats):
          calculon.write_json_file(model.get_stats_json(layers), stats)
      else:
          assert False, f'unknown stats extension: {stats}'

      if peers:
          calculon.write_json_file(exe.get_peers_json(), peers)

      res = Runner.get_model_stats_json(model)

      return res

  @staticmethod
  def get_model_stats_json(model: Llm):
    global_time, tp_comm, tp_fw_comm, tp_bw_comm, pp_comm, pp_fw_comm, pp_bw_comm, dp_comm, total_comm = model.get_total_flow_network_time()
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
            "dp_comm_size": model._dp_comm_size,
            "tp_comm_fw_size": model._tp_fw_comm_size,
            "tp_comm_bw_size": model._tp_bw_comm_size,
            "pp_comm_fw_size": model._pp_fw_comm_size,
            "pp_comm_bw_size": model._pp_bw_comm_size,
            "batch_dp_comm_time": dp_comm,
            "batch_tp_comm_time": tp_comm,
            "batch_tp_fw_comm_time": tp_fw_comm,
            "microbatch_tp_fw_comm_time": tp_fw_comm / model.exe._num_microbatches,
            "batch_tp_bw_comm_time": tp_bw_comm,
            "microbatch_tp_bw_comm_time": tp_bw_comm / model.exe._num_microbatches,
            "batch_pp_comm_time": pp_comm,
            "batch_pp_fw_comm_time": pp_fw_comm,
            "microbatch_pp_fw_comm_time": pp_fw_comm / model.exe._num_microbatches,
            "batch_pp_bw_comm_time": pp_bw_comm, 
            "microbatch_pp_bw_comm_time": pp_bw_comm / model.exe._num_microbatches,
        },
        "timeline": {
            "per_device_blocks": model._blocks_per_proc,
            "num_microbatches": model.exe._num_microbatches,
            "batch_forward_computation_time": model.get_fw_time(),
            "microbatch_forward_computation_time": model._block_fw_time * model._blocks_per_proc,
            "batch_backward_computation_time": model.get_bw_time(),
            "microbatch_backward_computation_time": (model._block_agrad_time + model._block_wgrad_time) * model._blocks_per_proc,
            "batch_dp_comm_time": dp_comm,
            "batch_tp_comm_time": tp_comm,
            "batch_tp_fw_comm_time": tp_fw_comm,
            "microbatch_tp_fw_comm_time": tp_fw_comm / model.exe._num_microbatches,
            "batch_tp_bw_comm_time": tp_bw_comm,
            "microbatch_tp_bw_comm_time": tp_bw_comm / model.exe._num_microbatches,
            "batch_pp_comm_time": pp_comm,
            "batch_pp_fw_comm_time": pp_fw_comm,
            "microbatch_pp_fw_comm_time": pp_fw_comm / model.exe._num_microbatches,
            "batch_pp_bw_comm_time": pp_bw_comm, 
            "microbatch_pp_bw_comm_time": pp_bw_comm / model.exe._num_microbatches,
            "warmup_time": getattr(model, "_baseblock_fw_time", 0), 
            "cooldown_time": getattr(model, "_edgeblock_fw_time", 0), 
            "batch_total_time": model.get_total_time()
        },
        "summary": {
            "global_batch_size": model.exe.global_batch_size,
            "local_batch_size": model.exe._local_batch_size,
            "batch_total_time": model.get_total_time(),
            "totoal_number_of_gpus": model.exe.num_procs,
            "total_efficiency": round(model.get_total_efficiency(), 4)
        }
    }


calculon.CommandLine.register(Runner)
