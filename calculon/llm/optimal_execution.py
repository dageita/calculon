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

import datetime
import gzip
import logging
import multiprocessing as mp
import psutil
import os

import calculon
from calculon.util import pick, arg_true_false_all
from calculon.llm import *
from calculon.llm.runner import Runner


class OptimalExecution(calculon.CommandLine):
  NAME = 'llm-optimal-execution'
  ALIASES = ['loe']

  @staticmethod
  def create_parser(subparser):
    sp = subparser.add_parser(
      OptimalExecution.NAME, aliases=OptimalExecution.ALIASES,
      help='run a search to find the optimal llm execution')
    sp.set_defaults(func=OptimalExecution.run_command)
    sp.add_argument('-d', '--debug', action='store_true',
                    help='Loop over executions, don\'t run them')
    sp.add_argument('application', type=str,
                    help='File path to application configuration')
    sp.add_argument('num_procs', type=int,
                    help='Number of processors in execution')
    sp.add_argument('max_batch_size', type=int,
                    help='Maximum batch size, will be largest multiple of DP')
    sp.add_argument('datatype', type=str, choices=System.supported_datatypes(),
                    help='The datatype to use')
    sp.add_argument('system', type=str,
                    help='File path to system configuration')
    sp.add_argument('output', type=str,
                    help='File path to the output file'
                    " ('*.csv', '*.csv.gz', '*.json', '*.json.gz')")
    sp.add_argument('-c', '--cpus', type=int, default=psutil.cpu_count(logical=False),
                    help='CPUs to use for parallelization')
    sp.add_argument('-n', '--noneok', action='store_true',
                    help='Don\'t give failure status when no good execution exists')
    sp.add_argument('-m', '--mbs-break', action='store_true',
                    help='Search across MBS and break earlier when possible')
    sp.add_argument('-t', '--top-n', type=int, default=1,
                    help='Number of best outputs')
    sp.add_argument('-l', '--layers', action='store_true',
                    help='Include layers information in output stats file')
    sp.add_argument('-f', '--fused_activation', type=arg_true_false_all,
                    default='true', help='Mode of fused activation')
    sp.add_argument('--no-tp-overlap', action='store_true',
                    help='Don\'t allow TP overlap')
    sp.add_argument('--no-dp-overlap', action='store_true',
                    help='Don\'t allow DP overlap')

  @staticmethod
  def run_command(logger, args):
    assert args.top_n > 0, 'top-n must be > 0'

    app = Llm.Application(calculon.io.read_json_file(args.application))
    syst = System(calculon.io.read_json_file(args.system), logger)

    params = []
    for tp in Llm.get_all_tensor_parallelisms(
        args.num_procs, app.hidden, app.attn_heads):
      for pp in Llm.get_all_pipeline_parallelisms(
          args.num_procs, tp, app.num_blocks):
        dp = Llm.get_data_parallelism(args.num_procs, tp, pp)
        for ppint in Llm.get_valid_pipeline_interleavings(app.num_blocks, pp):
          batch_size = OptimalExecution.get_batch_size(dp, args.max_batch_size)
          if batch_size is None:
            continue
          for activation_recompute in ['full', 'attn_only', 'none']:
            for optimizer_sharding in pick(dp>1, [True, False], [False]):
              for tensor_par_comm_type in ['ar', 'p2p_rs_ag', 'rs_ag']:
                params.append(
                  (args.debug, args.top_n, args.layers, args.num_procs,
                   args.max_batch_size, args.datatype, app, syst, tp, pp, dp,
                   ppint, batch_size, activation_recompute, optimizer_sharding,
                   tensor_par_comm_type, args.fused_activation, args.mbs_break,
                   not args.no_tp_overlap, not args.no_dp_overlap))

    # Runs parallel searches
    start_time = datetime.datetime.now()
    with mp.Pool(args.cpus) as pool:
      searches = pool.starmap(OptimalExecution.search, params)
    end_time = datetime.datetime.now()

    # Combines parallel search result into one data structure
    best = []
    exe_count = 0
    good_exe_count = 0
    bad_exe_count = 0
    for cbest, ec, gec, bec, tp, pp in searches:
      best = OptimalExecution.update_list(best, cbest, args.top_n)
      exe_count += ec
      good_exe_count += gec
      bad_exe_count += bec

    logger.info(f'Total executions: {exe_count}')
    logger.info(f'Good executions: {good_exe_count}')
    logger.info(f'Bad executions: {bad_exe_count}')
    calc_rate = exe_count / (end_time - start_time).total_seconds()
    logger.info(f'Calculation rate: {calc_rate:.2f} calcs/sec')
    if args.debug:
      return 0

    if len(best) == 0:
      if not args.noneok:
        logger.fatal('No acceptable configurations found :(')
        return -1
      else:
        logger.info('No acceptable configurations found :(')
    else:
      logger.info(f'Best sample rate: {best[0][0]}')

    output = {}
    for index, run in enumerate(best):
      _, execution, stats = run
      output[index] = {
        'execution': execution,
        'stats': stats
      }

    if calculon.io.is_json_extension(args.output):
      logger.info(f'Output: {args.output}')
      calculon.io.write_json_file(output, args.output)
    elif args.output.endswith('.csv') or args.output.endswith('.csv.gz'):
      logger.info(f'Output: {args.output}')
      exe_keys = list(output[0]['execution'].keys())
      stats_keys = list(output[0]['stats'].keys())
      opener = gzip.open if args.output.endswith('.gz') else open
      with opener(args.output, 'wb') as fd:
        fd.write(bytes(f',{",".join(exe_keys)},{",".join(stats_keys)}\n',
                       'utf-8'))
        for index in sorted(output.keys()):
          fd.write(bytes(f'{index}', 'utf-8'))
          for exe_key in exe_keys:
            fd.write(bytes(f',{output[index]["execution"][exe_key]}', 'utf-8'))
          for stats_key in stats_keys:
            fd.write(bytes(f',{output[index]["stats"][stats_key]}', 'utf-8'))
          fd.write(bytes('\n', 'utf-8'))
    else:
      assert False, f'Unknown file type: {args.output}'

    return 0

  @staticmethod
  def isinstance_run_command(logger, app, syst, optimal_config):
    """API版本的isinstance_run_command，接受直接的对象参数而不是命令行参数"""
    # 创建一个模拟的args对象来兼容现有的search方法
    class Args:
      def __init__(self, optimal_config):
        self.debug = False
        self.top_n = 1
        self.layers = None  # 不使用layers参数
        self.num_procs = optimal_config.num_procs
        self.max_batch_size = optimal_config.max_batch_size
        self.datatype = optimal_config.datatype
        self.cpus = 1  # 默认使用1个CPU
        self.noneok = True  # 允许没有找到结果
        self.fused_activation = ['none']  # 默认不使用融合激活
        self.mbs_break = False  # 默认不中断microbatch
        self.no_tp_overlap = True  # 默认禁用TP重叠
        self.no_dp_overlap = True  # 默认禁用DP重叠
        self.output = None  # 不需要输出文件

    args = Args(optimal_config)
    
    params = []
    for tp in Llm.get_all_tensor_parallelisms(
        args.num_procs, app.hidden, app.attn_heads):
      for pp in Llm.get_all_pipeline_parallelisms(
          args.num_procs, tp, app.num_blocks):
        dp = Llm.get_data_parallelism(args.num_procs, tp, pp)
        for ppint in Llm.get_valid_pipeline_interleavings(app.num_blocks, pp):
          batch_size = OptimalExecution.get_batch_size(dp, args.max_batch_size)
          if batch_size is None:
            continue
          for activation_recompute in ['full', 'attn_only', 'none']:
            for optimizer_sharding in [False]:  # 固定为False，不使用ZeRO优化
              for tensor_par_comm_type in ['ar', 'p2p_rs_ag', 'rs_ag']:
                params.append(
                  (args.debug, args.top_n, args.layers, args.num_procs,
                    args.max_batch_size, args.datatype, app, syst, tp, pp, dp,
                    ppint, batch_size, activation_recompute, optimizer_sharding,
                    tensor_par_comm_type, args.fused_activation, args.mbs_break,
                    not args.no_tp_overlap, not args.no_dp_overlap))

    # Runs parallel searches
    start_time = datetime.datetime.now()
    with mp.Pool(args.cpus) as pool:
      searches = pool.starmap(OptimalExecution.isinstance_search, params)
    end_time = datetime.datetime.now()

    # Combines parallel search result into one data structure
    best = []
    exe_count = 0
    good_exe_count = 0
    bad_exe_count = 0
    for cbest, ec, gec, bec, tp, pp in searches:
      best = OptimalExecution.update_list(best, cbest, args.top_n)
      exe_count += ec
      good_exe_count += gec
      bad_exe_count += bec

    logger.info(f'Total executions: {exe_count}')
    logger.info(f'Good executions: {good_exe_count}')
    logger.info(f'Bad executions: {bad_exe_count}')
    calc_rate = exe_count / (end_time - start_time).total_seconds()
    logger.info(f'Calculation rate: {calc_rate:.2f} calcs/sec')
    
    if args.debug:
      return {
        "executions": {
          "total_executions": exe_count,
          "good_executions": good_exe_count,
          "bad_executions": bad_exe_count,
          "calculation_rate": calc_rate,
        },
        "debug": True
      }

    if len(best) == 0:
      logger.info('No acceptable configurations found :(')
      return {
        "executions": {
          "total_executions": exe_count,
          "good_executions": good_exe_count,
          "bad_executions": bad_exe_count,
          "calculation_rate": calc_rate,
        },
        "message": "No acceptable configurations found"
      }
    else:
      logger.info(f'Best sample rate: {best[0][0]}')

    # 构建返回结果
    if len(best) > 0:
      # 获取最佳配置
      _, best_execution, best_stats = best[0]
      
      # 从best_execution中移除网络相关参数
      filtered_execution = {k: v for k, v in best_execution.items() 
                           if k not in ['tensor_par_net', 'pipeline_par_net', 'data_par_net']}
      
      # 创建logger用于最佳配置的详细分析
      best_logger = logging.getLogger('best_config')
      best_logger.propagate = False
      if not best_logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        best_logger.addHandler(handler)
        best_logger.setLevel(logging.INFO)
      
      # 使用最佳配置创建模型并获取详细信息
      try:
        best_model = Llm(app, best_logger)
        best_model.compile(syst, Llm.Execution.from_json(best_execution))
        best_model.run(syst)
        
        # 获取详细的模拟结果
        detailed_result = Runner.get_simulator_res_json(best_model)
        
        # 构建最终返回结果
        return {
          "executions": {
            "total_executions": exe_count,
            "good_executions": good_exe_count,
            "bad_executions": bad_exe_count,
            "calculation_rate": calc_rate,
          },
          "optimal_result": filtered_execution,
          "memory_usage": detailed_result["memory_usage"],
          "computation": detailed_result["computation"],
          "communication": detailed_result["communication"],
          "summary": detailed_result["summary"]
        }
      except Exception as e:
        logger.error(f"Failed to generate detailed result for best configuration: {str(e)}")
        # 如果详细分析失败，返回基本信息
        return {
          "executions": {
            "total_executions": exe_count,
            "good_executions": good_exe_count,
            "bad_executions": bad_exe_count,
            "calculation_rate": calc_rate,
          },
          "optimal_result": filtered_execution,
          "message": f"Failed to generate detailed analysis: {str(e)}"
        }
    else:
      return {
        "executions": {
          "total_executions": exe_count,
          "good_executions": good_exe_count,
          "bad_executions": bad_exe_count,
          "calculation_rate": calc_rate,
        },
        "message": "No acceptable configurations found"
      }

  @staticmethod
  def get_batch_size(data_par, max_batch_size):
    if data_par > max_batch_size:
      return None
    last = data_par
    while True:
      if last + data_par > max_batch_size:
        return last
      else:
        last += data_par


  @staticmethod
  def search(debug, top_n, layers, num_procs, max_batch_size, datatype,
          app, syst, tp, pp, dp, ppint, batch_size, activation_recompute,
          optimizer_sharding, tensor_par_comm_type, fused_acts, mbs_break,
          allow_tp_overlap, allow_dp_overlap):
    num_nets = syst.num_networks

    best = []
    exe_count = 0
    good_exe_count = 0
    bad_exe_count = 0

    # 固定Megatron不使用的参数
    weight_offload = False       # 不使用ZeRO优化
    activations_offload = False  # Megatron无此参数
    optimizer_offload = False    # 不使用ZeRO
    seq_par_ag_redo = False      # 固定AG重计算
    data_par_overlap = False     # 禁用DP重叠
    tensor_par_overlap = 'none'  # 禁用TP重叠
    
    can_redo = Llm.can_redo_ag(tensor_par_comm_type, activation_recompute)
    for fused_act in fused_acts:
        for microbatch_size in Llm.get_valid_microbatch_sizes(
                app.seq_size, tp, dp, batch_size, pp):
            mbs_break_good = good_exe_count
            # 固定网络分配策略（根据Megatron实现）
            tn = 0 if tp == 1 else 0  # 假设使用默认网络
            pn = 0 if pp == 1 else 0  # 假设使用默认网络
            dn = 0 if dp == 1 else 0  # 假设使用默认网络
            exe_count += 1
            exe_json = {
                'num_procs': num_procs,
                'tensor_par': tp,
                'pipeline_par': pp,
                'data_par': dp,
                'tensor_par_net': tn,
                'pipeline_par_net': pn,
                'data_par_net': dn,
                'batch_size': batch_size,
                'microbatch_size': microbatch_size,
                'datatype': datatype,
                'fused_activation': fused_act,
                'attention_type': 'multihead',
                'activation_recompute': activation_recompute,
                'pipeline_interleaving': ppint,
                'optimizer_sharding': optimizer_sharding,
                'tensor_par_comm_type': tensor_par_comm_type,
                'tensor_par_overlap': tensor_par_overlap,
                'seq_par_ag_redo': seq_par_ag_redo,
                'data_par_overlap': data_par_overlap,
                'weight_offload': weight_offload,
                'activations_offload': activations_offload,
                'optimizer_offload': optimizer_offload,
                'training': True
            }

            if not debug:
                try:
                    # 创建并配置logger
                    logger = logging.getLogger('sub')
                    logger.propagate = False  # 禁用传播到父logger
                    if not logger.handlers:  # 避免重复添加handler
                        handler = logging.StreamHandler()
                        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
                        handler.setFormatter(formatter)
                        logger.addHandler(handler)
                        logger.setLevel(logging.INFO)
                    model = Llm(app, logger)
                    model.compile(syst, Llm.Execution.from_json(exe_json))
                    model.run(syst)
                    stats = model.get_stats_json(layers)
                    good_exe_count += 1
                    curr = (stats['total_time'], exe_json, stats)
                    best = OptimalExecution.update_list(best, curr, top_n)
                except Exception as ex:
                    # 使用更详细的错误日志
                    error_logger = logging.getLogger('error')
                    error_logger.setLevel(logging.INFO)
                    if not error_logger.handlers:
                        handler = logging.StreamHandler()
                        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
                        handler.setFormatter(formatter)
                        error_logger.addHandler(handler)
                    error_logger.error(f'Execution failed for config: {exe_json}')
                    error_logger.error(f'Error details: {str(ex)}')
                    error_logger.error(f'Error type: {type(ex).__name__}')
                    import traceback
                    error_logger.error(f'Traceback: {traceback.format_exc()}')
                    bad_exe_count += 1
            if mbs_break and good_exe_count == mbs_break_good:
                break
    return (best, exe_count, good_exe_count, bad_exe_count, tp, pp)

  @staticmethod
  def isinstance_search(debug, top_n, layers, num_procs, max_batch_size, datatype,
          app, syst, tp, pp, dp, ppint, batch_size, activation_recompute,
          optimizer_sharding, tensor_par_comm_type, fused_acts, mbs_break,
          allow_tp_overlap, allow_dp_overlap):
    num_nets = syst.num_networks

    best = []
    exe_count = 0
    good_exe_count = 0
    bad_exe_count = 0

    # 固定Megatron不使用的参数
    weight_offload = False       # 不使用ZeRO优化
    activations_offload = False  # Megatron无此参数
    optimizer_offload = False    # 不使用ZeRO
    seq_par_ag_redo = False      # 固定AG重计算
    data_par_overlap = False     # 禁用DP重叠
    tensor_par_overlap = 'none'  # 禁用TP重叠
    
    can_redo = Llm.can_redo_ag(tensor_par_comm_type, activation_recompute)
    for fused_act in fused_acts:
        for microbatch_size in Llm.get_valid_microbatch_sizes(
                app.seq_size, tp, dp, batch_size, pp):
            mbs_break_good = good_exe_count
            # 固定网络分配策略（根据Megatron实现）
            tn = 0 if tp == 1 else 0  # 假设使用默认网络
            pn = 0 if pp == 1 else 0  # 假设使用默认网络
            dn = 0 if dp == 1 else 0  # 假设使用默认网络
            exe_count += 1
            exe_json = {
                'num_procs': num_procs,
                'tensor_par': tp,
                'pipeline_par': pp,
                'data_par': dp,
                'tensor_par_net': tn,
                'pipeline_par_net': pn,
                'data_par_net': dn,
                'batch_size': batch_size,
                'microbatch_size': microbatch_size,
                'datatype': datatype,
                'fused_activation': fused_act,
                'attention_type': 'multihead',
                'activation_recompute': activation_recompute,
                'pipeline_interleaving': ppint,
                'optimizer_sharding': optimizer_sharding,
                'tensor_par_comm_type': tensor_par_comm_type,
                'tensor_par_overlap': tensor_par_overlap,
                'seq_par_ag_redo': seq_par_ag_redo,
                'data_par_overlap': data_par_overlap,
                'weight_offload': weight_offload,
                'activations_offload': activations_offload,
                'optimizer_offload': optimizer_offload,
                'training': True
            }

            if not debug:
                try:
                    # 创建并配置logger
                    logger = logging.getLogger('sub')
                    logger.propagate = False  # 禁用传播到父logger
                    if not logger.handlers:  # 避免重复添加handler
                        handler = logging.StreamHandler()
                        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
                        handler.setFormatter(formatter)
                        logger.addHandler(handler)
                        logger.setLevel(logging.INFO)
                    model = Llm(app, logger)
                    model.compile(syst, Llm.Execution.from_json(exe_json))
                    model.run(syst)
                    stats = model.get_stats_json(layers)
                    good_exe_count += 1
                    curr = (stats['total_time'], exe_json, stats)
                    best = OptimalExecution.update_list(best, curr, top_n)
                except Exception as ex:
                    # 使用更详细的错误日志
                    error_logger = logging.getLogger('error')
                    error_logger.setLevel(logging.INFO)
                    if not error_logger.handlers:
                        handler = logging.StreamHandler()
                        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
                        handler.setFormatter(formatter)
                        error_logger.addHandler(handler)
                    error_logger.error(f'Execution failed for config: {exe_json}')
                    error_logger.error(f'Error details: {str(ex)}')
                    error_logger.error(f'Error type: {type(ex).__name__}')
                    import traceback
                    error_logger.error(f'Traceback: {traceback.format_exc()}')
                    bad_exe_count += 1
            if mbs_break and good_exe_count == mbs_break_good:
                break
    return (best, exe_count, good_exe_count, bad_exe_count, tp, pp)


  @staticmethod
  def update_list(current, candidate, quantity):
    if not isinstance(candidate, list):
      current.append(candidate)
    else:
      current.extend(candidate)
    current.sort(reverse=False, key=lambda x: x[0]) # Sort in ascending order
    return current[:quantity]

calculon.CommandLine.register(OptimalExecution)
