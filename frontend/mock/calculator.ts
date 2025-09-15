export default {

    'POST /llm_training_calculator/calculator': (req: any, res: any) => {
        // 添加跨域请求头
        let result = {
            "memory_usage": {
                "optimizer": "864.562 MiB",
                "weights": "144.094 MiB",
                "weight_gradients": "156.102 MiB",
                "activation": "40.797 GiB",
                "activation_gradients": "1.914 GiB",
                "overall_usage": "43.848 GiB"
            },
            "computation": {
                "per_device_blocks": 12,
                "num_microbatches": 16,
                "batch_forward_computation_time": 1.0841839542856806,
                "microbatch_forward_computation_time": 0.06776149714285504,
                "batch_backward_computation_time": 2.0534329437902685,
                "microbatch_backward_computation_time": 0.1283395589868918
            },
            "communication": {
                "dp_comm_size": 151093248,
                "tp_comm_fw_size": 1409286144,
                "tp_comm_bw_size": 1409286144,
                "pp_comm_fw_size": 704643072,
                "pp_comm_bw_size": 704643072,
                "batch_dp_comm_time": 0.037773312,
                "batch_tp_comm_time": 0.7046430719999999,
                "batch_tp_fw_comm_time": 0.35232153599999994,
                "microbatch_tp_fw_comm_time": 0.022020095999999996,
                "batch_tp_bw_comm_time": 0.35232153599999994,
                "microbatch_tp_bw_comm_time": 0.022020095999999996,
                "batch_pp_comm_time": 0.35232153599999994,
                "batch_pp_fw_comm_time": 0.17616076799999997,
                "microbatch_pp_fw_comm_time": 0.011010047999999998,
                "batch_pp_bw_comm_time": 0.17616076799999997,
                "microbatch_pp_bw_comm_time": 0.011010047999999998
            },
            "timeline": {
                "per_device_blocks": 12,
                "num_microbatches": 16,
                "batch_forward_computation_time": 1.0841839542856806,
                "microbatch_forward_computation_time": 0.06776149714285504,
                "batch_backward_computation_time": 2.0534329437902685,
                "microbatch_backward_computation_time": 0.1283395589868918,
                "batch_dp_comm_time": 0.037773312,
                "batch_tp_comm_time": 0.7046430719999999,
                "batch_tp_fw_comm_time": 0.35232153599999994,
                "microbatch_tp_fw_comm_time": 0.022020095999999996,
                "batch_tp_bw_comm_time": 0.35232153599999994,
                "microbatch_tp_bw_comm_time": 0.022020095999999996,
                "batch_pp_comm_time": 0.35232153599999994,
                "batch_pp_fw_comm_time": 0.17616076799999997,
                "microbatch_pp_fw_comm_time": 0.011010047999999998,
                "batch_pp_bw_comm_time": 0.17616076799999997,
                "microbatch_pp_bw_comm_time": 0.011010047999999998,
                "warmup_time": 0.01140539082412681,
                "cooldown_time": 0.014252068157460144,
                "batch_total_time": 5.810847859875839
            },
            "summary": {
                "global_batch_size": 896,
                "local_batch_size": 448,
                "batch_total_time": 5.810847859875839,
                "totoal_number_of_gpus": 8,
                "total_efficiency": 0.3511
            }
        }
        res.send(result);
    }

}