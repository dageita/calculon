import json
import logging
import unittest
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from calculon.llm import Llm
from calculon.llm.layers import Layer
from calculon.system import System



def execution_config(**updates):
    cfg = {
        'num_procs': 1, 'tensor_par': 1, 'pipeline_par': 1, 'data_par': 1,
        'expert_par': 1, 'context_par': 1, 'tensor_par_net': 0,
        'pipeline_par_net': 0, 'data_par_net': 0, 'expert_par_net': 0,
        'context_par_net': 0, 'batch_size': 1, 'microbatch_size': 1,
        'datatype': 'bfloat16', 'matrix_dtype': 'bfloat16',
        'vector_dtype': 'bfloat16', 'fused_activation': True,
        'attention_type': 'multihead', 'activation_recompute': 'none',
        'pipeline_interleaving': 1, 'optimizer_sharding': False,
        'tensor_par_comm_type': 'ar', 'tensor_par_overlap': 'none',
        'seq_par_ag_redo': False, 'data_par_overlap': False,
        'weight_offload': False, 'activations_offload': False,
        'optimizer_offload': False, 'training': True,
    }
    cfg.update(updates)
    return cfg


class OptimizerStrategyTest(unittest.TestCase):
    def test_megatron_defaults_are_backfilled(self):
        exe = Llm.Execution.from_json(execution_config())
        self.assertFalse(exe.optimizer_sharding)
        self.assertFalse(exe.use_precision_aware_optimizer)
        self.assertEqual(exe.main_grads_dtype, 'fp32')
        self.assertEqual(exe.main_params_dtype, 'fp32')
        self.assertEqual(exe.exp_avg_dtype, 'fp32')
        self.assertEqual(exe.exp_avg_sq_dtype, 'fp32')
        self.assertFalse(exe.optimizer_offload)
        self.assertEqual(exe.optimizer_offload_fraction, 1.0)
        self.assertTrue(exe.pin_cpu_grads)
        self.assertTrue(exe.pin_cpu_params)

    def test_distributed_optimizer_is_valid_at_dp_one(self):
        exe = Llm.Execution.from_json(execution_config(
            optimizer_sharding=True,
            use_precision_aware_optimizer=True,
            main_grads_dtype='bf16',
            main_params_dtype='fp16',
            exp_avg_dtype='fp16',
            exp_avg_sq_dtype='fp8',
        ))
        self.assertEqual(exe.data_par, 1)
        self.assertTrue(exe.optimizer_sharding)

    def test_precision_aware_requires_distributed_optimizer(self):
        with self.assertRaises(Llm.Error):
            Llm.Execution.from_json(execution_config(
                use_precision_aware_optimizer=True))

    def test_precision_aware_state_sizes_drive_capacity_and_traffic(self):
        system = System(json.loads((ROOT / 'systems/L20.json').read_text()),
                        logging.getLogger())
        layer = Layer('weights', system, weight_space=100, weight_grads=100)
        layer.set_bytes_per_element(2)
        layer.configure_optimizer('bf16', 'fp16', 'fp16', 'fp8')
        layer.shard_optimizer(2)
        self.assertEqual(layer.get_weight_grad(), 100)
        self.assertEqual(layer.get_optimizer(), 250)
        self.assertEqual(layer.get_optim_step_mem_accessed(), 700)


if __name__ == '__main__':
    unittest.main()
