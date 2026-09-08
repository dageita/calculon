import json
import logging
import sys
import unittest
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from calculon.llm.llm import Llm
from calculon.system import System
from calculon.llm.decoder_output import decoder_output_stats, VocabParallelCrossEntropy
from calculon.llm.flash_attention import FlashAttention
from calculon.llm.gated_activation import GatedActivation
from calculon.llm.layers import Linear, RMSNorm
from calculon.llm.framework_scheduler import RuntimeEventDAG


class TrainingOperatorsTest(unittest.TestCase):
    def model(self, activation_recompute="attn_only", **changes):
        app = json.loads((ROOT/"models/qwen3-0.6b.json").read_text())
        app.update(changes)
        exe = dict(num_procs=4, tensor_par=1, pipeline_par=1, data_par=4,
                   tensor_par_net=0, pipeline_par_net=0, data_par_net=0,
                   batch_size=8, microbatch_size=1, datatype="float16",
                   fused_activation=True, attention_type="multihead",
                   attention_kernel="flash", activation_recompute=activation_recompute,
                   pipeline_interleaving=1, optimizer_sharding=False,
                   tensor_par_comm_type="ar", tensor_par_overlap="none",
                   seq_par_ag_redo=False, data_par_overlap=False,
                   weight_offload=False, activations_offload=False,
                   optimizer_offload=False, training=True)
        log = logging.getLogger("test")
        system = System(json.loads((ROOT/"systems/L20.json").read_text()), log)
        model = Llm(Llm.Application(app), log)
        model.compile(system, Llm.Execution.from_json(exe))
        return model

    def test_head_scales_with_vocabulary_not_depth(self):
        a = decoder_output_stats(self.model())
        b = decoder_output_stats(self.model(num_blocks=56))
        c = decoder_output_stats(self.model(vocab_size=2*151936))
        self.assertEqual(a, b)
        self.assertGreater(c["forward"], a["forward"])
        self.assertGreater(c["gradient_bytes"], a["gradient_bytes"])

    def test_tied_embeddings_do_not_duplicate_optimizer(self):
        a = decoder_output_stats(self.model())
        b = decoder_output_stats(self.model(untied_embeddings=True))
        self.assertGreater(b["parameters"], a["parameters"])
        self.assertGreater(b["optimizer"], a["optimizer"])
        self.assertEqual(a["projection_forward"], b["projection_forward"])

    def test_flash_replaces_entire_core_and_recomputes(self):
        model = self.model()
        layers = model._llm_block
        flash = [x for x in layers if isinstance(x, FlashAttention)]
        self.assertEqual(len(flash), 1)
        self.assertTrue(flash[0].needs_recompute)
        self.assertFalse(any(x.name == "AttnBlock_Multihead_SoftMax" for x in layers))
        self.assertGreater(flash[0].get_agrad_flops(), flash[0].get_fw_flops())
        self.assertEqual(model.exe.get_json()["attention_kernel"], "flash")

    def test_flash_balanced_context_partition(self):
        model = self.model()
        a = FlashAttention("a", model.sys, 1, 1024, 16, 8, 128)
        b = FlashAttention("b", model.sys, 1, 1024, 16, 8, 128, context_partitions=2)
        a.set_bytes_per_element(2)
        b.set_bytes_per_element(2)
        self.assertEqual(a.get_fw_flops(), 2*b.get_fw_flops())
        self.assertEqual(a.get_fw_mem_accessed(), 2*b.get_fw_mem_accessed())

    def test_gated_fusion_is_not_free(self):
        model = self.model()
        layer = next(x for x in model._llm_block if isinstance(x, GatedActivation))
        self.assertGreater(layer.compute_processing_time("fw"), 0)
        self.assertGreater(layer.compute_processing_time("agrad"), 0)
        self.assertEqual(layer.get_agrad_mem_accessed()/layer.get_fw_mem_accessed(), 5/3)

    def test_loss_collectives_are_token_sized_fp32(self):
        class Network:
            def __init__(self): self.calls = []
            def collective_time(self, op, size, tp):
                self.calls.append((op, size, tp))
                return 0.001
        model = self.model()
        net = Network()
        loss = VocabParallelCrossEntropy(model.sys, 1024, 75968, 2, net)
        self.assertGreater(loss.compute_processing_time("fw"), 0)
        self.assertEqual(net.calls, [("all_reduce", 4096, 2)])

    def test_framework_runtime_is_graph_derived_and_disableable(self):
        small = self.model()
        large = self.model(num_blocks=56)
        small.run(small.sys)
        large.run(large.sys)
        a = small.get_framework_runtime_breakdown()
        b = large.get_framework_runtime_breakdown()
        self.assertGreater(a["flow_forward"], 0)
        self.assertGreater(a["flow_backward"], 0)
        self.assertEqual(a["events"]["checkpoint_nodes"], 2 * 28)
        dag = small.get_framework_runtime_dag()
        self.assertEqual(dag["backward"]["sync_count"], 1)
        self.assertGreater(dag["backward"]["sync_wait"], 0)
        optimizer = small.get_framework_boundary_dag()["optimizer"]
        self.assertEqual(optimizer["sync_count"], 2)
        self.assertGreater(optimizer["sync_wait"], 0)
        # Endpoint loss runs once per microbatch, whereas block work doubles.
        self.assertEqual(b["events"]["forward_kernels"],
                         2 * a["events"]["forward_kernels"] - 20)
        config = json.loads((ROOT/"systems/L20.json").read_text())
        config["framework_runtime"]["enabled"] = False
        system = System(config, logging.getLogger("test"))
        self.assertEqual(system.get_framework_runtime_time(
            kernels=100, autograd_nodes=100, checkpoint_nodes=10,
            scalar_syncs=2, optimizer_bytes=1 << 30), 0)

    def test_runtime_operator_profile_keeps_autograd_logical(self):
        model = self.model()
        linear = next(x for x in model._llm_block if isinstance(x, Linear))
        norm = next(x for x in model._llm_block if isinstance(x, RMSNorm))
        self.assertEqual(linear.get_framework_events("wgrad")["kernels"], 3)
        self.assertEqual(linear.get_framework_events("wgrad")["autograd_nodes"], 1)
        self.assertGreater(linear.get_framework_events("wgrad")["host_s"], 0)
        self.assertEqual(norm.get_framework_events("wgrad")["kernels"], 1)
        self.assertEqual(norm.get_framework_events("wgrad")["autograd_nodes"], 1)
        self.assertGreater(norm.get_framework_events("wgrad")["host_s"], 0)

    def test_dual_resource_scheduler_overlaps_submission_and_waits_at_sync(self):
        system = self.model().sys
        dag = RuntimeEventDAG(system)
        dag.enqueue("a", 0.1, kernels=1, host_s=0.02)
        dag.enqueue("b", 0.1, kernels=1, host_s=0.02)
        self.assertAlmostEqual(dag.elapsed, 0.22)
        self.assertLess(dag.elapsed, dag.host_work + dag.gpu_work)
        dag.synchronize("item", 0.01)
        self.assertAlmostEqual(dag.sync_wait, 0.18)
        self.assertAlmostEqual(dag.elapsed, 0.23)

    def test_full_recompute_has_one_block_checkpoint_boundary(self):
        model = self.model(activation_recompute="full")
        model.run(model.sys)
        dag = model.get_framework_runtime_dag()
        self.assertEqual(dag["backward"]["sync_count"], 1)
        self.assertEqual(dag["ffn_backward"]["sync_count"], 1)
        self.assertEqual(dag["attention_backward"]["sync_count"], 0)
        split = (dag["ffn_backward"]["flow_elapsed"] +
                 dag["attention_backward"]["flow_elapsed"])
        self.assertAlmostEqual(split, dag["backward"]["elapsed"])

    def test_vocabulary_loss_exposes_each_unfused_kernel(self):
        model = self.model()
        loss = VocabParallelCrossEntropy(model.sys, 1024, 75968, 2, None)
        self.assertEqual(loss.get_framework_events("fw")["kernels"], 6)
        self.assertEqual(loss.get_framework_events("agrad")["kernels"], 2)


if __name__ == "__main__":
    unittest.main()
