import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(1, str(ROOT / "backend"))

from app.core.calculate_repository import CalculateRepository


class BackendImportPathTest(unittest.TestCase):
    def test_frontend_execution_contract_accepts_attention_kernel(self):
        repo = CalculateRepository()
        gpu = {"name": "L20", "num_procs": 4, "network_bandwidth": 32.0}
        training = {
            "tensor_par": 1, "pipeline_par": 1, "data_par": 4,
            "expert_par": 1, "context_par": 1, "batch_size": 8,
            "microbatch_size": 1, "matrix_dtype": "bfloat16",
            "vector_dtype": "bfloat16", "attention_kernel": "flash",
            "activation_recompute": "attn_only",
        }
        execution = repo.build_exe(
            gpu, training, {}, {"network_topology": "Single machine"})
        self.assertEqual(execution.attention_kernel, "flash")
        self.assertEqual(execution.attention_type, "multihead")

    def test_kv_lora_selects_mla_without_q_lora(self):
        repo = CalculateRepository()
        gpu = {"name": "L20", "num_procs": 4, "network_bandwidth": 32.0}
        training = {
            "tensor_par": 1, "pipeline_par": 1, "data_par": 4,
            "expert_par": 1, "context_par": 1, "batch_size": 8,
            "microbatch_size": 1, "matrix_dtype": "bfloat16",
            "vector_dtype": "bfloat16", "attention_kernel": "flash",
        }
        execution = repo.build_exe(
            gpu, training, {"q_lora_rank": 0, "kv_lora_rank": 512},
            {"network_topology": "Single machine"})
        self.assertEqual(execution.attention_type, "mla")

    def test_script_entrypoint_prefers_checkout_over_stale_install(self):
        with tempfile.TemporaryDirectory() as directory:
            fake = Path(directory) / "calculon"
            fake.mkdir()
            (fake / "__init__.py").write_text(
                "raise RuntimeError('stale Calculon package was imported')\n")
            code = """
import inspect
import runpy
import sys
sys.path.insert(0, {backend!r})
runpy.run_path({main!r}, run_name='backend_main_import_test')
from calculon.llm.llm import Llm
print(inspect.getfile(Llm))
""".format(backend=str(ROOT / "backend"),
           main=str(ROOT / "backend" / "main.py"))
            env = os.environ.copy()
            env["PYTHONPATH"] = directory
            output = subprocess.check_output(
                [sys.executable, "-c", code], cwd="/tmp", env=env, text=True)
            self.assertEqual(
                Path(output.strip()).resolve(),
                (ROOT / "calculon" / "llm" / "llm.py").resolve())


if __name__ == "__main__":
    unittest.main()
