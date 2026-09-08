import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(1, str(ROOT / "backend"))
sys.path.insert(2, str(ROOT / "calibration" / "l20"))
sys.path.insert(2, str(ROOT / "calibration" / "l20"))

from app.config import _MODEL_CATALOG
from calculon.llm.llm import Llm
from model_catalog import CASES, application_from_hf, default_model_path


class ModelArchitectureTest(unittest.TestCase):
    def load(self, name):
        return json.loads((ROOT / "models" / name).read_text())

    def test_catalog_and_json_directory_are_one_to_one(self):
        catalog_files = {filename for _, filename in _MODEL_CATALOG}
        json_files = {path.name for path in (ROOT / "models").glob("*.json")}
        self.assertEqual(catalog_files, json_files)

    def test_all_models_compile_as_applications(self):
        for path in (ROOT / "models").glob("*.json"):
            with self.subTest(path=path.name):
                app = Llm.Application(json.loads(path.read_text()))
                self.assertLessEqual(app.seq_size, app.max_position_embeddings)
                self.assertGreater(app.num_parameters(), 0)

    def test_calibration_cases_match_local_hf_configs_and_model_jsons(self):
        for case, (_, _, filename) in CASES.items():
            with self.subTest(case=case):
                expected = application_from_hf(case, default_model_path(case), 1024)
                self.assertEqual(self.load(filename), expected)

    def test_all_requested_qwen3_sizes(self):
        expected = {
            "qwen3-0.6b.json": (1024, 3072, 28, 16, 8, 128),
            "qwen3-1.7b.json": (2048, 6144, 28, 16, 8, 128),
            "qwen3-4b.json": (2560, 9728, 36, 32, 8, 128),
            "qwen3-8b.json": (4096, 12288, 36, 32, 8, 128),
            "qwen3-14b.json": (5120, 17408, 40, 40, 8, 128),
        }
        for filename, values in expected.items():
            with self.subTest(filename=filename):
                cfg = self.load(filename)
                actual = (cfg["hidden"], cfg["feedforward"], cfg["num_blocks"],
                          cfg["attn_heads"], cfg["kv_heads"], cfg["attn_size"])
                self.assertEqual(actual, values)

    def test_deepseek_v2_lite_uses_direct_q_mla_and_moe(self):
        v2 = Llm.Application(self.load("deepseek-v2-lite.json"))
        coder = Llm.Application(self.load("deepseek-coder-v2-lite.json"))
        self.assertTrue(v2.is_mla)
        self.assertEqual((v2.q_lora_rank, v2.kv_lora_rank), (0, 512))
        self.assertEqual((v2.num_dense_blocks, v2.num_moe_blocks), (1, 26))
        self.assertEqual(v2.num_parameters(), coder.num_parameters())

    def test_qwen3_hf_architectures(self):
        qwen = self.load("qwen3-0.6b.json")
        self.assertEqual(
            {key: qwen[key] for key in (
                "hidden", "feedforward", "attn_heads", "kv_heads",
                "attn_size", "num_blocks", "vocab_size",
                "max_position_embeddings")},
            dict(hidden=1024, feedforward=3072, attn_heads=16, kv_heads=8,
                 attn_size=128, num_blocks=28, vocab_size=151936,
                 max_position_embeddings=40960))
        moe = Llm.Application(self.load("qwen3-30b-a3b.json"))
        self.assertEqual((moe.router_score_func, moe.num_moe_blocks),
                         ("softmax", 48))
        self.assertAlmostEqual(moe.num_parameters() / 1e9, 30.532, places=2)

    def test_deepseek_v3_hf_architecture(self):
        app = Llm.Application(self.load("deepseek-v3-671b.json"))
        self.assertEqual((app.num_dense_blocks, app.num_moe_blocks), (3, 58))
        self.assertEqual(
            (app.router_score_func, app.router_n_groups,
             app.router_topk_groups, app.router_has_bias),
            ("sigmoid", 8, 4, True))
        self.assertAlmostEqual(app.num_parameters() / 1e9, 671.026, places=2)
        self.assertAlmostEqual(app.num_activated_parameters() / 1e9,
                               37.552, places=2)

    def test_palm_uses_mqa_swiglu_and_parallel_block(self):
        app = Llm.Application(self.load("palm-540B.json"))
        self.assertEqual(app.kv_heads, 1)
        self.assertEqual(app.ffn_type, "swiglu")
        self.assertTrue(app.parallel_block)
        self.assertAlmostEqual(app.num_parameters() / 1e9, 540.356, places=2)


if __name__ == "__main__":
    unittest.main()
