"""Unit tests for recipe config loading (no GPU / no torch import)."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tsmoe.config import config_to_flags, expand_config_argv, load_config


ROOT = Path(__file__).resolve().parents[1]


class ConfigTests(unittest.TestCase):
    def test_tokyo_peak_extends_win_and_base(self):
        cfg = load_config(ROOT / "configs/tokyo/peak.yaml")
        self.assertEqual(cfg["city"], "tokyo")
        self.assertEqual(cfg["recipe"], "peak")
        self.assertEqual(cfg["recipe_id"], "on_rev_soft10r")
        self.assertEqual(cfg["dim"], 256)
        self.assertEqual(cfg["lambda_bpr"], 0.29)
        self.assertEqual(cfg["transition_attn"], "gate")
        self.assertEqual(cfg["geo_score_bias"], 0.15)
        self.assertEqual(cfg["soft_target_prior"], "revisit")
        self.assertEqual(cfg["cache"], "runs/tokyo/cache_abl_tgt_last")
        self.assertEqual(cfg["out"], "runs/tokyo/on_rev_soft10r")

    def test_win_recipe_id(self):
        cfg = load_config(ROOT / "configs/recipes/win.yaml")
        self.assertEqual(cfg["recipe_id"], "hc_sg_Atime_bpr029_ssb026")
        self.assertEqual(cfg["surd_bias_gate"], "unique_p")
        self.assertEqual(cfg["candidate_context_time"], "last_slot")

    def test_base_dim128(self):
        cfg = load_config(ROOT / "configs/shanghai/base.yaml")
        self.assertEqual(cfg["dim"], 128)
        self.assertEqual(cfg["out"], "runs/shanghai/base")

    def test_config_to_flags_and_cli_override(self):
        argv = expand_config_argv([
            "train", "--config", str(ROOT / "configs/tokyo/win.yaml"),
            "--device", "cpu",
        ])
        self.assertEqual(argv[0], "train")
        self.assertIn("--dim", argv)
        self.assertEqual(argv[argv.index("--dim") + 1], "256")
        # last --device wins
        devices = [argv[i + 1] for i, t in enumerate(argv) if t == "--device"]
        self.assertEqual(devices[-1], "cpu")

    def test_evaluate_strips_train_only_keys(self):
        argv = expand_config_argv([
            "evaluate",
            "--config", str(ROOT / "configs/tokyo/peak.yaml"),
            "--checkpoint", "runs/tokyo/on_rev_soft10r/best.pt",
        ])
        self.assertNotIn("--out", argv)
        self.assertNotIn("--lambda-bpr", argv)
        self.assertIn("--cache", argv)
        self.assertIn("--checkpoint", argv)

    def test_json_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "mini.json"
            path.write_text(json.dumps({
                "cache": "runs/x/cache",
                "out": "runs/x/out",
                "dim": 128,
                "seed": 7,
            }))
            cfg = load_config(path)
            flags = config_to_flags(cfg)
            self.assertEqual(flags[flags.index("--seed") + 1], "7")


if __name__ == "__main__":
    unittest.main()
