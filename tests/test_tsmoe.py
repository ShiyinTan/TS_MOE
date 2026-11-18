import contextlib
import csv
import io
import json
import random
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from examples.make_demo import generate
from tsmoe.__main__ import main
from tsmoe.data import Categories, build_split, make_batch, read_split
from tsmoe.engine import Cache, candidates, metric_row, ranks_of
from tsmoe.model import ModelConfig, TSMoE, objective
from tsmoe.quantizer import BasicSourceQuantizer
from tsmoe.surd import _self_test


def run_cli(*args):
    with patch.object(sys, "argv", ["tsmoe", *map(str, args)]), contextlib.redirect_stdout(io.StringIO()):
        main()


class TestDesign(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_analytic_distributions(self):
        with contextlib.redirect_stdout(io.StringIO()):
            _self_test()

    def test_features_windows_and_unseen_ids(self):
        with tempfile.TemporaryDirectory() as temp:
            path = generate(temp) / "demo_checkins_train.csv"
            with path.open() as f:
                rows = list(csv.DictReader(f))
            rows = [r for r in rows if r["trail_id"] == "train_1"]
            # One exact duplicate, one quarantined row, one missing historical coordinate.
            rows[1]["latitude"] = ""
            rows.append(dict(rows[-1]))
            rows.append({**rows[-1], "timestamp": "invalid"})
            with path.open("w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=rows[0])
                w.writeheader()
                w.writerows(reversed(rows))
            groups, audit = read_split(path)
            self.assertEqual((audit["duplicates_removed"], audit["invalid_timestamp_removed"]), (1, 1))
            cat = Categories(groups)
            event = next(iter(groups.values()))[0]
            from dataclasses import replace
            self.assertEqual(cat(event), cat(replace(event, category_code=123)))
            data = build_split(groups, cat, [1, 3, 5, 7], [0, 1, 2, 3], 2, "all")
            h, v, _ = make_batch(data, torch.arange(3), offline=True)
            self.assertEqual(h["P"].tolist(), [[1, -1], [1, 3], [3, 5]])
            self.assertEqual(data["P"][data["end"]].tolist(), [3, 5, 7])
            self.assertTrue((h["T"][:, 0, 4:] == 0).all())
            self.assertTrue((h["L"][:, 0, 7] == 0).all())
            self.assertEqual(h["L"][2, 0, 6].item(), 0)
            self.assertEqual(h["L"][2, 1, 7].item(), 0)
            q = BasicSourceQuantizer(max_length=2, n_clusters=8)
            summary = q.summarize(h, v)
            self.assertEqual([a.shape[1] for a in summary.values()], [193, 193, 19, 25])
            changed = {**h, "C": h["C"].masked_fill(v, 17)}
            alt = q.summarize(changed, v)
            for s in "PTL":
                torch.testing.assert_close(summary[s], alt[s])
            q.fit(h, v)
            q.save(Path(temp) / "q.pt")
            q2 = BasicSourceQuantizer.load(Path(temp) / "q.pt")
            self.assertTrue(torch.equal(q.transform(changed, v), q2.transform(changed, v)))

    def test_sparse_dispatch_masks_and_gradients(self):
        torch.manual_seed(1)
        model = TSMoE(5, 3, 3, ModelConfig(8, 2, 1, 0., 4))
        h = {"P": torch.tensor([[2, 3, 0]]), "C": torch.tensor([[2, 4, 0]]),
             "T": torch.randn(1, 3, 6), "L": torch.randn(1, 3, 8)}
        valid = torch.tensor([[True, True, False]])
        rows = torch.tensor([[0, 3, -1]])
        mask = rows >= 0
        p = torch.zeros(1, 3, 26)
        p[..., 14] = 1.
        p.requires_grad_(True)
        w = torch.ones(1, 3, requires_grad=True)
        with torch.no_grad():
            for param in model.router.parameters():
                param.zero_()
        out = model(h, valid, rows, p, w, candidate_mask=mask, details=True)
        self.assertTrue(torch.isneginf(out["score"][0, 2]))
        self.assertEqual(out["top"][0, 0].tolist(), [0, 1, 2, 3])
        self.assertTrue((out["mass"] == 0).all())
        expected = out["base"] + out["moe_bias"] + out["contributions"].sum(-1)
        torch.testing.assert_close(out["score"][mask], expected[mask])
        loss, _ = objective(out, torch.tensor([0]))
        loss.backward()
        self.assertIsNone(p.grad)
        self.assertIsNone(w.grad)
        self.assertTrue(all(p.grad is None for expert in model.experts[4:] for p in expert.parameters()))
        self.assertTrue(all(next(expert.parameters()).grad is not None for expert in model.experts[:4]))
        off = model(h, valid, rows, p, torch.zeros_like(w), candidate_mask=mask)
        _, (_, aux, bal) = objective(off, torch.tensor([1]))
        self.assertEqual((float(aux), float(bal)), (0., 0.))

    def test_candidate_chunks_padding_and_prior_ablation(self):
        torch.manual_seed(8)
        model = TSMoE(6, 3, 3, ModelConfig(8, 2, 1, 0., 4, use_surd=False)).eval()
        valid = torch.tensor([[True, True, False]])
        h = {"P": torch.tensor([[2, 3, 0]]), "C": torch.tensor([[3, 4, 0]]),
             "T": torch.randn(1, 3, 6), "L": torch.randn(1, 3, 8)}
        rows = torch.arange(6)[None, :]
        with torch.no_grad():
            p, w = torch.rand(1, 6, 26).softmax(-1), torch.rand(1, 6)
            a = model(h, valid, rows, p, w)
            changes = {s: x.clone() for s, x in h.items()}
            for s, x in changes.items():
                x[:, -1] = 2 if s in ("P", "C") else 100
            hist = model.encode(changes, valid)
            parts = [model.score(hist, valid, rows[:, sl], 1 - p[:, sl], 1 - w[:, sl])["score"]
                     for sl in (slice(None, 2), slice(2, None))]
            torch.testing.assert_close(a["score"], torch.cat(parts, 1))
            self.assertTrue((a["confidence"] == 0).all())

    def test_ranking_and_sampling(self):
        target = torch.tensor([0, 5, 2])
        rows, label = candidates(target, 6, 3, random.Random(77))
        self.assertEqual(rows.gather(1, label[:, None]).flatten().tolist(), target.tolist())
        self.assertTrue(all(len(set(row)) == 4 for row in rows.tolist()))
        ranks = ranks_of(torch.ones(3, 6), torch.tensor([0, 3, -1]))
        self.assertEqual(ranks.tolist(), [1, 4, 0])
        row = metric_row(ranks)
        self.assertEqual(row["Recall@5"], 2/3)
        self.assertEqual(row["Acc@5"], row["Hit@5"])
        self.assertEqual(row["Acc@5"], row["Recall@5"])
        self.assertEqual(row["MAP@5"], row["MRR@5"])
        self.assertAlmostEqual(row["mean_rank"], 2.5)
        self.assertAlmostEqual(row["MRR"], (1 + 0.25) / 3)

    def test_entire_cli(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            directory = generate(root / "data")
            cache_path, run = root / "cache", root / "run"
            run_cli("prepare", "--data-root", directory, "--city", "demo", "--out", cache_path)
            cache = Cache(cache_path)
            train = cache.split("train")
            run_cli("inspect", "--cache", cache_path, "--poi-id", cache.pois[0], "--sample-index", 0,
                    "--output", root / "trace.json")
            trace = json.loads((root / "trace.json").read_text())
            self.assertEqual(len(trace["sample"]["q_state"]), 4)
            self.assertEqual(len(trace["specific_bits"]), 15)
            self.assertEqual(len(trace["raw_surd"]), 26)
            for tab in cache.meta["tables"][:-1]:
                if tab["reference_events"]:
                    self.assertLess(tab["last_reference_target_utc_us"], tab["prediction_cutoff_utc_us"])
                ref = (train["us"][train["end"]] < tab["prediction_cutoff_utc_us"])
                expected = (train["poi_token"][train["end"][ref]] - 2).bincount(minlength=len(cache.pois))
                table = cache.table(tab["name"], "cpu")
                torch.testing.assert_close(table.target_count, expected.float())
            self.assertEqual(cache.meta["tables"][-1]["reference_events"], len(train["end"]))
            run_cli("train", "--cache", cache_path, "--out", run, "--epochs", 1, "--dim", 8,
                    "--layers", 1, "--heads", 2, "--device", "cpu", "--negatives", 3, "--threads", 1)
            # train itself writes best-val + final-test result dirs.
            train_test = json.loads((run / "eval_test" / "metrics.json").read_text())
            self.assertEqual(train_test["split"], "test")
            self.assertEqual(train_test["selection"], "best_checkpoint_by_NDCG@10")
            self.assertTrue((run / "eval_test" / "report.txt").is_file())
            self.assertTrue((run / "eval_validation" / "metrics.json").is_file())
            self.assertIn("Acc@10", train_test["all_targets"])
            self.assertIn("device", train_test)
            self.assertIn("timestamp_utc", train_test)
            common = ["--cache", cache_path, "--checkpoint", run / "best.pt", "--device", "cpu"]
            run_cli("evaluate", *common, "--output", root / "evaluation.json", "--chunk", 3)
            metrics = json.loads((root / "evaluation.json").read_text())
            self.assertLess(metrics["all_targets"]["coverage"], 1)
            self.assertEqual(metrics["known_only"]["coverage"], 1)
            self.assertIn("Acc@10", metrics["all_targets"])
            self.assertIn("MAP@20", metrics["all_targets"])
            self.assertEqual(metrics["all_targets"]["Acc@10"], metrics["all_targets"]["Recall@10"])
            # Default persistence beside checkpoint + optional --output mirror.
            saved = json.loads((run / "eval_test" / "metrics.json").read_text())
            self.assertEqual(saved["cache_id"], metrics["cache_id"])
            self.assertTrue((run / "eval_test" / "report.txt").is_file())
            run_cli("explain", *common, "--output", root / "case.json", "--sample-index", 0)
            case = json.loads((root / "case.json").read_text())
            self.assertAlmostEqual(case["final_score"], case["base_score"] + case["moe_bias_contribution"]
                                   + sum(case["expert_score_contributions"].values()), places=5)
            run_cli("recommend", *common, "--history", directory / "demo_checkins_validation.csv",
                    "--top-n", 3, "--output", root / "pred.json")
            ranked = json.loads((root / "pred.json").read_text())
            self.assertEqual(len(ranked), 6)
            self.assertTrue(all(len(r["recommendations"]) == 3 for r in ranked))

            alt = root / "last"
            run_cli("prepare", "--data-root", directory.parent, "--city", "demo", "--out", alt,
                    "--train-targets", "last", "--eval-targets", "last")
            for split in ("train", "validation", "test"):
                info = Cache(alt).meta["audit"][split]
                self.assertEqual(info["samples"], info["trails"])


if __name__ == "__main__":
    unittest.main()
