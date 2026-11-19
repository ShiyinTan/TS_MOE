import contextlib
import csv
import io
import json
import math
import random
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from examples.make_demo import generate
from tsmoe.__main__ import main
from tsmoe.candidates import (
    catalog_coordinates, consecutive_visit_distances, format_pool_size_line,
    haversine_km, neighborhood_indices, pick_radius_km, resolve_candidate_pool,
    sample_candidates, summarize_distances, summarize_pool_sizes,
)
from tsmoe.prepare import dump_json
from tsmoe.data import Categories, Event, build_split, make_batch, read_split
from tsmoe.engine import (
    Cache, GROUP_LABELS, PRIMARY_METRICS, candidates, format_eval_report, metric_row,
    ranks_of, results_run_dir,
)
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
        # Acc@1 is always computed (primary metric); only rank-1 hits.
        self.assertEqual(row["Acc@1"], 1 / 3)
        self.assertEqual(row["NDCG@1"], 1 / 3)

    def test_distance_stats_and_spatial_pools(self):
        # ~111 km per degree latitude; controlled consecutive hops.
        from datetime import datetime, timezone
        t0 = datetime(2018, 1, 1, tzinfo=timezone.utc)

        def event(poi, lat, lon, line):
            return Event(poi, "cat", 0, t0, (lat, lon), line)

        groups = {
            ("u", "a"): [event(1, 0.0, 0.0, 1), event(2, 0.0, 1.0, 2), event(3, 0.0, 2.0, 3)],
            ("u", "b"): [event(2, 0.0, 1.0, 4), event(4, 1.0, 1.0, 5)],
        }
        distances = consecutive_visit_distances(groups)
        self.assertEqual(len(distances), 3)
        self.assertAlmostEqual(distances[0], haversine_km(0.0, 0.0, 0.0, 1.0), places=6)
        stats = summarize_distances(distances)
        self.assertEqual(stats["count"], 3)
        self.assertAlmostEqual(stats["max_km"], max(distances))
        self.assertAlmostEqual(stats["mean_km"], sum(distances) / 3)
        self.assertAlmostEqual(pick_radius_km(stats, "max"), stats["max_km"])
        self.assertAlmostEqual(pick_radius_km(stats, "p90"), stats["p90_km"])
        self.assertIsNone(pick_radius_km(stats, "full"))
        pois = [1, 2, 3, 4]
        coord_list, audit = catalog_coordinates(groups, pois)
        self.assertEqual(coord_list[0], [0.0, 0.0])
        self.assertEqual(audit["counts"]["observed"], 4)
        self.assertEqual(audit["imputed"], 0)
        # POI never visited → global mean of observed catalog means (still finite).
        filled, filled_audit = catalog_coordinates(groups, [1, 99])
        self.assertEqual(filled[0], [0.0, 0.0])
        self.assertEqual(filled_audit["counts"]["global_mean"], 1)
        self.assertTrue(all(math.isfinite(v) for pair in filled for v in pair))
        coords = torch.tensor(coord_list, dtype=torch.float64)
        near = neighborhood_indices(0.0, 0.0, coords, radius_km=stats["mean_km"] * 0.5)
        # Only POI 1 is at the origin; mean_km/2 should exclude the ~111 km neighbors.
        self.assertEqual(near.tolist(), [0])
        pool, how = resolve_candidate_pool(0.0, 0.0, target_row=3, catalog_coords=coords,
                                           radius_km=1.0, catalog_size=4)
        self.assertEqual(how, "spatial")
        self.assertIn(0, pool.tolist())  # origin POI
        self.assertIn(3, pool.tolist())  # force-included target
        full, how = resolve_candidate_pool(float("nan"), 0.0, 1, coords, 10.0, 4)
        self.assertEqual(how, "fallback_full_catalog")
        self.assertEqual(full.tolist(), [0, 1, 2, 3])
        empty, how = resolve_candidate_pool(10.0, 10.0, 0, coords, 0.01, 4)
        self.assertEqual(how, "fallback_full_catalog")
        # Pool-size stats use the resolved set (after target include / fallback).
        pools = [
            resolve_candidate_pool(0.0, 0.0, 3, coords, 1.0, 4)[0],
            resolve_candidate_pool(float("nan"), 0.0, 1, coords, 10.0, 4)[0],
            resolve_candidate_pool(10.0, 10.0, 0, coords, 0.01, 4)[0],
        ]
        resolutions = ["spatial", "fallback_full_catalog", "fallback_full_catalog"]
        stats = summarize_pool_sizes(pools, resolutions)
        self.assertEqual(stats["n"], 3)
        self.assertEqual(stats["min"], 2)  # origin + force-included target
        self.assertEqual(stats["max"], 4)
        self.assertAlmostEqual(stats["mean"], (2 + 4 + 4) / 3)
        self.assertEqual(stats["median"], 4.0)
        self.assertEqual(stats["resolutions"]["spatial"], 1)
        self.assertEqual(stats["resolutions"]["fallback_full_catalog"], 2)
        line = format_pool_size_line("train", stats)
        self.assertTrue(line.startswith("candidates train: n=3 min=2 max=4 mean="))
        self.assertIn("median=4.00", line)
        self.assertIn("spatial=1", line)
        rows, labels, mask = sample_candidates(
            torch.tensor([0]), 4, 1, random.Random(0), pools=[pool])
        self.assertTrue(mask[0, 0].item())
        self.assertEqual(int(rows[0, 0]), 0)
        self.assertEqual(int(labels[0]), 0)
        self.assertEqual(int(mask[0].sum()), 2)

    def test_prepare_prints_coordinate_imputation_counts(self):
        """prepare logs per-source catalog coordinate counts from the imputation audit."""
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            directory = generate(root / "data")
            train_csv = directory / "demo_checkins_train.csv"
            with train_csv.open(encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            fields = list(rows[0])
            for row in rows:
                if row["venue_id"] == "0":
                    row["latitude"] = row["longitude"] = ""
            with train_csv.open("w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)
            buf = io.StringIO()
            with patch.object(sys, "argv", ["tsmoe", "prepare", "--data-root", str(directory),
                                            "--city", "demo", "--out", str(root / "cache")]), \
                    contextlib.redirect_stdout(buf):
                main()
            text = buf.getvalue()
            self.assertRegex(
                text,
                r"coordinate_imputation: observed=\d+ adjacent_midpoint=\d+ "
                r"trail_mean=\d+ global_mean=\d+ \(imputed=\d+/\d+\)",
            )
            self.assertIn("adjacent_midpoint=", text)
            self.assertRegex(text, r"imputed=[1-9]\d*/\d+")
            audit = json.loads((root / "cache" / "manifest.json").read_text())[
                "candidate_construction"]["coordinate_imputation"]
            self.assertGreater(audit["imputed"], 0)
            self.assertEqual(audit["counts"]["observed"] + audit["imputed"], audit["catalog_size"])

    def test_catalog_coordinate_imputation_and_strict_json(self):
        """Missing POI coords are imputed from trail neighbors; dump stays allow_nan=False."""
        from datetime import datetime, timezone
        t0 = datetime(2018, 1, 1, tzinfo=timezone.utc)

        def event(poi, coord, line):
            return Event(poi, "cat", 0, t0, coord, line)

        # POI 2 has no coords; adjacent visits at (0,0) and (0,2) → midpoint (0,1).
        groups = {
            ("u", "a"): [
                event(1, (0.0, 0.0), 1),
                event(2, None, 2),
                event(3, (0.0, 2.0), 3),
            ],
            ("u", "b"): [
                event(4, (10.0, 10.0), 4),
                event(5, None, 5),  # only one adjacent neighbor → that location
            ],
        }
        coords, audit = catalog_coordinates(groups, [1, 2, 3, 4, 5])
        self.assertEqual(coords[0], [0.0, 0.0])
        self.assertEqual(coords[1], [0.0, 1.0])
        self.assertEqual(coords[2], [0.0, 2.0])
        self.assertEqual(coords[3], [10.0, 10.0])
        self.assertEqual(coords[4], [10.0, 10.0])
        self.assertEqual(audit["counts"]["observed"], 3)
        self.assertEqual(audit["counts"]["adjacent_midpoint"], 2)
        self.assertEqual(audit["imputed"], 2)
        self.assertTrue(all(math.isfinite(v) for pair in coords for v in pair))

        # trail_mean: missing POI sandwiched by neighbors that also lack locations,
        # while other check-ins on the same trail still have coords.
        trail_only = {
            ("u", "c"): [
                event(6, (0.0, 0.0), 1),
                event(10, None, 2),
                event(7, None, 3),   # no adjacent locations (10 and 11 lack coords/means)
                event(11, None, 4),
                event(6, (4.0, 4.0), 5),
            ],
            ("u", "d"): [event(8, None, 6)],  # no trail coords → global_mean
            ("u", "e"): [event(9, (6.0, 6.0), 7)],
        }
        coords2, audit2 = catalog_coordinates(trail_only, [6, 7, 8, 9, 10, 11])
        self.assertEqual(coords2[0], [2.0, 2.0])  # observed mean of (0,0) and (4,4)
        self.assertEqual(coords2[1], [2.0, 2.0])  # trail mean for POI 7
        self.assertEqual(audit2["counts"]["trail_mean"], 1)
        self.assertEqual(coords2[2], [4.0, 4.0])  # global mean of POI 6 and 9
        self.assertEqual(audit2["counts"]["global_mean"], 1)
        self.assertEqual(coords2[3], [6.0, 6.0])
        # 10 and 11 get adjacent midpoints from neighboring observed/resolved locations.
        self.assertEqual(audit2["counts"]["adjacent_midpoint"], 2)
        self.assertTrue(all(math.isfinite(v) for pair in coords2 for v in pair))

        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "manifest.json"
            payload = {
                "radius_km": 9.2401577249972,
                "poi_coordinates": coords,
                "coordinate_imputation": audit,
            }
            dump_json(payload, path)
            loaded = json.loads(path.read_text())
            self.assertEqual(loaded["radius_km"], 9.2401577249972)
            self.assertEqual(loaded["poi_coordinates"][1], [0.0, 1.0])
            self.assertNotIn("nan", path.read_text().lower())

    def test_results_run_dir_strips_runs_prefix(self):
        with tempfile.TemporaryDirectory() as temp:
            cwd = Path(temp)
            (cwd / "runs" / "tokyo" / "model").mkdir(parents=True)
            mapped = results_run_dir(cwd / "runs" / "tokyo" / "model", cwd=cwd)
            self.assertEqual(mapped, cwd / "results" / "tokyo" / "model")
            outside = Path(temp) / "nested" / "run"
            outside.mkdir(parents=True)
            # Force a cwd that does not contain `outside` so the sibling layout applies.
            other = cwd / "other"
            other.mkdir()
            self.assertEqual(results_run_dir(outside, cwd=other),
                             outside.parent / "results" / "run")

    def test_eval_report_leads_with_acc1_and_group_labels(self):
        all_row = metric_row([1, 2, 0])
        known = metric_row([1, 2])
        head = metric_row([1, 1])
        tail = metric_row([2, 0])
        payload = dict(
            split="test",
            all_targets=all_row,
            known_only=known,
            groups={"poi/head_10_plus": head, "poi/tail_1_2": tail},
        )
        text = format_eval_report(payload)
        self.assertIn("summary  Acc@1=", text)
        self.assertIn("primary  Acc@1=", text)
        self.assertIn("NDCG@1=", text)
        self.assertIn("groups — Acc@1 / NDCG@1 / MRR", text)
        self.assertIn(GROUP_LABELS["poi/head_10_plus"], text)
        self.assertIn(GROUP_LABELS["poi/tail_1_2"], text)
        # Group summary must not fall back to @10-only headlines.
        self.assertNotIn("groups (Acc@10", text)
        self.assertEqual(PRIMARY_METRICS, ("Acc@1", "NDCG@1", "MRR"))

    def test_entire_cli(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            directory = generate(root / "data")
            cache_path, run = root / "cache", root / "run"
            run_cli("prepare", "--data-root", directory, "--city", "demo", "--out", cache_path)
            cache = Cache(cache_path)
            train = cache.split("train")
            self.assertEqual(cache.cfg["candidate_radius"], "p90")
            self.assertIsNotNone(cache.radius_km)
            self.assertEqual(cache.candidate_construction["method"], "spatial_radius")
            self.assertEqual(len(cache.meta["poi_coordinates"]), len(cache.pois))
            self.assertTrue(all(
                math.isfinite(v) for pair in cache.meta["poi_coordinates"] for v in pair))
            self.assertIn("coordinate_imputation", cache.candidate_construction)
            self.assertIn("lat", train)
            self.assertIn("lon", train)
            pools, resolutions = cache.candidate_pools(train)
            self.assertEqual(len(pools), len(train["end"]))
            self.assertTrue(any(r in ("spatial", "fallback_full_catalog") for r in resolutions))
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
            train_out = io.StringIO()
            with patch.object(sys, "argv", ["tsmoe", "train", "--cache", str(cache_path),
                                            "--out", str(run), "--epochs", "1", "--dim", "8",
                                            "--layers", "1", "--heads", "2", "--device", "cpu",
                                            "--negatives", "3", "--threads", "1"]), \
                    contextlib.redirect_stdout(train_out):
                main()
            train_log = train_out.getvalue()
            self.assertIn("candidates train:", train_log)
            self.assertIn("candidates validation:", train_log)
            self.assertIn("candidates test:", train_log)
            self.assertRegex(train_log, r"candidates train: n=\d+ min=\d+ max=\d+ mean=")
            # Metrics/reports land under results/, not beside the checkpoint.
            results = results_run_dir(run)
            self.assertEqual(results, root / "results" / "run")
            self.assertFalse((run / "eval_test").exists())
            self.assertTrue((run / "best.pt").is_file())
            pool_stats_path = results / "candidate_pool_stats.json"
            self.assertTrue(pool_stats_path.is_file())
            pool_stats = json.loads(pool_stats_path.read_text())
            self.assertEqual(set(pool_stats["splits"]), {"train", "validation", "test"})
            for name, stats in pool_stats["splits"].items():
                self.assertEqual(stats["n"], len(cache.split(name)["end"]))
                self.assertIsNotNone(stats["min"])
                self.assertIsNotNone(stats["max"])
                self.assertIsNotNone(stats["mean"])
                self.assertIsNotNone(stats["median"])
            history = json.loads((run / "history.json").read_text())
            self.assertIn("candidate_pool_stats", history)
            train_test = json.loads((results / "eval_test" / "metrics.json").read_text())
            self.assertEqual(train_test["split"], "test")
            self.assertEqual(train_test["selection"], "best_checkpoint_by_NDCG@10")
            self.assertTrue((results / "eval_test" / "report.txt").is_file())
            self.assertTrue((results / "eval_validation" / "metrics.json").is_file())
            self.assertIn("Acc@1", train_test["all_targets"])
            self.assertIn("Acc@10", train_test["all_targets"])
            self.assertEqual(train_test["primary_metrics"], ["Acc@1", "NDCG@1", "MRR"])
            self.assertEqual(train_test["candidate_construction"]["method"], "spatial_radius")
            self.assertEqual(train_test["candidate_construction"]["radius_statistic"], "p90")
            self.assertEqual(train_test["candidate_pool_stats"]["n"],
                             pool_stats["splits"]["test"]["n"])
            self.assertIn("group_notes", train_test)
            self.assertTrue(train_test.get("groups"))
            self.assertIn("poi/head_10_plus", train_test["group_labels"])
            self.assertIn("device", train_test)
            self.assertIn("timestamp_utc", train_test)
            val_metrics = json.loads((results / "eval_validation" / "metrics.json").read_text())
            self.assertIn("Acc@1", val_metrics["all_targets"])
            self.assertTrue(val_metrics.get("groups"))
            self.assertEqual(val_metrics["candidate_pool_stats"]["n"],
                             pool_stats["splits"]["validation"]["n"])
            val_report = (results / "eval_validation" / "report.txt").read_text()
            self.assertIn("summary  Acc@1=", val_report)
            self.assertIn("groups — Acc@1 / NDCG@1 / MRR", val_report)
            self.assertIn("candidates validation:", val_report)
            test_report = (results / "eval_test" / "report.txt").read_text()
            self.assertIn("summary  Acc@1=", test_report)
            self.assertIn("candidates test:", test_report)
            common = ["--cache", cache_path, "--checkpoint", run / "best.pt", "--device", "cpu"]
            run_cli("evaluate", *common, "--output", root / "evaluation.json", "--chunk", 3)
            metrics = json.loads((root / "evaluation.json").read_text())
            self.assertLess(metrics["all_targets"]["coverage"], 1)
            self.assertEqual(metrics["known_only"]["coverage"], 1)
            self.assertIn("Acc@1", metrics["all_targets"])
            self.assertIn("Acc@10", metrics["all_targets"])
            self.assertIn("MAP@20", metrics["all_targets"])
            self.assertEqual(metrics["all_targets"]["Acc@10"], metrics["all_targets"]["Recall@10"])
            self.assertEqual(metrics["primary_metrics"], ["Acc@1", "NDCG@1", "MRR"])
            self.assertTrue(metrics.get("groups"))
            # Default persistence under results/ + optional --output mirror.
            saved = json.loads((results / "eval_test" / "metrics.json").read_text())
            self.assertEqual(saved["cache_id"], metrics["cache_id"])
            self.assertTrue((results / "eval_test" / "report.txt").is_file())
            eval_report = (results / "eval_test" / "report.txt").read_text()
            self.assertIn("summary  Acc@1=", eval_report)
            self.assertTrue(
                any(label in eval_report for label in GROUP_LABELS.values())
                or "poi/" in eval_report
            )
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
