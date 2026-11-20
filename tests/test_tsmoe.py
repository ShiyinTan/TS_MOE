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
    build_user_frequent, candidate_construction_record, catalog_coordinates,
    catalog_visit_counts, consecutive_visit_distances, format_pool_size_line,
    haversine_km, neighborhood_indices, pick_popular_radius_km, pick_radius_km,
    popular_indices, resolve_candidate_pool, sample_candidates,
    summarize_distances, summarize_pool_sizes,
)
from tsmoe.prepare import dump_json
from tsmoe.counts import (
    COUNT_FEAT_DIM, DEFAULT_VISIT_BINS, CausalCountTables, chronological_indices,
    equal_frequency_edges, gather_batch_context, sample_hop, visit_bin_id,
)
from tsmoe.data import (
    FEATURE_METHOD, L_DIM, L_STEP_VALID_INDEX, STHGCN_BASIS_DIM, T_DIM,
    T_HOUR_DIM, T_RELATIVE_SLICE, T_WEEKDAY_DIM, Categories, Event,
    build_split, make_batch, numeric_features, read_split, sthgcn_basis_freqs,
    sthgcn_phi,
)
from tsmoe.ablation_summary import (
    collect_summary_rows, expected_settings_for_run_name, extract_recorded_settings,
    leaf_run_name, parse_key_flags, validate_run_name_settings, write_summary_table,
)
from tsmoe.engine import (
    Cache, GROUP_LABELS, PRIMARY_METRICS, ablation_row_from_eval, append_ablation_log,
    candidates, format_ablation_key_flags, format_eval_report, metric_row,
    model_config_from_checkpoint, ranks_of, results_root_dir, results_run_dir,
    run_name_for_results,
)
from tsmoe.features import (
    GEO_FEAT_DIM, build_poi_cat_table, delta_geo_features, gather_side_features,
)
from tsmoe.model import (
    ModelConfig, TSMoE, TransitionCrossAttention, apply_surd_gate,
    compute_surd_bias_gate, objective,
)
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
            data, _ = build_split(groups, cat, [1, 3, 5, 7], [0, 1, 2, 3], 2, "all")
            h, v, _ = make_batch(data, torch.arange(3), offline=True)
            self.assertEqual(h["P"].tolist(), [[1, -1], [1, 3], [3, 5]])
            self.assertEqual(data["P"][data["end"]].tolist(), [3, 5, 7])
            self.assertEqual(tuple(h["T"].shape[-1:]), (T_DIM,))
            self.assertEqual(tuple(h["L"].shape[-1:]), (L_DIM,))
            self.assertTrue((h["T"][:, 0, T_RELATIVE_SLICE] == 0).all())
            self.assertTrue((h["L"][:, 0, :STHGCN_BASIS_DIM] == 0).all())
            self.assertTrue((h["L"][:, 0, L_STEP_VALID_INDEX] == 0).all())
            self.assertEqual(h["L"][2, 0, STHGCN_BASIS_DIM].item(), 0)  # missing coord
            self.assertEqual(h["L"][2, 1, L_STEP_VALID_INDEX].item(), 0)
            q = BasicSourceQuantizer(max_length=2, n_clusters=8)
            summary = q.summarize(h, v)
            self.assertEqual(q.config["feature_method"], FEATURE_METHOD)
            self.assertEqual(
                [a.shape[1] for a in summary.values()],
                [193, 193, 3 * T_DIM + 1, 3 * L_DIM + 1],
            )
            changed = {**h, "C": h["C"].masked_fill(v, 17)}
            alt = q.summarize(changed, v)
            for s in "PTL":
                torch.testing.assert_close(summary[s], alt[s])
            q.fit(h, v)
            q.save(Path(temp) / "q.pt")
            q2 = BasicSourceQuantizer.load(Path(temp) / "q.pt")
            self.assertTrue(torch.equal(q.transform(changed, v), q2.transform(changed, v)))

    def test_sthgcn_tl_feature_dims_and_phi(self):
        from datetime import datetime, timezone, timedelta

        freqs = sthgcn_basis_freqs()
        self.assertEqual(len(freqs), STHGCN_BASIS_DIM)
        self.assertAlmostEqual(freqs[0], 1.0)
        self.assertAlmostEqual(freqs[-1], 1e-9)
        phi0 = sthgcn_phi(0.0)
        self.assertEqual(len(phi0), STHGCN_BASIS_DIM)
        self.assertTrue(all(abs(v - 1.0) < 1e-12 for v in phi0))
        # Δt=1 hour: cos(1 * 1) on the first basis (matches TimeEncoder init).
        self.assertAlmostEqual(sthgcn_phi(1.0)[0], math.cos(1.0), places=12)

        t0 = datetime(2018, 1, 1, 15, 0, tzinfo=timezone.utc)  # Monday hour 15
        a = Event(1, "cat", 0, t0, (35.0, 139.0), 1)
        b = Event(2, "cat", 0, t0 + timedelta(hours=2), (35.1, 139.1), 2)
        t_feat, l_feat = numeric_features(b, a)
        self.assertEqual(len(t_feat), T_DIM)
        self.assertEqual(len(l_feat), L_DIM)
        self.assertEqual(sum(t_feat[:T_HOUR_DIM]), 1.0)
        self.assertEqual(t_feat[b.time.hour], 1.0)
        self.assertEqual(sum(t_feat[T_HOUR_DIM:T_HOUR_DIM + T_WEEKDAY_DIM]), 1.0)
        self.assertEqual(t_feat[T_HOUR_DIM + b.time.weekday()], 1.0)
        self.assertEqual(t_feat[-1], 1.0)
        torch.testing.assert_close(
            torch.tensor(t_feat[T_HOUR_DIM + T_WEEKDAY_DIM:-1]),
            torch.tensor(sthgcn_phi(2.0)),
        )
        self.assertEqual(l_feat[STHGCN_BASIS_DIM], 1.0)
        self.assertEqual(l_feat[L_STEP_VALID_INDEX], 1.0)
        from tsmoe.candidates import haversine_km
        dist = haversine_km(*a.coord, *b.coord)
        torch.testing.assert_close(
            torch.tensor(l_feat[:STHGCN_BASIS_DIM]),
            torch.tensor(sthgcn_phi(dist)),
        )
        t0_feat, l0_feat = numeric_features(a, None)
        self.assertEqual(t0_feat[-1], 0.0)
        self.assertTrue(all(v == 0.0 for v in t0_feat[T_HOUR_DIM + T_WEEKDAY_DIM:-1]))
        self.assertEqual(l0_feat[L_STEP_VALID_INDEX], 0.0)
        self.assertTrue(all(v == 0.0 for v in l0_feat[:STHGCN_BASIS_DIM]))
        self.assertEqual(FEATURE_METHOD, "sthgcn_tl_features_v1")
        model = TSMoE(3, 2, 2, ModelConfig(8, 2, 1, 0., 2, use_surd=False))
        self.assertEqual(model.projections["T"][0].in_features, T_DIM)
        self.assertEqual(model.projections["L"][0].in_features, L_DIM)

    def test_sparse_dispatch_masks_and_gradients(self):
        torch.manual_seed(1)
        # Hybrid + zeroed sequence router isolates λ_prior · SURD bias (legacy path).
        model = TSMoE(5, 3, 3, ModelConfig(8, 2, 1, 0., 4, router_mode="hybrid",
                                           hybrid_lambda=1.))
        h = {"P": torch.tensor([[2, 3, 0]]), "C": torch.tensor([[2, 4, 0]]),
             "T": torch.randn(1, 3, T_DIM), "L": torch.randn(1, 3, L_DIM)}
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

    def test_router_modes_dispatch_and_shapes(self):
        torch.manual_seed(0)
        h = {"P": torch.tensor([[2, 3, 0]]), "C": torch.tensor([[2, 4, 0]]),
             "T": torch.randn(1, 3, T_DIM), "L": torch.randn(1, 3, L_DIM)}
        valid = torch.tensor([[True, True, False]])
        rows = torch.tensor([[0, 1, 2]])
        prior = torch.zeros(1, 3, 26)
        # Distinct U/S mass on expert slots 0 and 14 (last synergy) so hard_prior is peaked.
        prior[0, 0, 0] = 0.8
        prior[0, 0, 15] = 0.2
        prior[0, 1, 1] = 1.0
        prior[0, 2, 3] = 0.5
        prior[0, 2, 25] = 0.5
        conf = torch.ones(1, 3)
        mass = torch.cat((prior[..., :4], prior[..., 15:]), -1)
        expected_hard = torch.log1p(15 * mass)

        for mode in ("hard_prior", "surd_mlp", "hybrid"):
            cfg = ModelConfig(8, 2, 1, 0., 4, router_mode=mode, hybrid_lambda=0.1)
            model = TSMoE(5, 3, 3, cfg).eval()
            with torch.no_grad():
                out = model(h, valid, rows, prior, conf, details=True)
            self.assertEqual(out["router_mode"], mode)
            self.assertEqual(tuple(out["log_pi"].shape), (1, 3, 15))
            self.assertEqual(tuple(out["top"].shape), (1, 3, 4))
            self.assertEqual(tuple(out["logits"].shape), (1, 3, 15))
            self.assertEqual(tuple(out["mass"].shape), (1, 3, 15))
            self.assertEqual(tuple(out["gate"].shape), (1, 3))
            torch.testing.assert_close(out["mass"], mass)

        hard = TSMoE(5, 3, 3, ModelConfig(8, 2, 1, 0., 4, router_mode="hard_prior")).eval()
        with torch.no_grad():
            # History change must not move hard_prior routing logits (SURD-only).
            a = hard(h, valid, rows, prior, conf, details=True)
            h2 = {**h, "P": torch.tensor([[1, 4, 0]])}
            b = hard(h2, valid, rows, prior, conf, details=True)
            torch.testing.assert_close(a["logits"], b["logits"])
            torch.testing.assert_close(a["logits"], expected_hard)
            self.assertTrue((a["learned"] == 0).all())
            # Top expert for candidate 0 is Unique-P (index 0).
            self.assertEqual(int(a["top"][0, 0, 0]), 0)

        # surd_mlp / hybrid: sequence change can move hybrid learned term; surd_mlp stays SURD-only.
        surd = TSMoE(5, 3, 3, ModelConfig(8, 2, 1, 0., 4, router_mode="surd_mlp")).eval()
        hybrid = TSMoE(5, 3, 3, ModelConfig(8, 2, 1, 0., 4, router_mode="hybrid",
                                            lambda_prior=1., hybrid_lambda=0.1)).eval()
        with torch.no_grad():
            s0 = surd(h, valid, rows, prior, conf, details=True)
            s1 = surd(h2, valid, rows, prior, conf, details=True)
            torch.testing.assert_close(s0["logits"], s1["logits"])
            hy0 = hybrid(h, valid, rows, prior, conf, details=True)
            hy1 = hybrid(h2, valid, rows, prior, conf, details=True)
            self.assertFalse(torch.allclose(hy0["learned"], hy1["learned"]))
            torch.testing.assert_close(hy0["logits"],
                                       hy0["bias"] + hybrid.cfg.hybrid_lambda * hy0["learned"])

        legacy = model_config_from_checkpoint(dict(
            dim=8, heads=2, layers=1, dropout=0., top_k=4, temperature=1.,
            lambda_prior=.1, use_surd=True))
        self.assertEqual(legacy.router_mode, "hybrid")
        self.assertEqual(legacy.hybrid_lambda, 1.0)
        self.assertEqual(ModelConfig().router_mode, "hard_prior")
        with self.assertRaises(ValueError):
            ModelConfig(router_mode="nope")

    def test_candidate_chunks_padding_and_prior_ablation(self):
        torch.manual_seed(8)
        model = TSMoE(6, 3, 3, ModelConfig(8, 2, 1, 0., 4, use_surd=False)).eval()
        valid = torch.tensor([[True, True, False]])
        h = {"P": torch.tensor([[2, 3, 0]]), "C": torch.tensor([[3, 4, 0]]),
             "T": torch.randn(1, 3, T_DIM), "L": torch.randn(1, 3, L_DIM)}
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
        self.assertAlmostEqual(pick_radius_km(stats, "p80"), stats["p80_km"])
        self.assertAlmostEqual(pick_radius_km(stats, "p70"), stats["p70_km"])
        self.assertLessEqual(stats["p70_km"], stats["p80_km"])
        self.assertLessEqual(stats["p80_km"], stats["p90_km"])
        self.assertIsNone(pick_radius_km(stats, "full"))
        self.assertIsNone(pick_radius_km(stats, "random"))
        self.assertAlmostEqual(pick_popular_radius_km(stats, 10.0, "p80", 2.0), stats["p80_km"])
        self.assertAlmostEqual(pick_popular_radius_km(stats, 10.0, "p70", 2.0), stats["p70_km"])
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
        self.assertEqual(how, "multi_source")
        self.assertIn(0, pool.tolist())  # origin POI
        self.assertIn(3, pool.tolist())  # force-included target
        full, how = resolve_candidate_pool(float("nan"), 0.0, 1, coords, 10.0, 4)
        self.assertEqual(how, "fallback_full_catalog")
        self.assertEqual(full.tolist(), [0, 1, 2, 3])
        empty, how = resolve_candidate_pool(10.0, 10.0, 0, coords, 0.01, 4)
        self.assertEqual(how, "fallback_full_catalog")
        # Sparse neighborhood + random negatives enlarges the pool; GT stays.
        with_random, how = resolve_candidate_pool(
            0.0, 0.0, target_row=3, catalog_coords=coords, radius_km=1.0, catalog_size=4,
            random_negatives=2, random_negatives_seed=7, sample_index=0)
        self.assertEqual(how, "multi_source")
        self.assertIn(3, with_random.tolist())
        self.assertGreaterEqual(int(with_random.numel()), 3)  # origin + target + ≥1 random
        again, _ = resolve_candidate_pool(
            0.0, 0.0, target_row=3, catalog_coords=coords, radius_km=1.0, catalog_size=4,
            random_negatives=2, random_negatives_seed=7, sample_index=0)
        self.assertEqual(sorted(with_random.tolist()), sorted(again.tolist()))
        # Random-only: no spatial / popular / user-frequent; GT + seeded draws.
        rnd_only, how = resolve_candidate_pool(
            0.0, 0.0, target_row=3, catalog_coords=coords, radius_km=None, catalog_size=4,
            popular_top_k=100, user_frequent_rows=[0, 1],
            random_negatives=2, random_negatives_seed=7, sample_index=0,
            random_only=True)
        self.assertEqual(how, "random_pool")
        self.assertIn(3, rnd_only.tolist())
        self.assertEqual(int(rnd_only.numel()), 3)  # GT + 2 random
        rnd_again, _ = resolve_candidate_pool(
            0.0, 0.0, target_row=3, catalog_coords=coords, radius_km=None, catalog_size=4,
            random_negatives=2, random_negatives_seed=7, sample_index=0,
            random_only=True)
        self.assertEqual(sorted(rnd_only.tolist()), sorted(rnd_again.tolist()))
        record = candidate_construction_record(
            "random", stats, None, random_negatives=50, random_negatives_seed=0)
        self.assertEqual(record["method"], "random_pool")
        self.assertEqual(record["sources"], ["random_negatives"])
        self.assertEqual(record["random_negatives"], 50)
        self.assertEqual(record["popular_top_k"], 0)
        self.assertIsNone(record["empty_union_fallback"])
        # Popular + user-frequent enrich an otherwise tiny neighborhood.
        popularity = torch.tensor([1.0, 50.0, 5.0, 2.0], dtype=torch.float64)
        rich, how = resolve_candidate_pool(
            0.0, 0.0, target_row=3, catalog_coords=coords, radius_km=1.0, catalog_size=4,
            popular_radius_km=200.0, popularity=popularity, popular_top_k=2,
            user_frequent_rows=[2])
        self.assertEqual(how, "multi_source")
        self.assertIn(0, rich.tolist())
        self.assertIn(1, rich.tolist())  # most popular within large radius
        self.assertIn(2, rich.tolist())  # user-frequent
        self.assertIn(3, rich.tolist())
        self.assertEqual(pick_popular_radius_km(stats, 10.0, "multiplier", 2.5), 25.0)
        self.assertAlmostEqual(pick_popular_radius_km(stats, 10.0, "max", 2.0), stats["max_km"])
        visits = catalog_visit_counts(groups, pois)
        self.assertEqual(visits, [1, 2, 1, 1])
        uf = build_user_frequent(groups, pois, top_k=2, min_visits=1)
        self.assertEqual(uf["u"][0], 1)  # POI 2 visited twice
        pop = popular_indices(0.0, 0.0, coords, 200.0, popularity, top_k=1)
        self.assertEqual(pop.tolist(), [1])
        # Pool-size stats use the resolved set (after target include / fallback).
        pools = [
            resolve_candidate_pool(0.0, 0.0, 3, coords, 1.0, 4)[0],
            resolve_candidate_pool(float("nan"), 0.0, 1, coords, 10.0, 4)[0],
            resolve_candidate_pool(10.0, 10.0, 0, coords, 0.01, 4)[0],
        ]
        resolutions = ["multi_source", "fallback_full_catalog", "fallback_full_catalog"]
        stats = summarize_pool_sizes(pools, resolutions)
        self.assertEqual(stats["n"], 3)
        self.assertEqual(stats["min"], 2)  # origin + force-included target
        self.assertEqual(stats["max"], 4)
        self.assertAlmostEqual(stats["mean"], (2 + 4 + 4) / 3)
        self.assertEqual(stats["median"], 4.0)
        self.assertEqual(stats["resolutions"]["multi_source"], 1)
        self.assertEqual(stats["resolutions"]["fallback_full_catalog"], 2)
        line = format_pool_size_line("train", stats)
        self.assertTrue(line.startswith("candidates train: n=3 min=2 max=4 mean="))
        self.assertIn("median=4.00", line)
        self.assertIn("multi_source=1", line)
        rows, labels, mask = sample_candidates(
            torch.tensor([0]), 4, 1, random.Random(0), pools=[pool])
        self.assertTrue(mask[0, 0].item())
        self.assertEqual(int(rows[0, 0]), 0)
        self.assertEqual(int(labels[0]), 0)
        self.assertEqual(int(mask[0].sum()), 2)
        capped, _, cap_mask = sample_candidates(
            torch.tensor([0]), 4, -1, random.Random(0), pools=[rich], max_candidates=2)
        self.assertEqual(int(cap_mask[0].sum()), 2)
        self.assertEqual(int(capped[0, 0]), 0)

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
            self.assertEqual(results_root_dir(cwd / "runs" / "tokyo" / "model", cwd=cwd),
                             cwd / "results")
            self.assertEqual(run_name_for_results(cwd / "runs" / "tokyo" / "model", cwd=cwd),
                             "tokyo/model")
            outside = Path(temp) / "nested" / "run"
            outside.mkdir(parents=True)
            # Force a cwd that does not contain `outside` so the sibling layout applies.
            other = cwd / "other"
            other.mkdir()
            self.assertEqual(results_run_dir(outside, cwd=other),
                             outside.parent / "results" / "run")

    def test_append_ablation_log_creates_md_and_csv(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            out = root / "runs" / "tokyo" / "abl_router_hard"
            out.mkdir(parents=True)
            row = dict(
                date="2026-11-19",
                run_name="tokyo/abl_router_hard",
                cache=str(root / "runs" / "tokyo" / "cache"),
                key_flags="--router-mode hard_prior --causal-counts --transition-attn bias",
                **{"Acc@1": "0.123456", "NDCG@1": "0.111", "MRR": "0.2"},
                notes="best_val Acc@1=0.1; selection=best_checkpoint_by_NDCG@10",
            )
            md_path, csv_path = append_ablation_log(row, out, cwd=root)
            self.assertEqual(md_path, root / "results" / "ablation_log.md")
            self.assertEqual(csv_path, root / "results" / "ablation_log.csv")
            md = md_path.read_text()
            self.assertIn("| date | run_name | cache | key flags | Acc@1 | NDCG@1 | MRR | notes |", md)
            self.assertIn("tokyo/abl_router_hard", md)
            self.assertIn("0.123456", md)
            csv_text = csv_path.read_text()
            self.assertTrue(csv_text.startswith(
                "date,run_name,cache,key_flags,Acc@1,NDCG@1,MRR,notes\n"))
            self.assertIn("abl_router_hard", csv_text)
            append_ablation_log({**row, "Acc@1": "0.2"}, out, cwd=root)
            self.assertEqual(csv_path.read_text().count("\n"), 3)  # header + 2 rows

    def test_ablation_row_metric_cells(self):
        class FakeCache:
            candidate_construction = {"method": "multi_source", "radius_statistic": "p90",
                                      "popular_top_k": 100, "popular_radius": "max",
                                      "user_frequent_top_k": 50, "random_negatives": 50}
            cfg = {"train_targets": "all", "eval_targets": "all", "candidate_radius": "p90"}

        flags = format_ablation_key_flags(
            FakeCache(),
            {"router_mode": "hard_prior", "causal_counts": True, "transition_attn": "bias",
             "transition_lambda": 1.0, "visit_bins": 10, "negatives": 63,
             "max_train_candidates": -1},
            {"router_mode": "hard_prior", "causal_counts": True, "use_surd": True},
        )
        self.assertIn("--candidate-radius p90", flags)
        self.assertIn("--router-mode hard_prior", flags)
        self.assertIn("--causal-counts", flags)
        self.assertIn("--transition-attn bias", flags)
        row = ablation_row_from_eval(
            FakeCache(), {"Acc@1": 1 / 3, "NDCG@1": 0.5, "MRR": 0.25},
            run_name="demo/run", cache_path="runs/demo/cache", key_flags=flags,
            notes="unit", date="2026-01-02")
        self.assertEqual(row["date"], "2026-01-02")
        self.assertEqual(row["Acc@1"], "0.333333")
        self.assertEqual(row["NDCG@1"], "0.5")
        self.assertEqual(row["MRR"], "0.25")

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
            self.assertEqual(cache.candidate_construction["method"], "multi_source")
            self.assertEqual(cache.cfg["popular_radius"], "max")
            self.assertEqual(cache.cfg["popular_top_k"], 100)
            self.assertEqual(cache.cfg["popular_by"], "visit")
            self.assertEqual(cache.cfg["user_frequent_top_k"], 50)
            self.assertEqual(cache.cfg["random_negatives"], 50)
            self.assertEqual(cache.cfg["random_negatives_seed"], 0)
            self.assertEqual(cache.random_negatives, 50)
            self.assertTrue(cache.candidate_construction.get("ground_truth_always_in_pool"))
            self.assertIsNotNone(cache.popular_radius_km)
            self.assertTrue((cache.path / "user_frequent.pt").is_file())
            self.assertEqual(len(cache.meta["poi_coordinates"]), len(cache.pois))
            self.assertTrue(all(
                math.isfinite(v) for pair in cache.meta["poi_coordinates"] for v in pair))
            self.assertIn("coordinate_imputation", cache.candidate_construction)
            self.assertEqual(len(cache.candidate_construction["poi_visit_counts"]), len(cache.pois))
            self.assertIn("lat", train)
            self.assertIn("lon", train)
            pools, resolutions = cache.candidate_pools(train)
            self.assertEqual(len(pools), len(train["end"]))
            self.assertTrue(any(r in ("multi_source", "fallback_full_catalog") for r in resolutions))
            # In-catalog GT must be present in every supervised pool (train/val/test).
            for split_name in ("train", "validation", "test"):
                split = cache.split(split_name)
                split_pools, _ = cache.candidate_pools(split)
                n_events = len(split["poi_token"])
                for i, pool in enumerate(split_pools):
                    end = int(split["end"][i])
                    if end >= n_events:
                        continue
                    target_row = int(split["poi_token"][end]) - 2
                    if target_row < 0:
                        continue
                    self.assertTrue(
                        (pool == target_row).any(),
                        msg=f"{split_name} sample {i} missing GT row {target_row}")
            run_cli("inspect", "--cache", cache_path, "--poi-id", cache.pois[0], "--sample-index", 0,
                    "--output", root / "trace.json")
            trace = json.loads((root / "trace.json").read_text())
            self.assertEqual(len(trace["sample"]["q_state"]), 4)
            self.assertEqual(len(trace["specific_bits"]), 15)
            self.assertEqual(len(trace["raw_surd"]), 26)
            # One shared train-all SURD table; no fold_*.surd.pt artifacts.
            self.assertEqual([t["name"] for t in cache.meta["tables"]], ["final"])
            self.assertFalse(any((cache.path).glob("fold_*.surd.pt")))
            self.assertNotIn("fold", train)
            self.assertNotIn("folds", cache.cfg)
            self.assertNotIn("warmup", cache.cfg)
            final_meta = cache.meta["tables"][0]
            self.assertEqual(final_meta["reference_events"], len(train["end"]))
            self.assertIsNone(final_meta["prediction_cutoff_utc_us"])
            self.assertEqual(final_meta["comparison"], "all official training reference events")
            expected = (train["poi_token"][train["end"]] - 2).bincount(minlength=len(cache.pois))
            table = cache.table("final", "cpu")
            torch.testing.assert_close(table.target_count, expected.float())
            # Same POI row → identical prior/confidence whether used for train or val/test.
            rows = torch.tensor([[0, 1, -1]], dtype=torch.long)
            prior_a, conf_a = table(rows)
            prior_b, conf_b = cache.table("final", "cpu")(rows)
            torch.testing.assert_close(prior_a, prior_b)
            torch.testing.assert_close(conf_a, conf_b)
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
            ckpt = torch.load(run / "best.pt", map_location="cpu", weights_only=True)
            self.assertEqual(ckpt["config"]["router_mode"], "hard_prior")
            self.assertEqual(ckpt["config"]["hybrid_lambda"], 0.1)
            self.assertIn("router_mode", ckpt["training_args"])
            self.assertEqual(ckpt["training_args"]["router_mode"], "hard_prior")
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
            self.assertIn("test", history["epochs"][0])
            self.assertTrue(history["training_args"].get("eval_test_every_epoch", True))
            # Per-epoch test + compact results history (default --eval-test-every-epoch).
            epoch_json = results / "eval_test" / "epoch_001.json"
            self.assertTrue(epoch_json.is_file())
            epoch_payload = json.loads(epoch_json.read_text())
            self.assertEqual(epoch_payload["epoch"], 1)
            self.assertEqual(epoch_payload["selection"], "current_epoch_weights")
            self.assertIn("Acc@1", epoch_payload["all_targets"])
            results_history = json.loads((results / "history.json").read_text())
            self.assertTrue(results_history["eval_test_every_epoch"])
            self.assertIn("test", results_history["epochs"][0])
            self.assertIn("Acc@1", results_history["epochs"][0]["test"])
            # Auto ablation log under the shared results/ root.
            abl_md = results.parent / "ablation_log.md"
            abl_csv = results.parent / "ablation_log.csv"
            self.assertTrue(abl_md.is_file())
            self.assertTrue(abl_csv.is_file())
            self.assertIn("Acc@1", abl_md.read_text())
            self.assertIn("key_flags", abl_csv.read_text().splitlines()[0])
            train_test = json.loads((results / "eval_test" / "metrics.json").read_text())
            self.assertEqual(train_test["split"], "test")
            self.assertEqual(train_test["selection"], "best_checkpoint_by_NDCG@10")
            self.assertTrue((results / "eval_test" / "report.txt").is_file())
            self.assertTrue((results / "eval_validation" / "metrics.json").is_file())
            self.assertIn("Acc@1", train_test["all_targets"])
            self.assertIn("Acc@10", train_test["all_targets"])
            self.assertEqual(train_test["primary_metrics"], ["Acc@1", "NDCG@1", "MRR"])
            self.assertEqual(train_test["candidate_construction"]["method"], "multi_source")
            self.assertEqual(train_test["candidate_construction"]["radius_statistic"], "p90")
            self.assertEqual(train_test["candidate_construction"]["popular_radius"], "max")
            self.assertTrue(train_test["candidate_construction"]["eval_uses_full_pool"])
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

    def test_filter_unknown_sthgcn_style_prepare(self):
        """``--filter-unknown on`` drops cold-start users and OOV targets from val/test data.

        Unlike eval ``known_only`` (metric denominator only), samples are removed at prepare.
        Default ``off`` keeps status-quo unknown_targets > 0 on the demo OOV POI.
        """
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            directory = generate(root / "data")
            # Add a validation trail owned by a user who never appears in train.
            val_path = directory / "demo_checkins_validation.csv"
            with val_path.open(newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
                fields = list(rows[0].keys())
            cold = dict(rows[0])
            cold.update(
                trail_id="validation_cold_user",
                user_id="9999",
                venue_id="1",
                timestamp="2018-06-01 10:00:00",
            )
            cold2 = dict(cold)
            cold2.update(timestamp="2018-06-01 10:15:00", venue_id="2")
            cold3 = dict(cold)
            cold3.update(timestamp="2018-06-01 10:30:00", venue_id="3")
            with val_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows + [cold, cold2, cold3])

            baseline = root / "cache_off"
            run_cli("prepare", "--data-root", directory, "--city", "demo", "--out", baseline)
            base_meta = json.loads((baseline / "manifest.json").read_text())
            self.assertEqual(base_meta["config"].get("filter_unknown", "off"), "off")
            self.assertGreater(base_meta["audit"]["validation"]["unknown_targets"], 0)
            self.assertGreater(base_meta["audit"]["validation"]["samples"], 0)
            off_val_n = base_meta["audit"]["validation"]["samples"]

            filtered = root / "cache_on"
            run_cli("prepare", "--data-root", directory, "--city", "demo", "--out", filtered,
                    "--filter-unknown", "on")
            on_meta = json.loads((filtered / "manifest.json").read_text())
            self.assertEqual(on_meta["config"]["filter_unknown"], "on")
            self.assertNotEqual(on_meta["fingerprint"], base_meta["fingerprint"])
            val_audit = on_meta["audit"]["validation"]
            self.assertEqual(val_audit["filter_unknown"], "on")
            self.assertGreaterEqual(val_audit["trails_dropped_unknown_user"], 1)
            self.assertGreaterEqual(val_audit["samples_skipped_unknown_poi_target"], 1)
            self.assertEqual(val_audit["unknown_targets"], 0)
            self.assertLess(val_audit["samples"], off_val_n)
            # All remaining val targets are in-catalog.
            val = Cache(filtered).split("validation")
            rows = val["poi_token"][val["end"]] - 2
            self.assertTrue((rows >= 0).all())

            # key_flags / ablation-summary expectation for abl_filter_unknown.
            flags = format_ablation_key_flags(Cache(filtered))
            self.assertIn("--filter-unknown on", flags)
            status, _ = validate_run_name_settings(
                "tokyo/abl_filter_unknown",
                extract_recorded_settings(
                    {"filter_unknown": "on", "cli": {"filter_unknown": "on"}},
                    key_flags=flags,
                ),
            )
            self.assertEqual(status, "ok")

    def test_prepare_overwrites_on_fingerprint_change(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            directory = generate(root / "data")
            cache_path = root / "cache"
            run_cli("prepare", "--data-root", directory, "--city", "demo", "--out", cache_path)
            first = json.loads((cache_path / "manifest.json").read_text())
            self.assertEqual(first["config"]["candidate_radius"], "p90")
            marker = cache_path / "stale_marker.txt"
            marker.write_text("old\n")

            # Same fingerprint → reuse (marker left alone).
            run_cli("prepare", "--data-root", directory, "--city", "demo", "--out", cache_path)
            self.assertTrue(marker.is_file())
            self.assertEqual(
                json.loads((cache_path / "manifest.json").read_text())["fingerprint"],
                first["fingerprint"])

            # Different fingerprint → rebuild in place (default overwrite).
            run_cli("prepare", "--data-root", directory, "--city", "demo", "--out", cache_path,
                    "--candidate-radius", "p80")
            second = json.loads((cache_path / "manifest.json").read_text())
            self.assertEqual(second["config"]["candidate_radius"], "p80")
            self.assertNotEqual(second["fingerprint"], first["fingerprint"])
            self.assertFalse(marker.exists())
            Cache(cache_path)  # verifies checksums / loads

            # --no-overwrite refuses a further fingerprint change.
            with self.assertRaises(ValueError) as ctx:
                run_cli("prepare", "--data-root", directory, "--city", "demo",
                        "--out", cache_path, "--candidate-radius", "p70",
                        "--no-overwrite")
            self.assertIn("different cache", str(ctx.exception))
            self.assertEqual(
                json.loads((cache_path / "manifest.json").read_text())["config"]["candidate_radius"],
                "p80")

            # Incomplete cache (checksum mismatch) rebuilds in place by default.
            train_pt = cache_path / "train.pt"
            train_pt.write_bytes(train_pt.read_bytes() + b"\x00")
            run_cli("prepare", "--data-root", directory, "--city", "demo", "--out", cache_path,
                    "--candidate-radius", "p80")
            Cache(cache_path)
            self.assertEqual(
                json.loads((cache_path / "manifest.json").read_text())["config"]["candidate_radius"],
                "p80")

    def test_shared_surd_table_one_artifact_same_poi_lookup(self):
        """Prepare writes only final; train-path lookup matches val/test table for same POI."""
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            directory = generate(root / "data")
            cache_path = root / "cache"
            run_cli("prepare", "--data-root", directory, "--city", "demo", "--out", cache_path)
            cache = Cache(cache_path)
            self.assertEqual([t["name"] for t in cache.meta["tables"]], ["final"])
            self.assertEqual(sorted(p.name for p in cache_path.glob("*.surd.pt")), ["final.surd.pt"])
            self.assertEqual(
                sorted(p.name for p in cache_path.glob("*.quantizer.pt")), ["final.quantizer.pt"])
            train = cache.split("train")
            self.assertNotIn("fold", train)
            # Engine train() loads the same Cache.table("final") used by evaluate.
            shared = cache.table("final", "cpu")
            n = min(4, len(cache.pois))
            rows = torch.arange(n, dtype=torch.long).unsqueeze(0)
            prior, confidence = shared(rows)
            again_prior, again_confidence = cache.table("final", "cpu")(rows)
            torch.testing.assert_close(prior, again_prior)
            torch.testing.assert_close(confidence, again_confidence)
            # Per-POI routing_prior rows are identity lookups into the shared buffer.
            for j in range(n):
                torch.testing.assert_close(prior[0, j], shared.routing_prior[j])
                self.assertEqual(float(confidence[0, j]), float(shared.confidence[j]))


class TestCausalCounts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def _tiny_split(self):
        """Two users, overlapping timestamps so order is nontrivial."""
        from datetime import datetime, timedelta, timezone
        t0 = datetime(2018, 1, 1, tzinfo=timezone.utc)

        def ev(poi, cat, hour, line):
            return Event(poi, f"c{cat}", cat, t0 + timedelta(hours=hour), (0.0, 0.0), line)

        groups = {
            ("u1", "t1"): [ev(1, 0, 0, 1), ev(2, 1, 1, 2), ev(3, 0, 4, 3)],
            ("u2", "t2"): [ev(2, 1, 2, 4), ev(1, 0, 3, 5), ev(4, 1, 5, 6)],
        }
        cats = Categories(groups)
        pois = [1, 2, 3, 4]
        cat_vocab = sorted({cats(e) for events in groups.values() for e in events})
        data, _ = build_split(groups, cats, pois, cat_vocab, 4, "all")
        return data, len(pois), len(cat_vocab)

    def test_chronological_order_no_shuffle(self):
        data, _, _ = self._tiny_split()
        order = chronological_indices(data)
        stamps = data["us"][data["end"]][order]
        self.assertTrue(torch.all(stamps[1:] >= stamps[:-1]))
        # Must not match a random permutation in general; at least sorted.
        self.assertEqual(order.tolist(), torch.argsort(data["us"][data["end"]], stable=True).tolist())

    def test_causal_update_score_then_write(self):
        data, n_pois, n_cats = self._tiny_split()
        tables = CausalCountTables(n_pois, n_cats, visit_bins=4)
        order = chronological_indices(data)
        first = int(order[0])
        user, prev_p, prev_c, next_p, next_c = sample_hop(data, first)
        # Before any update, counts are zero.
        self.assertEqual(tables.poi_poi_count(prev_p, next_p), 0.0)
        self.assertEqual(int(tables.user_visits[user]), 0)
        feat_before = tables.poi_feat(user, next_p)
        tables.update_sample(data, first)
        self.assertEqual(tables.poi_poi_count(prev_p, next_p), 1.0)
        self.assertEqual(int(tables.user_visits[user]), 1)
        self.assertGreater(tables.poi_feat(user, next_p)[0], feat_before[0])
        # Replay of remaining samples is strictly increasing global mass.
        before = float(tables.global_poi.sum())
        for i in order[1:].tolist():
            tables.update_sample(data, i)
        self.assertGreater(float(tables.global_poi.sum()), before)

    def test_equal_frequency_binning(self):
        values = list(range(20))
        edges = equal_frequency_edges(values, 5)
        self.assertTrue(torch.all(edges[1:] >= edges[:-1]))
        bins = [visit_bin_id(v, edges) for v in values]
        self.assertEqual(min(bins), 0)
        self.assertLessEqual(max(bins), 5)
        data, n_pois, n_cats = self._tiny_split()
        tables = CausalCountTables(n_pois, n_cats, visit_bins=DEFAULT_VISIT_BINS)
        edges = tables.fit_visit_bins(data)
        self.assertEqual(tables.visit_bins, DEFAULT_VISIT_BINS)
        # After fit, tables are reset but edges frozen.
        self.assertEqual(int(tables.global_poi.sum()), 0)
        self.assertIsNotNone(tables.visit_bin_edges)
        clone = tables.clone()
        tables.replay(data)
        # Val/test start from train-end clone, not the live mutating table mid-stream.
        frozen = tables.clone()
        self.assertEqual(float(frozen.global_poi.sum()), float(tables.global_poi.sum()))
        self.assertEqual(clone.visit_bin_edges.tolist(), tables.visit_bin_edges.tolist())

    def test_transition_attn_modes_shapes(self):
        torch.manual_seed(0)
        b, c, length, d, heads = 2, 3, 4, 8, 2
        query = torch.randn(b, c, d)
        key = torch.randn(b, length, d)
        valid = torch.tensor([[True, True, True, False], [True, True, False, False]])
        counts = torch.randint(0, 5, (b, c, length)).float()
        for mode in ("bias", "gate", "rel_key"):
            layer = TransitionCrossAttention(d, heads, 0.0, mode, transition_lambda=1.0)
            out = layer(query, key, key, ~valid, counts)
            self.assertEqual(tuple(out.shape), (b, c, d), msg=mode)
            self.assertTrue(torch.isfinite(out).all(), msg=mode)

    def test_model_forward_with_count_ctx_all_modes(self):
        torch.manual_seed(2)
        data, n_pois, n_cats = self._tiny_split()
        tables = CausalCountTables(n_pois, n_cats, visit_bins=4)
        tables.fit_visit_bins(data, 4)
        idx = chronological_indices(data)[:2]
        h, valid, target = make_batch(data, idx)
        cand = torch.arange(n_pois).expand(len(idx), -1)
        cand_lists = [list(range(n_pois)) for _ in idx.tolist()]
        ctx = gather_batch_context(tables, data, idx, cand_lists, int(valid.shape[1]), update=True)
        prior = torch.zeros(len(idx), n_pois, 26)
        prior[..., 0] = 1.0
        conf = torch.ones(len(idx), n_pois)
        for mode in ("bias", "gate", "rel_key"):
            cfg = ModelConfig(8, 2, 1, 0., 2, router_mode="hard_prior",
                              causal_counts=True, transition_attn=mode, visit_bins=4)
            model = TSMoE(n_pois, n_cats, 4, cfg).eval()
            with torch.no_grad():
                out = model(h, valid, cand, prior, conf, count_ctx=ctx)
            self.assertEqual(tuple(out["score"].shape), (len(idx), n_pois), msg=mode)
            self.assertTrue(torch.isfinite(out["score"]).all(), msg=mode)
            # Count feature dims feed the MLPs.
            self.assertEqual(model.poi_count_mlp[0].in_features, COUNT_FEAT_DIM)
            self.assertEqual(model.visit_bin_emb.num_embeddings, 4)

    def test_transition_score_bias_raises_last_to_cand_scores(self):
        """λ·log1p(last→cand transition) must lift scores for high-count candidates."""
        torch.manual_seed(3)
        data, n_pois, n_cats = self._tiny_split()
        tables = CausalCountTables(n_pois, n_cats, visit_bins=4)
        tables.fit_visit_bins(data, 4)
        idx = chronological_indices(data)[:1]
        h, valid, _ = make_batch(data, idx)
        cand = torch.arange(n_pois).expand(1, -1)
        cand_lists = [list(range(n_pois))]
        ctx = gather_batch_context(
            tables, data, idx, cand_lists, int(valid.shape[1]), update=False)
        # Force a large last→cand[0] transition so the bias is detectable.
        last = int(valid[0].long().sum().item()) - 1
        ctx = dict(ctx)
        tr = ctx["poi_transition"].clone()
        tr[0, 0, last] = 20.0
        tr[0, 1, last] = 0.0
        ctx["poi_transition"] = tr
        prior = torch.zeros(1, n_pois, 26)
        prior[..., 0] = 1.0
        conf = torch.ones(1, n_pois)
        base_cfg = dict(dim=8, heads=2, layers=1, dropout=0., top_k=2,
                        router_mode="hard_prior", causal_counts=True, visit_bins=4)
        off = TSMoE(n_pois, n_cats, 4, ModelConfig(**base_cfg, transition_score_bias=0.)).eval()
        on = TSMoE(n_pois, n_cats, 4, ModelConfig(**base_cfg, transition_score_bias=2.)).eval()
        on.load_state_dict(off.state_dict())
        with torch.no_grad():
            s0 = off(h, valid, cand, prior, conf, count_ctx=ctx)["score"][0]
            s1 = on(h, valid, cand, prior, conf, count_ctx=ctx)["score"][0]
        # Candidate 0 should gain ≈ 2·log1p(20); candidate 1 unchanged (count 0).
        self.assertGreater(float(s1[0] - s0[0]), 1.0)
        torch.testing.assert_close(s1[1], s0[1], atol=1e-5, rtol=1e-5)
        legacy = model_config_from_checkpoint(dict(
            dim=8, heads=2, layers=1, dropout=0., top_k=4, temperature=1.,
            lambda_prior=.1, use_surd=True, causal_counts=True))
        self.assertEqual(legacy.transition_score_bias, 0.0)
        self.assertEqual(legacy.category_score_bias, 0.0)
        self.assertEqual(legacy.pop_score_bias, 0.0)
        self.assertEqual(legacy.geo_score_bias, 0.0)
        self.assertEqual(legacy.geo_score_power, 1.0)
        self.assertEqual(legacy.surd_bias_gate, "off")
        self.assertEqual(legacy.surd_score_bias, 0.0)
        self.assertEqual(legacy.tsb_gate_mode, "off")
        self.assertEqual(legacy.surd_attn_gate, "off")

    def test_category_pop_geo_score_biases(self):
        """New TM-35 score priors: cat→poi, popularity, and geo proximity."""
        torch.manual_seed(5)
        data, n_pois, n_cats = self._tiny_split()
        tables = CausalCountTables(n_pois, n_cats, visit_bins=4)
        tables.fit_visit_bins(data, 4)
        idx = chronological_indices(data)[:1]
        h, valid, _ = make_batch(data, idx)
        cand = torch.arange(n_pois).expand(1, -1)
        ctx = gather_batch_context(
            tables, data, idx, [list(range(n_pois))], int(valid.shape[1]), update=False)
        ctx = dict(ctx)
        last = int(valid[0].long().sum().item()) - 1
        cat_tr = ctx["cat_transition"].clone()
        cat_tr[0, 0, last] = 20.0
        cat_tr[0, 1, last] = 0.0
        ctx["cat_transition"] = cat_tr
        feats = ctx["cand_poi_feat"].clone()
        feats[0, 0, 1] = 3.0  # log1p(global) already-scaled slot
        feats[0, 1, 1] = 0.0
        ctx["cand_poi_feat"] = feats
        m = cand.shape[1]
        delta = torch.zeros(1, m, 3)
        delta[0, 0, 0] = 0.0   # near
        delta[0, 1, 0] = 2.0   # far (log1p km)
        ctx["delta_geo"] = delta
        prior = torch.zeros(1, n_pois, 26)
        prior[..., 0] = 1.0
        conf = torch.ones(1, n_pois)
        base = dict(dim=8, heads=2, layers=1, dropout=0., top_k=2,
                    router_mode="hard_prior", causal_counts=True, visit_bins=4,
                    candidate_context="on")
        off = TSMoE(n_pois, n_cats, 4, ModelConfig(**base)).eval()
        csb = TSMoE(n_pois, n_cats, 4, ModelConfig(**base, category_score_bias=2.)).eval()
        psb = TSMoE(n_pois, n_cats, 4, ModelConfig(**base, pop_score_bias=1.)).eval()
        gsb = TSMoE(n_pois, n_cats, 4, ModelConfig(**base, geo_score_bias=1.)).eval()
        csb.load_state_dict(off.state_dict())
        psb.load_state_dict(off.state_dict())
        gsb.load_state_dict(off.state_dict())
        with torch.no_grad():
            s0 = off(h, valid, cand, prior, conf, count_ctx=ctx)["score"][0]
            s_csb = csb(h, valid, cand, prior, conf, count_ctx=ctx)["score"][0]
            s_psb = psb(h, valid, cand, prior, conf, count_ctx=ctx)["score"][0]
            s_gsb = gsb(h, valid, cand, prior, conf, count_ctx=ctx)["score"][0]
        self.assertGreater(float(s_csb[0] - s0[0]), 1.0)
        torch.testing.assert_close(s_csb[1], s0[1], atol=1e-5, rtol=1e-5)
        self.assertGreater(float(s_psb[0] - s0[0]), 2.0)
        self.assertGreater(float(s0[1] - s_gsb[1]), 1.0)  # far cand penalized
        # Gravity α>1 penalizes far candidates more than α=1 for same λ.
        gsb_p = TSMoE(n_pois, n_cats, 4, ModelConfig(
            **base, geo_score_bias=1., geo_score_power=2.)).eval()
        gsb_p.load_state_dict(off.state_dict())
        with torch.no_grad():
            s_p2 = gsb_p(h, valid, cand, prior, conf, count_ctx=ctx)["score"][0]
        self.assertGreater(float(s_gsb[1] - s_p2[1]), 0.5)
        with self.assertRaises(ValueError):
            ModelConfig(8, 2, 1, 0., 2, geo_score_bias=1.0)  # needs candidate_context
        with self.assertRaises(ValueError):
            ModelConfig(8, 2, 1, 0., 2, candidate_context="on",
                        geo_score_bias=1.0, geo_score_power=0.0)

    def test_surd_bias_gate_helpers_and_score_bias(self):
        """SURD/SMI gates: helpers + additive surd_score_bias lifts high-gate cands."""
        prior = torch.zeros(1, 3, 26)
        prior[0, 0, 0] = 0.8  # Unique_P high on cand 0
        prior[0, 1, 0] = 0.1
        prior[0, 2, 1] = 0.9  # Unique_C only on cand 2
        conf = torch.tensor([[1.0, 1.0, 0.5]])
        g = compute_surd_bias_gate("unique_p", prior, conf)
        self.assertIsNone(compute_surd_bias_gate("off", prior, conf))
        self.assertAlmostEqual(float(g[0, 0]), 0.8, places=5)
        self.assertAlmostEqual(float(g[0, 1]), 0.1, places=5)
        self.assertAlmostEqual(float(g[0, 2]), 0.0, places=5)
        smi = torch.zeros(1, 3, 15)
        smi[0, 0, 0] = 4.0
        smi[0, 0, 1:] = 1.0  # total 4+14=18 → frac P = 4/18
        smi[0, 1, 0] = 0.0
        smi[0, 1, 1:] = 1.0
        g_smi = compute_surd_bias_gate("smi_p", prior, conf, smi)
        self.assertGreater(float(g_smi[0, 0]), float(g_smi[0, 1]))
        term = torch.tensor([[2.0, 2.0, 2.0]])
        torch.testing.assert_close(apply_surd_gate(term, g, "off"), term)
        torch.testing.assert_close(
            apply_surd_gate(term, g, "mul")[0, 0], torch.tensor(1.6))
        torch.testing.assert_close(
            apply_surd_gate(term, g, "residual")[0, 0], torch.tensor(3.6))
        with self.assertRaises(ValueError):
            ModelConfig(8, 2, 1, 0., 2, surd_score_bias=1.0)  # needs gate
        # Score path: surd_score_bias alone (no tsb) raises cand0 over cand1.
        torch.manual_seed(0)
        data, n_pois, n_cats = self._tiny_split()
        h, valid, _ = make_batch(data, chronological_indices(data)[:1])
        cand = torch.arange(min(3, n_pois)).expand(1, -1)
        n_c = cand.shape[1]
        prior_b = torch.zeros(1, n_c, 26)
        prior_b[0, 0, 0] = 0.9
        if n_c > 1:
            prior_b[0, 1, 0] = 0.05
        conf_b = torch.ones(1, n_c)
        cfg = ModelConfig(
            8, 2, 1, 0., 2, router_mode="hard_prior", causal_counts=False,
            surd_bias_gate="unique_p", surd_score_bias=3.0)
        model = TSMoE(n_pois, n_cats, 4, cfg).eval()
        with torch.no_grad():
            s = model(h, valid, cand, prior_b, conf_b)["score"][0]
        if n_c > 1:
            self.assertGreater(float(s[0] - s[1]), 1.0)

    def test_objective_bpr_and_margin(self):
        scores = torch.tensor([[3.0, 2.5, 0.0]])
        mask = torch.tensor([[True, True, True]])
        conf = torch.ones(1, 3)
        result = dict(
            score=scores, mask=mask, confidence=conf,
            log_pi=torch.zeros(1, 3, 15), mass=torch.zeros(1, 3, 15),
            gate=torch.ones(1, 3), gate_logits=torch.zeros(1, 3),
            alpha=torch.zeros(1, 3))
        # CE-only: bpr term is 0.
        _, (_, _, _, bpr0) = objective(result, torch.tensor([0]), lambda_bpr=0.)
        self.assertEqual(float(bpr0), 0.0)
        # Hard-neg BPR against max other score (2.5); margin increases the loss.
        _, (_, _, _, bpr) = objective(result, torch.tensor([0]), lambda_bpr=1.)
        _, (_, _, _, bpr_m) = objective(
            result, torch.tensor([0]), lambda_bpr=1., bpr_margin=1.0)
        self.assertGreater(float(bpr), 0.0)
        self.assertGreater(float(bpr_m), float(bpr))

    def test_legacy_checkpoint_disables_causal_counts(self):
        legacy = model_config_from_checkpoint(dict(
            dim=8, heads=2, layers=1, dropout=0., top_k=4, temperature=1.,
            lambda_prior=.1, use_surd=True))
        self.assertFalse(legacy.causal_counts)
        self.assertEqual(legacy.transition_attn, "bias")
        self.assertEqual(ModelConfig().causal_counts, True)
        self.assertEqual(ModelConfig().transition_attn, "bias")
        self.assertEqual(ModelConfig().visit_bins, DEFAULT_VISIT_BINS)
        self.assertEqual(ModelConfig().transition_score_bias, 0.0)

    def test_cli_causal_train_smoke(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            directory = generate(root / "data")
            cache_path, run = root / "cache", root / "run_causal"
            run_cli("prepare", "--data-root", directory, "--city", "demo", "--out", cache_path)
            run_cli("train", "--cache", cache_path, "--out", run, "--epochs", 1,
                    "--dim", 8, "--layers", 1, "--heads", 2, "--device", "cpu",
                    "--negatives", 3, "--threads", 1, "--transition-attn", "bias",
                    "--visit-bins", 4)
            ckpt = torch.load(run / "best.pt", map_location="cpu", weights_only=False)
            self.assertTrue(ckpt["config"]["causal_counts"])
            self.assertEqual(ckpt["config"]["transition_attn"], "bias")
            self.assertEqual(ckpt["config"]["visit_bins"], 4)
            self.assertIn("count_tables", ckpt)
            self.assertIn("visit_bin_edges", ckpt)
            # evaluate rebuilds / loads train-end tables
            run_cli("evaluate", "--cache", cache_path, "--checkpoint", run / "best.pt",
                    "--device", "cpu", "--split", "test", "--chunk", 3)


class TestCandidateContextOverlays(unittest.TestCase):
    """A/B/C/D/E overlay flags: defaults off ≡ baseline; on-path smoke."""

    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_overlay_defaults_off(self):
        cfg = ModelConfig()
        self.assertEqual(cfg.candidate_context, "off")
        self.assertEqual(cfg.candidate_context_time, "off")
        self.assertEqual(cfg.session_summary, "off")
        self.assertEqual(cfg.user_pref, "off")
        self.assertEqual(cfg.transition_pmi, "off")
        self.assertEqual(cfg.expert_cross_attn, "off")
        self.assertEqual(cfg.expert_cross_attn_sparse, "on")
        legacy = model_config_from_checkpoint(dict(
            dim=8, heads=2, layers=1, dropout=0., top_k=4, temperature=1.,
            lambda_prior=.1, use_surd=True, causal_counts=True))
        self.assertEqual(legacy.candidate_context, "off")
        self.assertEqual(legacy.expert_cross_attn, "off")

    def test_flags_off_match_baseline_scores(self):
        """Explicit off overlays must match a default ModelConfig forward (same seed)."""
        torch.manual_seed(0)
        h = {"P": torch.tensor([[2, 3, 0]]), "C": torch.tensor([[2, 4, 0]]),
             "T": torch.randn(1, 3, T_DIM), "L": torch.randn(1, 3, L_DIM)}
        valid = torch.tensor([[True, True, False]])
        rows = torch.tensor([[0, 1, 2]])
        prior = torch.zeros(1, 3, 26)
        prior[..., 0] = 1.0
        conf = torch.ones(1, 3)
        base_cfg = ModelConfig(8, 2, 1, 0., 2, router_mode="hard_prior", causal_counts=False)
        off_cfg = ModelConfig(
            8, 2, 1, 0., 2, router_mode="hard_prior", causal_counts=False,
            candidate_context="off", candidate_context_time="off",
            session_summary="off", user_pref="off", transition_pmi="off",
            expert_cross_attn="off")
        torch.manual_seed(1)
        a = TSMoE(5, 3, 3, base_cfg).eval()
        torch.manual_seed(1)
        b = TSMoE(5, 3, 3, off_cfg).eval()
        with torch.no_grad():
            sa = a(h, valid, rows, prior, conf)["score"]
            sb = b(h, valid, rows, prior, conf)["score"]
        torch.testing.assert_close(sa, sb)
        # hard_prior logits still SURD-only with overlays off.
        with torch.no_grad():
            det = a(h, valid, rows, prior, conf, details=True)
        torch.testing.assert_close(det["logits"], torch.log1p(15 * det["mass"]))

    def test_candidate_context_on_smoke_and_hard_prior_invariant(self):
        torch.manual_seed(3)
        data, n_pois, n_cats = TestCausalCounts()._tiny_split()
        poi_cat = build_poi_cat_table(data, n_pois)
        coords = torch.tensor([[35.0 + i * 0.01, 139.0] for i in range(n_pois)],
                              dtype=torch.float64)
        idx = chronological_indices(data)[:2]
        h, valid, _ = make_batch(data, idx)
        cand = torch.arange(n_pois).expand(len(idx), -1)
        cand_lists = [list(range(n_pois)) for _ in idx.tolist()]
        side = gather_side_features(
            data, idx, cand_lists, poi_coords=coords, poi_cat_table=poi_cat,
            n_categories=n_cats, max_length=int(valid.shape[1]),
            candidate_context=True, candidate_context_time="last_slot",
            session_summary=True)
        self.assertEqual(tuple(side["delta_geo"].shape[-1:]), (GEO_FEAT_DIM,))
        self.assertIn("last_slot", side)
        self.assertIn("session_cat_hist", side)
        prior = torch.zeros(len(idx), n_pois, 26)
        prior[..., 0] = 1.0
        conf = torch.ones(len(idx), n_pois)
        cfg = ModelConfig(
            8, 2, 1, 0., 2, router_mode="hard_prior", causal_counts=False,
            candidate_context="on", candidate_context_time="last_slot",
            session_summary="on")
        model = TSMoE(n_pois, n_cats, 4, cfg).eval()
        with torch.no_grad():
            out = model(h, valid, cand, prior, conf, count_ctx=side, details=True)
            h2 = {**h, "P": h["P"].clone()}
            h2["P"][:, 0] = 1
            out2 = model(h2, valid, cand, prior, conf, count_ctx=side, details=True)
        self.assertEqual(tuple(out["score"].shape), (len(idx), n_pois))
        self.assertTrue(torch.isfinite(out["score"]).all())
        # New features must not enter hard_prior logits.
        torch.testing.assert_close(out["logits"], out2["logits"])
        torch.testing.assert_close(out["logits"], torch.log1p(15 * out["mass"]))

    def test_user_pref_and_pmi_smoke(self):
        data, n_pois, n_cats = TestCausalCounts()._tiny_split()
        tables = CausalCountTables(n_pois, n_cats, visit_bins=4)
        tables.fit_visit_bins(data, 4)
        tables.replay(data)
        # PMI helpers return finite floats.
        self.assertTrue(math.isfinite(tables.poi_poi_pmi(0, 1)))
        self.assertEqual(len(tables.user_hour_pref_vector("u1")), 24)
        self.assertEqual(len(tables.user_cat_pref_vector("u1")), n_cats)
        idx = chronological_indices(data)[:2]
        h, valid, _ = make_batch(data, idx)
        cand_lists = [list(range(n_pois)) for _ in idx.tolist()]
        ctx = gather_batch_context(
            tables, data, idx, cand_lists, int(valid.shape[1]),
            update=False, transition_pmi=True)
        poi_cat = build_poi_cat_table(data, n_pois)
        coords = torch.zeros(n_pois, 2, dtype=torch.float64)
        side = gather_side_features(
            data, idx, cand_lists, poi_coords=coords, poi_cat_table=poi_cat,
            n_categories=n_cats, max_length=int(valid.shape[1]),
            tables=tables, user_pref=True)
        ctx = {**ctx, **side}
        prior = torch.zeros(len(idx), n_pois, 26)
        prior[..., 0] = 1.0
        conf = torch.ones(len(idx), n_pois)
        cfg = ModelConfig(
            8, 2, 1, 0., 2, router_mode="hard_prior", causal_counts=True,
            transition_attn="bias", visit_bins=4, user_pref="on",
            transition_pmi="on")
        model = TSMoE(n_pois, n_cats, 4, cfg).eval()
        with torch.no_grad():
            out = model(h, valid, torch.arange(n_pois).expand(len(idx), -1),
                        prior, conf, count_ctx=ctx, details=True)
        self.assertTrue(torch.isfinite(out["score"]).all())
        torch.testing.assert_close(out["logits"], torch.log1p(15 * out["mass"]))

    def test_expert_cross_attn_hybrid_and_full_smoke(self):
        torch.manual_seed(5)
        h = {"P": torch.tensor([[2, 3, 0]]), "C": torch.tensor([[2, 4, 0]]),
             "T": torch.randn(1, 3, T_DIM), "L": torch.randn(1, 3, L_DIM)}
        valid = torch.tensor([[True, True, False]])
        rows = torch.tensor([[0, 1, 2]])
        prior = torch.zeros(1, 3, 26)
        prior[..., 0] = 0.7
        prior[..., 15] = 0.3
        conf = torch.ones(1, 3)
        for mode, sparse in (("hybrid", "on"), ("hybrid", "off"), ("full", "on")):
            cfg = ModelConfig(
                8, 2, 1, 0., 2, router_mode="hard_prior", causal_counts=False,
                expert_cross_attn=mode, expert_cross_attn_sparse=sparse)
            model = TSMoE(5, 3, 3, cfg).eval()
            with torch.no_grad():
                out = model(h, valid, rows, prior, conf, details=True)
            self.assertEqual(tuple(out["score"].shape), (1, 3), msg=mode)
            self.assertTrue(torch.isfinite(out["score"]).all(), msg=mode)
            # Still SURD-only hard_prior.
            torch.testing.assert_close(
                out["logits"], torch.log1p(15 * out["mass"]),
                msg=f"{mode}/{sparse}")

    def test_delta_geo_features_finite(self):
        feats = delta_geo_features(35.0, 139.0, 35.1, 139.2)
        self.assertEqual(len(feats), 3)
        self.assertTrue(all(math.isfinite(v) for v in feats))
        self.assertGreater(feats[0], 0.0)

    def test_cli_overlay_train_smoke(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            directory = generate(root / "data")
            cache_path, run = root / "cache", root / "run_overlay"
            run_cli("prepare", "--data-root", directory, "--city", "demo", "--out", cache_path)
            run_cli(
                "train", "--cache", cache_path, "--out", run, "--epochs", 1,
                "--dim", 8, "--layers", 1, "--heads", 2, "--device", "cpu",
                "--negatives", 3, "--threads", 1,
                "--candidate-context", "on",
                "--candidate-context-time", "last_slot",
                "--session-summary", "on",
                "--user-pref", "on",
                "--transition-pmi", "on",
                "--expert-cross-attn", "hybrid",
                "--expert-cross-attn-sparse", "on",
                "--no-eval-test-every-epoch",
            )
            ckpt = torch.load(run / "best.pt", map_location="cpu", weights_only=False)
            cfg = ckpt["config"]
            self.assertEqual(cfg["candidate_context"], "on")
            self.assertEqual(cfg["candidate_context_time"], "last_slot")
            self.assertEqual(cfg["session_summary"], "on")
            self.assertEqual(cfg["user_pref"], "on")
            self.assertEqual(cfg["transition_pmi"], "on")
            self.assertEqual(cfg["expert_cross_attn"], "hybrid")
            self.assertEqual(cfg["expert_cross_attn_sparse"], "on")
            flags = format_ablation_key_flags(
                Cache(cache_path), ckpt.get("training_args"), cfg)
            self.assertIn("--candidate-context on", flags)
            self.assertIn("--expert-cross-attn hybrid", flags)
            run_cli("evaluate", "--cache", cache_path, "--checkpoint", run / "best.pt",
                    "--device", "cpu", "--split", "test", "--chunk", 3)


class AblationSummaryTests(unittest.TestCase):
    def test_leaf_and_expected_overlay_names(self):
        self.assertEqual(leaf_run_name("tokyo/abl_ctx_cat_geo"), "abl_ctx_cat_geo")
        self.assertEqual(
            expected_settings_for_run_name("tokyo/abl_ctx_cat_geo"),
            {"candidate_context": "on"},
        )
        self.assertEqual(
            expected_settings_for_run_name("abl_ctx_cat_geo_time")["candidate_context_time"],
            "last_slot",
        )
        self.assertEqual(
            expected_settings_for_run_name("abl_e1_dense"),
            {"expert_cross_attn": "hybrid", "expert_cross_attn_sparse": "off"},
        )
        self.assertEqual(
            expected_settings_for_run_name("tokyo/abl_filter_unknown"),
            {"filter_unknown": "on"},
        )
        self.assertEqual(
            expected_settings_for_run_name("tokyo/abl_cand_baseline"),
            {"candidate_radius": "p90"},
        )
        self.assertEqual(
            expected_settings_for_run_name("abl_cand_random"),
            {"candidate_radius": "random"},
        )
        self.assertIsNone(expected_settings_for_run_name("tokyo/model"))

    def test_parse_key_flags_and_validate(self):
        flags = (
            "--candidate-radius p90 --router-mode hard_prior --causal-counts "
            "--transition-attn bias --candidate-context on"
        )
        parsed = parse_key_flags(flags)
        self.assertEqual(parsed["router_mode"], "hard_prior")
        self.assertTrue(parsed["causal_counts"])
        self.assertEqual(parsed["candidate_context"], "on")
        status, detail = validate_run_name_settings("tokyo/abl_ctx_cat_geo", parsed)
        self.assertEqual(status, "ok")
        self.assertEqual(detail, "")
        bad = {**parsed, "candidate_context": "off"}
        status, detail = validate_run_name_settings("abl_ctx_cat_geo", bad)
        self.assertEqual(status, "mismatch")
        self.assertIn("candidate_context", detail)
        status, _ = validate_run_name_settings("abl_mystery_xyz", parsed)
        self.assertEqual(status, "unknown_name")

    def test_extract_recorded_settings_from_payload(self):
        payload = {
            "model_config": {
                "router_mode": "surd_mlp",
                "causal_counts": True,
                "candidate_context": "on",
                "expert_cross_attn": "off",
            },
            "training_args": {"negatives": 63, "max_train_candidates": -1},
            "candidate_construction": {
                "method": "multi_source",
                "radius_statistic": "p80",
                "popular_top_k": 0,
                "random_negatives": 50,
            },
            "target_mode": "all",
        }
        recorded = extract_recorded_settings(payload)
        self.assertEqual(recorded["router_mode"], "surd_mlp")
        self.assertEqual(recorded["candidate_radius"], "p80")
        self.assertEqual(recorded["popular_top_k"], 0)
        self.assertEqual(recorded["candidate_context"], "on")
        self.assertEqual(recorded["eval_targets"], "all")
        status, _ = validate_run_name_settings("abl_cand_p80", recorded)
        self.assertEqual(status, "ok")
        status, detail = validate_run_name_settings("abl_cand_p90", recorded)
        self.assertEqual(status, "mismatch")
        self.assertIn("candidate_radius", detail)

    def test_summary_scan_writes_md_csv_with_known_only(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            results = root / "results"
            run = results / "tokyo" / "abl_ctx_cat_geo"
            eval_dir = run / "eval_test"
            eval_dir.mkdir(parents=True)
            payload = {
                "split": "test",
                "cache": str(root / "runs" / "tokyo" / "cache"),
                "timestamp_utc": "2026-11-20T00:00:00Z",
                "target_mode": "all",
                "model_config": {
                    "router_mode": "hard_prior",
                    "causal_counts": True,
                    "candidate_context": "on",
                    "candidate_context_time": "off",
                    "session_summary": "off",
                    "user_pref": "off",
                    "transition_pmi": "off",
                    "expert_cross_attn": "off",
                },
                "training_args": {"router_mode": "hard_prior", "causal_counts": True},
                "candidate_construction": {
                    "method": "multi_source",
                    "radius_statistic": "p90",
                    "popular_top_k": 100,
                    "popular_radius": "max",
                    "user_frequent_top_k": 50,
                    "random_negatives": 50,
                },
                "all_targets": {
                    "Acc@1": 0.25, "NDCG@1": 0.25, "MRR": 0.4,
                    "samples": 4, "known_targets": 3, "coverage": 0.75,
                },
                "known_only": {
                    "Acc@1": 1 / 3, "NDCG@1": 1 / 3, "MRR": 0.5,
                    "samples": 3, "known_targets": 3, "coverage": 1.0,
                },
            }
            (eval_dir / "metrics.json").write_text(
                json.dumps(payload), encoding="utf-8")
            # Mismatching sibling run to exercise name_check.
            bad_run = results / "tokyo" / "abl_router_hard"
            bad_eval = bad_run / "eval_test"
            bad_eval.mkdir(parents=True)
            bad_payload = dict(payload)
            bad_payload["model_config"] = dict(payload["model_config"], router_mode="surd_mlp")
            bad_payload["training_args"] = dict(payload["training_args"], router_mode="surd_mlp")
            bad_payload["all_targets"] = dict(payload["all_targets"], **{"Acc@1": 0.1})
            (bad_eval / "metrics.json").write_text(
                json.dumps(bad_payload), encoding="utf-8")

            rows = collect_summary_rows(results)
            self.assertEqual(len(rows), 2)
            by_name = {r["run_name"]: r for r in rows}
            good = by_name["tokyo/abl_ctx_cat_geo"]
            self.assertEqual(good["name_check"], "ok")
            self.assertEqual(good["Acc@1"], "0.25")
            self.assertTrue(good["known_Acc@1"].startswith("0.333"))
            self.assertEqual(good["known_coverage"], "0.75")
            bad = by_name["tokyo/abl_router_hard"]
            self.assertEqual(bad["name_check"], "mismatch")
            self.assertIn("router_mode", bad["mismatch"])

            md_path, csv_path = write_summary_table(rows, results)
            self.assertTrue(md_path.is_file())
            self.assertTrue(csv_path.is_file())
            md = md_path.read_text(encoding="utf-8")
            self.assertIn("known_Acc@1", md)
            self.assertIn("name_check", md)
            self.assertIn("tokyo/abl_ctx_cat_geo", md)
            csv_text = csv_path.read_text(encoding="utf-8")
            self.assertTrue(csv_text.startswith("run_name,leaf_name,"))

            # CLI smoke
            run_cli("ablation-summary", "--results", results)
            self.assertIn("tokyo/abl_router_hard", (results / "ablation_summary.md").read_text())


if __name__ == "__main__":
    unittest.main()
