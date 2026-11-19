"""Training, exact catalog ranking, and candidate-level case study output."""
import json
import math
import os
import random
import tempfile
from collections import Counter, defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import torch
from tqdm.auto import tqdm

from .candidates import (
    format_pool_size_line, pools_for_split, sample_candidates, summarize_pool_sizes,
)
from .data import Categories, build_split, file_digest, make_batch, read_split
from .model import ModelConfig, TSMoE, objective
from .prepare import digest_json, dump_json, verify_files
from .runtime import check_interrupt, install_interrupt_handler
from .surd import ATOM_ORDER, EXPERT_ORDER, SOURCE_ORDER, FrozenSURDTable, gather_fold_priors

# Cutoffs used for Acc/Hit/Recall/NDCG/MRR/MAP tables (next-POI papers often include @20).
METRIC_KS = (1, 5, 10, 20)

# Primary headline metrics for terminal / report.txt (Acc@1 is the default focus).
PRIMARY_METRICS = ("Acc@1", "NDCG@1", "MRR")

# Human-readable labels for evaluate() group keys (JSON keys stay machine-stable).
# poi/* bins use SURD table target_count = # of times that POI is a train next-POI target.
GROUP_LABELS = {
    "poi/outside_catalog": "poi/outside_catalog — target not in train catalog",
    "poi/no_train_target": "poi/no_train_target — in catalog, 0 train targets",
    "poi/tail_1_2": "poi/tail — 1–2 train targets (long-tail)",
    "poi/mid_3_9": "poi/mid — 3–9 train targets",
    "poi/head_10_plus": "poi/head — ≥10 train targets (popular)",
    "history/1": "history/len=1 — prefix length 1",
    "history/2_5": "history/len=2–5 — prefix length 2–5",
    "history/6_plus": "history/len≥6 — prefix length ≥6",
    "last_category/0": "last_category/freq=0 — last-event category unseen in train",
    "last_category/1_9": "last_category/freq=1–9 — last-event category rare in train",
    "last_category/10_plus": "last_category/freq≥10 — last-event category common in train",
}

GROUP_NOTES = (
    "groups slice each eval sample by (1) poi/* = train-target popularity of the "
    "true next POI via SURD target_count: outside_catalog / no_train_target / "
    "tail_1_2 / mid_3_9 / head_10_plus; (2) history/* = observed prefix length "
    "bins 1 / 2_5 / 6_plus; (3) last_category/* = train frequency of the last "
    "history event's category bins 0 / 1_9 / 10_plus. Headline group columns are "
    "Acc@1, NDCG@1, full MRR."
)


def load_file(path):
    return torch.load(path, map_location="cpu", weights_only=True)


class Cache:
    def __init__(self, path):
        self.path = Path(path)
        self.meta = json.loads((self.path / "manifest.json").read_text())
        self.pois = self.meta["poi_ids"]
        self.cfg = self.meta["config"]
        if self.cfg["version"] != "tsmoe.massivesteps.v1" or digest_json(self.cfg) != self.meta["fingerprint"]:
            raise ValueError("incompatible cache or manifest fingerprint")
        self.id = file_digest(self.path / "manifest.json")
        # Candidate construction is fixed at prepare time so train/eval share one pool rule.
        construction = self.meta.get("candidate_construction")
        if construction is None:
            # Pre-spatial caches: full catalog (legacy).
            self.candidate_construction = dict(
                method="full_catalog", radius_statistic="full", radius_km=None,
                empty_union_fallback="full_catalog", train_eval_shared=True,
                eval_uses_full_pool=True)
            self.radius_km = None
            self.popular_radius_km = None
            self.popular_top_k = 0
            self.popularity = None
            self.user_frequent = {}
            self.random_negatives = 0
            self.random_negatives_seed = 0
            self.poi_coords = torch.full((len(self.pois), 2), float("nan"), dtype=torch.float64)
        else:
            self.candidate_construction = construction
            self.radius_km = construction.get("radius_km")
            self.popular_radius_km = construction.get("popular_radius_km")
            self.popular_top_k = int(construction.get("popular_top_k") or 0)
            self.random_negatives = int(construction.get("random_negatives") or 0)
            self.random_negatives_seed = int(
                construction.get("random_negatives_seed")
                if construction.get("random_negatives_seed") is not None else 0)
            coords = self.meta.get("poi_coordinates")
            if coords is None or len(coords) != len(self.pois):
                raise ValueError("cache missing poi_coordinates aligned with poi_ids; re-run prepare")
            self.poi_coords = torch.tensor(coords, dtype=torch.float64)
            popular_by = construction.get("popular_by") or "visit"
            visit = construction.get("poi_visit_counts")
            target = construction.get("poi_target_counts")
            if popular_by == "target" and target is not None and len(target) == len(self.pois):
                scores = target
            elif visit is not None and len(visit) == len(self.pois):
                scores = visit
            else:
                scores = [0] * len(self.pois)
            self.popularity = torch.tensor(scores, dtype=torch.float64)
            if "user_frequent.pt" in self.meta.get("files", {}):
                verify_files(self.path, {"user_frequent.pt": self.meta["files"]["user_frequent.pt"]})
                raw = load_file(self.path / "user_frequent.pt")
                self.user_frequent = {str(k): list(v) for k, v in raw.items()}
            else:
                self.user_frequent = {}

    def load(self, name):
        verify_files(self.path, {name: self.meta["files"][name]})
        return load_file(self.path / name)

    def split(self, name):
        return self.load(f"{name}.pt")

    def table(self, name, device):
        stats = self.load(f"{name}.surd.pt")
        info = stats["manifest"]
        if (stats["poi_ids"] != self.pois or info["cache_id"] != self.meta["fingerprint"]
                or info["feature_method"] != "basic_fixed_features_v1"
                or info["preprocessing_sha256"] != digest_json(self.meta["preprocessing"])):
            raise ValueError("SURD prior, catalog or preprocessing mismatch")
        pair = info["quantizer_manifest"]
        if pair:
            verify_files(self.path, {pair["file"]: pair["sha256"]})
        return FrozenSURDTable(stats).to(device)

    def model(self, cfg):
        return TSMoE(len(self.pois), len(self.meta["cat_vocab"]), self.cfg["max_length"], cfg)

    def uses_full_catalog_pools(self):
        """True when every sample scores against the full training catalog."""
        return self.radius_km is None or self.candidate_construction.get("method") == "full_catalog"

    def candidate_pools(self, data):
        """Same multi-source (or full-catalog) pools for train and evaluate.

        In-catalog ground-truth targets are always present (verified inside
        ``pools_for_split``); val/test ranking never drops the true next POI.
        """
        return pools_for_split(
            data, self.poi_coords, self.radius_km, len(self.pois),
            popular_radius_km=self.popular_radius_km,
            popularity=self.popularity,
            popular_top_k=self.popular_top_k,
            user_frequent=self.user_frequent,
            random_negatives=self.random_negatives,
            random_negatives_seed=self.random_negatives_seed,
        )

    def pool_size_stats(self, data):
        """Min/max/mean/median candidate counts for one split (scoring definition)."""
        pools, resolutions = self.candidate_pools(data)
        return summarize_pool_sizes(pools, resolutions), pools, resolutions


def select_device(name):
    """Resolve --device. Logs selection; warns when auto falls back to CPU."""
    cuda_ok = torch.cuda.is_available()
    if name == "auto":
        if cuda_ok:
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
            print("WARNING: --device auto fell back to CPU because "
                  "torch.cuda.is_available() is False "
                  "(CPU-only torch build, missing NVIDIA driver, or no GPU).",
                  flush=True)
    else:
        device = torch.device(name)
        if device.type == "cuda" and not cuda_ok:
            raise RuntimeError(
                f"--device {name} requested CUDA, but torch.cuda.is_available() "
                "is False. Install a CUDA PyTorch wheel (see README) and check "
                "nvidia-smi / drivers.")
        if device.type == "mps" and not (hasattr(torch.backends, "mps")
                                         and torch.backends.mps.is_available()):
            raise RuntimeError(f"--device {name} requested MPS, but it is not available")
    parts = [f"device={device}", f"cuda_available={cuda_ok}"]
    if device.type == "cuda":
        idx = device.index if device.index is not None else torch.cuda.current_device()
        parts.append(f"gpu={torch.cuda.get_device_name(idx)!r}")
    print("Using " + " ".join(parts), flush=True)
    return device


def device_batch(data, indices, device, labels=True):
    history, valid, target = make_batch(data, indices, labels=labels)
    return {s: x.to(device) for s, x in history.items()}, valid.to(device), target


def candidates(targets, size, negatives, rng, pools=None):
    """Uniform or spatial negative sampling. Returns (rows, labels) for tests; train uses mask too."""
    rows, labels, _mask = sample_candidates(targets, size, negatives, rng, pools=pools)
    return rows, labels


def all_scores(model, table, encoded, valid, size, chunk):
    """Keep only scalar scores on the CPU across chunks. Encode the batch history once."""
    parts = []
    for start in range(0, size, chunk):
        check_interrupt()
        rows = torch.arange(start, min(size, start + chunk), device=valid.device).expand(len(valid), -1)
        prior, confidence = table(rows)
        parts.append(model.score(encoded, valid, rows, prior, confidence)["score"].cpu())
    scores = torch.cat(parts, 1)
    if not torch.isfinite(scores).all():
        raise FloatingPointError("nonfinite full-catalog scores")
    return scores


def score_pools(model, table, encoded, valid, pools, chunk):
    """Score variable-size candidate pools; pad with mask (shared train/eval path)."""
    width = max(int(pool.numel()) for pool in pools)
    rows = torch.zeros(len(pools), width, dtype=torch.long, device=valid.device)
    mask = torch.zeros(len(pools), width, dtype=torch.bool, device=valid.device)
    for i, pool in enumerate(pools):
        n = int(pool.numel())
        rows[i, :n] = pool.to(valid.device)
        mask[i, :n] = True
    parts = []
    for start in range(0, width, chunk):
        check_interrupt()
        sl = slice(start, min(width, start + chunk))
        chunk_rows, chunk_mask = rows[:, sl], mask[:, sl]
        safe = chunk_rows.masked_fill(~chunk_mask, -1)
        prior, confidence = table(safe)
        scored = model.score(encoded, valid, chunk_rows, prior, confidence, candidate_mask=chunk_mask)
        part = scored["score"].cpu()
        part = part.masked_fill(~chunk_mask.cpu(), float("-inf"))
        parts.append(part)
    scores = torch.cat(parts, 1)
    finite = torch.isfinite(scores) | torch.isneginf(scores)
    if not finite.all():
        raise FloatingPointError("nonfinite candidate-pool scores")
    return scores, rows.cpu(), mask.cpu()


def ranks_in_pools(pool_scores, pool_rows, pool_mask, targets):
    """Rank the target inside each sample's candidate pool (1-based; 0 = unknown)."""
    ranks = []
    for scores, rows, mask, target in zip(pool_scores, pool_rows, pool_mask, targets.tolist()):
        if target < 0:
            ranks.append(0)
            continue
        valid_scores = scores[mask]
        valid_rows = rows[mask]
        hit = (valid_rows == target).nonzero(as_tuple=False)
        if hit.numel() == 0:
            ranks.append(0)
            continue
        pos = int(hit[0])
        truth = valid_scores[pos]
        earlier = torch.arange(len(valid_scores)) < pos
        rank = 1 + int((valid_scores > truth).sum()) + int(((valid_scores == truth) & earlier).sum())
        ranks.append(rank)
    return ranks


def ranks_of(scores, target):
    known = target >= 0
    row = target.clamp_min(0)
    truth = scores.gather(1, row[:, None])
    earlier = torch.arange(scores.shape[1])[None, :] < row[:, None]
    rank = 1 + (scores > truth).sum(1) + ((scores == truth) & earlier).sum(1)
    return rank.masked_fill(~known, 0)  # zero is the sentinel for out-of-catalog


def metric_row(ranks):
    """Aggregate ranks into ranking metrics.

    Single correct next-POI target ⇒ Acc@K = Hit@K = Recall@K, and MAP@K = MRR@K
    (AP of one relevant item is 1/rank when rank ≤ K). Unknown / out-of-catalog
    targets use rank sentinel 0 and contribute 0 to hit-based metrics under the
    all_targets denominator; known_only drops them first.
    """
    ranks = torch.as_tensor(ranks)
    known = ranks > 0
    n = len(ranks)
    known_ranks = ranks[known].double()
    out = dict(samples=n, known_targets=int(known.sum()),
               coverage=float(known.double().mean()) if n else None)
    if known_ranks.numel():
        out["mean_rank"] = float(known_ranks.mean())
        out["median_rank"] = float(known_ranks.median())
        # Full MRR (no K cutoff) over the same denominator as @K metrics.
        recip = torch.zeros(n, dtype=torch.float64)
        recip[known] = 1.0 / known_ranks
        out["MRR"] = float(recip.mean()) if n else None
    else:
        out["mean_rank"] = out["median_rank"] = out["MRR"] = None
    for k in METRIC_KS:
        hit = known & (ranks <= k)
        discount = ranks[hit].double()
        recall = float(hit.sum()) / n if n else None
        ndcg = ((1 / (1 + discount).log2()).sum().item() / n) if n else None
        mrr_at = ((1 / discount).sum().item() / n) if n else None
        out[f"Recall@{k}"] = recall
        out[f"Hit@{k}"] = recall
        out[f"Acc@{k}"] = recall
        out[f"NDCG@{k}"] = ndcg
        out[f"MRR@{k}"] = mrr_at
        out[f"MAP@{k}"] = mrr_at  # single relevant ⇒ MAP@K == MRR@K
    return out


def _fmt(value, width=8):
    if value is None:
        return f"{'—':>{width}}"
    if isinstance(value, float):
        return f"{value:{width}.4f}"
    return f"{value:>{width}}"


def format_primary_line(row, indent="  "):
    """One-line Acc@1 / NDCG@1 / MRR headline (default / primary metrics)."""
    return (f"{indent}primary  "
            f"Acc@1={_fmt(row.get('Acc@1')).strip()}  "
            f"NDCG@1={_fmt(row.get('NDCG@1')).strip()}  "
            f"MRR={_fmt(row.get('MRR')).strip()}")


def format_metric_block(row, title):
    """Compact human-readable table for one metric_row dict; Acc@1 leads."""
    lines = [title,
             format_primary_line(row),
             f"  samples={row['samples']}  known={row['known_targets']}  "
             f"coverage={_fmt(row['coverage']).strip()}  "
             f"mean_rank={_fmt(row['mean_rank']).strip()}  "
             f"median_rank={_fmt(row['median_rank']).strip()}"]
    header = f"  {'metric':<8}" + "".join(f"{'@' + str(k):>8}" for k in METRIC_KS)
    lines.append(header)
    for name in ("Acc", "Recall", "Hit", "NDCG", "MRR", "MAP"):
        # Acc/Hit/Recall share values; still print Acc + NDCG + MRR as the main trio,
        # and MAP once (aliases documented in README).
        if name in ("Recall", "Hit"):
            continue
        cells = "".join(_fmt(row.get(f"{name}@{k}")) for k in METRIC_KS)
        label = "Acc/Hit" if name == "Acc" else name
        lines.append(f"  {label:<8}{cells}")
    return "\n".join(lines)


def format_group_key(key):
    """Prefer the annotated label; fall back to the raw group key."""
    return GROUP_LABELS.get(key, key)


def format_eval_report(payload, group_keys=None):
    """Terminal / report.txt text: overview + main tables + short group summary."""
    lines = ["=== Evaluation ==="]
    for key in ("split", "device", "cache", "cache_id", "checkpoint", "catalog_size",
                "target_mode", "timestamp_utc"):
        value = payload.get(key)
        if value not in (None, ""):
            lines.append(f"  {key}={value}")
    pool_stats = payload.get("candidate_pool_stats")
    if pool_stats:
        lines.append(f"  {format_pool_size_line(payload.get('split') or 'split', pool_stats)}")
    # Lead the whole report with Acc@1 on all_targets (default / primary metric).
    all_row = payload["all_targets"]
    lines.append(
        f"  summary  Acc@1={_fmt(all_row.get('Acc@1')).strip()}  "
        f"NDCG@1={_fmt(all_row.get('NDCG@1')).strip()}  "
        f"MRR={_fmt(all_row.get('MRR')).strip()}  "
        f"(primary; full tables below)")
    lines.append("")
    lines.append(format_metric_block(all_row, "all_targets (OOV / unknown → miss)"))
    lines.append("")
    lines.append(format_metric_block(payload["known_only"], "known_only (in-catalog targets)"))
    groups = payload.get("groups") or {}
    if groups:
        keys = group_keys or sorted(groups)
        lines.append("")
        lines.append("groups — Acc@1 / NDCG@1 / MRR / n_known  "
                     "(poi = train-target popularity; history = prefix length; "
                     "last_category = train freq of last event's category)")
        for key in keys:
            row = groups[key]
            label = format_group_key(key)
            lines.append(
                f"  {label}")
            lines.append(
                f"    Acc@1={_fmt(row.get('Acc@1')).strip()}  "
                f"NDCG@1={_fmt(row.get('NDCG@1')).strip()}  "
                f"MRR={_fmt(row.get('MRR')).strip()}  "
                f"known={row['known_targets']}/{row['samples']}")
    return "\n".join(lines) + "\n"


def print_eval_tables(payload):
    """Print a tqdm-friendly summary (plain lines, no JSON dump)."""
    text = format_eval_report(payload)
    print(text, end="", flush=True)


def results_run_dir(path, *, cwd=None):
    """Map a train --out or checkpoint-parent path to results/<run_name>/.

    Under the working directory, strips a leading ``runs/`` segment so
    ``runs/tokyo/model`` → ``<cwd>/results/tokyo/model``. Paths outside the
    working directory (e.g. temp dirs in tests) use a sibling
    ``<parent>/results/<basename>``.
    """
    p = Path(path).expanduser().resolve()
    base = Path(cwd or Path.cwd()).resolve()
    try:
        rel = p.relative_to(base)
    except ValueError:
        return p.parent / "results" / p.name
    parts = rel.parts
    if parts and parts[0] == "runs":
        run_name = Path(*parts[1:]) if len(parts) > 1 else Path(p.name)
    else:
        run_name = rel if parts else Path(p.name)
    return base / "results" / run_name


def resolve_eval_out(args):
    """Directory for metrics.json / report.txt; default under results/."""
    if getattr(args, "out", None):
        return Path(args.out)
    # Mirror train --out under results/: runs/<city>/model → results/<city>/model/eval_<split>
    return results_run_dir(Path(args.checkpoint).resolve().parent) / f"eval_{args.split}"


def resolve_train_eval_dir(train_out, split):
    """Where train writes final eval artifacts (not checkpoints)."""
    return results_run_dir(train_out) / f"eval_{split}"


def build_eval_payload(args, cache, metrics, device, checkpoint_path=None, checkpoint=None):
    settings = {key: value for key, value in vars(args).items()
                if isinstance(value, (str, float, int, bool))}
    ckpt_path = checkpoint_path or getattr(args, "checkpoint", None)
    ckpt = checkpoint
    if ckpt is None and ckpt_path and Path(ckpt_path).is_file():
        ckpt = load_file(ckpt_path)
    ckpt = ckpt or {}
    return dict(
        timestamp_utc=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        split=getattr(args, "split", None),
        catalog_size=len(cache.pois),
        target_mode=cache.cfg["eval_targets"],
        cache=str(Path(args.cache).resolve()),
        cache_id=cache.id,
        checkpoint=str(Path(ckpt_path).resolve()) if ckpt_path else None,
        device=str(device),
        cuda_available=torch.cuda.is_available(),
        model_config=ckpt.get("config"),
        training_args=ckpt.get("training_args"),
        checkpoint_epoch=ckpt.get("epoch"),
        checkpoint_validation_ndcg10=ckpt.get("validation_ndcg10"),
        cli=settings,
        metric_ks=list(METRIC_KS),
        primary_metrics=list(PRIMARY_METRICS),
        metric_notes=("Primary headline metrics are Acc@1, NDCG@1, and full MRR. "
                      "Acc@K=Hit@K=Recall@K and MAP@K=MRR@K for single next-POI target; "
                      "rank ties break by lower catalog row index; "
                      "unknown targets use rank 0 and count as misses under all_targets. "
                      "When candidate_construction.method is multi_source, ranks are within "
                      "the per-sample constructed pool (full union at eval; training may "
                      "subsample negatives from the same pool)."),
        candidate_construction=cache.candidate_construction,
        group_notes=GROUP_NOTES,
        group_labels=dict(GROUP_LABELS),
        **metrics,
    )


def report_candidate_pool_stats(cache, splits):
    """Build and print per-split pool-size stats (same pools as train negatives / eval).

    ``splits`` is an ordered mapping of split name → split tensor dict.
    Returns ``(stats_by_split, pools_by_split)``.
    """
    radius_info = cache.candidate_construction
    print(
        f"candidates method={radius_info.get('method')} "
        f"radius_statistic={radius_info.get('radius_statistic')} "
        f"radius_km={cache.radius_km} "
        f"popular_radius={radius_info.get('popular_radius')} "
        f"popular_radius_km={cache.popular_radius_km} "
        f"popular_top_k={cache.popular_top_k} "
        f"popular_by={radius_info.get('popular_by')} "
        f"user_frequent_top_k={radius_info.get('user_frequent_top_k')} "
        f"random_negatives={cache.random_negatives} "
        f"eval_pool=full_constructed ground_truth_in_pool=always",
        flush=True,
    )
    all_stats = {}
    pools_by_split = {}
    for name, data in splits.items():
        stats, pools, _resolutions = cache.pool_size_stats(data)
        all_stats[name] = stats
        pools_by_split[name] = pools
        print(format_pool_size_line(name, stats), flush=True)
    return all_stats, pools_by_split


def save_eval_results(payload, out_dir, legacy_output=None):
    """Write metrics.json + report.txt under out_dir; optionally mirror JSON to --output."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    metrics_path = out / "metrics.json"
    report_path = out / "report.txt"
    dump_json(payload, metrics_path)
    report_path.write_text(format_eval_report(payload), encoding="utf-8")
    print(f"Wrote {metrics_path}", flush=True)
    print(f"Wrote {report_path}", flush=True)
    if legacy_output:
        legacy = Path(legacy_output)
        if legacy.resolve() != metrics_path.resolve():
            legacy.parent.mkdir(parents=True, exist_ok=True)
            dump_json(payload, legacy)
            print(f"Wrote {legacy}", flush=True)
    return metrics_path, report_path


@torch.inference_mode()
def evaluate(model, table, data, cache, batch_size, chunk, cat_frequency=None, desc="eval"):
    model.eval()
    ranks, grouped = [], defaultdict(list)
    counts = table.target_count.cpu()
    device = table.routing_prior.device
    pools, _resolutions = cache.candidate_pools(data)
    batches = list(torch.arange(len(data["end"])).split(batch_size))
    for idx in tqdm(batches, desc=desc, leave=False, dynamic_ncols=True):
        check_interrupt()
        h, valid, target = device_batch(data, idx, device)
        encoded = model.encode(h, valid)
        if cache.uses_full_catalog_pools():
            scores = all_scores(model, table, encoded, valid, len(cache.pois), chunk)
            batch_ranks = ranks_of(scores, target).tolist()
        else:
            # Eval always ranks inside the full constructed multi-source pool.
            batch_pools = [pools[i] for i in idx.tolist()]
            pool_scores, pool_rows, pool_mask = score_pools(
                model, table, encoded, valid, batch_pools, chunk)
            batch_ranks = ranks_in_pools(pool_scores, pool_rows, pool_mask, target.cpu())
        ranks.extend(batch_ranks)
        if cat_frequency is not None:
            # Slice ranks into Head/Mid/Tail-style popularity bins plus context bins.
            # count = SURD target_count (# times this POI is a train next-POI target).
            for i, row, rank in zip(idx.tolist(), target.tolist(), batch_ranks):
                end = int(data["end"][i])
                length = end - int(data["start"][i])
                count = int(counts[row]) if row >= 0 else 0
                cat_count = cat_frequency[int(data["C"][end - 1])]
                popularity = ("outside_catalog" if row < 0 else "no_train_target" if count == 0 else
                              "tail_1_2" if count <= 2 else "mid_3_9" if count < 10 else "head_10_plus")
                length_bin = "1" if length == 1 else "2_5" if length <= 5 else "6_plus"
                cat_bin = "0" if not cat_count else "1_9" if cat_count < 10 else "10_plus"
                for key in (f"poi/{popularity}", f"history/{length_bin}", f"last_category/{cat_bin}"):
                    grouped[key].append(rank)
    return dict(all_targets=metric_row(ranks), known_only=metric_row([r for r in ranks if r]),
                groups={key: metric_row(rs) for key, rs in sorted(grouped.items())})


def save_checkpoint(payload, path):
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as tmp:
        staged = Path(tmp.name)
    try:
        torch.save(payload, staged)
        os.replace(staged, path)
    finally:
        staged.unlink(missing_ok=True)


def train(args):
    install_interrupt_handler()
    cache = Cache(args.cache)
    cfg = ModelConfig(args.dim, args.heads, args.layers, args.dropout, args.top_k,
                      args.temperature, args.lambda_prior, use_surd=not args.no_surd)
    device = select_device(args.device)
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    batches = torch.Generator().manual_seed(args.seed)
    model = cache.model(cfg).to(device)
    param_device = next(model.parameters()).device
    print(f"model_param_device={param_device}", flush=True)
    if param_device.type != device.type or (
            device.index is not None and param_device.index not in (None, device.index)):
        raise RuntimeError(f"model landed on {param_device}, expected {device}")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    data, val = cache.split("train"), cache.split("validation")
    tables = [cache.table(info["name"], device) for info in cache.meta["tables"] if info["name"] != "final"]
    final = cache.table("final", device)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    best, history, best_metrics = -math.inf, [], None
    settings = {key: value for key, value in vars(args).items() if isinstance(value, (str, float, int, bool))}
    settings["candidate_radius"] = cache.cfg.get("candidate_radius", "full")
    settings["candidate_radius_km"] = cache.radius_km
    settings["popular_radius_km"] = cache.popular_radius_km
    settings["random_negatives"] = cache.random_negatives
    settings["max_train_candidates"] = args.max_train_candidates
    max_train_candidates = None if args.max_train_candidates < 0 else args.max_train_candidates
    category_frequency = Counter(data["C"].tolist())
    test = cache.split("test")
    pool_stats, pools_by_split = report_candidate_pool_stats(
        cache, {"train": data, "validation": val, "test": test})
    train_pools = pools_by_split["train"]
    # Persist pool-size audit under results/ (not beside checkpoints).
    stats_dir = results_run_dir(out)
    stats_dir.mkdir(parents=True, exist_ok=True)
    dump_json(dict(
        candidate_construction=cache.candidate_construction,
        catalog_size=len(cache.pois),
        cache=str(Path(args.cache).resolve()),
        cache_id=cache.id,
        train_sampling=dict(
            negatives=args.negatives,
            max_train_candidates=args.max_train_candidates,
            eval_uses_full_pool=True,
        ),
        definition=("per-sample pool size after multi-source union "
                    "(neighborhood + popular + user-frequent + random negatives) "
                    "+ force-include in-catalog target (always, including val/test) "
                    "+ empty-union fallback to full catalog; stats match the full "
                    "constructed pool used at evaluation (train may subsample)"),
        splits=pool_stats,
    ), stats_dir / "candidate_pool_stats.json")
    print(f"Wrote {stats_dir / 'candidate_pool_stats.json'}", flush=True)
    epoch_bar = tqdm(range(1, args.epochs + 1), desc="epochs", dynamic_ncols=True)
    for epoch in epoch_bar:
        check_interrupt()
        model.train()
        loss_sum, components = 0., torch.zeros(3, dtype=torch.float64)
        counts = torch.zeros(15, dtype=torch.long, device=device)
        steps = list(torch.randperm(len(data["end"]), generator=batches).split(args.batch_size))
        for idx in tqdm(steps, desc=f"train ep{epoch}", leave=False, dynamic_ncols=True):
            check_interrupt()
            h, valid, target = device_batch(data, idx, device)
            batch_pools = [train_pools[i] for i in idx.tolist()]
            rows, positive, cand_mask = sample_candidates(
                target.detach().cpu(), len(cache.pois), args.negatives, rng,
                pools=batch_pools, max_candidates=max_train_candidates)
            rows, cand_mask = rows.to(device), cand_mask.to(device)
            prior, confidence = gather_fold_priors(
                tables, data["fold"][idx], rows.masked_fill(~cand_mask, -1))
            result = model(h, valid, rows, prior, confidence, candidate_mask=cand_mask)
            loss, parts = objective(result, positive.to(device), args.lambda_surd, args.lambda_balance)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip, error_if_nonfinite=True)
            opt.step()
            counts += torch.bincount(result["top"].detach().reshape(-1), minlength=15)
            loss_sum += float(loss.detach()) * len(idx)
            components += torch.stack(parts).double().cpu() * len(idx)
        # Pass category_frequency so val (and saved best) always include groups + Acc@1.
        metrics = evaluate(model, final, val, cache, args.eval_batch_size, args.chunk,
                           category_frequency, desc=f"val ep{epoch}")
        score = metrics["all_targets"]["NDCG@10"]
        record = dict(epoch=epoch, loss=loss_sum / len(data["end"]),
                      loss_terms=dict(zip(("rec", "surd", "balance"), (components / len(data["end"])).tolist())),
                      dispatch_counts=dict(zip(EXPERT_ORDER, counts.cpu().tolist())), validation=metrics)
        history.append(record)
        if score > best:
            best = score
            best_metrics = metrics
            save_checkpoint(dict(cache_id=cache.id, config=asdict(cfg), weights=model.state_dict(),
                                 epoch=epoch, validation_ndcg10=score, training_args=settings), out / "best.pt")
        dump_json({"training_args": settings, "candidate_pool_stats": pool_stats,
                   "epochs": history}, out / "history.json")
        row = metrics["all_targets"]
        epoch_bar.set_postfix(loss=f"{record['loss']:.5f}", acc1=f"{row['Acc@1']:.4f}",
                              ndcg10=f"{score:.6f}")
        print(f"epoch={epoch} loss={record['loss']:.5f}  "
              f"val Acc@1={row['Acc@1']:.4f} NDCG@1={row['NDCG@1']:.4f} MRR={row['MRR']:.4f}  "
              f"Acc@10={row['Acc@10']:.4f} NDCG@10={row['NDCG@10']:.4f}", flush=True)
    if best_metrics is not None:
        best_path = out / "best.pt"
        val_payload = build_eval_payload(args, cache, best_metrics, device,
                                         checkpoint_path=best_path)
        val_payload["split"] = "validation"
        val_payload["selection"] = "best_checkpoint_by_NDCG@10"
        val_payload["candidate_pool_stats"] = pool_stats.get("validation")
        print_eval_tables(val_payload)
        save_eval_results(val_payload, resolve_train_eval_dir(out, "validation"))
        # Final test once on the selected checkpoint (not the last-epoch weights).
        checkpoint = load_file(best_path)
        model.load_state_dict(checkpoint["weights"])
        model.eval()
        test_metrics = evaluate(model, final, test, cache, args.eval_batch_size,
                                args.chunk, category_frequency, desc="eval test")
        test_payload = build_eval_payload(args, cache, test_metrics, device,
                                          checkpoint_path=best_path, checkpoint=checkpoint)
        test_payload["split"] = "test"
        test_payload["selection"] = "best_checkpoint_by_NDCG@10"
        test_payload["candidate_pool_stats"] = pool_stats.get("test")
        print_eval_tables(test_payload)
        save_eval_results(test_payload, resolve_train_eval_dir(out, "test"))


def load_model(args):
    cache, device = Cache(args.cache), select_device(args.device)
    checkpoint = load_file(args.checkpoint)
    if cache.id != checkpoint["cache_id"]:
        raise ValueError("checkpoint trained with a different cache; check --cache and --checkpoint")
    model = cache.model(ModelConfig(**checkpoint["config"])).to(device)
    model.load_state_dict(checkpoint["weights"])
    print(f"model_param_device={next(model.parameters()).device}", flush=True)
    return cache, model.eval(), cache.table("final", device), checkpoint


def report(value, dest):
    if dest:
        Path(dest).parent.mkdir(parents=True, exist_ok=True)
        dump_json(value, dest)
        print("Wrote", dest)
    else:
        print(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))


def evaluate_cli(args):
    install_interrupt_handler()
    cache, model, table, checkpoint = load_model(args)
    device = table.routing_prior.device
    training = cache.split("train")
    category_frequency = Counter(training["C"].tolist())
    data = cache.split(args.split)
    pool_stats, _ = report_candidate_pool_stats(cache, {args.split: data})
    metrics = evaluate(model, table, data, cache, args.eval_batch_size,
                       args.chunk, category_frequency, desc=f"eval {args.split}")
    payload = build_eval_payload(args, cache, metrics, device,
                                 checkpoint_path=args.checkpoint, checkpoint=checkpoint)
    payload["candidate_pool_stats"] = pool_stats.get(args.split)
    print_eval_tables(payload)
    out_dir = resolve_eval_out(args)
    save_eval_results(payload, out_dir, legacy_output=args.output)


@torch.inference_mode()
def explain(args):
    install_interrupt_handler()
    cache, model, table, _ = load_model(args)
    data = cache.split(args.split)
    i = args.sample_index
    if not 0 <= i < len(data["end"]):
        raise ValueError("sample-index outside split")
    h, valid, target = device_batch(data, torch.tensor([i]), table.routing_prior.device)
    encoded = model.encode(h, valid)
    pools, resolutions = cache.candidate_pools(data)
    pool, resolution = pools[i], resolutions[i]
    if args.poi_id is not None and args.poi_id not in table.row_of:
        raise ValueError("candidate POI outside the training catalog")
    if cache.uses_full_catalog_pools():
        scores = all_scores(model, table, encoded, valid, len(cache.pois), args.chunk)
        if args.poi_id is None:
            row = int(target[0]) if target[0] >= 0 else int(scores.argmax())
        else:
            row = table.row_of[args.poi_id]
        candidate_rank = int(ranks_of(scores, torch.tensor([row]))[0])
    else:
        pool_scores, pool_rows, pool_mask = score_pools(
            model, table, encoded, valid, [pool], args.chunk)
        if args.poi_id is None:
            if target[0] >= 0:
                row = int(target[0])
            else:
                active = pool_mask[0]
                row = int(pool_rows[0][active][pool_scores[0][active].argmax()])
        else:
            row = table.row_of[args.poi_id]
        candidate_rank = ranks_in_pools(pool_scores, pool_rows, pool_mask, torch.tensor([row]))[0]
    rows = torch.tensor([[row]], device=valid.device)
    prior, confidence = table(rows)
    result = model.score(encoded, valid, rows, prior, confidence, details=True)
    def get(key):
        return result[key][0, 0].tolist()
    top = [EXPERT_ORDER[j] for j in get("top")]
    start, end = int(data["start"][i]), int(data["end"][i])
    raw_poi = cache.pois[row]
    user, trail = data["trails"][data["trail"][end]]
    record = dict(sample_index=i, split=args.split, user_id=user, trail_id=trail,
                  prefix_end_utc_us=int(data["us"][end - 1]), target_utc_us=int(data["us"][end]),
                  candidate_poi_id=raw_poi, target_poi_id=int(data["P"][end]), is_true_next_poi=row == int(target[0]),
                  history_poi_ids=data["P"][start:end].tolist(), history_category_ids=data["C"][start:end].tolist(),
                  history_time_features=h["T"][0].tolist(), history_location_features=h["L"][0].tolist(),
                  prior_version=cache.id, source_order=SOURCE_ORDER, atom_order=ATOM_ORDER, expert_order=EXPERT_ORDER,
                  candidate_construction=cache.candidate_construction,
                  candidate_pool_size=int(pool.numel()), candidate_pool_resolution=resolution,
                  target_count=int(table.target_count[row]), total_information_bits=float(table.total_information[row]),
                  near_tie_in_surd=bool(table.near_tie[row]),
                  exact_tie_in_route=len(set(get("logits"))) < len(EXPERT_ORDER),
                  raw_surd=table.raw_surd[row].tolist(), normalized_surd=table.normalized_surd[row].tolist(),
                  routing_prior=prior[0, 0].tolist(), confidence=float(confidence),
                  effective_confidence=get("confidence"), us_mass=get("mass"),
                  alpha=get("alpha"), redundancy_mass=float(prior[..., 4:15].sum()), route_mlp_logits=get("learned"),
                  explicit_prior_bias=get("bias"), final_route_logits=get("logits"),
                  dense_route_probability=result["log_pi"][0, 0].exp().tolist(), selected_experts=top,
                  selected_weights=get("weights"), gate=get("gate"), base_score=get("base"),
                  expert_projected_scores=dict(zip(top, get("projected"))),
                  expert_score_contributions=dict(zip(top, get("contributions"))),
                  moe_bias_contribution=get("moe_bias"), final_score=get("score"),
                  candidate_rank=candidate_rank)
    report(record, args.output)


@torch.inference_mode()
def recommend(args):
    install_interrupt_handler()
    cache, model, table, _ = load_model(args)
    groups, _ = read_split(args.history)
    cats = Categories.from_manifest(cache.meta["preprocessing"])
    data = build_split(groups, cats, cache.pois, cache.meta["cat_vocab"], cache.cfg["max_length"], "predict")
    pools, resolutions = cache.candidate_pools(data)
    records = []
    batches = list(torch.arange(len(data["end"])).split(args.eval_batch_size))
    for idx in tqdm(batches, desc="recommend", dynamic_ncols=True):
        check_interrupt()
        h, valid, _ = device_batch(data, idx, table.routing_prior.device, labels=False)
        encoded = model.encode(h, valid)
        if cache.uses_full_catalog_pools():
            scores = all_scores(model, table, encoded, valid, len(cache.pois), args.chunk)
            top = torch.argsort(scores, descending=True, stable=True)[:, :args.top_n]
            for i, js, row_scores in zip(idx.tolist(), top.tolist(), scores):
                user, trail = data["trails"][data["trail"][data["end"][i] - 1]]
                records.append(dict(user_id=user, trail_id=trail,
                                    candidate_pool_resolution=resolutions[i],
                                    recommendations=[
                                        dict(venue_id=cache.pois[j], score=float(row_scores[j]))
                                        for j in js]))
        else:
            batch_pools = [pools[i] for i in idx.tolist()]
            pool_scores, pool_rows, pool_mask = score_pools(
                model, table, encoded, valid, batch_pools, args.chunk)
            for bi, i in enumerate(idx.tolist()):
                active = pool_mask[bi]
                scores = pool_scores[bi][active]
                rows = pool_rows[bi][active]
                order = torch.argsort(scores, descending=True, stable=True)[:args.top_n]
                user, trail = data["trails"][data["trail"][data["end"][i] - 1]]
                records.append(dict(
                    user_id=user, trail_id=trail,
                    candidate_pool_resolution=resolutions[i],
                    candidate_pool_size=int(active.sum()),
                    recommendations=[
                        dict(venue_id=cache.pois[int(rows[j])], score=float(scores[j]))
                        for j in order.tolist()]))
    report(records, args.output)
