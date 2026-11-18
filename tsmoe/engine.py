"""Training, exact catalog ranking, and candidate-level case study output."""
import json
import math
import os
import random
import tempfile
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path

import torch

from .data import Categories, build_split, file_digest, make_batch, read_split
from .model import ModelConfig, TSMoE, objective
from .prepare import digest_json, dump_json, verify_files
from .surd import ATOM_ORDER, EXPERT_ORDER, SOURCE_ORDER, FrozenSURDTable, gather_fold_priors


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


def select_device(name):
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def device_batch(data, indices, device, labels=True):
    history, valid, target = make_batch(data, indices, labels=labels)
    return {s: x.to(device) for s, x in history.items()}, valid.to(device), target


def candidates(targets, size, negatives, rng):
    if negatives < 0 or negatives >= size - 1:
        return torch.arange(size).expand(len(targets), -1), targets
    rows = []
    for target in targets.tolist():
        chosen = rng.sample(range(size - 1), negatives)
        rows.append([target, *(n + (n >= target) for n in chosen)])
    return torch.tensor(rows), torch.zeros_like(targets)


def all_scores(model, table, encoded, valid, size, chunk):
    """Keep only scalar scores on the CPU across chunks. Encode the batch history once."""
    parts = []
    for start in range(0, size, chunk):
        rows = torch.arange(start, min(size, start + chunk), device=valid.device).expand(len(valid), -1)
        prior, confidence = table(rows)
        parts.append(model.score(encoded, valid, rows, prior, confidence)["score"].cpu())
    scores = torch.cat(parts, 1)
    if not torch.isfinite(scores).all():
        raise FloatingPointError("nonfinite full-catalog scores")
    return scores


def ranks_of(scores, target):
    known = target >= 0
    row = target.clamp_min(0)
    truth = scores.gather(1, row[:, None])
    earlier = torch.arange(scores.shape[1])[None, :] < row[:, None]
    rank = 1 + (scores > truth).sum(1) + ((scores == truth) & earlier).sum(1)
    return rank.masked_fill(~known, 0)  # zero is the sentinel for out-of-catalog


def metric_row(ranks):
    ranks = torch.as_tensor(ranks)
    known = ranks > 0
    out = dict(samples=len(ranks), known_targets=int(known.sum()),
               coverage=float(known.double().mean()) if len(ranks) else None)
    for k in (1, 5, 10):
        hit = known & (ranks <= k)
        discount = ranks[hit].double()
        for label, total in (("Recall", float(hit.sum())), ("NDCG", (1 / (1 + discount).log2()).sum().item()),
                             ("MRR", (1 / discount).sum().item())):
            out[f"{label}@{k}"] = total / len(ranks) if len(ranks) else None
    return out


@torch.inference_mode()
def evaluate(model, table, data, cache, batch_size, chunk, cat_frequency=None):
    model.eval()
    ranks, grouped = [], defaultdict(list)
    counts = table.target_count.cpu()
    device = table.routing_prior.device
    for idx in torch.arange(len(data["end"])).split(batch_size):
        h, valid, target = device_batch(data, idx, device)
        scores = all_scores(model, table, model.encode(h, valid), valid, len(cache.pois), chunk)
        batch_ranks = ranks_of(scores, target).tolist()
        ranks.extend(batch_ranks)
        if cat_frequency is not None:
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
    cache = Cache(args.cache)
    cfg = ModelConfig(args.dim, args.heads, args.layers, args.dropout, args.top_k,
                      args.temperature, args.lambda_prior, use_surd=not args.no_surd)
    device = select_device(args.device)
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    batches = torch.Generator().manual_seed(args.seed)
    model = cache.model(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    data, val = cache.split("train"), cache.split("validation")
    tables = [cache.table(info["name"], device) for info in cache.meta["tables"] if info["name"] != "final"]
    final = cache.table("final", device)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    best, history = -math.inf, []
    settings = {key: value for key, value in vars(args).items() if isinstance(value, (str, float, int, bool))}
    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum, components = 0., torch.zeros(3, dtype=torch.float64)
        counts = torch.zeros(15, dtype=torch.long, device=device)
        for idx in torch.randperm(len(data["end"]), generator=batches).split(args.batch_size):
            h, valid, target = device_batch(data, idx, device)
            rows, positive = candidates(target, len(cache.pois), args.negatives, rng)
            rows = rows.to(device)
            prior, confidence = gather_fold_priors(tables, data["fold"][idx], rows)
            result = model(h, valid, rows, prior, confidence)
            loss, parts = objective(result, positive.to(device), args.lambda_surd, args.lambda_balance)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip, error_if_nonfinite=True)
            opt.step()
            counts += torch.bincount(result["top"].detach().reshape(-1), minlength=15)
            loss_sum += float(loss.detach()) * len(idx)
            components += torch.stack(parts).double().cpu() * len(idx)
        metrics = evaluate(model, final, val, cache, args.eval_batch_size, args.chunk)
        score = metrics["all_targets"]["NDCG@10"]
        record = dict(epoch=epoch, loss=loss_sum / len(data["end"]),
                      loss_terms=dict(zip(("rec", "surd", "balance"), (components / len(data["end"])).tolist())),
                      dispatch_counts=dict(zip(EXPERT_ORDER, counts.cpu().tolist())), validation=metrics)
        history.append(record)
        if score > best:
            best = score
            save_checkpoint(dict(cache_id=cache.id, config=asdict(cfg), weights=model.state_dict(),
                                 epoch=epoch, validation_ndcg10=score, training_args=settings), out / "best.pt")
        dump_json({"training_args": settings, "epochs": history}, out / "history.json")
        print(f"epoch={epoch} loss={record['loss']:.5f} validation NDCG@10={score:.6f}", flush=True)


def load_model(args):
    cache, device = Cache(args.cache), select_device(args.device)
    checkpoint = load_file(args.checkpoint)
    if cache.id != checkpoint["cache_id"]:
        raise ValueError("checkpoint trained with a different cache; check --cache and --checkpoint")
    model = cache.model(ModelConfig(**checkpoint["config"])).to(device)
    model.load_state_dict(checkpoint["weights"])
    return cache, model.eval(), cache.table("final", device)


def report(value, dest):
    if dest:
        Path(dest).parent.mkdir(parents=True, exist_ok=True)
        dump_json(value, dest)
        print("Wrote", dest)
    else:
        print(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))


def evaluate_cli(args):
    cache, model, table = load_model(args)
    training = cache.split("train")
    category_frequency = Counter(training["C"].tolist())
    metrics = evaluate(model, table, cache.split(args.split), cache, args.eval_batch_size,
                       args.chunk, category_frequency)
    report(dict(split=args.split, catalog_size=len(cache.pois), target_mode=cache.cfg["eval_targets"],
                cache_id=cache.id, **metrics), args.output)


@torch.inference_mode()
def explain(args):
    cache, model, table = load_model(args)
    data = cache.split(args.split)
    i = args.sample_index
    if not 0 <= i < len(data["end"]):
        raise ValueError("sample-index outside split")
    h, valid, target = device_batch(data, torch.tensor([i]), table.routing_prior.device)
    encoded = model.encode(h, valid)
    scores = all_scores(model, table, encoded, valid, len(cache.pois), args.chunk)
    if args.poi_id is None:
        row = int(target[0]) if target[0] >= 0 else int(scores.argmax())
    else:
        if args.poi_id not in table.row_of:
            raise ValueError("candidate POI outside the training catalog")
        row = table.row_of[args.poi_id]
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
                  candidate_rank=int(ranks_of(scores, torch.tensor([row]))[0]))
    report(record, args.output)


@torch.inference_mode()
def recommend(args):
    cache, model, table = load_model(args)
    groups, _ = read_split(args.history)
    cats = Categories.from_manifest(cache.meta["preprocessing"])
    data = build_split(groups, cats, cache.pois, cache.meta["cat_vocab"], cache.cfg["max_length"], "predict")
    records = []
    for idx in torch.arange(len(data["end"])).split(args.eval_batch_size):
        h, valid, _ = device_batch(data, idx, table.routing_prior.device, labels=False)
        scores = all_scores(model, table, model.encode(h, valid), valid, len(cache.pois), args.chunk)
        top = torch.argsort(scores, descending=True, stable=True)[:, :args.top_n]
        for i, js, row_scores in zip(idx, top.tolist(), scores):
            user, trail = data["trails"][data["trail"][data["end"][i] - 1]]
            records.append(dict(user_id=user, trail_id=trail, recommendations=[
                dict(venue_id=cache.pois[j], score=float(row_scores[j])) for j in js]))
    report(records, args.output)
