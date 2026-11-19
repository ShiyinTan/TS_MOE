"""Time-expanding cross-fitting and reproducible, versioned cache construction."""
import hashlib
import json
import math
import tempfile
from pathlib import Path

import torch
from tqdm.auto import tqdm

from .candidates import (
    DEFAULT_RADIUS, candidate_construction_record, catalog_coordinates,
    consecutive_visit_distances, pick_radius_km, summarize_distances,
)
from .data import SPLITS, Categories, build_split, file_digest, make_batch, read_split
from .quantizer import BasicSourceQuantizer, SOURCES
from .runtime import check_interrupt, install_interrupt_handler
from .surd import ATOM_ORDER, EXPERT_ORDER, SOURCE_ORDER, SUBSET_ORDER, build_offline_surd, save_surd_table


def digest_json(value):
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def dump_json(value, path):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def empty_stats(pois, meta):
    """An unobserved warm-up has no estimable entropy and zero guidance."""
    n = len(pois)
    zero = lambda *shape: torch.zeros(shape, dtype=torch.float64)
    return dict(schema_version="PCTL_TORCH_SURD_v2", source_order=list(SOURCE_ORDER),
                subset_order=list(SUBSET_ORDER), atom_order=list(ATOM_ORDER), expert_order=list(EXPERT_ORDER),
                poi_ids=pois, specific_information=zero(n, 15), raw_surd=zero(n, 26),
                normalized_surd=zero(n, 26), routing_prior=zero(n, 26), global_raw=zero(26),
                global_prior=zero(26), global_mi=zero(15), total_information=zero(n),
                confidence=zero(n), count_factor=zero(n), target_count=zero(n), target_probability=zero(n),
                observed=torch.zeros(n, dtype=torch.bool), has_information=torch.zeros(n, dtype=torch.bool),
                near_tie=torch.zeros(n, dtype=torch.bool), joint_states=torch.empty((0, 5), dtype=torch.long),
                joint_count=zero(0), total_weight=zero(), H_Y=None, H_cond=None, H_cond_via_mi=None,
                manifest=meta)


def reference_plan(train, folds, warmup):
    pred = train["us"][train["end"] - 1]
    target = train["us"][train["end"]]
    order = torch.argsort(pred, stable=True)
    n_warm = min(len(order) - 1, math.ceil(len(order) * warmup))
    blocks = ([order[:n_warm]] if n_warm else []) + list(order[n_warm:].tensor_split(folds))
    for block in blocks:
        if block.numel():
            cutoff = pred[block].min()
            ref = torch.where(target < cutoff)[0]
            yield block, ref, int(cutoff)


def event_digest(data, refs):
    acc = hashlib.sha256()
    for j in data["end"][refs].tolist():
        key = [*data["trails"][int(data["trail"][j])], int(data["us"][j]), int(data["P"][j])]
        acc.update(json.dumps(key, ensure_ascii=True, separators=(",", ":")).encode() + b"\n")
    return acc.hexdigest()


def prepare(args):
    install_interrupt_handler()
    root, city, out = Path(args.data_root), args.city, Path(args.out)
    if not (root / f"{city}_checkins_train.csv").is_file():
        root = root / city
    paths = {s: root / f"{city}_checkins_{s}.csv" for s in SPLITS}
    config = {k: getattr(args, k) for k in ("city", "max_length", "clusters", "hash_dim", "quant_seed",
              "folds", "warmup", "tau_pos", "train_targets", "eval_targets", "kappa_n", "kappa_i",
              "candidate_radius")}
    config.update(version="tsmoe.massivesteps.v1", files={s: file_digest(p) for s, p in paths.items()})
    fingerprint = digest_json(config)
    if out.exists():
        if (out / "manifest.json").exists():
            old = json.loads((out / "manifest.json").read_text())
            if old["fingerprint"] == fingerprint:
                verify_files(out, old["files"])
                print("Reusing verified cache:", out)
                return
        if any(out.iterdir()):
            raise ValueError(f"{out} contains a different cache. Select a new --out for this configuration.")
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=out.parent) as tmp:
        stage = Path(tmp) / "prepared"
        stage.mkdir()
        build_cache(config, paths, fingerprint, stage)
        stage.replace(out)
    print("Prepared:", out)


def verify_files(root, files):
    for name, expected in files.items():
        if file_digest(root / name) != expected:
            raise ValueError(f"cache checksum mismatch: {name}")


def build_cache(cfg, paths, fingerprint, stage):
    train_groups, train_audit = read_split(paths["train"])
    cats = Categories(train_groups)
    pois = sorted({e.poi for events in train_groups.values() for e in events if e.poi >= 0})
    cat_vocab = sorted({cats(e) for events in train_groups.values() for e in events if cats(e) >= 0})
    if not pois:
        raise ValueError("no observed training POIs")
    distance_stats = summarize_distances(consecutive_visit_distances(train_groups))
    radius_method = cfg.get("candidate_radius", DEFAULT_RADIUS)
    radius_km = pick_radius_km(distance_stats, radius_method)
    poi_coordinates, coordinate_imputation = catalog_coordinates(train_groups, pois)
    counts = coordinate_imputation["counts"]
    print(
        "coordinate_imputation: "
        + " ".join(f"{name}={counts[name]}" for name in coordinate_imputation["fallback_order"])
        + f" (imputed={coordinate_imputation['imputed']}/{coordinate_imputation['catalog_size']})",
        flush=True,
    )
    candidate_construction = candidate_construction_record(
        radius_method, distance_stats, radius_km, coordinate_imputation)
    preprocessing = dict(source_fields=["venue_id", "venue_category_id_code/venue_category_id",
                                        "timestamp", "latitude/longitude"],
                         time="UTC; naive timestamps interpreted as UTC; interval in hours",
                         location="WGS84 degrees; great-circle distance in km; bearing in radians",
                         tie_break="input CSV row within trail", dedup_key=["user_id", "trail_id", "timestamp", "venue_id"],
                         missing="quarantine invalid timestamps; missing historical IDs use UNK; missing target ID skipped",
                         categories="training code via stable category ID; ambiguous/missing mapping uses SHA-256 stable ID",
                         category_mapping=cats.codes, ambiguous_category_codes=sorted(cats.ambiguous),
                         max_length=cfg["max_length"], history="shared right-padding; first step/interval invalid",
                         candidate_construction=candidate_construction)
    audit, owners, held_trails = {}, {}, set()
    for split in SPLITS:
        groups, audit[split] = ((train_groups, train_audit) if split == "train" else read_split(paths[split]))
        if groups.keys() & held_trails:
            raise ValueError("trail shared by official split files; refusing overlapping histories")
        held_trails.update(groups)
        for user, trail in groups:
            if owners.setdefault(trail, user) != user:
                raise ValueError(f"trail {trail!r} has conflicting owners across splits")
        mode = cfg["train_targets"] if split == "train" else cfg["eval_targets"]
        data = build_split(groups, cats, pois, cat_vocab, cfg["max_length"], mode)
        rows = data["poi_token"][data["end"]] - 2
        audit[split].update(samples=len(rows), known_targets=int((rows >= 0).sum()),
                            unknown_targets=int((rows < 0).sum()), target_mode=mode)
        if split == "train":
            train = data
        else:
            torch.save(data, stage / f"{split}.pt")
    args_q = dict(n_clusters=cfg["clusters"], hash_dim=cfg["hash_dim"], seed=cfg["quant_seed"],
                  max_length=cfg["max_length"], tau_pos=cfg["tau_pos"])
    # These deterministic per-history summaries have no fitted parameters. Reuse across folds.
    ids = torch.arange(len(train["end"]))
    fixed = BasicSourceQuantizer(**args_q)
    pieces = {s: [] for s in SOURCES}
    summary_batches = list(ids.split(256))
    for batch in tqdm(summary_batches, desc="prepare summaries", dynamic_ncols=True):
        check_interrupt()
        history, valid, _ = make_batch(train, batch, offline=True)
        for s, matrix in fixed.summarize(history, valid).items():
            pieces[s].append(matrix)
    summaries = {s: torch.cat(xs) for s, xs in pieces.items()}
    del pieces
    target_rows = train["poi_token"][train["end"]] - 2
    folds = list(reference_plan(train, cfg["folds"], cfg["warmup"]))
    train["fold"] = torch.full_like(ids, -1)
    tables = []
    jobs = [(f"fold_{i}", block, ref, cutoff) for i, (block, ref, cutoff) in enumerate(folds)]
    jobs.append(("final", ids[:0], ids, None))
    for i, (name, block, ref, cutoff) in enumerate(tqdm(jobs, desc="prepare SURD tables", dynamic_ncols=True)):
        check_interrupt()
        if len(block):
            train["fold"][block] = i
        targets = train["us"][train["end"][ref]]
        split_info = dict(name=name, scored=len(block), reference_events=len(ref), prediction_cutoff_utc_us=cutoff,
                          first_reference_target_utc_us=int(targets.min()) if len(ref) else None,
                          last_reference_target_utc_us=int(targets.max()) if len(ref) else None,
                          comparison="strict target timestamp < earliest scoring prediction timestamp" if cutoff is not None
                                     else "all official training reference events",
                          reference_event_sha256=event_digest(train, ref))
        meta = dict(cache_id=fingerprint, feature_method="basic_fixed_features_v1",
                    source_fields=preprocessing["source_fields"], preprocessing_sha256=digest_json(preprocessing),
                    split_manifest=split_info, quantizer_manifest=None)
        if len(ref):
            fit_summaries = {s: matrix[ref] for s, matrix in summaries.items()}
            q = BasicSourceQuantizer(**args_q).fit_summaries(fit_summaries)
            state = q.transform_summaries(fit_summaries)
            q_file = f"{name}.quantizer.pt"
            q.save(stage / q_file)
            meta["quantizer_manifest"] = dict(file=q_file, sha256=file_digest(stage / q_file), config=q.config)
            stats = build_offline_surd(state, target_rows[ref], pois, manifest=meta,
                                       kappa_n=cfg["kappa_n"], kappa_i=cfg["kappa_i"])
            split_info["cluster_sizes"] = {s: state[:, j].bincount().tolist() for j, s in enumerate(SOURCES)}
            split_info["occupied_four_source_states"] = int(state.unique(dim=0).shape[0])
        else:
            stats = empty_stats(pois, meta)
        stats["manifest"]["split_manifest"] = split_info
        save_surd_table(stats, stage / f"{name}.surd.pt")
        tables.append(split_info)
        print(f"{name}: {len(ref)} reference targets; {len(block)} scored", flush=True)
    assert (train["fold"] >= 0).all()
    torch.save(train, stage / "train.pt")
    print(f"candidate_radius={radius_method} radius_km={radius_km} "
          f"consecutive_pairs={distance_stats['count']}", flush=True)
    manifest = dict(fingerprint=fingerprint, config=cfg, poi_ids=pois, cat_vocab=cat_vocab,
                    poi_coordinates=poi_coordinates,
                    candidate_construction=candidate_construction,
                    preprocessing=preprocessing, audit=audit, tables=tables,
                    torch_version=str(torch.__version__),
                    files={f.name: file_digest(f) for f in sorted(stage.glob("*.pt"))})
    dump_json(manifest, stage / "manifest.json")
