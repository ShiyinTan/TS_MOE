"""Trace a reference history through quantization and one candidate's SURD table."""
import math

import torch

from .data import make_batch
from .engine import Cache, report
from .prepare import verify_files
from .quantizer import BasicSourceQuantizer
from .surd import ATOM_ORDER, SOURCE_ORDER, SUBSETS, SUBSET_ORDER


def inspect_prior(args):
    cache = Cache(args.cache)
    if args.table not in {f["name"] for f in cache.meta["tables"]}:
        raise ValueError("unknown table; see tables[].name in manifest.json")
    stats = cache.load(f"{args.table}.surd.pt")
    table = cache.table(args.table, "cpu")
    if args.poi_id not in table.row_of:
        raise ValueError("venue_id is outside the training catalog")
    row = table.row_of[args.poi_id]
    meta = stats["manifest"]
    record = dict(venue_id=args.poi_id, table=args.table, reference=meta["split_manifest"],
                  observed=bool(stats["observed"][row]), target_count=int(stats["target_count"][row]),
                  near_tie=bool(stats["near_tie"][row]), total_information_bits=float(stats["total_information"][row]),
                  confidence=float(stats["confidence"][row]), specific_bits=dict(zip(
                      SUBSET_ORDER, stats["specific_information"][row].tolist())))
    for name in ("raw_surd", "normalized_surd", "routing_prior"):
        record[name] = dict(zip(ATOM_ORDER, stats[name][row].tolist()))
    if args.sample_index is not None:
        pair = meta["quantizer_manifest"]
        if pair is None:
            raise ValueError("the empty warm-up table has no fitted quantizer; select another table")
        data = cache.split("train")
        if not 0 <= args.sample_index < len(data["end"]):
            raise ValueError("training sample-index outside range")
        # Transforming a history is read-only. The selected event need not be in the reference set.
        verify_files(cache.path, {pair["file"]: pair["sha256"]})
        quantizer = BasicSourceQuantizer.load(cache.path / pair["file"])
        history, valid, target = make_batch(data, torch.tensor([args.sample_index]), offline=True)
        states = quantizer.transform(history, valid)[0]
        joint, count = stats["joint_states"], stats["joint_count"]
        ny, total = record["target_count"], int(stats["total_weight"])
        marginal = {}
        for subset, name in zip(SUBSETS, SUBSET_ORDER):
            cols = list(subset)
            match = (joint[:, [j + 1 for j in cols]] == states[cols]).all(1)
            nq = int(count[match].sum())
            nyq = int(count[match & (joint[:, 0] == row)].sum())
            marginal[name] = dict(count_qA=nq, count_y_qA=nyq,
                                  pointwise_bits=math.log2(nyq * total / (ny * nq)) if nyq else None)
        record["sample"] = dict(index=args.sample_index, true_next_venue_id=cache.pois[int(target)],
                                history_poi_ids=history["P"][0].tolist(), summary_dims=quantizer.config["summary_dims"],
                                q_state=dict(zip(SOURCE_ORDER, states.tolist())), counts_for_observed_state=marginal)
    report(record, args.output)
