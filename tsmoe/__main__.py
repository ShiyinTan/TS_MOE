"""Command-line entry point. Run python -m tsmoe COMMAND --help."""
import argparse
import math
import sys

import torch

from .ablation_summary import add_ablation_summary_parser
from .counts import DEFAULT_VISIT_BINS, TRANSITION_ATTN_MODES
from .engine import evaluate_cli, explain, recommend, train
from .model import (
    CANDIDATE_CONTEXT_MODES, CANDIDATE_CONTEXT_TIME_MODES, EXPERT_CROSS_ATTN_MODES, ON_OFF,
    SURD_BIAS_GATE_MODES, SURD_GATE_APPLY_MODES,
)
from .prepare import prepare
from .inspect import inspect_prior


def positive(value):
    n = int(value)
    if n < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return n


def main():
    root = argparse.ArgumentParser(description="TS-MoE for Massive-STEPS check-in CSVs")
    commands = root.add_subparsers(dest="command", required=True)
    p = commands.add_parser(
        "prepare",
        help="build samples and one train-all per-POI SURD table shared by train/val/test")
    p.add_argument("--data-root", required=True, help="official data directory or its city subdirectory")
    p.add_argument("--city", required=True, help="official city filename prefix, e.g. tokyo")
    p.add_argument("--out", required=True)
    p.add_argument(
        "--no-overwrite", action="store_true",
        help="refuse to replace an existing --out when the fingerprint differs "
             "or the cache is incomplete (default: rebuild and replace in place)")
    for name, value in [("max-length", 50), ("clusters", 4), ("hash-dim", 64)]:
        p.add_argument("--" + name, type=positive, default=value)
    p.add_argument("--quant-seed", type=int, default=0)
    p.add_argument("--tau-pos", type=float, default=10.)
    p.add_argument("--kappa-n", type=float, default=20.)
    p.add_argument("--kappa-i", type=float, default=.05)
    for name in ("train-targets", "eval-targets"):
        p.add_argument("--" + name, choices=("all", "last"), default="all")
    p.add_argument(
        "--filter-unknown", choices=("off", "on"), default="off",
        help="STHGCN-style filter (default off): when on, drop val/test trails whose "
             "user never appears in train, and drop prediction samples whose true "
             "next POI is outside the train catalog. Unlike eval known_only (metric "
             "denominator only), this removes those samples from the prepared data. "
             "Requires re-prepare; fingerprinted.")
    p.add_argument(
        "--candidate-radius",
        choices=("p70", "p80", "p90", "mean", "max", "full", "random"), default="p90",
        help="local spatial neighborhood radius from consecutive visit distances "
             "(default p90; also p70/p80/mean/max); full = global catalog only "
             "(disables popular / user-frequent / random-negative sources); "
             "random = random-only pool (GT + --random-negatives; no spatial / "
             "popular / user-frequent; requires --random-negatives >= 1). "
             "Stored in the cache so train and evaluate share the same construction.")
    p.add_argument(
        "--popular-radius",
        choices=("max", "p90", "p80", "p70", "mean", "multiplier"), default="max",
        help="larger radius for popular-POI source (default max of consecutive "
             "distances; also p90/p80/p70/mean); multiplier = local radius × "
             "--popular-radius-multiplier")
    p.add_argument(
        "--popular-radius-multiplier", type=float, default=2.0,
        help="scale for --popular-radius multiplier (default 2.0)")
    p.add_argument(
        "--popular-top-k", type=int, default=100,
        help="top-K popular catalog POIs inside the larger radius (default 100; "
             "0 disables the popular source)")
    p.add_argument(
        "--popular-by", choices=("visit", "target"), default="visit",
        help="popularity rank key: train check-in visits or next-POI targets "
             "(default visit)")
    p.add_argument(
        "--user-frequent-top-k", type=int, default=50,
        help="per-user top-K training venues for the user-frequent source "
             "(default 50; 0 disables)")
    p.add_argument(
        "--user-frequent-min-visits", type=positive, default=2,
        help="minimum training visits for a venue to count as user-frequent "
             "(default 2)")
    p.add_argument(
        "--random-negatives", type=int, default=50,
        help="extra uniform random catalog negatives added to each constructed "
             "pool (default 50; 0 disables). Under --candidate-radius random this "
             "is the pool size excluding GT (requires >= 1). Seeded and "
             "fingerprinted; ignored for --candidate-radius full / empty-union "
             "fallback")
    p.add_argument(
        "--random-negatives-seed", type=int, default=0,
        help="seed mixed with sample index for deterministic random negatives "
             "(default 0)")
    p.set_defaults(func=prepare)

    a = commands.add_parser("inspect", help="trace a training prefix, sparse joint counts and one POI's 26 atoms")
    a.add_argument("--cache", required=True)
    a.add_argument("--poi-id", type=int, required=True)
    a.add_argument("--sample-index", type=int, help="optional training prefix to transform with this table's quantizer")
    a.add_argument("--table", default="final")
    a.add_argument("--output")
    a.set_defaults(func=inspect_prior)

    t = commands.add_parser(
        "train",
        help="train with the shared train-all SURD table; checkpoint by "
             "validation NDCG@10 (default) or Acc@1 via --checkpoint-metric")
    t.add_argument("--out", required=True)
    for name, value in [("epochs", 20), ("batch-size", 32), ("dim", 128), ("heads", 4),
                        ("layers", 2), ("top-k", 4)]:
        t.add_argument("--" + name, type=positive, default=value)
    for name, value in [("dropout", .1), ("temperature", 1.), ("lambda-prior", .1),
                        ("hybrid-lambda", .1), ("lambda-surd", .05), ("lambda-balance", .001),
                        ("lr", .001), ("weight-decay", .01), ("grad-clip", 1.)]:
        t.add_argument("--" + name, type=float, default=value)
    t.add_argument(
        "--lambda-bpr", type=float, default=0.,
        help="weight on hard-negative BPR aux softplus(-(s_pos-max_neg-margin)) "
             "(default 0 = off). Acc@1 hillclimbs may try 0.1–1.0")
    t.add_argument(
        "--bpr-margin", type=float, default=0.,
        help="logit margin inside hard-neg BPR (default 0); only used when "
             "--lambda-bpr > 0. Try 0.5–2.0 to force a clearer Acc@1 gap")
    t.add_argument(
        "--router-mode", choices=("hard_prior", "surd_mlp", "hybrid"), default="hard_prior",
        help="expert routing: hard_prior (default) = π from candidate U/S SURD "
             "mass only (no sequence in router); surd_mlp = softmax(MLP(SURD)/T); "
             "hybrid = λ_prior·surd_bias + hybrid_lambda·learned(seq×cand×SURD). "
             "Stored in the checkpoint so evaluate/explain/recommend match train.")
    t.add_argument("--negatives", type=int, default=63,
                   help="distinct negatives per positive drawn from the cache "
                        "candidate pool (multi-source by default; full catalog "
                        "when prepare used --candidate-radius full); "
                        "-1 uses every other pool member (still subject to "
                        "--max-train-candidates if set)")
    t.add_argument(
        "--max-train-candidates", type=int, default=-1,
        help="cap total training candidates per sample including the positive "
             "(default -1 = no extra cap beyond --negatives); evaluation always "
             "uses the full constructed pool")
    t.add_argument("--no-surd", action="store_true", help="context-only U/S MoE ablation: disable all statistical guidance")
    t.add_argument(
        "--causal-counts", dest="no_causal_counts", action="store_false",
        help="enable causal online visit/transition count tables + transition "
             "cross-attn (default: on). Train is chronological (no shuffle); "
             "val/test start from train-end tables. See docs/causal-counts.md")
    t.add_argument(
        "--no-causal-counts", dest="no_causal_counts", action="store_true",
        help="disable the causal count module (restore shuffled minibatches)")
    t.set_defaults(no_causal_counts=False)
    t.add_argument(
        "--transition-attn", choices=TRANSITION_ATTN_MODES, default="bias",
        help="how global transition counts enter P/C cross-attn: "
             "bias (default) = λ·log1p(count) on logits; "
             "gate = sigmoid MLP gate; rel_key = MLP(φ) added to keys. "
             "P-source uses POI→POI; C-source uses category→POI")
    t.add_argument(
        "--transition-lambda", type=float, default=1.0,
        help="scale on log1p(transition) bias terms (default 1.0; bias mode)")
    t.add_argument(
        "--transition-score-bias", type=float, default=0.0,
        help="add λ·log1p(last→cand POI transition) to ranking scores "
             "(default 0 = off). Light graph Acc@1 prior; needs causal counts")
    t.add_argument(
        "--category-score-bias", type=float, default=0.0,
        help="add λ·log1p(last_cat→cand_poi) via causal cat_transition to scores "
             "(default 0 = off). Hypergraph-style Acc@1 prior; needs causal counts. "
             "Shares --tsb-gate-mode with transition-score-bias when SURD gate is on")
    t.add_argument(
        "--pop-score-bias", type=float, default=0.0,
        help="add λ·log1p(global POI visits) from causal cand feats to scores "
             "(default 0 = off). Popularity Acc@1 prior; needs causal counts")
    t.add_argument(
        "--geo-score-bias", type=float, default=0.0,
        help="subtract λ·log1p(haversine km^α) from scores so nearer POIs rank higher "
             "(default 0 = off). Needs --candidate-context on (delta_geo). No re-prepare")
    t.add_argument(
        "--geo-score-power", type=float, default=1.0,
        help="gravity exponent α on geo-score-bias: −λ·log1p(km^α) "
             "(default 1 = classic log1p km; try 0.5–2). No-op when geo-score-bias=0")
    t.add_argument(
        "--gauss-geo-bias", type=float, default=0.0,
        help="add λ·exp(−km²/(2σ²)) proximity kernel to scores (default 0 = off). "
             "Prefer replacing --geo-score-bias. Needs --candidate-context on")
    t.add_argument(
        "--gauss-sigma-km", type=float, default=0.5,
        help="σ in km for --gauss-geo-bias (default 0.5; sweep 0.25–2)")
    t.add_argument(
        "--geo-adapt", choices=("off", "sparse"), default="off",
        help="modulate geo/gauss by last→cand transition mass: sparse = more "
             "spatial when tsb is thin (default off)")
    t.add_argument(
        "--geo-adapt-temp", type=float, default=1.0,
        help="strength α for --geo-adapt sparse: scale=1/(1+α·log1p(count))")
    t.add_argument(
        "--revisit-score-bias", type=float, default=0.0,
        help="add λ·log1p(user→cand visits) to scores (default 0 = off). "
             "Needs causal counts; Acc@1 revisit prior")
    t.add_argument(
        "--tsb-decay-hours", type=float, default=0.0,
        help="if >0, multiply --transition-score-bias by exp(−Δt/τ) with Δt = "
             "hours from last history tip to target (Hawkes-lite; 0 = off)")
    t.add_argument(
        "--surd-bias-gate", choices=SURD_BIAS_GATE_MODES, default="off",
        help="per-candidate gate from SURD / Specific MI controlling score/attn "
             "biases: off (default); unique_p; smi_p; smi_pc; u_mass; confidence. "
             "See docs/causal-counts.md. No re-prepare.")
    t.add_argument(
        "--surd-score-bias", type=float, default=0.0,
        help="add λ·gate to ranking scores so SURD/SMI contributes directly "
             "(default 0 = off). Requires --surd-bias-gate ≠ off")
    t.add_argument(
        "--tsb-gate-mode", choices=SURD_GATE_APPLY_MODES, default="off",
        help="how --surd-bias-gate modulates transition-score-bias and "
             "category-score-bias: off (default, flat λ); mul = term·g; "
             "residual = term·(1+g). Needs gate ≠ off")
    t.add_argument(
        "--surd-attn-gate", choices=SURD_GATE_APPLY_MODES, default="off",
        help="how --surd-bias-gate modulates transition-attn λ (bias mode only): "
             "off (default); mul; residual. Needs gate ≠ off + causal counts")
    t.add_argument(
        "--visit-bins", type=positive, default=DEFAULT_VISIT_BINS,
        help=f"equal-frequency bins of user cumulative visit count for T/L "
             f"(default {DEFAULT_VISIT_BINS}; fit on train stream, freeze for val/test)")
    t.add_argument(
        "--candidate-context", choices=CANDIDATE_CONTEXT_MODES, default="off",
        help="fuse candidate category emb + Δgeo (haversine/east/north log1p) into "
             "the candidate query before shared cross-attn (default off). "
             "Does not enter hard_prior router logits. See docs/candidate-context.md")
    t.add_argument(
        "--candidate-context-time", choices=CANDIDATE_CONTEXT_TIME_MODES, default="off",
        help="optional time side of candidate-context: off (default); last_slot = "
             "last hour/weekday one-hot; causal_slot = + causal same-hour POI freq "
             "(requires causal counts). Never enters hard_prior")
    t.add_argument(
        "--session-summary", choices=ON_OFF, default="off",
        help="session/intent summary (window cat hist, traj-length bin, "
             "activity-center→last distance) added to shared base/context "
             "(default off; never hard_prior)")
    t.add_argument(
        "--user-pref", choices=ON_OFF, default="off",
        help="causal user–cat / user–hour preference MLP into base/context "
             "(default off; requires causal counts; never hard_prior)")
    t.add_argument(
        "--user-id-emb", choices=ON_OFF, default="off",
        help="learned Embedding of train user_id into base/context "
             "(default off; UNK for users unseen in train; never hard_prior). "
             "Distinct from --user-pref (count prefs). No re-prepare.")
    t.add_argument(
        "--transition-pmi", choices=ON_OFF, default="off",
        help="use POI→POI / cat→POI PMI instead of raw counts in transition "
             "cross-attn (default off; requires causal counts). Text embeddings deferred")
    t.add_argument(
        "--expert-cross-attn", choices=EXPERT_CROSS_ATTN_MODES, default="off",
        help="expert-local cross-attn: off (default, shared only); hybrid (E1) = "
             "keep shared + typed expert MHA (transition stays on shared P/C); "
             "full (E2) = drop shared MoE path for expert-local only")
    t.add_argument(
        "--expert-cross-attn-sparse", choices=ON_OFF, default="on",
        help="when expert-cross-attn ≠ off: only run expert-local attn for sources "
             "in router top-k experts (E3; default on)")
    t.add_argument("--seed", type=int, default=42)
    t.add_argument(
        "--checkpoint-metric", choices=("ndcg10", "acc1"), default="ndcg10",
        help="validation metric used to select best.pt: ndcg10 (default, "
             "all_targets NDCG@10) or acc1 (all_targets Acc@1). Experimental "
             "Acc@1 hillclimbs may use acc1; default keeps historical protocol.")
    t.add_argument(
        "--eval-test-every-epoch",
        action=argparse.BooleanOptionalAction, default=True,
        help="after each training epoch, also evaluate on the test split and write "
             "results/<run>/eval_test/epoch_*.json + results/<run>/history.json "
             "(default: on). Checkpoint selection uses --checkpoint-metric "
             "(default validation NDCG@10); final best-ckpt test still runs. "
             "Use --no-eval-test-every-epoch to skip per-epoch test for speed.")
    t.set_defaults(func=train)

    e = commands.add_parser(
        "evaluate",
        help="rank within the full constructed cache candidate pool "
             "(multi-source neighborhood+popular+user-frequent+random by "
             "default; or --candidate-radius random / full; in-catalog GT "
             "always present); report Acc@1 / NDCG@1 / MRR (+ full @K tables "
             "and Head/Mid/Tail groups) to metrics.json + report.txt")
    e.set_defaults(func=evaluate_cli)
    e.add_argument("--out", help="directory for metrics.json and report.txt "
                   "(default: results/<run_from_checkpoint>/eval_<split>)")
    e.add_argument(
        "--score-temperature", type=float, default=1.0,
        help="divide ranking logits by T before Acc@1 (eval-time calibration; "
             "default 1 = unchanged). Sweep on val, apply on test")
    x = commands.add_parser("explain", help="audit one candidate's prior, router and score contributions")
    x.add_argument("--sample-index", type=int, default=0)
    x.add_argument("--poi-id", type=int, help="external venue_id; default is the true POI or top prediction for OOV")
    x.set_defaults(func=explain)
    r = commands.add_parser("recommend", help="rank next POIs for each trail in a new observed-history CSV")
    r.add_argument("--history", required=True)
    r.add_argument("--top-n", type=positive, default=10)
    r.set_defaults(func=recommend)
    add_ablation_summary_parser(commands)
    for parser in (e, x):
        parser.add_argument("--split", choices=("validation", "test"), default="test")
    for parser in (e, x, r):
        parser.add_argument("--checkpoint", required=True)
        parser.add_argument("--output", help="optional extra JSON path (evaluate also writes --out/metrics.json)")
    for parser in (t, e, x, r):
        parser.add_argument("--cache", required=True)
        parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, or mps")
        parser.add_argument("--chunk", type=positive, default=128, help="catalog candidates per evaluation chunk")
        if parser != x:
            parser.add_argument("--eval-batch-size", type=positive, default=16)
    for parser in (p, a, t, e, x, r):
        parser.add_argument("--threads", type=positive, default=2, help="PyTorch CPU threads")
    args = root.parse_args()
    if any(not math.isfinite(v) for v in vars(args).values() if isinstance(v, float)):
        root.error("numeric options must be finite")
    if args.command == "prepare" and min(args.tau_pos, args.kappa_i, args.kappa_n) <= 0:
        root.error("tau-pos and reliability scales must be positive")
    if args.command == "prepare" and (
            args.popular_top_k < 0 or args.user_frequent_top_k < 0
            or args.random_negatives < 0
            or not math.isfinite(args.popular_radius_multiplier)
            or args.popular_radius_multiplier <= 0):
        root.error("popular-top-k / user-frequent-top-k / random-negatives must be >= 0; "
                   "popular-radius-multiplier must be positive and finite")
    if args.command == "prepare" and args.candidate_radius == "random" and args.random_negatives < 1:
        root.error("--candidate-radius random requires --random-negatives >= 1 "
                   "(random-only pool = GT + uniform catalog draws)")
    if args.command == "train" and (args.negatives < -1 or args.negatives == 0 or
                                    args.max_train_candidates == 0 or args.max_train_candidates < -1 or
                                    min(args.lambda_prior, args.hybrid_lambda, args.lambda_surd,
                                        args.lambda_balance, args.weight_decay) < 0
                                    or min(args.lr, args.grad_clip) <= 0
                                    or getattr(args, "transition_lambda", 1.0) < 0
                                    or getattr(args, "transition_score_bias", 0.0) < 0
                                    or getattr(args, "category_score_bias", 0.0) < 0
                                    or getattr(args, "pop_score_bias", 0.0) < 0
                                    or getattr(args, "geo_score_bias", 0.0) < 0
                                    or getattr(args, "surd_score_bias", 0.0) < 0
                                    or getattr(args, "revisit_score_bias", 0.0) < 0
                                    or getattr(args, "tsb_decay_hours", 0.0) < 0
                                    or getattr(args, "geo_score_power", 1.0) <= 0):
        root.error("use --negatives -1 or a positive count; --max-train-candidates -1 or >=1; "
                   "nonnegative strengths and positive lr/grad-clip; "
                   "transition-lambda / transition-score-bias / category-score-bias / "
                   "pop-score-bias / geo-score-bias / surd-score-bias / "
                   "revisit-score-bias / tsb-decay-hours must be nonnegative; "
                   "geo-score-power must be positive")
    if args.command == "evaluate" and float(getattr(args, "score_temperature", 1.0) or 1.0) <= 0:
        root.error("--score-temperature must be positive")
    if (args.command == "train"
            and float(getattr(args, "geo_score_bias", 0.0) or 0.0) > 0
            and getattr(args, "candidate_context", "off") == "off"):
        root.error("--geo-score-bias > 0 requires --candidate-context on")
    if args.command == "train":
        gate = getattr(args, "surd_bias_gate", "off")
        needs_gate = (
            float(getattr(args, "surd_score_bias", 0.0) or 0.0) > 0
            or getattr(args, "tsb_gate_mode", "off") != "off"
            or getattr(args, "surd_attn_gate", "off") != "off")
        if needs_gate and gate == "off":
            root.error("--surd-score-bias / --tsb-gate-mode / --surd-attn-gate "
                       "require --surd-bias-gate ≠ off")
        if gate != "off" and getattr(args, "no_surd", False):
            root.error("--surd-bias-gate requires SURD (omit --no-surd)")
    if hasattr(args, "threads"):
        torch.set_num_threads(args.threads)
    try:
        args.func(args)
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr, flush=True)
        raise SystemExit(130) from None


if __name__ == "__main__":
    main()
