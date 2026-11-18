"""Command-line entry point. Run python -m tsmoe COMMAND --help."""
import argparse
import math
import sys

import torch

from .engine import evaluate_cli, explain, recommend, train
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
    p = commands.add_parser("prepare", help="build samples, fold quantizers and per-POI SURD tables")
    p.add_argument("--data-root", required=True, help="official data directory or its city subdirectory")
    p.add_argument("--city", required=True, help="official city filename prefix, e.g. tokyo")
    p.add_argument("--out", required=True)
    for name, value in [("max-length", 50), ("clusters", 4), ("hash-dim", 64), ("folds", 3)]:
        p.add_argument("--" + name, type=positive, default=value)
    p.add_argument("--quant-seed", type=int, default=0)
    p.add_argument("--warmup", type=float, default=.2)
    p.add_argument("--tau-pos", type=float, default=10.)
    p.add_argument("--kappa-n", type=float, default=20.)
    p.add_argument("--kappa-i", type=float, default=.05)
    for name in ("train-targets", "eval-targets"):
        p.add_argument("--" + name, choices=("all", "last"), default="all")
    p.set_defaults(func=prepare)

    a = commands.add_parser("inspect", help="trace a training prefix, sparse joint counts and one POI's 26 atoms")
    a.add_argument("--cache", required=True)
    a.add_argument("--poi-id", type=int, required=True)
    a.add_argument("--sample-index", type=int, help="optional training prefix to transform with this table's quantizer")
    a.add_argument("--table", default="final")
    a.add_argument("--output")
    a.set_defaults(func=inspect_prior)

    t = commands.add_parser("train", help="train with fold-specific priors and select by validation NDCG@10")
    t.add_argument("--out", required=True)
    for name, value in [("epochs", 20), ("batch-size", 32), ("dim", 128), ("heads", 4),
                        ("layers", 2), ("top-k", 4)]:
        t.add_argument("--" + name, type=positive, default=value)
    for name, value in [("dropout", .1), ("temperature", 1.), ("lambda-prior", .1),
                        ("lambda-surd", .05), ("lambda-balance", .001), ("lr", .001),
                        ("weight-decay", .01), ("grad-clip", 1.)]:
        t.add_argument("--" + name, type=float, default=value)
    t.add_argument("--negatives", type=int, default=63,
                   help="uniform distinct negatives per positive; -1 selects full-catalog training")
    t.add_argument("--no-surd", action="store_true", help="context-only U/S MoE ablation: disable all statistical guidance")
    t.add_argument("--seed", type=int, default=42)
    t.set_defaults(func=train)

    e = commands.add_parser(
        "evaluate",
        help="exact ranking over every training POI; report Acc@1 / NDCG@1 / MRR "
             "(+ full @K tables and Head/Mid/Tail groups) to metrics.json + report.txt")
    e.set_defaults(func=evaluate_cli)
    e.add_argument("--out", help="directory for metrics.json and report.txt "
                   "(default: results/<run_from_checkpoint>/eval_<split>)")
    x = commands.add_parser("explain", help="audit one candidate's prior, router and score contributions")
    x.add_argument("--sample-index", type=int, default=0)
    x.add_argument("--poi-id", type=int, help="external venue_id; default is the true POI or top prediction for OOV")
    x.set_defaults(func=explain)
    r = commands.add_parser("recommend", help="rank next POIs for each trail in a new observed-history CSV")
    r.add_argument("--history", required=True)
    r.add_argument("--top-n", type=positive, default=10)
    r.set_defaults(func=recommend)
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
    if args.command == "prepare" and not (0 <= args.warmup < 1 and min(args.tau_pos, args.kappa_i, args.kappa_n) > 0):
        root.error("warmup must be in [0,1); tau-pos and reliability scales must be positive")
    if args.command == "train" and (args.negatives < -1 or args.negatives == 0 or
                                    min(args.lambda_prior, args.lambda_surd, args.lambda_balance, args.weight_decay) < 0
                                    or min(args.lr, args.grad_clip) <= 0):
        root.error("use --negatives -1 or a positive count, nonnegative strengths and positive lr/grad-clip")
    torch.set_num_threads(args.threads)
    try:
        args.func(args)
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr, flush=True)
        raise SystemExit(130) from None


if __name__ == "__main__":
    main()
