"""Command-line entry point. Run python -m tsmoe COMMAND --help."""
import argparse
import math
import sys

import torch

from .ablation_summary import add_ablation_summary_parser
from .config import expand_config_argv
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


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    argv = expand_config_argv(argv)
    root = argparse.ArgumentParser(
        description="TS-MoE for Massive-STEPS check-in CSVs",
        epilog="Recipes: python -m tsmoe train --config configs/<city>/<base|win|peak>.yaml "
               "[--device cuda]. Config flags apply first; later CLI flags override. "
               "See docs/runbook.md.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
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
             "validation NDCG@10 (default) or Acc@1 via --checkpoint-metric. "
             "Prefer --config configs/<city>/<recipe>.yaml for Base/WIN/Peak.")
    t.add_argument(
        "--config",
        help="YAML/JSON recipe (expanded before other flags; CLI overrides). "
             "See configs/ and docs/runbook.md")
    t.add_argument("--out", required=True,
                   help="run directory under runs/ (also set via --config)")
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
        "--bpr-mode", choices=("max", "bpr_max", "simce"), default="max",
        help="hard-neg aux form when --lambda-bpr > 0: max (historical "
             "softplus vs max-neg), bpr_max (GRU4Rec+ softmax-weighted negs), "
             "simce (binary CE on [s_pos, max_neg]). Do not pair with focal/CE-τ")
    t.add_argument(
        "--ce-focal-gamma", type=float, default=0.,
        help="focal CE γ on rec term: (1−p_t)^γ · (−log p_t); 0 = plain CE. "
             "Try 1–2 for Acc@1 hard-miss emphasis (do not pair with CE-τ)")
    t.add_argument(
        "--soft-target-beta", type=float, default=0.,
        help="mix β of Acc-positive prior into CE targets: "
             "(1−β)·onehot + β·normalize(priors). 0 = hard labels. Try 0.05–0.1")
    t.add_argument(
        "--soft-target-prior", type=str, default="off",
        help="comma list of soft-target priors: tsb,revisit,ema; "
             "off disables. Requires --soft-target-beta > 0. "
             "tsb|revisit need causal counts; ema uses EMA teacher softmax (#39)")
    t.add_argument(
        "--ema-decay", type=float, default=0.999,
        help="EMA teacher decay θ'←d·θ'+(1−d)·θ when soft-target-prior includes "
             "ema (default 0.999; try 0.99 for short ep1)")
    t.add_argument(
        "--kd-teacher", type=str, default=None,
        help="frozen teacher checkpoint for top-K KD (#41); requires --lambda-kd > 0")
    t.add_argument(
        "--lambda-kd", type=float, default=0.,
        help="weight on top-K KD KL term vs teacher logits (default 0 = off)")
    t.add_argument(
        "--kd-top-k", type=int, default=20,
        help="teacher top-K positions for KD soft labels (default 20)")
    t.add_argument(
        "--kd-temp", type=float, default=2.0,
        help="KD softmax temperature (default 2); loss scaled by T²")
    t.add_argument(
        "--poi-geo-laplace", type=float, default=0.,
        help="λ for geo kNN Laplacian POI-emb regularizer (#40); 0 = off")
    t.add_argument(
        "--geo-knn", type=int, default=8,
        help="kNN neighbors per POI for --poi-geo-laplace (default 8)")
    t.add_argument(
        "--lambda-cat-aux", type=float, default=0.,
        help="η for next-category MTL aux CE on shared tip encoding (#45); "
             "0 = off. Try 0.1–0.3 on soft10r; inference still POI-only")
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
        "--neg-sample", choices=("uniform", "adaptive_mix"), default="uniform",
        help="negative sampler (#37): uniform (default) or adaptive_mix = "
             "top model-scored hard negs ⊕ random pool negs (GenNi-style)")
    t.add_argument(
        "--neg-hard-frac", type=float, default=0.25,
        help="fraction of --negatives taken as hard (top probe scores) when "
             "--neg-sample adaptive_mix (default 0.25)")
    t.add_argument(
        "--neg-probe-mult", type=float, default=4.0,
        help="probe size ≈ max(negatives, negatives×mult) for adaptive_mix "
             "hard mining (default 4)")
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
        "--history-hit-bias", type=float, default=0.0,
        help="add λ·1[cand∈last-K history tips] (default 0 = off). "
             "Orthogonal to --revisit-score-bias; keep revisit locked at 0.10")
    t.add_argument(
        "--history-hit-k", type=int, default=5,
        help="K for --history-hit-bias last-K tips (default 5)")
    t.add_argument(
        "--revisit-recency-bias", type=float, default=0.0,
        help="add λ·exp(−age/τ) for most recent hist visit of cand "
             "(default 0 = off). Orthogonal to flat revisit λ")
    t.add_argument(
        "--revisit-recency-tau", type=float, default=3.0,
        help="τ in history-steps for --revisit-recency-bias (default 3)")
    t.add_argument(
        "--radiation-score-bias", type=float, default=0.0,
        help="add λ·Simini radiation log-score over the candidate pool "
             "(default 0 = off). Prefer replacing --geo-score-bias. "
             "Needs --candidate-context on + causal counts (visit mass)")
    t.add_argument(
        "--radiation-kernel", choices=("hard", "soft"), default="hard",
        help="intervening mass: hard = strict closer disk among cands; "
             "soft = sigmoid-smoothed (default hard)")
    t.add_argument(
        "--radiation-soft-km", type=float, default=0.25,
        help="soft-kernel bandwidth in km for --radiation-kernel soft")
    t.add_argument(
        "--jump-length-bias", type=float, default=0.0,
        help="add λ·log P(Δr) jump-length prior to scores (default 0 = off). "
             "Prefer replacing --geo-score-bias. Needs --candidate-context on")
    t.add_argument(
        "--jump-dist", choices=("lognormal", "trunc_power"), default="lognormal",
        help="parametric form for --jump-length-bias (default lognormal)")
    t.add_argument(
        "--jump-mu", type=float, default=None,
        help="lognormal μ on log(km); default = fit from train hops when bias>0")
    t.add_argument(
        "--jump-sigma", type=float, default=None,
        help="lognormal σ; default = fit from train hops when bias>0 (else 1.0)")
    t.add_argument(
        "--jump-beta", type=float, default=1.5,
        help="truncated power-law exponent β for --jump-dist trunc_power")
    t.add_argument(
        "--jump-d0", type=float, default=0.1,
        help="truncated power-law offset d0 (km) for --jump-dist trunc_power")
    t.add_argument(
        "--return-gate", choices=("off", "on"), default="off",
        help="mix revisit vs explore spatial by predicted P(return): "
             "π·(λ_ret·revisit)+(1−π)·(λ_exp·explore). Keep --revisit-score-bias; "
             "do not raise flat revisit λ")
    t.add_argument(
        "--explore-spatial-bias", choices=("off", "gauss", "geo", "radiation"),
        default="off",
        help="explore branch spatial form for --return-gate on (default off)")
    t.add_argument(
        "--explore-score-bias", type=float, default=0.0,
        help="λ_exp for return-gate explore branch (default 0). "
             "gauss uses --gauss-sigma-km; radiation uses --radiation-kernel")
    t.add_argument(
        "--fam-area-blend", choices=("off", "on"), default="off",
        help="familiar/unfamiliar area soft blend (#44): "
             "(1−f)·λ_exp·gauss + f·λ_ret·revisit; keeps locked revisit λ")
    t.add_argument(
        "--fam-radius-km", type=float, default=0.5,
        help="local area radius (km) for fam_area_blend familiarity (default 0.5)")
    t.add_argument(
        "--fam-explore-bias", type=float, default=0.0,
        help="λ_exp on gauss for unfamiliar area (1−f); keep ≪ geo 0.15")
    t.add_argument(
        "--fam-return-bias", type=float, default=0.0,
        help="extra λ_ret·revisit mass scaled by familiarity f (default 0)")
    t.add_argument(
        "--tsb-history-k", type=int, default=1,
        help="blend last K history→cand transitions into --transition-score-bias "
             "(default 1 = last-only; try 2–3)")
    t.add_argument(
        "--tsb-history-decay", type=float, default=0.5,
        help="κ for multi-hop tsb weights w_o∝exp(−o/κ); o=0 is last tip (default 0.5)")
    t.add_argument(
        "--transition-dt-bins", type=str, default="off",
        help="Δt-bin edges in hours for transition-score-bias, e.g. 1h,6h,24h "
             "(bins <1 / 1–6 / 6–24 / >24). Uses bin counts with pooled fallback; "
             "off = last-only/multi-hop pooled. Prefer over --tsb-decay-hours")
    t.add_argument(
        "--cat-stage-bias", type=float, default=0.0,
        help="add λ·log1p(last_cat→cand_cat) soft category-stage prior to scores "
             "(default 0 = off). Needs --candidate-context on + causal counts. "
             "Soft only (no hard top-C filter)")
    t.add_argument(
        "--transition-2hop-bias", type=float, default=0.0,
        help="add λ·log1p((A²)_{last,cand}) 2-hop transition prior (#4); "
             "A from train consecutive hops (default 0 = off). Try 0.3–0.5")
    t.add_argument(
        "--pmi-score-bias", type=float, default=0.0,
        help="add λ·PMI(last_poi→cand) to ranking scores (default 0 = off). "
             "Keeps raw-count tsb/attn; distinct from --transition-pmi. Try 0.1–0.5")
    t.add_argument(
        "--hour-pop-score-bias", type=float, default=0.0,
        help="add λ·log1p(global_poi_hour[cand, tip_hour]) (#8); "
             "needs causal counts (default 0 = off). Try 0.1–0.2")
    t.add_argument(
        "--tsb-decay-hours", type=float, default=0.0,
        help="if >0, multiply --transition-score-bias by exp(−Δt/τ) with Δt = "
             "hours from last history tip to target (Hawkes-lite; 0 = off)")
    t.add_argument(
        "--tsb-session-gap", type=float, default=0.0, dest="tsb_session_gap_hours",
        help="PRME-style gate: keep full tsb only when Δt(last→now) ≤ τ hours; "
             "else zero tsb (default 0 = off). Sweep {6,12,24}; not continuous decay")
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
             "Does not enter hard_prior router logits. See docs/candidates.md")
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
        "--large-city-fast", choices=ON_OFF, default="off",
        help="when on *and* city is Bandung-or-larger: shrink model structure "
             "(dim128/heads4, transition-attn=bias, no last_slot/session) and "
             "validation candidate pool C via geo top-M=2048. Final test keeps "
             "full constructed pool. No-op on small cities. See tsmoe/city_scale.py")
    t.add_argument(
        "--candidate-spatial-prior", type=str, default="off",
        choices=("off", "geo", "gravity", "radiation"),
        help="train validation pool shrink (#10): keep top-M by geo/gravity/"
             "radiation (GT always included). large_city_fast defaults to geo "
             "on Bandung-or-larger; final test keeps full pool")
    t.add_argument(
        "--candidate-prior-top-m", type=int, default=0,
        help="top-M for --candidate-spatial-prior (0 = off). "
             "large_city_fast default on large cities: 2048 for val only")
    t.add_argument(
        "--candidate-prior-gravity-beta", type=float, default=1.0,
        help="β in gravity prior log1p(pop)−β·log1p(km) (default 1)")
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
             "and Head/Mid/Tail groups) to metrics.json + report.txt. "
             "May use --config for cache/device/shared knobs.")
    e.set_defaults(func=evaluate_cli)
    e.add_argument(
        "--config",
        help="YAML/JSON recipe (expanded before other flags; CLI overrides)")
    e.add_argument("--out", help="directory for metrics.json and report.txt "
                   "(default: results/<run_from_checkpoint>/eval_<split>)")
    e.add_argument(
        "--score-temperature", type=float, default=1.0,
        help="divide ranking logits by T before Acc@1 (eval-time calibration; "
             "default 1 = unchanged). Sweep on val, apply on test")
    e.add_argument(
        "--geo-score-bias", type=float, default=None,
        help="evaluate-time override of geo λ (None = keep checkpoint). "
             "Use with soft10r for locked-bias ε-mix (#38)")
    e.add_argument(
        "--revisit-score-bias", type=float, default=None,
        help="evaluate-time override of revisit λ (None = keep checkpoint). "
             "Use with soft10r for locked-bias ε-mix (#38)")
    e.add_argument(
        "--tta", type=str, default="off",
        help="eval-only TTA (#36): off or comma list mask,substitute "
             "(keep last tip; fuse across views)")
    e.add_argument(
        "--tta-views", type=int, default=1,
        help="number of random Mask/Substitute draws per mode (default 1)")
    e.add_argument(
        "--tta-fuse", type=str, default="mean", choices=("mean", "rrf"),
        help="how to fuse TTA view scores (mean or reciprocal-rank fusion)")
    e.add_argument(
        "--tta-no-keep-last", action="store_true",
        help="allow masking/substituting the last history tip (default: keep last)")
    e.add_argument(
        "--score-calib", type=str, default="off",
        choices=("off", "gauss_pop", "gauss_rev"),
        help="Acc-aware Gaussian CDF + feature mix (#34): gauss_pop subtracts "
             "λ·log1p(global visits); gauss_rev adds λ·user revisit")
    e.add_argument(
        "--calib-lambda", type=float, default=0.0,
        help="λ for --score-calib feature term (Acc-aware; default 0)")
    e.add_argument(
        "--candidate-spatial-prior", type=str, default="off",
        choices=("off", "geo", "gravity", "radiation"),
        help="eval-only Acc@1 pool shrink (#10): keep top-M by geo/gravity/"
             "radiation prior (always force-include GT; log natural coverage)")
    e.add_argument(
        "--candidate-prior-top-m", type=int, default=0,
        help="top-M kept by --candidate-spatial-prior (0 = off)")
    e.add_argument(
        "--candidate-prior-gravity-beta", type=float, default=1.0,
        help="β in gravity prior log1p(pop)−β·log1p(km) (default 1)")
    e.add_argument(
        "--mc-dropout", type=int, default=0,
        help="MC-dropout logit average at eval (#46): T stochastic forwards "
             "with dropout on; 0 = off. Try 10–20 on soft10r")
    e.add_argument(
        "--mc-keep-train-dropout", action="store_true",
        help="document flag for #46 (MC always enables train-mode dropout)")
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
    # --config is consumed by expand_config_argv; suppress leftover if present
    args = root.parse_args(argv)
    if getattr(args, "config", None):
        # Should have been expanded away; ignore leftover dest if parser kept it
        pass
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
                                    or getattr(args, "radiation_score_bias", 0.0) < 0
                                    or getattr(args, "jump_length_bias", 0.0) < 0
                                    or getattr(args, "cat_stage_bias", 0.0) < 0
                                    or getattr(args, "tsb_decay_hours", 0.0) < 0
                                    or getattr(args, "geo_score_power", 1.0) <= 0):
        root.error("use --negatives -1 or a positive count; --max-train-candidates -1 or >=1; "
                   "nonnegative strengths and positive lr/grad-clip; "
                   "transition-lambda / transition-score-bias / category-score-bias / "
                   "pop-score-bias / geo-score-bias / surd-score-bias / "
                   "revisit-score-bias / radiation-score-bias / jump-length-bias / "
                   "cat-stage-bias / tsb-decay-hours must be nonnegative; "
                   "geo-score-power must be positive")
    if args.command == "evaluate" and float(getattr(args, "score_temperature", 1.0) or 1.0) <= 0:
        root.error("--score-temperature must be positive")
    if args.command in ("evaluate", "train"):
        cprior = str(getattr(args, "candidate_spatial_prior", "off") or "off")
        ctop = int(getattr(args, "candidate_prior_top_m", 0) or 0)
        if cprior not in ("off", "none", "") and ctop < 1:
            root.error("--candidate-spatial-prior requires --candidate-prior-top-m >= 1")
        if ctop < 0:
            root.error("--candidate-prior-top-m must be nonnegative")
    if args.command == "evaluate":
        geo_ov = getattr(args, "geo_score_bias", None)
        rev_ov = getattr(args, "revisit_score_bias", None)
        if geo_ov is not None and float(geo_ov) < 0:
            root.error("--geo-score-bias override must be nonnegative")
        if rev_ov is not None and float(rev_ov) < 0:
            root.error("--revisit-score-bias override must be nonnegative")
        if int(getattr(args, "tta_views", 1) or 1) < 1:
            root.error("--tta-views must be >= 1")
    if (args.command == "train"
            and float(getattr(args, "geo_score_bias", 0.0) or 0.0) > 0
            and getattr(args, "candidate_context", "off") == "off"):
        root.error("--geo-score-bias > 0 requires --candidate-context on")
    if (args.command == "train"
            and float(getattr(args, "radiation_score_bias", 0.0) or 0.0) > 0
            and getattr(args, "candidate_context", "off") == "off"):
        root.error("--radiation-score-bias > 0 requires --candidate-context on")
    if (args.command == "train"
            and float(getattr(args, "jump_length_bias", 0.0) or 0.0) > 0
            and getattr(args, "candidate_context", "off") == "off"):
        root.error("--jump-length-bias > 0 requires --candidate-context on")
    if (args.command == "train"
            and float(getattr(args, "cat_stage_bias", 0.0) or 0.0) > 0
            and getattr(args, "candidate_context", "off") == "off"):
        root.error("--cat-stage-bias > 0 requires --candidate-context on")
    if args.command == "train" and getattr(args, "return_gate", "off") == "on":
        if float(getattr(args, "revisit_score_bias", 0.0) or 0.0) <= 0:
            root.error("--return-gate on requires --revisit-score-bias > 0")
        if (getattr(args, "explore_spatial_bias", "off") == "off"
                or float(getattr(args, "explore_score_bias", 0.0) or 0.0) <= 0):
            root.error("--return-gate on requires --explore-spatial-bias ≠ off "
                       "and --explore-score-bias > 0")
        if getattr(args, "candidate_context", "off") == "off":
            root.error("--return-gate explore spatial requires --candidate-context on")
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
