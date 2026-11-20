"""Scan ``results/`` ablation runs and write a master summary table.

Discovers ``**/eval_test/metrics.json`` (and optionally merges ``ablation_log.*``),
checks run-name ↔ recorded flags against ``docs/ablation.md`` naming, and includes
``known_only`` Acc@1 / NDCG@1 / MRR alongside the primary ``all_targets`` metrics.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

from .engine import (
    ABLATION_LOG_COLUMNS,
    PRIMARY_METRICS,
    _csv_escape,
    _metric_cell,
)

# Columns written to results/ablation_summary.{md,csv}
SUMMARY_COLUMNS = (
    "run_name",
    "leaf_name",
    "split",
    "cache",
    "key_flags",
    "Acc@1",
    "NDCG@1",
    "MRR",
    "known_Acc@1",
    "known_NDCG@1",
    "known_MRR",
    "known_coverage",
    "name_check",
    "mismatch",
    "notes",
    "metrics_path",
)

# Expected settings inferred from leaf run names in docs/ablation.md §3 / §3.1.
# Longer / more specific names must appear before shorter prefixes when matching
# (we use exact leaf match, so order only matters for documentation).
RUN_NAME_EXPECTATIONS: dict[str, dict] = {
    # Router
    "abl_router_hard": {"router_mode": "hard_prior"},
    "abl_router_surd_mlp": {"router_mode": "surd_mlp"},
    "abl_router_hybrid": {"router_mode": "hybrid"},
    "abl_nosurd": {"use_surd": False},
    # Negatives / caps
    "abl_neg_63": {"negatives": 63},
    "abl_neg_all_max256": {"negatives": -1, "max_train_candidates": 256},
    "abl_neg_all": {"negatives": -1},
    "abl_maxcand_256": {"max_train_candidates": 256},
    # Causal / transition
    "abl_causal_on": {"causal_counts": True},
    "abl_causal_off": {"causal_counts": False},
    "abl_tr_bias": {"transition_attn": "bias"},
    "abl_tr_gate": {"transition_attn": "gate"},
    "abl_tr_rel_key": {"transition_attn": "rel_key"},
    "abl_visit_bins_10": {"visit_bins": 10},
    # Candidate construction (prepare)
    "abl_cand_p90": {"candidate_radius": "p90"},
    "abl_cand_p80": {"candidate_radius": "p80"},
    "abl_cand_p70": {"candidate_radius": "p70"},
    "abl_cand_full": {"candidate_radius": "full"},
    "abl_cand_rnd0": {"random_negatives": 0},
    "abl_cand_rnd50": {"random_negatives": 50},
    "abl_cand_no_pop": {"popular_top_k": 0},
    "abl_cand_pop_p80": {"popular_radius": "p80"},
    "abl_cand_no_uf": {"user_frequent_top_k": 0},
    "abl_cand_nbhd": {"popular_top_k": 0, "user_frequent_top_k": 0},
    # §5 head-to-head: default multi-source vs random-only pools
    "abl_cand_baseline": {"candidate_radius": "p90"},
    "abl_cand_random": {"candidate_radius": "random"},
    # Target protocol
    "abl_tgt_all": {"train_targets": "all", "eval_targets": "all"},
    "abl_tgt_last": {"train_targets": "last", "eval_targets": "last"},
    "abl_tgt_train_all_eval_last": {"train_targets": "all", "eval_targets": "last"},
    # STHGCN-style unknown user/POI filter (prepare; default off)
    "abl_filter_unknown": {"filter_unknown": "on"},
    # Overlays A–E (§3.1)
    "abl_ctx_off": {
        "candidate_context": "off",
        "candidate_context_time": "off",
        "session_summary": "off",
        "user_pref": "off",
        "transition_pmi": "off",
        "expert_cross_attn": "off",
    },
    "abl_ctx_cat_geo_causal_slot": {
        "candidate_context": "on",
        "candidate_context_time": "causal_slot",
    },
    "abl_ctx_cat_geo_time": {
        "candidate_context": "on",
        "candidate_context_time": "last_slot",
    },
    "abl_ctx_cat_geo": {"candidate_context": "on"},
    "abl_sess_on": {"session_summary": "on"},
    "abl_upref_on": {"user_pref": "on"},
    "abl_tr_pmi": {"transition_pmi": "on"},
    "abl_e1_hybrid": {"expert_cross_attn": "hybrid"},
    "abl_e1_e3": {"expert_cross_attn": "hybrid", "expert_cross_attn_sparse": "on"},
    "abl_e1_dense": {"expert_cross_attn": "hybrid", "expert_cross_attn_sparse": "off"},
    "abl_e2_full": {"expert_cross_attn": "full"},
    "abl_ctx_sess_upref": {
        "candidate_context": "on",
        "session_summary": "on",
        "user_pref": "on",
    },
    "abl_ctx_sess": {"candidate_context": "on", "session_summary": "on"},
    "abl_ctx_e1": {"candidate_context": "on", "expert_cross_attn": "hybrid"},
}

# Keys compared when both expected and recorded have a value.
_COMPARE_KEYS = (
    "router_mode",
    "use_surd",
    "causal_counts",
    "transition_attn",
    "visit_bins",
    "negatives",
    "max_train_candidates",
    "candidate_radius",
    "popular_top_k",
    "popular_radius",
    "user_frequent_top_k",
    "random_negatives",
    "train_targets",
    "eval_targets",
    "filter_unknown",
    "candidate_context",
    "candidate_context_time",
    "session_summary",
    "user_pref",
    "transition_pmi",
    "expert_cross_attn",
    "expert_cross_attn_sparse",
)

_FLAG_PATTERNS = {
    "router_mode": re.compile(r"--router-mode\s+(\S+)"),
    "use_surd": re.compile(r"--no-surd\b"),
    "causal_counts_off": re.compile(r"--no-causal-counts\b"),
    "causal_counts_on": re.compile(r"--causal-counts\b"),
    "transition_attn": re.compile(r"--transition-attn\s+(\S+)"),
    "visit_bins": re.compile(r"--visit-bins\s+(\S+)"),
    "negatives": re.compile(r"--negatives\s+(\S+)"),
    "max_train_candidates": re.compile(r"--max-train-candidates\s+(\S+)"),
    "candidate_radius": re.compile(r"--candidate-radius\s+(\S+)"),
    "popular_top_k": re.compile(r"--popular-top-k\s+(\S+)"),
    "popular_radius": re.compile(r"--popular-radius\s+(\S+)"),
    "user_frequent_top_k": re.compile(r"--user-frequent-top-k\s+(\S+)"),
    "random_negatives": re.compile(r"--random-negatives\s+(\S+)"),
    "train_targets": re.compile(r"--train-targets\s+(\S+)"),
    "eval_targets": re.compile(r"--eval-targets\s+(\S+)"),
    "filter_unknown": re.compile(r"--filter-unknown\s+(\S+)"),
    "candidate_context": re.compile(r"--candidate-context\s+(\S+)"),
    "candidate_context_time": re.compile(r"--candidate-context-time\s+(\S+)"),
    "session_summary": re.compile(r"--session-summary\s+(\S+)"),
    "user_pref": re.compile(r"--user-pref\s+(\S+)"),
    "transition_pmi": re.compile(r"--transition-pmi\s+(\S+)"),
    "expert_cross_attn": re.compile(r"--expert-cross-attn\s+(\S+)"),
    "expert_cross_attn_sparse": re.compile(r"--expert-cross-attn-sparse\s+(\S+)"),
}


def leaf_run_name(run_name: str) -> str:
    """Last path segment of a results run name (``tokyo/abl_ctx_cat_geo`` → leaf)."""
    text = (run_name or "").strip().rstrip("/")
    if not text:
        return ""
    return Path(text).name


def expected_settings_for_run_name(run_name: str) -> dict | None:
    """Return expected settings for a documented ablation leaf name, or None."""
    leaf = leaf_run_name(run_name)
    if leaf in RUN_NAME_EXPECTATIONS:
        return dict(RUN_NAME_EXPECTATIONS[leaf])
    return None


def _coerce_scalar(value):
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if isinstance(value, float) and value.is_integer():
            return int(value)
        return value
    text = str(value).strip()
    low = text.lower()
    if low in ("true", "yes"):
        return True
    if low in ("false", "no"):
        return False
    if low in ("off", "on"):
        return low
    try:
        return int(text)
    except ValueError:
        pass
    try:
        f = float(text)
        return int(f) if f.is_integer() else f
    except ValueError:
        return text


def _norm_boolish(value):
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    text = str(value).strip().lower()
    if text in ("true", "yes", "on", "1"):
        return True
    if text in ("false", "no", "off", "0"):
        return False
    return value


def settings_equal(key: str, expected, recorded) -> bool:
    """Compare one setting with light normalization (bool / int / on-off)."""
    if expected is None or recorded is None:
        return expected is recorded
    if key in ("causal_counts", "use_surd"):
        return _norm_boolish(expected) == _norm_boolish(recorded)
    if key in (
        "candidate_context", "candidate_context_time", "session_summary",
        "user_pref", "transition_pmi", "expert_cross_attn",
        "expert_cross_attn_sparse",
    ):
        e, r = str(expected).strip().lower(), str(recorded).strip().lower()
        return e == r
    return _coerce_scalar(expected) == _coerce_scalar(recorded)


def parse_key_flags(flags: str) -> dict:
    """Parse a compact ``key_flags`` / CLI string into a settings dict."""
    text = flags or ""
    out = {}
    if _FLAG_PATTERNS["use_surd"].search(text):
        out["use_surd"] = False
    if _FLAG_PATTERNS["causal_counts_off"].search(text):
        out["causal_counts"] = False
    elif _FLAG_PATTERNS["causal_counts_on"].search(text):
        out["causal_counts"] = True
    for key, pattern in _FLAG_PATTERNS.items():
        if key in ("use_surd", "causal_counts_off", "causal_counts_on"):
            continue
        match = pattern.search(text)
        if match:
            out[key] = _coerce_scalar(match.group(1))
    return out


def extract_recorded_settings(payload: dict, key_flags: str | None = None) -> dict:
    """Merge settings from metrics.json (config / training_args / construction / cli)."""
    payload = payload or {}
    cfg = dict(payload.get("model_config") or {})
    args = dict(payload.get("training_args") or {})
    cli = dict(payload.get("cli") or {})
    cc = dict(payload.get("candidate_construction") or {})
    out = {}

    def take(key, *sources, transform=None):
        for src in sources:
            if key in src and src[key] is not None:
                value = src[key]
                out[key] = transform(value) if transform else value
                return

    take("router_mode", args, cfg, cli)
    if args.get("no_surd") or cfg.get("use_surd") is False or cli.get("no_surd"):
        out["use_surd"] = False
    elif "use_surd" in cfg:
        out["use_surd"] = cfg["use_surd"]

    if args.get("no_causal_counts") is True or cli.get("no_causal_counts") is True:
        out["causal_counts"] = False
    else:
        take("causal_counts", args, cfg, cli)

    take("transition_attn", args, cfg, cli)
    take("visit_bins", args, cfg, cli)
    take("negatives", args, cli)
    take("max_train_candidates", args, cli)
    take("candidate_context", args, cfg, cli)
    take("candidate_context_time", args, cfg, cli)
    take("session_summary", args, cfg, cli)
    take("user_pref", args, cfg, cli)
    take("transition_pmi", args, cfg, cli)
    take("expert_cross_attn", args, cfg, cli)
    take("expert_cross_attn_sparse", args, cfg, cli)

    # Cache / prepare fingerprint fields.
    if cc.get("radius_statistic") is not None:
        out["candidate_radius"] = cc["radius_statistic"]
    elif cli.get("candidate_radius") is not None:
        out["candidate_radius"] = cli["candidate_radius"]
    for src_key, dst_key in (
        ("popular_top_k", "popular_top_k"),
        ("popular_radius", "popular_radius"),
        ("user_frequent_top_k", "user_frequent_top_k"),
        ("random_negatives", "random_negatives"),
    ):
        if cc.get(src_key) is not None:
            out[dst_key] = cc[src_key]
        elif cli.get(dst_key) is not None:
            out[dst_key] = cli[dst_key]

    if payload.get("target_mode") is not None:
        out["eval_targets"] = payload["target_mode"]
    take("train_targets", cli)
    take("eval_targets", cli)
    if payload.get("filter_unknown") is not None:
        out["filter_unknown"] = payload["filter_unknown"]
    take("filter_unknown", cli)

    if key_flags:
        for key, value in parse_key_flags(key_flags).items():
            out.setdefault(key, value)
    return out


def validate_run_name_settings(run_name: str, recorded: dict) -> tuple[str, str]:
    """Return ``(name_check, mismatch_detail)``.

    ``name_check`` is one of: ``ok``, ``mismatch``, ``unknown_name``, ``no_settings``.
    """
    expected = expected_settings_for_run_name(run_name)
    if expected is None:
        leaf = leaf_run_name(run_name)
        if leaf.startswith("abl_"):
            return "unknown_name", f"no documented expectation for leaf '{leaf}'"
        return "unknown_name", "run name not in ablation naming table"
    if not recorded:
        return "no_settings", "no recorded flags/config to compare"
    problems = []
    for key in _COMPARE_KEYS:
        if key not in expected:
            continue
        if key not in recorded or recorded[key] is None:
            problems.append(f"{key}: expected={expected[key]!r} (missing in recorded)")
            continue
        if not settings_equal(key, expected[key], recorded[key]):
            problems.append(
                f"{key}: expected={expected[key]!r} recorded={recorded[key]!r}")
    if problems:
        return "mismatch", "; ".join(problems)
    return "ok", ""


def discover_metrics_paths(results_root: Path, *, split: str = "test") -> list[Path]:
    """Find ``eval_<split>/metrics.json`` under ``results_root`` (sorted)."""
    root = Path(results_root)
    if not root.is_dir():
        return []
    pattern = f"**/eval_{split}/metrics.json"
    return sorted(p for p in root.glob(pattern) if p.is_file())


def run_name_from_metrics_path(metrics_path: Path, results_root: Path) -> str:
    """POSIX path of the run dir under results/ (parent of eval_*)."""
    run_dir = Path(metrics_path).resolve().parent.parent
    root = Path(results_root).resolve()
    try:
        return run_dir.relative_to(root).as_posix()
    except ValueError:
        return run_dir.name


def load_ablation_log_index(results_root: Path) -> dict[str, dict]:
    """Latest ablation_log row per run_name (prefers CSV, falls back to MD)."""
    root = Path(results_root)
    csv_path, md_path = root / "ablation_log.csv", root / "ablation_log.md"
    rows: list[dict] = []
    if csv_path.is_file():
        with csv_path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("run_name"):
                    rows.append(row)
    elif md_path.is_file():
        for line in md_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line.startswith("|") or set(line.replace("|", "").strip()) <= {"-", " "}:
                continue
            cells = [c.strip() for c in line.strip("|").split("|")]
            if len(cells) < 8 or cells[0] == "date":
                continue
            rows.append(dict(zip(ABLATION_LOG_COLUMNS, cells)))
    index = {}
    for row in rows:
        index[row["run_name"]] = row  # last wins
    return index


def _format_key_flags_from_settings(settings: dict) -> str:
    """Best-effort compact flags string when ablation_log has no key_flags."""
    parts = []
    mapping = [
        ("candidate_radius", "--candidate-radius"),
        ("popular_top_k", "--popular-top-k"),
        ("popular_radius", "--popular-radius"),
        ("user_frequent_top_k", "--user-frequent-top-k"),
        ("random_negatives", "--random-negatives"),
        ("train_targets", "--train-targets"),
        ("eval_targets", "--eval-targets"),
        ("router_mode", "--router-mode"),
        ("transition_attn", "--transition-attn"),
        ("visit_bins", "--visit-bins"),
        ("candidate_context", "--candidate-context"),
        ("candidate_context_time", "--candidate-context-time"),
        ("session_summary", "--session-summary"),
        ("user_pref", "--user-pref"),
        ("transition_pmi", "--transition-pmi"),
        ("expert_cross_attn", "--expert-cross-attn"),
        ("expert_cross_attn_sparse", "--expert-cross-attn-sparse"),
        ("negatives", "--negatives"),
        ("max_train_candidates", "--max-train-candidates"),
    ]
    if settings.get("use_surd") is False:
        parts.append("--no-surd")
    if settings.get("causal_counts") is False:
        parts.append("--no-causal-counts")
    elif settings.get("causal_counts") is True:
        parts.append("--causal-counts")
    for key, flag in mapping:
        value = settings.get(key)
        if value is None:
            continue
        if key in (
            "candidate_context", "candidate_context_time", "session_summary",
            "user_pref", "transition_pmi", "expert_cross_attn",
        ) and str(value) == "off":
            continue
        parts.append(f"{flag} {value}")
    return " ".join(parts)


def build_summary_row(metrics_path: Path, results_root: Path,
                      log_index: dict[str, dict] | None = None,
                      *, split: str = "test") -> dict:
    """Build one master-table row from a metrics.json path."""
    payload = json.loads(Path(metrics_path).read_text(encoding="utf-8"))
    run_name = run_name_from_metrics_path(metrics_path, results_root)
    leaf = leaf_run_name(run_name)
    log_row = (log_index or {}).get(run_name) or {}
    key_flags = log_row.get("key_flags") or log_row.get("key flags") or ""
    recorded = extract_recorded_settings(payload, key_flags or None)
    if not key_flags:
        key_flags = _format_key_flags_from_settings(recorded)
    name_check, mismatch = validate_run_name_settings(run_name, recorded)
    all_row = payload.get("all_targets") or {}
    known_row = payload.get("known_only") or {}
    notes = log_row.get("notes") or ""
    if payload.get("timestamp_utc"):
        stamp = f"metrics_ts={payload['timestamp_utc']}"
        notes = f"{notes}; {stamp}" if notes else stamp
    return {
        "run_name": run_name,
        "leaf_name": leaf,
        "split": payload.get("split") or split,
        "cache": payload.get("cache") or log_row.get("cache") or "",
        "key_flags": key_flags,
        "Acc@1": _metric_cell(all_row, "Acc@1"),
        "NDCG@1": _metric_cell(all_row, "NDCG@1"),
        "MRR": _metric_cell(all_row, "MRR"),
        "known_Acc@1": _metric_cell(known_row, "Acc@1"),
        "known_NDCG@1": _metric_cell(known_row, "NDCG@1"),
        "known_MRR": _metric_cell(known_row, "MRR"),
        "known_coverage": _metric_cell(all_row, "coverage"),
        "name_check": name_check,
        "mismatch": mismatch,
        "notes": notes,
        "metrics_path": str(metrics_path),
    }


def collect_summary_rows(results_root: Path, *, split: str = "test") -> list[dict]:
    """Scan results and return sorted summary rows (Acc@1 descending when present)."""
    root = Path(results_root)
    log_index = load_ablation_log_index(root)
    rows = [
        build_summary_row(path, root, log_index, split=split)
        for path in discover_metrics_paths(root, split=split)
    ]

    def sort_key(row):
        try:
            acc = float(row["Acc@1"]) if row["Acc@1"] not in ("", None) else -1.0
        except ValueError:
            acc = -1.0
        return (-acc, row["run_name"])

    rows.sort(key=sort_key)
    return rows


SUMMARY_MD_HEADER = (
    "# Ablation master summary (regenerated)\n"
    "\n"
    "Generated by `python -m tsmoe ablation-summary`. "
    "Primary metrics are **all_targets** Acc@1 / NDCG@1 / MRR; "
    "**known_*** columns are the `known_only` slice "
    "(in-catalog ground-truth targets only — see [`ablation.md`](../docs/ablation.md) §6). "
    "`name_check` validates leaf run names against documented flags "
    "(`ok` / `mismatch` / `unknown_name` / `no_settings`).\n"
    "\n"
    "| " + " | ".join(SUMMARY_COLUMNS) + " |\n"
    "| " + " | ".join("---" for _ in SUMMARY_COLUMNS) + " |\n"
)


def write_summary_table(rows: list[dict], results_root: Path) -> tuple[Path, Path]:
    """Write ``ablation_summary.md`` and ``.csv`` under ``results_root``."""
    root = Path(results_root)
    root.mkdir(parents=True, exist_ok=True)
    md_path, csv_path = root / "ablation_summary.md", root / "ablation_summary.csv"
    md_lines = [SUMMARY_MD_HEADER]
    for row in rows:
        cells = ["" if row.get(col) is None else str(row.get(col)) for col in SUMMARY_COLUMNS]
        md_lines.append("| " + " | ".join(cells) + " |\n")
    md_path.write_text("".join(md_lines), encoding="utf-8")
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        f.write(",".join(SUMMARY_COLUMNS) + "\n")
        for row in rows:
            f.write(",".join(_csv_escape(row.get(col)) for col in SUMMARY_COLUMNS) + "\n")
    return md_path, csv_path


def ablation_summary_cli(args):
    """CLI entry: regenerate master summary under ``--results``."""
    results_root = Path(args.results)
    split = getattr(args, "split", "test")
    rows = collect_summary_rows(results_root, split=split)
    md_path, csv_path = write_summary_table(rows, results_root)
    n_ok = sum(1 for r in rows if r["name_check"] == "ok")
    n_bad = sum(1 for r in rows if r["name_check"] == "mismatch")
    n_unk = sum(1 for r in rows if r["name_check"] == "unknown_name")
    print(
        f"Scanned {len(rows)} eval_{split} run(s) under {results_root} "
        f"(name_check ok={n_ok} mismatch={n_bad} unknown={n_unk})",
        flush=True,
    )
    print(f"Wrote {md_path}", flush=True)
    print(f"Wrote {csv_path}", flush=True)
    if n_bad and getattr(args, "strict", False):
        raise SystemExit(1)
    return md_path, csv_path


def add_ablation_summary_parser(commands):
    """Register ``ablation-summary`` on the root argparse subparsers."""
    p = commands.add_parser(
        "ablation-summary",
        help="scan results/*/eval_*/metrics.json (+ ablation_log.*) and write "
             "results/ablation_summary.md + .csv with known_only metrics and "
             "run-name ↔ flags checks",
    )
    p.add_argument(
        "--results", default="results",
        help="results root to scan (default: results)",
    )
    p.add_argument(
        "--split", choices=("test", "validation"), default="test",
        help="which eval_* folder to prefer (default: test → eval_test)",
    )
    p.add_argument(
        "--strict", action="store_true",
        help="exit with status 1 if any documented run name mismatches recorded flags",
    )
    p.set_defaults(func=ablation_summary_cli)
    return p
