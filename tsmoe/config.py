"""Load train/eval recipe configs (YAML or JSON) and expand them into CLI flags.

Config keys use argparse ``dest`` names (snake_case), e.g. ``lambda_bpr``,
``transition_score_bias``. Optional ``extends:`` merges parent files first
(child wins). Meta keys (``recipe``, ``recipe_id``, ``city``, ``description``,
``extends``) are not turned into flags.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

META_KEYS = frozenset({
    "recipe", "recipe_id", "city", "description", "extends", "notes",
})

# Keys safe to pass through to `evaluate` from a train recipe config.
# (evaluate has no --seed; geo/revisit here are optional eval-time overrides.)
EVAL_CONFIG_KEYS = frozenset({
    "cache", "device", "chunk", "eval_batch_size", "threads",
    "score_temperature", "geo_score_bias", "revisit_score_bias",
})

# argparse store_true / BooleanOptionalAction destinations that become bare flags
BOOL_TRUE_FLAGS = frozenset({
    "no_surd",
    "no_causal_counts",
    "tta_no_keep_last",
    "mc_keep_train_dropout",
    "eval_test_every_epoch",
})
BOOL_FALSE_FLAGS = frozenset({
    "eval_test_every_epoch",  # --no-eval-test-every-epoch when False
})


def _load_raw(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    suffix = path.suffix.lower()
    if suffix == ".json":
        data = json.loads(text)
    elif suffix in (".yaml", ".yml"):
        try:
            import yaml  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise SystemExit(
                "PyYAML is required for .yaml configs. "
                "Install with: pip install 'pyyaml>=6,<7' "
                "(or use a .json config)."
            ) from exc
        data = yaml.safe_load(text)
    else:
        raise SystemExit(f"unsupported config type: {path} (use .yaml/.yml/.json)")
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise SystemExit(f"config root must be a mapping: {path}")
    return data


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a config file, resolving ``extends`` (relative to the file)."""
    path = Path(path).resolve()
    if not path.is_file():
        raise SystemExit(f"config not found: {path}")
    raw = _load_raw(path)
    parent_ref = raw.get("extends")
    merged: dict[str, Any] = {}
    if parent_ref:
        parent_path = (path.parent / str(parent_ref)).resolve()
        merged.update(load_config(parent_path))
    for key, value in raw.items():
        if key == "extends":
            continue
        merged[key] = value
    return merged


def _flag_name(dest: str) -> str:
    return "--" + dest.replace("_", "-")


def config_to_flags(cfg: Mapping[str, Any]) -> list[str]:
    """Convert a loaded config mapping into argv flags (no program name)."""
    flags: list[str] = []
    for key, value in cfg.items():
        if key in META_KEYS or value is None:
            continue
        if key == "no_causal_counts":
            # dual-opt: True → --no-causal-counts; False → omit (causal on)
            if value is True:
                flags.append("--no-causal-counts")
            continue
        if key == "eval_test_every_epoch":
            flags.append(
                "--eval-test-every-epoch" if value else "--no-eval-test-every-epoch"
            )
            continue
        if key in BOOL_TRUE_FLAGS:
            if value is True:
                flags.append(_flag_name(key))
            continue
        if isinstance(value, bool):
            # remaining bools: only emit when True as store_true-style
            if value:
                flags.append(_flag_name(key))
            continue
        flags.extend([_flag_name(key), str(value)])
    return flags


def expand_config_argv(argv: list[str]) -> list[str]:
    """Replace ``--config PATH`` with expanded flags; later CLI wins.

    Keeps the subcommand (first token if it does not start with ``-``) at the
    front. Supports ``--config PATH`` and ``--config=PATH``.
    """
    if not argv:
        return argv
    head: list[str] = []
    rest = list(argv)
    if rest and not rest[0].startswith("-"):
        head = [rest.pop(0)]

    config_path: str | None = None
    out: list[str] = []
    i = 0
    while i < len(rest):
        tok = rest[i]
        if tok == "--config":
            if i + 1 >= len(rest):
                raise SystemExit("--config requires a path")
            config_path = rest[i + 1]
            i += 2
            continue
        if tok.startswith("--config="):
            config_path = tok.split("=", 1)[1]
            i += 1
            continue
        out.append(tok)
        i += 1

    if config_path is None:
        return head + out

    cfg = load_config(config_path)
    command = head[0] if head else None
    if command == "evaluate":
        cfg = {k: v for k, v in cfg.items() if k in META_KEYS or k in EVAL_CONFIG_KEYS}
    # Config flags first, then explicit CLI so argparse last-wins for values.
    return head + config_to_flags(cfg) + out
