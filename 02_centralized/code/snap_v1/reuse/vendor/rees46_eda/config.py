"""Frozen protocol configuration and workspace paths.

The split fences are frozen here BEFORE any fitting, recovery validation,
vocabulary construction, or feature statistics are computed.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import yaml

# ---------------------------------------------------------------------------
# Workspace layout
# ---------------------------------------------------------------------------

def workspace_root() -> Path:
    """Resolve the next_item_eda workspace root.

    ``REES46_WORKSPACE_ROOT`` overrides the location so the end-to-end
    integration smoke test can run the REAL stages against a throwaway
    workspace in a separate process without touching the primary run.
    """
    override = os.environ.get("REES46_WORKSPACE_ROOT")
    if override:
        return Path(override).resolve()
    return Path(__file__).resolve().parents[2]


def project_root() -> Path:
    return workspace_root().parent


@dataclass(frozen=True)
class Paths:
    root: Path

    @property
    def config(self) -> Path: return self.root / "config"

    @property
    def reports(self) -> Path: return self.root / "reports"

    @property
    def tables(self) -> Path: return self.root / "reports" / "tables"

    @property
    def figures(self) -> Path: return self.root / "reports" / "figures"

    @property
    def private_audit(self) -> Path: return self.root / "reports" / "private_audit"

    @property
    def sealed_test(self) -> Path: return self.root / "reports" / "sealed_test"

    @property
    def artifacts(self) -> Path: return self.root / "artifacts"

    @property
    def logs(self) -> Path: return self.root / "logs"

    @property
    def state(self) -> Path: return self.root / "state"

    @property
    def spill(self) -> Path: return self.root / "tmp_spill"

    def run_dir(self, run_id: str) -> Path:
        return self.artifacts / run_id


PATHS = Paths(workspace_root())

# ---------------------------------------------------------------------------
# Special categorical tokens (identical for every categorical field)
# ---------------------------------------------------------------------------
PAD = 0
MISSING = 1
OOV = 2
KNOWN_START = 3

# Sentinel used for "no rankable target"
NO_TARGET_CLASS = -1

# ---------------------------------------------------------------------------
# Split fences: frozen UTC half-open intervals
# ---------------------------------------------------------------------------
SPLIT_ORDER = ["TRAIN", "VALIDATION_SELECT", "VALIDATION_CONFIRM", "TEST", "LABEL_GRACE"]

SUPERVISED_PARTITIONS = ["TRAIN", "VALIDATION_SELECT", "VALIDATION_CONFIRM", "TEST"]
FUTURE_PARTITIONS = ["VALIDATION_SELECT", "VALIDATION_CONFIRM", "TEST"]


@dataclass(frozen=True)
class ProtocolConfig:
    raw: dict[str, Any]

    # -- convenience accessors -------------------------------------------------
    @property
    def schema_version(self) -> str:
        return self.raw["schema_version"]

    @property
    def splits(self) -> dict[str, tuple[str, str]]:
        s = self.raw["splits"]
        return {k: (v[0], v[1]) for k, v in s.items() if k in SPLIT_ORDER}

    def split_bounds_sql(self, name: str) -> tuple[str, str]:
        lo, hi = self.splits[name]
        return lo.replace("T", " ").replace("Z", ""), hi.replace("T", " ").replace("Z", "")

    @property
    def gap_seconds(self) -> int:
        return int(self.raw["sessions"]["gap_seconds"])

    @property
    def catalogue_min_support(self) -> int:
        return int(self.raw["catalogue"]["minimum_support"])

    @property
    def product_min_distinct_prior_timestamps(self) -> int:
        return int(self.raw["metadata"]["product_mapping_min_distinct_prior_timestamps"])

    @property
    def category_min_distinct_prior_products(self) -> int:
        return int(self.raw["metadata"]["category_code_mapping_min_distinct_prior_products"])

    @property
    def validation_cutoffs(self) -> list[str]:
        return [c.replace("T", " ").replace("Z", "") for c in self.raw["metadata"]["validation_cutoffs"]]

    @property
    def wilson_lower_bound(self) -> float:
        return float(self.raw["metadata"]["validation_wilson_accuracy_lower_bound"])

    @property
    def validation_min_products(self) -> int:
        return int(self.raw["metadata"]["validation_min_independent_products"])

    @property
    def user_context_window(self) -> int:
        return int(self.raw["features"]["user_context_window_previous_representatives"])

    @property
    def context_length_candidates(self) -> list[int]:
        return list(self.raw["loaders"]["context_length_candidates"])

    @property
    def clip_quantiles(self) -> tuple[float, float]:
        q = self.raw["features"]["numeric_clip_quantiles"]
        return float(q[0]), float(q[1])

    @property
    def sampling_seed(self) -> int:
        return int(self.raw["resources"]["sampling_seed"])

    @property
    def sensitivity_max_users(self) -> int:
        return int(self.raw["resources"]["sensitivity_max_users"])

    @property
    def sensitivity_gaps(self) -> list[int]:
        return list(self.raw["sessions"]["sensitivity_gaps_seconds"])


def load_protocol(path: Path | None = None) -> ProtocolConfig:
    p = path or (PATHS.config / "protocol.yaml")
    with open(p, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    return ProtocolConfig(raw=raw)


def config_hash(cfg: ProtocolConfig) -> str:
    blob = json.dumps(cfg.raw, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def module_hash(*module_names: str) -> str:
    """SHA-256 over the contents of specific pipeline modules only.

    Stage cache keys use this rather than the whole-package hash, so adding an
    unrelated module does not invalidate completed expensive stages.
    """
    src = Path(__file__).resolve().parent
    h = hashlib.sha256()
    for name in sorted(module_names):
        f = src / f"{name}.py"
        h.update(name.encode("utf-8"))
        h.update(f.read_bytes() if f.exists() else b"<absent>")
    return h.hexdigest()


def code_hash() -> str:
    """SHA-256 over the sorted contents of every pipeline source file."""
    src = Path(__file__).resolve().parent
    h = hashlib.sha256()
    for f in sorted(src.rglob("*.py")):
        if "__pycache__" in f.parts:
            continue
        h.update(f.name.encode("utf-8"))
        h.update(f.read_bytes())
    scripts = workspace_root() / "scripts"
    if scripts.exists():
        for f in sorted(scripts.rglob("*.py")):
            if "__pycache__" in f.parts:
                continue
            h.update(f.name.encode("utf-8"))
            h.update(f.read_bytes())
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Feature-set identity
# ---------------------------------------------------------------------------
FEATURE_SET_ID = "rees46_next_item_shared_v4_localtime"

# ---------------------------------------------------------------------------
# Contextual local time
# ---------------------------------------------------------------------------
# The canonical timestamp is and stays UTC. This is a SOURCE-INFORMED
# HYPOTHESIS about the audience's local clock, not a claim about where the
# store or its customers are. REES46 is historically associated with Saint
# Petersburg, which puts Europe/Moscow on the table as a defensible contextual
# timezone - it does NOT establish that the users are Russian.
#
# Verified rather than assumed: Europe/Moscow has exactly ONE offset (+3h)
# across 2019-10-01..2019-11-30, because Russia has observed permanent standard
# time since October 2014. There is no DST transition inside the data window,
# so a fixed +3h shift is exact here and needs no tz database at query time.
CONTEXTUAL_TZ = "Europe/Moscow"
CONTEXTUAL_TZ_OFFSET_HOURS = 3
CONTEXTUAL_TZ_STATUS = "HYPOTHESIS_NOT_CONFIRMED_STORE_GEOGRAPHY"
# Offsets compared in the TRAIN-only timezone audit (stage 04c).
TZ_AUDIT_OFFSETS = [0, 2, 3, 4]

# Frozen for v1. Boundaries are local hours, lower bound inclusive.
# NEVER re-tuned after looking at TEST.
DAYPARTS = [
    ("NIGHT", 0, 6),
    ("MORNING", 6, 12),
    ("AFTERNOON", 12, 17),
    ("EVENING", 17, 21),
    ("LATE_EVENING", 21, 24),
]
WEEKDAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday",
                 "Friday", "Saturday", "Sunday"]


def local_ts_sql(col: str = "block_ts") -> str:
    """The same instant expressed on the contextual local clock."""
    return f"({col} + INTERVAL {CONTEXTUAL_TZ_OFFSET_HOURS} HOUR)"


def daypart_case_sql(col: str = "block_ts") -> str:
    """Frozen v1 daypart, from the LOCAL hour. Never re-tuned against TEST."""
    lt = local_ts_sql(col)
    arms = "\n       ".join(
        f"WHEN extract(hour FROM {lt}) >= {lo} AND extract(hour FROM {lt}) < {hi} "
        f"THEN '{name}'" for name, lo, hi in DAYPARTS)
    return f"CASE {arms} END"


def daypart_index() -> dict[str, int]:
    """Deterministic contiguous daypart indices after the reserved tokens.

    The daypart set is closed and every value occurs in TRAIN, so "fitted on
    TRAIN" and "declared here" are the same mapping; declaring it keeps the
    index stable even if a future TRAIN window happened to miss an hour band.
    """
    return {name: KNOWN_START + i for i, (name, _, _) in enumerate(DAYPARTS)}


def weekday_index() -> dict[str, int]:
    """Monday..Sunday -> contiguous indices after the reserved tokens."""
    return {n: KNOWN_START + i for i, n in enumerate(WEEKDAY_NAMES)}


# ---------------------------------------------------------------------------
# Ablation feature groups
# ---------------------------------------------------------------------------
# Materialised so training can run ablations WITHOUT rebuilding the dataset.
# Each group names batch channels and numeric feature names; a group is a VIEW
# over one shared batch, never a separate extraction. Nothing here decides
# which group wins - that is a training question, and deciding it during
# preprocessing would be tuning the data against model scores.
# Which quality flag MUST travel with a numeric feature. The protocol owner's
# price decision (C5) is explicit: a neutral value reaches the model ONLY
# together with its missing indicator. That applies to every zero-filled
# numeric, not just price - a group that ships `log1p_price = 0.0` with no
# `price_missing` channel is telling the model the item is free.
REQUIRED_FLAG_FOR: dict[str, list[str]] = {
    "log1p_price": ["price_missing", "price_zero", "price_conflict"],
    "log1p_prev_raw_view_gap_s": ["prev_gap_unavailable"],
    "log1p_item_observed_age_s": ["item_age_unavailable"],
}

FEATURE_GROUPS: dict[str, dict[str, list[str]]] = {
    "F0_item_only": {
        "quality_flags": [],
        "categorical": ["item_tokens"],
        "numeric": [],
        "user_context": [],
    },
    "F1_core_categorical": {
        "quality_flags": [],
        "categorical": ["item_tokens", "category_tokens", "brand_tokens",
                        "main_category_tokens"],
        "numeric": [],
        "user_context": [],
    },
    "F2_time_context": {
        "quality_flags": ["ALL"],
        "categorical": ["item_tokens", "category_tokens", "brand_tokens",
                        "main_category_tokens", "daypart_tokens", "weekday_tokens"],
        "numeric": ["local_hour_sin", "local_hour_cos", "local_weekday_sin",
                    "local_weekday_cos", "is_weekend",
                    "utc_hour_sin", "utc_hour_cos", "utc_weekday_sin",
                    "utc_weekday_cos",
                    "log1p_prev_raw_view_gap_s", "log1p_segment_elapsed_s",
                    "log1p_item_observed_age_s", "prefix_representative_count"],
        "user_context": [],
    },
    "F3_user_session_context": {
        "quality_flags": ["ALL"],
        "categorical": ["item_tokens", "category_tokens", "brand_tokens",
                        "main_category_tokens", "daypart_tokens", "weekday_tokens"],
        "numeric": ["local_hour_sin", "local_hour_cos", "local_weekday_sin",
                    "local_weekday_cos", "is_weekend",
                    "utc_hour_sin", "utc_hour_cos", "utc_weekday_sin",
                    "utc_weekday_cos",
                    "log1p_prev_raw_view_gap_s", "log1p_segment_elapsed_s",
                    "log1p_item_observed_age_s", "prefix_representative_count",
                    "prefix_item_count", "prefix_known_category_count",
                    "prefix_category_switch_count",
                    "current_known_category_run_length", "log1p_prefix_elapsed_s"],
        "user_context": ["ALL"],
    },
    "F4_rich_side_information": {
        "quality_flags": ["ALL"],
        "categorical": ["item_tokens", "category_tokens", "brand_tokens",
                        "main_category_tokens", "daypart_tokens", "weekday_tokens"],
        "numeric": ["ALL"],
        "user_context": ["ALL"],
    },
}

FEATURE_SET_ID_PREVIOUS = "rees46_next_item_shared_v3"

EVENT_NUMERIC_FEATURES = [
    "log1p_price",
    # UTC periodic context - retained so a later ablation can compare UTC
    # against audience-local context WITHOUT rebuilding the dataset.
    "utc_hour_sin",
    "utc_hour_cos",
    "utc_weekday_sin",
    "utc_weekday_cos",
    # Audience-local periodic context (contextual hypothesis timezone).
    "local_hour_sin",
    "local_hour_cos",
    "local_weekday_sin",
    "local_weekday_cos",
    "is_weekend",
    "log1p_prev_raw_view_gap_s",
    "log1p_segment_elapsed_s",
    "log1p_item_observed_age_s",
    "prefix_representative_count",
]

# Cyclical encodings and binary indicators are passed through verbatim: they are
# already bounded and centred, and z-scoring them would destroy the geometry
# that makes them useful.
NEVER_NORMALIZED = [
    "utc_hour_sin", "utc_hour_cos", "utc_weekday_sin", "utc_weekday_cos",
    "local_hour_sin", "local_hour_cos", "local_weekday_sin", "local_weekday_cos",
    "is_weekend",
]

EVENT_QUALITY_FLAGS = [
    "price_missing",
    "price_zero",
    "price_conflict",
    "brand_missing",
    "brand_recovered",
    "brand_conflict",
    "category_missing",
    "category_recovered",
    "category_conflict",
    "category_code_missing",
    "category_code_recovered",
    "main_category_missing",
    "prev_gap_unavailable",
    "item_age_unavailable",
    "simultaneous_block",
]

USER_CONTEXT_FEATURES = [
    "log1p_prev_view_count",
    "log1p_prev_session_count",
    "recent_history_length",
    "has_history",
    "recent_distinct_items",
    "recent_distinct_known_main_categories",
    "recent_category_match_share",
    "recent_brand_match_share",
    "recent_mean_log_price",
    "current_minus_recent_mean_log_price",
    "log1p_time_since_prev_user_event_s",
    "log1p_time_since_prev_session_last_event_s",
]

USER_CONTEXT_MASKS = [
    "has_history_mask",
    "recent_category_match_supported",
    "recent_brand_match_supported",
    "recent_price_supported",
    "prev_user_event_available",
    "prev_session_available",
]

SESSION_PREFIX_FEATURES = [
    "prefix_item_count",
    "prefix_known_category_count",
    "prefix_category_switch_count",
    "current_known_category_run_length",
    "log1p_prefix_elapsed_s",
]

# Numeric features that are normalized (clip + z-score). Sine/cosine and
# boolean flags are explicitly excluded from clipping and scaling.
NORMALIZED_EVENT_NUMERIC = [
    "log1p_price",
    "log1p_prev_raw_view_gap_s",
    "log1p_segment_elapsed_s",
    "log1p_item_observed_age_s",
    "prefix_representative_count",
]

NORMALIZED_USER_CONTEXT = [
    "log1p_prev_view_count",
    "log1p_prev_session_count",
    "recent_history_length",
    "recent_distinct_items",
    "recent_distinct_known_main_categories",
    "recent_mean_log_price",
    "current_minus_recent_mean_log_price",
    "log1p_time_since_prev_user_event_s",
    "log1p_time_since_prev_session_last_event_s",
]

NORMALIZED_SESSION_PREFIX = [
    "prefix_item_count",
    "prefix_known_category_count",
    "prefix_category_switch_count",
    "current_known_category_run_length",
    "log1p_prefix_elapsed_s",
]


def resolve_and_write(cfg: ProtocolConfig, run_id: str, extra: dict[str, Any]) -> Path:
    """Write config/protocol.resolved.yaml capturing what was actually used."""
    resolved = dict(cfg.raw)
    resolved["_resolved"] = {
        "run_id": run_id,
        "task_version": cfg.schema_version,
        "config_hash": config_hash(cfg),
        "code_hash": code_hash(),
        "feature_set_id": FEATURE_SET_ID,
        "event_numeric_features": EVENT_NUMERIC_FEATURES,
        "event_quality_flags": EVENT_QUALITY_FLAGS,
        "user_context_features": USER_CONTEXT_FEATURES,
        "user_context_masks": USER_CONTEXT_MASKS,
        "session_prefix_features": SESSION_PREFIX_FEATURES,
        **extra,
    }
    out = PATHS.config / "protocol.resolved.yaml"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        yaml.safe_dump(resolved, fh, sort_keys=False, allow_unicode=True)
    return out
