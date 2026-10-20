"""v4eval.manifest — the in-memory evaluation manifest, its validation, the TEST gate and manifest-only strata.

Implements the MANIFEST_SPEC_V4 evaluation-manifest rules the evaluator depends on:
  * no raw-ID and no client_idx column (global_rules.no_raw_ids);
  * canonical order = strictly ascending decision_id (global_rules.ordering);
  * masks by construction (QA-01 BOUNDARY_DEFINITIONS L115-L153): eval_mask rows carry an observed target; an eval row
    with target_class = -1 is exactly an OOV row; a non-eval row has target_class = -1 and target_oov = False;
  * censor_class = NONE iff eval_mask (EVALUATION_CONTRACT_V4 §2.2, E2.1);
  * rankable = eval_mask AND target_class >= 0 (E3.2);
  * the ordered hashes, recomputed and compared with any value the header declares;
  * the TEST gate (E10.1, T48): a RETRO_TEST manifest, or any manifest holding a TEST row, is refused unless an
    ALL_METHODS_LOCK file exists whose sha256 equals the header's all_methods_lock_sha256, and the header names its
    TEST_EXPOSURE_REGISTER entry;
  * strata only from allowed manifest fields (E3.5): client_idx and any other non-manifest cut are refused.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Optional

import numpy as np

from .errors import LockGateError, ManifestError, StrataError
from .hashing import file_sha256, ordered_decision_id_sha256, population_sha256

FORBIDDEN_COLUMNS = ("user_id", "product_id", "target_product_id", "user_session_raw", "derived_session_id",
                     "category_id", "brand_resolved", "category_code", "client_idx", "client_id")
REQUIRED_COLUMNS = ("decision_id", "eval_user_key", "eval_mask", "target_class", "target_oov", "censor_class")
CENSOR_CLASSES = ("NONE", "CEN_NO_NEXT", "CEN_LEGACY_BARRIER", "CEN_RIGHT_EDGE", "CEN_AMBIGUOUS_NEXT_BARRIER")
ALLOWED_STRATA_FIELDS = ("cold_stratum", "subperiod", "old_split", "panel_label", "oov_cause")
ENUMS = {
    "cold_stratum": ("WARM", "COLD_NO_TRAIN_EVENTS", "COLD_TRAIN_EVENTS_NO_LABEL"),
    "subperiod": ("VAL_A_20191116_20191120", "VAL_B_20191120_20191123", "NONE"),
    "old_split": ("TRAIN", "VALIDATION_SELECT", "VALIDATION_CONFIRM", "TEST"),
    "v4_split": ("TRAIN", "VALIDATION", "RETRO_TEST"),
    "panel_label": ("COLD_START", "WARM_TRUNCATION", "RECIPE_SELECTION"),
    "oov_cause": ("NONE", "BELOW_SUPPORT", "ABSENT_FROM_TRAIN", "NOT_APPLICABLE"),
}
TEST_MANIFEST_PREFIXES = ("M07", "M08", "M09")
HEADER_HASH_KEYS = ("ordered_decision_id_sha256", "eval_ordered_decision_id_sha256", "rankable_ordered_decision_id_sha256")


@dataclass
class EvalManifest:
    header: dict
    decision_id: np.ndarray
    eval_user_key: np.ndarray
    eval_mask: np.ndarray
    target_class: np.ndarray
    target_oov: np.ndarray
    censor_class: np.ndarray
    catalogue_K: int
    fields: dict = field(default_factory=dict)
    selection_key: Optional[np.ndarray] = None   # per row, optional (population_sha256 check)

    @property
    def n(self) -> int:
        return int(self.decision_id.shape[0])

    @property
    def rankable(self) -> np.ndarray:
        return self.eval_mask & (self.target_class >= 0)

    @property
    def hashes(self) -> dict:
        return {"ordered_decision_id_sha256": ordered_decision_id_sha256(self.decision_id),
                "eval_ordered_decision_id_sha256": ordered_decision_id_sha256(self.decision_id, self.eval_mask),
                "rankable_ordered_decision_id_sha256": ordered_decision_id_sha256(self.decision_id, self.rankable)}

    @property
    def manifest_id(self) -> str:
        return str(self.header.get("manifest_id", ""))


def _is_test_manifest(header: Mapping, fields: Mapping[str, np.ndarray]) -> bool:
    if str(header.get("v4_split", "")) == "RETRO_TEST":
        return True
    if str(header.get("manifest_id", "")).startswith(TEST_MANIFEST_PREFIXES):
        return True
    if "v4_split" in fields and bool(np.any(fields["v4_split"] == "RETRO_TEST")):
        return True
    if "old_split" in fields and bool(np.any(fields["old_split"] == "TEST")):
        return True
    return False


def check_test_gate(header: Mapping, fields: Mapping[str, np.ndarray], all_methods_lock_path: Optional[Path]) -> None:
    """E10.1: no RETRO_TEST data is opened before ALL_METHODS_LOCK exists with the registered sha256."""
    if not _is_test_manifest(header, fields):
        return
    want = header.get("all_methods_lock_sha256")
    if not want:
        raise LockGateError("TEST manifest refused: the header does not carry all_methods_lock_sha256")
    if not header.get("test_exposure_register_entry_id"):
        raise LockGateError("TEST manifest refused: no TEST_EXPOSURE_REGISTER entry id in the header")
    if all_methods_lock_path is None:
        raise LockGateError("TEST manifest refused: no ALL_METHODS_LOCK path given")
    p = Path(all_methods_lock_path)
    if not p.is_file():
        raise LockGateError(f"TEST manifest refused: ALL_METHODS_LOCK file {p} does not exist")
    got = file_sha256(p)
    if got != want:
        raise LockGateError(f"TEST manifest refused: ALL_METHODS_LOCK sha256 {got} != header {want}")


def build_eval_manifest(header: Mapping, columns: Mapping[str, np.ndarray], *,
                        all_methods_lock_path: Optional[Path] = None) -> EvalManifest:
    """Validate and wrap an evaluation manifest. The TEST gate runs before any column is interpreted."""
    header = dict(header)
    bad = sorted(set(columns) & set(FORBIDDEN_COLUMNS))
    if bad:
        raise ManifestError(f"forbidden columns in an evaluation manifest: {bad}")
    missing = [c for c in REQUIRED_COLUMNS if c not in columns]
    if missing:
        raise ManifestError(f"missing required columns: {missing}")
    fields = {k: np.asarray(v) for k, v in columns.items() if k not in REQUIRED_COLUMNS
              and k not in ("selection_key", "rankable")}
    check_test_gate(header, fields, all_methods_lock_path)

    if "catalogue_K" not in header:
        raise ManifestError("header must declare catalogue_K")
    K = int(header["catalogue_K"])
    did = np.asarray(columns["decision_id"])
    if did.dtype.kind != "i":
        raise ManifestError("decision_id must be an integer column")
    did = did.astype(np.int64)
    n = did.shape[0]
    arrs = {}
    for c in REQUIRED_COLUMNS[1:]:
        a = np.asarray(columns[c])
        if a.shape != (n,):
            raise ManifestError(f"column {c} has shape {a.shape}, expected ({n},)")
        arrs[c] = a
    for k, v in fields.items():
        if v.shape != (n,):
            raise ManifestError(f"column {k} has shape {v.shape}, expected ({n},)")
        if k in ENUMS:
            unknown = sorted(set(v.tolist()) - set(ENUMS[k]))
            if unknown:
                raise ManifestError(f"unregistered {k} values: {unknown}")
    if n > 1 and not bool(np.all(did[1:] > did[:-1])):
        raise ManifestError("decision_id must be strictly ascending (canonical order, unique)")
    euk = arrs["eval_user_key"].astype(np.int64)
    em = arrs["eval_mask"].astype(bool)
    tc = arrs["target_class"].astype(np.int64)
    oov = arrs["target_oov"].astype(bool)
    cc = arrs["censor_class"].astype(str)
    if n and (tc.min() < -1 or tc.max() > K - 1):
        raise ManifestError(f"target_class outside [-1, {K - 1}]")
    if n and euk.min() < 0:
        raise ManifestError("eval_user_key must be >= 0 (a dense per-manifest key, never an OOV token)")
    if bool(np.any(oov & ~(em & (tc == -1)))):
        raise ManifestError("target_oov must imply eval_mask AND target_class = -1")
    if bool(np.any(em & (tc == -1) & ~oov)):
        raise ManifestError("an evaluated row with target_class = -1 must be an OOV row (target_oov)")
    if bool(np.any(~em & ((tc != -1) | oov))):
        raise ManifestError("a non-evaluated row carries no target (target_class = -1, target_oov = False)")
    unknown_cc = sorted(set(cc.tolist()) - set(CENSOR_CLASSES))
    if unknown_cc:
        raise ManifestError(f"unregistered censor_class values: {unknown_cc}")
    if bool(np.any(em != (cc == "NONE"))):
        raise ManifestError("censor_class must be NONE iff eval_mask")
    if "rankable" in columns and not np.array_equal(np.asarray(columns["rankable"], dtype=bool), em & (tc >= 0)):
        raise ManifestError("declared rankable column != eval_mask AND target_class >= 0")
    sel = None
    if "selection_key" in columns:
        sel = np.asarray(columns["selection_key"], dtype=np.uint64)
    m = EvalManifest(header=header, decision_id=did, eval_user_key=euk, eval_mask=em, target_class=tc,
                     target_oov=oov, censor_class=cc, catalogue_K=K, fields=fields, selection_key=sel)
    got = m.hashes
    for k in HEADER_HASH_KEYS:
        if header.get(k) is not None and header[k] != got[k]:
            raise ManifestError(f"header {k} {header[k]} != recomputed {got[k]}")
    if sel is not None and header.get("population_sha256") is not None:
        users = {}
        for u, s in zip(euk.tolist(), sel.tolist()):
            if users.setdefault(u, s) != s:
                raise ManifestError("selection_key differs between rows of one user")
        ph = population_sha256(np.fromiter(users.values(), dtype=np.uint64))
        if ph != header["population_sha256"]:
            raise ManifestError(f"population_sha256 {header['population_sha256']} != recomputed {ph}")
    return m


def stratum_masks(manifest: EvalManifest, field_name: str) -> dict:
    """Row masks per value of an allowed manifest field. Anything else (client_idx included) is refused."""
    if field_name not in ALLOWED_STRATA_FIELDS:
        raise StrataError(f"stratum {field_name!r} is not an allowed manifest field {ALLOWED_STRATA_FIELDS}")
    if field_name not in manifest.fields:
        raise StrataError(f"manifest {manifest.manifest_id!r} has no {field_name!r} field")
    col = manifest.fields[field_name]
    return {str(v): (col == v) for v in sorted(set(col.tolist()))}
