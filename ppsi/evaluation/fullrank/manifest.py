"""The in-memory evaluation manifest, its validation and manifest-only strata.

An evaluation manifest is one row per next-item decision. The rules the evaluator depends on:
  * no raw-ID column and no client index (privacy, and strata may not be cut by client);
  * canonical order = strictly ascending decision_id;
  * masks by construction: an eval_mask row carries an observed target; an evaluated row with target_class = -1 is
    exactly an OOV row; a non-evaluated (censored) row has target_class = -1 and target_oov = False;
  * censor_class = NONE iff eval_mask;
  * rankable = eval_mask AND target_class >= 0;
  * the ordered hashes, recomputed and compared with any value the header declares;
  * strata only from allowed manifest fields: a client index and any other non-manifest cut are refused.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np

from .errors import ManifestError, StrataError
from .hashing import ordered_decision_id_sha256, population_sha256

FORBIDDEN_COLUMNS = ("user_id", "product_id", "target_product_id", "user_session_raw", "derived_session_id",
                     "category_id", "brand_resolved", "category_code", "client_idx", "client_id")
REQUIRED_COLUMNS = ("decision_id", "eval_user_key", "eval_mask", "target_class", "target_oov", "censor_class")
CENSOR_CLASSES = ("NONE", "CEN_NO_NEXT", "CEN_LEGACY_BARRIER", "CEN_RIGHT_EDGE", "CEN_AMBIGUOUS_NEXT_BARRIER")
ALLOWED_STRATA_FIELDS = ("cold_stratum", "subperiod", "split", "panel_label", "oov_cause")
ENUMS = {
    "cold_stratum": ("WARM", "COLD_NO_TRAIN_EVENTS", "COLD_TRAIN_EVENTS_NO_LABEL"),
    "oov_cause": ("NONE", "BELOW_SUPPORT", "ABSENT_FROM_TRAIN", "NOT_APPLICABLE"),
}
HEADER_HASH_KEYS = ("ordered_decision_id_sha256", "eval_ordered_decision_id_sha256",
                    "rankable_ordered_decision_id_sha256")


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
    selection_key: np.ndarray | None = None   # per row, optional (population_sha256 check)

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
                "rankable_ordered_decision_id_sha256": ordered_decision_id_sha256(self.decision_id,
                                                                                  self.rankable)}

    @property
    def manifest_id(self) -> str:
        return str(self.header.get("manifest_id", ""))


def build_eval_manifest(header: Mapping, columns: Mapping[str, np.ndarray], *,
                        enums: Mapping[str, tuple] | None = None) -> EvalManifest:
    """Validate and wrap an evaluation manifest. `enums` registers the allowed values of optional fields
    (default `ENUMS`); a field with registered values may not carry any other value."""
    header = dict(header)
    enums = ENUMS if enums is None else enums
    bad = sorted(set(columns) & set(FORBIDDEN_COLUMNS))
    if bad:
        raise ManifestError(f"forbidden columns in an evaluation manifest: {bad}")
    missing = [c for c in REQUIRED_COLUMNS if c not in columns]
    if missing:
        raise ManifestError(f"missing required columns: {missing}")
    fields = {k: np.asarray(v) for k, v in columns.items() if k not in REQUIRED_COLUMNS
              and k not in ("selection_key", "rankable")}

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
        if k in enums:
            unknown = sorted(set(v.tolist()) - set(enums[k]))
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
        for u, s in zip(euk.tolist(), sel.tolist(), strict=True):
            if users.setdefault(u, s) != s:
                raise ManifestError("selection_key differs between rows of one user")
        ph = population_sha256(np.fromiter(users.values(), dtype=np.uint64))
        if ph != header["population_sha256"]:
            raise ManifestError(f"population_sha256 {header['population_sha256']} != recomputed {ph}")
    return m


def stratum_masks(manifest: EvalManifest, field_name: str,
                  allowed: tuple[str, ...] = ALLOWED_STRATA_FIELDS) -> dict:
    """Row masks per value of an allowed manifest field. Anything else (a client index included) is refused."""
    if field_name not in allowed:
        raise StrataError(f"stratum {field_name!r} is not an allowed manifest field {allowed}")
    if field_name not in manifest.fields:
        raise StrataError(f"manifest {manifest.manifest_id!r} has no {field_name!r} field")
    col = manifest.fields[field_name]
    return {str(v): (col == v) for v in sorted(set(col.tolist()))}
