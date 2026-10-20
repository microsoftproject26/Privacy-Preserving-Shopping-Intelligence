"""v4data.transform (owner: Data Lead, DATA-04) - the load-time Clean V2 transform, exactly per schema 2.1.0.

Two equivalent entry points (contract DATA_CONTRACT_CLEAN_V2 section 10):

* `encode_rows(raw, spec)`   - the REFERENCE: stored v1 raw columns (one row per canonical representative) ->
                               V2 per-row blocks. Every normalized channel is clip -> z-score in float64 with the
                               frozen V2 parameters, cast to float32, then re-zeroed (rezero_map_v2). This is the
                               INJECTED_NORMALIZERS formula.
* `clean_v2_view(batch, spec)` - a view over a v1-rich batch that the snapshot CompactStore already encoded with
                               the v1 normalizers. Kept columns pass through byte-identically; the three re-encoded
                               user-context columns (F-1, M1 x2) are recovered by the exact float64 inverse of the v1
                               z on available rows, re-encoded with the V2 values and re-zeroed by their gate
                               (m1_application_path = EXACT_INVERSE).

Both run the runtime identity checks RC1-RC8 BEFORE any column is dropped and refuse the batch on any failure.
Nothing here writes to disk. No v1 or lean weight is involved (from-scratch topology).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional

import numpy as np

from .common import SCHEMA_SELF_SHA256, canonical_json_bytes, load_schema, sha256_bytes

RC_TOL = 1e-6
SQRT2 = float(np.sqrt(2.0))


class IdentityCheckError(AssertionError):
    """A runtime identity check (RC1-RC8) failed: the batch is refused (fail-closed)."""


class SchemaError(ValueError):
    pass


class StateError(ValueError):
    """Illegal transform-state transition (T11): double transform, wrong input schema, wrong encoding."""


# ------------------------------------------------------------------------------------------------ spec
@dataclass(frozen=True)
class CleanV2Spec:
    v1_numeric: tuple
    v1_flags: tuple
    v1_uctx: tuple
    v1_masks: tuple
    v2_numeric: tuple
    num_src: tuple
    v2_flags: tuple
    flag_src: tuple
    v2_uctx: tuple
    uctx_src: tuple
    v2_masks: tuple
    mask_src: tuple
    norm_v2: Mapping[str, tuple]           # name -> (clip_lo, clip_hi, mean, std), all 17 normalized kept columns
    reencode: Mapping[str, dict]           # the 3 re-encoded user-context columns: old/new values and gate
    states_numeric: tuple
    states_uctx: tuple
    input_schema_sha256: str
    schema_self_sha256: str
    dead_flags: tuple = field(default=("price_conflict", "brand_conflict", "category_missing",
                                       "category_recovered", "category_conflict", "category_code_recovered"))


def input_schema_sha256_of(numeric, flags, uctx, masks) -> str:
    """Hash of the v1-rich input layout a batch must carry (names and order of the four blocks)."""
    return sha256_bytes(canonical_json_bytes({
        "event_numeric_features": list(numeric), "event_quality_flags": list(flags),
        "user_context": list(uctx), "user_context_masks": list(masks)}))


def validate_schema(schema: Mapping) -> None:
    """The schema invariants DATA-04 relies on (T12/T14 schema validator). Raises SchemaError."""
    ch = schema["channels"]
    if len(ch) != 59 or [c["n"] for c in ch] != list(range(1, 60)):
        raise SchemaError("schema must hold exactly the 59 v1 rich channels, numbered 1..59")
    dec = {}
    for c in ch:
        dec[c["decision"]] = dec.get(c["decision"], 0) + 1
        if not c.get("locality") or c["locality"] not in (
                "LOCAL_EVENT", "LOCAL_HISTORY", "SHARED_CATALOGUE", "SHARED_TRAIN_STATISTIC"):
            raise SchemaError(f"channel {c['name']} carries no valid locality label")
        if c["decision"] == "DROP_REDUNDANT" and (not c.get("reconstruction") or c.get("v2_slot") is not None):
            raise SchemaError(f"DROP_REDUNDANT channel {c['name']} lacks its reconstruction or keeps a V2 slot")
        if c["decision"] != "DROP_REDUNDANT" and c.get("v2_slot") is None:
            raise SchemaError(f"kept channel {c['name']} has no V2 slot")
    if dec != {"KEEP": 51, "DROP_REDUNDANT": 6, "RE_ENCODE_M1": 2}:
        raise SchemaError(f"decision counts {dec} != KEEP 51 / DROP_REDUNDANT 6 / RE_ENCODE_M1 2")
    lay = schema["tensor_layout_v2"]
    widths = (len(lay["event_numeric_features"]["names"]), len(lay["event_quality_flags"]["names"]),
              len(lay["user_context"]["names"]), len(lay["user_context_masks"]["names"]))
    if widths != (15, 15, 12, 4):
        raise SchemaError(f"V2 widths {widths} != (15, 15, 12, 4)")
    flags = set(lay["event_quality_flags"]["names"])
    for f in ("price_conflict", "brand_conflict", "category_missing", "category_recovered", "category_conflict",
              "category_code_recovered", "price_zero", "main_category_missing"):
        if f not in flags:
            raise SchemaError(f"future-relevant flag {f} must stay in the V2 tensor (PKG/03 section 7 L42)")
    if len(schema["normalized_channels_v2"]) != 17:
        raise SchemaError("normalized_channels_v2 must list the 17 normalized kept channels")


def build_spec(schema: Optional[Mapping] = None) -> CleanV2Spec:
    s = schema if schema is not None else load_schema()
    validate_schema(s)
    ch = s["channels"]
    blk = lambda b: tuple(c["name"] for c in ch if c["block"] == b)  # noqa: E731
    v1_num, v1_flag, v1_uctx, v1_mask = blk("event_numeric"), blk("event_flag"), blk("user_context"), blk("user_mask")
    lay = s["tensor_layout_v2"]

    def sel(key, v1names):
        names = tuple(lay[key]["names"])
        src = tuple(int(i) for i in lay[key]["v1_source_indices"])
        if tuple(v1names[i] for i in src) != names:
            raise SchemaError(f"{key}: v1_source_indices do not map onto the V2 names")
        return names, src

    v2_num, num_src = sel("event_numeric_features", v1_num)
    v2_flag, flag_src = sel("event_quality_flags", v1_flag)
    v2_uctx, uctx_src = sel("user_context", v1_uctx)
    v2_mask, mask_src = sel("user_context_masks", v1_mask)
    norm = {k: (float(v["clip_lo"]), float(v["clip_hi"]), float(v["mean"]), float(v["std"]))
            for k, v in s["normalized_channels_v2"].items()}
    m1 = s["m1_reencoding"]["features"]
    f1 = s["f1_reencoding"]
    re = {
        "current_minus_recent_mean_log_price": {
            "uctx_index": int(f1["user_context_index"]), "gate": "F1",
            "clip": (float(f1["clip_lo"]), float(f1["clip_hi"])),
            "old": (float(f1["old_mean"]), float(f1["old_std"])),
            "new": (float(f1["new_mean"]), float(f1["new_std"]))},
    }
    for name, e in m1.items():
        re[name] = {"uctx_index": int(e["user_context_index"]), "gate": e["governing_mask_stored"],
                    "clip": (float(e["clip_lo"]), float(e["clip_hi"])),
                    "old": (float(e["old_mean"]), float(e["old_std"])),
                    "new": (float(e["new_mean"]), float(e["new_std"]))}
    for name, e in re.items():
        lo, hi, mu, sd = norm[name]
        if (lo, hi) != e["clip"] or (mu, sd) != e["new"]:
            raise SchemaError(f"{name}: normalized_channels_v2 disagrees with its re-encoding block")
        if not sd > 1e-12 * max(1.0, abs(mu)):
            raise SchemaError(f"degenerate valid-only std for {name}: {sd}")
    return CleanV2Spec(
        v1_numeric=v1_num, v1_flags=v1_flag, v1_uctx=v1_uctx, v1_masks=v1_mask,
        v2_numeric=v2_num, num_src=num_src, v2_flags=v2_flag, flag_src=flag_src,
        v2_uctx=v2_uctx, uctx_src=uctx_src, v2_masks=v2_mask, mask_src=mask_src,
        norm_v2=norm, reencode=re,
        states_numeric=tuple(lay["event_numeric_features"]["states"]),
        states_uctx=tuple(lay["user_context"]["states"]),
        input_schema_sha256=input_schema_sha256_of(v1_num, v1_flag, v1_uctx, v1_mask),
        schema_self_sha256=s["schema_self_sha256"],
    )


# --------------------------------------------------------------------------------------- reference path
def _z(x: np.ndarray, p: tuple) -> np.ndarray:
    lo, hi, mu, sd = p
    return ((np.clip(x.astype(np.float64), lo, hi) - mu) / sd).astype(np.float32)


def _finite_or_raise(name: str, a: np.ndarray) -> None:
    if not np.all(np.isfinite(a)):
        raise IdentityCheckError(f"RC8: non-finite value in {name}")


def identity_checks_raw(raw: Mapping[str, np.ndarray]) -> dict:
    """RC1-RC7 on stored raw columns (before any drop). Returns the contradiction counts; raises if any > 0."""
    f64 = lambda n: np.asarray(raw[n], dtype=np.float64)  # noqa: E731
    out = {
        "RC1": int(np.count_nonzero(np.abs(f64("log1p_prefix_elapsed_s") - f64("log1p_segment_elapsed_s")) > RC_TOL)),
        "RC2": int(np.count_nonzero(np.abs(f64("prefix_item_count") - (f64("prefix_representative_count") + 1.0))
                                    > RC_TOL)),
        "RC3": int(np.count_nonzero(np.maximum(
            np.abs(f64("local_hour_sin") - (f64("utc_hour_sin") + f64("utc_hour_cos")) / SQRT2),
            np.abs(f64("local_hour_cos") - (f64("utc_hour_cos") - f64("utc_hour_sin")) / SQRT2)) > RC_TOL)),
        "RC4": int(np.count_nonzero(f64("has_history_mask") != f64("has_history"))),
        "RC5": int(np.count_nonzero(f64("prev_user_event_available") != f64("has_history"))),
        "RC6": int(np.count_nonzero((f64("prev_session_available") > 0) & ~(f64("prev_user_event_available") > 0))),
        "RC7": int(np.count_nonzero((f64("recent_category_match_supported") == 0)
                                    & (f64("recent_category_match_share") != 0))
                   + np.count_nonzero((f64("recent_brand_match_supported") == 0)
                                      & (f64("recent_brand_match_share") != 0))),
    }
    bad = {k: v for k, v in out.items() if v}
    if bad:
        raise IdentityCheckError(f"runtime identity checks failed (batch refused): {bad}")
    return out


def encode_rows(raw: Mapping[str, np.ndarray], spec: CleanV2Spec, *, check: bool = True) -> dict:
    """Stored v1 raw columns -> V2 blocks. Rows are canonical representatives (store rows)."""
    need = set(spec.v1_numeric) | set(spec.v1_flags) | set(spec.v1_uctx) | set(spec.v1_masks)
    missing = sorted(need - set(raw))
    if missing:
        raise StateError(f"raw input lacks v1 stored columns {missing}")
    for n in need:
        _finite_or_raise(f"raw {n}", np.asarray(raw[n], dtype=np.float64))
    rc = identity_checks_raw(raw) if check else {}
    n = int(np.asarray(raw[spec.v1_numeric[0]]).shape[0])
    flag = {f: np.asarray(raw[f]).astype(bool) for f in spec.v1_flags}
    pmiss = flag["price_missing"] | flag["price_conflict"]

    num = np.zeros((n, len(spec.v2_numeric)), dtype=np.float32)
    for j, name in enumerate(spec.v2_numeric):
        x = np.asarray(raw[name])
        v = _z(x, spec.norm_v2[name]) if name in spec.norm_v2 else x.astype(np.float64).astype(np.float32)
        if name == "log1p_price":
            v = np.where(pmiss, 0.0, v).astype(np.float32)
        elif name == "log1p_prev_raw_view_gap_s":
            v = np.where(flag["prev_gap_unavailable"], 0.0, v).astype(np.float32)
        elif name == "log1p_item_observed_age_s":
            v = np.where(flag["item_age_unavailable"], 0.0, v).astype(np.float32)
        num[:, j] = v
    flg = np.zeros((n, len(spec.v2_flags)), dtype=np.int8)
    for j, name in enumerate(spec.v2_flags):
        flg[:, j] = flag[name].astype(np.int8)
    m = {k: np.asarray(raw[k], dtype=np.float64) for k in spec.v1_masks}
    uct = np.zeros((n, len(spec.v2_uctx)), dtype=np.float32)
    for j, name in enumerate(spec.v2_uctx):
        x = np.asarray(raw[name])
        v = _z(x, spec.norm_v2[name]) if name in spec.norm_v2 else x.astype(np.float64).astype(np.float32)
        gate = None
        if name == "recent_mean_log_price":
            gate = m["recent_price_supported"] == 0.0
        elif name == "current_minus_recent_mean_log_price":
            gate = (m["recent_price_supported"] == 0.0) | pmiss
        elif name == "log1p_time_since_prev_user_event_s":
            gate = m["prev_user_event_available"] == 0.0
        elif name == "log1p_time_since_prev_session_last_event_s":
            gate = m["prev_session_available"] == 0.0
        if gate is not None:
            v = np.where(gate, 0.0, v).astype(np.float32)
        uct[:, j] = v
    msk = np.zeros((n, len(spec.v2_masks)), dtype=np.float32)
    for j, name in enumerate(spec.v2_masks):
        msk[:, j] = m[name].astype(np.float32)
    for nm, a in (("event_numeric_features", num), ("user_context", uct), ("user_context_masks", msk)):
        _finite_or_raise(nm, a)
    return {"event_numeric_features": num, "event_quality_flags": flg, "user_context": uct,
            "user_context_masks": msk, "rc_counts": rc}


# ------------------------------------------------------------------------------------------ view path
def _np(x):
    try:
        import torch
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy(), True
    except ImportError:                                                   # pragma: no cover
        pass
    return np.asarray(x), False


def identity_checks_v1_batch(num, flags, uctx, masks, real) -> dict:
    """RC1-RC7 on a v1-ENCODED batch, real positions only. num/flags [B,L,*], uctx/masks [B,*], real [B,L]."""
    n64 = num.astype(np.float64)
    r = real.astype(bool)
    rc1 = np.abs(n64[..., 18] - n64[..., 11])[r]
    rc2 = np.abs(n64[..., 14] - n64[..., 13])[r]
    rc3 = np.maximum(np.abs(n64[..., 5] - (n64[..., 1] + n64[..., 2]) / SQRT2),
                     np.abs(n64[..., 6] - (n64[..., 2] - n64[..., 1]) / SQRT2))[r]
    u, mk = uctx.astype(np.float64), masks.astype(np.float64)
    out = {"RC1": int(np.count_nonzero(rc1 > RC_TOL)), "RC2": int(np.count_nonzero(rc2 > RC_TOL)),
           "RC3": int(np.count_nonzero(rc3 > RC_TOL)),
           "RC4": int(np.count_nonzero(mk[:, 0] != u[:, 3])), "RC5": int(np.count_nonzero(mk[:, 4] != u[:, 3])),
           "RC6": int(np.count_nonzero((mk[:, 5] > 0) & ~(mk[:, 4] > 0))),
           "RC7": int(np.count_nonzero((mk[:, 1] == 0) & (u[:, 6] != 0)) + np.count_nonzero((mk[:, 2] == 0)
                                                                                              & (u[:, 7] != 0)))}
    bad = {k: v for k, v in out.items() if v}
    if bad:
        raise IdentityCheckError(f"runtime identity checks failed (batch refused): {bad}")
    return out


V1_VIEW_TAG = "v1_rich"
V2_VIEW_TAG = "clean_v2_rich"


def clean_v2_view(batch: Mapping[str, Any], spec: CleanV2Spec, *, encoded_with: str = "V1_NORMALIZERS",
                  state_sha256: Optional[str] = None, check: bool = True) -> dict:
    """v1-rich batch -> Clean V2 batch. Pure: returns a NEW dict and never mutates `batch`."""
    if batch.get("feature_view") == V2_VIEW_TAG:
        raise StateError("batch is already Clean V2 encoded; a second transform would double-standardize (T11)")
    if batch.get("feature_view") != V1_VIEW_TAG:
        raise StateError("batch carries no v1_rich provenance tag; refusing an unknown encoding state")
    if batch.get("v1_input_schema_sha256") != spec.input_schema_sha256:
        raise StateError("input schema hash mismatch: the v1 column layout of this batch is not the frozen one")
    if encoded_with not in ("V1_NORMALIZERS", "V2_NORMALIZERS"):
        raise StateError(f"unknown encoding {encoded_with!r}")
    num, is_t = _np(batch["event_numeric_features"])
    flags, _ = _np(batch["event_quality_flags"])
    uctx, _ = _np(batch["user_context"])
    masks, _ = _np(batch["user_context_masks"])
    real, _ = _np(batch["attention_mask"])
    lvi, _ = _np(batch["last_valid_index"])
    B = num.shape[0]
    if (num.ndim, flags.ndim, uctx.ndim, masks.ndim) != (3, 3, 2, 2) or num.shape[2] != len(spec.v1_numeric) \
            or flags.shape[2] != len(spec.v1_flags) or uctx.shape[1] != len(spec.v1_uctx) \
            or masks.shape[1] != len(spec.v1_masks):
        raise StateError(f"v1-rich shapes expected, got {num.shape} {flags.shape} {uctx.shape} {masks.shape}")
    for nm, a in (("event_numeric_features", num), ("user_context", uctx), ("user_context_masks", masks)):
        _finite_or_raise(f"input {nm}", a)
    rc = identity_checks_v1_batch(num, flags, uctx, masks, real) if check else {}

    fi = {f: i for i, f in enumerate(spec.v1_flags)}
    mi = {m: i for i, m in enumerate(spec.v1_masks)}
    rows = np.arange(B)
    cur_bad = (flags[rows, lvi, fi["price_missing"]] != 0) | (flags[rows, lvi, fi["price_conflict"]] != 0)
    u_new = uctx.astype(np.float32, copy=True)
    for name, e in spec.reencode.items():
        c = e["uctx_index"]
        if e["gate"] == "F1":
            avail = (masks[:, mi["recent_price_supported"]] != 0) & ~cur_bad
        else:
            avail = masks[:, mi[e["gate"]]] != 0
        z1 = uctx[:, c].astype(np.float64)
        if encoded_with == "V1_NORMALIZERS":
            lo, hi = e["clip"]
            x = np.clip(z1 * e["old"][1] + e["old"][0], lo, hi)             # exact inverse of the v1 z (float64)
            z2 = (x - e["new"][0]) / e["new"][1]
        else:
            z2 = z1
        u_new[:, c] = np.where(avail, z2, 0.0).astype(np.float32)

    out = {k: v for k, v in batch.items()}
    n2 = np.ascontiguousarray(num[..., list(spec.num_src)])
    f2 = np.ascontiguousarray(flags[..., list(spec.flag_src)])
    u2 = np.ascontiguousarray(u_new[:, list(spec.uctx_src)])
    m2 = np.ascontiguousarray(masks[:, list(spec.mask_src)])
    for nm, a in (("V2 event_numeric_features", n2), ("V2 user_context", u2), ("V2 user_context_masks", m2)):
        _finite_or_raise(nm, a)
    if is_t:
        import torch
        n2, f2, u2, m2 = (torch.from_numpy(a) for a in (n2, f2, u2, m2))
    out.update({
        "event_numeric_features": n2, "event_quality_flags": f2, "user_context": u2, "user_context_masks": m2,
        "event_numeric_feature_names": spec.v2_numeric, "event_quality_flag_names": spec.v2_flags,
        "user_context_names": spec.v2_uctx, "user_context_mask_names": spec.v2_masks,
        "feature_view": V2_VIEW_TAG, "clean_v2_schema_self_sha256": spec.schema_self_sha256,
        "clean_v2_state_sha256": state_sha256, "clean_v2_rc_counts": rc,
        "m1_application_path": "EXACT_INVERSE" if encoded_with == "V1_NORMALIZERS" else "INJECTED_NORMALIZERS",
    })
    out.pop("v1_input_schema_sha256", None)
    return out


def reference_formula(name: str, raw_value: np.ndarray, gate_available: np.ndarray, spec: CleanV2Spec) -> np.ndarray:
    """The frozen per-column formula for a re-encoded column (used by the spot-parity test)."""
    lo, hi, mu, sd = spec.norm_v2[name]
    z = (np.clip(np.asarray(raw_value, dtype=np.float64), lo, hi) - mu) / sd
    return np.where(gate_available, z, 0.0).astype(np.float32)


assert SCHEMA_SELF_SHA256  # pinned in common; build_spec reads it from the verified schema
