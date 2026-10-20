"""v4int.renorm (owner: Data/integration engineer, DATA-05; DECISION_LOG 42) — the optional `transform_state` of the
providers: re-normalize the 17 z-normalized Clean V2 channels with another TRANSFORM_STATE's normalizers.

NORMALIZERS ONLY. Everything else is unchanged: the item-side tables (catalogue, vocabularies and the first_seen_train
table that drives item age and its item_age_unavailable flag), every flag and mask, the rezero map, the clip rule, the
tokens and the windows. Only the (clip_lo, clip_hi, mean, std) of the 17 channels change.

The default (no state given) is the Clean V2 state: the batch is returned untouched, so it is bitwise the pre-DATA-05
behaviour. When a state file is given:
  * the file is strict JSON with a self-hash `transform_state_sha256` (the v4data.transform_state rule: sha256 of the
    canonical JSON without that field), verified; when it carries `schema_self_sha256` / `v1_version_hash` they must
    be the frozen ones; `normalized_channels_v2` must name exactly the 17 channels of the frozen schema, each with finite
    clip_lo < clip_hi, a finite mean and std > 0;
  * per channel, the served z (encoded with the Clean V2 values: the v1 normalizers for the 14 v1-rule channels, which
    are bit-identical to FEATURE_SCHEMA_CLEAN_V2 normalized_channels_v2, and the V2 values for the 3 re-encoded ones)
    is inverted exactly as v4data.transform.clean_v2_view's EXACT_INVERSE does, in float64:
        x = clip(z * std_cv2 + mean_cv2, clip_lo_new, clip_hi_new);  z_new = float32((x - mean_new) / std_new)
    and re-zeroed by the frozen gates (rezero_map_v2): log1p_price by price_missing | price_conflict,
    log1p_prev_raw_view_gap_s by prev_gap_unavailable, log1p_item_observed_age_s by item_age_unavailable (the flag is
    unchanged), recent_mean_log_price by recent_price_supported == 0, current_minus_recent_mean_log_price by
    recent_price_supported == 0 OR the decision event's price_missing | price_conflict (F-1),
    log1p_time_since_prev_user_event_s by prev_user_event_available == 0 (== has_history, runtime check RC5), and
    log1p_time_since_prev_session_last_event_s by prev_session_available == 0; padded positions stay 0.
  * With the Clean V2 values passed explicitly the arithmetic is an exact identity on float32 inputs, so the output is
    bitwise the default (tested).

Exactness limit (disclosed in `record()`): the served z is the float32 encoding of clip(raw, clip_lo_cv2, clip_hi_cv2).
Inside the Clean V2 clip range the inverse returns the raw value up to that float32 rounding. A state whose clip range
EXTENDS beyond the Clean V2 range on a channel cannot recover values beyond the Clean V2 bound (they arrive saturated
at it); such channels are listed in `clip_extends_beyond_clean_v2`.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping, Optional

import numpy as np

from v4data.common import SCHEMA_SELF_SHA256, VERSION_HASH, self_hash, sha256_file


class TransformStateError(ValueError):
    """An unusable transform state (self-hash, schema, dataset or normalizer block)."""


def _strict_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"),
                      parse_constant=lambda c: (_ for _ in ()).throw(TransformStateError(f"non-strict constant {c}")))


def load_transform_state(path, spec) -> dict:
    """Load and verify a TRANSFORM_STATE file; returns {name: (clip_lo, clip_hi, mean, std)} plus provenance."""
    p = Path(path)
    if not p.is_file():
        raise TransformStateError(f"transform state {p} does not exist")
    st = _strict_json(p)
    if not isinstance(st, dict):
        raise TransformStateError("transform state is not a JSON object")
    rec = st.get("transform_state_sha256")
    if not rec or self_hash(st, "transform_state_sha256") != rec:
        raise TransformStateError(f"{p.name}: transform_state_sha256 self-hash does not verify")
    if "schema_self_sha256" in st and st["schema_self_sha256"] != SCHEMA_SELF_SHA256:
        raise TransformStateError(f"{p.name}: built for another feature schema")
    if "v1_version_hash" in st and st["v1_version_hash"] != VERSION_HASH:
        raise TransformStateError(f"{p.name}: built for another dataset version")
    block = st.get("normalized_channels_v2")
    if not isinstance(block, dict) or set(block) != set(spec.norm_v2):
        got = sorted(block) if isinstance(block, dict) else None
        raise TransformStateError(f"{p.name}: normalized_channels_v2 must name exactly the 17 channels "
                                  f"{sorted(spec.norm_v2)}, got {got}")
    norm = {}
    for name, e in block.items():
        try:
            lo, hi, mu, sd = (float(e["clip_lo"]), float(e["clip_hi"]), float(e["mean"]), float(e["std"]))
        except (KeyError, TypeError, ValueError) as ex:
            raise TransformStateError(f"{p.name}: channel {name} lacks clip_lo / clip_hi / mean / std") from ex
        if not all(np.isfinite([lo, hi, mu, sd])) or not lo < hi or not sd > 0.0:
            raise TransformStateError(f"{p.name}: channel {name} has unusable values {(lo, hi, mu, sd)}")
        norm[name] = (lo, hi, mu, sd)
    return {"norm": norm, "transform_state_sha256": rec, "file_sha256": sha256_file(p), "path": str(p),
            "fit": st.get("fit"), "status": st.get("status")}


def _np(x):
    import torch
    return (x.detach().cpu().numpy(), True) if isinstance(x, torch.Tensor) else (np.asarray(x), False)


class Renormalizer:
    """Applies a transform state's 17 normalizers to Clean V2 batches (pure: returns a new dict)."""

    def __init__(self, path, spec):
        self.spec = spec
        loaded = load_transform_state(path, spec)
        self.new = loaded["norm"]
        self.old = {k: tuple(float(v) for v in spec.norm_v2[k]) for k in spec.norm_v2}
        self.state_sha256 = loaded["transform_state_sha256"]
        self.file_sha256 = loaded["file_sha256"]
        self.path = loaded["path"]
        self.identity = all(self.new[k] == self.old[k] for k in self.old)
        self.clip_extends = sorted(k for k in self.old if self.new[k][0] < self.old[k][0]
                                   or self.new[k][1] > self.old[k][1])
        self._num = [(j, n) for j, n in enumerate(spec.v2_numeric) if n in self.old]
        self._uctx = [(j, n) for j, n in enumerate(spec.v2_uctx) if n in self.old]
        if len(self._num) + len(self._uctx) != len(self.old):
            raise TransformStateError("the 17 normalized channels are not all in the V2 layout")
        f = {n: i for i, n in enumerate(spec.v2_flags)}
        m = {n: i for i, n in enumerate(spec.v2_masks)}
        u = {n: i for i, n in enumerate(spec.v2_uctx)}
        self._fi, self._mi, self._ui = f, m, u

    def record(self) -> dict:
        return {"transform_state_sha256": self.state_sha256, "file_sha256": self.file_sha256, "path": self.path,
                "scope": "NORMALIZERS_ONLY (17 channels; item-side tables, flags, masks and rezero map unchanged)",
                "identity_with_clean_v2": self.identity, "clip_extends_beyond_clean_v2": self.clip_extends,
                "method": "EXACT_INVERSE of the served Clean V2 z (float64), clip, z-score with the state, rezero"}

    def _z(self, z32: np.ndarray, name: str) -> np.ndarray:
        lo_o, hi_o, mu_o, sd_o = self.old[name]
        lo, hi, mu, sd = self.new[name]
        x = np.clip(z32.astype(np.float64) * sd_o + mu_o, lo, hi)
        return ((x - mu) / sd).astype(np.float32)

    def apply(self, batch: Mapping) -> dict:
        import torch
        num, is_t = _np(batch["event_numeric_features"])
        flags, _ = _np(batch["event_quality_flags"])
        uctx, _ = _np(batch["user_context"])
        masks, _ = _np(batch["user_context_masks"])
        real, _ = _np(batch["attention_mask"])
        lvi, _ = _np(batch["last_valid_index"])
        real = real.astype(bool)
        fi, mi, ui = self._fi, self._mi, self._ui
        pmiss = (flags[..., fi["price_missing"]] != 0) | (flags[..., fi["price_conflict"]] != 0)
        ev_gate = {"log1p_price": pmiss, "log1p_prev_raw_view_gap_s": flags[..., fi["prev_gap_unavailable"]] != 0,
                   "log1p_item_observed_age_s": flags[..., fi["item_age_unavailable"]] != 0}
        n2 = num.astype(np.float32, copy=True)
        for j, name in self._num:
            gate = ~real
            if name in ev_gate:
                gate = gate | ev_gate[name]
            n2[..., j] = np.where(gate, np.float32(0.0), self._z(num[..., j], name))
        rows = np.arange(uctx.shape[0])
        cur_bad = pmiss[rows, lvi]
        no_price = masks[:, mi["recent_price_supported"]] == 0
        u_gate = {"recent_mean_log_price": no_price, "current_minus_recent_mean_log_price": no_price | cur_bad,
                  "log1p_time_since_prev_user_event_s": uctx[:, ui["has_history"]] == 0,
                  "log1p_time_since_prev_session_last_event_s": masks[:, mi["prev_session_available"]] == 0}
        u2 = uctx.astype(np.float32, copy=True)
        for j, name in self._uctx:
            z = self._z(uctx[:, j], name)
            u2[:, j] = np.where(u_gate[name], np.float32(0.0), z) if name in u_gate else z
        for nm, a in (("event_numeric_features", n2), ("user_context", u2)):
            if not np.all(np.isfinite(a)):
                raise TransformStateError(f"non-finite {nm} after re-normalization")
        out = dict(batch)
        out["event_numeric_features"] = torch.from_numpy(np.ascontiguousarray(n2)) if is_t else n2
        out["user_context"] = torch.from_numpy(np.ascontiguousarray(u2)) if is_t else u2
        out["clean_v2_state_sha256"] = self.state_sha256                  # provenance: the state actually applied
        return out


def resolve_transform_state(transform_state, spec, default_state_sha256: Optional[str]):
    """(Renormalizer or None, record). None = the default Clean V2 state (the batch is untouched)."""
    if transform_state is None:
        return None, {"transform_state_sha256": default_state_sha256, "source": "DEFAULT_CLEAN_V2 (the locked state)",
                      "scope": "NORMALIZERS_ONLY", "identity_with_clean_v2": True, "clip_extends_beyond_clean_v2": []}
    r = Renormalizer(transform_state, spec)
    return r, dict(r.record(), source="transform_state argument")


__all__ = ["Renormalizer", "TransformStateError", "load_transform_state", "resolve_transform_state"]
