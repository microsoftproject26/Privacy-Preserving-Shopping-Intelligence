"""feature_view.py (owner: Opus) — feature views, probes, exact weight folding and D3 features.

Scope: SASREC_FEATURE_OPT_V1, executing Fable's scope decision in review/AUTO_DECISION_SCOPE.md.
Everything here operates on the batch tensors the campaign's F4 store already produces
(`event_numeric_features` [B,T,19], `event_quality_flags` [B,T,15], `user_context` [B,12],
`user_context_masks` [B,6], categorical side tokens [B,T]). No canonical artifact is read or
written, and no value is reconstructed from outside the batch.

Column names and orders come from the pipeline code in the snapshot (rees46_eda.loaders/config), never
from memory; tests pin them against FEATURE_GROUPS.json.

Rules implemented (decision items 3–5):
- "Unavailable" encodings are the real pipeline transform applied to the sentinel
  (`Normalizers.apply` + the loader's re-zero rule), never a hand-set 0.0 unless the loader itself
  produces 0.0.
- Duplicated channels move together inside a probe.
- The lean view is an exact re-parametrisation: the baseline's predictions are reproduced up to
  float rounding by folding weights (see `fold_lean_state_dict`).
- D3 features use known item tokens (>= 3) inside the retained window only, which is causal
  (positions <= t). OOV/MISSING never create a repeat.
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch

import fo_env  # noqa: F401  (isolation: snapshot first on sys.path)
from rees46_eda import loaders as L  # snapshot copy
from sasrec_model import SIDE_CHANNELS

NUM_NAMES = list(L.EVENT_NUMERIC_BLOCK)          # 19, model order
FLAG_NAMES = list(L.EVENT_QUALITY_FLAGS)         # 15
UCTX_NAMES = list(L.USER_CONTEXT_FEATURES)       # 12
UMASK_NAMES = list(L.USER_CONTEXT_MASKS)         # 6
NUM = {n: i for i, n in enumerate(NUM_NAMES)}
FLAG = {n: i for i, n in enumerate(FLAG_NAMES)}
UCTX = {n: i for i, n in enumerate(UCTX_NAMES)}
UMASK = {n: i for i, n in enumerate(UMASK_NAMES)}
NORMALIZED_NUM = set(L.NORMALIZED_EVENT_NUMERIC) | set(L.NORMALIZED_SESSION_PREFIX)
NORMALIZED_UCTX = set(L.NORMALIZED_USER_CONTEXT)
PAD, MISSING, KNOWN_START = 0, 1, 3

# ------------------------------------------------------------------ normalizers (pipeline code)
_NORM: Optional[L.Normalizers] = None


def normalizers_path() -> Path:
    return Path(os.environ.get("FO_NORMALIZERS", str(fo_env.NORMALIZERS_PATH)))


def normalizers() -> L.Normalizers:
    global _NORM
    if _NORM is None:
        _NORM = L.Normalizers.load(normalizers_path())
    return _NORM


def enc_event(name: str, raw: float) -> float:
    """Model-input value of event numeric `name` for raw value `raw` (loader path, without re-zero)."""
    if name in NORMALIZED_NUM:
        return float(normalizers().apply(name, np.array([raw], dtype=np.float64))[0])
    return float(np.float32(raw))


def enc_user(name: str, raw: float, supported: Optional[dict] = None) -> float:
    """Model-input value of user-context `name` for `raw`, including the loader's re-zero rule
    (USER_MISSING_FLAG_OF): if the governing support mask is 0 the loader writes 0.0."""
    if name in NORMALIZED_UCTX:
        v = float(normalizers().apply(name, np.array([raw], dtype=np.float64))[0])
    else:
        v = float(np.float32(raw))
    sup = L.USER_MISSING_FLAG_OF.get(name)
    if sup is not None and supported is not None and float(supported.get(sup, 1.0)) == 0.0:
        v = 0.0
    return v


def cold_user_vector() -> tuple[np.ndarray, np.ndarray]:
    """user_context / masks of a genuinely cold user: every raw value 0 and every mask 0, passed
    through the loader transform. The probe runner verifies it against real cold rows."""
    masks = {n: 0.0 for n in UMASK_NAMES}
    uc = np.array([enc_user(n, 0.0, masks) for n in UCTX_NAMES], dtype=np.float32)
    return uc, np.zeros(len(UMASK_NAMES), dtype=np.float32)


# ------------------------------------------------------------------ calendar instant (P8)
DAYPART_BANDS = {3: (0.0, 6.0), 4: (6.0, 12.0), 5: (12.0, 17.0), 6: (17.0, 21.0), 7: (21.0, 24.0)}
LOCAL_SHIFT_H = 3.0   # FEATURE_DICTIONARY: local = UTC + 3h (contextual hypothesis, kept as-is)


def calendar_instant(local_weekday_idx: int, daypart_idx: int) -> dict:
    """All 9 calendar channels for the local midpoint of `daypart_idx` on `local_weekday_idx`
    (Monday=3 .. Sunday=9), using the dictionary formulas. Returns tokens + numeric values."""
    lo, hi = DAYPART_BANDS[int(daypart_idx)]
    h_loc = (lo + hi) / 2.0
    dow_loc = int(local_weekday_idx) - 2                 # isodow 1..7
    h_utc = h_loc - LOCAL_SHIFT_H
    dow_utc = dow_loc
    if h_utc < 0:
        h_utc += 24.0
        dow_utc = (dow_loc - 2) % 7 + 1
    two_pi = 2.0 * math.pi
    return {
        "daypart_tokens": int(daypart_idx), "weekday_tokens": int(local_weekday_idx),
        "utc_hour_sin": math.sin(two_pi * h_utc / 24.0), "utc_hour_cos": math.cos(two_pi * h_utc / 24.0),
        "utc_weekday_sin": math.sin(two_pi * (dow_utc - 1) / 7.0), "utc_weekday_cos": math.cos(two_pi * (dow_utc - 1) / 7.0),
        "local_hour_sin": math.sin(two_pi * h_loc / 24.0), "local_hour_cos": math.cos(two_pi * h_loc / 24.0),
        "local_weekday_sin": math.sin(two_pi * (dow_loc - 1) / 7.0), "local_weekday_cos": math.cos(two_pi * (dow_loc - 1) / 7.0),
        "is_weekend": 1.0 if dow_loc >= 6 else 0.0,
        "_local_hour": h_loc, "_utc_hour": h_utc, "_isodow_local": dow_loc, "_isodow_utc": dow_utc,
    }


# ------------------------------------------------------------------ probes
PROBES = ("P1_ALL_SIDE_OFF", "P2_MAIN_CATEGORY_UNAVAILABLE", "P3_BRAND_UNAVAILABLE",
          "P4_CURRENT_PRICE_UNAVAILABLE", "P5_PREV_GAP_UNAVAILABLE", "P6_ITEM_AGE_UNAVAILABLE",
          "P7_COLD_USER", "P8_CALENDAR_FROZEN")


def _real(kw) -> torch.Tensor:
    return kw["attention_mask"].to(torch.bool)


def _set_event_num(num: torch.Tensor, real: torch.Tensor, name: str, value: float) -> None:
    col = num[..., NUM[name]]
    num[..., NUM[name]] = torch.where(real, torch.full_like(col, value), col)


def _set_event_flag(flg: torch.Tensor, real: torch.Tensor, name: str, value: int) -> None:
    col = flg[..., FLAG[name]]
    flg[..., FLAG[name]] = torch.where(real, torch.full_like(col, value), col)


def _set_token(tok: torch.Tensor, real: torch.Tensor, value: int) -> torch.Tensor:
    return torch.where(real, torch.full_like(tok, value), tok)


def last_position_mask(kw) -> torch.Tensor:
    """[B,T] bool, True only at the decision's current event (index lengths-1)."""
    am = _real(kw)
    B, T = am.shape
    idx = kw["lengths"].to(am.device).long().clamp(min=1) - 1
    return torch.arange(T, device=am.device).unsqueeze(0) == idx.unsqueeze(1)


@dataclass(frozen=True)
class ProbeSpec:
    name: str
    calendar_cell: Optional[tuple[int, int]] = None   # P8: (local_weekday_idx, daypart_idx), registered


def apply_probe(kw: dict, spec: ProbeSpec) -> dict:
    """Return a NEW kwargs dict with the probe's coherent absence applied. Inputs are never
    modified in place, and PAD positions are left untouched (masked out by the model anyway)."""
    name = spec.name
    if name not in PROBES:
        raise KeyError(name)
    out = dict(kw)
    if name == "P1_ALL_SIDE_OFF":
        return out                      # handled by the model wrapper (side_enabled=False)
    real = _real(kw)
    num = kw["event_numeric_features"].clone()
    flg = kw["event_quality_flags"].clone()
    uc = kw["user_context"].clone()
    um = kw["user_context_masks"].clone()
    if name == "P2_MAIN_CATEGORY_UNAVAILABLE":
        out["main_category_tokens"] = _set_token(kw["main_category_tokens"], real, MISSING)
        _set_event_flag(flg, real, "main_category_missing", 1)
        _set_event_flag(flg, real, "category_code_missing", 1)       # exact duplicate moves with it
        _set_event_flag(flg, real, "category_code_recovered", 0)
        for n in ("prefix_known_category_count", "prefix_category_switch_count",
                  "current_known_category_run_length"):
            _set_event_num(num, real, n, enc_event(n, 0.0))
        uc[:, UCTX["recent_distinct_known_main_categories"]] = enc_user("recent_distinct_known_main_categories", 0.0)
        uc[:, UCTX["recent_category_match_share"]] = enc_user("recent_category_match_share", 0.0)
        um[:, UMASK["recent_category_match_supported"]] = 0.0
    elif name == "P3_BRAND_UNAVAILABLE":
        out["brand_tokens"] = _set_token(kw["brand_tokens"], real, MISSING)
        _set_event_flag(flg, real, "brand_missing", 1)
        _set_event_flag(flg, real, "brand_recovered", 0)
        _set_event_flag(flg, real, "brand_conflict", 0)
        uc[:, UCTX["recent_brand_match_share"]] = enc_user("recent_brand_match_share", 0.0)
        um[:, UMASK["recent_brand_match_supported"]] = 0.0
    elif name == "P4_CURRENT_PRICE_UNAVAILABLE":
        cur = last_position_mask(kw) & real
        _set_event_num(num, cur, "log1p_price", 0.0)                 # loader re-zero (price missing)
        _set_event_flag(flg, cur, "price_missing", 1)
        _set_event_flag(flg, cur, "price_zero", 1)                   # exact duplicate moves with it
        _set_event_flag(flg, cur, "price_conflict", 0)
        sup = um[:, UMASK["recent_price_supported"]]
        ccol = uc[:, UCTX["current_minus_recent_mean_log_price"]]
        v_sup = enc_user("current_minus_recent_mean_log_price", 0.0, {"recent_price_supported": 1.0})
        uc[:, UCTX["current_minus_recent_mean_log_price"]] = torch.where(
            sup == 0, torch.zeros_like(ccol), torch.full_like(ccol, v_sup))
    elif name == "P5_PREV_GAP_UNAVAILABLE":
        _set_event_num(num, real, "log1p_prev_raw_view_gap_s", 0.0)  # loader re-zero by the flag
        _set_event_flag(flg, real, "prev_gap_unavailable", 1)
    elif name == "P6_ITEM_AGE_UNAVAILABLE":
        _set_event_num(num, real, "log1p_item_observed_age_s", 0.0)
        _set_event_flag(flg, real, "item_age_unavailable", 1)
    elif name == "P7_COLD_USER":
        cu, cm = cold_user_vector()
        uc = torch.as_tensor(cu, dtype=uc.dtype, device=uc.device).expand_as(uc).clone()
        um = torch.as_tensor(cm, dtype=um.dtype, device=um.device).expand_as(um).clone()
    elif name == "P8_CALENDAR_FROZEN":
        if spec.calendar_cell is None:
            raise ValueError("P8 needs its registered calendar cell")
        c = calendar_instant(*spec.calendar_cell)
        out["daypart_tokens"] = _set_token(kw["daypart_tokens"], real, c["daypart_tokens"])
        out["weekday_tokens"] = _set_token(kw["weekday_tokens"], real, c["weekday_tokens"])
        for n in ("utc_hour_sin", "utc_hour_cos", "utc_weekday_sin", "utc_weekday_cos", "local_hour_sin",
                  "local_hour_cos", "local_weekday_sin", "local_weekday_cos", "is_weekend"):
            _set_event_num(num, real, n, enc_event(n, c[n]))
    out["event_numeric_features"], out["event_quality_flags"] = num, flg
    out["user_context"], out["user_context_masks"] = uc, um
    return out


# ------------------------------------------------------------------ lean view + M1 repair
# M2 exact duplicates (MEASURED_FINDINGS.md) and M3 dead flags. Each dropped column maps to its keeper.
DROP_NUM_AFFINE = {"log1p_prefix_elapsed_s": "log1p_segment_elapsed_s",
                   "prefix_item_count": "prefix_representative_count"}
MEASURED_AFFINE_SHIFT = {"log1p_prefix_elapsed_s": 0.0, "prefix_item_count": 1.0}   # MEASURED_FINDINGS M2 / stream_fit.AFFINE
DROP_NUM_ROTATION = ("local_hour_sin", "local_hour_cos")        # = R(+3h) · (utc_hour_sin, utc_hour_cos)
DROP_FLAG_DUP = {"price_zero": "price_missing", "main_category_missing": "category_code_missing"}
DROP_FLAG_DEAD = ("price_conflict", "brand_conflict", "category_missing", "category_recovered",
                  "category_conflict", "category_code_recovered")
DROP_MASK_DUP = {"has_history_mask": "has_history", "prev_user_event_available": "has_history"}  # mask -> user value
M1_FEATURES = {"log1p_time_since_prev_session_last_event_s": "prev_session_available",
               "log1p_time_since_prev_user_event_s": "prev_user_event_available"}

LEAN_NUM = [n for n in NUM_NAMES if n not in DROP_NUM_AFFINE and n not in DROP_NUM_ROTATION]
LEAN_FLAG = [n for n in FLAG_NAMES if n not in DROP_FLAG_DUP and n not in DROP_FLAG_DEAD]
LEAN_UMASK = [n for n in UMASK_NAMES if n not in DROP_MASK_DUP]


@dataclass(frozen=True)
class M1State:
    """TRAIN valid-only refit for one M1 feature. Clip bounds stay the RECORDED ones, so the
    re-encoding is affine on available rows and the fold is exact."""
    name: str
    mask: str
    clip_lo: float
    clip_hi: float
    old_mean: float
    old_std: float
    new_mean: float
    new_std: float


def m1_states_from_fit(fit: dict) -> dict:
    """`fit`: {feature: {"valid_mean", "valid_std"}} from the exact TRAIN pass (code/stream_fit.py)."""
    spec = normalizers().spec
    out = {}
    for f, m in M1_FEATURES.items():
        p = spec[f]
        s = float(fit[f]["valid_std"])
        if not (s > 1e-12 * max(1.0, abs(float(fit[f]["valid_mean"])))):
            raise ValueError(f"degenerate valid-only std for {f}: {s}")
        out[f] = M1State(f, m, float(p["clip_lo"]), float(p["clip_hi"]), float(p["mean"]), float(p["std"]),
                         float(fit[f]["valid_mean"]), s)
    return out


def to_lean(kw: dict, m1: dict) -> dict:
    """Full F4 kwargs -> lean view (15 numeric + 7 flags; 12 user values with 2 re-encoded; 4 masks)."""
    out = dict(kw)
    num, flg = kw["event_numeric_features"], kw["event_quality_flags"]
    uc, um = kw["user_context"], kw["user_context_masks"]
    out["event_numeric_features"] = num[..., [NUM[n] for n in LEAN_NUM]]
    out["event_quality_flags"] = flg[..., [FLAG[n] for n in LEAN_FLAG]]
    uc2 = uc.clone()
    for f, st in m1.items():
        x = uc[:, UCTX[f]].to(torch.float64)
        avail = um[:, UMASK[st.mask]].to(torch.float64)
        raw_clipped = x * st.old_std + st.old_mean          # exact inverse on available rows
        new = (raw_clipped - st.new_mean) / st.new_std
        uc2[:, UCTX[f]] = torch.where(avail > 0, new, torch.zeros_like(new)).to(uc.dtype)
    out["user_context"] = uc2
    out["user_context_masks"] = um[:, [UMASK[n] for n in LEAN_UMASK]]
    return out


def fold_lean_state_dict(sd: dict, m1: dict) -> dict:
    """Baseline A1 state_dict -> lean-view state_dict with identical function (exact in real
    arithmetic): folds duplicate/rotated/affine columns into their keepers, drops dead inputs, and
    absorbs the M1 re-encoding into user_mlp (weight scale, mask column, bias)."""
    out = {k: v.clone() for k, v in sd.items()}
    W = sd["numflag_proj.weight"].to(torch.float64).clone()   # [32, 19+15]
    b = sd["numflag_proj.bias"].to(torch.float64).clone()
    nN = len(NUM_NAMES)
    col = lambda n: NUM[n]
    fcol = lambda n: nN + FLAG[n]
    spec = normalizers().spec
    for drop, keep in DROP_NUM_AFFINE.items():                # z_drop = a*z_keep + c
        pd, pk = spec[drop], spec[keep]
        if not (pd["clip_lo"] - pk["clip_lo"] == pd["clip_hi"] - pk["clip_hi"]):
            raise ValueError(f"{drop}/{keep}: clip bounds are not a common shift; fold would be inexact")
        shift = pd["clip_lo"] - pk["clip_lo"]                  # raw_drop = raw_keep + shift
        if shift != MEASURED_AFFINE_SHIFT[drop]:               # Review C LOW fix: cross-check the measured relation
            raise ValueError(f"{drop}: clip-inferred shift {shift} != measured shift {MEASURED_AFFINE_SHIFT[drop]}")
        a = pk["std"] / pd["std"]
        c = (pk["mean"] + shift - pd["mean"]) / pd["std"]
        W[:, col(keep)] += a * W[:, col(drop)]
        b += c * W[:, col(drop)]
    phi = 2.0 * math.pi * LOCAL_SHIFT_H / 24.0                 # ls = cos*us + sin*uc ; lc = -sin*us + cos*uc
    wls, wlc = W[:, col("local_hour_sin")].clone(), W[:, col("local_hour_cos")].clone()
    W[:, col("utc_hour_sin")] += math.cos(phi) * wls - math.sin(phi) * wlc
    W[:, col("utc_hour_cos")] += math.sin(phi) * wls + math.cos(phi) * wlc
    for drop, keep in DROP_FLAG_DUP.items():
        W[:, fcol(keep)] += W[:, fcol(drop)]
    keep_cols = [col(n) for n in LEAN_NUM] + [fcol(n) for n in LEAN_FLAG]
    out["numflag_proj.weight"] = W[:, keep_cols].to(sd["numflag_proj.weight"].dtype)
    out["numflag_proj.bias"] = b.to(sd["numflag_proj.bias"].dtype)

    U = sd["user_mlp.weight"].to(torch.float64).clone()        # [64, 12+6]
    ub = sd["user_mlp.bias"].to(torch.float64).clone()
    nU = len(UCTX_NAMES)
    mcol = lambda n: nU + UMASK[n]
    for f, st in m1.items():                                   # old x = a x' + (c - z0)*mask + z0
        a = st.new_std / st.old_std
        c = (st.new_mean - st.old_mean) / st.old_std
        z0 = (min(max(0.0, st.clip_lo), st.clip_hi) - st.old_mean) / st.old_std
        wf = U[:, UCTX[f]].clone()
        U[:, UCTX[f]] = a * wf
        U[:, mcol(st.mask)] += (c - z0) * wf
        ub += z0 * wf
    for drop, keep in DROP_MASK_DUP.items():                   # masks identical to has_history
        U[:, UCTX[keep]] += U[:, mcol(drop)]
    ucols = list(range(nU)) + [mcol(n) for n in LEAN_UMASK]
    out["user_mlp.weight"] = U[:, ucols].to(sd["user_mlp.weight"].dtype)
    out["user_mlp.bias"] = ub.to(sd["user_mlp.bias"].dtype)
    return out


# ------------------------------------------------------------------ D3 (candidate C1)
D3_NAMES = ("d3_log_prior_count", "d3_log_steps_since_prev", "d3_has_prev", "d3_known_distinct_ratio",
            "d3_current_item_known")
LOG50 = math.log(50.0)


def d3_features(item_tokens: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """[B,T,5] causal repeat features of the CURRENT observed product at each real position t,
    computed only from known tokens (>= 3) at window positions <= t. PAD rows are 0."""
    real = attention_mask.to(torch.bool)
    tok = item_tokens.long()
    B, T = tok.shape
    known = real & (tok >= KNOWN_START)
    eq = (tok.unsqueeze(2) == tok.unsqueeze(1))                               # [B,t,s]
    strict_past = torch.ones(T, T, dtype=torch.bool, device=tok.device).tril(-1)
    prev = eq & strict_past & known.unsqueeze(2) & known.unsqueeze(1)
    prior = prev.sum(-1).to(torch.float32)                                   # occurrences before t
    s_idx = torch.arange(T, device=tok.device).view(1, 1, T).expand(B, T, T)
    last = torch.where(prev, s_idx, torch.full_like(s_idx, -1)).amax(-1)     # last s<t with same token
    has_prev = last >= 0
    t_idx = torch.arange(T, device=tok.device).view(1, T).expand(B, T)
    steps = torch.where(has_prev, (t_idx - last).to(torch.float32), torch.zeros(B, T, device=tok.device))
    # distinct known tokens among positions <= t: token at s counts once, at its first occurrence
    first_occ = known & ~(eq & strict_past & known.unsqueeze(1)).any(-1)     # no equal known token before
    distinct = torch.cumsum(first_occ.to(torch.float32), dim=1)
    ratio = distinct / (t_idx.to(torch.float32) + 1.0)
    f = torch.stack([torch.log1p(prior) / LOG50,
                     torch.where(has_prev, torch.log1p(steps) / LOG50, torch.zeros_like(steps)),
                     has_prev.to(torch.float32),
                     ratio,
                     known.to(torch.float32)], dim=-1)
    return torch.where(real.unsqueeze(-1), f, torch.zeros_like(f))


def with_d3(kw: dict) -> dict:
    out = dict(kw)
    d3 = d3_features(kw["item_tokens"], kw["attention_mask"]).to(kw["event_numeric_features"].dtype)
    out["event_numeric_features"] = torch.cat([kw["event_numeric_features"], d3], dim=-1)
    return out


def d3_state_dict(sd: dict) -> dict:
    """Baseline state_dict -> C1 model with numflag_proj [32, 19+5+15]; new columns are ZERO so the
    candidate equals the control exactly at initialisation."""
    out = {k: v.clone() for k, v in sd.items()}
    W = sd["numflag_proj.weight"]
    nN = len(NUM_NAMES)
    out["numflag_proj.weight"] = torch.cat([W[:, :nN], torch.zeros(W.shape[0], len(D3_NAMES), dtype=W.dtype),
                                            W[:, nN:]], dim=1)
    return out


def schema_record() -> dict:
    return {"numeric": NUM_NAMES, "flags": FLAG_NAMES, "user_context": UCTX_NAMES, "user_masks": UMASK_NAMES,
            "side_channels": list(SIDE_CHANNELS), "lean_numeric": LEAN_NUM, "lean_flags": LEAN_FLAG,
            "lean_user_masks": LEAN_UMASK, "d3": list(D3_NAMES), "probes": list(PROBES)}
