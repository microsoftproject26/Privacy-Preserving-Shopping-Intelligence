"""Per-round update-norm monitoring and the no-op criteria for warm-started federated runs.

Monitoring only (descriptive; it explains a selection outcome, it never selects). Nothing here changes a tensor,
draws a random number or runs a forward pass: every quantity is computed from values the round already has.

One record per server step (`UpdateNormObserver`, attached as `server.Server.observer`; the server calls it
inside `Server.apply` with (theta_r, the aggregate it was given, theta_{r+1} after the server step and the family's
post-aggregate step)). Over the TRAINED subset T (= the manifest's shared keys = the uploaded keys; frozen item tables
are FIXED buffers and never in T), per tensor group (`group_of`) and in total:
  agg_delta_l2   ||aggregate - theta_r||_2            (Delta_agg; for DP runs the noisy fixed-denominator mean, i.e. the
                                                       post-noise released quantity — never a per-client value)
  step_l2        ||theta_{r+1} - theta_r||_2          (the server step: optimiser + SASRec recentering)
  pre_dist_l2    ||theta_{r+1} - theta_pre||_2        (theta_pre = the run's start, captured before any round / resume)
  pre_l2         ||theta_pre||_2 (constant; the denominator of rel_pre_dist = pre_dist_l2 / pre_l2)
  cos_prev       cos(Delta_agg,r, Delta_agg,r-1) over T (None for the first logged step and after a resume)
Squares are summed in float64 from the float32 tensors; a non-finite result is recorded as None with finite False.

No-op criteria (fixed before the warm-start runs; did federated training actually move a pretrained model?):
`param_noop_criterion` (a) and `functional_noop_criterion` (b) are pure functions; `classify_noop` combines them
(NO_OP iff (a) and (b) hold at the best checkpoint (PRACTICAL_BEST) and at endpoint_6.0; MOVED otherwise;
DRIFT = MOVED with calibration-set quality <= that of the pretrained start). Callers read checkpoints / probe rows
and pass the values in.
"""
from __future__ import annotations

import math
from collections import OrderedDict
from collections.abc import Mapping, Sequence

import torch
from torch import Tensor

ITEM_TABLE_KEYS = ("item_embed.weight", "output_embed", "output_bias")   # = runtime.FROZEN_ITEM_KEYS
BLOCK_GROUPS = ("qkv", "out", "ff1", "ff2")                               # per SASRec CausalBlock
LAYERNORM_MODULES = ("ln1", "ln2", "final_norm", "input_norm", "side_norm", "user_norm")
GATE_KEYS = ("side_gate", "user_gate")
# no-op thresholds
NOOP_REL = 0.01                     # (a) relative distance on T < 1 %, and no large group moved >= 1 %
NOOP_GROUP_MIN_NUMEL = 4096         # (a) "tensor group with >= 4,096 elements"
NOOP_JACCARD_MIN = 0.95             # (b) mean top-20 Jaccard >= 0.95
NOOP_TOPK = 20
PROBE_ROWS = 4096                   # (b) probe set size (calibration-set query rows)
PROBE_SEED = 20261004               # (b) seeded probe selection
VERDICTS = ("NO_OP", "MOVED", "DRIFT", "INCOMPLETE")


def group_of(key: str) -> str:
    """The tensor group of a canonical state key: 'block<i>.<qkv|out|ff1|ff2>' per CausalBlock, 'layernorm' (every
    LayerNorm), 'gates' (side / user gates), 'item_tables' (item_embed.weight, output_embed, output_bias), else the
    top-level module name ('pos_embed', 'side_proj', 'side_embeds', 'numflag_proj', 'user_mlp', 'user_proj', ...)."""
    if key in ITEM_TABLE_KEYS:
        return "item_tables"
    parts = key.split(".")
    if parts[0] == "blocks" and len(parts) >= 3 and parts[1].isdigit():
        sub = parts[2]
        return "layernorm" if sub in LAYERNORM_MODULES else f"block{int(parts[1])}.{sub}"
    if parts[0] in LAYERNORM_MODULES:
        return "layernorm"
    if key in GATE_KEYS:
        return "gates"
    return parts[0]


def _sq(t: Tensor) -> float:
    return float(torch.sum(torch.square(t.detach().to(torch.float64))))


def _dot(a: Tensor, b: Tensor) -> float:
    return float(torch.sum(a.detach().to(torch.float64) * b.detach().to(device=a.device, dtype=torch.float64)))


def _fin(x: float | None) -> float | None:
    return x if x is None or math.isfinite(x) else None


def pre_squares(theta_pre_T: Mapping[str, Tensor]) -> dict:
    """{key: ||theta_pre[key]||^2} (float64 sums), computed once per run."""
    return {k: _sq(v) for k, v in theta_pre_T.items()}


def round_update_record(theta_r: Mapping[str, Tensor], agg_state: Mapping[str, Tensor],
                        theta_new: Mapping[str, Tensor], theta_pre_T: Mapping[str, Tensor],
                        prev_delta: Mapping[str, Tensor] | None = None,
                        pre_sq: Mapping[str, float] | None = None) -> tuple:
    """(record, delta): the record of one server step over T = the keys of `theta_pre_T` (see the module
    docstring) and Delta_agg (float32, detached clones) for the next step's cos_prev. Read-only on every input."""
    keys = list(theta_pre_T)
    pre_sq = pre_squares(theta_pre_T) if pre_sq is None else pre_sq
    groups: OrderedDict[str, dict] = OrderedDict()
    delta = OrderedDict()
    tot = {"numel": 0, "agg": 0.0, "step": 0.0, "pre_dist": 0.0, "pre": 0.0}
    dot = sq_prev = 0.0
    for k in keys:
        th = theta_r[k].detach()
        d = torch.sub(agg_state[k].detach().to(th.device), th)
        st = torch.sub(theta_new[k].detach().to(th.device), th)
        pdist = torch.sub(theta_new[k].detach().to(th.device), theta_pre_T[k].detach().to(th.device))
        g = groups.setdefault(group_of(k), {"numel": 0, "agg": 0.0, "step": 0.0, "pre_dist": 0.0, "pre": 0.0})
        vals = {"numel": int(th.numel()), "agg": _sq(d), "step": _sq(st), "pre_dist": _sq(pdist), "pre": float(pre_sq[k])}
        for acc in (g, tot):
            for f, v in vals.items():
                acc[f] += v
        if prev_delta is not None:
            p = prev_delta[k].to(d.device)
            dot += _dot(d, p)
            sq_prev += _sq(p)
        delta[k] = d.clone()

    def norms(a: dict) -> dict:
        out = {"numel": a["numel"]}
        for f, name in (("agg", "agg_delta_l2"), ("step", "step_l2"), ("pre_dist", "pre_dist_l2"), ("pre", "pre_l2")):
            out[name] = _fin(math.sqrt(a[f]) if math.isfinite(a[f]) else float("nan"))
        out["rel_pre_dist"] = (_fin(out["pre_dist_l2"] / out["pre_l2"])
                               if out["pre_dist_l2"] is not None and out["pre_l2"] else None)
        return out

    cos = None
    if prev_delta is not None and tot["agg"] > 0 and sq_prev > 0:
        cos = _fin(dot / math.sqrt(tot["agg"] * sq_prev))
    rec = {"total_T": norms(tot), "groups": OrderedDict((g, norms(a)) for g, a in groups.items()),
           "cos_prev": cos, "cos_prev_available": prev_delta is not None, "n_keys_T": len(keys)}
    rec["finite"] = all(v is not None for v in (rec["total_T"]["agg_delta_l2"], rec["total_T"]["step_l2"],
                                                rec["total_T"]["pre_dist_l2"]))
    return rec, delta


class UpdateNormObserver:
    """A `server.Server.observer`. `theta_pre` = the run's start state (any superset of T); `keys` = T (the
    manifest's shared keys after any freeze). Each call computes one record (round_update_record) and keeps it until
    `pop()`; Delta_agg of the last call is kept for cos_prev (in memory only: a resumed run restarts it)."""

    def __init__(self, theta_pre: Mapping[str, Tensor], keys: Sequence[str]):
        self.keys = tuple(keys)
        missing = [k for k in self.keys if k not in theta_pre]
        if missing:
            raise KeyError(f"theta_pre lacks trained keys {missing[:5]}")
        self.theta_pre_T = OrderedDict((k, theta_pre[k].detach().clone()) for k in self.keys)
        self.pre_sq = pre_squares(self.theta_pre_T)
        self.prev_delta: dict | None = None
        self.last: dict | None = None
        self.calls = 0

    def __call__(self, theta_r: Mapping[str, Tensor], agg_state: Mapping[str, Tensor],
                 theta_new: Mapping[str, Tensor]) -> None:
        rec, delta = round_update_record(theta_r, agg_state, theta_new, self.theta_pre_T, self.prev_delta, self.pre_sq)
        if self.last is not None:                          # two server steps without a pop: never silently merged
            raise RuntimeError("UpdateNormObserver: a record was not collected before the next server step")
        self.prev_delta, self.last = delta, rec
        self.calls += 1

    def pop(self) -> dict | None:
        rec, self.last = self.last, None
        return rec


def noop_round_record() -> dict:
    """The record of a round without a server step (0 survivors / empty sweep round): theta unchanged."""
    return {"server_step": False}


# ------------------------------------------------------------------------------------------------ no-op criteria (a) / (b)
def param_noop_criterion(theta: Mapping[str, Tensor], theta_pre: Mapping[str, Tensor], keys: Sequence[str]) -> dict:
    """(a): rel = ||theta_T - theta_pre,T|| / ||theta_pre,T|| < 1 % AND no group with >= 4,096 elements moved by >= 1 %
    relative (group rel = ||delta_g|| / ||theta_pre,g||). Returns the per-group table and `holds`."""
    keys = list(keys)
    sq_d, sq_p, groups = 0.0, 0.0, OrderedDict()
    for k in keys:
        d, p = _sq(theta[k].to(torch.float32) - theta_pre[k].to(torch.float32)), _sq(theta_pre[k])
        g = groups.setdefault(group_of(k), {"numel": 0, "sq_delta": 0.0, "sq_pre": 0.0})
        g["numel"] += int(theta_pre[k].numel())
        g["sq_delta"] += d
        g["sq_pre"] += p
        sq_d += d
        sq_p += p
    table, large_moved = OrderedDict(), []
    for name, g in groups.items():
        rel = math.sqrt(g["sq_delta"]) / math.sqrt(g["sq_pre"]) if g["sq_pre"] > 0 else (
            0.0 if g["sq_delta"] == 0 else float("inf"))
        large = g["numel"] >= NOOP_GROUP_MIN_NUMEL
        table[name] = {"numel": g["numel"], "rel": _fin(rel), "large": large, "moved": rel >= NOOP_REL}
        if large and rel >= NOOP_REL:
            large_moved.append(name)
    rel_T = math.sqrt(sq_d) / math.sqrt(sq_p) if sq_p > 0 else (0.0 if sq_d == 0 else float("inf"))
    return {"criterion": "a", "rel_T": _fin(rel_T), "threshold": NOOP_REL, "group_min_numel": NOOP_GROUP_MIN_NUMEL,
            "groups": table, "large_groups_moved": large_moved, "n_keys_T": len(keys),
            "numel_T": sum(int(theta_pre[k].numel()) for k in keys),
            "holds": bool(rel_T < NOOP_REL and not large_moved)}


def functional_rows(scores: Tensor, scores_pre: Tensor, k: int = NOOP_TOPK) -> dict:
    """Per-row (b) statistics of one chunk of probe rows (rows x K logits of theta and theta_pre on the SAME query rows;
    no labels): top-k Jaccard of the two rankings, |s_theta[i*] - s_pre[i*]| with i* = argmax s_pre, and theta_pre's
    top-1 / top-2 score gap. float64 CPU tensors; ties in top-k follow torch.topk."""
    if scores.shape != scores_pre.shape or scores.dim() != 2:
        raise ValueError(f"score matrices must be equal-shape 2-D, got {tuple(scores.shape)} / {tuple(scores_pre.shape)}")
    a = scores.detach().to(torch.float64)
    b = scores_pre.detach().to(torch.float64)
    ta = torch.topk(a, k, dim=1).indices
    tb = torch.topk(b, k, dim=1).indices
    inter = (ta.unsqueeze(2) == tb.unsqueeze(1)).any(dim=2).sum(dim=1).to(torch.float64)
    top2 = torch.topk(b, 2, dim=1)
    istar = top2.indices[:, 0]
    dscore = (a.gather(1, istar[:, None]) - b.gather(1, istar[:, None])).abs().squeeze(1)
    return {"jaccard": (inter / (2 * k - inter)).cpu(), "dscore": dscore.cpu(),
            "gap": (top2.values[:, 0] - top2.values[:, 1]).cpu()}


def functional_noop_from_rows(rows: Mapping[str, Tensor], k: int = NOOP_TOPK) -> dict:
    """(b) from the per-row statistics of the whole probe set (functional_rows, chunks concatenated): mean top-k Jaccard
    >= 0.95 AND mean |Delta score| of theta_pre's top-1 item < the median top-1 / top-2 gap of theta_pre."""
    jac, dsc, gap = rows["jaccard"], rows["dscore"], rows["gap"]
    if not (jac.numel() == dsc.numel() == gap.numel() > 0):
        raise ValueError("functional rows must be non-empty and of equal length")
    mean_j, mean_d, med_gap = float(jac.mean()), float(dsc.mean()), float(gap.median())
    return {"criterion": "b", "rows": int(jac.numel()), "k": int(k), "mean_jaccard": mean_j,
            "jaccard_threshold": NOOP_JACCARD_MIN, "mean_abs_dscore_top1": mean_d, "median_top1_top2_gap_pre": med_gap,
            "top1_definition": "i* = argmax theta_pre scores; |s_theta[i*] - s_pre[i*]|",
            "holds": bool(mean_j >= NOOP_JACCARD_MIN and mean_d < med_gap)}


def functional_noop_criterion(scores: Tensor, scores_pre: Tensor, k: int = NOOP_TOPK) -> dict:
    """(b) on one probe set given as two full score matrices (see functional_rows / functional_noop_from_rows)."""
    return functional_noop_from_rows(functional_rows(scores, scores_pre, k), k)


def classify_noop(points: Mapping[str, Mapping], *, calib_value: float | None = None,
                  pre_calib: float | None = None) -> dict:
    """points = {"PRACTICAL_BEST": {"a": <param record>, "b": <functional record or None>},
    "CONTROLLED_BUDGET": {...}} (both required). NO_OP iff (a) and (b) hold at both; MOVED if (a) or (b) fails at either;
    a MOVED run with calib_value <= pre_calib is DRIFT (only when both values are given); INCOMPLETE when nothing failed
    but some (b) is missing."""
    need = ("PRACTICAL_BEST", "CONTROLLED_BUDGET")
    if sorted(points) != sorted(need):
        raise ValueError(f"classify_noop needs exactly {need}, got {sorted(points)}")
    fails, missing = [], []
    for name in need:
        for c in ("a", "b"):
            rec = points[name].get(c)
            if rec is None:
                missing.append(f"{name}:{c}")
            elif not rec["holds"]:
                fails.append(f"{name}:{c}")
    if fails:
        verdict = "MOVED"
        if calib_value is not None and pre_calib is not None and float(calib_value) <= float(pre_calib):
            verdict = "DRIFT"
    else:
        verdict = "INCOMPLETE" if missing else "NO_OP"
    return {"verdict": verdict, "failed": fails, "missing": missing, "calib_value": calib_value,
            "pre_calib": pre_calib, "drift_rule": "MOVED and calibration quality <= the pretrained start's"}
