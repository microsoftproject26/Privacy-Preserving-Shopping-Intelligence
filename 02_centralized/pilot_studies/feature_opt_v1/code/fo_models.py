"""fo_models.py (owner: Opus) — model construction for the feature study.

FOModel wraps one SASRecCE. To the audited evaluator and training adapters it presents the full F4
interface (the same four attributes that decide which batch keys a model receives), so the batch is
selected exactly as for the campaign model. Inside forward it applies:
- the registered probe (coherent absence), which also serves as the adapter of a removal candidate;
- a view: "full" (baseline), "lean" (M2/M3 drops plus the M1 repair, with folded weights) or "d3"
  (C1: five causal repeat channels, zero-initialised).

Checkpoints always load with strict=True. When the shapes change, the new and remapped parameters
are enumerated explicitly by `parameter_mapping`.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
from torch import nn

import fo_env  # noqa: F401
import feature_view as fv
from sasrec_model import SASRecConfig, SASRecCE, SIDE_CHANNELS, build_sasrec

VIEWS = ("full", "lean", "d3")


def load_weights_file(path) -> tuple[dict, dict]:
    obj = torch.load(Path(path), map_location="cpu", weights_only=True)
    if isinstance(obj, dict) and "state_dict" in obj:
        return obj["state_dict"], {k: v for k, v in obj.items() if k != "state_dict"}
    return obj, {}


def _resize_inputs(inner: SASRecCE, numflag_in: int, user_in: int) -> None:
    if inner.numflag_proj.in_features != numflag_in:
        inner.numflag_proj = nn.Linear(numflag_in, inner.numflag_proj.out_features)
    if inner.user_mlp.in_features != user_in:
        inner.user_mlp = nn.Linear(user_in, inner.user_mlp.out_features)


class FOModel(nn.Module):
    def __init__(self, inner: SASRecCE, *, view: str = "full", probe: Optional[fv.ProbeSpec] = None,
                 m1: Optional[dict] = None):
        super().__init__()
        if view not in VIEWS:
            raise ValueError(view)
        if not inner.cfg.rich:
            raise ValueError("the feature study wraps the rich (A1) architecture only")
        if view == "lean" and not m1:
            raise ValueError("lean view needs the M1 transform state")
        self.inner, self.view, self.probe, self.m1 = inner, view, probe, m1
        # the evaluator / predictor_kwargs contract: receive the full F4 batch
        self._categorical_channels = ("item_tokens",) + tuple(SIDE_CHANNELS)
        self.numeric_dim, self.flags_dim, self.use_user_context = 19, 15, True
        self.K = inner.K

    def recenter_output_(self) -> None:
        self.inner.recenter_output_()

    def forward(self, **kw):
        if self.probe is not None:
            kw = fv.apply_probe(kw, self.probe)
        if self.view == "lean":
            kw = fv.to_lean(kw, self.m1)
        elif self.view == "d3":
            kw = fv.with_d3(kw)
        side_off = self.probe is not None and self.probe.name == "P1_ALL_SIDE_OFF"
        self.inner.side_enabled = not side_off
        try:
            return self.inner(**kw)
        finally:
            self.inner.side_enabled = True


def build(registry_entry: dict, baseline_sd: dict, *, view: str = "full", probe: Optional[fv.ProbeSpec] = None,
          m1: Optional[dict] = None, device="cpu", vocab_sizes=None, product_idx_of_class=None,
          seed: int = 2026) -> FOModel:
    """Build a model from the pinned baseline weights. The init seed never matters, because every
    parameter is overwritten by a strict load: the folded weights for "lean", the baseline plus
    zero columns for "d3"."""
    cfg = SASRecConfig.from_registry(registry_entry)
    inner = build_sasrec(cfg, seed=seed, device="cpu", vocab_sizes=vocab_sizes,
                         product_idx_of_class=product_idx_of_class)
    if view == "full":
        sd = baseline_sd
    elif view == "lean":
        sd = fv.fold_lean_state_dict(baseline_sd, m1)
        _resize_inputs(inner, len(fv.LEAN_NUM) + len(fv.LEAN_FLAG), len(fv.UCTX_NAMES) + len(fv.LEAN_UMASK))
    else:
        sd = fv.d3_state_dict(baseline_sd)
        _resize_inputs(inner, len(fv.NUM_NAMES) + len(fv.D3_NAMES) + len(fv.FLAG_NAMES), inner.user_mlp.in_features)
    inner.load_state_dict(sd, strict=True)
    return FOModel(inner, view=view, probe=probe, m1=m1).to(device)


def parameter_mapping(baseline_sd: dict, view: str, m1: Optional[dict] = None) -> dict:
    """Explicit record of inherited / remapped / new parameters for a view (package test 17)."""
    if view == "full":
        return {"inherited": sorted(baseline_sd), "remapped": [], "new": []}
    new_sd = fv.fold_lean_state_dict(baseline_sd, m1) if view == "lean" else fv.d3_state_dict(baseline_sd)
    remapped = sorted(k for k in new_sd if k in baseline_sd and (new_sd[k].shape != baseline_sd[k].shape
                                                                or not torch.equal(new_sd[k], baseline_sd[k])))
    inherited = sorted(k for k in new_sd if k in baseline_sd and k not in remapped)
    rec = {"inherited": inherited, "remapped": remapped, "new": sorted(set(new_sd) - set(baseline_sd))}
    if view == "d3":
        rec["new_columns"] = {"numflag_proj.weight": {"columns": list(range(19, 24)), "names": list(fv.D3_NAMES),
                                                      "init": "zeros"}}
    return rec
