"""Pre-registered separate systems for registry v3 amendment 2 (owner: Opus). Evaluation only.

- LogProbEnsemble: equal-weight mean of per-member log-softmax over the K classes (weights fixed
  a priori, never tuned on validation). Exposes the union of its members' channel attributes so
  the reused evaluator's build_named_kwargs hands it every channel a member needs; each member
  receives only its own declared channels (an item-only member never sees side channels).
- average_state_dicts: uniform average of several checkpoints of ONE run (S4 "epoch average").
These are system-level results, reported separately from the single-model headline.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

from sasrec_model import SIDE_CHANNELS, SASRecCE

_RICH_KEYS = SIDE_CHANNELS + ("event_numeric_features", "event_quality_flags", "user_context", "user_context_masks")


class LogProbEnsemble(nn.Module):
    def __init__(self, members: list[SASRecCE]):
        super().__init__()
        if len(members) < 2:
            raise ValueError("an ensemble needs >= 2 members")
        Ks = {m.K for m in members}
        if len(Ks) != 1:
            raise ValueError("members disagree on K")
        self.members = nn.ModuleList(members)
        self.K = Ks.pop()
        rich = [m for m in members if m.cfg.rich]
        self._categorical_channels = ("item_tokens",) + (SIDE_CHANNELS if rich else ())
        self.numeric_dim = rich[0].numeric_dim if rich else 0
        self.flags_dim = rich[0].flags_dim if rich else 0
        self.use_user_context = bool(rich)

    def forward(self, *, item_tokens, lengths, attention_mask, **side_kw):
        acc: Optional[torch.Tensor] = None
        for m in self.members:
            kw = dict(item_tokens=item_tokens, lengths=lengths, attention_mask=attention_mask)
            if m.cfg.rich:
                kw.update({k: side_kw[k] for k in _RICH_KEYS})
            lp = F.log_softmax(m(**kw), dim=1)
            acc = lp if acc is None else acc + lp
        return acc / len(self.members)


def load_state_dict_file(path) -> dict:
    obj = torch.load(Path(path), map_location="cpu", weights_only=False)
    return obj["state_dict"] if isinstance(obj, dict) and "state_dict" in obj else obj


def average_state_dicts(paths) -> dict:
    sds = [load_state_dict_file(p) for p in paths]
    keys = list(sds[0].keys())
    if any(list(sd.keys()) != keys for sd in sds):
        raise ValueError("checkpoints have different parameter sets")
    out = {}
    for k in keys:
        t = sds[0][k]
        if torch.is_floating_point(t):
            out[k] = sum(sd[k].double() for sd in sds).div(len(sds)).to(t.dtype)
        else:
            if any(not torch.equal(sd[k], t) for sd in sds):
                raise ValueError(f"non-float tensor {k} differs between checkpoints")
            out[k] = t.clone()
    return out
