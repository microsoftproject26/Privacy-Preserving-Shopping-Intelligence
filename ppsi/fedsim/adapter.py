"""The model-agnostic adapter protocol and the parameter-role manifest.

A family plugs into the simulator through `ModelAdapter`:
  query(batch)            -> q [B, d]   (the query BEFORE the item head; PF adds p_u here)
  head_weight()           -> W [K, d]   (SASRec: the untied output table; GRU: the tied item_embed rows, taken with the
                                         model's own indexing, so the gradient reaches the one item Parameter)
  head_bias()             -> b [K] or None
  logits(q)               -> [B, K]     (the family's own scoring formula, bit-identical to module.forward)
  post_step()             optional per-optimizer-step hook (SASRec: recenter_output_)
  server_post_aggregate() optional server hook after aggregation and at init (SASRec: recenter_output_)
  manifest                ParamManifest: every state_dict key with exactly one role

Roles (build_manifest):
  SHARED   trainable, FP32, aggregated, sent down and up once per visit.
  ALIAS    a second state_dict key for a tied Parameter; never aggregated or counted, it follows its canonical key.
  BUFFER   persistent buffer (or frozen parameter) with an explicit rule:
             FIXED             integer maps / class orders / frozen tensors: never averaged, verified identical;
             SERVER_COPY       float buffer the server owns; a client's copy is ignored.
           There is deliberately no averaged-buffer rule: no supported family has a client-updated buffer, and every
           BN / FedBN branch is out of scope.
           Integer buffers can only be FIXED. A float buffer without an explicit rule is refused.
  PRIVATE  declared per-client state outside the module (the PF vector p_u and its AdamW moments): never uploaded,
           never aggregated, 0 bytes on the wire.
Non-persistent buffers (e.g. SASRec's causal mask and class map) are rebuilt from the config and are listed only.
"""
from __future__ import annotations

import hashlib
import json
from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Protocol, runtime_checkable

import torch
from torch import Tensor, nn


class Role(str, Enum):
    SHARED = "shared"
    ALIAS = "alias"
    BUFFER = "buffer"
    PRIVATE = "private"


class BufferRule(str, Enum):
    FIXED = "fixed"
    SERVER_COPY = "server_copy"


class ManifestError(ValueError):
    pass


@dataclass(frozen=True)
class Entry:
    key: str
    role: Role
    dtype: str
    shape: tuple
    numel: int
    element_size: int
    alias_of: str | None = None
    rule: BufferRule | None = None

    @property
    def nbytes(self) -> int:
        return self.numel * self.element_size


@dataclass
class ParamManifest:
    entries: OrderedDict[str, Entry]
    nonpersistent_buffers: tuple = ()
    private_extra: OrderedDict[str, Entry] = field(default_factory=OrderedDict)

    def keys(self, role: Role, rule: BufferRule | None = None) -> tuple:
        return tuple(k for k, e in self.entries.items() if e.role == role and (rule is None or e.rule == rule))

    @property
    def shared_keys(self) -> tuple:
        return self.keys(Role.SHARED)

    @property
    def alias_keys(self) -> tuple:
        return self.keys(Role.ALIAS)

    @property
    def buffer_keys(self) -> tuple:
        return self.keys(Role.BUFFER)

    @property
    def shared_numel(self) -> int:
        return sum(self.entries[k].numel for k in self.shared_keys)

    @property
    def shared_bytes(self) -> int:
        return sum(self.entries[k].nbytes for k in self.shared_keys)

    def buffer_bytes(self, rule: BufferRule | None = None) -> int:
        return sum(self.entries[k].nbytes for k in self.keys(Role.BUFFER, rule))

    @property
    def private_resident_bytes(self) -> int:
        return sum(e.nbytes for e in self.private_extra.values())

    def with_private(self, key: str, shape: tuple, dtype: torch.dtype = torch.float32) -> ParamManifest:
        """Declare per-client state held outside the module (e.g. the PF vector and its AdamW moments)."""
        ex = OrderedDict(self.private_extra)
        numel = 1
        for s in shape:
            numel *= int(s)
        ex[key] = Entry(key, Role.PRIVATE, str(dtype), tuple(shape), numel, torch.empty((), dtype=dtype).element_size())
        return ParamManifest(self.entries, self.nonpersistent_buffers, ex)

    def as_json(self) -> dict:
        def e2j(e: Entry) -> dict:
            return {"role": e.role.value, "dtype": e.dtype, "shape": list(e.shape), "numel": e.numel,
                    "alias_of": e.alias_of, "rule": e.rule.value if e.rule else None}
        return {"entries": {k: e2j(e) for k, e in self.entries.items()},
                "private_extra": {k: e2j(e) for k, e in self.private_extra.items()},
                "nonpersistent_buffers": list(self.nonpersistent_buffers),
                "shared_numel": self.shared_numel, "shared_bytes": self.shared_bytes}

    def digest(self) -> str:
        return hashlib.sha256(json.dumps(self.as_json(), sort_keys=True).encode()).hexdigest()


def build_manifest(module: nn.Module, *, buffer_rules: Mapping[str, BufferRule] | None = None) -> ParamManifest:
    """Assign exactly one role to every state_dict key of `module` (see the module docstring)."""
    buffer_rules = dict(buffer_rules or {})
    sd = module.state_dict(keep_vars=True)
    canonical = {id(p): n for n, p in module.named_parameters()}           # first registration wins
    all_params = dict(module.named_parameters(remove_duplicate=False))
    entries: OrderedDict[str, Entry] = OrderedDict()
    for key, t in sd.items():
        es = t.element_size()
        meta = {"key": key, "dtype": str(t.dtype), "shape": tuple(t.shape), "numel": int(t.numel()), "element_size": int(es)}
        if key in all_params:
            canon = canonical[id(all_params[key])]
            if canon != key:
                entries[key] = Entry(role=Role.ALIAS, alias_of=canon, **meta)
            elif not t.requires_grad:
                entries[key] = Entry(role=Role.BUFFER, rule=BufferRule.FIXED, **meta)
            else:
                if t.dtype != torch.float32:
                    raise ManifestError(f"shared parameter {key} is {t.dtype}; strict FP32 requires float32")
                entries[key] = Entry(role=Role.SHARED, **meta)
            continue
        rule = buffer_rules.pop(key, None)
        if not t.is_floating_point():
            if rule not in (None, BufferRule.FIXED):
                raise ManifestError(f"integer buffer {key} can only be FIXED (integer maps are never averaged)")
            rule = BufferRule.FIXED
        elif rule is None:
            raise ManifestError(f"float buffer {key} has no explicit rule (FIXED or SERVER_COPY)")
        entries[key] = Entry(role=Role.BUFFER, rule=BufferRule(rule), **meta)
    if buffer_rules:
        raise ManifestError(f"buffer rules for unknown keys: {sorted(buffer_rules)}")
    for key, e in entries.items():
        if e.role == Role.ALIAS and entries[e.alias_of].role != Role.SHARED:
            raise ManifestError(f"alias {key} points to non-shared {e.alias_of}")
    nonpersistent = tuple(n for n, _ in module.named_buffers() if n not in sd)
    return ParamManifest(entries, nonpersistent)


@runtime_checkable
class ModelAdapter(Protocol):
    module: nn.Module
    manifest: ParamManifest
    query_dim: int
    K: int

    def query(self, batch: Mapping[str, Tensor]) -> Tensor: ...
    def head_weight(self) -> Tensor: ...
    def head_bias(self) -> Tensor | None: ...
    def logits(self, q: Tensor) -> Tensor: ...
    def post_step(self) -> None: ...
    def server_post_aggregate(self) -> None: ...


class BaseAdapter:
    """Shared plumbing for concrete adapters. Subclasses implement `query`, `head_weight`, `head_bias`."""

    target_key = "target_class"
    loss_mask_key = "loss_mask"
    has_post_step = False

    def __init__(self, module: nn.Module, manifest: ParamManifest, *, query_dim: int, K: int):
        self.module = module
        self.manifest = manifest
        self.query_dim = int(query_dim)
        self.K = int(K)
        named = dict(module.named_parameters())
        self._shared = [named[k] for k in manifest.shared_keys]
        sd = module.state_dict(keep_vars=True)
        self._buf = {k: sd[k] for k in manifest.buffer_keys}
        self.device = next(module.parameters()).device

    # -- family hooks -------------------------------------------------------------------------------------------
    def query(self, batch: Mapping[str, Tensor]) -> Tensor:  # pragma: no cover - abstract
        raise NotImplementedError

    def head_weight(self) -> Tensor:  # pragma: no cover - abstract
        raise NotImplementedError

    def head_bias(self) -> Tensor | None:  # pragma: no cover - abstract
        raise NotImplementedError

    def logits(self, q: Tensor) -> Tensor:
        b = self.head_bias()
        W = self.head_weight()
        return q @ W.t() if b is None else torch.addmm(b, q, W.t())

    def post_step(self) -> None:
        return None

    def server_post_aggregate(self) -> None:
        return None

    # -- parameters and state -----------------------------------------------------------------------------------
    def shared_parameters(self) -> list:
        return list(self._shared)

    def param_groups(self, weight_decay: float) -> list:
        """The central training rule: ndim >= 2 decays, biases / LayerNorm do not."""
        decay = [p for p in self._shared if p.ndim >= 2]
        no_decay = [p for p in self._shared if p.ndim < 2]
        return [{"params": decay, "weight_decay": float(weight_decay)}, {"params": no_decay, "weight_decay": 0.0}]

    @torch.no_grad()
    def load_state_(self, state: Mapping[str, Tensor]) -> None:
        """Copy a broadcast state (shared + buffers, canonical keys) into the module. Aliases follow automatically."""
        for k, p in zip(self.manifest.shared_keys, self._shared):
            p.copy_(state[k])
        for k, b in self._buf.items():
            rule = self.manifest.entries[k].rule
            if rule == BufferRule.FIXED:
                src = state[k]
                if src.shape != b.shape or src.dtype != b.dtype or not torch.equal(src.to(b.device), b):
                    raise ManifestError(f"FIXED buffer {k} differs between broadcast and worker")
            else:
                b.copy_(state[k])

    def extract_shared(self, clone: bool = False) -> OrderedDict[str, Tensor]:
        out = OrderedDict()
        for k, p in zip(self.manifest.shared_keys, self._shared):
            out[k] = p.detach().clone() if clone else p.detach()
        return out

    def extract_buffers(self, clone: bool = False) -> OrderedDict[str, Tensor]:
        return OrderedDict((k, b.detach().clone() if clone else b.detach()) for k, b in self._buf.items())

    def broadcast_state(self, clone: bool = False) -> OrderedDict[str, Tensor]:
        s = self.extract_shared(clone)
        s.update(self.extract_buffers(clone))
        return s

    def scores(self, batch: Mapping[str, Tensor], p: Tensor | None = None) -> Tensor:
        """The one forward used by FA/FP/PF/CPF/LO: logits(query(batch) [+ p])."""
        q = self.query(batch)
        if p is not None:
            q = q + p
        return self.logits(q)
