"""Private personal (PF) state keyed by a collision-free private client key; central personalised training (CPF)
with the same forward.

PF: q_u = q_shared(history, metadata) + p_u, then the SAME item head. p_u has the query dimension, starts at 0, and
p_u = 0 reproduces the shared model exactly (adapter.scores(batch, p=0) == adapter.scores(batch)).

`PersonalStore` holds p_u and its persistent AdamW moments per private key:
  * keys are non-empty strings (e.g. a salted hash of the raw user id); integers, and the shared special tokens
    PAD / MISSING / OOV in any spelling, are refused, so no two users can share state through an OOV token;
  * `get` returns a CLONE (a worker can never mutate committed state in place) or None for a new user (p = 0);
  * `commit` stores a clone on the host; the round driver commits only after the round's aggregation succeeded,
    so an aborted round leaves every p_u unchanged;
  * nothing in the store is ever put in a server payload (the aggregator refuses any non-shared upload key).

CPF (central personalized control) uses the same forward with p rows gathered per decision, and the matched
regularizer (lambda/2) * sum over the DISTINCT users of the batch of ||p_u||^2 — each touched user's vector is penalized
once per step, exactly what a PF client pays for its own p_u in each local step.
"""
from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping

import torch
import torch.nn.functional as F
from torch import Tensor

from .client import PersonalState
from .numerics import tensor_bytes

_RESERVED = {"", "0", "1", "2", "pad", "missing", "oov", "<pad>", "<missing>", "<oov>", "none", "null"}


class PrivateKeyError(ValueError):
    pass


def validate_private_key(key) -> str:
    if not isinstance(key, str):
        raise PrivateKeyError(f"private key must be a str, got {type(key).__name__} (shared tokens are not keys)")
    if key.strip().lower() in _RESERVED:
        raise PrivateKeyError(f"private key {key!r} is a shared special token or empty")
    return key


class PersonalStore:
    def __init__(self, query_dim: int):
        self.query_dim = int(query_dim)
        self._s: dict = {}

    def __contains__(self, key: str) -> bool:
        return key in self._s

    def __len__(self) -> int:
        return len(self._s)

    def keys(self) -> list:
        return sorted(self._s)

    def get(self, key: str) -> PersonalState | None:
        validate_private_key(key)
        st = self._s.get(key)
        return None if st is None else st.clone()

    def p(self, key: str) -> Tensor:
        st = self._s.get(validate_private_key(key))
        return torch.zeros(self.query_dim) if st is None else st.p.detach().clone()

    def commit(self, key: str, state: PersonalState) -> None:
        validate_private_key(key)
        if tuple(state.p.shape) != (self.query_dim,):
            raise ValueError(f"p_u shape {tuple(state.p.shape)} != ({self.query_dim},)")
        st = state.clone()
        st.p = st.p.to("cpu")
        if st.opt_state is not None:
            st.opt_state = {k: v.to("cpu") for k, v in st.opt_state.items()}
        self._s[key] = st

    def resident_bytes(self) -> int:
        n = 0
        for st in self._s.values():
            n += st.p.numel() * st.p.element_size()
            if st.opt_state:
                n += sum(v.numel() * v.element_size() for v in st.opt_state.values())
        return n

    def state_dict(self) -> dict:
        """Dense packing for checkpoints (262,144 users x d stays a handful of tensors, not 10^6 small ones)."""
        keys = sorted(self._s)
        U, d = len(keys), self.query_dim
        out = {"query_dim": d, "keys": keys, "p": torch.zeros(U, d), "exp_avg": torch.zeros(U, d),
               "exp_avg_sq": torch.zeros(U, d), "step": torch.zeros(U), "has_opt": torch.zeros(U, dtype=torch.bool),
               "visits": torch.zeros(U, dtype=torch.long), "n_consumed": torch.zeros(U, dtype=torch.long)}
        for i, k in enumerate(keys):
            st = self._s[k]
            out["p"][i] = st.p
            out["visits"][i] = st.visits
            out["n_consumed"][i] = st.n_consumed
            if st.opt_state is not None:
                if set(st.opt_state) != {"step", "exp_avg", "exp_avg_sq"}:
                    raise ValueError(f"unexpected personal optimizer state keys {sorted(st.opt_state)}")
                out["exp_avg"][i] = st.opt_state["exp_avg"]
                out["exp_avg_sq"][i] = st.opt_state["exp_avg_sq"]
                out["step"][i] = st.opt_state["step"].to(torch.float32)
                out["has_opt"][i] = True
        return out

    @classmethod
    def from_state_dict(cls, d: dict) -> PersonalStore:
        store = cls(int(d["query_dim"]))
        for i, k in enumerate(d["keys"]):
            opt = None
            if bool(d["has_opt"][i]):
                opt = {"step": d["step"][i].clone(), "exp_avg": d["exp_avg"][i].clone(),
                       "exp_avg_sq": d["exp_avg_sq"][i].clone()}
            store.commit(k, PersonalState(d["p"][i].clone(), opt, int(d["visits"][i]), int(d["n_consumed"][i])))
        return store

    def digest(self) -> str:
        h = hashlib.sha256()
        for k in sorted(self._s):
            st = self._s[k]
            h.update(k.encode())
            h.update(tensor_bytes(st.p))
            for name in sorted(st.opt_state or {}):
                h.update(name.encode())
                h.update(tensor_bytes(st.opt_state[name]))
        return h.hexdigest()


def assert_unique_keys(keys: Iterable[str]) -> None:
    ks = [validate_private_key(k) for k in keys]
    if len(set(ks)) != len(ks):
        dup = sorted({k for k in ks if ks.count(k) > 1})
        raise PrivateKeyError(f"duplicate client keys in one round: {dup}")


def cpf_scores(adapter, batch: Mapping[str, Tensor], P: Tensor, row_user: Tensor) -> Tensor:
    """CPF forward: logits(query(batch) + P[row_user]) — the same adapter.scores path as PF."""
    return adapter.scores(batch, P.index_select(0, row_user))


def cpf_loss(adapter, batch: Mapping[str, Tensor], P: Tensor, row_user: Tensor, lam: float) -> Tensor:
    logits = cpf_scores(adapter, batch, P, row_user)
    tgt = batch[adapter.target_key]
    ce = F.cross_entropy(logits, tgt, reduction="sum") / tgt.shape[0]
    if lam <= 0:
        return ce
    touched = torch.unique(row_user)
    return ce + 0.5 * lam * (P.index_select(0, touched) ** 2).sum()


def pf_step_loss(adapter, batch: Mapping[str, Tensor], p: Tensor, lam: float) -> Tensor:
    """The PF client's per-step loss (as in client.local_passes), exposed for the CPF/PF match check."""
    logits = adapter.scores(batch, p)
    tgt = batch[adapter.target_key]
    ce = F.cross_entropy(logits, tgt, reduction="sum") / tgt.shape[0]
    return ce + (0.5 * lam) * (p * p).sum() if lam > 0 else ce
