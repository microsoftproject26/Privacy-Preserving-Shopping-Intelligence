"""Shared helpers for the independent cross-check tests (not collected by pytest: no test_ prefix).

Only thin, generic construction helpers live here (no expected-value logic, which stays in each
test module next to its hand-computed numbers).
"""
from __future__ import annotations

from collections import OrderedDict

import numpy as np

from ppsi.fedsim.adapter import Entry, ParamManifest, Role
from ppsi.fedsim.client import ClientData
from ppsi.fedsim.synthetic import client_examples, make_tiny_adapter


def flat_manifest(shared: dict, buffers: dict | None = None) -> ParamManifest:
    """A hand-built manifest with SHARED float32 keys `{name: shape}` and BUFFER keys
    `{name: (shape, dtype, rule)}` — no model involved, for pure aggregation-arithmetic tests."""
    entries: OrderedDict[str, Entry] = OrderedDict()
    for k, shape in shared.items():
        numel = 1
        for s in shape:
            numel *= int(s)
        entries[k] = Entry(k, Role.SHARED, "torch.float32", tuple(shape), numel, 4)
    for k, (shape, dtype, rule) in (buffers or {}).items():
        numel = 1
        for s in shape:
            numel *= int(s)
        es = 8 if dtype == "torch.int64" else 4
        entries[k] = Entry(k, Role.BUFFER, dtype, tuple(shape), numel, es, rule=rule)
    return ParamManifest(entries)


def one_client(key: str, n: int, K: int = 8, L: int = 5, seed: int = 0, invalid_frac: float = 0.0,
              pref: np.ndarray | None = None) -> ClientData:
    rng = np.random.default_rng(seed)
    return ClientData(key, client_examples(n, K, L, rng, pref=pref, invalid_frac=invalid_frac))


def fresh_adapter(K: int = 8, d: int = 4, tied: bool = True, seed: int = 0):
    return make_tiny_adapter(K=K, d=d, tied=tied, dropout=0.0, seed=seed)
