"""Shared helpers for the federated simulator tests (synthetic fixtures only)."""
from __future__ import annotations

from collections import OrderedDict

import numpy as np
import pytest
import torch
from torch import nn

from ppsi.fedsim.adapter import BaseAdapter, BufferRule, build_manifest
from ppsi.fedsim.client import ClientData, LocalSolver
from ppsi.fedsim.server import Server
from ppsi.fedsim.synthetic import client_examples, make_clients, make_tiny_adapter

K, D, LW = 24, 8, 6


def nc(reason: str):
    """Negative control: the wrapped key-property check must raise AssertionError (strict xfail)."""
    return pytest.mark.xfail(strict=True, raises=AssertionError, reason="NEGATIVE CONTROL: " + reason)


def assert_raises(exc, fn, *a, **kw):
    """Like pytest.raises, but a missing exception is an AssertionError (so negative controls can use it)."""
    try:
        fn(*a, **kw)
    except exc:
        return
    raise AssertionError(f"{getattr(fn, '__name__', fn)} did not raise {exc}")


def tiny(seed: int = 1, **kw):
    kw.setdefault("K", K)
    kw.setdefault("d", D)
    return make_tiny_adapter(seed=seed, **kw)


def server_and_worker(seed: int = 1, n_workers: int = 1, **kw):
    return Server(tiny(seed, **kw)), [tiny(seed + 100 + i, **kw) for i in range(n_workers)]


def solver(**kw) -> LocalSolver:
    base = {"lr": 0.05, "passes": 2, "batch_size": 16, "clip": 1.0, "weight_decay": 1e-5}
    base.update(kw)
    return LocalSolver(**base)


def clients(n: int = 6, seed: int = 3, sizes=None, prefix: str = "user-", invalid_frac: float = 0.1):
    return make_clients(n, K=K, L=LW, seed=seed, sizes=sizes, prefix=prefix, invalid_frac=invalid_frac)


def one_client(key: str, n: int, seed: int = 0, invalid_frac: float = 0.0) -> ClientData:
    rng = np.random.default_rng(seed)
    return ClientData(key, client_examples(n, K, LW, rng, invalid_frac=invalid_frac))


def clone_state(s):
    return OrderedDict((k, v.detach().clone()) for k, v in s.items())


def sq_dist(a, b, keys) -> float:
    return float(sum(((a[k].double() - b[k].double()) ** 2).sum() for k in keys))


# ---------------------------------------------------------------- a hand-checkable toy family (aggregation, bytes)
class Toy(nn.Module):
    """w [2] shared; emb [4, 2] shared with a tied alias key `head.weight`; int map (FIXED); float scale
    (SERVER_COPY); a non-persistent buffer."""

    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.zeros(2))
        self.emb = nn.Embedding(4, 2)
        self.head = nn.Linear(2, 4, bias=False)
        self.head.weight = self.emb.weight
        self.register_buffer("imap", torch.tensor([3, 4, 5], dtype=torch.long))
        self.register_buffer("scale", torch.tensor(2.0))
        self.register_buffer("tmp", torch.ones(3), persistent=False)


class ToyAdapter(BaseAdapter):
    def __init__(self, m: Toy):
        super().__init__(m, build_manifest(m, buffer_rules={"scale": BufferRule.SERVER_COPY}), query_dim=2, K=4)

    def query(self, batch):
        return batch["x"] + self.module.w

    def head_weight(self):
        return self.module.emb.weight

    def head_bias(self):
        return None
