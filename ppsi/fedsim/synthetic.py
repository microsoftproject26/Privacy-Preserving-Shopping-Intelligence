"""A tiny synthetic family and synthetic clients (tests and micro-benchmarks only; no data rows).

TinyRec mirrors the structural features the simulator must handle, at toy size:
  * query(batch) -> q [B, d]: masked mean of item embeddings -> Linear -> tanh -> dropout (dropout draws the global
    RNG, like the real models);
  * tied=True  (GRU-like): the head is item_embed rows 3..3+K, AND a classic tied Linear (`out_proj.weight is
    item_embed.weight`) so the state_dict carries a genuine ALIAS key that must be counted once;
  * tied=False (SASRec-like): an untied output table [K, d] + bias with `recenter_output_()`;
  * a persistent int64 class map (FIXED), a float scale buffer (SERVER_COPY) and a non-persistent buffer.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import Tensor, nn

from .adapter import BaseAdapter, BufferRule, build_manifest
from .client import ClientData


class TinyRec(nn.Module):
    def __init__(self, K: int = 24, d: int = 8, tied: bool = True, dropout: float = 0.1, seed: int = 0,
                 recenter: bool | None = None):
        super().__init__()
        V = K + 3
        self.K, self.d, self.tied = K, d, tied
        self.recenter = (not tied) if recenter is None else bool(recenter)
        if self.recenter and tied:
            raise ValueError("recentering applies to the untied output table only")
        self.item_embed = nn.Embedding(V, d, padding_idx=0)
        self.enc = nn.Linear(d, d)
        self.drop = nn.Dropout(dropout)
        if tied:
            self.out_proj = nn.Linear(d, V, bias=False)
            self.out_proj.weight = self.item_embed.weight            # genuine alias key in the state_dict
        else:
            self.output_embed = nn.Parameter(torch.empty(K, d))
        self.output_bias = nn.Parameter(torch.zeros(K))
        self.register_buffer("class_map", torch.arange(3, 3 + K, dtype=torch.long))
        self.register_buffer("input_scale", torch.tensor(1.0))
        self.register_buffer("scratch", torch.zeros(2), persistent=False)
        g = torch.Generator().manual_seed(int(seed))
        with torch.no_grad():
            nn.init.normal_(self.item_embed.weight, 0.0, 0.5, generator=g)
            self.item_embed.weight[0].zero_()
            nn.init.normal_(self.enc.weight, 0.0, 0.5, generator=g)
            nn.init.normal_(self.enc.bias, 0.0, 0.1, generator=g)
            if not tied:
                nn.init.normal_(self.output_embed, 0.0, 0.5, generator=g)
                self.output_embed.add_(0.3)                         # a visible common mode (recentering test)
            nn.init.normal_(self.output_bias, 0.0, 0.1, generator=g)

    def query(self, batch: dict) -> Tensor:
        tok, m = batch["item_tokens"], batch["attention_mask"].to(torch.float32)
        e = self.item_embed(tok) * m.unsqueeze(-1)
        h = e.sum(1) / m.sum(1, keepdim=True).clamp_min(1.0)
        return self.drop(torch.tanh(self.enc(h * self.input_scale)))

    def head_weight(self) -> Tensor:
        if self.tied:
            return self.item_embed.weight.index_select(0, self.class_map)
        return self.output_embed

    def forward(self, **batch) -> Tensor:
        return torch.addmm(self.output_bias, self.query(batch), self.head_weight().t())

    @torch.no_grad()
    def recenter_output_(self) -> None:
        self.output_embed.sub_(self.output_embed.mean(0, keepdim=True))
        self.output_bias.sub_(self.output_bias.mean())


class TinyAdapter(BaseAdapter):
    def __init__(self, module: TinyRec):
        man = build_manifest(module, buffer_rules={"input_scale": BufferRule.SERVER_COPY})
        super().__init__(module, man, query_dim=module.d, K=module.K)
        self.has_post_step = module.recenter

    def query(self, batch):
        return self.module.query(batch)

    def head_weight(self):
        return self.module.head_weight()

    def head_bias(self):
        return self.module.output_bias

    def post_step(self):
        if self.module.recenter:
            self.module.recenter_output_()

    def server_post_aggregate(self):
        if self.module.recenter:
            self.module.recenter_output_()


def make_tiny_adapter(K: int = 24, d: int = 8, tied: bool = True, dropout: float = 0.1, seed: int = 0,
                      recenter: bool | None = None) -> TinyAdapter:
    return TinyAdapter(TinyRec(K=K, d=d, tied=tied, dropout=dropout, seed=seed, recenter=recenter))


def client_examples(n: int, K: int, L: int, rng: np.random.Generator, *, pref: np.ndarray | None = None,
                    invalid_frac: float = 0.0) -> dict:
    """n synthetic right-padded decisions; targets drawn from a user preference (so p_u has something to learn)."""
    lengths = rng.integers(1, L + 1, size=n)
    toks = rng.integers(3, K + 3, size=(n, L))
    am = np.arange(L)[None, :] < lengths[:, None]
    toks = np.where(am, toks, 0)
    if pref is None:
        tgt = rng.integers(0, K, size=n)
    else:
        tgt = rng.choice(K, size=n, p=pref)
    if invalid_frac > 0:
        tgt = np.where(rng.random(n) < invalid_frac, -1, tgt)
    return {"item_tokens": torch.as_tensor(toks, dtype=torch.long),
            "attention_mask": torch.as_tensor(am),
            "lengths": torch.as_tensor(lengths, dtype=torch.long),
            "target_class": torch.as_tensor(tgt, dtype=torch.long)}


def census_like_sizes(n_clients: int, rng: np.random.Generator) -> np.ndarray:
    """Heavy-tailed client sizes (median ~6, p90 ~35, like the real cohort), capped for test speed."""
    s = np.maximum(1, np.round(rng.lognormal(mean=np.log(6.0), sigma=1.3, size=n_clients))).astype(int)
    return np.minimum(s, 120)


def make_clients(n_clients: int, K: int = 24, L: int = 6, seed: int = 0, sizes=None, invalid_frac: float = 0.1,
                 prefix: str = "user-") -> list:
    rng = np.random.default_rng(seed)
    if sizes is None:
        sizes = census_like_sizes(n_clients, rng)
    out = []
    for i, n in enumerate(sizes):
        alpha = np.full(K, 0.3)
        pref = rng.dirichlet(alpha)
        out.append(ClientData(f"{prefix}{i:05d}", client_examples(int(n), K, L, rng, pref=pref,
                                                                    invalid_frac=invalid_frac)))
    return out
