"""SASRecCE: a SASRec-family causal Transformer for the one-decision next-item task.

A SASRec-family model (Kang & McAuley 2018) adapted to the one-decision task; not a literal reproduction of the
original TF1 code.

Input layout (checked on every forward): right-padded windows, real events at columns 0..len-1 in chronological
order, PAD after. Real-event positions are 0..len-1 counted from the oldest retained event and are derived from the
validity mask, never from batch padding or numeric values.

Per block (Pre-LN on Q/K/V, 4d GELU FFN):
    u = x + Dropout(Wo * CausalAttn(Wq LN1(x), Wk LN1(x), Wv LN1(x)))
    x = u + Dropout(W2 * Dropout(GELU(W1 LN2(u))))
    x = x * valid_mask
Attention-probability dropout = cfg.dropout in training, exactly 0 in eval.

Two numerically equivalent implementations (tests compare them):
- "unpadded" (production): LN/QKV/FFN/residual/dropout run on the valid tokens only; q/k/v are scattered into the
  right-padded [B,T] layout just for the causal product. A real query at position i only sees keys j <= i, all of
  which are real, so PAD never influences a real position (PAD keys are strictly in every real query's future).
- "padded_reference": the literal padded computation with x*mask after every block.
Attention is explicit FP32 math (lower-triangular mask including the diagonal): no fully masked softmax rows exist by
construction, so no NaN repair is ever needed.

Recipe choices that keep dense-softmax Adam training stable (a tied item table drifted in its common mode, amplified
by a sqrt(d) input scale into a Pre-LN stream with no input LayerNorm):
- input: z = LN_in(E[item] + P[pos] (+ gated rich side residual)), then input dropout (input LayerNorm, no sqrt(d)
  scaling);
- head: logits = LN_final(h_last) @ W_out^T + b with a SEPARATE output table W_out [K, d] (untied, as gSASRec /
  Transformers4Rec allow); class j <-> item token j+3;
- `recenter_output_()` removes the common mode of W_out / b after each optimizer step: a per-query constant logit
  shift, exactly loss-neutral under softmax shift invariance.

The four float widths (numeric / flags / user context / user mask) come from a `widths` object (InputWidths), with a
width check on every rich float input block. The side-token list is configurable (`side_channels`, a subsequence of
SIDE_CHANNELS in its order), and a branch exists only when it has an input (numflag_proj iff numeric + flags > 0; the
side branch iff side tokens or numflag; the user branch iff user ctx + masks > 0).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

__all__ = ["SIDE_CHANNELS", "SIDE_DIMS", "SASRecCE", "SASRecConfig", "build_sasrec", "parameter_inventory"]

SIDE_CHANNELS = ("category_tokens", "brand_tokens", "main_category_tokens",
                 "daypart_tokens", "weekday_tokens")
SIDE_DIMS = {"category_tokens": 32, "brand_tokens": 32, "main_category_tokens": 8,
             "daypart_tokens": 4, "weekday_tokens": 4}
NUMFLAG_OUT = 32       # input widths (numeric / flags / user ctx / user mask) come from `widths`
USER_HIDDEN = 64
GATE_INIT = -3.0
RICH_RNG_OFFSET = 1_000_003
RICH_KEYS = SIDE_CHANNELS + ("event_numeric_features", "event_quality_flags", "user_context", "user_context_masks")


def resolve_side_channels(side_channels=None) -> tuple:
    """None -> SIDE_CHANNELS; else a duplicate-free subsequence of SIDE_CHANNELS in that order (the init order)."""
    if side_channels is None:
        return SIDE_CHANNELS
    side = tuple(str(c) for c in side_channels)
    if side != tuple(c for c in SIDE_CHANNELS if c in side):
        raise ValueError(f"side_channels {side} must be a duplicate-free subsequence of {SIDE_CHANNELS}")
    return side


@dataclass(frozen=True)
class SASRecConfig:
    run_id: str
    group: str                 # "item_only" | "rich_frozen_channels"
    d_model: int
    blocks: int
    heads: int
    ff_multiplier: int = 4
    dropout: float = 0.2
    max_len: int = 50
    ln_eps: float = 1e-5

    @property
    def rich(self) -> bool:
        return self.group == "rich_frozen_channels"

    @classmethod
    def from_entry(cls, entry: dict) -> SASRecConfig:
        group = entry["group"]
        if group not in ("item_only", "rich_frozen_channels"):
            raise ValueError(f"unknown group {group!r}")
        return cls(run_id=entry["id"], group=group, d_model=int(entry["d_model"]),
                   blocks=int(entry["blocks"]), heads=int(entry["heads"]),
                   ff_multiplier=int(entry["ff_multiplier"]), dropout=float(entry["dropout"]))


class CausalBlock(nn.Module):
    def __init__(self, d: int, heads: int, ff_mult: int, p: float, eps: float):
        super().__init__()
        if d % heads:
            raise ValueError("d_model must be divisible by heads")
        self.d, self.h, self.dh, self.p = d, heads, d // heads, p
        self.ln1 = nn.LayerNorm(d, eps=eps)
        self.qkv = nn.Linear(d, 3 * d)      # rows [0:d]=Wq, [d:2d]=Wk, [2d:3d]=Wv (one GEMM)
        self.out = nn.Linear(d, d)
        self.ln2 = nn.LayerNorm(d, eps=eps)
        self.ff1 = nn.Linear(d, ff_mult * d)
        self.ff2 = nn.Linear(ff_mult * d, d)

    def _attend(self, qkv_bt: Tensor, causal: Tensor) -> Tensor:
        """qkv_bt [B,T,3d] -> [B,T,d]; explicit FP32 causal attention."""
        B, T, _ = qkv_bt.shape
        q, k, v = qkv_bt.view(B, T, 3, self.h, self.dh).permute(2, 0, 3, 1, 4)
        s = torch.matmul(q, k.transpose(-1, -2)) * (1.0 / math.sqrt(self.dh))
        s = s.masked_fill(~causal[:T, :T], float("-inf"))
        a = torch.softmax(s, dim=-1)
        a = F.dropout(a, self.p, self.training)
        return torch.matmul(a, v).transpose(1, 2).reshape(B, T, self.d)

    def _ffn(self, u: Tensor) -> Tensor:
        f = self.ff2(F.dropout(F.gelu(self.ff1(self.ln2(u))), self.p, self.training))
        return u + F.dropout(f, self.p, self.training)

    def forward_tokens(self, xf: Tensor, flat_idx: Tensor, B: int, T: int, causal: Tensor) -> Tensor:
        """xf [N,d] valid tokens (row-major); flat_idx their cells in the right-padded [B,T]."""
        qkv = self.qkv(self.ln1(xf))                                         # [N,3d]
        buf = qkv.new_zeros(B * T, 3 * self.d).index_copy(0, flat_idx, qkv)  # right-padded
        o = self._attend(buf.view(B, T, 3 * self.d), causal).reshape(B * T, self.d)
        u = xf + F.dropout(self.out(o.index_select(0, flat_idx)), self.p, self.training)
        return self._ffn(u)

    def forward_padded(self, x: Tensor, mask: Tensor, causal: Tensor) -> Tensor:
        o = self._attend(self.qkv(self.ln1(x)), causal)
        u = x + F.dropout(self.out(o), self.p, self.training)
        return self._ffn(u) * mask.unsqueeze(-1).to(x.dtype)


class SASRecCE(nn.Module):
    def __init__(self, cfg: SASRecConfig, *, vocab_sizes: dict, product_idx_of_class,
                 seed: int, impl: str = "unpadded", widths=None, side_channels=None) -> None:
        super().__init__()
        if impl not in ("unpadded", "padded_reference"):
            raise ValueError(f"unknown impl {impl!r}")
        if cfg.rich and widths is None:
            raise ValueError("a rich model needs `widths` (numeric/flags/user ctx/user mask) from the feature layout")
        self.cfg, self.impl = cfg, impl
        d = cfg.d_model
        self.d = d
        self.item_vocab = int(vocab_sizes["item_tokens"])

        poc = torch.as_tensor(product_idx_of_class, dtype=torch.long)
        K = int(poc.shape[0])
        if poc.ndim != 1 or K < 1 or not torch.equal(poc, torch.arange(3, 3 + K, dtype=torch.long)):
            raise ValueError("known-class map: product_idx_of_class must equal arange(K)+3")
        if 3 + K > self.item_vocab:
            raise ValueError("catalogue exceeds the item table")
        self.K = K
        self.register_buffer("product_idx_of_class", poc, persistent=False)

        # attributes read by the batch helpers (features.gru_kwargs) and the adapters
        self.side_channels = resolve_side_channels(side_channels) if cfg.rich else ()
        self._categorical_channels = ("item_tokens",) + self.side_channels
        self.numeric_dim = int(widths.numeric_dim) if cfg.rich else 0
        self.flags_dim = int(widths.flags_dim) if cfg.rich else 0
        self.user_context_dim = int(widths.user_context_dim) if cfg.rich else 0
        self.user_mask_dim = int(widths.user_mask_dim) if cfg.rich else 0
        self.use_user_context = bool(cfg.rich) and self.user_context_dim + self.user_mask_dim > 0
        self.side_enabled = bool(cfg.rich)
        self.has_numflag = bool(cfg.rich) and self.numeric_dim + self.flags_dim > 0
        self.has_side_branch = bool(self.side_channels) or self.has_numflag
        self.has_user_branch = self.use_user_context

        # ---- common (ID) parameters; created and initialised in a fixed order ----
        self.item_embed = nn.Embedding(self.item_vocab, d, padding_idx=0)
        self.pos_embed = nn.Embedding(cfg.max_len, d)
        self.blocks = nn.ModuleList(CausalBlock(d, cfg.heads, cfg.ff_multiplier, cfg.dropout, cfg.ln_eps)
                                    for _ in range(cfg.blocks))
        self.final_norm = nn.LayerNorm(d, eps=cfg.ln_eps)
        self.input_norm = nn.LayerNorm(d, eps=cfg.ln_eps)
        self.output_embed = nn.Parameter(torch.empty(K, d))    # untied output table
        self.output_bias = nn.Parameter(torch.zeros(K))

        # ---- rich-only parameters ----
        self.side_embeds = self.numflag_proj = self.side_proj = self.side_norm = self.side_gate = None
        self.user_mlp = self.user_proj = self.user_norm = self.user_gate = None
        if cfg.rich:
            if self.side_channels:
                self.side_embeds = nn.ModuleDict({ch: nn.Embedding(int(vocab_sizes[ch]), SIDE_DIMS[ch])
                                                  for ch in self.side_channels})
            if self.has_numflag:
                self.numflag_proj = nn.Linear(self.numeric_dim + self.flags_dim, NUMFLAG_OUT)
            if self.has_side_branch:
                side_width = sum(SIDE_DIMS[ch] for ch in self.side_channels) + (NUMFLAG_OUT if self.has_numflag
                                                                                 else 0)
                self.side_proj = nn.Linear(side_width, d)
                self.side_norm = nn.LayerNorm(d, eps=cfg.ln_eps)
                self.side_gate = nn.Parameter(torch.tensor(GATE_INIT))
            if self.has_user_branch:
                self.user_mlp = nn.Linear(self.user_context_dim + self.user_mask_dim, USER_HIDDEN)
                self.user_proj = nn.Linear(USER_HIDDEN, d)
                self.user_norm = nn.LayerNorm(d, eps=cfg.ln_eps)
                self.user_gate = nn.Parameter(torch.tensor(GATE_INIT))

        self.register_buffer("causal", torch.ones(cfg.max_len, cfg.max_len, dtype=torch.bool).tril(),
                             persistent=False)
        self._init_parameters(int(seed))

    # ------------------------------------------------------------------ init
    @staticmethod
    def _xavier_linear(lin: nn.Linear, g: torch.Generator, split: int = 1) -> None:
        with torch.no_grad():
            rows = lin.weight.shape[0] // split
            for i in range(split):   # Q, K, V each get their own [d,d] Xavier fan
                nn.init.xavier_uniform_(lin.weight[i * rows:(i + 1) * rows], generator=g)
            lin.bias.zero_()

    def _init_parameters(self, seed: int) -> None:
        g = torch.Generator().manual_seed(seed)
        with torch.no_grad():
            nn.init.normal_(self.item_embed.weight, 0.0, 0.02, generator=g)
            self.item_embed.weight[0].zero_()
            nn.init.normal_(self.pos_embed.weight, 0.0, 0.02, generator=g)
            for blk in self.blocks:
                self._xavier_linear(blk.qkv, g, split=3)
                self._xavier_linear(blk.out, g)
                self._xavier_linear(blk.ff1, g)
                self._xavier_linear(blk.ff2, g)
                for ln in (blk.ln1, blk.ln2):
                    ln.weight.fill_(1.0); ln.bias.zero_()
            self.final_norm.weight.fill_(1.0); self.final_norm.bias.zero_()
            self.input_norm.weight.fill_(1.0); self.input_norm.bias.zero_()
            nn.init.normal_(self.output_embed, 0.0, 0.02, generator=g)
            self.output_bias.zero_()
            if self.cfg.rich:  # separate RNG scope: common weights identical to the ID arm
                g2 = torch.Generator().manual_seed(seed + RICH_RNG_OFFSET)
                for ch in self.side_channels:
                    nn.init.normal_(self.side_embeds[ch].weight, 0.0, 0.02, generator=g2)
                for lin in (self.numflag_proj, self.side_proj, self.user_mlp, self.user_proj):
                    if lin is not None:
                        self._xavier_linear(lin, g2)
                for ln in (self.side_norm, self.user_norm):
                    if ln is not None:
                        ln.weight.fill_(1.0); ln.bias.zero_()
                for gate_param in (self.side_gate, self.user_gate):
                    if gate_param is not None:
                        gate_param.fill_(GATE_INIT)

    # --------------------------------------------------------------- helpers
    def _layout(self, item_tokens: Tensor, lengths: Tensor, attention_mask: Tensor,
                position_ids: Tensor | None = None):
        """Validate the right-padded layout; positions are derived from the validity mask.

        If `position_ids` (store convention: 1..len on real events, 0 on PAD) is supplied,
        it must agree exactly with the derived 0..len-1 positions."""
        if item_tokens.ndim != 2 or attention_mask.shape != item_tokens.shape:
            raise ValueError("item_tokens/attention_mask must be [B,T] with equal shapes")
        B, T0 = item_tokens.shape
        if T0 > self.cfg.max_len:
            raise ValueError(f"window width {T0} exceeds max_len {self.cfg.max_len}")
        lengths = lengths.to(item_tokens.device).long()
        T = int(lengths.max().item()) if B else 0
        if T < 1:
            raise ValueError("every decision needs >= 1 real context event (length==0 is invalid)")
        ar = torch.arange(T, device=item_tokens.device)
        valid = ar.unsqueeze(0) < lengths.unsqueeze(1)
        am = attention_mask.to(torch.bool)
        bad = (lengths < 1).any() | (lengths > T0).any() | (am[:, :T] != valid).any()
        if T < T0:
            bad = bad | am[:, T:].any()
        if position_ids is not None:
            if position_ids.shape != item_tokens.shape:
                raise ValueError("position_ids must be [B,T] like item_tokens")
            expect = torch.where(valid, ar.unsqueeze(0) + 1, torch.zeros_like(ar).unsqueeze(0))
            bad = bad | (position_ids[:, :T].long() != expect).any()
            if T < T0:
                bad = bad | (position_ids[:, T:] != 0).any()
        if bool(bad.item()):
            raise ValueError("input is not right-padded: the valid mask must be a contiguous prefix "
                             "of length `lengths` with store positions 1..len (repack before calling the model)")
        return B, T, valid, lengths, ar

    def _side_residual(self, tok_sel, num_sel, flag_sel) -> Tensor:
        parts = [self.side_embeds[ch](tok_sel[ch]) for ch in self.side_channels]
        if self.numflag_proj is not None:
            nf = torch.cat([num_sel.to(torch.float32), flag_sel.to(torch.float32)], dim=-1)
            parts.append(F.gelu(self.numflag_proj(nf)))
        side = self.side_norm(self.side_proj(torch.cat(parts, dim=-1)))
        return torch.sigmoid(self.side_gate) * side

    def _user_residual(self, user_context: Tensor, user_context_masks: Tensor) -> Tensor:
        uc = torch.cat([user_context.to(torch.float32), user_context_masks.to(torch.float32)], dim=-1)
        u = self.user_norm(self.user_proj(F.gelu(self.user_mlp(uc))))
        return torch.sigmoid(self.user_gate) * u

    def _require_rich(self, kw: dict) -> None:
        need = self.side_channels + (("event_numeric_features", "event_quality_flags") if self.has_numflag else ()) \
            + (("user_context", "user_context_masks") if self.has_user_branch else ())
        missing = [k for k in need if kw.get(k) is None]
        if missing:
            raise ValueError(f"rich model requires {missing}")
        unused = [k for k in RICH_KEYS if k not in need and kw.get(k) is not None]
        if unused:
            raise ValueError(f"this model has no input {unused} (project the batch first)")
        for k, width in (("event_numeric_features", self.numeric_dim), ("event_quality_flags", self.flags_dim),
                         ("user_context", self.user_context_dim), ("user_context_masks", self.user_mask_dim)):
            if k in need and kw[k].shape[-1] != width:
                raise ValueError(f"{k} has width {kw[k].shape[-1]}; the model (feature layout) expects {width}")

    # ---------------------------------------------------------------- encode
    def _encode(self, item_tokens: Tensor, lengths: Tensor, attention_mask: Tensor,
                side_kw: dict, *, want_sequence: bool,
                position_ids: Tensor | None = None) -> tuple[Tensor, Tensor | None]:
        """Returns (h_last [B,d] pre final LN, optional sequence [B,T,d] with PAD rows 0)."""
        B, T, valid, lengths, ar = self._layout(item_tokens, lengths, attention_mask, position_ids)
        d = self.d
        rich_on = self.cfg.rich and self.side_enabled and self.has_side_branch
        if self.cfg.rich:
            self._require_rich(side_kw)

        if self.impl == "unpadded":
            flat_idx = valid.reshape(-1).nonzero(as_tuple=True)[0]              # [N], row-major
            pos = ar.unsqueeze(0).expand(B, T).reshape(-1).index_select(0, flat_idx)
            tok = item_tokens[:, :T].reshape(-1).index_select(0, flat_idx)
            x = self.item_embed(tok) + self.pos_embed(pos)
            if rich_on:
                tok_sel = {ch: side_kw[ch][:, :T].reshape(-1).index_select(0, flat_idx) for ch in self.side_channels}
                num = flg = None
                if self.has_numflag:
                    num = side_kw["event_numeric_features"][:, :T].reshape(B * T, self.numeric_dim).index_select(
                        0, flat_idx)
                    flg = side_kw["event_quality_flags"][:, :T].reshape(B * T, self.flags_dim).index_select(0, flat_idx)
                x = x + self._side_residual(tok_sel, num, flg)
            x = F.dropout(self.input_norm(x), self.cfg.dropout, self.training)
            for blk in self.blocks:
                x = blk.forward_tokens(x, flat_idx, B, T, self.causal)
            last = torch.cumsum(lengths, 0) - 1                                   # compact index
            h_last = x.index_select(0, last)
            seq = None
            if want_sequence:
                seq = x.new_zeros(B * T, d).index_copy(0, flat_idx, x).view(B, T, d)
            return h_last, seq

        # padded reference
        vm = valid.to(torch.float32).unsqueeze(-1)
        pos = ar.unsqueeze(0).expand(B, T)
        x = self.item_embed(item_tokens[:, :T]) + self.pos_embed(pos)
        if rich_on:
            tok_sel = {ch: side_kw[ch][:, :T] for ch in self.side_channels}
            nf_on = self.has_numflag
            x = x + self._side_residual(tok_sel, side_kw["event_numeric_features"][:, :T] if nf_on else None,
                                        side_kw["event_quality_flags"][:, :T] if nf_on else None)
        x = F.dropout(self.input_norm(x) * vm, self.cfg.dropout, self.training)
        for blk in self.blocks:
            x = blk.forward_padded(x, valid, self.causal)
        h_last = x.gather(1, (lengths - 1).view(B, 1, 1).expand(B, 1, d)).squeeze(1)
        return h_last, (x if want_sequence else None)

    def _readout(self, h_last: Tensor, side_kw: dict) -> Tensor:
        if self.cfg.rich and self.side_enabled and self.has_user_branch:
            h_last = h_last + self._user_residual(side_kw["user_context"], side_kw["user_context_masks"])
        return self.final_norm(h_last)

    # ------------------------------------------------------------------ API
    def query(self, *, item_tokens: Tensor, lengths: Tensor, attention_mask: Tensor,
              position_ids: Tensor | None = None, **side_kw) -> Tensor:
        h_last, _ = self._encode(item_tokens, lengths, attention_mask, side_kw, want_sequence=False,
                                 position_ids=position_ids)
        return self._readout(h_last, side_kw)

    def hidden_states(self, *, item_tokens: Tensor, lengths: Tensor, attention_mask: Tensor,
                      position_ids: Tensor | None = None, **side_kw) -> Tensor:
        _, seq = self._encode(item_tokens, lengths, attention_mask, side_kw, want_sequence=True,
                              position_ids=position_ids)
        return seq

    def output_weight(self) -> Tensor:
        return self.output_embed                              # untied output table [K, d]

    @torch.no_grad()
    def recenter_output_(self) -> None:
        """Remove the common mode of the output table and bias (after each optimizer step).
        Subtracting the row-mean vector from every W_out row and the mean from b shifts every
        logit of a query by the same constant: exactly loss-neutral (softmax shift invariance)."""
        self.output_embed.sub_(self.output_embed.mean(0, keepdim=True))
        self.output_bias.sub_(self.output_bias.mean())

    @torch.no_grad()
    def embedding_stats(self, sample_rows: Tensor) -> dict:
        """Cheap drift monitor: common-mode and median row norms of the input and output tables."""
        E = self.item_embed.weight[3:3 + self.K]
        W = self.output_embed
        return {"E_mean_row_norm": float(E.mean(0).norm()),
                "E_median_row_norm": float(E.index_select(0, sample_rows).norm(dim=1).median()),
                "W_mean_row_norm": float(W.mean(0).norm()),
                "W_median_row_norm": float(W.index_select(0, sample_rows).norm(dim=1).median()),
                "b_mean": float(self.output_bias.mean()), "b_std": float(self.output_bias.std())}

    def forward(self, *, item_tokens: Tensor, lengths: Tensor, attention_mask: Tensor,
                position_ids: Tensor | None = None, category_tokens: Tensor | None = None, brand_tokens: Tensor | None = None,
                main_category_tokens: Tensor | None = None, daypart_tokens: Tensor | None = None,
                weekday_tokens: Tensor | None = None, event_numeric_features: Tensor | None = None,
                event_quality_flags: Tensor | None = None, user_context: Tensor | None = None,
                user_context_masks: Tensor | None = None) -> Tensor:
        side_kw = {"category_tokens": category_tokens, "brand_tokens": brand_tokens,
                   "main_category_tokens": main_category_tokens, "daypart_tokens": daypart_tokens,
                   "weekday_tokens": weekday_tokens, "event_numeric_features": event_numeric_features,
                   "event_quality_flags": event_quality_flags, "user_context": user_context,
                   "user_context_masks": user_context_masks}
        if not self.cfg.rich:
            given = [k for k, v in side_kw.items() if v is not None]
            if given:
                raise ValueError(f"item-only model must not receive side channels {given}")
        q = self.query(item_tokens=item_tokens, lengths=lengths, attention_mask=attention_mask,
                       position_ids=position_ids, **side_kw)
        return torch.addmm(self.output_bias, q, self.output_weight().t())


def build_sasrec(cfg: SASRecConfig, *, seed: int, device, product_idx_of_class, vocab_sizes: dict,
                 impl: str = "unpadded", widths=None) -> SASRecCE:
    """Construct a SASRecCE and move it to `device`."""
    model = SASRecCE(cfg, vocab_sizes=vocab_sizes, product_idx_of_class=product_idx_of_class,
                     seed=seed, impl=impl, widths=widths)
    return model.to(device)


def parameter_inventory(model: SASRecCE) -> dict:
    inv = {name: sum(p.numel() for p in mod.parameters()) for name, mod in model.named_children()}
    for name, p in model.named_parameters(recurse=False):
        inv[name] = p.numel()
    inv["total"] = sum(p.numel() for p in model.parameters())
    return inv
