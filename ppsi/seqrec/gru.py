"""ContextGRU: a feature-group-driven GRU next-item model with a tied item-embedding output head.

Per decision, the window of past events (right-padded, zero initial state, no state carried between decisions) is
embedded channel by channel, the numeric features and quality flags are projected together, everything is
concatenated and projected to the item dimension, LayerNorm-ed and fed to a one-layer GRU. The readout is the final
hidden state of the last REAL event (packed sequence, never the last stored column), optionally late-fused with the
user-context vector, projected back to the item dimension, and scored against the item-embedding rows of the
catalogue classes (one shared Parameter: the head is a view of the item table) plus an output bias.

Feature groups (channel presence only; the caller hands over float blocks already restricted to the group's columns):

    F0_item_only             item only
    F1_core_categorical      + category, brand, main_category
    F2_time_context          + daypart, weekday; + numeric + flags
    F3_user_session_context  + more numeric + user context
    F4_rich_side_information + every numeric feature + user context

The float widths come from the spec (built from `InputWidths`); every float block is refused if its width differs
from the declared one. `item_dim` (default 128) is the item embedding / GRU input / query width; the hidden size is
256. Parameter creation order (hence the seeded default initialisation) does not depend on `item_dim`.
"""
from __future__ import annotations

from typing import ClassVar

import torch
from torch import Tensor, nn

__all__ = ["ContextGRU", "build_model", "parameter_inventory", "resolve_group_spec"]

HIDDEN = 256

_KNOWN_GROUPS: dict[str, dict] = {
    "F0_item_only": {"categorical": ["item_tokens"], "numeric_dim": 0, "flags_dim": 0, "user_context": False},
    "F1_core_categorical": {
        "categorical": ["item_tokens", "category_tokens", "brand_tokens", "main_category_tokens"],
        "numeric_dim": 0, "flags_dim": 0, "user_context": False},
    "F2_time_context": {
        "categorical": ["item_tokens", "category_tokens", "brand_tokens", "main_category_tokens",
                        "daypart_tokens", "weekday_tokens"],
        "numeric_dim": 13, "flags_dim": 15, "user_context": False},
    "F3_user_session_context": {
        "categorical": ["item_tokens", "category_tokens", "brand_tokens", "main_category_tokens",
                        "daypart_tokens", "weekday_tokens"],
        "numeric_dim": 18, "flags_dim": 15, "user_context": True},
    "F4_rich_side_information": {
        "categorical": ["item_tokens", "category_tokens", "brand_tokens", "main_category_tokens",
                        "daypart_tokens", "weekday_tokens"],
        "numeric_dim": 19, "flags_dim": 15, "user_context": True},
}


def resolve_group_spec(group) -> dict:
    """Resolve `group` (a known group name, or an already-resolved spec dict) into a spec dict with keys
    ``categorical``, ``numeric_dim``, ``flags_dim``, ``user_context`` (+ the user widths when it is True)."""
    if isinstance(group, dict):
        return group
    try:
        return _KNOWN_GROUPS[group]
    except KeyError:
        raise KeyError(f"unknown feature group {group!r}; known groups: {sorted(_KNOWN_GROUPS)}") from None


def _check_width(name: str, t: Tensor, width: int) -> None:
    """Refuse an input block whose last dimension differs from the declared width."""
    if t.shape[-1] != width:
        raise ValueError(f"{name} has width {t.shape[-1]}; the model (feature layout) expects {width}")


class ContextGRU(nn.Module):
    """GRU sequence encoder + tied output head, built from a resolved feature-group spec::

        spec = {
            "categorical": [...channel names, must include "item_tokens"...],
            "numeric_dim": int,       # width of event_numeric_features this group carries
            "flags_dim": int,         # width of event_quality_flags this group carries
            "user_context": bool,     # late-fuse user_context / user_context_masks at readout
            "user_context_dim": int,  # width of user_context (required if user_context)
            "user_mask_dim": int,     # width of user_context_masks (required if user_context)
        }

    Channels absent from `spec` have NO corresponding parameters at all (no embedding table, no projection)."""

    _EMBED_DIM: ClassVar[dict[str, int]] = {"item_tokens": 128, "category_tokens": 32, "brand_tokens": 32, "main_category_tokens": 8,
                  "daypart_tokens": 4, "weekday_tokens": 4}
    _ATTR: ClassVar[dict[str, str]] = {"item_tokens": "item_embed", "category_tokens": "category_embed", "brand_tokens": "brand_embed",
             "main_category_tokens": "main_category_embed", "daypart_tokens": "daypart_embed",
             "weekday_tokens": "weekday_embed"}

    def __init__(self, spec: dict, vocab_sizes: dict[str, int], *, product_idx_of_class,
                 proj_dropout: float = 0.20, readout_dropout: float = 0.10, item_dim: int = 128) -> None:
        super().__init__()
        d = int(item_dim)
        if d < 1:
            raise ValueError("item_dim must be a positive int")
        self.d_item = d

        categorical = list(spec["categorical"])
        if "item_tokens" not in categorical:
            raise ValueError("item_tokens is mandatory in every feature group")

        numeric_dim = int(spec.get("numeric_dim", 0))
        flags_dim = int(spec.get("flags_dim", 0))
        use_user_context = bool(spec.get("user_context", False))
        if use_user_context:
            try:
                user_context_dim = int(spec["user_context_dim"])
                user_mask_dim = int(spec["user_mask_dim"])
            except KeyError as e:
                raise KeyError("a user-context spec must carry user_context_dim and user_mask_dim "
                               f"(taken from the feature layout); missing {e}") from None
        else:
            user_context_dim = user_mask_dim = 0

        self.spec = dict(spec)
        self._categorical_channels = tuple(categorical)
        self.numeric_dim = numeric_dim
        self.flags_dim = flags_dim
        self.use_user_context = use_user_context
        self.user_context_dim = user_context_dim
        self.user_mask_dim = user_mask_dim

        # per-channel embeddings: ONLY for the channels this group provides
        width = 0
        for ch in categorical:
            if ch not in self._EMBED_DIM:
                raise ValueError(f"unknown categorical channel {ch!r}")
            if ch not in vocab_sizes:
                raise KeyError(f"vocab_sizes missing entry for required channel {ch!r}")
            dim = d if ch == "item_tokens" else self._EMBED_DIM[ch]
            padding_idx = 0 if ch == "item_tokens" else None
            setattr(self, self._ATTR[ch], nn.Embedding(int(vocab_sizes[ch]), dim, padding_idx=padding_idx))
            width += dim
        self.item_vocab_size = int(vocab_sizes["item_tokens"])

        # numeric + flags -> Linear(numeric + flags, 32) + GELU, only if present
        if numeric_dim > 0 or flags_dim > 0:
            self.numeric_flags_proj = nn.Sequential(nn.Linear(numeric_dim + flags_dim, 32), nn.GELU())
            width += 32
        else:
            self.numeric_flags_proj = None

        self.concat_width = width
        self.input_proj = nn.Linear(width, d)
        self.input_norm = nn.LayerNorm(d)
        self.input_dropout = nn.Dropout(proj_dropout)

        # backbone: zero initial state (nn.GRU default when h_0 is omitted)
        self.gru = nn.GRU(d, HIDDEN, num_layers=1, batch_first=True, dropout=0)

        # user context: late fusion at readout ONLY
        if use_user_context:
            self.user_ctx_proj = nn.Sequential(nn.Linear(user_context_dim + user_mask_dim, 64), nn.GELU())
            readout_in_dim = HIDDEN + 64
        else:
            self.user_ctx_proj = None
            readout_in_dim = HIDDEN

        self.readout_proj = nn.Linear(readout_in_dim, d)
        self.readout_dropout = nn.Dropout(readout_dropout)

        # tied output head
        poc = torch.as_tensor(product_idx_of_class, dtype=torch.long)
        if poc.ndim != 1:
            raise ValueError("product_idx_of_class must be 1-D")
        K = int(poc.shape[0])
        if K == 0:
            raise ValueError("product_idx_of_class must be non-empty")
        if int(poc.min()) < 0 or int(poc.max()) >= self.item_vocab_size:
            raise IndexError("product_idx_of_class references rows outside the item table "
                             f"(item vocab size {self.item_vocab_size})")
        contiguous = bool(torch.equal(poc, torch.arange(3, 3 + K, dtype=torch.long)))
        self.contiguous_head = contiguous
        self._contig_start = 3 if contiguous else None
        self.register_buffer("product_idx_of_class", poc)
        self.K = K
        # Exactly one item Parameter exists (self.item_embed.weight); the output head reuses it at forward time
        # (view or gather), never copies or detaches it. `output_bias` is the only head-owned Parameter.
        self.output_bias = nn.Parameter(torch.zeros(K))

    # internal building blocks (also exercised directly by tests)
    def _embed_and_project(self, *, item_tokens: Tensor, category_tokens: Tensor | None = None,
                           brand_tokens: Tensor | None = None, main_category_tokens: Tensor | None = None,
                           daypart_tokens: Tensor | None = None, weekday_tokens: Tensor | None = None,
                           event_numeric_features: Tensor | None = None,
                           event_quality_flags: Tensor | None = None) -> Tensor:   # [B, L, item_dim]
        parts = [self.item_embed(item_tokens)]
        for ch, t in (("category_tokens", category_tokens), ("brand_tokens", brand_tokens),
                      ("main_category_tokens", main_category_tokens), ("daypart_tokens", daypart_tokens),
                      ("weekday_tokens", weekday_tokens)):
            if ch in self._categorical_channels:
                if t is None:
                    raise ValueError(f"this model requires {ch}")
                parts.append(getattr(self, self._ATTR[ch])(t))

        if self.numeric_flags_proj is not None:
            bits = []
            if self.numeric_dim > 0:
                if event_numeric_features is None:
                    raise ValueError("this model requires event_numeric_features")
                _check_width("event_numeric_features", event_numeric_features, self.numeric_dim)
                bits.append(event_numeric_features)
            if self.flags_dim > 0:
                if event_quality_flags is None:
                    raise ValueError("this model requires event_quality_flags")
                _check_width("event_quality_flags", event_quality_flags, self.flags_dim)
                bits.append(event_quality_flags.to(torch.float32))  # int8 -> float32
            nf = bits[0] if len(bits) == 1 else torch.cat(bits, dim=-1)
            parts.append(self.numeric_flags_proj(nf))

        x = torch.cat(parts, dim=-1)          # [B, L, concat_width]
        x = self.input_proj(x)                # [B, L, item_dim]
        x = self.input_norm(x)
        return self.input_dropout(x)

    def encode(self, *, item_tokens: Tensor, lengths: Tensor, attention_mask: Tensor,
               category_tokens: Tensor | None = None, brand_tokens: Tensor | None = None,
               main_category_tokens: Tensor | None = None, daypart_tokens: Tensor | None = None,
               weekday_tokens: Tensor | None = None, event_numeric_features: Tensor | None = None,
               event_quality_flags: Tensor | None = None) -> Tensor:   # [B, 256], before user-context fusion
        if attention_mask.shape != item_tokens.shape:
            raise ValueError("attention_mask and item_tokens must share shape [B,L]")
        x = self._embed_and_project(
            item_tokens=item_tokens, category_tokens=category_tokens, brand_tokens=brand_tokens,
            main_category_tokens=main_category_tokens, daypart_tokens=daypart_tokens,
            weekday_tokens=weekday_tokens, event_numeric_features=event_numeric_features,
            event_quality_flags=event_quality_flags)
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, h_n = self.gru(packed)             # zero initial state, no state carry
        return h_n[-1]                        # [B, 256]: the last REAL event, never output[:, L-1]

    def query(self, *, item_tokens: Tensor, lengths: Tensor, attention_mask: Tensor,
              category_tokens: Tensor | None = None, brand_tokens: Tensor | None = None,
              main_category_tokens: Tensor | None = None, daypart_tokens: Tensor | None = None,
              weekday_tokens: Tensor | None = None, event_numeric_features: Tensor | None = None,
              event_quality_flags: Tensor | None = None, user_context: Tensor | None = None,
              user_context_masks: Tensor | None = None) -> Tensor:   # [B, item_dim]: post-readout query
        h_last = self.encode(
            item_tokens=item_tokens, lengths=lengths, attention_mask=attention_mask,
            category_tokens=category_tokens, brand_tokens=brand_tokens,
            main_category_tokens=main_category_tokens, daypart_tokens=daypart_tokens,
            weekday_tokens=weekday_tokens, event_numeric_features=event_numeric_features,
            event_quality_flags=event_quality_flags)
        if self.user_ctx_proj is not None:
            if user_context is None or user_context_masks is None:
                raise ValueError("this model requires user_context and user_context_masks")
            _check_width("user_context", user_context, self.user_context_dim)
            _check_width("user_context_masks", user_context_masks, self.user_mask_dim)
            uc = self.user_ctx_proj(torch.cat([user_context, user_context_masks], dim=-1))   # [B, 64]
            readout_in = torch.cat([h_last, uc], dim=-1)                                       # [B, 320]
        else:
            readout_in = h_last                                                                # [B, 256]
        q = self.readout_proj(readout_in)
        return self.readout_dropout(q)

    def forward(self, *, item_tokens: Tensor, lengths: Tensor, attention_mask: Tensor,
                category_tokens: Tensor | None = None, brand_tokens: Tensor | None = None,
                main_category_tokens: Tensor | None = None, daypart_tokens: Tensor | None = None,
                weekday_tokens: Tensor | None = None, event_numeric_features: Tensor | None = None,
                event_quality_flags: Tensor | None = None, user_context: Tensor | None = None,
                user_context_masks: Tensor | None = None) -> Tensor:   # [B, K]
        q = self.query(
            item_tokens=item_tokens, lengths=lengths, attention_mask=attention_mask,
            category_tokens=category_tokens, brand_tokens=brand_tokens,
            main_category_tokens=main_category_tokens, daypart_tokens=daypart_tokens,
            weekday_tokens=weekday_tokens, event_numeric_features=event_numeric_features,
            event_quality_flags=event_quality_flags, user_context=user_context,
            user_context_masks=user_context_masks)
        if self.contiguous_head:
            w_out = self.item_embed.weight[self._contig_start:self._contig_start + self.K]
        else:
            w_out = self.item_embed.weight.index_select(0, self.product_idx_of_class)
        return q @ w_out.T + self.output_bias


def build_model(group, vocab_sizes: dict[str, int], *, product_idx_of_class, proj_dropout: float = 0.20,
                readout_dropout: float = 0.10, item_dim: int = 128) -> ContextGRU:
    """Resolve `group` (name or spec dict) and construct a ContextGRU. `vocab_sizes` maps channel name -> embedding
    row count; only the entries needed by the group's categorical channels are required."""
    return ContextGRU(resolve_group_spec(group), vocab_sizes, product_idx_of_class=product_idx_of_class,
                      proj_dropout=proj_dropout, readout_dropout=readout_dropout, item_dim=item_dim)


def parameter_inventory(model: ContextGRU) -> dict[str, int]:
    """Per-component parameter counts plus the total; the tied item embedding is counted exactly once."""
    inv: dict[str, int] = {}
    for name, module in model.named_children():
        inv[name] = sum(p.numel() for p in module.parameters())
    for name, p in model.named_parameters(recurse=False):
        inv[name] = p.numel()
    inv["total"] = sum(p.numel() for p in model.parameters())
    return inv
