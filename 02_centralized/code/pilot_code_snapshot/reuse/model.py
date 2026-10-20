"""ContextGRU-v1 -- Task T-A.

Implements the model module specified in ``FABLE_PLAN.md`` section B
(``.local_ai/gru_v1/FABLE_PLAN.md``): a per-decision, feature-group-driven GRU
encoder over a bounded, zero-initial-state window with a tied item-embedding
output head. This module is data-free: it consumes whatever tensors a caller
(loader adapter or test) hands it and has no dependency on the frozen dataset,
``rees46_eda``, or ``ppsi/``.

Feature-group channel presence/widths mirror
``rees46_nextitem_v1/model_ready/FEATURE_GROUPS.json``
(``feature_set_id: rees46_next_item_shared_v4_localtime``):

    F0_item_only            -- item only
    F1_core_categorical     -- + category, brand, main_category
    F2_time_context         -- + daypart, weekday; + 13 numeric + 15 flags
    F3_user_session_context -- + 18 numeric (all but log1p_price) + user ctx
    F4_rich_side_information-- + all 19 numeric (incl. log1p_price) + user ctx

Only channel *counts* (not per-column semantics) matter to this module: the
loader/adapter is responsible for handing over ``event_numeric_features`` /
``event_quality_flags`` tensors already restricted to a group's columns, in
whatever fixed order it uses; this module simply concatenates and projects
them.

Implementation only -- no architecture changes. If something here looks like
a design decision that FABLE_PLAN.md section B does not pin down, it is
resolved conservatively per the literal wording of the task brief, not
"improved".
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor, nn

__all__ = [
    "ContextGRU",
    "build_model",
    "resolve_group_spec",
    "parameter_inventory",
]


# ---------------------------------------------------------------------------
# Feature-group specs (FEATURE_GROUPS.json mirror -- channel presence/widths
# only; the loader/adapter owns exact per-column semantics and ordering).
# ---------------------------------------------------------------------------

_KNOWN_GROUPS: dict[str, dict] = {
    "F0_item_only": {
        "categorical": ["item_tokens"],
        "numeric_dim": 0,
        "flags_dim": 0,
        "user_context": False,
    },
    "F1_core_categorical": {
        "categorical": [
            "item_tokens", "category_tokens", "brand_tokens", "main_category_tokens",
        ],
        "numeric_dim": 0,
        "flags_dim": 0,
        "user_context": False,
    },
    "F2_time_context": {
        "categorical": [
            "item_tokens", "category_tokens", "brand_tokens", "main_category_tokens",
            "daypart_tokens", "weekday_tokens",
        ],
        "numeric_dim": 13,
        "flags_dim": 15,
        "user_context": False,
    },
    "F3_user_session_context": {
        "categorical": [
            "item_tokens", "category_tokens", "brand_tokens", "main_category_tokens",
            "daypart_tokens", "weekday_tokens",
        ],
        "numeric_dim": 18,  # F4 - log1p_price
        "flags_dim": 15,
        "user_context": True,
    },
    "F4_rich_side_information": {
        "categorical": [
            "item_tokens", "category_tokens", "brand_tokens", "main_category_tokens",
            "daypart_tokens", "weekday_tokens",
        ],
        "numeric_dim": 19,
        "flags_dim": 15,
        "user_context": True,
    },
}


def resolve_group_spec(group) -> dict:
    """Resolve `group` (a known group name, or an already-resolved spec dict)
    into a spec dict with keys ``categorical``, ``numeric_dim``, ``flags_dim``,
    ``user_context``.
    """
    if isinstance(group, dict):
        return group
    try:
        return _KNOWN_GROUPS[group]
    except KeyError:
        raise KeyError(
            f"unknown feature group {group!r}; known groups: {sorted(_KNOWN_GROUPS)}"
        ) from None


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class ContextGRU(nn.Module):
    """ContextGRU-v1 sequence encoder + tied output head (FABLE_PLAN.md sec. B).

    Built from a resolved feature-group spec::

        spec = {
            "categorical": [...channel names, must include "item_tokens"...],
            "numeric_dim": int,   # width of event_numeric_features this group carries
            "flags_dim": int,     # width of event_quality_flags this group carries
            "user_context": bool, # late-fuse user_context/user_context_masks at readout
        }

    Channels absent from `spec` have NO corresponding parameters at all (no
    embedding table, no projection) -- verified by inspecting `state_dict()`.
    """

    _EMBED_DIM = {
        "item_tokens": 128,
        "category_tokens": 32,
        "brand_tokens": 32,
        "main_category_tokens": 8,
        "daypart_tokens": 4,
        "weekday_tokens": 4,
    }
    _ATTR = {
        "item_tokens": "item_embed",
        "category_tokens": "category_embed",
        "brand_tokens": "brand_embed",
        "main_category_tokens": "main_category_embed",
        "daypart_tokens": "daypart_embed",
        "weekday_tokens": "weekday_embed",
    }
    _USER_CONTEXT_DIM = 12
    _USER_CONTEXT_MASK_DIM = 6

    def __init__(
        self,
        spec: dict,
        vocab_sizes: dict[str, int],
        *,
        product_idx_of_class,
        proj_dropout: float = 0.20,
        readout_dropout: float = 0.10,
    ) -> None:
        super().__init__()

        categorical = list(spec["categorical"])
        if "item_tokens" not in categorical:
            raise ValueError("item_tokens is mandatory in every feature group")

        numeric_dim = int(spec.get("numeric_dim", 0))
        flags_dim = int(spec.get("flags_dim", 0))
        use_user_context = bool(spec.get("user_context", False))

        self.spec = dict(spec)
        self._categorical_channels = tuple(categorical)
        self.numeric_dim = numeric_dim
        self.flags_dim = flags_dim
        self.use_user_context = use_user_context

        # -- per-channel embeddings: ONLY for channels this group provides --
        width = 0
        for ch in categorical:
            if ch not in self._EMBED_DIM:
                raise ValueError(f"unknown categorical channel {ch!r}")
            if ch not in vocab_sizes:
                raise KeyError(f"vocab_sizes missing entry for required channel {ch!r}")
            dim = self._EMBED_DIM[ch]
            padding_idx = 0 if ch == "item_tokens" else None
            emb = nn.Embedding(int(vocab_sizes[ch]), dim, padding_idx=padding_idx)
            setattr(self, self._ATTR[ch], emb)
            width += dim
        self.item_vocab_size = int(vocab_sizes["item_tokens"])

        # -- numeric(19)+flags(15)=34 -> Linear(34,32)+GELU, only if present --
        if numeric_dim > 0 or flags_dim > 0:
            self.numeric_flags_proj = nn.Sequential(
                nn.Linear(numeric_dim + flags_dim, 32), nn.GELU()
            )
            width += 32
        else:
            self.numeric_flags_proj = None

        self.concat_width = width
        self.input_proj = nn.Linear(width, 128)
        self.input_norm = nn.LayerNorm(128)
        self.input_dropout = nn.Dropout(proj_dropout)

        # -- backbone: zero initial state (nn.GRU default when h_0 omitted) --
        self.gru = nn.GRU(128, 256, num_layers=1, batch_first=True, dropout=0)

        # -- user context: late fusion at readout ONLY --
        if use_user_context:
            self.user_ctx_proj = nn.Sequential(
                nn.Linear(self._USER_CONTEXT_DIM + self._USER_CONTEXT_MASK_DIM, 64),
                nn.GELU(),
            )
            readout_in_dim = 256 + 64
        else:
            self.user_ctx_proj = None
            readout_in_dim = 256

        self.readout_proj = nn.Linear(readout_in_dim, 128)
        self.readout_dropout = nn.Dropout(readout_dropout)

        # -- tied output head --
        poc = torch.as_tensor(product_idx_of_class, dtype=torch.long)
        if poc.ndim != 1:
            raise ValueError("product_idx_of_class must be 1-D")
        K = int(poc.shape[0])
        if K == 0:
            raise ValueError("product_idx_of_class must be non-empty")
        if int(poc.min()) < 0 or int(poc.max()) >= self.item_vocab_size:
            raise IndexError(
                "product_idx_of_class references rows outside the item table "
                f"(item vocab size {self.item_vocab_size})"
            )
        contiguous = bool(torch.equal(poc, torch.arange(3, 3 + K, dtype=torch.long)))
        self.contiguous_head = contiguous
        self._contig_start = 3 if contiguous else None
        self.register_buffer("product_idx_of_class", poc)
        self.K = K
        # Exactly one item Parameter exists (self.item_embed.weight); the
        # output head reuses it directly at forward time (view or gather),
        # never copies or detaches it. `output_bias` is the only head-owned
        # Parameter.
        self.output_bias = nn.Parameter(torch.zeros(K))

    # -- internal building blocks (also exercised directly by tests) --------

    def _embed_and_project(
        self,
        *,
        item_tokens: Tensor,
        category_tokens: Optional[Tensor] = None,
        brand_tokens: Optional[Tensor] = None,
        main_category_tokens: Optional[Tensor] = None,
        daypart_tokens: Optional[Tensor] = None,
        weekday_tokens: Optional[Tensor] = None,
        event_numeric_features: Optional[Tensor] = None,
        event_quality_flags: Optional[Tensor] = None,
    ) -> Tensor:  # [B, L, 128]
        parts = [self.item_embed(item_tokens)]

        if "category_tokens" in self._categorical_channels:
            if category_tokens is None:
                raise ValueError("this model requires category_tokens")
            parts.append(self.category_embed(category_tokens))
        if "brand_tokens" in self._categorical_channels:
            if brand_tokens is None:
                raise ValueError("this model requires brand_tokens")
            parts.append(self.brand_embed(brand_tokens))
        if "main_category_tokens" in self._categorical_channels:
            if main_category_tokens is None:
                raise ValueError("this model requires main_category_tokens")
            parts.append(self.main_category_embed(main_category_tokens))
        if "daypart_tokens" in self._categorical_channels:
            if daypart_tokens is None:
                raise ValueError("this model requires daypart_tokens")
            parts.append(self.daypart_embed(daypart_tokens))
        if "weekday_tokens" in self._categorical_channels:
            if weekday_tokens is None:
                raise ValueError("this model requires weekday_tokens")
            parts.append(self.weekday_embed(weekday_tokens))

        if self.numeric_flags_proj is not None:
            bits = []
            if self.numeric_dim > 0:
                if event_numeric_features is None:
                    raise ValueError("this model requires event_numeric_features")
                bits.append(event_numeric_features)
            if self.flags_dim > 0:
                if event_quality_flags is None:
                    raise ValueError("this model requires event_quality_flags")
                bits.append(event_quality_flags.to(torch.float32))  # int8 -> float32
            nf = bits[0] if len(bits) == 1 else torch.cat(bits, dim=-1)
            parts.append(self.numeric_flags_proj(nf))

        x = torch.cat(parts, dim=-1)          # [B, L, concat_width]
        x = self.input_proj(x)                # [B, L, 128]
        x = self.input_norm(x)
        x = self.input_dropout(x)
        return x

    def encode(
        self,
        *,
        item_tokens: Tensor,
        lengths: Tensor,
        attention_mask: Tensor,
        category_tokens: Optional[Tensor] = None,
        brand_tokens: Optional[Tensor] = None,
        main_category_tokens: Optional[Tensor] = None,
        daypart_tokens: Optional[Tensor] = None,
        weekday_tokens: Optional[Tensor] = None,
        event_numeric_features: Optional[Tensor] = None,
        event_quality_flags: Optional[Tensor] = None,
    ) -> Tensor:  # [B, 256] -- backbone readout, BEFORE user-context fusion
        if attention_mask.shape != item_tokens.shape:
            raise ValueError("attention_mask and item_tokens must share shape [B,L]")

        x = self._embed_and_project(
            item_tokens=item_tokens, category_tokens=category_tokens,
            brand_tokens=brand_tokens, main_category_tokens=main_category_tokens,
            daypart_tokens=daypart_tokens, weekday_tokens=weekday_tokens,
            event_numeric_features=event_numeric_features,
            event_quality_flags=event_quality_flags,
        )

        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, h_n = self.gru(packed)             # zero initial state, no state carry
        return h_n[-1]                        # [B, 256] -- NEVER output[:, L-1]

    def query(
        self,
        *,
        item_tokens: Tensor,
        lengths: Tensor,
        attention_mask: Tensor,
        category_tokens: Optional[Tensor] = None,
        brand_tokens: Optional[Tensor] = None,
        main_category_tokens: Optional[Tensor] = None,
        daypart_tokens: Optional[Tensor] = None,
        weekday_tokens: Optional[Tensor] = None,
        event_numeric_features: Optional[Tensor] = None,
        event_quality_flags: Optional[Tensor] = None,
        user_context: Optional[Tensor] = None,
        user_context_masks: Optional[Tensor] = None,
    ) -> Tensor:  # [B, 128] -- post-readout query, pre-head
        h_last = self.encode(
            item_tokens=item_tokens, lengths=lengths, attention_mask=attention_mask,
            category_tokens=category_tokens, brand_tokens=brand_tokens,
            main_category_tokens=main_category_tokens, daypart_tokens=daypart_tokens,
            weekday_tokens=weekday_tokens, event_numeric_features=event_numeric_features,
            event_quality_flags=event_quality_flags,
        )

        if self.user_ctx_proj is not None:
            if user_context is None or user_context_masks is None:
                raise ValueError("this model requires user_context and user_context_masks")
            uc_in = torch.cat([user_context, user_context_masks], dim=-1)  # [B, 18]
            uc = self.user_ctx_proj(uc_in)                                 # [B, 64]
            readout_in = torch.cat([h_last, uc], dim=-1)                   # [B, 320]
        else:
            readout_in = h_last                                           # [B, 256]

        q = self.readout_proj(readout_in)     # [B, 128]
        q = self.readout_dropout(q)
        return q

    def forward(
        self,
        *,
        item_tokens: Tensor,
        lengths: Tensor,
        attention_mask: Tensor,
        category_tokens: Optional[Tensor] = None,
        brand_tokens: Optional[Tensor] = None,
        main_category_tokens: Optional[Tensor] = None,
        daypart_tokens: Optional[Tensor] = None,
        weekday_tokens: Optional[Tensor] = None,
        event_numeric_features: Optional[Tensor] = None,
        event_quality_flags: Optional[Tensor] = None,
        user_context: Optional[Tensor] = None,
        user_context_masks: Optional[Tensor] = None,
    ) -> Tensor:  # [B, K]
        q = self.query(
            item_tokens=item_tokens, lengths=lengths, attention_mask=attention_mask,
            category_tokens=category_tokens, brand_tokens=brand_tokens,
            main_category_tokens=main_category_tokens, daypart_tokens=daypart_tokens,
            weekday_tokens=weekday_tokens, event_numeric_features=event_numeric_features,
            event_quality_flags=event_quality_flags, user_context=user_context,
            user_context_masks=user_context_masks,
        )

        if self.contiguous_head:
            w_out = self.item_embed.weight[self._contig_start:self._contig_start + self.K]
        else:
            w_out = self.item_embed.weight.index_select(0, self.product_idx_of_class)

        logits = q @ w_out.T + self.output_bias  # [B, K]
        return logits


def build_model(
    group,
    vocab_sizes: dict[str, int],
    *,
    product_idx_of_class,
    proj_dropout: float = 0.20,
    readout_dropout: float = 0.10,
) -> ContextGRU:
    """Factory: resolve `group` (name or spec dict) and construct a ContextGRU.

    `vocab_sizes` maps channel name -> embedding row count (vocab size passed
    in, never hard-coded), e.g. ``{"item_tokens": 158489, "category_tokens":
    663, "brand_tokens": 3956, "main_category_tokens": 16, "daypart_tokens": 8,
    "weekday_tokens": 10}``. Only entries needed by `group`'s categorical
    channels are required; extras are ignored.
    """
    spec = resolve_group_spec(group)
    return ContextGRU(
        spec,
        vocab_sizes,
        product_idx_of_class=product_idx_of_class,
        proj_dropout=proj_dropout,
        readout_dropout=readout_dropout,
    )


def parameter_inventory(model: ContextGRU) -> dict[str, int]:
    """Per-component parameter counts plus the total.

    Uses `named_children()`/`named_parameters(recurse=False)` so the tied
    item embedding is counted exactly once (its output-head use is a
    reference into the same Parameter, not a second one).
    """
    inv: dict[str, int] = {}
    for name, module in model.named_children():
        inv[name] = sum(p.numel() for p in module.parameters())
    for name, p in model.named_parameters(recurse=False):
        inv[name] = p.numel()
    inv["total"] = sum(p.numel() for p in model.parameters())
    return inv
