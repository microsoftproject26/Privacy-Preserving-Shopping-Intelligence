"""v4base.rule — BASELINE_RULE (contracts/PREG1_AMENDMENT_04.md §2 arms table; DOWNSTREAM_FINAL_PLAN_FABLE.md (2)):

    score(i) = 0.5 * P_u(main_category(i)) * pop_within_category(i) + 0.5 * P_u(brand(i)) * pop_within_brand(i)

P_u(*) = the user's OWN permitted-history SHARE OF VIEWS (the "share of views" wording the same amendment uses for
brand loyalty) in that category/brand: a per-decision, strictly causal quantity read straight off the batch's own
`main_category_tokens`/`brand_tokens` columns over the WHOLE causal window (v4base.last's convention: every valid
column, not just the current session) -- no catalogue lookup needed for the user's side.
pop_within_category/brand(i) = item i's OWN MAIN-TRAIN target popularity (`v4base.pop.fit`'s own [K] counts, reused
verbatim, never rebuilt) as a SHARE of its assigned category's/brand's total popularity -- the standard
"popularity conditioned on category/brand" reading that actually differentiates items sharing one category/brand
(the alternative reading, repeating the category's/brand's total for every member item, collapses to a pure
category/brand-affinity score with no per-item information and was rejected).

Judgment call — no per-item category/brand table exists anywhere in this codebase (confirmed): category/brand are
only ever stored PER EVENT, in the window's own `category_tokens`/`brand_tokens`/`main_category_tokens` columns
(contracts/DATA_CONTRACT_CLEAN_V2.md rows 2-4: "as-of in TRAIN, frozen snapshot in the future" — an item's category
CAN drift across events), never as a static item -> category catalogue (`V1/mappings/catalogue.parquet` only maps
product_id -> output_class; `vocab_main_category.parquet` / `vocab_brand.parquet` only map string -> token). Each
catalogue item's own (main_category, brand) is therefore derived by a single MAIN-TRAIN scan: for every valid
(item, main_category_token, brand_token) triple seen at ANY window position of ANY TRAIN decision (any user's
history, not just the label-producing row), accumulate a vote; each item's assigned category/brand is the MODE vote
(ties -> the smaller token id, this repo's general "fewer/smaller wins" tie spirit — v4base.grid.select_best does
the same for its own tie). This is the SAME class of information POP's own global popularity already is: a
population-level, MAIN-TRAIN-only aggregate, never a per-user secret, never touching VALIDATION/TEST. An item with
zero evidence anywhere in TRAIN gets the MISSING(1) sentinel for both, exactly like a real never-resolved event.

Token convention (contracts/DATA_CONTRACT_CLEAN_V2.md: "token: PAD 0 / MISSING 1 / OOV 2 / known >= 3", universal
across the item/category/brand/main_category channels): "known" reuses `sessions.ITEM_TOKEN_OFFSET` (=3) as the
shared boundary — the same integer, the same meaning, not a duplicated magic number. "Items with unknown
category/brand: their term is 0" (task packet): whenever an item's assigned main_category/brand token is < 3
(MISSING/OOV, or never observed), that HALF of its score is exactly 0.0 regardless of P_u — an explicit mask, never
a value that merely happens to come out near zero.

De-duplication (review fix, MED): a user's TRAIN decisions' windows overlap (a later decision's window is a
superset of an earlier one's, up to the L-cap), so scanning every decision's own window would count the SAME real
(item, category, brand) event once per later decision that still covers it (up to ~L times) — a real user with many
TRAIN decisions would dominate an item's mode vote out of proportion to how many DISTINCT users actually saw it.
`fit_category_brand_map` therefore uses `sessions.scan_best_window_per_user` (per-user grouping key supplied by the
caller — see `v4base.identity` for how a real caller resolves it; a within-this-scan-only key, never a cross-
manifest one) to read each user's single longest window exactly once.

Disclosure (LOW, review item 5): `pop_within_category/brand(i)` is a SHARE of the category's/brand's own total
popularity, not an absolute count — a brand with very few total TRAIN views (a tiny/rare brand) can hand its one or
two items a share near 1.0, the same as a dominant item of a huge, popular brand. This is the standard behaviour of
any "share within group" statistic with a small group denominator, not a bug; Ops/Fable should read
`pop_within_category`/`pop_within_brand` alongside the RAW `pop_counts` (or the group total) when auditing RULE's
rows, not as a substitute for it.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import torch

from .guard import RssGuard
from .sessions import ITEM_TOKEN_OFFSET, history_valid_mask, scan_best_window_per_user

try:
    from v4hist.windows import MAIN_CATEGORY_VOCAB, MAIN_UNKNOWN       # read-only: documented public constants
except Exception:                                                      # pragma: no cover (always importable here)
    MAIN_CATEGORY_VOCAB, MAIN_UNKNOWN = 16, 1

DEFAULT_FIT_CHUNK = 2048
KNOWN_START = ITEM_TOKEN_OFFSET          # known >= 3, the universal Clean V2 token rule (== 3)
UNKNOWN_TOKEN = MAIN_UNKNOWN             # = 1 (MISSING), the never-observed item default


def _as_tensor(x):
    return x if torch.is_tensor(x) else torch.as_tensor(x)


def _mode_map(counts: Dict[int, Dict[int, int]], K: int, default: int) -> np.ndarray:
    """item class -> the mode-vote token (ties -> the smaller token id)."""
    out = np.full(int(K), int(default), dtype=np.int64)
    for item, c in counts.items():
        best_tok, best_n = default, -1
        for tok in sorted(c):                          # ascending token id first, so a strict '>' keeps the smaller
            if c[tok] > best_n:
                best_tok, best_n = tok, c[tok]
        out[item] = best_tok
    return out


def fit_category_brand_map(provider, K: int, *, user_keys, chunk: int = DEFAULT_FIT_CHUNK,
                           guard: Optional[RssGuard] = None) -> Tuple[np.ndarray, np.ndarray]:
    """One user-de-duplicated MAIN-TRAIN scan: (main_category_of_item, brand_of_item), both int64 [K], token-valued
    (see module docstring). `user_keys` is REQUIRED (one entry per canonical TRAIN position, aligned exactly like
    `provider.batch(positions)` — see `sessions.scan_best_window_per_user`); for a real provider this is
    `v4base.identity.train_client_ord_by_canonical_position(provider, manifest_dir=...)` (a within-this-scan-only
    grouping key is enough here — RULE never persists it, unlike `v4base.strata`'s registry key)."""
    best = scan_best_window_per_user(provider, user_keys=user_keys,
                                     extra_fields=("main_category_tokens", "brand_tokens"), chunk=chunk, guard=guard)
    cat_counts: Dict[int, Dict[int, int]] = {}
    brand_counts: Dict[int, Dict[int, int]] = {}
    for row in best.values():
        L_valid = int(row["lengths"])
        item_tokens, cats, brs = row["item_tokens"], row["main_category_tokens"], row["brand_tokens"]
        for pos in range(L_valid):
            tok = item_tokens[pos]
            if tok < ITEM_TOKEN_OFFSET:
                continue
            it = tok - ITEM_TOKEN_OFFSET
            if it < 0 or it >= K:
                continue
            d = cat_counts.setdefault(it, {})
            d[cats[pos]] = d.get(cats[pos], 0) + 1
            d2 = brand_counts.setdefault(it, {})
            d2[brs[pos]] = d2.get(brs[pos], 0) + 1
    cat_of = _mode_map(cat_counts, K, UNKNOWN_TOKEN)
    brand_of = _mode_map(brand_counts, K, UNKNOWN_TOKEN)
    return cat_of, brand_of


class RuleTables:
    """Precomputed once (task packet: "popularity from MAIN TRAIN counts, like POP"): `pop_counts`
    (`v4base.pop`'s own [K] TRAIN target histogram, reused verbatim), `cat_of`/`brand_of` (this module's own
    MAIN-TRAIN map), and each item's popularity SHARE of its own category's/brand's total popularity."""

    def __init__(self, pop_counts: torch.Tensor, cat_of, brand_of):
        K = int(pop_counts.shape[0])
        self.K = K
        self.cat_of = torch.as_tensor(cat_of, dtype=torch.int64)
        self.brand_of = torch.as_tensor(brand_of, dtype=torch.int64)
        if int(self.cat_of.shape[0]) != K or int(self.brand_of.shape[0]) != K:
            raise ValueError("cat_of/brand_of must both have K entries, matching pop_counts")
        pop = pop_counts.to(torch.float64)
        self.cat_vocab = int(max(MAIN_CATEGORY_VOCAB, int(self.cat_of.max()) + 1)) if K else MAIN_CATEGORY_VOCAB
        self.brand_vocab = int(self.brand_of.max()) + 1 if K else 1
        cat_totals = torch.zeros(self.cat_vocab, dtype=torch.float64).scatter_add_(0, self.cat_of, pop)
        brand_totals = torch.zeros(self.brand_vocab, dtype=torch.float64).scatter_add_(0, self.brand_of, pop)
        cat_tot_per_item = cat_totals[self.cat_of]
        brand_tot_per_item = brand_totals[self.brand_of]
        self.pop_within_cat = torch.where(cat_tot_per_item > 0, pop / cat_tot_per_item.clamp(min=1.0),
                                          torch.zeros_like(pop)).to(torch.float32)
        self.pop_within_brand = torch.where(brand_tot_per_item > 0, pop / brand_tot_per_item.clamp(min=1.0),
                                            torch.zeros_like(pop)).to(torch.float32)
        self.cat_known = self.cat_of >= KNOWN_START
        self.brand_known = self.brand_of >= KNOWN_START


class RuleScorer:
    def __init__(self, tables: RuleTables):
        self.t = tables
        self.K = tables.K

    def score_batch(self, batch: dict) -> torch.Tensor:
        item_tokens = _as_tensor(batch["item_tokens"])
        B, L = int(item_tokens.shape[0]), int(item_tokens.shape[1])
        valid = history_valid_mask(batch["lengths"], L)
        cat_tok = _as_tensor(batch["main_category_tokens"]).to(torch.int64)
        brand_tok = _as_tensor(batch["brand_tokens"]).to(torch.int64)
        out = torch.zeros((B, self.K), dtype=torch.float32)
        t = self.t
        zeros_k = torch.zeros(self.K, dtype=torch.float32)
        for r in range(B):
            m = valid[r]
            n_valid = int(m.sum())
            if n_valid == 0:
                continue                                             # no permitted history: all-zero (no sentinel needed)
            cats = cat_tok[r][m]
            brs = brand_tok[r][m]
            pu_cat = (torch.bincount(cats, minlength=t.cat_vocab)[:t.cat_vocab].to(torch.float32) / n_valid)
            pu_brand = (torch.bincount(brs, minlength=t.brand_vocab)[:t.brand_vocab].to(torch.float32) / n_valid)
            term_cat = torch.where(t.cat_known, pu_cat[t.cat_of] * t.pop_within_cat, zeros_k)
            term_brand = torch.where(t.brand_known, pu_brand[t.brand_of] * t.pop_within_brand, zeros_k)
            out[r] = 0.5 * term_cat + 0.5 * term_brand
        return out


__all__ = ["KNOWN_START", "UNKNOWN_TOKEN", "DEFAULT_FIT_CHUNK", "fit_category_brand_map", "RuleTables", "RuleScorer"]
