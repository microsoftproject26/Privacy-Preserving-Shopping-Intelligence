# Task T-A result — ContextGRU-v1 model module

## Files created

- `F:\Queens University\Microsoft Project\eda\.local_ai\gru_v1\code\model.py`
  — `ContextGRU` (`nn.Module`), `build_model(group, vocab_sizes, *, product_idx_of_class, ...)`,
  `resolve_group_spec(group)`, `parameter_inventory(model)`.
- `F:\Queens University\Microsoft Project\eda\.local_ai\gru_v1\code\test_model.py`
  — the 5 required test scenarios (7 test functions incl. 2 strict-xfail variants).
- `F:\Queens University\Microsoft Project\eda\.local_ai\gru_v1\task_T-A_result.md` — this report.

**Deviation from the allowed-files list:** `code\__init__.py` was **not created**.
Reason (discovered empirically, not a design choice): with `code/` on `sys.path`
(which `python -m pytest` puts there via the CWD), a package literally named
`code` shadows the Python **standard-library** `code` module. `pdb` (imported
internally by `pytest`'s own `pytest_configure`, before any test collection)
does `import code`, resolves it to our package instead of the stdlib module,
and `torch.distributed`'s `class _DistributedPdb(pdb.Pdb)` then fails with
`AttributeError: partially initialized module 'pdb' ... no attribute 'Pdb'`
(a circular-import `INTERNALERROR` that aborts the whole pytest session before
any test runs, independent of `--import-mode`). This reproduced identically
with an **empty** `__init__.py` — the mere presence of the file makes `code`
an importable package and triggers the collision, regardless of content.
Deleting `__init__.py` fixed it: `code/` is then only a plain directory (no
namespace/regular package identity forcing early resolution), and Python's
import machinery correctly prefers the stdlib `code.py` found later on
`sys.path` over the (now non-package) local directory. `model.py` and
`test_model.py` do not depend on `code` being a package (`test_model.py` uses
`from model import ...`, a plain same-directory import). Functionally nothing
is lost — the brief said `__init__.py` "may be empty" — but strictly speaking
this is one allowed file not written. Flagging this rather than silently
picking a rule.

## Test command and result

```
cd "F:\Queens University\Microsoft Project\eda\.local_ai\gru_v1"
"F:\Queens University\Microsoft Project\eda\next_item_eda\.venv\Scripts\python.exe" -m pytest code\test_model.py -v
```

Exit code: **0**

```
============================= test session starts =============================
platform win32 -- Python 3.13.11, pytest-9.1.1, pluggy-1.6.0
rootdir: F:\Queens University\Microsoft Project\eda\.local_ai\gru_v1
collected 7 items

code/test_model.py::test_shapes_f4_returns_2d PASSED                     [ 14%]
code/test_model.py::test_t4_class_map_contiguous_vs_permuted PASSED      [ 28%]
code/test_model.py::test_t5_tied_gradient_and_inventory PASSED           [ 42%]
code/test_model.py::test_t6_feature_group_exclusion_f0 PASSED            [ 57%]
code/test_model.py::test_t1_padding_invariance PASSED                    [ 71%]
code/test_model.py::test_t1_broken_trailing_pad_readout XFAIL (trailing... [ 85%]
code/test_model.py::test_t1_broken_left_padded_input XFAIL (left-padded... [100%]

======================== 5 passed, 2 xfailed in 1.55s =========================
```

**Test counts: 5 passed / 0 failed / 2 xfailed (both strict, both genuinely
xfail — not xpass) / 7 collected total.**

Coverage against the brief's 5 required scenarios:

1. **Shapes** — `test_shapes_f4_returns_2d`: F4 model, B=8, L=50, mixed
   lengths incl. 1 and 50, K=1000 → output `[8, 1000]`, `logits.dim() == 2`. PASSED.
2. **T4 class-map** — `test_t4_class_map_contiguous_vs_permuted`: contiguous
   map (`arange(K)+3`) → `contiguous_head is True`; permuted map →
   `contiguous_head is False` and logits match the brute-force
   `q @ E_item.weight[product_idx_of_class].T + b` within 1e-5 (max|Δ|
   observed ≈ 0, same computation path). PASSED.
3. **T5 tied gradient + inventory** — `test_t5_tied_gradient_and_inventory`:
   after one CE backward, `E_item.weight.grad[3+t]` non-zero for an in-batch
   target, `grad[0]` (PAD) exactly zero (guaranteed by `padding_idx=0`);
   exactly one parameter tensor of shape `[item_rows, 128]`; `parameter_inventory`
   printed (see below) and its `total` matches `sum(p.numel() for p in
   model.parameters())`. PASSED.
4. **T6 feature-group exclusion** — `test_t6_feature_group_exclusion_f0`: F0
   output is bit-identical (`torch.equal`) when every non-item channel in the
   batch is permuted; F0 `state_dict()` has no key matching
   `category|brand|main|daypart|weekday|user|numeric|flags`. PASSED.
5. **T1 padding invariance** — `test_t1_padding_invariance`: batched packed
   `encode()` (readout `h_n[-1]`, `[B,256]`) vs independent per-row unpadded
   forward, `model.eval()` (dropout off), FP32, `allow_tf32=False`, rows of
   length 1 and 50 included; max|Δ| ≤ 1e-5. PASSED.
   - `test_t1_broken_trailing_pad_readout` (`xfail(strict=True)`): readout
     taken as `output[:, L-1]` (fixed index) instead of `h_n[-1]` — diverges
     for short rows as expected. **XFAIL (genuine)**.
   - `test_t1_broken_left_padded_input` (`xfail(strict=True)`): same rows
     re-laid out LEFT-padded while still packed with the same (unchanged)
     `lengths` — `pack_padded_sequence` then reads the wrong (pad+partial)
     prefix, diverges as expected. **XFAIL (genuine)**.

## Parameter inventory — real vocabulary

`vocab_sizes = {item_tokens: 158489, category_tokens: 663, brand_tokens: 3956,
main_category_tokens: 16, daypart_tokens: 8, weekday_tokens: 10}`, `K = 158486`,
`product_idx_of_class = arange(K) + 3` (contiguous head both cases).

### F0_item_only

```
item_embed:        20,286,592
input_proj:            16,512
input_norm:               256
input_dropout:              0
gru:                   296,448
readout_proj:           32,896
readout_dropout:            0
output_bias:           158,486
------------------------------
total:              20,791,190
```

**F0 total = 20,791,190 — matches expected 20,791,190 exactly.**

### F4_rich_side_information

```
item_embed:        20,286,592
category_embed:        21,216
brand_embed:           126,592
main_category_embed:       128
daypart_embed:              32
weekday_embed:              40
numeric_flags_proj:      1,120
input_proj:              30,848
input_norm:                  256
input_dropout:                0
gru:                    296,448
user_ctx_proj:             1,216
readout_proj:             41,088
readout_dropout:               0
output_bias:              158,486
--------------------------------
total:                20,964,062
```

**F4 total = 20,964,062 — matches expected 20,964,062 exactly.**

Both totals were verified with a standalone script
(`build_model("F0_item_only"/"F4_rich_side_information", VOCAB,
product_idx_of_class=arange(158486)+3)` + `parameter_inventory(model)`), run
under the same local venv python, printed and diffed against the two numbers
in `FABLE_PLAN.md` section B. `contiguous_head` was `True` for both, as
expected for `product_idx_of_class = arange(K)+3`.

## Design notes (implementation choices, not architecture changes)

- `ContextGRU` is built from a resolved spec dict (`categorical` channel
  list, `numeric_dim`, `flags_dim`, `user_context` bool); `build_model`
  resolves a known group name (`F0_item_only` … `F4_rich_side_information`,
  channel/width mirror of `FEATURE_GROUPS.json`) or accepts an already-resolved
  spec dict directly. Only channel **counts** matter to the module — exact
  per-column numeric ordering is the loader/adapter's responsibility.
- Channels absent from a group's spec have **no corresponding `nn.Module`**
  at all (verified by T6): no `category_embed`/`brand_embed`/etc.,
  no `numeric_flags_proj`, no `user_ctx_proj` for F0.
- `padding_idx=0` is set **only** on the item embedding, per the literal
  wording of the brief ("item 128 dims, `padding_idx=0`; category 32; brand
  32; …" — the qualifier is attached only to item). This has no effect on
  any tested behavior either way: padded timesteps are excluded from the GRU
  entirely by `pack_padded_sequence`, so any category/brand/etc. embedding
  looked up at a padded position never reaches the backbone regardless of
  whether its row 0 is zeroed.
- Internal helpers `encode()` (→ `[B,256]`, the GRU readout before
  user-context fusion) and `query()` (→ `[B,128]`, post-readout,
  pre-head) were factored out of `forward()` so the T1 padding-invariance
  test (which must compare the `[B,256]` backbone readout, per the brief)
  doesn't have to duplicate model internals. `forward()` calls `query()`
  then applies the tied head; behavior/outputs are unchanged from a single
  monolithic `forward()`. `_embed_and_project()` (→ `[B,L,128]`, pre-GRU) is
  also exposed for the same reason (used by the deliberately-broken
  trailing-PAD test variant).
- Tied head: exactly one `nn.Parameter` of item-embedding shape exists
  (`self.item_embed.weight`); the output head reuses it via a slice **view**
  (contiguous case) or `index_select` (permuted case) — never a copy,
  `.detach()`, or a second `nn.Linear`. `output_bias` (`[K]`, zero-init) is
  the only head-owned parameter.

## Unresolved concerns

1. **`code\__init__.py` was not created** — see "Deviation" above. This is
   the one departure from the literal allowed-files list; it is forced by a
   name collision with the Python stdlib `code` module in this exact
   environment (Windows, Python 3.13, torch 2.9.1+cpu with
   `torch.distributed`'s `pdb.Pdb` subclassing at import time), not a design
   choice, and does not affect the frozen dataset, `ppsi/`, or the
   architecture. If a package marker is required for a later integration
   step, it should be added at that point together with a matching
   `--import-mode=importlib` (or equivalent) pytest invocation, since a plain
   empty `__init__.py` reproduces the same `INTERNALERROR` under the exact
   command given in the brief.
2. Numeric/flags channel **order** is not validated by `ContextGRU` itself —
   only total width. The module trusts the caller (loader/adapter, out of
   scope for T-A) to hand over `event_numeric_features`/`event_quality_flags`
   already restricted and ordered per `FEATURE_GROUPS.json` /
   `feature_group_view()`. This seemed like the correct boundary (T-A is
   model-only, data-free) but flagging it in case a later integration test
   expects the model to assert column identity itself.
3. F1/F2/F3 group specs are included in `build_model`'s known-group table for
   completeness/generality (`ContextGRU` is explicitly "built from a
   feature-group spec"), but per `FABLE_PLAN.md` §I these arms are
   **deferred** and were not covered by dedicated tests here — only F0 and F4
   were required and tested, matching the pilot plan (§F: "GRU-F0 vs GRU-F4").
