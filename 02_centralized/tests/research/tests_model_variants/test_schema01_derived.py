"""SCHEMA-01: compact derived schemas in v4models (derived.py, build.py, schema.gru_spec, sasrec_v2 class-C hunks).

  * L0 bitwise guard: the full Clean V2 as a derived schema builds bitwise the MODEL-01 module (state_dict, theta0 /
    raw hashes, manifest digest, inventory), the projection is the identity (the batch object itself), and eval logits
    (both SASRec implementations), train-mode logits and EVERY gradient are bitwise equal under the same dropout RNG;
    at the full K the L0 theta0 equals the REGISTERED hash of the running torch build (MODEL-01 inventory / FL-02 GPU
    receipt) and the manifest digest equals the committed inventory;
  * projection correctness against a hand-built batch (every kept column is the named base column; unkept tokens and
    absent blocks are removed; name arrays follow; the base layout is checked first);
  * widths and parameter counts of EVERY FEATSTAT rung at the full K equal the ladder table, for both families (and the
    first-layer widths: GRU concat / numeric_flags_proj / user_ctx_proj, SASRec numflag_proj / user_mlp);
  * the user branch is absent at L5 (no parameter, no input taken; user inputs cannot change the output), and L9 is
    the item-only topology;
  * derived-schema refusals (self-hash, base line, order, item_tokens, widths, unknown channels).
Negative controls (strict xfail): a projection that drops item_tokens; an L0 build that is not bitwise (tokens in a
non-canonical order change the init order).
"""
from __future__ import annotations

import copy
import dataclasses
import gc
import json

import pytest
import torch

from fl.numerics import states_equal
from models_testkit import FAMS, K_SMALL, SCHEMA, V2, assert_raises, batch, nc
from test_inventory import INV, V1_ROOT, needs_v1, registered_init
from v4fs import schemas as S
from v4models.build import build, make_adapter, module_forward
from v4models.derived import derived_from_obj, full_keep, keep_from_names, make_derived, self_hash
from v4models.schema import SchemaError, WidthError

SEED = 2026
L0 = derived_from_obj(make_derived(SCHEMA, "L0_CLEAN_V2", full_keep(SCHEMA)), SCHEMA)
LADDER = S.ladder()
RUNGS = {r["id"]: r for r in LADDER["rungs"]}


def rung_ds(rid: str):
    return derived_from_obj(make_derived(SCHEMA, rid, keep_from_names(SCHEMA, S.rung_channel_names(RUNGS[rid]))),
                            SCHEMA)


L3 = rung_ds("L3_NEAR_DUPLICATE")
L5 = rung_ds("L5_NO_USER_CONTEXT_BLOCK")
L9 = rung_ds("L9_ID_ONLY_ANCHOR")


def _fwd_grads(bm, b, train: bool, impl_seed: int = 7):
    m = bm.module
    m.train(train)
    m.zero_grad(set_to_none=True)
    torch.manual_seed(impl_seed)                       # the dropout stream (global RNG)
    y = bm.adapter.scores(b)
    if not train:
        return y.detach(), {}
    y.float().pow(2).mean().backward()
    return y.detach(), {n: p.grad.detach().clone() for n, p in m.named_parameters() if p.grad is not None}


# ------------------------------------------------------------------------------------------------ L0 bitwise guard
@pytest.mark.parametrize("family", FAMS)
def test_l0_derived_is_bitwise_the_model01_build(family):
    assert L0.is_full and L0.projection.is_identity and L0.widths == V2
    a = build(family, SCHEMA, SEED, "cpu", K=K_SMALL)
    d = build(family, L0, SEED, "cpu", K=K_SMALL)
    assert states_equal(a.module.state_dict(), d.module.state_dict())
    assert list(a.module.state_dict()) == list(d.module.state_dict())           # same keys, same order
    assert (a.init_sha256, a.raw_init_sha256, a.manifest_digest) == (d.init_sha256, d.raw_init_sha256,
                                                                      d.manifest_digest)
    assert a.inventory == d.inventory and a.widths == d.widths
    assert d.schema_id == "L0_CLEAN_V2" and d.schema_self_sha256 == L0.self_sha256 and a.schema_id is None
    b = batch(9, seed=11, with_names=True)
    assert L0.projection.apply(b) is b                                          # no op is added at L0
    with torch.no_grad():
        ya, _ = _fwd_grads(a, b, train=False)
        yd, _ = _fwd_grads(d, b, train=False)
    assert torch.equal(ya, yd)
    ya, ga = _fwd_grads(a, b, train=True)
    yd, gd = _fwd_grads(d, b, train=True)
    assert torch.equal(ya, yd) and sorted(ga) == sorted(gd) and ga
    assert all(torch.equal(ga[k], gd[k]) for k in ga), "an L0 gradient differs"


def test_l0_derived_sasrec_padded_reference_is_bitwise():
    a = build("SASREC", SCHEMA, SEED, "cpu", K=K_SMALL, impl="padded_reference")
    d = build("SASREC", L0, SEED, "cpu", K=K_SMALL, impl="padded_reference")
    b = batch(6, seed=12)
    with torch.no_grad():
        assert torch.equal(_fwd_grads(a, b, False)[0], _fwd_grads(d, b, False)[0])


@needs_v1
@pytest.mark.parametrize("family", FAMS)
def test_l0_theta0_equals_the_registered_hash_full_k(family, record_property):
    torch.set_num_threads(4)
    rec = json.loads((INV / f"{family}_CLEAN_V2_MANIFEST_INVENTORY.json").read_text(encoding="utf-8"))
    bm = build(family, L0, 2026, "cpu", v1_root=V1_ROOT)
    try:
        got = (bm.init_sha256, bm.raw_init_sha256, bm.manifest_digest, bm.manifest.shared_numel)
    finally:
        del bm
        gc.collect()
    assert got[2] == rec["manifest_digest"], "the L0 manifest digest differs from the MODEL-01 inventory"
    assert got[3] == rec["shared_numel"] == {"GRU": 20_963_806, "SASREC": 83_094_112}[family]
    want, source = registered_init(family, rec)
    record_property("init_hash_source", source)
    if want is None:
        pytest.skip(f"L0 init-hash assertion only (the rest passed): {source}")
    assert got[0] == want["init_sha256"] and got[1] == want["raw_init_sha256"], f"L0 theta0 differs ({source})"


# ------------------------------------------------------------------------------------------------ projection
def _hand_batch(n: int = 3, L: int = 5) -> dict:
    """Every float column holds its own base index (numeric 100 + c, user 200 + c, masks 300 + c; flags c % 2 at c
    odd positions), every token channel a distinct constant: a wrong column is visible by value."""
    w = V2
    am = torch.ones(n, L, dtype=torch.bool)
    b = {"item_tokens": torch.full((n, L), 5), "category_tokens": torch.full((n, L), 6),
         "brand_tokens": torch.full((n, L), 7), "main_category_tokens": torch.full((n, L), 8),
         "daypart_tokens": torch.full((n, L), 4), "weekday_tokens": torch.full((n, L), 9),
         "lengths": torch.full((n,), L), "attention_mask": am, "position_ids": torch.arange(1, L + 1).expand(n, L),
         "event_numeric_features": (100 + torch.arange(w.numeric_dim, dtype=torch.float32)).expand(n, L, -1).clone(),
         "event_quality_flags": (torch.arange(w.flags_dim) % 2).to(torch.int8).expand(n, L, -1).clone(),
         "user_context": (200 + torch.arange(w.user_context_dim, dtype=torch.float32)).expand(n, -1).clone(),
         "user_context_masks": (300 + torch.arange(w.user_mask_dim, dtype=torch.float32)).expand(n, -1).clone(),
         "target_class": torch.zeros(n, dtype=torch.long), "decision_id": torch.arange(n)}
    b["event_quality_flags"][..., 13] = 7                         # item_age_unavailable, a distinct marker
    from v4models.schema import names_sequence
    b.update(names_sequence(w))
    b["feature_view"] = w.view_id
    return b


def test_projection_selects_the_named_columns_of_a_hand_built_batch():
    b = _hand_batch()
    p = L3.projection.apply(b)
    bw, mw = V2, L3.widths
    for key, names, base_names in (("event_numeric_features", mw.numeric_names, bw.numeric_names),
                                   ("event_quality_flags", mw.flag_names, bw.flag_names),
                                   ("user_context", mw.user_context_names, bw.user_context_names),
                                   ("user_context_masks", mw.user_mask_names, bw.user_mask_names)):
        assert p[key].shape[-1] == len(names)
        for j, n in enumerate(names):
            assert torch.equal(p[key][..., j], b[key][..., base_names.index(n)]), f"{key}:{n}"
    assert torch.equal(p["event_numeric_features"][0, 0], torch.tensor([100., 101, 102, 108, 109, 110, 111, 112,
                                                                         113, 114]))
    assert p["event_quality_flags"][0, 0].tolist() == [0, 0, 7]           # price_missing, prev_gap, item_age
    assert torch.equal(p["user_context"][0], torch.tensor([200., 201, 205, 206, 207, 208, 209, 211]))
    assert "main_category_tokens" not in p and "daypart_tokens" not in p
    for t in ("item_tokens", "category_tokens", "brand_tokens", "weekday_tokens"):
        assert p[t] is b[t]
    assert p["event_numeric_feature_names"] == list(mw.numeric_names) and p["feature_view"] == bw.view_id
    assert b["event_numeric_features"].shape[-1] == 15, "the provider's batch must not be modified"
    p5 = L5.projection.apply(b)
    assert "user_context" not in p5 and "user_context_masks" not in p5 and "user_context_names" not in p5
    p9 = L9.projection.apply(b)
    assert set(k for k in p9 if k.endswith("_tokens")) == {"item_tokens"} and "event_numeric_features" not in p9


@pytest.mark.parametrize("family", FAMS)
def test_adapter_checks_the_base_layout_then_projects(family):
    bm = build(family, L3, SEED, "cpu", K=K_SMALL)
    a = bm.adapter
    a.module.eval()
    b = batch(5, seed=3, with_names=True)
    with torch.no_grad():
        y = a.scores(b)
        assert torch.equal(y, module_forward(a, b))
        m = a.module
        from models_testkit import kwargs_for
        assert torch.equal(y, m(**kwargs_for(m, L3.projection.apply(b))))
    small = L3.projection.apply(b)                                  # an already-projected batch: base widths refused
    assert_raises(WidthError, a.query, small)
    bad = dict(b)
    bad["feature_view"] = "v1_rich"
    assert_raises(WidthError, a.query, bad)
    a2 = make_adapter(m, L3)
    assert a2.projection is L3.projection and a2.input_widths == V2
    with torch.no_grad():
        assert torch.equal(a2.scores(b), y)


@pytest.mark.parametrize("family", FAMS)
def test_dropped_channels_cannot_change_the_output(family):
    bm = build(family, L3, SEED, "cpu", K=K_SMALL)
    bm.module.eval()
    b = batch(6, seed=5)
    b2 = dict(b)
    b2["main_category_tokens"] = torch.randint(3, 16, b["main_category_tokens"].shape) * b["attention_mask"]
    nf = b["event_numeric_features"].clone()
    nf[..., 3] += 5.0                                                 # utc_weekday_sin: dropped at L3
    b2["event_numeric_features"] = nf
    uc = b["user_context"].clone()
    uc[:, 3] += 9.0                                                   # has_history: dropped at L3
    b2["user_context"] = uc
    with torch.no_grad():
        assert torch.equal(bm.adapter.scores(b), bm.adapter.scores(b2))
        nf2 = b["event_numeric_features"].clone()
        nf2[..., 0] += 1.0                                            # log1p_price: kept
        b3 = dict(b, event_numeric_features=nf2)
        assert not torch.equal(bm.adapter.scores(b), bm.adapter.scores(b3))


# ------------------------------------------------------------------------------------------------ rungs
@pytest.mark.parametrize("rid", [r["id"] for r in LADDER["rungs"]])
def test_rung_widths_and_parameter_counts_at_full_k(rid):
    r = RUNGS[rid]
    ds = rung_ds(rid)
    assert ds.n_information_channels == r["n_information_channels"]
    iw = r["input_widths"]
    w = ds.widths
    assert (len(w.categorical), w.numeric_dim, w.flags_dim, w.user_context_dim, w.user_mask_dim) == \
        (iw["tokens"], iw["numeric_dim"], iw["flags_dim"], iw["user_context_dim"], iw["user_mask_dim"])
    torch.set_num_threads(4)
    for fam, key in (("GRU", "gru"), ("SASREC", "sasrec")):
        bm = build(fam, ds, SEED, "cpu", K=158_486)
        try:
            m = bm.module
            n = bm.manifest.shared_numel
            assert n == sum(p.numel() for p in m.parameters()) == bm.inventory["total"]
            assert n == r[key]["params_total"], f"{rid}/{fam}: {n} != ladder {r[key]['params_total']}"
            if fam == "GRU":
                assert m.concat_width == r["gru"]["concat_width"]
                nfi = m.numeric_flags_proj[0].in_features if m.numeric_flags_proj is not None else 0
                uci = m.user_ctx_proj[0].in_features if m.user_ctx_proj is not None else 0
                assert (nfi, uci) == (r["gru"]["numeric_flags_proj_in"], r["gru"]["user_ctx_proj_in"])
            else:
                nfi = m.numflag_proj.in_features if m.numflag_proj is not None else 0
                umi = m.user_mlp.in_features if m.user_mlp is not None else 0
                assert (nfi, umi) == (r["sasrec"]["numflag_proj_in"], r["sasrec"]["user_mlp_in"])
                assert m.side_channels == tuple(c for c in w.categorical if c != "item_tokens")
        finally:
            del bm
            gc.collect()


@pytest.mark.parametrize("family", FAMS)
def test_user_branch_absent_at_l5(family):
    bm = build(family, L5, SEED, "cpu", K=K_SMALL)
    m = bm.module
    keys = list(m.state_dict())
    if family == "GRU":
        assert m.user_ctx_proj is None and not m.use_user_context and m.readout_proj.in_features == 256
        assert not any(k.startswith("user_ctx_proj") for k in keys)
    else:
        assert m.user_mlp is None and m.user_gate is None and not m.has_user_branch
        assert not any(k.startswith("user_") for k in keys) and m.side_proj is not None
    m.eval()
    b = batch(6, seed=8)
    b2 = dict(b, user_context=b["user_context"] + 3.0, user_context_masks=1.0 - b["user_context_masks"])
    with torch.no_grad():
        assert torch.equal(bm.adapter.scores(b), bm.adapter.scores(b2))
    if family == "SASREC":                            # the module itself refuses an input its schema lacks
        from models_testkit import kwargs_for
        kw = kwargs_for(m, L5.projection.apply(b))
        kw["user_context"] = b["user_context"][:, :0]
        assert_raises(ValueError, lambda: m(**kw))


def test_l9_is_the_item_only_topology():
    s = build("SASREC", L9, SEED, "cpu", K=K_SMALL).module
    assert not s.has_side_branch and not s.has_user_branch and s.side_channels == ()
    assert not any(k.split(".")[0] in ("side_embeds", "numflag_proj", "side_proj", "side_norm", "side_gate",
                                       "user_mlp", "user_proj", "user_norm", "user_gate") for k in s.state_dict())
    g = build("GRU", L9, SEED, "cpu", K=K_SMALL).module
    assert g.numeric_flags_proj is None and g.user_ctx_proj is None and g.concat_width == 128


# ------------------------------------------------------------------------------------------------ refusals
def test_derived_schema_refusals():
    obj = make_derived(SCHEMA, "X", keep_from_names(SCHEMA, ["log1p_price", "brand_tokens"]))
    assert derived_from_obj(obj, SCHEMA).widths.categorical == ("item_tokens", "brand_tokens")
    bad = copy.deepcopy(obj)
    bad["keep"]["event_numeric_features"].append("utc_hour_sin")
    assert_raises(SchemaError, derived_from_obj, bad, SCHEMA)                       # self-hash
    for mutate in (lambda o: o["keep"]["categorical"].remove("item_tokens"),
                   lambda o: o["keep"]["categorical"].reverse(),
                   lambda o: o["keep"]["event_numeric_features"].append("not_a_channel"),
                   lambda o: o["widths"].update(numeric_dim=2),
                   lambda o: o.update(base=o["base"].replace("2.1.0", "2.1.1")),
                   lambda o: o["keep"].pop("user_context")):
        o = copy.deepcopy(obj)
        mutate(o)
        o["self_sha256"] = self_hash(o)                                              # a consistent hash
        assert_raises(SchemaError, derived_from_obj, o, SCHEMA)
    assert_raises(SchemaError, keep_from_names, SCHEMA, ["item_tokens", "nope"])
    assert_raises(ValueError, __import__("v4models.sasrec_v2", fromlist=["x"]).resolve_side_channels,
                  ("brand_tokens", "category_tokens"))


# ------------------------------------------------------------------------------------------------ negative controls
@nc("a projection that drops item_tokens removes the task's item-ID sequence")
def test_nc_projection_dropping_item_tokens():
    bad = dataclasses.replace(L3.projection, tokens_dropped=L3.projection.tokens_dropped + ("item_tokens",))
    p = bad.apply(batch(3, seed=1))
    assert "item_tokens" in p, "the projected batch lost item_tokens"


@nc("an L0 build whose token channels are in a non-canonical order (init order) is not bitwise the MODEL-01 build")
def test_nc_l0_build_not_bitwise():
    wrong = dataclasses.replace(V2, categorical=("item_tokens", "brand_tokens", "category_tokens",
                                                 "main_category_tokens", "daypart_tokens", "weekday_tokens"))
    a = build("GRU", SCHEMA, SEED, "cpu", K=K_SMALL)
    d = build("GRU", wrong, SEED, "cpu", K=K_SMALL)
    assert d.init_sha256 == a.init_sha256, "the L0 theta0 differs from the MODEL-01 theta0"
