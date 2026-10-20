"""build(family, schema, seed, device) and the adapter registration (fl/adapter.py protocol).

  * deterministic init per seed: the same seed gives bitwise-identical theta0 and init_sha256; another seed differs;
    the caller's global RNG is untouched by a build;
  * SASRec theta0 is the RECENTERED init (Fable G0(b) C7): head common mode ~0, raw and recentered hashes differ,
    init_sha256 = state_digest(broadcast_state) = the digest fl.local_only records as theta0 provenance; GRU theta0 =
    its raw init;
  * the adapters satisfy fl.adapter.ModelAdapter; `ADAPTERS` registers both families; scores == the module's own
    forward (bitwise, eval); head weight / bias are the family's own tensors; post_step / server_post_aggregate
    recenter SASRec and are no-ops for the GRU; manifest roles as in FL-01 (no alias keys, int map FIXED);
  * the V2 families run through the FL core unchanged: one FA round (server + 2 clients) and one LO run per family.
"""
from __future__ import annotations

import pytest
import torch

import v4models
from fl.adapter import BufferRule, ModelAdapter, Role
from fl.client import ClientData, LocalSolver
from fl.local_only import LORecipe, run_local_only
from fl.numerics import state_digest, states_equal
from fl.server import Server, run_round
from models_testkit import FAMS, K_SMALL, SCHEMA, V2, batch, nc
from v4models.build import ADAPTERS, GRUV2Adapter, SASRecV2Adapter, build, module_forward
from v4models.monitors import common_mode_report

SEED = 2026


@pytest.mark.parametrize("family", FAMS)
def test_build_is_deterministic_per_seed(family):
    before = torch.get_rng_state()
    a, b = build(family, SCHEMA, SEED, "cpu", K=K_SMALL), build(family, SCHEMA, SEED, "cpu", K=K_SMALL)
    assert torch.equal(before, torch.get_rng_state()), "build must not advance the caller's global RNG"
    c = build(family, SCHEMA, SEED + 1, "cpu", K=K_SMALL)
    assert a.init_sha256 == b.init_sha256 and states_equal(a.theta0(), b.theta0())
    assert a.init_sha256 != c.init_sha256
    torch.manual_seed(12345)                                            # a different caller RNG state
    d = build(family, SCHEMA, SEED, "cpu", K=K_SMALL)
    assert d.init_sha256 == a.init_sha256, "the init must depend on the seed only"
    assert a.init_sha256 == state_digest(a.adapter.broadcast_state())
    assert a.widths == V2 and a.K == K_SMALL and a.family == family


def test_sasrec_theta0_is_recentered_and_gru_is_raw():
    s = build("SASREC", SCHEMA, SEED, K=K_SMALL)
    assert s.recentered and s.raw_init_sha256 != s.init_sha256
    head = common_mode_report(s.adapter)["head"]
    assert head["common_mode_norm"] < 1e-6 and abs(head["b_mean"]) < 1e-7
    g = build("GRU", SCHEMA, SEED, K=K_SMALL)
    assert not g.recentered and g.raw_init_sha256 == g.init_sha256
    rec = s.record()
    assert rec["recentered_theta0"] and rec["widths"]["user_mask_dim"] == 4 and rec["shared_numel"] > 0


@pytest.mark.parametrize("family", FAMS)
def test_adapter_protocol_and_registration(family):
    bm = build(family, SCHEMA, SEED, K=K_SMALL)
    a = bm.adapter
    assert isinstance(a, ModelAdapter) and type(a) is ADAPTERS[family]
    assert set(ADAPTERS) == set(v4models.FAMILIES) == {"GRU", "SASREC"}
    a.module.eval()
    b = batch(6, seed=2)
    with torch.no_grad():
        assert torch.equal(a.scores(b), module_forward(a, b)), "adapter scores must equal the module's own forward"
        q = a.query(b)
    assert q.shape == (6, a.query_dim) and a.query_dim == {"GRU": 128, "SASREC": 256}[family]
    assert a.head_bias() is a.module.output_bias and a.head_weight().shape == (K_SMALL, a.query_dim)
    m = bm.manifest
    assert m.alias_keys == () and m.shared_numel == sum(p.numel() for p in a.module.parameters())
    assert all(m.entries[k].dtype == "torch.float32" for k in m.shared_keys)
    if family == "GRU":
        assert isinstance(a, GRUV2Adapter)
        assert m.entries["product_idx_of_class"].role == Role.BUFFER
        assert m.entries["product_idx_of_class"].rule == BufferRule.FIXED
    else:
        assert isinstance(a, SASRecV2Adapter)
        assert set(m.nonpersistent_buffers) == {"product_idx_of_class", "causal"}


@pytest.mark.parametrize("family", FAMS)
def test_post_step_hooks(family):
    a = build(family, SCHEMA, SEED, K=K_SMALL).adapter
    with torch.no_grad():
        a.head_bias().add_(1.0)
    before = a.broadcast_state(clone=True)
    a.post_step()
    if family == "GRU":
        assert states_equal(before, a.broadcast_state()), "GRU post_step must be a no-op"
    else:
        assert abs(float(a.head_bias().mean())) < 1e-6
        a.head_bias().data.add_(1.0)
        a.server_post_aggregate()
        assert abs(float(a.head_bias().mean())) < 1e-6
    assert a.has_post_step == (family == "SASREC")


@pytest.mark.parametrize("family", FAMS)
def test_v2_families_run_through_the_fl_core(family):
    srv = Server(build(family, SCHEMA, SEED, K=K_SMALL).adapter)
    workers = [build(family, SCHEMA, SEED + 1, K=K_SMALL).adapter]
    cs = [ClientData("user-a", batch(5, seed=3)), ClientData("user-b", batch(20, seed=4))]
    sol = LocalSolver(lr=1e-3, passes=2, batch_size=16, clip=1.0, weight_decay=1e-5)
    rep = run_round(srv, workers, cs, sol, round_idx=0, seed=SEED)
    assert rep.weight == 2 * 25 and [v.steps for v in rep.visits] == [2, 4]
    assert all(torch.isfinite(t).all() for t in srv.adapter.broadcast_state().values() if t.is_floating_point())
    bm = build(family, SCHEMA, SEED, K=K_SMALL)
    theta0 = bm.theta0()
    recipe = LORecipe(bm.init_sha256, SEED, passes=2, eval_at=(1, 2), solver=sol)
    lo = run_local_only(workers[0], theta0, cs[1], recipe)
    assert lo.steps == 4 and lo.n_consumed == 40


def test_recentering_idempotence_is_measured():
    """C7 + FL server init recentering: recentering an already recentered theta0 again is a no-op up to FP32 rounding
    of the row mean; recorded here so the lock can state whether the server re-recenters a registered theta0."""
    bm = build("SASREC", SCHEMA, SEED, K=K_SMALL)
    s0 = bm.theta0()
    bm.adapter.server_post_aggregate()
    s1 = bm.adapter.broadcast_state()
    d = max(float((s0[k] - s1[k]).abs().max()) for k in ("output_embed", "output_bias"))
    assert d < 1e-8


@nc("a build that forgets the C7 recentering leaves a visible common mode in theta0")
def test_nc_unrecentered_theta0():
    from v4models.build import construct_module
    from models_testkit import poc, vocab
    m = construct_module("SASREC", V2, SEED, vocab_sizes=vocab(K_SMALL), poc=poc(K_SMALL))
    a = ADAPTERS["SASREC"](m, V2)
    assert common_mode_report(a)["head"]["common_mode_norm"] < 1e-6
