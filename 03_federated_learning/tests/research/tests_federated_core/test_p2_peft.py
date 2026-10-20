"""P2-04 — fl.peft, LoRA on qkv / out / ff1 / ff2 of both CausalBlocks of SASREC_D64_B2 (PREG2 §5.4), CPU, small K.

Proves: the TO-CHECK (CausalBlock's forward paths call the Linear modules and never read `.weight`; LoRALinear.weight is
the effective weight anyway); B = 0 -> scores identical to theta_pre and the merged state hashes to theta_pre's digest
(lora_init_sha256; refused when B != 0); the merged base-architecture forward equals the wrapped forward to 1e-6, with
the base model's exact key list; after freeze_all_but only the LoRA keys are shared (extract_shared, requires_grad,
optimizer groups), every other parameter is a FIXED buffer and the recentering hooks are off; a visit moves only A / B;
bytes per visit = 4 x 2,048 r (+ the 8-byte header up), the frozen base is reported once per client; the A init is
seeded and device-independent; an FLRun resumes A / B bitwise and finishes on the straight run's digest; best / endpoint
payloads carry the merged base theta + p2_lora, from which A / B are restored bitwise. Refusals: double application,
another trainable set, combining with freeze_item_tables, a structure mismatch on load.
Negative control: a worker whose frozen base differs is refused at the visit (FIXED buffer check).
"""
from __future__ import annotations

import inspect
from collections import OrderedDict

import pytest
import torch

import fl.peft as P
from fl.adapter import BufferRule, ManifestError, Role
from fl.checkpoint import CheckpointManager, load_checkpoint
from fl.client import ClientData, client_update, make_optimizer, valid_rows
from fl.comm import UPLOAD_HEADER_BYTES, VirtualChannel, visit_bytes
from fl.numerics import state_digest
from fl.participation import ParticipationPlan
from fl.runtime import FLRun, RunConfig, freeze_item_tables
from fl.server import Server, run_round
from fl_testkit import assert_raises, nc, solver

SEED = 2026
KC = 64
D, FF, NB = 64, 4, 2


def _built(seed=SEED):
    from v4models import build, load_schema
    return build("SASREC", load_schema(), seed, K=KC, variant="SASREC_D64_B2")


def _examples(n, seed):
    from fl.families import synthetic_examples_v2
    return synthetic_examples_v2(n, KC, torch.Generator().manual_seed(seed))


def _lora(r=4, seed=SEED, lora_seed=7, source=None):
    bm = _built(seed)
    P.apply_lora(bm.adapter.module, r=r, alpha=float(r), seed=lora_seed)
    P.freeze_all_but(bm.adapter, P.lora_keys(bm.adapter.module), source=source)
    return bm


def _randomize_lora(module, seed=11, scale=0.05):
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for k, p in OrderedDict(module.named_parameters()).items():
            if k.endswith(P.LORA_B) or k.endswith(P.LORA_A):
                p.copy_(torch.randn(p.shape, generator=g) * scale)


def _scores(adapter, ex):
    adapter.module.eval()
    with torch.no_grad():
        return adapter.scores(ex)


# ------------------------------------------------------------------------------------------------ the TO-CHECK
def test_to_check_causal_block_never_reads_linear_weight_and_property_is_effective():
    from v4models.sasrec_v2 import CausalBlock
    for fn in (CausalBlock.forward_tokens, CausalBlock.forward_padded, CausalBlock._attend, CausalBlock._ffn):
        src = inspect.getsource(fn)
        assert ".weight" not in src and ".bias" not in src, fn.__name__
    bm = _lora(r=4)
    _randomize_lora(bm.adapter.module)
    lin = bm.adapter.module.blocks[0].qkv
    assert isinstance(lin, P.LoRALinear) and (lin.in_features, lin.out_features) == (D, 3 * D)
    eff = lin.base_weight + lin.lora_B @ lin.lora_A * lin.scale
    assert torch.equal(lin.weight, eff) and lin.bias is lin.base_bias
    x = torch.randn(5, D, generator=torch.Generator().manual_seed(1))
    assert torch.equal(lin(x), torch.nn.functional.linear(x, eff, lin.base_bias))


# ------------------------------------------------------------------------------------------------ B = 0
def test_b_zero_scores_identical_to_theta_pre_and_init_hash():
    base = _built()
    ex = _examples(12, 3)
    ref = _scores(base.adapter, ex)
    bm = _lora(r=16)
    assert torch.equal(_scores(bm.adapter, ex), ref)
    assert state_digest(P.merged_state(bm.adapter.module)) == base.init_sha256
    h = P.lora_init_sha256(bm.adapter, base.init_sha256)
    assert h == state_digest(bm.adapter.broadcast_state()) != base.init_sha256
    assert_raises(P.PeftError, P.lora_init_sha256, bm.adapter, "0" * 64)
    _randomize_lora(bm.adapter.module)
    assert_raises(P.PeftError, P.lora_init_sha256, bm.adapter, base.init_sha256)


def test_lora_a_init_is_seeded_and_b_is_zero():
    a1, a2, a3 = _lora(lora_seed=7), _lora(lora_seed=7), _lora(lora_seed=8)
    s1, s2, s3 = (P.lora_state(b.adapter.module) for b in (a1, a2, a3))
    assert all(torch.equal(s1[k], s2[k]) for k in s1)
    assert any(not torch.equal(s1[k], s3[k]) for k in s1 if k.endswith(P.LORA_A))
    assert all(not s1[k].any() for k in s1 if k.endswith(P.LORA_B))
    A = s1["blocks.0.qkv.lora_A"]
    assert A.shape == (4, D) and float(A.abs().max()) <= 1.0 / D ** 0.5 + 1e-7   # kaiming_uniform(a=sqrt 5) bound
    meta = a1.adapter.module.fl_lora
    assert meta["r"] == 4 and meta["scale"] == 1.0 and meta["n_blocks"] == NB and meta["targets"] == list(P.LORA_TARGETS)


def test_alpha_over_r_scaling_of_effective_and_merged_weights():
    """REV-P2-03_04 m1: with alpha != r, W_eff and the merged Delta W are B A alpha / r numerically (a scale of
    alpha * r or r / alpha fails), and the merged forward still equals the wrapped forward."""
    for r, alpha, want in ((4, 8.0, 2.0), (16, 4.0, 0.25)):
        bm = _built()
        P.apply_lora(bm.adapter.module, r=r, alpha=alpha, seed=3)
        P.freeze_all_but(bm.adapter, P.lora_keys(bm.adapter.module))
        _randomize_lora(bm.adapter.module, seed=r)
        merged = P.merged_state(bm.adapter.module)
        for i in range(NB):
            for t in P.LORA_TARGETS:
                lin = getattr(bm.adapter.module.blocks[i], t)
                assert lin.scale == want == alpha / r
                ba = (lin.lora_B.detach().double() @ lin.lora_A.detach().double()) * (alpha / r)   # independent, FP64
                d_eff = lin.weight.detach().double() - lin.base_weight.detach().double()
                d_mrg = merged[f"blocks.{i}.{t}.weight"].double() - lin.base_weight.detach().double()
                for d in (d_eff, d_mrg):
                    assert torch.allclose(d, ba, rtol=1e-5, atol=1e-7), (r, alpha, i, t, float((d - ba).abs().max()))
                assert float(ba.abs().max()) > 1e-4                         # a non-trivial update was checked
        ex = _examples(12, 9)
        fresh = _built(seed=31)
        fresh.adapter.load_state_(merged)
        assert float((_scores(fresh.adapter, ex) - _scores(bm.adapter, ex)).abs().max()) <= 1e-6


# ------------------------------------------------------------------------------------------------ merge
def test_merged_forward_equals_wrapped_forward_and_base_keys():
    for r in (4, 16):
        bm = _lora(r=r)
        _randomize_lora(bm.adapter.module, seed=r)
        ex = _examples(16, 5)
        wrapped = _scores(bm.adapter, ex)
        merged = P.merged_state(bm.adapter.module)
        fresh = _built(seed=999)                                    # another init, then the merged weights loaded
        assert list(merged) == list(fresh.adapter.broadcast_state()) # the base SASREC_D64_B2 key list and order
        fresh.adapter.load_state_(merged)
        diff = float((_scores(fresh.adapter, ex) - wrapped).abs().max())
        assert diff <= 1e-6, (r, diff)
        assert not torch.equal(merged["blocks.1.ff2.weight"], bm.adapter.module.blocks[1].ff2.base_weight)


# ------------------------------------------------------------------------------------------------ freeze
def test_only_lora_keys_are_shared_and_the_rest_is_fixed():
    for r in (4, 16):
        bm = _lora(r=r)
        a, m = bm.adapter, bm.adapter.manifest
        keys = P.lora_keys(a.module)
        assert len(keys) == 2 * 4 * NB and all(k.endswith((P.LORA_A, P.LORA_B)) for k in keys)
        assert tuple(m.shared_keys) == keys and list(a.extract_shared()) == list(keys)
        assert m.shared_numel == P.lora_numel_formula(D, FF, NB, r) == 2048 * r
        named = dict(a.module.named_parameters())
        assert all(p.requires_grad == (k in keys) for k, p in named.items())
        opt_ids = {id(p) for g in make_optimizer(a, solver(optimizer="sgd_m")).param_groups for p in g["params"]}
        assert opt_ids == {id(named[k]) for k in keys}
        fz = a.fl_frozen_keys
        assert set(fz) == set(named) - set(keys) and "output_embed" in fz and "item_embed.weight" in fz
        assert all(m.entries[k].role == Role.BUFFER and m.entries[k].rule == BufferRule.FIXED for k in fz)
        before = state_digest(a.broadcast_state())
        a.post_step()
        a.server_post_aggregate()
        assert state_digest(a.broadcast_state()) == before           # recentering off for the frozen output layer
        assert P.freeze_all_but(a, keys)["manifest_digest"] == m.digest()   # idempotent


def test_refusals():
    bm = _lora()
    assert_raises(P.PeftError, P.apply_lora, bm.adapter.module, r=4, alpha=4.0, seed=1)      # twice
    assert_raises(P.PeftError, P.freeze_all_but, bm.adapter, P.lora_keys(bm.adapter.module)[:2])
    b2 = _built()
    freeze_item_tables(b2.adapter)
    P.apply_lora(b2.adapter.module, r=4, alpha=4.0, seed=1)
    assert_raises(P.PeftError, P.freeze_all_but, b2.adapter, P.lora_keys(b2.adapter.module))
    b3 = _built()
    assert_raises(P.PeftError, P.apply_lora, b3.adapter.module, ("qkv", "nope"), r=4, alpha=4.0, seed=1)
    assert_raises(P.PeftError, P.apply_lora, b3.adapter.module, r=0, alpha=1.0, seed=1)
    other = _lora(r=16)
    payload = P.lora_weights_payload({"theta": {}}, other.adapter)
    assert_raises(P.PeftError, P.load_lora_from_payload, bm.adapter, payload)               # r 16 into r 4


# ------------------------------------------------------------------------------------------------ visits / bytes
def _client(key, n, seed):
    return ClientData(key, _examples(n, seed))


def test_a_visit_moves_only_a_and_b():
    bm = _lora()
    theta = bm.adapter.broadcast_state(clone=True)
    w = _lora(seed=SEED, source=theta)
    res = client_update(w.adapter, theta, _client("u0", 9, 21), solver(optimizer="sgd_m", lr=0.05, passes=1),
                        round_idx=0, seed=SEED, clone_upload=True)
    assert list(res.upload) == list(P.lora_keys(w.adapter.module))
    assert any(not torch.equal(res.upload[k], theta[k]) for k in res.upload if k.endswith(P.LORA_B))
    after = w.adapter.broadcast_state()
    for k in w.adapter.fl_frozen_keys:
        assert torch.equal(after[k], theta[k]), k


def test_bytes_per_visit_lora_only():
    for r in (4, 16):
        bm = _lora(r=r)
        init = P.lora_init_sha256(bm.adapter, _built().init_sha256)
        server = Server(bm.adapter, init_sha256=init)
        w = _lora(r=r, source=server.adapter.broadcast_state())
        cl = [_client(f"u{i}", 5 + i, 30 + i) for i in range(3)]
        ch = VirtualChannel(server.manifest)
        rep = run_round(server, [w.adapter], cl, solver(optimizer="sgd_m", lr=0.05, passes=1), round_idx=0, seed=SEED,
                        n_shards=1, channel=ch)
        per = 4 * 2048 * r
        assert server.manifest.shared_bytes == per
        assert rep.bytes_down == 3 * per and rep.bytes_up == 3 * (per + UPLOAD_HEADER_BYTES)
        vb = visit_bytes(server.manifest, frozen_keys=bm.adapter.fl_frozen_keys)
        assert vb["upload_bytes"] == per + 8 and vb["uploaded_numel"] == 2048 * r
        base_bytes = _built().adapter.manifest.shared_bytes
        assert vb["frozen_bytes_once"] == base_bytes                    # the frozen base, downloaded once


# ------------------------------------------------------------------------------------------------ FLRun / checkpoints
CLIENTS = {f"user-{i:03d}": _client(f"user-{i:03d}", 3 + (i % 5), 100 + i) for i in range(8)}
N_DEC = sum(int(valid_rows(c.examples).numel()) for c in CLIENTS.values())


def _cfg():
    return RunConfig(run_id="p2-04-lora", method="FA", seed=SEED, lr_peak_local=0.05,
                     solver=solver(optimizer="sgd_m", lr=0.05, passes=1), n_shards=1, exposure_point="end",
                     dropout_p=0.1, endpoint_rounds=6, ckpt_every=2)


def _flrun(tmp, resume=False, name="ck"):
    bm = _lora()
    init = P.lora_init_sha256(bm.adapter, _built().init_sha256)
    server = Server(bm.adapter, init_sha256=init)
    w = _lora(source=server.adapter.broadcast_state())
    plan = ParticipationPlan.build(sorted(CLIENTS), seed=SEED, manifest_hash="p2-04", group_size=3, sampling="sweep")
    ck = CheckpointManager(tmp / name, "p2-04-lora") if tmp is not None else None
    run = (FLRun.resume if resume else FLRun)(_cfg(), plan, server, CLIENTS.__getitem__, N_DEC,
                                               workers=[w.adapter], ckpt=ck)
    rec = P.install_peft(run)
    assert rec["shared_numel"] == 2048 * 4
    return run


def test_flrun_resume_restores_a_b_bitwise_and_payloads(tmp_path):
    straight = _flrun(None)
    straight.run()
    part = _flrun(tmp_path)
    for _ in range(3):
        part.step()
    part.ckpt.save_latest(part.state_dict())
    saved = P.lora_state(part.server.adapter.module)
    res = _flrun(tmp_path, resume=True)
    assert res.state.cursor == 3
    got = P.lora_state(res.server.adapter.module)
    assert list(got) == list(saved) and all(torch.equal(got[k], saved[k]) for k in saved)   # A / B bitwise
    res.run()
    assert res.state.done and res.server.digest() == straight.server.digest()
    ep = load_checkpoint(res.ckpt.dir / res.ckpt.ENDPOINT)
    base_keys = list(_built().adapter.broadcast_state())
    assert sorted(ep["theta"]) == sorted(base_keys)                 # the merged base-architecture theta
    rec = ep["p2_lora"]
    assert rec["format"] == P.P2_LORA_FORMAT and rec["meta"]["r"] == 4
    final = P.lora_state(res.server.adapter.module)
    assert all(torch.equal(rec["state"][k], final[k]) for k in final)
    assert rec["merged_theta_digest"] == state_digest(ep["theta"])
    assert rec["lora_theta_digest"] == res.server.digest() == state_digest(res.server.adapter.broadcast_state())
    assert rec["lineage"] == {"theta_pre_digest": _built().init_sha256, "lora_init_sha256": res.server.init_sha256}
    fresh = _lora()                                                 # a later leg rebuilds theta_pre + LoRA ...
    P.load_lora_from_payload(fresh.adapter, ep)                     # ... and restores A / B from the payload
    back = P.lora_state(fresh.adapter.module)
    assert all(torch.equal(back[k], final[k]) for k in final)
    ex = _examples(10, 77)
    b = _built(seed=5)
    b.adapter.load_state_(ep["theta"])
    assert float((_scores(b.adapter, ex) - _scores(res.server.adapter, ex)).abs().max()) <= 1e-6
    assert P.install_peft(res)["shared_numel"] == 2048 * 4         # idempotent (no double wrapping)


# ------------------------------------------------------------------------------------------------ negative control
@nc("a worker whose frozen base differs from the server's is refused at the visit (FIXED buffers verified)")
def test_nc_worker_with_another_frozen_base_is_accepted():
    bm = _lora()
    theta = bm.adapter.broadcast_state(clone=True)
    w = _lora(seed=SEED + 1)                                        # another base init, no bootstrap copy
    try:
        client_update(w.adapter, theta, _client("u0", 6, 3), solver(optimizer="sgd_m", lr=0.05, passes=1),
                      round_idx=0, seed=SEED)
    except ManifestError:
        raise AssertionError("refused (as intended)")
