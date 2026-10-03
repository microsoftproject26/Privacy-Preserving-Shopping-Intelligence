"""sparse_rows.py, the row-sparse item upload codec, on the synthetic FL fixtures (CPU).

Proves: touched_rows = the history tokens of the valid decisions + their target tokens (class + 3), sorted / unique;
the round trip is exact on every sent row and untouched rows decode to theta_r; the output layer's joint top-k rows are
the k largest delta norms (stable ties); payload bytes = the formula (dense FP32 + per group 4 + 4 n +
4 n width) and the ledger adds the 8-byte n_consumed header per visit; the server-side decode reads only the payload and
theta_r (a poisoned upload is never seen, through decode and through the channel); with SGD_M and k = K the sparse run
equals the dense FA run bitwise (round-level and FLRun-level); a sparse round equals the FedAvg of the decoded uploads
bitwise; an FLRun with the channel resumes exactly after re-installing it; pools, non-FA configs and frozen item tables
are refused. Negative control: k < K equals dense.
"""
from __future__ import annotations

import math
from collections import OrderedDict

import torch
from fedsim_testkit import K, assert_raises, clients, nc, solver, tiny

import ppsi.fedsim.sparse_rows as SR
from ppsi.fedsim.aggregate import aggregate_uploads
from ppsi.fedsim.checkpoint import CheckpointManager
from ppsi.fedsim.client import client_update, valid_rows
from ppsi.fedsim.comm import UPLOAD_HEADER_BYTES
from ppsi.fedsim.numerics import state_digest
from ppsi.fedsim.participation import ParticipationPlan
from ppsi.fedsim.runtime import FROZEN_ITEM_KEYS, FLRun, RunConfig, freeze_item_tables
from ppsi.fedsim.server import Server, run_round

SEED = 2026
CS = clients(12, seed=707, invalid_frac=0.1)
BY_KEY = {c.key: c for c in CS}
N_DEC = sum(int((c.examples["target_class"] >= 0).sum()) for c in CS)
SGDM = {"optimizer": "sgd_m", "lr": 0.05}


def _untied(seed=1):
    return tiny(seed, tied=False)


def _registered_server():
    a = _untied(1)
    a.server_post_aggregate()
    th, init = a.broadcast_state(clone=True), state_digest(a.broadcast_state())
    return Server(a, init_sha256=init), th, init


def _perturbed_upload(theta, keys, seed=0, scale=1e-2):
    g = torch.Generator().manual_seed(seed)
    return OrderedDict((k, theta[k] + torch.randn(theta[k].shape, generator=g) * scale) for k in keys)


# ------------------------------------------------------------------------------------------------ touched rows
def test_touched_rows_history_and_targets_of_valid_rows_only():
    ex = {"item_tokens": torch.tensor([[5, 7, 0], [9, 9, 4], [11, 12, 13]]),
          "attention_mask": torch.tensor([[1, 1, 0], [1, 1, 1], [1, 1, 1]], dtype=torch.bool),
          "lengths": torch.tensor([2, 3, 3]),
          "target_class": torch.tensor([2, 0, -1])}                  # row 2 is invalid: none of its tokens count
    got = SR.touched_rows(ex)
    assert got.dtype == torch.long and got.tolist() == [3, 4, 5, 7, 9]   # history 5 7 9 4 + targets 2+3, 0+3
    ex["loss_mask"] = torch.tensor([1, 0, 1], dtype=torch.bool)          # loss_mask removes row 1 as well
    assert SR.touched_rows(ex).tolist() == [5, 7]
    ex["target_class"] = torch.tensor([-1, -1, -1])
    assert SR.touched_rows(ex).numel() == 0


# ------------------------------------------------------------------------------------------------ the codec
def test_round_trip_exact_on_sent_rows_and_theta_elsewhere():
    a = _untied(1)
    theta = a.broadcast_state(clone=True)
    keys = a.manifest.shared_keys
    up = _perturbed_upload(theta, keys, 1)
    rows = torch.tensor([0, 3, 4, 10, K + 2])
    k_top = 5
    pay = SR.encode(up, theta, keys, touched_rows=rows, k_top=k_top)
    dec = SR.decode(pay, theta)
    assert list(dec) == list(keys)
    gi, go = pay.group_of(SR.ITEM_KEY), pay.group_of("output_embed")
    assert gi.ids.dtype == torch.int32 and gi.ids.tolist() == rows.tolist() and go.n == k_top
    assert pay.group_of("output_bias") is go                                  # one joint row set
    for k, g in ((SR.ITEM_KEY, gi), ("output_embed", go), ("output_bias", go)):
        sent = g.ids.long()
        unsent = torch.ones(theta[k].shape[0], dtype=torch.bool)
        unsent[sent] = False
        assert torch.equal(dec[k][sent], up[k][sent]), k                    # exact on the sent rows
        assert torch.equal(dec[k][unsent], theta[k][unsent]), k             # zero delta elsewhere
    for k in keys:
        if k not in (SR.ITEM_KEY,) + SR.OUTPUT_KEYS:
            assert torch.equal(dec[k], up[k]), k                            # dense keys unchanged


def test_output_top_k_is_the_largest_joint_delta_norms_with_stable_ties():
    a = _untied(1)
    theta = a.broadcast_state(clone=True)
    up = OrderedDict((k, v.clone()) for k, v in theta.items() if k in a.manifest.shared_keys)
    mags = {3: 5.0, 7: 4.0, 1: 3.0, 9: 3.0, 20: 3.0}                          # rows 1 / 9 / 20 tie
    for j, m in mags.items():
        up["output_embed"][j] += m / math.sqrt(2.0)
        up["output_bias"][j] += m / math.sqrt(2.0)                            # joint norm sqrt(W^2 + b^2) = m-ish
    ids = SR.top_k_rows(up, theta, SR.OUTPUT_KEYS, 4)
    assert ids.tolist() == [1, 3, 7, 9]                                       # 3, 7, then the tie -> lower ids 1, 9
    assert SR.top_k_rows(up, theta, SR.OUTPUT_KEYS, K + 5).tolist() == list(range(K))
    up2 = OrderedDict(up)
    up2["output_bias"] = theta["output_bias"].clone()
    up2["output_bias"][15] += 100.0                                           # the bias alone decides row 15
    assert 15 in SR.top_k_rows(up2, theta, SR.OUTPUT_KEYS, 1).tolist()
    assert_raises(SR.SparseError, SR.top_k_rows, up, theta, SR.OUTPUT_KEYS, 0)


def test_top_k_tie_break_lower_id_first_deterministic_large_tie_blocks():
    """4,000 rows with only four distinct (exact) joint norms, k cutting inside a tie block; the
    result must equal an explicit lexicographic (-norm, id) reference (an unstable sort does not, MEASURED on CPU)."""
    n, k = 4000, 1500
    g = torch.Generator().manual_seed(0)
    mags = torch.tensor([2.0, 1.5, 1.0, 0.5])[torch.randint(0, 4, (n,), generator=g)]
    theta = {"output_embed": torch.zeros(n, 3), "output_bias": torch.zeros(n)}
    up = {"output_embed": torch.zeros(n, 3), "output_bias": torch.zeros(n)}
    up["output_embed"][:, 0] = mags                                           # joint norm^2 = mag^2 exactly
    nsq = [float(m) ** 2 for m in mags]
    ref = sorted(sorted(range(n), key=lambda j: (-nsq[j], j))[:k])
    boundary = nsq[sorted(range(n), key=lambda j: (-nsq[j], j))[k - 1]]
    tied = [j for j in range(n) if nsq[j] == boundary]
    assert any(j in ref for j in tied) and any(j not in ref for j in tied)   # k cuts inside a tie block
    runs = [SR.top_k_rows(up, theta, SR.OUTPUT_KEYS, k).tolist() for _ in range(3)]
    assert runs[0] == ref and runs[1] == runs[0] and runs[2] == runs[0]
    inside = sorted(j for j in tied if j in ref)
    assert inside == sorted(tied)[:len(inside)]                              # the LOWEST ids of the tie block


def test_bytes_formula_and_hand_count():
    a = _untied(1)
    m = a.manifest
    theta = a.broadcast_state(clone=True)
    up = _perturbed_upload(theta, m.shared_keys, 2)
    d = a.module.d
    for rows, k_top in ((torch.tensor([3, 4, 8]), 5), (torch.zeros(0, dtype=torch.long), 1), (torch.arange(K + 3), K)):
        pay = SR.encode(up, theta, m.shared_keys, touched_rows=rows, k_top=k_top)
        n_item, n_out = int(rows.numel()), min(k_top, K)
        dense = sum(m.entries[k].numel for k in m.shared_keys if k not in (SR.ITEM_KEY,) + SR.OUTPUT_KEYS) * 4
        hand = dense + (4 + 4 * n_item + 4 * n_item * d) + (4 + 4 * n_out + 4 * n_out * (d + 1))
        assert SR.payload_bytes_sparse(pay) == SR.sparse_payload_bytes(m, n_item, n_out) == hand
        assert SR.sparse_upload_bytes(m, n_item, n_out) == hand + UPLOAD_HEADER_BYTES
        vb = SR.sparse_visit_bytes(m, k_top, n_item)
        assert vb["upload_bytes"] == hand + 8 and vb["download_bytes"] == m.shared_bytes
    # the output-layer group at the study setting (d 64, k 3,170): 4 + 4 k + 4 k (d + 1) bytes
    assert 4 + 4 * 3170 + 4 * 3170 * 65 == 836_884


def test_decode_never_reads_the_upload():
    a = _untied(1)
    theta = a.broadcast_state(clone=True)
    keys = a.manifest.shared_keys
    up = _perturbed_upload(theta, keys, 3)
    pay = SR.encode(up, theta, keys, touched_rows=torch.tensor([4, 5]), k_top=3)
    want = SR.decode(pay, theta)
    want = OrderedDict((k, v.clone()) for k, v in want.items())
    for k in (SR.ITEM_KEY,) + SR.OUTPUT_KEYS:
        up[k].fill_(float("nan"))                                             # poison the client's dense upload
    got = SR.decode(pay, theta)
    for k in (SR.ITEM_KEY,) + SR.OUTPUT_KEYS:
        assert torch.isfinite(got[k]).all() and torch.equal(got[k], want[k]), k


def test_channel_never_lets_the_server_see_an_unsent_row():
    server, th, _ = _registered_server()
    touched = torch.tensor([3, 5, 6])
    ch = SR.SparseChannel(server.manifest, k_top=4, theta_fn=lambda: server.adapter.broadcast_state(),
                          touched_fn=lambda _k: touched)
    ch.download(0, "u")
    up = OrderedDict((k, v.clone()) for k, v in server.adapter.extract_shared(clone=True).items())
    poison = torch.ones(up[SR.ITEM_KEY].shape[0], dtype=torch.bool)
    poison[touched] = False
    up[SR.ITEM_KEY][poison] = float("nan")                                     # unsent rows carry garbage
    up[SR.ITEM_KEY][touched] += 0.5
    n = ch.upload(0, "u", up)
    assert torch.isfinite(up[SR.ITEM_KEY]).all()
    assert torch.equal(up[SR.ITEM_KEY][poison], th[SR.ITEM_KEY][poison])
    assert torch.equal(up[SR.ITEM_KEY][touched], th[SR.ITEM_KEY][touched] + 0.5)
    assert n == SR.sparse_upload_bytes(server.manifest, 3, 4) and ch.local.snapshot()[2] == n
    ch.check_round(0, ["u"])
    assert_raises(SR.SparseError, ch.check_round, 0, ["u", "v"])
    assert_raises(SR.SparseError, ch.upload_counted, 0, "u")
    assert_raises(SR.SparseError, ch.upload, 1, "u", dict(up))                 # no round-1 download


def test_encode_refusals():
    a = _untied(1)
    theta = a.broadcast_state(clone=True)
    keys = a.manifest.shared_keys
    up = _perturbed_upload(theta, keys, 4)
    assert_raises(SR.SparseError, SR.encode, up, theta, keys, touched_rows=torch.tensor([K + 3]), k_top=2)
    assert_raises(SR.SparseError, SR.encode, up, theta, keys, touched_rows=torch.tensor([5, 4]), k_top=2)
    half = tuple(k for k in keys if k != "output_bias")
    assert_raises(SR.SparseError, SR.encode, up, theta, half, touched_rows=torch.tensor([4]), k_top=2)
    up64 = OrderedDict(up)
    up64[SR.ITEM_KEY] = up[SR.ITEM_KEY].double()
    assert_raises(SR.SparseError, SR.encode, up64, theta, keys, touched_rows=torch.tensor([4]), k_top=2)


# ------------------------------------------------------------------------------------------------ one round
def _round(k_top, sol, channel=True, rnd=7):
    server, th, init = _registered_server()
    workers = [_untied(101)]
    cl = CS[:5]
    ch = None
    if channel:
        ch = SR.SparseChannel(server.manifest, k_top=k_top, theta_fn=lambda: server.adapter.broadcast_state(),
                              touched_fn=lambda key: SR.touched_rows(BY_KEY[key].examples))
    rep = run_round(server, workers, cl, sol, round_idx=rnd, seed=SEED, n_shards=2, channel=ch)
    if ch is not None:
        ch.check_round(rnd, [c.key for c in cl])
    return server, th, init, rep, ch, cl


def test_sparse_round_equals_fedavg_of_decoded_uploads_bitwise():
    sol = solver(**SGDM)
    server, th, init, rep, ch, cl = _round(5, sol)
    ref_w = _untied(101)
    uploads, n_up = [], 0
    for c in cl:
        res = client_update(ref_w, th, c, sol, round_idx=7, seed=SEED, clone_upload=True)
        rows = SR.touched_rows(c.examples)
        pay = SR.encode(res.upload, th, server.manifest.shared_keys, touched_rows=rows, k_top=5)
        uploads.append((c.key, SR.decode(pay, th), res.n_consumed))
        n_up += SR.payload_bytes_sparse(pay) + UPLOAD_HEADER_BYTES
    a2 = _untied(1)
    a2.load_state_(th)
    s2 = Server(a2, init_sha256=init)
    s2.apply(aggregate_uploads(s2.manifest, th, uploads, n_shards=2))
    assert server.digest() == s2.digest() == rep.state_digest
    assert rep.bytes_up == n_up and rep.bytes_down == len(cl) * server.manifest.shared_bytes
    st = ch.round_stats[7]
    assert st["visits"] == 5 and st["item_dropped_max_abs"] == 0.0               # SGD_M: untouched rows never move
    assert 0.0 < st["output_dropped_frac_max"] < 1.0


def test_k_equal_K_with_sgd_m_equals_dense_upload_bitwise():
    sol = solver(**SGDM)
    s_sparse, *_ = _round(K, sol)
    s_dense, *_ = _round(K, sol, channel=False)
    assert s_sparse.digest() == s_dense.digest()


def test_adamw_weight_decay_moves_untouched_rows_and_the_audit_reports_it():
    sol = solver(lr=0.05)                                                     # AdamW, weight decay 1e-5
    _, _, _, _, ch, _ = _round(K, sol)
    assert ch.round_stats[7]["item_dropped_max_abs"] > 0.0                   # disclosed, not hidden


# ------------------------------------------------------------------------------------------------ FLRun integration
def _plan(group=4):
    return ParticipationPlan.build(list(BY_KEY), seed=SEED, manifest_hash="sparse-tests", group_size=group,
                                   sampling="sweep")


def _cfg(**kw):
    base = {"run_id": "sparse", "method": "FA", "seed": SEED, "lr_peak_local": 0.05, "solver": solver(**SGDM), "n_shards": 2,
                "exposure_point": "end", "dropout_p": 0.1, "endpoint_rounds": 10}
    base.update(kw)
    return RunConfig(**base)


def _flrun(k_top=5, sparse=True, ckpt=None, resume=False, **kw):
    server, _, _ = _registered_server()
    args = (_cfg(**kw), _plan(), server, BY_KEY.__getitem__, N_DEC)
    kwa = {"workers": [_untied(101)], "ckpt": ckpt}
    run = FLRun.resume(*args, **kwa) if resume else FLRun(*args, **kwa)
    ch = SR.install_sparse(run, k_top=k_top) if sparse else None
    return run, ch


def _steps(run, ch, n):
    for _ in range(n):
        out = run.step()
        if ch is not None:
            assert run.channel is ch
            ch.check_round(out["round"], out["survived"])


def test_flrun_sparse_ledger_bytes_and_k_equal_K_equals_dense():
    run, ch = _flrun()
    _steps(run, ch, 10)
    surv = sum(x["n_survived"] for x in run.participation_log)
    _msgs, down, up, _retry = run.ledger.snapshot()
    assert down == surv * run.server.manifest.shared_bytes and up == ch.totals["bytes_up"]
    assert ch.totals["visits"] == surv and up < surv * (run.server.manifest.shared_bytes + UPLOAD_HEADER_BYTES)
    full, chf = _flrun(k_top=K)
    _steps(full, chf, 10)
    dense, _ = _flrun(sparse=False)
    _steps(dense, None, 10)
    assert full.server.digest() == dense.server.digest() != run.server.digest()
    again, ch2 = _flrun()
    _steps(again, ch2, 10)
    assert again.server.digest() == run.server.digest()                       # determinism (bitwise)
    assert ch.summary()["codec"] == SR.CODEC


def test_flrun_sparse_resume_is_exact(tmp_path):
    straight, ch = _flrun()
    _steps(straight, ch, 10)
    part, chp = _flrun(ckpt=CheckpointManager(tmp_path / "ck", "sparse", retain_weights=False))
    _steps(part, chp, 4)
    part.ckpt.save_latest(part.state_dict())
    res, chr_ = _flrun(ckpt=CheckpointManager(tmp_path / "ck", "sparse", retain_weights=False), resume=True)
    assert res.state.cursor == 4
    _steps(res, chr_, 6)
    assert res.server.digest() == straight.server.digest()
    assert res.ledger.snapshot() == straight.ledger.snapshot()
    # bytes of record come from the checkpointed ledger + counters and survive the resume, while
    # the channel's in-process summary covers only the last leg
    rec_s = SR.sparse_bytes_record(straight.server.manifest, straight.counters, straight.ledger, k_top=5)
    rec_r = SR.sparse_bytes_record(res.server.manifest, res.counters, res.ledger, k_top=5)
    assert rec_r == rec_s
    n_vis = int(sum(straight.counters.visit_counts))
    assert rec_s["fleet"]["visits"] == n_vis == ch.totals["visits"]
    assert rec_s["per_visit"]["upload_bytes_mean"] == straight.ledger.bytes_up / n_vis == ch.summary()["mean_upload_bytes"]
    assert chr_.totals["visits"] < n_vis and chr_.summary()["scope"].startswith("THIS PROCESS LEG ONLY")


def test_codec_dispatch_by_name():
    """The launcher must dispatch on the codec NAME (FA_Q8's record also sets upload_codec)."""
    q8 = {"name": "Q8_SR_PER_TENSOR", "bits": 8}
    rec = SR.sparse_codec_record()
    assert SR.is_sparse_codec(rec) and rec["k_top"] == SR.K_TOP_REGISTERED == 3170
    assert SR.is_sparse_codec(SR.sparse_codec_record(17)) and SR.sparse_codec_record(17)["k_top"] == 17
    for other in (None, q8, {"name": "SPARSE_ROWS"}, "SPARSE_ROWS_TOPK", {"k_top": 3170}):
        assert not SR.is_sparse_codec(other), other


def test_install_sparse_refusals():
    run, _ = _flrun(sparse=False, method="FP", mu=0.01)
    assert_raises(SR.SparseError, SR.install_sparse, run, k_top=5)
    a = _untied(1)
    freeze_item_tables(a, FROZEN_ITEM_KEYS)                                    # frozen tables: nothing to sparsify
    a.server_post_aggregate()
    srv = Server(a, init_sha256=state_digest(a.broadcast_state()))
    w = _untied(101)
    freeze_item_tables(w, FROZEN_ITEM_KEYS, source=a.broadcast_state())
    fr = FLRun(_cfg(), _plan(), srv, BY_KEY.__getitem__, N_DEC, workers=[w])
    assert_raises(SR.SparseError, SR.install_sparse, fr, k_top=5)
    run2, _ = _flrun(sparse=False)
    run2.pool = object()                                                      # a pool-driven run is refused
    assert_raises(SR.SparseError, SR.install_sparse, run2, k_top=5)
    assert all(int(valid_rows(c.examples).numel()) >= 0 for c in CS)


# ------------------------------------------------------------------------------------------------ negative control
@nc("k < K truncates the output layer's dense softmax delta: the run is not the dense run")
def test_nc_k_below_K_equals_dense():
    sol = solver(**SGDM)
    s_sparse, *_ = _round(3, sol)
    s_dense, *_ = _round(3, sol, channel=False)
    assert s_sparse.digest() == s_dense.digest()
