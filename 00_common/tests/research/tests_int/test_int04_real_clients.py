"""INT-04 — launch_device's real-data wiring on the INT-01 test environment (real v1 TRAIN / VALIDATION rows, bucket 0,
CPU, the real SASREC_D64_B2): arbitrary-position evaluation batches equal the provider's own stream batches; support
rows carry the manifest's target_ts; LO end to end (2 real clients with support, every other evaluation user without
support) through the frozen evaluator; a no-support user's ranks match predict_adapter's with theta0 up to float32 near-ties
(per-client batch composition; MEASURED here: < 1% of rows, |delta rank| <= 1). (The M06 test-kit
users table has no client_key_hex, so the eval-user -> client mapping is the test's own.)"""
from __future__ import annotations

import numpy as np
import torch

import v4int.launch_device as D
import v4int.launch_fl as F
from intkit import REAL_DS


def test_eval_batch_at_equals_stream_batches(providers):
    _, ep = providers
    ep_b = ep._make(0, 7)
    pos = np.arange(7, dtype=np.int64)
    got = D.eval_batch_at(ep, pos)
    for k, v in ep_b.items():
        if torch.is_tensor(v):
            assert torch.equal(v, got[k]), k
    sub = D.eval_batch_at(ep, np.array([3, 5], dtype=np.int64))
    assert torch.equal(sub["decision_id"], ep_b["decision_id"][[3, 5]])


def test_lo_real_clients_end_to_end(tmp_path, real_env, providers):
    from fl.numerics import state_digest
    from v4eval import frozen_config as fc
    from v4eval.predict import predict_adapter
    from v4models import build, load_schema
    tp, ep = providers
    mdir = real_env["paths"].manifest_dir
    keys, pos = F.client_index(tp, mdir)
    ts = D.train_target_ts_by_position(tp, mdir)
    from v4data.common import TRAIN_END_US
    assert ts.shape == (tp.n_decisions,) and int(ts.max()) < TRAIN_END_US
    m = ep.manifest
    euk = m.eval_user_key
    row_key = np.array([f"eval-user-{int(u):05d}" for u in euk], dtype=object)
    users = np.unique(euk[m.rankable])[:2]
    for u, k in zip(users, keys[:2]):                    # the test's own mapping: two eval users get real support
        row_key[euk == u] = k
    plans = D.build_client_plans(m, row_key, row_decision_ts=None, support_pos_by_key={k: pos[k] for k in keys[:2]},
                                 warm=True)
    assert sum(p.support is not None for p in plans) == 2
    mk = lambda: build("SASREC", load_schema(), 2026, "cpu", v1_root=REAL_DS, variant="SASREC_D64_B2",  # noqa: E731
                       dropout=(0.1,)).adapter
    harness, worker = mk(), mk()
    start = harness.broadcast_state(clone=True)
    spec = D.device_spec("LO", seed=2026, peak_lr=1e-3, role="MAIN", start_digest=state_digest(start))

    def get_support(p):
        b = tp.batch(p.support)
        b[D.TS_KEY] = torch.as_tensor(ts[p.support], dtype=torch.long)
        return b
    fz = fc.load()
    out = D.execute(spec=spec, run_dir=tmp_path / "lo", worker=worker, start_state=start, plans=plans,
                    get_support=get_support, get_eval_batch=lambda sp: D.eval_batch_at(ep, sp), manifest=m, frozen=fz,
                    n_shards=3, say=lambda *_: None)
    assert out["status"] == "DONE" and out["n_no_support"] == len(plans) - 2
    art = predict_adapter(harness, ep.batches(fz.eval_batch), m, config=fz, model_sha256="theta0")
    got = np.load(tmp_path / "lo" / "cells" / "LO_P2.ranks_int32.npy")
    nosup = np.concatenate([p.rows for p in plans if p.support is None])
    sup = np.concatenate([p.rows for p in plans if p.support is not None])
    a, b = got[nosup].astype(np.int64), art.ranks[nosup]
    diff = a != b
    # MEASURED (CPU, real SASREC_D64_B2): per-client batch composition changes float32 near-ties on a few rows only
    # (6 / 1,251 rows, |delta rank| = 1, all at ranks > 20); disclosed in the module docstring and the handoff
    assert diff.mean() < 0.01 and int(np.abs(a - b).max()) <= 1
    assert np.array_equal(np.minimum(a, 21), np.minimum(b, 21))               # identical @20 credit on this fixture
    assert not np.array_equal(got[sup], art.ranks[sup].astype(np.int32))    # the two trained clients moved
