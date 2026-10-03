"""One full-batch SGD local step, aggregated over clients, equals the corresponding pooled SGD step (synthetic
sanity only; not Adam or multi-local-step equivalence).

Hand derivation (plain SGD, no momentum, no weight decay, no gradient clipping): for client i with n_i
examples and mean CE gradient g_i(theta) = (1/n_i) sum_{examples} grad, a single full-batch SGD step gives
theta_i' = theta - lr * g_i(theta). FedAvg's weighted average with weight n_i is
  sum_i n_i * theta_i' / sum_i n_i = theta - lr * (1/N) * sum_i n_i * g_i(theta)
                                    = theta - lr * (1/N) * sum_i sum_{examples in i} grad
which is exactly one pooled full-batch SGD step over the union of all N = sum_i n_i examples (mean CE
gradient over the pooled batch). This is basic gradient linearity.

`solver.optimizer="sgd"` is reserved in client.py ONLY for this pooled-step sanity check.
"""
from __future__ import annotations

import torch
from fedsim_crosscheck_kit import fresh_adapter, one_client

from ppsi.fedsim.aggregate import aggregate_uploads
from ppsi.fedsim.client import LocalSolver, batch_ce, client_update, make_optimizer


def test_one_full_batch_sgd_step_aggregation_equals_pooled_sgd_step():
    K, d = 8, 5
    clients = [one_client("c0", n=3, K=K, L=4, seed=1), one_client("c1", n=5, K=K, L=4, seed=2),
              one_client("c2", n=4, K=K, L=4, seed=3)]
    lr = 0.05
    solver = LocalSolver(lr=lr, passes=1, batch_size=64, optimizer="sgd", clip=None, weight_decay=0.0)

    theta_r = fresh_adapter(K=K, d=d, tied=True, seed=7).broadcast_state(clone=True)
    manifest = fresh_adapter(K=K, d=d, tied=True, seed=7).manifest

    uploads = []
    for c in clients:
        adapter = fresh_adapter(K=K, d=d, tied=True, seed=7)
        res = client_update(adapter, theta_r, c, solver, round_idx=0, seed=13)
        uploads.append((c.key, res.upload, res.n_consumed))
    fedavg_theta = aggregate_uploads(manifest, theta_r, uploads, n_shards=1)

    # Pooled reference: one big batch of all N=12 examples, one manual SGD step from the SAME theta_r.
    pooled_adapter = fresh_adapter(K=K, d=d, tied=True, seed=7)
    pooled_adapter.load_state_(theta_r)
    pooled_examples = {k: torch.cat([c.examples[k] for c in clients], dim=0) for k in clients[0].examples}
    opt = make_optimizer(pooled_adapter, solver)
    opt.zero_grad(set_to_none=True)
    logits = pooled_adapter.scores(pooled_examples)
    loss = batch_ce(logits, pooled_examples["target_class"])
    loss.backward()
    opt.step()
    pooled_theta = pooled_adapter.extract_shared(clone=True)

    n_total = sum(int(c.examples["target_class"].shape[0]) for c in clients)
    assert n_total == 12
    for k in manifest.shared_keys:
        diff = float((fedavg_theta[k] - pooled_theta[k]).abs().max())
        # FP32 summation order differs (per-client mean-then-weight vs one pooled mean): expect a tiny,
        # not-exactly-zero difference, similar in spirit to the ~1e-7 of test_fedsim_sgd_pooled.py (a different fixture).
        assert diff < 2e-5, f"FedAvg(single SGD step) must equal the pooled SGD step for {k} (diff={diff})"


def test_uniform_weights_would_have_given_a_visibly_different_wrong_answer():
    # Negative-control-style contrast: using UNIFORM client weights instead of n_i-weighting breaks the pooled-step
    # equivalence when client sizes are unequal (3, 5, 4) -- proving the n_i-weighting is load-bearing, not
    # merely stylistic.
    K, d = 8, 5
    clients = [one_client("c0", n=3, K=K, L=4, seed=1), one_client("c1", n=5, K=K, L=4, seed=2),
              one_client("c2", n=4, K=K, L=4, seed=3)]
    lr = 0.05
    solver = LocalSolver(lr=lr, passes=1, batch_size=64, optimizer="sgd", clip=None, weight_decay=0.0)
    theta_r = fresh_adapter(K=K, d=d, tied=True, seed=7).broadcast_state(clone=True)
    manifest = fresh_adapter(K=K, d=d, tied=True, seed=7).manifest

    uploads_correct, uploads_uniform = [], []
    for c in clients:
        adapter = fresh_adapter(K=K, d=d, tied=True, seed=7)
        res = client_update(adapter, theta_r, c, solver, round_idx=0, seed=13)
        uploads_correct.append((c.key, res.upload, res.n_consumed))
        uploads_uniform.append((c.key, res.upload, 1))       # WRONG: uniform weight regardless of n_i

    correct = aggregate_uploads(manifest, theta_r, uploads_correct, n_shards=1)
    uniform = aggregate_uploads(manifest, theta_r, uploads_uniform, n_shards=1)
    any_key = manifest.shared_keys[0]
    assert not torch.allclose(correct[any_key], uniform[any_key], atol=1e-6), (
        "n_i-weighting is load-bearing for the pooled-step equivalence; uniform weights must give a different answer")
