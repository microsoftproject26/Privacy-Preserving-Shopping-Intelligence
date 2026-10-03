"""The fixed evaluation numerics (EvalConfig) and the prediction path that runs under them.

Proves: the default config values; the config is immutable and a caller override is refused (kwargs that differ,
a look-alike object, invalid values); evaluate() refuses an artifact produced under another config (research score
block, another eval_batch, a tampered eval_config); the batching rule (exactly eval_batch rows, unpadded tail,
manifest order) and the runtime flags (TF32 off, deterministic algorithms with warn_only=False) are enforced; when
score_block >= K the scoring path computes exactly ONE block [0, K) (structural, recorded by a spy on
ranking.block_scores; 32,768-wide blocks fail this on every platform) and that block equals the adapter's logits bit
for bit on CPU; two invocations give the identical rank vector; rows carry eval_score_block / eval_batch /
eval_device_class / eval_stream and the tie diagnostic; trained-model rows must come from the adapter path.
Does not prove: GPU behaviour.
"""
from __future__ import annotations

import dataclasses
import operator
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from fullrank_testkit import (
    artifact_from_scores,
    assert_raises,
    config,
    manifest_from_rows,
    nc,
    random_manifest,
    scores_for,
)

from ppsi.evaluation.fullrank import config as fc
from ppsi.evaluation.fullrank import ranking
from ppsi.evaluation.fullrank.errors import AlignmentError, EvalConfigError, RowKeyError
from ppsi.evaluation.fullrank.evaluate import evaluate
from ppsi.evaluation.fullrank.predict import (
    _research_predict_adapter,
    align,
    predict_adapter,
    predict_scores,
)
from ppsi.evaluation.fullrank.ranking import rank_from_query, rank_from_scores
from ppsi.evaluation.fullrank.rows import (
    COMPARISON_KEY,
    METHODS,
    comparison_status,
    rows_from_result,
)
from ppsi.fedsim.synthetic import client_examples, make_tiny_adapter

K_BIG = 70000                      # > two 32,768-wide blocks, < the default score_block


def _tiny_case(K=24, n=64, seed=3, tied=False):
    a = make_tiny_adapter(K=K, d=8, tied=tied, dropout=0.1, seed=seed)
    ex = client_examples(n, K, 6, np.random.default_rng(seed), invalid_frac=0.2)
    tgt = ex["target_class"].numpy()
    rows = [(int(u), "R" if t >= 0 else "O", int(t)) for u, t in zip(np.arange(n) // 4, tgt, strict=True)]
    m = manifest_from_rows(rows, K_=K)
    ex["decision_id"] = torch.as_tensor(m.decision_id)
    return a, ex, m


def _batches(ex, size):
    n = ex["decision_id"].shape[0]
    return [{k: v[s:s + size] for k, v in ex.items()} for s in range(0, n, size)]


def _key_base(*drop):
    skip = {"metric", "k", "aggregation", "target_denominator", *drop}
    return {f: f"x-{f}" for f in COMPARISON_KEY if f not in skip}


# ------------------------------------------------------------------------------------------------ checks
def check_mismatch_refused(evaluate_fn):
    cfg = config()
    a, ex, m = _tiny_case()
    good = predict_adapter(a, _batches(ex, cfg.eval_batch), m, config=cfg, model_sha256="tiny")
    evaluate_fn(m, good, cfg)
    research = _research_predict_adapter(a, _batches(ex, 16), m, model_sha256="tiny", score_block=7, class_chunk=5)
    assert_raises(EvalConfigError, evaluate_fn, m, research, cfg)
    small = fc.EvalConfig(eval_batch=16)
    art16 = predict_adapter(a, _batches(ex, 16), m, config=small, model_sha256="tiny")
    assert_raises(EvalConfigError, evaluate_fn, m, art16, cfg)                 # another config's artifact
    tampered = dataclasses.replace(good, header=dict(good.header, eval_config=dict(good.header["eval_config"],
                                                                                   score_block=32768)))
    assert_raises(EvalConfigError, evaluate_fn, m, tampered, cfg)
    lookalike = SimpleNamespace(**{f.name: getattr(cfg, f.name) for f in dataclasses.fields(cfg)})
    assert_raises(EvalConfigError, evaluate_fn, m, good, lookalike)            # not an EvalConfig


class BlockSpy:
    """Records every block the REAL scoring code computes: ranking.block_scores is the one primitive that
    rank_from_query (and so predict_adapter) calls per score block; it is wrapped via monkeypatch, never replaced."""

    def __init__(self, monkeypatch):
        self.calls = []
        real = ranking.block_scores

        def spy(q, W, b, s, e):
            out = real(q, W, b, s, e)
            self.calls.append((int(s), int(e), out))
            return out

        monkeypatch.setattr(ranking, "block_scores", spy)

    def spans(self):
        return [(s, e) for s, e, _ in self.calls]

    def pass1_scores(self, K):
        """The first consecutive run of blocks covering [0, K) (pass 1), concatenated; None if it does not cover K."""
        parts, cov = [], 0
        for s, e, out in self.calls:
            if s != cov:
                break
            parts.append(out)
            cov = e
            if cov == K:
                return torch.cat(parts, 1)
        return None


def check_single_block_bitwise(scoring_path, spy):
    """STRUCTURAL first (platform-independent): the scoring path computed exactly ONE block, spanning [0, K).
    NUMERICAL second: that block equals the adapter's logits bit for bit."""
    g = torch.Generator().manual_seed(1)
    cases = []
    for d in (8, 32):
        q, W, b = (torch.randn(64, d, generator=g), torch.randn(K_BIG, d, generator=g),
                   torch.randn(K_BIG, generator=g))
        cases.append((f"random_d{d}", q, W, b, lambda q, W, b: torch.addmm(b, q, W.t())))   # the logits formula
    a = make_tiny_adapter(K=K_BIG, d=8, tied=False, seed=0)
    a.module.eval()
    with torch.no_grad():
        cases.append(("tiny_adapter", torch.randn(64, 8, generator=g), a.head_weight(), a.head_bias(),
                      lambda q, W, b: a.logits(q)))
    checks = []
    for name, q, W, b, ref in cases:
        spy.calls.clear()
        scoring_path(q, W, b)
        K = int(W.shape[0])
        spans = spy.spans()
        S = spy.pass1_scores(K)
        with torch.no_grad():
            eq = S is not None and bool(torch.equal(S, ref(q, W, b)))
        single = bool(spans) and all(sp == (0, K) for sp in spans)
        checks.append((name, K, spans, single, eq))
    for name, K, spans, single, _ in checks:                                   # 1. structural (every platform)
        assert single, f"{name}: the scoring path computed blocks {sorted(set(spans))}, not one block [0, {K})"
    for name, _, _, _, eq in checks:                                           # 2. numerical
        assert eq, f"{name}: the single block is not bit-identical to the logits"


def check_batching(predict_fn):
    rows = [(i // 3, "R", i % 24) for i in range(2500)]
    m = manifest_from_rows(rows)
    s = scores_for(m)
    ok = [(torch.as_tensor(m.decision_id[i:i + 1024]), s[i:i + 1024]) for i in (0, 1024, 2048)]
    predict_fn(ok, m)
    bad_size = [(torch.as_tensor(m.decision_id[i:i + 1000]), s[i:i + 1000]) for i in range(0, 2500, 1000)]
    assert_raises(EvalConfigError, predict_fn, bad_size, m)
    too_big = [(torch.as_tensor(m.decision_id[0:1025]), s[0:1025])]
    assert_raises(EvalConfigError, predict_fn, too_big, m)
    assert_raises(AlignmentError, predict_fn, ok[:2], m)                                   # rows not covered


# ------------------------------------------------------------------------------------------------ variants
def default_evaluate(m, art, cfg):
    return evaluate(m, art, config=cfg)


def unchecked_evaluate(m, art, cfg):
    """Bug: any artifact is evaluated, whatever produced it."""
    align(m, art)
    return {"ok": True}


def configured_scoring_path(q, W, b):
    """The real scoring path with the default config (what predict_adapter calls)."""
    cfg = config()
    rank_from_query(q, W, b, torch.zeros(q.shape[0], dtype=torch.long), score_block=cfg.score_block,
                    class_chunk=cfg.class_chunk, cache_bytes=cfg.cache_bytes, K_expected=int(W.shape[0]))


def narrow_block_scoring_path(q, W, b):
    """Bug: the same real code with 32,768-wide score blocks (three blocks at K = 70,000)."""
    cfg = config()
    rank_from_query(q, W, b, torch.zeros(q.shape[0], dtype=torch.long), score_block=32768,
                    class_chunk=cfg.class_chunk, cache_bytes=cfg.cache_bytes, K_expected=int(W.shape[0]))


def default_predict(batches, m):
    return predict_scores(batches, m, config=config(), model_sha256="x")


def unbatched_predict(batches, m):
    """Bug: accepts any batching (no eval_batch rule)."""
    pos = 0
    for d, _ in batches:
        pos += len(d)
    if pos != m.n:
        raise AlignmentError("coverage")


# ------------------------------------------------------------------------------------------------ tests
def test_default_values_and_immutability():
    cfg = fc.EvalConfig()
    assert (cfg.score_block, cfg.class_chunk, cfg.eval_batch, cfg.cache_bytes) == (262144, 32768, 1024, 1 << 30)
    assert cfg.metric_accumulation == "FP64" and dict(cfg.flags) == dict(fc.DEFAULT_FLAGS)
    assert_raises(dataclasses.FrozenInstanceError, setattr, cfg, "score_block", 4096)
    assert_raises(TypeError, operator.setitem, cfg.flags, "cudnn.benchmark", True)
    assert cfg.sha256 == fc.EvalConfig().sha256 != fc.EvalConfig(eval_batch=512).sha256
    for bad in ({"score_block": 0}, {"eval_batch": 1.5}, {"class_chunk": True}, {"cache_bytes": -1},
                {"metric_accumulation": "FP32"}):
        assert_raises(EvalConfigError, fc.EvalConfig, **bad)


def test_overrides_refused():
    cfg = config()
    assert fc.resolve(None) == cfg and fc.resolve(cfg, score_block=cfg.score_block) is cfg
    for kw in ({"score_block": 4096}, {"eval_batch": 512}, {"cache_bytes": 0}, {"class_chunk": 7}):
        assert_raises(EvalConfigError, fc.resolve, cfg, **kw)
    assert_raises(EvalConfigError, fc.resolve, {"eval_batch": 512})


def test_mismatched_config_refused():
    check_mismatch_refused(default_evaluate)


@nc("evaluate() ignores the configuration that produced the artifact")
def test_nc_unchecked_evaluate():
    check_mismatch_refused(unchecked_evaluate)


def test_single_block_equals_adapter_logits_bitwise_cpu(monkeypatch):
    check_single_block_bitwise(configured_scoring_path, BlockSpy(monkeypatch))


def test_predict_uses_one_block_per_batch(monkeypatch):
    """End to end: predict_adapter computes exactly one [0, K) block per batch when score_block >= K."""
    spy = BlockSpy(monkeypatch)
    a = make_tiny_adapter(K=K_BIG, d=8, tied=False, seed=0)
    n = 40
    ex = client_examples(n, K_BIG, 6, np.random.default_rng(1), invalid_frac=0.0)
    m = manifest_from_rows([(i // 4, "R", int(t)) for i, t in enumerate(ex["target_class"].numpy())], K_=K_BIG)
    ex["decision_id"] = torch.as_tensor(m.decision_id)
    predict_adapter(a, [ex], m, config=config(), model_sha256="t")
    assert spy.spans() and all(sp == (0, K_BIG) for sp in spy.spans()), sorted(set(spy.spans()))


def test_negative_control_fails_structurally_even_where_blocks_are_bit_identical(monkeypatch):
    """Platform independence of the negative control: simulate a GEMM on which 32,768-wide blocks ARE bit-identical
    to the single block by forcing torch.equal to True; the check must still fail, with an AssertionError raised by
    the STRUCTURAL assertion."""
    spy = BlockSpy(monkeypatch)
    monkeypatch.setattr(torch, "equal", lambda a, b: True)
    with pytest.raises(AssertionError, match="not one block"):
        check_single_block_bitwise(narrow_block_scoring_path, spy)


@nc("the same scoring code with 32,768-wide score blocks: several blocks, not one block of width K")
def test_nc_multi_block_not_single(monkeypatch):
    check_single_block_bitwise(narrow_block_scoring_path, BlockSpy(monkeypatch))


def test_batching():
    check_batching(default_predict)


@nc("a prediction path without the eval_batch rule")
def test_nc_unbatched():
    check_batching(unbatched_predict)


def test_runtime_flags_enforced():
    cfg = config()
    a, ex, m = _tiny_case()
    torch.backends.cuda.matmul.allow_tf32 = True
    assert_raises(EvalConfigError, predict_adapter, a, _batches(ex, 1024), m, config=cfg, model_sha256="t")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    assert_raises(EvalConfigError, predict_adapter, a, _batches(ex, 1024), m, config=cfg, model_sha256="t")
    fc.apply_runtime_flags(cfg)
    predict_adapter(a, _batches(ex, 1024), m, config=cfg, model_sha256="t")


def test_two_invocations_identical_and_equal_to_logits_ranks():
    cfg = config()
    for tied in (False, True):
        a, ex, m = _tiny_case(tied=tied, n=200)
        r1 = predict_adapter(a, _batches(ex, cfg.eval_batch), m, config=cfg, model_sha256="t")
        r2 = predict_adapter(a, _batches(ex, cfg.eval_batch), m, config=cfg, model_sha256="t")
        assert np.array_equal(r1.ranks, r2.ranks) and np.array_equal(r1.n_tied, r2.n_tied)
        a.module.eval()
        rk = m.rankable
        with torch.no_grad():
            q = a.query(ex)[torch.as_tensor(rk)]
            ref = rank_from_scores(a.logits(q), torch.as_tensor(m.target_class[rk])).ranks
        assert np.array_equal(r1.ranks[rk], ref.numpy())
        assert r1.header["eval_config"]["eval_device_class"] == "CPU-FP32-DET"


def test_rows_carry_key_additions_and_tie_diagnostic():
    cfg = config()
    a, ex, m = _tiny_case(n=120)
    art = predict_adapter(a, _batches(ex, cfg.eval_batch), m, config=cfg, model_sha256="t")
    res = evaluate(m, art, config=cfg)
    base = _key_base("eval_score_block", "eval_batch", "eval_device_class")
    rows = rows_from_result(res, base, method="PF", seed=2026, selection_rule="PRACTICAL_BEST", efe=6.0)
    assert {"eval_score_block", "eval_batch", "eval_device_class"} <= set(COMPARISON_KEY)
    assert all(r["eval_score_block"] == cfg.score_block and r["eval_batch"] == 1024
               and r["eval_device_class"] == "CPU-FP32-DET" for r in rows)
    assert all(r["status"] == "MEASURED" for r in rows)                       # no reference device class set
    assert all("n_target_tied" in r and "value_pessimistic" in r for r in rows)
    pess = art.pessimistic_ranks
    rk = m.rankable
    assert np.all(pess[rk] >= art.ranks[rk]) and np.all(pess[~rk] == -1)
    small = fc.EvalConfig(eval_batch=16)
    art16 = predict_adapter(a, _batches(ex, 16), m, config=small, model_sha256="t")
    rows16 = rows_from_result(evaluate(m, art16, config=small), base, method="PF", seed=2026)
    assert rows16[0]["eval_batch"] == 16
    assert comparison_status(rows[0], rows16[0]) == ("HISTORICAL_REFERENCE", ["eval_batch"])
    gpu_ref = fc.EvalConfig(reference_device_class="SOME_GPU-FP32-DET")
    art_ref = predict_adapter(a, _batches(ex, 1024), m, config=gpu_ref, model_sha256="t")
    rows_ref = rows_from_result(evaluate(m, art_ref, config=gpu_ref), base, method="PF", seed=2026)
    assert all(r["status"] == "MEASURED_OTHER_DEVICE" for r in rows_ref)


def test_stream_forms_e2e_rows_and_all_rows():
    cfg = config()
    a, ex, _ = _tiny_case(n=40)
    kinds = ["C" if i % 5 == 0 else ("R" if int(ex["target_class"][i]) >= 0 else "O") for i in range(40)]
    rows = [(i // 4, k, max(int(ex["target_class"][i]), 0)) for i, k in enumerate(kinds)]
    m = manifest_from_rows(rows, K_=24)
    keep = np.flatnonzero(m.eval_mask)
    sub = {k: v[torch.as_tensor(keep)] for k, v in ex.items() if k != "target_class"}
    sub["decision_id"] = torch.as_tensor(m.decision_id[keep])
    art = predict_adapter(a, [sub], m, config=cfg, model_sha256="t")            # E2E_ROWS: censored rows skipped
    assert art.header["eval_config"]["eval_stream"] == "E2E_ROWS"
    assert np.all(art.ranks[~m.eval_mask] == -1) and np.all(art.ranks[m.rankable] >= 1)
    full = dict(ex, decision_id=torch.as_tensor(m.decision_id))
    full.pop("target_class")
    art_all = predict_adapter(a, [full], m, config=cfg, model_sha256="t")      # ALL_ROWS: every manifest row
    assert art_all.header["eval_config"]["eval_stream"] == "ALL_ROWS"
    assert np.all(art_all.ranks[~m.eval_mask] == -1)
    mixed = [{k: v[:10] for k, v in full.items()}, {k: v[5:] for k, v in sub.items()}]   # neither form: refused
    assert_raises((AlignmentError, EvalConfigError), predict_adapter, a, mixed, m, config=cfg, model_sha256="t")
    base = _key_base("eval_score_block", "eval_batch", "eval_device_class", "eval_stream")
    r_e2e = rows_from_result(evaluate(m, art, config=cfg), base, method="C", seed=2026)
    r_all = rows_from_result(evaluate(m, art_all, config=cfg), base, method="C", seed=2026)
    assert comparison_status(r_e2e[0], r_all[0]) == ("HISTORICAL_REFERENCE", ["eval_stream"])


def test_kwargs_must_equal_the_configured_values():
    """A caller may pass the numerics only if they equal the configured ones."""
    cfg = config()
    a, ex, m = _tiny_case()
    b = _batches(ex, 1024)
    predict_adapter(a, b, m, model_sha256="t", score_block=cfg.score_block, class_chunk=cfg.class_chunk,
                    cache_bytes=cfg.cache_bytes)
    for kw in ({"score_block": 32768}, {"class_chunk": 1024}, {"cache_bytes": 0}, {"eval_batch": 512}):
        assert_raises(EvalConfigError, predict_adapter, a, b, m, model_sha256="t", **kw)
    art = predict_adapter(a, b, m, model_sha256="t")                                     # config defaults
    evaluate(m, art)
    assert_raises(EvalConfigError, predict_scores, [], m, model_sha256="t", class_chunk=7)


def test_score_block_smaller_than_the_catalogue_refused():
    a, ex, m = _tiny_case()
    narrow = fc.EvalConfig(score_block=8)
    assert_raises(EvalConfigError, predict_adapter, a, _batches(ex, 1024), m, config=narrow, model_sha256="t")


def test_method_rows_require_the_adapter_path():
    m = random_manifest(n_users=10, seed=3)
    res = evaluate(m, artifact_from_scores(m, scores_for(m)), config=config())
    base = _key_base()
    for method in METHODS:
        assert_raises(RowKeyError, rows_from_result, res, base, method=method, seed=2026)
    rows_from_result(res, base, method="TRANS", seed=2026)                               # a baseline may use it
