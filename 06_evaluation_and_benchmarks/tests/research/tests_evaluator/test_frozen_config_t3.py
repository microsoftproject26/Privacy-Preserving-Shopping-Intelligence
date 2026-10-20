"""EVAL-03 Part A — the frozen T3 evaluation numerics (contracts/PREG1_LOCK_VALUES.json, T3_eval_numerics).

Proves: load() returns exactly the frozen T3 values and the lock file's sha256; a lock whose T3 values differ is refused;
every caller override is refused (kwargs, dict "amendments", direct construction, mutation); the one permitted change is
a recorded EvalBatchAmendment that lowers eval_batch; every public evaluate path requires the frozen config, and an
artifact produced under a different config (research score block, amended eval_batch, tampered eval_config, a lock
file changed after load) is refused; the frozen batching (exactly eval_batch rows, unpadded tail, manifest order) and
runtime flags (TF32 off, deterministic algorithms with warn_only=False) are enforced; at K = 158,486 the frozen
scoring path computes exactly ONE block [0, K) (structural, recorded by a spy on v4eval.ranking.block_scores; the
negative control's 32,768-wide blocks fail this on every platform) and that block equals BaseAdapter.logits
bit-for-bit on CPU (whether multi-block results also differ is platform-dependent and only recorded in
receipts/T3_PLATFORM_BITWISE_NOTE.json); two frozen invocations give the
identical rank vector; rows carry eval_score_block / eval_batch / eval_device_class and the tie diagnostic.
Does not prove: the A16 GPU behaviour (EVAL-03 Part B receipt).
"""
from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from evaltestkit import (artifact_from_scores, assert_raises, frozen, manifest_from_rows, nc, random_manifest,
                         scores_for)
from fl.synthetic import client_examples, make_tiny_adapter
from v4eval import frozen_config as fc
from v4eval.errors import AlignmentError
from v4eval.evaluate import evaluate
from v4eval.hashing import file_sha256
from v4eval.predict import _research_predict_adapter, align, predict_adapter, predict_scores
from v4eval.ranking import rank_from_query, rank_from_scores
from v4eval.rows import COMPARISON_KEY, comparison_status, rows_from_result

K_FROZEN = 158486


def _amend(eval_batch=512, **kw):
    base = dict(amendment_id="AMEND-EVAL-BATCH-0001", eval_batch=eval_batch, reason="GPU memory contention (synthetic)",
                recorded_utc="2026-09-26T19:40:00Z", recorded_by="Operations Director")
    base.update(kw)
    return fc.EvalBatchAmendment(**base)


def _tiny_case(K=24, n=64, seed=3, tied=False):
    a = make_tiny_adapter(K=K, d=8, tied=tied, dropout=0.1, seed=seed)
    ex = client_examples(n, K, 6, np.random.default_rng(seed), invalid_frac=0.2)
    tgt = ex["target_class"].numpy()
    rows = [(int(u), "R" if t >= 0 else "O", int(t)) for u, t in zip(np.arange(n) // 4, tgt)]
    m = manifest_from_rows(rows, K_=K)
    ex["decision_id"] = torch.as_tensor(m.decision_id)
    return a, ex, m


def _batches(ex, size):
    n = ex["decision_id"].shape[0]
    return [{k: v[s:s + size] for k, v in ex.items()} for s in range(0, n, size)]


# ------------------------------------------------------------------------------------------------ checks
def check_frozen_values(loader):
    cfg = loader()
    assert (cfg.score_block, cfg.class_chunk, cfg.eval_batch, cfg.cache_bytes) == (158486, 32768, 1024, 1073741824)
    assert cfg.metric_accumulation == "FP64" and cfg.device_class == "A16-FP32-DET"
    assert dict(cfg.flags) == fc.EXPECTED_T3["flags"]
    assert cfg.lock_sha256 == file_sha256(fc.DEFAULT_LOCK_PATH)
    assert set(fc.METHODS) <= set(cfg.applies_to)


def check_override_refused(load_fn):
    assert_raises(fc.FrozenConfigError, load_fn, score_block=4096)
    assert_raises(fc.FrozenConfigError, load_fn, eval_batch=512)
    assert_raises(fc.FrozenConfigError, load_fn, cache_bytes=0)
    assert_raises(fc.FrozenConfigError, load_fn, amendment={"eval_batch": 512})
    assert_raises(fc.FrozenConfigError, load_fn, amendment=_amend(eval_batch=2048))       # only a reduction
    assert_raises(fc.FrozenConfigError, load_fn, amendment=_amend(reason=""))
    assert_raises(fc.FrozenConfigError, load_fn, amendment=_amend(recorded_utc="yesterday"))
    cfg = load_fn()
    assert_raises(dataclasses.FrozenInstanceError, setattr, cfg, "score_block", 4096)
    assert_raises(fc.FrozenConfigError, fc.FrozenEvalConfig, **{f.name: getattr(cfg, f.name)
                                                                 for f in dataclasses.fields(cfg) if f.name != "_token"})


def check_mismatch_refused(evaluate_fn, tmp_path):
    cfg = frozen()
    a, ex, m = _tiny_case()
    good = predict_adapter(a, _batches(ex, cfg.eval_batch), m, config=cfg, model_sha256="tiny")
    evaluate_fn(m, good, cfg)
    research = _research_predict_adapter(a, _batches(ex, 16), m, model_sha256="tiny", score_block=7, class_chunk=5)
    assert_raises(fc.FrozenConfigError, evaluate_fn, m, research, cfg)
    amended = fc.load(amendment=_amend(eval_batch=16))
    art16 = predict_adapter(a, _batches(ex, 16), m, config=amended, model_sha256="tiny")
    assert_raises(fc.FrozenConfigError, evaluate_fn, m, art16, cfg)                       # amended artifact, default cfg
    tampered = dataclasses.replace(good, header=dict(good.header, eval_config=dict(good.header["eval_config"],
                                                                                   score_block=32768)))
    assert_raises(fc.FrozenConfigError, evaluate_fn, m, tampered, cfg)
    lookalike = SimpleNamespace(**{f.name: getattr(cfg, f.name) for f in dataclasses.fields(cfg)})
    assert_raises(fc.FrozenConfigError, evaluate_fn, m, good, lookalike)                  # not from load()
    lock = tmp_path / "LOCK.json"
    lock.write_bytes(fc.DEFAULT_LOCK_PATH.read_bytes())
    moved = fc.load(lock)
    lock.write_bytes(lock.read_bytes() + b"\n")                                            # changed after load
    assert_raises(fc.FrozenConfigError, evaluate_fn, m, good, moved)


class BlockSpy:
    """Records every block the REAL scoring code computes: v4eval.ranking.block_scores is the one primitive that
    rank_from_query (and so predict_adapter) calls per score block; it is wrapped via monkeypatch, never replaced."""

    def __init__(self, monkeypatch):
        import v4eval.ranking as ranking
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


NOTE = Path(__file__).resolve().parent / "receipts" / "T3_PLATFORM_BITWISE_NOTE.json"


def _write_note(section: str, cases: dict) -> None:
    """Informational only: whether the platform's GEMM made a multi-block result bit-identical (never asserted)."""
    import platform
    NOTE.parent.mkdir(exist_ok=True)
    doc = json.loads(NOTE.read_text(encoding="utf-8")) if NOTE.is_file() else {}
    doc["platform"] = {"python": platform.python_version(), "machine": platform.machine(),
                       "system": platform.system(), "torch": torch.__version__}
    doc[section] = cases
    NOTE.write_text(json.dumps(doc, indent=1, sort_keys=True), encoding="utf-8")


def check_single_block_bitwise(scoring_path, spy, section):
    """T3 single-block property. STRUCTURAL first (platform-independent): the scoring path computed exactly ONE block,
    spanning [0, K). NUMERICAL second: that block equals BaseAdapter.logits bit-for-bit."""
    g = torch.Generator().manual_seed(1)
    cases = []
    for d in (8, 32):
        q, W, b = torch.randn(64, d, generator=g), torch.randn(K_FROZEN, d, generator=g), torch.randn(K_FROZEN, generator=g)
        cases.append((f"random_d{d}", q, W, b, lambda q, W, b: torch.addmm(b, q, W.t())))   # BaseAdapter.logits formula
    a = make_tiny_adapter(K=K_FROZEN, d=8, tied=False, seed=0)
    a.module.eval()
    with torch.no_grad():
        cases.append(("tiny_adapter", torch.randn(64, 8, generator=g), a.head_weight(), a.head_bias(),
                      lambda q, W, b: a.logits(q)))
    info, checks = {}, []
    for name, q, W, b, ref in cases:
        spy.calls.clear()
        scoring_path(q, W, b)
        K = int(W.shape[0])
        spans = spy.spans()
        S = spy.pass1_scores(K)
        with torch.no_grad():
            eq = S is not None and bool(torch.equal(S, ref(q, W, b)))
        single = bool(spans) and all(sp == (0, K) for sp in spans)
        info[name] = {"K": K, "n_block_calls": len(spans), "block_spans": sorted(set(spans)),
                      "single_block_of_width_K": single, "bitwise_equal_to_logits": eq}
        checks.append((name, K, spans, single, eq))
    _write_note(section, info)
    for name, K, spans, single, _ in checks:                                   # 1. structural (every platform)
        assert single, f"{name}: the scoring path computed blocks {sorted(set(spans))}, not one block [0, {K})"
    for name, _, _, _, eq in checks:                                           # 2. numerical
        assert eq, f"{name}: the single block is not bit-identical to BaseAdapter.logits"


def check_batching(predict_fn):
    rows = [(i // 3, "R", i % 24) for i in range(2500)]
    m = manifest_from_rows(rows)
    s = scores_for(m)
    ok = [(torch.as_tensor(m.decision_id[i:i + 1024]), s[i:i + 1024]) for i in (0, 1024, 2048)]
    predict_fn(ok, m)
    bad_size = [(torch.as_tensor(m.decision_id[i:i + 1000]), s[i:i + 1000]) for i in range(0, 2500, 1000)]
    assert_raises(fc.FrozenConfigError, predict_fn, bad_size, m)
    too_big = [(torch.as_tensor(m.decision_id[0:1025]), s[0:1025])]
    assert_raises(fc.FrozenConfigError, predict_fn, too_big, m)
    assert_raises(AlignmentError, predict_fn, ok[:2], m)                                   # rows not covered


# ------------------------------------------------------------------------------------------------ variants
def unverified_loader():
    """Bug: trusts whatever the JSON says (here a tampered score_block) without the frozen checks."""
    t3 = dict(json.loads(fc.DEFAULT_LOCK_PATH.read_text(encoding="utf-8"))["T3_eval_numerics"], score_block=32768)
    return SimpleNamespace(score_block=t3["score_block"], class_chunk=t3["class_chunk"], eval_batch=t3["eval_batch"],
                           cache_bytes=t3["cache_bytes"], metric_accumulation=t3["metric_accumulation"],
                           device_class=t3["comparison_key_additions"]["eval_device_class"], flags=t3["flags"],
                           lock_sha256=file_sha256(fc.DEFAULT_LOCK_PATH), applies_to=t3["applies_to"])


def permissive_load(**kw):
    kw.pop("amendment", None)
    return SimpleNamespace(**{"score_block": 158486, **kw})


def v4_evaluate(m, art, cfg):
    return evaluate(m, art, config=cfg)


def unchecked_evaluate(m, art, cfg):
    """Bug: the EVAL-02 behaviour — any artifact is evaluated, whatever produced it."""
    align(m, art)
    return {"ok": True}


def frozen_scoring_path(q, W, b):
    """The real frozen scoring path: rank_from_query with the T3 values (what predict_adapter calls)."""
    cfg = frozen()
    rank_from_query(q, W, b, torch.zeros(q.shape[0], dtype=torch.long), score_block=cfg.score_block,
                    class_chunk=cfg.class_chunk, cache_bytes=cfg.cache_bytes, K_expected=int(W.shape[0]))


def pre_t3_scoring_path(q, W, b):
    """Bug: the same real code with the pre-T3 score block of 32,768 (the EVAL-02 default): five blocks at K = 158,486."""
    cfg = frozen()
    rank_from_query(q, W, b, torch.zeros(q.shape[0], dtype=torch.long), score_block=32768,
                    class_chunk=cfg.class_chunk, cache_bytes=cfg.cache_bytes, K_expected=int(W.shape[0]))


def v4_predict(batches, m):
    return predict_scores(batches, m, config=frozen(), model_sha256="x")


def unbatched_predict(batches, m):
    """Bug: accepts any batching (no frozen eval_batch rule)."""
    pos = 0
    for d, _ in batches:
        pos += len(d)
    if pos != m.n:
        raise AlignmentError("coverage")


# ------------------------------------------------------------------------------------------------ tests
def test_t3_frozen_values():
    check_frozen_values(fc.load)


@nc("a loader that trusts a tampered lock value without verification")
def test_nc_t3_unverified_loader():
    check_frozen_values(unverified_loader)


def test_t3_tampered_lock_refused(tmp_path):
    doc = json.loads(fc.DEFAULT_LOCK_PATH.read_text(encoding="utf-8"))
    for mutate in (lambda d: d["T3_eval_numerics"].__setitem__("score_block", 32768),
                   lambda d: d["T3_eval_numerics"]["flags"].__setitem__("use_deterministic_algorithms", False),
                   lambda d: d["T3_eval_numerics"]["comparison_key_additions"].__setitem__("eval_batch", 512),
                   lambda d: d.__setitem__("status", "DRAFT")):
        d = json.loads(json.dumps(doc))
        mutate(d)
        p = tmp_path / "PREG1_LOCK_VALUES.json"
        p.write_text(json.dumps(d), encoding="utf-8")
        assert_raises(fc.FrozenConfigError, fc.load, p)
    p.write_text('{"document": NaN}', encoding="utf-8")
    assert_raises(fc.FrozenConfigError, fc.load, p)


def test_t3_override_refused():
    check_override_refused(fc.load)
    cfg = fc.load(amendment=_amend(eval_batch=512))
    assert cfg.eval_batch == 512 and cfg.score_block == 158486 and cfg.amendment_id == "AMEND-EVAL-BATCH-0001"


@nc("a loader that accepts caller overrides")
def test_nc_t3_permissive_overrides():
    check_override_refused(permissive_load)


def test_t3_amendment_record_verified(tmp_path):
    rec = tmp_path / "amendment.json"
    rec.write_text('{"amendment": "eval_batch 512"}', encoding="utf-8")
    fc.load(amendment=_amend(record_path=str(rec), record_sha256=file_sha256(rec)))
    assert_raises(fc.FrozenConfigError, fc.load, amendment=_amend(record_path=str(rec), record_sha256="0" * 64))


def test_t3_mismatched_config_refused(tmp_path):
    check_mismatch_refused(v4_evaluate, tmp_path)


@nc("evaluate() ignores the configuration that produced the artifact")
def test_nc_t3_unchecked_evaluate(tmp_path):
    check_mismatch_refused(unchecked_evaluate, tmp_path)


def test_t3_single_block_equals_adapter_logits_bitwise_cpu(monkeypatch):
    check_single_block_bitwise(frozen_scoring_path, BlockSpy(monkeypatch), "frozen_path")


def test_t3_frozen_predict_uses_one_block_per_batch(monkeypatch):
    """End to end: predict_adapter under the frozen config computes exactly one [0, K) block per batch at K = 158,486."""
    spy = BlockSpy(monkeypatch)
    a = make_tiny_adapter(K=K_FROZEN, d=8, tied=False, seed=0)
    n = 40
    ex = client_examples(n, K_FROZEN, 6, np.random.default_rng(1), invalid_frac=0.0)
    m = manifest_from_rows([(i // 4, "R", int(t)) for i, t in enumerate(ex["target_class"].numpy())], K_=K_FROZEN)
    ex["decision_id"] = torch.as_tensor(m.decision_id)
    predict_adapter(a, [ex], m, config=frozen(), model_sha256="t")
    assert spy.spans() and all(sp == (0, K_FROZEN) for sp in spy.spans()), sorted(set(spy.spans()))


def test_t3_negative_control_fails_structurally_even_where_blocks_are_bit_identical(monkeypatch):
    """Platform independence of the negative control: simulate a GEMM on which 32,768-wide blocks ARE bit-identical to
    the single block (as observed on the Linux server, OPS-07) by forcing torch.equal to True; the check must still
    fail, with an AssertionError raised by the STRUCTURAL assertion."""
    spy = BlockSpy(monkeypatch)
    monkeypatch.setattr(torch, "equal", lambda a, b: True)
    with pytest.raises(AssertionError, match="not one block"):
        check_single_block_bitwise(pre_t3_scoring_path, spy, "meta_simulated_bit_identical_platform")


@nc("the same scoring code with 32,768-wide score blocks: five blocks, not one block of width K (structural, every "
    "platform); whether the platform also breaks bit-identity is recorded in receipts/T3_PLATFORM_BITWISE_NOTE.json")
def test_nc_t3_multi_block_not_bitwise(monkeypatch):
    check_single_block_bitwise(pre_t3_scoring_path, BlockSpy(monkeypatch), "negative_control_32768_blocks")


def test_t3_frozen_batching():
    check_batching(v4_predict)


@nc("a prediction path without the frozen eval_batch rule")
def test_nc_t3_unbatched():
    check_batching(unbatched_predict)


def test_t3_runtime_flags_enforced():
    cfg = frozen()
    a, ex, m = _tiny_case()
    torch.backends.cuda.matmul.allow_tf32 = True
    assert_raises(fc.FrozenConfigError, predict_adapter, a, _batches(ex, 1024), m, config=cfg, model_sha256="t")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    assert_raises(fc.FrozenConfigError, predict_adapter, a, _batches(ex, 1024), m, config=cfg, model_sha256="t")
    fc.apply_runtime_flags(cfg)
    predict_adapter(a, _batches(ex, 1024), m, config=cfg, model_sha256="t")


def test_t3_two_invocations_identical_and_equal_to_logits_ranks():
    cfg = frozen()
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


def test_t3_rows_carry_key_additions_and_tie_diagnostic():
    cfg = frozen()
    a, ex, m = _tiny_case(n=120)
    art = predict_adapter(a, _batches(ex, cfg.eval_batch), m, config=cfg, model_sha256="t")
    res = evaluate(m, art, config=cfg)
    base = {f: f"x-{f}" for f in COMPARISON_KEY if f not in ("metric", "k", "aggregation", "target_denominator",
                                                               "eval_score_block", "eval_batch", "eval_device_class")}
    rows = rows_from_result(res, base, method="PF", seed=2026, selection_rule="PRACTICAL_BEST", efe=6.0)
    assert {"eval_score_block", "eval_batch", "eval_device_class"} <= set(COMPARISON_KEY)
    assert all(r["eval_score_block"] == 158486 and r["eval_batch"] == 1024 and r["eval_device_class"] == "CPU-FP32-DET"
               for r in rows)
    assert all(r["status"] == "REHEARSAL_NON_FROZEN_DEVICE" for r in rows)            # CPU is not the frozen A16 class
    assert all("n_target_tied" in r and "value_pessimistic" in r for r in rows)
    pess = art.pessimistic_ranks
    rk = m.rankable
    assert np.all(pess[rk] >= art.ranks[rk]) and np.all(pess[~rk] == -1)
    amended = fc.load(amendment=_amend(eval_batch=16))
    art16 = predict_adapter(a, _batches(ex, 16), m, config=amended, model_sha256="t")
    rows16 = rows_from_result(evaluate(m, art16, config=amended), base, method="PF", seed=2026)
    assert rows16[0]["eval_batch"] == 16 and rows16[0]["eval_batch_amendment_id"] == "AMEND-EVAL-BATCH-0001"
    assert comparison_status(rows[0], rows16[0]) == ("HISTORICAL_REFERENCE", ["eval_batch"])


def test_t3_stream_forms_e2e_rows_and_all_rows():
    cfg = frozen()
    a, ex, m0 = _tiny_case(n=40)
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
    assert_raises((AlignmentError, fc.FrozenConfigError), predict_adapter, a, mixed, m, config=cfg,
                  model_sha256="t")
    base = {f: f"x-{f}" for f in COMPARISON_KEY if f not in ("metric", "k", "aggregation", "target_denominator",
                                                               "eval_score_block", "eval_batch", "eval_device_class",
                                                               "eval_stream")}
    r_e2e = rows_from_result(evaluate(m, art, config=cfg), base, method="C", seed=2026)
    r_all = rows_from_result(evaluate(m, art_all, config=cfg), base, method="C", seed=2026)
    assert comparison_status(r_e2e[0], r_all[0]) == ("HISTORICAL_REFERENCE", ["eval_stream"])


def test_t3_kwargs_must_equal_the_frozen_values():
    """ "require it (or equal it)": a caller may pass the numerics only if they equal the frozen ones."""
    a, ex, m = _tiny_case()
    b = _batches(ex, 1024)
    predict_adapter(a, b, m, model_sha256="t", score_block=158486, class_chunk=32768, cache_bytes=1073741824)
    for kw in ({"score_block": 32768}, {"class_chunk": 1024}, {"cache_bytes": 0}, {"eval_batch": 512}):
        assert_raises(fc.FrozenConfigError, predict_adapter, a, b, m, model_sha256="t", **kw)
    art = predict_adapter(a, b, m, model_sha256="t")                                     # config defaults to load()
    evaluate(m, art)
    assert_raises(fc.FrozenConfigError, predict_scores, [], m, model_sha256="t", class_chunk=7)


def test_t3_v4train_call_pattern_compat():
    """Read-only replay of v4train.trainer._evaluate on a real Clean V2 family module (tiny K), with the v4train
    EvalConfig built from this loader and the v4train SyntheticEvalProvider (ALL_ROWS batches, censored rows fed)."""
    from v4models.build import build
    from v4models.schema import load_schema
    from v4train.evalconfig import EvalConfig
    from v4train.provider import SyntheticEvalProvider
    ec = EvalConfig.from_frozen(fc.load())
    assert (ec.score_block, ec.class_chunk, ec.eval_batch_size, ec.cache_bytes) == (158486, 32768, 1024, 1073741824)
    assert ec.eval_device_class == "A16-FP32-DET" and ec.lock_sha256 == fc.load().lock_sha256
    schema = load_schema()
    for fam in ("GRU", "SASREC"):
        bm = build(fam, schema, seed=2026, K=24)
        evalp = SyntheticEvalProvider(90, 24, bm.widths, seed=5)
        art = predict_adapter(bm.adapter, evalp.batches(ec.eval_batch_size), evalp.manifest, model_sha256="theta0",
                              score_block=ec.score_block, class_chunk=ec.class_chunk, cache_bytes=ec.cache_bytes)
        assert art.header["eval_config"]["eval_stream"] == "ALL_ROWS"
        res = evaluate(evalp.manifest, art, evaluator_code_sha256="x")
        base = {f: f"x-{f}" for f in COMPARISON_KEY if f not in ("metric", "k", "aggregation", "target_denominator",
                                                                   "eval_score_block", "eval_batch",
                                                                   "eval_device_class", "eval_stream")}
        rows = rows_from_result(res, base, method="C", seed=2026, efe=0.5)
        assert rows and all(r["eval_stream"] == "ALL_ROWS" and r["eval_batch"] == 1024 for r in rows)


def test_t3_method_rows_require_the_adapter_path():
    m = random_manifest(n_users=10, seed=3)
    res = evaluate(m, artifact_from_scores(m, scores_for(m)), config=frozen())
    base = {f: f"x-{f}" for f in COMPARISON_KEY if f not in ("metric", "k", "aggregation", "target_denominator")}
    from v4eval.errors import RowKeyError
    for method in fc.METHODS:
        assert_raises(RowKeyError, rows_from_result, res, base, method=method, seed=2026)
    rows_from_result(res, base, method="TRANS", seed=2026)                               # a baseline may use it
