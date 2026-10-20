"""Evaluation batches follow the frozen config: exactly eval_batch rows except an unpadded tail, manifest order over the
registered stream (never hard-coded), prefetch byte-identical, and the frozen predict path runs on them."""
from __future__ import annotations

import threading
from types import SimpleNamespace

import numpy as np
import pytest

import v4int.providers as P
from intkit import REAL_DS, assert_raises, batches_equal, nc
from v4eval import frozen_config as fc
from v4eval.predict import stream_rows
from v4int.providers import ProviderError, StoreEvalProvider, ordered_prefetch, resolve_eval_stream

Q = lambda *_: None  # noqa: E731


def test_batches_exact_size_unpadded_tail_manifest_order(providers):
    _, ep = providers
    fz = fc.load()
    assert ep.eval_batch == fz.eval_batch == 1024
    bs = list(ep.batches(fz.eval_batch))
    sizes = [int(b["decision_id"].shape[0]) for b in bs]
    assert len(sizes) >= 2 and all(s == 1024 for s in sizes[:-1]) and 0 < sizes[-1] < 1024
    ids = np.concatenate([b["decision_id"].numpy() for b in bs])
    st = stream_rows(ep.manifest)
    assert np.array_equal(ids, ep.manifest.decision_id[st])                  # the E2E stream, manifest order
    assert ep.n_stream_rows < ep.manifest.n                                   # censored rows are not fed
    tg = np.concatenate([b["target_class"].numpy() for b in bs])
    assert np.array_equal(tg, ep.manifest.target_class[st])


def test_batch_size_must_equal_the_frozen_eval_batch(providers):
    _, ep = providers
    for bad in (512, 1023, 1025, 2048):
        assert_raises(ProviderError, ep.batches, bad)


def test_stream_comes_from_the_registered_source(tmp_path):
    """R2 (G1 combined gate, INT-02 item 4): v4eval.frozen_config.load() now carries `eval_stream` itself
    ("EVAL_ROWS_ONLY"), so resolve_eval_stream reads it straight from the frozen config (no PREG1_AMENDMENT_01 file
    read); the file-reading fallback below is exercised through a bare object that carries no eval_stream attribute
    (a pre-R2 frozen config stand-in), proving the fallback path itself still works."""
    fz = fc.load()
    s, src = resolve_eval_stream(fz)
    assert (s, src["value"]) == ("E2E_ROWS", "EVAL_ROWS_ONLY") and src["source"] == "v4eval.frozen_config.load().eval_stream"
    assert len(src["lock_sha256"]) == 64
    for v, want in (("EVAL_ROWS_ONLY", "E2E_ROWS"), ("E2E_ROWS", "E2E_ROWS"), ("ALL_ROWS", "ALL_ROWS")):
        got, src2 = resolve_eval_stream(SimpleNamespace(eval_stream=v, lock_sha256="L"))
        assert got == want and src2["source"].endswith(".eval_stream")
    assert_raises(ProviderError, resolve_eval_stream, SimpleNamespace(eval_stream="EVERYTHING", lock_sha256="L"))
    no_stream = SimpleNamespace(lock_sha256="L")                          # a pre-R2 frozen config: no eval_stream attr
    assert_raises(ProviderError, resolve_eval_stream, no_stream, tmp_path / "missing.md")
    two = tmp_path / "two.md"
    two.write_text('`eval_stream = "EVAL_ROWS_ONLY"` ... `eval_stream = "ALL_ROWS"`', encoding="utf-8")
    assert_raises(ProviderError, resolve_eval_stream, no_stream, two)
    ok = tmp_path / "one.md"
    ok.write_text('`eval_stream = "EVAL_ROWS_ONLY"`', encoding="utf-8")
    got3, src3 = resolve_eval_stream(no_stream, ok)
    assert got3 == "E2E_ROWS" and src3["value"] == "EVAL_ROWS_ONLY" and "carries no eval_stream" in src3["source"] \
        and len(src3["file_sha256"]) == 64


def test_all_rows_stream_refused_by_the_store(real_env, monkeypatch):
    monkeypatch.setattr(P, "resolve_eval_stream", lambda fz, p=None: ("ALL_ROWS", {"source": "test"}))
    assert_raises(ProviderError, StoreEvalProvider, "M06_CALIB_VALIDATION_WARM", history_policy="SESSION_L50", paths=real_env["paths"],
                  ds_root=REAL_DS, buckets=[0], log=Q)


def test_prefetch_is_byte_identical_and_ordered(providers):
    _, ep = providers
    old = ep.prefetch
    try:
        ep.prefetch = 0
        plain = list(ep.batches(ep.eval_batch))
        ep.prefetch = 2
        pre = list(ep.batches(ep.eval_batch))
    finally:
        ep.prefetch = old
    assert len(plain) == len(pre) and all(batches_equal(a, b)[0] for a, b in zip(plain, pre))


def test_prefetch_propagates_errors_and_stops_its_thread():
    def fn(i):
        if i == 3:
            raise ValueError("boom")
        return i
    got = []
    with pytest.raises(ValueError):
        for x in ordered_prefetch(fn, range(10), 2):
            got.append(x)
    assert got == [0, 1, 2]
    g = ordered_prefetch(lambda i: i, range(1000), 2)
    assert next(g) == 0
    g.close()                                                                # abandoned early
    assert not [t for t in threading.enumerate() if t.name == "v4int-eval-prefetch"]


def test_frozen_predict_path_runs_on_provider_batches(providers):
    """The trainer's evaluation call pattern on CPU (the real GRU at K = 158,486, seed 2026): the artifact records the
    E2E_ROWS stream, censored and OOV rows keep rank -1, and evaluate() accepts it."""
    from v4eval import evaluator_code_sha256
    from v4eval.evaluate import evaluate
    from v4eval.predict import predict_adapter
    from v4models.build import build
    from v4models.schema import load_schema
    _, ep = providers
    fz = fc.load()
    bm = build("GRU", load_schema(), 2026, "cpu", v1_root=REAL_DS)
    fc.apply_runtime_flags(fz)
    art = predict_adapter(bm.adapter, ep.batches(fz.eval_batch), ep.manifest, config=fz, model_sha256=bm.init_sha256)
    assert art.header["eval_config"]["eval_stream"] == "E2E_ROWS"
    rk = ep.manifest.rankable
    assert np.all(art.ranks[~rk] == -1) and np.all(art.ranks[rk] >= 1)
    res = evaluate(ep.manifest, art, config=fz, evaluator_code_sha256=evaluator_code_sha256())
    assert res["E2E"]["n_decisions"] == ep.n_stream_rows


@nc("feeding every manifest row (censored rows included) is not the registered E2E stream")
def test_nc_all_rows_are_not_the_stream(providers):
    _, ep = providers
    assert ep.manifest.n == ep.n_stream_rows
