"""OPS-B13-P2 / DL 368: INT8 pass 2 (`v4dev.bench --latency-only`). onnxruntime is replaced by a fake session so the whole
flow (sha assertions, protocol match, write-once output, no export, no accuracy) is tested without server-only packages."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pytest
import torch

import v4dev.bench as B
from dev_testkit import K_SMALL, assert_raises, nc, tiny_scoring

SENTINEL = 987654.321          # a pass-1 latency value: it must never reach LATENCY_PASS2.json
N_Q, WARM = 6, 2


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


class _In:
    def __init__(self, name):
        self.name = name


class _FakeSession:
    instances = []

    def __init__(self, path, hook=None):
        self.path, self.calls, self.hook = Path(path), 0, hook
        _FakeSession.instances.append(self)
        self.names = list(_FakeSession.names)

    def get_inputs(self):
        return [_In(n) for n in self.names]

    def run(self, outs, feed):
        self.calls += 1
        assert set(feed) == set(self.names)
        if self.hook:
            self.hook(self)
        return [None]


_FakeSession.names = []


@pytest.fixture
def pinned(tmp_path, monkeypatch):
    made = tiny_scoring("GRU", seed=2026, k=K_SMALL)
    ck = tmp_path / "ckpt.pt"
    torch.save(dict(made["built"].module.state_dict()), str(ck))
    d = tmp_path / "ONNX_P1"
    d.mkdir()
    blobs = {"model_fp32.onnx": b"fp32-bytes", "model_int8.onnx": b"int8-bytes", "rows.npz": b"rows-bytes",
             "rows.meta.json": b'{"manifest_n": 20}'}
    for n, b in blobs.items():
        (d / n).write_bytes(b)
    rows_sha, meta_sha = _sha(blobs["rows.npz"]), _sha(blobs["rows.meta.json"])
    ret = {t: {"tag": t, "rows_npz_sha256": rows_sha, "rows_meta_sha256": meta_sha} for t in ("onnx_fp32", "onnx_int8")}
    rec = {"family": "GRU", "variant": made["built"].variant, "K": int(made["built"].K), "seed": 2026,
           "intra_op_num_threads": 1, "checkpoint_sha256": _sha(ck.read_bytes()),
           "model_sha256": {"fp32": _sha(blobs["model_fp32.onnx"]), "int8": _sha(blobs["model_int8.onnx"])},
           "latency_fp32_ms": {"median_ms": SENTINEL, "p95_ms": SENTINEL, "n_queries": N_Q, "warmup": WARM},
           "latency_int8_ms": {"median_ms": SENTINEL, "p95_ms": SENTINEL, "n_queries": N_Q, "warmup": WARM},
           "retention": ret, "rows_first_users": 16384}
    (d / "BENCH_RESULT.json").write_text(json.dumps(rec), encoding="utf-8")
    _FakeSession.instances.clear()
    _FakeSession.names = list(made["scoring"].input_keys)
    monkeypatch.setattr(B, "make_session", lambda p, intra_op_num_threads=1: _FakeSession(p))
    monkeypatch.setattr(B, "onnxruntime_info", lambda n=1: {"onnxruntime_version": "fake", "intra_op_num_threads": n})
    calls = {"export": 0, "quant": 0, "pipeline": 0}

    def _boom(name):
        def f(*a, **k):
            calls[name] += 1
            raise AssertionError(f"{name} must not run in --latency-only")
        return f
    monkeypatch.setattr(B, "export_model", _boom("export"))
    monkeypatch.setattr(B, "quantize_int8", _boom("quant"))
    monkeypatch.setattr(B, "run_pipeline", _boom("pipeline"))
    args = argparse.Namespace(ckpt=str(ck), family="GRU", variant=None, out=str(d), seed=2026, v1_root=None,
                              catalogue=None, k=K_SMALL, n_queries=N_Q, warmup=WARM, intra_threads=1, rows=None,
                              manifest_n=None, tag="onnx_int8", latency_only=True)
    return args, d, calls, rec


def _snapshot(d: Path) -> dict:
    return {p.name: _sha(p.read_bytes()) for p in sorted(d.iterdir()) if p.is_file()}


def test_happy_path_writes_only_latency_pass2(pinned):
    args, d, calls, rec = pinned
    before = _snapshot(d)
    res = B.run_latency_only(args)
    after = _snapshot(d)
    assert set(after) - set(before) == {"LATENCY_PASS2.json"}
    assert {k: v for k, v in after.items() if k != "LATENCY_PASS2.json"} == before        # every pass-1 file untouched
    assert calls == {"export": 0, "quant": 0, "pipeline": 0}
    out = json.loads((d / "LATENCY_PASS2.json").read_text())
    assert out["mode"] == "latency_only_pass2" and out["no_export_no_accuracy_no_rss"] is True
    assert out["latency_fp32_ms"]["n_queries"] == N_Q and out["latency_int8_ms"]["n_queries"] == N_Q
    assert out["latency_fp32_ms"]["warmup"] == WARM and out["intra_op_num_threads"] == 1
    assert out["asserted_sha256"]["int8"] == rec["model_sha256"]["int8"]
    assert out["pass1_record"]["pass1_record_file_sha256"] == before["BENCH_RESULT.json"]
    for k in ("retention", "parity_synthetic_fp32_vs_production", "parity_real_int8_vs_production"):
        assert k not in out                                                  # no accuracy, no parity
    assert "peak_rss_bytes" not in out["latency_fp32_ms"]                    # no RSS
    assert str(SENTINEL) not in (d / "LATENCY_PASS2.json").read_text()        # pass-1 latency is never copied
    assert len(res["result_sha256"]) == 64 and "result_sha256" not in out
    # both sessions timed the same n queries + warmup calls
    assert [s.calls for s in _FakeSession.instances] == [N_Q + WARM, N_Q + WARM]


def test_refuses_each_sha_mismatch(pinned):
    args, d, calls, rec = pinned
    for name in ("model_int8.onnx", "model_fp32.onnx", "rows.npz", "rows.meta.json"):
        orig = (d / name).read_bytes()
        (d / name).write_bytes(orig + b"x")
        assert_raises(SystemExit, B.run_latency_only, args)
        assert not (d / "LATENCY_PASS2.json").exists()
        (d / name).write_bytes(orig)
    assert not _FakeSession.instances                                       # refused before any session was built
    B.run_latency_only(args)                                                # restored -> accepted


def test_refuses_missing_or_incomplete_pass1_record(pinned):
    args, d, calls, rec = pinned
    p = d / "BENCH_RESULT.json"
    for mutate in (lambda r: r.pop("retention"), lambda r: r.update(retention={}), lambda r: r.pop("model_sha256"),
                   lambda r: r["model_sha256"].update(int8="")):
        r = json.loads(json.dumps(rec))
        mutate(r)
        p.write_text(json.dumps(r))
        assert_raises(SystemExit, B.run_latency_only, args)
    p.unlink()
    assert_raises(SystemExit, B.run_latency_only, args)


@pytest.mark.parametrize("field,value", [("n_queries", N_Q + 1), ("warmup", WARM + 1), ("intra_threads", 2), ("seed", 7)])
def test_refuses_a_different_timing_protocol(pinned, field, value):
    args, d, calls, rec = pinned
    setattr(args, field, value)
    assert_raises(SystemExit, B.run_latency_only, args)
    assert not (d / "LATENCY_PASS2.json").exists()


def test_refuses_other_checkpoint(pinned, tmp_path):
    args, d, calls, rec = pinned
    other = tmp_path / "other.pt"
    other.write_bytes(b"not the checkpoint")
    args.ckpt = str(other)
    assert_raises(SystemExit, B.run_latency_only, args)


def test_latency_pass2_is_write_once(pinned):
    args, d, calls, rec = pinned
    B.run_latency_only(args)
    first = (d / "LATENCY_PASS2.json").read_bytes()
    assert_raises(SystemExit, B.run_latency_only, args)
    assert (d / "LATENCY_PASS2.json").read_bytes() == first


def test_pinned_file_changing_during_timing_refuses_and_writes_nothing(pinned, monkeypatch):
    args, d, calls, rec = pinned
    monkeypatch.setattr(B, "make_session", lambda p, intra_op_num_threads=1: _FakeSession(
        p, hook=lambda s: (d / "model_int8.onnx").write_bytes(b"tampered")))
    assert_raises(SystemExit, B.run_latency_only, args)
    assert not (d / "LATENCY_PASS2.json").exists()


def test_main_dispatches_latency_only_without_the_pipeline(pinned, capsys):
    args, d, calls, rec = pinned
    argv = ["--latency-only", "--ckpt", args.ckpt, "--family", "GRU", "--out", str(d), "--k", str(K_SMALL),
            "--n-queries", str(N_Q), "--warmup", str(WARM)]
    assert B.main(argv) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["mode"] == "latency_only_pass2" and (d / "LATENCY_PASS2.json").is_file()
    assert calls["pipeline"] == 0


def test_cli_flag_is_registered_and_default_off():
    src = Path(B.__file__).read_text(encoding="utf-8")
    assert '"--latency-only"' in src and "run_pipeline(args)" in src


def test_pass1_record_reads_identity_only(pinned):
    args, d, calls, rec = pinned
    r = B.read_pass1_record(d)
    assert SENTINEL not in [v for v in r.values() if isinstance(v, float)]
    assert r["n_queries"] == N_Q and r["warmup"] == WARM and r["rows_npz_sha256"] == rec["retention"]["onnx_int8"]["rows_npz_sha256"]



def test_end_to_end_pass1_then_pass2_on_real_onnx(tmp_path):
    """Tiny synthetic fixture only (K=64, 8 queries): real export + quantise in pass 1, then pass 2 on that dir."""
    pytest.importorskip("onnx", reason="onnx is a server-only dependency (not in the local dev venv)")
    pytest.importorskip("onnxruntime", reason="onnxruntime is a server-only dependency (not in the local dev venv)")
    import numpy as np
    from test_bench import _valid_rows_data
    made = tiny_scoring("GRU", seed=2026, k=K_SMALL)
    torch.save(dict(made["built"].module.state_dict()), str(tmp_path / "ckpt.pt"))
    out = tmp_path / "ONNX_P1"
    out.mkdir()
    np.savez(out / "rows.npz", **_valid_rows_data(made["built"], made["scoring"], n=6, seed=77))
    a1 = argparse.Namespace(ckpt=str(tmp_path / "ckpt.pt"), family="GRU", variant=None, out=str(out), seed=2026,
                            v1_root=None, catalogue=None, k=K_SMALL, n_queries=8, warmup=2, intra_threads=1,
                            rows=str(out / "rows.npz"), manifest_n=20, tag="onnx_int8")
    B.run_pipeline(a1)
    before = _snapshot(out)
    a2 = argparse.Namespace(**{**vars(a1), "rows": None, "manifest_n": None, "latency_only": True})
    res = B.run_latency_only(a2)
    after = _snapshot(out)
    assert set(after) - set(before) == {"LATENCY_PASS2.json"}
    assert {k: v for k, v in after.items() if k != "LATENCY_PASS2.json"} == before
    assert res["latency_fp32_ms"]["n_queries"] == 8 and res["latency_int8_ms"]["median_ms"] > 0
    assert res["asserted_sha256"]["int8"] == json.loads((out / "BENCH_RESULT.json").read_text())["model_sha256"]["int8"]

# --------------------------------------------------------------------------------------------- negative controls
def _accepts(args):
    """A refusal (SystemExit) is turned into the AssertionError an NC must fail with."""
    try:
        return B.run_latency_only(args)
    except SystemExit as e:
        raise AssertionError(f"refused as it should: {e}") from None


@nc("a model sha mismatch must be refused, not accepted")
def test_nc_sha_mismatch_would_be_accepted(pinned):
    args, d, calls, rec = pinned
    (d / "model_int8.onnx").write_bytes(b"other")
    _accepts(args)


@nc("an existing LATENCY_PASS2.json must not be overwritten")
def test_nc_existing_result_would_be_overwritten(pinned):
    args, d, calls, rec = pinned
    (d / "LATENCY_PASS2.json").write_text("{}")
    _accepts(args)


@nc("pass-1 latency must NOT be copied into pass 2")
def test_nc_pass1_latency_would_be_copied(pinned):
    args, d, calls, rec = pinned
    _accepts(args)
    assert str(SENTINEL) in (d / "LATENCY_PASS2.json").read_text()


@nc("pass 2 must not export or quantise")
def test_nc_pass2_would_reexport(pinned):
    args, d, calls, rec = pinned
    _accepts(args)
    assert calls["export"] > 0 or calls["quant"] > 0


@nc("a different warmup must be refused")
def test_nc_different_warmup_would_be_accepted(pinned):
    args, d, calls, rec = pinned
    args.warmup = WARM + 5
    _accepts(args)


@nc("pass 2 must not write accuracy / retention / RSS fields")
def test_nc_pass2_would_carry_accuracy(pinned):
    args, d, calls, rec = pinned
    _accepts(args)
    assert "retention" in json.loads((d / "LATENCY_PASS2.json").read_text())
