"""The GPU determinism receipt's code paths, dry-run on the local CPU (small K) before the server run; plus the
on-device fingerprint's bit sensitivity.

Proves: every receipt check (init hashes + device round trip, T34 with 2 dedicated worker processes and a shard
rotation, LO over clients in forward vs reverse order, checkpoint -> fresh process -> resume) runs end to end and
yields bitwise verdicts; a strict-determinism failure is recorded verbatim and re-run in warn-only mode; the
fingerprint changes under a single-bit flip or an element swap. Does not prove: GPU determinism (the server receipt).
"""
from __future__ import annotations

import contextlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import torch

import fl.gpu_receipt as gr
from fl.numerics import fingerprint
from fl_testkit import assert_raises, nc

CODE = Path(__file__).resolve().parents[1]
WORK = Path(__file__).resolve().parent / "_tmp_gpu_dry"


@pytest.fixture(scope="module")
def work():
    shutil.rmtree(WORK, ignore_errors=True)
    WORK.mkdir(parents=True)
    yield WORK
    shutil.rmtree(WORK, ignore_errors=True)


def _cli(*args):
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    p = subprocess.run([sys.executable, "-B", "-m", "fl.gpu_receipt", *args], cwd=str(CODE), env=env,
                       capture_output=True, text=True, timeout=900)
    assert p.returncode == 0, p.stderr[-3000:]


def test_receipt_dry_run_end_to_end(work):
    K = "64"
    _cli("init", "--K", K, "--device", "cpu", "--out", str(work / "init.json"))
    for fam in ("GRU", "SASREC"):
        _cli("t34", "--family", fam, "--K", K, "--devices", "cpu,cpu", "--out", str(work / f"t34_{fam}.json"))
        _cli("lo", "--family", fam, "--K", K, "--device", "cpu", "--order", "forward", "--n", "20",
             "--out", str(work / f"lo_{fam}_A.json"))
        _cli("lo", "--family", fam, "--K", K, "--device", "cpu", "--order", "reverse", "--n", "20",
             "--out", str(work / f"lo_{fam}_B.json"))
    for ph in ("full", "first", "second"):
        _cli("resume", "--family", "GRU", "--K", K, "--devices", "cpu,cpu", "--phase", ph,
             "--ckpt-root", str(work / "ckpt"), "--out", str(work / f"resume_{ph}.json"))
    _cli("summarize", "--dir", str(work), "--out", str(work / "GPU_DETERMINISM_RECEIPT.json"))
    rec = json.loads((work / "GPU_DETERMINISM_RECEIPT.json").read_text())
    assert rec["all_bitwise"], rec["checks"]
    assert len([k for k in rec["checks"] if k.startswith("t34_")]) == 8
    assert {"lo_GRU_1024_bitwise_A_vs_B", "lo_SASREC_1024_bitwise_A_vs_B", "resume_GRU_bitwise",
            "init_hash_round_trip"} <= set(rec["checks"])
    t34 = json.loads((work / "t34_SASREC.json").read_text())
    assert t34["mode"] == "strict" and len(t34["worker_pids"]) == 2
    rot = t34["methods"]["PF"]["runs"]
    assert rot[0]["rotation"] == 0 and rot[2]["rotation"] == 1
    second = json.loads((work / "resume_second.json").read_text())
    assert second["resumed_from_cursor"] == 2 and second["files_after_cleanup"] == []
    assert second["removal_log"][0]["sha256"] == second["checkpoint_sha256"]


def test_strict_failure_recorded_verbatim_then_warn_only():
    calls = []

    def fn(strict):
        calls.append(strict)
        if strict:
            raise RuntimeError("index_add_cuda_ does not have a deterministic implementation, but you set "
                               "'torch.use_deterministic_algorithms(True)'")
        return {"ok": 1}
    res = gr._with_fallback(fn)
    assert calls == [True, False] and res["mode"] == "warn_only_fallback" and res["ok"] == 1
    assert res["strict_error_verbatim"].startswith("RuntimeError: index_add_cuda_ does not have")

    def other(strict):
        raise RuntimeError("out of memory")
    assert_raises(RuntimeError, gr._with_fallback, other)


def test_invisible_cuda_device_refused_with_cause():
    """FL-02 finding: snap_v1/reuse/train.py setdefaults CUDA_VISIBLE_DEVICES='0' on import."""
    from fl.pool import require_visible
    try:
        require_visible("cuda:%d" % (torch.cuda.device_count() + 1))
    except RuntimeError as e:
        assert "CUDA_VISIBLE_DEVICES" in str(e) and "setdefault" in str(e)
    else:
        raise AssertionError("an invisible CUDA device must be refused")


def check_fingerprint(fp):
    torch.manual_seed(0)
    s = {"a": torch.randn(1000, 7), "b": torch.arange(10, dtype=torch.long), "c": torch.randn(3)}
    base = fp(s)
    assert fp({k: v.clone() for k, v in s.items()}) == base
    flip = {k: v.clone() for k, v in s.items()}
    flip["a"].view(torch.int32)[123, 4] ^= 1                      # one low mantissa bit
    assert fp(flip) != base, "a single-bit flip must change the fingerprint"
    swap = {k: v.clone() for k, v in s.items()}
    swap["a"][[5, 9]] = swap["a"][[9, 5]]
    assert fp(swap) != base, "swapping two rows must change the fingerprint"
    long_flip = {k: v.clone() for k, v in s.items()}
    long_flip["b"][3] += 1 << 40
    assert fp(long_flip) != base


def test_fingerprint_bit_and_position_sensitive():
    check_fingerprint(fingerprint)


@nc("an order-insensitive float-sum fingerprint")
def test_nc_fingerprint_sum_only():
    check_fingerprint(lambda s: repr(sorted((k, float(v.double().sum())) for k, v in s.items())))
