"""INT8 dynamic quantisation. Needs onnxruntime (the module importorskips it), plus `onnx` to produce the FP32
model it quantises."""
from __future__ import annotations

import pytest

pytest.importorskip("onnx", reason="onnx is not installed")
pytest.importorskip("onnxruntime", reason="onnxruntime is not installed")

from ondevice_testkit import K_SMALL, assert_raises, nc, tiny_scoring

from ppsi.ondevice.export import export_model
from ppsi.ondevice.quantize import quantize_int8
from ppsi.ondevice.scoring import example_inputs
from ppsi.seqrec.synthetic import L_MAX


def test_quantize_int8_produces_a_smaller_hashed_artifact(tmp_path):
    made = tiny_scoring("GRU", seed=6, k=K_SMALL)
    scoring, built = made["scoring"], made["built"]
    example = example_inputs(built, scoring, n=1, seed=6, min_len=L_MAX)
    export_model(scoring, example, tmp_path / "model_fp32.onnx")

    rec = quantize_int8(tmp_path / "model_fp32.onnx", tmp_path / "model_int8.onnx")
    assert (tmp_path / "model_int8.onnx").is_file()
    assert rec["fp32"]["sha256"] and rec["int8"]["sha256"]
    assert rec["fp32"]["sha256"] != rec["int8"]["sha256"]
    assert rec["int8"]["bytes"] < rec["fp32"]["bytes"]


def test_quantize_int8_refuses_missing_fp32(tmp_path):
    assert_raises(FileNotFoundError, quantize_int8, tmp_path / "nope.onnx", tmp_path / "out.onnx")


# --------------------------------------------------------------------------------------------- negative control
@nc("quantize_int8 must refuse a missing FP32 model, not silently produce nothing")
def test_nc_quantize_would_accept_a_missing_fp32_model(tmp_path):
    try:
        quantize_int8(tmp_path / "nope.onnx", tmp_path / "out.onnx")
        refused = False
    except FileNotFoundError:
        refused = True
    assert not refused
