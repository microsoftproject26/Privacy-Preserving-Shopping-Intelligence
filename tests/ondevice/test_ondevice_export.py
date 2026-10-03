"""Deterministic ONNX opset-17 export. Needs the `onnx` package (the module importorskips it) but not
onnxruntime: `onnx.checker` alone verifies the graph."""
from __future__ import annotations

import pytest

onnx = pytest.importorskip("onnx", reason="onnx is not installed")

from ondevice_testkit import K_SMALL, nc, tiny_scoring

from ppsi.ondevice.export import OPSET, export_model
from ppsi.ondevice.scoring import example_inputs
from ppsi.seqrec.synthetic import L_MAX


@pytest.mark.parametrize("family", ["GRU", "SASREC"])
def test_export_is_deterministic_and_well_formed(tmp_path, family):
    made = tiny_scoring(family, seed=3, k=K_SMALL)
    scoring, built = made["scoring"], made["built"]
    example = example_inputs(built, scoring, n=1, seed=3, min_len=L_MAX)     # full-length trace
    rec = export_model(scoring, example, tmp_path / "model_fp32.onnx")

    assert rec["opset"] == OPSET
    assert rec["K"] == K_SMALL
    assert rec["sha256"] == rec["deterministic_reexport_sha256"]     # the re-export is byte-identical
    assert (tmp_path / "model_fp32.onnx").is_file()
    assert rec["declared_input_keys"] == list(scoring.input_keys)
    # the graph's own retained inputs are a SUBSET of the nominal keys (a key used only for a static-shape/validation
    # check, never real op data -- attention_mask/position_ids here -- is legitimately pruned by tracing; item_tokens
    # and lengths are load-bearing for every family and must always survive)
    assert set(rec["input_names"]) <= set(scoring.input_keys)
    assert {"item_tokens", "lengths"} <= set(rec["input_names"])

    m = onnx.load(rec["path"])
    onnx.checker.check_model(m)
    assert [i.name for i in m.graph.input] == rec["input_names"]
    assert [o.name for o in m.graph.output] == ["scores"]
    for inp in m.graph.input:                                       # fixed batch 1: every axis is a static int, none symbolic
        dim0 = inp.type.tensor_type.shape.dim[0]
        assert dim0.dim_value == 1 and not dim0.dim_param


def test_export_manifest_sha256_matches_file(tmp_path):
    made = tiny_scoring("GRU", seed=42, k=K_SMALL)
    scoring, built = made["scoring"], made["built"]
    example = example_inputs(built, scoring, n=1, seed=42, min_len=L_MAX)
    rec = export_model(scoring, example, tmp_path / "m.onnx")
    import hashlib
    assert hashlib.sha256((tmp_path / "m.onnx").read_bytes()).hexdigest() == rec["sha256"]


# --------------------------------------------------------------------------------------------- negative control
@nc("the exported graph must have a STATIC batch dim of 1, never a dynamic/symbolic one")
def test_nc_export_would_allow_a_dynamic_batch_dim(tmp_path):
    made = tiny_scoring("SASREC", seed=8, k=K_SMALL)
    scoring, built = made["scoring"], made["built"]
    example = example_inputs(built, scoring, n=1, seed=8, min_len=L_MAX)
    rec = export_model(scoring, example, tmp_path / "model_fp32.onnx")
    m = onnx.load(rec["path"])
    dim0 = m.graph.input[0].type.tensor_type.shape.dim[0]
    assert bool(dim0.dim_param)          # true only for a dynamic/symbolic batch dim; this export has none
