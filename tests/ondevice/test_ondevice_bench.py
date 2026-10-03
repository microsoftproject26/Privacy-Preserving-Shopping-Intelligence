"""The bench: pure helper logic (latency stats, parity, rows validation, the rank-array scatter) needs no
onnxruntime; only the onnxruntime-session tests do, and importorskip it."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from ondevice_testkit import K_SMALL, assert_raises, nc, tiny_scoring

from ppsi.ondevice.bench import (
    latency_stats,
    load_rows_npz,
    parity_stats,
    production_scores_batch,
    pytorch_scores_batch,
    ranks_from_session,
    real_rows_from_npz,
    rows_to_batches,
    run_pipeline,
    validate_rows_layout,
)
from ppsi.ondevice.scoring import example_batch

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_latency_stats_median_and_p95():
    seconds = np.array([0.001, 0.002, 0.002, 0.003, 0.100])
    stats = latency_stats(seconds)
    assert stats["n_queries"] == 5
    assert stats["median_ms"] == pytest.approx(2.0, abs=1e-9)
    assert stats["p95_ms"] >= stats["median_ms"]


def test_latency_stats_refuses_empty():
    assert_raises(ValueError, latency_stats, np.array([]))


def test_parity_stats_identical_arrays():
    rng = np.random.default_rng(0)
    a = rng.normal(size=(10, 64))
    stats = parity_stats(a, a.copy(), k=20)
    assert stats["max_abs_delta"] == 0.0
    assert stats["top_k_identity_rate"] == 1.0


def test_parity_stats_is_order_insensitive_within_topk():
    a = np.array([[5.0, 4.0, 3.0, 2.0, 1.0]])
    b = np.array([[4.0, 5.0, 3.0, 2.0, 1.0]])          # top-2 set {0,1} in both, just reordered
    stats = parity_stats(a, b, k=2)
    assert stats["top_k_identity_rate"] == 1.0
    assert stats["max_abs_delta"] == pytest.approx(1.0)


def test_parity_stats_detects_a_changed_topk_member():
    a = np.array([[5.0, 4.0, 3.0, 2.0, 1.0]])
    b = np.array([[5.0, 1.0, 3.0, 2.0, 4.0]])           # rank-2 item swapped out of the top-2
    stats = parity_stats(a, b, k=2)
    assert stats["top_k_identity_rate"] == 0.0


def test_rows_to_batches_shapes():
    made = tiny_scoring("GRU", seed=1, k=K_SMALL)
    rows = rows_to_batches(made["built"], made["scoring"], 4, seed=1)
    assert len(rows) == 4
    for row in rows:
        assert row["item_tokens"].shape[0] == 1


def test_load_rows_npz_roundtrip(tmp_path):
    p = tmp_path / "rows.npz"
    np.savez(p, decision_id=np.array([1, 2]), target_class=np.array([0, 1]), manifest_row=np.array([3, 7]),
             item_tokens=np.zeros((2, 5), dtype=np.int64))
    data = load_rows_npz(p)
    assert list(data["manifest_row"]) == [3, 7]


def test_load_rows_npz_refuses_missing_required_arrays(tmp_path):
    p = tmp_path / "rows.npz"
    np.savez(p, decision_id=np.array([1]))
    assert_raises(ValueError, load_rows_npz, p)


# --------------------------------------------------------------------------------------- validate_rows_layout / real rows
def _valid_rows_data(built, scoring, n=3, seed=5):
    """A --rows-shaped dict built from real synthetic decisions (so padding/lengths are correct by construction)."""
    batch = example_batch(built, n=n, seed=seed, min_len=1)
    data = {k: batch[k].numpy() for k in scoring.input_keys}
    data["decision_id"] = np.arange(100, 100 + n, dtype=np.int64)
    data["target_class"] = batch["target_class"].numpy()
    data["manifest_row"] = np.arange(n, dtype=np.int64)
    return data


def test_validate_rows_layout_accepts_real_synthetic_rows():
    made = tiny_scoring("GRU", k=K_SMALL)
    data = _valid_rows_data(made["built"], made["scoring"])
    validate_rows_layout(data, made["scoring"].input_keys)   # must not raise


def test_validate_rows_layout_refuses_length_zero():
    made = tiny_scoring("GRU", k=K_SMALL)
    data = _valid_rows_data(made["built"], made["scoring"])
    data["lengths"][0] = 0
    assert_raises(ValueError, validate_rows_layout, data, made["scoring"].input_keys)


def test_validate_rows_layout_refuses_broken_padding():
    made = tiny_scoring("GRU", k=K_SMALL)
    data = _valid_rows_data(made["built"], made["scoring"])
    data["attention_mask"][0, 0] = 1 - data["attention_mask"][0, 0]    # flip one bit: no longer a contiguous prefix
    assert_raises(ValueError, validate_rows_layout, data, made["scoring"].input_keys)


def test_real_rows_from_npz_respects_limit_and_validates():
    made = tiny_scoring("GRU", k=K_SMALL)
    data = _valid_rows_data(made["built"], made["scoring"], n=5)
    rows = real_rows_from_npz(data, made["scoring"].input_keys, limit=2)
    assert len(rows) == 2
    assert rows[0]["item_tokens"].shape[0] == 1


def test_real_rows_from_npz_propagates_layout_refusal():
    made = tiny_scoring("GRU", k=K_SMALL)
    data = _valid_rows_data(made["built"], made["scoring"])
    data["lengths"][0] = 0
    assert_raises(ValueError, real_rows_from_npz, data, made["scoring"].input_keys, limit=10)


def test_real_rows_from_npz_refuses_missing_input_key():
    made = tiny_scoring("GRU", k=K_SMALL)
    data = _valid_rows_data(made["built"], made["scoring"])
    del data[made["scoring"].input_keys[-1]]
    assert_raises(ValueError, real_rows_from_npz, data, made["scoring"].input_keys, limit=10)


# --------------------------------------------------------------------------------------------- production parity
@pytest.mark.parametrize("family", ["GRU", "SASREC"])
def test_production_scores_batch_close_to_export_wrapper(family):
    """The export wrapper (gru_dense_query / SASRec padded_reference) and the production forward (packed GRU /
    SASRec unpadded) use different kernels but must agree to float tolerance."""
    made = tiny_scoring(family, seed=3, k=K_SMALL)
    scoring, built = made["scoring"], made["built"]
    rows = rows_to_batches(built, scoring, 4, seed=11)
    wrapper = pytorch_scores_batch(scoring, rows)
    production = production_scores_batch(built, rows)
    assert wrapper.shape == production.shape == (4, built.K)
    assert np.allclose(wrapper, production, atol=1e-3, rtol=1e-3)


def test_production_scores_batch_restores_sasrec_impl():
    made = tiny_scoring("SASREC", seed=3, k=K_SMALL)
    scoring, built = made["scoring"], made["built"]
    assert built.module.impl == "padded_reference"        # set by io.build_scoring_module for export
    rows = rows_to_batches(built, scoring, 2, seed=1)
    production_scores_batch(built, rows)
    assert built.module.impl == "padded_reference"         # restored, not left at "unpadded"


# --------------------------------------------------------------------------------------------- onnxruntime-only
def test_make_session_and_onnxruntime_info_record_fixed_config(tmp_path):
    pytest.importorskip("onnx", reason="onnx is not installed")
    pytest.importorskip("onnxruntime", reason="onnxruntime is not installed")
    from ppsi.ondevice.bench import make_session, onnxruntime_info
    from ppsi.ondevice.export import export_model
    from ppsi.ondevice.scoring import example_inputs
    from ppsi.seqrec.synthetic import L_MAX

    made = tiny_scoring("GRU", seed=1, k=K_SMALL)
    scoring, built = made["scoring"], made["built"]
    example = example_inputs(built, scoring, n=1, seed=1, min_len=L_MAX)
    rec = export_model(scoring, example, tmp_path / "m.onnx")
    session = make_session(rec["path"], intra_op_num_threads=1)
    assert session.get_session_options().intra_op_num_threads == 1

    info = onnxruntime_info(intra_op_num_threads=1)
    assert info["graph_optimization_level"] == "ORT_ENABLE_ALL"
    assert info["onnxruntime_version"]
    assert info["providers"] == ["CPUExecutionProvider"]


def test_rss_worker_subprocess_fails_loudly_on_a_missing_model(tmp_path):
    """The worker module is invocable via `-m` from the repository root and fails (non-zero exit, error on stderr)
    on a missing model instead of printing a partial result."""
    npz = tmp_path / "feed.npz"
    np.savez(npz, item_tokens=np.zeros((1, 1), dtype=np.int64))
    proc = subprocess.run([sys.executable, "-B", "-m", "ppsi.ondevice._rss_worker", "--onnx", "nope.onnx",
                           "--feed", str(npz), "--intra-threads", "1", "--warmup", "0"],
                          cwd=str(REPO_ROOT), capture_output=True, text=True, check=False)
    assert proc.returncode != 0
    assert proc.stderr and "peak_rss_bytes" not in proc.stdout


def test_full_pipeline_needs_onnxruntime(tmp_path):
    pytest.importorskip("onnx", reason="onnx is not installed")
    pytest.importorskip("onnxruntime", reason="onnxruntime is not installed")
    made = tiny_scoring("GRU", seed=2026, k=K_SMALL)
    torch.save(dict(made["built"].module.state_dict()), str(tmp_path / "ckpt.pt"))
    args = _args(tmp_path, rows=None, manifest_n=None)
    result = run_pipeline(args)
    assert result["K"] == K_SMALL
    assert result["latency_fp32_ms"]["n_queries"] == 8
    assert result["latency_fp32_ms"]["peak_rss_bytes"] is not None       # measured in its own subprocess
    assert result["latency_int8_ms"]["peak_rss_bytes"] is not None
    assert result["parity_synthetic_fp32_vs_export_wrapper"]["max_abs_delta"] < 1e-3
    assert result["parity_synthetic_fp32_vs_production"]["max_abs_delta"] < 1e-2
    assert result["onnxruntime"]["onnxruntime_version"]
    assert result["onnxruntime"]["graph_optimization_level"] == "ORT_ENABLE_ALL"
    assert result["parity_real_fp32_vs_production"] is None              # no --rows given this run
    assert result["parity_real_int8_vs_production"] is None


def test_full_pipeline_with_rows_computes_real_parity_and_retention(tmp_path):
    pytest.importorskip("onnx", reason="onnx is not installed")
    pytest.importorskip("onnxruntime", reason="onnxruntime is not installed")
    made = tiny_scoring("GRU", seed=2026, k=K_SMALL)
    scoring, built = made["scoring"], made["built"]
    torch.save(dict(built.module.state_dict()), str(tmp_path / "ckpt.pt"))

    n_rows = 6
    rows_data = _valid_rows_data(built, scoring, n=n_rows, seed=77)
    rows_data["target_class"][2] = -1                      # an OOV row: must be skipped, not crash
    rows_path = tmp_path / "rows.npz"
    np.savez(rows_path, **rows_data)

    args = _args(tmp_path, rows=str(rows_path), manifest_n=20)
    result = run_pipeline(args)
    assert result["parity_real_fp32_vs_production"]["n_rows"] == n_rows
    assert result["parity_real_int8_vs_production"]["n_rows"] == n_rows
    assert set(result["retention"]) == {"onnx_fp32", "onnx_int8"}
    for tag, block in result["retention"].items():
        assert block["n_oov_skipped"] == 1 and block["n_rows_scored"] == n_rows - 1
        assert block["onnxruntime_version"] and block["intra_op_num_threads"] == 1
        assert block["graph_optimization_level"] == "ORT_ENABLE_ALL"
        assert len(block["ranks_sha256"]) == 64 and len(block["rows_npz_sha256"]) == 64
        ranks = np.load(Path(block["ranks_path"]))
        assert Path(block["ranks_path"]).name == f"{tag}.ranks_int32.npy"
        assert ranks.shape == (20,)
        assert ranks[2] == -1                               # OOV: credit 0
        assert (ranks[[0, 1, 3, 4, 5]] > 0).all()
        assert (ranks[n_rows:] == -1).all()


def _args(tmp_path, *, rows, manifest_n):
    return argparse.Namespace(ckpt=str(tmp_path / "ckpt.pt"), family="GRU", variant=None, out=str(tmp_path / "out"),
                              seed=2026, widths="15,15,12,4", catalogue=None, k=K_SMALL, n_queries=8, warmup=2,
                              intra_threads=1, rows=rows, manifest_n=manifest_n, tag="onnx_int8")


class _FakeSession:
    """A minimal onnxruntime.InferenceSession stand-in: `.run(None, feed)` returns the fixed score row for
    `feed["_row"]` (pure Python/numpy; no onnxruntime needed to test the rank-array scatter logic)."""

    def __init__(self, scores_by_row):
        self._scores = scores_by_row

    def run(self, output_names, feed):
        row = int(feed["_row"][0])
        return [self._scores[row]]


def test_ranks_scatter_into_manifest_positions():
    scores = {0: np.array([[9.0, 1.0, 1.0, 1.0, 1.0, 1.0]]),      # target 0 is the top score -> rank 1
             1: np.array([[1.0, 9.0, 9.0, 9.0, 9.0, 9.0]])}       # target 0 is strictly below every other -> rank 6
    session = _FakeSession(scores)
    rows = {"_row": np.array([0, 1]), "decision_id": np.array([100, 101]), "target_class": np.array([0, 0]),
           "manifest_row": np.array([2, 5])}
    ranks = ranks_from_session(session, rows, ["_row"], manifest_n=8)[0]
    assert ranks.shape == (8,)
    assert ranks[2] == 1
    assert ranks[5] == 6
    untouched = [i for i in range(8) if i not in (2, 5)]
    assert all(ranks[i] == -1 for i in untouched)


def test_ranks_skip_oov_rows_and_count_them():
    """A target_class < 0 row (evaluated but OOV) is left at -1 (credit 0) and counted, never scored / raised on."""
    scores = {0: np.array([[9.0, 1.0, 1.0, 1.0]]), 1: np.array([[1.0, 2.0, 3.0, 4.0]]),
             2: np.array([[1.0, 9.0, 1.0, 1.0]])}
    session = _FakeSession(scores)
    rows = {"_row": np.array([0, 1, 2]), "decision_id": np.array([100, 101, 102]),
           "target_class": np.array([0, -1, 1]), "manifest_row": np.array([1, 3, 6])}
    ranks, n_oov = ranks_from_session(session, rows, ["_row"], manifest_n=8)
    assert n_oov == 1
    assert ranks[1] == 1 and ranks[6] == 1 and ranks[3] == -1
    assert (ranks[[0, 2, 4, 5, 7]] == -1).all()


def test_ranks_oov_row_still_range_checks_manifest_row():
    session = _FakeSession({0: np.array([[1.0, 2.0]])})
    rows = {"_row": np.array([0]), "decision_id": np.array([100]), "target_class": np.array([-1]),
           "manifest_row": np.array([99])}
    assert_raises(ValueError, ranks_from_session, session, rows, ["_row"], 8)


def test_ranks_refuse_out_of_range_manifest_row():
    session = _FakeSession({0: np.array([[1.0, 2.0]])})
    rows = {"_row": np.array([0]), "decision_id": np.array([100]), "target_class": np.array([0]),
           "manifest_row": np.array([99])}
    assert_raises(ValueError, ranks_from_session, session, rows, ["_row"], 8)


# --------------------------------------------------------------------------------------------- negative controls
@nc("a set-based top-k identity check must NOT treat a genuinely different top-k member as identical")
def test_nc_parity_stats_would_ignore_a_changed_topk_member():
    a = np.array([[5.0, 4.0, 3.0, 2.0, 1.0]])
    b = np.array([[5.0, 1.0, 3.0, 2.0, 4.0]])
    stats = parity_stats(a, b, k=2)
    assert stats["top_k_identity_rate"] == 1.0


@nc("ranks_from_session must default un-scored manifest rows to -1, not 0")
def test_nc_ranks_default_would_be_zero():
    session = _FakeSession({0: np.array([[2.0, 1.0]])})
    rows = {"_row": np.array([0]), "decision_id": np.array([100]), "target_class": np.array([0]),
           "manifest_row": np.array([0])}
    ranks = ranks_from_session(session, rows, ["_row"], manifest_n=4)[0]
    assert ranks[3] == 0


@nc("an OOV row (target_class -1) must be credited a rank, not skipped (the evaluator gives it credit 0)")
def test_nc_oov_row_would_get_a_rank():
    session = _FakeSession({0: np.array([[2.0, 1.0]])})
    rows = {"_row": np.array([0]), "decision_id": np.array([100]), "target_class": np.array([-1]),
           "manifest_row": np.array([0])}
    ranks, n_oov = ranks_from_session(session, rows, ["_row"], manifest_n=2)
    assert ranks[0] > 0 and n_oov == 0


@nc("validate_rows_layout must refuse a length-0 row, not silently accept it (the scoring clamp is defensive only)")
def test_nc_validate_rows_layout_would_accept_length_zero():
    made = tiny_scoring("GRU", k=K_SMALL)
    data = _valid_rows_data(made["built"], made["scoring"])
    data["lengths"][0] = 0
    try:
        validate_rows_layout(data, made["scoring"].input_keys)
        refused = False
    except ValueError:
        refused = True
    assert not refused, "validate_rows_layout correctly refused the length-0 row (this NC claims the opposite)"


@nc("validate_rows_layout must refuse padding that is not a contiguous prefix of lengths")
def test_nc_validate_rows_layout_would_accept_broken_padding():
    made = tiny_scoring("GRU", k=K_SMALL)
    data = _valid_rows_data(made["built"], made["scoring"])
    data["attention_mask"][0, 0] = 1 - data["attention_mask"][0, 0]
    try:
        validate_rows_layout(data, made["scoring"].input_keys)
        refused = False
    except ValueError:
        refused = True
    assert not refused, "validate_rows_layout correctly refused the broken padding (this NC claims the opposite)"

