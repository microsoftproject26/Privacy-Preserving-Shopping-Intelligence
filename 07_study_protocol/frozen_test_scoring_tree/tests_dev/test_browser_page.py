"""DEV-03 tests: static checks on `v4dev/browser/` (files exist, the onnxruntime-web version is pinned, only
cdn.jsdelivr.net is ever referenced) plus a round-trip of `v4dev.browser_queries` on synthetic fixtures (pattern of
code/tests_dev/test_bench.py's `_valid_rows_data`; no onnx/onnxruntime import anywhere in this file)."""
from __future__ import annotations

import json
import re
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import pytest
import torch

from dev_testkit import K_SMALL, assert_raises, nc, tiny_scoring
from v4dev.browser_queries import (META_KEYS, BrowserQueriesError, build_provenance, build_queries, main,
                                   reference_top20, resolve_n_rows, select_input_keys)
from v4dev.scoring import example_batch

BROWSER_DIR = Path(__file__).resolve().parents[1] / "v4dev" / "browser"
INDEX_HTML = BROWSER_DIR / "index.html"
BENCH_JS = BROWSER_DIR / "bench.js"
README = BROWSER_DIR / "README.md"

URL_RE = re.compile(r"https?://[^\s\"'<>]+")


# ------------------------------------------------------------------------------------------------- static checks
def test_browser_files_exist():
    for p in (INDEX_HTML, BENCH_JS, README):
        assert p.is_file(), f"missing {p}"


def test_onnxruntime_web_version_is_pinned_not_latest():
    html = INDEX_HTML.read_text(encoding="utf-8")
    m = re.search(r"onnxruntime-web@([^/\"']+)/dist/", html)
    assert m, "index.html must load onnxruntime-web with an explicit @version in its CDN URL"
    version = m.group(1)
    assert version not in ("latest", ""), f"onnxruntime-web version must be pinned, not {version!r}"
    assert re.match(r"^\d+\.\d+\.\d+", version), f"expected a semver-looking pin, got {version!r}"


def test_only_cdn_jsdelivr_host_is_referenced():
    html = INDEX_HTML.read_text(encoding="utf-8")
    js = BENCH_JS.read_text(encoding="utf-8")
    urls = URL_RE.findall(html) + URL_RE.findall(js)
    assert urls, "expected at least one https:// reference (the onnxruntime-web CDN tag)"
    hosts = {urlparse(u).netloc for u in urls}
    assert hosts == {"cdn.jsdelivr.net"}, f"unexpected external host(s) referenced: {hosts - {'cdn.jsdelivr.net'}}"


def test_bench_js_makes_no_network_calls_of_its_own():
    js = BENCH_JS.read_text(encoding="utf-8")
    assert not URL_RE.findall(js), "bench.js itself must reference no external URL (only index.html's <script> tag)"
    assert "fetch(" not in js and "XMLHttpRequest" not in js


def test_ort_script_tag_carries_sri_and_crossorigin():
    """D-F3: the pinned onnxruntime-web <script> has a sha384 SRI integrity value and crossorigin=anonymous."""
    html = INDEX_HTML.read_text(encoding="utf-8")
    tag = re.search(r"<script\b[^>]*onnxruntime-web@[^>]*>", html, re.S)
    assert tag, "no onnxruntime-web <script> tag"
    t = tag.group(0)
    m = re.search(r'integrity="sha384-([A-Za-z0-9+/]{64})"', t)
    assert m, "the onnxruntime-web <script> must carry integrity=\"sha384-<64 base64 chars>\""
    assert 'crossorigin="anonymous"' in t


def test_bench_js_sets_threads_before_first_session_and_records_applied_value():
    js = BENCH_JS.read_text(encoding="utf-8")
    assert "firstWasmThreads" in js and "wasm_num_threads_applied" in js and "wasm_num_threads_requested" in js
    assert js.index("ort.env.wasm.numThreads = ep.numThreads") < js.index("ort.InferenceSession.create")


def test_bench_js_checks_model_sha_against_queries_provenance():
    js = BENCH_JS.read_text(encoding="utf-8")
    assert "queries.provenance" in js and "model_${label}_sha256" in js and "queries_sha_match" in js


def test_index_html_references_bench_js_and_ort():
    html = INDEX_HTML.read_text(encoding="utf-8")
    assert 'src="bench.js"' in html
    assert "onnxruntime-web@" in html


# -------------------------------------------------------------------------------------- browser_queries round-trip
def _rows_npz_dict(built, scoring, n=5, seed=7):
    """A --rows-NPZ-shaped dict (pattern of test_bench.py's `_valid_rows_data`): scoring.input_keys arrays + the
    three meta fields `v4dev.bench.load_rows_npz` requires."""
    batch = example_batch(built, n=n, seed=seed, min_len=1)
    data = {k: batch[k].numpy() for k in scoring.input_keys}
    data["decision_id"] = np.arange(1000, 1000 + n, dtype=np.int64)
    data["target_class"] = batch["target_class"].numpy()
    data["manifest_row"] = np.arange(n, dtype=np.int64)
    return data


def test_select_input_keys_excludes_meta_fields_without_ckpt():
    made = tiny_scoring("GRU", k=K_SMALL)
    data = _rows_npz_dict(made["built"], made["scoring"], n=4)
    keys = select_input_keys(data, None)
    for meta in META_KEYS:
        assert meta not in keys
    for k in made["scoring"].input_keys:
        assert k in keys


def test_select_input_keys_uses_exact_input_keys_when_given():
    made = tiny_scoring("GRU", k=K_SMALL)
    data = _rows_npz_dict(made["built"], made["scoring"], n=4)
    keys = select_input_keys(data, list(made["scoring"].input_keys))
    assert keys == list(made["scoring"].input_keys)


def test_select_input_keys_refuses_missing_required_key():
    made = tiny_scoring("GRU", k=K_SMALL)
    data = _rows_npz_dict(made["built"], made["scoring"], n=4)
    del data[made["scoring"].input_keys[-1]]
    assert_raises(BrowserQueriesError, select_input_keys, data, list(made["scoring"].input_keys))


def test_resolve_n_rows_clamps_to_n_queries_and_refuses_mismatch():
    made = tiny_scoring("GRU", k=K_SMALL)
    data = _rows_npz_dict(made["built"], made["scoring"], n=5)
    keys = select_input_keys(data, None)
    assert resolve_n_rows(data, keys, 3) == 3
    assert resolve_n_rows(data, keys, 1000) == 5
    data[keys[0]] = data[keys[0]][:-1]           # now mismatched row counts
    assert_raises(BrowserQueriesError, resolve_n_rows, data, keys, 5)


@pytest.mark.parametrize("family", ["GRU", "SASREC"])
def test_build_queries_round_trips_values_and_dtypes(family):
    made = tiny_scoring(family, k=K_SMALL, seed=3)
    scoring, built = made["scoring"], made["built"]
    data = _rows_npz_dict(built, scoring, n=6, seed=11)
    keys = select_input_keys(data, list(scoring.input_keys))
    n = resolve_n_rows(data, keys, 4)
    doc = build_queries(data, keys, n)

    assert doc["schema"] == "v4dev.browser_queries.v1"
    assert doc["n_rows"] == 4
    assert doc["input_keys"] == keys
    assert doc["reference"]["available"] is False
    assert doc["reference"]["top20"] is None

    for k in keys:
        expected = np.asarray(data[k])[:4]
        got = np.asarray(doc["inputs"][k])
        assert got.shape == expected.shape
        assert np.allclose(got.astype(np.float64), expected.astype(np.float64))
        assert doc["shapes"][k] == list(expected.shape[1:])
        assert doc["dtypes"][k]                          # every included key gets a recognised ONNX dtype string

    # round-trips through JSON (the actual on-disk format) without loss for integer keys
    text = json.dumps(doc)
    reloaded = json.loads(text)
    lengths_key = "lengths"
    assert reloaded["inputs"][lengths_key] == doc["inputs"][lengths_key]


def test_build_queries_refuses_unrecognised_dtype():
    data = {"item_tokens": np.zeros((2, 3), dtype=np.complex64)}
    assert_raises(BrowserQueriesError, build_queries, data, ["item_tokens"], 2)


def test_reference_top20_shape_and_values(tmp_path):
    made = tiny_scoring("GRU", k=K_SMALL, seed=5)
    scoring, built = made["scoring"], made["built"]
    data = _rows_npz_dict(built, scoring, n=6, seed=21)
    keys = list(scoring.input_keys)
    ref = reference_top20(data, scoring, keys, n=6, top_k=20)
    assert ref["available"] is True
    assert ref["source"] == "pytorch_cpu_wrapper"
    top = np.asarray(ref["top20"])
    assert top.shape == (6, 20)
    assert (top >= 0).all() and (top < built.K).all()


def test_cli_round_trip_without_checkpoint(tmp_path):
    made = tiny_scoring("SASREC", k=K_SMALL, seed=9)
    scoring, built = made["scoring"], made["built"]
    data = _rows_npz_dict(built, scoring, n=10, seed=1)
    rows_path = tmp_path / "rows.npz"
    np.savez(rows_path, **data)
    out_path = tmp_path / "queries.json"

    rc = main(["--rows", str(rows_path), "--out", str(out_path), "--n-queries", "4"])
    assert rc == 0
    doc = json.loads(out_path.read_text(encoding="utf-8"))
    assert doc["n_rows"] == 4
    assert doc["reference"]["available"] is False
    for meta in META_KEYS:
        assert meta not in doc["inputs"]
    for k in scoring.input_keys:
        assert k in doc["inputs"]
        expected = np.asarray(data[k])[:4]
        assert np.asarray(doc["inputs"][k]).shape == expected.shape


def test_cli_round_trip_with_checkpoint_writes_reference(tmp_path):
    made = tiny_scoring("GRU", k=K_SMALL, seed=13)
    scoring, built = made["scoring"], made["built"]
    data = _rows_npz_dict(built, scoring, n=8, seed=2)
    rows_path = tmp_path / "rows.npz"
    np.savez(rows_path, **data)
    ckpt_path = tmp_path / "ckpt.pt"
    torch.save(dict(built.module.state_dict()), str(ckpt_path))
    out_path = tmp_path / "queries.json"

    rc = main(["--rows", str(rows_path), "--out", str(out_path), "--n-queries", "5",
              "--ckpt", str(ckpt_path), "--family", "GRU", "--k", str(K_SMALL), "--seed", "13"])
    assert rc == 0
    doc = json.loads(out_path.read_text(encoding="utf-8"))
    assert doc["n_rows"] == 5
    assert doc["reference"]["available"] is True
    top = np.asarray(doc["reference"]["top20"])
    assert top.shape == (5, 20)


def test_build_provenance_binds_rows_ckpt_and_bench_result_shas(tmp_path):
    import hashlib
    rows = tmp_path / "rows.npz"
    rows.write_bytes(b"rows-bytes")
    (tmp_path / "rows.meta.json").write_text("{}", encoding="utf-8")
    ckpt = tmp_path / "ckpt.pt"
    ckpt.write_bytes(b"ckpt-bytes")
    ck_sha = hashlib.sha256(b"ckpt-bytes").hexdigest()
    br = tmp_path / "BENCH_RESULT.json"
    br.write_text(json.dumps({"checkpoint_sha256": ck_sha, "model_sha256": {"fp32": "a" * 64, "int8": "b" * 64}}),
                  encoding="utf-8")
    prov = build_provenance(rows, ckpt_path=ckpt, bench_result_path=br)
    assert prov["rows_npz_sha256"] == hashlib.sha256(b"rows-bytes").hexdigest()
    assert prov["ckpt_sha256"] == ck_sha and prov["model_fp32_sha256"] == "a" * 64
    assert prov["model_int8_sha256"] == "b" * 64 and len(prov["rows_meta_sha256"]) == 64
    assert len(prov["bench_result_sha256"]) == 64
    other = tmp_path / "other.pt"
    other.write_bytes(b"different")
    assert_raises(BrowserQueriesError, build_provenance, rows, ckpt_path=other, bench_result_path=br)
    br.write_text(json.dumps({"checkpoint_sha256": ck_sha}), encoding="utf-8")       # a pre-DEV-04 BENCH_RESULT
    assert_raises(BrowserQueriesError, build_provenance, rows, bench_result_path=br)


def test_cli_writes_provenance_into_queries_json(tmp_path):
    made = tiny_scoring("GRU", k=K_SMALL, seed=21)
    scoring, built = made["scoring"], made["built"]
    rows_path = tmp_path / "rows.npz"
    np.savez(rows_path, **_rows_npz_dict(built, scoring, n=4, seed=3))
    out_path = tmp_path / "queries.json"
    assert main(["--rows", str(rows_path), "--out", str(out_path), "--n-queries", "3"]) == 0
    doc = json.loads(out_path.read_text(encoding="utf-8"))
    assert len(doc["provenance"]["rows_npz_sha256"]) == 64


def test_cli_refuses_ckpt_without_family(tmp_path):
    made = tiny_scoring("GRU", k=K_SMALL)
    data = _rows_npz_dict(made["built"], made["scoring"], n=3)
    rows_path = tmp_path / "rows.npz"
    np.savez(rows_path, **data)
    with pytest.raises(SystemExit):
        main(["--rows", str(rows_path), "--out", str(tmp_path / "q.json"), "--ckpt", "somewhere.pt"])


# ------------------------------------------------------------------------------------------------- negative controls
@nc("select_input_keys must exclude decision_id/target_class/manifest_row, not treat them as model inputs")
def test_nc_select_input_keys_would_include_meta_fields():
    made = tiny_scoring("GRU", k=K_SMALL)
    data = _rows_npz_dict(made["built"], made["scoring"], n=3)
    keys = select_input_keys(data, None)
    assert "decision_id" in keys


@nc("build_queries must refuse an unrecognised numpy dtype (e.g. complex64), not silently serialize it")
def test_nc_build_queries_would_accept_unrecognised_dtype():
    data = {"item_tokens": np.zeros((2, 3), dtype=np.complex64)}
    try:
        build_queries(data, ["item_tokens"], 2)
        refused = False
    except BrowserQueriesError:
        refused = True
    assert not refused, "build_queries correctly refused the unrecognised dtype (this NC claims the opposite)"


@nc("resolve_n_rows must refuse mismatched row counts across the selected keys, not silently truncate")
def test_nc_resolve_n_rows_would_accept_mismatched_arrays():
    made = tiny_scoring("GRU", k=K_SMALL)
    data = _rows_npz_dict(made["built"], made["scoring"], n=5)
    keys = select_input_keys(data, None)
    data[keys[0]] = data[keys[0]][:-1]
    try:
        resolve_n_rows(data, keys, 5)
        refused = False
    except BrowserQueriesError:
        refused = True
    assert not refused, "resolve_n_rows correctly refused the mismatched arrays (this NC claims the opposite)"
