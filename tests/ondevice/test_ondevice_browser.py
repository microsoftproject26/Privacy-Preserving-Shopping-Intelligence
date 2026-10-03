"""The browser pages (`ppsi/ondevice/browser/`) and `browser_queries`.

Static checks (no browser): the files exist, onnxruntime-web is pinned to an exact version with SRI, only
cdn.jsdelivr.net is referenced, the latency page makes no network call of its own, the device page's only POST is
the same-origin loopback sink, wasm threads are set before the first session, and the device page carries no
hard-coded device defaults. serve_coi.py: COOP/COEP headers, read-only, and the opt-in loopback sink with tag,
schema, size and origin checks. browser_queries: a round-trip on synthetic fixtures.
"""
from __future__ import annotations

import json
import re
import sys
import threading
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import pytest
import torch
from ondevice_testkit import K_SMALL, assert_raises, nc, tiny_scoring

from ppsi.ondevice.browser_queries import (
    META_KEYS,
    BrowserQueriesError,
    build_provenance,
    build_queries,
    main,
    reference_top20,
    resolve_n_rows,
    select_input_keys,
)
from ppsi.ondevice.scoring import example_batch

BROWSER_DIR = Path(__file__).resolve().parents[2] / "ppsi" / "ondevice" / "browser"
INDEX_HTML = BROWSER_DIR / "index.html"
BENCH_JS = BROWSER_DIR / "bench.js"
DEVICE_HTML = BROWSER_DIR / "device_bench.html"
DEVICE_JS = BROWSER_DIR / "device_bench.js"
SERVE = BROWSER_DIR / "serve_coi.py"
URL_RE = re.compile(r"https?://[^\s\"'<>]+")
SCHEMA = "ppsi.ondevice.device_bench.v1"


def _serve_coi():
    sys.path.insert(0, str(BROWSER_DIR))
    try:
        import serve_coi
    finally:
        sys.path.remove(str(BROWSER_DIR))
    return serve_coi


# ------------------------------------------------------------------------------------------------- static checks
def test_browser_files_exist():
    for p in (INDEX_HTML, BENCH_JS, DEVICE_HTML, DEVICE_JS, SERVE):
        assert p.is_file(), f"missing {p}"


def test_onnxruntime_web_version_is_pinned_and_identical_on_both_pages():
    versions = []
    for page in (INDEX_HTML, DEVICE_HTML):
        m = re.search(r"onnxruntime-web@([^/\"']+)/dist/", page.read_text(encoding="utf-8"))
        assert m, f"{page.name} must load onnxruntime-web with an explicit @version in its CDN URL"
        assert re.match(r"^\d+\.\d+\.\d+$", m.group(1)), f"expected an exact version pin, got {m.group(1)!r}"
        versions.append(m.group(1))
    assert versions[0] == versions[1]


def test_ort_script_tag_carries_sri_and_crossorigin():
    for page in (INDEX_HTML, DEVICE_HTML):
        tag = re.search(r"<script\b[^>]*onnxruntime-web@[^>]*>", page.read_text(encoding="utf-8"), re.DOTALL)
        assert tag, f"no onnxruntime-web <script> tag in {page.name}"
        t = tag.group(0)
        assert re.search(r'integrity="sha384-([A-Za-z0-9+/]{64})"', t), page.name
        assert 'crossorigin="anonymous"' in t


def test_only_cdn_jsdelivr_host_is_referenced():
    urls = URL_RE.findall(INDEX_HTML.read_text(encoding="utf-8")) + URL_RE.findall(BENCH_JS.read_text(encoding="utf-8"))
    assert {urlparse(u).netloc for u in urls} == {"cdn.jsdelivr.net"}
    urls = URL_RE.findall(DEVICE_HTML.read_text(encoding="utf-8")) + URL_RE.findall(
        DEVICE_JS.read_text(encoding="utf-8"))
    hosts = {urlparse(u).netloc for u in urls}
    assert hosts <= {"cdn.jsdelivr.net", "localhost:8765"} and "cdn.jsdelivr.net" in hosts


def test_latency_page_makes_no_network_calls_of_its_own():
    js = BENCH_JS.read_text(encoding="utf-8")
    assert not URL_RE.findall(js), "bench.js itself must reference no external URL"
    assert "fetch(" not in js and "XMLHttpRequest" not in js


def test_latency_page_sets_threads_before_first_session_and_checks_model_sha():
    js = BENCH_JS.read_text(encoding="utf-8")
    assert "firstWasmThreads" in js and "wasm_num_threads_applied" in js and "wasm_num_threads_requested" in js
    assert js.index("ort.env.wasm.numThreads = ep.numThreads") < js.index("ort.InferenceSession.create")
    assert "queries.provenance" in js and "model_${label}_sha256" in js and "queries_sha_match" in js
    html = INDEX_HTML.read_text(encoding="utf-8")
    assert 'src="bench.js"' in html


def test_device_page_has_no_upload_channels():
    js = DEVICE_JS.read_text(encoding="utf-8")
    for bad in ("XMLHttpRequest", "sendBeacon", "WebSocket", "EventSource", "FormData"):
        assert bad not in js, bad
    assert js.count('method: "POST"') == 1 and '"__bench_result?tag="' in js   # the loopback sink only
    calls = re.findall(r"fetch\(([^,)]*)", js)
    assert calls and all(not c.strip().startswith(("\"http", "'http", "`http", "\"//")) for c in calls), calls
    body = js[js.index("function isSameOriginRelative"):js.index("function numParam")]
    assert 'p.includes("\\\\")' in body and 'startsWith("//")' in body and "new URL(p, location.href).origin" in body
    assert '".."' in body


def test_device_page_record_keys_and_no_accuracy_fields():
    js = DEVICE_JS.read_text(encoding="utf-8")
    assert f'SCHEMA = "{SCHEMA}"' in js and 'MARKER = "DEVICE_BENCH_JSON:"' in js and "console.log(MARKER" in js
    for key in ("identity", "arm", "cross_origin_isolated", "protocol", "models", "cold_start", "fetch_ms",
                "session_create_ms", "first_inference_ms", "warm", "p50_ms", "p95_ms", "p99_ms", "memory",
                "sha256", "hardware_concurrency", "device_memory_gb", "wasm_threads_applied", "timer_resolution_ms"):
        assert re.search(rf"\b{key}\b", js), key
    for bad in ("mrr", "recall", "parity", "ndcg"):
        assert bad not in js.lower(), bad


def test_device_page_one_model_per_load_explicit_threads_and_no_device_defaults():
    js, html = DEVICE_JS.read_text(encoding="utf-8"), DEVICE_HTML.read_text(encoding="utf-8")
    assert "modelSel" in html and "one_model_per_page_load" in js and "doc.models[model]" in js
    assert 'id="devModel" value=""' in html and 'id="devId" value=""' in html and 'id="devPower" value=""' in html
    assert 'numParam("threads", 4)' in js and "hardwareConcurrency || 4" not in js
    assert 'src="device_bench.js"' in html


def _post(port, path, body, headers=None):
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=body, method="POST",
                                 headers={"Content-Type": "application/json", **(headers or {})})
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.status
    except urllib.error.HTTPError as e:
        return e.code
    except (urllib.error.URLError, ConnectionError) as e:
        # the server refuses an oversize body without reading it; the client may then see a reset or a broken pipe
        reason = getattr(e, "reason", e)
        if isinstance(reason, ConnectionError):
            return "RESET"
        raise


def test_serve_coi_headers_and_read_only():
    serve_coi = _serve_coi()
    assert serve_coi.COI_HEADERS["Cross-Origin-Opener-Policy"] == "same-origin"
    assert serve_coi.COI_HEADERS["Cross-Origin-Embedder-Policy"] == "require-corp"
    srv = serve_coi.make_server(BROWSER_DIR, "127.0.0.1", 0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        port = srv.server_address[1]
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/device_bench.html", timeout=10) as r:
            assert r.headers["Cross-Origin-Opener-Policy"] == "same-origin"
            assert r.headers["Cross-Origin-Embedder-Policy"] == "require-corp"
        assert _post(port, "/x", b"a") == 405
        assert _post(port, "/__bench_result?tag=x", json.dumps({"schema": SCHEMA}).encode()) == 405  # sink off
        assert srv.server_address[0] == "127.0.0.1"
    finally:
        srv.shutdown()
        srv.server_close()


def test_sink_accepts_valid_and_rejects_bad(tmp_path):
    srv = _serve_coi().make_server(BROWSER_DIR, "127.0.0.1", 0, results_dir=tmp_path)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    port = srv.server_address[1]
    try:
        good = json.dumps({"schema": SCHEMA, "models": {}}).encode()
        assert _post(port, "/__bench_result?tag=fp32_wasm-1t_s1", good) == 204
        assert json.loads((tmp_path / "fp32_wasm-1t_s1.json").read_text())["schema"] == SCHEMA
        assert _post(port, "/__bench_result?tag=../evil", good) == 400            # tag validation, no traversal
        assert _post(port, "/__bench_result?tag=a%2F..%2Fb", good) == 400
        assert _post(port, "/__bench_result?tag=x", b"{" + b" " * 10) == 400       # not json
        assert _post(port, "/__bench_result?tag=x", b'{"schema":"other"}') == 400  # schema
        assert _post(port, "/__bench_result?tag=big", b'{"schema":"x","pad":"' + b"a" * 1_100_000 + b'"}') in (413,
                                                                                                         "RESET")
        assert _post(port, "/__bench_result?tag=x", good, {"Origin": "http://evil.example"}) == 403
        assert _post(port, "/other", good) == 405
        assert sorted(p.name for p in tmp_path.iterdir()) == ["fp32_wasm-1t_s1.json"]
    finally:
        srv.shutdown()
        srv.server_close()


# -------------------------------------------------------------------------------------- browser_queries round-trip
def _rows_npz_dict(built, scoring, n=5, seed=7):
    """A rows-NPZ-shaped dict: scoring.input_keys arrays + the three meta fields `bench.load_rows_npz` requires."""
    batch = example_batch(built, n=n, seed=seed, min_len=1)
    data = {k: batch[k].numpy() for k in scoring.input_keys}
    data["decision_id"] = np.arange(1000, 1000 + n, dtype=np.int64)
    data["target_class"] = batch["target_class"].numpy()
    data["manifest_row"] = np.arange(n, dtype=np.int64)
    return data


def test_select_input_keys_with_and_without_a_model():
    made = tiny_scoring("GRU", k=K_SMALL)
    data = _rows_npz_dict(made["built"], made["scoring"], n=4)
    keys = select_input_keys(data, None)
    assert not set(META_KEYS) & set(keys) and set(made["scoring"].input_keys) <= set(keys)
    assert select_input_keys(data, list(made["scoring"].input_keys)) == list(made["scoring"].input_keys)
    del data[made["scoring"].input_keys[-1]]
    assert_raises(BrowserQueriesError, select_input_keys, data, list(made["scoring"].input_keys))


def test_resolve_n_rows_clamps_to_n_queries_and_refuses_mismatch():
    made = tiny_scoring("GRU", k=K_SMALL)
    data = _rows_npz_dict(made["built"], made["scoring"], n=5)
    keys = select_input_keys(data, None)
    assert resolve_n_rows(data, keys, 3) == 3 and resolve_n_rows(data, keys, 1000) == 5
    data[keys[0]] = data[keys[0]][:-1]
    assert_raises(BrowserQueriesError, resolve_n_rows, data, keys, 5)


@pytest.mark.parametrize("family", ["GRU", "SASREC"])
def test_build_queries_round_trips_values_and_dtypes(family):
    made = tiny_scoring(family, k=K_SMALL, seed=3)
    scoring, built = made["scoring"], made["built"]
    data = _rows_npz_dict(built, scoring, n=6, seed=11)
    keys = select_input_keys(data, list(scoring.input_keys))
    doc = build_queries(data, keys, resolve_n_rows(data, keys, 4))
    assert doc["schema"] == "ppsi.ondevice.browser_queries.v1" and doc["n_rows"] == 4 and doc["input_keys"] == keys
    assert doc["reference"]["available"] is False and doc["reference"]["top20"] is None
    for k in keys:
        expected = np.asarray(data[k])[:4]
        got = np.asarray(doc["inputs"][k])
        assert got.shape == expected.shape
        assert np.allclose(got.astype(np.float64), expected.astype(np.float64))
        assert doc["shapes"][k] == list(expected.shape[1:]) and doc["dtypes"][k]
    assert json.loads(json.dumps(doc))["inputs"]["lengths"] == doc["inputs"]["lengths"]


def test_build_queries_refuses_unrecognised_dtype():
    assert_raises(BrowserQueriesError, build_queries, {"item_tokens": np.zeros((2, 3), dtype=np.complex64)},
                  ["item_tokens"], 2)


def test_reference_top20_shape_and_values():
    made = tiny_scoring("GRU", k=K_SMALL, seed=5)
    scoring, built = made["scoring"], made["built"]
    ref = reference_top20(_rows_npz_dict(built, scoring, n=6, seed=21), scoring, list(scoring.input_keys), n=6,
                          top_k=20)
    top = np.asarray(ref["top20"])
    assert ref["available"] is True and ref["source"] == "pytorch_cpu_wrapper"
    assert top.shape == (6, 20) and (top >= 0).all() and (top < built.K).all()


def test_cli_round_trip_with_rows_and_without_a_model(tmp_path):
    made = tiny_scoring("SASREC", k=K_SMALL, seed=9)
    data = _rows_npz_dict(made["built"], made["scoring"], n=10, seed=1)
    rows_path = tmp_path / "rows.npz"
    np.savez(rows_path, **data)
    out_path = tmp_path / "queries.json"
    assert main(["--rows", str(rows_path), "--out", str(out_path), "--n-queries", "4"]) == 0
    doc = json.loads(out_path.read_text(encoding="utf-8"))
    assert doc["n_rows"] == 4 and doc["reference"]["available"] is False
    assert not set(META_KEYS) & set(doc["inputs"]) and set(made["scoring"].input_keys) <= set(doc["inputs"])
    assert len(doc["provenance"]["rows_npz_sha256"]) == 64


def test_cli_round_trip_with_checkpoint_writes_reference(tmp_path):
    made = tiny_scoring("GRU", k=K_SMALL, seed=13)
    rows_path = tmp_path / "rows.npz"
    np.savez(rows_path, **_rows_npz_dict(made["built"], made["scoring"], n=8, seed=2))
    ckpt_path = tmp_path / "ckpt.pt"
    torch.save(dict(made["built"].module.state_dict()), str(ckpt_path))
    out_path = tmp_path / "queries.json"
    assert main(["--rows", str(rows_path), "--out", str(out_path), "--n-queries", "5", "--ckpt", str(ckpt_path),
                 "--family", "GRU", "--k", str(K_SMALL), "--seed", "13"]) == 0
    doc = json.loads(out_path.read_text(encoding="utf-8"))
    assert doc["n_rows"] == 5 and doc["reference"]["available"] is True
    assert np.asarray(doc["reference"]["top20"]).shape == (5, 20) and len(doc["provenance"]["ckpt_sha256"]) == 64


def test_cli_synthetic_queries_from_a_model(tmp_path):
    out_path = tmp_path / "queries.json"
    assert main(["--out", str(out_path), "--n-queries", "3", "--family", "SASREC", "--k", str(K_SMALL)]) == 0
    doc = json.loads(out_path.read_text(encoding="utf-8"))
    assert doc["n_rows"] == 3 and doc["reference"]["available"] is True and "rows_npz_sha256" not in doc["provenance"]


def test_build_provenance_binds_rows_ckpt_and_bench_result_shas(tmp_path):
    import hashlib
    rows = tmp_path / "rows.npz"
    rows.write_bytes(b"rows-bytes")
    ckpt = tmp_path / "ckpt.pt"
    ckpt.write_bytes(b"ckpt-bytes")
    ck_sha = hashlib.sha256(b"ckpt-bytes").hexdigest()
    br = tmp_path / "BENCH_RESULT.json"
    br.write_text(json.dumps({"checkpoint_sha256": ck_sha, "model_sha256": {"fp32": "a" * 64, "int8": "b" * 64}}),
                  encoding="utf-8")
    prov = build_provenance(rows_path=rows, ckpt_path=ckpt, bench_result_path=br)
    assert prov["rows_npz_sha256"] == hashlib.sha256(b"rows-bytes").hexdigest()
    assert prov["ckpt_sha256"] == ck_sha and prov["model_fp32_sha256"] == "a" * 64
    assert prov["model_int8_sha256"] == "b" * 64 and len(prov["bench_result_sha256"]) == 64
    other = tmp_path / "other.pt"
    other.write_bytes(b"different")
    assert_raises(BrowserQueriesError, build_provenance, rows_path=rows, ckpt_path=other, bench_result_path=br)
    br.write_text(json.dumps({"checkpoint_sha256": ck_sha}), encoding="utf-8")       # no model sha256 block
    assert_raises(BrowserQueriesError, build_provenance, rows_path=rows, bench_result_path=br)


def test_cli_refuses_ckpt_without_family(tmp_path):
    made = tiny_scoring("GRU", k=K_SMALL)
    rows_path = tmp_path / "rows.npz"
    np.savez(rows_path, **_rows_npz_dict(made["built"], made["scoring"], n=3))
    with pytest.raises(SystemExit):
        main(["--rows", str(rows_path), "--out", str(tmp_path / "q.json"), "--ckpt", "somewhere.pt"])


# ------------------------------------------------------------------------------------------------- negative controls
@nc("select_input_keys must exclude decision_id/target_class/manifest_row, not treat them as model inputs")
def test_nc_select_input_keys_would_include_meta_fields():
    made = tiny_scoring("GRU", k=K_SMALL)
    assert "decision_id" in select_input_keys(_rows_npz_dict(made["built"], made["scoring"], n=3), None)


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
