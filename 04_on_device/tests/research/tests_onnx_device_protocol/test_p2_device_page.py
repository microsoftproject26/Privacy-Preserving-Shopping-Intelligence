"""P2-14 static tests: p2_bench.html / p2_bench.js / serve_coi.py / p2_edge_server.sh (no browser, no benchmarks)."""
from __future__ import annotations

import json
import re
import subprocess
import sys
import threading
import urllib.request
from urllib.parse import urlparse

import pytest
import ppsi_paths

BROWSER_DIR = ppsi_paths.code_dir("v4dev") / "browser"
HTML = BROWSER_DIR / "p2_bench.html"
JS = BROWSER_DIR / "p2_bench.js"
SERVE = BROWSER_DIR / "serve_coi.py"
README = BROWSER_DIR / "README_P2_DEVICE.md"
SH = ppsi_paths.locate("tools/p2_edge_server.sh")
INDEX_HTML = BROWSER_DIR / "index.html"
URL_RE = re.compile(r"https?://[^\s\"'<>]+")


def test_files_exist():
    for p in (HTML, JS, SERVE, README, SH):
        assert p.is_file(), p


def test_ort_pinned_and_same_as_existing_page():
    m = re.search(r"onnxruntime-web@([^/\"']+)/dist/", HTML.read_text(encoding="utf-8"))
    assert m and re.match(r"^\d+\.\d+\.\d+$", m.group(1))
    m0 = re.search(r"onnxruntime-web@([^/\"']+)/dist/", INDEX_HTML.read_text(encoding="utf-8"))
    assert m.group(1) == m0.group(1)
    assert 'integrity="sha384-' in HTML.read_text(encoding="utf-8")


def test_only_pinned_cdn_host_referenced():
    urls = URL_RE.findall(HTML.read_text(encoding="utf-8")) + URL_RE.findall(JS.read_text(encoding="utf-8"))
    hosts = {urlparse(u).netloc for u in urls}
    # the example localhost URL in the page note is fine; everything else must be the pinned CDN
    assert hosts <= {"cdn.jsdelivr.net", "localhost:8765"}, hosts
    assert "cdn.jsdelivr.net" in hosts


def test_no_upload_channels():
    js = JS.read_text(encoding="utf-8")
    for bad in ("XMLHttpRequest", "sendBeacon", "WebSocket", "EventSource", "FormData"):
        assert bad not in js, bad
    # exactly one POST: the same-origin relative loopback sink
    assert js.count('method: "POST"') == 1 and '"__p2_result?tag="' in js
    # every fetch() argument is a same-origin relative path variable from fetchBuf / the queries path, never a URL
    calls = re.findall(r"fetch\(([^,)]*)", js)
    assert calls and all(not c.strip().startswith(("\"http", "'http", "`http", "\"//")) for c in calls), calls
    assert "isSameOriginRelative" in js


def test_json_schema_keys_and_marker_present():
    js = JS.read_text(encoding="utf-8")
    assert 'MARKER = "P2_DEVICE_BENCH_JSON:"' in js
    assert "console.log(MARKER" in js
    assert "DEVICE_BENCH_P2.json" in js
    for key in ("schema", "identity", "ort", "arm", "cross_origin_isolated", "protocol", "models", "energy_proxy",
                "cold_start", "fetch_ms", "session_create_ms", "first_inference_ms", "warm", "p50_ms", "p95_ms",
                "p99_ms", "memory", "bytes", "sha256", "label", "user_agent", "hardware_concurrency",
                "device_memory_gb", "wasm_threads_applied"):
        assert re.search(rf"\b{key}\b", js), key
    assert "measureUserAgentSpecificMemory" in js
    assert "userAgent" in js and "hardwareConcurrency" in js and "deviceMemory" in js
    assert "SERVER_BROWSER" in HTML.read_text(encoding="utf-8")
    # latency/memory only: no accuracy fields
    for bad in ("mrr", "recall", "parity", "ndcg"):
        assert bad not in js.lower(), bad


def test_shell_script_contract():
    sh = SH.read_text(encoding="utf-8")
    assert "\r" not in sh, "CRLF would break bash"
    assert "MODE=plan" in sh and "--run" in sh
    assert sh.count("microsoft-edge-stable_") >= 1 and "packages.microsoft.com" in sh
    assert len(re.findall(r'^\s+"\d+\.[\d.]+-1"', sh, re.M)) >= 3
    assert "154." in sh and "dpkg-deb -x" in sh and "ldd" in sh and "sha256sum" in sh
    assert "--headless=new" in sh and "--no-sandbox" in sh
    assert "taskset" in sh and "P2_DEVICE_BENCH_JSON:" in sh
    assert "apt install" not in sh.replace("never apt install", "").replace("do not apt install", "").replace(
        "we never apt install", "")
    assert "sudo" not in sh


def test_shell_syntax_if_bash_available():
    try:
        r = subprocess.run(["bash", "-n", str(SH)], capture_output=True, text=True, timeout=30)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pytest.skip("bash unavailable")
    if "WSL" in r.stderr:
        pytest.skip("bash is a broken WSL stub")
    assert r.returncode == 0, r.stderr


def test_serve_coi_headers_and_read_only():
    sys.path.insert(0, str(BROWSER_DIR))
    try:
        import serve_coi
    finally:
        sys.path.remove(str(BROWSER_DIR))
    assert serve_coi.COI_HEADERS["Cross-Origin-Opener-Policy"] == "same-origin"
    assert serve_coi.COI_HEADERS["Cross-Origin-Embedder-Policy"] == "require-corp"
    srv = serve_coi.make_server(BROWSER_DIR, "127.0.0.1", 0)
    port = srv.server_address[1]
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/p2_bench.html", timeout=10) as r:
            assert r.headers["Cross-Origin-Opener-Policy"] == "same-origin"
            assert r.headers["Cross-Origin-Embedder-Policy"] == "require-corp"
        req = urllib.request.Request(f"http://127.0.0.1:{port}/x", data=b"a", method="POST")
        with pytest.raises(urllib.error.HTTPError) as ei:
            urllib.request.urlopen(req, timeout=10)
        assert ei.value.code == 405
    finally:
        srv.shutdown()
        srv.server_close()


def test_readme_mentions_owner_steps():
    t = README.read_text(encoding="utf-8")
    for s in ("serve_coi.py", "powercfg /batteryreport", "OPTIONAL", "session", "AC power", "DEVICE_BENCH_P2.json"):
        assert s in t, s


# ---------------------------------------------------------------------------------------------- REV-P2-14 fixes
def test_same_origin_rule_rejects_backslash_and_protocol_relative():
    js = JS.read_text(encoding="utf-8")
    body = js[js.index("function isSameOriginRelative"):js.index("function numParam")]
    assert 'p.includes("\\\\")' in body and 'startsWith("//")' in body and "new URL(p, location.href).origin" in body
    assert '".."' in body


def test_one_model_per_page_load_and_no_hardcoded_laptop_defaults():
    js, html = JS.read_text(encoding="utf-8"), HTML.read_text(encoding="utf-8")
    assert "modelSel" in html and "one_model_per_page_load" in js and "doc.models[model]" in js
    assert 'id="devModel" value=""' in html and 'id="devId" value=""' in html and 'id="devPower" value=""' in html
    assert "LENOVO" not in html
    assert 'Q.get("d1") === "1"' in js and "headless_hint" in js
    assert 'numParam("warmup"' in js and "timer_resolution_ms" in js


def test_threads_explicit_everywhere():
    sh, readme, js = SH.read_text(encoding="utf-8"), README.read_text(encoding="utf-8"), JS.read_text(encoding="utf-8")
    assert "threads=${THR}" in sh and "count_cpus" in sh
    assert "arm=wasm-mt&threads=4" in readme
    assert 'numParam("threads", 4)' in js and "hardwareConcurrency || 4" not in js


def test_shell_process_group_liveness_and_failed_semantics():
    sh = SH.read_text(encoding="utf-8")
    assert "setsid" in sh and 'kill -TERM -- "-$1"' in sh and "--user-data-dir=$2" in sh and "pkill" in sh
    assert 'kill -0 "$PID"' in sh and "DIED" in sh
    assert 'doc.get("error")' in sh and 'print("ERR:' in sh
    assert "ldd" in sh and "SKIP[$V]=1" in sh
    assert "SERVER_BROWSER" in sh and "model_desc=" in sh and "LENOVO" not in sh
    assert "--results-dir" in sh and "sink=${TAG}" in sh
    # interleaving: session outer loop, rotation
    assert "(i + S - 1) % NV" in sh and "(j + S - 1) % NC" in sh
    assert sh.index("for S in $(seq 1") < sh.index("for ((i = 0; i < NV")
    # one model per launch: each combo is its own run_one call
    assert "run_one " in sh and "--no-first-run" in sh


def _start_sink(tmp_path):
    sys.path.insert(0, str(BROWSER_DIR))
    try:
        import serve_coi
    finally:
        sys.path.remove(str(BROWSER_DIR))
    srv = serve_coi.make_server(BROWSER_DIR, "127.0.0.1", 0, results_dir=tmp_path)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, srv.server_address[1]


def _post(port, path, body, headers=None):
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=body, method="POST",
                                 headers={"Content-Type": "application/json", **(headers or {})})
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.status
    except urllib.error.HTTPError as e:
        return e.code


def test_sink_accepts_valid_and_rejects_bad(tmp_path):
    srv, port = _start_sink(tmp_path)
    try:
        good = json.dumps({"schema": "ppsi.v4.device_bench_p2.v1", "models": {}}).encode()
        assert _post(port, "/__p2_result?tag=v1_fp32_wasm-1t_s1", good) == 204
        assert json.loads((tmp_path / "v1_fp32_wasm-1t_s1.json").read_text())["schema"].startswith("ppsi")
        assert _post(port, "/__p2_result?tag=../evil", good) == 400            # tag validation, no traversal
        assert _post(port, "/__p2_result?tag=a%2F..%2Fb", good) == 400
        assert _post(port, "/__p2_result?tag=x", b"{" + b" " * 10) == 400       # not json
        assert _post(port, "/__p2_result?tag=x", b'{"schema":"other"}') == 400  # schema
        assert _post(port, "/__p2_result?tag=big", b'{"schema":"x","pad":"' + b"a" * 1_100_000 + b'"}') == 413
        assert _post(port, "/__p2_result?tag=x", good, {"Origin": "http://evil.example"}) == 403
        assert _post(port, "/other", good) == 405                               # any other path
        assert sorted(p.name for p in tmp_path.iterdir()) == ["v1_fp32_wasm-1t_s1.json"]
    finally:
        srv.shutdown()
        srv.server_close()


def test_sink_disabled_by_default():
    sys.path.insert(0, str(BROWSER_DIR))
    try:
        import serve_coi
    finally:
        sys.path.remove(str(BROWSER_DIR))
    srv = serve_coi.make_server(BROWSER_DIR, "127.0.0.1", 0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        good = json.dumps({"schema": "ppsi.v4.device_bench_p2.v1"}).encode()
        assert _post(srv.server_address[1], "/__p2_result?tag=x", good) == 405
        assert srv.server_address[0] == "127.0.0.1"
    finally:
        srv.shutdown()
        srv.server_close()


def test_readme_gaps_filled():
    t = README.read_text(encoding="utf-8")
    for s in ("py serve_coi.py", "Add python.exe to PATH", "Ctrl+C", "crossOriginIsolated=true", "edge://version",
              "Efficiency mode", "hybrid P/E", "WebGPU", "brotli", "transfer channel", "ROTATE", "ready-to-copy zip"):
        assert s in t, s


# ------------------------------------------------------------------------------- Packages SHA256 verification
PKGS_FIXTURE = """Package: microsoft-edge-beta
Version: 155.0.1.1-1
SHA256: 1111111111111111111111111111111111111111111111111111111111111111

Package: microsoft-edge-stable
Version: 154.0.4258.53-1
Architecture: amd64
Description: Microsoft Edge
 continued line
SHA256: AABBCC0000000000000000000000000000000000000000000000000000000000

Package: microsoft-edge-stable
Version: 152.0.4191.66-1
SHA256: 2222222222222222222222222222222222222222222222222222222222222222
"""


def _sha_mod():
    tools = str(ppsi_paths.locate("tools/p2_packages_sha.py").parent)
    sys.path.insert(0, tools)
    try:
        import p2_packages_sha
    finally:
        sys.path.remove(tools)
    return p2_packages_sha


def test_packages_parse_and_mismatch_refused(tmp_path):
    m = _sha_mod()
    d = m.parse_packages(PKGS_FIXTURE)
    assert d == {"154.0.4258.53-1": "aabbcc" + "0" * 58, "152.0.4191.66-1": "2" * 64}   # beta stanza ignored, lowercased
    assert m.verify(PKGS_FIXTURE, "154.0.4258.53-1", "AABBCC" + "0" * 58) == (True, "aabbcc" + "0" * 58)
    ok, repo = m.verify(PKGS_FIXTURE, "154.0.4258.53-1", "f" * 64)
    assert ok is False and repo is not None
    assert m.verify(PKGS_FIXTURE, "999.0-1", "f" * 64) == (False, None)                  # absent version refused
    # CLI: gz index, real file hash, exit codes
    import gzip, hashlib
    deb = tmp_path / "x.deb"
    deb.write_bytes(b"not a real deb")
    sha = hashlib.sha256(deb.read_bytes()).hexdigest()
    idx = tmp_path / "Packages.gz"
    idx.write_bytes(gzip.compress(PKGS_FIXTURE.replace("2" * 64, sha).encode()))
    assert m.main(["x", "verify", str(idx), "152.0.4191.66-1", str(deb)]) == 0
    assert m.main(["x", "verify", str(idx), "154.0.4258.53-1", str(deb)]) == 1
    assert m.main(["x", "lookup", str(idx), "nope"]) == 2


def test_shell_script_verifies_debs_against_packages():
    sh = SH.read_text(encoding="utf-8")
    assert "dists/stable/main/binary-amd64/Packages.gz" in sh and "p2_packages_sha.py" in sh
    assert "REFUSING" in sh and "sha256_repo" in sh
