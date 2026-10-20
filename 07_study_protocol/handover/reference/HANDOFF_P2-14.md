# P2-14 handoff: real-device measurement code (O8; DL 415)

Status: code + static tests only. No benchmark run, no server contact, no TEST data. Existing Phase-1 files untouched.

## Files (all new)
- code/v4dev/browser/p2_bench.html, p2_bench.js: page. Same pinned onnxruntime-web 1.24.1 + SRI as index.html; if the CDN
  fails, falls back to same-origin vendor/ort.min.js (offline server). Query string: fp32, int8, queries (same-origin
  relative paths only), arm=wasm-1t|wasm-mt, threads, n (1000), warmup (20), auto=1, label, id, session.
- code/v4dev/browser/serve_coi.py: stdlib static server, COOP same-origin / COEP require-corp / CORP, read-only (405 on POST).
- tools/p2_edge_server.sh: plan by default; `--run`, `--list` (lists pool versions), `--site --out --cpus --port`.
- code/v4dev/browser/README_P2_DEVICE.md: owner steps for D1 laptop.
- code/tests_dev/test_p2_device_page.py: 9 static tests.

## Design
- One arm per page load (ort reads numThreads only at the first session). Per model: fetch ms, session-create ms, first
  inference ms (= cold start), 20 warm-ups, 1000 timed batch-1 runs -> p50/p95/p99/mean/min/max; bytes + sha256;
  memory via measureUserAgentSpecificMemory (needs crossOriginIsolated) before/after, else n/a; JS heap if exposed.
  Gzip/brotli sizes are NOT measured in-page (compute offline from the .onnx).
- JSON schema `ppsi.v4.device_bench_p2.v1`: identity{label DEVICE|SERVER_BROWSER, device_id, user_agent,
  hardware_concurrency, device_memory_gb, session, power_and_conditions}, ort, arm, protocol, models{fp32,int8}, energy_proxy
  (null; filled by owner offline). No accuracy fields. No upload: download button + one console line prefixed
  `P2_DEVICE_BENCH_JSON:`.
- Server script: per-version dirs via `dpkg-deb -x`, `ldd | grep 'not found'` reported per version (no apt),
  `taskset -c 0-3` (documented, same for all versions), 3 sessions x 2 arms x versions with 120 s cool-down, console parsed
  from `--enable-logging=stderr`, MANIFEST.tsv (version, URL, sha256, bytes) + ENV.txt.

## Owner steps
See README_P2_DEVICE.md (copy folder to laptop, `python serve_coi.py`, Edge, 3 sessions x 2 arms, save JSON; optional powercfg energy proxy).

## Server commands
`tools/p2_edge_server.sh --list` then edit EDGE_VERSIONS; `tools/p2_edge_server.sh --site <dir> --out <dir>` (plan);
add `--run`. `<dir>` needs p2_bench.html/js, queries.json, models/fp32.onnx, models/int8.onnx, vendor/ (ort.min.js + wasm).

## Tests
test_p2_device_page.py: 8 passed + 1 skipped here (bash on this Windows box is a broken WSL stub; `bash -n` run via Git Bash: OK). Full tests_dev:
152 passed, 8 skipped, 29 xfailed.

## Open questions
1. Edge versions other than 154.0.4258.53-1 are PLACEHOLDERS (150.0.0.0-1, 146.0.0.0-1); real pool versions TO-VERIFY with `--list`; the 154 build may no longer be in the pool.
2. Does the server reach cdn.jsdelivr.net and packages.microsoft.com? If not, files must be staged via the existing transfer channel; vendor/ ort dist needs the pinned 1.24.1 files.
3. Headless console capture relies on Chromium stderr `CONSOLE` lines (long-line behaviour untested); fallback is a CDP client.
4. Edge on server may lack shared libs (ldd report decides; no root).
5. WebGPU arm and PC_PROXY on-device-training cost are not in this packet.
6. `--no-sandbox` run in a JupyterHub container with few cores: hardwareConcurrency may reflect the host, not the taskset set.

## REV fixes (REV-P2-14, PASS_WITH_MINORS)
- M1: each run is `setsid` (own process group); kill TERM then KILL on `-PGID` + `pkill -f --user-data-dir=<profile>`; asserts no survivor; profile deleted.
- M2: wasm-mt always gets explicit `threads=`: page default 4, README URL `&threads=4`, script passes the CPU count of `--cpus` (4 for 0-3).
- M3: `kill -0` liveness check aborts a dead run (DIED); error JSON or missing models = FAILED (`ERR:`), never OK; versions with ldd-missing libs are skipped; exit code non-zero if any FAILED.
- M4: `model=fp32|int8` one model per page load; one browser launch per (version, model, arm, session).
- L1: device description / id / power default empty; `d1=1` fills D1; the server script injects SERVER_BROWSER, `model_desc=` (CPU + Edge version + CPUs), id `S-<ver>`; `headless_hint`/`webdriver` recorded.
- Sink: `serve_coi.py --results-dir` enables `POST /__p2_result?tag=` (loopback client, same-origin Origin, <=1 MB, schema check, tag regex, one path; all else 405); page posts only with `sink=<tag>`; console marker kept as fallback.
- `isSameOriginRelative` rejects backslash, `//`, schemes, `..`, and checks resolved origin.
- Interleaving: session outer loop; version order and model/arm order rotated per session (README rotates the laptop order too).
- Also: `vendor=1` same-origin wasm, timer resolution, SharedArrayBuffer flag, wasm resource names, n/warmup 0 handling, MANIFEST.tsv appended, extra Chromium flags, ENV.txt (flags, vendor sha256), LAN unsupported; README gaps R1-R5 filled.
- Not done: Packages/Release signature check of .deb (S2), peak-memory evidence outside the page (m3).
- Tests: 159 passed, 8 skipped, 29 xfailed in tests_dev (new file now 18 tests incl. sink accept/reject cases).

## .deb signature-of-content check (added)
tools/p2_packages_sha.py parses the repo `Packages.gz` (https://packages.microsoft.com/repos/edge/dists/stable/main/binary-amd64/); p2_edge_server.sh verifies each downloaded .deb against its SHA256 entry, refuses and deletes on mismatch or absent version (version skipped), and records `sha256_actual` and `sha256_repo` in MANIFEST.tsv. If Packages.gz cannot be fetched the run aborts. Tests: fixture parse, mismatch/absent refusal, CLI exit codes. tests_dev total after this: see last run (all pass). This checks integrity against the HTTPS-served index, not the GPG-signed Release (not done).
