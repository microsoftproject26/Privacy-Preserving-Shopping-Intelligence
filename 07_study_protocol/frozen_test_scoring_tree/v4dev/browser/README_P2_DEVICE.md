# P2-14 device benchmark: owner steps (D1 laptop) and server note

Scope (owner decision O8, DL 415): D1 = the owner's laptop (Lenovo 82RC, i7-12700H, 15.8 GB, RTX 3050 Ti Laptop,
Windows 11 Pro, Microsoft Edge 154.0.4258.53). Server Edge versions are labelled SERVER_BROWSER (headless, server
CPUs, NOT a consumer device; see `tools/p2_edge_server.sh`). No Android / iPhone. Metrics are latency, cold start,
memory and model bytes only; nothing about accuracy. Results are never uploaded: the page downloads
`DEVICE_BENCH_P2.json` to your disk. Thresholds in PREG2 section 8 are DESCRIPTIVE.
The WebGPU arm listed for D1 in PREG2 section 8 is NOT in this page (wasm-1T and wasm-MT only). Gzip / brotli sizes are
computed by Ops offline from the .onnx files. Cold start here = fetch + session create + first inference; it excludes
Edge start-up and page load. Peak memory is not measured by the page (it takes one snapshot after the timed loop);
Ops adds peak evidence outside the page.

## What Ops gives you
Ops sends ONE ready-to-copy zip `p2bench_<pair>.zip` per model pair (PRE, P2_WSTAR, C_SMALL as agreed by Ops; each pair is
fp32 + int8, i.e. 6 runs per session: 2 models x 2 arms, plus repeats below). It contains `p2_bench.html`, `p2_bench.js`,
`serve_coi.py`, `queries.json`, `models\fp32.onnx`, `models\int8.onnx` and `vendor\` (the pinned onnxruntime-web 1.24.1
files: `ort.min.js` and the `ort-wasm*` files). You do not export or build anything.

## Steps
1. Unzip to `C:\p2bench\`. Install Python 3 if missing (python.org, tick "Add python.exe to PATH"). In PowerShell:
   `cd C:\p2bench` then `python serve_coi.py --root . --port 8765` (if `python` is not found, use `py serve_coi.py ...`).
   It only serves files on 127.0.0.1 (no firewall prompt expected). Stop it with Ctrl+C when finished. LAN / other
   machines are NOT supported (secure context needed for cross-origin isolation).
2. Close every other app (browser tabs, Teams, OneDrive sync; pause antivirus scans if possible). Plug in AC power,
   Windows power mode "Best performance", idle 2 minutes. In Edge turn off Efficiency mode and Sleeping tabs
   (edge://settings/system). Open `edge://version` and check it says 154.0.4258.53; write it down.
3. Open Edge in a fresh InPrivate window (no extensions) and go to (all four of these URLs differ only in `model` and `arm`):
   - `http://localhost:8765/p2_bench.html?d1=1&queries=queries.json&fp32=models/fp32.onnx&int8=models/int8.onnx&model=fp32&arm=wasm-1t&session=1`
   - `... &model=int8&arm=wasm-1t&session=1`
   - `... &model=fp32&arm=wasm-mt&threads=4&session=1`
   - `... &model=int8&arm=wasm-mt&threads=4&session=1`
   `d1=1` fills the D1 identity (id D1, laptop description, AC); without it those fields stay empty. ONE model and ONE arm
   per page load, so each cold start is genuinely cold: close the tab and open a new one for each URL.
4. The first log line must show `crossOriginIsolated=true`. If it says false, stop and tell Ops (you are not on
   serve_coi.py). Click "Run benchmark" (1000 timed queries after 20 warm-ups). Keep the tab in the foreground,
   not minimised; do not touch the machine until the log ends with `done` and "Download DEVICE_BENCH_P2.json" is enabled
   (expect a few minutes per run). Click it and rename the file `D1_<model>_<arm>_s<session>.json`.
5. Cool down 2 minutes between runs. Do the four runs above as one session. For sessions 2 and 3 (separate reboots or
   separate days) repeat with `session=2`, `session=3` and ROTATE the order (session 2: start with int8 1T; session 3: start
   with fp32 MT) so that no arm is always first. Wasm-MT on this hybrid P/E-core CPU is expected to vary more than 1T;
   that is not an error. Memory shows `n/a` where `measureUserAgentSpecificMemory` is unavailable; that is expected.
6. Send all JSON files (4 per session x 3 sessions per model pair) to Ops through the agreed project transfer channel
   (the same channel Ops used for the zip; ask Ops if unsure). We report the median over sessions and the range.

## Optional: energy proxy on Windows (OPTIONAL, coarse; do not block on it)
1. Unplug, battery between 80 and 95 percent, same closed-apps setup.
2. `powercfg /batteryreport /output C:\p2bench\before.html`
3. Run the benchmark in a loop for 30 minutes on battery (reload the page repeatedly).
4. `powercfg /batteryreport /output C:\p2bench\after.html`
5. Report the drop in mWh between the reports and the number of inferences done. Battery report granularity is
   coarse; label any number INFERENCE-grade and do not read it as a measurement of the model alone.

## Server (SERVER_BROWSER)
`tools/p2_edge_server.sh --site <dir> --out <dir>` prints the plan; add `--run` to execute. Per run: one headless Edge
launch (own process group, killed afterwards), one model, one arm; wasm-mt gets `threads=<number of pinned CPUs>`;
results are posted to a loopback-only sink (`serve_coi.py --results-dir`) with the console marker as fallback;
`MANIFEST.tsv` lists versions and .deb sha256.
