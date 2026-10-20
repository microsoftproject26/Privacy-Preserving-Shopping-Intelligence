# DEV-03 — browser latency page (ONNX Runtime Web), C_SMALL, HISTORICAL_REFERENCE

Measures single-query (batch 1) latency of the exported C_SMALL ONNX models **in a real browser**, on the owner's
own PC, as a HISTORICAL_REFERENCE point beside the server (`v4dev.bench`) numbers — never a substitute for them
(contracts `PREG1_AMENDMENT_06B.md` §6 order item 6). No npm, no build step, no venv install: this folder is three
static files (`index.html`, `bench.js`, `README.md`) plus one you generate (`queries.json`); the only network
resource the page loads is `onnxruntime-web` from `cdn.jsdelivr.net`, pinned to an exact version in `index.html`.

## 1. Get the three inputs onto this PC
1. `model_fp32.onnx` and `model_int8.onnx` — from Ops's `v4dev.bench` / `v4dev.export` run for C_SMALL
   (`<out>/model_fp32.onnx`, `<out>/model_int8.onnx`).
2. `queries.json` — built from the DEV-02 rows NPZ with:
   ```
   python -B -m v4dev.browser_queries --rows <rows.npz> --out queries.json \
       --ckpt <C_SMALL checkpoint> --family GRU|SASREC [--variant NAME] --v1-root <path>
   ```
   (Ops runs this on a machine with the real checkpoint + catalogue; `--ckpt`/`--family` are optional — omit them
   for a latency-only `queries.json` with no reference top-20, e.g. before a checkpoint exists.)

These three files do **not** need to sit in this folder — the page opens each one through a normal file-picker
dialog, so they can stay wherever Ops copied them.

## 2. Serve this folder (do not double-click index.html)
`onnxruntime-web`'s WASM loader requires an http(s) origin, not `file://`. From a terminal, `cd` into
`code/v4dev/browser/` and run:
```
python -m http.server 8000
```
Leave this running for the duration of the measurement.

## 3. Open the page in Microsoft Edge
1. Open Microsoft Edge and navigate to `http://localhost:8000/`.
2. Under **Inputs**, click each file field and pick, from wherever you put them in step 1:
   `model_fp32.onnx`, `model_int8.onnx` (optional — leave blank to bench FP32 only), `queries.json`.
3. Click **Run benchmark**. The **Log** panel prints each model / execution-provider pair as it runs; the
   **Results** table fills in below as each one finishes.
4. Click **Download BROWSER_LATENCY.json** and save it — this is the file Ops attaches next to the ONNX artifacts
   (`<out>/browser/BROWSER_LATENCY.json`).

## 4. What to expect
- **wasm (1 thread)** always runs.
- **wasm (multi-thread)** needs cross-origin isolation (`SharedArrayBuffer`), which requires
  `Cross-Origin-Opener-Policy` / `Cross-Origin-Embedder-Policy` response headers. Plain `python -m http.server`
  (step 2) does **not** send these, so this row will normally show "skipped" — that is expected, not a bug; the
  page still records `cross_origin_isolated: false` in the output so this is auditable. (If Ops wants this arm too,
  serve with a small wrapper that adds those two headers instead of stdlib `http.server`, then re-run.)
- **webgpu** runs only if this PC + Edge build expose `navigator.gpu` (recent Edge on a machine with a supported
  GPU/driver); otherwise it shows "skipped" with the reason.
- Each run is exactly 20 untimed warm-up calls + 1,000 timed calls per (model × execution provider) — the same
  20/1,000 discipline `v4dev.bench` uses on the server, so the two are comparable.
- Top-20 parity (first 50 queries vs `queries.json`'s reference) only appears when `queries.json` was built with
  `--ckpt`/`--family` (step 1.2); otherwise it reads "n/a" — there is no ground truth to compare against.
- Nothing on this page uploads anything anywhere: models and queries are read from local file-picker selections
  (browser File API) and the result is written to a `Blob` you download yourself.

## 5. If jsdelivr does not have the pinned onnxruntime-web version
`index.html`'s one `<script src="https://cdn.jsdelivr.net/npm/onnxruntime-web@X.Y.Z/dist/ort.min.js">` tag pins an
exact version (matching the server's `onnxruntime` 1.24.1, `company/RESOURCE_BOARD.md`); this was not independently
verified against the live CDN from the offline dev sandbox this page was written in (INFERENCE). If Edge reports the
script failed to load, open that URL directly to see jsdelivr's available `onnxruntime-web` tags, substitute the
closest published version in that one line, and record the substitution in the run's notes — `bench.js` reads the
version back out of this same tag (`getPinnedOrtVersion()`), so there is only one place to edit.

## 6. Integrity, threads and provenance (REV-ANA-01 D-F3)
- **SRI.** `index.html`'s onnxruntime-web `<script>` carries `integrity="sha384-..."` + `crossorigin="anonymous"`, computed
  from the pinned file (`ort.min.js`, v1.24.1, 358,028 bytes). If you change the version in step 5 you MUST recompute it:
  `curl -sSL <the exact URL> | openssl dgst -sha384 -binary | openssl base64 -A`, then write `sha384-<output>`. A browser
  refuses the script on a mismatch (the page then reports `ort is not defined`). SRI covers `ort.min.js` only; the `.wasm`
  files it loads from the same pinned path cannot carry SRI.
- **Threads.** `ort.env.wasm.numThreads` only takes effect at the first wasm session of a page load. The first wasm arm
  fixes it; a later wasm arm with a different count is reported as skipped. To run the multi-thread arm open
  `http://localhost:8000/?arm=wasm-mt` (fresh load; needs cross-origin isolation, step 4); `?arm=wasm-1t` / `?arm=webgpu`
  select the others. Every run records `wasm_num_threads_requested` and `wasm_num_threads_applied`.
- **Provenance.** Build queries.json with `--bench-result <out>/BENCH_RESULT.json` (and `--ckpt`): it then carries the
  `ckpt_sha256`, `rows_npz_sha256` and the FP32/INT8 model sha256 of that bench run; the page hashes each picked model and
  REFUSES one that differs (`queries_sha_match` is recorded; `null` = queries.json had no sha to check). The rows-NPZ sha
  binds the file used, not reproducible content (`np.savez` zip timestamps).
