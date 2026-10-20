# On-device inference benchmark (`ppsi.ondevice`)

The on-device arm of the deployment study runs the recommender in the browser, so its latency, memory and accuracy
have to be measured on the same model that would ship. `ppsi.ondevice` exports a `ppsi.seqrec` model to ONNX,
quantises it to INT8, benchmarks both under onnxruntime, and provides two static pages that run the same files under
ONNX Runtime Web.

## Pipeline

| Step | Module | What it guarantees |
|---|---|---|
| Scoring wrapper | `scoring.ScoringModule` | one positional tensor per model input, `[B, K]` scores = the model's own logits over the whole catalogue (the evaluator's single-block scoring); the GRU uses a pack-free equivalent of its query because ONNX cannot trace "pack without unpack" |
| Export | `export.export_model` | opset 17, fixed batch 1 (static shapes), classic TorchScript exporter; every export is re-exported and must be byte-identical |
| INT8 | `quantize.quantize_int8` | weight-only dynamic INT8 (`onnxruntime.quantization.quantize_dynamic`), sizes and sha256s recorded |
| Bench | `bench.run_pipeline` | onnxruntime CPU, one thread, batch 1: median / p95 latency after warm-up, peak RSS of each session in its own subprocess, FP32 parity against both the traced wrapper and the production forward (max abs delta, top-20 set identity) |
| Retention (optional) | `bench.ranks_from_session` | FP32 and INT8 ranks of real decisions (rows NPZ) as full-manifest int32 rank arrays for `ppsi.evaluation.fullrank` |

```powershell
uv run python -m ppsi.ondevice.bench --family SASREC --variant SASREC_D64_B2 --k 1000 --out out/bench
# with trained weights and real decisions:
uv run python -m ppsi.ondevice.bench --family SASREC --variant SASREC_D64_B2 --catalogue catalogue.parquet `
    --ckpt model.pt --rows rows.npz --manifest-n 120000 --out out/bench
```

`--ckpt` takes a bare state dict or a `{"model": state_dict, "meta": {...}}` record; `--widths` sets the four float
input widths (default `15,15,12,4`). The command writes `model_fp32.onnx`, `model_int8.onnx` and `BENCH_RESULT.json`.
A rows NPZ holds one array per model input key plus `decision_id`, `target_class` and `manifest_row`; rows with a
length below 1 or a padding that is not a contiguous prefix are refused, and OOV targets are left unscored (rank -1).

## Browser pages

`ppsi/ondevice/browser/` contains two static pages (no build step; onnxruntime-web is pinned to an exact version with
a sha384 integrity attribute and is the only external resource):

* `index.html` + `bench.js`: pick `model_fp32.onnx`, `model_int8.onnx` and a `queries.json`; runs warm-up plus 1,000
  single-query inferences per execution provider (wasm 1 thread, wasm multi-thread, WebGPU when available), checks
  each model's sha256 against the bench result, reports load time, median / p95 / mean latency, JS heap and top-20
  parity against the PyTorch reference, and downloads `BROWSER_LATENCY.json`.
* `device_bench.html` + `device_bench.js`: latency, cold start (fetch, session create, first inference) and memory
  only, one model per page load, for consumer devices; downloads `DEVICE_BENCH.json`.

Nothing is uploaded. Build the queries and serve the directory with the cross-origin-isolation headers that
multi-threaded wasm needs:

```powershell
uv run python -m ppsi.ondevice.browser_queries --family SASREC --variant SASREC_D64_B2 --k 1000 `
    --bench-result out/bench/BENCH_RESULT.json --out out/bench/queries.json
python ppsi/ondevice/browser/serve_coi.py --root ppsi/ondevice/browser --port 8765
```

Then open `http://localhost:8765/index.html` (or `device_bench.html?fp32=...&int8=...&queries=...&model=fp32&arm=wasm-1t`
with the files copied next to the pages). `serve_coi.py` binds 127.0.0.1, is read-only, and only accepts results on
an opt-in loopback sink (`--results-dir`).

## Tests

```powershell
uv run python -m pytest tests/ondevice -q
```

The export, INT8 and bench tests run real onnx / onnxruntime on tiny synthetic models; the page tests are static
checks plus a live `serve_coi.py`. Tests named `test_nc_*` are negative controls that must fail.
