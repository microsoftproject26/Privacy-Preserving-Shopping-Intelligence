# 04_on_device - export, quantisation, payloads and browser latency

## Purpose

The deployable side of the question: exporting models to ONNX, INT8 dynamic quantisation, the
communication byte ledger, and latency of the exported model in a real browser runtime on a
laptop, a phone and a server CPU. The on-device *adaptation* arms (device-only training,
fine-tuning a shipped model) are trained by the
[`03_federated_learning`](../03_federated_learning/README.md) engine and summarised here
because they answer the on-device question.

## Experiments, in order

1. **Prototype export and the INT8 reversal (August-September 2026).** ONNX export at opset 17
   matched PyTorch outputs to within 1.79e-7 after a dynamic-batch defect was fixed.
   Descriptive. Dynamic INT8 quantisation was accepted on the five-channel prototype (3.6662x
   smaller, 1.0709x faster at the median) and rejected on the real candidate (3.5111x smaller
   but slower: median 2.3623 ms against 1.8826 ms FP32), so the prototype shipped FP32. The
   byte ledger measured 800 bytes per transmission against an estimate of 544 (the estimate was
   32% low). Descriptive.
2. **Main-study device payloads (28 September - 1 October 2026).** The registered device
   payload cap is 170 MB per visit. Dense federated visits move 165,669,672 bytes (FP32); the
   frozen-item-table warm variant moves 2,110,584 bytes per visit plus a one-time download of
   81,779,544 bytes. Descriptive (measured).
3. **On-device arms in the confirmatory test evaluation (5-7 October 2026).** Confirmatory:
   device-only training keeps 2.10% [1.88, 2.34] of the matched cloud model (K8) - one user's
   history cannot learn a 158,486-product catalogue; fine-tuning the shipped pretrained model
   on each device adds +0.002283 [0.001588, 0.003076] MRR@20 (K9c), the one reliable deployable
   gain; INT8 against FP32 on the ONNX CPU run retains 99.98% [99.75, 100.23] (K16, on the
   registered fallback subset of 16,384 users).
4. **Browser latency of the tuned model (4-5 October 2026).** The tuned warm-start federated
   model exported at 82,863,741 bytes (FP32) and 21,243,628 bytes (INT8), about 3.9x smaller.
   In Microsoft Edge via WebAssembly, warm median latency is between 1.66 and 5.09 ms on the
   laptop, 2.19 and 3.70 ms on the phone and 2.66 and 4.56 ms on the server CPU; INT8 always
   starts faster (670 against 915 ms cold on the laptop; 2,573 against 5,168 ms on the phone)
   but is not uniformly faster warm. Descriptive.
5. **Closure latency pass (12-17 October 2026).** A single-thread CPU pass on a shared server:
   FP32 median 2.250102 ms, INT8 median 0.993833 ms. Descriptive; not a laptop or a phone and
   never read together with the browser numbers as one result.

## Code map

| Package | What it does | Origin |
|---|---|---|
| `code/ppsi/ondevice` | ONNX export (opset 17) and INT8 dynamic quantisation of the public package | Prototype stage (unchanged) |
| `code/ppsi/deployment` | Deployment package of the prototype: byte ledger, export verification, browser demo assets | Prototype stage (unchanged) |
| `code/v4dev` | Research-tree device tooling: `export.py`, `quantize.py`, `bench.py`, `scoring.py`, checkpoint IO | Main study |
| `code/v4dev/browser` | The Edge WebAssembly latency bench (`bench.js`, `p2_bench.js`, pages, local server) | Main study |
| `code/experiment_extensions/on_device` | Closure-phase device extensions (latency pass, cloud-model probe) | Closure phase |
| `scripts/prototype/deployment` | Prototype deployment scripts (export, quantisation benchmarks) | Prototype stage |
| `tests/prototype/{deployment,ondevice}`, `tests/research/tests_dev` | CPU tests and the research device suite | Prototype / main study |

## How to run

```bash
uv run python 04_on_device/run.py --from-results   # rebuild this stage's tables and figures from saved results
uv run python 04_on_device/run.py --mode auto      # per unit: skip if complete, resume if supported, else start
uv run python 04_on_device/run.py --force          # redo from scratch (old outputs are moved aside first)
```

Export and quantisation need a saved checkpoint (from the Release assets or a server run tree
under `PPSI_RUNS_ROOT`); the ONNX CPU scoring of K16 additionally needs the frozen dataset.
The browser bench needs a local COI/COOP server (`code/v4dev/browser/`) and a Chromium-based
browser; the latency tables in `results/` were measured on the devices named in the report.
`--from-results` rebuilds the tables without any of that.

## results/ and figures/

`results/` holds the curated device summaries, including `results/device_latency.csv`, and the
saved run records under `results/runs/main_study/` and `results/runs/closure_phase/`.
`figures/` holds the rebuilt payload and latency figures.

## Known limitations

Latency runs per cell are few (two or three on the laptop and phone), so the numbers describe
the setup and carry no inference. The headless browser on the server is not a consumer device.
No battery, energy or memory-over-session measurement was made. The prototype's INT8 quality
condition was never answered on a public evaluation set; the main study answered it only for
the registered fallback subset (K16). No recommender was deployed to real users.
