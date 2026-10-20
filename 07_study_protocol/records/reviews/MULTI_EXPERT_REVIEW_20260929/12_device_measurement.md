# 12 — Device deployment measurements (ONNX FP32/INT8, latency/memory, browser ONNX Runtime Web bench)

Expert: on-device ML / browser ML runtime reviewer (Claude Opus 5.5), multi-expert review 2026-09-29. READ-ONLY. I did not run
any script, test, page or bench. I did not contact the server, use git, or read any RESULT_ROWS / RETRO_TEST / ranks / BENCH_RESULT
file. Tags: **MEASURED** (read from a file, path:line; paths relative to `.local_ai/ppsi_v4/`), **REPORTED** (a number a project
document states), **INFERENCE**, **OPINION**. Literature is cited only where I checked it by web search in this session (§2).

---

## 1. What the project does (MEASURED)

### 1.1 The registered protocol
- **Amendment 04 §3** (`contracts/PREG1_AMENDMENT_04.md:34-35`): "ONNX: C_SMALL seed 2026, opset 17, onnxruntime CPU single-thread
  batch 1, latency median/p95 over 1,000 queries, peak RSS, FP32 and INT8 size, INT8 retention via T3, parity vs PyTorch."
- **Lock value T3** (`contracts/PREG1_LOCK_VALUES.json:67`): `onnx_device_study` = "SERVER-CPU PROXY deployment/latency only; parity
  receipt vs frozen evaluator on the WARM_TRUNCATION panel, labelled DEVICE_PARITY / HISTORICAL_REFERENCE; never a headline row".
- **06A §6** (`PREG1_AMENDMENT_06A.md:34-35`): INT8 retention rows on the full MAIN_VALIDATION "if the single-thread bench is ≤ 2 h
  (INFERENCE)". Otherwise the first 16,384 MAIN_VALIDATION users in population-selection-key order.
- **06B §2 K16** (`PREG1_AMENDMENT_06B.md:47`): K16 = R(INT8 / FP32) for C_SMALL is a *confirmatory* contrast ("on-device quantisation
  cost"). It is tertiary in the 06D §4 hierarchy (`PREG1_AMENDMENT_06D.md:74-78`). **06B §6.6** (`:88`): browser measurement (ONNX
  Runtime Web on the owner's PC) after the export, labelled HISTORICAL_REFERENCE. Ladder L1 = "C_SMALL shipped (ONNX FP32/INT8)" (`:54`).
- **06E §3** (`PREG1_AMENDMENT_06E.md:23-28`): K16 is computed inside an ONNX-CPU sub-block. Numerator and denominator are both
  onnxruntime CPU rows with the same thread count, opset, row subset, ORT version and optimisation level. The ratio against the A40
  PyTorch C_SMALL is DESCRIPTIVE (`DEVICE_MIXED`). The subset is the first 16,384 E2E MAIN_VALIDATION users in selection_key_v2
  order, with the set sha in rows.meta.json. K16 is single-seed 2026 (`:19`).
- **06D §8** (`PREG1_AMENDMENT_06D.md:97-100`): "C_SMALL INT8 (CPU)" is in the MUST tier. "browser ONNX (owner's PC)" is SHOULD-high.
  06I §8 keeps "browser" in SHOULD-high (`PREG1_AMENDMENT_06I.md:74`).
- **Disclosures already registered:** battery is not measured (`PREG1_AMENDMENT_06A.md:46`, item 9e). On-device training
  availability in a browser (C3) and the display-catalogue leak (C2) are listed (`PREG1_AMENDMENT_06I.md:70`).
- **Plan** (`company/plans/MASTER_PLAN_V5.md:161` §8.2, `:164` §8.5, `:193` §11.5, `:216` ch. 7.7): dump_rows → bench → browser_queries →
  the owner runs the page in Edge → BROWSER_LATENCY.json.

### 1.2 The code (`code/v4dev/`)
- **Export** (`export.py:55-104`): classic TorchScript exporter (`dynamo=False`), opset 17, `do_constant_folding=True`, no
  `dynamic_axes` (fully static batch 1 × L_MAX = 50). The export is traced with full-length rows so SASRec's data-dependent
  `T = lengths.max()` bakes to 50 (`scoring.py:158-167`). It is re-exported and must be byte-identical (`export.py:92-101`). The graph
  outputs the full `[1, K]` score vector, `q @ W.t() (+ b)`, i.e. the frozen T3 single block (`scoring.py:3-8`, `:142-147`).
- **Quantisation** (`quantize.py:165-174`): `quantize_dynamic(fp32, int8, weight_type=QuantType.QInt8)`. All other arguments are ORT
  defaults. ORT's own signature gives `per_channel=False`, `reduce_range=False`, activations QUInt8 and `MatMulConstBOnly=True`, and
  the dynamic-mode registry covers MatMul and Gather (verified on the ORT source, §2). So both 158,486 × 64 item tables are expected
  to be quantised per-tensor: the Gather input table and the MatMul output table (INFERENCE until the INT8 file size is seen).
- **Bench** (`bench.py:206-216`): CPUExecutionProvider, `intra_op_num_threads=1`, `inter_op=1`, ORT_SEQUENTIAL, ORT_ENABLE_ALL.
  - Latency (`bench.py:228-244`, `:323-333`): **1,000 synthetic queries** (`rows_to_batches`, `:109-113`), 50 warm-up (`:60`).
    Only `session.run` is timed. Median/p95. One pass, no repetition.
  - Peak RSS (`bench.py:247-269`): a fresh subprocess `v4dev._rss_worker`, measured with `ru_maxrss`.
  - Parity (`bench.py:341-373`): FP32 vs export wrapper and vs the production forward on synthetic queries, plus FP32/INT8 vs
    production on ≤ 1,000 real rows. Reported as max |Δ| and top-20 set identity.
  - K16 ranks (`bench.py:375-396`): FP32 and INT8 ranks over every row of the NPZ with the frozen `rank_from_scores`. The run records
    the ORT version, threads, optimisation level and hashes.
  - Every invocation **re-exports and re-quantises** (`bench.py:320-321`).
- **Rows dump** (`dump_rows.py:295-343`): it concatenates **every** provider batch field (`:319-327`) with no type filter.
- **Browser page** (`browser/index.html:96-98`, `browser/bench.js`):
  - onnxruntime-web 1.24.1 is pinned with SRI.
  - The run plan (`bench.js:147-163`) has three arms: wasm-1t, wasm-mt (only if `crossOriginIsolated`) and webgpu (if `navigator.gpu`).
  - Each arm does 20 warm-up and 1,000 timed calls (`bench.js:13-14`, `:194-209`). Median/p95/mean.
  - `load_ms` is the time of `InferenceSession.create` (`:182-185`).
  - Memory is `performance.memory.usedJSHeapSize`, read once after the loop (`:216`).
  - Top-20 parity is checked on the first 50 queries against a PyTorch-FP32 reference (`:121-136`; `browser_queries.py:96-106`).
  - The model sha is checked against BENCH_RESULT (`bench.js:230-241`).
  - The page runs on the main thread, and nothing is uploaded.

### 1.3 Facts that matter below
- **Server CPU = AMD EPYC 7H12 (Zen 2)**, a shared host (`company/RESOURCE_BOARD.md:8`). The G0 export smoke ran on "AMD EPYC"
  (`company/handoffs/EDGE-01.md`, latency table).
- **The owner/dev PC** has 20 logical CPUs and 15.7 GiB RAM (`company/RESOURCE_BOARD.md:15`). Local installs of onnxruntime are forbidden
  (`company/DECISION_LOG.md:11`, DL 7).
- **C_SMALL size**: 20,710,048 params ≈ 82.84 MB FP32 (REPORTED, `company/reviews/REALISM_AUDIT_20260929.md` §C1). The item tables are
  untied (`code/v4models/variants.py:15`, `:79-81`).
- **MAIN_VALIDATION** has 519,714 E2E rows, of which 515,636 are rankable (REPORTED, `company/reviews/REV-ANA-01_OPUS_REVIEW.md:297`).
- **No ONNX run of C_SMALL exists yet.** 06E states none exists (`PREG1_AMENDMENT_06E.md:7`). No DL row up to 225 records one
  (`company/DECISION_LOG.md`, rows 209–225). `company/gates/` holds no ANA_SCRIPT_FREEZE or ALL_METHODS_LOCK (directory listing).
  **K16 and every device value are therefore unread, so every recommendation below can still be registered before its governed value
  is read.**

---

## 2. State of the art (verified in this session)

**Runtimes and platform**
- **ONNX Runtime quantisation docs** (https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html):
  - "On x86-64 machines with AVX2 and AVX512 extensions, ONNX Runtime uses the VPMADDUBSW instruction for U8S8 … This instruction
    might suffer from saturation issues", with the fix "try reduce_range or the U8U8 format".
  - "it is recommended to use dynamic quantization for RNNs and transformer-based models".
  - "Per-channel quantization can improve the accuracy for models whose weight ranges are large … On AVX2 and AVX512 machines, you
    will generally need to enable reduce-range as well".
  - ORT source `quantize.py` (main): `quantize_dynamic(..., per_channel=False, reduce_range=False, weight_type=QInt8)`,
    `MatMulConstBOnly` defaults to True. `registry.py`: the IntegerOpsRegistry includes MatMul and Gather.
- **The server CPU**: Zen 2 has AVX2 but no AVX-512 and no VNNI (Wikipedia "Zen 2"/"Epyc" and a 7V12 cpuid dump, via search). This is
  the exact configuration the ORT saturation warning describes.
- **ONNX Runtime Web** (https://onnxruntime.ai/docs/tutorials/web/env-flags-and-session-options.html; WebGPU EP
  https://onnxruntime.ai/docs/execution-providers/WebGPU-ExecutionProvider.html; WebNN
  https://onnxruntime.ai/docs/tutorials/web/ep-webnn.html):
  - The EPs are wasm (default CPU), webgpu, webnn and webgl.
  - Multi-threaded wasm works only under cross-origin isolation (COOP/COEP). Otherwise it falls back to 1 thread.
  - WebNN is "not supported by default in browsers".
- **ORT training**:
  - ORT Web on-device training was introduced in 1.17.0 for "federated learning and developer exploration"
    (https://github.com/microsoft/onnxruntime/releases/tag/v1.17.0).
  - The ORT Training packages were deprecated in 1.20.0, and 1.19.2 was the last release
    (https://github.com/microsoft/onnxruntime/releases/tag/v1.20.0).
- **WebNN**: W3C Candidate Recommendation, updated 22 Jan 2026 (https://www.w3.org/TR/webnn/). It is in Chrome origin trial from
  Chrome 146 (Phoronix / Neowin news items). It is not a default production path in 2026 (REPORTED by secondary sources; OPINION agrees).
- **Edge built-in AI**:
  - The Prompt API with Phi-4-mini is a developer preview in Canary/Dev. It requires Windows 10/11 or macOS 13.3+, ≥ 20 GB free on
    the profile volume, ≥ 5.5 GB VRAM and an unmetered network.
  - Edge exposes a "Device performance class" at `edge://on-device-internals`.
  - Aion-1.0-Instruct (Edge ≥ 150.0.4070) adds a CPU-inference path for Medium/Low classes.
  - Sources: https://learn.microsoft.com/en-us/microsoft-edge/web-platform/prompt-api, and Edge blog 2026-06-02 (Sohum Chatterjee)
    https://blogs.windows.com/msedgedev/2026/06/02/expanding-on-device-ai-in-microsoft-edge-new-models-and-apis-for-the-web/.
  - **Real-system lesson:** Microsoft gates on-device models by a device class, and it budgets disk, VRAM and network conditions.
- **Chrome/Edge extensions (MV3)**:
  - An extension service worker terminates after 30 s idle, or when one event takes more than 5 min
    (https://developer.chrome.com/docs/extensions/develop/concepts/service-workers/lifecycle).
  - DOM/WebGPU work needs an offscreen document (https://developer.chrome.com/blog/Offscreen-Documents-in-Manifest-v3).
- **Browser memory**: `performance.measureUserAgentSpecificMemory()` requires a secure, cross-origin-isolated context
  (https://developer.mozilla.org/en-US/docs/Web/API/Performance/measureUserAgentSpecificMemory). A secondary source reports that
  `usedJSHeapSize` excludes WebAssembly linear memory (REPORTED, not from a spec; I treat it as likely).

**Measurement methodology**
- **Reddi et al., "MLPerf Mobile Inference Benchmark", MLSys 2022**
  (https://proceedings.mlsys.org/paper_files/paper/2022/hash/a2b2702ea7e682c5ea2c20e8f71efb0c-Abstract.html). Rules I read in the PDF:
  - single-stream reports the **90th-percentile latency over ≥ 1,024 samples and ≥ 60 s**;
  - room temperature 20–25 °C, ventilation, **cool-down 0–5 min between tests** (10 min between suite repeats);
  - runs on **battery** after a full charge;
  - "Power measurement is a major area of potential improvement" (power was not yet measured);
  - accuracy targets for INT8 are expressed relative to FP32.
- **Hoefler & Belli, "Scientific benchmarking of parallel computing systems: twelve ways to tell the masses when reporting performance
  results", SC 2015**, DOI 10.1145/2807591.2807644. It covers repeated measurements, non-parametric CIs of medians, and a full report
  of the environment.
- **Almeida et al., "Smart at what cost? Characterising Mobile Deep Neural Networks in the wild", IMC 2021**, DOI 10.1145/3487552.3487863.
  It measures energy on devices of different tiers and generations and treats energy as a core cost dimension.

**In-browser ML**
- **Ma et al., "Moving Deep Learning into Web Browser: How Far Can We Go?", WWW 2019**, DOI 10.1145/3308558.3313639.
- **Wang et al., "Anatomizing Deep Learning Inference in Web Browsers", ACM TOSEM 34(2), 2025**, DOI 10.1145/3688843 (arXiv 2402.05981).
  - Scope: 50 PC and 20 mobile devices.
  - In-browser inference is on average **16.9× slower on CPU and 4.9× slower on GPU than native** on PCs.
  - Memory can exceed **334.6× the model size**.
  - It adds 67.2 % to GUI render time.
- **Smilkov et al., "TensorFlow.js: Machine Learning for the Web and Beyond", SysML/MLSys 2019** (arXiv 1901.05350). It supports in-browser
  training.

**On-device recommendation, federated training and embedding compression**
- **Gong et al., "EdgeRec: Recommender System on Edge in Mobile Taobao", CIKM 2020** (arXiv 2005.08416). This is production on-device
  reranking.
- **Yao et al., "Device-Cloud Collaborative Learning for Recommendation", KDD 2021**, DOI 10.1145/3447548.3467097.
- **Hard et al., "Federated Learning for Mobile Keyboard Prediction", arXiv 1811.03604, 2018** (Gboard). It limits client CPU and battery
  use; the training eligibility conditions are REPORTED in the paper, and I did not re-read them.
- **Guan et al., "Post-Training 4-bit Quantization on Embedding Tables", NeurIPS 2019 MLSys workshop** (arXiv 1911.02079).
  - Row-wise uniform / codebook quantisation of recsys tables.
  - A production Facebook model at 13.89 % of FP32 size with neutral quality (REPORTED).

---

## 3. Comparison

### 3.1 Strengths (MEASURED)
1. **Engineering discipline is above typical MSc practice.**
   - Deterministic re-export check (`export.py:92-101`).
   - Parity against both the export wrapper and the *production* forward (`bench.py:341-345`).
   - Rows validated rather than clamped (`bench.py:127-154`).
   - ORT version, threads and optimisation level recorded (`bench.py:219-225`).
   - SRI-pinned CDN script and model-sha provenance refusal in the page (`bench.js:230-241`).
   - The server's ORT version is matched in the browser (1.24.1).
2. **K16 is defined correctly.** It is an ONNX-CPU/ONNX-CPU ratio with the frozen tie rule, never a runtime-mixed ratio
   (`PREG1_AMENDMENT_06E.md:24-26`; `bench.py:375-396`). OOV rows are handled as in v4eval (DEV-04).
3. **The page design is honest.**
   - Nothing is uploaded, and there is a file-picker only.
   - Skipped arms are recorded with a reason (`bench.js:147-163`).
   - Thread-count conflicts are detected and recorded, not silently mis-applied (`bench.js:166-181`).
   - The output is labelled HISTORICAL_REFERENCE.
4. **The full-K scoring is inside the graph** (`scoring.py:142-147`). The latency therefore includes the dominant 158,486 × 64 score
   GEMV and does not measure the encoder only. G0 found the SASRec-vs-GRU export issue early and designed around it (EDGE-01).
5. **The realism audit already names the key gaps** (C1–C3), and battery is registered as a disclosure (06A §9e).

### 3.2 Errors (MEASURED, with consequences)
- **E1 — `dump_rows` crash still open. The ONNX MUST step will fail at its first command.**
  - Finding (DL 209, `company/DECISION_LOG.md:214`; `company/handoffs/INT-07.md:186-189`): real StoreEvalProvider batches carry
    str/tuple/dict provenance fields. `dump_from_provider` concatenates every field (`dump_rows.py:319-327`) and fails with "zero-dimensional
    arrays cannot be concatenated".
  - The finding was listed as m12 for the "DEV owner" (`company/handoffs/FIX-DELTA-02.md:80`;
    `company/reviews/REV-DELTA-02_OPUS_REVIEW.md:143`, `:295`). The owner named there (eda-3b) closed at DL 214, and no later row
    assigns it.
  - `code/v4dev/dump_rows.py` is unchanged since 2026-09-29 11:45 local (≈ 08:45Z; sha256 `1bde41bd…22a1`). That is before the finding,
    so the deployed **code_ds3r3** (packed 16:19Z, DL 223) carries the unfixed file (INFERENCE from mtime vs pack time).
  - The TEST path uses a different wrapper (`v4int/dump_rows_retro.py:62`, `ArrayFieldsView`, array fields only). VALIDATION and TEST
    rows would therefore be produced by two different field filters unless the fix reuses it.
- **E2 — Peak RSS is not "the session's own footprint", contrary to the docstring.**
  - `_rss_worker.run_worker` imports `from .bench import make_session, peak_rss_bytes` (`_rss_worker.py:23`).
  - `bench.py` imports **torch at module top** (`bench.py:52`), plus `.checkpointio` → `v4models` and `.export`/`.scoring`
    (`bench.py:54-57`). The worker's `ru_maxrss` therefore includes the Python + numpy + onnxruntime + **torch (CUDA build 2.13+cu130)**
    + v4models import footprint.
  - For an 83 MB/21 MB model, that baseline can dominate the number (INFERENCE: a torch import is typically hundreds of MB RSS).
  - No baseline is subtracted, and ORT's arena/mmap behaviour is not separated.
  - As coded, "peak RSS" cannot support a memory guideline.
- **E3 — The INT8 configuration is the one ORT warns about on this exact CPU.**
  - U8S8 dynamic, per-tensor, `reduce_range=False` (`quantize.py:173`) is run on an AVX2 non-VNNI EPYC 7H12. ORT's docs name
    VPMADDUBSW saturation for exactly this case (§2).
  - K16 may therefore measure a **server-kernel artefact** (INFERENCE). The browser's wasm SIMD INT8 kernels are a different code
    path, so the server K16 may not transfer to the device it stands for.
  - Per-tensor scales on a 158,486-row output table also squeeze rare items with small norms. Recsys practice is row-wise
    (per-channel) scales (Guan et al. 2019).
  - The registered text says only "INT8" (`PREG1_AMENDMENT_04.md:35`). The mechanism is fixed only in code and in the DEV-01 packet
    (`company/tasks/DEV-01.md` item 2).
  - Nothing in BENCH_RESULT records the CPU model or flags (`bench.py:347-361`).
- **E4 — TEST-look model identity is not enforced.**
  - `run_pipeline` always re-exports and re-quantises (`bench.py:320-321`), and export determinism is checked only within one run.
  - I found no check that the TEST ONNX `model_sha256 {fp32,int8}` equals the VALIDATION run's. `grep model_sha256` in
    `tools/retention_report.py` finds it only as a copied field (L1360), and `tools/ana_block_index.py` has none.
  - The locked artefact at the single TEST look should be the *same file* (INFERENCE: a dependency drift between the two runs would
    silently change the INT8 model).
- **E5 — Documentation inconsistencies (minor).**
  - README claims "the same 20/1,000 discipline `v4dev.bench` uses on the server" (`browser/README.md:49-50`), but the server uses 50
    warm-up calls (`bench.py:60`).
  - The server bench times **synthetic** queries (`bench.py:323`), while the browser times **real** queries.json rows. With a static
    L = 50 graph the effect is small (INFERENCE), but it should be stated.
  - The lock's "parity receipt … on the WARM_TRUNCATION panel" (`PREG1_LOCK_VALUES.json:67`) is not what is implemented: parity runs on
    the MAIN rows NPZ. No v4dev or plan text binds WARM_TRUNCATION to ONNX (grep).
- **E6 — The browser `load_ms` of the first arm conflates two costs** (INFERENCE from ORT Web's lazy backend init): wasm
  download/compile from jsdelivr and session creation. FP32 wasm-1t always runs first (`bench.js:324-325`), so FP32 and INT8 load times
  are not comparable. The page also copies the model buffer per arm (`modelBuffer.slice(0)`, `:183`), which inflates memory.

### 3.3 Weaknesses and unrealistic choices (vs §2)
- **W1 — The proxy has little predictive value for the browser.** Native ORT single-thread on a server core is not a bound on
  Edge/wasm latency. TOSEM 2025 measured a 16.9× average CPU gap between browser and native on PCs. The browser page is the only
  representative measurement, yet it is SHOULD-high, one PC and HISTORICAL_REFERENCE. OPINION: for the proposal's "browser deployment
  guidelines" contribution, the page *is* the evidence, and the server numbers are a sanity/parity instrument.
- **W2 — Weak latency methodology** (MLPerf / Hoefler–Belli):
  - one pass of 1,000 queries;
  - no repetitions or CI of the median;
  - no p90 (the MLPerf single-stream statistic);
  - no ≥ 60 s floor;
  - no core pinning on a shared 256-CPU host running GPU jobs (DL 216–224: 11 GPU processes);
  - no record of CPU model, frequency or load;
  - no cool-down or thermal protocol in the browser;
  - no record of power source or Windows power mode;
  - reduced user agent only (`navigator.userAgent`, `bench.js:139`), which Chromium freezes. `userAgentData` high-entropy values are
    not captured.
- **W3 — Memory is not measured in the browser.** `usedJSHeapSize` (non-standard) is read once *after* the loop, not at peak. It likely
  misses wasm linear memory and GPU buffers. The standard `measureUserAgentSpecificMemory` needs COOP/COEP, which the plain `http.server`
  setup lacks (`browser/README.md:42-46`). TOSEM 2025 shows browser memory can be hundreds of times the model size, so this is the
  number a guideline needs.
- **W4 — Multi-thread and WebGPU are likely skipped or mis-attributed.**
  - Without COOP/COEP, wasm-mt is always skipped (README §4).
  - For INT8, WebGPU has no integer MatMul path, so nodes fall back to wasm (UNVERIFIED for 1.24.1; flagged in
    `REV-ANA-01_OPUS_REVIEW.md:305`). The page does not record node placement.
  - An "INT8 on WebGPU" latency would therefore be a mixed-EP number.
  - The natural GPU precision (FP16) is not tested.
- **W5 — What is timed is not the end-to-end recommendation.**
  - Top-k selection over 158,486 scores is outside the timed region on both server and browser (`bench.py:238-241`;
    `bench.js:200-204`; `top20Indices` does a full sort in JS).
  - The `[1, 158,486]` FP32 output (≈ 634 KB/query) must cross wasm→JS, or GPU→CPU on WebGPU.
  - Feature construction (history window, token lookup in the 158 k vocabulary) is not timed.
  - Cold start is not measured on the server: EDGE-01 measured first-load 0.9 s for C_BIG, but v4dev does not record session creation.
  - Download size is recorded as file bytes only: no compressed size, and no cache/OPFS path.
- **W6 — On-device training cost is absent from the device protocol.**
  - FT_C / FT_FA / LO / PF and every FL client assume a whole-model update in the browser: B16 × passes, fresh AdamW or SGD-m, full
    158 k softmax. Headline device contrasts K6–K11 depend on it.
  - Memory (INFERENCE): 83 MB params + 83 MB grads + 166 MB AdamW moments + logits 16 × 158,486 × 4 B ≈ 10 MB (×2 with grad) ≈ 0.35–0.4 GB.
  - Time per visit: not measured.
  - Toolchain: ORT-Web training was experimental in 1.17 and ORT training packages were deprecated in 1.20 (verified). TF.js is the
    maintained browser-training path (Smilkov 2019).
  - The MV3 service-worker limits (30 s idle / 5 min per event) constrain where a training step can run in an extension.
  - This is disclosed (06I C3) but never quantified.
- **W7 — Energy/battery is not measured.** The proposal names battery explicitly. MLPerf Mobile also did not measure power, but it
  fixed the battery and thermal conditions. A disclosure is registered (06A §9e). A cheap descriptive energy estimate is feasible (R8).
- **W8 — One device class.**
  - Only the owner's PC is measured (20 threads / 15.7 GiB, a high-end class; INFERENCE if the page is run on that PC).
  - There is no low-end laptop or phone. Edge's own built-in-AI program gates models by device performance class (§2).
  - Hybrid P/E-core CPUs make single-thread wasm latency depend on where the scheduler places the thread (OPINION; record the CPU model).
- **W9 — The "≤ 2 h" rule for full vs fallback rows is not operational.**
  - It is conditioned on a measurement that the bench only produces while scoring (`PREG1_AMENDMENT_06A.md:34`).
  - The DEV-04 command hard-codes `--first-users 16384` (`company/handoffs/DEV-04.md`, server commands).
  - 06E §3 reads as if the subset *is* the K16 rows.
  - INFERENCE: 519,714 rows × 2 sessions × a few ms ≈ 1–2 h single-thread, so the choice is genuinely borderline and should be fixed
    in writing before the run.
- **W10 — Ties in INT8.** Coarser INT8 scores raise the number of exact ties. The frozen rule `rank = 1 + gt + eq_before`
  (`code/v4eval/ranking.py:101`) breaks ties by catalogue index. BENCH_RESULT does not record `n_tied`, although `RankResult` exposes it.

### 3.4 Missing pieces (checklist from the brief)

| Item | Status |
|---|---|
| Model download size vs browser/extension limits | File bytes only. No compressed size, and no mention of extension package / storage quota (limits UNVERIFIED, §6) |
| WASM vs WebGPU vs WebNN | wasm-1t measured; wasm-mt skipped by default; WebGPU opportunistic, placement unrecorded; WebNN absent (acceptable: origin trial only) |
| Multi-thread / SharedArrayBuffer / COOP-COEP | Not served; README says so |
| Cold-start load time | Browser only, confounded (E6); server none |
| Full-K scoring cost | In graph (good); top-k and output transfer untimed |
| ANN index | Not needed: brute force is 10.1 M MAC/query at d = 64 (INFERENCE). Table memory, not search, is the constraint |
| Table memory | Not reported per component; RSS contaminated (E2) |
| INT8 method | Dynamic, per-tensor, U8S8, no reduce_range; not registered in text; CPU-dependent (E3) |
| On-device training cost | Absent (W6) |
| Battery/energy | Absent; disclosed (W7) |
| Low-end devices | Absent (W8) |
| Methodology (warm-up, variance, thermal) | Warm-up yes; variance/thermal/pinning no (W2) |
| DEV-04 `dump_rows` crash (DL 209) | **Not fixed** (E1) |

---

## 4. Recommendations

Cost is INFERENCE. "Prereg" means whether a registration (amendment or DL row) is needed before the governed value is read. The
governed values (K16, BENCH_RESULT, BROWSER_LATENCY) are all **unread**: no ONNX run exists (§1.3).

| ID | Item | Class | Action | Cost | Affects | Prereg needed / status |
|---|---|---|---|---|---|---|
| R1 | Fix `v4dev.dump_rows` to keep only numeric ndarray fields. Reuse the exact `ArrayFieldsView` filter of `v4int/dump_rows_retro.py:62` so VALIDATION and TEST NPZs have one schema. Add a test with str/tuple/dict fields. Run a server dry run on `--role MAIN` before §8.2. Assign an owner (m12 is orphaned since DL 214). Redeploy to the tree used for §8.2 | **MUST** | CHANGE | 1–2 h Sonnet + short review | Blocks the MUST-tier "C_SMALL INT8 (CPU)" and K16; nothing already run | No (bug fix, no rule change); DL row. Open |
| R2 | Make peak RSS mean what it says. The RSS worker must not import torch/v4models (move `make_session`/`peak_rss_bytes` into a torch-free module). Record `rss_after_imports`, `rss_after_session_create`, `peak_rss` and the delta. Add the file sizes of the two table initializers from the ONNX graph (weights vs runtime overhead) | **MUST** (for any memory claim) | CHANGE | 1–2 h | ONNX bench not yet run | No (definition unchanged: "peak RSS"); state the definition in a DL row before the run |
| R3 | Register the quantiser before the run: `quantize_dynamic`, weights QInt8, activations QUInt8, per-tensor, `reduce_range=False`, MatMul + Gather, ORT 1.24.1 → the K16 cell. Record `lscpu` model/flags (Zen 2, AVX2, no VNNI) in BENCH_RESULT. Add one DESCRIPTIVE sensitivity cell on the same rows: `per_channel=True, reduce_range=True` (ORT's own advice for AVX2). Report both; never swap them after reading | **MUST** | ADD | +1 INT8 file and +1 rank pass (≈ 0.5–1 h CPU) + 1 h code | ONNX not run | **Yes**: short amendment (06L-type) before the ONNX run. K16 unread → allowed. Open |
| R4 | Cross-runtime INT8 check: put the **server INT8** top-20 of the queries into `queries.json` (next to the FP32 reference). Have the page report browser-INT8 vs server-INT8 set identity. This shows whether K16 (server) transfers to the browser kernel (E3) | **MUST** (if K16 is presented as an *on-device* cost) | ADD | 1 h | browser_queries.py, bench.js | No (descriptive); freeze page sha before the owner run |
| R5 | Upgrade the browser protocol into the device evidence: | **MUST** for the "browser deployment guidelines" contribution (else downgrade the claim, R13) | CHANGE | 3 h Sonnet + 1–2 h owner | Browser run not done | No (HISTORICAL_REFERENCE, descriptive). Register the page sha + protocol in a DL row before the owner run |
| | (a) serve with COOP/COEP (tiny Python server) so wasm-mt and `measureUserAgentSpecificMemory` work; | | | | | |
| | (b) separate ORT/wasm init from session create (throwaway init, or time `ort.env` init) and time both models on fresh page loads; | | | | | |
| | (c) ≥ 3 repetitions per arm with a 2–5 min cool-down, reporting p50/p90/p95 and the spread across repetitions (MLPerf ≥ 1,024 queries, ≥ 60 s); | | | | | |
| | (d) record `userAgentData.getHighEntropyValues`, `hardwareConcurrency`, `deviceMemory`, power source/mode, and Edge's `edge://on-device-internals` device performance class (manual field); | | | | | |
| | (e) peak memory by `measureUserAgentSpecificMemory` + an Edge Browser Task Manager reading; | | | | | |
| | (f) an in-Worker arm (deployment runs off the main thread); | | | | | |
| | (g) for WebGPU, record node placement (ORT verbose log/profiling) and label INT8-WebGPU "mixed EP" | | | | | |
| R6 | End-to-end per-recommendation cost: time top-20 selection (argpartition server-side; a partial selection in JS) and, DESCRIPTIVE, an in-graph `TopK` export variant (20 ints out instead of 634 KB). Report output-transfer bytes, compressed (gzip/brotli) model sizes and server session-create time | SHOULD | ADD | 2 h | bench/page | No (descriptive) |
| R7 | Server latency hygiene: `taskset` to one idle core, record load average and co-running jobs, 5 repetitions × 1,024 queries (≥ 60 s), p50/p90/p95 plus a bootstrap CI of the median (Hoefler & Belli), and the CPU model | SHOULD | CHANGE | 1 h | bench | No; DL row with the protocol before the run |
| R8 | Energy, descriptive: on the owner's laptop (on battery, full charge, fixed brightness, MLPerf conditions), run a 10-min continuous inference loop vs 10-min idle and read the battery report/SRUM difference → J/query (coarse). Otherwise publish an explicit INFERENCE bound (≈ package-power × latency) labelled as such, next to the registered disclosure 06A §9e | SHOULD | ADD | owner 1 h + 1 h page tweak | Browser run | No (descriptive) |
| R9 | On-device training cost proxy: time and peak memory of ONE client visit (FT_C step and one FL client update; B16 × 2 passes; AdamW vs SGD-m) on one CPU thread in PyTorch, plus an analytic memory table. State the toolchain fact (ORT-Web training experimental 1.17 → training packages deprecated 1.20; TF.js maintained) and the MV3 30 s / 5 min limits. NICE: a TF.js timing of one visit on the owner's PC (audit C3) | SHOULD | ADD | 2–3 h script + CPU minutes | New descriptive artefact; no arm changes | No (descriptive; no quality value) |
| R10 | TEST-look identity: bench gets `--onnx-fp32/--onnx-int8` (use pre-built files), or the TEST run must assert `model_sha256` equal to the VALIDATION BENCH_RESULT's (and ORT version equal); ana_block_index checks it | SHOULD | CHANGE | 1 h | TEST look only | **Yes**: a DL/registration note before ALL_METHODS_LOCK (governs the TEST look; unread) |
| R11 | Fix the full-vs-fallback rule in writing before the run (W9). Either "the 16,384-user subset is the K16 row set" (06E wording), or project the time as 1,000-query median × 2 × 519,714 and apply "≤ 2 h" with a stated formula. The FP32 and INT8 passes may run as two parallel single-thread processes | SHOULD | CHANGE (clarify) | 0.5 h | K16 rows | **Yes** (clarification before the run). Open |
| R12 | Record `n_tied` per row (INT8 and FP32 ONNX) in the retention block; disclose the index tie-break | SHOULD | ADD | 0.5 h | bench output | No (descriptive) |
| R13 | Thesis wording: server numbers are a *parity/sanity* proxy, not a device estimate (cite TOSEM 2025's 16.9× CPU gap). Also state: static L = 50 / batch 1 graph; one device class; WebNN not evaluated (origin trial); extension MV3 constraints; the Edge Prompt-API hardware bar (5.5 GB VRAM, 20 GB disk) as the industry analogue of device classes; the lock-vs-implementation parity-panel difference (E5); README warm-up mismatch; synthetic (server) vs real (browser) queries | DISCLOSE-ONLY | KEEP + text | 1 h writing | Thesis ch. 7.7 / 9 | No |
| R14 | Second device class: the same page on the owner's phone (Edge Android) and/or an older laptop. If unavailable, a DevTools-CPU-throttled run labelled EMULATED | NICE | ADD | owner 30 min | Browser | No |
| R15 | Guideline analysis (no run): table memory FP32 2 × 40.6 MB → INT8 ≈ 2 × 10 MB → row-wise 4-bit ≈ 2 × 5 MB (Guan 2019); ANN not needed at 158 k × 64; on-demand fetching of input rows would leak viewed items; the display-catalogue leak (audit C2) | NICE | ADD | 1 h writing | Thesis | No |
| R16 | Keep: opset 17 static export, deterministic re-export check, production-forward parity, ONNX-CPU K16 sub-block, SRI/sha provenance, HISTORICAL_REFERENCE labelling, single-seed K16 | — | KEEP | 0 | — | Already registered |

**The top MUSTs in order:** R1 (without it the MUST-tier INT8 step cannot start), R3 (register the quantiser and record the CPU before
K16 exists), R2 (a memory number that is not dominated by torch), then R5 + R4 (the browser evidence the guidelines need).

---

## 5. Score: **5 / 10**

The software under this part is careful and well reviewed:
- deterministic export;
- parity against the real production forward;
- a correctly defined ONNX-CPU K16 block;
- SRI- and sha-bound browser provenance;
- honest labels.

As a *measurement* that can answer the proposal's "practical guidelines … memory, latency, battery", it is not yet adequate:
- The one representative instrument, the browser page, measures latency only, on one high-end PC, with multi-thread skipped by
  default, no valid memory number and a confounded load time.
- The server proxy has a peak-RSS number contaminated by the torch import.
- The INT8 configuration is the one ORT warns will saturate on this exact AVX2 non-VNNI CPU.
- On-device training cost is absent, although the headline device arms depend on it.
- Energy is absent.
- The first command of the MUST-tier ONNX step (`dump_rows --role MAIN`) is known to crash, and the fix has no owner.

All of this can be fixed cheaply, and nothing has been read yet, so no fix conflicts with a registered rule. With R1–R5 done the part
would merit about 7–8/10.

---

## 6. UNVERIFIED items / open questions
1. Whether ORT Web 1.24.1's wasm build includes `DynamicQuantizeMatMul`/`MatMulInteger` (so INT8 runs at all in the browser), and how
   the WebGPU EP places INT8 nodes. **Check on the owner's first run** (the page logs errors per arm).
2. Chrome Web Store / Edge Add-ons package size limits and the per-origin storage quota for caching an 83 MB model: not verified here;
   do not cite from memory.
3. Whether `usedJSHeapSize` excludes wasm linear memory: REPORTED by a secondary source only. `measureUserAgentSpecificMemory`'s COOP/COEP
   requirement is verified on MDN.
4. The actual VPMADDUBSW saturation impact on C_SMALL: unknown until R3's sensitivity cell exists (it may be negligible).
5. Whether the real MAIN StoreEvalProvider carries the same str/tuple/dict fields as the synthetic TEST provider. INT-07 states it as
   INFERENCE (`company/handoffs/INT-07.md:189`); an R1 dry run settles it.
6. Whether the page will be run on the 20-thread / 15.7 GiB dev PC listed in RESOURCE_BOARD, or on another owner device (affects the
   device-class statement).
7. Gboard/Hard et al. 2018 client-eligibility details (charging, unmetered, idle) are commonly cited. I did not re-read the paper's
   exact wording, so quote it only after checking.
