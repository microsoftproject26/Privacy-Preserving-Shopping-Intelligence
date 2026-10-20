"use strict";
/* browser/bench.js: the browser latency page's logic. Loads onnxruntime-web (pinned version, <script> tag in
 * index.html: the only external host this page talks to), lets the user pick model_fp32.onnx / model_int8.onnx and a
 * queries.json (built by ppsi.ondevice.browser_queries), runs WARMUP + N_TIMED single-query (batch 1) inferences per
 * execution provider, and produces / downloads BROWSER_LATENCY.json (execution provider, model sha256 via
 * SubtleCrypto, session load time, median / p95 / mean latency, JS heap if available, navigator.userAgent, top-20 parity of
 * the first PARITY_N queries against the reference).
 *
 * No networking of its own: model and queries files are read from local <input type=file> selections only (File API),
 * never uploaded anywhere. This script issues zero HTTP requests itself.
 */

const WARMUP = 20;
const N_TIMED = 1000;
const PARITY_N = 50;
const DEFAULT_TOP_K = 20;

let lastResultDoc = null;
// ort.env.wasm.numThreads only takes effect when the FIRST wasm session of a page load is created: the first
// wasm arm fixes it; a later wasm arm asking for a different count is skipped (reload the page and
// pick it with ?arm=wasm-mt) instead of silently running with the wrong thread count.
let firstWasmThreads = null;

// --------------------------------------------------------------------------------------------------------- logging
function log(msg) {
  const el = document.getElementById("log");
  const line = `[${new Date().toISOString().slice(11, 19)}] ${msg}`;
  el.textContent += (el.textContent ? "\n" : "") + line;
  el.scrollTop = el.scrollHeight;
  // eslint-disable-next-line no-console
  console.log(line);
}

// ------------------------------------------------------------------------------------------------------- utilities
async function sha256Hex(arrayBuffer) {
  const digest = await crypto.subtle.digest("SHA-256", arrayBuffer);
  return Array.from(new Uint8Array(digest)).map((b) => b.toString(16).padStart(2, "0")).join("");
}

function getPinnedOrtVersion() {
  const el = document.querySelector('script[src*="onnxruntime-web@"]');
  if (!el) return null;
  const m = el.src.match(/onnxruntime-web@([^/]+)/);
  return m ? m[1] : null;
}

function flattenNested(x, out) {
  out = out || [];
  if (Array.isArray(x)) {
    for (const v of x) flattenNested(v, out);
  } else {
    out.push(x);
  }
  return out;
}

function makeTypedArray(dtype, flat) {
  switch (dtype) {
    case "int64": return BigInt64Array.from(flat, (v) => BigInt(Math.trunc(v)));
    case "uint64": return BigUint64Array.from(flat, (v) => BigInt(Math.trunc(v)));
    case "int32": return Int32Array.from(flat);
    case "uint32": return Uint32Array.from(flat);
    case "int16": return Int16Array.from(flat);
    case "uint16": return Uint16Array.from(flat);
    case "int8": return Int8Array.from(flat);
    case "uint8": return Uint8Array.from(flat);
    case "bool": return Uint8Array.from(flat, (v) => (v ? 1 : 0));
    case "float32": return Float32Array.from(flat);
    case "float64": return Float64Array.from(flat);
    default: throw new Error(`unsupported dtype ${dtype} (queries.json and bench.js disagree)`);
  }
}

/** One row's feed, restricted to the ONNX session's OWN declared input names (mirrors the Python bench's
 * onnx_input_names narrowing: exported graphs drop e.g. attention_mask/position_ids that are never read as real
 * tensor data). Refuses (does not silently zero-fill) a graph input queries.json never carried. */
function buildFeed(queries, rowIndex, inputNames) {
  const feed = {};
  for (const name of inputNames) {
    if (!(name in queries.inputs)) {
      throw new Error(`queries.json has no "${name}" (a declared input of this ONNX graph)`);
    }
    const shape = [1, ...(queries.shapes[name] || [])];
    const flat = flattenNested(queries.inputs[name][rowIndex]);
    feed[name] = new ort.Tensor(queries.dtypes[name], makeTypedArray(queries.dtypes[name], flat), shape);
  }
  return feed;
}

/** numpy `np.percentile(x, p, method="linear")`-equivalent, so this matches the Python bench's latency_stats exactly. */
function percentile(sortedAsc, p) {
  const n = sortedAsc.length;
  if (n === 0) throw new Error("no timed queries");
  if (n === 1) return sortedAsc[0];
  const pos = (p / 100) * (n - 1);
  const lo = Math.floor(pos);
  const hi = Math.ceil(pos);
  const frac = pos - lo;
  return sortedAsc[lo] + (sortedAsc[hi] - sortedAsc[lo]) * frac;
}

function latencyStats(msArray) {
  const sorted = msArray.slice().sort((a, b) => a - b);
  const mean = msArray.reduce((a, b) => a + b, 0) / msArray.length;
  return { median_ms: percentile(sorted, 50), p95_ms: percentile(sorted, 95), mean_ms: mean,
          n_queries: msArray.length };
}

function top20Indices(scores, k) {
  const idx = Array.from(scores, (_, i) => i);
  idx.sort((a, b) => scores[b] - scores[a]);
  return idx.slice(0, k).sort((a, b) => a - b);        // ascending, set-comparison-ready
}

function sameSet(a, b) {
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) return false;
  return true;
}

/** Top-k SET identity rate of `capturedScores` (this session's first PARITY_N outputs) vs `reference.top20` (from
 * browser_queries' PyTorch CPU path) -- a set, not an ordered list, exactly like the Python bench's parity_stats: a
 * tie can legally reorder an argsort without changing the retrieved set. */
function parityVsReference(capturedScores, reference) {
  if (!reference || !reference.available || !reference.top20 || !reference.top20.length) return null;
  const topK = reference.top_k || DEFAULT_TOP_K;
  const n = Math.min(capturedScores.length, reference.top20.length);
  if (n === 0) return null;
  let identical = 0;
  for (let i = 0; i < n; i++) {
    const top = top20Indices(capturedScores[i], topK);
    const ref = reference.top20[i].slice(0, topK).sort((a, b) => a - b);
    if (sameSet(top, ref)) identical++;
  }
  return { n_rows: n, top_k: topK, top_k_identity_rate: identical / n };
}

function detectEnvironment() {
  return { crossOriginIsolated: !!self.crossOriginIsolated, webgpuAvailable: !!navigator.gpu,
          userAgent: navigator.userAgent, jsHeapAvailable: !!(performance && performance.memory) };
}

function selectedArm() {
  try { return new URLSearchParams(location.search).get("arm"); } catch (e) { return null; }
}

function epPlan(env) {
  const plan = [{ id: "wasm-1t", label: "wasm (1 thread)", providers: ["wasm"], numThreads: 1, skipReason: null }];
  plan.push({ id: "wasm-mt", label: "wasm (multi-thread)", providers: ["wasm"],
             numThreads: Math.max(2, navigator.hardwareConcurrency || 4),
             skipReason: env.crossOriginIsolated ? null :
               "cross-origin isolation not available (self.crossOriginIsolated is false — plain "
               + "`python -m http.server` does not send COOP/COEP headers; use serve_coi.py)" });
  plan.push({ id: "webgpu", label: "webgpu", providers: ["webgpu"], numThreads: null,
             skipReason: env.webgpuAvailable ? null : "navigator.gpu is not available in this browser/PC" });
  const arm = selectedArm();
  if (arm) {
    for (const ep of plan) {
      if (ep.id !== arm && !ep.skipReason) ep.skipReason = `not selected (page opened with ?arm=${arm})`;
    }
  }
  return plan;
}

// ----------------------------------------------------------------------------------------------------- the bench
function threadConflict(ep) {
  if (ep.numThreads === null || firstWasmThreads === null || firstWasmThreads === ep.numThreads) return null;
  return `numThreads=${ep.numThreads} cannot be applied: this page load already created a wasm session with `
    + `numThreads=${firstWasmThreads} (ort.env.wasm.numThreads is read only at the first session). Reload the page `
    + `with ?arm=${ep.id} to run this arm`;
}

async function benchOneEP(modelBuffer, ep, queries) {
  const conflict = threadConflict(ep);
  if (conflict) return { skipped: true, reason: conflict };
  if (ep.numThreads !== null) {
    if (firstWasmThreads === null) {
      ort.env.wasm.numThreads = ep.numThreads;          // BEFORE the first wasm session of this page load
      firstWasmThreads = ep.numThreads;
    }
  }
  const t0 = performance.now();
  const session = await ort.InferenceSession.create(modelBuffer.slice(0),
    { executionProviders: ep.providers, graphOptimizationLevel: "all" });
  const loadMs = performance.now() - t0;
  const appliedThreads = (ep.numThreads !== null) ? ort.env.wasm.numThreads : null;   // recorded per run

  const inputNames = session.inputNames;
  const nRows = queries.n_rows;
  if (nRows < 1) throw new Error("queries.json has no rows");
  const feeds = [];
  for (let i = 0; i < nRows; i++) feeds.push(buildFeed(queries, i, inputNames));

  for (let i = 0; i < WARMUP; i++) {
    await session.run(feeds[i % nRows]);
  }

  const times = new Array(N_TIMED);
  const captured = [];
  for (let i = 0; i < N_TIMED; i++) {
    const feed = feeds[i % nRows];
    const t1 = performance.now();
    const out = await session.run(feed);
    times[i] = performance.now() - t1;
    if (i < PARITY_N) {
      const outTensor = out.scores || Object.values(out)[0];
      captured.push(Array.from(outTensor.data));
    }
  }

  const stats = latencyStats(times);
  stats.warmup = WARMUP;
  stats.load_ms = loadMs;
  stats.wasm_num_threads_requested = ep.numThreads;
  stats.wasm_num_threads_applied = appliedThreads;
  stats.js_heap_bytes = (performance.memory && performance.memory.usedJSHeapSize) || null;
  stats.parity_vs_reference = parityVsReference(captured, queries.reference);

  if (typeof session.release === "function") {
    try { await session.release(); } catch (e) { /* best-effort cleanup */ }
  }
  return stats;
}

async function benchModel(label, file, epPlanList, queries) {
  if (!file) return { provided: false };
  log(`${label}: reading ${file.name} (${file.size} bytes)`);
  const buffer = await file.arrayBuffer();
  const sha256 = await sha256Hex(buffer);
  const expected = queries.provenance && queries.provenance[`model_${label}_sha256`];
  let queriesShaMatch = null;                       // null = queries.json carried no model sha to check against
  if (expected) {
    queriesShaMatch = (expected === sha256);
    if (!queriesShaMatch) {
      throw new Error(`${label}: ${file.name} sha256 ${sha256} != the model sha256 ${expected} that queries.json `
        + `(the bench result) was built for; refusing to benchmark a different model`);
    }
    log(`${label}: sha256 matches the bench result (${sha256.slice(0, 12)}…)`);
  } else {
    log(`${label}: no model sha256 in queries.json provenance — not verified against the bench result`);
  }
  const runs = {};
  for (const ep of epPlanList) {
    if (ep.skipReason) {
      log(`${label} / ${ep.label}: skipped (${ep.skipReason})`);
      runs[ep.id] = { skipped: true, reason: ep.skipReason };
      continue;
    }
    log(`${label} / ${ep.label}: loading session…`);
    try {
      runs[ep.id] = await benchOneEP(buffer, ep, queries);
      log(`${label} / ${ep.label}: median ${runs[ep.id].median_ms.toFixed(3)} ms, `
        + `p95 ${runs[ep.id].p95_ms.toFixed(3)} ms (load ${runs[ep.id].load_ms.toFixed(1)} ms)`);
    } catch (err) {
      log(`${label} / ${ep.label}: ERROR ${err.message}`);
      runs[ep.id] = { error: String(err && err.message ? err.message : err) };
    }
  }
  return { provided: true, sha256, bytes: buffer.byteLength, queries_sha_match: queriesShaMatch, runs };
}

function renderResults(doc) {
  const container = document.getElementById("results");
  const rows = [];
  for (const [modelLabel, modelRec] of Object.entries(doc.models)) {
    if (!modelRec.provided) continue;
    for (const [epId, run] of Object.entries(modelRec.runs)) {
      if (run.skipped || run.error) {
        rows.push(`<tr><td>${modelLabel}</td><td>${epId}</td><td colspan="6">${run.error ? "ERROR: " + run.error
          : "skipped: " + run.reason}</td></tr>`);
        continue;
      }
      const parity = run.parity_vs_reference
        ? `${(run.parity_vs_reference.top_k_identity_rate * 100).toFixed(1)}% (n=${run.parity_vs_reference.n_rows})`
        : "n/a";
      rows.push(`<tr><td>${modelLabel}</td><td>${epId}</td><td>${run.load_ms.toFixed(1)}</td>`
        + `<td>${run.median_ms.toFixed(3)}</td><td>${run.p95_ms.toFixed(3)}</td><td>${run.mean_ms.toFixed(3)}</td>`
        + `<td>${run.js_heap_bytes !== null ? (run.js_heap_bytes / 1e6).toFixed(1) : "n/a"}</td><td>${parity}</td></tr>`);
    }
  }
  container.innerHTML = `<table><thead><tr><th>model</th><th>EP</th><th>load ms</th><th>median ms</th>
    <th>p95 ms</th><th>mean ms</th><th>JS heap MB</th><th>top-20 parity</th></tr></thead>
    <tbody>${rows.join("")}</tbody></table>`;
}

async function runBenchmark() {
  const runButton = document.getElementById("runButton");
  const downloadButton = document.getElementById("downloadButton");
  runButton.disabled = true;
  downloadButton.disabled = true;
  document.getElementById("log").textContent = "";
  document.getElementById("results").innerHTML = "<p class=\"note\">running…</p>";
  try {
    const fp32File = document.getElementById("fp32File").files[0];
    const int8File = document.getElementById("int8File").files[0];
    const queriesFile = document.getElementById("queriesFile").files[0];
    if (!fp32File || !queriesFile) {
      throw new Error("model_fp32.onnx and queries.json are both required");
    }
    log(`onnxruntime-web version (pinned in index.html): ${getPinnedOrtVersion()}`);
    log(`reading ${queriesFile.name}…`);
    const queries = JSON.parse(await queriesFile.text());
    if (!queries || !queries.inputs || !queries.n_rows) {
      throw new Error("queries.json does not look like a browser_queries document");
    }
    log(`queries.json: ${queries.n_rows} rows, reference available = `
      + `${!!(queries.reference && queries.reference.available)}`);

    const env = detectEnvironment();
    log(`crossOriginIsolated=${env.crossOriginIsolated}, webgpuAvailable=${env.webgpuAvailable}, `
      + `jsHeapAvailable=${env.jsHeapAvailable}`);
    const plan = epPlan(env);

    const doc = {
      schema: "ppsi.ondevice.browser_latency.v1", created_at: new Date().toISOString(), user_agent: env.userAgent,
      cross_origin_isolated: env.crossOriginIsolated, webgpu_available: env.webgpuAvailable,
      onnxruntime_web_version: getPinnedOrtVersion(),
      queries: { schema: queries.schema || null, n_rows: queries.n_rows, n_warmup: WARMUP, n_timed: N_TIMED,
                reference_available: !!(queries.reference && queries.reference.available),
                provenance: queries.provenance || null },
      arm: selectedArm(),
      models: {},
    };
    doc.models.fp32 = await benchModel("fp32", fp32File, plan, queries);
    doc.models.int8 = await benchModel("int8", int8File, plan, queries);

    lastResultDoc = doc;
    renderResults(doc);
    downloadButton.disabled = false;
    log("done — click \"Download BROWSER_LATENCY.json\" to save the result.");
  } catch (err) {
    log(`FATAL: ${err && err.message ? err.message : err}`);
    document.getElementById("results").innerHTML = `<p class="note">run failed: ${err && err.message ? err.message
      : err}</p>`;
  } finally {
    runButton.disabled = false;
  }
}

function downloadResult() {
  if (!lastResultDoc) return;
  const blob = new Blob([JSON.stringify(lastResultDoc, null, 1)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = "BROWSER_LATENCY.json";
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}

document.addEventListener("DOMContentLoaded", () => {
  document.getElementById("runButton").addEventListener("click", runBenchmark);
  document.getElementById("downloadButton").addEventListener("click", downloadResult);
});
