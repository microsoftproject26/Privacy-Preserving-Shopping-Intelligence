"use strict";
/* Device bench. Latency / cold start / memory / model bytes ONLY (no accuracy). Model + queries.json are fetched from
 * the SAME origin (relative paths from the query string); the server is serve_coi.py on localhost. Results are NEVER
 * sent anywhere: they are downloaded as DEVICE_BENCH.json and (always) printed once to the console on a marker line
 * for headless capture.
 *
 * Query string: model=fp32|int8 (ONE model per page load: cold start must be cold) fp32=<rel path> int8=<rel path>
 *   queries=<rel path> arm=wasm-1t|wasm-mt threads=<n, wasm-mt; default 4, always pass it explicitly>
 *   vendor=1 (ort wasm from vendor/) sink=<tag> (POST JSON to the localhost-only __bench_result sink)
 *   model_desc= power= n=<timed, default 1000> warmup=<default 20> auto=1 (run on load)
 *   label=DEVICE|SERVER_BROWSER id=<device id> session=<n> note=<free text>
 */

const SCHEMA = "ppsi.ondevice.device_bench.v1";
const MARKER = "DEVICE_BENCH_JSON:";
const DEFAULT_N = 1000;
const DEFAULT_WARMUP = 20;

let lastDoc = null;
const Q = new URLSearchParams(location.search);

function log(msg) {
  const el = document.getElementById("log");
  const line = `[${new Date().toISOString().slice(11, 19)}] ${msg}`;
  el.textContent += (el.textContent ? "\n" : "") + line;
  el.scrollTop = el.scrollHeight;
  console.log(line);
}

function isSameOriginRelative(p) {
  // same-origin relative paths only: reject backslashes (browsers read "\\evil.com" as "//evil.com"), protocol-relative
  // and absolute URLs, ".." segments, and anything whose resolved origin differs from ours.
  if (typeof p !== "string" || p.length === 0) return false;
  if (p.includes("\\") || p.startsWith("//") || /^[a-zA-Z][a-zA-Z0-9+.-]*:/.test(p)) return false;
  if (p.split("/").includes("..")) return false;
  try { return new URL(p, location.href).origin === location.origin; } catch (e) { return false; }
}

function numParam(name, dflt) {
  const v = Q.get(name);
  if (v === null || v === "") return dflt;
  const x = Number(v);
  return Number.isFinite(x) && x >= 0 ? x : dflt;
}

function timerResolutionMs() {
  let best = Infinity, last = performance.now();
  for (let i = 0; i < 200000 && best > 0.001; i++) {
    const t = performance.now(), d = t - last;
    if (d > 0 && d < best) best = d;
    last = t;
  }
  return Number.isFinite(best) ? best : null;
}

async function postSink(tag, doc) {
  // ONLY the same-origin, relative, localhost sink (serve_coi.py --results-dir); SERVER_BROWSER runs only (?sink=<tag>)
  if (!/^[A-Za-z0-9._-]+$/.test(tag)) throw new Error("bad sink tag");
  const r = await fetch("__bench_result?tag=" + encodeURIComponent(tag), { method: "POST",
    headers: { "Content-Type": "application/json" }, body: JSON.stringify(doc) });
  if (!r.ok) throw new Error("sink HTTP " + r.status);
}

async function sha256Hex(buf) {
  const d = await crypto.subtle.digest("SHA-256", buf);
  return Array.from(new Uint8Array(d)).map((b) => b.toString(16).padStart(2, "0")).join("");
}

function ortVersion() {
  const el = document.querySelector('script[src*="onnxruntime-web@"]');
  const m = el && el.src.match(/onnxruntime-web@([^/]+)/);
  return { pinned: m ? m[1] : null, runtime: (window.ort && ort.env && ort.env.versions && ort.env.versions.web) || null,
           cdn_failed_used_local_vendor: !!window.__ortCdnFailed };
}

function flatten(x, out) {
  out = out || [];
  if (Array.isArray(x)) for (const v of x) flatten(v, out); else out.push(x);
  return out;
}

function typed(dtype, flat) {
  switch (dtype) {
    case "int64": return BigInt64Array.from(flat, (v) => BigInt(Math.trunc(v)));
    case "int32": return Int32Array.from(flat);
    case "uint8": return Uint8Array.from(flat);
    case "int8": return Int8Array.from(flat);
    case "bool": return Uint8Array.from(flat, (v) => (v ? 1 : 0));
    case "float32": return Float32Array.from(flat);
    case "float64": return Float64Array.from(flat);
    default: throw new Error(`unsupported dtype ${dtype}`);
  }
}

function buildFeed(queries, row, inputNames) {
  const feed = {};
  for (const name of inputNames) {
    if (!(name in queries.inputs)) throw new Error(`queries.json has no "${name}"`);
    const shape = [1, ...(queries.shapes[name] || [])];
    feed[name] = new ort.Tensor(queries.dtypes[name], typed(queries.dtypes[name], flatten(queries.inputs[name][row])), shape);
  }
  return feed;
}

function percentile(sorted, p) {
  const n = sorted.length;
  if (n === 1) return sorted[0];
  const pos = (p / 100) * (n - 1), lo = Math.floor(pos), hi = Math.ceil(pos);
  return sorted[lo] + (sorted[hi] - sorted[lo]) * (pos - lo);
}

function warmStats(ms) {
  const s = ms.slice().sort((a, b) => a - b);
  return { n: ms.length, p50_ms: percentile(s, 50), p95_ms: percentile(s, 95), p99_ms: percentile(s, 99),
           mean_ms: ms.reduce((a, b) => a + b, 0) / ms.length, min_ms: s[0], max_ms: s[s.length - 1] };
}

async function measureMemory() {
  const rec = { api: "performance.measureUserAgentSpecificMemory", available: false, bytes: null, error: null,
                js_heap_used_bytes: (performance.memory && performance.memory.usedJSHeapSize) || null };
  if (typeof performance.measureUserAgentSpecificMemory !== "function") return rec;
  if (!self.crossOriginIsolated) { rec.error = "needs crossOriginIsolated"; return rec; }
  try { const r = await performance.measureUserAgentSpecificMemory(); rec.available = true; rec.bytes = r.bytes; }
  catch (e) { rec.error = String(e && e.message ? e.message : e); }
  return rec;
}

async function fetchBuf(path) {
  const t0 = performance.now();
  const r = await fetch(path, { cache: "no-store" });
  if (!r.ok) throw new Error(`fetch ${path}: HTTP ${r.status}`);
  const buf = await r.arrayBuffer();
  return { buf, ms: performance.now() - t0 };
}

async function benchModel(label, path, queries, arm, n, warmup) {
  const out = { path, arm };
  const memBefore = await measureMemory();
  const f = await fetchBuf(path);                     // cold start part 1: fetch
  out.bytes = f.buf.byteLength;
  out.sha256 = await sha256Hex(f.buf);
  const t1 = performance.now();
  const session = await ort.InferenceSession.create(f.buf, { executionProviders: ["wasm"], graphOptimizationLevel: "all" });
  const createMs = performance.now() - t1;            // part 2: session create
  const feeds = [];
  for (let i = 0; i < queries.n_rows; i++) feeds.push(buildFeed(queries, i, session.inputNames));
  const t2 = performance.now();
  await session.run(feeds[0]);                        // part 3: first inference
  const firstMs = performance.now() - t2;
  out.cold_start = { fetch_ms: f.ms, session_create_ms: createMs, first_inference_ms: firstMs,
                     total_ms: f.ms + createMs + firstMs };
  for (let i = 0; i < warmup; i++) await session.run(feeds[i % feeds.length]);
  const times = new Array(n);
  for (let i = 0; i < n; i++) {
    const t = performance.now();
    await session.run(feeds[i % feeds.length]);
    times[i] = performance.now() - t;
  }
  out.warm = Object.assign(warmStats(times), { warmup });
  out.memory = { before_model: memBefore, after_timed_loop: await measureMemory() };
  if (typeof session.release === "function") { try { await session.release(); } catch (e) { /* best effort */ } }
  return out;
}

function field(id) { return document.getElementById(id).value; }

function identity() {
  return { label: field("devLabel"), device_id: field("devId"), model_description: field("devModel"),
           user_agent: field("devUA"), hardware_concurrency: Number(field("devCores")) || null,
           device_memory_gb: field("devMem") === "" ? null : Number(field("devMem")),
           session: field("devSession"), power_and_conditions: field("devPower"),
           webdriver: !!navigator.webdriver,
           headless_hint: /HeadlessChrome|HeadlessEdg/i.test(navigator.userAgent) || !!navigator.webdriver };
}

async function runBenchmark() {
  const runBtn = document.getElementById("runButton"), dlBtn = document.getElementById("downloadButton");
  runBtn.disabled = true; dlBtn.disabled = true;
  try {
    const model = field("modelSel");
    const mpath = Q.get(model), qpath = Q.get("queries");
    if (!isSameOriginRelative(mpath)) throw new Error(`${model} must be a same-origin relative path (got ${mpath})`);
    if (!isSameOriginRelative(qpath)) throw new Error(`queries must be a same-origin relative path (got ${qpath})`);
    if (!window.ort) throw new Error("onnxruntime-web did not load (CDN and vendor/ort.min.js both failed)");
    const arm = field("arm");
    const n = numParam("n", DEFAULT_N), warmup = numParam("warmup", DEFAULT_WARMUP);
    if (n < 1) throw new Error("n must be >= 1");
    const coi = !!self.crossOriginIsolated;
    let threads = 1;
    if (arm === "wasm-mt") {
      if (!coi) throw new Error("wasm-mt needs crossOriginIsolated; serve with serve_coi.py");
      threads = numParam("threads", 4) || 4;   // explicit; match it to the pinned CPU set
    }
    if (Q.get("vendor") === "1") ort.env.wasm.wasmPaths = "vendor/";   // same-origin wasm: no CDN variance in session create
    ort.env.wasm.numThreads = threads;                // before the first session of this page load
    log(`arm=${arm} threads=${threads} crossOriginIsolated=${coi} n=${n} warmup=${warmup}`);
    const queries = await (await fetch(qpath, { cache: "no-store" })).json();
    if (!queries || !queries.inputs || !queries.n_rows) throw new Error("queries.json is not a browser_queries document");
    const doc = { schema: SCHEMA, created_at: new Date().toISOString(), identity: identity(),
                  ort: ortVersion(), arm, model, one_model_per_page_load: true,
                  cold_start_excludes: "browser start and page load; includes fetch + session create + first inference",
                  timer_resolution_ms: timerResolutionMs(), shared_array_buffer: typeof SharedArrayBuffer !== "undefined",
                  timed_loop_note: "includes await overhead; excludes tensor construction (feeds prebuilt); linear-interpolated percentiles", wasm_threads_requested: threads,
                  wasm_threads_applied: ort.env.wasm.numThreads, cross_origin_isolated: coi,
                  protocol: { n_timed: n, n_warmup: warmup, batch: 1, queries_file: qpath, n_query_rows: queries.n_rows },
                  note: Q.get("note"), models: {}, energy_proxy: null };
    doc.models[model] = await benchModel(model, mpath, queries, arm, n, warmup);
    log(`${model} p50 ${doc.models[model].warm.p50_ms.toFixed(3)} ms p95 ${doc.models[model].warm.p95_ms.toFixed(3)} ms`);
    doc.ort = ortVersion();
    try { doc.ort.wasm_resources = performance.getEntriesByType("resource").map((e) => e.name).filter((u) => /\.wasm|\.mjs/.test(u)); }
    catch (e) { doc.ort.wasm_resources = null; }
    lastDoc = doc;
    document.getElementById("results").textContent = JSON.stringify(doc, null, 1);
    dlBtn.disabled = false;
    console.log(MARKER + JSON.stringify(doc));        // single line; fallback capture, no network involved
    if (Q.get("sink")) { try { await postSink(Q.get("sink"), doc); log("sink: posted"); } catch (e) { log("sink FAILED: " + e.message); } }
    log("done; click Download DEVICE_BENCH.json");
  } catch (err) {
    const msg = String(err && err.message ? err.message : err);
    log("FATAL: " + msg);
    const errDoc = { schema: SCHEMA, error: msg, identity: identity() };
    console.log(MARKER + JSON.stringify(errDoc));
    if (Q.get("sink")) { try { await postSink(Q.get("sink"), errDoc); } catch (e) { /* console marker remains */ } }
  } finally {
    runBtn.disabled = false;
  }
}

function download() {
  if (!lastDoc) return;
  const blob = new Blob([JSON.stringify(lastDoc, null, 1)], { type: "application/json" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = "DEVICE_BENCH.json";
  document.body.appendChild(a); a.click(); a.remove();
  URL.revokeObjectURL(a.href);
}

document.addEventListener("DOMContentLoaded", () => {
  const set = (id, v) => { document.getElementById(id).value = v; };
  set("devUA", navigator.userAgent);
  set("devModel", Q.get("model_desc") || "");
  set("devPower", Q.get("power") || "");
  set("devCores", String(navigator.hardwareConcurrency || ""));
  set("devMem", navigator.deviceMemory !== undefined ? String(navigator.deviceMemory) : "");
  for (const [q, id] of [["label", "devLabel"], ["id", "devId"], ["session", "devSession"], ["arm", "arm"], ["model", "modelSel"]]) {
    if (Q.get(q)) set(id, Q.get(q));
  }
  document.getElementById("runButton").addEventListener("click", runBenchmark);
  document.getElementById("downloadButton").addEventListener("click", download);
  if (Q.get("auto") === "1") runBenchmark();
});
