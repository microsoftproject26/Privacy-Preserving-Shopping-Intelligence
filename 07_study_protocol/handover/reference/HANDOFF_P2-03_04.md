# P2-03 / P2-04 handoff: WB3 row-sparse item uploads + WC LoRA (Claude Opus 5.5, 2026-10-03)
Status: **code and CPU tests DONE (MEASURED, local only)**. Nothing ran on the server, no GPU, no training run, no TEST data, no git, no Codex.
**Review:** REV-P2-03_04 = PASS_WITH_MINORS. The minors are closed in this revision (§8).
Authority: company/plans/PHASE2_IMPLEMENTATION_PLAN.md P2-03 / P2-04; contracts/PREG2.md (REGISTERED, be0561ab) §5.3 (WB) / §5.4 (WC). The draft's §5.3 / §5.4 differ only in wording.
**No existing file was edited.** launch_fl.py, fl/server.py, fl/runtime.py and fl/client.py belong to the concurrent P2-02 worker. The wiring below is for that worker.

## 1. Files (new; sha256[:12]; final, after the review fixes)
| file | sha | content |
|---|---|---|
| `code/fl/sparse_rows.py` | 0d8235e24333 | `touched_rows`, `top_k_rows`, `encode`, `decode`, `payload_bytes_sparse`, `sparse_payload_bytes` / `sparse_upload_bytes` / `sparse_visit_bytes`, `SparseChannel(VirtualChannel)`, `install_sparse(run, *, k_top=3170, touched_fn=None, audit=True)`; **review additions:** `is_sparse_codec(codec)`, `sparse_codec_record(k_top)`, `sparse_bytes_record(manifest, counters, ledger, *, k_top)` |
| `code/fl/peft.py` | bc1012b08e41 | `LoRALinear`, `apply_lora`, `lora_keys`, `freeze_all_but` (beside `fl.runtime.freeze_item_tables`, which is unchanged), `merged_state`, `lora_state` / `load_lora_state`, `lora_init_sha256` (now also records `adapter.fl_lora_lineage`), `lora_weights_payload` (p2_lora now carries `lineage` = {theta_pre_digest, lora_init_sha256}) / `load_lora_from_payload`, `install_peft(run)`, `lora_numel_formula` |
| `code/tests_fl/test_p2_sparse.py` | 266172d4f6e0 | 16 tests + 1 NC |
| `code/tests_fl/test_p2_peft.py` | 994c4ea746f7 | 10 tests + 1 NC |

**Code-identity warning:** `launch_fl.fl_code_sha256()` hashes `fl/*.py`. Two new files in `fl/` change it. Phase-1 runs that are still in flight must keep resuming from their own tree (06E §8). Deploy this code only in the new P2 tree (code_p2r1), as the plan says.

## 2. Design
**WB3 (`fl.sparse_rows`)** follows the `fl.quant` Q8 pattern exactly. `SparseChannel.upload()` is called by `fl.server.run_round` before `acc.add()`. It does four things:
1. encodes the upload against theta_r (snapshotted at the round's first download; only the item tensors are cloned);
2. decodes it on the server side;
3. replaces the dict entries with the decoded values;
4. records the exact byte count on the run's ledger.

`check_round` verifies that every survivor was decoded once, in logical order. fl.aggregate is unchanged.
- **item_embed.weight:** the rows the client touched. These are the item tokens of the real events (attention_mask) of its **valid** decisions (`fl.client.valid_rows`, i.e. the rows it trains on), plus their target tokens (class + 3), sorted and unique.
- **output_embed + output_bias:** **one joint row set** `[W_j | b_j]`. These are the top-k classes ranked by `sqrt(‖ΔW_j‖² + Δb_j²)`, with k = 3,170. A stable descending sort breaks ties toward the lower id. If k ≥ K, every row is sent.
- **Other keys:** dense FP32.
- **What a sent row carries:** the client's **new FP32 row value**, so the decode is exact on sent rows. An unsent row decodes to theta_r's row, i.e. zero delta. There is no error feedback.
- **`decode(payload, theta)`:** reads only the payload and theta_r.
- **Bytes per visit:** `up = dense FP32 non-item keys + per sparse group (4-byte int32 row count + 4n int32 ids + 4n·width FP32 rows) + 8-byte n_consumed header`. Width is d for item_embed and d + 1 for the output layer. `down` is the dense FP32 shared state.
- **Losslessness:**
  - Under the registered client SGD_M (weight decay 0), an untouched item row has an exactly-zero gradient and does not move. The item part is therefore lossless, and the k = K run equals the dense run bitwise (tested).
  - The output layer receives a dense softmax gradient, so its top-k truncation is the only lossy part.
  - The **audit** reports per round the largest unsent item-row |Δ| (0 under SGD_M; > 0 under AdamW with weight decay, tested) and the fraction of the output delta that was not sent. It is monitoring only.
- **Production numbers (MEASURED with the formula on the full-K SASREC_D64_B2 manifest; Clean V2 schema):**
  - non-item shared numel = 265,162 (PREG2 quotes S8's 263,822; the difference is presumably the derived schema; TO-VERIFY, §7.11);
  - dense FA up = 82,840,200 B;
  - WB3 up = 1,900,144 B at 10 touched rows, 1,949,544 at 200, 2,157,544 at 1,000. The draft's "≈ 2.7 MB" was INFERENCE.
  - The CPU smoke with a synthetic 48-decision client (1,366 touched rows) gave 2,252,704 B.

**WC (`fl.peft`)**
- **The layer:** `LoRALinear` computes W_eff = W + (B A)·α/r and y = F.linear(x, W_eff, b).
  - W and b are frozen copies of theta_pre (keys `base_weight` / `base_bias`).
  - A [r, in] is `kaiming_uniform_(a=√5)` (the loralib / HF-PEFT default). It is drawn on the CPU from one generator seeded `derive_seed(seed, "p2_lora")`, consumed block 0 → 1 and then qkv → out → ff1 → ff2, so the init does not depend on the device.
  - B [out, r] starts at zeros.
- **`apply_lora`** wraps qkv, out, ff1 and ff2 of both CausalBlocks in place, in the same module slots, so the state-dict order is kept. It refuses to run twice.
- **`freeze_all_but(adapter, lora_keys(module), source=None)`:**
  - rebuilds the adapter's manifest from the wrapped module;
  - every non-LoRA parameter becomes a **FIXED buffer** (no grad, no optimizer, no clip, no upload, no aggregation, no DP dimension; workers are verified bitwise at each visit);
  - the frozen output layer turns its recentering hooks into no-ops;
  - `adapter.fl_frozen_keys` lists every frozen parameter, so FLRun's server/worker check and `visit_bytes(frozen_keys=…)` work;
  - `source` is the worker bootstrap copy;
  - it refuses to combine with `freeze_item_tables`, because it subsumes it.
- **`merged_state(module)`** returns the **base SASREC_D64_B2 key list and order**, with W + BA·α/r, and exactly W where BA·α/r == 0. B = 0 therefore merges to theta_pre bit for bit (no signed-zero flip).
- **Bytes per visit:** 4·2,048·r (r 4: 32,768 B; r 16: 131,072 B) plus the 8-byte header up. The frozen base is reported once per client (`frozen_bytes_once` = the full base model).
- **Checkpoints:**
  - latest.pt (unchanged runtime `state_dict`) holds the LoRA broadcast state (A/B shared plus the frozen base as FIXED buffers). `FLRun.resume` therefore restores A/B bitwise with no runtime change (tested).
  - best.pt / endpoint_6.0.pt go through the `install_peft` wrapper of `run.weights_payload`. `"theta"` is the **merged base-architecture state** (reeval / export see the ordinary graph). `"p2_lora"` = {format `PPSI_V4_P2_LORA_1`, meta (r, α, targets, seed, keys), the A/B tensors, `lora_theta_digest`, `merged_theta_digest`}.
  - `load_lora_from_payload` restores A/B from such a payload and checks the merged digest.

## 3. TO-CHECK resolution (P2-04)
`CausalBlock.forward_tokens`, `forward_padded`, `_attend` and `_ffn` (v4models/sasrec_v2.py) call `self.qkv(...)`, `self.out(...)`, `self.ff1(...)` and `self.ff2(...)` as **modules**. They never read `.weight` or `.bias`. The only direct reader is `SASRecCE._xavier_linear`, which runs at construction time, before LoRA is applied. A plain module wrapper would therefore have been enough.

`LoRALinear` exposes **`weight` as a property returning the effective W_eff** anyway, and `bias` as the base bias, so a future direct reader cannot get the stale base weight. `test_to_check_…` pins this: it greps the four method sources for `.weight` / `.bias` and checks that the property and forward equal W_eff.

## 4. Tests and counts (CPU; Windows; torch 2.9.1+cpu; basetemp on F:)
- New files alone (final versions): **26 passed, 2 xfailed**, on both the snapshot tree and the live tree with P2-02's current in-progress edits. With `--runxfail`, both NCs fail with AssertionError, as intended.
  - **Review additions (4 tests):**
    - α/r scaling: with (r 4, α 8) → scale 2.0 and (r 16, α 4) → 0.25, both W_eff − W and the merged ΔW equal B·A·α/r computed independently in FP64 (rtol 1e-5), and merged forward = wrapped forward ≤ 1e-6.
    - Top-k tie-break: 4,000 rows with four distinct exact joint norms, and k = 1,500 cutting inside a tie block. The result equals an explicit lexicographic (−norm, id) reference, is identical over 3 calls, and keeps the lowest ids of the tie block.
    - Codec dispatch by name (Q8 record, None and other names → not WB3).
    - Real SASREC_D64_B2 (K 3,000; some targets invalid): moved item rows ⊆ touched_rows, and the audit's item_dropped_max_abs is 0.0 under SGD_M.
    - The resume test now also checks that `sparse_bytes_record` is identical for the straight and the resumed run, while the resumed channel's `summary()` covers only the last leg.
  - **Mutation checks (MEASURED, on a throw-away copy, since deleted):** `argsort(stable=False)` makes the tie-break test fail, and `scale = alpha * r` makes the α/r test fail. Both mutants survived the earlier suite (review m1 / m2).
  - sparse: touched rows; exact round trip on sent rows with theta elsewhere; joint top-k with stable ties; bytes = hand formula (+8 on the ledger); decode never reads the upload (poisoned upload, both directly and through the channel); the sparse round equals the FedAvg of decoded uploads bitwise; **k = K with SGD_M equals dense, bitwise** (round level and FLRun level); the AdamW audit; the FLRun ledger; resume exact after reinstall; refusals for FP, frozen tables and pool. NC: k < K equals dense.
  - peft: the TO-CHECK; **B = 0 gives scores identical (torch.equal) to theta_pre**, the merged digest equals theta_pre's, and the init hash is refused if B ≠ 0; seeded A init with B = 0; **merged forward equals wrapped forward ≤ 1e-6** (r 4 and 16; the base key list matches); **only LoRA keys in extract_shared**, requires_grad and the optimizer groups, with the rest FIXED and recentering off; a visit moves only A/B; **bytes** = 4·2048·r (+8 up) with the frozen base reported once; **FLRun resume restores A/B bitwise**, finishes on the straight run's digest, and the endpoint payload has the merged base theta plus a p2_lora that restores A/B bitwise; refusals. NC: a worker with another frozen base is accepted (it is not: ManifestError).
- **Full `tests_fl` suite (MEASURED, 2026-10-03):** two copies of one tree snapshot under `_pytest_tmp/p2_0304_trees/{before,after}`, byte-identical except for my 4 files (checked with `diff -rq`). The snapshot is fl/server.py 8c94ee530ba6, runtime.py e627e943d93a (the wave-A archive baseline) and client.py 91ac088d48e1 (a P2-02 in-progress snapshot). Both copies ran with basetemp/TMP on F:.

| run | passed | skipped | xfailed | failed | exit |
|---|---|---|---|---|---|
| before (without the 4 files) | 239 | 9 | 75 | 0 | 0 |
| after (final 4 files) | 265 | 9 | 77 | 0 | 0 |

  The difference is exactly the 26 new tests and the 2 new NCs; nothing else changed.
  - Earlier attempts are not counted, and all were environmental:
    - C: free space was below the checkpoint disk floor (DiskRefused);
    - one access violation and one MemoryError in full-K tests, plus one pool-test failure, all under RAM pressure from parallel suites;
    - one before/after pair was invalid because P2-02 edited client.py between the two copies. The before run was redone with the same client.py.
- CPU smoke at **full K** (the script in §6 with `P2_DEV=cpu`):
  - LoRA r 4 / r 16: shared numel 8,192 / 32,768; merged-vs-wrapped max |d| = 0.0; frozen base unchanged after a visit.
  - WB3: item_dropped_max_abs 0.0; bytes equal the formula.

## 5. Wiring for the P2_W hook (launch_fl.execute; one call each)
**WB3 (sparse upload):**
- **The spec:**
  - item tables **unfrozen** (no `frozen_item_tables`);
  - method FA;
  - **the item-LR multiplier × 0.1 on the three item tensors** (PREG2 §5.3 WB3; P2-02's `item_lr_mult`);
  - `upload_codec = fl.sparse_rows.sparse_codec_record(3170)` (name `SPARSE_ROWS_WB3` + k_top + the wire rule), so it enters the arm-spec digest.
- **RunConfig.digest does not see the codec**, and the manifest is the same as WB1's. Only the ARM_SPEC sha keeps a WB3 resume from silently running dense.
- **M1, codec dispatch by NAME (required).** Today `execute` calls `install_q8` whenever `spec["upload_codec"] is not None`, and `bytes_record` switches to `q8_visit_bytes` on any non-None codec. Both branches must dispatch on the name:
  - `fl.sparse_rows.is_sparse_codec(codec)` → WB3;
  - `codec["name"] == "Q8_SR_PER_TENSOR"` → Q8;
  - anything else → refuse.
  - Otherwise a WB3 arm either fails closed (`run.channel is not q8`) or, worse, gets a silently wrong Q8 bytes block. Bytes per visit are selection tie-break (1) in §5.6. (Tested: `test_codec_dispatch_by_name`.)
- After `run = FLRun(...)` **and** after `FLRun.resume(...)` (resume rebuilds `run.channel`), do this in the codec branch:
```python
from fl.sparse_rows import install_sparse, is_sparse_codec, sparse_bytes_record
if is_sparse_codec(codec):
    sp = install_sparse(run, k_top=int(codec["k_top"]))   # default touched_fn = touched_rows(run.get_client(k).examples)
```
- In the round loop, as for Q8: `assert run.channel is sp; sp.check_round(out_r["round"], out_r["survived"])`.
- Optional: log `sp.round_stats[r]` (monitoring).
- **M2, bytes of record from checkpointed state (required).** In `bytes_record`, for WB3 use `sparse_bytes_record(server.manifest, run.counters, run.ledger, k_top=codec["k_top"])`:
  - the ledger and counters are restored by `FLRun.resume`;
  - `per_visit.upload_bytes_mean` = ledger.bytes_up / Σ visit_counts;
  - the download is dense FP32;
  - fleet totals are included;
  - per-client figures are approximate (visits × (down + fleet-mean up)), because the ledger has no per-client upload.
  - **Do not** use `sp.summary()` / `sp.totals` as bytes. They live in the process only and restart after every resume (the summary's `scope` field says so). Keep them, if at all, as `bytes["sparse_codec_this_leg_monitoring"]`.
  - (Tested: the straight and resumed runs give identical `sparse_bytes_record`.)
  - The definition "WB3 upload bytes per visit = ledger.bytes_up / Σ visit_counts" should be fixed in P2_METHODS_LOCK (P2-06 / Ops).
- The default `touched_fn` calls `get_client` a second time per visit. If `ctp.batch` is costly, pass `touched_fn=` from a cache.
- **Tree timing (review m4).** The plan cuts code_p2r1 after Part A, but WB3 / WC run in RUN-2. Either this wiring lands in launch_fl **before** code_p2r1 is frozen, or RUN-2 needs a code_p2r2 tree. In the second case, no WA run may be resumed across the two trees (`fl_code_sha256` differs; resume is refused).

**WC1–WC4 (LoRA):**
- The spec has `lora = {"r": 4|16, "alpha": r, "targets": [...], "seed": <s>}`, **no** `frozen_item_tables` (freeze_all_but subsumes it; combining them is refused), and method FA.
- Before `Server(...)`, after theta_pre is loaded into `server_adapter` (`apply_init_state` gives `init_sha256` = theta_pre's digest):
```python
from fl.peft import apply_lora, freeze_all_but, lora_keys, lora_init_sha256, install_peft
L = spec["lora"]
apply_lora(server_adapter.module, tuple(L["targets"]), r=L["r"], alpha=L["alpha"], seed=L["seed"])
freeze_all_but(server_adapter, lora_keys(server_adapter.module))
init_sha256 = lora_init_sha256(server_adapter, init_sha256)          # checks B = 0 and merged == theta_pre; LoRA-θ0 hash
src = server_adapter.broadcast_state()
for w in workers:
    apply_lora(w.module, tuple(L["targets"]), r=L["r"], alpha=L["alpha"], seed=L["seed"])
    freeze_all_but(w, lora_keys(w.module), source=src)               # one-time bootstrap of the frozen base
server = Server(server_adapter, init_sha256=init_sha256)
```
- After `FLRun(...)` / `FLRun.resume(...)`: call `install_peft(run)`. This makes best.pt / endpoint_6.0.pt carry the merged theta + `p2_lora`.
- `bytes_record(..., frozen_keys=frozen_keys_of(server_adapter))`, so the frozen base is counted once per client.
- **Evaluation:** the harness evaluates `server_adapter`, i.e. the wrapped forward. reeval of best.pt uses the merged weights, which give the same scores (≤ 1e-6; 0.0 measured on CPU).
- `save_pf_store`, PF and DP: unchanged (WC5 = the PF path, no peft code).
- **Note for P2-02:** with LoRA the item keys are not shared, so the item-LR param group must tolerate absent item keys (WB1–WB3 only).

## 6. GPU smoke (server, SMOKE_NOT_EVIDENCE; about 1 min; synthetic data only, no store, no TEST)
From the P2 tree root (the folder that holds `code/` and `contracts/`):
```bash
CUDA_VISIBLE_DEVICES=<g> P2_DEV=cuda P2_CODE=code python -B - <<'EOF'
import os, sys, time
sys.path.insert(0, os.environ.get("P2_CODE", "code")); sys.dont_write_bytecode = True
import ppsi_env  # noqa
import torch
from fl.numerics import enforce_strict_fp32; enforce_strict_fp32()
from v4models import build, load_schema
from fl.families import synthetic_examples_v2
from fl.client import ClientData, LocalSolver, client_update
import fl.peft as P, fl.sparse_rows as SR
DEV, K, sch = os.environ.get("P2_DEV", "cuda"), 158_486, load_schema()
ex = synthetic_examples_v2(48, K, torch.Generator().manual_seed(1)); cl = ClientData("smoke-u0", ex)
sol = LocalSolver(lr=0.03, optimizer="sgd_m", passes=1, batch_size=16, clip=1.0)
for r in (4, 16):
    s, w = (build("SASREC", sch, 2026, DEV, K=K, variant="SASREC_D64_B2") for _ in range(2))
    pre = s.init_sha256
    P.apply_lora(s.adapter.module, r=r, alpha=float(r), seed=2026); P.freeze_all_but(s.adapter, P.lora_keys(s.adapter.module))
    init = P.lora_init_sha256(s.adapter, pre); th = s.adapter.broadcast_state(clone=True)
    P.apply_lora(w.adapter.module, r=r, alpha=float(r), seed=2026); P.freeze_all_but(w.adapter, P.lora_keys(w.adapter.module), source=th)
    t0 = time.perf_counter(); res = client_update(w.adapter, th, cl, sol, round_idx=0, seed=2026, clone_upload=True)
    torch.cuda.synchronize() if DEV == "cuda" else None; dt = time.perf_counter() - t0
    after = w.adapter.broadcast_state(); assert all(torch.equal(after[k], th[k]) for k in w.adapter.fl_frozen_keys)
    s.adapter.load_state_({**th, **res.upload}); m = P.merged_state(s.adapter.module)
    b = build("SASREC", sch, 5, DEV, K=K, variant="SASREC_D64_B2"); b.adapter.load_state_(m)
    s.adapter.module.eval(); b.adapter.module.eval()
    with torch.no_grad(): diff = float((s.adapter.scores(ex) - b.adapter.scores(ex)).abs().max())
    print(f"LoRA r={r}: shared {s.adapter.manifest.shared_numel}, visit {dt:.3f}s, merged-vs-wrapped {diff:.3e}, init {init[:12]}")
    assert s.adapter.manifest.shared_numel == 2048 * r and diff <= 1e-6
s, w = (build("SASREC", sch, 2026, DEV, K=K, variant="SASREC_D64_B2") for _ in range(2))
th = s.adapter.broadcast_state(clone=True)
res = client_update(w.adapter, th, cl, sol, round_idx=0, seed=2026, clone_upload=True)
rows = SR.touched_rows(ex)
ch = SR.SparseChannel(s.adapter.manifest, k_top=SR.K_TOP_REGISTERED, theta_fn=lambda: th, touched_fn=lambda _k: rows)
ch.download(0, cl.key); up = dict(res.upload)
t0 = time.perf_counter(); n = ch.upload(0, cl.key, up); torch.cuda.synchronize() if DEV == "cuda" else None
st = ch.round_stats[0]
print(f"WB3: rows {rows.numel()}, up {n} B (dense {s.adapter.manifest.shared_bytes + 8} B), codec {time.perf_counter() - t0:.3f}s, "
      f"item_dropped {st['item_dropped_max_abs']}, out_dropped_frac {st['output_dropped_frac_max']:.4f}")
assert st["item_dropped_max_abs"] == 0.0 and n == SR.sparse_upload_bytes(s.adapter.manifest, rows.numel(), 3170)
print("P2-03/04 SMOKE OK (SMOKE_NOT_EVIDENCE)", DEV, torch.__version__)
EOF
```
Also run on the server (CPU is enough): `python -B -m pytest -q -p no:cacheprovider --basetemp <tmp> code/tests_fl/test_p2_sparse.py code/tests_fl/test_p2_peft.py`. Expected: 26 passed, 2 xfailed.

After the P2_W hook exists, smoke each arm as usual: `python -B code/v4int/launch_fl.py --arm P2_W … --role CALIBRATION --smoke --smoke-rounds 3 --run-dir <R>/smoke_WB3|smoke_WC1`, then the same with `--smoke-interrupt-at 2` followed by `--resume`. This checks the reinstall-after-resume path for both modules on real rows.

## 7. Open questions / judgement calls (INFERENCE unless marked)
1. **Joint output-layer row set.** §5.3 says "output_embed / output_bias: top-k rows by ‖row delta‖₂". I read this as one class set for [W_j | b_j], ranked by the joint norm (the logit of class j depends on both). Separate top-k sets would add 4k bytes and break the class pairing. Reviewer decision.
2. **Framing bytes.** I count a 4-byte int32 row count per sparse group (8 B per visit; a variable-length id list needs it) **plus** the 8-byte n_consumed header (as FA / Q8). If the plan's "8-byte header" was meant to be the only framing, set `ROW_COUNT_BYTES = 0`; the change is 8 B on about 2 MB.
3. **Touched rows** use the valid decisions only (the rows that can receive gradient). History tokens of invalid or masked rows are excluded, and PAD is never included. Target tokens are included as §5.3 says, even though they move only if they also appear in the history (a superset costs bytes only).
4. **Sent values, not differences.** This gives an exact decode. On the wire the information is the same; the server then treats unsent rows as zero delta.
5. **Server recentering with WB3:** the decoded aggregate's output-layer common mode is removed by the server's `server_post_aggregate` (unchanged), so truncation adds no gauge drift.
6. The LoRA α = r gives scale 1. Kaiming(a = √5) is my reading of "seeded Kaiming".
7. The LoRA seed should be in the spec (I suggest = the run seed). r ≤ min(in, out) is enforced.
8. **Environment, local only:**
   - C: had 0.2–2 GB free, which makes the checkpoint tests' disk floor (8 GiB) refuse whenever a path lands on C:. The suites ran with basetemp/TMP on F:.
   - One earlier attempt crashed (access violation in `test_t32_dense_fullk`, full-K AdamW) while another worker's suites were running in parallel. This was memory pressure, unrelated to this code.
   - Leftover temp folders: `ppsi_v4/_pytest_tmp/p2_03_04_*` and `ppsi_v4/_pytest_tmp/p2_0304_trees` (before/after copies and suite logs `before.txt` / `after.txt`; deletable once the counts are accepted).
9. **O1 (review; Ops / Fable).** §5.5 carries only W*'s server optimiser and LRs into WD.
   - If W* = WB3: `install_sparse` refuses DP (fails closed). Rule whether WD then runs dense with the same item-LR, or whether WB3 + DP must be built.
   - If W* = WC: DP over the LoRA keys works in fl.dp. However, launch_fl's `dp_noise_dimension.frozen_keys_excluded` reads `spec["frozen_item_tables"]` and would list nothing; it should use `frozen_keys_of(server_adapter)`.
10. **O2 (review; P2-01 / Ops).** With LoRA, the trained subset T of D1-ext would be the A / B keys. A starts random, so `rel_pre_dist` on T does not measure how far the model moved from θ_pre. For WC, define T on the merged weights vs θ_pre: `fl.peft.merged_state(module)` gives them directly.
11. The non-item numel 265,162 (local Clean V2 schema) vs 263,822 (S8) is **TO-VERIFY** against the server manifest before any bytes claim.

## 8. Review closure (REV-P2-03_04 PASS_WITH_MINORS)
| item | closed by |
|---|---|
| m1, α/r untested | `test_alpha_over_r_scaling_of_effective_and_merged_weights`; kills the `alpha * r` mutant |
| m2, tie test not discriminating | `test_top_k_tie_break_lower_id_first_deterministic_large_tie_blocks`; kills the `stable=False` mutant |
| m5, lossless only on TinyRec | `test_real_sasrec_item_rows_lossless_under_sgd_m` (SASREC_D64_B2, K 3,000) |
| M1, codec dispatch | `is_sparse_codec` / `sparse_codec_record` + §5 instruction + test |
| M2, bytes after resume | `sparse_bytes_record` (checkpointed ledger + counters) + §5 instruction + resume test; `summary()` labelled "this leg, monitoring" |
| m3, item-LR × 0.1 for WB3 | §5 spec bullet |
| m4, code tree timing | §5 last bullet |
| m6, disclosure | §9; the audit is labelled SIMULATOR ORACLE in code (`SparseChannel` docstring, `summary()["audit_note"]`) |
| full-suite numbers missing | §4 table (239/9/75 → 265/9/77, 0 failed) |
| nits | citations now point to PREG2.md be0561ab; the peft docstring names `BaseAdapter.load_state_`; p2_lora carries `lineage` (theta_pre digest + LoRA init hash; tested); 265,162 marked TO-VERIFY |
| O1, O2 | §7.9 / §7.10, for Ops / Fable / P2-01 |

## 9. Privacy disclosure: WB3 reveals touched item ids (for P2_METHODS_LOCK and the report)
- **What leaks.** A WB3 upload sends the client's touched item_embed row ids **in the clear**. This is the set of item tokens in its trained history plus its target items, i.e. what the user viewed / carted / bought. The output-layer top-k ids correlate strongly with the client's targets too.
  - A dense upload (WB1 / WB2, S7) hides the item set inside a full-size update. A sparse upload does not.
  - This is the known item-set leakage of sparse federated-recommendation updates. The server learns at least the item set of every visit even without any gradient-inversion attack.
- **Interaction with other methods.**
  - Standard secure aggregation (P2-09) needs a common dense index set, so per-client index sets are incompatible with SecAgg as registered. WB3 + SecAgg would need padding to a public superset or a different protocol.
  - `install_sparse` refuses DP (O1). No DP / LDP guarantee covers the index set.
  - The MIA / audit claims (E3 / E4, PREG2 §7.3 / §7.4) for a WB3 model must state that the index set was disclosed.
- **Required wording.** Wherever WB3 bytes or WB3 accuracy are reported: "WB3 discloses each client's touched item ids (history and target items) and correlated output-row ids to the server in every round; the byte saving is not privacy-neutral; WB3 is not compatible with standard SecAgg and has no DP protection of the index set." Ops will carry this into P2_METHODS_LOCK and the Phase-2 report, next to the WB3 bytes.
- **Simulator note.** The channel's audit fields (`item_dropped_max_abs`, `output_dropped_frac`) read the client's dense upload. They are a simulator-only oracle that a real server would not have, and must not be presented as server-side measurements.
- WC (LoRA) uploads only dense A / B matrices of fixed shape, so no index set leaks. It has the same gradient-leakage profile as any dense low-rank update (INFERENCE).
