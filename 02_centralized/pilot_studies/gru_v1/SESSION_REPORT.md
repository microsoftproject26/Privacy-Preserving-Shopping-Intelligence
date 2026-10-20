# REES46 Next-Item — Full Session Report

**Scope:** everything performed in this working session, end to end.
**Period:** 2026-09-21 → 2026-09-23
**Owner:** Ahmed Abdelhamed (GitHub `AhmedAbdelhamed01`)
**Server:** Queen's University Lobot, group `group-digi2026-g8`, host `jupyter-group-digi2026-g8`
**Status at close:** dataset transferred and verified on the server; GRU pilot executed; **pilot returned NO-GO against its pre-registered gate; full training deliberately NOT launched, awaiting an owner decision.**

---

## ملخّص تنفيذي بالعربي

اتعمل في الجلسة دي أربع مراحل رئيسية:

1. **تجهيز السيرفر:** الدخول على Lobot، إنشاء مجلد شخصي منفصل، ضبط هوية الـgit صح، وإنشاء tunnel خاص للاتصال من VS Code. اتصلّحت مشكلة مهمة: كان فيه خلط في الهوية — الأدوات استنتجت إن المستخدم هو الزميل Ahmed Sherif من ملفات موجودة على حساب Linux مشترك، وده غلط. الهوية الصحيحة اتثبتت من مصدر مستقل.
2. **نقل الداتا:** نقل `rees46_nextitem_v1` (8.53 جيجا، 409 ملف) من الويندوز للسيرفر. اتقسمت لـ364 جزء ورفعت على مدى ~15.4 ساعة، واتأكدنا إنها سليمة بايت-بايت (409/409 hash مطابق).
3. **بناء قناة تحكم آلي:** اتعمل اتصال تنفيذ حقيقي بالسيرفر عبر بروتوكول Jupyter kernel، فبقى ممكن تشغيل أوامر على السيرفر أوتوماتيك بدل النسخ واللصق اليدوي.
4. **تجربة الـGRU:** مراجعة معمارية مستقلة، تنفيذ كامل مع اختبارات صارمة، قياس أداء العتاد، وتشغيل pilot.
   **النتيجة: الموديل لم يتجاوز الـbaseline البسيط (TRANS) عند ميزانية التدريب المحدّدة مسبقًا، فاتوقف التدريب الكامل ولم يُشغَّل.** التفاصيل والأسباب في القسم 5.

**أهم رقم:** الموديل وصل macro MRR@20 = 0.1008 مقابل 0.1331 للـbaseline، **لكنه كان لسه بيتحسّن بمعدل ~21% في الـpass الأخير** ولم يصل لمرحلة التشبّع — أي أن الميزانية كانت صغيرة (5% من epoch واحد)، وليس بالضرورة أن المعمارية فاشلة. القرار النهائي للمالك.

---

# 1. Lobot server setup

## 1.1 Environment established (measured, not assumed)

| Item | Value |
|---|---|
| Host | `jupyter-group-digi2026-g8` |
| Linux user | `jovyan` — **shared by all three teammates** |
| GPUs | 2 × NVIDIA A16, 15,356 MiB each (14.61 GiB usable per torch) |
| RAM / cores | 64 GB / 8 |
| Disk | 49 GB volume |
| Shared GPU env | `~/venvs/ppsi-gpu` — torch 2.13.0+cu130, numpy 2.4.6, pyarrow 21.0.0, yaml 6.0.3 |
| Repo | `~/ahmedabdelhamed01/Privacy-Preserving-Shopping-Intelligence` |

## 1.2 Identity collision — corrected

An early inference claimed the user was teammate **Ahmed Sherif**, based on the git config inside `~/ahmed`. **This was wrong and was retracted.** On a shared Linux account, file contents never establish ownership.

Correct identity, established from an independent source (Windows Credential Manager + GitHub public API):

```
GitHub login   AhmedAbdelhamed01   (id 114183798)
git user.name  AhmedAbdelhamed01
git user.email 114183798+AhmedAbdelhamed01@users.noreply.github.com
```

The **GitHub noreply address** was chosen over the personal Gmail because the Gmail could not be proven verified on the account; the noreply always links commits correctly and keeps the personal address out of public history.

Configuration is **repository-local only** with `user.useConfigOnly=true`; the machine-level global config was verified empty. On a shared account a `--global` identity would attribute every teammate's commits to one person.

Teammate resources left untouched throughout: `~/ahmed`, `~/.vscode-tunnel-ahmed`, tmux `tunnel-ahmed`, tunnel `g8-ahmed`.

## 1.3 Separate workspace and tunnel

| Resource | Value |
|---|---|
| Personal directory | `~/ahmedabdelhamed01/` |
| Repo clone | `~/ahmedabdelhamed01/Privacy-Preserving-Shopping-Intelligence` (clean, `main` @ `bfc90a0`) |
| VS Code tunnel | `g8-ahmedabdelhamed01` (tmux `tunnel-ahmedabdelhamed01`, cli-data-dir `~/.vscode-tunnel-ahmedabdelhamed01`) |

Verified from inside the remote VS Code session: correct host, correct user, correct path, and git author/committer resolving to the correct identity.

---

# 2. Dataset transfer: `rees46_nextitem_v1`

## 2.1 A stale hash was caught before transfer

The task brief quoted `76a4bda2e385…` as the expected version hash. The actual frozen dataset carried `860e5a1c3a81…`.

Investigation of the project's own logs showed the dataset had been **re-frozen** at 14:45 UTC after fixes landed in `src/` and `scripts/` (both covered by `code_hash`), and the project's own review log states the hash change is correct behaviour, not a regression. **`860e5a1c…` is authoritative; `76a4bda2…` is pre-re-freeze and must not be used.**

## 2.2 Source verified before sending

```
VERSION INTACT   409 files hashed, 0 mismatches, exit 0
version_hash        860e5a1c3a81b8e4bbfeff4f245fa37638a43e4f260df179585964d2d4da8dbd
FILES.json sha256   4d4d20d38eeb1b8cd9a7ba76eaea46e79312efc534cd15325604a03ef7004b1b
VERSION.json sha256 65f03c594d3d34dad1359c96c99bfff229d0602f569946933e6018199b4d1f01
tree                411 files on disk = 409 hash-covered + FILES.json + LAST_VERIFICATION.json
size                9,155,162,711 bytes (8.53 GiB)
problems_recorded_at_freeze  []
```

Byte arithmetic closed exactly: tree bytes − recorded bytes = 96,335 = the two control files.

## 2.3 Transfer method and the problems solved on the way

There is **no SSH/scp/rsync** to Lobot (confirmed against Queen's own cluster documentation) — only the JupyterHub browser and the VS Code tunnel. Transfer therefore ran over the **Jupyter Contents API** (HTTPS, base64 PUT).

Problems encountered and resolved:

| Problem | Resolution |
|---|---|
| Manual browser upload stalled ~6 MiB into the first file | Replaced with scripted, retryable per-chunk uploads |
| 512 MiB chunks rejected — nginx `413 Request Entity Too Large` | Re-chunked to **364 × 24 MiB** |
| "Cancelled" attempts left **orphaned `curl.exe` processes** still uploading, competing for bandwidth and corrupting timing measurements | Found via process inspection and killed; explained most of the apparent "network flakiness" |
| Hung requests survived `--max-time`, `timeout`, and `taskkill` | Root cause: bash `$!` returns the **MSYS** PID, not the Windows PID. Fixed via `/proc/<pid>/winpid` translation before `taskkill` |
| Contents API refuses dot-prefixed paths | Staging directory renamed without the leading dot |

## 2.4 Result

```
364 / 364 chunks uploaded   (3 needed a second pass: part-ih, part-ii, part-ij)
total transport             9,155,686,400 bytes
elapsed                     ~15.4 hours
sampled content check       34 / 34 chunks sha256-MATCH (all 14 ever-retried + 20 random)
chunks.sha256 manifest      byte-identical (773ab837…)
```

Final on-server verification **at the promoted path**:

```
root            /home/jovyan/shared_data/rees46_nextitem_v1
version_hash    860e5a1c…    (identical to source)
FILES.json      4d4d20d3…    (identical to source)
VERSION.json    65f03c59…    (identical to source)
verified_files  409 / 409     missing 0     mismatched 0     intact: true
```

Supporting files written outside the frozen dataset: `shared_data/REES46_NEXTITEM_V1_TRANSFER.json`, `shared_data/README_REES46.md`.

## 2.5 Incident, honestly recorded

A command block issued during promotion placed **cleanup (`rm -rf staging`, `rm -rf chunks`) in the same paste as the `mv` promotion, with no guard to halt on failure**. The `mv` failed (a directory already existed from an earlier mangled paste), but the subsequent cleanup lines still executed and deleted the 364 uploaded chunks and the reassembled tar from the server.

- **No data was lost** — a correctly extracted copy already existed, and the canonical Windows copy was untouched.
- Recovery required only fixing a directory-nesting error, not re-uploading.
- **Lesson applied since:** destructive steps are never chained after a promotion step; commands are `&&`-chained so a failure halts the rest.

A second recurring hazard: the user's terminal **mangles long multi-line pastes containing Arabic (RTL) comments**, which caused commands to execute out of order at least twice. All later command blocks were issued as pure ASCII.

---

# 3. Automated server-control channel (capability built)

Because there is no SSH, a real execution channel was built on the **standard Jupyter kernel WebSocket protocol** (the same mechanism `nbclient`/`jupyter_client` use):

1. `POST /api/kernels` → start a python3 kernel
2. connect `wss://…/api/kernels/<id>/channels`, send an `execute_request`
3. collect `stream`/`execute_result`/`error` until `status: idle`
4. `DELETE /api/kernels/<id>` so nothing is left holding memory on the shared server

Two capabilities now exist:

| Channel | Use | Limitation |
|---|---|---|
| Contents API | file read/write/delete | cannot see or create dot-prefixed paths; nginx body limit |
| Kernel WebSocket | **arbitrary shell execution** | none relevant |

This removed the need for manual copy-paste and is what all subsequent work used.

---

# 4. GRU experiment — preparation

## 4.1 Independent architecture review (Fable 5.1)

Verdict: **FIX_REQUIRED** — architecture compatible, but 8 corrections needed first. The five highest-impact claims were **independently verified against source** before being accepted:

| Finding | Severity | Verified |
|---|---|---|
| `client_id` collapses all unseen users to OOV=2 → macro-MRR would merge ~500k users into one "user" | HIGH | ✅ `loaders.py:316-321` |
| Loader reads whole buckets (all columns, per-row dicts, no eviction) → RAM exhaustion | HIGH | ✅ `loaders.py:85-115,186-193` |
| `target_class=-1` means **both** "no target" and "OOV target"; mishandling silently drops 49,545 rows and inflates every metric | HIGH | ✅ `pipeline5.py:394-404` (SQL comment: "OOV targets retained end-to-end and scored zero") |
| `post_freeze_smoke.py:101` is vacuous — `bool(...) or True` | MED | ✅ literal |
| Pilot manifest must filter `loss_mask` (52,760,314 raw vs 40,180,608 contributing) | MED | ✅ `loaders.py:194-215` |

**Notable finding:** F4 − F3 is **exactly one channel** (`log1p_price`). F3 already carries all 15 quality flags (including the three price flags) and all 12 user-context values (including `recent_mean_log_price`). The F0/F4 comparison is therefore honestly **item-only vs. all side information**, not a price ablation.

## 4.2 Key architecture decisions

| Decision | Choice | Reason |
|---|---|---|
| Workspace | all work under git-excluded `.local_ai/gru_v1/`, not `ppsi/` | shared repo; avoids a permanently dirty tree that invites accidental commits |
| Cache | one shared F4-complete cache for both arms | makes F0/F4 fairness **structural** — impossible for arms to see different decisions |
| Macro grouping | dense `eval_user_key` from sorted distinct `user_id` | structurally unreachable by the model (not in `OPTIONAL_CHANNELS`); `user_id` never persisted |
| Objective | exact full-catalogue CE primary | pre-registered E1/E2 rules decide; sampled CE only as a separately-named config |
| GPU | one A16, `CUDA_VISIBLE_DEVICES=0` | shared server etiquette |

**Acceptance criteria were frozen in writing before any result existed**, including two ambiguities resolved pre-pilot (selection metric = end-to-end-denominator macro MRR@20; GO-1 required on **both** micro and macro Recall, strictly harder than either single reading).

## 4.3 Implementation and correctness gates

| Component | Independent verification |
|---|---|
| ContextGRU-v1 model | tied head is a true **view** of one Parameter (no detach/copy/second Linear); output `[B,158486]` never `[B,L,K]`; readout `h_n[-1]`; params **20,964,062** (F4) / **20,791,190** (F0) matching derivation exactly |
| Objective | returns loss **sum + count** separately → accumulation-correct; uneven-microbatch equivalence tested |
| Evaluator | OOV rows counted with **zero credit** (sentinel rank −1), never dropped; macro grouped only by `eval_user_key`; no `topk`; chunk sizes {1,4,11} identical |
| Loader on Lobot | vendored 3-file closure, `layout == "frozen"` |
| Cross-platform digest | **identical** blake2b across torch 2.9/2.13, pyarrow 22/21 — `8f964013…` |
| Window cache | 6.78 GB, **byte-exact** parity vs direct loader on all 19 arrays (re-verified independently, 183 s) |
| Resume (T13) | **bit-identical** losses at updates 301–310 (max rel. diff **0.0**); broken variant diverged 3–4% |
| Synthetic memorisation | F0 & F4: loss 8.04 → 2.7e-7, **top-1 = 1.0**; negative control correctly fails at 0.0078 |

**Two real bugs were caught by tests rather than luck:** `decision_id` exceeds 2³¹ (an int32 instruction was wrong — caught by the parity test), and a `numpy.unique` over 8.8M boxed strings that hung 15 min (replaced with `pyarrow.compute`, 9 s).

## 4.4 Hardware benchmark

| Pre-registered gate | Threshold | Measured | Verdict |
|---|---|---|---|
| **E1** VRAM @ F4/256/FP32 | ≤ 8 GiB | **0.94 GiB** | **PASS** |
| **E2** GPU-only throughput | ≥ 2000 dec/s | **4,655 dec/s** | **PASS** (2.3×) |
| Input path not bottleneck | ≥ 0.5 × GPU-only | **0.945×** (4,401 dec/s) | **PASS** |
| Host RSS | ≤ 24 GB | 2.23 GB | **PASS** |

Exact CE stays primary — the sampled-CE trigger needed throughput < 800 dec/s **and** head+loss ≥ 60% of step time; measured head+loss is **25.4%** (backward dominates at 49.9%).

**The cache decision was decisively justified:** direct loader 439 dec/s cold with RSS 2,048 → 9,387 MB for 4 of 64 buckets, versus 4,401 dec/s at 2,226 MB from cache — **10× faster, 4× less RAM**.

Implied full epoch: **2.4 h GPU / 2.54 h wall-clock** for all 40,180,608 decisions.
Other observations: F4 costs only ~3% throughput over F0; mean context length is **~4.9 events**, not 50.

---

# 5. Pilot results and the NO-GO decision

## 5.1 Protocol

Two arms (`F0_item_only`, `F4_rich_side_information`), exact CE, microbatch 256, FP32, AdamW lr 1e-3, seed 2026, **exactly 2 passes × 1,000,000 decisions = 2,000,000 exposures per arm**, identical data order, evaluated on a fixed 10,000-user VALIDATION_SELECT cohort (112,365 decisions). TEST and VALIDATION_CONFIRM were never scored.

## 5.2 Baselines (fit on the same 1M manifest, same evaluator and tie rule)

| Baseline | micro R@20 | micro MRR@20 | macro R@20 | macro MRR@20 |
|---|---|---|---|---|
| POP | 0.04890 | 0.01145 | 0.08620 | 0.02189 |
| **TRANS** | 0.28530 | 0.10639 | 0.34828 | **0.13308** |

## 5.3 Results

| Gate | Threshold | Measured (F0 best) | Verdict |
|---|---|---|---|
| Engineering P1/P3/P5, learning floor, overfitting | — | exactly 2,000,000 exposures, 0 non-finite, bit-identical re-eval, 2.0 GiB VRAM, 9.73 GB RSS | **ALL PASS** |
| **GO-1** sanity floor | micro R ≥ 0.07335, macro R ≥ 0.12930, MRR > POP | **0.24238 / 0.29621 / 0.08175 / 0.10079** | **PASS** (3–7× over) |
| **GO-2** vs TRANS | ≥ 0.13308 approve; ≥ 0.11977 conditional | **macro MRR@20 = 0.100795** | **NO_GO** |
| F4 vs F0 (never gating) | — | +0.000365, 95% CI **[−0.00101, +0.00170]** | no detectable gain |

## 5.4 The decisive diagnostic

The model was **still improving steeply when the frozen budget ran out** — not plateaued, not overfitting:

| | pass 1 | pass 2 | change |
|---|---|---|---|
| F0 macro MRR@20 | 0.08342 | **0.10079** | **+20.8%** |
| F0 micro MRR@20 | 0.06428 | **0.08175** | **+27.2%** |

The pilot's exposure was 2,000,000 decisions ≈ **5% of a single epoch** of the 40,180,608 available. Reaching TRANS requires +32% from here; the model gained +21% in the last pass alone.

## 5.5 Interpretation

**The NO-GO is honest and stands.** Thresholds were frozen in writing before any result existed, and full training was withheld accordingly.

However the failure signature is **"under-trained", not "architecture inadequate"**, and that distinction is evidenced rather than asserted:

- implementation correctness is independently verified (memorisation top-1 = 1.0, bit-identical resume, byte-exact cache parity, cross-platform digest match);
- GO-1 was cleared by 3–7× margins — the model vastly outperforms popularity, so it is clearly learning real sequential structure;
- the metric was rising ~21% per pass at cutoff;
- TRANS is a strong, **fast-saturating** baseline fit on the same 1M manifest; a neural model at 5% of one epoch is not expected to have overtaken it yet.

The F4 result is a genuine, reportable finding: **at this exposure the rich side information buys nothing measurable** over item IDs alone.

## 5.6 Why full training was not launched

Extending the exposure budget or proceeding despite a failed gate would **change the frozen protocol**, which the governing brief forbids doing unilaterally. Three options remain open for the owner:

1. **Authorise full training** (~2.5 h/epoch measured), recording explicitly that GO-2 was overridden and why.
2. **Re-run the pilot at a larger budget** (e.g. 10M exposures) as an openly amended protocol, then re-apply GO-2. *Recommended if the gate should stay meaningful.*
3. **Accept the scientific stop** and report that the GRU does not beat a first-order transition baseline at pilot scale.

---

# 6. Current state

## 6.1 On the server

```
/home/jovyan/shared_data/rees46_nextitem_v1        frozen dataset, VERIFIED, READ-ONLY, 8.6 GB
/home/jovyan/shared_data/REES46_NEXTITEM_V1_TRANSFER.json
/home/jovyan/shared_data/README_REES46.md
/home/jovyan/shared_data/verify_frozen_readonly.py
~/ahmedabdelhamed01/Privacy-Preserving-Shopping-Intelligence/   clean tree, nothing committed
    .local_ai/gru_v1/          git-excluded workspace
        FABLE_PLAN.md  DECISIONS.md  EXECUTION_PLAN.md  STATE.md  task_*_result.md
        code/          model, objective, evaluator, trainer, cache, baselines, tests, vendored loader
        cache/         train (1,000,000 rows) + vs (112,365 rows), 6.78 GB
        artifacts/     PILOT_RESULTS, BASELINES, BENCHMARK, PILOT_MANIFEST, SYNTHETIC_MEMORISATION
        experiments/gru/T-M_F0_item_only/ , T-M_F4_rich_side_information/
```

Disk: ~26 GB free of 49 GB. Both GPUs idle at close.

## 6.2 Invariants maintained throughout

- Frozen dataset never modified; nothing ever written inside it.
- Shared GPU environment never modified; `uv run` never used for GPU work.
- No git commits, no pushes, no PRs, no remote mutation.
- TEST and VALIDATION_CONFIRM never scored.
- No acceptance threshold changed after results were seen.
- Teammate resources (`~/ahmed`, `tunnel-ahmed`, `g8-ahmed`) never touched.
- No credentials stored on the shared server.

## 6.3 Open items

| Item | Status |
|---|---|
| **GRU full-training decision** | **awaiting owner** (section 5.6) |
| `post_freeze_smoke.py:101` vacuous check | identified, repair specified, **not yet applied** |
| Targeted reproducibility checks (normalizers, feature invariants) | specified, deferred to pre-full-run |
| Local `rees46_transfer/` (8.5 GB of chunks on the Windows machine) | can be deleted to reclaim space; the canonical dataset copy must be kept |
| SASRec | explicitly out of scope for this session |

---

*Prepared as the session record. All figures in this document were measured and independently re-verified rather than taken from tool self-reports.*
