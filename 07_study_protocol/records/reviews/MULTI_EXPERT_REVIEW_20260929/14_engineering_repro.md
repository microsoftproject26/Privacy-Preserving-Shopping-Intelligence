# 14 — Engineering and reproducibility (Part 13 of the multi-expert review, 2026-09-29)

Reviewer: Claude Opus 5.5, acting as a senior ML-infrastructure / reproducibility-track reviewer. READ-ONLY. This file is the only
file written. I did not contact the server, use git, run any project script or test suite, or open any RESULT_ROWS / ranks / RETRO_TEST
/ metrics file. I only read files and hashed them locally (sha256, plus a Python re-implementation of `v4eval.hashing.code_sha256`
that reads bytes only). Tags: **MEASURED** = I read or hashed it (path:line, or DL row = `company/DECISION_LOG.md` row number);
**REPORTED** = a project document states it and I did not re-check it; **INFERENCE**; **OPINION**. Paths are relative to
`.local_ai/ppsi_v4/` unless absolute. The review time was ≈ 17:25Z (MEASURED, `datetime.now(utc)`), and the DL tail was row 225 (16:45Z).

---------------------------------------------------------------------------------------------------------------------
## 1. What the project does for this part (MEASURED unless tagged)

### 1.1 STRICT FP32 determinism
- **Frozen numerics.** The frozen T3 flags are TF32 off for matmul and for cuDNN, `cudnn.benchmark` False, `cudnn.deterministic`
  True, `use_deterministic_algorithms` True, and `CUBLAS_WORKSPACE_CONFIG ":4096:8"` (`code/v4eval/frozen_config.py:53-54`). The
  launchers set them (`apply_runtime_flags`, :243-252, including `set_float32_matmul_precision("highest")`), and every public
  evaluation path checks them again (`check_runtime_flags`, :255-273). A mismatch fails closed.
- **Trainer.** The trainer refuses STRICT unless the launcher exported `:4096:8` or `:16:8` (`code/v4train/trainer.py:388-395`). A
  CUDA run is further pinned to the T3 value (:309-311). The trainer applies the determinism state and reads it back (:397-411), and it
  refuses a resume whose mode or CUBLAS setting differs from the recorded one (:413 onward). The mode enters `precision_id` and so
  the comparison key (`v4train/recipe.py:114-122`).
- **Evaluator self-check.** On every block, pass 2 recomputes the target scores and refuses anything that is not bit-for-bit equal
  (`code/v4eval/ranking.py:18`, :156).
- **RNG and data order.**
  - Per-seed data order comes from a PCG64 SeedSequence (`v4train/order.py:22-30`).
  - Checkpoints save the CPU and CUDA RNG states and restore them on resume (`v4train/trainer.py:649`, :748-750, :952-953;
    `fl/checkpoint.py:226-247`).
  - DP noise uses a per-round derived seed (`fl/dp.py:160`), so noise does not depend on the resume point.
- **Evidence.**
  - A16: STRICT training is bitwise run-to-run at full K (REPORTED, DL 29).
  - A40: STRICT r2 passed for D64_B2 and D256_B3 with two reps (REPORTED, DL 167/169). The pin writer refuses to run without it
    (`code/v4eval/device_pin.py:93-99`).
  - A40 vs A16: the T3 GPU receipt shows 12,635 (GRU) / 14,063 (SASREC) rank-diff rows (REPORTED, DL 163).
  - I read the local receipt copy `company/receipts/phase1/T3_GPU_RECEIPT_A40.json`: torch 2.13.0+cu130, CUDA 13.0, "NVIDIA A40",
    runtime flags equal to the frozen ones, `all_checks_pass` present.

### 1.2 Device classes and the pin
- **Class rule.** `eval_device_class` returns "A16-FP32-DET" when the GPU name contains "A16". Any other GPU gets its own class
  built from the name (`frozen_config.py:276-282`), so rows from different devices never mix.
- **Pin reader.** The A40 class is accepted only through `contracts/eval_device/EVAL_DEVICE_PIN_A40.json`. The reader checks the
  format, the authority, both receipts, the canonical self-sha, the exact live GPU name, and the torch and CUDA versions
  (`device_pin.py:31-70`). It does not check the driver.
- **Pin writer.** The pin is written once with `O_CREAT|O_EXCL` and fsync (`device_pin.py:122-130`).
- **Block invariant.**
  - `evaluator_code_sha256` (a sha over `v4eval/*.py`, `v4eval/__init__.py:21-23`) is a comparison-key field.
  - The A40 block is defined by the value 61b64e61…, and `code/v4eval/*.py` is frozen (README in `contracts/eval_device/`; DL 180).
  - **MEASURED now:** my recomputation over the local `code/v4eval/*.py` gives
    `61b64e61c1a2e4f5a6fecd42f0ad3b98de26562d13a07758573cc9496b8d962e`, which is the pinned value.
- **The T3 receipt tested an older evaluator bundle.** The A40 T3 receipt was made with the frozen A16-era bundle, whose
  `evaluator_code_sha256` is 6cbc3e6f… (field in `company/receipts/phase1/T3_GPU_RECEIPT_A40.json`). I hashed the bundle's files
  against local code: `v4eval/predict.py` and `v4eval/rows.py` DIFFER from the production evaluator, and the other 13 v4eval files are
  the same. So the T3 receipt certifies the old bundle on the A40, not the production evaluator. The production evaluator's A40
  determinism rests on three things:
  - the built-in pass-2 check;
  - STRICT r2;
  - the 8 A40 re-evaluations (REPORTED).
  REV-DELTA-01 F10 (`company/reviews/REV-DELTA-01_OPUS_REVIEW.md:360`) proposes re-running one ds3r2 re-evaluation under ds3r3 and
  comparing the ranks sha. DL 221–225 does not record that it was done.

### 1.3 Immutable code trees and provenance
- **How a tree is made.** `tools/code_sync.py` packs `code/` and `contracts/` into a PAX tar and writes a SYNC_MANIFEST
  {rel: sha256} plus the tar sha. The deploy scripts check the tar sha, check that member names are safe, extract to a new directory,
  re-hash every file, and `mv` it to a new `code_<name>` that must not already exist (`company/plans/ops_scripts/deploy_code_ds3r2.sh:4-31`).
  Contracts are NOT per-tree: the tree reads the shared `$V/contracts` registration store.
- **Local archive (MEASURED).** I searched for tars and manifests under `.local_ai/` and in the Claude temp folders.

| Tree | Local tar / manifest | Where | Used for (REPORTED) |
|---|---|---|---|
| runs_g1 r5/r5b/ops10 | yes | `runs_g1/*/` | G1 era |
| code_p3 | yes (f89b9ff4…) | `runs_p3/deploy_code_p3/` | GRU contest |
| code_p3b | yes (739053b8…) | `runs_p3/deploy_code_p3b/` | contest + G2 |
| code_ds / code_ds2 | yes (c50fd606… / 0456cb20…) | `runs_ds/deploy*/` | C_SMALL, SMALL ladder, baselines |
| code_ds3 / ds3r1 / ds3r2 | yes (49e06d12… / b707868b… / 217d1c9d…) | `runs_ds3*_deploy/` | A40 receipts, pin, 8 re-evaluations, LR screen ×0.5–×2, **C_FULL s2026/s2027**, KNN_EMB |
| code_ds3pre | **none found** | — | "interim measurement tree" (DL 161), incl. the DATA-08 C_FULL manifest build (INFERENCE from DL 161/162) |
| code_ds3r2s | **none found** | — | S1b screens ×4/×8/×16 (→ LR_SELECTION_EXT, binding for every FL arm) and LADDER_S1a (DL 193, 198) |
| code_ds3r2p | **none found** | — | PRE (θ_pre for FA_WARM), with the on-server patch of `pretrain_small.py` (DL 207, 212–213) |
| code_ds3s2 | tar ccda9d64… **only in an OS temp scratchpad** (`C:/Users/Ahmed/AppData/Local/Temp/claude/.../bd85e2a3-.../scratchpad/code_snap_s2.tar`); sha not in DL 218 | temp | S2 screens (DL 218) |
| **code_ds3r3 (production)** | tar **15d8c994…** (= DL 223) **only in the same temp scratchpad** (`.../scratchpad/code_ds3r3.tar`) | temp | every later stage, device lanes 7/8 (DL 223–224) |

- **ds3r3 matches local code.** Its 659 members are byte-identical to the current local `code/`: 0 missing, 0 differing (MEASURED
  by hashing each member). No local code or contracts file changed after 16:19Z (MEASURED by mtime), so the local `code/` IS
  code_ds3r3 at present. That holds only until the next local edit.
- **Composition of r2s.** The DL reconstructs code_ds3r2s as "code_ds3r2 + launch_fl ca93ceea + lr_screen 651d1c84" (DL 193).
  DL 194 records the FL-09 worker's judgement that the interim tree "must be the full current local tree … not ds3r2 + one file".
  The two statements conflict, and no manifest of r2s exists to settle which one is true.
- **What a run records about its code.** `LAUNCH_*.json` records `argv` (relative), the host, and the smoke flag
  (`v4int/launch_central.py:295`). The trainer records `trainer_code_sha256`, `models_code_sha256` and `evaluator_code_sha256`
  (`v4train/trainer.py:133-138`, :693-721). FL and device runs record `fl/*.py` + launcher (`v4int/launch_fl.py:969-972`,
  `launch_device.py:346-349`).
  - No record carries the tree name or a whole-tree hash.
  - Data-path modules (`v4int/providers.py`, `cache.py`, `v4hist`, `v4data*`) and shims (`pretrain_small.py`,
    `launch_central.py`) are covered by no recorded hash.
  - So the on-server patch of `pretrain_small.py` in ds3r2p (DL 212) shows up in no run record.
  - REV-DELTA-02 m4 asked for a DL row with the ds3r2p shas of `pretrain_small.py` / `launch_central.py` against a5550479 /
    a4ded7da (`company/reviews/REV-DELTA-02_OPUS_REVIEW.md:185-188`; carried as "record-only for Ops" in
    `company/handoffs/FIX-DELTA-02.md:71-72`). **No such DL row exists (MEASURED, grep over company/).**
- **"Production only from the reviewed tree" was broken.** The rule is "production runs only from the reviewed tree (code_ds3r3) —
  never from code_ds3pre / r1 / r2" (`company/plans/MASTER_PLAN_V5.md:281`; DL 168). Yet C_FULL s2026/s2027 (the K15/K17–K19
  denominators) were launched from code_ds3r2 (DL 175–176, 193). ds3r2 carried Codex's then-unreviewed edits (DL 164–166). They were
  accepted afterwards: INT-06 (DL 180) and REV-DELTA-01 §8 ("C_FULL. Valid … Resume it with the tree that started it",
  `REV-DELTA-01_OPUS_REVIEW.md:311-314`).
- **Tree write protection.** Nothing makes the trees read-only: no `chmod` in any ops script (MEASURED: grep). The server account
  is shared (`jovyan`; memory note "files never prove ownership").

### 1.4 Registration files, self-hashes, O_EXCL
- **O_EXCL.** It is used for the pin, the selection files, ARCHITECTURE_LOCK and G2_RESULT (REPORTED DL 131; code for the pin
  MEASURED).
- **Self-hashes verified by recomputation (MEASURED).**
  - `company/gates/ANA_CONTRASTS.json` (51f2f116…) and `ANA_CONTRASTS_v2.json` (bac7aa4f…) verify under compact JSON with
    **ensure_ascii=False**.
  - `contracts/fl/LR_SELECTION_EXT.json` (e23431af…, file 7d922e4a… = DL 218) verifies under compact **ASCII** JSON.
  - `G1_APPROVED.json` matched none of my four candidate rules (its rule is presumably different; not an error claim).
  - So two canonicalisation rules coexist, which is the bug class of DL 166.
- **Lock hashes (MEASURED sha256).** ARCHITECTURE_LOCK fd201013… (= DL 131); DATA_PROTOCOL_LOCK_V2 file 905ee4f9… (= DL 54 "file
  905ee4f9…"). MASTER_PLAN §2's acceptance line lists "bc699a97…", which is that lock's *self*-sha, beside file shas. A literal
  `sha256sum` check against it would fail.
- **Registration files that exist only on the server (MEASURED absent locally).** The local `contracts/` mirror is incomplete:
  - `contracts/eval_device/EVAL_DEVICE_PIN_A40.json` (only the README is local);
  - `contracts/fl/LR_SELECTION.json` (06B, sha cdf6663d…; only the EXT file is local);
  - `contracts/strata/STRATA_CUTPOINTS_MAIN.json` (the local dir is empty);
  - `REGISTRY_V2_PRETRAIN_EXT`;
  - later `contracts/device/*`, `contracts/dp/*`.
- **TEST_EXPOSURE_REGISTER.** It is hash-chained (`tools/test_exposure_register.py:44-56`) with 4 local entries (MEASURED). The DATA-09
  gate binds to a MANIFEST_BUILD entry on the **server** copy (DL 174). Two copies of an append-only chain can diverge, and `append`
  takes no file lock (INFERENCE).

### 1.5 Coordination between concurrent AI sessions (DECISION_LOG as the only channel)
- **Incidents on 2026-09-29 alone (MEASURED from the DL).** Seven, most within about 45 minutes:
  - duplicate row id 169 (DL 171);
  - typed-ahead timestamps corrected twice (DL 171, 179);
  - duplicate workers ANA-02 / FL-09 / FIX-B-D (DL 172, 185);
  - amendment 06F **overwritten** by a second session's Write (DL 185), then restored byte-identical (DL 188);
  - two fixture collisions in concurrent test runs (DL 201, 205);
  - one false "digest changed" alarm (DL 205), corrected in DL 210 as a stale test helper.
- **Rules adopted after the incidents.** "Tail max + 1", `date -u` inside the writing command, one owner per file set (DL 172), DL
  only (DL 188).
- **Current state.** The Fable session closed (DL 214–216), so one operating session remains (REPORTED). The present multi-expert
  review adds readers but not writers.

### 1.6 Test-suite health
- **Convention.** NC = negative-control tests, marked xfail(strict). They are re-run with `--runxfail -k test_nc_` to prove they
  fail.
- **Quiescent evidence (MEASURED).** REV-DELTA-02 recorded pre and post hashes (556 files) and ran all 12 suites with rc 0. Every NC
  set failed as intended (`.local_ai/_pytest_tmp/REV-DELTA-02/SUMMARY.txt`), e.g. tests_fl 248 / 75, tests_int 239 / 67,
  tests_train 214 / 61. REV-DELTA-02b then shows tests_int 253 / 76, fl 248 / 75, ana 101 / 25 (`.../REV-DELTA-02b/SUMMARY.txt`),
  consistent with DL 221–222.
- **Count trail (REPORTED).**

| DL | tests_int | tests_fl |
|---|---|---|
| 205 | 194 / 51 (NC set changed mid-run) | — |
| 209 | 204 / 55 | — |
| 212 | 215 / 57 | 239 / 73 |
| 219 | 238 / 67 | 248 / 75 |
| 221 | 253 / 76 | — |

  The trail is monotone and explained by new tests.
- **In-tree receipts are stale (MEASURED).**
  - `code/tests_fl/receipts/pytest_output.txt` = "1 failed, 231 passed" (12:29Z).
  - `code/tests_train/receipts/pytest_output.txt` = "2 failed, 208 passed, 9 errors" (12:19Z, the DL 201 collision).
  - Both files were packed into code_ds3r3.
- **Environment of the local suites.** They run on Windows with the `next_item_eda` venv (`.../REV-DELTA-02/run_suites.sh:5`),
  torch 2.9.1+cpu (`company/RESOURCE_BOARD.md`). Production runs Linux, torch 2.13.0+cu130 on the A40. The ds3r3 deploy recorded
  only "imports OK" (DL 223). MASTER_PLAN §2 requires server suites on deploy (`MASTER_PLAN_V5.md:50`). ds3r2 had 10 targeted
  server tests (DL 166).

### 1.7 Run management and resources
- **Lanes.** Detached `nohup setsid` lane scripts with `job`/`wait_file`/`wait_grep`. The lane stops on the first failure, and resume
  is "same command + `--resume`" (`company/plans/ops_scripts/setup_lanes_v1.sh:23-35`). OMP/MKL threads are 2 per process (:22).
- **The launch script is edited in place.** `setup_lanes_v1.sh` was modified at 16:19:50Z (MEASURED mtime). The 08:40Z launch
  (lanes v2) used an earlier version, which no longer exists locally.
- **Throughput.** The FL loop is CPU-bound: ≈ 10.2 visits/s per process, GPU 19 %, 1 core (REPORTED, MASTER_PLAN addendum B).
- **Contention.**
  - C_FULL was at 2.0 EFE after ≈ 7.3 h at 16:05Z (REPORTED, DL 221).
  - INFERENCE: ≈ 3.6 h/EFE, so 6.0 EFE ≈ 06:45Z on 30 Sep and 8.0 EFE (continuation) ≈ 14:00Z. That is ≈ 22–29 h per seed,
    consistent with DL 198/213 "25–30 h".
  - Per-run wall time is not a registered quantity (DL 179).
- **Resume evidence.** Resume is bitwise on CPU and synthetic data (REPORTED, INT-01 DL 32; FL-10 server moments DL 211). I found
  no A40 bitwise-resume receipt at production scale, yet DL 213/216 plan to pause C_FULL s2027 "(it resumes bitwise)".

### 1.8 Records
- **Thesis living reference (`.local_ai/thesis_reference/`).**
  - STATE.md header still says "Last update: 2026-09-26T19:27Z" (`STATE.md:3`). Its last appended block is 2026-09-28T22:25Z.
  - CHANGELOG's last entry is 2026-09-28T22:25Z. The EN reference ends at §14.17. All three have mtime 2026-09-28T22:06Z.
  - SOURCE_REGISTER ends at S508 (DL 190). DL 191–225 is not captured: 06I/06J/06K, REV-DELTA-01/02/02b, FL-10, INT-07, DATA-10,
    code_ds3r3, the S1b selection, the GitHub pushes. RESUME_HERE §2n itself lists "next source S509".
- **Company boards.**
  - RESUME_HERE was last modified at 13:44Z, before DL 216–225 (MEASURED mtime).
  - `RISK_REGISTER.md` and `RESOURCE_BOARD.md` date from 2026-09-26 (A16, 64 GiB, 49 GiB disk). They contain no risk entries for the
    multi-session concurrency, the A40 device switch or the CPU-bound FL.
  - TASK_BOARD.json dates from 2026-09-27.
  - The charter's G3 gate (FL/PFL/LO protocol, `company/COMPANY_CHARTER.md` "Gates") has no gate file; amendments 06B–06K and the
    REV-FL-01 review did that work.
- **GitHub (DL 217, 225; memory `github-daily-push-workflow.md`).**
  - A daily docs push to the owner's branch, human-style commits.
  - Forbidden in pushed files: AI attribution, internal codes (amendment ids, DL numbers), "ppsi_v4", ".local_ai".
  - The clean repo port comes after the TEST look.
  - One factual inaccuracy was caught and fixed (DL 225).

---------------------------------------------------------------------------------------------------------------------
## 2. State of the art (verified citations; others marked UNVERIFIED)

- **Pineau et al., 2021. "Improving Reproducibility in Machine Learning Research (A Report from the NeurIPS 2019 Reproducibility
  Program)."** J. Pineau, P. Vincent-Lamarre, K. Sinha, V. Larivière, A. Beygelzimer, F. d'Alché-Buc, E. Fox, H. Larochelle. JMLR
  22(164). https://www.jmlr.org/papers/v22/20-303.html (arXiv:2003.12206).
  - Three levers: a code-submission policy, a reproducibility challenge, and the ML reproducibility checklist.
  - Reproducibility means the same results from the same code and data.
- **NeurIPS Paper Checklist (2025 Call for Papers / Reviewer Guidelines).**
  https://neurips.cc/Conferences/2025/CallForPapers, https://neurips.cc/Conferences/2025/ReviewerGuidelines.
  - It asks for "the exact command and environment needed to run to reproduce the results".
  - It also asks for a description of the compute resources.
- **ACM, "Artifact Review and Badging — Current" (v1.1, 2020).**
  https://www.acm.org/publications/policies/artifact-review-and-badging-current.
  - "Artifacts Available" requires a permanent archival repository with a DOI or another unique identifier.
  - "Results Reproduced" accepts agreement "within a tolerance"; exact bitwise equality is not required.
- **PyTorch docs, "Reproducibility"** (notes/randomness, 2.14). https://docs.pytorch.org/docs/2.14/notes/randomness.html.
  - Quote: "Completely reproducible results are not guaranteed across PyTorch releases, individual commits, or different platforms
    … may not be reproducible between CPU and GPU executions, even when using identical seeds."
  - `cudnn.benchmark=False` makes the algorithm selection deterministic.
- **NVIDIA cuBLAS documentation, "Results reproducibility"** (cuBLAS 13.x). https://docs.nvidia.com/cuda/cublas/.
  - Bitwise-identical results hold only "on GPUs with the same architecture and the same number of SMs" and for the same toolkit
    version.
  - Multiple concurrent streams break this unless `CUBLAS_WORKSPACE_CONFIG` is `:16:8` or `:4096:8`.
  - This directly supports the A16/A40 class separation and the co-scheduling-independence argument.
- **Sculley et al., 2015. "Hidden Technical Debt in Machine Learning Systems."** D. Sculley, G. Holt, D. Golovin, E. Davydov, T.
  Phillips, D. Ebner, V. Chaudhary, M. Young, J.-F. Crespo, D. Dennison. NeurIPS 28.
  https://papers.nips.cc/paper/5656-hidden-technical-debt-in-machine-learning-systems. Relevant items:
  - configuration debt;
  - glue code and pipeline jungles;
  - undeclared consumers;
  - dead experimental code paths.
- **Bouthillier et al., 2021. "Accounting for Variance in Machine Learning Benchmarks."** MLSys 3, 747–769.
  https://proceedings.mlsys.org/paper_files/paper/2021/hash/0184b0cd3cfb185989f858a1d9f5c1eb-Abstract.html. Bitwise determinism
  removes variance from reruns; it does not replace sampling variance over seeds, data and hyperparameters.
- **Pre-registration in ML.** NeurIPS 2020 workshop "The pre-registration experiment" (Bertinetto, Henriques, Albanie, Paganini,
  Varol; PMLR vol. 148, https://proceedings.mlr.press/v148/) and the 2021 edition (PMLR vol. 181, https://proceedings.mlr.press/v181/).
  The protocol is reviewed and time-stamped by a third party before results are known.
- **FL simulation frameworks.**
  - Flower: Beutel et al., arXiv:2007.14390 (https://arxiv.org/abs/2007.14390), which simulates up to 15 M clients on a pair of GPUs.
  - FedScale: Lai et al., ICML 2022, PMLR 162 (https://proceedings.mlr.press/v162/lai22a.html), a reproducible FL benchmark and
    runtime.
  - FLUTE: Dimitriadis et al., arXiv:2203.13789 (https://arxiv.org/abs/2203.13789; Microsoft Research), reporting up to 42× speed-up
    against other platforms.
  - These frameworks show that the CPU-bound per-visit loop is a known and solved engineering problem, through batched and parallel
    client simulation.
  - TFF: not re-verified here (UNVERIFIED as a citation; the tool itself is well known).
- **Experiment-tracking practice** (MLflow / W&B / DVC-style run registries: immutable run id → code commit + environment + data
  hash + config). This is general industry practice and not a single citable source (OPINION).

---------------------------------------------------------------------------------------------------------------------
## 3. Comparison with the state of the art

### 3.1 Strengths
1. **Determinism exceeds typical top-venue practice.**
   - Every flag of the PyTorch reproducibility note is applied and then *read back*, and a mismatch fails closed.
   - The CUBLAS setting is exported before CUDA initialises and checked.
   - The evaluator recomputes scores (pass-2) on every block.
   - STRICT is bitwise run-to-run, measured on both GPU classes.
   - The per-round DP noise seed makes DP arms reproducible after a resume.
   - This is better than the NeurIPS checklist asks for. Few MSc (or many published) FL papers can claim bitwise reruns.
2. **The device-class separation is correct and evidence-based.**
   - The cuBLAS guarantee requires the same architecture and SM count. A16 and A40 differ, and the measured 12,635 / 14,063 rank
     diffs confirm it.
   - The project made the A40 its own comparison block, re-evaluated C_BIG/C_SMALL inside it, and pinned it with O_EXCL and
     receipts.
   - This is a textbook application of the vendor guarantee.
3. **Evaluator-identity-in-the-key works.**
   - It caught a real incident: Codex's post-pack edit of `v4eval/device_pin.py` (DL 180), which would have split the A40 block
     silently.
   - It also caught the ensure_ascii canonicalisation bug before the pin was written (DL 166).
   - I re-verified 61b64e61 locally.
4. **Immutable, content-addressed deploys.** The pack builds a tar with a manifest. The deploy verifies the tar, checks member
   names, re-hashes every file and moves the tree to a new name. A running run is never exposed to a code change. This exceeds most
   academic practice.
5. **Negative-control tests with a `--runxfail` proof.** Quiescent-tree review runs with pre and post hash files (REV-DELTA-02)
   are strong QA discipline. The failure modes that did occur (fixture collisions, a stale helper) were all detected.
6. **Sealing.** Every metric-bearing artefact of a run goes to `sealed/`. TEST is gated and exposure-registered with a hash
   chain. This is operationally stronger than a normal pre-registration because the analysis cannot see values early by accident.

### 3.2 Weaknesses
1. **Provenance is traceable, but mostly through DL prose and a temp folder, not durable artefacts.**
   - The production tree code_ds3r3 and the S2 tree exist locally only in an OS temp scratchpad.
   - r2s, r2p and ds3pre have no local tar or manifest at all.
   - The r2s composition is internally contradictory (DL 193 vs 194).
   - The link from a run to its tree is not recorded in the run itself.
   - Measured against ACM "Artifacts Available" and the NeurIPS "exact command and environment", this is the largest gap.
2. **No environment lock.** No pip freeze, collect_env, cuDNN version, driver/VBIOS or SM count is archived. The pin records the
   driver but does not check it. This matters because cuBLAS bitwise identity is toolkit-version-specific, and PyTorch makes no
   guarantee across releases.
3. **The test evidence runs on a different stack than production.**
   - The suites run on Windows with torch 2.9.1 CPU; production runs Linux, torch 2.13 CUDA.
   - The θ0 hashes were already found to be torch-build-dependent (DL 25; R15).
   - ds3r3 got no server suite run, only "imports OK" (DL 223).
4. **Records lag the work.**
   - The thesis reference is ≈ 35 DL rows behind.
   - RISK_REGISTER and RESOURCE_BOARD are three days stale.
   - RESUME_HERE was not updated after DL 215.
   - The in-tree test receipts show failures that were later fixed.
   - CLAUDE.md requires updates "after every major decision".
5. **Coordination by a markdown file edited by several agents** had no locking. It produced seven incidents in one morning, one of
   them a registered amendment overwritten (DL 185). Pre-registration credibility rests on "registered before any read". The
   registration timestamps are self-attested, hand-typed in places (DL 171/179 corrected them), and not externally anchored.
6. **Process overhead (OPINION).**
   - In ≈ 4 days: 226 DL rows, 79 handoffs, 73 task packets, 51 review files, 11 amendments in 24 h (06–06K), 8+ code trees
     (MEASURED counts: `ls company/handoffs|tasks|reviews`).
   - For an MSc this is disproportionate as *human-facing* complexity, even though agents did the work.
   - The owner must be able to defend every tree and amendment at the defence.
   - Tree proliferation (r2 → r2s → r2p → s2 → r3) is itself the main provenance risk.

### 3.3 Errors (factual or internal inconsistencies)
- **E-a.** The MASTER_PLAN §2 acceptance line compares `sha256sum DATA_PROTOCOL_LOCK_V2.json` with "bc699a97…", which is the lock's
  self-sha; the file sha is 905ee4f9… (MEASURED). Minor, but a literal re-run of the package's verification would fail.
- **E-b.** DL 193 (r2s = ds3r2 + 2 files) and DL 194 ("must be the full current local tree") cannot both describe the tree that
  produced the ×4/×8/×16 CALIB values. Those values feed LR_SELECTION_EXT (×8, DL 218), which is binding for every FL arm.
  - INFERENCE: REV-DELTA-01 §8 argued that the digests are equal across trees (`_strip_lr` spec equality), so the values are very
    likely valid either way. The record is still wrong or ambiguous.
- **E-c.** "Production runs only from the reviewed tree" (MASTER_PLAN_V5.md:281) was not true for C_FULL (ds3r2), whose K15/K17–K19
  are confirmatory denominators. It was accepted by review afterwards, but it must be disclosed and not described as "reviewed-tree
  production".
- **E-d.** DL 213/216 assert that C_FULL s2027 "resumes bitwise" if paused. I found no A40 production-scale resume receipt, so this
  is INFERENCE from CPU tests.

### 3.4 Unrealistic choices
- Relying on a single shared server as the only home of irreplaceable evidence: sealed rows, checkpoints of 335 MB each, the pin,
  selection files. Bulk transfer to the local PC measured 0.05–0.07 MiB/s (RISK R12), and the server was shut down or restarted
  three times in 36 h (DL 145, 153, 158). One disk incident before the TEST look ends the study. This is the real engineering risk,
  more than any numerics issue.
- Co-scheduling up to 11 GPU processes (DL 221) to "reach 100 %", while the FL loop is 1-core CPU-bound. This stretches C_FULL to
  ≈ 25–30 h per seed, and every added lane increases the chance of an OOM or kill that stops a lane. It is scientifically harmless
  under STRICT (the cuBLAS doc supports independence from co-scheduling with the workspace config). It is operationally fragile.

### 3.5 Missing pieces
- A reproducibility package manifest: trees with tar shas, environment lock, data rebuild commands, per-run command lines,
  hardware, and a mapping from run to tree.
- An external timestamp for registrations.
- A durable archive with a DOI (for example Zenodo) for "Artifacts Available".
- A canonical-hash specification for the package verifier.
- A statement of the reproducibility tolerance across hardware: bitwise on the same SKU and stack; ACM-style agreement within a
  tolerance elsewhere, using the measured A16-vs-A40 difference as the empirical tolerance.

---------------------------------------------------------------------------------------------------------------------
## 4. Recommendations

Legend: priority MUST / SHOULD / NICE / DISCLOSE-ONLY; action KEEP / CHANGE / REDO / ADD. Costs are INFERENCE. "Pre-reg" = whether a
new registration (amendment) is needed. No recommendation below requires reading any sealed value or re-running any finished run.

| ID | Item | Priority | Action | Cost | Affects | Pre-reg (status) |
|---|---|---|---|---|---|---|
| R1 | **Archive every evidence-bearing tree durably.** Copy `code_ds3r3.tar` (15d8c994…) and `code_snap_s2.tar` (ccda9d64…) out of the OS temp scratchpad into `ppsi_v4/runs_ds3r3_deploy/` and `runs_ds3s2_deploy/`, with a SYNC_MANIFEST each and a DL row with both tar shas. For code_ds3r2s, code_ds3r2p and code_ds3pre, have Ops run a read-only `sha256sum` over the server trees, store the manifests locally, and record in a DL row which composition r2s really had (settles E-b). | MUST | ADD | 1 h (Ops, read-only on server) | records of finished and running runs; none re-run | no |
| R2 | **Run → tree provenance table** (thesis Table 6.1 / App. E). Per run dir: tree, tree-manifest sha, lane script, LAUNCH config sha, trainer/models/evaluator/fl code shas, GPU index, device class, pin sha, co-scheduled with, start/end. Build it by script from lane logs and LAUNCH/RUN_START (metadata only, never RESULT_ROWS). | MUST | ADD | 2–4 h Sonnet | reporting only | no |
| R3 | **Close θ_pre provenance** (REV-DELTA-02 m4, still open). DL row with the server sha of code_ds3r2p `v4int/pretrain_small.py` and `launch_central.py` vs a5550479 / a4ded7da. If they differ, diff and review before any FA_WARM launch. | MUST (before FA_WARM) | ADD | 10 min | PRE (running); FA_WARM, K20/K21 | no (yes only if the bytes differ in behaviour) |
| R4 | **Mirror every server-side registration file locally** at registration time, with a sha check: EVAL_DEVICE_PIN_A40.json, LR_SELECTION.json (06B, cdf6663d…), STRATA_CUTPOINTS_MAIN, REGISTRY_V2_PRETRAIN_EXT / _TEST_EXT, S_RECORD, SELECTION_*, FL_TUNING_SELECTION, the server TEST_EXPOSURE_REGISTER. Record in App. B of the pre-registration record. | MUST | ADD | 30 min + per event | records | no |
| R5 | **Environment lock.** Read-only on the server: `python -m torch.utils.collect_env`, `pip freeze`, `torch.backends.cudnn.version()`, `nvidia-smi -q` (driver, VBIOS, SM count) for both A40s. Store and hash it, and cite it next to the pin. Add the local test stack (Windows, torch 2.9.1+cpu) as a separate record. | MUST | ADD | 15 min | package; none re-run | no |
| R6 | **Reproducibility package definition** (MASTER_PLAN §12.3, extended). Contents: byte-identical trees (R1); environment (R5); data rebuild from the frozen ledger 860e5a1c… with exact commands for data_v3 / data_cfull / data_pretrain / data_test; lane scripts as launched; seeds; hardware; the canonical-hash rules (R15); a verifier script that re-checks every sha. Deposit it in an archival repository with a DOI if the data licence permits (REES46 licence UNVERIFIED). State the tolerance: bitwise on the same SKU and stack, otherwise ACM "Results Reproduced" agreement, with the A16-vs-A40 C_BIG / C_SMALL deltas (MASTER_PLAN §12.2 row 7.2) as the empirical device tolerance. | MUST (for any reproducibility claim in the thesis) | ADD | 1 day | write-up | no |
| R7 | **Public-repo compatibility.** The owner's rules (no internal ids, cleaned human-readable port after TEST) conflict with a verifiable package. A rewritten port is not the code that produced the numbers. Either (a) publish or deposit the byte-identical archive (R6) as a separate artefact linked from the clean repo, or (b) prove the port equivalent (for example, reproduce one A40 re-evaluation ranks sha and one short STRICT training leg bitwise), or label the port "re-implementation, not the evidence code". Keep the pre-registration ids and hashes in the deposited record even if they are scrubbed from commit messages. AI-assistance disclosure: follow the university's policy (UNVERIFIED which policy applies); the commit history will not show it. | MUST (claim accuracy) | CHANGE (package plan) | 0.5 day at port time | public artefacts | no |
| R8 | **Server suite run for code_ds3r3** (CPU, own basetemp under `runs_ds3/deploy_r3/tmp`, sequential, with receipts), as MASTER_PLAN §2 requires. Replace the stale in-tree receipts (tests_fl "1 failed", tests_train "2 failed / 9 errors") with the quiescent REV-DELTA-02/02b summaries, or add a pointer file. | SHOULD (high) | ADD | 40 min CPU | none (tests only) | no |
| R9 | **A40 bitwise-resume receipt** before any planned pause of C_FULL s2027 (DL 213/216). A `--smoke --smoke-interrupt-at` leg on the A40 from code_ds3r2 (the C_FULL tree), compared bitwise with an uninterrupted leg. | SHOULD (high) | ADD | 15–20 min GPU | the C_FULL pause decision | no |
| R10 | **Production-evaluator A40 check** (REV-DELTA-01 F10). Re-run one ds3r2 re-evaluation (C_BIG s2026 PB) under ds3r3 into a new dir and compare the ranks and rows shas (hash equality only; never open the values). This closes the gap that the T3 A40 receipt certified the old bundle 6cbc3e6f…, not 61b64e61…. | SHOULD (high) | ADD | 5 min GPU | A40 block integrity | no |
| R11 | **Off-server backup** of the small irreplaceable evidence: sealed RESULT_ROWS / ranks files as opaque blobs (copied, never opened), all JSON receipts and registrations, the SHA lists of large checkpoints. Ask the owner for university storage for the checkpoints. Copy by hash-verified transfer. | SHOULD (high) | ADD | 1–3 h wall | risk only | no |
| R12 | **Write-protect and verify trees.** `chmod -R a-w` each `code_*` after deploy and each registration file after O_EXCL. At lane start, re-hash the tree against its manifest (a small addition to `_common_<tree>.sh`, no code-tree edit). | SHOULD | ADD | 30 min | future launches only | no |
| R13 | **Coordination hardening** (if a second writer session ever runs again). DL rows appended by one script (max + 1, `date -u`, file lock) or rows as O_EXCL files. contracts/ and gates/ written only through an O_EXCL helper, never an editor Write. One session is the single writer of RESUME_HERE and the thesis reference. | SHOULD | CHANGE (process) | 1 h | process | no |
| R14 | **Bring the records up to date.** Thesis reference sources S509+ (DL 191–225), STATE (fix its "Last update" header), CHANGELOG, EN §14.18; RESUME_HERE block for DL 216–225; RISK_REGISTER (new: server single point, multi-session coordination, A16→A40, CPU-bound FL / contention, tree proliferation, temp-folder archives); RESOURCE_BOARD (2× A40, 384 GiB, 345 GB); TASK_BOARD. | SHOULD | ADD | 1–2 h Sonnet | records | no |
| R15 | **Canonical-hash spec.** Document per file which rule its self-sha uses: ensure_ascii False (ANA_CONTRASTS*) vs ASCII (pin, LR_SELECTION_EXT, TEST_EXPOSURE_REGISTER), and G1_APPROVED's rule. Ship one verifier. Do not rewrite existing files. Fix the MASTER_PLAN §2 self-sha / file-sha mix-up (E-a) in the package README. | SHOULD | ADD | 1 h | package | no |
| R16 | **External timestamp anchor for registrations.** Each day, push or e-mail one sha256 of a tar of `contracts/` + `company/gates/` + DL. It contains no internal ids, so it is compatible with the owner's GitHub rule. Anchor retroactively now as well. This turns self-attested "registered before any read" into third-party-timestamped. | SHOULD | ADD | 5 min/day | credibility of pre-registration | no |
| R17 | **Version the ops scripts as launched.** Keep an immutable copy (with sha) of every launcher script version actually run (setup_lanes_v1.sh was edited in place at 16:19:50Z). The server's generated `runs_ds3/queues/*.sh` are the ground truth; mirror them. | SHOULD (low) | ADD | 20 min | records | no |
| R18 | **Future runs record the tree identity** (code root + tree-manifest sha + git-like tree id in LAUNCH). This needs a v4int edit (not v4eval) and so a new tree; do it only at the next natural tree boundary, never for running runs. R2 covers the past. | NICE | ADD | 1 h code + review | future runs | no |
| R19 | **Co-scheduling independence as MEASURED.** One short STRICT leg alone vs co-scheduled on the A40, compared bitwise. Today this rests on the DL 82 ruling and the cuBLAS documentation (which supports it). | NICE | ADD | 20 min GPU | disclosure strength | no |
| R20 | **Do not optimise the CPU-bound FL loop now.** A batched or vmap client simulation (FLUTE / Flower style) would change the code identity and split the FL block mid-study. Disclose the throughput (≈ 10 visits/s, GPU 19 %), co-scheduling, and the per-seed wall times; list batched simulation as future work. | DISCLOSE-ONLY | KEEP | 0 | write-up | no |
| R21 | **Disclose the provenance deviations.** C_FULL from code_ds3r2 (reviewed afterwards: INT-06, REV-DELTA-01 §8). S1b ×4/×8/×16 and LADDER_S1a from interim r2s (DL 193/197/204). S2 from ds3s2 (REV-DELTA-02 §7). PRE from r2p with the on-server patch (DL 212–213; R3). The C_FULL manifest built from interim ds3pre with d8 code 720f8892… (`company/receipts/phase1/DATA08_SERVER_RECEIPT.json` "code"), content-verified by its registry. Codex's overnight deviations (serial queue, omitted MUST arms, post-pack v4eval edit reverted: DL 168, 180). | DISCLOSE-ONLY | ADD (text) | 1 h | thesis threats / App. H | no |
| R22 | **Keep** STRICT FP32 with read-back, the frozen v4eval (61b64e61), the A40 pin, O_EXCL registrations, sealed outputs, NC tests with `--runxfail`, and quiescent-tree review runs with pre/post hash files. | — | KEEP | 0 | — | no |
| R23 | **Proportionality (OPINION).** Freeze the process from here: no new tree except one per ladder wave, each with R1 archiving; no new amendment letter unless a registered question changes; one writer. The owner should get a one-page "tree and amendment map" for the defence. | SHOULD | CHANGE (process) | 1 h | defendability | no |

MUST summary: R1–R7. None of them requires re-running finished work or a new pre-registration. R3 blocks FA_WARM until it is
closed.

---------------------------------------------------------------------------------------------------------------------
## 5. Score: **7 / 10**

The numerics engineering alone would score 9. That covers STRICT FP32 with fail-closed read-back, per-block bitwise recomputation,
RNG and noise-seed discipline, and a device-class separation that follows the cuBLAS guarantee and is backed by measured A16/A40
diffs. The evaluator-hash-in-key design demonstrably caught an unauthorised edit, and the NC-tested suites run on a quiescent tree
with hash proofs. That is above what the NeurIPS checklist or typical FL papers deliver.

The score is pulled down by the gap between being traceable and being reproducible from artefacts:
- the production tree and the S2 tree exist only in an OS temp folder, and three interim trees that produced binding or
  confirmatory inputs have no archive at all;
- no environment lock exists;
- several registration files live only on the server;
- a run does not record its tree;
- one production-relevant provenance item (θ_pre) is still open;
- the suites are evidenced on a different OS and torch stack than production;
- the records (thesis reference, risk and resource boards, RESUME_HERE) lag by a day or more;
- all evidence sits on one shared server.

None of this invalidates a finished run, and every fix is cheap (R1–R7 ≈ 1.5 days in total). Until they are done, the thesis cannot
honestly claim "Artifacts Available" or a reproducible package. The process overhead is disproportionate for an MSc and is itself the
source of most provenance risk (tree proliferation, multi-session races).

---------------------------------------------------------------------------------------------------------------------
## 6. UNVERIFIED items and open questions

1. The actual file composition of code_ds3r2s and code_ds3r2p on the server (DL 193 vs 194; REV-DELTA-02 m4). Only a server-side
   read-only hash listing can settle it.
2. Whether STRICT r2 (the pin's strict receipt) ran while other processes were resident. This bears on R19.
3. Whether the server TEST_EXPOSURE_REGISTER and the local copy (4 entries) are the same chain.
4. The REES46 dataset licence terms for redistribution of derived manifests (affects R6's DOI deposit). UNVERIFIED.
5. Which Queen's University (or supervisor) AI-use disclosure policy applies to the thesis and repository. UNVERIFIED (R7).
6. The TFF citation was not re-verified (UNVERIFIED). The other citations in §2 were verified by web search on 2026-09-29.
7. The `G1_APPROVED.json` self-sha rule: it matched none of the four rules I tried. It is not claimed to be wrong; the rule should be
   documented (R15).
8. Whether the C_FULL s2026/s2027 run dirs will resume under code_ds3r2 if a pause or kill happens. REV-DELTA-01 advises "resume with
   the tree that started it". This is not tested on the A40 (R9).
9. The C_FULL and PRE ETAs in §1.7 are INFERENCE from REPORTED marks (DL 221). They are not results.
