# PREG1 amendment 07 — addendum §B16: ML-1M moves to the deterministic A40 GPU device class (REGISTERED 2026-10-03, DL 433)

- **Basis:** Fable re-review of DL 368 Decision 1, made at the owner's request (company/reviews/FABLE_GATE_ENDGAME_SCRIPTS_20261003.md
  §8, DL 404). Code: BM-03 + REV-BM-03 (PASS_WITH_MINORS), BM-05 + REV-BM-05 (PASS_WITH_MINORS), BM-04 + REV-BM-04 (PASS_WITH_MINORS; driver run_ml1m_gpu.sh 1a623877…). Fable concurrence: §11 (CONCUR WITH EDITS; E1–E5 applied).
- **Status (MEASURED by Ops):**
  - No ML-1M inner-validation or TEST value has been read; trackb-select has not run on ML-1M (B15 Status).
  - The GPU evidence below comes from smoke runs only (SMOKE_NOT_EVIDENCE, separate runs roots, checkpoint sha256 values and
    timing lines read; no metric).
  - Amendment 07 B1–B15 are unchanged except where B16 says so. S3 Beauty / Sports / Toys and Wave 2 are unaffected.

**B16.1 Device class for ML-1M.**
- Every ML-1M run (Track A, Track B, decide-warm, seed3, trackb-select, the winners' seed-2027 runs, the LOCK) uses the
  device class **NVIDIA_A40-FP32-DET**: `--device cuda`, strict deterministic algorithms, CUBLAS_WORKSPACE_CONFIG set before
  CUDA init, cuDNN deterministic, TF32 off, float32 matmul precision "highest".
- The code tree is **code_bench5b** = code_bench5 (BM-03) + the BM-05 delta (`final --device`, accel fix; REV-BM-05
  PASS_WITH_MINORS; delta tar 17fc7c90…, runs_ds3/deploy_bench5b/DELTA.sha256), deployed beside code_bench4 and code_bench5
  with both untouched. code_bench5 itself:
  - deploy tar f39b786a…;
  - the 11 changed / new files are pinned in runs_ds3/deploy_bench5/NEW_FILES.sha256;
  - code_bench4's v4bench and tests_bench were verified equal to the archived originals.
- The driver is run_ml1m_gpu.sh (BM-04). It writes into a fresh runs root, **runs_bench_gpu**.
- The device class is written into every record. A mix of device classes inside ML-1M is refused by the runner (resume,
  θ_pre, S record, FT base, LOCK, trackb-select, decide-warm, seed3; BM-03 tests).

**B16.2 Evidence (MEASURED 2026-10-03, code_bench5 on the A40, GPU 1; smoke-scale, Fable §11 E4).**
These are smoke-scale checks. The production guard is the runner's refusals (B16.1), not this evidence.
- **Determinism:**
  - test_gpu_two_runs_identical PASSED;
  - two independent ML-1M smoke runs gave identical checkpoint sha256 values: C_FULL L200 B24 ac7aeadb… = ac7aeadb…, and FA
    L200 572958017… = 572958017…;
  - the CPU suite against the real code_bench4 passed, so the CPU path is byte-identical (146 / 146 files, BM-03).
- **Timing (smoke):**
  - C_FULL L200 B24: 0.0223 s mean step × 92,088 updates ≈ 34 min per full run, against ≈ 5.4 days on CPU-FP32 at the measured
    ≈ 5.1 s step;
  - FA L200: 8.57 s mean round × 630 rounds ≈ 1.5 h, against 14–17 s per round on CPU at L50 (ml_1m FA_s2026 /
    FA__B6_L50_D0.5_s2026).
- **Projection (INFERENCE):** the full ML-1M graph on the GPU takes about 1–1.5 days, so ML-1M final ≈ 2026-10-05/06. This is
  ≤ 10-08, which meets the §8 gate of at least 3 days gained.

**B16.3 Hard switch (no two complete sets; relaunch guard, Fable §11 E2).** When B16 is registered, Ops does these steps in
order:
1. Mark every ML-1M job of the CPU Wave-1 driver state as SUPERSEDED_DEVICE, so that `--resume` never relaunches one.
2. Stop the CPU ML-1M jobs (their processes only).
3. Verify that the CPU driver's plan holds no runnable ml_1m job.
4. Seal every CPU ML-1M run directory under runs_bench/ml_1m with a SUPERSEDED_DEVICE marker and read-only permissions. It
   is never opened, never lockable, never a winner, and never compared.

S3 is unaffected. The GPU set is run from scratch and takes its own decisions (decide-warm, seed3, trackb-select). The CPU
decision records are SUPERSEDED_DEVICE. A fallback to the CPU is allowed only by a DL row that records a mechanical cause
before any ML-1M result is opened. It is never chosen on a value.

**B16.4 TEST look device: one device class for training AND the TEST look (Fable §11 E1).**
- For ML-1M, `final` runs with `--device cuda` (BM-05: the BM-03 plumbing applied to `final`, with CPU byte-identity and GPU
  two-run tests, and a Sonnet review). This covers the FT arms' per-user fine-tune and all TEST scoring.
- `final` refuses a LOCK whose device class differs from its --device.
- The ML-1M LOCK pins code_bench5 including `final`. S3 Beauty / Sports / Toys stay CPU-FP32 on code_bench4.
- A mix of device classes is REFUSED.
- If BM-05 is not reviewed by the time the ML-1M graph completes, the ML-1M LOCK waits.

**B16.5 GPU precedence and other lanes.**
- **Phase 2 has GPU priority (PREG2 S7).** The ML-1M GPU driver uses the GPUs only until Phase-2 RUN-1 or an H step needs
  them. It yields through a yield file ($RUNS/_gpu_state/YIELD) and the presence of an H or Phase-2 GPU process: it starts no
  new job, lets the running ones finish, and exits. It resumes later under the identical-hash resume rule (Fable §11 E3).
- Exactly one ML-1M driver runs at a time.
- **K16 INT8 pass 2** treats the GPU driver as a Wave-1 blocker (§8 condition 6): the pass-2 lane's W1_BUSY_RE pattern
  includes run_ml1m_gpu.sh.
- **Wave 2 stays CPU-FP32.** The FL rounds there are host-bound (0.95 s per round on CPU for Video_Games), so no ≥ 3× gain
  per FL job is expected (INFERENCE).

**B16.6 Disclosure.** The ML-1M LOCK records:
- this addendum's sha;
- the evidence shas (deploy_bench5, determinism digests);
- the sentence "device class switched before any ML-1M value was read, on schedule grounds; the CPU runs are sealed
  SUPERSEDED_DEVICE and were never opened".

The same disclosure appears in the Track-B table and in the cross-dataset tables (Fable §11 E5).
