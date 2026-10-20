# REV-FL-01 — independent READ-ONLY review of FL-05 / FL-06 / FL-07 / INT-05 (+ the A40 device-pin edits)

Reviewer: Claude Opus 5.5, 2026-09-29. READ-ONLY: no repo file edited except this one; no server command. Tags: MEASURED
(file:line, or a command I ran) / INFERENCE / OPINION. All `file:line` refs are to local `code/`.

**Deployed tree = local tree (MEASURED).** `runs_ds3r2_deploy/SYNC_MANIFEST.json` (packed 2026-09-29T00:07:59Z) sha256s
of every file under `code/fl/`, `code/v4int/`, `code/v4eval/`, `v4train/trainer.py` match local, with one exception.
`v4eval/device_pin.py` has a 2-line guard added after packing, inside `write_pin` only (`device_pin.py:100-101`, live-CUDA
check). It does not affect runs. So every finding below applies to `code_ds3r2` as deployed.

---------------------------------------------------------------------------------------------------------------------
## BLOCKERS FOR RUNNING MAIN FL NOW

Scope: FA / FP / PF / FA_Q8 (launch_fl) and LO / FT_C (launch_device) MAIN runs. Headline: **FA, FP and FA_Q8 can
launch. PF must NOT launch until B1 and B2 are fixed.** LO / FT_C have no numerics blocker, but fix M1 first (it is cheap).

**B1 — BLOCKER (PF). A resumed PF run evaluates and saves the WRONG personal store (MEASURED, reproduced).**
- `launch_fl.execute` creates `store = PersonalStore(...)` (`launch_fl.py:379`). `eval_fn` captures that local
  `store` (`launch_fl.py:386-387`), and the end of the run saves it (`launch_fl.py:484-487`).
- On `--resume`, `FLRun.load_state_dict` REPLACES `run.store` with a new object (`fl/runtime.py:554-555`). Training
  keeps using `run.store`, which is correct. But every later PF evaluation scores p_u from the stale, empty store
  (p = 0 for everyone). That also corrupts the best.pt / PRACTICAL_BEST choice, and `PF_STORE.pt` is written EMPTY.
- Reproduction: a scratch test outside the repo, using the repo's own `test_int03_launch_fl` fixtures.
  - Straight run: evaluations saw store sizes `[(2.5,4),(3.0,8),(5.5,10),(6.0,10)]`.
  - Interrupt at round 3, then resume: `[(2.5,4),(3.0,8),(5.5,0),(6.0,0)]`.
  - The resumed `PF_STORE.pt` held 0 clients against 10 in the straight run.
  - `summary.store_digest` (from run.store) still matched, which is why `test_int03::test_resume_is_exact_pf_store_written`
    passes. It never uses `evaluate_pf` and only checks that PF_STORE.pt exists.
- A crash after the endpoint and before RESULT_ROWS also gives an empty PF_STORE.pt.
- Fix (Sonnet, ~5 lines):
  - In `eval_fn`, use `holder["run"].store`. At the end, use `run.store` (or rebind `store = run.store` after construction/resume).
  - Add a test: interrupt + resume with `evaluate_pf`. Assert the per-mark store digests and the PF_STORE.pt digest equal the straight run's.
- Until fixed: never `--resume` a PF run.

**B2 — BLOCKER (PF). The PRACTICAL_BEST PF model is not persisted: best.pt has θ but not the p_u store of that mark
(MEASURED).**
- `FLRun._evaluate` saves `weights_payload()` (`runtime.py:457-458`). That payload holds θ only (`runtime.py:506-510`).
- `PF_STORE.pt` is written once, at the END (`launch_fl.py:484-486`). `latest.pt` has the store, but it is overwritten
  every 64 rounds and deleted at completion (`fl/checkpoint.py:215-216`).
- Consequence: the VALIDATION PRACTICAL_BEST rows were scored with (θ_best, store at the best mark). 06D §2 requires the
  TEST look to re-evaluate "PF (with its PF_STORE)" at PRACTICAL_BEST. It can only pair θ_best with the 6.0 store, which
  is a different model unless the best mark is 6.0.
- That PF run would have to be redone for TEST.
- Fix (Sonnet): when a PF mark improves, also write `PF_STORE_best.pt`. Either atomic_save in `eval_fn` gated on the
  improvement, or include `store.state_dict()` in `weights_payload()` for PF only (≈131,072 × 64 × 3 FP32 ≈ 100 MB).
  Record its digest in RESULT_ROWS `checkpoint_id` and RUN_SUMMARY.

**No blocker found for FA / FP / FA_Q8** (sections 2, 5, 6, 9):
- LR table and 12 marks are exact.
- The planned == realised n_consumed check fires every round (`runtime.py:379-381, 404-406`).
- Resume is bitwise (runtime tests).
- The Q8 codec is re-installed after resume and checked every round (`launch_fl.py:426-445`).

**Operational precondition (not a correctness bug).** On an A40 without `contracts/eval_device/EVAL_DEVICE_PIN_A40.json`,
every non-smoke launch_fl / launch_device run dies at harness construction with `TrainerError`
(`v4train/trainer.py:303-308`). It fails closed and never writes wrong rows. That pin file does not exist locally (only
README.md). Write it (06D §1) before launching.

**Fix before LO / FT_C (not numerics; recoverable):** M1 (device rows mislabelled `REHEARSAL_NON_FROZEN_DEVICE` on the
pinned A40). M2/M3 are guards against mixing device classes across resume legs / shard processes (low risk with ONE A40).
---------------------------------------------------------------------------------------------------------------------

## 1. Poisson sampling (`fl/participation.py:84-91`) — VERDICT: CORRECT
- MEASURED: `u = rng.integers(0, n, size=n)` from `PCG64(derive_seed(seed,"poisson",r))`; member iff `u_i < group_size`.
  So P(member) = m/N exactly: 1,024/131,072 = 2^-7, representable exactly.
- numpy `Generator.integers` uses Lemire's bounded-integer method with rejection, which is unbiased (not modulo).
- Draws are independent across clients within a round (one variate per sorted key). The seed depends only on
  (seed, "poisson", r): no RNG state, no manifest hash, no group size. So nothing couples rounds except the shared seed
  derivation. PCG64 streams from distinct BLAKE2b seeds are treated as independent (standard; INFERENCE).
- The same cohorts are reused by FA_1024 and DP at one seed, and the drop-out stream is separate
  (`derive_seed(seed,"client_dropout",r)`). Both are intended.
- `inclusion_probability` (`participation.py:156-158`) is min(m,N)/N, the same q as `accountant._check_cohort` (m/N).
- Severity: none. MINOR (cosmetic): the docstring at `participation.py:9-12` still cites N = 262,144 / 65,536.

## 2. DP mechanism — VERDICT: CORRECT as implemented (fixed denominator, add/remove, sensitivity S)
- The released update:
  - `finalize_dp` (`fl/dp.py:179-192`) computes θ_r + (Σ_{u∈C_r} clip_S(Δ_u) + z·S·ξ_r)/m, with m = `dp.denominator`
    (never |C_r|).
  - ξ_r ~ N(0, I), drawn on the CPU in FP32 from `derive_seed(seed,"dp_noise",r)` (`dp.py:157-165`).
  - Flat clip over all shared canonical keys (`dp.py:122-129`), factor exactly 1 when norm ≤ S.
- Drop-out is forced to 0 for DP (`runtime.py:115-116`). SASRec recentering in `Server.apply` is post-processing.
- Add/remove sensitivity of the unweighted clipped sum = S → σ = z. MEASURED by reading.
- Data-independence (MEASURED):
  - LR = peak·s(6(r+1)/T) (`runtime.py:382-384`, table `:238-239`).
  - Marks = ⌊12·cursor/T⌋ (`runtime.py:361-362`).
  - Round-count endpoint; continuation refused (`runtime.py:493-494`).
  - No early stop.
  - z > 0 rounds return no norms and no clipped count (`dp.py:201-206`).
  - Norms are logged only when `clip_norm is None`, the S_CAL pass (`runtime.py:420-424`).
- No double count or skip after FL-07 removed the bound:
  - `finalize_dp` requires the aggregated key list to equal the survivors list in logical order (`dp.py:175-178`) and
    `len(added) ≤ n_sampled` (`:180`).
  - `ExposureCounters.update` requires visited == survivors (`participation.py:253-258`).
  - Each shard enforces strictly increasing positions (`dp.py:105-107`).
  - Result: exactly-once per sampled client.
  - Empty rounds give a noise-only release for z > 0 (`runtime.py:439-449`) and a recorded no-op for z = 0.
- NOTE (not fixed, as asked): best.pt is chosen on MAIN_VALIDATION of the same users (post-hoc selection outside the
  accountant). 06D §3 already makes K12/K13 confirmatory at CONTROLLED_BUDGET only. Correct.
- MINOR disclosures:
  - (a) The noise seed is public and deterministic (`dp.py:160`) and torch's PRNG is not cryptographic. The ε claim is
    for the mathematical mechanism in simulation, not a deployable implementation (floating-point Gaussian;
    Mironov 2012).
  - (b) RUN_SUMMARY and the LR-table records hold data-dependent totals (realised exposures, `E_realised_total`,
    steps histogram, per-round realised n_r). They are internal logs, not part of the released model. State that the
    ε covers θ only.

## 3. Accountant (`fl/accountant.py`) — VERDICT: CORRECT; z values independently confirmed
- `_log_a_int` (`:91-100`) is the exact binomial expansion Σ C(α,k)(1-q)^{α-k} q^k exp((k²-k)/2σ²).
- `_log_a_frac` (`:103-136`) is MTZ 2019 §3.3: two series around z0 = σ² log(1/q-1) + 1/2, terms with ½·erfc, the
  generalised binomial via the recurrence with sign tracking (as TF-Privacy `_compute_log_a_frac`).
- Standard conversion ε = RDP + log(1/δ)/(α-1) (`:167-168`). Composition T·RDP (`:159-163`).
- Bisection returns the safe side (`:225-231`). The record rounds z UP to 4 dp (`:241-243`).
- Independent check (MEASURED; mpmath 1.3.0, 30 digits, direct quadrature of
  A_α = E_{x~N(0,σ²)}[((1-q)+q·e^{(2x-1)/2σ²})^α]; scratch script, not the repo code). Max relative RDP difference
  vs the code: 2.6e-11 at the orders tested.

| target | z reg. | ε(z) indep. | best α | ε(z − 0.0001) indep. | code z bisection |
|---|---|---|---|---|---|
| 8 | 0.5642 | 7.998246 | 2.9 | 8.001725 (> 8) | 0.5641495 |
| 1 | 1.3094 | 0.999846 | 16 | 1.000067 (> 1) | 1.3093302 |
| 4 | 0.7244 | 3.999855 | 4.7 | 4.001367 (> 4) | 0.7243904 |

- All three registered z are the correct 4-dp round-ups (m 1,024, N 131,072, T 384, δ 5e-6).
- q = m/N is exactly the sampler's inclusion probability (section 1).
- MINOR (deviation, FIX text only): the module docstring (`accountant.py:25-31`) and `registration_record`'s
  `sampling_disclosure` (`:250-253`, copied into `DP_Z_RECORD_FL04.json`) still say "fixed-size uniform sampling
  without replacement … not proven". Since 06C this is stale: the sampler IS Poisson and the bound is exact.
  Do not edit the locked record. Put a correction in `amendments/` and the thesis.

## 4. `fl/accountant_wor.py` (FL-06) — VERDICT: CORRECT; c = 2 is the right like-for-like number
- Theorem 9 (WBK 2019) for the Gaussian (ε(∞) = ∞ → every min{2,·} = 2) is implemented term for term (`:94-104`):
  1 + γ²C(α,2)·min{4(e^{ε(2)}−1), 2e^{ε(2)}} + Σ_{j≥3} 2γ^j C(α,j) e^{(j−1)ε(j)}.
- Integer α ≥ 2 only (`:114-115`). All sums in log space.
- The min with α/(2σ²) (`:128`) is valid: coupled uniform subsets plus joint convexity of ∫P^αQ^{1−α}, as argued at
  `:26-30`. I agree.
- Replace-one sensitivity of the clipped sum is ‖clip(u)−clip(u′)‖ ≤ 2S → σ = z/2 (c = 2). This is the
  mechanism-faithful WOR value. c = 1 is not a valid bound for fl.dp.
- The adjacency statement (`:169-183`) is accurate (group privacy runs add/remove → replace-one; zero-out ≡ Poisson
  add/remove for the sum).
- Moot for the runs after 06C, but it is the correct documented reason for switching. Severity: none.

## 5. LR schedule for round-count runs — VERDICT: CORRECT, no off-by-one
- Sweep + drop-out (FA/FP/PF/FA_Q8):
  - n_r = passes × survivor valid rows. E_total = Σ n_r over T rounds (`runtime.py:223-226`), which is exactly the
    planned consumed exposures.
  - Row LR = peak·s(6·E_end,r/E_total) at the "end" point (`:241-243`).
  - The final row has e1 = 6·E_total/E_total = 6.0 exactly (integer E, exact FP division), and the table refuses unless
    `final_lr == peak·s(6.0)` (`:250-252`).
  - Marks = ⌊12·E/E_total⌋ (`:363`), so unit 12 fires exactly when the last exposure-bearing round finishes (round T−1
    unless all its survivors drop, P ≈ 0.1^64).
  - `execute` requires `evals[-1][0] == 12` (`launch_fl.py:475-476`).
  - The per-round recomputation must equal the table (`runtime.py:389-391`).
- Poisson: progress (r+1)/T (`:382-384`). Marks ⌊12(r+1)/T⌋ fire after 32, 64, …, 384 rounds done (T = 384 divisible
  by 12). Unit 12 is after the last round. `units_after_row` (`:365-370`) is consistent.
- MINOR: `planned_efe` for Poisson = E_expected/N_dec equals 6.0 only if N_dec = V (sum of per-key valid rows). If they
  differ, only the reported raw number moves, not the LR.

## 6. FA_Q8 — VERDICT: CORRECT for the registered in-process path
- Stochastic rounding q = clamp(floor(Δ/s + U), ±127), U ~ U[0,1) (`fl/quant.py:58-64`). E[q·s] = Δ up to FP32
  granularity of U (2^-24) and the rare clamp at |Δ/s| = 127(1+ulp). Negligible; disclose "unbiased up to float32
  rounding".
- One generator per visit, seeded `derive_seed(seed,"q8",round,key)` (`:45-46, 75`).
- The server decode replaces the upload dict entries BEFORE `acc.add` (`fl/server.py:189-191`, `quant.py:148-151`).
  FedAvg with n_consumed weights is unchanged (`fl/aggregate.py:84-89`).
- θ_r snapshot (`quant.py:133-137`) is cloned at the round's first download, before any `apply`, so its bits equal
  run_round's θ_r.
- `check_round` (`:163-168`) proves every survivor was decoded exactly once, in logical order. The bitwise
  "aggregate == FedAvg of dequantised" guarantee rests on the driver order plus the FL-05 test (INFERENCE for any future
  fl.server refactor).
- Bytes = Σnumel·1 + 4·n_tensors + 8 (`quant.py:152-156`), ratified by 06D §6.
- Resume: `FLRun.resume` rebuilds a plain channel (`runtime.py:544`), then `install_q8` re-installs Q8 with the
  restored ledger (`launch_fl.py:426-429`). `run.channel is q8` is asserted every round (`:442-445`). ARM_SPEC.json
  guards FA ↔ FA_Q8 cross-resume (`:409-417`). No silent bypass found.
- The pool path is refused (`quant.py:160-161, 174-175`).

## 7. LR screen — VERDICT: CORRECT; two MINOR hardening items
- The three runs are identical apart from m: `_strip_lr` removes exactly lr_mult / lr_peak_local / solver.lr
  (`lr_screen.py:65-71`), then compares whole specs (`:132-135`) and one eval manifest (`:136-137`).
- PRACTICAL_BEST row checks (`:100-115`): exactly one row, value equal to the run's own evaluation at `best_half_efe`,
  and the mark matches.
- Tie rule (`:51-58`): best − v ≤ 0.0005, first of (1, 0.5, 2) = 06B §1 exactly.
- `require_lr_selection` (`launch_fl.py:616-621`) runs first in `run()` for every non-smoke MAIN arm, S_CAL included.
  CALIBRATION is exempt, the dry run too.
- PF personal LR stays at the base peak (`launch_fl.py:241`). μ, DP and knobs are unchanged.
- Deviation (ACCEPT, register): PRACTICAL_BEST = the strictly-best running mark (>0.0002, `runtime.py:453`), not the
  plain max of 12. The max is recorded; the difference is ≤ 0.0002 < tie tolerance.
- MINOR (FIX): `load_lr_selection` (`launch_fl.py:596-613`) accepts any self-consistent file.
  - Pin the expected sha256 (DL row) or the path `contracts/fl/LR_SELECTION.json`.
  - Check `obj["lr_peak_base"] == recipe peak_lr`.

## 8. INT-05 (`launch_device.py`) — VERDICT: CORRECT
- `--select` is refused except on UNSEEN_VAL (`:117-125, 132-133`), also in preflight (`:659-660`).
- `load_selection` refuses `selection_role != "UNSEEN_VAL"` (`:525-526`) and a wrong arm (`:142-143`).
- The panel comes from the registry roles (`:834-840`). MEASURED: `REGISTRY_V2.json` roles.UNSEEN_VAL =
  {train UNEW_VAL_SUPPORT_32768, eval UNEW_VAL_VALIDATION_32768}.
- Cutoff = earliest `decision_ts` of the client's eval rows. It is capped by TRAIN_END only for WARM_ROLES
  (MAIN, CALIBRATION) (`:229-232`, `:90`). UNSEEN_VAL is not capped. Correct.
- MINOR:
  - `load_selection` does not re-verify `eval_manifest_content_sha256` against the registry's UNSEEN_VAL eval manifest.
  - The 12,792-user count (06A §1) is not asserted.
- OPEN (planning, not code): LO runs all 3 cells and has no registered MAIN cell when no LO selection exists
  (`:26-31, 385-399`). K8 / K11 need a registered LO cell rule (e.g. `--select` LO on UNSEEN_VAL like FT) BEFORE
  MAIN LO rows are read. Otherwise the LO cell is a post-hoc choice.

## 9. Resume safety — VERDICT: bitwise for FA/FP/FA_Q8/FA_1024/DP/S_CAL; PF launcher-level BROKEN (B1)
Tests I ran (all pass):
- `tests_fl/test_checkpoint_resume.py::test_t20_resume_bitwise[FA|FP|PF]`, plus the sweep-boundary / first-round and
  foreign-config refusal tests.
- `test_fl03_dropout.py::test_resume_bitwise_with_dropout`.
- `test_fl04_dp.py::test_flrun_dp_resume_bitwise`.
- `test_fl07_poisson.py::test_flrun_poisson_dp_resume_bitwise`, `::test_empty_round_resume_bitwise`.
- `test_fl05_quant.py::test_flrun_q8_resume_is_exact`, `tests_int/test_fl05_launch.py::test_fa_q8_resume_exact_and_arm_spec_guard`.
- `tests_int/test_int03_launch_fl.py::test_resume_is_exact_pf_store_written`.

Mechanics:
- Plan, drop-out and Poisson are stateless re-derivations.
- Counters, ledger, evals, best, lr_log, participation/norm logs and RNG are restored (`runtime.py:532-563`).
- Digests are checked: config, plan, manifest, init, LR table.

Gaps:
- **B1** (PF store in the launcher).
- **M2** (MAJOR): no device-class / pin / code-hash continuity check across FL resume legs
  (`launch_fl.py:409-420`, `runtime.py:535-539`). A resumed leg on another GPU class would silently mix training
  numerics. ANA-01 would later refuse mixed `eval_device_class` among marks, but only after the compute is spent.
  Fix: store `eval_device_class`, `device_pin_sha256` and the fl/launcher code sha256 in ARM_SPEC.json or latest.pt,
  and refuse on mismatch.

## 10. GPU-device dependence (A40) — what happens, and the (already implemented) design
- Pre-INT-06 text (06D §1): `trainer.py` refused any class ≠ `A16-FP32-DET` unless `test_overrides`.
  `launch_fl.run` passes `test_overrides=bool(args.smoke)` (`launch_fl.py:783`), so a non-smoke FL run on an A40
  died at harness construction.
- Code as deployed now (MEASURED; the INT-06 / Codex edits are already in code_ds3r2):
  - `trainer.py:303-308` accepts `accepted_classes(frozen, pin)` = {A16-FP32-DET} ∪ {pin class}.
  - `pin_for_device` (`v4eval/device_pin.py:54-70`) returns the pin only if the live class equals the pin class
    (`eval_device_class` maps "NVIDIA A40" → `NVIDIA_A40-FP32-DET`, `frozen_config.py:276-282`). It raises if the live
    GPU name or torch/CUDA version differs from the pin.
  - `load_pin` checks format, authority, required fields, PASS receipts and the self-hash (`:31-47`).
  - `write_pin` is O_EXCL, only after T3 PASS on both families and STRICT two identical reps for both SASRec
    variants (`:81-133`).
- Hence on an A40:
  - (a) without the pin → `TrainerError` before any training (fail closed);
  - (b) with a valid pin → accepted. Every FL / PF / CALIB evaluation goes through `predict_adapter`, which adds
    `device_pin_sha256` to eval_config (`v4eval/predict.py:196-199`). Rows are labelled
    `eval_device_class = NVIDIA_A40-FP32-DET`, `status = MEASURED_PINNED_DEVICE` (`v4eval/rows.py:93-98`), plus the
    pin sha. The launch record also carries the pin sha (`launch_fl.py:848`).
- This is the smallest safe change I would have proposed. Remaining issues:
  - **M1** (MAJOR, FIX before LO/FT_C):
    - `launch_device.assemble` builds eval_config with `frozen.eval_config(PATH_ADAPTER, device_class)` directly
      (`launch_device.py:432-434`) and never adds the pin. So LO / FT_C / FT_FA rows on the pinned A40 get
      `status = REHEARSAL_NON_FROZEN_DEVICE` (`rows.py:97-98`), while carrying `device_pin_sha256` via
      `tr._row_labels()` (`launch_device.py:856`). Numerics are correct; the label is wrong against 06D §1.
      ANA-01 keys on `eval_device_class`, not status (`tools/retention_report.py:600-609`), so it would not refuse.
    - Fix: in `assemble`/`finalize`, call `pin_for_device(dev)` and set `eval_config["device_pin_sha256"]`.
    - If runs already exist: re-run with `--resume` once every shard exists. That re-finalizes without retraining
      (RESULT_ROWS is not O_EXCL). A `--select` run would hit SELECTION.json O_EXCL.
  - **M3** (MAJOR guard): device shards do not record the device class or pin (`launch_device.py:411-412`). `finalize`
    labels the whole artifact with the class of the process that assembles (`:472`). Shards from two GPU classes would
    be merged silently. Record the class and pin per shard and refuse mixing.
  - Deviation (ACCEPT, register): 06D §1 lists "the C_BIG / C_SMALL re-evaluation rows sha256" as a pin field. This is
    impossible by construction: `reeval.py:117` needs the pin first. Record those shas in the DL row instead.
    - The driver is recorded but not checked at runtime (`device_pin.py:66-69`). ACCEPT, disclose.
  - MINOR: the STRICT receipt covers central SASRec training, not the FL client loop. Recommendation (OPINION): add two
    identical FA smokes on the A40 (compare `SMOKE_SUMMARY.server_digest`) before FA. Cheap.

## 11. Other issues and deviations from 06 / 06A / 06B / 06C / 06D
| # | item | where | verdict |
|---|---|---|---|
| D1 | accountant docstring + DP_Z_RECORD "fixed-size WOR" disclosure stale after 06C | accountant.py:25-31, 250-253 | FIX text (amendment note; do not touch the locked record) |
| D2 | PRACTICAL_BEST = strictly-best mark (>0.0002) vs 06B "best of the 12 marks" | runtime.py:453; lr_screen.py:14-16 | ACCEPT (register) |
| D3 | FA_Q8 +8-byte header vs 06B §5 formula | quant.py:152-156 | ACCEPT (06D §6 ratified) |
| D4 | 06C "planned n_r = q·N·E[visit rows]": E_total expected; rows hold realised n_r | runtime.py:200-208, 230-235 | ACCEPT |
| D5 | 06D §1 pin lacks the re-evaluation rows sha (ordering impossible) | device_pin.py:114-120 | ACCEPT (DL row) |
| D6 | 06D §1 "pinned rows carry MEASURED_PINNED_DEVICE" violated for device arms | launch_device.py:432-434 | FIX (M1) |
| D7 | 06A §1 selection tie rule: exact ties only (no tolerance) | launch_device.py:452-459 | ACCEPT (06A registers none) |
| D8 | dp.py docstring: "\|C_r\| = m" (`:5`); "norm log for every DP-configured run" (`:27-28`) | fl/dp.py | FIX text (cosmetic) |
| D9 | participation docstring cites N 262,144 / 65,536 | participation.py:9-12 | FIX text (cosmetic) |
| D10 | 06B §1 "select on CALIB_VALIDATION" + 06D §3 DP disclosure list (LR multiplier from CALIB ⊂ MAIN) | — | ACCEPT (already registered) |
| D11 | DP noise from a public deterministic seed (simulation-only guarantee) | dp.py:160 | ACCEPT + disclose |
| D12 | fleet bytes do not charge dropped clients a download | launch_fl.py:339 | ACCEPT (already disclosed) |
| D13 | LO MAIN cell rule not registered (K8/K11) | launch_device.py:26-31 | FIX (register before read) |

OPINION on "a DP claim FALSE": none found. The ε is exact for the mechanism as run, for θ_1..θ_384 under add/remove
user-level adjacency, with S, m-multiplier and the PRACTICAL_BEST choice excluded as 06D §3 registers.

## Findings table
| finding | severity | file:line | fix | who |
|---|---|---|---|---|
| B1 PF resume: evals + PF_STORE use stale empty store | BLOCKER (PF) | launch_fl.py:379,386-387,484-487; runtime.py:554-555 | use `run.store` after resume; add resume+evaluate_pf digest test | Sonnet |
| B2 PF best-mark p_u store never saved | BLOCKER (PF) | runtime.py:457-458,506-510; launch_fl.py:484-486 | save PF_STORE_best.pt on improvement (or store in best payload) | Sonnet |
| M1 device rows on pinned A40 labelled REHEARSAL_NON_FROZEN_DEVICE | MAJOR | launch_device.py:432-434,472 | add pin sha to assembled eval_config | Sonnet |
| M2 no device/pin/code continuity check on FL resume | MAJOR | launch_fl.py:409-420; runtime.py:535-539 | record + refuse mismatch | Sonnet |
| M3 device shards from different GPU classes merge silently | MAJOR | launch_device.py:411-412,472 | per-shard class/pin, refuse mix | Sonnet |
| D13 LO MAIN cell rule unregistered | MAJOR (plan) | launch_device.py:26-31 | register LO selection on UNSEEN_VAL (or a fixed cell) before read | Ops |
| LR_SELECTION not bound to registered file / peak base | MINOR | launch_fl.py:596-613 | pin sha / path; check lr_peak_base | Sonnet |
| load_selection does not re-verify UNSEEN_VAL manifest sha | MINOR | launch_device.py:518-527 | compare with registry binding | Sonnet |
| stale sampling disclosure in accountant + z record | MINOR | accountant.py:25-31,250-253 | amendment note + thesis | Ops |
| stale docstrings (dp, participation) | MINOR | dp.py:5,27-28; participation.py:9-12 | text | Sonnet |
| Q8 unbiased only up to FP32 rounding; DP seed public | MINOR | quant.py:58-64; dp.py:160 | disclose | Ops |
| FL A40 determinism receipt = central only | MINOR | device_pin.py:95-99 | 2 identical FA smokes on A40 | Ops |

**VERDICT: ACCEPT_WITH_FIXES.** FA / FP / FA_Q8 MAIN may launch now, once the A40 pin exists. PF waits for B1 + B2. LO /
FT_C should get M1 first (or re-finalize later). M2 / M3 are guards.

## Tests run (local venv, python 3.13, torch 2.9.1 CPU; TMP/TEMP/basetemp = `_pytest_tmp/REV-FL-01`)
- `tests_fl`: test_fl04_accountant, test_fl04_dp, test_fl05_quant, test_fl06_wor, test_fl07_poisson, test_fl03_dropout,
  test_checkpoint_resume → **99 passed, 22 xfailed** (scipy present here, so the quad tests ran).
- `tests_int`: test_fl05_launch, test_fl05_lr_screen, test_fl07_launch, test_int04_device, test_int03_launch_fl →
  **43 passed, 18 xfailed**.
- `tests_eval/test_device_pin.py` → **7 passed**.
- Scratch (outside the repo, reuses the test_int03 fixtures): PF interrupt + resume with `evaluate_pf` → **1 failed**,
  which demonstrates B1 (store sizes at marks 5.5 / 6.0 are 0 against 10; resumed PF_STORE.pt has 0 clients).
- Scratch mpmath accountant check (section 3 table).
- Not run: `test_fl05_real_calib.py`, `test_int03_real_clients.py` (need the data store).
