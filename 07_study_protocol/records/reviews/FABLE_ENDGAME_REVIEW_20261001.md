# FABLE END-GAME REVIEW — REES46: from "all runs done" to the single TEST look (2026-10-01, read-only)

- **Reviewer:** Claude Fable 5.1 (chief reviewer), read-only, 40-minute cap. No Codex, no git, no server access, no MAIN or TEST
  value read (none was on the pages read; the only numbers seen are the already-disclosed CALIB screen values and the 06O
  descriptive table of DL 292–293).
- **Read:** RESUME_HERE §2q/§2p/§2o-ter/§2o-bis/§2o; DECISION_LOG rows 270–312; PREG1 amendments 06D §2/§3/§8, 06E §2, 06H §3,
  06K, 06L (A), 06L_C, 06L_D, 06L_E, 06M, 06M_B, 06N, 06O; handoffs ANA-05, FIX-REEVAL-06O, FL-16, FL-16b; MASTER_PLAN_V5 §6, §8–§11;
  company/gates/ANA_CONTRASTS_v3.json (26 confirmatory + S8−S7); the N33 and warm lane scripts (plans/ops_scripts/lanes_r6w/);
  the LOCAL, in-progress ANA-06 copies of tools/ana_block_index.py and tools/retention_report.py (modified 2026-10-01 23:27 — these
  are the worker's uncommitted edits, not a deployed tree; read for their stated rules only).
- **Labels:** MEASURED = read from a file or DL row; PLAN = registered intent; INFERENCE = my reading; PENDING = not yet done.

---------------------------------------------------------------------------------------------------------------------------------
## 0. Verdict in one paragraph

The end-game is **correctly ordered in outline** (§2o item 4: reevals → ANA tooling → freeze → `--check-only` → one §9 read →
ALL_METHODS_LOCK → one TEST look) and the automation of the remaining runs is sound. It is **not complete**: (1) the contrast
registry that the frozen script will refuse without is not written (ANA_CONTRASTS_v4 carrying 06N — the local ANA-06 script itself
lists this under `N06_NEEDS_REGISTRATION`); (2) the warm-line method contrasts, K_INIT and FT_FA_WARM* − FA_WARM* have no registered
id / formula / rule / seeds; (3) the cold DP pair is being re-run at ONE seed (K12 registered at two) and nothing registers the cut,
so the block index will refuse the VALIDATION build unless `--allow-single-seed` is registered first; (4) six registered items are
neither scheduled nor declared "not run" (cold FT_FA → K6/K10; device panels → **K9b, a secondary confirmatory contrast**; Study-A
panel reevals; C_SMALL INT8 → **K16, in 06D §8's MUST tier**; DP_OPT_SENS; TRUNC); (5) the §9 / §11 command lines in MASTER_PLAN
still point at the pre-06N world (`--fl $R/FL`, `ANA_CONTRASTS.json`, no pin, device list without FT_PRE / FT_FA_WARM*). None of
this needs a new run of anything already running; all of it is registration + a few short jobs, and all of it must be done
**before the ANA freeze**, because every item changes the frozen script's inputs or the block's arm set.

---------------------------------------------------------------------------------------------------------------------------------
## 1. The ordered checklist: "all runs done" → "TEST look done" (37 steps in 8 phases)

Legend: **what is computed** is stated per step. Nothing in A–E reads a MAIN metric. Steps marked ★ need a registration (section 3).

### Phase A — run completion and the arm set (server, automatic lanes + decisions)
A1. **Warm line finishes (automatic, r5 / r6).** FA_WARM ×2 + FA_WARM_FROZEN ×2 (r5) → laneFTW_WF_2026/2027 (FT_FA_WARM_FROZEN,
    primary) and laneFTW_W_2026/2027 (FT_FA_WARM): smoke (`--smoke-clients 8`, selection e47965d2, start kind fa_warm) → MAIN.
    FP / PF / FA_Q8 _WARM_FROZEN ×2; FA_1024_WARM_FROZEN ×2, DP8_FROZEN ×2, DP1_WARM_FROZEN s2026 (r6).
    *Computed/checked per run (paths and hashes only):* RUN_SUMMARY `status == DONE_MAIN`; LAUNCH `warm_selection.sha256` ==
    FL_WARM_SELECTION_S8 8b79d305… (frozen arms) / S7 d67a0a85… (FA_WARM, FT_FA_WARM); DP chain LAUNCH `s_record.format ==
    PPSI_V4_DP_S_RECORD_WARM_1`, `plan_digest_checked == plan_digest` (m1, DL 295), `dp_z` ε8 0.5642 / ε1 1.3094; FT lanes' `--base`
    is a FA_WARM(_FROZEN) best.pt (m3) and never a *_WARM* method run; no LANE_STOP; `device_pin_sha256` == db744ca3….
A2. **Cold N3.3 re-runs finish (automatic, r6, DL 306):** N33_FP / N33_PF / N33_FA_Q8 × s2026/s2027 (wait for the six FA-class
    RUN_SUMMARYs), N33_FA_1024_s2026 → N33_DP8_s2026, N33_DP1_s2026 (cold `contracts/dp/S_RECORD.json`, S4 tuning file).
    *Checked:* same as A1 minus the warm fields; LAUNCH `s_record.format == PPSI_V4_DP_S_RECORD_1`.
A3. ★ **Cold DP second seed decision.** Not queued: cold FA_1024_s2027 → DP8_s2027. K12 = R(DP8 / FA_1024) is registered at
    seeds {2026, 2027} in v3 (`default_seeds`); 06N N3.3 allows "single seed acceptable, labelled". Decide now (3a) queue the pair
    behind the N33 chain (≈ 26 h each, INFERENCE from DL 281/303) or (3b) register the cut and the `--allow-single-seed FA_1024 DP8`
    index flag (section 3, D2). Without either the VALIDATION index refuses (`missing registered seed(s)`, ana_block_index L758).
A4. ★ **Registered-but-unscheduled items: run or declare "not run" (06D §8) — decide now, one DL row.**
    (a) **Cold FT_FA** (MASTER_PLAN 6.4/6.5): `SEL_FT_FA` on UNSEEN_VAL (base FL_T/FA_s2026 best.pt, `--base-kind fl`) →
        contracts/device/SELECTION_FT_FA.json O_EXCL → FT_FA MAIN ×2. Feeds K6 (tertiary), K10 (tertiary) and the C16 "no-FT"
        descriptive references (C_SMALL + cold FA on UNSEEN_VAL). Cold FA is DONE (DL 307), so the prerequisite exists; the lane
        was never created. Cost ≈ 1–3 h + 2 × 4–10 h (MASTER_PLAN §6 INFERENCE). Recommendation: **run** (cheap, the "deployed
        on-device system" row of the cold line; also keeps 06L_C C16 honest).
    (b) **Device panels 6.6** (LO / FT_C / FT_FA on UNSEEN_HOLDOUT, SPARSE_PANEL, HEAVY_PANEL, TRUE_COLD, s2026). **K9b = FT_C −
        C_SMALL on UNSEEN_HOLDOUT is SECONDARY confirmatory** (06L_C C7; v3 population UNSEEN_HOLDOUT) and needs FT_C_UNSEEN_HOLDOUT_s2026
        (device) + a C_SMALL best reeval on UNSEEN_HOLDOUT (A5). No DL row shows either ran. Recommendation: **run at least
        FT_C + LO on UNSEEN_HOLDOUT s2026**; declare the SPARSE / HEAVY / TRUE_COLD device panels "not run" if time forbids.
    (c) **C_SMALL INT8 (K16)**: MASTER_PLAN 8.2 (`dump_rows --role MAIN` → `v4dev.bench … --tag onnx_int8`, CPU). K16 is in the
        **MUST** tier of 06D §8 and a confirmatory contrast. No DL row shows it ran. Recommendation: **run now on CPU** (spare
        CPU ≈ 110 cores, DL 304); fallback `--first-users 16384` (06A §6) if > 2 h.
    (d) **DP_OPT_SENS** (06H §3, SHOULD-high, trigger FIRED: FedAdam selected): FA_1024 s2026 + DP8 s2026 under plain FedAvg with
        the same m*, same S_RECORD. 06N does not mention it; §2o item 3 says "check their launch paths". Recommendation: declare
        **"not run — superseded by the 06N re-allocation"** unless a card is idle after A2; it is a cold-line sensitivity.
    (e) **FA_X2** (S5, SHOULD per 06I/06N memo "NICE-if-idle"): declare "not run" unless idle.
    (f) **TRUNC** (DATA-07 T41 both panels + Study A/B; SHOULD-medium; MUST only for an "events to parity" sentence): CPU T41 is
        20–40 min each; Study A/B share a GPU. Recommendation: run T41 + Study A (eval-only) if time; otherwise "not run" and no
        parity sentence in the thesis.
    (g) **Browser ONNX** (owner's PC, HISTORICAL_REFERENCE): optional, not part of any block.
A5. **Panel reevals for the FL / central arms** (MASTER_PLAN 6.7 + 06N): see Phase B3.
A6. **Server pre-freeze inventory (paths only):** list every run dir under runs_ds3/FL_T, DEV, REEVAL_A40, BASE_R4, CFULL,
    runs_pretrain, and the trees each run used (LAUNCH `code_tree` / sha); confirm the STOPPED_REALLOCATED dirs (FL_T/FP_s2026,
    FP_s2027, FA_1024_s2026; stale RUNNING, DL 303) are present and untouched; confirm no other RUNNING dir remains under FL_T
    (a non-DONE dir that is not in the exclusion tuple refuses the whole index, L422).

### Phase B — re-evaluations (reeval 67bb26f3 on code_ds3r6, or the r7 reeval once ANA-06 lands; dry run first, every line)
B1. **Already done (MEASURED):** C_FULL s2026 × {MAIN, UNSEEN_HOLDOUT, SPARSE, HEAVY} × {PB, CB} (8) + PRE θ_pre MAIN PB/CB +
    UNSEEN_HOLDOUT / SPARSE / HEAVY CB (5) — DL 268 "13/13"; C_BIG / C_SMALL × 2 seeds × PB/CB on MAIN (8, r2, DL 169); the 06O
    UNSEEN_VAL previews (5; excluded from every block by design). TRUE_COLD is refused for C_FULL / PRE by design.
B2. **C_FULL s2027 × 4 roles × PB/CB (8), DESCRIPTIVE** (ANA-05 step 2; refused on r5, admitted since r6). *Computed:* the rank
    files + REEVAL_RECORD with `central_source.seed_status DESCRIPTIVE`, binding hashes c9e48641… / c2c8d5e5…; `--expect-sha256` from
    the s2027 sealed/CHECKPOINTS.json. Needs `--allow-shared-gpu` while FL jobs run (approved DL 292) or an idle card.
B3. ★ **Panel reevals, seed 2026, roles UNSEEN_HOLDOUT / SPARSE_PANEL / HEAVY_PANEL (+ TRUE_COLD where the source admits it):**
    registered list (6.7): C_BIG best, C_SMALL best, FA best (cold, FL_T/FA_s2026), DP8 endpoint (now = N33_DP8_s2026). **Add for
    the 06N primary line (not registered anywhere yet):** FA_WARM_FROZEN best (s2026) and DP8_FROZEN endpoint (s2026); optional
    FA_WARM best. These are DESCRIPTIVE panel rows (different populations) but they are what the "realistic Edge" chapter needs
    beside PRE's and C_FULL's panel rows (B1). *Constraint (ANA-05 F1):* a warm REEVAL dir must bind to its FL run dir with the
    original LAUNCH files → run them on the server with FL_T readable, and only on a tree whose reeval knows the warm labels
    (FL-16: "reeval.py does not know the new arms" → ANA-06 scope, so **B3 warm rows wait for r7**).
B4. **No MAIN reeval for FL or device arms:** their own RESULT_ROWS are the A40 rows (pinned class db744ca3…). Nothing to run.
B5. *Checked after B2–B3:* every REEVAL_RECORD `role`, `arm`, `checkpoint sha`, `device_pin_sha256`; no REFUSED; dirs named
    `<ARM>_<ROLE>_s2026_<PB|CB>` as the index expects.

### Phase C — ANA-06 → review → tree r7 (local, then server deploy BESIDE r6)
C1. **ANA-06 content (in progress, DL 308; local tools already carry the rules — MEASURED from the file headers):**
    label map FP/PF/FA_Q8/FA_1024/S_CAL/DP × _WARM / _WARM_FROZEN; `DP_WARM_FROZEN` ε8 → DP8_FROZEN, ε1 → DP1_WARM_FROZEN; `DP_WARM`
    ε → DP8_WARM / DP1_WARM; directory arm must equal the block arm; `N33_<ARM>_s<S>` indexes as the cold arm with entry tag
    `cold_rerun "06N N3.3"`; STOPPED_REALLOCATED = (FP_s2026, FP_s2027, FA_1024_s2026) excluded by name; S_CAL_WARM* excluded;
    records keys S_RECORD_WARM / S_RECORD_WARM_FROZEN; FA_WARM* resolved from the two selection files only (STAR_RULE, tie 0.0005 →
    S8); device runs with a *_WARM* base other than FT_FA_WARM(_FROZEN) refuse (m3); REV-06O-ANA05 minor 1 (REEVAL-beside-run
    binding); reeval learns the warm labels (for B3). **The 06N rows are DESCRIPTIVE until a contrasts file carrying 06N is loaded**
    (N06_LABEL) — this is the hook for D1.
C2. ★ **Write and register `company/gates/ANA_CONTRASTS_v4.json`** (section 3, D1). The report refuses a contrasts file whose
    `registered_seeds` / `hierarchy` differ from the script's constants (retention_report L2889–2891), so v4 and the ANA-06
    constants are written together and frozen together.
C3. **Independent reviews:** REV-ANA-06 (tools + reeval + tests + NCs; old-vs-new: cold-line values and `--claims-only` output —
    NOTE the claims output WILL change once v4 moves the primary to K20b/K21, so "byte-identical claims" is no longer the
    acceptance; the acceptance is "equal to the v3 output when v3 is loaded") and REV-FL-16b (done on CPU, DL 309; registered
    behaviour byte-identical).
C4. **Pack r7 = r6 + launch_fl (FL-16b) + reeval (ANA-06) + tests; deploy BESIDE r6** (m6; pack_r6.sh / deploy_r6.tmpl.sh pattern;
    restore code/tests_fl/receipts/*.json to r6 first); server per-file sha verify; **tiny-run digests r6 == r7 for all 7 cold arms
    AND the warm forms** (FL-16b claims byte-identical run dirs — verify on the server); deploy `ppsi_v4/tools/` r7 (read-only).
    In-flight r6 runs finish on r6; the N33 lanes source `_common_code_ds3r6.sh` and will launch on r6 after r7 exists — acceptable
    (cold digests r5 == r6 == r7) but **record in the DL which tree each MAIN run used** (LAUNCH records it; the lock lists r2–r7).
C5. **Mirror the server-only registration files locally, read-only, sha-checked:** contracts/fl/FL_WARM_SELECTION_S7.json
    (d67a0a85…), _S8.json (8b79d305…), contracts/dp/S_RECORD_WARM_FROZEN.json (2e47ba39…) — DL 310 notes the local tree lacks the
    S7/S8 files; the freeze and the lock hash them.

### Phase D — `company/gates/ANA_SCRIPT_FREEZE.json` (O_EXCL; MASTER_PLAN 1.7; Fable or Ops; one DL row with its sha)
D1. **Contents (ANA-05 step 4 + FIX-REEVAL-06O "for the ANA freeze" + ANA-06 + this review):**
    - sha256 of `tools/retention_report.py`, `tools/ana_block_index.py` (the r7 finals), `code/v4int/reeval.py` (r7), the
      `library_files_sha256` map exactly as the report prints it, v4eval `evaluator_code_sha256` 61b64e61…;
    - `ANA_CONTRASTS_v4.json` self_sha256 (+ v3 a3d1456d… as lineage), the K texts verbatim, the hierarchy (primary pair K20b+K21;
      secondary; tertiary), the confirmatory-rule map (K12, K13 and their warm analogs CONTROLLED_BUDGET/CONTROLLED_BUDGET; K14
      and its analog TOST ±1 %; everything else PRACTICAL_BEST with CONTROLLED_BUDGET beside), the registered seed map **including
      the cuts of A3/A4**, the Holm family statement, the FA_WARM* resolution rule (06N N2) and its outcome S8 (DL 292, CALIB only);
    - the `--claims-only` output hashes regenerated under v4 (CLAIM_REGISTER.json, HEADLINE_TEMPLATE.md);
    - the pin `EVAL_DEVICE_PIN_A40.json` self sha db744ca3…; the code_ds3r7 aggregate tree sha (+ r2–r6 for the runs made on them);
    - the exact §9 command lines (corrected: see E1) and the list of arms declared "not run" (A4);
    - the exclusion rules by name: STOPPED_REALLOCATED dirs, runs_ds3/FL (old-recipe ladder evidence), WARM_PREVIEW, S7S8_T, S7S8,
      SMOKE_R6, LADDER_S1a, S_CAL* dirs, the 06O UNSEEN_VAL preview dirs.
D2. **After the freeze nothing in tools/, reeval.py or the contrasts file changes**; a refusal in Phase E means a NEW tree and a
    NEW freeze (new sha, new DL row), never an edited flag.

### Phase E — server `--check-only` on the real VALIDATION block (reads paths / status / hashes only; the LAST step before the read)
E1. **Index build (corrected command; INFERENCE of the exact flags from the local index header — verify on the dry build):**
    `python -B tools/ana_block_index.py --split VALIDATION --device-class NVIDIA_A40-FP32-DET --pin contracts/eval_device/EVAL_DEVICE_PIN_A40.json
    --reeval $R/REEVAL_A40 --fl $R/FL_T --dev $R/DEV --base $R/BASE_R4 [+ runs_ds/baselines runs_p3/baselines_main only if BASE_R4 does
    not cover all six] --onnx $R/ONNX (if K16 ran) --trunc $R/TRUNC (if run) --records contracts/strata/STRATA_CUTPOINTS_MAIN.json
    contracts/dp/S_RECORD.json contracts/dp/S_RECORD_WARM_FROZEN.json contracts/device/SELECTION_FT_C.json contracts/device/SELECTION_LO.json
    [SELECTION_FT_FA.json] contracts/fl/FL_WARM_SELECTION_S7.json contracts/fl/FL_WARM_SELECTION_S8.json [LR_SELECTION.json if the index still
    requires the key for FA] code/tests_fl/receipts/DP_WOR_SENSITIVITY_FL06.json --allow-single-seed <exactly the registered cuts of A3/A4>
    --out $R/ANA/block_val`. **Never pass `$R/FL`** (the pre-S4 ladder runs; dir names collide with FL_T and a DONE_MAIN old-recipe
    FA_s2026 would either duplicate-refuse or, worse, index). *Computed:* the block index (entries, bindings, exclusions, hashes).
E2. **`retention_report.py --block $R/ANA/block_val --split VALIDATION --contrasts company/gates/ANA_CONTRASTS_v4.json --out
    $R/ANA/out_val_check --expect-script-sha256 <frozen> --check-only` → exit 0.** *Verified here and only here:* comparison keys equal
    (pin), rules paired, ranks reproduce rows, the real warm bindings (LAUNCH `warm_selection.sha256` == records), FT_FA_WARM* bases,
    CALIBRATION counts, FA_WARM* == S8 from the files, N33 entries as the cold arms, STOPPED dirs excluded, every missing seed is a
    registered cut (INCOMPLETE_SEEDS / NOT_RUN labels, never a refusal), every "not run" arm absent and declared.
E3. Fetch the check output locally; DL row ("VALIDATION check-only PASS, no value read"); owner informed that the next command reads.

### Phase F — the single §9 VALIDATION read
F1. Same E2 line without `--check-only`, `--out $R/ANA/out_val`. **This is the FIRST read of any MAIN value.** *Computed:* every K
    (point, 95 % paired user-bootstrap CI, Holm band, per-seed ratios, SEED_SENSITIVE, harmed-client share, LAST anchoring, R*),
    the 06N rows as CONFIRMATORY under v4, the B5 rollout-quality rows, the S8−S7 bytes row, the claim register filled.
F2. DL row; fetch RETENTION_RESULT.json + REPORT.md + curves; thesis-reference batch (Appendix C). Anything added after this moment
    is POST_VALIDATION_EXPLORATORY (06L_C C5 / v3 `post_validation_rule`); no re-run, no selection, no recipe change.

### Phase G — `company/gates/ALL_METHODS_LOCK.json` (O_EXCL)
G1. Contents per MASTER_PLAN §10 **plus**: FA_WARM* = FA_WARM_FROZEN (S8) statement with both selection-file shas; the warm S record
    sha; FL_TUNING_SELECTION 9bca186e…; the N33_<ARM>_s<S> → arm mapping and the STOPPED_REALLOCATED exclusion; the trees r2–r7 and
    which runs used which; the arms declared "not run"; the VALIDATION_RESULT sha (out_val/RETENTION_RESULT.json); the ANA freeze
    sha; the TEST manifests to build (06D §2: MAIN_RETRO_TEST_5_100_131072, UNEW_HOLDOUT_RETRO_TEST_32768, TRUE_COLD_RETRO_TEST,
    SPARSE/HEAVY_PANEL_16384_RETRO_TEST); "device arms re-run their locked procedure" incl. FT_PRE and FT_FA_WARM(_FROZEN) with
    SELECTION_FT_C e47965d2… and the locked bases; the 06L_C D1 contest singles (02B_GRU, 02B_SASREC on A40) to restore in the look.
G2. `tools/test_exposure_register.py append --purpose MANIFEST_BUILD --lock company/gates/ALL_METHODS_LOCK.json …` = **entry 5**
    (the register has 4 entries today — MEASURED) → DL row → upload lock + register to the server. After this: no recipe, arm,
    selection or script change.

### Phase H — the single TEST look (MASTER_PLAN §11, with the 06N arm set; owner informed before it starts)
H1. 11.1 gate + build (CPU): job_test_build.sh / job_test_final.sh → data_test/, REGISTRY_V2_TEST_EXT.json (sha in DL), counts-only
    integrity check; 06L_C D2 fence + counts receipt.
H2. 11.2 reeval every locked checkpoint on `--role MAIN_TEST --registry-ext <TEST_EXT>`: C_BIG ×2, C_SMALL ×2, C_FULL s2026 (+ s2027
    DESCRIPTIVE), PRE (CB), cold FA ×2, N33 FP/PF/FA_Q8 ×2, N33 FA_1024/DP8/DP1 s2026, **FA_WARM ×2, FA_WARM_FROZEN ×2, FP/PF/FA_Q8
    _WARM_FROZEN ×2, FA_1024_WARM_FROZEN ×2, DP8_FROZEN ×2, DP1_WARM_FROZEN s2026** (PB and CB each; warm TEST entries checked against
    `ana05.warm_selection.stages[*].bound_runs` of the VALIDATION_RESULT); TEST panels only for the arms that have VALIDATION panel rows
    (B1/B3). Keep the 06O preview dirs out of the TEST `--reeval` root (they refuse at `role_of`).
H3. 11.3 device arms on MAIN_TEST with the LOCKED selections and bases: LO ×2, FT_C ×2, **FT_PRE ×2, FT_FA_WARM_FROZEN ×2, FT_FA_WARM ×2**
    (and FT_FA ×2 if A4a ran); UNSEEN_HOLDOUT_TEST s2026 for the arms with VALIDATION panel rows (K9b needs FT_C there).
H4. 11.4 baselines on MAIN_TEST (CPU): POP, LAST, SKNN, KNN_EMB SMALL/BIG, RULE (rank files; LAST anchoring on TEST).
H5. 11.5 ONNX: `dump_rows --role MAIN_TEST` → bench INT8 ranks (only if K16 ran on VALIDATION).
H6. 11.6 `ana_block_index.py --split TEST … --records … VALIDATION_RESULT=$R/ANA/out_val/RETENTION_RESULT.json` →
    `retention_report.py --split TEST --contrasts company/gates/ALL_METHODS_LOCK.json --out $R/ANA/out_test --expect-script-sha256 <frozen>`
    (TEST refuses non-locked contrasts, FIX-DELTA-03 M5). *Computed:* the TEST table + the VALIDATION→TEST shift indicator.
H7. 11.7 register entry 6 (RESULT), DL row, fetch everything, thesis-reference batch. No re-run except an interrupted computation
    with identical hashes (E10.3).

---------------------------------------------------------------------------------------------------------------------------------
## 2. Gaps and conflicts found

G-1 **No contrasts file carries 06N.** v3 (a3d1456d…) has primary = K2, K20/K20b/K21 written on `FA_WARM` (= S7), and no K_INIT, no
    warm method contrasts. 06N N1 says "the analysis script is updated before the ANA freeze" but names no file. The local ANA-06
    script explicitly reports the 06N rows as DESCRIPTIVE until a v4 is loaded and refuses a file whose seeds/hierarchy differ from
    its constants. **Blocking for a confirmatory read of the primary.** → D1.
G-2 **FA_WARM* = S8 but every registered formula says FA_WARM.** K20 = R(FA_WARM / C_BIG), K20b = R(FA_WARM / C_FULL), K21 = FA_WARM − PRE
    (06K §2, 06L_C C7, v3). 06N N1 re-defines them on FA_WARM* without re-stating the resolved arm. What happens to the S7 rows
    (FA_WARM ×2, FT_FA_WARM ×2) is unspecified beyond S8−S7. → D1 (b).
G-3 **Warm method contrasts have no ids / formulas / denominators / rules / seeds.** 06N N1: "the warm-line method contrasts of N3".
    ANA-06 is implementing "arm-substituted analogs of K4, K5, K12, K13, K14 and the descriptive PF − FA" (INFERENCE, its own label).
    Literal substitution would give R(FP_WARM_FROZEN / C_BIG), but DL 273 says the warm line is reported against **C_FULL**. The
    denominator must be fixed by registration, not by the worker. → D1 (c).
G-4 **K_INIT and FT_FA_WARM* − FA_WARM* lack rule / seeds / tier text.** 06M_B B4 calls FT_FA_WARM − FA_WARM DESCRIPTIVE; 06N N1 lists
    it as secondary → the later amendment wins, but the resolution must be written. → D1 (d).
G-5 **Cold DP pair at one seed, K12 registered at two.** N33 queues FA_1024/DP8/DP1 s2026 only; the index refuses a missing registered
    seed pair unless `--allow-single-seed` names the arm; the flag is a registered cut, so the cut must be a DL row before E1. → D2.
G-6 **Six registered items neither scheduled nor declared "not run":** cold FT_FA (K6, K10, C16 refs), device panels 6.6 (**K9b
    secondary**), Study-A panel reevals 6.7, **C_SMALL INT8 (K16, MUST tier)**, DP_OPT_SENS (06H trigger fired), FA_X2, TRUNC. 06D §8
    allows "not run", but the declaration belongs in the freeze and the lock, and K16 / K9b are cheap enough to run. → D3.
G-7 **Panels for the new primary arm are not registered.** 6.7 lists C_BIG / C_SMALL / FA / DP8 only; nothing registers FA_WARM_FROZEN
    or DP8_FROZEN panel rows, yet PRE and C_FULL already have panel rows (B1) and the Edge chapter compares them. → D3 (b).
G-8 **MASTER_PLAN §9 / §11 commands are stale:** `--fl $R/FL` (MAIN runs are in FL_T since DL 271; FL holds old-recipe dirs with the
    same names), `--contrasts ANA_CONTRASTS.json` (v1; ANA-04 already asked for v3, now v4), no `--pin`, `LR_SELECTION.json` (the S4
    tuning replaced the LR screen), no S_RECORD_WARM_FROZEN / FL_WARM_SELECTION records, §11.2 arm list without the warm line and PRE,
    §11.3 device list without FT_PRE / FT_FA_WARM*. → D4.
G-9 **N33 and STOPPED_REALLOCATED treatment is in code but not in a registration.** ANA-06 hard-codes STOPPED_REALLOCATED =
    (FP_s2026, FP_s2027, FA_1024_s2026) and N33_ARMS = (FP, PF, FA_Q8, FA_1024, DP8, DP1). 06N N4 also names "the queued FA_Q8, DP8,
    DP1, PF, FA_1024/DP8 s2027" — those never created dirs (DL 306 "no dir created"), so the tuple is complete today (INFERENCE); but
    the rule "an N33_ dir IS the registered cold run; a plain-named stopped dir is void" is only in a script docstring. Also the K
    texts say "FA" for the cold FA — the cold FA dirs are plain `FA_s2026/2027` (not N33) and the N33 cold FP/PF/FA_Q8 use the same S4
    tuning file as the cold FA, so the K3/K4/K5/K14 pairings are consistent (MEASURED from the lane scripts). → D2.
G-10 **The claims-only output will change.** Every freeze note so far says "claims-only byte-identical". Moving the primary to K20b/K21
    changes HEADLINE_TEMPLATE / CLAIM_REGISTER by construction. The REV-ANA-06 acceptance must be rewritten (C3), or the reviewer will
    fail a correct change.
G-11 **Reeval of warm arms needs r7.** FL-16: "reeval.py does not know the new arms". B3's FA_WARM_FROZEN / DP8_FROZEN panel rows and every
    H2 warm TEST reeval depend on the ANA-06 reeval change being in the deployed tree. Order: C4 before B3.
G-12 **Local registration copies missing** (DL 310): FL_WARM_SELECTION_S7/S8.json are server-only; the freeze (D1) and the lock (G1) hash
    them. Fetch read-only and verify shas before D. Same for S_RECORD_WARM_FROZEN.json (sha 2e47ba39… in DL 300).
G-13 **Rules that would block the read if ignored (checklist):** non-DONE dir under FL_T not in the exclusion tuple (L422); duplicate
    (arm, seed, rule, role) if `$R/FL` is passed (L747); missing registered seed without the flag (L758); pin absent for the A40 class
    (L788); warm REEVAL dir without a readable FL run dir (ANA-05 F1); FL_WARM_SELECTION_S7/S8 absent from `--records` when warm entries
    exist (F3); a contrasts file whose seeds / hierarchy differ from the script constants (L2889); `--expect-script-sha256` mismatch;
    reeval's GPU-ownership check while FL jobs run (use `--allow-shared-gpu`, DL 292); C_FULL s2027 reeval on r5 (refused; use r6+).
G-14 **Minor / consistency:** 06N's FL-16 fallback (not deployed by 10-02 12:00Z) did not trigger — record "not triggered" in the DL
    for the thesis. 06K §3 "S8 MAIN seed 2026 (2027 if time)" is superseded by 06L_C C8 (2 seeds unconditionally) — already so.
    TEST_EXPOSURE_REGISTER has 4 entries, so the lock is entry 5 and the result entry 6, as MASTER_PLAN says.

---------------------------------------------------------------------------------------------------------------------------------
## 3. Decisions to register NOW (before ANA-06 freezes anything), with exact text

### D1 — PREG1_AMENDMENT_06P (short) + `company/gates/ANA_CONTRASTS_v4.json` — contrasts for the 06N hierarchy
Proposed text (registration only; no MAIN or TEST value read; before the ANA freeze):

> **P1. Resolution of FA_WARM\*.** FA_WARM\* = FA_WARM_FROZEN (S8), by 06N N2 on the registered CALIB selections (DL 292). In K20, K20b
> and K21 the arm `FA_WARM` is read as `FA_WARM_FROZEN`: K20 = R(FA_WARM_FROZEN / C_BIG), K20b = R(FA_WARM_FROZEN / C_FULL) (2 seeds
> vs C_FULL s2026), K21 = FA_WARM_FROZEN − PRE (paired Δ, 2 seeds, PRACTICAL_BEST). **Primary = the pair {K20b, K21}**, read unadjusted
> and reported together; a claim about the deployment branch requires both. K2 = R(FA / C_BIG) becomes SECONDARY ("random-start
> ablation"). K20c = R(PRE / C_FULL) unchanged (secondary, seed 2026).
> **P2. The S7 variant.** FA_WARM (full-model warm start) keeps 2 MAIN seeds. Its rows are reported as the registered descriptive
> comparison S8 − S7 (06L_C C8, with the bytes ratio) and as DESCRIPTIVE R rows (R(FA_WARM / C_FULL), FA_WARM − PRE) beside the
> primary; no confirmatory contrast is defined on FA_WARM.
> **P3. K_INIT = FA_WARM_FROZEN − FA** (paired Δ, seeds 2026/2027, PRACTICAL_BEST, CONTROLLED_BUDGET beside; "value of the pretrained
> start"), SECONDARY, in the Holm family.
> **P4. Warm-line method contrasts** (all on MAIN, the warm reference = C_FULL per DL 273): K4w = R(FP_WARM_FROZEN / C_FULL), K5w =
> R(PF_WARM_FROZEN / C_FULL) (2 seeds, PRACTICAL_BEST, TERTIARY; PF_WARM_FROZEN − FA_WARM_FROZEN DESCRIPTIVE with the 06L A5 caveat);
> K12w = R(DP8_FROZEN / FA_1024_WARM_FROZEN) (2 seeds, CONTROLLED_BUDGET / CONTROLLED_BUDGET, SECONDARY; PRACTICAL_BEST ratio
> DESCRIPTIVE with the 06D §3 disclosure); K13w = R(DP1_WARM_FROZEN / FA_1024_WARM_FROZEN) (seed 2026, CONTROLLED_BUDGET, TERTIARY);
> K14w = R(FA_Q8_WARM_FROZEN / FA_WARM_FROZEN) (2 seeds, TOST ±1 % relative as 06L_C C15, TERTIARY); R(FA_1024_WARM_FROZEN /
> FA_WARM_FROZEN) DESCRIPTIVE (clipping + Poisson-cohort cost). The cold K4, K5, K12, K13, K14 keep their text and tiers on the N33
> re-runs.
> **P5. FT_FA_WARM_FROZEN − FA_WARM_FROZEN** (paired Δ, 2 seeds, PRACTICAL_BEST, base = the same-seed best.pt, harmed-client share
> beside) is SECONDARY (06N N1 supersedes 06M_B B4's DESCRIPTIVE); FT_FA_WARM − FA_WARM stays DESCRIPTIVE.
> **P6. Multiplicity.** Primary pair unadjusted; Holm over the secondary family (K1, K3, K8, K9b, K9c, K11b, K12, K12w, K17, K20,
> K20c, K_INIT, P5, K2); tertiary unadjusted with the Holm band beside, as in v3. `registered_seeds` as v3 plus the cuts of D2.
> **P7.** `ANA_CONTRASTS_v4.json` carries P1–P6 verbatim, the algebraic dependencies (K20b_s = K20_s × C_BIG_s / C_FULL; K20b − K20c
> = 100·K21 / C_FULL), `post_validation_rule` unchanged; its sha256 is registered by the DL row and copied into ANA_SCRIPT_FREEZE and
> ALL_METHODS_LOCK. The `--claims-only` output is regenerated under v4 and frozen with it.

(INFERENCE on P4 tiers: 06N N1 says "secondary" for the whole warm method set; I keep the DP contrast secondary and the others
tertiary to mirror 06D §4's cold hierarchy — Ops may register all five as secondary instead; either is defensible, but it must be
written before the read.)

### D2 — DL row: cold-line seed cuts, N33 directories, STOPPED dirs (06N N3.3 / N4 made executable)
> "06N N3.3/N4 execution rules, registered before any index build: (1) `runs_ds3/FL_T/N33_<ARM>_s<S>` (ARM ∈ FP, PF, FA_Q8, FA_1024, DP8,
> DP1) IS the registered cold run of that arm and seed (same S4 tuning file 9bca186e…, cold S_RECORD 7ab91eb2…); the plain-named
> `FL_T/FP_s2026`, `FP_s2027`, `FA_1024_s2026` are STOPPED_REALLOCATED: kept on disk, never indexed, never evidence. The cold FA runs are
> `FL_T/FA_s2026/s2027` (not N33). `runs_ds3/FL/` (pre-S4 ladder runs) is never passed to the index. (2) Cold FA_1024 and DP8 are run at
> seed 2026 only [or: s2027 pair queued behind N33_DP1 — choose one]; K12 (cold) is read with `seeds_status INCOMPLETE_SEEDS`, labelled
> single-seed, never substituted; the VALIDATION and TEST index builds pass `--allow-single-seed FA_1024 DP8` and nothing else [plus FT_FA
> if A4a is cut]. (3) The 06N fallback (FL-16 not deployed by 10-02 12:00Z) did not trigger."

### D3 — DL row: run-or-declare for the unscheduled registered items + panels for the primary arm
> "Before the ANA freeze: (a) RUN on CPU now: C_SMALL INT8 (K16; MASTER_PLAN 8.2; `--first-users 16384` fallback). (b) RUN on the next
> idle GPU: SEL_FT_FA (UNSEEN_VAL, base FL_T/FA_s2026 best.pt, `--base-kind fl`) → SELECTION_FT_FA.json O_EXCL → FT_FA MAIN ×2; FT_C and
> LO on UNSEEN_HOLDOUT s2026 (K9b); reeval s2026 on UNSEEN_HOLDOUT / SPARSE / HEAVY (+ TRUE_COLD where admitted): C_BIG best, C_SMALL
> best, FA best, N33_DP8 endpoint, **FA_WARM_FROZEN best, DP8_FROZEN endpoint** (the last two added for the 06N primary line;
> DESCRIPTIVE panel rows; on r7). (c) DECLARED NOT RUN (06D §8; listed in the freeze and the lock): DP_OPT_SENS (superseded by the 06N
> re-allocation), FA_X2, device panels on SPARSE / HEAVY / TRUE_COLD, TRUNC Study B [and T41 / Study A unless run]; no 'events to
> parity' sentence without TRUNC."

### D4 — DL row: MASTER_PLAN §9 / §11 command corrections (text in E1, H2, H3 above)
> "§9 index: `--fl $R/FL_T`, `--pin EVAL_DEVICE_PIN_A40.json`, `--contrasts ANA_CONTRASTS_v4.json`, `--records` += S_RECORD_WARM_FROZEN.json,
> FL_WARM_SELECTION_S7/S8.json, SELECTION_FT_FA.json (if run); `--allow-single-seed` per D2. §11.2 += PRE (CB), C_FULL s2027 (DESCRIPTIVE),
> FA_WARM ×2, FA_WARM_FROZEN ×2, FP/PF/FA_Q8_WARM_FROZEN ×2, FA_1024_WARM_FROZEN ×2, DP8_FROZEN ×2, DP1_WARM_FROZEN; N33 cold arms by their
> N33 dirs. §11.3 += FT_PRE ×2, FT_FA_WARM_FROZEN ×2, FT_FA_WARM ×2 with SELECTION_FT_C e47965d2… and the locked bases."

### D5 — REV-ANA-06 acceptance wording (task packet line, not a registration)
> "Old-vs-new: with `ANA_CONTRASTS_v3.json` loaded, every cold-line value, label and the `--claims-only` output are byte-identical to
> r6; with v4 loaded, the 06N rows become CONFIRMATORY and the claims output changes exactly in the primary / K20–K21 / K_INIT / warm
> method items — list the diff."

---------------------------------------------------------------------------------------------------------------------------------
## 4. What is NOT a problem (checked)
- The order §2o item 4 (reevals → freeze → check-only → read → lock → look) matches 06D §2/§8 and MASTER_PLAN §9–§11.
- FA_WARM* resolution is mechanical and from CALIB files only (DL 292; both the value rule and the tie rule give S8).
- The warm DP chain's S record is the warm-frozen one (06K §3 "S re-calibrated on the S8 trajectory"; DL 300), the cold N33 chain's
  is the cold one (DL 279) — the lane scripts pass them correctly (MEASURED).
- m1 (plan digest) passed on LAUNCH_0 of the smoke (DL 295); m3 (no *_WARM* FT base) is enforced in the FTW lanes; m6 (r6 beside r5,
  in-flight runs finish on their tree) is respected.
- 06O preview rows are excluded from every block by design and refused on TEST; the index still builds with them present.
- TEST tooling (DATA-09, INT-07) is deployed since code_ds3r3 (DL 223); the register entry numbering (5 = lock, 6 = result) is
  consistent with the 4 entries present.
- Nothing in this review requires stopping, re-running or re-selecting anything that is running.

---------------------------------------------------------------------------------------------------------------------------------
## 5. Addendum — joint Ops/Fable settlement of D1–D4 (2026-10-01, read-only reply to the Ops positions)

- **D1 — CONCUR** with the change: K4w, K5w, K12w, K13w, K14w are all SECONDARY (06N N1 text literal: "the warm-line method contrasts
  of N3" are listed under Secondary). My tertiary split was an inference and is withdrawn. **Corrected P6:**
  > Primary pair {K20b, K21}: read unadjusted, reported together, outside the Holm family. **Holm family (secondary):** K1, K2, K3,
  > K8, K9b, K9c, K11b, K12, K17, K20, K20c, K_INIT, P5 (FT_FA_WARM_FROZEN − FA_WARM_FROZEN), K4w, K5w, K12w, K13w, K14w — 19 contrasts.
  > Tertiary (unadjusted, Holm band beside): K4, K5, K6, K7, K9, K10, K11, K13, K14, K15, K16, K18, K19. Convention unchanged from
  > 06L_C C13: unadjusted 95 % paired user-bootstrap CIs with the Holm band beside; K14 / K14w keep their TOST ±1 % reading on the
  > unadjusted 90 % CI with the Holm band reported beside, never as the claim rule. Descriptive, no family: S8−S7, the S7 R rows (P2),
  > PF_WARM_FROZEN − FA_WARM_FROZEN, R(FA_1024_WARM_FROZEN / FA_WARM_FROZEN), the PRACTICAL_BEST ratios of K12/K13/K12w/K13w.
  > `registered_seeds`: v3 map unchanged (K13, K15–K19, K20c, K9b at [2026]) plus K13w [2026]; all other new ids [2026, 2027]; no cuts.
- **D2 — CONCUR** with (2) = RUN the cold s2027 pair. Notes for the lanes: `N33_FA_1024_s2027` and `N33_DP8_s2027` need only the cold
  `contracts/dp/S_RECORD.json` (seed-2026 S applies to both seeds, as in the cold chain), so they can run in PARALLEL on two cards,
  not sequentially (≈ 26 h each at the DL 303 rate, INFERENCE); dry run → MAIN as the other N33 lanes; dirs fit ANA-06's N33_ARMS
  tuple unchanged. DP1 cold stays s2026 (registered). Consequence: `--allow-single-seed` is EMPTY on both index builds; D2(2) text
  becomes "cold FA_1024 and DP8 run at seeds 2026 and 2027 as N33 re-runs; no seed cut registered".
- **D3 — CONCUR** as written (INT8 on CPU now; SEL_FT_FA → SELECTION_FT_FA.json O_EXCL → FT_FA ×2; FT_C / LO on UNSEEN_HOLDOUT s2026;
  panel reevals incl. FA_WARM_FROZEN best / DP8_FROZEN endpoint on r7; DP_OPT_SENS, FA_X2, SPARSE/HEAVY/TRUE_COLD device panels and
  TRUNC Study B declared not run). One addition: SEL_FT_FA must bind the cold FA base `FL_T/FA_s2026` (S4 recipe), never a
  `runs_ds3/FL/` ladder run, and SELECTION_FT_FA.json joins `--records` of both index builds and the lock.
- **D4 — CONCUR** as written, with the two consequential edits: drop `--allow-single-seed` entirely (D2) and add
  `contracts/device/SELECTION_FT_FA.json` to `--records` and FT_FA ×2 to §11.3 (D3).
