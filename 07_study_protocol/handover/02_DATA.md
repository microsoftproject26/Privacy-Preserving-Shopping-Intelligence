# 02 — Data: source, frozen dataset, task, splits, populations

*Paths relative to `.local_ai/ppsi_v4/` unless stated. Every count below is MEASURED from a lock, manifest receipt or
decision-log row cited beside it.*

## 1. Source and the frozen dataset

- **Source.** REES46 e-commerce behaviour logs of one multi-category online store, October and November 2019 (events
  `view`, `cart`, `purchase`; columns event_time, event_type, product_id, category_id, category_code, brand, price,
  user_id, user_session). Local raw files `data/raw/2019-Oct.csv` (5,668,612,855 bytes, 42,448,764 events, DL 234) and
  `data/raw/2019-Nov.csv` in the `eda` folder. REPORTED, not verified: REES46 publishes later months for the same store
  (PREG2 §6.1); they were never used.
- **Frozen dataset `rees46_nextitem_v1`.** Version hash `860e5a1c3a81b8e4bbfeff4f245fa37638a43e4f260df179585964d2d4da8dbd`,
  409 hash-covered files, 8.6 GiB. Server path (read-only, shared by the team): `/home/jovyan/shared_data/rees46_nextitem_v1`.
  Canonical local copy: `F:\Queens University\Microsoft Project\eda\rees46_nextitem_v1`. Its `VERSION.json` lists source
  months 2019-10 and 2019-11 only (PREG2 §6.1).
- **Policy.** The dataset is immutable. Never edit it, never make a per-person copy for training, never "fix" it. Any
  preprocessing change requires a new version (for example `rees46_nextitem_v2`) and the owner's approval; the v1 pipeline
  code hash is `02fbb071…` (PREG2 §6.2). Verify before an experiment with
  `python ~/shared_data/verify_frozen_readonly.py --root ~/shared_data/rees46_nextitem_v1` (from the earlier campaigns).
- **V4 data version.** `CLEAN_V2_POP5100_V2`, built on the server by DATA-06 (15 manifests; receipt 98 / 98; DL 50).
  Production data lock: `contracts/DATA_PROTOCOL_LOCK_V2.json`, lock_self `bc699a97…` (DL 54; Fable G1 approval DL 55).
  The v1 lock (`DATA_PROTOCOL_LOCK.json`, `da51d271…`) is historical: its population (262,144 MAIN users) was superseded by
  the owner's 5–100 amendment (DL 31).

## 2. The task (what one "decision" is)

- **Target:** the next *different* viewed product within the session (views only).
- **Input:** the user's last 50 representative events of the whole prior stream, across sessions and partition fences,
  strictly causal, identical for TRAIN / VALIDATION / TEST — history policy **USER_HISTORY_L50** (owner decision DL 36). A
  per-event channel tells the model which events belong to the current session (session_offset, DL 47); a droppable
  SESSION_INTENT group carries the session's dominant main category (DL 48).
- **Features.** The locked schema **S\*** has 25 information channels = 24 input columns + the output catalogue (schema
  self `c53d47fe…`; DL 53–54). It came from a registered minimal-sufficient feature search on a separate cohort (FS band):
  the worse family's seed-mean relative drop vs the full schema had to be ≤ 1.0 % with UB95 ≤ 2.0 %; results GRU −0.90 %,
  SASRec −0.13 % (S* passed, DL 53). REPORTED: 21 of the 24 inputs are computable on the device (deck slide 6).
- **Catalogue A:** every product with global TRAIN support ≥ 5 = **158,486 products**; every prediction ranks all of them
  (full ranking; pessimistic ties; `CLIENT_CONTRACT_V1.md` §2).
- **Denominators:** E2E = all evaluation rows, out-of-vocabulary targets count as misses (the registered denominator);
  RANKABLE = rows whose target is in the catalogue (descriptive).
- **Normalisers:** feature z-statistics fitted on full TRAIN for all arms (Fable ruling R2 + R4, DL 43); for FL arms this
  is disclosed as one federated-statistics round / the public bootstrap block.

## 3. Chronological splits (global timeline, half-open UTC; `DATA_PROTOCOL_LOCK_V2.json` splits_and_labels)

| Split | Window | Use |
|---|---|---|
| TRAIN | [2019-10-01, 2019-11-16) | training; labels must be observed before TRAIN_END |
| VALIDATION | [2019-11-16, 2019-11-23) | every selection-free evaluation; read once at §9 (DL 400) |
| RETRO_TEST ("TEST") | [2019-11-23, 2019-11-30) | the single TEST sitting (pending) |
| LABEL_GRACE | [2019-11-30, 2019-12-01) | targets only; emits 0 decisions |

- One timeline for every user, so no future leakage across users (multi-expert review §5 item 3).
- **TEST was previously inspected** at period level by the v1 GRU / SASRec campaigns (TEST_EXPOSURE_REGISTER entry 3;
  the lock's `exposure_status`: "VALIDATION and RETRO_TEST are previously inspected; all V4 results are
  retrospective/adaptive"). Disclose.
- **TEST contains Black Friday (29 Nov 2019).** The VALIDATION → TEST change of C_BIG is reported as a shift indicator
  (06B §7(b)); on VALIDATION that section reads NOT_APPLICABLE (REPORT §7).
- **Partition-fence detail to disclose:** at TEST some user-context features replay TRAIN then TEST and skip the VALIDATION
  week (multi-expert review §3.1, item D2; 09-30 handover §4.3).

## 4. Populations and roles

**Selection key v2.** Each user gets a pseudonymous key (sha256 of the salted user id, salt `PPSI_V4_POP_5_100_SEL_20260926`,
DL 40; `CLIENT_CONTRACT_V1.md` §1 C1.2). Users of the 5–100 pool are ranked by this key, and every band is a deterministic,
outcome-free rank range (DL 41–42).

**Strata (TRAIN-only definition; owner amendment DL 31):** TRUE COLD (0 eligible TRAIN examples), SPARSE (1–4), **MAIN pool
(5–100)**, HEAVY (> 100). All strata together: 4,804,933 users / 40,180,608 examples (DL 33). Why 5–100: a simulated device
needs some history; thousands of events are often bots (deck guide slide 8).

| Population / role | Definition | Users | Notes and source |
|---|---|---|---|
| 5–100 pool | 5 ≤ eligible TRAIN examples ≤ 100 | 1,446,986 (28,623,386 examples; 71.24 % of supervision, median 13) | DL 31, 33 |
| **MAIN** | key ranks [0, 131,072) | 131,072; TRAIN 2,587,776 decisions | DL 41–42, 50 |
| MAIN_VALIDATION | MAIN users with VALIDATION rows | **51,211 manifest users; 40,764 users with ≥ 1 E2E row; 519,714 decisions** | DL 50; REPORT §11 |
| **CALIB** | ranks [0, 32,768), nested inside MAIN; the only users on which tuning values are read | 32,768; 646,067 TRAIN decisions | DL 42, 50 |
| CALIB_VALIDATION | CALIB users with VALIDATION rows | 12,715 manifest users; 10,070 E2E users; 127,530 decisions | DL 50; DL 389–391 |
| CALIB-disjoint MAIN | MAIN minus CALIB (diagnostic) | 38,496 manifest; 30,694 E2E | REPORT §11 |
| **PRE band** (pretraining "opted-in" users) | ranks [131,072, 1,348,682) = TIEBREAK minus MAIN | 1,217,610; 24,089,824 decisions (band hash `716beeb9…`) | DL 207; DATA-10 |
| TIEBREAK band = **C_FULL population** | ranks [0, 1,348,682) = MAIN + PRE band | 1,348,682; 26,677,600 decisions | DL 151, 162 |
| **UNSEEN_VAL** (U_NEW_VAL) | ranks [1,348,682, 1,381,450); never trained on | 32,768 support; 12,792 validation users | the device-recipe **selection panel** (06A §1); 06O descriptive table (DL 293) |
| **UNSEEN_HOLDOUT** (U_NEW_TEST) | ranks [1,381,450, 1,414,218); never trained on | 32,768 support; 12,862 validation users (10,224 E2E users in K9b) | participation-gap / new-user panel |
| FS band | ranks ≥ 1,414,218 (top 32,768 keys) | 32,768 | feature search only (DL 41) |
| **TRUE_COLD** panel | users with 0 eligible TRAIN examples and VALIDATION decisions | 381,295 (2,857,665 decisions) | context-only reevaluation (some arms not admitted) |
| **SPARSE** panel | 16,384-user adaptive subset of the 1–4 stratum (full set 155,059) | 16,384 (129,696 decisions) | DL 42 |
| **HEAVY** panel | 16,384-user adaptive subset of the > 100 stratum (full set 36,373) | 16,384 (617,944 decisions) | DL 42 |
| COLD_H, WARM_TRUNCATION | 16,384 each, for the controlled cold-start studies | — | studies declared not run (DL 317 D3) |

**Disjointness.** MAIN, the PRE band, UNSEEN_VAL, UNSEEN_HOLDOUT and FS are disjoint rank ranges of the pool, so UNSEEN_VAL
is disjoint from the whole C_FULL band (06O). TRUE_COLD, SPARSE and HEAVY lie outside the pool (other strata), so they are
disjoint from every training cohort. CALIB is **not** disjoint from MAIN: it is the first quarter of MAIN, and
CALIB_VALIDATION rows lie inside the VALIDATION window that was read. This is why device recipes were selected on
UNSEEN_VAL instead of CALIB (06A §1) and why every CALIB-tuned arm carries a CALIB-disjoint diagnostic (06L_C C14).
Disclosed again for Phase 2 (PREG2 D-b).

**"Full pool" is not 100 % of the pool:** C_FULL uses about 93 % of the 5–100 pool (1,348,682 of 1,446,986), because the
unseen and feature-search bands are held out (DL 236).

## 5. Manifests and registries (where the files are)

| File | Content | Status |
|---|---|---|
| `contracts/MANIFEST_SPEC_V4.json` (+ `_ADDENDUM_01.json`) | manifest format, roles, register spec | binding spec |
| `contracts/manifest_registry/REGISTRY_V2.json` | the 15 DATA-06 manifests (self `f6a08207…`) | byte-identical since G1 (DL 55) |
| `contracts/manifest_registry/REGISTRY_V2_FULL_EXT.json` | C_FULL train / validation manifests (`c9e48641…`) | DL 162, 283 |
| `contracts/manifest_registry/REGISTRY_V2_PRETRAIN_EXT.json` | PRE band manifests | DATA-10, DL 207 |
| `REGISTRY_V2_TEST_EXT.json` (server `contracts/manifest_registry/`; local copy `company/gates/test_build_v3/`) | the 5 RETRO_TEST manifests: MAIN_RETRO_TEST_5_100_131072, UNEW_HOLDOUT_RETRO_TEST_32768, TRUE_COLD_RETRO_TEST, SPARSE_PANEL_16384_RETRO_TEST, HEAVY_PANEL_16384_RETRO_TEST (manifest header `bfaa2118…`) | built after ALL_METHODS_LOCK, counts-only (DL 406–407) |
| `contracts/manifest_registry/REGISTRY_FS.json` | feature-search band | historical |
| `contracts/strata/STRATA_CUTPOINTS_MAIN.json` | strata cut points (file `0406e24c…`, 21.8 MB) | **server copy only** (large-file fetch failed, DL 226) |
| `contracts/DATA_CONTRACT_CLEAN_V2.md`, `FEATURE_SCHEMA_CLEAN_V2.json`, `CLIENT_CONTRACT_V1.md`, `EVALUATION_CONTRACT_V4.md` | feature, client and evaluation contracts | binding |
| `contracts/derived_schemas/` | S* and other derived schemas | S* registered (DL 53–54) |

## 6. The CALIB-counts definition issue and its fix (DL 389–397)

- **What happened (MEASURED, value-free).** On 2026-10-03 the first value-free `--check-only` run of the frozen analysis
  (phase E) refused: the tool expected the CALIB / MAIN / disjoint user counts {51,211, 12,715, 38,496} but counted
  {40,764, 10,070, 30,694} (DL 389).
- **Cause.** Two correct numbers with two definitions. The registered constant holds **manifest users** (users listed in
  the VALIDATION manifests, as `tools/g2_run.py` enforced at G2), while the check counted **E2E users** (users with ≥ 1
  evaluation row). DATA-06 documents both as exact. A tool-definition bug; the check could never pass on production data.
  No data problem, no value read (Fable decision, `company/reviews/FABLE_GATE_ENDGAME_SCRIPTS_20261003.md` §5; DL 390).
- **Fix (ANA-07, DL 391).** `retention_report.load_calib_context` now enforces the registered manifest counts exactly as
  `g2_run.check_counts` does (all CALIB manifest users mapped by `client_key_hex` into the MAIN manifest users), reports the
  E2E counts beside them and pins MAIN 40,764 / CALIB 10,070 E2E users. retention_report `1d50f7f2 → d0247370`; tests 215
  passed / 84 xfailed; `--claims-only` byte-identical; independent review REV-ANA-07 PASS_WITH_MINORS; deployed as tools_r8.
  The analysis freeze was re-issued as v2.
- **Second refusal (DL 394–397).** E v2 refused on the warm S-record check: the analysis required `n_norms` equal to the
  number of calibration rounds, while the registered writer / validator (`launch_fl.check_s_content`) defines `n_norms` as
  the count of client-delta norms behind the median (20,625) over rounds 0–19. Another tool-definition drift (masked by a
  test fixture with n = rounds = 3). Fix ANA-08: retention_report `d0247370 → 9a3b2c2a`; a value-free pre-freeze dry check
  on the read-only block passed (DL 396); REV-ANA-08 PASS_WITH_MINORS; tools_r9; freeze v3 `59c83c86…`; E v3 PASS
  (DL 397–398). Freezes v1 (`03028475…`) and v2 (`d96116d8…`) are SUPERSEDED and were never used for a read.
- **What the thesis says.** The VALIDATION population is 40,764 E2E users (51,211 manifest users); the CALIB-disjoint
  diagnostic uses 38,496 manifest users of which 30,694 have evaluation rows (REPORT §11).

## 7. Data used by the external benchmarks (summary; details in file 06)

The benchmarks use other public datasets with their own leave-one-out protocol (S3 Beauty / Sports / Toys, MovieLens-1M,
Amazon Reviews 2023 Video_Games and Baby_Products). Raw files: server `ppsi_v4/bench_raw/`, processed `bench_proc/`,
`bench_proc_w2/`, each with SHA256SUMS. They never touch REES46 files, locks or selections (amendment 07 R8).
