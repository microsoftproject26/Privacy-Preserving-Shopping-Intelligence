# EVALUATION_CONTRACT_V4 — shared evaluator, populations, statistics and cold start (specification)

- **Task:** EVAL-01 (Wave A). **Author:** Evaluation Lead (Opus). **Written:** 2026-09-26.
- **Status:** PLAN / SPECIFICATION. It contains no code, no model outcome and no TEST-partition number.
- **Revision R1** (EVAL-01R1, 2026-09-26): the QA-01 receipts in `V4\g0\receipts_qa01\` are incorporated.
  - Every former PENDING_QA-01 item is replaced by receipt facts.
  - §2.2 is re-derived from the code-derived status definitions, as E2.4 required. The counts are unchanged.
  - QA-01's cold-h panel does not match this contract's definition, so the choice is raised to Fable (§11.1, P11).
- **Revision R2** (EVAL-01R2, 2026-09-26): Fable's binding G0(b) ruling (`REV/G0B_PROTOCOL_FABLE_POSITION.md`) is
  applied.
  - §11.4 uses the Clean V2 channel table (C1(a)).
  - The 140 AB VS→VC rows move to CEN_AMBIGUOUS_NEXT_BARRIER, and the censor LOCK_VALUES are verified (C5).
  - §12 states the binding stopping rule (C3).
  - M04 is candidate A, plus a WARM_TRUNCATION panel (C4).
  - C_ref, M04S, the recipe grid and the h conventions are fixed (C6).
  - P3–P6, P11 and P12 are RESOLVED.
- **Binding:** once the Operations Director accepts it and the Fable gate freezes DATA_PROTOCOL_LOCK and
  POPULATION_PLAN (PKG 03 §9 L51; OWNER_SCOPE_V4 "Gate order", item 3).
- **Companions:**
  - `CLIENT_CONTRACT_V1.md`: identity, cohort, cold panels, data ownership and I_META_RANDOM.
  - `MANIFEST_SPEC_V4.json`: the exact manifests, fields, ordering and hashes.

## 0. Conventions

**Path roots**

| Root | Path |
|---|---|
| PKG | `F:\Queens University\Microsoft Project\eda\PPSI_SINGLE_BACKBONE_V4_2026_09_26\PPSI_SINGLE_BACKBONE_V4_2026_09_26` |
| V4 | `F:\Queens University\Microsoft Project\eda\.local_ai\ppsi_v4` |
| G0 | `V4\g0` |
| CEN | `V4\g0\data_census` |
| REV | `V4\company\reviews` |
| REUSE | `V4\code\snap_v1\reuse` |
| GRU1 | `F:\Queens University\Microsoft Project\eda\.local_ai\gru_v1` |
| DS1 | `F:\Queens University\Microsoft Project\eda\rees46_nextitem_v1` |
| QA1 | `V4\g0\receipts_qa01` |

**Citation forms**
- `PKG 09 §4 L19` = package file `09_EVALUATION_AND_COLD_START.md`, section 4, line 19.
- `FABLE D8 L49` = `REV/G0_PLAN_FABLE_POSITION.md` decision 8, line 49. `FABLE LOCK` = its LOCK_VALUES (L55-L65).
- `FABLE B4 L71` = its blocker 4.
- `OPUS-R L22-33` = `REV/G0_PLAN_OPUS_RESPONSE.md`.
- `SPLIT_COUNTS L73` = `CEN/SPLIT_COUNTS.json` line 73. The other CEN JSON files are cited the same way.
- `BARRIER_COUNT` = `G0/BARRIER_COUNT.json`. `LOCALITY_MAP` = `G0/feature_map/FEATURE_LOCALITY_MAP.md`.
- `eval_full.py:L89` and `evaluator.py:L42` = files under REUSE.
- `BOUNDARY_DEFS L49` = `QA1/BOUNDARY_DEFINITIONS.md` line 49.
- `COHORT_RECEIPTS`, `COLD_STRATA`, `COLD_H_PANEL` and `CODE_FLAGS` = the QA1 files `COHORT_RECEIPTS.json`,
  `COLD_STRATA.json`, `COLD_H_PANEL.json` and `CODE_AND_FLAG_CHECKS.json`.
- `pipeline4.py`, `pipeline5.py` and `pipeline6.py` = `next_item_eda/src/rees46_eda/` sources. Their line numbers are
  quoted from BOUNDARY_DEFS.
- `G0B C3 L18` = `REV/G0B_PROTOCOL_FABLE_POSITION.md`, decision C3, line 18. `G0B LOCK L32` = its LOCK_VALUES delta
  (L30-L35). This ruling is binding.
- `SCHEMA_V2 L113` = `V4/contracts/FEATURE_SCHEMA_CLEAN_V2.json` line 113, at schema version 2.1.0 (L3; self-hash L4).
- `OI03` = `V4/g0/receipts_qa02/OI03_F1_PRICE_REZERO.json`.

**Tags** (company charter): MEASURED (a number read verbatim from the cited line), PLAN (a rule adopted here),
INFERENCE, LIMITATION, PENDING_FABLE and PENDING_RECEIPT (waiting on a named Wave A receipt). Since R1, no
PENDING_QA-01 item remains. Since R2, no PENDING_FABLE item remains (§14).

**Arithmetic.** Numbers are quoted, never recomputed. A sum of quoted cells is marked "consistency check"; it is not
a new measurement.

**TEST.** No RETRO_TEST number appears in this file. The G0 census files and the QA-01 receipts contain TEST
structural counts; they are not quoted (see §10.4).

## 1. Splits and label availability

### 1.1 Bounds (all 2019, half-open, UTC)

| Split | Interval | Governs | Source |
|---|---|---|---|
| TRAIN | 2019-10-01T00:00Z ≤ t < 2019-11-16T00:00Z | decision_ts | PKG 03 §2 L10; FABLE LOCK SPLITS L58 |
| VALIDATION | 2019-11-16T00:00Z ≤ t < 2019-11-23T00:00Z | decision_ts | PKG 03 §2 L11 |
| RETRO_TEST | 2019-11-23T00:00Z ≤ t < 2019-11-30T00:00Z | decision_ts | PKG 03 §2 L12 |
| LABEL_GRACE | 2019-11-30T00:00Z ≤ t < 2019-12-01T00:00Z | target event_ts only; **zero** decisions | PKG 03 §2 L13 |

- **E1.1** (PLAN, T03): endpoints are half-open. A decision at exactly 2019-11-16T00:00:00Z is in VALIDATION. A target
  observed at exactly TRAIN_END is *not* "strictly before TRAIN_END".
- The legacy `splits.json` bounds are identical. VALIDATION_SELECT (VS) = [11-16, 11-20) and VALIDATION_CONFIRM (VC)
  = [11-20, 11-23) (CEN/CENSUS_README.md L47-L55).

### 1.2 Merged VALIDATION (minimal-diff)
- **E1.2 Membership** (FABLE D3 L23-L25; PKG 03 §2 L14-L15; T03):
  - VALIDATION = old VS rows ∪ old VC rows. The union is a plain sum because the splits are time-disjoint
    (SPLIT_COUNTS L5).
  - MEASURED: 14,357,645 decisions (SPLIT_COUNTS L73); 11,288,056 eval rows (L76); 1,578,912 distinct users (L79).
  - The two parts are 10,930,090 (L25) and 3,427,555 (L40).
- **E1.3 Minimal-diff:**
  - The old decision, target and context bindings are reused.
  - Legacy model-segment and barrier semantics are kept, including the 2019-11-20 VS/VC barrier (PKG 03 §2 L15;
    PKG 12 T03 L8, "old internal barriers preserved").
  - Disclosure: this is not a newly generated session timeline (PKG 03 §2 L15).
  - A raw rebuild is not done. The raw CSVs are not on the server (FABLE D3 L24), and raw resessionization is listed
    as skipped (FABLE L77).

### 1.3 Subperiods
- **E1.4** (FABLE D3 L25; PKG 03 §2 L14): the subperiods are defined on decision_ts:

  | Subperiod | Interval | Old split | Decisions | Eval rows | Users |
  |---|---|---|---|---|---|
  | VAL_A | [11-16, 11-20) | VS (same bounds) | 10,930,090 (SPLIT_COUNTS L25) | 8,826,488 (L28) | 1,139,657 (L31) |
  | VAL_B | [11-20, 11-23) | VC (same bounds) | 3,427,555 (L40) | 2,461,568 (L43) | 700,093 (L46) |

- **E1.5** (PLAN): subperiods are descriptive only.
  - They are never used for selection, stopping, LR/μ/λ calibration or any recipe choice.
  - The full registered metric grid (§5.3) is reported per subperiod, with n_decisions and n_users.
- INFERENCE: each subperiod macro averages over the users present in that subperiod, and a user present in both
  counts in both. The census notes the union removes about 260,838 double-counted users (CENSUS_README L166-L168).
  So subperiod macros do not decompose the full-window macro.

### 1.4 Label availability
- **E1.6 TRAIN** (PKG 03 §3 L18; T04):
  - A supervised transition is admissible only if its target was observed strictly before TRAIN_END.
  - An input before the cutoff whose label comes after the cutoff contributes no loss.
  - MEASURED: 40,180,608 TRAIN loss rows (LABEL_TIME_AUDIT L10; SPLIT_COUNTS L12). Of these, 0 have
    target_ts ≥ TRAIN_END (LABEL_TIME_AUDIT L11-L14) and 0 have a null target_ts (L15-L18).
  - The cohort TRAIN manifest re-asserts this rule on every row (MANIFEST_SPEC M01).
- **E1.7 Target before filter** (PKG 03 §3 L18; DS1/DATA_CONTRACT.md L132-L134; T06):
  - Targets are created before catalogue filtering.
  - An OOV target keeps its raw identity, and in TRAIN its loss_mask is false.
  - MEASURED: 56,560 TRAIN OOV targets (SPLIT_COUNTS L14). Consistency check: this equals the s=5 excluded count
    (CATALOGUE_SUPPORT L73).
  - Removing an unrankable target never reconnects A→rare B→C into A→C.
- **E1.8 VALIDATION** (PKG 03 §3 L19; T04):
  - No outcome observed at or after VALIDATION_END is used for any selection.
  - Unresolved boundary targets are censored and reported (§2).
  - MEASURED: 0 of 11,288,056 VALIDATION eval rows have target_ts ≥ 2019-11-23T00:00Z (LABEL_TIME_AUDIT L19-L23).
  - PLAN: neither RETRO_TEST-period events nor LABEL_GRACE ever mature a VALIDATION target.
- **E1.9 LABEL_GRACE** (PKG 03 §2 L13, §3 L19; T05):
  - It matures RETRO_TEST targets only, within the existing session and ambiguity constraints.
  - It emits zero decisions.
  - MEASURED: 0 decisions have decision_ts ≥ 2019-11-30 (LABEL_TIME_AUDIT L34), and the ledger has no LABEL_GRACE
    split value (SPLIT_COUNTS L71; CENSUS_README L141-L145).
  - Code (QA-01; BOUNDARY_DEFS L78-L85):
    - OBSERVED_MATURED_IN_GRACE is assigned only to a TEST-split tail decision whose next run-run falls in
      LABEL_GRACE.
    - LABEL_GRACE representatives never originate decisions (`pipeline5.py:88-92`).
    - `grace_decisions_emitted` is asserted to be 0 (`pipeline5.py:211-212`).
- **E1.10** (PKG 03 §3 L19): OOV is an observed-but-unrankable target. It is never a censoring class.

### 1.5 Chronological evaluation
- **E1.11** (PKG 03 §4 L23):
  - Targets are successive: each decision's context stops at that decision.
  - A later VALIDATION or TEST event may be context for a later decision.
  - There are no model or personal-parameter updates from VALIDATION or TEST labels in the offline primary
    evaluation.
  - Multi-future-item retrieval is never relabelled as next-item prediction.
- **E1.12** (PKG 03 §3 L20):
  - L = 50 is the maximum same-derived-session prefix. A one-event context with a known target is valid.
  - Sparse users are never stripped.

## 2. Target status and censoring classes

### 2.1 Status vocabulary and the eval mask
- **Observed values** (CENSUS_README L121-L126): OBSERVED_IN_PARTITION, NO_OBSERVED_NEXT, AMBIGUOUS_BOUNDARY (AB),
  BOUNDARY_UNRESOLVED (BU), and OBSERVED_MATURED_IN_GRACE (RETRO_TEST only).
  - None is prose-defined in DATA_CONTRACT or FEATURE_DICTIONARY (CENSUS_README L121-L129; FABLE B3 L70).
  - The code-derived definitions are in §2.3 (QA-01).
- **Eval mask — holds by construction** (QA-01; BOUNDARY_DEFS L115-L145, quoting `pipeline6.py:386-404`):
  - `has_observed_target = (target_product_id IS NOT NULL)` (BOUNDARY_DEFS L118).
  - `eval_mask = split ∈ {VALIDATION_SELECT, VALIDATION_CONFIRM, TEST} AND target_product_id IS NOT NULL`
    (L124-L125).
  - target_product_id is non-null in exactly the OBSERVED_IN_PARTITION and OBSERVED_MATURED_IN_GRACE branches
    (L129-L132).
  - So eval_mask == has_observed_target in the future partitions by construction, not only by aggregate agreement.
    This resolves the one-bucket LIMITATION of CENSUS_README L134-L140, and agrees with the census measurement
    (CENSUS_README L62-L66).
  - VALIDATION: OBSERVED_IN_PARTITION 11,288,056 (SPLIT_COUNTS L81) = eval_mask_true 11,288,056 (L76).
- **VALIDATION status totals:** OBSERVED_IN_PARTITION 11,288,056, NO_OBSERVED_NEXT 3,064,778, AB 4,489 and BU 322
  (SPLIT_COUNTS L81-L84). Consistency check: they sum to the 14,357,645 decisions (L73).

### 2.2 Censoring classes (full VALIDATION population; rows with eval_mask = false) — re-derived in R1

`next_split = TEST` below is the ledger's legacy label; under V4 that period is RETRO_TEST.

**Why R1 changed the classes.** The status CASE tests the type of the next block *before* it tests the session and
the split (§2.3; BOUNDARY_DEFS L49-L60, `pipeline5.py:122-132`).
- **BU needs a same-session continuation.** A BU row's next raw block is a clean SINGLE_ITEM in the **same behavioural
  session** but a different split (BOUNDARY_DEFS L94-L100). It is therefore a proven continuation across a split
  fence.
- **AB checks neither session nor split.** It needs only that the user's next raw block is a barrier (BOUNDARY_DEFS
  L86-L93). An AB row's next_split tells where that barrier block lies, not that the session continued.
  - INFERENCE: when the next block is days later (VS→TEST, VS→GRACE, VC→GRACE), it must be in a new session, because
    sessions break at gaps > 3600 s (PKG 03 §5 L27).
- **Consequence.** Only BU rows can be right-censored or barrier-censored. AB rows are ambiguity-censored wherever the
  barrier lies.
  - EVAL-01 had put the AB→TEST and AB→GRACE rows in CEN_RIGHT_EDGE. R1 moves them to CEN_AMBIGUOUS_NEXT_BARRIER,
    which replaces CEN_AMBIGUOUS_WITHIN, as E2.4 required. No count changes.
  - R2 applies G0B C5 (L22). The 140 AB VS→VC rows also move to CEN_AMBIGUOUS_NEXT_BARRIER, so every AB row is in that
    class and CEN_LEGACY_BARRIER holds BU rows only.

| Class | Membership rule | VAL_A (old VS) | VAL_B (old VC) | Authority |
|---|---|---|---|---|
| **CEN_NO_NEXT** | target_status = NO_OBSERVED_NEXT | 2,101,485 (SPLIT_COUNTS L34) | 963,293 (L49) | Union 3,064,778 (L82); BOUNDARY_DEFS L101-L113 |
| **CEN_LEGACY_BARRIER** | status = BU ∧ old split VS ∧ next_split = VC | BU 170 (BARRIER_COUNT L21, L23) | none by definition | G0B C5 L22 |
| **CEN_RIGHT_EDGE** (right-censored at VALIDATION_END) | status = BU ∧ next_split ∈ {TEST, LABEL_GRACE} | none observed | BU→TEST 152 (L16) | PKG 03 §3 L19; BOUNDARY_DEFS L94-L100 |
| **CEN_AMBIGUOUS_NEXT_BARRIER** | status = AB (any next_split) | AB→VS 1,773 (L20); AB→VC 140 (L19); AB→TEST 32 (L18); AB→GRACE 2 (L17) | AB→VC 2,364 (L15); AB→TEST 174 (L14); AB→GRACE 4 (L13) | G0B C5 L22; PKG 03 §5 L27; BOUNDARY_DEFS L86-L93 |

- **Censor LOCK_VALUES** (G0B LOCK L32), each checked against the receipts before writing:

  | Class | LOCK_VALUE | Receipt |
  |---|---|---|
  | CEN_LEGACY_BARRIER | 170 | BARRIER_COUNT L21 (also L23) |
  | CEN_AMBIGUOUS_NEXT_BARRIER | 4,489 | SPLIT_COUNTS L83, the union AB total; it equals the sum of the seven AB cells above |
  | CEN_RIGHT_EDGE | 152 | BARRIER_COUNT L16 |
  | CEN_NO_NEXT | 3,064,778 | SPLIT_COUNTS L82 |

- **Consistency checks** (not new measurements):
  - VS AB cells 1,773 + 140 + 32 + 2 = 1,947 (SPLIT_COUNTS L35).
  - VC AB cells 2,364 + 174 + 4 = 2,542 (L50).
  - BU cells 170 + 152 = 322 (L84).
  - The four lock values plus 11,288,056 OBSERVED_IN_PARTITION rows (censor_class = NONE) = 14,357,645 decisions
    (L73). The classes therefore exhaust VALIDATION.
- **E2.1** (PLAN; precedence unchanged per G0B C5 L22): the classes are mutually exclusive.
  - Their membership rules partition the non-evaluated statuses: NO_OBSERVED_NEXT → CEN_NO_NEXT; AB →
    CEN_AMBIGUOUS_NEXT_BARRIER; BU → CEN_LEGACY_BARRIER or CEN_RIGHT_EDGE, by next_split.
  - The builder assigns exactly one class to every eval_mask = false row, with precedence CEN_LEGACY_BARRIER >
    CEN_RIGHT_EDGE > CEN_AMBIGUOUS_NEXT_BARRIER > CEN_NO_NEXT.
  - A row that matches no rule stops the build.
- **E2.2 Legacy-barrier class** (G0B C5 L22; supersedes the "310" of FABLE D3 L24):
  - CEN_LEGACY_BARRIER = the **170 BU rows** only. They stay censored under minimal-diff and are never re-targeted.
  - They are sessions that continue across the 2019-11-20 fence. Segments break whenever the split changes
    (`pipeline4.py:97-103`; BOUNDARY_DEFS L36-L40), and the minimal-diff merge keeps those segments.
  - **The 140 AB VS→VC rows** are ambiguous whatever the fence, because the CASE assigns AB before the session and
    split tests (`pipeline5.py:122-132`; BOUNDARY_DEFS L49-L60). They are in CEN_AMBIGUOUS_NEXT_BARRIER.
  - Fable's G0 "310" figure becomes 170 (G0B risks L44). No total changes.
- **E2.3 Cohort-level class counts:** PENDING_RECEIPT.
  - The manifest builder emits them (MANIFEST_SPEC `counts_by_cause`); they are not recomputed here.
  - QA-01 gave cohort rankable and OOV counts (§3.2), not censor-class counts.

### 2.3 Code-derived status definitions (QA-01, `QA1/BOUNDARY_DEFINITIONS.md`) — resolves the former PENDING_QA-01

The source of truth is `pipeline5.py` (`ledger_sql()`, Stage 07) and `pipeline6.py` (Stage 08 annotation)
(BOUNDARY_DEFS L9-L15). Answers FABLE B3 L70 and FABLE D3 L25.

- **Blocks** (`pipeline4.py:78-81`; BOUNDARY_DEFS L17-L34). A block is one user's raw rows with the same
  event_time_utc. It is one of:
  - SINGLE_ITEM: exactly one distinct, valid product;
  - AMBIGUOUS_TIME_BLOCK_BARRIER: more than one distinct product at that timestamp;
  - INVALID_PRODUCT_BARRIER: at least one invalid-product row.
- **Segments** (`pipeline4.py:97-103`; BOUNDARY_DEFS L36-L45).
  - A segment is a maximal run of consecutive SINGLE_ITEM blocks within one behavioural session and one split.
  - Each representative event is one decision. `is_tail` marks the last decision of a segment
    (`pipeline5.py:96`).
  - Only tail decisions can have a target beyond their segment.
- **target_status, in CASE order** (`pipeline5.py:122-132`; BOUNDARY_DEFS L49-L60 and L73-L113):
  1. **OBSERVED_IN_PARTITION:** a later representative event exists in the same segment, and it becomes the target
     (`pipeline5.py:85`, `112-121`).
  2. **OBSERVED_MATURED_IN_GRACE:** TEST split only. A tail decision whose next run-run falls in LABEL_GRACE.
  3. **NO_OBSERVED_NEXT:** a non-tail decision with no in-partition target (a defensive branch), or a tail decision
     with no next raw block (end of the user's stream).
  4. **AMBIGUOUS_BOUNDARY:** a tail decision whose next raw block exists but is a barrier
     (`next_block_type <> 'SINGLE_ITEM'`). No target is assigned.
  5. **NO_OBSERVED_NEXT:** a tail decision whose next clean block starts a new behavioural session.
  6. **BOUNDARY_UNRESOLVED:** a tail decision whose next raw block is a clean SINGLE_ITEM, in the same behavioural
     session, but in a different split. By design no target crosses a split fence, so the row is left unresolved.
  7. **NO_OBSERVED_NEXT:** the final ELSE.
- **Masks, by construction** (`pipeline6.py:386-404`; BOUNDARY_DEFS L115-L153).
  - For NO_OBSERVED_NEXT, AMBIGUOUS_BOUNDARY and BOUNDARY_UNRESOLVED, target_product_id is NULL. So in every split,
    TRAIN included:
    - has_observed_target, loss_mask and eval_mask are all FALSE;
    - target_class = −1 and target_oov = FALSE.
  - AB and BU rows carry no target anywhere, never contribute loss and are never evaluated.
  - They differ only in why no target exists: a barrier block versus a split-crossing clean next item
    (BOUNDARY_DEFS L147-L153).
- **Measured pairing** (BOUNDARY_DEFS L169-L177; BARRIER_COUNT):
  - Every AB row pairs with next_block_type AMBIGUOUS_TIME_BLOCK_BARRIER. INVALID_PRODUCT_BARRIER never occurs with
    AB in this dataset (L171-L172).
  - Every BU row pairs with SINGLE_ITEM and a next_split different from its own.
- **E2.4** (PLAN): applied in R1. §2.2 was re-derived from these definitions before DATA_PROTOCOL_LOCK, and the quoted
  counts stayed.
- **Reporting note** (INFERENCE from the CASE order): AB is tested before the session check, so the AB count includes
  tails whose next barrier block lies in a later session. It therefore overstates within-session ambiguity. Reports
  should say "next raw block is a barrier", not "ambiguous next item within the session".

### 2.4 TRAIN-side boundary rows
- TRAIN statuses: NO_OBSERVED_NEXT 12,512,673, AB 10,472, BU 1 (SPLIT_COUNTS L19-L21).
- The single BU row's next event is in VALIDATION_SELECT (BARRIER_COUNT L12). It contributes no loss, which is
  consistent with LABEL_TIME_AUDIT L11-L14.
- **Resolved by QA-01** (BOUNDARY_DEFS L94-L100, L138-L140, L174-L176):
  - The row is a TRAIN tail whose next clean item, in the same behavioural session, falls in VALIDATION_SELECT.
  - Its loss_mask is FALSE by construction.
  - The mechanism is expected, which answers CENSUS_README L130-L133.
- **Observation for DATA** (INFERENCE, non-blocking): the receipt explains the mechanism but not the magnitude.
  - The TRAIN_END fence has 1 BU row, against 170 at the Nov-20 fence (BARRIER_COUNT L21) and 152 at VALIDATION_END
    (L16).
  - The row carries no loss, so evaluation is unaffected (P13).

### 2.5 Reporting rule
- **E2.5** (PKG 03 §3 L19; PKG 09 §1 L5; PKG 04 §2 L14):
  - Every result table carries, per manifest, the decision count per censor class and the censored share
    (censored / all manifest decisions).
  - These are reported separately from OOV.
  - A censored row is never a miss and never enters a denominator.

## 3. Evaluation populations

### 3.1 Denominators
- **E3.1 E2E** (end-to-end):
  - The rows with eval_mask = true, i.e. every observed, uncensored target. OOV rows score 0.
  - Sources: PKG 04 §2 L9; `eval_full.py:L161` (`end_to_end_pop = eval_mask_t`); DS1/DATA_CONTRACT.md L132-L134.
- **E3.2 RANKABLE:**
  - eval_mask ∧ target_class ≥ 0 (`eval_full.py:L162`, where it is called `closed`; V4 calls it RANKABLE).
  - It is common to every method because the catalogue is common: K = 158,486, support ≥ 5, fitted on full TRAIN
    (FABLE D2 L19-L21; FABLE LOCK CATALOGUE L57; PKG 04 §1 L6).
- **E3.3 Wording reconciliation** (INFERENCE):
  - DATA_CONTRACT L130-L131 says rows with target_class = −1 are "excluded from the ranking denominator". That applies
    to RANKABLE and to censored rows.
  - For E2E the OOV rows stay in the denominator (L132-L134).
  - eval_mask == has_observed_target holds by construction (§2.1; BOUNDARY_DEFS L115-L145) and is measured
    (CENSUS_README L62-L66). So E2E is exactly the set of observed targets.

### 3.2 Registered evaluation manifests (full schema in MANIFEST_SPEC_V4.json)

| ID | Population | Role | MEASURED size | Source |
|---|---|---|---|---|
| M02 VAL_WARM | VALIDATION decisions of the N = 262,144 cohort | **Primary**: family selection and every VALIDATION comparison | 808,610 eval rows = 801,762 rankable + 6,848 OOV; 62,368 users with ≥1 eval row | COHORT_PROJECTIONS L6056-L6057; COHORT_RECEIPTS L45-L47; FABLE D8 L49 |
| M06 CALIB_VAL_WARM | Same, for the nested 65,536 prefix | LR/μ/λ calibration only | 200,985 eval rows = 199,276 rankable + 1,709 OOV; 15,487 users | COHORT_PROJECTIONS L3084-L3085; COHORT_RECEIPTS L24-L26; FABLE D7 L44 |
| M03 COLD_PANEL_VAL | VALIDATION users with ≥1 eval row and zero eligible TRAIN examples | Secondary; every method | 381,295 users: COLD_TRAIN_EVENTS_NO_LABEL 82,260 users / 587,149 eval rows, and COLD_NO_TRAIN_EVENTS 299,035 users / 2,270,516 eval rows | CLIENT_CENSUS L3971; COLD_STRATA L9-L17; FABLE D1 L17 |
| M04 COLD_H_PANEL | Candidate A: the 16,384 M03 users with the smallest population keys; all cold | Cold-start h study (§11) | 16,384 users by construction; strata mix and sub-panel eligibility reported at build | G0B C4 L20; G0B LOCK L34 |
| M04W WARM_TRUNCATION | The 16,384 smallest-population-key users of M02's 62,368 E2E users (nested; no new salt) | Warm-history truncation: T41 control and warm-p_u evaluation (PKG 09 §5 L27); never labelled cold start | 16,384 users by construction | G0B C4 L20; G0B LOCK L34; FABLE D8 L49 |
| M04S RECIPE_SELECTION | M03 users at population-key ranks 16,384–32,767 | Selects the FEW_SHOT recipe once, on FIXED_20, before any M04 outcome | 16,384 users by construction | G0B C6 L24; G0B LOCK L34 |
| M07+ RETRO_TEST | Built only after ALL_METHODS_LOCK | Retrospective TEST | not computed | §10 |

- **Full-VALIDATION reference** (all users; not a V4 comparison block):
  - 11,288,056 E2E rows, 96,871 OOV targets and 11,191,185 rankable rows (SPLIT_COUNTS L76-L78).
  - Coverage 0.991418274324649 (CATALOGUE_SUPPORT L78).
- **Consistency checks** (not new measurements):
  - For M02 and M06, rankable + OOV equals the COHORT_PROJECTIONS eval-row totals.
  - For M03, 82,260 + 299,035 = 381,295 users (CLIENT_CENSUS L3971).
- **E3.4 Cohort receipts — resolved by QA-01** (FABLE B4 L71; COHORT_RECEIPTS):

  | Quantity | N = 262,144 | N = 65,536 |
  |---|---|---|
  | VALIDATION coverage at s=5 | 801,762 rankable / 808,610 eval rows (L45, L47) | 199,276 / 200,985 (L24, L26) |
  | catalogue classes (of 158,486) with ≥1 cohort-TRAIN positive | 133,154 (L51) | 90,545 (L30) |
  | classes with ≥5 cohort-TRAIN positives | 72,682 (L52) | 31,425 (L31) |
  | classes that are VALIDATION targets but have zero cohort-TRAIN positives | 5,158 (L54) | 7,619 (L33) |

  - QA-01 recomputed the universe as 2,694,998, matching the census (L9-L11).
  - Both cohort key hashes match the census (L18-L20, L39-L41).
  - The coverage ratios are left as quoted count pairs and not restated as decimals. The builder emits the fractions.
- The primary covers warm-cohort users only. This is a disclosed choice (FABLE risks L84).

### 3.3 Legacy cohort cuts are not V4 strata
- The legacy evaluator splits cohorts on client_idx: "warm" is client_idx ≥ 3 and "unseen" is client_idx == 2
  (`eval_full.py:L174-L175`; `evaluator.py:L213-L214`).
- client_idx is the TRAIN-fitted user vocabulary:
  - its row count equals the number of distinct TRAIN users (CENSUS_README L82-L87);
  - TRAIN-absent users share OOV = 2 (LOCALITY_MAP L178-L182).
- So the legacy "warm" cut includes users who have TRAIN events but zero eligible examples; 1,509,381 TRAIN users are
  in that state (CLIENT_CENSUS L3967). It is not the V4 warm definition, which requires ≥1 eligible example
  (COHORT_PROJECTIONS L3).
- **E3.5** (PLAN): V4 strata come only from manifest fields (`cold_stratum`), never from client_idx.
  - The legacy cut may be printed as a diagnostic labelled LEGACY_CLIENT_IDX_CUT.

## 4. OOV reporting

### 4.1 End-to-end
- **E4.1** (PKG 04 §2 L9, L15; DATA_CONTRACT L135):
  - An OOV target gets rank −1 (`eval_full.py:L288`, `L313-L320`), which gives zero credit for every metric
    (`evaluator.py:L75-L79`).
  - It stays in the denominator.
  - A PAD, MISSING or OOV token is never a correct answer, and never counts as a prediction of any unseen product.

### 4.2 Conditional common-rankable metric
- **E4.2** (PKG 04 §2 L10, L15; T35):
  - The RANKABLE set is identical for every method in a block. This is proven by an equal
    `rankable_ordered_decision_id_sha256` (MANIFEST_SPEC) on every result row.
  - One method's RANKABLE value is never compared with another method's E2E value.
  - Every result row states its `target_denominator` (PKG templates/RESULT_ROW.template.json L17).

### 4.3 Coverage breakdown
Counts are over the manifest's E2E rows unless stated.

| Category | Definition | Full-VALIDATION reference (MEASURED) | Manifest level |
|---|---|---|---|
| coverage | \|RANKABLE\| / \|E2E\| | 0.991418274324649 (CATALOGUE_SUPPORT L78) | M02 801,762 / 808,610; M06 199,276 / 200,985 (COHORT_RECEIPTS L45-L47, L24-L26); M03, M04 builder emits |
| product OOV, total | target_oov = true | 96,871 (SPLIT_COUNTS L77; also CATALOGUE_SUPPORT L79) | M02 6,848; M06 1,709 (COHORT_RECEIPTS L46, L25); others builder emits |
| rankable target with a zero-cohort-positive class | target_class ≥ 0 but the class has no cohort-TRAIN positive: it is ranked, but no cohort client ever trained on it | not applicable (full TRAIN defines K) | class level: 5,158 classes for M02 (COHORT_RECEIPTS L54) and 7,619 for M06 (L33); the builder emits row counts; diagnostic |
| OOV_ABSENT_FROM_TRAIN | TRAIN support = 0 under the declared support definition (CATALOGUE_SUPPORT L3) | 50,124 (excluded at s=1, CATALOGUE_SUPPORT L29) | builder emits |
| OOV_BELOW_SUPPORT | known in TRAIN, 1 ≤ support ≤ 4 | 96,871 − 50,124 (difference of L79 and L29; the builder emits the measured value) | builder emits |
| metadata state, target side | category / brand / main-category token state of the target ∈ {KNOWN, MISSING, OOV} under the frozen end-of-TRAIN snapshot; reported separately for product-OOV and rankable targets | not measured | builder emits; diagnostic |
| metadata state, context side | share of E2E decisions whose current event has category / brand / main-category token MISSING or OOV | not measured | evaluator emits; diagnostic |
| censored / unavailable | the §2.2 classes; denominator = all manifest decisions; never part of the OOV rate | §2.2 | builder emits |

- **E4.3** (PKG 04 §2 L11-L14; PKG 09 §6 L31): metadata state is a feature-availability diagnostic. It never removes a
  row and never changes a denominator.
- **E4.4** (INFERENCE / LIMITATION): the split between "known in raw TRAIN" and "absent" uses representative-view
  support (CATALOGUE_SUPPORT L3).
  - A product seen in raw TRAIN only through non-view events is classed by its support.
  - Refining this is a DATA task.
- **E4.5** (PKG 04 §2 L15): a content-based cold-item scorer is outside the headline. It would need pre-exposure
  metadata and a new matched block.

### 4.4 Identities and the invalid macro factorization (PKG 04 §2 L16; T36)

**Notation**
- E = the E2E rows. R ⊆ E = the RANKABLE rows. c_i = the credit of row i; c_i = 0 for i ∈ E \ R.
- U = the users with ≥1 E row. E_u and R_u are user u's rows. U_R = {u : R_u ≠ ∅}.

**Micro (exact because OOV credit is 0):**
M_e2e = (1/|E|) Σ_{i∈E} c_i = (|R|/|E|) · (1/|R|) Σ_{i∈R} c_i = **coverage × M_rankable**.

**Macro (the correct identity):**
M_e2e^macro = (1/|U|) Σ_{u∈U} (1/|E_u|) Σ_{i∈E_u} c_i = **(1/|U|) Σ_{u∈U} cov_u × m_u**, where:
- cov_u = |R_u| / |E_u|;
- m_u = (1/|R_u|) Σ_{i∈R_u} c_i, with the term defined as 0 when R_u = ∅.

This is **not** in general equal to mean_u(cov_u) × M_rankable^macro. The rankable macro averages m_u over U_R only,
because `evaluator.py:L98-L111` groups only the users present in the population.

**Counterexample fixture T36-A** (MRR@20):
- User a has one rankable row at rank 1 (c = 1).
- User b has one rankable row at rank 2 (c = 0.5) and one OOV row (c = 0).

| Quantity | Value |
|---|---|
| micro M_e2e | 1.5 / 3 = 0.5 |
| coverage × micro M_rankable | (2/3) × (1.5/2 = 0.75) = 0.5 ✓ |
| macro M_e2e | (1 + 0.25) / 2 = 0.625 |
| mean_u(cov_u × m_u) | (1·1 + 0.5·0.5) / 2 = 0.625 ✓ |
| **invalid** mean(cov) × macro M_rankable | 0.75 × 0.75 = 0.5625 ≠ 0.625 |

**Fixture T36-B** (T36-A plus user c, who has one OOV row only; U = {a, b, c}, U_R = {a, b}):

| Quantity | Value |
|---|---|
| macro M_e2e | 1.25 / 3 = 0.41666… |
| mean_u(cov_u × m_u) | (1 + 0.25 + 0) / 3 = 0.41666… ✓ |
| **invalid** mean(cov) × macro M_rankable | 0.5 × 0.75 = 0.375 ≠ 0.41666… |
| micro M_e2e | 1.5 / 4 = 0.375 = (2/4) × 0.75 ✓ |

- **E4.6** (PLAN):
  - Both identities are asserted in float64, to within 1e-9 absolute, on the synthetic fixtures and on every real
    result row.
  - The invalid factorization must be shown to differ on T36-A and T36-B (negative control).
  - A report that presents mean coverage × rankable macro as an E2E macro fails review.

## 5. Metrics

### 5.1 Per-decision credits (single target)
For a decision i with exact rank r_i ∈ {1, …, K} (i ∈ R):
- HR@k_i = Recall@k_i = 1[r_i ≤ k] (`evaluator.py:L75-L76`).
- RR@k_i (the MRR credit) = 1[r_i ≤ k] / r_i (`evaluator.py:L78`).
- NDCG@k_i = 1[r_i ≤ k] / log2(r_i + 1) (`evaluator.py:L79`). With one relevant item the ideal DCG is 1/log2(2) = 1,
  so no normalizer is needed.
- For i ∈ E \ R every credit is 0 (rank −1 fails the `ranks >= 1` test, `evaluator.py:L75`).

**E5.1** (PKG 09 §1 L4):
- With one target, Recall@k ≡ HR@k. It is reported once, as "HR@k (= Recall@k)", and never as two pieces of evidence.
- INFERENCE from the formulas: at k = 1, HR@1 = MRR@1 = NDCG@1 = 1[r_i = 1]. That is one number, not three.

### 5.2 Aggregation
- **Micro** over population P: (1/|P|) Σ_{i∈P} c_i (`evaluator.py:L104`).
- **Macro (client-macro):** (1/|U_P|) Σ_{u∈U_P} (1/|P_u|) Σ_{i∈P_u} c_i (`evaluator.py:L98-L111`), where:
  - u = `eval_user_key`, and U_P = the users with ≥1 row in P;
  - grouping is by eval_user_key, never by client_id (`evaluator.py:L6-L8`, `L140`; `eval_full.py:L364-L367`;
    GRU1/FULL_RUN_PROTOCOL.md L62);
  - the V4 eval_user_key is defined in CLIENT_CONTRACT §1.

### 5.3 Registered grid
- **Primary:** macro MRR@20 on E2E, over M02 (PKG 09 §1 L4; FABLE D8 L49; FABLE LOCK EVAL L64; OWNER_SCOPE_V4 binding
  scope item 1).
- **Prominent:** macro NDCG@10 and macro HR@10, on E2E.
- **Full grid:** {micro, macro} × {MRR, HR (= Recall), NDCG} × k ∈ {1, 5, 10, 20} × {E2E, RANKABLE}.
  - Larger k only if preregistered (PKG 09 §1 L4).
  - The v1 protocol reported only k = 20 (GRU1 FULL_RUN_PROTOCOL L83); V4 extends it.
- Every value is a result row (PKG templates/RESULT_ROW.template.json L13-L22) with n_decisions, n_users, oov_count and
  coverage.

### 5.4 Numerics and empty populations
- **E5.2** (PLAN):
  - Legacy code computes credits in float32 and accumulates the macro in float32 (`evaluator.py:L74`, `L106-L107`).
    The bootstrap per-user means use float64 (`eval_full.py:L395-L399`).
  - V4 computes credits from the int64 ranks in float64, and does every sum and mean in float64.
  - Parity against the legacy `aggregate` on a real-data sample: ≤ 1e-6 absolute.
- **E5.3** (PKG 12 L3; T49 L62; T45 L58):
  - The legacy code returns NaN for an empty population (`evaluator.py:L91-L96`). V4 never writes a result row for an
    empty registered population; it records MISSING_EVIDENCE instead.
  - NaN is never serialized.

## 6. Ranking

### 6.1 Rank rule and tie rule (quoted)
- **Rank:** `eval_full.py:L63` states `rank = 1 + #{s_j > s_t} + #{j < t: s_j == s_t}`. It is implemented at `L89-L103`:
  - target score gather, L89;
  - strictly-greater count, L98;
  - lower-index equal count, L100-L101;
  - return, L103.
- **Tie rule:** "Ascending class id wins ties; the target NEVER benefits from a tie (only ties with a strictly LOWER
  class id count against it)" (`evaluator.py:L44-L46`). GRU1 FULL_RUN_PROTOCOL L61 states the same.
- **E6.1** (PKG 04 §1 L6): the class id is the frozen catalogue's output_class 0..K−1 (DATA_CONTRACT L126-L129).
  - Because the tie rule depends on this order, it is bound by `candidate_order_hash` (§9, MANIFEST_SPEC).
  - Order, availability and ties are shared by all methods.

### 6.2 Streaming
- **Class-chunked accumulation** over the K axis: `for start in range(0, K, step)` (`eval_full.py:L94-L101`), with a
  default class_chunk of 32,768 (L55).
  - "Never `torch.topk`, never candidate filtering; correctness is independent of `class_chunk` by construction (an
    exact partition of the K axis)" (L68-L70).
  - Scores are float32 or int64 and are never coerced (L64-L67, L73-L77).
- **Which rows are ranked:** only rows with eval_mask ∧ target_class ≥ 0 (L311-L320). Every other row keeps −1 (L288).
  A row-count check follows (L341-L342).
- **E6.2 Full candidate set** (PKG 09 §1 L5; T37):
  - All K = 158,486 output classes are candidates, including seen and revisited items. Nothing is filtered
    (DATA_CONTRACT L240-L242; `evaluator.py:L134-L136`).
  - No sampled negatives and no ANN (FABLE D8 L49, which cites R14).

### 6.3 V4 guards (PLAN)
- **G1 Non-finite scores are refused.**
  - INFERENCE from `eval_full.py:L98-L101`: if s_t is NaN, every comparison is false and the rank becomes 1, a silent
    best case. ±inf values create artificial ties.
  - The V4 evaluator asserts that every score in a scored row is finite. Otherwise it fails and writes no success
    marker (T45).
- **G2 Sampled, ANN or top-k evaluation is refused for headline rows.** Any API that accepts a candidate subset,
  sampled negatives or an approximate index raises when the row is tagged headline (T37 L48).
- **G3 Chunk and batch invariance** (T37):
  - Ranks must be identical, with tolerance 0 because they are integer counts. This is checked for class_chunk ∈ {1 on
    a small synthetic K, 7, 32,768, K}, and across eval batch sizes.
  - Given identical ranks, the float64 metrics must be identical.
- **G4 Alignment** (PKG 09 §1 L6): prediction rows align by decision_id and the manifest's
  `eval_ordered_decision_id_sha256`. A mismatch is fatal.
- **G5 TEST access:** the evaluator never loads TEST itself; that is left to the caller (`eval_full.py:L21-L22`). In
  V4 the manifest loader enforces §10.

### 6.4 Prediction artifact
- **E6.3** (PKG 09 §1 L6):
  - Per scored manifest, the artifact holds one int64 exact rank per row, in manifest order (−1 for non-RANKABLE
    rows).
  - Its header holds the manifest `eval_ordered_decision_id_sha256`, the model/checkpoint sha256, the evaluator code
    sha256 and `candidate_order_hash`.
  - Ranks above 20 are kept exactly. An exported top-20 is not a substitute.

### 6.5 Tie diagnostic (PLAN; descriptive only)
- **E6.4:**
  - Each result row reports `n_target_tied`, the number of scored rows where at least one other class has s_j == s_t.
  - If it is > 0, the fully pessimistic rank 1 + #{s_j > s_t} + #{j ≠ t : s_j == s_t} is also reported, as a
    sensitivity.
  - Motivation: Local-only may tie many classes (PKG 04 §1 L6). The headline always uses the quoted rule.

## 7. Statistics

### 7.1 Unit, seed and resamples
- **E7.1** (PKG 09 §4 L19; FABLE D8 L49; FABLE LOCK EVAL L64): paired user bootstrap.
  - The unit is the user (eval_user_key) of the registered manifest.
  - **1000 resamples, seed 20260926.**
  - This supersedes the legacy seed 2026 (`eval_full.py:L405`; GRU1 FULL_RUN_PROTOCOL L91).

### 7.2 Shared resample matrix W (PLAN)
- **E7.2 Construction:**
  - Users are sorted ascending by eval_user_key (as in `eval_full.py:L418-L421`).
  - `rng = numpy.random.default_rng(20260926)`; then for r = 0..999, `W[r] = rng.integers(0, n_users, size=n_users)`.
    This is the algorithm of `eval_full.py:L433-L436` with the V4 seed.
  - W is built once per distinct evaluated user set and registered with its sha256 (MANIFEST_SPEC `bootstrap`).
- **Sharing** (PKG 09 §4 L19, "same sampled user weights"):
  - W is reused by every method, seed and contrast on that user set.
  - Every method must cover the identical user set (`eval_full.py:L423-L428`).
- **Regeneration:** W may be regenerated rather than stored, but the registered sha256 is checked before every use.
  - INFERENCE: the hash, not the numpy version, defines W. Storing W for M03 at int32 would take about 1.5 GB.
- **E7.3 Statistics per resample:**
  - Macro: M_b = (1/n) Σ_j m_{W[b,j]}, using the per-user means m_u (`eval_full.py:L362-L401`).
  - Micro, in cluster form: Σ_j S_{W[b,j]} / Σ_j N_{W[b,j]}, where S_u and N_u are user u's credit sum and row count.
  - RANKABLE statistics use the same draw: m_u is averaged over the drawn users (with multiplicity) that have
    R_u ≠ ∅.
  - The CI is the 95% percentile interval [2.5, 97.5] (`eval_full.py:L439`).

### 7.3 Contrasts
- **E7.4** (PKG 09 §4 L19; T38): every preregistered contrast is bootstrapped, whatever its sign. Each reports:
  - the point estimate and CI;
  - n_users and n_decisions;
  - the W sha256.
- **Minimum registry** (PLAN; frozen per stage before that stage's first outcome):
  - the family contest GRU − SASRec (Central, cohort), per seed and as the seed mean;
  - FA, FP, PF and LO each against C (delta and retained);
  - FP against FA;
  - PF against CPF (capacity-matched) and PF against C (practical augmented system) (PKG 09 §3 L16);
  - the same contrasts on M03;
  - per cold-start panel and h.
- The final registry is PENDING; it is frozen at ARCHITECTURE_LOCK for the downstream stage.

### 7.4 Ratios are bootstrapped directly
- **E7.5** (PKG 09 §4 L19; T39):
  - On each W row b: retained_b = 100·M_b/C_b and loss_b = 100·(C_b − M_b)/C_b. The CI comes from their percentiles.
  - Marginal CI endpoints are never divided.
  - If C_b = 0 in any resample, that resample's ratio is undefined. The CI is then reported as UNDEFINED_DENOMINATOR,
    with the count. No resample is dropped.

### 7.5 Per-seed reporting
- **E7.6** (PKG 09 §4 L20; FABLE D8 L49):
  - Values are reported for seeds 2026 and 2027 separately.
  - Paired per-seed deltas pair seed s with seed s; the initialization is the same under I_META_RANDOM.
  - The seed mean is also reported.
  - The seed-mean contrast uses the same W for both seeds: Δ_b = mean_s(M^A_{s,b} − M^B_{s,b}).
  - Reporting only the best seed is forbidden.
- **Family selection** is not a bootstrap decision. The winner is the family with the highest seed-mean macro MRR@20
  on M02 (OWNER_SCOPE_V4 binding scope item 1; T22), using [0, 6]-EFE values only (E12.1, G0B C3 L18).

### 7.6 Claim vocabulary (carried over from GRU1 FULL_RUN_PROTOCOL L92; PLAN)
- "A exceeds B" iff the CI lower bound is > 0.
- "B exceeds A" iff the CI upper bound is < 0.
- Otherwise "no detectable difference".

### 7.7 Scope (LIMITATION; PKG 09 §4 L20)
- The intervals are conditional on the trained models and on the adaptive VALIDATION/TEST exposure history.
- They do not capture all temporal dependence.
- Any day-level sensitivity table is descriptive.

## 8. Retained quality

- **E8.1** (PKG 09 §3 L13-L15): for a central C and a method M on the *same* population, metric, aggregation,
  denominator and seed pairing:
  - `retained_percent = 100·M/C`;
  - `relative_loss_percent = 100·(C − M)/C`;
  - `delta = M − C`.
  - Each row carries `matched_central_row_id` (RESULT_ROW L39-L41).
- **E8.2 No clamping:** retained > 100% and a negative loss are both valid (PKG 09 §3 L15; T39 L50). Central is an
  empirical reference, not an upper bound.
- **E8.3 Zero denominator** (T39): if C = 0, retained and loss are null with status UNDEFINED_ZERO_DENOMINATOR, never 0.
  The delta is still reported. C < 0 is impossible because every credit is ≥ 0; this is asserted.
- **E8.4 Bootstrap:** ratios are bootstrapped directly (§7.4).
- **Labelling** (PKG 09 §3 L16):
  - PF/CPF is the capacity-matched comparison. PF/C is the practical augmented-system comparison. The two are labelled
    distinctly.
  - LO's lack of other-user supervision is intentional.
- **E8.5:**
  - A cohort method is never divided by a full-data central score (PKG 04 §3 L23).
  - A RANKABLE retained value needs a RANKABLE central value.

## 9. Comparison key (must match within a block; PKG 09 §2 L9)

| Identity (PKG 09 §2 L9) | RESULT_ROW field (template line) | V4 binding |
|---|---|---|
| data | `data_protocol_hash` (L7) | DATA_PROTOCOL_LOCK sha256 (PENDING) over the frozen dataset `860e5a1c3a81b8e4bbfeff4f245fa37638a43e4f260df179585964d2d4da8dbd` (CENSUS_README L3-L5). Dataset build `code_hash` `02fbb07147ed9c0ba54092dd4ee042ce6af42a06145178017999fb7b242dde73`, recomputed by QA-01 and matching both frozen records (CODE_FLAGS L4-L9) |
| feature | `feature_contract_hash` (L10) | clean Rich column list with the per-column (a)-(d) table (FABLE D4 L29); PENDING |
| split | `split` (L13) + `label_policy_hash` (L12) | §1 bounds and §2 status/censor rules |
| cohort | `population_hash` (L8) | `selected_keys_sha256_hex`; for M01/M02 it is `54e1d3c2539fd4cd7fe22d2b8ef91bed1a786d69be27867406bb19141ac48184` (COHORT_PROJECTIONS L7188) |
| query | `query_order_hash` (L11) | the manifest `eval_ordered_decision_id_sha256` |
| candidate | `candidate_order_hash` (L9) | catalogue order hash, K = 158,486 |
| initial resource | `initialization_block` (L6) | I_META_RANDOM snapshot hash + per-seed init hash (PKG 04 §4 L26) |
| topology | *(added)* `topology_hash` | ARCHITECTURE_LOCK |
| objective | *(added)* `objective_id` | exact full-K CE (PENDING lock) |
| precision | *(added)* `precision_id` | strict FP32, TF32 off for matmul and cuDNN (FABLE D5 L31-L33) |

- **Evaluation identity:** `metric`, `k`, `aggregation`, `target_denominator` (L14-L17), plus the added fields
  `evaluator_code_sha256` and `bootstrap_W_sha256`.
- **E9.1** (PKG 09 §2 L9):
  - Intended differences (training paradigm, local personalization, optimizer lifecycle) are recorded per row.
  - A row with any mismatched key is a HISTORICAL_REFERENCE and is never labelled a same-conditions paradigm effect.
- **E9.2:** every V4 row has `previously_seen_evaluation_period = true` (RESULT_ROW L44; PKG 03 §4 L24).
- **E9.3** (PLAN): the template has no topology, objective, precision, evaluator or bootstrap fields. V4 result rows add
  them (flagged to the owner of the results schema).

## 10. TEST exposure

### 10.1 ALL_METHODS_LOCK comes first
- **E10.1:** no RETRO_TEST manifest is built, and no RETRO_TEST decision, label or feature is loaded, until
  `ALL_METHODS_LOCK.json` exists and its sha256 is in the TEST_EXPOSURE_REGISTER.
  - Sources: OWNER_SCOPE_V4 gate order items 11-12; PKG 01 L19; PKG 09 §2 L10; PKG 13 L22; T48 L61.
  - The builder and the loader both refuse otherwise.
- TEST runs once per registered method and block (PKG 01 L19). Architecture selection is validation-only
  (PKG 09 §2 L10).

### 10.2 TEST_EXPOSURE_REGISTER
- **E10.2** (PKG 03 §4 L24; PKG 13 L22): an append-only, hash-chained JSON register. Its fields cover:
  - source, split, access purpose, observations, decision and timestamp (the PKG 03 §4 list);
  - actor role, the ALL_METHODS_LOCK sha256, the manifest, model and code hashes, and the receipt path.
  - The schema is in MANIFEST_SPEC `registers.TEST_EXPOSURE_REGISTER`.
- **Seed entries:** the prior TEST exposures of earlier campaigns ("Old TEST has been observed", PKG 09 §2 L10) are
  recorded by path and date only, with no values. PENDING: to be filled by the Operations Director.

### 10.3 Immutability and restart
- **E10.3** (PKG 09 §2 L10): after ALL_METHODS_LOCK, the TEST evaluation of the registered methods is immutable.
  - Only an interrupted, incomplete computation may be restarted, with identical model and input hashes and a
    receipt.
  - Weights are never edited because of interim scores. Nothing from TEST feeds back into any choice.
- **LO:** the TEST-time LO row is a deterministic retrain, backed by a receipt that reproduces the VALIDATION ranks
  exactly (FABLE D7 L46; FABLE B7 L74).

### 10.4 Structural TEST counts already present in G0 files and QA-01 receipts
- RETRO_TEST structural counts exist in these files:
  - G0 census: SPLIT_COUNTS L54-L69 and L87-L102, CLIENT_CENSUS L3973-L3977, LABEL_TIME_AUDIT L29-L33, and
    BARRIER_COUNT L4-L6;
  - QA-01 receipts: BOUNDARY_DEFS L164 and L195-L196, and COLD_STRATA L19-L28 (users and eval rows per cold stratum).
- They are counts, not model outcomes, and this contract quotes none of them. It only states that they exist.
- **E10.4** (PLAN): register the G0 census and the QA-01 receipts as STRUCTURAL_COUNT entries in the
  TEST_EXPOSURE_REGISTER.

## 11. Cold start

### 11.1 Budgets and panels
- **E11.1 Budgets:** h ∈ {0, 1, 5, 10, 20, 50}, where possible (PKG 09 §5 L23). h counts observed events, not
  supervised labels.
- **E11.1a Panels (binding; G0B C4 L20, G0B LOCK L34; resolves P11):**
  - **M04 COLD_H_PANEL = candidate A.** The 16,384 M03 users with the smallest population selection keys (salt
    PPSI_V4_POPULATION_20260926, CLIENT_CONTRACT C2.2). All of them are cold.
    - The stratum mix (COLD_NO_TRAIN_EVENTS vs COLD_TRAIN_EVENTS_NO_LABEL) and the FIXED_50, FIXED_20 and
      ALL_AVAILABLE(h) eligibility are reported at build and never used to select.
  - **M04W WARM_TRUNCATION panel.** The 16,384 users with the smallest population selection keys among M02's 62,368
    E2E users (FABLE D8 L49). It is nested in the population key order and uses no new salt.
    - It serves T41 (E11.7) and the warm-p_u evaluation of PKG 09 §5 L27.
    - It is always labelled WARM_TRUNCATION and never pooled with, or reported as, cold start.
    - INFERENCE, reading "M02 users" as the evaluable users: it is not the 16,384-user cohort prefix (COHORT_PROJECTIONS
      L1584). That prefix includes users with no VALIDATION query.
  - **Candidate B is historical only.** QA-01's panel (COLD_H_PANEL L3-L16: 45,572 qualifying; 16,384 selected under
    salt PPSI_V4_COLD_H_20260926; keys sha256 `1ce796f115cf347199e83736f84ff679a67897898eaf7fa2b7bfa81a88bfed91`;
    13,767 warm / 2,617 cold) is retained as a historical receipt. It is not a V4 panel.
    - Fable's reason (G0B C4 L20, INFERENCE): B qualifies users on evaluation-period activity, and 84% of it is warm,
      so it measures truncation of trained history, not cold start.
- **LIMITATION** (G0B risks L42): candidate A gives small FIXED_50 support, so ALL_AVAILABLE and FIXED_20 carry the
  study.
- **Sub-panels** (confirmed by G0B C6 L24): FIXED_50, FIXED_20 and ALL_AVAILABLE(h) (§11.2), for M04 and for M04W
  alike.

### 11.2 Query and history definitions (PLAN)
- **Observed event of user u:** one ledger decision row of u in TRAIN or VALIDATION, i.e. a representative view after
  consecutive-view compression (PKG 03 §1 L6, §5 L27; DATA_CONTRACT L228-L230). Events are ordered by
  (decision_ts, decision_id).
- **Query q:** an E2E VALIDATION decision of u.
  - The query's own event e_q is always visible and is **not** counted in h; h counts strictly earlier events.
  - So h = 0 gives a context length of 1 (PKG 03 §3 L20; DATA_CONTRACT L121, `lengths >= 1`). **Confirmed** by G0B C6
    L24: h = 0 means the query event only.
- **One query per user per sub-panel** (confirmed by G0B C6 L24).
- **Permitted history:** H_h(q) = the h most recent observed events of u strictly before e_q.
- **FIXED_50:** users with at least one E2E VALIDATION decision that has ≥ 50 prior events.
  - q_u is the earliest such decision, and the same q_u is used for every h (PKG 09 §5 L25).
- **FIXED_20:** the same rule with 20 prior events, for h ∈ {0, 1, 5, 10, 20}. Added and confirmed by G0B C6 L24.
- **ALL_AVAILABLE(h):** users with at least one E2E VALIDATION decision that has ≥ h prior events; q_u(h) is the
  earliest such decision.
  - The queries differ by h, so this panel is reported separately (PKG 09 §5 L25).
- **E11.2:** eligibility counts are reported per sub-panel and stratum. Selection never uses outcomes (PKG 04 §3 L22).

### 11.3 Support labels (T40)
- **E11.3** (PKG 09 §5 L23): a support label is a transition e_j → e_{j+1} that meets all of:
  - e_j ∈ H_h(q);
  - its target is observed and uncensored, with target_class ≥ 0 (the TRAIN loss rule, E1.6);
  - its target timestamp is **strictly before** decision_ts(q).
- **Consequences:**
  - The transition into e_q is never support, and the query target is never support.
  - A raw event alone supplies no label, and h = 1 may yield 0 labels.
  - Labels matured at or after VALIDATION_END, or in LABEL_GRACE, never support a VALIDATION query.

### 11.4 Truncated-history recompute rules per locality class
- **E11.4 Governing rule** (PLAN; PKG 09 §5 L27; PKG 03 §8 L47): the query's features are exactly what the clean builder
  produces for a user whose observed history is H_h(q) followed by e_q.
  - This is testable by golden replay (T13, T41).

**Clean V2 table** (G0B C1(a) L14; R2).
- All SCHEMA_V2 line numbers below refer to schema version 2.1.0 (SCHEMA_V2 L3), with `schema_self_sha256`
  `55a244e1608bcdcdf6f1728595f534871b52154faebf2977e11ac1fb487052f9` (L4).
- The counts are the Clean V2 kept channels: 53 in total (SCHEMA_V2 L105, L113). The v1 counts were
  27 / 26 / 3 / 3 / 0 (SCHEMA_V2 L114; LOCALITY_MAP L22-L29).
- Channel positions follow `tensor_layout_v2` (SCHEMA_V2 L574-L587).
- Six v1 channels are not in V2 (SCHEMA_V2 L106): `local_hour_sin`, `local_hour_cos`, `prefix_item_count`,
  `log1p_prefix_elapsed_s`, `has_history_mask` and `prev_user_event_available`.

| Class (LOCALITY_MAP L14-L18) | V2 count (SCHEMA_V2 L113) | V2 channels | Rule under truncation | Notes |
|---|---|---|---|---|
| LOCAL_EVENT | **25** | `daypart_tokens`, `weekday_tokens` (L576); `event_numeric_features[0..7]`: `log1p_price`, the four `utc_*` cyclic channels, `local_weekday_sin/cos`, `is_weekend` (L577-L578); all 15 `event_quality_flags` (L581-L582) | Taken unchanged from the permitted event rows | Exception (INFERENCE from the LOCALITY_MAP L59/L85 pairing): `prev_gap_unavailable` (V2 flags[12], L582) is set from the permitted history. For the earliest permitted event the previous event is outside the budget, so the flag = 1 and `log1p_prev_raw_view_gap_s` (V2 event_numeric[8]) is re-zeroed (rezero_map_v2, L564). |
| LOCAL_HISTORY | **22** | `event_numeric_features[8, 9, 11, 12, 13, 14]`: `log1p_prev_raw_view_gap_s`, `log1p_segment_elapsed_s`, `prefix_representative_count`, `prefix_known_category_count`, `prefix_category_switch_count`, `current_known_category_run_length` (L578); `user_context[12]` (L583-L584); `user_context_masks[4]`: `recent_category_match_supported`, `recent_brand_match_supported`, `recent_price_supported`, `prev_session_available` (L586-L587) | Recomputed from H_h(q) ∪ {e_q} only | 21 are recomputable (YES in LOCALITY_MAP). One is PARTIAL: `log1p_segment_elapsed_s` (LOCALITY_MAP L60), which takes its origin at the earliest permitted event of the segment and is disclosed as a truncated-origin value. `log1p_prefix_elapsed_s` is not a V2 channel. Counts, gap availability, price affinities, recency and masks all come from permitted events only (PKG 09 §5 L27). The stored, un-batched `prev_user_event_available` is still recomputed from the permitted history, because the M1 re-zero of `log1p_time_since_prev_user_event_s` reads it (rezero_map_v2, L568). |
| SHARED_CATALOGUE | 3 | `category_tokens`, `brand_tokens`, `main_category_tokens` (L576) | Frozen end-of-TRAIN snapshot; identical across h; outside the local budget | LOCALITY_MAP L39-L41; frozen future-partition snapshot, L233-L237 |
| SHARED_TRAIN_STATISTIC | 3 | `item_tokens` (L576); `log1p_item_observed_age_s` (V2 event_numeric[10], L578); output catalogue | Frozen; identical across h | item observed age from the pooled `first_seen_train` (LOCALITY_MAP L61, L245-L258); catalogue (L121) |
| UNSUPPORTED_FUTURE | 0 | none | Not permitted; any occurrence fails the build | SCHEMA_V2 L113; LOCALITY_MAP L28 |

- Consistency check (derived from the schema rows; not a measurement): the classes sum to 25 + 22 + 3 + 3 = 53, the
  V2 total (SCHEMA_V2 L105). The block sizes (numeric 15 / flags 15 / uctx 12 / masks 4) match G0B LOCK L31 and
  SCHEMA_V2 L112.
- **E11.5 Normalization** (PKG 03 §6 L35, L38; G0B C1(b)-(d) L14, C2 L16): recomputed raw values are normalized with
  the frozen V2 normalizers.
  - There are **17 normalized channels** (`normalized_channels_v2`, SCHEMA_V2 L540-L558). The schema breaks them down
    as "14 with unchanged v1 normalizers + 2 M1 valid-only refits + 1 F-1 valid-only refit" (L110-L111). So they
    include:
    - the two M1 valid-only re-encodings (SCHEMA_V2 L107, L556-L557);
    - the **F-1** valid-only refit of `current_minus_recent_mean_log_price` (SCHEMA_V2 L555; OI03 L26-L31;
      G0B C2 L16): mean 0.002065693626402541, std 0.9945344391661521, n_valid 48,434,230, at the recorded clip
      bounds −4.090688228607178 / 3.681367874145508.
  - Unavailable values are re-zeroed through `rezero_map_v2` (SCHEMA_V2 L562-L571). This includes the F-1 extension:
    the value is zero when `recent_price_supported` = 0, or when `price_missing` or `price_conflict` holds at the
    decision's own event (L567).
  - Under truncation, `recent_price_supported` is recomputed from H_h(q), and the price flags of e_q are its own.
  - There is no per-h or per-client refit.
  - G0B C1(d) is applied in schema version 2.1.0 (SCHEMA_V2 L3-L4, L9): the F-1 values are in `normalized_channels_v2`
    and `rezero_map_v2`, and the schema carries a self-hash. The builder and evaluator must verify
    `schema_self_sha256` before use.
- **E11.6 Private state** (PKG 09 §5 L25, L27; T29):
  - Cold users have no previously trained p_u; p_u = 0 at the start.
  - A p_u trained during global training is evaluated only on warm clients (the M04W panel), with its prior-support
    budget disclosed.
- **E11.7 Negative control** (T41 L52; G0B C4 L20):
  - On the M04W WARM_TRUNCATION panel (warm cohort users), a deliberately leaking builder (full-history
    `user_context`) must produce features that differ from the E11.4 output, and the test must catch the difference.
  - The E11.4 output must not change when every event before the permitted window is deleted from storage.

### 11.5 Study A — CONTEXT_ONLY (PKG 09 §5 L25)
- **E11.8:**
  - All model and personal parameters are frozen. Only the permitted context and the LOCAL_HISTORY aggregates vary.
  - p = 0 for unseen users.
  - The FIXED panels use the same queries for every h. ALL_AVAILABLE is reported separately.
- Every method runs from its locked checkpoint.
- For a zero-label cold user, LO is its per-seed initialization plus the shared bootstrap information
  (CLIENT_CONTRACT §4). This is intentional (PKG 09 §3 L16).

### 11.6 Study B — FEW_SHOT_ADAPTATION (PKG 09 §5 L26)
- **E11.9:**
  - A method or its personal component learns only from its own support (§11.3).
  - It uses a frozen adaptation recipe selected on VALIDATION.
  - Query targets never feed back into training. There is no online/prequential variant unless one is separately
    registered and uses only previously revealed labels.
- **E11.9a Recipe-selection panel and grid** (binding; G0B C6 L24, G0B LOCK L34; resolves P5):
  - **M04S** = the M03 users at population-key ranks 16,384–32,767 (0-based), disjoint from M04.
  - **The recipe grid** is registered at ARCHITECTURE_LOCK:
    - p_u-only adaptation for PF and CPF;
    - full local-model adaptation for LO;
    - local passes ∈ {1, 2, 4};
    - locked personal LR and λ.
  - It is selected **once**, on M04S FIXED_20, before any M04 outcome is observed. M04 outcomes never select a recipe.
- **Support permissions** (G0B C6 L24):

  | Method | May adapt | Support |
  |---|---|---|
  | PF, CPF | Personal component p_u only (initialized to 0), under the registered grid | Own support only |
  | LO | Full local model, under the registered grid | Own support only |
  | C, FA, FP | No adaptation is registered in the G0(b) grid, so they appear in Study A (CONTEXT_ONLY) only (INFERENCE from the C6 wording) | — |

### 11.7 Events-to-90% / 95%
- **E11.10** (binding; G0B C6 L24; PKG 09 §5 L28; resolves P4).
  - **C_ref(seed, panel, sub-panel)** is the central C method, evaluated with:
    - the same seed;
    - the PRACTICAL_BEST checkpoint (§12);
    - CONTEXT_ONLY;
    - the same queries;
    - the full available prior history. For ALL_AVAILABLE it is computed per h, on that h's queries.
  - **events-to-X%** is the smallest registered h with M(h) ≥ X · C_ref, for X ∈ {0.90, 0.95}.
    - It is reported only if reached at a registered h. No interpolation, no extrapolation.
  - M's own full-history value is always reported alongside.

## 12. Interfaces with other stages: checkpoint selection and the binding stopping rule

- Checkpoint selection uses only the VALIDATION primary: M02, or M06 for calibration runs.
- **E12.1 Stopping rule** (binding; G0B C3 L18; G0B LOCK L33; resolves P3 and the ambiguity in OPUS-R L22-L33):
  1. **No early stop before 6.0 EFE**, for either family. The "3.0-EFE reference" of FABLE D6 L38 is void.
  2. **Evaluate every 0.5 EFE** on M02, or M06 for calibration.
  3. **Save two checkpoints:** the strictly-best one (Δ > 0.0002 over the running best) and `endpoint_6.0`.
  4. **PRACTICAL_BEST** = the strictly-best checkpoint in [0.5, 6.0]. **CONTROLLED_BUDGET** = the 6.0 endpoint.
  5. **Only [0, 6] values** are used for family selection and for every matched block.
  6. **STILL_IMPROVING:** at 6.0 a run is STILL_IMPROVING iff its strictly-best checkpoint is at 5.5 or 6.0 (patience
     2).
     - Only then, and only with a pre-recorded budget, may it continue from the 6.0 resumable to 8.0.
     - The continuation LR is 0.1 × peak on (6.0, 7.5], then linear to 0.01 at 8.0 (fractions of peak, as in the
       FABLE D6 L36 schedule).
     - The continuation stops after 2 non-improving evaluations.
  7. **Continuation rows** are labelled **PRACTICAL_BEST_EXTENDED**. They are secondary and never enter family selection
     or a matched-block contrast.
  8. **Disk:** the resumable is deleted at 6.0 unless the run is STILL_IMPROVING (disk plan rule 2).
- **E12.2 Result-row labels** (PLAN): each central or downstream result row records:
  - its `selection_rule` ∈ {PRACTICAL_BEST, CONTROLLED_BUDGET, PRACTICAL_BEST_EXTENDED};
  - its `efe` (PKG templates/RESULT_ROW.template.json L24, L26).
  - Which of PRACTICAL_BEST and CONTROLLED_BUDGET is the family-selection value is not fixed by this contract. Both come
    from [0, 6].
- The stopping rule decides which checkpoints are evaluated. It does not change this evaluator.

## 13. Acceptance-test map (PKG 12)

| Test | Line | What this contract requires | Evidence now |
|---|---|---|---|
| T03 | L8 | E1.1-E1.3: half-open bounds; VALIDATION = VS ∪ VC exact membership; legacy barriers kept | MEASURED union counts (SPLIT_COUNTS L72-L85) |
| T04 | L9 | E1.6, E1.8: TRAIN label strictly before TRAIN_END; no post-VALIDATION target selects a model | MEASURED 0 / 0 / 0 (LABEL_TIME_AUDIT L11-L23) |
| T05 | L10 | E1.9: GRACE emits 0 decisions and matures RETRO_TEST labels only | MEASURED 0 decisions ≥ 11-30 (LABEL_TIME_AUDIT L34); by construction `pipeline5.py:88-92`, `211-212` (BOUNDARY_DEFS L78-L85) |
| T06 | L11 | E1.7: target before filter; no A→C reconnection; A→B→A kept | DATA-owned builder test; PENDING |
| T07 | L12 | Session gap 3600 vs 3601, multi-user sessions, equal timestamps; consumed by the §2 status rules | DATA-owned builder test; PENDING |
| T35 | L46 | E4.1-E4.2: OOV = 0 and identical E2E/RANKABLE populations across methods (hash equality) | spec |
| T36 | L47 | E4.6: micro identity, macro identity, T36-A/T36-B counterexamples | spec with fixtures |
| T37 | L48 | E6.2, G2, G3: no sampled/ANN/top-k evaluation; deterministic ties; chunk invariance | legacy code satisfies the rule (`eval_full.py:L68-L70`) |
| T38 | L49 | E7.2-E7.6: shared W, all contrasts regardless of sign, per-seed output | spec |
| T39 | L50 | E8.2-E8.3: retained may exceed 100%; zero denominator → undefined | spec |
| T40 | L51 | E11.3: h = 1 may have 0 labels; support strictly before query; no query target in support | spec |
| T41 | L52 | E11.4-E11.7: recompute under truncation; leakage negative control | spec |

Passing these tests supports the tested contract only. It does not establish convergence, causal attribution or
privacy (PKG 12 L65).

## 14. PENDING register

| ID | Item | Owner | Blocks |
|---|---|---|---|
| P1 | Prose definitions of AMBIGUOUS_BOUNDARY / BOUNDARY_UNRESOLVED (§2.3); status of the single TRAIN BU row (§2.4) | **RESOLVED in R1** by QA-01 (BOUNDARY_DEFS) | — |
| P2 | Cohort s=5 coverage, class counts and cold-strata counts: **RESOLVED in R1** (§3.2, E3.4; COHORT_RECEIPTS, COLD_STRATA). Still open: per-manifest censor-class, OOV-cause and metadata counts, and M03/M04 rankable counts | PENDING_RECEIPT (builder) | POPULATION_PLAN |
| P3 | Stopping/patience interpretation (OPUS-R L22-L33) | **RESOLVED by G0(b)** C3 L18 (E12.1) | — |
| P4 | C_ref for events-to-X% (§11.7) | **RESOLVED by G0(b)** C6 L24 (E11.10) | — |
| P5 | FEW_SHOT recipe-selection panel and recipes (§11.6) | **RESOLVED by G0(b)** C6 L24 (E11.9a); the grid values are registered at ARCHITECTURE_LOCK (P15) | — |
| P6 | Confirm h = 0 semantics, the FIXED_20 addition, and one query per user per sub-panel (§11.2) | **RESOLVED by G0(b)** C6 L24 | — |
| P7 | DATA_PROTOCOL_LOCK, feature contract, I_META_RANDOM snapshot and topology/objective hashes (§9) | lock stage | every result row |
| P8 | TEST_EXPOSURE_REGISTER location, seed entries and the G0 structural-count entry (§10) | Operations Director | before any TEST action |
| P9 | Contrast registry frozen per stage (§7.3) | Evaluation Lead + Fable | each stage's first outcome |
| P10 | RESULT_ROW additions (§9, E9.3) | results-schema owner | reporting |
| P11 | COLD_H_PANEL definition (§11.1) | **RESOLVED by G0(b)** C4 L20: M04 = candidate A, plus M04W WARM_TRUNCATION; candidate B is historical only (E11.1a) | — |
| P12 | Placement of the 140 AB VS→VC rows (E2.2) | **RESOLVED by G0(b)** C5 L22: moved to CEN_AMBIGUOUS_NEXT_BARRIER; CEN_LEGACY_BARRIER = 170 | — |
| P14 | F-1 values written into the schema's `normalized_channels_v2` and `rezero_map_v2`, with version bump and self-hash (E11.5) | **RESOLVED** by DATA in schema version 2.1.0 (SCHEMA_V2 L3-L4, L555, L567; G0B C1(d) L14) | — |
| P15 | Recipe-grid values (passes {1,2,4}, locked personal LR and λ) registered | ARCHITECTURE_LOCK (G0B C6 L24) | Study B |
| P13 | Why the TRAIN_END fence has 1 BU row against 170 and 152 at the VALIDATION fences (§2.4); non-blocking for evaluation | DATA | none |
