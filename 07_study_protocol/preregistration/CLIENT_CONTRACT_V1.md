# CLIENT_CONTRACT_V1 — logical clients, cohort, cold strata, data ownership and shared bootstrap information (specification)

- **Task:** EVAL-01 (Wave A). **Author:** Evaluation Lead (Opus). **Written:** 2026-09-26.
- **Status:** PLAN / SPECIFICATION. It contains no model outcome and no TEST-partition number.
- **Revision R1** (EVAL-01R1, 2026-09-26): the QA-01 receipts are incorporated.
  - The cohort receipts and the cold-strata counts are quoted.
  - QA-01's cold-h panel does not match C3.4, so the choice goes to Fable (C3.4).
- **Revision R2** (EVAL-01R2, 2026-09-26): Fable's binding G0(b) ruling (`V4/company/reviews/G0B_PROTOCOL_FABLE_POSITION.md`,
  cited as G0B) is applied.
  - M04 = candidate A, plus a WARM_TRUNCATION panel and M04S (C4, C6).
  - §6 is updated to Clean V2: 17 normalized channels plus the F-1 refit (C1(b)), and 53 channels.
  - The recentered SASRec θ0 is noted in §6 item 1 (C7). The Operations Director did not list C7; it was added so that
    §6 matches the binding ruling.
- **Binding:** at POPULATION_PLAN, once the Operations Director accepts it and Fable approves.
- **Companions:** `EVALUATION_CONTRACT_V4.md` (path roots, citation forms and tags are defined in its §0) and
  `MANIFEST_SPEC_V4.json`.
- **Extra citation forms:** `COHORT` = `CEN/COHORT_PROJECTIONS.json`, `CLIENT_CENSUS` = `CEN/CLIENT_CENSUS.json`,
  `README` = `CEN/CENSUS_README.md`. `COHORT_RECEIPTS`, `COLD_STRATA`, `COLD_H_PANEL` and `CODE_FLAGS` =
  `V4/g0/receipts_qa01/` files `COHORT_RECEIPTS.json`, `COLD_STRATA.json`, `COLD_H_PANEL.json` and
  `CODE_AND_FLAG_CHECKS.json`.

## 1. Identity: one persistent user = one logical client

- **C1.1 One user, one client** (PKG 04 §3 L19; T23 L32):
  - One persistent user is one logical client.
  - Storage buckets are not clients.
  - User histories are never merged to make a client larger.
  - There is no generic "≥ 5 interactions" filter.
- **C1.2 Private client key** (PLAN; T02 L7):
  - `client_key_hex = sha256(b"PPSI_V4_POPULATION_20260926" + b"|" + user_id.encode("utf-8")).hexdigest()`, the full
    32-byte digest in lowercase hex.
  - Its first 8 bytes, read big-endian, are exactly the census selection key (COHORT L4).
  - user_id is the canonical raw STRING identity (DS1/DATA_CONTRACT.md L177-L181).
  - **Collision-free by assertion:** in every manifest, and across the union of all manifests built so far, the count
    of distinct `client_key_hex` must equal the count of distinct user_id strings. Any collision is fatal.
- **C1.3 Never the OOV token** (PKG 09 §1 L5; PKG 04 §3 L19; T02, T29 L38):
  - client_idx is the TRAIN-fitted user vocabulary. TRAIN-absent users all share OOV = 2 (LOCALITY_MAP L178-L182;
    DATA_CONTRACT L194-L196).
  - It is never used as a client identity, a private-state key or a macro key.
  - Future users who share the OOV input id stay distinct in private state and in evaluation.
- **C1.4 Manifest keys** (PLAN):
  - `client_ord` (TRAIN manifests) and `eval_user_key` (evaluation manifests) are the 0-based dense rank of
    `client_key_hex`, ascending, among the users present in that manifest. They are int64, as the evaluator requires
    (`eval_full.py:L156`).
  - Joins across manifests use `client_key_hex` only.
  - Ascending hex order equals ascending selection-key order; the remaining digest bytes break ties deterministically.
- **C1.5 A pseudonym, not anonymity** (LIMITATION; OWNER_SCOPE_V4 constraints; README L5-L8):
  - Anyone who holds the salt and a user_id can recompute the key.
  - Manifests and private state are therefore private artifacts. Shared reports carry counts and aggregates only.
  - Raw user_id is never written (MANIFEST_SPEC `global_rules.no_raw_ids`).

## 2. Cohort definition (FABLE D1 L13-L16; FABLE LOCK POPULATION L56)

- **C2.1 Universe** (COHORT L3, L8; CLIENT_CENSUS L3, L7):
  - U = the users with at least one TRAIN-eligible example at support ≥ 5.
  - An eligible example is: split = TRAIN, loss_mask = true, target observed with target_ts < 2019-11-16T00:00Z, and
    target in the support ≥ 5 catalogue.
  - |U| = 2,694,998.
- **C2.2 Key rule** (COHORT L4, L13043-L13046):
  - The key is `int.from_bytes(sha256(b'PPSI_V4_POPULATION_20260926' + b'|' + user_id.encode('utf-8')).digest()[:8], 'big')`.
  - cohort(N) = the N users of U with the smallest keys.
  - Every smaller cohort is a prefix of every larger one (checked, L13044-L13045).
- **C2.3 N = 262,144** (FABLE D1 L13-L16; PKG 04 §3 L21): chosen once, from resource projections and evaluation
  support, not from scores.
- **C2.4 Reproducibility hashes and sizes (MEASURED).** `selected_keys_sha256_hex` is defined at COHORT L5.

| Cohort | selected_keys_sha256_hex (line) | Users | Eligible TRAIN examples | VALIDATION eval rows | Users with ≥1 eval row | Users with 0 eval rows |
|---|---|---|---|---|---|---|
| **262,144 (main)** | `54e1d3c2539fd4cd7fe22d2b8ef91bed1a786d69be27867406bb19141ac48184` (L7188) | 262,144 (L3884) | 3,906,929 (L3885) | 808,610 (L6056) | 62,368 (L6057) | 199,776 (L6060-L6061) |
| **65,536 (nested calibration prefix)** | `3d680aa7222fb5c9246d03fe6088464c78095d90a344e3453acb68e98d53a6f4` (L3880) | 65,536 (L1588) | 990,362 (L1589) | 200,985 (L3084) | 15,487 (L3085) | 50,049 (L3088-L3089) |

- **Other recorded projections that are not V4 populations:**
  - The 16,384 prefix, `3b845f51260fab7d10590827eafd0c9d8b58a3813f482d88b5ca4826f77a8c92` (L1584). It is a *warm*
    prefix and is **not** the cold-h panel of §3.4.
  - ALL, `db46b81d6ceb7d0c766e76ca2d0f5dede905cedf4b91881e168be4476b9bee14` (L13040).
- **Census script hashes** (README L31-L36):
  - `common.py` `8823d7e039a6ffb16a13d83e3ef400f3a67f6472979f84654906bb60a02334ab`
  - `pass1_support.py` `a2f9cf33ef0a2b6ed5befbcae87244e9e3a15f1946453040cecd98b035bf58db`
  - `pass2_census.py` `3e79b764a43e0616d692ba724226beb2e8ae8212c2a640ad645377c4d38783d0`
  - `validate_rerun.py` `4605a0dbaaf0465b6f80013c1aee2c4d6bc9351c0afe2d252f716be931ad796a`
- **C2.5 Rebuild acceptance** (PLAN): the manifest builder recomputes the keys from the frozen ledger. It must reproduce
  all of the following byte-for-byte:
  - both hashes above;
  - the counts above;
  - the nested-prefix property.
  - It must also assert that no two users in U share a selection key; if they do, the census tie rule at COHORT L4
    could differ.
  - Any mismatch is BLOCK.
- **C2.6 Main-cohort per-client distribution** (MEASURED, COHORT L3886-L3897; linear quantile method,
  CLIENT_CENSUS L4):

  | min | p10 | p25 | p50 | p75 | p90 | p95 | p99 | p99.9 | max |
  |---|---|---|---|---|---|---|---|---|---|
  | 1 | 1 | 2 | 6 | 15 | 35 | 57 | 137.57000000000698 | 353 | 3014 |

- **C2.7 No outcomes used** (PKG 04 §3 L22):
  - Selection uses only user_id and TRAIN eligibility. No VALIDATION or TEST outcome is used.
  - Selected users with no later query stay in the manifest: 199,776 of them in the main cohort.
- **C2.8 Cohort receipts — resolved by QA-01** (FABLE B4 L71; COHORT_RECEIPTS):
  - QA-01 recomputed the universe independently as 2,694,998, matching the census (L9-L11).
  - The 65,536 and 262,144 key hashes both match the census (L18-L20, L39-L41).

  | Quantity | N = 262,144 | N = 65,536 |
  |---|---|---|
  | VALIDATION eval rows: rankable / OOV / total | 801,762 / 6,848 / 808,610 (L45-L47) | 199,276 / 1,709 / 200,985 (L24-L26) |
  | classes (of 158,486) with ≥1 cohort-TRAIN positive | 133,154 (L51) | 90,545 (L30) |
  | classes with ≥5 cohort-TRAIN positives | 72,682 (L52) | 31,425 (L31) |
  | classes that are VALIDATION targets but have zero cohort-TRAIN positives | 5,158 (L54) | 7,619 (L33) |

  - Every cohort member is warm by construction, so all of the cohort's VALIDATION eval rows are warm rows (L23, L44).
- The contest on the cohort, rather than on ALL, is Fable's decision (FABLE D1 L16; risks L80).

## 3. Cold strata and cold panels

- **C3.1 Cold definition** (MEASURED; FABLE D1 L17):
  - "Cold" means a VALIDATION user with ≥1 eval row and **zero eligible TRAIN examples**: 381,295 users
    (CLIENT_CENSUS L3971).
  - For scale: of the 1,023,523 VALIDATION users with ≥1 eval row (L3969), 642,228 are warm (L3970).
  - Cold is not the same as "no TRAIN events": 1,509,381 TRAIN users have zero eligible examples (L3967).
- **C3.2 Strata** (definitions per COLD_STRATA L4-L6; counts MEASURED by QA-01):

  | Stratum | Definition | VALIDATION users | VALIDATION eval rows |
  |---|---|---|---|
  | **COLD_NO_TRAIN_EVENTS** | zero decision-ledger rows with split = TRAIN (L4). Equivalently, absent from `mappings/clients.parquet`, whose row count equals the distinct TRAIN users (README L82-L87) | 299,035 (COLD_STRATA L15) | 2,270,516 (L16) |
  | **COLD_TRAIN_EVENTS_NO_LABEL** | ≥1 TRAIN ledger row but zero eligible examples (L5): every TRAIN target is unobserved, boundary-censored or OOV | 82,260 (L11) | 587,149 (L12) |

  - Consistency check: 299,035 + 82,260 = 381,295 users (CLIENT_CENSUS L3971).
  - COLD_STRATA L19-L28 also holds TEST stratum counts. They exist but are not used here.
- **C3.3 COLD_PANEL_VAL (M03):**
  - All 381,295 cold users.
  - Evaluated as a secondary metric for every method (FABLE D1 L17).
  - It is not part of the matched warm block.
- **C3.4 COLD_H_PANEL (M04) = candidate A** (binding; `V4/company/reviews/G0B_PROTOCOL_FABLE_POSITION.md` C4 L20,
  LOCK L34; FABLE D1 L17; FABLE LOCK L56):
  - the 16,384 COLD_PANEL_VAL users with the smallest C2.2 population keys; all of them are cold;
  - selection is independent of activity and outcomes;
  - the stratum mix and the FIXED_50 / FIXED_20 / ALL_AVAILABLE(h) eligibility are reported at build and never used
    to select (PKG 09 §5 L25);
  - the panel is not stratified;
  - INFERENCE, asserted at build: U and the cold users are disjoint (U needs ≥1 eligible example), so reusing the salt
    creates no overlap;
  - its `selected_keys_sha256_hex` is computed at build.
  - LIMITATION (G0B risks L42): FIXED_50 support will be small, so ALL_AVAILABLE and FIXED_20 carry the study.
- **C3.4b WARM_TRUNCATION panel (M04W)** (binding; G0B C4 L20, LOCK L34):
  - the 16,384 users with the smallest C2.2 population keys among M02's 62,368 E2E users (FABLE D8 L49); nested, with
    no new salt;
  - for T41 and the warm-p_u evaluation of PKG 09 §5 L27;
  - always labelled WARM_TRUNCATION; never pooled with or reported as cold start;
  - its members are cohort (M01) clients, whose parameters and any p_u were trained on their TRAIN history;
  - INFERENCE: it is not the 16,384 cohort prefix (COHORT L1584), which includes users without a VALIDATION query.
- **QA-01's candidate B is historical only** (G0B C4 L20; COLD_H_PANEL L3-L16):
  - it drew all VALIDATION users with ≥50 VALIDATION decision rows, with salt `PPSI_V4_COLD_H_20260926`;
  - 45,572 qualified and 16,384 were selected (keys sha256
    `1ce796f115cf347199e83736f84ff679a67897898eaf7fa2b7bfa81a88bfed91`), 13,767 warm / 2,617 cold;
  - the receipt is kept as history and is not a V4 panel.
- The client key `client_key_hex` (C1.2) is the same in every panel.
- **C3.5 FEW_SHOT recipe-selection panel (M04S)** (binding; G0B C6 L24, LOCK L34):
  - the M03 users at population-key ranks 16,384–32,767 (0-based), disjoint from M04;
  - the recipe grid is registered at ARCHITECTURE_LOCK and selected once on M04S FIXED_20, before any M04 outcome
    (EVALUATION_CONTRACT_V4 E11.9a).
- **C3.6** INFERENCE: warm VALIDATION users outside the cohort belong to no V4 comparison block (FABLE D1; a disclosed
  risk, FABLE risks L84). M04W does not change this, because its users are cohort clients.

## 4. Zero-label clients

- **C4.1 Definition** (PLAN): a client with zero admissible supervised examples in the data it may train on, whether
  its block's TRAIN manifest or, in a few-shot study, its matured support.
- **C4.2 None in the warm cohort** (COHORT L3; min examples 1.0 at L3887): every U user has ≥1 eligible example, so the
  main cohort has no zero-label clients. The builder asserts this.
- **C4.3 Where they occur:**
  - every M03 and M04 user, with respect to TRAIN;
  - FEW_SHOT at h ≤ 1 (T40 L51), or at any h with no matured support.
- **C4.4 Rule** (PKG 04 §5 L31; T24 L33; T29 L38):
  - **No update:** no local step, no message, no optimizer-state creation.
  - The aggregation weight is 0, and aggregation rejects an empty total weight instead of dividing by it.
  - p_u stays 0.
  - The client is **still evaluated** whenever the query manifest includes it.
  - The population receipt reports the count as `no_label_users`.
- **C4.5 Zero-query clients** (PKG 04 §3 L22; `evaluator.py:L98-L100`):
  - These are cohort clients with 0 VALIDATION eval rows: 199,776 of 262,144 (COHORT L6060-L6061).
  - They train and send updates as usual.
  - They have no E2E row, so no macro weight.
  - They are kept, never dropped.

## 5. Per-client data ownership and the simulator trust boundary

- **C5.1 Own rows only** (T23 L32; PKG 04 §3 L19):
  - A client loader is constructed from M01 with one `client_key_hex`.
  - It can read only that client's contiguous row range; M01 is ordered by `(client_ord, decision_id)`.
  - Any read outside the range raises. Buckets are storage only.
- **C5.2 Own history only** (T14 L19; LOCALITY_MAP L15): LOCAL_HISTORY features come only from the client's own
  strictly-earlier events. There is no cross-client history contamination.
- **C5.3 Shared quantities:** these come only from the frozen I_META_RANDOM snapshot (§6), never from other clients' rows
  at run time.
- **C5.4 Private state** (T28 L37):
  - Private state (p_u, and personal or LO optimizer state) is keyed by `client_key_hex` or `client_ord`.
  - It never enters a server payload.
  - Two clients on one worker cannot swap state.
- **C5.5 Evaluation reads:** a client's evaluation reads only its own rows of the registered manifest. The shared
  evaluator aggregates.
- **C5.6 Trust-boundary disclosure** (LIMITATION):
  - **(a) One orchestrator.** All clients are simulated on one server by one orchestrator that can read the whole frozen
    dataset. Ownership is a software access discipline, tested by T23 and T28. It is not physical, cryptographic or
    differential-privacy isolation.
  - **(b) No privacy mechanism.** There is no secure aggregation and no differential privacy, and the server sees every
    client update.
  - **(c) Central feature pipeline.** The builder, vocabularies, normalizers, metadata recovery, item age and catalogue
    were all computed centrally over pooled TRAIN (LOCALITY_MAP L140-L187). They are pooled bootstrap information
    (PKG 03 §8 L46).
  - **(d) Central selection.** The population itself was selected centrally.
  - **(e) Tests are not privacy.** Passing the tests does not establish data privacy (PKG 12 L65).
  - Simulated FL results are therefore comparisons of parameter learning under a shared information block, not privacy
    claims.

## 6. Shared bootstrap information block I_META_RANDOM (PKG 04 §4 L26)

**Definition:**
- The same untrained model initialization per seed.
- The same declared TRAIN-derived catalogue, normalizers and permitted item metadata.
- It is given identically to C, FA, FP, PF, CPF and LO (PKG 03 §8 L47).
- It "tests decentralized parameter learning CONDITIONAL on common metadata, not zero historical global information"
  (PKG 04 §4 L26).
- In Clean V2, 34 of the 53 kept channels depend on other users' pooled interactions (SCHEMA_V2 L105, L115; schema
  version 2.1.0, L3). In v1 it was 36 of 59 (LOCALITY_MAP L31).

| # | Item | Pooled over | Source |
|---|---|---|---|
| 1 | Per-seed untrained initialization, seeds 2026 and 2027. If SASRec wins, the registered per-seed θ0 is the **recentered** init for every method, C and LO included, so the init hash is one tensor (G0B C7 L26, LOCK L35) | nothing (random) | PKG 04 §4 L26; FABLE LOCK CENTRAL L61 |
| 2 | Output catalogue: K = 158,486, support ≥ 5. The main cohort's TRAIN has ≥1 positive for 133,154 of these classes (COHORT_RECEIPTS L51); the rest are ranked but never trained on by any cohort client | **all** TRAIN users (full TRAIN, not only the cohort) | FABLE D2 L19-L21; CATALOGUE_SUPPORT L7-L13, L68; LOCALITY_MAP L121, L154-L160 |
| 3 | Input vocabularies: item table of K+3 rows (PAD/MISSING/OOV); category; brand; main category; category_code (stored, not batched) | all TRAIN representatives | LOCALITY_MAP L125-L128, L154-L157; DATA_CONTRACT L194-L196; FABLE LOCK CATALOGUE L57 |
| 4 | Normalizers for the **17 Clean V2 normalized channels** (`normalized_channels_v2`, SCHEMA_V2 L540-L558): "14 with unchanged v1 normalizers + 2 M1 valid-only refits + 1 F-1 valid-only refit" (L110-L111). The M1 refits of the two time-gap channels are at L556-L557. The **F-1** valid-only refit of `current_minus_recent_mean_log_price` is at L555: mean 0.002065693626402541, std 0.9945344391661521, n_valid 48,434,230, recorded clip bounds (OI03 L26-L31). The V2 re-zero map (`rezero_map_v2`, L562-L571) includes the F-1 extension (L567) | pooled full-TRAIN rows | G0B C1(b) L14, C2 L16; FABLE D4 L29; LOCALITY_MAP L147-L153, L189-L212 |
| 5 | Brand / category / category_code recovery, as a frozen end-of-TRAIN snapshot for future partitions | per product across all TRAIN users; the category fallback spans products (≥ 3) | LOCALITY_MAP L161-L171, L221-L243 |
| 6 | Item-observed-age anchor `first_seen_train` | MIN over all TRAIN rows | LOCALITY_MAP L172-L177, L245-L258 |
| 7 | Feature contract (columns, order, units): Clean V2, 53 channels (numeric 15 / flags 15 / uctx 12 / masks 4; 6 DROP_REDUNDANT, 2 RE_ENCODE_M1), per G0B LOCK L31 and SCHEMA_V2 L100-L113 (schema version 2.1.0, self-hash `55a244e1608bcdcdf6f1728595f534871b52154faebf2977e11ac1fb487052f9`, L3-L4). QA-01 found `category_code_missing` ≠ `main_category_missing` on 0 of 52,760,314 TRAIN rows and 0 of 14,357,645 VALIDATION rows (CODE_FLAGS L65-L86). Whether that makes one flag redundant is a DATA decision under PKG 03 §7, not decided here | protocol decision | FABLE D4 L27-L29; FABLE B5 L72 |
| 8 | Contextual timezone +3h, a hypothesis constant | nothing (declared constant) | LOCALITY_MAP L260-L270; PKG 03 §5 L28 |
| 9 | Cohort and panel definitions, whose eligibility depends on item 2 | pooled TRAIN eligibility | COHORT L3 |
| 10 | Recipe information: central LR calibration on cohort VALIDATION and the derived local-LR grid {peak_C, 0.5 × peak_C} | central runs on pooled cohort data | FABLE D6 L37; FABLE D7 L43-L44 |

**Not part of the block:**
- Centrally trained winner weights never initialize FL or LO (PKG 04 §4 L27-L28).
- I_COMMON_PRIOR is not used; it is optional and none is registered (PKG 04 §4 L27).
- The client_idx vocabulary is not consumed as a feature or identity by any V4 method (C1.3).

**Snapshot hash:** `I_META_RANDOM_sha256` covers items 2-8. PENDING (DATA_PROTOCOL_LOCK).

## 7. Population lock receipt (PKG 04 §5 L31)

| Required field | Value | Source / status |
|---|---|---|
| client count | 262,144 | COHORT L3884 |
| eligible-example total | 3,906,929 | COHORT L3885 |
| per-client min / median / quantiles / max | 1 / 6 / §2 C2.6 table / 3014 | COHORT L3886-L3897 |
| label-time coverage | full TRAIN: 0 loss rows with target_ts ≥ TRAIN_END (LABEL_TIME_AUDIT L11-L14). Cohort: 0 by the eligibility definition (target_ts < TRAIN_END; COHORT_RECEIPTS L3, L6), re-asserted row by row in M01 | this file; builder |
| VALIDATION label coverage (cohort) | 801,762 rankable / 6,848 OOV of 808,610 | COHORT_RECEIPTS L45-L47 |
| no-label users | 0 by construction (C4.2); asserted | builder |
| candidate K | 158,486; 133,154 classes have ≥1 cohort-TRAIN positive | FABLE LOCK CATALOGUE L57; COHORT_RECEIPTS L51 |
| per-block query manifests | M02, M03, M04 (candidate A), M04W, M04S and M06 hashes | PENDING_RECEIPT (build) |
| bootstrap source | I_META_RANDOM (§6) | this file |
| expected dense payloads | — | PENDING (FL/SYS owner) |

## 8. PENDING (client side)

| ID | Item | Owner |
|---|---|---|
| CP1 | Cold-strata counts; cohort s=5 coverage; classes with ≥1 cohort positive (FABLE B4 L71) | **RESOLVED in R1** (C2.8, C3.2) |
| CP2 | COLD_H_PANEL definition (C3.4) | **RESOLVED by G0(b)** C4 L20: candidate A. The M04 and M04W `selected_keys_sha256_hex` are computed at build (PENDING_RECEIPT) |
| CP3 | FEW_SHOT recipe-selection panel (C3.5) | **RESOLVED by G0(b)** C6 L24: M04S. The grid values are registered at ARCHITECTURE_LOCK |
| CP6 | F-1 values written into the schema (C1(d)) before the I_META_RANDOM hash is taken | **RESOLVED** in schema version 2.1.0 (SCHEMA_V2 L3-L4, L555, L567) |
| CP4 | I_META_RANDOM snapshot hash | DATA_PROTOCOL_LOCK |
| CP5 | Expected dense payload bytes per client and round | FL/SYS owner |
