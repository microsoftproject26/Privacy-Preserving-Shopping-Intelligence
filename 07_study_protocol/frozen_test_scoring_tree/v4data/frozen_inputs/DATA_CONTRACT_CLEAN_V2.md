# DATA_CONTRACT_CLEAN_V2 — Clean V2 feature and transform contract (design + specification)

- **Task:** DATA-03 (Data Lead), revised by **DATA-03R1**. **Date:** 2026-09-26.
- **Status:** Fable's G0(b) fixes are applied and the contract is ready for DATA_PROTOCOL_LOCK. Fable's verdict was
  APPROVE_WITH_FIXES (`company/reviews/G0B_PROTOCOL_FABLE_POSITION.md:7`). Nothing here has been trained or evaluated.
- **Companion (machine-readable, binding when locked):** `contracts/FEATURE_SCHEMA_CLEAN_V2.json`.
  - Schema version: **2.1.0**.
  - Self-hash: `schema_self_sha256 = 55a244e1608bcdcdf6f1728595f534871b52154faebf2977e11ac1fb487052f9`.
  - How the self-hash is computed: sha256 over the UTF-8 bytes of
    `json.dumps(obj, sort_keys=True, ensure_ascii=True, separators=(',', ':'), allow_nan=False)`, where `obj` is the
    parsed schema with only the top-level key `schema_self_sha256` removed.
- **Authority:**
  - package `03_DATA_PROTOCOL.md` §5–§9 (`PKG/03_DATA_PROTOCOL.md:26-51`);
  - Fable G0 decision 4 (`company/reviews/G0_PLAN_FABLE_POSITION.md:27-29`, LOCK_VALUES `:59`);
  - Fable G0(b) C1, C2, C8 and the LOCK_VALUES delta (`company/reviews/G0B_PROTOCOL_FABLE_POSITION.md:11-16,28,31`),
    which are binding.
- **Builder:** DATA-04 (later task). This document is the spec that DATA-04 implements and tests.
- **QA-01 receipts** (`g0/receipts_qa01/CODE_AND_FLAG_CHECKS.json`) arrived while this contract was being written and
  are incorporated:
  - **5a:** `code_hash()` recomputed on the live tree equals the frozen `02fbb071…`. The 49 covered files are therefore
    byte-identical to the freeze, and every `EDA/*.py` citation below is to the frozen build's code.
  - **5b:** `category_code_missing ≠ main_category_missing` on 0 rows in TRAIN and 0 rows in merged VALIDATION.

**Path roots.**
- `EDA` = `next_item_eda/src/rees46_eda/`
- `V1` = `rees46_nextitem_v1/`
- `FOI` = `.local_ai/feature_opt_v1/final_inputs/`
- `FO` = `.local_ai/feature_opt_v1/`
- `SNAP` = `.local_ai/ppsi_v4/code/snap_v1/`
- `PKG` = `PPSI_SINGLE_BACKBONE_V4_2026_09_26/PPSI_SINGLE_BACKBONE_V4_2026_09_26/`
- `MAP` = `.local_ai/ppsi_v4/g0/feature_map/FEATURE_LOCALITY_MAP.md`
- `DUP` = `FOI/audit/DUPLICATES_FULL.json`
- `M1FIT` = `FOI/audit/M1_FIT.json`

All paths are relative to `F:\Queens University\Microsoft Project\eda\`.

**Evidence labels.**
- **MEASURED:** quoted from an existing measurement file (DUP, M1FIT, normalizers.json), or a file hash taken while
  writing this contract.
- **CODE:** follows from the cited source lines by reading. Nothing was executed.
- **DERIVED:** arithmetic on quoted numbers only. No data was read.
- **INFERENCE:** reasoned, not measured.
- **PENDING:** a required receipt is not present yet.

No new data statistic was computed for this contract.

---

## 1. Binding principles

1. **v1 is immutable.** `rees46_nextitem_v1` has `version_hash 860e5a1c…` (`V1/VERSION.json:3`). Clean V2 is a
   *load-time transform sidecar* over the v1 `model_ready` features (`PKG/03_DATA_PROTOCOL.md:50`).
2. **No new per-decision copy of the data** is written. DATA-04 writes only a TRANSFORM_STATE json (a few KB) and test
   receipts.
3. **From-scratch topology.** Input widths change and every model is initialised anew. There is no weight transfer, no
   fold, and no claim of equivalence with v1 or lean weights (`PKG/03_DATA_PROTOCOL.md:43`).
4. **Dropped channels stay in v1 storage**, and in the v1 batch before the view is applied. They are left out only of
   the V2 neural tensors (`PKG/03_DATA_PROTOCOL.md:42`).
5. **Every fitted object is a SHARED_TRAIN_STATISTIC.** Each is fitted on pooled full TRAIN and is byte-identical
   across C/FA/FP/PF/CPF/LO. There is no per-client refit (`PKG/03_DATA_PROTOCOL.md:38`).
6. **A numeric unknown is written as exactly 0.0 after normalization,** and its governing flag or mask always travels
   with it (`PKG/03_DATA_PROTOCOL.md:35`; `V1/FEATURE_DICTIONARY.md:73-74`).
7. **Removal requires the (a)–(d) proof** (`PKG/03_DATA_PROTOCOL.md:41-42`):
   - (a) the exact meaning;
   - (b) an expression that rebuilds the column from kept inputs;
   - (c) a full-TRAIN contradiction count of 0, plus synthetic missing/OOV/edge cases;
   - (d) no loss of useful future availability semantics.

   A flag that is all-zero only empirically stays in.

## 2. Decision summary

The 59 channels are the rich set from DATA-02 (`MAP:20-29`): 6 tokens, 19 numeric, 15 flags, 12 user-context values,
6 masks and the output catalogue.

| decision | count | channels |
|---|---|---|
| KEEP | 51 | all others |
| DROP_REDUNDANT | 6 | `local_hour_sin`, `local_hour_cos`, `prefix_item_count`, `log1p_prefix_elapsed_s`, `has_history_mask`, `prev_user_event_available` |
| DROP_SPEC_DEAD | 0 | none. No flag's impossibility follows from the spec or code that was read (§4.6) |
| RE_ENCODE_M1 | 2 | `log1p_time_since_prev_user_event_s`, `log1p_time_since_prev_session_last_event_s` |

In total, **53 channels are kept.**

The **final Clean V2 widths** are:
- numeric 15 (v1: 19)
- flags 15 (v1: 15)
- user context 12 (v1: 12)
- masks 4 (v1: 6)
- plus 6 token channels and the output catalogue: 15+15+12+4+6+1 = **53 channels**

These match Fable's LOCK_VALUES: "Clean V2 = 53 channels (numeric 15 / flags 15 / uctx 12 / masks 4); 6
DROP_REDUNDANT, 0 DROP_SPEC_DEAD, 2 RE_ENCODE_M1, F-1 re-zero + refit" (`G0B_PROTOCOL_FABLE_POSITION.md:31`).

**F-1 was ADOPTED inside scope (Fable G0(b) C2, `G0B_PROTOCOL_FABLE_POSITION.md:16`).** For
`current_minus_recent_mean_log_price` (§6):
- **Re-zero gate:** the value is written as 0.0 when `recent_price_supported=0 OR price_missing OR price_conflict` at
  `last_valid_index`.
- **Refit:** its normalizer is refit on valid rows only, keeping the recorded clip bounds.
- **Refit values:** mean 0.002065693626402541, population std 0.9945344391661521, n_valid 48,434,230
  (`g0/receipts_qa02/OI03_F1_PRICE_REZERO.json:23-32`).
- **Label:** the channel stays labelled KEEP with F-1. It is not counted as RE_ENCODE_M1.

Clean V2 is **not the lean view** (§11). It keeps 8 columns the lean view dropped:
- `price_zero`;
- `main_category_missing`;
- six flags that are all-zero in TRAIN.

Those 8 columns were dropped on empirical evidence only.

## 3. Per-channel (a)–(d) table (59 rows)

The locality labels are DATA-02's (`MAP` line cited). A script check found that all 59 labels and pooled-dependency
bits match `g0/feature_map/FEATURE_LOCALITY_MAP.json` row by row. For the six removals, the full (b)/(c)/(d) proofs
are in §4.

### A. Categorical tokens `[B,50]` int64 (6) — all KEEP, V2 slots unchanged

| # | channel | (a) exact meaning | decision / basis | (d) availability | locality |
|---|---|---|---|---|---|
| 1 | `item_tokens` (product_idx) | Catalogue index of the representative's raw STRING product_id. PAD=0; OOV=2 when outside the support≥5 catalogue; MISSING=1 is never valid (`EDA/pipeline6.py:320`; `V1/FEATURE_DICTIONARY.md:10-15,25`) | KEEP; the primary sequence input (`EDA/loaders.py:419-420`) | OOV retained | SHARED_TRAIN_STATISTIC (`MAP:38`) |
| 2 | `category_tokens` | Vocab index of `category_id_resolved`: as-of in TRAIN, frozen snapshot in the future. NULL→1, unseen→2 (`EDA/pipeline6.py:321-322`; `EDA/metadata.py:217-224`) | KEEP; no reconstruction | MISSING/OOV retained | SHARED_CATALOGUE (`MAP:39`) |
| 3 | `brand_tokens` | Vocab index of `brand_resolved` (`EDA/pipeline6.py:323-324`; `EDA/metadata.py:207-212`) | KEEP | retained | SHARED_CATALOGUE (`MAP:40`) |
| 4 | `main_category_tokens` | First dot-segment of `category_code_resolved`. An empty segment counts as NULL, and NULL→1 (`EDA/pipeline6.py:69,74,112,325-326`) | KEEP; `category_code_token` is not batched (`MAP:125-130`) | retained | SHARED_CATALOGUE (`MAP:41`) |
| 5 | `daypart_tokens` | Closed daypart band of the local (UTC+3h) hour, NIGHT=3 … LATE_EVENING=7 (`EDA/pipeline6.py:53-58,146,347`) | KEEP; a lossy, non-affine function of the hour with no full-population count (`FO/audit/REDUNDANCY_FINDINGS.md:121-132`) | closed vocab | LOCAL_EVENT (`MAP:42`) |
| 6 | `weekday_tokens` | Local weekday Mon=3 … Sun=9, taken after the shift (`EDA/pipeline6.py:59-62,145,348`) | KEEP; the token path, no count against sin/cos | closed vocab | LOCAL_EVENT (`MAP:43`) |

### B. `event_numeric_features` `[B,50,19]` → `[B,50,15]` float32

| # | channel (v1→V2 idx) | (a) exact meaning | decision / basis | (d) availability | locality |
|---|---|---|---|---|---|
| 7 | `log1p_price` (0→0) | ln(1+price) for a finite, strictly positive block price. Stored as 0.0 when unavailable (`EDA/pipeline6.py:114-118,330`) | KEEP | price_missing or price_conflict → 0 after z (`EDA/loaders.py:284-296`) | LOCAL_EVENT (`MAP:49`) |
| 8 | `utc_hour_sin` (1→1) | sin(2π(h+m/60+s/3600)/24), UTC (`EDA/pipeline6.py:119-121`) | KEEP; keeper for the rotation | never missing | LOCAL_EVENT (`MAP:50`) |
| 9 | `utc_hour_cos` (2→2) | cos(…), UTC (`EDA/pipeline6.py:122-124`) | KEEP; keeper for the rotation | never missing | LOCAL_EVENT (`MAP:51`) |
| 10 | `utc_weekday_sin` (3→3) | sin(2π(isodow−1)/7), UTC (`EDA/pipeline6.py:125`) | KEEP | never missing | LOCAL_EVENT (`MAP:52`) |
| 11 | `utc_weekday_cos` (4→4) | cos(…) (`EDA/pipeline6.py:126`) | KEEP | never missing | LOCAL_EVENT (`MAP:53`) |
| 12 | `local_hour_sin` (5→—) | Hour angle on the clock `block_ts + INTERVAL 3 HOUR` (`EDA/pipeline6.py:127-134`; `EDA/config.py:235,253-255`) | **DROP_REDUNDANT**: (utc_sin+utc_cos)/√2; max residual 6.544e-08 in TRAIN and VS (`DUP:11,26`); §4.3 | never missing; exactly recoverable | LOCAL_EVENT (`MAP:54`) |
| 13 | `local_hour_cos` (6→—) | Same clock (`EDA/pipeline6.py:135-137`) | **DROP_REDUNDANT**: (utc_cos−utc_sin)/√2; §4.3 | never missing | LOCAL_EVENT (`MAP:55`) |
| 14 | `local_weekday_sin` (7→5) | sin(2π(isodow(UTC+3h)−1)/7) (`EDA/pipeline6.py:138-140`) | KEEP; differs from UTC on rows with UTC hour 21–24; non-affine; sample check only (`FO/audit/REDUNDANCY_FINDINGS.md:95-119`; `FO/audit/MEASURED_FINDINGS.md:54-55`) | never missing | LOCAL_EVENT (`MAP:56`) |
| 15 | `local_weekday_cos` (8→6) | cos(…) (`EDA/pipeline6.py:141`) | KEEP (as #14) | never missing | LOCAL_EVENT (`MAP:57`) |
| 16 | `is_weekend` (9→7) | 1 when isodow(local) ≥ 6 (`EDA/pipeline6.py:142-143`) | KEEP; lossy, and crosses between the token and numeric paths | never missing | LOCAL_EVENT (`MAP:58`) |
| 17 | `log1p_prev_raw_view_gap_s` (10→8) | ln(1+max(0, s since the user's previous VALID raw view block, before compression)). Stored as 0.0 when there is none (`EDA/pipeline6.py:147-149,349`) | KEEP | prev_gap_unavailable → 0 (`EDA/loaders.py:57,297-298`); fitted on valid rows only (`EDA/pipeline6.py:401`) | LOCAL_HISTORY (`MAP:59`) |
| 18 | `log1p_segment_elapsed_s` (11→9) | ln(1+max(0, s since the model-segment start)) (`EDA/pipeline6.py:150,350`) | KEEP; keeper for #25 | never missing | LOCAL_HISTORY (`MAP:60`) |
| 19 | `log1p_item_observed_age_s` (12→10) | ln(1+s since `first_seen_train`) when first_seen < block_ts, else stored as 0.0. first_seen is taken from pooled TRAIN and frozen (`EDA/pipeline6.py:151-152,351`; `MAP:245-258`) | KEEP | item_age_unavailable → 0 (`EDA/loaders.py:58`) | SHARED_TRAIN_STATISTIC (`MAP:61`) |
| 20 | `prefix_representative_count` (13→11) | position_in_segment − 1 (`EDA/pipeline6.py:153`) | KEEP; keeper for #21 | never missing | LOCAL_HISTORY (`MAP:62`) |
| 21 | `prefix_item_count` (14→—) | position_in_segment (`EDA/pipeline6.py:155`) | **DROP_REDUNDANT**: raw = #20 + 1; the z value is identical; 0 contradictions (`DUP:10,25`); §4.2 | never missing | LOCAL_HISTORY (`MAP:63`) |
| 22 | `prefix_known_category_count` (15→12) | Number of distinct KNOWN main categories in the segment prefix (`EDA/pipeline6.py:92-94,100,156`) | KEEP; 0 is a defined count, not M1 (`g0/feature_map/OPEN_ITEMS.md:41-49`); z(0)=−1.3206 by design | main_category_missing travels | LOCAL_HISTORY (`MAP:64`) |
| 23 | `prefix_category_switch_count` (16→13) | Running count of CHANGED transitions; UNKNOWN transitions are not counted (`EDA/pipeline6.py:85-88,99,157`) | KEEP; z(0)=−0.2583 by design | — | LOCAL_HISTORY (`MAP:65`) |
| 24 | `current_known_category_run_length` (17→14) | Length of the current run of the same known main category; 0 when the current category is unknown (`EDA/pipeline6.py:89-91,101-103,158`) | KEEP; the value 0 is defined by the dictionary (`V1/FEATURE_DICTIONARY.md:47`) and lies outside the M1 decision; z(0)=−0.5278 is disclosed | 0 exactly when main_category_missing=1 (CODE: `:101-103` with `:179`) | LOCAL_HISTORY (`MAP:66`) |
| 25 | `log1p_prefix_elapsed_s` (18→—) | Same SQL text as #18 (`EDA/pipeline6.py:159` vs `:150`) | **DROP_REDUNDANT**: equals #18; 0 contradictions (`DUP:9,24`); §4.1 | never missing | LOCAL_HISTORY (`MAP:67`) |

### C. `event_quality_flags` `[B,50,15]` int8 — all KEEP, V2 slots unchanged

| # | flag (idx) | (a) exact meaning | decision / basis | locality |
|---|---|---|---|---|
| 26 | `price_missing` (0) | `price_obs IS NULL AND NOT price_block_conflict`: the block has no finite, positive price. Covers NULL, negative, NaN, unparseable and zero prices (`EDA/pipeline6.py:161-166`) | KEEP; governs the log1p_price re-zero | LOCAL_EVENT (`MAP:73`) |
| 27 | `price_zero` (1) | A literal zero price was OBSERVED in the block (`EDA/pipeline6.py:167-169`) | KEEP. It equals price_missing empirically (0 contradictions, `DUP:12,27`), but the spec allows the two to diverge: a NULL, negative or non-finite source price gives price_missing=1 with price_zero=0 (`EDA/pipeline6.py:161-169`). Empirical equality is not enough (`PKG/03_DATA_PROTOCOL.md:41`) | LOCAL_EVENT (`MAP:74`) |
| 28 | `price_conflict` (2) | The block's eligible prices conflict (`EDA/pipeline6.py:170`) | KEEP, **DEAD_IN_TRAIN** (0 non-zero, `DUP:16,31`); no proof that it is impossible; also a re-zero trigger (`EDA/loaders.py:287`) | LOCAL_EVENT (`MAP:75`) |
| 29 | `brand_missing` (3) | `brand_resolved IS NULL` (`EDA/pipeline6.py:171`) | KEEP. It equals `brand_tokens==1` by code (`:171` vs `:323`), but that identity crosses from the token path to the flag path and has no full-population count | LOCAL_EVENT (`MAP:76`) |
| 30 | `brand_recovered` (4) | brand_source = PRODUCT_EXACT_ASOF (`EDA/pipeline6.py:172`; `EDA/metadata.py:159-169,207-212`) | KEEP | LOCAL_EVENT (`MAP:77`) |
| 31 | `brand_conflict` (5) | CONFLICT_ABSTAIN (recovery abstained on a visible contradiction) or a block brand conflict (`EDA/pipeline6.py:173`; `EDA/metadata.py:172-173,195-197,210-212`) | KEEP, **DEAD_IN_TRAIN** (`DUP:17,32`); the state machine can emit it | LOCAL_EVENT (`MAP:78`) |
| 32 | `category_missing` (6) | `category_id_resolved IS NULL` (`EDA/pipeline6.py:174`) | KEEP, **DEAD_IN_TRAIN** (`DUP:18,33`); NULL→MISSING is in the spec (`V1/FEATURE_DICTIONARY.md:26`) | LOCAL_EVENT (`MAP:79`) |
| 33 | `category_recovered` (7) | category_id_source = PRODUCT_EXACT_ASOF (`EDA/pipeline6.py:175`; `EDA/metadata.py:217-221`) | KEEP, **DEAD_IN_TRAIN** (`DUP:19,34`); reachable in code when its family is enabled (OI-08) | LOCAL_EVENT (`MAP:80`) |
| 34 | `category_conflict` (8) | CONFLICT_ABSTAIN or a block category conflict (`EDA/pipeline6.py:176`) | KEEP, **DEAD_IN_TRAIN** (`DUP:20,35`) | LOCAL_EVENT (`MAP:81`) |
| 35 | `category_code_missing` (9) | `category_code_resolved IS NULL` (`EDA/pipeline6.py:177`) | KEEP; cannot be rebuilt from batched inputs | LOCAL_EVENT (`MAP:82`) |
| 36 | `category_code_recovered` (10) | Source ∈ {PRODUCT_EXACT_ASOF, CATEGORY_EXACT_ASOF} (`EDA/pipeline6.py:178`; `EDA/metadata.py:226-233`) | KEEP, **DEAD_IN_TRAIN** (`DUP:21,36`); reachable in code (OI-08) | LOCAL_EVENT (`MAP:83`) |
| 37 | `main_category_missing` (11) | `main_cat IS NULL`, i.e. code NULL **or** an empty first segment (`EDA/pipeline6.py:69,74,179`) | KEEP. It equals category_code_missing empirically (`DUP:13,28`; QA-01 5b: 0 of 52,760,314 TRAIN rows and 0 of 14,357,645 merged-VALIDATION rows, `g0/receipts_qa01/CODE_AND_FLAG_CHECKS.json:65-86`), but the spec lets them diverge on an empty-root code. It equals `main_category_tokens==1` by code, with no count | LOCAL_EVENT (`MAP:84`) |
| 38 | `prev_gap_unavailable` (12) | `prev_valid_view_ts IS NULL` (`EDA/pipeline6.py:180`) | KEEP; governs the gap re-zero; event grain | LOCAL_EVENT (`MAP:85`) |
| 39 | `item_age_unavailable` (13) | first_seen NULL or block_ts ≤ first_seen (`EDA/pipeline6.py:181`) | KEEP; governs the age re-zero | LOCAL_EVENT (`MAP:86`) |
| 40 | `simultaneous_block` (14) | Same-timestamp multi-item ambiguity indicator, passed through from the upstream sequence stage (`EDA/pipeline6.py:113,363`; `PKG/03_DATA_PROTOCOL.md:27`). The upstream definition was not re-read | KEEP | LOCAL_EVENT (`MAP:87`) |

### D. `user_context` `[B,12]` float32 — order unchanged

| # | channel (idx) | (a) exact meaning | decision / basis | (d) availability | locality |
|---|---|---|---|---|---|
| 41 | `log1p_prev_view_count` (0) | ln(1 + number of the user's earlier representatives in the replay stream) (`EDA/userctx.py:176`; `EDA/pipeline6.py:288`) | KEEP; a cold user gets z(0)=−1.8249 | 0 is genuine for a cold user | LOCAL_HISTORY (`MAP:93`) |
| 42 | `log1p_prev_session_count` (1) | ln(1 + number of distinct sessions among earlier events) (`EDA/userctx.py:180-186`; `EDA/pipeline6.py:289`) | KEEP | genuine 0 | LOCAL_HISTORY (`MAP:94`) |
| 43 | `recent_history_length` (2) | min(number of earlier events, 100) (`EDA/userctx.py:178`) | KEEP; equals min(expm1(#41),100) by code, but non-affine with no count | genuine 0 | LOCAL_HISTORY (`MAP:95`) |
| 44 | `has_history` (3) | 1 if the user has an earlier event in the stream (`EDA/userctx.py:177`; `EDA/pipeline6.py:291`) | KEEP; keeper for #53 and #57; never normalized | the history indicator in V2 | LOCAL_HISTORY (`MAP:96`) |
| 45 | `recent_distinct_items` (4) | Exact number of distinct products among the ≤100 most recent earlier representatives (`EDA/userctx.py:188-191`) | KEEP | genuine 0 | LOCAL_HISTORY (`MAP:97`) |
| 46 | `recent_distinct_known_main_categories` (5) | Distinct known main categories in the window (`EDA/userctx.py:192-195`) | KEEP | genuine 0 | LOCAL_HISTORY (`MAP:98`) |
| 47 | `recent_category_match_share` (6) | Share of known recent main categories equal to the current known one; 0 when unsupported (`EDA/userctx.py:197-203`) | KEEP; never scaled | already 0 when the mask is 0 (asserted, RC7) | LOCAL_HISTORY (`MAP:99`) |
| 48 | `recent_brand_match_share` (7) | The same for brands (`EDA/userctx.py:205-210`) | KEEP | already 0 (RC7) | LOCAL_HISTORY (`MAP:100`) |
| 49 | `recent_mean_log_price` (8) | Mean log1p price over valid recent prices; 0 when there are none (`EDA/userctx.py:212-220`) | KEEP; valid-only fit (`EDA/pipeline6.py:403`) | recent_price_supported=0 → 0 (`EDA/loaders.py:61`) | LOCAL_HISTORY (`MAP:101`) |
| 50 | `current_minus_recent_mean_log_price` (9) | log1p(current) − recent mean when **both** are available, else stored as 0 (`EDA/userctx.py:221-222`; `EDA/pipeline6.py:213`) | KEEP with **F-1 (ADOPTED)**: re-zero gate extended plus valid-only refit (§6; `G0B_PROTOCOL_FABLE_POSITION.md:16`) | unknown when recent unsupported **or** the current price is unknown; 117,315 TRAIN rows affected (`OI03_F1_PRICE_REZERO.json:6-9`) | LOCAL_HISTORY (`MAP:102`) |
| 51 | `log1p_time_since_prev_user_event_s` (10) | ln(1+s since the user's previous representative in the stream); 0 when there is none (`EDA/userctx.py:224-232`; `EDA/pipeline6.py:300-301`) | **RE_ENCODE_M1** (§5) | prev_user_event_available=0 → 0 | LOCAL_HISTORY (`MAP:103`) |
| 52 | `log1p_time_since_prev_session_last_event_s` (11) | At a new session's first event, ln(1+s since the previous event); else 0 (`EDA/userctx.py:233-235`; `EDA/pipeline6.py:302-303`) | **RE_ENCODE_M1** (§5) | prev_session_available=0 → 0 | LOCAL_HISTORY (`MAP:104`) |

### E. `user_context_masks` `[B,6]` → `[B,4]` float32

| # | mask (v1→V2 idx) | (a) exact meaning | decision / basis | locality |
|---|---|---|---|---|
| 53 | `has_history_mask` (0→—) | `coalesce(c.has_history,0.0)`, the same expression as has_history (`EDA/pipeline6.py:291,304`) | **DROP_REDUNDANT**: equals #44; 0 contradictions (`DUP:14,29`); §4.4 | LOCAL_HISTORY (`MAP:110`) |
| 54 | `recent_category_match_supported` (1→0) | Current main category known and at least 1 known category in the window (`EDA/userctx.py:199-203`) | KEEP | LOCAL_HISTORY (`MAP:111`) |
| 55 | `recent_brand_match_supported` (2→1) | The same for brand (`EDA/userctx.py:205-210`) | KEEP | LOCAL_HISTORY (`MAP:112`) |
| 56 | `recent_price_supported` (3→2) | At least 1 valid price in the window (`EDA/userctx.py:213-220`) | KEEP; not a duplicate of has_history (204 TRAIN sample rows differ, `FO/audit/MEASURED_FINDINGS.md:59`) | LOCAL_HISTORY (`MAP:113`) |
| 57 | `prev_user_event_available` (4→—) | A previous same-user event exists in the stream (`EDA/userctx.py:224-229,232`) | **DROP_REDUNDANT**: equals #44; 0 contradictions (`DUP:15,30`); still read from storage as the M1 gate; §4.5 | LOCAL_HISTORY (`MAP:114`) |
| 58 | `prev_session_available` (5→3) | A previous event exists and the current event opens a new session (`EDA/userctx.py:233,235`) | KEEP; the M1 gate | LOCAL_HISTORY (`MAP:115`) |

### F. Output catalogue

| # | channel | (a) exact meaning | decision | locality |
|---|---|---|---|---|
| 59 | output catalogue | `catalogue.parquet` maps product_id → output_class 0..K−1 (K=158,486) → product_idx = class+3. Support ≥ 5 on full TRAIN. −1 means no rankable target; OOV targets score zero (`V1/DATA_CONTRACT.md:3-4,124-135`; `V1/FEATURE_DICTIONARY.md:17-19,64`; `company/reviews/G0_PLAN_FABLE_POSITION.md:19-21,57`) | KEEP; both families gate the class map (`SNAP/sasrec_model.py:137-143`; `SNAP/reuse/model.py:222-237`) | SHARED_TRAIN_STATISTIC (`MAP:121`) |

## 4. Removal proofs and non-removals

The removal proofs rest on these population sizes and scope:
- **Population sizes:** TRAIN has 52,760,314 rows and VALIDATION_SELECT has 10,930,090 (`DUP:3-6`). Both are labelled
  `FULL_POPULATION` (`DUP:2`).
- **Merged VALIDATION, including the VC half:** DUP does not cover the old VALIDATION_CONFIRM. QA-02 ran RC1–RC6 over
  merged VALIDATION, 14,357,645 rows including the 3,427,555 VC rows, and found **0 contradictions for every check**
  (`g0/receipts_qa02/OI06_RC_VALIDATION.json:10-89`). TEST is not measured. Every removal below is also a code
  identity, so it holds by construction on every split, and it is re-checked at runtime (RC1–RC6).
- **Capacity:** every reconstruction below is affine, and it enters the same first `Linear` layer as its keeper:
  - SASRec `numflag_proj` / `user_mlp` (`SNAP/sasrec_model.py:167,172,251-261`);
  - GRU `numeric_flags_proj` / `user_ctx_proj` (`SNAP/reuse/model.py:190-195,207-212`).

  So the V2 input layer can represent every function of the dropped columns exactly. This is a **capacity statement
  only**. It does not claim equal trained accuracy.

### 4.1 `log1p_prefix_elapsed_s` → `log1p_segment_elapsed_s`
- **(a) Meaning.** Both columns use the identical expression `ln(1 + greatest(date_diff('second', segment_start_ts,
  block_ts), 0))` (`EDA/pipeline6.py:150` and `:159`). The dictionary states they are "identical … because prefix
  state resets at the model-segment boundary" (`V1/FEATURE_DICTIONARY.md:48`).
- **(b) Reconstruction.** Raw: prefix = segment. Normalizers (`V1/mappings/normalizers.json:23-29,72-78`): the clip
  bounds (0, 9.561067581176758) and mean 4.548892813156198 are identical. The std is 2.8300548438973165 for segment and
  2.8300548438974196 for prefix. So z_prefix = z_segment·(1 − 3.6e-14) (DERIVED), which is identical in float32.
- **(c) Count and synthetic cases.** 0 contradictions in TRAIN and in VS (`DUP:9,24`). Synthetic cases:
  - segment start (0 s);
  - equal timestamps, and a clock-regression row (the `greatest(…,0)` guard);
  - elapsed time above expm1(9.561) ≈ 14,200 s (DERIVED), where the upper clip binds identically;
  - PAD positions;
  - a future-partition row (the SQL has no split dependence).

  Negative control: a row perturbed by 1e-3 must fail RC1.
- **(d) Availability.** Never missing (`V1/FEATURE_DICTIONARY.md:41,48`); no availability semantics exist.

### 4.2 `prefix_item_count` → `prefix_representative_count`
- **(a) Meaning.** item = `position_in_segment`; rep = `position_in_segment − 1` (`EDA/pipeline6.py:153,155`).
- **(b) Reconstruction.** Raw: item = rep + 1. The clip bounds (1,141) equal (0,140)+1 and the mean 7.716780248881764
  equals 6.716780248881764+1. The std is 11.90825842230758 vs 11.908258422307725 (`V1/mappings/normalizers.json:37-50`).
  Because clip(r+1,1,141) = clip(r,0,140)+1, z_item = z_rep·(1+1.2e-14) (DERIVED): the z-score removes the offset.
- **(c) Count and synthetic cases.** 0 contradictions in TRAIN and in VS (`DUP:10,25`). Synthetic cases:
  - position 1;
  - positions 141 and 142, where the upper clip binds on both;
  - a segment longer than L=50;
  - PAD positions.

  Negative control: item = rep+2 must fail RC2.
- **(d) Availability.** Never missing.

### 4.3 `local_hour_sin/cos` → rotation of `utc_hour_sin/cos`
- **(a) Meaning.** The local clock is `block_ts + INTERVAL 3 HOUR` (`EDA/config.py:235,253-255`), with minutes and
  seconds taken from block_ts (`EDA/pipeline6.py:127-137`).
- **(b) Reconstruction.** hour(local) = (hour(utc)+3) mod 24, so θ_local = θ_utc + π/4 (mod 2π). Therefore:
  - `local_hour_sin = (utc_hour_sin + utc_hour_cos)/√2`
  - `local_hour_cos = (utc_hour_cos − utc_hour_sin)/√2`

  The offset is a code constant, not a timezone database. The identity therefore does not depend on the date
  (INFERENCE from code). It holds only while the contextual clock is a constant whole-hour shift; any DST-aware or
  different local clock must re-derive it.
- **(c) Count and synthetic cases.** The maximum |residual| is 6.544382358253387e-08 in both TRAIN and VS (`DUP:11,26`),
  which is float32 rounding. No row exceeds 1e-6: the maximum is below it, and `FOI/TRANSFORM_STATE.json:61` records
  `rotation_residual_le_1e-6 = true`. Synthetic cases:
  - UTC 20:59:59, 21:00:00, 23:59:59 and 00:00:00 (the local date wrap);
  - PAD positions.

  Negative control: a rotation by 2π·2/24 must fail RC3.
- **(d) Availability.** Never missing. The local clock stays exactly recoverable, and `weekday_tokens`,
  `daypart_tokens` and `local_weekday_sin/cos` all stay.

### 4.4 `has_history_mask` → `has_history`
- **(a) Meaning.** Both columns are `coalesce(c.has_history, 0.0)::FLOAT` (`EDA/pipeline6.py:291,304,311`).
- **(b) Reconstruction.** has_history_mask = has_history.
- **(c) Count and synthetic cases.** 0 contradictions in TRAIN and in VS (`DUP:14,29`). Synthetic cases:
  - a user's first and second events;
  - a decision missing from the context join, where both coalesce to 0 (`EDA/pipeline6.py:370`);
  - the first future-partition event of a user with TRAIN history (`EDA/pipeline6.py:42-47`);
  - restricted-history streams with h=0 and h=1.

  Negative control: flipping one bit must fail RC4.
- **(d) Availability.** Retained bit for bit.

### 4.5 `prev_user_event_available` → `has_history`
- **(a) Meaning.** avail = `prev_ts ≥ 0`, where prev_ts = ts[i−1] if the previous row belongs to the same user, else −1
  (`EDA/userctx.py:224-229`).
- **(b) Reconstruction.** has_history = `pu > 0` (`EDA/userctx.py:177`). On the user-sorted stream
  (`EDA/pipeline6.py:216`; `EDA/userctx.py:36-44`), pu > 0 ⇔ same user as the previous row ⇔ prev_ts = ts[i−1]. That
  value is ≥ 0 because ts is the epoch second of a 2019 timestamp (`EDA/pipeline6.py:203`;
  `PKG/03_DATA_PROTOCOL.md:8-13`). The slow reference sets both columns from the same test `i > start`
  (`EDA/userctx.py:263,296-298`).
- **(c) Count and synthetic cases.** 0 contradictions in TRAIN and in VS (`DUP:15,30`). Synthetic cases: those of 4.4,
  plus:
  - equal timestamps (dt=0 still counts as available);
  - a user boundary in the stream.

  Precondition negative control: a fixture with ts ≤ 0 must be **rejected**, because the −1 sentinel would collide.
- **(d) Availability.**
  - The identical `has_history` stays in the tensor.
  - The stored `prev_user_event_available` is still read from storage, and it stays the M1 re-zero gate.
  - RC5 asserts that the two are equal.

  The cited `userctx.py` is byte-identical to the frozen build: its sha256 is `f02ff2ea…2ae254`, and `code_hash`
  matches (`g0/receipts_qa01/CODE_AND_FLAG_CHECKS.json:2-11`).

### 4.6 Why no column is DROP_SPEC_DEAD
Six flags are all-zero in full TRAIN and in VS (`DUP:16-21,31-36`), and also on the VC half
(`g0/receipts_qa02/OI06_RC_VALIDATION.json:91-122`). None of them is impossible by the code that was read:
- **price_conflict** is `price_block_conflict` (`EDA/pipeline6.py:170`).
- **brand_conflict and category_conflict** can arise through CONFLICT_ABSTAIN when a contradiction is visible
  (`EDA/metadata.py:172-173,195-197,210-221`).
- **category_missing** follows from NULL→MISSING, which is in the spec (`V1/FEATURE_DICTIONARY.md:26`).
- **category_recovered and category_code_recovered** are reachable whenever their families are enabled
  (`EDA/metadata.py:187-202,217-233`). The executed enabled state was not read (OI-08).

All six are therefore **KEEP**, labelled DEAD_IN_TRAIN, with this disclosure:
- An all-zero input gets zero gradient in its first-layer weight column (`FO/audit/MEASURED_FINDINGS.md:67-68`).
- That column stays at its seeded initialisation, apart from the decoupled weight decay of 1e-5 applied to ndim≥2
  parameters (the executed recipe, §14).
- If one of these flags becomes non-zero in TEST or in future data, the model receives an untrained column.
- RC9 counts non-zeros per split for disclosure.

### 4.7 Why `price_zero` and `main_category_missing` are KEEP
Both are empirically equal to a kept flag, with 0 contradictions (`DUP:12-13,27-28`). For main_category_missing,
QA-01 also found 0 on merged VALIDATION. In both cases the spec allows the two to diverge (§3 rows 27 and 37), so
empirical equality alone cannot justify removal (`PKG/03_DATA_PROTOCOL.md:41`).

`main_category_missing` is exactly `main_category_tokens == MISSING` by code (`EDA/pipeline6.py:179,325`). But no
full-population count exists for that identity, and it crosses from the token path to the flag path. Kept.

### 4.8 Other deterministic relations examined and kept
These relations are real, but each is non-affine, crosses paths or is clip-gated, and none has a full-population count:
- `local_weekday` vs `utc_weekday`+`utc_hour` (`FO/audit/REDUNDANCY_FINDINGS.md:95-119`).
- `is_weekend`, `daypart_tokens` and `weekday_tokens` vs the local clock (`:121-132`).
- `recent_history_length = min(expm1(log1p_prev_view_count),100)` and `has_history = 1[log1p_prev_view_count>0]`
  (`EDA/userctx.py:176-178`; `EDA/pipeline6.py:288,290-291`).
- `current_minus_recent_mean_log_price` vs `log1p_price` − `recent_mean_log_price` (clipped and masked;
  `FO/audit/REDUNDANCY_FINDINGS.md:140-144`).
- `brand_missing`, `category_missing` and `main_category_missing` vs their token being MISSING
  (`EDA/pipeline6.py:171,174,179,321-326`).
- `prev_gap_unavailable` vs `has_history` (event grain vs decision grain).

These are candidates only (OI-09).

## 5. M1 re-encoding (RE_ENCODE_M1)

**Defect (CODE).**
- Both raw values are filled with 0 when unavailable (`EDA/userctx.py:224-235`).
- Both are missing from the fit's `VALID_PREDICATE` (`EDA/pipeline6.py:399-407`).
- Both are missing from the load-time re-zero map `USER_MISSING_FLAG_OF` (`EDA/loaders.py:60-63`) (`MAP:189-219`).

**Formula (binding).** For feature f with governing mask m:

```
x = stored raw value (read as float64)                       # EDA/pipeline6.py:300-303, already log1p
c = min(max(x, clip_lo), clip_hi)                            # RECORDED bounds, not refit
z = (c - mean_valid) / std_valid   if m == 1   else 0.0      # re-zero via the availability mask
store float32(z)
```

| feature | mask m (stored) | clip_lo / clip_hi (normalizers.json) | old mean / std (normalizers.json) | **new mean / std (M1FIT valid_mean / valid_std)** | n valid / n TRAIN (M1FIT) |
|---|---|---|---|---|---|
| `log1p_time_since_prev_user_event_s` (user_context[10]) | `prev_user_event_available` (in the tensor as `has_history`) | 0.0 / 14.877890586853027 (`:128-131`) | 4.950253279346413 / 3.3790111405204004 (`:132-133`) | **5.3788876148270335 / 3.178172706038617** (`M1FIT:128-129`) | 48,555,935 / 52,760,314 (`M1FIT:70,113-121`) |
| `log1p_time_since_prev_session_last_event_s` (user_context[11]) | `prev_session_available` | 0.0 / 14.877890586853027 (`:135-138`) | 1.8615436239173029 / 4.354925521013689 (`:139-140`) | **11.8168906913688 / 1.6580237946197298** (`M1FIT:100-101`) | 8,311,461 / 52,760,314 (`M1FIT:70,85-93`) |

- **Fit population.** Full TRAIN: all 64 user buckets (`M1FIT:3-68`), split TRAIN (`M1FIT:73`), rows with m = 1.
  This is all TRAIN users, not the 262,144 cohort (Fable decision 4, `G0_PLAN_FABLE_POSITION.md:29`).
- **Fit grain (the executed convention, stated explicitly).**
  - **Grain:** one unweighted row per TRAIN row of `model_ready/…/features.parquet`. That is one per canonical
    representative, which is one per TRAIN decision (`V1/DATA_CONTRACT.md:227-234`).
  - **Row selection:** every such row counts, whether or not it has `loss_mask` or target availability. The fit reads
    inputs only, never labels.
  - **Estimator:** values are clipped at the recorded bounds (`M1FIT:74`), and the std is the population std, ddof 0
    (`EDA/pipeline6.py:427-433`).
  - **Why only the filter changes:** M1FIT reproduces the recorded all-row normalizer with this same grain
    (`normalizer_reproduced_by_all_rows = true`, `M1FIT:90-94,102,118-122,130`). Only the row filter changes.
  - **Protocol match:** this matches the rules "once per canonical representative" and "user-context once per TRAIN
    decision" (`PKG/03_DATA_PROTOCOL.md:37`). It is used identically for every method.
- **Source of numbers.** `M1FIT` `valid_mean` / `valid_std`. The `train_valid` blocks repeat them (`M1FIT:85-89,113-117`).
  The `vs_valid_drift_only` blocks (`:95-99,123-127`) are drift reports and are **not** used.
- **Reproduced (OI-04 CLOSED).** QA-02 recomputed n_valid, mean and std for both features on full TRAIN at this grain.
  They equal the M1FIT values exactly: abs diff 0.0, rel diff 0.0, within 1e-9 relative
  (`g0/receipts_qa02/OI04_M1_REPRODUCTION.json:10-37`).
- **The re-zero is mandatory.** Without it, an unavailable row would encode as z = −1.6924 (user) and −7.1271
  (session) under the new statistics. In v1 the values were −1.4650 and −0.4275 (DERIVED). For comparison, the valid
  encoded upper ends are +2.989 and +1.846.
- **Degeneracy guard.** Refuse to proceed if `std_valid ≤ 1e-12·max(1,|mean_valid|)` (the pattern at
  `FO/code/feature_view.py:262-265`).
- **Disclosure (NOT established).** The clip bounds are the recorded all-row quantiles. clip_lo = 0.0 is pinned by the
  masked zeros (`EDA/pipeline6.py:395-398`). They are kept by decision, and the valid-only clip rates were not measured.
- **Does any other quantity depend on these two normalizers?** No stored feature, vocabulary or other normalizer
  does:
  - each normalizer is fitted independently per column (`EDA/pipeline6.py:406-439`);
  - stored parquet values are raw (`EDA/pipeline6.py:114,300-303`).

  What changes in V2:
  - the encoded values of these two channels;
  - the encoded cold-user vector (these two entries become 0.0);
  - any golden tensor, probe or export-parity fixture built from encoded tensors;
  - the TRANSFORM_STATE and schema hashes.

  Historical objects that depend on the old values are **not used**:
  - v1 checkpoints;
  - the lean fold (`FO/code/feature_view.py:326-333`);
  - the FO probe cold vector (`:84-89`);
  - `06_numeric_clipping_rates.csv`.

## 6. The unknown→0-after-normalization rule and the re-zero map

**Order of operations:**
1. Start from the stored raw value. Columns named `log1p_*` are already log1p.
2. Clip at the recorded bounds.
3. z-score.
4. Re-zero.
5. Select the V2 columns.

Never apply log1p to a stored value, never re-standardize a z-score, and never reject a valid negative z
(`PKG/03_DATA_PROTOCOL.md:33`).

| feature (V2 slot) | written as 0.0 after z when | vs v1 |
|---|---|---|
| `log1p_price` (num 0) | price_missing OR price_conflict at that event (`EDA/loaders.py:284-296`) | same |
| `log1p_prev_raw_view_gap_s` (num 8) | prev_gap_unavailable (`EDA/loaders.py:57`) | same |
| `log1p_item_observed_age_s` (num 10) | item_age_unavailable (`EDA/loaders.py:58`) | same |
| `recent_mean_log_price` (uctx 8) | recent_price_supported == 0 (`EDA/loaders.py:61`) | same |
| `current_minus_recent_mean_log_price` (uctx 9) | recent_price_supported == 0 **OR** price_missing/price_conflict at the decision's own event (`last_valid_index`; `EDA/loaders.py:264-271`) | **EXTENDED + valid-only refit (F-1, ADOPTED)** |
| `log1p_time_since_prev_user_event_s` (uctx 10) | stored prev_user_event_available == 0 | **NEW (M1)** |
| `log1p_time_since_prev_session_last_event_s` (uctx 11) | prev_session_available == 0 | **NEW (M1)** |
| every `[B,L,*]` tensor on PAD | attention_mask False (`EDA/loaders.py:241-249`) | same |

**F-1 — ADOPTED inside scope (Fable G0(b) C2, `G0B_PROTOCOL_FABLE_POSITION.md:16`; binding).**

**The defect (CODE).**
- The raw value is 0 whenever the current price is invalid, even if recent prices exist (`EDA/userctx.py:221-222`).
- v1 gates the re-zero on `recent_price_supported` only (`EDA/loaders.py:62`).
- v1 fits on `recent_price_supported > 0` (`EDA/pipeline6.py:404`), so these sentinel zeros are both encoded as
  z(0) = −0.00207 and included in the fit. This is the Q3 case in `g0/feature_map/OPEN_ITEMS.md:50-56`, and the same
  defect class as M1.

**Measured (QA-02, `g0/receipts_qa02/OI03_F1_PRICE_REZERO.json`).**
- The predicate is `recent_price_supported = 1 AND (price_missing = 1 OR price_conflict = 1)`, counted per decision
  row (`:4-5`).
- TRAIN: **117,315** of 52,760,314 rows, 0.22% (`:6-9`).
- Merged VALIDATION: **64,201** of 14,357,645 rows — VS 56,318 and VC 7,883 (`:10-21`).

**Binding rule.** Both parts apply:
- **(i) Re-zero gate.**

  ```
  unknown := recent_price_supported == 0 OR price_missing OR price_conflict   (at last_valid_index)
  ```

- **(ii) Valid-only refit.** It uses the **recorded** clip bounds [−4.090688228607178, 3.681367874145508]
  (`V1/mappings/normalizers.json:122-123`):
  - fit predicate `recent_price_supported > 0 AND NOT price_missing AND NOT price_conflict`;
  - fit on full TRAIN, one row per TRAIN decision, population std ddof 0;
  - **mean 0.002065693626402541, std 0.9945344391661521, n_valid 48,434,230** (`OI03_F1_PRICE_REZERO.json:23-32`).

  The old values, 0.0020607022950704136 / 0.9933321718650419 (`V1/mappings/normalizers.json:124-125`), become
  historical.

```
c = min(max(x, -4.090688228607178), 3.681367874145508)
z = (c - 0.002065693626402541) / 0.9945344391661521   if NOT unknown   else 0.0
```

**Consistency (DERIVED here; Fable states the same check at `G0B_PROTOCOL_FABLE_POSITION.md:16`).**
- Adding the 117,315 zeros back to the valid-only fit gives mean 0.0020607022950704192 and std 0.9933321718650323.
  The recorded values are 0.0020607022950704136 and 0.9933321718650419, a relative difference of 2.7e-15 and
  −9.7e-15. So the receipt is internally coherent.
- 48,434,230 + 117,315 = 48,551,545, which is ≤ the 48,555,935 has-history rows.

**Effect.**
- An affected row's encoding moves from −0.00207 to 0.0.
- The std changes by +0.12% (DERIVED).
- No other stored quantity depends on this normalizer (`EDA/pipeline6.py:406-439`).

**Genuine zeros that are z-scored by design (not unknowns; disclosed, DERIVED from normalizers.json):**
- log1p_segment_elapsed_s −1.6074
- prefix_representative_count −0.5640
- prefix_known_category_count −1.3206
- prefix_category_switch_count −0.2583
- current_known_category_run_length −0.5278
- log1p_prev_view_count −1.8249
- log1p_prev_session_count −1.6628
- recent_history_length −0.8937
- recent_distinct_items −0.9121
- recent_distinct_known_main_categories −1.2033

**Expected V2 cold-user vector** (has_history=0, every mask 0):
- user_context =
  `[−1.8249, −1.6628, −0.8937, 0, −0.9121, −1.2033, 0, 0, 0, 0, 0, 0]`
- masks = `[0,0,0,0]`

This is a golden value for T09/T14 (DERIVED). DATA-04 confirms it on real cold rows.

## 7. Final tensor layout per family

| key | shape / dtype | V2 order (v1 source index) | state |
|---|---|---|---|
| `item_tokens`, `category_tokens`, `brand_tokens`, `main_category_tokens`, `daypart_tokens`, `weekday_tokens` | `[B,50]` int64 | unchanged | token: PAD 0 / MISSING 1 / OOV 2 / known ≥ 3 |
| `event_numeric_features` | `[B,50,15]` float32 | see the list below | see the list below |
| `event_quality_flags` | `[B,50,15]` int8; models cast it to float32 | the v1 order, all 15 (§3 C) | raw 0/1 |
| `user_context` | `[B,12]` float32 | the v1 order, all 12 (§3 D) | [0,1] LOG1P(count)→clip→z; [2,4,5] count→clip→z; [3] binary raw; [6,7] share raw; [8] LOG1P→clip→z→rezero; [9] logdiff→clip→z valid-only→rezero (F-1); [10,11] LOG1P→clip→z valid-only→rezero (M1) |
| `user_context_masks` | `[B,4]` float32 | recent_category_match_supported(1), recent_brand_match_supported(2), recent_price_supported(3), prev_session_available(5) | raw 0/1 |
| unchanged | — | attention_mask, position_ids, lengths, last_valid_index, target_class, loss_mask, eval_mask, decision_id, client_id, model_segment_id (`V1/DATA_CONTRACT.md:38-57`) | — |
| added | — | `event_numeric_feature_names`, `event_quality_flag_names`, `user_context_names`, `user_context_mask_names`, `feature_view="clean_v2_rich"`, `clean_v2_state_sha256` | — |

**`event_numeric_features` in V2 order (v1 source index in parentheses):**

| V2 idx | channel (v1 idx) | units | state |
|---|---|---|---|
| 0 | log1p_price (0) | log(1+currency) | LOG1P→clip[1.15688, 7.84912]→z(5.04455, 1.20851)→rezero |
| 1 | utc_hour_sin (1) | dimensionless | cyclic, raw |
| 2 | utc_hour_cos (2) | dimensionless | cyclic, raw |
| 3 | utc_weekday_sin (3) | dimensionless | cyclic, raw |
| 4 | utc_weekday_cos (4) | dimensionless | cyclic, raw |
| 5 | local_weekday_sin (7) | dimensionless | cyclic, raw |
| 6 | local_weekday_cos (8) | dimensionless | cyclic, raw |
| 7 | is_weekend (9) | indicator | binary, raw |
| 8 | log1p_prev_raw_view_gap_s (10) | log(1+s) | LOG1P→clip→z→rezero |
| 9 | log1p_segment_elapsed_s (11) | log(1+s) | LOG1P→clip→z |
| 10 | log1p_item_observed_age_s (12) | log(1+s) | LOG1P→clip→z→rezero |
| 11 | prefix_representative_count (13) | count | count→clip→z |
| 12 | prefix_known_category_count (15) | count | count→clip→z |
| 13 | prefix_category_switch_count (16) | count | count→clip→z |
| 14 | current_known_category_run_length (17) | count | count→clip→z |

The full-precision parameters of all 17 normalized kept columns are in
`FEATURE_SCHEMA_CLEAN_V2.json → normalized_channels_v2`. The 17 break down as follows:
- **14 unchanged:** quoted from `V1/mappings/normalizers.json`;
- **2 M1 refits:** quoted from M1FIT;
- **1 F-1 refit:** quoted from `OI03_F1_PRICE_REZERO.json`.

None is constant. The constant rule remains: a column with no valid row, hi ≤ lo, or
std ≤ 1e-12 is encoded as 0 with scale 1 (`EDA/pipeline6.py:412-426,434-436`; `EDA/loaders.py:79-80`).

**GRU sizing** (`SNAP/reuse/model.py`):
- **Group spec:** `{"categorical": [the 6 token channels], "numeric_dim": 15, "flags_dim": 15, "user_context": true,
  "user_context_dim": 12, "user_mask_dim": 4}`. `resolve_group_spec` already accepts a dict (`:97-103`).
- **Layers:**
  - `numeric_flags_proj` = Linear(30, 32) followed by GELU (`:190-195`);
  - the concat width stays 240 = 128+32+32+8+4+4+32 (`:132-139,177-199`);
  - `input_proj` is Linear(240,128), unchanged;
  - **`user_ctx_proj` = Linear(16, 64)** followed by GELU;
  - `readout_proj` is Linear(320,128), unchanged.
- **Required change:** `_USER_CONTEXT_DIM=12` and `_USER_CONTEXT_MASK_DIM=6` are hard-coded (`:148-149,209-212`), and
  the reuse file is read-only. A V2 module or subclass must take the mask width, 4, from the schema (OI-05).

**SASRec sizing** (`SNAP/sasrec_model.py`):
- **Constants:** the module constants `NUMERIC_DIM, FLAGS_DIM = 19, 15` and `USER_CTX_DIM, USER_MASK_DIM = 12, 6`
  (`:53-54`) must become **15, 15 and 12, 4**. They must be constructor arguments from the schema, with a width
  assertion.
- **Layers:**
  - `numflag_proj` = Linear(30, 32), then GELU in `_side_residual` (`:167,251-256`);
  - `side_proj` stays Linear(112, d): 32+32+8+4+4 side dims plus 32 (`:49-52,168-169`);
  - **`user_mlp` = Linear(16, 64)**; `user_proj` stays Linear(64, d) (`:172-173,258-261`);
  - the attributes are `numeric_dim=15, flags_dim=15` (`:146-149`).

**Both families:**
- **Vocab sizes** are unchanged. They are read from the mapping files. The reuse docstring gives item 158,489 (=K+3),
  category 663, brand 3,956, main_category 16, daypart 8 and weekday 10 (`SNAP/reuse/model.py:412-415`); the builder
  asserts these against the files.
- **Checkpoints:** a v1 or lean checkpoint must never be loaded into a V2 model. A shape mismatch is a hard refusal.

## 8. Locality labels and bootstrap-information disclosure

**Locality of the kept channels:** LOCAL_EVENT 25, LOCAL_HISTORY 22, SHARED_CATALOGUE 3, SHARED_TRAIN_STATISTIC 3,
UNSUPPORTED_FUTURE 0 (53 in total). In v1 the counts were 27/26/3/3 (`MAP:20-29`). The labels per channel are in §3.
34 of the 53 kept channels depend on pooled interactions of other users. In v1 it was 36 of 59 (`MAP:31`; counted from
the MAP rows).

**Bootstrap-information disclosure** (pooled information, not independent public information,
`PKG/03_DATA_PROTOCOL.md:46`):
1. **B1.** All 17 clipped and z-scored kept channels use pooled full-TRAIN normalizers, even for purely local raw
   values (`MAP:147-153`).
2. **B2.** The item, category, brand and main-category vocabularies and the support≥5 catalogue are pooled over all
   TRAIN users (`MAP:154-160`).
3. **B3.** Brand and category recovery is pooled per product. category_code is also pooled per category_id with ≥3
   distinct products. Future partitions use the frozen end-of-TRAIN snapshot (`MAP:161-171,221-243`).
4. **B4.** Item age is anchored to the pooled TRAIN first-seen time (`MAP:172-177`).
5. **B5.** client_idx is a TRAIN-pooled identifier and is not a V2 feature (`MAP:178-182`).
6. **B6.** The masked-zero user features keep a pooled normalizer. V2 removes only the M1 contamination
   (`MAP:183-187`).
7. **B7 (new).** The M1 valid-only mean/std are pooled statistics over 48,555,935 and 8,311,461 valid TRAIN decisions
   (`M1FIT:85-89,113-117`).
   - **B7′ (new, F-1).** The F-1 valid-only mean/std of `current_minus_recent_mean_log_price` are pooled statistics
     over **n_valid = 48,434,230** valid TRAIN decisions (`g0/receipts_qa02/OI03_F1_PRICE_REZERO.json:29-31`).
8. **B8 (new).** Every SHARED_TRAIN_STATISTIC is fitted on all TRAIN users, including users outside the cohort and
   outside any client's own data. C, FL and LO all receive the same snapshot (`PKG/03_DATA_PROTOCOL.md:47`).
9. **B9 (new, provenance).** Under the minimal-diff VALIDATION, user-context replay for a decision uses TRAIN plus that
   decision's own legacy partition. So for old VC decisions, the user's old VS events are omitted
   (`EDA/pipeline6.py:5-13,42-47`; `PKG/03_DATA_PROTOCOL.md:15`). This is disclosed, not changed.

## 9. Transform-state provenance

Each object below is a **SHARED_TRAIN_STATISTIC** fitted on pooled full TRAIN, and each is identical across
C/FA/FP/PF/CPF/LO:

| object | where it lives | fitted on | cite |
|---|---|---|---|
| item catalogue and vocabulary, support ≥ 5, K=158,486 | `V1/mappings/catalogue.parquet` | TRAIN representatives of all users | `V1/FEATURE_DICTIONARY.md:25`; `MAP:154-160` |
| category_id, brand, main_category and category_code vocabularies | `V1/mappings/vocab_*.parquet` | TRAIN resolved values | `V1/DATA_CONTRACT.md:194-201` |
| 14 unchanged numeric normalizers | `V1/mappings/normalizers.json` (sha256 9815…5f45, matching `V1/FILES.json:16-19`) | TRAIN feature rows (valid-only where listed) | `EDA/pipeline6.py:391-449` |
| 2 M1 valid-only mean/std | M1FIT → TRANSFORM_STATE_CLEAN_V2 (reproduced by `OI04_M1_REPRODUCTION.json`) | TRAIN rows with mask=1 | `M1FIT:73-131` |
| 1 F-1 valid-only mean/std (`current_minus_recent_mean_log_price`) | `OI03_F1_PRICE_REZERO.json` → TRANSFORM_STATE_CLEAN_V2 | TRAIN rows with recent_price_supported > 0 AND NOT price_missing AND NOT price_conflict (n_valid 48,434,230) | `g0/receipts_qa02/OI03_F1_PRICE_REZERO.json:23-32` |
| recovery tables (product-level and category-level as-of states; frozen snapshot) | baked into the v1 features | TRAIN original observations of all users | `EDA/metadata.py:73-169,180-244` |
| first_seen_train | baked into item age and its flag | MIN(event_time_utc) per product over TRAIN | `MAP:245-258` |
| client_idx vocabulary (an identifier only) | `V1/mappings/clients.parquet` | TRAIN users | `V1/FEATURE_DICTIONARY.md:62` |
| clip quantiles 0.001 / 0.999 | `V1/mappings/normalizers.json:3-6` | declared | — |

The daypart and weekday vocabularies are declared closed forms and are not fitted (`EDA/pipeline6.py:53-62`).

## 10. Sidecar builder design (for DATA-04)

**Inputs and hashes.** These are MEASURED at authoring time; the full list is in the schema under `base`.

**v1 dataset:**
- `V1/VERSION.json` `version_hash 860e5a1c3a81…8dbd`; frozen `code_hash 02fbb071…2dde73` (`:3,20`).
- `V1/FILES.json` sha256 `4d4d20d3…4b1b`. It is the manifest the builder verifies for each of the 64
  `model_ready/user_bucket=*/features.parquet` files and for every `mappings/*` file (OI-07).
- `normalizers.json` `98150624…5f45` and `FEATURE_GROUPS.json` `6b9b0589…6f26`. Both match FILES.json.

**Audit inputs:**
- M1FIT `7575010b…3c60`
- DUP `17d4403d…a8d8`
- FO TRANSFORM_STATE `6a25857b…2ff1`

**Receipts** (the full hashes are in the schema under `base.receipt_hashes_sha256`):
- `OI03_F1_PRICE_REZERO.json` `54aea414…4b92`, the number of record for F-1
- `OI04_M1_REPRODUCTION.json` `d03bb794…7f67`
- `OI06_RC_VALIDATION.json` `82ee227b…2227`
- `OI07_FILE_INTEGRITY.json` `db4bc10a…c070`
- `EXECUTED_RECIPES.md` `01ae1bd4…23b4`
- `G0B_PROTOCOL_FABLE_POSITION.md` `23eefa74…c91b`

**File integrity (OI-07 CLOSED).** QA-02 recomputed `version_hash` and it equals the recorded value. All 73/73
in-scope files match `FILES.json` in size and sha256: 64 `features.parquet`, 7 mappings, `splits.json` and
`FEATURE_GROUPS.json` (`g0/receipts_qa02/OI07_FILE_INTEGRITY.json:4-21`).

**Code:**
- Live EDA code:
  - `loaders.py` `212b1b4c…ac73`
  - `pipeline6.py` `500a3c63…81bb`
  - `norm.py` `4c741cbc…603e`
  - `metadata.py` `07093765…167f`
  - `userctx.py` `f02ff2ea…2ae254` (QA-01).
- The live tree equals the frozen build, because `code_hash` matches (QA-01 5a).
- The builder must also hash the loader module that training *actually imports*, i.e. the snapshot's vendored copy
  (T43).

**Outputs.** DATA-04 writes `TRANSFORM_STATE_CLEAN_V2.json` and test receipts only. There are no per-decision or
per-window tensors, caches or copies (`PKG/03_DATA_PROTOCOL.md:50`).

**Load-time API (names indicative; the signatures are binding):**

```python
load_clean_v2_state(transform_state_path: Path, *, schema_path: Path, v1_root: Path,
                    verify_hashes: bool = True) -> CleanV2State            # fail-closed on any hash mismatch
clean_v2_normalizers(state: CleanV2State) -> dict[str, dict[str, float]]   # v1 spec with ONLY the two M1 mean/std replaced
clean_v2_view(batch: Mapping[str, np.ndarray], state: CleanV2State, *,
              encoded_with: Literal["V2_NORMALIZERS", "V1_NORMALIZERS"] = "V2_NORMALIZERS"
              ) -> dict[str, np.ndarray]
```

**`clean_v2_view` behaviour:**
- It is pure and deterministic, does no I/O, and never mutates its input.
- It takes the **v1-rich** batch (`[B,50,19]`, `[B,50,15]`, `[B,12]`, `[B,6]`).
- Steps:
  1. check shapes and names;
  2. run RC1–RC8;
  3. apply the M1 encoding;
  4. apply the M1 and F-1 re-zero;
  5. select the columns;
  6. attach the name arrays and `clean_v2_state_sha256`.
- **Preferred path:** build the v1 loader with the V2 normalizer spec. `DecisionWindowLoader` already accepts
  `normalizers=` (`EDA/loaders.py:159-173`).
- **Fallback path:** `EXACT_INVERSE`. If the reused store cannot inject normalizers, first invert the v1 z exactly in
  float64 on available rows (the pattern at `FO/code/feature_view.py:279-284`). Record which path was used in
  TRANSFORM_STATE.
- A torch twin must be tested equal to the numpy reference.

**Runtime identity checks.** These run on every batch **before** any drop, on real positions. RC1–RC8 are fail-closed:
- **RC1:** |num[18]−num[11]| ≤ 1e-6
- **RC2:** |num[14]−num[13]| ≤ 1e-6
- **RC3:** rotation residual ≤ 1e-6
- **RC4:** mask[0] == uctx[3]
- **RC5:** mask[4] == uctx[3]
- **RC6:** mask[5] = 1 ⇒ mask[4] = 1 (`EDA/userctx.py:233`)
- **RC7:** the shares are 0 where their masks are 0
- **RC8:** all values finite
- **RC9** is not fail-closed: it logs the dead-flag non-zero counts and never raises.

**TRANSFORM_STATE_CLEAN_V2.json fields:**
- **Identity:** `schema_version`, `view_id="clean_v2_rich"`, `status` (DRAFT | LOCKED), `created_utc`,
  `created_by_task`.
- **Input hashes:** `v1_version_hash`, `v1_files_manifest_sha256`, `v1_input_sha256{}`, `base_normalizers_sha256`,
  `schema_sha256`, `contract_sha256`, `runtime_module_sha256{}`.
- **Fit:** `fit{population=FULL_TRAIN, split, buckets 0..63, grain, estimator, rows_all=52,760,314, source_sha256}`.
- **M1:** `m1{per feature: mask, clip_lo/hi, old/new mean/std, n_valid, unavailable_encoding=0.0, rule,
  reproduction_receipt}`.
- **F-1:** `f1{gate predicate, clip_lo/hi, old mean/std, new mean 0.002065693626402541, new std 0.9945344391661521,
  n_valid 48434230, unavailable_encoding=0.0, receipt sha256}`.
- **Transform spec:** `rezero_map`, `drops{keeper, expression, DUP counts, check id}`, `layout{names,
  v1_source_indices}`, `shared_train_statistics[{object, sha256, identical_across_methods=true}]`,
  `m1_application_path`.
- **Status:** `equivalence=null`, `prior_lean_gate="PARITY_FAILED (unchanged)"`.
- **Self-hash:** `transform_state_sha256`, the sha256 of the canonical JSON with this field removed.

**Required tests** (`PKG/12_ACCEPTANCE_TESTS.md:13-19`). The case lists in full are in the schema under
`required_tests`.

- **T08 — missing/invalid price, finite values, zero variance, all-missing.**
  - **Cases:**
    - a NULL, negative, NaN, +inf or zero price, and a conflicting block, all give log1p_price = 0.0;
    - every value is finite;
    - a zero-variance column and an all-missing column are both encoded as constant 0;
    - a degenerate M1 std is refused.
  - **Negative controls:**
    - an injected NaN must raise;
    - a std=0 column not marked constant must raise;
    - price_missing=1 with a non-zero raw value must still give 0.
  - **Proves:** the missing, finite and constant rules. **Does not prove:** that future data has no new invalid mode.
- **T09 — valid-only fitting, re-zero after the transform.**
  - **Cases:**
    - a synthetic fit on mask=1 rows only, with PAD excluded;
    - real data, M1: the two mean/std recomputed on full TRAIN at the stated grain match M1FIT within 1e-9 relative
      (done: `OI04_M1_REPRODUCTION.json`, abs diff 0.0);
    - real data, F-1: the valid-only fit equals mean 0.002065693626402541, std 0.9945344391661521 and n_valid
      48,434,230 (`OI03_F1_PRICE_REZERO.json:23-32`);
    - every re-zero row gives exactly 0.0;
    - the cold-user vector equals the golden value;
    - an F-1 row gives 0.0.
  - **Negative controls:**
    - a fit that includes mask=0 rows reproduces the old normalizer (M1FIT `normalizer_reproduced_by_all_rows`) and
      must be rejected;
    - removing an M1 entry from the re-zero map must surface −1.6924 / −7.1271 and fail;
    - **F-1:** a `current_minus_recent_mean_log_price` fit that includes the 117,315 current-price-unknown zeros must
      reproduce the old std 0.9933321718650419 (and mean 0.0020607022950704136) and be rejected
      (`G0B_PROTOCOL_FABLE_POSITION.md:16`).
  - **Proves:** the fit population and the re-zero semantics. **Does not prove:** any ranking effect.
- **T10 — TRAIN-only fitting; a future-only category is rejected as known.**
  - **Cases:**
    - every fitted statistic traces to TRAIN rows;
    - a category, brand, main category or product first seen after TRAIN maps to OOV=2;
    - a product outside the catalogue is OOV in context and is not a class.
  - **Negative controls:**
    - a TRAIN+VS fit gives a different mean (the M1FIT drift blocks show the shift) and must be rejected;
    - a vocabulary containing a future-only value must be detected.
  - **Proves:** TRAIN-only fitting. **Does not prove:** the absence of bootstrap information, which is present by
    design.
- **T11 — no double-log or double-standardize; explicit state contract.**
  - **Cases:**
    - each column carries a declared state, and the transform is a state machine RAW/LOG1P → CLIP → ZSCORE → REZERO in
      which illegal transitions raise;
    - valid negative z values pass;
    - cyclic, flag and mask columns pass through unchanged.
  - **Negative controls:**
    - applying log1p twice must fail;
    - re-standardizing a z-scored batch must be detected.
  - **Proves:** the declared transform chain. **Does not prove:** that the encoding is optimal.
- **T12 — duplicate algebra, full-population contradictions, negative controls; dead flags retained.**
  - **Cases:**
    - synthetic fixtures for each §4.1–4.5 case list;
    - RC1–RC7 on every batch;
    - the DUP counts for TRAIN and VS;
    - the feature-only pass over merged VALIDATION, including VC. It is done: RC1–RC6 all 0, and the dead flags are 0
      on VC (`OI06_RC_VALIDATION.json`);
    - the 6 DEAD_IN_TRAIN flags, price_zero and main_category_missing are present at their v1 indices.
  - **Negative controls:**
    - a one-row perturbation for each identity must fail;
    - a schema that drops a dead flag must be rejected.
  - **Proves:** the removed columns are recoverable on the checked populations, and the dead flags stay available.
    **Does not prove:** equal trained accuracy, or anything about TEST before it is loaded.
- **T14 — local provenance, as-of item age, no cross-client history contamination.**
  - **Cases:**
    - every kept channel carries a locality label;
    - in a two-user interleaved replay, user A's LOCAL_HISTORY values are invariant to user B's events
      (`EDA/userctx.py:156-236`, checked against the slow reference `:243-309`);
    - item age: an item first seen after TRAIN is unavailable, and one first seen exactly at block_ts is unavailable
      (`EDA/pipeline6.py:151-152,181`);
    - the SHARED_TRAIN_STATISTIC hashes are identical in every method's manifest.
  - **Negative controls:**
    - an interleaved, unsorted stream must be detected;
    - a first-seen computed over all months must change the fixture's item age and be rejected.
  - **Proves:** provenance and per-user isolation. **Does not prove:** privacy.

## 11. Prior lean gate — unchanged

**What the feature study recorded:**
- The feature study's lean view is `PARITY_FAILED_NOT_ADOPTED` (`FOI/TRANSFORM_STATE.json:56-68`).
- These four checks passed:
  - the full-population label;
  - all duplicate and dead counts 0;
  - rotation ≤ 1e-6;
  - max |logit| ≤ 1e-3.
- Only `rank_identical_ge_0.999` failed, at 0.998683 (`PKG/03_DATA_PROTOCOL.md:4`).
- The selection was KEEP_CURRENT (`FOI/SELECTED_FEATURES.json:2-3,188-190`).

**How Clean V2 relates to it:**
- Clean V2 is a different, from-scratch topology.
- It drops 6 columns instead of 14, and it keeps the 8 columns that were dropped on empirical evidence.
- It makes no parity or equivalence claim. The lean gate is therefore neither re-run nor weakened, and it stays
  **FAILED**.

## 12. Open items

**The lock blockers — all now CLOSED:**

| id | item | owner | status |
|---|---|---|---|
| OI-01 | QA-01 receipt 5b: counts of `category_code_missing ≠ main_category_missing` in TRAIN and merged VALIDATION (Fable blocker 5, `G0_PLAN_FABLE_POSITION.md:72`) | QA-01 | **CLOSED**: 0 in TRAIN and 0 in VALIDATION (VS 0 of 10,930,090, VC 0 of 3,427,555) (`g0/receipts_qa01/CODE_AND_FLAG_CHECKS.json:65-86`). The decision is unchanged: KEEP (§4.7) |
| OI-02 | QA-01 receipt 5a: sha256 of `userctx.py` and `code_hash()` vs the frozen `02fbb071…` | QA-01 | **CLOSED**: the recomputed hash equals both frozen records; userctx.py is `f02ff2ea…2ae254` (`g0/receipts_qa01/CODE_AND_FLAG_CHECKS.json:2-11`) |
| OI-03 | F-1: a full-TRAIN count of rows with recent_price_supported=1 and a current price_missing or price_conflict | QA-02 (count); Fable (ruling) | **CLOSED**: 117,315 of 52,760,314 TRAIN rows and 64,201 of 14,357,645 VALIDATION rows. The refit was triggered and F-1 ADOPTED with re-zero + valid-only refit (§6) (`g0/receipts_qa02/OI03_F1_PRICE_REZERO.json:6-32`; `G0B_PROTOCOL_FABLE_POSITION.md:16`) |
| OI-04 | M1 reproduction receipt (the T09 real-data case) | QA-02 | **CLOSED**: both features reproduced exactly (abs diff 0.0) (`g0/receipts_qa02/OI04_M1_REPRODUCTION.json:10-37`). The builder still writes `TRANSFORM_STATE_CLEAN_V2.json` from this schema |
| OI-06 | A feature-only full pass of RC1–RC6 and the dead-flag counts over merged VALIDATION | QA-02 | **CLOSED**: RC1–RC6 found 0 contradictions on merged VALIDATION (14,357,645 rows), including the 3,427,555-row VC half. The six dead flags are 0 on VS and VC. No target was read (`g0/receipts_qa02/OI06_RC_VALIDATION.json:4-122`) |
| OI-07 | Verify the v1 version_hash and every FILES.json entry that is read | QA-02 | **CLOSED**: the recomputed version_hash equals `860e5a1c…`, and **73/73** in-scope files match (`g0/receipts_qa02/OI07_FILE_INTEGRITY.json:4-21`) |

**Still open (not blocking the lock):**
- **OI-05 — schema-driven model input widths. Owner: Wave B MODEL-01.**
  - **What it blocks:** G1/G2 (Fable G0(b) BLOCKERS B, `G0B_PROTOCOL_FABLE_POSITION.md:39`), the code tests and the
    central contest.
  - **What is needed:** parameterize the GRU user mask width and the SASRec NUMERIC/FLAGS/USER widths (§7).
  - **The final Clean V2 widths MODEL-01 must consume:** numeric 15 / flags 15 / uctx 12 / masks 4. With the 6 token
    channels and the output catalogue, that is **53 channels** (Fable LOCK_VALUES, `G0B_PROTOCOL_FABLE_POSITION.md:31`).
- **OI-08.** Read the executed recovery-family enabled state (`EDA/metadata.py:187-202`). A later variant could
  reclassify flags as DROP_SPEC_DEAD. There is no effect on V2. Owner: Data Lead, optional.
- **OI-09.** Full-population counts for the token-identity flags and the non-affine relations (§4.8). There is no
  effect on V2. Owner: Data Lead, optional.
- **OI-10.** Export parity fixtures must be rebuilt on the V2 layout (T42). Owner: Deployment/QA (EDGE-02 follow-up).

## 13. What is NOT established

- **Unchecked split.** The behaviour of the six removal identities and the dead flags on TEST. TRAIN, VS and VC are
  measured (DUP; `OI06_RC_VALIDATION.json`).
- **Dead flags in future data.** Whether any DEAD_IN_TRAIN flag fires in TEST or in future data. All six are 0 on
  TRAIN, VS and VC. If one fires, the model applies an untrained weight column.
- **Clip statistics.** The valid-only clip rates and quantile bounds of the two M1 features and the F-1 feature. The
  recorded bounds are kept.
- **F-1 ranking effect.** Any ranking effect of F-1. Numerically it is a 0.12% std change on 0.22% of TRAIN rows
  (`G0B_PROTOCOL_FABLE_POSITION.md:44` calls it "≈0.1% std change").
- **Model effect.** Any accuracy, ranking or calibration effect of Clean V2 vs v1 rich. No training has been run. The
  expected ranking impact of M1 was only ever a "HYPOTHESIS, small" (`FO/audit/MEASURED_FINDINGS.md:45`).
- **Equivalence.** Any equivalence between Clean V2 models and v1 or lean weights. None is claimed.
- **Local clock.** That the UTC+3h clock reflects the audience. It is labelled HYPOTHESIS_NOT_CONFIRMED_STORE_GEOGRAPHY
  (`V1/model_ready/FEATURE_GROUPS.json:174-176`).
- **`simultaneous_block`.** Its upstream definition was not re-read.
- **Recovery families.** The recovery-family enabled state of the executed build (OI-08).
- **Privacy.** Privacy of any kind. Pooled bootstrap information is present by design (§8).

## 14. Reference for the lock: executed optimizer values (Fable G0(b) C8)

These values are transcribed as the **reference for DATA_PROTOCOL_LOCK** (`G0B_PROTOCOL_FABLE_POSITION.md:28`). They
are the executed values of the v1 campaigns, not new tuning. The source is `g0/receipts_res02/EXECUTED_RECIPES.md`,
which is the transcript Fable read; the `.json` next to it carries path-level citations. The V4 precision and schedule
are set elsewhere: strict FP32 with matmul and cuDNN TF32 off (G0 decision 5, `G0_PLAN_FABLE_POSITION.md:31-33,60`),
and the EFE schedule with the C3 stopping rule (`G0B_PROTOCOL_FABLE_POSITION.md:18`).

| field | GRU (ContextGRU) | SASRec (A1_RICH256) |
|---|---|---|
| optimizer | AdamW, fused (`EXECUTED_RECIPES.md:12`) | AdamW, fused (cuda) (`:51`) |
| betas / eps | (0.9, 0.999) / 1e-8 (`:13`) | (0.9, 0.999) / 1e-8 (`:52`) |
| weight decay | 1e-5 on ndim≥2, 0 elsewhere (`:14`) | 1e-5 on ndim≥2 including the untied output table, 0 elsewhere (`:53`) |
| grad clip | 1.0 global norm (`:15`) | 1.0 (`:54`) |
| peak LR | 1e-3 (`:16`) | 1e-3 (`:56`) |
| effective batch / microbatch | 256 / 256 (`:17`) | 256 / 256 (`:58`) |
| dropout | input 0.20, readout 0.10 (`:18`) | 0.2 shared (`:59`) |
| label smoothing | 0: not applicable, since `exact_ce_loss` has no smoothing argument (`:19`) | 0.0 (`:60`) |
| output head | tied item embedding + `output_bias` (`:20`) | untied `output_embed [K,d]` + `output_bias`, recentred after every step (`:61`) |
| architecture | — | d_model 256, 2 blocks, 4 heads, ff×4 (`:55`) |

Fable's summary (C8) is: AdamW fused, betas (0.9, 0.999), eps 1e-8, wd 1e-5 on ndim≥2, clip 1.0, peak 1e-3 for both
families, B=256, GRU dropout 0.20/0.10, SASRec 0.2 with an untied head and per-step recentering, and label smoothing 0.

The executed SASRec run *trained* in TF32 (`EXECUTED_RECIPES.md:65`). V4 overrides this with strict FP32.

## 15. Provenance of this contract

- **Files read:** the files listed in the DATA-03 packet, plus its named authorities:
  - `PKG/03_DATA_PROTOCOL.md`;
  - `PKG/12_ACCEPTANCE_TESTS.md`;
  - `G0_PLAN_FABLE_POSITION.md`.
- **Also read, for context:** `OWNER_SCOPE_V4.md` and the DATA-02 handoff.
- **Targeted lookups for exact citations or owners:**
  - `EDA/config.py:235,253-255` (grep);
  - the hash lines of `V1/FILES.json` and `V1/VERSION.json`;
  - `g0/feature_map/FEATURE_LOCALITY_MAP.json`, used for the row-by-row label cross-check;
  - the QA-01 task, its handoff and `g0/receipts_qa01/CODE_AND_FLAG_CHECKS.json`;
  - the task board.
- File hashes were computed with `sha256sum`. The arithmetic labelled DERIVED was computed in the local venv python,
  from the quoted numbers only.
- No data file was opened, and nothing was trained, installed or sent to the server.
- **DATA-03R1 revision.** It applied Fable's G0(b) C1(d), C2 and C8 and closed OI-03, OI-04, OI-06 and OI-07 from the
  QA-02 receipts. It read only the ruling, the four QA-02 receipts and `EXECUTED_RECIPES.md`. Every quoted value was
  checked programmatically against its receipt, and the F-1 consistency arithmetic is DERIVED from the quoted numbers.
- **Revision history:**
  - 2.0.0-draft.1 (DATA-03): initial version.
  - 2.1.0 (DATA-03R1): the F-1 values, the self-hash and the closed items.
