# 04 — Part 3: Features and history policy (S*, USER_HISTORY_L50, device-computability, pooled statistics)

Expert reviewer (FL / recommender systems), multi-expert audit 2026-09-29. READ-ONLY. Nothing on the server was touched. No
RETRO_TEST row, RESULT_ROWS, ranks or run-result file was opened. The only screen values quoted are the relative drops that
DECISION_LOG row 53 and DATA_PROTOCOL_LOCK_V2 publish. Paths are relative to `.local_ai/ppsi_v4/` unless stated otherwise.
`EDA/` = `next_item_eda/src/rees46_eda/`; `V1/` = `rees46_nextitem_v1/` (both under `eda/`).

Tags: **MEASURED** = read from a file (path:line); **REPORTED** = a number a project document states;
**INFERENCE** = my reasoning; **OPINION** = my judgement.

---------------------------------------------------------------------------------------------------------------------
## 1. What the project does (MEASURED)

### 1.1 The locked schema S*
- **Identity.** S* = `L3S_NEAR_DUPLICATE_SESSION-minus-G11+G12a+G3+G4+G7`, self-hash c53d47fe…, with 25 "information
  channels" (24 input columns plus the output catalogue). Sources: `contracts/DATA_PROTOCOL_LOCK_V2.json:99-137`;
  `contracts/derived_schemas/L3S_NEAR_DUPLICATE_SESSION-minus-G11+G12a+G3+G4+G7.json:64-101`.
- **Widths:** 5 token channels, 7 numeric, 2 flags, 6 user-context values and 4 masks (`…minus…json:140-146`).
- **Channels** (`…minus…json:64-99`):
  - **tokens (per event, [B,50]):**
    - `item_tokens`, `category_tokens`, `brand_tokens`;
    - `session_offset_tokens` (owner, DL 47): min(sessions back, 7) + 1;
    - `session_mode_main_category_tokens` (owner design G13, DL 48).
  - **numeric (per event):**
    - `log1p_price`;
    - `log1p_prev_raw_view_gap_s`;
    - `log1p_segment_elapsed_s`;
    - `prefix_representative_count`, `prefix_known_category_count`, `prefix_category_switch_count`;
    - `current_known_category_run_length`.
  - **flags (per event):** `price_missing`, `prev_gap_unavailable`.
  - **user context (per decision):**
    - `recent_category_match_share`, `recent_brand_match_share`;
    - `recent_mean_log_price`, `current_minus_recent_mean_log_price`;
    - `current_matches_session_mode`, `session_mode_share`.
  - **masks:** three `recent_*_supported` masks and `current_matches_session_mode_available`.
- **Event type is not a channel.** "purchase / cart derived signals | excluded | this is a view-only next-item task"
  (`V1/FEATURE_DICTIONARY.md:110`).
- **No calendar channel survived.** daypart, local weekday and is_weekend went at L1 (`featstat_v5_100/FEATSTAT_SUMMARY.md:207`).
  UTC weekday went at L3 (`:209`). G3 weekday_tokens and G4 utc_hour were dropped by the search (`company/DECISION_LOG.md:56`).
  So the +3 h timezone hypothesis no longer matters.
- **The pooled-first-seen item-age group G7 was dropped by the search** (`DECISION_LOG.md:56`).

### 1.2 How S* was chosen
1. **Statistics-only ladder (FEATSTAT-01), 53 → 30 channels.** It removed exact functional dependencies, dead or rare
   flags and near-duplicates (|ρ| ≥ 0.95) (`FEATSTAT_SUMMARY.md:201-215`), without training (owner speed directive,
   `DECISION_LOG.md:43`).
2. **Owner additions:** the session channels (`DECISION_LOG.md:51-52`).
3. **Grouped permutation importance** on a disjoint FS cohort (the top 32,768 selection keys), both families, 2 seeds.
   Permutation was LENGTH_MATCHED (`DECISION_LOG.md:49`). A group was dropped iff its importance was within noise for
   both families (`DECISION_LOG.md:45`).
4. **Retrain of S* and the mechanical acceptance rule:** worse family ≤ 1.0 % relative with UB95 ≤ 2.0 %.
   - Result (REPORTED): GRU −0.90 % (UB −0.15 %), SASRec −0.13 % (UB +0.86 %). PASS, no fallback
     (`DATA_PROTOCOL_LOCK_V2.json:139-161`; `DECISION_LOG.md:57`).
   - The owner closed the search (`DECISION_LOG.md:58`).

### 1.3 Transforms
- **Clean V2 is a load-time sidecar over the frozen v1 features** (`contracts/FEATURE_SCHEMA_CLEAN_V2.json`, "principles").
- **Every clip/z normaliser is fitted on pooled full TRAIN** (all users), identical byte-for-byte across arms (schema
  `:634`). The M1 and F-1 valid-only refits are recorded in `normalized_channels_v2`.
- **Numeric unknowns** become 0.0 after z, and a mask travels with them.
- **Production rule:** "shared statistics = full TRAIN as fitted (TRANSFORM_STATE_CLEAN_V2, unchanged), identical for every
  arm" (`DATA_PROTOCOL_LOCK_V2.json:167`).
- **D5 is still a [PROXY] decision.** The Opus cohort-refit position awaits the owner
  (`company/decisions/ADR-005-population-5-100-and-fast-feature-search.md:19,78-84`).
- **Nine S* channels are z-scored:** price, view gap, segment elapsed, the 4 prefix counts, recent mean price and the price
  difference. That is 9 × (clip_lo, clip_hi, mean, std) = 36 scalars (INFERENCE from the schema's
  `normalized_channels_v2` and the S* keep list).

### 1.4 Locality (the project's own map, re-counted for S*)
- **Class definitions:** LOCAL_EVENT / LOCAL_HISTORY / SHARED_CATALOGUE / SHARED_TRAIN_STATISTIC
  (`g0/feature_map/FEATURE_LOCALITY_MAP.md:14-18`).
- **Pooled objects are itemised** (MAP `:140-187`; schema `bootstrap_information_disclosure` B1–B9, `:637-646`).
- **My re-count for S*** (DERIVED from the schema channel labels plus the DATA-05 channels, which are pure functions of
  the user's own window, `code/v4hist/windows.py:45-57`):

  | class | count | channels |
  |---|---|---|
  | LOCAL_EVENT | 3 | price, price_missing, prev_gap_unavailable |
  | LOCAL_HISTORY | 18 | the other numeric, user-context and mask channels, and the 5 session channels |
  | SHARED_CATALOGUE | 2 | category, brand |
  | SHARED_TRAIN_STATISTIC | 1, plus the output catalogue | item_tokens |

- **What the pooled dependence actually amounts to in S*:**
  - (i) the item catalogue and vocabulary: support ≥ 5 **representative views** on full TRAIN, K = 158,486. The file
    holds only `product_id, representative_views, output_class, product_idx`, and no distinct-user count (read from the
    parquet schema of `V1/mappings/catalogue.parquet`).
  - (ii) the category, brand and main-category vocabularies: 660 / 3,953 / 13 values (parquet row counts).
  - (iii) the 36 normaliser scalars.
  - (iv) metadata recovery pooled across users. It is almost empty on the pool (`FEATSTAT_SUMMARY.md:15,18,20`):
    - brand_recovered fires on 0.00190 % of pool examples;
    - category_recovered and category_code_recovered fire on 0 %.
  - (v) θ0, which is random, not data.

### 1.5 History policy USER_HISTORY_L50
- **The rule:** "last 50 representative events of the user's whole prior stream, across sessions and partition fences,
  strictly causal, identical for TRAIN / VALIDATION / TEST" (`DATA_PROTOCOL_LOCK_V2.json:88-93`; owner DL 36,
  `DECISION_LOG.md:40`). It is not to be reopened (`DECISION_LOG.md:41`).
- **Implementation:** window = rows [max(user_start, r−49), r] of the user's retained stream (`code/v4hist/windows.py:20-27`;
  `code/v4hist/policy.py:3-5,41-63`).
- **Causality is proven exhaustively on the pool** (0 violations):
  - 28,623,386 TRAIN decisions (568.8 M window rows);
  - 5,823,491 VALIDATION decisions (`company/handoffs/DATA-05.md:273`).
- **Identical across arms.** Every launcher passes the same policy to both providers:
  - `code/v4int/launch_central.py:601-603`, `launch_fl.py:1912-1914,1935-1937`, `launch_device.py:928`;
  - `c_full.py:248-250`, `pretrain_small.py:53`;
  - `code/v4base/run.py:275-277`, `code/v4trunc`-side `launch_trunc.py:590`.
- **Window lengths** (`DATA-05.md:305-310`):

  | decisions | mean | at the cap of 50 |
  |---|---|---|
  | TRAIN | 19.87 | 11.35 % |
  | VALIDATION | 39.98 | 52.31 % |

  86.98 % of VALIDATION rows now see TRAIN-period events (`:315`).
- **The user-context and masks are NOT recomputed from the window.** "user_context / user_context_masks are the decision
  row's values, unchanged" (`code/v4hist/windows.py:26`; `DATA-05.md:110-111`).
  - They are v1 aggregates over the ≤ 100 most recent earlier events of a **partition-scoped replay stream**:
    "TRAIN replay TRAIN only; VALIDATION_SELECT replay TRAIN, then this partition's earlier events; VALIDATION_CONFIRM
    replay TRAIN, then this partition's …; TEST replay TRAIN, then this partition's earlier events … the observations
    between fences are intentionally omitted" (`EDA/pipeline6.py:1-13,42-47`; window = 100, `EDA/userctx.py:156`).
  - Likewise `log1p_prev_raw_view_gap_s` "is partition-scoped, skips barrier blocks, and at a VS→VC fence it measured
    624,176 s where the real gap was 160 s" (`DATA-05.md:46-47`).
  - The segment-relative channels (segment elapsed, the prefix counts and the run length) restart at every model segment,
    and segments also break at every partition fence (`DATA-05.md:410-419`; `DECISION_LOG.md:39`).

### 1.6 How features enter the model (SASRec v2, the winning family)
- **Input side only.** item_embed + pos_embed + a gated side residual (the side embeddings and Linear(numeric + flags) →
  side_proj) + a gated user residual (Linear(uctx + masks) → user_proj) (`code/v4models/sasrec_v2.py:196-223,307-318`).
- **The output is an untied ID table** [K, d] plus a bias (`sasrec_v2.py:202-203,446`). No attribute reaches the output side.
- **Positions:** `pos = ar` (0 … T−1) counted from the **oldest** window event, right-padded. So the current event's position
  equals the window length minus 1 (`sasrec_v2.py:347,370`; `windows.py:23-26`).
- **Parameter budget:** side information is < 0.3 % of parameters.
  - SASRec L0 has 83.09 M parameters; the ID-only anchor has 82.90 M (`FEATSTAT_SUMMARY.md:206,215`).
  - The model is dominated by the two 158 k-row item tables.

### 1.7 The pooled bootstrap block (CLIENT_CONTRACT C5.6 / §6)
- **Definition:** θ0 per seed, the catalogue, vocabularies, normalisers, recovery and first_seen, "given identically to C,
  FA, FP, PF, CPF and LO" (`contracts/CLIENT_CONTRACT_V1.md:215-237`).
- **Disclosed limits:** a central feature pipeline, and "not privacy claims" (`:201-213`).
- **The DP accounting explicitly lists the block as unaccounted** (`contracts/PREG1_AMENDMENT_06D.md:62-64`).

---------------------------------------------------------------------------------------------------------------------
## 2. State of the art (literature and real systems)

All citations below were verified by web search on 2026-09-29.

### Side information in sequential recommenders
- **Kang & McAuley, "Self-Attentive Sequential Recommendation," ICDM 2018.** DOI 10.1109/ICDM.2018.00035.
  - Positions: the reference sampler **left-pads**, so the most recent item always sits at the last position
    (`github.com/kang205/SASRec/blob/master/sampler.py`, fetched).
  - Positional indices therefore encode recency.
- **Hidasi, Quadrana, Karatzoglou & Tikk, "Parallel RNN Architectures for Feature-rich Session-based Recommendations,"
  RecSys 2016.** DOI 10.1145/2959100.2959167.
- **Quadrana et al., "Personalizing Session-based Recommendations with Hierarchical RNNs," RecSys 2017.**
  DOI 10.1145/3109859.3109896. This is the cross-session user history that DL 47 cites.
- **Zhang et al., "Feature-level Deeper Self-Attention Network for Sequential Recommendation" (FDSA), IJCAI 2019,
  pp. 4320-4326.** https://www.ijcai.org/proceedings/2019/600. It adds a separate feature-level self-attention stream.
- **Zhou et al., "S3-Rec," CIKM 2020.** DOI 10.1145/3340531.3411954. It pre-trains with item-attribute mutual information.
- **Rashed, Elsayed & Schmidt-Thieme, "Context and Attribute-Aware Sequential Recommendation via Cross-Attention" (CARCA),
  RecSys 2022.** DOI 10.1145/3523227.3546777. It uses attributes and context on the target (output) side through
  cross-attention.
- **de Souza Pereira Moreira et al., "Transformers4Rec," RecSys 2021.** DOI 10.1145/3460231.3474255. It is the production
  (NVIDIA Merlin) practice of feature normalisation and aggregation blocks for session features.

### Leakage and evaluation rigour
- **Ji, Sun, Zhang & Li, "A Critical Study on Data Leakage in Recommender System Offline Evaluation," ACM TOIS 2023.**
  DOI 10.1145/3569930. Global-timeline splits: every feature and statistic must be available at prediction time.

### Federated or private feature engineering and vocabularies
- **Kairouz, McMahan et al., "Advances and Open Problems in Federated Learning," FnTML 14(1-2), 2021.**
  DOI 10.1561/2200000083.
- **Ramage & Mazzocchi, "Federated Analytics: Collaborative Data Science without Data Collection," Google AI Blog, May 2020.**
  https://ai.googleblog.com/2020/05/federated-analytics-collaborative-data.html
- **Apple Differential Privacy Team, "Learning with Privacy at Scale," Apple ML Journal, 2017** (machinelearning.apple.com).
- **Zhu, Kairouz, McMahan, Sun & Li, "Federated Heavy Hitters Discovery with Differential Privacy," AISTATS 2020,
  PMLR 108:3837-3847.** https://proceedings.mlr.press/v108/zhu20a.html
- **Gopi, Gulhane, Kulkarni, Shen, Shokouhi & Yekhanin, "Differentially Private Set Union," ICML 2020, PMLR 119:3627-3636.**
  https://proceedings.mlr.press/v119/gopi20a.html. It covers DP vocabulary extraction.
- **Chen, Mathews, Ouyang & Beaufays, "Federated Learning of Out-of-Vocabulary Words," arXiv:1903.10635 (2019).**
- **Sun, Kairouz, Sun, Gascon & Suresh, "Private federated discovery of out-of-vocabulary words for Gboard,"
  arXiv:2404.11607 (2024).**
- **Andrew, Thakkar, McMahan & Ramaswamy, "Differentially Private Learning with Adaptive Clipping," NeurIPS 2021.** It shows
  DP quantile estimation with a negligible privacy budget.
- **Xu, Zhang, Andrew, Choquette, Kairouz, McMahan, Rosenstock & Zhang, "Federated Learning of Gboard Language Models with
  Differential Privacy," ACL 2023 Industry, pp. 629-639.** https://aclanthology.org/2023.acl-industry.60/

### Browsers computing features locally
- **Chrome Topics API** (Privacy Sandbox docs, https://privacysandbox.google.com/private-advertising/topics/web):
  - topics are computed on device from browsing history with a downloaded TFLite model and a human-curated taxonomy;
  - the default epoch is one week.
- **Protected Audience k-anonymity:** creatives must reach k = 50 users over 30 days
  (https://developers.google.com/privacy-sandbox/private-advertising/protected-audience-api/k-anonymity).

### Data minimisation
- **Biega, Potash, Daumé III, Diaz & Finck, "Operationalizing the Legal Principle of Data Minimization for Personalization,"
  SIGIR 2020.** DOI 10.1145/3397271.3401034. It proposes performance-based minimisation (GDPR Art. 5(1)(c)): the quality loss
  may be small but disparate across users.

### Consensus of these sources (INFERENCE)
- Production FL and private-analytics systems treat three things as a **separately privatised or public bootstrap**, not
  as free information:
  - vocabularies (heavy hitters / DP set union / OOV discovery);
  - normalisation or clipping statistics (federated analytics / DP quantiles);
  - taxonomies (Topics).
- Thresholds count **distinct users or devices**, not events.
- Sequential recommenders with rich side information commonly let attributes help the **target** side (CARCA, S3-Rec),
  which matters most when ID embeddings are poorly trained, as in FL or on small local models.

---------------------------------------------------------------------------------------------------------------------
## 3. Comparison

### 3.1 Strengths (MEASURED unless marked)
- **S1 — provenance.** The per-channel provenance and locality map is exemplary.
  - Every channel has a locality class, a pooled bit and a code citation (MAP; schema `channels`).
  - Pooled objects are enumerated B1–B9 (schema `:637-646`).
  - The project states outright that "privacy of any kind" is not established (schema `:769`).
  - I have not seen this level of feature provenance in published FL-recsys benchmarks (OPINION).
- **S2 — S* is almost fully device-computable.**
  - 21 of the 24 inputs are LOCAL_EVENT or LOCAL_HISTORY (§1.4).
  - The one feature that truly needed a pooled per-item table (item age from pooled first_seen, G7) and the timezone
    hypothesis both left the schema through the search.
  - What remains is a catalogue, three vocabularies and 36 scalars. That is the "public catalogue + one federated-analytics
    round" assumption production systems make (INFERENCE).
- **S3 — the feature search is well designed.**
  - A disjoint FS cohort (`company/handoffs/FS-01.md:37-42,63-73`).
  - The worse family judges (`DECISION_LOG.md:45`).
  - LENGTH_MATCHED permutation avoids the PAD artefact (`:49`).
  - A retrain confirms the joint drop (point and UB rule).
  - Seeds 2026 and 2027.
  - The margin is pre-declared before the reads.
- **S4 — the history policy.** It is strictly causal, one builder for train and eval, proven exhaustively
  (`DATA-05.md:271-276`), and passed identically to every arm (§1.5). Storage on a device is trivial: 50 events plus a
  100-event aggregate window, a few KB (INFERENCE).
- **S5 — normaliser hygiene.**
  - M1 and F-1 valid-only refits, mask-and-zero discipline, state-machine tests T08–T14 (schema `required_tests`).
  - The full-TRAIN vs pool refit difference is measured (`FEATSTAT_SUMMARY.md:172-199`).
- **S6 — fairness.** The bootstrap block is one hash-bound object shared by every neural arm (`DATA_PROTOCOL_LOCK_V2.json:167`;
  CLIENT_CONTRACT `:220`). It is correctly listed as outside the DP accountant (06D `:62-64`).

### 3.2 Weaknesses and errors
- **E1 — the history policy is only partly "all prior events" (MEASURED, not disclosed for TEST).**
  - The owner requirement DL 35(d) is "all of a user's previous events must enter VALIDATION/TEST prediction"
    (`DECISION_LOG.md:39`). Under USER_HISTORY_L50 it is met by the **sequence** channels only.
  - The S* **user-context** channels are still v1 partition-scoped replays (`EDA/pipeline6.py:1-13,42-47`; `windows.py:26`):
    the 2 match shares, 2 price aggregates and 3 support masks.
    - For VALIDATION_CONFIRM decisions they omit the VALIDATION_SELECT events. This is disclosed as B9 in
      `contracts/DATA_CONTRACT_CLEAN_V2.md:592-593`, written before L50.
    - For **every RETRO_TEST decision they omit the whole VALIDATION week**. This is not disclosed anywhere I could find.
  - The same holds for `log1p_prev_raw_view_gap_s` (`DATA-05.md:46-47`).
  - The segment-prefix channels restart at the fences.
  - Consequences (INFERENCE):
    - (a) The TEST inputs contain a one-week "hole" that TRAIN inputs never contain. VALIDATION_SELECT, which is most of
      VALIDATION, has none. So TEST is mildly out of distribution relative to what selection saw.
    - (b) A real browser would have no hole, so absolute device and cloud levels on TEST are slightly pessimistic.
    - (c) Fairness across arms is preserved to first order, because every arm reads the same stored values. Arms that
      differ in robustness (C vs LO/FT, BIG vs SMALL) could be affected differently, so exact fairness is not guaranteed.
- **E2 — positions no longer encode recency (MEASURED code + INFERENCE effect).**
  - SASRec v2 counts positions from the oldest window event (`sasrec_v2.py:347,370`). Kang & McAuley left-pad, so recency
    sits at a fixed position.
  - Under SESSION_L50 the index meant "position in session". Under USER_HISTORY_L50 it means "distance from an arbitrary
    truncation point".
  - The current event sits at position 49 on 52 % of VALIDATION decisions but only 11 % of TRAIN decisions
    (`DATA-05.md:308,310`).
  - `session_offset_tokens` and the gap channels partly restore recency.
  - The effect is identical across SASRec-based arms (fair). It is a modelling weakness that the thesis should name.
    Part 4 (backbone contest) should note that the GRU is unaffected.
- **E3 — the catalogue threshold counts events, not people (MEASURED).**
  - Support is ≥ 5 **representative views**, not distinct users, and the file has no user count (§1.4).
  - A single user can create a catalogue entry. So the catalogue is not even k-anonymous.
  - Production thresholds count distinct users over a window (Protected Audience k = 50), or are privatised (heavy hitters,
    DP set union).
  - The catalogue is also interaction-derived rather than "public". A real Edge feature would use the retailer's full
    published catalogue or a DP-discovered head.
  - The utility impact is small: K_A vs pool-only K_B changes 0.12–0.14 % of items and targets (`FEATSTAT_SUMMARY.md:194`).
- **E4 — stale framing (MEASURED).**
  - CLIENT_CONTRACT §6 still says "34 of the 53 kept channels depend on other users' pooled interactions" (`:223-224`),
    which is Clean V2, not S*.
  - It still says "Centrally trained winner weights never initialize FL or LO" (`:240`), yet FT_C and FA_WARM (06A/06K)
    do exactly that by design.
  - 06B's L6 row says RULE uses "shared catalogue/θ0 only" (`PREG1_AMENDMENT_06B.md:59`), yet RULE, KNN_EMB and POP use
    MAIN-TRAIN popularity tables (`code/v4base/rule.py:9-11,117-119`; `knn_emb.py:25-29`), which is pooled interaction data.
  - The bootstrap is identical across the neural arms, but **it is not the same information set for every arm**. The
    thesis needs a per-arm information table.
- **E5 — D5 (full-TRAIN normalisers) is still an unratified PROXY decision (MEASURED; ADR-005 `:19,78-84`).**
  - Realism-wise, full-TRAIN statistics include the UNSEEN and U_NEW users' TRAIN rows.
  - The measured size is small but not zero: most channels |Δz| ≤ 0.04; `prefix_representative_count` max |Δz| 2.85 because
    the clip is 140 vs 61 (`FEATSTAT_SUMMARY.md:176-192`).
- **E6 — S* was chosen for the cloud models only (INFERENCE).**
  - It was selected under central training of the BIG models on the FS cohort, then applied to C_SMALL, the FL arms,
    LO/FT and the DP arms.
  - Side information matters most when ID embeddings are under-trained: FL (CALIB FA ≈ 0.026 vs central ≈ 0.11–0.12,
    REPORTED in REALISM_AUDIT §0), small models and cold users.
  - Removing G7 (a popularity/recency proxy) was realism-positive, but could cost FL more than it cost the central models.
    Nothing measures this.
- **E7 — device-observable signals are absent (MEASURED + OPINION).** These are deliberate, but they bear on the Edge
  question:
  - cart and purchase events (`V1/FEATURE_DICTIONARY.md:110`) and dwell time;
  - above all, **cross-site** history. Here the device sees exactly what the single retailer's server sees.
  - The real asymmetry is that a browser sees more than any one retailer while the cloud sees less (or must collect across
    sites). The simulation cannot represent it. This is a framing point, not an error.
- **E8 — the ID-only output side (MEASURED design).**
  - All attributes enter only as inputs. The 158 k × d ID tables are > 99.7 % of the parameters.
  - For an Edge-wide (multi-retailer) deployment, ID catalogues do not scale on device. Attribute-composed item
    representations (CARCA / FDSA / S3-Rec lines) are the realistic path.
  - This is a discussion item, not a fix now.

### 3.3 Unrealistic (flag, no change needed for validity)
- **The retailer taxonomy.** `category_id` is a retailer-internal key. A browser sees breadcrumbs or schema markup and would
  need a mapping to a shipped taxonomy (Topics-style). UNVERIFIED how Edge's shopping features identify product pages.
- **Static catalogue:** a frozen end-of-TRAIN catalogue for the whole evaluation week. It is symmetric across arms and
  already reported as OOV (EVALUATION_CONTRACT §4).
- **The normaliser fit is not simulated as a federated-analytics round.** It is a free central computation (disclosed as
  bootstrap, not modelled).

### 3.4 Missing
- A statement of exactly what the device must download for S*: catalogue, vocabularies and 36 scalars, with bytes.
- A count of the rows affected by E1.
- A per-arm information-provenance table (E4).

---------------------------------------------------------------------------------------------------------------------
## 4. Recommendations

**Governed-value status (MEASURED):**
- S* was locked after its governed value (FS_FINAL) was read (`DECISION_LOG.md:57-58`).
- USER_HISTORY_L50 is owner-locked and not reopened (`:40-41`).
- VALIDATION values have been read (the P3 contest and the CALIB screens).
- TEST is sealed.

So **no change to S*, the history policy or VALIDATION inputs is admissible for the primary analysis**. Only disclosures
and pre-registered SECONDARY or DESCRIPTIVE analyses remain possible. Where a pre-registration is needed, it must be written
before the TEST read (the "06L-type" amendment).

| ID | Item | Class | Action | Cost (INFERENCE) | Affects | Pre-registration needed / status |
|---|---|---|---|---|---|---|
| F1 | Disclose E1 precisely: the S* user-context (7 channels), the view gap and the segment-prefix channels are partition-scoped. VC decisions omit VS events; **RETRO_TEST decisions omit the entire VALIDATION week**. DL 35(d) "all previous events" holds for the sequence channels only. Correct any thesis sentence that says otherwise | **MUST** | ADD (text) | 0 GPU; 1 h writing | Interpretation of every TEST number; no run | Text-only registration before the TEST read (not yet registered; B9 in `DATA_CONTRACT_CLEAN_V2.md:592-593` covers VC only and predates L50) |
| F2 | Replace the Clean V2 "34 of 53" framing with the **S*-specific** locality statement (§1.4): 21/24 inputs local; pooled content = support-thresholded catalogue + 3 vocabularies (660/3,953/13) + 36 normaliser scalars + brand recovery on 0.0019 % of examples; θ0 random. State the bytes the device must download once | **MUST** (it is the direct answer to "what can a device compute") | CHANGE (text) | 0; 1 h (bytes from file sizes) | Thesis Edge chapter; no run | None (description of locked facts) |
| F3 | A per-arm information-provenance table: which arms use the bootstrap block only (FA/FP/PF/LO/DP); centrally trained weights (FT_C, FA_WARM, KNN_EMB, device-served C_SMALL); MAIN-TRAIN popularity (RULE, POP, KNN fallback). Fix the 06B L6 row (`:59`) and note that CLIENT_CONTRACT `:240` is superseded by 06A/06K | **MUST** (the fairness claim "identical bootstrap" is true only for the neural arms) | CHANGE (text) | 0; 1 h | Ladder/privacy framing | Text registration before the TEST read |
| F4 | Counts-only census of E1: the share of VC and RETRO_TEST eval decisions whose recent-100 replay or previous-gap skips ≥ 1 real prior event, and the median number skipped. VALIDATION now; TEST under the INT-07 gate, counts only, logged in TEST_EXPOSURE_REGISTER | SHOULD | ADD | CPU ≈ 1–2 h + a Sonnet task | Quantifies F1; no model run | Counts-only (the precedent is DATA-05 entry 4); register the TEST-exposure entry |
| F5 | SECONDARY sensitivity: recompute the 7 user-context channels and the view gap on a **continuous** stream (TRAIN→VS→VC→TEST) as a new sidecar for RETRO_TEST decisions (v1 untouched). Evaluate the headline arms (C_BIG, C_SMALL, FA, FT_C, LO, RULE) evaluation-only on the same rows; report the retention ratios under both inputs. The primary stays as registered | SHOULD | ADD | ≈ 1 day data engineering + review; ≈ 3 min eval per arm | New descriptive rows only; nothing finished changes | **Yes: must be registered before the TEST read** (label SECONDARY/DESCRIPTIVE; not yet registered) |
| F6 | Disclose E3: catalogue support = ≥ 5 **views**, not distinct users, so it is not k-anonymous. Name the production alternatives: the full public catalogue, DP heavy hitters / DP set union, distinct-user thresholds | SHOULD | ADD (text) | 0 | Framing of L1–L6 "only the catalogue is shared" | Text |
| F7 | NICE census: the number of catalogue items with < k distinct TRAIN users (k = 2, 5, 50), from TRAIN only | NICE | ADD | CPU ≈ 1 h | Supports F6 | None (TRAIN counts) |
| F8 | D5: obtain the owner's ratification of full-TRAIN normalisers, or write it in the thesis as a proxy decision. Describe the normaliser fit as "one un-simulated federated-analytics round (9 means/variances + clip quantiles)", citing federated analytics / DP quantiles | **MUST** (an unratified proxy item in the locked protocol) | KEEP + DISCLOSE | 0 GPU; one owner question | None | Ratification only (ADR-005 `:84`) |
| F9 | NICE: an offline DP-noised normaliser sensitivity (Gaussian noise on the 36 scalars at a stated ε; report max/mean \|Δz\| as FEATSTAT §7 did). No training | NICE | ADD | CPU < 1 h | Supports F8 | None |
| F10 | Disclose E2: SASRec positional indices count from the oldest window event (`sasrec_v2.py:347,370`), unlike left-padded SASRec. Give the train/eval length shift (19.87 vs 39.98 mean; 11.35 % vs 52.31 % at the cap). Note that the GRU is unaffected. A recency-aligned variant is future work (a change now would REDO all runs) | SHOULD | ADD (text) | 0 | Backbone-contest interpretation (flag to Part 4) | Text |
| F11 | Disclose E6: S* was selected under central BIG training only and not validated for the FL, SMALL or device arms. Optional exploratory FA-CALIB pair L0S vs S* (2 × ≈ 2.7 h) registered as EXPLORATORY; S* is not changed whatever it shows | NICE (the run) / SHOULD (the text) | ADD | 0 / ≈ 5.4 GPU-h | None finished | Exploratory registration if run |
| F12 | Disclose E7/E8: view-only (no cart/purchase); single retailer (no cross-site history, the browser's real advantage); retailer-internal taxonomy; ID tables > 99.7 % of parameters, so they do not scale to Edge-wide catalogues. Point to attribute-aware models (CARCA/FDSA/S3-Rec) as the device-scalable path | SHOULD | ADD (text) | 0 | Discussion / device guidelines | Text |
| F13 | Keep USER_HISTORY_L50 as locked: causal, identical across arms, realistic storage; and keep S* (search design sound) | — | **KEEP** | 0 | — | Locked (DL 36, DL 53) |
| F14 | Keep the permutation-importance findings G8 (mixed sign) / G9 / G12b as THESIS FINDINGS with their CIs (DL 54) | — | KEEP | 0 | — | Recorded |

---------------------------------------------------------------------------------------------------------------------
## 5. Score: 7 / 10

**Why not lower.** The feature engineering is careful and well documented. Every channel's provenance and locality is
established from code. Pooled statistics are enumerated. Normaliser defects (M1, F-1) were found and fixed with receipts.
The feature search used a disjoint cohort, the worse of two families, length-matched permutation and a retrain
confirmation. The resulting S* is, in substance, device-computable: a local event/history stream plus a downloaded catalogue,
three vocabularies and 36 scalars. The history policy is strictly causal, exhaustively proven and passed identically to
every arm. This already answers "what could an Edge client compute locally", and answers it well.

**Why not higher.**
- **The "all previous events" property claimed for USER_HISTORY_L50 is false for the S* user-context, gap and prefix
  channels.** They keep v1's partition-scoped replay, which omits the whole VALIDATION week for every RETRO_TEST decision.
  This is undisclosed for TEST and makes TEST inputs slightly unlike both TRAIN and a real browser.
- **The SASRec position semantics silently changed under the new policy.**
- **The catalogue threshold counts views, not users.** The "public catalogue" is interaction-derived and not k-anonymous.
- **D5 is still an unratified proxy item.**
- **The published framing is stale:** "34/53", "never initialise from central weights", and RULE's L6 row.

None of these invalidates the paired contrasts, because inputs are identical across arms. But three of them must be fixed
in text before the TEST read, and one (F5) is the only way to put a number on the E1 effect.

---------------------------------------------------------------------------------------------------------------------
## 6. UNVERIFIED items and open questions
- **U1.** How Microsoft Edge's shopping features identify product pages and attributes (product id, category, price).
  UNVERIFIED; not cited.
- **U2.** Whether the S* 36-scalar and catalogue bytes have been counted anywhere. `fl/comm.py:56` counts model buffers only,
  per the REALISM_AUDIT C2 (REPORTED, not re-read).
- **U3.** The magnitude of E1 on TEST. Unknown by design (TEST sealed); F4 would measure it counts-only.
- **U4.** The magnitude of E2 (positions). Untested; an ablation would need retraining (future work).
- **U5.** Whether the owner has ratified D5 in a channel I did not read. ADR-005 `:84` and the 2026-09-27 RESUME_HERE list it
  as open, and I found no later ratification row in the DECISION_LOG.
- **U6.** BERT4Rec (Sun et al., CIKM 2019) and the FL benchmarks FedScale / LEAF / FLUTE were not needed for this part and
  were not re-verified. They are not cited here.
- **Open question to the lead.** Should F1/F3 (text) and F5 (secondary) go into one pre-TEST amendment together with the
  REALISM_AUDIT M2 disclosures? It must precede the ALL_METHODS_LOCK / TEST read.
