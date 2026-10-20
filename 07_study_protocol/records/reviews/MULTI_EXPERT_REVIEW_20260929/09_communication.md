# 09 — Communication efficiency (payload bytes, FA_Q8, encoder-only uploads, bits-vs-quality curves)

Expert: part 8 of the multi-expert review. Viewpoint: communication-efficient FL, production cross-device FL, and federated recommendation.
READ-ONLY. I read contracts, code, handoffs, reviews, the DECISION_LOG (DL) and the model inventory. I read no RETRO_TEST row, no RESULT_ROWS,
no ranks and no eval-row file. I ran no project script. Every byte figure below is either quoted from a file or worked out by arithmetic from
registered parameter counts. None of them is a result. Tags: MEASURED (file:line), REPORTED (a project document's own claim), INFERENCE,
OPINION. Paths are relative to `.local_ai/ppsi_v4/`.

---------------------------------------------------------------------------------------------------------------------------------------

## 1. What the project does (MEASURED)

### 1.1 The model and its payload
- The SMALL model is SASREC_D64_B2 with **20,710,048 shared parameters** (`contracts/ARCHITECTURE_LOCK.json:172`; DL 125, line 129). Amendment
  03 §1 wrote "≈ 20.4 M" (`contracts/PREG1_AMENDMENT_03.md:16`). Amendment 06 §1 corrects this to 20,710,048 (`PREG1_AMENDMENT_06.md:9`).
- **The per-visit FP32 payload is 165,680,392 B** (down plus up, header included) (`ARCHITECTURE_LOCK.json:259`), under the 170 MB cap
  (`PREG1_AMENDMENT_03.md:11-13`). BIG = 671,054,112 B (`ARCHITECTURE_LOCK.json:258`).
- **Parameter breakdown** (INFERENCE by exact arithmetic; the code's formulas were applied to the D256 inventory
  `code/v4models/inventory/SASREC_CLEAN_V2_MANIFEST_INVENTORY.json` at d = 64; the sum matches the registered 20,710,048 exactly):
  | Tensor group | Params | Share | FP32 bytes |
  |---|---|---|---|
  | `item_embed` (158,489 × 64; `v4models/sasrec_v2.py:196`) | 10,143,296 | 49.0 % | 40.57 MB |
  | `output_embed` (158,486 × 64, untied; `:202`) + `output_bias` (`:203`) | 10,301,590 | 49.7 % | 41.21 MB |
  | side embeddings (category 663, brand 3,956, main-cat 16, daypart 8, weekday 10 tokens; fixed widths) | 148,008 | 0.71 % | 0.59 MB |
  | 2 transformer blocks, pos_embed (50 × 64), norms, side/user projections, gates | 117,154 | 0.57 % | 0.47 MB |
  | **Total** | **20,710,048** | 100 % | 82.84 MB per direction |
  **The two item tables are 98.7 % of the model.**

### 1.2 How bytes are counted (`code/fl/comm.py`)
- Everything is counted "virtually": tensors never leave the process (`comm.py:1-11`). Per visit, download = every SHARED tensor once
  (tied aliases are deduplicated, `comm.py:36-46`). Upload = the same key set plus an 8-byte int64 `n_consumed` header
  (`comm.py:33, 70-78`). Private PF state costs 0 B (`comm.py:74`). FIXED/SERVER_COPY buffers are a one-time bootstrap (`comm.py:76`).
  Retried visits re-download, and those bytes are counted separately (`comm.py:10, 93-96`). The ledger checks that an upload is exactly
  the shared key set (`comm.py:138-143`).
- The run record is `v4int/launch_fl.py:885-918` `bytes_record`: per_client = visits × (down + up) for surviving visits. The fleet total
  comes from the ledger. Disclosure: "dropped clients … are not charged a download" (`:900`).
- **The "charged bytes" decision (DL 218, line 223; DL 221, line 226; FIX-DELTA-02 M4).** The registered per-client figure (06A §3) stays
  visits × (down + up). The 06I §7b column adds **only the download of each dropped visit**, since a dropped client received the model and
  never uploaded (`tools/retention_report.py:1023-1044`, `DISCLOSURE_BYTES_06I_7B` :311-317). The variant that also charges an upload for
  a dropped visit is DESCRIPTIVE. **Correction to the brief's wording:** the decision is *not* "charged bytes = downloads only". Surviving
  visits are still charged down + up. Only the dropped-visit surcharge is download-only.

### 1.3 FA_Q8 (`code/fl/quant.py`; 06B §5 `PREG1_AMENDMENT_06B.md:69-74`; 06D §6 `PREG1_AMENDMENT_06D.md:86-88`)
- The codec quantises the upload delta Δ_k = upload_k − θ_r,k per SHARED tensor. It is symmetric, with scale = max|Δ_k|/127 sent as FP32,
  q = clamp(floor(Δ/s + U), ±127) as int8, and stochastic rounding U ~ U[0,1) seeded by `derive_seed(seed,"q8",round,client)`
  (`quant.py:49-64`). It is unbiased. The server dequantises θ_r + q·s and aggregates exactly as FA (`quant.py:140-158`).
- Bytes: up = Σ numel + 4 × n_tensors + 8 (`quant.py:87-98`). The download stays dense FP32 (`quant.py:101-117`). The code refuses the
  process pool (in-process only), DP, PF and μ ≠ 0 (`quant.py:171-181`). It works with any server optimiser, because it is only a
  channel (`v4int/launch_fl.py:1106-1108`).
- An exact zero stays exactly zero: floor(0 + U) = 0 for U ∈ [0,1) (`quant.py:61`). **So Q8 keeps the row support of the delta.**
- INFERENCE: with 51 shared tensors, Q8 up = 20,710,260 B and the Q8 visit = **103.55 MB** (62.5 % of FA). The download is 80 % of Q8 traffic.
- K14 = R(FA_Q8 / FA) is a confirmatory contrast of the tertiary tier (`PREG1_AMENDMENT_06B.md:45`; 06D §4 hierarchy).
  **The FA_Q8 MAIN runs were stopped before any mark ("never evidence", DL 183, line 188). No FA_Q8 value exists.**

### 1.4 S8 — FA_WARM_FROZEN (06K §3, `PREG1_AMENDMENT_06K.md:29-35`; FL-12b)
- `item_embed.weight`, `output_embed` and `output_bias` become FIXED buffers: not trained, not uploaded, not aggregated
  (`code/fl/runtime.py:128, 145-192`). They come from θ_pre, a model pretrained on 1.22 M non-MAIN "opted-in" users.
- Per-visit bytes count only the uploaded parameters. The frozen tables are charged once per visited client (`comm.py:55-67`,
  `launch_fl.py:902-917` `per_client_incl_frozen_once`).
- INFERENCE (arithmetic): the uploaded set is **265,162 params → 1.06 MB per direction, 2.12 MB per visit** (a 78× cut from 165.68 MB).
  The frozen tables are **81.78 MB once**. 06K says "≈ 0.4 M" uploaded (`PREG1_AMENDMENT_06K.md:31`) and the ladder research says
  "≈ 1.7 MB per direction". Both are high by about 1.6×: the 0.42 M figure is 20.71 − 20.29 M, which leaves `output_bias` and the three
  extra input rows in the "encoder".
- **Side embeddings (brand 3,956 rows, category 663 rows) are NOT frozen** (`runtime.py:128` lists only the three item keys). They are
  56 % of the S8 upload.

### 1.5 The bits-vs-quality curve (06A §3, `PREG1_AMENDMENT_06A.md:25-26`; `tools/retention_report.py:1982-2014`)
- R vs cumulative bytes per client is built from the **12 in-run VALIDATION marks** of every FL run (no extra runs). Each point uses
  denominator C_BIG PRACTICAL_BEST of the same seed. Bytes at a mark = the run's per-client mean × round/rounds ("PRORATED_BY_ROUND
  (INFERENCE)"). There are three series: registered, charged-dropped, and up+down (DESCRIPTIVE). Label DESCRIPTIVE.
- `FL_ARMS` = FA, FP, PF, CPF, FA_1024, DP8/4/1, FA_Q8, FA_WARM (`retention_report.py:154`). **`FA_WARM_FROZEN` is not in `ARMS`**
  (`:125-127`), and any entry with an unknown arm is refused (`:764-765`). `_fl_meta` reads `per_client.mean` only
  (`:1047-1056`), never `per_client_incl_frozen_once`.

### 1.6 Governance state (for the pre-registration column)
- MEASURED from DL 218/221 and `company/registers/FL_IMPROVEMENT_LADDER.md`: the only FL values read so far are the CALIB screens S0, S1a
  and S1b. S2 is running. **No MAIN FL value, no FA_Q8 value and no S7/S8 value has been read.** The ANA freeze is pending.
- Consequence: everything below can still be registered legitimately. The rule is to do it before the S4 selection / MAIN FL launch for
  runs, and before the ANA freeze for analysis code.

---------------------------------------------------------------------------------------------------------------------------------------

## 2. State of the art (literature and real systems; each entry verified by web search today unless marked)

### 2.1 Production payloads and eligibility
- **Gboard** (Hard et al., *Federated Learning for Mobile Keyboard Prediction*, arXiv:1811.03604, 2018; PDF p.2–3 read today):
  - "Overall, 1.4 million parameters … After weight quantization, the model shipped to Gboard devices is 1.4 megabytes."
  - The vocabulary is capped at V = 10,000 *because* "client device requirements … limit the vocabulary and model sizes".
  - Tied input/output embeddings are used "to reduce the model size".
  - Eligibility: ≥ 2 GB RAM, charging, **un-metered network**, idle. 100–500 client updates per round; server Nesterov momentum 0.9.
  - https://arxiv.org/abs/1811.03604
- **Google FL system** (Bonawitz et al., *Towards Federated Learning at Scale: System Design*, MLSys 2019, p.4 read today): jobs run only
  when the device is "idle, charging, and connected to an unmetered network such as WiFi". Update sizes "can range … from kilobytes to
  tens of megabytes". https://mlsys.org/Conferences/2019/doc/2019/193.pdf
- **Browser precedent** (Chrome built-in AI / Prompt API docs, developer.chrome.com/docs/ai/prompt-api, read today): Gemini Nano is
  downloaded once, needs "at least 22 GB of free space", and needs "unlimited data or an unmetered connection … for initial model
  download only". A browser vendor already ships GB-scale models *once* over unmetered links. Its exact model size is not stated.
  Third-party posts say ≈ 4 GB (UNVERIFIED).
- **Taobao / Alibaba submodel FL** (Niu, Wu, Tang, Hua, Jia, Lv, Wu, Chen, *Billion-scale federated learning on mobile clients: a submodel
  design with tunable privacy*, MobiCom 2020): clients download and upload only the embedding rows they need. The index set is protected
  by randomized response, secure aggregation and a private set union. This is the closest real-system precedent for our e-commerce
  item-table problem. https://dblp.org/rec/conf/mobicom/NiuWTHJLWC20.html

### 2.2 Compression methods
- **Konečný, McMahan, Yu, Richtárik, Suresh, Bacon**, *Federated Learning: Strategies for Improving Communication Efficiency*,
  arXiv:1610.05492 (2016). Uplink only: structured updates (low-rank, random mask) and sketched updates (quantisation, random rotations,
  subsampling). https://arxiv.org/abs/1610.05492
- **Suresh, Yu, Kumar, McMahan**, *Distributed Mean Estimation with Limited Communication*, ICML 2017, PMLR 70:3329-3337. Naive stochastic
  rounding has MSE Θ(d/n). A structured random rotation before quantisation gives O(log d / n), and better coding gives O(1/n).
  https://proceedings.mlr.press/v70/suresh17a.html
- **Alistarh, Grubic, Li, Tomioka, Vojnovic**, *QSGD*, NeurIPS 2017. Uses **bucketed** scales: 8-bit with bucket 512 recovers full
  accuracy, and a scale is stored per bucket to control variance. https://proceedings.neurips.cc/paper/2017/hash/6c340f25839e6acdc73414517203f5f0-Abstract.html
- **Reisizadeh, Mokhtari, Hassani, Jadbabaie, Pedarsani**, *FedPAQ*, AISTATS 2020, PMLR 108. Combines periodic averaging, partial
  participation and quantised uploads. https://proceedings.mlr.press/v108/reisizadeh20a.html
- **Caldas, Konečný, McMahan, Talwalkar**, *Expanding the Reach of Federated Learning by Reducing Client Resource Requirements*,
  arXiv:1812.07210 (2018). Adds **lossy compression of the server-to-client model** and Federated Dropout. Reports up to 14× less
  download and 28× less upload. https://arxiv.org/abs/1812.07210
- **Sattler, Wiedemann, Müller, Samek**, *Robust and Communication-Efficient Federated Learning From Non-i.i.d. Data* (STC), IEEE TNNLS
  31(9):3400-3413, 2020. Uses top-k, ternarisation and Golomb coding, and compresses **downstream too**. It relies on error feedback,
  i.e. client state. https://arxiv.org/abs/1903.02891
- **Charles, Bonawitz, Chiknavaryan, McMahan, Agüera y Arcas**, *Federated Select*, arXiv:2208.09432 (2022). Clients "select" data-dependent
  slices of the server model, with "drastic reductions in communication and client memory". The paper discusses the privacy and trust
  implications of the selection keys. https://arxiv.org/abs/2208.09432
- **Kairouz, Liu, Steinke**, *The Distributed Discrete Gaussian Mechanism for FL with Secure Aggregation*, ICML 2021, PMLR 139:5201-5212.
  SecAgg forces discretisation and modular arithmetic, which creates a three-way tension between communication, privacy and accuracy.
  https://proceedings.mlr.press/v139/kairouz21a.html
- **Zhu, Liu, Han**, *Deep Leakage from Gradients*, NeurIPS 2019, pp. 14747–14756. Shared gradients can reveal training data token by token.

### 2.3 Federated recommendation and embedding tables
- Efficient-FedRec (Yi, Wu, Wu, Liu, Sun, Xie, EMNLP 2021, pp. 2814–2824): a large model on the server and a light user model on the client.
  https://aclanthology.org/2021.emnlp-main.223/
- FedRecon (Singhal et al., NeurIPS 2021): partially local, stateless, deployed in Gboard.
  https://proceedings.neurips.cc/paper/2021/hash/5d44a2b0d85aa1a4dd3f218be6422c66-Abstract.html
- LightFR (Zhang, Luo, Wu, He, Li, ACM TOIS 41(4), 2023, doi:10.1145/3578361): binary (hashed) codes for federated MF.
- CoLR (Nguyen et al., arXiv:2401.03748, 2024; the WWW 2024 venue is from a search snippet, UNVERIFIED): low-rank federated
  recommendation, "up to 93.75 %" payload cut for ≈ 8 % quality loss, and SecAgg/HE compatible.
- Compositional QR embeddings (Shi, Mudigere, Naumov, Yang, KDD 2020, arXiv:1909.02107): up to 15× smaller tables.

### 2.4 What real systems do with huge ID tables (OPINION synthesis of the above)
They take one or more of these routes:
- (i) Cap the vocabulary (Gboard 10 k).
- (ii) Tie or compress the tables (hashing, QR, low rank, binary codes).
- (iii) Keep the table on the server or freeze it, and federate a small user/encoder model (Efficient-FedRec, FedRecon, and our S8).
- (iv) Move only the rows a client touches (federated select, Taobao submodel), plus a privacy layer on the index set.
- (v) Quantise both directions.

Nobody ships an FP32 166 MB round trip on a device.

---------------------------------------------------------------------------------------------------------------------------------------

## 3. Comparison

### 3.1 Strengths (MEASURED)
- **The accounting is careful and auditable.**
  - Alias deduplication, and the upload key set is enforced.
  - The header is counted in both FA and Q8, so the comparison is symmetric (06D §6).
  - Retry downloads are separate, and private state is 0 B.
  - The fleet ledger is checkpointed.
  - The byte formula is asserted against the actual payload at every Q8 upload (`quant.py:152-154`).
  - The dropped-visit surcharge is correctly **download-only**. This matches the Bonawitz 2019 failure model: a device that received the
    plan and then fails never uploads. Charging up+down for a dropped visit would overstate the traffic.
- **FA_Q8 is a textbook unbiased stochastic quantiser** (Suresh 2017 "naive stochastic rounding"; FedPAQ). It is seeded, deterministic,
  pinned bitwise ("aggregate equals the FedAvg of the dequantised uploads", FL-05 handoff), and inserted without touching the aggregator.
- **S8 is exactly the production pattern (iii).** Its bytes record charges the one-time table download per visited client, keeps a
  separate once-column, and the m6 note avoids double counting (`launch_fl.py:902-917`).
- The realism audit (B12) and 06I §7(a) already disclose that the payload is 10–30× a production payload and that the item set leaks.
  The "only weights leave" claim is correctly downgraded to data minimisation.

### 3.2 Weaknesses and errors

**E1 — S8 has no analysis path (MEASURED; the most serious).** `FA_WARM_FROZEN` is not in `retention_report.ARMS`, and an index entry
with that arm raises `AnaRefused` (`retention_report.py:764-765`). Even if it were added, `_fl_meta` would read `per_client.mean`
(training traffic only). The 81.78 MB one-time table download would then be omitted, understating S8's bytes about 13×
(INFERENCE: 6.4 MB vs 88.5 MB). `curve_bytes` would prorate a step cost linearly over rounds. So the only phone-class configuration in
the study cannot appear in the retention table, the privacy ladder or the bits curve produced by the frozen script. The ANA freeze has
not happened yet, so this is cheap to fix now.

**E2 — The bits-vs-quality "curve" is mostly a learning curve.**
- Six of the eight non-S8 FL arms have the *same* per-visit payload: FA, FP, PF, FA_1024, DP8, DP1 (and FA_WARM) all use 165.68 MB.
  FA_1024/DP use 384 × 1,024 ≈ 393 k visits, equal to FA's 6,827 × 57.6 ≈ 393 k by the equal-EFE design.
- So along the bytes axis the arms differ only in *quality at equal bytes*. That is a convergence comparison, not a compression frontier.
- The bytes axis really has three points: 165.7 MB (FP32), 103.6 MB (Q8 upload) and 2.1 MB + 81.8 MB once (S8).
- The proposal (§5) asks for "bits transmitted vs quality retained for various federated protocols". As built, the thesis can show one
  compression ratio (1.6×, upload only) and one structural cut (S8), and S8 is confounded by warm start.

**E3 — S8's quality cost is confounded with warm start.** The only registered S7/S8 contrasts are K20 = R(FA_WARM/C_BIG) and K21 =
FA_WARM − PRE (06K §2). There is no registered S8 − S7 paired Δ. That paired Δ is the clean "cost of not transmitting the tables"
(same θ_pre, same recipe). S8 vs FA mixes the data from 1.22 M pretraining users with the compression.

**E4 — FA_Q8 per-tensor scaling may interact with the tuned recipe (INFERENCE).**
- One FP32 scale per 10 M-element table means s = max|Δ|/127 over the whole table.
- With the fresh-AdamW client the delta is sign-like (≈ lr per step on every coordinate that gets a gradient). Its magnitudes are near
  the max, so 8 bits is almost lossless and K14 ≈ 100 is expected. That is a weakly informative contrast.
- If S3 selects **client SGD** (06I §3 / 06J §2), the output-table delta becomes heavy-tailed: target rows get ≈ (1−p)h and the other
  158 k rows get p_j·h. Most entries then fall far below s and are rounded to 0 or ±s at random.
  - Per client, the noise std for an entry at x = s/1000 is ≈ 32 x.
  - Over 57.6 survivors, the SNR on the dense "push-down" signal is ≈ 1/4.
  - Under FedAdam, per-coordinate normalisation of this unbiased noise can inflate steps on coordinates that carry only noise.
- So K14 would partly measure an optimiser × codec interaction, not a property of 8-bit uploads. QSGD's remedy is bucketed scales
  (per row here: +4 B × 316,975 rows ≈ 1.27 MB, i.e. +6 % over Q8). Suresh's remedy is a random rotation.

**E5 — There is no equivalence margin for K14.** K14 is confirmatory, but "the CI contains 100" is not evidence that Q8 is harmless.
A compression claim needs a registered equivalence (TOST) margin, for example ±1 retention point. The user-level paired bootstrap also
ignores training-run variance: two seeds only, and the Q8 noise sends the trajectory down a different path.

**E6 — Leakage statement, refinements.**
- (a) Under the executed AdamW (wd 1e-5 on ndim ≥ 2; `launch_fl.py:15, 164`; `fl/adapter.py:225-229`), untouched rows of `item_embed`
  are multiplied by (1 − lr·wd). The delta is therefore not exactly zero off-support, as 06I §7(a) says, but ~1e-9 against ~1e-4 on
  support. The support is still trivially recoverable by magnitude (INFERENCE). Under client SGD (wd 0) the support is exact.
- (b) The support is the rows of the items in the client's *history window and examples* (USER_HISTORY_L50), not just the visit's
  targets. The target rows of `output_embed`/`output_bias` carry large opposite-sign updates, which is a second channel (audit B2).
- (c) **S8 does not remove the categorical leak.** The brand (3,956) and category (663) side embeddings are still uploaded, and their
  row support reveals the brands and categories the user viewed. 06K §3 says S8 "removes the input-table item-set leak". That is true
  for items only. The disclosure should add that brands and categories still leak.
- (d) Q8 keeps exact zeros (`quant.py:61`), so FA_Q8 leaks the same row support as FA (the audit already notes this).

**E7 — Lossless compression is left on the table.** The dense FP32 upload already reveals the input-table support (E6). A **sparse
encoding of the input-table rows** therefore adds *no* leakage and is lossless (exactly lossless under SGD; under AdamW the server can
re-apply the known wd shrink).
- For a client with ≈ 100 distinct history items, that row block costs ≈ 26 KB instead of 40.57 MB.
- The FA upload falls from 82.84 MB to ≈ 42.3 MB and the visit from 165.7 to ≈ 125 MB. With Q8 on the dense part, ≈ 93.4 MB.
- These are analytic bytes on the *same trained model*, so they add a free, honest point to the bits axis. The ladder research dismisses
  "sending only touched rows" because it "leaks the item set" (`FL_STRONGEST_LADDER_RESEARCH_20260929.md:170`), but it concedes that
  the dense upload already leaks it. That argument is internally inconsistent for the non-DP arms.

**E8 — The download is never compressed and dominates the traffic.** Downloads are 82.84 MB per visit in every non-S8 arm, 80 % of Q8
traffic. Production compresses downloads (Gboard ships a quantised 1.4 MB model; Caldas 2018 reports 14× less download; STC compresses
downstream). The audit lists the FP32 download as "disclose only" (B15). That is acceptable, but then the thesis must not call Q8 "the
compression point". It is an upload-compression point with a 1.6× saving on the total.

**E9 — The deployment byte budget is incomplete** (OPINION + INFERENCE). The accounting counts training traffic only. A comparable
Edge ladder should put every level on the same axis:
- L0 cloud: raw events up. At ≈ 20 TRAIN decisions per MAIN client (2.59 M / 131,072) × ~100 B that is ≈ 2 KB per client (INFERENCE,
  order of magnitude).
- L1/L5: one model download, 82.84 MB FP32 or ≈ 21 MB INT8.
- L2: ≈ 525 MB per client charged for FA (3.0 surviving visits × 165.68 + 0.33 dropped × 82.84), ≈ 338 MB for Q8, ≈ 88.5 MB for S8,
  plus the final model download for inference (+82.84 MB FP32; S8 +1.06 MB).
- The thesis-grade finding is "privacy costs about five orders of magnitude more bytes than the cloud path at this catalogue size".
  It is available at zero GPU cost and is not planned anywhere I could find.

**E10 — Minor accounting points.**
- The 06K and ladder-research encoder sizes are off by 1.6× (§1.4).
- The prorated cumulative bytes are fine for constant per-visit payloads. They are not fine for S8 (step at the first visit) or for
  FA_WARM if its once-costs are ever added.
- `bootstrap_bytes_once` lists the frozen tables twice (REV-DELTA-02 m6, documented).
- A client that drops at its first visit is not charged the table download. That understates slightly and is disclosed.
- No protocol framing, TLS or SecAgg expansion is counted. SecAgg adds modular-range bits and O(n) key agreement per client
  (Kairouz 2021). This matters if the thesis proposes SecAgg as the remedy for the leak.

### 3.3 Unrealistic choices (OPINION, measured against the §2 evidence)
- **166 MB per visit / ≈ 0.5 GB per client over training is unrealistic for phones and for metered links.** It is about 120× Gboard's
  shipped model, and several times Bonawitz's "tens of megabytes" upper range.
- **It is plausible for an Edge desktop on an unmetered connection with production-style eligibility** (idle, charging, unmetered). The
  browser vendor precedent (Chrome's once-only GB-scale model download over unmetered links) supports "download once, cache". It does not
  support "download 83 MB again on every one of ~3 visits". That pattern is unprecedented.
- The realistic shape for Edge is S8-like: a one-time ~82 MB (or ~21 MB INT8) catalogue pack, then ~1–2 MB encoder rounds. The project
  already built this. It must now be *reported*.
- The 158,486-item full-catalogue full-softmax design is the root cause. The project's rule "no model changes disguised as FL tuning"
  (same model as C_SMALL) is a sound internal-validity choice. Its external cost should be stated plainly: production would cap the
  vocabulary (Gboard 10 k), compress it, or keep it off the device.

### 3.4 Missing pieces
- A multi-point compression frontier: bit-width sweep (8/4/2), bucketed vs per-tensor scales, and download quantisation.
- Iso-bytes reporting: R at equal cumulative bytes, the natural "compression benefit" read of the existing curves.
- An S8 − S7 contrast.
- The categorical leak in S8.
- A cross-level byte budget (E9).
- Error-feedback / top-k methods are *correctly* excluded, because they need client state (06J §4(iv) realism budget: "clients stateful
  across rounds" is out of scope). This exclusion should be stated with the reason. With ≈ 3 visits per client, error feedback could not
  work in cross-device FL here anyway.

---------------------------------------------------------------------------------------------------------------------------------------

## 4. Recommendations

Cost is INFERENCE. "Pre-reg" = whether a registration (a DL row or an amendment) is needed and whether it is still legitimate. Status for
every row: **no governed value has been read**. No FA_Q8 / S7 / S8 / MAIN FL value exists (DL 183, 218), and the ANA script is not frozen.

| ID | Item | Tier | Type | Cost | Affects | Pre-reg needed / status |
|---|---|---|---|---|---|---|
| C1 | **Admit `FA_WARM_FROZEN` into `retention_report`**: add it to ARMS/FL_ARMS (method alt "FA"). Use `per_client_incl_frozen_once` as S8's registered bytes (plus the charged-dropped variant). Put the 81.78 MB once-cost as a **step at the first visit** in `curve_bytes`, not prorated. Add a test. | **MUST** | ADD (code) | ≈ 2–3 h Sonnet + review; 0 GPU | ANA freeze; no running run | Yes, a DL row before the ANA freeze. Legitimate: no S8 value exists |
| C2 | **Register a paired Δ S8 − S7** (FA_WARM_FROZEN − FA_WARM, same θ_pre, seed 2026, PRACTICAL_BEST) with the bytes ratio beside it, DESCRIPTIVE. This is the clean "cost of not sending the item tables". | **MUST** | ADD (analysis) | 0 GPU; 1 h | ANA (K-list or descriptive block) | Yes, before any S7/S8 read. Legitimate now |
| C3 | **Bytes for every ladder level on one axis** (E9): L0 raw-event bytes (a registered per-event size × TRAIN events per client), L1/L5 model download (FP32 and INT8 from the ONNX bench), L2 training traffic plus the final download, S8 once + rounds. Report next to R in the privacy-ladder table. | **MUST** (the Edge question is a bytes-vs-privacy-vs-quality question) | ADD (analysis) | 0 GPU; 2–4 h | ANA report / thesis ch. | A DL row fixing the event-size convention before the ANA freeze. Bytes are design quantities, not results, so it is legitimate at any time before the freeze |
| C4 | **Correct the S8 disclosure**: S8 still uploads brand/category side-embedding deltas, whose row support reveals viewed brands and categories. Correct the uploaded size to 265,162 params / 1.06 MB per direction (06K "≈ 0.4 M", ladder research "1.7 MB"). Refine 06I §7(a): under AdamW wd 1e-5 off-support rows are ~1e-9, not 0 (support recoverable by magnitude); the support = the history window plus the examples. | **MUST** (a wrong privacy claim otherwise) | CHANGE (text) | 0.5 h | thesis text, 06K/06I disclosure lists | Disclosure text only; register in a DL row (numbers in amendments are not rewritten, corrections go to amendments/) |
| C5 | **Iso-bytes read of the existing curves**: R(FA_Q8) vs R(FA) interpolated at equal cumulative charged bytes (FA_Q8's endpoint ≈ FA at 62.5 % of its bytes), and R vs bytes for S7/S8. DESCRIPTIVE. | SHOULD | ADD (analysis) | 0 GPU; 2 h | ANA | DL row before the ANA freeze |
| C6 | **Equivalence margin for K14** (e.g. TOST: R(FA_Q8/FA) UB/LB within ±1 retention point, paired bootstrap), plus a statement that seed variance is not in the CI. | SHOULD | ADD (rule) | 0 GPU | K14 reading | **Must be registered before any FA_Q8 value is read** (none exists: DL 183) |
| C7 | **Analytic lossless sparse-input-table bytes** (E7) as a DESCRIPTIVE bytes row for FA and FA_Q8: bytes = dense output layer + encoder + (distinct input rows × (64×4 + 4 index)), with the per-visit distinct-row count taken from the TRAIN manifests (no result read). The thesis states it adds no leakage beyond the dense upload in the non-DP arms. | SHOULD | ADD (analysis) | 0 GPU; ≈ 0.5 day (row counts from manifests) | ANA / thesis | DL row. No governed value involved |
| C8 | **Per-row (bucketed) Q8 variant `FA_Q8R`** (QSGD-style scale per embedding row / per 512 elements; +≈ 1.27 MB), or at least a CALIB pair Q8 vs Q8R at the final S4 recipe. It guards K14 against the codec × client-SGD × FedAdam interaction (E4). Keep the registered FA_Q8 unchanged. | SHOULD (becomes MUST if S3 selects client SGD) | ADD | CALIB 2 × 2.7 h; MAIN 1 × ≈ 10.7 h; code 0.5 day | new arm only; FA_Q8 untouched | A new amendment before its runs and before any FA_Q8 read. Legitimate now |
| C9 | **A small bits-vs-quality frontier on CALIB** (DESCRIPTIVE, stateless unbiased codecs only): uploads at 8/4/2 bits (per-row scales), optionally with a Suresh-style random rotation, plus download INT8 (Caldas-style lossy broadcast, server keeps FP32). Run at the final S4 configuration. This gives the proposal's §5 curve with ≥ 5 points instead of 3. | SHOULD | ADD | CALIB 4–5 × 2.7 h (parallel ≈ 3 h wall); code ≈ 1 day (quant.py bit-width + download codec) | new CALIB runs only | Amendment before launch. The CALIB tuning is still running (S2), so register now and launch after S4 |
| C10 | **State the exclusions with reasons**: error feedback / top-k (needs client state: 06J §4(iv)), tied / hashed / low-rank / sampled-softmax tables (change the model vs C_SMALL), federated select / sparse download (leaks the selection keys at inference; the full output table is needed for full-K scoring). Cite Konečný 2016, Suresh 2017, QSGD, Sattler 2020, Charles 2022, Niu 2020, Shi 2020, CoLR. | SHOULD | ADD (text) | 1–2 h | thesis Discussion | none |
| C11 | Disclose that byte counts are payload only (no framing, TLS or SecAgg expansion, which Kairouz 2021 shows is not free), and the first-visit-dropout table-download omission (m6). | DISCLOSE-ONLY | ADD (text) | 0.5 h | thesis | none |
| C12 | Keep the charged-dropped series (download-only surcharge) as the number quoted in the Edge text. It is closest to what devices receive. The registered 06A §3 series is printed beside it. | DISCLOSE-ONLY | KEEP | 0 | thesis text | none (both series are already registered) |
| C13 | Keep FA_Q8 exactly as registered (06B §5 / 06D §6). Keep the byte ledger, the header symmetry, retry accounting and the O_EXCL/seeded design. | — | KEEP | 0 | — | — |
| C14 | NICE: DP8_FROZEN (already NICE in 06K §3): the per-round noise norm falls about 8.8× (√(20.71 M / 0.265 M)), the realistic DP + bytes point. A one-page empirical item-set recovery from one recorded delta (audit B2 NICE). | NICE | ADD | ≈ 10.7 h + S re-cal; 1 h | new runs | 06K already registers DP8_FROZEN (z/q/T/δ); S re-calibration per 06K |

**Priority order if time is short:** C1 → C4 → C2 → C3 (all zero GPU, all before the ANA freeze) → C6 (before any FA_Q8 read) → C5, C7 →
C8/C9 (GPU).

---------------------------------------------------------------------------------------------------------------------------------------

## 5. Score: **6 / 10** for the communication-efficiency part

The engineering of the byte accounting is very good: exact, symmetric, deduplicated, retry-aware and dropout-aware. The download-only
dropped-visit surcharge is the correct reading of production failure modes. FA_Q8 is a correct, unbiased, tested stochastic quantiser.
S8 already implements the production-style answer to the 98.7 %-table problem, with once-per-client table accounting. The disclosure
that "only weights leave" is data minimisation and not privacy is honest and correct.

The part falls short of a convincing answer to the proposal's "bits vs quality for various federated protocols":
- The bytes axis has three points in practice. The only registered compression (8-bit uploads, dense FP32 downloads) saves 1.6× and is
  probably near-lossless under the current client optimiser. That makes K14 weakly informative, and potentially confounded if client SGD
  is selected.
- The one phone-class configuration (S8) cannot enter the frozen analysis script. If it were forced in, it would be undercounted about
  13×. It has no clean contrast against its warm-start twin.
- The S8 privacy claim overlooks the brand/category leak.
- No cross-level byte budget places FL's ~0.5 GB per client next to the cloud path's ~KB.

Every one of these gaps is fixable with zero or little GPU and without touching any governed value. Fixed, the part would be about 8/10.

---------------------------------------------------------------------------------------------------------------------------------------

## 6. UNVERIFIED items and open questions
- Chrome's Gemini Nano on-disk size (≈ 4 GB) comes from third-party posts. Only the 22 GB-free and unmetered-connection requirements
  were verified on developer.chrome.com. There is no verified public figure for Edge component-updater or Edge on-device model sizes.
- CoLR's venue (WWW 2024) is from a search snippet only. The arXiv abstract was verified.
- Whether the executed AdamW recipe's weight decay is exactly 1e-5 for every FL arm after the S2–S4 tuning. Client SGD sets wd = 0
  (`launch_fl.py:390`). This changes E6(a) only cosmetically.
- The per-visit distinct-item count (for C7) is unknown to me. I did not read the TRAIN manifests' contents. The ≈ 100-item example is
  illustrative.
- E4 (the Q8 × client-SGD × FedAdam interaction) is an INFERENCE from the codec's per-tensor scale and the gradient structure of a full
  softmax. It is not measured. C8's CALIB pair would measure it.
- The parameter breakdown in §1.1 is my arithmetic from the D256 inventory formulas. It reproduces the registered total exactly, but it is
  not a D64 inventory file. A D64 manifest inventory, if one exists on the server, should confirm it.
- ONNX INT8 size of C_SMALL (≈ 21 MB) is audit INFERENCE (REALISM_AUDIT C1). The K16 bench records the real value.
- Web references used: arxiv.org/abs/1811.03604; mlsys.org/Conferences/2019/doc/2019/193.pdf; developer.chrome.com/docs/ai/prompt-api;
  dblp.org/rec/conf/mobicom/NiuWTHJLWC20.html; arxiv.org/abs/1610.05492; proceedings.mlr.press/v70/suresh17a.html;
  proceedings.neurips.cc/paper/2017/hash/6c340f25839e6acdc73414517203f5f0-Abstract.html; proceedings.mlr.press/v108/reisizadeh20a.html;
  arxiv.org/abs/1812.07210; arxiv.org/abs/1903.02891; arxiv.org/abs/2208.09432; proceedings.mlr.press/v139/kairouz21a.html;
  aclanthology.org/2021.emnlp-main.223/; proceedings.neurips.cc/paper/2021/hash/5d44a2b0d85aa1a4dd3f218be6422c66-Abstract.html;
  dl.acm.org/doi/10.1145/3578361; arxiv.org/abs/2401.03748; arxiv.org/abs/1909.02107.
