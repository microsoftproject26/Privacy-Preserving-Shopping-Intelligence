# PREG1 amendment 06E — review-driven registrations after REV-FL-01 / REV-ANA-01 (registration only)

- **Status:** REGISTERED 2026-09-29T08:42Z by Claude Fable 5.1 (chief scientist; Ops authority shared with the Ops session per
  DL 168–169). Sources: `company/reviews/REV-FL-01_OPUS_REVIEW.md` (ACCEPT_WITH_FIXES: B1, B2 PF blockers; M1–M3; D1–D13),
  `company/reviews/REV-ANA-01_OPUS_REVIEW.md` (ACCEPT_WITH_FIXES: A-F1…A-F17, B-F1…B-F7, C-F1/2, D-F1…D-F3).
- **Registered BEFORE any affected read.** Unread at registration: every downstream RESULT_ROWS (C_SMALL sealed rows; the 8 A40
  re-evaluations; every FL/device/ONNX/C_FULL/truncation arm — none of these runs exists yet except the re-evaluations, the LR-screen
  runs x0.5/x1 (their selection values are registered reads, not yet taken) and smokes). The A16 contest values (G2) are final.
- **Unchanged:** amendments 03–06D except where a numbered item says "adds" or "clarifies".

## 1. LO cell rule (adds to 06A §1; closes REV-FL-01 D13)
LO runs its three budgets {1, 2, 4} passes on every role. For K8 = R(LO / C_BIG) and K11 = FA − LO the LO cell is the one selected
ONCE on the UNSEEN_VAL role (`launch_device --arm LO --role UNSEEN_VAL --select`, seed 2026) by the same rule as FT_C (highest E2E
client-macro MRR@20 on UNEW_VAL_VALIDATION_32768; ties → fewer passes), recorded O_EXCL in `contracts/device/SELECTION_LO.json`
with a DECISION_LOG row BEFORE any MAIN LO value is read. The other two cells are reported as DESCRIPTIVE (`LO_CELL_NOT_SELECTED`).
Applied to both seeds and to every panel. Code: FL-08.

## 2. Registered seed sets for the confirmatory contrasts (clarifies 06B §2; closes A-F4)
K13, K15, K16 are single-seed (2026) by registration; every other K uses seeds {2026, 2027}. A run of the second seed of a
single-seed contrast (NICE tier) is reported as DESCRIPTIVE; a missing registered seed is flagged `INCOMPLETE_SEEDS` beside the
value (06D §8 cut rule: reported, never substituted).

## 3. K16 comparison block (clarifies 06B §2 K16 and 06A §6; closes A-F2)
K16 = R(C_SMALL_INT8 / C_SMALL_FP32) is computed INSIDE an ONNX-CPU sub-block: both numerator and denominator are onnxruntime
CPU rows (single thread, batch 1, opset 17, the same rows subset, the same onnxruntime version and optimisation level), never the
A40 PyTorch C_SMALL rows. The ratio against the A40 PyTorch C_SMALL is DESCRIPTIVE (`DEVICE_MIXED`). The INT8 rows subset (06A §6
fallback) = the first 16,384 E2E users of MAIN_VALIDATION in **selection_key_v2 order** (`sha256(SEL_PREFIX_V2 + digest)[:8]`), the
population key of the 5–100 pool; the exact user-set sha256 is recorded in rows.meta.json (DEV-04, option b of D-F2).

## 4. Comparison-key checks of the analysis (clarifies EVALUATION_CONTRACT §9 for ANA; closes A-F5)
Within a retention block every non-historical row must agree on eval_device_class, split, evaluator_code_sha256, query_order_hash
(per role), candidate_order_hash, label_policy_hash, data_protocol_hash, feature_contract_hash, precision_id, objective_id,
eval_score_block and eval_stream; intended differences recorded per entry: population_hash, topology_hash, initialization_block,
eval_batch (06A §9f). On the pinned A40 block every non-historical row carries status MEASURED_PINNED_DEVICE and one device_pin_sha256.

## 5. Evidence-integrity rules of the analysis (adds; closes A-F3, A-F6, A-F7, A-F10, A-F13)
RULE_FREE only for the baselines (RULE, SKNN, LAST, POP, KNN_EMB_SMALL/BIG); device arms and ONNX rows are PRACTICAL_BEST; every
rankable row of an evaluated view has a rank ≥ 1; C_FULL rows are identified by train_manifest_id = FULL_TRAIN_5_100_TIEBREAK,
C_BIG/C_SMALL by MAIN_TRAIN_5_100_131072, FL/device rows by method = arm; CONTROLLED_BUDGET rows are the 6.0 endpoint and the
C_BIG PRACTICAL_BEST row is best@3.0 (ARCHITECTURE_LOCK); smoke, LR-screen and CALIBRATION rows are never evidence.

## 6. Sanity trigger (clarifies 04 §1; closes A-F12)
"C_SMALL 6.0 value < 0.5 × C_BIG" is evaluated as C_SMALL CONTROLLED_BUDGET vs C_BIG PRACTICAL_BEST (both printed).

## 7. PF evidence (adds; closes B1, B2)
A PF run's PRACTICAL_BEST model = (θ at the best mark, the personal store at that mark); the store of the best mark is persisted
(`PF_STORE_best.pt`) and its digest is part of the checkpoint id; a PF run is never resumed before FL-08 lands; the TEST look
re-evaluates PF with the best-mark store.

## 8. Device-class continuity (adds to 06D §1; closes M1–M3, D5)
Device rows carry the pin (status MEASURED_PINNED_DEVICE); an FL resume leg and every device shard record eval_device_class +
device_pin_sha256 (+ code sha256) and refuse a mismatch. The pin file cannot contain the re-evaluation rows' sha256 (the
re-evaluations need the pin): those hashes are recorded in the DECISION_LOG row of the re-evaluation instead (D5 accepted).

## 9. Disclosures added (thesis §5.5, §9, DP appendix)
(a) The DP_Z_RECORD's "fixed-size WOR" sampling disclosure predates 06C; the sampler is Poisson and the bound is exact — the record
is not edited (locked receipt); this note supersedes its text. (b) The DP noise is drawn from a public deterministic seed with a
non-cryptographic PRNG in FP32: the ε is the guarantee of the mathematical mechanism in simulation, not of a deployable
implementation (Mironov 2012); ε covers the released θ_1..θ_384 only, not the run's internal logs. (c) FA_Q8 stochastic rounding is
unbiased up to FP32 rounding of U. (d) PRACTICAL_BEST = the strictly-best running mark (improvement > 0.0002), the plain max is
recorded (differs by ≤ 0.0002). (e) The seed-mean retention is the mean of per-seed ratios (06A §7 reading). (f) The contrast-text
check folds whitespace and dash glyphs. (g) FL cumulative bytes at a mark are the fleet mean prorated by rounds. (h) A40 vs A16
rank differences for a random-init model: 12,635 (GRU) / 14,063 (SASRec) of the M02 rows (T3 receipt) — the reason for the pinned
class; the C_BIG/C_SMALL device sensitivity is reported from the two blocks. (i) The truncation study: C0 carry-in cancels
algebraically (last float32 bit of two price channels); h = 50 holds 49 prior events + e_q in the model sequence; VC queries' h
counts VS events the user context never replays; T41 admission may concern columns 5–7 on flagged rows.
