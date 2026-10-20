# Brief to Fable 5.1 — scope decision for SASREC_FEATURE_OPT_V1 (owner-delegated)

## Owner directive (2026-09-25, verbatim)
"خلي fable يراجع كل حاAجه ويعمل الاصلح بدون الرجوع ليا"
Translation: "Have Fable review everything and do what is best, without coming back to me."

You are the owner's delegated decision authority for this add-on study. Your decision is recorded
verbatim and treated as OWNER-APPROVED. The lead (Opus) executes it. You are read-only: do not
execute, edit or delegate. Use WebSearch/WebFetch only if one specific primary fact is essential.

## Authority boundary (you may NOT authorize any of these)
- changing the frozen dataset, targets, catalogue, sessionization or eligibility;
- deleting scientific evidence;
- excluding hard users or items;
- TEST/VC use for selection;
- weakening OOV handling;
- approximate or sampled evaluation;
- touching or delaying the running campaign (seed 2027, cooldowns, lock, final VC/TEST evaluation);
- teammate resources;
- git commit/push/PR or any remote mutation;
- credentials;
- shared-environment changes.

The package's hard contract (docs 01–05) stays in force. Within it, you may change the plan's scope,
gates, thresholds, ordering and budgets, and you may stop the study.

## Read (compact; do not browse broadly)
- Package: `F:\Queens University\Microsoft Project\eda\SASREC_FEATURE_OPT_V1\01_OPUS_FEATURE_OPT_MASTER.md`,
  `03_EXPERIMENT_PLAN.md`, and `02_AUDIT_AND_CANDIDATE_RULES.md` sections C–E.
- Actual schema: `F:\Queens University\Microsoft Project\eda\.local_ai\feature_opt_v1\evidence\snap\dataset\FEATURE_DICTIONARY.md`
  and `FEATURE_GROUPS.json`.
- Actual model ingestion: `F:\Queens University\Microsoft Project\eda\.local_ai\sasrec_v1\code\sasrec_model.py`,
  lines 49–56, 147–176, 251–261 and 271–320.
- Audit so far (Sonnet A, read-only, from code): `F:\Queens University\Microsoft Project\eda\.local_ai\feature_opt_v1\audit\`.
  Start with `REDUNDANCY_FINDINGS.md`, `OPEN_QUESTIONS.md` and `FEATURE_DEPENDENCIES.json`; open
  `FEATURE_INVENTORY.csv` only as needed.
- **MEASURED audit findings on a bounded real-data sample (read this first):**
  `F:\Queens University\Microsoft Project\eda\.local_ai\feature_opt_v1\audit\MEASURED_FINDINGS.md`
  - M1: fit-population defect in two user time features;
  - M2: 9 exact duplicate channels;
  - M3: 6 dead flags;
  - M4: missingness/OOV rates;
  - M5: TRAIN→VS drift.
- Study state and baseline rule: `...\feature_opt_v1\STATE.md`, `...\feature_opt_v1\config\BASELINE_RULE.md`.
- Earlier ceiling/levers review (context only): `F:\Queens University\Microsoft Project\eda\.local_ai\sasrec_v1\review\deep\FABLE_HIGH_RECALL_RESPONSE.md`.

## Measured facts (PROJECT_MEASURED unless marked otherwise)
- **Full-VS macro MRR@20, epoch 3:**
  - A1 rich (all side features, F4): 0.18142, micro R@20 0.4636.
  - A0 item-only: 0.17376, micro R@20 0.4477.
  - GRU F0: 0.17387.

  So the entire side-feature bundle is worth +0.0077 macro MRR (one seed, same recipe).
- **Epoch 4** was lower than epoch 3 for both arms (A1 0.17976, A0 0.17282).
- **Seed variation is not yet measured.** A1 seed 2027 (same recipe) finishes about
  2026-09-26T00:50Z, and its VS e3/e4 scores will give the first s2026-vs-s2027 difference.
- **Campaign timeline:**
  - A1 s2026 cooldown epoch 5 on GPU1 until about 13:40Z today, plus evaluation;
  - possibly A0 and s2027 cooldowns afterwards;
  - final stage (lock, then one retrospective VC/TEST) on both GPUs, done about
    2026-09-26T07:30Z.

  The study gets no GPU or heavy scan before then.
- **Resources:**
  - disk free 12.37 GiB (target ≥10, floor 8);
  - cgroup RAM 61.3/64 GiB used, mostly page cache; the two trainers hold about 18 GB RSS each;
  - GPUs: 2 × A16, 15.4 GB each.
- **Historical rate** (inference, from campaign throughput): about 3.8 h per full-TRAIN epoch,
  including a VS eval. A from-scratch 5-epoch pair is therefore about 20–21 h on both GPUs, and a
  4-arm 5M-decision warm-start screen is about 2–2.5 GPU-h.
- **Disk estimate for a pair:** each run holds latest.pt and previous.pt with Adam state
  (about 0.95 GB each) plus per-epoch weights (about 0.33 GB each), so about 5–7 GiB for two runs.

## The lead's own assessment (a hypothesis, for you to accept or reject)
- The plan is well built, but for an accuracy goal the expected gain is small.
  - The whole side bundle is only +0.0077.
  - The existing normalization is already log, then TRAIN-quantile clip, then z-score, fitted on
    TRAIN only.
  - Most D1/D2 engineering already exists (`current_minus_recent_mean_log_price`,
    switch/run-length/match shares).
  - Two numeric pairs look like exact duplicates after the transform
    (`log1p_prefix_elapsed_s` == `log1p_segment_elapsed_s`; `prefix_item_count` ==
    `prefix_representative_count` + 1 with the same std), which is an efficiency matter, not accuracy.
  - D3 (revisit counts of the current observed product, OOV-safe) is the only clearly new signal.
- A +0.001 acceptance threshold on a single-seed pair may sit inside seed noise.
- The full pair costs about a day on both GPUs and makes the disk tight.
- **Proposal:**
  1. Finish the audit, plus a bounded sample profile, now.
  2. Run the full profile and baseline + ≤8 coherent probes after the campaign (about 1–2 GPU-h).
  3. Hold an evidence gate. Run the screen and the pair only if (a) a real transform/mask bug is
     found, or (b) a candidate shows a signal clearly above measured seed noise. Otherwise
     KEEP_CURRENT.

## Decide (answer every item)
1. **SCOPE:** FULL_PLAN / STAGED_WITH_EVIDENCE_GATES / AUDIT_AND_PROBES_ONLY / STOP. Give the reason.
2. **Gates:** the exact numeric go/no-go criteria for (a) probes to screen and (b) screen to full
   pair.
   - Include how seed noise enters: the s2026-vs-s2027 VS difference, and whether and how to use
     the paired bootstrap on the cohort.
   - Include whether the package's +0.001 / −0.001 defaults stand or are replaced (replacement is
     allowed only now, before candidate training).
3. **Probes:** is baseline + ≤8 coherent group probes on the 10k-user VS cohort worth running? Must
   any be dropped or required? Is the probe semantics the lead intends acceptable? The lead intends
   "metadata-unavailable counterfactual" encodings where the pipeline defines missing semantics
   (category, brand, price, prev gap, item age, cold-user user-context), and a time-frozen reference
   instant for calendar. Is anything prohibited?
4. **Bug path:** M1 is a measured fit-population/re-zero defect in two user time features. What is
   the required action? Options include a repaired study view (valid-only refit + re-zero by mask)
   entering as a candidate, report-only, or other. The same question applies to any further defect.
5. **Candidate priorities** if a screen happens, max 3 plus control, from what is known now.
   Duplicate removal as efficiency-only? D3? A normalization alternative? Anything to skip outright?
6. **Resource and disk policy for a full pair**, within the rules that forbid deleting evidence and
   keep ≥10 GiB target / 8 GiB floor. For example: run one arm at a time or both concurrently? Can
   previous.pt be dropped for this study's own runs?
7. **Baseline rule:** accept `config/BASELINE_RULE.md` or amend it.
8. Anything in the lead's assessment that is wrong, and anything else you order changed.

## Output format (≤1,400 tokens)
Line 1: `ROUTING: <exact model id from your system prompt>`
Line 2: `VERDICT: <SCOPE value>`
Then `DECISIONS:` as numbered items 1–8 matching the list above. Tag every claim
[evidence: file:line / value], [literature: URL] or [inference]. End with `SKIP:` (a short list).
