# G0 — Opus position (Operations Director). Independent memo; withheld from Fable until Fable's own position exists.

Written 2026-09-26, about 13:52Z. Evidence tags follow the charter. The evidence is in `g0/` and `company/handoffs/`.

## Q1. Population and the client cohort
**Proposed decision.** Lock a deterministic TRAIN-only cohort of **N = 262,144** warm users, chosen by salted hash, the
largest nested option that stays feasible for either possible winner. Every method, including the central contest,
uses this cohort. Cold users are evaluated in a separate common cold block.

**Evidence:**
- PROJECT_MEASURED (DATA-01): U = 2,694,998 warm users and 40.18M examples (median 6 per user, p99 138). At N = 262,144
  the cohort has 3,906,929 examples, 62,368 users with a VALIDATION query and 808,610 VALIDATION rows.
- INFERENCE (SYS-02): the downstream core over ALL costs 908 GPU-h for GRU and 2,701 GPU-h for SASRec (two seeds),
  which is months. At 262,144 it costs 88 and 263 GPU-h; at 65,536, 22 and 66 GPU-h.
- Statistical support (INFERENCE, from √n scaling of the SASRec-campaign CI half-width of ±0.00024 at 746k users):
  about ±0.0008 at 62k validation users and about ±0.0017 at 15.5k.

**Implementation implications:**
- The worst case is SASRec winning: about 263 GPU-h plus calibration (+20–40%), roughly 6–8 days on two GPUs.
- If that is judged wasteful, 65,536 is the fallback. It is 4× cheaper, but its CI is about 2× wider.

**Main uncertainty (material, therefore a deliberation topic).** Should the central family contest run on the same
cohort or on ALL?
- My view: on the same cohort. It keeps the selection population identical to the downstream population, it is cheap
  (1.4 / 3.6 h per seed), and it avoids the "cohort-FL divided by full-data central" confound the package forbids.
- Cost of that choice: the winner on 3.9M examples may differ from the winner on 40M (transformers are often more
  data-hungry). A full-data central run could be kept as a descriptive secondary reference; it is not needed for the
  thesis question.

## Q2. Catalogue threshold
**Proposed decision.** Keep support ≥ 5 on the full TRAIN (K = 158,486) as the shared catalogue, and disclose it as
bootstrap information pooled from all TRAIN users, including those outside the cohort (information block
I_META_RANDOM).

**Evidence (PROJECT_MEASURED):**
- s = 1 raises K by 21% (to 191,890) for +0.414 pp of VALIDATION label coverage (99.142% → 99.556%).
- s = 2 raises K by 13% for +0.322 pp.

**Reasoning (INFERENCE).** Every client step pays for the K × d head, so s = 1 costs about 21% more in every FL and LO
cell and in communication, for less than 0.5 pp more coverage. The end-to-end metric already charges OOV as zero,
and the conditional metric is reported beside it.

**Alternative.** A cohort-only support ≥ 5 catalogue would reduce K but diverge from the historical central
references.

## Q3. VALIDATION merge and target barrier
**Proposed decision.** Merge VS and VC into one VALIDATION window (Nov 16 to Nov 23, 7 days), minimal-diff: keep the
v1 in-partition target bindings and the internal Nov 20 barrier. Report the count of affected decisions.

**Evidence:**
- PROJECT_MEASURED: v1 targets never cross a partition boundary.
- PROJECT_MEASURED: the union has 14,357,645 decisions and 11,288,056 eval rows over 1,578,912 users.

**Reason.** This is the package's minimal-diff default; re-deriving targets across Nov 20 changes the decision set.

**Resolved (PROJECT_MEASURED; `g0/BARRIER_COUNT.json`, exact over 64 buckets).** Only **170** VS decisions are censored
only by the Nov 20 barrier (BOUNDARY_UNRESOLVED with the next view in VC). Other counts:
- 152 VC decisions censored at the Nov 23 boundary, correctly censored under V4;
- 10 TEST decisions unresolved into LABEL_GRACE;
- 1 TRAIN decision unresolved into VS.

The minimal-diff merge therefore differs from a re-derived merge by 170 targets out of 11.29M VALIDATION eval rows
(0.0015%). Targets are within-session only, so NO_OBSERVED_NEXT with a later next view means a session break, not a
barrier.

## Q4. Features and transforms
**Proposed decision.** Clean Rich = the v1 rich channels, with:
- the proven exact redundancies removed (re-derived from code, full-population contradiction 0);
- specification-dead flags removed;
- the M1 valid-only refit applied with re-zeroing afterwards;
- the new topology trained from scratch (no weight folding and no lean-parity claim);
- normalizers fitted once on the full TRAIN, shared by every method.

**Evidence (PROJECT_MEASURED, feature study):** the full-population duplicate checks were 0; M1 was confirmed on all of
TRAIN.

## Q5. Precision
**Proposed decision.** Strict FP32 everywhere. The cohort makes it affordable, so no TF32 branch is needed.

## Q6. Budget and schedules
**Proposed decision:**
- Central recipes follow the executed ones: GRU 6 passes; SASRec 5, including the cooldown.
- A bounded calibration of at most 2 LR candidates per family is allowed, because the cohort cuts the step count about
  10× (about 15k steps per epoch against 157k).
- FL endpoint 6 EFE, practical cap 8 EFE; patience 2 after the low-LR phase; tolerance 0.0002.

**Reasoning (INFERENCE).** The cohort's smaller step count may leave the executed warmup and decay mis-scaled; that is
the "genuine recipe mismatch" case in package 05 §C.

## Q7. FL settings
**Proposed decision:**
- 64 clients per round; 2 local passes; local batch 64.
- AdamW local solver, reset at each download; server weighted average by consumed examples.
- FedProx μ grid {0.001, 0.01, 0.1}, with μ = 0 as the FedAvg identity test.
- PF private query offset of size 128 or 256, zero-initialised, L2 1e-4; CPF matched to it.
- LO starts each client from θ0 and trains 6 local passes.

## Q8. Device
**Proposed decision.** The server-CPU ONNX Runtime is a declared proxy. The owner may approve a local-PC ORT install for
a second proxy. Physical phone and browser measurements stay PENDING.

## Deliberation topics I expect
1. Whether the central contest runs on the cohort or on ALL (Q1).
2. The cohort size: 262,144 or 65,536 (Q1).
3. The catalogue scope: full TRAIN or cohort TRAIN (Q2).
