# G1b part 2 packet (SASRec) — Ops, 2026-09-28T03:41Z. Per STAGGERED_P3_FABLE_RULING.md C3, amendment 05, MAIN_SCALE ruling.

Question: approve G1b part 2 (pin the SASRec candidates) and the SASRec MAIN runs (4 = 2 candidates × seeds 2026/2027)?
Also pin retroactively part 1R (GRU C2), registered mechanically under the MAIN-scale ruling (DL 103, 106).

## 1. SASRec P2 (PROJECT_MEASURED; HP_SEARCH on CALIB; macro MRR@20). Queue done 03:39:30Z rc=0; 12/12 + 3 stage-2 runs OK,
no exclusions, no triggers. Stage 1 self 2d948bbd…964e; stage 2 self 8bf4a706…0ffa.

| hp_id | size | peak_lr | @2.0 | @4.0 | @6.0 | ckpt_avg |
|---|---|---|---|---|---|---|
| D256_B2_S0_s0 | D256_B2 | 0.000400 | 0.12465 | 0.13346 | **0.13042** | 0.13090 |
| D256_B2_S0_s3 | D256_B2 | 0.002250 | 0.07377 | 0.09439 | 0.09760 | 0.09730 |
| D256_B2_S1_s1 | D256_B2 | 0.000711 | 0.12030 | 0.11848 | 0.11658 | 0.11991 |
| D256_B2_S1_s2 | D256_B2 | 0.001265 | 0.10314 | 0.12127 | 0.12269 | 0.12474 |
| D256_B3_S0_s0 | D256_B3 | 0.000400 | 0.12125 | 0.13553 | **0.13323** | 0.13393 |
| D256_B3_S0_s3 | D256_B3 | 0.002250 | 0.07270 | 0.08881 | 0.09159 | 0.09182 |
| D256_B3_S1_s1 | D256_B3 | 0.000711 | 0.10946 | 0.12048 | 0.12079 | 0.12286 |
| D256_B3_S1_s2 | D256_B3 | 0.001265 | 0.09083 | 0.10172 | 0.11678 | 0.11650 |
| D384_B2_S0_s0 | D384_B2 | 0.000400 | 0.12593 | 0.13083 | **0.12760** | 0.12834 |
| D384_B2_S0_s3 | D384_B2 | 0.002250 | 0.06979 | 0.08950 | 0.09384 | 0.09315 |
| D384_B2_S1_s1 | D384_B2 | 0.000711 | 0.11650 | 0.11946 | 0.11529 | 0.11809 |
| D384_B2_S1_s2 | D384_B2 | 0.001265 | 0.08228 | 0.09130 | 0.11104 | 0.11026 |

Stage 2 (seed 2027) two-seed means: **D256_B3_S0_s0 0.132402** (83.9M params), **D256_B2_S0_s0 0.130120** (83.1M),
D384_B2_S0_s0 0.127326 (125.7M). p2_driver tuned = D256_B3_S0_s0 (dropout [0.1], S0, lr 0.000400); CKPT_AVG carry = True for
all sizes (gains 0.00061–0.00085 > 0.0005; affects P4 C_BEST only).
**Candidates (amendment 05 §2, computed by code_p3b register-02b --with-c2 in a TEMP copy; the real registry is untouched):**
C1 02B_SASREC = P2_SASREC_D256_B3_S0_s0 (sha 9a253087…7f6d); C2 02B_SASREC_C2 = P2_SASREC_D256_B2_S0_s0 (sha 9c9978c6…f292);
no_c2 = []; registry v3 would be self **0b0bc3b4d54d0444ab04045008d8401980ee724cae16a963598e59834175dc42** (02B_GRU and
02B_GRU_C2 byte-identical). The real registration must reproduce this self-hash (Ops checks).

Ops observations (not decisions): (a) the SASRec optimum is the LOWEST LR of the grid (0.000400) for all three sizes, and
the S0 runs peak at 4.0 then fall by 6.0 (e.g. D256_B2 0.1335 → 0.1304) — an edge on the low side (GRU's was on the high
side; both unextended per the MAIN-scale ruling); MAIN has 4× more updates per EFE-scaled schedule, so behaviour may
differ. (b) SASRec CALIB means exceed GRU's (0.132 vs 0.125) — tuning evidence only, not the contest.
(c) INCIDENT (DL 105): Ops incidentally printed one SASRec TUNING STATUS.json value (D384_B2_S1_s2 @2.0 = 0.08228) at 00:5xZ;
the procedure and the mechanical selection were not changed.

## 2. GRU status (public state only)
C1 D128 s2026/s2027 DONE_CONTINUATION (7.0); C2 D256 s2026 DONE_CONTINUATION (7.0) ~03:3xZ; C2 s2027 starting on GPU0.
Part 1R (registry v2 self d8172874…, 02B_GRU_C2 = P2_GRU_D256_S1_s2 sha ea92d3d1…) is in G1B (_2, parts [1, 1R], file
ac71ef09…); C2 launch hashes s2026 3e115847… == dry run. Please pin 1R too.

## 3. Plan on approval
Register SASREC (C1 + C2) and rewrite G1B to parts [1, 1R, 2] back-to-back (seconds), only while no GRU run is starting;
dry-run the 4 SASRec runs (code_p3b); then **GPU1 immediately: C1 s2026 → C2 s2026; GPU0 after GRU C2 s2027 ends: C1 s2027
→ C2 s2027** (staggered per-GPU queues; STRICT numerics are timing-independent per your staggered ruling). INFERENCE per
SASRec MAIN run ≈ 4–6 h → SASRec done ≈ 14:00–18:00Z. Disk: free 23.6 GB; a SASRec run peaks ≈ 2 GB in flight and keeps
≈ 0.7 GB; worst case (2 in flight + 6 finished GRU/SASRec kept) ≈ 8 GB → ≥ 15 GiB free > 8 GiB guard.
MAIN baselines (POP, LAST, SKNN k=500/s=5000) finished on CPU (not read).
