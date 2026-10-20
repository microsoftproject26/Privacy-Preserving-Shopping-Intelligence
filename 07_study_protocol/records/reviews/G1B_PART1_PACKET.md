# G1b part 1 packet (GRU only) — Ops, 2026-09-27T21:40Z. Per STAGGERED_P3_FABLE_RULING.md condition 3.

Question: approve G1b part 1 (pin the GRU 02b tuned recipe) and the GRU P3 launch (MAIN contest, seeds 2026 then 2027,
sequential, GPU0, from the separate tree code_p3/)? SASRec P2 is still running on GPU1 (its table comes in part 2).

## 1. GRU P2 (PROJECT_MEASURED; label HP_SEARCH, never selection evidence; tuning-split macro MRR@20)
Stage 1 (seed 2026, 12 trials, all exit 0, finite, no exclusions, no triggers; file self b1641155…59c1):

| hp_id | size | sched | peak_lr | @2.0 | @4.0 | @6.0 endpoint | ckpt_avg |
|---|---|---|---|---|---|---|---|
| D128_S0_s0 | GRU_D128 | S0 | 0.000363 | 0.08750 | 0.10110 | 0.10342 | 0.10321 |
| D128_S0_s3 | GRU_D128 | S0 | 0.001149 | 0.09489 | 0.10867 | 0.11148 | 0.11093 |
| D128_S0_s5 | GRU_D128 | S0 | 0.002476 | 0.09731 | 0.10928 | 0.11107 | 0.11092 |
| D128_S1_s1 | GRU_D128 | S1 | 0.000533 | 0.08333 | 0.09615 | 0.10855 | 0.10793 |
| D128_S1_s2 | GRU_D128 | S1 | 0.000783 | 0.09083 | 0.10621 | 0.11826 | 0.11774 |
| **D128_S1_s4** | GRU_D128 | S1 | 0.001687 | 0.10249 | 0.11664 | **0.12377** | 0.12385 |
| D256_S0_s0 | GRU_D256 | S0 | 0.000363 | 0.09446 | 0.10911 | 0.11053 | 0.11054 |
| D256_S0_s3 | GRU_D256 | S0 | 0.001149 | 0.09311 | 0.10864 | 0.11028 | 0.11046 |
| D256_S0_s5 | GRU_D256 | S0 | 0.002476 | 0.09217 | 0.10716 | 0.10858 | 0.10817 |
| D256_S1_s1 | GRU_D256 | S1 | 0.000533 | 0.09019 | 0.10243 | 0.11353 | 0.11271 |
| **D256_S1_s2** | GRU_D256 | S1 | 0.000783 | 0.09284 | 0.10522 | **0.11576** | 0.11609 |
| D256_S1_s4 | GRU_D256 | S1 | 0.001687 | 0.09893 | 0.10716 | 0.11211 | 0.11271 |

Stage 2 (best per size re-run at seed 2027; file self da08395b…5962): D128_S1_s4 endpoint 0.12592 (2-seed mean
**0.124844**); D256_S1_s2 endpoint 0.11481 (2-seed mean 0.115286). Mechanical choice (p2_driver): **tuned =
P2_GRU_D128_S1_s4** (GRU_D128, schedule S1, peak_lr 0.001687, dropout [0.1, 0.05], 20.96M params); tie rule not
applied (gap 0.0096 ≫ tie); CKPT_AVG not carried (gain +0.00016 < 0.0005).
Ops observations (not decisions): (a) the tuned LR is the highest S1 point; the S0 curve is flat at the top (s3 0.11148
vs s5 0.11107) and D256 declines above 0.00078, so no sign that the grid edge truncates a large gain — but it is an
edge. (b) The smaller size wins by ~0.0096; relevant to the owner's open 'bigger size' idea (not registered).

## 2. Registration (PROJECT_MEASURED)
`code_p3/v4train/p2_driver.py register-02b --family GRU --p2-dir runs_p2` → contracts/hp_registry/TUNED_RECIPES_02B.json
file e72b9ff8…4035, **self 63704173e6081b1c7fb38d4e52152898952026804ba07a922434c453848d0306**, version 1, recipe
02B_GRU sha bbe3b88f…096b, source_hp_id P2_GRU_D128_S1_s4 (stage files re-verified, selection recomputed by the code).
Minor defect found: register-02b imports v4train before inserting its path → run with PYTHONPATH=code_p3 (no effect on
content; fix queued).

## 3. Code and seal (condition 1)
code_p3/ = local reviewed tree (TRAIN-02R3 + TRAIN-03 + F1 fix; DECISION_LOG 90 ACCEPT_WITH_FIXES), pack tar
f89b9ff4…3b2b; 361 code files hash-verified on the server, 0 mismatches, 0 extras; code/ untouched (P2 SASRec still
uses it). Seal cite (code_p3/v4train/trainer.py:338, file 00d5d420…5235):
`self.art = self.run_dir / "sealed" if self.sealed else self.run_dir     # every metric-bearing artifact`
The ruling file (sha f2767229…926c) and contracts PREG1_AMENDMENT_03/04 were uploaded to the server (new files).

## 4. Disk (condition 5; MEASURED + INFERENCE)
Free now 22.80 GB = 21.23 GiB. Measured largest SASRec P2 run (D256_B3) in flight: snapshots 5.0/5.5 = 0.34 GB each +
latest.pt 1.01 GB → peak ≈ 2.0 GB, pruned to 3.7 MB at run end (one run at a time on GPU1). GRU_D128 is 20.96M params
(≈ 1/4 of the SASRec footprint); conservative bound: treat each GRU P3 seed like the SASRec run (≈ 2 GB kept, incl.
7.5/8.0 continuation) → 2 seeds 4 GiB. Projection: 21.23 − 2.0 − 4.0 ≈ **15.2 GiB > 8 GiB guard** (trainer guards stay
armed: refuse if free − peak < 8 GiB, pause if free < 10 GiB). P2 run dirs kept for all of P3 (≈ 52 MB GRU).

## 5. Planned action on approval
Write company/gates/G1B_APPROVED.json (PPSI_V4_G1B_APPROVED_1, part 1: {part 1, GRU, self 63704173…}; ruling path +
sha), upload it, then `p3_driver --family GRU --gpus 0 --out-dir runs_p3 --code-root code_p3` (dry run first). Nobody
reads a MAIN value until all four contest runs are done (seal).
