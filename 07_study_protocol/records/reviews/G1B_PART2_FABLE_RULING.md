# G1b part 2 — Fable 5.1 ruling (verbatim; fresh agent, one message; received 2026-09-28 ~03:43Z; packet G1B_PART2_PACKET.md)

**VERDICT:** APPROVE_WITH_CONDITIONS

**PINS**
- Part 1: 63704173e6081b1c7fb38d4e52152898952026804ba07a922434c453848d0306 (unchanged)
- Part 1R: d8172874f4bc8a8913b69ce8b93bb1be3aee0dc10bce131bfc3782755c840d0a (02B_GRU_C2 = P2_GRU_D256_S1_s2, ea92d3d1…) — confirmed on disk in G1B_APPROVED.json
- Part 2: 0b0bc3b4d54d0444ab04045008d8401980ee724cae16a963598e59834175dc42 (C1 02B_SASREC = D256_B3_S0_s0 9a253087…; C2 02B_SASREC_C2 = D256_B2_S0_s0 9c9978c6…)
- P2 evidence: stage-1 2d948bbd…, stage-2 8bf4a706…

**CONDITIONS**
1. Real `register-02b --with-c2 SASREC` reproduces self 0b0bc3b4… exactly; 02B_GRU and 02B_GRU_C2 byte-identical to v2. Any mismatch: stop, no launch, new packet.
2. G1B → format _2, parts [1, 1R, 2], append-only, part-1 archive kept; written only outside a GRU launch window; record which registry/G1B hashes GRU C2 s2027's RUN_START pinned.
3. Four SASRec dry runs from code_p3b with the sealed=True line cited; each LAUNCH_0 config hash equals its dry run; label PRODUCTION.
4. GPU0 SASRec queue starts only after GRU C2 s2027 reaches DONE/DONE_CONTINUATION; continuation automatic on the flag; 8 GiB guard armed; no value read before G2.
5. DL 105 entered in the tuning ledger; cite the p2_driver log line showing stage 2 (8bf4a706…) was derived mechanically after stage 1 closed 01:21:39Z.
6. No LR-grid extension; MAIN_SCALE ruling stands.

**DISCLOSURES**
- SASRec's CALIB optimum is the lowest grid LR (4e-4) for all sizes; S0 curves peak at 4.0 and fall 0.002–0.005 by the registered 6.0 endpoint. Unextended, mirroring GRU's unextended high edge; MAIN's 4× more updates may shift both optima. Sealed 0.5-EFE MAIN marks reported post-unseal as diagnostic only.
- Both SASRec candidates are D256 (2 vs 3 blocks); D384_B2 dropped on CALIB only.
- SASRec > GRU on CALIB (0.132 vs 0.125) is tuning evidence, not the contest.
- Incident: one intermediate value of a non-candidate trial (D384_B2_S1_s2 @2.0) was seen before P2 closed; selection mechanical, procedure unchanged.

**BASIS:** Candidates follow amendment 05 §2 mechanically from the stage-2 means (0.1324/0.1301 > 0.1273), with GRU entries byte-identical. Staggered per-GPU queues are fair under STRICT provided the seal holds. The incident is immaterial: the value belonged to a dropped size and preceded the mechanical stage-2 derivation. Symmetric non-extension of both LR edges preserves fairness; the residual risk is a disclosure, not a design change.
