# G1b part 1 — Fable 5.1 ruling (verbatim; fresh agent, one message; received 2026-09-27T21:42Z; packet G1B_PART1_PACKET.md)

VERDICT: APPROVE_WITH_CONDITIONS

CONDITIONS:
1. trainer.py:338 only proves the redirect exists (`if self.sealed`). Before the real launch Ops cites the p3_driver line that sets sealed=True for MAIN runs, and the dry run prints a `sealed/` artifact path.
2. G1B_APPROVED.json is append-only: the part-1 entry (self 63704173…, 02B_GRU sha bbe3b88f…096b, source P2_GRU_D128_S1_s4) stays byte-identical when part 2 is added, and the part-2 registration must reproduce the same 02B_GRU sha.
3. code_p3/ (pack f89b9ff4…) stays frozen for all of GRU P3; the register-02b import fix goes elsewhere, not into code_p3/.
4. SASRec P2 grid/procedure hash unchanged from pre-registration through part 2 — nothing reacts to this GRU table.
5. Dry-run RunConfig hashes (both seeds) filed against the pinned recipe before the seed-2026 launch; disk guards stay armed.

DISCLOSURES:
- Grid edge: tuned peak_lr 0.001687 is the top of the S1 sub-grid (s5 ran only under S0). Flat S0 top and D256's decline above 0.00078 argue against a truncated gain; grid not extended (any extension must be pre-registered and applied to both families).
- D128 beat D256 by 0.0096 (2-seed means); the "bigger size" idea stays unregistered, not a post-hoc contest option.
- CKPT_AVG not carried (+0.00016 < 0.0005); register-02b needed the PYTHONPATH workaround.

BASIS: Selection was mechanical (p2_driver) from 12 clean stage-1 trials plus a seed-2027 confirmation, recomputed at registration; hash 63704173… is pinned. Ruling conditions 3, 5 and 6 are met; the disk margin (15.2 GiB vs 8 GiB guard) is ample. Condition 1 needs the seal shown engaged, not merely present — the planned dry run makes that cheap.
