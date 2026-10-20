# PREG1 amendment 06O — descriptive C_FULL / PRE evaluation on the non-MAIN new-user panel (registration only)

- **Status:** REGISTERED 2026-09-30 ~21:35Z (server clock) by the Operations Director on the OWNER's request ("give me a picture of
  C_FULL without opening MAIN"). No MAIN value of C_FULL, PRE or any FL arm has been read; nothing in this amendment reads MAIN or TEST.
- **Panel:** role UNSEEN_VAL = UNEW_VAL_SUPPORT_32768 → UNEW_VAL_VALIDATION_32768 (12,792 users with evaluation rows). Its users are
  disjoint from MAIN, CALIB and the pretraining band (06K §1), hence from the whole C_FULL band (= pretraining band ∪ MAIN). PRE's own
  descriptive marks on this set are already read (DL 284: θ_pre endpoint 6.0 = 0.173081).

## O1. What is added
- reeval admits, on UNSEEN_VAL only: C_FULL s2026 and s2027 (best = PRACTICAL_BEST, endpoint_6.0 = CONTROLLED_BUDGET) and PRE (θ_pre =
  endpoint_6.0), with every existing source check (FIX-REEVAL binding, checkpoint sha, pair, roles).
- Rows are labelled DESCRIPTIVE with `preview_reference: NON_MAIN_UNSEEN_VAL`. They never enter a contrast, a selection, a stopping
  rule or a claim; they are reported in one descriptive table ("central models on never-trained new users").

## O2. What this must not change
- No registered rule, contrast, selection or the FT_FA selection on UNSEEN_VAL (06A / 06E) depends on these values; the FT_FA selection
  stays the mechanical rule over its own FT cells.
- C_FULL remains the upper-bound reference for 06N K20b on MAIN; its MAIN values stay sealed until the §9 read.
