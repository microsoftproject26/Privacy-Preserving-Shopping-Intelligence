# G0(b) — protocol approval packet for Fable (Clean V2 freeze). Facts, options and questions only.

You are Fable 5.1, the Research Director. You are read-only and your ruling is binding.
- **Context:** you approved the G0 plan with fixes (`company/reviews/G0_PLAN_FABLE_POSITION.md`). Wave A has written
  the contracts and closed your seven blockers with receipts.
- **Independence:** do NOT open `company/reviews/G0B_PROTOCOL_OPUS_POSITION.md`; form your own view first.
- **Root:** `F:\Queens University\Microsoft Project\eda\.local_ai\ppsi_v4\`.

## A. Contracts to approve (read them; judge them against your G0 decisions and package 03, 04 and 09)
- `contracts/DATA_CONTRACT_CLEAN_V2.md` and `contracts/FEATURE_SCHEMA_CLEAN_V2.json` (handoff `company/handoffs/DATA-03.md`)
- `contracts/EVALUATION_CONTRACT_V4.md` (R1) (handoffs `EVAL-01.md`, `EVAL-01R1.md`)
- `contracts/CLIENT_CONTRACT_V1.md` (R1)
- `contracts/MANIFEST_SPEC_V4.json`

## B. The seven blocker receipts
| # | Blocker | Receipt | State |
|---|---|---|---|
| 1 | Disk retention plan | `company/plans/DISK_RETENTION_PLAN.md` | server-only; bulk transfer measured impractical; an owner decision is needed only if SASRec wins |
| 2 | Executed recipes | `g0/receipts_res02/EXECUTED_RECIPES.{json,md}` | transcribed with file:line |
| 3 | Boundary definitions | `g0/receipts_qa01/BOUNDARY_DEFINITIONS.md` | code: `pipeline5.py:122-132`; no target, so masks are false |
| 4 | Cohort receipts | `g0/receipts_qa01/COHORT_RECEIPTS.json`, `COLD_STRATA.json`, `COLD_H_PANEL.json` | 262,144: 801,762 rankable / 6,848 OOV; 133,154 classes with ≥ 1 positive |
| 5 | Code hash and flags | `g0/receipts_qa01/CODE_AND_FLAG_CHECKS.json` | the code hash matches the frozen build; category_code_missing ≠ main_category_missing on 0 rows |
| 6 | FL/LO mechanics smoke | `company/handoffs/FL-01.md`; `code/tests_fl/receipts/*` | 67 passed and 26 strict-xfail negative controls, independently re-run by Ops (exit 0); μ = 0 ≡ FA bitwise; p = 0 recovery; worker restore; 2-worker aggregation bit-exact. CPU only; the GPU receipt is PENDING |
| 7 | LO determinism | `code/tests_fl/receipts/LO_DETERMINISM_1024.json` | 1,024 clients reproduce exactly, CPU |

Clean V2 lock receipts:
- `g0/receipts_qa02/OI03_F1_PRICE_REZERO.json`
- `OI04_M1_REPRODUCTION.json` (bit-exact)
- `OI06_RC_VALIDATION.json` (0 contradictions on merged VALIDATION)
- `OI07_FILE_INTEGRITY.json` (73/73)

## C. Decisions requested
1. **Approve the contracts?** APPROVE / APPROVE_WITH_FIXES / FIX_REQUIRED, with exact fixes.
2. **F-1** (DATA-03 §6; QA-02 OI-03): `current_minus_recent_mean_log_price` is computed from an unknown current price
   in 117,315 of 52,760,314 TRAIN rows. The proposal is to extend the re-zero gate and refit on valid rows only (mean
   0.002066, population std 0.994534; clip bounds as recorded). Inside or outside your approved M1 scope? Adopt or
   reject?
3. **P3 (stopping).** Your decision 6 says both "run to 6.0 EFE" and "patience 2 after 3.0". One reading is: always
   complete 6.0; PRACTICAL_BEST is the best 0.5-EFE checkpoint in [0, 6]; patience and tolerance govern only a 6→8
   continuation. Confirm, or give the rule.
4. **P11 (cold-h panel), EVALUATION_CONTRACT §11.1 / CLIENT_CONTRACT C3.4:**
   - A = a cold-only subset of the cold VALIDATION users;
   - B = QA-01's panel: 16,384 users with ≥ 50 VALIDATION events (13,767 warm, 2,617 cold), salt
     PPSI_V4_COLD_H_20260926, 45,572 qualifying.

   If B, EVAL-01 proposes splitting it into COLD_START and WARM_TRUNCATION panels, never pooled. Decide, including
   the panel size.
5. **P12:** the 140 AMBIGUOUS_BOUNDARY VS→VC rows. Keep them in CEN_LEGACY_BARRIER, or move them to
   CEN_AMBIGUOUS_NEXT_BARRIER? No total changes.
6. **P4 / P5 / P6** (EVALUATION_CONTRACT §11): C_ref for events-to-X%; the few-shot recipe-selection panel; h = 0
   semantics; the FIXED_20 addition; one query per user per sub-panel.
7. **FL-01 mechanics choices** (handoff §"Risks, assumptions and choices"):
   - (1) the server recenters θ0 before the first broadcast (SASRec);
   - (2) a separate gradient clip for p;
   - (3) the CPF penalty counts each distinct user once per step;
   - (4) a frozen `n_shards` (proposal: 2, one per GPU).
8. **Anything else that blocks the lock.** Known non-Fable items are:
   - OI-05: the model input widths are hard-coded, a Wave B code task;
   - the GPU determinism receipt, which belongs to G1;
   - P13, the single TRAIN_END BOUNDARY_UNRESOLVED row, which is a non-blocking data question.

## D. Return format (target ≤ 1,200 words)
ROUTING, VERDICT, numbered DECISIONS with evidence, LOCK_VALUES delta (only what changes or is newly fixed), BLOCKERS,
OPEN_DISAGREEMENT_RISKS. Read-only.
