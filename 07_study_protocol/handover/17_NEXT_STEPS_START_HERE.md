# 17 — NEXT STEPS: where you start and what you do (final, 2026-10-10)

*For the colleagues who continue the project. The owner (Ahmed Abdelhamed) has left the project. This file is the entry point. Read it first, then the files it points to.*

> **ملخص بالعربية.**
> - **الحالة:** كل التجارب خلصت. النتائج التأكيدية نهائية، والتجارب الاستكشافية كلها خلصت ومعاها اختباراتها.
> - **المطلوب منكم:** دمج فروع GitHub، ثم كتابة الرسالة، ثم التسليم النهائي.
> - **السيرفر:** فاضي. ممنوع تلمسوا النتائج المسجّلة.
> - **بأي ترتيب:** القسم 2 بالترتيب، والقسم 4 فيه القواعد.

## 1. State on 2026-10-10 (nothing is running)

- **Confirmatory.** The REES46 TEST sitting (2026-10-07) and the 6 benchmark TEST looks are FINAL.
  - Headline: the realistic Edge system (federated, warm-started from a server pretrain) keeps 88.5 % of the full-data cloud model.
  - Federated training adds nothing over a strong 1.2M-user server pretrain.
  - On-device fine-tuning adds a small, significant gain. Device-only training fails (1.6 %).
- **The tau finding.** The registered FedAdam tau = 1e-3 froze federated learning (updates below tau make the server step tiny). Fixed with tau = 1e-6:
  - on the benchmarks, FL climbs from 13–77 % to 75–92 % of the cloud model on TEST;
  - on REES46 with the strong pretrain, nothing changes, so the confirmatory answer is robust.
- **Value of FL depends on the server-side data.**
  - With a 100k-user pretrain, FL adds about 5 retention points.
  - With a disjoint server band of 50 % on the S3 benchmarks, the warm start beats cold by 26–53 %.
- **Privacy.** FL that really learns leaks membership (E1 AUC 0.75); DP at ε = 8 removes the leak.
  - The DP cost is large on a cold start and small on a warm start.
- **Capacity.** A central D64 model reaches 96–110 % of the central D256. FL at D256 is no better than at D64. So the federation cost (≈ 15–22 % at matched size) is the real gap, not model size.

All numbers with CIs: `14_UPDATE_v2_20261008.md` §1–2c, and on GitHub under `results/` and `docs/results.md`.

## 2. What to do, in order

1. **GitHub status (updated 2026-10-10 23:00Z).**
   - PRs #103–#106 are MERGED.
   - **FINAL: PRs #107–#112 are MERGED as well** (merge commits f5bcff6, 1244654, 537230f, e8917e3, 50d9cea for #108–#112), so `main` holds everything. No open PRs remain from this work. The list below is kept for reference:
   - PRs #107 (result-table corrections) and #108–#113: #108 secure aggregation, #109 membership-inference audit, #110 contrast statistics, #111 warm-start recipe, #112 REES46 protocol doc. PR #113 (`docs/next-steps.md`) was closed without merging by the owner's decision: a team-handover page does not belong in the public repo, and this file replaces it.
   - **Check on github.com that #107–#113 show "Merged".** If one failed its checks, fix the test on its branch and merge it yourself, in that order.
   - The text below is kept for reference.
1. **GitHub: merge the pushed branches into `main` through pull requests.** (Historical note; see the status above.)
   - The branches are pushed, but no PR exists yet: the operator's machine had no GitHub CLI and must not handle tokens.
   - Open one PR per branch on github.com, using the descriptions in §3, in this order:
     1. `fix/fedadam-tau`
     2. `results/rees46-final`
     3. `results/benchmarks-final`
     4. `docs/final-results-report` (its notebook reads the two results branches' CSVs)
     5. `docs/next-steps` and any later topic branches listed in `docs/next-steps.md` on GitHub
   - `feature/benchmark-runner` is already in `main` (PR #102); its extra merge commit changes no file, so no PR is needed.
   - Use the repo's usual "Create a merge commit".
2. **Thesis write-up.**
   - Report the confirmatory results exactly as printed.
   - Put the exploratory results **beside** them, labelled "exploratory, after the test look", never instead of them.
   - Disclosures to include: the tau finding; θ_pre is a single seed; FedProx is inert at 1 local step; FROZEN arms score 0.0 for out-of-band targets; the extra exploratory TEST exposures (register entries 12–19); the device-class difference on ML-1M.
   - The structure is in `09_THESIS_WRITING_GUIDE.md`; the published comparison is in GitHub `results/benchmarks/`.
3. **Optional exploratory TEST looks** for the disjoint-band (A1), capacity (A2) and DP8 / PB100-DP8 studies. They are validation-only today. Each needs:
   - its own exploratory lock;
   - an independent review;
   - a BIND register entry;
   - scoring with the gated wrappers;
   - a shas-only log row;
   - an inventory check;
   - a LOOK entry.

   Templates: `company/gates/EXP_*`; procedure: `15_TEAM_TASKS_AND_SERVER_FOLLOWUP.md` §3–4. Skip these if time is short; the validation results already answer the questions.
4. **Wrap-up.**
   - Optionally capture `.local_ai/thesis_reference` sources with `tools/add_source_batch.py`.
   - Final presentation: `10_PRESENTATION_GUIDE.md`.
   - Archive this ZIP.

## 3. Pull-request descriptions (copy and paste)

- **fix/fedadam-tau** — *Validate the FedAdam tau and document why tau = 1e-3 froze the federated runs.*
  - FedAdam divides each server update by √v + tau. Our pseudo-gradients are around 1e-7, so with tau = 1e-3 the step collapsed to update / tau and the global model barely moved: about 0.06 % drift over 13k rounds.
  - This PR validates `fl.server_tau` in the recipe and adds a unit test that shows the effect. It also adds `docs/fedadam-tau.md`.
  - The default stays 1e-3 for reproducibility; 1e-6 is recommended. Tests: fedsim + benchrun pass.
- **results/rees46-final** — *Add the final REES46 result tables.*
  - The confirmatory TEST results: retention contrasts with 95 % CIs, and the Phase-2 contrasts.
  - The exploratory re-runs: tau-fixed warm and cold FL, and the 100k-user pretrain.
  - Privacy (membership-inference audit, DP) and device summaries.
  - CSVs with a README. The `.gitignore` gets an exception for `results/**/*.csv`.
- **results/benchmarks-final** — *Add the benchmark result tables and the published comparison.*
  - Confirmatory TEST per dataset (S3 Beauty / Sports / Toys, Amazon-2023 Video Games / Baby Products, ML-1M).
  - The exploratory fixed-recipe FL test, with the central D64 reference.
  - The disjoint-server-band and capacity validation studies.
  - The comparison with Spotify's published Table 1 (full ranking).
- **docs/final-results-report** — *Write the final results report and notebook.*
  - `docs/results.md`: question, setup, results, the tau finding, privacy, devices, limitations.
  - `notebooks/final_results.ipynb`, with figures from the results CSVs. Merge after the two results branches.

## 4. Rules (do not break)

- Never edit, re-run or re-select confirmatory artifacts, locks or the register history.
- No TEST evaluation without the full gate.
- Log each action as a new row in `company/DECISION_LOG.md`, which is append-only.
- Never put secrets in files.
- On GitHub, use your own accounts. There are no direct pushes to `main`.
- Server rules: `16_TEAM_SERVER_README_20261010.md`. They are also on the server as `CLAUDE.md` / `AGENTS.md`, so AI agents read them automatically.

## 5. Where things are

| What | Where |
|---|---|
| This archive (code, results, records) | `PPSI_V4_Handover_…_FINAL.zip` → `ppsi_v4/` |
| Results narrative | `HANDOVER/14_UPDATE_v2_20261008.md` |
| Task details and server follow-up | `HANDOVER/15_TEAM_TASKS_AND_SERVER_FOLLOWUP.md` |
| Server rules for people and agents | `HANDOVER/16_TEAM_SERVER_README_20261010.md` |
| Full action record | `company/DECISION_LOG.md` (rows up to 586+) |
| Server root | `/home/jovyan/ahmedabdelhamed01/Privacy-Preserving-Shopping-Intelligence/.local_ai/ppsi_v4` (idle; outputs under `runs_*`) |
| GitHub | `microsoftproject26/Privacy-Preserving-Shopping-Intelligence`: branches above; `results/`, `docs/results.md`, `docs/next-steps.md` |
