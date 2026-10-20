# 10 — Presentation guide (15–20 slides)

*Reuse the 30 Sep supervisor deck where possible: `company/presentations/PPSI_Supervisor_Deck_Final.pptx` (+ `DECK_SCRIPT_EN.md`
speaker script, `DECK_GUIDE_AR.md`, `SUPERVISOR_DECK_DATA.md`, build scripts `build_supervisor_deck*.js`). That deck predates
every MAIN / VALIDATION number. For the final talk, replace VALIDATION numbers by TEST numbers when the sitting is done and say
which split each number comes from. Labels on slides: "VALIDATION (max-of-marks)" or "TEST (single look)".*

| # | Slide | Content / figure | Talking points (source) |
|---|---|---|---|
| 1 | Title | PPSI: how much recommendation quality survives the move from cloud to federated / on-device | Queen's + Microsoft; owner; date |
| 2 | The question | verbatim question; "how much is retained", no threshold | DL 86 |
| 3 | Three ways a browser could learn | diagram: cloud-only / federated (weights only) / small local model (+ fine-tune) | deck slide 2; file 01 §2 |
| 4 | Data and task | REES46 Oct–Nov 2019, next different viewed product, 158,486-item full ranking, USER_HISTORY_L50, splits timeline | file 02 |
| 5 | Fairness and pre-registration | register → read once → lock → one TEST look; amendments timeline as a strip | file 03; deck slide 9 |
| 6 | Backbone contest | SASRec D256_B3 selected under rule T1 (2-seed PB means 0.1612 vs 0.1569); order reverses at the 6.0 endpoint | DL 129, 138 (contest values, final) |
| 7 | Models and payloads | C_BIG 83.9 M vs device model 20.7 M; 165.7 MB per FL visit vs 2.1 MB with frozen tables (S8); 170 MB cap | file 01 §3; REPORT §12b |
| 8 | Fixing federated learning | ladder S0 0.0235 → S4 0.1076 on CALIB (× 4.58): server optimiser, client SGD, LR shape, 1 pass | FL_IMPROVEMENT_LADDER; DL 271 |
| 9 | Pretrained start (realistic) | PRE on 1.2 M opted-in users; S7 / S8 warm screens; FA_WARM* = S8 | 06N; DL 292 |
| 10 | Headline result | bar chart of R % vs C_BIG / C_FULL: C_SMALL 95.8, FA 86.1, FA_WARM_FROZEN 107.0 (88.8 % of C_FULL), PRE 107.3, FT_PRE 109.7, LAST 44.1, LO 3.1 | file 04 §1, §3 (VALIDATION) |
| 11 | The primary pair | K20b 88.81 % [88.27, 89.36]; K21 −0.00043 [−0.00076, −0.00010] | "the pretrained federated system keeps ~89 % of the full-data cloud, but FL rounds did not add over θ_pre — under this recipe" |
| 12 | Why K21 ≈ 0 (D1) | table of ‖θ − θ_pre‖ / ‖θ_pre‖: 0.5–0.7 % (S8), DP runs ≤ 0.08 % | exploratory; recipe effect, not proof FL cannot help (DL 407) |
| 13 | Communication | quality vs bytes per client curve; FA_Q8 and S8 save traffic at no measurable cost (K14, K14w equivalent) | REPORT §5c, §9 |
| 14 | Privacy | privacy ladder L0–L6 with retention; DP cost small vs its twin but the cold DP schedule is weak (~21 % of C_BIG); warm DP ≈ PRE; trusted-aggregator DP; non-DP updates reveal items | file 04 §7; 06L_C C1–C2 |
| 15 | On-device | FT gains (K9, K9b, K9c, P5 ≈ +0.004); INT8 99.84 %; latency = pass 2 | file 04 §7–§8 |
| 16 | Phase 2 | design: W-grid (server optimiser × server LR × client LR, unfrozen tables, LoRA), C1 / C2 controls, Q1 reading grid; status | file 05 |
| 17 | External benchmarks | datasets, protocol, comparison rule, status (S3 ready to lock, ML-1M on GPU, Wave 2 on CPU) | file 06 |
| 18 | Limitations | single dataset for the main study; CALIB ⊂ MAIN; one θ_pre seed; no secure aggregation; simulated DP scale; TEST previously inspected; VALIDATION optimism | file 09 §2 |
| 19 | Answer and guidance | retention by system with CIs (TEST); deploy pretrained + on-device fine-tuning; FL only with a measured gain; payload / INT8 guidance | file 04 §10 |
| 20 | Future work / questions | teacher-student distillation, DP at full scale, real SecAgg, phones | DL 276 |

**Talking rules.** Name the split and the recipe with every number; never say "SASRec beats GRU", "FL retains X %" without the
recipe, or call a non-DP arm private; mark VALIDATION numbers as provisional until the TEST sitting; keep benchmark numbers
separate from REES46. Figures can be drawn from `results/VALIDATION_v3/` (REPORT tables, `curves/*.csv` on the server) and
`METRIC_TABLE.csv`; refresh them from the TEST outputs later.
