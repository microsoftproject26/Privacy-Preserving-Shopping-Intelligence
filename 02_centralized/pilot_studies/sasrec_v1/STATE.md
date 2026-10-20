# STATE.md — SASRec campaign v1 (registry v2, owner 38 h deadline)

**Phase:** COMPLETE (FINAL STAGE DONE 2026-09-26T07:24:17Z; lead report 07:37Z). Nothing may be re-run: RETRO_CLAIM refuses a second VC/TEST look. Owner offline since ~2026-09-23T23:05Z; Fable 5.1 is the owner's delegate for in-scope decisions (owner directive, review/AUTO_DECISION_*).

## Milestones
- [x] ROUTING_VERIFIED: transcripts show Fable = `claude-fable-5-1` (Read/Grep/Glob only) and all 3 workers = `claude-sonnet-5` with no nested agents (AGENT_ROUTING.json)
- [x] PLAN_REVIEWED: Gate P APPROVE_WITH_FIXES (9 fixes accepted); owner rejected the 5–7 day plan → Gate P2 → owner chose "TF32 + 6 epochs"; registry v2 locked, sha 74414b38…
- [x] CODE_REVIEWED: Gate C FIX_REQUIRED (evidence attribution only) → attested run → correction review **APPROVE_FOR_REAL_TRAINING** (review/FABLE_GATE_C_RESPONSE.md); review/GATE_C_PASS.json pins the training code + runtime files
- [x] CORRECTNESS_PASS: full suite 112/112 on the exact approved hashes; tiny end-to-end + crash/resume bitwise-identical + mechanics (tests/evidence/GATE_C_ATTESTED_EVIDENCE.json)
- [x] TF32 parity gate (pre-registered review/TF32_PARITY_PREREG.md): all 4 criteria pass (loss Δ 1.6e-5 vs seed 5.9e-3; cohort Δ 8.8e-5 vs seed 2.2e-4; grad ratio 1.0001) → Fable ruling **TF32 ACCEPTABLE** (review/AUTO_DECISION_TF32.md; bench/TF32_PARITY.json)
- [x] SYSTEMS_LOCKED (bench/SYSTEMS_LOCK.json + SYSTEMS_LOCK_NOTES.md)
- [x] v2 RUN (00:18Z–17:05Z) FAILED: training pathology (common-mode drift of the tied item table; rows ~162 vs GRU 11–13; loss rising 7.55→9.0; VS macro MRR@20 A0 0.091, A1 0.118). Stopped at epoch 5 by Fable (owner delegate, owner online); artifacts archived intact in runs_v2/, state/v2/.
- [x] Fable deep review + web research (review/deep/FABLE_DEEP_REVIEW_RESPONSE.md) → v3 recipe: untied output table + per-step recentering, input LN, no √d, betas (0.9,0.999), peak 1e-3, 4 epochs. Registry v3 sha a303bc3f…; delta approved (review/deep/FABLE_V3_DELTA_RESPONSE.md); attested 113/113; GATE_C_PASS v3.
- [x] v3 RUNS_IN_PROGRESS since 17:28Z (A0 GPU0, A1 GPU1; 4 epochs, eval 3–4). ETA train end ≈ 08:15Z / 08:34Z Sep 25.
- [x] 20k go/no-go GO both arms (review/V3_GATE_20K.json; amendment 1 criterion fix decided before the data point)
- [x] v3 COMPLETE 4 epochs: full-VS macro MRR@20 A1 e3 0.18142 / e4 0.17976; A0 e3 0.17376 / e4 0.17282 (GRU 0.17387, TRANS 0.16164). Queue stopped before LOCK (no VC/TEST touched).
- [x] Owner (2026-09-25): "5 epochs" + "best accuracy" → Fable research (review/deep/FABLE_HIGH_RECALL_RESPONSE.md): floor 5th epoch NO-GO; cooldown epoch 5 + seed 2027 + pre-registered systems S1–S4; amendment 2 (config/run_registry.v3_amendment_2.json, 09:24Z); new deadline 2026-09-26T12:20Z.
- [x] continue_cooldown.py approved (review/GATE_C_COOLDOWN_PASS.json, sha 2f33832f); A1_s2026_cool5 running GPU1 since 09:34Z; A1_RICH256_s2027 running GPU0 since 09:21Z; orchestrator tmux sasrec-v3b (tools/campaign_v3b.py, state/v3b_state.json).
- [x] final stage code APPROVED by Fable (3 passes; review/deep/FABLE_FINAL_STAGE_REVIEW.md; review/GATE_FINAL_STAGE_PASS.json pins shas); tools/final_stage.sh ARMED 09:49Z — the orchestrator runs it once all training is settled (or at the 06:30Z Sep 26 hard stop)
- [x] A1_RICH256_s2027 20k go/no-go GO (review/GATE20K_A1_RICH256_s2027.json, 09:51Z)
- [x] A1_s2026_cool5 gate GO (cohort 0.18294/0.18510 ≥ 0.1806); COMPLETE 13:41Z e5 0.186298 → cooldowns_go = true
- [x] A0_ID256_s2026_cool5 COMPLETE 17:40Z (e5 0.178585); A1_RICH256_s2027 COMPLETE 00:48Z (e3 0.181504); A1_RICH256_s2027_cool5 COMPLETE 05:04:31Z (e5 0.186513)
- [x] VS-descriptive S2/S4a on idle GPU1 (../sasrec_vs_systems_v1; Fable GO_WITH_CHANGES; not used for selection)
- [x] MODEL_LOCKED 2026-09-26T05:05:18Z: headline A1_RICH256_s2027 e5 (VS-only rule), 8 entries, none dropped, sha 004130c6…a47d0; RETRO_CLAIM 05:05:21Z
- [x] RETROSPECTIVE_EVAL_COMPLETE 07:24:16Z (retro_eval/RETRO_RESULTS.json sha 1ac73dcd…; fingerprint gate 8/8). TEST macro MRR@20 headline 0.170475 vs GRU F0 0.168695 (+0.001780, CI [+0.001538, +0.002022]); vs TRANS +0.022871; A0 0.166484 (< GRU F0, descriptive); best system S3 0.173224. PREVIOUSLY INSPECTED.
- [x] REPORT_COMPLETE 07:37Z: FINAL_SASREC_REPORT_AR.md (+ _EN companion), FINAL_SASREC_RESULTS.json, SOTA_CLAIM_STATUS.md (lead, from result files only: tools/build_final_results.py + tools/write_final_report.py). The automatic v2-era render (wrong bootstrap key → 'Not yet measured') is preserved in final/auto_render/.

## Plan (registry v2)
- A0_ID256 (GPU0) ‖ A1_RICH256 (GPU1), seed 2026, full TRAIN 40,180,608, exact [B,K] CE, batch 256.
- S′ schedule: warmup 2000 → peak 5e-4 through epoch 1 → cosine to 5e-5 over epoch 2 → floor epochs 3–6.
- Full-VS strict-FP32 eval at epochs 3–6; best = raw max. No C0/C1/R0, no replicas.
- Deadline 2026-09-25T12:20Z. Cut rule after epoch 2 (systems only, both arms equal, ≥4 epochs).

## Resume / operations
- Launch: `bash $S/tools/launch_campaign.sh` (tmux `sasrec-queue`, restart loop; supervisors in tmux `sasrec-<run>-s2026`).
- Status: `python code/queue_runner.py --status`; `runs/<run>_s2026/status.json` (heartbeat + ETA); `logs/resources*.jsonl`.
- Never launch a second queue runner (PID lock). Resume = rerun the launcher after a pod restart; the supervisors resume from latest/previous.pt.

## Owner standing instruction
Never stop on a blocker. A real decision point → one concise Fable call, act on it as the owner's decision, record it. Hard limits: frozen data, GRU artifacts, teammates, no push/commit, shared env, VC/TEST only retrospectively after the SASRec lock.
