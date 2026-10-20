# Fable code review of vs_systems.py (owner-delegated), 2026-09-25

- **Runtime routing:** ['claude-fable-5-1']
- **Passes:**
  1. FIX_REQUIRED (2 blockers);
  2. APPROVE_FOR_GPU1.

ROUTING: claude-fable-5-1
VERDICT: FIX_REQUIRED

Checklist (files: `F:\Queens University\Microsoft Project\eda\.local_ai\sasrec_vs_systems_v1\code\vs_systems.py` = V, `...\code\run_gpu1.sh` = R, `...\code\snap\` = S):

- No claim/lock import [evidence]: V imports only `paths, numpy, torch, ensembles, eval_full, precision, sasrec_model, full_store`; nothing in S imports `lock_and_eval.py`; `snap/paths.py:39-41` puts only snapshot dirs on `sys.path`, never live `sasrec_v1/code`.
- VS-only split [evidence]: V:58 `SPLIT="VALIDATION_SELECT"`, V:357 `splits=(SPLIT,)`, V:359 `len(ds)==paths.N_VS`.
- Output root [evidence]: `W`=workspace (V:56); all writes go to `W/state`, `W/logs`, `W/results`; `RUNS` is only read (V:96,108,112,120,137). Nothing written under `sasrec_v1`; `PYTHONDONTWRITEBYTECODE=1` in R:8.
- TF32 off [evidence]: V:352 `configure_strict_fp32` (asserted, `snap/precision.py:28-34`) + `strict_eval()` at V:299/323; `CUBLAS_WORKSPACE_CONFIG` in R:8.
- Hook capture [evidence]: V:259-288 with `prefetch=False` (V:300); same `rank_targets` path as `evaluate_model` (eval_full.py:309-320); local test incl. negative control passed.
- Order check [evidence]: V:121 compares `decision_id_sha256` to `ds.sha256()`, raises on mismatch; consistency recorded V:305-308.
- Guard thread [evidence]: V:127-202; thresholds V:59 match the plan (10%, 10 GiB, 56 GiB, 12 GiB RSS, 12 GB VRAM); hard stop V:182; no-new-stage gate V:344 and V:372.
- Bootstrap unconditional [evidence]: V:377-380, both differences regardless of sign.
- Rank −1 counted as miss [evidence]: `evaluator.py:75`.

Blockers:
1. V:106-107 vs V:110 [inference]: A1 e3/e4 WEIGHTS come from the base run (`A1_RICH256_s2026`), but A1 e3/e4 RANKS come from the cool5 summary (`a1c["ranks"]["3"/"4"]`). If the cooldown re-trained epoch 4, those ranks belong to different weights than the S4a member, and the P0 pairwise numbers would be mislabelled silently. Fix: take e3/e4 ranks from `a1b["ranks"]`, or assert `a1c["ranks"][e]["sha256"] == a1b["ranks"][e]["sha256"]` for e in {3,4}.
2. R:10 [evidence]: `>> $V/logs/vs_systems.log` runs before Python creates `logs/` (V:132); the workspace mirror has no `logs/`, so the launch would die with "No such file or directory". Fix: `mkdir -p "$V/logs"` before `exec`.

Non-blocking notes:
- V:69 `ts()` uses `mktime - time.timezone`; correct on a UTC pod, fires one hour EARLY (safe direction) under DST. Prefer `calendar.timegm`.
- V:189 re-baselines only if `self.phase is not None`; if GPU0 status is stale (>900 s) at start, the throughput guard is silently disabled for the run. During a 3-min re-baseline (V:192) no resource/hard-stop checks run, so the 03:00Z stop can lag by up to 3 min.
- V:167 cgroup v2 path only; on v1 the cgroup check is silently skipped (RSS check remains).
- V:114 A0_e5/A1_e5/A1_e4 weight shas are recorded, not cross-checked against summary.json (plan requires verification only for the rank files; A1_e3 is verified, V:108).
- V:373 if S4a is time-gated, the S2 bootstrap is also not written.
- Missing context [inference]: the campaign writer of `ranks/epoch5.npz` is not among the named files, so I could not confirm `decision_id_sha256` is stored as 32 raw uint8 bytes as V:121 assumes; a format mismatch fails closed (RuntimeError before any GPU stage).

---

ROUTING: claude-fable-5-1
VERDICT: APPROVE_FOR_GPU1

Re-check of the changed files (`F:\Queens University\Microsoft Project\eda\.local_ai\sasrec_vs_systems_v1\code\vs_systems.py` = V, `...\code\run_gpu1.sh` = R):

Blocker 1 — resolved [evidence]: V:106-114 takes each member from `summary.json["checkpoints"]["best_weights"]`, verifying both the epoch (A0 cool5 = 5, A1 cool5 = 5, A1 base = 3) and the sha, raising on either mismatch. V:117-119 asserts the cool5 copies of the A1 e3/e4 rank files have the same sha as the base run's; V:120 then sources e3/e4 ranks from the base summary, i.e. the weights' own run. V:115 records the A1 e4 weight sha (no summary field exists). V:129-133 still applies the `decision_id_sha256` order check to every loaded rank file.

Blocker 2 — resolved [evidence]: R:7 `mkdir -p "$V/logs" "$V/state" "$V/results"` runs before the `cd` and the `>>` redirect at R:11.

Non-blocking notes — all addressed [evidence]:
- V:70 `calendar.timegm`, so the 02:00Z/03:00Z gates are exact regardless of pod TZ.
- V:163 `check_resources()` inside every baseline iteration: disk/RAM/VRAM/hard stop are checked during the 3-min (re)baseline windows too.
- V:200-202 retries the baseline while it is `None` and a GPU0 run is visible, instead of leaving the throughput guard disabled.
- V:386-388 `break` on the time gate; V:392-396 bootstraps every completed system; V:398 DONE vs PARTIAL.

Format check [evidence]: `F:\Queens University\Microsoft Project\eda\.local_ai\sasrec_v1\code\train_sasrec.py:371` writes `decision_id_sha256=np.frombuffer(bytes.fromhex(ds.sha256()), dtype=np.uint8)`, so V:131 `bytes(...).hex()` reads the same 32 raw bytes.

Unchanged and still passing: no claim/lock import, VS-only split (V:59, V:371-374), workspace-only writes, strict FP32 (V:366, V:313, V:337), hook capture with `prefetch=False` (V:314), abort limits (V:60), unconditional bootstrap.

Residual, not blocking [inference]: I cannot recompute the sha256 values in `review\REVIEW2_HASHES.json` (read-only reviewer); the lead should confirm the server copies match `3cc11d89…` (V) and `820fb52c…` (R) before writing `CODE_REVIEW_PASS.json`. The cgroup v1 path is still silently skipped (V:177-182); the RSS and VRAM checks remain.
