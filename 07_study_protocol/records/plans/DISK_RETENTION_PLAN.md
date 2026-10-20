# DISK_RETENTION_PLAN — PPSI V4 (OPS-04; Fable G0 blocker 1). Transfer measurement done (14:42Z).

## Measured state (PROJECT_MEASURED; server volume 48.9 GiB, shared by the three team members)
- **Free:** 12.81 GiB at 13:31Z; `df` showed 13 G at 14:1xZ.
- **Policy:** the target is ≥ 10 GiB free; the floor is 8 GiB plus the next atomic temp file (package 10 §3). The
  budget above the floor is ≈ 4.8 GiB, and ≈ 2.8 GiB above the target.

| Use | Size | Note |
|---|---|---|
| Frozen dataset `shared_data/rees46_nextitem_v1` | 8.6 GB | immutable |
| venvs | 5.7 GB | shared |
| `~/.cache` | 5.6 GB | shared account; not ours to touch |
| gru_v1 | 9.7 GB | 6.4 GB `cache/` (regenerable window caches), 3.1 GB experiments |
| sasrec_v1 | 7.8 GB | 5.1 GB runs, 2.4 GB runs_v2 |
| feature_opt_v1 | 0.64 GB | |
| ppsi_v4 | 0.48 GB | mostly the snapshot plus export ONNX files: 317 + 162 MB, regenerable |

- **Transfer channel (PROJECT_MEASURED 14:1x–14:42Z):**
  - the kernel-channel chunked copy runs at 0.07 MiB/s GET and 0.05 MiB/s PUT on a 16 MiB probe (a 332 MB
    checkpoint takes ≈ 80–110 min);
  - the Jupyter contents-API HTTP GET of a 64 MiB probe (exposed through a hard link, because the API refuses hidden
    directories) did not finish in ≈ 30 min and was stopped. The probe and the staging link were removed.

  **Offloading checkpoints to the local PC is not practical. Option (b) below is withdrawn;** the plan is
  server-only.

## Checkpoint sizes (PROJECT_MEASURED from SYS-01)
| | GRU | SASRec |
|---|---|---|
| Weights, FP32 | 84 MB | 332 MB |
| Resumable (weights + AdamW m, v) | ≈ 252 MB | ≈ 1.0 GB |
| FA/FP resumable (optimizer resets every round, so no optimizer state) | ≈ 84 MB | ≈ 332 MB |
| PF resumable (θ + p_u for 262,144 clients × dim + p_u AdamW m, v) | 84 MB + 262,144 × 128 × 4 × 3 = 84 + 403 MB | 332 MB + 262,144 × 256 × 4 × 3 = 332 + 805 MB |

## Retention rules (base plan: server-only)
1. **Calibration runs** (LR ×2 per family, FP μ ×3, PF λ ×3): metrics and curves only, with **no weights retained**.
   A resumable checkpoint exists while the run is live and is deleted when it completes. Only files that run created
   are removed, and the manifest keeps their sha256.
2. **Central contest runs** (2 families × 2 seeds):
   - while live: one `latest` resumable, replaced atomically, plus `best` weights and the `endpoint_6.0` weights;
   - after completion: keep `best` and `endpoint_6.0` weights; delete `latest` (it is regenerable only by
     re-running).

   Peak, two runs at once: SASRec 1.0 (latest) + 1.0 (atomic temp) + 0.33 + 0.33 ≈ 2.7 GB; GRU ≈ 0.7 GB; together
   ≈ 3.4 GB. That fits within the 4.8 GiB above the floor, but only with nothing else running.

   Retained after the contest: 2 × 0.66 (SASRec) + 2 × 0.17 (GRU) ≈ **1.66 GB**. The losing family's weights are
   kept, as the package requires.
3. **Winner downstream runs** (FA, FP, PF, CPF × 2 seeds; C reused from the contest; LO procedural with no weights):
   retain `best` and `endpoint` weights only.
   - If GRU wins: 8 × 0.17 ≈ 1.35 GB. Fits.
   - If SASRec wins: 8 × 0.66 ≈ **5.3 GB**, plus the live resumables (PF ≈ 1.1 GB). This **does NOT fit** alongside
     the contest's 1.66 GB. Options, in order of preference:
     - (a) The owner approves removing regenerable old caches, i.e. `gru_v1/cache/` (6.4 GB of window caches that
       can be rebuilt from v1 by the gru_v1 code). This is the owner's decision because it sits in a prior workspace.
     - ~~(b) Offload completed checkpoints to the local archive~~: withdrawn, because the measured channels are too
       slow.
     - (c) Retain only the `endpoint_6.0` weights plus a deterministic **re-derivation recipe** for `best`, and
       re-train `best` only if TEST needs it. This is discouraged: it spends compute.
4. **Evaluation rank files:** int32 ranks per evaluation (about 3.2 MB for 808k VALIDATION rows) are retained. They are
   small.
5. **Guards:**
   - **Refuse:** the supervisor refuses to start a run when free − (planned peak including the atomic temp) < **8 GiB**
     (the floor).
   - **Pause:** it pauses scheduling new runs while free < **10 GiB** (the target).
   - **Why:** this corrects the earlier rule, "refuse if free − peak < 10 GiB". Fable G0(b) showed that rule would
     block the planned concurrent contest (12.81 − 3.4 = 9.4 GiB).
   - Every write is flush + fsync + reload/validate + atomic replace.
   - No evidence is ever overwritten.
   - Everything removed is logged with its sha256 (only files this workspace created and marked regenerable).

## Owner decision (received 2026-09-26; DECISION_LOG item 19)
**APPROVED, conditionally.** IF AND ONLY IF SASRec wins G2, delete ONLY the regenerable `gru_v1/cache/` directory.
- **Before deleting:** write a complete path/size/sha256 manifest and verify it by re-hashing every file.
- **Never delete:** historical experiments, final model weights, locks, reports or evidence.
- **If GRU wins:** nothing is deleted.

## Update 2026-09-26T18:33Z (OPS-08)
- **Measured:** free space was 11.12 GiB before the reclaim and 12.10 GiB after it (18:3xZ). The reclaim removed the SYS-04 probe dumps, 1.05 GB (manifest `runs_g1/ops08_reclaim/`).
- **Contest margin:** 12.10 − 3.4 = 8.7 GiB, which is above the 8 GiB floor. The volume is shared with teammates, so the supervisor's guard stays authoritative.
- **Rule:** probe and smoke jobs keep digests only and delete their own dumps and weights inside the same job.

