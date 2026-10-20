# Fable Gate P response (verbatim)

- date: 2026-09-24
- agent type: built-in `claude-code-guide` (tools Read/Grep/Glob/WebFetch/WebSearch; no shell, edit or agent tools), model override `fable`
- transcript model ids: ['claude-fable-5-1']
- tool calls: ['Read', 'Read', 'Read', 'Read', 'Read', 'Read']
- usage (sum over turns, as exposed by the harness): {"input_tokens": 260, "output_tokens": 18854, "cache_read_input_tokens": 153590, "cache_creation_input_tokens": 85152}

---

ROUTING: claude-fable-5-1
TOOLS_USED: Read

**VERDICT:** APPROVE_WITH_FIXES

**MUST_FIX** (ordered; all before Gate C unless marked prereg)
1. [evidence] D1 keep. Store emits `position_ids` 1..len / 0 on PAD; model uses `pos=(position_ids-1)*mask`, asserted equal to `cumsum(mask)-1` on real cells; PAD rows contribute exactly 0 (table [50,d]). Window-start anchoring stays; recency-anchored positions recorded as future work.
2. [evidence] D9 is unenforced as written: A16 is Ampere, cuDNN TF32 defaults on. Assert `matmul.allow_tf32=False`, `cudnn.allow_tf32=False`, `set_float32_matmul_precision('highest')`, deterministic-algorithms flag; log values in the run record; TF32 canary test.
3. [inference] D3 partner liveness: explicit status file {RUNNING, EPOCH_DONE(e, record_hash), STOPPED, FAILED} plus heartbeat; timeout ≥2× partner's measured epoch time → waiting arm exits resumable and never continues solo. Joint rule is a pure function unit-tested on synthetic series including the GRU F0/F4 trajectories.
4. [evidence] Resume: checkpoint stores u, epoch, permutation cursor, torch CPU/CUDA RNG states, optimizer state; patience/best recomputed from persisted epoch records, not memory; per-epoch coverage bitset == N; assert u==156,956 after epoch 1 with the 128-tail mean-normalised.
5. [evidence] Eval self-term: target score gathered from the same [B,K] logits used for the counts; eval-mode dropout 0; int32 rank vector saved per epoch-end (≈35 MB, ≤2 GB total — add to disk budget); population/target fingerprint asserted equal to the archived baseline fingerprint before any comparison.
6. [evidence] D6 accept: logits exactly K rows (no PAD/MISSING/OOV classes); `F.cross_entropy(reduction='sum', label_smoothing=ε)/n_contributing` across accumulation; hand-formula test on small K; init loss ≈ ln K sanity.
7. [inference] Diagnostic cohort: seeded fixed VS subset, separate file/code path; stopping/selection read only epoch-end records. Store loader refuses any split other than TRAIN/VALIDATION_SELECT.
8. [inference] One checkpoint-rotation function (latest+previous, best weights-only) with pre-write free-space check (refuse <10 GiB, clean abort <8). Measure RSS of two concurrent rich processes before two rich replicas run in parallel; else sequential.
9. [inference, prereg] Registry adds `position_convention`, `tf32=false`, `attention_impl=explicit_fp32_causal`, `barrier_from_epoch=5`, `partner_dead_policy`, D8 ruling; dated hash saved before the first real-data update.

**DECISIONS**
| Item | Ruling |
|---|---|
| D1 positions | KEEP window-start 0..len−1 (enables prefix-consistency test) |
| D2 attention | ACCEPT explicit FP32 causal over unpadded ops; SDPA only if it passes equivalence tests |
| D3 barrier | ACCEPT with fix 3 |
| D4 queue | ACCEPT; critical path is GPU1 (C1→R0, 41–56 h) |
| D5 eval | ACCEPT with fixes 5, 7 |
| D6 smoothing | ACCEPT full-catalogue formula |
| D7 rich fusion | ACCEPT; zero-gate equivalence test mandatory |
| D8 budget | KEEP C0_WIDE384; no replacement |
| D9 precision | ACCEPT with fix 2 |
| Schedule / floor / min-low-LR | UNCHANGED (warmup 2000, peak to 2U, cosine to 0.1 in epoch 3, floor from 4; min 5, max 8) |
| Stopping | UNCHANGED; clarify: first post-floor miss is judged at the epoch-4 eval against the running best incl. pre-floor epochs |
| Selection | UNCHANGED pure function; unit-test tie chain |

D8 rationale [inference]: the item table dominates parameters (158,489×d), median length 5 and 18% length-1 windows leave a third block little leverage, so width is the more plausible binding capacity. Replacing with d256×3 saves ~8–10 h on GPU0, but GPU0 is not the critical path, so wall-clock is unchanged. Depth-3 = future work.

**TESTS_THAT_MUST_FAIL_WHEN_BROKEN**
1. PAD invariance + layout equivalence: randomise PAD-column tokens/features/positions → real outputs and loss unchanged; unpadded path == naive padded FP32 reference (≤1e-5) on real-histogram batches incl. len 1 and 50.
2. Prefix consistency / causality: hidden at position p of a length-L window equals the readout of the same window truncated to p+1; perturbing token p leaves positions <p unchanged; altering target_class never changes forward output.
3. Tying: head weight `is` embedding Parameter, appears once in optimizer groups; PAD row zero after 100 updates; `product_idx_of_class==arange(K)+3` gate; logits [B,K] not [B,T,K].
4. Schedule/stopping: `lr(u)` at u∈{1,2000,2001,2U,2U+1,3U,3U+1,8U} for both peaks; u counts optimizer steps not microbatches; joint and single-arm rules on synthetic series stop at expected epoch, never <5.
5. Exposure: epoch digest == `default_rng(2026+e).permutation(N)`; matched arms equal; coverage exactly N across a mid-epoch resume.
6. Objective: 2×128 accumulation equals one 256 batch gradient before clip (≤1e-6); smoothing formula; init loss ≈ ln K.
7. Rich: gates forced to 0 + copied common weights reproduces ID logits (≤1e-5); independently permuting a side tensor changes output; masked field value change → no output change.
8. Evaluator/precision: rank rule reproduces audited fixture incl. exact ties and OOV credit 0; repeated eval bitwise identical; TF32 canary.

**RESOURCE_RISKS**
1. [evidence] Disk: 21.87 GiB free, +9 GiB peak plus rank vectors; any leaked `previous` checkpoint after a crash can breach the 8 GiB stop — fix 8.
2. [inference] RAM: two rich replicas ≈ 31 GiB stores + overhead vs measured 26 GiB for rich+ID pair; 12 GiB headroom rule may force sequential replicas (+16–36 h).
3. [evidence] Wall-clock 3.6–6.0 days; barrier idle ≈0.3 h/epoch; deterministic kernels may cost 10–20%. Re-benchmark after Gate C; never cut epochs to compensate.
4. [inference] Shared jovyan account: a foreign GPU job or kill stalls the pair; fix 3 plus resume semantics prevent double exposure.

**CLAIM_BOUNDARIES**
Retrospective benchmark on exposed VC/TEST; no confirmatory-holdout claim. Rich-vs-ID is a matched-exposure/LR comparison of one fusion design, not proof of feature value; dropout trajectories differ. Best-found SASRec vs GRU is not architecture attribution (budgets/widths differ). Challengers are single seed-2026 runs; differences smaller than the replica spread of the selected config are uninterpretable. Bootstrap CI is descriptive on a reused period. Report `training_complete`, `evaluation_complete`, `better_than_GRU`, `better_than_TRANS`, `confirmatory_holdout_available=false` separately.

**PRETRAIN_CODE_REVIEW_CHECKLIST**
- `model.py`: `forward` (gather/scatter, causal mask, LN_final placement, readout gather), tied `scores`, position derivation/assert, PAD-zero hook, rich event/user residual gates.
- `train.py`: `lr_at`, param groups (no decay on bias/LN; E once), accumulation normalisation + clip order, u increment, save/load state, TF32/determinism/thread setup.
- `exposure.py`: permutation, digest, cursor, tail.
- `supervisor.py`: status/heartbeat, joint rule, single-arm patience, queue, selection function, disk/RAM guards, split allowlist.
- `eval_adapter.py`: forward allowlist, eval mode, same-tensor target score, cohort isolation.
- `tests/`: the 8 tests above with fixtures.
- `config/run_registry.json`: fix 9 fields and dated hash.

Files read: `F:\Queens University\Microsoft Project\eda\.local_ai\sasrec_v1\review\GATE_P_BRIEF.md`, `F:\Queens University\Microsoft Project\eda\SASREC_PERFORMANCE_EXECUTION_V1\{10_FABLE_PLAN_REVIEW.md, 02_HANDOFF_AND_LESSONS.md, 03_ARCHITECTURE_CONTRACT.md, 04_EXPERIMENT_PROTOCOL.md, config\run_registry.json}`.
