# Fable Gate P2 response (owner 38 h deadline), verbatim

- agent type: sasrec-fable-reviewer (tools Read/Grep/Glob only)
- transcript model ids: ['claude-fable-5-1']
- tool calls: ['Read', 'Read', 'Read', 'Glob', 'Read', 'Read', 'Read', 'Read']

---

ROUTING: claude-fable-5-1

**VERDICT: FEASIBLE within 38 h only as a 3-run, fixed-epoch, no-replica campaign.** [evidence] Registered protocol (7 runs × ≥5 epochs) projects 5–7 days (brief §"Measured speed"); d256 FP32 = 5.24 h per epoch+eval, so at most two d256 5-epoch runs fit in a 32 h window at FP32, three runs only with TF32.

**RECOMMENDED plan (binding; item 4 needs owner approval)**
Schedule S′ (all runs): u≤2000 linear warmup; 2000<u≤U peak; U<u≤2U cosine to 0.1×; u>2U floor (U=156,956). Batch 256 (micro 128×2). No early stopping.

| run | GPU | epochs | floor epochs | precision (train/eval) | eval epochs | hours |
|---|---|---|---|---|---|---|
| A0_ID256 | 0 | 5 | 3,4,5 | TF32 / strict FP32 | 3,4,5 | 5×3.66+3×0.44+0.2 = 19.8 |
| A1_RICH256 | 1 | 5 | 3,4,5 | TF32 / strict FP32 | 3,4,5 | 5×4.03+1.32+0.2 = 21.7 |
| R0_ID128 | 0 (after A0) | 4 | 3,4 | TF32 / strict FP32 | 3,4 | 4×2.29+0.60+0.2 = 10.0 |

Critical path = pre-launch 3.5 + TF32 parity 0.3 + GPU0 29.8 + post 2.0 = **35.6 h; slack 2.4 h**. GPU1 finishes at 21.7 h (idle ≈ 8 h; no TEST eval before lock).
Pre-registered cut rule (systems numbers only, evaluated at launch from the real-data concurrent benchmark and again when A0 finishes): if projected total > 36 h, drop R0; if still > 36 h, anchors to 4 epochs (evals 3,4). Never below 4 epochs, never skip a strict-FP32 eval, never cut context/batch. [inference]
Owner option: if headline quality outranks the GRU-budget reference, replace R0 by a 6th anchor epoch (evals 3–6): A1 = 25.9 h, slack 6.6 h.

**ALTERNATIVE (strict FP32 only, no numerical approval)**
A0: 5×4.80+1.32+0.2 = 25.5 h (GPU0); A1: 5×5.28+1.32+0.2 = 27.9 h (GPU1). Total 3.5+27.9+2.0 = 33.4 h, slack 4.6 h. Nothing else fits (R0 FP32 = 11.6 h → GPU0 37.1 h). Expected quality cost: [inference] none on the anchors (same epochs, same schedule); loses the R0 reference and the parity evidence; gains 2.2 h robustness. If the owner declines TF32 this alternative is nearly equivalent for the headline claim.

**Changes from the registered protocol**
1. [inference] Schedule S′ replaces peak-through-2/cosine-3/floor-4: same warmup, floor LR and ≥2 full floor epochs (GRU lesson kept); halves peak exposure. Untested trade; pre-registered.
2. [evidence→inference] Fixed epochs (5 anchors, 4 R0) replace min5/max8, patience 2, +0.0002 delta and the D3 pair barrier. Matched exposure now holds by construction (equal epochs, shared permutation digests).
3. [inference] Full-VS eval only at floor epochs; best = raw max over evaluated epochs, ties earlier. Diagnostic cohort logging unchanged (monitoring only).
4. [evidence] TF32 matmul for training steps (+31% d256, +20% d128 per brief table); canonical and epoch-end eval strict FP32 (`allow_tf32=False`, `'highest'`). Affects attention and head GEMMs alike. **Needs explicit owner approval** and the parity gate below. Gate-P fix 2 inverts to a per-phase assertion.
5. [evidence] Runs dropped: C0_WIDE384, C1_REG256, seeds 2027/2028. max_scored_runs=3. Owner should confirm this reading of the deadline.
6. [inference] Levers 5 and 6 rejected: batch 512/1024 is an untested LR treatment that invalidates U-based schedule tests; d128 anchors would weaken the headline. d128 kept only as R0.
7. [inference] Registry adds `deadline_hours=38`, `schedule=S_prime`, `fixed_epochs`, `eval_epochs`, `tf32_training`, `eval_precision=strict_fp32`, `cut_rule`, owner-approval reference; dated hash before the first real-data update.

**Revised rules**
- Stopping: none; supervisor asserts epoch count and permutation digests equal for A0/A1 before any comparison; a dead arm that cannot resume within 3 attempts → pair reported incomplete, never compared at unequal epochs.
- Selection: unchanged pure function over {A0,A1,R0} best VS macro MRR; tie chain unchanged.
- Replication: none. `select_luckiest_seed` moot.
- Claims narrowed: fixed-budget training, no convergence/early-stop claim; single-seed A0–A1 difference with user-bootstrap CI only, training-seed variance unmeasured; capacity (d384) and regularisation (C1) untested; R0 is a reference with 4 epochs, not a matched arm; "trained TF32 / evaluated FP32" stated in every table, parity result reported with its threshold, not as FP32 equivalence; best-SASRec-vs-GRU remains non-attributional.

**Minimum tests to add**
1. TF32 parity gate (pre-launch, ≈15 min on benchmark states, discarded): three 3,000-update real-data runs — FP32 seed A, FP32 seed B, TF32 seed A; pass iff |mean loss(TF32_A) − mean loss(FP32_A)| over updates 2001–3000 ≤ |FP32_A − FP32_B| + 0.5% relative, no NaN/Inf, grad-norm medians within 5%. Fail → ALTERNATIVE plan automatically.
2. Precision phase assertion: training steps read `allow_tf32=True/'high'`, every eval reads `False/'highest'`; logged in run record; eval of the same checkpoint from within the trainer and from a fresh FP32 process yields bitwise-identical rank vectors on a fixture.
3. Schedule S′: `lr(u)` at u∈{1,2000,2001,U,U+1,1.5U,2U,2U+1,5U}; fixed-epoch stop at exactly 5/4, never earlier.
4. Cut rule and equal-epoch assertion as pure functions with unit tests (including the "A0 slow → skip R0" path).
5. Gate-P tests 1–8 remain mandatory; the barrier test is replaced by the equal-epoch test.

Files read: `F:\Queens University\Microsoft Project\eda\.local_ai\sasrec_v1\review\GATE_P2_BUDGET_BRIEF.md`, `...\review\GATE_P_BRIEF.md`, `...\review\FABLE_GATE_P_RESPONSE.md`, `F:\Queens University\Microsoft Project\eda\SASREC_PERFORMANCE_EXECUTION_V1\{03_ARCHITECTURE_CONTRACT.md, 04_EXPERIMENT_PROTOCOL.md, 07_GPU_STORAGE_AND_AUTONOMY.md, config\run_registry.json}`.
