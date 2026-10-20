# AUTO_DECISION: TF32 training (owner delegated to Fable 5.1; owner offline)

**Situation.**
- The owner approved TF32 for TRAINING only, conditional on a real-data parity gate, with you as the delegate. Canonical evaluation stays strict FP32.
- Gate C approved the code (`review/FABLE_GATE_C_RESPONSE.md`), and the gate has now run exactly as pre-registered (`review/TF32_PARITY_PREREG.md`, sha256 3612bb87…, written before any comparison existed).

**Measured evidence.**
- `F:\Queens University\Microsoft Project\eda\.local_ai\sasrec_v1\bench\TF32_PARITY_EVIDENCE.json`: per-state window statistics, pass-criteria checks, cohort metrics, and the descriptive parameter-trajectory difference.
- Raw per-update logs: `bench/parity_{FP32_A,TF32_A,FP32_B}.losses.json`.

**Options.**
- (a) **TF32 ACCEPTABLE**: TF32 training, strict-FP32 evaluation, 6 epochs (4 low-LR).
- (b) **TF32 NOT ACCEPTABLE**: strict FP32 training, 5 epochs (3 low-LR); the registered fallback, applied automatically.

**Time impact.** d256 step ≈ 84 ms (TF32) vs ≈ 110 ms (FP32), synthetic measurement.
- (a) projects to ≈ 26 h for the slower arm.
- (b) projects to ≈ 28 h, but with one fewer low-LR epoch.
- Both fit the 38 h deadline.

**Accuracy/scientific impact.**
- TF32 rounds matmul inputs to a 10-bit mantissa during training only.
- Every validation, cohort and retrospective number is computed in strict FP32.
- The claim wording must say "trained TF32 / evaluated strict FP32", with the gate threshold stated.

**Recommendation needed.** Answer:
- `ROUTING` line;
- `RULING: TF32 ACCEPTABLE | TF32 NOT ACCEPTABLE`;
- which pre-registered criteria you relied on, and any concern about the descriptive parameter difference;
- ≤ 400 tokens.
