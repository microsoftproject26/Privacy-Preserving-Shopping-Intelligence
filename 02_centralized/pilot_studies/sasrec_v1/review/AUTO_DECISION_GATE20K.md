# AUTO_DECISION: 20k gate criterion 4 (verbatim ruling)

- model: ['claude-fable-5-1']
- brief: review/AUTO_DECISION_GATE20K_BRIEF.md
- amendment: config/run_registry.v3_amendment_1.json

ROUTING: claude-fable-5-1

DECISION: b (minimal one-for-one substitution, with disclosure conditions)

Basis: init 0.02·√256 ≈ 0.32, so the 3× cap is 0.96; measured medians are already 0.633/0.822 (E/W_out) at 3.7k [evidence: brief table; registry `median_row_norms_E_and_W_le_x_init: 3.0`, line 277]. The criterion is deterministically failed by any run, so this is a calibration error I made (never computed the absolute value), not data-dependent tuning [inference]. The v2 pathology was common-mode rows ≈ 162; scale-free/absolute caps detect it [evidence: brief].

Final 20k criteria (both arms, 100-update means):
1. mean_loss(20k) ≤ mean_loss(10k) − 0.02 — unchanged
2. mean_grad_norm(20k) ≤ 1.25× value at 6k — unchanged
3. ‖mean row E[3:]‖ ≤ 0.5 — unchanged
4. median row norm of E and of W_out ≤ 5.0 absolute; W_out ‖mean row‖ ≤ 1e-3 (recentring sanity) — replaces item at registry line 277
5. all finite — unchanged
Descriptive only (log, non-gating): ‖mean row E‖ / median row E.

Conditions: (i) amend `run_registry.locked.json` before 17:58Z with timestamp, original threshold, and this rationale; (ii) at 20k report the original 3×-init outcome alongside the amended one; (iii) no further gate changes after the 20k value is seen. Caveat: brief figures are unverified by me (no logs named) [evidence: brief only].
