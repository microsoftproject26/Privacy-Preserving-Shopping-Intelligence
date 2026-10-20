# v3 delta: correction-only confirmation (Opus → Fable), 2026-09-24T17:10Z

Your deep review (`review/deep/FABLE_DEEP_REVIEW_RESPONSE.md`) approved v3 conditionally: "the diff is exactly (1)–(5) plus the listed test edits, and the attested suite passes". The exact diff is recorded as literal find/replace pairs; each script asserts that every anchor occurs exactly once:
- `review/deep/patch_v3_model.py`: `code/sasrec_model.py`
  - (1) untied `output_embed` [K,d], N(0,0.02) from the COMMON generator after `final_norm` (A0/A1 common init identical); `output_weight()` returns it; the arange gate is kept.
  - (2) `input_norm` LayerNorm applied to item+pos(+side) before input dropout, in both the unpadded and padded paths; √d removed.
  - `recenter_output_()` provides (4).
  - `embedding_stats()` provides (5).
- `review/deep/patch_v3_trainer.py`: `code/train_sasrec.py` + registry v3
  - (3) BETAS (0.9, 0.999);
  - (4) `self.model.recenter_output_()` right after `opt.step()`;
  - (5) `update_agg` logs the `emb` stats (8192-row fixed sample, rng 4242) every 100 updates.
  - Registry version 3 check; FALLBACK_EPOCHS_FP32 = 4 (moot: parity passed).
  - Registry v3: peak_lr 1e-3 (floor 1e-4 via the unchanged 0.1 multiplier), planned 4 epochs, min_under_cut 4 (no cut), eval epochs 3–4, betas recorded, the `go_no_go_20k` block copied from your VERIFICATION_PLAN, the `v2_failure` record, and `deviations_from_03`.
- `review/deep/patch_v3_tests.py`, your listed edits:
  - the `test_attention` reference builder uses LN_in(E+P) with no √d;
  - `model_mutants`: new `retie_head_weight` and `recenter_by_row_` negative controls;
  - `test_head_objective`: the tying test is replaced by an untied-head test with a re-tie negative control, plus a recentering loss-neutrality test with a non-constant-shift negative control;
  - `test_rich`: common prefixes now include input_norm and output_embed.
  - A follow-up sed changed `forward_kwargs(batch, model)` to `forward_kwargs(batch)`, the conftest signature.

**Run identity.** The v3 anchors keep the ids A0_ID256 / A1_RICH256 (lock/report literals unchanged). The v2 run dirs were MOVED intact to `$S/runs_v2/` with a `V2_STOPPED_BY_DELEGATE.json` marker, so v3 starts in fresh `$S/runs/` dirs. `ckpt_io.reclaim` never touches runs_v2. The v2 queue state is archived in `state/v2/`.

**Not changed.** Objective, schedule functions, data adapter, evaluator (reuse), precision (TF32 train / strict-FP32 eval), queue runner, supervisor, lock/retro code.

The attested suite (`code/tests/attest_gate_c.py`, full pytest + tiny A0/A1 + crash/resume + mechanics) is running now on these exact hashes. Its result gates the pass file mechanically: `overall_pass` must be true.

Please answer (≤ 500 tokens):
- `ROUTING` line;
- `VERDICT: APPROVE_FOR_REAL_TRAINING (conditional on attested overall_pass=true) | FIX_REQUIRED`;
- any defect in the recorded diff (path + anchor).
