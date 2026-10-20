# Task packet — Sonnet B (independent tests), SASREC_FEATURE_OPT_V1

## Purpose
Write INDEPENDENT tests for the lead's study code. You may NOT modify the code under test to make
tests pass. If you find a real defect, report it with a minimal failing test and stop there; the lead
fixes it. Every critical test needs a NEGATIVE CONTROL: a deliberately wrong input or implementation
that the test must catch. Invalid evidence includes a literal `True`, an empty assertion, a
`... or True`, and a grep standing in for a check.

## Owned paths (write only here)
`F:\Queens University\Microsoft Project\eda\.local_ai\feature_opt_v1\code\tests\test_fo_*.py`, plus
`...\code\tests\conftest_fo.py` if you need shared fixtures. Also write the evidence file
`...\feature_opt_v1\review\TEST_EVIDENCE_B.md`.

## Read-only code under test (`...\feature_opt_v1\code\`)
- `fo_env.py` (isolation, path guards)
- `feature_view.py` (probes, lean view + fold, M1 re-encoding, D3)
- `fo_models.py` (wrapper, strict loads, parameter mapping)
- `fo_eval.py` (metrics/bootstrap wrappers)
- `screen.py` (manifest, schedule, train_arm)
- `stream_fit.py` (exact TRAIN fit)
- `pin_baseline.py` (VS-only pin rule, noise reference)
- `probe_runner.py` (encoding check, gates)
- `finalize_fo.py`
- `config\candidate_registry.json` (the frozen registry the code must match)

Context documents:
- `...\feature_opt_v1\review\AUTO_DECISION_SCOPE.md` (Fable's binding decision);
- `...\feature_opt_v1\audit\MEASURED_FINDINGS.md`;
- `...\evidence\snap\dataset\FEATURE_GROUPS.json` and `FEATURE_DICTIONARY.md`;
- package test list: `F:\Queens University\Microsoft Project\eda\SASREC_FEATURE_OPT_V1\05_AGENT_RESOURCES_TESTS.md`.

## Environment
- Local only. No server, no `tools/lobot.py`, no network, no git, no pip.
- Python: `"/f/Queens University/Microsoft Project/eda/next_item_eda/.venv/Scripts/python.exe"`, CPU.
- Imports: put `...\feature_opt_v1\code` on sys.path and import `fo_env` first. It puts the
  byte-identical campaign snapshot `code\snap_sasrec` first; `synthetic_store.SyntheticStore` is there.
- Set `FO_NORMALIZERS` to `...\feature_opt_v1\evidence\snap\dataset\normalizers.json`, the real fitted
  normalizers.
- Set `FO_ROOT_OVERRIDE` to a pytest tmp dir for anything that writes. Never write into the real
  study folders.
- A tiny rich model:
  ```python
  SASRecCE(SASRecConfig.from_registry({"id":"A1_RICH256","group":"rich_frozen_channels","d_model":16,
           "blocks":1,"heads":2,"ff_multiplier":4,"dropout":0.0,"peak_lr":1e-3,"label_smoothing":0.0}),
           vocab_sizes=store.vocab_sizes, product_idx_of_class=store.product_idx_of_class, seed=1)
  ```
  with `store = SyntheticStore.build(channels="F4", n_train=..., n_vs=..., K=97, seed=0, task="mapping")`.
  Perturb its weights and set `side_gate`/`user_gate` to 0.5 so the side paths matter.
- The lead's smoke script `...\feature_opt_v1\scratch\smoke_fv.py` shows how to build batches that
  satisfy the measured duplicate relations. You may copy its idea, not its assertions.

## Required tests
Map each one to the package list numbers.
1. **(2) Identity view.** FOModel(view="full") equals the raw SASRecCE outputs exactly.
   **(2/17) Lean fold parity.** On batches satisfying the measured relations: max|Δlogit| is small
   (report the value) and the ranks are equal. Negative control: break one relation (e.g. +1.0 on
   `log1p_prefix_elapsed_s`) and the difference must exceed a threshold.
2. **(3) Column order.** NUM/FLAG/UCTX/UMASK orders equal FEATURE_GROUPS.json. Swapping two
   columns in the batch must break lean parity (negative control).
3. **(4) Poisoned PAD.** Random garbage at PAD positions (tokens, numerics, flags) never changes
   outputs, for every view and every probe. D3 is 0 at PAD.
4. **(5/7) M1 re-encoding.**
   - Available rows map to (clip(raw)−m')/s'.
   - Unavailable rows map to 0.
   - An available row whose OLD encoding equals z0 is NOT mapped to 0.
   - No second log is applied: the inverse recovers the clipped log-space raw.
5. **(6) Fit-on-VS trap.** A synthetic parquet run of `stream_fit.main` (tmp dataset root with
   `model_ready/user_bucket=0/features.parquet` holding every column it reads, split TRAIN and
   VALIDATION_SELECT) gives an M1 fit that does not change when extreme values are added only to
   VS rows. Valid-only statistics exclude unavailable rows (negative control: include them and see
   the mean move).
6. **(8) Degenerate std.** `m1_states_from_fit` raises on a degenerate std.
7. **(9) Calendar.**
   - `calendar_instant`: sin²+cos²=1;
   - local hour = utc hour + 3h mod 24;
   - weekday index and is_weekend are consistent for all 7×5 cells;
   - utc weekday rolls back when the utc hour would be negative (construct the case, or show it
     cannot occur with the band midpoints and state that).
8. **(10/16) Probe closure.** Each probe changes EXACTLY its declared columns/positions and nothing
   else. P4 touches only the last real position of event data. P2 leaves `category_tokens` and
   `has_history` untouched. Duplicated flags move together.
9. **(11/12/14/15) D3.**
   - Future change: tokens at positions > t never change the features at t.
   - Boundaries: t=0, full window of 50, lengths 1.
   - OOV(2)/MISSING(1) repeats never count.
   - Prior counts exclude the current event.
   - Check exact values on a hand-built example.
10. **(13) No labels.** No target/label/decision-id key ever reaches `inner.forward`. Spy on the
    kwargs.
11. **(17) Parameters.** `parameter_mapping` for d3 and lean is explicit. D3's new columns are
    zeros, and the candidate equals the control exactly at init. A strict load of lean-folded
    weights into a full-view model raises (wrong view refuses load).
12. **(18/19) Screen.**
    - `manifest()` is deterministic, unique, in range.
    - Its sha256 equals `candidate_registry.json["manifest_sha256"]` for n_train=40,180,608. This
      is allowed: it is ~350 MB RAM for a few seconds.
    - `n_updates()` is 19,532 with a final batch of 64.
    - `lr_at` values: u=1 is 5e-7, u=200 is 1e-4, u=U is 1e-5, and the cosine is monotone after
      warmup.
    - `train_arm` on a SyntheticStore (tiny model, CPU, e.g. 1,000 positions, batch 256 → 4
      updates, last of 232) covers every position once, has exposures == len, and runs a fresh
      optimizer.
    - CONTROL and C1_D3 given the same seed and data produce the same loss at update 1.
13. **(20)** `fo_eval.metrics` counts rank −1 rows in the denominator (OOV zero credit): compare
    with a hand computation.
14. **(22/24)**
    - `refuse_split` raises for TEST/VALIDATION_CONFIRM/LABEL_GRACE.
    - `study_path` refuses escaping paths.
    - `write_once` refuses to overwrite and raises on NaN.
    - `assert_isolated` raises when a module is loaded from outside the snapshot (negative control:
      a fake module object in sys.modules with a foreign `__file__`).
15. **pin_baseline**
    - `choose()` tie rule: highest score, then earlier epoch, then seed 2026.
    - `noise_reference`: max abs diff over matched epochs; the fallback of 0.002 when s2027 is absent.
    - Test on fabricated summary dicts in a tmp campaign root via FO_CAMPAIGN_ROOT; `candidates()`
      reads them.

## Acceptance
- Report: test file names, count, pass/fail, the exit code of `pytest code/tests/test_fo_*.py -q`,
  and every negative control with its observed number.
- Also report ANY defect you found in the code under test, with the file:line and a failing test.
- Keep the final answer to ≤ 400 words. Details go in TEST_EVIDENCE_B.md.
