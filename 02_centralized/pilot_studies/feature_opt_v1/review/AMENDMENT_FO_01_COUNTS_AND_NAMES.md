# Amendment FO-01 — duplicate count, D3 width and the `sigma_hat` name (documentation only)

- **Timestamp:** 2026-09-25T12:25Z.
- **Decider:** the lead (Opus). This is a documentation correction with no scientific or
  operational change. No code, registry value, gate or threshold is changed.
- **Found by:** the thesis-reference reconciliation. The package's chapter 16.3 flagged it.

## 1. "Nine duplicated channels" → 8 redundant columns in 6 exact relations
The original text is RETAINED unchanged in:
- `audit/MEASURED_FINDINGS.md:48` ("Nine duplicated channels");
- `review/SCOPE_DECISION_BRIEF.md:40`;
- `review/AUTO_DECISION_SCOPE.md:29,35` (Fable);
- `config/candidate_registry.json:47` (hygiene text);
- `FEATURE_EXPERIMENT_PROTOCOL.md:46`.

The authoritative source is the code, `code/feature_view.py`. It drops 8 redundant columns:
- `DROP_NUM_AFFINE`, 2 columns: `log1p_prefix_elapsed_s`, `prefix_item_count`;
- `DROP_NUM_ROTATION`, 2 columns: `local_hour_sin`, `local_hour_cos`;
- `DROP_FLAG_DUP`, 2 columns: `price_zero`, `main_category_missing`;
- `DROP_MASK_DUP`, 2 columns: `has_history_mask`, `prev_user_event_available`.

The code also drops 6 dead flags (`DROP_FLAG_DEAD`). The total is 14 removed input dimensions:
34 → 22 in the event branch and 18 → 16 in the user branch.

These 8 columns come from 6 exact relations:
1. prefix_elapsed ≡ segment_elapsed;
2. item_count ≡ rep + 1;
3. local-hour pair ≡ R(+3h)·utc-hour pair;
4. price_zero ≡ price_missing;
5. main_category_missing ≡ category_code_missing;
6. has_history ≡ has_history_mask ≡ prev_user_event_available.

The inventory (`audit/REDUNDANCY_FINDINGS.md`, findings 1–4) lists relations 1–4 in code. Finding 5
(local weekday) is NOT an exact duplicate. Relations 5 and 6 were added by the measured check
(`audit/MASK_DUP_TIME_CHECK.json`). "Nine" was a miscount in the M2 header: the M2 list also contains
two items given for contrast, the local weekday and `recent_price_supported`, which are not
duplicates. **Corrected statement: 8 redundant columns (6 exact relations) + 6 dead flags.**

## 2. D3 width: "≤4 channels" versus 5 scalars
- Fable's decision (item 5) lists 4 conceptual channels, the second of which includes "+
  availability flag".
- The frozen registry and `feature_view.D3_NAMES` implement 5 scalars: the 4 channels plus the
  separate `d3_has_prev` availability flag.
- This is within the package cap of "up to six scalars plus required masks". **Statement: 4
  conceptual channels = 5 scalars.**

## 3. The name `sigma_hat`
- In `config/NOISE_REFERENCE.json` and `finalize_fo.py`, `sigma_hat` is the MAXIMUM ABSOLUTE
  difference between two seeds over matched epochs, or the fallback 0.002. It is NOT a standard
  deviation or a confidence interval.
- The code name is kept, and every report must describe it as a "seed-gap reference", a
  conservative operational hurdle.

## Effect
None on execution. The code drops exactly the columns listed above, and parity is checked in any
case. Claims in reports must use the corrected wording.
