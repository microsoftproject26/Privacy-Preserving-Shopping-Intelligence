# Task packet — Sonnet A (inventory / lineage / profiler), SASREC_FEATURE_OPT_V1

Purpose: build the ACTUAL feature inventory of the corrected SASRec v3 rich arm (A1, group
`rich_frozen_channels` = F4) from source code and frozen-dataset documents. No assumptions from chat
summaries; every claim carries a source `file:line` or schema key. Unknowns are marked, never filled
with plausible prose.

## Hard rules
- LOCAL, READ-ONLY work. Do NOT contact the server (no `tools/lobot.py`, no ssh/curl/network).
- Write ONLY under `F:\Queens University\Microsoft Project\eda\.local_ai\feature_opt_v1\audit\` and the
  one script `...\feature_opt_v1\code\profile_train_sample.py` (+ its test `...\code\tests\test_profile_train_sample.py`).
- Never edit anything under `.local_ai\sasrec_v1\`, `.local_ai\gru_v1\`, `next_item_eda\`, or the package.
- No git commands. No pip install. Local python for tests only:
  `/f/Queens University/Microsoft Project/eda/next_item_eda/.venv/Scripts/python.exe`.

## Read-only sources (authoritative, in this order)
1. Frozen dataset documents (copied from the server, byte-identical):
   `.local_ai\feature_opt_v1\evidence\snap\dataset\` — FEATURE_DICTIONARY.md, FEATURE_GROUPS.json,
   normalizers.json, DATA_CONTRACT.md, VERSION.json, protocol.resolved.yaml, TRAINING_READINESS.json,
   PROTOCOL_DEVIATIONS.md, FILES.json, splits.json, model_ready_listing.txt.
2. Data-pipeline source that built the dataset: `next_item_eda\src\rees46_eda\*.py`
   (norm.py, userctx.py, pipeline*.py, loaders.py, torch_adapters.py, ...). First establish whether
   this local source is the code identity recorded for the frozen build (look for code hashes /
   commit ids in VERSION.json, COLD_REBUILD_REPRODUCIBILITY-like records, FILES.json). If no identity
   can be established, say NOT_VERIFIED and still cite lines, flagged.
3. Reused loader/store code the trainer actually imports (byte-identical copy):
   `.local_ai\feature_opt_v1\evidence\snap\reuse\` — full_store.py (CompactStore: which parquet
   columns are loaded, dtype, where each tensor/index comes from), train.py (build_named_kwargs,
   VOCAB_SIZES), vendor\rees46_eda\loaders.py + config.py.
4. SASRec model + batch adapter: `.local_ai\sasrec_v1\code\sasrec_model.py`, `sasrec_data.py`,
   `train_sasrec.py` (only where the batch is built / fed to the model).

## Deliverables (in `feature_opt_v1\audit\`)
1. `FEATURE_INVENTORY.csv` — one row per channel/array position INCLUDING unused/stored columns,
   identifiers, structural masks (attention_mask, lengths, loss/eval masks), category_code token,
   audit-only columns. Columns:
   `feature, batch_key, tensor_index, dtype, shape, grain(event|decision|structural), semantic_units,
   formula_short, as_of_rule, fit_population(grain + split + which rows, e.g. "valid TRAIN
   representative events" vs "TRAIN decisions"), transform(chain + fitted params), missing_sentinel,
   masks(names), used_by_A1(yes/no/stored-only), model_entry(module + sasrec_model.py line),
   direct_group(CATEGORY|BRAND|CALENDAR|ELAPSED_TIME|PRICE|SESSION_INTENT|USER_HISTORY|QUALITY_PROVENANCE|PRODUCT|STRUCTURAL|IDENTIFIER|LABEL),
   depends_on, duplicates_or_derivations, source_refs(file:line;file:line), confidence(high|medium|low), notes`.
   Use the actual names; the FEATURE_GROUPS.json orders give the tensor indices.
2. `TRANSFORM_LINEAGE.json` — per feature: canonical source -> recovery/derivation -> raw/semantic
   value -> transform -> model token/float -> projection -> fusion (sasrec_model.py lines), with refs.
3. `FEATURE_DEPENDENCIES.json` — dependency graph: each feature's single direct owner group plus
   dependency list; for each semantic group two closures: (a) "direct channels only" and (b) "total"
   (all descendants that disclose the same information, e.g. category-derived counts/shares, their
   flags and masks). List masks shared across groups (must survive removal of other columns).
4. `REDUNDANCY_FINDINGS.md` — exact/algebraic duplicates or deterministic derivations found IN CODE
   (e.g. two columns identical after the recorded transform, a sin/cos pair that is an exact rotation
   of another pair, a flag identical to a mask), each with proof from formulas + fitted params, and
   whether equality can break (clipping bounds, special cases). Mark which need a data check.
5. `OPEN_QUESTIONS.md` — only questions the code cannot answer and a bounded data sample can; for
   each, the exact measurement needed. Must cover at least: exact encoded vector of a cold user
   (has_history=0) in user_context/masks; encoded values of category-derived counts/shares when the
   category is unknown (raw 0 -> z(0) or 0-after-normalization?); the grain/population each scaler
   was fitted on (event rows? representatives? decisions?); whether raw product identity is available
   read-only for OOV-safe repeat features; how often OOV tokens occur in context.
6. `code\profile_train_sample.py` — a read-only, deterministic, bounded profiler to be run LATER on the
   server by the lead (you do not run it on real data). Constraints: 1 thread (set OMP/MKL/POLARS
   threads = 1 before imports), <= 400 MiB RSS target, column projection, reads a declared
   deterministic subset of `model_ready/user_bucket=*` partitions (arg `--buckets 0,17,42` default 3
   buckets) for split TRAIN (and VALIDATION_SELECT for drift, same buckets), never VC/TEST. Per
   feature on VALID values only (use the masks/flags; exclude PAD): count, missing-by-mask, exact
   zeros, min, q001, q01, q25, q50, q75, q99, q999, max, mean, std, n_distinct (capped), constant
   flag; exact equality checks for the redundancy hypotheses; the cold-user encoded vector table;
   unknown-category encoded values; OOV rate in item tokens; strict JSON output (nonfinite encoded
   explicitly, no NaN literals). Label all statistics SAMPLE (not exact population). Output path arg.
   Include a small synthetic unit test (`code\tests\test_profile_train_sample.py`) that builds a tiny
   fake partition with the same column names and checks the script runs and the numbers are right
   (with one negative control).
   Read full_store.py / loaders.py to learn the real on-disk column names and layout; if they cannot
   be determined locally, write the script to FAIL LOUDLY on unknown columns, never guess.

## Acceptance
- Every inventory row has source_refs or an explicit `NOT_FOUND` + confidence low.
- Counts reported: numeric channels, quality flags, user-context values, user-context masks,
  categorical side channels (actual numbers from code/schema; compare to 19/15/12/6/5 and report any mismatch).
- `pytest code/tests/test_profile_train_sample.py` passes locally (report exit code).
- Return (<= 400 words): files written, counts, top 5 findings (with refs), open risks. Raw detail stays in files.
