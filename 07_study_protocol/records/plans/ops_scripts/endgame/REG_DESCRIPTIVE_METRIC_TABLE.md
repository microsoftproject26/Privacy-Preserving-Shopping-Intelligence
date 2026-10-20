# Registration text: the full DESCRIPTIVE metric table (FIX-5, Fable gate 2026-10-03)

**For Ops:** paste the block below as ONE Decision Log row before D runs. Then run D with `MT_REG_DL=<that row number>`.
D refuses without it and writes the row number into the freeze. No MAIN / VALIDATION / TEST value has been read for this
text.

**Script:** `company/plans/ops_scripts/endgame/metric_table.py`, deployed at `runs_ds3/queues/endgame/metric_table.py`. Its
sha256 is recorded at registration (§ below) and frozen by D in ANA_SCRIPT_FREEZE `scripts`, which `freeze_check`
re-hashes in E, F, G and H.

---

> **REGISTERED (before the ANA freeze; no MAIN / VALIDATION / TEST value read): the full descriptive metric table
> (owner timeline DL 382; Fable gate FIX-5 option (a)).**
>
> **(1) Content.** For every block entry of the §9 VALIDATION block and of the §11.6 TEST block — every arm, seed, rule
> (PRACTICAL_BEST / CONTROLLED_BUDGET / RULE_FREE) and role (MAIN and the panel roles present), central, FL, device and
> baseline — the table reports the registered T3 evaluator's grid as already computed in the sealed evidence rows:
> - metric families HR (= Recall@k for one target per decision), NDCG and MRR;
> - k = 1, 5, 10, 20;
> - aggregation micro and macro;
> - target denominator E2E (the registered one) and RANKABLE.
>
> That is 48 cells per entry. Each cell carries `value`, `value_status`, `value_pessimistic` (the tie diagnostic),
> `n_decisions` and `n_users`. ONNX entries (K16) and TRUNC curves are reported elsewhere (BENCH_RESULT / retention_report)
> and are listed as not in the table.
>
> **(2) Status.** Every cell is DESCRIPTIVE, reported beside the confirmatory results. The primary reading is unchanged:
> macro MRR@20 of RETENTION_RESULT.json (06P P1–P7, 06D §3). NDCG@10 / HR@10 and their CIs stay as retention_report
> computes them. No claim, contrast, Holm band, selection, stopping rule or recipe change may use a cell of this table.
>
> **(3) No new computation.** The extractor `metric_table.py` (sha256 `<METRIC_TABLE_SHA256>`):
> - binds the block by its BLOCK_INDEX.json sha256 (VALIDATION: E_CHECK_PASS `block_index_sha256`; TEST: the index the
>   TEST report read);
> - verifies each copied rows file against the index `sources` sha256;
> - applies the entry's row_filter (device cell) and declared rule exactly as retention_report.select_rows does;
> - **copies** the five fields verbatim.
>
> It computes no mean, ratio, CI, bootstrap or rounding. A duplicate or missing grid cell refuses (exit 2, nothing written).
> Seed means, CIs or any other statistic on these metrics are NOT part of this registration. If ever wanted, they would be
> POST_VALIDATION_EXPLORATORY (06L-C C5) for VALIDATION, and are not allowed after the TEST look.
>
> **(4) When.** VALIDATION: inside F_validation_read.sh, right after the single §9 read, on the same block
> (`$R/ANA/metric_table_val_<tag>/`). TEST: inside H `analysis`, right after the TEST report (`$R/ANA/metric_table_test_<tag>/`).
> It runs once each, outputs are write-once and read-only, and stdout is counts only. A refusal stops only the table, never
> the read. It is re-run, with the same frozen script, into a new `--out` only after a DL row records the cause.
>
> **(5) Freeze / lock.** ANA_SCRIPT_FREEZE carries `descriptive_metric_table` {script, script_sha256, this DL row} and the
> script in `scripts`. ALL_METHODS_LOCK copies `descriptive_metric_table`. After the freeze the script never changes:
> a change means a new freeze.

---

**Placeholders Ops fills before registering:** `<METRIC_TABLE_SHA256>` = the sha256 of the deployed `metric_table.py`
(local = server, CRLF stripped) = **561d758ae2431d2a659e42e310d439f446bbf179dcd6d83f8400c1f3d2e77b57** (OPS-ENDGAME-3, 2026-10-03).
