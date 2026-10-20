"""v4int.smoke — SMOKE_NOT_EVIDENCE labelling: a smoke run can never become a selection, C or C_MATCHED row.

A `--smoke` launch (INT-01) runs the real trainer with test_overrides=True, a run id prefixed SMOKE_NOT_EVIDENCE__,
an update cap far below the first 0.5-EFE mark (so the trainer itself never writes a mark row or RESULT_ROWS), and
one frozen evaluation called by the launcher. Its rows are relabelled here before they are written:
  split              "SMOKE_NOT_EVIDENCE"  -> v4train.selection._refuse_common refuses the row ("VALIDATION only") on
                                              EVERY selection entry point (select_family, cohort_evidence, the
                                              tie-break stages), and v4eval.rows.check_block / comparison_status never
                                              treat it as same-conditions with a real row (split is a key field);
  status / row_status / comparison_status / evidence_label  "SMOKE_NOT_EVIDENCE";
  value_status       "SMOKE_NOT_EVIDENCE:<original>"  -> refused as evidence even if split were restored;
  selection_rule     None;  c_matched_eligible False (assert_c_matched_eligible raises);  downstream_seed_allowed False.
The original split is kept in `smoke_manifest_split`. v4train.selection is not modified; code/tests_int proves the
refusals, with a negative control showing that an unlabelled smoke row would have been accepted.
"""
from __future__ import annotations

from typing import Iterable, List, Mapping

SMOKE_LABEL = "SMOKE_NOT_EVIDENCE"
SMOKE_RUN_PREFIX = SMOKE_LABEL + "__"
MAX_SMOKE_UPDATES = 600


def smoke_row(row: Mapping) -> dict:
    r = dict(row)
    r["smoke_manifest_split"] = r.get("split")
    r["smoke_original_value_status"] = r.get("value_status")
    r.update(split=SMOKE_LABEL, status=SMOKE_LABEL, row_status=SMOKE_LABEL, comparison_status=SMOKE_LABEL,
             evidence_label=SMOKE_LABEL, value_status=f"{SMOKE_LABEL}:{r.get('value_status')}", selection_rule=None,
             c_matched_eligible=False, downstream_seed_allowed=False)
    return r


def smoke_rows(rows: Iterable[Mapping]) -> List[dict]:
    return [smoke_row(r) for r in rows]


def is_smoke(row: Mapping) -> bool:
    return any(row.get(k) == SMOKE_LABEL for k in ("split", "status", "row_status", "evidence_label"))


def smoke_cap_before_first_mark(updates: int, effective_batch: int, N: int) -> bool:
    """True iff `updates` full effective batches end strictly before the first 0.5-EFE mark (ceil(N / 2) exposures):
    the trainer then never evaluates a mark, so it writes no mark row, no RESULT_ROWS and no RUN_SUMMARY."""
    return 0 < int(updates) and int(updates) * int(effective_batch) < -(-int(N) // 2)


__all__ = ["SMOKE_LABEL", "SMOKE_RUN_PREFIX", "MAX_SMOKE_UPDATES", "smoke_row", "smoke_rows", "is_smoke",
           "smoke_cap_before_first_mark"]
