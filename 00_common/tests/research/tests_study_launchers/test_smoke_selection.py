"""Smoke rows are refused by the (unmodified) v4train.selection code, as C_MATCHED and as downstream C rows."""
from __future__ import annotations

from intkit import assert_raises, evidence_rows, nc
from v4int.smoke import SMOKE_LABEL, is_smoke, smoke_row, smoke_rows
from v4train.selection import (SelectionError, assert_c_matched_eligible, cohort_evidence, downstream_c_rows,
                               select_family)

CI = (0.005, 0.015)


def test_valid_evidence_is_accepted():
    d = select_family(evidence_rows(), delta_bar_ci=CI)
    assert d["status"] == "DECISIVE_STABLE" and d["winner_family"] == "GRU"


def test_one_smoke_row_refuses_the_whole_selection():
    rows = evidence_rows() + [smoke_row(evidence_rows()[0])]
    assert_raises(SelectionError, select_family, rows, delta_bar_ci=CI)
    assert_raises(SelectionError, cohort_evidence, rows)


def test_smoke_rows_alone_refused():
    assert_raises(SelectionError, select_family, smoke_rows(evidence_rows()), delta_bar_ci=CI)


def test_smoke_row_with_restored_split_and_rule_still_refused():
    rows = evidence_rows()
    r = smoke_row(rows[0])
    r.update(split="VALIDATION", selection_rule="PRACTICAL_BEST")            # tampered back toward evidence
    assert_raises(SelectionError, select_family, [r] + rows[1:], delta_bar_ci=CI)


def test_smoke_rows_never_c_matched_or_downstream_c():
    rows = evidence_rows()
    for r in smoke_rows(rows):
        assert is_smoke(r) and r["split"] == SMOKE_LABEL and r["smoke_manifest_split"] == "VALIDATION"
        assert_raises(SelectionError, assert_c_matched_eligible, r)
    d = select_family(rows, delta_bar_ci=CI)
    ctrl = [dict(r, selection_rule="CONTROLLED_BUDGET") for r in rows]
    got = downstream_c_rows(d, rows + ctrl + smoke_rows(rows + ctrl))
    assert got and not any(is_smoke(r) for r in got)


@nc("a smoke-run row that is NOT relabelled (only its run id says smoke) would be accepted as evidence")
def test_nc_unlabelled_smoke_row_is_accepted():
    rows = evidence_rows()
    rows[0] = dict(rows[0], run_id="SMOKE_NOT_EVIDENCE__GRU_CALIBRATION_s2026")
    assert_raises(SelectionError, select_family, rows, delta_bar_ci=CI)
