"""The secure-aggregation grid command (ppsi.fedsim.secagg_grid): deterministic cells, the JSON report, no overwrite."""
from __future__ import annotations

import json

import pytest

from ppsi.fedsim import secagg_grid as grid

N = 64


@pytest.mark.parametrize("k", [None, 32])
def test_cells_are_deterministic(k):
    c1, c2 = grid.run_cell(N, 256, k, 0.1, 8, 1.0), grid.run_cell(N, 256, k, 0.1, 8, 1.0)
    for key in ("success", "bit_identical", "bytes_up_by_round", "bytes_down_by_round", "n_survivors",
                "decoded_max_abs"):
        assert c1[key] == c2[key]


def test_cell_records_unmasking_time_and_work():
    c = grid.run_cell(N, 256, None, 0.3, 11, 1.0)
    assert c["success"] and c["bit_identical"] and c["unmask_cpu_s"] > 0 and c["rounds"] == 4
    assert c["prg_words_server"] >= 256 * c["n_survivors"]
    assert c["decoded_max_abs"] == 0.0


def test_command_writes_the_report_once(tmp_path):
    out = tmp_path / "secagg.json"
    rc = grid.main(["--out", str(out), "--update-dim", "128", "--seeds", "1", "--k-values", "32"])
    assert rc == 0 and out.exists()
    doc = json.loads(out.read_text(encoding="utf-8"))
    assert doc["n_cells"] == 2 * 3 and doc["n_bit_identity_violations"] == 0
    assert doc["grid"]["threshold_t"] == 43 and doc["protocol"].startswith("BONAWITZ2017")
    with pytest.raises(SystemExit):
        grid.main(["--out", str(out), "--update-dim", "128"])
