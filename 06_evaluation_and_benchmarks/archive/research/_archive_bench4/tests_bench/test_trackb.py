"""Track B plumbing: max_len and dropout reach the data windows and the model of the central and FL arms."""
from __future__ import annotations

import numpy as np

from v4bench import central, data as D, fl_engine
from v4bench.common import read_json


def test_track_b_max_len_and_dropout(release, tmp_path, monkeypatch):
    monkeypatch.setattr(central, "EFFECTIVE_BATCH", 8)
    d = D.load(release["name"], release["root"], max_len=6)
    assert d.batch(np.arange(d.N))["item_tokens"].shape[1] <= 6
    out = central.run_central(data=d, arm="C_FULL", variant="SASREC_D64_B2", peak_lr=4e-4, dropout=(0.5,), seed=2026,
                              rows=np.arange(d.N), select_users=None, run_dir=tmp_path / "c", run_id="tb_c", max_len=6)
    spec = read_json(tmp_path / "c" / "ARM_SPEC.json")
    assert out["status"] == "DONE_MAIN" and spec["max_len"] == 6 and spec["dropout"] == [0.5]
    out = fl_engine.run_fl(data=d, arm="FA", seed=2026, run_dir=tmp_path / "f", run_id="tb_f", dropout=(0.2,), max_len=6)
    s = read_json(tmp_path / "f" / "RESULT.json")["spec"]
    assert out["status"] == "DONE_MAIN" and s["max_len"] == 6 and s["dropout"] == [0.2]
