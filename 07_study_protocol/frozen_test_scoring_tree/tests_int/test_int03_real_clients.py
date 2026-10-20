"""INT-03 — launch_fl's real-data wiring on the INT-01 test environment (real v1 TRAIN rows, bucket 0, CPU):
one client = one TRAIN-manifest user keyed by client_key_hex (v4base.identity over the manifest's own verified tables),
its examples = its own canonical rows served by the real StoreTrainProvider, and two real FL rounds of SASREC_D64_B2
(the registered FA knobs, a small group) through launch_fl.execute (smoke path, no evaluation)."""
from __future__ import annotations

import numpy as np
import pytest

import v4int.launch_fl as F
from intkit import REAL_DS, nc


def test_one_client_is_one_user_with_its_own_rows(real_env, providers):
    tp, _ = providers
    keys, pos = F.client_index(tp, real_env["paths"].manifest_dir)
    tab = real_env["tables"]["M05_CALIB_TRAIN"]
    assert keys == sorted(tab["clients"]["client_key_hex"].tolist())
    allpos = np.sort(np.concatenate([pos[k] for k in keys]))
    assert np.array_equal(allpos, np.arange(tp.n_decisions))
    dec = tab["decisions"]
    hex_of_did = dict(zip(dec["decision_id"].tolist(), tab["clients"]["client_key_hex"][dec["client_ord"]].tolist()))
    did = tp.decision_ids()
    for k in keys[:10]:
        assert {hex_of_did[int(d)] for d in did[pos[k]]} == {k}


def test_two_real_rounds_of_sasrec_d64_b2(tmp_path, real_env, providers):
    from fl.client import ClientData
    from v4models import build, load_schema
    tp, _ = providers
    keys, pos = F.client_index(tp, real_env["paths"].manifest_dir)
    mk = lambda: build("SASREC", load_schema(), 2026, "cpu", v1_root=REAL_DS, variant="SASREC_D64_B2",  # noqa: E731
                       dropout=(0.1,))
    srv, wrk = mk(), mk()
    assert srv.adapter.manifest.shared_numel == wrk.adapter.manifest.shared_numel
    spec = F.arm_spec("FA", seed=2026, peak_lr=1e-3)
    spec["plan"]["group_size"] = 3
    out = F.execute(spec=spec, run_id="real", run_dir=tmp_path / "real", server_adapter=srv.adapter,
                    init_sha256=srv.init_sha256, workers=[wrk.adapter], keys=keys,
                    n_valid_by_key={k: int(v.size) for k, v in pos.items()},
                    get_client=lambda k: ClientData(k, tp.batch(pos[k])), n_decisions=int(tp.n_decisions),
                    manifest_hash=tp.fingerprint, evaluate=None, smoke_rounds=2, say=lambda *_: None)
    assert out["status"] == "SMOKE_DONE_NOT_EVIDENCE" and out["summary"]["rounds"] == 2
    assert out["summary"]["server_digest"] != srv.init_sha256 or out["summary"]["participation"]["survived"] == 0
    assert out["summary"]["bytes"]["per_visit"]["shared_numel"] == srv.adapter.manifest.shared_numel


@nc("keying clients by the per-manifest dense client_ord (not client_key_hex) does not give the manifest's user keys")
def test_nc_dense_key_is_not_the_user_key(real_env, providers):
    from v4base.identity import train_client_ord_by_canonical_position
    tp, _ = providers
    ords = train_client_ord_by_canonical_position(tp, manifest_dir=real_env["paths"].manifest_dir)
    keys = sorted({str(int(o)) for o in ords})
    assert keys == sorted(real_env["tables"]["M05_CALIB_TRAIN"]["clients"]["client_key_hex"].tolist())
