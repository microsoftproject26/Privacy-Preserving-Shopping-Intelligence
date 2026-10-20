"""FL-05 item 2 — the CALIBRATION role on real rows (INT-01 test environment: real v1 TRAIN / VALIDATION rows, bucket 0,
CPU, the real SASREC_D64_B2). The INT-01 kit's pair IS a CALIBRATION pair (M05_CALIB_TRAIN / M06_CALIB_VALIDATION_WARM):

  * launch_fl.role_evaluator scores the harness model on the role's evaluation manifest exactly like the frozen T3 path
    (predict_adapter -> evaluate): same value, same ranks; the rows carry the harness labels with the role's identity
    fields (manifest_role / run_role CALIBRATION, the CALIB manifest ids) on the given key base;
  * two real FA rounds of the CALIBRATION spec (clients = M05 users keyed by client_key_hex) through launch_fl.execute
    (smoke path) evaluated by role_evaluator: the smoke rows are LR-screen rows (role CALIBRATION, fl_lr_mult) and
    SMOKE_NOT_EVIDENCE.
The CentralTrainer harness itself (MAIN pair, CUDA) is not built here (as INT-03 / INT-04: server-side only).
"""
from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np

import v4int.launch_fl as F
from intkit import REAL_DS


def _harness(adapter):
    from fl.numerics import enforce_strict_fp32
    from v4eval import frozen_config as fc
    labels = {"run_label": "SMOKE_NOT_EVIDENCE", "manifest_role": "MAIN", "run_role": "MAIN",
              "train_manifest_id": "M01_COHORT_TRAIN", "eval_manifest_id": "M02_VALIDATION_WARM"}
    return SimpleNamespace(frozen=fc.load(), adapter=adapter, evaluator_sha="TEST_EVALUATOR_SHA", device="cpu",
                           _apply_determinism=enforce_strict_fp32, cfg=SimpleNamespace(seed=2026),
                           state=SimpleNamespace(exposures=0), N=1, _row_labels=lambda: dict(labels))


def _key_base():
    from v4eval.rows import COMPARISON_KEY, T3_KEY_ADDITIONS
    kb = {f: f"TEST_{f}" for f in COMPARISON_KEY if f not in T3_KEY_ADDITIONS}
    kb.update(train_manifest_id="M05_CALIB_TRAIN", eval_manifest_id="M06_CALIB_VALIDATION_WARM")
    return kb


OVER = {"manifest_role": "CALIBRATION", "run_role": "CALIBRATION", "train_manifest_id": "M05_CALIB_TRAIN",
        "eval_manifest_id": "M06_CALIB_VALIDATION_WARM"}


def _mk():
    from v4models import build, load_schema
    return build("SASREC", load_schema(), 2026, "cpu", v1_root=REAL_DS, variant="SASREC_D64_B2", dropout=(0.1,))


def test_role_evaluator_is_the_frozen_t3_path(providers):
    from v4eval import frozen_config as fc
    from v4eval.evaluate import evaluate
    from v4eval.predict import predict_adapter
    _, ep = providers
    srv = _mk()
    ev = F.role_evaluator(_harness(srv.adapter), ep, _key_base(), OVER)
    v, rows, ranks, res = ev(0.5)
    fz = fc.load()
    fc.apply_runtime_flags(fz)
    art = predict_adapter(srv.adapter, ep.batches(fz.eval_batch), ep.manifest, config=fz, model_sha256="ref")
    ref = evaluate(ep.manifest, art, config=fz)
    assert v == float(ref["E2E"]["macro"]["mrr@20"]) and np.array_equal(ranks, art.ranks)
    main = [r for r in rows if (r["aggregation"], r["target_denominator"], r["metric"], r["k"]) ==
            ("macro", "E2E", "mrr", 20)]
    assert len(main) == 1 and main[0]["value"] == v
    assert all(r["manifest_role"] == r["run_role"] == "CALIBRATION" and r["eval_manifest_id"] ==
               "M06_CALIB_VALIDATION_WARM" and r["train_manifest_id"] == "M05_CALIB_TRAIN" and r["method"] == "C"
               and r["run_label"] == "SMOKE_NOT_EVIDENCE" for r in rows)


def test_two_real_calibration_rounds_with_the_role_evaluator(tmp_path, real_env, providers):
    from fl.client import ClientData
    tp, ep = providers
    keys, pos = F.client_index(tp, real_env["paths"].manifest_dir)
    assert keys == sorted(real_env["tables"]["M05_CALIB_TRAIN"]["clients"]["client_key_hex"].tolist())
    srv, wrk = _mk(), _mk()
    spec = F.arm_spec("FA", seed=2026, peak_lr=1e-3, role="CALIBRATION", lr_mult=0.5)
    assert spec["cfg"]["endpoint_rounds"] == 1707 and spec["cfg"]["lr_peak_local"] == 0.5e-3
    spec["plan"]["group_size"] = 3
    ev = F.role_evaluator(_harness(srv.adapter), ep, _key_base(), OVER)
    out = F.execute(spec=spec, run_id="SMOKE_NOT_EVIDENCE__calib", run_dir=tmp_path / "c", server_adapter=srv.adapter,
                    init_sha256=srv.init_sha256, workers=[wrk.adapter], keys=keys,
                    n_valid_by_key={k: int(v.size) for k, v in pos.items()},
                    get_client=lambda k: ClientData(k, tp.batch(pos[k])), n_decisions=int(tp.n_decisions),
                    manifest_hash=tp.fingerprint, evaluate=ev, smoke_rounds=2, say=lambda *_: None)
    assert out["status"] == "SMOKE_DONE_NOT_EVIDENCE" and out["summary"]["rounds"] == 2
    rows = json.loads((tmp_path / "c" / "SMOKE_EVAL.json").read_text(encoding="utf-8"))["rows"]
    assert rows and all(r["role"] == "CALIBRATION" and r["fl_lr_mult"] == 0.5 and r["split"] == "SMOKE_NOT_EVIDENCE"
                        and r["row_status"] == "SMOKE_NOT_EVIDENCE" and r["run_label"] == "SMOKE_NOT_EVIDENCE"
                        and r["eval_manifest_id"] ==
                        "M06_CALIB_VALIDATION_WARM" for r in rows)
