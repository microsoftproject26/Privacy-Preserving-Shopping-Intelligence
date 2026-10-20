"""INT-06 item 3 — v4int.reeval (CPU; the GPU / pin / data-root path is server-only, like launch_fl.run).

  * preflight refusals BEFORE any checkpoint deserialisation: TEST role (even for a path that does not exist), sha256,
    variant / recipe pair, source RUN_START / LAUNCH identity, smoke sources, unregistered roles, PF store pairing;
  * on the INT-01 real-row CPU environment (real v1 VALIDATION rows, the real SASREC_D64_B2): a saved central
    checkpoint, an FL theta and a PF theta + store, each loaded into a FRESH model by reeval, reproduce the harness
    evaluation (launch_fl.role_evaluator / launch_device.pf_evaluator on the original weights) bitwise — value, every
    row, every rank; an unloaded theta0 / an empty store does not (negative controls);
  * the role key replacement (launch_device.run rule), the additions-only registry extension check + verify wrapper,
    the relabel / pinned-row guards and the DATA-09 TEST hook.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import v4int.launch_device as D
import v4int.launch_fl as F
from intkit import REAL_DS, nc
from v4int import reeval as R

V4 = Path(__file__).resolve().parents[2]
VAR, REC = "SASREC_D64_B2", "SMALL_SASREC_D64_B2"


def _sha(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


# ------------------------------------------------------------------------------------------------ preflight (no data)
def _central(tmp_path, cfg=None, kb=None, run_id="C_SMALL_s2026"):
    run = tmp_path / "run"
    cp = run / "sealed" / "checkpoints" / "best.pt"
    cp.parent.mkdir(parents=True)
    cp.write_bytes(b"synthetic-checkpoint-for-preflight")
    start = {"run_id": run_id, "config": dict({"family": "SASREC", "model_variant": VAR, "recipe_id": REC,
                                               "seed": 2026}, **(cfg or {})),
             "key_base": dict({"split": "VALIDATION", "run_label": "PRODUCTION"}, **(kb or {}))}
    (run / "RUN_START.json").write_text(json.dumps(start), encoding="utf-8")
    return cp


def _argv(cp, tmp_path, *, kind="central", sha=None, extra=(), **kw):
    a = {"--ckpt": str(cp), "--ckpt-kind": kind, "--expect-sha256": sha or _sha(cp), "--variant": VAR,
         "--recipe-id": REC, "--seed": "2026", "--role": "MAIN", "--arm-label": "C_SMALL",
         "--selection-rule": "PRACTICAL_BEST", "--checkpoint-id": "best", "--gpu": "0",
         "--out-dir": str(tmp_path / "out"), "--history-policy": "USER_HISTORY_L50"}
    a.update({f"--{k.replace('_', '-')}": str(v) for k, v in kw.items()})
    return R.parse_args([x for kv in a.items() for x in kv] + list(extra) + ["--dry-run"])


def test_dry_run_verifies_checkpoint_identity(tmp_path):
    got = R.run(_argv(_central(tmp_path), tmp_path))
    assert got["dry_run"] is True and got["source_run_id"] == "C_SMALL_s2026" and not (tmp_path / "out").exists()


def test_wrong_checkpoint_hash_refused(tmp_path):
    with pytest.raises(R.ReevalRefused, match="sha256"):
        R.run(_argv(_central(tmp_path), tmp_path, sha="0" * 64))


def test_test_role_refused_before_any_read(tmp_path):
    missing = tmp_path / "does_not_exist" / "best.pt"
    for role in ("MAIN_RETRO_TEST", "UNEW_HOLDOUT_RETRO_TEST", "test"):
        with pytest.raises(R.ReevalRefused, match="TEST role"):
            R.run(_argv(missing, tmp_path, sha="0" * 64, role=role))
    assert R.test_gate_ok("MAIN") and not R.test_gate_ok("TRUE_COLD_RETRO_TEST")


def test_argument_contract_refusals(tmp_path):
    cp = _central(tmp_path)
    cases = [(dict(variant="SASREC_D256_B3"), "variant / recipe"), (dict(role="COLD_H"), "re-evaluation role"),
             (dict(role="CALIBRATION"), "re-evaluation role"), (dict(arm_label="c big"), "arm label"),
             (dict(kind="pf"), "needs --pf-store")]
    for kw, msg in cases:
        with pytest.raises(R.ReevalRefused, match=msg):
            R.run(_argv(cp, tmp_path, **kw))
    with pytest.raises(R.ReevalRefused, match="--ckpt-kind pf only"):
        R.run(_argv(cp, tmp_path, pf_store=cp))
    with pytest.raises(R.ReevalRefused, match="SMALL recipe"):
        R.run(_argv(cp, tmp_path, kind="fl", variant="SASREC_D256_B3", recipe_id="02B_SASREC"))


def test_source_run_identity_refusals(tmp_path):
    for i, (cfg, kb, rid, msg) in enumerate([({"seed": 2027}, None, "C", "seed"),
                                             ({"model_variant": "SASREC_D256_B3"}, None, "C", "model_variant"),
                                             (None, {"run_label": "SMOKE_NOT_EVIDENCE"}, "C", "smoke"),
                                             (None, None, "SMOKE_NOT_EVIDENCE__C", "smoke"),
                                             (None, {"split": "TEST"}, "C", "VALIDATION")]):
        d = tmp_path / f"c{i}"
        cp = _central(d, cfg=cfg, kb=kb, run_id=rid)
        with pytest.raises(R.ReevalRefused, match=msg):
            R.run(_argv(cp, d))


# ------------------------------------------------------------------------------------------------ FL / PF sources
def _theta():
    from fl.synthetic import make_tiny_adapter
    return {k: v.clone() for k, v in make_tiny_adapter(seed=1).broadcast_state().items()}


def _fl_run(tmp_path, *, arm="FA", legs=1, label="PRODUCTION", role="MAIN", ckpt="best.pt", digest="C", **launch_over):
    from fl.checkpoint import atomic_save
    run, rid = tmp_path / "fl", f"FL_{arm}_SMALL_s2026"
    run.mkdir(parents=True, exist_ok=True)
    launch = {"arm": arm, "run_id": rid, "arm_spec_sha256": "S", "fl_config_digest": "C", "label": label,
              "fl05": {"role": role, "lr_mult": 1.0}, "arm_spec": {"cfg": {"seed": 2026}},
              "small_recipe": {"model_variant": VAR, "recipe_id": REC, "recipe_sha256": "R"}}
    launch.update(launch_over)
    for k in range(legs):
        (run / f"LAUNCH_{k}.json").write_text(json.dumps(launch), encoding="utf-8")
    theta = _theta()
    cp = run / "ckpt" / rid / ckpt
    atomic_save({"kind": "weights", "run_id": rid, "config_digest": digest, "round": 5, "efe": 3.0, "phase": "MAIN",
                 "theta": theta}, cp)
    return cp, rid, theta


def _store_file(tmp_path, rid, theta_digest=None, name="PF_STORE_best.pt"):
    from fl.checkpoint import atomic_save
    from fl.personal import PersonalStore
    obj = {"kind": "pf_store", "run_id": rid, "config_digest": "C", "store": PersonalStore(8).state_dict()}
    if theta_digest is not None:
        obj["theta_digest"] = theta_digest
    p = tmp_path / "fl" / name
    atomic_save(obj, p)
    return p


def test_fl_dry_run_and_launch_identity(tmp_path):
    cp, rid, _ = _fl_run(tmp_path)
    got = R.run(_argv(cp, tmp_path, kind="fl", arm_label="FA"))
    assert got["dry_run"] and got["source_run_id"] == rid and got["fl_round"] == 5


def test_fl_source_refusals(tmp_path):
    cases = [(dict(arm="PF"), "fl", "--ckpt-kind pf"), (dict(label="SMOKE_NOT_EVIDENCE"), "fl", "smoke"),
             (dict(role="CALIBRATION"), "fl", "LR-screen"), (dict(digest="OTHER"), "fl", "config digest"),
             (dict(small_recipe={"model_variant": VAR, "recipe_id": "02B_SASREC"}), "fl", "SMALL recipe"),
             (dict(arm_spec={"cfg": {"seed": 2027}}), "fl", "seed")]
    for i, (kw, kind, msg) in enumerate(cases):
        d = tmp_path / f"f{i}"
        cp, _, _ = _fl_run(d, **kw)
        with pytest.raises(R.ReevalRefused, match=msg):
            R.run(_argv(cp, d, kind=kind, arm_label="FA"))
    d = tmp_path / "legs"
    cp, _, _ = _fl_run(d, legs=2)
    other = json.loads((d / "fl" / "LAUNCH_1.json").read_text(encoding="utf-8"))
    other["arm_spec_sha256"] = "S2"
    (d / "fl" / "LAUNCH_1.json").write_text(json.dumps(other), encoding="utf-8")
    with pytest.raises(R.ReevalRefused, match="arm_spec_sha256 differs"):
        R.run(_argv(cp, d, kind="fl", arm_label="FA"))


def test_pf_store_pairing(tmp_path):
    from fl.numerics import state_digest
    cp, rid, theta = _fl_run(tmp_path, arm="PF")
    good = _store_file(tmp_path, rid, theta_digest=state_digest(theta))
    got = R.run(_argv(cp, tmp_path, kind="pf", arm_label="PF", pf_store=good))
    assert got["pf_store_sha256"] == _sha(good)
    bad = _store_file(tmp_path, rid, theta_digest="0" * 64, name="PF_STORE_other.pt")
    with pytest.raises(R.ReevalRefused, match="theta_digest"):
        R.run(_argv(cp, tmp_path, kind="pf", arm_label="PF", pf_store=bad))
    legacy = _store_file(tmp_path, rid, None, name="PF_STORE.pt")
    with pytest.raises(R.ReevalRefused, match="endpoint checkpoint only"):
        R.run(_argv(cp, tmp_path, kind="pf", arm_label="PF", pf_store=legacy))
    d = tmp_path / "ep"
    cpe, rid_e, _ = _fl_run(d, arm="PF", ckpt="endpoint_6.0.pt")
    legacy_e = _store_file(d, rid_e, None, name="PF_STORE.pt")
    assert R.run(_argv(cpe, d, kind="pf", arm_label="PF", pf_store=legacy_e))["dry_run"]
    with pytest.raises(R.ReevalRefused, match="expect-pf-store-sha256"):
        R.run(_argv(cp, tmp_path, kind="pf", arm_label="PF", pf_store=good, expect_pf_store_sha256="1" * 64))


# ------------------------------------------------------------------------------------------------ real-row equivalence
def _mk():
    from v4models import build, load_schema
    return build("SASREC", load_schema(), 2026, "cpu", v1_root=REAL_DS, variant=VAR, dropout=(0.1,))


def _perturbed():
    bm = _mk()
    g = torch.Generator().manual_seed(20260929)
    with torch.no_grad():
        for p in bm.adapter.shared_parameters():
            p.add_(0.05 * torch.randn(p.shape, generator=g, dtype=p.dtype))
    return bm


LABELS = {"run_label": "PRODUCTION", "manifest_role": "MAIN", "run_role": "MAIN", "exposure_point": "end",
          "train_manifest_id": "M05_CALIB_TRAIN", "eval_manifest_id": "M06_CALIB_VALIDATION_WARM"}


def _kb():
    from v4eval.rows import COMPARISON_KEY, T3_KEY_ADDITIONS
    kb = {f: f"TEST_{f}" for f in COMPARISON_KEY if f not in T3_KEY_ADDITIONS}
    kb.update(train_manifest_id="M05_CALIB_TRAIN", eval_manifest_id="M06_CALIB_VALIDATION_WARM")
    return kb


def _harness(adapter, ep):
    """The harness contract of launch_fl.role_evaluator / launch_device.pf_evaluator (test_fl05_real_calib pattern)."""
    from v4eval import frozen_config as fc
    return R.central_harness(adapter, frozen=fc.load(), device=torch.device("cpu"), key_base=_kb(), labels=LABELS,
                             seed=2026, evaluator_sha="TEST_EVALUATOR_SHA", evalp=ep)


def _same(ref, got):
    assert got[0] == ref[0]
    assert np.array_equal(np.asarray(got[2]), np.asarray(ref[2]))
    assert got[1] == ref[1]
    assert json.dumps(F._json_safe(got[3]), sort_keys=True) == json.dumps(F._json_safe(ref[3]), sort_keys=True)


@pytest.fixture(scope="module")
def ref_central(providers):
    _, ep = providers
    src = _perturbed()
    ref = F.role_evaluator(_harness(src.adapter, ep), ep, _kb(), {})(0.0)
    return src, ref


def test_central_checkpoint_reproduces_the_harness_rows(tmp_path, providers, ref_central):
    from fl.numerics import state_digest
    _, ep = providers
    src, ref = ref_central
    obj = {"meta": {"model_variant": VAR, "seed": 2026, "config_sha256": "CFG", "init_sha256": src.init_sha256,
                    "state_digest": state_digest(src.adapter.broadcast_state()), "efe": 3.0, "exposures": 7},
           "model": src.module.state_dict()}
    torch.save(obj, tmp_path / "best.pt")
    back = torch.load(tmp_path / "best.pt", map_location="cpu", weights_only=False)
    fresh = _mk()
    start = {"config_sha256": "CFG", "init_sha256": fresh.init_sha256}
    loaded = R.load_central(fresh, back, start, SimpleNamespace(variant=VAR, seed=2026))
    fresh.module.eval()
    assert loaded["efe"] == 3.0 and loaded["exposures"] == 7
    _same(ref, R.evaluate_role(_harness(fresh.adapter, ep), ep, _kb(), {}))
    tampered = dict(back, meta=dict(back["meta"], state_digest="0" * 64))
    with pytest.raises(R.ReevalRefused, match="state digest"):
        R.load_central(_mk(), tampered, start, SimpleNamespace(variant=VAR, seed=2026))
    with pytest.raises(R.ReevalRefused, match="config_sha256"):
        R.load_central(_mk(), back, dict(start, config_sha256="OTHER"), SimpleNamespace(variant=VAR, seed=2026))


def test_fl_theta_reproduces_the_harness_rows(tmp_path, providers, ref_central):
    from fl.checkpoint import atomic_save, load_checkpoint
    _, ep = providers
    src, ref = ref_central
    theta = {k: v.detach().clone() for k, v in src.adapter.broadcast_state().items()}
    atomic_save({"kind": "weights", "run_id": "R", "config_digest": "C", "round": 9, "efe": 3.0, "phase": "MAIN",
                 "theta": theta}, tmp_path / "ckpt" / "R" / "best.pt")
    payload = load_checkpoint(tmp_path / "ckpt" / "R" / "best.pt")
    fresh = _mk()
    R.load_fl(fresh.adapter, payload)
    _same(ref, R.evaluate_role(_harness(fresh.adapter, ep), ep, _kb(), {}))
    wrong = {k: v for k, v in theta.items() if k != sorted(theta)[0]}
    with pytest.raises(R.ReevalRefused, match="broadcast state"):
        R.load_fl(_mk().adapter, dict(payload, theta=wrong))


def _hex_and_store(ep, adapter):
    from fl.client import PersonalState
    from fl.personal import PersonalStore
    m = ep.manifest
    n = int(m.eval_user_key.max()) + 1
    hexes = np.array([hashlib.sha256(f"int06-user-{u}".encode()).hexdigest() for u in range(n)], dtype=object)
    store = PersonalStore(adapter.query_dim)
    g = torch.Generator().manual_seed(7)
    for u in np.unique(m.eval_user_key[m.rankable])[:40]:
        store.commit(str(hexes[int(u)]), PersonalState(0.5 * torch.randn(adapter.query_dim, generator=g), None, 1, 1))
    return hexes, store


def test_pf_theta_and_store_reproduce_the_pf_evaluator(tmp_path, providers, ref_central):
    from fl.checkpoint import atomic_save, load_checkpoint
    from fl.numerics import state_digest
    from fl.personal import PersonalStore
    _, ep = providers
    src, _ = ref_central
    hexes, store = _hex_and_store(ep, src.adapter)
    ref = D.pf_evaluator(_harness(src.adapter, ep), hexes)(0.0, store)
    theta = {k: v.detach().clone() for k, v in src.adapter.broadcast_state().items()}
    atomic_save({"kind": "weights", "run_id": "R", "config_digest": "C", "theta": theta}, tmp_path / "w.pt")
    atomic_save({"kind": "pf_store", "run_id": "R", "config_digest": "C", "store": store.state_dict(),
                 "theta_digest": state_digest(theta), "store_digest": store.digest()}, tmp_path / "s.pt")
    payload, sobj = load_checkpoint(tmp_path / "w.pt"), load_checkpoint(tmp_path / "s.pt")
    fresh = _mk()
    loaded = R.load_fl(fresh.adapter, payload)
    R.check_pf_store(sobj, payload, theta_digest=loaded["state_digest"], ckpt_name="best.pt")
    store2 = PersonalStore.from_state_dict(sobj["store"])
    assert store2.digest() == store.digest()
    _same(ref, R.evaluate_role(_harness(fresh.adapter, ep), ep, _kb(), {}, store=store2, hex_by_euk=hexes))
    assert all(r["pf_personal_component"] == "EVALUATED" and r["method"] == "PF" for r in ref[1])


@nc("an UNLOADED fresh model (theta0) reproduces the saved checkpoint's ranks")
def test_nc_unloaded_theta0_reproduces(providers, ref_central):
    _, ep = providers
    _, ref = ref_central
    got = R.evaluate_role(_harness(_mk().adapter, ep), ep, _kb(), {})
    assert np.array_equal(np.asarray(got[2]), np.asarray(ref[2]))


@nc("scoring the PF weights with an EMPTY store (p = 0, the B1 symptom) reproduces the PF evaluator's ranks")
def test_nc_empty_store_reproduces_pf(providers, ref_central):
    from fl.personal import PersonalStore
    _, ep = providers
    src, _ = ref_central
    hexes, store = _hex_and_store(ep, src.adapter)
    ref = D.pf_evaluator(_harness(src.adapter, ep), hexes)(0.0, store)
    got = R.evaluate_role(_harness(src.adapter, ep), ep, _kb(), {}, store=PersonalStore(src.adapter.query_dim),
                          hex_by_euk=hexes)
    assert np.array_equal(np.asarray(got[2]), np.asarray(ref[2]))


# ------------------------------------------------------------------------------------------------ pure helpers
def test_role_key_base_is_the_launch_device_rule():
    base = {"population_hash": "P0", "candidate_order_hash": "C0", "train_manifest_id": "T0", "eval_manifest_id": "E0",
            "split": "VALIDATION", "data_protocol_hash": "D"}
    rk = {"population_hash": "P1", "candidate_order_hash": "C1", "train_manifest_id": "T1", "eval_manifest_id": "E1",
          "split": "OTHER", "data_protocol_hash": "OTHER"}
    kb, over = R.role_key_base(base, rk, "SPARSE_PANEL")
    assert kb == {"population_hash": "P1", "candidate_order_hash": "C1", "train_manifest_id": "T1",
                  "eval_manifest_id": "E1", "split": "VALIDATION", "data_protocol_hash": "D"}
    assert over == {"manifest_role": "SPARSE_PANEL", "train_manifest_id": "T1", "eval_manifest_id": "E1"}


def test_resolve_role_rules():
    base = json.loads((V4 / "contracts" / "manifest_registry" / "REGISTRY_V2.json").read_text(encoding="utf-8"))["roles"]
    assert R.resolve_role(base, "TRUE_COLD", base) == {"eval": "TRUE_COLD_VALIDATION"}
    for role in R.ROLES:
        assert "eval" in R.resolve_role(base, role, base)
    for roles, role, msg in ((base, "COLD_H", "not a train/eval"), (base, "NOPE", "not in the manifest registry"),
                             (dict(base, MAIN={"train": "X", "eval": "Y"}), "MAIN", "differs between"),
                             (dict(base, MAIN_RETRO_TEST={"eval": "MAIN_RETRO_TEST_5_100_131072"}), "MAIN_RETRO_TEST",
                              "TEST manifest")):
        with pytest.raises(R.ReevalRefused, match=msg):
            R.resolve_role(roles, role, base)


def test_registry_extension_rule_and_wrapper(tmp_path):
    reg_dir = V4 / "contracts" / "manifest_registry"
    base_p, ext_p = reg_dir / "REGISTRY_V2.json", reg_dir / "REGISTRY_V2_FULL_EXT.json"
    base, ext = json.loads(base_p.read_text(encoding="utf-8")), json.loads(ext_p.read_text(encoding="utf-8"))
    assert R.verify_generic_ext(ext, base, _sha(base_p))["all_ok"]          # the DATA-08 extension is additions-only
    bad = json.loads(json.dumps(ext))
    bad["roles"]["MAIN"] = {"train": "FULL_TRAIN_5_100_TIEBREAK", "eval": "MAIN_VALIDATION_5_100_131072"}
    assert not R.verify_generic_ext(bad, base, _sha(base_p))["all_ok"]
    assert not R.verify_generic_ext(ext, base, "0" * 64)["all_ok"]
    calls = []

    def orig(path, lock_self, lock_file, *, lock=None, root=None):
        calls.append(Path(path).name)
        return base, _sha(base_p), base["self_sha256"], None
    good_p = tmp_path / "EXT.json"
    good_p.write_text(ext_p.read_text(encoding="utf-8"), encoding="utf-8")
    w = R.make_ext_verify_wrapper(orig, good_p, base_p)
    assert w(base_p, "L", "F")[0] is base and calls == ["REGISTRY_V2.json"]           # other paths: delegated
    got = w(good_p, "L", "F")
    assert got[0]["self_sha256"] == ext["self_sha256"] and got[1] == _sha(good_p)
    bad_p = tmp_path / "BAD.json"
    bad_p.write_text(json.dumps(bad), encoding="utf-8")
    with pytest.raises(ValueError, match="refused"):
        R.make_ext_verify_wrapper(orig, bad_p, base_p)(bad_p, "L", "F")


def test_relabel_and_pinned_row_guards():
    from v4eval.rows import COMPARISON_KEY
    rows = [{"status": "MEASURED_PINNED_DEVICE", "device_pin_sha256": "p", "eval_device_class": R.PINNED_CLASS,
             "method": "C"}]
    out = R.relabel(rows, arm_label="C_BIG", selection_rule="PRACTICAL_BEST", checkpoint_id="x:best",
                    checkpoint_sha256="s", kind="central", source_run_id="x", efe=3.0)
    assert out[0]["method"] == "C_BIG" and out[0]["reeval"] is True and out[0]["row_status"] == "REEVAL"
    R.check_pinned_rows(out, "p")
    with pytest.raises(R.ReevalRefused, match="comparison-key"):
        R.relabel(rows, arm_label="C_BIG", selection_rule="PRACTICAL_BEST", checkpoint_id="x", checkpoint_sha256="s",
                  kind="central", source_run_id="x", efe=3.0, extra={COMPARISON_KEY[0]: "tampered"})
    for bad in (dict(rows[0], status="REHEARSAL_NON_FROZEN_DEVICE"), dict(rows[0], device_pin_sha256="q"),
                dict(rows[0], eval_device_class="A16-FP32-DET")):
        with pytest.raises(R.ReevalRefused, match="MEASURED_PINNED_DEVICE"):
            R.check_pinned_rows([bad], "p")
    with pytest.raises(R.ReevalRefused):
        R.check_pinned_rows([], "p")


@nc("a TEST role passes the pre-lock gate")
def test_nc_test_role_passes_gate():
    assert R.test_gate_ok("MAIN_RETRO_TEST")


def test_write_outputs_layout_and_no_overwrite(tmp_path):
    out = tmp_path / "REEVAL_A40" / "C_SMALL_s2026_PB"
    out.mkdir(parents=True)
    (out / "REGISTRY_V2_TRAINER_VIEW.json").write_text("{}", encoding="utf-8")     # the FL harness's view file
    rows = [{"metric": "mrr", "k": 20, "value": 0.1}]
    rec = R.write_outputs(out, tag="MAIN", run_id="REEVAL_C_SMALL_MAIN_s2026_PRACTICAL_BEST", role="MAIN", rows=rows,
                          result={"E2E": {"n": 1}}, ranks=np.arange(4), record={"format": R.RECORD_FORMAT})
    assert rec["status"] == "COMPLETE" and rec["n_rows"] == 1
    assert json.loads((out / "RESULT_ROWS.json").read_text(encoding="utf-8"))["rows"] == rows
    assert np.load(out / "evals" / "MAIN.ranks_int32.npy").dtype == np.int32
    saved = json.loads((out / "REEVAL_RECORD.json").read_text(encoding="utf-8"))
    assert saved["outputs"]["RESULT_ROWS.json"] == _sha(out / "RESULT_ROWS.json")
    with pytest.raises(R.ReevalRefused, match="already holds"):
        R.write_outputs(out, tag="MAIN", run_id="x", role="MAIN", rows=rows, result={}, ranks=np.arange(4),
                        record={})


# FIX-DELTA-01 F1: tools/ana_block_index.py reads the ds3r2 top-level names result_rows_sha256 / ranks_sha256.
def _index_view_ok(rec: dict, out: Path) -> bool:
    """The two checks ana_block_index.reeval() makes on a REEVAL_RECORD (file sha == top-level field)."""
    ranks = sorted((out / "evals").glob("*.ranks_int32.npy"))
    return (len(ranks) == 1 and rec.get("result_rows_sha256") == _sha(out / "RESULT_ROWS.json")
            and rec.get("ranks_sha256") == _sha(ranks[0]))


def _written_record(tmp_path) -> tuple:
    out = tmp_path / "REEVAL_A40" / "C_SMALL_s2026_PB"
    R.write_outputs(out, tag="MAIN", run_id="REEVAL_C_SMALL_MAIN_s2026_PRACTICAL_BEST", role="MAIN",
                    rows=[{"metric": "mrr", "k": 20, "value": 0.1}], result={"E2E": {"n": 1}},
                    ranks=np.arange(5), record={"format": R.RECORD_FORMAT})
    return json.loads((out / "REEVAL_RECORD.json").read_text(encoding="utf-8")), out


def test_write_outputs_top_level_hashes_for_the_ana_index(tmp_path):
    saved, out = _written_record(tmp_path)
    assert _index_view_ok(saved, out)
    assert saved["result_rows_sha256"] == saved["outputs"]["RESULT_ROWS.json"]
    assert saved["ranks_sha256"] == saved["outputs"]["evals/MAIN.ranks_int32.npy"]
    src = (V4 / "tools" / "ana_block_index.py").read_text(encoding="utf-8")
    assert "result_rows_sha256" in src and "ranks_sha256" in src            # the index still reads these names


@nc("a record with the hashes only under `outputs` (the pre-FIX-DELTA-01 form) passes the index view")
def test_nc_outputs_only_record_passes_index_view(tmp_path):
    saved, out = _written_record(tmp_path)
    old = {k: v for k, v in saved.items() if k not in ("result_rows_sha256", "ranks_sha256")}
    assert _index_view_ok(old, out)


def test_out_dir_must_be_new(tmp_path):
    cp = _central(tmp_path)
    (tmp_path / "out").mkdir()
    args = _argv(cp, tmp_path)
    args.dry_run = False
    with pytest.raises(R.ReevalRefused, match="output directory exists"):
        R.run(args)
