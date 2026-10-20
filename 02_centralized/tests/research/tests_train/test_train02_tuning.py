"""TRAIN-02 items 2 (trainer side), 4 and 5 — the TUNING role in the trainer, its evaluation marks, the 5.0 / 5.5 / 6.0
snapshots and CKPT_AVG, HP_SEARCH rows refused by selection, resume-exact per variant.

  * a TUNING run evaluates at 2 / 4 / 6 EFE only (the endpoint read is 6.0), saves weights-only snapshots at 5.0 / 5.5 /
    6.0 (sha256 in CHECKPOINTS.json), writes RESULT_ROWS (6.0, CONTROLLED_BUDGET, checkpoint_variant ENDPOINT), deletes
    `latest` (no continuation; continue_run refused), then CKPT_AVG: the module evaluated is exactly the uniform average
    of the three snapshot files (float64 mean cast to fp32), through the trainer's own _evaluate; the rows are CKPT_AVG;
    the snapshots and the average are deleted with their sha256 logged; no weight file is left; the disk peak is
    reported; the LR follows the trial's schedule (the planned table of RUN_START is the TUNING / S0|S1 table);
  * the real v4eval path end to end (synthetic providers) for the CKPT_AVG evaluation;
  * non-test TUNING runs (device class patched to the frozen A16 class) write HP_SEARCH rows (RESULT_ROWS, CKPT_AVG,
    every mark), refused by every selection consumer; theta0_init_sha256 must equal the built theta0;
  * resume-exact for each of the five variants (interrupted between the 5.0 and 5.5 snapshots) and crash recovery
    inside CKPT_AVG (before and after its evaluation); a resume with a changed HP registry file is refused;
  * the dropout of the HP trial is checked against the built module; model_variant / theta0_init_sha256 in key_base and
    every row; the current builds keep their topology hash.
Negative controls: snapshots kept after CKPT_AVG; CKPT_AVG evaluating the endpoint weights instead of the average; a
TUNING run evaluated at every 0.5 EFE; selection accepting HP_SEARCH rows.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
from collections import OrderedDict
from pathlib import Path

import pytest
import torch

import v4train.ckpt_avg as CA
import v4train.trainer as T
from trainkit import (BIG_DISK, K, KEY, N, NEW_DATA_VERSION, SCHEMA, assert_raises, cfg, interrupt_after, nc,
                      providers, raises_with, read, scripted, trainer, write_registry)
from test_t22_selection import WIDE
from fl.numerics import state_digest
from v4models import build
from v4models import variants as V
from v4train import hp_registry as H
from v4train import selection as SEL
from v4train.evalconfig import frozen
from v4train.schedule import MAIN_END, eval_marks, planned_lr_table
from v4train.selection import SelectionError, assert_c_matched_eligible, calibration_choice, select_family
from v4train.trainer import TrainerError

TRIALS = {t["hp_id"]: t for fam in H.load_hp_registry()["families"].values() for t in fam}
ONE_PER_VARIANT = {}
for _t in TRIALS.values():
    ONE_PER_VARIANT.setdefault(_t["model_variant"], _t["hp_id"])
VALS = {**{round(0.5 * i, 6): 0.2 for i in range(1, 13)}, 2.0: 0.21, 4.0: 0.24, 6.0: 0.25}


def tcfg(run_id, hp_id, **kw):
    t = TRIALS[hp_id]
    return cfg(run_id, t["family"], phase="CALIBRATION", role="TUNING", hp_id=hp_id, peak=t["peak_lr"],
               model_variant=t["model_variant"], dropout=tuple(t["dropout"]), schedule=t["schedule"], **kw)


class Recorder:
    """A scripted evaluator that records every call and, at the CKPT_AVG call, compares the module with the manual
    uniform average of the three snapshot files on disk."""

    def __init__(self, values=VALS):
        self.values, self.calls, self.avg_ok, self.differs_from_endpoint = dict(values), [], None, None
        self._ev = scripted(self.values)

    def __call__(self, tr, mark):
        in_avg = "ckpt_avg" in tr.ckpt.index
        self.calls.append((round(mark, 6), "CKPT_AVG" if in_avg else "MARK"))
        if in_avg:
            snaps = [torch.load(tr.run_dir / tr.ckpt.index[k]["file"], map_location="cpu", weights_only=True)["model"]
                     for k in ("snapshot_5.0", "snapshot_5.5", "snapshot_6.0")]
            manual = OrderedDict()
            for key, t0 in snaps[0].items():
                if t0.is_floating_point():
                    acc = torch.zeros(t0.shape, dtype=torch.float64)
                    for s in snaps:
                        acc += s[key].to(torch.float64)
                    manual[key] = (acc / 3).to(t0.dtype)
                else:
                    manual[key] = t0
            now = tr.module.state_dict()
            self.avg_ok = state_digest(now) == state_digest(manual)
            self.differs_from_endpoint = state_digest(now) != state_digest(snaps[2])
        return self._ev(tr, mark)


def _run_tuning(work, name, hp_id, rec=None, **kw):
    rec = rec or Recorder()
    tr = trainer(work / name, tcfg(name, hp_id, **kw), eval_override=rec)
    return tr, tr.run(), rec


def _log(run_dir):
    return [json.loads(x) for x in (Path(run_dir) / "RETENTION_LOG.jsonl").read_text(encoding="utf-8").splitlines()]


def _check_tuning_run(work, hp_id):
    tr, s, rec = _run_tuning(work, "t", hp_id)
    rd = work / "t"
    assert [c for c in rec.calls] == [(2.0, "MARK"), (4.0, "MARK"), (6.0, "MARK"), (6.0, "CKPT_AVG")]
    assert rec.avg_ok is True and rec.differs_from_endpoint is True
    assert s["status"] == "DONE_TUNING" and s["values_at_marks"] == {"2.0": 0.21, "4.0": 0.24, "6.0": 0.25}
    assert s["endpoint"]["value"] == 0.25 and s["run_role"] == "TUNING" and s["model_variant"] == TRIALS[hp_id][
        "model_variant"]
    rows = read(rd / "RESULT_ROWS.json")["rows"]
    assert rows and all(r["selection_rule"] == "CONTROLLED_BUDGET" and r["checkpoint_variant"] == "ENDPOINT"
                        and abs(r["efe"] - 6.0) < 1e-12 and r["row_status"] == "HP_SEARCH" for r in rows)
    ca = read(rd / "CKPT_AVG.json")
    assert ca["status"] == "COMPLETE" and [x["mark"] for x in ca["snapshots"]] == [5.0, 5.5, 6.0]
    log = _log(rd)
    deleted = {Path(e["deleted"]).stem: e["sha256"] for e in log}
    assert set(deleted) == {"latest", "snapshot_5.0", "snapshot_5.5", "snapshot_6.0", "ckpt_avg"}
    for x in ca["snapshots"]:
        assert deleted[x["kind"]] == x["sha256"], "each snapshot's deletion logged with its recorded sha256"
    assert deleted["ckpt_avg"] == ca["average"]["sha256"] == ca["deleted_sha256"]["ckpt_avg"]
    assert tr.ckpt.weight_files() == [] and s["weight_files_left"] == [] and not list((rd / "checkpoints").glob("*"))
    sizes = {Path(e["deleted"]).stem: e["bytes"] for e in log}
    assert s["disk_peak_bytes"] >= sizes["latest"] + sizes["snapshot_5.0"] + sizes["snapshot_5.5"]
    ca_rows = read(rd / "CKPT_AVG_ROWS.json")["rows"]
    assert ca_rows and all(r["checkpoint_variant"] == "CKPT_AVG" and r["row_status"] == "HP_SEARCH" for r in ca_rows)
    start = read(rd / "RUN_START.json")
    t = TRIALS[hp_id]
    plan = planned_lr_table(t["peak_lr"], N, 32, "TUNING", "end", schedule=t["schedule"])
    assert start["planned_lr_table"]["sha256"] == plan["sha256"] and start["planned_lr_table"]["schedule"] == \
        t["schedule"] and start["train_phase"] == "TUNING"
    assert start["hp"]["hp_sha256"] == t["sha256"] and start["hp"]["hp_id"] == hp_id
    assert start["model_variant_registry"]["file_sha256"] == V.load_variant_registry()["file_sha256"]
    ev = [json.loads(x) for x in (rd / "EVENTS.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [e["mark"] for e in ev if e.get("event") == "SNAPSHOT"] == [5.0, 5.5]
    for r in rows + ca_rows:
        assert r["model_variant"] == t["model_variant"] and r["theta0_init_sha256"] == tr.built.init_sha256
        assert r["hp_id"] == hp_id and r["schedule"] == t["schedule"] and r["dropout"] == t["dropout"]
    raises_with(TrainerError, "has no continuation", trainer(work / "t", tr.cfg, tr.train, tr.evalp).continue_run)
    return tr, s


@pytest.mark.parametrize("hp_id", ["P2_GRU_D256_S1_s4", "P2_SASREC_D384_B2_S0_s3"])
def test_tuning_run_marks_snapshots_and_ckpt_avg(work, hp_id):
    _check_tuning_run(work, hp_id)


@nc("a CKPT_AVG step that keeps the snapshots and the average")
def test_nc_snapshots_kept(work, monkeypatch):
    def keep(tr, rec, kinds):
        from v4train.checkpoint import write_json_atomic
        rec = dict(rec, status="COMPLETE", deleted_sha256={}, weight_files_left=tr.ckpt.weight_files())
        write_json_atomic(Path(tr.run_dir) / CA.RECORD, rec)
        return rec
    monkeypatch.setattr(CA, "_finish", keep)                                                     # MUTANT
    _check_tuning_run(work, "P2_GRU_D256_S1_s4")


@nc("a CKPT_AVG that evaluates the 6.0 weights instead of the uniform average")
def test_nc_ckpt_avg_is_the_endpoint(work, monkeypatch):
    monkeypatch.setattr(CA, "uniform_average", lambda states: OrderedDict((k, v.clone()) for k, v in states[-1].items()))
    _check_tuning_run(work, "P2_GRU_D256_S1_s4")


@nc("a TUNING run evaluated at every 0.5 EFE (the MAIN marks)")
def test_nc_tuning_every_half_efe(work, monkeypatch):
    real = T.eval_marks
    monkeypatch.setattr(T, "eval_marks", lambda phase: real("MAIN") if phase == "TUNING" else real(phase))
    _check_tuning_run(work, "P2_GRU_D256_S1_s4")


def test_uniform_average_unit():
    a = OrderedDict(w=torch.tensor([1.0, 2.0]), i=torch.tensor([3, 4]))
    b = OrderedDict(w=torch.tensor([2.0, 4.0]), i=torch.tensor([3, 4]))
    c = OrderedDict(w=torch.tensor([3.0, 9.0]), i=torch.tensor([3, 4]))
    avg = CA.uniform_average([a, b, c])
    assert torch.equal(avg["w"], torch.tensor([2.0, 5.0])) and avg["w"].dtype == torch.float32
    assert torch.equal(avg["i"], torch.tensor([3, 4]))
    assert_raises(CA.CkptAvgError, CA.uniform_average, [a, OrderedDict(w=a["w"], i=torch.tensor([3, 5]))])
    assert_raises(CA.CkptAvgError, CA.uniform_average, [a])
    assert_raises(CA.CkptAvgError, CA.uniform_average, [a, OrderedDict(i=a["i"], w=a["w"])])


def test_real_evaluator_ckpt_avg_end_to_end(work):
    tr = trainer(work / "real", tcfg("real", "P2_GRU_D128_S0_s3"))
    s = tr.run()
    ca = read(work / "real" / "CKPT_AVG.json")
    assert s["status"] == "DONE_TUNING" and ca["status"] == "COMPLETE" and 0.0 <= ca["value"] <= 1.0
    assert ca["endpoint_value"] == s["endpoint"]["value"] and ca["evaluator"].startswith("CentralTrainer._evaluate")
    assert (work / "real" / "evals" / "ckpt_avg_6.0.result.json").exists()
    rows = read(work / "real" / "CKPT_AVG_ROWS.json")["rows"]
    assert rows and all(r["method"] == "C" and r["checkpoint_variant"] == "CKPT_AVG" for r in rows)
    assert sorted(p.name for p in (work / "real" / "evals").glob("eval_*.rows.json")) == \
        ["eval_2.0.rows.json", "eval_4.0.rows.json", "eval_6.0.rows.json"]


# ------------------------------------------------------------------------------------------------ non-test runs
def _frozen_device_class(monkeypatch):
    monkeypatch.setattr(T, "runtime_device_class", lambda device: frozen().device_class)


def _non_test(work, name, hp_id, theta0=None, **kw):
    t = TRIALS[hp_id]
    tp, ep = providers("CALIBRATION", n=1024, role="TUNING")
    reg = write_registry({"CALIBRATION": (tp, ep)}, data_version=NEW_DATA_VERSION)
    theta0 = theta0 or build(t["family"], SCHEMA, 2026, K=K, variant=t["model_variant"]).init_sha256
    c = dataclasses.replace(tcfg(name, hp_id, manifest_registry=reg, data_version=NEW_DATA_VERSION,
                                 effective_batch=256, theta0_init_sha256=theta0, **kw), test_overrides=False)
    return trainer(work / name, c, tp, ep, eval_override=scripted(VALS))


def _check_hp_rows_refused(rows, ca_rows):
    for bad in (rows, ca_rows):
        raises_with(SelectionError, "HP_SEARCH row is refused", select_family, bad, delta_bar_ci=WIDE)
        raises_with(SelectionError, "HP_SEARCH", calibration_choice, {1.0: bad[:1], 0.5: bad[:1]})
        raises_with(SelectionError, "HP_SEARCH row is refused", assert_c_matched_eligible,
                    dict(bad[0], c_matched_eligible=True))
        assert_raises(SelectionError, assert_c_matched_eligible, bad[0])


@pytest.mark.parametrize("hp_id", ["P2_GRU_D256_S0_s0", "P2_SASREC_D256_B3_S1_s1"])
def test_non_test_tuning_rows_are_hp_search_and_refused(work, monkeypatch, hp_id):
    _frozen_device_class(monkeypatch)
    tr = _non_test(work, "nt", hp_id)
    assert tr.run_label == "HP_SEARCH" and tr.cfg.manifest_role == "CALIBRATION"
    s = tr.run()
    rows = read(work / "nt" / "RESULT_ROWS.json")["rows"]
    ca_rows = read(work / "nt" / "CKPT_AVG_ROWS.json")["rows"]
    marks = [r for p in sorted((work / "nt" / "evals").glob("eval_*.rows.json")) for r in read(p)["rows"]]
    assert s["run_label"] == "HP_SEARCH" and s["status"] == "DONE_TUNING"
    for r in rows + ca_rows + marks:
        assert r["run_label"] == "HP_SEARCH" and r["c_matched_eligible"] is False and r["downstream_seed_allowed"] is False
        assert r["manifest_role"] == "CALIBRATION" and r["run_role"] == "TUNING"
    _check_hp_rows_refused(rows, ca_rows)


@nc("a selection that accepts HP_SEARCH rows as evidence")
def test_nc_selection_accepts_hp_search(work, monkeypatch):
    _frozen_device_class(monkeypatch)
    for stage in ("COHORT", "CALIBRATION", "C_MATCHED"):
        monkeypatch.setitem(SEL.EVIDENCE_LABELS, stage, ("PRODUCTION", "HP_SEARCH"))           # MUTANT
    tr = _non_test(work, "nc", "P2_GRU_D256_S0_s0")
    tr.run()
    _check_hp_rows_refused(read(work / "nc" / "RESULT_ROWS.json")["rows"],
                           read(work / "nc" / "CKPT_AVG_ROWS.json")["rows"])


def test_theta0_mismatch_refused(work, monkeypatch):
    _frozen_device_class(monkeypatch)
    raises_with(TrainerError, "!= the registered theta0_init_sha256", _non_test, work, "bad", "P2_GRU_D256_S0_s0",
                theta0="0" * 64)
    assert not (work / "bad" / "RUN_START.json").exists()


# ------------------------------------------------------------------------------------------------ resume
def _fingerprint(rd):
    s = read(rd / "RUN_SUMMARY.json")
    ca = read(rd / "CKPT_AVG.json")
    rows = read(rd / "RESULT_ROWS.json")["rows"]
    return {"final_state_digest": s["final_state_digest"], "status": s["status"], "evals": [
        {k: e[k] for k in ("mark", "value", "exposures", "updates")} for e in s["evals"]],
        "snapshots": ca["snapshots"], "avg": ca["average"]["sha256"], "ca_value": ca["value"],
        "rows": [{k: v for k, v in r.items() if k not in ("checkpoint_id",)} for r in rows]}


@pytest.mark.parametrize("name", sorted(n for n in V.VARIANTS if n in ONE_PER_VARIANT))   # Ops: downstream-only SMALL variants have no P2 hp_id (TRAIN-04)
def test_resume_exact_per_variant(work, name):
    hp_id = ONE_PER_VARIANT[name]
    trainer(work / "full", tcfg("r", hp_id), eval_override=scripted(VALS)).run()
    a = trainer(work / "cut", tcfg("r", hp_id), eval_override=scripted(VALS), fault=interrupt_after(37))
    assert_raises(T.InjectedInterruption, a.run)                                  # between the 5.0 and 5.5 snapshots
    assert read(work / "cut" / "CHECKPOINTS.json").get("snapshot_5.0") and \
        not read(work / "cut" / "CHECKPOINTS.json").get("snapshot_5.5")
    b = trainer(work / "cut", tcfg("r", hp_id), eval_override=scripted(VALS))
    b.resume()
    assert _fingerprint(work / "full") == _fingerprint(work / "cut"), name


def test_crash_inside_ckpt_avg_resumes(work, monkeypatch):
    trainer(work / "ref", tcfg("c", "P2_SASREC_D384_B2_S1_s2"), eval_override=scripted(VALS)).run()
    real_finish, real_run = CA._finish, CA.run_ckpt_avg
    # (a) a crash before CKPT_AVG runs: RUN_SUMMARY is PENDING_CKPT_AVG, `latest` already deleted
    monkeypatch.setattr(CA, "run_ckpt_avg", lambda tr: (_ for _ in ()).throw(RuntimeError("crash before CKPT_AVG")))
    assert_raises(RuntimeError, trainer(work / "a", tcfg("c", "P2_SASREC_D384_B2_S1_s2"), eval_override=scripted(VALS)).run)
    assert read(work / "a" / "RUN_SUMMARY.json")["status"] == "PENDING_CKPT_AVG"
    monkeypatch.setattr(CA, "run_ckpt_avg", real_run)
    s = trainer(work / "a", tcfg("c", "P2_SASREC_D384_B2_S1_s2"), eval_override=scripted(VALS)).resume()
    assert s["status"] == "DONE_TUNING" and _fingerprint(work / "a") == _fingerprint(work / "ref")
    # (b) a crash after the evaluation, during the deletions: the record is EVALUATED; resume only deletes
    calls = []

    def crash_once(tr, rec, kinds):
        if not calls:
            calls.append(1)
            raise RuntimeError("crash during the deletions")
        return real_finish(tr, rec, kinds)
    monkeypatch.setattr(CA, "_finish", crash_once)
    assert_raises(RuntimeError, trainer(work / "b", tcfg("c", "P2_SASREC_D384_B2_S1_s2"), eval_override=scripted(VALS)).run)
    assert read(work / "b" / "CKPT_AVG.json")["status"] == "EVALUATED"
    s = trainer(work / "b", tcfg("c", "P2_SASREC_D384_B2_S1_s2"), eval_override=scripted(VALS)).resume()
    assert s["status"] == "DONE_TUNING" and _fingerprint(work / "b") == _fingerprint(work / "ref")
    assert not list((work / "b" / "checkpoints").glob("*.pt"))


def test_resume_refuses_a_changed_hp_registry_file(work):
    obj = json.loads(H.REGISTRY_PATH.read_text(encoding="utf-8"))
    p = work / "HP_COPY.json"
    p.write_text(json.dumps(obj, indent=1, sort_keys=True), encoding="utf-8")
    a = trainer(work / "r", tcfg("h", "P2_GRU_D128_S1_s4", hp_registry=str(p)), eval_override=scripted(VALS),
                fault=interrupt_after(20))
    assert_raises(T.InjectedInterruption, a.run)
    p.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")        # same content, other bytes
    before = sorted(x.name for x in (work / "r").rglob("*"))
    raises_with(TrainerError, "hp_registry_file_sha256",
                trainer(work / "r", tcfg("h", "P2_GRU_D128_S1_s4", hp_registry=str(p)), eval_override=scripted(VALS)).resume)
    assert sorted(x.name for x in (work / "r").rglob("*")) == before
    p.write_text(json.dumps(obj, indent=1, sort_keys=True), encoding="utf-8")
    s = trainer(work / "r", tcfg("h", "P2_GRU_D128_S1_s4", hp_registry=str(p)), eval_override=scripted(VALS)).resume()
    assert s["status"] == "DONE_TUNING"


# ------------------------------------------------------------------------------------------------ build checks, keys
def test_trial_dropout_checked_and_keys(work):
    tr = trainer(work / "d", tcfg("d", "P2_GRU_D256_S1_s1"))
    assert (tr.module.input_dropout.p, tr.module.readout_dropout.p) == (0.3, 0.15)
    tr._check_dropout()
    tr.module.readout_dropout.p = 0.1                                             # the recipe value, not the trial's
    raises_with(TrainerError, "GRU dropout", tr._check_dropout)
    s = trainer(work / "s", tcfg("s", "P2_SASREC_D384_B2_S0_s0"))
    assert s.module.cfg.dropout == 0.1 and s.built.variant == "SASREC_D384_B2"
    object.__setattr__(s.module.cfg, "dropout", 0.2)
    raises_with(TrainerError, "SASRec dropout", s._check_dropout)
    kb = tr.key_base
    assert kb["model_variant"] == "GRU_D256" and kb["theta0_init_sha256"] == tr.built.init_sha256
    cur = trainer(work / "c", cfg("c", "GRU"))
    assert cur.key_base["model_variant"] == "GRU_D128"
    topo = {"topology": T.TOPOLOGY["GRU"], "widths": cur.built.widths.as_json(),
            "manifest_digest": cur.built.manifest_digest, "schema_self_sha256": SCHEMA.self_sha256}
    assert cur.key_base["topology_hash"] == T._sha_obj(json.loads(json.dumps(topo, default=list))), \
        "the current build keeps its TRAIN-01R2 topology hash"
    assert kb["topology_hash"] != cur.key_base["topology_hash"]
    raises_with(TrainerError, "config.key model_variant", trainer, work / "k",
                cfg("k", "GRU", key=dict(KEY, model_variant="GRU_D256")))
