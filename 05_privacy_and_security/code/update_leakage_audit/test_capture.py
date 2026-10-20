"""Scientific invariants for audit-only upload observation (synthetic data)."""
from dataclasses import asdict, replace
from pathlib import Path
import json
import random

import numpy as np
import pytest
import torch

from fl.client import FaultPlan, LocalSolver, PFConfig, valid_rows
from fl.dp import DPConfig
from fl.numerics import state_digest, enforce_strict_fp32
from fl.participation import ParticipationPlan
from fl.runtime import FLRun, RunConfig, freeze_item_tables
from fl.server import Server, ServerOptConfig, run_round
from fl.synthetic import make_clients, make_tiny_adapter, TinyRec, TinyAdapter
from e3_audit.capture import UploadCapture, declare_population, sparse_input, canonical_sha
from e3_audit.replay import build_audit_spec, validate_args, normalize_isolated_path


def fixture(tmp_path, *, capture_on=True, frozen=False, mutate=False, tied=False, device="cpu"):
    torch.set_num_threads(1)
    enforce_strict_fp32(warn_only=False)
    torch.manual_seed(917)
    np.random.seed(918)
    random.seed(919)
    server_adapter = TinyAdapter(TinyRec(seed=11, tied=tied, K=12, d=4).to(device))
    worker = TinyAdapter(TinyRec(seed=12, tied=tied, K=12, d=4).to(device))
    if frozen:
        freeze_item_tables(server_adapter)
        freeze_item_tables(worker, source=server_adapter.broadcast_state())
    server = Server(server_adapter, init_sha256=state_digest(server_adapter.broadcast_state()))
    clients = make_clients(32, K=12, L=4, seed=21, sizes=[3+i%4 for i in range(32)], invalid_frac=0)
    by_key = {c.key: c for c in clients}
    cfg = RunConfig(run_id="E3_TOY", method="FA", seed=2026, lr_peak_local=0.01,
                    solver=LocalSolver(lr=0.01, passes=1, batch_size=3, weight_decay=0),
                    n_shards=2, exposure_point="end", endpoint_rounds=8,
                    server_opt=ServerOptConfig("fedadam", lr=0.00003, tau=1e-6), lr_shape="fl_const")
    plan = ParticipationPlan.build(list(by_key), seed=2026, manifest_hash="SYNTHETIC_CALIB", group_size=4)
    population = declare_population(plan, cfg, rounds=4, sample_size=8, seed=73, holdout_rounds=1)
    capture = None
    if capture_on:
        capture = UploadCapture(parent=tmp_path, population=population, truth_fn=lambda k: {"key": k},
                                manifest=server.manifest)
        if mutate:
            original = capture.observe_upload
            def observe(**kw):
                original(**kw)
                for t in kw["delta"].values():
                    t.fill_(9)  # adversarial observer owns these copies, never training state
            capture.observe_upload = observe
    run = FLRun(cfg, plan, server, by_key.__getitem__, sum(len(c.examples["target_class"]) for c in clients),
                workers=[worker], upload_observer=capture)
    return run, capture, by_key


def signature(run):
    def plain(x):
        if isinstance(x, (np.ndarray, torch.Tensor)):
            return x.tolist()
        if isinstance(x, dict):
            return {k:plain(v) for k,v in x.items()}
        if isinstance(x, (tuple,list)):
            return [plain(v) for v in x]
        if isinstance(x, set):
            return sorted(plain(v) for v in x)
        return x
    return plain({"theta": run.server.digest(), "opt": run.server.server_opt.digest(),
            "cfg": run.cfg.digest(), "plan": run.plan.digest(), "lr_table": run.lr_table["digest"],
            "counters": asdict(run.counters), "ledger": asdict(run.ledger), "lr_log": run.lr_log,
            "state": asdict(run.state), "torch_rng": torch.get_rng_state().tolist(),
            "cuda_rng": [x.tolist() for x in torch.cuda.get_rng_state_all()] if next(
                run.server.adapter.module.parameters()).device.type == "cuda" else None,
            "numpy_rng": np.random.get_state()[1].tolist(), "python_rng": random.getstate()})


@pytest.mark.parametrize("frozen", [False, True])
def test_capture_preserves_every_bit_rng_counter_and_communication(tmp_path, frozen):
    off, _, _ = fixture(tmp_path/"off", capture_on=False, frozen=frozen)
    off.run(max_rounds=4)
    expected = signature(off)
    on, capture, _ = fixture(tmp_path/"on", frozen=frozen)
    with capture:
        on.run(max_rounds=4)
        assert signature(on) == expected
        assert capture.verify_complete()["n_individual"] == 8
        assert len(list(capture.read_cohorts())) == 4
    assert not capture.directory.exists()


def test_observer_cannot_change_live_upload_or_server(tmp_path):
    off, _, _ = fixture(tmp_path/"off", capture_on=False)
    off.run(max_rounds=4)
    expected = signature(off)
    on, capture, _ = fixture(tmp_path/"on", mutate=True)
    with capture:
        on.run(max_rounds=4)
        assert signature(on) == expected


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires the audited server GPU")
@pytest.mark.parametrize("frozen", [False, True])
def test_gpu_capture_preserves_state_and_cuda_rng(tmp_path, frozen):
    off, _, _ = fixture(tmp_path/"off", capture_on=False, frozen=frozen, device="cuda:0")
    off.run(max_rounds=4)
    expected = signature(off)
    on, capture, _ = fixture(tmp_path/"on", frozen=frozen, device="cuda:0")
    with capture:
        on.run(max_rounds=4)
        assert signature(on) == expected
        capture.verify_complete()


def test_retry_is_observed_once_and_no_failed_attempt_survives(tmp_path):
    run, capture, by_key = fixture(tmp_path)
    first = run.plan.round_keys(0)[0]
    fault = FaultPlan(frozenset({(first, 0, 1)}))
    with capture:
        for r in range(4):
            clients = [by_key[k] for k in run.plan.round_keys(r)]
            report = run_round(run.server, run.workers, clients, run.cfg.solver, round_idx=r, seed=2026,
                               n_shards=2, fault=fault, upload_observer=capture)
            if r == 0:
                assert report.visits[0].attempts == 2
        assert len(capture.seen) == 16
        assert capture.verify_complete()["n_individual"] == 8
        assert len({(x["round"], x["key"]) for x in capture.read_records()}) == 8


def test_duplicate_and_unplanned_successful_observations_refused(tmp_path):
    run, capture, _ = fixture(tmp_path)
    with capture:
        run.step()
        with pytest.raises(ValueError, match="duplicate"):
            capture.begin_round(round_idx=0, cohort_keys=run.plan.round_keys(0))
        capture.begin_round(round_idx=1, cohort_keys=run.plan.round_keys(1))
        kw = dict(round_idx=1, position=0, key="UNDECLARED", n_consumed=2, n_valid=2, attempts=1, delta={})
        with pytest.raises(ValueError, match="invalid"):
            capture.observe_upload(**kw)


def test_exception_cleanup_removes_all_transient_update_files(tmp_path):
    run, capture, _ = fixture(tmp_path)
    with pytest.raises(RuntimeError, match="analysis failure"):
        with capture:
            run.step()
            assert any(capture.directory.iterdir())
            raise RuntimeError("analysis failure")
    receipt = json.loads(capture.receipt_path.read_text())
    assert receipt["outcome"] == "FAILED_OR_INCOMPLETE"
    assert receipt["deleted_files"] > 0 and receipt["private_directory_removed"]
    assert not list(tmp_path.glob("E3_PRIVATE_UPLOADS_*"))


def test_cleanup_after_injected_client_failure(tmp_path):
    run, capture, by_key = fixture(tmp_path)
    first = run.plan.round_keys(0)[1]
    fault = FaultPlan(frozenset({(first, 0, 1), (first, 1, 1)}))
    with pytest.raises(RuntimeError):
        with capture:
            run_round(run.server, run.workers, [by_key[k] for k in run.plan.round_keys(0)], run.cfg.solver,
                      round_idx=0, seed=2026, n_shards=1, fault=fault, upload_observer=capture)
    assert not capture.directory.exists()


def test_cohort_matches_weighted_actual_raw_delta_not_mean_row_norms(tmp_path):
    run, capture, _ = fixture(tmp_path)
    contributions = []
    original = capture.observe_upload
    def observe(**kw):
        contributions.append((kw["n_consumed"], {k: t.clone() for k,t in kw["delta"].items()}))
        original(**kw)
    capture.observe_upload = observe
    with capture:
        run.step()
        cohort = next(capture.read_cohorts())
        weight = sum(w for w,_ in contributions)
        inp = sum(w*d["item_embed.weight"].double() for w,d in contributions)/weight
        ids, norms = sparse_input(inp)
        assert torch.equal(ids, cohort["input_row_ids"])
        assert torch.equal(norms, cohort["input_row_norms"])
        enc = torch.cat([sum(w*d[k].double() for w,d in contributions).flatten()/weight for k in capture.encoder_keys])
        assert torch.equal(enc.float(), cohort["encoder_delta"])
        assert cohort["weight"] == weight


@pytest.mark.parametrize("role", ["MAIN", "TEST", "SMOKE"])
def test_noncalib_roles_refused_before_capture_directory_creation(tmp_path, role):
    run, capture, _ = fixture(tmp_path/"valid")
    capture.cleanup()
    with pytest.raises(ValueError, match="CALIBRATION"):
        UploadCapture(parent=tmp_path/"invalid", population=capture.population, role=role,
                      manifest=run.server.manifest, truth_fn=lambda k:{})
    assert not (tmp_path/"invalid").exists()


def test_resume_refused(tmp_path):
    run, capture, _ = fixture(tmp_path)
    with capture:
        with pytest.raises(RuntimeError, match="resume"):
            FLRun.resume(run.cfg, run.plan, run.server, run.get_client, run.n_decisions_total,
                         workers=run.workers, upload_observer=capture)
        with pytest.raises(RuntimeError, match="resumed state"):
            run.load_state_dict(run.state_dict())


@pytest.mark.parametrize("mode", ["dp", "personal", "pool"])
def test_dp_personal_and_process_pool_capture_refused(tmp_path, mode):
    run, capture, _ = fixture(tmp_path)
    with capture:
        cfg = replace(run.cfg, dp=DPConfig(1.,1.,4)) if mode=="dp" else (
            replace(run.cfg, method="PF", pf=PFConfig(.01)) if mode=="personal" else run.cfg)
        kw = dict(workers=run.workers) if mode!="pool" else dict(pool=object())
        with pytest.raises(ValueError, match="raw-upload audit"):
            FLRun(cfg, run.plan, run.server, run.get_client, run.n_decisions_total,
                  upload_observer=capture, **kw)


def test_tied_or_frozen_input_is_explicitly_not_applicable(tmp_path):
    for name, kwargs, status in [("tied", {"tied":True}, "TIED_OUTPUT_NOT_APPLICABLE"),
                                  ("frozen", {"frozen":True}, "FROZEN_NOT_UPLOADED")]:
        run, capture, _ = fixture(tmp_path/name, **kwargs)
        with capture:
            run.run(max_rounds=4)
            records = list(capture.read_records())
            assert all(x["input_table_status"] == status and x["input_row_ids"].numel()==0 for x in records)


def test_population_predeclared_reproducible_and_disjoint(tmp_path):
    run, capture, _ = fixture(tmp_path)
    with capture:
        pop = declare_population(run.plan, run.cfg, rounds=4, sample_size=8, seed=73, holdout_rounds=1)
        assert pop == capture.population
        train = {x["key"] for x in pop["selected"] if x["split"]=="train"}
        hold = {x["key"] for x in pop["selected"] if x["split"]=="holdout"}
        assert not train.intersection(hold)
        assert len(train|hold)==8
        assert {x["cohort_id"] for x in pop["selected"] if x["split"]=="train"}.isdisjoint(
            {x["cohort_id"] for x in pop["selected"] if x["split"]=="holdout"})


def test_tampered_population_refused(tmp_path):
    run, capture, _ = fixture(tmp_path)
    capture.cleanup()
    bad = dict(capture.population, sample_size=9)
    with pytest.raises(ValueError, match="self-hash"):
        UploadCapture(parent=tmp_path/"invalid", population=bad, manifest=run.server.manifest, truth_fn=lambda k:{})


@pytest.mark.parametrize("mutation", ["round", "duplicate", "partition", "split_counts", "repeat_user"])
def test_freshly_hashed_malformed_population_refused(tmp_path, mutation):
    import copy
    from e3_audit.capture import check_population
    run, capture, _ = fixture(tmp_path)
    capture.cleanup()
    bad = copy.deepcopy(capture.population)
    if mutation=="round":
        bad["cohorts"][1]["round"]=0
    elif mutation=="duplicate":
        bad["cohorts"][0]["planned_keys"].append(bad["cohorts"][0]["planned_keys"][0])
    elif mutation=="partition":
        bad["cohorts"][0]["dropped_keys"] = [bad["cohorts"][0]["observed_keys"][0]]
    elif mutation=="split_counts":
        bad["split_counts"]["train"] += 1
    else:
        old = bad["cohorts"][1]["planned_keys"][0]
        new = bad["cohorts"][0]["planned_keys"][0]
        for name in ("planned_keys", "observed_keys", "dropped_keys"):
            bad["cohorts"][1][name] = [new if x==old else x for x in bad["cohorts"][1][name]]
    bad["self_sha256"] = canonical_sha({k:v for k,v in bad.items() if k!="self_sha256"})
    with pytest.raises(ValueError):
        check_population(bad)


def test_set_support_uses_actual_nonzero_input_rows_and_excludes_padding():
    delta = torch.tensor([[2.,2.],[0.,0.],[1.,0.],[0.,0.],[-1.,1.]])
    ids, norms = sparse_input(delta)
    assert ids.tolist()==[2,4]
    assert torch.allclose(norms, torch.tensor([1.,2**.5]))


def test_physical_virtual_eid_paths_normalize_without_widening_output_guard(monkeypatch):
    physical = Path('/home/jovyan/EID PROJECT NEW')
    virtual = Path('/home/jovyan/ahmedabdelhamed01')
    review = virtual/'DELIVERY_REVIEW'
    monkeypatch.setenv('EID_ISOLATED','1')
    monkeypatch.setenv('EID_PROJECT_ROOT',str(physical/'Privacy-Preserving-Shopping-Intelligence'))
    for base in [physical,virtual,Path('/home/jovyan/ahmed')]:
        assert normalize_isolated_path(base/'DELIVERY_REVIEW/evidence/E3', review=review)==review/'evidence/E3'
    assert normalize_isolated_path('/tmp/outside',review=review)==Path('/tmp/outside')
    assert not normalize_isolated_path(physical/'DELIVERY_REVIEW/../Privacy-Preserving-Shopping-Intelligence',
                                       review=review).is_relative_to(review)
