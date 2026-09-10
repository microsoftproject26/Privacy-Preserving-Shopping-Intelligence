"""Focused tests for the S2-PR-01/03 runtime benchmark.

Everything here runs on toy inputs and is safe for CI. The 1k-5k matrix is never
launched from pytest; the one integration test uses a twelve-node toy population
so that the real strategy, the real client callback and the real oracle path are
all exercised without a large run.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import polars as pl
import psutil
import pytest
import torch

from ppsi.federated.runtime_monitor import (
    MonitorLimits,
    ProcessTreeMonitor,
    summarize_samples,
)
from ppsi.federated.sampling import sample_clients
from ppsi.training.fixtures import default_batch_spec
from ppsi.training.state import shared_state_digest
from scripts.federated.fl_real_smoke import build_smoke_model, server_round_to_sampler_round
from scripts.federated.fl_runtime_benchmark import (
    RuntimeFedAvg,
    build_runtime_client_app,
    capture_hardware,
    check_disk,
    choose_scale_concurrency,
    run_trial_child,
    trial_id,
)
from scripts.federated.fl_synthetic_smoke import SmokeValidationError

CATEGORY_COUNT = 588
SEED = 13
GIB = 1024**3


# ---------------------------------------------------------------------------
# Toy fixtures
# ---------------------------------------------------------------------------


def _toy_ids(count: int) -> list[str]:
    return [f"toy-client-{i:04d}" for i in range(count)]


@pytest.fixture(scope="module")
def toy_slices(tmp_path_factory: pytest.TempPathFactory) -> dict[str, object]:
    """A twelve-client toy population with exactly 32 TRAIN rows each."""
    tmp = tmp_path_factory.mktemp("runtime-toy")
    ids = _toy_ids(12)
    rows = [
        {
            "client_id": cid,
            "session": f"s{index}",
            "decision_order": order,
            "label_code": (index * 7 + order) % CATEGORY_COUNT,
        }
        for index, cid in enumerate(ids)
        for order in range(32)
    ]
    train_path = tmp / "train.parquet"
    pl.DataFrame(rows).write_parquet(train_path)

    val_rows = [
        {
            "client_id": ids[i % len(ids)],
            "session": "v",
            "decision_order": i,
            "label_code": i % CATEGORY_COUNT,
        }
        for i in range(64)
    ]
    val_path = tmp / "validation.parquet"
    pl.DataFrame(val_rows).write_parquet(val_path)

    spec = default_batch_spec()
    torch.manual_seed(SEED)
    model = build_smoke_model(CATEGORY_COUNT, spec)
    initial = {k: v.detach().clone() for k, v in model.state_dict().items()}
    from scripts.federated.fl_synthetic_smoke import get_digest

    return {
        "dir": tmp,
        "ids": ids,
        "train_path": str(train_path),
        "validation_path": str(val_path),
        "initial_digest": get_digest(initial),
    }


def _toy_workload(**overrides: object) -> dict[str, object]:
    base = {
        "seed": SEED,
        "torch_num_threads": 1,
        "clients_per_round": 4,
        "num_rounds": 3,
        "learning_rate": 0.02,
        "momentum": 0.0,
        "local_epochs": 1,
        "batch_size": 8,
        "examples_per_client": 32,
        "aggregation_atol": 1e-6,
        "warmup_rounds_excluded_from_timing_summary": [1],
    }
    base.update(overrides)
    return base


def _toy_samples(ids: list[str], rounds: int, participants: int) -> dict[str, object]:
    samples = {}
    for server_round in range(1, rounds + 1):
        index = server_round_to_sampler_round(server_round)
        result = sample_clients(ids, SEED, index, participants)
        samples[str(index)] = {
            "selected_client_ids": result.selected_client_ids,
            "selected_digest": result.selected_digest,
        }
    return samples


def _toy_spec(toy_slices: dict[str, object], **overrides: object) -> dict[str, object]:
    ids = list(toy_slices["ids"])
    workload = _toy_workload()
    spec = {
        "trial_id": "toy-n12-m4-c1-rep1",
        "workload": workload,
        "population_ids": ids,
        "samples": _toy_samples(ids, workload["num_rounds"], workload["clients_per_round"]),
        "concurrency": 1,
        "category_count": CATEGORY_COUNT,
        "train_slice_path": toy_slices["train_path"],
        "validation_slice_path": toy_slices["validation_path"],
        "expected_initial_digest": toy_slices["initial_digest"],
        "node_registration_timeout_seconds": 90,
        "round_timeout_seconds": 180.0,
        "output_path": str(Path(toy_slices["dir"]) / "out.json"),
    }
    spec.update(overrides)
    return spec


# ---------------------------------------------------------------------------
# Population, participants and concurrency stay three different numbers
# ---------------------------------------------------------------------------


def test_trial_id_encodes_all_three_quantities() -> None:
    assert trial_id(2500, 20, 2, 1) == "n2500-m20-c2-rep1"


def test_nested_population_prefixes_are_deterministic_and_not_deduplicated() -> None:
    pool = sorted(_toy_ids(5000))
    points = [1000, 2500, 5000]
    populations = {n: pool[:n] for n in points}
    for n in points:
        assert len(populations[n]) == n
        assert len(set(populations[n])) == n
    # Nested: every smaller point is a prefix of every larger one.
    assert populations[1000] == populations[2500][:1000]
    assert populations[2500] == populations[5000][:2500]


def test_sampler_round_mapping_is_zero_based() -> None:
    assert server_round_to_sampler_round(1) == 0
    assert server_round_to_sampler_round(6) == 5
    with pytest.raises(ValueError):
        server_round_to_sampler_round(0)


def test_selected_ids_are_exactly_the_frozen_sampler_choice() -> None:
    ids = _toy_ids(100)
    samples = _toy_samples(ids, 6, 20)
    for index in range(6):
        selected = samples[str(index)]["selected_client_ids"]
        assert len(selected) == 20
        assert len(set(selected)) == 20
        assert set(selected) <= set(ids)
        assert selected == sample_clients(ids, SEED, index, 20).selected_client_ids


def test_benchmark_filter_excludes_only_from_the_pool() -> None:
    counts = pl.DataFrame(
        {
            "client_id": _toy_ids(5),
            "t1_train_example_count": [5, 31, 32, 100, 33],
        }
    )
    eligible = sorted(counts.filter(pl.col("t1_train_example_count") >= 32)["client_id"].to_list())
    assert eligible == ["toy-client-0002", "toy-client-0003", "toy-client-0004"]
    # The source frame is untouched: nobody is removed from the cohort itself.
    assert counts.height == 5


# ---------------------------------------------------------------------------
# Concurrency admission and disk guard
# ---------------------------------------------------------------------------


def _scale_config() -> dict[str, object]:
    return json.loads(Path("config/federated/fl_scale_matrix.v1.json").read_text(encoding="utf-8"))


@pytest.mark.parametrize(
    "logical,total_gib,available_gib,expected",
    [
        (8, 32, 16, 2),
        (8, 16, 5, 1),
        (1, 8, 4, 1),
        (8, 4, 2, None),
        (8, 16, 1, None),
    ],
)
def test_concurrency_admission_is_frozen_before_any_run(
    logical: int, total_gib: float, available_gib: float, expected: int | None
) -> None:
    hardware = {
        "logical_cpus": logical,
        "total_ram_bytes": int(total_gib * GIB),
        "available_ram_bytes": int(available_gib * GIB),
    }
    decision = choose_scale_concurrency(hardware, _scale_config())
    assert decision["concurrency"] == expected
    assert decision["admissible"] is (expected is not None)


def test_disk_guard_blocks_a_full_volume() -> None:
    config = _scale_config()
    ok, _ = check_disk(
        {"repo_volume_free_bytes": 100 * GIB, "temp_volume_free_bytes": 100 * GIB}, config
    )
    assert ok is True
    ok, reason = check_disk(
        {"repo_volume_free_bytes": 1 * GIB, "temp_volume_free_bytes": 100 * GIB}, config
    )
    assert ok is False
    assert "repository" in reason


def test_captured_hardware_is_measured_not_assumed() -> None:
    hardware = capture_hardware()
    assert hardware["logical_cpus"] == psutil.cpu_count(logical=True)
    assert hardware["total_ram_bytes"] == psutil.virtual_memory().total
    # CPU-only means the GPU figure is not applicable, never a measured zero.
    assert hardware["gpu_memory"] == "NOT_APPLICABLE"


# ---------------------------------------------------------------------------
# Sampled memory: simultaneous sums, PID reuse, incomplete measurement
# ---------------------------------------------------------------------------


def test_monitor_includes_child_processes_in_the_tree_sum() -> None:
    child = subprocess.Popen(
        [sys.executable, "-c", "import time; x=bytearray(40_000_000); time.sleep(2.5)"],
        cwd=os.getcwd(),
        shell=False,
    )
    try:
        monitor = ProcessTreeMonitor(os.getpid(), poll_seconds=0.05)
        monitor.start()
        time.sleep(1.5)
        report = monitor.stop()
    finally:
        child.terminate()
        child.wait(timeout=10)
    assert report.sample_count > 0
    assert child.pid in report.observed_pids
    assert report.peak_tree_rss_bytes > psutil.Process(os.getpid()).memory_info().rss // 2


def test_peak_is_a_simultaneous_sum_not_a_sum_of_separate_peaks() -> None:
    # Two processes that never peak at the same instant.
    a = [10, 1, 1]
    b = [1, 1, 10]
    simultaneous_sums = [x + y for x, y in zip(a, b, strict=True)]
    assert max(simultaneous_sums) == 11
    # Summing individual maxima would overstate the peak.
    assert max(a) + max(b) == 20


def test_pid_reuse_is_not_followed() -> None:
    monitor = ProcessTreeMonitor(os.getpid(), poll_seconds=0.05)
    monitor._known[os.getpid()] = psutil.Process(os.getpid()).create_time() + 1234.0
    live, _ = monitor._live_processes()
    assert live == []


def test_incomplete_measurement_cannot_be_called_verified() -> None:
    monitor = ProcessTreeMonitor(os.getpid(), poll_seconds=0.05)
    monitor.start()
    time.sleep(0.3)
    report = monitor.stop()
    assert report.measurement_complete is True
    report.samples_with_inaccessible_processes = 1
    assert report.measurement_complete is False
    assert report.to_public_dict()["measurement_complete"] is False


def test_monitor_records_interval_and_gap() -> None:
    monitor = ProcessTreeMonitor(os.getpid(), poll_seconds=0.05)
    monitor.start()
    time.sleep(0.4)
    report = monitor.stop()
    public = report.to_public_dict()
    assert public["poll_interval_seconds"] == 0.05
    assert public["max_sample_gap_seconds"] is not None
    assert public["min_available_ram_bytes"] is not None
    assert "not unique physical RAM" in " ".join(public["limitations"])


def test_guard_floor_uses_the_larger_of_the_two_bounds() -> None:
    limits = MonitorLimits(min_available_ram_gib=2.0, min_available_ram_fraction_total=0.10)
    assert limits.available_floor_bytes(8 * GIB) == 2.0 * GIB
    assert limits.available_floor_bytes(64 * GIB) == 0.10 * 64 * GIB


def test_cleanup_only_touches_verified_owned_processes() -> None:
    monitor = ProcessTreeMonitor(os.getpid(), poll_seconds=0.05)
    # An unrelated PID with a mismatched creation time must never be terminated.
    monitor._known[os.getpid()] = psutil.Process(os.getpid()).create_time() + 999.0
    result = monitor.terminate_owned_tree(grace_seconds=0.1)
    assert result["owned_process_count"] == 0
    assert result["cleanup_complete"] is True


# ---------------------------------------------------------------------------
# Timing statistics
# ---------------------------------------------------------------------------


def test_p50_and_p95_use_the_declared_definitions() -> None:
    stats = summarize_samples([1.0, 2.0, 3.0, 4.0, 5.0])
    assert stats["n"] == 5
    assert stats["p50"] == 3.0
    # nearest rank: ceil(0.95 * 5) - 1 = 4 -> the largest of five samples
    assert stats["p95"] == 5.0
    ten = summarize_samples([float(i) for i in range(1, 11)])
    assert ten["p50"] == 5.5
    # ceil(0.95 * 10) - 1 = 9 -> the tenth sample
    assert ten["p95"] == 10.0


def test_empty_sample_summary_is_null_not_zero() -> None:
    stats = summarize_samples([])
    assert stats["n"] == 0
    assert stats["p50"] is None
    assert stats["p95"] is None


def test_warm_up_round_is_excluded_from_the_timing_summary() -> None:
    workload = _toy_workload()
    rounds = [
        {"server_round": 1, "round_wall_seconds": 100.0},
        {"server_round": 2, "round_wall_seconds": 1.0},
        {"server_round": 3, "round_wall_seconds": 2.0},
    ]
    excluded = set(workload["warmup_rounds_excluded_from_timing_summary"])
    warm = [r["round_wall_seconds"] for r in rounds if r["server_round"] not in excluded]
    stats = summarize_samples(warm)
    assert stats["n"] == 2
    assert stats["max"] == 2.0  # the 100s warm-up never enters the summary


def test_stage_timings_are_non_negative(toy_trial_output: dict[str, object]) -> None:
    for row in toy_trial_output["rounds"]:
        assert row["dispatch_to_replies_seconds"] >= 0
        assert row["aggregation_seconds"] >= 0
        assert row["server_evaluation_seconds"] >= 0
        assert row["round_wall_seconds"] >= 0
        assert row["client_local_fit_seconds"]["n"] == 4


# ---------------------------------------------------------------------------
# The toy integration trial: real strategy, real callback, real oracle
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def toy_trial_output(toy_slices: dict[str, object]) -> dict[str, object]:
    return run_trial_child(_toy_spec(toy_slices))


def test_population_is_n_and_participation_is_m(toy_trial_output: dict[str, object]) -> None:
    assert toy_trial_output["declared_population"] == 12
    assert toy_trial_output["participants_per_round"] == 4
    assert toy_trial_output["node_binding"]["registered_nodes"] == 12
    assert toy_trial_output["node_binding"]["map_frozen"] is True
    # Not everyone trains: 3 rounds of 4 clients is 12 fit calls over fewer unique users.
    assert toy_trial_output["total_fit_calls"] == 12
    assert toy_trial_output["unique_participating_clients"] <= 12


def test_every_round_contributes_the_expected_examples(toy_trial_output: dict[str, object]) -> None:
    for row in toy_trial_output["rounds"]:
        assert row["selected_client_count"] == 4
        assert row["contributing_examples"] == 4 * 32
        assert row["aggregation_oracle_pass"] is True
        assert row["max_abs_diff"] <= 1e-6


def test_server_evaluations_cover_round_zero_through_the_last(
    toy_trial_output: dict[str, object],
) -> None:
    rounds = [e["server_round"] for e in toy_trial_output["server_evaluations"]]
    assert rounds == [0, 1, 2, 3]
    for evaluation in toy_trial_output["server_evaluations"]:
        assert evaluation["support"] == 64
        assert 0.0 <= evaluation["accuracy_at_1"] <= 1.0
        assert evaluation["cross_entropy"] == pytest.approx(evaluation["cross_entropy"])


def test_the_global_model_actually_changed(toy_trial_output: dict[str, object]) -> None:
    assert toy_trial_output["initial_state_digest"] != toy_trial_output["final_state_digest"]


def test_a_wrong_initial_digest_stops_the_trial(toy_slices: dict[str, object]) -> None:
    with pytest.raises(SmokeValidationError):
        run_trial_child(_toy_spec(toy_slices, expected_initial_digest="0" * 64))


def test_a_population_smaller_than_the_node_map_is_rejected(
    toy_slices: dict[str, object],
) -> None:
    # Declaring 11 nodes while the simulation registers 12 must fail loudly.
    spec = _toy_spec(toy_slices)
    spec["population_ids"] = list(toy_slices["ids"])[:11]
    spec["samples"] = _toy_samples(spec["population_ids"], 3, 4)
    strategy = RuntimeFedAvg(
        population_ids=spec["population_ids"],
        node_registration_timeout=0.5,
        train_slice_path=spec["train_slice_path"],
        learning_rate=0.02,
        momentum=0.0,
        local_epochs=1,
        batch_size=8,
        examples_per_client=32,
        seed=SEED,
        expected_clients=4,
        aggregation_atol=1e-6,
        selected_ids_by_round={},
        selection_digests_by_round={},
        fraction_train=0.0,
        fraction_evaluate=0.0,
        min_train_nodes=4,
        min_evaluate_nodes=0,
        min_available_nodes=4,
        weighted_by_key="num-examples",
    )

    class _Grid:
        def get_node_ids(self):
            return [1, 2, 3]

    with pytest.raises(SmokeValidationError, match="registered SuperNodes"):
        strategy.bind_population_to_nodes(_Grid())


def test_each_message_carries_its_own_config_object(toy_slices: dict[str, object]) -> None:
    """A shared mutable config would give every client the last written id."""
    output = run_trial_child(_toy_spec(toy_slices))
    per_round = output["private_selected_ids"]
    for ids in per_round.values():
        assert len(set(ids)) == len(ids) == 4


def test_client_app_factory_captures_no_frames() -> None:
    app = build_runtime_client_app(CATEGORY_COUNT)
    closure = app.__dict__
    payload = json.dumps(str(closure))
    assert "DataFrame" not in payload


def test_reply_digest_uses_the_same_packing_function_the_client_used() -> None:
    spec = default_batch_spec()
    torch.manual_seed(SEED)
    model = build_smoke_model(CATEGORY_COUNT, spec)
    packed = {k: v.detach().clone() for k, v in model.state_dict().items()}
    assert shared_state_digest(packed) == shared_state_digest(dict(packed))


def test_public_trial_output_holds_no_raw_client_ids(toy_trial_output: dict[str, object]) -> None:
    public = {k: v for k, v in toy_trial_output.items() if k != "private_selected_ids"}
    text = json.dumps(public)
    for client_id in ("toy-client-0000", "toy-client-0005"):
        assert client_id not in text


# ---------------------------------------------------------------------------
# Targeted closeout regressions
# ---------------------------------------------------------------------------


def test_configured_round_timeout_reaches_the_production_start_call(
    toy_slices: dict[str, object], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The declared round wait must be passed, not left at the library default."""
    import scripts.federated.fl_runtime_benchmark as bench

    seen: dict[str, object] = {}
    original = bench.RuntimeFedAvg.start

    def _spy(self, *args, **kwargs):
        seen["timeout"] = kwargs.get("timeout")
        return original(self, *args, **kwargs)

    monkeypatch.setattr(bench.RuntimeFedAvg, "start", _spy, raising=False)
    spec = _toy_spec(toy_slices)
    spec["round_timeout_seconds"] = 123.0
    bench.run_trial_child(spec)
    assert seen["timeout"] == 123.0


def test_trial_spec_carries_the_configured_round_timeout() -> None:
    config = _scale_config()
    assert config["safety"]["round_timeout_seconds"] == 180


def test_measured_startup_field_names_the_interval_it_measures(
    toy_trial_output: dict[str, object],
) -> None:
    # The old name claimed whole-simulation startup; the measurement is narrower.
    assert "startup_seconds" not in toy_trial_output
    assert toy_trial_output["server_entry_to_node_binding_seconds"] >= 0


def test_unreadable_discovery_marks_the_measurement_incomplete(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """AccessDenied while discovering the tree must not report a verified profile."""
    monitor = ProcessTreeMonitor(os.getpid(), poll_seconds=0.05)
    monitor.start()
    time.sleep(0.2)
    report = monitor.stop()
    assert report.measurement_complete is True

    denied = ProcessTreeMonitor(os.getpid(), poll_seconds=0.05)

    def _deny(_pid):
        raise psutil.AccessDenied(_pid)

    monkeypatch.setattr(psutil, "Process", _deny)
    denied._discover()
    monkeypatch.undo()
    assert denied._discovery_inaccessible > 0
    blocked = denied.report()
    assert blocked.measurement_complete is False
    assert blocked.to_public_dict()["discovery_inaccessible_events"] > 0


def test_source_snapshot_pins_untracked_execution_code() -> None:
    from scripts.federated.fl_runtime_benchmark import build_source_snapshot

    snapshot = build_source_snapshot()
    assert snapshot["execution_provenance"] == "WORKING_TREE_BUNDLE_RUN"
    assert len(snapshot["git_head"]) == 40
    files = snapshot["files_sha256"]
    # The new untracked driver and monitor are pinned by content, not by commit.
    assert "scripts/federated/fl_runtime_benchmark.py" in files
    assert "ppsi/federated/runtime_monitor.py" in files
    # Inherited helpers actually used by a trial are pinned too.
    assert "ppsi/training/flower.py" in files
    assert all(len(v) == 64 for v in files.values())


def test_verify_fails_closed_when_a_required_deliverable_is_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An empty evidence tree must never report PASS."""
    import scripts.federated.fl_runtime_benchmark as bench

    monkeypatch.setattr(bench, "_REPO_ROOT", tmp_path)
    report = bench.stage_verify()
    assert report["status"] == "FAIL"
    assert any("required deliverable missing" in f for f in report["failures"])
    assert report["declared_trials"] == []


def test_retirement_never_removes_a_tracked_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import scripts.federated.fl_runtime_benchmark as bench

    monkeypatch.setattr(bench, "_REPO_ROOT", tmp_path)
    results = tmp_path / "artifacts" / "experiment-results"
    results.mkdir(parents=True)
    config_path = Path("cfg.json")
    (tmp_path / config_path).write_text(json.dumps({"a": 1}), encoding="utf-8")

    stale = results / "tracked.result.json"
    stale.write_text(
        json.dumps({"config_ref": {"uri": "cfg.json", "sha256": "0" * 64}}), encoding="utf-8"
    )
    monkeypatch.setattr(
        bench.subprocess,
        "run",
        lambda *a, **k: subprocess.CompletedProcess(
            a[0],
            0,
            stdout="artifacts/experiment-results/tracked.result.json",
            stderr="",
        ),
    )
    retired = bench._retire_superseded_results(config_path, keep=Path("other.json"))
    assert retired == []
    assert stale.is_file()  # a tracked result is never touched


def test_retirement_archives_an_untracked_superseded_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import scripts.federated.fl_runtime_benchmark as bench

    monkeypatch.setattr(bench, "_REPO_ROOT", tmp_path)
    results = tmp_path / "artifacts" / "experiment-results"
    results.mkdir(parents=True)
    config_path = Path("cfg.json")
    (tmp_path / config_path).write_text(json.dumps({"a": 1}), encoding="utf-8")
    stale = results / "old.result.json"
    stale.write_text(
        json.dumps({"config_ref": {"uri": "cfg.json", "sha256": "0" * 64}}), encoding="utf-8"
    )
    monkeypatch.setattr(
        bench.subprocess,
        "run",
        lambda *a, **k: subprocess.CompletedProcess(a[0], 0, stdout="", stderr=""),
    )
    retired = bench._retire_superseded_results(config_path, keep=Path("other.json"))
    assert retired == ["old.result.json"]
    assert not stale.exists()
    # Preserved with its metadata in the ignored private area, not erased.
    archived = (
        tmp_path / "artifacts" / "federated-runtime" / "superseded-results" / "old.result.json"
    )
    assert archived.is_file()
    assert json.loads(archived.read_text(encoding="utf-8"))["config_ref"]["sha256"] == "0" * 64
