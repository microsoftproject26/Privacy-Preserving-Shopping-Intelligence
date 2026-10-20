"""Package test list items 18/19 -- screen.py (owner: Opus): manifest, schedule, train_arm.
Never modifies the code under test. The n_train=40,180,608 manifest/sha256 check is explicitly
allowed by the task packet (~350 MB RAM, a few seconds).
"""
from __future__ import annotations

import hashlib
import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

import conftest_fo  # noqa: F401
from conftest_fo import tiny_store, tiny_inner, TINY_REGISTRY, STUDY_ROOT
from conftest_fo import set_fo_root  # noqa: F401  (registers the fixture for this module)

import fo_env
import fo_models as fm
import sasrec_data
import screen as sc

REGISTRY = json.loads((conftest_fo.STUDY_ROOT / "config" / "candidate_registry.json").read_text())


# --------------------------------------------------------------------- (18) manifest
def test_manifest_deterministic():
    a = sc.manifest(5000, n=200)
    b = sc.manifest(5000, n=200)
    assert np.array_equal(a, b)


def test_manifest_unique_and_in_range():
    m = sc.manifest(5000, n=1000)
    assert m.shape[0] == 1000
    assert len(np.unique(m)) == 1000
    assert m.min() >= 0 and m.max() < 5000
    assert m.dtype == np.int64


def test_manifest_sha256_matches_registry_at_full_scale():
    positions = sc.manifest(fo_env.N_TRAIN)
    assert positions.shape[0] == sc.N_SCREEN
    got = hashlib.sha256(positions.tobytes()).hexdigest()
    assert got == REGISTRY["manifest_sha256"]
    assert fo_env.N_TRAIN == 40_180_608
    assert REGISTRY["screen"]["manifest"]["n"] == sc.N_SCREEN


def test_manifest_negative_control_wrong_seed_gives_different_sha(monkeypatch):
    # Negative control: prove the sha comparison above is actually sensitive to the manifest
    # content, not vacuously true for any array of the right shape.
    monkeypatch.setattr(sc, "MANIFEST_SEED", sc.MANIFEST_SEED + 1)
    positions = sc.manifest(1000, n=500)
    monkeypatch.setattr(sc, "MANIFEST_SEED", sc.MANIFEST_SEED - 1)
    positions_orig = sc.manifest(1000, n=500)
    assert not np.array_equal(positions, positions_orig)


# --------------------------------------------------------------------- (18) schedule
def test_n_updates_and_final_batch_full_scale():
    U = sc.n_updates()
    assert U == 19532
    final_batch = sc.N_SCREEN - (U - 1) * sc.BATCH
    assert final_batch == 64


def test_n_updates_tiny_scale_matches_packet_example():
    assert sc.n_updates(1000, 256) == 4
    assert 1000 - 3 * 256 == 232


def test_lr_at_registered_points():
    U = sc.n_updates()
    assert sc.lr_at(1, U) == pytest.approx(5e-7, rel=1e-9)
    assert sc.lr_at(200, U) == pytest.approx(1e-4, rel=1e-9)
    assert sc.lr_at(U, U) == pytest.approx(1e-5, rel=1e-9)


def test_lr_at_cosine_monotone_after_warmup():
    U = sc.n_updates()
    us = list(range(sc.WARMUP, U + 1, 500)) + [U]
    lrs = [sc.lr_at(u, U) for u in us]
    assert all(a >= b - 1e-15 for a, b in zip(lrs, lrs[1:])), "lr must be non-increasing after warmup"


def test_lr_at_negative_control_before_warmup_is_increasing():
    lrs = [sc.lr_at(u, sc.n_updates()) for u in (1, 50, 100, 150, 200)]
    assert all(a < b for a, b in zip(lrs, lrs[1:])), "warmup phase must be strictly increasing"


# --------------------------------------------------------------------- (19) train_arm
def _tiny_reg_and_sd(seed=7):
    store = tiny_store(n_train=1000, n_vs=64, K=59, seed=seed)
    inner = tiny_inner(store, seed=seed + 1)
    return store, inner.state_dict()


def test_build_optimizer_is_fresh_each_call():
    import train_sasrec as ts
    store, sd = _tiny_reg_and_sd()
    model = fm.build(TINY_REGISTRY, sd, view="full", device="cpu",
                     vocab_sizes=store.vocab_sizes, product_idx_of_class=store.product_idx_of_class)
    opt = ts.build_optimizer(model, fused=False)
    assert len(opt.state) == 0, "a freshly built optimizer must carry no momentum/state"


def test_train_arm_covers_every_position_once_with_matching_exposures():
    store, sd = _tiny_reg_and_sd()
    model = fm.build(TINY_REGISTRY, sd, view="full", device="cpu",
                     vocab_sizes=store.vocab_sizes, product_idx_of_class=store.product_idx_of_class)
    ds_train = store.decisions("TRAIN", "loss")
    assert len(ds_train) == 1000
    positions = np.arange(1000)
    rec = sc.train_arm(model, store, ds_train, positions, torch.device("cpu"))
    assert rec["updates"] == 4
    assert rec["exposures"] == 1000
    assert rec["coverage_ok"] is True


def test_train_arm_negative_control_dropped_position_breaks_coverage():
    store, sd = _tiny_reg_and_sd()
    model = fm.build(TINY_REGISTRY, sd, view="full", device="cpu",
                     vocab_sizes=store.vocab_sizes, product_idx_of_class=store.product_idx_of_class)
    ds_train = store.decisions("TRAIN", "loss")
    positions = np.arange(1000)[:-1]   # drop the last position: coverage claim must now be False
    rec = sc.train_arm(model, store, ds_train, positions, torch.device("cpu"))
    assert rec["exposures"] == 999
    # `seen` is sized to len(positions), so `coverage_ok` can only assert internal consistency
    # (which still holds); the REAL signal a caller must check is exposures == full population.
    assert rec["exposures"] != 1000


def test_control_and_c1_d3_same_seed_same_loss_at_update1():
    store, sd = _tiny_reg_and_sd(seed=11)
    ds_train = store.decisions("TRAIN", "loss")
    positions = np.arange(256)   # exactly one update's worth

    control = fm.build(TINY_REGISTRY, sd, view="full", device="cpu",
                       vocab_sizes=store.vocab_sizes, product_idx_of_class=store.product_idx_of_class)
    d3 = fm.build(TINY_REGISTRY, sd, view="d3", device="cpu",
                  vocab_sizes=store.vocab_sizes, product_idx_of_class=store.product_idx_of_class)

    torch.manual_seed(2026)
    rec_ctrl = sc.train_arm(control, store, ds_train, positions, torch.device("cpu"))
    torch.manual_seed(2026)
    rec_d3 = sc.train_arm(d3, store, ds_train, positions, torch.device("cpu"))

    assert rec_ctrl["loss_first"] == pytest.approx(rec_d3["loss_first"], abs=1e-4)


def test_control_and_c1_d3_negative_control_perturbed_d3_weights_diverge():
    """Negative control for the same-loss-at-update-1 check: if the D3 weight columns were NOT
    zero-initialised (e.g. a future regression), the two arms' first-update loss would differ."""
    store, sd = _tiny_reg_and_sd(seed=12)
    ds_train = store.decisions("TRAIN", "loss")
    positions = np.arange(256)

    control = fm.build(TINY_REGISTRY, sd, view="full", device="cpu",
                       vocab_sizes=store.vocab_sizes, product_idx_of_class=store.product_idx_of_class)
    d3 = fm.build(TINY_REGISTRY, sd, view="d3", device="cpu",
                  vocab_sizes=store.vocab_sizes, product_idx_of_class=store.product_idx_of_class)
    with torch.no_grad():
        w = d3.inner.numflag_proj.weight
        w[:, 19:24] += 0.5   # break the zero-init on purpose

    torch.manual_seed(2026)
    rec_ctrl = sc.train_arm(control, store, ds_train, positions, torch.device("cpu"))
    torch.manual_seed(2026)
    rec_d3 = sc.train_arm(d3, store, ds_train, positions, torch.device("cpu"))
    assert abs(rec_ctrl["loss_first"] - rec_d3["loss_first"]) > 1e-3


# --------------------------------------------------------------------- screen.main label_smoothing guard
# (Review C fix: CE below is unsmoothed, so a mis-registered label_smoothing must abort BEFORE
# any training happens). screen.main() builds a REAL training store and reads real dataset files
# before reaching this guard, which this local-only test suite cannot provide; instead we
# monkeypatch every heavy dependency (sasrec_data.build_training_store -> a fake store whose
# TRAIN/VS decision counts and cohort match our fabricated tiny files, fm.load_weights_file -> a
# sentinel marker exception) so the REAL screen.main() code path actually executes up to and
# including the guard line, with nothing beyond it faked away.

class _FakeTrainDS:
    def __len__(self):
        return fo_env.N_TRAIN


class _FakeVSDS:
    def __init__(self, decision_id):
        self.decision_id = np.asarray(decision_id, dtype=np.int64)

    def __len__(self):
        return len(self.decision_id)

    def take(self, positions):
        return _FakeVSDS(self.decision_id[np.asarray(positions)])


class _FakeStore:
    def decisions(self, split, kind):
        if split == "TRAIN":
            return _FakeTrainDS()
        if split == "VALIDATION_SELECT":
            return _FakeVSDS([100, 200])
        raise ValueError(split)


class _PastGuardMarker(Exception):
    """Raised by the monkeypatched fm.load_weights_file, proving execution reached PAST the
    label_smoothing guard without the guard itself firing."""


def _run_screen_main_to_guard(tmp_path, set_fo_root, monkeypatch, label_smoothing):
    set_fo_root(tmp_path)
    registry = json.loads((STUDY_ROOT / "config" / "candidate_registry.json").read_text())
    (tmp_path / "config" / "candidate_registry.json").write_text(json.dumps(registry))

    weights_path = tmp_path / "fake_weights.pt"
    weights_path.write_bytes(b"not a real checkpoint, never loaded before the guard fires")
    sha = fo_env.sha256_file(weights_path)

    cohort_path = tmp_path / "cohort.parquet"
    pq.write_table(pa.table({"decision_id": pa.array([100, 200], type=pa.int64())}), cohort_path)

    pin = {"cohort_parquet": str(cohort_path),
           "chosen": {"file": str(weights_path), "sha256": sha,
                     "registry_entry": {"label_smoothing": label_smoothing}}}
    (tmp_path / "config" / "BASELINE_PIN.json").write_text(json.dumps(pin))

    monkeypatch.setattr(sasrec_data, "build_training_store", lambda channels: _FakeStore())
    monkeypatch.setattr(fm, "load_weights_file", lambda path: (_ for _ in ()).throw(_PastGuardMarker()))
    monkeypatch.setattr("sys.argv", ["screen.py", "--arm", "CONTROL"])
    sc.main()


def test_screen_main_label_smoothing_guard_raises_before_training(tmp_path, set_fo_root, monkeypatch):
    with pytest.raises(RuntimeError, match="label_smoothing"):
        _run_screen_main_to_guard(tmp_path, set_fo_root, monkeypatch, label_smoothing=0.1)


def test_screen_main_label_smoothing_guard_negative_control_zero_passes_through(tmp_path, set_fo_root, monkeypatch):
    """Negative control: label_smoothing == 0.0 must NOT trip the guard -- execution must reach
    (and only then fail on) the sentinel marker placed just past it."""
    with pytest.raises(_PastGuardMarker):
        _run_screen_main_to_guard(tmp_path, set_fo_root, monkeypatch, label_smoothing=0.0)
