"""Package test list item 'pin_baseline' -- pin_baseline.py (owner: Opus): choose() tie rule,
noise_reference() (incl. the 0.002 fallback), and candidates() reading fabricated summary dicts
from a FO_CAMPAIGN_ROOT tmp dir. Never modifies the code under test.
"""
from __future__ import annotations

import json

import pytest

import conftest_fo  # noqa: F401
from conftest_fo import set_fo_campaign_root  # noqa: F401

import pin_baseline as pb


REG_ENTRY = {"id": "A1_RICH256", "group": "rich_frozen_channels", "d_model": 256, "blocks": 2,
             "heads": 4, "ff_multiplier": 4, "dropout": 0.2, "peak_lr": 1e-3, "label_smoothing": 0.0}


def _write_run(runs_dir, name, *, epochs: dict, complete=True):
    """`epochs`: {epoch:int -> macro_mrr20:float}. Writes summary.json + empty checkpoint files."""
    base = runs_dir / name
    (base / "checkpoints").mkdir(parents=True, exist_ok=True)
    records = []
    for ep, mrr in epochs.items():
        records.append({"epoch": ep, "full_vs": {"macro_mrr20": mrr, "micro_recall20": mrr + 0.1}})
        (base / "checkpoints" / f"epoch{ep}_weights.pt").write_bytes(b"fake")
    summary = {"final_status": "COMPLETE" if complete else "RUNNING", "epoch_records": records,
               "config": {"registry_entry": REG_ENTRY}, "registry_sha256": "deadbeef"}
    (base / "summary.json").write_text(json.dumps(summary))
    return base


# --------------------------------------------------------------------- choose() tie rule
def test_choose_highest_score_wins_outright():
    cands = [
        {"seed": 2027, "epoch": 4, "macro_mrr20": 0.5},
        {"seed": 2026, "epoch": 3, "macro_mrr20": 0.6},   # highest, must win despite later criteria
        {"seed": 2026, "epoch": 4, "macro_mrr20": 0.5},
    ]
    best = pb.choose(cands)
    assert best["macro_mrr20"] == 0.6 and best["seed"] == 2026 and best["epoch"] == 3


def test_choose_tie_breaks_by_earlier_epoch_then_seed_2026():
    cands = [
        {"seed": 2027, "epoch": 3, "macro_mrr20": 0.5},
        {"seed": 2026, "epoch": 4, "macro_mrr20": 0.5},
        {"seed": 2026, "epoch": 3, "macro_mrr20": 0.5},   # earliest epoch AND seed 2026 -> must win
    ]
    best = pb.choose(cands)
    assert best["seed"] == 2026 and best["epoch"] == 3


def test_choose_negative_control_seed_tiebreak_only_matters_after_epoch():
    # Negative control: swap which seed has the earlier epoch and confirm the winner follows the
    # epoch, not a fixed seed preference -- proving the ordering is (score, epoch, seed) not just
    # "seed 2026 always wins."
    cands = [
        {"seed": 2027, "epoch": 3, "macro_mrr20": 0.5},   # earlier epoch, off-seed -> must win here
        {"seed": 2026, "epoch": 4, "macro_mrr20": 0.5},
    ]
    best = pb.choose(cands)
    assert best["seed"] == 2027 and best["epoch"] == 3


# --------------------------------------------------------------------- noise_reference()
def test_noise_reference_matched_epochs_sigma_is_max_abs_diff():
    cands = [
        {"seed": 2026, "epoch": 3, "macro_mrr20": 0.550},
        {"seed": 2027, "epoch": 3, "macro_mrr20": 0.548},
        {"seed": 2026, "epoch": 4, "macro_mrr20": 0.560},
        {"seed": 2027, "epoch": 4, "macro_mrr20": 0.551},
    ]
    n = pb.noise_reference(cands)
    assert n["fallback_used"] is False
    assert n["sigma_hat"] == pytest.approx(max(abs(0.550 - 0.548), abs(0.560 - 0.551)), abs=1e-12)
    assert n["sigma_hat"] == pytest.approx(0.009, abs=1e-9)


def test_noise_reference_fallback_when_s2027_absent():
    cands = [{"seed": 2026, "epoch": 3, "macro_mrr20": 0.55}, {"seed": 2026, "epoch": 4, "macro_mrr20": 0.56}]
    n = pb.noise_reference(cands)
    assert n["fallback_used"] is True
    assert n["sigma_hat"] == pytest.approx(0.002, abs=1e-12)


# --------------------------------------------------------------------- candidates()
def test_candidates_reads_fabricated_summaries(tmp_path, set_fo_campaign_root):
    campaign = tmp_path / "campaign"
    runs = campaign / "runs"
    _write_run(runs, "A1_RICH256_s2026", epochs={3: 0.550, 4: 0.560})
    _write_run(runs, "A1_RICH256_s2027", epochs={3: 0.548, 4: 0.551})
    set_fo_campaign_root(campaign)
    import importlib
    import pin_baseline as pb2
    importlib.reload(pb2)

    cands = pb2.candidates()
    got = {(c["seed"], c["epoch"]): c["macro_mrr20"] for c in cands}
    assert got == {(2026, 3): 0.550, (2026, 4): 0.560, (2027, 3): 0.548, (2027, 4): 0.551}
    for c in cands:
        assert c["registry_entry"] == REG_ENTRY


def test_candidates_negative_control_incomplete_run_excluded(tmp_path, set_fo_campaign_root):
    campaign = tmp_path / "campaign"
    runs = campaign / "runs"
    _write_run(runs, "A1_RICH256_s2026", epochs={3: 0.550, 4: 0.560})
    _write_run(runs, "A1_RICH256_s2027", epochs={3: 0.548, 4: 0.551}, complete=False)   # RUNNING, not COMPLETE
    set_fo_campaign_root(campaign)
    import importlib
    import pin_baseline as pb2
    importlib.reload(pb2)

    cands = pb2.candidates()
    seeds = {c["seed"] for c in cands}
    assert seeds == {2026}, "an incomplete (non-COMPLETE) run must never produce a candidate"


def test_candidates_negative_control_missing_weights_file_excluded(tmp_path, set_fo_campaign_root):
    campaign = tmp_path / "campaign"
    runs = campaign / "runs"
    base = _write_run(runs, "A1_RICH256_s2026", epochs={3: 0.550, 4: 0.560})
    (base / "checkpoints" / "epoch4_weights.pt").unlink()   # summary claims epoch4 but the file is gone
    set_fo_campaign_root(campaign)
    import importlib
    import pin_baseline as pb2
    importlib.reload(pb2)

    cands = pb2.candidates()
    epochs = {c["epoch"] for c in cands if c["seed"] == 2026}
    assert epochs == {3}, "a candidate whose weights file does not exist must be excluded"
