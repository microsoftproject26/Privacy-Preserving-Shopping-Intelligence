"""T30 / T31 — LO sends no messages and gets no server refresh; theta0 provenance is checked; LO replay is
checksum-repeatable; the 1,024-client determinism receipt reproduces exactly in a fresh interpreter.

Proves: LO mechanics (single theta0 load, persistent AdamW, no channel, procedural replay determinism on CPU).
Does not prove: GPU replay determinism (needs its own receipt on the server: PENDING) or LO quality.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import torch

import fl.local_only as flo
from fl.client import LocalSolver
from fl.comm import LEDGER, VirtualChannel
from fl.local_only import LORecipe, ProvenanceError, run_local_only
from fl.numerics import state_digest
from fl_testkit import assert_raises, clients, nc, one_client, tiny

RECEIPTS = Path(__file__).resolve().parent / "receipts"
CODE = Path(__file__).resolve().parents[1]
SEED = 2026


def _recipe(theta0, passes=6, eval_at=(2, 4, 6)):
    return LORecipe(theta0_digest=state_digest(theta0), seed=SEED, passes=passes, eval_at=eval_at,
                    solver=LocalSolver(lr=0.02, passes=passes, batch_size=16, clip=1.0, weight_decay=1e-5))


def _hook(adapter, budget):
    b = one_client("user-q", 5, seed=77).examples
    return {"ce": float(torch.nn.functional.cross_entropy(adapter.scores(b), b["target_class"]))}


def lo_correct(adapter, theta0, c, recipe):
    return run_local_only(adapter, theta0, c, recipe, eval_hook=_hook)


def lo_with_refresh(adapter, theta0, c, recipe):
    """WRONG: an 'LO' that downloads the server state again at every evaluation budget."""
    ch = VirtualChannel(adapter.manifest)

    def hook(a, budget):
        ch.download(0, c.key)
        a.load_state_(theta0)
        return _hook(a, budget)
    return run_local_only(adapter, theta0, c, recipe, eval_hook=hook)


def check_no_messages(lo_fn):
    a = tiny(3, tied=False)
    theta0 = a.broadcast_state(clone=True)
    recipe = _recipe(theta0)
    loads = []
    orig = a.load_state_
    a.load_state_ = lambda s: (loads.append(1), orig(s))
    before = LEDGER.snapshot()
    cs = clients(6, seed=71, sizes=[3, 17, 40, 1, 9, 26])
    theta0_shared = state_digest({k: theta0[k] for k in a.manifest.shared_keys})
    for c in cs:
        r = lo_fn(a, theta0, c, recipe)
        assert (r.final_digest != theta0_shared) == (r.n_valid > 0), "LO must train exactly the labelled clients"
    assert LEDGER.snapshot() == before, "LO must not send or receive any message"
    assert len(loads) == len(cs), "LO loads theta0 exactly once per client and never refreshes"


def check_replay(n=24):
    a = tiny(4, tied=False)
    theta0 = a.broadcast_state(clone=True)
    recipe = _recipe(theta0)
    cs = clients(n, seed=72)
    first = {c.key: lo_correct(a, theta0, c, recipe).record() for c in cs}
    other = tiny(44, tied=False)                     # a differently initialized worker, clients in reverse order
    second = {c.key: lo_correct(other, theta0, c, recipe).record() for c in reversed(cs)}
    assert first == second, "LO replay must be checksum-identical"


# ------------------------------------------------------------------------------------------------ tests
def test_t30_lo_sends_no_messages():
    check_no_messages(lo_correct)


def test_t30_lo_theta0_provenance_checked():
    a = tiny(3, tied=False)
    theta0 = a.broadcast_state(clone=True)
    recipe = _recipe(theta0)
    tampered = {k: v.clone() for k, v in theta0.items()}
    tampered["enc.bias"].add_(1e-3)
    assert_raises(ProvenanceError, run_local_only, a, tampered, one_client("user-a", 10), recipe)


def test_t30_lo_persistent_optimizer_and_budgets():
    a = tiny(5, tied=False)
    theta0 = a.broadcast_state(clone=True)
    made = []
    orig = flo.make_optimizer
    flo.make_optimizer = lambda *x, **k: made.append(1) or orig(*x, **k)
    try:
        r = run_local_only(a, theta0, one_client("user-a", 40), _recipe(theta0), eval_hook=_hook)
    finally:
        flo.make_optimizer = orig
    assert len(made) == 1, "one persistent AdamW for the whole LO trajectory"
    assert r.steps == 6 * 3 and r.n_consumed == 6 * 40 and sorted(r.evals) == [2, 4, 6]
    assert r.evals[2] != r.evals[6]


def test_t30_lo_no_label_client_uses_theta0():
    a = tiny(5, tied=False)
    theta0 = a.broadcast_state(clone=True)
    recipe = _recipe(theta0)
    r = run_local_only(a, theta0, one_client("user-empty", 7, invalid_frac=1.0), recipe, eval_hook=_hook)
    assert r.no_label and r.steps == 0 and r.n_consumed == 0
    assert r.final_digest == state_digest({k: theta0[k] for k in a.manifest.shared_keys})
    assert r.evals[2] == r.evals[4] == r.evals[6]


def test_t31_lo_replay_checksum_repeatable():
    check_replay()


def test_t31_lo_determinism_receipt_1024_fresh_interpreter():
    from fl.lo_determinism import run_receipt
    RECEIPTS.mkdir(exist_ok=True)
    a = run_receipt(1024, SEED, "forward", threads=1)
    out_b = RECEIPTS / "LO_DETERMINISM_1024_reverse_subprocess.json"
    env = dict(os.environ, PYTHONHASHSEED="12345", PYTHONDONTWRITEBYTECODE="1", PYTHONPATH=str(CODE))
    p = subprocess.run([sys.executable, "-B", "-m", "fl.lo_determinism", "--n", "1024", "--seed", str(SEED),
                        "--order", "reverse", "--threads", "1", "--out", str(out_b)],
                       cwd=str(CODE), env=env, capture_output=True, text=True, timeout=900)
    assert p.returncode == 0, p.stderr[-2000:]
    b = json.loads(out_b.read_text())
    mism = [k for k in a["per_client"] if a["per_client"][k] != b["per_client"].get(k)]
    receipt = {"what": "LO determinism receipt: 1,024 synthetic clients, run A in-process (forward order) vs run B "
                       "in a fresh interpreter (PYTHONHASHSEED=12345, reverse order, same reused-worker procedure)",
               "tolerance": "exact (bitwise final-state sha256 and bit-identical eval floats)",
               "n_clients": a["n_clients"], "n_no_label": a["n_no_label"], "steps_total": a["steps_total"],
               "recipe_digest": a["recipe_digest"], "theta0_digest": a["theta0_digest"],
               "combined_digest_A": a["combined_digest"], "combined_digest_B": b["combined_digest"],
               "mismatched_clients": len(mism), "numerics": a["numerics"], "wall_s_A": a["wall_s"],
               "wall_s_B": b["wall_s"], "subprocess_exit_code": p.returncode,
               "verdict": "REPRODUCED_EXACTLY" if not mism and a["combined_digest"] == b["combined_digest"] else "MISMATCH"}
    (RECEIPTS / "LO_DETERMINISM_1024.json").write_text(json.dumps(receipt, indent=1))
    assert not mism and a["combined_digest"] == b["combined_digest"], f"{len(mism)} clients differ"
    assert a["n_clients"] == 1024 and a["n_no_label"] > 0


@nc("LO with a server refresh at each evaluation budget")
def test_nc_t30_lo_refresh():
    check_no_messages(lo_with_refresh)


@nc("LO replay with an unseeded (entropy) RNG")
def test_nc_t31_unseeded_replay(monkeypatch):
    orig = flo.derive_seed
    monkeypatch.setattr(flo, "derive_seed", lambda *p: orig(*p, int.from_bytes(os.urandom(4), "big")))
    check_replay(n=6)
