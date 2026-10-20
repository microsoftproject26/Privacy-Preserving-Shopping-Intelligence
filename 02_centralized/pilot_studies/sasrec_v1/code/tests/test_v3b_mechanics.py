"""End-to-end mechanics of lock_retro_v3b eval (2 shards) + finalize on SYNTHETIC stores, CPU.

Exercises the real cmd_eval / cmd_finalize code paths (claim, sharding, F0/F4 store routing, the
mixed-ensemble F0/F4 equality check, build_model for singles / ensembles / weight averages, strict
FP32 evaluation, rank + manifest writing, finalize fingerprint checks and bootstraps) with every
real-data access monkeypatched to a SyntheticStore. VC/TEST are never touched.
"""
from __future__ import annotations

import json
import types
from pathlib import Path

import numpy as np
import torch

import lock_retro as base
import lock_retro_v3b as lr
import sasrec_data
from synthetic_store import SyntheticStore
from sasrec_model import SASRecConfig, SASRecCE


class _SplitAlias:
    """Serves the synthetic VALIDATION_SELECT population under any requested split name."""

    def __init__(self, inner):
        self.inner = inner

    def decisions(self, split, kind):
        return self.inner.decisions("VALIDATION_SELECT", kind)

    def batch(self, ds, positions):
        return self.inner.batch(ds, positions)


def test_v3b_eval_and_finalize_end_to_end(tmp_path, monkeypatch):
    K = 97
    stores = {ch: SyntheticStore.build(channels=ch, n_train=512, n_vs=600, K=K, seed=0, task="mapping") for ch in ("F0", "F4")}
    monkeypatch.setattr(sasrec_data, "build_eval_store", lambda ch, sp, lock_path=None: _SplitAlias(stores[ch]))
    import train as reuse_train
    monkeypatch.setattr(reuse_train, "VOCAB_SIZES", dict(stores["F4"].vocab_sizes))
    monkeypatch.setattr(reuse_train, "load_product_idx_of_class", lambda: stores["F4"].product_idx_of_class)
    for name in ("LOCK_PATH", "CLAIM_PATH", "DONE_PATH"):
        monkeypatch.setattr(base, name, tmp_path / "locks" / f"{name}.json")
    monkeypatch.setattr(base, "OUT", tmp_path / "retro")
    monkeypatch.setattr(base, "ARCHIVED", {})
    monkeypatch.setenv("SASREC_EVAL_DEVICE", "cpu")
    (tmp_path / "locks").mkdir()

    def entry(group, seed, fname):
        reg = {"id": "A1_RICH256" if group == "rich_frozen_channels" else "A0_ID256", "group": group, "d_model": 16,
               "blocks": 1, "heads": 2, "ff_multiplier": 4, "dropout": 0.0, "peak_lr": 1e-3, "label_smoothing": 0.0}
        m = SASRecCE(SASRecConfig.from_registry(reg), vocab_sizes=stores["F4"].vocab_sizes,
                     product_idx_of_class=stores["F4"].product_idx_of_class, seed=seed)
        p = tmp_path / fname
        torch.save({"state_dict": m.state_dict()}, p)
        return {"kind": "single", "group": group, "seed": seed, "registry_entry": reg, "weights": [str(p)],
                "weights_sha256": [lr.sha(p)]}

    e = {"A0_ID256_s2026": entry("item_only", 1, "a0.pt"), "A1_RICH256_s2026": entry("rich_frozen_channels", 2, "a1.pt"),
         "A1_RICH256_s2027": entry("rich_frozen_channels", 3, "a1b.pt")}
    e["S2"] = {"kind": "ensemble_logprob_mean", "members": ["A0_ID256_s2026", "A1_RICH256_s2026"]}
    e["S3"] = {"kind": "ensemble_logprob_mean", "members": ["A0_ID256_s2026", "A1_RICH256_s2026", "A1_RICH256_s2027"]}
    e["S4a"] = dict(e["A1_RICH256_s2026"], kind="weight_average",
                    weights=[e["A1_RICH256_s2026"]["weights"][0], e["A1_RICH256_s2027"]["weights"][0]],
                    weights_sha256=[e["A1_RICH256_s2026"]["weights_sha256"][0], e["A1_RICH256_s2027"]["weights_sha256"][0]])
    lock = {"entries": e, "headline": "A1_RICH256_s2026", "pair_check": {"comparable": True}}
    base.LOCK_PATH.write_text(json.dumps(lock))

    assert lr.cmd_eval(types.SimpleNamespace(worker=0, workers=2)) == 0
    assert lr.cmd_eval(types.SimpleNamespace(worker=1, workers=2)) == 0
    for sp in lr.SPLITS:
        for n in e:
            assert (base.OUT / sp / f"{n}.ranks.npz").exists(), f"{sp}/{n} not evaluated"
    # a single model evaluated here equals a direct evaluation (rank vector identical)
    from eval_full import evaluate_model
    import ensembles
    m = lr.build_model(e["A1_RICH256_s2026"], lock, torch.device("cpu"))
    ds = stores["F4"].decisions("VALIDATION_SELECT", "eval")
    direct = evaluate_model(m, stores["F4"], ds, torch.device("cpu"), eval_batch=128, k=20, return_ranks=True)["ranks"]
    got = np.load(base.OUT / "TEST" / "A1_RICH256_s2026.ranks.npz")["ranks"]
    assert np.array_equal(direct.astype(np.int32), got)
    # the S4a average really is the mean of its two members
    avg = ensembles.average_state_dicts(e["S4a"]["weights"])
    sd1, sd2 = (ensembles.load_state_dict_file(p) for p in e["S4a"]["weights"])
    assert torch.allclose(avg["output_embed"], (sd1["output_embed"] + sd2["output_embed"]) / 2)
    # re-running a worker is a no-op (immutable manifest), and a different lock refuses a second look
    assert lr.cmd_eval(types.SimpleNamespace(worker=0, workers=2)) == 0
    assert lr.cmd_finalize(types.SimpleNamespace()) == 0
    res = json.loads((base.OUT / "RETRO_RESULTS.json").read_text())
    assert set(res["models"]) >= {"A0_ID256_s2026", "A1_RICH256_s2026", "A1_RICH256_s2027"}
    assert set(res["systems"]) == {"S2", "S3", "S4a"}
    assert "A1_RICH256_s2026_minus_A0_ID256_s2026" in res["bootstraps"]["TEST"]
    assert "A1_RICH256_s2026_minus_GRU_F0_locked" in res["bootstraps_skipped"]      # no archived ranks in the test
    assert base.DONE_PATH.exists()
    base.DONE_PATH.unlink()
    base.LOCK_PATH.write_text(json.dumps(dict(lock, headline="changed")))
    import pytest
    with pytest.raises(SystemExit, match="second look"):
        lr.cmd_eval(types.SimpleNamespace(worker=0, workers=2))
