"""P2-01 (PREG2 §3.2 registered no-op definition) — tools/p2_noop_check.py (criterion (a) from checkpoints),
fl.update_log's (a) / (b) / classification functions and the pure parts of v4int/p2_probe.py (CPU, synthetic SASRec-like
checkpoints; no data, no evaluation value).

Proves: (a) rel on the trained subset T < 1 % AND no group with >= 4,096 elements moved >= 1 % (a small group may move);
T = RUN_SUMMARY update_norm_log.trained_keys when recorded, else the float keys minus the ARM_SPEC frozen tables;
frozen tables checked bit-identical; (b) top-20 Jaccard >= 0.95 and mean |Delta s(i*)| < the median top-1 / top-2 gap;
classification NO_OP / MOVED / DRIFT / INCOMPLETE at PRACTICAL_BEST + endpoint_6.0; the CLI writes once (O_EXCL),
refuses --calib-value without --pre-calib; the probe rows are a pure seeded function (4,096 sorted distinct positions,
seed 20261004), labels are stripped before scoring, chunked scoring equals the full-matrix criterion and draws no RNG.
NC (strict xfail): a criterion (a) that ignores the large-group rule calls a run with a 2 %-moved qkv block NO_OP.
"""
from __future__ import annotations

import importlib.util
import json
import math
from collections import OrderedDict
from pathlib import Path

import torch

import fl.update_log as U
import v4int.p2_probe as PR
from fl.checkpoint import atomic_save
from fl.numerics import state_digest
from fl.synthetic import make_clients, make_tiny_adapter
from intkit import assert_raises, nc

V4 = Path(__file__).resolve().parents[2]
FROZEN = ["item_embed.weight", "output_embed", "output_bias"]


def _tool():
    spec = importlib.util.spec_from_file_location("p2_noop_check", V4 / "tools" / "p2_noop_check.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def theta_pre() -> OrderedDict:
    g = torch.Generator().manual_seed(11)
    shapes = {"blocks.0.qkv.weight": (192, 64), "blocks.0.out.weight": (64, 64), "blocks.0.ff1.weight": (128, 64),
              "blocks.0.ln1.weight": (64,), "side_gate": (), "item_embed.weight": (100, 64), "output_embed": (50, 64),
              "output_bias": (50,)}
    st = OrderedDict((k, torch.randn(s, generator=g)) for k, s in shapes.items())
    st["product_idx_of_class"] = torch.arange(3, 53, dtype=torch.long)
    return st


def moved(pre, rel: dict, seed=5) -> OrderedDict:
    """theta_pre with group keys moved by the given relative L2 amount (exactly, along a random direction)."""
    g = torch.Generator().manual_seed(seed)
    out = OrderedDict((k, v.clone()) for k, v in pre.items())
    for k, r in rel.items():
        d = torch.randn(pre[k].shape, generator=g)
        out[k] = pre[k] + d * (r * pre[k].norm() / d.norm())
    return out


def make_run(root: Path, pre, best, end, *, trained_keys=None, name="run") -> Path:
    rd = root / name
    (rd / "ckpt" / "rid").mkdir(parents=True)
    spec = {"arm": "P2_W", "frozen_item_tables": FROZEN, "init_from": {"state_digest": state_digest(pre)}}
    (rd / "ARM_SPEC.json").write_text(json.dumps({"arm": "P2_W", "run_id": "rid", "arm_spec_sha256": "0" * 64,
                                                  "arm_spec": spec}), encoding="utf-8")
    for f, th in (("best.pt", best), ("endpoint_6.0.pt", end)):
        atomic_save({"kind": "weights", "run_id": "rid", "theta": dict(th)}, rd / "ckpt" / "rid" / f)
    if trained_keys is not None:
        (rd / "RUN_SUMMARY.json").write_text(json.dumps({"update_norm_log": {"trained_keys": trained_keys}}),
                                             encoding="utf-8")
    return rd


def pre_file(root: Path, pre) -> Path:
    p = root / "endpoint_6.0.pt"
    torch.save({"kind": "endpoint_6.0", "meta": {"state_digest": state_digest(pre)}, "model": dict(pre)}, p)
    return p


def b_rec(holds: bool) -> dict:
    return {"criterion": "b", "holds": holds, "mean_jaccard": 0.99 if holds else 0.5}


def probe(holds_pb=True, holds_cb=True) -> dict:
    return {"points": {"PRACTICAL_BEST": b_rec(holds_pb), "CONTROLLED_BUDGET": b_rec(holds_cb)}, "self_sha256": "p"}


# ------------------------------------------------------------------------------------------------ criterion (a)
def test_param_criterion_thresholds():
    pre = theta_pre()
    keys = [k for k in pre if k not in FROZEN and pre[k].is_floating_point()]
    a = U.param_noop_criterion(moved(pre, {"blocks.0.qkv.weight": 0.001}), pre, keys)
    assert a["holds"] and a["rel_T"] < 0.01 and a["n_keys_T"] == 5
    assert a["groups"]["block0.qkv"]["large"] and not a["groups"]["layernorm"]["large"]
    small = U.param_noop_criterion(moved(pre, {"blocks.0.ln1.weight": 0.05, "side_gate": 0.5}), pre, keys)
    assert small["holds"] and small["groups"]["layernorm"]["moved"] and not small["large_groups_moved"]
    big = U.param_noop_criterion(moved(pre, {"blocks.0.out.weight": 0.02}), pre, keys)
    assert not big["holds"] and big["large_groups_moved"] == ["block0.out"] and big["rel_T"] < 0.01
    tot = U.param_noop_criterion(moved(pre, {k: 0.015 for k in keys}), pre, keys)
    assert not tot["holds"] and math.isclose(tot["rel_T"], 0.015, rel_tol=1e-6)


# ------------------------------------------------------------------------------------------------ the tool
def test_noop_check_classifications(tmp_path):
    T = _tool()
    pre = theta_pre()
    pp = pre_file(tmp_path, pre)
    still = moved(pre, {"blocks.0.qkv.weight": 1e-4})
    big = moved(pre, {"blocks.0.qkv.weight": 0.02})
    cases = {"noop": (still, still, probe(), None, None, "NO_OP"),
             "moved_a": (still, big, probe(), None, None, "MOVED"),
             "moved_b": (still, still, probe(True, False), None, None, "MOVED"),
             "drift": (big, big, probe(), 0.10, 0.11, "DRIFT"),
             "drift_equal": (big, big, probe(), 0.11, 0.11, "DRIFT"),
             "moved_better": (big, big, probe(), 0.12, 0.11, "MOVED"),
             "incomplete": (still, still, None, None, None, "INCOMPLETE"),
             "moved_no_probe": (big, still, None, None, None, "MOVED")}
    for name, (b, e, pr, cv, pc, want) in cases.items():
        rd = make_run(tmp_path, pre, b, e, name=name)
        rec = T.check(rd, pp, probe=pr, calib_value=cv, pre_calib=pc)
        assert rec["classification"]["verdict"] == want, name
        assert rec["theta_pre_matches_init_from"] is True and all(rec["frozen_tables_bit_identical"].values())
        assert rec["trained_keys_source"].startswith("theta_pre float keys") and len(rec["trained_keys"]) == 5
        assert "product_idx_of_class" in rec["excluded_keys"] and set(FROZEN) <= set(rec["excluded_keys"])
    drift = T.check(tmp_path / "drift", pp, probe=probe(), calib_value=0.10, pre_calib=0.11)["classification"]
    assert drift["failed"] == ["PRACTICAL_BEST:a", "CONTROLLED_BUDGET:a"]
    # T from RUN_SUMMARY (a Phase-2 run records it); a frozen key there is refused
    rd = make_run(tmp_path, pre, still, still, trained_keys=["blocks.0.qkv.weight", "blocks.0.out.weight"], name="tk")
    rec = T.check(rd, pp, probe=probe())
    assert rec["trained_keys_source"].startswith("RUN_SUMMARY") and rec["points"]["PRACTICAL_BEST"]["a"]["n_keys_T"] == 2
    bad = make_run(tmp_path, pre, still, still, trained_keys=["output_embed"], name="tkbad")
    assert_raises(T.NoopCheckRefused, T.check, bad, pp)
    # a changed frozen table is reported
    tab = moved(still, {"output_embed": 0.1})
    rec = T.check(make_run(tmp_path, pre, tab, still, name="tab"), pp, probe=probe())
    assert rec["frozen_tables_bit_identical"]["output_embed"] is False


def test_noop_check_cli(tmp_path):
    T = _tool()
    pre = theta_pre()
    pp = pre_file(tmp_path, pre)
    rd = make_run(tmp_path, pre, pre, pre)
    pf = tmp_path / "P2_PROBE.json"
    pf.write_text(json.dumps(probe()), encoding="utf-8")
    out = tmp_path / "o" / "NOOP.json"
    args = ["--run-dir", str(rd), "--theta-pre", str(pp), "--out", str(out), "--probe", str(pf)]
    assert T.main(args) == 0
    rec = json.loads(out.read_text(encoding="utf-8"))
    assert rec["format"] == T.FORMAT and rec["classification"]["verdict"] == "NO_OP" and rec["probe_sha256"] == "p"
    try:
        T.main(args)
        raise AssertionError("an existing output was overwritten")
    except FileExistsError:
        pass
    assert T.main(["--run-dir", str(rd), "--theta-pre", str(pp), "--out", str(tmp_path / "x.json"),
                   "--calib-value", "0.1"]) == 2
    empty = tmp_path / "empty"
    (empty / "ckpt").mkdir(parents=True)
    (empty / "ARM_SPEC.json").write_text(json.dumps({"arm_spec": {}}), encoding="utf-8")
    assert T.main(["--run-dir", str(empty), "--theta-pre", str(pp), "--out", str(tmp_path / "y.json")]) == 2


# ------------------------------------------------------------------------------------------------ criterion (b) / probe
def test_functional_criterion_and_classify():
    g = torch.Generator().manual_seed(3)
    s = torch.randn(64, 200, generator=g)
    same = U.functional_noop_criterion(s.clone(), s)
    assert same["holds"] and same["mean_jaccard"] == 1.0 and same["mean_abs_dscore_top1"] == 0.0
    near = U.functional_noop_criterion(s + 1e-4 * torch.randn(64, 200, generator=g), s)
    assert near["holds"] and near["mean_jaccard"] > 0.95
    far = U.functional_noop_criterion(torch.randn(64, 200, generator=g), s)
    assert not far["holds"] and far["mean_jaccard"] < 0.5
    shift = U.functional_noop_criterion(s + 10.0, s)     # same ranking, every score moved: fails the |Delta s| part
    assert shift["mean_jaccard"] == 1.0 and not shift["holds"]
    assert_raises(ValueError, U.functional_noop_criterion, s, s[:, :10])
    a_ok = {"holds": True}
    assert U.classify_noop({"PRACTICAL_BEST": {"a": a_ok, "b": a_ok},
                            "CONTROLLED_BUDGET": {"a": a_ok, "b": a_ok}})["verdict"] == "NO_OP"
    assert_raises(ValueError, U.classify_noop, {"PRACTICAL_BEST": {"a": a_ok, "b": a_ok}})


def test_probe_positions_labels_and_chunked_scoring():
    p1, p2 = PR.probe_positions(50_000), PR.probe_positions(50_000)
    assert p1.dtype.kind == "i" and len(p1) == U.PROBE_ROWS == 4096 and (p1 == p2).all()
    assert (p1[1:] > p1[:-1]).all() and p1.min() >= 0 and p1.max() < 50_000
    assert (PR.probe_positions(50_000, seed=1) != p1).any() and U.PROBE_SEED == 20261004
    assert_raises(PR.ProbeRefused, PR.probe_positions, 4095)
    cl = make_clients(6, seed=2)
    batches = [PR.strip_labels(c.examples) for c in cl]
    assert all(not set(PR.LABEL_FIELDS) & set(b) for b in batches) and "item_tokens" in batches[0]
    a, b = make_tiny_adapter(seed=1, tied=False), make_tiny_adapter(seed=2, tied=False)
    rng0 = torch.get_rng_state().clone()
    rows = PR.score_chunks(a, b, batches, k=5)
    assert torch.equal(torch.get_rng_state(), rng0)                       # eval mode: no dropout draw
    with torch.no_grad():
        full = U.functional_rows(torch.cat([a.scores(x) for x in batches]), torch.cat([b.scores(x) for x in batches]), 5)
    assert all(torch.equal(rows[f], full[f]) for f in full)
    again = PR.score_chunks(a, b, batches, k=5)
    assert all(torch.equal(rows[f], again[f]) for f in rows)
    rec = U.functional_noop_from_rows(rows, 5)
    assert rec["rows"] == sum(int(x["item_tokens"].shape[0]) for x in batches) and rec["k"] == 5


@nc("criterion (a) without the >= 4,096-element group rule would call a 2 %-moved qkv block a no-op")
def test_nc_total_only_criterion():
    pre = theta_pre()
    keys = [k for k in pre if k not in FROZEN and pre[k].is_floating_point()]
    a = U.param_noop_criterion(moved(pre, {"blocks.0.out.weight": 0.02}), pre, keys)
    assert a["rel_T"] < U.NOOP_REL                         # the total-only reading ...
    assert a["holds"]                                      # ... is not the registered criterion
