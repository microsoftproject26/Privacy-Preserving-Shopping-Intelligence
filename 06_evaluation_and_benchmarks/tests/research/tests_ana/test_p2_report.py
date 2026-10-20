"""P2-12: tools/p2_report.py + tools/p2_block_index.py on synthetic Phase-2 blocks only (no server, no real data, no
TEST file; tiny K, 48 MAIN users). Known answers: P2_WSTAR_SECAGG = P2_WSTAR s2026 ranks (Q13 = 100 exactly, TOST
equivalent), P2_WSTAR_DP8 = P2_WSTAR_FA1024 ranks (Q6 = 100 exactly), every Q1 / Q2 per-seed point = the difference of
the macro-E2E MRR@20 recomputed here from the rank files, Q10_s = Q1_s / Q2_s."""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import anakit
import layoutkit
import retention_report as R

TOOLS = R.V4 / "tools"
CODE = R.V4 / "code"
RR_FILE, ABI_FILE = TOOLS / "retention_report.py", TOOLS / "ana_block_index.py"
FROZEN = {"rr": "9a3b2c2a918ef2baacc5299426941f1eb94c366f5bb6aa7c29b4bfdbca0afe58",
          "abi": "6e7043bbe42967b0de313206496de38fb0cf20cc85aaa34945dfdd19863cb662"}


def _sha(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


PRE_IMPORT = {"rr": _sha(RR_FILE), "abi": _sha(ABI_FILE)}
import p2_report as P2  # noqa: E402
import p2_block_index as PBI  # noqa: E402
POST_IMPORT = {"rr": _sha(RR_FILE), "abi": _sha(ABI_FILE)}

CONTRASTS = R.V4 / "company" / "gates" / "P2_CONTRASTS_v3.json"
PB, CB = "PRACTICAL_BEST", "CONTROLLED_BUDGET"
TAG = {PB: "PB", CB: "CB"}
ADD = {"PRE": 4, "P2_WSTAR": 2, "FA_WARM_FROZEN": 3, "P2_C1_FT_CENTRAL_FULL": 1, "P2_C1_FT_CENTRAL_FROZEN": 2,
       "FT_PRE": 2, "FT_P2_WSTAR": 1, "P2_WSTAR_FA1024": 2, "P2_WSTAR_DP8": 2, "P2_WSTAR_DP1": 3,
       "P2_C2_CFULL_SMALL": 3, "P2_WSTAR_DPFTRL8": 3, "P2_WSTAR_SECAGG": 2}
DEVICE = ("FT_PRE", "FT_P2_WSTAR")
C2_LABELS = {"central_source": "P2_C2_CFULL_SMALL", "source_train_manifest_id": "FULL_TRAIN_5_100_TIEBREAK",
             "train_manifest_id": "FULL_TRAIN_5_100_TIEBREAK"}
SINGLE = tuple(P2.P2_SINGLE_SEED) + ("PRE", "C_FULL")


def seeds_of(arm):
    return (2026,) if arm in SINGLE else (2026, 2027)


def rules_of(arm):
    return (PB,) if arm in DEVICE else (PB, CB)


def ranks(arm, seed, rule, base, rankable):
    n = base.size
    if arm == "C_FULL":
        r = np.maximum(1, base - 1)
    else:
        r = base + ADD[arm]
    if arm != "PRE":
        if seed == 2027:
            r = np.where(anakit._mask(0.2, 13, n), r + 1, r)
        if rule == CB:
            r = np.where(anakit._mask(0.1, 17, n), r + 1, r)
    r = r.astype(np.int64)
    r[~rankable] = -1
    return r


def ckpt(arm, seed, rule):
    return layoutkit.ckpt_sha(arm, seed, rule)


def entry_dir(arm, seed, rule):
    return f"DEV/{arm}_s{seed}" if arm in DEVICE else f"REEVAL/{arm}_s{seed}_{TAG[rule]}"


def build_world(tmp: Path, split: str = "TEST", gate=None, *, drop=(), row_extra=None) -> SimpleNamespace:
    """A Phase-2 block (study PREG2, role MAIN) with every P2 arm and the four reused Phase-1 arms."""
    root = tmp / "block"
    root.mkdir(parents=True)
    m = anakit.make_manifest("MAIN", split, gate)
    base = np.random.default_rng(501).integers(1, 9, m.n).astype(np.int64)
    rsplit = "RETRO_TEST" if split == "TEST" else "VALIDATION"
    hexes = np.asarray([anakit.hexkey("MAIN", k) for k in range(anakit.ROLES["MAIN"][2])], dtype=object)
    ctx = {"MAIN": R.RoleCtx("MAIN", m, hexes, None, {"eval_manifest_id": m.manifest_id, "synthetic": True})}
    entries, rk, files = [], {}, {}
    for arm in P2.P2_BLOCK_ARMS:
        if arm in drop:
            continue
        for seed in seeds_of(arm):
            dev_rows = []
            for rule in rules_of(arm):
                r = ranks(arm, seed, PB if arm == "PRE" else rule, base, m.rankable)
                rk[(arm, seed, rule)] = r
                d = entry_dir(arm, seed, rule)
                extra = {"study": "PREG2"} if arm in P2.P2_ARMS else {}
                if arm == "P2_C2_CFULL_SMALL":                   # the central-source provenance reeval_p2 writes
                    extra.update(C2_LABELS)
                extra.update((row_extra or {}).get(arm, {}))
                if arm in DEVICE:
                    rel_rk = f"{d}/cells/{anakit.DEVICE_CELL}.ranks_int32.npy"
                    rows = anakit.rows_for(arm, seed, rule, r, m, carry_rule=False, cid=None, split=rsplit,
                                           extra=dict(extra, device_cell=anakit.DEVICE_CELL, device_selected=True,
                                                      efe=None))
                    dev_rows += rows
                else:
                    rel_rk = f"{d}/evals/MAIN.ranks_int32.npy"
                    kw = {"cid": "PRE_s2026:endpoint_6.0", "efe": 6.0} if arm == "PRE" else {}
                    rows = anakit.rows_for(arm, seed, rule, r, m, split=rsplit, extra=extra, **kw)
                    anakit.jdump(root / d / "RESULT_ROWS.json", {"rows": rows})
                (root / rel_rk).parent.mkdir(parents=True, exist_ok=True)
                np.save(root / rel_rk, r.astype(np.int32))
                e = {"arm": arm, "seed": seed, "rule": rule, "role": "MAIN", "rows": f"{d}/RESULT_ROWS.json",
                     "ranks": rel_rk}
                if arm in P2.PHASE1_REUSED_ARMS:
                    e["phase1_reused"] = True
                if arm in DEVICE:
                    e["row_filter"] = {"device_cell": anakit.DEVICE_CELL}
                if arm == "PRE":
                    e.update(pre_init=dict(anakit.PRE_INIT), checkpoint_sha256=anakit.INIT_FROM["checkpoint_sha256"])
                elif arm == "FT_PRE":
                    e.update(start_run_id=anakit.PRE_RUN, start_init=dict(anakit.INIT_FROM),
                             start_init_record_sha256=anakit.PRE_INIT_RECORD_SHA)
                elif arm == "FT_P2_WSTAR":
                    e.update(start_run_id=f"P2_W_MAIN_s{seed}",
                             p2_base={"arm": "P2_WSTAR", "seed": seed, "checkpoint": "best.pt",
                                      "checkpoint_sha256": ckpt("P2_WSTAR", seed, PB), "run_id": f"P2_W_MAIN_s{seed}"})
                else:
                    e["checkpoint_sha256"] = ckpt(arm, seed, rule)
                entries.append(e)
            if arm in DEVICE:
                anakit.jdump(root / entry_dir(arm, seed, PB) / "RESULT_ROWS.json", {"rows": dev_rows})
    for e in entries:
        for k in ("rows", "ranks"):
            files[e[k]] = R._sha_file(root / e[k])
    idx = {"format": R.INDEX_FORMAT, "study": "PREG2", "split": split, "eval_device_class": anakit.DEV,
           "roles": {"MAIN": {"eval_manifest_id": m.manifest_id, "train_manifest_id": "MAIN_TRAIN_5_100_131072"}},
           "entries": entries, "records": {}}
    anakit.jdump(root / R.INDEX_NAME, idx)

    def loader(role, spec):
        return None if role == "CALIBRATION" else ctx[role]
    return SimpleNamespace(root=root, m=m, base=base, ctx=ctx, loader=loader, entries=entries, ranks=rk, files=files,
                           split=split)


def write_lock(path: Path, w, *, unseal=(), declared=(), selection=None) -> Path:
    sealed, reused = [], []
    for e in w.entries:
        k = (e["arm"], e["seed"], e["rule"])
        if e["arm"] in P2.PHASE1_REUSED_ARMS:
            reused.append({"arm": k[0], "seed": k[1], "rule": k[2], "checkpoint_sha256": e.get("checkpoint_sha256")})
        elif k[0] not in unseal:
            it = {"arm": k[0], "seed": k[1], "rule": k[2]}
            if k[0] in P2.P2_DEVICE_ARMS:
                it["base_checkpoint_sha256"] = e["p2_base"]["checkpoint_sha256"]
            else:
                it["checkpoint_sha256"] = e["checkpoint_sha256"]
            sealed.append(it)
    obj = {"format": P2.P2_LOCK_FORMAT, "document": "P2_METHODS_LOCK", "draft": False, "preg2_sha256": P2.PREG2_SHA256,
           "p2_contrasts_self_sha256": P2.P2_CONTRASTS_SELF_SHA256, "bootstrap_seed": 20261004,
           "sealed_arms": sealed, "phase1_reused": reused,
           "declared_not_run": [{"arm": a, "seed": None, "reason": "synthetic: not finished at lock time"}
                                for a in declared],
           "selection_files_sha256": dict(selection or {})}
    obj["self_sha256"] = R._canon_compact(obj)
    anakit.jdump(path, obj)
    return path


def write_freeze(path: Path, **override) -> Path:
    files = {k: _sha(R.V4 / k) for k in P2.FREEZE_FILES}
    files.update(override)
    obj = {"format": P2.P2_FREEZE_FORMAT, "document": "P2_SCRIPT_FREEZE", "files_sha256": files}
    obj["self_sha256"] = R._canon_compact(obj)
    anakit.jdump(path, obj)
    return path


def append_register(reg_path: Path, items) -> None:
    import test_exposure_register as TER
    reg = json.loads(reg_path.read_text(encoding="utf-8"))
    for purpose, lsha, desc in items:
        prev = reg["entries"][-1]["entry_sha256"]
        i = len(reg["entries"]) + 1
        e = {"entry_id": i, "utc": "2026-10-04T00:00:0%dZ" % i, "actor_role": "Ops", "source": "synthetic",
             "split": "RETRO_TEST", "access_purpose": purpose, "observations": "NONE", "observations_description": desc,
             "decision": "synthetic", "all_methods_lock_sha256": lsha, "manifest_header_sha256": None,
             "model_checkpoint_sha256": None, "code_sha256": None, "receipt_path": None, "prev_entry_sha256": prev}
        e["entry_sha256"] = hashlib.sha256(TER.canon(e)).hexdigest()
        reg["entries"].append(e)
    anakit.jdump(reg_path, reg)


def write_phase1_result(path: Path, lock_sha: str, files: dict) -> Path:
    obj = R.with_self_sha({"schema": R.SCHEMA, "split": "TEST",
                           "script": {"file": "tools/retention_report.py", "sha256": R.script_sha256()},
                           "inputs": {"files_sha256": dict(files), "test_gate": {"all_methods_lock_sha256": lock_sha}}})
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(R.dumps(obj), encoding="utf-8")
    return path


def reused_files(w) -> dict:
    return {rel: sha for e in w.entries if e["arm"] in P2.PHASE1_REUSED_ARMS for rel, sha in
            ((e["rows"], w.files[e["rows"]]), (e["ranks"], w.files[e["ranks"]]))}


def make_test_world(tmp: Path, *, p2_entry=True, lock_kw=None, drop=(), row_extra=None) -> SimpleNamespace:
    """The TEST world: Phase-1 gate (anakit), the P2 block, P2 lock / freeze, the register with both
    LOCKED_EVALUATION entries, the sitting's Phase-1 RETENTION_RESULT, and the argv."""
    g = anakit.make_test_gate(tmp / "gate")
    w = build_world(tmp, "TEST", g, drop=drop, row_extra=row_extra)
    lock = write_lock(tmp / "P2_METHODS_LOCK.json", w, **(lock_kw or {}))
    items = [("LOCKED_EVALUATION", g["all_methods_lock_sha256"], "Phase-1 locked evaluation")]
    if p2_entry:
        items.append(("LOCKED_EVALUATION", R._sha_file(lock), "Phase-2 locked evaluation under P2_METHODS_LOCK"))
    append_register(g["register"], items)
    freeze = write_freeze(tmp / "P2_SCRIPT_FREEZE.json")
    p1 = write_phase1_result(tmp / "p1" / R.RESULT_NAME, g["all_methods_lock_sha256"], reused_files(w))
    w.gate, w.lock, w.freeze, w.p1 = g, lock, freeze, p1
    w.argv = ["--block", str(w.root), "--split", "TEST", "--contrasts", str(CONTRASTS), "--p2-lock", str(lock),
              "--p2-freeze", str(freeze), "--all-methods-lock", str(g["lock"]), "--phase1-result", str(p1),
              "--test-exposure-register", str(g["register"]), "--registry-ext", str(g["ext"])]
    return w


def run(argv, w, out: Path):
    lines = []
    rc = P2.main(list(argv) + ["--out", str(out)], context_loader=w.loader, say=lines.append)
    return rc, "\n".join(lines)


def replace_arg(argv, flag, value):
    a = list(argv)
    i = a.index(flag)
    if value is None:
        del a[i:i + 2]
    else:
        a[i + 1] = str(value)
    return a


@pytest.fixture(scope="module")
def sitting(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("p2_test_run")
    w = make_test_world(tmp)
    out = tmp / "p2_out_test_1"
    rc, txt = run(w.argv, w, out)
    assert rc == 0, txt
    res = json.loads((out / R.RESULT_NAME).read_text(encoding="utf-8"))
    return SimpleNamespace(w=w, out=out, res=res, txt=txt, by={c["id"]: c for c in res["contrasts"]})


# ------------------------------------------------------------------------------------------------ library integrity
def test_frozen_tools_unchanged_by_import(tmp_path):
    assert PRE_IMPORT == POST_IMPORT == FROZEN
    assert P2.RR is R and PBI.RR is R and PBI.ABI.RR is R
    code = ("import sys; sys.dont_write_bytecode = True; "
            f"sys.path[:0] = [{str(TOOLS)!r}, {str(CODE)!r}]; import ppsi_env; import p2_block_index, p2_report; "
            "print(p2_report.RR.__name__)")
    before = {"rr": _sha(RR_FILE), "abi": _sha(ABI_FILE)}
    cp = subprocess.run([sys.executable, "-B", "-c", code], capture_output=True, text=True, cwd=str(tmp_path))
    assert cp.returncode == 0, cp.stderr
    assert cp.stdout.strip() == "retention_report"
    assert {"rr": _sha(RR_FILE), "abi": _sha(ABI_FILE)} == before == FROZEN
    assert P2.assert_frozen_library() == {"tools/retention_report.py": FROZEN["rr"],
                                          "tools/ana_block_index.py": FROZEN["abi"]}


def test_frozen_sha_mismatch_refuses(monkeypatch):
    monkeypatch.setattr(P2, "RR_SHA256", "0" * 64)
    with pytest.raises(P2.P2Refused, match="retention_report"):
        P2.assert_frozen_library()
    with pytest.raises(ImportError, match="frozen"):
        P2._load_library()


def test_profile_restores_library_tables():
    saved = {k: getattr(R, k) for k in P2._PROFILE_KEYS}
    with P2.p2_profile(("P2_WSTAR",)):
        assert "P2_WSTAR" in R.ARMS and "FT_P2_WSTAR" in R.PB_ONLY_ARMS and R.REF == "PRE"
        assert "P2_WSTAR" in R.FROZEN_ONCE_ARMS and R.DP_Z["P2_WSTAR_DP8"] == 0.5642 and R.is_warm_start("P2_WSTAR")
        c2 = "P2_C2_CFULL_SMALL"
        assert R.SOURCE_TRAIN_MANIFEST[c2] == R.TRAIN_MANIFEST_OF[c2] == "FULL_TRAIN_5_100_TIEBREAK"
        with pytest.raises(P2.P2Refused, match="re-entrant"):
            with P2.p2_profile():
                pass
    assert all(getattr(R, k) is v for k, v in saved.items())
    with pytest.raises(RuntimeError):
        with P2.p2_profile():
            raise RuntimeError("boom")
    assert all(getattr(R, k) is v for k, v in saved.items())
    assert "P2_WSTAR" not in R.ARMS and R.REF == "C_BIG"


# ------------------------------------------------------------------------------------------------ registry
def test_contrasts_loader_registered_file():
    c = P2.load_p2_contrasts(CONTRASTS)
    assert c["self_sha256"] == P2.P2_CONTRASTS_SELF_SHA256 and c["n_contrasts"] == 14
    assert [i["id"] for i in c["items"]] == list(P2.P2_ORDER)
    assert c["holm_family"] == ["Q2", "Q3", "Q4", "Q5", "Q6", "Q7", "Q8"]
    assert c["retro_test_band_family"] == ["Q9", "Q11", "Q12a", "Q12b", "Q13"]
    by = {i["id"]: i for i in c["items"]}
    assert (by["Q6"]["kind"], by["Q6"]["num"], by["Q6"]["den"]) == ("retained", "P2_WSTAR_DP8", "P2_WSTAR_FA1024")
    assert (by["Q1"]["kind"], by["Q1"]["num"], by["Q1"]["den"]) == ("delta", "P2_WSTAR", "PRE")


@pytest.mark.parametrize("name", ["P2_CONTRASTS_v1.json", "P2_CONTRASTS_v2.json",
                                  "_withdrawn_P2_CONTRASTS_v2_never_registered.json",
                                  "_drafts_P2_CONTRASTS_v1_18fa8b67.json"])
def test_contrasts_loader_refuses_superseded(name):
    with pytest.raises(P2.P2Refused):
        P2.load_p2_contrasts(R.V4 / "company" / "gates" / name)


def test_contrasts_loader_refuses_tampering(tmp_path):
    obj = json.loads(CONTRASTS.read_text(encoding="utf-8"))
    obj["holm_family"] = obj["holm_family"][:-1]                      # Q8 dropped, self sha stale
    p = tmp_path / "c.json"
    p.write_text(json.dumps(obj, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(P2.P2Refused, match="self_sha256"):
        P2.load_p2_contrasts(p)
    obj["self_sha256"] = P2._utf8_compact_self(obj)                     # re-sealed: not the registered sha
    p.write_text(json.dumps(obj, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(P2.P2Refused, match="registered"):
        P2.load_p2_contrasts(p)


def test_disclosures_verbatim_in_preg2():
    text = R._norm_text((R.V4 / "contracts" / "PREG2.md").read_text(encoding="utf-8")).replace("**", "")
    assert _sha(R.V4 / "contracts" / "PREG2.md") == P2.PREG2_SHA256
    for k, v in P2.DISCLOSURES.items():
        assert R._norm_text(v) in text, k


# ------------------------------------------------------------------------------------------------ labels (§6.4)
def test_labels_per_6_4():
    for k in ("Q1", "Q2", "Q3", "Q4", "Q5", "Q6", "Q7", "Q8"):
        assert P2.p2_label(k, "TEST") == "CONFIRMATORY_P2 (PREREGISTERED_FOLLOWUP)"
    for k in ("Q9", "Q11", "Q12a", "Q12b", "Q13"):
        assert P2.p2_label(k, "TEST") == "TERTIARY_P2 (unadjusted; DESCRIPTIVE Holm band)"
    for k in P2.P2_RETRO_TEST_IDS:
        assert P2.p2_label(k, "VALIDATION") == "POST_VALIDATION_EXPLORATORY"
    assert P2.p2_label("Q14", "TEST") == P2.p2_label("Q14", "VALIDATION") == "TEST2_NOT_AVAILABLE"
    assert P2.p2_label("Q1_TEST2", "TEST") == "TEST2_NOT_AVAILABLE"
    assert dict(P2.LABELS_6_4)["audits E1–E5, devices §8, D1-ext"] == "DESCRIPTIVE (tertiary)"


# ------------------------------------------------------------------------------------------------ statistics units
class FakeEng:
    def __init__(self):
        self.boot = {}


def _std_normal(n=1000, seed=7):
    z = np.random.default_rng(seed).standard_normal(n)
    return (z - z.mean()) / z.std(ddof=1)                               # mean 0, sample sd exactly 1


def _fake_rec(kid, point, se, eng, kind="delta"):
    boot = point + se * _std_normal(seed=sum(map(ord, kid)))
    eng.boot[(kid, PB)] = boot
    b = {"point": float(point), "ci95": [float(x) for x in np.percentile(boot, [2.5, 97.5])], "status": "COMPUTED"}
    return {"id": kid, "kind": kind, "confirmatory_rule": PB, "by_rule": {PB: b}, "decision": {}}


def test_holm_over_q2_q8_known_answer():
    eng = FakeEng()
    zs = {"Q1": 10.0, "Q2": 4.0, "Q3": 3.0, "Q4": 2.6, "Q5": 2.4, "Q6": 2.2, "Q7": 1.0, "Q8": 0.5,
          "Q9": 5.0, "Q11": 0.1, "Q12a": 3.5, "Q12b": 0.2, "Q13": 0.3}
    conf = [_fake_rec(k, z * 0.001, 0.001, eng) for k, z in zs.items()]
    h = P2.p2_holm(eng, conf, "TEST")
    assert h["family"] == ["Q2", "Q3", "Q4", "Q5", "Q6", "Q7", "Q8"] and h["family_size"] == 7
    assert h["n_rejected"] == 3 and h["alpha"] == 0.05
    by = {r["id"]: r["by_rule"][PB]["holm"] for r in conf}
    assert [k for k in zs if (by[k].get("holm_reject") is True)] == ["Q2", "Q3", "Q4"]
    for j, k in enumerate(("Q2", "Q3", "Q4", "Q5", "Q6", "Q7", "Q8"), start=1):   # sorted by p = by z here
        assert by[k]["holm_rank"] == j and by[k]["holm_level"] == pytest.approx(0.05 / (7 - j + 1))
    assert by["Q5"]["holm_reject"] is False and by["Q5"]["holm_band_level"] == pytest.approx(0.05 / 4)
    assert by["Q1"] == {"status": "NOT_IN_HOLM_FAMILY", "tier": "primary",
                        "rule": "PREG2 §9.3: Q1 alone is primary, read unadjusted at alpha 0.05, outside Holm"}
    assert h["tertiary_band"]["family"] == ["Q9", "Q11", "Q12a", "Q12b", "Q13"]
    assert h["tertiary_band"]["family_size"] == 5
    for k in ("Q9", "Q11", "Q12a", "Q12b", "Q13"):
        assert by[k]["family"] == "TERTIARY_BAND" and by[k]["holm_reject"] is None
        assert by[k]["decision"].startswith("none (TERTIARY_P2")
    assert {r["id"]: r["decision"].get("holm_reject") for r in conf if r["id"] in P2.P2_HOLM_FAMILY} == \
        {"Q2": True, "Q3": True, "Q4": True, "Q5": False, "Q6": False, "Q7": False, "Q8": False}
    assert "Q14" not in h["family"] and h["test2_family"] == ["Q14"]


def test_tost_q1_margin():
    inside = np.linspace(-0.00045, 0.00045, 1000)
    t = P2.tost(inside, P2.Q1_INTERVAL)
    assert t["margin"] == [-0.0005, 0.0005] and t["equivalent"] is True
    wide = np.linspace(-0.0002, 0.0008, 1000)                           # 95th percentile 0.00075 > 0.0005
    assert P2.tost(wide, P2.Q1_INTERVAL)["equivalent"] is False
    edge = np.linspace(-0.00055, 0.00055, 1001)                         # 5th / 95th pct = ±0.00044: inside
    assert P2.tost(edge, P2.Q1_INTERVAL)["equivalent"] is True
    eng = FakeEng()
    eng.boot[("Q1", PB)] = inside
    q = P2.q1_equivalence(eng, ("Q1", PB), {"ci95": [-0.0004, 0.0004]})
    assert q["equivalent"] is True and q["margin"] == [-0.0005, 0.0005] and q["ci95_inside_margin"] is True


def _q1(ci, per, eq, status="COMPLETE"):
    return {"confirmatory_rule": PB, "by_rule": {PB: {
        "point": float(np.mean(per)) if per else 0.0, "ci95": ci, "seeds_status": status,
        "per_seed": [{"point": p} for p in per], "equivalence": {"equivalent": eq, "ci90": ci}}}}


def test_q1_reading_grid_9_4():
    assert P2.q1_reading(_q1([0.0002, 0.0009], [0.0004, 0.0007], False))["reading"] == "SUCCESS"
    r = P2.q1_reading(_q1([0.0001, 0.0004], [0.0002, 0.0003], True))
    assert r["reading"] == "SUCCESS" and r["conditions_true"] == ["SUCCESS", "EQUIVALENT"] and r["overlap"] is True
    assert P2.q1_reading(_q1([-0.0003, 0.0003], [0.0001, -0.0001], True))["reading"] == "EQUIVALENT"
    assert P2.q1_reading(_q1([-0.002, -0.001], [-0.0015, -0.0014], False))["reading"] == "FAILURE"
    assert P2.q1_reading(_q1([-0.0002, 0.0012], [0.0004, 0.0006], False))["reading"] == "INCONCLUSIVE"
    assert P2.q1_reading(_q1([0.0001, 0.0009], [0.0009, -0.0001], False))["reading"] == "INCONCLUSIVE"
    r = P2.q1_reading(_q1([0.0002, 0.0009], [0.0005], False, status="INCOMPLETE_SEEDS"))
    assert r["reading"] == "INCONCLUSIVE" and "success_blocked" in r
    assert P2.q1_reading({"confirmatory_rule": PB, "by_rule": {PB: {"status": "NOT_AVAILABLE"}}})["reading"] == \
        "NOT_AVAILABLE"
    assert r["D-h"] == P2.DISCLOSURES["D-h"]


def test_interpretation_grid_rows():
    eng = FakeEng()

    def q2(point, reject, eqboot):
        eng.boot[("Q2", PB)] = eqboot
        return {"id": "Q2", "confirmatory_rule": PB,
                "by_rule": {PB: {"point": point, "holm": {"holm_reject": reject, "reference": 0.0}}}}
    pos, eqb = np.linspace(0.002, 0.004, 1000), np.linspace(-0.0004, 0.0004, 1000)
    g = P2.interpretation_grid({"reading": "SUCCESS", "conditions_true": ["SUCCESS"]}, q2(0.003, True, pos), eng)
    assert g["cell"] == "Q2_POSITIVE x Q1_POSITIVE"
    g = P2.interpretation_grid({"reading": "EQUIVALENT", "conditions_true": ["EQUIVALENT"]}, q2(0.003, True, pos), eng)
    assert g["cell"] == "Q2_POSITIVE x Q1_NONPOSITIVE_OR_EQUIVALENT"
    g = P2.interpretation_grid({"reading": "EQUIVALENT", "conditions_true": ["EQUIVALENT"]}, q2(0.0, False, eqb), eng)
    assert g["cell"] == "Q2_EQUIVALENT x Q1_EQUIVALENT"
    g = P2.interpretation_grid({"reading": "SUCCESS", "conditions_true": ["SUCCESS"]}, q2(0.0, False, eqb), eng)
    assert g["cell"] == "Q2_EQUIVALENT x Q1_POSITIVE"
    g = P2.interpretation_grid({"reading": "INCONCLUSIVE", "conditions_true": []}, q2(0.003, True, pos), eng)
    assert g["cell"] == "NO_REGISTERED_CELL"


def test_tost_q13_margin_99_101():
    eng = FakeEng()
    eng.boot[("Q13", PB)] = np.linspace(99.2, 100.8, 1000)
    q = P2.q13_equivalence(eng, ("Q13", PB), {"ci95": [99.1, 100.9]}, "Q13 rule text")
    assert q["status"] == "COMPUTED" and q["margin"] == [99.0, 101.0] and q["equivalent"] is True
    assert q["rule"] == "Q13 rule text" and "PREG2" in q["registered"]
    eng.boot[("Q13", PB)] = np.linspace(98.5, 100.5, 1000)              # 5th percentile 98.6 < 99
    assert P2.q13_equivalence(eng, ("Q13", PB), {"ci95": [98.5, 100.5]}, "x")["equivalent"] is False


# ------------------------------------------------------------------------------------------------ end to end (TEST)
def test_test_run_labels_and_disclosures(sitting):
    res, by = sitting.res, sitting.by
    assert res["schema"] == "PPSI_V4_P2_RETENTION_1" and res["split"] == "TEST" and R.verify_self_sha(res)
    for k in ("Q1", "Q2", "Q3", "Q4", "Q5", "Q6", "Q7", "Q8"):
        assert by[k]["label"] == "CONFIRMATORY_P2 (PREREGISTERED_FOLLOWUP)"
        assert by[k]["disclosure_D_a"] == P2.DISCLOSURES["D-a"]
        assert by[k]["by_rule"][by[k]["confirmatory_rule"]]["label"] == by[k]["label"]
        other = CB if by[k]["confirmatory_rule"] == PB else PB
        assert by[k]["by_rule"][other]["label"] == "DESCRIPTIVE"
    for k in ("Q9", "Q11", "Q12a", "Q12b", "Q13"):
        assert by[k]["label"] == "TERTIARY_P2 (unadjusted; DESCRIPTIVE Holm band)" and "disclosure_D_a" not in by[k]
    assert by["Q14"]["label"] == by["Q14"]["status"] == "TEST2_NOT_AVAILABLE"
    assert res["test2"]["Q1_replication"] == res["test2"]["Q14"] == "TEST2_NOT_AVAILABLE"
    assert res["q1_reading"]["D-h"] == P2.DISCLOSURES["D-h"]
    assert by["Q6"]["disclosure_D_i"] == P2.DISCLOSURES["D-i"]
    assert by["Q6"]["by_rule"][PB]["disclosure"].startswith("Q6/Q7/Q11/Q12a/Q12b PRACTICAL_BEST ratios")
    assert res["labels"]["by_contrast"]["Q14"] == "TEST2_NOT_AVAILABLE"
    assert set(res["disclosures"]) == {f"D-{c}" for c in "abcdefghijk"}
    rep = (sitting.out / R.REPORT_NAME).read_text(encoding="utf-8")
    assert P2.DISCLOSURES["D-a"] in rep and "TEST2_NOT_AVAILABLE" in rep and P2.DISCLOSURES["D-h"] in rep
    man = json.loads((sitting.out / R.MANIFEST_NAME).read_text(encoding="utf-8"))
    assert man["result_self_sha256"] == res["self_sha256"] and R.verify_self_sha(man)
    assert str(sitting.w.root) not in json.dumps(res)                  # no absolute block path in the result


def test_test_run_bootstrap_seed_20261004(sitting):
    res, m = sitting.res, sitting.w.m
    users = np.unique(m.eval_user_key[m.eval_mask])
    assert res["bootstrap"]["seed"] == 20261004 and res["bootstrap"]["n_resamples"] == 1000
    want = R.LIB.B.make_plan(users, seed=20261004).W_sha256
    assert res["bootstrap"]["main_view"]["W_sha256"] == want != R.LIB.BC._plan(m).W_sha256


def test_test_run_holm_and_families(sitting):
    h, by = sitting.res["holm"], sitting.by
    assert h["family"] == list(P2.P2_HOLM_FAMILY) and h["family_size"] == 7
    assert h["tertiary_band_family"] == ["Q9", "Q11", "Q12a", "Q12b", "Q13"] and h["test2_family"] == ["Q14"]
    assert by["Q1"]["by_rule"][PB]["holm"]["status"] == "NOT_IN_HOLM_FAMILY"
    for k in P2.P2_HOLM_FAMILY:
        hk = by[k]["by_rule"][by[k]["confirmatory_rule"]]["holm"]
        assert hk["status"] in ("COMPUTED", "NOT_COMPUTED") and by[k]["decision"].get("holm_reject") in (True, False, None)


def test_test_run_known_answers(sitting):
    w, by = sitting.w, sitting.by
    mac = {k: anakit.macro(v, w.m) for k, v in w.ranks.items()}
    pre = mac[("PRE", 2026, PB)]
    q1 = by["Q1"]["by_rule"][PB]
    assert [p["seed"] for p in q1["per_seed"]] == [2026, 2027]
    for p in q1["per_seed"]:
        assert p["point"] == pytest.approx(mac[("P2_WSTAR", p["seed"], PB)] - pre, abs=1e-12)
    assert q1["equivalence"]["margin"] == [-0.0005, 0.0005] and q1["seeds_status"] == "COMPLETE"
    q13 = by["Q13"]["by_rule"][PB]
    assert q13["point"] == pytest.approx(100.0, abs=1e-9) and q13["equivalence"]["margin"] == [99.0, 101.0]
    assert q13["equivalence"]["equivalent"] is True and q13["equivalence"]["ci90"] == pytest.approx([100.0, 100.0])
    assert by["Q6"]["by_rule"][CB]["point"] == pytest.approx(100.0, abs=1e-9) and by["Q6"]["confirmatory_rule"] == CB
    assert [p["seed"] for p in by["Q9"]["by_rule"][PB]["per_seed"]] == [2026, 2027]       # one C_FULL s2026 shared
    assert by["Q8"]["registered_seeds"] == [2026] and by["Q8"]["by_rule"][PB]["seeds_status"] == "COMPLETE"
    assert by["Q5"]["by_rule"][PB]["seeds_status"] == "COMPLETE"                            # base_best_only kept both
    q10 = sitting.res["q10_headroom_capture"]
    q2 = {p["seed"]: p["point"] for p in by["Q2"]["by_rule"][PB]["per_seed"]}
    for p in q10["per_seed"]:
        q1s = mac[("P2_WSTAR", p["seed"], PB)] - pre
        assert p["point"] == pytest.approx(q1s / q2[p["seed"]], rel=1e-12)
    assert q10["status"] == "COMPUTED" and q10["point"] == pytest.approx(np.mean([p["point"] for p in q10["per_seed"]]))
    assert sitting.res["q1_reading"]["reading"] in ("SUCCESS", "EQUIVALENT", "FAILURE", "INCONCLUSIVE")
    assert sitting.res["deployment_grade"]["rule"] == P2.DEPLOYMENT_RULE
    assert sitting.res["descriptive_sections"]["E1_MIA"]["status"] == "NOT_RUN"
    assert all(v["label"] == "DESCRIPTIVE (tertiary)" for v in sitting.res["descriptive_sections"].values())
    assert sitting.res["checks"]["sealed"]["n_sealed_entries_checked"] == sum(
        1 for e in w.entries if e["arm"] in P2.P2_ARMS)
    assert len(sitting.res["checks"]["phase1_binding_S5"]) == sum(
        1 for e in w.entries if e["arm"] in P2.PHASE1_REUSED_ARMS)


def test_check_only_prints_no_value(tmp_path):
    w = make_test_world(tmp_path)
    out = tmp_path / "p2_out_test_check"
    rc, txt = run(w.argv + ["--check-only"], w, out)
    assert rc == 0, txt
    assert "CHECK-ONLY: PASS" in txt and not out.exists()


# ------------------------------------------------------------------------------------------------ refusals
def _refused(w, argv, out, match):
    rc, txt = run(argv, w, out)
    assert rc == 2, txt
    assert match in txt, txt
    assert not (out / R.RESULT_NAME).exists()


def test_refuses_without_p2_lock(tmp_path):
    w = make_test_world(tmp_path)
    _refused(w, replace_arg(w.argv, "--p2-lock", None), tmp_path / "p2_out_test_a", "P2_METHODS_LOCK")
    _refused(w, replace_arg(w.argv, "--p2-lock", tmp_path / "missing.json"), tmp_path / "p2_out_test_b",
             "does not exist")


def test_refuses_without_p2_register_entry(tmp_path):
    w = make_test_world(tmp_path, p2_entry=False)
    _refused(w, w.argv, tmp_path / "p2_out_test_c", "LOCKED_EVALUATION")


def test_refuses_without_phase1_result(tmp_path):
    w = make_test_world(tmp_path)
    _refused(w, replace_arg(w.argv, "--phase1-result", None), tmp_path / "p2_out_test_d", "RETENTION_RESULT")


def test_refuses_freeze_mismatch(tmp_path):
    w = make_test_world(tmp_path)
    bad = write_freeze(tmp_path / "bad_freeze.json", **{"tools/p2_report.py": "0" * 64})
    _refused(w, replace_arg(w.argv, "--p2-freeze", bad), tmp_path / "p2_out_test_e", "P2_SCRIPT_FREEZE")


def test_refuses_reused_row_hash_differs_S5(tmp_path):
    w = make_test_world(tmp_path)
    files = reused_files(w)
    rel = next(e["rows"] for e in w.entries if e["arm"] == "PRE")
    files[rel] = "0" * 64
    p1 = write_phase1_result(tmp_path / "p1b" / R.RESULT_NAME, w.gate["all_methods_lock_sha256"], files)
    _refused(w, replace_arg(w.argv, "--phase1-result", p1), tmp_path / "p2_out_test_f", "S5")


def test_refuses_unsealed_and_declared_not_run(tmp_path):
    w = make_test_world(tmp_path, lock_kw={"unseal": ("P2_WSTAR_SECAGG",)})
    _refused(w, w.argv, tmp_path / "p2_out_test_g", "not a sealed arm")
    w2 = make_test_world(tmp_path / "w2", lock_kw={"unseal": ("P2_WSTAR_SECAGG",), "declared": ("P2_WSTAR_SECAGG",)})
    _refused(w2, w2.argv, tmp_path / "p2_out_test_h", "declared not run")


def test_declared_not_run_arm_absent_reads_declared(tmp_path):
    w = make_test_world(tmp_path, drop=("P2_WSTAR_SECAGG",), lock_kw={"declared": ("P2_WSTAR_SECAGG",)})
    out = tmp_path / "p2_out_test_i"
    rc, txt = run(w.argv, w, out)
    assert rc == 0, txt
    res = json.loads((out / R.RESULT_NAME).read_text(encoding="utf-8"))
    q13 = next(c for c in res["contrasts"] if c["id"] == "Q13")
    assert q13["by_rule"][PB]["status"] == "DECLARED_NOT_RUN" and q13["declared_not_run"][0]["arm"] == "P2_WSTAR_SECAGG"
    assert res["declared_not_run"][0]["arm"] == "P2_WSTAR_SECAGG"


def test_refuses_test_out_name_and_p2_rows_without_study(tmp_path):
    w = make_test_world(tmp_path)
    _refused(w, w.argv, tmp_path / "out_plain", "p2_out_test_")
    w2 = make_test_world(tmp_path / "w2", row_extra={"P2_WSTAR": {"study": None}})
    _refused(w2, w2.argv, tmp_path / "p2_out_test_j", "study PREG2")


def test_validation_gate_after_the_sitting(tmp_path, sitting):
    """S8: VALIDATION needs the P2 lock, the Phase-1 sitting result AND the Phase-2 sitting result."""
    tw = sitting.w
    v = build_world(tmp_path / "val", "VALIDATION")
    base = ["--block", str(v.root), "--split", "VALIDATION", "--contrasts", str(CONTRASTS), "--p2-lock", str(tw.lock),
            "--p2-freeze", str(tw.freeze), "--all-methods-lock", str(tw.gate["lock"]), "--phase1-result", str(tw.p1)]
    p2t = str(sitting.out / R.RESULT_NAME)
    _refused(v, base, tmp_path / "v_a", "--p2-test-result")
    _refused(v, replace_arg(base + ["--p2-test-result", p2t], "--phase1-result", None), tmp_path / "v_b",
             "RETENTION_RESULT")
    _refused(v, replace_arg(base + ["--p2-test-result", p2t], "--p2-lock", None), tmp_path / "v_c", "P2_METHODS_LOCK")
    out = tmp_path / "v_ok"
    rc, txt = run(base + ["--p2-test-result", p2t], v, out)
    assert rc == 0, txt
    res = json.loads((out / R.RESULT_NAME).read_text(encoding="utf-8"))
    assert res["split"] == "VALIDATION"
    for c in res["contrasts"]:
        if c["id"] == "Q14":
            assert c["label"] == "TEST2_NOT_AVAILABLE"
            continue
        assert c["label"] == "POST_VALIDATION_EXPLORATORY" and "disclosure_D_a" not in c
        for b in c["by_rule"].values():
            assert b["label"] == "POST_VALIDATION_EXPLORATORY"
    assert all(r["label"] == "POST_VALIDATION_EXPLORATORY" for r in res["values"])
    assert res["checks"]["phase1_binding_S5"]["status"] == "NOT_APPLICABLE"


# ------------------------------------------------------------------------------------------------ p2_block_index
def _phase1_block(tmp: Path, w) -> Path:
    """A Phase-1 TEST block (ana_block_index layout, verified self_sha256) holding the reused entries' files."""
    src = tmp / "p1block"
    ents = []
    for e in w.entries:
        if e["arm"] not in P2.PHASE1_REUSED_ARMS:
            continue
        for k in ("rows", "ranks"):
            (src / e[k]).parent.mkdir(parents=True, exist_ok=True)
            (src / e[k]).write_bytes((w.root / e[k]).read_bytes())
        ents.append({k: v for k, v in e.items() if k != "phase1_reused"})
    idx = {"format": R.INDEX_FORMAT, "split": w.split, "eval_device_class": anakit.DEV,
           "roles": {"MAIN": {"eval_manifest_id": w.m.manifest_id, "train_manifest_id": "MAIN_TRAIN_5_100_131072"}},
           "entries": ents, "records": {}}
    idx["self_sha256"] = R._canon_indent(json.loads(json.dumps(idx)))
    anakit.jdump(src / R.INDEX_NAME, idx)
    return src


def _p2_sources(tmp: Path, w) -> tuple:
    """P2 REEVAL dirs through the real v4int.reeval.write_outputs (layoutkit) + FT_P2_WSTAR --p2-dev dirs."""
    rroot, droot = tmp / "runs_p2" / "REEVAL_A40", tmp / "runs_p2" / "DEV"
    for e in w.entries:
        if e["arm"] not in P2.P2_ARMS:
            continue
        if e["arm"] in P2.P2_DEVICE_ARMS:
            d = droot / f"FT_P2_WSTAR_s{e['seed']}"
            layoutkit._cp(w.root / e["rows"], d / "RESULT_ROWS.json")
            layoutkit._cp(w.root / e["ranks"], d / "cells" / f"{anakit.DEVICE_CELL}.ranks_int32.npy")
            anakit.jdump(d / "RUN_SUMMARY.json", {"status": "DONE", "arm": "FT_P2_WSTAR", "role": "MAIN_TEST",
                                                  "selected_cell": anakit.DEVICE_CELL, "spec": {"seed": e["seed"]},
                                                  "device_pin_sha256": anakit.PIN})
            anakit.jdump(d / "LAUNCH_0.json", {"label": "PRODUCTION", "arm": "FT_P2_WSTAR",
                                               "start": {"kind": "P2_WSTAR", "seed": e["seed"], "checkpoint": "best.pt",
                                                         "file_sha256": e["p2_base"]["checkpoint_sha256"],
                                                         "run_id": e["p2_base"]["run_id"]}})
            continue
        layoutkit.write_reeval_dir(rroot / f"{e['arm']}_s{e['seed']}_{TAG[e['rule']]}", w.root / e["rows"],
                                   w.root / e["ranks"], {
                                       "format": "PPSI_V4_REEVAL_RECORD_1", "arm_label": e["arm"], "seed": e["seed"],
                                       "role": "MAIN_TEST", "selection_rule": e["rule"],
                                       "checkpoint_id": anakit.ckpt_id(e["arm"], e["seed"], e["rule"]),
                                       "checkpoint_sha256": e["checkpoint_sha256"], "device_class": anakit.DEV,
                                       "device_pin_sha256": anakit.PIN, "eval_manifest_id": w.m.manifest_id,
                                       **({"central_source": {"source": e["arm"], "arm_label": e["arm"],
                                                              "pair": {"train": C2_LABELS["source_train_manifest_id"],
                                                                       "eval": w.m.manifest_id}}}
                                          if e["arm"] == "P2_C2_CFULL_SMALL" else {})},
                                   "MAIN_TEST")
    pin = tmp / "EVAL_DEVICE_PIN_A40.json"
    anakit.jdump(pin, layoutkit.PIN_OBJ)
    return rroot, droot, pin


def test_block_index_builds_a_p2_block_that_p2_report_accepts(tmp_path):
    w = make_test_world(tmp_path)
    p1src = _phase1_block(tmp_path, w)
    rroot, droot, pin = _p2_sources(tmp_path, w)
    out_blk = tmp_path / "p2block"
    lines = []
    rc = PBI.main(["--split", "TEST", "--device-class", anakit.DEV, "--pin", str(pin), "--phase1-block", str(p1src),
                   "--reeval", str(rroot), "--p2-dev", str(droot), "--p2-lock", str(w.lock), "--out", str(out_blk)],
                  say=lines.append)
    assert rc == 0, "\n".join(lines)
    idx = json.loads((out_blk / R.INDEX_NAME).read_text(encoding="utf-8"))
    assert idx["study"] == "PREG2" and idx["self_sha256"] == R._canon_indent(idx)
    assert idx["builder"]["ana_block_index_sha256"] == FROZEN["abi"] and idx["p2_lock_checks"]["status"] == "APPLIED"
    reused = [e for e in idx["entries"] if e.get("phase1_reused")]
    assert {(e["arm"], e["seed"], e["rule"]) for e in reused} == \
        {(e["arm"], e["seed"], e["rule"]) for e in w.entries if e["arm"] in P2.PHASE1_REUSED_ARMS}
    for e in reused:                                                     # same block paths, byte-identical files
        assert R._sha_file(out_blk / e["rows"]) == w.files[e["rows"]]
        assert R._sha_file(out_blk / e["ranks"]) == w.files[e["ranks"]]
    assert {e["arm"] for e in idx["entries"]} == set(P2.P2_BLOCK_ARMS)
    w.root = out_blk                                                     # the built block through p2_report
    out = tmp_path / "p2_out_test_built"
    rc, txt = run(replace_arg(w.argv, "--block", out_blk), w, out)
    assert rc == 0, txt
    res = json.loads((out / R.RESULT_NAME).read_text(encoding="utf-8"))
    assert {c["id"]: c["label"] for c in res["contrasts"]}["Q1"] == "CONFIRMATORY_P2 (PREREGISTERED_FOLLOWUP)"


def test_block_index_refusals(tmp_path):
    w = make_test_world(tmp_path)
    p1src = _phase1_block(tmp_path, w)
    rroot, _droot, pin = _p2_sources(tmp_path, w)
    common = ["--split", "TEST", "--device-class", anakit.DEV, "--pin", str(pin)]

    def go(extra, name):
        lines = []
        rc = PBI.main(common + extra + ["--out", str(tmp_path / name)], say=lines.append)
        return rc, "\n".join(lines)
    pre_dir = tmp_path / "phase1_as_p2" / "PRE_RESCORE_s2026_PB"                # a Phase-1 arm re-scored for Phase 2
    e = next(x for x in w.entries if x["arm"] == "PRE" and x["rule"] == PB)
    layoutkit.write_reeval_dir(pre_dir, w.root / e["rows"], w.root / e["ranks"], {
        "format": "PPSI_V4_REEVAL_RECORD_1", "arm_label": "PRE", "seed": 2026, "role": "MAIN_TEST",
        "selection_rule": PB, "checkpoint_id": "PRE_s2026:endpoint_6.0", "checkpoint_sha256": e["checkpoint_sha256"],
        "device_class": anakit.DEV, "device_pin_sha256": anakit.PIN, "eval_manifest_id": w.m.manifest_id,
        "central_source": layoutkit.central_source_record("PRE")}, "MAIN_TEST")
    rc, txt = go(["--phase1-block", str(p1src), "--reeval", str(pre_dir.parent)], "o1")
    assert rc == 2 and "S5" in txt, txt
    rc, txt = go(["--phase1-block", str(p1src), "--records", f"STRATA={w.p1}"], "o2")
    assert rc == 2 and "Phase-2 record" in txt, txt
    idx = json.loads((p1src / R.INDEX_NAME).read_text(encoding="utf-8"))
    idx.pop("self_sha256")
    anakit.jdump(p1src / R.INDEX_NAME, idx)
    rc, txt = go(["--phase1-block", str(p1src)], "o3")
    assert rc == 2 and "self_sha256" in txt, txt
    assert not any((tmp_path / n).exists() and any((tmp_path / n).iterdir()) for n in ("o1", "o2", "o3"))
