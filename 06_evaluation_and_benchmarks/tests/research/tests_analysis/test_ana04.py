"""ANA-04 (PREG1_AMENDMENT_06L_C, MULTI_EXPERT_REVIEW_20260929 §4 Gate C): C3 (baseline rank files in v4base.run, the LAST
anchor, R*, BELOW_DEVICE_HEURISTIC, K11b, the device-only envelope), C4 (company/gates/ANA_CONTRASTS_v3.json), C8 (one
bytes axis, S8 − S7 with the bytes ratio), C1 (privacy matrix), C11 (GRU_D128 vs C_SMALL), C12 (claim register +
number-free headline), C13-C15 (seed spread / SEED_SENSITIVE, Holm band, CALIB-disjoint, harmed share + seed null, K14
margin, repeat / novel strata) and the arm admissions (FT_PRE / K9c, K9b on UNSEEN_HOLDOUT, S8, DP8_FROZEN, the no-FT
UNSEEN_VAL references) — synthetic blocks only.

Known answers come from the block's own rank files (anakit.macro / the v4eval library), never from the script's tables.
Each MAJOR item has a positive test and a negative control (test_nc_*, strict xfail on AssertionError: the same
positive check run against a mutant that re-introduces the defect)."""
from __future__ import annotations

import dataclasses
import json
import re
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import ana_block_index as BI
import anakit
import layoutkit
import retention_report as R
import test_gate_b as GB
from anakit import macro
from test_ana02 import edit_index
from test_retention_report import _set_rows_field, check_refuses, conf, nc, result, run
from v4eval.metrics import credits_from_ranks, population_block

M = "MAIN"
PB, CB = "PRACTICAL_BEST", "CONTROLLED_BUDGET"
V3_ORDER = ["K1", "K2", "K3", "K4", "K5", "K6", "K7", "K8", "K9", "K9b", "K9c", "K10", "K11", "K11b", "K12", "K13", "K14",
            "K15", "K16", "K17", "K18", "K19", "K20", "K20b", "K20c", "K21"]          # 06L-C §C4, written out here
V3_FILE = R.V4 / "company" / "gates" / "ANA_CONTRASTS_v3.json"
AMEND = R.V4 / "contracts" / "PREG1_AMENDMENT_06L_C.md"


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def mmacro(ranks, m, mask) -> float:
    return population_block(credits_from_ranks(ranks, (20,)), mask, m.eval_user_key)["macro"]["mrr@20"]


def rk(blk, arm, seed, rule=PB, role=M):
    return blk.ranks[(role, arm, seed, rule)]


def pbv(blk, arm, seed, rule=PB, role=M):
    return macro(rk(blk, arm, seed, rule, role), blk.manifests[role])


def last_v(blk, role=M):
    return macro(blk.ranks[(role, "LAST", None, "RULE_FREE")], blk.manifests[role])


def add_fl(blk, arm, seeds=(2026, 2027), patch=None):
    """FL run dirs (anakit._fl_run) for `arm`, optionally patching RUN_SUMMARY."""
    idx = blk.index()
    m, base = blk.manifests[M], blk.base[M]
    for seed in seeds:
        rel = f"MAIN/FL_{arm}_s{seed}"
        anakit._fl_run(blk.root, rel, arm, seed, m, base, m.rankable, idx["entries"], "VALIDATION")
        if patch is not None:
            p = blk.root / rel / "RUN_SUMMARY.json"
            anakit.jdump(p, patch(json.loads(p.read_text(encoding="utf-8"))))
        for rule in R.RULES:
            blk.ranks[(M, arm, seed, rule)] = anakit.arm_ranks(arm, seed, rule, base, m.rankable)
    anakit.ensure_warm_records(blk.root, idx)                          # REV-ANA-05 F3: the S7 / S8 file in records
    blk.write_index(idx)


def dp8_frozen_patch(z=0.5642):
    def p(summ):
        s = GB.s8_summary_patch(summ)
        s["arm"] = "DP_WARM_FROZEN"                                    # ANA-06: FL-16 runs DP8_FROZEN as DP_WARM_FROZEN
        s["arm_spec"] = anakit.warm_line_spec("DP8_FROZEN", z=z)       # eps 8 (theta_pre, S8 frozen keys, Poisson DP)
        return s
    return p


def full_block(tmp, *, version=3, calib=True):
    """Every ANA-04 arm: v3 registry, warm start (FA_WARM / PRE), ranked LAST / RULE with tie counts, CALIBRATION,
    FT_PRE, S8 (two seeds), GRU_D128, the UNSEEN_VAL no-FT reference of C_SMALL."""
    blk = anakit.build(tmp)
    anakit.jdump(blk.contrasts, anakit.contrasts_obj(version))
    anakit.add_warm_start(blk)
    anakit.add_baselines_ranked(blk)
    if calib:
        anakit.add_calib(blk)
    anakit.add_ft_pre(blk)
    GB.add_s8(blk)
    anakit.add_plain(blk, "GRU_D128", (2026, 2027))
    anakit.add_role(blk, "UNSEEN_VAL")
    anakit.add_plain(blk, "C_SMALL", (2026,), (PB,), role="UNSEEN_VAL")
    return blk


@pytest.fixture(scope="module")
def full(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("ana04_full")
    blk = full_block(tmp)
    rc, text = run(blk, tmp / "out")
    assert rc == 0, text
    return blk, tmp / "out", result(tmp / "out")


def _rec(res, kid):
    for key in ("confirmatory", "extra_registered", "registered_descriptive"):
        for c in res.get(key) or []:
            if c["id"] == kid:
                return c
    raise KeyError(kid)


def _anchor(res, arm, rule=PB):
    row = next((r for r in res["last_anchor"]["rows"] if r["arm"] == arm and r["rule"] == rule), None)
    assert row is not None, (arm, rule, res["last_anchor"].get("status"))
    return row


# ================================================================================================ C3: v4base.run rank files
def _basekit():
    p = str(R.V4 / "code" / "tests_base")
    if p not in sys.path:
        sys.path.append(p)
    import basekit
    return basekit


def check_rank_files(out: Path, manifest, L: int) -> dict:
    """The rank file reproduces rows.json's macro-E2E MRR@20 (1e-12), -1 exactly where the tie file is -1, RANKS_RECORD
    binds the three files by sha256, and LAST's tie count 0 rows rank <= L (target in the window)."""
    rec = json.loads((out / "RANKS_RECORD.json").read_text(encoding="utf-8"))
    ranks = np.load(out / rec["ranks_file"]).astype(np.int64)
    ties = np.load(out / rec["n_tied_file"]).astype(np.int64)
    assert ranks.dtype == np.int64 and ranks.shape == (manifest.n,) and ties.shape == (manifest.n,)
    assert rec["ranks_sha256"] == R._sha_file(out / rec["ranks_file"])
    assert rec["n_tied_sha256"] == R._sha_file(out / rec["n_tied_file"])
    assert rec["rows_sha256"] == R._sha_file(out / "rows.json") and rec["n_rows"] == manifest.n
    body = {k: v for k, v in rec.items() if k != "self_sha256"}
    assert rec["self_sha256"] == R._sha_bytes(json.dumps(body, sort_keys=True, separators=(",", ":")).encode())
    rows = json.loads((out / "rows.json").read_text(encoding="utf-8"))
    want = next(r["value"] for r in rows if r["metric"] == "mrr" and r["k"] == 20 and r["aggregation"] == "macro"
                and r["target_denominator"] == "E2E")
    assert abs(mmacro(ranks, manifest, manifest.eval_mask) - want) <= 1e-12
    assert np.array_equal(ranks < 1, ties < 0)
    if rec["method"] == "LAST":
        assert np.all(ranks[(ranks > 0) & (ties == 0)] <= L)
    return rec


def run_last(tmp):
    bk = _basekit()
    from v4base import run as BR
    tp, ep = bk.FixtureTrainProvider(n=80, K=bk.K, seed=0), bk.FixtureEvalProvider(n=60, K=bk.K, seed=1)
    out = BR.execute("LAST", train_provider=tp, eval_provider=ep, key_base=bk.MINIMAL_KEY_BASE, out_dir=tmp, K=bk.K)
    return out, ep, bk


def test_baseline_run_writes_rank_files(tmp_path):
    out, ep, bk = run_last(tmp_path)
    rec = check_rank_files(tmp_path, ep.manifest, bk.L)
    assert rec["format"] == "PPSI_V4_BASELINE_RANKS_1" and rec["eval_manifest_id"] == ep.manifest.manifest_id
    assert out["receipt"]["ranks"]["ranks_sha256"] == rec["ranks_sha256"]
    rec_text = (tmp_path / "RANKS_RECORD.json").read_text(encoding="utf-8")
    assert '"value"' not in rec_text and "mrr" not in rec_text                           # identity + sha256 only
    from v4base import run as BR
    from v4base import scoring as S
    from v4base.last import LastItemScorer
    from v4eval import frozen_config as fc
    ep2 = bk.FixtureEvalProvider(n=60, K=bk.K, seed=1)
    cfg = fc.load()
    old = S.run_baseline_evaluation(LastItemScorer(bk.K), ep2.manifest, ep2.batches(cfg.eval_batch),
                                    model_sha256=S.v4base_code_sha256(), config=cfg)
    ep3 = bk.FixtureEvalProvider(n=60, K=bk.K, seed=1)
    new, art = BR.evaluate_with_ranks(LastItemScorer(bk.K), ep3.manifest, ep3.batches(cfg.eval_batch),
                                      model_sha256=S.v4base_code_sha256(), config=cfg)
    assert json.dumps(old, sort_keys=True, default=str) == json.dumps(new, sort_keys=True, default=str)
    assert np.array_equal(np.load(tmp_path / "ranks_int32.npy"), np.asarray(art.ranks, dtype=np.int32))


@nc("a rank-file writer that saves the ranks in another row order than the manifest")
def test_nc_rank_file_row_order(tmp_path, monkeypatch):
    from v4base import run as BR
    real = BR.write_rank_files

    def reversed_rows(out_dir, art, manifest, **kw):                                  # MUTANT
        art = SimpleNamespace(decision_id=art.decision_id, ranks=np.asarray(art.ranks)[::-1].copy(),
                              n_tied=np.asarray(art.n_tied)[::-1].copy())
        return real(out_dir, art, manifest, **kw)
    monkeypatch.setattr(BR, "write_rank_files", reversed_rows)
    _out, ep, bk = run_last(tmp_path)
    check_rank_files(tmp_path, ep.manifest, bk.L)


# ================================================================================================ C4: ANA_CONTRASTS_v3
def check_v3_file(path: Path) -> dict:
    got = R.load_contrasts(path, split="TEST")
    assert got["format"] == "PPSI_V4_ANA_CONTRASTS_3" and [c["id"] for c in got["items"]] == V3_ORDER
    assert [c["id"] for c in got["descriptive_items"]] == ["S8-S7"]
    obj = json.loads(path.read_text(encoding="utf-8"))
    by = {c["id"]: c for c in obj["confirmatory_contrasts"]}
    assert by["K11b"]["contrast"] == "FA − LAST" and by["K9c"]["contrast"] == "FT_PRE − PRE"
    assert by["K9b"]["contrast"] == "FT_C − C_SMALL" and by["K9b"]["population"] == "UNSEEN_HOLDOUT"
    assert by["K20b"]["contrast"] == "R(FA_WARM / C_FULL)" and by["K20c"]["contrast"] == "R(PRE / C_FULL)"
    assert obj["registered_descriptive_contrasts"][0]["contrast"] == "FA_WARM_FROZEN − FA_WARM"
    assert set(obj["hierarchy"]["secondary"]) == {"K1", "K3", "K8", "K12", "K17", "K20", "K21", "K11b", "K9b", "K9c",
                                                  "K20b", "K20c"} and obj["hierarchy"]["primary"] == ["K2"]
    assert obj["registered_seeds"]["K9b"] == [2026] and obj["registered_seeds"]["K20c"] == [2026]
    assert "K20b" not in obj["registered_seeds"] and "K11b" not in obj["registered_seeds"]       # default 2026 + 2027
    deps = {d["id"]: d["of"] for d in obj["algebraic_dependencies"]}
    assert deps["K3"] == ["K2", "K1"] and all(deps[k][-1] == "K15" for k in ("K17", "K18", "K19"))
    assert obj["reading_rules"]["K14"]["interval"] == [99.0, 101.0]
    assert "co-headline" not in path.read_text(encoding="utf-8").lower()
    assert obj["multiplicity_statement"].startswith("26 pre-registered contrasts")
    assert got["self_sha256_status"] == "VERIFIED"
    for k in V3_ORDER[:21]:                                                          # K1-K21 exactly as v2 / 06K
        assert (by[k]["contrast"], by[k]["question"]) == R.registered_items()[k]
    return obj


def test_v3_registry_file_matches_the_amendment():
    check_v3_file(V3_FILE)


def check_v3_deviations(tmp_path):
    p = tmp_path / "v3.json"

    def refused(obj, needle):                                    # plain asserts (the NC below must fail by assertion)
        anakit.jdump(p, obj)
        try:
            R.load_contrasts(p)
        except R.AnaRefused as e:
            assert needle in str(e), str(e)
            return
        raise AssertionError(f"not refused ({needle})")
    base = anakit.contrasts_obj(3)
    for key, edit, needle in (
            ("populations", lambda o: o["populations"].update(K9b="MAIN"), "populations"),
            ("pairing", lambda o: o["pairing_notes"].update(K20b="seed-matched"), "pairing_notes"),
            ("deps", lambda o: o["algebraic_dependencies"].pop(), "algebraic_dependencies"),
            ("margin", lambda o: o["reading_rules"]["K14"].update(interval=[98.0, 102.0]), "reading_rules"),
            ("desc", lambda o: o.update(registered_descriptive_contrasts=[]), "registered_descriptive"),
            ("k11b", lambda o: _item(o, "K11b").update(question="x"), "differs"),
            ("coh", lambda o: o.update(note="K17 is the co-headline"), "co-headline"),
            ("hier", lambda o: [o["hierarchy"]["secondary"].remove("K11b"), o["hierarchy"]["tertiary"].append("K11b")],
             "hierarchy"),
            ("seeds", lambda o: o["registered_seeds"].update(K9b=[2026, 2027]), "registered_seeds")):
        obj = json.loads(json.dumps(base))
        edit(obj)
        refused(obj, needle)


def _item(obj, kid):
    return next(c for c in obj["confirmatory_contrasts"] if c["id"] == kid)


def test_v3_deviations_refused(tmp_path):
    check_v3_deviations(tmp_path)


@nc("a v3 loader that checks the texts but not the structured 06L-C fields")
def test_nc_v3_structured_fields_ignored(tmp_path, monkeypatch):
    real = R.v3_registry_content
    monkeypatch.setattr(R, "v3_registry_content", lambda: {k: v for k, v in real().items()
                                                           if k == "registered_descriptive_contrasts"})    # MUTANT
    check_v3_deviations(tmp_path)


def test_embedded_06lc_text_matches_the_contract():
    text = _norm(AMEND.read_text(encoding="utf-8"))
    for key, t in R.REGISTERED_06LC.items():
        assert _norm(t) in text, key
    assert "POST_VALIDATION_EXPLORATORY" in text and R.SECAGG_NOT_SIMULATED in text


# ================================================================================================ C3: LAST anchor, K11b
def check_anchor(blk, res):
    L = last_v(blk)
    fa = _anchor(res, "FA")
    want = np.mean([pbv(blk, "FA", s) - L for s in (2026, 2027)])
    assert fa["delta_vs_LAST"]["point"] == pytest.approx(want, abs=1e-12)
    assert fa["BELOW_DEVICE_HEURISTIC"] is False and fa["delta_vs_LAST"]["comparison"] == "BASELINE_CPU_EXACT_RANKS"
    rstar = np.mean([100.0 * (pbv(blk, "FA", s) - L) / (pbv(blk, "C_BIG", s) - L) for s in (2026, 2027)])
    assert fa["headroom_vs_C_BIG"]["point"] == pytest.approx(rstar, abs=1e-9)
    lo, hi = fa["headroom_vs_C_BIG"]["ci95"]
    assert lo <= fa["headroom_vs_C_BIG"]["point"] <= hi
    lo_row = _anchor(res, "LO")                                   # LO (C_BIG + 10) is below LAST (C_BIG + 6)
    assert lo_row["delta_vs_LAST"]["ci95"][1] < 0 and lo_row["BELOW_DEVICE_HEURISTIC"] is True
    k11b = conf(res, "K11b")
    assert k11b["label"] == "CONFIRMATORY" and k11b["seeds"] == [2026, 2027] and k11b["seeds_status"] == "COMPLETE"
    assert k11b["point"] == pytest.approx(want, abs=1e-12)
    assert [p["b"] for p in k11b["per_seed"]] == ["LAST|None|RULE_FREE|MAIN"] * 2
    assert _rec(res, "K11b")["hierarchy"] == "secondary"
    warm = _anchor(res, "FA_WARM")                                # deployment arms also against C_FULL s2026
    cf = pbv(blk, "C_FULL", 2026)
    assert warm["headroom_vs_C_FULL"]["point"] == pytest.approx(
        np.mean([100.0 * (pbv(blk, "FA_WARM", s) - L) / (cf - L) for s in (2026, 2027)]), abs=1e-9)


def test_last_anchor_k11b_and_headroom_known_answers(full):
    blk, _out, res = full
    check_anchor(blk, res)


@nc("the ANA-03 pairing: a seed-free LAST denominator never pairs (no anchor), so K11b and Δ(arm − LAST) vanish")
def test_nc_no_last_anchor(tmp_path, monkeypatch):
    monkeypatch.setattr(R, "is_anchor", lambda e: False)                            # MUTANT
    blk = full_block(tmp_path / "b", calib=False)
    rc, text = run(blk, tmp_path / "o")
    assert rc == 0, text
    check_anchor(blk, result(tmp_path / "o"))


def test_device_envelope(full):
    blk, _out, res = full
    env = res["device_only_envelope"]
    rows = {r["arm"]: r for r in env["rows"]}
    assert list(rows) == ["LO", "LAST", "RULE"]
    assert rows["LAST"]["value"] == pytest.approx(last_v(blk), abs=1e-12)
    assert rows["LO"]["value"] == pytest.approx(np.mean([pbv(blk, "LO", s) for s in (2026, 2027)]), abs=1e-12)
    assert env["best_point"]["arm"] == "LAST"                     # C_BIG + 6 beats RULE (+9) and LO (+10)
    assert rows["RULE"]["delta_vs_LAST"]["point"] == pytest.approx(macro(blk.ranks[(M, "RULE", None, "RULE_FREE")],
                                                                         blk.manifests[M]) - last_v(blk), abs=1e-12)


def test_k11b_not_available_without_the_last_rank_file(tmp_path):
    blk = anakit.build(tmp_path)
    anakit.jdump(blk.contrasts, anakit.contrasts_obj(3))
    rc, text = run(blk, tmp_path / "o")
    assert rc == 0, text
    res = result(tmp_path / "o")
    k = conf(res, "K11b")
    assert k["status"] == "NOT_AVAILABLE" and "rank file" in k["reason"] and k["seeds_status"] == "NOT_RUN"
    assert res["last_anchor"]["status"] == "NOT_AVAILABLE" and res["target_strata"]["status"] == "NOT_AVAILABLE"


def check_baseline_block_refusals(tmp_path):
    b = full_block(tmp_path / "lp", calib=False)
    _set_rows_field(b, "MAIN/BASE_LAST/rows.json", "label_policy_hash", "x" * 64)
    check_refuses(b, tmp_path / "lp", "BASELINE_CPU row label_policy_hash")
    b = full_block(tmp_path / "qh", calib=False)
    _set_rows_field(b, "MAIN/BASE_LAST/rows.json", "query_order_hash", "q" * 64)
    check_refuses(b, tmp_path / "qh", "is not on the loaded")


def test_baseline_block_key_checks(tmp_path):
    check_baseline_block_refusals(tmp_path)
    b = full_block(tmp_path / "hist", calib=False)                # a ranked baseline declared historical: refused
    edit_index(b, lambda idx: [e.__setitem__("historical_reference", True) for e in idx["entries"]
                               if e.get("comparison_block") == "BASELINE_CPU" and e["arm"] == "RULE"])
    check_refuses(b, tmp_path / "hist", "never a historical reference")
    b = full_block(tmp_path / "ties", calib=False)
    edit_index(b, lambda idx: [e.__setitem__("row_ties", "MAIN/BASE_LAST/n_tied_int32.npy") for e in idx["entries"]
                               if e["arm"] == "FP"])
    check_refuses(b, tmp_path / "ties", "row_ties")


@nc("the BASELINE_CPU rows are never key-checked against the block")
def test_nc_baseline_block_unchecked(tmp_path, monkeypatch):
    monkeypatch.setattr(R, "check_baseline_block", lambda *a, **k: None)             # MUTANT
    check_baseline_block_refusals(tmp_path)


# ================================================================================================ C15: repeat / novel strata
def check_target_strata(blk, res):
    m = blk.manifests[M]
    last = blk.ranks[(M, "LAST", None, "RULE_FREE")]
    ties = anakit.last_ties(last)
    rep = m.eval_mask & (last > 0) & (ties == 0)
    ts = res["target_strata"]
    assert ts["counts"]["REPEAT_rows"] == int(rep.sum()) and ts["counts"]["NOVEL_rows"] == int((m.eval_mask & ~rep).sum())
    for name, mask in (("REPEAT", rep), ("NOVEL", m.eval_mask & ~rep)):
        row = next(r for r in ts["rows"] if r["stratum"] == name and r["arm"] == "FA")
        want = np.mean([100.0 * mmacro(rk(blk, "FA", s), m, mask) / mmacro(rk(blk, "C_BIG", s), m, mask)
                        for s in (2026, 2027)])
        assert row["R_vs_C_BIG"] == pytest.approx(want, abs=1e-9)
        assert row["n_decisions"] == int(mask.sum())


def test_repeat_novel_target_strata_known_answers(full):
    blk, _out, res = full
    check_target_strata(blk, res)


@nc("a repeat flag that ignores LAST's tie count (every RANKABLE row counted as a repeat)")
def test_nc_repeat_flag_without_ties(tmp_path, monkeypatch):
    real = R.check_row_ties

    def all_repeat(e, manifest):                                                   # MUTANT
        real(e, manifest)
        e.ties = np.where(e.ranks > 0, 0, -1)
    monkeypatch.setattr(R, "check_row_ties", all_repeat)
    blk = full_block(tmp_path / "b", calib=False)
    rc, text = run(blk, tmp_path / "o")
    assert rc == 0, text
    check_target_strata(blk, result(tmp_path / "o"))


def test_last_tie_file_consistency(tmp_path):
    b = full_block(tmp_path / "t", calib=False)
    p = b.root / "MAIN" / "BASE_LAST" / "n_tied_int32.npy"
    t = np.load(p)
    t[t == 0] = -1                                                # -1 where the rank is >= 1: not the same artifact
    np.save(p, t)
    check_refuses(b, tmp_path / "t", "row_ties is -1 on other rows")
    b = full_block(tmp_path / "r", calib=False)
    rp = b.root / "MAIN" / "BASE_LAST" / "ranks_int32.npy"
    ranks = np.load(rp)
    ties = np.load(b.root / "MAIN" / "BASE_LAST" / "n_tied_int32.npy")
    i = int(np.flatnonzero((ties == 0) & (ranks > 0))[0])
    ranks[i] = 51                                                 # a "window" row ranked below 50 items
    np.save(rp, ranks)
    anakit.jdump(b.root / "MAIN" / "BASE_LAST" / "rows.json",   # rows re-derived: only the tie check can refuse
                 anakit.rows_for("LAST", None, None, ranks.astype(np.int64), b.manifests[M], device=anakit.BASE_DEV,
                                 carry_rule=False, cid=None, extra=anakit.BASE_EXTRA))
    check_refuses(b, tmp_path / "r", "has a LAST rank > 50")


# ================================================================================================ C14: MAIN \ CALIB
def check_calib(blk, res):
    m = blk.manifests[M]
    cd = res["calib_disjoint"]
    assert cd["status"] == "COMPUTED"
    assert cd["counts"] == {"manifest": {"main_users": 48, "calib_users": 16, "disjoint_users": 32},
                            "e2e": {"main_users": 48, "calib_users": 16, "disjoint_users": 32}}
    calib_users = np.arange(anakit.CALIB_OFFSET, anakit.CALIB_OFFSET + anakit.CALIB_USERS)
    mask = m.eval_mask & ~np.isin(m.eval_user_key, calib_users)
    fa = next(a for a in cd["arms"] if a["arm"] == "FA")
    want = np.mean([100.0 * mmacro(rk(blk, "FA", s), m, mask) / mmacro(rk(blk, "C_BIG", s), m, mask)
                    for s in (2026, 2027)])
    assert fa["R_vs_C_BIG_disjoint"]["point"] == pytest.approx(want, abs=1e-9)
    assert fa["R_vs_C_BIG_disjoint"]["n_users"] == 32
    ids = {c["id"] for c in cd["contrasts"]}
    assert {"K2", "K3", "K11b", "K20", "K20b"} <= ids and "K15" not in ids and "K9b" not in ids
    k2 = next(c for c in cd["contrasts"] if c["id"] == "K2")["stat"]
    assert k2["point"] == pytest.approx(want, abs=1e-9) and k2["label"] == "DESCRIPTIVE"


def test_calib_disjoint_known_answers(full):
    blk, _out, res = full
    check_calib(blk, res)


@nc("MAIN \\ CALIB joined on eval_user_key (a per-manifest dense key) instead of client_key_hex")
def test_nc_calib_join_by_eval_user_key(tmp_path, monkeypatch):
    real = R.load_calib_context

    def by_euk(context_loader, ctx, split):                                        # MUTANT
        out = real(context_loader, ctx, split)
        c = context_loader("CALIBRATION", {})
        users = R._euks(ctx["MAIN"])
        keys = users[~np.isin(users, np.unique(c.manifest.eval_user_key))]
        return dict(out, disjoint_keys=keys.astype(np.int64))
    monkeypatch.setattr(R, "load_calib_context", by_euk)
    blk = full_block(tmp_path / "b")
    rc, text = run(blk, tmp_path / "o")
    assert rc == 0, text
    check_calib(blk, result(tmp_path / "o"))


def test_calib_counts_enforced_for_the_production_context(tmp_path):
    b = full_block(tmp_path / "e")
    b.ctx["CALIBRATION"].record["enforce_registered_counts"] = True
    check_refuses(b, tmp_path / "e", "registered")


def _patch_registered(monkeypatch, manifest=None, e2e=None):
    monkeypatch.setattr(R, "CALIB_REGISTERED_COUNTS",
                        manifest or {"main_users": 48, "calib_users": 16, "disjoint_users": 32})
    monkeypatch.setattr(R, "CALIB_E2E_REGISTERED", e2e or {"main_users": 48, "calib_users": 16})


def test_calib_registered_counts_pass_when_they_match(tmp_path, monkeypatch):
    _patch_registered(monkeypatch)
    b = full_block(tmp_path / "ok")
    b.ctx["CALIBRATION"].record["enforce_registered_counts"] = True
    rc, text = run(b, tmp_path / "o")
    assert rc == 0, text
    check_calib(b, result(tmp_path / "o"))
    assert "32 CALIB-disjoint users, of whom 32 have" in (tmp_path / "o" / "REPORT.md").read_text(encoding="utf-8")


def test_nc_wrong_manifest_counts_refuse(tmp_path, monkeypatch):
    _patch_registered(monkeypatch, manifest={"main_users": 51211, "calib_users": 12715, "disjoint_users": 38496})
    b = full_block(tmp_path / "wm")
    b.ctx["CALIBRATION"].record["enforce_registered_counts"] = True
    check_refuses(b, tmp_path / "wm", "manifest user counts")


def test_nc_wrong_e2e_pins_refuse(tmp_path, monkeypatch):
    _patch_registered(monkeypatch, e2e={"main_users": 40764, "calib_users": 10070})
    b = full_block(tmp_path / "we")
    b.ctx["CALIBRATION"].record["enforce_registered_counts"] = True
    check_refuses(b, tmp_path / "we", "E2E user counts")


def test_nc_calib_user_missing_from_main_refuses(tmp_path):
    b = full_block(tmp_path / "ms")
    b.ctx["CALIBRATION"].hex_by_euk[0] = "not_a_main_user_hex"
    check_refuses(b, tmp_path / "ms", "not a subset of MAIN")


def test_no_calibration_context_is_not_available(tmp_path):
    b = full_block(tmp_path / "n", calib=False)
    rc, text = run(b, tmp_path / "o")
    assert rc == 0, text
    assert result(tmp_path / "o")["calib_disjoint"]["status"] == "NOT_AVAILABLE"


# ================================================================================================ C13: seeds and Holm
def _k2_boot(blk):
    m = blk.manifests[M]
    lib = R.LIB
    plan = lib.BC._plan(m)
    arr = []
    for s in (2026, 2027):
        a = lib.B.resampled(plan, lib.BC._stats(m, rk(blk, "FA", s), "fa"), "macro_e2e")
        c = lib.B.resampled(plan, lib.BC._stats(m, rk(blk, "C_BIG", s), "cb"), "macro_e2e")
        arr.append(100.0 * a / c)
    return np.mean(arr, axis=0)


def test_seed_statistics_beside_every_k(full):
    blk, _out, res = full
    k2 = conf(res, "K2")
    ss = k2["seed_stats"]
    r = [100.0 * pbv(blk, "FA", s) / pbv(blk, "C_BIG", s) for s in (2026, 2027)]
    assert ss["seed_spread"] == pytest.approx(abs(r[0] - r[1]), abs=1e-9)
    half = (k2["ci95"][1] - k2["ci95"][0]) / 2
    assert ss["seed_sensitive"] is bool(abs(r[0] - r[1]) > half)
    lo, hi = ss["user_seed_interval"]
    assert lo < k2["point"] < hi
    assert conf(res, "K13", CB)["seed_stats"]["status"] == "SINGLE_SEED"
    for c in res["confirmatory"]:
        b = c["by_rule"][c["confirmatory_rule"]]
        if "point" in b:
            assert "seed_stats" in b and "claim" in b


def check_holm(blk, res, m_family=26):
    hb = res["holm_band"]
    assert hb["family_size"] == m_family and hb["family"] == V3_ORDER and 0 < hb["n_computed"] < m_family
    k2 = conf(res, "K2")["holm"]
    se = float(np.std(_k2_boot(blk), ddof=1))
    assert k2["se_bootstrap"] == pytest.approx(se, rel=1e-12)
    comp = sorted((c["by_rule"][c["confirmatory_rule"]]["holm"] for c in res["confirmatory"]
                   if c["by_rule"][c["confirmatory_rule"]].get("holm", {}).get("status") == "COMPUTED"),
                  key=lambda h: h["holm_rank"])
    assert [h["holm_rank"] for h in comp] == list(range(1, len(comp) + 1))
    for h in comp:
        assert h["holm_level"] == pytest.approx(0.05 / (m_family - h["holm_rank"] + 1))
        assert h["bonferroni_band"][0] <= h["holm_band"][0] and h["holm_band"][1] <= h["bonferroni_band"][1]
        lo, hi = h["holm_band"]                                   # FIX-DELTA-03 M3: the band agrees with the decision
        assert (not (lo <= h["reference"] <= hi)) == h["holm_reject"], h
    assert all(a["p_holm"] <= b["p_holm"] for a, b in zip(comp, comp[1:]))


def test_holm_band_beside_the_unadjusted_cis(full):
    blk, out, res = full
    check_holm(blk, res)
    rep = (out / R.REPORT_NAME).read_text(encoding="utf-8")
    assert "Holm band" in rep.split("## 2.")[0] and "### 1c. Per-K statistics" in rep


@nc("a Holm family that shrinks to the computed contrasts (a missing K would make the band narrower)")
def test_nc_holm_family_shrinks(tmp_path, monkeypatch):
    real = R.holm_band

    def shrunk(eng, conf_, contrasts, alpha=0.05):                                 # MUTANT
        n = sum(1 for c in conf_ if (eng.boot.get((c["id"], c["confirmatory_rule"])) is not None))
        return real(eng, conf_, dict(contrasts, n_contrasts=n), alpha)
    monkeypatch.setattr(R, "holm_band", shrunk)
    blk = full_block(tmp_path / "b", calib=False)
    rc, text = run(blk, tmp_path / "o")
    assert rc == 0, text
    check_holm(blk, result(tmp_path / "o"))


# ================================================================================================ C15: harmed share, K14
def check_harmed_ratio(res):
    k1 = conf(res, "K1")                                          # C_SMALL = 2 x C_BIG ranks: nobody improves
    assert k1.get("improved_share") == 0.0 and k1.get("harmed_share", 0.0) > 0.5
    assert k1["harmed_share"] + k1["tied_share"] == pytest.approx(1.0)
    assert set(k1["per_user_delta_quantiles"]) == {"P10", "P25", "P50", "P75", "P90"}
    null = k1["harmed_share_seed_null"]["numerator"]
    assert null["arm"] == "C_SMALL" and null["seeds"] == [2027, 2026]
    assert null["harmed_share"] + null["improved_share"] + null["tied_share"] == pytest.approx(1.0)
    assert "harmed_share" in conf(res, "K2") and "harmed_share" in conf(res, "K20b")


def test_harmed_share_for_every_k_with_a_seed_null(full):
    _blk, _out, res = full
    check_harmed_ratio(res)


@nc("the ANA-03 harmed share: deltas only (ratio contrasts print '-')")
def test_nc_harmed_share_only_for_deltas(tmp_path, monkeypatch):
    real = R.ratio

    def no_harm(*a, **k):                                                          # MUTANT
        out = real(*a, **k)
        for key in ("harmed_share", "improved_share", "tied_share", "per_user_delta_quantiles"):
            out.pop(key, None)
        return out
    monkeypatch.setattr(R, "ratio", no_harm)
    blk = full_block(tmp_path / "b", calib=False)
    rc, text = run(blk, tmp_path / "o")
    assert rc == 0, text
    check_harmed_ratio(result(tmp_path / "o"))


def check_k14_rule():
    eng = SimpleNamespace(boot={})
    rng = np.random.default_rng(3)
    eng.boot[("K14", PB)] = 100.0 + rng.normal(0.0, 0.1, 1000)                       # tight around 100: equivalent
    e1 = R.k14_equivalence(eng, ("K14", PB), {"ci95": [99.8, 100.2]})
    assert e1["equivalent"] is True and e1["margin"] == [99.0, 101.0]
    x = np.concatenate([np.full(40, 98.5), np.full(920, 100.0), np.full(40, 101.5)])  # 90% inside, 95% outside
    eng.boot[("K14", PB)] = x
    e2 = R.k14_equivalence(eng, ("K14", PB), {"ci95": R.LIB.B.ci95(x)})
    assert e2["equivalent"] is True and e2["ci95_inside_margin"] is False
    eng.boot[("K14", PB)] = 97.0 + rng.normal(0.0, 0.1, 1000)
    assert R.k14_equivalence(eng, ("K14", PB), {"ci95": [96.8, 97.2]})["equivalent"] is False


def test_k14_equivalence_margin(full):
    check_k14_rule()
    _blk, _out, res = full
    eq = conf(res, "K14")["equivalence"]
    assert eq["status"] == "COMPUTED" and eq["ci90"][0] >= conf(res, "K14")["ci95"][0] - 1e-12
    assert eq["equivalent"] is (99.0 <= eq["ci90"][0] and eq["ci90"][1] <= 101.0)
    assert "equivalence" not in conf(res, "K14", CB)


@nc("a K14 equivalence read on the 95% interval (TOST at alpha 0.05 is the 90% interval)")
def test_nc_k14_equivalence_on_the_95_interval(monkeypatch):
    real = R.k14_equivalence

    def on95(eng, key, b):                                                         # MUTANT
        out = real(eng, key, b)
        if out.get("status") == "COMPUTED":
            lo, hi = R.LIB.B.ci95(eng.boot[key])
            out["equivalent"] = bool(99.0 <= lo and hi <= 101.0)
        return out
    monkeypatch.setattr(R, "k14_equivalence", on95)
    check_k14_rule()


# ================================================================================================ C12: claims
_NUM = re.compile(r"\d+\.\d+|\d+\s?%")


def number_free(text: str) -> bool:
    """No result number outside the {placeholders} (design constants such as the '95% CI' level are not results)."""
    text = re.sub(r"\b(90|95)% (CI|paired|interval)", "", re.sub(r"\{[^{}]*\}", "", text))
    return not _NUM.search(text)


def check_claims(tmp_path):
    lines = []
    rc = R.main(["--claims-only", "--contrasts", str(V3_FILE), "--out", str(tmp_path / "claims")], say=lines.append)
    assert rc == 0, lines
    reg = json.loads((tmp_path / "claims" / "CLAIM_REGISTER.json").read_text(encoding="utf-8"))
    head = (tmp_path / "claims" / "HEADLINE_TEMPLATE.md").read_text(encoding="utf-8")
    assert R.verify_self_sha(reg) and reg["format"] == "PPSI_V4_ANA_CLAIM_REGISTER_1"
    assert [c["id"] for c in reg["contrasts"]] == V3_ORDER + ["S8-S7"]
    for c in reg["contrasts"]:
        assert set(c["allowed"]) == {"ABOVE", "BELOW", "INCLUDES", "NOT_AVAILABLE", "UNDEFINED"}
        assert all(number_free(t) for t in c["allowed"].values()), c["id"]
        assert c["forbidden"], c["id"]
    by = {c["id"]: c for c in reg["contrasts"]}
    assert by["S8-S7"]["tier"] == "descriptive" and by["K2"]["tier"] == "primary" and by["K11b"]["tier"] == "secondary"
    assert any("lower bound" in f for f in by["K2"]["forbidden"]) and any("K15" in f for f in by["K17"]["forbidden"])
    assert number_free(head) and "{K2}" in head and "{K11b}" in head and "{K20b}" in head
    assert all(number_free(r) for r in reg["general_rules"])
    return reg


def test_claim_register_and_number_free_headline(tmp_path):
    check_claims(tmp_path)


@nc("a headline template that carries a result number")
def test_nc_headline_with_a_number(tmp_path, monkeypatch):
    monkeypatch.setattr(R, "HEADLINE_TEMPLATE", R.HEADLINE_TEMPLATE + "\nFL retains 25.0 % of the cloud.\n")  # MUTANT
    check_claims(tmp_path)


def test_claim_keys_stamped_on_every_k(full):
    _blk, out, res = full
    for c in res["confirmatory"] + res["registered_descriptive"]:
        b = c["by_rule"][c["confirmatory_rule"]]
        ref = 100.0 if c["kind"] == "retained" else 0.0
        key = b["claim"]["template_key"]
        if b.get("status") == "NOT_AVAILABLE":
            assert key == "NOT_AVAILABLE"
        elif isinstance(b.get("ci95"), list):
            lo, hi = b["ci95"]
            assert key == ("ABOVE" if lo > ref else "BELOW" if hi < ref else "INCLUDES"), c["id"]
    assert (out / "CLAIM_REGISTER.json").is_file() and (out / "HEADLINE_TEMPLATE.md").is_file()
    man = json.loads((out / R.MANIFEST_NAME).read_text(encoding="utf-8"))
    assert {"CLAIM_REGISTER.json", "HEADLINE_TEMPLATE.md"} <= set(man["files_sha256"])


def test_claims_only_needs_no_block_and_refuses_extra_options(tmp_path):
    lines = []
    assert R.main(["--claims-only", "--contrasts", str(V3_FILE), "--out", str(tmp_path / "x"), "--split", "VALIDATION"],
                  say=lines.append) == R.REFUSED
    with pytest.raises(SystemExit):
        R.parse_args(["--contrasts", str(V3_FILE), "--out", str(tmp_path / "y")])       # a report needs --block


# ================================================================================================ C8 / C1: bytes and privacy
def test_bytes_axis_and_privacy_matrix(full):
    _blk, _out, res = full
    axis = {(r["arm"], r["step"]): r for r in res["curves"]["bytes_axis"]}
    fa = axis[("FA", 3)]
    assert fa["bytes_per_client"] == anakit.FA_PER_CLIENT
    assert fa["bytes_per_client_incl_final_download"] == anakit.FA_PER_CLIENT + anakit.FA_DOWN_BYTES
    assert axis[("FA_WARM_FROZEN", 4)]["bytes_per_client"] == pytest.approx(GB.S8_INCL)
    raw = next(r for r in res["curves"]["bytes_axis"] if r["step"] == 1)
    assert raw["bytes_per_client"] is None and raw["status"].startswith("NOT_REGISTERED")
    assert axis[("C_SMALL", 2)]["bytes_per_client"] == 82_840_192 and axis[("LAST", 0)]["bytes_per_client"] == 0
    steps = [r["step"] for r in res["curves"]["bytes_axis"]]
    assert steps == sorted(steps)                                  # raw events -> download -> FL -> S8 on one axis
    mat = {lv["level"]: lv for lv in res["privacy_matrix"]}
    arms_at = {a["arm"]: lv for lv in res["privacy_matrix"] for a in lv["arms"]}
    assert arms_at["LAST"]["level"] == "L6 device-only" and arms_at["LO"]["level"] == "L6 device-only"
    assert arms_at["FT_C"]["level"] == "L1 base (server-trained on raw events) + local fine-tune"
    assert arms_at["FT_PRE"]["level"].startswith("L5") and arms_at["PRE"]["level"].startswith("L1 on opted-in")
    assert arms_at["RULE"]["level"] == "device + published aggregate"
    assert arms_at["FA"]["secure_aggregation"] == "secure aggregation: not simulated"
    assert "265,162" in arms_at["FA_WARM_FROZEN"]["server_sees_training"]
    assert "trusted aggregator" in mat["L3 federated + DP"]["formal_guarantee"]
    assert next(a for a in arms_at["FA"]["arms"] if a["arm"] == "FA")["bytes_per_client"] == anakit.FA_PER_CLIENT


def test_event_size_record(tmp_path):
    b = anakit.build(tmp_path / "a")
    anakit.jdump(b.root / "records" / "EVENT_SIZE.json", anakit.canon({"bytes_per_event": 64, "rule": "synthetic",
                                                                        "decision_log_row": 999}))
    edit_index(b, lambda idx: idx["records"].update(EVENT_SIZE="records/EVENT_SIZE.json"))
    rc, text = run(b, tmp_path / "a" / "o")
    assert rc == 0, text
    raw = next(r for r in result(tmp_path / "a" / "o")["curves"]["bytes_axis"] if r["step"] == 1)
    sup = [anakit.SUPPORT_CYCLE[k % len(anakit.SUPPORT_CYCLE)] for k in range(anakit.ROLES["MAIN"][2])]
    assert raw["bytes_per_client"] == pytest.approx(64 * np.mean(sup)) and raw["status"] == "COMPUTED"
    b2 = anakit.build(tmp_path / "b")
    anakit.jdump(b2.root / "records" / "EVENT_SIZE.json", {"bytes_per_event": 64})
    edit_index(b2, lambda idx: idx["records"].update(EVENT_SIZE="records/EVENT_SIZE.json"))
    check_refuses(b2, tmp_path / "b", "EVENT_SIZE")


def s8_s7_block(tmp):
    blk = anakit.build(tmp)
    anakit.jdump(blk.contrasts, anakit.contrasts_obj(3))
    add_fl(blk, "FA_WARM")
    GB.add_s8(blk)
    return blk


def check_s8_s7(blk, res):
    d = _rec(res, "S8-S7")
    b = d["by_rule"][PB]
    want = np.mean([pbv(blk, "FA_WARM_FROZEN", s) - pbv(blk, "FA_WARM", s) for s in (2026, 2027)])
    assert b["point"] == pytest.approx(want, abs=1e-12) and b["seeds"] == [2026, 2027] and b["label"] == "DESCRIPTIVE"
    assert d["label"] == "DESCRIPTIVE" and "S8-S7" not in res["labels"]["CONFIRMATORY"]
    br = d["bytes_ratio"][PB]
    for s in ("2026", "2027"):
        assert br["per_seed"][s]["ratio"] == pytest.approx(GB.S8_INCL / anakit.FA_PER_CLIENT)
    assert br["mean_ratio"] == pytest.approx(GB.S8_INCL / anakit.FA_PER_CLIENT)


def test_s8_minus_s7_with_the_bytes_ratio(tmp_path):
    blk = s8_s7_block(tmp_path / "b")
    rc, text = run(blk, tmp_path / "o")
    assert rc == 0, text
    res = result(tmp_path / "o")
    check_s8_s7(blk, res)
    assert res["labels"]["registered_descriptive"] == ["S8-S7"]


@nc("S8's bytes without the frozen tables (the pre-B3 per_client mean) in the S8 / S7 ratio")
def test_nc_s8_s7_ratio_without_the_frozen_tables(tmp_path, monkeypatch):
    monkeypatch.setattr(R, "frozen_once_meta", lambda summ, label: {})                # MUTANT
    blk = s8_s7_block(tmp_path / "b")
    rc, text = run(blk, tmp_path / "o")
    assert rc == 0, text
    check_s8_s7(blk, result(tmp_path / "o"))


# ================================================================================================ arms: FT_PRE, K9b, S8, ...
def check_ft_pre(tmp_path):
    blk = full_block(tmp_path / "a", calib=False)
    rc, text = run(blk, tmp_path / "a" / "o")
    assert rc == 0, text
    res = result(tmp_path / "a" / "o")
    k9c = conf(res, "K9c")
    want = np.mean([pbv(blk, "FT_PRE", s) - pbv(blk, "PRE", 2026) for s in (2026, 2027)])
    assert k9c["point"] == pytest.approx(want, abs=1e-12) and k9c["label"] == "CONFIRMATORY"
    assert [p["b"] for p in k9c["per_seed"]] == ["PRE|2026|PRACTICAL_BEST|MAIN"] * 2
    bad = full_block(tmp_path / "b", calib=False)                 # FT_PRE from another base
    edit_index(bad, lambda idx: [e.__setitem__("start_run_id", "C_SMALL_02B_s2026") for e in idx["entries"]
                                 if e["arm"] == "FT_PRE"])
    check_refuses(bad, tmp_path / "b", "FT_PRE's base must be θ_pre")
    swap = full_block(tmp_path / "c", calib=False)                # FT_C rows from theta_pre
    edit_index(swap, lambda idx: [e.__setitem__("start_run_id", anakit.PRE_RUN) for e in idx["entries"]
                                  if e["arm"] == "FT_C" and e["role"] == M])
    check_refuses(swap, tmp_path / "c", "a device run from θ_pre is FT_PRE")


def test_ft_pre_and_k9c(tmp_path):
    check_ft_pre(tmp_path)


@nc("FT_PRE admitted without its theta_pre base check")
def test_nc_ft_pre_base_unchecked(tmp_path, monkeypatch):
    monkeypatch.setattr(R, "ft_pre_base_problem", lambda *a: None)                   # MUTANT (DEV-FTPRE: the
    monkeypatch.setattr(R, "ft_pre_vs_pre_problem", lambda *a: None)                 # init-record checks removed)
    check_ft_pre(tmp_path)


def test_k9b_on_unseen_holdout(full):
    blk, _out, res = full
    k = conf(res, "K9b")
    r = "UNSEEN_HOLDOUT"
    assert _rec(res, "K9b")["population"] == r and k["seeds"] == [2026] and k["label"] == "CONFIRMATORY"
    want = pbv(blk, "FT_C", 2026, role=r) - pbv(blk, "C_SMALL", 2026, role=r)
    assert k["point"] == pytest.approx(want, abs=1e-12)
    assert k["population"] == f"{r}/ALL"
    assert _rec(res, "K9b")["extra_seeds"][PB]["seeds"] == [2026, 2027]              # seed 2027: DESCRIPTIVE


def test_dp8_frozen_labelled_row(tmp_path):
    blk = s8_s7_block(tmp_path / "a")
    add_fl(blk, "DP8_FROZEN", (2026,), patch=dp8_frozen_patch())
    rc, text = run(blk, tmp_path / "a" / "o")
    assert rc == 0, text
    res = result(tmp_path / "a" / "o")
    fb = res["frozen_branch"]
    assert fb["status"] == "COMPUTED" and fb["epsilon"] == 8.0 and fb["dp_z"] == 0.5642
    for rule in R.RULES:
        want = 100.0 * pbv(blk, "DP8_FROZEN", 2026, rule) / pbv(blk, "FA_WARM_FROZEN", 2026, rule)
        assert fb["by_rule"][rule]["point"] == pytest.approx(want, abs=1e-9)
        assert fb["by_rule"][rule]["seeds"] == [2026] and fb["by_rule"][rule]["label"] == "DESCRIPTIVE"
    meta = next(r for r in res["retention"]["MAIN"][PB] if r["arm"] == "DP8_FROZEN")["entries"]["2026"]["meta"]
    assert meta["bytes_per_client"] == pytest.approx(GB.S8_INCL) and meta["epsilon"] == 8.0
    assert any(a["arm"] == "DP8_FROZEN" for lv in res["privacy_matrix"] if lv["level"] == "L3 federated + DP"
               for a in lv["arms"])
    blk2 = s8_s7_block(tmp_path / "b")
    add_fl(blk2, "DP8_FROZEN", (2026,), patch=dp8_frozen_patch(z=1.3094))
    check_refuses(blk2, tmp_path / "b", "noise multiplier")


def test_gru_d128_family_comparison(full):
    blk, _out, res = full
    fc_ = res["family_comparison"]
    assert fc_["status"] == "COMPUTED" and fc_["label"] == "DESCRIPTIVE"
    for rule in R.RULES:
        b = fc_["by_rule"][rule]
        want = np.mean([100.0 * pbv(blk, "GRU_D128", s, rule) / pbv(blk, "C_SMALL", s, rule) for s in (2026, 2027)])
        assert b["R"]["point"] == pytest.approx(want, abs=1e-9)
        assert b["delta"]["point"] == pytest.approx(
            np.mean([pbv(blk, "GRU_D128", s, rule) - pbv(blk, "C_SMALL", s, rule) for s in (2026, 2027)]), abs=1e-12)
    assert "never a downstream family" in fc_["text"]


def check_unseen_val(tmp_path):
    blk = full_block(tmp_path / "a", calib=False)
    rc, text = run(blk, tmp_path / "a" / "o")
    assert rc == 0, text
    nf = result(tmp_path / "a" / "o")["no_ft_unseen_val"]
    row = next(r for r in nf["rows"] if r["base_arm"] == "C_SMALL")
    v = pbv(blk, "C_SMALL", 2026, role="UNSEEN_VAL")
    assert row["no_ft_value"] == pytest.approx(v, abs=1e-12) and row["selected_cell_value"] == 0.2
    assert row["selected_minus_no_ft"] == pytest.approx(0.2 - v, abs=1e-12) and row["label"] == "DESCRIPTIVE"
    lo, hi = row["no_ft_ci95"]
    assert lo <= v <= hi
    bad = full_block(tmp_path / "b", calib=False)
    anakit.add_plain(bad, "C_BIG", (2026,), (PB,), role="UNSEEN_VAL")
    check_refuses(bad, tmp_path / "b", "selection panel")


def test_no_ft_unseen_val_references(tmp_path):
    check_unseen_val(tmp_path)


@nc("any arm admitted on the UNSEEN_VAL selection panel")
def test_nc_unseen_val_any_arm(tmp_path, monkeypatch):
    monkeypatch.setattr(R, "NO_FT_ARMS", R.ARMS)                                      # MUTANT
    check_unseen_val(tmp_path)


def test_report_sections_and_descriptive_labels(full):
    _blk, out, res = full
    rep = (out / R.REPORT_NAME).read_text(encoding="utf-8")
    for head in ("## 8. LAST anchor", "## 9. Privacy matrix", "## 10. Labelled rows", "## 11. CALIB-disjoint",
                 "## 12. Claim register"):
        line = next(ln for ln in rep.splitlines() if ln.startswith(head))
        assert "DESCRIPTIVE" in line, line
    for text in R.REGISTERED_06LC.values():
        assert text in rep
    assert R.POST_VALIDATION_NOTE in rep

    def walk(o, path=""):
        if isinstance(o, dict):
            if "label" in o and not path.startswith("confirmatory"):
                assert o["label"] != "CONFIRMATORY", path
            for k, v in o.items():
                walk(v, f"{path}/{k}" if path else k)
        elif isinstance(o, list):
            for v in o:
                walk(v, path)
    walk(res)


def test_check_only_covers_the_new_pairings(tmp_path):
    blk = full_block(tmp_path / "a")
    rc, text = run(blk, tmp_path / "a" / "chk", "--check-only")
    assert rc == 0 and "PASS" in text and not re.search(r"0\.\d{4,}", text), text


# ================================================================================================ the index builder
def _layout_with_ranked_last(tmp):
    """anakit block -> runs layout + BASE/LAST written in v4base.run's own format (write_rank_files)."""
    from v4base import run as BR
    b = anakit.build(tmp / "a")
    anakit.jdump(b.contrasts, anakit.contrasts_obj(3))
    anakit.add_baselines_ranked(b, ("LAST",))
    lay = layoutkit.build_layout(tmp / "lay", b)
    d = lay["R"] / "BASE" / "LAST"
    d.mkdir(parents=True)
    shutil.copyfile(b.root / "MAIN" / "BASE_LAST" / "rows.json", d / "rows.json")
    m = b.manifests[M]
    ranks = b.ranks[(M, "LAST", None, "RULE_FREE")]
    art = SimpleNamespace(decision_id=m.decision_id, ranks=ranks, n_tied=anakit.last_ties(ranks))
    BR.write_rank_files(d, art, m, method="LAST", baseline="LAST", key_base={"split": "VALIDATION",
                                                                                 "train_manifest_id": anakit.ROLES[M][1]})
    return b, lay, d


def test_index_builder_admits_ranked_baselines_end_to_end(tmp_path):
    b, lay, _d = _layout_with_ranked_last(tmp_path)
    lines = []
    assert BI.main(lay["args"] + ["--out", str(tmp_path / "idx")], say=lines.append) == 0, lines
    idx = json.loads((tmp_path / "idx" / R.INDEX_NAME).read_text(encoding="utf-8"))
    last = next(e for e in idx["entries"] if e["arm"] == "LAST")
    assert last["comparison_block"] == "BASELINE_CPU" and last["rule"] == "RULE_FREE" and last["seed"] is None
    assert last["row_ties"].endswith("n_tied_int32.npy") and not last.get("historical_reference")
    assert next(e for e in idx["entries"] if e["arm"] == "POP")["historical_reference"] is True   # no RANKS_RECORD
    argv = ["--block", str(tmp_path / "idx"), "--split", "VALIDATION", "--contrasts", str(b.contrasts)]
    assert R.main(argv + ["--out", str(tmp_path / "o")], context_loader=b.loader, say=lines.append) == 0, lines
    res = result(tmp_path / "o")
    want = np.mean([pbv(b, "FA", s) - last_v(b) for s in (2026, 2027)])
    assert conf(res, "K11b")["point"] == pytest.approx(want, abs=1e-12)
    assert res["target_strata"]["status"] == "COMPUTED"


def check_index_rank_record(tmp_path):
    for name, edit, needle in (
            ("ranks", lambda d: np.save(d / "ranks_int32.npy", np.zeros(3, np.int32)), "ranks_sha256"),
            ("rows", lambda d: (d / "rows.json").write_text("[]", encoding="utf-8"), "rows_sha256"),
            ("ties", lambda d: np.save(d / "n_tied_int32.npy", np.zeros(3, np.int32)), "n_tied_sha256")):
        _b, lay, d = _layout_with_ranked_last(tmp_path / name)
        edit(d)
        lines = []
        rc = BI.main(lay["args"] + ["--out", str(tmp_path / name / "idx")], say=lines.append)
        assert rc == BI.REFUSED and needle in lines[0], lines


def test_index_refuses_a_rank_record_that_does_not_bind(tmp_path):
    check_index_rank_record(tmp_path)


@nc("an index builder that trusts RANKS_RECORD without checking the file hashes")
def test_nc_index_rank_record_unchecked(tmp_path, monkeypatch):
    def trusting(self, d, rel, method, rows_sha, rec):                              # MUTANT
        role = BI._role_of_manifest(self.a.split, rec.get("eval_manifest_id"))
        self.copy_file(d / rec["ranks_file"], f"{rel}/{rec['ranks_file']}")
        return {"arm": method, "seed": None, "rule": "RULE_FREE", "role": role, "rows": f"{rel}/rows.json",
                "ranks": f"{rel}/{rec['ranks_file']}", "comparison_block": "BASELINE_CPU", "source": "BASE"}
    monkeypatch.setattr(BI.Builder, "base_ranked", trusting)
    check_index_rank_record(tmp_path)


def test_index_builder_maps_the_new_arms(tmp_path):
    b = anakit.build(tmp_path / "a")
    lay = layoutkit.build_layout(tmp_path / "lay", b)
    R_ = lay["R"]
    for seed in (2026, 2027):                                     # FT_PRE: launch_device --arm FT_PRE (DEV-FTPRE)
        d = R_ / "DEV" / f"FT_PRE_s{seed}"
        shutil.copytree(R_ / "DEV" / f"FT_C_s{seed}", d)
        summ = json.loads((d / "RUN_SUMMARY.json").read_text(encoding="utf-8"))
        anakit.jdump(d / "RUN_SUMMARY.json", dict(summ, arm="FT_PRE"))
        anakit.jdump(d / "LAUNCH_0.json", {"label": "PRODUCTION", "arm": "FT_PRE", "start": {
            "run_id": anakit.PRE_RUN, "kind": "PRE", "init_from": dict(anakit.INIT_FROM),
            "init_record_sha256": anakit.PRE_INIT_RECORD_SHA}})
    for seed in (2026, 2027):                                     # DP8_FROZEN = DP_WARM_FROZEN eps 8, 2 seeds (ANA-06)
        d = R_ / "FL" / f"DP8_FROZEN_s{seed}"
        shutil.copytree(R_ / "FL" / f"FA_s{seed}", d)
        s = dp8_frozen_patch()(json.loads((d / "RUN_SUMMARY.json").read_text("utf-8")))
        s["arm_spec"]["cfg"]["seed"] = seed
        anakit.jdump(d / "RUN_SUMMARY.json", s)
        anakit.jdump(d / "LAUNCH_0.json", {"label": "PRODUCTION", "arm": "DP_WARM_FROZEN", "fl05": {"role": "MAIN"},
                                           "device_pin_sha256": anakit.PIN, "dp_z": {"epsilon": 8.0},
                                           "warm_selection": anakit.warm_binding("DP8_FROZEN")})   # ANA-06 (FL-16)
    lines = []
    assert BI.main(lay["args"] + ["--out", str(tmp_path / "idx")], say=lines.append) == 0, lines
    idx = json.loads((tmp_path / "idx" / R.INDEX_NAME).read_text(encoding="utf-8"))
    keys = {(e["arm"], e["seed"], e["rule"], e["role"]) for e in idx["entries"]}
    assert {("FT_PRE", s, PB, M) for s in (2026, 2027)} <= keys and ("DP8_FROZEN", 2026, CB, M) in keys
    ftp = next(e for e in idx["entries"] if e["arm"] == "FT_PRE")
    assert ftp["start_run_id"] == anakit.PRE_RUN and ftp["start_init"] == anakit.INIT_FROM
    assert ftp["start_init_record_sha256"] == anakit.PRE_INIT_RECORD_SHA
    shutil.copytree(R_ / "DEV" / "FT_C_s2026", R_ / "DEV" / "FT_C_X_s2026")          # theta_pre base under FT_C: refused
    anakit.jdump(R_ / "DEV" / "FT_C_X_s2026" / "LAUNCH_0.json",
                 {"label": "PRODUCTION", "arm": "FT_C", "start": {"run_id": anakit.PRE_RUN}})
    lines = []
    assert BI.main(lay["args"] + ["--out", str(tmp_path / "idx2")], say=lines.append) == BI.REFUSED
    assert "is FT_PRE" in lines[0], lines
    shutil.rmtree(R_ / "DEV" / "FT_C_X_s2026")
    src = R_ / "REEVAL_A40" / "C_BIG_s2026_PB"                    # C_BIG on the UNSEEN_VAL selection panel: refused
    dst = R_ / "REEVAL_A40" / "C_BIG_UNSEEN_VAL_s2026_PB"
    shutil.copytree(src, dst)
    rec = json.loads((dst / "REEVAL_RECORD.json").read_text(encoding="utf-8"))
    rec.update(role="UNSEEN_VAL", eval_manifest_id=R.ROLE_MAP["VALIDATION"]["UNSEEN_VAL"][0])
    anakit.jdump(dst / "REEVAL_RECORD.json", rec)
    lines = []
    assert BI.main(lay["args"] + ["--out", str(tmp_path / "idx3")], say=lines.append) == BI.REFUSED
    assert "no-FT references" in lines[0], lines
