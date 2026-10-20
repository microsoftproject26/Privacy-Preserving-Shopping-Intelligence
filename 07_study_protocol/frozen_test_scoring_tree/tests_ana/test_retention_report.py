"""ANA-01: tools/retention_report.py on synthetic blocks (tiny K, 2 seeds).

Known answers (anakit docstring), CI reproducibility with the seed, equality with the reused library contrasts, the
refusals (mixed device class, rule mismatch, zero denominator, ...) and bin / events-to-X known answers.
Negative controls (test_nc_*, strict xfail on AssertionError): each runs a positive check against a mutant."""
from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

import numpy as np
import pytest

import anakit
import retention_report as R
from anakit import SUBSET_USERS, macro
from v4eval import bootstrap as B
from v4train import bootstrap_ci as BC


def nc(reason: str):
    return pytest.mark.xfail(strict=True, raises=AssertionError, reason="NEGATIVE CONTROL: " + reason)


def run(blk, out, *extra, split=None):                     # split None = the block's own split (anakit.Block)
    lines = []
    rc = R.main(blk.argv(out, *extra, split=split), context_loader=blk.loader, say=lines.append)
    return rc, "\n".join(lines)


def result(out) -> dict:
    return json.loads((out / R.RESULT_NAME).read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def done(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("done")
    blk = anakit.build(tmp)
    out = tmp / "out"
    rc, text = run(blk, out)
    assert rc == 0, text
    return blk, out, result(out)


@pytest.fixture
def blk(tmp_path):
    return anakit.build(tmp_path)


def conf(res, kid, rule="PRACTICAL_BEST"):
    return next(c for c in res["confirmatory"] if c["id"] == kid)["by_rule"][rule]


def tab_row(res, arm, rule="PRACTICAL_BEST", role=None):
    tab = res["retention"]["MAIN"] if role is None else res["panels"][role]
    return next(r for r in tab[rule] if r["arm"] == arm)


# ------------------------------------------------------------------------------------------------ end to end
def test_writes_every_output_and_self_hashes(done):
    blk, out, res = done
    assert R.verify_self_sha(res)
    man = json.loads((out / R.MANIFEST_NAME).read_text())
    assert R.verify_self_sha(man) and man["result_self_sha256"] == res["self_sha256"]
    for rel, sha in man["files_sha256"].items():
        assert R._sha_file(out / rel) == sha, rel
    for name in R.CURVE_COLUMNS:
        assert (out / "curves" / f"{name}.csv").is_file() and (out / "curves" / f"{name}.json").is_file()
    assert res["script"]["sha256"] == R.script_sha256()
    assert set(res["library_files_sha256"]) >= {"v4train/bootstrap_ci.py", "v4eval/bootstrap.py", "v4eval/metrics.py",
                                                "v4base/strata.py", "v4base/identity.py"}


def test_deterministic_rerun_byte_identical(done, tmp_path):
    blk, out, _res = done
    out2 = tmp_path / "out2"
    rc, text = run(blk, out2)
    assert rc == 0, text
    for f in sorted(p.relative_to(out) for p in out.rglob("*") if p.is_file()):
        assert (out / f).read_bytes() == (out2 / f).read_bytes(), f


def test_labels_only_the_registry_contrasts_confirmatory(done):
    _blk, out, res = done
    assert res["labels"]["CONFIRMATORY"] == [f"K{i}" for i in range(1, 20)]         # ANA_CONTRASTS_v2 (06I §1)
    assert all(c["label"] == "CONFIRMATORY" for c in res["confirmatory"])

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
    rep = (out / R.REPORT_NAME).read_text(encoding="utf-8")
    assert "CONFIRMATORY" in rep.split("## 2.")[0]
    for sec in rep.split("\n## ")[2:8]:
        assert "DESCRIPTIVE" in sec.split("\n")[0], sec.split("\n")[0]


def test_report_prints_disclosures_verbatim(done):
    _blk, out, _res = done
    rep = (out / R.REPORT_NAME).read_text(encoding="utf-8")
    for text in (R.DISCLOSURES_06A_9, R.DISCLOSURES_06B_7, R.DISCLOSURE_06B_1, R.LADDER_NOTE):
        assert text in rep


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def test_embedded_registered_text_matches_the_contracts():
    a06a = _norm((R.V4 / "contracts" / "PREG1_AMENDMENT_06A.md").read_text(encoding="utf-8"))
    a06b = _norm((R.V4 / "contracts" / "PREG1_AMENDMENT_06B.md").read_text(encoding="utf-8"))
    assert _norm(R.DISCLOSURES_06A_9) in a06a
    assert _norm(R.DISCLOSURES_06B_7) in a06b
    assert _norm(R.DISCLOSURE_06B_1.replace(".", "")) in a06b.replace(".", "")
    a06i = _norm((R.V4 / "contracts" / "PREG1_AMENDMENT_06I.md").read_text(encoding="utf-8"))
    a06k = _norm((R.V4 / "contracts" / "PREG1_AMENDMENT_06K.md").read_text(encoding="utf-8"))
    for kid, c, q in R.REGISTERED_CONTRASTS:
        if int(kid[1:]) <= 16:
            assert f"| {kid} | {c} | {q} |" in a06b, kid
        elif int(kid[1:]) <= 19:                                                     # 06I §1 (question verbatim)
            assert f"| {kid} | {c} | {q.rsplit(' (seed 2026)', 1)[0]} |" in a06i, kid
        else:                                                                        # 06K §2 (formula verbatim)
            assert f"{kid} = {c}" in a06k or f"**{kid} = {c}**" in a06k, kid
    for level, _arms, train, infer, guarantee in R.PRIVACY_LADDER:
        assert f"| {level} |" in a06b and train in a06b and guarantee in a06b, level


# ------------------------------------------------------------------------------------------------ known answers
def test_known_answer_size_cost_is_exactly_50(done):
    _blk, _out, res = done
    for rule in R.RULES:
        k1 = conf(res, "K1", rule)
        assert k1["seeds"] == [2026, 2027]
        assert abs(k1["point"] - 50.0) < 1e-9
        assert abs(k1["ci95"][0] - 50.0) < 1e-9 and abs(k1["ci95"][1] - 50.0) < 1e-9
        for p in k1["per_seed"]:
            assert abs(p["point"] - 50.0) < 1e-9


def test_known_answer_retention_equals_direct_computation(done):
    blk, _out, res = done
    m = blk.manifests["MAIN"]
    for rule in R.RULES:
        want = np.mean([100.0 * macro(blk.ranks[("MAIN", "FA", s, rule)], m) / macro(blk.ranks[("MAIN", "C_BIG", s, rule)], m)
                        for s in (2026, 2027)])
        got = conf(res, "K2", rule)
        assert abs(got["point"] - want) < 1e-9
        row = tab_row(res, "FA", rule)
        assert abs(row["vs_C_BIG"]["mrr@20"]["point"] - want) < 1e-9
        for s in ("2026", "2027"):
            assert abs(row["entries"][s]["values"]["mrr@20"] - macro(blk.ranks[("MAIN", "FA", int(s), rule)], m)) < 1e-12
            assert abs(row["entries"][s]["values"]["ndcg@10"]
                       - macro(blk.ranks[("MAIN", "FA", int(s), rule)], m, "ndcg", 10)) < 1e-12


def test_ci_reproducible_with_the_registered_seed(done):
    """The ratio CI = percentiles of 100 M_b / C_b on W = make_W(n_users, seed 20260926); another seed differs."""
    blk, _out, res = done
    m = blk.manifests["MAIN"]
    users = np.unique(m.eval_user_key[m.eval_mask])
    per = conf(res, "K2")["per_seed"]
    for p in per:
        s = p["seed"]
        W = B.make_W(users.size, seed=20260926)
        mu = {arm: (lambda st: st.S / st.N)(BC._stats(m, blk.ranks[("MAIN", arm, s, "PRACTICAL_BEST")], arm))
              for arm in ("FA", "C_BIG")}
        rm = np.array([mu["FA"][w].mean() for w in W])
        rc = np.array([mu["C_BIG"][w].mean() for w in W])
        want = np.percentile(100.0 * rm / rc, [2.5, 97.5])
        assert np.allclose(p["ci95"], want, rtol=0, atol=1e-9)
        W2 = B.make_W(users.size, seed=20260927)
        st_a = BC._stats(m, blk.ranks[("MAIN", "FA", s, "PRACTICAL_BEST")], "a")
        st_c = BC._stats(m, blk.ranks[("MAIN", "C_BIG", s, "PRACTICAL_BEST")], "c")
        other = np.percentile(100.0 * np.array([(st_a.S / st_a.N)[w].mean() for w in W2])
                              / np.array([(st_c.S / st_c.N)[w].mean() for w in W2]), [2.5, 97.5])
        assert not np.allclose(p["ci95"], other)
    assert conf(res, "K2")["W_sha256"] == BC._plan(m).W_sha256


def test_ratio_and_delta_equal_the_library_contrasts(done):
    blk, _out, res = done
    m = blk.manifests["MAIN"]
    plan = BC._plan(m)
    st = {(a, s): BC._stats(m, blk.ranks[("MAIN", a, s, "PRACTICAL_BEST")], f"{a}:{s}")
          for a in ("FA", "C_BIG", "FT_C", "C_SMALL") for s in (2026, 2027)}
    for p in conf(res, "K2")["per_seed"]:
        lib = B.ratio(plan, st[("FA", p["seed"])], st[("C_BIG", p["seed"])], "macro_e2e", "x")
        assert p["point"] == pytest.approx(lib["point"], abs=1e-12)
        assert np.allclose(p["ci95"], lib["ci95"], rtol=0, atol=1e-12)
    k9 = conf(res, "K9")
    for p in k9["per_seed"]:
        lib = B.contrast(plan, st[("FT_C", p["seed"])], st[("C_SMALL", p["seed"])], "macro_e2e", "x")
        assert p["point"] == pytest.approx(lib["point"], abs=1e-15)
        assert np.allclose(p["ci95"], lib["ci95"], rtol=0, atol=1e-15)
    lib = B.seed_contrast(plan, {s: st[("FT_C", s)] for s in (2026, 2027)},
                          {s: st[("C_SMALL", s)] for s in (2026, 2027)}, "macro_e2e", "x")
    assert k9["point"] == pytest.approx(lib["point"], abs=1e-15)
    assert np.allclose(k9["ci95"], lib["ci95"], rtol=0, atol=1e-15)


def test_known_answer_personalisation_gain_and_harmed_share(done):
    """FT_C = C_BIG ranks, C_SMALL = 2x: FT_C - C_SMALL = C_BIG / 2 per seed; nobody harmed; LO worse than FA."""
    blk, _out, res = done
    m = blk.manifests["MAIN"]
    k9 = conf(res, "K9")
    for p in k9["per_seed"]:
        assert p["point"] == pytest.approx(macro(blk.ranks[("MAIN", "C_BIG", p["seed"], "PRACTICAL_BEST")], m) / 2,
                                           abs=1e-12)
        assert p["harmed_share"] == 0.0
    assert k9["harmed_share"] == 0.0
    k11 = conf(res, "K11")
    assert k11["point"] > 0 and k11["harmed_share"] == 0.0
    pers = {p["contrast"]: p for p in res["personalisation"]}
    assert pers["FT_C - C_SMALL"]["by_rule"]["PRACTICAL_BEST"]["point"] == pytest.approx(k9["point"], abs=1e-15)
    assert pers["FT_C - C_SMALL"]["by_rule"]["CONTROLLED_BUDGET"] == {"status": "NOT_AVAILABLE"}
    pf = pers["PF - FA"]["by_rule"]["PRACTICAL_BEST"]
    assert pf["point"] > 0 and 0.0 <= pf["harmed_share"] <= 1.0


def test_single_seed_contrasts_and_subset_int8(done):
    """K13 / K15 / K16 are single-seed 2026 (06E §2); K16 without ONNX FP32 rows: the CONFIRMATORY row is NOT_AVAILABLE
    and the A40 C_SMALL fallback is DESCRIPTIVE, DEVICE_MIXED, on the INT8 users subset (A-F2)."""
    _blk, _out, res = done
    k13 = conf(res, "K13", "CONTROLLED_BUDGET")
    assert k13["seeds"] == [2026] and k13["n_seeds"] == 1 and k13["seeds_status"] == "COMPLETE"
    k15 = conf(res, "K15")
    assert k15["seeds"] == [2026] and k15["point"] < 100.0          # C_FULL ranks better than C_BIG
    k16 = conf(res, "K16")
    assert k16["status"] == "NOT_AVAILABLE" and k16["label"] == "CONFIRMATORY" and "ONNX_CPU" in k16["reason"]
    k16_full = next(c for c in res["confirmatory"] if c["id"] == "K16")
    fb = k16_full["descriptive_fallback"]["PRACTICAL_BEST"]
    assert fb["label"] == "DESCRIPTIVE" and fb["comparison"] == "DEVICE_MIXED" and fb["seeds"] == [2026]
    assert fb["n_users"] == SUBSET_USERS.size and "population_note" in fb
    assert fb["point"] == pytest.approx(100.0, abs=1e-12)
    assert conf(res, "K16", "CONTROLLED_BUDGET")["status"] == "NOT_AVAILABLE"


def test_rule_free_baseline_is_point_only(done):
    blk, _out, res = done
    m = blk.manifests["MAIN"]
    pop = tab_row(res, "POP", "CONTROLLED_BUDGET")
    assert pop["vs_C_BIG"]["mrr@20"]["ci95"]["reason"] == "NO_RANKS"
    rk = anakit.arm_ranks("POP", 2026, "PRACTICAL_BEST", blk.base["MAIN"], m.rankable)
    want = np.mean([100 * macro(rk, m) / macro(blk.ranks[("MAIN", "C_BIG", s, "CONTROLLED_BUDGET")], m) for s in (2026, 2027)])
    assert pop["vs_C_BIG"]["mrr@20"]["point"] == pytest.approx(want, abs=1e-12)


# ------------------------------------------------------------------------------------------------ curves
def test_bytes_curve_from_the_12_marks(done):
    blk, _out, res = done
    rows = [r for r in res["curves"]["bytes"] if r["arm"] == "FA"]
    assert len(rows) == 24 and sorted({r["seed"] for r in rows}) == [2026, 2027]
    m = blk.manifests["MAIN"]
    for r in rows:
        i = int(round(r["mark"] * 2))
        assert r["round"] == 10 * i
        assert r["cum_bytes_per_client"] == pytest.approx(anakit.FA_PER_CLIENT * 10 * i / anakit.FA_ROUNDS)
        if r["mark"] == 5.5:
            want = 100 * macro(blk.ranks[("MAIN", "FA", r["seed"], "PRACTICAL_BEST")], m) / \
                macro(blk.ranks[("MAIN", "C_BIG", r["seed"], "PRACTICAL_BEST")], m)
            assert r["R_vs_C_BIG_PB"] == pytest.approx(want, abs=1e-9)
    fa = tab_row(res, "FA")["entries"]["2026"]["meta"]
    assert fa["bytes_per_visit"] == 165680392 and fa["param_count"] == 20710048 and fa["rounds"] == anakit.FA_ROUNDS


def test_epsilon_curve_and_both_accountants(done):
    _blk, _out, res = done
    eps = {e["arm"]: e for e in res["curves"]["epsilon"]}
    assert eps["FA_1024"]["epsilon_registered_accountant"] == "inf"
    assert eps["DP8"]["epsilon_registered_accountant"] == 8.0
    assert eps["DP8"]["epsilon_receipt_fields"] == {"epsilon_poisson": 7.99825, "epsilon_wor": 6.5}
    assert eps["DP1"]["epsilon_receipt_fields"]["epsilon_wor"] == 0.8
    k12 = conf(_res := res, "K12")
    assert eps["DP8"]["by_rule"]["PRACTICAL_BEST"]["R_vs_FA_1024"]["point"] == pytest.approx(k12["point"], abs=1e-12)


def test_size_curve_and_privacy_ladder(done):
    _blk, _out, res = done
    size = {s["arm"]: s for s in res["curves"]["model_size"]}
    assert size["C_SMALL"]["by_rule"]["PRACTICAL_BEST"]["point"] == pytest.approx(50.0, abs=1e-9)
    assert size["C_SMALL_INT8"]["model_bytes"] == 21_000_000
    ladder = {lv["level"]: lv for lv in res["privacy_ladder"]}
    assert len(ladder) == 7
    l3 = [a for a in ladder["L3 federated + DP"]["arms"] if a["arm"] == "DP8"][0]
    assert "ε = 8.0" in l3["guarantee"]
    assert any(a["arm"] == "FA_Q8" for a in ladder["L2 federated"]["arms"])


def test_strata_groups_cover_main_users(done):
    blk, _out, res = done
    m = blk.manifests["MAIN"]
    n = np.unique(m.eval_user_key[m.eval_mask]).size
    for fam in ("category_diversity", "brand_loyalty", "persona", "activity"):
        rows = [r for r in res["curves"]["strata"] if r["family"] == fam and r["arm"] == "C_SMALL"]
        assert sum(r["n_users"] for r in rows) == n, fam
        assert any(r["group"] == "NO_TRAIN" for r in rows)
        for r in rows:
            assert r["R_vs_C_BIG"] == pytest.approx(50.0, abs=1e-9)          # halving holds on every subset
    assert {r["group"] for r in res["curves"]["strata"] if r["family"] == "persona"} <= \
        {f"d{d}_l{l}" for d in range(3) for l in range(3)} | {"NO_TRAIN"}


def test_support_bins_known_counts_and_events_to_x(done):
    blk, _out, res = done
    counts = res["curves"]["support_bin_counts"]
    assert counts["TRUE_COLD"] == {"0": 16}
    n_main = anakit.ROLES["MAIN"][2]
    want = {}
    for k in range(n_main):
        lab = R.support_bin(anakit.SUPPORT_CYCLE[k % len(anakit.SUPPORT_CYCLE)])
        want[lab] = want.get(lab, 0) + 1
    assert counts["MAIN"] == dict(sorted(want.items()))
    assert not any(r["group"].startswith("OUT_OF") for r in res["curves"]["support_bins"])
    ev = {(e["panel"], e["arm"], e["seed"]): e for e in res["curves"]["events_to_x_bins"]}
    ft = ev[("UNSEEN_HOLDOUT", "FT_C", 2026)]                         # FT_C = C_BIG ranks -> 100% at bin 0
    assert ft["bins"][0]["bin"] == "0" and ft["bins"][0]["role"] == "TRUE_COLD"
    assert ft["events_to_90"] == "0" and ft["events_to_95"] == "0"
    lo = ev[("UNSEEN_HOLDOUT", "LO", 2026)]                            # no TRUE_COLD LO rows; far below -> never
    assert lo["bins"][0]["bin"] != "0" and lo["events_to_90"] == "NOT_REACHED"
    cs = ev[("MAIN", "C_SMALL", 2027)]
    assert cs["events_to_90"] == "NOT_REACHED"                         # exactly 50% everywhere


def test_support_bin_edges():
    assert R.support_bin(0) == "0"
    assert [R.support_bin(n) for n in (5, 10, 11, 20, 21, 50, 51, 100)] == \
        ["5-10", "5-10", "11-20", "11-20", "21-50", "21-50", "51-100", "51-100"]
    assert R.support_bin(3).startswith("OUT_OF") and R.support_bin(101).startswith("OUT_OF")


def test_events_to_x_known_answer_no_interpolation():
    rows = [{"role": "MAIN", "arm": "FA", "group": g, "per_seed": [{"seed": 2026, "value": m, "C_ref": 1.0}]}
            for g, m in (("5-10", 0.5), ("11-20", 0.91), ("21-50", 0.94), ("51-100", 0.96))]
    rows.append({"role": "TRUE_COLD", "arm": "FA", "group": "0", "per_seed": [{"seed": 2026, "value": 0.1, "C_ref": 1.0}]})
    ev = R.events_to_x_bins(rows)
    main = next(e for e in ev if e["panel"] == "MAIN")
    assert [b["bin"] for b in main["bins"]] == ["0", "5-10", "11-20", "21-50", "51-100"]
    assert main["events_to_90"] == "11-20" and main["events_to_95"] == "51-100"
    rows[3]["per_seed"][0]["value"] = 0.949999
    ev = R.events_to_x_bins(rows)
    assert next(e for e in ev if e["panel"] == "MAIN")["events_to_95"] == "NOT_REACHED"


def test_trunc_curves_known_answer(done):
    """A-F1 known answer from launch_trunc.trunc_labels-shaped rows: FIXED_20 / FIXED_50 are separate series with their
    own C_ref (the C_BIG FULL cell), ALL_AVAILABLE takes C_ref per query_h, M's FULL value is reported, Study B keeps
    the selected LO cell only."""
    _blk, _out, res = done
    ev = {(e["arm"], e["panel"], e["panel_type"], e["seed"]): e for e in res["curves"]["trunc_events_to_x"]}
    f20 = ev[("C_SMALL", "WARM_TRUNCATION_16384", "FIXED_20", 2026)]
    assert f20["events_to_90"] == 10 and f20["events_to_95"] == 20            # 0.185 / 0.2 = 0.925 first at h = 10
    assert f20["M_full_history"] == 0.1 and f20["C_ref_method"] == "C_BIG"
    f50 = ev[("C_SMALL", "WARM_TRUNCATION_16384", "FIXED_50", 2026)]
    assert f50["events_to_90"] == 50 and f50["events_to_95"] == 50            # 0.08 / 0.2 < 0.9 until h = 50
    pts = {(t["arm"], t["panel_type"], t["seed"], t["h"]): t for t in res["curves"]["trunc"]}
    assert pts[("C_SMALL", "FIXED_20", 2026, 5)]["M"] == 0.17 and pts[("C_SMALL", "FIXED_50", 2026, 5)]["M"] == 0.06
    c = pts[("C_SMALL", "ALL_AVAILABLE", 2027, 10)]
    assert c["C_ref"] == pytest.approx(0.21) and c["R"] == pytest.approx(100 * 0.185 / 0.21)
    assert c["M_full_history"] == 0.1
    lo = [t for t in res["curves"]["trunc"] if t["arm"] == "LO"]
    assert lo and all(t["M"] == 0.05 for t in lo)                             # the selected cell only
    assert ev[("LO", "COLD_H_16384", "ALL_AVAILABLE", 2026)]["events_to_90"] == "NOT_REACHED"


def test_shift_indicator_on_test(tmp_path, done):
    """The full TEST path (A-F8): gate + register + lock-as-contrasts + extension + TEST role map + gate-bound headers +
    the same-script VALIDATION_RESULT; the manifests come from the injected loader (the DATA-09 hook's place)."""
    blk, out, res = done
    b2 = anakit.build_test(tmp_path, anakit.validation_result_v3(tmp_path / "v3"))     # FIX-DELTA-03 M5: v3 on TEST
    rc, text = run(b2, tmp_path / "o2", split="TEST")
    assert rc == 0, text
    r2 = result(tmp_path / "o2")
    si = r2["shift_indicator"]
    for rule in R.RULES:
        for seed in ("2026", "2027"):
            assert si["by_rule"][rule][seed]["delta"] == pytest.approx(0.0, abs=1e-15)
    assert r2["inputs"]["test_gate"]["test_exposure_register_entry_id"] == 2
    assert r2["inputs"]["validation_result_check"]["script_sha256"] == R.script_sha256()


# ------------------------------------------------------------------------------------------------ refusals
def check_refuses(blk, tmp_path, needle: str):
    rc, text = run(blk, tmp_path / "out_refused")
    assert rc == R.REFUSED, text
    assert needle in text, text
    assert not (tmp_path / "out_refused" / R.RESULT_NAME).exists()


def _set_rows_field(blk, rel, field, value, where=None):
    p = blk.root / rel
    obj = json.loads(p.read_text(encoding="utf-8"))
    for r in R._rows_of(obj):
        if where is None or all(r.get(k) == v for k, v in where.items()):
            r[field] = value
    p.write_text(json.dumps(obj), encoding="utf-8")


def mix_device(blk):
    _set_rows_field(blk, "MAIN/FP_s2027/RESULT_ROWS.json", "eval_device_class", "A16-FP32-DET")


def test_mixed_device_class_refused(blk, tmp_path):
    mix_device(blk)
    check_refuses(blk, tmp_path, "eval_device_class")


@nc("the comparison key without eval_device_class")
def test_nc_mixed_device_class(blk, tmp_path, monkeypatch):
    monkeypatch.setattr(R, "KEY_FIELDS", ("split", "evaluator_code_sha256"))           # MUTANT
    idx = blk.index()
    idx.pop("eval_device_class")                                                        # (the second guard off too)
    idx["records"].pop("TRUNC_ROWS")                                                    # (its own device key check)
    blk.write_index(idx)
    mix_device(blk)
    check_refuses(blk, tmp_path, "eval_device_class")


def test_mixed_evaluator_and_manifest_refused(tmp_path):
    b = anakit.build(tmp_path / "a")
    _set_rows_field(b, "MAIN/FP_s2026/RESULT_ROWS.json", "evaluator_code_sha256", "f" * 64)
    check_refuses(b, tmp_path / "a", "evaluator_code_sha256")
    b = anakit.build(tmp_path / "b")
    _set_rows_field(b, "MAIN/FA_Q8_s2026/RESULT_ROWS.json", "query_order_hash", "0" * 64)
    check_refuses(b, tmp_path / "b", "query_order_hash")
    b = anakit.build(tmp_path / "c")
    _set_rows_field(b, "MAIN/FL_FA_s2026/evals/eval_3.0.rows.json", "eval_device_class", "A16-FP32-DET")
    check_refuses(b, tmp_path / "c", "eval_device_class")


def test_historical_reference_is_exempt_and_labelled(blk, tmp_path):
    mix_device(blk)
    idx = blk.index()
    for e in idx["entries"]:
        if e["arm"] == "FP" and e["seed"] == 2027:
            e["historical_reference"] = True
    blk.write_index(idx)
    rc, text = run(blk, tmp_path / "o")
    assert rc == 0, text
    res = result(tmp_path / "o")
    hist = {h["entry"] for h in res["historical_reference"]}
    assert hist == {"FP|2027|PRACTICAL_BEST|MAIN", "FP|2027|CONTROLLED_BUDGET|MAIN"}
    assert tab_row(res, "FP")["vs_C_BIG"]["mrr@20"]["seeds"] == [2026]


def rule_mismatch(blk):
    """C_BIG seed 2026 PRACTICAL_BEST pointed at a rows file that only carries CONTROLLED_BUDGET rows."""
    p = blk.root / "MAIN" / "C_BIG_s2026" / "RESULT_ROWS.json"
    obj = json.loads(p.read_text(encoding="utf-8"))
    cb = [r for r in obj["rows"] if r["selection_rule"] == "CONTROLLED_BUDGET"]
    anakit.jdump(blk.root / "MAIN" / "C_BIG_s2026" / "CB_ONLY.json", {"rows": cb})
    idx = blk.index()
    for e in idx["entries"]:
        if e["arm"] == "C_BIG" and e["seed"] == 2026 and e["rule"] == "PRACTICAL_BEST" and e["role"] == "MAIN":
            e["rows"] = "MAIN/C_BIG_s2026/CB_ONLY.json"
            e["ranks"] = "MAIN/C_BIG_s2026/ranks_CONTROLLED_BUDGET.npy"
    blk.write_index(idx)


def test_rule_mismatch_refused(blk, tmp_path):
    rule_mismatch(blk)
    check_refuses(blk, tmp_path, "RULE_MISMATCH")


@nc("a row selection that ignores the declared checkpoint rule")
def test_nc_rule_mismatch(blk, tmp_path, monkeypatch):
    real = R.select_rows

    def ignore_rule(entry, rows):                                                       # MUTANT
        if entry.rule in R.RULES and not any(r.get("selection_rule") == entry.rule for r in rows):
            rows = [dict(r, selection_rule=None) for r in rows]
        return real(entry, rows)
    monkeypatch.setattr(R, "select_rows", ignore_rule)
    rule_mismatch(blk)
    check_refuses(blk, tmp_path, "RULE_MISMATCH")


def _pair_entries(done):
    blk, _out, _res = done
    e1 = R.Entry("FA", 2026, "PRACTICAL_BEST", "MAIN", "x", None, {}, None, None, False, {})
    e2 = R.Entry("C_BIG", 2026, "CONTROLLED_BUDGET", "MAIN", "x", None, {}, None, None, False, {})
    e3 = R.Entry("C_BIG", 2027, "PRACTICAL_BEST", "MAIN", "x", None, {}, None, None, False, {})
    return e1, e2, e3


def check_pairs_refuse(done):
    e1, e2, e3 = _pair_entries(done)
    for den, word in ((e2, "RULE_MISMATCH"), (e3, "SEED_MISMATCH")):
        with pytest.raises(R.AnaRefused, match=word):
            R.check_pairs([(e1, den)])


def test_pairing_rule_and_seed_refused(done):
    check_pairs_refuse(done)


@nc("a pairing check that accepts any rule / seed")
def test_nc_pairing_rule_and_seed(done, monkeypatch):
    monkeypatch.setattr(R, "check_pairs", lambda pairs: None)                         # MUTANT
    e1, e2, _e3 = _pair_entries(done)
    try:
        R.check_pairs([(e1, e2)])
    except R.AnaRefused:
        return
    raise AssertionError("the mismatched pair was accepted")


def zero_denominator(blk):
    """C_BIG seed 2027 PRACTICAL_BEST: every rank > 20 -> macro MRR@20 = 0."""
    m = blk.manifests["MAIN"]
    rk = np.where(m.rankable, 25, -1).astype(np.int64)
    np.save(blk.root / "MAIN" / "C_BIG_s2027" / "ranks_PRACTICAL_BEST.npy", rk.astype(np.int32))
    p = blk.root / "MAIN" / "C_BIG_s2027" / "RESULT_ROWS.json"
    obj = json.loads(p.read_text(encoding="utf-8"))
    keep = [r for r in obj["rows"] if r["selection_rule"] != "PRACTICAL_BEST"]
    obj["rows"] = keep + anakit.rows_for("C_BIG", 2027, "PRACTICAL_BEST", rk, m)
    p.write_text(json.dumps(obj), encoding="utf-8")


def test_zero_denominator_refused(blk, tmp_path):
    zero_denominator(blk)
    check_refuses(blk, tmp_path, "zero denominator")


@nc("a block-level denominator check that lets C = 0 through")
def test_nc_zero_denominator(blk, tmp_path, monkeypatch):
    monkeypatch.setattr(R, "require_nonzero_denominator", lambda value, what: None)    # MUTANT
    zero_denominator(blk)
    check_refuses(blk, tmp_path, "zero denominator")


def swap_ranks(blk):
    idx = blk.index()
    for e in idx["entries"]:
        if e["arm"] == "C_SMALL" and e["seed"] == 2026 and e["rule"] == "PRACTICAL_BEST" and e["role"] == "MAIN":
            e["ranks"] = "MAIN/C_SMALL_s2026/ranks_CONTROLLED_BUDGET.npy"
    blk.write_index(idx)


def test_rank_file_of_another_rule_refused(blk, tmp_path):
    swap_ranks(blk)
    check_refuses(blk, tmp_path, "disagrees with its rank file")


@nc("no rank-file / row-value binding check")
def test_nc_rank_file_swapped(blk, tmp_path, monkeypatch):
    monkeypatch.setattr(R, "check_rank_rows", lambda label, manifest, ranks, rows: 0.0)  # MUTANT
    swap_ranks(blk)
    check_refuses(blk, tmp_path, "disagrees with its rank file")


def test_split_seed_and_index_refusals(tmp_path):
    b = anakit.build(tmp_path / "a")
    anakit.jdump(b.contrasts, anakit.contrasts_obj(3))                      # FIX-DELTA-03 M5: TEST takes v3 only
    rc, text = run(b, tmp_path / "a" / "o", split="TEST")
    assert rc == 2 and "split" in text
    b = anakit.build(tmp_path / "b")
    _set_rows_field(b, "MAIN/FP_s2026/RESULT_ROWS.json", "seed", 2027)
    check_refuses(b, tmp_path / "b", "SEED_MISMATCH")
    b = anakit.build(tmp_path / "c")
    idx = b.index()
    idx["entries"][0]["rule"] = "PRACTICAL_BEST_EXTENDED"
    b.write_index(idx)
    check_refuses(b, tmp_path / "c", "PRACTICAL_BEST_EXTENDED")
    b = anakit.build(tmp_path / "d")
    idx = b.index()
    idx["entries"][0]["rows"] = "../../outside.json"
    b.write_index(idx)
    check_refuses(b, tmp_path / "d", "leaves the block")
    b = anakit.build(tmp_path / "e")
    idx = b.index()
    idx["entries"] = [e for e in idx["entries"] if e["arm"] != "C_BIG" or e["role"] != "MAIN"]
    b.write_index(idx)
    check_refuses(b, tmp_path / "e", "no C_BIG entry")


def test_contrasts_must_be_the_registered_text(blk, tmp_path):
    obj = anakit.contrasts_obj()
    obj["confirmatory_contrasts"][1]["contrast"] = "R(FP / C_BIG)"
    anakit.jdump(blk.contrasts, obj)
    check_refuses(blk, tmp_path, "differs from the registered")
    obj = anakit.contrasts_obj()
    obj["confirmatory_contrasts"] = obj["confirmatory_contrasts"][:15]
    anakit.jdump(blk.contrasts, obj)
    check_refuses(blk, tmp_path, "not exactly K1-K19")
    rows = [f"| {k} | {c.replace(chr(0x2212), '-')} | {q} |" for k, c, q in R.REGISTERED_CONTRASTS[:19]]
    obj = anakit.contrasts_obj()
    obj.pop("confirmatory_contrasts")
    obj["contrasts"] = rows
    anakit.jdump(blk.contrasts, obj)                                                                   # accepted
    assert R.load_contrasts(blk.contrasts)["items"][8]["kind"] == "delta"


def test_contrast_parsing():
    assert R.parse_contrast("R(INT8 / FP32) for C_SMALL") == {"kind": "retained", "num": "C_SMALL_INT8",
                                                              "den": "C_SMALL_FP32", "for": "C_SMALL"}
    assert R.parse_contrast("FT_FA − FA")["kind"] == "delta"
    with pytest.raises(R.AnaRefused):
        R.parse_contrast("R(XX / C_BIG)")


def test_out_dir_never_overwritten_and_expected_sha(done, tmp_path):
    blk, out, _res = done
    rc, text = run(blk, out)
    assert rc == 2 and "not empty" in text
    rc, text = run(blk, tmp_path / "o", "--expect-script-sha256", "0" * 64)
    assert rc == 2 and "frozen" in text
    rc, text = run(blk, tmp_path / "o2", "--expect-script-sha256", R.script_sha256())
    assert rc == 0, text


def test_check_only_writes_nothing_and_prints_no_value(done, tmp_path):
    blk, _out, res = done
    rc, text = run(blk, tmp_path / "o", "--check-only")
    assert rc == 0 and "PASS" in text, text
    assert not (tmp_path / "o").exists()
    assert not re.search(r"0\.\d{4,}", text)


def test_tampered_records_refused(blk, tmp_path):
    p = blk.root / "records" / "S_RECORD.json"
    obj = json.loads(p.read_text())
    obj["S"] = 9.0
    p.write_text(json.dumps(obj))
    check_refuses(blk, tmp_path, "S_RECORD")


def test_records_reported(done):
    _blk, out, res = done
    rec = res["records"]
    assert rec["S_RECORD"]["S"] == 1.2345 and rec["SELECTION"][0]["role"] == "UNSEEN_VAL"
    assert rec["LR_SELECTION"]["self_sha256_status"] == "VERIFIED" and rec["LR_SELECTION"]["label"] == "CALIB (not MAIN)"
    assert rec["STRATA"]["registry_self_sha256"]
    st = res["sanity_trigger"]                                          # 06E §6: C_SMALL CB vs C_BIG PB
    m = _blk.manifests["MAIN"]
    want = np.mean([100 * macro(_blk.ranks[("MAIN", "C_SMALL", s, "CONTROLLED_BUDGET")], m)
                    / macro(_blk.ranks[("MAIN", "C_BIG", s, "PRACTICAL_BEST")], m) for s in (2026, 2027)])
    assert st["R_C_SMALL_CB_vs_C_BIG_PB"] == pytest.approx(want, abs=1e-12) and st["fired"] is bool(want < 50)
    assert st["R_C_SMALL_PB_vs_C_BIG_PB"] == pytest.approx(50.0, abs=1e-9)
    rep = (out / R.REPORT_NAME).read_text(encoding="utf-8")
    assert "LR_SELECTION" in rep and "CALIB" in rep


# ------------------------------------------------------------------------------------------------ added behaviours
def test_fl_ranks_derived_from_run_dir_and_marks_complete(tmp_path):
    b = anakit.build(tmp_path / "a")
    idx = b.index()
    for e in idx["entries"]:
        if e["arm"] == "FA" and e["role"] == "MAIN":
            e.pop("ranks")
            e["run_dir"] = f"MAIN/FL_FA_s{e['seed']}"
    b.write_index(idx)
    rc, text = run(b, tmp_path / "a" / "o")
    assert rc == 0, text
    res = result(tmp_path / "a" / "o")
    notes = {c["entry"]: c["notes"] for c in res["checks"]["entries"]}
    assert "mark 5.5" in notes["FA|2026|PRACTICAL_BEST|MAIN"][0]
    assert "mark 6.0" in notes["FA|2027|CONTROLLED_BUDGET|MAIN"][0]
    assert len([r for r in res["curves"]["bytes"] if r["arm"] == "FA"]) == 24         # PB entries only
    b = anakit.build(tmp_path / "b")
    mj = b.root / "MAIN" / "FL_FA_s2027" / "metrics.jsonl"
    mj.write_text("\n".join(mj.read_text().splitlines()[:-1]) + "\n")
    check_refuses(b, tmp_path / "b", "not the 12 registered")


def test_support_and_hex_from_verified_tables(tmp_path, lib):
    import pyarrow as pa
    import pyarrow.parquet as pq
    root = tmp_path / "v4"
    (root / "m").mkdir(parents=True)
    hexes = [anakit.hexkey("T", k) for k in range(4)]
    pq.write_table(pa.table({"client_ord": np.arange(4, dtype=np.int64), "client_key_hex": hexes}),
                   root / "m" / "T.clients.parquet")
    pq.write_table(pa.table({"client_ord": np.array([0, 0, 2, 2, 2, 3], dtype=np.int64)}), root / "m" / "T.decisions.parquet")
    pq.write_table(pa.table({"eval_user_key": np.arange(3, dtype=np.int64), "client_key_hex": hexes[1:]}),
                   root / "m" / "E.users.parquet")

    def tab(rel, n):
        return {"file": rel, "file_sha256": R._sha_file(root / rel), "row_count": n}
    train = {"clients": tab("m/T.clients.parquet", 4), "decisions": tab("m/T.decisions.parquet", 6)}
    sup = R.support_from_tables(train, root)
    assert sup == {hexes[0]: 2, hexes[1]: 0, hexes[2]: 3, hexes[3]: 1}
    hx = R.hex_from_tables({"users": tab("m/E.users.parquet", 3)}, root)
    assert list(hx) == hexes[1:]
    bad = dict(train, decisions=dict(train["decisions"], file_sha256="0" * 64))
    with pytest.raises(ValueError, match="file_sha256"):
        R.support_from_tables(bad, root)


def check_bin_edges():
    assert [R.support_bin(n) for n in (0, 4, 5, 10, 11, 20, 21, 50, 51, 100, 101)] == \
        ["0", "OUT_OF_REGISTERED_BINS:1-4", "5-10", "5-10", "11-20", "11-20", "21-50", "21-50", "51-100", "51-100",
         "OUT_OF_REGISTERED_BINS:>100"]


def test_support_bin_edges_registered():
    check_bin_edges()


@nc("support bins with an exclusive lower edge (5 -> out of bins, 11 -> 5-10)")
def test_nc_support_bin_edges(monkeypatch):
    def off_by_one(n):                                                                  # MUTANT
        for lo, hi, lab in R.SUPPORT_BINS:
            if lo < n <= hi + 1 or (lo == hi == n == 0):
                return lab
        return "OUT_OF_REGISTERED_BINS:" + ("1-4" if 1 <= n <= 4 else ">100")
    monkeypatch.setattr(R, "support_bin", off_by_one)
    check_bin_edges()


def check_events_to_x_no_interpolation():
    rows = [{"role": "MAIN", "arm": "FA", "group": g, "per_seed": [{"seed": 2026, "value": m, "C_ref": 1.0}]}
            for g, m in (("5-10", 0.80), ("11-20", 0.89), ("21-50", 0.99))]
    ev = R.events_to_x_bins(rows)[0]
    assert ev["events_to_90"] == "21-50"                   # 0.89 < 0.90: never interpolated between 11-20 and 21-50


def test_events_to_x_never_interpolates():
    check_events_to_x_no_interpolation()


@nc("events-to-X that rounds M / C_ref to two decimals (0.89 -> reached at 0.9 tolerance)")
def test_nc_events_to_x_interpolates(monkeypatch):
    real = R.events_to_x_bins

    def lenient(rows):                                                                  # MUTANT
        rows = [dict(r, per_seed=[dict(p, value=p["value"] + 0.011) for p in r["per_seed"]]) for r in rows]
        return real(rows)
    monkeypatch.setattr(R, "events_to_x_bins", lenient)
    check_events_to_x_no_interpolation()
