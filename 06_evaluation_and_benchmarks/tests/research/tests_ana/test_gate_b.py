"""GATE-B (MULTI_EXPERT_REVIEW_20260929 §4 Gate B) — the ANA side of B3 and B4.

  B3 (review 09 [8-C1]): FA_WARM_FROZEN (S8) is an admitted arm of tools/retention_report.py and tools/ana_block_index.py;
     its registered bytes per client are launch_fl's per_client_incl_frozen_once (the frozen item tables charged ONCE
     per visited client), the charged-dropped variants carry the same once component, and curve_bytes charges that
     component as a step at the first visit (never prorated). A run with frozen tables is never FA_WARM, and an S8 run
     without the frozen-once bytes record refuses.
  B4 (review 11 [10-R4]): trunc_curves reads NO_C_REF (never NOT_REACHED) when the C_ref row is missing.
Negative controls (test_nc_*, strict xfail on AssertionError): the positive check against a mutant.
"""
from __future__ import annotations

import json
import shutil

import pytest

import anakit
import layoutkit
import retention_report as R
from test_ana02 import build_index
from test_retention_report import check_refuses, nc, result, run, tab_row

S8 = "FA_WARM_FROZEN"
S8_VISIT, S8_DOWN, S8_UP = 2_121_296, 1_060_648, 1_060_648      # 265,162 uploaded params x 4 B, each direction
S8_TRAIN_PER_CLIENT = 6.4e6
ONCE = 81_780_000                                                # the frozen tables (item_embed / output_embed / bias)
VISITED_SHARE = 0.9
S8_INCL = S8_TRAIN_PER_CLIENT + VISITED_SHARE * ONCE


def s8_summary_patch(summ: dict, *, frozen=True, incl=True) -> dict:
    summ = dict(summ)
    b = dict(summ["bytes"])
    b["per_visit"] = dict(b["per_visit"], send_plus_receive_bytes=S8_VISIT, upload_bytes=S8_UP, download_bytes=S8_DOWN,
                          shared_numel=265_162)
    b["per_client"] = {"mean": S8_TRAIN_PER_CLIENT}
    if frozen:
        b["per_visit"]["frozen_bytes_once"] = ONCE
        summ["frozen_item_tables"] = {"keys": ["item_embed.weight", "output_embed", "output_bias"]}
    if incl:
        b["per_client_incl_frozen_once"] = {"mean": S8_INCL, "rule": "visits x (down + up) + the frozen tables once"}
    summ["bytes"] = b
    return summ


def add_s8(blk, *, arm=S8, **patch) -> None:
    """Two S8 FL run dirs (seeds 2026 / 2027) in the anakit layout, with launch_fl's FL-12b bytes record."""
    idx = blk.index()
    m, base = blk.manifests["MAIN"], blk.base["MAIN"]
    for seed in (2026, 2027):
        rel = f"MAIN/FL_{arm}_s{seed}"
        anakit._fl_run(blk.root, rel, arm, seed, m, base, m.rankable, idx["entries"], "VALIDATION")
        p = blk.root / rel / "RUN_SUMMARY.json"
        anakit.jdump(p, s8_summary_patch(json.loads(p.read_text(encoding="utf-8")), **patch))
        for rule in R.RULES:
            blk.ranks[("MAIN", arm, seed, rule)] = anakit.arm_ranks(arm, seed, rule, base, m.rankable)
    anakit.ensure_warm_records(blk.root, idx)                          # REV-ANA-05 F3: the S7 / S8 file in records
    blk.write_index(idx)


@pytest.fixture(scope="module")
def s8_done(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("s8")
    blk = anakit.build(tmp)
    add_s8(blk)
    rc, text = run(blk, tmp / "out")
    assert rc == 0, text
    return blk, result(tmp / "out")


# ------------------------------------------------------------------------------------------------ B3: retention_report
def check_s8_bytes(res) -> None:
    """B3 key property: S8's bytes per client include the frozen tables once; the charged-dropped value = incl + dropped
    visits x download / n_clients; the bytes curve starts with the full once step and prorates only the training."""
    for seed, dropped in anakit.FA_DROPPED.items():
        meta = tab_row(res, S8)["entries"][str(seed)]["meta"]
        assert meta["bytes_per_client"] == pytest.approx(S8_INCL)
        assert meta["bytes_per_client_excl_frozen_once"] == pytest.approx(S8_TRAIN_PER_CLIENT)
        assert meta["frozen_bytes_once"] == ONCE and "PER_CLIENT_INCL_FROZEN_ONCE" in meta["bytes_rule"]
        assert meta["frozen_once_per_client_mean"] == pytest.approx(VISITED_SHARE * ONCE)
        assert meta["bytes_per_client_charged_dropped"] == pytest.approx(
            S8_INCL + dropped * S8_DOWN / anakit.FA_N_CLIENTS)
        assert meta["bytes_per_client_charged_dropped_up_down"] == pytest.approx(
            S8_INCL + dropped * S8_VISIT / anakit.FA_N_CLIENTS)
    rows = [r for r in res["curves"]["bytes"] if r["arm"] == S8]
    assert len(rows) == 24
    step = VISITED_SHARE * ONCE
    for r in rows:
        want = step + S8_TRAIN_PER_CLIENT * r["round"] / anakit.FA_ROUNDS
        assert r["cum_bytes_per_client"] == pytest.approx(want)
        assert r["frozen_once_step_bytes_per_client"] == pytest.approx(step)
        assert "STEP_AT_FIRST_VISIT" in r["bytes_rule"]
    first = min(rows, key=lambda r: r["round"])
    assert first["cum_bytes_per_client"] > step                     # the whole once cost at the first mark


def test_s8_is_an_admitted_arm_with_frozen_once_bytes(s8_done):
    _blk, res = s8_done
    check_s8_bytes(res)
    assert S8 in R.ARMS and S8 in R.FL_ARMS and R.FL_METHOD_ALT[S8] == ("FA",)


def test_other_fl_arms_keep_the_prorated_curve_and_their_meta(s8_done):
    _blk, res = s8_done
    fa = tab_row(res, "FA")["entries"]["2026"]["meta"]
    assert "bytes_rule" not in fa and "frozen_bytes_once" not in fa
    for r in (r for r in res["curves"]["bytes"] if r["arm"] == "FA"):
        assert r["cum_bytes_per_client"] == pytest.approx(anakit.FA_PER_CLIENT * r["round"] / anakit.FA_ROUNDS)
        assert "frozen_once_step_bytes_per_client" not in r and r["bytes_rule"].startswith("PRORATED_BY_ROUND")


def test_s8_without_its_frozen_once_record_refuses(tmp_path):
    for i, patch in enumerate((dict(incl=False), dict(frozen=False))):
        blk = anakit.build(tmp_path / f"b{i}")
        add_s8(blk, **patch)
        check_refuses(blk, tmp_path / f"b{i}", "FA_WARM_FROZEN (S8) run needs")


def test_a_frozen_run_under_another_arm_refuses(tmp_path):
    blk = anakit.build(tmp_path / "w")
    add_s8(blk, arm="FA_WARM")                                        # an S8-shaped run declared as FA_WARM
    check_refuses(blk, tmp_path / "w", "froze its item tables")


@nc("S8 bytes read from per_client.mean only (the pre-B3 _fl_meta; the 81.78 MB table download omitted) must not pass "
    "the frozen-once bytes property")
def test_nc_s8_bytes_without_the_frozen_tables(tmp_path, monkeypatch):
    monkeypatch.setattr(R, "frozen_once_meta", lambda summ, label: {})
    blk = anakit.build(tmp_path / "nc")
    add_s8(blk)
    rc, text = run(blk, tmp_path / "nc" / "out")
    assert rc == 0, text
    check_s8_bytes(result(tmp_path / "nc" / "out"))


@nc("a prorated once cost (the pre-B3 curve: incl mean x round / rounds) must not pass the first-visit step property")
def test_nc_s8_curve_prorates_the_once_cost(tmp_path, monkeypatch):
    real = R.curve_bytes

    def prorating(eng, book):                                        # the step component forgotten by the curve only
        for ents in book.by.values():
            for e in ents.values():
                e.meta_auto.pop("frozen_once_per_client_mean", None)
        return real(eng, book)

    monkeypatch.setattr(R, "curve_bytes", prorating)
    blk = anakit.build(tmp_path / "nc2")
    add_s8(blk)
    rc, text = run(blk, tmp_path / "nc2" / "out")
    assert rc == 0, text
    check_s8_bytes(result(tmp_path / "nc2" / "out"))


# ------------------------------------------------------------------------------------------------ B3: the index builder
def _s8_layout(tmp_path, *, frozen=True, name=S8, summary_arm=S8):
    b = anakit.build(tmp_path / "a")
    lay = layoutkit.build_layout(tmp_path / "lay", b)
    for seed in (2026, 2027):
        d = lay["R"] / "FL" / f"{name}_s{seed}"
        shutil.copytree(lay["R"] / "FL" / f"FA_s{seed}", d)
        p = d / "RUN_SUMMARY.json"
        summ = json.loads(p.read_text(encoding="utf-8"))
        summ = s8_summary_patch(summ, frozen=frozen, incl=frozen)
        summ["arm"] = summary_arm
        summ["arm_spec"]["init_from"] = dict(anakit.INIT_FROM)           # FIX-DELTA-03 M2: warm from theta_pre
        p.write_text(json.dumps(summ, indent=1, sort_keys=True) + "\n", encoding="utf-8")
        lp = d / "LAUNCH_0.json"                                        # ANA-05: the 2-D S8 warm-selection binding
        anakit.jdump(lp, dict(json.loads(lp.read_text(encoding="utf-8")), warm_selection=anakit.warm_binding(S8)))
    return b, lay


def check_index_admits_s8(tmp_path, **kw):
    b, lay = _s8_layout(tmp_path, **kw)
    rc, text, _b, _lay = build_index(tmp_path, layout=lay, blk_=b)
    assert rc == 0, text
    idx = json.loads((tmp_path / "idx" / R.INDEX_NAME).read_text(encoding="utf-8"))
    keys = {(e["arm"], e["seed"], e["rule"], e["role"]) for e in idx["entries"]}
    assert {(S8, s, r, "MAIN") for s in (2026, 2027) for r in R.RULES} <= keys
    assert ("FA", 2026, "PRACTICAL_BEST", "MAIN") in keys


def test_index_admits_s8(tmp_path):
    check_index_admits_s8(tmp_path / "x")
    check_index_admits_s8(tmp_path / "y", summary_arm="FA")          # the launcher may keep method "FA"


def test_index_refuses_s8_without_frozen_record_and_frozen_fa_warm(tmp_path):
    b, lay = _s8_layout(tmp_path / "n", frozen=False)
    rc, text, _b, _lay = build_index(tmp_path / "n", layout=lay, blk_=b)
    assert rc == R.REFUSED and "FA_WARM_FROZEN (S8) run without" in text, text
    b, lay = _s8_layout(tmp_path / "w", name="FA_WARM", summary_arm="FA")
    rc, text, _b, _lay = build_index(tmp_path / "w", layout=lay, blk_=b)
    assert rc == R.REFUSED and "froze its item tables" in text, text


@nc("an index builder whose FL arm list lacks FA_WARM_FROZEN (the pre-B3 retention_report.FL_ARMS) must not admit S8")
def test_nc_index_without_s8_in_fl_arms(tmp_path, monkeypatch):
    monkeypatch.setattr(R, "FL_ARMS", tuple(a for a in R.FL_ARMS if a != S8))
    check_index_admits_s8(tmp_path)


# ------------------------------------------------------------------------------------------------ B4: trunc_curves
def check_no_c_ref(curves=R.trunc_curves) -> None:
    rows = anakit.trunc_rows()
    no_cref = [r for r in rows if not (r["method"] == "C_BIG" and r["trunc_full_history"])]
    _curve, ev = curves(no_cref)
    assert ev and all(e["events_to_90"] == e["events_to_95"] == "NO_C_REF" for e in ev)
    partial = [r for r in rows if not (r["method"] == "C_BIG" and r["trunc_panel"] == "COLD_H_16384"
                                       and r["trunc_full_history"] and r["trunc_query_h"] == 0)]
    _curve, ev = curves(partial)
    cold = [e for e in ev if e["panel"] == "COLD_H_16384" and e["arm"] == "LO"]
    assert cold and all(e["events_to_90"] == "NO_C_REF" for e in cold)          # h = 0 has no C_ref: undecidable
    warm = [e for e in ev if e["panel"] == "WARM_TRUNCATION_16384"]
    assert warm and all(e["events_to_90"] not in ("NO_C_REF",) for e in warm)   # complete C_ref: unchanged


def test_trunc_curves_no_c_ref_status():
    check_no_c_ref()


def test_trunc_curves_complete_cref_values_unchanged():
    _curve, ev = R.trunc_curves(anakit.trunc_rows())
    by = {(e["arm"], e["panel"], e["panel_type"], e["seed"]): e for e in ev}
    assert by[("C_SMALL", "WARM_TRUNCATION_16384", "FIXED_20", 2026)]["events_to_90"] == 10
    assert by[("LO", "COLD_H_16384", "ALL_AVAILABLE", 2026)]["events_to_90"] == "NOT_REACHED"


@nc("the pre-B4 first-hit rule (a missing C_ref skipped, the series read as NOT_REACHED) must not pass the NO_C_REF "
    "property")
def test_nc_missing_cref_reads_not_reached(monkeypatch):
    def old(seq, x):
        hit = next((h for h, m, c, _R in seq if c is not None and m >= x * c), None)
        return hit if hit is not None else "NOT_REACHED"
    monkeypatch.setattr(R, "_first_hit_or_status", old)
    check_no_c_ref()
