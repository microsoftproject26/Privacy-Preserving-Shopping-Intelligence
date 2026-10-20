"""G2-01: tools/g2_run.py on synthetic contests (registry / G1B / unseal-valid runs from the TRAIN-05 fixtures).

Negative controls (test_nc_*, strict xfail on AssertionError): each runs a key-property check against a mutant."""
from __future__ import annotations

import builtins
import hashlib
import io
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import pytest

import g2_run as G
import g2kit
from trainkit import nc
from v4train import bootstrap_ci as BC, selection as S, unseal as U

FLOAT = re.compile(r"\d\.\d{3,}")


@pytest.fixture(scope="session")
def reg_full(tmp_path_factory, _clean_work):
    return g2kit.register(tmp_path_factory.mktemp("reg_full"))


@pytest.fixture(scope="session")
def reg_noc2(tmp_path_factory, _clean_work):
    return g2kit.register(tmp_path_factory.mktemp("reg_noc2"), sasrec_no_c2=True)


@pytest.fixture(autouse=True)
def _synthetic_counts(monkeypatch):
    monkeypatch.setattr(G, "REGISTERED_COUNTS", dict(g2kit.SYN_COUNTS))     # 400 / 100 / 300 users


@pytest.fixture
def contest(tmp_path, reg_full):
    return g2kit.build_contest(tmp_path, reg_full)


def run(c, *extra, **kw):
    lines = []
    rc = G.main(c.argv(*extra), gate_path=c.gate, tuned_path=c.tuned, manifest_loader=c.loader,
                say=lines.append, **kw)
    return rc, "\n".join(lines)


def result(c) -> dict:
    return json.loads((c.out / G.RESULT_NAME).read_text(encoding="utf-8"))


def norm(obj):
    return json.loads(G.dumps(obj))


def _leaks(text: str, values) -> list:
    hits = [m.group(0) for m in FLOAT.finditer(text)]
    hits += [repr(v) for v in values if isinstance(v, float) and repr(v) in text]
    return hits


# ------------------------------------------------------------------------------------------------ preflight checks
def spy_sealed(monkeypatch) -> list:
    """Record (and for the library readers, refuse) any sealed read."""
    seen = []
    real_open, real_load = io.open, np.load

    def is_sealed(f):
        try:
            return "sealed" in Path(os.fspath(f)).parts
        except TypeError:
            return False

    def guarded_open(file, *a, **k):
        if is_sealed(file):
            seen.append(str(file))
        return real_open(file, *a, **k)

    def guarded_load(file, *a, **k):
        if is_sealed(file):
            seen.append(str(file))
        return real_load(file, *a, **k)

    monkeypatch.setattr(io, "open", guarded_open)
    monkeypatch.setattr(builtins, "open", guarded_open)
    monkeypatch.setattr(np, "load", guarded_load)
    for mod, name in ((U, "read_sealed_json"), (U, "sealed_root"), (BC, "practical_best_ranks")):
        def boom(*a, _n=name, **k):
            seen.append(_n)
            raise AssertionError(f"sealed reader {_n} called in preflight")
        monkeypatch.setattr(mod, name, boom)
    return seen


def check_preflight_clean(c, monkeypatch):
    seen = spy_sealed(monkeypatch)
    rc, out = run(c, "--preflight")
    assert seen == [], f"preflight read sealed files: {seen}"
    assert rc == 0, out
    assert "GATE: PASS" in out and "PREFLIGHT: PASS" in out
    assert _leaks(out, c.values) == [], f"preflight printed values: {_leaks(out, c.values)}"
    assert not c.out.exists() or not any(c.out.iterdir()), "preflight wrote files"
    return out


def check_refuses_extra(c):
    (c.runs_root / "GRU" / "02B_GRU_s2028").mkdir()
    rc, out = run(c, "--preflight")
    assert rc == G.REFUSED and "extra" in out and "02B_GRU_s2028" in out, out
    rc, out = run(c)
    assert rc == G.REFUSED and not (c.out / G.RESULT_NAME).exists()


def test_preflight_passes_without_sealed_read(contest, monkeypatch):
    out = check_preflight_clean(contest, monkeypatch)
    assert out.count("RUN ") == 8 and "for 8 runs" in out


def test_preflight_refuses_missing_run(contest):
    import shutil
    shutil.rmtree(contest.runs[("SASREC", "02B_SASREC_C2", 2027)])
    rc, out = run(contest, "--preflight")
    assert rc == G.REFUSED and "missing" in out and "02B_SASREC_C2/2027" in out


def test_preflight_refuses_unfinished_run(contest):
    rd = contest.runs[("GRU", "02B_GRU", 2026)]
    pub = json.loads((rd / "RUN_SUMMARY.json").read_text())
    pub["still_improving"] = True                  # DONE_MAIN with a continuation pending: not complete
    (rd / "RUN_SUMMARY.json").write_text(json.dumps(pub))
    rc, out = run(contest, "--preflight")
    assert rc == G.REFUSED and "GATE: REFUSED" in out and "complete=False" in out
    rc, out = run(contest)
    assert rc == G.REFUSED and not (contest.out / G.RESULT_NAME).exists()


def test_preflight_refuses_extra_run(contest):
    check_refuses_extra(contest)


def test_preflight_refuses_missing_sealed_dir(contest):
    import shutil
    shutil.rmtree(contest.runs[("GRU", "02B_GRU_C2", 2027)] / "sealed")
    rc, out = run(contest, "--preflight")
    assert rc == G.REFUSED and "sealed_dir=False" in out


# ------------------------------------------------------------------------------------------------ full run
def direct(c) -> dict:
    """select_g2_candidates called directly (no runner) on the same sealed files."""
    token = U.unseal_gate(c.runs, gate_path=c.gate, tuned_path=c.tuned)
    rows = [r for rd in c.runs.values() for r in json.loads((rd / "sealed" / "RESULT_ROWS.json").read_text())["rows"]]
    within = S.within_family_candidates(rows, c.reg["recipes"], unseal_token=token)
    sel = {f: within[f]["selected_recipe_id"] for f in S.FAMILIES}
    ranks = {(f, s): BC.practical_best_ranks(c.runs[(f, sel[f], s)], token) for f in S.FAMILIES for s in (2026, 2027)}
    ci = BC.delta_bar_ci(c.manifest, ranks)
    return S.select_g2_candidates(rows, c.reg["recipes"], unseal_token=token, delta_bar_ci=ci["ci95"])


def check_reproduces_direct(c):
    rc, out = run(c)
    got, want = result(c), norm(direct(c))
    assert got["between_families"] == want["between_families"]
    for fam in S.FAMILIES:
        for k in ("selected_recipe_id", "means", "excluded", "labels"):
            assert got["within_family"][fam][k] == want["within_family"][fam][k], (fam, k)
    return rc, out, got


def test_full_run_reproduces_direct_call(contest):
    rc, out, got = check_reproduces_direct(contest)
    assert rc == 0 and got["outcome"] == "DECISIVE_STABLE"
    assert got["selected_recipe_ids"] == {"GRU": "02B_GRU_C2", "SASREC": "02B_SASREC_C2"}
    assert got["between_families"]["winner_family"] == "SASREC"
    assert got["within_family"]["SASREC"]["labels"] == {"02B_SASREC": "CANDIDATE_NOT_SELECTED",
                                                         "02B_SASREC_C2": "SELECTED"}
    assert set(got["code_files_sha256"]) == {f"v4train/{n}.py" for n in G.LIB_FILES}
    assert got["script"]["sha256"] == G.script_sha256() and len(got["inputs"]["per_run"]) == 8
    assert got["unseal_token"]["record"]["parameter_counts"]["02B_GRU_C2"] == 10
    assert got["checks"]["rank_row_rows_checked"] == 8 * 4 and got["checks"]["rank_row_max_abs_diff"] <= 1e-9
    assert got["manifest"]["counts"] == g2kit.SYN_COUNTS
    assert got["manifest"]["calib_users_main_keys_sha256"] == hashlib.sha256(g2kit.CALIB_MAIN_USERS.tobytes()).hexdigest()
    assert got["diagnostic"]["between_families"]["flip"] is False
    assert got["lock_draft"]["written"] is True and got["lock_draft"]["recipe_id"] == "02B_SASREC_C2"
    started = json.loads((contest.out / G.STARTED_NAME).read_text())
    assert started["script_sha256"] == G.script_sha256() and len(started["runs"]) == 8
    assert got["started_record_sha256"] == hashlib.sha256((contest.out / G.STARTED_NAME).read_bytes()).hexdigest()
    text = (contest.out / G.RESULT_NAME).read_bytes()
    assert b"\r\n" not in text and b"NaN" not in text and json.loads(text) == got
    assert _leaks(out, contest.values) == [], "console printed values without --print-values"
    assert "G2 WITHIN GRU: SELECTED 02B_GRU_C2 | CANDIDATE_NOT_SELECTED 02B_GRU" in out


def test_print_values_prints_values(contest):
    rc, out = run(contest, "--print-values")
    assert rc == 0 and FLOAT.search(out)                  # the leak check above can see a value


@pytest.mark.parametrize("mode,winner", [("tie_c1_slightly_better", "02B_GRU_C2"), ("c1_better", "02B_GRU")])
def test_within_family_tie_rule(tmp_path, reg_full, mode, winner):
    c = g2kit.build_contest(tmp_path, reg_full, g2kit.profile(gru=mode))
    rc, _, got = check_reproduces_direct(c)
    w = got["within_family"]["GRU"]
    assert rc == 0 and w["selected_recipe_id"] == winner
    assert w["means"]["02B_GRU"] > w["means"]["02B_GRU_C2"]       # C1 has the higher mean in both cases
    if mode.startswith("tie"):
        assert w["means"]["02B_GRU"] - w["means"]["02B_GRU_C2"] <= 0.0005   # tie -> the smaller model (10 < 20)


def test_diagnostic_flip_reported(tmp_path, reg_full):
    c = g2kit.build_contest(tmp_path, reg_full, g2kit.profile(gru="flip"))
    rc, out, got = check_reproduces_direct(c)
    assert got["within_family"]["GRU"]["selected_recipe_id"] == "02B_GRU"
    assert got["within_family"]["GRU"]["diagnostic_selected_recipe_id"] == "02B_GRU_C2"
    assert got["within_family"]["GRU"]["diagnostic_flip"] is True
    assert got["diagnostic"]["within_family"]["GRU"]["diagnostic_flip"] is True
    assert "G2 WITHIN GRU:" in out and "diagnostic_flip True" in out


def test_family_with_no_c2(tmp_path, reg_noc2, monkeypatch):
    c = g2kit.build_contest(tmp_path, reg_noc2)
    assert len(c.runs) == 6 and "02B_SASREC_C2" not in c.reg["recipes"]
    out = check_preflight_clean(c, monkeypatch)
    assert "for 6 runs" in out
    monkeypatch.undo()
    monkeypatch.setattr(G, "REGISTERED_COUNTS", dict(g2kit.SYN_COUNTS))
    rc, _, got = check_reproduces_direct(c)
    assert rc == 0 and got["tuned_recipes"]["no_c2"] == ["SASREC"]
    assert got["within_family"]["SASREC"]["labels"] == {"02B_SASREC": "SELECTED"}
    assert got["selected_recipe_ids"]["SASREC"] == "02B_SASREC"


def test_hold_is_never_resolved(tmp_path, reg_full):
    c = g2kit.build_contest(tmp_path, reg_full, g2kit.profile(families_equal=True))
    receipt = tmp_path / "receipt.json"
    receipt.write_text(json.dumps({"source": "SERVER_MEASURED", "host": "h", "torch": "t",
                                   "hashes": {f: {"2026": "a" * 64, "2027": "b" * 64} for f in S.FAMILIES}}))
    rc, out = run(c, "--init-hash-receipt", str(receipt))
    got = result(c)
    assert got["between_families"] == norm(direct(c))["between_families"]
    assert rc == G.EXIT["HOLD"] and got["outcome"] == "HOLD"
    assert got["between_families"]["status"] == "TIEBREAK_REQUIRED" and got["between_families"]["winner_family"] is None
    assert got["hold"]["status"] == "TIEBREAK_REQUIRED"
    assert got["lock_draft"]["written"] is False and not (c.out / S.DRAFT_NAME).exists()
    assert "stage1" not in got["between_families"]                   # no tie-break was run
    assert "G2 BETWEEN: TIEBREAK_REQUIRED" in out


def test_lock_draft_from_run_start_theta0(contest):
    rc, _ = run(contest)
    got = result(contest)
    assert rc == 0 and got["lock_draft"]["written"] is True
    draft = json.loads((contest.out / S.DRAFT_NAME).read_text())
    assert draft["status"] == "DRAFT_NOT_LOCKED" and draft["winner_family"] == "SASREC"
    assert draft["selected_recipe_id"] == "02B_SASREC_C2" and draft["g2_script_sha256"] == G.script_sha256()
    want = {str(s): g2kit.theta0("02B_SASREC_C2", s) for s in (2026, 2027)}
    assert draft["initial_untrained_weights_hash_by_seed"] == want           # the SELECTED recipe, not C1
    assert draft["init_hash_receipt"]["recipe_id"] == "02B_SASREC_C2"
    assert draft["initial_hash_source"]["torch"] == "2.5.1+cu121"
    assert not (contest.out / S.LOCK_NAME).exists()


def test_lock_draft_cross_check_disagreeing_receipt(contest, tmp_path):
    receipt = tmp_path / "receipt.json"
    receipt.write_text(json.dumps({"hashes": {"SASREC": {"2026": "c" * 64, "2027": "d" * 64}}}))
    rc, _ = run(contest, "--init-hash-receipt", str(receipt))
    got = result(contest)
    assert rc == 0 and got["lock_draft"]["written"] is False and "cross-check" in got["lock_draft"]["reason"]
    assert not (contest.out / S.DRAFT_NAME).exists()


def _edit_start(rd, **kw):
    p = rd / "RUN_START.json"
    s = json.loads(p.read_text())
    for k, v in kw.items():
        s[k] = dict(s.get(k) or {}, **v) if isinstance(v, dict) else v
    p.write_text(json.dumps(s))


def _no_gate(*a, **k):
    raise AssertionError("the unseal gate was called")


@pytest.mark.parametrize("edit,text", [
    ({"theta0_init_sha256": "f" * 64}, "initialization_block"),
    ({"key_base": {"eval_manifest_id": "CALIB_VALIDATION_5_100_32768"}}, "eval_manifest_id"),
    ({"key_base": {"manifest_registry_sha256": "0" * 64}}, "neither the loaded REGISTRY_V2"),
    ({"manifest_registry": {"bindings": {"eval": dict(g2kit.ENTRY, manifest_id=g2kit.PRIMARY,
                                                      content_sha256="0" * 64)}}}, "content_sha256")])
def test_public_run_start_checks_refuse_before_unsealing(contest, monkeypatch, edit, text):
    _edit_start(contest.runs[("GRU", "02B_GRU_C2", 2027)], **edit)
    rc, out = run(contest, "--preflight")
    assert rc == G.REFUSED and "PUBLIC CHECKS: REFUSED" in out and text in out, out
    monkeypatch.setattr(U, "unseal_gate", _no_gate)
    rc, out = run(contest)
    assert rc == G.REFUSED and text in out and not (contest.out / G.STARTED_NAME).exists()


def test_trainer_view_registry_route(contest, tmp_path):
    view = tmp_path / "REGISTRY_V2_TRAINER_VIEW.json"
    view.write_text(json.dumps({"derived_from": {"file_sha256": g2kit.REGSHA},
                                "manifests": {g2kit.PRIMARY: dict(g2kit.ENTRY)}}))
    vsha = hashlib.sha256(view.read_bytes()).hexdigest()
    for rd in contest.runs.values():
        _edit_start(rd, key_base={"manifest_registry_sha256": vsha},
                    manifest_registry={"path": str(view), "file_sha256": vsha})
    rc, out = run(contest, "--preflight")
    assert rc == 0 and "registry ['TRAINER_VIEW']" in out, out
    view.write_text(json.dumps({"derived_from": {"file_sha256": "0" * 64},
                                "manifests": {g2kit.PRIMARY: dict(g2kit.ENTRY)}}))
    rc, out = run(contest, "--preflight")
    assert rc == G.REFUSED and "neither the loaded REGISTRY_V2" in out


# ------------------------------------------------------------------------------------------------ CALIB mapping
def check_calib_mapping(c):
    """The diagnostic CALIB users are the true CALIB users in MAIN's key space (via decision_id)."""
    lib = G.bootstrap(g2kit.CODE)
    m = G.calib_mapping(lib, c.manifest, c.calib)
    assert np.array_equal(m["calib_users"], g2kit.CALIB_MAIN_USERS), "wrong CALIB users"
    assert m["counts"] == g2kit.SYN_COUNTS


def test_calib_mapping_through_decision_id(contest):
    check_calib_mapping(contest)
    assert not np.array_equal(np.unique(contest.calib.eval_user_key), g2kit.CALIB_MAIN_USERS)   # separate key spaces


def test_calib_mapping_refusals(contest, monkeypatch):
    m = contest.manifest
    bad = g2kit.calib_manifest(m)
    bad.decision_id = bad.decision_id.copy()
    bad.decision_id[0] += 1                                           # a CALIB decision that MAIN does not hold
    contest.calib = bad
    rc, out = run(contest, "--preflight")
    assert rc == G.REFUSED and "not in MAIN_VALIDATION" in out, out
    contest.calib = g2kit.calib_manifest(m, key_of=lambda k: (k - 150) // 2)     # two MAIN users -> one CALIB key
    rc, out = run(contest, "--preflight")
    assert rc == G.REFUSED and "not 1:1" in out, out
    contest.calib = g2kit.calib_manifest(m, users=g2kit.CALIB_MAIN_USERS[:-1])    # 99 users: counts differ
    rc, out = run(contest, "--preflight")
    assert rc == G.REFUSED and "registered" in out, out
    monkeypatch.setattr(U, "unseal_gate", _no_gate)
    rc, out = run(contest)
    assert rc == G.REFUSED and not (contest.out / G.STARTED_NAME).exists()


def test_diagnostic_rows_use_the_mapped_users(contest):
    from v4eval.metrics import credits_from_ranks, population_block
    rc, _ = run(contest)
    got = result(contest)
    sub = contest.manifest.eval_mask & ~np.isin(contest.manifest.eval_user_key, g2kit.CALIB_MAIN_USERS)
    rd = contest.runs[("GRU", "02B_GRU", 2026)]
    end = np.load(rd / "sealed" / "evals" / "eval_6.0.ranks_int32.npy").astype(np.int64)
    want = population_block(credits_from_ranks(end, (20,)), sub, contest.manifest.eval_user_key)["macro"]["mrr@20"]
    assert rc == 0 and got["diagnostic"]["values_mrr20_macro_e2e"]["CONTROLLED_BUDGET"]["02B_GRU"]["2026"] == want


def test_started_record_survives_a_failed_final_write(contest, monkeypatch):
    real = G._write_new

    def fail_result(path, text):
        if Path(path).name == G.RESULT_NAME:
            raise OSError("disk full")
        return real(path, text)
    monkeypatch.setattr(G, "_write_new", fail_result)
    with pytest.raises(OSError):
        run(contest)
    started = json.loads((contest.out / G.STARTED_NAME).read_text())
    assert started["note"].startswith("attempt record") and len(started["runs"]) == 8
    monkeypatch.undo()
    monkeypatch.setattr(G, "REGISTERED_COUNTS", dict(g2kit.SYN_COUNTS))
    rc, out = run(contest)
    assert rc == G.REFUSED and G.STARTED_NAME in out                  # a second attempt needs a new --out


# ------------------------------------------------------------------------------------------------ P2 smoke tool
def _p2_runs(root, m, base, n_runs=2, tamper=False):
    from v4eval.metrics import credits_from_ranks, population_block
    qhash = m.hashes["eval_ordered_decision_id_sha256"]
    for i in range(n_runs):
        rd = root / "GRU" / "stage1" / f"P2_GRU_D128_S0_s{i}_s2026"
        (rd / "evals").mkdir(parents=True)
        ranks = np.where(m.rankable, base + i, -1).astype(np.int64)
        np.save(rd / "evals" / "eval_6.0.ranks_int32.npy", ranks.astype(np.int32))
        blk = population_block(credits_from_ranks(ranks), m.eval_mask, m.eval_user_key)
        rows = [{"metric": mt, "k": k, "aggregation": "macro", "target_denominator": "E2E", "value": blk["macro"][f"{mt}@{k}"],
                 "value_status": "OK", "selection_rule": "CONTROLLED_BUDGET", "checkpoint_variant": "ENDPOINT",
                 "efe": 6.0, "eval_manifest_id": g2kit.CALIB, "query_order_hash": qhash, "hp_id": "x", "seed": 2026}
                for mt, k in (("mrr", 20), ("ndcg", 10))]
        if tamper and i == 1:
            rows[0]["value"] += 0.001
        (rd / "RESULT_ROWS.json").write_text(json.dumps({"rows": rows}))
        (rd / "RUN_SUMMARY.json").write_text(json.dumps({"status": "DONE_TUNING"}))
    (root / "SASREC" / "stage2" / "P2_SASREC_unfinished_s2027").mkdir(parents=True)


@pytest.mark.parametrize("tamper", [False, True])
def test_g2_smoke_p2(tmp_path, tamper):
    import g2_smoke_p2 as SM
    m, calib, base = g2kit.make_manifest()
    _p2_runs(tmp_path / "runs_p2", calib, base[np.isin(m.decision_id, calib.decision_id)], tamper=tamper)
    lines = []
    rc = SM.main(["--code-root", str(g2kit.CODE), "--runs-root", str(tmp_path / "runs_p2")],
                 calib_loader=lambda lib: calib, say=lines.append)
    out = "\n".join(lines)
    assert "runs checked 2" in out and "query_order_hash match 2/2" in out
    if tamper:
        assert rc == 1 and out.endswith("FAIL") and "runs failed 1" in out and "rows checked 2" in out
    else:
        assert rc == 0 and out.endswith("PASS") and "rows checked 4" in out and "undefined 0" in out
    assert not re.search(r"0\.\d{3,}", out.replace("max_abs_diff", ""))          # no metric value printed


# ------------------------------------------------------------------------------------------------ O_EXCL and guards
def check_o_excl(c, monkeypatch):
    rc, _ = run(c)
    assert rc == 0
    first = (c.out / G.RESULT_NAME).read_bytes()
    try:
        G._write_new(c.out / G.RESULT_NAME, "{}\n")
    except FileExistsError:
        pass
    else:
        raise AssertionError("the writer replaced an existing G2_RESULT.json")
    def no_unseal(*a, **k):
        raise AssertionError("the gate was called although G2_RESULT.json exists")
    monkeypatch.setattr(U, "unseal_gate", no_unseal)
    rc, out = run(c)
    assert rc == G.REFUSED and "already exists" in out
    assert (c.out / G.RESULT_NAME).read_bytes() == first


def test_o_excl_refusal(contest, monkeypatch):
    check_o_excl(contest, monkeypatch)


def test_expect_script_sha256(contest):
    rc, out = run(contest, "--expect-script-sha256", "0" * 64)
    assert rc == G.REFUSED and "frozen" in out and not (contest.out / G.RESULT_NAME).exists()


def test_rank_row_mismatch_is_an_error_not_a_decision(contest):
    rd = contest.runs[("SASREC", "02B_SASREC", 2026)]
    p = rd / "sealed" / "RESULT_ROWS.json"
    obj = json.loads(p.read_text())
    for r in obj["rows"]:
        if r["selection_rule"] == "CONTROLLED_BUDGET" and r["metric"] == "mrr" and r["aggregation"] == "macro":
            r["value"] += 0.01
    p.write_text(json.dumps(obj))
    rc, out = run(contest)
    got = result(contest)
    assert rc == G.EXIT["ERROR"] and got["outcome"] == "ERROR" and "rank file" in got["error"]
    assert "within_family" not in got and "G2 OUTCOME: ERROR" in out


def test_code_root_module_origin(tmp_path, monkeypatch):
    fake = tmp_path / "fake_code"
    (fake / "v4train").mkdir(parents=True)
    for n in G.LIB_FILES:
        (fake / "v4train" / f"{n}.py").write_text("")
    monkeypatch.setattr(sys, "path", list(sys.path))
    with pytest.raises(G.G2Refused, match="imported from"):
        G.bootstrap(fake)
    with pytest.raises(G.G2Refused, match="not a V4 code tree"):
        G.bootstrap(tmp_path / "nothing")


# ------------------------------------------------------------------------------------------------ negative controls
@nc("a preflight that reads a sealed file through the library after the gate")
def test_nc_preflight_reads_sealed_via_library(contest, monkeypatch):
    monkeypatch.setattr(G, "_discard_token", lambda lib, token, runs: lib.U.read_sealed_json(
        token, next(iter(runs.values())), "RUN_SUMMARY.json"))                           # MUTANT
    check_preflight_clean(contest, monkeypatch)


@nc("a preflight that opens a sealed file directly (bypassing the token)")
def test_nc_preflight_opens_sealed_directly(contest, monkeypatch):
    monkeypatch.setattr(G, "_discard_token", lambda lib, token, runs: (
        Path(next(iter(runs.values()))) / "sealed" / "RESULT_ROWS.json").read_text())      # MUTANT
    check_preflight_clean(contest, monkeypatch)


@nc("a preflight that prints a metric value")
def test_nc_preflight_prints_value(contest, monkeypatch):
    real = G._format_run_line
    leak = contest.values[0]
    monkeypatch.setattr(G, "_format_run_line", lambda *a: real(*a) + f" value={leak!r}")   # MUTANT
    check_preflight_clean(contest, monkeypatch)


@nc("discovery that silently accepts an extra run directory")
def test_nc_extra_run_accepted(contest, monkeypatch):
    monkeypatch.setattr(G, "_extra_run_dirs", lambda *a: [])                             # MUTANT
    check_refuses_extra(contest)


@nc("a result writer without O_EXCL")
def test_nc_overwriting_writer(contest, monkeypatch):
    monkeypatch.setattr(G, "_write_new", lambda path, text: Path(path).write_text(text))  # MUTANT
    check_o_excl(contest, monkeypatch)


@nc("a bootstrap CI computed on the C1 ranks instead of the selected recipes")
def test_nc_ci_on_unselected_ranks(contest, monkeypatch):
    monkeypatch.setattr(G, "_selected_ranks", lambda ev, selected: {                    # MUTANT
        (f, s): ev["pb_ranks"][(f, f"02B_{f}", s)] for f in selected for s in (2026, 2027)})
    check_reproduces_direct(contest)


@nc("CALIB users taken by eval_user_key (a per-manifest dense key) instead of the decision_id mapping")
def test_nc_calib_users_by_key_space(contest, monkeypatch):
    def by_keys(lib, main, calib):                                                     # MUTANT (the old bug)
        users = np.intersect1d(np.unique(main.eval_user_key), np.unique(calib.eval_user_key))
        n = int(np.unique(main.eval_user_key).size)
        return {"calib_users": users, "counts": {"main_users": n, "calib_users": int(users.size),
                                                 "disjoint_users": n - int(users.size)},
                "calib_users_main_keys_sha256": ""}
    monkeypatch.setattr(G, "calib_mapping", by_keys)
    check_calib_mapping(contest)
