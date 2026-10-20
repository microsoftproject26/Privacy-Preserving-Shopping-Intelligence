"""TRAIN-03 — the staggered-P3 conditions (company/reviews/STAGGERED_P3_FABLE_RULING.md).

  * condition 1, seal by construction: a synthetic MAIN contest run (scripted distinctive values, then the real v4eval
    path) writes NO eval metric value — nor a training loss / gradient value — in any file outside <run_dir>/sealed/
    (every recorded value is searched for in every unsealed file); STATUS / EVENTS / the unsealed metrics.jsonl / the
    public RUN_SUMMARY carry only allowed keys; TUNING, CALIBRATION, the screen and the tie-break keep no sealed dir;
  * condition 4, automatic continuation: STILL_IMPROVING at 6.0 continues in the same run under the C3 rule (an S1
    recipe too), only the boolean is public, the rows are PRACTICAL_BEST_EXTENDED (sealed) and never T1 evidence;
    resume-exact inside the automatic continuation; the RunConfig rules (non-test: automatic, no budget file);
  * condition 2, the unseal gate: four complete runs; both G1b parts; the GRU entry byte-identical in the v1 and current
    02b; the four RunConfig hashes match the pinned recipes; a sealed read without a token is refused;
  * the G1B_APPROVED format (tuned_recipes.g1b_obj / load_g1b);
  * the P3 queue driver (sequential / parallel, --code-root, resume / skip, public states only).
Negative controls: an unsealed contest run; each unseal clause removed (4); a driver that prints a value.
"""
from __future__ import annotations

import dataclasses
import json
import re
import shutil
from pathlib import Path

import pytest

import p2kit
import v4train.trainer as T
import v4train.tuned_recipes as TR
import v4train.unseal as U
import v4train.p3_driver as P3
from p2kit import make_p2, make_runner, under_v4
from trainkit import (NEW_DATA_VERSION, K, art, assert_raises, cfg, interrupt_after, nc, providers, raises_with, read,
                      scripted, trainer, write_registry)
from test_train02_tuning import tcfg
from v4models import build
from v4train.evalconfig import frozen
from v4train.recipe import RecipeError, RunConfig
from v4train.schedule import eval_marks
from v4train.selection import SelectionError, select_family

ALL_MARKS = eval_marks("MAIN") + eval_marks("CONTINUATION")
DISTINCT = {m: round(0.1 + 0.021 * i + 0.000123457, 9) for i, m in enumerate(ALL_MARKS)}     # always improving
FLAT = {m: (0.2 + 0.01 * i if m <= 3.0 else 0.2 + 0.01 * 5) for i, m in enumerate(ALL_MARKS)}  # best at 3.0
ALLOWED_STATUS = {"run_id", "status", "phase", "updates", "efe", "utc", "last_mark", "attempt", "still_improving",
                  "continuation", "error", "recovered", "last_snapshot"}
FORBIDDEN_KEYS = ("\"value\"", "last_value", "loss_mean", "grad_norm", "improved", "practical_best", "controlled_budget",
                  "secondary", "ndcg", "mrr")


@pytest.fixture(autouse=True)
def _p2_cleanup():
    yield
    p2kit.cleanup()


def _values_in_sealed(run_dir: Path) -> set:
    """Every metric / loss value recorded in the run's artifact root (sealed/ for a contest run)."""
    root, vals = art(run_dir), set()
    for line in (root / "metrics.jsonl").read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        for k in ("value", "loss_mean", "grad_norm_mean_preclip", "grad_norm_max_preclip", "head_common_mode_norm"):
            if isinstance(r.get(k), float):
                vals.add(r[k])
        vals.update(v for v in (r.get("secondary") or {}).values() if isinstance(v, float))
    for p in list((root / "evals").glob("*.rows.json")) + [q for q in (root / "RESULT_ROWS.json",
                                                                      root / "RESULT_ROWS_EXTENDED.json") if q.exists()]:
        vals.update(r["value"] for r in read(p)["rows"] if isinstance(r.get("value"), float))
    return {v for v in vals if len(repr(v)) >= 8}           # distinctive only (no 0.0 / 1.0 / 6.5 collisions)


def _unsealed_files(run_dir: Path) -> list:
    sealed = Path(run_dir) / "sealed"
    return [p for p in Path(run_dir).rglob("*") if p.is_file() and sealed not in p.parents]


def _check_seal(run_dir: Path) -> None:
    vals = _values_in_sealed(run_dir)
    assert len(vals) > 10, "the run recorded values"
    leaks = []
    for p in _unsealed_files(run_dir):
        text = p.read_bytes().decode("latin-1")
        for v in vals:
            for form in {repr(v), json.dumps(v), f"{v:.6f}"}:
                if re.search(r"(?<![\d.])" + re.escape(form) + r"(?!\d)", text):
                    leaks.append((p.name, form))
        if p.suffix in (".json", ".jsonl"):
            leaks += [(p.name, k) for k in FORBIDDEN_KEYS if k in text]
    assert not leaks, f"metric values / keys outside sealed/: {sorted(set(leaks))[:10]}"
    st = read(Path(run_dir) / "STATUS.json")
    assert set(st) <= ALLOWED_STATUS, set(st) - ALLOWED_STATUS
    for line in (Path(run_dir) / "metrics.jsonl").read_text(encoding="utf-8").splitlines():
        assert set(json.loads(line)) <= {"kind", "attempt", "phase", "update", "exposures", "efe", "window_updates",
                                         "step_wall_s", "data_wait_s", "mark", "loss_finite", "utc"}


def _contest_run(work, name="c", vals=DISTINCT, **kw):
    kw.setdefault("auto_continuation", True)
    tr = trainer(work / name, cfg(name, "GRU", **kw), eval_override=scripted(vals))
    return tr, tr.run()


def test_seal_by_construction_scripted(work):
    tr, s = _contest_run(work)
    assert tr.sealed and (work / "c" / "sealed").is_dir() and s["status"] == "DONE_CONTINUATION"
    _check_seal(work / "c")
    pub = read(work / "c" / "RUN_SUMMARY.json")
    assert pub["sealed"] is True and pub["still_improving"] is True and pub["status"] == "DONE_CONTINUATION"
    assert pub["marks_done"] == [m for m in ALL_MARKS if m <= s["continuation_stop_mark"]]


@nc("a contest run that writes its values outside sealed/ (the pre-TRAIN-03 layout)")
def test_nc_unsealed_contest_run(work, monkeypatch):
    monkeypatch.setattr(RunConfig, "sealed_outputs", property(lambda self: False))                 # MUTANT
    monkeypatch.setattr(RunConfig, "effective_auto_continuation", property(lambda self: False))
    tr = trainer(work / "c", cfg("c", "GRU"), eval_override=scripted(DISTINCT))
    tr.run()
    _check_seal(work / "c")


def test_seal_real_evaluator(work):
    tr = trainer(work / "r", cfg("r", "SASREC"))
    tr.run()
    assert (work / "r" / "sealed" / "evals").is_dir() and not (work / "r" / "evals").exists()
    assert not (work / "r" / "checkpoints").exists() and (work / "r" / "sealed" / "CHECKPOINTS.json").is_file()
    _check_seal(work / "r")


def test_other_roles_keep_their_layout(work):
    t = trainer(work / "t", tcfg("t", "P2_GRU_D256_S1_s4"), eval_override=scripted({2.0: 0.1, 4.0: 0.2, 6.0: 0.3}))
    t.run()
    c = trainer(work / "cal", cfg("cal", "GRU", phase="CALIBRATION"), eval_override=scripted(FLAT))
    c.run()
    f = trainer(work / "tb", cfg("tb", "GRU", population="FULL_TRAIN"), eval_override=scripted(FLAT))
    f.run()
    for d in ("t", "cal", "tb"):
        assert not (work / d / "sealed").exists(), d
    assert (work / "cal" / "evals").is_dir() and "value" in (work / "cal" / "metrics.jsonl").read_text(encoding="utf-8")
    assert not t.sealed and not c.sealed and not f.sealed
    assert not RunConfig(**{**_base(), "role": "FEATURE_SEARCH_SCREEN"}).sealed_outputs


def _base():
    from trainkit import KEY, registry_keys
    return dict(run_id="x", family="GRU", seed=2026, exposure_point="end", cuda_visible_devices="0",
                determinism_mode="STRICT", key=dict(KEY), **registry_keys("FEATURE_SEARCH_SCREEN"))


# ------------------------------------------------------------------------------------------------ condition 4
def test_automatic_continuation_and_extended_rows(work):
    tr, s = _contest_run(work)
    ev = [json.loads(x) for x in (work / "c" / "EVENTS.jsonl").read_text(encoding="utf-8").splitlines()]
    dm = [e for e in ev if e["event"] == "DONE_MAIN"][0]
    cs = [e for e in ev if e["event"] == "CONTINUATION_START"][0]
    assert dm["still_improving"] is True and dm["continuation"] == "AUTO_C3"
    assert cs["automatic"] is True and cs["ceiling"] == 8.0 and "C3" in cs["rule"]
    ext = read(art(work / "c") / "RESULT_ROWS_EXTENDED.json")["rows"]
    assert ext and {r["selection_rule"] for r in ext} == {"PRACTICAL_BEST_EXTENDED"}
    main = read(art(work / "c") / "RESULT_ROWS.json")["rows"]
    rows = [dict(r, run_label="PRODUCTION") for r in main + ext]
    raises_with(SelectionError, "never T1 selection evidence", select_family, rows, delta_bar_ci=(-1.0, 1.0))
    t2, s2 = _contest_run(work, "flat", vals=FLAT)
    assert s2["status"] == "DONE_MAIN" and s2["still_improving"] is False
    assert read(work / "flat" / "RUN_SUMMARY.json")["still_improving"] is False
    assert not any(json.loads(x)["event"] == "CONTINUATION_START"
                   for x in (work / "flat" / "EVENTS.jsonl").read_text(encoding="utf-8").splitlines())


def test_resume_exact_inside_the_automatic_continuation(work):
    _contest_run(work, "full")
    a = trainer(work / "cut", cfg("cut", "GRU", auto_continuation=True), eval_override=scripted(DISTINCT),
                fault=interrupt_after(47))
    assert_raises(T.InjectedInterruption, a.run)
    assert read(art(work / "cut") / "CHECKPOINTS.json")["latest"]["meta"]["exposures"] > 1200
    s = trainer(work / "cut", cfg("cut", "GRU", auto_continuation=True), eval_override=scripted(DISTINCT)).resume()
    f = read(art(work / "full") / "RUN_SUMMARY.json")
    for k in ("final_state_digest", "status", "continuation_stop_mark", "exposures"):
        assert s[k] == f[k], k
    assert [(e["mark"], e["value"]) for e in s["evals"]] == [(e["mark"], e["value"]) for e in f["evals"]]


def test_s1_recipe_continues_with_the_c3_knots(work, tmp_path):
    out, _ = p2kit.register(tmp_path)
    r = TR.recipe(TR.load_tuned_recipes(out), "02B_GRU")
    c = cfg("s1", "GRU", recipe_id="02B_GRU", recipe_registry=str(out), peak=r["peak_lr"], auto_continuation=True,
            model_variant=r["model_variant"], dropout=tuple(r["dropout"]), schedule=r["schedule"])
    tr = trainer(work / "s1", c, eval_override=scripted(DISTINCT))
    s = tr.run()
    assert r["schedule"] == "S1" and s["status"] == "DONE_CONTINUATION"
    mon = [json.loads(x) for x in (art(work / "s1") / "metrics.jsonl").read_text(encoding="utf-8").splitlines()]
    cont = [m["lr_last"] for m in mon if m["kind"] == "monitor" and m["phase"] == "CONTINUATION"]
    assert cont and all(abs(x - 0.1 * r["peak_lr"]) < 1e-15 or x < 0.1 * r["peak_lr"] for x in cont), \
        "the C3 knots (0.1 x peak, then the cooldown) for an S1 recipe too"


def test_runconfig_continuation_rules():
    from trainkit import KEY, registry_keys
    base = dict(run_id="x", family="GRU", seed=2026, exposure_point="end", cuda_visible_devices="0",
                determinism_mode="STRICT", key=dict(KEY), **registry_keys())
    c = RunConfig(**base)
    assert c.sealed_outputs and c.effective_auto_continuation, "a non-test contest run continues automatically"
    raises_with(RecipeError, "auto_continuation False is refused", RunConfig(**base, auto_continuation=False).validate)
    raises_with(RecipeError, "continuation_budget file is refused", RunConfig(**base, continuation_budget="b.json")
                .validate)
    t = RunConfig(**base, test_overrides=True)
    assert not t.effective_auto_continuation and RunConfig(**base, test_overrides=True,
                                                           auto_continuation=True).effective_auto_continuation
    raises_with(RecipeError, "contest run only", RunConfig(**{**base, "phase": "CALIBRATION"}, auto_continuation=True)
                .validate)
    assert "auto_continuation" not in c.as_json(), "left out of the config hash while None"


# ------------------------------------------------------------------------------------------------ G1B format
def _parts(*hashes):
    return [{"part": i + 1, "family": ("GRU", "SASREC")[i], "self_sha256": h} for i, h in enumerate(hashes)]


def test_g1b_format():
    g = TR.g1b_obj(_parts("a" * 64, "b" * 64))
    assert g["tuned_recipes_self_sha256"] == "b" * 64 and g["ruling"]["path"] == TR.RULING_PATH
    TR.load_g1b_obj(g)
    bad = [dict(g, tuned_recipes_self_sha256="a" * 64),
           dict(g, tuned_recipes_parts=[dict(p, family="SASREC") if p["part"] == 1 else p for p in g["tuned_recipes_parts"]]),
           dict(g, ruling={"path": TR.RULING_PATH, "sha256": "0" * 64}), dict(g, format="OTHER"),
           dict(g, tuned_recipes_parts=[])]
    for b in bad:
        assert_raises(TR.TunedRecipeError, TR.load_g1b_obj, b)


def _interrupted_gru_across_part2(tmp_path, monkeypatch, *, part1r=False):
    """Start a non-test GRU run at G1b part 1, then register and pin part 2."""
    monkeypatch.setattr(T, "runtime_device_class", lambda device: frozen().device_class)
    p2 = make_p2(tmp_path / "p2", ("GRU", "SASREC"))
    out, gate = tmp_path / "TUNED_RECIPES_02B.json", tmp_path / "G1B_APPROVED.json"
    from v4train.p2_driver import register_02b
    w1 = register_02b(p2, out, families=("GRU",))
    gate.write_text(json.dumps(TR.g1b_obj(_parts(w1["self_sha256"]))), encoding="utf-8")
    monkeypatch.setattr(TR, "REGISTRY_PATH", out)
    monkeypatch.setattr(TR, "G1B_GATE", gate)
    r1 = TR.load_tuned_recipes(out)["recipes"]["02B_GRU"]
    tp, ep = providers(n=1024)
    mreg = write_registry({"MAIN": (tp, ep), "CONTINUATION": (tp, ep)}, data_version=NEW_DATA_VERSION)
    theta = build("GRU", p2kit_schema(), 2026, K=K, variant=r1["model_variant"]).init_sha256
    c = cfg("P3_GRU_2026", "GRU", manifest_registry=mreg, data_version=NEW_DATA_VERSION,
            effective_batch=256, checkpoint_every=1, recipe_id="02B_GRU", recipe_registry=str(out),
            peak=r1["peak_lr"], model_variant=r1["model_variant"], dropout=tuple(r1["dropout"]),
            schedule=r1["schedule"], theta0_init_sha256=theta)
    c = dataclasses.replace(c, test_overrides=False)
    d = tmp_path / "run"
    first = trainer(d, c, tp, ep, eval_override=scripted(FLAT), fault=interrupt_after(2))
    assert_raises(T.InjectedInterruption, first.run)
    before = read(d / "RUN_START.json")
    assert before["tuned_recipe"]["recipe_sha256"] == r1["sha256"]
    w2 = (register_02b(p2, out, families=("GRU",), c2_only=True) if part1r
          else register_02b(p2, out, families=("SASREC",)))
    parts = ([{"part": 1, "family": "GRU", "self_sha256": w1["self_sha256"]},
              {"part": "1R", "family": "GRU", "self_sha256": w2["self_sha256"]}] if part1r
             else _parts(w1["self_sha256"], w2["self_sha256"]))
    gate.write_text(json.dumps(TR.g1b_obj(parts)), encoding="utf-8")
    reg2 = TR.load_tuned_recipes(out)
    assert reg2["self_sha256"] != w1["self_sha256"]
    return d, c, tp, ep, before, r1, reg2


def test_production_gru_resume_across_g1b_part2(tmp_path, monkeypatch):
    d, c, tp, ep, before, r1, reg2 = _interrupted_gru_across_part2(tmp_path, monkeypatch)
    assert reg2["recipes"]["02B_GRU"] == r1
    resumed = trainer(d, c, tp, ep, eval_override=scripted(FLAT))
    assert resumed.config_sha256 == before["config_sha256"]
    assert resumed.tuned_recipe["recipe_sha256"] == r1["sha256"]
    assert resumed.tuned_recipe["g1b_pin"]["pinned"] == reg2["self_sha256"]
    assert resumed.resume()["status"] == "DONE_MAIN"
    assert read(d / "RUN_START.json")["config_sha256"] == before["config_sha256"]


@nc("a GRU recipe changed by part 2 must not resume under the old run identity")
def test_nc_production_resume_with_changed_gru_recipe(tmp_path, monkeypatch):
    d, c, tp, ep, _, r1, _ = _interrupted_gru_across_part2(tmp_path, monkeypatch)
    original = TR.recipe

    def changed_recipe(reg, recipe_id):
        r = original(reg, recipe_id)
        if recipe_id == "02B_GRU":
            r["sha256"] = "f" * 64 if r1["sha256"] != "f" * 64 else "e" * 64
        return r

    monkeypatch.setattr(TR, "recipe", changed_recipe)  # MUTANT: part-2 changed the GRU entry
    resumed = trainer(d, c, tp, ep, eval_override=scripted(FLAT))
    # MUTANT: if resume ignored the recipe identity, this assertion would fail.
    old_check = T.CentralTrainer._check_start_identity

    def check_without_recipe(self):
        current = self.tuned_recipe
        self.tuned_recipe = dict(current, recipe_sha256=r1["sha256"])
        try:
            return old_check(self)
        finally:
            self.tuned_recipe = current

    monkeypatch.setattr(T.CentralTrainer, "_check_start_identity", check_without_recipe)
    raises_with(T.TrainerError, "tuned_recipe_sha256", resumed.resume)


# ------------------------------------------------------------------------------------------------ condition 2
@pytest.fixture(scope="module")
def contest(tmp_path_factory):
    """Four non-test contest runs of 02b recipes (GRU from part 1, SASREC from part 2), complete, plus the G1b file."""
    base = tmp_path_factory.mktemp("contest")
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(T, "runtime_device_class", lambda device: frozen().device_class)
        croot = p2kit.contest_root()               # under V4; per session (FIX-DELTA-01 F5); not the per-test cleanup
        shutil.rmtree(croot, ignore_errors=True)
        p2 = make_p2(croot / "p2", ("GRU", "SASREC"))
        out = base / "TUNED_RECIPES_02B.json"
        from v4train.p2_driver import register_02b
        w1 = register_02b(p2, out, families=("GRU",))
        w2 = register_02b(p2, out, families=("SASREC",))
        gate = base / "G1B_APPROVED.json"
        gate.write_text(json.dumps(TR.g1b_obj(_parts(w1["self_sha256"], w2["self_sha256"]))), encoding="utf-8")
        mp.setattr(TR, "REGISTRY_PATH", out)
        mp.setattr(TR, "G1B_GATE", gate)
        reg = TR.load_tuned_recipes(out)
        dirs = {}
        for fam in ("GRU", "SASREC"):
            r = reg["recipes"][f"02B_{fam}"]
            for seed in (2026, 2027):
                tp, ep = providers(n=1024)
                mreg = write_registry({"MAIN": (tp, ep), "CONTINUATION": (tp, ep)}, data_version=NEW_DATA_VERSION)
                theta = build(fam, p2kit_schema(), seed, K=K, variant=r["model_variant"]).init_sha256
                c = cfg(f"P3_{fam}_{seed}", fam, seed=seed, manifest_registry=mreg, data_version=NEW_DATA_VERSION,
                        effective_batch=256, recipe_id=r["recipe_id"], recipe_registry=str(out), peak=r["peak_lr"],
                        model_variant=r["model_variant"], dropout=tuple(r["dropout"]), schedule=r["schedule"],
                        theta0_init_sha256=theta)
                c = dataclasses.replace(c, test_overrides=False)
                d = base / "runs" / f"{fam}_{seed}"
                s = trainer(d, c, tp, ep, eval_override=scripted(FLAT)).run()
                assert s["status"] == "DONE_MAIN" and s["still_improving"] is False and s["run_label"] == "PRODUCTION"
                dirs[(fam, seed)] = d
    yield {"base": base, "dirs": dirs, "gate": gate, "out": out, "p2": p2, "w1": w1, "w2": w2}
    shutil.rmtree(croot, ignore_errors=True)
    try:
        croot.parent.rmdir()                               # only when no other session's contest dir is left
    except OSError:
        pass


def p2kit_schema():
    from trainkit import SCHEMA
    return SCHEMA


def _copy(contest, tmp_path):
    dirs = {}
    for k, d in contest["dirs"].items():
        dst = tmp_path / "runs" / d.name
        shutil.copytree(d, dst)
        dirs[k] = dst
    return dirs


def _gate(contest, dirs=None, gate=None, out=None):
    return U.unseal_gate(dirs or contest["dirs"], gate_path=gate or contest["gate"], tuned_path=out or contest["out"])


def test_unseal_gate_passes_and_guards_every_sealed_read(contest):
    d = contest["dirs"][("GRU", 2026)]
    from v4train.bootstrap_ci import practical_best_ranks
    raises_with(U.UnsealRefused, "is sealed", U.read_sealed_json, None, d, "RESULT_ROWS.json")
    raises_with(U.UnsealRefused, "is sealed", practical_best_ranks, d)
    tok = _gate(contest)
    rows = U.read_sealed_json(tok, d, "RESULT_ROWS.json")["rows"]
    assert rows and {r["run_label"] for r in rows} == {"PRODUCTION"} and rows[0]["recipe_id"] == "02B_GRU"
    assert set(tok.record["configs"]) == {"GRU/2026", "GRU/2027", "SASREC/2026", "SASREC/2027"}
    raises_with(U.UnsealRefused, "test runs only", U.smoke_token, [d])
    forged = U.UnsealToken(frozenset([str(d.resolve())]), {}, object())
    raises_with(U.UnsealRefused, "is sealed", U.read_sealed_json, forged, d, "RESULT_ROWS.json")


def _check_clause1(contest, tmp_path):
    three = {k: v for k, v in contest["dirs"].items() if k != ("SASREC", 2027)}
    raises_with(U.UnsealRefused, "exactly the four contest runs", _gate, contest, three)
    dirs = _copy(contest, tmp_path)
    p = dirs[("SASREC", 2027)] / "RUN_SUMMARY.json"
    pub = read(p)
    pub["still_improving"] = True                                   # an automatic continuation still pending
    p.write_text(json.dumps(pub), encoding="utf-8")
    raises_with(U.UnsealRefused, "not complete", _gate, contest, dirs)


def test_unseal_clause1_all_four_complete(contest, tmp_path):
    _check_clause1(contest, tmp_path)


@nc("an unseal gate without the completeness clause")
def test_nc_unseal_without_clause1(contest, tmp_path, monkeypatch):
    monkeypatch.setattr(U, "clause_runs_complete", lambda run_dirs: {})                              # MUTANT
    _check_clause1(contest, tmp_path)


def _check_clause2(contest, tmp_path):
    g = tmp_path / "G1B_part1_only.json"
    g.write_text(json.dumps(TR.g1b_obj(_parts(contest["w1"]["self_sha256"]), format=TR.G1B_FORMAT_1)),
                 encoding="utf-8")
    raises_with(U.UnsealRefused, "part 1 (GRU) AND part 2", _gate, contest, None, g)
    raises_with(U.UnsealRefused, "does not exist", _gate, contest, None, tmp_path / "none.json")


def test_unseal_clause2_both_g1b_parts(contest, tmp_path):
    _check_clause2(contest, tmp_path)


@nc("an unseal gate that accepts a G1b file with one part")
def test_nc_unseal_without_clause2(contest, tmp_path, monkeypatch):
    monkeypatch.setattr(U, "clause_g1b_parts", lambda gp: TR.load_g1b(gp))                          # MUTANT
    _check_clause2(contest, tmp_path)


def _check_clause3(contest, tmp_path):
    """A current registry whose 02B_GRU entry differs from the part-1 (v1) one (a GRU recipe from another P2 table),
    pinned consistently as part 2: refused."""
    work2 = tmp_path / "reg"
    work2.mkdir()
    shutil.copy(contest["out"].with_name("TUNED_RECIPES_02B.v1.json"), work2 / "TUNED_RECIPES_02B.v1.json")
    obj = read(contest["out"])
    p2b = make_p2(tmp_path / "p2b", ("GRU",), run=make_runner(0.003))
    hp = TR.load_hp_registry()
    obj["recipes"]["02B_GRU"] = TR.family_recipe("GRU", p2b / "P2_GRU_STAGE1.json", p2b / "P2_GRU_STAGE2.json", hp)
    obj["self_sha256"] = TR.canonical_sha256(obj)
    cur = work2 / "TUNED_RECIPES_02B.json"
    cur.write_text(json.dumps(obj, indent=1, sort_keys=True), encoding="utf-8")
    g = tmp_path / "G1B.json"
    g.write_text(json.dumps(TR.g1b_obj(_parts(contest["w1"]["self_sha256"], obj["self_sha256"]))), encoding="utf-8")
    raises_with(U.UnsealRefused, "not byte-identical", _gate, contest, None, g, cur)


def test_unseal_clause3_gru_entry_byte_identical(contest, tmp_path):
    _check_clause3(contest, tmp_path)


@nc("an unseal gate without the GRU byte-identity clause")
def test_nc_unseal_without_clause3(contest, tmp_path, monkeypatch):
    monkeypatch.setattr(U, "clause_gru_identical",                                                   # MUTANT
                        lambda gate, tuned_path, hp_path: TR.load_tuned_recipes(tuned_path, hp_path))
    _check_clause3(contest, tmp_path)


def _check_clause4(contest, tmp_path):
    dirs = _copy(contest, tmp_path / "a")
    p = dirs[("GRU", 2027)] / "RUN_START.json"
    st = read(p)
    st["config"]["peak_lr"] = 1e-3                                  # edited, stale config hash
    p.write_text(json.dumps(st), encoding="utf-8")
    raises_with(U.UnsealRefused, "RunConfig hash does not match", _gate, contest, dirs)
    dirs = _copy(contest, tmp_path / "b")
    p, q = dirs[("SASREC", 2026)] / "RUN_START.json", dirs[("SASREC", 2026)] / "RUN_SUMMARY.json"
    st, pub = read(p), read(q)
    st["config"]["dropout"] = [0.3]                                 # re-hashed consistently, but not the pinned recipe
    st["config_sha256"] = U._config_sha256(st["config"])
    pub["config_sha256"] = st["config_sha256"]
    p.write_text(json.dumps(st), encoding="utf-8")
    q.write_text(json.dumps(pub), encoding="utf-8")
    raises_with(U.UnsealRefused, "not the pinned recipe's", _gate, contest, dirs)


def test_unseal_clause4_config_hashes(contest, tmp_path):
    _check_clause4(contest, tmp_path)


@nc("an unseal gate without the RunConfig-hash clause")
def test_nc_unseal_without_clause4(contest, tmp_path, monkeypatch):
    monkeypatch.setattr(U, "clause_config_hashes", lambda run_dirs, reg, gate: {})                   # MUTANT
    _check_clause4(contest, tmp_path)


# ------------------------------------------------------------------------------------------------ P3 driver
class FakeProc:
    log = []

    def __init__(self, cmd, **kw):
        self.cmd = cmd
        FakeProc.log.append(("start", cmd[cmd.index("--seed") + 1]))
        rd = Path(cmd[cmd.index("--run-dir") + 1])
        rd.mkdir(parents=True, exist_ok=True)
        (rd / "RUN_START.json").write_text("{}", encoding="utf-8")
        (rd / "RUN_SUMMARY.json").write_text(json.dumps({"status": "DONE_MAIN", "still_improving": False, "sealed": True,
                                                         "marks_done": [0.5], "config_sha256": "c" * 64}),
                                             encoding="utf-8")
        (rd / "STATUS.json").write_text(json.dumps({"status": "DONE_MAIN"}), encoding="utf-8")

    def wait(self):
        FakeProc.log.append(("wait", self.cmd[self.cmd.index("--seed") + 1]))
        return 0


def _args(tmp_path, gpus, **kw):
    import argparse
    return argparse.Namespace(family=kw.get("family", "GRU"), gpus=gpus, out_dir=tmp_path / "p3",
                              code_root=kw.get("code_root", P3.CODE), seeds=[2026, 2027], recipe_id=None,
                              tuned_recipes=None, history_policy="USER_HISTORY_L50", dry_run=kw.get("dry", False))


def test_p3_driver_plan_and_modes(tmp_path):
    lines = []
    assert P3.run_queue(_args(tmp_path, [0], dry=True, code_root=Path("/srv/v4/code_p3")), say=lines.append) == 0
    assert len(lines) == 2 and all("--phase MAIN" in ln and "--recipe-id 02B_GRU" in ln and "--gpu 0" in ln
                                   for ln in lines)
    assert all(str(Path("/srv/v4/code_p3") / "v4int" / "launch_central.py") in ln for ln in lines)
    two = P3.plan(_args(tmp_path, [0, 1], family="SASREC"))
    assert [j["gpu"] for j in two] == [0, 1] and [j["seed"] for j in two] == [2026, 2027]
    assert_raises(P3.P3Error, P3.plan, _args(tmp_path, [0, 0]))
    FakeProc.log = []
    out = []
    assert P3.run_queue(_args(tmp_path / "s", [0]), popen=FakeProc, say=out.append) == 0
    assert FakeProc.log == [("start", "2026"), ("wait", "2026"), ("start", "2027"), ("wait", "2027")], "sequential"
    FakeProc.log = []
    assert P3.run_queue(_args(tmp_path / "p", [0, 1]), popen=FakeProc, say=out.append) == 0
    assert FakeProc.log[:2] == [("start", "2026"), ("start", "2027")], "parallel: both launched before waiting"
    FakeProc.log = []
    assert P3.run_queue(_args(tmp_path / "s", [0]), popen=FakeProc, say=out.append) == 0
    assert FakeProc.log == [], "finished runs are not launched again"
    rd = P3.run_dir_of(tmp_path / "r" / "p3", "GRU", "02B_GRU", 2026)
    rd.mkdir(parents=True)
    (rd / "RUN_START.json").write_text("{}", encoding="utf-8")
    assert "--resume" in P3.plan(_args(tmp_path / "r", [0]))[0]["cmd"]
    rec = json.loads(sorted((tmp_path / "s" / "p3").glob("P3_GRU_QUEUE_*.json"))[0].read_text(encoding="utf-8"))
    assert rec["mode"] == "sequential" and all("value" not in json.dumps(j["public_state"]) for j in rec["jobs"])
    assert P3.child_env().get("CUDA_VISIBLE_DEVICES") is None
