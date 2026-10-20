"""Amendment 05: candidate registration, three-part G1B, eight-run gate, and G2."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import p2kit
from trainkit import nc, raises_with
from v4train import p2_driver as D, selection as S, tuned_recipes as TR, unseal as U
from v4int import launch_central as L
from test_train03 import _interrupted_gru_across_part2, trainer, scripted, FLAT, read


@pytest.fixture(scope="module")
def registered(tmp_path_factory):
    base = tmp_path_factory.mktemp("train05")
    p2 = p2kit.make_p2(base / "p2")
    out = base / "TUNED_RECIPES_02B.json"
    try:
        w1 = D.register_02b(p2, out, families=("GRU",))
        before = TR.load_tuned_recipes(out)
        args = SimpleNamespace(tuned_recipes=out, hp_registry=None, recipe_id="02B_GRU", smoke=True, peak_lr=None)
        launcher_before = L._recipe_values(args, "GRU")
        w2 = D.register_02b(p2, out, families=("GRU",), c2_only=True)
        w3 = D.register_02b(p2, out, families=("SASREC",), with_c2=True)
        gate = base / "G1B_APPROVED.json"
        parts = [{"part": part, "family": fam, "self_sha256": w["self_sha256"]}
                 for part, fam, w in ((1, "GRU", w1), ("1R", "GRU", w2), (2, "SASREC", w3))]
        gate.write_text(json.dumps(TR.g1b_obj(parts)), encoding="utf-8")
        yield base, p2, out, gate, before, launcher_before
    finally:
        p2kit.cleanup()


def test_c2_ranking_and_c1_launcher_identity(registered):
    _, p2, out, _, before, launcher_before = registered
    after = TR.load_tuned_recipes(out)
    s2 = json.loads((p2 / "P2_GRU_STAGE2.json").read_text())
    c1 = before["recipes"]["02B_GRU"]
    c2 = after["recipes"]["02B_GRU_C2"]
    assert TR.recipe(after, "02B_GRU") == TR.recipe(before, "02B_GRU")
    assert c2["source_hp_id"] != c1["source_hp_id"]
    remaining = [c for c in s2["selection"]["configs"].values()
                 if c["valid"] and c["hp_id"] != c1["source_hp_id"]]
    assert c2["source_hp_id"] == max(remaining, key=lambda c: c["mean"])["hp_id"]
    args = SimpleNamespace(tuned_recipes=out, hp_registry=None, recipe_id="02B_GRU", smoke=True, peak_lr=None)
    got = L._recipe_values(args, "GRU")
    expected = {"recipe_id": c1["recipe_id"], "recipe_sha256": c1["sha256"], "recipe_registry": str(out),
                "peak_lr": float(c1["peak_lr"]), "dropout": list(c1["dropout"]),
                "schedule": c1["schedule"], "model_variant": c1["model_variant"]}
    assert got == expected == launcher_before


def test_registry_versions_and_g1b(registered):
    _, _, out, gate, before, _ = registered
    v2 = TR.load_tuned_recipes(out.with_name(out.stem + ".v2.json"))
    v3 = TR.load_tuned_recipes(out)
    assert before["recipes"]["02B_GRU"] == v2["recipes"]["02B_GRU"] == v3["recipes"]["02B_GRU"]
    assert v2["recipes"]["02B_GRU_C2"] == v3["recipes"]["02B_GRU_C2"]
    assert set(v3["recipes"]) == {"02B_GRU", "02B_GRU_C2", "02B_SASREC", "02B_SASREC_C2"}
    g = TR.load_g1b(gate)
    assert g["format"] == TR.G1B_FORMAT and [p["part"] for p in g["tuned_recipes_parts"]] == [1, "1R", 2]
    for count in (1, 2, 3):
        assert TR.load_g1b_obj(TR.g1b_obj(g["tuned_recipes_parts"][:count]))
    with pytest.raises(TR.TunedRecipeError):
        TR.g1b_obj([g["tuned_recipes_parts"][0], g["tuned_recipes_parts"][2]], format=TR.G1B_FORMAT)
    old = TR.g1b_obj([g["tuned_recipes_parts"][0]], format=TR.G1B_FORMAT_1)
    assert TR.load_g1b_obj(old)


def test_candidate_pin_order_starts_at_v1():
    versions = [{"self_sha256": "a", "version": 1}, {"self_sha256": "b", "version": 2},
                {"self_sha256": "c", "version": 3}]
    by_pin = {v["self_sha256"]: v for v in versions}
    with pytest.raises(U.UnsealRefused, match="v1 registry"):
        U._require_strict_pins(["b", "c"], versions, by_pin)


def test_production_resume_across_part1r_format_2(tmp_path, monkeypatch):
    d, c, tp, ep, before, r1, reg2 = _interrupted_gru_across_part2(tmp_path, monkeypatch, part1r=True)
    assert TR.load_g1b()["format"] == TR.G1B_FORMAT
    resumed = trainer(d, c, tp, ep, eval_override=scripted(FLAT))
    assert resumed.config_sha256 == before["config_sha256"]
    assert resumed.tuned_recipe["recipe_sha256"] == r1["sha256"]
    assert resumed.tuned_recipe["g1b_pin"]["pinned"] == reg2["self_sha256"]
    assert resumed.resume()["status"] == "DONE_MAIN"
    assert read(d / "RUN_START.json")["config_sha256"] == before["config_sha256"]


def test_refuse_changed_existing_entry(registered, tmp_path):
    _, p2, out, _, _, _ = registered
    obj = json.loads(out.read_text())
    obj["recipes"]["02B_GRU"]["peak_lr"] = 0.123
    obj["recipes"]["02B_GRU"]["sha256"] = TR.canonical_sha256(obj["recipes"]["02B_GRU"], exclude=("sha256",))
    obj["self_sha256"] = TR.canonical_sha256(obj)
    fake = tmp_path / out.name
    fake.write_text(json.dumps(obj))
    with pytest.raises(TR.TunedRecipeError):
        TR.load_tuned_recipes(fake)


@nc("a self-hashed stage-2 C2 config that differs from the recomputed selection")
def test_nc_train05_tampered_c2_stage_config(registered, tmp_path, monkeypatch):
    _, p2, out, _, _, _ = registered
    stage = p2 / "P2_GRU_STAGE2.json"
    original_bytes = stage.read_bytes()                        # Ops fix: restore exact bytes (Windows CRLF)
    original = original_bytes.decode("utf-8")
    obj = json.loads(original)
    c2 = next(c for c in obj["selection"]["configs"].values()
              if c["hp_id"] != obj["selection"]["tuned"]["hp_id"] and c["valid"])
    c2["mean"] += 0.01
    try:
        p2kit.rehash_stage(stage, obj)
        with pytest.raises(TR.TunedRecipeError, match="configs"):
            D.register_02b(p2, tmp_path / "bad.json", families=("GRU",), with_c2=True)
        with pytest.raises(TR.TunedRecipeError, match="changed"):
            TR.load_tuned_recipes(out)
        forged = json.loads(out.read_text())
        for rid in ("02B_GRU", "02B_GRU_C2"):
            rec = forged["recipes"][rid]
            rec["stage2_file"]["file_sha256"] = TR._sha_file(stage)
            rec["stage2_file"]["self_sha256"] = json.loads(stage.read_text())["self_sha256"]
            rec["sha256"] = TR.canonical_sha256(rec, exclude=("sha256",))
        forged["self_sha256"] = TR.canonical_sha256(forged)
        fake = tmp_path / "forged_registry.json"
        fake.write_text(json.dumps(forged))
        with pytest.raises(TR.TunedRecipeError, match="configs"):
            TR.load_tuned_recipes(fake)
        monkeypatch.setattr(TR, "_canon", lambda obj: "ignored")  # MUTANT: no canonical config comparison
        raises_with(TR.TunedRecipeError, "configs", TR.load_tuned_recipes, fake)
    finally:
        stage.write_bytes(original_bytes)


def test_load_retains_stored_run_dir_suffix_tolerance(tmp_path):
    p2 = p2kit.make_p2(tmp_path / "p2", families=("GRU",))
    out = tmp_path / "TUNED_RECIPES_02B.json"
    D.register_02b(p2, out, families=("GRU",))
    stage = p2 / "P2_GRU_STAGE2.json"
    obj = json.loads(stage.read_text(encoding="utf-8"))
    obj["runs"][0]["run_dir"] = "foreign/prefix/" + obj["runs"][0]["run_dir"].replace("\\", "/")
    p2kit.rehash_stage(stage, obj)
    reg = json.loads(out.read_text())
    rec = reg["recipes"]["02B_GRU"]
    rec["stage2_file"]["file_sha256"] = TR._sha_file(stage)
    rec["stage2_file"]["self_sha256"] = json.loads(stage.read_text())["self_sha256"]
    rec["sha256"] = TR.canonical_sha256(rec, exclude=("sha256",))
    reg["self_sha256"] = TR.canonical_sha256(reg)
    out.write_text(json.dumps(reg))
    assert TR.load_tuned_recipes(out)["recipes"]["02B_GRU"] == rec


def test_one_valid_size_records_no_c2(tmp_path, monkeypatch):
    p2 = tmp_path / "p2"
    monkeypatch.setattr(TR, "V4", tmp_path)
    args = argparse.Namespace(family="GRU", stage=1, gpu=0, out_dir=p2, history_policy="USER_HISTORY_L50",
                              hp_registry=None, theta0_variants=None, dry_run=False)
    assert D.run_stage(args, runner=p2kit.runner, say=lambda *_: None) == 0

    def one_size_runner(cmd, **kw):
        result = p2kit.runner(cmd, **kw)
        hp_id = cmd[cmd.index("--hp-id") + 1]
        if "D128" in hp_id:
            rd = Path(cmd[cmd.index("--run-dir") + 1])
            summary = json.loads((rd / "RUN_SUMMARY.json").read_text())
            summary["endpoint"]["value"] = 0.01
            summary["values_at_marks"]["6.0"] = 0.01
            (rd / "RUN_SUMMARY.json").write_text(json.dumps(summary))
            rows = json.loads((rd / "RESULT_ROWS.json").read_text())
            rows["rows"][0]["value"] = 0.01
            (rd / "RESULT_ROWS.json").write_text(json.dumps(rows))
        return result

    args.stage = 2
    assert D.run_stage(args, runner=one_size_runner, say=lambda *_: None) == 3
    out = tmp_path / "TUNED_RECIPES_02B.json"
    with pytest.raises(TR.TunedRecipeError, match="requires --with-c2"):
        D.register_02b(p2, out, families=("GRU",))
    result = D.register_02b(p2, out, families=("GRU",), with_c2=True)
    assert result["no_c2"] == ["GRU"] and result["version"] == 1
    reg = TR.load_tuned_recipes(out)
    assert set(reg["recipes"]) == {"02B_GRU"} and reg["no_c2"] == ["GRU"]


def _fake_runs(base: Path, reg, pins):
    runs = {}
    for rid, r in reg["recipes"].items():
        for seed in (2026, 2027):
            fam = r["family"]
            rd = base / f"{rid}_{seed}"
            (rd / "sealed").mkdir(parents=True, exist_ok=True)
            cfg = {"family": fam, "seed": seed, "phase": "MAIN", "recipe_id": rid,
                   "test_overrides": False, "model_variant": r["model_variant"], "peak_lr": r["peak_lr"],
                   "dropout": r["dropout"], "schedule": r["schedule"]}
            sha = U._config_sha256(cfg)
            pin = pins[0] if rid == "02B_GRU" else pins[1] if rid == "02B_GRU_C2" else pins[2]
            (rd / "RUN_START.json").write_text(json.dumps({"config": cfg, "config_sha256": sha,
                "shared_numel": {"02B_GRU": 20, "02B_GRU_C2": 10, "02B_SASREC": 30,
                                  "02B_SASREC_C2": 40}[rid],
                "tuned_recipe": {"recipe_sha256": r["sha256"], "tuned_recipes_self_sha256": pin}}))
            pub = {"status": "DONE_MAIN", "still_improving": False, "sealed": True, "config_sha256": sha}
            (rd / "RUN_SUMMARY.json").write_text(json.dumps(pub))
            (rd / "STATUS.json").write_text(json.dumps({"status": "DONE_MAIN"}))
            (rd / "sealed" / "RUN_SUMMARY.json").write_text("{}")
            runs[(fam, rid, seed)] = rd
    return runs


@pytest.fixture
def contest05(registered, tmp_path):
    _, _, out, gate, _, _ = registered
    reg = TR.load_tuned_recipes(out)
    pins = [p["self_sha256"] for p in TR.load_g1b(gate)["tuned_recipes_parts"]]
    return _fake_runs(tmp_path, reg, pins), gate, out


def _unseal(c):
    runs, gate, out = c
    return U.unseal_gate(runs, gate_path=gate, tuned_path=out)


def test_eight_run_unseal(contest05):
    token = _unseal(contest05)
    assert len(token.record["configs"]) == 8
    assert all(token.covers(d) for d in contest05[0].values())
    assert token.record["parameter_counts"]["02B_GRU_C2"] == 10


@nc("a gate that accepts candidate seeds with disagreeing RUN_START shared_numel")
def test_nc_train05_parameter_count_seed_mismatch(contest05, monkeypatch):
    runs, gate, out = contest05
    rd = runs[("GRU", "02B_GRU_C2", 2027)]
    p = rd / "RUN_START.json"
    obj = json.loads(p.read_text())
    obj["shared_numel"] += 1
    p.write_text(json.dumps(obj))
    with pytest.raises(U.UnsealRefused, match="shared_numel differs"):
        U.unseal_gate(runs, gate_path=gate, tuned_path=out)
    monkeypatch.setattr(U, "_agree_parameter_counts", lambda cs: {rid: next(iter(s.values()))
                                                          for rid, s in cs.items()})  # MUTANT
    raises_with(U.UnsealRefused, "shared_numel differs", U.unseal_gate, runs,
                gate_path=gate, tuned_path=out)


def test_unseal_four_clauses(contest05, tmp_path):
    runs, gate, out = contest05
    short = dict(runs)
    short.pop(next(iter(short)))
    with pytest.raises(U.UnsealRefused, match="exactly 8"):
        U.unseal_gate(short, gate_path=gate, tuned_path=out)
    g = TR.load_g1b(gate)
    bad_gate = tmp_path / "short_gate.json"
    bad_gate.write_text(json.dumps(TR.g1b_obj(g["tuned_recipes_parts"][:2])))
    with pytest.raises(U.UnsealRefused, match="part 1R"):
        U.unseal_gate(runs, gate_path=bad_gate, tuned_path=out)
    rd = next(iter(runs.values()))
    start = json.loads((rd / "RUN_START.json").read_text())
    start["config"]["peak_lr"] = 12.0
    (rd / "RUN_START.json").write_text(json.dumps(start))
    with pytest.raises(U.UnsealRefused, match="RunConfig hash"):
        _unseal(contest05)


def test_unseal_registry_byte_identity(contest05, monkeypatch):
    runs, gate, out = contest05
    real = TR.load_tuned_recipes
    def changed(path, hp_path, **kw):
        reg = real(path, hp_path, **kw)
        if Path(path).name.endswith(".v2.json"):
            reg["recipes"]["02B_GRU"] = dict(reg["recipes"]["02B_GRU"], peak_lr=123.0)
        return reg
    monkeypatch.setattr(TR, "load_tuned_recipes", changed)
    with pytest.raises(U.UnsealRefused, match="byte-identical"):
        U.unseal_gate(runs, gate_path=gate, tuned_path=out)


@nc("a six-run registry with SASREC C2 silently omitted")
def test_nc_train05_six_runs_without_sasrec_c2(registered, tmp_path, monkeypatch):
    _, p2, _, _, _, _ = registered
    out = tmp_path / "TUNED_RECIPES_02B.json"
    w1 = D.register_02b(p2, out, families=("GRU",))
    w2 = D.register_02b(p2, out, families=("GRU",), c2_only=True)
    w3 = D.register_02b(p2, out, families=("SASREC",))
    gate = tmp_path / "G1B.json"
    gate.write_text(json.dumps(TR.g1b_obj([
        {"part": part, "family": fam, "self_sha256": w["self_sha256"]}
        for part, fam, w in ((1, "GRU", w1), ("1R", "GRU", w2), (2, "SASREC", w3))])))
    reg = TR.load_tuned_recipes(out)
    runs = _fake_runs(tmp_path / "runs", reg, [w1["self_sha256"], w2["self_sha256"], w3["self_sha256"]])
    assert len(runs) == 6
    with pytest.raises(U.UnsealRefused, match="SASREC"):
        U.unseal_gate(runs, gate_path=gate, tuned_path=out)
    monkeypatch.setattr(U, "_require_candidate_completeness", lambda current: None)  # MUTANT
    raises_with(U.UnsealRefused, "SASREC", U.unseal_gate, runs, gate_path=gate, tuned_path=out)


@nc("a repeated G1B registry version pin")
def test_nc_train05_repeated_registry_pin(contest05, tmp_path, monkeypatch):
    runs, gate, out = contest05
    g = TR.load_g1b(gate)
    parts = [dict(p) for p in g["tuned_recipes_parts"]]
    parts[1]["self_sha256"] = parts[2]["self_sha256"]
    for seed in (2026, 2027):
        p = runs[("GRU", "02B_GRU_C2", seed)] / "RUN_START.json"
        start = json.loads(p.read_text())
        start["tuned_recipe"]["tuned_recipes_self_sha256"] = parts[2]["self_sha256"]
        p.write_text(json.dumps(start))
    bad = tmp_path / "repeated.json"
    bad.write_text(json.dumps(TR.g1b_obj(parts)))
    with pytest.raises(U.UnsealRefused, match="strictly increasing"):
        U.unseal_gate(runs, gate_path=bad, tuned_path=out)
    monkeypatch.setattr(U, "_require_strict_pins", lambda *a: None)  # MUTANT
    raises_with(U.UnsealRefused, "strictly increasing", U.unseal_gate, runs, gate_path=bad, tuned_path=out)


@nc("a gate that accepts an incomplete eight-run set")
def test_nc_train05_missing_run(contest05, monkeypatch):
    monkeypatch.setattr(U, "_candidate_runs", lambda *a: ({}, {}, {}))
    short = dict(contest05[0]); short.pop(next(iter(short)))
    raises_with(U.UnsealRefused, "exactly 8", U.unseal_gate, short, gate_path=contest05[1],
                tuned_path=contest05[2])


@nc("a gate that accepts a missing G1B part")
def test_nc_train05_missing_part(contest05, monkeypatch, tmp_path):
    runs, gate, out = contest05
    valid = TR.load_g1b(gate)
    bad = tmp_path / "G1B.json"
    bad.write_text(json.dumps(TR.g1b_obj(valid["tuned_recipes_parts"][:2])))
    original = U._candidate_registries
    monkeypatch.setattr(U, "_candidate_registries", lambda gate_arg, tuned_path, hp_path:
                        original(valid, tuned_path, hp_path))
    raises_with(U.UnsealRefused, "part 1R", U.unseal_gate, runs, gate_path=bad, tuned_path=out)


@nc("a gate that skips registry-version identity")
def test_nc_train05_registry_identity(contest05, monkeypatch, tmp_path):
    runs, gate, out = contest05
    valid = TR.load_g1b(gate)
    bad = tmp_path / "G1B.json"
    obj = TR.g1b_obj(valid["tuned_recipes_parts"])
    obj["tuned_recipes_parts"][1]["self_sha256"] = "f" * 64
    bad.write_text(json.dumps(obj))
    original = U._candidate_registries
    monkeypatch.setattr(U, "_candidate_registries", lambda gate_arg, tuned_path, hp_path:
                        original(valid, tuned_path, hp_path))
    raises_with(U.UnsealRefused, "pin", U.unseal_gate, runs, gate_path=bad, tuned_path=out)


@nc("a gate that skips candidate RunConfig checks")
def test_nc_train05_config_hash(contest05, monkeypatch):
    runs, gate, out = contest05
    rd = next(iter(runs.values()))
    start = json.loads((rd / "RUN_START.json").read_text())
    start["config"]["peak_lr"] = 123.0
    (rd / "RUN_START.json").write_text(json.dumps(start))
    monkeypatch.setattr(U, "_candidate_runs", lambda *a: ({}, {}, {}))
    raises_with(U.UnsealRefused, "RunConfig hash", U.unseal_gate, runs, gate_path=gate, tuned_path=out)


def _rows(values):
    rows = []
    for rid, pair in values.items():
        fam = "SASREC" if "SASREC" in rid else "GRU"
        for seed, value in zip((2026, 2027), pair):
            rows.append({"family": fam, "recipe_id": rid, "seed": seed, "method": "C", "split": "VALIDATION",
                         "run_label": "PRODUCTION", "selection_rule": "CONTROLLED_BUDGET",
                         "metric": "mrr", "k": 20, "aggregation": "macro", "target_denominator": "E2E",
                         "efe": 6.0, "checkpoint_variant": "ENDPOINT", "value_status": "OK", "value": value,
                         "eval_manifest_id": "MAIN_VALIDATION_5_100_131072"})
    return rows


def test_g2_tie_exclusion_and_diagnostic_flip(contest05):
    recipes = {rid: {"family": "SASREC" if "SASREC" in rid else "GRU", "peak_lr": 0.001}
               for rid in ("02B_GRU", "02B_GRU_C2", "02B_SASREC", "02B_SASREC_C2")}
    token = _unseal(contest05)
    main = _rows({"02B_GRU": (0.6, 0.6), "02B_GRU_C2": (0.6003, 0.6003),
                  "02B_SASREC": (0.7, 0.7), "02B_SASREC_C2": (0.71, 0.71)})
    diag = _rows({"02B_GRU": (0.61, 0.61), "02B_GRU_C2": (0.59, 0.59),
                  "02B_SASREC": (0.7, 0.7), "02B_SASREC_C2": (0.71, 0.71)})
    picked = S.within_family_candidates(main, recipes, unseal_token=token, diagnostic_rows=diag)
    assert picked["GRU"]["selected_recipe_id"] == "02B_GRU_C2"
    assert picked["GRU"]["diagnostic_flip"] is True
    assert picked["GRU"]["labels"]["02B_GRU"] == "CANDIDATE_NOT_SELECTED"
    main[0]["value"] = float("nan")
    picked = S.within_family_candidates(main, recipes, unseal_token=token)
    assert picked["GRU"]["selected_recipe_id"] == "02B_GRU_C2"
    main[2]["value"] = float("nan")
    with pytest.raises(S.NeedsFable, match="needs Fable"):
        S.within_family_candidates(main, recipes, unseal_token=token)


def test_g2_diagnostic_error_is_reported(contest05):
    token = _unseal(contest05)
    recipes = {rid: {"family": r["family"], "peak_lr": r["peak_lr"]}
               for rid, r in TR.load_tuned_recipes(contest05[2])["recipes"].items()}
    main = _rows({rid: (0.6, 0.61) for rid in recipes})
    diagnostic = _rows({rid: (float("nan"), float("nan")) for rid in recipes})
    picked = S.within_family_candidates(main, recipes, unseal_token=token, diagnostic_rows=diagnostic)
    assert all("needs Fable" in picked[f]["diagnostic_error"] for f in ("GRU", "SASREC"))


@nc("foreign eval-manifest rows passed as MAIN primary evidence")
def test_nc_train05_foreign_primary_rows(contest05, monkeypatch):
    token = _unseal(contest05)
    recipes = {rid: {"family": r["family"], "peak_lr": r["peak_lr"]}
               for rid, r in TR.load_tuned_recipes(contest05[2])["recipes"].items()}
    rows = _rows({rid: (0.6, 0.61) for rid in recipes})
    rows[0]["eval_manifest_id"] = "DIAGNOSTIC"
    with pytest.raises(S.SelectionError, match="MAIN_VALIDATION"):
        S.within_family_candidates(rows, recipes, unseal_token=token)
    monkeypatch.setattr(S, "_require_primary_manifest", lambda rows: None)  # MUTANT
    raises_with(S.SelectionError, "MAIN_VALIDATION", S.within_family_candidates,
                rows, recipes, unseal_token=token)


def test_g2_passes_only_selected_recipes_to_t1(contest05):
    recipes = {rid: {"family": "SASREC" if "SASREC" in rid else "GRU", "peak_lr": 0.001}
               for rid in ("02B_GRU", "02B_GRU_C2", "02B_SASREC", "02B_SASREC_C2")}
    token = _unseal(contest05)
    rows = _rows({"02B_GRU": (0.6, 0.6), "02B_GRU_C2": (0.61, 0.61),
                  "02B_SASREC": (0.7, 0.7), "02B_SASREC_C2": (0.71, 0.71)})
    for row in list(rows):
        rows.append(dict(row, selection_rule="PRACTICAL_BEST"))
        rows.append(dict(row, selection_rule="PRACTICAL_BEST", metric="ndcg", k=10))
    result = S.select_g2_candidates(rows, recipes, unseal_token=token, delta_bar_ci=(-0.11, -0.09))
    assert result["within_family"]["GRU"]["selected_recipe_id"] == "02B_GRU_C2"
    assert result["within_family"]["SASREC"]["selected_recipe_id"] == "02B_SASREC_C2"
    assert result["between_families"]["status"] == "DECISIVE_STABLE"
    assert result["between_families"]["winner_family"] == "SASREC"
