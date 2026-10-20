"""v4train.tuned_recipes — the P3 tuned-recipes registry (amendment 02b; TRAIN-02R1 item F6, TRAIN-02R2 F1 / F2 / (a)).

After P2, `python -B code/v4train/p2_driver.py register-02b --p2-dir <P2 dir> [--family GRU|SASREC ...]` registers the
tuned recipe of each named family into contracts/hp_registry/TUNED_RECIPES_02B.json: recipe_id 02B_GRU / 02B_SASREC,
holding the tuned model variant, peak LR, dropout and schedule (02R1 §3), the source HP trial (hp_id and its sha256),
the 2-seed 6.0 mean, the CKPT_AVG carry flag and the sha256 (file and self) of BOTH stage files; each recipe carries
the sha256 of its canonical JSON.
TRAIN-02R2 F1 — nothing in a stage file is trusted on its self-hash alone. For each family, registration (and every
later load, which re-reads the stage files at their recorded paths and checks their file sha256):
  * verifies P2_<F>_STAGE1.json and P2_<F>_STAGE2.json (self-hash, family, stage, the same HP registry), requires
    stage2.stage1_file.self_sha256 == stage 1's self-hash;
  * recomputes stage 1's plan (p2_driver.stage1_plan over stage 1's values) and requires it to equal the recorded plan;
  * recomputes the stage-2 selection (p2_driver.stage2_select(trials, stage1.stage2_plan, stage1.values,
    stage2.values, levels)) and requires its status OK and its tuned config and ckpt_avg_carried_to_p4 to equal the
    recorded ones; the tuned values must also equal the named HP trial.
(a) One family at a time: a family already in the registry is never replaced (refused). The first registration creates
the file with O_EXCL; a later one writes the new version (all earlier recipes byte-identical, `previous_self_sha256`
set) atomically, after archiving the previous version as TUNED_RECIPES_02B.v<n>.json with O_EXCL (never overwritten).
TRAIN-02R2 F2 — a NON-TEST run may use only the production registry: `check_production_registry` requires the path to
be contracts/hp_registry/TUNED_RECIPES_02B.json and its self-hash to equal `tuned_recipes_self_sha256` of
company/gates/G1B_APPROVED.json (written by Ops at G1b; a missing file or field refuses). Test runs
(test_overrides) may use any verified registry.
A MAIN-phase RunConfig may carry `recipe_id` (+ optional `recipe_registry`): validate() accepts a model variant, peak
LR, dropout and schedule that differ from the registered recipe ONLY when all four equal that tuned recipe. The TUNING
/ HP_SEARCH rules are unchanged; an HP-trial id is never a recipe id.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
from typing import Iterable, Mapping, Optional

from .hp_registry import FAMILIES, REGISTRY_PATH as HP_REGISTRY_PATH, VARIANT_LEVELS, HPRegistryError, \
    canonical_sha256, load_hp_registry, trial

V4 = Path(__file__).resolve().parents[2]
REGISTRY_PATH = V4 / "contracts" / "hp_registry" / "TUNED_RECIPES_02B.json"
G1B_GATE = V4 / "company" / "gates" / "G1B_APPROVED.json"
FORMAT = "PPSI_V4_TUNED_RECIPES_02B_1"
RECIPE_FIELDS = ("model_variant", "peak_lr", "dropout", "schedule")


class TunedRecipeError(ValueError):
    pass


def recipe_id_of(family: str) -> str:
    return f"02B_{family}"


def c2_id_of(family: str) -> str:
    return f"02B_{family}_C2"


def _sha_file(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _stage_obj(path) -> dict:
    from .p2_driver import P2Error, load_stage
    try:
        return load_stage(Path(path))
    except P2Error as e:
        raise TunedRecipeError(str(e)) from None


def verify_stages(fam: str, s1: Mapping, s2: Mapping, hp: Mapping) -> dict:
    """F1: the stage-2 selection recomputed from the two verified stage files; returns the tuned config and the
    carry flag (raises TunedRecipeError on any disagreement)."""
    from .p2_driver import stage1_plan, stage2_select
    for s, k in ((s1, 1), (s2, 2)):
        if s.get("family") != fam or int(s.get("stage", 0)) != k:
            raise TunedRecipeError(f"{fam}: not the stage-{k} file of {fam}")
        if (s.get("hp_registry") or {}).get("self_sha256") != hp["self_sha256"]:
            raise TunedRecipeError(f"{fam}: stage {k} was produced under another HP registry")
    if (s2.get("stage1_file") or {}).get("self_sha256") != s1.get("self_sha256"):
        raise TunedRecipeError(f"{fam}: the stage-2 file does not name this stage-1 file (stage1_file.self_sha256)")
    trials, levels = hp["families"][fam], VARIANT_LEVELS[fam]
    plan = s1["selection"]["stage2_plan"]
    p1 = stage1_plan(trials, s1["values"], levels)
    if p1["stage2_plan"] != plan or p1["status"] != "OK":
        raise TunedRecipeError(f"{fam}: the recorded stage-2 plan differs from the one recomputed from stage 1")
    sel = stage2_select(trials, plan, s1["values"], s2["values"], levels)
    rec = s2.get("selection") or {}
    valid_sizes = sum(c.get("valid") is True for c in (sel.get("configs") or {}).values())
    if (sel["status"] not in ("OK", "FABLE_TRIGGER") or rec.get("status") != sel["status"]
            or valid_sizes < 1 or "tuned" not in sel):
        raise TunedRecipeError(f"{fam}: stage 2 ended in {sel['status']} / recorded {rec.get('status')} "
                               "with no valid tuned size or mismatched status")
    if json.dumps(sel.get("tuned"), sort_keys=True) != json.dumps(rec.get("tuned"), sort_keys=True):
        raise TunedRecipeError(f"{fam}: the recorded tuned config {rec.get('tuned')} differs from the recomputed "
                               f"{sel.get('tuned')}")
    if bool(sel.get("ckpt_avg_carried_to_p4")) != bool(rec.get("ckpt_avg_carried_to_p4")):
        raise TunedRecipeError(f"{fam}: the recorded CKPT_AVG carry differs from the recomputed one")
    tuned = sel["tuned"]
    t = trial(hp, tuned["hp_id"])
    if t["family"] != fam or any(t[k] != tuned[k] for k in RECIPE_FIELDS):
        raise TunedRecipeError(f"{fam}: the tuned values differ from HP trial {tuned['hp_id']}")
    if _canon(sel.get("configs")) != _canon(rec.get("configs")):
        raise TunedRecipeError(f"{fam}: the recorded stage-2 configs differ from the recomputed selection")
    return {"tuned": tuned, "trial": t, "carry": bool(sel["ckpt_avg_carried_to_p4"]),
            "selection": sel}


# ------------------------------------------------------------------------------------------------ TRAIN-02R3
def v4_rel(p) -> str:
    """R3 item 2: a path stored in 02b is relative to the V4 root (resolved first); P2 output must live under V4."""
    r = Path(p).resolve()
    try:
        return r.relative_to(V4.resolve()).as_posix()
    except ValueError:
        raise TunedRecipeError(f"{p} is not under the V4 root {V4}: the P2 output directory must live under it so "
                               "02b can store V4-relative paths") from None


def v4_abs(rel: str) -> Path:
    if Path(rel).is_absolute() or ".." in Path(rel).parts:
        raise TunedRecipeError(f"{rel!r} is not a V4-relative path")
    return V4 / rel


def _canon(obj) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=True, separators=(",", ":"), allow_nan=True)


def _endpoint_row_check(rows_obj: Mapping, summary: Mapping, hp_id: str, seed: int, where: str) -> None:
    rows = [r for r in (rows_obj.get("rows") or [])
            if r.get("metric") == "mrr" and int(r.get("k", -1)) == 20 and r.get("aggregation") == "macro"
            and r.get("target_denominator") == "E2E" and r.get("selection_rule") == "CONTROLLED_BUDGET"]
    if len(rows) != 1:
        raise TunedRecipeError(f"{where}: RESULT_ROWS.json must hold exactly one CONTROLLED_BUDGET macro MRR@20 E2E "
                               f"row, found {len(rows)}")
    r = rows[0]
    end = (summary.get("endpoint") or {}).get("value")
    if r.get("value") != end or abs(float(r.get("efe") or 0.0) - 6.0) > 1e-9 or r.get("hp_id") != hp_id \
            or int(r.get("seed", -1)) != int(seed) or r.get("run_label") != "HP_SEARCH":
        raise TunedRecipeError(f"{where}: the endpoint row (value {r.get('value')}, efe {r.get('efe')}, hp_id "
                               f"{r.get('hp_id')}, seed {r.get('seed')}, label {r.get('run_label')}) does not match the "
                               f"RUN_SUMMARY endpoint {end} of {hp_id} seed {seed}")


def _suffix_match(recorded_run_dir: str, rel: str) -> bool:
    rp = [x for x in str(recorded_run_dir).replace("\\", "/").split("/") if x]
    rr = [x for x in rel.split("/") if x]
    return len(rp) >= len(rr) and rp[-len(rr):] == rr


def run_evidence(stage: Mapping, stored: Optional[Mapping] = None) -> dict:
    """R3 item 1: every run of a stage file is bound to its evidence. For each `runs` entry (written by the R5-pack
    driver: hp_id, seed, run_dir, exit_code, ...) the RUN_SUMMARY.json of its run dir is re-read, the run values are
    recomputed with p2_driver.run_values(summary, exit_code) and must equal the stage file's `values[hp_id]`; for a
    DONE_TUNING run the endpoint row of RESULT_ROWS.json must carry the same endpoint value. Registration (stored None)
    derives each V4-relative run dir from the recorded run_dir; a load uses the stored relative dirs (the recorded
    run_dir must end with them) and requires the RUN_SUMMARY / RESULT_ROWS sha256 recorded at registration."""
    from .p2_driver import run_values
    runs = stage.get("runs") or []
    values = stage.get("values") or {}
    if sorted(r["hp_id"] for r in runs) != sorted(values):
        raise TunedRecipeError(f"stage {stage.get('stage')}: the runs and the recorded values name different trials")
    out = {}
    for run in runs:
        h, seed = run["hp_id"], int(run["seed"])
        where = f"{stage.get('family')} stage {stage.get('stage')} {h} seed {seed}"
        if stored is None:
            rel = v4_rel(run["run_dir"])
        else:
            if h not in stored:
                raise TunedRecipeError(f"{where}: no evidence recorded at registration")
            rel = stored[h]["run_dir"]
            if not _suffix_match(run["run_dir"], rel):
                raise TunedRecipeError(f"{where}: the recorded run dir {run['run_dir']} does not end with {rel}")
        rd = v4_abs(rel)
        sp, rp = rd / "RUN_SUMMARY.json", rd / "RESULT_ROWS.json"
        summary = json.loads(sp.read_text(encoding="utf-8")) if sp.is_file() else None
        got = run_values(summary, run.get("exit_code"))
        if _canon(got) != _canon(values[h]):
            raise TunedRecipeError(f"{where}: the recorded run values differ from the ones recomputed from "
                                   f"{sp} (RUN_SUMMARY)")
        rows_sha = None
        if summary is not None and summary.get("status") == "DONE_TUNING":
            if not rp.is_file():
                raise TunedRecipeError(f"{where}: {rp} (the endpoint rows) is missing")
            _endpoint_row_check(json.loads(rp.read_text(encoding="utf-8")), summary, h, seed, where)
            rows_sha = _sha_file(rp)
        ev = {"run_dir": rel, "summary_sha256": _sha_file(sp) if summary is not None else None,
              "result_rows_sha256": rows_sha}
        if stored is not None and {k: stored[h].get(k) for k in ev} != ev:
            raise TunedRecipeError(f"{where}: the RUN_SUMMARY / RESULT_ROWS changed since registration")
        out[h] = ev
    return out


def _c2_config(selection: Mapping) -> Optional[Mapping]:
    """Second valid size by CALIB mean, with the stage-2 deterministic tie order."""
    first = selection["tuned"]["hp_id"]
    configs = [c for c in selection["configs"].values() if c.get("valid") is True and c["hp_id"] != first]
    configs.sort(key=lambda c: (-float(c["mean"]), int(c["n_params"]), float(c["peak_lr"]), c["hp_id"]))
    return configs[0] if configs else None


def family_recipe(fam: str, stage1_path, stage2_path, hp: Mapping, *, candidate: int = 1,
                  stored_evidence: Optional[Mapping] = None) -> Optional[dict]:
    s1, s2 = _stage_obj(stage1_path), _stage_obj(stage2_path)
    v = verify_stages(fam, s1, s2, hp)
    ev1 = run_evidence(s1, (stored_evidence or {}).get("stage1") if stored_evidence is not None else None)
    ev2 = run_evidence(s2, (stored_evidence or {}).get("stage2") if stored_evidence is not None else None)
    if candidate not in (1, 2):
        raise TunedRecipeError(f"candidate must be 1 or 2, got {candidate}")
    chosen = v["tuned"] if candidate == 1 else _c2_config(v["selection"])
    if chosen is None:
        return None
    t = v["trial"] if candidate == 1 else trial(hp, chosen["hp_id"])
    r = {"recipe_id": recipe_id_of(fam) if candidate == 1 else c2_id_of(fam), "family": fam,
         **{k: copy.deepcopy(t[k]) for k in RECIPE_FIELDS},
         "source_hp_id": t["hp_id"], "source_hp_sha256": t["sha256"], "mean_6.0_two_seeds":
         chosen["mean_6.0"] if candidate == 1 else chosen["mean"],
         "ckpt_avg_carried_to_p4": v["carry"] if candidate == 1 else bool(chosen["ckpt_avg_rule"]["carry"]),
         "stage1_file": {"path": v4_rel(stage1_path), "file_sha256": _sha_file(stage1_path),
                         "self_sha256": s1["self_sha256"]},
         "stage2_file": {"path": v4_rel(stage2_path), "file_sha256": _sha_file(stage2_path),
                         "self_sha256": s2["self_sha256"]},
         "run_evidence": {"stage1": ev1, "stage2": ev2},
         "paths": "V4-relative (TRAIN-02R3)"}
    r["sha256"] = canonical_sha256(r, exclude=("sha256",))
    return r


def _write_atomic_new_version(path: Path, text: str) -> str:
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    data = text.encode("utf-8")
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    return hashlib.sha256(data).hexdigest()


def write_registry(p2_dir, families: Iterable[str] = FAMILIES, path=REGISTRY_PATH, hp_path=HP_REGISTRY_PATH,
                   *, with_c2: bool = False, c2_only: bool = False) -> dict:
    """Register the tuned recipe of `families` (one or both). A registered family is never replaced (a)."""
    from v4models.variants import write_exclusive
    hp = load_hp_registry(hp_path)
    path = Path(path)
    fams = [str(f).upper() for f in families]
    if not fams or any(f not in FAMILIES for f in fams) or len(set(fams)) != len(fams):
        raise TunedRecipeError(f"families must be distinct values of {FAMILIES}, got {fams}")
    prev = load_tuned_recipes(path, hp_path) if path.exists() else None
    recipes = dict(prev["recipes"]) if prev else {}
    no_c2 = list(prev.get("no_c2", [])) if prev else []
    for fam in fams:
        if c2_only and recipe_id_of(fam) not in recipes:
            raise TunedRecipeError(f"{recipe_id_of(fam)} must be registered before C2")
        if not c2_only and recipe_id_of(fam) in recipes:
            raise TunedRecipeError(f"{recipe_id_of(fam)} is already registered in {path}; a recipe is never replaced")
        s1, s2 = Path(p2_dir) / f"P2_{fam}_STAGE1.json", Path(p2_dir) / f"P2_{fam}_STAGE2.json"
        stage2 = _stage_obj(s2)
        if stage2.get("selection", {}).get("status") == "FABLE_TRIGGER" and not (with_c2 or c2_only):
            raise TunedRecipeError(f"{fam}: FABLE_TRIGGER requires --with-c2 or --c2-only to record no_c2")
        if not c2_only:
            recipes[recipe_id_of(fam)] = family_recipe(fam, s1, s2, hp)
        if with_c2 or c2_only:
            if c2_id_of(fam) in recipes:
                raise TunedRecipeError(f"{c2_id_of(fam)} is already registered in {path}")
            if c2_only and family_recipe(fam, s1, s2, hp) != recipes[recipe_id_of(fam)]:
                raise TunedRecipeError(f"{recipe_id_of(fam)} changed since its registration")
            c2 = family_recipe(fam, s1, s2, hp, candidate=2)
            if c2 is not None:
                recipes[c2_id_of(fam)] = c2
            elif fam not in no_c2:
                no_c2.append(fam)
    obj = {"format": FORMAT, "task": "TRAIN-02R1 F6 / TRAIN-02R2 (amendment 02b registry)",
           "hp_registry": {"path": hp["path"], "file_sha256": hp["file_sha256"], "self_sha256": hp["self_sha256"]},
           "recipes": dict(sorted(recipes.items())), "previous_self_sha256": prev["self_sha256"] if prev else None,
           "version": (prev["version"] + 1) if prev else 1,
           "rule": "a MAIN run may differ from the registered recipe only as one of these recipes (recipe_id); a "
                   "non-test run needs this file's self-hash pinned in company/gates/G1B_APPROVED.json"}
    if no_c2:
        obj["no_c2"] = sorted(no_c2)
    obj["self_sha256"] = canonical_sha256(obj)
    text = json.dumps(obj, indent=1, sort_keys=True, ensure_ascii=True, allow_nan=False) + "\n"
    archived = None
    if prev is None:
        fsha = write_exclusive(path, text)
    else:
        archived = path.with_name(f"{path.stem}.v{prev['version']}.json")
        write_exclusive(archived, path.read_text(encoding="utf-8"))            # O_EXCL: an archive is never rewritten
        fsha = _write_atomic_new_version(path, text)
    return {"path": str(path), "file_sha256": fsha, "self_sha256": obj["self_sha256"], "version": obj["version"],
            "archived_previous": str(archived) if archived else None,
            "no_c2": sorted(no_c2),
            "recipes": {k: {f: v[f] for f in ("source_hp_id", "sha256")} for k, v in obj["recipes"].items()}}


def load_tuned_recipes(path=REGISTRY_PATH, hp_path=HP_REGISTRY_PATH, *, reverify_stages: bool = True) -> dict:
    p = Path(path)
    if not p.is_file():
        raise TunedRecipeError(f"tuned-recipes registry {p} does not exist (written after P2 by register-02b)")
    data = p.read_bytes()
    try:
        obj = json.loads(data.decode("utf-8"),
                         parse_constant=lambda c: (_ for _ in ()).throw(ValueError(f"non-strict constant {c}")))
    except ValueError as e:
        raise TunedRecipeError(f"{p} is not strict JSON ({e})") from None
    if not isinstance(obj, dict) or obj.get("format") != FORMAT:
        raise TunedRecipeError(f"{p} is not a {FORMAT} object")
    if obj.get("self_sha256") != canonical_sha256(obj):
        raise TunedRecipeError(f"{p}: self-hash mismatch")
    try:
        hp = load_hp_registry(hp_path)
    except HPRegistryError as e:
        raise TunedRecipeError(f"HP registry refused: {e}") from None
    if (obj.get("hp_registry") or {}).get("self_sha256") != hp["self_sha256"]:
        raise TunedRecipeError(f"{p} was registered under another HP registry")
    for rid, r in (obj.get("recipes") or {}).items():
        if r.get("sha256") != canonical_sha256(r, exclude=("sha256",)) or r.get("recipe_id") != rid:
            raise TunedRecipeError(f"{p}: recipe {rid} sha256 / id mismatch")
        if r.get("family") not in FAMILIES or rid not in (recipe_id_of(r["family"]), c2_id_of(r["family"])):
            raise TunedRecipeError(f"{p}: invalid recipe id / family {rid}")
        try:
            t = trial(hp, r["source_hp_id"])
        except HPRegistryError as e:
            raise TunedRecipeError(f"{p}: recipe {rid}: {e}") from None
        if t["sha256"] != r["source_hp_sha256"] or any(t[k] != r[k] for k in RECIPE_FIELDS):
            raise TunedRecipeError(f"{p}: recipe {rid} differs from its source HP trial {r['source_hp_id']}")
        if reverify_stages:                                                    # F1 / R3: stage files and run evidence
            stages = {}
            for k in ("stage1_file", "stage2_file"):
                sf = r.get(k) or {}
                ap = v4_abs(sf["path"]) if sf.get("path") else None
                if ap is None or not ap.is_file() or _sha_file(ap) != sf.get("file_sha256"):
                    raise TunedRecipeError(f"{p}: recipe {rid}: {k} {sf.get('path')} is missing or changed")
                stages[k] = _stage_obj(ap)
            v = verify_stages(r["family"], stages["stage1_file"], stages["stage2_file"], hp)
            ev = r.get("run_evidence") or {}
            run_evidence(stages["stage1_file"], ev.get("stage1") or {})
            run_evidence(stages["stage2_file"], ev.get("stage2") or {})
            expected = family_recipe(r["family"], v4_abs(r["stage1_file"]["path"]),
                                     v4_abs(r["stage2_file"]["path"]), hp,
                                     candidate=2 if rid == c2_id_of(r["family"]) else 1,
                                     stored_evidence=ev)
            if expected != r:
                raise TunedRecipeError(f"{p}: recipe {rid} differs from its re-verified stage files")
    no_c2 = obj.get("no_c2", [])
    if not isinstance(no_c2, list) or len(no_c2) != len(set(no_c2)) or any(f not in FAMILIES for f in no_c2):
        raise TunedRecipeError(f"{p}: invalid no_c2 record")
    if any(recipe_id_of(f) not in obj["recipes"] or c2_id_of(f) in obj["recipes"] for f in no_c2):
        raise TunedRecipeError(f"{p}: no_c2 conflicts with registered recipes")
    if reverify_stages:
        for fam in no_c2:
            c1 = obj["recipes"][recipe_id_of(fam)]
            if family_recipe(fam, v4_abs(c1["stage1_file"]["path"]),
                             v4_abs(c1["stage2_file"]["path"]), hp, candidate=2,
                             stored_evidence=c1.get("run_evidence")) is not None:
                raise TunedRecipeError(f"{p}: no_c2 conflicts with the re-verified stage-2 ranking")
    return {"path": str(p), "file_sha256": hashlib.sha256(data).hexdigest(), "self_sha256": obj["self_sha256"],
            "recipes": obj["recipes"], "version": int(obj.get("version", 1)), "no_c2": no_c2}


# ------------------------------------------------------------------------------------------------ TRAIN-03: G1B format
G1B_FORMAT = "PPSI_V4_G1B_APPROVED_2"
G1B_FORMAT_1 = "PPSI_V4_G1B_APPROVED_1"
RULING_PATH = "company/reviews/STAGGERED_P3_FABLE_RULING.md"
PARTS_2 = ((1, "GRU"), ("1R", "GRU"), (2, "SASREC"))
PARTS_1 = ((1, "GRU"), (2, "SASREC"))


def _hex64(v) -> bool:
    return isinstance(v, str) and len(v) == 64 and all(c in "0123456789abcdef" for c in v)


def g1b_obj(parts: Iterable[Mapping], *, ruling_path: str = RULING_PATH, ruling_sha256: Optional[str] = None,
            note: str = "", format: Optional[str] = None) -> dict:
    """The company/gates/G1B_APPROVED.json object Ops writes at G1b (part 1, then again at part 2):
      {"format": "PPSI_V4_G1B_APPROVED_1", "gate": "G1b",
       "tuned_recipes_self_sha256": <the CURRENT TUNED_RECIPES_02B.json self-hash = the last part's>,
       "tuned_recipes_parts": [{"part": 1, "family": "GRU", "self_sha256": ...},
                               {"part": 2, "family": "SASREC", "self_sha256": ...}],   # part 2 once registered
       "ruling": {"path": "company/reviews/STAGGERED_P3_FABLE_RULING.md", "sha256": <file sha256>}, "note": ...}"""
    parts = [dict(p) for p in parts]
    if format is None:
        format = G1B_FORMAT_1 if len(parts) == 2 and parts[1].get("part") == 2 else G1B_FORMAT
    if ruling_sha256 is None:
        ruling_sha256 = _sha_file(V4 / ruling_path)
    obj = {"format": format, "gate": "G1b", "tuned_recipes_self_sha256": parts[-1]["self_sha256"] if parts else None,
           "tuned_recipes_parts": parts, "ruling": {"path": ruling_path, "sha256": ruling_sha256}, "note": note}
    load_g1b_obj(obj)
    return obj


def load_g1b_obj(gate: Mapping, where: str = "G1B_APPROVED") -> dict:
    """Validate the G1B format (see g1b_obj); raises TunedRecipeError."""
    if not isinstance(gate, Mapping) or gate.get("format") not in (G1B_FORMAT, G1B_FORMAT_1):
        raise TunedRecipeError(f"{where} is not a {G1B_FORMAT_1} object or a {G1B_FORMAT} object")
    expected = PARTS_2 if gate["format"] == G1B_FORMAT else PARTS_1
    cur, parts = gate.get("tuned_recipes_self_sha256"), gate.get("tuned_recipes_parts")
    if not _hex64(cur):
        raise TunedRecipeError(f"{where}: tuned_recipes_self_sha256 must be a sha256")
    if not isinstance(parts, list) or not parts or len(parts) > len(expected):
        raise TunedRecipeError(f"{where}: tuned_recipes_parts must be a valid prefix")
    for i, p in enumerate(parts):
        part, family = expected[i]
        if not isinstance(p, Mapping) or p.get("part") != part or p.get("family") != family \
                or not _hex64(p.get("self_sha256")):
            raise TunedRecipeError(f"{where}: part {part} must be {{part: {part}, family: {family}, self_sha256}}")
    if parts[-1]["self_sha256"] != cur:
        raise TunedRecipeError(f"{where}: tuned_recipes_self_sha256 must equal the last part's self_sha256")
    ruling = gate.get("ruling") or {}
    if not isinstance(ruling, Mapping) or not ruling.get("path") or not _hex64(ruling.get("sha256")):
        raise TunedRecipeError(f"{where}: ruling {{path, sha256}} is required")
    rp = V4 / str(ruling["path"])
    if not rp.is_file():                                     # fail closed (delta review TRAIN-03 finding 1)
        raise TunedRecipeError(f"{where}: the ruling file {rp} is missing")
    if _sha_file(rp) != ruling["sha256"]:
        raise TunedRecipeError(f"{where}: the ruling file {rp} does not hash to the pinned sha256")
    return dict(gate)


def load_g1b(gate_path: Optional[Path] = None) -> dict:
    g = Path(gate_path or G1B_GATE)
    if not g.is_file():
        raise TunedRecipeError(f"{g} does not exist: the tuned recipes are not approved (G1b)")
    try:
        gate = json.loads(g.read_text(encoding="utf-8"),
                          parse_constant=lambda c: (_ for _ in ()).throw(ValueError(f"non-strict constant {c}")))
    except ValueError as e:
        raise TunedRecipeError(f"{g} is not strict JSON ({e})") from None
    return dict(load_g1b_obj(gate, str(g)), _path=str(g), _sha256=_sha_file(g))


def check_production_registry(path, reg: Mapping, gate_path: Optional[Path] = None) -> dict:
    """F2 (TRAIN-02R2) with the TRAIN-03 G1B format: a non-test run may use only
    contracts/hp_registry/TUNED_RECIPES_02B.json whose self-hash is the CURRENT pin (tuned_recipes_self_sha256);
    the unseal gate (v4train.unseal) additionally requires both parts."""
    if Path(path).resolve() != Path(REGISTRY_PATH).resolve():
        raise TunedRecipeError(f"a non-test run must use the production tuned-recipes registry {REGISTRY_PATH}, not "
                               f"{path}")
    gate = load_g1b(gate_path)
    pinned = gate["tuned_recipes_self_sha256"]
    if pinned != reg["self_sha256"]:
        raise TunedRecipeError(f"the tuned-recipes self-hash {reg['self_sha256']} is not the one pinned at G1b "
                               f"({pinned})")
    return {"gate": gate["_path"], "gate_sha256": gate["_sha256"], "pinned": pinned,
            "parts": [p["part"] for p in gate["tuned_recipes_parts"]]}


def recipe(reg: Mapping, recipe_id: str) -> dict:
    r = (reg.get("recipes") or {}).get(recipe_id)
    if r is None:
        raise TunedRecipeError(f"recipe_id {recipe_id!r} is not in the tuned-recipes registry {reg.get('path')} "
                               f"(registered: {sorted(reg.get('recipes') or {})})")
    return copy.deepcopy(r)


def recipe_record(reg: Mapping, r: Mapping) -> dict:
    return {"recipe_id": r["recipe_id"], "recipe_sha256": r["sha256"], "tuned_recipes_path": reg["path"],
            "tuned_recipes_file_sha256": reg["file_sha256"], "tuned_recipes_self_sha256": reg["self_sha256"],
            "source_hp_id": r["source_hp_id"]}


__all__ = ["REGISTRY_PATH", "G1B_GATE", "FORMAT", "TunedRecipeError", "recipe_id_of", "verify_stages",
           "family_recipe", "write_registry", "load_tuned_recipes", "check_production_registry", "recipe",
           "recipe_record"]
