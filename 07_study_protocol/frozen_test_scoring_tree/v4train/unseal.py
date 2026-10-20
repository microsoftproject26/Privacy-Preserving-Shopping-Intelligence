"""v4train.unseal — the G2 unseal gate of the P3 contest (TRAIN-03; Fable staggered-P3 ruling, condition 2).

The contest runs (MAIN phase, RunConfig.sealed_outputs) keep every metric value under <run_dir>/sealed/. The G2
selection script's FIRST read of any sealed value goes through `unseal_gate(...)`, which reads only UNSEALED files
(RUN_START.json, the public RUN_SUMMARY.json, STATUS.json) and the gate / registry files, and refuses unless:
  clause 1 (complete)  exactly the four contest runs (GRU, SASREC) x (2026, 2027) are given and each is complete: status
                       DONE_CONTINUATION, or DONE_MAIN with STILL_IMPROVING false (an automatic continuation pending or
                       running is not complete); STATUS.json agrees; the sealed summary exists;
  clause 2 (G1b)       company/gates/G1B_APPROVED.json (v4train.tuned_recipes format) holds BOTH parts — part 1 (GRU)
                       and part 2 (SASREC) self-hashes — and its current pin is part 2;
  clause 3 (GRU)       the current TUNED_RECIPES_02B.json has self-hash = part 2, the archived TUNED_RECIPES_02B.v1.json
                       has self-hash = part 1 (both fully re-verified), and the 02B_GRU entry is byte-identical
                       (canonical JSON) in the two versions;
  clause 4 (configs)   for each run, RUN_START's config_sha256 equals the sha256 recomputed from its recorded config (the
                       RunConfig.sha256 rule), the public summary names the same config_sha256, the config is a non-test
                       run of that family and seed with recipe_id 02B_<FAMILY> whose model variant, peak LR, dropout and
                       schedule equal the pinned recipe, the recorded recipe_sha256 is the pinned recipe's and the recorded
                       registry self-hash is a pinned part (GRU: part 1 or 2; SASREC: part 2).
It returns an UnsealToken; `read_sealed_json(token, run_dir, name)` / `practical_best_ranks` (bootstrap_ci) accept a
sealed read only with a token that covers the run. `smoke_token(run_dirs)` covers ONLY test (SMOKE_NOT_EVIDENCE) runs.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Optional

from . import tuned_recipes as TRM
from .checkpoint import read_json

FAMILIES = ("GRU", "SASREC")
SEEDS = (2026, 2027)
_ISSUER = object()                     # only this module mints tokens


class UnsealRefused(RuntimeError):
    pass


@dataclass(frozen=True)
class UnsealToken:
    run_dirs: frozenset
    record: Mapping = field(default_factory=dict, compare=False)
    issuer: object = field(default=None, repr=False, compare=False)

    def covers(self, run_dir) -> bool:
        return self.issuer is _ISSUER and str(Path(run_dir).resolve()) in self.run_dirs


def is_sealed(run_dir) -> bool:
    return (Path(run_dir) / "sealed").is_dir()


def _config_sha256(config: Mapping) -> str:
    """RunConfig.sha256 over the recorded as_json() (json.dumps sort_keys, allow_nan False; tuples are lists)."""
    return hashlib.sha256(json.dumps(config, sort_keys=True, allow_nan=False, default=list).encode("utf-8")).hexdigest()


def _canon(obj) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=True, separators=(",", ":"), allow_nan=False)


def clause_runs_complete(run_dirs: Mapping) -> dict:
    want = {(f, s) for f in FAMILIES for s in SEEDS}
    got = {(str(f).upper(), int(s)) for f, s in run_dirs}
    if got != want or len(run_dirs) != 4:
        raise UnsealRefused(f"the unseal gate needs exactly the four contest runs {sorted(want)}, got {sorted(got)}")
    out = {}
    for (fam, seed), rd in run_dirs.items():
        rd = Path(rd)
        pub = rd / "RUN_SUMMARY.json"
        if not pub.is_file() or not is_sealed(rd) or not (rd / "sealed" / "RUN_SUMMARY.json").is_file():
            raise UnsealRefused(f"{fam}/{seed}: {rd} is not a finished sealed contest run")
        s, st = read_json(pub), read_json(rd / "STATUS.json")
        done = s.get("status") == "DONE_CONTINUATION" or (s.get("status") == "DONE_MAIN" and s.get("still_improving")
                                                            is False)
        if not done or st.get("status") != s.get("status") or s.get("sealed") is not True:
            raise UnsealRefused(f"{fam}/{seed}: the run is not complete (status {s.get('status')}, STATUS "
                                f"{st.get('status')}, still_improving {s.get('still_improving')})")
        out[f"{fam}/{seed}"] = {"status": s["status"], "config_sha256": s.get("config_sha256")}
    return out


def clause_g1b_parts(gate_path: Optional[Path]) -> dict:
    gate = TRM.load_g1b(gate_path)
    parts = gate["tuned_recipes_parts"]
    if len(parts) != 2:
        raise UnsealRefused(f"G1B_APPROVED holds {len(parts)} part(s); unsealing needs part 1 (GRU) AND part 2 (SASREC)")
    return gate


def _two_parts(gate: Mapping) -> tuple:
    parts = gate.get("tuned_recipes_parts") or []
    if len(parts) != 2:
        raise UnsealRefused("the G1b pin lacks part 2: both part self-hashes are needed")
    return parts[0]["self_sha256"], parts[1]["self_sha256"]


def clause_gru_identical(gate: Mapping, tuned_path: Path, hp_path) -> dict:
    p1, p2 = _two_parts(gate)
    cur = TRM.load_tuned_recipes(tuned_path, hp_path)
    arch_path = Path(tuned_path).with_name(f"{Path(tuned_path).stem}.v1.json")
    v1 = TRM.load_tuned_recipes(arch_path, hp_path)
    if cur["self_sha256"] != p2 or v1["self_sha256"] != p1:
        raise UnsealRefused(f"the registry versions (v1 {v1['self_sha256']}, current {cur['self_sha256']}) are not "
                            f"the pinned parts (part 1 {p1}, part 2 {p2})")
    g1, g2 = v1["recipes"].get("02B_GRU"), cur["recipes"].get("02B_GRU")
    if g1 is None or g2 is None or _canon(g1) != _canon(g2):
        raise UnsealRefused("the 02B_GRU entry is not byte-identical in the part-1 (v1) and part-2 registries")
    if "02B_SASREC" not in cur["recipes"]:
        raise UnsealRefused("the part-2 registry holds no 02B_SASREC recipe")
    return cur


def clause_config_hashes(run_dirs: Mapping, reg: Mapping, gate: Mapping) -> dict:
    p1, p2 = _two_parts(gate)
    out = {}
    for (fam, seed), rd in run_dirs.items():
        fam, seed = str(fam).upper(), int(seed)
        start, pub = read_json(Path(rd) / "RUN_START.json"), read_json(Path(rd) / "RUN_SUMMARY.json")
        cfg = start.get("config") or {}
        if _config_sha256(cfg) != start.get("config_sha256") or pub.get("config_sha256") != start.get("config_sha256"):
            raise UnsealRefused(f"{fam}/{seed}: the RunConfig hash does not match its recorded config")
        r = reg["recipes"].get(TRM.recipe_id_of(fam))
        if r is None:
            raise UnsealRefused(f"{fam}: no pinned recipe")
        if cfg.get("test_overrides") is not False or cfg.get("family") != fam or int(cfg.get("seed", -1)) != seed \
                or cfg.get("phase") != "MAIN" or cfg.get("recipe_id") != r["recipe_id"]:
            raise UnsealRefused(f"{fam}/{seed}: not a non-test MAIN contest run of recipe {r['recipe_id']}")
        got = {"peak_lr": cfg.get("peak_lr"), "model_variant": cfg.get("model_variant"),
               "dropout": list(cfg.get("dropout") or []), "schedule": cfg.get("schedule")}
        if got != {k: r[k] for k in got}:
            raise UnsealRefused(f"{fam}/{seed}: the RunConfig values {got} are not the pinned recipe's")
        tr = start.get("tuned_recipe") or {}
        allowed = {p1, p2} if fam == "GRU" else {p2}
        if tr.get("recipe_sha256") != r["sha256"] or tr.get("tuned_recipes_self_sha256") not in allowed:
            raise UnsealRefused(f"{fam}/{seed}: the run's recorded recipe / registry is not the pinned one")
        out[f"{fam}/{seed}"] = {"config_sha256": start["config_sha256"], "recipe_sha256": r["sha256"],
                                "registry_self_sha256": tr.get("tuned_recipes_self_sha256")}
    return out


def unseal_gate(run_dirs: Mapping, *, gate_path: Optional[Path] = None, tuned_path: Optional[Path] = None,
                hp_path: Optional[Path] = None) -> UnsealToken:
    """The G2 script calls this before its first sealed read; see the module docstring for the four clauses."""
    from .hp_registry import REGISTRY_PATH as HP_PATH
    tuned_path = Path(tuned_path or TRM.REGISTRY_PATH)
    hp_path = hp_path or HP_PATH
    try:
        gate = TRM.load_g1b(gate_path)
        if gate["format"] == TRM.G1B_FORMAT:
            return _unseal_candidates(run_dirs, gate, tuned_path, hp_path)
        runs = clause_runs_complete(run_dirs)
        gate = clause_g1b_parts(gate_path)
        reg = clause_gru_identical(gate, tuned_path, hp_path)
        cfgs = clause_config_hashes(run_dirs, reg, gate)
    except TRM.TunedRecipeError as e:
        raise UnsealRefused(str(e)) from None
    rec = {"runs": runs, "configs": cfgs, "g1b": {"path": gate["_path"], "sha256": gate["_sha256"]},
           "tuned_recipes_self_sha256": reg["self_sha256"]}
    return UnsealToken(frozenset(str(Path(d).resolve()) for d in run_dirs.values()), rec, _ISSUER)


def _candidate_registries(gate: Mapping, tuned_path: Path, hp_path) -> tuple:
    """Load every version, match all three G1B pins, and compare every persistent entry."""
    parts = gate["tuned_recipes_parts"]
    if [p["part"] for p in parts] != [1, "1R", 2]:
        raise UnsealRefused("G1B_APPROVED needs part 1 (GRU) AND part 2 (SASREC), with part 1R between them")
    current = TRM.load_tuned_recipes(tuned_path, hp_path)
    versions = []
    previous = None
    for n in range(1, current["version"] + 1):
        p = (Path(tuned_path).with_name(f"{Path(tuned_path).stem}.v{n}.json")
             if n < current["version"] else Path(tuned_path))
        loaded = TRM.load_tuned_recipes(p, hp_path)
        raw = read_json(p)
        if loaded["version"] != n or raw.get("previous_self_sha256") != previous:
            raise UnsealRefused(f"registry version {n} breaks the append-only hash chain")
        versions.append(loaded)
        previous = loaded["self_sha256"]
    pins = [p["self_sha256"] for p in parts]
    if current["self_sha256"] != pins[-1]:
        raise UnsealRefused("the current registry does not match the part-2 pin")
    by_pin = {v["self_sha256"]: v for v in versions}
    if len(by_pin) != len(versions) or any(pin not in by_pin for pin in pins):
        raise UnsealRefused("each G1B part must pin a loaded registry version")
    _require_strict_pins(pins, versions, by_pin)
    seen = {}
    for v in versions:
        if not set(seen).issubset(v["recipes"]):
            raise UnsealRefused("a recipe entry was removed from a later registry version")
        for rid, entry in v["recipes"].items():
            encoded = _canon(entry)
            if rid in seen and seen[rid] != encoded:
                raise UnsealRefused(f"{rid} is not byte-identical across registry versions")
            seen[rid] = encoded
    required = [(1, "02B_GRU"), (2, "02B_SASREC")]
    if "02B_GRU_C2" in current["recipes"]:
        required.append(("1R", "02B_GRU_C2"))
    _require_candidate_completeness(current)
    for part, expected in required:
        v = by_pin[parts[[1, "1R", 2].index(part)]["self_sha256"]]
        if expected not in v["recipes"]:
            raise UnsealRefused(f"part {part} lacks {expected}")
    return current, by_pin, pins


def _require_strict_pins(pins: list, versions: list, by_pin: Mapping) -> None:
    if pins[0] != versions[0]["self_sha256"]:
        raise UnsealRefused("part-1 pin must equal the v1 registry self-hash")
    pin_versions = [by_pin[p]["version"] for p in pins]
    if any(a >= b for a, b in zip(pin_versions, pin_versions[1:])):
        raise UnsealRefused("G1B registry versions must be strictly increasing")


def _require_candidate_completeness(current: Mapping) -> None:
    for fam in FAMILIES:
        if (TRM.c2_id_of(fam) in current["recipes"]) == (fam in current["no_c2"]):
            raise UnsealRefused(f"{fam}: exactly one of C2 recipe or no_c2 record is required")


def _candidate_runs(run_dirs: Mapping, reg: Mapping, by_pin: Mapping, pins: list) -> tuple:
    want = {(fam, rid, seed) for rid, r in reg["recipes"].items()
            for fam in (r["family"],) for seed in SEEDS}
    if not {"02B_GRU", "02B_SASREC"}.issubset(reg["recipes"]):
        raise UnsealRefused("the registry lacks a C1 recipe for each family")
    got = {(str(f).upper(), rid, int(s)) for f, rid, s in run_dirs}
    if got != want or len(run_dirs) != len(want):
        raise UnsealRefused(f"the unseal gate needs exactly {len(want)} candidate runs {sorted(want)}, got {sorted(got)}")
    runs, configs, counts_by_recipe = {}, {}, {}
    for (fam, rid, seed), rd in run_dirs.items():
        fam, seed, rd = str(fam).upper(), int(seed), Path(rd)
        pub_path = rd / "RUN_SUMMARY.json"
        if not pub_path.is_file() or not (rd / "sealed" / "RUN_SUMMARY.json").is_file():
            raise UnsealRefused(f"{fam}/{rid}/{seed}: the finished sealed summary is missing")
        pub, status = read_json(pub_path), read_json(rd / "STATUS.json")
        done = pub.get("status") == "DONE_CONTINUATION" or (pub.get("status") == "DONE_MAIN"
                and pub.get("still_improving") is False)
        if not done or status.get("status") != pub.get("status") or pub.get("sealed") is not True:
            raise UnsealRefused(f"{fam}/{rid}/{seed}: run is not complete and sealed")
        start = read_json(rd / "RUN_START.json")
        count = start.get("shared_numel")
        if type(count) is not int or count <= 0:
            raise UnsealRefused(f"{fam}/{rid}/{seed}: RUN_START shared_numel must be a positive integer")
        counts_by_recipe.setdefault(rid, {})[seed] = count
        cfg = start.get("config") or {}
        if _config_sha256(cfg) != start.get("config_sha256") or pub.get("config_sha256") != start.get("config_sha256"):
            raise UnsealRefused(f"{fam}/{rid}/{seed}: RunConfig hash mismatch")
        r = reg["recipes"][rid]
        if cfg.get("test_overrides") is not False or cfg.get("family") != fam or cfg.get("seed") != seed \
                or cfg.get("phase") != "MAIN" or cfg.get("recipe_id") != rid:
            raise UnsealRefused(f"{fam}/{rid}/{seed}: wrong non-test MAIN recipe / seed")
        values = {"model_variant": cfg.get("model_variant"), "peak_lr": cfg.get("peak_lr"),
                  "dropout": list(cfg.get("dropout") or []), "schedule": cfg.get("schedule")}
        if values != {k: r[k] for k in values}:
            raise UnsealRefused(f"{fam}/{rid}/{seed}: RunConfig differs from its pinned recipe")
        tr = start.get("tuned_recipe") or {}
        pin = tr.get("tuned_recipes_self_sha256")
        if tr.get("recipe_sha256") != r["sha256"] or pin not in by_pin or pin not in pins \
                or rid not in by_pin[pin]["recipes"]:
            raise UnsealRefused(f"{fam}/{rid}/{seed}: run registry pin did not already contain its recipe")
        key = f"{fam}/{rid}/{seed}"
        runs[key] = {"status": pub["status"], "config_sha256": start["config_sha256"]}
        configs[key] = {"config_sha256": start["config_sha256"], "recipe_sha256": r["sha256"],
                        "registry_self_sha256": pin}
    return runs, configs, _agree_parameter_counts(counts_by_recipe)


def _agree_parameter_counts(counts_by_recipe: Mapping) -> dict:
    counts = {}
    for rid, seeds in counts_by_recipe.items():
        if set(seeds) != set(SEEDS) or seeds[SEEDS[0]] != seeds[SEEDS[1]]:
            raise UnsealRefused(f"{rid}: RUN_START shared_numel differs across seeds")
        counts[rid] = seeds[SEEDS[0]]
    return counts


def _unseal_candidates(run_dirs: Mapping, gate: Mapping, tuned_path: Path, hp_path) -> UnsealToken:
    reg, by_pin, pins = _candidate_registries(gate, tuned_path, hp_path)
    runs, configs, counts = _candidate_runs(run_dirs, reg, by_pin, pins)
    rec = {"runs": runs, "configs": configs, "g1b": {"path": gate["_path"], "sha256": gate["_sha256"]},
           "tuned_recipes_self_sha256": reg["self_sha256"], "parameter_counts": counts}
    return UnsealToken(frozenset(str(Path(d).resolve()) for d in run_dirs.values()), rec, _ISSUER)


def smoke_token(run_dirs) -> UnsealToken:
    """A token for TEST runs only (SMOKE_NOT_EVIDENCE; e.g. synthetic end-to-end tests): refuses any non-test run."""
    dirs = list(run_dirs.values()) if isinstance(run_dirs, Mapping) else list(run_dirs)
    for d in dirs:
        cfg = read_json(Path(d) / "RUN_START.json").get("config") or {}
        if cfg.get("test_overrides") is not True:
            raise UnsealRefused(f"{d}: a smoke token covers test runs only")
    return UnsealToken(frozenset(str(Path(d).resolve()) for d in dirs), {"smoke": True}, _ISSUER)


def sealed_root(run_dir, token: Optional[UnsealToken]) -> Path:
    """The directory a reader may read metric artifacts from: run_dir for an unsealed run; <run_dir>/sealed only with
    a token that covers the run."""
    rd = Path(run_dir)
    if not is_sealed(rd):
        return rd
    if token is None or not isinstance(token, UnsealToken) or not token.covers(rd):
        raise UnsealRefused(f"{rd} is sealed: pass the UnsealToken of the G2 unseal gate (v4train.unseal)")
    return rd / "sealed"


def read_sealed_json(token: UnsealToken, run_dir, name: str):
    return read_json(sealed_root(run_dir, token) / name)


__all__ = ["UnsealRefused", "UnsealToken", "unseal_gate", "smoke_token", "sealed_root", "read_sealed_json",
           "is_sealed", "clause_runs_complete", "clause_g1b_parts", "clause_gru_identical", "clause_config_hashes"]
