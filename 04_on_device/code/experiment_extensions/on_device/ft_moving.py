"""Own validation-only FT arms around the unchanged device/FT_C numerical path.

Never admits a TEST role. The plan binds four existing moving-model best.pt
files and their historical MAIN validation ranks. Root schedules GPU work.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
from pathlib import Path

import ppsi_paths

LABEL = "POST_HOC_EXPLORATORY"
DEVICE_SHA = "1968ca9550ed9ac70d9310753aa5aa33243e7bb79cabc6e4fa7cd95ad701642c"
SELECTION_SHA = "e47965d2f4486fc2c30140a8ee14e79686a842decb091b3dff1de5646716c10c"
_ACTIVE = False


class Refused(ValueError):
    pass


def physical(path):
    """Resolve the two read-only legacy aliases to the owned physical copy."""
    p = str(path)
    base = ppsi_paths.SERVER_BASE.as_posix()
    for alias in ppsi_paths.LEGACY_ALIASES:
        if p == alias or p.startswith(alias + "/"):
            p = base + p[len(alias):]
            break
    return Path(p)


def phase_root():
    # The extension phase folder on the server (the code was run from <phase>/ppsi_v4/code).
    return physical(os.environ.get("PPSI_EXPERIMENT_EXTENSIONS", ppsi_paths.EXTENSIONS_ROOT))


def sha(path):
    h = hashlib.sha256()
    with physical(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(physical(path).read_text(encoding="utf-8"),
                      parse_constant=lambda x: (_ for _ in ()).throw(Refused(f"nonfinite JSON {x}")))


def canonical_sha(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def load_plan(path):
    d = read(path)
    if d.get("format") != "PPSI_FT_MOVING_MODELS_DESIGN_1" or d.get("label") != LABEL:
        raise Refused("not an own exploratory FT moving-model plan")
    if canonical_sha({k: v for k, v in d.items() if k != "self_sha256"}) != d.get("self_sha256"):
        raise Refused("plan self SHA differs")
    if d.get("roles_allowed") != ["MAIN"] or d.get("test_allowed") is not False:
        raise Refused("only MAIN validation is admitted")
    pairs = [(b["line"], b["seed"]) for b in d["bases"]]
    if sorted(pairs) != [(l, s) for l in ("cold", "pb100") for s in (2026, 2027)]:
        raise Refused("plan must bind exactly four own moving bases")
    proc = d["procedure"]
    if (proc["selected_cell"], proc["passes"], proc["lr_frac"], proc["selection"]["sha256"]) != (
            "FT_C_f0.3_P4", 4, 0.3, SELECTION_SHA):
        raise Refused("only unchanged fixed FT_C f0.3/P4 is admitted")
    return d


def pin_check(pin):
    if sha(pin["path"]) != pin["sha256"]:
        raise Refused(f"bound file changed: {pin['path']}")


def select_base(plan, line, seed, *, role="MAIN"):
    if role != "MAIN" or line not in ("cold", "pb100") or seed not in (2026, 2027):
        raise Refused("only the own cold/PB100 seeds on MAIN validation are admitted")
    hits = [b for b in plan["bases"] if (b["line"], b["seed"]) == (line, seed)]
    if len(hits) != 1:
        raise Refused("base not uniquely bound")
    return hits[0]


def check_base(entry):
    """Torch-free complete source-record binding, before GPU initialization."""
    for p in [entry["base"], *entry["provenance_files"],
              *[entry["validation_baseline"][k] for k in ("ranks", "result", "rows")]]:
        pin_check(p)
    q = physical(entry["base"]["path"])
    if q.name != "best.pt" or q.parent.parent.name != "ckpt":
        raise Refused("FT base must be own best.pt")
    directory = physical(entry["base_directory"])
    if q.parent.parent.parent.resolve() != directory.resolve():
        raise Refused("base outside its bound source directory")
    st, side, launch = [read(directory / f) for f in ("STATUS.json", "ARM_SPEC.json", "LAUNCH_0.json")]
    spec = side["arm_spec"]
    from v4int import launch_fl as F
    if F.spec_digest(spec) != entry["arm_spec_sha256"] or side["arm_spec_sha256"] != entry["arm_spec_sha256"]:
        raise Refused("own source arm spec does not verify")
    for rec in (st, side, launch):
        if rec.get("run_id") != entry["base_run_id"] or rec.get("arm") != entry["launch_arm"]:
            raise Refused("own source run/arm mismatch")
    if st.get("status") != "DONE_MAIN" or spec.get("role") != "MAIN" or spec["cfg"]["seed"] != entry["seed"]:
        raise Refused("not the own completed MAIN base seed")
    if launch.get("arm_spec") != spec or launch.get("arm_spec_sha256") != entry["arm_spec_sha256"]:
        raise Refused("own LAUNCH and sidecar disagree")
    if spec["cfg"].get("dp") or spec["cfg"].get("pf") or (spec.get("p2_recipe") or {}).get("personal_head"):
        raise Refused("DP/personal component bases are not admitted")
    if (spec.get("p2_recipe") or {}).get("lora_rank"):
        raise Refused("LoRA bases are not admitted")
    baseline = read(entry["validation_baseline"]["result"]["path"])
    if baseline["manifest_id"] != "MAIN_VALIDATION_5_100_131072" or baseline["model_sha256"] != entry["validation_baseline"]["model_state_digest"]:
        raise Refused("baseline is not own MAIN validation state")
    return spec, launch


def check_checkpoint(entry):
    """CPU-only payload check; export the launch CUDA environment before calling."""
    spec, launch = check_base(entry)
    from fl.checkpoint import load_checkpoint
    from fl.numerics import state_digest
    from v4int.launch_fl import fl_run_config
    q = physical(entry["base"]["path"])
    obj = load_checkpoint(q)
    if obj.get("kind") != "weights" or obj.get("run_id") != entry["base_run_id"]:
        raise Refused("checkpoint kind/run mismatch")
    expected = fl_run_config(spec, entry["base_run_id"], 1).digest()
    if obj.get("config_digest") != expected or expected != launch.get("fl_config_digest"):
        raise Refused("checkpoint config not its own source")
    efe = float(obj.get("efe", float("nan")))
    if not math.isfinite(efe) or efe < 0.5:
        raise Refused("best checkpoint is not a scored positive EFE")
    digest = state_digest(obj["theta"])
    if digest != entry["validation_baseline"]["model_state_digest"]:
        raise Refused("best theta does not match the bound unchanged baseline ranks")
    pin_check(entry["base"])
    return obj, spec


def numerical_binding():
    """Bind inherited numerical modules plus this shim; exclude other owned extensions."""
    code = physical((ppsi_paths.EXTENSIONS_ROOT / "ppsi_v4" / "code").resolve())
    files = {p.relative_to(code).as_posix(): sha(p) for p in sorted(code.rglob("*.py"))
             if "experiment_extensions" not in p.relative_to(code).parts and "__pycache__" not in p.parts}
    files["experiment_extensions/ft_moving.py"] = sha(__file__)
    return {"files": files, "sha256": canonical_sha(files)}


def output_guard(out, binding, *, resume=False, root=None):
    """Write-once identity; partial resumes keep exact code/plan/base binding."""
    root = physical(root or phase_root()).resolve()
    out = physical(out)
    if out.resolve() == root or root not in out.resolve().parents:
        raise Refused("output must be strictly inside the fresh extension")
    if any(p.is_symlink() for p in [out, *out.parents] if p == root or root in p.parents):
        raise Refused("symlink output refused")
    marker = out / "FT_BINDING.json"
    if out.exists() and any(out.iterdir()):
        if not resume or not marker.is_file() or read(marker) != binding:
            raise Refused("existing output is not an exact bound partial resume")
        if (out / "STATUS.json").is_file() and read(out / "STATUS.json").get("status") == "DONE":
            raise Refused("completed validation FT never reruns")
    elif resume:
        raise Refused("resume requires an existing own partial binding")
    else:
        out.mkdir(parents=True, exist_ok=True)
        with marker.open("x", encoding="utf-8") as f:
            json.dump(binding, f, sort_keys=True, indent=1, allow_nan=False)
            f.write("\n")
    return out


@contextlib.contextmanager
def device_context(entry, binding):
    """Metadata/base admission only. Training/evaluation functions stay unchanged."""
    global _ACTIVE
    if _ACTIVE:
        raise Refused("nested FT moving context")
    from v4int import launch_device as D
    if sha(D.__file__) != DEVICE_SHA:
        raise Refused("the wrapped launch_device changed")
    arm, base_arm = entry["new_arm"], "X2_FA_PB100K" if entry["line"] == "pb100" else "X_FA_COLD"
    original_write, original_code = D.write_json, D.device_code_sha256

    def load(adapter, path, *, base_arm, seed, want_sha256=None):
        if physical(path).resolve() != physical(entry["base"]["path"]).resolve() or seed != entry["seed"] or want_sha256 != entry["base"]["sha256"]:
            raise Refused("runtime base differs from own plan")
        obj, spec = check_checkpoint(entry)
        from fl.numerics import state_digest
        adapter.load_state_(obj["theta"])
        state = adapter.broadcast_state(clone=True)
        if state_digest(state) != state_digest(obj["theta"]):
            raise Refused("worker did not load exact own theta")
        return {"state": state, "digest": state_digest(state), "kind": base_arm, "base_kind": D.WARM_BASE_KIND,
                "run_id": entry["base_run_id"], "path": str(physical(path)), "file_sha256": want_sha256,
                "checkpoint": "best.pt", "base_arm_spec_sha256": entry["arm_spec_sha256"], "init_from": spec.get("init_from")}

    def labels(start, selection):
        return {"study": "FT_MOVING_EXTENSION_20261013", "exploratory_label": LABEL, "base_source": base_arm,
                "base_run_id": start["run_id"], "base_checkpoint_sha256": start["file_sha256"],
                "base_arm_spec_sha256": start["base_arm_spec_sha256"], "base_training_paradigm": "FEDERATED",
                "ft_selection_self_sha256": selection["self_sha256"], "extension_binding_sha256": canonical_sha(binding)}

    def write(path, obj):
        if isinstance(obj, dict):
            obj = dict(obj, extension_binding_sha256=canonical_sha(binding), exploratory_label=LABEL)
            if Path(path).name.startswith("LAUNCH_"):
                obj.update(label=LABEL, task="FT moving model validation extension", extension_binding=binding)
        return original_write(path, obj)

    patches = {"ARMS": D.ARMS + (arm,), "load_warm_base": load, "ft_warm_row_labels": labels,
               "write_json": write, "device_code_sha256": lambda: canonical_sha({"numerical": original_code(), "extension": canonical_sha(binding)})}
    for key, value in {"LOCK_CODE": "FA", "PARADIGM": "CLIENT_FINETUNE", "LIFECYCLE": "FRESH_ADAMW_PER_CLIENT",
                       "FT_PROCEDURE": "FT_C", "SELECTION_ARM": "FT_C", "FT_WARM_BASE": base_arm}.items():
        patches[key] = dict(getattr(D, key), **{arm: value})
    saved = {k: getattr(D, k) for k in patches}
    _ACTIVE = True
    try:
        for key, value in patches.items():
            setattr(D, key, value)
        yield D
    finally:
        for key, value in saved.items():
            setattr(D, key, value)
        _ACTIVE = False


def review_gate(path, plan_path, numerical):
    r = read(path)
    if r.get("status") != "PASS" or r.get("protocol_sha256") != sha(plan_path) or r.get("ft_module_sha256") != sha(__file__) or r.get("numerical_code_sha256") != numerical["sha256"]:
        raise Refused("independent review does not bind exact current FT bytes and protocol")
    return {"path": str(physical(path)), "sha256": sha(path)}


def validation_stats(manifest, ranks, label=""):
    """Full E2E denominator, including OOV-only clients; never decision macro."""
    import numpy as np
    from v4eval.metrics import credits_from_ranks
    from v4eval.bootstrap import user_stats
    ranks = np.asarray(ranks)
    if ranks.shape != (manifest.n,) or not np.issubdtype(ranks.dtype, np.integer):
        raise Refused("ranks do not cover the exact whole validation manifest")
    if np.any(ranks[manifest.rankable] < 1) or np.any(ranks[manifest.rankable] > manifest.catalogue_K):
        raise Refused("invalid full-ranking rankable row")
    if np.any(ranks[~manifest.rankable] >= 1):
        raise Refused("censored/OOV row has a positive rank")
    credit = credits_from_ranks(ranks, (20,))["mrr@20"]
    return user_stats(credit, manifest.eval_mask, manifest.rankable, manifest.eval_user_key, label)


def analyze(plan_path, outputs, out):
    """Read all four completed own validation arms and bootstrap exact paired users."""
    import numpy as np
    from v4eval import bootstrap as B
    from v4int import launch_device as D
    from v4int.providers import StoreEvalProvider
    from v4int.verify import DataPaths
    from v4base.identity import eval_client_key_hex_by_eval_user_key
    plan = load_plan(plan_path)
    numerical = numerical_binding()
    outputs, out = physical(outputs), physical(out)
    if phase_root().resolve() not in out.resolve().parents or out.exists():
        raise Refused("analysis output must be fresh inside the extension")
    code_root = physical((ppsi_paths.EXTENSIONS_ROOT / "ppsi_v4").resolve())
    paths = DataPaths.default(code_root)
    ep = StoreEvalProvider("MAIN_VALIDATION_5_100_131072", paths=paths, history_policy="USER_HISTORY_L50",
                          manifest_registry=code_root / "contracts/manifest_registry/REGISTRY_V2.json",
                          registry_root=code_root, prefetch=0)
    m = ep.manifest
    external = eval_client_key_hex_by_eval_user_key(ep, manifest_dir=paths.manifest_dir)
    points, stats = {}, {}
    for entry in plan["bases"]:
        check_base(entry)
        directory = outputs / entry["new_run_id"]
        complete = read(directory / "FT_COMPLETE.json")
        binding = complete["binding"]
        if complete.get("status") != "DONE" or read(directory / "STATUS.json").get("status") != "DONE":
            raise Refused("analysis requires all four completed validation arms")
        if read(directory / "FT_BINDING.json") != binding or binding["protocol_sha256"] != sha(plan_path) or binding["numerical_code_sha256"] != numerical["sha256"]:
            raise Refused("completed run identity/code differs from this exact analysis")
        if (binding["line"], binding["seed"], binding["base_checkpoint_sha256"], binding["run_id"]) != (entry["line"], entry["seed"], entry["base"]["sha256"], entry["new_run_id"]):
            raise Refused("completed arm is not the own paired base")
        launches = list(directory.glob("LAUNCH_*.json"))
        if not launches:
            raise Refused("completed FT has no launch provenance")
        for lp in launches:
            launch = read(lp)
            if launch["start"]["digest"] != entry["validation_baseline"]["model_state_digest"] or launch["extension_binding"] != binding:
                raise Refused("own FT did not start from the matched baseline theta")
        cell = f"{entry['new_arm']}_f0.3_P4"
        actual = read(directory / "cells" / f"{cell}.result.json")
        baseline = read(entry["validation_baseline"]["result"]["path"])
        for result in (actual, baseline):
            if result["manifest_id"] != m.manifest_id or result["hashes"] != dict(m.hashes) or result["n_rows"] != m.n:
                raise Refused("metric result is not the exact aligned full validation manifest")
        if actual["evaluator_code_sha256"] != baseline["evaluator_code_sha256"]:
            raise Refused("baseline/FT evaluator source differs")
        for key, value in baseline["eval_config"].items():
            if actual["eval_config"].get(key) != value:
                raise Refused(f"baseline/FT frozen evaluator setting differs: {key}")
        line, seed = entry["line"], entry["seed"]
        for name, rank_path, result in (("FT", directory / "cells" / f"{cell}.ranks_int32.npy", actual),
                                        ("BASE", physical(entry["validation_baseline"]["ranks"]["path"]), baseline)):
            st = validation_stats(m, np.load(rank_path, allow_pickle=False), f"{line}_{name}_s{seed}")
            point = B.point(st, "macro_e2e")
            if not math.isclose(point, result["E2E"]["macro"]["mrr@20"], abs_tol=1e-12, rel_tol=0):
                raise Refused("client-macro reconstruction disagrees with scored rows")
            if len(st.user_keys) != 40764 or st.n_decisions != 519714 or int(np.count_nonzero(st.NR == 0)) != 46:
                raise Refused("full E2E population/OOV-only user count changed")
            stats[(line, seed, name)] = st
            points[st.label] = {"MRR20": point, "NDCG10": result["E2E"]["macro"]["ndcg@10"],
                                "HR10": result["E2E"]["macro"]["hr@10"], "n_users": len(st.user_keys),
                                "n_decisions": st.n_decisions, "ranks": {"path": str(rank_path), "sha256": sha(rank_path)}}
    first = next(iter(stats.values()))
    keys = [str(external[k]) for k in first.user_keys]
    if len(set(keys)) != len(keys):
        raise Refused("external validation client keys are not one-to-one")
    sample = B.make_plan(first.user_keys, seed=20260926, n_resamples=1000, client_key_hex=keys)
    contrasts = {}
    for line in ("cold", "pb100"):
        ft = {s: stats[(line, s, "FT")] for s in (2026, 2027)}
        base = {s: stats[(line, s, "BASE")] for s in (2026, 2027)}
        delta = B.seed_contrast(sample, ft, base, "macro_e2e", line + "_FT-BASE")
        rf = np.mean([B.resampled(sample, ft[s], "macro_e2e") for s in (2026, 2027)], axis=0)
        rb = np.mean([B.resampled(sample, base[s], "macro_e2e") for s in (2026, 2027)], axis=0)
        if np.any(rb <= 0):
            raise Refused("retention has zero baseline in a paired bootstrap draw")
        point_f = float(np.mean([B.point(ft[s], "macro_e2e") for s in (2026, 2027)]))
        point_b = float(np.mean([B.point(base[s], "macro_e2e") for s in (2026, 2027)]))
        contrasts[line] = {"delta": delta, "retention": {"point": 100 * point_f / point_b,
                            "ci95": B.ci95(100 * rf / rb), "rule": "ratio of seed-mean client-macro values per draw"}}
    result = {"format": "PPSI_FT_MOVING_VALIDATION_ANALYSIS_1", "label": LABEL, "role": "MAIN_VALIDATION",
              "points": points, "contrasts": contrasts, "binding": {"protocol_sha256": sha(plan_path),
              "numerical_code_sha256": numerical["sha256"], "manifest_hashes": dict(m.hashes),
              "user_set_sha256": sample.user_set_sha256, "W_sha256": sample.W_sha256}, "CI_scope": plan["analysis"]["CI_scope"],
              "limitations": plan["limitations"]}
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("x", encoding="utf-8") as f:
        json.dump(result, f, indent=1, sort_keys=True, allow_nan=False)
        f.write("\n")
    return result


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plan", type=Path, default=phase_root() / "planning/FT_MOVING_MODELS_DESIGN.json")
    p.add_argument("--line", choices=("cold", "pb100"))
    p.add_argument("--seed", type=int, choices=(2026, 2027))
    p.add_argument("--gpu", type=int)
    p.add_argument("--out", type=Path)
    p.add_argument("--review", type=Path)
    p.add_argument("--role", default="MAIN")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--shards")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--analyze-outputs", type=Path)
    return p


def main(argv=None):
    a = parser().parse_args(argv)
    try:
        if a.analyze_outputs:
            if a.out is None or a.role != "MAIN":
                raise Refused("fresh MAIN validation analysis output required")
            analyze(a.plan, a.analyze_outputs, a.out)
            print("FT VALIDATION ANALYSIS COMPLETE")
            return 0
        if a.line is None or a.seed is None or a.gpu is None or a.gpu < 0 or a.out is None:
            raise Refused("line/seed/physical GPU/fresh output required")
        # check_base imports launch_fl/v4train, which capture the launch env.
        # Require the physical device export before that first numerical import.
        if not a.dry_run and os.environ.get("CUDA_VISIBLE_DEVICES") != str(a.gpu):
            raise Refused("export CUDA_VISIBLE_DEVICES=<--gpu> before Python starts")
        plan = load_plan(a.plan)
        entry = select_base(plan, a.line, a.seed, role=a.role)
        for pin in plan["source_pins"]:
            pin_check(pin)
        check_base(entry)
        numerical = numerical_binding()
        review = None
        if not a.dry_run:
            if a.review is None:
                raise Refused("exact independent review required before real training")
            review = review_gate(a.review, a.plan, numerical)
        binding = {"format": "PPSI_FT_MOVING_BINDING_1", "label": LABEL, "role": "MAIN", "line": a.line,
                   "seed": a.seed, "run_id": entry["new_run_id"], "protocol_sha256": sha(a.plan),
                   "base_checkpoint_sha256": entry["base"]["sha256"], "base_arm_spec_sha256": entry["arm_spec_sha256"],
                   "numerical_code_sha256": numerical["sha256"], "ft_module_sha256": sha(__file__), "review": review,
                   "n_shards": 64, "history_policy": "USER_HISTORY_L50"}
        with device_context(entry, binding) as D:
            av = ["--arm", entry["new_arm"], "--seed", str(a.seed), "--gpu", str(a.gpu), "--role", "MAIN",
                  "--run-id", entry["new_run_id"], "--run-dir", str(physical(a.out)), "--base", str(physical(entry["base"]["path"])),
                  "--base-kind", D.WARM_BASE_KIND, "--base-sha256", entry["base"]["sha256"],
                  "--history-policy", "USER_HISTORY_L50", "--n-shards", "64", "--n-threads", "4"]
            if a.resume:
                av.append("--resume")
            if a.shards:
                av += ["--shards", a.shards]
            if a.dry_run:
                av.append("--dry-run")
            args = D.parse_args(av)
            pre = D.preflight(args)
            from v4int.launch_central import export_environment
            env = export_environment(a.gpu)
            # Full payload/base-rank provenance check remains CPU-only.
            obj, _ = check_checkpoint(entry)
            del obj
            if a.dry_run:
                print(json.dumps({"status": "PASS_DRY_RUN", "binding": binding, "preflight": pre}, indent=1))
                return 0
            output_guard(a.out, binding, resume=a.resume)
            pre["ft_moving"] = binding
            rc = D.run(args, pre, env)
            if rc == 0 and read(physical(a.out) / "STATUS.json").get("status") == "DONE":
                D.write_json(physical(a.out) / "FT_COMPLETE.json", {"status": "DONE", "binding": binding,
                    "baseline": entry["validation_baseline"], "source_pins": plan["source_pins"], "numerical_binding": numerical})
            return rc
    except Refused as e:
        print(f"FT MOVING REFUSED: {e}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
