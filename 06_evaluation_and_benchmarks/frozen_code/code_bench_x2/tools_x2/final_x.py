"""final_x — the GATED exploratory TEST look of the fixed-recipe POST_HOC_X benchmark winners (task T1 / backlog C1).
POST_HOC_EXPLORATORY. Prepared 2026-10-09; NOT RUN. code_bench_x refuses `lock` / `final` on purpose and stays unchanged:
this tool scores with that tree's UNCHANGED v4bench (final.score_run, data._load_holdout_view, evaluate.summarize) and
adds only the gate, the output location and the analysis.

  python -B tools_x2/final_x.py build-lock [--final]       candidate (draft) lock, or the write-once lock (all runs DONE)
  python -B tools_x2/final_x.py check --dataset <ds>       the gate only (dry run): prints PASS or the refusal
  python -B tools_x2/final_x.py score --dataset <ds>       gate -> preflight -> holdout -> TEST_LOOK -> results
  python -B tools_x2/final_x.py analyze                    after every dataset is scored: X_RESULT.json

Gate (all before the holdout file is opened; a refusal writes nothing):
  1. company/gates/EXP_BENCH_X_TEST_LOCK.json exists, format / self_sha256 verify, draft is false;
  2. the TEST_EXPOSURE_REGISTER chain verifies AND holds a BIND entry for this lock: access_purpose LOCKED_EVALUATION,
     observations NONE, all_methods_lock_sha256 = sha256 of the lock FILE, source = the lock's path;
     no METRICS entry for this lock exists yet (that would be the LOOK: the look was taken);
  3. runs_bench_x/TEST/<ds>/TEST_LOOK.json does not exist (one look per dataset);
  4. this file's sha256 and the scoring tree (code_bench_x X_SHA256SUMS, v4bench code digest) equal the lock's;
  5. every locked run is DONE_MAIN and its RESULT.json / checkpoint sha256 equal the lock's;
  6. (Fable T1, DL 561) the confirmatory LOCK.json / data.py identity still match, and right before the holdout is
     opened the data view's fingerprint must equal the confirmatory LOCK.json data_fingerprint.
Pairing (Fable T1): every bootstrap pairs users by EXTERNAL USER ID. This side saves its TEST view's user ids
(TEST_USERS.int64.npy); the confirmatory rank files carry no user list, so their ids are those of the confirmatory view,
which the lock proves identical: the confirmatory tree (found by its LOCK code_sha256) has a data.py byte-identical to
code_bench_x's, the fingerprint is equal, and the user / cold / unknown counts are equal. Different id sets are refused.
Wave 2 (B14.5): r8_claim is null (ours-vs-ours, no claim rule). FT runs (kind ft) are refused in a lock.
Outputs only under runs_bench_x/TEST/<ds>/ (and runs_bench_x/TEST/ANALYSIS_EXP_BENCH_X/): TEST_ATTEMPTS.jsonl (start /
end / error, after the gate), TEST_LOOK.json (atomic write-once, after every checkpoint is scored in memory),
<key>/TEST_<rule>.json + ranks, TEST_RESULTS.json. Confirmatory files are only READ (their sha256 is bound in the lock).
Analysis (DL 551-553; amendment 07 R8 / B14.5): NDCG@10, ALL holdout users, PRACTICAL_BEST (CONTROLLED_BUDGET reported);
paired user bootstrap 1,000 resamples seed 20260926, per seed then the seed mean (the same resampled users for every seed);
R8 claim rule: |seed-mean delta| > 0.005 AND every seed on the same side, else "on par"; retention = 100 x seed-mean X /
seed-mean C_FULL (Track A, all its seeds). Wave 2 (amazon2023) is ours-vs-ours; ML-1M's confirmatory side is the GPU class,
this side CPU (disclosed). Every number is POST_HOC_EXPLORATORY, reported beside the confirmatory TEST, never instead of it.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
V4 = os.path.dirname(os.path.dirname(HERE))                    # .../ppsi_v4
X_TREE = os.path.join(V4, "code_bench_x")                       # the training tree of every locked run (unchanged)
os.environ["CUDA_VISIBLE_DEVICES"] = ""                         # CPU only, before any torch import
sys.path.insert(0, X_TREE)

RX = os.path.join(V4, "runs_bench_x")
LOCK_REL = "company/gates/EXP_BENCH_X_TEST_LOCK.json"
CAND_REL = "company/gates/EXP_BENCH_X_TEST_LOCK.candidate.json"
REG_REL = "company/registers/TEST_EXPOSURE_REGISTER.json"
TEST_ROOT = os.path.join(RX, "TEST")
ANALYSIS_DIR = os.path.join(TEST_ROOT, "ANALYSIS_EXP_BENCH_X")
FORMAT = "PPSI_V4_X_BENCH_TEST_LOCK_1"
DATASETS = ("s3_beauty", "s3_sports", "s3_toys", "amazon2023_video_games", "amazon2023_baby_products", "ml_1m")
WAVE2 = ("amazon2023_video_games", "amazon2023_baby_products")
SEEDS = (2026, 2027)
RULES = ("PRACTICAL_BEST", "CONTROLLED_BUDGET")
BOOT_N, BOOT_SEED, R8_TOL = 1000, 20260926, 0.005
CONF_ARMS = ("C_FULL", "FA", "FA_WARM", "PRE")
CONTRASTS = [  # (id, kind, X, Y): delta = X - Y (paired bootstrap), ratio = 100 X / Y (seed means, descriptive)
    ("R_FA_X/C_FULL", "ratio", "FA_X", "C_FULL"), ("R_FA_WARM_X/C_FULL", "ratio", "FA_WARM_X", "C_FULL"),
    ("R_C_SMALL_X/C_FULL", "ratio", "C_SMALL_X", "C_FULL"), ("R_FA_X/C_SMALL_X", "ratio", "FA_X", "C_SMALL_X"),
    ("R_FA_WARM_X/C_SMALL_X", "ratio", "FA_WARM_X", "C_SMALL_X"),
    ("D_FA_X-FA", "delta", "FA_X", "FA"), ("D_FA_WARM_X-FA_WARM", "delta", "FA_WARM_X", "FA_WARM"),
    ("D_FA_X-PRE", "delta", "FA_X", "PRE"), ("D_FA_WARM_X-PRE", "delta", "FA_WARM_X", "PRE"),
    ("D_FA_WARM_X-FA_X", "delta", "FA_WARM_X", "FA_X"),
    ("D_FA_X-C_SMALL_X", "delta", "FA_X", "C_SMALL_X"), ("D_C_SMALL_X-C_FULL", "delta", "C_SMALL_X", "C_FULL"),
    ("D_FA_X-C_FULL", "delta", "FA_X", "C_FULL"), ("D_FA_WARM_X-C_FULL", "delta", "FA_WARM_X", "C_FULL"),
]


class Refused(RuntimeError):
    pass


def sha256_file(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def canon_sha256(o) -> str:
    return hashlib.sha256(json.dumps(o, sort_keys=True, separators=(",", ":"), default=str,
                                     allow_nan=False).encode()).hexdigest()


def rj(p):
    return json.load(open(p, encoding="utf-8"))


def utc():
    import time
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def x_root(ds):
    return os.path.join(RX, "ml1m") if ds == "ml_1m" else RX


def conf_root(ds):
    return os.path.join(V4, "runs_bench_gpu" if ds == "ml_1m" else "runs_bench")


# ------------------------------------------------------------------------------------------------ scoring tree identity
def x_tree_identity() -> dict:
    sums = os.path.join(X_TREE, "X_SHA256SUMS")
    bad = []
    for line in open(sums, encoding="utf-8"):
        if line.strip():
            want, rel = line.split()
            if sha256_file(os.path.join(X_TREE, rel)) != want:
                bad.append(rel)
    if bad:
        raise Refused(f"code_bench_x differs from its X_SHA256SUMS: {bad}")
    from v4bench import common
    if os.path.realpath(str(common.CODE)) != os.path.realpath(X_TREE):
        raise Refused(f"v4bench was imported from {common.CODE}, not {X_TREE}")
    return {"tree": "code_bench_x", "x_sha256sums_sha256": sha256_file(sums),
            "bench_code_sha256": common.bench_code_sha256(),
            "final_py_sha256": sha256_file(os.path.join(X_TREE, "v4bench", "final.py")),
            "final_x_sha256": sha256_file(os.path.abspath(__file__))}


# ------------------------------------------------------------------------------------------------ lock
def run_record(run_dir) -> dict:
    res_p = os.path.join(run_dir, "RESULT.json")
    if not os.path.isfile(res_p):
        return {"run_dir": run_dir, "status": "PENDING"}
    res = rj(res_p)
    if res.get("status") != "DONE_MAIN":
        return {"run_dir": run_dir, "status": f"PENDING ({res.get('status')})"}
    cks = {}
    for rule, c in res["checkpoints"].items():
        p = c["file"] if os.path.isabs(c["file"]) else os.path.join(run_dir, c["file"])
        got = sha256_file(p)
        if got != c["sha256"]:
            raise Refused(f"{run_dir}: checkpoint {rule} sha256 differs from RESULT.json")
        cks[rule] = {"file": c["file"], "sha256": got, "mark": c["mark"]}
    seed = int(res["spec"]["seed"] if "seed" in res["spec"] else res["spec"]["cfg"]["seed"])
    if res.get("spec", {}).get("exploratory") != "POST_HOC_X":
        raise Refused(f"{run_dir}: not a POST_HOC_X run")
    if res.get("kind") not in ("fl", "central"):                 # Fable T1: no FT (per-user fine-tune) arm here
        raise Refused(f"{run_dir}: kind {res.get('kind')!r} is not allowed in this lock (fl / central only)")
    rec = {"run_dir": run_dir, "status": "DONE_MAIN", "arm": res["arm"], "kind": res["kind"], "seed": seed,
           "result_sha256": sha256_file(res_p), "checkpoints": cks,
           "device_class": res.get("device_class", "CPU-FP32")}
    ini = res.get("init_from")
    if ini:                                                      # warm arm: the PRE provenance of THIS seed
        pdir = ini["pre_run_dir"]
        pres = rj(os.path.join(pdir, "RESULT.json"))
        pck = pres["checkpoints"][ini["pre_ckpt_rule"]]
        pfile = pck["file"] if os.path.isabs(pck["file"]) else os.path.join(pdir, pck["file"])
        if sha256_file(pfile) != ini["checkpoint_sha256"]:
            raise Refused(f"{run_dir}: the PRE checkpoint {pfile} differs from init_from")
        if int(pres["spec"]["seed"]) != seed:
            raise Refused(f"{run_dir}: PRE seed {pres['spec']['seed']} != run seed {seed}")
        rec["pre_provenance"] = {"pre_run_dir": pdir, "pre_result_sha256": sha256_file(os.path.join(pdir, "RESULT.json")),
                                 "pre_ckpt_rule": ini["pre_ckpt_rule"], "pre_mark": ini.get("pre_mark"),
                                 "pre_budget_efe": ini.get("pre_budget_efe"), "checkpoint_sha256": ini["checkpoint_sha256"],
                                 "state_digest": ini["state_digest"], "pre_device_class": ini.get("pre_device_class"),
                                 "pre_seed": int(pres["spec"]["seed"])}
    return rec


def tree_bench_code(tree: str) -> str:
    import glob
    rows = [(os.path.relpath(p, tree).replace(os.sep, "/"), sha256_file(p))
            for p in sorted(glob.glob(os.path.join(tree, "v4bench", "*.py")))]
    return canon_sha256(rows)


def confirmatory_view_identity(conf_lock: dict) -> dict:
    """The confirmatory tree = the code_bench* tree whose v4bench digest equals the confirmatory LOCK's code_sha256; its
    data.py must be byte-identical to code_bench_x's (then its TEST view's user order = this view's)."""
    import glob
    want = conf_lock["code_sha256"]
    hits = [t for t in sorted(glob.glob(os.path.join(V4, "code_bench*")))
            if os.path.isdir(os.path.join(t, "v4bench")) and tree_bench_code(t) == want]
    if len(hits) != 1:
        raise Refused(f"confirmatory tree for code_sha256 {want[:12]}: {len(hits)} matches")
    a = sha256_file(os.path.join(hits[0], "v4bench", "data.py"))
    b = sha256_file(os.path.join(X_TREE, "v4bench", "data.py"))
    if a != b:
        raise Refused(f"{hits[0]} data.py {a[:12]} != code_bench_x data.py {b[:12]}: user ids cannot be proven")
    return {"conf_tree": hits[0], "conf_code_sha256": want, "data_py_sha256": a,
            "user_id_rule": "confirmatory TEST view users = this TEST view's users (same data.py, same fingerprint, "
                            "same counts); paired by external user id"}


def build_lock(final: bool) -> dict:
    ident = x_tree_identity()
    dsets, pending = {}, []
    for ds in DATASETS:
        sp = os.path.join(RX, "lean", f"SELECT_{ds}.json")
        sel = rj(sp)
        runs = {}
        for arm, key in (("FA", "FA_X"), ("FA_WARM", "FA_WARM_X")):
            a = sel["arms"][arm]
            lab = a["final"]
            seeds = sorted(int(s) for s, v in (a.get("seeds") or {}).items() if v is not None)
            if not set(SEEDS) <= set(seeds):
                raise Refused(f"{ds} {arm}: the selected cell lacks seed 2026 / 2027")
            for s in seeds:                # the seed's PRE may live in runs_bench (label __pre...) or runs_bench_x (__prex...)
                cands = [lab, lab.replace("__preB", "__prexB"), lab.replace("__prexB", "__preB")]
                hit = [c for c in dict.fromkeys(cands) if os.path.isdir(os.path.join(x_root(ds), ds, f"{c}_s{s}"))]
                if len(hit) != 1:
                    raise Refused(f"{ds} {arm} s{s}: {len(hit)} run dirs match the selected cell {lab}")
                runs[f"{key}_s{s}"] = dict(run_record(os.path.join(x_root(ds), ds, f"{hit[0]}_s{s}")), key=key,
                                           label=hit[0])
        for s in SEEDS:
            lab = "C_SMALL__B24_L50_D0.1"
            runs[f"C_SMALL_X_s{s}"] = dict(run_record(os.path.join(x_root(ds), ds, f"{lab}_s{s}")), key="C_SMALL_X",
                                           label=lab)
        pending += [f"{ds}/{k}" for k, r in runs.items() if r["status"] != "DONE_MAIN"]
        cr = os.path.join(conf_root(ds), ds)
        conf = {"root": cr, "test_results_sha256": sha256_file(os.path.join(cr, "TEST_RESULTS.json")),
                "test_look_sha256": sha256_file(os.path.join(cr, "TEST_LOOK.json")),
                "lock_sha256": sha256_file(os.path.join(cr, "LOCK.json")), "ranks": {}}
        clock = rj(os.path.join(cr, "LOCK.json"))
        conf["data_fingerprint"] = clock["data_fingerprint"]
        conf["view_identity"] = confirmatory_view_identity(clock)
        tr = rj(os.path.join(cr, "TEST_RESULTS.json"))          # run NAMES and counts only (values are not read here)
        conf["n_test_users"] = int(tr["n_test_users"])
        conf["n_cold_holdout_items_as_misses"] = int(tr.get("n_cold_holdout_items_as_misses", 0))
        conf["n_holdout_rows_unknown_user"] = int(tr.get("n_holdout_rows_unknown_user", 0))
        conf["device_class"] = tr.get("device_class", "CPU-FP32")
        for name in sorted(tr["runs"]):
            arm, _, sd = name.rpartition("_s")
            if arm in CONF_ARMS and sd.isdigit():
                for rule in RULES:
                    p = os.path.join(cr, name, "test", f"test_{rule}.ranks_int32.npy")
                    if os.path.isfile(p):
                        conf["ranks"][f"{arm}_s{sd}|{rule}"] = {"path": p, "sha256": sha256_file(p)}
        dsets[ds] = {"runs_root": x_root(ds), "select_file": sp, "select_sha256": sha256_file(sp), "runs": runs,
                     "confirmatory": conf, "wave2_ours_vs_ours": ds in WAVE2,
                     "device_note": ("confirmatory side NVIDIA_A40-FP32-DET (B16), exploratory side CPU-FP32"
                                     if ds == "ml_1m" else "CPU-FP32 both sides")}
    if final and pending:
        raise Refused(f"the lock needs every run DONE_MAIN; pending: {pending}")
    obj = {"format": FORMAT, "document": "EXP_BENCH_X_TEST_LOCK", "label": "POST_HOC_EXPLORATORY",
           "study": "PPSI V4 benchmarks POST_HOC_X fixed recipe (tau 1e-6, B24, unfrozen) + C_SMALL",
           "authority": "owner backlog C1 (DL 553); Fable gate, then a BIND register entry, then scoring, then a LOOK "
                        "entry; results beside the confirmatory benchmark TEST, never instead of it",
           "binds": "checkpoints and RESULT files by sha256, the scoring tree by X_SHA256SUMS + code digest, final_x by "
                    "sha256, the confirmatory TEST rank files by sha256; no TEST value was opened to build it "
                    "(validation values of these runs were seen, DL 551-557)",
           "draft": not final, "pending": pending, "written_utc": utc(), "code": ident, "datasets": dsets,
           "analysis": {"metric": "ndcg@10", "group": "ALL", "primary_rule": "PRACTICAL_BEST",
                        "secondary_rule": "CONTROLLED_BUDGET (where both sides have it)",
                        "bootstrap": {"n": BOOT_N, "seed": BOOT_SEED, "unit": "paired user bootstrap per seed, then "
                                      "the seed mean (same resampled users for every seed)"},
                        "claim_rule": "R8: |seed-mean delta| > 0.005 and every seed on the same side, else on par",
                        "retention": "100 x seed-mean X / seed-mean C_FULL (Track A C_FULL_s*, all its seeds)",
                        "contrasts": [{"id": c[0], "kind": c[1], "x": c[2], "y": c[3]} for c in CONTRASTS],
                        "script": "code_bench_x2/tools_x2/final_x.py analyze"}}
    obj["self_sha256"] = canon_sha256({k: v for k, v in obj.items() if k != "self_sha256"})
    return obj


def write_lock(obj, final: bool) -> str:
    p = os.path.join(V4, LOCK_REL if final else CAND_REL)
    data = (json.dumps(obj, indent=1, sort_keys=True, allow_nan=False) + "\n").encode()
    if final:
        fd = os.open(p, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)       # write-once
        try:
            os.write(fd, data)
        finally:
            os.close(fd)
    else:
        tmp = p + f".tmp.{os.getpid()}"
        open(tmp, "wb").write(data)
        os.replace(tmp, p)
    return p


# ------------------------------------------------------------------------------------------------ gate
def load_lock(path=None) -> tuple:
    p = path or os.path.join(V4, LOCK_REL)
    if not os.path.isfile(p):
        raise Refused(f"no exploratory lock {p} (build-lock --final after the Fable gate)")
    lock = rj(p)
    if lock.get("format") != FORMAT or lock.get("self_sha256") != canon_sha256(
            {k: v for k, v in lock.items() if k != "self_sha256"}):
        raise Refused(f"{p}: not a verified {FORMAT} file")
    return lock, p, sha256_file(p)


def register_gate(lock_file_sha: str, lock_path: str) -> dict:
    sys.path.insert(1, os.path.join(V4, "tools"))
    import test_exposure_register as ter
    reg = rj(os.path.join(V4, REG_REL))
    errs = ter.verify(reg)
    if errs:
        raise Refused(f"TEST_EXPOSURE_REGISTER does not verify: {errs[:3]}")
    rel = os.path.relpath(lock_path, V4).replace(os.sep, "/")
    mine = [e for e in reg["entries"] if e.get("all_methods_lock_sha256") == lock_file_sha]
    bind = [e for e in mine if e.get("access_purpose") == "LOCKED_EVALUATION" and e.get("observations") == "NONE"
            and e.get("source") == rel]
    if not bind:
        raise Refused(f"no BIND entry for {rel} (sha256 {lock_file_sha[:16]}) in {REG_REL}: append it with "
                      "tools/test_exposure_register.py append after the Fable gate, BEFORE any scoring")
    if any(e.get("observations") == "METRICS" for e in mine):
        raise Refused("the register already holds the LOOK (METRICS) entry for this lock: the look was taken")
    return {"bind_entry_id": bind[-1]["entry_id"], "bind_entry_sha256": bind[-1]["entry_sha256"],
            "register_entries": len(reg["entries"])}


def gate(ds: str, lock_path=None) -> tuple:
    if ds not in DATASETS:
        raise Refused(f"{ds} is not a locked dataset {DATASETS}")
    lock, lp, lsha = load_lock(lock_path)
    reg = register_gate(lsha, lp)                                 # BIND first: the dry run shows this refusal
    if lock.get("draft") or lock.get("pending"):
        raise Refused("the lock is a draft / candidate (pending runs); only the final write-once lock opens TEST")
    if os.path.exists(os.path.join(TEST_ROOT, ds, "TEST_LOOK.json")):
        raise Refused(f"runs_bench_x/TEST/{ds}/TEST_LOOK.json exists: the exploratory look of {ds} was taken")
    ident = x_tree_identity()
    for k in ("x_sha256sums_sha256", "bench_code_sha256", "final_py_sha256", "final_x_sha256"):
        if ident[k] != lock["code"][k]:
            raise Refused(f"scoring code {k} differs from the lock")
    dl = lock["datasets"][ds]
    for key, rec in dl["runs"].items():
        now = run_record(rec["run_dir"])
        if now.get("status") != "DONE_MAIN" or any(now.get(f) != rec.get(f) for f in ("result_sha256", "checkpoints",
                                                                                         "seed", "arm", "pre_provenance")):
            raise Refused(f"locked run {key} changed after the lock (or is not DONE_MAIN)")
    for k, c in dl["confirmatory"]["ranks"].items():
        if sha256_file(c["path"]) != c["sha256"]:
            raise Refused(f"confirmatory rank file {k} changed after the lock")
    cl = os.path.join(dl["confirmatory"]["root"], "LOCK.json")
    if sha256_file(cl) != dl["confirmatory"]["lock_sha256"]:
        raise Refused("the confirmatory LOCK.json changed after the lock")
    if confirmatory_view_identity(rj(cl)) != dl["confirmatory"]["view_identity"]:
        raise Refused("the confirmatory view identity (tree / data.py) changed after the lock")
    return lock, lp, lsha, reg


# ------------------------------------------------------------------------------------------------ score
def score(ds: str, threads: int, lock_path=None) -> int:
    import secrets
    import numpy as np
    lock, lp, lsha, reg = gate(ds, lock_path)
    import torch
    torch.set_num_threads(int(threads))
    from v4bench import common, data as D, final
    from v4bench.data import EvalView
    from v4bench.evaluate import summarize
    out_dir = os.path.join(TEST_ROOT, ds)
    os.makedirs(out_dir, exist_ok=True)
    att = os.path.join(out_dir, "TEST_ATTEMPTS.jsonl")

    def log(ev):
        with open(att, "a") as f:
            f.write(json.dumps(dict(ev, utc=utc()), sort_keys=True) + "\n")

    log({"event": "start", "threads": threads, "lock_file_sha256": lsha, **reg})
    try:
        root = common.default_data_root(ds)
        d = D.load(ds, root)                                     # every locked run is Track A (max_len 50)
        runs = []
        for key, rec in sorted(lock["datasets"][ds]["runs"].items()):
            res = rj(os.path.join(rec["run_dir"], "RESULT.json"))
            if int(res["spec"].get("max_len") or 50) != 50:
                raise Refused(f"{key}: max_len {res['spec'].get('max_len')} (the lock is Track A only)")
            runs.append((key, rec, res))
        vv = d.validation_view()                                 # --- preflight on 3 validation rows per checkpoint
        sel = np.unique(np.asarray([0, len(vv.users) // 2, len(vv.users) - 1]))
        sub = EvalView(d, "VALIDATION", vv.users[sel], vv.ends[sel], vv.targets[sel], source="inner")
        for key, rec, res in runs:
            for rule, c in rec["checkpoints"].items():
                r = final.score_run(rec, res, rule, c, __import__("pathlib").Path(rec["run_dir"]), d, sub, dataset=ds,
                                    data_root=root, workers=1, threads=threads, say=lambda *_: None)
                if r.shape != (len(sel),) or (r < 0).any():
                    raise Refused(f"preflight: {key} / {rule} did not score its validation rows")
        fp_want = lock["datasets"][ds]["confirmatory"]["data_fingerprint"]
        if d.fingerprint() != fp_want:                           # Fable T1: before the holdout is opened
            raise Refused(f"data fingerprint {d.fingerprint()[:16]} != the confirmatory LOCK's {fp_want[:16]}")
        tok = secrets.token_hex(16)                              # the unchanged final.py gate token, minted after
        final._TOKENS[tok] = ds                                  # the exploratory gate above
        v = D._load_holdout_view(d, root, _gate=tok)
        if ds in common.DATASETS and v.n_cold_items != 0:
            raise Refused(f"{v.n_cold_items} cold holdout items on Wave-1 {ds}")
        cf = lock["datasets"][ds]["confirmatory"]
        got = (len(v.users), v.n_cold_items, v.n_unknown_users)
        want = (cf["n_test_users"], cf["n_cold_holdout_items_as_misses"], cf["n_holdout_rows_unknown_user"])
        if got != want:
            raise Refused(f"holdout view counts {got} != the confirmatory look's {want}")
        test_users = d.user_ids[v.users].astype(np.int64)        # external user ids, in this view's row order
        scored = {}
        for key, rec, res in runs:                               # --- score in memory
            for rule, c in rec["checkpoints"].items():
                ranks = final.score_run(rec, res, rule, c, __import__("pathlib").Path(rec["run_dir"]), d, v, dataset=ds,
                                        data_root=root, workers=1, threads=threads, say=print)
                scored[(key, rule)] = (ranks, summarize(ranks, v, d.band))
        look = {"dataset": ds, "utc": utc(), "lock_file_sha256": lsha, "lock_self_sha256": lock["self_sha256"],
                "runs": sorted(k for k, _r, _s in runs), "label": "POST_HOC_EXPLORATORY", **reg}
        common.write_atomic_exclusive(os.path.join(out_dir, "TEST_LOOK.json"), look)     # --- the look, only now
        up = os.path.join(out_dir, "TEST_USERS.int64.npy")
        np.save(up, test_users)
        res_all = {"dataset": ds, "utc": utc(), "label": "POST_HOC_EXPLORATORY", "n_test_users": int(len(v.users)),
                   "test_view_digest": v.digest(), "n_cold_holdout_items_as_misses": v.n_cold_items,
                   "n_holdout_rows_unknown_user": v.n_unknown_users, "lock_file_sha256": lsha,
                   "data_fingerprint": d.fingerprint(), "test_users_file": "TEST_USERS.int64.npy",
                   "test_users_sha256": sha256_file(up), "runs": {}}
        for key, rec, _res in runs:
            kd = os.path.join(out_dir, key)
            os.makedirs(kd, exist_ok=True)
            per = {}
            for rule, c in rec["checkpoints"].items():
                ranks, summ = scored[(key, rule)]
                np.save(os.path.join(kd, f"test_{rule}.ranks_int32.npy"), ranks.astype(np.int32))
                common.write_exclusive(os.path.join(kd, f"TEST_{rule}.json"),
                                       {"key": key, "run_dir": rec["run_dir"], "rule": rule, "mark": c["mark"],
                                        "checkpoint_sha256": c["sha256"], "summary": summ, "utc": utc(),
                                        "label": "POST_HOC_EXPLORATORY"})
                per[rule] = summ
            res_all["runs"][key] = per
        common.write_exclusive(os.path.join(out_dir, "TEST_RESULTS.json"), res_all)
    except BaseException as e:                                   # noqa: BLE001 - logged, then re-raised
        log({"event": "error", "type": type(e).__name__, "message": str(e)[:500]})
        raise
    log({"event": "end", "n_runs": len(runs), "n_test_users": int(len(v.users))})
    print(f"TEST_X done {ds}: {len(runs)} runs, {len(v.users)} holdout users (values not printed)")
    return 0


# ------------------------------------------------------------------------------------------------ analysis
def ndcg10(r):
    import numpy as np
    r = np.asarray(r, dtype=np.float64)
    hit = (r >= 1) & (r <= 10)
    return np.where(hit, 1.0 / np.log2(np.where(hit, r, 1.0) + 1.0), 0.0)


def analyze(lock_path=None) -> int:
    import numpy as np
    lock, lp, lsha = load_lock(lock_path)
    if lock.get("draft"):
        raise Refused("draft lock")
    out = {"label": "POST_HOC_EXPLORATORY", "lock_file_sha256": lsha, "utc": utc(), "datasets": {},
           "beside": "the confirmatory benchmark TEST (runs_bench*/<ds>/TEST_RESULTS.json), never instead of it"}
    for ds, dl in lock["datasets"].items():
        if not os.path.isfile(os.path.join(TEST_ROOT, ds, "TEST_LOOK.json")):
            raise Refused(f"{ds}: not scored yet")
        for k, c in dl["confirmatory"]["ranks"].items():
            if sha256_file(c["path"]) != c["sha256"]:
                raise Refused(f"confirmatory rank file {k} changed")
        tres = rj(os.path.join(TEST_ROOT, ds, "TEST_RESULTS.json"))
        up = os.path.join(TEST_ROOT, ds, tres["test_users_file"])
        if sha256_file(up) != tres["test_users_sha256"]:
            raise Refused(f"{ds}: TEST_USERS file changed")
        ids_x = np.load(up)
        cf = dl["confirmatory"]
        if (confirmatory_view_identity(rj(os.path.join(cf["root"], "LOCK.json"))) != cf["view_identity"]
                or tres["data_fingerprint"] != cf["data_fingerprint"]):
            raise Refused(f"{ds}: the confirmatory view identity cannot be proven (user ids)")
        ids_c = ids_x.copy()                                    # proven identical view (lock view_identity), see docstring
        ref = np.sort(ids_x)
        if np.unique(ref).size != ref.size:
            raise Refused(f"{ds}: duplicate user ids in the TEST view")

        def by_id(ids, vec):                                    # Fable T1: pair by external user id, refuse other sets
            if ids.shape != vec.shape:
                raise Refused(f"{ds}: {vec.shape[0]} ranks for {ids.shape[0]} user ids")
            o = np.argsort(ids, kind="stable")
            if not np.array_equal(ids[o], ref):
                raise Refused(f"{ds}: the user-id sets differ between the paired arms")
            return vec[o]

        per_user = {}                                           # (arm, rule) -> {seed: per-user ndcg@10, sorted by id}
        for key, rec in dl["runs"].items():
            for rule in rec["checkpoints"]:
                r = np.load(os.path.join(TEST_ROOT, ds, key, f"test_{rule}.ranks_int32.npy"))
                per_user.setdefault((rec["key"], rule), {})[rec["seed"]] = by_id(ids_x, ndcg10(r))
        for k, c in cf["ranks"].items():
            name, rule = k.split("|")
            arm, _, sd = name.rpartition("_s")
            per_user.setdefault((arm, rule), {})[int(sd)] = by_id(ids_c, ndcg10(np.load(c["path"])))
        n = int(ref.size)
        rng = np.random.Generator(np.random.PCG64(BOOT_SEED))
        idx = rng.integers(0, n, size=(BOOT_N, n))
        dres = {"n_test_users": n, "means": {}, "contrasts": {}, "wave2_ours_vs_ours": dl["wave2_ours_vs_ours"],
                "device_note": dl["device_note"]}
        for (arm, rule), vecs in sorted(per_user.items()):
            dres["means"][f"{arm}|{rule}"] = {"per_seed": {str(s): float(x.mean()) for s, x in sorted(vecs.items())},
                                              "seed_mean": float(np.mean([x.mean() for x in vecs.values()]))}
        for cid, kind, xa, ya in CONTRASTS:
            for rule in RULES:
                X, Y = per_user.get((xa, rule)), per_user.get((ya, rule))
                if not X or not Y:
                    continue
                key = f"{cid}|{'PB' if rule == 'PRACTICAL_BEST' else 'CB'}"
                if kind == "ratio":
                    mx = float(np.mean([v.mean() for v in X.values()]))
                    my = float(np.mean([v.mean() for v in Y.values()]))
                    dres["contrasts"][key] = {"kind": "ratio", "value": 100.0 * mx / my if my else None,
                                              "x_seed_mean": mx, "y_seed_mean": my,
                                              "x_seeds": sorted(X), "y_seeds": sorted(Y)}
                    continue
                seeds = sorted(set(X) & set(Y))
                if not seeds:
                    continue
                per = {s: float(X[s].mean() - Y[s].mean()) for s in seeds}
                boot = np.mean([(X[s][idx] - Y[s][idx]).mean(axis=1) for s in seeds], axis=0)
                m = float(np.mean(list(per.values())))
                same = all(v > 0 for v in per.values()) or all(v < 0 for v in per.values())
                claim = ("above" if m > 0 else "below") if (abs(m) > R8_TOL and same) else "on par"
                if dl["wave2_ours_vs_ours"]:                    # B14.5: Wave 2 is ours-vs-ours, no claim rule
                    claim = None
                dres["contrasts"][key] = {"kind": "delta", "seed_mean": m, "per_seed": {str(s): v for s, v in per.items()},
                                          "ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
                                          "r8_claim": claim, "paired_seeds": seeds}
        out["datasets"][ds] = dres
    os.makedirs(ANALYSIS_DIR, exist_ok=True)
    p = os.path.join(ANALYSIS_DIR, "X_RESULT.json")
    data = (json.dumps(out, indent=1, sort_keys=True, allow_nan=False) + "\n").encode()
    fd = os.open(p, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        os.write(fd, data)
    finally:
        os.close(fd)
    print(f"X_RESULT written: {p} sha256 {sha256_file(p)} (values not printed)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build-lock")
    b.add_argument("--final", action="store_true")
    for name in ("check", "score"):
        p = sub.add_parser(name)
        p.add_argument("--dataset", required=True)
        p.add_argument("--lock", default=None, help="tests only: another lock file (the gate still needs its BIND)")
        p.add_argument("--threads", type=int, default=16)
    sub.add_parser("fingerprints", help="dry run: train-only data fingerprint vs every confirmatory LOCK (no holdout)")
    a_ = sub.add_parser("analyze")
    a_.add_argument("--lock", default=None)
    a = ap.parse_args()
    try:
        if a.cmd == "build-lock":
            obj = build_lock(a.final)
            p = write_lock(obj, a.final)
            print(f"{'LOCK' if a.final else 'CANDIDATE'} written: {p} file sha256 {sha256_file(p)} self "
                  f"{obj['self_sha256']} pending {len(obj['pending'])}: {obj['pending']}")
            return 0
        if a.cmd == "check":
            gate(a.dataset, a.lock)
            print(f"GATE PASS {a.dataset} (nothing scored)")
            return 0
        if a.cmd == "score":
            return score(a.dataset, a.threads, a.lock)
        if a.cmd == "fingerprints":
            from v4bench import common, data as D
            bad = 0
            for ds in DATASETS:
                d = D.load(ds, common.default_data_root(ds))
                want = rj(os.path.join(conf_root(ds), ds, "LOCK.json"))["data_fingerprint"]
                ok = d.fingerprint() == want
                bad += not ok
                print(f"{ds}: fingerprint {d.fingerprint()[:16]} confirmatory {want[:16]} {'OK' if ok else 'DIFFERS'}")
            return 1 if bad else 0
        return analyze(a.lock)
    except Refused as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
