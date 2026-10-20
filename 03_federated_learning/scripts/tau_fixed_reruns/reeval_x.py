"""X-REES46 exploratory scoring gate (POST_HOC_EXPLORATORY, owner approval DL 539). It never scores by itself: it checks an
EXPLORATORY lock (format PPSI_V4_X_EXP_TEST_LOCK_1, e.g. company/gates/EXP_PB100_TEST_LOCK.json) and then runs the
UNCHANGED scorer named by the lock entry (v4int.reeval / exp_pb100/reeval_pb100.py / x_rees46/reeval_p2_x.py /
v4int.launch_device) from the lock's scoring code tree.

  python -B x_rees46/reeval_x.py --lock <EXP lock> --lock-sha <its DL-registered file sha256> --entry <key> --gpu 0
         [--plan]                  print the exact scorer command and every check; never runs the scorer
         [--proof-role MAIN]       VALIDATION proof: the same entry on role MAIN, out dir runs_x_rees46/PROOF/<key>
         [--dry-run]               pass --dry-run to the scorer

Checks, in this order, before any checkpoint byte is read by the scorer (each failure exits 2):
  1 the lock file sha256 = --lock-sha; strict JSON; format / draft false / label POST_HOC_EXPLORATORY; self_sha256
  2 the entry exists once; role is MAIN_TEST unless --proof-role MAIN (no other role is admitted)
  3 the scoring tree digest (P2-lock rule) = the lock's; every pinned script sha256 = the lock's
  4 the checkpoint (and the FT base) sha256 = the lock's
  5 TEST role only: the hash-chain-verified TEST_EXPOSURE_REGISTER (tools/test_exposure_register.verify, unchanged) holds
    a LOCKED_EVALUATION / RETRO_TEST entry whose all_methods_lock_sha256 = the EXP lock file sha256 and whose
    observations_description names the lock document; otherwise refused (no TEST role is ever resolved)
  6 the output dir is under runs_x_rees46 and absent (a non-empty existing dir is a manual decision)
After the scorer exits 0 a X_RECEIPT.json (label, lock sha, entry, argv, rc, utc) is written next to its outputs."""
import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path

V = Path(__file__).resolve().parents[1]
PY = "/home/jovyan/venvs/ppsi-gpu/bin/python"
FORMAT = "PPSI_V4_X_EXP_TEST_LOCK_1"
LABEL = "POST_HOC_EXPLORATORY"
REG = V / "company" / "registers" / "TEST_EXPOSURE_REGISTER.json"
TEST_EXT = V / "contracts" / "manifest_registry" / "REGISTRY_V2_TEST_EXT.json"
XR = V / "runs_x_rees46"


def refuse(msg):
    print(f"reeval_x REFUSED: {msg}", file=sys.stderr)
    sys.exit(2)


def sha_file(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def strict(p):
    return json.loads(Path(p).read_text(encoding="utf-8"),
                      parse_constant=lambda c: (_ for _ in ()).throw(ValueError(f"non-strict constant {c}")))


def self_sha(obj) -> str:
    body = {k: v for k, v in obj.items() if k != "self_sha256"}
    return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
                          .encode("utf-8")).hexdigest()


def tree_digest(root: Path) -> tuple:
    """P2_METHODS_LOCK code_tree rule: sha256(compact sort_keys JSON of {posix relpath: file sha256}); __pycache__ /
    .pytest_cache / *.pyc skipped, symlinks refused."""
    files = {}
    for p in sorted(root.rglob("*")):
        rel = p.relative_to(root)
        if any(part in ("__pycache__", ".pytest_cache") for part in rel.parts) or p.suffix == ".pyc":
            continue
        if p.is_symlink():
            refuse(f"symlink in the scoring tree: {rel}")
        if p.is_file():
            files[rel.as_posix()] = sha_file(p)
    return hashlib.sha256(json.dumps(files, sort_keys=True, separators=(",", ":")).encode()).hexdigest(), len(files)


def register_binding(lock_sha: str, document: str) -> dict:
    spec = importlib.util.spec_from_file_location("x_ter", V / "tools" / "test_exposure_register.py")
    ter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ter)
    reg = strict(REG)
    errs = ter.verify(reg)
    if errs:
        refuse(f"TEST_EXPOSURE_REGISTER hash chain does not verify: {errs[:3]}")
    hits = [e for e in reg["entries"] if e.get("access_purpose") == "LOCKED_EVALUATION"
            and e.get("split") == "RETRO_TEST" and e.get("all_methods_lock_sha256") == lock_sha]
    if not hits:
        refuse(f"TEST refused: no LOCKED_EVALUATION / RETRO_TEST register entry binds the EXP lock {lock_sha} "
               "(Ops appends it after the Fable gate)")
    if any(document not in str(e.get("observations_description") or "") for e in hits):
        refuse(f"TEST refused: a binding register entry does not name {document} in its observations_description")
    e = min(hits, key=lambda x: int(x["entry_id"]))
    return {"entry_id": int(e["entry_id"]), "entry_sha256": e["entry_sha256"], "register_file_sha256": sha_file(REG)}


def build_argv(e: dict, role: str, out: Path, gpu: int, dry: bool) -> tuple:
    s = e["scorer"]
    if s == "device":                                        # v4int.launch_device (FT on top of a base)
        a = list(e["device_args"])
        a[a.index("--role") + 1] = role
        a[a.index("--run-dir") + 1] = str(out)
        a += ["--gpu", str(gpu), "--allow-shared-gpu", "--history-policy", "USER_HISTORY_L50"]
        if role.endswith("TEST"):
            a += ["--registry-ext", str(TEST_EXT)]
        return a + (["--dry-run"] if dry else []), "v4int/launch_device.py"
    a = ["--history-policy", "USER_HISTORY_L50", "--gpu", str(gpu), "--allow-shared-gpu",
         "--ckpt", e["ckpt"], "--ckpt-kind", e["ckpt_kind"], "--expect-sha256", e["checkpoint_sha256"],
         "--variant", e["variant"], "--recipe-id", e["recipe_id"], "--seed", str(e["seed"]), "--role", role,
         "--arm-label", e["arm_label"], "--selection-rule", e["rule"], "--checkpoint-id", e["checkpoint_id"]]
    if role.endswith("TEST"):
        a += ["--registry-ext", str(TEST_EXT)]
    a += ["--out-dir", str(out)] + (["--dry-run"] if dry else [])
    script = {"reeval": "v4int/reeval.py", "reeval_pb100": str(V / "exp_pb100" / "reeval_pb100.py"),
              "reeval_p2_x": str(V / "x_rees46" / "reeval_p2_x.py"),
              "reeval_cold_x": str(V / "x_rees46" / "reeval_cold_x.py")}[s]
    return a, script


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lock", required=True, type=Path)
    ap.add_argument("--lock-sha", required=True)
    ap.add_argument("--entry", required=True)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("--proof-role", choices=("MAIN",), default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
    if args.gpu != 0:
        refuse("GPU 0 only (GPU 1 carries the benchmark TEST look)")
    # 1 lock
    if not args.lock.is_file() or sha_file(args.lock) != args.lock_sha.lower():
        refuse(f"{args.lock} is absent or its sha256 is not --lock-sha {args.lock_sha[:12]}")
    lock = strict(args.lock)
    if lock.get("format") != FORMAT or lock.get("draft") is not False or lock.get("label") != LABEL:
        refuse(f"{args.lock} is not a final {FORMAT} lock labelled {LABEL}")
    if self_sha(lock) != lock.get("self_sha256"):
        refuse("lock self_sha256 does not verify")
    # 2 entry / role
    hits = [e for e in lock["sealed_arms"] + lock.get("device_optional", []) if e["key"] == args.entry]
    if len(hits) != 1:
        refuse(f"entry {args.entry!r} bound {len(hits)} times in the lock")
    e = hits[0]
    if e["role"] != "MAIN_TEST":
        refuse(f"entry role {e['role']} is not MAIN_TEST")
    role = args.proof_role or "MAIN_TEST"
    out = (XR / "PROOF" / f"{e['key']}_{role}" if args.proof_role else Path(e["out_dir"]))
    if XR.resolve() not in out.resolve().parents:
        refuse(f"output {out} is not under {XR}")
    # 3 code
    tree = V / lock["code"]["scoring_tree"]["name"]
    dig, n = tree_digest(tree)
    if dig != lock["code"]["scoring_tree"]["aggregate_sha256"]:
        refuse(f"scoring tree {tree.name} digest {dig[:12]} != the lock's")
    for rel, want in lock["code"]["pinned_files"].items():
        if sha_file(V / rel) != want:
            refuse(f"{rel} sha256 != the lock's")
    # 4 checkpoint / FT base
    for k in ("ckpt", "base"):
        if e.get(k):
            if sha_file(e[k]) != e[f"{'checkpoint' if k == 'ckpt' else 'base'}_sha256"]:
                refuse(f"{k} {e[k]} sha256 != the lock's")
    # 5 TEST binding
    binding = None
    if role.endswith("TEST"):
        binding = register_binding(args.lock_sha.lower(), lock["document"])
    # 6 output dir
    if out.exists() and any(out.iterdir()):
        refuse(f"{out} exists and is not empty (manual decision)")
    a, script = build_argv(e, role, out, args.gpu, args.dry_run)
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", CUBLAS_WORKSPACE_CONFIG=":4096:8",
               V4_DATASET_ROOT="/home/jovyan/shared_data/rees46_nextitem_v1", PB100_CODE=str(tree), X_CODE=str(tree),
               X_LOCK=str(args.lock.resolve()), X_LOCK_SHA=args.lock_sha.lower(),
               **{k: str(v) for k, v in (lock.get("scorer_env") or {}).items()})
    env.pop("CUDA_VISIBLE_DEVICES", None)
    cmd = [PY, "-B", script] + a
    print(json.dumps({"label": LABEL, "entry": e["key"], "role": role, "out": str(out), "cwd": str(tree),
                      "tree_digest": dig, "n_files": n, "register_binding": binding, "cmd": cmd}, indent=1))
    if args.plan:
        print("PLAN ONLY: scorer not run")
        return 0
    rc = subprocess.run(cmd, cwd=str(tree), env=env).returncode
    if rc == 0 and out.is_dir() and not args.dry_run:
        rec = {"label": LABEL, "lock": str(args.lock), "lock_sha256": args.lock_sha.lower(), "entry": e,
               "role": role, "cmd": cmd, "rc": rc, "register_binding": binding,
               "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        (out / "X_RECEIPT.json").write_text(json.dumps(rec, indent=1, sort_keys=True), encoding="utf-8")
    return rc


if __name__ == "__main__":
    sys.exit(main())
