#!/bin/bash
# FIX-1 (Fable gate 2026-10-03) — LOCAL, Git Bash: pack code_ds3r7b = r7 + DEV-05 + the OPS-B13-P2 --latency-only bench.
# Built FROM the deployed r7 tar (deploy_archive/r7/code_ds3r7.tar): every r7 member is copied byte for byte, and only the
# 8 reviewed files are replaced / added from the local code/ tree, each asserted against its reviewed sha
# (DEV-04.md DEV-05 table + REV-DEV-05 PASS, DL 358 / 359; OPS-B13-P2.md + REV-OPS-B13-P2 PASS, DL 378 / 380).
# Refuses (exit 3) unless the manifest diff r7 -> r7b is exactly those 8 paths, and (exit 4) if any v4eval/*,
# v4int/reeval.py, v4int/launch_*.py, fl/* or receipt differs from r7. No tools tar: tools/ are unchanged (r7).
set -euo pipefail
cd "/f/Queens University/Microsoft Project/eda/.local_ai/ppsi_v4"
OUT=deploy_archive/r7b
[ ! -e $OUT ] || { echo "REFUSED: $OUT exists (a pack is never overwritten)"; exit 2; }
mkdir -p $OUT
python - <<'PY'
import hashlib, io, json, sys, tarfile
from pathlib import Path
R7T, R7M = Path("deploy_archive/r7/code_ds3r7.tar"), Path("deploy_archive/r7/SYNC_MANIFEST.json")
def h(b): return hashlib.sha256(b).hexdigest()
m7 = json.loads(R7M.read_text())
if h(R7T.read_bytes()) != m7["tar_sha256"]: print("REFUSED: r7 tar != its manifest"); sys.exit(3)
DELTA = {  # path in tar -> (r7 sha prefix or None for NEW, reviewed r7b sha)
 "code/v4dev/dump_rows.py": ("50c1a25d", "3961e2306258869c3d146b45a1d0cce517d89b51dbbd8b953046e00727a1d523"),
 "code/v4dev/bench.py": ("0b74e77a", "9662cba5e706fa90fb4f96f46c4d1653f64ec2769b2cdc7fe515cf23385d210b"),
 "code/v4int/dump_rows_retro.py": ("9853cc55", "83f5f2f7e48cc92b3dd0a28e384faedd397b6c2577c49f5bfe3dbb23b064c0cc"),
 "code/tests_int/test_int07_test_path.py": ("7dfdd802", "c9558aad998f6aa5b331ecb6212c38f1d395566c7feb6d770b010ed80051b291"),
 "code/tests_dev/test_dump_rows_retro_view.py": (None, "6efd04bd1efa4df263a090a9c588bac8641e8d330c6b32aeeb27805f87e0ddba"),
 "code/tests_dev/r6_ref/dump_rows_r6.py.txt": (None, "4696c4af329813e9984c15e5658ef2fe6704ea06643ec0b26ebbbc73c7cf3cc9"),
 "code/tests_dev/r6_ref/dump_rows_retro_r6.py.txt": (None, "9853cc554799542605c3e0e896a895ba10b55b5ca7a8ce030b338f72a0844889"),
 "code/tests_dev/test_bench_latency_only.py": (None, "71f79745f533e19d0550ab0bbae813dadeab9fd61b851ffcf76d28b73bb4e49e"),
}
data = {}
for k, (old, new) in DELTA.items():
    b = Path(k).read_bytes()
    if h(b) != new: print("REFUSED: local", k, "is not the reviewed", new[:12]); sys.exit(5)
    if (old is None) != (k not in m7["files"]) or (old and not m7["files"][k].startswith(old)):
        print("REFUSED: r7 state of", k, "is not", old); sys.exit(3)
    data[k] = b
out = Path("deploy_archive/r7b/code_ds3r7b.tar")
with tarfile.open(R7T) as src, tarfile.open(out, "w", format=tarfile.GNU_FORMAT) as dst:
    have_dirs = set()
    for ti in src:
        if ti.isdir(): have_dirs.add(ti.name.rstrip("/"))
        if ti.isfile() and ti.name in data:
            ni = tarfile.TarInfo(ti.name); ni.size = len(data[ti.name]); ni.mode = ti.mode; ni.mtime = ti.mtime
            ni.uid, ni.gid, ni.uname, ni.gname = ti.uid, ti.gid, ti.uname, ti.gname
            dst.addfile(ni, io.BytesIO(data[ti.name]))
        else:
            dst.addfile(ti, src.extractfile(ti) if ti.isfile() else None)
    for k in sorted(k for k, (old, _) in DELTA.items() if old is None):
        d = str(Path(k).parent).replace("\\", "/")
        if d not in have_dirs:
            di = tarfile.TarInfo(d); di.type = tarfile.DIRTYPE; di.mode = 0o755; di.mtime = int(Path(k).parent.stat().st_mtime)
            dst.addfile(di); have_dirs.add(d)
        ni = tarfile.TarInfo(k); ni.size = len(data[k]); ni.mode = 0o644; ni.mtime = int(Path(k).stat().st_mtime)
        dst.addfile(ni, io.BytesIO(data[k]))
files = {}
with tarfile.open(out) as tf:
    for mm in tf:
        if mm.isfile(): files[mm.name] = h(tf.extractfile(mm).read())
agg = h("".join(f"{k}\0{v}\n" for k, v in sorted(files.items())).encode())
man = {"tar_sha256": h(out.read_bytes()), "aggregate_sha256": agg, "files": files,
       "base": {"tree": "code_ds3r7", "tar_sha256": m7["tar_sha256"], "aggregate_sha256": m7["aggregate_sha256"]},
       "delta_reviewed": {k: v[1] for k, v in DELTA.items()}}
Path("deploy_archive/r7b/SYNC_MANIFEST.json").write_text(json.dumps(man, indent=1, sort_keys=True))
diff = sorted(k for k in set(files) | set(m7["files"]) if files.get(k) != m7["files"].get(k))
print("CODE TAR", man["tar_sha256"]); print("CODE AGG", agg); print("N r7", len(m7["files"]), "r7b", len(files))
for k in diff: print("  ", "NEW" if k not in m7["files"] else "CHANGED" if k in files else "GONE", k, (files.get(k) or "")[:12])
if diff != sorted(DELTA): print("REFUSED: r7 -> r7b diff is not exactly the 8 reviewed paths"); sys.exit(3)
frozen = [k for k in diff if k.startswith(("code/v4eval/", "code/fl/")) or k in ("code/v4int/reeval.py",)
          or (k.startswith("code/v4int/launch_")) or "/receipts/" in k]
if frozen: print("REFUSED: frozen r7 file changed", frozen); sys.exit(4)
for k in ("code/v4int/reeval.py", "code/v4int/launch_fl.py", "code/v4int/launch_device.py"):
    print("KEY", k, files[k][:16], "= r7" if files[k] == m7["files"][k] else "!= r7")
print("PACK_R7B_OK")
PY
# then: fill deploy_r7b.tmpl.sh (__CODE_SHA__) -> upload tar + manifest to $R/deploy_r7b/ -> deploy -> traj_cmp7b.sh
