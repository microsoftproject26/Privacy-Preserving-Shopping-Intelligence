"""make_snapshot.py (owner: Opus): byte-identical, hashed import snapshot of the campaign's SASRec code for this study.

The study imports ONLY from code/snap_sasrec (never from the live sasrec_v1/code tree). The source is
never modified; every copied file is verified sha256-equal to its source, and the manifest is written
to code/snap_sasrec/SNAPSHOT_MANIFEST.json.
Usage: make_snapshot.py <sasrec_code_dir> <reuse_dir> <dest_snap_dir>
"""
import hashlib, json, shutil, sys, time
from pathlib import Path
FILES = ["paths.py", "sasrec_model.py", "sasrec_data.py", "sasrec_objective.py", "sasrec_schedule.py",
         "precision.py", "train_sasrec.py", "synthetic_store.py", "ckpt_io.py"]
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
src, reuse, dst = (Path(a) for a in sys.argv[1:4])
if dst.exists():
    raise SystemExit(f"{dst} exists; snapshots are immutable (delete only this study's own snapshot deliberately)")
dst.mkdir(parents=True)
man = {"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "source": str(src), "reuse_source": str(reuse), "files": {}}
for f in FILES:
    shutil.copy2(src / f, dst / f); a, b = sha(src / f), sha(dst / f)
    assert a == b, f; man["files"][f] = a
for p in sorted(reuse.rglob("*")):
    if p.is_file() and "__pycache__" not in p.parts:
        rel = Path("reuse") / p.relative_to(reuse); (dst / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, dst / rel); a, b = sha(p), sha(dst / rel)
        assert a == b, rel; man["files"][rel.as_posix()] = a
(dst / "SNAPSHOT_MANIFEST.json").write_text(json.dumps(man, indent=1, sort_keys=True))
print(len(man["files"]), "files snapshotted ->", dst)
