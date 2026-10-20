"""Draft register entries (NOT appended) for an EXP lock: <doc>_BIND (observations NONE, before any TEST scoring; the one
reeval_x requires) and <doc>_LOOK (observations METRICS, appended after X_RESULT is read). Format = REGISTER_ENTRY_9 / 11.
  python make_entries.py <lock> <out_dir> <code_sha256>"""
import hashlib
import json
import sys
from pathlib import Path

lock_p, out, code = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
sha = hashlib.sha256(lock_p.read_bytes()).hexdigest()
lock = json.loads(lock_p.read_text())
doc = lock["document"]
rel = f"company/gates/{doc}.json"
base = {"access_purpose": "LOCKED_EVALUATION", "actor_role": "Operations Director", "all_methods_lock_sha256": sha,
        "code_sha256": code, "decision": "NO_FEEDBACK", "manifest_header_sha256": None, "model_checkpoint_sha256": None,
        "split": "RETRO_TEST"}
bind = dict(base, observations="NONE", receipt_path=rel, source=rel, observations_description=(
    f"LOCKED_EVALUATION for {doc} (POST_HOC_EXPLORATORY lock, self_sha256 {lock['self_sha256'][:16]}, owner approval "
    "DL 539, Fable gate passed): appended before the first exploratory TEST scoring; no TEST row read, no model scored, "
    "no metric; results are reported beside the confirmatory T0a results, never instead of them"))
res = f"runs_x_rees46/TEST/ANALYSIS_{doc}/X_RESULT.json"
look = dict(base, observations="METRICS", receipt_path=res, source=res, observations_description=(
    f"Exploratory TEST look under {doc} (POST_HOC_EXPLORATORY, bind entry appended first): X_RESULT under "
    f"runs_x_rees46/TEST/ANALYSIS_{doc}, written by x_rees46/x_analysis.py; read for the write-up beside the "
    "confirmatory results; nothing re-run, re-selected or added; confirmatory T0a results unchanged"))
for name, e in (("BIND", bind), ("LOOK", look)):
    p = out / f"REGISTER_ENTRY_DRAFT_{doc}_{name}.json"
    p.write_text(json.dumps(e, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(p, hashlib.sha256(p.read_bytes()).hexdigest())
