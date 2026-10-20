"""protect_diff.py (owner: Opus) — compare two protection snapshots (state/PROTECT_<a>.json vs <b>).

Package test 1 (current-job protection): an existing campaign code/config/lock file whose hash changed
between T0 (study start) and a later snapshot is reported. Added files are listed separately; the
campaign's own final stage legitimately adds lock/result files. Writes audit/PROTECT_DIFF_<a>_<b>.json.
"""
import json
import sys
from pathlib import Path

F = Path(__file__).resolve().parents[1]
a, b = sys.argv[1], sys.argv[2]
A = json.loads((F / "state" / f"PROTECT_{a}.json").read_text())
B = json.loads((F / "state" / f"PROTECT_{b}.json").read_text())
fa, fb = A["files"], B["files"]
changed = sorted(k for k in fa if k in fb and fa[k] != fb[k])
out = {"a": a, "b": b, "a_utc": A["utc"], "b_utc": B["utc"],
       "changed_existing": changed, "removed": sorted(set(fa) - set(fb)), "added": sorted(set(fb) - set(fa)),
       "dataset_metadata_changed": sorted(k for k in A["dataset_metadata"] if A["dataset_metadata"][k] != B["dataset_metadata"].get(k)),
       "dataset_tree_equal": A["dataset_tree"] == B["dataset_tree"],
       "code_or_config_changed": [k for k in changed if "/code/" in k or "/tools/" in k or "/config/" in k]}
out["verdict"] = "UNCHANGED" if not out["code_or_config_changed"] and not out["removed"] and not out["dataset_metadata_changed"] \
    and out["dataset_tree_equal"] else "CHANGES_FOUND"
(F / "audit" / f"PROTECT_DIFF_{a}_{b}.json").write_text(json.dumps(out, indent=1))
print(json.dumps({k: (v if not isinstance(v, list) else len(v)) for k, v in out.items()}))
