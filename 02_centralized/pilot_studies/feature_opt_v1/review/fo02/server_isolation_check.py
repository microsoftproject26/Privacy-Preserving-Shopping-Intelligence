"""FO-02 server isolation check (CPU only, read-only): import every study stage module the way the queue
does, force the lazy imports that failed in the probes stage, then run assert_isolated()."""
import json
import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "code"))
import fo_env  # noqa: E402
import feature_view, fo_models, fo_eval, precision  # noqa: E402,F401
import probe_runner, screen, finalize_fo, pin_baseline, stream_fit  # noqa: E402,F401
from full_store import CompactStore  # noqa: E402,F401  (the lazy import that failed at 07:35Z)
import sasrec_data, sasrec_model, train_sasrec  # noqa: E402,F401
import baselines_full, lock_and_eval, verify_frozen  # noqa: E402,F401
seen = fo_env.assert_isolated()
foreign = [p for p in sys.path if "gru_v1" in p or "sasrec_v1/code" in p]
man = fo_env.verify_snapshot()
print(json.dumps({"assert_isolated": "OK", "modules_checked": len(seen), "not_loaded": [m for m in fo_env.ISOLATED_MODULES if m not in seen],
                  "foreign_sys_path": foreign, "snapshot_files_verified": len(man["files"]),
                  "sample": {k: seen[k] for k in ("full_store", "train", "model", "eval_full") if k in seen}}, indent=1))
