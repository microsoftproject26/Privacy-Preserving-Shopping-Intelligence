"""X-REES46 cold-FA scorer (POST_HOC_EXPLORATORY): the UNCHANGED v4int.reeval (code_x1; its v4int/reeval.py is
byte-identical to code_ds3r7's, sha256 833f0c98..., the file the confirmatory Phase-1 TEST scoring ran) for the C1 cold
FA runs (x_rees46/launch_fl_x3.py, runs_x_rees46/COLD/MAIN). One in-process change: launch_fl admits FedAdam tau 1e-6
and eta {1e-4, 3e-4, 3e-3} (the same admission as launch_fl_x3.py) so reeval's launch_fl spec checks re-derive the run's
own spec. Run ONLY through x_rees46/reeval_x.py (EXP lock + register gate first)."""
import os
import sys
from pathlib import Path

sys.path.insert(0, os.environ["X_CODE"])
import v4int.launch_fl as LF  # noqa: E402

LF.FEDADAM_TAUS = tuple(LF.FEDADAM_TAUS) + (1e-6,)
LF.FEDADAM_EDGE_LRS = tuple(LF.FEDADAM_EDGE_LRS) + (1e-4, 3e-4, 3e-3)
from v4int import reeval as R  # noqa: E402

if __name__ == "__main__":
    XR = Path(__file__).resolve().parents[1] / "runs_x_rees46" / "COLD" / "MAIN"
    ck = Path(sys.argv[sys.argv.index("--ckpt") + 1]).resolve() if "--ckpt" in sys.argv else None
    if ck is None or XR.resolve() not in ck.parents:
        sys.exit("reeval_cold_x: the checkpoint is not under runs_x_rees46/COLD/MAIN")
    raise SystemExit(R.main())
