"""FIX-DELTA-01 (a) — review of Ops' unreviewed pretrain_small.py patch (DECISION_LOG 212; 06G §5 / 06K §1): the shim
refuses --allow-shared-gpu for the PRETRAIN MAIN-phase run unless --coschedule-06g is also given; with both flags the
launch contract passes, both flags reach launch_central unchanged (launch_argv) and launch_central's own preflight
accepts the busy card and records preflight.coschedule_06g = True; --coschedule-06g alone is refused (shim and
launcher). Negative control: the shim accepts --allow-shared-gpu alone.
"""
from __future__ import annotations

import json
import subprocess

from data_testkit import assert_raises, nc

ARGV = ["--family", "SASREC", "--seed", "2026", "--phase", "MAIN", "--recipe-id", "SMALL_SASREC_D64_B2", "--gpu", "0",
        "--run-dir", "RUNDIR", "--history-policy", "USER_HISTORY_L50"]
BUS = {"0": "00000000:46:00.0", "1": "00000000:47:00.0"}


def _args(extra=()):
    from v4int.launch_central import parse_args
    return parse_args(ARGV + list(extra))


def _busy_runner():
    """nvidia-smi stand-in (the tests_int/test_launcher pattern): GPU 0 runs a foreign process with 3,000 MiB."""
    def run(argv, **kw):
        q = argv[1]
        if q.startswith("--query-gpu=index,pci.bus_id"):
            out = "\n".join(f"{i}, {b}" for i, b in BUS.items())
        elif q.startswith("--query-compute-apps"):
            out = f"{BUS['0']}, 4242"
        elif q.startswith("--query-gpu=index,uuid"):
            out = "\n".join(f"{i}, GPU-{i}, NVIDIA A40, {3000.0 if i == '0' else 0.0}, 46068, 0" for i in BUS)
        else:
            out = ""
        return type("R", (), {"returncode": 0, "stdout": out + "\n", "stderr": ""})()
    return run


def test_pretrain_coschedule_flag_pair(tmp_path):
    from v4int import launch_central as LC
    from v4int.pretrain_small import PretrainRefused, check_launch_args, normalise_argv
    assert_raises(PretrainRefused, check_launch_args, _args(["--allow-shared-gpu"]))
    assert_raises(PretrainRefused, check_launch_args, _args(["--coschedule-06g"]))
    assert_raises(PretrainRefused, check_launch_args, _args(["--allow-shared-gpu", "--smoke"]))
    a = _args(["--allow-shared-gpu", "--coschedule-06g"])
    rid = check_launch_args(a)
    assert rid == "PRETRAIN_FOR_FL_SMALL_SASREC_D64_B2_s2026"
    largv = normalise_argv(ARGV + ["--allow-shared-gpu", "--coschedule-06g"], a, rid)
    assert "--allow-shared-gpu" in largv and "--coschedule-06g" in largv                 # passed to launch_central
    gate = tmp_path / "G1_APPROVED.json"
    gate.write_text(json.dumps({"gate": "G1", "note": "test fixture only"}), encoding="utf-8")
    la = LC.parse_args(largv)
    rec = LC.preflight(la, runner=_busy_runner(), environ={}, gate_path=gate)
    assert rec["coschedule_06g"] is True and rec["gpu_check"]["shared_gpu"]["allowed"] is True
    no_flag = LC.parse_args(normalise_argv(ARGV + ["--allow-shared-gpu"], _args(["--allow-shared-gpu"]), rid))
    assert_raises(LC.LaunchRefused, LC.preflight, no_flag, runner=_busy_runner(), environ={}, gate_path=gate)
    assert check_launch_args(_args()) == rid                                           # no flags: unchanged


@nc("the PRETRAIN shim accepts --allow-shared-gpu without --coschedule-06g")
def test_nc_shim_accepts_shared_gpu_alone():
    from v4int.pretrain_small import PretrainRefused, check_launch_args
    try:
        check_launch_args(_args(["--allow-shared-gpu"]))
        accepted = True
    except PretrainRefused:
        accepted = False
    assert accepted, "refused (as it must be)"
