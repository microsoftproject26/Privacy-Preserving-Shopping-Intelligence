"""FIX-DELTA-01 (a) — review of Ops' unreviewed launch_central edit `--coschedule-06g` (PREG1_AMENDMENT_06G §5 / 06K §1,
DECISION_LOG 207-208): a non-smoke central MAIN run on a busy card is refused with --allow-shared-gpu alone, allowed with
--allow-shared-gpu --coschedule-06g (recorded: preflight.coschedule_06g + gpu_check.shared_gpu); --coschedule-06g alone
is refused (it takes effect only with --allow-shared-gpu); every other launch keeps its old behaviour.
Negative control: without --allow-shared-gpu the 06G flag makes a busy GPU acceptable.
"""
from __future__ import annotations

from intkit import assert_raises, nc
from test_int03_small import SR, _targs
from test_launcher import fake_runner, gate
from v4int import launch_central as L

MAIN = ["--phase", "MAIN", "--recipe-id", SR.RECIPE_ID]


def _busy():
    return fake_runner(apps=[("00000000:46:00.0", "4242")], mem={"0": 3000.0, "1": 0.0})


def test_coschedule_06g_main_run(tmp_path):
    g = gate(tmp_path)
    pf = dict(runner=_busy(), environ={}, gate_path=g)
    assert_raises(L.LaunchRefused, L.preflight, _targs(tmp_path, *MAIN, smoke=False), **pf)            # busy card
    assert_raises(L.LaunchRefused, L.preflight, _targs(tmp_path, *MAIN, "--allow-shared-gpu", smoke=False), **pf)
    assert_raises(L.LaunchRefused, L.preflight, _targs(tmp_path, *MAIN, "--coschedule-06g", smoke=False), **pf)
    rec = L.preflight(_targs(tmp_path, *MAIN, "--allow-shared-gpu", "--coschedule-06g", smoke=False), **pf)
    assert rec["coschedule_06g"] is True and rec["gpu_check"]["shared_gpu"]["allowed"] is True
    assert rec["gpu_check"]["shared_gpu"]["memory_used_mib_at_launch"] == 3000.0
    free = L.preflight(_targs(tmp_path, *MAIN, smoke=False), runner=fake_runner(), environ={}, gate_path=g)
    assert free["coschedule_06g"] is False and "shared_gpu" not in free["gpu_check"]  # default path unchanged
    smoke = L.preflight(_targs(tmp_path, *MAIN, "--allow-shared-gpu"), **pf)          # smoke: allowed as before
    assert smoke["coschedule_06g"] is False and smoke["gpu_check"]["shared_gpu"]["allowed"] is True
    assert_raises(L.LaunchRefused, L.preflight,                                        # nvidia-smi failure: refused
                  _targs(tmp_path, *MAIN, "--allow-shared-gpu", "--coschedule-06g", smoke=False),
                  runner=fake_runner(fail=True), environ={}, gate_path=g)
    assert L.parse_args(["--family", "SASREC", "--seed", "2026", "--gpu", "0", "--run-dir", "r", "--phase", "MAIN",
                         "--history-policy", "USER_HISTORY_L50"]).coschedule_06g is False     # default off


@nc("--coschedule-06g without --allow-shared-gpu makes a busy GPU acceptable for a central MAIN run")
def test_nc_coschedule_alone_accepts_busy_gpu(tmp_path):
    try:
        L.preflight(_targs(tmp_path, *MAIN, "--coschedule-06g", smoke=False), runner=_busy(), environ={},
                    gate_path=gate(tmp_path))
        accepted = True
    except L.LaunchRefused:
        accepted = False
    assert accepted, "refused (as it must be)"
