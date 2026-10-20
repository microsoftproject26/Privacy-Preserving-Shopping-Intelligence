"""FL-07 — v4int.launch_fl after PREG1_AMENDMENT_06C (CPU, the tiny FL world of test_int03_launch_fl):

  * FA_1024 / S_CAL / DP arm specs use the Poisson plan {group_size 1024, sampling poisson} (DP denominator 1024);
    their arm-spec digests change (old -> new pinned here), their fl RunConfig digests do not; FA / FP / PF / FA_Q8 /
    the CALIBRATION screen (all seeds, fused modes and LR multipliers) keep the pre-FL-07 digests byte for byte;
  * the engine end to end on Poisson: S_CAL -> S_RECORD -> FA_1024 / DP (same cohorts, variable sizes, 12th mark at the
    end), a non-Poisson DP spec refused by fl.runtime, the smoke first-evaluation bound on the round-index units;
  * item 5a: a non-smoke MAIN run of any arm (S_CAL included) is refused without --lr-selection;
  * item 5b: FA_Q8 upload bytes carry the same 8-byte n_consumed header as FA.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import v4int.launch_fl as F
from intkit import assert_raises, nc
from test_fl05_launch import BASELINE, PEAK_REG, _combos
from test_int03_launch_fl import _run, _tiny_spec

from fl.synthetic import make_tiny_adapter

OLD_SPEC = {   # pre-FL-07 arm-spec sha256 (uniform plan) of the arms whose plan changed; see test_fl05_launch.BASELINE
    "DP|2026|None|0.5642": "a3deaf34fdcac1e89e240c028398441bdec87bd7a2626055544a2fa035d8a687",
    "DP|2026|None|1.3094": "8e53895cc31a27391af3cab163abcca78c4ba252baea6b8360a82be73962c526",
    "DP|2026|True|0.5642": "12eb9b592f49bf212c510a26ed7e8bb0ca7474e3b5bda065eb90889d05dd415b",
    "DP|2026|True|1.3094": "baacf3b96a3f8010a26a78c22de14eb51f09be11aa554f7d6a610e81a521dff7",
    "DP|2027|None|0.5642": "af3102d75d5099f2be8db02d436cf541961691b3a8abe881cf5159a1ff97c4ac",
    "DP|2027|None|1.3094": "4ff6b23e82b8a33652d715aa3fa578cc2a23e6ccb861d91358b3d58957a50628",
    "DP|2027|True|0.5642": "2232a78fa386f08d1a3168d9120999f8d6442276d930d5e9107546805be49f7c",
    "DP|2027|True|1.3094": "15c1ca2b86de426845e71d619927ac9068cf7f932faddca0a4594af9ec824026",
    "FA_1024|2026|None|None": "364f12d78da97ad53e16f4fe5906d9fd6185803083e0b0cd65cd780cd6610bcb",
    "FA_1024|2026|True|None": "d571ca47a8471db4a38528bc56b38464c6db6074cc2e7d7e5b9467b5d945eda3",
    "FA_1024|2027|None|None": "b3a9547f3caa3c43580b66081f7f133a2b98a1b85f80c7d42bab1b6b2e77e1e6",
    "FA_1024|2027|True|None": "3524ea0fbd55e65ffa84ea9ef7464cdc69184053293d42e864fc802acee1de00",
    "S_CAL|2026|None|None": "b4952be822d1643252d9f2994ee95babf70b4a5cc058e9e960624f67eb1ffbd0",
    "S_CAL|2026|True|None": "08eae95b979746727b17bd84f7ec9ea63564ba2bea72f662385471833f6c0af8",
}
# sha256 of the sorted JSON {tag: [spec sha256, fl RunConfig digest]} of FA / FP / PF / FA_Q8 (seeds 2026/2027, fused
# True/None, lr_mult 1/0.5/2) and the FA CALIBRATION screen (fused True/None, lr_mult 0.5/1/2): 54 pairs computed from
# the PRE-FL-07 code (launch_fl.py sha256 4614375511aa5cbde4f293c5c5ca86463ecee46e06a76fe7eb874e107c2b0dd7).
UNCHANGED_SHA = "ac0c1f257c71627175200c1d207c63981478a00a29bc3691a75e66a2c640c03e"
POISSON_PLAN = {"group_size": 1024, "sampling": "poisson"}


def unchanged_digests() -> dict:
    out = {}
    for arm in ("FA", "FP", "PF", "FA_Q8"):
        for seed in (2026, 2027):
            for fused in (True, None):
                for m in (1.0, 0.5, 2.0):
                    s = F.arm_spec(arm, seed=seed, peak_lr=PEAK_REG, fused=fused, lr_mult=m)
                    out[f"{arm}|{seed}|{fused}|None|{m}"] = [F.spec_digest(s),
                                                             F.fl_run_config(s, f"FL_{arm}_SMALL_s{seed}", 64).digest()]
    for m in (0.5, 1.0, 2.0):
        for fused in (True, None):
            s = F.arm_spec("FA", seed=2026, peak_lr=PEAK_REG, fused=fused, lr_mult=m, role="CALIBRATION")
            out[f"FA_CALIB|2026|{fused}|None|{m}"] = [F.spec_digest(s), F.fl_run_config(s, "FL_FA_LRSCREEN", 64).digest()]
    return out


# ------------------------------------------------------------------------------------------------ specs / digests
def test_dp_arm_specs_are_poisson_with_fixed_denominator():
    for arm, kw in (("FA_1024", {"clip_norm": 1.0}), ("DP", {"clip_norm": 1.0, "z": 0.5642}), ("S_CAL", {})):
        s = F.arm_spec(arm, seed=2026, peak_lr=PEAK_REG, **kw)
        assert s["plan"] == POISSON_PLAN and s["cfg"]["dp"]["denominator"] == s["plan"]["group_size"] == 1024
        assert s["cfg"]["dropout_p"] == 0.0 and s["cfg"]["endpoint_rounds"] == 384
    for arm in ("FA", "FP", "PF", "FA_Q8"):
        assert F.arm_spec(arm, seed=2026, peak_lr=PEAK_REG)["plan"] == {"group_size": 64, "sampling": "sweep"}


def test_changed_digests_old_to_new_and_unchanged_arms_byte_identical():
    changed = 0
    for tag, arm, seed, fused, kw in _combos():
        s = F.arm_spec(arm, seed=seed, peak_lr=PEAK_REG, fused=fused, **kw)
        got = (F.spec_digest(s), F.fl_run_config(s, f"FL_{arm}_SMALL_s{seed}", 64).digest())
        assert got == BASELINE[tag], tag                       # FA / FP / PF: the FL-05 values; others: re-pinned
        if tag in OLD_SPEC:
            assert got[0] != OLD_SPEC[tag]
            changed += 1
    assert changed == len(OLD_SPEC) == 14
    body = json.dumps(unchanged_digests(), sort_keys=True).encode()
    assert hashlib.sha256(body).hexdigest() == UNCHANGED_SHA


# ------------------------------------------------------------------------------------------------ the engine
def _poisson_tiny(arm, rounds=24, group=4, **kw):
    """A tiny Poisson spec: group_size = the DP denominator = `group` on the 10-client world (q = 0.4)."""
    s = _tiny_spec(arm, rounds=rounds, **kw)
    assert s["plan"]["sampling"] == "poisson"
    s["plan"]["group_size"] = group
    s["cfg"]["dp"]["denominator"] = group
    return s


def test_s_cal_fa1024_dp_end_to_end_on_poisson(tmp_path):
    out, _ = _run(tmp_path, _poisson_tiny("S_CAL"), "cal")
    assert out["status"] == "DONE_S_CAL"
    rec = F.load_s_record(tmp_path / "cal" / "S_RECORD.json", smoke=False)
    assert rec["rounds_0based"] == list(range(20)) and rec["S"] > 0
    logs = {}
    for arm, z in (("FA_1024", None), ("DP", 0.5642)):
        o, _ = _run(tmp_path, _poisson_tiny(arm, clip_norm=rec["S"], z=z), arm)
        s = json.loads((tmp_path / arm / "RUN_SUMMARY.json").read_text(encoding="utf-8"))
        assert o["status"] == "DONE_MAIN" and s["evals"][-1][0] == 12
        part = s["participation"]
        assert part["sampling"] == "poisson" and part["inclusion_probability"] == 0.4
        assert part["cohort_size_min"] < part["cohort_size_max"] and s["dp"]["denominator"] == 4
        logs[arm] = {k: part[k] for k in ("sampled", "cohort_size_min", "cohort_size_max", "empty_rounds")}
        logs[arm]["lr"] = s["lr_table"]["digest"]
    assert logs["FA_1024"] == logs["DP"]                     # FA_1024 -> DP: same cohorts and LR table, noise only


def test_expected_exposure_total_in_the_summary(tmp_path):
    spec = _poisson_tiny("FA_1024", clip_norm=0.5)
    o, w = _run(tmp_path, spec, "x")
    t = o["summary"]["lr_table"]
    V = sum(w["n_valid_by_key"].values())
    assert t["E_total"] == 24 * 4 * 2 * V / 10 and t["planned_efe"] == t["E_total"] / w["n_decisions"]
    assert o["summary"]["participation"]["E_realised_total"] == o["summary"]["counters"]["exposures"]


def test_non_poisson_dp_spec_refused_by_the_runtime(tmp_path):
    for sampling in ("uniform", "sweep"):
        spec = _poisson_tiny("DP", clip_norm=0.5, z=0.5642)
        spec["plan"]["sampling"] = sampling
        assert_raises(ValueError, _run, tmp_path, spec, f"u_{sampling}")


def test_smoke_first_evaluation_bound_uses_round_units(tmp_path):
    spec = _poisson_tiny("FA_1024", clip_norm=0.5)                       # T 24: units after row 1 = 1
    out, _ = _run(tmp_path, spec, "sm", smoke_rounds=1)
    assert out["status"] == "SMOKE_DONE_NOT_EVIDENCE"
    assert_raises(F.LaunchRefused, _run, tmp_path, _poisson_tiny("FA_1024", clip_norm=0.5), "sm2", smoke_rounds=2)


# ------------------------------------------------------------------------------------------------ 5a / 5b
def _args(tmp_path, *extra, smoke=False, arm="FA", seed="2026"):
    a = ["--arm", arm, "--seed", seed, "--gpu", "0", "--run-dir", str(tmp_path / "r"),
         "--history-policy", "USER_HISTORY_L50"] + list(extra)
    if smoke:
        a.append("--smoke")
    return F.parse_args(a)


ARM_EXTRA = {"FA": (), "FP": (), "PF": (), "FA_Q8": (), "S_CAL": (), "FA_1024": ("--clip-record", "s.json"),
             "DP": ("--clip-record", "s.json", "--epsilon", "8")}


def test_non_smoke_main_run_requires_lr_selection(tmp_path, monkeypatch):
    for name in ("LR_SELECTION_EXT_PATH", "FL_TUNING_PATH", "FIRST_SCREEN_PATH"):   # nothing registered
        monkeypatch.setattr(F, name, tmp_path / "registered" / getattr(F, name).name)
    for arm, extra in ARM_EXTRA.items():
        a = _args(tmp_path, *extra, arm=arm)
        assert_raises(F.LaunchRefused, F.run, a, {}, {})                  # refused before any import / data / GPU
        assert_raises(F.LaunchRefused, F.require_lr_selection, _args(tmp_path, *extra, "--lr-mult", "1", arm=arm))
        # FIX-DELTA-01 F2: an unregistered (06B / any) --lr-selection no longer satisfies a non-smoke MAIN run
        assert_raises(F.LaunchRefused, F.require_lr_selection,
                      _args(tmp_path, *extra, "--lr-selection", "LR_SELECTION.json", arm=arm))
        F.require_lr_selection(_args(tmp_path, *extra, arm=arm, smoke=True))
    F.require_lr_selection(_args(tmp_path, "--role", "CALIBRATION", "--lr-mult", "2"))   # the screen itself


def test_fa_q8_upload_bytes_carry_the_fa_header():
    from fl.comm import UPLOAD_HEADER_BYTES, visit_bytes
    from fl.quant import q8_payload_bytes, q8_upload_bytes, q8_visit_bytes
    man = make_tiny_adapter(seed=1).manifest
    q8, fa = q8_visit_bytes(man), visit_bytes(man)
    assert UPLOAD_HEADER_BYTES == 8 and q8["header_bytes"] == 8
    assert q8["upload_bytes"] == q8_upload_bytes(man) == q8_payload_bytes(man) + 8
    assert fa["upload_bytes"] - q8["upload_bytes"] == man.shared_bytes - q8_payload_bytes(man)
    assert F.Q8_CODEC["upload_bytes"] == "sum(numel) + 4 x n_tensors"      # the codec payload (spec digest unchanged)


# ------------------------------------------------------------------------------------------------ negative controls
@nc("a check on --lr-mult instead of --lr-selection lets a non-smoke MAIN run through at a hand-picked multiplier")
def test_nc_lr_mult_satisfies_the_requirement(tmp_path):
    def mutant(args):                                                    # MUTANT: any LR option counts
        if not args.smoke and args.role == "MAIN" and args.lr_selection is None and args.lr_mult is None:
            raise F.LaunchRefused("no LR option")
    assert_raises(F.LaunchRefused, mutant, _args(tmp_path, "--lr-mult", "1"))


@nc("the pre-FL-07 (uniform-plan) DP spec is not the registered spec any more")
def test_nc_old_uniform_spec_still_registered():
    s = F.arm_spec("DP", seed=2026, peak_lr=PEAK_REG, fused=True, clip_norm=1.2345, z=0.5642)
    s["plan"] = {"group_size": 1024, "sampling": "uniform"}                # MUTANT: the old plan
    assert F.spec_digest(s) == BASELINE["DP|2026|True|0.5642"][0]
