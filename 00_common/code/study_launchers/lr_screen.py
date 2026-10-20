"""v4int.lr_screen — the federated local-LR selection of PREG1_AMENDMENT_06B §1 (task FL-05 item 3).

  python -B code/v4int/lr_screen.py select --runs <run dir m=0.5> <run dir m=1> <run dir m=2> --out <LR_SELECTION.json>

Input: the three finished CALIBRATION-role FA runs of v4int.launch_fl (--role CALIBRATION --lr-mult 0.5 | 1 | 2, seed
2026). Each run is verified: RUN_SUMMARY.json status DONE_MAIN with 12 evaluations ending at mark 6.0; its arm spec is
FA / CALIBRATION / seed 2026 and hashes (launch_fl.spec_digest) to the recorded arm_spec_sha256; ARM_SPEC.json agrees;
the three specs are identical apart from the multiplier (lr_mult, lr_peak_local = m x lr_peak_base, solver LR) and
cover exactly {0.5, 1, 2}; RESULT_ROWS.json holds exactly one PRACTICAL_BEST row of E2E / macro / mrr@20 (the
CALIB_VALIDATION client-macro MRR@20) with role CALIBRATION, run_label FL_LR_SCREEN, method FA, the run's spec sha256
and fl_lr_mult = m, and its value equals the run's own evaluation at its PRACTICAL_BEST mark; the three runs share one
evaluation manifest.

Value used (PRACTICAL_BEST): the launcher's registered PRACTICAL_BEST row = the strictly-best of the 12 marks (running
best, improvement > 0.0002, fl.runtime); the plain maximum of the 12 marks is recorded beside it (`max_of_12_marks`)
and can exceed it by at most 0.0002 (below the 0.0005 tie tolerance).
Rule (06B §1): the best value wins; every m within |delta| <= 0.0005 of the best is tied, and the tied m closest to 1
wins (order 1, 0.5, 2).
Output (O_EXCL, never overwritten): LR_SELECTION.json {format PPSI_V4_FL_LR_SELECTION_1, the three values, the selected
multiplier, the tie set, the run dirs with their RESULT_ROWS / RUN_SUMMARY sha256, self_sha256}. Only the selected
multiplier and the file's sha256 are printed. launch_fl --lr-selection verifies the file and re-applies the rule.

FL-09 (PREG1_AMENDMENT_06F §3) — the extended screen (the CLI `select` now writes this format):
  python -B code/v4int/lr_screen.py select --runs <dir> <dir> <dir> [<dir> ...] [--first-screen <LR_SELECTION.json>] \
         --out <LR_SELECTION_EXT.json>
  >= 3 CALIBRATION FA runs, any registered subset of launch_fl.LR_MULTS (0.5 .. 32), each multiplier at most once, each
  verified exactly as above (FA / CALIBRATION / 2026 / DONE_MAIN / 12 marks / one PRACTICAL_BEST row = the run's own
  evaluation / specs identical apart from the multiplier / one evaluation manifest). A run whose PRACTICAL_BEST value is
  non-finite (or missing) is EXCLUDED and recorded (never selected). With --first-screen, the 06B file must verify and
  every one of its runs must be among --runs with the same RESULT_ROWS sha256 and value (a missing screened run is
  refused). Rule: the best value over the screened set; ties |delta| <= 0.0005 -> the tied m with the smallest |log2 m|,
  then the smaller m (order 1, 0.5, 2, 4, 8, 16, 32 — the 06B order extended). Output O_EXCL, format
  PPSI_V4_FL_LR_SELECTION_EXT_1: values (every finite screened m), screened, excluded, selected_lr_mult, tie_set,
  first_screen_file_sha256, the runs with their RESULT_ROWS / RUN_SUMMARY sha256, an informational edge record,
  self_sha256. Only the selected multiplier and the file's sha256 are printed.

FL-10 (PREG1_AMENDMENT_06G §3 as tightened by 06J §1-§2 and 06I §3) — the staged FL tuning selection:
  python -B code/v4int/lr_screen.py select-joint --stage S2 --runs <S1b winner run> <S2 runs> \
         --first-selection contracts/fl/LR_SELECTION_EXT.json --out <S2 file>
  python -B code/v4int/lr_screen.py select-joint --stage S3 --runs <S2 winner run> <S3 runs> \
         --first-selection <S2 file> --out <S3 file>
  Every run verified as above plus its tuning point (launch_fl.spec_tuning: server optimizer / eta_s / tau, m, LR shape,
  client optimizer / client LR, each recorded in the spec, cfg and rows exactly as the launcher writes them).
  S2 (06G §2 + 06J §1): the incumbent = the S1b winner run of the verified EXT file (same RESULT_ROWS sha256 / value);
  REQUIRED FedAdam eta_s in {0.01, 0.1} x m in {1, m*} (m* = 1 -> {1, 4}) and FedAvgM (beta 0.9, eta 1) at m*; OPTIONAL
  (only when the registered edge condition holds) FedAdam eta_s {0.003, 0.001} at the best base FedAdam run's m when its
  eta_s is 0.01 (both runs), and one FedAdam tau = 1e-5 run at the (eta_s, m) of a FedAdam S2 winner. Ties: FedAvg
  (incumbent) -> FedAvgM -> FedAdam, then smallest |log2 m|, then smaller eta_s, then larger tau.
  S3 (06I §3 + 06J §2): the incumbent = the S2 winner run (from the verified S2 file); REQUIRED AdamW-fresh x FL_CONST at
  the S2 winner's point and client SGD-m0.9 x FL_CONST at lr {0.01, 0.1, 1.0} (m = 1, the S2 server optimizer); OPTIONAL
  SGD lr {3, 10} if lr 1.0 is the raw best SGD run (both), lr 0.003 if 0.01 is. Ties: incumbent, then AdamW before SGD,
  then the smaller SGD lr.
  A missing required run, an unregistered or disallowed configuration, a configuration twice, runs that differ beyond the
  tuning dimensions, or two evaluation manifests are refused; a non-finite value is excluded and recorded (06J §4 v).
  Output O_EXCL, format PPSI_V4_FL_TUNING_SELECTION_1: stage, every configuration (tuning point, value, incumbent flag,
  RESULT_ROWS / RUN_SUMMARY sha256), values, excluded, tie set, winner, m*, the stage history (EXT -> S2 -> S3 with
  sha256), self_sha256. Prints only the winner + sha256. The file Ops writes as contracts/fl/FL_TUNING_SELECTION.json
  is the last stage's (S4). `select` (06B / 06F) refuses any run with a non-default tuning point.

FL-13 (PREG1_AMENDMENT_06J §3) — stage S4, local computation:
  python -B code/v4int/lr_screen.py select-joint --stage S4 --runs <S3 winner run> <3 S4 runs> \
         --first-selection <S3 file> --out <S4 file>
  The incumbent = the S3 winner run (from the verified S3 file); REQUIRED, one at a time at the S3 winner's point:
  local passes 1, local passes 4, local batch 8 (no edge runs, no LR re-screen). Ties |delta| <= 0.0005 -> the incumbent
  (the default 2 passes / batch 16), then passes closest to 2 (|log2(P / 2)|: batch 8 before passes 1 / 4), then
  batch 16, then fewer passes (passes 1 before 4).
  Each run's round count must be the 06J §3 recomputed 6.0-EFE budget (launch_fl.spec_tuning).

FIX-DELTA-02 (REV-DELTA-02 B1 / M3 / m5; amendment 06J §4(ii), §1-§2):
  * 06J §4(ii) "no measurable gain", applied by stage_decision (= apply_rule_stage) at S2 / S3 / S4, in the S2 FedAdam
    tau edge condition and in launch_fl.load_fl_tuning (the same function): after the 0.0005 tie band, the stage's
    best candidate (the highest finite non-incumbent value) replaces the incumbent only if it gains >= MIN_GAIN = 0.002
    over the incumbent's value; otherwise the incumbent carries on (winner = incumbent, tie set = [incumbent]). The
    stage file records gain, best_candidate, no_measurable_gain, min_gain and the raw 0.0005 band (tie_band_set).
  * Registered edge runs are REQUIRED: when a 06J edge condition holds and none of its runs is given, select-joint
    refuses (EdgeRunsMissing) and names the runs to launch (label + launch_fl flags); the S2 tau condition is evaluated
    only once a required eta edge is complete. load_fl_tuning re-checks required / edge runs from the file's configs.
  * The runs of select-joint / select-warm must share one ARM_SPEC.json `device` record (evaluation device class and
    pin); the file records it.
  * The S4 file notes the DL 219 exception in `applies_to_exception` (FA_1024 / DP / S_CAL keep 2 local passes and
    batch 16; launch_fl applies it).

FL-14 (PREG1_AMENDMENT_06L A1 / A2 as read by 06L_D D1-D3; no registered value, threshold or grid changed):
  * S2 REQUIRED = the 06J §1 set + FedAvgM (beta 0.9, eta 1) at m in {1, 2} (a cell equal to FedAvgM at m* is one
    run). A1 edge: if the raw-best required FedAvgM cell is m = 1, FedAvgM m = 0.5 is REQUIRED (EdgeRunsMissing;
    no group when m = 0.5 is already required, m* = 0.5). Edge order: 06J eta edge -> A1 FedAvgM edge -> 06J tau
    edge; the tau condition is evaluated once every condition-true edge is complete, on the S2 winner (06J §4(ii))
    over the incumbent + required + those edge runs.
  * S3 REQUIRED = the 06J §2 cells on the S2 winner + the two A2 cells (FedAdam eta_s 0.01, tau 1e-3, v0 = tau^2;
    m 1; FL_CONST; client SGD-m 0.9 at lr {0.1, 1.0}); an A2 cell equal to a 06J cell is one run. The 06J SGD edge
    reads only the 06J SGD cells (the S2 winner's server optimizer); A2 cells never trigger edges. Ties: incumbent,
    AdamW before SGD, the smaller SGD lr, then FedAvg family (FedAvg / FedAvgM) before FedAdam.
  * S4 unchanged (built on any S3 winner, an A2 cell included). launch_fl.load_fl_tuning re-checks D1 / D2 through
    stage_required / check_stage_runs.

FL-12 (PREG1_AMENDMENT_06K §2-§3) — the warm-start LR screens S7 / S8:
  python -B code/v4int/lr_screen.py select-warm --stage S7|S8 --runs <factor-1 run> <other run> \
         --tuning contracts/fl/FL_TUNING_SELECTION.json --out <FL_WARM_SELECTION_S7|S8.json>
  Two CALIBRATION runs of FA_WARM (S7, factors {1, 1/4} = m*, m*/4) or FA_WARM_FROZEN (S8, {1, 4} = m*, 4 m*), each at
  the tuning file's winner point, from one theta_pre (same init_from identity), otherwise identical. Rule: the best
  value; ties |delta| <= 0.0005 -> factor 1 (m*); a non-finite value is excluded and recorded. Output O_EXCL, format
  PPSI_V4_FL_WARM_SELECTION_1 (values, screened, excluded, winner_factor, tie_set, runs, tuning_selection sha, init_from,
  self_sha256); Ops copies it to contracts/fl/FL_WARM_SELECTION_<stage>.json for the MAIN warm runs.

FL-15 (PREG1_AMENDMENT_06M_B B2, amending 06K §2-§3) — the registered S7 / S8 screens are 2-D; the 2-run screen above
is superseded (its PPSI_V4_FL_WARM_SELECTION_1 file never configures a run):
  python -B code/v4int/lr_screen.py select-warm --stage S7|S8 --runs <4 run dirs> \
         --tuning contracts/fl/FL_TUNING_SELECTION.json --out <FL_WARM_SELECTION_S7|S8.json>
  4 CALIBRATION runs at the tuning file's winner point from one theta_pre: local factor (S7 {1, 1/4}; S8 {1, 4}) x
  server-LR factor (launch_fl --warm-server-lr-factor: FedAdam {1, 0.1} = {eta*, eta*/10}; FedAvgM / FedAvg {1, 0.25}
  = {eta*, eta*/4}), otherwise identical. Rule (warm_decision) = 06J restricted to these runs: the 0.0005 band, then
  06J §4(ii) with the incumbent (m*, eta*) = (1, 1): the best candidate replaces it only if it gains >= 0.002, else no
  measurable gain and (1, 1) carries on; ties -> (1, 1), then fewer changed axes, then the smaller |log2| local factor,
  then server factor; a non-finite run is excluded and recorded; an excluded incumbent refuses (Ops decision). Output
  O_EXCL, format PPSI_V4_FL_WARM_SELECTION_2 (values / screened / excluded by cell label 'f=<local>|sf=<server>',
  winner {warm_lr_factor, warm_server_lr_factor, label, server_lr}, winner_factor, winner_server_lr_factor, tie_set,
  tie_band_set, best_candidate, gain, no_measurable_gain, min_gain, incumbent, server_axis {optimizer, eta_star,
  factors, server_lrs}, tuning_selection sha, init_from, runs, self_sha256). launch_fl verifies it
  (load_warm_selection_2d) and a MAIN warm run takes BOTH factors from the registered copy (bind_warm_selection).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Callable, Mapping, Optional, Sequence

HERE = Path(__file__).resolve().parent
CODE = HERE.parent
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))
sys.dont_write_bytecode = True

TIE = 0.0005
ORDER = (1.0, 0.5, 2.0)                                   # closest to 1 first (06B §1)
EVAL_MARKS = 12
RULE_TEXT = ("select the multiplier with the highest CALIB_VALIDATION client-macro MRR@20 (E2E) at PRACTICAL_BEST; "
             "every multiplier within |delta| <= 0.0005 of the best is tied; the tied multiplier closest to 1 wins "
             "(order 1, 0.5, 2) — PREG1_AMENDMENT_06B §1")


RULE_TEXT_EXT = ("select the multiplier with the highest CALIB_VALIDATION client-macro MRR@20 (E2E) at PRACTICAL_BEST "
                 "over every screened multiplier (runs with a non-finite value excluded and recorded); every multiplier "
                 "within |delta| <= 0.0005 of the best is tied; the tied multiplier with the smallest |log2 m| wins, then "
                 "the smaller m — PREG1_AMENDMENT_06F §3 (extends 06B §1)")
MIN_EXT_RUNS = 3
MIN_GAIN = 0.002                                          # 06J §4(ii): "no measurable gain" below this gain
MIN_GAIN_TEXT = ("06J §4(ii): if the best candidate (the highest finite non-incumbent value) gains < 0.002 over the "
                 "incumbent, the stage is recorded 'no measurable gain' and the incumbent carries on (winner = "
                 "incumbent, tie set = [incumbent])")
POISSON_EXCEPTION_TEXT = ("DECISION_LOG row 219 (registered before any S4 read): FA_1024 / DP / S_CAL take this "
                          "file's optimizer / LR / shape fields (optimizer, server_lr, server_tau, lr_mult, lr_shape, "
                          "client_opt, client_lr) with local passes 2 and local batch 16 whatever S4 selects (their "
                          "T = 384 and z values are registered); launch_fl records poisson_local_rule")
STAGES = ("S2", "S3", "S4")
PREV_STAGE = {"S2": "S1b", "S3": "S2", "S4": "S3"}
FEDAVGM_A1_MULTS = (1.0, 2.0)                             # 06L A1 / 06L_D D1: FedAvgM also at m in {1, 2}
FEDAVGM_A1_EDGE = 0.5                                     # 06L A1 edge: FedAvgM m = 0.5 if m = 1 is the raw best
A2_CLIENT_LRS = (0.1, 1.0)                                # 06L A2 / 06L_D D2: the two S3 combination cells
RULE_TEXT_STAGE = {
    "S2": ("S2 winner: the highest CALIB_VALIDATION E2E client-macro MRR@20 at PRACTICAL_BEST over the S2 set (FedAdam "
           "eta_s {0.01, 0.1} x m {1, m*}, FedAvgM at m* and at m {1, 2} (06L A1), the registered edge runs: 06J eta "
           "edge -> 06L A1 FedAvgM m 0.5 edge -> 06J tau edge) and the S1b winner (the incumbent); "
           "non-finite values excluded and recorded; |delta| <= 0.0005 ties -> FedAvg (incumbent) -> FedAvgM -> FedAdam, "
           "then smallest |log2 m|, then smaller eta_s, then larger tau — PREG1_AMENDMENT_06G §3 / 06J §1 / 06L A1 "
           "(06L_D D1)"),
    "S3": ("S3 winner: the highest CALIB_VALIDATION E2E client-macro MRR@20 at PRACTICAL_BEST over the S3 set "
           "(AdamW-fresh x FL_CONST, SGD-m0.9 x FL_CONST lr {0.01, 0.1, 1.0} on the S2 winner, the registered 06J SGD "
           "edge runs, and the 06L A2 cells FedAdam eta_s 0.01 tau 1e-3 x m 1 x FL_CONST x SGD-m0.9 lr {0.1, 1.0}) and "
           "the S2 winner (the incumbent); non-finite values excluded and recorded; |delta| <= 0.0005 ties -> incumbent, "
           "then AdamW before SGD, then the smaller SGD lr, then FedAvg family before FedAdam — PREG1_AMENDMENT_06I §3 "
           "/ 06J §2 / 06L A2 (06L_D D2)"),
    "S4": ("S4 winner: the highest CALIB_VALIDATION E2E client-macro MRR@20 at PRACTICAL_BEST over the S4 set (local "
           "passes 1, local passes 4, local batch 8, one at a time at the S3 winner; rounds recomputed to the 6.0-EFE "
           "budget; no LR re-screen) and the S3 winner (the incumbent); non-finite values excluded and recorded; "
           "|delta| <= 0.0005 ties -> the incumbent (2 passes / batch 16) — PREG1_AMENDMENT_06J §3")}
RULE_TEXT_STAGE = {k: f"{v}; {MIN_GAIN_TEXT}" for k, v in RULE_TEXT_STAGE.items()}      # FIX-DELTA-02 B1
TIE_ORDER_TEXT = {"S2": "FedAvg (incumbent) -> FedAvgM -> FedAdam, then smallest |log2 m|, then smaller eta_s, then larger tau",
                  "S3": "incumbent, then AdamW before SGD, then the smaller SGD lr, then FedAvg family before FedAdam",
                  "S4": "incumbent (default), then |log2(passes / 2)| (batch 8 keeps 2 passes), then batch 16 before 8, "
                        "then fewer passes"}
SERVER_RANK = {"fedavg": 0, "fedavgm": 1, "fedadam": 2}


class SelectionRefused(RuntimeError):
    pass


class EdgeRunsMissing(SelectionRefused):
    """FIX-DELTA-02 M3: a registered 06J edge condition holds and its runs are absent; `needed` = the tuning points
    (dicts with TUNING_KEYS + label) to launch before the stage file can be written."""

    def __init__(self, msg: str, needed: Sequence[Mapping]):
        super().__init__(msg)
        self.needed = [dict(t) for t in needed]


def ext_order_key(m: float) -> tuple:
    """06F §3 tie order: smallest |log2 m|, then the smaller m (1, 0.5, 2, 4, 8, 16, 32)."""
    return (abs(math.log2(float(m))), float(m))


def apply_rule_ext(values: Mapping[float, float]) -> tuple:
    """(selected multiplier, tie set in the 06F §3 order) over any non-empty set of registered multipliers."""
    import v4int.launch_fl as F
    if not values:
        raise SelectionRefused("no screened multiplier with a finite value")
    vals = {}
    for m, v in values.items():
        m, v = float(m), float(v)
        if m not in F.LR_MULTS:
            raise SelectionRefused(f"multiplier {m} is not registered ({F.LR_MULTS})")
        if not math.isfinite(v):
            raise SelectionRefused(f"multiplier {m}: non-finite value {v!r} (it must be excluded, not ranked)")
        vals[m] = v
    best = max(vals.values())
    tie = sorted((m for m in vals if best - vals[m] <= TIE), key=ext_order_key)
    return tie[0], tie


def stage_order_key(stage: str, c: Mapping) -> tuple:
    """06J tie orders (S2: simpler server optimizer first; S3: incumbent, AdamW, smaller SGD lr, then FedAvg family
    before FedAdam (06L A2)); the label is the final deterministic key (two distinct registered points never share it)."""
    if stage == "S2":
        return (SERVER_RANK[c["optimizer"]], abs(math.log2(float(c["lr_mult"]))),
                0.0 if c["optimizer"] != "fedadam" else float(c["server_lr"]),
                0.0 if c["server_tau"] is None else -float(c["server_tau"]), float(c["lr_mult"]), c["label"])
    if stage == "S4":
        lp, lb = int(c.get("local_passes") or 2), int(c.get("local_batch") or 16)
        return (0 if c["incumbent"] else 1, abs(math.log2(lp / 2.0)), 0 if lb == 16 else 1, lp, c["label"])
    return (0 if c["incumbent"] else 1, 0 if c["client_opt"] == "adamw_fresh" else 1,
            0.0 if c["client_lr"] is None else float(c["client_lr"]),
            0 if c["optimizer"] in ("fedavg", "fedavgm") else 1, c["label"])      # FL-14 (06L A2): FedAvg family first


def apply_rule_stage(stage: str, configs: Sequence[Mapping]) -> tuple:
    """(winner tuning point, tie set of labels in the stage order) over the finite configurations of a stage; exactly
    one configuration carries incumbent = True. FIX-DELTA-02 B1: 06J §4(ii) applied (stage_decision)."""
    d = stage_decision(stage, configs)
    return d["winner"], d["tie_set"]


def stage_decision(stage: str, configs: Sequence[Mapping]) -> dict:
    """The full stage decision over the finite configurations (exactly one incumbent): the 0.0005 tie band in the stage
    order (tie_band_set), then 06J §4(ii): the best candidate (highest finite non-incumbent value; its label, gain =
    its value - the incumbent's) replaces the incumbent only if gain >= MIN_GAIN; else no_measurable_gain, winner =
    the incumbent and tie set = [incumbent] (also when no finite candidate exists: gain None)."""
    import v4int.launch_fl as F
    if stage not in STAGES:
        raise SelectionRefused(f"stage {stage!r} is not one of {STAGES}")
    if not configs:
        raise SelectionRefused("no screened configuration with a finite value")
    norm, seen = [], set()
    for c in configs:
        try:
            t = F.tuning_config(**{k: c.get(k) for k in F.TUNING_KEYS})
        except F.LaunchRefused as e:
            raise SelectionRefused(f"unregistered configuration {dict(c)}: {e}") from e
        v = float(c["value"])
        lab = F.tuning_label(t)
        if not math.isfinite(v):
            raise SelectionRefused(f"{lab}: non-finite value {v!r} (excluded, not ranked)")
        if lab in seen:
            raise SelectionRefused(f"configuration {lab} given twice")
        seen.add(lab)
        norm.append(dict(t, value=v, label=lab, incumbent=bool(c.get("incumbent"))))
    n_inc = sum(c["incumbent"] for c in norm)
    if n_inc != 1:
        raise SelectionRefused(f"{n_inc} incumbents among the finite configurations (exactly one is required: 06J "
                               "§4(ii) measures the gain over the incumbent)")
    best = max(c["value"] for c in norm)
    band = sorted((c for c in norm if best - c["value"] <= TIE), key=lambda c: stage_order_key(stage, c))
    inc = next(c for c in norm if c["incumbent"])
    cands = [c for c in norm if not c["incumbent"]]
    top = min(cands, key=lambda c: (-c["value"], stage_order_key(stage, c))) if cands else None
    gain = None if top is None else top["value"] - inc["value"]
    no_gain = gain is None or gain < MIN_GAIN
    tie = [inc] if no_gain else band
    return {"winner": {k: tie[0][k] for k in F.TUNING_KEYS}, "tie_set": [c["label"] for c in tie],
            "tie_band_set": [c["label"] for c in band], "best_candidate": None if top is None else top["label"],
            "gain": gain, "no_measurable_gain": bool(no_gain), "min_gain": MIN_GAIN,
            "incumbent": inc["label"]}


def apply_rule(values: Mapping[float, float]) -> tuple:
    """(selected multiplier, tie set in ORDER) for {m: value} over exactly the multipliers of ORDER."""
    if sorted(float(m) for m in values) != sorted(ORDER):
        raise SelectionRefused(f"values must be given for exactly the multipliers {sorted(ORDER)}")
    vals = {float(m): float(v) for m, v in values.items()}
    best = max(vals.values())
    tie = [m for m in ORDER if best - vals[m] <= TIE]
    return tie[0], tie


def _sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _strip_lr(spec: Mapping) -> dict:
    import v4int.launch_fl as F
    d = json.loads(json.dumps(spec))
    for k in ("lr_mult", "server_opt", "server_lr", "server_tau", "lr_shape", "client_opt", "client_lr",   # FL-10
              "local_passes", "local_batch", "warm_lr_factor",                                          # FL-12/13
              "warm_server_lr_factor"):                                                                  # FL-15
        d.pop(k, None)
    eb = d.pop("exposure_budget", None)                    # FL-13: a non-default pass count compares at the
    if eb is not None:                                     # default-pass round count of its registered N
        d["cfg"]["endpoint_rounds"] = d["max_rounds"] = F.fa_rounds(eb["n_clients"], F.KNOBS["passes"])
    for k in ("server_opt", "lr_shape", "lr_peak_local"):
        d["cfg"].pop(k, None)
    for k in ("lr", "optimizer", "weight_decay", "fused", "passes", "batch_size"):   # the client optimizer / local
        d["cfg"]["solver"].pop(k, None)              # computation are screened; AdamW (wd, fused) checked separately
    return d


def read_run(run_dir: Path, arms: Sequence[str] = ("FA",)) -> dict:
    """One verified CALIBRATION run of one of `arms` (default the cold-start FA screen; see the module docstring)."""
    import v4int.launch_fl as F
    rd = Path(run_dir)
    try:
        summ = json.loads((rd / "RUN_SUMMARY.json").read_text(encoding="utf-8"))
        rr = json.loads((rd / "RESULT_ROWS.json").read_text(encoding="utf-8"))
        side = json.loads((rd / "ARM_SPEC.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise SelectionRefused(f"{rd}: RUN_SUMMARY / RESULT_ROWS / ARM_SPEC unreadable ({e})") from e
    spec = summ.get("arm_spec") or {}
    digest = F.spec_digest(spec)
    if summ.get("status") != "DONE_MAIN":
        raise SelectionRefused(f"{rd}: status {summ.get('status')} is not DONE_MAIN")
    arm = spec.get("arm")
    if arm not in arms or spec.get("role") != "CALIBRATION" or spec.get("cfg", {}).get("seed") != F.CALIB["seed"]:
        raise SelectionRefused(f"{rd}: not an {'/'.join(arms)} / CALIBRATION / seed {F.CALIB['seed']} run")
    if digest != summ.get("arm_spec_sha256") or digest != side.get("arm_spec_sha256") or side.get("arm") != arm:
        raise SelectionRefused(f"{rd}: arm spec digest does not match RUN_SUMMARY / ARM_SPEC.json")
    m = F.lr_mult_value(spec.get("lr_mult"))
    base = float(spec["lr_peak_base"])
    try:                                                   # FL-10: the registered tuning point, cfg-consistent
        t = F.spec_tuning(spec)
        F.check_warm_spec(spec)                            # FL-12: warm keys exactly as arm_spec writes them
    except F.LaunchRefused as e:
        raise SelectionRefused(f"{rd}: {e}") from e
    want_local = t["client_lr"] if t["client_opt"] == "sgd_m" else (base if m == 1.0 else m * base)
    wf = float(spec.get("warm_lr_factor", 1.0))
    if wf != 1.0:
        want_local = wf * want_local
    if spec["cfg"]["lr_peak_local"] != want_local or spec["cfg"]["solver"]["lr"] != spec["cfg"]["lr_peak_local"]:
        raise SelectionRefused(f"{rd}: lr_peak_local / solver LR are not m x lr_peak_base (or the client SGD LR)")
    evals = summ.get("evals") or []
    if [e[0] for e in evals] != list(range(1, EVAL_MARKS + 1)):
        raise SelectionRefused(f"{rd}: expected the 12 evaluations at marks 0.5 .. 6.0")
    hits = [r for r in rr.get("rows", []) if r.get("selection_rule") == "PRACTICAL_BEST"
            and r.get("aggregation") == "macro" and r.get("target_denominator") == "E2E"
            and r.get("metric") == "mrr" and int(r.get("k", -1)) == 20]
    if len(hits) != 1:
        raise SelectionRefused(f"{rd}: {len(hits)} PRACTICAL_BEST E2E macro mrr@20 rows (expected exactly 1)")
    row = hits[0]
    ident = (("role", "CALIBRATION"), ("run_label", F.CALIB["run_label"]), ("method", arm), ("arm_spec_sha256", digest))
    bad = {k: row.get(k) for k, want in ident if row.get(k) != want}
    if bad or float(row.get("fl_lr_mult", -1)) != m:
        raise SelectionRefused(f"{rd}: the PRACTICAL_BEST row is not a verified LR-screen row ({bad})")
    want_rl = dict(F.row_labels(t), **F.warm_row_labels(spec))
    got_rl = {k: row[k] for k in ("fl_server_opt", "fl_server_lr", "fl_server_tau", "fl_lr_shape", "fl_client_opt",
                                  "fl_client_lr", "fl_local_passes", "fl_local_batch", "fl_init", "fl_init_run_id",
                                  "fl_init_checkpoint_sha256", "fl_init_state_digest", "fl_warm_lr_factor",
                                  "fl_frozen_item_tables", "fl_warm_server_lr_factor", "fl_warm_server_lr")
              if k in row}
    if got_rl != want_rl:
        raise SelectionRefused(f"{rd}: the PRACTICAL_BEST row's tuning labels {got_rl} differ from the spec {want_rl}")
    sk = {"optimizer": t["optimizer"], "tuning": t, "arm": arm, "warm_lr_factor": wf,
          "warm_server_lr_factor": float(spec.get("warm_server_lr_factor", 1.0)),        # FL-15 (06M_B B2)
          "device": side.get("device")}                    # FIX-DELTA-02 m5: the evaluation device class / pin
    try:
        value = float(row.get("value"))
    except (TypeError, ValueError):
        value = float("nan")
    if not math.isfinite(value):                           # FL-09 / 06F §3: excluded and recorded, never ranked
        return {"run_dir": str(rd), "run_id": rr.get("run_id"), "lr_mult": m, "lr_peak_base": base, **sk,
                "excluded": True, "reason": f"non-finite PRACTICAL_BEST value {row.get('value')!r} "
                                            f"(value_status {row.get('value_status')!r})",
                "eval_manifest_id": row.get("eval_manifest_id"), "split": row.get("split"),
                "arm_spec_sha256": digest, "result_rows_sha256": _sha(rd / "RESULT_ROWS.json"),
                "run_summary_sha256": _sha(rd / "RUN_SUMMARY.json"), "_spec": spec}
    if row.get("value_status") != "OK":
        raise SelectionRefused(f"{rd}: the PRACTICAL_BEST row has value_status {row.get('value_status')!r}")
    best_u = summ.get("best_half_efe")
    at_best = [e[2] for e in evals if e[0] == best_u]
    if len(at_best) != 1 or float(at_best[0]) != value or abs(float(row["schedule_mark"]) - best_u / 2) > 1e-12:
        raise SelectionRefused(f"{rd}: the PRACTICAL_BEST row does not match the run's evaluation at its best mark")
    return {"run_dir": str(rd), "run_id": rr.get("run_id"), "lr_mult": m, "lr_peak_base": base, **sk, "value": value,
            "practical_best_mark": best_u / 2, "max_of_12_marks": max(float(e[2]) for e in evals),
            "eval_manifest_id": row.get("eval_manifest_id"), "split": row.get("split"),
            "arm_spec_sha256": digest, "result_rows_sha256": _sha(rd / "RESULT_ROWS.json"),
            "run_summary_sha256": _sha(rd / "RUN_SUMMARY.json"), "_spec": spec}


_NOT_DUMPED = ("_spec", "optimizer", "tuning", "arm", "warm_lr_factor", "device",    # 06B / 06F: pre-FL-10 run records
               "warm_server_lr_factor")                                               # FL-15: files unchanged


def _fedavg_only(runs: Sequence[dict]) -> list:
    """FL-10: the 06B / 06F local-LR selections are FedAvg screens; a FedAdam run belongs to select-joint."""
    import v4int.launch_fl as F
    for r in runs:
        if dict(r["tuning"], lr_mult=1.0) != F.TUNING_DEFAULTS:
            raise SelectionRefused(f"{r['run_dir']}: tuning point {F.tuning_label(r['tuning'])}; the local-LR selection "
                                   "(06B / 06F) takes default FedAvg / AdamW / central runs only (use select-joint)")
    return list(runs)


def selection(run_dirs: Sequence[Path]) -> dict:
    """The LR_SELECTION object (with self_sha256) from exactly three verified runs."""
    import v4int.launch_fl as F
    if len(run_dirs) != 3:
        raise SelectionRefused("exactly three run dirs (m = 0.5, 1, 2)")
    runs = _fedavg_only([read_run(d) for d in run_dirs])
    if any(r.get("excluded") for r in runs):
        raise SelectionRefused("the 06B §1 selection needs three finite values")
    ms = sorted(r["lr_mult"] for r in runs)
    if ms != sorted(ORDER):
        raise SelectionRefused(f"the runs cover multipliers {ms}, not exactly {sorted(ORDER)}")
    ref = _strip_lr(runs[0]["_spec"])
    for r in runs[1:]:
        if _strip_lr(r["_spec"]) != ref:
            raise SelectionRefused(f"{r['run_dir']}: its spec differs from {runs[0]['run_dir']} beyond the multiplier")
    if len({(r["eval_manifest_id"], r["split"]) for r in runs}) != 1:
        raise SelectionRefused("the three runs were not evaluated on one manifest")
    values = {r["lr_mult"]: r["value"] for r in runs}
    sel, tie = apply_rule(values)
    obj = {"format": F.LR_SELECTION_FORMAT, "task": "FL-05 v4int.lr_screen", "rule": RULE_TEXT, "tie_tolerance": TIE,
           "order": list(ORDER), "metric": "CALIB_VALIDATION E2E client-macro MRR@20 at PRACTICAL_BEST",
           "values": {F.LR_MULT_TAG[m]: values[m] for m in ORDER}, "selected_lr_mult": sel, "tie_set": tie,
           "best_value_lr_mult": max(ORDER, key=lambda m: (values[m], -ORDER.index(m))),
           "lr_peak_base": runs[0]["lr_peak_base"], "eval_manifest_id": runs[0]["eval_manifest_id"],
           "runs": [{k: v for k, v in r.items() if k not in _NOT_DUMPED} for r in sorted(runs, key=lambda r: ORDER.index(
               r["lr_mult"]))],
           "applies_to": ["FA", "FP", "PF", "FA_1024", "DP", "S_CAL", "FA_Q8"],
           "unchanged": "PF personal LR, FedProx mu, the FT grid, LO (amendment 06B §1)",
           "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    obj["self_sha256"] = F._canon(obj)
    return obj


def write_selection(run_dirs: Sequence[Path], out: Path) -> dict:
    from v4models.variants import write_exclusive
    obj = selection(run_dirs)
    if Path(out).exists():
        raise SelectionRefused(f"{out} exists (O_EXCL: an LR selection is written once)")
    sha = write_exclusive(Path(out), json.dumps(obj, indent=1, sort_keys=True) + "\n")
    return {"path": str(out), "sha256": sha, "selected_lr_mult": obj["selected_lr_mult"],
            "self_sha256": obj["self_sha256"]}


def _first_screen(path: Path, runs: Sequence[dict]) -> dict:
    """The verified 06B LR_SELECTION.json and the check that each of its runs is screened here, unchanged."""
    import v4int.launch_fl as F
    try:
        obj = F.load_lr_selection(path)
    except F.LaunchRefused as e:
        raise SelectionRefused(f"--first-screen: {e}") from e
    if obj.get("format") != F.LR_SELECTION_FORMAT:
        raise SelectionRefused(f"--first-screen {path} is not the 06B {F.LR_SELECTION_FORMAT} file")
    by_m = {r["lr_mult"]: r for r in runs}
    for fr in obj.get("runs") or []:
        m = float(fr["lr_mult"])
        got = by_m.get(m)
        if got is None:
            raise SelectionRefused(f"the first-screen run x{F.LR_MULT_TAG.get(m, m)} ({fr.get('run_dir')}) is missing "
                                   "from --runs: the extended selection covers every screened run")
        if got["result_rows_sha256"] != fr.get("result_rows_sha256") or got.get("value") != fr.get("value"):
            raise SelectionRefused(f"{got['run_dir']}: RESULT_ROWS sha256 / value differ from the first-screen "
                                   f"record of x{F.LR_MULT_TAG.get(m, m)}")
    return {"path": str(path), "sha256": _sha(Path(path)), "self_sha256": obj["self_sha256"],
            "selected_lr_mult": obj["selected_lr_mult"]}


def selection_ext(run_dirs: Sequence[Path], first_screen: Optional[Path] = None) -> dict:
    """The LR_SELECTION_EXT object (06F §3, with self_sha256) from >= 3 verified runs."""
    import v4int.launch_fl as F
    if len(run_dirs) < MIN_EXT_RUNS:
        raise SelectionRefused(f"at least {MIN_EXT_RUNS} run dirs")
    runs = _fedavg_only([read_run(d) for d in run_dirs])
    ms = [r["lr_mult"] for r in runs]
    if len(set(ms)) != len(ms):
        raise SelectionRefused(f"each multiplier at most once, got {sorted(ms)}")
    ref = _strip_lr(runs[0]["_spec"])
    for r in runs[1:]:
        if _strip_lr(r["_spec"]) != ref:
            raise SelectionRefused(f"{r['run_dir']}: its spec differs from {runs[0]['run_dir']} beyond the multiplier")
    if len({(r["eval_manifest_id"], r["split"]) for r in runs}) != 1:
        raise SelectionRefused("the runs were not evaluated on one manifest")
    if len({r["lr_peak_base"] for r in runs}) != 1:
        raise SelectionRefused("the runs do not share one lr_peak_base")
    first = _first_screen(first_screen, runs) if first_screen is not None else None
    values = {r["lr_mult"]: r["value"] for r in runs if not r.get("excluded")}
    sel, tie = apply_rule_ext(values)
    top = max(ms)
    obj = {"format": F.LR_SELECTION_EXT_FORMAT, "task": "FL-09 v4int.lr_screen", "rule": RULE_TEXT_EXT,
           "tie_tolerance": TIE, "tie_order": "smallest |log2 m|, then the smaller m",
           "metric": "CALIB_VALIDATION E2E client-macro MRR@20 at PRACTICAL_BEST",
           "values": {F.LR_MULT_TAG[m]: values[m] for m in sorted(values, key=ext_order_key)},
           "screened": sorted(ms, key=ext_order_key),
           "excluded": [{k: r[k] for k in ("lr_mult", "run_dir", "reason", "result_rows_sha256")}
                        for r in sorted(runs, key=lambda r: ext_order_key(r["lr_mult"])) if r.get("excluded")],
           "selected_lr_mult": sel, "tie_set": tie,
           "best_value_lr_mult": sorted(values, key=lambda m: (-values[m], ext_order_key(m)))[0],
           "edge": {"largest_screened": top, "selected_is_largest_screened": sel == top,
                    "next_registered_multiplier": 2 * top if 2 * top in F.LR_MULTS else None,
                    "note": "informational only: the 06F edge rule decides whether one more (doubled) run is screened"},
           "lr_peak_base": runs[0]["lr_peak_base"], "eval_manifest_id": runs[0]["eval_manifest_id"],
           "first_screen_file_sha256": first["sha256"] if first else None, "first_screen": first,
           "runs": [{k: v for k, v in r.items() if k not in _NOT_DUMPED}
                    for r in sorted(runs, key=lambda r: ext_order_key(r["lr_mult"]))],
           "applies_to": ["FA", "FP", "PF", "FA_1024", "DP", "S_CAL", "FA_Q8"],
           "unchanged": "PF personal LR, FedProx mu, the FT grid, LO (amendments 06B §1 / 06F)",
           "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    obj["self_sha256"] = F._canon(obj)
    return obj


def write_selection_ext(run_dirs: Sequence[Path], out: Path, first_screen: Optional[Path] = None) -> dict:
    from v4models.variants import write_exclusive
    obj = selection_ext(run_dirs, first_screen)
    if Path(out).exists():
        raise SelectionRefused(f"{out} exists (O_EXCL: an LR selection is written once)")
    sha = write_exclusive(Path(out), json.dumps(obj, indent=1, sort_keys=True) + "\n")
    return {"path": str(out), "sha256": sha, "selected_lr_mult": obj["selected_lr_mult"],
            "self_sha256": obj["self_sha256"]}


def _config_record(r: Mapping, incumbent: bool) -> dict:
    import v4int.launch_fl as F
    c = dict(r["tuning"], label=F.tuning_label(r["tuning"]), value=r.get("value"), excluded=bool(r.get("excluded")),
             incumbent=bool(incumbent))
    c.update({k: r.get(k) for k in ("reason", "practical_best_mark", "run_dir", "run_id", "arm_spec_sha256",
                                    "result_rows_sha256", "run_summary_sha256") if k in r})
    return c


def _raw_best(configs: Sequence[Mapping]):
    fin = [c for c in configs if not c["excluded"]]
    return max(fin, key=lambda c: c["value"]) if fin else None


def _check_incumbent(run: Mapping, record: Mapping, what: str) -> None:
    if run["result_rows_sha256"] != record.get("result_rows_sha256") or run.get("value") != record.get("value"):
        raise SelectionRefused(f"{run['run_dir']}: RESULT_ROWS sha256 / value differ from the {what} record")


def a2_cells() -> list:
    """06L A2 (06L_D D2): the two S3 combination cells, whatever S2 selects: FedAdam (eta_s 0.01, tau 1e-3, v0 =
    tau^2), m 1, FL_CONST, client SGD-m 0.9 at lr {0.1, 1.0}."""
    import v4int.launch_fl as F
    return [F.tuning_config(optimizer="fedadam", server_lr=0.01, server_tau=F.FEDADAM_HYPER["tau"], lr_mult=1.0,
                            lr_shape="fl_const", client_opt="sgd_m", client_lr=lr) for lr in A2_CLIENT_LRS]


def _append_new(req: list, pts: Sequence[Mapping]) -> list:
    """06L_D: a cell that coincides with one already in `req` is one run (first occurrence kept)."""
    import v4int.launch_fl as F
    seen = {F.tuning_label(t) for t in req}
    for t in pts:
        if F.tuning_label(t) not in seen:
            req.append(t)
            seen.add(F.tuning_label(t))
    return req


def _server_point(c: Mapping) -> tuple:
    """The server-optimizer part of a tuning point (optimizer, FedAdam eta_s, tau)."""
    return c["optimizer"], (float(c["server_lr"]) if c["optimizer"] == "fedadam" else None), c["server_tau"]


def stage_required(stage: str, inc_t: Mapping, m_star) -> list:
    """The REQUIRED tuning points of a stage given its incumbent point (S2: FedAvg at m*, 06G §2 + 06J §1 + 06L A1;
    S3: the S2 winner, 06I §3 + 06J §2 + 06L A2; S4: the S3 winner, 06J §3). Used by select-joint and by
    launch_fl.load_fl_tuning. FL-14 (06L_D D1 / D2): the 06J cells first, then the 06L cells not already listed."""
    import v4int.launch_fl as F
    T = F.tuning_config
    if stage == "S2":
        req = [T(optimizer="fedadam", server_lr=lr, lr_mult=m) for lr, m in F.fedadam_grid(m_star)]
        req.append(T(optimizer="fedavgm", lr_mult=m_star))
        return _append_new(req, [T(optimizer="fedavgm", lr_mult=m) for m in FEDAVGM_A1_MULTS])     # 06L A1
    base = {k: inc_t[k] for k in F.TUNING_KEYS}
    if stage == "S4":                                      # FL-13 (06J §3): one at a time at the S3 winner
        return [T(**dict(base, local_passes=1)), T(**dict(base, local_passes=4)), T(**dict(base, local_batch=8))]
    srv = {"optimizer": base["optimizer"], "server_lr": base["server_lr"] if base["optimizer"] == "fedadam" else None,
           "server_tau": base["server_tau"]}
    req = [T(**srv, lr_mult=base["lr_mult"], lr_shape="fl_const")]
    req += [T(**srv, lr_mult=1.0, lr_shape="fl_const", client_opt="sgd_m", client_lr=lr) for lr in F.CLIENT_LRS]
    return _append_new(req, a2_cells())                                                        # 06L A2


def _stage_sets(stage: str, first: Mapping, runs: Sequence[dict]) -> tuple:
    """(incumbent tuning point, incumbent source record, required labels, m*) of a stage."""
    import v4int.launch_fl as F
    T = F.tuning_config
    L = F.tuning_label
    if stage == "S2":
        m_star = F.lr_mult_value(first["selected_lr_mult"])
        rec = next((r for r in first.get("runs") or [] if float(r["lr_mult"]) == m_star), None)
        if rec is None:
            raise SelectionRefused(f"the EXT file has no run record for its selected m* = {m_star}")
        inc = T(lr_mult=m_star)
        return inc, rec, [L(t) for t in stage_required(stage, inc, m_star)], m_star
    w2 = T(**{k: first["winner"].get(k) for k in F.TUNING_KEYS})
    rec = next((c for c in first.get("configs") or [] if c["label"] == L(w2)), None)
    if rec is None:
        raise SelectionRefused(f"the {PREV_STAGE[stage]} file has no configuration record of its winner")
    return w2, rec, [L(t) for t in stage_required(stage, w2, first.get("m_star"))], first.get("m_star")


def edge_groups(stage: str, configs: Sequence[Mapping], inc_label: str, req_labels: Sequence[str]) -> tuple:
    """(the registered 06J / 06L edge groups whose condition HOLDS, in registration order, as [(rule, [tuning
    points])]; a pending note or None). Conditions read the base (incumbent + required) configurations; the S2 tau
    condition reads the S2 winner (stage_decision, 06J §4(ii) included) over the base + the condition-true eta /
    FedAvgM edge runs, and is evaluated only once every condition-true edge is complete (else pending).
    FL-14 (06L_D D1 / D2): S2 order eta edge -> 06L A1 FedAvgM edge -> tau edge; the S3 SGD edge reads only the 06J
    SGD cells (the S2 winner's server optimizer), never the 06L A2 cells."""
    import v4int.launch_fl as F
    T, L = F.tuning_config, F.tuning_label
    by = {c["label"]: c for c in configs}
    groups = []
    if stage == "S4":                                      # 06J §3: no edge runs
        return groups, None
    if stage == "S2":
        base_adam = [by[lab] for lab in req_labels if lab in by and by[lab]["optimizer"] == "fedadam"]
        b = _raw_best(base_adam)
        if b is not None and b["server_lr"] == 0.01:
            eta = [T(optimizer="fedadam", server_lr=lr, lr_mult=b["lr_mult"]) for lr in F.FEDADAM_EDGE_LRS]
            groups.append(("06J §1 eta edge: the best base FedAdam run has eta_s 0.01 -> eta_s {0.003, 0.001} at "
                           "its m", eta))
        base_avgm = [by[lab] for lab in req_labels if lab in by and by[lab]["optimizer"] == "fedavgm"]
        b = _raw_best(base_avgm)                           # 06L A1: raw values, the required FedAvgM cells only
        a1 = T(optimizer="fedavgm", lr_mult=FEDAVGM_A1_EDGE)
        if b is not None and b["lr_mult"] == 1.0 and L(a1) not in req_labels:
            groups.append(("06L A1 FedAvgM edge: the raw-best required FedAvgM cell is m 1 -> FedAvgM m 0.5", [a1]))
        edge_pts = [t for _rule, pts in groups for t in pts]
        if not all(L(t) in by for t in edge_pts):
            return groups, ("the 06J §1 tau edge condition (a FedAdam S2 winner) is evaluated once the condition-true "
                            "eta / FedAvgM edge runs are complete")
        pool = [by[lab] for lab in [inc_label, *req_labels, *map(L, edge_pts)]
                if lab in by and not by[lab]["excluded"]]
        if pool:
            win = stage_decision("S2", pool)["winner"]
            if win["optimizer"] == "fedadam":
                groups.append(("06J §1 tau edge: a FedAdam run is the S2 winner (06J §4(ii) applied) -> tau 1e-5 at "
                               "its (eta_s, m)", [T(**dict(win, server_tau=1e-5))]))
        return groups, None
    if inc_label not in by:
        raise SelectionRefused(f"S3 edge rule: the incumbent {inc_label} is not among the configurations")
    srv = _server_point(by[inc_label])                     # 06L_D D2: the 06J SGD cells (the S2 winner's server)
    sgd = [by[lab] for lab in req_labels if lab in by and by[lab]["client_opt"] == "sgd_m"
           and _server_point(by[lab]) == srv]
    b = _raw_best(sgd)
    if b is not None:
        base = {k: b[k] for k in F.TUNING_KEYS}
        if b["client_lr"] == 1.0:
            groups.append(("06J §2 SGD edge: lr 1.0 is the raw best SGD run -> lr {3, 10}",
                           [T(**dict(base, client_lr=lr)) for lr in (3.0, 10.0)]))
        elif b["client_lr"] == 0.01:
            groups.append(("06J §2 SGD edge: lr 0.01 is the raw best SGD run -> lr 0.003",
                           [T(**dict(base, client_lr=0.003))]))
    return groups, None


def launch_flags(t: Mapping) -> str:
    """The launch_fl CALIBRATION-screen options of a tuning point (for the EdgeRunsMissing message)."""
    import v4int.launch_fl as F
    out = []
    if t["optimizer"] != "fedavg":
        out.append(f"--server-opt {t['optimizer']}")
        if t["optimizer"] == "fedadam":
            out.append(f"--server-lr {F.num_tag(t['server_lr'])}")
            if t["server_tau"] != F.FEDADAM_HYPER["tau"]:
                out.append(f"--server-tau {F.num_tag(t['server_tau'])}")
    out.append(f"--lr-mult {F.num_tag(t['lr_mult'])}")
    if t["lr_shape"] != "central":
        out.append(f"--lr-shape {t['lr_shape']}")
    if t["client_opt"] != "adamw_fresh":
        out.append(f"--client-opt {t['client_opt']} --client-lr {F.num_tag(t['client_lr'])}")
    if t["local_passes"] != F.KNOBS["passes"]:
        out.append(f"--local-passes {t['local_passes']}")
    if t["local_batch"] != F.KNOBS["batch"]:
        out.append(f"--local-batch {t['local_batch']}")
    return " ".join(out)


def check_stage_runs(stage: str, configs: Sequence[Mapping], inc_label: str, req_labels: Sequence[str]) -> list:
    """The run set of a stage (configuration records: label, tuning keys, excluded, incumbent): exactly one incumbent
    (inc_label), every required label, every registered edge group whose condition holds COMPLETE (FIX-DELTA-02 M3:
    absent -> EdgeRunsMissing naming the runs; partial -> refused) and nothing else. Returns the edge groups as
    [{"rule", "labels"}] (the file's `edges`). Used by select-joint and launch_fl.load_fl_tuning."""
    import v4int.launch_fl as F
    L = F.tuning_label
    labels = [c["label"] for c in configs]
    if len(set(labels)) != len(labels):
        raise SelectionRefused(f"each configuration at most once, got {sorted(labels)}")
    incs = [c["label"] for c in configs if c.get("incumbent")]
    if incs != [inc_label]:
        raise SelectionRefused(f"the incumbent must be exactly {inc_label} (got {incs})")
    missing = [lab for lab in req_labels if lab not in labels]
    if missing:
        raise SelectionRefused(f"required {stage} run(s) missing: {missing}")
    groups, pending = edge_groups(stage, configs, inc_label, req_labels)
    needed = []
    for rule, pts in groups:                                  # a "two runs" edge is complete or absent
        glabs = [L(t) for t in pts]
        present = sorted(lab for lab in glabs if lab in labels)
        if present and len(present) != len(glabs):
            raise SelectionRefused(f"the 06J edge runs {sorted(glabs)} must all be present (got {present})")
        if not present:
            needed += [(rule, t) for t in pts]
    if needed:
        lines = "; ".join(f"{L(t)} [{rule}] -> launch_fl --role CALIBRATION {launch_flags(t)}" for rule, t in needed)
        raise EdgeRunsMissing(f"{stage}: a registered 06J / 06L A1 edge condition holds and its runs are missing "
                              f"(FIX-DELTA-02 M3: the stage file is written only with them): {lines}"
                              + (f"; then {pending}" if pending else ""),
                              [dict(t, label=L(t)) for _rule, t in needed])
    allowed = {L(t) for _rule, pts in groups for t in pts}
    bad = [lab for lab in labels if lab not in req_labels and lab != inc_label and lab not in allowed]
    if bad:
        raise SelectionRefused(f"{stage} runs not registered / not allowed by the 06J / 06L A1 edge conditions: {bad} "
                               f"(allowed now: {sorted(allowed)})")
    return [{"rule": rule, "labels": [L(t) for t in pts]} for rule, pts in groups]


def _one_device(runs: Sequence[Mapping]) -> Optional[dict]:
    """FIX-DELTA-02 m5: the runs share one ARM_SPEC.json device record (evaluation device class + pin)."""
    devs = {json.dumps(r.get("device"), sort_keys=True) for r in runs}
    if len(devs) != 1:
        raise SelectionRefused(f"the runs do not share one evaluation device class / pin (ARM_SPEC device): "
                               f"{sorted(devs)}")
    return runs[0].get("device")


def selection_stage(stage: str, run_dirs: Sequence[Path], first_selection: Path) -> dict:
    """A stage file (06J §1-§3 + §4(ii), with self_sha256) from the incumbent run + the stage's runs."""
    import v4int.launch_fl as F
    if stage not in STAGES:
        raise SelectionRefused(f"--stage must be one of {STAGES}")
    try:
        first = F.load_lr_selection(first_selection) if stage == "S2" else F.load_fl_tuning(first_selection)
    except F.LaunchRefused as e:
        raise SelectionRefused(f"--first-selection: {e}") from e
    want_fmt = F.LR_SELECTION_EXT_FORMAT if stage == "S2" else F.FL_TUNING_FORMAT
    if first.get("format") != want_fmt or (stage != "S2" and first.get("stage") != PREV_STAGE[stage]):
        raise SelectionRefused(f"--first-selection {first_selection} is not the "
                               f"{'06F EXT' if stage == 'S2' else PREV_STAGE[stage]} file")
    inc_t, inc_rec, req_labels, m_star = _stage_sets(stage, first, [])
    runs = [read_run(d) for d in run_dirs]
    labels = [F.tuning_label(r["tuning"]) for r in runs]
    if len(set(labels)) != len(labels):
        raise SelectionRefused(f"each configuration at most once, got {sorted(labels)}")
    inc_label = F.tuning_label(inc_t)
    by_run = dict(zip(labels, runs))
    if inc_label not in by_run:
        raise SelectionRefused(f"the incumbent {inc_label} run is missing from --runs")
    _check_incumbent(by_run[inc_label], inc_rec, "previous stage's incumbent")
    configs = [_config_record(by_run[lab], lab == inc_label) for lab in labels]
    ref = _strip_lr(runs[0]["_spec"])
    for r in runs[1:]:
        if _strip_lr(r["_spec"]) != ref:
            raise SelectionRefused(f"{r['run_dir']}: its spec differs from {runs[0]['run_dir']} beyond the tuning "
                                   "dimensions")
    adamw = {(r["_spec"]["cfg"]["solver"].get("weight_decay"), r["_spec"]["cfg"]["solver"].get("fused"))
             for r in runs if r["tuning"]["client_opt"] == "adamw_fresh"}
    if len(adamw) > 1:
        raise SelectionRefused(f"the AdamW runs differ in (weight_decay, fused): {sorted(map(str, adamw))}")
    if len({(r["eval_manifest_id"], r["split"]) for r in runs}) != 1:
        raise SelectionRefused("the runs were not evaluated on one manifest")
    if len({r["lr_peak_base"] for r in runs}) != 1 or runs[0]["lr_peak_base"] != first.get("lr_peak_base"):
        raise SelectionRefused("the runs do not share the previous selection's lr_peak_base")
    device = _one_device(runs)
    edges = check_stage_runs(stage, configs, inc_label, req_labels)
    configs.sort(key=lambda c: stage_order_key(stage, c))
    dec = stage_decision(stage, [c for c in configs if not c["excluded"]])
    win, tie = dec["winner"], dec["tie_set"]
    prev = {"stage": PREV_STAGE[stage], "path": str(first_selection), "sha256": _sha(Path(first_selection)),
            "self_sha256": first["self_sha256"],
            "winner": F.tuning_label(inc_t)}
    history = ([] if stage == "S2" else list(first.get("stage_history") or [])) + [prev]
    obj = {"format": F.FL_TUNING_FORMAT, "stage": stage, "task": "FL-10 v4int.lr_screen select-joint",
           "rule": RULE_TEXT_STAGE[stage], "tie_tolerance": TIE, "tie_order": TIE_ORDER_TEXT[stage],
           "metric": "CALIB_VALIDATION E2E client-macro MRR@20 at PRACTICAL_BEST",
           "configs": configs, "values": {c["label"]: c["value"] for c in configs if not c["excluded"]},
           "excluded": [c for c in configs if c["excluded"]], "tie_set": tie,
           "winner": dict(win, label=F.tuning_label(win)), "incumbent": inc_label, "required": list(req_labels),
           "min_gain": MIN_GAIN, "gain": dec["gain"], "best_candidate": dec["best_candidate"],
           "no_measurable_gain": dec["no_measurable_gain"], "tie_band_set": dec["tie_band_set"],
           "edges": edges, "device": device,
           "m_star": m_star, "fedadam": {"algorithm": "Reddi et al. 2021 ICLR Alg. 2 (FEDADAM), no bias correction, "
                                                      "m_0 = 0, v_0 = tau^2", **F.FEDADAM_HYPER},
           "fedavgm": {"algorithm": "Hsu, Qi & Brown 2019 server momentum", "lr": F.FEDAVGM_LR, **F.FEDAVGM_HYPER},
           "client_sgd": {"momentum": 0.9, "dampening": 0.0, "nesterov": False, "weight_decay": 0.0, "clip": 1.0,
                          "fresh_per_visit": True},
           "lr_peak_base": runs[0]["lr_peak_base"], "eval_manifest_id": runs[0]["eval_manifest_id"],
           "first_selection": {"path": str(first_selection), "sha256": prev["sha256"], "self_sha256": prev["self_sha256"]},
           "stage_history": history,
           "applies_to": ["FA", "FP", "PF", "FA_1024", "DP", "S_CAL", "FA_Q8"],
           "unchanged": "PF personal LR, FedProx mu, the FT grid, LO, beta1 / beta2 / tau defaults"
                        + ("" if stage == "S4" else ", passes / batch") + " (amendments 06B §1 / 06F / 06G / 06I / 06J)",
           "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    if stage == "S4":                                      # FIX-DELTA-02 M1 (DL 219)
        obj["applies_to_exception"] = {"arms": list(F.POISSON_ARMS), "local_passes": F.KNOBS["passes"],
                                       "local_batch": F.KNOBS["batch"], "rule": POISSON_EXCEPTION_TEXT}
    obj["self_sha256"] = F._canon(obj)
    return obj


def write_selection_stage(stage: str, run_dirs: Sequence[Path], out: Path, first_selection: Path) -> dict:
    from v4models.variants import write_exclusive
    obj = selection_stage(stage, run_dirs, first_selection)
    if Path(out).exists():
        raise SelectionRefused(f"{out} exists (O_EXCL: a stage selection is written once)")
    sha = write_exclusive(Path(out), json.dumps(obj, indent=1, sort_keys=True) + "\n")
    import v4int.launch_fl as F
    return {"path": str(out), "sha256": sha, "self_sha256": obj["self_sha256"], "stage": stage,
            "winner": {k: obj["winner"][k] for k in F.TUNING_KEYS}}


# ------------------------------------------------------------------------------------------------ FL-12 warm screens
RULE_TEXT_WARM = ("warm-start LR screen: the highest CALIB_VALIDATION E2E client-macro MRR@20 at PRACTICAL_BEST over the "
                  "two runs (S7 FA_WARM: factor 1 = m*, 1/4 = m*/4; S8 FA_WARM_FROZEN: 1 = m*, 4 = 4 m*) at the tuning "
                  "selection's point from one theta_pre; non-finite values excluded and recorded; |delta| <= 0.0005 "
                  "ties -> factor 1 (m*) — PREG1_AMENDMENT_06K §2-§3 (06I §3 / 06J rule restricted to the two runs)")


def apply_rule_warm(arm: str, values: Mapping[float, float]) -> tuple:
    """(winner factor, tie set: factor 1 first) over the finite values of a warm screen."""
    import v4int.launch_fl as F
    if arm not in F.WARM_ARMS:
        raise SelectionRefused(f"{arm} is not a warm arm")
    if not values:
        raise SelectionRefused("no warm run with a finite value")
    vals = {}
    for f, v in values.items():
        f, v = float(f), float(v)
        if f not in F.WARM_FACTORS[arm]:
            raise SelectionRefused(f"factor {f} is not registered for {arm} ({F.WARM_FACTORS[arm]})")
        if not math.isfinite(v):
            raise SelectionRefused(f"factor {f}: non-finite value {v!r} (excluded, not ranked)")
        vals[f] = v
    best = max(vals.values())
    tie = sorted((f for f in vals if best - vals[f] <= TIE), key=lambda f: (f != 1.0, f))
    return tie[0], tie


# FL-15 (PREG1_AMENDMENT_06M_B B2): the registered S7 / S8 screens are 2-D (4 runs); the 2-run screen above is superseded
RULE_TEXT_WARM_2D = ("warm-start screen (06M_B B2, amends 06K §2-§3): the highest CALIB_VALIDATION E2E client-macro "
                     "MRR@20 at PRACTICAL_BEST over the 4 runs at the tuning selection's point from one theta_pre — S7 "
                     "FA_WARM {m*, m*/4} x {eta*, eta*/10} (FedAdam) or {eta*, eta*/4} (FedAvgM / FedAvg); S8 "
                     "FA_WARM_FROZEN {m*, 4 m*} x the same server axis; the 06J rule restricted to these runs: the "
                     "0.0005 tie band, then 06J §4(ii) with the incumbent (m*, eta*): the best candidate replaces it only "
                     "if it gains >= 0.002, else no measurable gain and (m*, eta*) carries on; ties -> (m*, eta*); "
                     "non-finite values excluded and recorded (06J §4(v)); raw values")
WARM_TIE_ORDER_TEXT = ("(m*, eta*) first, then fewer changed axes, then the smaller |log2| of the local factor, then of "
                       "the server-LR factor")


def warm_order_key(cell) -> tuple:
    """FL-15 tie order inside the 0.0005 band (FL-15 reading of 'ties -> (m*, eta*)'): the incumbent (1, 1), then the
    cell that changes fewer axes, then the smaller |log2 local factor|, then the smaller |log2 server factor|."""
    wf, sf = float(cell[0]), float(cell[1])
    return ((wf, sf) != (1.0, 1.0), (wf != 1.0) + (sf != 1.0), abs(math.log2(wf)), abs(math.log2(sf)), wf, sf)


def warm_incumbent(vals: Mapping) -> tuple:
    """The incumbent cell (m*, eta*) = (1, 1) of a 2-D warm screen; refused when it has no finite value (06J §4(ii)
    measures the gain over the incumbent; Ops decides what happens then)."""
    inc = (1.0, 1.0)
    if inc not in vals:
        raise SelectionRefused("the incumbent (m*, eta*) run is excluded / missing: 06J §4(ii) measures the gain over "
                               "the incumbent, so no warm selection is written without it (Ops decision)")
    return inc


def warm_decision(arm: str, optimizer: str, values: Mapping) -> dict:
    """The 2-D warm decision (06M_B B2: the 06J rule restricted to these runs) over the finite cells {(local factor,
    server factor): value}: every cell registered (launch_fl.warm_grid), the incumbent (1, 1) finite (06J §4(ii)
    measures the gain over it; a non-finite incumbent is refused for an Ops decision), the 0.0005 band in
    warm_order_key order (tie_band_set), then the best candidate (highest finite non-incumbent value) replaces the
    incumbent only if gain >= MIN_GAIN; else no_measurable_gain, winner = (1, 1) and tie set = [(1, 1)]."""
    import v4int.launch_fl as F
    if arm not in F.WARM_ARMS:
        raise SelectionRefused(f"{arm} is not a warm arm")
    if optimizer not in F.WARM_SERVER_FACTORS:
        raise SelectionRefused(f"server optimiser {optimizer!r} has no registered 06M_B B2 server axis")
    grid = F.warm_grid(arm, optimizer)
    vals = {}
    for c, v in values.items():
        c, v = (float(c[0]), float(c[1])), float(v)
        if c not in grid:
            raise SelectionRefused(f"cell {c} is not registered for {arm} / {optimizer} ({grid})")
        if not math.isfinite(v):
            raise SelectionRefused(f"cell {F.warm_cell_label(*c)}: non-finite value {v!r} (excluded, not ranked)")
        vals[c] = v
    inc = warm_incumbent(vals)
    best = max(vals.values())
    band = sorted((c for c in vals if best - vals[c] <= TIE), key=warm_order_key)
    cands = [c for c in vals if c != inc]
    top = min(cands, key=lambda c: (-vals[c], warm_order_key(c))) if cands else None
    gain = None if top is None else vals[top] - vals[inc]
    no_gain = gain is None or gain < MIN_GAIN
    tie = [inc] if no_gain else band
    L = F.warm_cell_label
    return {"winner": tie[0], "tie_set": [L(*c) for c in tie], "tie_band_set": [L(*c) for c in band],
            "best_candidate": None if top is None else L(*top), "gain": gain, "no_measurable_gain": bool(no_gain),
            "min_gain": MIN_GAIN, "incumbent": L(*inc)}


def check_warm_tuning_stage(tobj: Mapping, tuning) -> None:
    """REV-FL-15 m2: the S7 / S8 screens run at the S4 winner (06K §2 / 06M_B B2), so select-warm takes the FINAL
    stage-S4 tuning selection only (an S2 / S3 stage file is refused before any warm file is written)."""
    import v4int.launch_fl as F
    if tobj.get("stage") != F.MAIN_TUNING_STAGE:
        raise SelectionRefused(f"--tuning {tuning}: stage {tobj.get('stage')!r}; the S7 / S8 warm screens are bound to "
                               f"the FINAL stage-{F.MAIN_TUNING_STAGE} FL_TUNING_SELECTION (06K §2 / 06M_B B2)")


def selection_warm(stage: str, run_dirs: Sequence[Path], tuning: Path) -> dict:
    """The 2-D warm selection object (06K §2-§3 as amended by 06M_B B2, with self_sha256) from the 4 warm CALIB runs of
    a stage (format PPSI_V4_FL_WARM_SELECTION_2)."""
    import v4int.launch_fl as F
    arm = {v: k for k, v in F.WARM_STAGE.items()}.get(stage)
    if arm is None:
        raise SelectionRefused(f"--stage must be one of {sorted(F.WARM_STAGE.values())}")
    try:
        tobj = F.load_fl_tuning(tuning)
    except F.LaunchRefused as e:
        raise SelectionRefused(f"--tuning: {e}") from e
    check_warm_tuning_stage(tobj, tuning)                  # REV-FL-15 m2: the FINAL (S4) tuning selection only
    tw = F.tuning_config(**{k: tobj["winner"].get(k) for k in F.TUNING_KEYS})
    opt = tw["optimizer"]
    grid = F.warm_grid(arm, opt)
    if len(run_dirs) != len(grid):
        raise SelectionRefused(f"{stage}: exactly {len(grid)} runs, the registered 2-D grid (local factor x server LR, "
                               f"06M_B B2): {[F.warm_cell_label(*c) for c in grid]} (the 06K 2-run screen is superseded)")
    runs = [read_run(d, arms=(arm,)) for d in run_dirs]
    cells = [(r["warm_lr_factor"], r["warm_server_lr_factor"]) for r in runs]
    if sorted(cells) != sorted(grid):
        raise SelectionRefused(f"{stage}: the runs cover cells {sorted(F.warm_cell_label(*c) for c in cells)}, not the "
                               f"registered {sorted(F.warm_cell_label(*c) for c in grid)}")
    for r in runs:
        if r["tuning"] != tw:
            raise SelectionRefused(f"{r['run_dir']}: tuning point {F.tuning_label(r['tuning'])} is not the tuning "
                                   f"selection's winner {F.tuning_label(tw)}")
    ref = _strip_lr(runs[0]["_spec"])
    for r in runs[1:]:
        if _strip_lr(r["_spec"]) != ref:
            raise SelectionRefused(f"{r['run_dir']}: the warm runs differ beyond the local / server LR factors "
                                   "(theta_pre, arm, knobs, ...)")
    if len({(r["eval_manifest_id"], r["split"]) for r in runs}) != 1:
        raise SelectionRefused("the runs were not evaluated on one manifest")
    if any(r["lr_peak_base"] != tobj.get("lr_peak_base") for r in runs):
        raise SelectionRefused("the runs do not share the tuning selection's lr_peak_base")
    device = _one_device(runs)                             # FIX-DELTA-02 m5
    values = {(r["warm_lr_factor"], r["warm_server_lr_factor"]): r["value"] for r in runs if not r.get("excluded")}
    dec = warm_decision(arm, opt, values)
    L = F.warm_cell_label
    eta = 1.0 if opt == "fedavg" else float(tw["server_lr"])
    lrs = {F.num_tag(f): (eta if f == 1.0 else eta / F.WARM_SERVER_DIVISORS[f]) for f in F.WARM_SERVER_FACTORS[opt]}
    wf, wsf = dec["winner"]
    obj = {"format": F.WARM_SELECTION_FORMAT_2D, "stage": stage, "arm": arm,
           "task": "FL-15 v4int.lr_screen select-warm (2-D)", "rule": RULE_TEXT_WARM_2D, "tie_tolerance": TIE,
           "tie_order": WARM_TIE_ORDER_TEXT, "metric": "CALIB_VALIDATION E2E client-macro MRR@20 at PRACTICAL_BEST",
           "values": {L(*c): values[c] for c in sorted(values, key=warm_order_key)},
           "screened": [L(*c) for c in sorted(grid, key=warm_order_key)],
           "excluded": [{"label": L(r["warm_lr_factor"], r["warm_server_lr_factor"]),
                         **{k: r[k] for k in ("warm_lr_factor", "warm_server_lr_factor", "run_dir", "reason",
                                              "result_rows_sha256")}}
                        for r in sorted(runs, key=lambda r: warm_order_key((r["warm_lr_factor"],
                                                                            r["warm_server_lr_factor"])))
                        if r.get("excluded")],
           "winner": {"warm_lr_factor": wf, "warm_server_lr_factor": wsf, "label": L(wf, wsf),
                      "server_lr": lrs[F.num_tag(wsf)]},
           "winner_factor": wf, "winner_server_lr_factor": wsf,
           **{k: dec[k] for k in ("tie_set", "tie_band_set", "best_candidate", "gain", "no_measurable_gain",
                                  "min_gain", "incumbent")},
           "server_axis": {"optimizer": opt, "eta_star": eta, "factors": list(F.WARM_SERVER_FACTORS[opt]),
                           "server_lrs": lrs, "fedavg_note": ("FedAvg at server LR 1/4 = FedAvgM beta 0 at LR 0.25 "
                                                              "(launch_fl.warm_server_opt)") if opt == "fedavg" else None},
           "local_factors": list(F.WARM_FACTORS[arm]),
           "winner_local_lr_rule": "local LR = factor x the tuning selection's local LR (m* x SMALL peak, or the "
                                   "client SGD LR)",
           "winner_server_lr_rule": "server LR = eta* / 10 (FedAdam) or eta* / 4 (FedAvgM / FedAvg) at factor != 1",
           "tuning_selection": {"path": str(tuning), "sha256": _sha(Path(tuning)), "self_sha256": tobj["self_sha256"],
                                "stage": tobj["stage"], "winner": F.tuning_label(tw)},
           "init_from": dict(runs[0]["_spec"]["init_from"]), "lr_peak_base": runs[0]["lr_peak_base"], "device": device,
           "eval_manifest_id": runs[0]["eval_manifest_id"],
           "runs": [{k: v for k, v in r.items() if k not in ("_spec", "optimizer", "tuning", "device")}
                    for r in sorted(runs, key=lambda r: warm_order_key((r["warm_lr_factor"],
                                                                        r["warm_server_lr_factor"])))],
           "authority": "PREG1_AMENDMENT_06K §2-§3, 06M_B B2, 06J §4(i)-(ii) / (v)",
           "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    obj["self_sha256"] = F._canon(obj)
    return obj


def write_selection_warm(stage: str, run_dirs: Sequence[Path], out: Path, tuning: Path) -> dict:
    from v4models.variants import write_exclusive
    obj = selection_warm(stage, run_dirs, tuning)
    if Path(out).exists():
        raise SelectionRefused(f"{out} exists (O_EXCL: a warm selection is written once)")
    sha = write_exclusive(Path(out), json.dumps(obj, indent=1, sort_keys=True) + "\n")
    return {"path": str(out), "sha256": sha, "self_sha256": obj["self_sha256"], "stage": stage,
            "winner_factor": obj["winner_factor"], "winner_server_lr_factor": obj["winner_server_lr_factor"]}


def main(argv=None, say: Callable = print) -> int:
    ap = argparse.ArgumentParser(description="PPSI V4 federated local-LR / tuning selection (FL-05 / FL-09 / FL-10, "
                                             "amendments 06B §1, 06F §3, 06G §3, 06I §3, 06J)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("select")
    s.add_argument("--runs", nargs="+", type=Path, required=True)
    s.add_argument("--first-screen", type=Path, default=None, help="the 06B LR_SELECTION.json (verified + recorded)")
    s.add_argument("--out", type=Path, required=True)
    j = sub.add_parser("select-joint")                    # FL-10 (06G §3 / 06J)
    j.add_argument("--stage", required=True, choices=STAGES)
    j.add_argument("--runs", nargs="+", type=Path, required=True)
    j.add_argument("--first-selection", type=Path, required=True,
                   help="S2: the 06F LR_SELECTION_EXT.json; S3: the S2 stage file; S4: the S3 stage file")
    j.add_argument("--out", type=Path, required=True)
    w = sub.add_parser("select-warm")                     # FL-12 (06K §2-§3)
    w.add_argument("--stage", required=True, choices=("S7", "S8"))
    w.add_argument("--runs", nargs="+", type=Path, required=True,
                   help="FL-15 (06M_B B2): the 4 runs of the 2-D grid (local factor x server-LR factor)")
    w.add_argument("--tuning", type=Path, required=True, help="the registered FL_TUNING_SELECTION.json (S4)")
    w.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    import ppsi_env  # noqa: F401  (isolation before v4 imports)
    try:
        if args.cmd == "select-joint":
            rec = write_selection_stage(args.stage, args.runs, args.out, args.first_selection)
        elif args.cmd == "select-warm":
            rec = write_selection_warm(args.stage, args.runs, args.out, args.tuning)
        else:
            rec = write_selection_ext(args.runs, args.out, args.first_screen)
    except SelectionRefused as e:
        print(f"LR SELECTION REFUSED: {e}", file=sys.stderr, flush=True)
        return 2
    say(json.dumps(rec, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
