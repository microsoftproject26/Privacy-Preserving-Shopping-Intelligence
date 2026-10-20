"""v4int.p2_select — the Phase-2 CALIB selection tool (work packet P2-06). It decides the Phase-2 arm on CALIB values only.
Binding spec: contracts/PREG2.md (REGISTERED, DL 411, sha256 be0561abe5ceec450d3a75bef49853a3c480d3847f75559dd94e9e0545df18a8)
§4.2, §5.1-§5.9, §6.3 S8, §11 D-j. REV-P2-06 fixes applied (company/handoffs/P2-06.md "REV fixes").

  python -B code/v4int/p2_select.py select --stage C1 --runs <8 C1 screen run dirs (+ C1 edge runs)> \
         --anchor <P2_ANCHOR_PRE_CALIB reeval out dir> [--out contracts/fl/P2_C1_SELECTION.json]
  python -B code/v4int/p2_select.py select --stage WA --runs <21 WA run dirs (+ <= 2 WA edge runs)> \
         --anchor <dir> [--incumbent-phase1 <S8Tc run dir> --p2t3-receipt <file>] [--out .../P2_WARM_SELECTION_A.json]
  python -B code/v4int/p2_select.py futility --wa-selection <A file> --c1-selection <C1 file>
  python -B code/v4int/p2_select.py select --stage FINAL --runs <A* run dir> <WB / WC run dirs (+ edge run)> \
         --anchor <dir> --wa-selection <A file> --c1-selection <C1 file> [--dropped <units> --cap-plan <file>] \
         [--incumbent-phase1 <dir> --p2t3-receipt <file>] [--out .../P2_WARM_SELECTION_FINAL.json]

Registered rule implemented (PREG2 REGISTERED):
  * Metric (§5.6): CALIB_VALIDATION E2E client-macro MRR@20 at PRACTICAL_BEST (marks 0.5 .. 6.0; mark 0 = theta_pre is
    never a candidate, §5.1). Every run must be a CALIBRATION run evaluated on CALIB_VALIDATION_5_100_32768 (MAIN /
    TEST refused, F3 / §6.3 S8). A run whose selection value is non-finite is EXCLUDED and recorded, never ranked. A
    non-finite value anywhere else in a run's record (D1-ext summary, other marks) never blocks the file: it is written
    as a string ("NaN" / "Infinity" / "-Infinity") and the run is FLAGGED (candidate `nonfinite_fields`, top-level
    `nonfinite_diagnostics`); D1-ext never selects (§3.3).
  * Order (§5.6): A* over WA core (18) + WA-ext (2) + the incumbent (FedAdam 1e-3, c 0.1); W* over {A*} + WB (3) +
    WC (5) minus the cells dropped under §5.9; C1 within each variant (§4.2).
  * Tie band 0.001 (§5.6), INCLUSIVE, in exact decimal arithmetic (Fraction of the shortest repr of each value, so
    0.1716 vs 0.1706 is a gap of exactly 0.001 and tied). Inside the band, p2_order_key: (1) fewer upload bytes per
    visit, (2) fewer trained parameters, (3) the smaller effective server step (FedAdam eta_s; FedAvgM eta_s / (1 -
    0.9); FedAvg eta_s; exact), (4) the smaller client LR applied, (5) the lexically first label. C1: bytes do not
    differ; then the smaller LR (§5.6).
  * Anchor flag (§5.6): NO_MEASURABLE_CALIB_GAIN iff CALIB_GAIN = winner - PRE_CALIB < 0.001 (STRICT, exact); the
    winner is carried on whatever the flag says.
  * Edge rule (§5.7, registered text): a winner on the edge of a numeric axis -> that axis extended ONCE, one step
    outward, at the winner's other coordinates (the registered step table EDGE_STEPS; no corner run — the registered
    text does not name one); at most 3 extension runs per stage; the extension runs are REQUIRED (P2EdgeRunsMissing,
    exit 3); the rule is re-applied over the union; if the winner is again on the new edge (the union winner is a
    registered extension cell): EDGE_UNRESOLVED; no second extension. Facts the text does not classify (a non-finite
    extension run; the winner moved to another base run) are recorded as edge `notes`, never as a status.
  * Futility (§5.8): FUTILITY_SKIP_WB_WC iff C1_best - PRE_CALIB < 0.001 AND WA_best - PRE_CALIB < 0.001 (STRICT,
    exact); D-j printed beside it.
  * Budget cap (§5.9): 250 GPU-h; ONE strict order: P2_WSTAR_DP1 -> P2_WSTAR_SECAGG (MAIN) -> P2_WSTAR_DPFTRL8 s2027 ->
    WC3/WC4 -> WB2 -> WC1/WC2 -> WB3 -> C2. Drops must be a prefix of that order AND equal cap_drop_plan of the given
    projection (FINAL --cap-plan). Part-B arms unfinished at lock time are declared not run under §2.1 (P2 lock), not
    here.

Reading runs. P2 warm-FL runs: read_p2_run re-implements lr_screen.read_run's checks one by one (R1-R12 below; lr_screen
unchanged — read_run itself cannot admit a P2 spec: spec_tuning / check_warm_spec are Phase-1 functions), with the two
Phase-1-specific checks replaced by their launch_fl P2 counterparts (check_p2_spec; warm_row_labels + p2_row_labels);
the value-free ARM_SPEC.json is verified BEFORE any value-bearing file is opened (§6.3 S8):
  R1  ARM_SPEC.json readable; its arm_spec hashes (launch_fl.spec_digest) to its arm_spec_sha256      (read_run :326-338)
  R2  arm in P2_ARMS, role CALIBRATION, cfg.seed = launch_fl.CALIB["seed"] (2026)                       (read_run :334-336)
  R3  launch_fl.check_p2_spec(spec): the spec is exactly p2_arm_spec of its own recipe        (replaces spec_tuning +
      check_warm_spec + the lr_peak_local check, read_run :339-351)
  R4  RUN_SUMMARY.json / RESULT_ROWS.json readable; RUN_SUMMARY status DONE_MAIN                        (read_run :324-333)
  R5  RUN_SUMMARY arm_spec_sha256 = digest of RUN_SUMMARY arm_spec = the ARM_SPEC digest                (read_run :337-338)
  R6  12 evaluations at half-EFE marks 1 .. 12                                                          (read_run :352-354)
  R7  exactly one PRACTICAL_BEST E2E macro mrr@20 row, on CALIB_VALIDATION_5_100_32768                  (read_run :355-359)
  R8  row identity: role CALIBRATION, run_label FL_LR_SCREEN, method = arm, arm_spec_sha256 = digest, fl_lr_mult 1
                                                                                                        (read_run :360-364)
  R9  row labels = warm_row_labels(spec) + p2_row_labels(spec) with launch_fl.fl_rows' CALIB relabel (row_status /
      comparison_status FL_LR_SCREEN), every key equal, AND no key of read_run's label set (READ_RUN_LABEL_KEYS, +
      p2_item_lr_mult) present beyond them — a set comparison as in read_run   (replaces row_labels(t), read_run :365-372)
  R10 value non-finite (or not a number) -> EXCLUDED and recorded                                       (read_run :376-386)
  R11 value_status OK                                                                                   (read_run :387-388)
  R12 value = the evaluation at best_half_efe; schedule_mark a finite number = best_half_efe / 2 (missing /
      null / non-numeric refused, as read_run's float(row["schedule_mark"]) refuses it)             (read_run :389-392)
then p2_run_adapter (THE alignment point for P2 recipe / bytes / D1-ext fields). C1 runs: delegated to the P2-05
verifier v4int.central_ft.read_c1_run (S8-strict: RUN_START identity before any value file). The anchor: read_anchor ->
anchor_adapter (P2-07 format, an assumption until P2-07 lands). S8's incumbent run (Phase-1 label FA_WARM_FROZEN) is
read by the unedited lr_screen.read_run, ONLY via --incumbent-phase1 with a P2-T3 receipt.

Output (O_EXCL, self_sha256 = launch_fl._canon, file text json.dumps(indent=1, sort_keys=True)): every candidate, the
excluded runs, the base decision, the edge record, the winner, the tie set, the anchor, flags, futility / drops (FINAL),
tool shas. Printed: winner label(s), flags, file sha256 (no value).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from fractions import Fraction
from pathlib import Path
from typing import Callable, Mapping, Optional, Sequence

HERE = Path(__file__).resolve().parent
CODE = HERE.parent
V4 = CODE.parent
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))
sys.dont_write_bytecode = True

import v4int.lr_screen as LR  # noqa: E402  (stdlib-only at import; launch_fl is imported lazily, as lr_screen does)

# ------------------------------------------------------------------------------------------------ registered values
PREG2_SHA256 = "be0561abe5ceec450d3a75bef49853a3c480d3847f75559dd94e9e0545df18a8"
TIE_CALIB = 0.001              # §5.6 CALIB tie band (inclusive)
FUTILITY_MIN = 0.001           # §5.8 (strict)
ANCHOR_MIN_GAIN = 0.001        # §5.6 anchor flag (strict; = the CALIB tie band)
TIE_FRAC = FUT_FRAC = ANCHOR_FRAC = Fraction(1, 1000)
CAP_GPU_H = 250.0              # §5.9
CALIB_EVAL_MANIFEST = "CALIB_VALIDATION_5_100_32768"
CALIB_SEED = 2026
STAGES = ("WA", "FINAL", "C1")
SERVER_OPTS = ("fedavg", "fedavgm", "fedadam")
FEDAVGM_BETA = Fraction(9, 10)                                        # §5.2: FedAvgM momentum 0.9
WA_SERVER_POINTS = (("fedavg", 0.3), ("fedavg", 1.0), ("fedavgm", 0.03), ("fedavgm", 0.1),
                    ("fedadam", 1e-4), ("fedadam", 3e-4))             # §5.2 WA core server points
WA_CLIENT_LRS = (0.003, 0.01, 0.03)                                   # §5.2 client SGD-m 0.9 LR
WA_EXT_POINTS = (("fedadam", 1e-5), ("fedadam", 1e-4))                # §5.2 WA-ext
WA_EXT_CLIENT_LR = 0.1
INCUMBENT_CELL = ("fedadam", 1e-3, 0.1)                               # §5.2 S8 f1/sf0.1 (run S8Tc)
WB_CELLS = {"WB1": {"items": "unfrozen", "item_lr_mult": 0.1, "upload": "dense"},                  # §5.3
            "WB2": {"items": "unfrozen", "item_lr_mult": 0.01, "upload": "dense"},
            "WB3": {"items": "unfrozen", "item_lr_mult": 0.1, "upload": "delta_rows"}}
WC_CELLS = {"WC1": {"lora_rank": 4, "lora_lr_factor": 10}, "WC2": {"lora_rank": 16, "lora_lr_factor": 10},   # §5.4
            "WC3": {"lora_rank": 4, "lora_lr_factor": 1}, "WC4": {"lora_rank": 16, "lora_lr_factor": 1},
            "WC5": {"personal_head": True}}
C1_LABELS = {"FULL": "P2_C1_FT_CENTRAL_FULL", "FROZEN": "P2_C1_FT_CENTRAL_FROZEN"}             # §4.2
C1_PEAK_LRS = (1e-5, 3e-5, 1e-4, 3e-4)
# §5.7 step table, verbatim (axis -> {edge value: the one step outward}).
EDGE_STEPS = {
    "client_lr": {0.003: 0.001, 0.03: 0.1},
    "server_lr:fedavg": {0.3: 0.1, 1.0: 2.0},
    "server_lr:fedavgm": {0.03: 0.01, 0.1: 0.3},
    "server_lr:fedadam": {1e-4: 3e-5, 3e-4: 1e-3},                     # "(at c*)"
    "server_lr:fedadam_ext": {1e-5: 3e-6},                             # "WA-ext 1e-5 -> 3e-6 (at c 0.1)"
    "item_lr_mult": {0.01: 0.001},
    "lora_rank": {16: 64, 4: 1},
    "c1_peak_lr": {1e-5: 3e-6, 3e-4: 1e-3},
}
MAX_EXTENSION_RUNS = 3                                                 # §5.7 "At most 3 extension runs per stage"
DROP_ORDER = ("DP1", "SECAGG_MAIN", "DPFTRL_S2027", "WC3/WC4", "WB2", "WC1/WC2", "WB3", "C2")       # §5.9, strict
DROP_ARMS = {"DP1": "P2_WSTAR_DP1", "SECAGG_MAIN": "P2_WSTAR_SECAGG on MAIN", "DPFTRL_S2027": "P2_WSTAR_DPFTRL8 s2027",
             "C2": "P2_C2_CFULL_SMALL"}
DROP_UNIT_CELLS = {"WC3/WC4": ("WC3", "WC4"), "WB2": ("WB2",), "WC1/WC2": ("WC1", "WC2"), "WB3": ("WB3",)}
CAP_PLAN_FORMAT = "PPSI_V4_P2_CAP_PLAN_1"
FORMATS = {"WA": "PPSI_V4_P2_WARM_SELECTION_A_1", "FINAL": "PPSI_V4_P2_WARM_SELECTION_FINAL_1",
           "C1": "PPSI_V4_P2_C1_SELECTION_1"}
DEFAULT_OUT = {"WA": V4 / "contracts" / "fl" / "P2_WARM_SELECTION_A.json",
               "FINAL": V4 / "contracts" / "fl" / "P2_WARM_SELECTION_FINAL.json",
               "C1": V4 / "contracts" / "fl" / "P2_C1_SELECTION.json"}
P2T3_RECEIPT_KEYS = {"test": "P2-T3", "verdict": "BIT_IDENTICAL", "rounds": 64}
METRIC = "CALIB_VALIDATION E2E client-macro MRR@20 at PRACTICAL_BEST (marks 0.5 .. 6.0; mark 0 never a candidate)"
TIE_ORDER_TEXT = ("(1) fewer upload bytes per visit, (2) fewer trained parameters, (3) the smaller effective server step "
                  "(FedAdam eta_s; FedAvgM eta_s / (1 - 0.9); FedAvg eta_s; exact rationals), (4) the smaller client LR "
                  "applied, (5) the lexically first label")
RULE_TEXT = {
    "WA": ("A* = the highest CALIB value over WA core (18) + WA-ext (2) + the incumbent (FedAdam 1e-3, c 0.1); "
           "non-finite runs excluded and recorded; ties best - value <= 0.001 (exact, inclusive) -> " + TIE_ORDER_TEXT +
           "; §5.7 edge rule — PREG2 (REGISTERED) §5.2 / §5.6 / §5.7"),
    "FINAL": ("W* = the highest CALIB value over {A*} + WB (3) + WC (5) minus the §5.9 drops (or {A*} alone under "
              "FUTILITY_SKIP_WB_WC, §5.8); non-finite runs excluded and recorded; ties best - value <= 0.001 -> "
              + TIE_ORDER_TEXT + "; §5.7 edge rule on a WB / WC winner — PREG2 (REGISTERED) §5.3-§5.9"),
    "C1": ("per variant (FULL / FROZEN) the highest CALIB value over peak LR {1e-5, 3e-5, 1e-4, 3e-4}; non-finite runs "
           "excluded and recorded; ties best - value <= 0.001 -> the smaller LR (bytes do not differ); §5.7 edge rule "
           "1e-5 -> 3e-6, 3e-4 -> 1e-3 — PREG2 (REGISTERED) §4.2 / §5.6 / §5.7")}
EDGE_RULE_TEXT = ("§5.7 (registered): a winner on the edge of a numeric axis -> that axis extended once, one step "
                  "outward, at the winner's other coordinates; at most 3 extension runs per stage; re-applied over the "
                  "union; the winner again on the new edge -> EDGE_UNRESOLVED; no second extension")
FUTILITY_DISCLOSURE = ("D-j: Futility uses CALIB headroom (32,768 users); MAIN has 131,072 users, so FUTILITY_SKIP_WB_WC "
                       "may under-read the headroom available on MAIN.")
AUTHORITY = (f"contracts/PREG2.md (REGISTERED, DL 411, sha256 {PREG2_SHA256}) §4.2, §5.1-§5.9, §6.3 S8, §11 D-j; "
             "company/plans/PHASE2_IMPLEMENTATION_PLAN.md P2-06; company/reviews/REV-P2-06.md")

# ------------------------------------------------------------------------------------------------ ADAPTER constants
P2_ARMS = ("P2_W",)            # = launch_fl.P2_ARMS (tested)
PHASE1_INCUMBENT_ARM = "FA_WARM_FROZEN"       # S8 (the only Phase-1 label admitted, and only via --incumbent-phase1)
RECIPE_KEYS = ("server_opt", "server_lr", "client_lr", "items", "item_lr_mult", "upload", "lora_rank",
               "lora_lr_factor", "personal_head")       # = launch_fl.P2_RECIPE_KEYS (tested)
P2_COMMON_FIELDS = (("role",), ("study",), ("cfg", "seed"), ("plan",), ("max_rounds",), ("cfg", "endpoint_rounds"),
                    ("cfg", "dropout_p"), ("cfg", "endpoint_efe"), ("cfg", "lr_shape"), ("cfg", "solver", "passes"),
                    ("cfg", "solver", "batch_size"), ("cfg", "solver", "clip"), ("cfg", "solver", "weight_decay"),
                    ("lr_peak_base",), ("exposure_budget",), ("p2_fixed",), ("init_from",))
INCUMBENT_COMPARED = ("cfg", "plan", "max_rounds", "frozen_item_tables", "init_from", "exposure_budget", "role")
# R9 (delta N1): the row-label key set lr_screen.read_run compares as a SET (lr_screen.py:366-369, copied verbatim; a test
# checks every key against the lr_screen source) + the optional P2 key of launch_fl.p2_row_labels: a key of this set
# present in the row but absent from the expected labels is refused, exactly as read_run refuses it.
READ_RUN_LABEL_KEYS = ("fl_server_opt", "fl_server_lr", "fl_server_tau", "fl_lr_shape", "fl_client_opt", "fl_client_lr",
                       "fl_local_passes", "fl_local_batch", "fl_init", "fl_init_run_id", "fl_init_checkpoint_sha256",
                       "fl_init_state_digest", "fl_warm_lr_factor", "fl_frozen_item_tables", "fl_warm_server_lr_factor",
                       "fl_warm_server_lr")
P2_OPTIONAL_ROW_KEYS = ("p2_item_lr_mult",)
C1_DIVERGED_ERROR_PREFIX = "TrainingDiverged"             # v4train.trainer: STATUS.json FAILED, error "<type>: <msg>"


class P2SelectionRefused(LR.SelectionRefused):
    pass


class P2EdgeRunsMissing(P2SelectionRefused):
    """A registered §5.7 extension is required and its runs are absent; `needed` = [{label, recipe, axis}]."""

    def __init__(self, msg: str, needed: Sequence[Mapping]):
        super().__init__(msg)
        self.needed = [dict(n) for n in needed]


# ------------------------------------------------------------------------------------------------ numbers
def _f(v, what: str) -> float:
    try:
        x = float(v)
    except (TypeError, ValueError):
        raise P2SelectionRefused(f"{what} {v!r} is not a number") from None
    if not math.isfinite(x) or x <= 0:
        raise P2SelectionRefused(f"{what} {v!r} must be a finite positive number")
    return x


def _tag(v) -> str:
    return format(float(v), "g")


def _frac(v) -> Fraction:
    """The exact decimal a recorded value denotes (Fraction of its shortest repr: the JSON text round-trips it), so
    registered thresholds compare exactly (0.1716 - 0.1706 == 1/1000; 0.03 / (1 - 0.9) == 3/10)."""
    x = float(v)
    if not math.isfinite(x):
        raise P2SelectionRefused(f"non-finite value {v!r} in an exact comparison")
    return Fraction(repr(x))


def _finite_json(x, path: str = ""):
    """(a JSON-safe copy of x, [paths of non-finite numbers]): NaN / +-inf -> "NaN" / "Infinity" / "-Infinity"."""
    if isinstance(x, float) and not math.isfinite(x):
        return ("NaN" if math.isnan(x) else ("Infinity" if x > 0 else "-Infinity")), [path or "."]
    if isinstance(x, Mapping):
        out, bad = {}, []
        for k, v in x.items():
            c, b = _finite_json(v, f"{path}.{k}" if path else str(k))
            out[k] = c
            bad += b
        return out, bad
    if isinstance(x, (list, tuple)):
        out, bad = [], []
        for i, v in enumerate(x):
            c, b = _finite_json(v, f"{path}[{i}]")
            out.append(c)
            bad += b
        return out, bad
    return x, []


# ------------------------------------------------------------------------------------------------ recipes and labels
def p2_recipe(*, server_opt, server_lr, client_lr, items="frozen", item_lr_mult=None, upload="dense", lora_rank=None,
              lora_lr_factor=None, personal_head=False) -> dict:
    """The normalised P2 warm-FL recipe (the selection-relevant coordinates; = launch_fl.check_p2_recipe's layout);
    structural combinations outside §5.2-§5.4 are refused (grid membership: the stage's registered cell sets)."""
    so = str(server_opt).lower()
    if so not in SERVER_OPTS:
        raise P2SelectionRefused(f"server_opt {server_opt!r} is not one of {SERVER_OPTS}")
    r = {"server_opt": so, "server_lr": _f(server_lr, "server_lr"), "client_lr": _f(client_lr, "client_lr"),
         "items": str(items), "item_lr_mult": None, "upload": str(upload), "lora_rank": None, "lora_lr_factor": None,
         "personal_head": bool(personal_head)}
    if r["items"] not in ("frozen", "unfrozen") or r["upload"] not in ("dense", "delta_rows"):
        raise P2SelectionRefused(f"items {items!r} / upload {upload!r} not registered")
    if personal_head not in (True, False):
        raise P2SelectionRefused(f"personal_head {personal_head!r} must be a boolean")
    if r["items"] == "unfrozen":
        r["item_lr_mult"] = _f(item_lr_mult, "item_lr_mult")
    elif item_lr_mult is not None or r["upload"] != "dense":
        raise P2SelectionRefused("item_lr_mult / delta_rows upload need unfrozen item tables (§5.3)")
    if lora_rank is not None or lora_lr_factor is not None:
        if lora_rank is None or lora_lr_factor is None or r["items"] != "frozen" or r["personal_head"]:
            raise P2SelectionRefused("a LoRA cell needs rank + LR factor, frozen item tables and no personal head (§5.4)")
        rk = _f(lora_rank, "lora_rank")
        if rk != int(rk):
            raise P2SelectionRefused(f"lora_rank {lora_rank!r} is not an integer")
        r["lora_rank"], r["lora_lr_factor"] = int(rk), _f(lora_lr_factor, "lora_lr_factor")
    if r["personal_head"] and r["items"] != "frozen":
        raise P2SelectionRefused("the personalised head (WC5) runs at A*'s frozen-table recipe (§5.4)")
    return r


def p2_label(r: Mapping) -> str:
    """'fedadam|eta=0.0003|c=0.01', '...|items=unfrozen|ilr=0.1', '...|up=delta_rows', '...|lora=r4|lrx=10', '...|pf'
    (= launch_fl.p2_label, tested)."""
    s = f"{r['server_opt']}|eta={_tag(r['server_lr'])}|c={_tag(r['client_lr'])}"
    if r["items"] == "unfrozen":
        s += f"|items=unfrozen|ilr={_tag(r['item_lr_mult'])}"
    if r["upload"] != "dense":
        s += f"|up={r['upload']}"
    if r["lora_rank"] is not None:
        s += f"|lora=r{int(r['lora_rank'])}|lrx={_tag(r['lora_lr_factor'])}"
    if r["personal_head"]:
        s += "|pf"
    return s


def c1_recipe(variant, peak_lr) -> dict:
    if variant not in C1_LABELS:
        raise P2SelectionRefused(f"C1 variant {variant!r} is not one of {sorted(C1_LABELS)}")
    return {"variant": variant, "peak_lr": _f(peak_lr, "peak_lr")}


def c1_label(r: Mapping) -> str:
    return f"{C1_LABELS[r['variant']]}|lr={_tag(r['peak_lr'])}"


def recipe_label(r: Mapping) -> str:
    return c1_label(r) if "variant" in r else p2_label(r)


def server_step(r: Mapping) -> Fraction:
    """§5.6 (3): FedAdam eta_s; FedAvgM eta_s / (1 - beta) with beta 0.9; FedAvg eta_s (exact)."""
    eta = _frac(r["server_lr"])
    return eta / (1 - FEDAVGM_BETA) if r["server_opt"] == "fedavgm" else eta


def client_lr_applied(r: Mapping) -> Fraction:
    """§5.6 (4): the client LR applied (c, or c x the LoRA LR factor of WC1 / WC2)."""
    c = _frac(r["client_lr"])
    return c * _frac(r["lora_lr_factor"]) if r.get("lora_rank") is not None else c


def p2_order_key(c: Mapping) -> tuple:
    """§5.6 tie preference inside the band (smaller = preferred): (1) upload bytes per visit, (2) trained parameters
    ([uploaded numel, 1 if a per-client private head is trained else 0], lexicographic), (3) effective server step,
    (4) client LR applied, (5) label. C1 (central): (1) 0, (2) trained numel, (3) 0, (4) peak LR, (5) label."""
    r = c["recipe"]
    tp = tuple(int(x) for x in c["trained_params"])
    if "variant" in r:
        return (0, tp, Fraction(0), _frac(r["peak_lr"]), c["label"])
    return (int(c["bytes_per_visit"]), tp, server_step(r), client_lr_applied(r), c["label"])


# ------------------------------------------------------------------------------------------------ registered cell sets
def wa_cells() -> dict:
    """label -> (set, recipe) of the REQUIRED WA runs: 18 core + 2 WA-ext + the incumbent (§5.2)."""
    out = {}
    for so, eta in WA_SERVER_POINTS:
        for c in WA_CLIENT_LRS:
            r = p2_recipe(server_opt=so, server_lr=eta, client_lr=c)
            out[p2_label(r)] = ("WA_CORE", r)
    for so, eta in WA_EXT_POINTS:
        r = p2_recipe(server_opt=so, server_lr=eta, client_lr=WA_EXT_CLIENT_LR)
        out[p2_label(r)] = ("WA_EXT", r)
    so, eta, c = INCUMBENT_CELL
    r = p2_recipe(server_opt=so, server_lr=eta, client_lr=c)
    out[p2_label(r)] = ("INCUMBENT", r)
    return out


def _a_star_base(a: Mapping) -> dict:
    if a["items"] != "frozen" or a["upload"] != "dense" or a["lora_rank"] is not None or a["personal_head"]:
        raise P2SelectionRefused(f"A* {p2_label(a)} is not a WA (frozen-table, dense) recipe")
    return {"server_opt": a["server_opt"], "server_lr": a["server_lr"], "client_lr": a["client_lr"]}


def final_cells(a_star: Mapping, dropped: Sequence[str] = (), futility_skip: bool = False) -> dict:
    """label -> (set, recipe) of the REQUIRED FINAL runs: A* + WB1-3 + WC1-5 at A*'s server / client recipe (§5.3 /
    §5.4), minus the cells of the declared §5.9 drops; {A*} alone under FUTILITY_SKIP_WB_WC (§5.8)."""
    base = _a_star_base(a_star)
    out = {p2_label(a_star): ("A_STAR", dict(a_star))}
    if futility_skip:
        return out
    gone = {cell for u in dropped if u in DROP_UNIT_CELLS for cell in DROP_UNIT_CELLS[u]}
    for name, kw in list(WB_CELLS.items()) + list(WC_CELLS.items()):
        if name in gone:
            continue
        r = p2_recipe(**base, **kw)
        out[p2_label(r)] = (name, r)
    return out


def c1_cells() -> dict:
    out = {}
    for v in C1_LABELS:
        for lr in C1_PEAK_LRS:
            r = c1_recipe(v, lr)
            out[c1_label(r)] = ("C1_SCREEN", r)
    return out


def p2_candidates(stage: str, *, a_star: Optional[Mapping] = None, dropped: Sequence[str] = (),
                  futility_skip: bool = False) -> dict:
    """The registered REQUIRED cells of a stage (label -> (set, recipe)); §5.6 candidate set before §5.7 extensions."""
    if stage == "WA":
        return wa_cells()
    if stage == "C1":
        return c1_cells()
    if stage == "FINAL":
        if a_star is None:
            raise P2SelectionRefused("FINAL needs A* (the verified P2_WARM_SELECTION_A winner)")
        return final_cells(a_star, dropped, futility_skip)
    raise P2SelectionRefused(f"stage {stage!r} is not one of {STAGES}")


def _wa_kind(r: Mapping) -> Optional[str]:
    if r["items"] != "frozen" or r["upload"] != "dense" or r["lora_rank"] is not None or r["personal_head"]:
        return None
    pt, c = (r["server_opt"], r["server_lr"]), r["client_lr"]
    if pt in WA_SERVER_POINTS and c in WA_CLIENT_LRS:
        return "WA_CORE"
    if pt in WA_EXT_POINTS and c == WA_EXT_CLIENT_LR:
        return "WA_EXT"
    if (r["server_opt"], r["server_lr"], c) == INCUMBENT_CELL:
        return "INCUMBENT"
    return None


def edge_extension(stage: str, winner: Mapping) -> list:
    """§5.7 (registered text): the extension cells of a base winner (recipe or candidate) — [{label, recipe, axis}],
    one per numeric axis on whose edge the winner sits, one step outward at the winner's other coordinates (no corner
    run: the registered text names none). WA core: the server axis of its optimiser (at c*) and the client axis (at
    eta*); WA-ext: 1e-5 -> 3e-6 at c 0.1 (1e-4 is interior: 1e-5 < 1e-4 < 1e-3 at c 0.1); the incumbent: no registered
    step; FINAL: WB item-LR multiplier 0.01 -> 0.001, LoRA rank 16 -> 64 / 4 -> 1 (A*, WB x0.1, WC5: none); C1: peak LR
    1e-5 -> 3e-6, 3e-4 -> 1e-3. A cell that is not a base cell (an extension cell) gets none: no second extension."""
    r = winner.get("recipe", winner)
    out = []
    if stage == "C1":
        new = EDGE_STEPS["c1_peak_lr"].get(r["peak_lr"])
        if new is not None:
            nr = c1_recipe(r["variant"], new)
            out.append({"label": c1_label(nr), "recipe": nr, "axis": "c1_peak_lr"})
        return out
    if stage == "WA":
        kind = _wa_kind(r)
        so, eta, c = r["server_opt"], r["server_lr"], r["client_lr"]
        cells = []
        if kind == "WA_CORE":
            s_new = EDGE_STEPS[f"server_lr:{so}"].get(eta)
            c_new = EDGE_STEPS["client_lr"].get(c)
            if s_new is not None:
                cells.append(("server_lr", s_new, c))
            if c_new is not None:
                cells.append(("client_lr", eta, c_new))
        elif kind == "WA_EXT":
            s_new = EDGE_STEPS["server_lr:fedadam_ext"].get(eta)
            if s_new is not None:
                cells.append(("server_lr", s_new, c))
        for axis, e, cc in cells:
            nr = p2_recipe(server_opt=so, server_lr=e, client_lr=cc)
            out.append({"label": p2_label(nr), "recipe": nr, "axis": axis})
        return out
    if stage == "FINAL":
        if r["items"] == "unfrozen" and r["upload"] == "dense":
            new = EDGE_STEPS["item_lr_mult"].get(r["item_lr_mult"])
            if new is not None:
                nr = dict(r, item_lr_mult=new)
                out.append({"label": p2_label(nr), "recipe": nr, "axis": "item_lr_mult"})
        elif r["lora_rank"] is not None:
            new = EDGE_STEPS["lora_rank"].get(r["lora_rank"])
            if new is not None:
                nr = dict(r, lora_rank=new)
                out.append({"label": p2_label(nr), "recipe": nr, "axis": "lora_rank"})
        return out
    raise P2SelectionRefused(f"stage {stage!r} is not one of {STAGES}")


def unstepped_edges(stage: str, winner_set: str) -> list:
    """Informational: axis ends of the base winner with no registered §5.7 step (no extension is run)."""
    notes = {("WA", "INCUMBENT"): "server LR 1e-3 is the top of the c = 0.1 axis (1e-5, 1e-4, 1e-3); §5.7 registers no "
                                  "step there and none for client LR 0.1",
             ("WA", "WA_EXT"): "client LR 0.1 has no registered step (§5.7)",
             ("FINAL", "A_STAR"): "A*'s axes were handled at stage WA (one extension only)",
             ("FINAL", "WB1"): "item-LR multiplier 0.1 is the top of the WB axis; §5.7 registers no step",
             ("FINAL", "WB3"): "item-LR multiplier 0.1 / delta upload: §5.7 registers no step",
             ("FINAL", "WC5"): "the personalised head has no numeric axis"}
    n = notes.get((stage, winner_set))
    return [n] if n else []


# ------------------------------------------------------------------------------------------------ the decision
def _anchor_value(anchor) -> float:
    a = anchor["value"] if isinstance(anchor, Mapping) else anchor
    try:
        a = float(a)
    except (TypeError, ValueError):
        raise P2SelectionRefused(f"anchor PRE_CALIB {a!r} is not a number") from None
    if not math.isfinite(a):
        raise P2SelectionRefused(f"anchor PRE_CALIB {a!r} is not finite")
    return a


def p2_decide(stage: str, runs: Sequence[Mapping], anchor) -> dict:
    """§5.6 over candidate records (label, recipe, value, excluded, bytes_per_visit, trained_params): excluded runs are
    never ranked; best = the highest finite value; the band = every finite candidate with best - value <= 0.001
    (exact decimal arithmetic, inclusive) in p2_order_key order; the winner = the band's first; CALIB_GAIN = winner -
    PRE_CALIB (exact), NO_MEASURABLE_CALIB_GAIN iff CALIB_GAIN < 0.001 (strict). C1: within one variant."""
    if stage not in STAGES:
        raise P2SelectionRefused(f"stage {stage!r} is not one of {STAGES}")
    labels = [c["label"] for c in runs]
    if len(set(labels)) != len(labels):
        raise P2SelectionRefused(f"each candidate at most once, got {sorted(labels)}")
    if stage == "C1" and len({c["recipe"].get("variant") for c in runs}) > 1:
        raise P2SelectionRefused("C1 is selected within each variant (FULL / FROZEN), never across them (§4.2)")
    fin = []
    for c in runs:
        if c.get("excluded"):
            continue
        v = c.get("value")
        if not isinstance(v, (int, float)) or isinstance(v, bool) or not math.isfinite(float(v)):
            raise P2SelectionRefused(f"{c['label']}: non-finite value {v!r} (it must be excluded, not ranked)")
        fin.append(c)
    if not fin:
        raise P2SelectionRefused(f"{stage}: no candidate with a finite value")
    a = _anchor_value(anchor)
    best = max(float(c["value"]) for c in fin)
    fb = _frac(best)
    band = sorted((c for c in fin if fb - _frac(c["value"]) <= TIE_FRAC), key=p2_order_key)
    top = min(fin, key=lambda c: (-float(c["value"]), p2_order_key(c)))
    w = band[0]
    gain = _frac(w["value"]) - _frac(a)
    return {"winner": w["label"], "winner_value": float(w["value"]), "tie_set": [c["label"] for c in band],
            "best_value": best, "best_value_label": top["label"], "tie_tolerance": TIE_CALIB,
            "anchor_value": a, "calib_gain": float(gain), "no_measurable_calib_gain": bool(gain < ANCHOR_FRAC),
            "anchor_min_gain": ANCHOR_MIN_GAIN, "n_ranked": len(fin),
            "excluded": sorted(c["label"] for c in runs if c.get("excluded"))}


def stage_outcome(stage: str, base: Sequence[Mapping], extra: Sequence[Mapping], anchor) -> dict:
    """§5.6 + §5.7 for one stage (one C1 variant): base = the registered required candidates, extra = every other run
    given. The base decision (d0) -> the registered extension of its winner; extra must be exactly the NEW extension
    cells (refused otherwise; absent -> P2EdgeRunsMissing). With an extension, the decision over base + extra is final;
    EDGE_UNRESOLVED iff the union winner is a registered extension cell ("again on the new edge"), else EDGE_RESOLVED;
    without one: NOT_ON_EDGE. Notes (no status effect): non-finite extension cells; a winner that moved to another base
    run."""
    d0 = p2_decide(stage, base, anchor)
    by = {c["label"]: c for c in base}
    w0 = by[d0["winner"]]
    reg = edge_extension(stage, w0)
    if len(reg) > MAX_EXTENSION_RUNS:
        raise P2SelectionRefused(f"{len(reg)} extension cells (> {MAX_EXTENSION_RUNS}; §5.7)")
    new = [e for e in reg if e["label"] not in by]
    existing = [e["label"] for e in reg if e["label"] in by]
    new_labels = [e["label"] for e in new]
    xl = [c["label"] for c in extra]
    if len(set(xl)) != len(xl) or set(xl) & set(by):
        raise P2SelectionRefused(f"{stage}: a configuration given twice: {sorted(xl)}")
    bad = sorted(set(xl) - set(new_labels))
    if bad:
        raise P2SelectionRefused(f"{stage}: runs not registered / not allowed by the §5.7 edge rule: {bad} (allowed now: "
                                 f"{sorted(new_labels)}; base winner {d0['winner']})")
    missing = [e for e in new if e["label"] not in xl]
    if missing:
        raise P2EdgeRunsMissing(f"{stage}: the base winner {d0['winner']} is on a registered edge; the §5.7 extension "
                                f"run(s) are missing (the file is written only with them): "
                                + "; ".join(f"{e['label']} [{e['axis']}]" for e in missing), missing)
    union = {c["label"]: c for c in list(base) + list(extra)}
    reg_labels = [e["label"] for e in reg]
    edge = {"rule": EDGE_RULE_TEXT, "winner_before_extension": d0["winner"],
            "registered_cells": [{"label": e["label"], "axis": e["axis"]} for e in reg],
            "new_cells": new_labels, "existing_cells": existing,
            "unstepped_edges": unstepped_edges(stage, w0.get("set", "")), "status": "NOT_ON_EDGE", "notes": []}
    if not reg:
        return {"base_decision": d0, "decision": d0, "edge": edge}
    d1 = p2_decide(stage, list(base) + list(extra), anchor)
    nonfin = sorted(lab for lab in reg_labels if union[lab].get("excluded"))
    if nonfin:
        edge["notes"].append(f"extension cell(s) {nonfin} non-finite: excluded, so that edge was not measured")
    if d1["winner"] != d0["winner"] and d1["winner"] not in reg_labels:
        edge["notes"].append(f"the union winner moved from {d0['winner']} to the base run {d1['winner']} (band "
                             "reshuffle); no second extension (§5.7)")
    edge["status"] = "EDGE_UNRESOLVED" if d1["winner"] in reg_labels else "EDGE_RESOLVED"
    return {"base_decision": d0, "decision": d1, "edge": edge}


def futility(c1_best, wa_best, anchor) -> dict:
    """§5.8: FUTILITY_SKIP_WB_WC iff C1_best - PRE_CALIB < 0.001 AND WA_best - PRE_CALIB < 0.001 (strict, exact)."""
    a = _anchor_value(anchor)
    vals = {}
    for k, v in (("c1_best", c1_best), ("wa_best", wa_best)):
        try:
            vals[k] = float(v)
        except (TypeError, ValueError):
            raise P2SelectionRefused(f"futility: {k} {v!r} is not a number") from None
        if not math.isfinite(vals[k]):
            raise P2SelectionRefused(f"futility: {k} {v!r} is not finite")
    c1_gain, wa_gain = _frac(vals["c1_best"]) - _frac(a), _frac(vals["wa_best"]) - _frac(a)
    skip = bool(c1_gain < FUT_FRAC and wa_gain < FUT_FRAC)
    return {"decision": "FUTILITY_SKIP_WB_WC" if skip else "CONTINUE_WB_WC", "skip_wb_wc": skip,
            "c1_best": vals["c1_best"], "wa_best": vals["wa_best"], "anchor_value": a, "c1_gain": float(c1_gain),
            "wa_gain": float(wa_gain), "threshold": FUTILITY_MIN, "disclosure": FUTILITY_DISCLOSURE,
            "rule": "PREG2 §5.8: skip WB and WC iff both gains < 0.001 (strict); A* = W* goes to MAIN"}


# ------------------------------------------------------------------------------------------------ §5.9 budget cap
def check_dropped(dropped: Sequence[str]) -> list:
    """§5.9: a declared drop set is valid iff it is a prefix of the ONE registered order DROP_ORDER (tertiary arms in
    their listed order, then the exploration cells, then C2 last). Returns it in drop order."""
    d = list(dropped)
    if len(set(d)) != len(d) or any(u not in DROP_ORDER for u in d):
        raise P2SelectionRefused(f"--dropped {d}: unknown or repeated unit (registered order: {list(DROP_ORDER)})")
    want = list(DROP_ORDER[:len(d)])
    if sorted(d) != sorted(want):
        raise P2SelectionRefused(f"--dropped {d}: not a prefix of the registered order {list(DROP_ORDER)} (PREG2 §5.9)")
    return want


def cap_drop_plan(projected_gpu_h: float, savings: Mapping[str, float], cap: float = CAP_GPU_H) -> dict:
    """§5.9: while the projected total exceeds the cap, drop the next unit of DROP_ORDER (savings[unit] = the GPU-h it
    saves; every unit must be given, 0 for a unit not planned). The result is always a prefix (check_dropped passes);
    the primary path is never dropped: within_cap False = it alone exceeds the cap."""
    missing = [u for u in DROP_ORDER if u not in savings]
    if missing:
        raise P2SelectionRefused(f"cap plan: savings missing for {missing} (every §5.9 unit, 0 if not planned)")
    total = float(projected_gpu_h)
    if not math.isfinite(total) or total < 0:
        raise P2SelectionRefused(f"cap plan: projected GPU-h {projected_gpu_h!r} invalid")
    out = []
    for u in DROP_ORDER:
        if total <= cap:
            break
        s = float(savings[u])
        if not math.isfinite(s) or s < 0:
            raise P2SelectionRefused(f"cap plan: savings[{u}] = {savings[u]!r} invalid")
        out.append(u)
        total -= s
    return {"dropped": out, "projected_gpu_h": float(projected_gpu_h), "projected_after": total, "cap": cap,
            "within_cap": bool(total <= cap), "status": "WITHIN_CAP" if total <= cap else "CAP_EXCEEDED_PRIMARY_PATH",
            "order": list(DROP_ORDER), "authority": "PREG2 §5.9"}


def read_cap_plan(path: Path) -> dict:
    """A cap-plan input {format PPSI_V4_P2_CAP_PLAN_1, projected_gpu_h, savings{unit: GPU-h}} (+ its sha256)."""
    obj = _read_json(Path(path), "cap plan")
    if not isinstance(obj, dict) or obj.get("format") != CAP_PLAN_FORMAT:
        raise P2SelectionRefused(f"{path} is not a {CAP_PLAN_FORMAT} object")
    plan = cap_drop_plan(obj.get("projected_gpu_h"), obj.get("savings") or {})
    return dict(plan, path=str(path), sha256=LR._sha(Path(path)), savings=dict(obj["savings"]))


# ------------------------------------------------------------------------------------------------ reading runs
def _read_json(p: Path, what: str) -> dict:
    try:
        return json.loads(Path(p).read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise P2SelectionRefused(f"{what} {p} unreadable ({e})") from e


def _get(d: Mapping, path: Sequence[str]):
    for k in path:
        if not isinstance(d, Mapping):
            return None
        d = d.get(k)
    return d


def _value_or_nan(v) -> float:
    if isinstance(v, bool):
        return float("nan")
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def p2_run_adapter(spec: Mapping, summary: Mapping, run_dir) -> dict:
    """THE ADAPTER to the P2-02 run format (align here, ONE edit; nothing else reads P2 recipe / bytes / D1-ext fields).
    Fields depended on (launch_fl sha recorded in every file; checked against the P2-01+02 handoff):
      * spec["p2_recipe"]: exactly RECIPE_KEYS (launch_fl.check_p2_recipe layout); spec["p2_label"] = p2_label(recipe);
      * spec cfg.solver.lr = the client LR applied (c; c x lora_lr_factor once P2-04 installs LoRA);
      * RUN_SUMMARY["p2"]["label"] / ["recipe"] = the spec's (launch_fl execute);
      * RUN_SUMMARY["bytes"]["per_visit"]["upload_bytes"] (P2-03 WB3 must publish its per-visit payload here) and
        ["shared_numel"] (the uploaded numel after the freeze; P2-04 LoRA: LoRA parameters only);
      * RUN_SUMMARY["update_norm_log"] (launch_fl.update_norm_summary; D1-ext, recorded, never selects);
      * P2_COMMON_FIELDS of the spec (identical across the P2 runs of a file).
    Returns {recipe, bytes_per_visit, trained_params, d1ext (raw), common}."""
    raw = spec.get("p2_recipe")
    if not isinstance(raw, Mapping) or sorted(raw) != sorted(RECIPE_KEYS):
        raise P2SelectionRefused(f"{run_dir}: arm spec p2_recipe {raw!r} is not a {RECIPE_KEYS} record")
    recipe = p2_recipe(**{k: raw[k] for k in RECIPE_KEYS})
    if spec.get("p2_label") != p2_label(recipe):
        raise P2SelectionRefused(f"{run_dir}: spec p2_label {spec.get('p2_label')!r} != {p2_label(recipe)!r}")
    want = float(client_lr_applied(recipe))
    got = _get(spec, ("cfg", "solver", "lr"))
    if not isinstance(got, (int, float)) or not math.isclose(float(got), want, rel_tol=1e-12, abs_tol=0.0):
        raise P2SelectionRefused(f"{run_dir}: cfg solver LR {got!r} != the recipe's client LR applied {want}")
    p2s = summary.get("p2") or {}
    if p2s.get("label") != p2_label(recipe) or p2s.get("recipe") != dict(raw):
        raise P2SelectionRefused(f"{run_dir}: RUN_SUMMARY p2 label / recipe differ from the arm spec")
    pv = _get(summary, ("bytes", "per_visit")) or {}
    up, sh = pv.get("upload_bytes"), pv.get("shared_numel")
    if not (isinstance(up, int) and isinstance(sh, int)) or isinstance(up, bool) or isinstance(sh, bool):
        raise P2SelectionRefused(f"{run_dir}: RUN_SUMMARY bytes.per_visit upload_bytes / shared_numel missing")
    return {"recipe": recipe, "bytes_per_visit": up, "trained_params": [sh, 1 if recipe["personal_head"] else 0],
            "d1ext": summary.get("update_norm_log"),
            "common": {".".join(p): _get(spec, p) for p in P2_COMMON_FIELDS}}


def _candidate(rec: Mapping, *, recipe: Mapping, bytes_per_visit, trained_params, d1ext, source: str) -> dict:
    d1, bad = _finite_json(d1ext, "d1ext")
    mx, bad2 = _finite_json(rec.get("max_of_12_marks"), "max_of_12_marks")
    return {"label": recipe_label(recipe), "recipe": dict(recipe), "value": None if rec.get("excluded") else rec["value"],
            "excluded": bool(rec.get("excluded")), "reason": rec.get("reason"),
            "practical_best_mark": rec.get("practical_best_mark"), "max_of_12_marks": mx,
            "bytes_per_visit": bytes_per_visit, "trained_params": list(trained_params), "d1ext": d1,
            "nonfinite_fields": sorted(set(bad + bad2 + list(rec.get("nonfinite_fields") or []))),
            "source": source, "arm": rec.get("arm"), "run_dir": rec.get("run_dir"), "run_id": rec.get("run_id"),
            "arm_spec_sha256": rec.get("arm_spec_sha256"), "result_rows_sha256": rec.get("result_rows_sha256"),
            "run_summary_sha256": rec.get("run_summary_sha256"), "eval_manifest_id": rec.get("eval_manifest_id"),
            "split": rec.get("split"), "device": rec.get("device")}


def read_p2_run(run_dir: Path) -> dict:
    """One verified P2 warm-FL CALIB run: lr_screen.read_run's checks R1-R12 (module docstring), P2 counterparts for the
    two Phase-1-specific ones, ARM_SPEC verified before any value-bearing file is opened; then p2_run_adapter."""
    import v4int.launch_fl as F
    rd = Path(run_dir)
    side = _read_json(rd / "ARM_SPEC.json", "ARM_SPEC")                                                    # R1
    spec = side.get("arm_spec") if isinstance(side, dict) else None
    if not isinstance(spec, dict):
        raise P2SelectionRefused(f"{rd}: ARM_SPEC.json carries no arm_spec")
    digest = F.spec_digest(spec)
    arm = spec.get("arm")
    if side.get("arm_spec_sha256") != digest or side.get("arm") != arm:
        raise P2SelectionRefused(f"{rd}: ARM_SPEC arm / arm_spec_sha256 do not match its arm_spec")
    if arm not in P2_ARMS or spec.get("role") != "CALIBRATION" or _get(spec, ("cfg", "seed")) != F.CALIB["seed"]:   # R2
        raise P2SelectionRefused(f"{rd}: not a {'/'.join(P2_ARMS)} / CALIBRATION / seed {F.CALIB['seed']} run "
                                 f"(arm {arm!r}, role {spec.get('role')!r}); nothing value-bearing was opened")
    try:                                                                                                   # R3
        F.check_p2_spec(spec)
    except F.LaunchRefused as e:
        raise P2SelectionRefused(f"{rd}: {e}") from e
    summ = _read_json(rd / "RUN_SUMMARY.json", "RUN_SUMMARY")                                              # R4
    rr = _read_json(rd / "RESULT_ROWS.json", "RESULT_ROWS")
    if summ.get("status") != "DONE_MAIN":
        raise P2SelectionRefused(f"{rd}: status {summ.get('status')} is not DONE_MAIN")
    if summ.get("arm_spec_sha256") != digest or F.spec_digest(summ.get("arm_spec") or {}) != digest:        # R5
        raise P2SelectionRefused(f"{rd}: RUN_SUMMARY arm spec digest differs from ARM_SPEC.json")
    ad = p2_run_adapter(spec, summ, rd)
    evals = summ.get("evals") or []
    if [e[0] for e in evals] != list(range(1, LR.EVAL_MARKS + 1)):                                         # R6
        raise P2SelectionRefused(f"{rd}: expected the 12 evaluations at marks 0.5 .. 6.0")
    hits = [r for r in rr.get("rows", []) if r.get("selection_rule") == "PRACTICAL_BEST"                   # R7
            and r.get("aggregation") == "macro" and r.get("target_denominator") == "E2E"
            and r.get("metric") == "mrr" and _value_or_nan(r.get("k")) == 20]
    if len(hits) != 1:
        raise P2SelectionRefused(f"{rd}: {len(hits)} PRACTICAL_BEST E2E macro mrr@20 rows (expected exactly 1)")
    row = hits[0]
    if row.get("eval_manifest_id") != CALIB_EVAL_MANIFEST:
        raise P2SelectionRefused(f"{rd}: evaluated on {row.get('eval_manifest_id')!r}, not {CALIB_EVAL_MANIFEST} (the "
                                 "selection reads CALIB only; MAIN / TEST refused)")
    lab = F.CALIB["run_label"]
    ident = {"role": "CALIBRATION", "run_label": lab, "method": arm, "arm_spec_sha256": digest}             # R8
    bad = {k: row.get(k) for k, want in ident.items() if row.get(k) != want}
    if bad or _value_or_nan(row.get("fl_lr_mult")) != 1.0:
        raise P2SelectionRefused(f"{rd}: the PRACTICAL_BEST row is not a verified P2 CALIB row ({bad})")
    want_rl = dict(F.warm_row_labels(spec), **F.p2_row_labels(spec))                                      # R9
    want_rl.update(row_status=lab, comparison_status=lab)                     # launch_fl.fl_rows' CALIBRATION relabel
    bad = {k: (row.get(k), v) for k, v in want_rl.items() if row.get(k) != v}
    extra = sorted(k for k in READ_RUN_LABEL_KEYS + P2_OPTIONAL_ROW_KEYS if k in row and k not in want_rl)
    if bad or extra:                                                       # delta N1: a set comparison, as read_run
        raise P2SelectionRefused(f"{rd}: the PRACTICAL_BEST row's labels differ from the spec: {bad}; labels not in the "
                                 f"spec: {extra}")
    marks = [_value_or_nan(e[2]) for e in evals]
    fin_marks = [m for m in marks if math.isfinite(m)]
    rec = {"run_dir": str(rd), "run_id": rr.get("run_id"), "arm": arm, "device": side.get("device"),
           "eval_manifest_id": row.get("eval_manifest_id"), "split": row.get("split"), "arm_spec_sha256": digest,
           "result_rows_sha256": LR._sha(rd / "RESULT_ROWS.json"), "run_summary_sha256": LR._sha(rd / "RUN_SUMMARY.json"),
           "max_of_12_marks": max(fin_marks) if fin_marks else None,
           "nonfinite_fields": [f"evals[{i}]" for i, m in enumerate(marks) if not math.isfinite(m)]}
    value = _value_or_nan(row.get("value"))
    if not math.isfinite(value):                                                                           # R10
        rec.update(excluded=True, reason=f"non-finite PRACTICAL_BEST value {row.get('value')!r} "
                                         f"(value_status {row.get('value_status')!r})")
    else:
        if row.get("value_status") != "OK":                                                                # R11
            raise P2SelectionRefused(f"{rd}: the PRACTICAL_BEST row has value_status {row.get('value_status')!r}")
        best_u = summ.get("best_half_efe")                                                                 # R12
        at_best = [_value_or_nan(e[2]) for e in evals if e[0] == best_u]
        sm = _value_or_nan(row.get("schedule_mark"))                       # delta N2: missing / null -> NaN -> refused
        if (len(at_best) != 1 or at_best[0] != value or not isinstance(best_u, int) or not math.isfinite(sm)
                or abs(sm - float(best_u) / 2) > 1e-12):
            raise P2SelectionRefused(f"{rd}: the PRACTICAL_BEST row does not match the run's evaluation at its best "
                                     "mark")
        rec.update(value=value, practical_best_mark=float(best_u) / 2)
    c = _candidate(rec, recipe=ad["recipe"], bytes_per_visit=ad["bytes_per_visit"],
                   trained_params=ad["trained_params"], d1ext=ad["d1ext"], source="P2_RUN")
    c["_common"], c["_spec"] = ad["common"], spec
    return c


def check_p2t3_receipt(receipt: Optional[Path], rec: Mapping) -> dict:
    """§5.2 "reused only if" P2-T3 is bit-identical: the receipt is a JSON object with test P2-T3, verdict
    BIT_IDENTICAL, rounds 64, s8tc_run_id = the run read, and its RESULT_ROWS / RUN_SUMMARY sha256 (alignment point with
    the P2-02 P2-T3 test's output)."""
    rp = Path(receipt) if receipt is not None else None
    if rp is None or not rp.is_file():
        raise P2SelectionRefused("--incumbent-phase1 needs --p2t3-receipt (§5.2: reuse only if P2-T3 is bit-identical)")
    obj = _read_json(rp, "P2-T3 receipt")
    if not isinstance(obj, dict):
        raise P2SelectionRefused(f"{rp}: not a receipt object")
    bad = {k: obj.get(k) for k, v in P2T3_RECEIPT_KEYS.items() if obj.get(k) != v}
    if obj.get("s8tc_run_id") != rec.get("run_id"):
        bad["s8tc_run_id"] = obj.get("s8tc_run_id")
    for k in ("result_rows_sha256", "run_summary_sha256"):
        if obj.get(k) != rec.get(k):
            bad[k] = obj.get(k)
    if bad:
        raise P2SelectionRefused(f"{rp}: not a P2-T3 BIT_IDENTICAL receipt of {rec.get('run_dir')} ({bad})")
    return {"path": str(rp), "sha256": LR._sha(rp), **{k: obj[k] for k in ("test", "verdict", "rounds", "s8tc_run_id")}}


def read_phase1_incumbent(run_dir: Path, receipt: Path) -> dict:
    """§5.2: S8's f1/sf0.1 run (FedAdam eta 0.01 x server factor 0.1 = 1e-3, client SGD-m 0.1, FL_CONST, frozen tables)
    reused as the incumbent ONLY with the P2-T3 receipt; read by the unedited lr_screen.read_run(arms=(FA_WARM_FROZEN,)).
    Its cfg / plan / rounds / theta_pre are compared with p2_arm_spec of the incumbent recipe in _same_common."""
    if receipt is None or not Path(receipt).is_file():
        raise P2SelectionRefused("--incumbent-phase1 needs --p2t3-receipt (§5.2: reuse only if P2-T3 is bit-identical)")
    rec = LR.read_run(Path(run_dir), arms=(PHASE1_INCUMBENT_ARM,))
    if rec.get("eval_manifest_id") != CALIB_EVAL_MANIFEST:
        raise P2SelectionRefused(f"{run_dir}: not evaluated on {CALIB_EVAL_MANIFEST}")
    spec, t = rec["_spec"], rec["tuning"]
    ok = (t["optimizer"] == "fedadam" and t["server_lr"] == 0.01 and t["client_opt"] == "sgd_m"
          and t["client_lr"] == 0.1 and t["lr_shape"] == "fl_const" and rec["warm_lr_factor"] == 1.0
          and rec["warm_server_lr_factor"] == 0.1 and spec.get("frozen_item_tables"))
    if not ok:
        raise P2SelectionRefused(f"{run_dir}: not S8's f=1|sf=0.1 cell (FedAdam 1e-3, client SGD-m 0.1, frozen tables)")
    rcpt = check_p2t3_receipt(receipt, rec)
    summ = _read_json(Path(run_dir) / "RUN_SUMMARY.json", "RUN_SUMMARY")
    if LR._sha(Path(run_dir) / "RUN_SUMMARY.json") != rec["run_summary_sha256"]:
        raise P2SelectionRefused(f"{run_dir}: RUN_SUMMARY.json changed while it was read")
    pv = _get(summ, ("bytes", "per_visit")) or {}
    up, sh = pv.get("upload_bytes"), pv.get("shared_numel")
    if not (isinstance(up, int) and isinstance(sh, int)):
        raise P2SelectionRefused(f"{run_dir}: RUN_SUMMARY bytes.per_visit missing")
    so_, eta, c = INCUMBENT_CELL
    c_ = _candidate(rec, recipe=p2_recipe(server_opt=so_, server_lr=eta, client_lr=c), bytes_per_visit=up,
                    trained_params=[sh, 0], d1ext=summ.get("update_norm_log"), source="PHASE1_INCUMBENT")
    c_["p2t3_receipt"] = rcpt
    c_["_common"], c_["_spec"] = None, spec
    return c_


def read_c1_run(run_dir: Path) -> dict:
    """One verified C1 CALIB screen run, delegated to the P2-05 verifier v4int.central_ft.read_c1_run (RUN_START identity
    role CALIBRATION + CALIB manifest checked before any value-bearing file is opened; sealed DONE_MAIN, 12 marks, one
    PRACTICAL_BEST E2E macro mrr@20 control row equal to the run's evaluation at its PB mark >= 0.5); a non-finite value
    -> excluded and recorded. THE C1 ADAPTER = the mapping of its record below (align here)."""
    from v4int import central_ft as CF
    try:
        r = CF.read_c1_run(Path(run_dir))
    except CF.CentralFTRefused as e:
        div = c1_nonfinite_record(Path(run_dir))                 # delta open item: a diverged C1 run is EXCLUDED
        if div is None:
            raise P2SelectionRefused(f"C1: {e}") from e
        r = dict(div, refused_by_central_ft=str(e))
    if r.get("seed") != CALIB_SEED or r.get("role") != "CALIBRATION":
        raise P2SelectionRefused(f"{run_dir}: C1 seed {r.get('seed')} / role {r.get('role')} (CALIB screen: 2026)")
    tn = r.get("trained_numel")
    if not isinstance(tn, int) or isinstance(tn, bool):
        raise P2SelectionRefused(f"{run_dir}: no trained numel in the C1 record")
    rec = {"run_dir": r["run_dir"], "run_id": r.get("run_id"), "arm": r.get("arm"), "device": None,
           "eval_manifest_id": CALIB_EVAL_MANIFEST, "split": None, "result_rows_sha256": r.get("result_rows_sha256"),
           "run_summary_sha256": None, "arm_spec_sha256": r.get("run_start_sha256")}
    if r.get("finite"):
        rec.update(value=float(r["value"]), practical_best_mark=float(r["mark"]))
    else:
        rec.update(excluded=True, reason=r.get("reason") or "non-finite PRACTICAL_BEST value (central_ft.read_c1_run "
                                                           "finite False)")
    c = _candidate(rec, recipe=c1_recipe(r.get("variant"), r.get("peak_lr")), bytes_per_visit=0,
                   trained_params=[tn, 0], d1ext=None, source="C1_RUN")
    c["run_start_sha256"] = r.get("run_start_sha256")
    if r.get("refused_by_central_ft"):
        c["c1_nonfinite_path"] = {"kind": r["kind"], "central_ft_refusal": r["refused_by_central_ft"],
                                  "status_sha256": r.get("status_sha256")}
    return c


def c1_nonfinite_record(rd: Path) -> Optional[dict]:
    """Delta open item (REV-P2-06 O-A; central_ft frozen at 4d95d12c, not edited): when central_ft.read_c1_run refuses
    a run, it is still recordable as EXCLUDED (§5.6 via §4.2) iff it is a verified C1 CALIB screen run — the RUN_START
    identity checks of central_ft.read_c1_run repeated here (p2_c1 arm / variant = run id, role CALIBRATION, eval
    manifest CALIB_VALIDATION, seed 2026), value-free, BEFORE anything else is opened — AND either
      (a) DIVERGED: STATUS.json status FAILED with error "TrainingDiverged: ..." (v4train.trainer) and no DONE_MAIN
          RUN_SUMMARY (the run never finished, so it has no value), or
      (b) NON-FINITE: a DONE_MAIN RUN_SUMMARY that only non-strict JSON parses (central_ft refuses NaN) whose
          practical_best.value is non-finite.
    Anything else returns None (the caller refuses)."""
    from v4int import central_ft as CF
    try:
        start = CF._strict(rd / "RUN_START.json")
    except (OSError, ValueError):
        return None
    ident = (start.get("p2_c1") or {}).get("identity") or {}
    if (ident.get("arm") not in CF.C1_ARMS.values() or CF.c1_variant_of_run_id(start.get("run_id")) != ident.get("variant")
            or CF.C1_ARMS.get(ident.get("variant")) != ident.get("arm") or ident.get("role") != "CALIBRATION"
            or (start.get("eval_provider") or {}).get("manifest_id") != CF.CALIB_EVAL_ID
            or start.get("seed") != CALIB_SEED):
        return None
    art = rd / "sealed" if (rd / "sealed").is_dir() else rd
    base = {"run_dir": str(rd), "run_id": start.get("run_id"), "arm": ident["arm"], "variant": ident["variant"],
            "role": ident["role"], "peak_lr": ident.get("peak_lr"), "seed": start.get("seed"),
            "trained_numel": (start.get("p2_c1") or {}).get("trained_numel"), "value": None, "finite": False,
            "mark": None, "run_start_sha256": LR._sha(rd / "RUN_START.json"),
            "result_rows_sha256": _file_sha(art / "RESULT_ROWS.json")}
    summ = None
    if (art / "RUN_SUMMARY.json").is_file():
        try:
            summ = json.loads((art / "RUN_SUMMARY.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
    done = isinstance(summ, dict) and summ.get("status") == "DONE_MAIN"
    status = None
    if (rd / "STATUS.json").is_file():
        try:
            status = json.loads((rd / "STATUS.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            status = None
    err = str((status or {}).get("error") or "")
    if not done and (status or {}).get("status") == "FAILED" and err.startswith(C1_DIVERGED_ERROR_PREFIX):
        return dict(base, kind="DIVERGED", reason=f"diverged (STATUS FAILED: {err})",
                    status_sha256=LR._sha(rd / "STATUS.json"))
    if done and (summ.get("p2_c1") or {}).get("arm") == ident["arm"]:
        v = _value_or_nan((summ.get("practical_best") or {}).get("value"))
        if not math.isfinite(v):
            return dict(base, kind="NON_FINITE", reason="non-finite PRACTICAL_BEST value (RUN_SUMMARY practical_best "
                                                        f"{(summ.get('practical_best') or {}).get('value')!r})",
                        status_sha256=_file_sha(rd / "STATUS.json"))
    return None


def anchor_adapter(rows_obj: Mapping, out_dir: Path) -> dict:
    """THE ANCHOR ADAPTER to the P2-07 reeval_p2 output (align here, one edit; P2-07 not yet written). Assumed
    (v4int.reeval conventions): <out>/RESULT_ROWS.json {role, rows}, rows relabelled method = P2_ANCHOR_PRE_CALIB;
    exactly one E2E macro mrr@20 row on CALIB_VALIDATION; <out>/REEVAL_RECORD.json present (sha recorded)."""
    if rows_obj.get("role") not in (None, "CALIBRATION"):
        raise P2SelectionRefused(f"{out_dir}: anchor role {rows_obj.get('role')!r} is not CALIBRATION")
    hits = [r for r in rows_obj.get("rows", []) if r.get("method") == "P2_ANCHOR_PRE_CALIB"
            and r.get("aggregation") == "macro" and r.get("target_denominator") == "E2E" and r.get("metric") == "mrr"
            and _value_or_nan(r.get("k")) == 20]
    if len(hits) != 1:
        raise P2SelectionRefused(f"{out_dir}: {len(hits)} P2_ANCHOR_PRE_CALIB E2E macro mrr@20 rows (expected 1)")
    return hits[0]


def read_anchor(out_dir: Path) -> dict:
    """PRE_CALIB (§4.1 C0): theta_pre on CALIB_VALIDATION (P2-07). Must be finite, value_status OK, CALIB manifest."""
    od = Path(out_dir)
    rows_obj = _read_json(od / "RESULT_ROWS.json", "anchor RESULT_ROWS")
    if not (od / "REEVAL_RECORD.json").is_file():
        raise P2SelectionRefused(f"{od}: REEVAL_RECORD.json missing")
    row = anchor_adapter(rows_obj, od)
    if row.get("eval_manifest_id") != CALIB_EVAL_MANIFEST:
        raise P2SelectionRefused(f"{od}: anchor evaluated on {row.get('eval_manifest_id')!r}, not {CALIB_EVAL_MANIFEST}")
    if row.get("value_status", "OK") != "OK":
        raise P2SelectionRefused(f"{od}: anchor value_status {row.get('value_status')!r}")
    v = _value_or_nan(row.get("value"))
    if not math.isfinite(v):
        raise P2SelectionRefused(f"{od}: anchor PRE_CALIB is non-finite: no selection is written (Ops decision)")
    return {"value": v, "path": str(od), "result_rows_sha256": LR._sha(od / "RESULT_ROWS.json"),
            "reeval_record_sha256": LR._sha(od / "REEVAL_RECORD.json"), "eval_manifest_id": row.get("eval_manifest_id")}


# ------------------------------------------------------------------------------------------------ files
def _file_sha(p: Path) -> Optional[str]:
    return LR._sha(p) if Path(p).is_file() else None


def _tool_record() -> dict:
    return {"p2_select_sha256": _file_sha(Path(__file__)), "lr_screen_sha256": _file_sha(HERE / "lr_screen.py"),
            "launch_fl_sha256": _file_sha(HERE / "launch_fl.py"), "central_ft_sha256": _file_sha(HERE / "central_ft.py"),
            "preg2_sha256_registered": PREG2_SHA256}


_PRIVATE = ("_common", "_spec")


def _dump(c: Mapping, set_name: str) -> dict:
    d = {k: v for k, v in c.items() if k not in _PRIVATE}
    d["set"] = set_name
    if "variant" not in c["recipe"]:
        d["server_step"] = {"exact": str(server_step(c["recipe"])), "float": float(server_step(c["recipe"]))}
        d["client_lr_applied"] = {"exact": str(client_lr_applied(c["recipe"])),
                                  "float": float(client_lr_applied(c["recipe"]))}
    return d


def _partition(stage: str, cands: Sequence[dict], required: Mapping) -> tuple:
    """(base candidates with their set, extra candidates): labels unique, recipes = the registered cell's, every
    required cell present."""
    labels = [c["label"] for c in cands]
    dup = sorted({x for x in labels if labels.count(x) > 1})
    if dup:
        raise P2SelectionRefused(f"{stage}: configuration(s) given twice: {dup}")
    missing = sorted(set(required) - set(labels))
    if missing:
        raise P2SelectionRefused(f"{stage}: required run(s) missing: {missing}")
    base, extra = [], []
    for c in cands:
        if c["label"] in required:
            st, r = required[c["label"]]
            if c["recipe"] != r:
                raise P2SelectionRefused(f"{c['run_dir']}: recipe {c['recipe']} != the registered cell {r}")
            base.append(dict(c, set=st))
        else:
            extra.append(dict(c, set=f"{stage}_EDGE"))
    return base, extra


def _same_eval(cands: Sequence[Mapping]) -> dict:
    if len({(c["eval_manifest_id"], c["split"]) for c in cands}) != 1:
        raise P2SelectionRefused("the runs were not evaluated on one manifest")
    devs = {json.dumps(c.get("device"), sort_keys=True) for c in cands}
    if len(devs) != 1:
        raise P2SelectionRefused(f"the runs do not share one evaluation device class / pin: {sorted(devs)}")
    return {"eval_manifest_id": cands[0]["eval_manifest_id"], "split": cands[0]["split"], "device": cands[0]["device"]}


def _same_common(cands: Sequence[Mapping]) -> Optional[dict]:
    """P2 runs differ only in their recipe (P2_COMMON_FIELDS); a Phase-1 incumbent must equal p2_arm_spec of the
    incumbent recipe at the P2 runs' theta_pre / seed / role / peak on INCUMBENT_COMPARED (cfg, plan, rounds, frozen
    tables, theta_pre, exposure budget, role)."""
    import v4int.launch_fl as F
    p2 = [c for c in cands if c["source"] == "P2_RUN"]
    if not p2:
        return None
    ref = p2[0]["_common"]
    for c in p2[1:]:
        if c["_common"] != ref:
            diff = sorted(k for k in ref if ref[k] != c["_common"].get(k))
            raise P2SelectionRefused(f"{c['run_dir']}: differs from {p2[0]['run_dir']} beyond the recipe: {diff}")
    for c in cands:
        if c["source"] != "PHASE1_INCUMBENT":
            continue
        rs = p2[0]["_spec"]
        so, eta, cl = INCUMBENT_CELL
        try:
            want = F.p2_arm_spec(P2_ARMS[0], recipe={"server_opt": so, "server_lr": eta, "client_lr": cl},
                                 seed=rs["cfg"]["seed"], peak_lr=rs["lr_peak_base"], role=rs["role"],
                                 init_from=rs.get("init_from"))
        except F.LaunchRefused as e:
            raise P2SelectionRefused(f"the incumbent's P2 reference spec cannot be built ({e})") from e
        bad = [k for k in INCUMBENT_COMPARED if c["_spec"].get(k) != want.get(k)]
        if bad:
            raise P2SelectionRefused(f"{c['run_dir']}: the Phase-1 incumbent differs from the P2 path's incumbent spec on "
                                     f"{bad} (theta_pre / cfg / rounds must be identical for reuse, §5.2)")
    return ref


def _decision_record(out: Mapping) -> dict:
    d = out["decision"]
    return {"winner": d["winner"], "tie_set": d["tie_set"], "best_value": d["best_value"],
            "best_value_label": d["best_value_label"], "winner_value": d["winner_value"],
            "calib_gain": d["calib_gain"], "no_measurable_calib_gain": d["no_measurable_calib_gain"],
            "base_decision": {k: out["base_decision"][k] for k in ("winner", "tie_set", "best_value_label")},
            "edge": out["edge"]}


def _flags(edge_statuses: Sequence[str], no_gain: bool, extra: Sequence[str] = ()) -> list:
    f = []
    if "EDGE_UNRESOLVED" in edge_statuses:
        f.append("EDGE_UNRESOLVED")
    if no_gain:
        f.append("NO_MEASURABLE_CALIB_GAIN")
    return f + list(extra)


def _nonfinite_diag(cands: Sequence[Mapping]) -> list:
    return sorted(c["label"] for c in cands if c.get("nonfinite_fields"))


def _base_obj(stage: str, anchor: Mapping) -> dict:
    return {"format": FORMATS[stage], "stage": stage, "task": "P2-06 v4int.p2_select", "rule": RULE_TEXT[stage],
            "metric": METRIC, "tie_tolerance": TIE_CALIB, "tie_band_rule": "best - value <= 0.001 (exact, inclusive)",
            "tie_order": TIE_ORDER_TEXT, "anchor_min_gain": ANCHOR_MIN_GAIN,
            "anchor_rule": "NO_MEASURABLE_CALIB_GAIN iff CALIB_GAIN < 0.001 (exact, strict)", "anchor": dict(anchor),
            "authority": AUTHORITY, "tool": _tool_record(),
            "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}


def _finish(obj: dict) -> dict:
    import v4int.launch_fl as F
    try:
        obj["self_sha256"] = F._canon(obj)
    except (TypeError, ValueError) as e:                   # cannot happen after _finite_json; fail closed anyway
        raise P2SelectionRefused(f"the selection object cannot be canonicalised ({e})") from e
    return obj


def _excluded(cands):
    return [{k: c[k] for k in ("label", "run_dir", "reason", "result_rows_sha256")}
            for c in sorted(cands, key=lambda c: c["label"]) if c["excluded"]]


def selection_c1(run_dirs: Sequence[Path], anchor_dir: Path) -> dict:
    anchor = read_anchor(anchor_dir)
    cands = [read_c1_run(d) for d in run_dirs]
    ev = _same_eval(cands)
    base, extra = _partition("C1", cands, c1_cells())
    variants, all_c = {}, []
    for v in C1_LABELS:
        b = [c for c in base if c["recipe"]["variant"] == v]
        x = [c for c in extra if c["recipe"]["variant"] == v]
        out = stage_outcome("C1", b, x, anchor)
        win = next(c for c in b + x if c["label"] == out["decision"]["winner"])
        variants[v] = dict(_decision_record(out), winner_peak_lr=win["recipe"]["peak_lr"], run_label=C1_LABELS[v],
                           winner_run_dir=win["run_dir"], winner_result_rows_sha256=win["result_rows_sha256"])
        all_c += [_dump(c, c["set"]) for c in b + x]
    fin = [c for c in cands if not c["excluded"]]
    obj = _base_obj("C1", anchor)
    obj.update(candidates=sorted(all_c, key=lambda c: c["label"]), variants=variants,
               values={c["label"]: c["value"] for c in sorted(fin, key=lambda c: c["label"])},
               excluded=_excluded(cands), required=sorted(c1_cells()),
               c1_best=max(c["value"] for c in fin), c1_best_label=min(fin, key=lambda c: (-c["value"], c["label"]))["label"],
               flags=_flags([variants[v]["edge"]["status"] for v in C1_LABELS], False),
               nonfinite_diagnostics=_nonfinite_diag(cands),
               applies_to="P2-05 central_ft: MAIN runs bind the winner peak LR of their variant", **ev)
    obj["c1_gain"] = float(_frac(obj["c1_best"]) - _frac(anchor["value"]))
    return _finish(obj)


def selection_wa(run_dirs: Sequence[Path], anchor_dir: Path, incumbent_phase1: Optional[Path] = None,
                 p2t3_receipt: Optional[Path] = None) -> dict:
    anchor = read_anchor(anchor_dir)
    cands = [read_p2_run(d) for d in run_dirs]
    inc = read_phase1_incumbent(incumbent_phase1, p2t3_receipt) if incumbent_phase1 is not None else None
    if inc is not None:
        if inc["label"] in {c["label"] for c in cands}:
            raise P2SelectionRefused("the incumbent is given twice (a P2 re-run and --incumbent-phase1)")
        cands.append(inc)
    ev = _same_eval(cands)
    common = _same_common(cands)
    base, extra = _partition("WA", cands, wa_cells())
    out = stage_outcome("WA", base, extra, anchor)
    win = next(c for c in base + extra if c["label"] == out["decision"]["winner"])
    fin = [c for c in cands if not c["excluded"]]
    obj = _base_obj("WA", anchor)
    obj.update(_decision_record(out),
               winner_recipe=win["recipe"], winner_source=win["source"], winner_run_dir=win["run_dir"],
               winner_result_rows_sha256=win["result_rows_sha256"],
               candidates=[_dump(c, c["set"]) for c in sorted(base + extra, key=lambda c: c["label"])],
               values={c["label"]: c["value"] for c in sorted(fin, key=lambda c: c["label"])},
               excluded=_excluded(cands), required=sorted(wa_cells()), wa_best=max(c["value"] for c in fin),
               incumbent={"label": p2_label(p2_recipe(server_opt=INCUMBENT_CELL[0], server_lr=INCUMBENT_CELL[1],
                                                      client_lr=INCUMBENT_CELL[2])),
                          "source": next(c["source"] for c in base if c["set"] == "INCUMBENT"),
                          "p2t3_receipt": inc.get("p2t3_receipt") if inc else None,
                          "init_from": inc["_spec"].get("init_from") if inc else None},
               common=common, nonfinite_diagnostics=_nonfinite_diag(cands),
               flags=_flags([out["edge"]["status"]], out["decision"]["no_measurable_calib_gain"]), **ev)
    return _finish(obj)


def _redecide(stage: str, cands: Sequence[Mapping], anchor, rec: Mapping, path) -> None:
    edge_sets = {f"{s}_EDGE" for s in STAGES}
    b = [c for c in cands if c["set"] not in edge_sets]
    x = [c for c in cands if c["set"] in edge_sets]
    out = stage_outcome(stage, b, x, anchor)
    d = out["decision"]
    if (d["winner"] != rec["winner"] or d["tie_set"] != rec["tie_set"] or out["edge"]["status"] != rec["edge"]["status"]
            or d["calib_gain"] != rec["calib_gain"] or d["no_measurable_calib_gain"] != rec["no_measurable_calib_gain"]):
        raise P2SelectionRefused(f"{path}: the recorded decision does not re-derive from its candidates")


def load_p2_selection(path: Path, stage: str) -> dict:
    """A written P2 selection file, verified: format, self_sha256 (launch_fl._canon), labels = their recipes, the
    registered required set = the base candidates, the recorded decision / calib_gain / edge status re-derived
    (stage_outcome), the flags re-derived, and (FINAL) the futility decision and the drop prefix re-derived."""
    import v4int.launch_fl as F
    obj = _read_json(Path(path), f"{stage} selection")
    if not isinstance(obj, dict) or obj.get("format") != FORMATS.get(stage) or obj.get("stage") != stage:
        raise P2SelectionRefused(f"{path} is not a {FORMATS.get(stage)} (stage {stage}) file")
    try:
        ok = obj.get("self_sha256") == F._canon(obj)
    except (TypeError, ValueError):
        ok = False
    if not ok:
        raise P2SelectionRefused(f"{path}: self_sha256 does not verify")
    try:
        cands = obj["candidates"]
        for c in cands:
            r = c["recipe"]
            r2 = c1_recipe(r["variant"], r["peak_lr"]) if "variant" in r else p2_recipe(**r)
            if r2 != r or recipe_label(r2) != c["label"]:
                raise P2SelectionRefused(f"{path}: candidate {c.get('label')} label / recipe mismatch")
        base_labels = sorted(c["label"] for c in cands if not c["set"].endswith("_EDGE"))
        if stage == "C1":
            if base_labels != sorted(c1_cells()) or obj["required"] != sorted(c1_cells()):
                raise P2SelectionRefused(f"{path}: the C1 base set is not the registered 8 cells")
            for v in C1_LABELS:
                _redecide("C1", [c for c in cands if c["recipe"]["variant"] == v], obj["anchor"], obj["variants"][v],
                          path)
            flags = _flags([obj["variants"][v]["edge"]["status"] for v in C1_LABELS], False)
        elif stage == "WA":
            if base_labels != sorted(wa_cells()) or obj["required"] != sorted(wa_cells()):
                raise P2SelectionRefused(f"{path}: the WA base set is not the registered 21 cells")
            _redecide("WA", cands, obj["anchor"], obj, path)
            flags = _flags([obj["edge"]["status"]], obj["no_measurable_calib_gain"])
        else:
            fut = obj["futility"]
            f2 = futility(fut["c1_best"], fut["wa_best"], fut["anchor_value"])
            if f2["decision"] != fut["decision"] or fut["anchor_value"] != obj["anchor"]["value"]:
                raise P2SelectionRefused(f"{path}: the futility decision does not re-derive")
            drops = check_dropped(obj["dropped"]["units"])
            a_star = p2_recipe(**obj["a_star"]["recipe"])
            req = sorted(final_cells(a_star, drops, fut["skip_wb_wc"]))
            if base_labels != req or obj["required"] != req:
                raise P2SelectionRefused(f"{path}: the FINAL base set is not the registered set at A* / drops / futility")
            _redecide("FINAL", cands, obj["anchor"], obj, path)
            flags = _flags([obj["edge"]["status"], obj["edge_wa"]["status"]], obj["no_measurable_calib_gain"],
                           ["FUTILITY_SKIP_WB_WC"] if fut["skip_wb_wc"] else [])
    except (KeyError, TypeError, IndexError) as e:
        raise P2SelectionRefused(f"{path}: malformed selection file ({type(e).__name__}: {e})") from e
    if obj.get("flags") != flags:
        raise P2SelectionRefused(f"{path}: the recorded flags {obj.get('flags')} do not re-derive ({flags})")
    return obj


def _check_a_star_run(c: Mapping, wa: Mapping) -> None:
    if (c["label"] != wa["winner"] or c["result_rows_sha256"] != wa["winner_result_rows_sha256"]
            or c["recipe"] != wa["winner_recipe"]):
        raise P2SelectionRefused(f"{c['run_dir']}: not the verified A* run of the WA file ({wa['winner']})")
    rec = next(x for x in wa["candidates"] if x["label"] == wa["winner"])
    if rec.get("value") != c.get("value"):
        raise P2SelectionRefused(f"{c['run_dir']}: A*'s value differs from the WA file record")


def futility_from_files(wa_path: Path, c1_path: Path) -> dict:
    wa, c1 = load_p2_selection(wa_path, "WA"), load_p2_selection(c1_path, "C1")
    if wa["anchor"]["result_rows_sha256"] != c1["anchor"]["result_rows_sha256"]:
        raise P2SelectionRefused("the WA and C1 files were decided against different PRE_CALIB anchors")
    if wa["eval_manifest_id"] != c1["eval_manifest_id"]:
        raise P2SelectionRefused("the WA and C1 files were not evaluated on one manifest")
    rec = futility(c1["c1_best"], wa["wa_best"], wa["anchor"])
    rec.update(wa_selection={"path": str(wa_path), "sha256": LR._sha(Path(wa_path)), "self_sha256": wa["self_sha256"]},
               c1_selection={"path": str(c1_path), "sha256": LR._sha(Path(c1_path)), "self_sha256": c1["self_sha256"]})
    return rec


def selection_final(run_dirs: Sequence[Path], anchor_dir: Path, wa_path: Path, c1_path: Path,
                    dropped: Sequence[str] = (), incumbent_phase1: Optional[Path] = None,
                    p2t3_receipt: Optional[Path] = None, cap_plan: Optional[Path] = None) -> dict:
    anchor = read_anchor(anchor_dir)
    fut = futility_from_files(wa_path, c1_path)
    wa = load_p2_selection(wa_path, "WA")
    if wa["anchor"]["result_rows_sha256"] != anchor["result_rows_sha256"]:
        raise P2SelectionRefused("FINAL's anchor differs from the WA file's anchor")
    drops = check_dropped(dropped)
    plan = read_cap_plan(cap_plan) if cap_plan is not None else None
    if drops and plan is None:
        raise P2SelectionRefused("--dropped needs --cap-plan (§5.9: drop only if the projected total exceeds 250 GPU-h)")
    if plan is not None and plan["dropped"] != drops:
        raise P2SelectionRefused(f"--dropped {drops} != the §5.9 plan of the given projection {plan['dropped']}")
    cands = [read_p2_run(d) for d in run_dirs]
    if wa["winner_source"] == "PHASE1_INCUMBENT":
        if incumbent_phase1 is None:
            raise P2SelectionRefused("A* is the reused Phase-1 incumbent: give it with --incumbent-phase1 "
                                     "(+ --p2t3-receipt)")
        cands.append(read_phase1_incumbent(incumbent_phase1, p2t3_receipt))
    elif incumbent_phase1 is not None:
        raise P2SelectionRefused("--incumbent-phase1 is admitted at FINAL only when A* is the reused incumbent")
    a_star = p2_recipe(**wa["winner_recipe"])
    a_c = [c for c in cands if c["label"] == wa["winner"]]
    if len(a_c) != 1:
        raise P2SelectionRefused(f"the A* run ({wa['winner']}) must be given exactly once")
    _check_a_star_run(a_c[0], wa)
    ev = _same_eval(cands)
    if ev["eval_manifest_id"] != wa["eval_manifest_id"] or ev["split"] != wa["split"] or \
            json.dumps(ev["device"], sort_keys=True) != json.dumps(wa["device"], sort_keys=True):
        raise P2SelectionRefused("the FINAL runs do not share the WA file's evaluation manifest / device")
    common = _same_common(cands)
    if common is not None and wa.get("common") is not None and common != wa["common"]:
        raise P2SelectionRefused("the FINAL runs differ from the WA runs beyond the recipe")
    if common is None:
        common = wa.get("common")                          # e.g. futility with the Phase-1 incumbent as A*
    required = final_cells(a_star, drops, fut["skip_wb_wc"])
    if fut["skip_wb_wc"] and len(cands) != 1:
        raise P2SelectionRefused("FUTILITY_SKIP_WB_WC (§5.8): WB / WC are skipped; only the A* run is admitted")
    base, extra = _partition("FINAL", cands, required)
    out = stage_outcome("FINAL", base, extra, anchor)
    win = next(c for c in base + extra if c["label"] == out["decision"]["winner"])
    fin = [c for c in cands if not c["excluded"]]
    obj = _base_obj("FINAL", anchor)
    obj.update(_decision_record(out),
               winner_recipe=win["recipe"], winner_source=win["source"], winner_set=win["set"],
               winner_run_dir=win["run_dir"], winner_result_rows_sha256=win["result_rows_sha256"],
               a_star={"label": wa["winner"], "recipe": wa["winner_recipe"], "source": wa["winner_source"]},
               candidates=[_dump(c, c["set"]) for c in sorted(base + extra, key=lambda c: c["label"])],
               values={c["label"]: c["value"] for c in sorted(fin, key=lambda c: c["label"])},
               excluded=_excluded(cands), required=sorted(required), futility=fut,
               dropped={"units": drops, "arms": [DROP_ARMS.get(u, u) for u in drops],
                        "cells": sorted({x for u in drops if u in DROP_UNIT_CELLS for x in DROP_UNIT_CELLS[u]}),
                        "order": list(DROP_ORDER), "cap_gpu_h": CAP_GPU_H, "cap_plan": plan,
                        "note": "every dropped unit is declared not run in P2_METHODS_LOCK (§5.9); Part-B arms "
                                "unfinished at lock time are declared not run under §2.1, not here"},
               wa_selection=fut["wa_selection"], c1_selection=fut["c1_selection"], common=common,
               nonfinite_diagnostics=_nonfinite_diag(cands),
               flags=_flags([out["edge"]["status"], wa["edge"]["status"]], out["decision"]["no_measurable_calib_gain"],
                            ["FUTILITY_SKIP_WB_WC"] if fut["skip_wb_wc"] else []),
               edge_wa={"status": wa["edge"]["status"], "notes": wa["edge"].get("notes", [])}, **ev)
    return _finish(obj)


def write_selection(stage: str, out: Path, **kw) -> dict:
    """Build the stage file and write it O_EXCL (refused if it exists; never overwritten)."""
    from v4models.variants import write_exclusive
    out = Path(out)
    if out.exists():
        raise P2SelectionRefused(f"{out} exists (O_EXCL: a P2 selection is written once)")
    if stage == "C1":
        obj = selection_c1(kw["runs"], kw["anchor"])
    elif stage == "WA":
        obj = selection_wa(kw["runs"], kw["anchor"], kw.get("incumbent_phase1"), kw.get("p2t3_receipt"))
    elif stage == "FINAL":
        obj = selection_final(kw["runs"], kw["anchor"], kw["wa_selection"], kw["c1_selection"], kw.get("dropped", ()),
                              kw.get("incumbent_phase1"), kw.get("p2t3_receipt"), kw.get("cap_plan"))
    else:
        raise P2SelectionRefused(f"--stage must be one of {STAGES}")
    sha = write_exclusive(out, json.dumps(obj, indent=1, sort_keys=True, allow_nan=False) + "\n")
    rec = {"path": str(out), "sha256": sha, "self_sha256": obj["self_sha256"], "stage": stage, "flags": obj["flags"]}
    if stage == "C1":
        rec["winners"] = {v: obj["variants"][v]["winner"] for v in C1_LABELS}
    else:
        rec["winner"] = obj["winner"]
    return rec


def main(argv=None, say: Callable = print) -> int:
    ap = argparse.ArgumentParser(description="PPSI V4 Phase-2 CALIB selection (P2-06; PREG2 REGISTERED §5.6-§5.9)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("select")
    s.add_argument("--stage", required=True, choices=STAGES)
    s.add_argument("--runs", nargs="*", type=Path, default=[],
                   help="the stage's run dirs (FINAL under futility with the Phase-1 incumbent as A*: none)")
    s.add_argument("--anchor", type=Path, required=True, help="the P2_ANCHOR_PRE_CALIB reeval output dir (P2-07)")
    s.add_argument("--wa-selection", type=Path, default=None)
    s.add_argument("--c1-selection", type=Path, default=None)
    s.add_argument("--dropped", nargs="*", default=[], help="§5.9 units dropped under the cap (registered order)")
    s.add_argument("--cap-plan", type=Path, default=None, help=f"a {CAP_PLAN_FORMAT} file (projection + savings)")
    s.add_argument("--incumbent-phase1", type=Path, default=None, help="S8Tc run dir (only with --p2t3-receipt)")
    s.add_argument("--p2t3-receipt", type=Path, default=None)
    s.add_argument("--out", type=Path, default=None)
    f = sub.add_parser("futility")
    f.add_argument("--wa-selection", type=Path, required=True)
    f.add_argument("--c1-selection", type=Path, required=True)
    args = ap.parse_args(argv)
    import ppsi_env  # noqa: F401  (isolation before v4 imports)
    import v4int.launch_fl as F
    from v4int.central_ft import CentralFTRefused
    try:
        if args.cmd == "futility":
            r = futility_from_files(args.wa_selection, args.c1_selection)
            say(json.dumps({"decision": r["decision"], "wa_selection_sha256": r["wa_selection"]["sha256"],
                            "c1_selection_sha256": r["c1_selection"]["sha256"]}, sort_keys=True))
            return 0
        if args.stage == "FINAL" and (args.wa_selection is None or args.c1_selection is None):
            raise P2SelectionRefused("--stage FINAL needs --wa-selection and --c1-selection")
        if args.stage != "FINAL" and (args.wa_selection or args.c1_selection or args.dropped or args.cap_plan):
            raise P2SelectionRefused("--wa-selection / --c1-selection / --dropped / --cap-plan are FINAL options")
        if args.stage == "C1" and args.incumbent_phase1 is not None:
            raise P2SelectionRefused("--incumbent-phase1 is a WA / FINAL option")
        rec = write_selection(args.stage, args.out or DEFAULT_OUT[args.stage], runs=args.runs, anchor=args.anchor,
                              wa_selection=args.wa_selection, c1_selection=args.c1_selection, dropped=args.dropped,
                              cap_plan=args.cap_plan, incumbent_phase1=args.incumbent_phase1,
                              p2t3_receipt=args.p2t3_receipt)
    except P2EdgeRunsMissing as e:
        print(f"P2 SELECTION REFUSED (edge runs missing): {e}", file=sys.stderr, flush=True)
        return 3
    except (LR.SelectionRefused, F.LaunchRefused, CentralFTRefused) as e:
        print(f"P2 SELECTION REFUSED: {e}", file=sys.stderr, flush=True)
        return 2
    say(json.dumps(rec, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
