"""v4int.audit_e1_mia (P2-14a, owner: Phase-2 privacy audits) - the E1 user-level membership-inference audit of the
final models, exactly as registered in contracts/PREG2.md §7.1. A read-only analysis: no training step, no arm is
selected, no VALIDATION / TEST row or label is read.

  python -B code/v4int/audit_e1_mia.py --targets PRE=<ckpt> FA_WARM_FROZEN=<ckpt> P2_WSTAR=<ckpt>
      [P2_WSTAR_DP8=<ckpt>] [P2_WSTAR_DP1=<ckpt>] P2_C1_FT_CENTRAL_FULL=<ckpt>
      --out <new dir> [--device cuda|cpu] [--dataset-root <frozen v1 root>] [--dry-run]

Binding spec: contracts/PREG2.md (REGISTERED, DL 411, sha256 be0561ab...) §7 preamble + §7.1 (the text is read from the
contract, hash-pinned as SPEC_SHA256 and copied into the output), §6.4 (label DESCRIPTIVE, tertiary), §6.3 S8 (no value
on stdout).

Targets (each a checkpoint FILE; every source check of v4int.reeval / v4int.reeval_p2 runs before a weight is used):
  PRE                    theta_pre: the PRETRAIN_FOR_FL central endpoint_6.0.pt, the registered sha256 cf08588c... and
                         state digest ee33345b..., its PRETRAIN_FOR_FL_INIT.json fl_init record and the PRE registry
                         extension binding. MANDATORY (the null control).
  FA_WARM_FROZEN         the Phase-1 FL run (reeval.source_fl + check_warm_line), read only. Mandatory.
  P2_WSTAR               the MAIN P2_W run P2_WSTAR_SMALL_s2026 (reeval_p2.p2_source_fl, payload mark, merged LoRA).
                         A WC5 personal-head run is audited on its SHARED weights (the server-side model; no PF store).
                         Mandatory.
  P2_WSTAR_DP8 / _DP1    the W* DP-FedAvg chain runs (reeval.source_fl + launch_fl.check_p2_dp_spec; run id
                         <arm>_SMALL_s2026; LAUNCH dp_z epsilon = the registered accountant epsilon 8 / 1). endpoint_6.0.pt
                         ONLY: the accountant covers the 6.0 endpoint, a best-of-12 checkpoint is outside it (PREG2 §9.1).
                         May be absent (recorded NOT_AVAILABLE).
  P2_C1_FT_CENTRAL_FULL  the MAIN C1 central run (reeval.source_central + reeval_p2.check_c1_source /
                         check_c1_checkpoint). Mandatory.
  Every target: SASREC_D64_B2 / SMALL, seed 2026; best.pt (PRACTICAL_BEST) or endpoint_6.0.pt (CONTROLLED_BUDGET) where
  admitted; the rule of the file is recorded.
Populations (TRAIN window ONLY, [2019-10-01, 2019-11-16) half-open, v4data.common TRAIN_START_US / TRAIN_END_US):
  MEMBERS     10,000 users of MAIN_TRAIN_5_100_131072 (REGISTRY_V2 role MAIN), np.random.default_rng(20261004).
  NONMEMBERS  10,000 users of UNEW_HOLDOUT_SUPPORT_32768 (role UNSEEN_HOLDOUT = the U_NEW_TEST band; its TRAIN-window
              rows), np.random.default_rng([20261004, 2]).
  Each manifest is verified (v4int.verify.verify_manifest_by_registry through the LOCK_V2 REGISTRY_V2, kind TRAIN), then
  the WINDOW CHECK refuses unless every row is old_split TRAIN, decision_ts_us AND target_ts_us inside the TRAIN window,
  target_status OBSERVED_IN_PARTITION, the manifest id / header name no VALIDATION / TEST split. Only the selected
  users' decisions are materialised (v4int.cache.build_subset_store, decision split TRAIN, USER_HISTORY_L50 whose store
  partitions for TRAIN decisions are TRAIN only - asserted), every batch is checked against the manifest (decision_id,
  target_class, loss_mask), and members / non-members must share no store client and no decision. No evaluation
  manifest is ever opened. The checks are recorded in E1_AUDIT.json (window_check).
Score: per user, mean CE over the user's TRAIN-window decisions (full-K CE, the trainer's objective, eval mode).
  s(u) = mean CE_theta(u) - mean CE_theta_pre(u) (reference-calibrated; the registered score) and the raw mean CE beside
  it. The attack statistic is -s (members have the lower loss): a user is called a member iff -s > tau.
Protocol / metrics (np.random.default_rng([20261004, 3]) splits each population 50/50 -> half A / half B):
  ROC AUC on all 20,000 users (Mann-Whitney, ties 1/2) with a user-bootstrap 95 % CI (2,000 replicates, members and
  non-members resampled separately, np.random.default_rng(20261004), the same replicate indices for every target;
  percentile CI, numpy 'linear' quantile). For FPR in {0.1 %, 1 %}: tau from half A's non-members (the (k+1)-th largest,
  k = floor(FPR * n), so at most k of them exceed it), applied to half B: TPR_B / FPR_B with Clopper-Pearson 95 % bounds.
  eps_LB = ln(TPR_lo / FPR_hi) at the registered threshold (FPR 1 %), clipped at 0 (0 when TPR_lo = 0); the 0.1 % value is
  recorded beside it.
Criteria (PREG2 §7.1): (a) VALIDITY FIRST: PRE is scored before every other target; its AUC CI on the RAW loss must
  include 0.5 (PRE's calibrated score is identically 0, so that test would be vacuous). Otherwise audit_status
  AUDIT_INVALID: no eps_LB is reported for any target, PRE is AUDIT_INVALID and no other target is scored
  (NOT_SCORED_AUDIT_INVALID). (b) DP arms: PASS iff eps_LB <= the accountant epsilon, else FAIL_INVESTIGATE. (c) non-DP
  arms: MATERIAL_LEAKAGE iff the AUC CI lower bound > 0.55 or the TPR@1 % Clopper-Pearson lower bound > 3 %, else
  NO_MATERIAL_LEAKAGE. (b) / (c) read the calibrated score. Label DESCRIPTIVE (tertiary).
Outputs in the NEW --out directory: E1_AUDIT.json (spec text + sha, protocol, user-selection sha256, window check, input
  checkpoint sha256s and source checks, per-target metrics and statuses, the per-user file sha256, self_sha256; a pure
  function of the inputs: the same inputs give the same bytes), E1_USER_SCORES.json (per-user raw mean CE and calibrated
  score per target; self_sha256), E1_RUN_RECORD.json (argv, utc, device, timings, cache build records; not
  deterministic). S8: stdout carries step names, statuses and file sha256 only - never a metric value; nobody opens the
  two value files before both sitting analyses are written (PREG2 §6.3 S3 / S8).
--dry-run: the target plan and every torch-free source check, nothing read from the data root, nothing written.
Exit: 0 written (or dry run passed), 1 refused / error.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace
from typing import Callable, Iterator, Mapping, Optional, Sequence

HERE = Path(__file__).resolve().parent
CODE = HERE.parent
V4 = CODE.parent
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))
sys.dont_write_bytecode = True

from v4int import reeval as R  # noqa: E402  (stdlib-only at import)
from v4int import reeval_p2 as P2  # noqa: E402  (stdlib-only at import)

TASK = "P2-14a v4int.audit_e1_mia"
FORMAT = "PPSI_V4_P2_E1_AUDIT_1"
SCORES_FORMAT = "PPSI_V4_P2_E1_USER_SCORES_1"
RUN_RECORD_FORMAT = "PPSI_V4_P2_E1_RUN_RECORD_1"
AUDIT_FILE, SCORES_FILE, RUN_RECORD_FILE = "E1_AUDIT.json", "E1_USER_SCORES.json", "E1_RUN_RECORD.json"
PREG2_PATH = V4 / "contracts" / "PREG2.md"
PREG2_SHA256 = P2.PREG2_SHA256
SPEC_START, SPEC_END = "## 7. ", "### 7.2 "           # the §7 preamble + §7.1, cut from the contract (spec_text)
SPEC_SHA256 = "61624bf9942fd8437f718cbabe5c10093ccf8b90c9e3939f174d10d8ec46ccc5"
AUTHORITY = (f"contracts/PREG2.md (REGISTERED, DL 411, sha256 {PREG2_SHA256}) §7 + §7.1 (E1), §6.4 (DESCRIPTIVE), "
             "§6.3 S8; company/plans/PHASE2_IMPLEMENTATION_PLAN.md RUN-4 (E1 MIA); task packet P2-14a")
LABEL = "DESCRIPTIVE (tertiary)"                       # PREG2 §6.4: audits E1-E5

SEED = 20261004                                        # PREG2 §7.1 member selection
BOOTSTRAP_SEED = 20261004                              # PREG2 §6.3 S1 bootstrap seed
NONMEMBER_STREAM, SPLIT_STREAM = [SEED, 2], [SEED, 3]   # np.random.default_rng entropy of the other two draws
N_MEMBERS = N_NONMEMBERS = 10_000
N_BOOT = 2_000
CI_QUANTILES = (0.025, 0.975)
CP_ALPHA = 0.05                                        # Clopper-Pearson 95 % (two-sided)
THRESHOLD_FPR = Fraction(1, 100)                       # "the threshold is set on half A at FPR 1 %"
FPR_LEVELS = (Fraction(1, 1000), Fraction(1, 100))
AUC_MATERIAL = 0.55
TPR1_MATERIAL = 0.03
SCORE_BATCH = 256
HISTORY_POLICY = "USER_HISTORY_L50"
FAMILY, VARIANT, RECIPE, TARGET_SEED = R.FAMILY, P2.VARIANT, P2.RECIPE, 2026
BEST, END = R.BEST_FILE, R.ENDPOINT_FILE
FILE_RULE = {BEST: P2.PB, END: P2.CB}
WINDOW = {"name": "TRAIN", "rule": "[2019-10-01T00:00:00Z, 2019-11-16T00:00:00Z) half-open (v4data.common)"}
WINDOW_COLUMNS = ("decision_id", "client_ord", "decision_ts_us", "target_ts_us", "target_class", "target_status",
                  "old_split")
FORBIDDEN_NAMES = ("VALIDATION", "TEST", "SUPPORT_QUERY")   # no evaluation / panel-query manifest is a population

POPULATIONS = {                                        # REGISTRY_V2 roles MAIN / UNSEEN_HOLDOUT, TRAIN-layout manifests
    "MEMBERS": {"manifest_id": "MAIN_TRAIN_5_100_131072", "status": "MAIN", "member": True, "n": N_MEMBERS},
    "NONMEMBERS": {"manifest_id": "UNEW_HOLDOUT_SUPPORT_32768", "status": "UNSEEN_HOLDOUT", "member": False,
                   "n": N_NONMEMBERS},
}
TARGET_ORDER = ("PRE", "FA_WARM_FROZEN", "P2_WSTAR", "P2_WSTAR_DP8", "P2_WSTAR_DP1", "P2_C1_FT_CENTRAL_FULL")
TARGETS = {
    "PRE": {"kind": "central", "class": "NULL_CONTROL", "epsilon": None, "files": (END,), "mandatory": True},
    "FA_WARM_FROZEN": {"kind": "fl", "class": "NON_DP", "epsilon": None, "files": (BEST, END), "mandatory": True},
    "P2_WSTAR": {"kind": "fl", "class": "NON_DP", "epsilon": None, "files": (BEST, END), "mandatory": True},
    "P2_WSTAR_DP8": {"kind": "fl", "class": "DP", "epsilon": 8.0, "files": (END,), "mandatory": False},
    "P2_WSTAR_DP1": {"kind": "fl", "class": "DP", "epsilon": 1.0, "files": (END,), "mandatory": False},
    "P2_C1_FT_CENTRAL_FULL": {"kind": "central", "class": "NON_DP", "epsilon": None, "files": (BEST, END),
                              "mandatory": True},
}
VALID, INVALID = "VALID", "AUDIT_INVALID"
S_VALID_NULL, S_PASS, S_FAIL = "VALID_NULL", "PASS", "FAIL_INVESTIGATE"
S_LEAK, S_NO_LEAK = "MATERIAL_LEAKAGE", "NO_MATERIAL_LEAKAGE"
S_NA, S_NOT_SCORED = "NOT_AVAILABLE", "NOT_SCORED_AUDIT_INVALID"


class E1Refused(RuntimeError):
    pass


def _refuse(msg: str):
    raise E1Refused(msg)


def _canon(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sha_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def with_self_sha(obj: dict) -> dict:
    body = {k: v for k, v in obj.items() if k != "self_sha256"}
    return dict(body, self_sha256=_sha_bytes(_canon(body).encode("utf-8")))


def json_safe(obj):
    """A JSON round trip (paths and other objects as str) so a source record can never fail the final write."""
    return json.loads(json.dumps(obj, default=str, allow_nan=False))


def write_json(path: Path, obj: Mapping) -> str:
    """Strict JSON (no NaN / inf), sorted keys, LF; returns the file sha256."""
    data = (json.dumps(obj, indent=1, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    Path(path).write_bytes(data)
    return _sha_bytes(data)


# ------------------------------------------------------------------------------------------------ the registered text
def spec_text(path: Path = PREG2_PATH) -> dict:
    """The §7 preamble + §7.1 as registered: PREG2.md must hash to the registered sha256 and the cut block to
    SPEC_SHA256 (so the audit never runs against an edited contract)."""
    p = Path(path)
    if not p.is_file():
        _refuse(f"the registered contract {p} is not on this host")
    data = p.read_bytes()
    if _sha_bytes(data) != PREG2_SHA256:
        _refuse(f"{p.name} sha256 is not the registered {PREG2_SHA256}")
    lines = data.decode("utf-8").split("\n")
    try:
        i = next(k for k, ln in enumerate(lines) if ln.startswith(SPEC_START))
        j = next(k for k, ln in enumerate(lines) if ln.startswith(SPEC_END))
    except StopIteration:
        _refuse(f"{p.name}: the §7 / §7.2 headings were not found")
    text = "\n".join(lines[i:j]).rstrip() + "\n"
    if _sha_bytes(text.encode("utf-8")) != SPEC_SHA256:
        _refuse(f"the §7.1 text of {p.name} does not hash to the registered SPEC_SHA256")
    return {"path": "contracts/PREG2.md", "file_sha256": PREG2_SHA256, "section": "§7 preamble + §7.1",
            "text": text, "text_sha256": SPEC_SHA256}


# ------------------------------------------------------------------------------------------------ metrics (numpy only)
def avg_ranks(x):
    """1-based ranks with ties at their average rank (deterministic: stable sort)."""
    import numpy as np
    x = np.asarray(x, dtype=np.float64)
    n = x.size
    order = np.argsort(x, kind="mergesort")
    xs = x[order]
    starts = np.flatnonzero(np.concatenate(([True], xs[1:] != xs[:-1]))) if n else np.zeros(0, dtype=np.int64)
    ends = np.concatenate((starts[1:], [n])) if n else starts
    r = np.empty(n, dtype=np.float64)
    r[order] = np.repeat((starts + ends - 1) / 2.0 + 1.0, ends - starts)
    return r


def auc(pos, neg) -> float:
    """ROC AUC of `pos` (members) above `neg` (non-members) = P(pos > neg) + P(tie) / 2 (Mann-Whitney U)."""
    import numpy as np
    pos, neg = np.asarray(pos, dtype=np.float64), np.asarray(neg, dtype=np.float64)
    if pos.size == 0 or neg.size == 0:
        _refuse("AUC needs at least one member and one non-member")
    r = avg_ranks(np.concatenate((pos, neg)))
    u = float(r[:pos.size].sum()) - pos.size * (pos.size + 1) / 2.0
    return u / (pos.size * neg.size)


def bootstrap_auc(pos, neg, *, n_boot: int, seed: int) -> dict:
    """AUC with a user-bootstrap percentile CI: members and non-members resampled separately with replacement from
    np.random.default_rng(seed) in a fixed order (member draw, then non-member draw, per replicate), so every target
    of one audit sees the same replicate indices."""
    import numpy as np
    pos, neg = np.asarray(pos, dtype=np.float64), np.asarray(neg, dtype=np.float64)
    rng = np.random.default_rng(seed)
    reps = np.empty(int(n_boot), dtype=np.float64)
    for b in range(int(n_boot)):
        ip = rng.integers(0, pos.size, size=pos.size)
        ineg = rng.integers(0, neg.size, size=neg.size)
        reps[b] = auc(pos[ip], neg[ineg])
    lo, hi = (float(v) for v in np.quantile(reps, CI_QUANTILES, method="linear"))
    return {"auc": auc(pos, neg), "ci95": [lo, hi], "n_boot": int(n_boot), "seed": int(seed),
            "ci_method": "percentile, numpy quantile 'linear'", "n_members": int(pos.size),
            "n_nonmembers": int(neg.size)}


def _betacf(a: float, b: float, x: float) -> float:
    """Continued fraction of the regularized incomplete beta (modified Lentz)."""
    tiny, eps = 1e-300, 1e-15
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    d = 1.0 / (d if abs(d) > tiny else tiny)
    h = d
    for m in range(1, 100_000):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c
        c = c if abs(c) > tiny else tiny
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c
        c = c if abs(c) > tiny else tiny
        de = d * c
        h *= de
        if abs(de - 1.0) < eps:
            return h
    _refuse(f"incomplete beta did not converge (a {a}, b {b})")


def betainc(a: float, b: float, x: float) -> float:
    """I_x(a, b), the regularized incomplete beta function (a, b > 0)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbt = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log1p(-x)
    bt = math.exp(lbt)
    if x < (a + 1.0) / (a + b + 2.0):
        return bt * _betacf(a, b, x) / a
    return 1.0 - bt * _betacf(b, a, 1.0 - x) / b


def beta_ppf(q: float, a: float, b: float) -> float:
    """The q-quantile of Beta(a, b) by bisection on I_x(a, b) (monotone; 200 halvings = float resolution)."""
    lo, hi = 0.0, 1.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if betainc(a, b, mid) < q:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def cp_bounds(k: int, n: int, alpha: float = CP_ALPHA) -> list:
    """Clopper-Pearson (exact, two-sided 1 - alpha) bounds of a binomial proportion k / n."""
    k, n = int(k), int(n)
    if n < 1 or not 0 <= k <= n:
        _refuse(f"Clopper-Pearson needs 0 <= k <= n, n >= 1 (k {k}, n {n})")
    lo = 0.0 if k == 0 else beta_ppf(alpha / 2.0, k, n - k + 1)
    hi = 1.0 if k == n else beta_ppf(1.0 - alpha / 2.0, k + 1, n - k)
    return [lo, hi]


def allowed_false_positives(n_neg: int, fpr: Fraction) -> int:
    """k = floor(FPR * n) in exact arithmetic."""
    return int(math.floor(Fraction(fpr) * int(n_neg)))


def threshold_at_fpr(neg, fpr: Fraction) -> float:
    """tau = the (k+1)-th largest non-member statistic, k = floor(FPR * n): at most k non-members exceed it (a user
    is called a member iff its statistic is > tau; ties at tau are called non-members)."""
    import numpy as np
    neg = np.sort(np.asarray(neg, dtype=np.float64))[::-1]
    if neg.size == 0:
        _refuse("a threshold needs at least one non-member on half A")
    k = allowed_false_positives(neg.size, fpr)
    if k >= neg.size:
        _refuse(f"FPR {fpr} admits every half-A non-member: no threshold")
    return float(neg[k])


def eps_lower_bound(tpr_lo: float, fpr_hi: float) -> dict:
    """eps_LB = ln(TPR_lo / FPR_hi), reported clipped at 0; raw None when TPR_lo = 0 (the bound is vacuous)."""
    raw = math.log(tpr_lo / fpr_hi) if tpr_lo > 0.0 and fpr_hi > 0.0 else None
    return {"eps_lb_raw": raw, "eps_lb": max(0.0, raw) if raw is not None else 0.0}


def transfer(pos_a, neg_a, pos_b, neg_b, fpr: Fraction, alpha: float = CP_ALPHA) -> dict:
    """The threshold from half A's non-members at `fpr`, evaluated on half B (TPR_B / FPR_B with CP bounds)."""
    import numpy as np
    tau = threshold_at_fpr(neg_a, fpr)
    pos_b, neg_b = np.asarray(pos_b, dtype=np.float64), np.asarray(neg_b, dtype=np.float64)
    tp, fp = int(np.count_nonzero(pos_b > tau)), int(np.count_nonzero(neg_b > tau))
    tpr_ci, fpr_ci = cp_bounds(tp, pos_b.size, alpha), cp_bounds(fp, neg_b.size, alpha)
    out = {"fpr_target": str(fpr), "threshold": tau, "rule": "member iff statistic > threshold (threshold = the "
           "(k+1)-th largest half-A non-member statistic, k = floor(FPR * n_A))",
           "n_nonmembers_a": int(np.asarray(neg_a).size), "allowed_fp_a": allowed_false_positives(
               np.asarray(neg_a).size, fpr), "fp_a": int(np.count_nonzero(np.asarray(neg_a, dtype=np.float64) > tau)),
           "tp_b": tp, "n_members_b": int(pos_b.size), "fp_b": fp, "n_nonmembers_b": int(neg_b.size),
           "tpr_b": tp / pos_b.size, "fpr_b": fp / neg_b.size, "tpr_cp95": tpr_ci, "fpr_cp95": fpr_ci}
    out.update(eps_lower_bound(tpr_ci[0], fpr_ci[1]))
    return out


def score_metrics(t_mem, t_non, half_a_mem, half_a_non, *, n_boot: int, seed: int) -> dict:
    """Every registered metric of one score: AUC + bootstrap CI on all users; per FPR level the A -> B transfer;
    eps_LB at the registered threshold (FPR 1 %)."""
    import numpy as np
    t_mem, t_non = np.asarray(t_mem, dtype=np.float64), np.asarray(t_non, dtype=np.float64)
    if not (np.all(np.isfinite(t_mem)) and np.all(np.isfinite(t_non))):
        _refuse("a user score is not finite")
    am, an = np.asarray(half_a_mem, dtype=bool), np.asarray(half_a_non, dtype=bool)
    lv = {str(f): transfer(t_mem[am], t_non[an], t_mem[~am], t_non[~an], f) for f in FPR_LEVELS}
    thr = lv[str(THRESHOLD_FPR)]
    return {"auc": bootstrap_auc(t_mem, t_non, n_boot=n_boot, seed=seed), "tpr_at_fpr": lv,
            "eps_lb": thr["eps_lb"], "eps_lb_raw": thr["eps_lb_raw"], "eps_lb_threshold_fpr": str(THRESHOLD_FPR),
            "eps_lb_at_fpr": {k: v["eps_lb"] for k, v in lv.items()}}


def strip_eps(m: Mapping) -> dict:
    """Criterion (a): no eps_LB is reported when the audit is invalid (the fields are removed, not zeroed)."""
    out = {k: v for k, v in m.items() if not k.startswith("eps_lb")}
    out["tpr_at_fpr"] = {f: {k: v for k, v in d.items() if not k.startswith("eps_lb")}
                         for f, d in m["tpr_at_fpr"].items()}
    out["eps_lb_reported"] = False
    return out


# ------------------------------------------------------------------------------------------------ criteria
def validity(pre_raw: Mapping) -> dict:
    """(a) PRE (trained on neither population) must not separate members from non-members: its RAW-loss AUC CI must
    include 0.5 (its calibrated score is identically 0)."""
    lo, hi = pre_raw["auc"]["ci95"]
    ok = lo <= 0.5 <= hi
    return {"rule": "PREG2 §7.1 (a): PRE's AUC 95 % CI (raw mean CE; the calibrated score of PRE is identically 0) "
                    "includes 0.5, checked before any other target is scored and before any eps_LB is reported",
            "score": "raw_mean_ce", "auc_ci95": [lo, hi], "includes_0_5": bool(ok), "status": VALID if ok else INVALID}


def classify(label: str, cal: Mapping, accountant_epsilon: Optional[float] = None) -> dict:
    """(b) DP arms: PASS iff eps_LB <= the accountant epsilon; (c) non-DP arms: MATERIAL_LEAKAGE iff AUC CI lower bound >
    0.55 or TPR@1 % CP lower bound > 3 %. Both on the calibrated score."""
    t = TARGETS[label]
    if t["class"] == "DP":
        eps = float(accountant_epsilon if accountant_epsilon is not None else t["epsilon"])
        if eps != t["epsilon"]:
            _refuse(f"{label}: accountant epsilon {eps} is not the registered {t['epsilon']}")
        ok = cal["eps_lb"] <= eps
        return {"status": S_PASS if ok else S_FAIL, "criterion": f"(b) eps_LB <= accountant epsilon {eps}",
                "accountant_epsilon": eps, "eps_lb": cal["eps_lb"]}
    if t["class"] == "NON_DP":
        a_lo = cal["auc"]["ci95"][0]
        t_lo = cal["tpr_at_fpr"][str(Fraction(1, 100))]["tpr_cp95"][0]
        leak = a_lo > AUC_MATERIAL or t_lo > TPR1_MATERIAL
        return {"status": S_LEAK if leak else S_NO_LEAK,
                "criterion": f"(c) AUC CI lower bound > {AUC_MATERIAL} or TPR@1 % lower bound > {TPR1_MATERIAL}",
                "auc_ci_lower_gt_0_55": bool(a_lo > AUC_MATERIAL), "tpr1_lower_gt_0_03": bool(t_lo > TPR1_MATERIAL)}
    _refuse(f"{label} is the null control: it is judged by validity(), not classify()")


# ------------------------------------------------------------------------------------------------ users
def select_users(keys, n: int, rng):
    """n distinct users drawn without replacement from the sorted unique keys; returned sorted."""
    import numpy as np
    u = np.unique(np.asarray(keys, dtype=np.int64))
    if u.size < int(n):
        _refuse(f"the population holds fewer users than the registered selection size ({u.size} < {n})")
    return np.sort(rng.choice(u, size=int(n), replace=False)).astype(np.int64)


def split_half(n: int, rng):
    """Boolean half-A mask with exactly floor(n / 2) users in A (a seeded permutation)."""
    import numpy as np
    a = np.zeros(int(n), dtype=bool)
    a[rng.permutation(int(n))[: int(n) // 2]] = True
    return a


def selection_record(members, nonmembers, half_a_mem, half_a_non) -> dict:
    import numpy as np

    def side(pop, half):
        body = {"manifest_id": pop.manifest_id, "client_ord": [int(x) for x in pop.user_keys],
                "half_a": [bool(x) for x in np.asarray(half)]}
        return {"manifest_id": pop.manifest_id, "n_users_population": int(pop.n_users_population),
                "n_selected": int(len(pop.user_keys)), "n_half_a": int(np.count_nonzero(half)),
                "n_decisions": int(pop.n_decisions), "selected_sha256": _sha_bytes(_canon(body).encode("utf-8"))}, body

    m, mb = side(members, half_a_mem)
    n, nb = side(nonmembers, half_a_non)
    return {"members": dict(m, seed=SEED, rng="np.random.default_rng(20261004).choice(sorted unique client_ord, "
                                              "replace=False), sorted"),
            "nonmembers": dict(n, seed=NONMEMBER_STREAM, rng="np.random.default_rng([20261004, 2]).choice(sorted "
                                                             "unique client_ord, replace=False), sorted"),
            "split": {"rng": "np.random.default_rng([20261004, 3]).permutation: members first, then non-members; "
                             "half A = the first floor(n / 2) of each", "seed": SPLIT_STREAM},
            "user_selection_sha256": _sha_bytes(_canon({"members": mb, "nonmembers": nb}).encode("utf-8"))}


# ------------------------------------------------------------------------------------------------ the TRAIN window
def refuse_window_manifest_id(manifest_id: str) -> None:
    """Before any file is opened: a VALIDATION / TEST / panel-query manifest is never a population."""
    mid = str(manifest_id).upper()
    bad = [w for w in FORBIDDEN_NAMES if w in mid]
    if bad:
        _refuse(f"manifest {manifest_id} names {bad}: E1 reads TRAIN-window rows only (PREG2 §7)")


def window_check(manifest_id: str, kind: str, header: Mapping, cols: Mapping) -> dict:
    """The TRAIN-window assertion over EVERY row of a verified manifest (refuses on any failure)."""
    import numpy as np
    from v4data.common import TRAIN_END_US, TRAIN_START_US
    refuse_window_manifest_id(manifest_id)
    missing = [c for c in WINDOW_COLUMNS if c not in cols]
    if missing:
        _refuse(f"{manifest_id}: the decisions table lacks {missing}: the TRAIN window cannot be asserted")
    ts = np.asarray(cols["decision_ts_us"], dtype=np.int64)
    tt = np.asarray(cols["target_ts_us"], dtype=np.int64)
    in_w = lambda a: bool(np.all((a >= TRAIN_START_US) & (a < TRAIN_END_US)))  # noqa: E731
    split = str(header.get("v4_split", "TRAIN") or "TRAIN").upper()
    checks = {
        "manifest_id_names_no_validation_test_query": True,                       # refuse_window_manifest_id above
        "manifest_kind_TRAIN": kind == "TRAIN",
        "header_split_TRAIN": split == "TRAIN" and not header.get("all_methods_lock_sha256"),
        "old_split_all_TRAIN": bool(np.all(np.asarray(cols["old_split"], dtype=object) == "TRAIN")),
        "decision_ts_in_TRAIN_window": in_w(ts),
        "target_ts_in_TRAIN_window": in_w(tt),
        "target_status_all_OBSERVED_IN_PARTITION": bool(np.all(np.asarray(cols["target_status"], dtype=object)
                                                               == "OBSERVED_IN_PARTITION")),
        "target_class_nonnegative": bool(np.all(np.asarray(cols["target_class"], dtype=np.int64) >= 0)),
    }
    if "v4_split" in cols:
        checks["v4_split_all_TRAIN"] = bool(np.all(np.asarray(cols["v4_split"], dtype=object) == "TRAIN"))
    rec = {"manifest_id": manifest_id, "window": dict(WINDOW, start_us=int(TRAIN_START_US), end_us=int(TRAIN_END_US)),
           "n_rows_checked": int(ts.size), "checks": checks, "all_ok": all(checks.values())}
    if not rec["all_ok"]:
        _refuse(f"{manifest_id}: TRAIN-window check failed {[k for k, v in checks.items() if not v]}: E1 reads no row "
                "and no label outside the TRAIN window (PREG2 §7)")
    return rec


def store_window_check(manifest_id: str, info, ds, ids, targets, store_partitions) -> dict:
    """The materialised store: decision split TRAIN, store partitions TRAIN only, exactly the selected decisions."""
    import numpy as np
    checks = {
        "decision_splits_TRAIN_only": tuple(getattr(info, "splits", ())) == ("TRAIN",),
        "store_partitions_TRAIN_only": tuple(store_partitions) == ("TRAIN",),
        "decision_set_split_TRAIN": str(getattr(ds, "split", "")) == "TRAIN",
        "decision_ids_eq_selected": bool(np.array_equal(np.asarray(ds.decision_id, dtype=np.int64), ids)),
        "target_class_eq_manifest": bool(np.array_equal(np.asarray(ds.target_class, dtype=np.int64), targets)),
        "all_loss_eligible": bool(np.all(np.asarray(ds.loss_mask, dtype=bool))),
    }
    if not all(checks.values()):
        _refuse(f"{manifest_id}: store window check failed {[k for k, v in checks.items() if not v]}")
    return {"checks": checks, "all_ok": True}


class Population:
    """One side of the audit: the selected users' TRAIN-window decisions in ascending decision_id. `batches(bs)` yields
    (positions, CPU batch); `dec_user[i]` is the index into `user_keys` of decision i."""

    def __init__(self, *, name: str, member: bool, manifest_id: str, user_keys, dec_user, batch_fn: Callable,
                 n_users_population: int, record: Mapping, client_idx=None, decision_ids=None):
        import numpy as np
        self.name, self.member, self.manifest_id = name, bool(member), manifest_id
        self.user_keys = np.asarray(user_keys, dtype=np.int64)
        self.dec_user = np.asarray(dec_user, dtype=np.int64)
        self.n_decisions = int(self.dec_user.size)
        self.n_users_population = int(n_users_population)
        self._batch_fn = batch_fn
        self.record = dict(record)
        self.client_idx = None if client_idx is None else np.unique(np.asarray(client_idx, dtype=np.int64))
        self.decision_ids = None if decision_ids is None else np.asarray(decision_ids, dtype=np.int64)
        if self.n_decisions == 0 or np.any(np.bincount(self.dec_user, minlength=self.user_keys.size) == 0):
            _refuse(f"{name}: a selected user has no TRAIN-window decision")

    def batches(self, batch_size: int) -> Iterator[tuple]:
        return self._batch_fn(int(batch_size))


def check_disjoint(members: Population, nonmembers: Population) -> dict:
    """Members and non-members share no store client and no decision (bands are disjoint by construction)."""
    import numpy as np
    checks = {}
    if members.client_idx is not None and nonmembers.client_idx is not None:
        checks["no_shared_store_client"] = int(np.intersect1d(members.client_idx, nonmembers.client_idx).size) == 0
    if members.decision_ids is not None and nonmembers.decision_ids is not None:
        checks["no_shared_decision"] = int(np.intersect1d(members.decision_ids, nonmembers.decision_ids).size) == 0
    checks["member_flags"] = members.member is True and nonmembers.member is False
    if not all(checks.values()):
        _refuse(f"members / non-members overlap: {[k for k, v in checks.items() if not v]}")
    return checks


def build_train_population(name: str, n_users: int, rng, *, registry_path: Path, paths=None, ds_root=None,
                           verify: Optional[Callable] = None, log: Callable = lambda *_: None) -> Population:
    """The real population: registry-verified TRAIN manifest -> window check -> seeded users -> a subset store of their
    TRAIN decisions only (v4int.cache, USER_HISTORY_L50) -> batches exactly as v4int.providers.StoreTrainProvider."""
    import numpy as np
    spec = POPULATIONS[name]
    mid = spec["manifest_id"]
    refuse_window_manifest_id(mid)                                   # before any file is opened
    from v4data.transform import build_spec
    from v4hist.policy import store_splits
    from v4int.cache import build_subset_store
    from v4int.providers import ProviderIntegrityError, add_session_offsets, manifest_binding_of
    from v4int.renorm import resolve_transform_state
    from v4int.verify import verify_manifest_by_registry
    if verify is None:
        vm = verify_manifest_by_registry(mid, registry_path, paths, root=V4, expect_kind="TRAIN", log=log)
    else:
        vm = verify(mid)
    if str(vm.header.get("status")) != spec["status"]:
        _refuse(f"{mid}: header status {vm.header.get('status')!r} is not the registered {spec['status']}")
    cols = vm.columns
    wc = window_check(mid, vm.kind, vm.header, cols)
    cord = np.asarray(cols["client_ord"], dtype=np.int64)
    did_all = np.asarray(cols["decision_id"], dtype=np.int64)
    tgt_all = np.asarray(cols["target_class"], dtype=np.int64)
    n_pop = int(np.unique(cord).size)
    chosen = select_users(cord, n_users, rng)
    m = np.isin(cord, chosen)
    o = np.argsort(did_all[m], kind="stable")
    ids, tgt, own = did_all[m][o], tgt_all[m][o], cord[m][o]
    if ids.size > 1 and not np.all(ids[1:] > ids[:-1]):
        _refuse(f"{mid}: the selected decision ids are not unique")
    dec_user = np.searchsorted(chosen, own)
    binding = manifest_binding_of(vm)
    tstate_sha = vm.transform_state_self_sha256
    vm.columns = None
    parts = store_splits(HISTORY_POLICY, ("TRAIN",))
    store, ds, info = build_subset_store(ids, splits=("TRAIN",), kind="loss", state_sha256=tstate_sha,
                                         history_policy=HISTORY_POLICY, ds_root=ds_root, log=log)
    sc = store_window_check(mid, info, ds, ids, tgt, parts)
    sc["checks"]["store_partitions_reported_TRAIN_only"] = tuple(info.store_splits) == ("TRAIN",)
    cidx = np.asarray(ds.client_idx, dtype=np.int64)
    pairs = np.unique(np.stack([dec_user, cidx], axis=1), axis=0)     # (manifest user, store client) must be 1:1
    sc["checks"]["one_store_client_per_user"] = bool(pairs.shape[0] == chosen.size
                                                     and np.unique(pairs[:, 0]).size == chosen.size
                                                     and np.unique(pairs[:, 1]).size == chosen.size)
    if not all(sc["checks"].values()):
        _refuse(f"{mid}: store window check failed {[k for k, v in sc['checks'].items() if not v]}")
    renorm, tstate = resolve_transform_state(None, build_spec(), tstate_sha)

    def batches(bs: int):
        for a in range(0, int(ids.size), bs):
            pos = np.arange(a, min(int(ids.size), a + bs), dtype=np.int64)
            b = add_session_offsets(store.batch(ds, pos), store, ds, pos)
            if renorm is not None:
                b = renorm.apply(b)
            if not (np.array_equal(b["decision_id"].numpy(), ids[pos])
                    and np.array_equal(b["target_class"].numpy(), tgt[pos]) and bool(b["loss_mask"].all())):
                raise ProviderIntegrityError(f"{mid}: an E1 batch disagrees with the manifest (decision_id, "
                                             "target_class or loss_mask)")
            yield pos, b

    record = {"manifest_id": mid, "registry_status": spec["status"], "member": spec["member"],
              "manifest_binding": binding, "population_sha256_selection_key_v2":
              vm.header.get("population_sha256_selection_key_v2"), "n_users_population": n_pop,
              "n_users_selected": int(chosen.size), "n_decisions_selected": int(ids.size),
              "history_policy": HISTORY_POLICY, "store_partitions": list(parts), "transform_state": tstate,
              "window_check": wc, "store_check": sc}
    pop = Population(name=name, member=spec["member"], manifest_id=mid, user_keys=chosen, dec_user=dec_user,
                     batch_fn=batches, n_users_population=n_pop, record=record, client_idx=cidx, decision_ids=ids)
    pop.cache_info = info.as_json()
    return pop


# ------------------------------------------------------------------------------------------------ targets
def parse_targets(items: Sequence[str]) -> dict:
    """LABEL=PATH -> {label: Path} in TARGET_ORDER. Unknown / repeated labels, a missing mandatory target, a directory
    or a checkpoint file the label does not admit are refused (before any file is read)."""
    out: dict = {}
    for it in items or ():
        if "=" not in str(it):
            _refuse(f"--targets entry {it!r} is not LABEL=PATH")
        label, path = str(it).split("=", 1)
        if label not in TARGETS:
            _refuse(f"--targets label {label!r} is not a registered E1 target {list(TARGET_ORDER)}")
        if label in out:
            _refuse(f"--targets names {label} twice")
        p = Path(path)
        if not p.is_file():
            _refuse(f"{label}: {p} is not a checkpoint file (give the file: best.pt / endpoint_6.0.pt)")
        if p.name not in TARGETS[label]["files"]:
            _refuse(f"{label}: {p.name} is not admitted ({'/'.join(TARGETS[label]['files'])}"
                    + ("; the accountant epsilon covers the 6.0 endpoint only, PREG2 §9.1" if TARGETS[label]["class"]
                       == "DP" else "") + ")")
        out[label] = p
    missing = [t for t in TARGET_ORDER if TARGETS[t]["mandatory"] and t not in out]
    if missing:
        _refuse(f"mandatory E1 targets missing: {missing} (only the DP arms may be absent; PRE is the null control)")
    return {t: out[t] for t in TARGET_ORDER if t in out}


def _args(label: str, ckpt: Path, kind: str) -> SimpleNamespace:
    return SimpleNamespace(ckpt=Path(ckpt), ckpt_kind=kind, variant=VARIANT, recipe_id=RECIPE, seed=TARGET_SEED,
                           arm_label=label, role="MAIN", selection_rule=FILE_RULE[Path(ckpt).name], dry_run=False,
                           pf_store=None, expect_pf_store_sha256=None, registry_ext=None)


def _launch_method(ckpt: Path) -> Optional[str]:
    run_dir = Path(ckpt).parent.parent.parent
    ls = sorted(run_dir.glob("LAUNCH_*.json"), key=lambda q: int(q.stem.split("_")[1]))
    if not ls:
        return None
    return ((R._strict(ls[0]).get("arm_spec") or {}).get("cfg") or {}).get("method")


def check_theta_pre_init(label: str, spec: Mapping) -> dict:
    init = spec.get("init_from") or {}
    bad = {k: init.get(k) for k, want in (("checkpoint_sha256", P2.PRE_CHECKPOINT_SHA256),
                                          ("state_digest", P2.PRE_STATE_DIGEST)) if init.get(k) != want}
    if bad:
        _refuse(f"{label}: the run's init_from is not the registered theta_pre ({bad})")
    return dict(init)


def check_dp_source(label: str, src: Mapping) -> dict:
    """The W* DP-chain run: LAUNCH arm = the label, spec = launch_fl.p2_dp_arm_spec of itself, run id
    <arm>_SMALL_s2026, theta_pre start, LAUNCH dp_z epsilon = the registered accountant epsilon."""
    import v4int.launch_fl as LF
    la = src.get("launch") or {}
    spec = la.get("arm_spec") or {}
    if src.get("arm") != label or not LF.is_p2_dp_arm(label):
        _refuse(f"{label}: the run's LAUNCH arm is {src.get('arm')!r}")
    if LF.spec_digest(spec) != la.get("arm_spec_sha256") or spec.get("arm") != label:
        _refuse(f"{label}: the LAUNCH arm spec does not hash to its arm_spec_sha256 / names another arm")
    try:
        LF.check_p2_dp_spec(spec)
    except LF.LaunchRefused as e:
        _refuse(f"{label}: launch_fl.check_p2_dp_spec refuses the run's spec: {e}")
    want_rid = f"{label}_SMALL_s{TARGET_SEED}"
    if la.get("run_id") != want_rid:
        _refuse(f"{label}: run id {la.get('run_id')!r} is not the bound MAIN run {want_rid}")
    try:
        eps = float((la.get("dp_z") or {}).get("epsilon"))
    except (TypeError, ValueError):
        eps = float("nan")
    if eps != TARGETS[label]["epsilon"] or LF.P2_DP_EPSILON.get(label) != TARGETS[label]["epsilon"]:
        _refuse(f"{label}: the LAUNCH dp_z epsilon is not the registered accountant epsilon {TARGETS[label]['epsilon']}")
    return {"launch_arm": label, "run_id": want_rid, "arm_spec_sha256": la.get("arm_spec_sha256"),
            "check_p2_dp_spec": True, "accountant_epsilon": eps, "dp_z": dict(la.get("dp_z") or {}),
            "theta_pre": check_theta_pre_init(label, spec)}


def verify_source(label: str, ckpt: Path) -> dict:
    """Every torch-free identity check of the target's run (reeval / reeval_p2 / launch_fl functions), before a weight
    is read. Returns {kind, sha256, file, rule, src, ctx, checks}."""
    ckpt = Path(ckpt)
    t = TARGETS[label]
    got = R._sha(ckpt)
    if label == "PRE" and got != P2.PRE_CHECKPOINT_SHA256:
        _refuse(f"PRE: checkpoint sha256 is not the registered theta_pre {P2.PRE_CHECKPOINT_SHA256}")
    ctx = {"plan": {"label": label}, "checks": {}, "spec": None}
    checks: dict = {}
    try:
        if t["kind"] == "central":
            args = _args(label, ckpt, "central")
            src = R.source_central(ckpt, args)
            if label == "PRE":
                if src.get("central_source") != "PRE":
                    _refuse("PRE: the checkpoint's run is not the registered PRE source (theta_pre)")
                checks["pre_init_record"] = R.check_pre_init_record(src, ckpt, got)
                reg = R.verify_source_registry("PRE", src["start"], R.EXT_DIR / "REGISTRY_V2.json")
                checks["pre_registry"] = {k: v for k, v in reg.items() if k != "checks"}
            else:
                checks["c1"] = P2.check_c1_source(src, ckpt, args, ctx)
            checks["run_start_sha256"] = src["start_sha256"]
        else:
            pf = _launch_method(ckpt) == "PF"
            args = _args(label, ckpt, "pf" if pf and label == "P2_WSTAR" else "fl")
            if label == "P2_WSTAR":
                src = P2.p2_source_fl(ckpt, args, ctx)
                checks["p2_w"] = dict(ctx["checks"].get("fl") or {})
                checks["personal_component"] = ("NOT_EVALUATED (WC5 personal head: the shared server-side weights "
                                                "are audited)" if pf else None)
            else:
                src = R.source_fl(ckpt, args)
                if label == "FA_WARM_FROZEN":
                    if src.get("arm") != "FA_WARM_FROZEN":
                        _refuse(f"FA_WARM_FROZEN: the run's LAUNCH arm is {src.get('arm')!r}")
                    R.check_warm_line(args, src)
                    checks["warm"] = {"launch_arm": src["arm"], "run_id": src["run_id"],
                                      "theta_pre": check_theta_pre_init(label, src["launch"].get("arm_spec") or {})}
                else:
                    checks["dp"] = check_dp_source(label, src)
            checks["launch_files"] = src["launch_files"]
    except R.ReevalRefused as e:
        _refuse(f"{label}: source check refused: {e}")
    return {"label": label, "kind": t["kind"], "sha256": got, "file": ckpt.name, "rule": FILE_RULE[ckpt.name],
            "path": str(ckpt), "src": src, "ctx": ctx, "args": args, "checks": checks}


def read_target_plain(label: str, path: Path) -> dict:
    """The checkpoint object, format checks only: central = a v4train {kind, meta, model} checkpoint whose kind is its
    file's (endpoint mark 6.0); fl = an fl.checkpoint weights payload."""
    from collections.abc import Mapping as _M
    path = Path(path)
    kind = TARGETS[label]["kind"]
    if kind == "central":
        import torch
        obj = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(obj, _M) or not isinstance(obj.get("model"), _M):
            _refuse(f"{label}: {path.name} is not a v4train {{meta, model}} checkpoint")
        try:
            R.check_source_checkpoint(label, obj, path)
        except R.ReevalRefused as e:
            _refuse(str(e))
        mv = (obj.get("meta") or {}).get("model_variant")
        if mv != VARIANT:
            _refuse(f"{label}: checkpoint meta model_variant {mv!r} is not {VARIANT}")
        return {"kind": kind, "obj": obj, "sha256": R._sha(path), "file": path.name, "rule": FILE_RULE[path.name]}
    from fl.checkpoint import load_checkpoint
    payload = load_checkpoint(path)
    if payload.get("kind") != "weights" or not isinstance(payload.get("theta"), _M):
        _refuse(f"{label}: {path.name} is not an FL weights payload")
    return {"kind": kind, "obj": payload, "sha256": R._sha(path), "file": path.name, "rule": FILE_RULE[path.name]}


def read_target_verified(label: str, path: Path, source: Mapping, schema_self_sha256: Optional[str]) -> dict:
    """The real reader: the source record of verify_source (same sha256), then the payload checks of reeval /
    reeval_p2 on the loaded object."""
    item = read_target_plain(label, path)
    if item["sha256"] != source["sha256"]:
        _refuse(f"{label}: the checkpoint changed between the source check and the read")
    obj, args = item["obj"], source["args"]
    try:
        if item["kind"] == "central":
            start = source["src"]["start"]
            R.check_central_meta(obj, start, args)
            if schema_self_sha256 is not None and (start.get("schema") or {}).get("schema_self_sha256") != \
                    schema_self_sha256:
                _refuse(f"{label}: the run's schema is not the LOCK_V2 production schema")
            if label == "P2_C1_FT_CENTRAL_FULL":
                P2.check_c1_checkpoint(obj, path, source["ctx"])
        else:
            R.check_fl_payload(obj, source["src"])
            if label != "FA_WARM_FROZEN":
                P2.check_payload_mark(obj, Path(path).name)
            if label == "P2_WSTAR":
                item["merged_lora"] = P2.check_lora_payload(obj, source["ctx"]["spec"])
    except R.ReevalRefused as e:
        _refuse(f"{label}: payload check refused: {e}")
    item["source_checks"] = source["checks"]
    if label == "PRE":
        item["expect_state_digest"] = P2.PRE_STATE_DIGEST
    return item


def apply_target(adapter, item: Mapping) -> str:
    """The target's weights into the one SASREC_D64_B2 adapter (central: strict load_state_dict; fl: reeval.load_fl -
    same keys / shapes / dtypes, FIXED buffers identical); returns the loaded state digest."""
    from fl.numerics import state_digest
    if item["kind"] == "central":
        adapter.module.load_state_dict(item["obj"]["model"], strict=True)
        d = state_digest(adapter.broadcast_state())
        meta_d = (item["obj"].get("meta") or {}).get("state_digest")
        if meta_d is not None and meta_d != d:
            _refuse("the loaded weights' state digest differs from the checkpoint's recorded state_digest")
    else:
        try:
            d = R.load_fl(adapter, item["obj"])["state_digest"]
        except R.ReevalRefused as e:
            _refuse(str(e))
    if item.get("expect_state_digest") is not None and d != item["expect_state_digest"]:
        _refuse(f"the loaded state digest is not the registered {item['expect_state_digest']}")
    adapter.module.eval()
    return d


# ------------------------------------------------------------------------------------------------ scoring
def user_mean_ce(adapter, pop: Population, device, batch_size: int = SCORE_BATCH):
    """Per selected user: the mean full-K CE of the current weights over the user's TRAIN-window decisions (eval mode,
    no gradient; summed in float64 in a fixed order)."""
    import numpy as np
    import torch
    import torch.nn.functional as F
    from v4train.objective import to_device
    n_u = int(pop.user_keys.size)
    sums, cnt = np.zeros(n_u, dtype=np.float64), np.zeros(n_u, dtype=np.int64)
    adapter.module.eval()
    with torch.no_grad():
        for pos, b in pop.batches(batch_size):
            tc = b["target_class"]
            lm = b.get("loss_mask")
            if bool((tc < 0).any()) or (lm is not None and not bool(lm.all())):
                _refuse(f"{pop.name}: a batch row is not loss-eligible")
            logits = adapter.scores(to_device(b, device))
            ce = F.cross_entropy(logits, tc.to(logits.device, dtype=torch.long), reduction="none")
            ce = ce.detach().to("cpu", dtype=torch.float64).numpy()
            if not np.all(np.isfinite(ce)):
                _refuse(f"{pop.name}: a CE value is not finite")
            u = pop.dec_user[np.asarray(pos, dtype=np.int64)]
            np.add.at(sums, u, ce)
            np.add.at(cnt, u, 1)
    if np.any(cnt == 0) or int(cnt.sum()) != pop.n_decisions:
        _refuse(f"{pop.name}: not every selected decision was scored exactly once")
    return sums / cnt


# ------------------------------------------------------------------------------------------------ the audit
def run_audit(out_dir: Path, *, targets: Mapping[str, Path], members: Population, nonmembers: Population, adapter,
              device, read_target: Callable, n_boot: int = N_BOOT, score_batch: int = SCORE_BATCH,
              say: Callable = print, extra_record: Optional[Mapping] = None) -> dict:
    """PRE first; validity gate; then every other available target. Writes E1_USER_SCORES.json and E1_AUDIT.json."""
    import numpy as np
    out = Path(out_dir)
    if out.exists():
        _refuse(f"output directory exists: {out}")
    spec = spec_text()
    say("E1 STEP spec_check: OK")
    if "PRE" not in targets:
        _refuse("PRE (the null control) is mandatory")
    disjoint = check_disjoint(members, nonmembers)
    say("E1 STEP populations: TRAIN_WINDOW_CHECK_PASS")
    srng = np.random.default_rng(SPLIT_STREAM)
    half_m, half_n = split_half(members.user_keys.size, srng), split_half(nonmembers.user_keys.size, srng)
    sel = selection_record(members, nonmembers, half_m, half_n)
    registered = (members.user_keys.size == N_MEMBERS and nonmembers.user_keys.size == N_NONMEMBERS
                  and int(n_boot) == N_BOOT and int(score_batch) == SCORE_BATCH)

    inputs, per_target, scores = {}, {}, {}
    ce_pre = None

    def score(label: str) -> tuple:
        item = read_target(label, targets[label])
        digest = apply_target(adapter, item)
        inputs[label] = {"path": str(targets[label]), "sha256": item["sha256"], "file": item["file"],
                         "selection_rule": item["rule"], "kind": item["kind"], "loaded_state_digest": digest,
                         "source_checks": json_safe(item.get("source_checks")),
                         "merged_lora": json_safe(item.get("merged_lora"))}
        return user_mean_ce(adapter, members, device, score_batch), user_mean_ce(adapter, nonmembers, device,
                                                                                  score_batch)

    def metrics(ce_m, ce_n) -> dict:
        raw = score_metrics(-ce_m, -ce_n, half_m, half_n, n_boot=n_boot, seed=BOOTSTRAP_SEED)
        cal = score_metrics(-(ce_m - ce_pre[0]), -(ce_n - ce_pre[1]), half_m, half_n, n_boot=n_boot,
                            seed=BOOTSTRAP_SEED)
        return {"calibrated": cal, "raw_mean_ce": raw}

    ce_pre = score("PRE")
    say("E1 STEP target PRE: SCORED")
    scores["PRE"] = ce_pre
    m_pre = metrics(*ce_pre)
    val = validity(m_pre["raw_mean_ce"])
    say(f"E1 STEP validity: {val['status']}")
    if val["status"] == VALID:
        per_target["PRE"] = dict(m_pre, status=S_VALID_NULL, role="NULL_CONTROL")
    else:
        per_target["PRE"] = {"calibrated": strip_eps(m_pre["calibrated"]), "raw_mean_ce": strip_eps(
            m_pre["raw_mean_ce"]), "status": INVALID, "role": "NULL_CONTROL"}
    for label in TARGET_ORDER[1:]:
        t = TARGETS[label]
        if label not in targets:
            per_target[label] = {"status": S_NA, "role": t["class"]}
        elif val["status"] != VALID:
            per_target[label] = {"status": S_NOT_SCORED, "role": t["class"]}
            inputs[label] = {"path": str(targets[label]), "sha256": R._sha(targets[label]), "scored": False}
        else:
            ce = score(label)
            scores[label] = ce
            m = metrics(*ce)
            acc = (((inputs[label].get("source_checks") or {}).get("dp") or {}).get("accountant_epsilon")
                   if t["class"] == "DP" else None)
            verdict = classify(label, m["calibrated"], acc)
            per_target[label] = dict(m, role=t["class"], **verdict)
        say(f"E1 TARGET {label}: {per_target[label]['status']}")
    audit_status = val["status"]

    out.mkdir(parents=True)
    users = {"population": [members.name] * int(members.user_keys.size) + [nonmembers.name] *
             int(nonmembers.user_keys.size),
             "manifest_id": [members.manifest_id] * int(members.user_keys.size) + [nonmembers.manifest_id] *
             int(nonmembers.user_keys.size),
             "client_ord": [int(x) for x in members.user_keys] + [int(x) for x in nonmembers.user_keys],
             "member": [True] * int(members.user_keys.size) + [False] * int(nonmembers.user_keys.size),
             "half": ["A" if x else "B" for x in np.concatenate((half_m, half_n))],
             "n_decisions": [int(x) for x in np.concatenate((np.bincount(members.dec_user,
                                                                         minlength=members.user_keys.size),
                                                             np.bincount(nonmembers.dec_user,
                                                                         minlength=nonmembers.user_keys.size)))]}
    tgt_scores = {}
    for label, (cm, cn) in scores.items():
        ce = np.concatenate((cm, cn))
        cal = ce - np.concatenate(ce_pre)
        tgt_scores[label] = {"mean_ce": [float(x) for x in ce], "score_calibrated": [float(x) for x in cal]}
    sc_obj = with_self_sha({"format": SCORES_FORMAT, "task": TASK, "label": LABEL,
                            "score_definition": "mean_ce = per-user mean full-K CE over the user's TRAIN-window "
                                                "decisions; score_calibrated = mean_ce(theta) - mean_ce(theta_pre); "
                                                "the attack statistic is -score",
                            "users": users, "targets": tgt_scores, "audit_status": audit_status,
                            "eps_lb_reported": audit_status == VALID})
    sc_sha = write_json(out / SCORES_FILE, sc_obj)
    audit = with_self_sha({
        "format": FORMAT, "task": TASK, "authority": AUTHORITY, "label": LABEL, "spec": spec,
        "protocol": {"seed_members": SEED, "seed_nonmembers": NONMEMBER_STREAM, "seed_split": SPLIT_STREAM,
                     "bootstrap_seed": BOOTSTRAP_SEED, "n_boot": int(n_boot), "ci_quantiles": list(CI_QUANTILES),
                     "cp_alpha": CP_ALPHA, "threshold_fpr": str(THRESHOLD_FPR),
                     "fpr_levels": [str(f) for f in FPR_LEVELS], "auc_material": AUC_MATERIAL,
                     "tpr1_material": TPR1_MATERIAL, "score_batch": int(score_batch),
                     "history_policy": HISTORY_POLICY, "attack_statistic": "-score (member iff -score > threshold)",
                     "auc_population": "all selected users (halves A + B)",
                     "validity_score": "raw_mean_ce (PRE's calibrated score is identically 0)",
                     "registered_scale": bool(registered)},
        "user_selection": sel, "window_check": {"members": members.record.get("window_check"),
                                                "nonmembers": nonmembers.record.get("window_check"),
                                                "members_store": members.record.get("store_check"),
                                                "nonmembers_store": nonmembers.record.get("store_check"),
                                                "disjoint": disjoint, "reads": "TRAIN-window rows only; no evaluation "
                                                "manifest, no VALIDATION / TEST row or label"},
        "populations": json_safe({"members": {k: v for k, v in members.record.items()
                                              if k not in ("window_check", "store_check")},
                                  "nonmembers": {k: v for k, v in nonmembers.record.items()
                                                 if k not in ("window_check", "store_check")}}),
        "inputs": inputs, "validity": val, "audit_status": audit_status, "eps_lb_reported": audit_status == VALID,
        "targets": per_target,
        "targets_not_available": [t for t in TARGET_ORDER if per_target[t]["status"] == S_NA],
        "per_user_scores": {"file": SCORES_FILE, "sha256": sc_sha},
        "code": {"audit_e1_mia.py": R._sha(Path(__file__)), "reeval.py": R._sha(Path(R.__file__)),
                 "reeval_p2.py": R._sha(Path(P2.__file__))},
        **dict(extra_record or {})})
    a_sha = write_json(out / AUDIT_FILE, audit)
    say(f"E1 AUDIT: {audit_status}")
    say(f"E1 OUTPUT {AUDIT_FILE} sha256 {a_sha}")
    say(f"E1 OUTPUT {SCORES_FILE} sha256 {sc_sha}")
    return audit


# ------------------------------------------------------------------------------------------------ CLI
def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="PPSI V4 E1 user-level membership-inference audit (PREG2 §7.1; P2-14a)")
    ap.add_argument("--targets", nargs="+", required=True, metavar="LABEL=CKPT")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    ap.add_argument("--dataset-root", type=Path, help="the frozen v1 root (default: $V4_DATASET_ROOT, as every tool)")
    ap.add_argument("--dry-run", action="store_true")
    return ap.parse_args(argv)


def run(args, say: Callable = print) -> dict:
    t0 = time.time()
    plan = parse_targets(args.targets)
    if Path(args.out).exists():
        _refuse(f"output directory exists: {args.out}")
    spec_text()
    sources = {}
    for label, p in plan.items():
        sources[label] = verify_source(label, p)
        say(f"E1 STEP source {label}: OK")
    if args.dry_run:
        return {"dry_run": True, "targets": {k: {"sha256": v["sha256"], "file": v["file"], "rule": v["rule"]}
                                             for k, v in sources.items()},
                "not_available": [t for t in TARGET_ORDER if t not in plan], "status": "DRY_RUN_PASS"}
    if args.dataset_root is not None:
        os.environ["V4_DATASET_ROOT"] = str(args.dataset_root)
    if args.device == "cuda":                                        # the T3 lock value, before CUDA initialises
        from v4int.launch_central import PREG1_LOCK
        t3 = json.loads(Path(PREG1_LOCK).read_text(encoding="utf-8"))["T3_eval_numerics"]["flags"][
            "CUBLAS_WORKSPACE_CONFIG"]
        if os.environ.get("CUBLAS_WORKSPACE_CONFIG") not in (None, t3):
            _refuse(f"inherited CUBLAS_WORKSPACE_CONFIG differs from the T3 lock value {t3!r}")
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = t3
    import v4train  # noqa: F401  (ppsi_env first)
    import numpy as np
    import torch
    from fl.numerics import enforce_strict_fp32
    from v4data.common import dataset_root
    from v4int.launch_central import LaunchRefused, bind_data_protocol_lock_v2, load_production_schema
    from v4int.verify import DataPaths
    from v4models.build import build
    if args.device == "cuda":
        if not torch.cuda.is_available():
            _refuse("--device cuda: no CUDA device is visible")
        dev = torch.device("cuda:0")
        torch.cuda.set_device(dev)
    else:
        dev = torch.device("cpu")
    numerics = enforce_strict_fp32()
    torch.set_num_threads(4)
    quiet = lambda *_: None                                          # noqa: E731  (no value is ever printed)
    paths = DataPaths.default()
    try:
        lockctx = bind_data_protocol_lock_v2(paths, smoke=False)
        schema = load_production_schema(lockctx["schema_id"], lockctx["schema_self_sha256"],
                                        lockctx["schema_registry_path"])
    except LaunchRefused as e:
        _refuse(str(e))
    root = dataset_root()
    bm = build(FAMILY, schema, TARGET_SEED, dev, v1_root=root, variant=VARIANT)
    say("E1 STEP model: BUILT")
    reg = Path(lockctx["manifest_registry_path"])
    members = build_train_population("MEMBERS", N_MEMBERS, np.random.default_rng(SEED), registry_path=reg,
                                     paths=paths, ds_root=root, log=quiet)
    say("E1 STEP population MEMBERS: TRAIN_WINDOW_CHECK_PASS")
    nonmembers = build_train_population("NONMEMBERS", N_NONMEMBERS, np.random.default_rng(NONMEMBER_STREAM),
                                        registry_path=reg, paths=paths, ds_root=root, log=quiet)
    say("E1 STEP population NONMEMBERS: TRAIN_WINDOW_CHECK_PASS")
    reader = lambda label, path: read_target_verified(label, path, sources[label], schema.self_sha256)  # noqa: E731
    lock_rec = {"data_protocol": {"lock_v2_self_sha256": lockctx["lock_self_sha256"],
                                  "registry": {"path": lockctx["manifest_registry_rel_path"],
                                               "file_sha256": lockctx["manifest_registry_file_sha256"],
                                               "self_sha256": lockctx["manifest_registry_self_sha256"]},
                                  "schema": {"id": lockctx["schema_id"], "self_sha256": lockctx["schema_self_sha256"]}}}
    audit = run_audit(Path(args.out), targets=plan, members=members, nonmembers=nonmembers, adapter=bm.adapter,
                      device=dev, read_target=reader, say=say, extra_record=lock_rec)
    rec = {"format": RUN_RECORD_FORMAT, "task": TASK, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "argv": list(sys.argv), "device": {"type": dev.type, "name": (torch.cuda.get_device_name(dev)
                                                                       if dev.type == "cuda" else "cpu"),
                                              "torch": torch.__version__, "cuda": torch.version.cuda},
           "numerics": numerics, "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
           "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "dataset_root": str(root), "wall_seconds": round(time.time() - t0, 3),
           "cache": {"members": getattr(members, "cache_info", None),
                     "nonmembers": getattr(nonmembers, "cache_info", None)},
           "audit_self_sha256": audit["self_sha256"]}
    write_json(Path(args.out) / RUN_RECORD_FILE, json.loads(json.dumps(rec, default=str)))
    return {"status": "COMPLETE", "audit_status": audit["audit_status"], "out": str(args.out)}


def main(argv=None) -> int:
    args = parse_args(argv)
    try:
        rec = run(args)
        for label, t in (rec.get("targets") or {}).items():          # dry run: hashes only, never a value
            print(f"E1 DRY_RUN target {label}: {t['file']} {t['rule']} sha256 {t['sha256']}")
        print(f"E1 DONE: {rec['status']}")
        return 0
    except Exception as e:                                          # noqa: BLE001  (every refusal exits 1)
        print(f"E1_AUDIT REFUSED: {type(e).__name__}: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
