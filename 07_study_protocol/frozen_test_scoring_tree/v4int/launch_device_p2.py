"""v4int.launch_device_p2 (P2-DEVLOCK piece A, owner: Phase-2 device stack) - a SHIM around the UNEDITED
v4int.launch_device (pinned by sha256, LAUNCH_DEVICE_SHA256, checked before every use). It runs the registered Phase-1
device procedure - the FT_C local fine-tune with the registered SELECTION_FT_C cell, exactly as FT_PRE /
FT_FA_WARM_FROZEN use it - on the W* base, as the Phase-2 arm FT_P2_WSTAR (contrast Q5 = FT_P2_WSTAR - FT_PRE). It adds
admissions and checks, never a new training or evaluation path.

  python -B code/v4int/launch_device_p2.py --arm FT_P2_WSTAR --seed 2026|2027 --gpu <id> \
         --run-dir <...>/FT_P2_WSTAR_s<seed> (MAIN) | <...>/FT_P2_WSTAR_MAIN_TEST_s<seed> (MAIN_TEST) \
         --history-policy USER_HISTORY_L50 --role MAIN|MAIN_TEST \
         --base <P2_W MAIN run>/ckpt/P2_WSTAR_SMALL_s<seed>/best.pt [--base-kind fa_warm] [--base-sha256 <hex>] \
         [--registry-ext <REGISTRY_V2_TEST_EXT.json> (MAIN_TEST)] [launch_device's shard / resume options] [--dry-run]

Binding spec: contracts/PREG2.md (REGISTERED, DL 411, sha256 be0561ab...) §2 (FT_P2_WSTAR = the registered FT_C procedure
on top of W*), §5 (W* = the bound P2_WARM_SELECTION_FINAL winner, MAIN run), §6.3 T0a S1-S8, §9.1 Q5, §9.5 (seeds
{2026, 2027}); company/plans/PHASE2_IMPLEMENTATION_PLAN.md "FT_P2_WSTAR x2 (needs W* best.pt)"; company/gates/
P2_CONTRASTS_v3.json base_best_only = Q5; the reader tools/p2_block_index.py --p2-dev (P2-12 §3: LAUNCH start = {kind
P2_WSTAR, seed, checkpoint, file_sha256, run_id}).

Admitted (everything else refuses before any GPU work):
  arm     FT_P2_WSTAR only (every Phase-1 arm runs through the unedited launch_device).
  roles   MAIN, and MAIN_TEST (the MAIN users' RETRO_TEST role) only inside the sitting. Never CALIBRATION, UNSEEN_VAL,
          a panel, another TEST role or TEST2.
  seeds   2026, 2027 (PREG2 §9.5). --smoke is refused (S8: a smoke run would score MAIN users outside the plan).
  run dir FT_P2_WSTAR_s<seed> (MAIN) / FT_P2_WSTAR_MAIN_TEST_s<seed> (MAIN_TEST): the directory names
          tools/p2_block_index.py --p2-dev reads.
  base    ONLY best.pt of the MAIN P2_W run P2_WSTAR_SMALL_s<seed> of the SAME seed (Q5 base_best_only; endpoint_6.0.pt,
          another seed, theta_pre, a Phase-1 FL run, a CALIBRATION screen and every other file refuse). Checked
          torch-free in preflight from value-free records only (STATUS.json, ARM_SPEC.json, LAUNCH_<k>.json; the
          RUN_SUMMARY is never opened, S8): STATUS DONE_MAIN; ARM_SPEC arm P2_W, run id, arm spec sha256 =
          launch_fl.spec_digest; every LAUNCH leg agrees (arm / run id / arm spec sha / config digest / label /
          p2_selection), PRODUCTION, fl05 role MAIN; launch_fl.check_p2_spec (= p2_arm_spec of its own recipe), spec
          role MAIN, seed = --seed, init_from = theta_pre (pinned sha256 / state digest); the bound
          P2_WARM_SELECTION_FINAL (registered path name, format, stage FINAL, recipe / label = the spec's, sha256 = the
          registered file's current bytes); a personal-head (WC5, PF) recipe is refused pending a decision (open question
          in company/handoffs/P2-DEVLOCK.md). At load (run time): the fl weights payload of that run, config digest =
          the LAUNCH digest = launch_fl.fl_run_config(spec).digest(), mark >= 0.5 EFE (theta_pre is never a candidate),
          the merged-LoRA rule of reeval_p2.check_lora_payload, the loaded state reproduces the payload's digest. The
          base file sha256 is recorded (LAUNCH start.file_sha256, preflight.p2_device.base, every row).
  recipe  the UNEDITED launch_device path of FT_FA_WARM / FT_FA_WARM_FROZEN: --selection defaults to the registered
          contracts/device/SELECTION_FT_C.json (pinned e47965d2...; any other file refuses), the ONE cell FT_C_f0.3_P4 ->
          FT_P2_WSTAR_f0.3_P4, fl.finetune FTRecipe FT_C, fresh AdamW, B16, clip 1.0, wd 1e-5; rows in the device
          RESULT_ROWS format, method FT_P2_WSTAR, ft_procedure FT_C, plus study PREG2, p2_arm and the base identity
          (base_source P2_WSTAR, base_run_id, base_checkpoint best.pt, base_checkpoint_sha256, base_arm_spec_sha256,
          base_p2_label, base_p2_selection_sha256, the theta_pre lineage).
TEST (MAIN_TEST; T0a, PREG2 §6.3 S1 / S2; the gate pattern of v4int/reeval_p2.p2_test_gate): --base-sha256 is required
and the gate runs BEFORE any base byte is read: the DATA-09 gate (launch_device's own preflight hook AND
d9_gate.check_test_gate), P2_METHODS_LOCK.json + P2_SCRIPT_FREEZE.json on this host, the hash-chain-verified
TEST_EXPOSURE_REGISTER holds a LOCKED_EVALUATION / RETRO_TEST entry for the Phase-1 ALL_METHODS_LOCK sha AND one whose
all_methods_lock_sha256 is the P2_METHODS_LOCK file sha (naming P2_METHODS_LOCK in observations_description), and the
lock seals (P2_WSTAR, seed, PRACTICAL_BEST) with checkpoint_sha256 = the base (reeval_p2.p2_lock_arm, the one adapter)
AND (FT_P2_WSTAR, seed, PRACTICAL_BEST) with base_checkpoint_sha256 = the base, neither declared not run. Then
launch_device's own gated TEST route (test_roles.install / resolve_role, retro_store) runs unchanged.
How: inside p2_device_context the launch_device attributes below are replaced for ONE call and restored afterwards
(identity): the tables ARMS / LOCK_CODE (FA, as FT_FA_WARM: the base is an FA-method FL model) / PARADIGM / LIFECYCLE /
FT_PROCEDURE (FT_C) / SELECTION_ARM (FT_C) / FT_WARM_BASE (FT_P2_WSTAR -> P2_WSTAR) gain the FT_P2_WSTAR entry;
load_warm_base and ft_warm_row_labels dispatch on base_arm / start kind P2_WSTAR to this module and call the originals
otherwise. launch_device.py is not edited. S8: nothing here prints or records a value (the base mark / round are not
recorded; a dry run prints no metric).
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Mapping, Optional

HERE = Path(__file__).resolve().parent
CODE = HERE.parent
V4 = CODE.parent
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))
sys.dont_write_bytecode = True

from v4int import launch_device as D  # noqa: E402  (no torch at import)

TASK = "P2-DEVLOCK v4int.launch_device_p2"
PREG2_SHA256 = "be0561abe5ceec450d3a75bef49853a3c480d3847f75559dd94e9e0545df18a8"
AUTHORITY = (f"contracts/PREG2.md (REGISTERED, DL 411, sha256 {PREG2_SHA256}) §2, §5, §6.3 S1-S8, §9.1 Q5, §9.5; "
             "company/plans/PHASE2_IMPLEMENTATION_PLAN.md FT_P2_WSTAR; company/gates/P2_CONTRASTS_v3.json base_best_only")
LAUNCH_DEVICE_SHA256 = "1968ca9550ed9ac70d9310753aa5aa33243e7bb79cabc6e4fa7cd95ad701642c"   # the wrapped launch_device.py
ARM = "FT_P2_WSTAR"
BASE_ARM = "P2_WSTAR"                       # the block arm of the base = the LAUNCH start kind (tools/p2_block_index)
BASE_LAUNCH_ARM = "P2_W"                    # launch_fl.P2_ARMS: the LAUNCH / ARM_SPEC arm of a P2 FL run
BASE_FILE = "best.pt"                       # Q5 base_best_only (P2_CONTRASTS_v3): the W* PRACTICAL_BEST checkpoint
BASE_RUN_ID = "P2_WSTAR_SMALL_s{seed}"      # = launch_fl.p2_run_id of a MAIN run bound to the selection
PB = "PRACTICAL_BEST"
SEEDS = (2026, 2027)
ROLES = ("MAIN",)
TEST_ROLES = ("MAIN_TEST",)
RUN_DIR_NAME = {"MAIN": "FT_P2_WSTAR_s{seed}", "MAIN_TEST": "FT_P2_WSTAR_MAIN_TEST_s{seed}"}   # p2_block_index P2_DEV_DIR
ADMIT_PF_BASE = False                       # a WC5 (personal head) W*: open question, fails closed (P2-DEVLOCK.md)
PRE_CHECKPOINT_SHA256 = "cf08588c11f019cc760f129ba7ba6b23806297660e0170f9938436a5cb88cf78"   # theta_pre (= reeval_p2)
PRE_STATE_DIGEST = "ee33345be4f20b61a53cd80fa11e1af702031bfcd1425af490825cd0019dc54a"
SELECTION_FINAL = V4 / "contracts" / "fl" / "P2_WARM_SELECTION_FINAL.json"   # the file the MAIN P2_W run was bound to
P2_METHODS_LOCK = V4 / "company" / "gates" / "P2_METHODS_LOCK.json"
P2_SCRIPT_FREEZE = V4 / "company" / "gates" / "P2_SCRIPT_FREEZE.json"
AML_PATH: Optional[Path] = None             # None -> v4data09.d9_gate defaults (ALL_METHODS_LOCK / TEST_EXPOSURE_REGISTER)
REGISTER_PATH: Optional[Path] = None
LOCKED_EVALUATION = "LOCKED_EVALUATION"
TABLE_ADD = {"LOCK_CODE": "FA", "PARADIGM": "CLIENT_FINETUNE", "LIFECYCLE": "FRESH_ADAMW_PER_CLIENT",
             "FT_PROCEDURE": "FT_C", "SELECTION_ARM": "FT_C", "FT_WARM_BASE": BASE_ARM}
PATCHED = ("ARMS",) + tuple(TABLE_ADD) + ("load_warm_base", "ft_warm_row_labels")


class P2DeviceRefused(D.LaunchRefused):
    pass


def _refuse(msg: str):
    raise P2DeviceRefused(msg)


def _sha(p) -> str:
    return D._sha_file(Path(p))


def _canon(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def _strict(p: Path) -> dict:
    try:
        obj = json.loads(Path(p).read_text(encoding="utf-8"),
                         parse_constant=lambda c: (_ for _ in ()).throw(ValueError(f"non-strict constant {c}")))
    except (OSError, ValueError) as e:
        _refuse(f"{p}: not readable strict JSON ({e})")
    if not isinstance(obj, dict):
        _refuse(f"{p}: not a JSON object")
    return obj


def check_launch_device_unchanged() -> str:
    """The wrapped module must be the registered, byte-identical v4int/launch_device.py (the shim assumes its code)."""
    got = _sha(Path(D.__file__))
    if got != LAUNCH_DEVICE_SHA256:
        _refuse(f"v4int/launch_device.py sha256 {got} is not the registered {LAUNCH_DEVICE_SHA256}: the P2 device shim "
                "refuses to wrap it")
    return got


# ------------------------------------------------------------------------------------------------ the context
@contextlib.contextmanager
def p2_device_context():
    """The FT_P2_WSTAR entries in launch_device's tables + the two dispatchers, for one call; restored by identity."""
    if getattr(D, "__p2_device_context__", False):
        _refuse("a P2 device launch is already running in this process (no nesting)")
    o_load, o_labels = D.load_warm_base, D.ft_warm_row_labels

    def load_warm_base(adapter, path, *, base_arm, seed, want_sha256=None):
        if base_arm == BASE_ARM:
            return load_p2_wstar_base(adapter, path, seed=seed, want_sha256=want_sha256)
        return o_load(adapter, path, base_arm=base_arm, seed=seed, want_sha256=want_sha256)

    def ft_warm_row_labels(start_rec, selection_rec):
        if (start_rec or {}).get("kind") == BASE_ARM:
            return p2_row_labels(start_rec, selection_rec)
        return o_labels(start_rec, selection_rec)

    patches = {"ARMS": tuple(D.ARMS) + (ARM,), "load_warm_base": load_warm_base,
               "ft_warm_row_labels": ft_warm_row_labels}
    patches.update({k: dict(getattr(D, k), **{ARM: v}) for k, v in TABLE_ADD.items()})
    saved = {k: getattr(D, k) for k in patches}
    try:
        for k, v in patches.items():
            setattr(D, k, v)
        D.__p2_device_context__ = True
        yield
    finally:
        for k, v in saved.items():
            setattr(D, k, v)
        D.__p2_device_context__ = False


# ------------------------------------------------------------------------------------------------ admissions (no file)
def check_args(args) -> None:
    """Every P2 refusal that needs no file (before launch_device's preflight)."""
    if args.arm != ARM:
        _refuse(f"launch_device_p2 runs --arm {ARM} only, not {args.arm} (every Phase-1 arm runs through the unedited "
                "v4int/launch_device.py)")
    role = str(args.role)
    if "TEST2" in role.upper():
        _refuse(f"role {role}: TEST2 is not part of this study (owner O2 = NO, DL 415)")
    if role not in ROLES + TEST_ROLES:
        _refuse(f"{ARM} runs on --role {ROLES + TEST_ROLES} only (MAIN users, PREG2 §9.1; MAIN_TEST inside the sitting), "
                f"not {role}")
    if int(args.seed) not in SEEDS:
        _refuse(f"{ARM} is registered at seeds {SEEDS} (PREG2 §9.5), not {args.seed}")
    if args.smoke:
        _refuse(f"{ARM}: --smoke is refused (PREG2 §6.3 S8: a smoke run would score MAIN users outside the plan)")
    if getattr(args, "init_from", None) is not None:
        _refuse(f"{ARM} starts from the W* best.pt (--base), not theta_pre: --init-from refused (theta_pre + FT_C is "
                "FT_PRE)")
    if args.base is None:
        _refuse(f"{ARM} needs --base <P2_W MAIN run>/ckpt/{BASE_RUN_ID.format(seed=args.seed)}/{BASE_FILE}")
    if args.base_kind is None:
        args.base_kind = D.WARM_BASE_KIND
    elif args.base_kind != D.WARM_BASE_KIND:
        _refuse(f"{ARM}: --base-kind is {D.WARM_BASE_KIND} (the warm FL base path of launch_device), not {args.base_kind}")
    want_dir = RUN_DIR_NAME[role].format(seed=int(args.seed))
    if Path(args.run_dir).name != want_dir:
        _refuse(f"{ARM} on {role}: --run-dir must be named {want_dir} (tools/p2_block_index.py --p2-dev), not "
                f"{Path(args.run_dir).name}")
    if role in TEST_ROLES and not args.base_sha256:
        _refuse(f"{ARM} on {role}: --base-sha256 is required (the TEST gate binds the sealed base before any base byte "
                "is read)")


# ------------------------------------------------------------------------------------------------ the W* base
def check_selection_binding(launch: Mapping, spec: Mapping) -> dict:
    """The MAIN P2_W run's bound P2_WARM_SELECTION_FINAL (launch_fl.bind_p2_selection record) = the spec's recipe and
    the registered file's current bytes (as reeval_p2.check_launch_selection)."""
    import v4int.launch_fl as LF
    b = launch.get("p2_selection")
    if not isinstance(b, Mapping):
        _refuse("the base run carries no bound p2_selection (a MAIN P2_W run takes its recipe from the registered "
                "P2_WARM_SELECTION_FINAL only)")
    p = Path(str(b.get("path") or ""))
    if (p.name, p.parent.name, p.parent.parent.name) != (SELECTION_FINAL.name, "fl", "contracts"):
        _refuse(f"the base run's p2_selection path {p} is not the registered contracts/fl/{SELECTION_FINAL.name}")
    if b.get("format") != LF.P2_SELECTION_FORMAT or b.get("stage") != "FINAL":
        _refuse(f"the base run's p2_selection is {b.get('format')!r} / stage {b.get('stage')!r}, not "
                f"{LF.P2_SELECTION_FORMAT} / FINAL")
    if _canon(b.get("recipe")) != _canon(spec.get("p2_recipe")) or b.get("label") != spec.get("p2_label"):
        _refuse("the base run's p2_selection recipe / label differ from its arm spec's p2_recipe / p2_label")
    reg = Path(SELECTION_FINAL)
    if not reg.is_file():
        _refuse(f"the registered selection {reg} is not on this host: the W* binding cannot be verified")
    sha = _sha(reg)
    if sha != str(b.get("sha256") or "").lower():
        _refuse(f"the registered {reg.name} (sha256 {sha}) is not the file the base run was bound to ({b.get('sha256')})")
    return {"path": str(reg), "sha256": sha, "label": b.get("label"), "self_sha256": b.get("self_sha256")}


def check_p2_wstar_base(path, *, seed: int, want_sha256: Optional[str] = None) -> tuple:
    """Torch-free: (record, arm spec) of the W* base = best.pt of the MAIN P2_W run P2_WSTAR_SMALL_s<seed>, from
    value-free records only (STATUS / ARM_SPEC / LAUNCH; RUN_SUMMARY is never opened). See the module docstring."""
    import v4int.launch_fl as LF
    p = Path(path)
    want_rid = BASE_RUN_ID.format(seed=int(seed))
    if p.name != BASE_FILE or p.parent.parent.name != "ckpt":
        _refuse(f"--base {p}: the FT_P2_WSTAR base is <P2_W run>/ckpt/{want_rid}/{BASE_FILE} only (Q5 base_best_only: "
                "never endpoint_6.0.pt, theta_pre or another checkpoint)")
    if not p.is_file():
        _refuse(f"--base {p} does not exist")
    rid, run_dir = p.parent.name, p.parent.parent.parent
    if rid != want_rid:
        _refuse(f"--base {p}: run id {rid!r} is not the MAIN W* run {want_rid} of seed {seed} (FT_P2_WSTAR - FT_PRE is "
                "paired per seed; a smoke / screen / Phase-1 run never qualifies)")
    st = _strict(run_dir / "STATUS.json")
    if st.get("status") != "DONE_MAIN" or st.get("run_id") != rid or st.get("arm") != BASE_LAUNCH_ARM:
        _refuse(f"--base {p}: STATUS {st.get('status')!r} / run id {st.get('run_id')!r} / arm {st.get('arm')!r} is not "
                f"a DONE_MAIN {BASE_LAUNCH_ARM} run {rid}")
    side = _strict(run_dir / "ARM_SPEC.json")
    spec = side.get("arm_spec")
    if side.get("arm") != BASE_LAUNCH_ARM or side.get("run_id") != rid or not isinstance(spec, dict):
        _refuse(f"--base {p}: ARM_SPEC arm {side.get('arm')!r} / run id {side.get('run_id')!r} is not the P2_W run {rid}")
    digest = LF.spec_digest(spec)
    if digest != side.get("arm_spec_sha256") or spec.get("arm") != BASE_LAUNCH_ARM:
        _refuse(f"--base {p}: the ARM_SPEC arm spec does not hash to its arm_spec_sha256 / names another arm")
    launches = sorted(run_dir.glob("LAUNCH_*.json"), key=lambda q: int(q.stem.split("_")[1]))
    if not launches:
        _refuse(f"--base {p}: {run_dir} has no LAUNCH_<k>.json")
    recs = [_strict(q) for q in launches]
    first = recs[0]
    for q, r in zip(launches, recs):
        for k in ("arm", "run_id", "arm_spec_sha256", "fl_config_digest", "label", "p2_selection"):
            if _canon(r.get(k)) != _canon(first.get(k)):
                _refuse(f"--base {p}: {q.name} {k} differs from {launches[0].name}")
    if first.get("label") != "PRODUCTION" or first.get("arm") != BASE_LAUNCH_ARM or first.get("run_id") != rid:
        _refuse(f"--base {p}: LAUNCH label {first.get('label')!r} / arm {first.get('arm')!r} / run id "
                f"{first.get('run_id')!r} is not a PRODUCTION {BASE_LAUNCH_ARM} run {rid}")
    if (first.get("fl05") or {}).get("role", "MAIN") != "MAIN":
        _refuse(f"--base {p}: a CALIBRATION screen run is never a base (LAUNCH fl05 role "
                f"{(first.get('fl05') or {}).get('role')!r})")
    if first.get("arm_spec_sha256") != digest or _canon(first.get("arm_spec")) != _canon(spec):
        _refuse(f"--base {p}: the LAUNCH arm spec differs from ARM_SPEC.json")
    try:
        LF.check_p2_spec(spec)
    except LF.LaunchRefused as e:
        _refuse(f"--base {p}: the run's arm spec is refused by launch_fl.check_p2_spec: {e}")
    if spec.get("role", "MAIN") != "MAIN" or int((spec.get("cfg") or {}).get("seed", -1)) != int(seed):
        _refuse(f"--base {p}: spec role {spec.get('role')!r} / seed {(spec.get('cfg') or {}).get('seed')!r} is not MAIN "
                f"/ {seed}")
    init = spec.get("init_from") or {}
    bad = {k: init.get(k) for k, want in (("checkpoint_sha256", PRE_CHECKPOINT_SHA256),
                                          ("state_digest", PRE_STATE_DIGEST)) if init.get(k) != want}
    if bad:
        _refuse(f"--base {p}: the run's init_from is not the registered theta_pre ({bad})")
    if (spec.get("p2_recipe") or {}).get("personal_head") and not ADMIT_PF_BASE:
        _refuse(f"--base {p}: a personal-head (WC5, method PF) W* is not admitted yet: local FT of the shared model would "
                "drop the personal component (open question in company/handoffs/P2-DEVLOCK.md; fails closed)")
    sel = check_selection_binding(first, spec)
    sha = _sha(p)
    if want_sha256 and sha != str(want_sha256).lower():
        _refuse(f"--base {p}: sha256 {sha} != --base-sha256 {want_sha256}")
    rec = {"kind": BASE_ARM, "base_kind": D.WARM_BASE_KIND, "arm": BASE_ARM, "seed": int(seed), "checkpoint": BASE_FILE,
           "path": str(p), "file_sha256": sha, "run_id": rid, "run_dir": str(run_dir), "launch_arm": BASE_LAUNCH_ARM,
           "base_arm_spec_sha256": digest, "fl_config_digest": first.get("fl_config_digest"),
           "p2_label": spec.get("p2_label"), "method": (spec.get("cfg") or {}).get("method"),
           "init_from": dict(init), "p2_selection": sel, "status": "DONE_MAIN",
           "launch_files": {q.name: _sha(q) for q in launches}, "rule": PB}
    return rec, spec


def load_p2_wstar_base(adapter, path, *, seed: int, want_sha256: Optional[str] = None) -> dict:
    """Run time: the record of check_p2_wstar_base (re-checked) + the fl weights payload loaded into `adapter`."""
    from fl.checkpoint import load_checkpoint
    from fl.numerics import state_digest
    import v4int.launch_fl as LF
    from v4int import reeval_p2 as RP
    rec, spec = check_p2_wstar_base(path, seed=seed, want_sha256=want_sha256)
    p = Path(path)
    if _sha(p) != rec["file_sha256"]:
        _refuse(f"--base {p} changed while it was verified")
    obj = load_checkpoint(p)
    if obj.get("kind") != "weights" or obj.get("run_id") != rec["run_id"]:
        _refuse(f"--base {p}: not the fl weights checkpoint of {rec['run_id']}")
    cd = LF.fl_run_config(spec, rec["run_id"], 1).digest()
    if obj.get("config_digest") != rec["fl_config_digest"] or obj.get("config_digest") != cd:
        _refuse(f"--base {p}: config_digest {str(obj.get('config_digest'))[:12]}... is not the LAUNCH / arm spec fl "
                "RunConfig digest")
    try:
        efe = float(obj.get("efe"))
    except (TypeError, ValueError):
        efe = float("nan")
    if not math.isfinite(efe) or efe < 0.5 - 1e-9:
        _refuse(f"--base {p}: best.pt is not at a scored mark >= 0.5 EFE (PREG2 §5.1: mark 0 = theta_pre is never a "
                "candidate)")
    try:
        lora = RP.check_lora_payload(obj, spec)
    except RP.P2ReevalRefused as e:
        _refuse(f"--base {p}: {e}")
    adapter.load_state_(obj["theta"])
    state = adapter.broadcast_state(clone=True)
    d = state_digest(state)
    if d != state_digest(obj["theta"]):
        _refuse(f"--base {p}: the loaded state digest {d[:12]}... != the payload theta digest")
    return dict(rec, state=state, digest=d, merged_lora=lora)


def p2_row_labels(start_rec: Mapping, selection_rec: Optional[Mapping]) -> dict:
    """FT_P2_WSTAR row fields (none a comparison-key field): study PREG2 (p2_report.check_rows_study) + the base."""
    ident = start_rec["init_from"]
    return {"study": "PREG2", "p2_arm": ARM, "p2_launch_task": TASK, "base_source": BASE_ARM,
            "base_run_id": start_rec["run_id"], "base_checkpoint": start_rec["checkpoint"],
            "base_checkpoint_sha256": start_rec["file_sha256"], "base_arm_spec_sha256": start_rec["base_arm_spec_sha256"],
            "base_training_paradigm": "FEDERATED", "base_p2_label": start_rec["p2_label"],
            "base_p2_selection_sha256": start_rec["p2_selection"]["sha256"], "base_init_run_id": ident["run_id"],
            "base_init_checkpoint_sha256": ident["checkpoint_sha256"],
            "base_init_train_manifest_id": ident["train_manifest"],
            "ft_selection_self_sha256": None if selection_rec is None else selection_rec.get("self_sha256")}


# ------------------------------------------------------------------------------------------------ TEST binding (T0a)
def p2_lock_device_arm(lock, *, seed: int, base_sha256: str) -> dict:
    """The P2 lock seals (P2_WSTAR, seed, PB) = the base (reeval_p2.p2_lock_arm: format, draft, PREG2 sha, contrasts
    self sha, not declared not run) and (FT_P2_WSTAR, seed, PB) with base_checkpoint_sha256 = the base."""
    from v4int import reeval_p2 as RP
    try:
        wstar = RP.p2_lock_arm(lock, label=BASE_ARM, seed=int(seed), rule=PB, checkpoint_sha256=str(base_sha256))
    except RP.P2ReevalRefused as e:
        _refuse(f"the W* base: {e}")
    for d in lock.get("declared_not_run") or []:
        if isinstance(d, Mapping) and d.get("arm") == ARM and d.get("seed") in (None, int(seed)):
            _refuse(f"{ARM} seed {seed} is declared not run in P2_METHODS_LOCK (S4): it is never scored on RETRO_TEST")
    hits = [e for e in lock.get("sealed_arms") or []
            if isinstance(e, Mapping) and (e.get("arm"), e.get("seed"), e.get("rule")) == (ARM, int(seed), PB)]
    if len(hits) != 1:
        _refuse(f"P2_METHODS_LOCK sealed_arms binds {len(hits)} entries for ({ARM}, seed {seed}, {PB}) (expected exactly "
                "1; S4)")
    if str(hits[0].get("base_checkpoint_sha256") or "").lower() != str(base_sha256).lower():
        _refuse(f"P2_METHODS_LOCK binds ({ARM}, seed {seed}) to base {hits[0].get('base_checkpoint_sha256')}, not "
                f"{base_sha256}")
    return {"wstar": wstar, "device": {k: hits[0].get(k) for k in ("arm", "seed", "rule", "base_checkpoint_sha256")}}


def p2_device_test_gate(args) -> dict:
    """T0a before any base byte is read (reeval_p2.p2_test_gate pattern): DATA-09 gate, P2 lock + freeze, the two
    LOCKED_EVALUATION / RETRO_TEST register entries, the sealed base."""
    from v4data09 import d9_gate as G
    from v4int import test_roles as TR
    if not TR.test_gate_ok(args.role):
        _refuse(f"TEST role {args.role} refused: the DATA-09 TEST gate is closed (v4int.test_roles.test_gate_ok)")
    try:
        g = G.check_test_gate(AML_PATH, REGISTER_PATH)
    except G.TestGateRefused as e:
        _refuse(f"TEST role {args.role} refused: the DATA-09 TEST gate is closed ({e})")
    lock_p, frz_p = Path(P2_METHODS_LOCK), Path(P2_SCRIPT_FREEZE)
    for p, what in ((lock_p, "P2_METHODS_LOCK"), (frz_p, "P2_SCRIPT_FREEZE")):
        if not p.is_file():
            _refuse(f"TEST role {args.role} refused: {what} ({p}) is not registered on this host - PREG2 §6.3 S2: the P2 "
                    "lock and freeze precede the first TEST scoring of any arm")
    lock_sha, frz_sha = _sha(lock_p), _sha(frz_p)
    reg = G._strict_json(Path(g.register_path))
    errs = G._register_tool().verify(reg)
    if errs:
        _refuse(f"TEST_EXPOSURE_REGISTER hash chain does not verify: {errs}")
    ents = [e for e in reg.get("entries") or [] if isinstance(e, Mapping)
            and e.get("access_purpose") == LOCKED_EVALUATION and e.get("split") == "RETRO_TEST"]
    p1 = [e for e in ents if e.get("all_methods_lock_sha256") == g.lock_sha256]
    p2 = [e for e in ents if e.get("all_methods_lock_sha256") == lock_sha]
    if not p1:
        _refuse("TEST refused: no LOCKED_EVALUATION / RETRO_TEST register entry for the Phase-1 ALL_METHODS_LOCK "
                f"{g.lock_sha256} (PREG2 §6.3 S2)")
    if not p2:
        _refuse(f"TEST refused: no LOCKED_EVALUATION / RETRO_TEST register entry whose all_methods_lock_sha256 binds "
                f"P2_METHODS_LOCK {lock_sha} (PREG2 §6.3 S2)")
    if any("P2_METHODS_LOCK" not in str(e.get("observations_description") or "") for e in p2):
        _refuse("TEST refused: the Phase-2 LOCKED_EVALUATION entry does not name P2_METHODS_LOCK in its "
                "observations_description (PREG2 §13 item 6)")
    lock = _strict(lock_p)
    arm = p2_lock_device_arm(lock, seed=int(args.seed), base_sha256=str(args.base_sha256))
    return {"data09_gate": g.binding(), "p2_methods_lock": {"path": str(lock_p), "sha256": lock_sha},
            "p2_script_freeze": {"path": str(frz_p), "sha256": frz_sha},
            "locked_evaluation_entry_ids": {"phase1": int(min(p1, key=lambda e: e["entry_id"])["entry_id"]),
                                            "p2": int(min(p2, key=lambda e: e["entry_id"])["entry_id"])},
            "arm_binding": arm, "register_file_sha256": g.register_file_sha256}


# ------------------------------------------------------------------------------------------------ the run
def parse_args(argv=None):
    with p2_device_context():
        return D.parse_args(argv)


def preflight(args, **kw) -> dict:
    """launch_device's preflight (unchanged, inside the context) + the P2 admissions, the TEST gate (MAIN_TEST, before
    any base byte is read) and the torch-free base binding. The record goes into LAUNCH.preflight.p2_device."""
    ld = check_launch_device_unchanged()
    check_args(args)
    with p2_device_context():
        pre = D.preflight(args, **kw)
    test_lock = p2_device_test_gate(args) if args.role in TEST_ROLES else None
    base, _spec = check_p2_wstar_base(args.base, seed=int(args.seed), want_sha256=args.base_sha256)
    pre["p2_device"] = {"task": TASK, "authority": AUTHORITY, "shim_sha256": _sha(Path(__file__)),
                        "launch_device_sha256": ld, "arm": ARM, "role": args.role, "seed": int(args.seed),
                        "base": base, "test_lock": test_lock}
    return pre


def main(argv=None) -> int:
    from v4int.launch_central import export_environment
    try:
        args = parse_args(argv)
        pre = preflight(args)
        if args.dry_run:
            print(json.dumps({"preflight": pre, "cells": D.grid_cells(ARM), "knobs": D.KNOBS}, indent=1, default=str))
            return 0
        env_rec = export_environment(int(args.gpu))
        with p2_device_context():
            return D.run(args, pre, env_rec)
    except D.LaunchRefused as e:
        print(f"LAUNCH REFUSED: {e}", file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
