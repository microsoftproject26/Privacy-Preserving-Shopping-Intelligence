"""v4int.reeval (INT-06) - evaluate ONE locked checkpoint on ONE registered role through the frozen T3 path on the
registered A40 device class (PREG1_AMENDMENT_06D §1). Evaluation only: no training step, no optimizer state is used.

  python -B code/v4int/reeval.py --ckpt <file> --ckpt-kind central|fl|pf [--pf-store <PF_STORE[_best].pt>]
      --expect-sha256 <hex> [--expect-pf-store-sha256 <hex>] --variant SASREC_D256_B3|SASREC_D64_B2
      --recipe-id 02B_SASREC|SMALL_SASREC_D64_B2 --seed S
      --role MAIN|UNSEEN_HOLDOUT|SPARSE_PANEL|HEAVY_PANEL|TRUE_COLD|UNSEEN_VAL
             |MAIN_TEST|UNSEEN_HOLDOUT_TEST|SPARSE_PANEL_TEST|HEAVY_PANEL_TEST|TRUE_COLD_TEST (INT-07; gated)
      --arm-label <C_BIG|C_SMALL|FA|DP8|...>
      --selection-rule PRACTICAL_BEST|CONTROLLED_BUDGET --checkpoint-id <id> --gpu g --out-dir <new dir>
      --history-policy USER_HISTORY_L50 [--allow-shared-gpu] [--registry-ext <REGISTRY_V2_*_EXT.json>] [--dry-run]

Sources (every identity check runs before a weight is used; --dry-run stops after them, no GPU):
  central  v4train checkpoint {"meta", "model"} under <run>/[sealed/]checkpoints/. <run>/RUN_START.json gives the
           config (variant / recipe / seed), schema, theta0 and the source harness's key base (the exact tr.key_base
           the source rows were built from). meta variant / seed / config_sha256 (/ init_sha256, state_digest when
           recorded) must match; the rebuilt theta0 must equal RUN_START's.
  fl / pf  fl.checkpoint "weights" payload at <run>/ckpt/<run_id>/<file>. Every <run>/LAUNCH_<k>.json must agree (arm,
           run id, arm spec, config digest); the payload's run id / config digest must equal them; a PRODUCTION MAIN FL
           run of SASREC_D64_B2 / SMALL_SASREC_D64_B2 only. The harness CentralTrainer is rebuilt EXACTLY as
           launch_fl.run builds it and must reproduce the LAUNCH record's theta0, init sha256 and comparison key; theta
           is loaded into the harness adapter (the FL run's server adapter). An FL PF run needs --ckpt-kind pf.
  pf       + --pf-store (launch_fl PF_STORE.pt = endpoint store, PF_STORE_best.pt = store of the PRACTICAL_BEST mark,
           REV-FL-01 B2): same run id / config digest, and its recorded theta_digest must equal the loaded weights'
           digest (a pre-B2 store without theta_digest is accepted with the endpoint checkpoint only). Scored with
           launch_device.pf_evaluator (shared + p_u; p = 0 for a user the store never saw).
Evaluation: launch_fl.role_evaluator (the frozen T3 path: predict_adapter -> evaluate -> rows_from_result + harness
labels) or launch_device.pf_evaluator. A role other than MAIN replaces population_hash, candidate_order_hash and the
train / eval manifest ids exactly as launch_device.run does (an eval-only role such as TRUE_COLD keeps the harness
pair's train manifest). The device must be the registered pinned class (v4eval.device_pin; NVIDIA_A40-FP32-DET):
rows are MEASURED_PINNED_DEVICE with device_pin_sha256 (checked before anything is written). TEST roles / manifests
are refused unless test_gate_ok(role) (the DATA-09 hook v4int.test_roles.test_gate_ok: a REGISTERED TEST role and an
open TEST gate; checked before any file is read). --registry-ext: an additions-only extension of the LOCK_V2
REGISTRY_V2 (every base entry byte-identical, `extends` = the base file / self hash) whose roles are verified through
the same wrapper pattern as v4int.c_full; only TEST-named roles may come from it, so they stay behind the gate.
TEST roles (INT-07; PREG1_AMENDMENT_06D §2): --registry-ext must be the DATA-09 REGISTRY_V2_TEST_EXT.json. The DATA-09
loader wrappers are installed (v4int.test_roles.install(role=None): the gate re-checked on every TEST resolution), the
role is resolved under the gate (test_roles.resolve_role), the RETRO_TEST StoreEvalProvider runs on the gated store /
window path (v4int.retro_store: USER_HISTORY_L50 over TRAIN + VALIDATION_SELECT + VALIDATION_CONFIRM + TEST), and the
key is v4int.retro_store.retro_role_key: population_hash, candidate_order_hash, train / eval manifest ids, split
(= RETRO_TEST) and manifest_registry_self_sha256 from the gated comparison_key_base of (support manifest verified
through the extension WITHOUT building its store, RETRO_TEST provider); the rows carry split RETRO_TEST, the TEST role,
the register entry id and the eval manifest content sha256. The checkpoint, harness, T3 path and pin rules are the
VALIDATION roles' unchanged.
Registered central sources (GATE-B B1; review 06 [5-R1]): besides the MAIN-pair central runs (C_BIG / C_SMALL / ...,
unchanged), exactly two central sources are accepted, each bound to its registry extension under contracts/
manifest_registry (never a user path):
  C_FULL  run id C_FULL_*, run_label PRODUCTION, SASREC_D256_B3 / 02B_SASREC, seed 2026, trained on
          FULL_TRAIN_5_100_TIEBREAK -> MAIN_VALIDATION_5_100_131072 = REGISTRY_V2_FULL_EXT.json role C_FULL;
          --arm-label C_FULL; best.pt with --selection-rule PRACTICAL_BEST or endpoint_6.0.pt with CONTROLLED_BUDGET only
          (REV-DELTA-03 m1: no endpoint_8.0 / intermediate file; the file's v4train kind must match its name).
          ANA-05: seed 2027 is admitted as a DESCRIPTIVE source seed (06D §8 NICE seed; 06E §2 / DL 176: the second seed
          of a single-seed contrast is DESCRIPTIVE; K15 / K17-K19 / K20b / K20c stay seed 2026): the same checks as seed
          2026 (source classification, FIX-REEVAL binding, checkpoint file / kind / mark), and its rows carry
          comparison_status DESCRIPTIVE + source_seed_status DESCRIPTIVE + source_seed_rule (REEVAL_RECORD
          central_source.seed_status DESCRIPTIVE); a seed-2026 record / row is byte-for-byte the pre-ANA-05 one.
  PRE     run id PRETRAIN_FOR_FL_*, run_label PRETRAIN_FOR_FL, SASREC_D64_B2 / SMALL_SASREC_D64_B2, seed 2026, trained on
          PRETRAIN_TRAIN_5_100_NONMAIN -> UNEW_VAL_VALIDATION_32768 = REGISTRY_V2_PRETRAIN_EXT.json role PRETRAIN;
          --arm-label PRE; the endpoint_6.0.pt checkpoint only (theta_pre, 06K §1: no selection), kind endpoint_6.0 and
          meta mark 6.0, and it must be the run's PRETRAIN_FOR_FL_INIT.json fl_init (path + sha256; REV-DELTA-03 m2).
A source is CLAIMED by any of: its train manifest (RUN_START providers or key base), its run-id prefix, its run label or
its arm label; a claimed source must then match EVERY field above, and the extension must verify additions-only
against REGISTRY_V2 (verify_generic_ext; `extends` = REGISTRY_V2's path / file / self hash), register the source's pair
under its role, and the run must be bound to it (check_source_binding, FIX-REEVAL): RUN_START config_sha256 =
RunConfig.sha256 of its config (the hash the checkpoint meta carries), config.key.manifest_registry_self_sha256 = the
extension's self hash (the comparison key the launcher's providers built THROUGH the extension), and key_base /
RUN_START.manifest_registry = the trainer view the c_full / pretrain_small shim derives from that extension
(v4data08 / v4data10 trainer_view, recomputed here: self + file sha256), every other config.key field unchanged in the
key base. (The trainer's key base always names its own registry = the per-run trainer VIEW, never the extension itself.)
Anything else claimed refuses (an unregistered source). Roles: MAIN, UNSEEN_HOLDOUT, SPARSE_PANEL, HEAVY_PANEL, UNSEEN_VAL and their TEST
roles; the eval-only TRUE_COLD / TRUE_COLD_TEST are refused for these sources (the key train manifest would be the
source's own, not a registered key-only reference). Keys:
  C_FULL on MAIN (its own eval manifest): the source key base unchanged (train FULL_TRAIN_5_100_TIEBREAK, the TIEBREAK
          population_hash) - the key of C_FULL's own sealed rows; the MAIN eval provider is verified through the
          FULL extension (wrapper as v4int.c_full) and its fingerprint must equal the source run's. On MAIN_TEST the
          gated key keeps the same train side (train_manifest_id / population_hash of the source; A-F7 of
          tools/retention_report.py). Every other role: the role key exactly as for any central source.
  PRE on MAIN / panels: the role key (MAIN pair for MAIN: train MAIN_TRAIN_5_100_131072), the train side verified
          key-only (v4int.retro_store.train_key_ref; no train store is built).
Rows of both sources carry central_source, source_train_manifest_id, source_eval_manifest_id, source_manifest_role and
source_registry_self_sha256 (none of them a comparison-key field); REEVAL_RECORD carries the verified source record.
--dry-run for a registered source also verifies the extension against contracts/manifest_registry/REGISTRY_V2.json and
the checkpoint's meta (variant / seed / config / init sha256; loaded on the CPU) - no GPU, no data root, nothing written.
Outputs in the NEW --out-dir: RESULT_ROWS.json (rows relabelled: method = arm label, selection_rule, checkpoint_id,
checkpoint_sha256, reeval = true, row_status / comparison_status REEVAL, efe of the checkpoint; the comparison key
untouched), evals/<role>.result.json, evals/<role>.ranks_int32.npy, REEVAL_RECORD.json (every input hash, device
class, pin sha, comparison key). No metric value is printed. Exit: 0 written (or dry run passed), 1 refused / error.
No-FT references on UNSEEN_VAL (PREG1_AMENDMENT_06L_C §C16; DEV-FTPRE): the UNSEEN_VAL role (the device selection
panel UNEW_VAL_SUPPORT_32768 -> UNEW_VAL_VALIDATION_32768) admits exactly two evaluation-only references - C_SMALL (a
MAIN-pair central SASREC_D64_B2 / SMALL checkpoint, --ckpt-kind central) and FA (a cold-start FA FL checkpoint: LAUNCH
arm FA, no warm-start init_from, --ckpt-kind fl). Every other arm label / kind on UNSEEN_VAL refuses before any file is
read (except the 06O preview references below). Their rows keep the role key (launch_device rule) and are labelled
DESCRIPTIVE (comparison_status DESCRIPTIVE, no_ft_reference true, no_ft_reference_role UNSEEN_VAL; never a headline,
never a selection input); REEVAL_RECORD carries `no_ft_reference`. Server directory form (tools/ana_block_index.py):
<R>/REEVAL_A40/<C_SMALL|FA>_UNSEEN_VAL_s<S>_<PB|CB>.
Preview references on UNSEEN_VAL (PREG1_AMENDMENT_06O O1-O2; FIX-REEVAL-06O): the same role also admits the registered
central sources C_FULL (seeds 2026 / 2027; best.pt = PRACTICAL_BEST, endpoint_6.0.pt = CONTROLLED_BUDGET) and PRE
(theta_pre = endpoint_6.0.pt), --ckpt-kind central with the source's own variant, through EVERY source check above
(classification, check_source_binding, checkpoint sha / file / kind / mark / meta, extension pair, roles, PRE init
record) and the role key (train side key-only, as on the panels). Their rows carry comparison_status DESCRIPTIVE +
preview_reference NON_MAIN_UNSEEN_VAL + preview_reference_rule (none a comparison-key field; not the no-FT labels);
REEVAL_RECORD and the dry run carry preview_reference. They never enter a contrast, a selection, a stopping rule or a
claim (tools/ana_block_index.py lists them as excluded, tools/retention_report.py refuses them). Directory form:
<R>/REEVAL_A40/<C_FULL|PRE>_UNSEEN_VAL_s<S>_<PB|CB>. The no-FT references and every other role are unchanged.
Warm FL runs (ANA-06; FL-16 warm method set, PREG1_AMENDMENT_06N): a run whose LAUNCH arm is a warm launch_fl label
(FA_WARM / FA_WARM_FROZEN and the 12 FL-16 labels) or whose arm_spec carries init_from is re-evaluated ONLY under its
block arm label (FA_WARM, FA_WARM_FROZEN, FP_WARM, PF_WARM, FA_Q8_WARM, FA_1024_WARM, DP8_WARM / DP1_WARM by the LAUNCH
dp_z epsilon, the *_WARM_FROZEN labels, DP8_FROZEN = DP_WARM_FROZEN eps 8, DP1_WARM_FROZEN), and such a label only on
that run; the arm_spec must carry theta_pre (init_from) and frozen item tables exactly for the S8 variant; S_CAL_WARM*
(no weights) refuse. --ckpt-kind pf follows the cold arm (PF_WARM* are pf). A warm-line run's dry run / REEVAL_RECORD
carry `warm_line` (launch arm, block arm, cold arm, variant, stage, frozen, epsilon); cold runs and FA_WARM(_FROZEN) are
unchanged byte for byte.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Callable, Mapping, Optional

CODE = Path(__file__).resolve().parents[1]
V4 = CODE.parent
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))
sys.dont_write_bytecode = True

RECORD_FORMAT = "PPSI_V4_REEVAL_RECORD_1"
PINNED_CLASS = "NVIDIA_A40-FP32-DET"
FAMILY = "SASREC"
VARIANT_RECIPE = {"SASREC_D256_B3": "02B_SASREC", "SASREC_D64_B2": "SMALL_SASREC_D64_B2"}
FL_VARIANT = "SASREC_D64_B2"
ROLES = ("MAIN", "UNSEEN_HOLDOUT", "SPARSE_PANEL", "HEAVY_PANEL", "TRUE_COLD", "UNSEEN_VAL")
TEST_ROLES = ("MAIN_TEST", "UNSEEN_HOLDOUT_TEST", "SPARSE_PANEL_TEST", "HEAVY_PANEL_TEST", "TRUE_COLD_TEST")  # DATA-09
ROLE_KEY_FIELDS = ("population_hash", "candidate_order_hash", "train_manifest_id", "eval_manifest_id")
IDENTITY_FIELDS = ("data_version", "manifest_role", "manifest_registry_sha256", "manifest_registry_self_sha256",
                   "train_manifest_id", "eval_manifest_id", "schema_id", "schema_self_sha256", "run_label",
                   "model_variant", "theta0_init_sha256", "run_role", "recipe_id", "recipe_sha256",
                   "determinism_mode", "cublas_workspace_config")
ARM_LABEL_RE = re.compile(r"^[A-Z][A-Z0-9_]{0,31}$")
SMOKE = "SMOKE_NOT_EVIDENCE"
ENDPOINT_FILE = "endpoint_6.0.pt"
BEST_FILE = "best.pt"
CKPT_FILE_KIND = {BEST_FILE: "best", ENDPOINT_FILE: "endpoint_6.0"}   # v4train.checkpoint kinds (the file's own `kind`)
ENDPOINT_MARK = 6.0
PRE_INIT_FILE = "PRETRAIN_FOR_FL_INIT.json"                           # v4int.pretrain_small.FL_INIT_FILE (DATA-10)
EXT_DIR = V4 / "contracts" / "manifest_registry"
BASE_REGISTRY_REL = "contracts/manifest_registry/REGISTRY_V2.json"
CENTRAL_SOURCES = {           # GATE-B B1 (review 06 [5-R1]): the registered non-MAIN central sources
    "C_FULL": {"arm_label": "C_FULL", "ext": "REGISTRY_V2_FULL_EXT.json", "ext_role": "C_FULL",
               "train": "FULL_TRAIN_5_100_TIEBREAK", "eval": "MAIN_VALIDATION_5_100_131072",
               "run_id_prefix": "C_FULL_", "run_label": "PRODUCTION", "variant": "SASREC_D256_B3",
               "recipe_id": "02B_SASREC", "seed": 2026, "descriptive_seeds": (2027,),     # ANA-05: s2027 DESCRIPTIVE
               "checkpoint_files": (BEST_FILE, ENDPOINT_FILE),
               "rule_files": {"PRACTICAL_BEST": BEST_FILE, "CONTROLLED_BUDGET": ENDPOINT_FILE},   # REV-DELTA-03 m1
               "own_eval_roles": ("MAIN", "MAIN_TEST"),
               "view_module": "v4data08.d8_registry"},        # the c_full shim's trainer view (FIX-REEVAL)
    "PRE": {"arm_label": "PRE", "ext": "REGISTRY_V2_PRETRAIN_EXT.json", "ext_role": "PRETRAIN",
            "train": "PRETRAIN_TRAIN_5_100_NONMAIN", "eval": "UNEW_VAL_VALIDATION_32768",
            "run_id_prefix": "PRETRAIN_FOR_FL_", "run_label": "PRETRAIN_FOR_FL", "variant": "SASREC_D64_B2",
            "recipe_id": "SMALL_SASREC_D64_B2", "seed": 2026, "descriptive_seeds": (),   # ONE theta_pre (06K §1)
            "checkpoint_files": (ENDPOINT_FILE,),
            "rule_files": None,                     # theta_pre = endpoint_6.0 under either rule label (06K §1)
            "own_eval_roles": (),
            "view_module": "v4data10.d10_registry"},   # the pretrain_small shim's trainer view (FIX-REEVAL)
}
SOURCE_ROLES = ("MAIN", "UNSEEN_HOLDOUT", "SPARSE_PANEL", "HEAVY_PANEL", "UNSEEN_VAL",
                "MAIN_TEST", "UNSEEN_HOLDOUT_TEST", "SPARSE_PANEL_TEST", "HEAVY_PANEL_TEST")
SOURCE_SEED_STATUS = "DESCRIPTIVE"            # ANA-05: the label of a source's descriptive seed (C_FULL s2027)
SOURCE_SEED_RULE = ("06D §8 NICE seed; 06E §2 / DL 176: a run of the second seed of a single-seed contrast is "
                    "DESCRIPTIVE (K15, K17-K19, K20b, K20c stay seed 2026 confirmatory); never a confirmatory denominator")
NO_FT_ROLE = "UNSEEN_VAL"                     # 06L-C §C16: the evaluation-only no-FT references on the selection panel
NO_FT_REFERENCES = {"C_SMALL": {"kind": "central", "variant": "SASREC_D64_B2"},
                    "FA": {"kind": "fl", "variant": "SASREC_D64_B2", "fl_arm": "FA"}}
NO_FT_STATUS = "DESCRIPTIVE"
NO_FT_RULE = ("06L-C §C16: an evaluation-only 'no-FT' reference of C_SMALL / FA on UNSEEN_VAL (the device selection "
              "panel); DESCRIPTIVE, never a headline and never a selection input")
PREVIEW_ROLE = NO_FT_ROLE                     # 06O: the non-MAIN new-user panel (UNEW_VAL_SUPPORT -> UNEW_VAL_VALIDATION)
PREVIEW_REFERENCE = "NON_MAIN_UNSEEN_VAL"     # 06O O1: the label of a C_FULL / PRE row on it
PREVIEW_SOURCES = ("C_FULL", "PRE")           # the registered central sources (CENTRAL_SOURCES) admitted there (06O O1)
PREVIEW_STATUS = "DESCRIPTIVE"
PREVIEW_RULE = ("PREG1_AMENDMENT_06O O1-O2: a registered central source (C_FULL s2026 / s2027, PRE theta_pre) evaluated "
                "on the non-MAIN new-user panel UNSEEN_VAL (12,792 eval users, rankable 10,237); DESCRIPTIVE, reported "
                "only in the table 'central models on never-trained new users'; never a contrast, a selection, a "
                "stopping rule or a claim")


class ReevalRefused(RuntimeError):
    pass


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _canon(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def _strict(p: Path) -> dict:
    return json.loads(Path(p).read_text(encoding="utf-8"),
                      parse_constant=lambda c: (_ for _ in ()).throw(ValueError(f"non-strict constant {c}")))


def test_gate_ok(role: str) -> bool:
    """The DATA-09 hook (INT-07 wiring): True for every non-TEST role; for a TEST role True only if it is a registered
    TEST role and the TEST gate is open (ALL_METHODS_LOCK + its MANIFEST_BUILD register entry). Never raises."""
    if "TEST" not in str(role).upper():
        return True
    from v4int import test_roles as TR
    return bool(TR.test_gate_ok(str(role)))


def is_test_role(role: str) -> bool:
    return "TEST" in str(role).upper()


# ------------------------------------------------------------------------------------------------ arguments
def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="PPSI V4 checkpoint re-evaluation on the registered A40 class (INT-06)")
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--ckpt-kind", required=True, choices=("central", "fl", "pf"))
    ap.add_argument("--pf-store", type=Path)
    ap.add_argument("--expect-pf-store-sha256")
    ap.add_argument("--expect-sha256", required=True)
    ap.add_argument("--variant", required=True, choices=tuple(VARIANT_RECIPE))
    ap.add_argument("--recipe-id", required=True, choices=tuple(VARIANT_RECIPE.values()))
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--role", required=True)
    ap.add_argument("--arm-label", required=True)
    ap.add_argument("--selection-rule", required=True, choices=("PRACTICAL_BEST", "CONTROLLED_BUDGET"))
    ap.add_argument("--checkpoint-id", required=True)
    ap.add_argument("--gpu", type=int, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--history-policy", required=True, choices=("USER_HISTORY_L50",))
    ap.add_argument("--allow-shared-gpu", action="store_true")
    ap.add_argument("--registry-ext", type=Path)
    ap.add_argument("--dry-run", action="store_true")
    return ap.parse_args(argv)


def check_args(args) -> None:
    """Every refusal that needs no file: TEST gate first (before any read), then the argument contract."""
    if not test_gate_ok(args.role):
        raise ReevalRefused("TEST role refused: not a registered TEST role or the DATA-09 TEST gate is closed "
                            "(test_gate_ok)")
    if is_test_role(args.role):
        if args.role not in TEST_ROLES:
            raise ReevalRefused(f"TEST role {args.role!r} is not a registered TEST role {TEST_ROLES}")
        if args.registry_ext is None:
            raise ReevalRefused("a TEST role needs --registry-ext <REGISTRY_V2_TEST_EXT.json> (DATA-09)")
    elif args.role not in ROLES:
        raise ReevalRefused(f"role {args.role!r} is not a registered re-evaluation role {ROLES}")
    if VARIANT_RECIPE[args.variant] != args.recipe_id:
        raise ReevalRefused("variant / recipe pair differs from the registered pair")
    if not ARM_LABEL_RE.match(args.arm_label or ""):
        raise ReevalRefused(f"--arm-label {args.arm_label!r} is not an arm label ([A-Z][A-Z0-9_]*)")
    if args.ckpt_kind == "pf" and args.pf_store is None:
        raise ReevalRefused("--ckpt-kind pf needs --pf-store (PF_STORE.pt or PF_STORE_best.pt of the same run)")
    if args.ckpt_kind != "pf" and (args.pf_store is not None or args.expect_pf_store_sha256):
        raise ReevalRefused("--pf-store is valid with --ckpt-kind pf only")
    if args.ckpt_kind in ("fl", "pf") and args.variant != FL_VARIANT:
        raise ReevalRefused(f"FL checkpoints are {FL_VARIANT} (the SMALL recipe) only")
    if args.ckpt_kind != "central" and args.arm_label in {d["arm_label"] for d in CENTRAL_SOURCES.values()}:
        raise ReevalRefused(f"--arm-label {args.arm_label} is a registered CENTRAL source (GATE-B B1)")
    if not re.fullmatch(r"[0-9a-fA-F]{64}", args.expect_sha256 or ""):
        raise ReevalRefused("--expect-sha256 must be a sha256 hex digest")
    if int(args.gpu) < 0:
        raise ReevalRefused("an explicit physical GPU id (--gpu >= 0) is required")
    check_no_ft_args(args)


def check_no_ft_args(args) -> None:
    """06L-C §C16: on UNSEEN_VAL only the no-FT references C_SMALL (central, SMALL variant) and FA (FL checkpoint)."""
    if args.role != NO_FT_ROLE:
        return
    if is_preview_reference(args.role, args.arm_label) and args.ckpt_kind == "central" and \
            args.variant == CENTRAL_SOURCES[args.arm_label]["variant"]:
        return                             # 06O: a C_FULL / PRE preview; every source check follows (classify, binding)
    ref = NO_FT_REFERENCES.get(args.arm_label)
    if ref is None or args.ckpt_kind != ref["kind"] or args.variant != ref["variant"]:
        allowed = ", ".join(f"{a} (--ckpt-kind {r['kind']}, {r['variant']})" for a, r in NO_FT_REFERENCES.items())
        raise ReevalRefused(f"role {NO_FT_ROLE} is the device selection panel: only the no-FT references {allowed} are "
                            f"evaluated there, not {args.arm_label} / --ckpt-kind {args.ckpt_kind} / {args.variant} "
                            "(06L-C §C16)")


def check_no_ft_source(args, src: Mapping) -> None:
    """06L-C §C16: the FA no-FT reference is a cold-start FA run (LAUNCH arm FA, no warm-start init_from); the
    C_SMALL reference is a MAIN-pair central run (never a registered PRE / C_FULL source)."""
    if args.role != NO_FT_ROLE:
        return
    if is_preview_reference(args.role, args.arm_label):              # 06O: the run must BE the registered source
        if src.get("kind") != "central" or src.get("central_source") != args.arm_label:
            raise ReevalRefused(f"the {args.arm_label} preview reference on {PREVIEW_ROLE} must be the registered "
                                f"{args.arm_label} central source; the checkpoint's run is "
                                f"{src.get('central_source') or 'not a registered source'} (06O O1)")
        return
    ref = NO_FT_REFERENCES.get(args.arm_label)
    if ref is None:                                                  # (check_args refuses first; re-asserted here)
        raise ReevalRefused(f"role {NO_FT_ROLE} admits only the no-FT references {sorted(NO_FT_REFERENCES)} "
                            "(06L-C §C16)")
    if ref["kind"] == "fl":
        la = src.get("launch") or {}
        if src.get("arm") != ref["fl_arm"] or (la.get("arm_spec") or {}).get("init_from") is not None:
            raise ReevalRefused(f"the {args.arm_label} no-FT reference must be a cold-start {ref['fl_arm']} run; the "
                                f"checkpoint's run is arm {src.get('arm')!r}"
                                + (" warm-started from theta_pre" if (la.get("arm_spec") or {}).get("init_from")
                                   else "") + " (06L-C §C16)")
    elif src.get("central_source") is not None:
        raise ReevalRefused(f"the {args.arm_label} no-FT reference is a MAIN-pair central run, not the registered "
                            f"{src.get('central_source')} source (06L-C §C16)")


def no_ft_labels(role: str, arm_label: str) -> dict:
    """The DESCRIPTIVE labels of a no-FT reference row (none of them a comparison-key field); {} on any other role."""
    if role != NO_FT_ROLE:
        return {}
    return {"comparison_status": NO_FT_STATUS, "no_ft_reference": True, "no_ft_reference_role": NO_FT_ROLE,
            "no_ft_reference_arm": arm_label, "no_ft_reference_rule": NO_FT_RULE}


def is_preview_reference(role: str, arm_label: str) -> bool:
    """06O: a C_FULL / PRE re-evaluation on UNSEEN_VAL (classify_central_source then requires the run to BE that
    registered source, with every source check)."""
    return role == PREVIEW_ROLE and arm_label in PREVIEW_SOURCES


def preview_labels(role: str, arm_label: str) -> dict:
    """06O O1: the DESCRIPTIVE labels of a preview row (none of them a comparison-key field); {} otherwise."""
    if not is_preview_reference(role, arm_label):
        return {}
    return {"comparison_status": PREVIEW_STATUS, "preview_reference": PREVIEW_REFERENCE,
            "preview_reference_rule": PREVIEW_RULE}


# ------------------------------------------------------------------------------------------------ sources
def source_central(ckpt: Path, args) -> dict:
    """<run>/[sealed/]checkpoints/<file> -> the source run's RUN_START (config, key base, providers)."""
    ckpt = Path(ckpt)
    if ckpt.parent.name != "checkpoints":
        raise ReevalRefused(f"{ckpt} is not under a v4train checkpoints/ directory")
    run_dir = ckpt.parent.parent.parent if ckpt.parent.parent.name == "sealed" else ckpt.parent.parent
    p = run_dir / "RUN_START.json"
    if not p.is_file():
        raise ReevalRefused(f"checkpoint run has no RUN_START: {p}")
    start = _strict(p)
    cfg = start.get("config") or {}
    for key, want in (("family", FAMILY), ("model_variant", args.variant), ("recipe_id", args.recipe_id),
                      ("seed", args.seed)):
        if cfg.get(key) != want:
            raise ReevalRefused(f"checkpoint run {key} {cfg.get(key)!r} differs from requested {want!r}")
    kb = start.get("key_base") or {}
    if kb.get("run_label") == SMOKE or str(start.get("run_id", "")).startswith(SMOKE):
        raise ReevalRefused("a smoke run's checkpoint is not evidence")
    if kb.get("split") != "VALIDATION":
        raise ReevalRefused("the source key base is not a VALIDATION key")
    csrc = classify_central_source(start, args, ckpt)
    return {"kind": "central", "run_dir": run_dir, "start": start, "start_path": p, "start_sha256": _sha(p),
            "run_id": start.get("run_id"), "central_source": csrc}


def central_train_binding(start: Mapping) -> Optional[dict]:
    """The source central run's train manifest binding (FIX-REEVAL). v4train.trainer's RUN_START train_provider carries
    only {manifest_id, fingerprint} (no manifest_binding); the binding is in RUN_START.manifest_registry.bindings.train
    (the trainer's registry record; content_sha256 = the provider's manifest_binding content_sha256). None when neither
    is recorded or the recorded binding names another manifest than train_provider."""
    tp = start.get("train_provider") or {}
    if isinstance(tp.get("manifest_binding"), Mapping):
        return dict(tp["manifest_binding"])
    b = ((start.get("manifest_registry") or {}).get("bindings") or {}).get("train")
    if isinstance(b, Mapping) and b.get("manifest_id") == tp.get("manifest_id"):
        return dict(b)
    return None


def classify_central_source(start: Mapping, args, ckpt: Path) -> Optional[str]:
    """GATE-B B1: None for a MAIN-pair central run (the unchanged path), else the registered source name (C_FULL /
    PRE) whose every identity field the run matches. A run CLAIMED by a source (train manifest, run-id prefix, run
    label or arm label) that fails any field refuses: an unregistered source is never evaluated."""
    kb = start.get("key_base") or {}
    tr = (start.get("train_provider") or {}).get("manifest_id")
    ev = (start.get("eval_provider") or {}).get("manifest_id")
    rid = str(start.get("run_id", ""))
    claims = sorted(n for n, d in CENTRAL_SOURCES.items()
                    if d["train"] in (tr, kb.get("train_manifest_id")) or rid.startswith(d["run_id_prefix"])
                    or (d["run_label"] != "PRODUCTION" and kb.get("run_label") == d["run_label"])
                    or args.arm_label == d["arm_label"])
    if not claims:
        return None
    if len(claims) > 1:
        raise ReevalRefused(f"the checkpoint run is claimed by more than one registered central source {claims}")
    name = claims[0]
    d = CENTRAL_SOURCES[name]
    want = {"run id prefix": (rid[:len(d["run_id_prefix"])], d["run_id_prefix"]),
            "key_base.run_label": (kb.get("run_label"), d["run_label"]),
            "train_provider.manifest_id": (tr, d["train"]), "eval_provider.manifest_id": (ev, d["eval"]),
            "key_base.train_manifest_id": (kb.get("train_manifest_id"), d["train"]),
            "key_base.eval_manifest_id": (kb.get("eval_manifest_id"), d["eval"]),
            "--variant": (args.variant, d["variant"]), "--recipe-id": (args.recipe_id, d["recipe_id"]),
            "--arm-label": (args.arm_label, d["arm_label"]), "--ckpt-kind": (args.ckpt_kind, "central")}
    bad = {k: got for k, (got, exp) in want.items() if got != exp}
    if int(args.seed) not in source_seeds(name):                     # ANA-05: the registered (+ descriptive) seeds
        bad["--seed"] = int(args.seed)
    if d["checkpoint_files"] is not None and Path(ckpt).name not in d["checkpoint_files"]:
        bad["checkpoint file"] = Path(ckpt).name
    elif d.get("rule_files") and Path(ckpt).name != d["rule_files"].get(args.selection_rule):
        bad["checkpoint file for --selection-rule"] = (Path(ckpt).name, args.selection_rule)   # PB = best, CB = 6.0
    if bad:
        raise ReevalRefused(f"not the registered {name} central source (GATE-B B1): {bad}")
    if args.role not in SOURCE_ROLES:
        raise ReevalRefused(f"role {args.role} is not registered for the {name} source {SOURCE_ROLES}")
    return name


def verify_source_registry(name: str, start: Mapping, base_path: Path, base_file_sha256: Optional[str] = None,
                           base_rel_path: str = BASE_REGISTRY_REL) -> dict:
    """The registered source's extension (contracts/manifest_registry/<ext>, never a user path): additions-only over the
    given REGISTRY_V2 (verify_generic_ext, `extends` = its path / file / self hash), the source's pair under its role,
    both manifests registered, and the run bound to the extension (check_source_binding)."""
    d = CENTRAL_SOURCES[name]
    ext_path = EXT_DIR / d["ext"]
    if not ext_path.is_file():
        raise ReevalRefused(f"the {name} registry extension {ext_path} does not exist")
    ext, base = _strict(ext_path), _strict(Path(base_path))
    chk = verify_generic_ext(ext, base, base_file_sha256 or _sha(Path(base_path)))
    if not chk["all_ok"] or (ext.get("extends") or {}).get("path") != base_rel_path:
        raise ReevalRefused(f"the {name} registry extension is refused: "
                            f"{[k for k, v in chk.items() if not v] or ['extends.path']}")
    pair = dict((ext.get("roles") or {}).get(d["ext_role"]) or {})
    if pair != {"train": d["train"], "eval": d["eval"]}:
        raise ReevalRefused(f"{d['ext']} role {d['ext_role']} is {pair}, not the registered {name} pair")
    missing = [m for m in (d["train"], d["eval"]) if m not in (ext.get("manifests") or {})]
    if missing:
        raise ReevalRefused(f"{d['ext']} does not register {missing}")
    binding = check_source_binding(name, start, ext, ext_path)
    return {"source": name, "arm_label": d["arm_label"], "ext_path": str(ext_path), "ext_file_sha256": _sha(ext_path),
            "ext_self_sha256": ext["self_sha256"], "ext_role": d["ext_role"], "pair": pair,
            "base_registry_file_sha256": base_file_sha256 or _sha(Path(base_path)), "checks": chk,
            "binding": binding}


def run_config_sha256(cfg: Mapping) -> Optional[str]:
    """v4train.recipe.RunConfig.sha256 on the recorded RUN_START config (= RunConfig.as_json(); JSON round-trip safe)."""
    try:
        return hashlib.sha256(json.dumps(cfg, sort_keys=True, allow_nan=False, default=list).encode("utf-8")).hexdigest()
    except (TypeError, ValueError):
        return None


def check_source_binding(name: str, start: Mapping, ext: Mapping, ext_path: Path) -> dict:
    """FIX-REEVAL: "trained on the registered extension's pair" as the REAL records show it. A central run launched
    through v4int.c_full / v4int.pretrain_small --ext-registry records THREE registry hashes:
      * config.key.manifest_registry_self_sha256 = the EXTENSION's self hash: the comparison key comparison_key_base
        built from the providers the launcher verified through the extension (LAUNCH_<k>.comparison_key is the same
        dict); it is inside the RunConfig, so config_sha256 (and the checkpoint meta's config_sha256) covers it;
      * key_base.manifest_registry_self_sha256 / manifest_registry_sha256 = the trainer's OWN registry = the per-run
        trainer VIEW (REGISTRY_V2_TRAINER_VIEW.json: roles MAIN / CONTINUATION = the extension's pair, derived_from =
        the extension) - v4train.trainer._data_identity_fields overwrites the config.key value with it;
      * RUN_START.manifest_registry = that same view (self / file sha256, bindings).
    All three are verified: config_sha256 recomputed, config.key's registry = the extension, the view recomputed from
    the extension (the shim's own trainer_view + check_trainer_view; derived_from.path = the extension's relative path)
    and equal to key_base / RUN_START.manifest_registry, and every other config.key field equal in the key base."""
    d = CENTRAL_SOURCES[name]
    import importlib
    vm = importlib.import_module(d["view_module"])
    cfg = start.get("config") or {}
    ck, kb = dict(cfg.get("key") or {}), dict(start.get("key_base") or {})
    mr = start.get("manifest_registry") or {}
    ext_file = _sha(ext_path)
    try:
        view = vm.trainer_view(ext, ext_rel_path=f"contracts/manifest_registry/{d['ext']}", ext_file_sha256=ext_file)
        vchk = vm.check_trainer_view(view, ext)
    except Exception as e:                                          # noqa: BLE001  (every refusal is a refusal)
        raise ReevalRefused(f"the {name} trainer view cannot be derived from {d['ext']}: {type(e).__name__}: {e}") from e
    view_file = hashlib.sha256((json.dumps(view, indent=1, sort_keys=True) + "\n").encode("utf-8")).hexdigest()
    checks = {
        "config_sha256 = RunConfig.sha256(config)": (run_config_sha256(cfg), start.get("config_sha256")),
        "config.key.manifest_registry_self_sha256 = extension self": (ck.get("manifest_registry_self_sha256"),
                                                                      ext.get("self_sha256")),
        "trainer view derived from the extension": (bool(vchk.get("all_ok")), True),
        "key_base.manifest_registry_self_sha256 = trainer view self": (kb.get("manifest_registry_self_sha256"),
                                                                       view["self_sha256"]),
        "key_base.manifest_registry_sha256 = trainer view file": (kb.get("manifest_registry_sha256"), view_file),
        "RUN_START.manifest_registry.self_sha256 = trainer view self": (mr.get("self_sha256"), view["self_sha256"]),
        "RUN_START.manifest_registry.file_sha256 = trainer view file": (mr.get("file_sha256"), view_file),
        "config.key fields unchanged in key_base": (
            sorted(f for f in ck if f != "manifest_registry_self_sha256" and kb.get(f) != ck[f]), [])}
    bad = {k: got for k, (got, want) in checks.items() if got != want or got is None}
    if bad:
        raise ReevalRefused(f"the {name} run was not built through {d['ext']} (manifest_registry_self_sha256 binding, "
                            f"FIX-REEVAL): {bad}")
    return {"config_sha256": start.get("config_sha256"), "comparison_key_registry_self_sha256": ext["self_sha256"],
            "trainer_view_self_sha256": view["self_sha256"], "trainer_view_file_sha256": view_file,
            "ext_file_sha256": ext_file, "view_checks": vchk}


def check_source_checkpoint(name: str, obj: Mapping, ckpt: Path) -> dict:
    """REV-DELTA-03 m1 / m2: a registered source's checkpoint is what its file name says: the v4train checkpoint `kind`
    of the file (best.pt -> best, endpoint_6.0.pt -> endpoint_6.0), and an endpoint file carries meta mark = 6.0 (for PRE
    theta_pre = the 6.0 endpoint, 06K §1; not only the file name)."""
    fname = Path(ckpt).name
    want = CKPT_FILE_KIND.get(fname)
    if want is None or obj.get("kind") != want:
        raise ReevalRefused(f"the {name} checkpoint {fname} carries kind {obj.get('kind')!r}, not {want!r} "
                            "(v4train.checkpoint; REV-DELTA-03 m1)")
    mark = (obj.get("meta") or {}).get("mark")
    if fname == ENDPOINT_FILE:
        try:
            ok = abs(float(mark) - ENDPOINT_MARK) < 1e-9
        except (TypeError, ValueError):
            ok = False
        if not ok:
            raise ReevalRefused(f"the {name} {fname} carries meta mark {mark!r}, not {ENDPOINT_MARK} (REV-DELTA-03 m2)")
    return {"file": fname, "kind": obj.get("kind"), "mark": mark}


def check_pre_init_record(src: Mapping, ckpt: Path, got: str) -> dict:
    """REV-DELTA-03 m2: PRE's checkpoint is the theta_pre of the run's own PRETRAIN_FOR_FL_INIT.json (v4int.pretrain_small
    fl_init_record): same run id, fl_init.kind endpoint_6.0, fl_init.path = this checkpoint file and fl_init.sha256 = its
    sha256 (so --expect-sha256 is tied to the record in code, not by operator procedure)."""
    p = Path(src["run_dir"]) / PRE_INIT_FILE
    if not p.is_file():
        raise ReevalRefused(f"the PRE run has no {PRE_INIT_FILE} (the theta_pre record of DATA-10): {p}")
    rec = _strict(p)
    fl = rec.get("fl_init") or {}
    path = (Path(src["run_dir"]) / str(fl["path"])).resolve() if fl.get("path") else None
    checks = (("run_id", rec.get("run_id"), src.get("run_id")), ("fl_init.kind", fl.get("kind"), "endpoint_6.0"),
              ("fl_init.path", str(path), str(Path(ckpt).resolve())),
              ("fl_init.sha256", str(fl.get("sha256") or "").lower(), got.lower()))
    bad = {k: got_v for k, got_v, want in checks if got_v != want}
    if bad:
        raise ReevalRefused(f"the PRE checkpoint is not the theta_pre of {PRE_INIT_FILE}: {bad} (REV-DELTA-03 m2)")
    return {"path": str(p), "file_sha256": _sha(p), "fl_init_sha256": fl["sha256"]}


def source_seeds(name: str) -> tuple:
    """ANA-05: the seeds a registered central source is re-evaluated at: its registered seed + its DESCRIPTIVE seeds
    (C_FULL: 2026 + 2027 (06D §8 NICE; 06E §2); PRE: 2026 only)."""
    d = CENTRAL_SOURCES[name]
    return (int(d["seed"]),) + tuple(int(s) for s in d.get("descriptive_seeds") or ())


def is_descriptive_source_seed(name: Optional[str], seed) -> bool:
    return name is not None and int(seed) in tuple(CENTRAL_SOURCES[name].get("descriptive_seeds") or ())


def source_seed_labels(name: Optional[str], seed) -> dict:
    """ANA-05: the row labels of a source's DESCRIPTIVE seed (none of them a comparison-key field); {} for the
    registered seed (so a seed-2026 row is the pre-ANA-05 row) and for every non-source run."""
    if not is_descriptive_source_seed(name, seed):
        return {}
    return {"comparison_status": SOURCE_SEED_STATUS, "source_seed_status": SOURCE_SEED_STATUS,
            "source_seed_rule": SOURCE_SEED_RULE}


def add_source_seed_record(csrc_rec: dict, name: Optional[str], seed) -> dict:
    """ANA-05: REEVAL_RECORD / dry-run central_source gains seed_status + seed_rule for a DESCRIPTIVE seed only."""
    if is_descriptive_source_seed(name, seed):
        csrc_rec.update(seed=int(seed), seed_status=SOURCE_SEED_STATUS, seed_rule=SOURCE_SEED_RULE)
    return csrc_rec


def source_row_labels(name: str, src_reg: Mapping) -> dict:
    d = CENTRAL_SOURCES[name]
    return {"central_source": name, "source_train_manifest_id": d["train"], "source_eval_manifest_id": d["eval"],
            "source_manifest_role": d["ext_role"], "source_registry_self_sha256": src_reg["ext_self_sha256"]}


def install_source_ext(ext_path: Path, base_path: Path) -> None:
    """v4int.verify.verify_registry for the source extension path (make_ext_verify_wrapper; chained, one wrapper per
    extension path, every other path delegated unchanged)."""
    import v4int.verify as VF
    ext_path = Path(ext_path).resolve()
    covered = getattr(VF.verify_registry, "__reeval_source_exts__", frozenset())
    if str(ext_path) in covered:
        return
    inner = VF.verify_registry
    w = make_ext_verify_wrapper(inner, ext_path, base_path, error=VF.ManifestVerificationError)
    w.__reeval_source_exts__ = frozenset(covered | {str(ext_path)})
    if getattr(inner, "__reeval_ext__", False):
        w.__reeval_ext__ = True
    VF.verify_registry = w


def source_train_side(kb: Mapping, overrides: Mapping, base_kb: Mapping) -> tuple:
    """C_FULL on its own eval population's TEST role: the gated key with the SOURCE train side (train_manifest_id,
    population_hash), as on MAIN VALIDATION (the key of C_FULL's own rows; retention_report A-F7)."""
    side = {"train_manifest_id": base_kb["train_manifest_id"], "population_hash": base_kb["population_hash"]}
    return dict(kb, **side), dict(overrides, train_manifest_id=side["train_manifest_id"])


def source_fl(ckpt: Path, args) -> dict:
    """<run>/ckpt/<run_id>/<file> -> the FL run's LAUNCH records (all legs must agree)."""
    ckpt = Path(ckpt)
    if ckpt.parent.parent.name != "ckpt":
        raise ReevalRefused(f"{ckpt} is not under an FL <run>/ckpt/<run_id>/ directory")
    run_dir = ckpt.parent.parent.parent
    launches = sorted(run_dir.glob("LAUNCH_*.json"), key=lambda q: int(q.stem.split("_")[1]))
    if not launches:
        raise ReevalRefused(f"FL run {run_dir} has no LAUNCH_<k>.json")
    recs = [_strict(q) for q in launches]
    first = recs[0]
    for q, r in zip(launches, recs):
        for k in ("arm", "run_id", "arm_spec_sha256", "fl_config_digest", "label"):
            if r.get(k) != first.get(k):
                raise ReevalRefused(f"{q.name}: {k} differs from {launches[0].name}")
    if first.get("label") != "PRODUCTION" or str(first.get("run_id", "")).startswith(SMOKE):
        raise ReevalRefused("a smoke FL run's checkpoint is not evidence")
    if ckpt.parent.name != first.get("run_id"):
        raise ReevalRefused(f"checkpoint directory {ckpt.parent.name!r} is not the run id {first.get('run_id')!r}")
    if (first.get("fl05") or {}).get("role", "MAIN") != "MAIN":
        raise ReevalRefused("only MAIN FL runs are re-evaluated (an LR-screen run is not a downstream arm)")
    arm = first.get("arm")
    from v4int.launch_fl import base_arm                             # ANA-06: PF_WARM* are PF runs (FL-16)
    if (args.ckpt_kind == "pf") != (base_arm(str(arm)) == "PF"):
        raise ReevalRefused(f"FL arm {arm!r} needs --ckpt-kind {'pf' if base_arm(str(arm)) == 'PF' else 'fl'}")
    rec = first.get("small_recipe") or {}
    if rec.get("model_variant") != args.variant or rec.get("recipe_id") != args.recipe_id:
        raise ReevalRefused("the FL run's SMALL recipe differs from --variant / --recipe-id")
    if int((first.get("arm_spec") or {}).get("cfg", {}).get("seed", -1)) != int(args.seed):
        raise ReevalRefused("the FL run's seed differs from --seed")
    return {"kind": args.ckpt_kind, "run_dir": run_dir, "launch": first, "launch_files":
            {q.name: _sha(q) for q in launches}, "run_id": first["run_id"], "arm": arm}


WARM_BLOCK_ARM = {                       # ANA-06: (launch_fl warm label, dp epsilon) -> the block arm (FL-16 run-dir names)
    ("FA_WARM", None): "FA_WARM", ("FA_WARM_FROZEN", None): "FA_WARM_FROZEN",
    ("FP_WARM", None): "FP_WARM", ("PF_WARM", None): "PF_WARM", ("FA_Q8_WARM", None): "FA_Q8_WARM",
    ("FA_1024_WARM", None): "FA_1024_WARM", ("DP_WARM", 8.0): "DP8_WARM", ("DP_WARM", 1.0): "DP1_WARM",
    ("FP_WARM_FROZEN", None): "FP_WARM_FROZEN", ("PF_WARM_FROZEN", None): "PF_WARM_FROZEN",
    ("FA_Q8_WARM_FROZEN", None): "FA_Q8_WARM_FROZEN", ("FA_1024_WARM_FROZEN", None): "FA_1024_WARM_FROZEN",
    ("DP_WARM_FROZEN", 8.0): "DP8_FROZEN", ("DP_WARM_FROZEN", 1.0): "DP1_WARM_FROZEN"}
WARM_BLOCK_LABELS = tuple(sorted(set(WARM_BLOCK_ARM.values())))


def check_warm_line(args, src: Mapping) -> Optional[dict]:
    """ANA-06: a warm FL run (LAUNCH arm a launch_fl warm label, or arm_spec init_from) is re-evaluated only under its
    block arm label, and a warm block label only on that run; theta_pre init_from present, frozen tables exactly for S8.
    Returns the `warm_line` record for an FL-16 warm-line arm, None for FA_WARM(_FROZEN) and every cold run."""
    if args.ckpt_kind not in ("fl", "pf"):
        return None
    from v4int import launch_fl as LF
    la = src.get("launch") or {}
    arm, spec = str(src.get("arm")), (la.get("arm_spec") or {})
    warm = LF.is_warm_arm(arm) or arm in LF.WARM_LINE or spec.get("init_from") is not None
    if not warm and args.arm_label not in WARM_BLOCK_LABELS:
        return None                                                  # cold: unchanged
    eps = (la.get("dp_z") or {}).get("epsilon")
    try:
        eps = None if eps is None else float(eps)
    except (TypeError, ValueError):
        eps = None
    want = WARM_BLOCK_ARM.get((arm, eps if LF.base_arm(arm) == "DP" else None))
    if want is None or args.arm_label != want:
        raise ReevalRefused(f"FL arm {arm!r} (dp_z epsilon {eps!r}) is re-evaluated as the block arm {want!r}, not "
                            f"--arm-label {args.arm_label} (a warm run keeps its own label; FL-16 / 06N; ANA-06)")
    if spec.get("init_from") is None:
        raise ReevalRefused(f"the {arm} run's LAUNCH arm_spec carries no init_from (a warm arm starts from theta_pre)")
    if spec.get("arm", arm) != arm or bool(spec.get("frozen_item_tables")) != LF.is_frozen_arm(arm):
        raise ReevalRefused(f"the {arm} run's LAUNCH arm_spec ({spec.get('arm')!r}, frozen_item_tables "
                            f"{spec.get('frozen_item_tables')!r}) is not that warm arm (S8 freezes the item tables)")
    if arm not in LF.WARM_LINE:
        return None                                                  # FA_WARM / FA_WARM_FROZEN: unchanged records
    v = LF.warm_variant(arm)
    return {"launch_arm": arm, "block_arm": want, "cold_arm": LF.base_arm(arm), "variant": v,
            "stage": LF.WARM_STAGE[v], "frozen_item_tables": LF.is_frozen_arm(arm), "epsilon": eps,
            "authority": "FL-16; PREG1_AMENDMENT_06N N3.2 (ANA-06)"}


def check_fl_payload(payload: Mapping, src: Mapping) -> None:
    if payload.get("kind") != "weights" or not isinstance(payload.get("theta"), Mapping):
        raise ReevalRefused("the FL checkpoint is not a weights payload")
    if payload.get("run_id") != src["run_id"] or payload.get("config_digest") != src["launch"]["fl_config_digest"]:
        raise ReevalRefused("the FL checkpoint's run id / config digest differs from the run's LAUNCH record")


def check_pf_store(store_obj: Mapping, payload: Mapping, *, theta_digest: str, ckpt_name: str) -> None:
    if store_obj.get("kind") != "pf_store" or not isinstance(store_obj.get("store"), Mapping):
        raise ReevalRefused("--pf-store is not a launch_fl PF store")
    if store_obj.get("run_id") != payload.get("run_id") or store_obj.get("config_digest") != payload.get(
            "config_digest"):
        raise ReevalRefused("--pf-store belongs to another run / config than the checkpoint")
    if "theta_digest" in store_obj:
        if store_obj["theta_digest"] != theta_digest:
            raise ReevalRefused("--pf-store was saved with other shared weights than the checkpoint's (theta_digest)")
    elif ckpt_name != ENDPOINT_FILE:
        raise ReevalRefused("a pre-B2 PF store (no theta_digest) is the END store: it pairs with the endpoint "
                            "checkpoint only (the best-mark p_u of that run was never saved)")


# ------------------------------------------------------------------------------------------------ weights
def check_central_meta(obj: Mapping, start: Mapping, args) -> dict:
    """The v4train checkpoint meta against the source RUN_START (variant, seed, config sha256, init sha256)."""
    meta = obj.get("meta") or {}
    for key, want in (("model_variant", args.variant), ("seed", args.seed),
                      ("config_sha256", start.get("config_sha256"))):
        if meta.get(key) != want:
            raise ReevalRefused(f"checkpoint meta {key} {meta.get(key)!r} differs from the source ({want!r})")
    if meta.get("init_sha256") is not None and meta["init_sha256"] != start.get("init_sha256"):
        raise ReevalRefused("checkpoint meta init_sha256 differs from RUN_START")
    return meta


def load_central(bm, obj: Mapping, start: Mapping, args) -> dict:
    """The v4train checkpoint's `model` into a freshly built model; every recorded identity must agree."""
    from fl.numerics import state_digest
    meta = check_central_meta(obj, start, args)
    if bm.init_sha256 != start.get("init_sha256"):
        raise ReevalRefused("the rebuilt theta0 differs from the source RUN_START init_sha256")
    bm.module.load_state_dict(obj["model"], strict=True)
    d = state_digest(bm.adapter.broadcast_state())
    if meta.get("state_digest") is not None and meta["state_digest"] != d:
        raise ReevalRefused("the loaded weights' state digest differs from the checkpoint's recorded state_digest")
    return {"state_digest": d, "efe": meta.get("efe"), "exposures": meta.get("exposures"), "mark": meta.get("mark")}


def load_fl(adapter, payload: Mapping) -> dict:
    """The FL payload's theta into the harness (server) adapter: same keys, shapes and dtypes; digest re-verified."""
    from fl.numerics import state_digest
    theta, cur = payload["theta"], adapter.broadcast_state()
    if set(theta) != set(cur) or any(tuple(theta[k].shape) != tuple(cur[k].shape) or theta[k].dtype != cur[k].dtype
                                     for k in cur):
        raise ReevalRefused("the FL theta does not match the harness adapter's broadcast state (keys / shapes)")
    adapter.load_state_(theta)
    d = state_digest(adapter.broadcast_state())
    if d != state_digest(theta):
        raise ReevalRefused("the loaded FL theta does not reproduce its own digest")
    return {"state_digest": d, "efe": payload.get("efe"), "round": payload.get("round"), "phase": payload.get("phase")}


# ------------------------------------------------------------------------------------------------ evaluation
class _View:
    """A harness with some attributes replaced (evalp / key_base for another role); the rest delegates."""

    def __init__(self, base, **over):
        self.__dict__["_base"], self.__dict__["_over"] = base, over

    def __getattr__(self, k):
        over = self.__dict__["_over"]
        return over[k] if k in over else getattr(self.__dict__["_base"], k)


def central_harness(adapter, *, frozen, device, key_base: Mapping, labels: Mapping, seed: int, evaluator_sha: str,
                    evalp) -> SimpleNamespace:
    """The role_evaluator / pf_evaluator harness contract for a central checkpoint: the source run's key base and
    labels, the loaded adapter; no training follows, so no determinism switch-back is needed."""
    return SimpleNamespace(frozen=frozen, adapter=adapter, evaluator_sha=evaluator_sha, device=device,
                           _apply_determinism=lambda: None, cfg=SimpleNamespace(seed=int(seed)),
                           state=SimpleNamespace(exposures=0), N=1, _row_labels=lambda: dict(labels),
                           key_base=dict(key_base), evalp=evalp)


def central_labels(start: Mapping) -> dict:
    """The source harness's _row_labels analogue: its data identity (from its key base) + exposure point / scope."""
    kb, cfg = start.get("key_base") or {}, start.get("config") or {}
    lab = {f: kb[f] for f in IDENTITY_FIELDS if f in kb}
    lab.update(exposure_point=cfg.get("exposure_point"), population_scope=cfg.get("population"))
    return lab


def evaluate_role(harness, rep, key_base: Mapping, overrides: Mapping, *, store=None,
                  hex_by_euk=None) -> tuple:
    """(value, rows, ranks, result) of the harness's CURRENT weights on `rep`: launch_fl.role_evaluator, or for PF
    launch_device.pf_evaluator on a view of the harness whose evalp / key_base are the role's."""
    from v4int.launch_fl import role_evaluator
    if store is None:
        return role_evaluator(harness, rep, key_base, overrides)(0.0)
    from v4int.launch_device import pf_evaluator
    v, rows, ranks, res = pf_evaluator(_View(harness, evalp=rep, key_base=dict(key_base)), hex_by_euk)(0.0, store)
    return v, [dict(r, **overrides) for r in rows], ranks, res


def role_key_base(base_kb: Mapping, rk: Mapping, role: str) -> tuple:
    """(key base, row overrides) for a role other than MAIN - exactly launch_device.run: population_hash,
    candidate_order_hash and the train / eval manifest ids come from the role's comparison key; the rows are
    labelled manifest_role = role with the role's manifest ids."""
    kb = dict(base_kb, **{f: rk[f] for f in ROLE_KEY_FIELDS})
    return kb, {"manifest_role": role, "train_manifest_id": rk["train_manifest_id"],
                "eval_manifest_id": rk["eval_manifest_id"]}


def relabel(rows, *, arm_label: str, selection_rule: str, checkpoint_id: str, checkpoint_sha256: str, kind: str,
            source_run_id: Optional[str], efe, extra: Optional[Mapping] = None) -> list:
    from v4eval.rows import COMPARISON_KEY
    lab = {"method": arm_label, "method_code": arm_label, "selection_rule": selection_rule,
           "checkpoint_id": checkpoint_id, "checkpoint_sha256": checkpoint_sha256, "reeval": True,
           "reeval_checkpoint_kind": kind, "source_run_id": source_run_id, "efe": efe, "row_status": "REEVAL",
           "comparison_status": "REEVAL", "c_matched_eligible": False, "downstream_seed_allowed": False,
           **dict(extra or {})}
    bad = sorted(set(lab) & set(COMPARISON_KEY))
    if bad:
        raise ReevalRefused(f"relabel may not touch comparison-key fields {bad}")
    return [dict(r, **lab) for r in rows]


def label_rows(rows, args, *, checkpoint_sha256: str, source_run_id: Optional[str], efe, extra: Mapping) -> list:
    """run()'s row labelling: relabel + the 06L-C §C16 DESCRIPTIVE no-FT labels (no_ft_labels; {} off UNSEEN_VAL), or
    for a C_FULL / PRE preview on UNSEEN_VAL the 06O labels (preview_labels) instead."""
    ref = (preview_labels(args.role, args.arm_label) if is_preview_reference(args.role, args.arm_label)
           else no_ft_labels(args.role, args.arm_label))            # 06O / 06L-C §C16: DESCRIPTIVE on UNSEEN_VAL
    extra = dict(extra, **ref)
    return relabel(rows, arm_label=args.arm_label, selection_rule=args.selection_rule,
                   checkpoint_id=args.checkpoint_id, checkpoint_sha256=checkpoint_sha256, kind=args.ckpt_kind,
                   source_run_id=source_run_id, efe=efe, extra=extra)


def add_no_ft_record(record: dict, args) -> dict:
    """run()'s REEVAL_RECORD no_ft_reference (06L-C §C16), or preview_reference for a C_FULL / PRE preview on
    UNSEEN_VAL (06O O1); every other role's record is unchanged."""
    if is_preview_reference(args.role, args.arm_label):
        record["preview_reference"] = PREVIEW_REFERENCE
        record["preview_reference_rule"] = PREVIEW_RULE
    elif args.role == NO_FT_ROLE:
        record["no_ft_reference"] = no_ft_labels(args.role, args.arm_label)
    return record


def check_pinned_rows(rows, pin_sha: str) -> None:
    bad = [i for i, r in enumerate(rows) if r.get("status") != "MEASURED_PINNED_DEVICE"
           or r.get("device_pin_sha256") != pin_sha or r.get("eval_device_class") != PINNED_CLASS]
    if bad or not rows:
        raise ReevalRefused(f"{len(bad)} row(s) are not MEASURED_PINNED_DEVICE rows of the registered pin")


def write_outputs(out: Path, *, tag: str, run_id: str, role: str, rows, result, ranks, record: dict) -> dict:
    import numpy as np
    from v4int.launch_fl import write_json
    out = Path(out)                                     # new at run start (the FL harness may have written its view)
    if (out / "RESULT_ROWS.json").exists() or (out / "evals").exists():
        raise ReevalRefused(f"{out} already holds re-evaluation outputs")
    out.mkdir(parents=True, exist_ok=True)
    (out / "evals").mkdir()
    write_json(out / "RESULT_ROWS.json", {"run_id": run_id, "role": role, "reeval": True, "rows": rows})
    write_json(out / "evals" / f"{tag}.result.json", result)
    np.save(out / "evals" / f"{tag}.ranks_int32.npy", np.asarray(ranks, dtype=np.int32))
    rows_sha = _sha(out / "RESULT_ROWS.json")
    ranks_sha = _sha(out / "evals" / f"{tag}.ranks_int32.npy")
    # FIX-DELTA-01 F1: the top-level `result_rows_sha256` / `ranks_sha256` are the ds3r2 REEVAL_RECORD field names that
    # tools/ana_block_index.py reads; `outputs` is kept (same values) for the per-file view.
    record = dict(record, outputs={"RESULT_ROWS.json": rows_sha,
                                   f"evals/{tag}.result.json": _sha(out / "evals" / f"{tag}.result.json"),
                                   f"evals/{tag}.ranks_int32.npy": ranks_sha},
                  result_rows_sha256=rows_sha, ranks_sha256=ranks_sha,
                  n_rows=len(rows), status="COMPLETE")
    write_json(out / "REEVAL_RECORD.json", record)
    return record


# ------------------------------------------------------------------------------------------------ registry extension
def verify_generic_ext(ext: Mapping, base: Mapping, base_file_sha256: str) -> dict:
    """An additions-only REGISTRY_V2 extension: both self hashes verify, `extends` names the base file / self hash,
    same data_version, and every base top-level value (every entry of a mapping) is canonical-JSON identical."""
    from v4data08.d8_registry import SELF, canon, self_ok
    c = {"ext.self_sha256": self_ok(ext), "base.self_sha256": self_ok(base)}
    ex = ext.get("extends") or {}
    c["extends.self_sha256_eq_base"] = ex.get("self_sha256") == base.get(SELF)
    c["extends.file_sha256_eq_base"] = ex.get("file_sha256") == base_file_sha256
    c["data_version_eq_base"] = ext.get("data_version") == base.get("data_version")
    for k, v in base.items():
        if k == SELF:
            continue
        e = ext.get(k)
        if isinstance(v, Mapping):
            c[f"{k}.base_entries_identical"] = isinstance(e, Mapping) and all(
                sk in e and canon(e[sk]) == canon(sv) for sk, sv in v.items())
        else:
            c[f"top.{k}.identical"] = k in ext and canon(e) == canon(v)
    c["all_ok"] = all(c.values())
    return c


def make_ext_verify_wrapper(orig, ext_path: Path, base_path: Path, error=ValueError) -> Callable:
    """v4int.verify.verify_registry for the extension path: the ORIGINAL verifies the base under the lock, then the
    extension must pass verify_generic_ext (the v4int.c_full wrapper pattern); any other path is delegated unchanged."""
    ext_path, base_path = Path(ext_path).resolve(), Path(base_path)

    def verify_registry(registry_path, lock_self, lock_file, *, lock=None, root=None):
        if Path(registry_path).resolve() != ext_path:
            return orig(registry_path, lock_self, lock_file, lock=lock, root=root)
        base, base_file, _base_self, v1_ctx = orig(base_path, lock_self, lock_file, lock=lock, root=root)
        ext = _strict(ext_path)
        chk = verify_generic_ext(ext, base, base_file)
        if not chk["all_ok"]:
            raise error(f"registry extension {ext_path} refused: {[k for k, v in chk.items() if not v]}")
        return ext, _sha(ext_path), ext["self_sha256"], v1_ctx

    verify_registry.__wrapped__ = orig
    return verify_registry


def install_ext(ext_path: Path, lockctx: Mapping) -> dict:
    import v4int.verify as VF
    base_path = Path(lockctx["manifest_registry_path"])
    ext, base = _strict(ext_path), _strict(base_path)
    chk = verify_generic_ext(ext, base, lockctx["manifest_registry_file_sha256"])
    if not chk["all_ok"] or (ext.get("extends") or {}).get("path") != lockctx["manifest_registry_rel_path"]:
        raise ReevalRefused(f"--registry-ext refused: {[k for k, v in chk.items() if not v] or ['extends.path']}")
    if not getattr(VF.verify_registry, "__reeval_ext__", False):
        w = make_ext_verify_wrapper(VF.verify_registry, ext_path, base_path, error=VF.ManifestVerificationError)
        w.__reeval_ext__ = True
        VF.verify_registry = w
    return {"path": str(ext_path), "file_sha256": _sha(ext_path), "self_sha256": ext["self_sha256"], "checks": chk,
            "roles": ext["roles"]}


def resolve_role(roles: Mapping, role: str, base_roles: Mapping) -> dict:
    """The role's {train?, eval} pair; refuses panel-only roles, a base role redefined, and ungated TEST names."""
    if role not in roles:
        raise ReevalRefused(f"role {role!r} is not in the manifest registry")
    if role in base_roles and dict(roles[role]) != dict(base_roles[role]):
        raise ReevalRefused(f"role {role!r} differs between the extension and REGISTRY_V2")
    pair = dict(roles[role])
    if "eval" not in pair or set(pair) - {"train", "eval"}:
        raise ReevalRefused(f"role {role!r} is not a train/eval evaluation role ({sorted(pair)})")
    if any("TEST" in str(x).upper() for x in [role, *pair.values()]) and not test_gate_ok(role):
        raise ReevalRefused(f"role {role!r} names a TEST manifest: refused before the DATA-09 gate")
    return pair


# ------------------------------------------------------------------------------------------------ the run
def _dry_run(args, src: Mapping, got: str) -> dict:
    rec = {"dry_run": True, "checkpoint_sha256": got, "checkpoint_kind": args.ckpt_kind,
           "source_run": str(src["run_dir"]), "source_run_id": src["run_id"]}
    if args.ckpt_kind in ("fl", "pf"):
        from fl.checkpoint import load_checkpoint
        payload = load_checkpoint(args.ckpt)
        check_fl_payload(payload, src)
        rec.update(fl_round=payload.get("round"), fl_efe=payload.get("efe"))
        if args.ckpt_kind == "pf":
            store_obj = load_checkpoint(args.pf_store)
            from fl.numerics import state_digest
            check_pf_store(store_obj, payload, theta_digest=state_digest(payload["theta"]), ckpt_name=args.ckpt.name)
            rec["pf_store_sha256"] = _sha(args.pf_store)
    elif src.get("central_source") is not None:                       # GATE-B B1: the server-runnable source check
        name = src["central_source"]
        rec["central_source"] = add_source_seed_record(
            verify_source_registry(name, src["start"], EXT_DIR / "REGISTRY_V2.json"), name, args.seed)   # ANA-05
        import torch
        obj = torch.load(args.ckpt, map_location="cpu", weights_only=False)
        if not isinstance(obj, Mapping) or not isinstance(obj.get("model"), Mapping):
            raise ReevalRefused("the checkpoint is not a v4train {meta, model} checkpoint")
        meta = check_central_meta(obj, src["start"], args)
        rec["checkpoint_file"] = check_source_checkpoint(name, obj, args.ckpt)
        if src.get("pre_init_record") is not None:
            rec["pre_init_record"] = src["pre_init_record"]
        own = args.role in CENTRAL_SOURCES[name]["own_eval_roles"]
        rec.update(checkpoint_meta={k: meta.get(k) for k in ("model_variant", "seed", "config_sha256", "init_sha256",
                                                              "efe", "mark")},
                   role=args.role, arm_label=args.arm_label,
                   planned_key=("SOURCE_TRAIN_SIDE (the source key base's train_manifest_id / population_hash)" if own
                                else "ROLE_KEY (the role's registered pair; train side key-only)"),
                   row_labels=dict(source_row_labels(name, rec["central_source"]),
                                   **source_seed_labels(name, args.seed)))                     # ANA-05 ({} at s2026)
    return rec


def run(args) -> dict:
    check_args(args)
    got = _sha(args.ckpt)
    if got.lower() != args.expect_sha256.lower():
        raise ReevalRefused("checkpoint sha256 differs from --expect-sha256")
    if args.pf_store is not None and args.expect_pf_store_sha256 and \
            _sha(args.pf_store).lower() != args.expect_pf_store_sha256.lower():
        raise ReevalRefused("PF store sha256 differs from --expect-pf-store-sha256")
    src = source_central(args.ckpt, args) if args.ckpt_kind == "central" else source_fl(args.ckpt, args)
    if src.get("central_source") == "PRE":                           # REV-DELTA-03 m2: the sha tied to the record
        src["pre_init_record"] = check_pre_init_record(src, args.ckpt, got)
    check_no_ft_source(args, src)                                    # 06L-C §C16 (UNSEEN_VAL only)
    warm_line = check_warm_line(args, src)                           # ANA-06 (after the no-FT messages)
    if args.dry_run:
        rec = _dry_run(args, src, got)
        if warm_line is not None:
            rec["warm_line"] = warm_line
        if is_preview_reference(args.role, args.arm_label):          # 06O: the labels the rows will carry
            rec["row_labels"] = dict(rec.get("row_labels") or {}, **preview_labels(args.role, args.arm_label))
            rec["preview_reference"] = PREVIEW_REFERENCE
        elif args.role == NO_FT_ROLE:
            rec["no_ft_reference"] = dict(no_ft_labels(args.role, args.arm_label), planned=True)
        return rec
    out = Path(args.out_dir)
    if out.exists():
        raise ReevalRefused(f"output directory exists: {out}")
    from v4int.launch_central import LaunchRefused, export_environment, gpu_checks
    from v4sup.gpu import default_runner
    try:
        gpu = gpu_checks(args, default_runner, os.environ)             # before CUDA / torch (launcher rule)
        env_rec = export_environment(int(args.gpu))
    except LaunchRefused as e:
        raise ReevalRefused(str(e)) from e
    t0 = time.time()
    import v4train                                                   # ppsi_env first; captures the export
    import ppsi_env
    if v4train.LAUNCH_CUDA_VISIBLE_DEVICES != str(int(args.gpu)):
        raise ReevalRefused("the CUDA_VISIBLE_DEVICES export did not precede the v4train import")
    snap = ppsi_env.verify_snapshot()
    iso = ppsi_env.assert_isolated()
    import torch
    from fl.checkpoint import load_checkpoint
    from fl.numerics import state_digest
    from v4data.common import dataset_root
    from v4eval import evaluator_code_sha256
    from v4eval.device_pin import pin_for_device
    from v4eval.frozen_config import apply_runtime_flags, check_runtime_flags, eval_device_class
    from v4eval.frozen_config import load as frozen_eval
    from v4eval.rows import COMPARISON_KEY
    from v4int.launch_central import bind_data_protocol_lock_v2
    from v4int.providers import StoreEvalProvider, StoreTrainProvider, comparison_key_base
    from v4int.verify import DataPaths
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ReevalRefused("expected exactly one visible CUDA device")
    dev = torch.device("cuda:0")
    torch.cuda.set_device(dev)
    torch.set_num_threads(4)
    pin = pin_for_device(dev)                                        # raises on a wrong GPU name / torch / CUDA
    if pin is None or pin["device_class"] != PINNED_CLASS or eval_device_class(dev) != PINNED_CLASS:
        raise ReevalRefused(f"re-evaluation runs only on the registered pinned class {PINNED_CLASS} "
                            f"(live {eval_device_class(dev)}; pin {'absent' if pin is None else pin['device_class']})")
    say = lambda *_: None                                            # noqa: E731  (no value is ever printed)
    paths = DataPaths.default()
    try:
        lockctx = bind_data_protocol_lock_v2(paths, smoke=False)
    except LaunchRefused as e:
        raise ReevalRefused(str(e)) from e
    base_path = Path(lockctx["manifest_registry_path"])
    base_reg = _strict(base_path)
    ext_rec, roles, reg_path, gate = None, dict(base_reg["roles"]), base_path, None
    if is_test_role(args.role):                                      # INT-07: the gated TEST route
        gate, pair, ext_rec = install_test_role(args, base_path)
        reg_path = Path(args.registry_ext)
    else:
        if args.registry_ext is not None:
            ext_rec = install_ext(Path(args.registry_ext), lockctx)
            roles = dict(ext_rec.pop("roles"))
            if args.role not in base_reg["roles"]:                  # only an extension-only role reads through it
                reg_path = Path(args.registry_ext)
        pair = resolve_role(roles, args.role, base_reg["roles"])
    reg_kw = dict(history_policy=args.history_policy, paths=paths, log=say, manifest_registry=reg_path,
                  registry_root=V4)
    fz = frozen_eval()
    evaluator_sha = evaluator_code_sha256()
    store = hex_by_euk = None
    rec_inputs: dict = {"checkpoint": {"path": str(args.ckpt), "sha256": got, "kind": args.ckpt_kind}}

    if args.ckpt_kind == "central":
        from v4fs.schemas import load_registered
        from v4models.build import build
        start, cfg = src["start"], src["start"]["config"]
        schema_rec = start["schema"]
        schema = load_registered(schema_rec["schema_id"], Path(lockctx["schema_registry_path"]))
        if schema.self_sha256 != schema_rec["schema_self_sha256"]:
            raise ReevalRefused("source schema self hash differs from the registered schema")
        bm = build(FAMILY, schema, args.seed, dev, v1_root=dataset_root(), variant=args.variant,
                   dropout=tuple(cfg["dropout"]))
        obj = torch.load(args.ckpt, map_location="cpu", weights_only=False)
        if src.get("central_source") is not None:                    # REV-DELTA-03 m1 / m2 (before any weight is used)
            rec_inputs["checkpoint_file"] = check_source_checkpoint(src["central_source"], obj, args.ckpt)
            if src.get("pre_init_record") is not None:
                rec_inputs["pre_init_record"] = src["pre_init_record"]
        loaded = load_central(bm, obj, start, args)
        bm.module.eval()
        main_eval_id, main_train_id = start["eval_provider"]["manifest_id"], start["train_provider"]["manifest_id"]
        csrc = src.get("central_source")
        if csrc is None and dict(base_reg["roles"]["MAIN"]) != {"train": main_train_id, "eval": main_eval_id}:
            raise ReevalRefused("the source run's manifests are not the registry's MAIN pair")
        base_kb, labels = dict(start["key_base"]), central_labels(start)
        adapter, harness_tp, harness_ep = bm.adapter, None, None
        efe, extra = loaded["efe"], {"exposures": loaded["exposures"], "source_config_sha256": start["config_sha256"]}
        rec_inputs["source"] = {"RUN_START.json": src["start_sha256"], "run_dir": str(src["run_dir"])}
        if csrc is not None:                                         # GATE-B B1: a registered non-MAIN source
            src_reg = verify_source_registry(csrc, start, base_path, lockctx["manifest_registry_file_sha256"],
                                             lockctx["manifest_registry_rel_path"])
            rec_inputs["central_source"] = add_source_seed_record(src_reg, csrc, args.seed)   # ANA-05
            extra.update(source_row_labels(csrc, src_reg))
            extra.update(source_seed_labels(csrc, args.seed))              # ANA-05: DESCRIPTIVE seed ({} at s2026)
    else:
        csrc = None
        tr, harness_tp, harness_ep, fl_rec = build_fl_harness(args, src, paths, lockctx, out, say)
        payload = load_checkpoint(args.ckpt)
        check_fl_payload(payload, src)
        loaded = load_fl(tr.adapter, payload)
        if tr.device_pin_sha256 != pin["self_sha256"]:
            raise ReevalRefused("the FL harness did not bind the registered device pin")
        adapter, base_kb, labels = tr.adapter, dict(tr.key_base), None
        main_train_id, main_eval_id = src["launch"]["manifests"]["train"], src["launch"]["manifests"]["eval"]
        la = src["launch"]
        efe = loaded["efe"]
        extra = {"training_paradigm": "FEDERATED", "optimizer_lifecycle": "FRESH_ADAMW_PER_VISIT",
                 "fl_run_id": src["run_id"], "fl_arm": src["arm"], "arm_spec_sha256": la["arm_spec_sha256"],
                 "fl_lr_mult": float((la.get("fl05") or {}).get("lr_mult", 1.0)), "fl_round": loaded["round"]}
        rec_inputs["source"] = {"launch_files": src["launch_files"], "run_dir": str(src["run_dir"]),
                                "harness": fl_rec}
        if args.ckpt_kind == "pf":
            from fl.personal import PersonalStore
            store_obj = load_checkpoint(args.pf_store)
            check_pf_store(store_obj, payload, theta_digest=loaded["state_digest"], ckpt_name=args.ckpt.name)
            store = PersonalStore.from_state_dict(store_obj["store"])
            if "store_digest" in store_obj and store.digest() != store_obj["store_digest"]:
                raise ReevalRefused("the PF store does not reproduce its recorded store_digest")
            extra.update(pf_store_sha256=_sha(args.pf_store), pf_store_digest=store.digest(),
                         pf_personal_component="EVALUATED")
            rec_inputs["pf_store"] = {"path": str(args.pf_store), "sha256": extra["pf_store_sha256"],
                                      "store_digest": store.digest(), "half_efe": store_obj.get("half_efe")}

    # ---- the evaluated role: provider + key base exactly as launch_device.run
    if gate is not None:                                             # INT-07: a TEST role
        src_train = (harness_tp.manifest_binding if harness_tp is not None else
                     central_train_binding(src["start"]))
        rep, kb, overrides, key_train = evaluated_test_role(
            args, gate=gate, pair=pair, reg_kw=reg_kw, paths=paths, base_kb=base_kb, main_train_id=main_train_id,
            source_train_binding=src_train)
        if csrc is not None and args.role in CENTRAL_SOURCES[csrc]["own_eval_roles"]:   # B1: C_FULL on MAIN_TEST
            kb, overrides = source_train_side(kb, overrides, base_kb)
            key_train = dict(key_train, key_train_side="SOURCE", source_train_manifest_id=base_kb["train_manifest_id"])
    elif args.role == "MAIN" and csrc is not None and "MAIN" in CENTRAL_SOURCES[csrc]["own_eval_roles"]:
        if pair.get("eval") != main_eval_id:                         # B1: C_FULL on its own eval manifest (MAIN)
            raise ReevalRefused(f"the registry's MAIN eval manifest {pair.get('eval')} is not the {csrc} source's")
        src_ext = Path(rec_inputs["central_source"]["ext_path"])
        install_source_ext(src_ext, base_path)
        rep = StoreEvalProvider(main_eval_id, prefetch=0, **dict(reg_kw, manifest_registry=src_ext))
        if rep.fingerprint != src["start"]["eval_provider"]["fingerprint"]:
            raise ReevalRefused(f"the MAIN evaluation provider (through {src_ext.name}) fingerprint differs from the "
                                f"{csrc} source run's")
        kb, overrides, key_train = base_kb, {}, None
    elif args.role == "MAIN" and csrc is None:
        if pair != {"train": main_train_id, "eval": main_eval_id}:
            raise ReevalRefused("the registry's MAIN pair differs from the source run's")
        rep = harness_ep if harness_ep is not None else StoreEvalProvider(main_eval_id, prefetch=0, **reg_kw)
        if args.ckpt_kind == "central" and rep.fingerprint != src["start"]["eval_provider"]["fingerprint"]:
            raise ReevalRefused("the MAIN evaluation provider fingerprint differs from the source run's")
        kb, overrides, key_train = base_kb, {}, None
    else:
        rep = StoreEvalProvider(pair["eval"], prefetch=0, **reg_kw)
        if csrc is not None:                                         # B1: the role key, train side key-only
            if "train" not in pair:
                raise ReevalRefused(f"role {args.role} is eval-only: not registered for the {csrc} source")
            from v4int import retro_store as RS
            rtp = RS.train_key_ref(pair["train"], history_policy=reg_kw["history_policy"], paths=paths,
                                   manifest_registry=reg_kw["manifest_registry"],
                                   registry_root=reg_kw.get("registry_root"))
        elif "train" in pair:
            rtp = StoreTrainProvider(pair["train"], **reg_kw)
        elif harness_tp is not None:
            rtp = harness_tp
        else:                                                        # eval-only role: the harness pair's train
            rtp = StoreTrainProvider(main_train_id, **dict(reg_kw, manifest_registry=base_path))
            if rtp.fingerprint != src["start"]["train_provider"]["fingerprint"]:
                raise ReevalRefused("the source train provider fingerprint differs from the source run's")
        kb, overrides = role_key_base(base_kb, comparison_key_base(paths, rtp, rep), args.role)
        key_train = {"manifest_id": rtp.manifest_id, "fingerprint": rtp.fingerprint}
        if csrc is not None:
            key_train["key_only"] = True
    if store is not None:
        from v4base.identity import eval_client_key_hex_by_eval_user_key
        mdir = paths.manifest_dir
        if gate is not None:
            from v4int.retro_store import retro_manifest_dir
            mdir = retro_manifest_dir(Path(args.registry_ext), rep.manifest_id, V4)
        hex_by_euk = eval_client_key_hex_by_eval_user_key(rep, manifest_dir=mdir)
    if args.ckpt_kind == "central":
        harness = central_harness(adapter, frozen=fz, device=dev, key_base=kb, labels=labels, seed=args.seed,
                                  evaluator_sha=evaluator_sha, evalp=rep)
    else:
        harness = tr
    apply_runtime_flags(fz)
    check_runtime_flags(fz, dev)
    _v, rows, ranks, res = evaluate_role(harness, rep, kb, overrides, store=store, hex_by_euk=hex_by_euk)
    rows = label_rows(rows, args, checkpoint_sha256=got, source_run_id=src["run_id"], efe=efe, extra=extra)
    check_pinned_rows(rows, pin["self_sha256"])
    if gate is not None:
        check_test_rows(rows)
    if res.get("evaluator_code_sha256") != evaluator_sha:
        raise ReevalRefused("the result does not carry this tree's evaluator identity")
    record = {"format": RECORD_FORMAT, "task": "INT-06 v4int.reeval", "authority": "PREG1_AMENDMENT_06D §1",
              "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "argv": list(sys.argv),
              "checkpoint_kind": args.ckpt_kind, "checkpoint_sha256": got, "source_run_id": src["run_id"],
              "variant": args.variant, "recipe_id": args.recipe_id, "seed": args.seed, "role": args.role,
              "arm_label": args.arm_label, "selection_rule": args.selection_rule, "checkpoint_id": args.checkpoint_id,
              "history_policy": args.history_policy, "loaded_state_digest": loaded["state_digest"],
              "device": {"name": torch.cuda.get_device_name(dev), "eval_device_class": eval_device_class(dev),
                         "physical_gpu": int(args.gpu), "torch": torch.__version__, "cuda": torch.version.cuda},
              "device_class": pin["device_class"], "device_pin_sha256": pin["self_sha256"],
              "evaluator_code_sha256": evaluator_sha, "frozen_t3_sha256": fz.sha256,
              "inputs": dict(rec_inputs, lock_v2_self_sha256=lockctx["lock_self_sha256"],
                             registry={"path": lockctx["manifest_registry_rel_path"],
                                       "file_sha256": lockctx["manifest_registry_file_sha256"],
                                       "self_sha256": lockctx["manifest_registry_self_sha256"]},
                             registry_ext=ext_rec,
                             schema={"id": lockctx["schema_id"], "self_sha256": lockctx["schema_self_sha256"]},
                             eval_manifest={"manifest_id": rep.manifest_id, "fingerprint": rep.fingerprint,
                                            "binding": rep.manifest_binding},
                             key_train_manifest=key_train),
              "comparison_key": {f: rows[0].get(f) for f in COMPARISON_KEY
                                 if f not in ("metric", "k", "aggregation", "target_denominator")},
              "eval_manifest_id": rows[0].get("eval_manifest_id") or rep.manifest_id,
              "split": rows[0].get("split"),
              "test_gate": None if gate is None else gate.binding(),
              "environment": env_rec, "gpu_check": gpu,
              "snapshot": {"verified_files": len(snap["files"]), "isolated_modules": len(iso)},
              "wall_seconds": round(time.time() - t0, 3)}
    if csrc is not None:                                             # B1 (the MAIN-pair record is unchanged)
        record["central_source"] = {k: v for k, v in rec_inputs["central_source"].items() if k != "checks"}
    add_no_ft_record(record, args)                                   # 06L-C §C16 (every other record is unchanged)
    if warm_line is not None:                                        # ANA-06: FL-16 warm-line arms only
        record["warm_line"] = warm_line
    rid = f"REEVAL_{args.arm_label}_{args.role}_s{args.seed}_{args.selection_rule}"
    rec = write_outputs(out, tag=args.role, run_id=rid, role=args.role, rows=rows, result=res, ranks=ranks,
                        record=record)
    return {"status": rec["status"], "record": str(out / "REEVAL_RECORD.json"), "n_rows": rec["n_rows"]}


# ------------------------------------------------------------------------------------------------ TEST roles (INT-07)
def install_test_role(args, base_path: Path) -> tuple:
    """(gate, {train?, eval} pair, ext record) of a TEST role: the DATA-09 loader wrappers installed for the extension
    (role=None: no trainer view), then the role resolved under the gate (gate FIRST, then the extension verified
    against the lock-pinned REGISTRY_V2)."""
    from v4int import test_roles as TR
    ext_path = Path(args.registry_ext)
    try:
        TR.install(ext_path, role=None, eager=True)
        res = TR.resolve_role(args.role, ext_path=ext_path, base_path=Path(base_path))
    except Exception as e:                                          # noqa: BLE001  (every refusal is a refusal)
        raise ReevalRefused(f"TEST role {args.role} refused: {type(e).__name__}: {e}") from e
    pair = {k: res[k] for k in ("train", "eval") if res.get(k)}
    rec = {"path": str(ext_path), "file_sha256": _sha(ext_path), "self_sha256": res["registry_self_sha256"],
           "test_gate": res["gate"].binding(), "roles": {args.role: pair}}
    return res["gate"], pair, rec


def evaluated_test_role(args, *, gate, pair: Mapping, reg_kw: Mapping, paths, base_kb: Mapping, main_train_id: str,
                        source_train_binding: Optional[Mapping] = None) -> tuple:
    """(RETRO_TEST provider, key base, row overrides, key-train record) of a TEST role. The support manifest (or, for
    the eval-only TRUE_COLD_TEST, the source harness's train manifest) is verified through the extension for the key
    only (no store is built: a re-evaluation never reads a support row)."""
    from v4int import retro_store as RS
    from v4int.providers import StoreEvalProvider, comparison_key_base
    rep = StoreEvalProvider(pair["eval"], prefetch=0, test_gate=gate, **reg_kw)
    train_id = pair.get("train") or main_train_id
    rtp = RS.train_key_ref(train_id, history_policy=reg_kw["history_policy"], paths=paths,
                           manifest_registry=reg_kw["manifest_registry"], registry_root=reg_kw.get("registry_root"))
    if "train" not in pair:                                         # FIX-REEVAL: fail closed on a missing binding
        if not (source_train_binding or {}).get("content_sha256"):
            raise ReevalRefused("the eval-only TEST role needs the source run's train manifest binding (none recorded)")
        if source_train_binding["content_sha256"] != rtp.manifest_binding["content_sha256"]:
            raise ReevalRefused("the eval-only TEST role's key train manifest differs from the source run's train "
                                "manifest")
    rk = comparison_key_base(paths, rtp, rep)
    kb, over = RS.retro_role_key(base_kb, rk, gate, role=args.role,
                                 eval_manifest_content_sha256=rep.manifest_binding["content_sha256"])
    key_train = {"manifest_id": rtp.manifest_id, "content_sha256": rtp.manifest_binding["content_sha256"],
                 "key_only": True}
    return rep, kb, over, key_train


def check_test_rows(rows) -> None:
    bad = [i for i, r in enumerate(rows) if r.get("split") != "RETRO_TEST" or not r.get("test_exposure_register_entry_id")]
    if bad or not rows:
        raise ReevalRefused(f"{len(bad)} TEST row(s) do not carry split RETRO_TEST and the register entry id")


def build_fl_harness(args, src: Mapping, paths, lockctx: Mapping, out: Path, say) -> tuple:
    """The FL run's evaluation harness, rebuilt EXACTLY as launch_fl.run builds it (MAIN COHORT pair, the SMALL recipe,
    the registered theta0, run id of the FL run), then bound to the LAUNCH record (theta0, init sha256, comparison key,
    recipe sha256, manifests)."""
    from v4data.common import dataset_root
    from v4int.launch_central import (DETERMINISM_MODE, LaunchRefused, _json_safe, _theta0_for,
                                      build_trainer_registry_view, load_production_schema)
    from v4int.launch_fl import FAMILY as FL_FAMILY, _small_recipe
    from v4int.providers import StoreEvalProvider, StoreTrainProvider, comparison_key_base
    from v4train.provider import expected_manifests
    from v4train.recipe import RunConfig
    from v4train.trainer import CentralTrainer
    la = src["launch"]
    try:
        recipe = _small_recipe(SimpleNamespace(small_recipe=None, smoke=False))
        theta0 = _theta0_for(argparse.Namespace(theta0_variants=None, smoke=False), lockctx["schema_id"],
                             {"model_variant": recipe["model_variant"]}, int(args.seed))
    except LaunchRefused as e:
        raise ReevalRefused(str(e)) from e
    if recipe["recipe_sha256"] != (la.get("small_recipe") or {}).get("recipe_sha256"):
        raise ReevalRefused("the SMALL recipe sha256 differs from the FL run's LAUNCH record")
    harness_rec = la.get("harness") or {}
    if theta0 != harness_rec.get("theta0_init_sha256"):
        raise ReevalRefused("the registered theta0 differs from the FL run's harness theta0")
    view = build_trainer_registry_view(lockctx["manifest_registry_path"], Path(out),
                                       registry_rel_path=lockctx["manifest_registry_rel_path"],
                                       registry_file_sha256=lockctx["manifest_registry_file_sha256"])
    train_id, eval_id = expected_manifests("COHORT", "MAIN", registry=str(view["path"]))
    if {"train": train_id, "eval": eval_id} != dict(la.get("manifests") or {}):
        raise ReevalRefused("the MAIN pair differs from the FL run's LAUNCH manifests")
    kw = dict(history_policy=args.history_policy, paths=paths, log=say,
              manifest_registry=lockctx["manifest_registry_path"], registry_root=V4)
    tp = StoreTrainProvider(train_id, **kw)
    ep = StoreEvalProvider(eval_id, prefetch=0, **kw)
    key = comparison_key_base(paths, tp, ep)
    if _canon(json.loads(json.dumps(_json_safe(key)))) != _canon(la.get("comparison_key")):
        raise ReevalRefused("the rebuilt comparison key differs from the FL run's LAUNCH comparison key")
    cfg = RunConfig(run_id=la["run_id"], family=FL_FAMILY, seed=int(args.seed), exposure_point="end",
                    cuda_visible_devices=str(int(args.gpu)), phase="MAIN", population="COHORT",
                    peak_lr=recipe["peak_lr"], log_every=50, checkpoint_every=0, key=key, test_overrides=False,
                    determinism_mode=DETERMINISM_MODE, manifest_registry=str(view["path"]),
                    data_version=lockctx["data_version"], schema_registry=str(lockctx["schema_registry_path"]),
                    recipe_id=recipe["recipe_id"], recipe_registry=recipe["recipe_registry"],
                    model_variant=recipe["model_variant"], dropout=tuple(recipe["dropout"]),
                    schedule=recipe["schedule"], theta0_init_sha256=theta0)
    schema = load_production_schema(lockctx["schema_id"], lockctx["schema_self_sha256"],
                                    lockctx["schema_registry_path"])
    tr = CentralTrainer(cfg, schema, tp, ep, Path(out) / "eval_harness", device="cuda:0", v1_root=dataset_root())
    if tr.built.init_sha256 != harness_rec.get("init_sha256"):
        raise ReevalRefused("the rebuilt harness init sha256 differs from the FL run's LAUNCH record")
    return tr, tp, ep, {"init_sha256": tr.built.init_sha256, "theta0_init_sha256": theta0,
                        "config_sha256": tr.config_sha256, "launch_config_sha256": harness_rec.get("config_sha256"),
                        "recipe_sha256": recipe["recipe_sha256"], "view_self_sha256": view["self_sha256"]}


def main(argv=None) -> int:
    args = parse_args(argv)
    try:
        rec = run(args)
        print(json.dumps(rec, sort_keys=True, default=str))
        return 0
    except Exception as e:                                          # noqa: BLE001  (every refusal exits 1)
        print(f"REEVAL REFUSED: {type(e).__name__}: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
