"""tools/ana_block_index.py — builds <out>/BLOCK_INDEX.json (format PPSI_V4_ANA_BLOCK_INDEX_1) for tools/retention_report.py
from the downstream run directories, WITHOUT reading any metric value (task ANA-02).

  python -B tools/ana_block_index.py --split VALIDATION|TEST --device-class NVIDIA_A40-FP32-DET --pin <EVAL_DEVICE_PIN_A40.json>
         [--reeval <dirs...>] [--fl <dirs...>] [--dev <dirs...>] [--base <dirs...>] [--onnx <dirs...>] [--trunc <dirs...>]
         [--records <files or KEY=file ...>] --out <new dir> [--registry-ext <REGISTRY_V2_TEST_EXT.json>]
         [--allow-single-seed ARM ...]

WHAT IS READ (parsed as JSON; `_open_json` is the single parse point and refuses any other file name): LAUNCH_*.json,
RUN_SUMMARY.json, STATUS.json, REEVAL_RECORD.json, BENCH_RESULT.json, rows.meta.json and the --pin file — for arm, seed,
rule, checkpoint id, role, status, labels, paths and file sha256 only. Evidence files (RESULT_ROWS.json, rows.json,
evals/*.rows.json, rank files, records) are COPIED byte for byte (sha256 recorded), never parsed: no `rows` list and no
`value` is ever opened. Prints counts only.

EXPECTED LAYOUTS (MASTER_PLAN_V5 §3-§8, §11; ADDENDUM V5.1 §C; <R> = runs_ds3):
  --reeval <R>/REEVAL_A40      <ARM>_s<S>_<PB|CB>/ (and <ARM>_<ROLE>_s<S>_<PB|CB>/ for the panel re-evaluations, 6.7):
                               REEVAL_RECORD.json (status COMPLETE; arm_label, seed, role, selection_rule, checkpoint_id,
                               device_class, device_pin_sha256, eval_manifest_id, result_rows_sha256, ranks_sha256;
                               the two hashes are read from the top level and/or from `outputs` {"RESULT_ROWS.json",
                               "evals/<rank file>"} (REV-DELTA-01 F1; both present -> they must agree)),
                               RESULT_ROWS.json, evals/<one>.ranks_int32.npy. The TEST look (§11.2) writes the same layout
                               for every locked checkpoint (C_BIG, C_FULL, C_SMALL, FA, FP, PF, FA_Q8, FA_1024, DP8, DP1,
                               [DP4]; arm_label = the 06B / 06D arm name) on role MAIN_TEST / *_TEST.
  --fl <R>/FL                  <ARM>_s<S>/ (FA, FP, PF, FA_Q8, FA_1024, DP8, DP1, DP4, FA_WARM, FA_WARM_FROZEN = S8;
                               S8 needs RUN_SUMMARY frozen_item_tables + bytes.per_client_incl_frozen_once, GATE-B B3,
                               and a run with frozen tables is never FA_WARM): LAUNCH_<k>.json (label PRODUCTION,
                               fl05.role MAIN, dp_z.epsilon for DP), RUN_SUMMARY.json (status DONE_MAIN, arm, arm_spec.cfg
                               .seed, best_half_efe), RESULT_ROWS.json, metrics.jsonl, evals/eval_<m>.rows.json +
                               .ranks_int32.npy (12 marks). S_CAL (DONE_S_CAL) and FA_X2 (BUDGET_SENSITIVITY) are listed
                               as excluded. VALIDATION only (on TEST the FL checkpoints come from --reeval).
  --dev <R>/DEV                <ARM>_s<S>/ and <ARM>_<ROLE>_s<S>/ (LO, FT_C, FT_FA): LAUNCH_<k>.json, RUN_SUMMARY.json
                               (status DONE, arm, role, spec.seed, selected_cell — the UNSEEN_VAL selection, 06A §1 /
                               06E §1), RESULT_ROWS.json, cells/<cell>.ranks_int32.npy. SEL_* / --select runs are
                               excluded (their SELECTION.json enters through --records).
  --base <R>/BASE runs_ds/baselines runs_p3/baselines_main
                               <METHOD>/rows.json (+ RECEIPT.json, copied, not read) with METHOD one of RULE, SKNN, LAST,
                               POP, KNN_EMB_SMALL, KNN_EMB_BIG (the directory name is the method; v4base.run --out
                               <root>/<METHOD>); every baseline is historical_reference + RULE_FREE (CPU rows).
  --onnx <R>/ONNX              BENCH_RESULT.json (retention[tag] per tag onnx_int8 / onnx_fp32, DEV-04), evals/<tag>
                               .ranks_int32.npy, rows.meta.json (first_users, first_users_set_sha256): entries
                               C_SMALL_INT8 / C_SMALL_FP32, comparison_block ONNX_CPU, rows null, users_subset
                               {"rule": "SELECTION_KEY_V2_FIRST", "n", "set_sha256"} when first_users is set.
  --trunc <R>/TRUNC            SA_<arm>_s<S>/ and SB_<arm>_s<S>/: LAUNCH_<k>.json (study, label, ckpt file sha256),
                               STATUS.json (DONE), RESULT_ROWS.json -> records.TRUNC_ROWS [{"rows", "rule"}]; Study A's
                               rule is the checkpoint's (matched through the REEVAL_RECORD checkpoint_sha256), Study B is
                               PRACTICAL_BEST; T41 / E2X dirs (no RESULT_ROWS.json) are skipped. VALIDATION only.
  --records                    STRATA_CUTPOINTS_MAIN.json, S_RECORD.json, SELECTION_*.json, LR_SELECTION.json, DP_WOR*.json,
                               DP_Z_EPS4*.json, or KEY=file (KEY in STRATA, S_RECORD, SELECTION, LR_SELECTION, DP_WOR,
                               DP_Z_EPS4, VALIDATION_RESULT) -> copied into <out>/records/.

ANA-04 (PREG1_AMENDMENT_06L_C): --base dirs with v4base.run's RANKS_RECORD.json (+ ranks_int32.npy, n_tied_int32.npy;
  06L-C §C3) become BASELINE_CPU entries (RULE_FREE, not historical; role from the record's eval_manifest_id; the rows /
  ranks / ties sha256 must equal the record's; RANKS_RECORD carries no value); dirs without it stay historical point-only
  rows. --dev FT_PRE_s<S>/ (launch_device FT_C from the theta_pre base: LAUNCH start.run_id PRETRAIN_FOR_FL*) -> arm FT_PRE,
  and every device entry records `start_run_id`; a theta_pre-based run under any other name refuses. --fl DP8_FROZEN_s<S>/
  (summary arm DP, eps 8, frozen tables) -> DP8_FROZEN (single seed 2026 by registration). --reeval GRU_D128_s<S>_<PB|CB>
  (06L-C §C11) and <C_SMALL|FA>_UNSEEN_VAL_s<S>_<PB|CB> (role UNSEEN_VAL: only the no-FT references C_SMALL / FA,
  06L-C §C16). --records EVENT_SIZE*.json (the per-event size convention of 06L-C §C8).

FIX-DELTA-03 (REV-DELTA-03): a PRE / C_FULL REEVAL_RECORD must carry v4int.reeval's central_source for that arm with
  the source's training manifest (PRETRAIN_TRAIN_5_100_NONMAIN / FULL_TRAIN_5_100_TIEBREAK) as its train side, and no
  other arm may carry one (M1); FA_WARM / FA_WARM_FROZEN / DP8_FROZEN need RUN_SUMMARY arm_spec.init_from = the theta_pre
  identity (a <WARM>_s<S> directory without one refuses), no other FL arm may carry one (M2); a --trunc dir's optional
  TRUNC_PER_USER.npz (06L-C §C10 per-user evidence) is copied beside its RESULT_ROWS as records.TRUNC_ROWS[i].per_user.
  FIX-DEV-FTPRE M4: only when its sha256 equals the run's recorded per_user.sha256 (STATUS.json, which launch_trunc
  writes with the same per_user record as RESULT_ROWS.json; RESULT_ROWS.json is copied, never parsed); a file without
  that record, a record without the file, or a mismatch refuses.

DEV-FTPRE (06K §2 / 06L-C §C7): the FT_PRE base check uses the PRE init record, not a run-id prefix. --dev FT_PRE_s<S>/
  (and FT_PRE_<ROLE>_s<S>/) must hold a launch_device --arm FT_PRE run (RUN_SUMMARY arm FT_PRE) whose last LAUNCH start
  is kind PRE with init_from = the theta_pre identity (retention_report.init_from_problem None), run_id = init_from.run_id
  and init_record_sha256 (the PRETRAIN_FOR_FL_INIT.json file sha256); the entry records start_init and
  start_init_record_sha256. A PRE REEVAL_RECORD's inputs.pre_init_record (v4int.reeval, REV-DELTA-03 m2) becomes the PRE
  entry's pre_init {checkpoint_sha256 (= its fl_init_sha256, checked), init_record_sha256, run_id}; when the block holds
  both, every FT_PRE base must equal that theta_pre (retention_report.ft_pre_vs_pre_problem; the report re-checks and
  refuses FT_PRE without a PRE entry). An FT_PRE run under another directory name, or any other device run that starts
  from theta_pre (init_from or a PRETRAIN_FOR_FL start run id), refuses.

ANA-05 (C_FULL s2027; 06M; 06M_B B2 / B4): a C_FULL REEVAL_RECORD of seed 2027 must carry central_source.seed_status
  DESCRIPTIVE (v4int.reeval ANA-05) and a seed-2026 one must not; PRE never carries a seed_status. --fl FA_WARM_s<S> /
  FA_WARM_FROZEN_s<S>: every LAUNCH_<k>.json must carry the SAME launch_fl warm_selection (format
  PPSI_V4_FL_WARM_SELECTION_2, the arm's stage S7 / S8, file FL_WARM_SELECTION_<stage>.json, not a dry run) whose winner
  factors equal the RUN_SUMMARY arm_spec factors; the entries record it as `warm_selection`; --records
  FL_WARM_SELECTION_S7.json / _S8.json (keys WARM_SELECTION_S7 / _S8, copied, never parsed) must have exactly the
  bound sha256. A warm-start CALIB run (WARM_PREVIEW, 06M M2, or an S7 / S8 screen; LAUNCH / arm_spec role CALIBRATION
  with init_from, or a WARM_PREVIEW parent directory) refuses with that label. --dev FT_FA_WARM_s<S> /
  FT_FA_WARM_FROZEN_s<S> (launch_device --base-kind fa_warm, FL-15): RUN_SUMMARY arm = the directory's arm, LAUNCH start
  kind = the base arm (FA_WARM / FA_WARM_FROZEN) with base_kind fa_warm, a best.pt / endpoint_6.0.pt checkpoint, init_from
  = the theta_pre identity (retention_report.ft_warm_base_problem); the entry records the start as `start_base` (the
  report binds it to the block's base run of the same seed); a device run from an FA_WARM base under any other arm refuses.
  REV-ANA-05: a FA_WARM / FA_WARM_FROZEN --reeval dir carries its source FL run's LAUNCH binding (warm_selection) and
  identity (warm_run) read from REEVAL_RECORD inputs.source (run dir + launch_files, sha-checked; F1); the S_CAL exclusion
  runs before the preview check (F4).

FIX-REEVAL-06O (PREG1_AMENDMENT_06O): a --reeval <C_FULL|PRE>_UNSEEN_VAL_s<S>_<PB|CB> dir whose REEVAL_RECORD carries
  v4int.reeval's preview_reference NON_MAIN_UNSEEN_VAL is listed under `excluded` (retention_report.PREVIEW_REFERENCE_LABEL;
  nothing copied, never an entry, so it can sit in REEVAL_A40 beside the block dirs); a C_FULL / PRE UNSEEN_VAL record
  without that label keeps the no-FT refusal, and a preview_reference on any other arm / role refuses.

ANA-06 (PREG1_AMENDMENT_06N; FL-16; DL 287 m3): --fl <ARM>_s<S> of the FL-16 warm method set: RUN_SUMMARY arm = the
  launch_fl label (FP_WARM, PF_WARM, FA_Q8_WARM, FA_1024_WARM, DP_WARM + dp_z.epsilon 8 / 1 -> DP8_WARM / DP1_WARM, and
  the *_WARM_FROZEN labels; DP_WARM_FROZEN eps 8 -> DP8_FROZEN, eps 1 -> DP1_WARM_FROZEN = the FL-16 run-dir names),
  the directory arm must be that block arm (refused otherwise), the run must be the label's cold-arm semantics from
  theta_pre with frozen tables exactly for S8 (retention_report.warm_line_base_problem), and every LAUNCH leg carries
  its variant's S7 / S8 warm_selection (as FA_WARM / FA_WARM_FROZEN); S_CAL_WARM* (DONE_S_CAL) stay excluded. A
  --reeval warm-line dir binds its FL run the same way (LAUNCH arm = the launch label, dp_z epsilon). The 06N N3.3
  cold re-runs N33_<FP|PF|FA_Q8|FA_1024|DP8|DP1>_s<S> index as the cold arm (entry cold_rerun "06N N3.3"); the
  STOPPED_REALLOCATED cold dirs of 06N N4 (FP_s2026, FP_s2027, FA_1024_s2026; DL 317 D2) and any FL dir holding Ops'
  STOPPED_REALLOCATED.json marker (Fable audit M5; existence only, never parsed) are listed as excluded, and a --reeval
  dir whose REEVAL_RECORD source run dir is such a dir refuses (REV-ANA-06 m2). DP8_FROZEN needs 2 seeds. A legacy
  DP8_FROZEN run of arm DP refuses (no launcher writes it; FL-16 runs it as DP_WARM_FROZEN). --dev: a device run other
  than FT_FA_WARM(_FROZEN) whose LAUNCH start run id / path names a *_WARM* run refuses (DL 287 m3). --records
  S_RECORD_WARM.json / S_RECORD_WARM_FROZEN.json (keys S_RECORD_WARM / _FROZEN; copied, verified by the report).

REFUSES (exit 2, nothing left behind but an empty --out): a non-DONE run, a smoke / LR-screen / CALIBRATION dir, a device
class or pin that is not --device-class / --pin, a record whose file hashes do not verify, a duplicate (arm, seed, rule,
role), an unknown arm or role, and on MAIN an arm with a missing registered seed pair unless --allow-single-seed ARM
(C_FULL, DP1, DP4, C_SMALL_INT8, C_SMALL_FP32 are single-seed 2026 by registration: 06D §8, 06E §2; PRE: 06K). Output: the index
with self_sha256 (the retention_report indent-1 rule, verified there) and every copied file under <out>.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REFUSED, ERROR = 2, 5
SINGLE_SEED_REGISTERED = ("C_FULL", "DP1", "DP4", "C_SMALL_INT8", "C_SMALL_FP32", "PRE",    # PRE: one theta_pre (06K)
                          "DP1_WARM", "DP1_WARM_FROZEN")                                   # 06N N3.2: DP1 (1 seed)
#   REV-ANA-06 m1: DP8_FROZEN is no longer single-seed (06N N3.2 / 06P P4 K12w: 2 seeds; the 06L-C §C17 cut is superseded)
ONNX_TAGS = {"onnx_int8": "C_SMALL_INT8", "onnx_fp32": "C_SMALL_FP32"}
RECORD_KEYS = ("STRATA", "S_RECORD", "SELECTION", "LR_SELECTION", "DP_WOR", "DP_Z_EPS4", "VALIDATION_RESULT",
               "EVENT_SIZE",                                     # 06L-C §C8: the registered per-event size convention
               "WARM_SELECTION_S7", "WARM_SELECTION_S8",         # ANA-05 (06M_B B2): the registered 2-D warm selections
               "S_RECORD_WARM", "S_RECORD_WARM_FROZEN")          # ANA-06 (FL-16): the warm DP chain's S records
START_BASE_KEYS = ("kind", "base_kind", "run_id", "checkpoint", "file_sha256", "base_arm_spec_sha256", "init_from",
                   "warm_lr_factor", "warm_server_lr_factor", "frozen_item_tables", "base_round", "base_efe")
WARM_SELECTION_KEYS = ("path", "sha256", "self_sha256", "stage", "format", "winner_factor", "winner_server_lr_factor",
                       "winner_server_lr", "server_optimizer")
ALLOWED_JSON = re.compile(r"^(LAUNCH_\d+\.json|RUN_SUMMARY\.json|STATUS\.json|REEVAL_RECORD\.json|BENCH_RESULT\.json|"
                          r"rows\.meta\.json|RANKS_RECORD\.json)$")
FROZEN_FL_ARMS = ("FA_WARM_FROZEN", "DP8_FROZEN",              # S8 and its DP variant (06K §3; 06L-C §C17)
                  "FP_WARM_FROZEN", "PF_WARM_FROZEN", "FA_Q8_WARM_FROZEN", "FA_1024_WARM_FROZEN",   # + ANA-06: every
                  "DP1_WARM_FROZEN")                                                              # S8-bound arm
N33_PREFIX = "N33_"                                             # 06N N3.3: the cold arms re-run from scratch
N33_ARMS = ("FP", "PF", "FA_Q8", "FA_1024", "DP8", "DP1")
STOPPED_REALLOCATED = ("FP_s2026", "FP_s2027", "FA_1024_s2026")  # 06N N4: stopped, no value read, never evidence
STOPPED_MARKER = "STOPPED_REALLOCATED.json"     # Ops' read-only marker in a stopped dir (Fable audit M5; never parsed)
STOPPED_LABEL = ("STOPPED_REALLOCATED (PREG1_AMENDMENT_06N N4): a cold run stopped for the pretrained line, no value "
                 "read; never block evidence (its arm is re-run from scratch as N33_<ARM>_s<S>, 06N N3.3)")
SMOKE_WORDS = ("SMOKE", "LRSCREEN", "LR_SCREEN", "CALIB")
TRUNC_PER_USER_NAME = "TRUNC_PER_USER.npz"      # 06L-C §C10: optional per-user evidence (PPSI_V4_TRUNC_PER_USER_1)


class IndexRefused(RuntimeError):
    pass


def _load_report():
    """tools/retention_report.py (its constants: ARMS, ROLE_MAP, the index format and self-sha rule), by path."""
    path = (HERE / "retention_report.py").resolve()
    mod = sys.modules.get("retention_report")
    if mod is not None and Path(mod.__file__).resolve() == path:
        return mod
    spec = importlib.util.spec_from_file_location("_ana_index_retention_report", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod                      # (dataclasses resolve their module through sys.modules)
    spec.loader.exec_module(mod)
    return mod


RR = _load_report()


def sha_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _open_json(p: Path, *, pin: bool = False) -> dict:
    """The ONLY JSON parse point: status / identity files only (never a rows file, never a value)."""
    p = Path(p)
    if not pin and not ALLOWED_JSON.match(p.name):
        raise IndexRefused(f"{p}: the index builder never parses {p.name} (only LAUNCH / RUN_SUMMARY / STATUS / "
                           "REEVAL_RECORD / BENCH_RESULT / rows.meta)")
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise IndexRefused(f"{p}: unreadable ({e})") from None


def check_central_source(arm: str, rec: dict, where) -> None:
    """REV-DELTA-03 M1: a PRE / C_FULL REEVAL_RECORD must carry the verified central source of v4int.reeval (GATE-B B1)
    naming that arm, with the source's registered training manifest (PRETRAIN_TRAIN_5_100_NONMAIN / FULL_TRAIN_5_100_
    TIEBREAK) as its pair's train side; a record written without the source check (the pre-B1 reeval) or for another
    source refuses, and no other arm may carry a central source."""
    cs = rec.get("central_source")
    if arm in RR.SOURCE_TRAIN_MANIFEST:
        src = cs.get("source") if isinstance(cs, dict) else None
        train = ((cs.get("pair") or {}).get("train") if isinstance(cs, dict) else None)
        if src != arm or cs.get("arm_label") != arm or train != RR.SOURCE_TRAIN_MANIFEST[arm]:
            raise IndexRefused(f"{where}: a {arm} REEVAL_RECORD must carry central_source.source = {arm} with the pair "
                               f"train {RR.SOURCE_TRAIN_MANIFEST[arm]} (v4int.reeval GATE-B B1); it carries "
                               f"{src!r} / {train!r} (REV-DELTA-03 M1)")
    elif cs is not None:
        raise IndexRefused(f"{where}: arm {arm} carries a central_source record (only PRE / C_FULL are registered "
                           "central sources; REV-DELTA-03 M1)")
    if arm in RR.SOURCE_TRAIN_MANIFEST:                             # ANA-05: C_FULL s2027 = a DESCRIPTIVE source seed
        st = cs.get("seed_status")
        desc = rec.get("seed") in RR.DESCRIPTIVE_SEEDS.get(arm, ())
        if desc != (st == RR.DESCRIPTIVE_SEED_STATUS):
            raise IndexRefused(f"{where}: a {arm} seed {rec.get('seed')} REEVAL_RECORD carries central_source.seed_status "
                               f"{st!r}; the DESCRIPTIVE source seeds are {RR.DESCRIPTIVE_SEEDS} (v4int.reeval ANA-05; "
                               "06E §2)")


def pre_init_of(arm: str, rec: dict, where):
    """DEV-FTPRE: a PRE REEVAL_RECORD's binding to its PRETRAIN_FOR_FL_INIT.json (v4int.reeval inputs.pre_init_record,
    REV-DELTA-03 m2) -> {checkpoint_sha256, init_record_sha256, run_id}, or None (no binding recorded; the report then
    refuses an FT_PRE entry in the same block). The recorded fl_init sha256 must be the re-evaluated checkpoint's."""
    if arm != "PRE":
        return None
    pir = (rec.get("inputs") or {}).get("pre_init_record")
    if not isinstance(pir, dict):
        return None
    if pir.get("fl_init_sha256") != rec.get("checkpoint_sha256"):
        raise IndexRefused(f"{where}: the PRE REEVAL_RECORD's pre_init_record fl_init_sha256 "
                           f"{str(pir.get('fl_init_sha256'))[:12]}… is not its checkpoint_sha256 "
                           f"{str(rec.get('checkpoint_sha256'))[:12]}… (REV-DELTA-03 m2)")
    return {"checkpoint_sha256": rec.get("checkpoint_sha256"), "init_record_sha256": pir.get("file_sha256"),
            "run_id": rec.get("source_run_id")}


def record_hashes(rec: dict, ranks_name: str, where) -> tuple:
    """(result_rows_sha256, ranks_sha256) of a REEVAL_RECORD in either registered form (REV-DELTA-01 F1): the top-level
    fields (ds3r2 records; v4int.reeval since FIX-DELTA-01) and/or `outputs` = {"RESULT_ROWS.json": sha, "evals/<rank
    file>": sha} (the ds3r3 per-file view). When both forms are present they must agree; when neither is, refuse."""
    out = rec.get("outputs")
    if out is not None and not isinstance(out, dict):
        raise IndexRefused(f"{where}: REEVAL_RECORD.outputs is not an object")
    o_rows = (out or {}).get("RESULT_ROWS.json")
    o_ranks = (out or {}).get(f"evals/{ranks_name}")
    t_rows, t_ranks = rec.get("result_rows_sha256"), rec.get("ranks_sha256")
    for what, top, alt in (("result_rows_sha256", t_rows, o_rows), ("ranks_sha256", t_ranks, o_ranks)):
        if top is not None and alt is not None and top != alt:
            raise IndexRefused(f"{where}: REEVAL_RECORD.{what} disagrees with REEVAL_RECORD.outputs")
    rows_sha, ranks_sha = t_rows or o_rows, t_ranks or o_ranks
    if not rows_sha or not ranks_sha:
        raise IndexRefused(f"{where}: REEVAL_RECORD carries no result_rows_sha256 / ranks_sha256 (neither top-level nor "
                           f"outputs['RESULT_ROWS.json'] / outputs['evals/{ranks_name}'])")
    return rows_sha, ranks_sha


class Builder:
    def __init__(self, args):
        self.a = args
        self.out = Path(args.out)
        self.entries, self.excluded, self.sources = [], [], {}
        self.records = {}
        self.ckpt_rule = {}                         # checkpoint sha256 -> selection rule (REEVAL_RECORDs)
        self.pin = None

    # ------------------------------------------------------------------ helpers
    def copy_file(self, src: Path, rel: str) -> str:
        """Byte copy into the block (O_EXCL); the sha256 is recorded; the file is never parsed here."""
        src = Path(src)
        if not src.is_file():
            raise IndexRefused(f"{src} is missing")
        dst = self.out / rel
        if rel in self.sources:
            raise IndexRefused(f"two sources map to the block path {rel}")
        dst.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(str(dst), os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0), 0o644)
        with os.fdopen(fd, "wb") as f, open(src, "rb") as g:
            shutil.copyfileobj(g, f)
        sha = sha_file(dst)
        if sha != sha_file(src):
            raise IndexRefused(f"{rel}: the copy differs from {src}")
        self.sources[rel] = {"from": str(src.resolve()), "sha256": sha}
        return sha

    def cleanup(self) -> None:
        """A refusal leaves --out as it was (every file here was created O_EXCL by this run)."""
        for rel in self.sources:
            try:
                (self.out / rel).unlink()
            except OSError:
                pass
        for d in sorted((p for p in self.out.rglob("*") if p.is_dir()), key=lambda p: len(p.parts), reverse=True):
            try:
                d.rmdir()
            except OSError:
                pass

    def exclude(self, path: Path, why: str) -> None:
        self.excluded.append({"dir": str(Path(path).resolve()), "reason": why})

    def role_of(self, role: str, where: Path) -> str:
        r = str(role or "")
        if self.a.split == "TEST":
            if not r.endswith("_TEST"):
                raise IndexRefused(f"{where}: role {r!r} is not a TEST role on a TEST block")
            r = r[:-len("_TEST")]
        elif "TEST" in r.upper():
            raise IndexRefused(f"{where}: TEST role {r!r} in a VALIDATION block")
        if r not in RR.ROLE_MAP[self.a.split]:
            raise IndexRefused(f"{where}: role {role!r} is not a registered {self.a.split} role")
        return r

    def check_smoke_name(self, d: Path) -> None:
        up = d.name.upper()
        if any(w in up for w in SMOKE_WORDS):
            raise IndexRefused(f"{d}: a smoke / LR-screen / calibration directory is never block evidence")

    def check_pin(self, sha, where) -> None:
        if self.pin is not None and sha is not None and sha != self.pin["self_sha256"]:
            raise IndexRefused(f"{where}: device_pin_sha256 {sha} is not the --pin {self.pin['self_sha256']}")

    def add(self, e: dict) -> None:
        self.entries.append(e)

    @staticmethod
    def _dirs(roots) -> list:
        out = []
        for r in roots or []:
            r = Path(r)
            if not r.is_dir():
                raise IndexRefused(f"{r} is not a directory")
            out += [d for d in sorted(r.iterdir(), key=lambda x: x.name) if d.is_dir()]
        return out

    @staticmethod
    def _launch(d: Path) -> dict:
        ls = sorted(d.glob("LAUNCH_*.json"), key=lambda p: int(re.findall(r"\d+", p.name)[0]))
        return _open_json(ls[-1]) if ls else {}

    # ------------------------------------------------------------------ sources
    def reeval(self) -> None:
        for d in self._dirs(self.a.reeval):
            self.check_smoke_name(d)
            rp = d / "REEVAL_RECORD.json"
            if not rp.is_file():
                self.exclude(d, "no REEVAL_RECORD.json")
                continue
            rec = _open_json(rp)
            if rec.get("status") != "COMPLETE":
                raise IndexRefused(f"{d}: REEVAL_RECORD status {rec.get('status')!r} is not COMPLETE")
            arm, seed, rule = rec.get("arm_label"), rec.get("seed"), rec.get("selection_rule")
            if arm not in RR.ARMS or arm in RR.BASELINE_ARMS or arm in RR.ONNX_ARMS:
                raise IndexRefused(f"{d}: arm_label {arm!r} is not a registered block arm")
            if rule not in RR.RULES or seed not in RR.SEEDS:
                raise IndexRefused(f"{d}: selection_rule {rule!r} / seed {seed!r} not registered")
            m = re.fullmatch(r"(?P<head>.+)_s(?P<seed>\d{4})_(?P<rule>PB|CB)", d.name)
            if m and (not (m.group("head") == arm or m.group("head").startswith(arm + "_"))
                      or int(m.group("seed")) != seed
                      or {"PB": "PRACTICAL_BEST", "CB": "CONTROLLED_BUDGET"}[m.group("rule")] != rule):
                raise IndexRefused(f"{d}: the directory name disagrees with REEVAL_RECORD ({arm}, {seed}, {rule})")
            if rec.get("device_class") != self.a.device_class:
                raise IndexRefused(f"{d}: device_class {rec.get('device_class')!r} is not {self.a.device_class}")
            self.check_pin(rec.get("device_pin_sha256"), d)
            role = self.role_of(rec.get("role"), d)
            if role == RR.NO_FT_ROLE and arm in RR.PREVIEW_REFERENCE_ARMS and \
                    rec.get("preview_reference") == RR.PREVIEW_REFERENCE:   # 06O: listed as excluded, never an entry
                self.exclude(d, RR.PREVIEW_REFERENCE_LABEL)
                continue
            if rec.get("preview_reference") is not None:
                raise IndexRefused(f"{d}: REEVAL_RECORD preview_reference {rec.get('preview_reference')!r} on {arm} / "
                                   f"{role}: only a {' / '.join(RR.PREVIEW_REFERENCE_ARMS)} record on {RR.NO_FT_ROLE} "
                                   f"carries {RR.PREVIEW_REFERENCE} (v4int.reeval; 06O)")
            if role == RR.NO_FT_ROLE and arm not in RR.NO_FT_ARMS:
                raise IndexRefused(f"{d}: role {role} admits only the no-FT references {RR.NO_FT_ARMS} (06L-C §C16)")
            if rec.get("eval_manifest_id") != RR.ROLE_MAP[self.a.split][role][0]:
                raise IndexRefused(f"{d}: eval_manifest_id {rec.get('eval_manifest_id')!r} is not the registered "
                                   f"{role} manifest")
            check_central_source(arm, rec, d)
            src_dir = ((rec.get("inputs") or {}).get("source") or {}).get("run_dir")   # REV-ANA-06 m2 (DL 317)
            if src_dir and (Path(str(src_dir)).name in STOPPED_REALLOCATED or
                            (Path(str(src_dir)) / STOPPED_MARKER).exists()):
                raise IndexRefused(f"{d}: its source run {src_dir} is a STOPPED_REALLOCATED dir (06N N4; DL 317 D2: "
                                   "never evidence)")
            ranks = sorted((d / "evals").glob("*.ranks_int32.npy"))
            if len(ranks) != 1:
                raise IndexRefused(f"{d}: {len(ranks)} rank files under evals/ (need exactly 1)")
            rel = f"REEVAL/{d.name}"
            want_rows, want_ranks = record_hashes(rec, ranks[0].name, d)
            if self.copy_file(d / "RESULT_ROWS.json", f"{rel}/RESULT_ROWS.json") != want_rows:
                raise IndexRefused(f"{d}: RESULT_ROWS.json sha256 differs from the REEVAL_RECORD result_rows_sha256")
            if self.copy_file(ranks[0], f"{rel}/evals/{ranks[0].name}") != want_ranks:
                raise IndexRefused(f"{d}: the rank file sha256 differs from the REEVAL_RECORD ranks_sha256")
            self.copy_file(rp, f"{rel}/REEVAL_RECORD.json")
            if rec.get("checkpoint_sha256"):
                self.ckpt_rule[rec["checkpoint_sha256"]] = rule
            e = {"arm": arm, "seed": seed, "rule": rule, "role": role, "rows": f"{rel}/RESULT_ROWS.json",
                 "ranks": f"{rel}/evals/{ranks[0].name}", "checkpoint_id": rec.get("checkpoint_id"),
                 "checkpoint_sha256": rec.get("checkpoint_sha256"), "source": "REEVAL"}
            pi = pre_init_of(arm, rec, d)
            if pi is not None:
                e["pre_init"] = pi
            if RR.warm_stage_of(arm) is not None:                          # REV-ANA-05 F1 (+ ANA-06 warm line)
                e["warm_selection"], e["warm_run"] = reeval_warm_binding(d, rec, arm)
            self.add(e)

    def fl(self) -> None:
        if self.a.fl and self.a.split == "TEST":
            raise IndexRefused("--fl on a TEST block: the FL run dirs hold VALIDATION marks; the TEST look re-evaluates "
                               "the locked FL checkpoints (--reeval)")
        for d in self._dirs(self.a.fl):
            self.check_smoke_name(d)
            if d.name in STOPPED_REALLOCATED or (d / STOPPED_MARKER).exists():   # ANA-06: 06N N4 / DL 317 D2 by name,
                self.exclude(d, STOPPED_LABEL)                                    # or Ops' marker (M5); nothing read
                continue
            sp = d / "RUN_SUMMARY.json"
            if not sp.is_file():
                raise IndexRefused(f"{d}: no RUN_SUMMARY.json (a run that is not DONE)")
            summ, launch = _open_json(sp), self._launch(d)
            if summ.get("status") == "DONE_S_CAL":
                self.exclude(d, "S_CAL (its S_RECORD enters through --records)")
                continue
            check_warm_preview(d, summ, launch)          # ANA-05 (06M M2), after the S_CAL exclusion (REV-ANA-05 F4)
            if summ.get("status") != "DONE_MAIN":
                raise IndexRefused(f"{d}: RUN_SUMMARY status {summ.get('status')!r} is not DONE_MAIN")
            if launch.get("label") != "PRODUCTION" or (launch.get("fl05") or {}).get("role", "MAIN") != "MAIN":
                raise IndexRefused(f"{d}: LAUNCH label {launch.get('label')!r} / role "
                                   f"{(launch.get('fl05') or {}).get('role')!r} is not a PRODUCTION MAIN run")
            self.check_pin(launch.get("device_pin_sha256"), d)
            arm = summ.get("arm")
            if arm in RR.LAUNCH_WARM_LINE:                          # ANA-06: the FL-16 label -> its block arm
                arm = warm_line_block_arm(d, arm, launch)
            elif arm == "DP":
                eps = (launch.get("dp_z") or {}).get("epsilon")
                arm = {8.0: "DP8", 4.0: "DP4", 1.0: "DP1"}.get(float(eps) if eps is not None else None)
                if arm is None:
                    raise IndexRefused(f"{d}: a DP run without a registered dp_z.epsilon")
            seed = ((summ.get("arm_spec") or {}).get("cfg") or {}).get("seed")
            m = re.fullmatch(r"(?P<n33>N33_)?(?P<arm>.+)_s(?P<seed>\d{4})", d.name)
            n33 = bool(m and m.group("n33"))
            if n33 and (m.group("arm") not in N33_ARMS or m.group("arm") != arm):      # ANA-06: 06N N3.3 re-runs
                raise IndexRefused(f"{d}: an {N33_PREFIX}<ARM>_s<S> directory is a 06N N3.3 cold re-run of one of "
                                   f"{N33_ARMS} whose run is that arm; the run is {arm!r}")
            frozen = bool(summ.get("frozen_item_tables")) or \
                (summ.get("bytes") or {}).get("per_client_incl_frozen_once") is not None
            init = (summ.get("arm_spec") or {}).get("init_from")
            if m and RR.is_warm_start(m.group("arm")) and init is None:                          # REV-DELTA-03 M2
                raise IndexRefused(f"{d}: a {m.group('arm')} directory whose run has no arm_spec.init_from (a warm "
                                   "start from theta_pre; a cold-start run is never FA_WARM / S8 / DP8_FROZEN; 06K §2)")
            if m and m.group("arm") == "FA_WARM" and arm == "FA":      # 06K: FA from theta_pre, the launcher keeps "FA"
                arm = "FA_WARM"
            if m and m.group("arm") == "FA_WARM_FROZEN" and arm == "FA" and frozen:   # GATE-B B3: S8, same rule
                arm = "FA_WARM_FROZEN"
            if arm in FROZEN_FL_ARMS and not (summ.get("frozen_item_tables") and (summ.get("bytes") or {}).get(
                    "per_client_incl_frozen_once") is not None):
                what = "an FA_WARM_FROZEN (S8)" if arm == "FA_WARM_FROZEN" else f"a {arm} (S8 + DP)"
                raise IndexRefused(f"{d}: {what} run without frozen_item_tables / "
                                   "bytes.per_client_incl_frozen_once (its bytes charge the frozen tables once; 06K §3)")
            if arm not in FROZEN_FL_ARMS and frozen:
                raise IndexRefused(f"{d}: arm {arm!r} froze its item tables: only FA_WARM_FROZEN (S8) / DP8_FROZEN may "
                                   "(06K §3; 06L-C §C17)")
            try:
                RR.check_warm_init(arm, summ, str(d))                       # REV-DELTA-03 M2: both directions
            except RR.AnaRefused as e:
                raise IndexRefused(str(e)) from None
            if arm in RR.WARM_LINE_ARMS:                                # ANA-06: the label's semantics (FL-16)
                why = RR.warm_line_base_problem(arm, summ)
                if why:
                    raise IndexRefused(f"{d}: the run is not {arm} (launch_fl {RR.WARM_LINE_ARMS[arm][0]}): {why} (FL-16)")
            if m and m.group("arm") != arm and (arm in RR.WARM_LINE_ARMS or m.group("arm") in RR.WARM_LINE_ARMS):
                raise IndexRefused(f"{d}: directory arm {m.group('arm')} is not the run's block arm {arm} (a warm-line run "
                                   "lives in its FL-16 run-dir name <ARM>_s<S>; ANA-06)")
            if m and m.group("arm") != arm:
                self.exclude(d, f"directory arm {m.group('arm')} is not the run's block arm {arm} (e.g. FA_X2 "
                                "BUDGET_SENSITIVITY; never a block entry)")
                continue
            if arm not in RR.FL_ARMS or seed not in RR.SEEDS or (m and int(m.group("seed")) != seed):
                raise IndexRefused(f"{d}: arm {arm!r} / seed {seed!r} not registered or not the directory's")
            best = summ.get("best_half_efe")
            if best is None:
                raise IndexRefused(f"{d}: RUN_SUMMARY has no best_half_efe")
            wsel = warm_selection_of(d, arm, summ) if RR.warm_stage_of(arm) is not None else None   # ANA-05 / -06
            rel = f"FL/{d.name}"
            for f in ("RUN_SUMMARY.json", "RESULT_ROWS.json", "metrics.jsonl"):
                self.copy_file(d / f, f"{rel}/{f}")
            for p in sorted(d.glob("LAUNCH_*.json")):
                self.copy_file(p, f"{rel}/{p.name}")
            for mark in RR.FL_MARKS:
                for suffix in ("rows.json", "ranks_int32.npy"):
                    self.copy_file(d / "evals" / f"eval_{mark:.1f}.{suffix}", f"{rel}/evals/eval_{mark:.1f}.{suffix}")
            pb = float(best) / 2.0
            extra = {"warm_selection": dict(wsel)} if wsel is not None else {}
            if n33:
                extra["cold_rerun"] = "06N N3.3"
            self.add({"arm": arm, "seed": seed, "rule": "PRACTICAL_BEST", "role": "MAIN",
                      "rows": f"{rel}/RESULT_ROWS.json", "ranks": f"{rel}/evals/eval_{pb:.1f}.ranks_int32.npy",
                      "run_dir": rel, "source": "FL", **extra})
            self.add({"arm": arm, "seed": seed, "rule": "CONTROLLED_BUDGET", "role": "MAIN",
                      "rows": f"{rel}/RESULT_ROWS.json", "ranks": f"{rel}/evals/eval_6.0.ranks_int32.npy",
                      "source": "FL", **json.loads(json.dumps(extra))})

    def dev(self) -> None:
        for d in self._dirs(self.a.dev):
            self.check_smoke_name(d)
            sp = d / "RUN_SUMMARY.json"
            if d.name.upper().startswith("SEL_"):
                self.exclude(d, "a selection run (its SELECTION.json enters through --records)")
                continue
            if not sp.is_file():
                raise IndexRefused(f"{d}: no RUN_SUMMARY.json (a run that is not DONE)")
            summ, launch = _open_json(sp), self._launch(d)
            spec = summ.get("spec") or {}
            if spec.get("select"):
                self.exclude(d, "a --select run (UNSEEN_VAL selection)")
                continue
            if summ.get("status") != "DONE":
                raise IndexRefused(f"{d}: RUN_SUMMARY status {summ.get('status')!r} is not DONE")
            if launch and launch.get("label", "PRODUCTION") != "PRODUCTION":
                raise IndexRefused(f"{d}: LAUNCH label {launch.get('label')!r} is not PRODUCTION")
            self.check_pin(summ.get("device_pin_sha256"), d)
            arm, seed, cell = summ.get("arm"), spec.get("seed"), summ.get("selected_cell")
            start = (launch.get("start") or {}) if launch else {}
            start_run, start_init = start.get("run_id"), start.get("init_from")
            from_pre = start_init is not None or str(start_run or "").startswith(RR.FT_PRE_BASE_RUN_PREFIX)
            ftpre_dir = re.fullmatch(r"FT_PRE(?:_[A-Z_]+)?_s\d{4}", d.name) is not None
            warm_dir = next((a for a in sorted(RR.FT_WARM_BASE, key=len, reverse=True)
                             if re.fullmatch(rf"{a}(?:_[A-Z_]+)?_s\d{{4}}", d.name)), None)
            warm_start = start.get("base_kind") == RR.WARM_BASE_KIND or start.get("kind") in RR.FT_WARM_BASE.values()
            if warm_dir or arm in RR.FT_WARM_BASE or warm_start:             # ANA-05 (06M_B B4; FL-15)
                if warm_dir != arm or arm not in RR.FT_WARM_BASE:
                    raise IndexRefused(f"{d}: an FA_WARM / S8-based device run is a launch_device --arm FT_FA_WARM / "
                                       f"FT_FA_WARM_FROZEN --base-kind fa_warm run (RUN_SUMMARY arm {arm!r}) in an "
                                       f"<ARM>_s<S> / <ARM>_<ROLE>_s<S> directory of that arm (06M_B B4)")
                why = RR.ft_warm_base_problem(arm, start, start_run)
                if why:
                    raise IndexRefused(f"{d}: {arm}'s base must be a registered {RR.FT_WARM_BASE[arm]} MAIN checkpoint: "
                                       f"{why} (06M_B B4)")
            elif ftpre_dir or arm == "FT_PRE":                                 # DEV-FTPRE: the init record, not a prefix
                if not ftpre_dir or arm != "FT_PRE":
                    raise IndexRefused(f"{d}: an FT_PRE entry is a launch_device --arm FT_PRE run (RUN_SUMMARY arm "
                                       f"{arm!r}) in an FT_PRE_s<S> / FT_PRE_<ROLE>_s<S> directory (06L-C §C7)")
                if start.get("kind") != "PRE":
                    raise IndexRefused(f"{d}: FT_PRE's LAUNCH start kind {start.get('kind')!r} is not PRE (theta_pre "
                                       "from --init-from; 06L-C §C7)")
                why = RR.ft_pre_base_problem(start_init, start_run, start.get("init_record_sha256"))
                if why:
                    raise IndexRefused(f"{d}: FT_PRE's base must be theta_pre of the PRE init record: {why} (06L-C §C7)")
            elif from_pre:
                raise IndexRefused(f"{d}: a device run from theta_pre ({start_run!r}) is FT_PRE: launch it with --arm "
                                   "FT_PRE --init-from and name it FT_PRE_s<S> (06L-C §C7)")
            why = RR.warm_base_run_problem(arm, start_run, start.get("path"))      # ANA-06 (DL 287 m3)
            if why:
                raise IndexRefused(f"{d}: {why}")
            if arm not in RR.DEVICE_ARMS or seed not in RR.SEEDS:
                raise IndexRefused(f"{d}: arm {arm!r} / seed {seed!r} not a registered device arm")
            if not cell:
                raise IndexRefused(f"{d}: no selected_cell (the arm's UNSEEN_VAL SELECTION must be applied; 06A §1 / "
                                   "06E §1)")
            role = self.role_of(summ.get("role"), d)
            rel = f"DEV/{d.name}"
            for f in ("RUN_SUMMARY.json", "RESULT_ROWS.json"):
                self.copy_file(d / f, f"{rel}/{f}")
            for p in sorted(d.glob("LAUNCH_*.json")):
                self.copy_file(p, f"{rel}/{p.name}")
            self.copy_file(d / "cells" / f"{cell}.ranks_int32.npy", f"{rel}/cells/{cell}.ranks_int32.npy")
            e = {"arm": arm, "seed": seed, "rule": "PRACTICAL_BEST", "role": role,
                 "rows": f"{rel}/RESULT_ROWS.json", "ranks": f"{rel}/cells/{cell}.ranks_int32.npy",
                 "row_filter": {"device_cell": cell}, "source": "DEV"}
            if start_run is not None:
                e["start_run_id"] = start_run
            if arm == "FT_PRE":
                e["start_init"] = dict(start_init)
                e["start_init_record_sha256"] = start.get("init_record_sha256")
            if arm in RR.FT_WARM_BASE:                                  # ANA-05: the base identity for the report
                e["start_base"] = {k: start[k] for k in START_BASE_KEYS if k in start}
            self.add(e)

    def base(self) -> None:
        cands = []
        for r in self.a.base or []:
            r = Path(r)
            if not r.is_dir():
                raise IndexRefused(f"{r} is not a directory")
            cands += [r] + [d for d in sorted(r.iterdir(), key=lambda x: x.name) if d.is_dir()]
        for d in cands:
            if not (d / "rows.json").is_file():
                continue
            method = d.name.upper()
            if method not in RR.BASELINE_ARMS:
                self.exclude(d, f"directory {d.name} is not a baseline method name {RR.BASELINE_ARMS}")
                continue
            rel = f"BASE/{method}"
            rows_sha = self.copy_file(d / "rows.json", f"{rel}/rows.json")
            if (d / "RECEIPT.json").is_file():
                self.copy_file(d / "RECEIPT.json", f"{rel}/RECEIPT.json")
            rr = d / "RANKS_RECORD.json"
            if not rr.is_file():                                   # the DL 132 CPU rows: point-only (historical)
                self.add({"arm": method, "seed": None, "rule": "RULE_FREE", "role": "MAIN",
                          "rows": f"{rel}/rows.json", "historical_reference": True, "source": "BASE"})
                continue
            self.add(self.base_ranked(d, rel, method, rows_sha, _open_json(rr)))

    def base_ranked(self, d: Path, rel: str, method: str, rows_sha: str, rec: dict) -> dict:
        """06L-C §C3: a baseline run with v4base.run's per-row rank files -> a BASELINE_CPU entry (RULE_FREE, not
        historical). RANKS_RECORD.json binds rows.json / ranks_int32.npy / n_tied_int32.npy by sha256 (no value read)."""
        if rec.get("format") != "PPSI_V4_BASELINE_RANKS_1" or rec.get("method") != method:
            raise IndexRefused(f"{d}: RANKS_RECORD format / method {rec.get('format')!r} / {rec.get('method')!r} is not "
                               f"PPSI_V4_BASELINE_RANKS_1 / {method}")
        body = {k: v for k, v in rec.items() if k != "self_sha256"}
        if rec.get("self_sha256") != hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":"),
                                                               allow_nan=False).encode()).hexdigest():
            raise IndexRefused(f"{d}: RANKS_RECORD self_sha256 does not verify")
        if rec.get("rows_sha256") != rows_sha:
            raise IndexRefused(f"{d}: rows.json sha256 differs from RANKS_RECORD rows_sha256")
        role = _role_of_manifest(self.a.split, rec.get("eval_manifest_id"))
        if role is None:
            raise IndexRefused(f"{d}: RANKS_RECORD eval_manifest_id {rec.get('eval_manifest_id')!r} is not a registered "
                               f"{self.a.split} role manifest")
        role = self.role_of(role, d)
        ranks_rel = f"{rel}/{rec.get('ranks_file')}"
        if self.copy_file(d / str(rec.get("ranks_file")), ranks_rel) != rec.get("ranks_sha256"):
            raise IndexRefused(f"{d}: the rank file sha256 differs from RANKS_RECORD ranks_sha256")
        self.copy_file(d / "RANKS_RECORD.json", f"{rel}/RANKS_RECORD.json")
        e = {"arm": method, "seed": None, "rule": "RULE_FREE", "role": role, "rows": f"{rel}/rows.json",
             "ranks": ranks_rel, "comparison_block": RR.BASELINE_BLOCK, "source": "BASE"}
        if rec.get("n_tied_file"):
            ties_rel = f"{rel}/{rec['n_tied_file']}"
            if self.copy_file(d / str(rec["n_tied_file"]), ties_rel) != rec.get("n_tied_sha256"):
                raise IndexRefused(f"{d}: the tie file sha256 differs from RANKS_RECORD n_tied_sha256")
            e["row_ties"] = ties_rel
        return e

    def onnx(self) -> None:
        for d in [Path(x) for x in (self.a.onnx or [])]:
            bp = d / "BENCH_RESULT.json"
            if not bp.is_file():
                raise IndexRefused(f"{d}: no BENCH_RESULT.json")
            bench = _open_json(bp)
            ret = bench.get("retention")
            if not isinstance(ret, dict) or not ret:
                raise IndexRefused(f"{d}: BENCH_RESULT has no retention blocks (v4dev.bench --rows; DEV-04 format)")
            meta_p = d / "rows.meta.json"
            meta = _open_json(meta_p) if meta_p.is_file() else {}
            role = (self.role_of(_role_of_manifest(self.a.split, meta.get("manifest_id")) or "?", d)
                    if meta.get("manifest_id") else "MAIN")
            ck = (bench.get("checkpoint_meta") or {}).get("seed")
            if ck is None:
                hit = re.findall(r"_s(\d{4})", str(bench.get("checkpoint", "")))
                ck = int(hit[-1]) if hit else None
            if ck not in RR.SEEDS:
                raise IndexRefused(f"{d}: cannot tell the C_SMALL checkpoint seed from BENCH_RESULT")
            rel = f"ONNX/{d.name}"
            self.copy_file(bp, f"{rel}/BENCH_RESULT.json")
            meta_sha = self.copy_file(meta_p, f"{rel}/rows.meta.json") if meta_p.is_file() else None
            sub = None
            if meta.get("first_users") is not None:
                if not meta.get("first_users_set_sha256"):
                    raise IndexRefused(f"{d}: rows.meta.json has first_users but no first_users_set_sha256 (DEV-04)")
                sub = {"rule": RR.SUBSET_RULE_V2, "n": int(meta["first_users"]),
                       "set_sha256": meta["first_users_set_sha256"]}
            for tag, blk in sorted(ret.items()):
                arm = ONNX_TAGS.get(tag)
                if arm is None:
                    self.exclude(d, f"BENCH_RESULT retention tag {tag!r} is not onnx_int8 / onnx_fp32")
                    continue
                if meta_sha is not None and blk.get("rows_meta_sha256") not in (None, meta_sha):
                    raise IndexRefused(f"{d}: retention[{tag}].rows_meta_sha256 is not rows.meta.json's")
                src = d / "evals" / f"{tag}.ranks_int32.npy"
                if self.copy_file(src, f"{rel}/evals/{tag}.ranks_int32.npy") != blk.get("ranks_sha256"):
                    raise IndexRefused(f"{d}: {src.name} sha256 differs from retention[{tag}].ranks_sha256")
                e = {"arm": arm, "seed": ck, "rule": "PRACTICAL_BEST", "role": role, "rows": None,
                     "ranks": f"{rel}/evals/{tag}.ranks_int32.npy", "bench_result": f"{rel}/BENCH_RESULT.json",
                     "comparison_block": RR.ONNX_BLOCK, "source": "ONNX"}
                if sub:
                    e["users_subset"] = dict(sub)
                self.add(e)

    def trunc(self) -> None:
        if self.a.trunc and self.a.split == "TEST":
            raise IndexRefused("--trunc on a TEST block: no TEST truncation panel is registered (06D §2: M09 not "
                               "registered)")
        items = []
        for d in self._dirs(self.a.trunc):
            if not (d / "RESULT_ROWS.json").is_file():
                self.exclude(d, "no RESULT_ROWS.json (T41 / E2X / unfinished)")
                continue
            self.check_smoke_name(d)
            st = _open_json(d / "STATUS.json") if (d / "STATUS.json").is_file() else {}
            if st.get("status") != "DONE":
                raise IndexRefused(f"{d}: STATUS {st.get('status')!r} is not DONE")
            launch = self._launch(d)
            if launch.get("label") != "PRODUCTION":
                raise IndexRefused(f"{d}: LAUNCH label {launch.get('label')!r} is not PRODUCTION")
            study = launch.get("study")
            if study == "B":
                rule = "PRACTICAL_BEST"
            elif study == "A":
                ck = (launch.get("ckpt") or {}).get("file_sha256") or (launch.get("spec") or {}).get("ckpt_file_sha256")
                rule = self.ckpt_rule.get(ck)
                if rule != "PRACTICAL_BEST":
                    self.exclude(d, f"Study A checkpoint rule {rule or 'UNKNOWN (not a re-evaluated checkpoint)'}: "
                                    "the curves use PRACTICAL_BEST only")
                    continue
            else:
                raise IndexRefused(f"{d}: LAUNCH study {study!r} is not A / B")
            rel = f"TRUNC/{d.name}/RESULT_ROWS.json"
            self.copy_file(d / "RESULT_ROWS.json", rel)
            item = {"rows": rel, "rule": rule}
            want_pu = trunc_per_user_sha(d, st)                         # FIX-DEV-FTPRE M4 (None: no file, no record)
            if want_pu is not None:                                     # 06L-C §C10 persistence CIs (never parsed here)
                item["per_user"] = f"TRUNC/{d.name}/{TRUNC_PER_USER_NAME}"
                if self.copy_file(d / TRUNC_PER_USER_NAME, item["per_user"]) != want_pu:
                    raise IndexRefused(f"{d}: {TRUNC_PER_USER_NAME} changed while it was copied")
            items.append(item)
        if items:
            self.records["TRUNC_ROWS"] = items

    def records_files(self) -> None:
        for spec in self.a.records or []:
            key, _, path = spec.partition("=") if "=" in spec and spec.split("=", 1)[0] in RECORD_KEYS else ("", "", spec)
            p = Path(path)
            name = p.name
            if not key:
                key = ("STRATA" if name.startswith("STRATA_CUTPOINTS") else "S_RECORD" if name == "S_RECORD.json" else
                       "S_RECORD_WARM" if name == "S_RECORD_WARM.json" else
                       "S_RECORD_WARM_FROZEN" if name == "S_RECORD_WARM_FROZEN.json" else
                       "SELECTION" if name.startswith("SELECTION") else "LR_SELECTION" if name == "LR_SELECTION.json"
                       else "DP_WOR" if name.startswith("DP_WOR") else "DP_Z_EPS4" if name.startswith("DP_Z_EPS4")
                       else "EVENT_SIZE" if name.startswith("EVENT_SIZE")
                       else "WARM_SELECTION_S7" if name.startswith("FL_WARM_SELECTION_S7")
                       else "WARM_SELECTION_S8" if name.startswith("FL_WARM_SELECTION_S8") else None)
            if key is None:
                raise IndexRefused(f"--records {spec}: unknown record (use KEY=file with KEY in {RECORD_KEYS})")
            rel = f"records/{'VALIDATION_RESULT.json' if key == 'VALIDATION_RESULT' else name}"
            self.copy_file(p, rel)
            if key == "SELECTION":
                self.records.setdefault("SELECTION", []).append(rel)
            elif key in self.records:
                raise IndexRefused(f"--records: two files for {key}")
            else:
                self.records[key] = rel

    # ------------------------------------------------------------------ checks + write
    def check_entries(self) -> dict:
        seen = set()
        for e in self.entries:
            k = (e["arm"], e["seed"], e["rule"], e["role"])
            if k in seen:
                raise IndexRefused(f"duplicate entry {k}")
            seen.add(k)
        allow = set(SINGLE_SEED_REGISTERED) | set(self.a.allow_single_seed or [])
        missing = []
        for arm, rule in sorted({(e["arm"], e["rule"]) for e in self.entries if e["role"] == "MAIN"
                                 and e["rule"] != "RULE_FREE"}):
            seeds = {e["seed"] for e in self.entries if (e["arm"], e["rule"], e["role"]) == (arm, rule, "MAIN")}
            need = {2026} if arm in allow else set(RR.SEEDS)
            if not need <= seeds:
                missing.append(f"{arm}/{rule}: seeds {sorted(seeds)}")
        if missing:
            raise IndexRefused(f"missing registered seed(s) (use --allow-single-seed ARM for a registered cut): {missing}")
        for arm, st in sorted((a, RR.warm_stage_of(a)) for a in RR.ARMS if RR.warm_stage_of(a)):   # ANA-05 / -06
            rel = self.records.get(f"WARM_SELECTION_{st}")
            if rel is None:
                continue
            fsha = self.sources[rel]["sha256"]
            bad = sorted({str((e.get("warm_selection") or {}).get("sha256")) for e in self.entries
                          if e["arm"] == arm and e.get("warm_selection") is not None} - {fsha})
            if bad:
                raise IndexRefused(f"--records {rel}: sha256 {fsha[:12]}… is not the warm selection the {arm} runs are "
                                   f"bound to {[b[:12] for b in bad]} (06M_B B2)")
        pres = [(f"{e['arm']}|{e['seed']}|{e['rule']}|{e['role']}", e.get("pre_init")) for e in self.entries
                if e["arm"] == "PRE"]
        if pres:                                         # DEV-FTPRE: FT_PRE's base = PRE's theta_pre (init record)
            why = RR.ft_pre_vs_pre_problem([(f"{e['arm']}|{e['seed']}|{e['rule']}|{e['role']}", e.get("start_init"),
                                             e.get("start_init_record_sha256"))
                                            for e in self.entries if e["arm"] == "FT_PRE"], pres)
            if why:
                raise IndexRefused(f"{why} (06L-C §C7; DEV-FTPRE)")
        return {"n_entries": len(self.entries), "n_excluded": len(self.excluded)}

    def build(self) -> dict:
        a = self.a
        if a.split not in RR.ROLE_MAP:
            raise IndexRefused(f"--split {a.split!r}")
        if self.out.exists() and any(self.out.iterdir()):
            raise IndexRefused(f"{self.out} exists and is not empty (O_EXCL; use a new --out)")
        self.out.mkdir(parents=True, exist_ok=True)
        if a.device_class == RR.PINNED_CLASS:
            if not a.pin:
                raise IndexRefused(f"--device-class {RR.PINNED_CLASS} needs --pin <EVAL_DEVICE_PIN_A40.json>")
        if a.pin:
            pin = _open_json(Path(a.pin), pin=True)
            if pin.get("device_class", a.device_class) != a.device_class or not pin.get("self_sha256"):
                raise IndexRefused(f"--pin {a.pin}: device_class {pin.get('device_class')!r} / self_sha256 missing")
            self.pin = {"file_sha256": sha_file(Path(a.pin)), "self_sha256": pin["self_sha256"]}
        try:
            self.reeval()
            self.fl()
            self.dev()
            self.base()
            self.onnx()
            self.trunc()
            self.records_files()
            counts = self.check_entries()
        except Exception:
            self.cleanup()
            raise
        roles = {}
        for r in sorted({e["role"] for e in self.entries}):
            ev, tr = RR.ROLE_MAP[a.split][r]
            roles[r] = {"eval_manifest_id": ev, **({"train_manifest_id": tr} if tr else {})}
        order = {arm: i for i, arm in enumerate(RR.ARMS)}
        ents = sorted(self.entries, key=lambda e: (e["role"] != "MAIN", e["role"], order[e["arm"]],
                                                   e["seed"] or 0, e["rule"]))
        idx = {"format": RR.INDEX_FORMAT, "split": a.split, "eval_device_class": a.device_class,
               "device_pin": self.pin, "roles": roles, "entries": ents, "records": self.records,
               "builder": {"script": "tools/ana_block_index.py", "sha256": sha_file(Path(__file__).resolve()),
                           "retention_report_sha256": sha_file(HERE / "retention_report.py"),
                           "reads": "LAUNCH / RUN_SUMMARY / STATUS / REEVAL_RECORD / BENCH_RESULT / rows.meta only; "
                                    "evidence files copied byte for byte, never parsed"},
               "test_registry_ext": ({"path": str(Path(a.registry_ext).resolve()),
                                      "file_sha256": sha_file(Path(a.registry_ext))} if a.registry_ext else None),
               "sources": dict(sorted(self.sources.items())), "excluded": self.excluded}
        idx = json.loads(json.dumps(idx))
        idx[RR.SELF_KEY] = RR._canon_indent(idx)
        text = json.dumps(idx, indent=1, sort_keys=True, ensure_ascii=True, allow_nan=False) + "\n"
        fd = os.open(str(self.out / RR.INDEX_NAME), os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0),
                     0o644)
        with os.fdopen(fd, "wb") as f:
            f.write(text.encode("utf-8"))
        return dict(counts, n_roles=len(roles), n_records=len(self.records), n_files=len(self.sources),
                    index_self_sha256=idx[RR.SELF_KEY])


def check_warm_preview(d: Path, summ: dict, launch: dict) -> None:
    """ANA-05 (06M M2; 06M_B B1-B2): a warm-start CALIB run (WARM_PREVIEW or an S7 / S8 screen) is DESCRIPTIVE CALIB
    evidence and never a block entry: refused with that label (a cold LR-screen run keeps its old refusal)."""
    spec = summ.get("arm_spec") or {}
    calib = ((launch.get("fl05") or {}).get("role", "MAIN") != "MAIN") or spec.get("role", "MAIN") != "MAIN"
    warm = spec.get("init_from") is not None or summ.get("arm") in RR.WARM_LAUNCH_LABELS   # ANA-06: + FL-16 labels
    if d.parent.name.upper() == "WARM_PREVIEW" or (calib and warm):
        raise IndexRefused(f"{d}: {RR.PREVIEW_LABEL}")


def reeval_warm_binding(d: Path, rec: dict, arm: str) -> tuple:
    """REV-ANA-05 F1: (warm_selection, warm_run) of a FA_WARM / FA_WARM_FROZEN REEVAL_RECORD: its source FL run's
    LAUNCH_<k>.json files (REEVAL_RECORD inputs.source.run_dir + launch_files, v4int.reeval) must exist with exactly the
    recorded sha256s; they must carry one warm_selection bound to the LAUNCH arm_spec factors (warm_selection_of) and
    name this arm and seed. warm_run = the LAUNCH run_id / arm_spec_sha256 / seed / factors (the report binds the
    entry's rows fl_run_id / arm_spec_sha256 to it)."""
    src = (rec.get("inputs") or {}).get("source") or {}
    files = src.get("launch_files") or {}
    rd = Path(str(src.get("run_dir") or ""))
    if not files or not src.get("run_dir") or not rd.is_dir():
        raise IndexRefused(f"{d}: a {arm} REEVAL_RECORD needs its FL run's LAUNCH files (inputs.source.run_dir + "
                           f"launch_files) to bind the 2-D warm selection; run dir {src.get('run_dir')!r} (REV-ANA-05 F1)")
    have = sorted(p.name for p in rd.glob("LAUNCH_*.json"))
    if have != sorted(files) or any(sha_file(rd / n) != str(v) for n, v in files.items()):
        raise IndexRefused(f"{d}: the FL run's LAUNCH files {have} are not the REEVAL_RECORD's launch_files "
                           f"{sorted(files)} with the recorded sha256 (REV-ANA-05 F1)")
    ls = sorted(have, key=lambda n: int(re.findall(r"\d+", n)[0]))
    la = _open_json(rd / ls[-1])
    spec = la.get("arm_spec") or {}
    want = RR.WARM_LINE_ARMS[arm][0] if arm in RR.WARM_LINE_ARMS else arm       # ANA-06: the FL-16 launch label
    if spec.get("arm") != want or int((spec.get("cfg") or {}).get("seed", -1)) != int(rec.get("seed", -2)):
        raise IndexRefused(f"{d}: the FL run's LAUNCH arm_spec is {spec.get('arm')!r} seed "
                           f"{(spec.get('cfg') or {}).get('seed')!r}, not {want} seed {rec.get('seed')} (REV-ANA-05 F1)")
    if arm in RR.WARM_LINE_ARMS:
        if warm_line_block_arm(rd, want, la) != arm:
            raise IndexRefused(f"{d}: the FL run's LAUNCH dp_z epsilon {(la.get('dp_z') or {}).get('epsilon')!r} makes "
                               f"it {warm_line_block_arm(rd, want, la)}, not {arm} (ANA-06)")
        why = RR.warm_line_base_problem(arm, {"arm": spec.get("arm"), "arm_spec": spec})
        if why:
            raise IndexRefused(f"{d}: the FL run is not {arm} (launch_fl {want}): {why} (FL-16; ANA-06)")
    ws = warm_selection_of(rd, arm, {"arm_spec": spec})
    run = {"run_id": la.get("run_id"), "arm_spec_sha256": la.get("arm_spec_sha256"), "seed": rec.get("seed"),
           "warm_lr_factor": float(spec.get("warm_lr_factor", 1.0)),
           "warm_server_lr_factor": float(spec.get("warm_server_lr_factor", 1.0))}
    return ws, run


def warm_selection_of(d: Path, arm: str, summ: dict) -> dict:
    """ANA-05 (06M_B B2; FL-15 launch_fl.bind_warm_selection): the LAUNCH warm_selection every leg of an FA_WARM /
    FA_WARM_FROZEN MAIN run carries (identical in every LAUNCH_<k>.json), checked by retention_report's
    warm_selection_problem against the run's arm_spec factors; the index keeps only its identity fields."""
    ls = sorted(d.glob("LAUNCH_*.json"), key=lambda p: int(re.findall(r"\d+", p.name)[0]))
    recs = [(p.name, _open_json(p).get("warm_selection")) for p in ls]
    if not recs or any(w != recs[0][1] for _n, w in recs):
        raise IndexRefused(f"{d}: the LAUNCH legs do not carry one warm_selection ({[n for n, _w in recs]}; 06M_B B2)")
    ws = recs[0][1]
    why = RR.warm_selection_problem(arm, ws, summ.get("arm_spec") or {})
    if why:
        raise IndexRefused(f"{d}: {arm} MAIN is not bound to the registered 2-D "
                           f"FL_WARM_SELECTION_{RR.warm_stage_of(arm)}.json: {why} (06M_B B2; FL-15)")
    return {k: ws[k] for k in WARM_SELECTION_KEYS if k in ws}


def warm_line_block_arm(d: Path, label: str, launch: dict) -> str:
    """ANA-06 (FL-16): the block arm of a launch_fl warm-line label (= the FL-16 run-dir name): DP_WARM / DP_WARM_FROZEN
    by the LAUNCH dp_z.epsilon (8 -> DP8_WARM / DP8_FROZEN, 1 -> DP1_WARM / DP1_WARM_FROZEN), every other label as is;
    S_CAL_WARM* (DONE_S_CAL, excluded before) and an unregistered epsilon refuse."""
    if label in RR.WARM_LINE_EXCLUDED:
        raise IndexRefused(f"{d}: a {label} run is the 20-round S calibration (status DONE_S_CAL, its S record enters "
                           "through --records), never a block entry")
    eps = (launch.get("dp_z") or {}).get("epsilon")
    try:
        eps = None if eps is None else float(eps)
    except (TypeError, ValueError):
        eps = "?"
    hits = [a for a, v in RR.WARM_LINE_ARMS.items() if v[0] == label and v[3] == eps]
    if len(hits) != 1:
        raise IndexRefused(f"{d}: launch_fl arm {label} with dp_z.epsilon {eps!r} is not a registered warm-line arm "
                           f"({sorted(a for a, v in RR.WARM_LINE_ARMS.items() if v[0] == label)}; FL-16)")
    return hits[0]


def trunc_per_user_sha(d: Path, status: dict):
    """FIX-DEV-FTPRE M4: the recorded sha256 of a --trunc dir's TRUNC_PER_USER.npz (STATUS.json per_user, written by
    launch_trunc with the RESULT_ROWS.json per_user record), verified against the file; None when neither the file nor
    a record exists (a pre-DEV-FTPRE run). A file without a record, a record without the file, or a mismatch refuses."""
    f = Path(d) / TRUNC_PER_USER_NAME
    rec = (status or {}).get("per_user")
    want = rec.get("sha256") if isinstance(rec, dict) else None
    if not f.is_file():
        if want is not None:
            raise IndexRefused(f"{d}: STATUS.json records per_user sha256 {str(want)[:12]}… but {f.name} is absent")
        return None
    if not isinstance(want, str) or len(want) != 64:
        raise IndexRefused(f"{d}: {f.name} is present but STATUS.json records no per_user.sha256 for it")
    got = sha_file(f)
    if got != want.lower():
        raise IndexRefused(f"{d}: {f.name} sha256 {got[:12]}… differs from the recorded per_user.sha256 "
                           f"{want[:12]}… (a replaced or stale file)")
    return got


def _role_of_manifest(split: str, mid):
    for role, (ev, _tr) in RR.ROLE_MAP[split].items():
        if ev == mid:
            return role + ("_TEST" if split == "TEST" else "")
    return None


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--split", required=True, choices=("VALIDATION", "TEST"))
    ap.add_argument("--device-class", required=True)
    ap.add_argument("--pin", default=None)
    for name in ("reeval", "fl", "dev", "base", "onnx", "trunc", "records"):
        ap.add_argument(f"--{name}", nargs="*", default=[])
    ap.add_argument("--out", required=True)
    ap.add_argument("--registry-ext", default=None)
    ap.add_argument("--allow-single-seed", nargs="*", default=[])
    return ap.parse_args(argv)


def main(argv=None, say=print) -> int:
    args = parse_args(argv)
    try:
        counts = Builder(args).build()
    except IndexRefused as e:
        say(f"INDEX REFUSED: {e}")
        return REFUSED
    except Exception as e:
        import traceback
        say(f"INDEX ERROR: {type(e).__name__}: {e}\n{traceback.format_exc()}")
        return ERROR
    say("INDEX: written " + json.dumps(counts, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
