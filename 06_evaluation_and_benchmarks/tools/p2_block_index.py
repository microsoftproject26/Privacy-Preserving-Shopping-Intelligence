"""tools/p2_block_index.py — builds <out>/BLOCK_INDEX.json for tools/p2_report.py (work packet P2-12; frozen with it by
P2_SCRIPT_FREEZE), WITHOUT reading any metric value.

  python -B tools/p2_block_index.py --split VALIDATION|TEST --device-class NVIDIA_A40-FP32-DET --pin <EVAL_DEVICE_PIN_A40.json>
         --phase1-block <the Phase-1 block of the same split (tools/ana_block_index.py output)>
         [--reeval <P2 REEVAL dirs (v4int.reeval_p2)>] [--p2-fl <P2 FL run dirs>] [--p2-dev <FT_P2_WSTAR run dirs>]
         [--records KEY=file ...] [--p2-lock company/gates/P2_METHODS_LOCK.json] --out <new dir>
         [--registry-ext <REGISTRY_V2_TEST_EXT.json>] [--allow-single-seed ARM ...]

LIBRARY (PREG2 F6). Imports tools/p2_report.py (which imports tools/retention_report.py after asserting its sha256) and
tools/ana_block_index.py (unchanged; its file sha256 must be ABI_SHA256 = 6e7043bb…, asserted before it is executed and
again at run time; it resolves the same `retention_report` module object). The Phase-2 builder subclasses
ana_block_index.Builder and reuses unchanged: copy_file (O_EXCL byte copy, sha256 recorded), cleanup, exclude, role_of,
check_smoke_name, check_pin, _dirs, _launch, reeval (REEVAL_RECORD layout: status COMPLETE, arm_label, seed, role,
selection_rule, checkpoint id / sha256, device class / pin, eval_manifest_id, result_rows / ranks sha256 — the layout
v4int.reeval_p2 (P2-13, a shim around the unchanged reeval.py) writes for the P2 arm labels) and check_entries
(duplicates, registered seed pairs, FT_PRE base = PRE's theta_pre). Every Builder call runs inside p2_report.p2_profile
(the P2 labels in the library tables, restored on exit).

SOURCES
  --phase1-block  PREG2 §6.3 S5: the Phase-1 rows Phase 2 reuses (PRE, FA_WARM_FROZEN, FT_PRE, C_FULL) are produced
                  ONCE by the Phase-1 plan. Their MAIN entries are copied from the Phase-1 block (BLOCK_INDEX.json with a
                  verified self_sha256, format PPSI_V4_ANA_BLOCK_INDEX_1, the same split) verbatim — every key kept — with
                  every file they name (rows, ranks, the run dir) copied byte for byte to the SAME block-relative path,
                  so p2_report can compare each file sha256 with the Phase-1 RETENTION_RESULT's inputs.files_sha256 (S5:
                  hash-bound into both blocks). The entries gain phase1_reused = true and phase1_block {index self /
                  file sha256}. With --p2-lock exactly the lock's phase1_reused (arm, seed, rule) list is taken (missing
                  -> refused); without it every MAIN entry of the four arms. PRE is required (the anchor).
  --reeval        P2 REEVAL dirs: ana_block_index.Builder.reeval with the P2 labels admitted; an entry of any other arm
                  refuses (S5: a Phase-1 arm is never re-scored for Phase 2), a non-MAIN role refuses.
  --p2-fl         VALIDATION only (on TEST the P2 checkpoints come from --reeval): <P2 FL ARM>_s<S>/ run dirs of
                  launch_fl's P2 arm (RUN_SUMMARY arm P2_W, or the arm's admitted run labels; status DONE_MAIN; LAUNCH
                  label PRODUCTION, fl05.role MAIN; arm_spec.study PREG2, role MAIN, cfg.seed = the directory seed;
                  init_from = the theta_pre identity (retention_report.check_warm_init); frozen item tables carry the
                  frozen-once bytes record; DP arms the registered z), 12 marks copied -> PRACTICAL_BEST (run_dir) +
                  CONTROLLED_BUDGET entries.
  --p2-dev        FT_P2_WSTAR_s<S>/ (and FT_P2_WSTAR_<ROLE>_s<S>/) device runs: RUN_SUMMARY arm FT_P2_WSTAR, status DONE,
                  selected_cell, LAUNCH PRODUCTION whose start is kind P2_WSTAR with seed, checkpoint (best.pt /
                  endpoint_6.0.pt), file_sha256 and run_id; the entry records it as `p2_base` (Q5 base_best_only).
  --records       KEY=file with KEY in p2_report.P2_RECORD_KEYS (P2_WARM_SELECTION_FINAL / _A, P2_C1_SELECTION,
                  S_RECORD_P2, E1_MIA, E2_CANARY, E3_UPDATE_LEAKAGE, E4_SECAGG, E5_DPFTRL_LDP, DEVICES, D1_EXT) -> copied
                  to records/<KEY>.json, never parsed.
  --p2-lock       optional early S4 / S5 checks on TEST (p2_report re-checks everything): every P2 entry a sealed (arm,
                  seed, rule) with the sealed checkpoint sha256; no declared-not-run arm.
WHAT IS READ: LAUNCH / RUN_SUMMARY / STATUS / REEVAL_RECORD (ana_block_index._open_json), the Phase-1 BLOCK_INDEX.json and
the P2 lock — identity, paths and hashes only. Evidence files and records are copied, never parsed; no value is opened.
Prints counts only. OUTPUT: <out>/BLOCK_INDEX.json (study PREG2, the retention_report index format, self_sha256 by the
indent-1 rule) and every copied file. A refusal leaves --out as it was (exit 2).
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REFUSED, ERROR = 2, 5


def _load_p2_report():
    path = (HERE / "p2_report.py").resolve()
    mod = sys.modules.get("p2_report")
    if mod is not None and Path(mod.__file__).resolve() == path:
        return mod
    spec = importlib.util.spec_from_file_location("p2_report", str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["p2_report"] = mod
    spec.loader.exec_module(mod)
    return mod


P2 = _load_p2_report()
RR = P2.RR


def _load_abi():
    """tools/ana_block_index.py, unchanged, only when its file sha256 is the frozen ABI_SHA256 (it then resolves the
    already-loaded retention_report module)."""
    path = (HERE / "ana_block_index.py").resolve()
    got = hashlib.sha256(path.read_bytes()).hexdigest()
    if got != P2.ABI_SHA256:
        raise ImportError(f"tools/ana_block_index.py sha256 {got} is not the frozen {P2.ABI_SHA256} (PREG2 F6)")
    mod = sys.modules.get("ana_block_index")
    if mod is None or Path(mod.__file__).resolve() != path:
        spec = importlib.util.spec_from_file_location("ana_block_index", str(path))
        mod = importlib.util.module_from_spec(spec)
        sys.modules["ana_block_index"] = mod
        spec.loader.exec_module(mod)
    if mod.RR is not RR:
        raise ImportError("tools/ana_block_index.py resolved another retention_report module object")
    return mod


ABI = _load_abi()
IndexRefused = ABI.IndexRefused
P2_FL_DIR = re.compile(r"(?P<arm>P2_[A-Z0-9_]+?)_s(?P<seed>\d{4})")
P2_DEV_DIR = re.compile(r"FT_P2_WSTAR(?:_(?P<role>[A-Z_]+?))?_s(?P<seed>\d{4})")
P2_FL_OPTIONAL = ("UPDATE_NORM_LOG.jsonl", "ARM_SPEC.json")      # D1-ext log (PREG2 §3.3) and the value-free spec
BASE_FILES = ("best.pt", "endpoint_6.0.pt")


class P2Builder(ABI.Builder):
    """ana_block_index.Builder for a Phase-2 block (see the module docstring)."""

    def __init__(self, args):
        super().__init__(args)
        self.lock = None
        self.phase1_rec = None

    def _copy_once(self, src: Path, rel: str) -> str:
        """copy_file, except that a block path already copied from a byte-identical file is kept (one reused run dir
        serves the PRACTICAL_BEST and CONTROLLED_BUDGET entries)."""
        if rel in self.sources:
            if ABI.sha_file(Path(src)) != self.sources[rel]["sha256"]:
                raise IndexRefused(f"two different files map to the block path {rel}")
            return self.sources[rel]["sha256"]
        return self.copy_file(src, rel)

    # ------------------------------------------------------------------ S5: the reused Phase-1 rows
    def phase1(self) -> None:
        src = Path(self.a.phase1_block)
        ip = src / RR.INDEX_NAME
        if not ip.is_file():
            raise IndexRefused(f"--phase1-block {src}: no {RR.INDEX_NAME}")
        idx = json.loads(ip.read_text(encoding="utf-8"))
        if idx.get("format") != RR.INDEX_FORMAT or str(idx.get("split", "")).upper() != self.a.split:
            raise IndexRefused(f"--phase1-block: format {idx.get('format')!r} / split {idx.get('split')!r} is not "
                               f"{RR.INDEX_FORMAT} / {self.a.split}")
        if idx.get("study") == P2.P2_STUDY:
            raise IndexRefused("--phase1-block is a Phase-2 block (study PREG2), not the Phase-1 block")
        if RR.SELF_KEY not in idx or idx[RR.SELF_KEY] != RR._canon_indent(idx):
            raise IndexRefused("--phase1-block: BLOCK_INDEX.json needs a verified self_sha256 (tools/ana_block_index.py)")
        want = None if self.lock is None else set(self.lock["phase1_reused"])
        taken = []
        for e in idx.get("entries") or []:
            if e.get("arm") not in P2.PHASE1_REUSED_ARMS or e.get("role", "MAIN") != "MAIN" or \
                    e.get("historical_reference"):
                continue
            k = (e.get("arm"), e.get("seed"), e.get("rule"))
            if want is None or k in want:
                taken.append(e)
        got = {(e.get("arm"), e.get("seed"), e.get("rule")) for e in taken}
        if want is not None and want - got:
            raise IndexRefused(f"--phase1-block lacks the P2_METHODS_LOCK phase1_reused entries {sorted(want - got, key=str)}")
        if not any(e.get("arm") == P2.P2_ANCHOR for e in taken):
            raise IndexRefused(f"--phase1-block has no MAIN {P2.P2_ANCHOR} entry (the Phase-2 anchor; S5)")
        rec = {"index_self_sha256": idx[RR.SELF_KEY], "index_file_sha256": ABI.sha_file(ip)}
        for e in taken:
            rels = [e[k] for k in ("rows", "ranks", "bench_result", "row_ties") if e.get(k)]
            if isinstance(e.get("users_subset"), str):
                rels.append(e["users_subset"])
            if e.get("run_dir"):
                rd = self._inside_src(src, e["run_dir"])
                rels += sorted(p.relative_to(src).as_posix() for p in rd.rglob("*") if p.is_file())
            files = {}
            for rel in rels:
                files[rel] = self._copy_once(self._inside_src(src, rel), rel)
            new = json.loads(json.dumps(e))
            new.update(phase1_reused=True, phase1_block=dict(rec), phase1_files_sha256=dict(sorted(files.items())))
            self.add(new)
        self.phase1_rec = dict(rec, path_note="the Phase-1 block of the same split", n_entries=len(taken))

    @staticmethod
    def _inside_src(src: Path, rel) -> Path:
        p = (src / str(rel)).resolve()
        if src.resolve() not in p.parents or not p.exists():
            raise IndexRefused(f"--phase1-block path {rel!r} leaves the block or does not exist")
        return p

    # ------------------------------------------------------------------ P2 sources
    def reeval(self) -> None:
        n0 = len(self.entries)
        super().reeval()
        for e in self.entries[n0:]:
            if e["arm"] not in P2.P2_ARMS:
                raise IndexRefused(f"--reeval {e['arm']} s{e['seed']}: only P2 arms are scored for a Phase-2 block; "
                                   f"{e['arm']} rows come once from the Phase-1 block (PREG2 §6.3 S5)")
            if e["role"] != "MAIN":
                raise IndexRefused(f"--reeval {e['arm']}: role {e['role']} is not MAIN (every sitting Q is on MAIN)")

    def p2_fl(self) -> None:
        dirs = self._dirs(self.a.p2_fl)
        if dirs and self.a.split == "TEST":
            raise IndexRefused("--p2-fl on a TEST block: the run dirs hold VALIDATION marks; the TEST look re-evaluates "
                               "the sealed P2 checkpoints (--reeval)")
        for d in dirs:
            self.check_smoke_name(d)
            m = P2_FL_DIR.fullmatch(d.name)
            arm = m.group("arm") if m else None
            if arm not in P2.P2_FL_ARMS:
                raise IndexRefused(f"{d}: a --p2-fl directory is <ARM>_s<S> with ARM one of {P2.P2_FL_ARMS}")
            sp = d / "RUN_SUMMARY.json"
            if not sp.is_file():
                raise IndexRefused(f"{d}: no RUN_SUMMARY.json (a run that is not DONE)")
            summ, launch = ABI._open_json(sp), self._launch(d)
            if summ.get("status") != "DONE_MAIN":
                raise IndexRefused(f"{d}: RUN_SUMMARY status {summ.get('status')!r} is not DONE_MAIN")
            if launch.get("label") != "PRODUCTION" or (launch.get("fl05") or {}).get("role", "MAIN") != "MAIN":
                raise IndexRefused(f"{d}: LAUNCH label {launch.get('label')!r} / role "
                                   f"{(launch.get('fl05') or {}).get('role')!r} is not a PRODUCTION MAIN run")
            self.check_pin(launch.get("device_pin_sha256"), d)
            if summ.get("arm") not in P2.P2_METHOD_ALT[arm]:
                raise IndexRefused(f"{d}: RUN_SUMMARY arm {summ.get('arm')!r} is not a run label of {arm} "
                                   f"{P2.P2_METHOD_ALT[arm]}")
            spec = summ.get("arm_spec") or {}
            if spec.get("study") != P2.P2_STUDY or spec.get("role", "MAIN") != "MAIN":
                raise IndexRefused(f"{d}: arm_spec study {spec.get('study')!r} / role {spec.get('role')!r} is not a "
                                   f"{P2.P2_STUDY} MAIN run (CALIB screens never enter a block; PREG2 F3 / S8)")
            seed = (spec.get("cfg") or {}).get("seed")
            if seed not in RR.SEEDS or int(m.group("seed")) != seed:
                raise IndexRefused(f"{d}: cfg.seed {seed!r} is not registered or not the directory's")
            try:
                RR.check_warm_init(arm, summ, str(d))
            except RR.AnaRefused as e:
                raise IndexRefused(str(e)) from None
            b = summ.get("bytes") or {}
            if summ.get("frozen_item_tables") and b.get("per_client_incl_frozen_once") is None:
                raise IndexRefused(f"{d}: a run with frozen item tables without bytes.per_client_incl_frozen_once")
            z = ((spec.get("cfg") or {}).get("dp") or {}).get("noise_multiplier")
            if arm in P2.P2_DP_Z and (z is None or abs(float(z) - P2.P2_DP_Z[arm]) > 1e-9):
                raise IndexRefused(f"{d}: noise multiplier {z!r} is not {arm}'s registered z {P2.P2_DP_Z[arm]} (PREG2 §5.5)")
            best = summ.get("best_half_efe")
            if best is None:
                raise IndexRefused(f"{d}: RUN_SUMMARY has no best_half_efe")
            rel = f"P2FL/{d.name}"
            for f in ("RUN_SUMMARY.json", "RESULT_ROWS.json", "metrics.jsonl"):
                self.copy_file(d / f, f"{rel}/{f}")
            for f in P2_FL_OPTIONAL:
                if (d / f).is_file():
                    self.copy_file(d / f, f"{rel}/{f}")
            for p in sorted(d.glob("LAUNCH_*.json")):
                self.copy_file(p, f"{rel}/{p.name}")
            for mark in RR.FL_MARKS:
                for suffix in ("rows.json", "ranks_int32.npy"):
                    self.copy_file(d / "evals" / f"eval_{mark:.1f}.{suffix}", f"{rel}/evals/eval_{mark:.1f}.{suffix}")
            pb = float(best) / 2.0
            extra = {"source": "P2_FL", "p2_label": spec.get("p2_label")}
            self.add({"arm": arm, "seed": seed, "rule": "PRACTICAL_BEST", "role": "MAIN",
                      "rows": f"{rel}/RESULT_ROWS.json", "ranks": f"{rel}/evals/eval_{pb:.1f}.ranks_int32.npy",
                      "run_dir": rel, **extra})
            self.add({"arm": arm, "seed": seed, "rule": "CONTROLLED_BUDGET", "role": "MAIN",
                      "rows": f"{rel}/RESULT_ROWS.json", "ranks": f"{rel}/evals/eval_6.0.ranks_int32.npy", **extra})

    def p2_dev(self) -> None:
        for d in self._dirs(self.a.p2_dev):
            self.check_smoke_name(d)
            m = P2_DEV_DIR.fullmatch(d.name)
            if not m:
                raise IndexRefused(f"{d}: a --p2-dev directory is FT_P2_WSTAR_s<S> / FT_P2_WSTAR_<ROLE>_s<S>")
            sp = d / "RUN_SUMMARY.json"
            if not sp.is_file():
                raise IndexRefused(f"{d}: no RUN_SUMMARY.json (a run that is not DONE)")
            summ, launch = ABI._open_json(sp), self._launch(d)
            spec = summ.get("spec") or {}
            if summ.get("status") != "DONE" or summ.get("arm") != "FT_P2_WSTAR":
                raise IndexRefused(f"{d}: RUN_SUMMARY status {summ.get('status')!r} / arm {summ.get('arm')!r} is not a "
                                   "DONE FT_P2_WSTAR run")
            if not launch or launch.get("label", "PRODUCTION") != "PRODUCTION":
                raise IndexRefused(f"{d}: no PRODUCTION LAUNCH record")
            self.check_pin(summ.get("device_pin_sha256"), d)
            seed, cell = spec.get("seed"), summ.get("selected_cell")
            if seed not in RR.SEEDS or int(m.group("seed")) != seed:
                raise IndexRefused(f"{d}: spec.seed {seed!r} is not registered or not the directory's")
            if not cell:
                raise IndexRefused(f"{d}: no selected_cell")
            start = launch.get("start") or {}
            if start.get("kind") != "P2_WSTAR" or start.get("checkpoint") not in BASE_FILES or \
                    not RR._hex64(start.get("file_sha256")) or not start.get("run_id") or \
                    str(start.get("run_id")).startswith(RR.PRETRAIN_RUN_PREFIX):
                raise IndexRefused(f"{d}: FT_P2_WSTAR's LAUNCH start must be kind P2_WSTAR with a best.pt / "
                                   "endpoint_6.0.pt checkpoint, its file_sha256 and run_id (never theta_pre)")
            role = self.role_of(summ.get("role"), d)
            if role != "MAIN":
                raise IndexRefused(f"{d}: role {role} is not MAIN")
            rel = f"P2DEV/{d.name}"
            for f in ("RUN_SUMMARY.json", "RESULT_ROWS.json"):
                self.copy_file(d / f, f"{rel}/{f}")
            for p in sorted(d.glob("LAUNCH_*.json")):
                self.copy_file(p, f"{rel}/{p.name}")
            self.copy_file(d / "cells" / f"{cell}.ranks_int32.npy", f"{rel}/cells/{cell}.ranks_int32.npy")
            self.add({"arm": "FT_P2_WSTAR", "seed": seed, "rule": "PRACTICAL_BEST", "role": role,
                      "rows": f"{rel}/RESULT_ROWS.json", "ranks": f"{rel}/cells/{cell}.ranks_int32.npy",
                      "row_filter": {"device_cell": cell}, "source": "P2_DEV", "start_run_id": start.get("run_id"),
                      "p2_base": {"arm": "P2_WSTAR", "seed": start.get("seed"), "checkpoint": start.get("checkpoint"),
                                  "checkpoint_sha256": start.get("file_sha256"), "run_id": start.get("run_id")}})

    def records_files(self) -> None:
        for spec in self.a.records or []:
            key, sep, path = str(spec).partition("=")
            if not sep or key not in P2.P2_RECORD_KEYS:
                raise IndexRefused(f"--records {spec}: a Phase-2 record is KEY=file with KEY in {P2.P2_RECORD_KEYS}")
            if key in self.records:
                raise IndexRefused(f"--records: two files for {key}")
            rel = f"records/{key}.json"
            self.copy_file(Path(path), rel)
            self.records[key] = rel

    def lock_checks(self) -> dict:
        """--p2-lock on TEST: S4 early (p2_report re-checks): sealed (arm, seed, rule) + checkpoint, no declared-not-run."""
        if self.lock is None or self.a.split != "TEST":
            return {"status": "NOT_APPLIED"}
        n = 0
        for e in self.entries:
            if e["arm"] not in P2.P2_ARMS:
                continue
            for dnr in self.lock["declared_not_run"]:
                if dnr["arm"] == e["arm"] and dnr["seed"] in (None, e["seed"]):
                    raise IndexRefused(f"{e['arm']} s{e['seed']} is declared not run in P2_METHODS_LOCK (S4)")
            k = (e["arm"], e["seed"], e["rule"])
            s = self.lock["sealed"].get(k)
            if e["arm"] in P2.P2_DEVICE_ARMS:
                ok = s is not None and s["base_checkpoint_sha256"] == (e.get("p2_base") or {}).get("checkpoint_sha256")
            else:
                ok = s is not None and s["checkpoint_sha256"] == e.get("checkpoint_sha256")
            if not ok:
                raise IndexRefused(f"{k}: not sealed in P2_METHODS_LOCK with this checkpoint / base sha256 (S4)")
            n += 1
        return {"status": "APPLIED", "n_checked": n}

    # ------------------------------------------------------------------ build
    def build(self) -> dict:
        a = self.a
        if a.split not in RR.ROLE_MAP:
            raise IndexRefused(f"--split {a.split!r}")
        if self.out.exists() and any(self.out.iterdir()):
            raise IndexRefused(f"{self.out} exists and is not empty (O_EXCL; use a new --out)")
        try:
            P2.assert_frozen_library()
        except RR.AnaRefused as e:
            raise IndexRefused(str(e)) from None
        if a.p2_lock:
            try:
                self.lock = P2.check_p2_lock(a.p2_lock)
            except RR.AnaRefused as e:
                raise IndexRefused(str(e)) from None
        self.out.mkdir(parents=True, exist_ok=True)
        if a.device_class == RR.PINNED_CLASS and not a.pin:
            raise IndexRefused(f"--device-class {RR.PINNED_CLASS} needs --pin <EVAL_DEVICE_PIN_A40.json>")
        if a.pin:
            pin = ABI._open_json(Path(a.pin), pin=True)
            if pin.get("device_class", a.device_class) != a.device_class or not pin.get("self_sha256"):
                raise IndexRefused(f"--pin {a.pin}: device_class {pin.get('device_class')!r} / self_sha256 missing")
            self.pin = {"file_sha256": ABI.sha_file(Path(a.pin)), "self_sha256": pin["self_sha256"]}
        with P2.p2_profile():
            try:
                self.phase1()
                self.reeval()
                self.p2_fl()
                self.p2_dev()
                self.records_files()
                counts = self.check_entries()
                lchk = self.lock_checks()
            except Exception:
                self.cleanup()
                raise
            order = {arm: i for i, arm in enumerate(RR.ARMS)}
        ents = sorted(self.entries, key=lambda e: (order[e["arm"]], e["seed"] or 0, e["rule"]))
        ev, tr = RR.ROLE_MAP[a.split]["MAIN"]
        idx = {"format": RR.INDEX_FORMAT, "study": P2.P2_STUDY, "split": a.split, "eval_device_class": a.device_class,
               "device_pin": self.pin, "roles": {"MAIN": {"eval_manifest_id": ev, "train_manifest_id": tr}},
               "entries": ents, "records": self.records, "phase1_block": self.phase1_rec, "p2_lock_checks": lchk,
               "p2_lock_file_sha256": None if self.lock is None else self.lock["file_sha256"],
               "builder": {"script": "tools/p2_block_index.py", "sha256": ABI.sha_file(Path(__file__).resolve()),
                           "p2_report_sha256": P2.script_sha256(), "ana_block_index_sha256": P2.ABI_SHA256,
                           "retention_report_sha256": P2.RR_SHA256,
                           "reads": "LAUNCH / RUN_SUMMARY / STATUS / REEVAL_RECORD + the Phase-1 BLOCK_INDEX and the P2 "
                                    "lock only; evidence files and records copied byte for byte, never parsed"},
               "test_registry_ext": ({"path": str(Path(a.registry_ext).resolve()),
                                      "file_sha256": ABI.sha_file(Path(a.registry_ext))} if a.registry_ext else None),
               "sources": dict(sorted(self.sources.items())), "excluded": self.excluded}
        idx = json.loads(json.dumps(idx))
        idx[RR.SELF_KEY] = RR._canon_indent(idx)
        text = json.dumps(idx, indent=1, sort_keys=True, ensure_ascii=True, allow_nan=False) + "\n"
        fd = os.open(str(self.out / RR.INDEX_NAME), os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0),
                     0o644)
        with os.fdopen(fd, "wb") as f:
            f.write(text.encode("utf-8"))
        return dict(counts, n_records=len(self.records), n_files=len(self.sources),
                    n_phase1_reused=sum(1 for e in ents if e.get("phase1_reused")), index_self_sha256=idx[RR.SELF_KEY])


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--split", required=True, choices=("VALIDATION", "TEST"))
    ap.add_argument("--device-class", required=True)
    ap.add_argument("--pin", default=None)
    ap.add_argument("--phase1-block", required=True)
    for name in ("reeval", "p2-fl", "p2-dev", "records"):
        ap.add_argument(f"--{name}", nargs="*", default=[])
    ap.add_argument("--p2-lock", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--registry-ext", default=None)
    ap.add_argument("--allow-single-seed", nargs="*", default=[])
    args = ap.parse_args(argv)
    args.allow_single_seed = list(P2.P2_SINGLE_SEED) + list(args.allow_single_seed or [])
    return args


def main(argv=None, say=print) -> int:
    args = parse_args(argv)
    try:
        counts = P2Builder(args).build()
    except IndexRefused as e:
        say(f"P2 INDEX REFUSED: {e}")
        return REFUSED
    except Exception as e:
        import traceback
        say(f"P2 INDEX ERROR: {type(e).__name__}: {e}\n{traceback.format_exc()}")
        return ERROR
    say("P2 INDEX: written " + json.dumps(counts, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
