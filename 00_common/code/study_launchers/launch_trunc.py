"""v4int.launch_trunc — controlled cold start on the truncated-history panels (DATA-07; PREG1_AMENDMENT_06B §6.4;
EVALUATION_CONTRACT_V4 §11.5-§11.7).

  T41 (CPU; the gate of everything below):
    python -B code/v4int/launch_trunc.py --study T41 --panel WARM_TRUNCATION --run-dir <R>/T41_WARM_TRUNCATION
  Study A — CONTEXT_ONLY (E11.8), a locked checkpoint through the frozen T3 kernel:
    python -B code/v4int/launch_trunc.py --study A --arm C_BIG|C_SMALL|FA|DP --seed S --gpu g --run-dir <R>/SA_<arm>_s<S> \
        --ckpt <checkpoint> --t41-receipt <R>/T41_WARM_TRUNCATION/T41_RECEIPT.json [--panel WARM_TRUNCATION|COLD_H]
  Study B — FEW_SHOT_ADAPTATION (E11.9) on the COLD_H panel, own support only:
    python -B code/v4int/launch_trunc.py --study B --arm FT_C|LO --seed S --gpu g --run-dir <R>/SB_<arm>_s<S> \
        --selection <UNSEEN_VAL SELECTION.json of the arm> [--base <C_SMALL ckpt>] --t41-receipt <COLD_H T41 receipt>
  Events-to-90/95% (E11.10; CPU, pure), Study A or Study B rows (GATE-B B4) against the C_BIG Study A rows of the SAME
  panel and seed:
    python -B code/v4int/launch_trunc.py --study E2X --run-dir <R>/E2X_s<S> --rows <R>/SA_<arm>_s<S>/RESULT_ROWS.json \
        [--rows <R>/SB_<arm>_s<S>/RESULT_ROWS.json] --cref-rows <R>/SA_C_BIG_s<S>/RESULT_ROWS.json
  Study B rows have no FULL cell (M_full_history = None) and only the UNSEEN_VAL-selected device cell enters; a cell whose
  C_ref row is missing reports NO_C_REF (never "not reached").

Features: v4trunc (E11.4): for a query q and budget h the builder serves exactly what the clean builder produces for a
user whose observed history is H_h(q) followed by e_q — LOCAL_EVENT unchanged, LOCAL_HISTORY recomputed from the
window only, SHARED frozen, the frozen normalizers, the PARTIAL channels of v4trunc.rules (recorded in every launch
file and row). Every Study A / B launch REFUSES unless a T41 receipt of the same panel exists whose builder code sha256
equals the running v4trunc code, whose golden replay passed (h = full history reproduces production bitwise, apart
from the declared PARTIAL cells) and whose leak / deletion controls held. A SMOKE receipt gates only a --smoke launch;
the receipt is bound to the verified panel (content_sha256; n_compared == its distinct query decisions) (REV-ANA-01
B-F1). The registered panel is re-derived from the loaded event table (§11.2) and must equal it (B-F2).
Cells: every registered sub-panel / budget of the panel — FIXED_50 h in {0,1,5,10,20,50}, FIXED_20 h in {0,1,5,10,20}
(the same query q_u for every h), ALL_AVAILABLE(h) (queries differ by h; reported separately) — plus the FULL view
(the full available prior history): one per FIXED sub-panel (its fixed queries) and one per ALL_AVAILABLE h (that h's
queries) — the C_ref of E11.10 when the arm is the central C. Study A arms evaluate one locked checkpoint
(C_BIG / C_SMALL: v4train central checkpoints; FA / DP: fl.checkpoint global models) with p = 0 for every user
(unseen users; no personal component, E11.6 / E11.8). Study B arms: FT_C (fl.finetune on the C_SMALL base) and LO
(fl.local_only from theta0), the ONE registered recipe of the arm's UNSEEN_VAL SELECTION.json (launch_device's
selection; LO runs its 3 pass budgets and marks the selected one), per client its own E11.3 support at that h
(support_batch), causality cutoff = decision_ts(q) (fl.finetune.check_support_causal refuses, never filters), h = 0 or
a zero-label support scores the start state; evaluated per client through launch_device.execute (the frozen T3 kernel,
one full-manifest artifact per cell, the frozen evaluator, RESULT_ROWS format).
Rows: the central RESULT_ROWS format (v4eval.rows.rows_from_result) + per-h labels (study, panel, panel_type, h,
history FULL | H_h, budget_queries_identical_across_h, partial_channels) and comparison_status = row_status
COLD_START_STUDY_A / COLD_START_STUDY_B (never pooled with MAIN rows); WARM_TRUNCATION rows are labelled
WARM_TRUNCATION_STUDY_A in both fields, never cold start (B-F3). Study A rows carry the cell's OOV-flagged query count.
Eligibility counts per sub-panel and stratum are written to ELIGIBILITY.json (reported, never used to select).
Per-user evidence (06L-C §C10 persistence CIs; DEV-FTPRE): every Study A / B run dir also gets TRUNC_PER_USER.npz in
exactly the format tools/retention_report.py reads (load_trunc_per_user, PPSI_V4_TRUNC_PER_USER_1): array `format` =
the format string, and per cell with an OK primary row a pair `u__<key>` (the sorted unique int64 eval_user_key of the
cell's E2E users) / `m__<key>` (their per-user macro-E2E MRR@20, float64, same order), key = trunc_cell_key(method,
seed, trunc_panel, trunc_panel_type, trunc_query_h, trunc_full_history) of the cell's primary row (Study B: the
UNSEEN_VAL-selected device cell only, the rows the report keeps). The per-user values are v4eval.metrics' own
per-user means of the frozen mrr@20 credit over the manifest's eval_mask rows (per_user), computed from the cell's rank
array, so their mean reproduces the row value (checked to 1e-9 before anything is written; else refused). The .npz is
written deterministically (sorted members, stored, fixed zip timestamps: byte-identical on a rerun) and its sha256 is
recorded in STATUS.json and RESULT_ROWS.json (`per_user`). Study A also keeps cells/<cell>.per_user.npz (sha in the
cell's rows.json) so a --resume leg reuses finished cells; a pre-DEV-FTPRE finished cell is recomputed from its saved
cells/<cell>.ranks_int32.npy. tools/ana_block_index.py copies TRUNC_PER_USER.npz beside RESULT_ROWS.json.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Callable, Mapping, Optional, Sequence

HERE = Path(__file__).resolve().parent
CODE = HERE.parent
V4 = CODE.parent
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))
sys.dont_write_bytecode = True

import numpy as np  # noqa: E402

from v4int.launch_central import G1_GATE, LaunchRefused, _json_safe, _next_launch_path, gpu_checks, utc  # noqa: E402

STUDIES = ("T41", "A", "B", "E2X")
ARMS_A = ("C_BIG", "C_SMALL", "FA", "DP")
ARMS_B = ("FT_C", "LO")
PANELS = {"WARM_TRUNCATION": "WARM_TRUNCATION_16384", "COLD_H": "COLD_H_16384"}
STUDY_PANELS = {"A": ("WARM_TRUNCATION", "COLD_H"), "B": ("COLD_H",), "T41": ("WARM_TRUNCATION", "COLD_H")}
DEFAULT_PANEL = {"A": "WARM_TRUNCATION", "B": "COLD_H", "T41": "WARM_TRUNCATION"}
CKPT_KIND = {"C_BIG": "central", "C_SMALL": "central", "FA": "fl", "DP": "fl"}
MODEL_RECIPE = {"C_BIG": "TUNED_02B", "C_SMALL": "SMALL", "FA": "SMALL", "DP": "SMALL", "FT_C": "SMALL", "LO": "SMALL"}
EVAL_METHOD = {"C_BIG": "C", "C_SMALL": "C", "FA": "FA", "DP": "FA"}      # the v4eval adapter-path method code
PARADIGM_A = {"C_BIG": "CENTRAL", "C_SMALL": "CENTRAL", "FA": "FEDERATED_AVERAGING", "DP": "FEDERATED_DP"}
PRIMARY = ("E2E", "macro", "mrr@20")
XS = (0.90, 0.95)
RECEIPT_FORMAT = "PPSI_V4_T41_RECEIPT_1"
FAMILY = "SASREC"
DELETION_SAMPLE = 512


# ------------------------------------------------------------------------------------------------ pure helpers
def v4trunc_code_sha256() -> str:
    from v4data.common import code_tree_sha256
    root = CODE / "v4trunc"
    return code_tree_sha256([p for p in root.glob("*.py")], root)


def canon_sha(obj: Mapping, field: str = "self_sha256") -> str:
    body = {k: v for k, v in obj.items() if k != field}
    return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":"), allow_nan=False,
                                     default=str).encode()).hexdigest()


def study_cells(subpanels: Sequence[tuple]) -> list:
    """Evaluation cells [(panel_type, h, full_history)] in the registered order: every (panel_type, h) at H_h, then
    FULL once per FIXED sub-panel (its identical queries; carried by the largest budget) and once per ALL_AVAILABLE h."""
    cells = [(p, int(h), False) for p, h in subpanels]
    fixed_done = set()
    for p, h in subpanels:
        if p == "ALL_AVAILABLE":
            cells.append((p, int(h), True))
        elif p not in fixed_done:
            hs = [int(x) for q, x in subpanels if q == p]
            cells.append((p, max(hs), True))
            fixed_done.add(p)
    return cells


def cell_name(panel_type: str, h: int, full: bool) -> str:
    if full:
        return f"{panel_type}_FULL" if panel_type != "ALL_AVAILABLE" else f"ALL_AVAILABLE_h{int(h)}_FULL"
    return f"{panel_type}_h{int(h)}"


NO_C_REF = "NO_C_REF"        # GATE-B B4 (review 11 [10-R4]): the C_ref row is missing -> undecidable, never "not reached"


def events_to_x(values_by_h: Mapping[int, float], cref, xs: Sequence[float] = XS) -> dict:
    """E11.10: the smallest REGISTERED h with M(h) >= X * C_ref, per X; None if no registered h reaches it (no
    interpolation, no extrapolation). `cref` is one value or {h: value} (ALL_AVAILABLE: C_ref per h). A missing C_ref
    (None, or an h absent from the mapping) before the first hit gives NO_C_REF (GATE-B B4)."""
    out = {}
    hs = sorted(int(h) for h in values_by_h)
    for x in xs:
        hit = None
        for h in hs:
            c = cref.get(h) if isinstance(cref, Mapping) else cref
            if c is None:
                hit = NO_C_REF
                break
            if float(values_by_h[h]) >= float(x) * float(c):
                hit = h
                break
        out[f"{int(round(x * 100))}"] = hit
    return out


def primary_rows(rows: Sequence[Mapping]) -> list:
    return [r for r in rows if r.get("target_denominator") == PRIMARY[0] and r.get("aggregation") == PRIMARY[1]
            and f"{r.get('metric')}@{r.get('k')}" == PRIMARY[2]]


CREF_ARM = "C_BIG"          # E11.10 / PREG1_AMENDMENT_06A §2: C_ref = the central C (C_BIG), PRACTICAL_BEST, Study A


def study_of_rows(rows: Sequence[Mapping]) -> str:
    """'A' or 'B' (trunc_study of every row; a row without it counts as Study A, the pre-B4 layout); mixed refuses."""
    studies = {r.get("trunc_study") or "A" for r in rows}
    if len(studies) != 1 or not studies <= {"A", "B"}:
        raise LaunchRefused(f"events-to-X: the M rows mix or carry unknown studies {sorted(map(str, studies))}")
    return studies.pop()


def selected_study_b_rows(rows: Sequence[Mapping]) -> list:
    """Study B evidence rows: the UNSEEN_VAL-selected device cell only (device_selected True, not LO_CELL_NOT_SELECTED;
    06E §1), exactly one primary row per (panel_type, h) (refused otherwise); no FULL cell may appear (E11.9)."""
    from v4int.launch_device import LO_NOT_SELECTED
    out = [r for r in rows if r.get("device_selected", True) is True and r.get("row_status") != LO_NOT_SELECTED]
    if any(r.get("trunc_full_history") for r in out):
        raise LaunchRefused("events-to-X: a Study B row claims a FULL-history cell (Study B has H_h cells only)")
    seen = {}
    for r in out:
        k = (r.get("trunc_panel_type"), int(r.get("trunc_query_h")))
        if k in seen:
            raise LaunchRefused(f"events-to-X: two selected Study B rows for cell {k} (one selected cell per h)")
        seen[k] = r
    return out


def events_to_x_table(rows_m: Sequence[Mapping], rows_cref: Sequence[Mapping], xs: Sequence[float] = XS,
                      cref_arm: str = CREF_ARM) -> dict:
    """events-to-X per sub-panel from Study A or Study B rows of arm M and the Study A rows of the C_ref arm (same seed,
    same panel). REV-ANA-01 B-F7: every C_ref row must be a Study A row of the C_ref arm (method == C_BIG); the per-cell
    count of OOV-flagged queries (builder audit, PARTIAL_OOV_IDENTITY) is reported next to each value (None if not
    recorded). GATE-B B4: Study B rows (FT_C / LO on COLD_H; no FULL cell) enter with their selected cell only,
    M_full_history = None; a C_ref cell missing from the C_ref rows gives C_ref None and NO_C_REF."""
    pm, pc = primary_rows(rows_m), primary_rows(rows_cref)
    study_m = study_of_rows(pm) if pm else "A"
    if study_m == "B":
        pm = selected_study_b_rows(pm)
    if not pc:
        raise LaunchRefused("events-to-X: the C_ref rows carry no primary (E2E macro mrr@20) row")
    bad = sorted({(r.get("method"), r.get("trunc_study")) for r in pc
                  if r.get("method") != cref_arm or r.get("trunc_study") != "A"}, key=str)
    if bad:
        raise LaunchRefused(f"events-to-X: C_ref rows must be Study A rows of arm {cref_arm} (E11.10), got "
                            f"(method, study) {bad}")
    seeds = {r.get("seed") for r in pm} | {r.get("seed") for r in pc}
    panels = {r.get("trunc_panel") for r in pm} | {r.get("trunc_panel") for r in pc}
    if len(seeds) != 1 or len(panels) != 1:
        raise LaunchRefused(f"events-to-X needs one seed and one panel, got seeds {seeds} panels {panels}")
    sel = lambda rows, p, h, full: [r for r in rows if r.get("trunc_panel_type") == p  # noqa: E731
                                    and int(r.get("trunc_query_h")) == int(h) and bool(r.get("trunc_full_history")) == full]
    val = lambda rows, p, h, full: [float(r["value"]) for r in sel(rows, p, h, full)]  # noqa: E731
    oov = lambda rows, p, h, full: (sel(rows, p, h, full) or [{}])[0].get("trunc_oov_flagged_queries")  # noqa: E731
    first = lambda xs_: xs_[0] if xs_ else None  # noqa: E731   (a missing cell -> None; B4: never an IndexError)
    out = {}
    for p in ("FIXED_50", "FIXED_20", "ALL_AVAILABLE"):
        hs = sorted({int(r["trunc_query_h"]) for r in pm if r.get("trunc_panel_type") == p
                     and not r.get("trunc_full_history")})
        if not hs:
            continue
        m_h = {h: val(pm, p, h, False)[0] for h in hs}
        oov_h = {h: oov(pm, p, h, False) for h in hs}
        if p == "ALL_AVAILABLE":
            cref = {h: first(val(pc, p, h, True)) for h in hs}
            m_full = {h: first(val(pm, p, h, True)) for h in hs}
            oov_full = {h: oov(pm, p, h, True) for h in hs}
            oov_cref = {h: oov(pc, p, h, True) for h in hs}
        else:
            hqs = [int(r["trunc_query_h"]) for r in pc if r.get("trunc_panel_type") == p and r.get("trunc_full_history")]
            hq = max(hqs) if hqs else None
            cref = None if hq is None else first(val(pc, p, hq, True))
            m_full = None if hq is None else first(val(pm, p, hq, True))
            oov_full = None if hq is None else oov(pm, p, hq, True)
            oov_cref = None if hq is None else oov(pc, p, hq, True)
        out[p] = {"M_by_h": m_h, "C_ref": cref, "M_full_history": m_full, "events_to_X": events_to_x(m_h, cref, xs),
                  "C_ref_arm": cref_arm,
                  "oov_flagged_queries": {"M_by_h": oov_h, "M_full_history": oov_full, "C_ref": oov_cref},
                  "rule": "smallest registered h with M(h) >= X*C_ref; reported only if reached (E11.10)"}
        if study_m == "B":                                  # GATE-B B4: labelled, only for the Study B tables
            out[p].update(study="B", M_full_history_note="Study B has H_h cells only (no FULL cell, E11.9)",
                          cells="UNSEEN_VAL-selected device cell only (06E §1)")
        if cref is None or (isinstance(cref, Mapping) and any(v is None for v in cref.values())):
            out[p]["C_ref_status"] = NO_C_REF
    return out


def trunc_device_spec(arm: str, *, seed: int, peak_lr: float, panel_id: str, panel_type: str, h: int,
                      start_digest: str, selection: Optional[Mapping], fused: Optional[bool]) -> dict:
    """Study B spec in launch_device.execute's format: FT applies the ONE selected UNSEEN_VAL cell; LO runs its 3 pass
    budgets and marks the selected one (a selection is REQUIRED for both: the E11.9 frozen recipe)."""
    from v4int.launch_device import KNOBS, SELECTION_ROLE, grid_cells
    if arm not in ARMS_B:
        raise LaunchRefused(f"Study B arm {arm} is not one of {ARMS_B}")
    if selection is None:
        raise LaunchRefused("Study B needs --selection (the arm's UNSEEN_VAL SELECTION.json: the frozen recipe, E11.9)")
    if selection.get("arm") != arm or selection.get("selection_role") != SELECTION_ROLE:
        raise LaunchRefused(f"--selection is for {selection.get('arm')} / {selection.get('selection_role')}, not "
                            f"{arm} / {SELECTION_ROLE}")
    cells = grid_cells(arm)
    if selection["cell"] not in {c["cell"] for c in cells}:
        raise LaunchRefused(f"selection cell {selection['cell']} is not a cell of the {arm} grid")
    if arm != "LO":
        cells = [c for c in cells if c["cell"] == selection["cell"]]
    return {"arm": arm, "seed": int(seed), "peak_lr": float(peak_lr), "role": "COLD_H", "select": False,
            "selected_cell": selection["cell"], "cells": cells, "start_digest": start_digest, "fused": fused,
            "knobs": {k: list(v) if isinstance(v, tuple) else v for k, v in KNOBS.items()},
            "selection_substitute": None, "study": "B", "panel": panel_id, "panel_type": panel_type, "h": int(h),
            "selection_self_sha256": selection.get("self_sha256")}


def panel_query_decision_count(pm) -> int:
    """The number of DISTINCT query decisions of a (verified) panel: exactly what the T41 golden replay compares."""
    return int(np.unique(np.asarray(pm.queries["query_decision_id"], dtype=np.int64)).size)


def check_t41_receipt(path: Path, panel_id: str, code_sha: Optional[str] = None, *, exact: bool = False,
                      smoke: bool = False, pm=None) -> dict:
    """Refuse a Study A / B launch unless a verified, PASSING T41 receipt of this panel exists for this v4trunc code.
    exact=True (--require-exact-t41) also refuses any admitted PARTIAL_OOV_IDENTITY cell in the golden replay.
    REV-ANA-01 B-F1: a SMOKE receipt (`--buckets`: smoke_buckets / label_smoke SMOKE_NOT_EVIDENCE) gates only a
    `--smoke` launch, never a production one. With `pm` (the verified panel of this launch) the receipt must be BOUND
    to it: panel_record.content_sha256 == the verified panel's registry content_sha256, and (production receipts)
    golden_replay.n_compared == the panel's distinct query-decision count and golden_replay.pass is true."""
    p = Path(path) if path is not None else None
    if p is None or not p.is_file():
        raise LaunchRefused(f"no T41 receipt at {path}: run --study T41 first (the E11.4 builder gate)")
    try:
        obj = json.loads(p.read_text(encoding="utf-8"))
    except ValueError as e:
        raise LaunchRefused(f"T41 receipt {p} is not strict JSON ({e})") from e
    if obj.get("format") != RECEIPT_FORMAT or obj.get("self_sha256") != canon_sha(obj):
        raise LaunchRefused(f"T41 receipt {p} is not a verified {RECEIPT_FORMAT} object")
    if obj.get("panel") != panel_id:
        raise LaunchRefused(f"T41 receipt {p} is for {obj.get('panel')}, not {panel_id}")
    want = code_sha or v4trunc_code_sha256()
    if obj.get("v4trunc_code_sha256") != want:
        raise LaunchRefused(f"T41 receipt {p} was made with v4trunc {obj.get('v4trunc_code_sha256')}, running {want}")
    if obj.get("status") != "PASS" or not all(bool(v) for v in (obj.get("gates") or {}).values()) \
            or not obj.get("gates"):
        raise LaunchRefused(f"T41 receipt {p} did not pass: status {obj.get('status')}, gates {obj.get('gates')}")
    receipt_is_smoke = bool(obj.get("smoke_buckets")) or obj.get("label_smoke") is not None
    if receipt_is_smoke and not smoke:
        raise LaunchRefused(f"T41 receipt {p} is a SMOKE receipt (buckets {obj.get('smoke_buckets')}, label "
                            f"{obj.get('label_smoke')}): it gates only a --smoke launch, never production (B-F1)")
    gold = obj.get("golden_replay") or {}
    admitted = int(gold.get("admitted_partial_rows", 0))
    if exact and admitted:
        raise LaunchRefused(f"T41 receipt {p}: {admitted} golden-replay rows differ in declared PARTIAL_OOV_IDENTITY "
                            "cells and --require-exact-t41 is set")
    bound = None
    if pm is not None:
        want_c = (getattr(pm, "record", None) or {}).get("content_sha256")
        got_c = (obj.get("panel_record") or {}).get("content_sha256")
        if not want_c or got_c != want_c:
            raise LaunchRefused(f"T41 receipt {p} is bound to panel content {got_c}, the verified panel of this launch "
                                f"is {want_c} (B-F1)")
        n_q = panel_query_decision_count(pm)
        if not receipt_is_smoke:
            if gold.get("pass") is not True or int(gold.get("n_compared", -1)) != n_q:
                raise LaunchRefused(f"T41 receipt {p}: golden replay compared {gold.get('n_compared')} decisions "
                                    f"(pass {gold.get('pass')}), the panel has {n_q} distinct query decisions (B-F1)")
        bound = {"panel_content_sha256": want_c, "panel_query_decisions": n_q,
                 "n_compared": gold.get("n_compared"), "receipt_is_smoke": receipt_is_smoke}
    return {"admitted_partial_rows": admitted, "path": str(p), "self_sha256": obj["self_sha256"],
            "file_sha256": hashlib.sha256(p.read_bytes()).hexdigest(), "receipt_is_smoke": receipt_is_smoke,
            "bound_to_panel": bound}


def trunc_labels(*, study: str, panel_id: str, label: str, panel_type: str, h: int, full: bool) -> dict:
    from v4trunc.rules import PARTIAL_CHANNELS
    return {"trunc_study": study, "trunc_panel": panel_id, "trunc_panel_label": label, "trunc_panel_type": panel_type,
            "trunc_query_h": int(h), "trunc_h": None if full else int(h), "trunc_full_history": bool(full),
            "trunc_history": "FULL" if full else "H_h",
            "trunc_queries_identical_across_h": panel_type in ("FIXED_50", "FIXED_20"),
            "trunc_feature_builder": "v4trunc E11.4", "trunc_partial_channels": sorted(PARTIAL_CHANNELS),
            "comparison_status": f"COLD_START_STUDY_{study}" if label != "WARM_TRUNCATION"
            else f"WARM_TRUNCATION_STUDY_{study}", "population_label": label}


def study_a_cell(adapter, provider, frozen, *, evaluator_sha: Optional[str], key_base: Mapping, labels: Mapping,
                 arm: str, seed: int, model_sha256: str, overrides: Mapping) -> tuple:
    """(result, rows): the locked checkpoint through the frozen T3 path (predict_adapter, one shared model, p = 0)."""
    from v4eval.evaluate import evaluate
    from v4eval.predict import predict_adapter
    from v4eval.rows import rows_from_result
    m = provider.manifest
    art = predict_adapter(adapter, provider.batches(int(frozen.eval_batch)), m, config=frozen,
                          model_sha256=model_sha256)
    res = evaluate(m, art, config=frozen, evaluator_code_sha256=evaluator_sha)
    kb = dict(key_base)
    kb["population_hash"] = overrides.get("trunc_population_hash", kb.get("population_hash"))
    kb["eval_manifest_id"] = m.manifest_id
    rows = rows_from_result(res, kb, method=EVAL_METHOD[arm], seed=int(seed), selection_rule=None, efe=None)
    oov = provider.oov_flag_counts() if hasattr(provider, "oov_flag_counts") else None
    out = []
    for r in rows:
        r = dict(r, **labels)
        # REV-ANA-01 B-F3: row_status follows the population label (E11.1a: WARM_TRUNCATION is never cold start)
        r.update(method=arm, method_code=arm, training_paradigm=PARADIGM_A[arm], local_personalization="NONE_P0",
                 row_status=labels["comparison_status"], c_matched_eligible=False, downstream_seed_allowed=False,
                 study_a_context_only=True, **overrides)
        if oov is not None:                                   # B-F7: OOV-flagged queries of this cell (reported)
            r.update(trunc_oov_flagged_queries=oov["any"], trunc_oov_flag_counts=oov)
        out.append(r)
    return res, out, art


def study_b_rows(result, key_base: Mapping, labels: Mapping, *, spec: Mapping, cell: Mapping, selected: bool,
                 down_bytes: int, overrides: Optional[Mapping] = None) -> list:
    """Study B rows: launch_device.device_rows (the device RESULT_ROWS format), then row_status = comparison_status =
    the study label (COLD_START_STUDY_B; E11.1a / REV-ANA-01 B-F3: a truncated-panel row is never DOWNSTREAM_ARM /
    DOWNSTREAM_RETENTION, so it cannot pool with MAIN device rows). device_rows' own LO_NOT_SELECTED / DESCRIPTIVE
    marking of the non-selected LO pass budgets (06E §1) is kept. The status is applied AFTER device_rows (passing it
    in `overrides` would collide with device_rows' own row_status keyword)."""
    from v4int.launch_device import LO_NOT_SELECTED, device_rows
    over = {k: v for k, v in (overrides or {}).items() if k not in ("row_status", "comparison_status")}
    rows = device_rows(result, key_base, labels, spec=spec, cell=cell, selected=selected, down_bytes=down_bytes,
                       overrides=over)
    status = labels["comparison_status"]
    for r in rows:
        if r.get("row_status") != LO_NOT_SELECTED:
            r.update(row_status=status, comparison_status=status)
    return rows


# ------------------------------------------------------------------------------------------------ per-user evidence
TRUNC_PER_USER_FORMAT = "PPSI_V4_TRUNC_PER_USER_1"     # = tools/retention_report.TRUNC_PER_USER_FORMAT (06L-C §C10)
TRUNC_PER_USER_NAME = "TRUNC_PER_USER.npz"             # = tools/ana_block_index.TRUNC_PER_USER_NAME
PER_USER_TOL = 1e-9                                    # = tools/retention_report.ROW_VALUE_TOL
_ZIP_DATE = (1980, 1, 1, 0, 0, 0)
PER_USER_RULE = ("06L-C §C10 per-user evidence: per cell, u__<key> = sorted unique eval_user_key of the E2E users, "
                 "m__<key> = their per-user macro-E2E MRR@20 (v4eval.metrics.per_user of the frozen mrr@20 credit of "
                 "the cell's rank array over eval_mask), key = trunc_cell_key of the cell's primary row")


def trunc_cell_key(arm: str, seed, panel: str, ptype: str, qh: int, full: bool) -> str:
    """= tools/retention_report.trunc_cell_key (the per-user evidence key of one DATA-07 cell)."""
    return f"{arm}__{seed}__{panel}__{ptype}__{int(qh)}__{'F' if full else 'H'}"


def cell_per_user(manifest, ranks) -> tuple:
    """(users int64 sorted unique, per-user MRR@20 float64) over the manifest's E2E (eval_mask) rows: exactly the
    per-user means v4eval.metrics.population_block averages into the macro value."""
    from v4eval.metrics import credits_from_ranks, per_user
    ranks = np.asarray(ranks)
    if ranks.shape != (manifest.n,):
        raise LaunchRefused(f"per-user evidence: {ranks.shape[0] if ranks.ndim else 0} ranks for {manifest.n} "
                            "manifest rows")
    k = int(PRIMARY[2].split("@")[1])
    credit = credits_from_ranks(ranks, (k,))[f"mrr@{k}"]
    keys, s, c = per_user(credit, manifest.eval_mask, manifest.eval_user_key)
    return np.asarray(keys, dtype=np.int64), np.asarray(s / c, dtype=np.float64)


def primary_cell_row(rows: Sequence[Mapping]) -> Optional[Mapping]:
    """The one primary (E2E macro mrr@20) row of a cell that tools/retention_report.trunc_curves keeps (a row carrying
    device_selected must have it True: Study B's selected cell), or None when its value is not OK (no evidence)."""
    pr = [r for r in primary_rows(rows) if "device_selected" not in r or r.get("device_selected") is True]
    if len(pr) != 1:
        raise LaunchRefused(f"per-user evidence: {len(pr)} kept primary rows in one cell (need exactly 1)")
    r = pr[0]
    if r.get("value") is None or r.get("value_status", "OK") != "OK":
        return None
    return r


def per_user_entry(rows: Sequence[Mapping], manifest, ranks) -> Optional[tuple]:
    """(key, users, values) of one evaluated cell, or None (no OK primary value / no E2E user). Refuses unless the
    per-user mean reproduces the row value within PER_USER_TOL (the report's own check)."""
    r = primary_cell_row(rows)
    if r is None:
        return None
    for f in ("trunc_panel", "trunc_panel_type", "trunc_query_h", "trunc_full_history"):
        if r.get(f) is None:
            raise LaunchRefused(f"per-user evidence: the primary row has no {f} (trunc_labels format)")
    arm = r.get("method") if r.get("method") is not None else r.get("arm")
    key = trunc_cell_key(arm, r.get("seed"), r["trunc_panel"], r["trunc_panel_type"], int(r["trunc_query_h"]),
                         bool(r["trunc_full_history"]))
    u, m = cell_per_user(manifest, ranks)
    if u.size == 0:
        return None
    if abs(float(m.mean()) - float(r["value"])) > PER_USER_TOL:
        raise LaunchRefused(f"per-user evidence {key}: the per-user mean {float(m.mean())!r} does not reproduce the "
                            f"row value {float(r['value'])!r}")
    return key, u, m


def add_per_user(cells: dict, entry: Optional[tuple]) -> None:
    if entry is None:
        return
    key, u, m = entry
    if key in cells:
        raise LaunchRefused(f"per-user evidence: cell {key} twice in one run")
    cells[key] = (u, m)


def write_npz_deterministic(path: Path, arrays: Mapping) -> str:
    """An .npz np.load reads (allow_pickle=False), byte-identical for identical arrays on any host: members sorted,
    stored (no compression), fixed timestamps / system / permissions; written atomically. Returns the file sha256."""
    import io
    import zipfile
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as zf:
        for name in sorted(arrays):
            buf = io.BytesIO()
            np.lib.format.write_array(buf, np.asarray(arrays[name]), allow_pickle=False)   # (0-d kept)
            zi = zipfile.ZipInfo(f"{name}.npy", date_time=_ZIP_DATE)
            zi.compress_type = zipfile.ZIP_STORED
            zi.create_system = 3
            zi.external_attr = 0o100644 << 16
            zf.writestr(zi, buf.getvalue())
    os.replace(tmp, path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def per_user_arrays(cells: Mapping) -> dict:
    arrays = {"format": np.array(TRUNC_PER_USER_FORMAT)}
    for key, (u, m) in cells.items():
        arrays[f"u__{key}"] = np.asarray(u, dtype=np.int64)
        arrays[f"m__{key}"] = np.asarray(m, dtype=np.float64)
    return arrays


def write_trunc_per_user(run_dir: Path, cells: Mapping) -> dict:
    """<run_dir>/TRUNC_PER_USER.npz (PPSI_V4_TRUNC_PER_USER_1) of every evaluated cell -> its record (sha256)."""
    sha = write_npz_deterministic(Path(run_dir) / TRUNC_PER_USER_NAME, per_user_arrays(cells))
    return {"file": TRUNC_PER_USER_NAME, "sha256": sha, "format": TRUNC_PER_USER_FORMAT, "n_cells": len(cells),
            "cells": sorted(cells), "rule": PER_USER_RULE}


def write_cell_per_user(path: Path, entry: Optional[tuple]) -> str:
    """Study A: one cell's per-user evidence (same format; no pair when the cell has no evidence) -> sha256."""
    return write_npz_deterministic(path, per_user_arrays({} if entry is None else {entry[0]: entry[1:]}))


def load_cell_per_user(path: Path, want_sha256: Optional[str]) -> tuple:
    """(found, entry) of a Study A cell file written by write_cell_per_user; found False when the file is absent, its
    sha256 is not the one recorded in the cell's rows.json, or none was recorded (a pre-DEV-FTPRE cell)."""
    p = Path(path)
    if not want_sha256 or not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest() != want_sha256:
        return False, None
    with np.load(p, allow_pickle=False) as z:
        if str(z["format"]) != TRUNC_PER_USER_FORMAT:
            return False, None
        us = [n for n in z.files if n.startswith("u__")]
        if not us:
            return True, None
        if len(us) != 1:
            raise LaunchRefused(f"{p}: a cell per-user file with {len(us)} cells")
        key = us[0][3:]
        return True, (key, z[us[0]].astype(np.int64), z["m__" + key].astype(np.float64))


def study_b_cell_per_user(cell_dir: Path, spec: Mapping, manifest, rows: Sequence[Mapping]) -> Optional[tuple]:
    """Study B: the per-user evidence of one finished cell = its UNSEEN_VAL-selected device cell (launch_device.finalize
    wrote cells/<selected>.ranks_int32.npy of the full-manifest artifact)."""
    cid = spec["selected_cell"]
    ranks = np.load(Path(cell_dir) / "cells" / f"{cid}.ranks_int32.npy")
    return per_user_entry([r for r in rows if r.get("device_cell") == cid], manifest, ranks)


# ------------------------------------------------------------------------------------------------ CLI
def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="PPSI V4 controlled cold start on the truncated-history panels (DATA-07)")
    ap.add_argument("--study", required=True, type=str.upper, choices=STUDIES)
    ap.add_argument("--arm", default=None, type=str.upper, choices=ARMS_A + ARMS_B)
    ap.add_argument("--panel", default=None, type=str.upper, choices=tuple(PANELS))
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--gpu", type=int, default=None)
    ap.add_argument("--run-dir", required=True, type=Path)
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--ckpt", type=Path, default=None, help="Study A: the locked checkpoint of the arm")
    ap.add_argument("--ckpt-sha256", default=None)
    ap.add_argument("--base", type=Path, default=None, help="Study B FT_C: the C_SMALL central checkpoint")
    ap.add_argument("--base-sha256", default=None)
    ap.add_argument("--selection", type=Path, default=None)
    ap.add_argument("--t41-receipt", type=Path, default=None)
    ap.add_argument("--require-exact-t41", action="store_true",
                    help="refuse unless the T41 golden replay admitted zero PARTIAL_OOV_IDENTITY cells")
    ap.add_argument("--recipe-id", default="02B_SASREC", help="Study A C_BIG: the tuned recipe id")
    ap.add_argument("--tuned-recipes", type=Path, default=None)
    ap.add_argument("--hp-registry", type=Path, default=None)
    ap.add_argument("--small-recipe", type=Path, default=None)
    ap.add_argument("--theta0-variants", type=Path, default=None)
    ap.add_argument("--rows", type=Path, action="append", default=None, help="E2X: Study A RESULT_ROWS of an arm")
    ap.add_argument("--cref-rows", type=Path, default=None, help="E2X: Study A RESULT_ROWS of the C_ref arm")
    ap.add_argument("--cells", default=None, help="comma list of cell names (default: every cell)")
    ap.add_argument("--n-shards", type=int, default=8)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--allow-shared-gpu", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--smoke-queries", type=int, default=64)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--n-threads", type=int, default=4)
    ap.add_argument("--manifest-dir", type=Path, default=None)
    ap.add_argument("--lock-path", type=Path, default=None)
    ap.add_argument("--manifest-hashes", type=Path, default=None)
    ap.add_argument("--header-list", type=Path, default=None)
    ap.add_argument("--transform-state", type=Path, default=None)
    ap.add_argument("--registry", type=Path, default=None, help="default contracts/manifest_registry/REGISTRY_V2.json")
    ap.add_argument("--ds-root", type=Path, default=None)
    ap.add_argument("--buckets", default=None, help="T41 smoke: comma list of buckets (default all 64)")
    return ap.parse_args(argv)


def preflight(args, *, runner=None, environ=None, gate_path: Path = G1_GATE) -> dict:
    """Torch-free refusals (+ the shared GPU checks for A / B)."""
    env = os.environ if environ is None else environ
    rec = {"utc": utc(), "argv": list(sys.argv), "host": platform.node(), "study": args.study, "arm": args.arm,
           "smoke": bool(args.smoke)}
    study = args.study
    panel = args.panel or DEFAULT_PANEL.get(study)
    if study in STUDY_PANELS and panel not in STUDY_PANELS[study]:
        raise LaunchRefused(f"--study {study} runs on {STUDY_PANELS[study]}, not {panel} (PREG1_AMENDMENT_06B §6.4)")
    rec["panel"] = panel
    if study == "E2X":
        if not args.rows or args.cref_rows is None:
            raise LaunchRefused("--study E2X needs --rows (one or more) and --cref-rows")
        return rec
    if study == "T41":
        if args.arm is not None or args.ckpt is not None:
            raise LaunchRefused("--study T41 is data-only: no --arm / --ckpt")
        return rec
    if args.seed is None:
        raise LaunchRefused(f"--study {study} needs --seed")
    if args.gpu is None or int(args.gpu) < 0:
        raise LaunchRefused("an explicit physical GPU id (--gpu >= 0) is required")
    if args.t41_receipt is None:
        raise LaunchRefused("--t41-receipt is required (the T41 golden-replay gate of the E11.4 builder)")
    if study == "A":
        if args.arm not in ARMS_A:
            raise LaunchRefused(f"Study A arms are {ARMS_A}")
        if args.ckpt is None:
            raise LaunchRefused(f"Study A {args.arm} needs --ckpt (a locked checkpoint)")
        if args.selection is not None or args.base is not None:
            raise LaunchRefused("Study A is CONTEXT_ONLY: no --selection / --base")
    if study == "B":
        if args.arm not in ARMS_B:
            raise LaunchRefused(f"Study B arms are {ARMS_B}")
        if args.selection is None:
            raise LaunchRefused("Study B needs --selection (the arm's UNSEEN_VAL SELECTION.json)")
        if args.arm == "FT_C" and args.base is None:
            raise LaunchRefused("FT_C needs --base <C_SMALL central checkpoint>")
        if args.arm == "LO" and args.base is not None:
            raise LaunchRefused("LO starts from theta0: no --base")
        if args.ckpt is not None:
            raise LaunchRefused("Study B takes --base (FT_C), not --ckpt")
    if args.smoke:
        if not 1 <= int(args.smoke_queries) <= 1024:
            raise LaunchRefused("--smoke-queries must be in [1, 1024]")
        rec["g1_gate"] = {"required": False, "reason": "smoke (SMOKE_NOT_EVIDENCE)"}
    else:
        if not Path(gate_path).is_file():
            raise LaunchRefused(f"non-smoke run refused: {gate_path} does not exist (G1 not approved)")
        rec["g1_gate"] = {"required": True, "path": str(gate_path),
                          "sha256": hashlib.sha256(Path(gate_path).read_bytes()).hexdigest()}
    if args.dry_run:
        rec["gpu_check"] = {"skipped": "dry-run"}
        return rec
    from v4sup.gpu import default_runner
    rec["gpu_check"] = gpu_checks(args, runner or default_runner, env)
    return rec


def write_json(path: Path, obj) -> None:
    from v4int.launch_device import write_json as wj
    wj(path, obj)


def _registry(args) -> Path:
    return Path(args.registry) if args.registry else V4 / "contracts" / "manifest_registry" / "REGISTRY_V2.json"


def load_panel_world(args, say: Callable) -> dict:
    """The verified panel, its users' event table, the builder and the panel-vs-events proof (fail-closed)."""
    from v4data.common import dataset_root
    from v4int.launch_central import data_paths
    from v4trunc import events as E
    from v4trunc.builder import TruncatedHistoryBuilder, load_normalizers
    from v4trunc.panel import check_panel_against_events, check_panel_rederived, eligibility, verify_panel
    panel = args.panel or DEFAULT_PANEL[args.study]
    pid = PANELS[panel]
    paths = data_paths(args)
    pm = verify_panel(pid, _registry(args), paths, root=V4, log=say)
    ds_root = Path(args.ds_root) if args.ds_root else dataset_root()
    buckets = [int(b) for b in args.buckets.split(",")] if args.buckets else None
    hexes = pm.users["client_key_hex"].tolist()
    if buckets is not None:                                    # smoke: only the users of these buckets
        tab = E.concat([E.load_bucket(ds_root, b, want_hex=hexes) for b in buckets])
    else:
        tab = E.load_users(ds_root, hexes, log=say)
    if buckets is not None:
        keep = set(tab.user_hex.tolist())
        pm = _restrict_panel(pm, keep)
    ev_checks = check_panel_against_events(pm, tab)
    # REV-ANA-01 B-F2: re-derive the panel (§11.2 earliest-q_u rule, every sub-panel / h of every loaded user) from
    # the loaded event table and refuse unless the (panel_type, h, client_key_hex, query_decision_id) sets are equal
    ev_checks.update(check_panel_rederived(pm, tab))
    bad = sorted(k for k, v in ev_checks.items() if v is not True)
    if bad:
        raise LaunchRefused(f"{pid}: the panel disagrees with the event data: {bad}")
    b = TruncatedHistoryBuilder(tab, normalizers=load_normalizers(ds_root),
                                state_sha256=pm.record["transform_state_self_sha256"])
    return {"panel": panel, "panel_id": pid, "pm": pm, "table": tab, "builder": b, "ds_root": ds_root,
            "panel_event_checks": ev_checks, "eligibility": eligibility(pm), "buckets": buckets}


def _restrict_panel(pm, keep_hex: set):
    """A panel view restricted to users present in the loaded buckets (T41 smoke only)."""
    from v4trunc.panel import PanelManifest
    hexes = pm.hex_of_key(pm.queries["eval_user_key"])
    qm = np.array([h in keep_hex for h in hexes.tolist()], dtype=bool)
    sh = pm.hex_of_key(pm.support["eval_user_key"])
    sm = np.array([h in keep_hex for h in sh.tolist()], dtype=bool)
    return PanelManifest(manifest_id=pm.manifest_id, header=pm.header, users=pm.users,
                         queries={k: v[qm] for k, v in pm.queries.items()},
                         support={k: v[sm] for k, v in pm.support.items()}, record=pm.record)


def run_t41(args, pre: dict, say: Callable = print) -> int:
    """T41 on the real panel (CPU): golden replay (h = FULL == production, bitwise) over every distinct query decision,
    deletion invariance on a deterministic sample of (q, h), the leak control, the panel-vs-events proof."""
    import v4train  # noqa: F401  (ppsi_env first)
    import ppsi_env
    snap = ppsi_env.verify_snapshot()
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    w = load_panel_world(args, say)
    obj = t41_receipt(w, pre=pre, snapshot_files=len(snap["files"]), say=say)
    name = "T41_RECEIPT_SMOKE.json" if w["buckets"] else "T41_RECEIPT.json"
    write_json(run_dir / name, obj)
    say(f"[T41] {w['panel_id']}: {obj['status']} ({name})")
    return 0 if obj["status"] == "PASS" else 1


def t41_receipt(w: Mapping, *, pre: Optional[Mapping] = None, snapshot_files: Optional[int] = None,
                say: Callable = print) -> dict:
    """The T41 receipt of a loaded panel world {panel_id, pm, table, builder, ds_root, buckets, panel_event_checks,
    eligibility} (self-hashed; status PASS iff every gate holds)."""
    from v4data.common import VALIDATION_SPLITS
    from v4trunc.golden import deletion_invariance, golden_replay
    from v4trunc.rules import rules_record
    t0 = time.time()
    pm, tab, b = w["pm"], w["table"], w["builder"]
    ids = np.unique(np.asarray(pm.queries["query_decision_id"], dtype=np.int64))
    say(f"[T41] golden replay over {ids.size} query decisions")
    gold = golden_replay(b, ids, splits=VALIDATION_SPLITS, kind="eval", ds_root=w["ds_root"], buckets=w["buckets"],
                         log=say)
    q = pm.queries
    cand = np.flatnonzero((np.asarray(q["h"]) > 0) & (np.asarray(q["h"]) < np.asarray(q["n_prior_events_available"])))
    rng = np.random.default_rng(20260928)
    pick = np.sort(rng.choice(cand, size=min(DELETION_SAMPLE, cand.size), replace=False)) if cand.size else cand
    items = [(tab.row_of(int(q["query_decision_id"][i])), int(q["h"][i])) for i in pick.tolist()]
    mk = lambda t2: type(b)(t2, normalizers=b.normalizers, state_sha256=b.state_sha256)  # noqa: E731
    dele = deletion_invariance(b, items, mk)
    leak = _leak_control(b, items)
    gates = {"golden_replay_bitwise_except_declared_partial": bool(gold["pass"]),
             "deletion_invariance": dele["n_checked"] > 0 and dele["n_changed"] == 0,
             "leak_control_detects_full_history_user_context": leak["n_items"] > 0 and leak["n_differ"] > 0,
             "panel_matches_event_data": all(bool(v) for v in w["panel_event_checks"].values())}
    obj = {"format": RECEIPT_FORMAT, "task": "DATA-07", "panel": w["panel_id"], "label": pm.label,
           "status": "PASS" if all(gates.values()) else "FAIL", "gates": gates, "golden_replay": gold,
           "deletion_invariance": dele, "leak_control": leak, "panel_event_checks": w["panel_event_checks"],
           "panel_record": pm.record, "eligibility": w["eligibility"], "rules": rules_record(),
           "v4trunc_code_sha256": v4trunc_code_sha256(), "snapshot_files_verified": snapshot_files,
           "event_table": {"users": tab.n_users, "rows": tab.n_rows},
           "smoke_buckets": w["buckets"], "label_smoke": "SMOKE_NOT_EVIDENCE" if w["buckets"] else None,
           "seconds": round(time.time() - t0, 1), "utc": utc(), "preflight": pre}
    obj["self_sha256"] = canon_sha(obj)
    return obj


def _leak_control(b, items) -> dict:
    """E11.7: a builder that feeds the FULL-history user context into the query row must produce different features
    from E11.4 on truncated queries (and the comparison must see it)."""
    from v4trunc.golden import batches_equal
    from v4trunc.builder import FULL
    n, differ = 0, 0
    for q_row, h in items[:128]:
        blk = b.block(int(q_row), int(h))
        full = b.block(int(q_row), FULL)
        leaky = type(blk)(**{**blk.__dict__, "raw": dict(blk.raw)})
        for c in full.raw:
            if c in ("log1p_prev_view_count", "log1p_prev_session_count", "recent_history_length", "has_history",
                     "recent_distinct_items", "recent_distinct_known_main_categories", "recent_category_match_share",
                     "recent_brand_match_share", "recent_mean_log_price", "current_minus_recent_mean_log_price",
                     "log1p_time_since_prev_user_event_s", "log1p_time_since_prev_session_last_event_s",
                     "has_history_mask", "recent_category_match_supported", "recent_brand_match_supported",
                     "recent_price_supported", "prev_user_event_available", "prev_session_available"):
                col = leaky.raw[c].copy()
                col[-1] = full.raw[c][-1]
                leaky.raw[c] = col
        n += 1
        differ += int(not batches_equal(b.batch_of_blocks([blk]), b.batch_of_blocks([leaky])))
    return {"n_items": n, "n_differ": differ,
            "rule": "full-history user_context substituted into the truncated block must change the served features"}


def _arm_recipe(arm: str, args) -> dict:
    """The arm's registered recipe: C_BIG = the tuned 02B recipe (TUNED_RECIPES_02B.json); every other arm = SMALL."""
    if MODEL_RECIPE[arm] == "TUNED_02B":
        from v4int.launch_central import _recipe_values
        return _recipe_values(argparse.Namespace(tuned_recipes=args.tuned_recipes, hp_registry=args.hp_registry,
                                                 smoke=args.smoke, recipe_id=args.recipe_id, peak_lr=None), FAMILY)
    from v4int.launch_fl import _small_recipe
    return _small_recipe(args)


def _harness(args, say: Callable, dev: str, recipe: Mapping):
    """The MAIN harness trainer of the arm's recipe (key base incl. topology, row labels, frozen T3 config, evaluator
    sha), as launch_device builds it."""
    import argparse as _ap
    from v4data.common import dataset_root
    from v4int.launch_central import (DETERMINISM_MODE, _theta0_for, bind_data_protocol_lock_v2,
                                      build_trainer_registry_view, data_paths, load_production_schema)
    from v4int.providers import StoreEvalProvider, StoreTrainProvider, comparison_key_base
    from v4int.smoke import SMOKE_RUN_PREFIX
    from v4train.provider import expected_manifests
    from v4train.recipe import RunConfig
    from v4train.trainer import CentralTrainer
    run_dir = Path(args.run_dir)
    run_id = args.run_id or f"{SMOKE_RUN_PREFIX if args.smoke else ''}TRUNC_{args.study}_{args.arm}_s{args.seed}"
    paths = data_paths(args)
    lockctx = bind_data_protocol_lock_v2(paths, smoke=bool(args.smoke))
    view = build_trainer_registry_view(lockctx["manifest_registry_path"], run_dir,
                                       registry_rel_path=lockctx["manifest_registry_rel_path"],
                                       registry_file_sha256=lockctx["manifest_registry_file_sha256"])
    reg_kw = dict(history_policy="USER_HISTORY_L50", paths=paths, log=say,
                  manifest_registry=lockctx["manifest_registry_path"], registry_root=V4)
    train_id, eval_id = expected_manifests("COHORT", "MAIN", registry=str(view["path"]))
    tp = StoreTrainProvider(train_id, **reg_kw)
    ep = StoreEvalProvider(eval_id, prefetch=0, **reg_kw)
    key = comparison_key_base(paths, tp, ep)
    theta0 = _theta0_for(_ap.Namespace(theta0_variants=args.theta0_variants, smoke=args.smoke),
                         lockctx["schema_id"], {"model_variant": recipe["model_variant"]}, int(args.seed))
    cfg = RunConfig(run_id=run_id, family=FAMILY, seed=int(args.seed), exposure_point="end",
                    cuda_visible_devices=str(int(args.gpu)), phase="MAIN", population="COHORT",
                    peak_lr=recipe["peak_lr"], log_every=50, checkpoint_every=0, key=key,
                    test_overrides=bool(args.smoke), determinism_mode=DETERMINISM_MODE,
                    manifest_registry=str(view["path"]), data_version=lockctx["data_version"],
                    schema_registry=str(lockctx["schema_registry_path"]), recipe_id=recipe["recipe_id"],
                    recipe_registry=recipe["recipe_registry"], model_variant=recipe["model_variant"],
                    dropout=tuple(recipe["dropout"]), schedule=recipe["schedule"], theta0_init_sha256=theta0)
    schema = load_production_schema(lockctx["schema_id"], lockctx["schema_self_sha256"], lockctx["schema_registry_path"])
    tr = CentralTrainer(cfg, schema, tp, ep, run_dir / "eval_harness", device=dev, v1_root=dataset_root())
    return {"tr": tr, "schema": schema, "recipe": recipe, "run_id": run_id, "paths": paths, "lockctx": lockctx}


def _model_for(args, h: dict, dev: str):
    """A fresh adapter of the harness recipe's topology (the arm's registered model variant)."""
    from v4data.common import dataset_root
    from v4models import build
    r = h["recipe"]
    ad = build(FAMILY, h["schema"], int(args.seed), dev, v1_root=dataset_root(), variant=r["model_variant"],
               dropout=tuple(r["dropout"])).adapter
    return ad, r


def run_a(args, pre: dict, env_rec: dict, say: Callable = print) -> int:
    import v4train
    import ppsi_env
    snap = ppsi_env.verify_snapshot()
    if v4train.LAUNCH_CUDA_VISIBLE_DEVICES != str(int(args.gpu)):
        raise LaunchRefused("the CUDA_VISIBLE_DEVICES export did not precede the v4train import")
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise LaunchRefused("expected exactly one visible CUDA device")
    torch.set_num_threads(int(args.n_threads))
    dev = "cuda:0"
    from v4eval.frozen_config import apply_runtime_flags, check_runtime_flags
    from v4int.launch_device import load_base
    from v4trunc.provider import TruncEvalProvider
    from v4trunc.rules import rules_record
    run_dir = Path(args.run_dir)
    panel_id = PANELS[args.panel or DEFAULT_PANEL["A"]]
    t41 = check_t41_receipt(args.t41_receipt, panel_id, exact=bool(args.require_exact_t41), smoke=bool(args.smoke))
    h = _harness(args, say, dev, _arm_recipe(args.arm, args))
    tr = h["tr"]
    adapter, recipe = _model_for(args, h, dev)
    ck = load_base(adapter, args.ckpt, CKPT_KIND[args.arm], args.ckpt_sha256)
    w = load_panel_world(args, say)
    pm, b = w["pm"], w["builder"]
    t41 = check_t41_receipt(args.t41_receipt, panel_id, exact=bool(args.require_exact_t41), smoke=bool(args.smoke),
                            pm=pm)                                  # B-F1: bound to the verified panel
    fz = tr.frozen
    apply_runtime_flags(fz)
    check_runtime_flags(fz, torch.device(dev))
    cells = study_cells(pm.subpanels())
    want = set(args.cells.split(",")) if args.cells else None
    spec = {"study": "A", "arm": args.arm, "seed": int(args.seed), "panel": panel_id, "ckpt_digest": ck["digest"],
            "ckpt_file_sha256": ck["file_sha256"], "cells": [cell_name(*c) for c in cells],
            "model_variant": recipe["model_variant"], "t41": t41["self_sha256"]}
    digest = canon_sha(spec, "_none_")
    write_json(_next_launch_path(run_dir), {"task": "DATA-07 launch_trunc", "study": "A", "spec": spec,
                                            "spec_sha256": digest, "t41_receipt": t41, "ckpt": ck | {"state": None},
                                            "rules": rules_record(), "eligibility": w["eligibility"],
                                            "preflight": pre, "environment": env_rec,
                                            "snapshot_files": len(snap["files"]), "torch": torch.__version__,
                                            "label": "SMOKE_NOT_EVIDENCE" if args.smoke else "PRODUCTION"})
    write_json(run_dir / "ELIGIBILITY.json", w["eligibility"])
    labels_base = tr._row_labels()
    all_rows = []
    per_user = {}                                                   # 06L-C §C10 (DEV-FTPRE): cell key -> (u, m)
    for (p, hh, full) in cells:
        name = cell_name(p, hh, full)
        if want is not None and name not in want:
            continue
        out = run_dir / "cells" / f"{name}.rows.json"
        pu_path = run_dir / "cells" / f"{name}.per_user.npz"
        if out.is_file() and args.resume:
            prev = json.loads(out.read_text(encoding="utf-8"))
            if prev.get("spec_sha256") == digest:
                all_rows += prev["rows"]
                found, ent = load_cell_per_user(pu_path, prev.get("per_user_sha256"))
                if not found:                                       # a pre-DEV-FTPRE cell: from its saved ranks
                    pv = TruncEvalProvider(b, pm, p, hh, full_history=full)
                    if args.smoke:
                        pv = _smoke_provider(pv, int(args.smoke_queries))
                    ent = per_user_entry(prev["rows"], pv.manifest,
                                         np.load(run_dir / "cells" / f"{name}.ranks_int32.npy"))
                add_per_user(per_user, ent)
                continue
        prov = TruncEvalProvider(b, pm, p, hh, full_history=full)
        if args.smoke:
            prov = _smoke_provider(prov, int(args.smoke_queries))
        msha = hashlib.sha256(f"STUDY_A|{args.arm}|{ck['digest']}|{name}".encode()).hexdigest()
        lab = dict(labels_base, **trunc_labels(study="A", panel_id=panel_id, label=pm.label, panel_type=p, h=hh,
                                               full=full))
        over = {"trunc_population_hash": pm.header.get("population_sha256_selection_key_v2") or
                pm.header.get("population_sha256"), "trunc_cell": name, "ckpt_file_sha256": ck["file_sha256"],
                "ckpt_digest": ck["digest"], "trunc_t41_receipt_self_sha256": t41["self_sha256"],
                "label_smoke": "SMOKE_NOT_EVIDENCE" if args.smoke else None}
        res, rows, art = study_a_cell(adapter, prov, fz, evaluator_sha=tr.evaluator_sha, key_base=tr.key_base,
                                      labels=lab, arm=args.arm, seed=int(args.seed), model_sha256=msha, overrides=over)
        ent = per_user_entry(rows, prov.manifest, art.ranks)       # checked against the row value before writing
        write_json(run_dir / "cells" / f"{name}.result.json", res)
        np.save(run_dir / "cells" / f"{name}.ranks_int32.npy", art.ranks.astype(np.int32))
        pu_sha = write_cell_per_user(pu_path, ent)
        write_json(out, {"spec_sha256": digest, "cell": name, "rows": rows, "per_user_sha256": pu_sha})
        add_per_user(per_user, ent)
        all_rows += rows
        say(f"[trunc A] {args.arm} {name}: done")
    pu_rec = write_trunc_per_user(run_dir, per_user)
    write_json(run_dir / "RESULT_ROWS.json", {"study": "A", "arm": args.arm, "seed": int(args.seed),
                                              "panel": panel_id, "spec_sha256": digest, "rows": all_rows,
                                              "per_user": pu_rec})
    write_json(run_dir / "STATUS.json", {"status": "SMOKE_DONE_NOT_EVIDENCE" if args.smoke else "DONE",
                                         "spec_sha256": digest, "utc": utc(), "per_user": pu_rec})
    return 0


def _smoke_provider(prov, n: int):
    """A provider restricted to its first n queries (smoke only; SMOKE_NOT_EVIDENCE)."""
    from v4eval.manifest import build_eval_manifest
    m = prov.manifest
    k = min(int(n), m.n)
    cols = {"decision_id": m.decision_id[:k], "eval_user_key": m.eval_user_key[:k], "eval_mask": m.eval_mask[:k],
            "target_class": m.target_class[:k], "target_oov": m.target_oov[:k], "censor_class": m.censor_class[:k]}
    cols.update({f: v[:k] for f, v in m.fields.items()})
    hdr = dict(m.header, manifest_id=m.manifest_id + "::SMOKE")
    prov.manifest = build_eval_manifest(hdr, cols)
    prov.event_rows, prov._ids, prov._target = prov.event_rows[:k], prov._ids[:k], prov._target[:k]
    if hasattr(prov, "_served"):
        prov._served, prov._oov = prov._served[:k], {f: v[:k] for f, v in prov._oov.items()}
    return prov


def run_b(args, pre: dict, env_rec: dict, say: Callable = print) -> int:
    import v4train
    import ppsi_env
    snap = ppsi_env.verify_snapshot()
    if v4train.LAUNCH_CUDA_VISIBLE_DEVICES != str(int(args.gpu)):
        raise LaunchRefused("the CUDA_VISIBLE_DEVICES export did not precede the v4train import")
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise LaunchRefused("expected exactly one visible CUDA device")
    torch.set_num_threads(int(args.n_threads))
    dev = torch.device("cuda:0")
    from fl.numerics import state_digest
    from v4eval.frozen_config import apply_runtime_flags, check_runtime_flags
    from v4int.launch_device import (_to, assert_arm_allowed, build_client_plans, execute, load_base,
                                     load_selection, parse_shards)
    from v4trunc.provider import TruncEvalProvider, support_batch
    from v4trunc.rules import rules_record
    panel_id = PANELS["COLD_H"]
    t41 = check_t41_receipt(args.t41_receipt, panel_id, exact=bool(args.require_exact_t41), smoke=bool(args.smoke))
    allowed = assert_arm_allowed(args.arm, FAMILY)
    selection = load_selection(args.selection)
    h = _harness(args, say, "cuda:0", _arm_recipe(args.arm, args))
    tr = h["tr"]
    worker, recipe = _model_for(args, h, "cuda:0")
    if args.arm == "LO":
        start = tr.adapter.broadcast_state(clone=True)
        start_rec = {"kind": "theta0", "init_sha256": tr.built.init_sha256}
    else:
        base = load_base(worker, args.base, "central", args.base_sha256)
        start, start_rec = base["state"], {k: v for k, v in base.items() if k != "state"}
    w = load_panel_world(args, say)
    pm, b = w["pm"], w["builder"]
    t41 = check_t41_receipt(args.t41_receipt, panel_id, exact=bool(args.require_exact_t41), smoke=bool(args.smoke),
                            pm=pm)                                  # B-F1: bound to the verified panel
    fz = tr.frozen
    apply_runtime_flags(fz)
    check_runtime_flags(fz, dev)
    run_dir = Path(args.run_dir)
    write_json(run_dir / "ELIGIBILITY.json", w["eligibility"])
    write_json(_next_launch_path(run_dir), {"task": "DATA-07 launch_trunc", "study": "B", "arm": args.arm,
                                            "seed": int(args.seed), "panel": panel_id, "t41_receipt": t41,
                                            "architecture_lock": allowed, "selection": {
                                                "path": str(args.selection), "self_sha256": selection["self_sha256"],
                                                "cell": selection["cell"]}, "start": start_rec,
                                            "rules": rules_record(), "preflight": pre, "environment": env_rec,
                                            "snapshot_files": len(snap["files"]), "torch": torch.__version__,
                                            "label": "SMOKE_NOT_EVIDENCE" if args.smoke else "PRODUCTION"})
    labels_base = tr._row_labels()
    down = 0 if args.arm == "LO" else int(worker.manifest.shared_bytes)
    want = set(args.cells.split(",")) if args.cells else None
    all_rows = []
    per_user = {}                                                   # 06L-C §C10 (DEV-FTPRE): cell key -> (u, m)
    for p, hh in pm.subpanels():
        name = cell_name(p, hh, False)
        if want is not None and name not in want:
            continue
        prov = TruncEvalProvider(b, pm, p, hh)
        if args.smoke:
            prov = _smoke_provider(prov, int(args.smoke_queries))
        m = prov.manifest
        spec = trunc_device_spec(args.arm, seed=int(args.seed), peak_lr=recipe["peak_lr"], panel_id=panel_id,
                                 panel_type=p, h=hh, start_digest=state_digest(start), selection=selection, fused=True)
        hexes = pm.hex_of_key(m.eval_user_key)
        qrow = prov.event_rows
        sup = {}
        for key, r in zip(hexes.tolist(), qrow.tolist()):
            sup[str(key)] = np.array([int(r)], dtype=np.int64)          # the query row stands for the support set
        plans = build_client_plans(m, hexes, row_decision_ts=b.table.meta["decision_ts_us"][qrow],
                                   support_pos_by_key=sup, warm=False)
        lab = dict(labels_base, **trunc_labels(study="B", panel_id=panel_id, label=pm.label, panel_type=p, h=hh,
                                               full=False))

        def get_support(plan, _h=hh):
            return support_batch(b, int(plan.support[0]), _h)

        def row_fn(res, cell, selected, _lab=lab, _spec=spec, _name=name, _m=m):
            return study_b_rows(res, dict(tr.key_base, eval_manifest_id=_m.manifest_id), _lab, spec=_spec, cell=cell,
                                selected=selected, down_bytes=down,
                                overrides={"trunc_cell": _name, "trunc_t41_receipt_self_sha256": t41["self_sha256"]})

        out = execute(spec=spec, run_dir=run_dir / "cells" / name, worker=worker, start_state=start, plans=plans,
                      get_support=_support_or_empty(get_support),
                      get_eval_batch=lambda sp, _p=prov: _to(_p.batch_at(sp), dev), manifest=m, frozen=fz,
                      row_fn=row_fn, evaluator_sha=tr.evaluator_sha, n_shards=int(args.n_shards),
                      resume=bool(args.resume), smoke_clients=None, say=say)
        if out.get("status") == "DONE":
            rr = json.loads((run_dir / "cells" / name / "RESULT_ROWS.json").read_text(encoding="utf-8"))["rows"]
            all_rows += rr
            add_per_user(per_user, study_b_cell_per_user(run_dir / "cells" / name, spec, m, rr))
    pu_rec = write_trunc_per_user(run_dir, per_user)
    write_json(run_dir / "RESULT_ROWS.json", {"study": "B", "arm": args.arm, "seed": int(args.seed), "panel": panel_id,
                                              "rows": all_rows, "per_user": pu_rec})
    write_json(run_dir / "STATUS.json", {"status": "SMOKE_DONE_NOT_EVIDENCE" if args.smoke else "DONE", "utc": utc(),
                                         "per_user": pu_rec})
    return 0


def _support_or_empty(fn: Callable) -> Callable:
    """launch_device.execute calls get_support only for plans with support; a zero-label support returns an empty
    example dict (valid_rows = 0 -> the start state is scored, recorded as no_support)."""
    import torch

    def g(plan):
        b = fn(plan)
        if b is None:
            return {"target_class": torch.zeros(0, dtype=torch.long)}
        return b
    return g


def run_e2x(args, pre: dict, say: Callable = print) -> int:
    cobj = json.loads(Path(args.cref_rows).read_text(encoding="utf-8"))
    if cobj.get("study") != "A" or cobj.get("arm") != CREF_ARM:
        raise LaunchRefused(f"--cref-rows {args.cref_rows} is study {cobj.get('study')} arm {cobj.get('arm')}; C_ref "
                            f"is Study A {CREF_ARM} (E11.10, B-F7)")
    rows_c = cobj["rows"]
    out = {"format": "PPSI_V4_EVENTS_TO_X_1", "cref_rows": str(args.cref_rows), "cref_arm": CREF_ARM, "arms": {}}
    for p in args.rows:
        obj = json.loads(Path(p).read_text(encoding="utf-8"))
        if obj.get("study", "A") not in ("A", "B"):
            raise LaunchRefused(f"--rows {p} is study {obj.get('study')!r}; events-to-X reads Study A / B rows only")
        if obj.get("study") == "B" and any(r.get("trunc_study") != "B" for r in obj["rows"]):
            raise LaunchRefused(f"--rows {p}: a Study B RESULT_ROWS file carries a row that is not trunc_study B")
        name = obj.get("arm", str(p))
        if name in out["arms"]:
            raise LaunchRefused(f"--rows: arm {name} given twice")
        out["arms"][name] = events_to_x_table(obj["rows"], rows_c)
    out["self_sha256"] = canon_sha(out)
    write_json(Path(args.run_dir) / "EVENTS_TO_X.json", out)
    say("[trunc E2X] EVENTS_TO_X.json written")
    return 0


def main(argv=None) -> int:
    from v4int.launch_central import export_environment
    args = parse_args(argv)
    try:
        pre = preflight(args)
        if args.dry_run:
            print(json.dumps(_json_safe(pre), indent=1))
            return 0
        if args.study == "E2X":
            return run_e2x(args, pre)
        if args.study == "T41":
            os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
            return run_t41(args, pre)
        env_rec = export_environment(int(args.gpu))
        return run_a(args, pre, env_rec) if args.study == "A" else run_b(args, pre, env_rec)
    except LaunchRefused as e:
        print(f"[launch_trunc] REFUSED: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
