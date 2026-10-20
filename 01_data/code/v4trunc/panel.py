"""v4trunc.panel — the COLD_H / WARM_TRUNCATION panel manifests (REGISTRY_V2 kind PANEL) for the truncated builder.

* `verify_panel` — the registry verification of a PANEL manifest. v4int.verify.verify_manifest_by_registry serves
  TRAIN / EVAL layouts only (its manifest_kind refuses the panel table set), so this is the same check list, built
  from the same helpers (verify_lock, verify_registry incl. the B1 superseded-lock route, verify_transform_state,
  strict_json, header_content_sha256, manifest_content_sha256, table_content_sha256 over reread_columns), applied to
  the panel tables panel_users / panel_queries / panel_support. Nothing is served from an unverified file.
* `derive_panel` — an INDEPENDENT implementation of the §11.2 panel rules from an event table (FIXED_50 / FIXED_20 /
  ALL_AVAILABLE(h), earliest qualifying E2E VALIDATION decision, one query per user per sub-panel and h) and of the
  §11.3 support rule; the real panel's queries and support must equal it (`check_panel_against_events`), and the
  registered query set must equal the re-derivation from the loaded event table (`check_panel_rederived`, B-F2).
* `subpanel_manifest` — the v4eval EvalManifest of one sub-panel and budget (one row per query, ascending decision id;
  eval_mask true; target_oov iff the query target is out of catalogue; strata fields panel_label / cold_stratum) for
  the frozen evaluator.
* `support_rows` — E11.3 (T40): e_j in H_h(q), target observed and uncensored with target_class >= 0 and loss_mask or
  eval_mask (the TRAIN loss rule, DATA-04 interpretation 6), target timestamp STRICTLY before decision_ts(q).
  So the transition into e_q and the query's own target are never support, h = 1 may give 0 labels, and a label that
  matures at / after the query never supports it.
Eligibility counts are reported per sub-panel and stratum and are never used to select anything (E11.2).
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Optional

import numpy as np

from .events import EventTable
from .rules import BUDGETS, FIXED_BUDGETS, SUBPANELS

PANEL_TABLES = ("panel_users", "panel_queries", "panel_support")
FIXED_MIN = {"FIXED_50": 50, "FIXED_20": 20}
PANEL_ROLES = ("COLD_H", "WARM_TRUNCATION")
LABEL_OF_ROLE = {"COLD_H": "COLD_START", "WARM_TRUNCATION": "WARM_TRUNCATION"}


class PanelError(AssertionError):
    """A panel manifest failed verification or disagrees with the event data: refused."""


@dataclass
class PanelManifest:
    manifest_id: str
    header: dict
    users: dict                  # eval_user_key, client_key_hex, selection_key, cold_stratum, fixed50/20_eligible
    queries: dict                # panel_type, h, eval_user_key, query_decision_id, ... (manifest row order)
    support: dict                # panel_type, h, eval_user_key, support_decision_id, support_target_ts_us, ...
    record: dict = field(default_factory=dict)

    @property
    def label(self) -> str:
        return str(self.header.get("panel_label", ""))

    @property
    def catalogue_K(self) -> int:
        return int(self.header["catalogue_K"])

    def subpanels(self) -> list:
        """(panel_type, h) pairs present, in the registered order."""
        have = set(zip(self.queries["panel_type"].tolist(), self.queries["h"].tolist()))
        out = []
        for p in SUBPANELS:
            for h in (FIXED_BUDGETS[p] if p in FIXED_BUDGETS else BUDGETS):
                if (p, int(h)) in have:
                    out.append((p, int(h)))
        return out

    def rows(self, panel_type: str, h: int) -> np.ndarray:
        """Query-table rows of one sub-panel and budget, ascending query_decision_id."""
        m = (self.queries["panel_type"] == panel_type) & (self.queries["h"] == int(h))
        r = np.flatnonzero(m)
        return r[np.argsort(self.queries["query_decision_id"][r], kind="stable")]

    def hex_of_key(self, euk: np.ndarray) -> np.ndarray:
        k = np.asarray(self.users["eval_user_key"], dtype=np.int64)
        if not np.array_equal(k, np.arange(k.size)):
            raise PanelError("panel_users.eval_user_key is not the dense 0..n-1 key")
        return np.asarray(self.users["client_key_hex"], dtype=object)[np.asarray(euk, dtype=np.int64)]


def _np_table(tbl) -> dict:
    out = {}
    for name in tbl.column_names:
        col = tbl[name]
        out[name] = np.asarray(col.to_pylist(), dtype=object) if str(col.type) in ("string", "large_string") \
            else col.to_numpy(zero_copy_only=False)
    return out


def verify_panel(manifest_id: str, registry_path, paths=None, *, root: Optional[Path] = None, log=None) -> PanelManifest:
    """Registry verification of a PANEL manifest (see the module docstring); returns its verified tables."""
    import pyarrow.parquet as pq
    from v4data.common import sha256_file, table_content_sha256
    from v4data.manifests import reread_columns
    from v4int.verify import (V4, DataPaths, header_content_sha256, manifest_content_sha256, refuse_test_manifest,
                              registry_statuses, strict_json, verify_lock, verify_registry, verify_transform_state)
    mid = str(manifest_id)
    refuse_test_manifest(mid)
    paths = paths or DataPaths.default()
    root = Path(root) if root is not None else V4
    say = log or (lambda *_: None)
    t0 = time.time()
    checks: dict = {}
    lock, lself, lfile = verify_lock(paths)
    reg, reg_file, reg_self, v1_ctx = verify_registry(registry_path, lself, lfile, lock=lock, root=root)
    ref_lock, ref_self, ref_file = v1_ctx if v1_ctx is not None else (lock, lself, lfile)
    entry = reg["manifests"].get(mid)
    if not isinstance(entry, dict):
        raise PanelError(f"{mid}: no entry in the manifest registry {registry_path}")
    checks["registry.kind_panel"] = entry.get("kind") == "PANEL"
    checks["registry.tables_are_panel_tables"] = set(entry.get("tables", {})) == set(PANEL_TABLES)
    checks["registry.role_is_panel_role"] = entry.get("role") in PANEL_ROLES
    ts_self, ts_file = verify_transform_state(paths, lock)
    rts = reg.get("transform_state") or {}
    checks["registry.transform_state_eq_lock"] = (rts.get("transform_state_sha256") == ts_self
                                                  and rts.get("file_sha256") == ts_file)
    _fail(mid, checks)
    tables, cols = {}, {}
    for t in PANEL_TABLES:
        e = entry["tables"][t]
        f = root / str(e.get("file", ""))
        if not f.is_file():
            raise PanelError(f"{mid}: table {t} missing from disk ({f})")
        fsha = sha256_file(f)
        checks[f"{t}.file_sha256_eq_registry"] = fsha == e.get("file_sha256")
        tbl = pq.read_table(f)
        checks[f"{t}.row_count_eq_registry"] = tbl.num_rows == int(e["row_count"])
        checks[f"{t}.table_content_sha256_eq_registry"] = table_content_sha256(reread_columns(tbl)) == \
            e["table_content_sha256"]
        tables[t] = {"file": f.name, "file_sha256": fsha, "row_count": int(e["row_count"]),
                     "table_content_sha256": e["table_content_sha256"]}
        cols[t] = _np_table(tbl)
        del tbl
    hp = root / str(entry.get("path", ""))
    if not hp.is_file():
        raise PanelError(f"{mid}: header file {hp} is missing")
    hsha = sha256_file(hp)
    header = strict_json(hp)
    refuse_test_manifest(mid, header=header)
    status = header.get("status")
    ref_code = (ref_lock.get("code") or {}).get("v4data_builder_code_sha256")
    ref_schema = (ref_lock.get("features_and_transforms") or {}).get("schema_self_sha256")
    ref_ver = (ref_lock.get("data_version") or {}).get("base_version_hash")
    ref_cand = ((ref_lock.get("population_and_manifests") or {}).get("manifest_hashes") or {}).get("candidate_order_hash")
    ref_K = (ref_lock.get("catalogue") or {}).get("K")
    checks.update({
        "registry.header_sha256": hsha == entry.get("header_sha256"),
        "header.manifest_id": header.get("manifest_id") == mid,
        "header.status_eq_registry_entry": status == entry.get("status", status),
        "header.status_allowed": status in registry_statuses(reg),
        "header.data_version_eq_registry": header.get("data_version") == reg.get("data_version"),
        "header.no_all_methods_lock": header.get("all_methods_lock_sha256") is None,
        "header.no_test_exposure_entry": header.get("test_exposure_register_entry_id") is None,
        "header.data_protocol_lock_sha256_eq_lock_self": header.get("data_protocol_lock_sha256") == ref_self,
        "header.data_protocol_lock_file_sha256_eq_lock_file": header.get("data_protocol_lock_file_sha256") == ref_file,
        "header.content_self_hash": header.get("header_content_sha256") == header_content_sha256(header),
        "registry.header_content_sha256": header.get("header_content_sha256") == entry.get(
            "header_content_sha256", header.get("header_content_sha256")),
        "header.tables_eq_registry": {x["table"]: {k: x[k] for k in ("row_count", "table_content_sha256", "file_sha256")}
                                      for x in header.get("tables", [])} == {
            t: {k: v[k] for k in ("row_count", "table_content_sha256", "file_sha256")} for t, v in entry["tables"].items()},
        "header.builder_code_sha256_eq_lock": bool(ref_code) and header.get("builder_code_sha256") == ref_code,
        "header.schema_self_sha256_eq_lock": bool(ref_schema) and header.get("feature_schema_self_sha256") == ref_schema,
        "header.dataset_version_eq_lock": bool(ref_ver) and header.get("dataset_version_hash") == ref_ver,
        "header.candidate_order_eq_lock": bool(ref_cand) and header.get("candidate_order_hash") == ref_cand,
        "header.catalogue_K_eq_lock": ref_K is not None and header.get("catalogue_K") == ref_K,
        "header.panel_label_eq_role": header.get("panel_label") == LABEL_OF_ROLE.get(entry.get("role")),
    })
    for k in ("ordered_decision_id_sha256", "eval_ordered_decision_id_sha256", "rankable_ordered_decision_id_sha256",
              "population_sha256", "n_users", "n_decisions", "n_eval_rows", "n_rankable_rows"):
        if k in entry:
            checks[f"header.{k}_eq_registry"] = header.get(k) == entry.get(k)
    try:
        checks["registry.content_sha256"] = manifest_content_sha256(header.get("tables", []), tables) == \
            entry.get("content_sha256")
    except Exception:                                           # noqa: BLE001 — any failure is a failed check
        checks["registry.content_sha256"] = False
    mk_path = root / str(entry.get("marker", ""))
    if entry.get("marker") and mk_path.is_file():
        mk = strict_json(mk_path).get("verified_sha256", {})
        base = mk_path.parent.parent
        checks["ok_marker_matches_disk"] = bool(mk) and all(
            (base / rel).is_file() and sha256_file(base / rel) == s for rel, s in mk.items())
    else:
        checks["ok_marker_matches_disk"] = False
    _fail(mid, checks)
    pm = PanelManifest(manifest_id=mid, header=header, users=cols["panel_users"], queries=cols["panel_queries"],
                       support=cols["panel_support"])
    struct = check_panel_structure(pm)
    checks.update({f"structure.{k}": v for k, v in struct.items()})
    _fail(mid, checks)
    pm.record = {"manifest_id": mid, "verified_by": "MANIFEST_REGISTRY (v4trunc.panel.verify_panel)",
                 "registry": {"path": str(registry_path), "file_sha256": reg_file, "self_sha256": reg_self,
                              "data_version": reg.get("data_version")},
                 "header_file_sha256": hsha, "header_content_sha256": header.get("header_content_sha256"),
                 "content_sha256": entry.get("content_sha256"), "tables": tables, "lock_self_sha256": lself,
                 "lock_file_sha256": lfile, "transform_state_self_sha256": ts_self,
                 "transform_state_file_sha256": ts_file, "checks_passed": len(checks),
                 "seconds": round(time.time() - t0, 2)}
    say(f"[v4trunc.panel] {mid}: {len(checks)} checks passed")
    return pm


def _fail(mid: str, checks: Mapping) -> None:
    bad = sorted(k for k, v in checks.items() if v is not True)
    if bad:
        raise PanelError(f"{mid}: panel verification failed: {bad}")


def check_panel_structure(pm: PanelManifest) -> dict:
    """The M04 assertions re-checked on the verified tables (never trusted from the header)."""
    q, s = pm.queries, pm.support
    c = {}
    key = list(zip(q["panel_type"].tolist(), q["h"].tolist(), q["eval_user_key"].tolist()))
    c["one_query_per_user_per_subpanel_h"] = len(set(key)) == len(key)
    c["n_permitted_eq_h"] = bool(np.array_equal(np.asarray(q["n_permitted_events"]), np.asarray(q["h"])))
    c["prior_ge_h"] = bool(np.all(np.asarray(q["n_prior_events_available"]) >= np.asarray(q["h"])))
    c["h0_no_permitted_first"] = bool(np.all(np.asarray(q["permitted_first_decision_id"])[q["h"] == 0] == -1))
    c["h0_no_support"] = bool(np.all(np.asarray(q["n_support_labels"])[q["h"] == 0] == 0))
    fixed_ok = True
    for p, hs in FIXED_BUDGETS.items():
        m = q["panel_type"] == p
        for u in np.unique(q["eval_user_key"][m]).tolist():
            mm = m & (q["eval_user_key"] == u)
            fixed_ok &= np.unique(q["query_decision_id"][mm]).size == 1 and \
                sorted(q["h"][mm].tolist()) == sorted(hs)
    c["fixed_query_identical_across_h"] = bool(fixed_ok)
    qk = {k: i for i, k in enumerate(key)}
    sk = list(zip(s["panel_type"].tolist(), s["h"].tolist(), s["eval_user_key"].tolist()))
    idx = np.array([qk.get(k, -1) for k in sk], dtype=np.int64)
    c["every_support_row_has_its_query"] = bool(np.all(idx >= 0))
    if idx.size and np.all(idx >= 0):
        c["support_strictly_before_query"] = bool(np.all(np.asarray(s["support_target_ts_us"])
                                                         < np.asarray(q["query_decision_ts_us"])[idx]))
        c["query_never_support"] = bool(np.all(np.asarray(s["support_decision_id"])
                                               != np.asarray(q["query_decision_id"])[idx]))
        c["support_count_matches"] = bool(np.array_equal(np.bincount(idx, minlength=len(key)),
                                                         np.asarray(q["n_support_labels"])))
    return c


# ------------------------------------------------------------------------------------------------ §11.2 / §11.3
def support_rows(table: EventTable, q_row: int, h: Optional[int]) -> np.ndarray:
    """E11.3 support labels of query q_row under budget h (event-table rows, ascending). h = None: every prior row."""
    u = int(table.row_user[q_row])
    start = int(table.user_start[u])
    first = start if h is None else int(q_row) - int(h)
    if first < start:
        raise PanelError(f"h={h} exceeds the query's prior events")
    perm = np.arange(first, int(q_row), dtype=np.int64)
    m = table.meta
    ok = m["has_obs"][perm] & (m["target_class"][perm] >= 0) & (m["loss_mask"][perm] | m["eval_mask"][perm]) \
        & (m["target_ts_us"][perm] >= 0) & (m["target_ts_us"][perm] < int(m["decision_ts_us"][q_row]))
    return perm[ok]


def _user_queries(table: EventTable, rows: np.ndarray) -> tuple:
    """§11.2 for one user's event-table rows (store order): ([(panel_type, h, index-in-user of q_u)], fixed50, fixed20).
    q_u = the EARLIEST E2E VALIDATION decision with at least h_min (FIXED) / h (ALL_AVAILABLE) prior events."""
    cand = np.flatnonzero((table.split_code[rows] > 0) & table.meta["eval_mask"][rows])  # prior events = index in user
    qs, elig = [], {"FIXED_50": False, "FIXED_20": False}
    if cand.size == 0:
        return qs, False, False
    for p, hmin in FIXED_MIN.items():
        c = cand[cand >= hmin]
        if c.size:
            elig[p] = True
            qs += [(p, h, int(c[0])) for h in FIXED_BUDGETS[p]]
    for h in BUDGETS:
        c = cand[cand >= h]
        if c.size:
            qs.append(("ALL_AVAILABLE", h, int(c[0])))
    return qs, elig["FIXED_50"], elig["FIXED_20"]


def rederived_query_keys(table: EventTable) -> tuple:
    """({(panel_type, h, client_key_hex, query_decision_id)}, {hex: fixed50_eligible}, {hex: fixed20_eligible}) of
    every user of `table` by the §11.2 rules (no support; that is check_panel_against_events' E11.3 check)."""
    keys, f50, f20 = set(), {}, {}
    for u in range(table.n_users):
        rows = table.rows_of_user(u)
        hx = str(table.user_hex[u])
        qs, f50[hx], f20[hx] = _user_queries(table, rows)
        for p, h, qi in qs:
            keys.add((p, int(h), hx, int(table.decision_id[rows[qi]])))
    return keys, f50, f20


def check_panel_rederived(pm: PanelManifest, table: EventTable) -> dict:
    """REV-ANA-01 B-F2: the (verified) panel re-derived from the loaded event table. The registered set of
    (panel_type, h, client_key_hex, query_decision_id) must EQUAL the §11.2 derivation over the panel users present in
    the table (a later-than-earliest q_u, a missing / extra sub-panel row or a missing eligible user all fail), every
    loaded user must be a panel user, every panel query's user must be loaded, and the fixed50/fixed20 eligibility
    flags must equal the derivation. Fail-closed: every value is a bool the caller refuses on."""
    want, f50, f20 = rederived_query_keys(table)
    q = pm.queries
    hexes = pm.hex_of_key(q["eval_user_key"])
    reg = list(zip([str(x) for x in q["panel_type"].tolist()], [int(x) for x in q["h"].tolist()],
                   [str(x) for x in hexes.tolist()], [int(x) for x in q["query_decision_id"].tolist()]))
    got = set(reg)
    panel_hex = [str(x) for x in np.asarray(pm.users["client_key_hex"], dtype=object).tolist()]
    loaded = {str(x) for x in table.user_hex.tolist()}
    pos = {hx: i for i, hx in enumerate(panel_hex)}
    flags_ok = all(bool(pm.users["fixed50_eligible"][pos[hx]]) == bool(f50[hx])
                   and bool(pm.users["fixed20_eligible"][pos[hx]]) == bool(f20[hx]) for hx in loaded if hx in pos)
    return {"rederived.query_rows_unique": len(got) == len(reg),
            "rederived.loaded_users_are_panel_users": loaded <= set(panel_hex),
            "rederived.query_users_loaded": {k[2] for k in got} <= loaded,
            "rederived.query_set_eq_panel": got == want,
            "rederived.fixed_eligibility_eq_panel": bool(flags_ok)}


def derive_panel(table: EventTable, *, label: str = "WARM_TRUNCATION", catalogue_K: int = 158_486,
                 cold_stratum: Optional[Mapping[str, str]] = None) -> PanelManifest:
    """An independent §11.2 / §11.3 panel over every user of `table` (eval_user_key = rank of client_key_hex)."""
    users = np.argsort(table.user_hex.astype(str), kind="stable")
    key_of = np.empty(table.n_users, dtype=np.int64)
    key_of[users] = np.arange(users.size)
    m = table.meta
    q_rows, s_rows = [], []
    f50 = np.zeros(table.n_users, bool)
    f20 = np.zeros(table.n_users, bool)
    for u in range(table.n_users):
        rows = table.rows_of_user(u)
        qs, e50, e20 = _user_queries(table, rows)
        f50[u], f20[u] = e50, e20
        for p, h, qi in qs:
            qr = int(rows[qi])
            sup = support_rows(table, qr, h)
            q_rows.append((p, h, int(key_of[u]), int(table.decision_id[qr]), int(m["decision_ts_us"][qr]), qi, h,
                           int(table.decision_id[qr - h]) if h > 0 else -1, int(sup.size),
                           bool(m["target_class"][qr] >= 0), int(m["target_class"][qr])))
            for j in sup.tolist():
                s_rows.append((p, h, int(key_of[u]), int(table.decision_id[j]), int(m["target_ts_us"][j]),
                               int(m["target_class"][j])))
    qa = sorted(q_rows, key=lambda r: (SUBPANELS.index(r[0]), r[1], r[2]))
    sa = sorted(s_rows, key=lambda r: (SUBPANELS.index(r[0]), r[1], r[2], r[3]))
    qn = ("panel_type", "h", "eval_user_key", "query_decision_id", "query_decision_ts_us", "n_prior_events_available",
          "n_permitted_events", "permitted_first_decision_id", "n_support_labels", "query_rankable",
          "query_target_class")
    sn = ("panel_type", "h", "eval_user_key", "support_decision_id", "support_target_ts_us", "support_target_class")
    Q = {k: np.array([r[i] for r in qa], dtype=object if k == "panel_type" else None) for i, k in enumerate(qn)}
    S = {k: np.array([r[i] for r in sa], dtype=object if k == "panel_type" else None) for i, k in enumerate(sn)}
    for d, names in ((Q, qn), (S, sn)):
        for k in names:
            if k != "panel_type" and d[k].size == 0:
                d[k] = np.zeros(0, dtype=np.int64)
        if d["panel_type"].size == 0:
            d["panel_type"] = np.zeros(0, dtype=object)
    Q["h"] = Q["h"].astype(np.int32)
    S["h"] = S["h"].astype(np.int32)
    Q["query_rankable"] = Q["query_rankable"].astype(bool)
    hexes = table.user_hex[users]
    strata = np.array([(cold_stratum or {}).get(str(x), "WARM") for x in hexes.tolist()], dtype=object)
    U = {"eval_user_key": np.arange(users.size, dtype=np.int64), "client_key_hex": hexes.astype(object),
         "cold_stratum": strata, "fixed50_eligible": f50[users], "fixed20_eligible": f20[users]}
    header = {"manifest_id": f"DERIVED_{label}", "panel_label": label, "catalogue_K": int(catalogue_K),
              "candidate_order_hash": None, "derived_by": "v4trunc.panel.derive_panel"}
    pm = PanelManifest(manifest_id=header["manifest_id"], header=header, users=U, queries=Q, support=S)
    return pm


def check_panel_against_events(pm: PanelManifest, table: EventTable) -> dict:
    """Every query and support row of the (verified) panel equals the event-table derivation (fail-closed). This also
    proves that the panel's (decision_ts, decision_id) event order equals the store order used by the builder."""
    hexes = pm.hex_of_key(pm.queries["eval_user_key"])
    q_row = np.array([table.row_of(int(d)) for d in pm.queries["query_decision_id"].tolist()], dtype=np.int64)
    u_of_q = table.row_user[q_row]
    same_user = np.array([table.user_hex[u] == hx for u, hx in zip(u_of_q.tolist(), hexes.tolist())], dtype=bool)
    rank = q_row - table.user_start[u_of_q]
    h = np.asarray(pm.queries["h"], dtype=np.int64)
    first = np.where(h > 0, table.decision_id[np.maximum(q_row - h, 0)], -1)
    m = table.meta
    checks = {
        "query_user_is_panel_user": bool(np.all(same_user)),
        "n_prior_events_available_eq_store_rank": bool(np.array_equal(rank, np.asarray(pm.queries["n_prior_events_available"]))),
        "permitted_first_decision_id_eq_store_order": bool(np.array_equal(first, np.asarray(pm.queries["permitted_first_decision_id"]))),
        "query_decision_ts_eq": bool(np.array_equal(m["decision_ts_us"][q_row], np.asarray(pm.queries["query_decision_ts_us"]))),
        "query_ts_eq_block_ts": bool(np.array_equal(table.ts_us[q_row], m["decision_ts_us"][q_row])),
        "query_target_class_eq": bool(np.array_equal(m["target_class"][q_row], np.asarray(pm.queries["query_target_class"]))),
        "query_is_e2e_validation": bool(np.all((table.split_code[q_row] > 0) & m["eval_mask"][q_row])),
    }
    # support rows, per query, as sets (decision id, target ts, target class)
    want: dict = {}
    for i in range(len(q_row)):
        k = (str(pm.queries["panel_type"][i]), int(h[i]), int(pm.queries["eval_user_key"][i]))
        rows = support_rows(table, int(q_row[i]), int(h[i]))
        want[k] = sorted(zip(table.decision_id[rows].tolist(), m["target_ts_us"][rows].tolist(),
                             m["target_class"][rows].tolist()))
    got: dict = {}
    s = pm.support
    for i in range(len(s["h"])):
        k = (str(s["panel_type"][i]), int(s["h"][i]), int(s["eval_user_key"][i]))
        got.setdefault(k, []).append((int(s["support_decision_id"][i]), int(s["support_target_ts_us"][i]),
                                      int(s["support_target_class"][i])))
    got = {k: sorted(v) for k, v in got.items()}
    checks["support_rows_eq_E11_3_rule"] = all(want[k] == got.get(k, []) for k in want) and set(got) <= set(want)
    return checks


# ------------------------------------------------------------------------------------------------ evaluation manifests
def subpanel_manifest(pm: PanelManifest, panel_type: str, h: int, *, full_history: bool = False):
    """The EvalManifest of one sub-panel / budget (`full_history` only relabels it: the C_ref / FULL view of the same
    queries)."""
    from v4eval.manifest import build_eval_manifest
    r = pm.rows(panel_type, h)
    if r.size == 0:
        raise PanelError(f"{pm.manifest_id}: no queries for {panel_type} h={h}")
    q = pm.queries
    tc = np.asarray(q["query_target_class"], dtype=np.int64)[r]
    euk = np.asarray(q["eval_user_key"], dtype=np.int64)[r]
    label = pm.label if pm.label in ("COLD_START", "WARM_TRUNCATION", "RECIPE_SELECTION") else "WARM_TRUNCATION"
    strata = np.asarray(pm.users["cold_stratum"], dtype=object)[euk]
    cols = {"decision_id": np.asarray(q["query_decision_id"], dtype=np.int64)[r], "eval_user_key": euk,
            "eval_mask": np.ones(r.size, dtype=bool), "target_class": tc, "target_oov": tc < 0,
            "censor_class": np.array(["NONE"] * r.size, dtype=object),
            "panel_label": np.array([label] * r.size, dtype=object), "cold_stratum": strata}
    header = {"manifest_id": f"{pm.manifest_id}::{panel_type}::h{int(h)}" + ("::FULL" if full_history else ""),
              "catalogue_K": pm.catalogue_K, "candidate_order_hash": pm.header.get("candidate_order_hash"),
              "panel_manifest_id": pm.manifest_id, "panel_label": label, "panel_type": panel_type,
              "h": None if full_history else int(h), "query_h": int(h), "history": "FULL" if full_history else "H_h",
              "panel_header_content_sha256": pm.header.get("header_content_sha256")}
    return build_eval_manifest(header, cols), r


def eligibility(pm: PanelManifest) -> dict:
    """Per sub-panel / h: users, rankable queries, support labels, zero-support users, by stratum (report only)."""
    q = pm.queries
    strata = np.asarray(pm.users["cold_stratum"], dtype=object)
    out = {}
    for p, h in pm.subpanels():
        r = pm.rows(p, h)
        euk = np.asarray(q["eval_user_key"], dtype=np.int64)[r]
        ns = np.asarray(q["n_support_labels"], dtype=np.int64)[r]
        st = strata[euk]
        out[f"{p}_h{h}"] = {"users": int(r.size), "rankable_queries": int(np.asarray(q["query_rankable"])[r].sum()),
                            "support_labels": int(ns.sum()), "users_with_zero_support": int((ns == 0).sum()),
                            "by_stratum": {s: int((st == s).sum()) for s in sorted(set(st.tolist()))}}
    return {"panel": pm.manifest_id, "label": pm.label, "n_users": int(len(pm.users["eval_user_key"])),
            "subpanels": out, "note": "reported, never used to select (EVALUATION_CONTRACT_V4 E11.2)"}


def digest(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()
