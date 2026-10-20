"""v4data09.d9_common (owner: Data engineer, DATA-09) - ids, roles, the extended field dictionary (one bool column,
grace_matured, MANIFEST_SPEC M07 "an extra bool field grace_matured"), the GATED table encoder / manifest writer and the
header re-labeller of the RETRO_TEST manifests.

Nothing here reads the dataset. The locked emitter (v4fs_data.fs_common.encode_tables / write_manifest) refuses every id
containing TEST by design (v4data.manifests.refuse_test_manifest); the two functions below are the same code path with
that refusal replaced by the TEST gate (d9_gate.revalidate + the header's gate binding) and the table re-read done with
the extended field dictionary (reread_columns_d9). The locked module-level FIELD_TYPES is never mutated: importing this
module has no side effect on any locked builder (tested: tests_data09 test_no_side_effect_*).
"""
from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
CODE = HERE.parent
V4 = CODE.parent

from v4data.common import (EnumValues, atomic_write_bytes, json_bytes_strict, sha256_bytes, sha256_file,  # noqa: E402
                           table_content_sha256, write_marker)
from v4data.manifests import FIELD_TYPES, M02_DECISION_FIELDS, BuildBlocked  # noqa: E402
from v4data06 import d6_common as DC  # noqa: E402
from v4fs_data import fs_common as FC  # noqa: E402

from .d9_gate import GateRecord, TestGateRefused, header_matches_gate, revalidate  # noqa: E402

TASK = "DATA-09"
DATA_VERSION = DC.DATA_VERSION                        # CLEAN_V2_POP5100_V2: the same data version, RETRO_TEST rows
OUT_REL = "data_test"                                 # server output root (relative to ppsi_v4)

MAIN_TEST_ID = "MAIN_RETRO_TEST_5_100_131072"
UNEW_TEST_ID = "UNEW_HOLDOUT_RETRO_TEST_32768"
TRUE_COLD_TEST_ID = "TRUE_COLD_RETRO_TEST"
SPARSE_TEST_ID = "SPARSE_PANEL_16384_RETRO_TEST"
HEAVY_TEST_ID = "HEAVY_PANEL_16384_RETRO_TEST"
TEST_IDS = (MAIN_TEST_ID, UNEW_TEST_ID, TRUE_COLD_TEST_ID, SPARSE_TEST_ID, HEAVY_TEST_ID)

# PREG1_AMENDMENT_06D §2 / DATA-09 packet: the five TEST roles (train = the UNCHANGED registered support manifest)
TEST_ROLES = {
    "MAIN_TEST": {"train": DC.MAIN_TRAIN_ID, "eval": MAIN_TEST_ID},
    "UNSEEN_HOLDOUT_TEST": {"train": DC.UNEW_TEST_TRAIN_ID, "eval": UNEW_TEST_ID},
    "TRUE_COLD_TEST": {"eval": TRUE_COLD_TEST_ID},
    "SPARSE_PANEL_TEST": {"train": DC.SPARSE_TRAIN_ID, "eval": SPARSE_TEST_ID},
    "HEAVY_PANEL_TEST": {"train": DC.HEAVY_TRAIN_ID, "eval": HEAVY_TEST_ID},
}
ROLE_OF_ID = {pair["eval"]: role for role, pair in TEST_ROLES.items()}
VALIDATION_COUNTERPART = {MAIN_TEST_ID: DC.MAIN_EVAL_ID, UNEW_TEST_ID: DC.UNEW_TEST_EVAL_ID,
                          TRUE_COLD_TEST_ID: DC.TRUE_COLD_ID, SPARSE_TEST_ID: DC.SPARSE_EVAL_ID,
                          HEAVY_TEST_ID: DC.HEAVY_EVAL_ID}
DEFINITION = {MAIN_TEST_ID: "M07_TEST_WARM", UNEW_TEST_ID: "M07_TEST_WARM", SPARSE_TEST_ID: "M07_TEST_WARM",
              HEAVY_TEST_ID: "M07_TEST_WARM", TRUE_COLD_TEST_ID: "M08_TEST_COLD_PANEL"}
PARENT = {MAIN_TEST_ID: DC.MAIN_TRAIN_ID, UNEW_TEST_ID: DC.UNEW_TEST_TRAIN_ID, SPARSE_TEST_ID: DC.SPARSE_TRAIN_ID,
          HEAVY_TEST_ID: DC.HEAVY_TRAIN_ID, TRUE_COLD_TEST_ID: None}
BAND_OF_ID = {MAIN_TEST_ID: "MAIN", UNEW_TEST_ID: "U_NEW_TEST", SPARSE_TEST_ID: "SPARSE", HEAVY_TEST_ID: "HEAVY"}
CLASS_POS_BAND = {MAIN_TEST_ID: "MAIN", UNEW_TEST_ID: "U_NEW_TEST", SPARSE_TEST_ID: "SPARSE", HEAVY_TEST_ID: "HEAVY",
                  TRUE_COLD_TEST_ID: "MAIN"}   # TRUE_COLD_VALIDATION counts class positives over MAIN_TRAIN (DATA-06)

# the extended field dictionary: the locked one plus grace_matured (never written back into v4data.manifests)
FIELD_TYPES_D9 = dict(FIELD_TYPES)
FIELD_TYPES_D9["grace_matured"] = ("bool", None)
TEST_DECISION_FIELDS = tuple(M02_DECISION_FIELDS) + ("grace_matured",)

G0_SPLIT_COUNTS = V4 / "g0" / "data_census" / "SPLIT_COUNTS.json"
G0_CLIENT_CENSUS = V4 / "g0" / "data_census" / "CLIENT_CENSUS.json"

SELECTION_REFUSAL = (
    "RETRO_TEST (EVALUATION_CONTRACT_V4 §10; PREG1_AMENDMENT_06D §2): built once, after ALL_METHODS_LOCK, under the "
    "TEST_EXPOSURE_REGISTER MANIFEST_BUILD entry named in this header. Never used for any selection, stopping, "
    "calibration, recipe or architecture choice; nothing is re-run after the look except an interrupted computation with "
    "identical hashes (E10.3). All V4 TEST results are retrospective (old TEST previously inspected, register entry 3).")

GRACE_NOTE = (
    "OBSERVED_MATURED_IN_GRACE rows (grace_matured = true) carry the ledger's own target (target_class / target_ts in "
    "LABEL_GRACE [2019-11-30, 2019-12-01)); they are eval rows (eval_mask = true, censor_class NONE) and rankable iff "
    "target_class >= 0, exactly like OBSERVED_IN_PARTITION rows. Their target_category/brand/main_category_state is "
    "NOT_APPLICABLE: the frozen v1 model_ready features exclude LABEL_GRACE rows by construction "
    "(next_item_eda pipeline6.py: WHERE e.split <> 'LABEL_GRACE'), so the target's own event row does not exist.")

BUILD_TIME_CHECK_M07 = (
    "MANIFEST_SPEC M07 build_time_check (static code inspection, DATA-09): next_item_eda/src/rees46_eda/pipeline5.py "
    "rr / rr_next (the run-run chain) is computed WINDOW PARTITION BY user_id, run_index ORDER BY run_run_index, so a "
    "TEST tail's grace target is the next distinct product INSIDE THE SAME behavioural run (fence-ignoring only across "
    "the TEST/LABEL_GRACE split fence); a run-run spanning the fence takes the split of its first block (arg_min(split, "
    "block_ts)). LABEL_GRACE representatives never originate decisions (WHERE r.split <> 'LABEL_GRACE'). The session "
    "constraint of PKG/03 §3 L19 therefore holds by construction; the builder additionally asserts every grace target "
    "lies in [TEST_END, LABEL_GRACE_END).")

CENSOR_NOTE = (
    "EVALUATION_CONTRACT_V4 §2.2 / E2.1 with VALIDATION_END replaced by the end of LABEL_GRACE: CEN_NO_NEXT = "
    "NO_OBSERVED_NEXT; CEN_LEGACY_BARRIER = BU with old split VALIDATION_SELECT and next_split VALIDATION_CONFIRM "
    "(impossible on TEST rows, kept for precedence fidelity); CEN_RIGHT_EDGE = BU with next_split in {TEST, "
    "LABEL_GRACE} (right-censored at the end of LABEL_GRACE); CEN_AMBIGUOUS_NEXT_BARRIER = every AMBIGUOUS_BOUNDARY row; "
    "precedence LEGACY > RIGHT_EDGE > AMBIGUOUS > NO_NEXT; NONE iff eval_mask; eval_mask == target_status in "
    "{OBSERVED_IN_PARTITION, OBSERVED_MATURED_IN_GRACE}; a row matching no rule stops the build.")


# ------------------------------------------------------------------------------------------------ code identity
BUILD_FILES = ("__init__.py", "d9_common.py", "d9_gate.py", "d9_read.py", "d9_build.py")


def d9_code_hashes() -> dict:
    files = [HERE / n for n in BUILD_FILES]
    return {"d9_code_sha256": FC.code_tree_sha256(files, HERE),
            "d9_code_sha256_rule": "v4data.common.code_tree_sha256 over code/v4data09/{__init__,d9_common,d9_gate,"
                                   "d9_read,d9_build}.py",
            "d9_code_files_sha256": {p.name: FC.sha256_file(p) for p in files},
            "v4data06_files_sha256": {n: FC.sha256_file(CODE / "v4data06" / n) for n in ("d6_common.py", "d6_build.py")},
            "census_v2_core_sha256": FC.sha256_file(CODE / "census_v2" / "core.py"),
            "v4fs_data_fs_common_sha256": FC.sha256_file(CODE / "v4fs_data" / "fs_common.py"),
            "test_exposure_register_tool_sha256": FC.sha256_file(V4 / "tools" / "test_exposure_register.py"),
            "v4data_builder_code_sha256": FC.builder_code_sha256()}


# ------------------------------------------------------------------------------------------------ tables
def reread_columns_d9(t: pa.Table) -> list:
    """v4data.manifests.reread_columns with FIELD_TYPES_D9 (identical for every locked column)."""
    cols = []
    for name in t.column_names:
        dtype = FIELD_TYPES_D9[name][0]
        col = t[name]
        if dtype == "string":
            d = col.combine_chunks().dictionary_encode() if col.num_chunks else pa.array([], pa.string()).dictionary_encode()
            vals = EnumValues(d.indices.to_numpy(zero_copy_only=False).astype(np.int64), d.dictionary.to_pylist())
        else:
            vals = col.to_numpy(zero_copy_only=False)
        cols.append((name, dtype, vals))
    return cols


def _require_test_id(mid: str) -> None:
    if mid not in TEST_IDS:
        raise BuildBlocked(f"{mid}: the gated DATA-09 emitter writes only the registered RETRO_TEST ids {TEST_IDS}")


def encode_tables_d9(b, mid: str, m: dict, gate: GateRecord) -> list:
    """v4fs_data.fs_common.encode_tables for a RETRO_TEST manifest: the gate replaces the id refusal."""
    revalidate(gate)
    _require_test_id(mid)
    out = []
    for tname, tbl in m["tables"].items():
        b._check_no_raw_ids(mid, tname, tbl)
        content = table_content_sha256(b._columns(tbl))
        buf = io.BytesIO()
        pq.write_table(b._arrow(tbl), buf, compression="zstd", compression_level=9)
        data = buf.getvalue()
        back = pq.read_table(pa.BufferReader(data))
        if table_content_sha256(reread_columns_d9(back)) != content:
            raise BuildBlocked(f"{mid}.{tname}: content hash changed after the in-memory Parquet round trip")
        del back
        out.append((tname, content, data, int(len(next(iter(tbl.values()))))))
    return out


def write_manifest_d9(out_root: Path, mid: str, enc: list, header: dict, gate: GateRecord,
                      out_dir: str = "manifests") -> dict:
    """v4fs_data.fs_common.write_manifest for a RETRO_TEST manifest (tables, header, then the .OK marker last)."""
    revalidate(gate)
    _require_test_id(mid)
    if not header_matches_gate(header, gate):
        raise TestGateRefused(f"{mid}: the header's all_methods_lock_sha256 / register entry id is not the open gate's")
    files = {}
    for tname, content, data, n in enc:
        rel = f"{out_dir}/{mid}.{tname}.parquet"
        p = atomic_write_bytes(out_root, rel, data)
        fsha = sha256_file(p)
        if fsha != sha256_bytes(data):
            raise BuildBlocked(f"{rel}: file bytes differ from the encoded bytes")
        if table_content_sha256(reread_columns_d9(pq.ParquetFile(p).read())) != content:
            raise BuildBlocked(f"{rel}: content hash changed after the disk round trip")
        files[rel] = fsha
    hb = json_bytes_strict(header)
    back = json.loads(hb, parse_constant=lambda c: (_ for _ in ()).throw(ValueError(c)))
    if back != header or FC.header_content_sha256(back) != header["header_content_sha256"]:
        raise BuildBlocked(f"{mid}: header does not round-trip or its content hash does not verify")
    hrel = f"{out_dir}/{mid}.header.json"
    hp = atomic_write_bytes(out_root, hrel, hb)
    files[hrel] = sha256_file(hp)
    for rel, fsha in files.items():
        if sha256_file(Path(out_root) / rel) != fsha:
            raise BuildBlocked(f"{rel}: file changed before the marker was written")
    write_marker(out_root, f"{out_dir}/{mid}.OK", files)
    return {"files": files, "header_file_sha256": files[hrel]}


# ------------------------------------------------------------------------------------------------ header
def d9_header(b, m: dict, tables_meta: list, *, out_id: str, role: str, parent, lock: dict, code: dict,
              population: dict, expected: list, users_keys_v2: np.ndarray, gate: GateRecord,
              extra: dict | None = None) -> dict:
    """v4data06.d6_common.d6_header (m['manifest_id'] = the MANIFEST_SPEC definition M07 / M08, so the builder's _header
    finds its spec entry), re-labelled to DATA-09: the two M07 header_extra fields from the gate, v4_split RETRO_TEST,
    the TEST selection refusal and the M07 notes; content hash recomputed by the builder rule."""
    h = DC.d6_header(b, m, tables_meta, out_id=out_id, role=role, parent=parent, lock=lock, code=code,
                     population=population, expected=expected, users_keys_v2=users_keys_v2, extra=extra)
    h.pop("header_content_sha256")
    definition = h["definition"]["manifest_spec_definition"]
    h["task"] = TASK
    h["v4_split"] = "RETRO_TEST"
    h["all_methods_lock_sha256"] = gate.lock_sha256
    h["test_exposure_register_entry_id"] = gate.entry_id
    h["test_gate"] = gate.binding()
    h["previously_seen_evaluation_period"] = True
    h["selection_refusal"] = SELECTION_REFUSAL
    h["definition"]["note"] = (
        f"MANIFEST_SPEC {definition}: tables = the M02 / M03 schema (users + decisions; built by the locked "
        "v4data.manifests._eval_manifest, unedited) with v4_split RETRO_TEST, subperiod NONE, target_status including "
        "OBSERVED_MATURED_IN_GRACE and the extra bool column grace_matured (last decisions column). The users-table "
        "column n_validation_decisions keeps its M02 name and counts this manifest's RETRO_TEST decisions. "
        "eval_user_key / rankable / target_class / oov_cause / target_class_cohort_train_positives follow "
        "MAIN_VALIDATION's code path exactly; class positives are counted over the parent (support) manifest's TRAIN "
        "rows (TRUE_COLD: over MAIN_TRAIN, as TRUE_COLD_VALIDATION)")
    h["censor_rules"] = CENSOR_NOTE
    h["grace_note"] = GRACE_NOTE
    h["build_time_check"] = BUILD_TIME_CHECK_M07
    h = FC.jsonable(h)
    h["header_content_sha256"] = FC.header_content_sha256(h)
    return h


chk = DC.chk

__all__ = [n for n in dir() if not n.startswith("_")]
