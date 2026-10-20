"""DATA-09 real-data tests (server, after job_test_build.sh and the registry / verify / g0 steps of job_test_final.sh):
the build receipt and its gate, every RETRO_TEST manifest on disk under the open gate, the registry extension rule on the
real files, the rebuilt bands against REGISTRY_V2, the loader-path receipt and the G0 integrity receipt. They read only
ppsi_v4/data_test, contracts/ and the register; never the v1 dataset. Skipped unless the build receipt exists AND the
real TEST gate is open (so they are always skipped on the local host, where ALL_METHODS_LOCK does not exist)."""
from __future__ import annotations

import copy
import json

from conftest import DATA_TEST, V4, real_only
from data_testkit import nc
from v4data.common import sha256_file
from v4data09 import d9_common as D9
from v4data09 import d9_finalize as F
from v4data09 import d9_registry as R
from v4fs_data import fs_common as FC


def _gate():
    from v4data09.d9_gate import check_test_gate
    return check_test_gate()


def _load(p):
    return json.loads(p.read_text(encoding="utf-8"))


@real_only
def test_real_build_receipt_full_pass_under_the_open_gate():
    rec = F.verify_build_receipt(DATA_TEST, _gate(), require_full=True)
    assert rec["n_failed"] == 0 and set(rec["manifests"]) == set(D9.TEST_IDS)
    names = {c["name"] for c in rec["checks"]}
    for band, mid in (("MAIN", "MAIN_TRAIN_5_100_131072"), ("U_NEW_TEST", "UNEW_HOLDOUT_SUPPORT_32768"),
                      ("SPARSE", "SPARSE_PANEL_16384_SUPPORT"), ("HEAVY", "HEAVY_PANEL_16384_SUPPORT")):
        assert f"{band}.population_sha256_selection_key_v2_eq_registry_{mid}" in names


@real_only
def test_real_manifests_verify_on_disk():
    g = _gate()
    for mid in D9.TEST_IDS:
        v = F.verify_manifest_on_disk_d9(V4, D9.OUT_REL, mid, g)
        assert v["all_ok"], {k: x for k, x in v["checks"].items() if not x}


@real_only
def test_real_extension_rule():
    ext = _load(V4 / F.EXT_REL)
    chk = R.verify_test_ext_registry(ext, _load(V4 / F.BASE_REL), sha256_file(V4 / F.BASE_REL))
    assert chk["all_ok"], {k: v for k, v in chk.items() if not v}
    g = _gate()
    assert ext["extension_test_gate"]["all_methods_lock_sha256"] == g.lock_sha256


@nc("a re-signed extension with an altered REGISTRY_V2 entry must fail on the real files")
@real_only
def test_nc_real_extension_tampered():
    ext = copy.deepcopy(_load(V4 / F.EXT_REL))
    ext["manifests"]["MAIN_TRAIN_5_100_131072"]["n_decisions"] += 1
    assert R.verify_test_ext_registry(R.with_self(ext), _load(V4 / F.BASE_REL), sha256_file(V4 / F.BASE_REL))["all_ok"]


@real_only
def test_real_loader_and_g0_receipts():
    lv = _load(DATA_TEST / "receipts" / "DATA09_LOADER_VERIFY.json")
    assert FC.verify_self_hash(lv, "receipt_sha256") and set(lv["roles"]) == set(D9.TEST_ROLES)
    g0 = _load(DATA_TEST / "receipts" / "DATA09_G0_INTEGRITY.json")
    assert FC.verify_self_hash(g0, "receipt_sha256") and g0["all_match"] is True
