"""FS-01 test configuration: ppsi_env FIRST (snapshot isolation, T43), then the shared fixtures.

Negative controls are strict-xfail tests named test_nc_* (data_testkit.nc): each runs a key-property check against a
deliberately wrong variant and MUST fail with AssertionError; an XPASS fails the suite.
Synthetic fixtures reuse the DATA-04 frozen-layout testkit (code/tests_data/data_testkit.py, read-only).
Real-data tests read only ppsi_v4/data_fs, data_v2 receipts / TRANSFORM_STATE and the registry; they never open the
v1 dataset, so no TEST row can be read. They are skipped where data_fs is absent (local host) or when
FS_TESTS_SYNTHETIC_ONLY=1 (the pre-build gate).
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.dont_write_bytecode = True
CODE = Path(__file__).resolve().parents[1]
TESTS = Path(__file__).resolve().parent
for p in (str(CODE), str(TESTS), str(CODE / "tests_data")):
    if p not in sys.path:
        sys.path.insert(0, p)
import ppsi_env  # noqa: E402,F401  (mandatory isolation, before any snapshot import)

import pytest  # noqa: E402

V4 = CODE.parent
DATA_FS = V4 / "data_fs"
REAL = (DATA_FS / "receipts" / "FS_BUILD_RECEIPT.json").is_file() and os.environ.get("FS_TESTS_SYNTHETIC_ONLY") != "1"
real_only = pytest.mark.skipif(not REAL, reason="real FS build not present on this host (or synthetic-only gate)")

# synthetic FS configuration: the 3 warm users of the testkit form the pool at 1 <= n <= 100; FS = the top 2 keys
SYN = {"lo": 1, "hi": 100, "n_fs": 2, "bands": {"MAIN": (0, 1), "FS": (1, 3)}, "disjoint_from": ("MAIN",),
       "train_id": "FS_TRAIN_SYN_2", "eval_id": "FS_VALIDATION_SYN_2"}


def quiet(*_a, **_k):
    return None


@pytest.fixture(scope="session")
def synth(tmp_path_factory):
    """The locked DATA-04 builder end-to-end on the synthetic dataset (reference M05 / M06), plus the FS paths."""
    from data_testkit import standard_users, write_dataset
    from v4data.manifests import ManifestBuilder
    users, cat = standard_users()
    root = tmp_path_factory.mktemp("fs_ds")
    write_dataset(root, users, cat)
    out_ref = tmp_path_factory.mktemp("fs_ref_out")
    ref = ManifestBuilder(ds_root=root, out_root=out_ref, buckets=[0], expect=False, log=quiet)
    ref.pass_support()
    ref.pass_aggregate()
    ref.select_populations()
    ref.pass_extract()
    man = ref.assemble()
    hashes = ref.write(man)
    mh = {"receipt": "MANIFEST_HASHES", "builder_code_sha256": ref.code_sha, "manifests": hashes,
          "cohort_key_hashes": {k: v for k, v in ref.pop_hash.items() if k != "universe_size"}}
    mh_path = out_ref / "receipts" / "MANIFEST_HASHES.json"
    mh_path.parent.mkdir(parents=True, exist_ok=True)
    mh_path.write_text(json.dumps(mh), encoding="utf-8")
    return {"root": root, "ref": ref, "man": man, "hashes": hashes, "mh_path": mh_path, "out_ref": out_ref,
            "users": users, "catalogue": cat}


@pytest.fixture(scope="session")
def synth_equiv(synth, tmp_path_factory):
    from v4fs_data.fs_build import PopulationBuilder, run_equivalence
    out = tmp_path_factory.mktemp("fs_equiv_out")
    b = PopulationBuilder(ds_root=synth["root"], out_root=out, buckets=[0], expect=False, log=quiet)
    rec = run_equivalence(b, out, manifest_hashes=synth["mh_path"],
                          server_manifest_dir=synth["out_ref"] / "manifests", write=True)
    return {"rec": rec, "out": out, "builder": b}


@pytest.fixture(scope="session")
def synth_fs(synth, tmp_path_factory):
    from v4fs_data.fs_build import PopulationBuilder, run_fs
    v4 = tmp_path_factory.mktemp("fs_v4")
    out = v4 / "data_fs"
    b = PopulationBuilder(ds_root=synth["root"], out_root=out, buckets=[0], expect=False, log=quiet)
    rec = run_fs(b, out, lo=SYN["lo"], hi=SYN["hi"], n_fs=SYN["n_fs"], bands=SYN["bands"],
                 disjoint_from=SYN["disjoint_from"], train_id=SYN["train_id"], eval_id=SYN["eval_id"], write=True,
                 transform_state_path=v4 / "absent.json")
    return {"rec": rec, "v4": v4, "out": out, "builder": b}
