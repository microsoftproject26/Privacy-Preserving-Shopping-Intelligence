"""DATA-09 no-side-effect tests: the LOCKED DATA-04 manifests and the DATA-06 v2 manifests (their VALIDATION manifests
included) of the same synthetic dataset are byte-identical (table content and Parquet file sha256, header content
sha256, ordered hashes) whether or not v4data09 / v4int.test_roles were imported, a gated RETRO_TEST build ran and the
test_roles wrappers were installed in the same interpreter - with a strict-xfail negative control."""
from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from data_testkit import nc

HERE = Path(__file__).resolve().parent


def _probe(ds: Path, out: Path, mode: str, gate_files=None) -> dict:
    args = [sys.executable, "-B", str(HERE / "side_effect_probe.py"), str(ds), str(out), mode]
    if gate_files:
        args += [str(gate_files[0]), str(gate_files[1])]
    r = subprocess.run(args, capture_output=True, text=True, cwd=HERE, env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"))
    assert r.returncode == 0, r.stderr[-3000:]
    return json.loads(r.stdout.strip().splitlines()[-1])


@pytest.fixture(scope="module")
def probes(synth_ds, gate_files, tmp_path_factory):
    a = _probe(synth_ds, tmp_path_factory.mktemp("probe_plain"), "plain")
    b = _probe(synth_ds, tmp_path_factory.mktemp("probe_after"), "after_d9", gate_files)
    return a, b


def test_locked_and_d6_manifests_identical_with_or_without_v4data09(probes):
    a, b = probes
    assert b["extra"]["wrappers_installed"] and len(b["extra"]["retro_test_manifests"]) == 5
    assert set(a["locked"]) >= {"M01_COHORT_TRAIN", "M02_VALIDATION_WARM", "M03_COLD_PANEL_VAL",
                                "M06_CALIB_VALIDATION_WARM"}
    assert set(a["d6"]) >= {"MAIN_VALIDATION_5_100_131072", "UNEW_HOLDOUT_VALIDATION_32768", "TRUE_COLD_VALIDATION",
                            "SPARSE_PANEL_16384_VALIDATION", "HEAVY_PANEL_16384_VALIDATION"}
    assert a["locked"] == b["locked"]
    assert a["d6"] == b["d6"]
    for mid, e in a["locked"].items():                      # the Parquet file bytes too, not only the content
        for t, x in e["tables"].items():
            assert x["file_sha256"] == b["locked"][mid]["tables"][t]["file_sha256"]


def test_no_manifest_id_of_the_locked_builds_contains_test(probes):
    a, _ = probes
    assert not any("TEST" in m.upper() for m in list(a["locked"]) + list(a["d6"]))


@nc("the identity comparison must detect a single changed table hash")
def test_nc_side_effect_detector(probes):
    a, b = probes
    b2 = copy.deepcopy(b)
    e = b2["d6"]["MAIN_VALIDATION_5_100_131072"]["tables"]["decisions"]
    e["table_content_sha256"] = "0" * 64
    assert a["d6"] == b2["d6"]
