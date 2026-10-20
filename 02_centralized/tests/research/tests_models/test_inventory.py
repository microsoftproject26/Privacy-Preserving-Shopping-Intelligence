"""Parameter-role manifest and inventory at the Clean V2 widths and the full catalogue (K = 158,486).

Reads only the v1 mapping files (sha256-checked against FILES.json), never a dataset row. The v1 root is the first
existing of (FL-02R3 / MODEL-01R1 portability; the root used is printed and recorded as a junit property):
  1. the owner-PC copy (V1_ROOT_WIN below);
  2. the frozen server dataset /home/jovyan/shared_data/rees46_nextitem_v1 (READ-ONLY: only read, never written);
  3. $PPSI_V1_ROOT, if set.
The committed record was produced on Windows torch 2.9.1+cpu: topology, shapes, parameter counts, bytes and the
manifest digest must match on EVERY build; the init-hash fields are compared build-specifically (the record's own
build, or the server values registered by the FL-02 GPU receipt for torch 2.13.0+cu130), and only those two
assertions are skipped — naming the build — when no value is registered for the running torch build.
  * vocab sizes from the mapping files equal the executed VOCAB_SIZES; the class map is arange(3, 3 + K);
  * a fresh full-K build per family reproduces the committed v4models/inventory/<FAMILY>_CLEAN_V2_MANIFEST_INVENTORY.json:
    manifest digest, shared numel / bytes, per-component inventory and the seed-2026 initial-state sha256;
  * at the v1 widths the same constructor gives exactly SYS-01's measured counts (GRU 20,964,062; SASRec 83,094,368);
    V2 - v1 = -256 per family, and only the two width matrices change shape (read from the committed record).
"""
from __future__ import annotations

import gc
import json
import os
from pathlib import Path, PurePosixPath

import numpy as np
import pytest
import torch

import ppsi_paths
import train as reuse_train
from fl.comm import visit_bytes
from models_testkit import SCHEMA
from v4models.build import build, vocab_sizes_from_mappings

V1_ROOT_WIN = r"F:\Queens University\Microsoft Project\eda\rees46_nextitem_v1"        # owner PC
V1_ROOT_SERVER = ppsi_paths.DATA_ROOT.as_posix()                                        # frozen dataset (READ-ONLY)


def v1_root_candidates() -> list:
    c = [Path(V1_ROOT_WIN), Path(V1_ROOT_SERVER)]
    if os.environ.get("PPSI_V1_ROOT"):
        c.append(Path(os.environ["PPSI_V1_ROOT"]))
    return c


V1_CANDIDATES = v1_root_candidates()
V1_ROOT = next((r for r in V1_CANDIDATES if (r / "mappings" / "catalogue.parquet").is_file()), None)
INV = ppsi_paths.code_dir("v4models") / "inventory"
RECEIPT = ppsi_paths.code_dir("tests_fl") / "receipts" / "gpu" / "init.json"
# Registered seed-2026 initial-state hashes by torch build (MODEL-01 risk 4). The server build's values are the
# production registration measured by the FL-02 GPU receipt (code/tests_fl/receipts/gpu/init.json; CPU build before
# the device move) — the same registry as tests_fl/test_families_smoke.py.
REGISTERED_INIT_2026 = {
    "2.13.0+cu130": {
        "GRU": {"init_sha256": "f1dd4ec4b91c8f2a108588f96d70f6e81e54867460b675a67c45cc19f12d681d",
                "raw_init_sha256": "f1dd4ec4b91c8f2a108588f96d70f6e81e54867460b675a67c45cc19f12d681d"},
        "SASREC": {"init_sha256": "e264064a1fc678ab4b0eea7aed0290ae522e58f809c82fef11e042c11a9063d4",
                   "raw_init_sha256": "ce38d24342fdc253edaa1f5030ae1d1b5c68386ac7d0372dde08adc232bde125"}},
}


def registered_init(family: str, rec: dict):
    """(expected init / raw hashes, source) for the running torch build, or (None, reason)."""
    v = torch.__version__
    if rec["environment"]["torch"] == v:
        return rec["init_state"]["2026"], f"the committed inventory record (torch {v})"
    if v in REGISTERED_INIT_2026:
        return REGISTERED_INIT_2026[v][family], f"FL-02 GPU receipt init.json (torch {v})"
    return None, (f"no registered initial-state hash for torch build {v} (record: {rec['environment']['torch']}; "
                  f"registered: {sorted(REGISTERED_INIT_2026)})")


needs_v1 = pytest.mark.skipif(V1_ROOT is None, reason="no v1 mapping root among the candidates: "
                              + ", ".join(str(r) for r in V1_CANDIDATES))


@needs_v1
def test_vocab_sizes_from_mapping_files(record_property):
    print(f"[v1 root] using {V1_ROOT} (candidates: {[str(r) for r in V1_CANDIDATES]}); torch {torch.__version__}")
    record_property("v1_root_used", str(V1_ROOT))
    vocab, poc = vocab_sizes_from_mappings(V1_ROOT)
    assert vocab == dict(reuse_train.VOCAB_SIZES)
    assert poc.shape == (158_486,) and np.array_equal(poc, np.arange(3, 3 + 158_486))


@needs_v1
@pytest.mark.parametrize("family", ["GRU", "SASREC"])
def test_full_k_manifest_and_inventory_match_the_record(family, record_property):
    print(f"[v1 root] using {V1_ROOT}; torch {torch.__version__}")
    record_property("v1_root_used", str(V1_ROOT))
    record_property("torch_build", torch.__version__)
    torch.set_num_threads(4)
    rec = json.loads((INV / f"{family}_CLEAN_V2_MANIFEST_INVENTORY.json").read_text(encoding="utf-8"))
    bm = build(family, SCHEMA, 2026, "cpu", v1_root=V1_ROOT)
    try:
        assert bm.K == rec["K"] == 158_486
        assert bm.manifest.digest() == rec["manifest_digest"]
        assert bm.manifest.shared_numel == rec["shared_numel"] and bm.manifest.shared_bytes == rec["shared_bytes"]
        assert bm.inventory == rec["parameter_inventory"]
        assert visit_bytes(bm.manifest) == rec["visit_bytes"]
        got_init, got_raw = bm.init_sha256, bm.raw_init_sha256          # compared build-specifically, LAST
        assert rec["schema"]["self_sha256"] == SCHEMA.self_sha256 and rec["widths"] == SCHEMA.widths.as_json()
        roles = {e["role"] for e in rec["manifest"]["entries"].values()}
        assert roles <= {"shared", "buffer"} and rec["manifest"]["entries"]
    finally:
        del bm
        gc.collect()
    ref = rec["v1_reference"]
    assert ref["equals_sys01"] and ref["shared_numel_at_v1_widths"] == ref["sys01_measured_v1"]
    assert ref["delta_v2_minus_v1"] == -256 and ref["only_width_matrices_changed"] and ref["same_key_set"]
    expect = {"GRU": 20_963_806, "SASREC": 83_094_112}[family]
    assert rec["shared_numel"] == expect and rec["parameter_inventory"]["total"] == expect
    assert rec["visit_bytes"]["download_bytes"] == 4 * expect
    want, source = registered_init(family, rec)
    print(f"[init hash] {family} seed 2026: init {got_init} raw {got_raw} vs {source}")
    record_property("init_hash_source", source)
    if want is None:
        pytest.skip(f"init-hash assertions only (all other assertions passed): {source}")
    assert got_init == want["init_sha256"], f"registered theta0 hash differs ({source})"
    assert got_raw == want["raw_init_sha256"], f"raw (pre-recentering) hash differs ({source})"


def test_v1_root_candidates_and_build_specific_registry(monkeypatch):
    """Platform-independent: candidate order, the PPSI_V1_ROOT override, the build-specific lookup, and the registry
    equals the FL-02 GPU receipt when that receipt is present."""
    monkeypatch.delenv("PPSI_V1_ROOT", raising=False)
    base = v1_root_candidates()
    assert len(base) == 2 and str(base[0]) == V1_ROOT_WIN and base[1].as_posix() == V1_ROOT_SERVER
    monkeypatch.setenv("PPSI_V1_ROOT", "/some/v1/root")
    over = v1_root_candidates()
    assert len(over) == 3 and over[2].as_posix() == PurePosixPath("/some/v1/root").as_posix()
    rec = json.loads((INV / "GRU_CLEAN_V2_MANIFEST_INVENTORY.json").read_text(encoding="utf-8"))
    monkeypatch.setattr(torch, "__version__", "2.13.0+cu130")
    want, src = registered_init("SASREC", rec)
    assert want == REGISTERED_INIT_2026["2.13.0+cu130"]["SASREC"] and "2.13.0+cu130" in src
    monkeypatch.setattr(torch, "__version__", rec["environment"]["torch"])
    assert registered_init("GRU", rec)[0] == rec["init_state"]["2026"]
    monkeypatch.setattr(torch, "__version__", "9.9.9+unregistered")
    want, why = registered_init("GRU", rec)
    assert want is None and "9.9.9+unregistered" in why
    if RECEIPT.is_file():
        r = json.loads(RECEIPT.read_text(encoding="utf-8"))
        reg = REGISTERED_INIT_2026[r["env"]["torch"]]
        for fam in ("GRU", "SASREC"):
            assert reg[fam]["init_sha256"] == r["families"][fam]["2026"]["init_sha256"]
            assert reg[fam]["raw_init_sha256"] == r["families"][fam]["2026"]["raw_init_sha256"]
