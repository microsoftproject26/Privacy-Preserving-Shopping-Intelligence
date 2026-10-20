"""Real-family smoke tests through the family-agnostic core. FL runs use the CLEAN V2 models (code/v4models: widths
15/15/12/4, `fl.families.build_v2`); the snapshot (v1-width) path is kept only for the explicit v1-parity test.
Synthetic inputs only; no dataset rows.

  * isolation: every snapshot module is loaded from code/snap_v1 and its bytes match SNAPSHOT_MANIFEST.json;
  * adapter == the model's own forward, bitwise (eval mode); GRU head = item_embed rows by the model's own indexing
    (a view: the gradient reaches the single item Parameter); SASRec head = the untied table;
  * manifest roles; one FA / FP / PF round and one LO run at a small K override (K = 64);
  * REGISTERED theta0 (Ops lock decision on MODEL-01 risk 1): Server(adapter, init_sha256=bm.init_sha256) keeps the
    registered hash bitwise; a wrong hash is refused; re-recentering the registered SASRec theta0 would change it;
  * full catalogue (K = 158,486; the first existing of: the owner-PC copy, the frozen READ-ONLY server dataset, an
    optional PPSI_V1_ROOT override — the path used is printed and recorded as a junit property): V2 payload = 4 x the
    V2 parameter count; the seed-2026 theta0 hash equals the REGISTERED value for this torch build (server torch
    2.13.0+cu130: FL-02 GPU receipt; Windows torch 2.9.1+cpu: MODEL-01's inventory) — only this last assertion is
    skipped, with the build named, on an unregistered build; v1 parity: the snapshot path gives SYS-01's counts.
"""
from __future__ import annotations

import json
import os
from pathlib import Path, PurePosixPath, PureWindowsPath

import numpy as np
import pytest
import torch

import ppsi_env
from fl.adapter import BufferRule, Role
from fl.client import ClientData, LocalSolver, PFConfig, client_update
from fl.comm import visit_bytes
from fl.families import build_family, build_v2, load_catalogue_poc, module_forward, synthetic_examples_v2
from fl.local_only import LORecipe, run_local_only
from fl.numerics import state_digest
from fl.personal import PersonalStore
from fl.server import InitProvenanceError, Server, run_round
from fl_testkit import assert_raises

CODE = Path(__file__).resolve().parents[1]


CATALOGUE_WIN = r"F:\Queens University\Microsoft Project\eda\rees46_nextitem_v1\mappings\catalogue.parquet"  # owner PC
CATALOGUE_SERVER = "/home/jovyan/shared_data/rees46_nextitem_v1/mappings/catalogue.parquet"   # frozen dataset (READ-ONLY)


def _catalogue_candidates() -> list:
    """Ordered candidates for the frozen v1 catalogue (FL-02R2; the first existing file is used, read-only)."""
    c = [Path(CATALOGUE_WIN), Path(CATALOGUE_SERVER)]
    if os.environ.get("PPSI_V1_ROOT"):
        c.append(Path(os.environ["PPSI_V1_ROOT"]) / "mappings" / "catalogue.parquet")    # optional override
    return c


CATALOGUE_CANDIDATES = _catalogue_candidates()
CATALOGUE = next((p for p in CATALOGUE_CANDIDATES if p.is_file()), None)
SYS01_V1_PARAMS = {"GRU": 20_964_062, "SASREC": 83_094_368}        # company/handoffs/SYS-01.md (v1 widths)
INVENTORY = {f: json.loads((CODE / "v4models" / "inventory" / f"{f}_CLEAN_V2_MANIFEST_INVENTORY.json").read_text())
             for f in ("GRU", "SASREC")}
# Registered seed-2026 theta0 hashes are tied to the torch build (MODEL-01 risk 4; FL-02 GPU receipt). The server
# values are the ones registered for production: FL-02 receipt code/tests_fl/receipts/gpu/init.json (Linux,
# torch 2.13.0+cu130, built on CPU before the device move). The local Windows build uses MODEL-01's inventory.
REGISTERED_INIT_2026 = {
    "2.13.0+cu130": {"GRU": "f1dd4ec4b91c8f2a108588f96d70f6e81e54867460b675a67c45cc19f12d681d",
                     "SASREC": "e264064a1fc678ab4b0eea7aed0290ae522e58f809c82fef11e042c11a9063d4"},
}


def registered_init_2026(family: str):
    """(hash, source) for this torch build, or (None, reason)."""
    v = torch.__version__
    if v in REGISTERED_INIT_2026:
        return REGISTERED_INIT_2026[v][family], f"FL-02 GPU receipt init.json (torch {v})"
    if INVENTORY[family]["environment"]["torch"] == v:
        return INVENTORY[family]["init_state"][str(SEED)]["init_sha256"], f"MODEL-01 inventory (torch {v})"
    return None, f"no registered theta0 hash for torch build {v} (registered: {sorted(REGISTERED_INIT_2026)} and " \
                 f"MODEL-01's {INVENTORY[family]['environment']['torch']})"


SEED = 2026
FAMS = ["GRU", "SASREC"]


def test_catalogue_candidates_and_build_specific_theta0_lookup(monkeypatch):
    """FL-02R2 portability: ordered catalogue candidates (+ optional PPSI_V1_ROOT) and the torch-build lookup."""
    monkeypatch.delenv("PPSI_V1_ROOT", raising=False)
    base = _catalogue_candidates()
    # platform-independent (FL-02R3): on POSIX a Windows path is one opaque name, so compare strings, not .name
    assert len(base) == 2 and str(base[0]) == CATALOGUE_WIN and base[1].as_posix() == CATALOGUE_SERVER
    assert PureWindowsPath(str(base[0])).name == PurePosixPath(base[1].as_posix()).name == "catalogue.parquet"
    monkeypatch.setenv("PPSI_V1_ROOT", "/some/v1/root")
    over = _catalogue_candidates()
    assert len(over) == 3 and str(over[0]) == CATALOGUE_WIN and over[1].as_posix() == CATALOGUE_SERVER
    assert over[2].as_posix().endswith("/some/v1/root/mappings/catalogue.parquet")          # the override comes last
    receipt = CODE / "tests_fl" / "receipts" / "gpu" / "init.json"                  # the registry = the GPU receipt
    if receipt.is_file():
        rec = json.loads(receipt.read_text(encoding="utf-8"))
        assert rec["env"]["torch"] in REGISTERED_INIT_2026
        for fam in FAMS:
            assert REGISTERED_INIT_2026[rec["env"]["torch"]][fam] == rec["families"][fam]["2026"]["init_sha256"]
    monkeypatch.setattr(torch, "__version__", "2.13.0+cu130")
    for fam in FAMS:
        h, src = registered_init_2026(fam)
        assert h == REGISTERED_INIT_2026["2.13.0+cu130"][fam] and "2.13.0+cu130" in src
    monkeypatch.setattr(torch, "__version__", INVENTORY["GRU"]["environment"]["torch"])
    assert registered_init_2026("GRU")[0] == INVENTORY["GRU"]["init_state"]["2026"]["init_sha256"]
    monkeypatch.setattr(torch, "__version__", "9.9.9+unregistered")
    h, why = registered_init_2026("SASREC")
    assert h is None and "9.9.9+unregistered" in why


def test_isolation_and_snapshot_hashes():
    build_v2("GRU", seed=SEED, K=8)
    seen = ppsi_env.assert_isolated()
    ppsi_env.verify_snapshot()
    snap = ppsi_env.SNAP.resolve()
    for name in ("model", "train", "sasrec_model"):
        assert Path(seen[name]).resolve().is_relative_to(snap), (name, seen[name])


@pytest.mark.parametrize("family", FAMS)
def test_adapter_equals_module_forward_and_head(family):
    a = build_v2(family, seed=SEED, K=64).adapter
    a.module.eval()
    b = synthetic_examples_v2(6, 64, torch.Generator().manual_seed(2))
    with torch.no_grad():
        assert torch.equal(a.scores(b), module_forward(a, b)), "adapter scores must equal the model's own forward"
    W = a.head_weight()
    assert W.shape == (64, a.query_dim)
    if family == "GRU":
        assert a.query_dim == 128 and W.data_ptr() == a.module.item_embed.weight[3].data_ptr()   # a view, not a copy
        a.module.train()
        a.module.zero_grad(set_to_none=True)
        a.scores(b).sum().backward()
        assert a.module.item_embed.weight.grad[3:67].abs().sum() > 0    # head gradient reaches the tied rows
    else:
        assert a.query_dim == 256 and W is a.module.output_embed


def test_family_manifests():
    g = build_v2("GRU", seed=SEED, K=64).adapter
    s = build_v2("SASREC", seed=SEED, K=64).adapter
    assert g.manifest.alias_keys == () and s.manifest.alias_keys == ()
    assert g.manifest.entries["product_idx_of_class"].role == Role.BUFFER
    assert g.manifest.entries["product_idx_of_class"].rule == BufferRule.FIXED
    assert set(s.manifest.nonpersistent_buffers) == {"product_idx_of_class", "causal"}
    for a in (g, s):
        assert a.manifest.shared_numel == sum(p.numel() for p in a.module.parameters())
        assert all(a.manifest.entries[k].dtype == "torch.float32" for k in a.manifest.shared_keys)


@pytest.mark.parametrize("family", FAMS)
def test_family_rounds_and_lo_small_k(family):
    bm = build_v2(family, seed=SEED, K=64)
    srv = Server(bm.adapter, init_sha256=bm.init_sha256)
    w = [build_v2(family, seed=SEED + 1, K=64).adapter]
    g = torch.Generator().manual_seed(3)
    cs = [ClientData("user-f0", synthetic_examples_v2(5, 64, g)), ClientData("user-f1", synthetic_examples_v2(20, 64, g))]
    sol = LocalSolver(lr=1e-3, passes=2, batch_size=16, clip=1.0, weight_decay=1e-5)
    store = PersonalStore(srv.adapter.query_dim)
    for r, kw in enumerate([{}, {"mu": 0.01}, {"pf": PFConfig(lr=1e-3), "store": store}]):
        rep = run_round(srv, w, cs, sol, round_idx=r, seed=SEED, **kw)
        assert rep.weight == 2 * 25 and [v.steps for v in rep.visits] == [2, 4]
        assert rep.bytes_down == 2 * srv.manifest.shared_bytes
        assert all(torch.isfinite(t).all() for t in srv.adapter.broadcast_state().values() if t.is_floating_point())
    assert sorted(store.keys()) == ["user-f0", "user-f1"]
    theta0 = build_v2(family, seed=SEED, K=64).theta0()
    recipe = LORecipe(state_digest(theta0), SEED, passes=2, eval_at=(1, 2), solver=sol)
    assert recipe.theta0_digest == bm.init_sha256                    # LO provenance = the registered init hash
    lo = run_local_only(w[0], theta0, cs[1], recipe)
    assert lo.steps == 4 and lo.n_consumed == 40


def check_registered_start(family, raw):
    bm = build_v2(family, seed=SEED, K=64)
    srv = Server(bm.adapter) if raw else Server(bm.adapter, init_sha256=bm.init_sha256)
    assert srv.init_provenance == ("raw_unregistered" if raw else "registered")
    assert state_digest(srv.broadcast()) == bm.init_sha256, "the broadcast theta0 must keep the registered hash"


@pytest.mark.parametrize("family", FAMS)
def test_registered_theta0_is_kept_bitwise(family):
    check_registered_start(family, raw=False)
    bm = build_v2(family, seed=SEED, K=64)
    assert_raises(InitProvenanceError, Server, bm.adapter, init_sha256="0" * 64)
    other = build_v2(family, seed=SEED + 1, K=64)
    assert_raises(InitProvenanceError, Server, other.adapter, init_sha256=bm.init_sha256)


def test_registered_theta0_refused_by_run_driver_when_raw():
    from fl.participation import ParticipationPlan
    from fl.runtime import FLRun, RunConfig
    bm = build_v2("GRU", seed=SEED, K=16)
    cfg = RunConfig("raw-refused", "FA", SEED, 1e-3, LocalSolver(lr=1e-3), exposure_point="end")
    plan = ParticipationPlan.build(["user-a", "user-b"], seed=SEED, manifest_hash="m", group_size=2)
    assert_raises(ValueError, FLRun, cfg, plan, Server(bm.adapter), lambda k: None, 10,
                  workers=[build_v2("GRU", seed=SEED + 1, K=16).adapter])


def test_raw_path_on_registered_sasrec_changes_the_hash():
    """Why the registered path exists: a second recentering of the registered SASRec theta0 is not idempotent."""
    bm = build_v2("SASREC", seed=SEED, K=158_486)
    srv = Server(bm.adapter)                                    # raw path: recenters again
    assert state_digest(srv.broadcast()) != bm.init_sha256


@pytest.mark.skipif(CATALOGUE is None, reason="no v1 catalogue among the candidates: "
                                               + ", ".join(str(p) for p in CATALOGUE_CANDIDATES))
@pytest.mark.parametrize("family", FAMS)
def test_full_catalogue_v2_bytes_init_hash_and_one_visit(family, record_property):
    print(f"[catalogue] using {CATALOGUE} (candidates: {[str(p) for p in CATALOGUE_CANDIDATES]}); "
          f"torch {torch.__version__}")
    record_property("catalogue_used", str(CATALOGUE))
    record_property("torch_build", torch.__version__)
    torch.set_num_threads(4)
    poc = load_catalogue_poc(CATALOGUE)                               # read-only (ParquetFile.read)
    K = int(poc.shape[0])
    assert K == 158_486 and np.array_equal(poc, np.arange(3, 3 + K))
    bm = build_v2(family, seed=SEED, poc=poc)
    inv = INVENTORY[family]
    assert bm.manifest.shared_numel == inv["shared_numel"]
    assert visit_bytes(bm.manifest)["download_bytes"] == 4 * inv["shared_numel"]
    srv = Server(bm.adapter, init_sha256=bm.init_sha256)
    w = build_v2(family, seed=SEED + 1, poc=poc).adapter
    c = ClientData("user-full", synthetic_examples_v2(3, K, torch.Generator().manual_seed(9)))
    r = client_update(w, srv.broadcast(), c, LocalSolver(lr=1e-4, passes=2, batch_size=16), round_idx=0, seed=SEED)
    assert r.n_consumed == 6 and r.steps == 2
    assert all(torch.isfinite(t).all() for t in r.upload.values())
    want, source = registered_init_2026(family)                      # the build-specific check comes LAST
    print(f"[theta0] {family} seed {SEED}: {bm.init_sha256} vs {source}")
    record_property("theta0_hash_source", source)
    if want is None:
        pytest.skip(f"theta0 hash assertion only (all other assertions passed): {source}")
    assert bm.init_sha256 == want, f"theta0 hash differs from the registered value ({source})"


@pytest.mark.parametrize("family", FAMS)
def test_v1_parity_snapshot_path_counts(family):
    """V1 PARITY ONLY: the snapshot-width path still builds SYS-01's measured models (19/15/12/6 widths)."""
    a = build_family(family, seed=SEED, K=158_486)
    assert a.manifest.shared_numel == SYS01_V1_PARAMS[family]
    assert SYS01_V1_PARAMS[family] - INVENTORY[family]["shared_numel"] == 256      # MODEL-01: V2 - v1 = -256
