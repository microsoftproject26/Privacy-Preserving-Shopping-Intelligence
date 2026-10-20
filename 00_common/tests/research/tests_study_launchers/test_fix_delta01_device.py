"""FIX-DELTA-01 F6 / F7 (REV-DELTA-01) — v4int.launch_device (CPU, torch-free parts).

F7 (06E §1): a non-smoke LO launch on any role other than UNSEEN_VAL is refused in preflight without --selection (the
UNSEEN_VAL --select run and smoke runs are not refused; the pure device_spec is unchanged).
F6: an applied SELECTION.json must carry the content_sha256 of the VERIFIED UNSEEN_VAL eval manifest of the registry
(and of the registry entry); a real run also needs the manifest's registered 12,792 users. verify_selection_manifest
reads the registry's UNSEEN_VAL eval id and verifies that manifest through v4int.providers._verify (stubbed here: the
real manifest is server data).
Negative controls: the pre-F7 preflight (check_role only) lets LO MAIN through without a selection; the pre-F6
load_selection applies a selection made on another manifest.
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import v4int.launch_device as D
import v4int.providers as P
from intkit import assert_raises, nc

CONTENT = "6802cc6cec658eae71985ea2f27d2aad3c81853263758e0fb4d45c9f719b8455"   # REGISTRY_V2 UNEW_VAL_VALIDATION_32768
EVAL_ID = "UNEW_VAL_VALIDATION_32768"


def _args(**kw):
    base = dict(arm="LO", family="SASREC", seed=2026, gpu=0, history_policy="USER_HISTORY_L50", role="MAIN",
                select=False, selection=None, base=None, base_kind=None, n_shards=4, shards=None, smoke=False,
                smoke_clients=4, dry_run=True, allow_shared_gpu=False)
    base.update(kw)
    return SimpleNamespace(**base)


def _selection(tmp_path, content=CONTENT, role="UNSEEN_VAL") -> Path:
    obj = {"format": D.SELECTION_FORMAT, "arm": "LO", "seed": 2026, "selection_role": role,
           "eval_manifest_content_sha256": content, "cell": "LO_P2"}
    obj["self_sha256"] = D._canon(obj)
    p = tmp_path / f"SELECTION_{content[:6]}_{role}.json"
    p.write_text(json.dumps(obj), encoding="utf-8")
    return p


# ------------------------------------------------------------------------------------------------ F7
def test_lo_needs_selection_off_unseen_val(tmp_path):
    gate = tmp_path / "G1_APPROVED.json"
    gate.write_text("{}", encoding="utf-8")
    for role in ("MAIN", "CALIBRATION", "UNSEEN_HOLDOUT", "SPARSE_PANEL", "HEAVY_PANEL", "TRUE_COLD"):
        with pytest.raises(D.LaunchRefused, match="needs --selection"):
            D.preflight(_args(role=role), environ={}, gate_path=gate)
        D.preflight(_args(role=role, selection="SELECTION.json"), environ={}, gate_path=gate)      # dry run passes
        D.preflight(_args(role=role, smoke=True), environ={}, gate_path=gate)                     # smoke: allowed
    for role in D.TEST_ROLES:
        with pytest.raises(D.LaunchRefused, match="needs --selection"):
            D.check_lo_selection(role, arm="LO", select=False, selection=None, smoke=False)
        D.check_lo_selection(role, arm="LO", select=False, selection="S.json", smoke=False)
    D.preflight(_args(role="UNSEEN_VAL", select=True), environ={}, gate_path=gate)                 # the selection run
    D.check_lo_selection("UNSEEN_VAL", arm="LO", select=False, selection=None, smoke=False)
    for arm in ("FT_C", "FT_FA"):                                                     # FT: check_role's own rule
        D.check_lo_selection("MAIN", arm=arm, select=False, selection=None, smoke=False)
    spec = D.device_spec("LO", seed=2026, peak_lr=1e-3, role="MAIN", start_digest="x")            # pure spec unchanged
    assert [c["cell"] for c in spec["cells"]] == ["LO_P1", "LO_P2", "LO_P4"]


# ------------------------------------------------------------------------------------------------ F6
def test_selection_manifest_binding_and_users(tmp_path):
    sel = D.load_selection(_selection(tmp_path))
    b = {"manifest_id": EVAL_ID, "content_sha256": CONTENT}
    rec = D.check_selection_manifest(sel, binding=b, n_users=12792, smoke=False, registry_entry={"content_sha256": CONTENT})
    assert rec["content_sha256"] == CONTENT and rec["users_checked"] is True and D.UNSEEN_VAL_USERS == 12792
    assert D.check_selection_manifest(sel, binding=b, n_users=64, smoke=True)["users_checked"] is False
    for kw in (dict(binding=dict(b, content_sha256="0" * 64), n_users=12792, smoke=False),       # other manifest
               dict(binding=b, n_users=12791, smoke=False), dict(binding=b, n_users=None, smoke=False),
               dict(binding=b, n_users=12792, smoke=False, registry_entry={"content_sha256": "1" * 64})):
        assert_raises(D.LaunchRefused, D.check_selection_manifest, sel, **kw)
    nosha = D.load_selection(_selection(tmp_path, content=""))
    assert_raises(D.LaunchRefused, D.check_selection_manifest, nosha, binding=b, n_users=12792, smoke=True)


def test_verify_selection_manifest_reads_the_registry_unseen_val_eval(tmp_path, monkeypatch):
    reg = tmp_path / "REGISTRY_V2.json"
    reg.write_text(json.dumps({"roles": {"UNSEEN_VAL": {"train": "UNEW_VAL_SUPPORT_32768", "eval": EVAL_ID}},
                               "manifests": {EVAL_ID: {"content_sha256": CONTENT, "n_users": 12792}}}), encoding="utf-8")
    seen = []

    def fake_verify(mid, paths, kind, registry, root, log):
        seen.append((mid, kind, registry))
        return SimpleNamespace(manifest_id=mid, tables={"users": {"row_count": users[0]}}, columns=None)
    users = [12792]
    monkeypatch.setattr(P, "_verify", fake_verify)
    monkeypatch.setattr(P, "manifest_binding_of", lambda vm: {"manifest_id": vm.manifest_id, "content_sha256": CONTENT})
    kw = {"paths": None, "manifest_registry": reg, "registry_root": tmp_path, "log": lambda *_: None}
    sel = D.load_selection(_selection(tmp_path))
    rec = D.verify_selection_manifest(sel, reg, kw, smoke=False)
    assert seen == [(EVAL_ID, "EVAL", reg)] and rec["n_users"] == 12792 and rec["manifest_id"] == EVAL_ID
    users[0] = 12000
    assert_raises(D.LaunchRefused, D.verify_selection_manifest, sel, reg, kw, smoke=False)
    assert D.verify_selection_manifest(sel, reg, kw, smoke=True)["users_checked"] is False
    other = D.load_selection(_selection(tmp_path, content="a" * 64))
    assert_raises(D.LaunchRefused, D.verify_selection_manifest, other, reg, kw, smoke=True)


# ------------------------------------------------------------------------------------------------ negative controls
@nc("the pre-F7 preflight rule (check_role only) lets a non-smoke LO MAIN run through without a selection")
def test_nc_lo_main_without_selection_passes_check_role():
    def mutant(role):                                                  # MUTANT: check_role alone
        D.check_role(role, select=False, arm="LO", selection=None)
    assert_raises(D.LaunchRefused, mutant, "MAIN")


@nc("the pre-F6 load_selection applies a selection made on another eval manifest")
def test_nc_selection_on_other_manifest_applies(tmp_path):
    p = _selection(tmp_path, content="b" * 64)
    assert_raises(D.LaunchRefused, D.load_selection, p)                # MUTANT: no manifest re-verification
