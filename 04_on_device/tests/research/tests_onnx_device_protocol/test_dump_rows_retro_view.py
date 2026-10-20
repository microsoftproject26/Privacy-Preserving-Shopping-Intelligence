"""DEV-05: the TEST-row dump (v4int.dump_rows_retro) carries the DEV-04 view binding, and its rows are exactly r6's.

The r6 reference is the FROZEN r6 code (tests_dev/r6_ref/*.py.txt, sha256-pinned below: code_ds3r6/v4dev/dump_rows.py and
code_ds3r6/v4int/dump_rows_retro.py, byte-equal to the server tree). On a synthetic RETRO_TEST-shaped provider that
yields the full Clean V2 batch of the C_SMALL view plus every provenance field a real StoreEvalProvider batch carries,
the r7b `dump_retro` and the r6 main() composition write NPZs whose members are byte-identical; the meta differs only
by the added keys. No real TEST row, label or value is read (synthetic rows only)."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import zipfile
from importlib.machinery import SourceFileLoader
from pathlib import Path

import numpy as np
import pytest
import torch

from dev_testkit import assert_raises, nc
from test_feature_view import HIST, VIEW, good_meta, load, make_run, rows_for
from v4dev.bench import check_bench_rows, load_rows_npz
from v4dev.dump_rows import DumpRowsError, write_rows
from v4eval.manifest import build_eval_manifest
from v4int import dump_rows_retro as DR

HERE = Path(__file__).resolve().parent
R6_SHA = {"dump_rows_r6.py.txt": "4696c4af329813e9984c15e5658ef2fe6704ea06643ec0b26ebbbc73c7cf3cc9",
          "dump_rows_retro_r6.py.txt": "9853cc554799542605c3e0e896a895ba10b55b5ca7a8ce030b338f72a0844889"}
TEST_ID = "MAIN_TEST_SYN"
ADDED_META = {"batch_view", "fields", "feature_view_binding"}
VOLATILE = {"rows_npz_sha256", "rows_path"}


def _r6(name: str):
    p = HERE / "r6_ref" / name
    assert hashlib.sha256(p.read_bytes()).hexdigest() == R6_SHA[name], f"{p} is not the frozen r6 file"
    mod_name = "r6ref_" + name.split(".")[0]
    loader = SourceFileLoader(mod_name, str(p))
    spec = importlib.util.spec_from_loader(mod_name, loader)
    mod = importlib.util.module_from_spec(spec)
    loader.exec_module(mod)
    return mod


R6_DUMP = _r6("dump_rows_r6.py.txt")
R6_RETRO = _r6("dump_rows_retro_r6.py.txt")


class _RetroProvider:
    """RETRO_TEST-shaped StoreEvalProvider stand-in: full Clean V2 batch tensors (C_SMALL view + extensions) plus the
    provenance fields of a real batch (str scalars, name tuples, a per-batch counts dict, a str ndarray)."""

    def __init__(self, manifest, rows: dict, bview: dict, *, eval_batch=5, change_at=None, drop=()):
        self.manifest, self._rows, self._bview = manifest, rows, bview
        self.eval_batch, self.history_policy, self.manifest_id = eval_batch, HIST, manifest.manifest_id
        self.manifest_binding = {"content_sha256": "e" * 64}
        self._change, self._drop = change_at, set(drop)

    def batches(self, bs):
        n = self._rows["lengths"].shape[0]
        for i, s in enumerate(range(0, n, bs)):
            b = {k: torch.as_tensor(v[s:s + bs]) for k, v in self._rows.items() if k not in self._drop}
            bv = {k: (tuple(v) if isinstance(v, list) else v) for k, v in self._bview.items()}
            if self._change is not None and i >= self._change:
                bv["feature_view"] = "OTHER_VIEW"
            m = int(b["lengths"].shape[0])
            yield dict(b, **bv, clean_v2_rc_counts={"rc0": i, "rc1": m}, clean_v2_state_sha256="5" * 64,
                       m1_application_path="EXACT_INVERSE", channel_labels=np.array(["x"] * m))


def _world(tmp_path, n=15, **run_kw):
    run = make_run(tmp_path, **run_kw)
    made = load(run)
    rows = rows_for(made, n=n, seed=21)
    rows.pop("manifest_row")
    euk = (np.arange(n) % 4).astype(np.int64)
    cols = {"decision_id": rows["decision_id"].astype(np.int64), "eval_user_key": euk,
            "eval_mask": np.ones(n, dtype=bool), "target_class": rows["target_class"].astype(np.int64),
            "target_oov": np.zeros(n, dtype=bool), "censor_class": np.array(["NONE"] * n)}
    man = build_eval_manifest({"catalogue_K": int(made["built"].K), "manifest_id": TEST_ID}, cols)
    hexes = [hashlib.sha256(f"test-user-{i}".encode()).hexdigest() for i in range(4)]
    pmeta = {"eval_manifest_id": TEST_ID, "manifest_dir": "/synthetic", "lock_self_sha256": "6" * 64,
             "data_version": "SYN", "split": "RETRO_TEST", "test_gate": {"entry": 2},
             "test_exposure_register_entry_id": 2, "registry_ext": {"path": "/synthetic/ext.json", "self_sha256": "7" * 64}}
    return {"run": run, "made": made, "rows": rows, "manifest": man, "hexes": hexes, "pmeta": pmeta,
            "bview": good_meta()["batch_view"]}


def _provider(w, **kw):
    return _RetroProvider(w["manifest"], w["rows"], w["bview"], **kw)


def r6_dump(w, out: Path, first_users=None) -> dict:
    """The r6 main() body, verbatim, with the frozen r6 modules."""
    p = _provider(w)
    view = R6_RETRO.ArrayFieldsView(p)
    rows, meta = R6_DUMP.dump_from_provider(view, first_users=first_users,
                                            client_key_hex=w["hexes"] if first_users is not None else None)
    meta.update(w["pmeta"], dropped_non_array_fields=sorted(view.dropped))
    meta["role"] = "MAIN_TEST"
    return R6_DUMP.write_rows(rows, meta, out)


def r7b_dump(w, out: Path, first_users=None, *, bind=True, **pkw) -> dict:
    ck = w["run"]["ckpt"]
    view = DR.resolve_ckpt_view(ck, "SASREC", "MAIN_TEST", HIST) if bind else None
    rows, meta = DR.dump_retro(_provider(w, **pkw), w["pmeta"], "MAIN_TEST", first_users=first_users,
                               client_key_hex=w["hexes"] if first_users is not None else None,
                               view=view, ckpt=ck if bind else None)
    return write_rows(rows, meta, out)


def _members(npz: Path) -> dict:
    with zipfile.ZipFile(npz) as z:
        return {n: z.read(n) for n in z.namelist()}


# --------------------------------------------------------------------------------------------- r6 identity
@pytest.mark.parametrize("first_users", [None, 2])
def test_test_rows_are_byte_identical_to_r6(tmp_path, first_users):
    w = _world(tmp_path)
    a = r6_dump(w, tmp_path / "r6" / "rows.npz", first_users)
    b = r7b_dump(w, tmp_path / "r7b" / "rows.npz", first_users)
    ma, mb = _members(a["rows_path"]), _members(b["rows_path"])
    assert list(ma) == list(mb)                                   # same arrays, same order
    assert all(ma[k] == mb[k] for k in ma), [k for k in ma if ma[k] != mb[k]]
    assert set(b["meta"]) - set(a["meta"]) == ADDED_META and not set(a["meta"]) - set(b["meta"])
    assert {k: v for k, v in a["meta"].items() if k not in VOLATILE} == \
        {k: v for k, v in b["meta"].items() if k not in VOLATILE | ADDED_META}
    if first_users is not None:
        assert b["meta"]["n_users"] == 2


def test_r7b_meta_carries_the_view_binding(tmp_path):
    w = _world(tmp_path)
    m = r7b_dump(w, tmp_path / "d" / "rows.npz")["meta"]
    assert m["batch_view"] == w["bview"]
    fb = m["feature_view_binding"]
    assert fb["schema_id"] == VIEW.schema_id and fb["schema_self_sha256"] == VIEW.self_sha256
    assert fb["split"] == "RETRO_TEST" and fb["test_role"] == "MAIN_TEST"
    assert fb["test_exposure_register_entry_id"] == 2 and fb["run_eval_manifest_id"] == "MAIN_VALIDATION_SYN"
    assert fb["checkpoint_sha256"] == hashlib.sha256(Path(w["run"]["ckpt"]).read_bytes()).hexdigest()
    assert m["manifest_id"] == TEST_ID and m["split"] == "RETRO_TEST"


def test_bench_accepts_the_bound_test_rows(tmp_path):
    w = _world(tmp_path)
    out = r7b_dump(w, tmp_path / "d" / "rows.npz")
    disk = json.loads(Path(out["meta_path"]).read_text(encoding="utf-8"))
    rec = check_bench_rows(w["made"], load_rows_npz(out["rows_path"]), disk, ckpt_path=w["run"]["ckpt"])
    assert rec["split"] == "RETRO_TEST" and rec["checkpoint_bound"] and rec["batch_view_checked"]


def test_unbound_retro_dump_still_records_the_batch_view(tmp_path):
    w = _world(tmp_path)
    m = r7b_dump(w, tmp_path / "d" / "rows.npz", bind=False)["meta"]
    assert m["batch_view"] == w["bview"] and "feature_view_binding" not in m


# --------------------------------------------------------------------------------------------- refusals
def test_bench_refuses_the_r6_test_dump(tmp_path):
    """The OPS-ENDGAME-2 / Q13 symptom: r6 TEST rows (no batch_view, no binding) are refused by the r7 bench."""
    w = _world(tmp_path)
    a = r6_dump(w, tmp_path / "r6" / "rows.npz")
    assert_raises(DumpRowsError, check_bench_rows, w["made"], load_rows_npz(a["rows_path"]), a["meta"],
                  ckpt_path=w["run"]["ckpt"])


def _bound(tmp_path):
    w = _world(tmp_path)
    out = r7b_dump(w, tmp_path / "d" / "rows.npz")
    return w, load_rows_npz(out["rows_path"]), json.loads(Path(out["meta_path"]).read_text(encoding="utf-8"))


@pytest.mark.parametrize("breaker", ["other_ckpt", "no_ckpt_arg", "no_binding", "binding_split", "meta_split",
                                     "entry_id", "run_eval_manifest", "history", "batch_view"])
def test_bench_refuses_a_broken_test_binding(tmp_path, breaker):
    w, rows, meta = _bound(tmp_path)
    ck = w["run"]["ckpt"]
    fb = meta["feature_view_binding"]
    if breaker == "other_ckpt":
        fb["checkpoint_sha256"] = "8" * 64
    elif breaker == "no_ckpt_arg":
        ck = None
    elif breaker == "no_binding":
        meta.pop("feature_view_binding")
    elif breaker == "binding_split":
        fb.pop("split")
    elif breaker == "meta_split":
        meta["split"] = "VALIDATION"
    elif breaker == "entry_id":
        fb["test_exposure_register_entry_id"] = 3
    elif breaker == "run_eval_manifest":
        fb["run_eval_manifest_id"] = "OTHER_VALIDATION"
    elif breaker == "history":
        meta["history_policy"] = "SESSION_ONLY"
    elif breaker == "batch_view":
        meta["batch_view"]["user_context_names"] = list(reversed(meta["batch_view"]["user_context_names"]))
    assert_raises(DumpRowsError, check_bench_rows, w["made"], rows, meta, ckpt_path=ck)


def test_validation_rows_bound_to_another_checkpoint_are_refused(tmp_path):
    w = _world(tmp_path)
    meta = good_meta()
    meta["feature_view_binding"]["checkpoint_sha256"] = "9" * 64
    assert_raises(DumpRowsError, check_bench_rows, w["made"], rows_for(w["made"]), meta, ckpt_path=w["run"]["ckpt"])


def test_dump_retro_refuses_a_changing_batch_view(tmp_path):
    w = _world(tmp_path)
    assert_raises(DR.DumpRowsRetroRefused, r7b_dump, w, tmp_path / "d" / "rows.npz", change_at=1)


def test_dump_retro_refuses_rows_without_the_view_keys(tmp_path):
    w = _world(tmp_path)
    assert_raises(DumpRowsError, r7b_dump, w, tmp_path / "d" / "rows.npz", drop=("session_mode_share",))


def test_resolve_ckpt_view_refuses_history_and_role(tmp_path):
    w = _world(tmp_path)
    assert_raises(DR.DumpRowsRetroRefused, DR.resolve_ckpt_view, w["run"]["ckpt"], "SASREC", "MAIN_TEST",
                  "SESSION_ONLY")
    run = make_run(tmp_path / "fa", start_over={"key_base": {"eval_manifest_id": "MAIN_VALIDATION_SYN",
                                                             "history_policy": HIST, "manifest_role": "FA"}})
    assert_raises(DR.DumpRowsRetroRefused, DR.resolve_ckpt_view, run["ckpt"], "SASREC", "MAIN_TEST", HIST)
    run2 = make_run(tmp_path / "main", start_over={"key_base": {"eval_manifest_id": "MAIN_VALIDATION_SYN",
                                                                "history_policy": HIST, "manifest_role": "MAIN"}})
    assert DR.resolve_ckpt_view(run2["ckpt"], "SASREC", "MAIN_TEST", HIST)["record"]["schema_id"] == VIEW.schema_id


def test_cli_refuses_ckpt_without_family_before_the_gate(tmp_path):
    assert DR.main(["--role", "MAIN_TEST", "--history-policy", HIST, "--out", str(tmp_path / "r.npz"),
                    "--ckpt", str(tmp_path / "x.pt")]) == 2


# --------------------------------------------------------------------------------------------- negative controls
def _refused(fn, *a, **kw) -> bool:
    try:
        fn(*a, **kw)
    except (DumpRowsError, DR.DumpRowsRetroRefused):
        return True
    return False


@nc("the r6 TEST dump (no view provenance, no binding) must be refused by the r7 bench, not accepted")
def test_nc_r6_test_rows_would_pass_the_bench(tmp_path):
    w = _world(tmp_path)
    a = r6_dump(w, tmp_path / "r6" / "rows.npz")
    assert not _refused(check_bench_rows, w["made"], load_rows_npz(a["rows_path"]), a["meta"],
                        ckpt_path=w["run"]["ckpt"])


@nc("the r7b TEST dump must not change any TEST row array relative to r6")
def test_nc_r7b_changes_a_test_row(tmp_path):
    w = _world(tmp_path)
    ma = _members(r6_dump(w, tmp_path / "r6" / "rows.npz")["rows_path"])
    mb = _members(r7b_dump(w, tmp_path / "r7b" / "rows.npz")["rows_path"])
    assert any(ma[k] != mb.get(k) for k in ma)


@nc("TEST rows bound to another checkpoint must be refused")
def test_nc_test_rows_of_another_checkpoint_would_pass(tmp_path):
    w, rows, meta = _bound(tmp_path)
    meta["feature_view_binding"]["checkpoint_sha256"] = "8" * 64
    assert not _refused(check_bench_rows, w["made"], rows, meta, ckpt_path=w["run"]["ckpt"])


@nc("a TEST provider whose batch view changes mid-stream must be refused")
def test_nc_changing_batch_view_would_dump(tmp_path):
    w = _world(tmp_path)
    assert not _refused(r7b_dump, w, tmp_path / "d" / "rows.npz", change_at=1)
