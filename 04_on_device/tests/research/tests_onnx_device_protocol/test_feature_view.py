"""DEV-04: the ONNX protocol builds a checkpoint with its OWN registered feature view (v4dev.checkpointio.
resolve_feature_view), the rows must carry that view (v4dev.dump_rows.check_rows_view), and every mismatch refuses.

Fixture: a C_SMALL-shaped v4train run directory <run>/sealed/checkpoints/best.pt + <run>/RUN_START.json, the model
built exactly like the trainer (v4models.build with the REGISTERED derived schema
L3S_NEAR_DUPLICATE_SESSION-minus-G11+G12a+G3+G4+G7 from the local contracts/derived_schemas registry, variant
SASREC_D64_B2, seed 2026), a tiny synthetic catalogue (K_SMALL), weights perturbed away from theta0 so a wrong load
would show. No dataset rows, no MAIN/TEST values. The onnxruntime pipeline test importorskips (server-only)."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from dev_testkit import K_SMALL, SCHEMA, assert_raises, nc
from fl.numerics import state_digest
from v4dev.bench import check_bench_rows, load_rows_npz, production_scores_batch, pytorch_scores_batch
from v4dev.checkpointio import (CheckpointIOError, build_scoring_module, resolve_feature_view,
                                view_required_row_keys)
from v4dev.dump_rows import DumpRowsError, batch_view_of, check_rows_view, dump_from_provider, write_rows
from v4dev.scoring import example_batch
from v4eval.ranking import rank_from_scores
from v4fs.schemas import REGISTRY_PATH, load_registered
from v4models import build as build_model
from v4models.schema import NAME_KEYS

VIEW_ID = "L3S_NEAR_DUPLICATE_SESSION-minus-G11+G12a+G3+G4+G7"
VIEW = load_registered(VIEW_ID, REGISTRY_PATH)
EXT_KEYS = ("session_offset_tokens", "session_mode_main_category_tokens", "current_matches_session_mode",
            "session_mode_share", "current_matches_session_mode_available")
DROPPED_TOKENS = ("main_category_tokens", "daypart_tokens", "weekday_tokens")
VARIANT = {"SASREC": "SASREC_D64_B2", "GRU": "GRU_D128"}
EVAL_ID, HIST = "MAIN_VALIDATION_SYN", "USER_HISTORY_L50"


def _sha(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def make_run(tmp_path, family="SASREC", *, schema=VIEW, seed=2026, meta_over=None, start_over=None,
             perturb=True, registry_sha=None) -> dict:
    """A v4train-shaped run directory whose best.pt was built with `schema` (trainer construction)."""
    built = build_model(family, schema, seed, "cpu", K=K_SMALL, variant=VARIANT[family])
    init = built.init_sha256
    if perturb:
        g = torch.Generator().manual_seed(7)
        with torch.no_grad():
            for p in built.module.parameters():
                p.add_(0.05 * torch.randn(p.shape, generator=g))
    sd = dict(built.module.state_dict())
    run = tmp_path / f"run_{family}"
    ck = run / "sealed" / "checkpoints"
    ck.mkdir(parents=True)
    sid, ssh = schema.schema_id, schema.self_sha256
    meta = {"family": family, "model_variant": VARIANT[family], "seed": seed, "config_sha256": "c" * 64,
            "init_sha256": init, "state_digest": state_digest(built.adapter.broadcast_state()),
            "schema": {"schema_id": sid, "schema_self_sha256": ssh}}
    meta.update(meta_over or {})
    torch.save({"kind": "best", "meta": meta, "model": sd}, str(ck / "best.pt"))
    derived = getattr(schema, "is_derived_schema", False)
    start = {"family": family, "model_variant": VARIANT[family], "seed": seed, "config_sha256": "c" * 64,
             "init_sha256": init, "config": {"family": family, "model_variant": VARIANT[family], "seed": seed},
             "schema": {"schema_id": sid, "schema_self_sha256": ssh, "kind": "DERIVED" if derived else "BASE",
                        "file_sha256": schema.file_sha256,
                        "registry": ({"path": str(REGISTRY_PATH), "file_sha256": registry_sha or _sha(REGISTRY_PATH)}
                                     if derived else None)},
             "key_base": {"eval_manifest_id": EVAL_ID, "history_policy": HIST}}
    start.update(start_over or {})
    (run / "RUN_START.json").write_text(json.dumps(start), encoding="utf-8")
    return {"ckpt": ck / "best.pt", "built": built, "state": sd, "run": run}


def load(run, family="SASREC", **kw):
    return build_scoring_module(family, variant=kw.pop("variant", VARIANT[family]), ckpt_path=run["ckpt"],
                                k=K_SMALL, **kw)


def rows_for(made, n=12, seed=3) -> dict:
    """--rows-shaped numpy rows (the FULL Clean V2 batch + extension keys, as dump_rows writes them)."""
    b = example_batch(made["built"], n=n, seed=seed, min_len=1)
    rows = {k: v.numpy() for k, v in b.items() if torch.is_tensor(v)}
    rows["decision_id"] = np.arange(500, 500 + n, dtype=np.int64)
    rows["manifest_row"] = np.arange(n, dtype=np.int64) * 2
    return rows


def good_meta(schema=VIEW, **over) -> dict:
    base = schema.base if getattr(schema, "is_derived_schema", False) else schema
    bw = base.widths
    bv = {"feature_view": bw.view_id, "clean_v2_schema_self_sha256": base.self_sha256}
    bv.update({NAME_KEYS[k]: list(bw.names_of(k)) for k in NAME_KEYS if bw.names_of(k)})
    m = {"manifest_id": EVAL_ID, "history_policy": HIST, "manifest_n": 40, "batch_view": bv,
         "feature_view_binding": {"schema_id": schema.schema_id, "schema_self_sha256": schema.self_sha256}}
    m.update(over)
    return m


def ranks_of(scores: np.ndarray, target: np.ndarray) -> np.ndarray:
    return rank_from_scores(torch.as_tensor(scores, dtype=torch.float64), torch.as_tensor(target)).ranks.numpy()


# --------------------------------------------------------------------------------------------- the view loads
def test_registered_view_is_the_c_small_layout():
    """The fixture view is the C_SMALL one (the INT8_BENCH.log shapes: numflag 9, user 10, ext side channels)."""
    w = VIEW.widths
    assert (w.numeric_dim + w.flags_dim, w.user_context_dim + w.user_mask_dim) == (9, 10)
    assert {"session_offset_tokens", "session_mode_main_category_tokens"} <= set(w.categorical)
    assert not set(DROPPED_TOKENS) & set(w.categorical)


@pytest.mark.parametrize("family", ["SASREC", "GRU"])
def test_extension_checkpoint_loads_with_its_own_view(tmp_path, family):
    run = make_run(tmp_path, family)
    made = load(run, family)
    fv = made["feature_view"]
    assert fv["schema_id"] == VIEW_ID and fv["kind"] == "DERIVED"
    assert fv["schema_self_sha256"] == VIEW.self_sha256
    assert set(fv["sources"]) == {"checkpoint_meta", "RUN_START"}
    assert fv["init_sha256"] == run["built"].init_sha256
    assert fv["eval_manifest_id"] == EVAL_ID and fv["history_policy"] == HIST
    got = made["built"].module.state_dict()
    for k, v in run["state"].items():
        assert torch.equal(got[k], v), k
    keys = made["scoring"].input_keys
    assert all(k in keys for k in EXT_KEYS)
    assert not set(DROPPED_TOKENS) & set(keys)


@pytest.mark.parametrize("family", ["SASREC", "GRU"])
def test_extension_view_wrapper_has_fp32_parity_ranks_with_production(tmp_path, family):
    """The ONNX-traced wrapper (padded_reference / dense GRU, projection inside) gives the production forward's
    (adapter.query, SASRec impl unpadded) scores to FP32 tolerance and IDENTICAL full-catalogue ranks."""
    run = make_run(tmp_path, family)
    made = load(run, family)
    rows = rows_for(made, n=16)
    keys = made["scoring"].input_keys
    tr = [{k: torch.as_tensor(rows[k][i:i + 1]) for k in keys} for i in range(16)]
    wrap = pytorch_scores_batch(made["scoring"], tr)
    prod = production_scores_batch(made["built"], tr)
    assert np.max(np.abs(wrap - prod)) < 1e-4
    assert np.array_equal(ranks_of(wrap, rows["target_class"]), ranks_of(prod, rows["target_class"]))


def test_extension_inputs_change_the_scores(tmp_path):
    """The extension channels are real inputs of the scoring graph (not silently ignored)."""
    made = load(make_run(tmp_path))
    rows = rows_for(made, n=4)
    keys = made["scoring"].input_keys
    a = pytorch_scores_batch(made["scoring"], [{k: torch.as_tensor(rows[k][i:i + 1]) for k in keys} for i in range(4)])
    rows["session_offset_tokens"] = np.where(rows["session_offset_tokens"] > 0, 8, 0)
    b = pytorch_scores_batch(made["scoring"], [{k: torch.as_tensor(rows[k][i:i + 1]) for k in keys} for i in range(4)])
    assert np.max(np.abs(a - b)) > 1e-6


def test_base_view_checkpoint_unchanged(tmp_path):
    """A base-schema run still builds the base view with the DEV-01 signature (no extension keys)."""
    run = make_run(tmp_path, "SASREC", schema=SCHEMA)
    made = load(run)
    assert made["feature_view"]["kind"] == "BASE"
    from fl.families import SASREC_KEYS
    assert made["scoring"].input_keys == tuple(SASREC_KEYS)


# --------------------------------------------------------------------------------------------- refusals
def test_default_view_refuses_the_extension_checkpoint(tmp_path):
    """The DEV-01 failure: the base Clean V2 build cannot take the extension weights (bare state dict, no record)."""
    run = make_run(tmp_path)
    bare = tmp_path / "bare.pt"
    torch.save(run["state"], str(bare))
    assert_raises(CheckpointIOError, build_scoring_module, "SASREC", variant="SASREC_D64_B2", ckpt_path=bare,
                  k=K_SMALL)


def test_explicit_schema_id_must_equal_the_record(tmp_path):
    run = make_run(tmp_path)
    assert_raises(CheckpointIOError, load, run, schema_id=SCHEMA.schema_id)
    assert load(run, schema_id=VIEW_ID)["feature_view"]["schema_id"] == VIEW_ID


def test_explicit_schema_with_ckpt_refuses(tmp_path):
    run = make_run(tmp_path)
    assert_raises(CheckpointIOError, build_scoring_module, "SASREC", variant="SASREC_D64_B2", ckpt_path=run["ckpt"],
                  k=K_SMALL, schema=SCHEMA)


def test_meta_and_run_start_view_disagreement_refuses(tmp_path):
    run = make_run(tmp_path, start_over={"schema": {"schema_id": SCHEMA.schema_id,
                                                    "schema_self_sha256": SCHEMA.self_sha256}})
    assert_raises(CheckpointIOError, load, run)


def test_self_hash_mismatch_refuses(tmp_path):
    run = make_run(tmp_path, meta_over={"schema": {"schema_id": VIEW_ID, "schema_self_sha256": "0" * 64}},
                   start_over={"schema": {"schema_id": VIEW_ID, "schema_self_sha256": "0" * 64}})
    assert_raises(CheckpointIOError, load, run)


def test_registry_file_sha_mismatch_refuses(tmp_path):
    assert_raises(CheckpointIOError, load, make_run(tmp_path, registry_sha="f" * 64))


def test_variant_mismatch_refuses(tmp_path):
    assert_raises(CheckpointIOError, load, make_run(tmp_path), variant="SASREC_D256_B2")


def test_init_sha_mismatch_refuses(tmp_path):
    run = make_run(tmp_path, meta_over={"init_sha256": "1" * 64}, start_over={"init_sha256": "1" * 64})
    assert_raises(CheckpointIOError, load, run)


def test_state_digest_mismatch_refuses(tmp_path):
    assert_raises(CheckpointIOError, load, make_run(tmp_path, meta_over={"state_digest": "2" * 64}))


def test_meta_run_start_identity_disagreement_refuses(tmp_path):
    assert_raises(CheckpointIOError, load, make_run(tmp_path, start_over={"seed": 2027}))


def test_family_mismatch_refuses(tmp_path):
    run = make_run(tmp_path)
    assert_raises(CheckpointIOError, build_scoring_module, "GRU", ckpt_path=run["ckpt"], k=K_SMALL)


def test_checkpoints_dir_without_run_start_refuses(tmp_path):
    run = make_run(tmp_path)
    (run["run"] / "RUN_START.json").unlink()
    assert_raises(CheckpointIOError, load, run)


def test_record_without_any_view_refuses(tmp_path):
    run = make_run(tmp_path)
    obj = torch.load(str(run["ckpt"]), weights_only=True)
    obj["meta"].pop("schema")
    p = tmp_path / "loose.pt"                      # not under checkpoints/: no RUN_START either
    torch.save(obj, str(p))
    assert_raises(CheckpointIOError, build_scoring_module, "SASREC", variant="SASREC_D64_B2", ckpt_path=p, k=K_SMALL)


# --------------------------------------------------------------------------------------------- the rows view
def test_rows_view_accepts_the_matching_rows(tmp_path):
    made = load(make_run(tmp_path))
    rec = check_bench_rows(made, rows_for(made), good_meta())
    assert rec["schema_id"] == VIEW_ID and rec["batch_view_checked"]
    assert set(EXT_KEYS) <= set(rec["required_row_keys"])


def test_required_row_keys_are_the_full_base_batch_plus_extensions():
    need = view_required_row_keys(VIEW)
    assert set(EXT_KEYS) <= set(need)
    assert not set(DROPPED_TOKENS) & set(need)
    assert {"event_numeric_features", "event_quality_flags", "user_context", "user_context_masks"} <= set(need)


@pytest.mark.parametrize("breaker", ["missing_ext", "projected_width", "no_batch_view", "name_order", "view_id",
                                     "base_self", "binding", "manifest", "history"])
def test_rows_view_refuses(tmp_path, breaker):
    made = load(make_run(tmp_path))
    rows, meta = rows_for(made), good_meta()
    if breaker == "missing_ext":
        rows.pop("session_mode_share")
    elif breaker == "projected_width":
        rows["event_numeric_features"] = rows["event_numeric_features"][..., :7]
    elif breaker == "no_batch_view":
        meta.pop("batch_view")
    elif breaker == "name_order":
        names = meta["batch_view"]["event_numeric_feature_names"]
        names[0], names[1] = names[1], names[0]
    elif breaker == "view_id":
        meta["batch_view"]["feature_view"] = "V1"
    elif breaker == "base_self":
        meta["batch_view"]["clean_v2_schema_self_sha256"] = "3" * 64
    elif breaker == "binding":
        meta["feature_view_binding"] = {"schema_id": SCHEMA.schema_id, "schema_self_sha256": SCHEMA.self_sha256}
    elif breaker == "manifest":
        meta["manifest_id"] = "OTHER_EVAL"
    elif breaker == "history":
        meta["history_policy"] = "SESSION_ONLY"
    assert_raises(DumpRowsError, check_bench_rows, made, rows, meta)


def test_base_view_rows_need_no_batch_view(tmp_path):
    """Identity projection (base view): legacy rows without a meta still pass the key / width checks."""
    made = load(make_run(tmp_path, schema=SCHEMA))
    check_bench_rows(made, rows_for(made), None)


# --------------------------------------------------------------------------------------------- dump_rows records the view
class _ViewProvider:
    def __init__(self, manifest, rows, bview, *, change_at=None):
        self.manifest, self._rows, self._bview, self._change = manifest, rows, bview, change_at
        self.eval_batch, self.history_policy, self.manifest_id = 5, HIST, manifest.manifest_id
        self.manifest_binding = {"content_sha256": "d" * 64}

    def batches(self, bs):
        n = self._rows["lengths"].shape[0]
        for i, s in enumerate(range(0, n, bs)):
            bv = dict(self._bview)
            if self._change is not None and i >= self._change:
                bv["feature_view"] = "OTHER"
            yield dict({k: v[s:s + bs] for k, v in self._rows.items()}, **bv)


def _manifest(n):
    from v4eval.manifest import build_eval_manifest
    cols = {"decision_id": np.arange(500, 500 + n, dtype=np.int64), "eval_user_key": (np.arange(n) % 3).astype(np.int64),
            "eval_mask": np.ones(n, dtype=bool), "target_class": (np.arange(n) % K_SMALL).astype(np.int64),
            "target_oov": np.zeros(n, dtype=bool), "censor_class": np.array(["NONE"] * n)}
    return build_eval_manifest({"catalogue_K": K_SMALL, "manifest_id": EVAL_ID}, cols)


def test_dump_rows_records_the_batch_view_and_bench_accepts_it(tmp_path):
    made = load(make_run(tmp_path))
    rows = rows_for(made, n=12)
    rows.pop("manifest_row")
    rows["decision_id"] = np.arange(500, 512, dtype=np.int64)
    bview = good_meta()["batch_view"]
    out, meta = dump_from_provider(_ViewProvider(_manifest(12), rows, bview))
    assert meta["batch_view"] == bview == batch_view_of(dict(rows, **bview))
    assert meta["fields"]["session_offset_tokens"]["row_shape"] == [rows["session_offset_tokens"].shape[1]]
    w = write_rows(out, dict(meta, history_policy=HIST), tmp_path / "d" / "rows.npz")
    back = load_rows_npz(w["rows_path"])
    disk_meta = json.loads(Path(w["meta_path"]).read_text())
    check_bench_rows(made, back, disk_meta)


def test_dump_rows_refuses_a_changing_batch_view(tmp_path):
    made = load(make_run(tmp_path))
    rows = rows_for(made, n=12)
    rows.pop("manifest_row")
    assert_raises(DumpRowsError, dump_from_provider, _ViewProvider(_manifest(12), rows, good_meta()["batch_view"],
                                                                   change_at=1))


# --------------------------------------------------------------------------------------------- negative controls
def _refuses(fn, *a, **kw) -> bool:
    try:
        fn(*a, **kw)
    except (CheckpointIOError, DumpRowsError):
        return True
    return False


@nc("the default Clean V2 build must NOT accept the extension-view weights (the DEV-01 bench bug)")
def test_nc_default_view_would_load_the_extension_checkpoint(tmp_path):
    run = make_run(tmp_path)
    bare = tmp_path / "bare.pt"
    torch.save(run["state"], str(bare))
    assert not _refuses(build_scoring_module, "SASREC", variant="SASREC_D64_B2", ckpt_path=bare, k=K_SMALL)


@nc("a non-identity view's rows without column-order provenance must be refused, not accepted")
def test_nc_rows_without_batch_view_would_pass(tmp_path):
    made = load(make_run(tmp_path))
    meta = good_meta()
    meta.pop("batch_view")
    assert not _refuses(check_bench_rows, made, rows_for(made), meta)


@nc("the extension-view scoring signature must carry the extension keys")
def test_nc_signature_would_lack_extension_keys(tmp_path):
    made = load(make_run(tmp_path))
    assert "session_offset_tokens" not in made["scoring"].input_keys


@nc("a disagreeing RUN_START view must refuse, not be ignored in favour of the checkpoint meta")
def test_nc_disagreeing_run_start_would_be_ignored(tmp_path):
    run = make_run(tmp_path, start_over={"schema": {"schema_id": SCHEMA.schema_id,
                                                    "schema_self_sha256": SCHEMA.self_sha256}})
    assert not _refuses(load, run)


@nc("a tampered theta0 record (init_sha256) must refuse")
def test_nc_init_sha_would_not_be_checked(tmp_path):
    run = make_run(tmp_path, meta_over={"init_sha256": "1" * 64}, start_over={"init_sha256": "1" * 64})
    assert not _refuses(load, run)


# --------------------------------------------------------------------------------------------- ONNX (server-only)
def test_full_pipeline_extension_view_fp32_onnx_ranks_equal_production(tmp_path):
    pytest.importorskip("onnx", reason="onnx is a server-only dependency (not in the local dev venv)")
    pytest.importorskip("onnxruntime", reason="onnxruntime is a server-only dependency (not in the local dev venv)")
    from v4dev.bench import run_pipeline
    run = make_run(tmp_path)
    made = load(run)
    n = 10
    rows = rows_for(made, n=n, seed=11)
    rows["target_class"][3] = -1                                          # one OOV row (credit 0, skipped)
    write_rows(rows, good_meta(manifest_n=2 * n), tmp_path / "rows" / "rows.npz")
    args = argparse.Namespace(ckpt=str(run["ckpt"]), family="SASREC", variant="SASREC_D64_B2",
                              out=str(tmp_path / "out"), seed=2026, v1_root=None, catalogue=None, k=K_SMALL,
                              n_queries=8, warmup=2, intra_threads=1, rows=str(tmp_path / "rows" / "rows.npz"),
                              manifest_n=2 * n, tag="onnx_int8", schema_id=None, schema_registry=None)
    result = run_pipeline(args)
    assert result["feature_view"]["schema_id"] == VIEW_ID
    assert result["rows_view_check"]["batch_view_checked"]
    assert set(EXT_KEYS) & set(result["onnx_input_names"])
    keys = made["scoring"].input_keys
    tr = [{k: torch.as_tensor(rows[k][i:i + 1]) for k in keys} for i in range(n)]
    prod = production_scores_batch(made["built"], tr)
    want = np.full(2 * n, -1, dtype=np.int64)
    ok = rows["target_class"] >= 0
    want[rows["manifest_row"][ok]] = ranks_of(prod[ok], rows["target_class"][ok])
    fp32 = np.load(result["retention"]["onnx_fp32"]["ranks_path"])
    assert np.array_equal(fp32, want)                                     # FP32 ONNX ranks = production ranks
    int8 = np.load(result["retention"]["onnx_int8"]["ranks_path"])
    assert int8.shape == fp32.shape and np.array_equal(int8 < 0, fp32 < 0)
