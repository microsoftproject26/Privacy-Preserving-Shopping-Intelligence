"""INT-07 test kit: a SYNTHETIC four-partition (TRAIN / VALIDATION_SELECT / VALIDATION_CONFIRM / TEST) frozen-layout F4
dataset that BOTH the DATA-09 RETRO_TEST builder and the snapshot CompactStore read, a tmp TEST gate, the DATA-09
synthetic RETRO_TEST build + registry extension in a tmp ppsi_v4-shaped workspace, and the gated provider stubs.

HARD RULE: no real RETRO_TEST row / label / feature / metric is ever read. Every TEST row here is fabricated:
  * users, ledger, decisions and the FEATURES_B columns = the DATA-09 synthetic population (code/tests_data09/conftest.py
    synthetic_users(), written by code/tests_data/data_testkit.write_dataset, both imported read-only by path);
  * the remaining F4 feature channels of every synthetic row (numeric / flags / user context / masks / daypart /
    weekday / prev-view count) are COPIED from the first real TRAIN rows of the frozen v1 bucket 0, read with a
    `split == "TRAIN"` scan filter (so no other partition is ever materialised) - only so that the per-row identity
    checks of v4data.transform.clean_v2_view (RC1-RC7) hold; the real normalizers.json (TRAIN statistics) is copied as is;
  * decisions_index/split=S/user_bucket=0 is written from the synthetic ledger (exactly as pipeline7 derives the index
    from the ledger: the same decision_id / target_class / loss_mask / eval_mask / position_in_segment).
"""
from __future__ import annotations

import copy
import importlib.util
import json
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as pads
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
CODE = HERE.parent
V4 = CODE.parent
if str(CODE / "tests_data") not in sys.path:
    sys.path.append(str(CODE / "tests_data"))

from intkit import REAL_DS  # noqa: E402

OOV_CLIENT = 1
_D9C = None


def d9c():
    """code/tests_data09/conftest.py imported read-only by path (unique module name; sys.path restored)."""
    global _D9C
    if _D9C is None:
        saved = list(sys.path)
        spec = importlib.util.spec_from_file_location("_int07_d9_conftest", str(CODE / "tests_data09" / "conftest.py"))
        mod = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(mod)
        finally:
            sys.path[:] = saved + [p for p in (str(CODE / "tests_data"), str(CODE / "tests_fs_data"))
                                   if p not in saved]
        _D9C = mod
    return _D9C


def have_real_bucket0() -> bool:
    return (Path(REAL_DS) / "model_ready" / "user_bucket=0" / "features.parquet").is_file() and \
        (Path(REAL_DS) / "mappings" / "normalizers.json").is_file()


def _donor_rows(n: int, cols: list) -> pa.Table:
    """The first n real TRAIN rows of bucket 0, F4 channels only (scan filter split == TRAIN)."""
    path = Path(REAL_DS) / "model_ready" / "user_bucket=0" / "features.parquet"
    t = pads.dataset(str(path), format="parquet").scanner(columns=cols + ["split"],
                                                            filter=pc.field("split") == "TRAIN").head(int(n))
    if t.num_rows < n or set(pc.unique(t["split"]).to_pylist()) != {"TRAIN"}:
        raise RuntimeError("donor scan did not return TRAIN rows only")
    return t.drop(["split"])


def write_f4_dataset(root: Path) -> Path:
    """The DATA-09 synthetic dataset + F4 channels + decisions_index + normalizers (module docstring)."""
    from full_store import _needed_feature_columns     # snapshot (ppsi_env)
    C = d9c()
    ds = C.write_synthetic_dataset(Path(root))
    led = pq.ParquetFile(ds / "decisions" / "user_bucket=0" / "ledger.parquet").read()
    fpath = ds / "model_ready" / "user_bucket=0" / "features.parquet"
    feat = pq.ParquetFile(fpath).read()
    assert feat["decision_id"].to_pylist() == led["decision_id"].to_pylist()
    n = feat.num_rows
    keep_syn = {"user_id", "segment_index", "position_in_segment", "decision_id", "split", "product_idx",
                "category_idx", "brand_idx", "main_category_idx"}
    f4 = [c for c in _needed_feature_columns("F4") if c not in keep_syn]
    donor = _donor_rows(n, f4)
    cat = pq.read_table(ds / "mappings" / "catalogue.parquet").to_pydict()
    pidx = dict(zip(cat["product_id"], cat["product_idx"]))
    cols = {name: feat[name] for name in feat.column_names}
    cols["session_index"] = led["session_index"]
    cols["product_idx"] = pa.array([int(pidx.get(p, 2)) for p in feat["product_id"].to_pylist()], pa.int64())
    for c in f4:
        cols[c] = donor[c]
    pq.write_table(pa.table(cols), fpath)
    cl = pq.read_table(ds / "mappings" / "clients.parquet").to_pydict()
    cidx = dict(zip(cl["user_id"], cl["client_idx"]))
    for split in ("TRAIN", "VALIDATION_SELECT", "VALIDATION_CONFIRM", "TEST"):
        t = led.filter(pc.equal(led["split"], split))
        d = ds / "decisions_index" / f"split={split}" / "user_bucket=0"
        d.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.table({
            "decision_id": t["decision_id"], "user_id": t["user_id"],
            "client_idx": pa.array([int(cidx.get(u, OOV_CLIENT)) for u in t["user_id"].to_pylist()], pa.int64()),
            "segment_index": t["segment_index"], "session_index": t["session_index"],
            "position_in_segment": t["position_in_segment"], "target_class": t["target_class"],
            "loss_mask": t["loss_mask"], "eval_mask": t["eval_mask"]}), d / "data_0.parquet")
    shutil.copyfile(Path(REAL_DS) / "mappings" / "normalizers.json", ds / "mappings" / "normalizers.json")
    return ds


# ------------------------------------------------------------------------------------------------ workspace
def relabel_full(root: Path, out_rel: str) -> None:
    """Test scaffolding (the DATA-09 registry-shim test's rule): header status = the TEST role (as a full build writes
    it), header content hash, .OK marker and build receipt recomputed."""
    from v4data.common import atomic_write_bytes, json_bytes_strict, sha256_file, write_marker
    from v4data09 import d9_common as D9
    from v4fs_data import fs_common as FC
    out = root / out_rel
    rp = out / "receipts" / "DATA09_BUILD_RECEIPT.json"
    rec = json.loads(rp.read_text(encoding="utf-8"))
    for mid in D9.TEST_IDS:
        hp = out / "manifests" / f"{mid}.header.json"
        h = json.loads(hp.read_text(encoding="utf-8"))
        h["status"] = D9.ROLE_OF_ID[mid]
        h.pop("header_content_sha256")
        h["header_content_sha256"] = FC.header_content_sha256(h)
        atomic_write_bytes(out, f"manifests/{mid}.header.json", json_bytes_strict(h))
        files = {f"manifests/{mid}.{t['table']}.parquet": t["file_sha256"] for t in h["tables"]}
        files[f"manifests/{mid}.header.json"] = sha256_file(hp)
        write_marker(out, f"manifests/{mid}.OK", files)
        rec["manifests"][mid] = FC.hashes_entry(h, h["tables"], sha256_file(hp))
    rec["full_build"] = True
    FC.write_json_self_hashed(out, "receipts/DATA09_BUILD_RECEIPT.json", {k: v for k, v in rec.items()
                                                                           if k != "receipt_sha256"})


def make_world(tmp: Path) -> SimpleNamespace:
    """ds (the synthetic F4 dataset), the tmp gate, the workspace with the synthetic RETRO_TEST build + extension."""
    from v4data09 import d9_common as D9
    from v4data09 import d9_finalize as F
    from v4data09.d9_gate import check_test_gate
    C = d9c()
    tmp = Path(tmp)
    ds = write_f4_dataset(tmp / "ds")
    lock, reg = C.make_gate_files(tmp / "gate")
    gate = check_test_gate(lock, reg)
    root = tmp / "ws"
    for rel in (F.BASE_REL, "contracts/DATA_PROTOCOL_LOCK_V2.json", "contracts/DATA_PROTOCOL_LOCK.json"):
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(V4 / rel, root / rel)
    C.run_synthetic_build(ds, root / D9.OUT_REL, gate)
    relabel_full(root, D9.OUT_REL)
    ext = F.build_extension(root, gate, D9.OUT_REL)
    F.write_extension(root, ext)
    return SimpleNamespace(ds=ds, lock=lock, register=reg, gate=gate, root=root, ext=ext,
                           ext_path=root / F.EXT_REL, base_path=root / F.BASE_REL,
                           mdir=root / D9.OUT_REL / "manifests")


def verification_stubs(w, mp) -> None:
    """The DATA-09 `verified` fixture's stubs (monkeypatch `mp`): the REGISTRY_V2 base verification returns the REAL
    registry and the REAL v1 lock except the two catalogue values the synthetic dataset cannot share (K and the
    candidate order hash); the transform-state check returns the registry's state. Everything TEST-specific runs."""
    import v4int.verify as VF
    from v4data.common import sha256_file
    from v4data09 import d9_common as D9
    from v4fs_data import fs_common as FC
    base = json.loads(w.base_path.read_text(encoding="utf-8"))
    v1 = json.loads((V4 / "contracts" / "DATA_PROTOCOL_LOCK.json").read_text(encoding="utf-8"))
    h = json.loads((w.mdir / f"{D9.MAIN_TEST_ID}.header.json").read_text(encoding="utf-8"))
    v1s = copy.deepcopy(v1)
    v1s["catalogue"]["K"] = h["catalogue_K"]
    v1s["population_and_manifests"]["manifest_hashes"]["candidate_order_hash"] = h["candidate_order_hash"]
    mp.setattr(VF, "verify_registry", lambda p, ls, lf, *, lock=None, root=None: (
        base, sha256_file(w.base_path), base["self_sha256"], (v1s, FC.LOCK_SELF_SHA256, FC.LOCK_FILE_SHA256)))
    ts = base["transform_state"]
    mp.setattr(VF, "verify_transform_state", lambda paths, lock: (ts["transform_state_sha256"], ts["file_sha256"]))


def support_key_standin(rep, manifest_id: str = "MAIN_TRAIN_5_100_131072") -> SimpleNamespace:
    """The support manifest's side of comparison_key_base for the synthetic world (the REGISTRY_V2 support manifests
    live on the server only): the same lock / registry / transform state as the verified RETRO_TEST provider and a
    synthetic population band identity. A stand-in for v4int.retro_store.train_key_ref."""
    vm = SimpleNamespace(lock_self_sha256=rep.verified.lock_self_sha256, registry=dict(rep.verified.registry),
                         header={"population_sha256_selection_key_v2": "SYNTHETIC_MAIN_BAND"})
    return SimpleNamespace(manifest_id=manifest_id, history_policy=rep.history_policy,
                           transform_state=dict(rep.transform_state), verified=vm,
                           manifest_binding={"content_sha256": "SYNTHETIC_SUPPORT"}, fingerprint=None, key_only=True)


def store_digest(store, ds) -> dict:
    """sha256 of every event array, decision-set field and session id of a subset store (bitwise identity)."""
    import hashlib
    from v4int.cache import DS_FIELDS, EVENT_ARRAYS
    out = {a: hashlib.sha256(np.ascontiguousarray(getattr(store.cs, a)).tobytes()).hexdigest() for a in EVENT_ARRAYS}
    out.update({f"ds.{f}": hashlib.sha256(np.ascontiguousarray(getattr(ds, f)).tobytes()).hexdigest()
                for f in DS_FIELDS})
    sid = getattr(store, "session_id", None)
    out["session_id"] = None if sid is None else hashlib.sha256(np.ascontiguousarray(sid).tobytes()).hexdigest()
    return out


def batch_digest(b: dict) -> str:
    import hashlib
    import torch
    h = hashlib.sha256()
    for k in sorted(b):
        v = b[k]
        if torch.is_tensor(v):
            h.update(k.encode() + str(v.dtype).encode() + str(tuple(v.shape)).encode())
            h.update(np.ascontiguousarray(v.numpy()).tobytes())
        elif isinstance(v, (str, int, float, bool, type(None))):
            h.update(k.encode() + repr(v).encode())
    return h.hexdigest()


__all__ = ["d9c", "have_real_bucket0", "write_f4_dataset", "make_world", "verification_stubs", "support_key_standin",
           "store_digest", "batch_digest", "relabel_full"]
