"""v4int.dump_rows_retro (INT-07) - `v4dev.dump_rows --role MAIN_TEST`, as a wrapper (code/v4dev is another worker's
tree and is NOT edited): the RETRO_TEST E2E rows of the MAIN_TEST eval manifest as the `--rows` NPZ v4dev.bench reads.

    python -B code/v4int/dump_rows_retro.py --role MAIN_TEST --history-policy USER_HISTORY_L50 --out <dir>/rows.npz
        [--first-users 16384] [--registry-ext <REGISTRY_V2_TEST_EXT.json>] [--eval-prefetch 0]

Order: the DATA-09 TEST gate FIRST (v4int.test_roles.test_gate_ok; refused -> exit 2 before any file of the dataset or a
manifest is opened); then the DATA-09 loader wrappers (test_roles.install(role=None)), the role resolved under the gate
(test_roles.resolve_role), the RETRO_TEST StoreEvalProvider on the gated store / window path (v4int.retro_store), and
the UNCHANGED v4dev.dump_rows functions: dump_from_provider (same layout, same first-users rule: the first N E2E users in
selection_key_v2 order over client_key_hex, read from the RETRO_TEST manifest's own users table) and write_rows.
rows.meta.json additionally records role MAIN_TEST, split RETRO_TEST, the gate binding (all_methods_lock_sha256,
test_exposure_register_entry_id) and the TEST registry extension's self sha256. Only MAIN_TEST is served (the packet).
The batches handed to dump_from_provider keep their ARRAY fields only (ArrayFieldsView): a real StoreEvalProvider batch
also carries provenance scalars / name tuples that np.concatenate cannot stack (found on the synthetic RETRO_TEST
provider; the same holds for v4dev.dump_rows on a real VALIDATION provider - reported to its owner, not edited here).

DEV-05 (the DEV-04 view binding on the TEST path; r7b). The r7 bench refuses rows of a derived view without the
column-order provenance (v4dev.dump_rows.check_rows_view), and ArrayFieldsView drops exactly those name fields. Now:
  * ArrayFieldsView ALSO records each raw batch's view provenance (v4dev.dump_rows.batch_view_of: feature_view, the
    Clean V2 base self-hash, the four name arrays; identical in every batch, else refused) before filtering; the
    filtered batches, hence the rows, are unchanged. The meta gets `batch_view` (and v4dev's `fields`);
  * `--ckpt <best.pt> --family SASREC` binds the dump to the checkpoint's own registered view
    (v4dev.checkpointio.resolve_feature_view; the run's history policy must be --history-policy and its manifest role
    MAIN, of which MAIN_TEST is the TEST counterpart), runs check_rows_view on the rows (the eval-manifest equality is
    replaced by the TEST binding, since the rows come from the MAIN_TEST manifest) and writes `feature_view_binding`
    (v4dev.dump_rows.view_binding_record + split RETRO_TEST, test_role, the run's VALIDATION eval manifest id, the
    gate's register entry id); v4dev.bench requires that binding, to the same checkpoint sha256, for RETRO_TEST rows.
Which rows are dumped (selection, first-users rule, order, arrays) is unchanged: `dump_retro` is the r6 main() body
plus the meta additions (tests_dev/test_dump_rows_retro_view.py proves it against the frozen r6 code).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Mapping, Optional

CODE = Path(__file__).resolve().parents[1]
V4 = CODE.parent
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))
sys.dont_write_bytecode = True

ROLES = ("MAIN_TEST",)


class DumpRowsRetroRefused(RuntimeError):
    pass


def check_role(role: str) -> None:
    """Before any read: a served TEST role and an open TEST gate."""
    if role not in ROLES:
        raise DumpRowsRetroRefused(f"--role {role!r}: dump_rows_retro serves {ROLES} only")
    from v4int import test_roles as TR
    if not TR.test_gate_ok(role):
        raise DumpRowsRetroRefused(f"--role {role} refused: the DATA-09 TEST gate is closed")


class ArrayFieldsView:
    """The provider as v4dev.dump_rows.dump_from_provider reads it, with every batch reduced to its ARRAY fields
    (tensors / ndarrays with a row axis). A real StoreEvalProvider batch also carries provenance scalars (feature_view,
    the channel-name tuples, clean_v2_rc_counts, ...) that np.concatenate cannot stack; they are constant per provider
    and are recorded in the meta instead (`dropped_non_array_fields`)."""

    def __init__(self, provider):
        self._p = provider
        self.dropped: set = set()
        self.batch_view: Optional[dict] = None          # DEV-05: the raw batches' view provenance (constant)
        for k in ("manifest", "eval_batch", "manifest_id", "manifest_binding", "history_policy"):
            setattr(self, k, getattr(provider, k))

    @staticmethod
    def _is_array(v) -> bool:
        import numpy as np
        return hasattr(v, "shape") and len(getattr(v, "shape")) >= 1 and not isinstance(v, (str, bytes)) and             (hasattr(v, "numpy") or isinstance(v, np.ndarray))

    def batches(self, bs: int):
        from v4dev.dump_rows import batch_view_of
        for b in self._p.batches(bs):
            bv = batch_view_of(b)                          # DEV-05: recorded before the filter; rows unchanged
            if self.batch_view is None:
                self.batch_view = bv
            elif bv != self.batch_view:
                raise DumpRowsRetroRefused(f"provider batches disagree on the batch view: {self.batch_view} vs {bv}")
            keep = {k: v for k, v in b.items() if self._is_array(v)}
            self.dropped |= set(b) - set(keep)
            yield keep


def build_retro_eval_provider(role: str, history_policy: str, *, ext_path=None, eval_prefetch: int = 0, paths=None,
                              lock_path=None, register_path=None, **provider_kw):
    """(RETRO_TEST StoreEvalProvider, provider meta) of `role` under the gate (the v4dev.dump_rows.build_eval_provider
    analogue; no lock binding / trainer view is needed: the role comes from the gated extension)."""
    from v4int import retro_store as RS
    from v4int import test_roles as TR
    from v4int.providers import StoreEvalProvider
    from v4int.verify import DataPaths
    ext = Path(ext_path) if ext_path is not None else Path(TR.DEFAULT_EXT)
    TR.install(ext, role=None, eager=True, lock_path=lock_path, register_path=register_path)
    kw = {} if provider_kw.get("base_path") is None else {"base_path": provider_kw.pop("base_path")}
    res = TR.resolve_role(role, ext_path=ext, lock_path=lock_path, register_path=register_path, **kw)
    gate = res["gate"]
    paths = paths or DataPaths.default()
    root = provider_kw.pop("registry_root", V4)
    provider = StoreEvalProvider(res["eval"], history_policy=history_policy, paths=paths, prefetch=int(eval_prefetch),
                                 manifest_registry=ext, registry_root=root, test_gate=gate, **provider_kw)
    meta = {"eval_manifest_id": res["eval"], "manifest_dir": str(RS.retro_manifest_dir(ext, res["eval"], root)),
            "lock_self_sha256": provider.verified.lock_self_sha256, "data_version": res.get("data_version")
            or (provider.verified.registry or {}).get("data_version"), "split": RS.RETRO_TEST,
            "test_gate": gate.binding(), "test_exposure_register_entry_id": gate.entry_id,
            "registry_ext": {"path": str(ext), "self_sha256": res["registry_self_sha256"]}}
    return provider, meta


SERVED_RUN_ROLE = {"MAIN_TEST": "MAIN"}     # the TEST role -> the run manifest role it is the TEST counterpart of


def resolve_ckpt_view(ckpt, family: str, role: str, history_policy: str, *, schema_registry=None) -> dict:
    """DEV-05: the checkpoint's registered view (v4dev.checkpointio.resolve_feature_view), refused unless the run's
    history policy is `history_policy` and its manifest role is the one `role` is the TEST counterpart of."""
    from v4dev.checkpointio import resolve_feature_view
    view = resolve_feature_view(ckpt, family, schema_registry=schema_registry)
    hp = view["record"].get("history_policy")
    if hp is not None and hp != history_policy:
        raise DumpRowsRetroRefused(f"--history-policy {history_policy} != the checkpoint run's {hp}")
    mr = ((view.get("run_start") or {}).get("key_base") or {}).get("manifest_role")
    if mr is not None and mr != SERVED_RUN_ROLE[role]:
        raise DumpRowsRetroRefused(f"the checkpoint run's manifest role is {mr}; {role} is the TEST counterpart of "
                                   f"{SERVED_RUN_ROLE[role]} only")
    return view


def dump_retro(provider, pmeta: Mapping, role: str, *, first_users=None, client_key_hex=None,
               view: Optional[Mapping] = None, ckpt=None) -> tuple:
    """(rows, meta): the r6 main() body (ArrayFieldsView -> v4dev.dump_rows.dump_from_provider; meta + provider meta,
    dropped fields, role), plus DEV-05's `batch_view` and, when `view` (resolve_ckpt_view) is given, the checked
    `feature_view_binding`. The rows are exactly r6's."""
    from v4dev.dump_rows import check_rows_view, dump_from_provider, view_binding_record
    av = ArrayFieldsView(provider)
    rows, meta = dump_from_provider(av, first_users=first_users, client_key_hex=client_key_hex)
    meta.update(pmeta, dropped_non_array_fields=sorted(av.dropped))
    meta["role"] = role
    meta["batch_view"] = dict(av.batch_view or {})
    if view is not None:
        if ckpt is None:
            raise DumpRowsRetroRefused("a view binding needs the checkpoint path")
        rec = view["record"]
        checked = check_rows_view(rows, meta, view["schema"], require_batch_view=True,
                                  run_binding={"history_policy": rec.get("history_policy")})
        meta["feature_view_binding"] = view_binding_record(
            view, ckpt, checked, split=meta.get("split"), test_role=role,
            run_eval_manifest_id=rec.get("eval_manifest_id"),
            test_exposure_register_entry_id=meta.get("test_exposure_register_entry_id"))
    return rows, meta


def main(argv=None) -> int:
    from v4hist.policy import HISTORY_POLICIES
    ap = argparse.ArgumentParser(description="INT-07: dump the MAIN_TEST RETRO_TEST E2E rows for v4dev.bench (gated)")
    ap.add_argument("--role", required=True)
    ap.add_argument("--history-policy", required=True, choices=HISTORY_POLICIES)
    ap.add_argument("--out", required=True)
    ap.add_argument("--first-users", type=int, default=None)
    ap.add_argument("--registry-ext", type=Path, default=None)
    ap.add_argument("--eval-prefetch", type=int, default=0)
    ap.add_argument("--ckpt", default=None, help="DEV-05: bind the dump to this checkpoint's registered feature view")
    ap.add_argument("--family", default=None, choices=["GRU", "SASREC"], help="the --ckpt family (required with it)")
    ap.add_argument("--schema-registry", default=None, help="derived-schema registry override (default: RUN_START's)")
    args = ap.parse_args(argv)
    if (args.ckpt is None) != (args.family is None):
        print("DUMP_ROWS_RETRO REFUSED: --ckpt and --family must be given together", file=sys.stderr, flush=True)
        return 2
    try:
        check_role(args.role)
    except DumpRowsRetroRefused as e:
        print(f"DUMP_ROWS_RETRO REFUSED: {e}", file=sys.stderr, flush=True)
        return 2
    import v4train  # noqa: F401  (ppsi_env first)
    from v4dev.dump_rows import write_rows
    view = None
    if args.ckpt is not None:                      # before any TEST file is opened: refuse a wrong checkpoint early
        try:
            view = resolve_ckpt_view(args.ckpt, args.family, args.role, args.history_policy,
                                     schema_registry=args.schema_registry)
        except (DumpRowsRetroRefused, ValueError) as e:
            print(f"DUMP_ROWS_RETRO REFUSED: {e}", file=sys.stderr, flush=True)
            return 2
    provider, pmeta = build_retro_eval_provider(args.role, args.history_policy, ext_path=args.registry_ext,
                                                eval_prefetch=args.eval_prefetch)
    key_hex = None
    if args.first_users is not None:
        from v4base.identity import eval_client_key_hex_by_eval_user_key
        key_hex = eval_client_key_hex_by_eval_user_key(provider, manifest_dir=pmeta["manifest_dir"])
    rows, meta = dump_retro(provider, pmeta, args.role, first_users=args.first_users, client_key_hex=key_hex,
                            view=view, ckpt=args.ckpt)
    out = write_rows(rows, meta, Path(args.out))
    print(json.dumps(out["meta"], indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
