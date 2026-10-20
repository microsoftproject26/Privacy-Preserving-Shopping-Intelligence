"""Subprocess probe for test_d9_no_side_effect.py (not a test module): builds the LOCKED DATA-04 manifests
(v4data.manifests.ManifestBuilder, all eight pre-lock ids) and the DATA-06 v2 manifests (v4data06.d6_build.run_build) of
the synthetic dataset, in a FRESH interpreter, either plainly or after importing v4data09 + v4int.test_roles, running a
gated synthetic RETRO_TEST build and installing the test_roles wrappers in the same process. Prints the writer-
independent identities (table content / file sha256, header content sha256, ordered hashes) as JSON.

  python -B side_effect_probe.py <dataset> <out_dir> plain|after_d9 [<lock> <register>]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import conftest as C  # noqa: E402  (ppsi_env first, via conftest)


def identities(mh: dict) -> dict:
    out = {}
    for mid, e in mh["manifests"].items():
        out[mid] = {"tables": e["tables"], **{k: e.get(k) for k in (
            "header_content_sha256", "ordered_decision_id_sha256", "eval_ordered_decision_id_sha256",
            "rankable_ordered_decision_id_sha256", "population_sha256", "n_users", "n_decisions", "n_eval_rows",
            "n_rankable_rows")}}
    return out


def main() -> int:
    ds, out, mode = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
    extra = {}
    if mode == "after_d9":
        from v4data09.d9_gate import check_test_gate
        from v4int import test_roles as TR
        g = check_test_gate(Path(sys.argv[4]), Path(sys.argv[5]))
        res = C.run_synthetic_build(ds, out / "data_test", g)
        TR.install(out / "absent_ext.json", role="MAIN_TEST", lock_path=Path(sys.argv[4]),
                   register_path=Path(sys.argv[5]), eager=True)
        extra = {"retro_test_manifests": sorted(res["rec"]["manifests"]), "wrappers_installed": True}
    from v4data.manifests import ManifestBuilder
    ref = ManifestBuilder(ds_root=ds, out_root=out / "locked", buckets=[0], expect=False, log=C.quiet)
    ref.pass_support()
    ref.pass_aggregate()
    ref.select_populations()
    ref.pass_extract()
    man = ref.assemble()
    locked = identities({"manifests": ref.write(man)})
    from v4data06.d6_build import D6Builder, run_build
    with C.synthetic_bands():
        b6 = D6Builder(ds_root=ds, out_root=out / "d6", buckets=[0], expect=False, log=C.quiet)
        run_build(b6, out / "d6", write=True)
    d6 = identities(json.loads((out / "d6" / "receipts" / "DATA06_MANIFEST_HASHES.json").read_text(encoding="utf-8")))
    print(json.dumps({"locked": locked, "d6": d6, "extra": extra}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
