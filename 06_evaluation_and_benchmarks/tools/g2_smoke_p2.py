"""tools/g2_smoke_p2.py — de-risk the G2 runner's strict row-vs-rank check on the finished P2 TUNING runs.

  python -B tools/g2_smoke_p2.py --code-root <V4>/code_p3b --runs-root <V4>/runs_p2

It uses the same code as tools/g2_run.py: bootstrap(code_root), CALIB_VALIDATION loaded through the production path
(lock V2 -> REGISTRY_V2 -> verify -> build_eval_manifest), and g2_run.check_bindings (manifest_id CALIB_VALIDATION,
with no recipe identity check because TUNING rows carry no recipe_id) run on each DONE_TUNING run under
<runs-root>/{GRU,SASREC}/stage{1,2}/, using RESULT_ROWS.json (6.0 endpoint rows) and evals/eval_6.0.ranks_int32.npy.
Output is counts only: runs checked / failed, rows checked, undefined, max_abs_diff, query_order_hash match and
PASS / FAIL. No metric value is printed. (Both families' P2 values were already read for the G1b packets.)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import g2_run as G  # noqa: E402


def tuning_runs(runs_root: Path, families=("GRU", "SASREC")) -> dict:
    out = {}
    for fam in families:
        for stage in (1, 2):
            d = Path(runs_root) / fam / f"stage{stage}"
            if not d.is_dir():
                continue
            for rd in sorted(p for p in d.iterdir() if p.is_dir()):
                s = rd / "RUN_SUMMARY.json"
                if s.is_file() and json.loads(s.read_text(encoding="utf-8")).get("status") == "DONE_TUNING":
                    out[(fam, f"stage{stage}", rd.name)] = rd
    return out


def load_calib_production(lib):
    pc = G.production_context(lib)
    return G.load_eval_manifest(lib, pc, G.CALIB_MANIFEST)[0]


def smoke(lib, runs: dict, calib, say=print) -> int:
    qhash = calib.hashes["eval_ordered_decision_id_sha256"]
    rows_checked = undefined = failed = q_ok = 0
    worst = 0.0
    for key, rd in runs.items():
        rows = [r for r in json.loads((rd / "RESULT_ROWS.json").read_text(encoding="utf-8"))["rows"]
                if r.get("checkpoint_variant", "ENDPOINT") == "ENDPOINT"]
        q = bool(rows) and all(r.get("query_order_hash") == qhash for r in rows)
        q_ok += q
        ev = {"rows": {key: rows}, "summary": {key: {}}, "pb_ranks": {},
              "end_ranks": {key: lib.np.load(rd / "evals" / G.ENDPOINT_RANKS).astype(lib.np.int64)}}
        try:
            if not q:
                raise G.G2Error("query_order_hash")
            c = G.check_bindings(lib, ev, [key], calib, manifest_id=G.CALIB_MANIFEST, check_identity=False)
        except (G.G2Error, KeyError, ValueError, TypeError):
            failed += 1
            continue
        rows_checked += c["rank_row_rows_checked"]
        undefined += c["rank_row_rows_undefined"]
        worst = max(worst, c["rank_row_max_abs_diff"])
    ok = bool(runs) and failed == 0 and q_ok == len(runs) and rows_checked > 0
    say(f"runs checked {len(runs)}")
    say(f"runs failed {failed}")
    say(f"rows checked {rows_checked}")
    say(f"undefined {undefined}")
    say(f"max_abs_diff {worst!r}")
    say(f"query_order_hash match {q_ok}/{len(runs)}")
    say("PASS" if ok else "FAIL")
    return 0 if ok else 1


def main(argv=None, *, calib_loader=None, say=print) -> int:
    ap = argparse.ArgumentParser(description="G2 strict-check smoke on P2 TUNING runs")
    ap.add_argument("--code-root", required=True, type=Path)
    ap.add_argument("--runs-root", required=True, type=Path)
    args = ap.parse_args(argv)
    lib = G.bootstrap(args.code_root)
    calib = (calib_loader or load_calib_production)(lib)
    return smoke(lib, tuning_runs(args.runs_root), calib, say=say)


if __name__ == "__main__":
    sys.exit(main())
