"""metric_table.py — FIX-5 (Fable gate 2026-10-03): the registered DESCRIPTIVE full metric table, extracted from the
sealed evidence rows of ONE built block (VALIDATION after F, TEST after H analysis). Frozen with the ANA script freeze
(its sha256 is in ANA_SCRIPT_FREEZE `scripts`; freeze_check refuses any later change). Registration text:
company/plans/ops_scripts/endgame/REG_DESCRIPTIVE_METRIC_TABLE.md (DL row before D).

  python -B metric_table.py --block <block dir> --split VALIDATION|TEST --expect-block-index-sha256 <hex> --out <new dir>

What it does (and nothing else):
  * reads <block>/BLOCK_INDEX.json; refuses unless its file sha256 = --expect-block-index-sha256 (E_CHECK_PASS / H record)
    and its split = --split;
  * for every entry with a rows file: refuses unless the copied rows file sha256 = the index `sources` sha256; applies the
    entry's row_filter (device cell) and its declared rule (rows with selection_rule = the entry rule; RULE_FREE / rows
    without a selection_rule exactly as retention_report.select_rows); keeps only the evaluator's registered grid
    (v4eval.metrics FAMILIES mrr / hr / ndcg x K_GRID 1 / 5 / 10 / 20; v4eval.rows AGGREGATIONS micro / macro;
    DENOMINATORS E2E / RANKABLE);
  * COPIES each row's `value`, `value_status`, `value_pessimistic`, `n_decisions`, `n_users` verbatim. No mean, no ratio,
    no CI, no bootstrap, no selection, no contrast, no rounding: zero new computation of any value;
  * refuses on a duplicate cell (more than one row for one entry x aggregation x denominator x metric x k) or a missing
    grid cell;
  * writes <out>/METRIC_TABLE.json (O_EXCL; self_sha256) and METRIC_TABLE.csv (O_EXCL); prints counts only.
Labels: every cell DESCRIPTIVE; the confirmatory reading stays RETENTION_RESULT.json (macro MRR@20, 06P / 06D §3).
ONNX entries (rows null; K16 lives in BENCH_RESULT) and TRUNC records are listed as `not_in_table`, not extracted.
"""
import argparse, csv, hashlib, io, json, os, sys
from pathlib import Path

FORMAT = "PPSI_V4_DESCRIPTIVE_METRIC_TABLE_1"
FAMILIES, K_GRID = ("mrr", "hr", "ndcg"), (1, 5, 10, 20)          # = v4eval.metrics FAMILIES / K_GRID (r7)
AGGREGATIONS, DENOMINATORS = ("micro", "macro"), ("E2E", "RANKABLE")   # = v4eval.rows (r7)
RULE_FREE = "RULE_FREE"
COPIED = ("value", "value_status", "value_pessimistic", "n_decisions", "n_users")


class Refused(RuntimeError):
    pass


def fsha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def rows_of(obj):
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict) and isinstance(obj.get("rows"), list):
        return obj["rows"]
    raise Refused("a rows file is neither a list nor an object with a 'rows' list")


def pick(entry, rows):
    flt = entry.get("row_filter") or {}
    rows = [r for r in rows if all(r.get(f) == v for f, v in flt.items())]
    rule = entry.get("rule")
    if rule == RULE_FREE:
        return rows, "RULE_FREE"
    hit = [r for r in rows if r.get("selection_rule") == rule]
    if hit:
        return hit, "ROW"
    if any(r.get("selection_rule") is not None for r in rows):
        raise Refused(f"{entry['arm']}/{entry.get('seed')}/{rule}/{entry['role']}: rows carry another selection_rule")
    return rows, "INDEX"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--block", required=True)
    ap.add_argument("--split", required=True, choices=("VALIDATION", "TEST"))
    ap.add_argument("--expect-block-index-sha256", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    blk, out = Path(a.block), Path(a.out)
    ip = blk / "BLOCK_INDEX.json"
    if fsha(ip) != a.expect_block_index_sha256:
        raise Refused("BLOCK_INDEX.json sha256 is not --expect-block-index-sha256")
    idx = json.loads(ip.read_text(encoding="utf-8"))
    if idx.get("split") != a.split:
        raise Refused(f"block split {idx.get('split')!r} is not --split {a.split}")
    if out.exists():
        raise Refused(f"{out} exists (O_EXCL)")
    src = idx.get("sources") or {}
    cells, skipped = [], []
    for e in idx["entries"]:
        tag = {k: e.get(k) for k in ("arm", "seed", "rule", "role", "source", "comparison_block")}
        if not e.get("rows"):
            skipped.append(dict(tag, reason="no rows file (ONNX entry: K16 is in BENCH_RESULT)"))
            continue
        rel = e["rows"]
        if rel not in src or fsha(blk / rel) != src[rel]["sha256"]:
            raise Refused(f"{rel}: copied rows file sha256 differs from the index sources record")
        rows, rule_source = pick(e, rows_of(json.loads((blk / rel).read_text(encoding="utf-8"))))
        seen = {}
        for r in rows:
            m, k, ag, dn = r.get("metric"), r.get("k"), r.get("aggregation"), r.get("target_denominator")
            if m not in FAMILIES or ag not in AGGREGATIONS or dn not in DENOMINATORS or k is None or int(k) not in K_GRID:
                continue
            key = (ag, dn, m, int(k))
            if key in seen:
                raise Refused(f"{tag}: two rows for {key} after the filter")
            seen[key] = r
        want = {(ag, dn, m, k) for ag in AGGREGATIONS for dn in DENOMINATORS for m in FAMILIES for k in K_GRID}
        if set(seen) != want:
            raise Refused(f"{tag}: grid cells missing {sorted(want - set(seen))[:4]}")
        for (ag, dn, m, k), r in sorted(seen.items()):
            cells.append(dict(tag, rows_file=rel, rows_sha256=src[rel]["sha256"], rule_source=rule_source,
                              checkpoint_id=r.get("checkpoint_id"), aggregation=ag, target_denominator=dn, metric=m, k=k,
                              label="DESCRIPTIVE", **{c: r.get(c) for c in COPIED}))
    for k in ("TRUNC_ROWS",):
        if (idx.get("records") or {}).get(k):
            skipped.append({"record": k, "reason": "TRUNC curves are reported by retention_report, not this table"})
    doc = {"format": FORMAT, "split": a.split, "label": "DESCRIPTIVE (registered full metric table; FIX-5)",
           "confirmatory_reading": "RETENTION_RESULT.json (macro MRR@20 primary; NDCG@10 / HR@10 as the report computes them)",
           "block_index_sha256": a.expect_block_index_sha256, "block_index_self_sha256": idx.get("self_sha256"),
           "script_sha256": fsha(Path(__file__).resolve()), "grid": {"families": FAMILIES, "k": K_GRID,
           "aggregations": AGGREGATIONS, "denominators": DENOMINATORS}, "copied_fields": COPIED,
           "rule": "values copied verbatim from the sealed evaluator rows; no new computation", "cells": cells,
           "not_in_table": skipped}
    body = json.dumps(doc, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    doc["self_sha256"] = hashlib.sha256(body.encode("utf-8")).hexdigest()
    out.mkdir(parents=True)
    fd = os.open(out / "METRIC_TABLE.json", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=1, sort_keys=True, ensure_ascii=False)
    cols = ["arm", "seed", "rule", "role", "source", "comparison_block", "aggregation", "target_denominator", "metric", "k",
            *COPIED, "label", "checkpoint_id", "rule_source", "rows_file", "rows_sha256"]
    buf = io.StringIO(); w = csv.DictWriter(buf, fieldnames=cols, extrasaction="ignore", lineterminator="\n")
    w.writeheader(); w.writerows(cells)
    fd = os.open(out / "METRIC_TABLE.csv", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(buf.getvalue())
    n_ent = len({(c["arm"], c["seed"], c["rule"], c["role"]) for c in cells})
    print(f"METRIC_TABLE written: {n_ent} entries x {len(cells) // max(n_ent, 1)} cells = {len(cells)}; "
          f"not_in_table {len(skipped)}; self {doc['self_sha256'][:16]}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Refused as e:
        print(f"METRIC_TABLE REFUSED: {e}", file=sys.stderr)
        sys.exit(2)
