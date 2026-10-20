#!/bin/bash
# 06O O1 DESCRIPTIVE table "central models on never-trained new users" (UNSEEN_VAL; non-MAIN). Reads only the 5 preview dirs.
R=/home/jovyan/ahmedabdelhamed01/Privacy-Preserving-Shopping-Intelligence/.local_ai/ppsi_v4/runs_ds3/REEVAL_A40
/home/jovyan/venvs/ppsi-gpu/bin/python -B - "$R" <<'PY'
import json, sys, pathlib
R = pathlib.Path(sys.argv[1])
dirs = ["C_FULL_UNSEEN_VAL_s2026_PB", "C_FULL_UNSEEN_VAL_s2026_CB", "C_FULL_UNSEEN_VAL_s2027_PB",
        "C_FULL_UNSEEN_VAL_s2027_CB", "PRE_UNSEEN_VAL_s2026_CB"]
want = [("mrr", 20, "macro"), ("mrr", 20, "micro"), ("recall", 20, "macro"), ("recall", 20, "micro")]
print("dir | preview_ref | status | n_users | " + " | ".join(f"{m}@{k} {a}" for m, k, a in want))
for d in dirs:
    p = R / d
    if not (p / "REEVAL_RECORD.json").exists():
        print(d, "| MISSING"); continue
    rec = json.loads((p / "REEVAL_RECORD.json").read_text())
    rows = json.loads((p / "RESULT_ROWS.json").read_text())
    rows = rows.get("rows", rows) if isinstance(rows, dict) else rows
    rows = list(rows.values()) if isinstance(rows, dict) else rows
    assert rec.get("preview_reference") == "NON_MAIN_UNSEEN_VAL" and rec.get("role") == "UNSEEN_VAL", d
    assert all(r.get("preview_reference") == "NON_MAIN_UNSEEN_VAL" and r.get("comparison_status") == "DESCRIPTIVE" for r in rows), d
    cells, n = [], None
    for m, k, a in want:
        hit = [r for r in rows if r.get("metric") == m and r.get("k") == k and r.get("aggregation") == a]
        if not hit: cells.append("-"); continue
        r = hit[0]; n = r.get("n_users"); ci = r.get("ci") or {}
        lo, hi = (ci.get("lo"), ci.get("hi")) if isinstance(ci, dict) else (ci[0], ci[1]) if ci else (None, None)
        cells.append(f"{r['value']:.6f}" + (f" [{lo:.4f},{hi:.4f}]" if lo is not None else ""))
    print(f"{d} | {rec['preview_reference']} | {rec.get('status')} | {n} | " + " | ".join(cells))
PY
