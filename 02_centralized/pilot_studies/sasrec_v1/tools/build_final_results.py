"""build_final_results.py (owner: Opus, lead): FINAL_SASREC_RESULTS.json and the result tables for the
final report, computed only from fetched result files. Runs locally, read-only on its inputs.

  python build_final_results.py --inputs final_inputs --out final

Inputs (copies of server files, fetched after locks/RETRO_DONE.json exists):
  retro_eval/RETRO_RESULTS.json, locks/RETRO_DONE.json, locks/SASREC_LOCK.json, locks/RETRO_CLAIM.json,
  runs/<run>/summary.json for the six v3 runs, bench/TF32_PARITY.json (optional).
Every VC/TEST number carries the label "PREVIOUSLY INSPECTED / RETROSPECTIVE BENCHMARK".
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime
from pathlib import Path

LABEL = "PREVIOUSLY INSPECTED / RETROSPECTIVE BENCHMARK"
SPLITS = ("VALIDATION_CONFIRM", "TEST")
RUNS = ("A0_ID256_s2026", "A1_RICH256_s2026", "A1_RICH256_s2027")
ARCH = ("GRU_F0_locked", "GRU_F4_descriptive", "TRANS_full", "POP_full")
SYSTEMS = {"S1": "A1 s2026 + A1 s2027 (log-prob mean)", "S2": "A0 s2026 + A1 s2026 (log-prob mean)",
           "S3": "A0 s2026 + A1 s2026 + A1 s2027 (log-prob mean)", "S4a": "weight average A1 s2026 e3/e4/e5",
           "S4b": "weight average A1 s2027 e3/e4/e5"}
SHOW = (("macro", "mrr@20"), ("micro", "mrr@20"), ("micro", "recall@20"), ("macro", "recall@20"), ("macro", "ndcg@20"))


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def jl(p: Path):
    return json.loads(p.read_text(encoding="utf-8"))


def clean(o):
    if isinstance(o, float) and not math.isfinite(o):
        return None
    if isinstance(o, dict):
        return {k: clean(v) for k, v in o.items()}
    if isinstance(o, list):
        return [clean(v) for v in o]
    return o


def f6(x):
    return "—" if x is None else f"{x:.6f}"


def row_metrics(hl: dict, den: str = "end_to_end") -> dict:
    b = hl[den]
    return {f"{agg}_{m}": b[agg][m] for agg, m in SHOW} | {"n": b["n"], "users": b["users"]}


def build(inp: Path) -> dict:
    retro = jl(inp / "retro_eval/RETRO_RESULTS.json")
    done = jl(inp / "locks/RETRO_DONE.json")
    lock_p = inp / "locks/SASREC_LOCK.json"
    lock = jl(lock_p)
    rp = inp / "retro_eval/RETRO_RESULTS.json"
    assert done["results_sha256"] == sha(rp), "RETRO_RESULTS.json differs from the RETRO_DONE hash"
    assert retro["lock_sha256"] == sha(lock_p), "RETRO_RESULTS lock hash differs from the fetched lock"
    assert retro["label"] == LABEL and lock["label"] == LABEL
    head = retro["headline"]
    summ = {}
    for r in RUNS:
        for d in (f"{r}", f"{r}_cool5"):
            p = inp / "runs" / d / "summary.json"
            if p.exists():
                summ[d] = jl(p)
    vs = []
    for d, s in sorted(summ.items()):
        for rec in s["epoch_records"]:
            fv = rec.get("full_vs")
            if not fv or rec.get("inherited_from_source"):
                continue
            vs.append({"run": d, "epoch": rec["epoch"], "macro_mrr@20": fv["macro_mrr20"], "micro_mrr@20": fv["micro_mrr20"],
                       "micro_recall@20": fv["micro_recall20"], "macro_recall@20": fv["macro_recall20"], "n": fv["n"], "users": fv["users"]})
    g = {e: {} for e in (3, 4, 5)}
    for x in vs:
        if x["run"].startswith("A1_RICH256"):
            g[x["epoch"]][x["run"].split("_s")[1][:4]] = x["macro_mrr@20"]
    seed_gap_vs = {f"e{e}": abs(v["2026"] - v["2027"]) for e, v in g.items() if len(v) == 2}
    out = {
        "label": LABEL, "note": "VC and TEST were inspected in the GRU project before this campaign; every VC/TEST number here is retrospective.",
        "lock": {"sha256": sha(lock_p), "created_utc": lock["created_utc"], "headline": head, "selection_rule": lock["selection"].get("rule"),
                 "selection_ranking": lock["selection"].get("ranking"), "pair_check": lock["pair_check"], "dropped_entries": lock["dropped_entries"],
                 "entries": {k: {kk: v.get(kk) for kk in ("kind", "members", "run", "best_epoch", "epochs_available", "best_metric_full_vs")}
                             for k, v in lock["entries"].items()}},
        "retro": {"results_sha256": done["results_sha256"], "completed_utc": done["completed_utc"],
                  "claim_utc": jl(inp / "locks/RETRO_CLAIM.json")["claimed_utc"]},
        "validation_select": {"rows": vs, "seed_gap_macro_mrr20_A1": seed_gap_vs,
                              "seed_gap_note": "two seeds give a range, not a standard deviation"},
        "splits": {sp: retro["splits"][sp]["fingerprint"] for sp in SPLITS},
        "single_models": {}, "archived_baselines": {}, "systems": {}, "bootstraps_TEST": retro["bootstraps"].get("TEST", {}),
        "bootstraps_skipped": retro.get("bootstraps_skipped", {}), "seed_variation_TEST": retro["seed_variation"],
        "archived_fingerprint_gate": retro["archived_fingerprint_gate"], "cohorts": {}, "resources": {},
    }
    for n in RUNS:
        m = retro["models"][n]
        out["single_models"][n] = {sp: {den: row_metrics(m[sp], den) for den in ("end_to_end", "closed")} for sp in SPLITS}
    for n in ARCH:
        m = retro["models"][n]
        out["archived_baselines"][n] = {sp: {den: row_metrics(m[sp], den) for den in ("end_to_end", "closed")} for sp in SPLITS}
    for n in SYSTEMS:
        if n in retro["systems"]:
            m = retro["systems"][n]
            out["systems"][n] = {"kind": m["kind"], "definition": SYSTEMS[n],
                                 **{sp: {den: row_metrics(m[sp], den) for den in ("end_to_end", "closed")} for sp in SPLITS}}
    for n in (head, "GRU_F0_locked"):
        src = retro["models"][n]
        out["cohorts"][n] = {sp: src[sp]["cohorts"] for sp in SPLITS}
    for d, s in sorted(summ.items()):
        recs = [r for r in s["epoch_records"] if not r.get("inherited_from_source")]
        out["resources"][d] = {"physical_gpu": s.get("physical_gpu"), "start_utc": s["start_utc"], "end_utc": s["end_utc"],
                               "wall_seconds_cumulative": s["wall_seconds"],
                               "elapsed_hours_this_run": (datetime.strptime(s["end_utc"], "%Y-%m-%dT%H:%M:%SZ")
                                                          - datetime.strptime(s["start_utc"], "%Y-%m-%dT%H:%M:%SZ")).total_seconds() / 3600, "resume_count": s["resume_count"], "updates": s["updates"],
                               "n_params": s["n_params"], "precision_training": s.get("precision_training"),
                               "precision_eval": s.get("precision_eval"),
                               "peak_gpu_mem_allocated_gib": max((r.get("peak_gpu_mem_allocated_bytes") or 0) for r in recs) / 2**30,
                               "rss_max_gib": max((r.get("rss_bytes_max") or 0) for r in recs) / 2**30}
    tp = inp / "bench/TF32_PARITY.json"
    if tp.exists():
        out["tf32_parity"] = jl(tp)
    v2 = {}
    for r in ("A0_ID256_s2026", "A1_RICH256_s2026"):
        p = inp / "runs_v2" / r / "epoch_records.json"
        if p.exists():
            v2[r] = {f"e{x['epoch']}": x["full_vs"]["macro_mrr20"] for x in jl(p) if x.get("full_vs")}
    if v2:
        out["v2_failed_full_vs_macro_mrr20"] = v2
    gp = inp / "gru" / "FINAL_FULL_GRU_RESULTS.json"
    if gp.exists():
        out["gru_v1_full_vs_macro_mrr20"] = {"GRU_F0_locked": jl(gp)["runs"]["F0_item_only_h256"]["summary"]["best_metric"],
                                             "source": "gru_v1/FINAL_FULL_GRU_RESULTS.json runs.F0_item_only_h256.summary.best_metric"}
    out["claims"] = {"sota": "NOT_CLAIMED", "privacy": "NOT_TESTED", "causality": "NOT_CLAIMED (bundled v3 repair; single Rich-vs-ID comparison)",
                     "systems_vs_gru": "NOT_A_MODEL_VS_MODEL_COMPARISON (systems reported separately)"}
    out["inputs_sha256"] = {str(p.relative_to(inp)).replace("\\", "/"): sha(p) for p in sorted(inp.rglob("*")) if p.is_file()}
    return clean(out)


def tables(r: dict) -> dict:
    head = r["lock"]["headline"]
    t = {}
    for sp in SPLITS:
        lines = ["| Model | macro MRR@20 | micro MRR@20 | micro R@20 | macro R@20 | macro NDCG@20 |", "|---|---|---|---|---|---|"]
        for n in RUNS + ARCH:
            src = r["single_models"].get(n) or r["archived_baselines"][n]
            m = src[sp]["end_to_end"]
            name = f"**{n} (headline)**" if n == head else n
            lines.append(f"| {name} | {f6(m['macro_mrr@20'])} | {f6(m['micro_mrr@20'])} | {f6(m['micro_recall@20'])} | {f6(m['macro_recall@20'])} | {f6(m['macro_ndcg@20'])} |")
        t[f"singles_{sp}"] = "\n".join(lines)
        lines = ["| System | definition | macro MRR@20 | micro MRR@20 | micro R@20 |", "|---|---|---|---|---|"]
        for n, s in r["systems"].items():
            m = s[sp]["end_to_end"]
            lines.append(f"| {n} | {s['definition']} | {f6(m['macro_mrr@20'])} | {f6(m['micro_mrr@20'])} | {f6(m['micro_recall@20'])} |")
        t[f"systems_{sp}"] = "\n".join(lines)
        lines = ["| Model | closed macro MRR@20 | closed micro R@20 | closed n |", "|---|---|---|---|"]
        for n in RUNS + ARCH:
            src = r["single_models"].get(n) or r["archived_baselines"][n]
            m = src[sp]["closed"]
            lines.append(f"| {n} | {f6(m['macro_mrr@20'])} | {f6(m['micro_recall@20'])} | {m['n']:,} |")
        t[f"closed_{sp}"] = "\n".join(lines)
    lines = ["| Paired difference (TEST, user-macro MRR@20) | diff | 95% CI | users | resamples |", "|---|---|---|---|---|"]
    for k, b in r["bootstraps_TEST"].items():
        lines.append(f"| {k} | {b['diff']:+.6f} | [{b['ci95'][0]:+.6f}, {b['ci95'][1]:+.6f}] | {b['n_users']:,} | {b['n_resamples']} |")
    t["bootstraps"] = "\n".join(lines)
    lines = ["| Run | epoch | macro MRR@20 | micro MRR@20 | micro R@20 | macro R@20 |", "|---|---|---|---|---|---|"]
    for x in r["validation_select"]["rows"]:
        lines.append(f"| {x['run']} | {x['epoch']} | {f6(x['macro_mrr@20'])} | {f6(x['micro_mrr@20'])} | {f6(x['micro_recall@20'])} | {f6(x['macro_recall@20'])} |")
    t["vs"] = "\n".join(lines)
    lines = ["| Model | split | warm n | warm macro MRR@20 | unseen n | unseen macro MRR@20 |", "|---|---|---|---|---|---|"]
    for n, c in r["cohorts"].items():
        for sp in SPLITS:
            w, u = c[sp]["warm"], c[sp]["unseen"]
            lines.append(f"| {n} | {sp} | {w['n']:,} | {f6(w['macro_mrr@20'])} | {u['n']:,} | {f6(u['macro_mrr@20'])} |")
    t["cohorts"] = "\n".join(lines)
    lines = ["| Run | GPU | start | end | elapsed h | cumulative wall h | updates | params | peak alloc GiB | RSS max GiB |", "|---|---|---|---|---|---|---|---|---|---|"]
    for d, x in r["resources"].items():
        lines.append(f"| {d} | {x['physical_gpu']} | {x['start_utc']} | {x['end_utc']} | {x['elapsed_hours_this_run']:.2f} | {x['wall_seconds_cumulative'] / 3600:.2f} | {x['updates']:,} | "
                     f"{x['n_params']:,} | {x['peak_gpu_mem_allocated_gib']:.2f} | {x['rss_max_gib']:.2f} |")
    t["resources"] = "\n".join(lines)
    return t


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()
    r = build(a.inputs)
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "FINAL_SASREC_RESULTS.json").write_text(json.dumps(r, indent=2, allow_nan=False, ensure_ascii=False), encoding="utf-8")
    (a.out / "_tables.json").write_text(json.dumps(tables(r), indent=1, ensure_ascii=False), encoding="utf-8")
    print("headline", r["lock"]["headline"], "TEST", r["single_models"][r["lock"]["headline"]]["TEST"]["end_to_end"]["macro_mrr@20"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
