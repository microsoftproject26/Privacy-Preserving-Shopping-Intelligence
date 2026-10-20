"""g0_forecast.py (owner: Opus, lead) — PPSI V4 G0 campaign cost forecast from MEASURED inputs only.

Inputs: g0/BENCH_GRU.json, g0/BENCH_SASREC.json (synthetic-shape GPU timings, strict FP32, one A16) and
g0/data_census/CLIENT_CENSUS.json + COHORT_PROJECTIONS.json (exact per-user eligible-example frequency tables).
Output: g0/COST_FORECAST.json. Every figure is a projection (INFERENCE), not a measurement of a real run: it excludes data
loading, evaluation passes, calibration trials and supervisor overhead, which are listed separately as unmodelled.

Visit model (FA/FP/PF, local solver = AdamW reset per visit): t_visit(n) = t_copy + t_adamw + sum_steps t_step(b) + t_accum,
steps = passes * ceil(n / B_local) with the last partial batch at its own size; visits per client = E / passes at a uniform
coverage schedule (E = controlled endpoint EFE). LO: one client trained alone from theta0 for E passes over its n examples
(copy + fresh AdamW + steps; no aggregation). Central: E * N / (decisions per second at B=256).
"""
from __future__ import annotations

import json
import math
from pathlib import Path

G0 = Path(__file__).resolve().parents[1] / "g0"
BS = (1, 4, 16, 64, 256)


def jl(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def step_ms(bench: dict, b: int) -> float:
    """Piecewise-linear interpolation of the measured step time over batch size (measured at 1, 4, 16, 64, 256)."""
    t = {int(k): v["median_ms"] for k, v in bench["t_local_step_by_batch"].items()}
    if b in t:
        return t[b]
    lo = max(x for x in BS if x <= b) if b >= 1 else 1
    hi = min((x for x in BS if x >= b), default=256)
    if b > 256:
        return t[256] * b / 256
    return t[lo] + (t[hi] - t[lo]) * (b - lo) / (hi - lo)


def visit_ms(bench: dict, n: int, passes: int, b_local: int, aggregate: bool = True) -> float:
    full, rem = divmod(n, b_local)
    per_pass = full * step_ms(bench, b_local) + (step_ms(bench, rem) if rem else 0.0)
    fixed = bench["t_copy_state"]["median_ms"] + bench["t_fresh_adamw"]["median_ms"] + (bench["t_accumulate"]["median_ms"] if aggregate else 0.0)
    return fixed + passes * per_pass


def total_hours(freq: list, fn) -> float:
    return sum(users * fn(n) for n, users in freq) / 3.6e6


def main() -> int:
    benches = {f: jl(G0 / f"BENCH_{f}.json") for f in ("GRU", "SASREC")}
    cc = jl(G0 / "data_census" / "CLIENT_CENSUS.json")
    cp = jl(G0 / "data_census" / "COHORT_PROJECTIONS.json")
    freqs = {"ALL": cc["s5"]["examples_frequency_table"]}
    for N in ("16384", "65536", "262144"):
        freqs[N] = cp["cohorts"][N]["examples_frequency_table"]
    val = {N: {"users_with_val_query": cp["cohorts"][N]["users_with_ge1_validation_eval_row"],
               "val_eval_rows": cp["cohorts"][N]["validation_union_eval_rows_total"]} for N in ("16384", "65536", "262144", "ALL")}
    out = {"basis": "INFERENCE from measured synthetic-shape step timings (one A16, strict FP32) and exact census frequency tables",
           "unmodelled": ["data loading / batch assembly", "validation passes", "calibration trials (FP mu x3, LR x<=2, PF l2 x<=3)",
                          "supervisor / checkpoint I/O", "second GPU parallelism (divide wall time by <=2 if two workers)"],
           "E_controlled": 6, "passes": 2, "visits_per_client": 3, "rows": []}
    for fam, b in benches.items():
        cen_dps = b["central_decisions_per_s_B256"]
        for pop, freq in freqs.items():
            N = sum(n * u for n, u in freq)
            U = sum(u for _, u in freq)
            row = {"family": fam, "population": pop, "clients": U, "examples": N, **val[pop],
                   "central_6EFE_hours_per_seed": 6 * N / cen_dps / 3600}
            for bl in (16, 64, 256):
                row[f"FA_visit_B{bl}_hours_per_method_seed"] = total_hours(freq, lambda n: 3 * visit_ms(b, n, 2, bl))
                row[f"LO_B{bl}_hours_per_seed"] = total_hours(freq, lambda n: visit_ms(b, n, 6, bl, aggregate=False))
            # GPU-hours for one seed of the downstream core: FA + FP + PF (visits) + LO + C/CPF central runs
            core = row["FA_visit_B64_hours_per_method_seed"] * 3 + row["LO_B64_hours_per_seed"] + 2 * row["central_6EFE_hours_per_seed"]
            row["downstream_core_gpu_hours_per_seed_B64"] = core
            row["downstream_core_gpu_hours_2seeds_B64"] = 2 * core
            rounds = math.ceil(3 * U / 64)
            row["FA_rounds_at_64_clients_per_round"] = rounds
            row["virtual_bytes_per_client_visit_down_up"] = 2 * b["param_bytes_fp32"]
            row["virtual_total_bytes_per_method_seed_TiB"] = 3 * U * 2 * b["param_bytes_fp32"] / 2**40
            out["rows"].append(row)
    (G0 / "COST_FORECAST.json").write_text(json.dumps(out, indent=1, allow_nan=False))
    for r in out["rows"]:
        print(r["family"], r["population"], r["clients"], r["examples"], "central/seed h", round(r["central_6EFE_hours_per_seed"], 1),
              "FA B64 h", round(r["FA_visit_B64_hours_per_method_seed"], 1), "LO B64 h", round(r["LO_B64_hours_per_seed"], 1),
              "core 2 seeds GPU-h", round(r["downstream_core_gpu_hours_2seeds_B64"], 1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
