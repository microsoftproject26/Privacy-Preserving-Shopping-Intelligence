"""probe_runner.py (owner: Opus) — baseline + 8 coherent group probes + lean parity on the VS cohort.

Implements Fable decision items 3 and 4 (review/AUTO_DECISION_SCOPE.md):
1. Verify the pinned checkpoint and cohort hashes against config/BASELINE_PIN.json.
2. ENCODING CHECK: every "unavailable" encoding the probes inject must equal what the real pipeline
   produced for real rows in that state. Any mismatch aborts before any score exists.
3. Score the baseline and P1..P8 (strict FP32, exhaustive K, the audited evaluator).
   - Paired user bootstrap against the baseline (B=1000, registered seed).
   - Weekday/weekend slices by the current event's local is_weekend.
4. Gate 2(a) for a removal candidate C2, written to state/C2_DECISION.json.
5. Lean parity (M2/M3 drops + the M1 repair, folded weights) against the baseline, if audit/M1_FIT.json exists.

Writes audit/GROUP_PROBE_RESULTS.json and audit/LEAN_PARITY.json, plus ranks under runs/probes/.
Never overwrites. Never reads VC/TEST.
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import torch

import fo_env
import feature_view as fv
import fo_models as fm
import fo_eval as E
import precision

GATE_MIN_DELTA = 0.001
ENCODING_MIN_ROWS = 100      # Review C fix: a state verified on fewer real rows is UNVERIFIED, not OK


def encoding_check(store, ds) -> dict:
    cu, cm = fv.cold_user_vector()
    cu, cm = torch.as_tensor(cu), torch.as_tensor(cm)
    v_sup = fv.enc_user("current_minus_recent_mean_log_price", 0.0, {"recent_price_supported": 1.0})
    z_run0 = fv.enc_event("current_known_category_run_length", 0.0)
    cnt = {k: 0 for k in ("cold_rows", "cur_price_missing_supported", "cur_price_missing_unsupported", "price_missing_events",
                          "brand_missing_events", "brand_unsupported_rows", "maincat_missing_events",
                          "cat_unsupported_rows", "prev_gap_unavail_events", "item_age_unavail_events")}
    bad = []
    for s in range(0, len(ds), 4096):
        pos = np.arange(s, min(s + 4096, len(ds)))
        b = {k: torch.as_tensor(v) for k, v in store.batch(ds, pos).items() if isinstance(v, (np.ndarray, torch.Tensor))}
        kw = {"attention_mask": b["attention_mask"], "lengths": b["lengths"]}
        real = b["attention_mask"].bool()
        last = fv.last_position_mask(kw) & real
        num, flg = b["event_numeric_features"].float(), b["event_quality_flags"]
        uc, um = b["user_context"].float(), b["user_context_masks"].float()
        F, N, U, M = fv.FLAG, fv.NUM, fv.UCTX, fv.UMASK

        def check(name, cond_rows, ok_rows):
            n = int(cond_rows.sum()); cnt[name] += n
            if n and not bool(ok_rows[cond_rows].all()):
                bad.append(f"{name}: {int((~ok_rows[cond_rows]).sum())} of {n} rows differ")
        cold = um[:, M["has_history_mask"]] == 0
        check("cold_rows", cold, torch.isclose(uc, cu.expand_as(uc), atol=1e-6).all(1) & (um == cm).all(1))
        pm_last = ((flg[..., F["price_missing"]] == 1) & last).any(1)
        sup = um[:, M["recent_price_supported"]] == 1
        cmr = uc[:, U["current_minus_recent_mean_log_price"]]
        check("cur_price_missing_supported", pm_last & sup, torch.isclose(cmr, torch.full_like(cmr, v_sup), atol=1e-6))
        check("cur_price_missing_unsupported", pm_last & ~sup, cmr == 0)
        pe = (flg[..., F["price_missing"]] == 1) & real
        check("price_missing_events", pe, (num[..., N["log1p_price"]] == 0) & (flg[..., F["price_zero"]] == 1))
        be = (flg[..., F["brand_missing"]] == 1) & real
        check("brand_missing_events", be, b["brand_tokens"] == fv.MISSING)
        bu = um[:, M["recent_brand_match_supported"]] == 0
        check("brand_unsupported_rows", bu, uc[:, U["recent_brand_match_share"]] == 0)
        ce = (flg[..., F["main_category_missing"]] == 1) & real
        check("maincat_missing_events", ce, (b["main_category_tokens"] == fv.MISSING)
              & (flg[..., F["category_code_missing"]] == 1)
              & torch.isclose(num[..., N["current_known_category_run_length"]], torch.tensor(z_run0), atol=1e-6))
        cu_rows = um[:, M["recent_category_match_supported"]] == 0
        check("cat_unsupported_rows", cu_rows, uc[:, U["recent_category_match_share"]] == 0)
        ge = (flg[..., F["prev_gap_unavailable"]] == 1) & real
        check("prev_gap_unavail_events", ge, num[..., N["log1p_prev_raw_view_gap_s"]] == 0)
        ae = (flg[..., F["item_age_unavailable"]] == 1) & real
        check("item_age_unavail_events", ae, num[..., N["log1p_item_observed_age_s"]] == 0)
    unverified = sorted(k for k, n in cnt.items() if n < ENCODING_MIN_ROWS)
    return {"counts": cnt, "violations": bad, "unverified": unverified, "min_rows": ENCODING_MIN_ROWS,
            "population": "all VALIDATION_SELECT eval decisions", "ok": not bad and not unverified,
            "values": {"cold_user_context": cu.tolist(), "current_minus_supported_sentinel": v_sup,
                       "run_length_unknown": z_run0}}


def main() -> int:
    """Idempotent stages (Review C fix): encoding check -> probes (+C2 decision) -> lean parity.
    Each stage is skipped only if ITS own output exists; a crash in a later stage never skips an
    earlier or a later one on retry."""
    camp = fo_env.Path(os.environ.get("FO_CAMPAIGN_ROOT", "/home/jovyan/ahmedabdelhamed01/Privacy-Preserving-Shopping-Intelligence/.local_ai/sasrec_v1"))
    if not (camp / "state" / "FINAL_STAGE_DONE").exists() and not os.environ.get("FO_ALLOW_BEFORE_FINAL"):
        print("campaign not finished: QUEUED_RESOURCE"); return 3
    enc_p, out_p = fo_env.study_path("audit", "PROBE_ENCODING_CHECK.json"), fo_env.study_path("audit", "GROUP_PROBE_RESULTS.json")
    c2_p, lean_p = fo_env.study_path("state", "C2_DECISION.json"), fo_env.study_path("audit", "LEAN_PARITY.json")
    if out_p.exists() and c2_p.exists() and lean_p.exists():
        print("probe stage complete (immutable outputs)"); return 0
    fit_p = fo_env.study_path("audit", "M1_FIT.json")
    if not fit_p.exists():
        raise RuntimeError("audit/M1_FIT.json missing: stream_fit must run before the probes")
    precision.configure_strict_fp32(2026)
    device = torch.device(os.environ.get("FO_DEVICE", "cuda:0"))
    pin = json.loads(fo_env.study_path("config", "BASELINE_PIN.json").read_text())
    cell = json.loads(fo_env.study_path("config", "P8_CALENDAR_CELL.json").read_text())
    ch = pin["chosen"]
    if fo_env.sha256_file(ch["file"]) != ch["sha256"]:
        raise RuntimeError("pinned checkpoint changed since the pin")
    if fo_env.sha256_file(pin["cohort_parquet"]) != pin["cohort_sha256"]:
        raise RuntimeError("cohort parquet changed since the pin")
    fo_env.verify_snapshot()
    t0 = time.time()
    store = E.build_vs_store()
    ds = E.load_cohort(store, pin["cohort_parquet"])
    fo_env.assert_isolated()
    fo_env.study_path("audit").mkdir(exist_ok=True)
    # ---- stage 1: encoding check over ALL VS eval decisions (not only the cohort), floor per state
    if enc_p.exists():
        enc = json.loads(enc_p.read_text())
    else:
        enc = encoding_check(store, store.decisions("VALIDATION_SELECT", "eval"))
        E.write_once(enc_p, enc)
    if not enc["ok"]:
        print("ENCODING CHECK FAILED / UNVERIFIED:", enc["violations"], enc.get("unverified")); return 5
    wk = E.decision_is_weekend(store, ds)
    sd, _ = fm.load_weights_file(ch["file"])
    reg = ch["registry_entry"]
    base = fm.build(reg, sd, view="full", device=device).eval()
    rdir = fo_env.study_path("runs", "probes"); rdir.mkdir(parents=True, exist_ok=True)
    # light copy of the cohort decision fields for finalize (internal pseudonymous keys only, never raw ids)
    np.savez_compressed(rdir / "COHORT_DS.npz", **{k: np.asarray(getattr(ds, k)) for k in
                                                   ("decision_id", "target_class", "eval_mask", "eval_user_key", "client_idx")},
                        is_weekend=wk)
    # ---- stage 2: baseline + probes (+ gate 2(a))
    if out_p.exists():
        res = json.loads(out_p.read_text())
        bz = np.load(rdir / "BASELINE.npz")
        if fo_env.sha256_file(rdir / "BASELINE.npz") != res["baseline"]["ranks_sha256"] or not np.array_equal(bz["decision_id"], ds.decision_id):
            raise RuntimeError("stored baseline ranks do not match GROUP_PROBE_RESULTS.json")
        base_ranks, base_m = bz["ranks"].astype(np.int64), res["baseline"]["metrics"]
    else:
        res = {"pin": {k: ch[k] for k in ("seed", "epoch", "file", "sha256", "macro_mrr20")}, "cohort_n": len(ds),
               "encoding_check": "audit/PROBE_ENCODING_CHECK.json", "calendar_cell": cell, "bootstrap": {"n": E.BOOT_N, "seed": E.BOOT_SEED},
               "slice_rule": "current event local is_weekend", "weekend_share": float(wk.mean()), "probes": {},
               "interpretation": "reliance of THIS checkpoint under coherent absence; not the effect of removal with retraining"}
        base_ranks, base_m, secs = E.evaluate(base, store, ds, device)
        res["baseline"] = {"metrics": base_m, "seconds": secs, "ranks_sha256": E.save_ranks(rdir / "BASELINE.npz", base_ranks, ds)}
        for name in fv.PROBES:
            spec = fv.ProbeSpec(name, calendar_cell=(cell["local_weekday_idx"], cell["daypart_idx"]) if name.startswith("P8") else None)
            model = fm.FOModel(base.inner, probe=spec).eval()
            ranks, m, secs = E.evaluate(model, store, ds, device)
            res["probes"][name] = {"metrics": m, "seconds": secs, "ranks_sha256": E.save_ranks(rdir / f"{name}.npz", ranks, ds),
                                   "delta_macro_mrr@20": m["macro_mrr@20"] - base_m["macro_mrr@20"],
                                   "delta_micro_recall@20": m["micro_recall@20"] - base_m["micro_recall@20"],
                                   "bootstrap_macro_mrr@20": E.paired(ranks, base_ranks, ds),
                                   "slices": {"weekday": E.sliced(ranks, base_ranks, ds, ~wk), "weekend": E.sliced(ranks, base_ranks, ds, wk)},
                                   "label": "OOD anchor" if name == "P1_ALL_SIDE_OFF" else ("single-point counterfactual" if name.startswith("P8") else "coherent absence")}
            print(name, round(res["probes"][name]["delta_macro_mrr@20"], 5), flush=True)
        res["gate_2a"] = gate_2a(res["probes"])
        res["seconds_total"] = round(time.time() - t0, 1)
        E.write_once(out_p, res)
    if not c2_p.exists():
        E.write_once(c2_p, res["gate_2a"])
    # ---- stage 3: lean parity (hygiene deliverable, decision items 4/5)
    if not lean_p.exists():
        m1 = fv.m1_states_from_fit(json.loads(fit_p.read_text())["features"])
        lean = fm.build(reg, sd, view="lean", m1=m1, device=device).eval()
        lr_, lm, secs = E.evaluate(lean, store, ds, device)
        with torch.no_grad(), precision.strict_eval():
            b = store.batch(ds, np.arange(min(1024, len(ds))))
            import sasrec_data
            kw = {k: v.to(device) for k, v in sasrec_data.predictor_kwargs(b, base).items()}
            dmax = float((base(**kw) - lean(**kw)).abs().max())
        par = {"max_abs_logit_diff_first_1024": dmax, "ranks_identical_fraction": float((lr_ == base_ranks).mean()),
               "n_rank_changes": int((lr_ != base_ranks).sum()), "baseline": base_m, "lean": lm,
               "delta_macro_mrr@20": lm["macro_mrr@20"] - base_m["macro_mrr@20"], "seconds": secs,
               "parameter_mapping": fm.parameter_mapping(sd, "lean", m1),
               "m1_state": {f: vars(s) for f, s in m1.items()}, "schema": fv.schema_record()}
        E.write_once(lean_p, par)
        print("lean parity: max|dlogit|", dmax, "rank-identical", par["ranks_identical_fraction"])
    print(json.dumps({"baseline_macro": base_m["macro_mrr@20"], "C2": res["gate_2a"]["C2"]}))
    return 0


def gate_2a(probes: dict) -> dict:
    """A removal candidate only from a POSITIVE, CI-supported, slice-consistent delta (P1 = OOD anchor excluded)."""
    passing = []
    for name, r in probes.items():
        if name == "P1_ALL_SIDE_OFF":
            continue
        d, lo = r["delta_macro_mrr@20"], r["bootstrap_macro_mrr@20"]["ci95"][0]
        s_wd, s_we = r["slices"]["weekday"]["delta_macro_mrr@20"], r["slices"]["weekend"]["delta_macro_mrr@20"]
        r["gate_2a"] = {"delta_ge_0.001": d >= GATE_MIN_DELTA, "ci_lower_gt_0": lo > 0, "slices_positive": s_wd > 0 and s_we > 0}
        if all(r["gate_2a"].values()):
            passing.append((d, name))
    passing.sort(reverse=True)
    return {"rule": "Fable decision 2(a)", "passing": [n for _, n in passing], "C2": passing[0][1] if passing else None}


if __name__ == "__main__":
    sys.exit(main())
