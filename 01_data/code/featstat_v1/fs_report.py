"""featstat_v1.fs_report - FEATSTAT_SUMMARY.md + FEATSTAT_PROVENANCE_CLASSIFICATION.json from the JSON outputs.

Numbers are read from the JSON files only (no recomputation). The provenance classification is a PROPOSAL for the
POPULATION_5_100 deliberation, not a decision.
Usage: python -B fs_report.py <featstat_v5_100 dir>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

POPS = ("POOL", "FULL")
METHODS = "C / FA / FP / PF / CPF / LO"

CLASSIFICATION = [
    {"statistic": "v1 z-normalizers baked into the store (14 kept columns: clip quantiles 0.001/0.999 + mean/std)",
     "fitted_on_now": "full TRAIN: all 4,204,379 TRAIN users (includes n = 0, 1-4 and > 100 users)",
     "proposed_class_as_fitted": "POOLED_INFORMATION_FROM_EXCLUDED_CLIENTS (mixture: pool + excluded strata)",
     "proposed_class_if_refit_on_pool": "COHORT_SPECIFIC_LEARNED_INFORMATION",
     "legitimately_available": {
         "as_fitted": "all six only if declared a pre-deployment historical bootstrap blob shipped identically "
                      "(historical-comparability option A); none of FA/FP/PF/LO could compute it from their own clients",
         "pool_refit": "C and CPF directly; FA/FP/PF via one accounted federated-statistics round (secure sums of "
                       "count, sum, sum of squares per column + a quantile sketch for the clip bounds); LO only as the "
                       "same shipped block (section 9: the same block for all six)"}},
    {"statistic": "M1 x2 valid-only mean/std (time since previous user event / previous session) and the F-1 "
                  "valid-only refit (current_minus_recent_mean_log_price)",
     "fitted_on_now": "full TRAIN valid rows, recorded clip bounds kept",
     "proposed_class_as_fitted": "POOLED_INFORMATION_FROM_EXCLUDED_CLIENTS",
     "proposed_class_if_refit_on_pool": "COHORT_SPECIFIC_LEARNED_INFORMATION",
     "legitimately_available": {"as_fitted": "as the v1 normalizers", "pool_refit": "as the v1 normalizers"}},
    {"statistic": "rezero_map_v2 (unknown -> 0.0 after normalization; 7 gated channels + PAD)",
     "fitted_on_now": "nothing (a rule over each row's own flags/masks)",
     "proposed_class_as_fitted": "DEPLOYMENT_AVAILABLE_BOOTSTRAP_INFORMATION (protocol constant)",
     "proposed_class_if_refit_on_pool": "unchanged (no fitted content)",
     "legitimately_available": {"as_fitted": METHODS + " (all six)", "pool_refit": METHODS + " (all six)"}},
    {"statistic": "catalogue support counts (support >= 5 -> K = 158,486 output classes and the item vocabulary)",
     "fitted_on_now": "TRAIN representative views of all TRAIN users",
     "proposed_class_as_fitted": "POOLED_INFORMATION_FROM_EXCLUDED_CLIENTS used as the candidate-space definition "
                                 "(owner section 8 default: keep A)",
     "proposed_class_if_refit_on_pool": "COHORT_SPECIFIC_LEARNED_INFORMATION (catalogue B, K_B)",
     "legitimately_available": {"as_fitted": "all six must share one class space (a server-side catalogue); counts "
                                             "never leave the server as features",
                                "pool_refit": "C/CPF directly; FL methods via a federated count round; LO as shipped"}},
    {"statistic": "category / brand / main_category / category_code vocabularies",
     "fitted_on_now": "values seen on TRAIN rows of all users (no support threshold)",
     "proposed_class_as_fitted": "DEPLOYMENT_AVAILABLE_BOOTSTRAP_INFORMATION (catalogue attribute tables; the only "
                                 "behavioural input is 'seen in TRAIN')",
     "proposed_class_if_refit_on_pool": "same class; changes no pool TRAIN token (index renumbering only)",
     "legitimately_available": {"as_fitted": METHODS + " (all six)", "pool_refit": METHODS + " (all six)"}},
    {"statistic": "first_seen_train per product (item observed age and item_age_unavailable)",
     "fitted_on_now": "MIN(event time) per product over pooled TRAIN views of all users",
     "proposed_class_as_fitted": "POOLED_INFORMATION_FROM_EXCLUDED_CLIENTS (a popularity/recency statistic of other "
                                 "users' behaviour)",
     "proposed_class_if_refit_on_pool": "COHORT_SPECIFIC_LEARNED_INFORMATION (pooled over cohort clients)",
     "legitimately_available": {"as_fitted": "all six only as a shipped historical blob; a real deployment would "
                                             "need an item listing time from the catalogue (not in this dataset)",
                                "pool_refit": "C/CPF directly; FL methods via a federated MIN round; LO as shipped"}},
    {"statistic": "brand / category / category_code recovery tables (as-of state per product; frozen end-of-TRAIN "
                  "snapshot for future partitions)",
     "fitted_on_now": "pooled TRAIN observations of all users (baked into the v1 features)",
     "proposed_class_as_fitted": "POOLED_INFORMATION_FROM_EXCLUDED_CLIENTS (catalogue cleaning from other users' "
                                 "observations)",
     "proposed_class_if_refit_on_pool": "COHORT_SPECIFIC_LEARNED_INFORMATION (not recomputed here: needs the raw "
                                        "view table; exposure bound reported)",
     "legitimately_available": {"as_fitted": "all six only as shipped catalogue metadata",
                                "pool_refit": "as the vocabularies once rebuilt"}},
    {"statistic": "daypart / weekday closed vocabularies, the UTC+3h clock, clip quantile levels 0.001/0.999",
     "fitted_on_now": "declared constants (not fitted)",
     "proposed_class_as_fitted": "DEPLOYMENT_AVAILABLE_BOOTSTRAP_INFORMATION",
     "proposed_class_if_refit_on_pool": "unchanged",
     "legitimately_available": {"as_fitted": METHODS + " (all six)", "pool_refit": METHODS + " (all six)"}},
    {"statistic": "client_idx vocabulary", "fitted_on_now": "TRAIN users (identifier, not a feature)",
     "proposed_class_as_fitted": "not an information channel of any model", "proposed_class_if_refit_on_pool": "n/a",
     "legitimately_available": {"as_fitted": "n/a", "pool_refit": "n/a"}},
]


GROUP_VALUE_INDICATORS = {
    "ITEM_ID": ["item_known"], "ITEM_METADATA_TOKENS": ["category_known", "brand_known", "main_category_known"],
    "CALENDAR": "ALWAYS", "PRICE_EVENT": ["price_available"], "VIEW_GAP_EVENT": ["prev_gap_available"],
    "ITEM_AGE_POOLED_FIRST_SEEN": ["item_age_available"], "SEGMENT_PREFIX": "ALWAYS",
    "METADATA_QUALITY_FLAGS": None, "SIMULTANEOUS_BLOCK": None, "USER_HISTORY_COUNTS": ["history_available"],
    "USER_MATCH": ["category_match_supported", "brand_match_supported"], "USER_PRICE": ["recent_price_supported"],
    "USER_TIME_SINCE": ["history_available"], "USER_CONTEXT_ALL": ["history_available", "recent_price_supported"]}


def group_value_avail(g, ao):
    spec = GROUP_VALUE_INDICATORS.get(g)
    if spec is None:
        return None
    if spec == "ALWAYS":
        return 1.0
    sh = ao["indicator_available_share"]
    if any(sh[k] >= 1.0 for k in spec):
        return 1.0
    if len(spec) == 1:
        return sh[spec[0]]
    names = ao["cooccurrence_counts"]["names"]
    A = ao["cooccurrence_counts"]["matrix"]
    i, j = names.index(spec[0]), names.index(spec[1])
    n = A[i][i] / sh[spec[0]] if sh[spec[0]] else None
    if len(spec) == 2 and n:
        return (A[i][i] + A[j][j] - A[i][j]) / n
    return max(sh[k] for k in spec)


def pct(x, nd=4):
    return "n/a" if x is None else f"{100 * x:.{nd}f}%"


def f(x, nd=6):
    if x is None:
        return "n/a"
    if isinstance(x, int):
        return f"{x:,}"
    return f"{x:.{nd}g}"


def main(d: Path):
    S = json.loads((d / "FEATSTAT_STATS.json").read_text(encoding="utf-8"))
    FD = json.loads((d / "FEATSTAT_FD.json").read_text(encoding="utf-8"))
    P = json.loads((d / "FEATSTAT_PROVENANCE.json").read_text(encoding="utf-8"))
    R = json.loads((d / "FEATSTAT_RECEIPT.json").read_text(encoding="utf-8"))
    LAD = json.loads((d / "FEATSTAT_LADDER.json").read_text(encoding="utf-8"))
    (d / "FEATSTAT_PROVENANCE_CLASSIFICATION.json").write_text(json.dumps(
        {"task": "FEATSTAT-01", "status": "PROPOSED (the POPULATION_5_100 deliberation decides)",
         "classes": ["DEPLOYMENT_AVAILABLE_BOOTSTRAP_INFORMATION", "COHORT_SPECIFIC_LEARNED_INFORMATION",
                     "POOLED_INFORMATION_FROM_EXCLUDED_CLIENTS"],
         "methods": {"C": "centralized", "FA": "FedAvg", "FP": "FedProx", "PF": "personalized FL",
                     "CPF": "centralized-personalized", "LO": "local-only"},
         "rows": CLASSIFICATION}, indent=1), encoding="utf-8")
    cs = S["channel_stats"]
    st = S["status"]
    tr = S["transfer"]
    reg = S["registry"]
    L = []
    w = L.append
    N, Np = S["n_rows"]["FULL"], S["n_rows"]["POOL"]
    rep = R["population_reproduction"]
    w("# FEATSTAT-01 — feature statistics and normalizer provenance on the 5–100 MAIN pool (TRAIN only)\n")
    w(f"- Evidence: PROJECT_MEASURED by `code/featstat_v1` on the server; receipt stage `{R['stage']}`, "
      f"`date -u` {R['date_u_start']} → {R['date_u_written']}, wall {R['wall_seconds']} s, peak RSS "
      f"{R['peak_rss'].get('VmHWM', 0):.0f} MiB (guard 16,384).")
    w(f"- Pool (5 ≤ n ≤ 100): {rep['pool_users']['measured']:,} users / {Np:,} supervised examples; comparison "
      f"(n ≥ 1): {rep['nge1_users']['measured']:,} / {N:,}. All {len(rep)} census counts reproduced: "
      f"{R['population_reproduction_all_equal']}.")
    v = R["clean_v2_verification"]
    w(f"- Clean V2 rows via v4data: {v['store_batch_windows_checked']:,} real CleanV2Store windows compared byte for "
      f"byte, mismatches {v['store_batch_mismatches']}; encode_rows max |diff| "
      f"{max(v['encode_rows_max_abs_diff'].values()):.3g}; RC1–RC7 on every decision row: "
      f"{sum(v['rc_counts_view'].values())} violations. Inputs sha256 = FILES.json: "
      f"{R['inputs_read_sha256_checked_against_FILES_json']} files, mismatches {len(R['input_hash_mismatches'])}.")
    w("- Grain: one row per supervised TRAIN example (its own event = last_valid_index). The JSON also holds a "
      "window-exposure grain computed under the v1 SESSION_L50 window; the owner has since approved USER_HISTORY_L50 "
      "(DECISION_LOG 35-36), so those window numbers are SUPERSEDED, descriptive only, and not used by any rule "
      "here; the window statistics of record are DATA-05's. VALIDATION/TEST not read.\n")

    w("## 1. Near-constant and dead channels on the pool (mode share)\n")
    w("| channel | pool mode share | full mode share | pool prevalence / default share | status change |")
    w("|---|---|---|---|---|")
    for r_ in reg:
        n = r_["name"]
        a, b = cs[n]["POOL"], cs[n]["FULL"]
        if a["mode_share"] >= 0.99:
            pv = a.get("prevalence", a.get("default_share"))
            w(f"| {n} | {a['mode_share']:.6f} | {b['mode_share']:.6f} | {pct(pv, 5)} | "
              f"{', '.join(tr[n]['status_changes_pool_vs_full']) or '—'} |")
    w("")
    w("## 2. Exact functional dependencies on the pool (0 violations; informative determinants only)\n")
    names = FD["names"]
    ix = {n: i for i, n in enumerate(names)}
    VP, VF = FD["violations"]["POOL"], FD["violations"]["FULL"]
    D = FD["distinct_full"]
    w("| X → Y | violations pool | violations full | note |")
    w("|---|---|---|---|")
    seen = set()
    for y in names:
        if D[y] == 1:
            continue
        for x in names:
            if x == y or D[x] == 1:
                continue
            if VP[ix[x]][ix[y]] == 0 and cs[x]["POOL"]["rows_in_multirow_value_groups_share"] >= 0.5:
                key = tuple(sorted((x, y)))
                both = VP[ix[y]][ix[x]] == 0
                if both and key in seen:
                    continue
                seen.add(key)
                w(f"| {x} {'↔' if both else '→'} {y} | {VP[ix[x]][ix[y]]} | {VF[ix[x]][ix[y]]} | "
                  f"{'one-to-one' if both else ''} |")
    w("")
    w("Multi-column determinants:\n")
    for m in FD["multi_column"]:
        if m["population"] == "POOL":
            full = next(z for z in FD["multi_column"] if z["population"] == "FULL" and z["determinant"] == m["determinant"]
                        and z["dependent"] == m["dependent"])
            w(f"- ({', '.join(m['determinant'])}) → {m['dependent']}: pool {m['violations']:,} "
              f"({pct(m['violation_rate'], 4)}), full {full['violations']:,}")
    w("")
    w("## 3. High-similarity clusters (single linkage; tensor order)\n")
    for pop in POPS:
        for k, cl in S["clusters"][pop].items():
            txt = "; ".join("{" + ", ".join(c["members"]) + f"}} min|r| {c['min_abs_r']:.4f}" for c in cl) or "none"
            w(f"- **{pop} {k}:** {txt}")
    w("")
    w("## 4. Missingness (mask-based) and availability overlap\n")
    w("| channel | availability indicator | pool unavailable | full unavailable | pool V2 exact-zero share |")
    w("|---|---|---|---|---|")
    for n, e in S["missingness"]["per_channel"].items():
        w(f"| {n} | {e['availability_indicator']} | {pct(e['POOL']['mask_based_unavailable_share'])} | "
          f"{pct(e['FULL']['mask_based_unavailable_share'])} | {pct(e['POOL']['sentinel_based_v2_exact_zero_share'])} |")
    w("")
    w("## 5. Feature-group prevalence (share of pool / full examples)\n")
    w("Value availability = any value channel of the group carries an observed (not unknown) value; DERIVED from the "
      "measured indicator co-occurrence counts (inclusion-exclusion). Non-default = any member differs from its "
      "default/unknown encoding (tokens: known), MEASURED.\n")
    w("| group | pool value available | pool non-default | full value available | full non-default |")
    w("|---|---|---|---|---|")
    ao = S["missingness"]["availability_overlap"]
    for g, e in S["feature_group_prevalence"].items():
        va = {pop: group_value_avail(g, ao[pop]) for pop in POPS}
        w(f"| {g} | {pct(va['POOL'], 3) if va['POOL'] is not None else 'n/a (indicators only)'} | "
          f"{pct(e['POOL']['any_non_default_share'], 3)} | "
          f"{pct(va['FULL'], 3) if va['FULL'] is not None else 'n/a'} | {pct(e['FULL']['any_non_default_share'], 3)} |")
    w("")
    w("## 6. Transfer check (pool vs n ≥ 1)\n")
    ch = {n: t["status_changes_pool_vs_full"] for n, t in tr.items() if t["status_changes_pool_vs_full"]}
    w(f"- Channels whose status changes: {json.dumps(ch) if ch else 'none'}")
    ct = S["correlation_transfer"]
    w(f"- Max |Δ Pearson| {ct['max_abs_pearson_diff']:.4f} at {ct['at']}; max |Δ Spearman| "
      f"{ct['max_abs_spearman_diff']:.4f} at {ct['at_spearman']}; pairs with |Δ| > 0.05: Pearson "
      f"{ct['pairs_abs_pearson_diff_gt_0.05']}, Spearman {ct['pairs_abs_spearman_diff_gt_0.05']}.")
    ft = FD.get("fd_transfer") or {}
    w(f"- Ordered pairs whose exact-FD status differs: {ft.get('ordered_pairs_exact_status_differs')}")
    big = sorted(((n, t["mean_shift_in_full_sd"]) for n, t in tr.items() if t.get("mean_shift_in_full_sd") is not None),
                 key=lambda z: -abs(z[1]))[:8]
    w("- Largest mean shifts (pool − full, in full SD): " + ", ".join(f"{n} {s:+.3f}" for n, s in big))
    w("")
    w("## 7. Normalizer and shared-statistic provenance (full TRAIN vs 5–100 pool refit)\n")
    w("| channel | full clip [lo, hi] / mean / std | pool refit clip [lo, hi] / mean / std | max / mean |Δz| on pool "
      "examples |")
    w("|---|---|---|---|")
    for n, e in P["normalizers"].items():
        r = e["recorded_v2"]
        c = e["pool_only_candidate"]
        ef = e["effect_on_transformed_values"].get("POOL_SUPERVISED_EXAMPLES|candidate", {})
        w(f"| {n} | [{f(r['clip_lo'], 5)}, {f(r['clip_hi'], 5)}] / {f(r['mean'], 6)} / {f(r['std'], 6)} | "
          f"[{f(c['clip_lo'], 5)}, {f(c['clip_hi'], 5)}] / {f(c['mean'], 6)} / {f(c['std'], 6)} | "
          f"{f(ef.get('max_abs_change_z'), 4)} / {f(ef.get('mean_abs_change_z_valid_rows'), 4)} |")
    cat = P["catalogue"]
    w("")
    w(f"- Catalogue: K_A {cat['K_A_full_train_support_ge5']:,}, K_B (pool-only support ≥ 5) "
      f"{cat['K_B_pool_only_support_ge5']:,}, A∖B {cat['A_minus_B']:,}; pool examples whose current item leaves B "
      f"{pct(cat['pool_examples_current_item_not_in_B_share'])} (SESSION_L50 window positions, superseded: "
      f"{pct(cat['pool_window_positions_item_not_in_B_share'])}); pool targets whose class leaves B "
      f"{cat['pool_targets_class_not_in_B']:,} ({pct(cat['pool_targets_class_not_in_B_share'])}); support = "
      f"catalogue.representative_views mismatches {cat['support_full_train_equals_catalogue_representative_views_mismatches']}.")
    fs = P["first_seen_item_age"]
    w(f"- First-seen proxy validation: availability agreement {pct(fs['proxy_validation_on_pool_train_rows']['availability_agreement_share'])}, "
      f"exact value agreement {pct(fs['proxy_validation_on_pool_train_rows']['value_exact_float32_agreement_share_when_available'])}.")
    for k, e in fs["effect_vs_current"].items():
        if k.startswith("POOL_SUPERVISED_EXAMPLES"):
            w(f"  - {k.split('|')[1]}: max |Δz| {e['max_abs_change_z']:.4f}, mean |Δz| {e['mean_abs_change_z']:.4f}")
    fl = fs["availability_flips"]["POOL_SUPERVISED_EXAMPLES"]
    w(f"  - availability flips (available under full first-seen, unavailable under pool first-seen): "
      f"{fl['available_full_but_not_pool']:,} of {fl['rows']:,}")
    w("")
    w("## 8. Proposed compact-candidate ladder (statistics only; nested; identical for both families)\n")
    w(f"- Rungs nested: {LAD.get('rungs_nested')}; judging rule: {LAD['judging_rule_reminder']}.")
    w("| rung | channels | tokens / num / flags / uctx / masks | GRU params (Δ) | SASRec params (Δ) | removed here |")
    w("|---|---|---|---|---|---|")
    for r in LAD["rungs"]:
        iw = r["input_widths"]
        rem = ", ".join(x["channel"] for x in r["removed_here"]) or "—"
        w(f"| {r['id']} | {r['n_information_channels']} | {iw['tokens']} / {iw['numeric_dim']} / {iw['flags_dim']} / "
          f"{iw['user_context_dim']} / {iw['user_mask_dim']} | {r['gru']['params_total']:,} ({r['gru']['delta_vs_clean_v2']:+,}) | "
          f"{r['sasrec']['params_total']:,} ({r['sasrec']['delta_vs_clean_v2']:+,}) | {rem} |")
    w("")
    w("### USER_HISTORY_L50 column (hypotheses for the experiments, not drop decisions)\n")
    w("| channel | would a cross-session 50-event window carry it directly? | basis |")
    w("|---|---|---|")
    for n, e in LAD["user_history_l50_hypotheses"].items():
        w(f"| {n} | {e['hypothesis']} | {e['basis']} |")
    (d / "FEATSTAT_SUMMARY.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("wrote", d / "FEATSTAT_SUMMARY.md")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
