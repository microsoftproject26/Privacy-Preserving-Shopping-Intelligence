"""write_fo_report.py (owner: Opus, lead): FINAL_FEATURE_OPT_REPORT_AR.md (+ _EN companion) for SASREC_FEATURE_OPT_V1,
written only from fetched study result files (copies under final_inputs/). Read-only on its inputs.

  python report_tools/write_fo_report.py --inputs final_inputs --out final
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


def jl(p: Path):
    return json.loads(p.read_text(encoding="utf-8"))


def f6(x):
    return "—" if x is None or x == "" else f"{float(x):.6f}"


def sg(x):
    return "—" if x is None or x == "" else f"{float(x):+.6f}"


def ci(b):
    return "—" if not b else f"[{b['ci95'][0]:+.6f}, {b['ci95'][1]:+.6f}]"


def facts(inp: Path) -> dict:
    fin = jl(inp / "FINAL_FEATURE_OPT_RESULTS.json")
    probes = jl(inp / "audit/GROUP_PROBE_RESULTS.json")
    lean = jl(inp / "audit/LEAN_PARITY.json")
    m1 = jl(inp / "audit/M1_FIT.json")
    dup = jl(inp / "audit/DUPLICATES_FULL.json")
    pin = jl(inp / "config/BASELINE_PIN.json")
    noise = jl(inp / "config/NOISE_REFERENCE.json")
    sel = jl(inp / "SELECTED_FEATURES.json")
    ts = jl(inp / "TRANSFORM_STATE.json")
    st = jl(inp / "state/fo_status.json")
    arms = {a: jl(p / "result.json") for a in ("CONTROL", "C1_D3", "C2") if (p := inp / "runs/screen" / a).joinpath("result.json").exists()}
    rows = list(csv.DictReader(open(inp / "FEATURE_SELECTION_RESULTS.csv", encoding="utf-8")))
    pd = sorted(inp.glob("audit/PROTECT_DIFF_T0_study_start_T2_study_done.json"))
    prot = jl(pd[0]) if pd else None
    return {"fin": fin, "probes": probes, "lean": lean, "m1": m1, "dup": dup, "pin": pin, "noise": noise, "sel": sel, "ts": ts, "st": st,
            "arms": arms, "rows": rows, "prot": prot,
            "inputs_sha256": {str(p.relative_to(inp)).replace("\\", "/"): hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in sorted(inp.rglob("*")) if p.is_file()}}


def tables(F: dict) -> dict:
    t = {}
    L = ["| Probe | type | Δ macro MRR@20 | 95% CI | Δ micro R@20 | weekday Δ | weekend Δ |", "|---|---|---|---|---|---|---|"]
    for n, p in F["probes"]["probes"].items():
        s = p.get("slices", {})
        L.append(f"| {n} | {p.get('label')} | {sg(p['delta_macro_mrr@20'])} | {ci(p['bootstrap_macro_mrr@20'])} | {sg(p['delta_micro_recall@20'])} | "
                 f"{sg(s.get('weekday', {}).get('delta_macro_mrr@20'))} | {sg(s.get('weekend', {}).get('delta_macro_mrr@20'))} |")
    t["probes"] = "\n".join(L)
    L = ["| Arm | cohort macro MRR@20 | cohort micro R@20 | Δ vs CONTROL | 95% CI | gate 2(b) |", "|---|---|---|---|---|---|"]
    for a, r in F["arms"].items():
        s = F["fin"]["screen"].get(a, {})
        g = s.get("gate_2b")
        L.append(f"| {a} | {f6(r['cohort_metrics']['macro_mrr@20'])} | {f6(r['cohort_metrics']['micro_recall@20'])} | "
                 f"{sg(s.get('delta_macro_mrr@20_vs_control')) if a != 'CONTROL' else '—'} | {ci(s.get('bootstrap')) if a != 'CONTROL' else '—'} | "
                 f"{('PASS' if s.get('passes') else 'FAIL ' + json.dumps(g)) if a != 'CONTROL' else 'control'} |")
    t["screen"] = "\n".join(L)
    L = ["| Stage | state | attempt | end (UTC) | disk free GiB |", "|---|---|---|---|---|"]
    for n, s in F["st"]["stages"].items():
        L.append(f"| {n} | {s.get('state')} | {s.get('attempt')} | {s.get('end_utc')} | {s.get('disk_free_gib')} |")
    t["stages"] = "\n".join(L)
    L = ["| Feature | mask | fitted mean / std (all TRAIN rows) | valid-only mean / std (TRAIN) | VS valid mean |", "|---|---|---|---|---|"]
    for n, x in F["m1"]["features"].items():
        fn = x["fitted_normalizer"]
        L.append(f"| {n} | {x['mask']} | {fn['mean']:.4f} / {fn['std']:.4f} | {x['train_valid']['mean']:.4f} / {x['train_valid']['std']:.4f} | "
                 f"{x['vs_valid_drift_only']['mean']:.4f} |")
    t["m1"] = "\n".join(L)
    return t


def report(F: dict, t: dict, lang: str) -> str:
    fin, lean, ts, sel, pin, noise = F["fin"], F["lean"], F["ts"], F["sel"], F["pin"], F["noise"]
    dec, thr = fin["decision"], fin["gate_2b_threshold"]
    par = ts["parity"]
    failed_checks = [k for k, v in par["checks"].items() if not v]
    c1 = fin["screen"].get("C1_D3", {})
    ar = lang == "ar"
    A = []
    w = A.append
    if ar:
        w("# التقرير النهائي لدراسة الخصائص — `SASREC_FEATURE_OPT_V1`")
        w("")
        w("> دراسة إضافية منفصلة عن حملة SASRec. **كل الأرقام على VALIDATION_SELECT فقط** (عيّنة 10k مستخدم، 112,365 قرار)، وهو قسم استُخدم")
        w("> سابقًا لاختيار نماذج الحملة؛ إذن هذا تطوير تكيّفي وليس holdout جديدًا. لم يُقرأ VC ولا TEST إطلاقًا.")
        w("")
        w("## 1. الخلاصة")
        w(f"- **القرار: `{dec}`**" + (" — يبقى النموذج الحالي المثبّت دون تغيير." if dec == "KEEP_CURRENT" else f" — المرشح النهائي `{fin['finalist']}` يحتاج زوجًا كاملًا مسجلًا مسبقًا."))
        w(f"- النموذج المثبّت (baseline pin): A1_RICH256 seed {pin['chosen']['seed']} epoch {pin['chosen']['epoch']} (VS {f6(pin['chosen']['macro_mrr20'])}). "
          f"σ̂ = {f6(noise['sigma_hat'])} (أقصى فرق بين بذرتين؛ مدى وليس انحرافًا معياريًا) → عتبة البوابة 2(b) = max(0.002, 2σ̂) = {f6(thr)}.")
        if c1:
            w(f"- **C1_D3** (5 قنوات تكرار سببية، تهيئة صفرية) مقابل CONTROL بعد نفس التكيّف الدافئ (5M قرار): Δ {sg(c1['delta_macro_mrr@20_vs_control'])}، "
              f"CI {ci(c1['bootstrap'])}، Δ micro R@20 {sg(c1['delta_micro_recall@20_vs_control'])} → البوابة 2(b): **{'نجح' if c1['passes'] else 'فشل'}**.")
        w(f"- **Probes (اعتماد النموذج الحالي، ليست إعادة تدريب):** لا يوجد مرشح إزالة يجتاز البوابة 2(a) (Δ ≥ +0.001 وCI > 0 وكلا الشريحتين موجبتان). "
          "كل خصائص الجانب تُستخدم فعلًا: إخفاؤها يضر.")
        w(f"- **النظافة (lean view):** max|Δlogit| {lean['max_abs_logit_diff_first_1024']:.2e}، تطابق الرتب {lean['ranks_identical_fraction']:.6f} "
          f"({lean['n_rank_changes']} تغيير)، Δ المقياس {lean['delta_macro_mrr@20']:.1e} → الحالة **{par['status']}**"
          + (f" (فشل: {', '.join(failed_checks)})؛ لا ادعاء تكافؤ، والمخطط المختار هو F4 الحالي." if failed_checks else "؛ إعادة تمثيل مكافئة (نظافة فقط، ليست مكسب دقة)."))
        base = F["fin"]["baseline_cohort"]["macro_mrr@20"]
        if "CONTROL" in F["arms"]:
            cm = F["arms"]["CONTROL"]["cohort_metrics"]["macro_mrr@20"]
            w(f"- **(MEASURED)** التكيّف الدافئ نفسه خفّض المقياس: CONTROL {f6(cm)} مقابل النموذج المثبّت {f6(base)} ({sg(cm - base)})"
              + (f"؛ وC1_D3 {f6(F['arms']['C1_D3']['cohort_metrics']['macro_mrr@20'])} ({sg(F['arms']['C1_D3']['cohort_metrics']['macro_mrr@20'] - base)}). "
                 "كلا الذراعين أقل من النموذج الحالي، إذن لا يوجد ما يُعتمد." if "C1_D3" in F["arms"] else "."))
        pr = F.get("prot")
        if pr:
            w(f"- **الحماية T0→T2:** شجرة البيانات متطابقة = {pr['dataset_tree_equal']}؛ ميتاداتا البيانات المتغيرة = {pr['dataset_metadata_changed']}؛ "
              f"ملفات قائمة تغيّرت = {pr['changed_existing']} (أداة المراقبة للقراءة فقط التي يملكها القائد)؛ الإضافات = ملفات قفل الحملة وأدوات تقريرها (متوقعة).")
        w("- **لا ادعاء سببي ولا SOTA.** Probes = اعتماد نقطة تفتيش واحدة؛ الـscreen تكيّف دافئ استكشافي، بذرة واحدة.")
        w("")
        w("## 2. النطاق والتفويض")
        w("- قرار Fable (مفوض المالك): STAGED_WITH_EVIDENCE_GATES (`review/AUTO_DECISION_SCOPE.md`)؛ مراجعة الكود C: APPROVE_FOR_ISOLATED_EXPERIMENTS بعد FIX_REQUIRED؛ "
          "تصحيح FO-02 (عزل الاستيراد): APPROVE_RESUME (`review/fo02/`).")
        w("- التقاط لقطة من كود الحملة بايتيًا (`code/snap_sasrec`)؛ لا GPU قبل انتهاء الحملة؛ الملاءمة على TRAIN فقط؛ لا حذف بالقناع فقط؛ لا log مزدوج؛ لا VC/TEST.")
        w("")
        w("## 3. التدقيق المقاس (MEASURED، كل السكان)")
        dchk = F["dup"]["checks"]
        nz = {sp: {k: v for k, v in d.items() if k != "local_hour_rotation_max_residual" and int(v)} for sp, d in dchk.items()}
        w(f"- التكرارات الدقيقة والأعلام الميتة على كل السكان (TRAIN {F['dup']['rows']['TRAIN']:,} صف، VS {F['dup']['rows']['VALIDATION_SELECT']:,}): "
          f"مخالفات غير صفرية = {nz}؛ أقصى باقي دوران local_hour = {max(d['local_hour_rotation_max_residual'] for d in dchk.values()):.1e}. "
          "(8 أعمدة مكررة في 6 علاقات + 6 أعلام ميتة؛ تصحيح العدد في AMENDMENT_FO_01.)")
        w("- **عيب M1 (مؤكد على كل TRAIN):** مُطبِّع ميزتين من ميزات الفجوة الزمنية مُلائم على كل الصفوف بما فيها الأصفار المقنّعة:")
        w(t["m1"])
        w("- الإصلاح (ملاءمة على الصفوف الصالحة فقط مع حدود القص المسجلة) جزء من TRANSFORM_STATE؛ هو نظافة، وليس مكسب دقة.")
        w("")
        w("## 4. Probes (اعتماد النموذج المثبّت على كل مجموعة)")
        w(t["probes"])
        w("")
        w("- التفسير: " + F["probes"]["interpretation"] + ". P1 مرساة خارج التوزيع؛ P8 counterfactual لنقطة واحدة.")
        w("")
        w("## 5. الـScreen (تكيّف دافئ استكشافي، ≤ 6 GPU-h)")
        w(t["screen"])
        w("")
        w("- كل الأذرع: نفس الأوزان المثبّتة، AdamW جديد، نفس 5,000,000 قرار TRAIN بنفس الترتيب، 19,532 تحديث، warmup 200 ثم cosine إلى 1e-5، seed 2026، TF32 تدريب / FP32 تقييم.")
        w("")
        w("## 6. المخرجات")
        w(f"- `SELECTED_FEATURES.json`: decision `{sel['decision']}`، model_in_use «{sel['model_in_use']}»، selected_schema «{sel['selected_schema']}».")
        w(f"- `TRANSFORM_STATE.json`: status `{ts['status']}`، fit_split {ts['fit_split']}، view_sha256 `{ts['view_sha256'][:16]}…`.")
        w("- `FEATURE_SELECTION_RESULTS.csv`، `FINAL_FEATURE_OPT_RESULTS.json`، `state/FINAL_DECISION.json` (كتابة مرة واحدة).")
        w("")
        w("## 7. مراحل التنفيذ والموارد")
        w(t["stages"])
        w("")
        w("## 8. حوادث وتصحيحات")
        w("- FO-01: عدّ «Nine» كان علاقات لا أعمدة → 8 أعمدة في 6 علاقات؛ D3 = 4 قنوات مفاهيمية = 5 قيم (AMENDMENT_FO_01).")
        w("- اختبارات Sonnet B المستقلة كشفت عيبين (هروب مسار بالاسم الشقيق؛ ملف فارغ عند رفض NaN) → أُصلحا. مراجعة C: 8 ملاحظات أُصلحت.")
        w("- P4: تجاوز uint8 في dtype القناع → أُصلح قبل التشغيل. تشغيل تجريبي تجاوز حارس الذاكرة → قُتل خلال 2.6 ث بلا أثر.")
        w("- FO-02: `train.py` المعاد استخدامه من GRU يضع `gru_v1/code` أولًا في sys.path؛ فشل probes مرتين (حارس العزل) → إصلاح في `fo_env.py` + 3 اختبارات → "
          "Fable APPROVE_RESUME؛ المراحل المكتملة صالحة (14/14 ملفًا متطابقًا بايتيًا).")
        w("")
        w("## 9. الحدود")
        w("- VS أعيد استخدامه؛ لا holdout جديد؛ لا VC/TEST. Probes تقيس اعتماد نقطة تفتيش، وليست أثر إزالة مع إعادة تدريب. "
          "الـscreen تكيّف دافئ على 5M قرار ببذرة واحدة، وليس تدريبًا من الصفر. لا ادعاء بأن أي خاصية «سببية».")
    else:
        w("# Feature study final report — `SASREC_FEATURE_OPT_V1` (English companion)")
        w("")
        w("> A separate add-on study to the SASRec campaign. **All numbers are on VALIDATION_SELECT only** (10k-user cohort, 112,365 decisions),")
        w("> a split already used for the campaign's selection; this is adaptive development, not a fresh holdout. VC and TEST were never read.")
        w("")
        w("## 1. Summary")
        w(f"- **Decision: `{dec}`**" + (": the pinned incumbent stays unchanged." if dec == "KEEP_CURRENT" else f": finalist `{fin['finalist']}` requires the registered full pair."))
        w(f"- Pin: A1_RICH256 seed {pin['chosen']['seed']} epoch {pin['chosen']['epoch']} (VS {f6(pin['chosen']['macro_mrr20'])}). σ̂ = {f6(noise['sigma_hat'])} "
          f"(max two-seed gap; a range, not an SD) → gate-2(b) threshold max(0.002, 2σ̂) = {f6(thr)}.")
        if c1:
            w(f"- **C1_D3** (5 causal repeat channels, zero-init) versus CONTROL after the same warm-start adaptation (5M decisions): Δ {sg(c1['delta_macro_mrr@20_vs_control'])}, "
              f"CI {ci(c1['bootstrap'])}, Δ micro R@20 {sg(c1['delta_micro_recall@20_vs_control'])} → gate 2(b): **{'PASS' if c1['passes'] else 'FAIL'}**.")
        w("- **Probes (reliance of the incumbent, not retraining):** no removal candidate passes gate 2(a) (Δ ≥ +0.001, CI > 0, both slices positive). "
          "Every side-feature group is used: blanking it hurts.")
        w(f"- **Hygiene (lean view):** max|Δlogit| {lean['max_abs_logit_diff_first_1024']:.2e}, rank identity {lean['ranks_identical_fraction']:.6f} "
          f"({lean['n_rank_changes']} changes), metric Δ {lean['delta_macro_mrr@20']:.1e} → status **{par['status']}**"
          + (f" (failed: {', '.join(failed_checks)}); no equivalence claim, and the selected schema is the current F4." if failed_checks else "; an equivalent re-parametrisation (hygiene only, not an accuracy gain)."))
        base = F["fin"]["baseline_cohort"]["macro_mrr@20"]
        if "CONTROL" in F["arms"]:
            cm = F["arms"]["CONTROL"]["cohort_metrics"]["macro_mrr@20"]
            w(f"- **(MEASURED)** The warm-start adaptation itself lowered the metric: CONTROL {f6(cm)} versus the pinned incumbent {f6(base)} ({sg(cm - base)})"
              + (f"; C1_D3 {f6(F['arms']['C1_D3']['cohort_metrics']['macro_mrr@20'])} ({sg(F['arms']['C1_D3']['cohort_metrics']['macro_mrr@20'] - base)}). "
                 "Both arms are below the incumbent, so nothing is adopted." if "C1_D3" in F["arms"] else "."))
        pr = F.get("prot")
        if pr:
            w(f"- **Protection T0→T2:** dataset tree equal = {pr['dataset_tree_equal']}; dataset metadata changed = {pr['dataset_metadata_changed']}; "
              f"changed existing files = {pr['changed_existing']} (the lead's read-only monitor); additions = the campaign's lock files and report tools (expected).")
        w("- **No causal and no SOTA claim.** Probes measure one checkpoint's reliance; the screen is exploratory warm-start adaptation with one seed.")
        w("")
        w("## 2. Measured audit (full population)")
        dchk = F["dup"]["checks"]
        nz = {sp: {k: v for k, v in d.items() if k != "local_hour_rotation_max_residual" and int(v)} for sp, d in dchk.items()}
        w(f"- Exact duplicates and dead flags on the full population (TRAIN {F['dup']['rows']['TRAIN']:,} rows, VS {F['dup']['rows']['VALIDATION_SELECT']:,}): "
          f"non-zero violations = {nz}; max local_hour rotation residual = {max(d['local_hour_rotation_max_residual'] for d in dchk.values()):.1e}.")
        w("- **M1 defect (confirmed on all TRAIN):** two time-gap normalizers were fitted on all rows including the masked zeros:")
        w(t["m1"])
        w("")
        w("## 3. Probes")
        w(t["probes"])
        w("")
        w("## 4. Screen (exploratory warm-start adaptation)")
        w(t["screen"])
        w("")
        w("## 5. Deliverables")
        w(f"- `SELECTED_FEATURES.json`: decision `{sel['decision']}`, model in use \"{sel['model_in_use']}\", selected schema \"{sel['selected_schema']}\".")
        w(f"- `TRANSFORM_STATE.json`: status `{ts['status']}`, fit split {ts['fit_split']}.")
        w("")
        w("## 6. Stages")
        w(t["stages"])
        w("")
        w("## 7. Incidents")
        w("- FO-01 (counts/names), Sonnet B defects, Review C findings, the P4 dtype bug, the memory-guard kill: all fixed before running.")
        w("- FO-02: the reused GRU `train.py` puts `gru_v1/code` first on sys.path; probes failed twice on the isolation guard → fix in `fo_env.py` "
          "(+3 tests) → Fable APPROVE_RESUME; DONE stages valid (14/14 byte-identical).")
        w("")
        w("## 8. Limits")
        w("- VS reused; no fresh holdout; no VC/TEST. Probes are reliance, not removal-with-retraining. The screen is a one-seed warm start on 5M decisions.")
    return "\n".join(A) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()
    F = facts(a.inputs)
    t = tables(F)
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "FINAL_FEATURE_OPT_REPORT_AR.md").write_text(report(F, t, "ar"), encoding="utf-8")
    (a.out / "FINAL_FEATURE_OPT_REPORT_EN.md").write_text(report(F, t, "en"), encoding="utf-8")
    (a.out / "_report_inputs_sha256.json").write_text(json.dumps(F["inputs_sha256"], indent=1), encoding="utf-8")
    print(json.dumps({"decision": F["fin"]["decision"], "parity": F["ts"]["parity"]["status"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
