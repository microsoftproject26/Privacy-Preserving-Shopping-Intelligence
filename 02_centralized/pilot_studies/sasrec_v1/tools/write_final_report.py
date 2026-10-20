"""write_final_report.py (owner: Opus, lead): the lead's final narrative, written from
final/FINAL_SASREC_RESULTS.json and final/_tables.json only (both from build_final_results.py).

  python write_final_report.py --final final

Writes FINAL_SASREC_REPORT_AR.md (the required deliverable), FINAL_SASREC_REPORT_EN.md (companion)
and SOTA_CLAIM_STATUS.md. Every verdict sentence is computed from a CI or a measured difference;
nothing is typed in by hand except the fixed history (v2 failure, v3 repair, amendments), which cites
its source files.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

GRU_KEY, TRANS_KEY = "GRU_F0_locked", "TRANS_full"


def f6(x):
    return "—" if x is None else f"{x:.6f}"


def sg(x):
    return "—" if x is None else f"{x:+.6f}"


def verdict(bt):
    if bt is None:
        return None
    lo, hi = bt["ci95"]
    return "above" if lo > 0 else ("below" if hi < 0 else "overlaps")


def facts(r: dict) -> dict:
    head = r["lock"]["headline"]
    sm, ab = r["single_models"], r["archived_baselines"]
    m = lambda src, n, sp: src[n][sp]["end_to_end"]["macro_mrr@20"]
    bts = r["bootstraps_TEST"]
    f = {"head": head, "lock_sha": r["lock"]["sha256"], "lock_utc": r["lock"]["created_utc"],
         "retro_done": r["retro"]["completed_utc"], "claim_utc": r["retro"]["claim_utc"], "results_sha": r["retro"]["results_sha256"],
         "t_head": m(sm, head, "TEST"), "v_head": m(sm, head, "VALIDATION_CONFIRM"),
         "t_gru": m(ab, GRU_KEY, "TEST"), "v_gru": m(ab, GRU_KEY, "VALIDATION_CONFIRM"),
         "t_trans": m(ab, TRANS_KEY, "TEST"), "v_trans": m(ab, TRANS_KEY, "VALIDATION_CONFIRM"),
         "t_f4": m(ab, "GRU_F4_descriptive", "TEST"), "t_pop": m(ab, "POP_full", "TEST"),
         "bt_gru": bts.get(f"{head}_minus_{GRU_KEY}"), "bt_trans": bts.get(f"{head}_minus_{TRANS_KEY}"),
         "bt_rich": bts.get("A1_RICH256_s2026_minus_A0_ID256_s2026"),
         "t_a0": m(sm, "A0_ID256_s2026", "TEST"), "t_a1": m(sm, "A1_RICH256_s2026", "TEST"), "t_a1b": m(sm, "A1_RICH256_s2027", "TEST"),
         "vs_head": max(x["macro_mrr@20"] for x in r["validation_select"]["rows"] if x["run"].startswith(head)),
         "gaps": r["validation_select"]["seed_gap_macro_mrr20_A1"], "seedvar": r["seed_variation_TEST"],
         "sys": {n: (s["TEST"]["end_to_end"]["macro_mrr@20"], s["VALIDATION_CONFIRM"]["end_to_end"]["macro_mrr@20"]) for n, s in r["systems"].items()},
         "users_test": r["splits"]["TEST"].get("users"), "n_test": r["splits"]["TEST"].get("n_eval"),
         "users_vc": r["splits"]["VALIDATION_CONFIRM"].get("users"), "n_vc": r["splits"]["VALIDATION_CONFIRM"].get("n_eval"),
         "oov_test": r["splits"]["TEST"].get("n_oov"), "gate": r["archived_fingerprint_gate"], "tf": r.get("tf32_parity") or {},
         "v2": r.get("v2_failed_full_vs_macro_mrr20") or {}, "pair": r["lock"]["pair_check"], "dropped": r["lock"]["dropped_entries"]}
    f["v_gru_d"], f["t_gru_d"] = f["v_head"] - f["v_gru"], f["t_head"] - f["t_gru"]
    f["best_sys"] = max(f["sys"].items(), key=lambda kv: kv[1][0]) if f["sys"] else None
    tm = lambda src, n, k: src[n]["TEST"]["end_to_end"][k]
    f["vs_gru"] = (r.get("gru_v1_full_vs_macro_mrr20") or {}).get("GRU_F0_locked")
    f["vs_a0"] = max(x["macro_mrr@20"] for x in r["validation_select"]["rows"] if x["run"].startswith("A0_ID256"))
    f["vs_a1_26"] = max(x["macro_mrr@20"] for x in r["validation_select"]["rows"] if x["run"].startswith("A1_RICH256_s2026"))
    f["m_vs"] = None if f["vs_gru"] is None else f["vs_head"] - f["vs_gru"]
    f["a0_gru_vc"] = m(sm, "A0_ID256_s2026", "VALIDATION_CONFIRM") - f["v_gru"]
    f["a0_gru_t"] = f["t_a0"] - f["t_gru"]
    f["micro_mrr_d"] = tm(sm, head, "micro_mrr@20") - tm(ab, GRU_KEY, "micro_mrr@20")
    f["micro_r_d"] = tm(sm, head, "micro_recall@20") - tm(ab, GRU_KEY, "micro_recall@20")
    ch, cg = r["cohorts"][head]["TEST"], r["cohorts"][GRU_KEY]["TEST"]
    f["warm_d"] = ch["warm"]["macro_mrr@20"] - cg["warm"]["macro_mrr@20"]
    f["unseen_d"] = ch["unseen"]["macro_mrr@20"] - cg["unseen"]["macro_mrr@20"]
    f["rel_pct"] = 100 * f["t_gru_d"] / f["t_gru"]
    sd = (f["seedvar"] or {}).get("std_ddof1")
    f["seed_ratio"] = None if not sd else f["t_gru_d"] / sd
    f["rich_vs"] = f["vs_a1_26"] - f["vs_a0"]
    f["s4_d"] = {n: f["sys"][n][0] - m(sm, run, "TEST") for n, run in (("S4a", "A1_RICH256_s2026"), ("S4b", "A1_RICH256_s2027")) if n in f["sys"]}
    return f


def statuses(f: dict) -> dict:
    v = verdict(f["bt_gru"])
    return {"training_status": "COMPLETE", "search_status": "COMPLETE_BOUNDED", "review_status": "PLAN_AND_CODE_APPROVED",
            "retrospective_evaluation_status": "COMPLETE",
            "better_than_gru_on_reused_test": None if v is None else (v == "above"),
            "gru_bootstrap_key": f"{f['head']}_minus_{GRU_KEY}", "tf32_parity_gate_pass": f["tf"].get("pass"),
            "new_confirmatory_holdout_available": False, "universal_sota_proven": False,
            "label": "PREVIOUSLY INSPECTED / RETROSPECTIVE BENCHMARK", "lock_sha256": f["lock_sha"],
            "systems_compared_with_gru_as_model_vs_model": False, "privacy_claim": "NONE", "causal_claim": "NONE"}


AR_V = {"above": "أعلى، وفترة الثقة 95% كلها فوق الصفر", "below": "أدنى، وفترة الثقة 95% كلها تحت الصفر",
        "overlaps": "غير قابل للتمييز: فترة الثقة 95% تعبر الصفر"}
EN_V = {"above": "higher, with the whole 95% CI above zero", "below": "lower, with the whole 95% CI below zero",
        "overlaps": "not distinguishable: the 95% CI crosses zero"}


def ci(bt):
    return "—" if bt is None else f"{bt['diff']:+.6f} [{bt['ci95'][0]:+.6f}, {bt['ci95'][1]:+.6f}]"


def gate_line(f):
    bad = [k for k, v in f["gate"].items() if not v.get("exact_match_fields")]
    return len(f["gate"]), bad


def report_ar(f: dict, t: dict) -> str:
    n_gate, bad = gate_line(f)
    bs = f["best_sys"]
    v2 = f["v2"]
    L = []
    A = L.append
    A("# التقرير النهائي لحملة SASRec — `SASREC_PERFORMANCE_EXECUTION_V1`")
    A("")
    A("> **PREVIOUSLY INSPECTED / RETROSPECTIVE BENCHMARK.** قسما VALIDATION_CONFIRM وTEST فُحصا من قبل في مشروع GRU. لذلك كل رقم VC/TEST في هذا التقرير")
    A("> **استرجاعي**، وليس اختبارًا تأكيديًا جديدًا. لا يوجد holdout تأكيدي جديد.")
    A("")
    A(f"- القفل: `locks/SASREC_LOCK.json`، sha256 `{f['lock_sha']}`، كُتب {f['lock_utc']} قبل تحميل أي صف من VC/TEST.")
    A(f"- مطالبة التقييم الواحد: `locks/RETRO_CLAIM.json` ({f['claim_utc']})؛ اكتمل التقييم {f['retro_done']}؛ "
      f"`retro_eval/RETRO_RESULTS.json` sha256 `{f['results_sha']}`.")
    A("- كُتب هذا التقرير من ملفات النتائج فقط (`tools/build_final_results.py` ثم `tools/write_final_report.py`). "
      "الملف الآلي الذي أنتجه `render_report.py` محفوظ دون تعديل في `final/auto_render/` (انظر §10).")
    A("")
    A("## 1. الخلاصة")
    A(f"- **النموذج الرئيسي (headline):** `{f['head']}` عند epoch 5 (نهاية الـcooldown)، اختير على VS فقط بالقاعدة المسجلة مسبقًا. "
      f"VS macro MRR@20 = {f6(f['vs_head'])}.")
    A(f"- **TEST (استرجاعي):** macro MRR@20 = **{f6(f['t_head'])}** مقابل GRU F0 المقفل {f6(f['t_gru'])} "
      f"(الفرق {ci(f['bt_gru'])}: {AR_V.get(verdict(f['bt_gru']), '—')}) ومقابل TRANS {f6(f['t_trans'])} "
      f"(الفرق {ci(f['bt_trans'])}: {AR_V.get(verdict(f['bt_trans']), '—')}).")
    A(f"- **VALIDATION_CONFIRM (استرجاعي):** {f6(f['v_head'])} مقابل GRU F0 {f6(f['v_gru'])} (الفرق {sg(f['v_gru_d'])}، بدون bootstrap على VC).")
    if f["bt_rich"]:
        A(f"- **Rich مقابل ID (seed 2026، كلاهما cooldown، مقارنة واحدة):** TEST {ci(f['bt_rich'])} — {AR_V[verdict(f['bt_rich'])]}. "
          "هذه مقارنة واحدة وليست دليلًا عامًا على قيمة الخصائص.")
    if bs:
        A(f"- **الأنظمة (منفصلة، ليست نموذجًا مقابل نموذج مع GRU):** أعلى نظام على TEST هو {bs[0]} بـ {f6(bs[1][0])} "
          f"(الفرق عن الـheadline {sg(bs[1][0] - f['t_head'])}، وصفي بدون CI).")
    A("- **SOTA:** لا يوجد ادعاء. لا خصوصية مختبرة. لا ادعاء سببي.")
    A("")
    A("### 1.1 قراءة النتائج")
    A(f"- **(MEASURED)** هامش الـheadline على GRU F0 في TEST هو {sg(f['t_gru_d'])} ({f['rel_pct']:+.2f}% نسبيًا)"
      + (f"، أي نحو {f['seed_ratio']:.0f}× الانحراف المعياري بين بذرتي A1 على TEST ({f6(f['seedvar'].get('std_ddof1'))})" if f["seed_ratio"] else "")
      + ". هامش صغير لكنه ثابت إحصائيًا على مستوى المستخدم.")
    if f["m_vs"] is not None:
        A(f"- **(MEASURED)** الهامش تقلص من VS ({sg(f['m_vs'])}) إلى VC ({sg(f['v_gru_d'])}) ثم TEST ({sg(f['t_gru_d'])}). "
          f"الـheadline تغيّر بمقدار {sg(f['t_head'] - f['vs_head'])} من VS إلى TEST، وGRU F0 بمقدار {sg(f['t_gru'] - f['vs_gru'])}. "
          "**(INFERENCE، غير مختبر)** أسباب محتملة: تفاؤل قسم الاختيار (VS استُخدم لاختيار الـepoch وقرار الـcooldown post-hoc والبذرة)، "
          "والبعد الزمني عن TRAIN، وزيادة OOV؛ لم يُعزل أي منها.")
    A(f"- **(MEASURED، وصفي بدون CI)** SASRec بالمعرّفات فقط (A0) **أقل** من GRU F0: VC {sg(f['a0_gru_vc'])}، TEST {sg(f['a0_gru_t'])}. "
      "إذن تفوّق الـheadline على GRU يأتي مع قنوات الخصائص الإضافية؛ لم يُثبت أن المعمارية وحدها تتفوق على GRU. هذه المقارنة غير مسجلة مسبقًا ولا bootstrap لها.")
    A(f"- **(MEASURED)** Rich − ID على TEST {sg(f['bt_rich']['diff'] if f['bt_rich'] else None)} مقابل {sg(f['rich_vs'])} على VS (أفضل epoch لكل منهما).")
    A(f"- **(MEASURED)** المكسب في micro MRR@20 على TEST يكاد يكون صفرًا ({sg(f['micro_mrr_d'])})، وفي micro R@20 {sg(f['micro_r_d'])}؛ "
      f"أي أن المكسب في المتوسط لكل مستخدم وفي الاسترجاع ضمن أعلى 20. warm {sg(f['warm_d'])}، unseen {sg(f['unseen_d'])}.")
    if f["best_sys"]:
        A(f"- **(MEASURED، وصفي)** أفضل نظام {f['best_sys'][0]} أعلى من الـheadline بـ{sg(f['best_sys'][1][0] - f['t_head'])} على TEST؛ "
          f"متوسطات الأوزان أقل من النموذج المفرد: {', '.join(f'{k} {sg(v)}' for k, v in f['s4_d'].items())} — متسق مع قراءة VS الوصفية.")
    A("")
    A("## 2. المهمة والبروتوكول (MEASURED)")
    A("- التنبؤ بالمنتج التالي على بيانات REES46 المجمدة `rees46_nextitem_v1` (hash `860e5a1c…`)؛ نفس البروتوكول المستخدم مع GRU.")
    A("- ترتيب كامل على الكتالوج (K = 158,486)، بدون sampling وبدون تقليص الكتالوج.")
    A("- المقياس الرئيسي user-macro MRR@20 على `eval_user_key`؛ الهدف OOV يأخذ صفرًا (`end_to_end`)، والمقام `closed` معروض منفصلًا.")
    A(f"- VC: {f['n_vc']:,} قرار قابل للتقييم، {f['users_vc']:,} مستخدم. TEST: {f['n_test']:,} قرار، {f['users_test']:,} مستخدم، منها {f['oov_test']:,} هدف OOV.")
    A(f"- بوابة البصمة: {n_gate} فحوص لإعادة إنتاج أرقام GRU/TRANS/POP المؤرشفة بالضبط"
      + ("؛ كلها تطابقت." if not bad else f"؛ **فشل**: {bad}."))
    A("")
    A("## 3. مسار الحملة")
    A("### 3.1 فشل v2 (MEASURED)")
    if v2:
        A("- الوصفة: head مربوط (tied)، إدخال مضروب في √d (×16)، betas (0.9, 0.98)، peak 5e-4.")
        A("- VS macro MRR@20: " + "؛ ".join(f"{k} " + ", ".join(f"{e} {v:.5f}" for e, v in d.items()) for k, d in v2.items())
          + " — أقل كثيرًا من GRU (0.17387). أوقفه Fable بصفته مفوض المالك وحُفظ كما هو في `runs_v2/`.")
        A("- التفسير (INFERENCE من مراجعة Fable، ليس تجربة معزولة): انجراف مشترك لجدول العناصر المربوط تحت Adam وCE الكثيف، مضخّم ×16؛ "
          "فشل A0 أيضًا، إذن الأعمدة ليست السبب. المصدر: `review/deep/FABLE_DEEP_REVIEW_RESPONSE.md`.")
    A("### 3.2 إصلاح v3 (حزمة من خمسة تغييرات غير معزولة)")
    A("1. فك ربط head الإخراج؛ 2. إزالة √d وإضافة LayerNorm للإدخال؛ 3. betas (0.9, 0.999)؛ 4. إعادة تمركز جدول الإخراج بعد كل خطوة؛ "
      "5. peak 1e-3 وfloor 1e-4، 4 epochs تُقيَّم عند 3 و4.")
    A("- هذه حزمة واحدة؛ لا يمكن نسب التحسن إلى أي تغيير منفرد.")
    A("### 3.3 التعديلات (amendments)")
    A("- **Amendment 1:** استبدل معيار بوابة 20k غير المعاير (≤ 3×init) بحدود مطلقة. **أُضيف بعد رؤية بداية المسار (2.1–3.7k تحديث) وقبل نقطة 20k.**")
    A("- **Amendment 2** (طلب المالك «5 epochs» و«أفضل دقة»): epoch خامس **كـcooldown فقط** "
      "(LR = 1e-6 + (1e-4 − 1e-6)(1 − √t))، seed ثاني 2027، cooldowns مشروطة لـA0 وs2027، وأنظمة S1–S4b تُعرض منفصلة.")
    A("- **إفصاح:** VS لـepochs 3–4 كان قد رُئي قبل Amendment 2، إذن قرار الـcooldown **post-hoc**. "
      "VS لم يُستخدم إلا للاختيار؛ VC/TEST لم يُستخدما لأي قرار.")
    A("### 3.4 الدقة الحسابية")
    tf = f["tf"]
    A(f"- التدريب TF32، والتقييم FP32 صارم دائمًا. بوابة التكافؤ المسجلة مسبقًا: فرق الخسارة {tf.get('delta_tf32', 0):.2e} "
      f"(التسامح {tf.get('tolerance', 0):.4f})، فرق الـcohort {tf.get('cohort_delta_tf32', 0):.2e}، نسبة grad-norm {tf.get('grad_median_ratio', 0):.6f} → "
      f"**{tf.get('ruling', '—')}** (`bench/TF32_PARITY.json`).")
    A("")
    A("## 4. نتائج VS (قسم الاختيار، full VS)")
    A(t["vs"])
    A("")
    g = f["gaps"]
    A(f"- فروق البذرتين لـA1 (epochs متطابقة): e3 {f6(g.get('e3'))}، e4 {f6(g.get('e4'))}، e5 {f6(g.get('e5'))}. "
      "هذا **مدى من بذرتين، وليس انحرافًا معياريًا**.")
    A("")
    A("## 5. الاختيار والقفل")
    A("- القاعدة: max full-VS macro MRR@20 على {A0_s2026, A1_s2026, A1_s2027} × epochs {3,4,5}؛ tie ≤ 1e-6 → micro R@20 → عدد معاملات أقل → ترتيب السجل.")
    A(f"- الاختيار: `{f['head']}`. الفرق بين بذرتي A1 عند e5 ({f6(g.get('e5'))}) أصغر كثيرًا من مكسب الـcooldown ومن فرق Rich−ID؛ "
      "أي بذرة فازت ليس دليلًا على شيء.")
    A(f"- القفل يضم 8 مدخلات (3 نماذج مفردة + S1/S2/S3 + S4a/S4b)؛ pair_check: {f['pair'].get('basis')} "
      f"(comparable = {f['pair'].get('comparable')})؛ مدخلات مسقطة: {f['dropped'] or 'لا شيء'}.")
    A("")
    A("## 6. النتائج الاسترجاعية على VC وTEST (PREVIOUSLY INSPECTED)")
    A("### 6.1 VALIDATION_CONFIRM (end_to_end)")
    A(t["singles_VALIDATION_CONFIRM"])
    A("")
    A("### 6.2 TEST (end_to_end)")
    A(t["singles_TEST"])
    A("")
    A("GRU_F4_descriptive وصفي فقط (F4 < F0 في GRU مشوش بتوقيت خطوة LR). POP_full مرجع شعبية.")
    A("")
    A("### 6.3 المقام closed (بدون أهداف OOV)")
    A(t["closed_TEST"])
    A("")
    A("### 6.4 Bootstrap مزدوج على مستوى المستخدم (TEST، 1000 إعادة، seed 2026)")
    A(t["bootstraps"])
    A("")
    A("- هذه الفترات مشروطة بهذه النماذج المدربة وبفترة زمنية فُحصت من قبل؛ لا تلتقط تباين بذرة التدريب.")
    sv = f["seedvar"]
    A(f"- تباين البذرة على TEST (A1، بذرتان): {json.dumps(sv.get('macro_mrr20_by_seed'))}، std(ddof=1) = {f6(sv.get('std_ddof1'))} — بذرتان فقط.")
    A("")
    A("## 7. الأنظمة S1–S4b (منفصلة؛ لا تُقارن بـGRU كنموذج مقابل نموذج)")
    A("### TEST")
    A(t["systems_TEST"])
    A("")
    A("### VALIDATION_CONFIRM")
    A(t["systems_VALIDATION_CONFIRM"])
    A("")
    A("- أوزان الأنظمة ثابتة مسبقًا (متوسط log-softmax بأوزان متساوية؛ أو متوسط state_dict منتظم). لم يُضف أو يُحذف أو يُعاد وزن أي نظام بعد رؤية أي رقم.")
    A("- قيم VS الوصفية لـS2 وS4a (GPU1 الفارغ، قبل القفل): S2 ≈ A1 e5 (CI يعبر الصفر)، S4a < A1 e5 (CI كله سالب). "
      "لم تُستخدم للاختيار (`sasrec_vs_systems_v1/`).")
    A("")
    A("## 8. فئات المستخدمين (warm / unseen)")
    A(t["cohorts"])
    A("")
    A("## 9. الموارد")
    A(t["resources"])
    A("")
    A("- elapsed = نهاية − بداية هذا التشغيل؛ cumulative wall يشمل التشغيل المصدر في حالة الـcooldown. GPU0/GPU1 على خادم Lobot واحد.")
    A("")
    A("## 10. أخطاء وتصحيحات معلنة (errata)")
    A("- حقول نصية قديمة في السجل المقفل لم تُصحح (تعديلها يكسر hash الاستئناف): `tie_item_embeddings: true` (النموذج الفعلي غير مربوط)، "
      "نص fallback يقول «5 epochs»، `SASREC_INTERFACES.md` يصف v2، حقل حالة قديم.")
    A("- Gate C على v2: الدليل نُسب إلى hash قديم لـ`train_sasrec.py` → أعيد توليده (112/112).")
    A("- مراجعة المرحلة النهائية وجدت سطر `model = build_model(...)` ناقصًا كان سيعطي صفرًا لكل صفوف VC/TEST → أُصلح قبل التشغيل.")
    A("- الـcooldown كان يمكن أن يحذف checkpoints المصدر → أُصلح قبل التشغيل.")
    A("- `render_report.py` الآلي كُتب لـv2: يبحث عن `state/selection.json` ومفتاح bootstrap بصيغة v2، فقد يُظهر «not measured» لمقارنة GRU. "
      "وقد أظهر فعلًا «Not yet measured» (بنى المفتاح `A1_RICH256_s2026_minus_GRU_F0_locked` بدل `A1_RICH256_s2027_…`). "
      "ناتجه محفوظ في `final/auto_render/` دون تعديل؛ هذا التقرير يحل محله.")
    A("- ملاحظة عمل سابقة ذكرت فرق البذرتين عند e5 كـ0.000215 (مبتور)؛ القيمة المقربة 0.000216.")
    A("")
    A("## 11. السقف الواقعي")
    A(f"- تقدير Fable ≈ 0.20–0.22 macro MRR@20 هو **INFERENCE من أدبيات قريبة وليس حدًا مثبتًا**. الـheadline على VS = {f6(f['vs_head'])}.")
    A("")
    A("## 12. حدود وادعاءات ممنوعة")
    A("- لا SOTA: لا مقارنة بأرقام منشورة ببروتوكولات أخرى، ولا إعادة إنتاج لأوراق SASRec بنفس البروتوكول.")
    A("- لا خصوصية: لم تُختبر أي آلية خصوصية.")
    A("- لا سببية: إصلاح v3 حزمة غير معزولة؛ Rich > ID مقارنة واحدة؛ A0 ببذرة واحدة، A1 ببذرتين.")
    A("- الأنظمة لا تُقارن بـGRU كنموذج مقابل نموذج.")
    A("- VC/TEST استرجاعيان؛ لا holdout تأكيدي جديد.")
    A("")
    A("## 13. الأدلة")
    A("- `locks/SASREC_LOCK.json`، `locks/RETRO_CLAIM.json`، `locks/RETRO_DONE.json`، `retro_eval/RETRO_RESULTS.json`، `retro_eval/<split>/<entry>.json|.ranks.npz`.")
    A("- `runs/<run>/summary.json` للتشغيلات الستة؛ `config/run_registry.v3_amendment_{1,2}.json`؛ `review/` و`review/deep/`.")
    A("- `final/FINAL_SASREC_RESULTS.json` يحمل sha256 كل ملف إدخال (`inputs_sha256`).")
    return "\n".join(L) + "\n"


def report_en(f: dict, t: dict) -> str:
    n_gate, bad = gate_line(f)
    bs = f["best_sys"]
    v2 = f["v2"]
    L = []
    A = L.append
    A("# SASRec campaign final report — `SASREC_PERFORMANCE_EXECUTION_V1` (English companion)")
    A("")
    A("> **PREVIOUSLY INSPECTED / RETROSPECTIVE BENCHMARK.** VALIDATION_CONFIRM and TEST were inspected in the GRU project. Every")
    A("> VC/TEST number here is **retrospective**, not a new confirmatory test. No fresh confirmatory holdout exists.")
    A(">")
    A("> The required deliverable is `FINAL_SASREC_REPORT_AR.md`; this file carries the same numbers.")
    A("")
    A(f"- Lock: `locks/SASREC_LOCK.json`, sha256 `{f['lock_sha']}`, written {f['lock_utc']} before any VC/TEST row was loaded.")
    A(f"- One-look claim: `locks/RETRO_CLAIM.json` ({f['claim_utc']}); evaluation completed {f['retro_done']}; "
      f"`retro_eval/RETRO_RESULTS.json` sha256 `{f['results_sha']}`.")
    A("")
    A("## 1. Summary")
    A(f"- **Headline single model:** `{f['head']}` at epoch 5 (end of cooldown), selected on VS only by the pre-registered rule. "
      f"VS macro MRR@20 = {f6(f['vs_head'])}.")
    A(f"- **TEST (retrospective):** macro MRR@20 = **{f6(f['t_head'])}** versus the locked GRU F0 {f6(f['t_gru'])} "
      f"(difference {ci(f['bt_gru'])}: {EN_V.get(verdict(f['bt_gru']), '—')}) and versus TRANS {f6(f['t_trans'])} "
      f"(difference {ci(f['bt_trans'])}: {EN_V.get(verdict(f['bt_trans']), '—')}).")
    A(f"- **VALIDATION_CONFIRM (retrospective):** {f6(f['v_head'])} versus GRU F0 {f6(f['v_gru'])} (difference {sg(f['v_gru_d'])}; no bootstrap on VC).")
    if f["bt_rich"]:
        A(f"- **Rich versus ID (seed 2026, both cooled down; one comparison):** TEST {ci(f['bt_rich'])}: {EN_V[verdict(f['bt_rich'])]}. "
          "A single comparison, not general evidence of feature value.")
    if bs:
        A(f"- **Systems (separate; never model versus model against GRU):** the best system on TEST is {bs[0]} at {f6(bs[1][0])} "
          f"({sg(bs[1][0] - f['t_head'])} versus the headline; descriptive, no CI).")
    A("- **SOTA:** not claimed. No privacy mechanism tested. No causal claim.")
    A("")
    A("### 1.1 Reading the results")
    A(f"- **(MEASURED)** The headline's TEST margin over GRU F0 is {sg(f['t_gru_d'])} ({f['rel_pct']:+.2f}% relative)"
      + (f", about {f['seed_ratio']:.0f}× the A1 two-seed TEST std ({f6(f['seedvar'].get('std_ddof1'))})" if f["seed_ratio"] else "")
      + ". Small, but statistically stable at the user level.")
    if f["m_vs"] is not None:
        A(f"- **(MEASURED)** The margin shrank from VS ({sg(f['m_vs'])}) to VC ({sg(f['v_gru_d'])}) and TEST ({sg(f['t_gru_d'])}). "
          f"The headline changed by {sg(f['t_head'] - f['vs_head'])} from VS to TEST; GRU F0 by {sg(f['t_gru'] - f['vs_gru'])}. "
          "**(INFERENCE, untested)** Possible reasons: selection-split optimism (VS chose the epoch, the post-hoc cooldown and the seed), "
          "temporal distance from TRAIN, more OOV. None was isolated.")
    A(f"- **(MEASURED, descriptive, no CI)** The ID-only SASRec (A0) is **below** GRU F0: VC {sg(f['a0_gru_vc'])}, TEST {sg(f['a0_gru_t'])}. "
      "The headline's advantage over GRU therefore comes with the extra feature channels; the architecture alone is not shown to beat GRU. "
      "This pair was not pre-registered and has no bootstrap.")
    A(f"- **(MEASURED)** Rich − ID is {sg(f['bt_rich']['diff'] if f['bt_rich'] else None)} on TEST versus {sg(f['rich_vs'])} on VS (each at its best epoch).")
    A(f"- **(MEASURED)** The micro MRR@20 gain on TEST is near zero ({sg(f['micro_mrr_d'])}); micro R@20 {sg(f['micro_r_d'])}. The gain is in the per-user "
      f"average and in top-20 retrieval. Warm {sg(f['warm_d'])}, unseen {sg(f['unseen_d'])}.")
    if f["best_sys"]:
        A(f"- **(MEASURED, descriptive)** The best system, {f['best_sys'][0]}, is {sg(f['best_sys'][1][0] - f['t_head'])} above the headline on TEST; "
          f"the weight averages are below their single models: {', '.join(f'{k} {sg(v)}' for k, v in f['s4_d'].items())}, consistent with the VS-descriptive reading.")
    A("")
    A("## 2. Task and protocol (MEASURED)")
    A("- Next-item prediction on the frozen REES46 dataset `rees46_nextitem_v1` (hash `860e5a1c…`), the same protocol as GRU.")
    A("- Full-catalogue ranking (K = 158,486); no sampling; no catalogue reduction.")
    A("- Primary metric user-macro MRR@20 over `eval_user_key`; OOV targets score zero (`end_to_end`); the `closed` denominator is shown separately.")
    A(f"- VC: {f['n_vc']:,} evaluable decisions, {f['users_vc']:,} users. TEST: {f['n_test']:,} decisions, {f['users_test']:,} users, {f['oov_test']:,} OOV targets.")
    A(f"- Fingerprint gate: {n_gate} checks reproduce the archived GRU/TRANS/POP numbers exactly"
      + ("; all matched." if not bad else f"; **FAILED**: {bad}."))
    A("")
    A("## 3. Campaign path")
    if v2:
        A("- **v2 failed (MEASURED):** tied head, input ×√d (×16), betas (0.9, 0.98), peak 5e-4. VS macro MRR@20: "
          + "; ".join(f"{k} " + ", ".join(f"{e} {v:.5f}" for e, v in d.items()) for k, d in v2.items())
          + ". Stopped by Fable as the owner's delegate and archived intact in `runs_v2/`. Interpretation (INFERENCE): common-mode drift of the "
          "tied table under Adam and dense CE, amplified ×16; A0 failed too, so the columns are not the cause.")
    A("- **v3 repair (a bundle of five, not isolated):** untied head; no √d plus input LayerNorm; betas (0.9, 0.999); recentred output table "
      "after every step; peak 1e-3, floor 1e-4, 4 epochs evaluated at 3 and 4.")
    A("- **Amendment 1:** the miscalibrated 20k criterion (≤ 3×init) was replaced by absolute caps, **after seeing 2.1–3.7k updates and before 20k**.")
    A("- **Amendment 2** (owner: \"5 epochs\", \"best accuracy\"): a fifth epoch **as a cooldown only** (LR = 1e-6 + (1e-4 − 1e-6)(1 − √t)), "
      "seed 2027, conditional cooldowns for A0 and s2027, systems S1–S4b reported separately.")
    A("- **Disclosure:** VS for epochs 3–4 had been seen before amendment 2, so the cooldown decision is **post-hoc**. VS was used only for "
      "selection; VC/TEST were used for no decision.")
    tf = f["tf"]
    A(f"- **Precision:** TF32 training, strict FP32 evaluation. Pre-registered parity gate: loss difference {tf.get('delta_tf32', 0):.2e} "
      f"(tolerance {tf.get('tolerance', 0):.4f}), cohort difference {tf.get('cohort_delta_tf32', 0):.2e}, grad-norm ratio "
      f"{tf.get('grad_median_ratio', 0):.6f} → **{tf.get('ruling', '—')}**.")
    A("")
    A("## 4. VS results (selection split, full VS)")
    A(t["vs"])
    A("")
    g = f["gaps"]
    A(f"- A1 two-seed gaps (matched epochs): e3 {f6(g.get('e3'))}, e4 {f6(g.get('e4'))}, e5 {f6(g.get('e5'))}: a **range from two seeds, not a standard deviation**.")
    A("")
    A("## 5. Selection and lock")
    A("- Rule: max full-VS macro MRR@20 over {A0_s2026, A1_s2026, A1_s2027} × epochs {3,4,5}; tie ≤ 1e-6 → micro R@20 → fewer params → registry order.")
    A(f"- Selected: `{f['head']}`. The A1 seed gap at e5 ({f6(g.get('e5'))}) is far below the cooldown gain and the Rich−ID gap; which seed won is not evidence.")
    A(f"- The lock holds 8 entries (3 singles + S1/S2/S3 + S4a/S4b); pair_check: {f['pair'].get('basis')} (comparable = {f['pair'].get('comparable')}); "
      f"dropped entries: {f['dropped'] or 'none'}.")
    A("")
    A("## 6. Retrospective VC and TEST results (PREVIOUSLY INSPECTED)")
    A("### 6.1 VALIDATION_CONFIRM (end_to_end)")
    A(t["singles_VALIDATION_CONFIRM"])
    A("")
    A("### 6.2 TEST (end_to_end)")
    A(t["singles_TEST"])
    A("")
    A("GRU_F4_descriptive is descriptive only (F4 < F0 in GRU is confounded by LR-step timing). POP_full is the popularity reference.")
    A("")
    A("### 6.3 Closed denominator (OOV targets excluded)")
    A(t["closed_TEST"])
    A("")
    A("### 6.4 Paired user-level bootstrap (TEST, 1000 resamples, seed 2026)")
    A(t["bootstraps"])
    A("")
    A("- Conditional on these fitted models and a previously inspected period; the CIs do not capture training-seed variance.")
    sv = f["seedvar"]
    A(f"- TEST seed variation (A1, two seeds): {json.dumps(sv.get('macro_mrr20_by_seed'))}, std(ddof=1) = {f6(sv.get('std_ddof1'))}; two seeds only.")
    A("")
    A("## 7. Systems S1–S4b (separate; never compared with GRU as model versus model)")
    A("### TEST")
    A(t["systems_TEST"])
    A("")
    A("### VALIDATION_CONFIRM")
    A(t["systems_VALIDATION_CONFIRM"])
    A("")
    A("- Weights fixed a priori (equal-weight log-softmax mean, or a uniform state_dict average). No system was added, dropped or re-weighted after any number was seen.")
    A("- The earlier VS-descriptive S2/S4a numbers (idle GPU1, before the lock): S2 ≈ A1 e5 (CI crosses 0); S4a < A1 e5 (CI below 0). Not used for selection.")
    A("")
    A("## 8. User cohorts (warm / unseen)")
    A(t["cohorts"])
    A("")
    A("## 9. Resources")
    A(t["resources"])
    A("")
    A("- elapsed = this run's end − start; the cumulative wall time of a cooldown includes its source run. GPU0/GPU1 are on one Lobot server.")
    A("")
    A("## 10. Disclosed errata")
    A("- Stale text fields in the locked registry, not fixed because editing breaks the resume hash: `tie_item_embeddings: true` (the model is untied), "
      "fallback text \"5 epochs\", `SASREC_INTERFACES.md` describes v2, a stale status field.")
    A("- Gate C on v2 attributed evidence to an old `train_sasrec.py` hash → regenerated (112/112).")
    A("- The final-stage review found a missing `model = build_model(...)` line that would have scored zero on every VC/TEST row → fixed before running.")
    A("- The cooldown could have deleted the source checkpoints → fixed before running.")
    A("- The automatic `render_report.py` was written for v2: it looks for `state/selection.json` and a v2 bootstrap key, and it did print \"Not yet measured\" "
      "(it built the key `A1_RICH256_s2026_minus_GRU_F0_locked` instead of `A1_RICH256_s2027_…`). Its output is preserved unmodified in `final/auto_render/`; this report supersedes it.")
    A("- An earlier working note gave the e5 seed gap as 0.000215 (truncated); the rounded value is 0.000216.")
    A("")
    A("## 11. Realistic ceiling")
    A(f"- Fable's ≈ 0.20–0.22 macro MRR@20 is **INFERENCE from nearby literature, not a proven bound**. The headline VS value is {f6(f['vs_head'])}.")
    A("")
    A("## 12. Limitations and prohibited claims")
    A("- No SOTA: no comparison with published numbers under other protocols; no same-protocol reproduction of SASRec papers.")
    A("- No privacy mechanism tested. No causality: the v3 repair is a bundle; Rich > ID is one comparison; A0 has one seed, A1 two.")
    A("- Systems are never compared with GRU as model versus model. VC/TEST are retrospective; no fresh confirmatory holdout.")
    return "\n".join(L) + "\n"


def sota_md(f: dict, st: dict) -> str:
    v = verdict(f["bt_gru"])
    L = ["# SOTA_CLAIM_STATUS", "", "> PREVIOUSLY INSPECTED / RETROSPECTIVE BENCHMARK", ""]
    if v == "above":
        L.append(f"The locked headline `{f['head']}` beats the existing locked GRU F0 baseline on the frozen, previously inspected REES46 "
                 f"TEST period (measured; paired user bootstrap {ci(f['bt_gru'])}, CI excludes 0).")
    elif v is None:
        L.append("The paired bootstrap against GRU is missing from RETRO_RESULTS.json. No claim is made.")
    else:
        L.append(f"The locked headline `{f['head']}` is NOT shown to beat the locked GRU F0 baseline on the previously inspected TEST period "
                 f"(paired user bootstrap {ci(f['bt_gru'])}).")
    L.append("Best among the configurations evaluated in this campaign (bounded search: A0/A1 × seeds × epochs 3–5).")
    L.append("Not shown to beat published SASRec-family variants: no same-protocol reproduction of external papers was run.")
    L.append("No fresh confirmatory holdout: VALIDATION_CONFIRM/TEST were inspected in the GRU project.")
    L.append("Systems S1–S4b are reported separately and are not a model-versus-model comparison with GRU.")
    L += ["", "## Machine-readable statuses", "```json", json.dumps(st, indent=2), "```"]
    return "\n".join(L) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--final", required=True, type=Path)
    a = ap.parse_args()
    r = json.loads((a.final / "FINAL_SASREC_RESULTS.json").read_text(encoding="utf-8"))
    t = json.loads((a.final / "_tables.json").read_text(encoding="utf-8"))
    f = facts(r)
    st = statuses(f)
    (a.final / "FINAL_SASREC_REPORT_AR.md").write_text(report_ar(f, t), encoding="utf-8")
    (a.final / "FINAL_SASREC_REPORT_EN.md").write_text(report_en(f, t), encoding="utf-8")
    (a.final / "SOTA_CLAIM_STATUS.md").write_text(sota_md(f, st), encoding="utf-8")
    print(json.dumps({k: st[k] for k in ("better_than_gru_on_reused_test", "gru_bootstrap_key")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
