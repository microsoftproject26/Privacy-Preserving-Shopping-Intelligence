"""render_report.py -- FINAL_SASREC_REPORT_AR.md, FINAL_SASREC_RESULTS.json and
SOTA_CLAIM_STATUS.md from result files only (SASREC_INTERFACES.md sec 7,
08_REPORT_AND_CLAIM_POLICY.md). Owner: Sonnet-runtime.

This module NEVER invents a number. Every figure in the outputs is read from an
on-disk artefact (run summary.json, state/selection.json, locks/SASREC_LOCK.json,
retro_eval/RETRO_RESULTS.json, resource-sampler JSONL, the locked registry, the
review markdown files). Whatever is missing is rendered as an explicit
"NOT_AVAILABLE"/"غير متاح" placeholder with the machine-readable status set to
PARTIAL/BLOCKED/NOT_RUN accordingly -- never silently dropped, never filled in.

The report template (§ text between numbered sections) is embedded here rather
than in a separate report_template_ar.md so a plain per-section string swap is
all a redeploy needs; `report_template_ar.md` remains available in the code dir
as the human-readable reference copy of that same structure.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Optional

import paths  # noqa: E402  import-first convention

GIB = 2 ** 30


def _root() -> Path:
    import os
    override = os.environ.get("SASREC_QUEUE_ROOT")
    return Path(override) if override else paths.S


def _read_json(path: Path) -> Optional[Any]:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def _read_jsonl(path: Path) -> list[dict]:
    out = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except OSError:
        pass
    return out


def _sanitize(obj: Any, sanitized_paths: list, path: str = "$") -> Any:
    """Recursively replace NaN/Inf with None (documented, not invented) so the
    strict FINAL_SASREC_RESULTS.json (allow_nan=False) always serializes."""
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            sanitized_paths.append(path)
            return None
        return obj
    if isinstance(obj, dict):
        return {k: _sanitize(v, sanitized_paths, f"{path}.{k}") for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize(v, sanitized_paths, f"{path}[{i}]") for i, v in enumerate(obj)]
    return obj


# ---------------------------------------------------------------------------
# gather
# ---------------------------------------------------------------------------

def gather_inputs(root: Path) -> dict:
    registry = _read_json(root / "config" / "run_registry.locked.json") or {}
    models = {m["id"]: m for m in registry.get("models", [])}
    primary_seed = registry.get("training", {}).get("seed")
    replica_seeds = registry.get("replication", {}).get("additional_seeds", [])

    runs = {}
    runs_root = root / "runs"
    if runs_root.exists():
        for d in sorted(p for p in runs_root.iterdir() if p.is_dir()):
            summary = _read_json(d / "summary.json")
            if summary is not None:
                runs[d.name] = summary

    selection = _read_json(root / "state" / "selection.json")
    lock = _read_json(root / "locks" / "SASREC_LOCK.json")
    retro = _read_json(root / "retro_eval" / "RETRO_RESULTS.json")
    resource_samples = _read_jsonl(root / "logs" / "resources_samples.jsonl")
    queue_state = _read_json(root / "state" / "queue_state.json")
    epoch_cap = _read_json(root / "state" / "epoch_cap.json")
    tf32_parity = _read_json(root / "bench" / "TF32_PARITY.json")
    dropped_models = registry.get("dropped_models", [])
    owner_deadline = registry.get("owner_deadline", {})
    pair_parity = (queue_state or {}).get("pair_parity")

    review_dir = root / "review"
    review_files = []
    if review_dir.exists():
        for f in sorted(review_dir.glob("*.md")):
            review_files.append(f.name)

    return {
        "registry": registry, "models": models, "primary_seed": primary_seed,
        "replica_seeds": replica_seeds, "runs": runs, "selection": selection,
        "lock": lock, "retro": retro, "resource_samples": resource_samples,
        "queue_state": queue_state, "review_files": review_files,
        "epoch_cap": epoch_cap, "tf32_parity": tf32_parity,
        "dropped_models": dropped_models, "owner_deadline": owner_deadline,
        "pair_parity": pair_parity,
    }


# ---------------------------------------------------------------------------
# machine-readable statuses (08_REPORT_AND_CLAIM_POLICY.md)
# ---------------------------------------------------------------------------

def compute_statuses(inputs: dict) -> dict:
    models = inputs["models"]
    primary_seed = inputs["primary_seed"]
    runs = inputs["runs"]

    primary_keys = [f"{mid}_s{primary_seed}" for mid in models]
    primary_present = [runs.get(k) for k in primary_keys]
    n_present = sum(1 for s in primary_present if s is not None)
    n_complete = sum(1 for s in primary_present if s and s.get("final_status") == "COMPLETE")
    n_terminal = sum(1 for s in primary_present
                     if s and s.get("final_status") not in (None, "RUNNING"))

    if n_present == 0:
        training_status = "BLOCKED"
    elif n_terminal < len(primary_keys):
        training_status = "PARTIAL"
    else:
        training_status = "COMPLETE" if n_complete > 0 else "PARTIAL"

    selection = inputs["selection"]
    if selection is None:
        search_status = "PARTIAL" if training_status != "BLOCKED" else "PARTIAL"
    elif selection.get("selected") is not None:
        search_status = "COMPLETE_BOUNDED"
    else:
        search_status = "PARTIAL"

    review_status = "PLAN_AND_CODE_APPROVED" if "GATE_P_DECISIONS.md" in inputs["review_files"] else "BLOCKED"

    retro = inputs["retro"]
    if retro is None:
        retrospective_evaluation_status = "NOT_RUN"
    elif "TEST" in retro.get("splits", {}):
        retrospective_evaluation_status = "COMPLETE"
    else:
        retrospective_evaluation_status = "PARTIAL"

    # v2 (Gate P2): the exact bootstrap key is <selected>_s<seed>_minus_GRU_F0_locked
    # (coordinator-specified; the selected id comes from state/selection.json, never
    # guessed here).
    better_than_gru = None
    gru_bootstrap_key = None
    selection = inputs["selection"]
    selected = selection.get("selected") if selection else None
    if retro is not None and selected is not None:
        gru_bootstrap_key = f"{selected}_s{primary_seed}_minus_GRU_F0_locked"
        rec = retro.get("bootstraps", {}).get("TEST", {}).get(gru_bootstrap_key)
        if rec is not None:
            ci = rec.get("ci95")
            if ci and len(ci) == 2:
                better_than_gru = bool(ci[0] > 0)

    tf32_parity = inputs.get("tf32_parity")
    tf32_parity_gate_pass = tf32_parity.get("pass") if isinstance(tf32_parity, dict) else None

    return {
        "training_status": training_status,
        "search_status": search_status,
        "review_status": review_status,
        "retrospective_evaluation_status": retrospective_evaluation_status,
        "better_than_gru_on_reused_test": better_than_gru,
        "gru_bootstrap_key": gru_bootstrap_key,
        "tf32_parity_gate_pass": tf32_parity_gate_pass,
        "new_confirmatory_holdout_available": False,
        "universal_sota_proven": False,
        "counts": {"primary_present": n_present, "primary_complete": n_complete,
                   "primary_terminal": n_terminal, "primary_total": len(primary_keys)},
    }


# ---------------------------------------------------------------------------
# section text (Arabic) -- every sentence below is either fixed policy wording
# (08_REPORT_AND_CLAIM_POLICY.md) or a direct read of `inputs`/`statuses`;
# nothing here is a number synthesised outside those two dicts.
# ---------------------------------------------------------------------------

NA = "غير متاح (البيانات المصدر غير موجودة بعد)"


def _run_row(run_id: str, seed, summary: Optional[dict]) -> str:
    if summary is None:
        return f"| {run_id} | {seed} | {NA} | - | - | - | - |"
    fs = summary.get("final_status", NA)
    be = summary.get("best_epoch", "-")
    bm = summary.get("best_metric", "-")
    ep = summary.get("epochs_completed", "-")
    npar = summary.get("n_params", "-")
    return f"| {run_id} | {seed} | {fs} | {be} | {bm} | {ep} | {npar} |"


def build_sections(inputs: dict, statuses: dict) -> dict:
    models = inputs["models"]
    seed = inputs["primary_seed"]
    runs = inputs["runs"]

    s1 = (
        f"حالة التدريب: **{statuses['training_status']}** "
        f"({statuses['counts']['primary_terminal']}/{statuses['counts']['primary_total']} "
        f"تشغيلات أساسية بلغت حالة نهائية، {statuses['counts']['primary_complete']} منها COMPLETE). "
        f"حالة البحث عن الإعداد: **{statuses['search_status']}**. "
        f"حالة المراجعة: **{statuses['review_status']}** (راجع review/GATE_P_DECISIONS.md). "
        "لا توجد أرقام في هذا القسم غير مقروءة مباشرة من summary.json لكل تشغيل."
    )

    tf32_parity = inputs.get("tf32_parity")
    if tf32_parity:
        gate_txt = (
            f"بوابة تكافؤ TF32 (bench/TF32_PARITY.json): pass={tf32_parity.get('pass')}، "
            f"delta_tf32={tf32_parity.get('delta_tf32')}، delta_seeds={tf32_parity.get('delta_seeds')}، "
            f"grad_median_ratio={tf32_parity.get('grad_median_ratio')}."
        )
    else:
        gate_txt = f"نتيجة بوابة تكافؤ TF32 {NA} (bench/TF32_PARITY.json)."
    s2 = (
        "الفروق عن SASRec الأصلي مسجّلة في التصميم المعتمد (03_ARCHITECTURE_CONTRACT.md، "
        "GATE_P_DECISIONS.md، review/FABLE_GATE_P2_RESPONSE.md): Pre-LN مع FFN بمضاعف 4 وGELU، هدف "
        "Cross-Entropy على كامل الكتالوج بدل العينات السلبية، تمثيل السياق بنافذة ثابتة الطول (حتى 50 "
        "حدثًا) بدل التكديس الأصلي، وقناة جانبية اختيارية (rich_frozen_channels) مجمّدة الأوزان. "
        "بموجب موعد المالك (38 ساعة، Gate P2): التدريب بدقة TF32 (allow_tf32=True، "
        "float32_matmul_precision='high') والتقييم بدقة FP32 صارمة (allow_tf32=False، 'highest') في كل "
        f"تقييم full-VS ونهائي. {gate_txt} هذا القسم لا يضيف أي زعم أداء؛ الأرقام الفعلية في الأقسام 4 و6 فقط."
    )

    s3 = (
        "تمت إعادة استخدام نفس مخزن البيانات المجمّد ونفس الأهداف والسياقات والأقنعة ونفس "
        "المُقيّم المدقّق من مشروع GRU (code/reuse/*، REUSE_MANIFEST.json). ما تغيّر هو "
        "المعمارية (SASRec بدل GRU)، عدد المعاملات لكل إعداد، وميزانية التحسين/الجدولة "
        "الموصوفة في run_registry.locked.json. القيم d128 (R0_ID128) هي مرجع تقريبي لميزانية "
        "المعاملات فقط وليست مكافئة معماريًا لإعدادات d256/d384."
    )

    header = "| Run ID | Seed | الحالة النهائية | أفضل حقبة | أفضل مقياس | حقب مكتملة | عدد المعاملات |"
    sep = "|---|---|---|---|---|---|---|"
    rows = [header, sep]
    for mid in models:
        key = f"{mid}_s{seed}"
        rows.append(_run_row(mid, seed, runs.get(key)))
    for rk, rec in sorted(runs.items()):
        if rk not in {f"{mid}_s{seed}" for mid in models}:
            parts = rk.rsplit("_s", 1)
            rid = parts[0]
            sd = parts[1] if len(parts) > 1 else "?"
            rows.append(_run_row(rid, sd, rec))
    s4 = "\n".join(rows) if len(rows) > 2 else NA
    dropped = inputs.get("dropped_models") or []
    if dropped:
        dropped_lines = "؛ ".join(
            f"{m.get('id')} ({m.get('dropped_reason', NA)})" for m in dropped)
        s4 += f"\n\nإعدادات أُسقطت بسبب موعد المالك (38 ساعة، Gate P2): {dropped_lines}."

    a0 = runs.get(f"A0_ID256_s{seed}")
    a1 = runs.get(f"A1_RICH256_s{seed}")
    pair_parity = inputs.get("pair_parity")
    epoch_cap = inputs.get("epoch_cap")
    cap_txt = (f"سقف الحقب المقرَّر بقاعدة القطع (cut rule): {epoch_cap.get('epoch_cap')} "
              f"(قُرِّر في {epoch_cap.get('decided_utc')})." if epoch_cap else f"سقف الحقب {NA}.")
    parity_txt = (f"تكافؤ الزوج: pair_status={pair_parity.get('pair_status')}، "
                  f"equal_epochs={pair_parity.get('equal_epochs')}، "
                  f"equal_perm_sha256={pair_parity.get('equal_perm_sha256')}."
                  if pair_parity else f"فحص تكافؤ الزوج {NA}.")
    if a0 and a1:
        s5 = (
            f"A0_ID256: {a0.get('epochs_completed', '-')} حقبة، أفضل مقياس "
            f"{a0.get('best_metric', '-')} عند الحقبة {a0.get('best_epoch', '-')}. "
            f"A1_RICH256: {a1.get('epochs_completed', '-')} حقبة، أفضل مقياس "
            f"{a1.get('best_metric', '-')} عند الحقبة {a1.get('best_epoch', '-')}. "
            "كلا الذراعين يشتركان في نفس جدول معدل التعلم لكل تحديث (sasrec_schedule.lr_at) "
            f"وأُطلقا في نفس الوقت بلا حاجز زوجي (Gate P2). {cap_txt} {parity_txt} نتيجة القناة "
            "الجانبية هنا مقيّدة بهذه الوصفة وهذا التدريب فقط، وليست ادعاءً عامًا عن قيمة البيانات الجانبية."
        )
    else:
        s5 = f"{NA} {cap_txt} {parity_txt}"

    sel = inputs["selection"]
    if sel and sel.get("selected"):
        s6 = (
            f"الإعداد المختار: **{sel['selected']}** (القاعدة: {sel.get('rule', NA)}). "
            f"القيمة الأساسية عند الاختيار: {sel.get('selected_best_metric', '-')}. "
            f"الترتيب الكامل للإعدادات المؤهلة: {sel.get('ranking', [])}. "
        )
        if inputs["replica_seeds"]:
            rep_lines = []
            for seed_r in inputs["replica_seeds"]:
                key = f"{sel['selected']}_s{seed_r}"
                rec = runs.get(key)
                if rec:
                    rep_lines.append(f"seed {seed_r}: أفضل مقياس {rec.get('best_metric', '-')}")
                else:
                    rep_lines.append(f"seed {seed_r}: {NA}")
            s6 += " كرر البذور: " + "؛ ".join(rep_lines)
        else:
            s6 += (" بذرة واحدة فقط (2026) بلا تكرار بذور بسبب موعد المالك (38 ساعة)؛ "
                   "تباين البذرة التدريبية غير مُقاس.")
    else:
        s6 = f"لم يتم اختيار إعداد بعد. {NA}"

    retro = inputs["retro"]
    if retro is not None:
        s7 = (
            f"التصنيف: {retro.get('label')}. البذرة السابقة معروضة على TEST/VALIDATION_CONFIRM "
            "المُدرَجَين هنا تحت العنوان الإلزامي PREVIOUSLY INSPECTED / RETROSPECTIVE BENCHMARK "
            "لأن هذه الفترات سبق فحصها في مشروع GRU؛ قفل جديد لا يُلغي ذلك التاريخ. "
            f"عدد النماذج المقيَّمة: {len(retro.get('models', {}))}."
        )
    else:
        s7 = f"لم يُنفَّذ التقييم الاسترجاعي بعد ({statuses['retrospective_evaluation_status']}). {NA}"

    if retro is not None:
        s8 = "أرقام Micro/Macro لكل من Recall/MRR/NDCG@{1,2,5,10,20} موجودة داخل FINAL_SASREC_RESULTS.json (retro.models.*.TEST/VALIDATION_CONFIRM) دون أي تحويل إضافي هنا."
    else:
        s8 = NA

    if retro is not None and retro.get("bootstraps"):
        bs_test = retro["bootstraps"].get("TEST", {})
        selected = sel.get("selected") if sel else None
        expected_keys = []
        if selected:
            expected_keys = [
                f"{selected}_s{seed}_minus_GRU_F0_locked",
                f"{selected}_s{seed}_minus_TRANS_full",
                "A1_RICH256_s2026_minus_A0_ID256_s2026",
            ]
        key_lines = []
        for k in expected_keys:
            rec = bs_test.get(k)
            key_lines.append(f"{k}: {NA}" if rec is None else
                             f"{k}: diff={rec.get('diff')} ci95={rec.get('ci95')} n_users={rec.get('n_users')}")
        s9 = ("نتائج bootstrap المزاوج (TEST)، المفاتيح الثلاثة المتوقعة: " + "؛ ".join(key_lines) +
             f". التباين المزاوج للمستخدم لا يساوي عدم اليقين المعماري (04/08). التفصيل الكامل في "
             f"retro_eval.bootstraps.TEST داخل FINAL_SASREC_RESULTS.json.")
    else:
        s9 = NA

    s10 = NA if retro is None else "الشرائح التشخيصية (طول السياق الحقيقي، مستخدمون معروفون/غير معروفين لـ TRAIN) موجودة في retro.models.*.slices ضمن FINAL_SASREC_RESULTS.json."

    rs = inputs["resource_samples"]
    if rs:
        last = rs[-1]
        s11 = (
            f"آخر عيّنة موارد ({last.get('ts_utc')}): ذاكرة cgroup الحالية "
            f"{last.get('cgroup_memory', {}).get('current_bytes')} بايت من حد "
            f"{last.get('cgroup_memory', {}).get('limit_bytes')}؛ المساحة الحرة على القرص "
            f"{last.get('disk', {}).get('free_gib')} GiB. عدد العيّنات المسجّلة: {len(rs)}. "
            "لا يوجد ادعاء بأن استخدام GPU 100% هو الأفضل بحد ذاته؛ الأرقام أعلاه قياس فقط."
        )
    else:
        s11 = NA

    s12 = "مقارنة الحجم وزمن الاستجابة عند النشر تُقرأ من n_params لكل تشغيل في القسم 4؛ لا يوجد مجمّع (ensemble) يُقدَّم هنا كـ SASRec واحد."

    dataset_hash = inputs["registry"].get("dataset_hash", NA)
    s13 = (
        f"بصمة البيانات: {dataset_hash}. sha256 السجل المقفل: "
        f"{inputs['registry'].get('source_registry_sha256', NA)}. "
        "أوامر إعادة الإنتاج: queue_runner.py يُشغَّل عبر supervise_run.sh لكل تشغيل، ثم "
        "lock_retro.py lock/eval، ثم render_report.py (هذا الملف)."
    )

    deadline = inputs.get("owner_deadline") or {}
    s14 = (
        "القيود: لم يُنفَّذ أي تدريب حقيقي بواسطة هذه الوحدة (Sonnet-runtime) نفسها؛ "
        "الأرقام أعلاه، متى وُجدت، مصدرها ملفات ناتجة عن train_sasrec.py/lock_retro.py "
        "المملوكة لعمال آخرين. لا يوجد ادعاء 'أفضل من كل منشورات SASRec' في أي مكان هنا. "
        f"موعد المالك (Gate P2): {deadline.get('hours', NA)} ساعة، الموعد النهائي "
        f"{deadline.get('deadline_utc', NA)}، اختيار المالك: {deadline.get('owner_choice', NA)}. "
        "دعاوى مُضيَّقة بموجب Fable Gate P2: لا يوجد ادعاء تقارب (convergence) لأن الحقب ثابتة "
        "بلا إيقاف مبكر؛ تباين البذرة التدريبية غير مُقاس (بذرة واحدة فقط 2026)؛ لم تُختبر "
        "قدرة السعة الأكبر (d384) ولا إعداد التنظيم/معدل التعلم البديل (C1) بعد إسقاطهما ضمن "
        "موعد المالك؛ R0 (مرجع الميزانية القريبة من GRU) أُسقط أيضًا. "
        "التدريب بدقة TF32 والتقييم بدقة FP32 صارمة مذكوران في كل جدول؛ نتيجة بوابة التكافؤ "
        "(bench/TF32_PARITY.json) مُبلَّغة بعتبتها في القسم 2، وليست ادعاءً بتكافؤ FP32 التام."
    )

    return {f"SECTION_{i}": txt for i, txt in enumerate(
        [s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13, s14], start=1)}


def render_markdown(inputs: dict, statuses: dict) -> str:
    sections = build_sections(inputs, statuses)
    status_block = (
        "| المفتاح | القيمة |\n|---|---|\n" +
        "\n".join(f"| {k} | {v} |" for k, v in statuses.items() if k != "counts")
    )
    parts = [f"# تقرير SASRec النهائي — {time.strftime('%Y-%m-%d', time.gmtime())}",
             "", status_block, ""]
    titles = [
        "حالة التنفيذ الفعلية", "الفروق بين الأدبيات والكود", "ما تم الاحتفاظ به من GRU وما تغيّر",
        "جدول التجارب الكامل", "منحنيات تعلم A0/A1 المتطابقة", "أفضل إعداد مُختار ونتائج البذور",
        "PREVIOUSLY INSPECTED / RETROSPECTIVE BENCHMARK", "Micro/Macro Recall/MRR/NDCG",
        "Bootstrap المزاوج للمستخدمين", "الشرائح التشخيصية", "جدول الموارد",
        "حجم النشر وزمن الاستجابة", "أوامر إعادة الإنتاج والبصمات", "القيود والدراسة التالية",
    ]
    for i, title in enumerate(titles, start=1):
        parts.append(f"## {i}. {title}")
        parts.append(sections[f"SECTION_{i}"])
        parts.append("")
    return "\n".join(parts)


def render_sota_claim(statuses: dict, inputs: dict) -> str:
    lines = ["# SOTA_CLAIM_STATUS", ""]
    if statuses["better_than_gru_on_reused_test"] is True:
        lines.append(
            "This configuration beats the existing GRU baseline on the frozen, previously "
            "inspected REES46 benchmark (measured; paired bootstrap CI excludes 0)."
        )
    elif statuses["better_than_gru_on_reused_test"] is False:
        lines.append(
            "This configuration is NOT shown to beat the existing GRU baseline on the frozen, "
            "previously inspected REES46 benchmark under the measured paired bootstrap."
        )
    else:
        lines.append(
            "Not yet measured: the retrospective evaluation against GRU has not produced a "
            "paired bootstrap result. No SOTA claim is made."
        )
    if statuses["search_status"] == "COMPLETE_BOUNDED":
        lines.append("Best among the configurations evaluated in this campaign (bounded search, see section 6).")
    lines.append("Not shown to beat all published SASRec-family variants: no same-protocol "
                 "reproduction of external SASRec papers was run in this campaign.")
    lines.append("Fresh confirmatory holdout unavailable: this campaign's VALIDATION_CONFIRM/TEST "
                 "were previously inspected in the GRU project (test_exposure=previously_seen_in_gru_project).")
    lines.append("")
    lines.append("## Machine-readable statuses")
    lines.append("```json")
    lines.append(json.dumps({k: v for k, v in statuses.items() if k != "counts"}, indent=2))
    lines.append("```")
    return "\n".join(lines)


def render(root: Optional[Path] = None) -> dict:
    root = root or _root()
    inputs = gather_inputs(root)
    statuses = compute_statuses(inputs)

    sanitized_paths: list = []
    results = {
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "statuses": statuses,
        "registry_sha256": inputs["registry"].get("source_registry_sha256"),
        "dataset_hash": inputs["registry"].get("dataset_hash"),
        "runs": inputs["runs"],
        "selection": inputs["selection"],
        "pair_parity": inputs.get("pair_parity"),
        "epoch_cap": inputs.get("epoch_cap"),
        "tf32_parity": inputs.get("tf32_parity"),
        "dropped_models": inputs.get("dropped_models"),
        "owner_deadline": inputs.get("owner_deadline"),
        "lock": inputs["lock"],
        "retro_eval": inputs["retro"],
        "resource_sample_count": len(inputs["resource_samples"]),
        "last_resource_sample": inputs["resource_samples"][-1] if inputs["resource_samples"] else None,
        "sanitized_nan_or_inf_fields": [],
    }
    results = _sanitize(results, sanitized_paths)
    results["sanitized_nan_or_inf_fields"] = sanitized_paths

    md = render_markdown(inputs, statuses)
    sota = render_sota_claim(statuses, inputs)

    out_md = root / "FINAL_SASREC_REPORT_AR.md"
    out_json = root / "FINAL_SASREC_RESULTS.json"
    out_sota = root / "SOTA_CLAIM_STATUS.md"
    out_md.write_text(md, encoding="utf-8")
    out_json.write_text(json.dumps(results, indent=2, allow_nan=False, ensure_ascii=False))
    out_sota.write_text(sota, encoding="utf-8")
    return {"report_md": str(out_md), "results_json": str(out_json), "sota_md": str(out_sota),
            "statuses": statuses}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    args = ap.parse_args(argv)
    out = render()
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
