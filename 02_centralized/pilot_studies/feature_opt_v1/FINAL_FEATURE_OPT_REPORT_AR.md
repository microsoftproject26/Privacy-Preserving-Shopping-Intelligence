# التقرير النهائي لدراسة الخصائص — `SASREC_FEATURE_OPT_V1`

> دراسة إضافية منفصلة عن حملة SASRec. **كل الأرقام على VALIDATION_SELECT فقط** (عيّنة 10k مستخدم، 112,365 قرار)، وهو قسم استُخدم
> سابقًا لاختيار نماذج الحملة؛ إذن هذا تطوير تكيّفي وليس holdout جديدًا. لم يُقرأ VC ولا TEST إطلاقًا.

## 1. الخلاصة
- **القرار: `KEEP_CURRENT`** — يبقى النموذج الحالي المثبّت دون تغيير.
- النموذج المثبّت (baseline pin): A1_RICH256 seed 2027 epoch 5 (VS 0.186513). σ̂ = 0.000869 (أقصى فرق بين بذرتين؛ مدى وليس انحرافًا معياريًا) → عتبة البوابة 2(b) = max(0.002, 2σ̂) = 0.002000.
- **C1_D3** (5 قنوات تكرار سببية، تهيئة صفرية) مقابل CONTROL بعد نفس التكيّف الدافئ (5M قرار): Δ +0.000262، CI [+0.000040, +0.000475]، Δ micro R@20 +0.000329 → البوابة 2(b): **فشل**.
- **Probes (اعتماد النموذج الحالي، ليست إعادة تدريب):** لا يوجد مرشح إزالة يجتاز البوابة 2(a) (Δ ≥ +0.001 وCI > 0 وكلا الشريحتين موجبتان). كل خصائص الجانب تُستخدم فعلًا: إخفاؤها يضر.
- **النظافة (lean view):** max|Δlogit| 2.10e-05، تطابق الرتب 0.998683 (148 تغيير)، Δ المقياس -7.5e-08 → الحالة **PARITY_FAILED** (فشل: rank_identical_ge_0.999)؛ لا ادعاء تكافؤ، والمخطط المختار هو F4 الحالي.
- **(MEASURED)** التكيّف الدافئ نفسه خفّض المقياس: CONTROL 0.185454 مقابل النموذج المثبّت 0.187117 (-0.001663)؛ وC1_D3 0.185716 (-0.001401). كلا الذراعين أقل من النموذج الحالي، إذن لا يوجد ما يُعتمد.
- **الحماية T0→T2:** شجرة البيانات متطابقة = True؛ ميتاداتا البيانات المتغيرة = []؛ ملفات قائمة تغيّرت = ['.local_ai/sasrec_v1/tools/event_probe.py'] (أداة المراقبة للقراءة فقط التي يملكها القائد)؛ الإضافات = ملفات قفل الحملة وأدوات تقريرها (متوقعة).
- **لا ادعاء سببي ولا SOTA.** Probes = اعتماد نقطة تفتيش واحدة؛ الـscreen تكيّف دافئ استكشافي، بذرة واحدة.

## 2. النطاق والتفويض
- قرار Fable (مفوض المالك): STAGED_WITH_EVIDENCE_GATES (`review/AUTO_DECISION_SCOPE.md`)؛ مراجعة الكود C: APPROVE_FOR_ISOLATED_EXPERIMENTS بعد FIX_REQUIRED؛ تصحيح FO-02 (عزل الاستيراد): APPROVE_RESUME (`review/fo02/`).
- التقاط لقطة من كود الحملة بايتيًا (`code/snap_sasrec`)؛ لا GPU قبل انتهاء الحملة؛ الملاءمة على TRAIN فقط؛ لا حذف بالقناع فقط؛ لا log مزدوج؛ لا VC/TEST.

## 3. التدقيق المقاس (MEASURED، كل السكان)
- التكرارات الدقيقة والأعلام الميتة على كل السكان (TRAIN 52,760,314 صف، VS 10,930,090): مخالفات غير صفرية = {'TRAIN': {}, 'VALIDATION_SELECT': {}}؛ أقصى باقي دوران local_hour = 6.5e-08. (8 أعمدة مكررة في 6 علاقات + 6 أعلام ميتة؛ تصحيح العدد في AMENDMENT_FO_01.)
- **عيب M1 (مؤكد على كل TRAIN):** مُطبِّع ميزتين من ميزات الفجوة الزمنية مُلائم على كل الصفوف بما فيها الأصفار المقنّعة:
| Feature | mask | fitted mean / std (all TRAIN rows) | valid-only mean / std (TRAIN) | VS valid mean |
|---|---|---|---|---|
| log1p_time_since_prev_session_last_event_s | prev_session_available | 1.8615 / 4.3549 | 11.8169 / 1.6580 | 11.4827 |
| log1p_time_since_prev_user_event_s | prev_user_event_available | 4.9503 / 3.3790 | 5.3789 / 3.1782 | 5.3559 |
- الإصلاح (ملاءمة على الصفوف الصالحة فقط مع حدود القص المسجلة) جزء من TRANSFORM_STATE؛ هو نظافة، وليس مكسب دقة.

## 4. Probes (اعتماد النموذج المثبّت على كل مجموعة)
| Probe | type | Δ macro MRR@20 | 95% CI | Δ micro R@20 | weekday Δ | weekend Δ |
|---|---|---|---|---|---|---|
| P1_ALL_SIDE_OFF | OOD anchor | -0.046669 | [-0.049070, -0.044077] | -0.164348 | -0.041016 | -0.049463 |
| P2_MAIN_CATEGORY_UNAVAILABLE | coherent absence | -0.006262 | [-0.007527, -0.004957] | -0.010822 | -0.005850 | -0.006314 |
| P3_BRAND_UNAVAILABLE | coherent absence | -0.005687 | [-0.006903, -0.004337] | -0.013848 | -0.006789 | -0.005328 |
| P4_CURRENT_PRICE_UNAVAILABLE | coherent absence | -0.009761 | [-0.011340, -0.008255] | -0.012789 | -0.007057 | -0.011124 |
| P5_PREV_GAP_UNAVAILABLE | coherent absence | -0.003619 | [-0.004456, -0.002867] | -0.011249 | -0.005136 | -0.003085 |
| P6_ITEM_AGE_UNAVAILABLE | coherent absence | -0.024654 | [-0.026842, -0.022451] | -0.052677 | -0.017636 | -0.028089 |
| P7_COLD_USER | coherent absence | -0.004068 | [-0.005173, -0.002887] | -0.013287 | -0.002999 | -0.004702 |
| P8_CALENDAR_FROZEN | single-point counterfactual | +0.000660 | [-0.000192, +0.001502] | +0.002608 | -0.000122 | +0.000979 |

- التفسير: reliance of THIS checkpoint under coherent absence; not the effect of removal with retraining. P1 مرساة خارج التوزيع؛ P8 counterfactual لنقطة واحدة.

## 5. الـScreen (تكيّف دافئ استكشافي، ≤ 6 GPU-h)
| Arm | cohort macro MRR@20 | cohort micro R@20 | Δ vs CONTROL | 95% CI | gate 2(b) |
|---|---|---|---|---|---|
| CONTROL | 0.185454 | 0.464273 | — | — | control |
| C1_D3 | 0.185716 | 0.464602 | +0.000262 | [+0.000040, +0.000475] | FAIL {"delta_ge_threshold": false, "ci_lower_gt_0": true, "recall_guard": true} |

- كل الأذرع: نفس الأوزان المثبّتة، AdamW جديد، نفس 5,000,000 قرار TRAIN بنفس الترتيب، 19,532 تحديث، warmup 200 ثم cosine إلى 1e-5، seed 2026، TF32 تدريب / FP32 تقييم.

## 6. المخرجات
- `SELECTED_FEATURES.json`: decision `KEEP_CURRENT`، model_in_use «pinned incumbent (unchanged)»، selected_schema «current full F4 schema (lean parity failed; no equivalence claim)».
- `TRANSFORM_STATE.json`: status `PARITY_FAILED_NOT_ADOPTED`، fit_split TRAIN، view_sha256 `86de261359151213…`.
- `FEATURE_SELECTION_RESULTS.csv`، `FINAL_FEATURE_OPT_RESULTS.json`، `state/FINAL_DECISION.json` (كتابة مرة واحدة).

## 7. مراحل التنفيذ والموارد
| Stage | state | attempt | end (UTC) | disk free GiB |
|---|---|---|---|---|
| T1_protect | DONE | 1 | 2026-09-26T07:24:28Z | 13.44 |
| T1_diff | DONE | 1 | 2026-09-26T07:24:28Z | 13.44 |
| stream_fit | DONE | 1 | 2026-09-26T07:24:58Z | 13.44 |
| full_profile | DONE | 1 | 2026-09-26T07:34:29Z | 13.44 |
| pin_baseline | DONE | 1 | 2026-09-26T07:34:30Z | 13.44 |
| probes | DONE | 1 | 2026-09-26T07:58:43Z | 13.43 |
| screen_control_c1 | DONE | 1 | 2026-09-26T08:27:57Z | 12.81 |
| finalize | DONE | 1 | 2026-09-26T08:27:59Z | 12.81 |
| T2_protect | DONE | 1 | 2026-09-26T08:28:00Z | 12.81 |
| T2_diff | DONE | 1 | 2026-09-26T08:28:00Z | 12.81 |

## 8. حوادث وتصحيحات
- FO-01: عدّ «Nine» كان علاقات لا أعمدة → 8 أعمدة في 6 علاقات؛ D3 = 4 قنوات مفاهيمية = 5 قيم (AMENDMENT_FO_01).
- اختبارات Sonnet B المستقلة كشفت عيبين (هروب مسار بالاسم الشقيق؛ ملف فارغ عند رفض NaN) → أُصلحا. مراجعة C: 8 ملاحظات أُصلحت.
- P4: تجاوز uint8 في dtype القناع → أُصلح قبل التشغيل. تشغيل تجريبي تجاوز حارس الذاكرة → قُتل خلال 2.6 ث بلا أثر.
- FO-02: `train.py` المعاد استخدامه من GRU يضع `gru_v1/code` أولًا في sys.path؛ فشل probes مرتين (حارس العزل) → إصلاح في `fo_env.py` + 3 اختبارات → Fable APPROVE_RESUME؛ المراحل المكتملة صالحة (14/14 ملفًا متطابقًا بايتيًا).

## 9. الحدود
- VS أعيد استخدامه؛ لا holdout جديد؛ لا VC/TEST. Probes تقيس اعتماد نقطة تفتيش، وليست أثر إزالة مع إعادة تدريب. الـscreen تكيّف دافئ على 5M قرار ببذرة واحدة، وليس تدريبًا من الصفر. لا ادعاء بأن أي خاصية «سببية».
