# مراجعة Fable 5.1 الشاملة + الخطة الحرفية لباقي المشروع — PPSI V4
كتبها Claude Fable 5.1 (بطلب المالك، جلسة 2026-09-28 ~17:00Z) بصفة مراجع ومدير عمليات. السيرفر مقفول للترقية ولم يُلمس.
المرجع الحاكم: سؤال البروبوزال — **"How much shopping recommendation quality can be retained when moving from
cloud-based to on-device or federated models?"** (Privacy-Preserving Shopping Intelligence — Research Proposal 1.pdf).
الاختلاف عن البروبوزال في المهمة/الموديلات/الداتا مقصود ومتفق عليه مع الفريق والمشرف؛ المراجعة تحكم على **الفكرة والسؤال**.
كل مدة زمنية هنا **INFERENCE**. كل قرار اتخذته هنا مسجّل في `contracts/PREG1_AMENDMENT_06A.md` قبل أي قراءة قيمة.

---
## 0. الحكم الإجمالي
- **اللي اتعمل سليم علميًا ولا يحتاج إعادة.** السباق (G2) مسجّل مسبقًا، محكوم بقواعد مجمّدة قبل القراءة، ونتيجته حاسمة
  (SASRec D256_B3 = C_BIG، 0.1612 على MAIN_VALIDATION، CI لا يشمل الصفر). C_SMALL (D64_B2) خلص ببذرتين. كود FL/DP/الجهاز
  متراجع (Opus) والاختبارات خضراء. **لا شيء قديم يحتاج إعادة، ولا شيء يحتاج سكيل أصغر.**
- **اللي هيخلّي الرسالة أقوى** (ترتيب الأثر): (1) فصل لوحة اختيار وصفة الـ FT عن مستخدمي التقييم (كانت CALIB ⊂ MAIN — عيب
  تسجيلي يُصلَّح بسطر كود قبل أي تشغيل)؛ (2) تسجيل تحليل الـ cold start الرصدي (حسب عدد أحداث المستخدم) كإجابة أساسية لـ
  Step 4 لأن باني التاريخ المقطوع (E11.4) غير موجود؛ (3) إعادة ترتيب المؤجَّلات: DP ε=1 قبل C_FULL لأنه أرخص بكثير ويكمّل
  منحنى الخصوصية؛ (4) منحنى "الجودة مقابل الـ bytes/الجولات" يُستخرج مجانًا من علامات التقييم الـ 12 داخل كل تشغيل FL؛
  (5) إفصاحات واجبة في الرسالة (انعكاس ترتيب GRU/SASRec عند نقطة 6.0، LR المحلي غير مضبوط، FedBN غير قابل للتطبيق).
- **سكيل أكبر؟** الوحيد المبرَّر هو C_FULL (مرجع "السحابة على السكيل الكامل"، 1.32M مستخدم) — يظل **اختياريًا بعد الأساس**،
  لأن الإجابة العلمية العادلة هي نفس الكوهورت (MAIN)، وC_FULL يضيف إطارًا لا يغيّر الإجابة.

---
## 1. مطابقة البروبوزال بما لدينا (خطوة بخطوة)
| بند البروبوزال | ما لدينا (مسجّل/منفّذ) | الحالة | الفجوة/القرار |
|---|---|---|---|
| Step 1: Centralized upper bound | C_BIG = SASRec D256_B3 (83.9M) على MAIN، مقاييس MRR@20 + NDCG@10 + HR@10 من المقيّم المجمّد | ✅ منفّذ | Purchase-AUC و"comparable offers" خارج النطاق بالاتفاق (مهمة next-item). C_FULL اختياري كمرجع ثانٍ |
| Step 2a FedAvg | FA (64/جولة، 10% drop-out، 6,827 جولة، B16×2) | ✅ كود جاهز | تشغيل على A40 |
| Step 2b FedProx | FP (μ=0.01 غير مضبوط، مُفصح) | ✅ كود جاهز | — |
| Step 2c FedBN | مستبعد (المالك). SASRec يستخدم LayerNorm لا BatchNorm | ⚪ غير قابل للتطبيق | يُذكر في الرسالة كحدّ (limitation) + عمل مستقبلي |
| Step 2d Personalized FL | PF (متّجه شخصي p_u على الجهاز) + FT_FA (تخصيص الموديل كله) | ✅ كود جاهز | CPF (الضابط المكافئ للسعة) مؤجّل — غير مُنفَّذ في launch_fl |
| Communication cost / convergence | bytes لكل زيارة/عميل/أسطول + 12 علامة تقييم داخل كل تشغيل FL | ✅ | **قرار:** منحنى الجودة مقابل الجولات/الـ bytes التراكمية من العلامات الـ 12 (بدون تشغيل إضافي) |
| Privacy levels | DP-FedAvg ε=8 (z=0.5642)، ε=1 مؤجّل | 🟡 | **قرار:** ε=1 ببذرة واحدة يسبق C_FULL في المؤجَّلات |
| Step 3a Local model on own history | LO (SASRec D64 يتدرّب من θ0 على بيانات الجهاز فقط) | ✅ كود جاهز | — |
| Step 3b Pre-trained + local fine-tuning | FT_C (C_SMALL ثم fine-tune كامل على الجهاز) | ✅ | لوحة الاختيار → UNSEEN_VAL (§3) |
| Step 3c kNN on cached embeddings | BASELINE_KNN_EMB (جدول SMALL؛ BIG مرجع) | ✅ | KNN_EMB_BIG انقطع عند الإغلاق → يُعاد |
| Step 3d Rule-based | BASELINE_RULE (فئة/براند) منفّذ على MAIN | ✅ منفّذ | — |
| Step 4 Cold start: events-to-parity | لوحات مسجّلة: TRUE_COLD، UNSEEN_VAL/HOLDOUT (Study A/B)، SPARSE/HEAVY؛ COLD_H وWARM_TRUNCATION تحتاج باني تاريخ مقطوع **غير موجود** | 🟡 | **قرار:** التحليل الرصدي حسب عدد أحداث الدعم (5–10/11–20/21–50/51–100 + 0) هو الإجابة الأساسية؛ القطع المتحكَّم فيه (COLD_H) اختياري إن بقيت كوتة |
| Category-diversity impact | strata: تنوّع الفئات، ولاء البراند، persona 3×3، النشاط (حدود مسجّلة قبل القراءة) | ✅ | — |
| Device guidelines (memory/latency/battery) | ONNX opset 17، FP32/INT8، latency median/p95، RSS، الحجم، retention INT8 | ✅ كود | البطارية غير قابلة للقياس هنا → يُفصح |
| Open-source benchmark | الريبو + العقود + السجلات | ✅ | لا push بدون إذن |

**الخلاصة:** كل بنود السؤال مغطاة أو مُفصح عنها. الفجوة الوحيدة الجوهرية هي القطع المتحكَّم فيه للتاريخ (E11.4)، وبديله الرصدي مسجّل الآن.

---
## 2. مراجعة ما اتعمل (النتائج بالحرف، كلها REPORTED من ملفات مقروءة)
1. **G2** (`runs_p3/g2/G2_RESULT.json` 3397a643…): DECISIVE_STABLE، الفائز SASREC. PRACTICAL_BEST: SASRec 0.161196/0.161221
   (best@3.0) مقابل GRU D128 0.157141/0.156733 (best@6.0)؛ Δ = −0.004271، CI [−0.005180, −0.003261]، 40,764 مستخدم،
   1,000 إعادة عيّنة. NDCG@10: 0.1967 مقابل 0.1910. **سليم:** القاعدة T1 مجمّدة قبل أي نتيجة؛ السكريبت مجمّد (3893a49a…) قبل
   الفتح؛ الـ CALIB-disjoint diagnostic يتفق. **إفصاح واجب:** عند نقطة 6.0 ينعكس الترتيب (GRU +0.004213، CI [+0.003283, +0.005216]).
   SASRec يبلغ قمته عند 3.0 EFE ويخسر ≈0.0085 حتى 6.0 (overfit على MAIN الصغير). هذا **لا يهدد سؤال البحث** (الـ retention
   داخل عائلة واحدة)، لكنه يُكتب في فصل السباق مع كلا الصفّين.
2. **C_BIG** = تشغيلا 02B_SASREC نفسهما (best.pt عند 3.0 + endpoint_6.0.pt، الهاشات في ARCHITECTURE_LOCK fd201013…). لا إعادة.
3. **C_SMALL** = SASREC_D64_B2 (20.7M، 165.7 MB/زيارة)، وصفة من سلّم 4 LR على CALIB (peak_lr 0.000711)، بذرتان DONE_MAIN عند 6.0
   (configs 7381b5ce…/02511c1c…). القيم **لم تُقرأ** بعد. سليم. ملاحظة: لم يُعمل dry run قبل الإطلاق (كوتة) — الهاش مسجّل من LAUNCH_0؛ مقبول.
4. **Baselines على MAIN:** SKNN 0.11138، LAST 0.07111، POP 0.01943 (مقروءة بعد G2)، RULE منفّذ (لم يُقرأ)، KNN_EMB_BIG انقطع.
5. **كود المرحلة:** FL-0304 (drop-out، FT كامل، DP-FedAvg + RDP؛ z مؤكَّد مستقلًا بـ mpmath)، BASE-02، DEV-01، INT-03، INT-04 — مراجعة
   Opus واحدة + suites: tests_fl 181/57، tests_int 118/30، tests_train 206/58. INT-03/INT-04 قُبلا بلا مراجعة منفصلة (كوتة، مُفصح) —
   **مقبول** لأن مسار التقييم هو المدرّب المركزي المقفول، والـ smokes على الـ GPU تسبق أي تشغيل حقيقي.
6. **العقود:** amendments 03–06 مسجّلة قبل القراءات؛ strata cut points مسجّلة (0406e24c…)؛ التقسيم زمني (TRAIN حتى 15/11،
   VALIDATION 16–22/11، TEST 23–29/11 مختوم حتى ALL_METHODS_LOCK). سليم.

### عيوب وجدتها (وكيف تُصلَّح)
- **W1 (متوسط، يُصلَّح قبل أي تشغيل FT):** INT-04 يختار وصفة الـ FT على دور CALIBRATION، و**CALIB ⊂ MAIN** (رتب 0–32,768 من
  0–131,072). أي أن 25% من مستخدمي التقرير شاركوا في اختيار الوصفة. العقد E11.9a نصّ على لوحة **منفصلة** (M04S من مستخدمين غير
  مرئيين). **القرار:** لوحة الاختيار = دور **UNSEEN_VAL** (UNEW_VAL_SUPPORT → UNEW_VAL_VALIDATION؛ 12,792 مستخدم تقييم؛ منفصلة عن
  MAIN وعن UNSEEN_HOLDOUT ولوحات التقرير). تغيير كود صغير (`SELECTION_ROLE` في launch_device.py + اختباره) يُنشر مع code_ds3.
  إن تعذّر تقنيًا: CALIBRATION مع تقرير إلزامي على المجموعة الفرعية CALIB-disjoint من MAIN_VALIDATION (كما في G2).
- **W2 (متوسط، تسجيل):** لا يوجد باني تاريخ مقطوع (E11.4) → COLD_H/WARM_TRUNCATION غير قابلة للتشغيل. **القرار:** التحليل
  الرصدي (§3 بند 2) هو الإجابة الأساسية لـ Step 4، مسجّل الآن؛ الباني (DATA-07) اختياري في آخر القائمة.
- **W3 (منخفض، إفصاح):** LR المحلي في FL = SMALL peak فقط (شبكة {1×, 0.5×} من الخطة الواقعية لم تُنقل إلى amendment 06 ولا الكود).
  يُفصح كقيمة واحدة غير مضبوطة (مثل μ). لا تغيير (تكلفة).
- **W4 (منخفض، إفصاح):** تجميع العملاء في تقييم الجهاز يغيّر ترتيب 6 من 1,251 صفًا بمركز واحد عند رتب > 20 (لا أثر على @≤20). يُسجَّل بجانب قاعدة eval_batch.
- **W5 (منخفض):** مقياس الأجهزة A40 يغيّر `eval_device_class` في مفتاح المقارنة → كل الـ retention يُحسب داخل كتلة A40 بعد إعادة
  تقييم C_BIG/C_SMALL على A40 (تقييم فقط). مسجّل في DL 139/141. يُضاف: إيصال حتمية STRICT على A40 قبل أي تشغيل.
- **W6 (منخفض):** bench الـ ONNX يحتاج ملف صفوف NPZ لا تنتجه أي أداة بعد (DEV-01 مفتوح). مهمة صغيرة DEV-02 (Sonnet).
- **W7 (توثيق):** السجلات المهيكلة للرسالة (DECISION/EXPERIMENT/CLAIM) متأخرة منذ 2026-09-27 — تُعمل دفعة واحدة بعد نظرة TEST.

---
## 3. القرارات المسجّلة الآن (تفاصيلها في PREG1_AMENDMENT_06A.md)
1. لوحة اختيار وصفة FT/LO = **UNSEEN_VAL** (بديل M04S)؛ الاختيار مرة واحدة على الشبكة {1,2,4}×{0.1,0.3}؛ MAIN ولوحات التقرير لا تختار أبدًا.
2. **Cold start الرصدي (Step 4):** لكل ذراع، الـ retention مقابل C_BIG (وC_SMALL) داخل حزم عدد أحداث الدعم في فترة TRAIN:
   {0 (TRUE_COLD), 5–10, 11–20, 21–50, 51–100} على MAIN وعلى UNSEEN_HOLDOUT؛ events-to-90%/95% = أصغر حزمة تحقق M ≥ X·C_ref
   في نفس الحزمة (يُبلَّغ فقط إن تحقق؛ بلا استيفاء). القطع المتحكَّم فيه (COLD_H) اختياري.
3. **منحنى الاتصال/التقارب:** من العلامات الـ 12 داخل كل تشغيل FL: الجودة مقابل الجولات والـ bytes التراكمية لكل عميل. بلا تشغيل إضافي.
4. **ترتيب المؤجَّلات:** DP ε=1 (بذرة 2026) → C_FULL (بذرة 2026) → CPF → KD_SMALL/FT_KD → DATA-07 (COLD_H). يُنفَّذ فقط بعد
   ALL_METHODS_LOCK-ready للأساس وإن بقيت كوتة، وكل واحد يُسجَّل قبل تشغيله.
5. **A40:** إيصال T3 + حتمية STRICT على A40 قبل أي تشغيل؛ إعادة تقييم C_BIG/C_SMALL على A40؛ كتلة مقارنة واحدة (A40).
6. **ONNX INT8 retention:** على كامل MAIN_VALIDATION إن كان الزمن ≤ 2 س، وإلا على أول 16,384 مستخدم بمفتاح السكان (مسجّل).
7. **قاعدة الـ retention:** البسط والمقام بنفس قاعدة الاختيار (PRACTICAL_BEST/PRACTICAL_BEST والـ CONTROLLED_BUDGET/CONTROLLED_BUDGET)
   وبنفس البذرة، ثم متوسط البذرتين؛ bootstrap مزدوج 1,000، بذرة 20260926. الجهاز A40.
8. **سكريبت التحليل (ANA-01)** يُجمَّد بالهاش قبل نظرة TEST (كما g2_run.py)، ويُشغَّل أولًا على VALIDATION.
9. **إفصاحات الرسالة:** انعكاس 6.0؛ LR محلي غير مضبوط؛ FedBN غير قابل؛ AUC الشراء خارج النطاق؛ البطارية غير مقيسة؛ VALIDATION
   = max-of-marks (تحيّز متماثل)، TEST هو الرقم غير المتحيّز الوحيد.

---
## 4. هل شيء قديم يحتاج إعادة أو تغيير سكيل؟
| البند | الحكم | السبب |
|---|---|---|
| السباق GRU/SASRec (8 تشغيلات MAIN) | **لا إعادة** | مسجّل مسبقًا، مجمّد، حاسم؛ الانعكاس عند 6.0 يُفصح لا يُعالَج |
| C_BIG بسكيل أكبر (D384) | **لا** | غير واقعي للجهاز ولا يخدم السؤال؛ C_FULL (داتا أكبر لا موديل أكبر) هو الإضافة المنطقية |
| C_SMALL | **لا إعادة** | سلّم LR مسجّل، بذرتان، لم تُقرأ |
| SMALL أصغر (85 MB) | **لا** | المالك لم يطلب؛ 170 MB هو السقف المسجّل؛ يُذكر كعمل مستقبلي |
| البذور 2→3 | **لا** | CI مزدوج بالمستخدم هو أداة الاستدلال؛ فجوة البذور تُبلَّغ |
| Baselines | KNN_EMB_BIG يُعاد (انقطع)؛ KNN_EMB_SMALL جديد | — |
| P2 على CALIB | **لا** | كان فرزًا فقط؛ الاختيار الحقيقي تم على MAIN |

---
## 5. الخطة الحرفية (تُنفَّذ بالترتيب؛ لا خطوة قبل اكتمال ما قبلها إلا ما هو موسوم "بالتوازي")
**الرموز:** `V=/home/jovyan/ahmedabdelhamed01/Privacy-Preserving-Shopping-Intelligence/.local_ai/ppsi_v4`،
`PY=/home/jovyan/venvs/ppsi-gpu/bin/python`، `CR=$V/code_ds3`، `R=$V/runs_ds/arms`. كل أمر من داخل `$CR` مع
`export PYTHONPATH=$CR PYTHONDONTWRITEBYTECODE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8; unset CUDA_VISIBLE_DEVICES`.
الوظائف الطويلة detached: `(nohup setsid $PY -B … > log 2>&1 < /dev/null &)`. **لا قراءة لأي قيمة قبل §5.10.**
أرقام GPU على A40: يُحدَّد بعد الرجوع (`nvidia-smi -L`).

### 5.0 قبل رجوع السيرفر (محلي، بلا سيرفر) — Opus/Sonnet
- **INT-05 (Sonnet، صغير):** في `code/v4int/launch_device.py`: `SELECTION_ROLE = "UNSEEN_VAL"`، تحديث نص
  `SELECTION_SUBSTITUTE`، ونوع القاعدة `SELECTED_ON_UNSEEN_VAL`؛ تحديث اختبارات test_int04 المرتبطة؛ تشغيل tests_int + tests_fl
  (TMP على F:). القبول: الحزم المقفولة بلا تغيير، NCs تفشل كما هو مقصود.
- **DEV-02 (Sonnet، صغير):** أداة `v4dev/dump_rows.py` تكتب NPZ صفوف MAIN_VALIDATION بالتخطيط الذي يتحقق منه
  `validate_rows_layout` (من مزوّد INT-01)، مع `--first-users 16384` اختياري بترتيب مفتاح السكان. اختبار على fixture.
- **ANA-01 (Opus، متوسط، بلا سيرفر):** `tools/retention_report.py`: يقرأ RESULT_ROWS + ranks + strata + SELECTION، ويُخرج:
  جدول الـ retention (كل ذراع × بذرة × قاعدة checkpoint) مع CI؛ منحنيات (bytes، ε، حجم، strata، حزم الدعم)، harmed-client share،
  events-to-90/95%، وقيم NDCG@10/HR@10. يُختبر على fixtures. يُجمَّد بالهاش في `company/gates/ANA_SCRIPT_FREEZE.json` قبل §5.11.
- حزمة الكود code_ds3: `tools/code_sync.py` مع `--exclude code/.train03r1_tmp` (وTMP dirs)؛ الهاش يُسجَّل.

### 5.1 عند رجوع السيرفر (المالك يبلّغ) — تحقق فقط
1. `curl`-مكافئ عبر lobot: `date -u; nvidia-smi -L; df -h /home/jovyan; ls $V; ls $V/shared_data $V/data_v3 | head`.
2. هاشات: `sha256sum $V/contracts/ARCHITECTURE_LOCK.json $V/contracts/hp_registry/SMALL_RECIPE_SASREC_D64_B2.json
   $V/runs_p3/SASREC/02B_SASREC_s2026/sealed/checkpoints/best.pt` … (تطابق fd201013… وهاشات القفل).
3. `$PY -c "import torch;print(torch.__version__, torch.cuda.get_device_name(0))"` — لو تغيّرت نسخة torch/CUDA: **توقّف وأبلغ المالك**
   (الحتمية STRICT قد تختلف).
4. نشر code_ds3 (نفس نمط small_autopilot: فحص sha256 ثم `cp -a x/code $CR`).

### 5.2 إيصالات A40 (≈ 30–60 دقيقة)
1. إيصال T3 للمقيّم على A40 (نفس أمر إيصال G1 R5 على الشجرة code_ds3) → يجب PASS bitwise ضد الإيصال المرجعي أو يُسجَّل الفرق كـ A40.
2. حتمية STRICT: تشغيلان smoke لـ launch_central (SMALL، `--smoke`) بنفس البذرة على نفس الكارت → هاش الأوزان النهائية متطابق.
3. **إعادة تقييم C_BIG وC_SMALL على A40 (تقييم فقط، بلا تدريب):** لكل بذرة، best.pt وendpoint_6.0.pt (C_BIG) + best/endpoint (C_SMALL)
   على MAIN_VALIDATION → صفوف بـ `eval_device_class=A40`. تُحفظ تحت `$R/REEVAL_A40/`. (لا قراءة للقيم.)
4. **بالتوازي على CPU:** `v4base.run --baseline KNN_EMB --table-role BIG` (يُعاد) ثم `--table-role SMALL` (جدول C_SMALL s2026 best).

### 5.3 FL smokes (≈ 30 دقيقة)
- `$PY -B v4int/launch_fl.py --arm FA --seed 2026 --gpu g --run-dir $R/smoke_FA --history-policy USER_HISTORY_L50 --smoke --smoke-rounds 3 --small-recipe $V/contracts/hp_registry/SMALL_RECIPE_SASREC_D64_B2.json`
  ثم نفس الشيء لـ PF وDP (`--epsilon 8` بقيمة clip مؤقتة من smoke S_CAL). القبول: exit 0، ذاكرة GPU مسجّلة، RESULT_ROWS بالتنسيق المركزي.
- قياس الاستفادة من الكارت (nvidia-smi خلال الـ smoke) → يقرّر عدد الوظائف المتزامنة لكل A40 (amendment 06 §4: اثنان إن كان كلٌّ تحت الاستغلال).

### 5.4 FA / FP / PF (بذرتان لكل ذراع = 6 تشغيلات؛ ≈ 1–1.5 يوم على A40 مع المشاركة)
- `$PY -B v4int/launch_fl.py --arm {FA|FP|PF} --seed {2026|2027} --gpu g --run-dir $R/FL_<ARM>_s<S> --history-policy USER_HISTORY_L50 --allow-shared-gpu`
- التوزيع: A40-0: FA_s2026 + FP_s2026؛ A40-1: FA_s2027 + FP_s2027؛ ثم PF_s2026/PF_s2027. طابور واحد لكل كارت بسكريبت `fl_queue_<g>.sh`
  يكتب `QUEUE_DONE` (grep بـ `^QUEUE_DONE` لا `QUEUE_DONE` وحدها — درس DL).
- الاستئناف: نفس الأمر + `--resume`.

### 5.5 S_CAL → FA_1024 → DP ε=8 (≈ 0.5 يوم)
1. `$PY -B v4int/launch_fl.py --arm S_CAL --seed 2026 --gpu g --run-dir $R/S_CAL --history-policy USER_HISTORY_L50` → `S_RECORD.json`
   (يُنسخ إلى `contracts/dp/S_RECORD.json` + صف DL **قبل** أول DP).
2. `--arm FA_1024 --seed S … --clip-record $R/S_CAL/S_RECORD.json` (بذرتان).
3. `--arm DP --epsilon 8 --seed S … --clip-record $R/S_CAL/S_RECORD.json` (بذرتان).

### 5.6 الجهاز: الاختيار ثم التطبيق (≈ 0.5–1 يوم؛ الشظايا تتوازى)
1. **اختيار FT_C** على UNSEEN_VAL: `$PY -B v4int/launch_device.py --arm FT_C --seed 2026 --gpu g --run-dir $R/SEL_FT_C --history-policy USER_HISTORY_L50 --role UNSEEN_VAL --select --base $V/runs_ds/C_SMALL_s2026/sealed/checkpoints/best.pt --base-kind central`
   → `SELECTION.json` (يُنسخ إلى contracts/device/ + صف DL قبل أي تطبيق).
2. **اختيار FT_FA** بنفس الأمر مع `--arm FT_FA --base $R/FL_FA_s2026/ckpt/best.pt --base-kind fl --run-dir $R/SEL_FT_FA`.
3. (اختياري) اختيار LO: `--arm LO --role UNSEEN_VAL --select --run-dir $R/SEL_LO`؛ وإلا LO بميزانيات {1,2,4} كلها تُبلَّغ والوصفة المطبّقة على MAIN = المختارة.
4. **التطبيق على MAIN** (بذرتان): `--arm LO --seed S --role MAIN --run-dir $R/DEV_LO_s<S> --n-shards 64 [--shards a:b --allow-shared-gpu]`؛
   `--arm FT_C … --selection $R/SEL_FT_C/SELECTION.json --base $V/runs_ds/C_SMALL_s<S>/sealed/checkpoints/best.pt --base-kind central`؛
   `--arm FT_FA … --selection $R/SEL_FT_FA/SELECTION.json --base $R/FL_FA_s<S>/ckpt/best.pt --base-kind fl`.
5. **اللوحات** (بذرة 2026 على الأقل، 2027 إن سمح الوقت): `--role UNSEEN_HOLDOUT | SPARSE_PANEL | HEAVY_PANEL | TRUE_COLD` لكل من LO/FT_C/FT_FA؛
   وC_SMALL/FA/DP على نفس اللوحات بالتقييم فقط (Study A، context-only).

### 5.7 ONNX (CPU، بالتوازي مع 5.4–5.6)
- `$PY -B v4dev/dump_rows.py --role MAIN --out $R/onnx/rows.npz` ثم
  `$PY -B -m v4dev.bench --ckpt $V/runs_ds/C_SMALL_s2026/sealed/checkpoints/best.pt --family SASREC --variant SASREC_D64_B2 --v1-root <v1-mappings> --out $R/onnx --rows $R/onnx/rows.npz --manifest-n <n> --tag onnx_int8`.
  إن تجاوز الزمن المتوقع ساعتين: `--first-users 16384` (القرار §3.6).

### 5.8 الـ strata والتقرير المؤقت على VALIDATION (CPU)
- `ANA-01` على كل صفوف A40 (C_BIG/C_SMALL المعاد تقييمها + كل الأذرع + baselines) → `reports/RETENTION_VALIDATION.md/json`.
  **أول قراءة للقيم تكون هنا، دفعة واحدة، بعد اكتمال كل الأذرع الأساسية** (القراءة الجزئية ممنوعة إلا للحالة/المعدل).

### 5.9 المؤجَّلات (فقط إن بقيت كوتة ووقت؛ كل بند صف DL قبل تشغيله)
1. DP ε=1، بذرة 2026: `--arm DP --epsilon 1 …` (z=1.3094 مسجّل).
2. C_FULL بذرة 2026 (amendment 06 §5): يحتاج أولًا مانيفست FULL_TIEBREAK_TRAIN (1,315,914 مستخدم، أحداث TRAIN فقط، أعمدة S*)،
   امتداد REGISTRY_V2 + قفل، dry run، قياس ذاكرة؛ الوصفة 02B_SASREC بلا تغيير؛ التقييم على نفس صفوف MAIN_VALIDATION.
3. CPF (كود: ذراع جديد في launch_fl)، ثم KD_SMALL/FT_KD (كود TRAIN-06)، ثم DATA-07 باني التاريخ المقطوع (COLD_H/WARM_TRUNCATION).

### 5.10 ALL_METHODS_LOCK
- ملف `company/gates/ALL_METHODS_LOCK.json`: قائمة كل الأذرع + هاشات RESULT_ROWS + checkpoints + SELECTION + S_RECORD + سكريبت
  ANA-01 المجمّد + مفتاح المقارنة (A40). صف DL. **بعده لا تعديل لأي وصفة أو ذراع.**

### 5.11 نظرة TEST الوحيدة
- تقييم كل الأذرع المقفولة (وC_BIG/C_SMALL) على MAIN RETRO_TEST (+ لوحات TEST المسجّلة إن وُجدت) دفعة واحدة، تسجيل
  `TEST_EXPOSURE_REGISTER`، ثم ANA-01 → `reports/RETENTION_TEST.*`. الأرقام النهائية للرسالة من هنا فقط.

### 5.12 التوثيق (بعد كل مرحلة؛ دفعة نهائية بعد TEST)
- DECISION_LOG صف لكل حدث؛ RESUME_HERE يتحدّث؛ مرجع الرسالة: sources جديدة (S467+)، EN §14.16+، STATE/CHANGELOG، ثم السجلات
  المهيكلة (DECISION/EXPERIMENT/INCIDENT/CLAIM) دفعة واحدة بعد TEST؛ `tools/verify_reference.py` صفر أخطاء.

---
## 6. مخرجات الرسالة (الجداول والرسوم المستهدفة)
1. **الجدول الرئيسي (TEST):** الذراع | البذرة 2026 | 2027 | المتوسط | R% مقابل C_BIG [CI] | R% مقابل C_SMALL [CI] | bytes/عميل | ε | الحجم.
   الصفوف: C_BIG، C_SMALL، FA، FP، PF، FA_1024، DP ε=8 (ε=1)، FT_C، FT_FA، LO، KNN_EMB_SMALL/BIG، RULE، SKNN، LAST، POP.
2. **منحنيات:** R مقابل bytes التراكمية (من علامات FL)؛ R مقابل ε (∞، 8، [1])؛ R مقابل حجم الموديل (BIG→SMALL→INT8)؛ R لكل stratum
   (تنوّع الفئات، ولاء البراند، persona 3×3، النشاط)؛ R مقابل حزمة أحداث الدعم (cold start) + events-to-90/95%.
3. **التخصيص:** المكسب المزدوج FT_C−C_SMALL وFT_FA−FA وPF−FA مع CI، ونسبة العملاء المتضررين.
4. **الجهاز:** latency median/p95، RSS، الحجم FP32/INT8، retention INT8، bytes لكل زيارة.
5. **السباق (فصل مستقل):** الجدول عند PRACTICAL_BEST وعند 6.0 معًا، مع الانعكاس والإفصاحات.

## 7. سياسة الكوتة (ملزمة)
- هذه المراجعة هي **الاستخدام الوحيد** لـ Fable. بعدها: العمليات على **Opus** (جلسة واحدة تستأنف من RESUME_HERE)، الأعمال
  الميكانيكية على **Sonnet**، بلا Fable إلا إن طلب المالك. لا مراجعات منفصلة إضافية للكود الصغير (INT-05/DEV-02)؛ ANA-01 يُراجع
  بمراجعة Opus واحدة لأنه يُنتج أرقام الرسالة.
- كل مرحلة = سكريبت طابور واحد على السيرفر + مراقب محلي واحد حدثي (لا polling قصير).

## 8. المخاطر
- **A40/بيئة جديدة:** إن اختلفت torch/CUDA أو فشلت الحتمية → البديل A16 (إن بقيت) أو تسجيل الفرق كإيصال؛ لا تشغيل قبل الإيصال.
- **ذاكرة FL بالمشاركة على كارت واحد:** يُقاس في الـ smoke؛ عند OOM يُعاد بوظيفة واحدة لكل كارت.
- **الكوتة:** RESUME_HERE يحمل كل الأوامر؛ الأساس (§5.1–5.8) له الأولوية المطلقة؛ المؤجَّلات لا تُبدأ إلا بعد اكتماله.
- **TEST مرة واحدة:** preflight لـ ANA-01 على VALIDATION + تجميد الهاش قبل الفتح.
