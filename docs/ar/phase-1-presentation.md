# Phase 1 — Building the Data Foundation

**Privacy-Preserving Shopping Intelligence via Federated / On-Device Learning**
Dataset: REES46 · October 2019 · 42,448,764 events
Scope: `S1-DS-02` → `S1-DS-06`

> **How to use this file:** each `## Slide` is one slide. The English block is what goes **on
> the screen**. The 🗣️ blocks are what you **say** — Arabic first, English second. Nothing in a
> 🗣️ block belongs on the slide.
>
> **Every number here is read from an executed artifact, not estimated.**

---

## Slide 1 — Title

# Building the Data Foundation
### Before building any model

| | |
|---|---|
| **Dataset** | REES46 e-commerce, October 2019 |
| **Events** | 42,448,764 |
| **Users** | 3,022,290 |
| **Window** | 31 days |

<div dir="rtl" align="right">

🗣️ **عربي**

> «النهاردة هعرض المرحلة الأولى — وهي **مش بناء موديل**. دي مرحلة بناء **الأساس** اللي كل
> الموديلات هتقف عليه.
>
> وهدفها تجاوب على سؤال واحد قبل أي كود تعلّم: **هل الداتا دي تنفع نجاوب بيها على سؤال البحث
> أصلًا؟**»

</div>

🗣️ **English**

> "Today I'm presenting Phase 1, which is deliberately **not** a modelling phase. It builds the
> foundation every later model stands on.
>
> Its only job is to answer one question before a single line of learning code: **can this
> dataset answer our research question at all?**"

---

## Slide 2 — The Research Question

### We compare four ways of training the same model

| Regime | How it trains | Data leaves device? |
|---|---|---|
| **`R1`** Centralized | one model on the server, everyone's data pooled | ❌ Yes |
| **`R2`** Federated | each device trains locally, server averages **weights only** | ✅ No |
| **`R3`** Personalized | federated + per-device fine-tuning | ✅ No |
| **`R4`** On-device | each device trains entirely alone | ✅ No |

### The question

> **How much recommendation quality do we lose if the data never leaves the phone?**

<div dir="rtl" align="right">

🗣️ **عربي**

> «السؤال مش «نبني أحسن recommender». السؤال: **لو الداتا مسابتش تليفون المستخدم، هنخسر قد
> إيه من الجودة؟**
>
> وعشان نجاوب، لازم نقارن الأربع طرق دي على **نفس الداتا وبنفس المقياس بالظبط**. وأي خلل في
> الأساس بيخلي المقارنة دي **بلا معنى** — مش بيخليها أقل دقة، بيخليها بلا معنى.»

</div>

🗣️ **English**

> "The question is not 'build the best recommender'. It is: **how much quality do we lose if
> the data never leaves the user's phone?**
>
> To answer that, all four regimes must be compared on **exactly the same data with exactly the
> same metric**. Any flaw in the foundation does not make the comparison less accurate — it
> makes it meaningless."

---

## Slide 3 — The Dataset

| | Count |
|---|---:|
| Events | **42,448,764** |
| Users | **3,022,290** |
| Sessions | **9,244,772** |
| Products | **166,794** |
| Categories | **624** |
| Brands | **3,445** |
| Days | **31** (Oct 1 → Oct 31, 2019) |

### Event vocabulary — exactly three

`view` · `cart` · `purchase`

> ⚠️ **No `remove_from_cart`.** The original plan assumed one.

<div dir="rtl" align="right">

🗣️ **عربي**

> «الداتا REES46 — أكتوبر 2019. **42 مليون حدث** من 3 مليون مستخدم على 31 يوم.
>
> وحاجة مهمة جدًا: **قاموس الأحداث تلاتة بس** — مشاهدة، إضافة للسلة، وشرا.
>
> **مافيش `remove_from_cart`** — والخطة الأصلية كانت مفترضة إنه موجود وبانية عليه قاعدة.
> اكتشفنا ده في مرحلة القياس وصلّحنا الخطة. **دي أول حاجة القياس صلّحها.**»

</div>

🗣️ **English**

> "REES46, October 2019 — 42 million events from 3 million users across 31 days.
>
> One thing matters immediately: the event vocabulary is **exactly three** — view, cart,
> purchase.
>
> **There is no `remove_from_cart`**, and the original plan assumed one and built a rule on it.
> We found this during the audit and corrected the plan. **That was the first thing measurement
> fixed.**"

---

## Slide 4 — Event Distribution

| Event type | Count | Share |
|---|---:|---:|
| `view` | 40,779,399 | **96.07%** |
| `cart` | 926,516 | 2.18% |
| `purchase` | 742,849 | **1.75%** |

### Consequence

| Metric choice | Why |
|---|---|
| ❌ Accuracy | a model predicting "no purchase" always scores **98%** |
| ✅ **PR-AUC** | rank-aware, honest under heavy imbalance |

📊 **Chart:** horizontal bar or pie — 96.07 / 2.18 / 1.75

<div dir="rtl" align="right">

🗣️ **عربي**

> «التوزيع ده هو أول قرار تصميمي في المشروع.
>
> **الشرا 1.75% بس من الأحداث.** يعني أي مهمة بتتوقع الشرا هتبقى **مختلة التوازن بشدة**.
>
> ودي مش ملاحظة عابرة — دي اللي خلّتنا نختار **PR-AUC** بدل الـaccuracy. لأن لو استخدمنا
> accuracy، موديل بيقول «مفيش شرا» طول الوقت هيجيب **98%** ويبان عبقري وهو مش بيعمل حاجة.»

</div>

🗣️ **English**

> "This distribution is the project's first design decision.
>
> **Purchases are only 1.75% of events.** Any task predicting purchase is severely imbalanced.
>
> This is not a passing observation — it is why we use **PR-AUC** rather than accuracy. Under
> accuracy, a model that always says 'no purchase' scores **98%** and looks brilliant while
> doing nothing."

---

## Slide 5 — First Finding: a Long Tail

### Events per user

| Percentile | Events |
|---|---:|
| p50 (median) | **4** |
| mean | 14.05 |
| p90 | 34 |
| p95 | 57 |
| p99 | 141 |
| **max** | **7,436** |

### Users and sessions with no sequence

| | Count | Share |
|---|---:|---:|
| Users with exactly **1 event** | 691,246 | **22.9%** |
| Users with exactly **1 session** | 1,475,463 | **48.8%** |
| Sessions with exactly **1 event** | 3,270,069 | **35.4%** |

📊 **Chart:** bar of percentiles — p50, p90, p95, p99 (log scale for max)

<div dir="rtl" align="right">

🗣️ **عربي**

> «هنا أول حاجة غيّرت تصميم الدراسة كلها.
>
> **الوسيط 4 أحداث، والمتوسط 14، والأقصى 7,436.** يعني المتوسط **مضلل تمامًا** — فيه أقلية
> ضخمة بتسحبه لفوق. عشان كده إحنا بنعرض **percentiles مش متوسطات** في كل حاجة.
>
> وأخطر رقم: **نص المستخدمين تقريبًا عندهم جلسة واحدة بس**. ومستخدم بجلسة واحدة **مالوش تسلسل
> يتعلّم منه أصلًا**.
>
> ودي المشكلة المباشرة لـ`R4` — التعلّم على الجهاز. مستخدم بأربع أحداث **مستحيل** يتدرب عليه
> موديل محلي. فلو شغّلنا الدراسة على الكل، `R4` هيفشل **لأسباب مالهاش علاقة بالفيدرالي** —
> هيفشل لأن مفيش داتا أصلًا.»

</div>

🗣️ **English**

> "This is the finding that changed the study design.
>
> **Median 4 events, mean 14, max 7,436.** The mean is badly misleading — a small heavy minority
> drags it up. That is why we report **percentiles, never averages**.
>
> The most consequential number: **almost half of users have exactly one session**. A user with
> one session has **no sequence to learn from**.
>
> That is a direct problem for `R4`, on-device learning. A model cannot be trained locally on
> four events. If we ran the study on everyone, `R4` would fail **for reasons unrelated to
> federation** — it would fail from having no data."

---

## Slide 6 — Phase Map

| Task | Question it answers | Type |
|---|---|---|
| `S1-DS-02` | How do we read 42M rows efficiently? | Engineering |
| `S1-DS-03` | What is the canonical shape of an event? | Convention |
| `S1-DS-04` | **What is actually in the data?** | **Measurement** |
| `S1-DS-05/06` | **What rules will every model obey?** | **Decisions** |

> **Order is deliberate: we measured before we decided.**

<div dir="rtl" align="right">

🗣️ **عربي**

> «أربع تاسكات بترتيب مقصود.
>
> **مابدأناش بالقرارات — بدأنا بالقياس.** كل قرار في التاسكة الأخيرة مبني على رقم من التاسكة
> اللي قبلها. **مافيش قرار مبني على افتراض.**
>
> ولو حد سأل بعدين «ليه قرّرتوا كده؟»، الإجابة موجودة في القياس مش في رأي.»

</div>

🗣️ **English**

> "Four tasks, in a deliberate order.
>
> **We did not start with decisions — we started with measurement.** Every decision in the last
> task rests on a number produced by the one before it. **No decision rests on an assumption.**
>
> If anyone later asks 'why did you decide that?', the answer is a measurement, not an opinion."

---

## Slide 7 — `S1-DS-03` · One Canonical Event

### The problem

Every team member could read the data differently — different columns, types, ordering.

### 12 canonical fields

| | |
|---|---|
| `event_time` · `event_type` · `product_id` | the action |
| `category_id` · `category_code` · `brand` · `price` | the item |
| `user_id` · `user_session` | who and when |
| `source_file` · `source_row_number` · `row_id` | provenance |

### Canonical order

```
(event_time, source_row_number)
```

<div dir="rtl" align="right">

🗣️ **عربي**

> «التاسكة دي بتحل مشكلة تنظيمية بتبان صغيرة وبتكلّف كتير جدًا.
>
> **الترتيب المعياري** هو أهم حاجة فيها: لو حدثين حصلوا في **نفس الثانية** بالظبط، مين الأول؟
>
> لو كل واحد رتّبهم بطريقة، **التسلسلات هتطلع مختلفة**، والموديلات مش هتبقى قابلة للمقارنة —
> ومحدش هيعرف السبب.
>
> فثبّتنا القاعدة: الوقت الأول، وبعدين **رقم السطر في الملف الأصلي**. حاجة deterministic
> ومستحيل تختلف.»

</div>

🗣️ **English**

> "This task solves an organisational problem that looks small and costs a lot.
>
> The **canonical order** is the important part. If two events share the same timestamp
> exactly, which comes first?
>
> If each person orders them differently, **the sequences differ**, the models stop being
> comparable, and nobody can see why.
>
> So we fixed it: timestamp first, then the **row number in the source file**. Deterministic,
> and impossible to disagree on."

---

## Slide 8 — `S1-DS-04` · What the Audit Found

### Six findings that changed the design

| # | Finding | Design consequence |
|---|---|---|
| 1 | Purchases are **1.75%** of events | use **PR-AUC**, not accuracy |
| 2 | median 4 · mean 14 · max 7,436 | report **percentiles**, never means |
| 3 | **48.8%** of users have one session | no sequence to learn from |
| 4 | **35.4%** of sessions have one event | same problem at session level |
| 5 | `category_code` missing **31.84%** · `brand` **14.4%** | model on `category_id` (complete) |
| 6 | **No `remove_from_cart`** | **the plan assumed one** |

### Quality checks

| Check | Result |
|---|---:|
| Duplicate `source_row_number` | **0** |
| Chronological order violations | **0** |
| Fully duplicated events | 35,207 (0.08%) |
| Empty `user_session` | **2 rows** |

<div dir="rtl" align="right">

🗣️ **عربي**

> «التاسكة دي **مابتاخدش قرارات — بتقيس بس**. وطلّعت ست نتايج كل واحدة فيهم غيّرت حاجة في
> التصميم.
>
> أهمهم رقم 6: **الخطة الأصلية كانت مفترضة إن فيه حدث `remove_from_cart`** وبانية عليه قاعدة.
> الداتا مافيهاش. **لو مكناش قِسنا، كنا هنبني قاعدة على حدث مش موجود.**
>
> ورقم 5 كمان مهم: اسم الفئة ناقص في تلت الصفوف، بس **رقم الفئة كامل 100%**. فالموديل بيشتغل
> على الرقم، والاسم للعرض بس.
>
> ولاحظوا الجدول التاني: **صفر مخالفات ترتيب وصفر صفوف مكررة**. الداتا نضيفة من الناحية دي —
> وده **مقيس مش مفترض**.»

</div>

🗣️ **English**

> "This task **makes no decisions — it measures**. It produced six findings, each of which
> changed something.
>
> The most important is number 6: **the original plan assumed a `remove_from_cart` event** and
> built a rule on it. The data has none. **Without measuring, we would have built a rule on an
> event that does not exist.**
>
> Number 5 also matters: the category *name* is missing on a third of rows, but the category
> *id* is **100% complete**. So the model uses the id; the name is for display only.
>
> And note the second table: **zero ordering violations, zero duplicate rows**. The data is
> clean in that respect — **measured, not assumed**."

---

## Slide 9 — Decision 1 · What Is a Session?

### The problem

`user_session` is **not a global identifier**.

### The measurement

> **348 raw session values appear under more than one user.**

### The decision

| | |
|---|---|
| **Session key** | `(user_id, user_session)` — composite |
| **Empty sessions** (2 rows) | each becomes its own singleton, keyed by provenance |
| **30-minute rule** | ❌ **rejected** — REES46 supplies boundaries; we respect them |

<div dir="rtl" align="right">

🗣️ **عربي**

> «قيمة الجلسة في الداتا **مش فريدة عالميًا**. وعشان نتأكد **قِسنا**: **348 قيمة** ظهرت تحت
> أكتر من مستخدم.
>
> رقم صغير — بس لو استخدمنا القيمة لوحدها كمفتاح، الـ348 دول كانوا **هيخلطوا سلوك مستخدمين
> مختلفين في جلسة واحدة**.
>
> وفيه قرار تاني مهم — **قرار بالرفض**: فيه طريقة شائعة في الأبحاث اسمها الـ**30-minute rule**
> بتقول أي فجوة أكتر من نص ساعة = جلسة جديدة.
>
> **رفضناها.** لأن إعادة بناء **9.2 مليون جلسة** بقاعدة من عندنا معناها إننا **غيّرنا دلالة
> المصدر في الداتا كلها** — من غير أي دليل إن قاعدتنا أحسن من بتاعتهم.»

</div>

🗣️ **English**

> "The session value in the data is **not globally unique**. To be sure, we **measured**:
> **348 raw values appear under more than one user**.
>
> A small number — but had we used the value alone as a key, those 348 would have **merged the
> behaviour of different users into one session**.
>
> There is also a second decision here, and it is a **rejection**. There is a common research
> convention, the **30-minute rule**: any gap over half an hour starts a new session.
>
> **We rejected it.** Rebuilding **9.2 million sessions** under our own rule would **change the
> source's semantics across the entire dataset** — with no evidence that our rule is better
> than theirs."

---

## Slide 10 — Decision 2 · The Temporal Split ⭐

### The most important decision in this phase

| Split | Range | Days | Events | Share | Decisions? |
|---|---|---:|---:|---:|---|
| `TRAIN` | Oct 1 → 22 | 21 | 29,218,702 | **68.8%** | ✅ |
| `VALIDATION` | Oct 22 → 27 | 5 | 6,897,095 | 16.3% | ✅ |
| `TEST` | Oct 27 → 31 | 4 | 5,087,488 | 12.0% | ✅ |
| `LABEL_GRACE` | Oct 31 | 1 | 1,245,479 | 2.9% | ❌ **observation only** |

### Why not a random split?

> The model would train on **Oct 25** and be tested on **Oct 10** — it would **see the future**.

### Half-open intervals `[start, end)`

An event exactly at `00:00:00` on Oct 22 belongs to **VALIDATION only** — never both, never lost.

📊 **Chart:** horizontal timeline, four coloured bands sized 21 / 5 / 4 / 1

<div dir="rtl" align="right">

🗣️ **عربي**

> «دي أهم سلايد في العرض كله.
>
> لو قسّمنا الداتا عشوائيًا، الموديل هيتدرب على أحداث من آخر الشهر ويتمتحن على أوله. يعني
> **هيشوف المستقبل وهو بيتعلم** — والرقم هيطلع عالي جدًا **ومالوش أي معنى**.
>
> في recommendation، **التقسيم لازم يكون بالزمن**. أي حاجة تانية leakage.
>
> ويوم 31 حالة خاصة تستاهل تتشرح: تخيلوا مستخدم شاف منتج **الساعة 11 مساءً يوم 30**. هل
> اشتراه؟ لو وقفنا الداتا يوم 30، **مش هنعرف** — لأن الشرا ممكن يكون حصل بعد نص ساعة.
>
> فسبنا يوم 31 **للمراقبة بس**: **مابنبدأش أي قرار تقييم جديد فيه**، بس بنشوف نتايج القرارات
> اللي اتاخدت قبله. ده اسمه **label maturation** — نسيب وقت للإجابة إنها تظهر.
>
> وتفصيلة تقنية صغيرة بس مهمة: الفترات **نصف-مفتوحة**. الحدث اللي بيقع بالظبط على نقطة القطع
> بيروح للفترة اليمين **وبس** — فمفيش حدث بيتعدّ مرتين ولا حدث بيضيع. وفي فحص بيختبر ده
> تحديدًا.»

</div>

🗣️ **English**

> "This is the most important slide in the deck.
>
> With a random split, the model trains on events from late in the month and is tested on
> early ones. It **sees the future while learning** — the score comes out very high and means
> **nothing**.
>
> In recommendation, **the split must be temporal**. Anything else is leakage.
>
> October 31 is a special case worth explaining. Imagine a user views a product at **11pm on
> the 30th**. Did they buy it? If we stop the data at the 30th, **we cannot know** — the
> purchase may have happened half an hour later.
>
> So we keep October 31 **for observation only**: **no new evaluation decision starts there**,
> but outcomes of earlier decisions can mature. This is **label maturation**.
>
> One small technical detail that matters: the intervals are **half-open**. An event exactly on
> a cut-off goes to the right-hand split **and only there** — never counted twice, never lost.
> A dedicated check tests exactly this."

---

## Slide 11 — Decision 3 · Boundary Sessions

### The problem

A user starts shopping at **11pm on Oct 21** (TRAIN) and continues to **1am on Oct 22**
(VALIDATION). The session sits in both.

### The alternatives

| Option | Problem |
|---|---|
| Split it in half | **changes what a session means** — half a session is not a session |
| Assign all to TRAIN | **leaks later behaviour into training** |
| Assign all to VALIDATION | same leak, reversed |

### The decision — **exclude the whole session**

| | Count | Share |
|---|---:|---:|
| Crossing sessions | **4,995** | **0.054%** of 9,244,772 |
| TRAIN events removed | 10,113 | 0.035% |
| `C1` clients touched | 690 | of 388,789 |

### And we publish **which sessions**, not just how many

| Column | Why |
|---|---|
| `session_key` | the identity |
| `user_id` · `user_session` | so any language can apply the rule |
| `touch_pattern` | which splits it crossed — auditable |

<div dir="rtl" align="right">

🗣️ **عربي**

> «الجلسة اللي بتعدّي بين فترتين — نعمل بيها إيه؟
>
> جرّبنا التلات بدائل ووقعوا كلهم. فقرّرنا **نستبعد الجلسة كاملة**. والتكلفة **0.054% من
> الجلسات** — **تافهة مقابل منع تسريب حقيقي**.
>
> بس أهم حاجة في السلايد دي هي الجدول الأخير: **مانشرناش العدد بس — نشرنا الجلسات نفسها
> بأسمائها في ملف**.
>
> **ليه؟** عشان أي تاسكة جاية شغالة على VALIDATION عايزة تعرف إن جلسة معيّنة عابرة، **لازم
> تشوف صفوف TEST** — وهي ممنوعة تعمل كده.
>
> فالتاسكة اللي بتملك القاعدة **بتقرا مرة واحدة وتنشر**، والباقي **بيستهلك** ومابيلمسش TEST
> أصلًا.
>
> وحطينا الزوج الخام كمان — لأن اللينات التانية بتكتب كودها بنفسها، ومن غيره **مش هيقدروا
> يستخدموا الملف**.»

</div>

🗣️ **English**

> "What do we do with a session that crosses a boundary?
>
> We considered all three alternatives and all three fail. So we **exclude the whole session**.
> The cost is **0.054% of sessions** — negligible against preventing real leakage.
>
> But the important part of this slide is the last table: **we publish the sessions themselves,
> not just the count**.
>
> **Why?** Because any later task working on VALIDATION that wants to know whether a session
> crosses **must look at TEST rows** — and it is forbidden to.
>
> So the task that owns the rule **reads once and publishes**, and everything downstream
> **consumes** without touching TEST at all.
>
> We also include the raw pair, because the other engineering lanes write their own code —
> without it, they could not use the file."

---

## Slide 12 — Decision 4 · Who Is in the Study?

### The rule is computed from **TRAIN only**

Using validation behaviour to qualify a user would mean **selecting on the future**.

### Three candidate thresholds

| Candidate | Rule (events / sessions / **active days**) | Users | Share | TRAIN events held | Median events | Median days |
|---|---|---:|---:|---:|---:|---:|
| `BROAD` | 5 / 1 / 1 | 1,130,834 | 48.91% | **91.97%** | 13 | **2** |
| **`MAIN`** ⭐ | **10 / 2 / 3** | **388,789** | **16.81%** | **60.21%** | **29** | **4** |
| `STRICT` | 20 / 3 / 7 | 65,340 | 2.83% | 22.51% | 73 | 8 |

### Why `MAIN`

| Candidate | Why not |
|---|---|
| `BROAD` | keeps 92% of events, but **median 2 active days** — repeat-visit signal is weak |
| `STRICT` | narrows the study to **2.83% of users** — a study of power users only |
| **`MAIN`** | **60% of the content from 17% of users** — non-trivial history without elitism |

📊 **Chart:** grouped bar — users vs TRAIN events held, for the three candidates

<div dir="rtl" align="right">

🗣️ **عربي**

> «مش كل المستخدمين ينفعوا للدراسة دي. `R4` معناه موديل لكل جهاز — ومستخدم بأربع أحداث
> **مستحيل** يتدرب عليه.
>
> فحطينا قاعدة أهلية، و**بتتحسب من فترة التدريب بس**. لو قلنا «مؤهل لأنه كان نشط في
> VALIDATION» نكون استخدمنا **معلومة من المستقبل** عشان نختار مين يتدرب — وده leakage.
>
> وجرّبنا **تلات مستويات** مش واحد.
>
> `BROAD` بيمسك **92% من الأحداث** — رقم مغري. بس المستخدمين اللي بيقبلهم **متوسطهم يومين
> نشطين بس**، يعني إشارة الزيارة المتكررة ضعيفة جدًا. والفيدرالي كله قايم على إن الجهاز يتعلم
> من تاريخه.
>
> و`STRICT` بيضيّق الدراسة على **2.83% من المستخدمين**. ودي مشكلة **بحثية مش تشغيلية**: لو
> الـcohort كله «مستخدمين خارقين»، **كل الأنظمة هتبان كويسة والمقارنة تبقى أقل صدقًا** — لأن
> الحالة المهمة للتعلّم على الجهاز هي **التاريخ المتوسط مش الضخم**.
>
> فاخترنا `MAIN`: **60% من المحتوى بـ17% من المستخدمين**.»

</div>

🗣️ **English**

> "Not every user is usable for this study. `R4` means one model per device — and a model
> **cannot** be trained on four events.
>
> So we set an eligibility rule, computed from **TRAIN only**. Saying 'eligible because they
> were active in validation' would use **information from the future** to choose who trains —
> that is leakage.
>
> We tested **three thresholds**, not one.
>
> `BROAD` retains **92% of events** — a tempting number. But the users it admits have a
> **median of two active days**, so the repeat-visit signal is very weak. And federated
> learning depends entirely on a device learning from its own history.
>
> `STRICT` narrows the study to **2.83% of users**. That is a **research** problem, not an
> operational one: if the cohort is all power users, **every regime looks good and the
> comparison becomes less honest** — because the case that matters for on-device learning is
> the **median history, not the huge one**.
>
> So we chose `MAIN`: **60% of the content from 17% of the users**."

---

## Slide 13 — A Correction: Active Days, Not Elapsed Span

### The earlier draft

Used **elapsed span** = `last_seen − first_seen` → produced **230,185 users**

### Why that was wrong

| User appears on | Elapsed span | Active days |
|---|---:|---:|
| Oct 1 and Oct 31 | **30 days** | **2** |

### The corrected rule

**Active days** = distinct calendar dates with at least one event, inside TRAIN

<div dir="rtl" align="right">

🗣️ **عربي**

> «دي غلطة اتصلحت، ومهم أوي تتقال — لأنها بتوضح **إزاي القياس بيصلّح التصميم**.
>
> المسودة الأولى كانت بتقيس **المدى الزمني** — الفرق بين أول وآخر ظهور.
>
> والقياس أثبت إن ده غلط: مستخدم ظهر **يوم 1** وظهر **يوم 31** مداه **30 يوم** — بس نشاطه
> **يومين بس**.
>
> فبقينا نستخدم **الأيام النشطة الفعلية**. والفرق مش تجميلي: القاعدة القديمة كانت هتدخل
> مستخدمين **تاريخهم فعليًا فاضي** وتحسبهم مؤهلين.»

</div>

🗣️ **English**

> "This is a correction we made, and it is worth stating because it shows **how measurement
> fixes design**.
>
> The earlier draft used **elapsed span** — the gap between first and last appearance.
>
> Measurement showed this is wrong: a user seen on **Oct 1** and again on **Oct 31** has a
> **30-day span** — but only **two active days**.
>
> So we switched to **actual active days**. The difference is not cosmetic: the old rule would
> have admitted users with **effectively no history** and counted them as eligible."

---

## Slide 14 — The Five Cohorts

### Two questions decide everything

| Question | Branch | Cohort |
|---|---|---|
| **1.** When was the user first seen? | in TRAIN → ask question 2 | — |
| | first in VALIDATION | `C3_VAL` |
| | first in TEST | `C3_TEST` |
| | first on Oct 31 | `GRACE_ONLY` |
| **2.** Is their TRAIN history enough? | yes (10 / 2 / 3) | **`C1`** |
| | no | `C2` |

### The result

| Cohort | Users | Share | What it is |
|---|---:|---:|---|
| **`C1`** | **388,789** | **12.86%** | **the only cohort a per-device model can be built on** |
| `C2` | 1,923,411 | 63.64% | **cold-start / low-history** — a research case, not waste |
| `C3_VAL` | 391,205 | 12.94% | the model has never seen them |
| `C3_TEST` | 256,884 | 8.50% | same, in the test window |
| `GRACE_ONLY` | 62,001 | 2.05% | **cannot produce an evaluation decision at all** |

📊 **Chart:** donut, or a two-level decision tree

<div dir="rtl" align="right">

🗣️ **عربي**

> «التقسيم ده **سؤالين بس**: **إمتى شفناه أول مرة؟** و**تاريخه كافي؟**
>
> `C1` هي **المجموعة الوحيدة اللي ينفع نبني عليها موديل لكل جهاز** — 13% من المستخدمين بس
> **60% من أحداث التدريب**.
>
> و`C2` بـ64% **مش مستبعدة ومش مهملة** — دي حالة **cold-start** اللي البحث محتاج يقيسها: **هل
> الفيدرالي بيساعد المستخدم قليل التاريخ ولا لأ؟** دي واحدة من أهم أسئلة المشروع.
>
> وملاحظة مهمة جدًا لو حد سأل: `C3` معناها **جديد على نافذة الرصد** — **مش عميل جديد**. إحنا
> شايفين **شهر واحد بس**، والمستخدم ممكن يكون زبون من سنين. لو قلنا «عميل جديد» في التقرير
> نكون قلنا حاجة غلط.»

</div>

🗣️ **English**

> "This split is **just two questions**: **when did we first see them?** and **is their history
> enough?**
>
> `C1` is **the only cohort a per-device model can be built on** — 13% of users, but **60% of
> training events**.
>
> `C2`, at 64%, is **not excluded and not waste** — it is the **cold-start** case the research
> needs to measure: **does federation help the low-history user?** That is one of the project's
> central questions.
>
> And an important caveat if anyone asks: `C3` means **new to the observation window**, **not a
> new customer**. We see **one month only**; these users may have been customers for years.
> Calling them 'new customers' in a report would be wrong."

---

## Slide 15 — Finding: How Much Does One Client Have? ⭐

### Everything so far describes the **cohort**. The project compares **clients**.

`R3` and `R4` produce **one model per device** and are scored **per device**.

| Split | Clients with data | Share of `C1` | Median events |
|---|---:|---:|---:|
| TRAIN | 388,789 | **100%** | 29 |
| VALIDATION | 169,841 | **43.68%** | **0** |
| **TEST** | **130,464** | **33.56%** | **0** |

> **Two thirds of `C1` clients have no data in the test window.**

### Why

The split is **by time, not by user**. A user who shopped in the first three weeks and did not
return has training history and **nothing to be evaluated on**.

### What it changes

| We cannot say | We can say |
|---|---|
| "`R4` works for `C1`" | "`R4` works for **the third who returned**" |

📊 **Chart:** bar — 100% / 43.68% / 33.56%

<div dir="rtl" align="right">

🗣️ **عربي**

> «دي أهم حاجة قِسناها، ومعظم المشاريع **مابتقيسهاش أصلًا**.
>
> كل اللي فات بيوصف **الـcohort**. بس المشروع بيقارن **العملاء** — لأن `R3` و`R4` بينتجوا
> **موديل لكل جهاز** وبيتقيّموا **لكل جهاز**.
>
> فالسؤال مش «الـcohort فيه كام مستخدم» — السؤال **«العميل الواحد عنده كام؟»**
>
> والنتيجة: **43.68% بس** من `C1` عندهم داتا تحقق، و**33.56%** عندهم داتا اختبار. يعني **تلتين
> الكوهورت مالهمش حاجة نمتحنهم بيها**.
>
> والسبب إن التقسيم **بالزمن**: مستخدم اشترى في أول الشهر وماتعاش تاني — **عنده تدريب ومعندوش
> تقييم**. وده **سلوك تسوق طبيعي**، وأي temporal split صحيح هيطلّع نفس الحاجة. **لو مالقيناهاش
> كان ده سبب للشك في التقسيم نفسه.**
>
> **ده مش باج — بس بيغيّر اللي نقدر ندّعيه.** مانقولش «`R4` بيشتغل لمستخدمي `C1`»، نقول
> **«بيشتغل للتلت اللي رجعوا»** — وهم **أنشط من المتوسط** بطبيعتهم لأنهم اللي رجعوا.»

</div>

🗣️ **English**

> "This is the most important thing we measured, and most projects **never measure it**.
>
> Everything so far describes the **cohort**. But the project compares **clients** — because
> `R3` and `R4` produce **one model per device** and are scored **per device**.
>
> So the question is not 'how many users are in the cohort' — it is **'how much does one client
> have?'**
>
> The answer: only **43.68%** of `C1` have validation data, and **33.56%** have test data. **Two
> thirds of the cohort have nothing to be evaluated on.**
>
> The reason is that the split is **by time**. A user who shopped early in the month and did not
> return has training history and no evaluation data. That is **normal shopping behaviour**, and
> any correct temporal split produces it. **Had we not found this, it would have been a reason
> to doubt the split.**
>
> **This is not a bug — but it changes what we may claim.** We do not say '`R4` works for the
> `C1` cohort'. We say '**it works for the third who returned**' — and those clients are **more
> active than average**, precisely because they returned."

---

## Slide 16 — Finding: Client Skew ⭐

### Among clients who **do** have data

| Largest share of clients | Hold this share of VALIDATION events |
|---|---:|
| **1%** | **20.19%** |
| 5% | 49.62% |
| **10%** | **68.15%** |
| 25% | 92.43% |

### The school analogy

| A school with one class of 40 and one of 5 | Result |
|---|---|
| Average over **students** | the big class dominates |
| Average over **classes** | both count equally |

> **Two different numbers for the same school — and neither is wrong.**

### Which forces a decision

| Regime | Reads naturally as |
|---|---|
| `R1` — one model for everyone | **micro** — average over decisions |
| `R4` — one model per device | **macro** — average over clients |

> Comparing `R1` under micro against `R4` under macro compares **two different questions**.
> **`FedAvg` also weights devices by sample count** — the same 10% would dominate the federated model.

📊 **Chart:** cumulative (Lorenz) curve — 1/5/10/25% on x, share of events on y

<div dir="rtl" align="right">

🗣️ **عربي**

> «**عُشر العملاء عندهم تلتين الداتا.**
>
> ودي مش ملاحظة وصفية — **بتفرض قرار**.
>
> خدوا مثال المدرسة: مدرسة فيها فصل بـ40 طالب وفصل بـ5. لو حسبت المتوسط **على الطلبة**، الفصل
> الكبير بيسيطر. ولو حسبته **على الفصول**، الاتنين بوزن واحد. **رقمين مختلفين لنفس المدرسة،
> ومحدش فيهم غلط** — بس لازم تقول إنت بتستخدم أنهي واحد.
>
> عندنا نفس الحاجة: `R1` موديل واحد للكل، فبيتقرا طبيعي بمتوسط **على القرارات**. و`R4` موديل
> لكل جهاز، فبيتقرا طبيعي بمتوسط **على العملاء**.
>
> **ولو قارنّاهم بمتوسطين مختلفين، مانكونش قارنّا نظامين — نكون قارنّا سؤالين مختلفين.**
>
> وحاجة تانية: `FedAvg` — الخوارزمية المعيارية في الفيدرالي — **بتوزّن الأجهزة بحجم داتاها**.
> يعني **نفس الـ10% دول هيسيطروا على الموديل الفيدرالي نفسه**.
>
> عشان كده كتبنا **ADR** بالقرار ده — **قبل ما نشوف أي نتيجة**، عشان مانختارش المقياس اللي
> يدّينا القصة الأحلى.»

</div>

🗣️ **English**

> "**A tenth of the clients hold two thirds of the data.**
>
> That is not a descriptive note — **it forces a decision**.
>
> Take the school analogy: a school with one class of 40 and one of 5. Average over
> **students**, and the big class dominates. Average over **classes**, and both count equally.
> **Two different numbers for the same school, and neither is wrong** — but you must say which
> one you used.
>
> We have exactly that. `R1` is one model for everyone, so it reads naturally as an average
> **over decisions**. `R4` is one model per device, so it reads naturally as an average **over
> clients**.
>
> **Comparing them under different averages is not comparing two systems — it is comparing two
> questions.**
>
> And `FedAvg`, the standard federated algorithm, **weights devices by sample count** — so the
> same 10% would dominate the federated model itself.
>
> That is why we wrote an **ADR** fixing this — **before seeing any result**, so we could not
> pick the average that flatters us."

---

## Slide 17 — The Censoring Rule

> ### An unobservable outcome is **not a negative label**

| Wrong | Right |
|---|---|
| user viewed and did not buy → **negative** | **`CENSORED`** — unknown |

### The agreed representation

| Field | Value |
|---|---|
| `label` | `null` |
| `task_mask` | `false` |
| `status` | `CENSORED` |

### Measured impact

**70,467 decisions (17.95%)** would have been counted as **false negatives**.

<div dir="rtl" align="right">

🗣️ **عربي**

> «لو المستخدم شاف منتج في **آخر حدث في الجلسة**، إحنا **مانعرفش** هل كان هيشتريه ولا لأ.
> الجلسة خلصت.
>
> لو حسبناها «مااشتراش»، الموديل **هيتعلم معلومة غلط** — هيتعلم إن الناس مابتشتريش وهو مش صح.
>
> فالتمثيل **تلات حقول**: الـlabel **فاضي**، والـ**mask** بيقول للموديل «تجاهل المثال ده في
> التدريب»، والحالة `CENSORED`.
>
> **وليه تلاتة مش واحد؟** لأن لو حطينا صفر بدل فاضي، الموديل **هيتعلم الصفر**. الـmask هو اللي
> بيمنع ده.
>
> والأثر **مقيس**: **17.95% من قرارات المهمة دي** — يعني حوالي 70 ألف قرار — كانوا هيتحسبوا
> سوالب كذبًا.»

</div>

🗣️ **English**

> "If a user views a product as the **last event in a session**, we **do not know** whether they
> would have bought it. The session ended.
>
> If we record that as 'did not buy', the model **learns something false** — it learns that
> people do not buy, which is not what the data says.
>
> So the representation has **three fields**: the label is **null**, the **mask** tells the
> model to ignore this example during training, and the status is `CENSORED`.
>
> **Why three fields and not one?** Because if we wrote zero instead of null, the model would
> **learn the zero**. The mask is what prevents that.
>
> And the impact is **measured**: **17.95% of this task's decisions** — around 70,000 — would
> have been counted as false negatives."

---

## Slide 18 — Validation

### **19 checks · all PASS**

| Check | What it guarantees |
|---|---|
| Row count reconciles with the audit | **42,448,764** — no row lost |
| Every event lands in exactly one split | no overlap, no gap |
| An event on a cut-off goes right | half-open intervals work |
| Same raw value under two users → two keys | **a semantic test, not a count comparison** |
| Active days never exceed 21 | a logical impossibility |
| `LABEL_GRACE` starts no decision | observation and decision stay separate |
| Five cohorts are exclusive and exhaustive | every user in exactly one |
| Every excluded session carries its raw pair | other lanes can apply the rule |
| Cohort rule still holds after exclusion | **231 of 388,789 fall below (0.059%)** |

> **If any check fails, the notebook stops. It does not print a warning and continue.**

<div dir="rtl" align="right">

🗣️ **عربي**

> «كل قرار في العرض ده وراه **فحص بيتأكد منه**، والفحوص **بتقرا من الداتا نفسها** مش من أرقام
> مكتوبة في الكود.
>
> والفرق ده مهم: فحص بيطبع تحذير **حد ممكن يعدّي عليه**. **فحص بيوقّف الشغل لأ.**
>
> ولاحظوا السطر قبل الأخير — فحص بيتأكد إن نفس القيمة الخام تحت مستخدمين مختلفين بتطلّع
> مفتاحين. **ده اختبار دلالي**، مش مقارنة أعداد. لأن مقارنة الأعداد ممكن تعدّي وهي فاضية.»

</div>

🗣️ **English**

> "Every decision in this deck is backed by a **check that verifies it**, and the checks read
> **from the data itself**, not from numbers typed into the code.
>
> That distinction matters: a check that prints a warning is one **someone can walk past**. A
> check that **halts the notebook** is not.
>
> Note the second-to-last row — a check that the same raw value under two different users
> produces two different keys. That is a **semantic test**, not a count comparison. A count
> comparison can pass while being vacuous."

---

## Slide 19 — Where We Are · Who Owes What

### Three lanes working in parallel

| Lane | Responsibility |
|---|---|
| **Us** — `DATA_MODEL` | data, rules, task examples |
| `SE` | turning data into model-readable features |
| `PR` | the shared trainer contract |

### The five blockers on Phase 2

| # | Blocker | Owner | Status |
|---|---|---|---|
| 1 | `S1-ALL-G1` protocol freeze | **Us** | ✅ **Closed** |
| 2 | `S1-DS-07` task examples | **Us** | ✅ **Closed** |
| 3 | `S1-SE-05` product representation | `SE` | ⏳ |
| 4 | `S1-SE-06` price and time encoding | `SE` | ⏳ |
| 5 | `S1-PR-05` trainer contract | `PR` | ⏳ |

### Features today

| # | Feature | Status |
|---|---|---|
| 1 | Category | ✅ embedding |
| 2 | Event type | ✅ embedding |
| 3 | Time gap | ✅ log-encoded |
| 4 | Product identity | ⚠️ **stand-in** — `SE-05` owns the real one |
| 5 | Price | ⚠️ **stand-in** — `SE-06` owns the real one |
| 6 | Brand | ❌ present in the data, unused |

> **A real GRU already beat the strongest simple baseline on three features alone.**
> The remaining features **raise the number** — they do not create the gain.

<div dir="rtl" align="right">

🗣️ **عربي**

> «إحنا لين الـ**data**، وأول واحد في السلسلة.
>
> **العائقين اللي كانوا علينا — التجميد وملفات الأمثلة — اتقفلوا.** فاضل تلاتة عند لينات
> تانية.
>
> وأهم نقطة في السلايد دي: **دول تحسينات مش عوائق**.
>
> إحنا دربنا **GRU حقيقي بـ3 features بس** — الفئة ونوع الحدث وفرق الوقت — **وكسر أقوى خط أساس
> بسيط**.
>
> لما `SE-05` و`SE-06` يخلّصوا ونضيف المنتج والسعر والبراند، **الرقم هيرتفع** — بس **الموديل
> شغال دلوقتي، مش مستني**.»

</div>

🗣️ **English**

> "We are the **data** lane, first in the chain.
>
> **The two blockers that were ours — the freeze and the task examples — are closed.** Three
> remain, owned by other lanes.
>
> The key point on this slide: **those are improvements, not blockers**.
>
> We trained a **real GRU on three features alone** — category, event type, time gap — and it
> **beat the strongest simple baseline**.
>
> When `SE-05` and `SE-06` land and we add product, price and brand, **the number rises**. But
> **the model works today. It is not waiting.**"

---

## Slide 20 — Summary

### What we delivered

| | |
|---|---|
| A complete protocol | temporal split · sessions · boundaries · cohort |
| **388,789 clients** in `C1` | 60% of training events from 17% of users |
| Excluded-session list | leakage prevented, identities published |
| Client density and skew | **the evidence that forced the metric decision** |
| **19 checks** | all PASS |

### Three sentences to leave them with

| # | |
|---|---|
| **1** | The split is **temporal, not random** — anything else makes every later number meaningless |
| **2** | Cohort eligibility uses **TRAIN only**, and **active days, not elapsed span** |
| **3** | We measured **the client**, not the cohort — evaluation covers **a third**, and every number must say so |

<div dir="rtl" align="right">

🗣️ **عربي**

> «الخلاصة في جملة واحدة:
>
> **إحنا مابنيناش الموديل الأول وبعدين دوّرنا على مقاييس تناسبه. قِسنا الأول، وكتبنا القواعد،
> وبعدين بنينا.**
>
> والمرحلة الجاية: بناء الـGRU الكامل ومقارنة الأنظمة الأربعة — على الأساس ده بالظبط.»

</div>

🗣️ **English**

> "The summary in one sentence:
>
> **We did not build the model first and then look for metrics that suit it. We measured first,
> wrote the rules, and then built.**
>
> The next phase is the full GRU and the four-regime comparison — on exactly this foundation."

---

## Appendix A — Chart Specifications

| Slide | Chart type | Data |
|---|---|---|
| 4 | Horizontal bar or pie | view 96.07% · cart 2.18% · purchase 1.75% |
| 5 | Bar (log scale) | p50=4 · p90=34 · p95=57 · p99=141 · max=7,436 |
| 10 | Horizontal timeline, 4 bands | 21 / 5 / 4 / 1 days |
| 12 | Grouped bar | BROAD / MAIN / STRICT × (users · % of TRAIN events) |
| 14 | Donut or 2-level tree | C1 12.86 · C2 63.64 · C3_VAL 12.94 · C3_TEST 8.50 · GRACE 2.05 |
| 15 | Bar | TRAIN 100% · VALIDATION 43.68% · TEST 33.56% |
| 16 | Cumulative (Lorenz) curve | 1→20.19 · 5→49.62 · 10→68.15 · 25→92.43 |

**Suggested palette:** blue = train · green = validation · orange = test · grey = observation only

---

## Appendix B — Anticipated Questions

| Question | Answer |
|---|---|
| Why not the 30-minute sessionization rule? | REES46 supplies session boundaries. Rebuilding 9.2M sessions under our own rule changes the source's semantics with no evidence ours is better. |
| Why `MAIN` and not `STRICT`? | `STRICT` narrows the study to 2.83% of users. On-device learning's important case is the **median** history, not the huge one. |
| Evaluation on only a third of clients? | Yes — and those who returned are **more active than average**. It is declared and travels with every number. |
| The cohort was computed before exclusion? | Yes, **deliberately**. We measured the effect: **231 of 388,789 (0.059%)** fall below the rule afterwards. Fixing it would make membership depend on future-derived information — the fix is itself the leak. |
| Only one month of data? | Yes. A declared limitation — but REES46 is the only public dataset with **user IDs + multi-behaviour + price + brand + a full month**. |
| What is genuinely new here? | **Measuring the headroom before building the model.** Most work does not. |

---

## Appendix C — Glossary

| Term | Arabic | Meaning |
|---|---|---|
| **Leakage** | تسريب | the model sees information it would not have at prediction time |
| **Temporal split** | تقسيم زمني | train on the past, test on the future |
| **Label maturation** | نضج الـlabel | leaving time for an outcome to become observable |
| **Censoring** | حجب | an unobservable outcome is marked unknown, never negative |
| **Cohort** | مجموعة | the set of users a study runs on |
| **Client / device** | عميل / جهاز | one user — the unit federated learning trains on |
| **Macro / micro average** | متوسط على العملاء / على القرارات | one vote per client vs one vote per decision |
| **`FedAvg`** | — | the standard federated algorithm; weights devices by sample count |
| **PR-AUC** | — | a rank-aware metric that stays honest under class imbalance |
