// Supervisor deck generator (pptxgenjs). Every number here is listed with its source in SUPERVISOR_DECK_DATA.md.
const pptxgen = require("pptxgenjs");
const path = require("path");

const OUT = process.argv[2] || path.join(__dirname, "PPSI_Supervisor_Deck.pptx");
const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE"; // 13.333 x 7.5 in
pres.title = "Privacy-Preserving Shopping Intelligence — supervisor update";
pres.author = "Ahmed Abdelhamed";

// ---------- palette & type ----------
const C = {
  ink: "13262F", body: "33434A", muted: "5E6E75", line: "D5DDDF",
  panel: "F1F4F4", white: "FFFFFF",
  teal: "1F7A80", tealDark: "164E57", tealLight: "D8ECEC",
  amber: "E0932B", amberDark: "9A5A0C", amberLight: "FBEBD3",
  slate: "5C6B73", slateLight: "E6EAEC", gray: "A7B1B6",
  onDark: "E9F1F2", onDarkMuted: "A9C0C4",
};
const HF = "Cambria";   // headings
const BF = "Calibri";   // body
const W = 13.333, H = 7.5, MX = 0.6;

let pageNo = 0;
function footer(s, src) {
  pageNo += 1;
  s.addText(String(pageNo), { x: W - MX - 0.6, y: H - 0.42, w: 0.6, h: 0.3, fontFace: BF, fontSize: 10, color: C.muted, align: "right", margin: 0, isTextBox: true });
  if (src) s.addText("Source: " + src, { x: MX, y: H - 0.42, w: W - 2 * MX - 0.8, h: 0.3, fontFace: BF, fontSize: 9, color: C.muted, margin: 0, isTextBox: true });
}
function header(s, kicker, title, dark) {
  s.addText(kicker.toUpperCase(), { x: MX, y: 0.38, w: 8, h: 0.3, fontFace: BF, fontSize: 11, bold: true, charSpacing: 2, color: dark ? C.amber : C.teal, margin: 0, isTextBox: true });
  s.addText(title, { x: MX, y: 0.68, w: W - 2 * MX, h: 0.75, fontFace: HF, fontSize: 30, bold: true, color: dark ? C.white : C.ink, margin: 0, valign: "top", isTextBox: true });
}
function light(bg) { const s = pres.addSlide(); s.background = { color: bg || C.white }; return s; }
function dark() { const s = pres.addSlide(); s.background = { color: C.ink }; return s; }
function card(s, x, y, w, h, fill, opts = {}) {
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, rectRadius: 0.08, fill: { color: fill }, line: opts.line ? { color: opts.line, width: 1 } : { type: "none" } });
}
function txt(s, t, x, y, w, h, o = {}) {
  s.addText(t, Object.assign({ x, y, w, h, fontFace: BF, fontSize: 15, color: C.body, margin: 0, valign: "top", isTextBox: true }, o));
}
function chip(s, t, x, y, w, fill, color) {
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h: 0.32, rectRadius: 0.16, fill: { color: fill }, line: { type: "none" } });
  s.addText(t, { x, y, w, h: 0.32, fontFace: BF, fontSize: 11, bold: true, color, align: "center", valign: "middle", margin: 0, isTextBox: true });
}
function bullets(items, o = {}) {
  return items.map((t, i) => ({ text: t, options: Object.assign({ bullet: { indent: 16 }, breakLine: i < items.length - 1, paraSpaceAfter: 6 }, o) }));
}
function discussion(n, title, qs, notes, src) {
  const s = dark();
  s.addText("DISCUSSION / DECISION " + n, { x: MX, y: 0.38, w: 8, h: 0.3, fontFace: BF, fontSize: 11, bold: true, charSpacing: 2, color: C.amber, margin: 0, isTextBox: true });
  s.addText(title, { x: MX, y: 0.68, w: W - 2 * MX, h: 0.75, fontFace: HF, fontSize: 30, bold: true, color: C.white, margin: 0, isTextBox: true });
  const top = 1.75, gap = 0.22, ch = (H - top - 0.75 - gap * (qs.length - 1)) / qs.length;
  qs.forEach((q, i) => {
    const y = top + i * (ch + gap);
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: MX, y, w: W - 2 * MX, h: ch, rectRadius: 0.08, fill: { color: "1D3640" }, line: { type: "none" } });
    s.addShape(pres.shapes.OVAL, { x: MX + 0.3, y: y + ch / 2 - 0.3, w: 0.6, h: 0.6, fill: { color: C.amber }, line: { type: "none" } });
    s.addText(String(i + 1), { x: MX + 0.3, y: y + ch / 2 - 0.3, w: 0.6, h: 0.6, fontFace: HF, fontSize: 20, bold: true, color: C.ink, align: "center", valign: "middle", margin: 0, isTextBox: true });
    s.addText([{ text: q[0], options: { bold: true, color: C.white, fontSize: 18, breakLine: true } }, { text: q[1], options: { color: C.onDarkMuted, fontSize: 13 } }],
      { x: MX + 1.2, y: y + 0.08, w: W - 2 * MX - 1.5, h: ch - 0.16, fontFace: BF, valign: "middle", margin: 0, isTextBox: true });
  });
  pageNo += 1;
  s.addText(String(pageNo), { x: W - MX - 0.6, y: H - 0.42, w: 0.6, h: 0.3, fontFace: BF, fontSize: 10, color: C.onDarkMuted, align: "right", margin: 0, isTextBox: true });
  if (src) s.addText("Context: " + src, { x: MX, y: H - 0.42, w: W - 2 * MX - 0.8, h: 0.3, fontFace: BF, fontSize: 9, color: C.onDarkMuted, margin: 0, isTextBox: true });
  s.addNotes(notes);
  return s;
}

// =====================================================================
// 1. Title
{
  const s = dark();
  s.addText("PRIVACY-PRESERVING SHOPPING INTELLIGENCE", { x: MX, y: 1.3, w: 11, h: 0.4, fontFace: BF, fontSize: 14, bold: true, charSpacing: 3, color: C.amber, margin: 0, isTextBox: true });
  s.addText("How much recommendation quality survives when shopping data stays on the device?", { x: MX, y: 1.8, w: 11.2, h: 1.9, fontFace: HF, fontSize: 40, bold: true, color: C.white, margin: 0, valign: "top", isTextBox: true });
  s.addText("Cloud vs federated vs on-device learning for Microsoft Edge — progress update and decisions", { x: MX, y: 3.85, w: 11, h: 0.5, fontFace: BF, fontSize: 18, color: C.onDarkMuted, margin: 0, isTextBox: true });
  // three deployment dots motif
  const lbl = [["Cloud", C.slate], ["Federated", C.teal], ["On-device", C.amber]];
  lbl.forEach((l, i) => {
    s.addShape(pres.shapes.OVAL, { x: MX + i * 2.2, y: 4.85, w: 0.28, h: 0.28, fill: { color: l[1] }, line: { type: "none" } });
    s.addText(l[0], { x: MX + 0.4 + i * 2.2, y: 4.82, w: 1.7, h: 0.34, fontFace: BF, fontSize: 14, color: C.onDark, margin: 0, isTextBox: true });
  });
  s.addText("Ahmed Abdelhamed · MSc, Queen's University · Microsoft supervisor meeting · 30 September 2026", { x: MX, y: 6.3, w: 11, h: 0.35, fontFace: BF, fontSize: 13, color: C.onDarkMuted, margin: 0, isTextBox: true });
  s.addText("Status of all facts: as recorded 29 Sep 2026, 18:36 UTC", { x: MX, y: 6.65, w: 11, h: 0.3, fontFace: BF, fontSize: 11, color: C.onDarkMuted, margin: 0, isTextBox: true });
  pageNo += 1;
  s.addNotes(`الشريحة 1 — العنوان.
ابدأ بجملة واحدة: المشروع بيجاوب على سؤال البروبوزال: لو خلّينا بيانات التسوق على جهاز المستخدم بدل ما نبعتها للسحابة، هنخسر قد إيه من جودة الترشيحات؟
قول إن العرض فيه: السؤال، اللي اتبنى، إزاي خلّينا المقارنة عادلة وواقعية، اللي اتقاس لحد دلوقتي، اللي لسه شغال ومواعيده المتوقعة، والقرارات اللي محتاجين رأيه فيها.
مهم: كل الأرقام في العرض من ملفات المشروع، وأي تاريخ مستقبلي مكتوب عليه "estimate" (تقدير) مش نتيجة.
سؤال محتمل: "ليه مفيش نتائج نهائية؟" — لأن نتائج الـ validation الرئيسية والـ test مقفولة (sealed) لحد القراءة المسجّلة مسبقاً؛ ده بيمنع إننا نختار حاجة بعد ما نشوف النتيجة.`);
}

// 2. The question and why it matters for Edge
{
  const s = light();
  header(s, "The question", "Three ways Edge could learn from shopping behaviour");
  txt(s, [{ text: "“How much shopping recommendation quality can be retained when moving from cloud-based to on-device or federated models?”", options: { italic: true } }],
    MX, 1.5, W - 2 * MX, 0.6, { fontSize: 17, color: C.ink });
  const cols = [
    ["CLOUD", C.slate, C.slateLight, "Events go to the server", "One large model is trained and served centrally.", "Quality ceiling — but raw browsing leaves the device."],
    ["FEDERATED", C.teal, C.tealLight, "Events stay on the device", "The device trains locally and sends only model updates; the server averages them.", "Optional differential privacy adds noise to the updates."],
    ["ON-DEVICE", C.amberDark, C.amberLight, "Nothing is uploaded", "A small model is shipped once and can be fine-tuned locally on the user's own history.", "Strongest privacy; limited by one user's data."],
  ];
  const cw = (W - 2 * MX - 0.6) / 3;
  cols.forEach((c, i) => {
    const x = MX + i * (cw + 0.3), y = 2.45;
    card(s, x, y, cw, 3.55, c[2]);
    txt(s, c[0], x + 0.3, y + 0.3, cw - 0.6, 0.3, { fontSize: 12, bold: true, charSpacing: 2, color: c[1] });
    txt(s, c[3], x + 0.3, y + 0.7, cw - 0.6, 0.5, { fontSize: 20, bold: true, fontFace: HF, color: C.ink });
    txt(s, c[4], x + 0.3, y + 1.4, cw - 0.6, 1.0, { fontSize: 15 });
    txt(s, c[5], x + 0.3, y + 2.5, cw - 0.6, 0.8, { fontSize: 14, italic: true, color: C.muted });
  });
  txt(s, [{ text: "Headline measure: ", options: { bold: true, color: C.ink } }, { text: "retention = quality of a deployment ÷ quality of the cloud model, with a paired confidence interval." }],
    MX, 6.25, W - 2 * MX, 0.4, { fontSize: 15 });
  footer(s, "Research proposal §1; public study description (federated-deployment-study.md)");
  s.addNotes(`الشريحة 2 — السؤال وأهميته لـ Edge.
السؤال مكتوب حرفياً من البروبوزال. Edge شغال على جهاز المستخدم، فعنده فرصة يعمل ترشيحات تسوق من غير ما يبعت التصفح الخام للسيرفر.
الثلاث طرق: (1) السحابة: الأحداث بتروح للسيرفر، ده سقف الجودة. (2) الفيدرالي: الجهاز بيتدرب محلياً وبيبعت تحديثات الموديل بس، وممكن نضيف Differential Privacy. (3) على الجهاز فقط: موديل صغير بينزل مرة واحدة وممكن يتظبط (fine-tune) على تاريخ المستخدم.
الرقم الأساسي اسمه retention: جودة الطريقة ÷ جودة السحابة، ومعاه فترة ثقة (confidence interval) من paired bootstrap.
سؤال محتمل: "ليه retention مش الفرق المطلق؟" — لأنه بيجاوب مباشرة على صيغة السؤال "قد إيه بنحتفظ بالجودة"، وبنعرض الفرق المطلق كمان.`);
}

// 3. What changed from the proposal
{
  const s = light();
  header(s, "Scope", "What changed from the proposal, and why");
  const rows = [
    ["Dataset", "Criteo, MovieLens, Amazon, Taobao, RetailRocket", "REES46 e-commerce events (Oct–Nov 2019)", "Public, per-user timestamped shopping sessions → each user can be simulated as one device"],
    ["Task", "Next category, purchase likelihood, best offers", "Next product viewed, ranked over the full catalogue", "One well-defined task, exact ranking, no sampled negatives"],
    ["Metric", "NDCG@10, HR@10, purchase AUC", "MRR@20 per user (NDCG@10, HR@10 also reported)", "Per-user ranking quality matches a browser serving one user"],
    ["Models", "Not fixed", "GRU vs SASRec contest → SASRec", "Backbone chosen by a pre-registered rule, not by preference"],
    ["Federated", "FedAvg, FedProx, FedBN, personalised FL", "FedAvg, FedProx, personalised FL, 8-bit uploads; tuned server optimizers", "FedBN not applicable: the model uses layer norm, not batch norm"],
    ["Privacy", "“Various privacy levels” (not defined)", "Differential privacy at ε = 8 and ε = 1; a six-level privacy matrix", "Makes “privacy level” measurable"],
  ];
  const hdr = ["", "Proposal", "Now", "Why"].map(t => ({ text: t, options: { bold: true, color: C.white, fill: { color: C.ink }, fontSize: 13 } }));
  const body = rows.map((r, i) => r.map((t, j) => ({ text: t, options: { bold: j === 0, color: j === 0 ? C.ink : C.body, fill: { color: i % 2 ? C.white : C.panel }, fontSize: 12.5 } })));
  s.addTable([hdr, ...body], { x: MX, y: 1.55, w: W - 2 * MX, colW: [1.3, 3.2, 3.6, 4.03], fontFace: BF, border: { type: "solid", pt: 0.5, color: C.line }, margin: [4, 6, 4, 6], rowH: 0.5 });
  txt(s, "Kept from the proposal: cloud upper bound, federated variants, on-device variants, cold-start analysis, persona analysis, communication cost, browser deployment check.", MX, 6.25, W - 2 * MX, 0.5, { fontSize: 14, italic: true, color: C.muted });
  footer(s, "Research proposal §§3–4; study description; project map; realism audit §F");
  s.addNotes(`الشريحة 3 — إيه اللي اتغير عن البروبوزال وليه (اتفقنا عليه معاه قبل كده).
- الداتا: بدل Criteo/MovieLens/... استخدمنا REES46: داتا تسوق عامة فيها كل مستخدم بأحداثه بالتوقيت، فنقدر نعامل كل مستخدم كجهاز.
- المهمة: بدل توقع الفئة والشراء، بقت "المنتج الجاي اللي المستخدم هيشوفه"، ومترتب على الكتالوج كله (158,486 منتج) من غير عينات سلبية.
- المقياس: MRR@20 لكل مستخدم وبعدين متوسط على المستخدمين، ومعاه NDCG@10 و HR@10.
- الموديل: عملنا مسابقة مسجلة مسبقاً بين GRU و SASRec.
- FedBN اتشال لأن SASRec بيستخدم LayerNorm مش BatchNorm، فـ FedBN مالوش معنى هنا.
- الخصوصية: البروبوزال قال "مستويات خصوصية" من غير تعريف، إحنا عرّفناها كمصفوفة من 6 مستويات + DP عند ε=8 و ε=1.
سؤال محتمل: "ليه مش داتا Edge نفسها؟" — البروبوزال نفسه بيقول السؤال عن المعمارية، وأي داتا مقسمة حسب المستخدم بتعمم؛ وكمان داتا عامة علشان النشر والخصوصية.`);
}

// 4. Data and task
{
  const s = light();
  header(s, "Data and task", "Next-product prediction over the whole catalogue");
  const stats = [["158,486", "products ranked for every prediction"], ["131,072", "users in the main cohort (5–100 events each)"], ["2.59 M", "training decisions in the main cohort"], ["1.35 M", "users in the full pool (for the realistic cloud)"]];
  const sw = (W - 2 * MX - 0.9) / 4;
  stats.forEach((t, i) => {
    const x = MX + i * (sw + 0.3);
    card(s, x, 1.6, sw, 1.55, C.panel);
    txt(s, t[0], x + 0.25, 1.75, sw - 0.5, 0.7, { fontFace: HF, fontSize: 34, bold: true, color: C.teal });
    txt(s, t[1], x + 0.25, 2.5, sw - 0.5, 0.6, { fontSize: 13, color: C.body });
  });
  // timeline strip
  txt(s, "Chronological split — the model never sees the future", MX, 3.5, 8, 0.35, { fontSize: 16, bold: true, color: C.ink });
  const tx = MX, tw = W - 2 * MX, ty = 4.0;
  const segs = [["Training — 1 Oct to 15 Nov 2019", 0.70, C.slate, C.white], ["Validation — 16–22 Nov", 0.15, C.teal, C.white], ["Test — 23–29 Nov (sealed)", 0.15, C.amber, C.ink]];
  let cx = tx;
  segs.forEach(sg => {
    const w = tw * sg[1];
    s.addShape(pres.shapes.RECTANGLE, { x: cx, y: ty, w: w - 0.04, h: 0.6, fill: { color: sg[2] }, line: { type: "none" } });
    s.addText(sg[0], { x: cx + 0.1, y: ty, w: w - 0.2, h: 0.6, fontFace: BF, fontSize: 13, bold: true, color: sg[3], valign: "middle", margin: 0, isTextBox: true });
    cx += w;
  });
  txt(s, bullets([
    "Each user = one simulated device; a prediction uses at most the user's last 50 events.",
    "Calibration users (32,768, part of the main cohort) are the only place where tuning results may be read before the final analysis.",
    "The test week is opened once, after every method is locked. It contains Black Friday — disclosed and reported as a sub-period.",
  ]), MX, 4.9, W - 2 * MX, 1.5, { fontSize: 15 });
  footer(s, "Data protocol lock (via realism audit A1, project map §3); study description “Data and task”");
  s.addNotes(`الشريحة 4 — الداتا والمهمة.
- 158,486 منتج: كل توقع بيرتب الكتالوج كله، مفيش عينات سلبية، فالرقم صعب وحقيقي.
- 131,072 مستخدم في الـ main cohort، كل واحد عنده من 5 لـ 100 حدث تدريب، إجمالي حوالي 2.59 مليون قرار تدريب.
- 1.35 مليون مستخدم في الـ full pool: ده بنستخدمه لموديل سحابة "واقعي" (C_FULL) زي خدمة Edge الحقيقية.
- التقسيم زمني: تدريب لحد 15 نوفمبر، validation أسبوع 16–22، test أسبوع 23–29 ومقفول.
- Calibration users: 32,768 مستخدم جزء من الـ main cohort، هما المكان الوحيد اللي مسموح نقرأ فيه نتائج الضبط (tuning) قبل التحليل النهائي.
سؤال محتمل: "أسبوع الـ test فيه Black Friday؟" — أيوه، ومكتوب كـ disclosure، وهنعرض الفترة دي لوحدها.
سؤال محتمل: "الـ test اتشاف قبل كده؟" — في مرحلة سابقة (v1) اتقرأ مرة؛ ده مكتوب كـ "previously inspected" في الرسالة.`);
}

// 5. Fairness
{
  const s = light();
  header(s, "Method", "How we keep the comparison fair");
  const items = [
    ["Pre-registered", "Every rule, selection criterion and contrast is written and hash-sealed before the result it governs is read. Changes are dated amendments."],
    ["Same budget, same everything", "Same data version, features, evaluator and training exposure for every method; tuning happens only on calibration users."],
    ["Full-catalogue ranking", "Exact ranking over all 158,486 products with a pessimistic tie rule; unknown items score zero."],
    ["Paired bootstrap", "1,000 user-level resamples on the same users for every comparison; two seeds for the main methods."],
    ["One sealed test look", "Validation is read once for all methods together, then everything is locked, then the test week is opened once."],
    ["Deterministic runs", "Strict FP32, fixed seeds, frozen evaluator; the evaluator hash is part of every comparison key."],
  ];
  const cw = (W - 2 * MX - 0.6) / 3, chh = 2.2;
  items.forEach((it, i) => {
    const x = MX + (i % 3) * (cw + 0.3), y = 1.6 + Math.floor(i / 3) * (chh + 0.3);
    card(s, x, y, cw, chh, i % 3 === 1 ? C.tealLight : C.panel);
    s.addShape(pres.shapes.OVAL, { x: x + 0.3, y: y + 0.3, w: 0.42, h: 0.42, fill: { color: C.teal }, line: { type: "none" } });
    s.addText(String(i + 1), { x: x + 0.3, y: y + 0.3, w: 0.42, h: 0.42, fontFace: HF, fontSize: 15, bold: true, color: C.white, align: "center", valign: "middle", margin: 0, isTextBox: true });
    txt(s, it[0], x + 0.9, y + 0.33, cw - 1.1, 0.4, { fontSize: 17, bold: true, fontFace: HF, color: C.ink });
    txt(s, it[1], x + 0.3, y + 0.95, cw - 0.6, 1.2, { fontSize: 14 });
  });
  footer(s, "Evaluation contract; study description “How fairness is kept”; multi-expert review strengths");
  s.addNotes(`الشريحة 5 — إزاي المقارنة عادلة.
1) Pre-registration: أي قاعدة أو اختيار بيتكتب ويتختم بـ hash قبل ما نشوف النتيجة. أي تغيير = amendment بتاريخ.
2) نفس الميزانية: نفس الداتا والفيتشرز والـ evaluator وكمية التدريب لكل الطرق. الضبط على calibration users بس.
3) ترتيب على الكتالوج كله مع قاعدة تعادل متشائمة.
4) Paired bootstrap: 1000 إعادة عينة على مستوى المستخدم، ونفس المستخدمين في كل مقارنة. بذرتين (seeds) للطرق الرئيسية.
5) نظرة واحدة على الـ test بعد القفل.
6) تشغيل حتمي FP32 وبذور ثابتة.
سؤال محتمل: "كام amendment؟ واتعملوا قبل القراءة؟" — كتير (أكتر من 20)، وكل واحد مسجل بوقت و hash قبل القراءة اللي بيأثر فيها؛ ده من نقاط القوة في المراجعة المستقلة.`);
}

// 6. Discussion 1
discussion(1, "Is this the right yardstick?", [
  ["Should retention be measured against the matched cloud or the full-pool cloud?", "Matched cloud = same 131k users (isolates the learning method). Full-pool cloud = 1.35M users (what Edge would really run). We plan to report both — which should lead?"],
  ["Is per-user MRR@20 the right headline metric for a browser feature?", "NDCG@10 and HR@10 are reported too. Would a product team prefer a top-10 metric?"],
  ["Are two seeds and one sealed test look enough for the headline methods?", "A third seed costs about half a day of GPU time per method (estimate)."],
], `ديسكشن 1 — المسطرة اللي بنقيس بيها.
1) نقيس الـ retention قدام أي سحابة؟ السحابة المطابقة (نفس 131 ألف مستخدم) بتعزل تأثير طريقة التعلم. السحابة الكاملة (1.35 مليون) هي الواقع بتاع Edge. هنعرض الاتنين، بس مين يبقى الـ headline؟ رأيي: الكاملة للقصة بتاعة Edge والمطابقة للعلم.
2) MRR@20 لكل مستخدم مناسب؟ ولا فريق المنتج يحب top-10؟
3) بذرتين ونظرة test واحدة كفاية؟ بذرة تالتة تقريباً نص يوم GPU لكل طريقة (تقدير).
اسمع رأيه واكتبه.`, "the matched cloud saw 131,072 users; the full-pool cloud 1,348,682 users");

// 7. Backbone contest result
{
  const s = light();
  header(s, "Backbone contest", "SASRec won the pre-registered contest against GRU");
  s.addChart(pres.charts.BAR, [{ name: "Validation MRR@20 (2-seed mean)", labels: ["GRU (hidden 128)", "SASRec (hidden 256, 3 blocks)"], values: [0.1569, 0.1612] }], {
    x: MX, y: 1.55, w: 6.6, h: 4.6, barDir: "col", chartColors: [C.slate, C.teal], varyColors: true,
    showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0.0000", dataLabelFontSize: 14, dataLabelColor: C.ink,
    valAxisMinVal: 0.14, valAxisMaxVal: 0.165, valAxisLabelFormatCode: "0.000", valAxisLabelColor: C.muted, catAxisLabelColor: C.body, catAxisLabelFontSize: 13,
    valGridLine: { color: "E3E8EA", size: 0.5 }, catGridLine: { style: "none" }, showLegend: false,
    showTitle: true, title: "Best checkpoint within the training budget (pre-registered rule)", titleFontSize: 13, titleColor: C.ink,
  });
  const x2 = 7.6, w2 = W - MX - x2;
  card(s, x2, 1.6, w2, 1.7, C.tealLight);
  txt(s, "−0.0043", x2 + 0.3, 1.75, w2 - 0.6, 0.7, { fontFace: HF, fontSize: 36, bold: true, color: C.tealDark });
  txt(s, "GRU − SASRec, 95% CI [−0.0052, −0.0033] — excludes zero; stable across both seeds", x2 + 0.3, 2.5, w2 - 0.6, 0.7, { fontSize: 13 });
  txt(s, bullets([
    "8 runs on the main cohort: 2 sizes per family × 2 seeds, equal tuning budget.",
    "NDCG@10 agrees: 0.1967 vs 0.1910.",
    "A check on users outside the calibration set agrees (no flip).",
    "Validation: 40,764 users, 519,714 predictions.",
  ]), x2, 3.55, w2, 2.6, { fontSize: 15 });
  footer(s, "Contest result file; decision log (contest run, 28 Sep)");
  s.addNotes(`الشريحة 7 — نتيجة مسابقة الموديل الأساسي (backbone).
عملنا 8 تشغيلات على الـ main cohort: حجمين لكل عائلة × بذرتين، بنفس ميزانية الضبط. القاعدة مسجلة مسبقاً: "أفضل checkpoint متحقق منه داخل ميزانية التدريب".
النتيجة: SASRec 0.1612 مقابل GRU 0.1569 (MRR@20 على الـ validation). الفرق −0.0043 وفترة الثقة 95% من −0.0052 لـ −0.0033، يعني مش صفر، ومستقر في البذرتين.
NDCG@10 نفس الاتجاه: 0.1967 مقابل 0.1910. وفي فحص على مستخدمين خارج الـ calibration، نفس النتيجة من غير انقلاب.
الأرقام دي مسموح نعرضها لأنها من المسابقة اللي اتقرت بعد الختم.
سؤال محتمل: "الفرق صغير (0.004)؟" — صغير لكن فترة الثقة بعيدة عن الصفر؛ والأهم إن الاختيار كان بقاعدة مسبقة، مش "SASRec أحسن دايماً" — وده الشريحة الجاية.`);
}

// 8. Honest caveat
{
  const s = light();
  header(s, "Backbone contest — the caveat", "The order reverses at the fixed training endpoint");
  s.addChart(pres.charts.BAR, [
    { name: "GRU", labels: ["Best checkpoint within budget (registered rule)", "Fixed endpoint (end of budget)"], values: [0.1569, 0.1569] },
    { name: "SASRec", labels: ["Best checkpoint within budget (registered rule)", "Fixed endpoint (end of budget)"], values: [0.1612, 0.1527] },
  ], {
    x: MX, y: 1.55, w: 7.2, h: 4.6, barDir: "col", barGrouping: "clustered", chartColors: [C.slate, C.teal],
    showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0.0000", dataLabelFontSize: 13, dataLabelColor: C.ink,
    valAxisMinVal: 0.14, valAxisMaxVal: 0.165, valAxisLabelFormatCode: "0.000", valAxisLabelColor: C.muted, catAxisLabelColor: C.body, catAxisLabelFontSize: 12,
    valGridLine: { color: "E3E8EA", size: 0.5 }, catGridLine: { style: "none" },
    showLegend: true, legendPos: "t", legendFontSize: 13, legendColor: C.body,
  });
  const x2 = 8.2, w2 = W - MX - x2;
  txt(s, "What it means", x2, 1.6, w2, 0.4, { fontSize: 18, bold: true, fontFace: HF, color: C.ink });
  txt(s, bullets([
    "SASRec peaks halfway through the budget, then declines; GRU was still improving at the end.",
    "At the endpoint GRU is ahead: +0.0042, CI [+0.0033, +0.0052].",
    "Both rules were fixed before any result, so the verdict stands — but it is reported as “selected under the pre-registered rule”, never “SASRec beats GRU”.",
    "Not size-matched: 83.9M vs 21.0M parameters. A size-matched comparison is registered as descriptive.",
  ]), x2, 2.1, w2, 4.2, { fontSize: 14.5 });
  footer(s, "Contest result file; contest diagnostics (decision log, 28 Sep 14:05Z); contest review ruling");
  s.addNotes(`الشريحة 8 — التحفظ الأمين.
لو بدل "أفضل checkpoint" أخدنا نهاية الميزانية الثابتة، الترتيب بيتقلب: GRU 0.1569 و SASRec 0.1527. الفرق +0.0042 لصالح GRU وفترة الثقة من +0.0033 لـ +0.0052.
السبب: SASRec بيوصل لأحسن نقطة في نص الميزانية وبعدين بينزل، و GRU كان لسه بيتحسن في الآخر.
القاعدتين كانوا متسجلين قبل أي نتيجة، فالقرار ثابت؛ بس هنكتب في الرسالة "اتختار بالقاعدة المسجلة" مش "SASRec أحسن من GRU".
كمان مش نفس الحجم: 83.9 مليون باراميتر مقابل 21.0 مليون. في مقارنة وصفية بنفس الحجم (GRU 20.96M مقابل الموديل الصغير 20.71M) مسجلة.
سؤال محتمل: "ليه ما اخترتوش GRU طالما أصغر؟" — القاعدة المسجلة هي اللي بتقرر؛ لو غيرناها بعد ما شفنا النتيجة يبقى تحيز. والموديل اللي بيروح الجهاز أصلاً هو SASRec الصغير.`);
}

// 9. Model sizes and device budget
{
  const s = light();
  header(s, "Model sizes", "A large cloud model, and a device model sized for a browser");
  const cards = [
    ["Cloud model", "83.9 M", "parameters (hidden 256, 3 blocks)", C.slateLight, C.slate],
    ["Device model", "20.7 M", "parameters (hidden 64, 2 blocks); 98% of them are the two item tables", C.tealLight, C.tealDark],
  ];
  cards.forEach((c, i) => {
    const y = 1.6 + i * 2.0;
    card(s, MX, y, 4.6, 1.75, c[3]);
    txt(s, c[0].toUpperCase(), MX + 0.3, y + 0.2, 4, 0.3, { fontSize: 12, bold: true, charSpacing: 2, color: c[4] });
    txt(s, c[1], MX + 0.3, y + 0.5, 4, 0.7, { fontFace: HF, fontSize: 36, bold: true, color: C.ink });
    txt(s, c[2], MX + 0.3, y + 1.2, 4.1, 0.5, { fontSize: 12.5 });
  });
  txt(s, "Device download: 82.8 MB (FP32), about 21 MB after INT8 quantisation (estimate until measured).", MX, 5.75, 4.6, 0.8, { fontSize: 13, color: C.muted, italic: true });
  s.addChart(pres.charts.BAR, [{ name: "MB per federated visit", labels: ["Cloud-size model", "Device model (FP32)", "Device model, 8-bit uploads", "Frozen item tables*"], values: [671.1, 165.7, 103.6, 2.1] }], {
    x: 5.6, y: 1.55, w: W - MX - 5.6, h: 4.4, barDir: "bar", chartColors: [C.slate, C.teal, C.teal, C.amber], varyColors: true,
    showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0.0", dataLabelFontSize: 13, dataLabelColor: C.ink,
    valAxisHidden: true, valGridLine: { style: "none" }, catGridLine: { style: "none" }, catAxisLabelColor: C.body, catAxisLabelFontSize: 13, catAxisOrientation: "maxMin",
    showLegend: false, showTitle: true, title: "MB per federated visit (download + upload); device cap 170 MB", titleFontSize: 13, titleColor: C.ink,
  });
  txt(s, "* 265,162 uploaded parameters ≈ 1.06 MB each way, plus a one-time 81.78 MB table download. To be re-measured.", 5.6, 6.05, W - MX - 5.6, 0.5, { fontSize: 12, color: C.muted, italic: true });
  footer(s, "Architecture lock (parameters, bytes per visit); realism audit B12, C1; multi-expert review §3.2");
  s.addNotes(`الشريحة 9 — أحجام الموديلات وميزانية الجهاز.
- موديل السحابة: 83.9 مليون باراميتر. لو كان هيتدرب فيدرالي، الزيارة الواحدة كانت هتبقى حوالي 671 ميجا — مستحيل على الجهاز.
- موديل الجهاز: 20.7 مليون باراميتر، و98% منهم جداول المنتجات (embedding tables). الزيارة الفيدرالية: 82.84 تحميل + 82.84 رفع = 165.7 ميجا، تحت سقف 170 ميجا.
- الرفع بـ 8 بت: حوالي 103.6 ميجا (82.84 تحميل + 20.71 رفع) — محسوبة من عدد الباراميترز.
- جداول المنتجات المجمدة (frozen): بنرفع 265,162 باراميتر بس ≈ 1.06 ميجا في كل اتجاه، مع تحميل جداول مرة واحدة 81.78 ميجا. الرقم ده هيتقاس تاني.
- تحميل الموديل للاستخدام بس: 82.8 ميجا FP32، وحوالي 21 ميجا INT8 (تقدير لحد ما يتقاس).
سؤال محتمل: "165 ميجا كتير؟" — أيوه، أكبر من الإنتاج (Gboard موديله 1.4 مليون باراميتر). على لابتوب مقبول، على موبايل لأ. علشان كده في خيار الجداول المجمدة والـ 8 بت.`);
}

// 10. The comparison matrix
{
  const s = light();
  header(s, "What is compared", "Every method, placed by what leaves the device");
  const lv = [
    ["Cloud", C.slate, C.white, "Large cloud model · small cloud model · full-pool cloud (1.35M users)", "Raw events", "None"],
    ["Cloud-trained, runs on device", "4F6A73", C.white, "Small model shipped (ONNX FP32 / INT8) · nearest items on cached embeddings", "Raw events, once, for training", "None"],
    ["Pretrained on opted-in users", C.amberDark, C.white, "Pretrained model · + federated on private users · + frozen item tables · + local fine-tune", "Opted-in users: events. Private users: updates only", "As federated (DP optional)"],
    ["Federated", C.teal, C.white, "FedAvg (ladder-tuned) · FedProx · personal-vector FL · 8-bit uploads · large-cohort FedAvg", "Model updates (reveal viewed items)", "None; secure aggregation not simulated"],
    ["Federated + DP", C.tealDark, C.white, "DP-FedAvg at ε = 8 and ε = 1", "Clipped, noised updates", "Client-level (ε, δ = 5e-6) DP"],
    ["Federated + personalisation", "3FA0A0", C.ink, "Personal vector stays on device · federated model fine-tuned on device", "As federated", "As federated"],
    ["Device only", C.amber, C.ink, "Local-only model (control) · “last viewed item” rule", "Nothing", "Data never leaves"],
    ["Reference baselines", C.gray, C.ink, "Popularity · session kNN · category/brand rule (uses published aggregates)", "—", "—"],
  ];
  const hdr = ["Privacy level", "Methods", "What leaves the device", "Formal guarantee"].map(t => ({ text: t, options: { bold: true, color: C.white, fill: { color: C.ink }, fontSize: 12.5 } }));
  const body = lv.map(r => [
    { text: r[0], options: { bold: true, color: r[2], fill: { color: r[1] }, fontSize: 12 } },
    { text: r[3], options: { color: C.body, fill: { color: C.white }, fontSize: 12 } },
    { text: r[4], options: { color: C.body, fill: { color: C.panel }, fontSize: 12 } },
    { text: r[5], options: { color: C.body, fill: { color: C.white }, fontSize: 12 } },
  ]);
  s.addTable([hdr, ...body], { x: MX, y: 1.5, w: W - 2 * MX, colW: [2.55, 5.0, 2.6, 1.98], fontFace: BF, border: { type: "solid", pt: 0.5, color: C.line }, margin: [3, 6, 3, 6], rowH: 0.5 });
  footer(s, "Privacy ladder (pre-registration) as revised after the multi-expert review; study description “What is compared”");
  s.addNotes(`الشريحة 10 — مصفوفة المقارنة: كل طريقة حسب "إيه اللي بيخرج من الجهاز".
- السحابة: الأحداث الخام بتروح. فيها 3 موديلات: الكبير، الصغير، وسحابة الـ full pool.
- متدرب في السحابة وشغال على الجهاز: الأحداث بتروح مرة للتدريب، وبعدين الموديل ينزل (ONNX FP32/INT8).
- مُدرب مسبقاً على مستخدمين موافقين (opted-in): ده الفرع الواقعي (شريحة 17).
- فيدرالي: FedAvg بعد الضبط، FedProx، personal-vector، رفع 8 بت، وFedAvg بعدد عملاء كبير. التحديثات بتكشف المنتجات اللي المستخدم شافها، و secure aggregation مش متحاكي — ده مكتوب بصراحة.
- فيدرالي + DP: ε=8 و ε=1، ضمان على مستوى العميل.
- فيدرالي + تخصيص محلي.
- على الجهاز فقط: موديل محلي (control) وقاعدة "آخر منتج اتشاف".
- baselines: popularity، session kNN، قاعدة الفئة/البراند.
سؤال محتمل: "ليه قاعدة آخر منتج مهمة؟" — لأنها ببلاش ومن غير تدريب ومفيش خصوصية بتتكسر، فأي طريقة لازم تغلبها علشان يبقى ليها معنى.`);
}

// 11. Discussion 2
discussion(2, "Which deployment should be the headline answer for Edge?", [
  ["Which row should carry the Edge headline?", "Candidates: plain federated · pretrained on opted-in users then federated · cloud model fine-tuned on the device. The independent review recommends the pretrained branch."],
  ["Should “last viewed item” be the floor every method must beat?", "It needs no training and sends nothing. We now report every method's gain over it, with a confidence interval."],
  ["Is the privacy matrix the right framing for a product audience?", "Or would a single ordered ladder (cloud → federated → DP → device) be clearer?"],
], `ديسكشن 2 — مين الـ headline بتاع Edge؟
1) أنهي صف يبقى الإجابة الرئيسية لـ Edge؟ الخيارات: فيدرالي عادي، أو تدريب مسبق على مستخدمين موافقين ثم فيدرالي، أو موديل سحابة يتظبط على الجهاز. المراجعة المستقلة بتقترح الفرع المُدرب مسبقاً لأنه شبه اللي المتصفحات بتعمله فعلاً.
2) هل "آخر منتج اتشاف" يبقى الأرضية؟ إحنا خلاص بنعرض فرق كل طريقة عنه بفترة ثقة.
3) المصفوفة مناسبة لجمهور منتج؟ ولا سلم واحد مرتب أبسط؟`, "rows as on the previous slide");

// 12. FL problem found
{
  const s = light();
  header(s, "Federated learning — the problem we found", "Untuned federated learning was far below the cloud");
  const st = [
    ["≈ 25%", "Federated (best tuned stage so far) vs a matched central model on calibration users", "0.0288 vs ≈ 0.11–0.12 (central value is an estimate)", C.teal],
    ["≈ 44%", "“Last viewed item” rule vs the cloud model on the main validation users", "0.0711 vs 0.1612 — no training, nothing uploaded", C.amberDark],
  ];
  st.forEach((t, i) => {
    const x = MX + i * 3.2;
    card(s, x, 1.6, 2.95, 3.1, i ? C.amberLight : C.tealLight);
    txt(s, t[0], x + 0.25, 1.75, 2.5, 0.8, { fontFace: HF, fontSize: 40, bold: true, color: t[3] });
    txt(s, t[1], x + 0.25, 2.6, 2.5, 1.2, { fontSize: 13.5, color: C.ink });
    txt(s, t[2], x + 0.25, 3.85, 2.5, 0.8, { fontSize: 12, color: C.muted, italic: true });
  });
  txt(s, "Different user sets and budgets — a warning sign, not a result.", MX, 4.9, 6.2, 0.4, { fontSize: 13, italic: true, color: C.muted });
  const x2 = 7.3, w2 = W - MX - x2;
  txt(s, "Diagnosis: the recipe, not a bug", x2, 1.6, w2, 0.4, { fontSize: 18, bold: true, fontFace: HF, color: C.ink });
  txt(s, bullets([
    "No server optimizer: plain averaging, server learning rate 1, no momentum.",
    "Client optimizer restarted every visit: about 4 sign-like steps per visit.",
    "The cloud learning-rate schedule mapped onto rounds: about half of the budget at ≤ 10% of peak rate.",
    "Client learning rate too small: the first screen's winner sat at the edge of its grid.",
    "Both 158k-row item tables start from random, with about 3 visits per user.",
  ]), x2, 2.1, w2, 3.4, { fontSize: 14.5 });
  card(s, MX, 5.5, W - 2 * MX, 0.95, C.panel);
  txt(s, [{ text: "Response: ", options: { bold: true, color: C.ink } }, { text: "a pre-registered improvement ladder — each stage changes one thing, is screened on calibration users only, and is kept only if it gains ≥ 0.002. The untuned start stays in the thesis as the registered reference." }],
    MX + 0.3, 5.62, W - 2 * MX - 0.6, 0.75, { fontSize: 14.5, valign: "middle" });
  footer(s, "Improvement-ladder register; multi-expert review §1; realism audit §0; ladder research §1.2; contest + baseline values (decision log, 28 Sep)");
  s.addNotes(`الشريحة 12 — المشكلة اللي لقيناها في الفيدرالي.
- الفيدرالي في أحسن مرحلة لحد دلوقتي بيجيب 0.0288 على calibration users، وموديل مركزي بنفس الحجم على نفس المستخدمين بيجيب حوالي 0.11–0.12 (تقدير من مرحلة الضبط). يعني حوالي 25%.
- في المقابل، قاعدة "آخر منتج اتشاف" بتجيب 0.0711 مقابل 0.1612 للسحابة على الـ main validation، يعني حوالي 44% — من غير تدريب.
- مهم تقول: دي علامة تحذير مش نتيجة، لأن المجموعتين والميزانيات مختلفة.
- التشخيص: الوصفة (recipe) مش bug: مفيش server optimizer، الـ optimizer عند العميل بيبدأ من الصفر كل زيارة (حوالي 4 خطوات بس)، جدول الـ learning rate بتاع السحابة اتطبق على الجولات فنص الميزانية ضاع، الـ learning rate صغير، والجداول بتبدأ عشوائي.
- الحل: سلم تحسين مسجل مسبقاً، كل مرحلة بتغير حاجة واحدة، ولازم تكسب ≥ 0.002 علشان نحتفظ بيها.
سؤال محتمل: "مش ده تحيز إنكم بتضبطوا الفيدرالي أكتر؟" — الضبط على calibration users بس، ومسجل قبل القراءة، والنقطة الأصلية فاضلة في الرسالة كمرجع.`);
}

// 13. Ladder staircase
{
  const s = light();
  header(s, "Federated improvement ladder", "Stages done so far: +22.7% over the registered start");
  s.addChart(pres.charts.BAR, [{ name: "Calibration MRR@20", labels: ["S0  Registered FedAvg", "S1a  Client LR ×2", "S1b  Client LR ×8"], values: [0.023473, 0.025562, 0.028796] }], {
    x: MX, y: 1.5, w: 7.4, h: 4.75, barDir: "col", chartColors: ["9FB7BA", "5E9DA1", C.teal], varyColors: true, barGapWidthPct: 40,
    showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0.0000", dataLabelFontSize: 15, dataLabelColor: C.ink,
    valAxisMinVal: 0, valAxisMaxVal: 0.032, valAxisLabelFormatCode: "0.000", valAxisLabelColor: C.muted, catAxisLabelColor: C.body, catAxisLabelFontSize: 13,
    valGridLine: { color: "E3E8EA", size: 0.5 }, catGridLine: { style: "none" }, showLegend: false,
    showTitle: true, title: "Calibration users, validation MRR@20 (seed 2026, full training budget)", titleFontSize: 13, titleColor: C.ink,
  });
  const x2 = 8.4, w2 = W - MX - x2;
  const steps = [["+8.9%", "S1a vs S0 — client learning rate grid {0.5, 1, 2}; winner ×2 at the edge → grid extended"], ["+12.7%", "S1b vs S1a — extended grid {4, 8, 16}; ×8 wins inside the grid (×16 falls)"], ["+22.7%", "S1b vs the registered start"]];
  steps.forEach((t, i) => {
    const y = 1.6 + i * 1.45;
    card(s, x2, y, w2, 1.25, i === 2 ? C.tealLight : C.panel);
    txt(s, t[0], x2 + 0.25, y + 0.15, 1.6, 0.6, { fontFace: HF, fontSize: 26, bold: true, color: C.tealDark });
    txt(s, t[1], x2 + 1.85, y + 0.15, w2 - 2.05, 1.0, { fontSize: 13, valign: "middle" });
  });
  txt(s, "Next stages are shown on the following slide. Main-cohort values of ladder rows are sealed until the final analysis.", x2, 6.0, w2, 0.55, { fontSize: 12, italic: true, color: C.muted });
  footer(s, "Improvement-ladder register (registered calibration screen values); learning-rate selection files");
  s.addNotes(`الشريحة 13 — سلم التحسين: المراحل اللي خلصت.
- S0: FedAvg زي ما هو مسجل = 0.023473.
- S1a: شبكة learning rate للعميل {0.5، 1، 2}: الفائز ×2 = 0.025562 (+8.9%). بس الفائز كان على حافة الشبكة والمنحنى لسه طالع، فمدينا الشبكة.
- S1b: {4، 8، 16}: ×8 = 0.028796 وده جوه الشبكة (×16 نزل لـ 0.022909)، فما احتجناش ×32. التحسن +12.7% عن S1a و +22.7% عن البداية.
- كل القيم دي على calibration users، ومسموح نقراها لأنها مسجلة كـ screen.
- قيم الـ main cohort لصفوف السلم مقفولة لحد التحليل النهائي.
سؤال محتمل: "لسه بعيد عن السحابة؟" — أيوه، المراحل الجاية (server optimizer، شكل الـ LR، التدريب المسبق) متوقع إنها تعمل الفرق الأكبر (تقدير من البحث).`);
}

// 14. Ladder running & rules
{
  const s = light();
  header(s, "Federated improvement ladder", "Next stages: strictly one at a time, with a stop rule");
  const st = [
    ["S2", "Server optimizer", "FedAdam and FedAvg with momentum (incl. small client-LR multipliers)", "Running — 7 screens; ≈ 30 Sep (estimate)", C.teal, C.white],
    ["S3", "LR shape × client optimizer", "Constant rate over rounds; client SGD-momentum vs Adam", "Queued after S2", C.slateLight, C.ink],
    ["S4", "Local computation", "Local passes 1 / 4, batch 8 — one at a time, same exposure", "Queued after S3", C.slateLight, C.ink],
    ["S5", "Longer budget", "Twice the training budget (descriptive budget test)", "After tuning", C.slateLight, C.ink],
    ["S6", "Personalisation", "Personal vector; federated model fine-tuned on device", "Registered arms", C.slateLight, C.ink],
    ["S7", "Start from a pretrained model", "Pretrained on 1.22M opted-in users, then federated", "Pretraining running", C.amber, C.ink],
    ["S8", "Freeze the item tables", "Only the encoder is trained and uploaded (≈ 1 MB each way)", "After S7", C.slateLight, C.ink],
  ];
  const rh = 0.6;
  st.forEach((r, i) => {
    const y = 1.55 + i * (rh + 0.08);
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: MX, y, w: 0.8, h: rh, rectRadius: 0.08, fill: { color: C.ink }, line: { type: "none" } });
    s.addText(r[0], { x: MX, y, w: 0.8, h: rh, fontFace: HF, fontSize: 16, bold: true, color: C.white, align: "center", valign: "middle", margin: 0, isTextBox: true });
    txt(s, r[1], MX + 1.0, y, 2.8, rh, { fontSize: 14.5, bold: true, color: C.ink, valign: "middle" });
    txt(s, r[2], MX + 3.9, y, 3.7, rh, { fontSize: 12.5, valign: "middle" });
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: MX + 7.75, y: y + 0.1, w: 2.0, h: rh - 0.2, rectRadius: 0.2, fill: { color: r[4] }, line: { type: "none" } });
    s.addText(r[3], { x: MX + 7.75, y: y + 0.1, w: 2.0, h: rh - 0.2, fontFace: BF, fontSize: 10.5, bold: true, color: r[5], align: "center", valign: "middle", margin: 0, isTextBox: true });
  });
  const x2 = MX + 10.0, w2 = W - MX - x2;
  card(s, x2, 1.55, w2, 4.7, C.panel);
  txt(s, "Rules", x2 + 0.2, 1.7, w2 - 0.4, 0.4, { fontSize: 16, bold: true, fontFace: HF, color: C.ink });
  txt(s, bullets(["Each stage builds on the previous winner; no going back.", "A challenger must gain ≥ 0.002.", "Ties within 0.0005 → the simpler option.", "No stage beyond 170 MB per visit or with stateful clients."]), x2 + 0.2, 2.15, w2 - 0.4, 4.0, { fontSize: 12.5 });
  footer(s, "Ladder register; ladder amendments (stages S2–S4, S7–S8); decision log 29 Sep (S2 launch 14:44Z, extra runs 17:56Z)");
  s.addNotes(`الشريحة 14 — المراحل الجاية والقواعد.
- S2 (شغالة دلوقتي): server optimizer = FedAdam و FedAvg مع momentum. 5 تشغيلات بدأوا الساعة 14:44 UTC، و2 اتضافوا 17:56 بعد المراجعة (FedAvgM عند multipliers صغيرة). متوقع تخلص حوالي 30 سبتمبر (تقدير).
- S3: شكل الـ learning rate ثابت على الجولات × optimizer العميل (SGD-momentum مقابل Adam).
- S4: عدد الـ passes المحلية وحجم الـ batch، واحدة واحدة بنفس التعرض.
- S5: ضعف ميزانية التدريب (وصفي).
- S6: التخصيص.
- S7: نبدأ من موديل مُدرب مسبقاً على 1.22 مليون مستخدم موافق (التدريب المسبق شغال دلوقتي).
- S8: نجمد جداول المنتجات ونرفع الـ encoder بس (حوالي 1 ميجا في كل اتجاه).
القواعد: كل مرحلة على الفائز اللي قبلها، لازم تكسب ≥ 0.002، التعادل ≤ 0.0005 نختار الأبسط، ومفيش أي حاجة فوق 170 ميجا أو عملاء بيحتفظوا بحالة.
تقديرات البحث (مش نتائج): S2 ممكن توصل 0.045–0.065، S3 حوالي 0.055–0.075 على calibration. لو اتسألت قولها كتقدير بس.`);
}

// 15. Discussion 3
discussion(3, "How much effort should federated learning get?", [
  ["Is the staged ladder with a 0.002 stop rule a fair amount of effort for FL?", "Or should we stop after the server-optimizer stage and spend the time on the deployment branch?"],
  ["If tuned FL still falls below “last viewed item”, how should we present it?", "Our plan: report it plainly as a finding about this recipe at this budget, with the ladder showing what was tried."],
  ["Is 165.7 MB per visit acceptable for a desktop browser?", "Or should the 8-bit and frozen-table variants be the main federated rows?"],
], `ديسكشن 3 — الفيدرالي ياخد قد إيه مجهود؟
1) السلم بقاعدة 0.002 عادل؟ ولا نقف بعد مرحلة الـ server optimizer ونركز على الفرع المُدرب مسبقاً؟
2) لو الفيدرالي بعد الضبط لسه تحت "آخر منتج اتشاف"، نعرضه إزاي؟ خطتنا: نقوله بوضوح كنتيجة عن الوصفة دي والميزانية دي، والسلم بيوضح إحنا جربنا إيه.
3) 165.7 ميجا للزيارة مقبولة لمتصفح ديسكتوب؟ ولا نخلي نسخ 8 بت والجداول المجمدة هي الصفوف الأساسية؟`, "ladder stages S0–S8; stop rule gain ≥ 0.002");

// 16. Privacy
{
  const s = light();
  header(s, "Privacy", "What the guarantee covers — and what it does not");
  const dp = [["ε = 8", "noise multiplier 0.5642"], ["ε = 1", "noise multiplier 1.3094"], ["ε = 4", "optional, if time allows"]];
  dp.forEach((d, i) => {
    const x = MX + i * 2.05;
    card(s, x, 1.6, 1.85, 1.5, i === 2 ? C.panel : C.tealLight);
    txt(s, d[0], x + 0.2, 1.72, 1.5, 0.6, { fontFace: HF, fontSize: 28, bold: true, color: i === 2 ? C.muted : C.tealDark });
    txt(s, d[1], x + 0.2, 2.4, 1.5, 0.6, { fontSize: 12 });
  });
  txt(s, bullets([
    "Client-level differential privacy, δ = 5 × 10⁻⁶: adding or removing one user's whole history barely changes the released model.",
    "Central DP with a trusted aggregator; Poisson client sampling; Rényi accounting (independently reproduced).",
    "Not accounted: the federated tuning on calibration users, and the public starting point (catalogue, pretrained weights).",
  ]), MX, 3.35, 6.0, 3.0, { fontSize: 14 });
  const x2 = 6.95, w2 = W - MX - x2;
  card(s, x2, 1.6, w2, 1.55, C.panel);
  txt(s, "Threat model", x2 + 0.3, 1.72, w2 - 0.6, 0.4, { fontSize: 16, bold: true, fontFace: HF, color: C.ink });
  txt(s, "Adversary = an honest-but-curious browser vendor (the server). The retailer's own logs are out of scope.", x2 + 0.3, 2.1, w2 - 0.6, 1.0, { fontSize: 14 });
  card(s, x2, 3.35, w2, 2.6, C.amberLight);
  txt(s, "What updates can reveal", x2 + 0.3, 3.5, w2 - 0.6, 0.4, { fontSize: 16, bold: true, fontFace: HF, color: C.amberDark });
  txt(s, bullets([
    "Without DP, an update shows which products the user viewed (the item-table rows that moved).",
    "Secure aggregation is not simulated — stated as a limitation.",
    "Frozen item tables remove most, not all, of this leak.",
  ]), x2 + 0.3, 3.95, w2 - 0.6, 1.9, { fontSize: 14 });
  footer(s, "Privacy statement and threat model (registered after the review); independent correctness review of the DP accountant");
  s.addNotes(`الشريحة 16 — الخصوصية.
- مستويين DP أساسيين: ε=8 (noise multiplier 0.5642) و ε=1 (1.3094). ε=4 اختياري لو في وقت (0.7244).
- الضمان على مستوى العميل (المستخدم كله)، δ = 5×10⁻⁶. معناه: إضافة أو شيل تاريخ مستخدم كامل مش هيغير الموديل النهائي تقريباً.
- DP مركزي مع aggregator موثوق، اختيار عملاء Poisson، وحساب Rényi DP؛ مراجعة مستقلة أعادت حساب القيم ووصلت لنفس النتيجة بدقة كبيرة جداً.
- مش محسوب في الـ ε: ضبط الفيدرالي على calibration users، ونقطة البداية العامة (الكتالوج، الأوزان المُدربة مسبقاً).
- نموذج التهديد: الخصم هو شركة المتصفح نفسها بس "أمينة وفضولية". سجلات المتجر نفسه خارج النطاق.
- بصراحة: من غير DP، التحديث بيكشف المنتجات اللي المستخدم شافها، و secure aggregation مش متحاكي. الجداول المجمدة بتقلل التسريب ده بس مش بتلغيه.
سؤال محتمل: "ε=8 ضعيف؟" — شائع في الإنتاج لـ user-level DP، و ε=1 قوي؛ السؤال للمشرف هل محتاج منحنى.`);
}

// 17. Realistic deployment branch
{
  const s = light();
  header(s, "Realistic deployment branch", "How a browser would actually ship it");
  const steps = [
    ["1", "Pretrain on opted-in users", "1,217,610 users who share data (24.1M decisions); none of them is in the main cohort", C.amberLight, C.amberDark],
    ["2", "Federated on private users", "Start from the pretrained model; private users send only updates", C.tealLight, C.tealDark],
    ["3", "Frozen item tables", "Only the small encoder moves: 265,162 parameters ≈ 1.06 MB each way; tables downloaded once", C.panel, C.ink],
  ];
  const bw = 3.55, gap = 0.62;
  steps.forEach((t, i) => {
    const x = MX + i * (bw + gap), y = 1.7;
    card(s, x, y, bw, 2.15, t[3]);
    s.addShape(pres.shapes.OVAL, { x: x + 0.25, y: y + 0.25, w: 0.5, h: 0.5, fill: { color: t[4] }, line: { type: "none" } });
    s.addText(t[0], { x: x + 0.25, y: y + 0.25, w: 0.5, h: 0.5, fontFace: HF, fontSize: 18, bold: true, color: C.white, align: "center", valign: "middle", margin: 0, isTextBox: true });
    txt(s, t[1], x + 0.95, y + 0.25, bw - 1.1, 0.6, { fontSize: 17, bold: true, fontFace: HF, color: C.ink, valign: "middle" });
    txt(s, t[2], x + 0.25, y + 1.0, bw - 0.5, 1.5, { fontSize: 14 });
    if (i < 2) s.addShape(pres.shapes.RIGHT_ARROW, { x: x + bw + 0.1, y: y + 1.05, w: gap - 0.2, h: 0.45, fill: { color: C.gray }, line: { type: "none" } });
  });
  card(s, MX, 4.25, W - 2 * MX, 1.85, C.white, { line: C.line });
  txt(s, "How it is judged (pre-registered, secondary confirmatory)", MX + 0.3, 4.4, 8, 0.4, { fontSize: 16, bold: true, color: C.ink });
  txt(s, bullets([
    "Against the full-pool cloud (same opted-in + main users): retention of the pretrained model alone, and of pretrained + federated.",
    "Value of federated training on private users = (pretrained + federated) − (pretrained alone); frozen vs unfrozen tables with the bytes ratio.",
    "Status: pretraining running since 29 Sep 13:03 UTC; results sealed until the first validation read.",
  ]), MX + 0.3, 4.85, W - 2 * MX - 0.6, 1.2, { fontSize: 14.5 });
  footer(s, "Warm-start amendment and its review; pretraining cohort build (decision log 29 Sep); multi-expert review §3.2");
  s.addNotes(`الشريحة 17 — الفرع الواقعي للنشر (أولوية عندي).
الفكرة: ده اللي المتصفحات بتعمله فعلاً:
1) ندرب موديل مسبقاً على مستخدمين موافقين يشاركوا الداتا: 1,217,610 مستخدم (24.1 مليون قرار)، ومفيش حد منهم في الـ main cohort.
2) نبدأ الفيدرالي من الموديل ده على المستخدمين الخاصين؛ هما بيبعتوا تحديثات بس.
3) نجمد جداول المنتجات، فبنرفع الـ encoder الصغير بس: 265,162 باراميتر ≈ 1.06 ميجا في كل اتجاه، والجداول بتنزل مرة واحدة (81.78 ميجا).
التقييم مسجل مسبقاً كـ secondary confirmatory: نقارن بالسحابة الكاملة (اللي شافت نفس المستخدمين الموافقين + الـ main)، ونحسب قيمة الفيدرالي = (مسبق + فيدرالي) − (مسبق لوحده)، ونقارن المجمد بغير المجمد مع نسبة البايتس.
الحالة: التدريب المسبق شغال من 29 سبتمبر 13:03 UTC. النتائج مقفولة.
سؤال محتمل: "ليه مش بنبدأ من موديل السحابة الكبير؟" — لأنه شاف مستخدمين الـ main، فده هيبقى غش؛ ممنوع بالتسجيل.`);
}

// 18. Discussion 4
discussion(4, "Privacy level and the deployment story", [
  ["Is DP at ε = 8 and ε = 1 enough, or do we need a curve?", "Adding ε = 4 needs one more registered run (~half a day, estimate). A full curve would need several."],
  ["Is “honest-but-curious browser vendor” the right adversary for Edge?", "Should secure aggregation be simulated, or is stating it as a limitation acceptable?"],
  ["Is pretraining on opted-in users a realistic assumption for Edge?", "If yes, we would lead the Edge answer with the pretrained → federated → frozen-tables branch."],
], `ديسكشن 4 — مستوى الخصوصية وقصة النشر.
1) DP عند ε=8 و ε=1 كفاية؟ ولا محتاج منحنى؟ ε=4 محتاج تشغيل مسجل واحد زيادة (حوالي نص يوم، تقدير).
2) الخصم "شركة المتصفح الأمينة الفضولية" مناسب لـ Edge؟ ولا لازم نحاكي secure aggregation؟
3) هل التدريب المسبق على مستخدمين موافقين افتراض واقعي لـ Edge؟ لو أيوه، هنخلي الفرع ده هو إجابة Edge الرئيسية.`, "ε = 8 and ε = 1 registered; ε = 4 optional");

// 19. On-device
{
  const s = light();
  header(s, "On-device", "What runs entirely on the user's device");
  const arms = [
    ["Local-only model", "Trained from scratch on the user's own 5–100 events. A control, not a product."],
    ["Cloud model, fine-tuned on device", "Download once, adapt locally; nothing uploaded after the download."],
    ["Federated model, fine-tuned on device", "Adds local adaptation on top of the federated model."],
    ["Pretrained model, fine-tuned on device", "The cleanest “download-only” option: its base never saw private users."],
  ];
  arms.forEach((a, i) => {
    const y = 1.6 + i * 1.18;
    card(s, MX, y, 6.2, 1.02, i === 3 ? C.amberLight : C.panel);
    txt(s, a[0], MX + 0.3, y + 0.12, 5.6, 0.35, { fontSize: 15.5, bold: true, color: C.ink });
    txt(s, a[1], MX + 0.3, y + 0.5, 5.6, 0.45, { fontSize: 13 });
  });
  const x2 = 7.2, w2 = W - MX - x2;
  txt(s, "Device realism checks", x2, 1.6, w2, 0.4, { fontSize: 17, bold: true, fontFace: HF, color: C.ink });
  const chk = [["Export", "Device model exported to ONNX, FP32 and INT8"], ["Size", "82.8 MB FP32; ≈ 21 MB INT8 (estimate)"], ["Latency", "Single-thread CPU latency and memory — running"], ["Browser", "ONNX Runtime Web in Edge on the owner's laptop — pending"]];
  chk.forEach((c, i) => {
    const y = 2.1 + i * 0.95;
    chip(s, c[0], x2, y + 0.05, 1.2, i < 2 ? C.teal : C.slateLight, i < 2 ? C.white : C.ink);
    txt(s, c[1], x2 + 1.4, y, w2 - 1.4, 0.8, { fontSize: 14 });
  });
  txt(s, "Fine-tuning settings are chosen on a separate panel of unseen users, never on the reported users. Device runs started 29 Sep 16:20 UTC.", x2, 5.95, w2, 0.6, { fontSize: 12, italic: true, color: C.muted });
  footer(s, "Study description “What is compared”; privacy matrix revision; realism audit C1; decision log 29 Sep (device lanes)");
  s.addNotes(`الشريحة 19 — على الجهاز فقط.
- موديل محلي بس: بيتدرب من الصفر على 5–100 حدث بتوع المستخدم. ده control مش منتج، ومتوقع يبقى ضعيف.
- موديل السحابة بيتظبط على الجهاز: تحميل مرة وبعدين تكيف محلي.
- الموديل الفيدرالي بيتظبط على الجهاز.
- الموديل المُدرب مسبقاً بيتظبط على الجهاز: ده أنضف خيار "تحميل بس" لأن أساسه ما شافش المستخدمين الخاصين.
- واقعية الجهاز: تصدير ONNX FP32 و INT8؛ الحجم 82.8 ميجا FP32 وحوالي 21 ميجا INT8 (تقدير)؛ السرعة والذاكرة على CPU بثريد واحد (شغال)؛ واختبار في المتصفح بـ ONNX Runtime Web على لابتوبي (لسه).
- إعدادات الـ fine-tuning بتتختار على مجموعة مستخدمين منفصلة مش هي اللي بنعرض عليها.
سؤال محتمل: "التدريب في المتصفح نفسه؟" — حزم التدريب في ONNX Runtime متوقفة، فالمتصفح بيعمل inference بس؛ التدريب المحلي بنحاكيه.`);
}

// 20. Cold start & segments
{
  const s = light();
  header(s, "Cold start and user segments", "Who benefits, and how much history is needed");
  const p = [["12,862", "unseen users (never participated) — does the model serve non-participants?"], ["381,295", "truly cold users (no training events at all)"], ["16,384 × 2", "sparse-history and heavy-history panels"]];
  p.forEach((t, i) => {
    const x = MX + i * 4.1;
    card(s, x, 1.6, 3.85, 1.55, C.panel);
    txt(s, t[0], x + 0.25, 1.72, 3.4, 0.6, { fontFace: HF, fontSize: 28, bold: true, color: C.teal });
    txt(s, t[1], x + 0.25, 2.35, 3.4, 0.75, { fontSize: 13 });
  });
  txt(s, "Analyses", MX, 3.45, 6, 0.4, { fontSize: 17, bold: true, fontFace: HF, color: C.ink });
  txt(s, bullets([
    "Quality by history length: 0, 5–10, 11–20, 21–50, 51–100 training events.",
    "Personas: 3 × 3 grid of category diversity × brand loyalty; activity quartiles.",
    "Controlled truncation study: cut each user's history to 0, 1, 5, 10, 20, 50 events, with a negative control that must show no change.",
    "“Events to parity” is claimed only if parity holds at the next history step too.",
  ]), MX, 3.9, 7.4, 2.6, { fontSize: 14.5 });
  card(s, 8.4, 3.45, W - MX - 8.4, 2.1, C.amberLight);
  txt(s, "Honest limit", 8.65, 3.6, W - MX - 8.9, 0.4, { fontSize: 16, bold: true, fontFace: HF, color: C.amberDark });
  txt(s, "Grouping by history length is observational: users with long histories differ in other ways. The truncation study is the controlled answer.", 8.65, 4.05, W - MX - 8.9, 2.2, { fontSize: 14 });
  footer(s, "Realism audit D2; project map §3; cold-start registration (after the review)");
  s.addNotes(`الشريحة 20 — البداية الباردة (cold start) والشرائح.
- 12,862 مستخدم ما شاركوش في التدريب خالص: هل الموديل بيخدمهم كويس؟ (فجوة المشاركة).
- 381,295 مستخدم بارد تماماً: صفر أحداث تدريب.
- لوحتين 16,384 مستخدم: تاريخ قليل وتاريخ كتير.
- التحليلات: الجودة حسب طول التاريخ، شخصيات 3×3 (تنوع الفئات × الولاء للبراند)، ودراسة قطع التاريخ لـ 0/1/5/10/20/50 حدث مع negative control لازم ما يتغيرش.
- "عدد الأحداث للوصول للتكافؤ" بنقوله بس لو التكافؤ ثابت في الخطوة اللي بعدها كمان.
- حدود صريحة: التقسيم حسب طول التاريخ ملاحظة (observational) مش تجربة؛ دراسة القطع هي الإجابة المتحكم فيها.
سؤال محتمل: "كام حدث محتاج المستخدم الجديد؟" — لسه مقفول؛ هيطلع من دراسة القطع.`);
}

// 21. Reviews
{
  const s = light();
  header(s, "Independent reviews", "14-part expert review: 7/10 now, 8–8.5 after the fixes");
  const dims = [["Data", 8], ["Evaluation & statistics", 8], ["Features & history", 7], ["Backbone contest", 7], ["Central references", 7], ["Differential privacy", 7], ["Engineering", 7], ["Question & Edge framing", 6.5], ["Federated training", 6], ["Communication", 6], ["On-device arms", 6], ["Cold start & strata", 6], ["Thesis write-up", 6], ["Device measurement", 5]];
  s.addChart(pres.charts.BAR, [{ name: "Score / 10", labels: dims.map(d => d[0]), values: dims.map(d => d[1]) }], {
    x: MX, y: 1.5, w: 5.9, h: 5.0, barDir: "bar", chartColors: [C.teal], barGapWidthPct: 35,
    showValue: true, dataLabelPosition: "outEnd", dataLabelFontSize: 11, dataLabelColor: C.ink, dataLabelFormatCode: "0.0",
    valAxisMinVal: 0, valAxisMaxVal: 10, valAxisHidden: true, valGridLine: { style: "none" }, catGridLine: { style: "none" },
    catAxisLabelColor: C.body, catAxisLabelFontSize: 11, catAxisOrientation: "maxMin", showLegend: false,
    showTitle: true, title: "Part scores (mean 6.96)", titleFontSize: 12, titleColor: C.ink,
  });
  const cols = [
    ["Strengths", C.tealLight, ["Pre-registration and a single gated test look", "Exact full-catalogue evaluation; no bug found", "Leak-free chronological data protocol", "DP accounting independently reproduced"]],
    ["Weaknesses", C.amberLight, ["Federated recipe too weak (pre-2019 practice)", "Privacy claims over-stated; no threat model", "Device evidence from a server CPU"]],
    ["Fixed or registered", C.panel, ["Ladder extended; server optimizers added", "Privacy matrix, threat model, “last item” anchor", "Deployment branch promoted; code blockers fixed"]],
  ];
  const x2 = 6.8, w2 = W - MX - x2;
  let y = 1.55;
  cols.forEach(c => {
    const hh = 0.45 + c[2].length * 0.33 + 0.15;
    card(s, x2, y, w2, hh, c[1]);
    txt(s, c[0], x2 + 0.25, y + 0.1, w2 - 0.5, 0.35, { fontSize: 14.5, bold: true, color: C.ink });
    txt(s, bullets(c[2], { paraSpaceAfter: 2 }), x2 + 0.25, y + 0.45, w2 - 0.5, c[2].length * 0.33 + 0.1, { fontSize: 12.5 });
    y += hh + 0.15;
  });
  footer(s, "Multi-expert review synthesis (§1 scorecard, §4 gates, §5 strengths); decision log 29 Sep 17:59–18:36Z");
  s.addNotes(`الشريحة 21 — المراجعات المستقلة.
مراجعة من 14 جزء، كل جزء خبير. الدرجة 7/10 (متوسط 6.96). لو خلصنا قائمة الـ MUST قبل أول قراءة validation، الهدف 8–8.5.
أعلى الدرجات: الداتا 8، التقييم والإحصاء 8. أقلها: قياس الجهاز 5.
نقاط القوة: الـ pre-registration ونظرة test واحدة، تقييم مضبوط على الكتالوج كله من غير bugs، بروتوكول داتا من غير تسريب زمني، وحساب DP اتأكد بشكل مستقل.
نقاط الضعف: وصفة الفيدرالي ضعيفة (ممارسات قبل 2019)، ادعاءات الخصوصية كانت مبالغ فيها ومفيش threat model، وأدلة الجهاز من CPU سيرفر.
اللي اتصلح/اتسجل: مدينا السلم وضفنا server optimizers، سجلنا مصفوفة الخصوصية و threat model وربطنا كل رقم بقاعدة "آخر منتج"، رقّينا الفرع الواقعي، وصلحنا عوائق الكود (إعادة تقييم السحابة الكاملة والتدريب المسبق، تصدير الصفوف للـ ONNX، إدخال الجداول المجمدة في التحليل).
لسه: بنود قبل الـ test (زي إفصاح عن حدود الـ test) وبنود الكتابة (بيان استخدام الذكاء الاصطناعي في Queen's محتاج موافقة المشرف).`);
}

// 22. Timeline
{
  const s = light();
  header(s, "Timeline", "What is running now and when results are expected");
  const days = ["27 Sep", "28", "29", "30", "1 Oct", "2", "3", "4", "5", "6", "7", "8"];
  const gx = MX + 3.4, gw = W - MX - gx, dw = gw / days.length, gy = 1.55;
  days.forEach((d, i) => {
    txt(s, d, gx + i * dw, gy, dw, 0.3, { fontSize: 11, color: C.muted, align: "center" });
    if (i % 2 === 0) s.addShape(pres.shapes.RECTANGLE, { x: gx + i * dw, y: gy + 0.35, w: dw, h: 4.75, fill: { color: "F6F8F8" }, line: { type: "none" } });
  });
  const rows = [
    ["Backbone contest", 0.0, 2.0, "done", C.slate],
    ["Ladder S0–S1b screens", 2.0, 2.65, "done", C.slate],
    ["Cloud models (full pool, 2 seeds)", 2.35, 3.6, "running", C.teal],
    ["Pretraining on opted-in users", 2.55, 3.6, "running", C.teal],
    ["Ladder S2 → S3 → S4", 2.6, 3.95, "running", C.teal],
    ["Main federated + DP runs", 3.9, 5.6, "estimate", C.gray],
    ["Device arms", 2.7, 5.4, "running", C.teal],
    ["Analysis freeze → first validation read", 5.4, 6.8, "estimate", C.amber],
    ["Lock → single test look", 6.8, 8.6, "estimate", C.amber],
    ["Final tables", 8.6, 10.2, "estimate", C.amber],
  ];
  const rh = 0.42;
  rows.forEach((r, i) => {
    const y = gy + 0.45 + i * (rh + 0.04);
    txt(s, r[0], MX, y, 3.3, rh, { fontSize: 12.5, color: C.ink, valign: "middle" });
    const bx = gx + r[1] * dw, bw = (r[2] - r[1]) * dw;
    const est = r[3] === "estimate";
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: bx, y: y + 0.08, w: bw, h: rh - 0.16, rectRadius: 0.08, fill: { color: r[4], transparency: est ? 35 : 0 }, line: est ? { color: C.amberDark, width: 1, dashType: "dash" } : { type: "none" } });
    const inside = bw >= 0.6;
    s.addText(r[3], { x: inside ? bx : bx + bw + 0.06, y: y + 0.06, w: inside ? bw : 0.9, h: rh - 0.12, fontFace: BF, fontSize: 9, bold: true, color: !inside || r[4] === C.gray || r[4] === C.amber ? C.ink : C.white, align: inside ? "center" : "left", valign: "middle", margin: 0, isTextBox: true });
  });
  // today marker (30 Sep = index 3)
  const tx = gx + 3.5 * dw;
  s.addShape(pres.shapes.LINE, { x: tx, y: gy + 0.35, w: 0, h: 4.75, line: { color: C.amberDark, width: 1.5, dashType: "dash" } });
  txt(s, "meeting 30 Sep", tx + 0.06, gy + 0.38, 1.4, 0.25, { fontSize: 10, color: C.amberDark, bold: true });
  txt(s, "All future dates are estimates from measured run rates; the plan's 1–2 Oct first read has slipped about a day because the tuning screens run slower when sharing GPUs.", MX, 6.62, W - 2 * MX, 0.4, { fontSize: 11.5, italic: true, color: C.muted });
  footer(s, "Master plan v5 (addendum B cost model); ladder research §6 schedule; decision log 29 Sep run status");
  s.addNotes(`الشريحة 22 — الجدول الزمني.
خلص: مسابقة الموديل (27–28 سبتمبر)، ومراحل السلم S0–S1b (29 سبتمبر).
شغال دلوقتي: موديلين السحابة الكاملة (بذرتين) — كانوا عند 2.0 من 6.0 الساعة 16:05 UTC، متوقع يخلصوا حوالي 30 سبتمبر (تقدير). التدريب المسبق (بدأ 13:03 UTC). مراحل السلم S2 ثم S3 ثم S4 بالترتيب. تشغيلات الجهاز بدأت 16:20 UTC.
بعد S4: التشغيلات الفيدرالية الرئيسية و DP (كل تشغيل حوالي 10.7 ساعة، مقاسة) — تقريباً 30 سبتمبر لـ 2 أكتوبر.
التقديرات: أول قراءة validation حوالي 2–3 أكتوبر؛ القفل ونظرة الـ test الواحدة حوالي 4–5 أكتوبر؛ الجداول النهائية حوالي 6–7 أكتوبر.
قول بوضوح: دي تقديرات، الخطة كانت 1–2 أكتوبر للقراءة الأولى واتأخرت حوالي يوم لأن تشغيلات الضبط أبطأ لما بتشارك الـ GPU.
سؤال محتمل: "إمتى الرسالة؟" — محتاجين نعرف ميعاد التسليم الرسمي علشان نحدد نقطة القطع.`);
}

// 23. Discussion 5
discussion(5, "Priorities if time is short", [
  ["If we must cut, what goes first?", "Our pre-registered cut order starts with: extra seeds on panels, the 8-bit second seed, ε = 1, sparse/heavy panels. Never cut: federated ×2, local-only ×2, cloud fine-tune ×2, INT8, the lock, the test look."],
  ["Is a server-CPU benchmark plus one browser run enough device evidence?", "Or should we measure on a second machine or a phone-class device?"],
  ["For cold start, which answer matters more to you?", "Observational (by history length) for all methods, or the controlled truncation study for a few methods."],
  ["Can you pre-approve the Queen's AI-use disclosure, and accept an emailed hash manifest as an external timestamp?", "Both are front-matter / integrity items the review asked for."],
], `ديسكشن 5 — الأولويات لو الوقت ضيق.
1) لو لازم نقطع، نقطع إيه الأول؟ عندنا ترتيب قطع مسجل: البذرة التانية للوحات، البذرة التانية للـ 8 بت، ε=1، لوحات التاريخ القليل والكتير... وحاجات مستحيل نقطعها: الفيدرالي ببذرتين، المحلي ببذرتين، fine-tune السحابة ببذرتين، INT8، القفل، ونظرة الـ test.
2) هل CPU السيرفر + تشغيل واحد في المتصفح كفاية كدليل للجهاز؟
3) البداية الباردة: الأهم عنده التحليل الملاحظ لكل الطرق ولا دراسة القطع المتحكم فيها لطرق قليلة؟
4) هل يوافق مسبقاً على بيان استخدام الذكاء الاصطناعي المطلوب من Queen's؟ وهل يقبل إننا نبعتله بالإيميل ملف hashes كطابع زمني خارجي؟`, "cut order is pre-registered in the master plan §13");

// 24. Risks
{
  const s = light();
  header(s, "Risks", "Main risks and how each is handled");
  const rk = [
    ["Federated looks like a strawman", "Pre-registered tuning ladder; every stage recorded; “last item” floor reported"],
    ["“Federated is not private”", "Threat model, privacy matrix, DP arms, frozen-table variant; secure aggregation stated as not simulated"],
    ["Retention against which cloud?", "Both matched and full-pool references reported, the full-pool recipe's upward bias disclosed"],
    ["Many protocol changes", "Every change is a dated, hash-sealed amendment registered before the read it affects"],
    ["Device claims from a server CPU", "Labelled a proxy; browser run on a real laptop; INT8 settings registered exactly"],
    ["Test week: Black Friday, seen in an earlier phase", "Disclosed; Black-Friday sub-period reported; one look only"],
    ["Time", "Pre-registered cut order; anything cut is listed as “registered, not run”"],
  ];
  const hdr = ["Risk", "Mitigation"].map(t => ({ text: t, options: { bold: true, color: C.white, fill: { color: C.ink }, fontSize: 13 } }));
  const body = rk.map((r, i) => [{ text: r[0], options: { bold: true, color: C.ink, fill: { color: i % 2 ? C.white : C.panel }, fontSize: 13 } }, { text: r[1], options: { color: C.body, fill: { color: i % 2 ? C.white : C.panel }, fontSize: 13 } }]);
  s.addTable([hdr, ...body], { x: MX, y: 1.55, w: W - 2 * MX, colW: [4.0, 8.13], fontFace: BF, border: { type: "solid", pt: 0.5, color: C.line }, margin: [5, 8, 5, 8], rowH: 0.56 });
  footer(s, "Multi-expert review §6 (defence risks); thesis research review §F; master plan §13");
  s.addNotes(`الشريحة 24 — المخاطر.
1) الفيدرالي يبان ضعيف عمداً: الحل السلم المسجل، وكل مرحلة متسجلة، وأرضية "آخر منتج".
2) "الفيدرالي مش خاص": threat model، مصفوفة الخصوصية، DP، الجداول المجمدة، وبنقول secure aggregation مش متحاكي.
3) retention قدام أي سحابة: بنعرض الاتنين، وبنفصح إن وصفة السحابة الكاملة ما اتظبطتش للداتا الأكبر.
4) تغييرات كتير في البروتوكول: كل تغيير amendment بتاريخ و hash قبل القراءة.
5) الجهاز من CPU سيرفر: مكتوب proxy، وفي تشغيل على لابتوب حقيقي.
6) أسبوع الـ test فيه Black Friday واتشاف في مرحلة سابقة: مفصح عنه ونظرة واحدة.
7) الوقت: ترتيب قطع مسجل.`);
}

// 25. Summary & next steps
{
  const s = dark();
  header(s, "Summary", "Where the project stands", true);
  const cols = [
    ["Done", C.slate, ["Frozen data protocol; full-catalogue evaluator", "Backbone contest: SASRec by the registered rule (caveat disclosed)", "Federated ladder S0 → S1b: +22.7%", "Independent reviews; fixes registered before any read"]],
    ["Running", C.teal, ["Full-pool cloud models (2 seeds)", "Pretraining on opted-in users", "Ladder stage S2 (server optimizers)", "Device arms (local-only, fine-tuning)"]],
    ["Next", C.amber, ["S3 → S4 → main federated + DP runs", "Analysis freeze → one validation read (≈ 2–3 Oct, est.)", "Lock → single test look (≈ 4–5 Oct, est.)", "Browser benchmark; thesis tables (≈ 6–7 Oct, est.)"]],
  ];
  const cw = (W - 2 * MX - 0.6) / 3;
  cols.forEach((c, i) => {
    const x = MX + i * (cw + 0.3), y = 1.7;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w: cw, h: 4.4, rectRadius: 0.08, fill: { color: "1D3640" }, line: { type: "none" } });
    chip(s, c[0].toUpperCase(), x + 0.3, y + 0.3, 1.4, c[1], c[1] === C.amber ? C.ink : C.white);
    s.addText(bullets(c[2], { paraSpaceAfter: 10 }), { x: x + 0.3, y: y + 0.85, w: cw - 0.6, h: 3.4, fontFace: BF, fontSize: 15, color: C.onDark, valign: "top", margin: 0, isTextBox: true });
  });
  pageNo += 1;
  s.addText(String(pageNo), { x: W - MX - 0.6, y: H - 0.42, w: 0.6, h: 0.3, fontFace: BF, fontSize: 10, color: C.onDarkMuted, align: "right", margin: 0, isTextBox: true });
  s.addText("Decisions requested today: headline deployment · DP levels · FL effort · cut priorities · AI-use disclosure", { x: MX, y: 6.35, w: W - 2 * MX, h: 0.4, fontFace: BF, fontSize: 14, bold: true, color: C.amber, margin: 0, isTextBox: true });
  s.addNotes(`الشريحة 25 — الملخص.
خلص: بروتوكول الداتا مجمد، الـ evaluator على الكتالوج كله، مسابقة الموديل (SASRec بالقاعدة المسجلة مع الإفصاح عن الانقلاب)، سلم الفيدرالي +22.7%، والمراجعات المستقلة واتسجلت الإصلاحات قبل أي قراءة.
شغال: موديلين السحابة الكاملة، التدريب المسبق، مرحلة S2، وتشغيلات الجهاز.
الجاي: S3 ثم S4 ثم التشغيلات الفيدرالية و DP الرئيسية؛ تجميد التحليل وقراءة validation واحدة (حوالي 2–3 أكتوبر، تقدير)؛ القفل ونظرة test واحدة (حوالي 4–5 أكتوبر)؛ اختبار المتصفح وجداول الرسالة (حوالي 6–7 أكتوبر).
اختم بالقرارات اللي محتاجها النهارده: الـ headline بتاع Edge، مستويات DP، قد إيه مجهود للفيدرالي، أولويات القطع، وبيان استخدام الذكاء الاصطناعي.`);
}

// =====================================================================
// Appendix divider
{
  const s = dark();
  s.addText("APPENDIX", { x: MX, y: 2.8, w: 8, h: 0.4, fontFace: BF, fontSize: 14, bold: true, charSpacing: 3, color: C.amber, margin: 0, isTextBox: true });
  s.addText("Backup slides for likely questions", { x: MX, y: 3.25, w: 11, h: 0.9, fontFace: HF, fontSize: 36, bold: true, color: C.white, margin: 0, isTextBox: true });
  pageNo += 1;
  s.addNotes("الملحق — شرائح احتياطية للأسئلة المتوقعة: الـ baselines، منحنى الـ learning rate كامل، تشغيلات المسابقة الثمانية، تفاصيل DP، والمصطلحات. ما تعرضهاش إلا لو اتسألت.");
}

// A1 baselines
{
  const s = light();
  header(s, "Appendix A — baselines", "Simple baselines vs the contest models (main validation users)");
  s.addChart(pres.charts.BAR, [{ name: "MRR@20", labels: ["Popularity", "Last viewed item", "Session kNN", "GRU (contest)", "SASRec (contest)"], values: [0.01943, 0.07111, 0.11138, 0.1569, 0.1612] }], {
    x: MX, y: 1.5, w: 8.2, h: 4.9, barDir: "col", chartColors: [C.gray, C.amber, C.gray, C.slate, C.teal], varyColors: true,
    showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0.0000", dataLabelFontSize: 13, dataLabelColor: C.ink,
    valAxisMinVal: 0, valAxisMaxVal: 0.18, valAxisLabelFormatCode: "0.00", valAxisLabelColor: C.muted, catAxisLabelColor: C.body, catAxisLabelFontSize: 12,
    valGridLine: { color: "E3E8EA", size: 0.5 }, catGridLine: { style: "none" }, showLegend: false,
    showTitle: true, title: "Validation MRR@20 (per-user mean)", titleFontSize: 13, titleColor: C.ink,
  });
  const x2 = 9.2, w2 = W - MX - x2;
  txt(s, bullets([
    "NDCG@10: popularity 0.0249, last item 0.1016, session kNN 0.1485.",
    "“Last viewed item” needs no training and sends nothing — it is the device floor.",
    "Session kNN reaches about 69% of the contest winner.",
  ]), x2, 1.7, w2, 4.3, { fontSize: 14.5 });
  footer(s, "Decision log (post-contest reported read of baselines, 28 Sep 11:39Z)");
  s.addNotes(`ملحق A — الـ baselines على الـ main validation: popularity 0.0194، آخر منتج 0.0711، session kNN 0.1114، مقابل GRU 0.1569 و SASRec 0.1612. NDCG@10: 0.0249 / 0.1016 / 0.1485.
دي أرقام مسموحة (اتقرت بعد المسابقة كـ reported). session kNN حوالي 69% من الفائز (0.1114/0.1612).`);
}

// A2 LR screen curve
{
  const s = light();
  header(s, "Appendix B — client learning-rate screen", "The full screen: rise, peak at ×8, fall");
  s.addChart(pres.charts.LINE, [{ name: "Calibration MRR@20", labels: ["×0.5", "×1", "×2", "×4", "×8", "×16"], values: [0.020263, 0.023473, 0.025562, 0.027570, 0.028796, 0.022909] }], {
    x: MX, y: 1.5, w: 8.2, h: 4.9, chartColors: [C.teal], lineSize: 3, lineDataSymbol: "circle", lineDataSymbolSize: 9,
    showValue: true, dataLabelPosition: "t", dataLabelFormatCode: "0.0000", dataLabelFontSize: 12, dataLabelColor: C.ink,
    valAxisMinVal: 0.018, valAxisMaxVal: 0.031, valAxisLabelFormatCode: "0.000", valAxisLabelColor: C.muted, catAxisLabelColor: C.body, catAxisLabelFontSize: 13,
    valGridLine: { color: "E3E8EA", size: 0.5 }, catGridLine: { style: "none" }, showLegend: false,
    showTitle: true, title: "FedAvg on 32,768 calibration users, full budget (1,707 rounds), seed 2026", titleFontSize: 13, titleColor: C.ink,
  });
  const x2 = 9.2, w2 = W - MX - x2;
  txt(s, bullets([
    "Multiplier of the client learning rate relative to the small model's central peak rate.",
    "First screen {0.5, 1, 2} ended at its edge → extension {4, 8, 16} registered before reading.",
    "×8 wins inside the grid, so ×32 was not needed.",
  ]), x2, 1.7, w2, 4.3, { fontSize: 14.5 });
  footer(s, "Learning-rate selection files; improvement-ladder register");
  s.addNotes("ملحق B — منحنى الـ learning rate كامل: 0.0203 / 0.0235 / 0.0256 / 0.0276 / 0.0288 / 0.0229. طلوع لحد ×8 وبعدين نزول عند ×16، فالفائز جوه الشبكة ومش محتاجين ×32. التمديد اتسجل قبل ما نقرا قيمه.");
}

// A3 contest runs table
{
  const s = light();
  header(s, "Appendix C — the eight contest runs", "Validation MRR@20 per run");
  const rows = [
    ["GRU, hidden 128", "2026", "0.15714", "0.15714"], ["GRU, hidden 128", "2027", "0.15673", "0.15673"],
    ["GRU, hidden 256", "2026", "0.15412", "0.15412"], ["GRU, hidden 256", "2027", "0.15345", "0.15345"],
    ["SASRec, 256 × 3 blocks", "2026", "0.16120", "0.15287"], ["SASRec, 256 × 3 blocks", "2027", "0.16122", "0.15258"],
    ["SASRec, 256 × 2 blocks", "2026", "0.16059", "0.15086"], ["SASRec, 256 × 2 blocks", "2027", "0.16006", "0.15011"],
  ];
  const hdr = ["Model", "Seed", "Best checkpoint within budget", "Fixed endpoint"].map(t => ({ text: t, options: { bold: true, color: C.white, fill: { color: C.ink }, fontSize: 13 } }));
  const body = rows.map((r, i) => r.map((t, j) => ({ text: t, options: { color: j === 0 ? C.ink : C.body, bold: j === 0, fill: { color: i % 2 ? C.white : C.panel }, fontSize: 13, align: j > 1 ? "right" : "left" } })));
  s.addTable([hdr, ...body], { x: MX, y: 1.55, w: 8.4, colW: [3.0, 1.1, 2.4, 1.9], fontFace: BF, border: { type: "solid", pt: 0.5, color: C.line }, margin: [5, 8, 5, 8], rowH: 0.5 });
  const x2 = 9.4, w2 = W - MX - x2;
  txt(s, bullets([
    "Within each family, the size was chosen by the 2-seed endpoint mean; between families by the best-checkpoint rule.",
    "SASRec's best checkpoint is at half the budget; GRU's at the end, so its two columns are equal.",
    "Winners: GRU hidden 128 and SASRec 3 blocks; SASRec then won between families.",
  ]), x2, 1.6, w2, 4.5, { fontSize: 13.5 });
  footer(s, "Decision log (reported read of the 8 runs, 28 Sep 11:39Z; contest run 11:25Z)");
  s.addNotes(`ملحق C — التشغيلات الثمانية. GRU 128: 0.15714 و 0.15673 (أحسن نقطة هي النهاية في الحالتين). GRU 256: 0.15412 و 0.15345 (أحسن نقطة هي النهاية). SASRec 3 blocks: 0.16120 و 0.16122 (أحسن نقطة في النص) والنهاية 0.15287 و 0.15258. SASRec 2 blocks: 0.16059 و 0.16006، والنهاية 0.15086 و 0.15011.
الاختيار جوه العائلة بمتوسط النهاية للبذرتين، وبين العائلتين بقاعدة أحسن checkpoint.`);
}

// A4 DP details
{
  const s = light();
  header(s, "Appendix D — differential privacy details", "Mechanism and accounting");
  const kv = [
    ["Mechanism", "DP-FedAvg: per-client clipping of the update, Gaussian noise on the sum"],
    ["Adjacency", "Client-level add/remove (one user's whole history)"],
    ["δ", "5 × 10⁻⁶"],
    ["Sampling", "Poisson, about 1,024 clients per round, 384 rounds; a non-noisy large-cohort twin is run for comparison"],
    ["Noise multipliers", "ε = 8 → 0.5642 · ε = 1 → 1.3094 · ε = 4 → 0.7244 (independently re-derived)"],
    ["Accounting", "Rényi DP; a without-replacement bound reported as sensitivity"],
    ["Clip norm", "Calibrated on a separate run before any DP run; recorded before launch"],
    ["Not covered", "Tuning on calibration users; public bootstrap (catalogue, pretrained weights)"],
  ];
  const body = kv.map((r, i) => [{ text: r[0], options: { bold: true, color: C.ink, fill: { color: i % 2 ? C.white : C.panel }, fontSize: 13.5 } }, { text: r[1], options: { color: C.body, fill: { color: i % 2 ? C.white : C.panel }, fontSize: 13.5 } }]);
  s.addTable(body, { x: MX, y: 1.55, w: W - 2 * MX, colW: [2.6, 9.53], fontFace: BF, border: { type: "solid", pt: 0.5, color: C.line }, margin: [5, 8, 5, 8], rowH: 0.56 });
  footer(s, "Independent federated-code review (DP accountant check); privacy statement registration; project map (DP settings)");
  s.addNotes("ملحق D — تفاصيل DP: قص التحديث لكل عميل + ضوضاء Gaussian على المجموع؛ الجوار على مستوى العميل؛ δ=5e-6؛ Poisson حوالي 1024 عميل في الجولة × 384 جولة؛ z = 0.5642 (ε=8)، 1.3094 (ε=1)، 0.7244 (ε=4) واتأكدوا بشكل مستقل؛ Rényi accounting؛ الـ clip norm بيتظبط في تشغيل منفصل قبل أي DP. مش محسوب: الضبط على calibration والنقطة العامة.");
}

// A5 Glossary + update log
{
  const s = light();
  header(s, "Appendix E — terms", "Plain-language glossary");
  const g = [
    ["Retention", "Quality of a method ÷ quality of the cloud reference, × 100"],
    ["MRR@20", "Mean reciprocal rank of the true next product within the top 20, averaged per user"],
    ["Calibration users", "32,768 users used only for tuning screens; their values may be read before the final analysis"],
    ["Pre-registered", "Rule written and hash-sealed before the result it governs is seen"],
    ["Sealed test week", "23–29 Nov 2019; opened once, after all methods are locked"],
    ["Training budget", "Six passes' worth of exposure over the cohort's data, equal for all methods"],
    ["Ladder stage", "One registered change to the federated recipe, kept only if it gains ≥ 0.002"],
    ["Frozen item tables", "Product tables fixed after pretraining; only the small encoder is trained and uploaded"],
  ];
  const body = g.map((r, i) => [{ text: r[0], options: { bold: true, color: C.ink, fill: { color: i % 2 ? C.white : C.panel }, fontSize: 13.5 } }, { text: r[1], options: { color: C.body, fill: { color: i % 2 ? C.white : C.panel }, fontSize: 13.5 } }]);
  s.addTable(body, { x: MX, y: 1.55, w: W - 2 * MX, colW: [2.8, 9.33], fontFace: BF, border: { type: "solid", pt: 0.5, color: C.line }, margin: [5, 8, 5, 8], rowH: 0.56 });
  footer(s, "Deck data file: company/presentations/SUPERVISOR_DECK_DATA.md (every number with its source)");
  s.addNotes(`ملحق E — المصطلحات.
سجل التحديثات (Update log):
- v1 — 2026-09-29: أول نسخة. الحالة من DECISION_LOG لحد الصف 228 (18:36 UTC)، والسلم لحد S1b. S2 شغالة ومفيش قيم ليها لسه.
عند التحديث: هنكتب هنا إيه اللي اتغير (مراحل سلم جديدة، حالة التشغيلات، التواريخ).`);
}

pres.writeFile({ fileName: OUT }).then(f => console.log("wrote " + f + " slides=" + pageNo));
