// Supervisor deck generator — NUMBERS version (work done, earlier phases, why we changed) (pptxgenjs). Every number here is listed with its source in SUPERVISOR_DECK_DATA.md.
const pptxgen = require("pptxgenjs");
const path = require("path");

const OUT = process.argv[2] || path.join(__dirname, "PPSI_Supervisor_Deck_Numbers.pptx");
const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE"; // 13.333 x 7.5 in
pres.title = "Privacy-Preserving Shopping Intelligence — supervisor update (numbers version)";
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

// ===================== NUMBERS VERSION: work done, earlier experiments, why we changed =====================
function statusChip(s, t, x, y, w, kind) {
  const m = { done: [C.slate, C.white], running: [C.teal, C.white], sealed: [C.amberLight, C.amberDark], next: [C.slateLight, C.ink] }[kind];
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h: 0.34, rectRadius: 0.17, fill: { color: m[0] }, line: kind === "sealed" ? { color: C.amber, width: 1 } : { type: "none" } });
  s.addText(t, { x, y, w, h: 0.34, fontFace: BF, fontSize: 11, bold: true, color: m[1], align: "center", valign: "middle", margin: 0, isTextBox: true });
}
function stat(s, x, y, w, h, big, small, fill, color) {
  card(s, x, y, w, h, fill || C.panel);
  txt(s, big, x + 0.25, y + 0.15, w - 0.5, 0.75, { fontFace: HF, fontSize: 32, bold: true, color: color || C.teal });
  txt(s, small, x + 0.25, y + 0.9, w - 0.5, h - 1.0, { fontSize: 13, color: C.body });
}
function oldTag(s) { chip(s, "EARLIER PHASE · full data · not comparable with the current study", W - MX - 5.6, 0.36, 5.6, C.slateLight, C.slate); }

// 1. Title
{
  const s = dark();
  s.addText("PRIVACY-PRESERVING SHOPPING INTELLIGENCE", { x: MX, y: 1.4, w: 11, h: 0.4, fontFace: BF, fontSize: 14, bold: true, charSpacing: 3, color: C.amber, margin: 0, isTextBox: true });
  s.addText("The project in numbers", { x: MX, y: 1.9, w: 11.5, h: 1.0, fontFace: HF, fontSize: 48, bold: true, color: C.white, margin: 0, isTextBox: true });
  s.addText("What was built, what earlier experiments showed, why we changed course, and what is measured so far", { x: MX, y: 3.05, w: 11, h: 0.9, fontFace: BF, fontSize: 20, color: C.onDarkMuted, margin: 0, isTextBox: true });
  s.addText("Ahmed Abdelhamed · MSc, Queen's University · supervisor meeting, 30 September 2026 · facts as of 29 Sep, 18:36 UTC", { x: MX, y: 6.5, w: 12, h: 0.35, fontFace: BF, fontSize: 13, color: C.onDarkMuted, margin: 0, isTextBox: true });
  pageNo += 1;
  s.addNotes(`الشريحة 1 — النسخة دي مركزة على الأرقام: الشغل اللي اتعمل، التجارب القديمة، وليه غيرنا الاتجاه، وإيه اللي اتقاس لحد دلوقتي.
قاعدة مهمة: نتائج الطرق الرئيسية في الدراسة الحالية (الفيدرالي، الجهاز، السحابة الكاملة) وأسبوع الـ test مقفولين لحد القراءة المسجلة مسبقاً.`);
}

// 2. Project in numbers
{
  const s = light();
  header(s, "At a glance", "Eight numbers that summarise the work");
  const st = [
    ["158,486", "products ranked for every prediction"], ["1.45 M", "users in the study pool (5–100 events each)"],
    ["131,072", "users in the main cohort"], ["2.59 M", "training decisions in the main cohort"],
    ["4", "phases: next category → next product (cloud) → features → deployment study"], ["8", "runs in the backbone contest (2 families × 2 sizes × 2 seeds)"],
    ["+22.7%", "federated gain so far from tuning (calibration users)"], ["7 / 10", "independent 14-part review; 8–8.5 after the fixes"],
  ];
  const cw = (W - 2 * MX - 0.9) / 4, ch = 2.1;
  st.forEach((t, i) => stat(s, MX + (i % 4) * (cw + 0.3), 1.6 + Math.floor(i / 4) * (ch + 0.3), cw, ch, t[0], t[1], i === 6 ? C.tealLight : C.panel));
  footer(s, "Data protocol and population lock; contest result; ladder register; review synthesis");
  s.addNotes(`الشريحة 2 — المشروع في 8 أرقام:
158,486 منتج بيترتبوا في كل توقع. 1.45 مليون مستخدم في الـ pool (كل واحد 5–100 حدث). 131,072 في الـ main cohort بـ 2.59 مليون قرار تدريب.
4 مراحل: الفئة الجاية، المنتج الجاي بموديل سحابة، الفيتشرز، ودراسة النشر الحالية. 8 تشغيلات في مسابقة الموديل.
+22.7% تحسن للفيدرالي من الضبط (calibration). والمراجعة المستقلة 7/10 والهدف 8–8.5.`);
}

// 3. Journey
{
  const s = light();
  header(s, "The journey", "Four phases, each answering one question");
  const ph = [
    ["1", "Next-category MVP", "Can we predict the next category?", "0.3474", "GRU vs 0.3143 transition table (Oct 2019)", C.slate],
    ["2", "Next product", "How good can a cloud model get?", "0.1713", "best central model (SASRec), validation, full data", C.slate],
    ["3", "Features", "Which columns help?", "58 → 24", "input columns; 21 of 24 computable on device", C.slate],
    ["4", "Deployment study", "How much survives off the cloud?", "+22.7%", "federated tuning so far; main results sealed", C.teal],
  ];
  const cw = (W - 2 * MX - 0.9) / 4;
  s.addShape(pres.shapes.LINE, { x: MX + 0.4, y: 1.95, w: W - 2 * MX - 0.8, h: 0, line: { color: C.line, width: 3 } });
  ph.forEach((p, i) => {
    const x = MX + i * (cw + 0.3);
    s.addShape(pres.shapes.OVAL, { x: x + 0.15, y: 1.7, w: 0.5, h: 0.5, fill: { color: p[5] }, line: { color: C.white, width: 2 } });
    s.addText(p[0], { x: x + 0.15, y: 1.7, w: 0.5, h: 0.5, fontFace: HF, fontSize: 16, bold: true, color: C.white, align: "center", valign: "middle", margin: 0, isTextBox: true });
    card(s, x, 2.45, cw, 3.9, i === 3 ? C.tealLight : C.panel);
    txt(s, p[1], x + 0.25, 2.6, cw - 0.5, 0.45, { fontSize: 19, bold: true, fontFace: HF, color: C.ink });
    txt(s, p[2], x + 0.25, 3.1, cw - 0.5, 0.7, { fontSize: 13.5, italic: true, color: C.muted });
    txt(s, p[3], x + 0.25, 3.95, cw - 0.5, 0.8, { fontSize: 36, bold: true, fontFace: HF, color: i === 3 ? C.tealDark : C.slate });
    txt(s, p[4], x + 0.25, 4.85, cw - 0.5, 1.3, { fontSize: 13 });
  });
  footer(s, "MVP report (next category); central-model campaign reports; feature dictionary and feature search; ladder register");
  s.addNotes(`الشريحة 3 — الرحلة في 4 مراحل:
1) MVP توقع الفئة الجاية (أكتوبر 2019): GRU 0.3474 مقابل جدول الانتقالات 0.3143.
2) توقع المنتج الجاي بموديل سحابة على الداتا الكاملة: أحسن موديل SASRec 0.1713 على الـ validation.
3) هندسة الفيتشرز: من 58 عمود لـ 24، و21 منهم ممكن يتحسبوا على الجهاز.
4) دراسة النشر الحالية: الفيدرالي +22.7% من الضبط لحد دلوقتي، والنتائج الرئيسية مقفولة.`);
}

// 4. Earlier phase 1: next-category task
{
  const s = light();
  header(s, "Earlier phase 1 — next-category task (MVP)", "Our next-category numbers");
  chip(s, "EARLIER PHASE · Oct 2019 · 97,279 users · 588 categories", W - MX - 5.6, 0.36, 5.6, C.slateLight, C.slate);
  s.addChart(pres.charts.BAR, [{ name: "MRR@20", labels: ["Popularity", "Transition table", "GRU (3 seeds)"], values: [0.1049, 0.3143, 0.3474] }], {
    x: MX, y: 1.5, w: 6.0, h: 4.9, barDir: "col", chartColors: [C.gray, C.gray, C.teal], varyColors: true, barGapWidthPct: 40,
    showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0.0000", dataLabelFontSize: 15, dataLabelColor: C.ink,
    valAxisMinVal: 0, valAxisMaxVal: 0.4, valAxisLabelFormatCode: "0.00", valAxisLabelColor: C.muted, catAxisLabelColor: C.body, catAxisLabelFontSize: 13,
    valGridLine: { color: "E3E8EA", size: 0.5 }, catGridLine: { style: "none" }, showLegend: false,
    showTitle: true, title: "Category-change decisions, validation, MRR@20", titleFontSize: 12, titleColor: C.ink,
  });
  const x2 = 6.9, w2 = W - MX - x2, cw = (w2 - 0.3) / 2;
  stat(s, x2, 1.6, cw, 1.75, "+0.0331", "GRU over the transition table (seed spread 0.0007)", C.tealLight, C.tealDark);
  stat(s, x2 + cw + 0.3, 1.6, cw, 1.75, "88.4%", "federated pilot kept of centralised quality (1,000 clients)", C.tealLight, C.tealDark);
  stat(s, x2, 3.55, cw, 1.75, "82%", "of the overall metric solved by a no-learning rule");
  stat(s, x2 + cw + 0.3, 3.55, cw, 1.75, "+0.066", "purchase PR-AUC gain: 0.076 (rule) → 0.142 (model)");
  card(s, x2, 5.5, w2, 0.9, C.amberLight);
  txt(s, [{ text: "Stronger or weaker than published work? ", options: { bold: true, color: C.amberDark } }, { text: "Unknown — no published next-category results on REES46 to compare with." }], x2 + 0.25, 5.55, w2 - 0.5, 0.8, { fontSize: 13.5, valign: "middle" });
  footer(s, "GitHub repo: docs/MVP_REPORT_AR.md §§1–2; ADR-001 (evaluation protocol); MVP federated pilot evidence (mvp-t1-003)");
  s.addNotes(`الشريحة 4 — المرحلة القديمة 1: مهمة الفئة الجاية (MVP) من الـ GitHub.
الداتا: أكتوبر 2019 بس، 97,279 مستخدم، 588 فئة.
على قرارات تغيير الفئة (validation، macro MRR@20): popularity 0.1049، جدول الانتقالات 0.3143، GRU (3 بذور) 0.3474 → +0.0331، والفرق بين البذور 0.0007.
الفيدرالي في تجربة صغيرة (1,000 عميل) حافظ على 88.4% من الجودة المركزية (0.0996 مقابل 0.1127، بطريقة قياس تانية، ما تخلطهاش بـ 0.3474).
مشكلة: 82% من المقياس الكلي بتتحل بقاعدة من غير تعلم (تكرار نفس الفئة) — المهمة سهلة بشكل مصطنع.
توقع الشراء: PR-AUC من 0.076 (قاعدة) لـ 0.142 (موديل).
أقوى ولا أضعف من الأبحاث؟ مانعرفش — مالقيناش أبحاث منشورة على REES46 لمهمة الفئة الجاية نقارن بيها.`);
}

// 5. Why we moved to next product
{
  const s = light();
  header(s, "Why we moved to next-product", "Three reasons, in numbers");
  const rs = [
    ["0", "published next-category results on REES46 — no way to know if we are stronger or weaker", C.amberLight, C.amberDark],
    ["82%", "of the next-category metric solved without learning — the task is artificially easy", C.panel, C.slate],
    ["1", "official published next-product benchmark on REES46 (GRU4Rec, RecSys 2023) — a reference point", C.tealLight, C.tealDark],
  ];
  const cw = (W - 2 * MX - 0.6) / 3;
  rs.forEach((r, i) => {
    const x = MX + i * (cw + 0.3);
    card(s, x, 1.7, cw, 3.3, r[2]);
    txt(s, r[0], x + 0.3, 1.9, cw - 0.6, 1.2, { fontFace: HF, fontSize: 64, bold: true, color: r[3] });
    txt(s, r[1], x + 0.3, 3.2, cw - 0.6, 1.6, { fontSize: 15.5, color: C.ink });
  });
  const cmp = [["Next category", "588 classes", C.slate], ["Next product", "158,486 products", C.teal]];
  cmp.forEach((c, i) => {
    const x = MX + i * 6.2;
    chip(s, c[0], x, 5.45, 2.0, c[2], C.white);
    txt(s, c[1], x + 2.2, 5.4, 3.5, 0.45, { fontSize: 20, bold: true, fontFace: HF, color: C.ink, valign: "middle" });
  });
  txt(s, "Harder, more standard task — agreed with the supervisor.", MX, 6.15, W - 2 * MX, 0.4, { fontSize: 13.5, italic: true, color: C.muted });
  footer(s, "ADR-001; thesis evidence note (proposal amendment table); literature register card R05; thesis research review A3");
  s.addNotes(`الشريحة 5 — ليه انتقلنا لتوقع المنتج الجاي.
1) مفيش أبحاث منشورة بأرقام على REES46 لمهمة الفئة الجاية، فمنقدرش نعرف إحنا أقوى ولا أضعف.
2) 82% من مقياس الفئة بيتحل من غير تعلم — المهمة سهلة بشكل مصطنع.
3) مهمة المنتج الجاي ليها benchmark منشور رسمي على REES46 (GRU4Rec، RecSys 2023) نقدر نقارن بيه.
كمان الصعوبة: 588 فئة مقابل 158,486 منتج. واتفقنا على التغيير ده مع المشرف.`);
}

// 5. Earlier phase 2: central models on full data
{
  const s = light();
  header(s, "Earlier phase 2 — next product, cloud models", "On the full data, sequence models beat the baselines");
  oldTag(s);
  s.addChart(pres.charts.BAR, [{ name: "Validation MRR@20", labels: ["Popularity", "Item-transition baseline", "GRU (item IDs)", "SASRec (rich features)"], values: [0.020168, 0.148740, 0.168988, 0.171339] }], {
    x: MX, y: 1.5, w: 7.8, h: 4.9, barDir: "bar", chartColors: [C.gray, C.gray, C.slate, C.teal], varyColors: true, barGapWidthPct: 40,
    showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0.0000", dataLabelFontSize: 14, dataLabelColor: C.ink,
    valAxisMinVal: 0, valAxisMaxVal: 0.2, valAxisHidden: true, valGridLine: { style: "none" }, catGridLine: { style: "none" },
    catAxisLabelColor: C.body, catAxisLabelFontSize: 14, catAxisOrientation: "maxMin", showLegend: false,
    showTitle: true, title: "Validation-confirm week (20–23 Nov), macro MRR@20, full data", titleFontSize: 13, titleColor: C.ink,
  });
  const x2 = 8.8, w2 = W - MX - x2;
  stat(s, x2, 1.6, w2, 1.6, "52.8 M", "training decisions (40.2M loss examples)");
  stat(s, x2, 3.4, w2, 1.6, "+0.0024", "SASRec over GRU — but ID-only SASRec was below GRU", C.amberLight, C.amberDark);
  txt(s, "The gain came from the extra feature channels, not the architecture.", x2, 5.2, w2, 1.0, { fontSize: 13, italic: true, color: C.muted });
  footer(s, "Central-model campaign final report §§1, 6.1; planning evidence packet (data census)");
  s.addNotes(`الشريحة 5 — المرحلة القديمة 2: موديلات مركزية على الداتا الكاملة (52.8 مليون قرار).
على أسبوع الـ validation-confirm (20–23 نوفمبر): popularity 0.0202، item-transition 0.1487، GRU 0.1690، SASRec 0.1713.
SASRec أعلى بـ +0.0024، بس نسخة SASRec بالـ IDs بس كانت أقل من GRU، يعني المكسب جه من الفيتشرز الزيادة مش من المعمارية.
ملاحظة لو اتسألت: أسبوع الـ test اتقرا مرة في المرحلة دي (SASRec 0.1705 مقابل GRU 0.1687)؛ مش حاطينه على الشريحة لأن نفس الأسبوع هو الـ test المقفول في الدراسة الحالية. والأرقام دي مش قابلة للمقارنة مع الدراسة الحالية (داتا ومستخدمين وفيتشرز مختلفة).`);
}

// 6b. Our numbers vs the published benchmark
{
  const s = light();
  header(s, "Our numbers vs the published benchmark", "Below the paper — and why");
  oldTag(s);
  s.addChart(pres.charts.BAR, [
    { name: "Published GRU4Rec (REES46)", labels: ["Recall@20", "MRR@20"], values: [0.5291, 0.2003] },
    { name: "Our SASRec", labels: ["Recall@20", "MRR@20"], values: [0.4315, 0.1542] },
    { name: "Our GRU", labels: ["Recall@20", "MRR@20"], values: [0.4242, 0.1531] },
  ], {
    x: MX, y: 1.5, w: 6.0, h: 4.95, barDir: "col", barGrouping: "clustered", chartColors: [C.amber, C.teal, C.slate],
    showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0.000", dataLabelFontSize: 12, dataLabelColor: C.ink,
    valAxisMinVal: 0, valAxisMaxVal: 0.6, valAxisLabelFormatCode: "0.0", valAxisLabelColor: C.muted, catAxisLabelColor: C.body, catAxisLabelFontSize: 14,
    valGridLine: { color: "E3E8EA", size: 0.5 }, catGridLine: { style: "none" },
    showLegend: true, legendPos: "t", legendFontSize: 12, legendColor: C.body,
    showTitle: true, title: "Per-event metrics (ours: validation-confirm week, full data)", titleFontSize: 12, titleColor: C.ink,
  });
  const x2 = 6.9, w2 = W - MX - x2;
  const rs = [
    ["−0.098", "Recall@20", "we rank all 158,486 products and count new items as misses"],
    ["−0.046", "MRR@20", "we predict the following week; the paper uses a last-day split"],
    ["5 epochs", "our training", "≈ 20.7 GPU-h on 2× A16 (15.4 GiB) — the paper is heavily tuned"],
    ["20×", "less data now", "current study 2.59M vs 52.8M decisions (chosen for federated) → 0.1612"],
  ];
  rs.forEach((r, i) => {
    const y = 1.55 + i * 1.02;
    card(s, x2, y, w2, 0.9, i % 2 ? C.white : C.panel, i % 2 ? { line: C.line } : {});
    txt(s, r[0], x2 + 0.2, y, 1.55, 0.9, { fontFace: HF, fontSize: 21, bold: true, color: C.amberDark, valign: "middle" });
    txt(s, r[1], x2 + 1.8, y, 1.35, 0.9, { fontSize: 12.5, bold: true, color: C.ink, valign: "middle" });
    txt(s, r[2], x2 + 3.15, y, w2 - 3.3, 0.9, { fontSize: 12.5, valign: "middle" });
  });
  card(s, x2, 5.7, w2, 0.75, C.tealLight);
  txt(s, [{ text: "Not a same-protocol re-run. ", options: { bold: true, color: C.tealDark } }, { text: "Now: 2× A40 server — experiments running." }], x2 + 0.2, 5.72, w2 - 0.4, 0.7, { fontSize: 13.5, valign: "middle" });
  footer(s, "Hidasi & Czapp, RecSys 2023 (official GRU4Rec REES46 page) as recorded in the literature register (R05) and benchmark note; central-model report §6.1; planning evidence (A16)");
  s.addNotes(`الشريحة — أرقامنا مقابل الـ benchmark المنشور (GRU4Rec الرسمي على REES46، Hidasi & Czapp، RecSys 2023).
المنشور: Recall@20 = 0.5291، MRR@20 = 0.2003 (على مستوى الحدث).
إحنا (المرحلة القديمة، أسبوع validation-confirm، داتا كاملة): SASRec Recall@20 0.4315 و MRR@20 0.1542؛ GRU 0.4242 و 0.1531.
ليه أقل:
- Recall أقل بـ 0.098: بنرتب الكتالوج كله (158,486 منتج) وأي منتج جديد بيتحسب غلط.
- MRR أقل بـ 0.046: بنتوقع الأسبوع اللي بعده، والبحث بيستخدم آخر يوم.
- التدريب: 5 epochs بس، حوالي 20.7 ساعة GPU على 2× A16 (15.4 جيجا) — البحث متظبط جامد. ده بسبب ضعف الموارد وقتها.
- الدراسة الحالية داتا أقل 20 مرة (2.59 مليون مقابل 52.8 مليون) علشان الفيدرالي، فالرقم 0.1612.
الخلاصة: مش إعادة تشغيل بنفس البروتوكول، فمنقدرش نقول أقوى ولا أضعف بشكل قاطع. دلوقتي معانا سيرفر 2× A40 وبدأنا نجرب.
الفجوة ما اتفككتش بتجارب (ablation)، فالأسباب دي هي الفروق المعروفة في البروتوكول مش قياس لكل سبب.
ملحوظة: رقمنا الرئيسي 0.1713 متوسط لكل مستخدم (macro) — مش نفس طريقة البحث، فما نقارنوش بـ 0.2003 مباشرة.`);
}

// 6c. Discussion: task and benchmark
discussion(1, "Task and benchmark", [
  ["Do you agree next-product is the right task for the thesis?", "Next-category had no published REES46 results and 82% was solvable without learning."],
  ["Should we re-run the published GRU4Rec protocol on our server?", "It would tell us directly whether we are stronger or weaker (≈ 1 extra run, estimate)."],
  ["Which metric should lead: per-user MRR@20 (ours) or per-event (papers)?", "We can report both."],
], `ديسكشن 1 — المهمة والـ benchmark.
1) موافق إن توقع المنتج الجاي هو المهمة الصح للرسالة؟ الفئة الجاية ماكانش ليها أبحاث منشورة على REES46، و82% منها بيتحل من غير تعلم.
2) نعيد تشغيل بروتوكول GRU4Rec المنشور على السيرفر؟ ده هيقولنا مباشرة إحنا أقوى ولا أضعف (تقريباً تشغيل واحد زيادة، تقدير).
3) أنهي مقياس يبقى الأساسي: MRR@20 لكل مستخدم (بتاعنا) ولا لكل حدث (بتاع الأبحاث)؟ نقدر نعرض الاتنين.`, "earlier phases vs the published benchmark");

// 6. Earlier phase 2b: SASRec repair
{
  const s = light();
  header(s, "Earlier phase 2 — the SASRec repair", "A failed first attempt, fixed and documented");
  oldTag(s);
  s.addChart(pres.charts.BAR, [{ name: "Selection-week MRR@20", labels: ["First attempt (failed)", "Repaired (5 fixes)", "+ cooldown epoch"], values: [0.11791, 0.181421, 0.186298] }], {
    x: MX, y: 1.5, w: 7.2, h: 4.9, barDir: "col", chartColors: ["C9A27A", "5E9DA1", C.teal], varyColors: true, barGapWidthPct: 40,
    showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0.0000", dataLabelFontSize: 15, dataLabelColor: C.ink,
    valAxisMinVal: 0, valAxisMaxVal: 0.21, valAxisLabelFormatCode: "0.00", valAxisLabelColor: C.muted, catAxisLabelColor: C.body, catAxisLabelFontSize: 13,
    valGridLine: { color: "E3E8EA", size: 0.5 }, catGridLine: { style: "none" }, showLegend: false,
    showTitle: true, title: "SASRec with features, seed 2026, validation-selection week", titleFontSize: 13, titleColor: C.ink,
  });
  const x2 = 8.2, w2 = W - MX - x2;
  stat(s, x2, 1.6, w2, 1.55, "+54%", "repaired vs failed attempt (0.1814 vs 0.1179)", C.tealLight, C.tealDark);
  txt(s, "Fixes: untied output table · no ×16 input scaling + input layer norm · Adam β2 0.999 · re-centred output table · new learning-rate schedule", x2, 3.35, w2, 1.3, { fontSize: 13.5 });
  stat(s, x2, 4.7, w2, 1.55, "0.0002", "gap between two seeds at the final epoch (stable)");
  footer(s, "Central-model campaign final report §§3–4");
  s.addNotes(`الشريحة 6 — إصلاح SASRec في المرحلة القديمة.
المحاولة الأولى فشلت: 0.1179. بعد 5 إصلاحات (فك ربط جدول الخرج، شيل الضرب ×16 مع LayerNorm على الدخل، Adam β2 = 0.999، إعادة توسيط جدول الخرج، وجدول learning rate جديد) وصلت 0.1814، يعني +54%. وبعد epoch تبريد إضافية 0.1863.
الفرق بين البذرتين في الآخر 0.0002 بس، يعني ثابت.
الدرس: كل فشل اتوثق ومتقفلش عليه — وده نفس الأسلوب اللي بنستخدمه في سلم الفيدرالي.`);
}

// 7a. Feature engineering: the columns
{
  const s = light();
  header(s, "Feature engineering", "From 58 input columns to 24");
  const groups = [
    ["6", "token columns", "item · category · brand · main category · daypart · weekday"],
    ["19", "numeric columns", "price · hour/weekday (sin, cos) · gap since last view · item age · session counts"],
    ["15", "quality flags", "price missing / zero · brand recovered · category conflict …"],
    ["12", "user-context values", "past views · sessions · brand / category match share · price vs usual"],
    ["6", "masks", "history available · previous session available …"],
  ];
  txt(s, "Earlier phase — 58 inputs", MX, 1.5, 7, 0.35, { fontSize: 15, bold: true, color: C.ink });
  groups.forEach((g, i) => {
    const y = 1.95 + i * 0.86;
    card(s, MX, y, 7.3, 0.76, i % 2 ? C.white : C.panel, i % 2 ? { line: C.line } : {});
    txt(s, g[0], MX + 0.2, y, 0.8, 0.76, { fontFace: HF, fontSize: 26, bold: true, color: C.teal, valign: "middle" });
    txt(s, g[1], MX + 1.05, y, 2.0, 0.76, { fontSize: 13.5, bold: true, color: C.ink, valign: "middle" });
    txt(s, g[2], MX + 3.05, y, 4.1, 0.76, { fontSize: 12, valign: "middle" });
  });
  const x2 = 8.3, w2 = W - MX - x2;
  txt(s, "Current study — 24 inputs", x2, 1.5, w2, 0.35, { fontSize: 15, bold: true, color: C.ink });
  s.addChart(pres.charts.DOUGHNUT, [{ name: "Inputs", labels: ["Token", "Numeric", "Flag", "User context", "Mask"], values: [5, 7, 2, 6, 4] }], {
    x: x2, y: 1.85, w: w2, h: 2.6, chartColors: [C.tealDark, C.teal, C.gray, C.amber, C.slate], holeSize: 55,
    showValue: true, dataLabelColor: C.white, dataLabelFontSize: 12, showLegend: true, legendPos: "r", legendFontSize: 12, legendColor: C.body, showTitle: false,
  });
  stat(s, x2, 4.6, (w2 - 0.2) / 2, 1.8, "24/53", "kept: statistics filter (→ 30), then grouped search", C.panel, C.slate);
  stat(s, x2 + (w2 - 0.2) / 2 + 0.2, 4.6, (w2 - 0.2) / 2, 1.8, "21 / 24", "computable on the device", C.tealLight, C.tealDark);
  footer(s, "Feature dictionary §15.1; execution reference §2.6; feature search (decision log rows 39, 52–54); review part 4 (features)");
  s.addNotes(`الشريحة — هندسة الفيتشرز (الأعمدة).
المرحلة القديمة: 58 عمود دخل:
- 6 أعمدة tokens: المنتج، الفئة، البراند، الفئة الرئيسية، جزء اليوم، يوم الأسبوع.
- 19 عمود رقمي: السعر، الساعة واليوم (sin/cos)، الوقت من آخر مشاهدة، عمر المنتج، عدادات الجلسة.
- 15 علامة جودة: السعر ناقص أو صفر، البراند متصلح، تعارض الفئة...
- 12 قيمة عن المستخدم: عدد المشاهدات والجلسات السابقة، نسبة تطابق البراند/الفئة، السعر مقارنة بالعادي.
- 6 masks: التاريخ متاح، الجلسة السابقة متاحة...
الدراسة الحالية: 24 عمود: 5 tokens، 7 أرقام، 2 علامات، 6 عن المستخدم، 4 masks. الاختيار: 53 → 30 بالإحصاء → 24 ببحث جماعي. ومفيش أي عمود تقويم فضل.
21 من 24 ممكن يتحسبوا على الجهاز نفسه؛ الـ 3 الباقيين محتاجين قاموس مشترك (المنتج، الفئة، البراند).`);
}

// 7b. Feature study: what each group is worth
{
  const s = light();
  header(s, "Feature study", "What each feature group is worth");
  oldTag(s);
  s.addChart(pres.charts.BAR, [{ name: "Change in MRR@20 when removed", labels: ["All side features", "Item age", "Current price", "Main category", "Brand", "Cold user", "Previous gap", "Calendar"], values: [-0.046669, -0.024654, -0.009761, -0.006262, -0.005687, -0.004068, -0.003619, 0.00066] }], {
    x: MX, y: 1.5, w: 7.6, h: 4.95, barDir: "bar", chartColors: [C.amber, C.amber, C.amber, C.amber, C.amber, C.amber, C.amber, C.gray], varyColors: true, barGapWidthPct: 35,
    showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "+0.0000;−0.0000", dataLabelFontSize: 12, dataLabelColor: C.ink,
    valAxisMinVal: -0.055, valAxisMaxVal: 0.01, valAxisHidden: true, valGridLine: { style: "none" }, catGridLine: { style: "none" },
    catAxisLabelColor: C.body, catAxisLabelFontSize: 13, catAxisOrientation: "maxMin", catAxisLabelPos: "low", showLegend: false,
    showTitle: true, title: "Macro MRR@20 change when a group is removed (10k users, 112,365 decisions)", titleFontSize: 12, titleColor: C.ink,
  });
  const x2 = 8.6, w2 = W - MX - x2;
  stat(s, x2, 1.6, w2, 1.5, "−0.0467", "without any side feature: features matter", C.amberLight, C.amberDark);
  stat(s, x2, 3.3, w2, 1.5, "+0.00026", "best new candidate; threshold 0.002 → keep current", C.panel, C.slate);
  stat(s, x2, 5.0, w2, 1.45, "+0.0007", "calendar: within noise → dropped later", C.panel, C.slate);
  footer(s, "Feature-study final report §3; living thesis reference (decision KEEP_CURRENT)");
  s.addNotes(`الشريحة — دراسة الفيتشرز: كل مجموعة تسوى قد إيه (بنشيلها ونشوف الفرق).
شيل كل الفيتشرز الجانبية: −0.0467. عمر المنتج: −0.0247. السعر الحالي: −0.0098. الفئة الرئيسية: −0.0063. البراند: −0.0057. مستخدم بارد: −0.0041. الوقت من آخر مشاهدة: −0.0036. التقويم: +0.0007 (ضوضاء).
يعني الفيتشرز مهمة، بس أحسن مرشح جديد كسب +0.00026 بس (الحد 0.002) → فضلنا على نفس الفيتشرز. والتقويم اتشال في الدراسة الحالية.`);
}

// 8. Why we changed
{
  const s = light();
  header(s, "Why we changed course", "The earlier phases answered a different question");
  const rs = [
    ["Wrong question", "Best cloud model ≠ the proposal's question (cloud vs federated vs on-device)", "→ a deployment study"],
    ["One user = one device", "Raw users: median 6 events, max 11,628; 473,402 users with a single event", "→ users with 5–100 events: 1.45M"],
    ["Federated cost", "52.8M decisions is too large to simulate federated training many times", "→ main cohort 2.59M (≈ 20× smaller)"],
    ["Unfair family comparison", "GRU and SASRec had different features and training schedules", "→ a new contest on one clean protocol"],
    ["Test week already seen", "The old test week was read once", "→ disclosed as seen; one new sealed look"],
    ["Limited compute", "2× A16 GPUs (15.4 GiB each), 12.8 GiB free disk", "→ now 2× A40 server; experiments running"],
  ];
  rs.forEach((r, i) => {
    const y = 1.5 + i * 0.84;
    card(s, MX, y, W - 2 * MX, 0.76, i % 2 ? C.white : C.panel, i % 2 ? { line: C.line } : {});
    txt(s, r[0], MX + 0.3, y, 3.0, 0.76, { fontSize: 16, bold: true, fontFace: HF, color: C.ink, valign: "middle" });
    txt(s, r[1], MX + 3.4, y, 5.4, 0.76, { fontSize: 13.5, valign: "middle" });
    txt(s, r[2], MX + 8.95, y, W - 2 * MX - 9.1, 0.76, { fontSize: 14.5, bold: true, color: C.tealDark, valign: "middle" });
  });
  footer(s, "Owner scope for the deployment study; planning evidence packet (data census); population decision; central-model report");
  s.addNotes(`الشريحة 8 — ليه غيرنا الاتجاه (بالأرقام).
1) المراحل القديمة جاوبت "إيه أحسن موديل سحابة"، بس سؤال البروبوزال هو سحابة مقابل فيدرالي مقابل جهاز → عملنا دراسة نشر.
2) مستخدم = جهاز: في الداتا الخام الوسيط 6 أحداث، والأقصى 11,628، و473,402 مستخدم عندهم حدث واحد بس → أخدنا المستخدمين اللي عندهم 5–100 حدث: 1.45 مليون.
3) تكلفة الفيدرالي: 52.8 مليون قرار كبير جداً نحاكي عليه فيدرالي مرات كتير → main cohort 2.59 مليون (حوالي 20 مرة أصغر).
4) مقارنة غير عادلة: GRU و SASRec كانوا بفيتشرز وجداول تدريب مختلفة → مسابقة جديدة على بروتوكول واحد نظيف.
5) أسبوع الـ test القديم اتقرا مرة → مكتوب إنه اتشاف، ونظرة واحدة جديدة مقفولة.
6) الموارد: كان عندنا 2× A16 (15.4 جيجا لكل واحدة) و12.8 جيجا مساحة فاضية بس؛ دلوقتي معانا سيرفر 2× A40 والتجارب شغالة.`);
}

// 9. New setup: population funnel
{
  const s = light();
  header(s, "The new setup", "From raw users to the calibration set");
  s.addChart(pres.charts.BAR, [{ name: "Users", labels: ["Warm users (≥ 1 training event)", "Study pool (5–100 events)", "Full-pool cloud users", "Pretraining (opted-in) users", "Main cohort", "Calibration users (tuning only)"], values: [2694998, 1446986, 1348682, 1217610, 131072, 32768] }], {
    x: MX, y: 1.5, w: 8.6, h: 4.9, barDir: "bar", chartColors: [C.gray, C.slate, C.slate, C.amber, C.teal, C.tealDark], varyColors: true, barGapWidthPct: 35,
    showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "#,##0", dataLabelFontSize: 13, dataLabelColor: C.ink,
    valAxisHidden: true, valAxisMaxVal: 3300000, valGridLine: { style: "none" }, catGridLine: { style: "none" },
    catAxisLabelColor: C.body, catAxisLabelFontSize: 13, catAxisOrientation: "maxMin", showLegend: false,
    showTitle: true, title: "Number of users", titleFontSize: 13, titleColor: C.ink,
  });
  const x2 = 9.5, w2 = W - MX - x2;
  txt(s, "Split", x2, 1.6, w2, 0.35, { fontSize: 15, bold: true, color: C.ink });
  const sp = [["Train", "1 Oct – 15 Nov", C.slate], ["Validation", "16–22 Nov", C.teal], ["Test (sealed)", "23–29 Nov", C.amber]];
  sp.forEach((p, i) => { const y = 2.05 + i * 0.62; chip(s, p[0], x2, y, 1.55, p[2], i === 2 ? C.ink : C.white); txt(s, p[1], x2 + 1.7, y, w2 - 1.7, 0.34, { fontSize: 13.5, valign: "middle" }); });
  txt(s, bullets(["Pretraining users never overlap the main cohort.", "Up to 50 past events per prediction.", "Unseen-user panel 12,862; truly cold users 381,295."]), x2, 4.1, w2, 2.3, { fontSize: 12.5 });
  footer(s, "Planning evidence packet (data census); population decision; pretraining cohort build; realism audit D2");
  s.addNotes(`الشريحة 9 — الإعداد الجديد: من المستخدمين الخام لمجموعة الضبط.
2,694,998 مستخدم عندهم حدث تدريب واحد على الأقل ← 1,446,986 في الـ pool (5–100) ← 1,348,682 للسحابة الكاملة ← 1,217,610 للتدريب المسبق (مستخدمين موافقين، مش متداخلين مع الـ main) ← 131,072 main cohort ← 32,768 calibration (للضبط بس).
التقسيم: تدريب لحد 15 نوفمبر، validation 16–22، test 23–29 مقفول. كل توقع بيشوف لحد 50 حدث.
كمان: لوحة مستخدمين ما شاركوش 12,862، ومستخدمين باردين تماماً 381,295.`);
}

// 10. Discussion 1
discussion(2, "Is the change of course sound?", [
  ["Is limiting users to 5–100 events acceptable for the Edge story?", "It keeps each simulated device realistic; very heavy and single-event users are reported as separate panels."],
  ["Which cloud should be the reference: 131k matched users or the 1.35M full pool?", "We report both — which leads?"],
  ["Should the earlier full-data results appear in the thesis, or only as context?", "They are not comparable (different data size and features)."],
], `ديسكشن 2 — هل تغيير الاتجاه سليم؟
1) تحديد المستخدمين بـ 5–100 حدث مقبول لقصة Edge؟ بيخلي كل جهاز واقعي، والمستخدمين التقال جداً وأصحاب الحدث الواحد ليهم لوحات منفصلة.
2) المرجع: السحابة المطابقة (131 ألف) ولا الكاملة (1.35 مليون)؟
3) نتائج المرحلة القديمة تدخل الرسالة ولا تفضل context بس؟ مش قابلة للمقارنة.`, "earlier phases and the new setup");

// 11. Contest
{
  const s = light();
  header(s, "Current study — backbone contest", "SASRec wins under the registered rule; GRU at the fixed end");
  s.addChart(pres.charts.BAR, [
    { name: "GRU (21.0M)", labels: ["Best checkpoint within budget (registered rule)", "Fixed end of budget"], values: [0.1569, 0.1569] },
    { name: "SASRec (83.9M)", labels: ["Best checkpoint within budget (registered rule)", "Fixed end of budget"], values: [0.1612, 0.1527] },
  ], {
    x: MX, y: 1.5, w: 7.6, h: 4.9, barDir: "col", barGrouping: "clustered", chartColors: [C.slate, C.teal],
    showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0.0000", dataLabelFontSize: 14, dataLabelColor: C.ink,
    valAxisMinVal: 0.14, valAxisMaxVal: 0.165, valAxisLabelFormatCode: "0.000", valAxisLabelColor: C.muted, catAxisLabelColor: C.body, catAxisLabelFontSize: 13,
    valGridLine: { color: "E3E8EA", size: 0.5 }, catGridLine: { style: "none" },
    showLegend: true, legendPos: "t", legendFontSize: 13, legendColor: C.body,
    showTitle: true, title: "Validation MRR@20, 2-seed mean, main cohort (40,764 users)", titleFontSize: 13, titleColor: C.ink,
  });
  const x2 = 8.6, w2 = W - MX - x2;
  stat(s, x2, 1.6, w2, 1.55, "+0.0043", "SASRec, CI [+0.0033, +0.0052] — selected", C.tealLight, C.tealDark);
  stat(s, x2, 3.3, w2, 1.55, "+0.0042", "GRU at the fixed end, CI [+0.0033, +0.0052] — disclosed", C.slateLight, C.slate);
  stat(s, x2, 5.0, w2, 1.35, "0.1967 vs 0.1910", "NDCG@10, SASRec vs GRU");
  footer(s, "Contest result file; contest diagnostics (decision log, 28 Sep)");
  s.addNotes(`الشريحة 11 — مسابقة الموديل في الدراسة الحالية (8 تشغيلات، 40,764 مستخدم).
القاعدة المسجلة: SASRec 0.1612 مقابل GRU 0.1569 → SASRec +0.0043، فترة الثقة [+0.0033، +0.0052] → اتختار.
نهاية الميزانية الثابتة: GRU 0.1569 مقابل SASRec 0.1527 → GRU +0.0042 → مفصح عنه.
NDCG@10: 0.1967 مقابل 0.1910.
ملحوظة: 0.1612 هنا مش قابل للمقارنة بـ 0.1713 القديم (داتا أصغر حوالي 20 مرة وفيتشرز مختلفة).`);
}

// 12. Baselines and model sizes
{
  const s = light();
  header(s, "Current study — the bar to beat, and the device budget", "Simple rules reach 44–69% of the cloud");
  s.addChart(pres.charts.BAR, [{ name: "MRR@20", labels: ["Popularity", "Last viewed item", "Session kNN", "GRU", "SASRec (cloud)"], values: [0.01943, 0.07111, 0.11138, 0.1569, 0.1612] }], {
    x: MX, y: 1.5, w: 6.2, h: 4.9, barDir: "bar", chartColors: [C.gray, C.amber, C.gray, C.slate, C.teal], varyColors: true, barGapWidthPct: 40,
    showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0.0000", dataLabelFontSize: 13, dataLabelColor: C.ink,
    valAxisMinVal: 0, valAxisMaxVal: 0.2, valAxisHidden: true, valGridLine: { style: "none" }, catGridLine: { style: "none" },
    catAxisLabelColor: C.body, catAxisLabelFontSize: 13, catAxisOrientation: "maxMin", showLegend: false,
    showTitle: true, title: "Validation MRR@20, main cohort", titleFontSize: 13, titleColor: C.ink,
  });
  s.addChart(pres.charts.BAR, [{ name: "MB per visit", labels: ["Cloud-size model", "Device model", "8-bit uploads", "Frozen item tables"], values: [671.1, 165.7, 103.6, 2.1] }], {
    x: 7.0, y: 1.5, w: W - MX - 7.0, h: 4.9, barDir: "bar", chartColors: [C.slate, C.teal, C.teal, C.amber], varyColors: true, barGapWidthPct: 40,
    showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0.0", dataLabelFontSize: 13, dataLabelColor: C.ink,
    valAxisHidden: true, valAxisMaxVal: 800, valGridLine: { style: "none" }, catGridLine: { style: "none" }, catAxisLabelColor: C.body, catAxisLabelFontSize: 13, catAxisOrientation: "maxMin",
    showLegend: false, showTitle: true, title: "MB per federated visit (cap 170 MB)", titleFontSize: 13, titleColor: C.ink,
  });
  txt(s, "Parameters: 83.9M cloud vs 20.7M device. Frozen tables: ≈ 1 MB each way + one 81.78 MB download (to be re-measured).", MX, 6.45, W - 2 * MX, 0.35, { fontSize: 12, italic: true, color: C.muted });
  footer(s, "Decision log (baselines, 28 Sep); architecture lock (parameters, bytes); review synthesis §3.2");
  s.addNotes(`الشريحة 12 — الحد اللي لازم نعديه، وميزانية الجهاز.
شمال: popularity 0.0194، آخر منتج اتشاف 0.0711 (44% من السحابة)، session kNN 0.1114 (69%)، GRU 0.1569، SASRec 0.1612.
يمين: الميجابايت في الزيارة: موديل بحجم السحابة 671، موديل الجهاز 165.7 (تحت سقف 170)، رفع 8 بت 103.6، جداول مجمدة 2.1.
الباراميترز: 83.9 مليون مقابل 20.7 مليون.`);
}

// 13. FL ladder with warning
{
  const s = light();
  header(s, "Current study — federated learning", "Tuning so far: 0.0235 → 0.0288 (+22.7%)");
  s.addChart(pres.charts.LINE, [{ name: "Calibration MRR@20", labels: ["×0.5", "×1 (start)", "×2", "×4", "×8", "×16"], values: [0.020263, 0.023473, 0.025562, 0.027570, 0.028796, 0.022909] }], {
    x: MX, y: 1.5, w: 7.4, h: 4.9, chartColors: [C.teal], lineSize: 3, lineDataSymbol: "circle", lineDataSymbolSize: 9,
    showValue: true, dataLabelPosition: "t", dataLabelFormatCode: "0.0000", dataLabelFontSize: 12, dataLabelColor: C.ink,
    valAxisMinVal: 0.018, valAxisMaxVal: 0.031, valAxisLabelFormatCode: "0.000", valAxisLabelColor: C.muted, catAxisLabelColor: C.body, catAxisLabelFontSize: 13,
    valGridLine: { color: "E3E8EA", size: 0.5 }, catGridLine: { style: "none" }, showLegend: false,
    showTitle: true, title: "Client learning-rate multiplier, 32,768 calibration users, 1,707 rounds", titleFontSize: 13, titleColor: C.ink,
  });
  const x2 = 8.4, w2 = W - MX - x2;
  stat(s, x2, 1.6, w2 / 2 - 0.1, 1.95, "≈ 25%", "federated vs matched central (calibration, est.)", C.tealLight, C.tealDark);
  stat(s, x2 + w2 / 2 + 0.1, 1.6, w2 / 2 - 0.1, 1.95, "≈ 44%", "“last viewed item” vs cloud (main cohort)", C.amberLight, C.amberDark);
  txt(s, "Next stages", x2, 3.8, w2, 0.35, { fontSize: 14, bold: true, color: C.ink });
  const st = [["Server optimizer", "running"], ["LR shape × client optimizer", "next"], ["Local passes / batch", "next"], ["Pretrained start", "running"], ["Frozen item tables", "next"]];
  st.forEach((r, i) => { const y = 4.2 + i * 0.44; txt(s, r[0], x2, y, w2 - 1.1, 0.4, { fontSize: 13, valign: "middle" }); statusChip(s, r[1], x2 + w2 - 1.0, y + 0.03, 1.0, r[1]); });
  footer(s, "Improvement-ladder register; learning-rate selection files; review synthesis §1");
  s.addNotes(`الشريحة 13 — الفيدرالي.
منحنى الـ learning rate: 0.0203 / 0.0235 (البداية) / 0.0256 / 0.0276 / 0.0288 (الأحسن عند ×8) / 0.0229. يعني +22.7% عن البداية.
الإنذار: حوالي 25% من الموديل المركزي المطابق على calibration (المركزي تقدير 0.11–0.12)، في حين آخر منتج اتشاف حوالي 44% من السحابة. مجموعات مختلفة، فده إنذار مش نتيجة.
المراحل الجاية: server optimizer (شغال)، شكل الـ LR، الـ passes/batch، البدء من موديل مُدرب مسبقاً (شغال)، والجداول المجمدة. المرحلة بتتحفظ بس لو كسبت ≥ 0.002.`);
}

// 14. Discussion 2
discussion(3, "Federated effort and privacy level", [
  ["How much tuning effort is fair for federated learning?", "Our ladder stops when a stage gains < 0.002."],
  ["If tuned federated stays below 0.0711 (“last viewed item”), how do we present it?", "Plan: report it plainly as a finding about this recipe and budget."],
  ["Is DP at ε = 8 and ε = 1 enough, or do you want a curve?", "ε = 4 costs one more run (≈ half a day, estimate)."],
], `ديسكشن 3 — مجهود الفيدرالي ومستوى الخصوصية.
1) قد إيه ضبط عادل؟ السلم بيقف لو المرحلة كسبت أقل من 0.002.
2) لو الفيدرالي فضل تحت 0.0711، نعرضه إزاي؟ خطتنا: بوضوح كنتيجة عن الوصفة والميزانية.
3) ε=8 و ε=1 كفاية ولا منحنى؟ ε=4 تشغيل زيادة حوالي نص يوم (تقدير).`, "federated ladder and baselines");

// 15. Comparison board
{
  const s = light();
  header(s, "Current study — comparison board", "Every method, compared with what, and its status");
  const rows = [
    ["Cloud", C.slate, C.white, "Large cloud model", "reference", "0.1612 (contest)", "done"],
    ["Cloud", C.slate, C.white, "Full-pool cloud (1.35M users)", "reference 2", "training ends ≈ 30 Sep", "running"],
    ["Cloud", C.slate, C.white, "Small cloud model (device size)", "large cloud", "sealed", "sealed"],
    ["Federated", C.teal, C.white, "Federated (tuned), FedProx, personal vector, 8-bit uploads", "cloud; floor", "tuning stage 2 of 4", "running"],
    ["Federated + DP", C.tealDark, C.white, "DP at ε = 8 and ε = 1", "federated", "after tuning", "next"],
    ["Pretrained branch", C.amberDark, C.white, "Pretrained (1.22M users) → + federated → + frozen tables", "full-pool cloud", "pretraining running", "running"],
    ["On-device", C.amber, C.ink, "Local-only model; cloud / federated / pretrained model fine-tuned on device", "cloud; its own base", "device runs started", "running"],
    ["Baselines", C.gray, C.ink, "Last viewed item 0.0711 · session kNN 0.1114 · popularity 0.0194", "floor", "done", "done"],
  ];
  const cx = [MX, MX + 2.1, MX + 8.1, MX + 10.0], cwid = [2.0, 5.9, 1.8, 2.13];
  ["Level", "Method", "Compared with", "Status"].forEach((h, j) => txt(s, h, cx[j] + (j === 0 ? 0.1 : 0), 1.5, cwid[j], 0.3, { fontSize: 12, bold: true, color: C.muted }));
  const rh = 0.52;
  rows.forEach((r, i) => {
    const y = 1.88 + i * (rh + 0.06);
    s.addShape(pres.shapes.RECTANGLE, { x: MX, y, w: W - 2 * MX, h: rh, fill: { color: i % 2 ? C.white : C.panel }, line: { type: "none" } });
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: MX + 0.05, y: y + 0.08, w: 1.9, h: rh - 0.16, rectRadius: 0.06, fill: { color: r[1] }, line: { type: "none" } });
    s.addText(r[0], { x: MX + 0.05, y: y + 0.08, w: 1.9, h: rh - 0.16, fontFace: BF, fontSize: 11.5, bold: true, color: r[2], align: "center", valign: "middle", margin: 0, isTextBox: true });
    txt(s, r[3], cx[1], y, cwid[1], rh, { fontSize: 13, color: C.ink, valign: "middle" });
    txt(s, r[4], cx[2], y, cwid[2], rh, { fontSize: 12.5, valign: "middle" });
    statusChip(s, r[5], cx[3], y + 0.09, cwid[3], r[6]);
  });
  txt(s, "Sealed results are read together: validation ≈ 2–3 Oct, single test look ≈ 4–5 Oct (estimates).", MX, 6.62, W - 2 * MX, 0.35, { fontSize: 12.5, italic: true, bold: true, color: C.amberDark });
  footer(s, "Contest result; baselines (decision log 28 Sep); run status (decision log 29 Sep); master plan v5");
  s.addNotes(`الشريحة 15 — لوحة المقارنة في الدراسة الحالية.
السحابة الكبيرة هي المرجع (0.1612). السحابة الكاملة بتتدرب وتخلص حوالي 30 سبتمبر. السحابة الصغيرة مقفولة. الفيدرالي في مرحلة الضبط التانية من أربعة. DP بعد الضبط. الفرع المُدرب مسبقاً (1.22 مليون مستخدم) شغال. تشغيلات الجهاز بدأت. الـ baselines خلصت.
النتائج المقفولة كلها هتتقرا مع بعض: validation حوالي 2–3 أكتوبر، test حوالي 4–5 أكتوبر (تقديرات).`);
}

// 16. Privacy + deployment branch in numbers
{
  const s = light();
  header(s, "Current study — privacy and the realistic branch", "The key numbers");
  const st = [
    ["ε = 8 · ε = 1", "client-level DP; noise multipliers 0.5642 · 1.3094", C.tealLight, C.tealDark],
    ["δ = 5 × 10⁻⁶", "adding or removing one user's history", C.panel, C.slate],
    ["1,217,610", "opted-in users for pretraining (24.1M decisions)", C.amberLight, C.amberDark],
    ["265,162", "parameters uploaded with frozen tables (≈ 1.06 MB each way)", C.panel, C.slate],
    ["81.78 MB", "one-time table download", C.panel, C.slate],
    ["98%", "of device-model parameters are item tables → updates can reveal viewed items", C.amberLight, C.amberDark],
  ];
  const cw = (W - 2 * MX - 0.6) / 3, ch = 2.15;
  st.forEach((t, i) => stat(s, MX + (i % 3) * (cw + 0.3), 1.6 + Math.floor(i / 3) * (ch + 0.3), cw, ch, t[0], t[1], t[2], t[3]));
  txt(s, "Adversary: an honest-but-curious browser vendor. Secure aggregation is not simulated — stated as a limitation.", MX, 6.5, W - 2 * MX, 0.35, { fontSize: 12.5, italic: true, color: C.muted });
  footer(s, "Privacy statement and threat model; independent DP check; pretraining cohort build; review synthesis §3.2; realism audit");
  s.addNotes(`الشريحة 16 — الخصوصية والفرع الواقعي بالأرقام.
DP عند ε=8 و ε=1 على مستوى المستخدم، noise multipliers 0.5642 و 1.3094، و δ = 5×10⁻⁶.
التدريب المسبق على 1,217,610 مستخدم موافق (24.1 مليون قرار). مع الجداول المجمدة بنرفع 265,162 باراميتر بس (حوالي 1.06 ميجا في كل اتجاه)، وتحميل الجداول مرة واحدة 81.78 ميجا.
98% من باراميترز موديل الجهاز جداول منتجات، فالتحديثات من غير DP بتكشف المنتجات اللي المستخدم شافها. secure aggregation مش متحاكي — مكتوب كحدود.`);
}

// 17. Reviews scorecard
{
  const s = light();
  header(s, "Independent reviews", "7/10 overall; strongest on data and evaluation");
  const dims = [["Data", 8], ["Evaluation & statistics", 8], ["Features & history", 7], ["Backbone contest", 7], ["Central references", 7], ["Differential privacy", 7], ["Engineering", 7], ["Question & Edge framing", 6.5], ["Federated training", 6], ["Communication", 6], ["On-device arms", 6], ["Cold start & strata", 6], ["Thesis write-up", 6], ["Device measurement", 5]];
  s.addChart(pres.charts.BAR, [{ name: "Score / 10", labels: dims.map(d => d[0]), values: dims.map(d => d[1]) }], {
    x: MX, y: 1.5, w: 7.6, h: 5.0, barDir: "bar", chartColors: [C.teal], barGapWidthPct: 35,
    showValue: true, dataLabelPosition: "outEnd", dataLabelFontSize: 11, dataLabelColor: C.ink, dataLabelFormatCode: "0.0",
    valAxisMinVal: 0, valAxisMaxVal: 10, valAxisHidden: true, valGridLine: { style: "none" }, catGridLine: { style: "none" },
    catAxisLabelColor: C.body, catAxisLabelFontSize: 11, catAxisOrientation: "maxMin", showLegend: false,
    showTitle: true, title: "14 part scores (mean 6.96)", titleFontSize: 12, titleColor: C.ink,
  });
  const x2 = 8.6, w2 = W - MX - x2;
  stat(s, x2, 1.6, w2, 1.5, "7 → 8–8.5", "overall now → after the registered fixes", C.tealLight, C.tealDark);
  stat(s, x2, 3.3, w2, 1.5, "0", "evaluation or data-leak bugs found");
  stat(s, x2, 5.0, w2, 1.45, "3 of 5", "fix groups done or registered before any read", C.panel, C.slate);
  footer(s, "Review synthesis (§1 scorecard, §4 fix gates); decision log 29 Sep 17:59–18:36Z");
  s.addNotes(`الشريحة 17 — المراجعات المستقلة.
14 جزء، المتوسط 6.96 → 7/10. أعلى حاجة الداتا والتقييم (8)، وأقل حاجة قياس الجهاز (5).
مفيش ولا bug في التقييم ولا تسريب داتا.
مجموعات الإصلاح: 5 مجموعات (A لـ E). A و B و C اتعملوا أو اتسجلوا قبل أي قراءة؛ D قبل الـ test و E في الكتابة. الهدف 8–8.5.`);
}

// 18. Timeline
{
  const s = light();
  header(s, "When results arrive", "Four milestones (estimates)");
  const ms = [
    ["≈ 30 Sep", "Tuning and pretraining finish", "running"],
    ["≈ 30 Sep – 2 Oct", "Main federated + DP runs (≈ 10.7 h each)", "next"],
    ["≈ 2–3 Oct", "One validation read, then lock", "sealed"],
    ["≈ 4–5 Oct", "Single test look; tables ≈ 6–7 Oct", "sealed"],
  ];
  const cw = (W - 2 * MX - 0.9) / 4;
  s.addShape(pres.shapes.LINE, { x: MX + 0.3, y: 2.25, w: W - 2 * MX - 0.6, h: 0, line: { color: C.line, width: 3 } });
  ms.forEach((m, i) => {
    const x = MX + i * (cw + 0.3);
    s.addShape(pres.shapes.OVAL, { x: x + 0.1, y: 2.05, w: 0.4, h: 0.4, fill: { color: i === 0 ? C.teal : i === 1 ? C.slate : C.amber }, line: { color: C.white, width: 2 } });
    txt(s, m[0], x, 1.5, cw, 0.4, { fontSize: 18, bold: true, fontFace: HF, color: C.ink });
    card(s, x, 2.75, cw, 2.4, C.panel);
    txt(s, m[1], x + 0.25, 2.95, cw - 0.5, 1.3, { fontSize: 16, bold: true, color: C.ink });
    statusChip(s, m[2], x + 0.25, 4.55, 1.3, m[2]);
  });
  txt(s, "Plan said 1–2 Oct for the first read; ≈ 1 day slip because tuning screens run slower when sharing GPUs.", MX, 5.55, W - 2 * MX, 0.4, { fontSize: 13, italic: true, color: C.muted });
  footer(s, "Master plan v5 (cost model, test-look duration); ladder schedule; decision log 29 Sep");
  s.addNotes(`الشريحة 18 — إمتى النتائج (تقديرات).
حوالي 30 سبتمبر: الضبط والتدريب المسبق يخلصوا. 30 سبتمبر–2 أكتوبر: التشغيلات الفيدرالية و DP (كل واحد حوالي 10.7 ساعة، معدل مقاس). حوالي 2–3 أكتوبر: قراءة validation واحدة ثم قفل. حوالي 4–5 أكتوبر: نظرة test واحدة، والجداول حوالي 6–7 أكتوبر.
اتأخرنا حوالي يوم عن الخطة بسبب إن تشغيلات الضبط أبطأ وهي بتشارك الـ GPU.`);
}

// 19. Discussion 3
discussion(4, "Headline and priorities", [
  ["Which comparison should be the Edge headline?", "Plain federated vs cloud, or pretrained branch vs full-pool cloud (the review recommends the latter)."],
  ["If time is short, what do we protect first?", "Never cut: federated ×2, local-only ×2, cloud fine-tune ×2, INT8, the lock, the test look."],
  ["Can you pre-approve the Queen's AI-use disclosure?", "Required front matter per the independent review."],
], `ديسكشن 4 — الـ headline والأولويات.
1) أنهي مقارنة تبقى الإجابة الرئيسية لـ Edge؟ الفيدرالي العادي ولا الفرع المُدرب مسبقاً قدام السحابة الكاملة؟
2) لو الوقت ضيق نحمي إيه الأول؟ مش هنقطع: الفيدرالي ببذرتين، المحلي ببذرتين، fine-tune السحابة ببذرتين، INT8، القفل، ونظرة الـ test.
3) يوافق مسبقاً على بيان استخدام الذكاء الاصطناعي المطلوب من Queen's؟
سجل التحديثات: v1-numbers — 2026-09-29، نفس الحالة (DECISION_LOG لحد الصف 228، السلم لحد S1b).`, "cut order is pre-registered in the master plan");

pres.writeFile({ fileName: OUT }).then(f => console.log("wrote " + f + " slides=" + pageNo));
