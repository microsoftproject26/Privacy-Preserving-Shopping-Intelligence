// Supervisor deck generator — SHORT comparison-focused version (pptxgenjs). Every number here is listed with its source in SUPERVISOR_DECK_DATA.md.
const pptxgen = require("pptxgenjs");
const path = require("path");

const OUT = process.argv[2] || path.join(__dirname, "PPSI_Supervisor_Deck_Simple.pptx");
const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE"; // 13.333 x 7.5 in
pres.title = "Privacy-Preserving Shopping Intelligence — supervisor update (short, comparison-focused)";
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

// ===================== SIMPLE VERSION: comparison-focused =====================
function statusChip(s, t, x, y, w, kind) {
  const m = { done: [C.slate, C.white], running: [C.teal, C.white], sealed: [C.amberLight, C.amberDark], next: [C.slateLight, C.ink] }[kind];
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h: 0.34, rectRadius: 0.17, fill: { color: m[0] }, line: kind === "sealed" ? { color: C.amber, width: 1 } : { type: "none" } });
  s.addText(t, { x, y, w, h: 0.34, fontFace: BF, fontSize: 11, bold: true, color: m[1], align: "center", valign: "middle", margin: 0, isTextBox: true });
}

// 1. Title
{
  const s = dark();
  s.addText("PRIVACY-PRESERVING SHOPPING INTELLIGENCE", { x: MX, y: 1.4, w: 11, h: 0.4, fontFace: BF, fontSize: 14, bold: true, charSpacing: 3, color: C.amber, margin: 0, isTextBox: true });
  s.addText("Cloud vs federated vs on-device: what we compare, and what we know so far", { x: MX, y: 1.9, w: 11.5, h: 1.9, fontFace: HF, fontSize: 40, bold: true, color: C.white, margin: 0, valign: "top", isTextBox: true });
  s.addText("Short version — supervisor meeting, 30 September 2026", { x: MX, y: 4.0, w: 11, h: 0.5, fontFace: BF, fontSize: 18, color: C.onDarkMuted, margin: 0, isTextBox: true });
  const lbl = [["Cloud", C.slate], ["Federated", C.teal], ["On-device", C.amber]];
  lbl.forEach((l, i) => {
    s.addShape(pres.shapes.OVAL, { x: MX + i * 2.2, y: 4.9, w: 0.28, h: 0.28, fill: { color: l[1] }, line: { type: "none" } });
    s.addText(l[0], { x: MX + 0.4 + i * 2.2, y: 4.87, w: 1.7, h: 0.34, fontFace: BF, fontSize: 14, color: C.onDark, margin: 0, isTextBox: true });
  });
  s.addText("Ahmed Abdelhamed · MSc, Queen's University · facts as recorded 29 Sep 2026, 18:36 UTC", { x: MX, y: 6.5, w: 11, h: 0.35, fontFace: BF, fontSize: 13, color: C.onDarkMuted, margin: 0, isTextBox: true });
  pageNo += 1;
  s.addNotes(`الشريحة 1 — العنوان (النسخة المختصرة).
قول: النسخة دي مركزة على المقارنات: بنقارن إيه بإيه، وإيه اللي اتقاس فعلاً، وإيه اللي لسه مقفول ومعاده المتوقع.
مهم: نتائج الطرق الرئيسية (الفيدرالي، الجهاز، السحابة الكاملة) مقفولة لحد القراءة المسجلة مسبقاً، فهتلاقيها مكتوبة "sealed" مع تاريخ متوقع (تقدير).`);
}

// 2. The three deployments side by side
{
  const s = light();
  header(s, "The question", "Three deployments, compared on the same users");
  const rowsL = ["What leaves the device", "Model", "Privacy", "Main risk"];
  const cols = [
    ["CLOUD", C.slate, C.slateLight, ["Raw browsing events", "Large (83.9M parameters)", "None — server sees everything", "Privacy"]],
    ["FEDERATED", C.teal, C.tealLight, ["Model updates only", "Device-size (20.7M)", "Optional differential privacy", "Quality and bandwidth"]],
    ["ON-DEVICE", C.amberDark, C.amberLight, ["Nothing (after one download)", "Device-size (20.7M)", "Data never leaves", "Too little data per user"]],
  ];
  const lx = MX, lw = 2.6, cx0 = MX + lw + 0.2, cw = (W - MX - cx0 - 0.4) / 3;
  cols.forEach((c, i) => {
    const x = cx0 + i * (cw + 0.2);
    card(s, x, 1.6, cw, 4.55, c[2]);
    txt(s, c[0], x, 1.75, cw, 0.35, { fontSize: 14, bold: true, charSpacing: 2, color: c[1], align: "center" });
    c[3].forEach((v, j) => txt(s, v, x + 0.25, 2.35 + j * 0.95, cw - 0.5, 0.8, { fontSize: 16, color: C.ink, align: "center", valign: "middle" }));
  });
  rowsL.forEach((r, j) => txt(s, r, lx, 2.35 + j * 0.95, lw, 0.8, { fontSize: 15, bold: true, color: C.muted, valign: "middle" }));
  txt(s, [{ text: "The number we report: ", options: { bold: true, color: C.ink } }, { text: "retention = method ÷ cloud, on the same users and the same test week, with a confidence interval." }], MX, 6.35, W - 2 * MX, 0.4, { fontSize: 15 });
  footer(s, "Research proposal §1; study description; architecture lock (parameters)");
  s.addNotes(`الشريحة 2 — الثلاث طرق جنب بعض.
السحابة: الأحداث الخام بتروح للسيرفر، موديل كبير 83.9 مليون باراميتر، مفيش خصوصية.
الفيدرالي: تحديثات الموديل بس، موديل بحجم الجهاز 20.7 مليون، و DP اختياري. الخطر: الجودة والباندويدث.
على الجهاز: مفيش حاجة بتطلع بعد التحميل، نفس الحجم. الخطر: داتا المستخدم الواحد قليلة.
الرقم اللي بنقارن بيه: retention = الطريقة ÷ السحابة، على نفس المستخدمين ونفس أسبوع الـ test، بفترة ثقة.`);
}

// 3. The four headline comparisons
{
  const s = light();
  header(s, "What we compare", "Four comparisons answer the question");
  // reference box
  card(s, MX, 1.65, 3.3, 4.5, C.ink);
  txt(s, "REFERENCE", MX + 0.3, 1.85, 2.7, 0.3, { fontSize: 12, bold: true, charSpacing: 2, color: C.amber });
  txt(s, "Cloud model", MX + 0.3, 2.2, 2.7, 0.5, { fontSize: 22, bold: true, fontFace: HF, color: C.white });
  txt(s, bullets(["Matched: same 131k users", "Full pool: 1.35M users (what Edge would run)"], { color: C.onDark }), MX + 0.3, 2.85, 2.8, 1.5, { fontSize: 14, color: C.onDark });
  txt(s, "FLOOR", MX + 0.3, 4.45, 2.7, 0.3, { fontSize: 12, bold: true, charSpacing: 2, color: C.amber });
  txt(s, "“Last viewed item” rule — no training, nothing uploaded", MX + 0.3, 4.8, 2.8, 1.0, { fontSize: 14, color: C.onDark });
  const cmp = [
    ["1", "Federated  vs  cloud", "How much quality is lost when events stay on the device?", C.tealLight, C.tealDark],
    ["2", "Federated + DP  vs  federated", "What does a formal privacy guarantee cost? (ε = 8, ε = 1)", C.tealLight, C.tealDark],
    ["3", "On-device  vs  cloud", "How much survives with no upload at all (local model, local fine-tuning)?", C.amberLight, C.amberDark],
    ["4", "Pretrained branch  vs  full-pool cloud", "Realistic Edge: pretrain on opted-in users → federated on private users.", C.panel, C.ink],
  ];
  const x2 = MX + 3.7, w2 = W - MX - x2, ch = 1.0;
  cmp.forEach((c, i) => {
    const y = 1.65 + i * (ch + 0.17);
    card(s, x2, y, w2, ch, c[3]);
    s.addShape(pres.shapes.OVAL, { x: x2 + 0.25, y: y + 0.22, w: 0.56, h: 0.56, fill: { color: c[4] }, line: { type: "none" } });
    s.addText(c[0], { x: x2 + 0.25, y: y + 0.22, w: 0.56, h: 0.56, fontFace: HF, fontSize: 18, bold: true, color: C.white, align: "center", valign: "middle", margin: 0, isTextBox: true });
    txt(s, c[1], x2 + 1.05, y + 0.12, 4.0, 0.76, { fontSize: 18, bold: true, fontFace: HF, color: C.ink, valign: "middle" });
    txt(s, c[2], x2 + 5.1, y + 0.12, w2 - 5.3, 0.76, { fontSize: 14, valign: "middle" });
  });
  txt(s, "Every method is also compared with the floor: gain over “last viewed item”, with a confidence interval.", x2, 6.4, w2, 0.35, { fontSize: 13, italic: true, color: C.muted });
  footer(s, "Pre-registered contrast list (as revised after the independent review)");
  s.addNotes(`الشريحة 3 — الأربع مقارنات اللي بتجاوب على السؤال.
المرجع: موديل السحابة بنسختين: المطابقة (نفس 131 ألف مستخدم) والكاملة (1.35 مليون، ده اللي Edge هيشغله فعلاً).
الأرضية: قاعدة "آخر منتج اتشاف" — من غير تدريب ومن غير رفع أي حاجة.
1) الفيدرالي مقابل السحابة: قد إيه بنخسر لما الأحداث تفضل على الجهاز.
2) الفيدرالي + DP مقابل الفيدرالي: تمن الضمان الرسمي للخصوصية عند ε=8 و ε=1.
3) على الجهاز مقابل السحابة: قد إيه بيفضل من غير أي رفع.
4) الفرع المُدرب مسبقاً مقابل السحابة الكاملة: السيناريو الواقعي لـ Edge.
وكل طريقة كمان بتتقارن بالأرضية.
سؤال محتمل: "دي مسجلة مسبقاً؟" — أيوه، قايمة المقارنات اتسجلت قبل أي قراءة.`);
}

// 4. The comparison board
{
  const s = light();
  header(s, "Comparison board", "Every method: what it is compared with, and where it stands");
  const rows = [
    ["Cloud", C.slate, C.white, "Large cloud model", "reference", "0.1612 (contest)", "done"],
    ["Cloud", C.slate, C.white, "Full-pool cloud (1.35M users)", "reference 2", "training ends ≈ 30 Sep", "running"],
    ["Cloud", C.slate, C.white, "Small cloud model (device size)", "large cloud", "sealed", "sealed"],
    ["Federated", C.teal, C.white, "Federated (tuned), FedProx, personal vector, 8-bit uploads", "cloud; floor", "tuning stage 2 of 4", "running"],
    ["Federated + DP", C.tealDark, C.white, "DP at ε = 8 and ε = 1", "federated", "after tuning", "next"],
    ["Pretrained branch", C.amberDark, C.white, "Pretrained → + federated → + frozen tables", "full-pool cloud", "pretraining running", "running"],
    ["On-device", C.amber, C.ink, "Local-only model; cloud / federated / pretrained model fine-tuned on device", "cloud; its own base", "device runs started", "running"],
    ["Baselines", C.gray, C.ink, "Last viewed item 0.0711 · session kNN 0.1114 · popularity 0.0194", "floor", "done", "done"],
  ];
  const hdr = ["Level", "Method", "Compared with", "Status"];
  const cx = [MX, MX + 2.1, MX + 8.1, MX + 10.0], cwid = [2.0, 5.9, 1.8, 2.13];
  hdr.forEach((h, j) => txt(s, h, cx[j] + (j === 0 ? 0.1 : 0), 1.5, cwid[j], 0.3, { fontSize: 12, bold: true, color: C.muted }));
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
  txt(s, "All sealed results are read together: validation ≈ 2–3 Oct, single test look ≈ 4–5 Oct (estimates).", MX, 6.62, W - 2 * MX, 0.35, { fontSize: 12.5, italic: true, color: C.amberDark, bold: true });
  footer(s, "Contest result; baselines (decision log 28 Sep); run status (decision log 29 Sep); master plan v5");
  s.addNotes(`الشريحة 4 — لوحة المقارنة (أهم شريحة).
كل صف: الطريقة، بتتقارن بإيه، وحالتها.
- موديل السحابة الكبير: هو المرجع، قيمته 0.1612 من المسابقة (خلص).
- السحابة الكاملة (1.35 مليون مستخدم): التدريب شغال ومتوقع يخلص حوالي 30 سبتمبر (تقدير).
- موديل السحابة الصغير: خلص تدريبه بس نتيجته مقفولة.
- الفيدرالي بأنواعه: لسه في مرحلة الضبط التانية من أربعة.
- DP عند ε=8 و ε=1: بعد الضبط.
- الفرع المُدرب مسبقاً: التدريب المسبق شغال.
- على الجهاز: تشغيلات الجهاز بدأت 29 سبتمبر.
- الـ baselines: خلصت — آخر منتج 0.0711، session kNN 0.1114، popularity 0.0194.
كل النتائج المقفولة هتتقرا مع بعض: validation حوالي 2–3 أكتوبر، والـ test مرة واحدة حوالي 4–5 أكتوبر (تقديرات).`);
}

// 5. Discussion 1
discussion(1, "Which comparison should lead the Edge answer?", [
  ["Which of the four comparisons should be the headline for Edge?", "Plain federated vs cloud, or the realistic pretrained branch vs the full-pool cloud (the independent review recommends the latter)."],
  ["Against which cloud: the matched 131k-user model or the full-pool 1.35M-user model?", "We report both; the question is which one leads."],
  ["Should “last viewed item” be the floor every method must beat?", "It needs no training and sends nothing."],
], `ديسكشن 1 — أنهي مقارنة تبقى الإجابة الرئيسية؟
1) من الأربعة، أنهي واحدة الـ headline لـ Edge؟ الفيدرالي العادي مقابل السحابة، ولا الفرع الواقعي مقابل السحابة الكاملة؟ المراجعة المستقلة بتقترح التاني.
2) نقارن بأنهي سحابة: المطابقة ولا الكاملة؟ هنعرض الاتنين، بس مين الأول؟
3) هل "آخر منتج اتشاف" يبقى الأرضية اللي كل طريقة لازم تغلبها؟`, "four comparisons and the floor, previous slides");

// 6. Comparison: GRU vs SASRec (both rules)
{
  const s = light();
  header(s, "Comparison 1 — model family", "SASRec vs GRU: the winner depends on the checkpoint rule");
  s.addChart(pres.charts.BAR, [
    { name: "GRU (21.0M)", labels: ["Best checkpoint within budget — the registered rule", "Fixed end of budget"], values: [0.1569, 0.1569] },
    { name: "SASRec (83.9M)", labels: ["Best checkpoint within budget — the registered rule", "Fixed end of budget"], values: [0.1612, 0.1527] },
  ], {
    x: MX, y: 1.5, w: 7.6, h: 4.9, barDir: "col", barGrouping: "clustered", chartColors: [C.slate, C.teal],
    showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0.0000", dataLabelFontSize: 14, dataLabelColor: C.ink,
    valAxisMinVal: 0.14, valAxisMaxVal: 0.165, valAxisLabelFormatCode: "0.000", valAxisLabelColor: C.muted, catAxisLabelColor: C.body, catAxisLabelFontSize: 13,
    valGridLine: { color: "E3E8EA", size: 0.5 }, catGridLine: { style: "none" },
    showLegend: true, legendPos: "t", legendFontSize: 13, legendColor: C.body,
    showTitle: true, title: "Validation MRR@20, 2-seed mean, main cohort", titleFontSize: 13, titleColor: C.ink,
  });
  const x2 = 8.6, w2 = W - MX - x2;
  const k = [["SASRec +0.0043", "under the registered rule, CI [+0.0033, +0.0052] → selected", C.tealLight, C.tealDark], ["GRU +0.0042", "at the fixed end, CI [+0.0033, +0.0052] → disclosed", C.slateLight, C.slate]];
  k.forEach((t, i) => {
    const y = 1.65 + i * 1.75;
    card(s, x2, y, w2, 1.5, t[2]);
    txt(s, t[0], x2 + 0.3, y + 0.15, w2 - 0.6, 0.6, { fontFace: HF, fontSize: 26, bold: true, color: t[3] });
    txt(s, t[1], x2 + 0.3, y + 0.8, w2 - 0.6, 0.6, { fontSize: 13.5 });
  });
  txt(s, "Both rules were fixed before any result. The thesis says “selected under the pre-registered rule”, not “SASRec is better”.", x2, 5.3, w2, 1.1, { fontSize: 13.5, italic: true, color: C.muted });
  footer(s, "Contest result file; contest diagnostics (decision log, 28 Sep)");
  s.addNotes(`الشريحة 6 — المقارنة 1: عائلة الموديل.
بالقاعدة المسجلة (أحسن checkpoint داخل الميزانية): SASRec 0.1612 مقابل GRU 0.1569، يعني SASRec +0.0043 وفترة الثقة من +0.0033 لـ +0.0052، فاتختار.
عند نهاية الميزانية الثابتة: GRU 0.1569 مقابل SASRec 0.1527، يعني GRU +0.0042 وفترة الثقة من +0.0033 لـ +0.0052. ده مفصح عنه.
السبب: SASRec بيوصل لأحسن نقطة في النص وبعدين بينزل، و GRU لسه بيتحسن.
كمان مش نفس الحجم (83.9M مقابل 21.0M)، وفي مقارنة وصفية بنفس الحجم متسجلة.
الرسالة هتقول "اتختار بالقاعدة المسجلة" مش "SASRec أحسن".`);
}

// 7. Comparison: models vs simple baselines
{
  const s = light();
  header(s, "Comparison 2 — models vs simple rules", "What a trained model adds over simple rules");
  s.addChart(pres.charts.BAR, [{ name: "MRR@20", labels: ["Popularity", "Last viewed item", "Session kNN", "GRU", "SASRec (cloud reference)"], values: [0.01943, 0.07111, 0.11138, 0.1569, 0.1612] }], {
    x: MX, y: 1.5, w: 8.4, h: 4.9, barDir: "bar", chartColors: [C.gray, C.amber, C.gray, C.slate, C.teal], varyColors: true, barGapWidthPct: 40,
    showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0.0000", dataLabelFontSize: 14, dataLabelColor: C.ink,
    valAxisMinVal: 0, valAxisMaxVal: 0.19, valAxisHidden: true, valGridLine: { style: "none" }, catGridLine: { style: "none" },
    catAxisLabelColor: C.body, catAxisLabelFontSize: 14, catAxisOrientation: "maxMin", showLegend: false,
    showTitle: true, title: "Validation MRR@20, main cohort", titleFontSize: 13, titleColor: C.ink,
  });
  const x2 = 9.4, w2 = W - MX - x2;
  const st = [["44%", "of the cloud reached by “last viewed item” — for free"], ["69%", "of the cloud reached by session kNN"]];
  st.forEach((t, i) => {
    const y = 1.65 + i * 1.9;
    card(s, x2, y, w2, 1.65, i ? C.panel : C.amberLight);
    txt(s, t[0], x2 + 0.3, y + 0.15, w2 - 0.6, 0.7, { fontFace: HF, fontSize: 36, bold: true, color: i ? C.slate : C.amberDark });
    txt(s, t[1], x2 + 0.3, y + 0.9, w2 - 0.6, 0.7, { fontSize: 13.5 });
  });
  txt(s, "So the bar for any private method is high: it must beat 0.0711 to be worth shipping.", x2, 5.6, w2, 0.8, { fontSize: 13.5, italic: true, color: C.muted });
  footer(s, "Decision log (post-contest reported read of baselines and contest runs, 28 Sep)");
  s.addNotes(`الشريحة 7 — المقارنة 2: الموديلات مقابل القواعد البسيطة.
popularity 0.0194، آخر منتج اتشاف 0.0711، session kNN 0.1114، GRU 0.1569، SASRec 0.1612.
آخر منتج اتشاف بيوصل لحوالي 44% من السحابة ببلاش (0.0711 ÷ 0.1612). session kNN حوالي 69%.
الرسالة: أي طريقة خاصة لازم تغلب 0.0711 علشان يبقى ليها لازمة.
الأرقام دي مسموحة: baselines ومسابقة.`);
}

// 8. Comparison: cloud vs device model cost
{
  const s = light();
  header(s, "Comparison 3 — cost on the device", "Cloud-size vs device-size model: megabytes per visit");
  s.addChart(pres.charts.BAR, [{ name: "MB per visit", labels: ["Cloud-size model", "Device model", "Device model, 8-bit uploads", "Frozen item tables"], values: [671.1, 165.7, 103.6, 2.1] }], {
    x: MX, y: 1.5, w: 8.4, h: 4.9, barDir: "bar", chartColors: [C.slate, C.teal, C.teal, C.amber], varyColors: true, barGapWidthPct: 40,
    showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0.0", dataLabelFontSize: 14, dataLabelColor: C.ink,
    valAxisHidden: true, valGridLine: { style: "none" }, catGridLine: { style: "none" }, catAxisLabelColor: C.body, catAxisLabelFontSize: 14, catAxisOrientation: "maxMin",
    showLegend: false, showTitle: true, title: "MB per federated visit (download + upload)", titleFontSize: 13, titleColor: C.ink,
  });
  const x2 = 9.4, w2 = W - MX - x2;
  card(s, x2, 1.65, w2, 1.65, C.tealLight);
  txt(s, "170 MB", x2 + 0.3, 1.8, w2 - 0.6, 0.7, { fontFace: HF, fontSize: 36, bold: true, color: C.tealDark });
  txt(s, "device cap per visit — only the device-size model fits", x2 + 0.3, 2.55, w2 - 0.6, 0.7, { fontSize: 13.5 });
  txt(s, bullets(["Parameters: 83.9M cloud vs 20.7M device.", "Frozen tables: ≈ 1 MB each way plus one 81.78 MB download (to be re-measured).", "Quality of each option: sealed until the first read."]), x2, 3.6, w2, 2.8, { fontSize: 13.5 });
  footer(s, "Architecture lock (bytes per visit); realism audit B12; review synthesis §3.2");
  s.addNotes(`الشريحة 8 — المقارنة 3: التكلفة على الجهاز.
الموديل بحجم السحابة: 671 ميجا في الزيارة — مستحيل. موديل الجهاز: 165.7 ميجا (تحت سقف 170). الرفع بـ 8 بت: حوالي 103.6. الجداول المجمدة: حوالي 1 ميجا في كل اتجاه (2.1 رايح جاي) + تحميل مرة واحدة 81.78 ميجا (هيتقاس تاني).
جودة كل خيار لسه مقفولة. المقارنة هنا بين التكاليف بس.
سؤال محتمل: "165 ميجا كتير؟" — أيوه للموبايل، مقبول للابتوب؛ علشان كده عندنا 8 بت والجداول المجمدة.`);
}

// 9. Comparison: federated vs cloud (calibration warning)
{
  const s = light();
  header(s, "Comparison 4 — federated vs cloud (early warning)", "Untuned federated learning was far below the cloud");
  const st = [
    ["≈ 25%", "Federated (best stage so far) vs a matched central model", "calibration users · 0.0288 vs ≈ 0.11–0.12 (central is an estimate)", C.tealLight, C.tealDark],
    ["≈ 44%", "“Last viewed item” vs the cloud model", "main validation users · 0.0711 vs 0.1612", C.amberLight, C.amberDark],
  ];
  st.forEach((t, i) => {
    const x = MX + i * 6.15;
    card(s, x, 1.7, 5.9, 2.7, t[3]);
    txt(s, t[0], x + 0.4, 1.9, 5.1, 0.9, { fontFace: HF, fontSize: 54, bold: true, color: t[4] });
    txt(s, t[1], x + 0.4, 2.95, 5.1, 0.6, { fontSize: 17, bold: true, color: C.ink });
    txt(s, t[2], x + 0.4, 3.6, 5.1, 0.6, { fontSize: 13.5, italic: true, color: C.muted });
  });
  card(s, MX, 4.7, W - 2 * MX, 1.65, C.panel);
  txt(s, "Why, and what we did", MX + 0.35, 4.85, 6, 0.4, { fontSize: 16, bold: true, fontFace: HF, color: C.ink });
  txt(s, "The federated recipe was the cause, not a bug: no server optimizer, a client optimizer restarted every visit, and too small a learning rate. We register a tuning ladder on calibration users only. Different users and budgets, so this is a warning, not a result.", MX + 0.35, 5.25, W - 2 * MX - 0.7, 1.0, { fontSize: 14 });
  footer(s, "Improvement-ladder register; review synthesis §1; baselines and contest (decision log 28 Sep)");
  s.addNotes(`الشريحة 9 — المقارنة 4: الفيدرالي مقابل السحابة (إنذار مبكر).
الفيدرالي في أحسن مرحلة لحد دلوقتي 0.0288 على calibration users، والموديل المركزي المطابق حوالي 0.11–0.12 (تقدير) → حوالي 25%.
في المقابل آخر منتج اتشاف حوالي 44% من السحابة.
قول بوضوح: مجموعات مستخدمين وميزانيات مختلفة، فده إنذار مش نتيجة.
السبب الوصفة: مفيش server optimizer، الـ optimizer بيبدأ من الصفر كل زيارة، والـ learning rate صغير. الحل: سلم ضبط مسجل على calibration users بس.`);
}

// 10. Comparison: federated before vs after tuning
{
  const s = light();
  header(s, "Comparison 5 — federated before vs after tuning", "Each tuning stage vs the one before: +22.7% so far");
  s.addChart(pres.charts.BAR, [{ name: "Calibration MRR@20", labels: ["Registered start", "Client LR ×2", "Client LR ×8"], values: [0.023473, 0.025562, 0.028796] }], {
    x: MX, y: 1.5, w: 6.8, h: 4.9, barDir: "col", chartColors: ["9FB7BA", "5E9DA1", C.teal], varyColors: true, barGapWidthPct: 40,
    showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0.0000", dataLabelFontSize: 15, dataLabelColor: C.ink,
    valAxisMinVal: 0, valAxisMaxVal: 0.032, valAxisLabelFormatCode: "0.000", valAxisLabelColor: C.muted, catAxisLabelColor: C.body, catAxisLabelFontSize: 14,
    valGridLine: { color: "E3E8EA", size: 0.5 }, catGridLine: { style: "none" }, showLegend: false,
    showTitle: true, title: "Calibration users, validation MRR@20", titleFontSize: 13, titleColor: C.ink,
  });
  const x2 = 7.6, w2 = W - MX - x2;
  txt(s, "Stages", x2, 1.55, w2, 0.4, { fontSize: 16, bold: true, fontFace: HF, color: C.ink });
  const st = [
    ["Client learning rate", "+8.9%, then +12.7%", "done"], ["Server optimizer", "FedAdam / momentum", "running"],
    ["LR shape × client optimizer", "", "next"], ["Local passes and batch", "", "next"],
    ["Start from a pretrained model", "pretraining", "running"], ["Freeze the item tables", "", "next"],
  ];
  st.forEach((r, i) => {
    const y = 2.05 + i * 0.62;
    txt(s, r[0], x2, y, 2.75, 0.5, { fontSize: 14, color: C.ink, valign: "middle" });
    txt(s, r[1], x2 + 2.75, y, 1.5, 0.5, { fontSize: 11.5, color: C.muted, valign: "middle" });
    statusChip(s, r[2], x2 + 4.3, y + 0.08, 0.83, r[2]);
  });
  txt(s, "A stage is kept only if it gains ≥ 0.002. Main-cohort values of each stage are sealed until the final read.", x2, 5.85, w2, 0.6, { fontSize: 12.5, italic: true, color: C.muted });
  footer(s, "Improvement-ladder register; ladder stage registrations; decision log 29 Sep");
  s.addNotes(`الشريحة 10 — المقارنة 5: الفيدرالي قبل وبعد الضبط.
البداية المسجلة 0.023473 ← learning rate ×2 = 0.025562 (+8.9%) ← ×8 = 0.028796 (+12.7%). الإجمالي +22.7%.
المراحل الجاية: server optimizer (شغال دلوقتي)، شكل الـ LR مع optimizer العميل، الـ passes والـ batch، البدء من موديل مُدرب مسبقاً (التدريب المسبق شغال)، وتجميد جداول المنتجات.
القاعدة: المرحلة بتتحفظ بس لو كسبت ≥ 0.002.
تقديرات البحث (مش نتائج، لو اتسألت بس): مرحلة الـ server optimizer ممكن توصل حوالي 0.045–0.065.`);
}

// 11. Comparison: privacy levels
{
  const s = light();
  header(s, "Comparison 6 — the cost of privacy", "No DP vs ε = 8 vs ε = 1, all on the same federated recipe");
  const lv = [
    ["No DP", "ε = ∞", "Updates can reveal which products a user viewed", C.slateLight, C.slate],
    ["Moderate", "ε = 8", "Noise multiplier 0.5642", C.tealLight, C.teal],
    ["Strong", "ε = 1", "Noise multiplier 1.3094", C.tealLight, C.tealDark],
  ];
  const cw = (W - 2 * MX - 0.6) / 3;
  lv.forEach((l, i) => {
    const x = MX + i * (cw + 0.3);
    card(s, x, 1.65, cw, 2.6, l[3]);
    txt(s, l[0].toUpperCase(), x + 0.3, 1.85, cw - 0.6, 0.3, { fontSize: 12, bold: true, charSpacing: 2, color: l[4] });
    txt(s, l[1], x + 0.3, 2.2, cw - 0.6, 0.8, { fontFace: HF, fontSize: 40, bold: true, color: C.ink });
    txt(s, l[2], x + 0.3, 3.1, cw - 0.6, 0.9, { fontSize: 14 });
    statusChip(s, "quality: sealed", x + 0.3, 3.8, 1.8, "sealed");
  });
  s.addShape(pres.shapes.RIGHT_ARROW, { x: MX, y: 4.5, w: W - 2 * MX, h: 0.35, fill: { color: C.tealLight }, line: { type: "none" } });
  txt(s, "more privacy →", MX + 0.2, 4.5, 3, 0.35, { fontSize: 12, bold: true, color: C.tealDark, valign: "middle" });
  txt(s, bullets([
    "Guarantee: one user's whole history barely changes the released model (client-level, δ = 5 × 10⁻⁶).",
    "Adversary: an honest-but-curious browser vendor. Secure aggregation is not simulated — stated as a limitation.",
  ]), MX, 5.1, W - 2 * MX, 1.3, { fontSize: 14 });
  footer(s, "Privacy statement and threat model (registered); independent check of the DP noise multipliers");
  s.addNotes(`الشريحة 11 — المقارنة 6: تمن الخصوصية.
نفس وصفة الفيدرالي بثلاث مستويات: من غير DP (التحديثات ممكن تكشف المنتجات اللي المستخدم شافها)، ε=8 (noise multiplier 0.5642)، ε=1 (1.3094).
الجودة لكل مستوى لسه مقفولة.
الضمان: تاريخ مستخدم كامل مش بيغير الموديل تقريباً، δ = 5×10⁻⁶. الخصم: شركة المتصفح أمينة وفضولية. secure aggregation مش متحاكي ومكتوب كحدود.`);
}

// 12. Discussion 2
discussion(2, "How far should each comparison go?", [
  ["Is ε = 8 and ε = 1 enough, or do you want a privacy curve?", "Adding ε = 4 costs one more registered run (about half a day, estimate)."],
  ["How much tuning effort is fair for federated learning?", "Our ladder stops when a stage gains < 0.002. Stop earlier and spend the time on the pretrained branch?"],
  ["If tuned federated is still below “last viewed item”, how do we present it?", "Our plan: report it plainly as a finding about this recipe and budget."],
], `ديسكشن 2 — كل مقارنة تروح لحد فين؟
1) ε=8 و ε=1 كفاية ولا محتاج منحنى؟ ε=4 تشغيل زيادة حوالي نص يوم (تقدير).
2) قد إيه ضبط للفيدرالي عادل؟ السلم بيقف لو المرحلة كسبت أقل من 0.002. نقف بدري ونركز على الفرع المُدرب مسبقاً؟
3) لو الفيدرالي بعد الضبط لسه تحت "آخر منتج اتشاف"، نعرضه إزاي؟ خطتنا: نقوله بوضوح كنتيجة عن الوصفة والميزانية دي.`, "comparisons 4–6");

// 13. Proposal vs now (short)
{
  const s = light();
  header(s, "Proposal vs now", "What changed, in one view");
  const rows = [
    ["Dataset", "Criteo, MovieLens, Amazon, Taobao, RetailRocket", "REES46 shopping events (per-user, timestamped)"],
    ["Task", "Next category, purchase, offers", "Next product viewed, over all 158,486 products"],
    ["Metric", "NDCG@10, HR@10, AUC", "MRR@20 per user (+ NDCG@10, HR@10)"],
    ["Federated", "FedAvg, FedProx, FedBN, personalised", "FedAvg, FedProx, personalised, 8-bit; FedBN n/a (layer norm)"],
    ["Privacy", "“Various levels” (undefined)", "DP at ε = 8 and ε = 1; privacy matrix"],
  ];
  const cw = [2.0, 4.8, 5.33];
  const hx = [MX, MX + 2.0, MX + 6.8];
  [["", C.ink], ["Proposal", C.slate], ["Now", C.teal]].forEach((h, j) => { if (h[0]) chip(s, h[0], hx[j], 1.55, 1.6, h[1], C.white); });
  rows.forEach((r, i) => {
    const y = 2.05 + i * 0.85;
    s.addShape(pres.shapes.RECTANGLE, { x: MX, y, w: W - 2 * MX, h: 0.75, fill: { color: i % 2 ? C.white : C.panel }, line: { type: "none" } });
    txt(s, r[0], hx[0] + 0.15, y, cw[0] - 0.2, 0.75, { fontSize: 15, bold: true, color: C.ink, valign: "middle" });
    txt(s, r[1], hx[1], y, cw[1] - 0.2, 0.75, { fontSize: 14, color: C.muted, valign: "middle" });
    txt(s, r[2], hx[2], y, cw[2], 0.75, { fontSize: 14, color: C.ink, valign: "middle" });
  });
  txt(s, "Kept: cloud upper bound, federated and on-device variants, cold start, personas, communication cost, browser check.", MX, 6.4, W - 2 * MX, 0.4, { fontSize: 13, italic: true, color: C.muted });
  footer(s, "Research proposal §§3–4; study description; realism audit §F");
  s.addNotes(`الشريحة 13 — البروبوزال مقابل دلوقتي (اتفقنا عليه قبل كده).
الداتا REES46 بدل الداتاسيتس القديمة لأنها تسوق حقيقي بكل مستخدم وتوقيته. المهمة: المنتج الجاي على الكتالوج كله. المقياس MRR@20 لكل مستخدم. FedBN اتشال لأن الموديل بيستخدم LayerNorm. الخصوصية بقت قابلة للقياس بـ DP.
اللي فضل زي ما هو: السحابة كسقف، الفيدرالي، الجهاز، الـ cold start، الشخصيات، تكلفة الاتصال، واختبار المتصفح.`);
}

// 14. Timeline (simple)
{
  const s = light();
  header(s, "When the comparisons are read", "Four milestones (dates are estimates)");
  const ms = [
    ["≈ 30 Sep", "Tuning and pretraining finish", "Full-pool cloud, pretraining, federated tuning stages", "running"],
    ["≈ 30 Sep – 2 Oct", "Main federated and DP runs", "Each run ≈ 10.7 h (measured rate)", "next"],
    ["≈ 2–3 Oct", "One validation read", "All methods read together, then everything is locked", "sealed"],
    ["≈ 4–5 Oct", "Single test look → final tables ≈ 6–7 Oct", "The test week is opened once", "sealed"],
  ];
  const cw = (W - 2 * MX - 0.9) / 4;
  s.addShape(pres.shapes.LINE, { x: MX + 0.3, y: 2.25, w: W - 2 * MX - 0.6, h: 0, line: { color: C.line, width: 3 } });
  ms.forEach((m, i) => {
    const x = MX + i * (cw + 0.3);
    s.addShape(pres.shapes.OVAL, { x: x + 0.1, y: 2.05, w: 0.4, h: 0.4, fill: { color: i === 0 ? C.teal : i === 1 ? C.slate : C.amber }, line: { color: C.white, width: 2 } });
    txt(s, m[0], x, 1.5, cw, 0.4, { fontSize: 16, bold: true, color: C.ink });
    card(s, x, 2.75, cw, 3.2, C.panel);
    txt(s, m[1], x + 0.25, 2.95, cw - 0.5, 1.0, { fontSize: 17, bold: true, fontFace: HF, color: C.ink });
    txt(s, m[2], x + 0.25, 4.05, cw - 0.5, 1.2, { fontSize: 14 });
    statusChip(s, m[3], x + 0.25, 5.4, 1.3, m[3]);
  });
  txt(s, "The plan said 1–2 Oct for the first read; it slipped about a day because tuning screens run slower when sharing GPUs.", MX, 6.25, W - 2 * MX, 0.4, { fontSize: 12.5, italic: true, color: C.muted });
  footer(s, "Master plan v5 (cost model, test-look duration); ladder schedule; decision log 29 Sep");
  s.addNotes(`الشريحة 14 — إمتى المقارنات هتتقرا (كلها تقديرات).
حوالي 30 سبتمبر: الضبط والتدريب المسبق والسحابة الكاملة يخلصوا.
30 سبتمبر لـ 2 أكتوبر: التشغيلات الفيدرالية و DP الرئيسية (كل تشغيل حوالي 10.7 ساعة، معدل مقاس).
حوالي 2–3 أكتوبر: قراءة validation واحدة لكل الطرق مع بعض، وبعدها القفل.
حوالي 4–5 أكتوبر: نظرة الـ test الواحدة، والجداول النهائية حوالي 6–7 أكتوبر.
الخطة كانت 1–2 أكتوبر واتأخرت حوالي يوم بسبب إن تشغيلات الضبط أبطأ لما بتشارك الـ GPU.`);
}

// 15. Discussion 3 + summary
discussion(3, "Priorities and next steps", [
  ["If time is short, which comparison do we protect first?", "Never cut: federated ×2 seeds, local-only ×2, cloud fine-tune ×2, INT8, the lock, the test look. First to cut: extra panel seeds, the 8-bit second seed, ε = 1."],
  ["Is a server-CPU benchmark plus one browser run on a laptop enough device evidence?", "Or should we add a second machine?"],
  ["Can you pre-approve the Queen's AI-use disclosure?", "The independent review lists it as required front matter."],
], `ديسكشن 3 — الأولويات والخطوات الجاية.
1) لو الوقت ضيق، نحمي أنهي مقارنة الأول؟ حاجات مش هنقطعها: الفيدرالي ببذرتين، المحلي ببذرتين، fine-tune السحابة ببذرتين، INT8، القفل، ونظرة الـ test. أول حاجات تتقطع: بذور اللوحات الزيادة، البذرة التانية للـ 8 بت، ε=1.
2) CPU السيرفر + تشغيل في المتصفح على لابتوب كفاية كدليل للجهاز؟
3) يوافق مسبقاً على بيان استخدام الذكاء الاصطناعي المطلوب من Queen's؟
اختم: خلص المسابقة والـ baselines ومرحلتين من الضبط؛ شغال السحابة الكاملة والتدريب المسبق والضبط والجهاز؛ النتائج المقفولة كلها هتتقرا مع بعض حوالي 2–3 أكتوبر.
سجل التحديثات: v1-simple — 2026-09-29، نفس حالة النسخة الكاملة (DECISION_LOG لحد الصف 228، السلم لحد S1b).`, "cut order is pre-registered in the master plan");

pres.writeFile({ fileName: OUT }).then(f => console.log("wrote " + f + " slides=" + pageNo));
