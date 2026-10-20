// Supervisor deck generator — FINAL version (full deck trimmed; few numbers) (pptxgenjs). Every number here is listed with its source in SUPERVISOR_DECK_DATA.md.
const pptxgen = require("pptxgenjs");
const path = require("path");

const OUT = process.argv[2] || path.join(__dirname, "PPSI_Supervisor_Deck_Final.pptx");
const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE"; // 13.333 x 7.5 in
pres.title = "Privacy-Preserving Shopping Intelligence — supervisor update (final)";
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

// ===================== FINAL VERSION: the full deck, trimmed; few numbers, clear story =====================
function statusChip(s, t, x, y, w, kind) {
  const m = { done: [C.slate, C.white], running: [C.teal, C.white], sealed: [C.amberLight, C.amberDark], next: [C.slateLight, C.ink] }[kind];
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h: 0.34, rectRadius: 0.17, fill: { color: m[0] }, line: kind === "sealed" ? { color: C.amber, width: 1 } : { type: "none" } });
  s.addText(t, { x, y, w, h: 0.34, fontFace: BF, fontSize: 11, bold: true, color: m[1], align: "center", valign: "middle", margin: 0, isTextBox: true });
}
function stat(s, x, y, w, h, big, small, fill, color) {
  card(s, x, y, w, h, fill || C.panel);
  txt(s, big, x + 0.25, y + 0.15, w - 0.5, 0.75, { fontFace: HF, fontSize: 32, bold: true, color: color || C.teal });
  txt(s, small, x + 0.25, y + 0.9, w - 0.5, h - 1.0, { fontSize: 14, color: C.body });
}
function tag(s, t) { chip(s, t, W - MX - 4.2, 0.36, 4.2, C.slateLight, C.slate); }

// 1. Title
{
  const s = dark();
  s.addText("PRIVACY-PRESERVING SHOPPING INTELLIGENCE", { x: MX, y: 1.3, w: 11, h: 0.4, fontFace: BF, fontSize: 14, bold: true, charSpacing: 3, color: C.amber, margin: 0, isTextBox: true });
  s.addText("How much recommendation quality survives when shopping data stays on the device?", { x: MX, y: 1.8, w: 11.2, h: 1.9, fontFace: HF, fontSize: 40, bold: true, color: C.white, margin: 0, valign: "top", isTextBox: true });
  s.addText("Cloud vs federated vs on-device learning for Microsoft Edge — progress and decisions", { x: MX, y: 3.85, w: 11, h: 0.5, fontFace: BF, fontSize: 18, color: C.onDarkMuted, margin: 0, isTextBox: true });
  [["Cloud", C.slate], ["Federated", C.teal], ["On-device", C.amber]].forEach((l, i) => {
    s.addShape(pres.shapes.OVAL, { x: MX + i * 2.2, y: 4.85, w: 0.28, h: 0.28, fill: { color: l[1] }, line: { type: "none" } });
    s.addText(l[0], { x: MX + 0.4 + i * 2.2, y: 4.82, w: 1.7, h: 0.34, fontFace: BF, fontSize: 14, color: C.onDark, margin: 0, isTextBox: true });
  });
  s.addText("Ahmed Abdelhamed · MSc, Queen's University · Microsoft supervisor meeting · 30 September 2026", { x: MX, y: 6.4, w: 11, h: 0.35, fontFace: BF, fontSize: 13, color: C.onDarkMuted, margin: 0, isTextBox: true });
  pageNo += 1;
  s.addNotes(`الشريحة 1 — العنوان.
جملة البداية: المشروع بيجاوب على سؤال البروبوزال: لو بيانات التسوق فضلت على جهاز المستخدم، هنخسر قد إيه من جودة الترشيحات؟
العرض: اللي عملناه قبل كده وليه غيرنا، الإعداد الحالي، اللي اتقاس، اللي لسه شغال، والقرارات اللي محتاجين رأيه فيها.
أي تاريخ مستقبلي في العرض تقدير مش نتيجة. ونتائج الطرق الرئيسية مقفولة لحد القراءة المسجلة مسبقاً.`);
}

// 2. The question
{
  const s = light();
  header(s, "The question", "Three ways Edge could learn from shopping behaviour");
  txt(s, [{ text: "“How much shopping recommendation quality can be retained when moving from cloud-based to on-device or federated models?”", options: { italic: true } }], MX, 1.5, W - 2 * MX, 0.6, { fontSize: 17, color: C.ink });
  const cols = [
    ["CLOUD", C.slate, C.slateLight, "Events go to the server", "One large model, trained and served centrally.", "Best quality — no privacy."],
    ["FEDERATED", C.teal, C.tealLight, "Events stay on the device", "The device trains locally and sends only model updates.", "Optional differential privacy."],
    ["ON-DEVICE", C.amberDark, C.amberLight, "Nothing is uploaded", "A small model is downloaded once and adapted locally.", "Strongest privacy — least data."],
  ];
  const cw = (W - 2 * MX - 0.6) / 3;
  cols.forEach((c, i) => {
    const x = MX + i * (cw + 0.3), y = 2.45;
    card(s, x, y, cw, 3.4, c[2]);
    txt(s, c[0], x + 0.3, y + 0.3, cw - 0.6, 0.3, { fontSize: 12, bold: true, charSpacing: 2, color: c[1] });
    txt(s, c[3], x + 0.3, y + 0.7, cw - 0.6, 0.5, { fontSize: 20, bold: true, fontFace: HF, color: C.ink });
    txt(s, c[4], x + 0.3, y + 1.45, cw - 0.6, 1.0, { fontSize: 15 });
    txt(s, c[5], x + 0.3, y + 2.6, cw - 0.6, 0.5, { fontSize: 14, italic: true, color: C.muted });
  });
  txt(s, [{ text: "Headline measure: ", options: { bold: true, color: C.ink } }, { text: "retention = quality of a deployment ÷ quality of the cloud model." }], MX, 6.2, W - 2 * MX, 0.4, { fontSize: 15 });
  footer(s, "Research proposal §1; study description");
  s.addNotes(`الشريحة 2 — السؤال.
السؤال من البروبوزال حرفياً. Edge شغال على جهاز المستخدم، فعنده فرصة يعمل ترشيحات تسوق من غير ما يبعت التصفح للسيرفر.
الثلاث طرق: السحابة (أحسن جودة، مفيش خصوصية)، الفيدرالي (التدريب على الجهاز وبيطلع تحديثات بس، وممكن DP)، وعلى الجهاز فقط (أقوى خصوصية وأقل داتا).
الرقم الأساسي: retention = جودة الطريقة ÷ جودة السحابة.`);
}

// 3. Journey
{
  const s = light();
  header(s, "The journey", "Four phases, each answering one question");
  const ph = [
    ["1", "Next category", "Can we predict the next category?", "Worked — but nothing published to compare with", C.slate],
    ["2", "Next product, cloud", "How good can a cloud model get?", "Strong models; compared with a published benchmark", C.slate],
    ["3", "Features", "Which columns help?", "Kept a small set that a device can compute", C.slate],
    ["4", "Deployment study", "How much survives off the cloud?", "Running now; main results sealed", C.teal],
  ];
  const cw = (W - 2 * MX - 0.9) / 4;
  s.addShape(pres.shapes.LINE, { x: MX + 0.4, y: 1.95, w: W - 2 * MX - 0.8, h: 0, line: { color: C.line, width: 3 } });
  ph.forEach((p, i) => {
    const x = MX + i * (cw + 0.3);
    s.addShape(pres.shapes.OVAL, { x: x + 0.15, y: 1.7, w: 0.5, h: 0.5, fill: { color: p[4] }, line: { color: C.white, width: 2 } });
    s.addText(p[0], { x: x + 0.15, y: 1.7, w: 0.5, h: 0.5, fontFace: HF, fontSize: 16, bold: true, color: C.white, align: "center", valign: "middle", margin: 0, isTextBox: true });
    card(s, x, 2.45, cw, 3.3, i === 3 ? C.tealLight : C.panel);
    txt(s, p[1], x + 0.25, 2.65, cw - 0.5, 0.5, { fontSize: 19, bold: true, fontFace: HF, color: C.ink });
    txt(s, p[2], x + 0.25, 3.25, cw - 0.5, 0.8, { fontSize: 14, italic: true, color: C.muted });
    txt(s, p[3], x + 0.25, 4.25, cw - 0.5, 1.3, { fontSize: 15, color: C.ink });
  });
  footer(s, "MVP report; central-model campaign reports; feature study; study description");
  s.addNotes(`الشريحة 3 — الرحلة في 4 مراحل.
1) توقع الفئة الجاية: نجح، بس مفيش أبحاث منشورة نقارن بيها.
2) توقع المنتج الجاي بموديل سحابة: موديلات قوية، وقارناها بـ benchmark منشور.
3) الفيتشرز: فضلنا على مجموعة صغيرة الجهاز يقدر يحسبها.
4) دراسة النشر (سحابة/فيدرالي/جهاز): شغالة دلوقتي والنتائج الرئيسية مقفولة.`);
}

// 4. Next category -> why we moved
{
  const s = light();
  header(s, "Phase 1 — next category", "It worked, but we could not tell if it was good");
  tag(s, "EARLIER PHASE · October 2019 data");
  s.addChart(pres.charts.BAR, [{ name: "MRR@20", labels: ["Popularity", "Transition table", "Our GRU"], values: [0.1049, 0.3143, 0.3474] }], {
    x: MX, y: 1.5, w: 6.0, h: 4.9, barDir: "col", chartColors: [C.gray, C.gray, C.teal], varyColors: true, barGapWidthPct: 40,
    showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0.00", dataLabelFontSize: 16, dataLabelColor: C.ink,
    valAxisMinVal: 0, valAxisMaxVal: 0.4, valAxisHidden: true, valGridLine: { style: "none" }, catAxisLabelColor: C.body, catAxisLabelFontSize: 14,
    catGridLine: { style: "none" }, showLegend: false, showTitle: true, title: "Next category (when the category changes), validation", titleFontSize: 12, titleColor: C.ink,
  });
  const x2 = 7.0, w2 = W - MX - x2;
  txt(s, "Why we moved to next product", x2, 1.6, w2, 0.4, { fontSize: 18, bold: true, fontFace: HF, color: C.ink });
  const rs = [
    ["No published results", "for next category on REES46 — no way to say if we are stronger or weaker", C.amberLight],
    ["Too easy overall", "most of the metric (≈ 82%) is solved by “repeat the same category”", C.panel],
    ["A real benchmark exists", "for next product on REES46 (GRU4Rec, RecSys 2023)", C.tealLight],
  ];
  rs.forEach((r, i) => {
    const y = 2.15 + i * 1.3;
    card(s, x2, y, w2, 1.15, r[2]);
    txt(s, r[0], x2 + 0.25, y + 0.12, w2 - 0.5, 0.4, { fontSize: 16, bold: true, color: C.ink });
    txt(s, r[1], x2 + 0.25, y + 0.52, w2 - 0.5, 0.6, { fontSize: 13.5 });
  });
  txt(s, "Agreed with the supervisor.", x2, 6.1, w2, 0.35, { fontSize: 13, italic: true, color: C.muted });
  footer(s, "GitHub repo: MVP report §2; evaluation-protocol decision (ADR-001); literature register");
  s.addNotes(`الشريحة 4 — المرحلة 1: توقع الفئة الجاية (من الـ GitHub).
النتيجة: الـ GRU بتاعنا 0.35 مقابل جدول الانتقالات 0.31 والـ popularity 0.10 — يعني الموديل اتعلم فعلاً.
ليه انتقلنا لتوقع المنتج:
1) مالقيناش أبحاث منشورة على REES46 لمهمة الفئة الجاية، فمنقدرش نعرف إحنا أقوى ولا أضعف.
2) المقياس الكلي سهل: حوالي 82% منه بيتحل بقاعدة "كرر نفس الفئة".
3) مهمة المنتج الجاي ليها benchmark منشور على نفس الداتا (GRU4Rec، RecSys 2023).
واتفقنا على التغيير ده معاه.`);
}

// 5. Next product vs the published benchmark
{
  const s = light();
  header(s, "Phase 2 — next product vs the published benchmark", "Close to the paper, with a harder setup");
  tag(s, "EARLIER PHASE · full data");
  s.addChart(pres.charts.BAR, [
    { name: "Published GRU4Rec", labels: ["Recall@20", "MRR@20"], values: [0.5291, 0.2003] },
    { name: "Our SASRec", labels: ["Recall@20", "MRR@20"], values: [0.4315, 0.1542] },
  ], {
    x: MX, y: 1.5, w: 6.2, h: 4.9, barDir: "col", barGrouping: "clustered", chartColors: [C.amber, C.teal],
    showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0.00", dataLabelFontSize: 15, dataLabelColor: C.ink,
    valAxisMinVal: 0, valAxisMaxVal: 0.6, valAxisHidden: true, valGridLine: { style: "none" }, catAxisLabelColor: C.body, catAxisLabelFontSize: 15,
    catGridLine: { style: "none" }, showLegend: true, legendPos: "t", legendFontSize: 13, legendColor: C.body,
    showTitle: true, title: "REES46, next product viewed", titleFontSize: 12, titleColor: C.ink,
  });
  const x2 = 7.2, w2 = W - MX - x2;
  txt(s, "Why ours is lower", x2, 1.6, w2, 0.4, { fontSize: 18, bold: true, fontFace: HF, color: C.ink });
  const rs = [
    ["Harder ranking", "we rank the whole catalogue; new products always count as misses"],
    ["Harder split", "we predict the following week, not the last day"],
    ["Less compute", "a few epochs on two small GPUs — the paper is heavily tuned"],
  ];
  rs.forEach((r, i) => {
    const y = 2.15 + i * 1.1;
    card(s, x2, y, w2, 0.95, i % 2 ? C.white : C.panel, i % 2 ? { line: C.line } : {});
    txt(s, r[0], x2 + 0.25, y, 1.9, 0.95, { fontSize: 15, bold: true, color: C.ink, valign: "middle" });
    txt(s, r[1], x2 + 2.2, y, w2 - 2.4, 0.95, { fontSize: 13.5, valign: "middle" });
  });
  card(s, x2, 5.55, w2, 0.9, C.tealLight);
  txt(s, [{ text: "Not a same-protocol re-run — ", options: { bold: true, color: C.tealDark } }, { text: "now we have a 2× A40 server and have started experimenting." }], x2 + 0.25, 5.6, w2 - 0.5, 0.8, { fontSize: 14, valign: "middle" });
  footer(s, "Hidasi & Czapp, RecSys 2023 (official GRU4Rec REES46 results) as recorded in the literature register; central-model report (validation week, per-event)");
  s.addNotes(`الشريحة 5 — المرحلة 2: توقع المنتج مقارنة بالبحث المنشور (GRU4Rec الرسمي على REES46).
البحث: Recall@20 = 0.53 و MRR@20 = 0.20. إحنا (SASRec، أسبوع validation، داتا كاملة): 0.43 و 0.15.
ليه أقل:
- الترتيب أصعب: بنرتب الكتالوج كله، وأي منتج جديد بيتحسب غلط.
- التقسيم أصعب: بنتوقع الأسبوع اللي بعده مش آخر يوم.
- موارد أقل: كام epoch على GPUين صغيرين (A16)، والبحث متظبط جامد.
الخلاصة: مش نفس البروتوكول، فمنقدرش نقول أقوى ولا أضعف بشكل قاطع. دلوقتي معانا سيرفر 2× A40 وبدأنا نجرب.
لو اتسأل: رقمنا الأساسي متوسط لكل مستخدم (0.17)، والبحث لكل حدث، علشان كده قارنت بنفس طريقة البحث.`);
}

// 6. Features
{
  const s = light();
  header(s, "Phase 3 — feature engineering", "From many columns to a small set a device can compute");
  const groups = [
    ["Tokens", "item · category · brand"],
    ["Numbers", "price · time since last view · session counts"],
    ["User context", "brand / category match share · price vs usual"],
    ["Quality flags & masks", "price missing · history available"],
  ];
  groups.forEach((g, i) => {
    const y = 1.65 + i * 1.08;
    card(s, MX, y, 7.2, 0.92, i % 2 ? C.white : C.panel, i % 2 ? { line: C.line } : {});
    txt(s, g[0], MX + 0.3, y, 2.6, 0.92, { fontSize: 16, bold: true, color: C.ink, valign: "middle" });
    txt(s, g[1], MX + 2.9, y, 4.1, 0.92, { fontSize: 14, valign: "middle" });
  });
  txt(s, "Removing all side features clearly hurts; adding more did not help — so we kept the set.", MX, 6.1, 7.2, 0.5, { fontSize: 13, italic: true, color: C.muted });
  const x2 = 8.2, w2 = W - MX - x2;
  stat(s, x2, 1.65, w2, 1.5, "58 → 24", "input columns (earlier → now)", C.panel, C.slate);
  stat(s, x2, 3.35, w2, 1.5, "21 of 24", "computable on the device itself", C.tealLight, C.tealDark);
  txt(s, "No calendar columns survived.", x2, 5.05, w2, 0.4, { fontSize: 13.5, color: C.body });
  footer(s, "Feature dictionary; feature study; feature search; independent review (features)");
  s.addNotes(`الشريحة 6 — هندسة الفيتشرز (الأعمدة).
الأعمدة 4 أنواع: tokens (المنتج، الفئة، البراند)، أرقام (السعر، الوقت من آخر مشاهدة، عدادات الجلسة)، سياق المستخدم (نسبة تطابق البراند والفئة، السعر مقارنة بالعادي)، وعلامات جودة و masks (السعر ناقص، التاريخ متاح).
في المرحلة القديمة كانوا 58 عمود؛ دلوقتي 24 بعد فلترة إحصائية وبحث. 21 منهم الجهاز يقدر يحسبهم لوحده، والـ 3 الباقيين محتاجين قاموس مشترك (المنتج/الفئة/البراند).
شيل كل الفيتشرز الجانبية بيوقع الجودة بوضوح، وإضافة فيتشرز جديدة ما فادتش — فثبتنا المجموعة. أعمدة التقويم اتشالت.`);
}

// 7. What changed from the proposal
{
  const s = light();
  header(s, "Scope", "What changed from the proposal, and why");
  const rows = [
    ["Task", "Next category, purchase, offers", "Next product viewed", "Harder, standard, has a benchmark"],
    ["Data", "Five public datasets", "REES46 (per-user, timestamped)", "Each user can be simulated as one device"],
    ["Metric", "NDCG@10, HR@10, AUC", "MRR@20 per user", "Matches a browser serving one user"],
    ["Federated", "FedAvg, FedProx, FedBN, personalised", "Same minus FedBN; + tuned optimizers", "Our model uses layer norm, not batch norm"],
    ["Privacy", "“Various levels”", "DP at ε = 8 and ε = 1", "Makes privacy measurable"],
  ];
  const hdr = ["", "Proposal", "Now", "Why"].map(t => ({ text: t, options: { bold: true, color: C.white, fill: { color: C.ink }, fontSize: 14 } }));
  const body = rows.map((r, i) => r.map((t, j) => ({ text: t, options: { bold: j === 0, color: j === 0 ? C.ink : C.body, fill: { color: i % 2 ? C.white : C.panel }, fontSize: 14 } })));
  s.addTable([hdr, ...body], { x: MX, y: 1.6, w: W - 2 * MX, colW: [1.5, 3.3, 3.4, 3.93], fontFace: BF, border: { type: "solid", pt: 0.5, color: C.line }, margin: [6, 8, 6, 8], rowH: 0.68 });
  txt(s, "Kept: cloud upper bound · federated and on-device variants · cold start · communication cost · browser check.", MX, 6.2, W - 2 * MX, 0.4, { fontSize: 13.5, italic: true, color: C.muted });
  footer(s, "Research proposal §§3–4; study description; realism audit");
  s.addNotes(`الشريحة 7 — إيه اللي اتغير عن البروبوزال (اتفقنا عليه قبل كده).
المهمة: المنتج الجاي بدل الفئة والشراء — أصعب ومعروفة وليها benchmark. الداتا: REES46 لأن كل مستخدم بأحداثه وتوقيتها فنقدر نعامله كجهاز. المقياس: MRR@20 لكل مستخدم. FedBN اتشال لأن الموديل بيستخدم LayerNorm. الخصوصية بقت DP عند ε=8 و ε=1.
واللي فضل: السحابة كسقف، الفيدرالي، الجهاز، الـ cold start، تكلفة الاتصال، واختبار المتصفح.`);
}

// 8. Data and task (current study)
{
  const s = light();
  header(s, "Phase 4 — data and task", "Next product, ranked over the whole catalogue");
  const st = [["158,486", "products ranked for every prediction"], ["131,072", "users in the main cohort — each one a simulated device"], ["1.35 M", "users behind the realistic full-pool cloud"]];
  const sw = (W - 2 * MX - 0.6) / 3;
  st.forEach((t, i) => stat(s, MX + i * (sw + 0.3), 1.6, sw, 1.6, t[0], t[1]));
  txt(s, "Chronological split — the model never sees the future", MX, 3.55, 8, 0.35, { fontSize: 16, bold: true, color: C.ink });
  const tw = W - 2 * MX, ty = 4.05;
  const segs = [["Training — Oct to 15 Nov 2019", 0.70, C.slate, C.white], ["Validation — 16–22 Nov", 0.15, C.teal, C.white], ["Test — 23–29 Nov (sealed)", 0.15, C.amber, C.ink]];
  let cx = MX;
  segs.forEach(sg => {
    const w = tw * sg[1];
    s.addShape(pres.shapes.RECTANGLE, { x: cx, y: ty, w: w - 0.04, h: 0.6, fill: { color: sg[2] }, line: { type: "none" } });
    s.addText(sg[0], { x: cx + 0.1, y: ty, w: w - 0.2, h: 0.6, fontFace: BF, fontSize: 13, bold: true, color: sg[3], valign: "middle", margin: 0, isTextBox: true });
    cx += w;
  });
  txt(s, bullets(["Users with 5–100 events, so every simulated device has a realistic history.", "Tuning only on a separate calibration subset; the test week is opened once, after everything is locked."]), MX, 5.0, W - 2 * MX, 1.3, { fontSize: 15 });
  footer(s, "Data protocol and population lock; study description");
  s.addNotes(`الشريحة 8 — الداتا والمهمة في الدراسة الحالية.
158,486 منتج بيترتبوا في كل توقع. 131,072 مستخدم في الـ main cohort، كل واحد جهاز. و1.35 مليون مستخدم للسحابة الكاملة الواقعية.
التقسيم زمني: تدريب لحد 15 نوفمبر، validation أسبوع، test أسبوع مقفول.
المستخدمين عندهم 5–100 حدث علشان كل جهاز يبقى واقعي. الضبط على مجموعة calibration منفصلة، والـ test بيتفتح مرة واحدة بعد القفل.
لو اتسأل "ليه داتا أصغر من المرحلة القديمة؟" — علشان محاكاة الفيدرالي مرات كتير تبقى ممكنة.`);
}

// 9. Fairness
{
  const s = light();
  header(s, "Method", "How we keep the comparison fair");
  const items = [
    ["Pre-registered", "Rules and comparisons are written and sealed before the result is seen."],
    ["Same budget", "Same data, features, evaluator and training exposure for every method."],
    ["Full catalogue", "Exact ranking over every product; no sampled negatives."],
    ["Paired statistics", "Same users for every comparison; two seeds for the main methods."],
    ["One test look", "Everything is locked, then the test week is opened once."],
    ["Independent review", "Every stage is checked by an independent reviewer before use."],
  ];
  const cw = (W - 2 * MX - 0.6) / 3, chh = 2.2;
  items.forEach((it, i) => {
    const x = MX + (i % 3) * (cw + 0.3), y = 1.6 + Math.floor(i / 3) * (chh + 0.3);
    card(s, x, y, cw, chh, i % 3 === 1 ? C.tealLight : C.panel);
    s.addShape(pres.shapes.OVAL, { x: x + 0.3, y: y + 0.3, w: 0.42, h: 0.42, fill: { color: C.teal }, line: { type: "none" } });
    s.addText(String(i + 1), { x: x + 0.3, y: y + 0.3, w: 0.42, h: 0.42, fontFace: HF, fontSize: 15, bold: true, color: C.white, align: "center", valign: "middle", margin: 0, isTextBox: true });
    txt(s, it[0], x + 0.9, y + 0.33, cw - 1.1, 0.4, { fontSize: 18, bold: true, fontFace: HF, color: C.ink });
    txt(s, it[1], x + 0.3, y + 0.95, cw - 0.6, 1.1, { fontSize: 15 });
  });
  footer(s, "Evaluation contract; study description");
  s.addNotes(`الشريحة 9 — إزاي المقارنة عادلة.
1) مسجل مسبقاً: القواعد والمقارنات بتتكتب وتتختم قبل ما نشوف النتيجة.
2) نفس الميزانية ونفس الداتا والفيتشرز والـ evaluator لكل الطرق.
3) ترتيب على الكتالوج كله من غير عينات.
4) نفس المستخدمين في كل مقارنة، وبذرتين للطرق الرئيسية.
5) نظرة واحدة على الـ test بعد القفل.
6) كل مرحلة بتتراجع من مراجع مستقل قبل ما نستخدمها.`);
}

// 10. Discussion 1
discussion(1, "Task, benchmark and yardstick", [
  ["Do you agree next product is the right task?", "Next category had nothing published to compare with."],
  ["Should we re-run the published benchmark protocol on our new server?", "It would tell us directly whether we are stronger or weaker."],
  ["Retention against which cloud: the matched 131k users or the 1.35M full pool?", "We plan to report both — which should lead?"],
], `ديسكشن 1 — المهمة والـ benchmark والمسطرة.
1) موافق إن توقع المنتج الجاي هو المهمة الصح؟ الفئة الجاية ماكانش ليها حاجة منشورة نقارن بيها.
2) نعيد تشغيل بروتوكول البحث المنشور على السيرفر الجديد؟ ده هيقول بشكل مباشر إحنا أقوى ولا أضعف.
3) الـ retention نقيسه قدام السحابة المطابقة (131 ألف) ولا الكاملة (1.35 مليون)؟ هنعرض الاتنين، بس مين الأول؟`, "phases 1–4");

// 11. Contest
{
  const s = light();
  header(s, "Backbone contest", "SASRec won the pre-registered contest against GRU");
  s.addChart(pres.charts.BAR, [
    { name: "GRU", labels: ["Registered rule: best checkpoint", "Fixed end of training"], values: [0.1569, 0.1569] },
    { name: "SASRec", labels: ["Registered rule: best checkpoint", "Fixed end of training"], values: [0.1612, 0.1527] },
  ], {
    x: MX, y: 1.5, w: 7.4, h: 4.9, barDir: "col", barGrouping: "clustered", chartColors: [C.slate, C.teal],
    showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0.000", dataLabelFontSize: 15, dataLabelColor: C.ink,
    valAxisMinVal: 0.14, valAxisMaxVal: 0.165, valAxisHidden: true, valGridLine: { style: "none" }, catAxisLabelColor: C.body, catAxisLabelFontSize: 14,
    catGridLine: { style: "none" }, showLegend: true, legendPos: "t", legendFontSize: 14, legendColor: C.body,
    showTitle: true, title: "Validation MRR@20, main cohort, 2 seeds", titleFontSize: 12, titleColor: C.ink,
  });
  const x2 = 8.4, w2 = W - MX - x2;
  card(s, x2, 1.6, w2, 1.9, C.tealLight);
  txt(s, "SASRec selected", x2 + 0.3, 1.75, w2 - 0.6, 0.5, { fontSize: 22, bold: true, fontFace: HF, color: C.tealDark });
  txt(s, "by the rule fixed before any result; stable across both seeds.", x2 + 0.3, 2.3, w2 - 0.6, 1.0, { fontSize: 14 });
  card(s, x2, 3.7, w2, 1.9, C.amberLight);
  txt(s, "Honest caveat", x2 + 0.3, 3.85, w2 - 0.6, 0.5, { fontSize: 22, bold: true, fontFace: HF, color: C.amberDark });
  txt(s, "At the fixed end of training GRU is ahead — disclosed in the thesis.", x2 + 0.3, 4.4, w2 - 0.6, 1.0, { fontSize: 14 });
  footer(s, "Contest result; contest diagnostics (decision log, 28 Sep)");
  s.addNotes(`الشريحة 11 — مسابقة الموديل الأساسي.
8 تشغيلات (عائلتين × حجمين × بذرتين) بنفس ميزانية الضبط. القاعدة المسجلة مسبقاً "أحسن checkpoint داخل الميزانية": SASRec 0.161 مقابل GRU 0.157، والفرق ثابت في البذرتين وفترة الثقة بعيدة عن الصفر → SASRec اتختار.
التحفظ: عند نهاية التدريب الثابتة GRU أعلى (0.157 مقابل 0.153)، لأن SASRec بيوصل لأحسن نقطة في النص. مكتوب في الرسالة إنه "اتختار بالقاعدة المسجلة".`);
}

// 12. Model sizes and device budget
{
  const s = light();
  header(s, "Model sizes", "A large cloud model and a device model sized for a browser");
  stat(s, MX, 1.6, 4.4, 1.7, "84 M", "parameters — cloud model", C.slateLight, C.slate);
  stat(s, MX, 3.55, 4.4, 1.7, "21 M", "parameters — device model", C.tealLight, C.tealDark);
  s.addChart(pres.charts.BAR, [{ name: "MB per visit", labels: ["Cloud-size model", "Device model", "8-bit uploads", "Frozen item tables"], values: [671, 166, 104, 2] }], {
    x: 5.5, y: 1.5, w: W - MX - 5.5, h: 4.9, barDir: "bar", chartColors: [C.slate, C.teal, C.teal, C.amber], varyColors: true, barGapWidthPct: 40,
    showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0", dataLabelFontSize: 15, dataLabelColor: C.ink,
    valAxisHidden: true, valAxisMaxVal: 780, valGridLine: { style: "none" }, catGridLine: { style: "none" }, catAxisLabelColor: C.body, catAxisLabelFontSize: 14, catAxisOrientation: "maxMin",
    showLegend: false, showTitle: true, title: "Megabytes per federated visit — device cap 170 MB", titleFontSize: 13, titleColor: C.ink,
  });
  txt(s, "Only the device-size model fits the cap.", MX, 5.5, 4.4, 0.6, { fontSize: 14, italic: true, color: C.muted });
  footer(s, "Architecture lock (parameters, bytes per visit); review synthesis (frozen tables)");
  s.addNotes(`الشريحة 12 — أحجام الموديلات وميزانية الجهاز.
موديل السحابة 84 مليون باراميتر، وموديل الجهاز 21 مليون.
الزيارة الفيدرالية الواحدة: موديل بحجم السحابة 671 ميجا (مستحيل)، موديل الجهاز 166 (تحت سقف 170)، رفع 8 بت 104، والجداول المجمدة حوالي 2 ميجا بس.
لو اتسأل "166 كتير؟" — أيوه للموبايل، مقبول للابتوب؛ علشان كده عندنا 8 بت والجداول المجمدة.`);
}

// 13. Comparison matrix
{
  const s = light();
  header(s, "What is compared", "Every method, placed by what leaves the device");
  const lv = [
    ["Cloud", C.slate, C.white, "Large cloud model · small cloud model · full-pool cloud", "Raw events"],
    ["Cloud-trained, runs on device", "4F6A73", C.white, "Small model shipped to the device (ONNX)", "Raw events, once"],
    ["Pretrained on opted-in users", C.amberDark, C.white, "Pretrained → + federated → + frozen item tables", "Opted-in: events · others: updates"],
    ["Federated", C.teal, C.white, "FedAvg · FedProx · personal vector · 8-bit uploads", "Model updates"],
    ["Federated + DP", C.tealDark, C.white, "DP at ε = 8 and ε = 1", "Noised updates"],
    ["On-device", C.amber, C.ink, "Local-only model · models fine-tuned on the device", "Nothing"],
    ["Baselines", C.gray, C.ink, "Last viewed item · popularity · session kNN", "—"],
  ];
  const hdr = ["Privacy level", "Methods", "What leaves the device"].map(t => ({ text: t, options: { bold: true, color: C.white, fill: { color: C.ink }, fontSize: 14 } }));
  const body = lv.map(r => [
    { text: r[0], options: { bold: true, color: r[2], fill: { color: r[1] }, fontSize: 13.5 } },
    { text: r[3], options: { color: C.body, fill: { color: C.white }, fontSize: 13.5 } },
    { text: r[4], options: { color: C.body, fill: { color: C.panel }, fontSize: 13.5 } },
  ]);
  s.addTable([hdr, ...body], { x: MX, y: 1.55, w: W - 2 * MX, colW: [3.0, 5.9, 3.23], fontFace: BF, border: { type: "solid", pt: 0.5, color: C.line }, margin: [5, 8, 5, 8], rowH: 0.6 });
  txt(s, "Every method is also compared with “last viewed item” — a free device rule with no training.", MX, 6.45, W - 2 * MX, 0.35, { fontSize: 13, italic: true, color: C.muted });
  footer(s, "Privacy matrix (pre-registered, revised after the independent review)");
  s.addNotes(`الشريحة 13 — مصفوفة المقارنة حسب "إيه اللي بيطلع من الجهاز".
السحابة: الأحداث الخام. متدرب في السحابة وشغال على الجهاز: الأحداث مرة واحدة. الفرع الواقعي: المستخدمين الموافقين بيشاركوا الداتا والباقي تحديثات بس. الفيدرالي: تحديثات الموديل. الفيدرالي + DP: تحديثات عليها ضوضاء. على الجهاز: ولا حاجة.
وكل طريقة بتتقارن كمان بقاعدة "آخر منتج اتشاف" — ببلاش ومن غير تدريب.`);
}

// 14. Discussion 2
discussion(2, "Which deployment should be the headline answer for Edge?", [
  ["Plain federated, or pretrained-then-federated, or cloud model fine-tuned on the device?", "The independent review recommends the pretrained branch — it mirrors how browsers ship features."],
  ["Should “last viewed item” be the floor every method must beat?", "It needs no training and sends nothing."],
], `ديسكشن 2 — مين الإجابة الرئيسية لـ Edge؟
1) الفيدرالي العادي، ولا تدريب مسبق ثم فيدرالي، ولا موديل سحابة يتظبط على الجهاز؟ المراجعة المستقلة بتقترح الفرع المُدرب مسبقاً لأنه شبه اللي المتصفحات بتعمله.
2) هل "آخر منتج اتشاف" يبقى الأرضية اللي كل طريقة لازم تغلبها؟`, "comparison matrix");

// 15. Federated: our mistake
{
  const s = light();
  header(s, "Federated learning", "Our first federated setup was wrong — and we fixed it");
  const cols = [
    ["WHAT WE DID", C.amberDark, C.amberLight, ["Started from an untrained model (random item tables)", "Reused the cloud's training settings", "No server optimizer; client optimizer reset every visit"]],
    ["WHAT IT CAUSED", C.slate, C.panel, ["Federated far below the cloud on calibration users (≈ 20%)", "Even below the free “last viewed item” rule", "A warning from tuning users — not a final result"]],
    ["WHAT WE CHANGED", C.tealDark, C.tealLight, ["Tuning ladder: +46% so far (learning rate, then server optimizer)", "Pretrained model ready — federated starts from it next", "Fixed before any final result is read"]],
  ];
  const cw = (W - 2 * MX - 0.6) / 3;
  cols.forEach((c, i) => {
    const x = MX + i * (cw + 0.3), y = 1.6;
    card(s, x, y, cw, 3.9, c[2]);
    txt(s, c[0], x + 0.3, y + 0.25, cw - 0.6, 0.3, { fontSize: 13, bold: true, charSpacing: 2, color: c[1] });
    s.addText(bullets(c[3], { paraSpaceAfter: 12 }), { x: x + 0.3, y: y + 0.75, w: cw - 0.6, h: 3.0, fontFace: BF, fontSize: 15.5, color: C.ink, valign: "top", margin: 0, isTextBox: true });
    if (i < 2) s.addShape(pres.shapes.RIGHT_ARROW, { x: x + cw + 0.02, y: y + 1.75, w: 0.26, h: 0.4, fill: { color: C.gray }, line: { type: "none" } });
  });
  card(s, MX, 5.75, W - 2 * MX, 0.8, C.ink);
  txt(s, "The gap mainly reflected our setup, not federated learning itself.", MX + 0.3, 5.75, W - 2 * MX - 0.6, 0.8, { fontSize: 18, bold: true, color: C.white, valign: "middle" });
  footer(s, "Realism audit; federated ladder research; independent review; ladder register");
  s.addNotes(`الشريحة 15 — الفيدرالي: الغلطة بتاعتنا وإزاي صلحناها (قولها بصراحة).
إيه اللي عملناه: بدأنا الفيدرالي من موديل مش متدرب (جداول المنتجات عشوائية)، واستخدمنا إعدادات تدريب السحابة زي ما هي، ومفيش server optimizer، والـ optimizer عند العميل بيبدأ من الصفر كل زيارة. يعني ما دربناش الموديل قبل الخطوة دي.
النتيجة: الإعداد الأول طلع حوالي 20% من السحابة على مستخدمين الضبط، وأقل من قاعدة "آخر منتج اتشاف" اللي ببلاش. ده إنذار من مستخدمين الضبط، مش نتيجة نهائية.
اللي غيرناه: سلم ضبط مسجل — +46% لحد دلوقتي (رفع الـ learning rate، وبعدين server optimizer اسمه FedAdam). والأهم: الموديل المُدرب مسبقاً خلص تدريبه، والفيدرالي هيبدأ منه — زي ما المتصفحات بتعمل فعلاً. وكل ده قبل ما نقرا أي نتيجة نهائية.
الرسالة: الفجوة كانت أساساً من الإعداد بتاعنا، مش من الفيدرالي نفسه (ده رأي المراجعة المستقلة كمان).
سؤال محتمل: "مش ده تحيز إنكم بتحسنوا الفيدرالي؟" — الضبط على مستخدمين الضبط بس ومسجل قبل القراءة، والنقطة الأولى فاضلة في الرسالة كمرجع.`);
}

// 16. Ladder
{
  const s = light();
  header(s, "Federated improvement ladder", "One change at a time, each kept only if it helps");
  s.addChart(pres.charts.BAR, [{ name: "Calibration MRR@20", labels: ["Start", "Client learning rate", "+ Server optimizer"], values: [0.023473, 0.028796, 0.034247] }], {
    x: MX, y: 1.5, w: 6.4, h: 4.9, barDir: "col", chartColors: ["9FB7BA", "5E9DA1", C.teal], varyColors: true, barGapWidthPct: 40,
    showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0.000", dataLabelFontSize: 16, dataLabelColor: C.ink,
    valAxisMinVal: 0, valAxisMaxVal: 0.038, valAxisHidden: true, valGridLine: { style: "none" }, catAxisLabelColor: C.body, catAxisLabelFontSize: 14,
    catGridLine: { style: "none" }, showLegend: false, showTitle: true, title: "Calibration users — +46% so far", titleFontSize: 13, titleColor: C.ink,
  });
  const x2 = 7.4, w2 = W - MX - x2;
  txt(s, "Stages", x2, 1.55, w2, 0.4, { fontSize: 17, bold: true, fontFace: HF, color: C.ink });
  const st = [["Client learning rate", "done"], ["Server optimizer", "done"], ["Learning-rate shape × client optimizer", "running"], ["Local passes and batch size", "next"], ["Start from the pretrained model", "next"], ["Freeze the item tables", "next"]];
  st.forEach((r, i) => {
    const y = 2.1 + i * 0.62;
    txt(s, r[0], x2, y, w2 - 1.2, 0.5, { fontSize: 15, color: C.ink, valign: "middle" });
    statusChip(s, r[1], x2 + w2 - 1.1, y + 0.08, 1.1, r[1]);
  });
  txt(s, "A stage stays only if it gains ≥ 0.002.", x2, 5.95, w2, 0.4, { fontSize: 13.5, italic: true, color: C.muted });
  footer(s, "Improvement-ladder register; ladder stage registrations");
  s.addNotes(`الشريحة 16 — سلم تحسين الفيدرالي.
كل مرحلة بتغير حاجة واحدة، وبتفضل بس لو كسبت 0.002 على الأقل.
لحد دلوقتي: البداية 0.023 ← رفع الـ learning rate عند العميل 0.029 (+22.7%) ← server optimizer (FedAdam) 0.034 (+18.9% كمان). الإجمالي +46% عن البداية.
مرحلة الـ server optimizer خلصت (فاضل تشغيل تأكيد أخير واحد قبل ما الاختيار يتكتب رسمياً).
الجاي: شكل الـ learning rate مع optimizer العميل (شغال دلوقتي)، الـ passes والـ batch، البدء من الموديل المُدرب مسبقاً (خلص تدريبه)، وتجميد جداول المنتجات.
سؤال محتمل: "لو دربنا أطول هيتحسن؟" — لأ غالباً: منحنى FedAdam بيوصل لأعلى نقطة في نص الميزانية وبعدين بينزل، والنزول بيبدأ بالظبط لما الـ learning rate بيتخفض (نفس جدول السحابة). علشان كده المرحلة الجاية بتجرب learning rate ثابت. الرافعة الأكبر هي داتا أكتر والبدء من الموديل المُدرب مسبقاً.`);
}

// 17. Privacy
{
  const s = light();
  header(s, "Privacy", "What the guarantee covers — and what it does not");
  [["ε = 8", "moderate"], ["ε = 1", "strong"]].forEach((d, i) => {
    const x = MX + i * 2.9;
    card(s, x, 1.6, 2.65, 1.6, C.tealLight);
    txt(s, d[0], x + 0.3, 1.72, 2.1, 0.7, { fontFace: HF, fontSize: 34, bold: true, color: C.tealDark });
    txt(s, d[1], x + 0.3, 2.5, 2.1, 0.5, { fontSize: 14 });
  });
  txt(s, bullets(["Per-user guarantee: one user's whole history barely changes the model.", "Adversary: an honest-but-curious browser vendor."]), MX, 3.5, 5.6, 2.2, { fontSize: 15 });
  const x2 = 6.9, w2 = W - MX - x2;
  card(s, x2, 1.6, w2, 2.9, C.amberLight);
  txt(s, "Stated limits", x2 + 0.3, 1.75, w2 - 0.6, 0.4, { fontSize: 17, bold: true, fontFace: HF, color: C.amberDark });
  txt(s, bullets(["Without DP, an update can reveal which products a user viewed.", "Secure aggregation is not simulated.", "Frozen item tables reduce this leak, not remove it."], { paraSpaceAfter: 12 }), x2 + 0.3, 2.3, w2 - 0.6, 3.4, { fontSize: 15 });
  footer(s, "Privacy statement and threat model (registered after the review)");
  s.addNotes(`الشريحة 17 — الخصوصية.
مستويين DP: ε=8 (متوسط) و ε=1 (قوي). الضمان على مستوى المستخدم: تاريخ مستخدم كامل مش هيغير الموديل تقريباً.
الخصم: شركة المتصفح نفسها "أمينة وفضولية".
الحدود بصراحة: من غير DP التحديث ممكن يكشف المنتجات اللي المستخدم شافها؛ secure aggregation مش متحاكي؛ والجداول المجمدة بتقلل التسريب مش بتلغيه.`);
}

// 18. Realistic deployment branch
{
  const s = light();
  header(s, "Realistic deployment branch", "How a browser would actually ship it");
  const steps = [
    ["1", "Pretrain on opted-in users", "≈ 1.2M users who share data; none of them in the main cohort", C.amberLight, C.amberDark],
    ["2", "Federated on private users", "Start from the pretrained model; private users send only updates", C.tealLight, C.tealDark],
    ["3", "Freeze the item tables", "Only a small encoder moves: about 1 MB each way", C.panel, C.ink],
  ];
  const bw = 3.55, gap = 0.62;
  steps.forEach((t, i) => {
    const x = MX + i * (bw + gap), y = 1.8;
    card(s, x, y, bw, 2.4, t[3]);
    s.addShape(pres.shapes.OVAL, { x: x + 0.25, y: y + 0.25, w: 0.5, h: 0.5, fill: { color: t[4] }, line: { type: "none" } });
    s.addText(t[0], { x: x + 0.25, y: y + 0.25, w: 0.5, h: 0.5, fontFace: HF, fontSize: 18, bold: true, color: C.white, align: "center", valign: "middle", margin: 0, isTextBox: true });
    txt(s, t[1], x + 0.95, y + 0.25, bw - 1.1, 0.6, { fontSize: 17, bold: true, fontFace: HF, color: C.ink, valign: "middle" });
    txt(s, t[2], x + 0.25, y + 1.1, bw - 0.5, 1.2, { fontSize: 15 });
    if (i < 2) s.addShape(pres.shapes.RIGHT_ARROW, { x: x + bw + 0.1, y: y + 1.0, w: gap - 0.2, h: 0.45, fill: { color: C.gray }, line: { type: "none" } });
  });
  card(s, MX, 4.6, W - 2 * MX, 1.5, C.white, { line: C.line });
  txt(s, "This directly fixes our first mistake: federated training no longer starts from an untrained model.", MX + 0.3, 4.72, W - 2 * MX - 0.6, 0.5, { fontSize: 16, bold: true, color: C.tealDark });
  txt(s, "Judged against the full-pool cloud. Pretraining has finished; federated training from it is next. Results sealed until the first read.", MX + 0.3, 5.3, W - 2 * MX - 0.6, 0.6, { fontSize: 14.5 });
  footer(s, "Warm-start registration and its review; pretraining cohort build");
  s.addNotes(`الشريحة 18 — الفرع الواقعي للنشر (ده الحل للغلطة).
1) ندرب موديل مسبقاً على حوالي 1.2 مليون مستخدم موافقين يشاركوا الداتا، ومفيش حد منهم في الـ main cohort.
2) الفيدرالي يبدأ من الموديل ده على المستخدمين الخاصين، وهما بيبعتوا تحديثات بس.
3) نجمد جداول المنتجات فيتحرك encoder صغير بس، حوالي 1 ميجا في كل اتجاه.
ده بيصلح الغلطة الأولى مباشرة: الفيدرالي مبقاش بيبدأ من موديل مش متدرب.
بنقارنه بالسحابة الكاملة. التدريب المسبق خلص ميزانيته، والفيدرالي هيبدأ منه بعد كده. النتائج مقفولة.`);
}

// 19. Discussion 3
discussion(3, "Federated effort, privacy and the deployment story", [
  ["How much tuning effort is fair for federated learning?", "Our ladder stops when a stage gains less than 0.002."],
  ["Is DP at ε = 8 and ε = 1 enough, or do you want a curve?", "One more level costs about half a day (estimate)."],
  ["Is pretraining on opted-in users a realistic assumption for Edge?", "If yes, it becomes the headline Edge answer."],
], `ديسكشن 3 — مجهود الفيدرالي والخصوصية وقصة النشر.
1) قد إيه ضبط للفيدرالي عادل؟ السلم بيقف لو المرحلة كسبت أقل من 0.002.
2) ε=8 و ε=1 كفاية ولا محتاج منحنى؟ مستوى زيادة حوالي نص يوم (تقدير).
3) هل التدريب المسبق على مستخدمين موافقين واقعي لـ Edge؟ لو أيوه، يبقى هو الإجابة الرئيسية.`, "federated slides and privacy");

// 20. On-device and cold start
{
  const s = light();
  header(s, "On-device and cold start", "What runs on the device, and who it serves");
  txt(s, "On the device", MX, 1.55, 6, 0.4, { fontSize: 17, bold: true, fontFace: HF, color: C.ink });
  const arms = [["Local-only model", "a control, not a product"], ["Cloud model, fine-tuned locally", "download once, adapt"], ["Pretrained model, fine-tuned locally", "cleanest download-only option"]];
  arms.forEach((a, i) => {
    const y = 2.05 + i * 1.0;
    card(s, MX, y, 5.9, 0.86, i === 2 ? C.amberLight : C.panel);
    txt(s, a[0], MX + 0.3, y + 0.08, 5.3, 0.4, { fontSize: 15, bold: true, color: C.ink });
    txt(s, a[1], MX + 0.3, y + 0.46, 5.3, 0.35, { fontSize: 13.5 });
  });
  const cks = [["Export", "ONNX, full and 8-bit", "done"], ["Latency", "single-thread CPU", "running"], ["Browser", "Edge on a laptop", "next"]];
  cks.forEach((c, i) => {
    const y = 5.2 + i * 0.45;
    txt(s, c[0] + " — " + c[1], MX, y, 4.6, 0.38, { fontSize: 13.5, valign: "middle" });
    statusChip(s, c[2], MX + 4.7, y + 0.02, 1.2, c[2]);
  });
  const x2 = 7.0, w2 = W - MX - x2;
  txt(s, "Cold start", x2, 1.55, w2, 0.4, { fontSize: 17, bold: true, fontFace: HF, color: C.ink });
  const cs = [["Users who never took part", "does the model still serve them?"], ["Users with no history at all", "what can any method do?"], ["Short vs long histories", "how many events until quality catches up?"]];
  cs.forEach((a, i) => {
    const y = 2.05 + i * 1.0;
    card(s, x2, y, w2, 0.86, C.panel);
    txt(s, a[0], x2 + 0.3, y + 0.08, w2 - 0.6, 0.4, { fontSize: 15, bold: true, color: C.ink });
    txt(s, a[1], x2 + 0.3, y + 0.46, w2 - 0.6, 0.35, { fontSize: 13.5 });
  });
  txt(s, "A controlled test cuts histories to fixed lengths, so the answer is not just “heavy users differ”.", x2, 5.2, w2, 0.9, { fontSize: 13.5, italic: true, color: C.muted });
  footer(s, "Study description; privacy matrix; cold-start registration");
  s.addNotes(`الشريحة 20 — على الجهاز والـ cold start.
على الجهاز: موديل محلي بس (control مش منتج)، موديل السحابة يتظبط محلياً، والموديل المُدرب مسبقاً يتظبط محلياً (أنضف خيار "تحميل بس").
الفحوصات: التصدير لـ ONNX خلص (كامل و 8 بت)، السرعة على CPU شغالة، واختبار المتصفح في Edge على لابتوبي لسه.
الـ cold start: مستخدمين ما شاركوش في التدريب، مستخدمين من غير أي تاريخ، وتاريخ قصير مقابل طويل. في تجربة متحكم فيها بتقطع التاريخ لأطوال ثابتة.`);
}

// 21. Reviews
{
  const s = light();
  header(s, "Independent reviews", "7/10 today, 8–8.5 after the registered fixes");
  const cols = [
    ["Strengths", C.tealLight, C.tealDark, ["Pre-registration and one sealed test look", "Exact evaluation — no bug found", "No data leakage"]],
    ["Weaknesses found", C.amberLight, C.amberDark, ["Federated setup too weak", "Privacy claims over-stated", "Device evidence from a server CPU"]],
    ["Already fixed or registered", C.panel, C.slate, ["Tuning ladder and pretrained start", "Threat model and privacy matrix", "Realistic deployment branch promoted"]],
  ];
  const cw = (W - 2 * MX - 0.6) / 3;
  cols.forEach((c, i) => {
    const x = MX + i * (cw + 0.3);
    card(s, x, 1.65, cw, 3.6, c[1]);
    txt(s, c[0], x + 0.3, 1.85, cw - 0.6, 0.45, { fontSize: 19, bold: true, fontFace: HF, color: c[2] });
    s.addText(bullets(c[3], { paraSpaceAfter: 14 }), { x: x + 0.3, y: 2.5, w: cw - 0.6, h: 3.4, fontFace: BF, fontSize: 15.5, color: C.ink, valign: "top", margin: 0, isTextBox: true });
  });
  footer(s, "Multi-expert review synthesis (14 parts)");
  s.addNotes(`الشريحة 21 — المراجعات المستقلة (14 جزء، كل جزء خبير).
الدرجة 7/10، والهدف 8–8.5 بعد الإصلاحات المسجلة.
القوة: التسجيل المسبق ونظرة test واحدة، تقييم مضبوط من غير bugs، ومفيش تسريب داتا.
الضعف: إعداد الفيدرالي كان ضعيف، ادعاءات الخصوصية كانت مبالغ فيها، وأدلة الجهاز من CPU سيرفر.
اللي اتصلح: سلم الضبط والبدء من موديل مُدرب مسبقاً، threat model ومصفوفة الخصوصية، والفرع الواقعي بقى أساسي.`);
}

// 22. Timeline
{
  const s = light();
  header(s, "Timeline", "What is running now and when results arrive");
  const days = ["27 Sep", "28", "29", "30", "1 Oct", "2", "3", "4", "5", "6", "7", "8"];
  const gx = MX + 3.4, gw = W - MX - gx, dw = gw / days.length, gy = 1.55;
  days.forEach((d, i) => {
    txt(s, d, gx + i * dw, gy, dw, 0.3, { fontSize: 11, color: C.muted, align: "center" });
    if (i % 2 === 0) s.addShape(pres.shapes.RECTANGLE, { x: gx + i * dw, y: gy + 0.35, w: dw, h: 4.2, fill: { color: "F6F8F8" }, line: { type: "none" } });
  });
  const rows = [
    ["Backbone contest", 0.0, 2.0, "done", C.slate],
    ["Cloud models (full pool)", 2.35, 3.35, "finishing", C.teal],
    ["Pretraining", 2.55, 3.3, "done", C.slate],
    ["Federated tuning", 2.0, 4.0, "running", C.teal],
    ["Device runs", 2.7, 5.4, "running", C.teal],
    ["Main federated + DP runs", 3.9, 5.6, "estimate", C.gray],
    ["One validation read", 5.4, 6.8, "estimate", C.amber],
    ["Single test look → tables", 6.8, 10.2, "estimate", C.amber],
  ];
  const rh = 0.46;
  rows.forEach((r, i) => {
    const y = gy + 0.45 + i * (rh + 0.04);
    txt(s, r[0], MX, y, 3.3, rh, { fontSize: 13.5, color: C.ink, valign: "middle" });
    const bx = gx + r[1] * dw, bw = (r[2] - r[1]) * dw, est = r[3] === "estimate";
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: bx, y: y + 0.07, w: bw, h: rh - 0.14, rectRadius: 0.08, fill: { color: r[4], transparency: est ? 35 : 0 }, line: est ? { color: C.amberDark, width: 1, dashType: "dash" } : { type: "none" } });
    const inside = bw >= 0.42;
    s.addText(r[3], { x: inside ? bx : bx + bw + 0.06, y: y + 0.06, w: inside ? bw : 0.9, h: rh - 0.12, fontFace: BF, fontSize: 9.5, bold: true, color: !inside || r[4] === C.gray || r[4] === C.amber ? C.ink : C.white, align: inside ? "center" : "left", valign: "middle", margin: 0, isTextBox: true });
  });
  const tx = gx + 3.5 * dw;
  s.addShape(pres.shapes.LINE, { x: tx, y: gy + 0.35, w: 0, h: 4.2, line: { color: C.amberDark, width: 1.5, dashType: "dash" } });
  txt(s, "meeting", tx + 0.06, gy + 0.38, 1.0, 0.25, { fontSize: 10, color: C.amberDark, bold: true });
  txt(s, "Future dates are estimates: validation read ≈ 2–3 Oct · test look ≈ 4–5 Oct · final tables ≈ 6–7 Oct.", MX, 6.3, W - 2 * MX, 0.4, { fontSize: 13, italic: true, color: C.muted });
  footer(s, "Master plan (measured run rates); decision log run status to 30 Sep 08:30 UTC");
  s.addNotes(`الشريحة 22 — الجدول الزمني.
خلص: مسابقة الموديل، والتدريب المسبق خلص ميزانيته. موديلات السحابة الكاملة: واحد خلص (بذرة 2027) والتاني وصل لآخر الميزانية. شغال: ضبط الفيدرالي (مرحلة شكل الـ learning rate) وتشغيلات الجهاز (الموديل المحلي بدأ، وبعده الـ fine-tuning).
بعد الضبط: التشغيلات الفيدرالية و DP الرئيسية.
التقديرات: قراءة validation واحدة حوالي 2–3 أكتوبر، نظرة الـ test حوالي 4–5 أكتوبر، والجداول النهائية حوالي 6–7 أكتوبر. اتأخرنا حوالي يوم عن الخطة لأن تشغيلات الضبط أبطأ لما بتشارك الـ GPU.`);
}

// 23. Discussion 4
discussion(4, "Priorities if time is short", [
  ["If we must cut, what should go first?", "Never cut: federated, local-only and cloud fine-tune with two seeds, the lock and the single test look."],
  ["Is one browser run on a laptop enough device evidence?", "Or should we add a second machine?"],
  ["Can you pre-approve the Queen's AI-use disclosure?", "Required front matter for the thesis."],
], `ديسكشن 4 — الأولويات لو الوقت ضيق.
1) لو لازم نقطع، نقطع إيه الأول؟ مش هنقطع: الفيدرالي والمحلي و fine-tune السحابة ببذرتين، والقفل، ونظرة الـ test الواحدة. أول حاجات تتقطع: البذور الزيادة للوحات، و ε=1 لو اضطرينا.
2) تشغيل واحد في المتصفح على لابتوب كفاية كدليل للجهاز؟
3) يوافق مسبقاً على بيان استخدام الذكاء الاصطناعي المطلوب من Queen's؟`, "cut order is pre-registered");

// 24. Summary
{
  const s = dark();
  header(s, "Summary", "Where the project stands", true);
  const cols = [
    ["Done", C.slate, ["Next category → next product, with reasons", "Compared with the published benchmark", "Features reduced to a device-friendly set", "Backbone chosen by a registered rule"]],
    ["Running", C.teal, ["Federated tuning: +46% so far, after fixing our first setup", "Pretraining finished → federated from it next", "Full-pool cloud models finishing", "Device runs (local-only model running)"]],
    ["Next", C.amber, ["One validation read (≈ 2–3 Oct, est.)", "Single test look (≈ 4–5 Oct, est.)", "Browser benchmark", "Thesis tables (≈ 6–7 Oct, est.)"]],
  ];
  const cw = (W - 2 * MX - 0.6) / 3;
  cols.forEach((c, i) => {
    const x = MX + i * (cw + 0.3), y = 1.7;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w: cw, h: 4.3, rectRadius: 0.08, fill: { color: "1D3640" }, line: { type: "none" } });
    chip(s, c[0].toUpperCase(), x + 0.3, y + 0.3, 1.4, c[1], c[1] === C.amber ? C.ink : C.white);
    s.addText(bullets(c[2], { paraSpaceAfter: 12 }), { x: x + 0.3, y: y + 0.85, w: cw - 0.6, h: 3.3, fontFace: BF, fontSize: 15.5, color: C.onDark, valign: "top", margin: 0, isTextBox: true });
  });
  pageNo += 1;
  s.addText(String(pageNo), { x: W - MX - 0.6, y: H - 0.42, w: 0.6, h: 0.3, fontFace: BF, fontSize: 10, color: C.onDarkMuted, align: "right", margin: 0, isTextBox: true });
  s.addText("Decisions today: task and benchmark · Edge headline · federated effort · DP levels · priorities", { x: MX, y: 6.3, w: W - 2 * MX, h: 0.4, fontFace: BF, fontSize: 14, bold: true, color: C.amber, margin: 0, isTextBox: true });
  s.addNotes(`الشريحة 24 — الملخص.
خلص: انتقلنا من الفئة للمنتج بأسباب واضحة، قارنا بالبحث المنشور، صغرنا الفيتشرز لمجموعة الجهاز يحسبها، واخترنا الموديل بقاعدة مسجلة.
شغال: ضبط الفيدرالي بعد ما صلحنا الإعداد الأول (+46% لحد دلوقتي)، التدريب المسبق خلص والفيدرالي هيبدأ منه، موديلات السحابة الكاملة قربت تخلص، وتشغيلات الجهاز (الموديل المحلي شغال).
الجاي: قراءة validation واحدة (حوالي 2–3 أكتوبر)، نظرة test واحدة (حوالي 4–5 أكتوبر)، اختبار المتصفح، وجداول الرسالة (حوالي 6–7 أكتوبر).
اختم بالقرارات: المهمة والـ benchmark، إجابة Edge الرئيسية، مجهود الفيدرالي، مستويات DP، والأولويات.
سجل التحديثات: final v1 — 2026-09-29 (DECISION_LOG لحد الصف 228). final v2 — 2026-09-30 (لحد الصف 251) — التفاصيل في نوتس آخر شريحة.`);
}

// Backup
{
  const s = dark();
  s.addText("BACKUP", { x: MX, y: 2.8, w: 8, h: 0.4, fontFace: BF, fontSize: 14, bold: true, charSpacing: 3, color: C.amber, margin: 0, isTextBox: true });
  s.addText("Details for likely questions", { x: MX, y: 3.25, w: 11, h: 0.9, fontFace: HF, fontSize: 36, bold: true, color: C.white, margin: 0, isTextBox: true });
  pageNo += 1;
  s.addNotes("شرايح احتياطية: الـ baselines، ومنحنى الـ learning rate كامل. ما تعرضهاش إلا لو اتسألت.");
}
{
  const s = light();
  header(s, "Backup — baselines", "Simple rules vs the contest models (validation)");
  s.addChart(pres.charts.BAR, [{ name: "MRR@20", labels: ["Popularity", "Last viewed item", "Session kNN", "GRU", "SASRec (cloud)"], values: [0.0194, 0.0711, 0.1114, 0.1569, 0.1612] }], {
    x: MX, y: 1.5, w: 8.6, h: 4.9, barDir: "bar", chartColors: [C.gray, C.amber, C.gray, C.slate, C.teal], varyColors: true, barGapWidthPct: 40,
    showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0.000", dataLabelFontSize: 14, dataLabelColor: C.ink,
    valAxisMinVal: 0, valAxisMaxVal: 0.19, valAxisHidden: true, valGridLine: { style: "none" }, catGridLine: { style: "none" },
    catAxisLabelColor: C.body, catAxisLabelFontSize: 14, catAxisOrientation: "maxMin", showLegend: false,
    showTitle: true, title: "Validation MRR@20, main cohort", titleFontSize: 12, titleColor: C.ink,
  });
  stat(s, 9.6, 1.6, W - MX - 9.6, 1.7, "44%", "of the cloud reached by “last viewed item” — for free", C.amberLight, C.amberDark);
  footer(s, "Decision log (baselines and contest, 28 Sep)");
  s.addNotes("احتياطي — الـ baselines: popularity 0.019، آخر منتج 0.071، session kNN 0.111، GRU 0.157، SASRec 0.161. آخر منتج اتشاف بيوصل لحوالي 44% من السحابة ببلاش.");
}
{
  const s = light();
  header(s, "Backup — client learning-rate screen", "Rise, peak at ×8, then fall");
  s.addChart(pres.charts.LINE, [{ name: "Calibration MRR@20", labels: ["×0.5", "×1", "×2", "×4", "×8", "×16"], values: [0.020263, 0.023473, 0.025562, 0.027570, 0.028796, 0.022909] }], {
    x: MX, y: 1.5, w: W - 2 * MX, h: 4.9, chartColors: [C.teal], lineSize: 3, lineDataSymbol: "circle", lineDataSymbolSize: 10,
    showValue: true, dataLabelPosition: "t", dataLabelFormatCode: "0.000", dataLabelFontSize: 13, dataLabelColor: C.ink,
    valAxisMinVal: 0.018, valAxisMaxVal: 0.031, valAxisHidden: true, catAxisLabelColor: C.body, catAxisLabelFontSize: 14,
    valGridLine: { style: "none" }, catGridLine: { style: "none" }, showLegend: false,
    showTitle: true, title: "Federated, calibration users, full training budget", titleFontSize: 12, titleColor: C.ink,
  });
  footer(s, "Learning-rate selection files; ladder register");
  s.addNotes(`احتياطي — منحنى الـ learning rate: بيطلع لحد ×8 (0.029) وبعدين بينزل عند ×16. الفائز جوه الشبكة.

سجل التحديثات (Update log)
final v1 — 2026-09-29: أول نسخة نهائية (DECISION_LOG لحد الصف 228، السلم لحد رفع الـ learning rate).
final v2 — 2026-09-30 (DECISION_LOG لحد الصف 251):
- مرحلة الـ server optimizer خلصت: FedAdam 0.034 على مستخدمين الضبط (+18.9% عن المرحلة اللي قبلها، +46% عن البداية). اتحدثت شرايح 15 و 16 و 24.
- شريحة 15: "اللي حصل" بقت حوالي 20% للإعداد الأول (بدل 25%، اللي كان بعد أول تحسين).
- التدريب المسبق خلص ميزانيته (شرايح 16 و 18 و 22 و 24).
- موديلات السحابة الكاملة: بذرة 2027 خلصت، وبذرة 2026 وصلت لآخر الميزانية (شريحة 22).
- اختيارات الجهاز اتسجلت، والموديل المحلي بدأ يتدرب (شرايح 22 و 24).
- المرحلة الجاية (شكل الـ learning rate × optimizer العميل) شغالة (شريحة 16).
- مواعيد القراءة زي ما هي (تقديرات).`);
}

pres.writeFile({ fileName: OUT }).then(f => console.log("wrote " + f + " slides=" + pageNo));
