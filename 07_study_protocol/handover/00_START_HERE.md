# 00 — START HERE: PPSI V4 handover

> **UPDATE v2 (2026-10-08): read `14_UPDATE_v2_20261008.md` first.** It covers DL 436–553: the T0a sitting is complete,
> confirmatory results are final, the FedAdam-tau finding, the exploratory runs still on the server, and what remains.

*Privacy-Preserving Shopping Intelligence, version 4 (PPSI V4). MSc project, Queen's University, with a Microsoft supervisor.*
*Owner: Ahmed Abdelhamed (GitHub `AhmedAbdelhamed01`). Written 2026-10-03 from the project records up to decision-log row
435 (2026-10-03, about 18:35Z server clock). Where this folder and a primary record disagree, the primary record wins
(contract, lock, selection file, decision-log row). Record the correction as a new decision-log row.*

> **ملخص بالعربية.** هذا المجلد يسلّم مشروع PPSI V4 لزملاء المالك. السؤال البحثي: كم من جودة التوصية يبقى عند
> الانتقال من نموذج سحابي إلى تعلّم اتحادي (يُرسل الأوزان فقط) أو إلى نموذج صغير على الجهاز، مع الحفاظ على الخصوصية.
> حتى 3 أكتوبر 2026 انتهت كل تشغيلات المرحلة الأولى على بيانات REES46، وتمت قراءة VALIDATION الوحيدة (السطر 400 في
> سجل القرارات)، وكُتب ALL_METHODS_LOCK (السطر 402)، وبُنيت ملفات TEST دون تقييم أي نموذج. نظرة TEST الوحيدة مؤجلة
> بقرار المالك لتُنفَّذ في جلسة واحدة مع المرحلة الثانية (PREG2)، وقد سُجّلت المرحلة الثانية ونُشر كودها (code_p2r1)
> ولم يبدأ أي تشغيل لها بعد. تجارب المقارنة الخارجية (Beauty/Sports/Toys و ML-1M على GPU و Amazon 2023) جارية.
> ابدأ بهذا الملف، ثم 08 (خطوات الإكمال)، ثم 03 (قواعد التسجيل المسبق). لا تفتح أي قيمة TEST، ولا أي قيمة MAIN
> للمرحلة الثانية، قبل الخطوة المسجلة.

---

## 1. What the project is (one paragraph)

The project answers the proposal's question: **"How much shopping recommendation quality can be retained when moving from
cloud-based to on-device or federated models?"** (amendment 04 header; DL 86). It simulates a browser vendor (the
"Edge" framing) that could serve next-product recommendations in three ways: (a) **cloud-only** (raw events go to a
server model), (b) **federated, weights only** (devices train locally and send model updates), or (c) **a small local
model** (shipped once, optionally fine-tuned on the device). Each REES46 user is one simulated device. The task is
next-product prediction over the full catalogue of 158,486 products; quality is client-macro MRR@20; every answer is a
retention ratio with a paired user-bootstrap CI and **no pass / fail threshold** (DL 86). The work is pre-registered:
every rule is hashed before the data it governs is read, VALIDATION is read once, TEST is looked at once.

## 2. Status at 2026-10-03 (one page)

| Stream | State (label) | Evidence |
|---|---|---|
| Backbone contest (GRU vs SASRec) | DONE. SASRec D256_B3 selected under rule T1; ARCHITECTURE_LOCK written (MEASURED, final) | DL 129–131, 138 |
| Phase 1 REES46 deployment study: training | DONE. Every registered MAIN run finished (cold line, pretrained "warm" line, DP chains, device arms, panels) (MEASURED) | DL 387 |
| Phase 1: analysis freeze | ANA_SCRIPT_FREEZE **v3** registered (`59c83c86…`); v1 and v2 superseded after two value-free tool refusals, never read (MEASURED) | DL 388–397 |
| Phase 1: the single VALIDATION read | DONE (F, 15:04–15:14Z). Output copied to `results/VALIDATION_v3/` (MEASURED, VALIDATION only) | DL 398–400 |
| Phase 1: ALL_METHODS_LOCK | REGISTERED `04a3cd19…`; TEST_EXPOSURE_REGISTER entry 5 (MEASURED) | DL 402 |
| Phase 1: TEST manifests | BUILT counts-only (H gate → build → final), register entry 6; **no model scored on TEST; no TEST value exists** (MEASURED) | DL 403, 406, 407 |
| Phase 1: the single TEST look | **ON HOLD by the owner** (DL 406). It will run in ONE sitting together with Phase 2 ("T0a") | DL 406, 410 |
| D1 diagnostic | DONE: warm FL moved θ by only 0.5–0.7 % of θ_pre (S8) → K21 ≈ 0 is a recipe effect (MEASURED, exploratory) | DL 407 |
| Phase 2 (PREG2: can a better warm-FL recipe add value over θ_pre?) | REGISTERED (`be0561ab…`); contrasts `P2_CONTRASTS_v3` (`673b547d…`); code tree `code_p2r1` deployed; **no Phase-2 run yet** (MEASURED) | DL 411, 427, 435 |
| Phase 2 code still open | P2-07/13 reeval_p2, P2-12 analysis tool, the RUN-1 lane review, WB3 / LoRA wiring, P2-08a DP chain, FT launch path for W*, P2-00 lock tool (PENDING) | DL 434; §3 of file 05 |
| External benchmarks | Amendment 07 + addenda B12–B16 registered. S3 Beauty / Sports / Toys: all Wave-1 jobs done, **not yet locked**. ML-1M: restarted from scratch on the A40 GPU (B16), running. Wave 2 (Amazon 2023 Video_Games, Baby_Products): running on CPU (MEASURED / RUNNING) | DL 369, 377, 433 |
| K16 INT8 (device quantisation) | Pass 1 done on a 16,384-user subset (fallback, disclosed); pass 2 (latency only) PENDING after the Wave-1 drivers exit | DL 371–372, 401 |
| Device measurement (Phase 2, Part B) | Page + server Edge script written and reviewed; Edge runs on the server proven feasible; no benchmark run yet | DL 416, 419–420 |
| Living thesis reference | Captured up to DL 353; **DL 354–435 not yet captured** (PENDING) | thesis_reference CHANGELOG |
| Thesis text | Draft chapter 3 (methods), chapter 4 (setup) and an outline exist, written before any result | DL 324; `company/thesis_draft/` |

**INFERENCE (schedule, not a result):** the Phase-1 + Phase-2 TEST sitting was estimated at about 2026-10-17 … 10-21
(PREG2 §6.3); ML-1M final about 2026-10-05/06 on the GPU (B16.2).

## 3. Reading order

1. This file.
2. `08_REMAINING_WORK_RUNBOOK.md` — what to do next, in order, with preconditions.
3. `03_GOVERNANCE_PREREGISTRATION.md` — the rules you must not break.
4. `01_RESEARCH_QUESTION_AND_DESIGN.md` and `02_DATA.md` — what is compared and on what data.
5. `04_RESULTS_VALIDATION.md` — the single VALIDATION read and its honest reading.
6. `05_PHASE2_FOLLOWUP.md` and `06_BENCHMARKS.md` — the two open streams.
7. `07_SERVER_AND_OPERATIONS.md` — how the server and the job lanes work.
8. `09_THESIS_WRITING_GUIDE.md` and `10_PRESENTATION_GUIDE.md` — write-up.
9. Reference: `11_FILE_MAP.md`, `12_GLOSSARY.md`, `13_OPEN_ISSUES_RISKS.md`.

**`HANDOVER/reference/`** holds byte copies of the key primary documents (VALIDATION report, claim register, PREG2,
P2_CONTRASTS_v3, the Phase-2 implementation plan, amendment 07 + B12–B16, the endgame README, the benchmark lock runbook,
the Fable endgame / Phase-2 gate review, the TEST exposure register, the thesis-reference STATE, the P2 device README, the
thesis outline draft and the 09-30 team handover). The originals stay authoritative; the copies are for convenient reading.

The older narrative `company/handover/PPSI_V4_TEAM_HANDOVER.md` (written 2026-09-30, state at DL 268) is still accurate
for everything before DL 268 and is more detailed on the FL ladder, the privacy design and the incidents. This folder
supersedes it for everything after DL 268.

## 4. The ten things to know first

1. **The decision log is the authority.** `company/DECISION_LOG.md` (about 435 rows, append-only). Cite rows by the `#`
   column, not the file line. Known quirks: row 169 exists twice (the second is cited **169b**, DL 171); rows 354–363
   exist twice (the second copies are monitoring-only rows of a cron session, DL 377 erratum); row **377 exists twice**
   (the erratum and the Wave-2 decision); several rows carry local time (UTC+3) labelled Z (corrections in DL 392 and
   DL 414). Always take new time stamps from the server or `date -u`.
2. **Some values are still sealed. Do not open them.** Every RETRO_TEST (TEST) output; every Phase-2 MAIN VALIDATION value
   (PREG2 §6.3 S8) until both sitting analyses are written; every benchmark holdout value before that dataset's LOCK and
   single `--final`. The unseal gates are not a security boundary: the shared server account can open anything (DL 90
   F3). The discipline is personal.
3. **Register first, read later.** A rule, grid, contrast or arm is written into an amendment (or a DL row), hashed and
   logged **before** any value it governs exists. Amendments are never edited; corrections are new files. Anything added
   after the 2026-10-03 VALIDATION read is `POST_VALIDATION_EXPLORATORY` (DL 400; 06L_C C5).
4. **There is exactly one TEST sitting left, and it covers both studies (T0a).** Phase-1 scoring → Phase-2 scoring →
   Phase-1 analysis → Phase-2 analysis; nothing under `runs_ds3/TEST`, `out_test_*` or `p2_out_test_*` is opened before
   both RETENTION_RESULT files exist; there is never a second RETRO_TEST sitting (PREG2 §6.3 S1–S8; DL 410).
5. **The headline VALIDATION result is mixed, and it is honest.** On the registered recipe the pretrained federated line
   keeps 88.81 % [88.27, 89.36] of the full-data cloud model (K20b), but federated training did not add anything over the
   pretrained model alone (K21 = −0.000433 [−0.000758, −0.000101]). The D1 diagnostic shows the warm FL barely moved the
   weights (DL 407), so K21 is reported "under the registered recipe", not as "FL cannot help". That is why Phase 2
   exists. (VALIDATION values, max-of-marks; TEST is the only unbiased number.)
6. **Frozen code stays frozen.** The evaluator `code/v4eval/*.py` (evaluator hash `61b64e61…`), the frozen analysis tools
   (`tools/retention_report.py` `9a3b2c2a…`, `tools/ana_block_index.py` `6e7043bb…`, `tools/test_exposure_register.py`
   `7d59e828…`) and every deployed server tree are immutable. A code change goes into the local `code/`, gets tests with
   negative controls and an independent review, and is deployed as a **new** tree beside the old one, with a byte-identity
   check of the Phase-1 paths.
7. **The server is shared and has no SSH.** Lobot (`lobot.cs.queensu.ca`, group server `group-digi2026-g8`), one Linux
   account `jovyan` for the whole team. Jobs run as detached "lanes". Each teammate needs their **own** Jupyter token;
   tokens are never shared, printed or copied onto the server. Never rewrite a running lane script; never touch another
   teammate's processes. GPU precedence: H (TEST) steps > Phase-2 lanes > the ML-1M benchmark driver (PREG2 S7; B16.5).
8. **Every statement carries a label.** MEASURED (read from a record), REPORTED (stated by someone else, not re-checked),
   INFERENCE (an interpretation), PLAN, PENDING. An ETA is never a result. Result labels: CONFIRMATORY (registered
   contrast under its registered rule), DESCRIPTIVE, POST_VALIDATION_EXPLORATORY, CONFIRMATORY_P2 (PREREGISTERED_FOLLOWUP),
   TERTIARY_P2.
9. **The data are frozen.** `rees46_nextitem_v1` (version hash `860e5a1c…8dbd`, 409 files) lives read-only at
   `/home/jovyan/shared_data/rees46_nextitem_v1`. Any preprocessing change means a new data version and the owner's
   approval. December 2019 data were **not** used (owner decision O2 = NO, DL 415).
10. **Mandatory disclosures go into the thesis verbatim** (list in file 09): central DP with a trusted aggregator; non-DP
    federated updates reveal the viewed items and secure aggregation is not simulated; θ_pre is one seed; PRE trained on
    9× more users than C_BIG; C_FULL is a larger (D256) model; the backbone was "selected under rule T1", never "SASRec
    beats GRU"; the B13 → B15 and B16 benchmark changes; the C2 decision history YES → NO → YES; the VALIDATION → TEST
    order and the post-validation labels.

## 5. People, roles and how the work was done (honest summary)

- **The owner** took every binding decision (scope, data, sharing, approvals). His decisions are logged as "OWNER" rows.
- **Operations, workers and reviewers were AI agents** run by the owner in Claude Code: an operations lead (Claude Opus /
  Sonnet sessions) ran the server, wrote most amendments, dispatched work packets and kept the decision log; worker agents
  implemented one packet each (`company/tasks/`, `company/handoffs/`); independent **read-only reviewer agents** checked
  each change (`company/reviews/REV-*.md`); "Fable" (a Claude Fable 5.1 model instance) acted as chief scientist and gate
  reviewer (`company/reviews/FABLE_*`), and for a while as owner-proxy on blocking decisions (DL 37). Some early code
  packets and one overnight server session (DL 162–167) were done by OpenAI Codex before the owner revoked Codex access
  (RESUME_HERE §0 item 2). This matters for the Queen's AI-use disclosure (file 09 §5).
- **The supervisor** saw the 30 September deck (`company/presentations/`); the meeting outcome is not in the records.

## 6. What you received and what you did not

- The ZIP holds the local project folder: records, contracts, gates, code, tools, scripts, the VALIDATION output and these
  documents. It does **not** hold the frozen dataset, model checkpoints, run directories or server logs: they exist only on
  the server (file 11 §3).
- Some registered files exist only on the server (for example `contracts/fl/FL_WARM_SELECTION_S7|S8.json`,
  `contracts/dp/S_RECORD_WARM_FROZEN.json`, `contracts/device/SELECTION_FT_FA.json`, `contracts/strata/STRATA_CUTPOINTS_MAIN.json`).
  Their shas are in the decision log; read-only copies of S7 / S8 are in `../thesis_reference/sources/local/S579_*`, `S580_*`.
- Before you zip anything further: exclude `_pytest_tmp/` (about 300 MB of test scratch), `code/_pytest_tmp/`,
  `code/.train03r1_tmp/` (unreadable ACLs) and every `__pycache__/`. The Lobot token is stored outside the project
  (`C:\Users\Ahmed\AppData\Local\Temp\lobot_upload\.tok`); no token is in the project files (MEASURED by a scan on
  2026-10-03).
