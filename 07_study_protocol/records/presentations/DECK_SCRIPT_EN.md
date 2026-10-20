# Speaker script — PPSI supervisor meeting (30 September 2026)

About 25–30 minutes of talking, plus discussion. Each block is roughly what to say on that slide; speak naturally, don't read.
On the dark "Discussion" slides, stop, ask the questions, and listen.

---

**Slide 1 — Title**
"Thank you for the time today. The question of the project is simple: how much recommendation quality survives when a
shopper's data stays on the device? I'll compare three ways Microsoft Edge could learn from shopping behaviour — the cloud,
federated learning, and a small model on the device — show where we are, and ask for your opinion on a few decisions."

**Slide 2 — Three ways Edge could learn**
"In the cloud setting, browsing events go to the server and one large model is trained and served there. That gives the best
quality and no privacy. In the federated setting, events stay on the device; the device trains locally and sends only model
updates, optionally with differential privacy. In the on-device setting nothing is uploaded: a small model is downloaded and
adapted locally. Our headline measure is retention — the quality of a deployment divided by the quality of the cloud model."

**Slide 3 — Four phases**
"The work went through four phases, each answering one question. First, can we predict the next category. Second, how good can a
cloud model get on the next product. Third, which input columns actually help. And fourth, the deployment study, which is running
now — its main results are sealed until the planned read."

**Slide 4 — Phase 1**
"The next-category model worked — our GRU beat both baselines — but we could not tell whether that was good, because nothing had
been published for that task on this dataset, and about 82% of the metric is solved by simply repeating the current category. A
real published benchmark exists for next product on REES46, so, as we agreed, we moved to next product."

**Slide 5 — Phase 2**
"Against the published GRU4Rec results we are close but lower: 0.43 versus 0.53 Recall at 20, and 0.15 versus 0.20 MRR. Our setup
is harder: we rank the whole catalogue, new products always count as misses, and we predict the following week rather than the
last day, with far less compute. This is not a same-protocol re-run — which is one of the questions I'll ask you."

**Slide 6 — Phase 3**
"We went from 58 input columns to 24, and 21 of those can be computed on the device itself. Removing all side features clearly hurt,
and adding more did not help, so we kept this set. It matters for the device story: the model only uses what a browser can see."

**Slide 7 — What changed from the proposal**
"Here is what changed and why. The task became next product viewed — harder, standard, and it has a benchmark. The data is REES46,
where every user can be simulated as one device. The metric is per-user MRR at 20, which matches a browser serving one user.
Federated methods are the proposal's list minus FedBN, because our model uses layer normalisation, not batch normalisation. And
'various privacy levels' became differential privacy at epsilon 8 and 1, so privacy is measurable."

**Slide 8 — Data and task**
"Every prediction ranks all 158,486 products. The main cohort has 131,072 users, each one a simulated device, and a full-pool cloud
reference uses 1.35 million users. The split is chronological: training until 15 November, validation the next week, and a test
week that stays sealed. We only tune on a separate calibration subset, and the test week is opened once."

**Slide 9 — Fairness**
"Six rules keep the comparison fair: rules and comparisons are written before results are seen; every method gets the same data,
features, evaluator and training budget; we rank the full catalogue; statistics are paired on the same users with two seeds for
the main methods; the test week is opened once; and every stage is checked by an independent review before use."

**Slide 10 — Discussion 1**
"Three questions here. Do you agree next product is the right task? Should we re-run the published benchmark protocol on our new
server to know directly whether we are stronger or weaker? And retention against which cloud: the matched 131 thousand users, or
the 1.35 million full pool — we plan to report both; which should lead?"

**Slide 11 — Backbone contest**
"We ran a pre-registered contest between GRU and SASRec on the main cohort with two seeds. Under the rule we fixed in advance — the
best validated checkpoint — SASRec won, 0.161 against 0.157, stable across seeds. The honest caveat: at the fixed end of training
GRU is ahead, because SASRec peaks early and then overfits. We report both, and we never claim SASRec is simply better."

**Slide 12 — Model sizes**
"The cloud model has 84 million parameters; the device model 21 million. For federated learning, one visit means downloading and
uploading the model: 671 MB for a cloud-size model, 166 MB for the device model, 104 with 8-bit uploads, and about 2 MB if we freeze
the item tables. Our cap is 170 MB, so only the device-size model fits."

**Slide 13 — Comparison matrix**
"Every method is placed by what leaves the device: raw events in the cloud; events once when a cloud-trained model is shipped;
events only from opted-in users in the pretrained branch; model updates in federated learning; noised updates with DP; and nothing
on the device. Every method is also compared with 'last viewed item', a free rule with no training."

**Slide 14 — Discussion 2**
"Which deployment should be the headline answer for Edge — plain federated, pretrained-then-federated, or a cloud model fine-tuned
on the device? The independent review recommends the pretrained branch. And should 'last viewed item' be the floor every method
must beat?"

**Slide 15 — Our first federated setup**
"I want to be transparent: our first federated setup was wrong. We started from an untrained model, reused the cloud's training
settings, and had no server optimiser, with the client optimiser reset every visit. On calibration users federated learning reached
only about a quarter of the cloud — even below 'last viewed item'. That is a warning from tuning users, not a final result. We
responded with a registered tuning ladder, already plus 22.7%, and a start from a pretrained model, all before reading any final
result. The gap mainly reflected our setup, not federated learning itself."

**Slide 16 — Improvement ladder**
"The ladder changes one thing at a time and keeps a change only if it gains at least 0.002. Tuning the client learning rate took us
from 0.023 to 0.029. The server optimiser stage is running now, then the learning-rate shape and client optimiser, local passes and
batch size, the pretrained start, and freezing the item tables."

**Slide 17 — Privacy**
"We use differential privacy at epsilon 8, moderate, and epsilon 1, strong. It is a per-user guarantee: one user's whole history
barely changes the model. Our adversary is an honest-but-curious browser vendor. We state the limits: without DP an update can
reveal which products a user viewed, secure aggregation is not simulated, and freezing the item tables reduces that leak but does
not remove it."

**Slide 18 — How a browser would ship it**
"This is how a browser would actually ship it: pretrain on about 1.2 million users who agreed to share data, none of them in the main
cohort; then run federated learning on the private users starting from that model; then freeze the item tables so only a small
encoder moves, about 1 MB each way. It directly fixes our first mistake, and it is judged against the full-pool cloud. Pretraining
is running now."

**Slide 19 — Discussion 3**
"How much tuning effort is fair for federated learning — our ladder stops when a stage gains less than 0.002? Are epsilon 8 and 1
enough, or do you want a curve — one more level costs about half a day? And is pretraining on opted-in users a realistic assumption
for Edge? If yes, it becomes our headline answer."

**Slide 20 — On-device and cold start**
"On the device we have a local-only model as a control, a cloud model fine-tuned locally, and the pretrained model fine-tuned locally,
which is the cleanest download-only option. The model is exported to ONNX in full and 8-bit precision; latency on a single CPU thread
is running, and a browser test in Edge is next. For cold start we look at users who never took part, users with no history, and
short versus long histories, with a controlled test that cuts histories to fixed lengths."

**Slide 21 — Independent reviews**
"We had the work reviewed independently in fourteen parts. The strengths: pre-registration and one sealed test look, exact
evaluation with no bug found, no data leakage. The weaknesses: the federated setup was too weak, privacy claims were over-stated,
and device evidence comes from a server CPU. Those are already fixed or registered, so the review expects 8 to 8.5 out of 10 after
the fixes, from 7 today."

**Slide 22 — Timeline**
"The contest is done. The full-pool cloud models, pretraining, federated tuning and device runs are running now. We expect the main
federated and DP runs around 1–2 October, one validation read around 2–3 October, the single test look around 4–5 October, and the
final tables around 6–7 October. These are estimates."

**Slide 23 — Discussion 4**
"If we must cut, what should go first? We will never cut federated, local-only and cloud fine-tune with two seeds, the lock and the
single test look. Is one browser run on a laptop enough device evidence? And Queen's requires a disclosure of AI tool use in the
thesis — could you pre-approve it?"

**Slide 24 — Summary**
"To summarise: we moved from next category to next product for good reasons, compared with the published benchmark, reduced the
features to a device-friendly set, and chose the backbone by a registered rule. Federated tuning, pretraining, the full-pool cloud
and the device runs are running. Next are the validation read, the single test look, the browser benchmark and the thesis tables.
Thank you — I'd value your views on these decisions."

**Backup slides (26–27)** — use only if asked: the baselines ("last viewed item reaches 44% of the cloud for free") and the learning-rate
curve ("rise, peak at ×8, then fall").
