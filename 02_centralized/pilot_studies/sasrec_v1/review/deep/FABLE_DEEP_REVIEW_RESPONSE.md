# Fable deep review + research (owner-requested, max effort), verbatim

- agent type: fable-gate-reviewer (effort max; has Bash but instructed not to use it)
- transcript model ids: ['claude-fable-5-1']
- tool calls: {'Read': 21, 'Glob': 6, 'Grep': 10, 'WebFetch': 24, 'WebSearch': 1}

---

ROUTING: claude-fable-5-1
TOOLS_USED: Read, Grep, Glob, WebFetch, WebSearch

**ROOT_CAUSE** (ranked; answers owner (a))
1. **Common-mode drift of the tied item table under dense-CE Adam, amplified 16× into the residual stream.** Every step all 158,486 output rows get the softmax-denominator gradient p_j·q [inference from `sasrec_model.py:343`]; AdamW's per-coordinate normalisation θ←θ−γ·m̂/(√v̂+ε) [literature: https://docs.pytorch.org/docs/2.14/generated/torch.optim.AdamW.html] turns those tiny, sign-consistent gradients into ≈γ-sized steps along −sign(q̄) for every row, so all rows move together. Evidence: row norms 161.7 in the 0-, 1–4- and 5–49-target buckets alike (p99 161.7), 164.5 for 500+ [evidence: `diag_item_embeddings.json:9-40`]; item-specific content is only √(164.5²−161.7²)≈30 vs a 162 common offset; Adam v 3e-17 for rare rows (√v=5.5e-9<ε → still 0.35·γ steps) vs 7e-8 for frequent [evidence: same file]. The head is shift-invariant, so the loss never pushes back; wd 1e-5 is nil. Harm comes only through tying: input z=16·E+P (`sasrec_model.py:269,275,294`) puts a ≈2,590-norm vector common to all tokens into a Pre-LN stream with no input LN (`:281`); LN1/final LN cannot remove a shared direction, so keys/queries and the final query collapse toward δ̂ and item identity is attenuated by ≈|u_j|/|δ|≤0.19 (≈0 for rare items) [inference]. Trajectory matches: loss min 7.54@10k, 8.19@156k, 9.02@320k, 8.93@720k; grad norm 1.85→6.6 [evidence: `trajectories.json:381-394,1111,1931,3931`]; cohort 0.105@80k→0.089@720k [`:4098,4198`]. Mechanism = Gao et al.'s representation degeneration (rare-word embeddings "pushed towards similar directions negatively correlated with most hidden states… move toward infinity", LayerNorm makes it likely) [literature: https://ar5iv.labs.arxiv.org/html/1907.12009].
2. **β2=0.98 (`train_sasrec.py:56`)** copied from the original SASRec (`AdamOptimizer(beta2=0.98)`, sampled BCE, 1 negative) [literature: https://github.com/kang205/SASRec/blob/master/model.py; https://github.com/pmixer/SASRec.pytorch/blob/main/python/main.py]. With 256 targets/step and K=158k an item is a target every 157k/f steps (f=50 → 3.1k steps), far beyond the 50-step β2 memory, so v collapses to the denominator regime between hits. GRU used β2=0.999 (`train_full.py:93`), no √d, Linear+LN input, unnormalised readout, and its rows stayed at 11–13 ≈ init √128 [evidence: `model.py:185,200-201,218`; diag GRU block] [inference on causality].
3. Why literature SASRec-CE (Dross, RecBole) does not collapse: all-position CE with ≤26.7k items gives ~25k targets/step, so v never decays [inference; item counts: https://ar5iv.labs.arxiv.org/html/2309.07602].
4. TF32: at |E|≈10/coord the head-GEMM rounding noise is ≈0.06 logit units — not causal; only a risk if norms stay unbounded [inference]. Dropout/warmup/lr: not causal (loss fell fast to 10k, then rose).
5. Columns: not causal; A1's edge is its un-tied, LN-gained side/user residuals (`:244-250`) that can out-shout the offset [inference], see COLUMNS_DECISION.

**CODE_REVIEW_FINDINGS**
- `sasrec_model.py:269,275,294` √d input scale on a tied table with no input LN — Critical (root path).
- `train_sasrec.py:56` betas (0.9,0.98) — High.
- `train_sasrec.py:875-880` update_agg logs no embedding statistics; pathology found only post hoc — Medium.
- `sasrec_model.py:184,186` item and position init both 0.02, then item ×16: positions start 16× weaker — Medium (removed by fix).
- `train_sasrec.py:696` clip 1.0 is active from update 100 (grad norm 1.85) — Low (Adam is scale-invariant; shifts ε regime only).
- Untying is evaluation-safe: the reused evaluator scores via `logits = model(**kw)` (`gru_v1/code/eval_full.py:309`).
- Must change with the fix: `tests/mutants/model_mutants.py:70-84` (tying negative control), `tests/test_attention.py:31-37,105,135` (reference builder uses `* scale`, item-grad checks), `SASREC_INTERFACES.md:56`, `train_sasrec.py:294` docstring; `bench_sizing.py:82` irrelevant.
- Clean: objective (`sasrec_objective.py:34-49`), schedule (`sasrec_schedule.py:48-64`), layout/causality, precision guards, resume.

**LITERATURE_AND_PROJECTS** (owner (b))
- Original SASRec: tied embeddings ("significantly improves"), BCE 1 negative, Adam 1e-3, dropout 0.2/0.5, d=50; beats GRU4Rec+ Hit@10 by 4.6–13.8% [https://ar5iv.labs.arxiv.org/html/1808.09781]; code: `scale=True`, β2 0.98, no bias, l2_emb [kang205 URL above]; pmixer: `seqs *= dim**0.5`, Adam(0.9,0.98), xavier_normal, last_layernorm [https://github.com/pmixer/SASRec.pytorch/blob/main/python/model.py].
- Dross: SASRec+full-CE (pmixer architecture) ≫ BERT4Rec; Adam 1e-3, batch 128, hidden 64/256; GRU4Rec ML-1M NDCG@10 0.1648 vs SASRec+ 0.1821; largest catalogue 26,744 [URL above]. 2024 follow-up: SASRec-CE vs GRU4Rec-CE, both CE/Adam 1e-3/hidden 64, 15 datasets; retrieved summary reports SASRec ahead on 13/15 (sequential-structure metric; absolute pairs unverified) [https://arxiv.org/html/2408.12008].
- gSASRec: softmax loss "overconfidence is less prevalent"; README: "a separate set of item embeddings improves model convergence" (`reuse_item_embeddings`) [https://ar5iv.labs.arxiv.org/html/2308.07192; https://github.com/asash/gsasrec-pytorch].
- eSASRec: pre-norm gated LiGR, sampled softmax, Adam 1e-3, d 256, dropout 0.2; no GRU4Rec [https://arxiv.org/html/2508.06450].
- RecBole SASRec: `input_emb = LayerNorm(item+pos)`, no √d, Post-LN, tied, no bias, init 0.02, CE, dropout 0.5, hidden 64 [https://raw.githubusercontent.com/RUCAIBox/RecBole/master/recbole/model/sequential_recommender/sasrec.py, .../layers.py, .../properties/model/SASRec.yaml]. T4Rec: tied head, no bias, optional projection to item dim, CE [https://raw.githubusercontent.com/NVIDIA-Merlin/Transformers4Rec/main/transformers4rec/torch/model/prediction_task.py]. Vaswani: share weights and "multiply those weights by √d_model" — but Post-LN renormalises the stream [https://ar5iv.labs.arxiv.org/html/1706.03762].
- Caveat: third-party GRU4Rec ports underperform the official one badly [https://ar5iv.labs.arxiv.org/html/2307.14956]; our ContextGRU (full CE, tied, tuned) is not a weak baseline, and REES46 sessions are short, so published SASRec>GRU margins (≈5–15%) are an upper bound here [inference].

**FIX** (minimal; each item cited above)
Model (`sasrec_model.py`): (1) untie: `self.output_embed = nn.Parameter([K,d])`, init N(0,0.02) drawn after `:195` (common-weight identity A0/A1 preserved); `output_weight()` returns it; `:343` unchanged; keep the arange gate [gSASRec, T4Rec option]. (2) drop √d (`:269,:275,:294`) and add `self.input_norm = LayerNorm(d)` applied after item+pos(+side) and before dropout (`:281,:299`) [RecBole]. Unchanged: Pre-LN blocks, final LN, bias, dropout 0.2, K, positions.
Optimizer (`train_sasrec.py`): (3) `BETAS=(0.9,0.999)` [PyTorch default; GRU; RecBole/Dross]; eps 1e-8, wd 1e-5, clip 1.0 unchanged. (4) after `opt.step()` (`:704`): `W.sub_(W.mean(0,keepdim=True)); b.sub_(b.mean())` on output table/bias — exactly loss-neutral (softmax shift invariance), bounds norms for FP32/TF32 [inference; post-hoc precedent: mean removal in Gao et al./anisotropy work]. (5) log every 100 updates: ‖mean row E[3:]‖, median row norm of E (8k-row sample) and of W_out, mean b.
Schedule: peak 1e-3, floor 1e-4 (mirrors GRU 1e-3→1e-4; SASRec-CE literature 1e-3), warmup 2000, S′ with planned 4 epochs (peak 1, cosine 2, floor 3–4), min_under_cut 3. Precision: TF32 train / strict-FP32 eval retained (owner-approved; v2 parity; no fresh twin — disclosed). Registry v3, Gate C re-opened: I approve conditionally on the diff being exactly (1)–(5) plus the listed test edits and the attested suite passing.

**COLUMNS_DECISION**: Partly-no. Columns are not the cause (A0 fails with items only; A1 degrades too: loss 7.17@80k→7.29@700k). A1's v2 lead is not evidence for the rich features. Relaunch keeps the matched pair: A0v3 = F0 (headline vs GRU F0), A1v3 = F4 (owner's rich question, same design as the GRU project; F1 would need store/model/dim changes the deadline cannot absorb). No new features.

**VERIFICATION_PLAN** (owner (d))
No separate A/B: the two production arms are the check; a 4-variant round costs the ≥30–60 min the 4-epoch plan cannot spare, and 20k-update loss is a weak selector between healthy variants. Pre-registered gates on training signals, both arms:
- 20k updates (≈25 min): mean_loss(20k) ≤ mean_loss(10k) − 0.02 [v2: +0.12 both arms]; mean_grad_norm(20k) ≤ 1.25× value at 6k [v2: 1.64×]; ‖mean row E[3:]‖ ≤ 0.5; median row norms of E and W_out ≤ 3× init; all finite. Any failure → NO-GO: stop, report v2+v3 honestly (no third attempt fits).
- 40k: cohort macro MRR@20 ≥ v2 same-arm +0.010 (A0 ≥ 0.113, A1 ≥ 0.123) — descriptive.
- 80k: "on track to beat GRU" iff A0 cohort ≥ 0.140 [inference: cohort ≈ full-VS +0.000–0.002, `trajectories.json:4158/4283`; healthy curves sit ≥90% of epoch-1 level at half epoch].
- 156,956 (epoch-1 diag): A0 cohort ≥ 0.156 ⇔ ≥ GRU epoch-1 0.1544; < 0.145 → pre-declare "unlikely to exceed 0.1739", continue, report.
If the owner prefers a variant A/B (accepting a 3-epoch relaunch): V1 = recipe; V2 = recipe@5e-4; V3 = tied + per-step centring of E[3:] + input LN; V4 = untie only (√d, β2 .98 kept). Rule: pass the 20k gates, lowest mean loss over 18–20k, ties <0.01 → fewest contract deviations.

**RELAUNCH_PLAN**
| run | GPU | cols | epochs | schedule | precision | evals | h |
|---|---|---|---|---|---|---|---|
| A0v3_ID256 | 0 | F0 | 4 | S′ 1e-3→cos→1e-4 | TF32/FP32 | 3,4 | 4×3.25+0.84=13.8 |
| A1v3_RICH256 | 1 | F4 | 4 | same | same | 3,4 | 4×3.33+0.88=14.2 |
Launch ≤18:45Z → train end ≈09:00Z → post 2 h ≈11:00Z; deadline 12:20Z; slack ≈1.3 h. Latest 4-epoch launch 19:45Z; later → systems cap 3 epochs (eval 3 only, 10.4 h), declared now. (c) expected effect [inference]: fix recovers the healthy regime, +0.06–0.08 over v2 A0 to ≈0.15–0.17 at epoch 1 and ≈0.17–0.19 best; margin over GRU 0.1739 −0.01…+0.02 — plausible, not guaranteed.

**CURRENT_RUNS**: Stop both now (epoch 5/6, floor LR, losses flat at 8.93/7.29, cohorts declining; remaining ≈5 GPU-h are needed for v3). Keep every v2 artifact (checkpoints incl. epoch-3 best, metrics.jsonl, epoch_records, rank npz 3–4, diag JSON, this brief) as the v2-failure evidence; label v2 INCOMPLETE-by-delegate; never compare v2 with v3 arms.

**CLAIMS_IMPACT**: v3 is a disclosed post-failure recipe change motivated by training diagnostics and literature; v2 VS numbers at epochs 3–4 were seen (state it). Deviations from 03 §3: untied head, no √d, input LN, β2, peak lr, 4 epochs. Single seed, fixed budget, TF32-trained/FP32-evaluated without a fresh parity twin, user-bootstrap CI only; "SASRec > GRU" only if the CI excludes 0; SASRec-vs-GRU stays non-attributional (architecture and optimiser differ). TEST untouched.
