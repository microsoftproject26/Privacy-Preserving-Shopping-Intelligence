# Owner-supplied proposal (verbatim), 2026-09-26 ~00:05 Cairo / 2026-09-25T21:05Z

The owner said:
«خلي fable يراجع الكلام ده ولو كويس ومش هيئاثر بالسلب خليه يخطط ليه ويعمله»

(Translation: "Have Fable review this. If it is good and will not have a negative effect, have it plan it
and do it.") The pasted proposal follows, unchanged:

---

أيوه، **GPU1 ماينفعش تفضل فاضية ~4 ساعات**. وأفضل استخدام لها دلوقتي **مش تدريب architecture جديدة**؛ أعلى عائد وأقل مخاطرة إننا نبدأ نحسب الـfinal systems اللي لا تعتمد على seed 2027 من دلوقتي.

خصوصًا إن عندنا الآن موديلين مكتملين وقويين: A1 cooldown وصلت `0.186298` macro MRR@20، وA0 cooldown وصلت `0.178585` على full VS.

## أعمل إيه على GPU1 الآن؟

1. **S2 Ensemble أولًا — أهم حاجة**: A0_ID256_s2026_cool5 epoch 5 + A1_RICH256_s2026_cool5 epoch 5، score = 0.5*log_softmax(A0) + 0.5*log_softmax(A1)، على full VALIDATION_SELECT بالكامل، strict FP32، full catalogue K=158,486.
2. **S4a checkpoint/state averaging**: uniform state_dict average of A1_s2026 epochs 3, 4, 5، وبعدها full-VS evaluation.
3. **Disagreement / oracle-headroom analysis** من ranks بتاعة A0 e5 وA1 e5: both hit@20، A1-only، A0-only، neither، union/oracle Recall@20، per-user macro differences.
4. **Paired bootstrap** لو S2 أو S4a أحسن من A1 e5: candidate − A1_e5، macro MRR@20، user-level paired bootstrap، 1000 resamples.

المتوقع: S2 ~45–60 min، S4a ~25–35 min، analysis + bootstrap ~10–20 min.

لو فضل وقت: final-stage evaluator rehearsal / throughput benchmark على VS فقط (S1/S3 code path بالـmodels الحالية كـmechanics test)، بدون VC وبدون TEST.

**ما نبدأش:** new d384 model، new architecture، seed 2028 full run، D3 feature training، feature-selection screen، normalization experiments. الـfeature study تستنى `FINAL_STAGE_DONE`.

**Safety guards:** measure s2027 dps for 3 min before GPU1؛ بعد 5 min من التشغيل measure again؛ يكمل طالما GPU0 throughput drop ≤ 10%، free disk ≥ 10 GiB، RAM RSS < ~58 GiB، no TEST/VC read، no TRAIN store build. استخدم VS-only store. لو throughput نزل > 10% يوقف evaluation على GPU1.

Save ranks, all @1/@5/@10/@20 metrics, runtime and hashes in a NEW directory. Do not change any preregistered ensemble weights. Document every result in the living thesis reference.
