import ast
import hashlib
import json
import pathlib

ROOT = pathlib.Path(r"F:\Queens University\Microsoft Project\eda\.local_ai\sasrec_v1")
CODE = ROOT / "code"


def patch(fn, pairs):
    p = CODE / fn
    s = p.read_text(encoding="utf-8")
    for a, b in pairs:
        assert s.count(a) == 1, (fn, a[:70])
        s = s.replace(a, b)
    ast.parse(s)
    p.write_bytes(s.encode())
    print("patched", fn)


patch("train_sasrec.py", [
    ('"""SASRec-CE campaign trainer (owner: Opus). Registry v2: schedule S_prime, fixed epochs.\n',
     '"""SASRec-CE campaign trainer (owner: Opus). Registry v3: v2 schedule mechanics + the Fable\n'
     'deep-review recipe fix (untied output table + per-step common-mode recentering, input LN,\n'
     'no sqrt(d) scale, AdamW betas (0.9, 0.999), peak 1e-3, 4 fixed epochs).\n'),
    ('BETAS, EPS, WEIGHT_DECAY, GRAD_CLIP = (0.9, 0.98), 1e-8, 1e-5, 1.0',
     'BETAS, EPS, WEIGHT_DECAY, GRAD_CLIP = (0.9, 0.999), 1e-8, 1e-5, 1.0   # v3: beta2 0.999 (Fable deep review)'),
    ('FALLBACK_EPOCHS_FP32 = 5', 'FALLBACK_EPOCHS_FP32 = 4'),
    ('''        if int(self.reg.get("version", 1)) != 2:
            raise ConfigError("this trainer implements registry v2")''',
     '''        if int(self.reg.get("version", 1)) != 3:
            raise ConfigError("this trainer implements registry v3")'''),
    ('''        self.opt.step()
        return loss_mean, gn, lr''',
     '''        self.opt.step()
        self.model.recenter_output_()          # v3: loss-neutral common-mode removal (Fable FIX 4)
        return loss_mean, gn, lr'''),
    ('''        self.model = build_model(self.cfg, self.seed, self.store, self.device, a)''',
     '''        self.model = build_model(self.cfg, self.seed, self.store, self.device, a)
        self.stat_rows = torch.as_tensor(np.sort(np.random.default_rng(4242).choice(
            self.model.K, min(8192, self.model.K), replace=False)), device=self.device)'''),
    ('''                                     "gpu_mem_reserved": torch.cuda.memory_reserved(self.device) if self.device.type == "cuda" else None})''',
     '''                                     "gpu_mem_reserved": torch.cuda.memory_reserved(self.device) if self.device.type == "cuda" else None,
                                     "emb": self.model.embedding_stats(self.stat_rows)})'''),
])

# registry v3
p = ROOT / "config" / "run_registry.locked.json"
old = p.read_bytes()
reg = json.loads(old)
reg["version"] = 3
reg["status"] = "LOCKED_V3_AFTER_V2_FAILURE_PENDING_GATE_C_DELTA"
reg["previous_locked_sha256_v2"] = hashlib.sha256(old).hexdigest()
reg["v2_failure"] = {
    "what": "v2 A0/A1 (tied, sqrt(d) input scale, betas 0.9/0.98, peak 5e-4) degenerated: train loss min 7.55 at 12k then rising to ~9.0; grad norm 1.8->6.6; full-VS macro MRR@20 A0 0.0913/0.0901, A1 0.1179/0.1168 at epochs 3/4",
    "root_cause": "common-mode drift of the tied item table under dense-CE AdamW (row norms ~162 uniform across frequency buckets vs GRU 11-13), amplified 16x into a Pre-LN stream without input LN",
    "evidence": ["review/deep/diag_item_embeddings.json", "review/deep/trajectories.json", "review/deep/FABLE_DEEP_REVIEW_RESPONSE.md"],
    "decision": "stopped at epoch 5 by Fable (owner delegate; owner online and requested the deep review); artifacts archived under runs_v2/",
    "stopped_utc": "2026-09-24T17:05Z"}
for m in reg["models"]:
    m["peak_lr"] = 0.001
reg["optimizer"] = {"name": "AdamW (fused)", "betas": [0.9, 0.999], "eps": 1e-8, "weight_decay_matrices": 1e-5, "grad_clip": 1.0}
reg["training"]["betas"] = [0.9, 0.999]
reg["schedule"].update({"planned_epochs": 4, "min_epochs_under_cut": 4, "eval_epochs": "floor epochs only: 3..4",
                        "lr_definition": "u<=2000: 1e-3*u/2000; u<=U: 1e-3; U<u<=2U: cosine 1e-3->1e-4; u>2U: 1e-4"})
reg["cut_rule"]["rule"] = "planned 4 = minimum 4: no cut (2 genuine low-LR epochs kept); a launch after 19:45Z would have required a separately declared 3-epoch plan"
reg["implementation"].update({
    "tying": "UNTIED: separate output table W_out [K,d] (N(0,0.02) from the common generator) + output bias; class j <-> item token j+3",
    "input": "z = LayerNorm_in(E[item] + P[pos] (+ rich side residual)), then input dropout; no sqrt(d) scale",
    "recentering": "after every optimizer step: W_out -= mean_row(W_out); b -= mean(b)  (loss-neutral)",
    "monitoring": "update_agg every 100 updates logs E/W_out common-mode and median row norms (8192-row sample) and bias mean/std",
    "position_convention": reg["implementation"]["position_convention"]})
reg["go_no_go_20k"] = {
    "source": "Fable deep review VERIFICATION_PLAN (pre-registered before any v3 update)",
    "at_20k_updates_both_arms": {"mean_loss_20k_le_mean_loss_10k_minus": 0.02, "mean_grad_norm_20k_le_x_value_at_6k": 1.25,
                                 "E_mean_row_norm_le": 0.5, "median_row_norms_E_and_W_le_x_init": 3.0, "all_finite": True},
    "on_fail": "NO-GO: stop both, report v2+v3 honestly (no third attempt fits the deadline)",
    "descriptive": {"40k_cohort_A0_ge": 0.113, "40k_cohort_A1_ge": 0.123, "80k_A0_cohort_on_track_ge": 0.140,
                    "epoch1_A0_cohort_ge_GRU_epoch1": 0.156, "epoch1_A0_cohort_lt_unlikely_to_beat_GRU": 0.145}}
reg["deviations_from_03"] = ["untied head", "no sqrt(d) input scale", "input LayerNorm", "betas (0.9,0.999)", "peak lr 1e-3", "4 fixed epochs", "per-step output recentering"]
data = (json.dumps(reg, indent=2) + "\n").encode()
p.write_bytes(data)
sha = hashlib.sha256(data).hexdigest()
(ROOT / "config" / "run_registry.locked.sha256").write_bytes((sha + "  run_registry.locked.json\n").encode())
print("registry v3", sha)
