"""P2-WIRE (PREG2 §5.3 WB3 / §5.4 WC1-WC4; company/handoffs/P2-03_04.md §5; REV-P2-03_04 M1 / M2) — WB3 row-sparse
uploads and WC LoRA wired into the P2_W path of v4int.launch_fl (CPU; tiny TinyRec warm worlds of test_fl12_launch and a
tiny real SASREC_D64_B2 world for LoRA; no data, no GPU).

Proves: RECIPES — WB3 (delta_rows, unfrozen) and WC1-WC4 (+ the §5.7 rank steps) admitted with p2_select's structure
rules (every FINAL cell of p2_select.final_cells is now launchable, labels equal), structural misuse refused; the codec
record = fl.sparse_rows.sparse_codec_record(3170), the LoRA targets = fl.peft.LORA_TARGETS; DRY RUNS of WB3 / WC cells
through launch_fl.dry_run (the codec / LoRA record in the ARM_SPEC sha; client LR = c x factor; run ids); CODEC DISPATCH
on the NAME (a P2 spec with the Q8 codec or an unknown codec is refused; WB3 never installs Q8); WB3 END TO END — a tiny
CALIB run DONE_MAIN through the sparse channel, bitwise equal to the dense WB1 run of the same recipe (k_top >= K, SGD_M:
lossless), bytes from the checkpointed ledger (per_visit.upload_bytes = ceil(bytes_up / visits), an int read by
v4int.p2_select.p2_run_adapter), the index-set disclosure recorded; LEDGER BYTES AFTER RESUME — an interrupted + resumed
WB3 smoke has the straight run's server digest and bytes record while the leg-local monitoring restarts; WC END TO END —
a tiny SASRec LoRA CALIB run DONE_MAIN (LoRA init hash != theta_pre digest, T = the LoRA keys, bytes = 4 x 2,048 x r + 8,
the whole base frozen once per client, best / endpoint carry the merged base theta + p2_lora with lineage, p2_run_adapter
accepts it), a resumed LoRA smoke equals the straight one; hook refusals; FIREWALL — the archived pre-wire launch_fl
(code/_archive_p2_wire, f4e56b80) and this one give byte-identical Phase-1 / P2_W WA spec tables and identical tiny
trajectories (skipped when the archive is absent).
NCs (strict xfail): HOOK 2 skipped for WB3 (the run would silently go dense: the bytes check bites); a LoRA run whose
HOOK 1 is skipped (the trained set would be the full model).
"""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

import v4int.launch_fl as F
from fl.numerics import state_digest
from intkit import assert_raises, nc
from test_p2_launch import FAKE_IDENT, PEAK, _dry, _recipe, _run, _tiny, pre  # noqa: F401  (pre = module fixture)

INC = dict(server_opt="fedadam", server_lr=1e-3, client_lr=0.1)
WB3 = dict(INC, items="unfrozen", item_lr_mult=0.1, upload="delta_rows")
WB1 = dict(INC, items="unfrozen", item_lr_mult=0.1)
WC1 = dict(INC, lora_rank=4, lora_lr_factor=10.0)


def _json(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


# ================================================================================================ recipes / records
def test_recipes_codec_record_and_alignment_with_p2_select():
    import fl.peft as PE
    import fl.sparse_rows as SR
    import v4int.p2_select as P
    assert F.P2_WB3_CODEC == SR.sparse_codec_record(SR.K_TOP_REGISTERED) and F.P2_WB3_CODEC["k_top"] == 3170
    assert F.P2_LORA_TARGETS == PE.LORA_TARGETS and F.P2_UPLOADS == ("dense", "delta_rows")
    assert SR.is_sparse_codec(F.P2_WB3_CODEC) and not F.is_q8_codec(F.P2_WB3_CODEC)
    assert not SR.is_sparse_codec(F.Q8_CODEC) and F.is_q8_codec(F.Q8_CODEC)
    a_star = P.p2_recipe(server_opt="fedadam", server_lr=1e-4, client_lr=0.1)
    cells = P.final_cells(a_star)
    assert sorted(s for s, _r in cells.values()) == ["A_STAR", "WB1", "WB2", "WB3", "WC1", "WC2", "WC3", "WC4", "WC5"]
    edges = [e["recipe"] for _l, (_s, r) in cells.items() for e in P.edge_extension("FINAL", r)]
    for r in [r for _s, r in cells.values()] + edges:                 # every FINAL / edge cell is now launchable
        assert F.check_p2_recipe(dict(r)) == r and F.p2_label(r) == P.p2_label(r)
    for rk in F.P2_LORA_RANKS:
        for fac in F.P2_LORA_LR_FACTORS:
            r = F.check_p2_recipe(dict(INC, lora_rank=rk, lora_lr_factor=fac))
            assert r["lora_rank"] == rk and F.p2_client_lr_applied(r) == float(P.client_lr_applied(r))
    bad = [dict(INC, upload="delta_rows"),                                           # WB3 needs unfrozen tables
           dict(INC, items="unfrozen", upload="delta_rows"),                         # ... and its multiplier
           dict(INC, upload="sparse"), dict(INC, lora_rank=4), dict(INC, lora_lr_factor=10.0),
           dict(INC, lora_rank=8, lora_lr_factor=10.0), dict(INC, lora_rank=4, lora_lr_factor=3.0),
           dict(INC, lora_rank=4.5, lora_lr_factor=10.0), dict(INC, lora_rank=True, lora_lr_factor=10.0),
           dict(INC, lora_rank=4, lora_lr_factor=10.0, items="unfrozen", item_lr_mult=0.1),
           dict(WB3, lora_rank=4, lora_lr_factor=10.0), dict(INC, lora_rank=4, lora_lr_factor=10.0, personal_head=True),
           dict(WB3, personal_head=True)]
    for r in bad:
        assert_raises(F.LaunchRefused, F.check_p2_recipe, r)
    # the test_p2_launch refusal list minus its two "not installed" entries (WB3, WC1): still refused
    still = [dict(server_opt="fedadam", server_lr=0.01, client_lr=0.1), dict(server_opt="fedavg", server_lr=1e-4, client_lr=0.01),
             dict(server_opt="fedadam", server_lr=1e-4, client_lr=1.0), dict(server_opt="sgd", server_lr=1e-4, client_lr=0.01),
             dict(server_opt="fedadam", server_lr=1e-4, client_lr=0.01, item_lr_mult=0.1),
             dict(server_opt="fedadam", server_lr=1e-4, client_lr=0.01, items="unfrozen"),
             dict(server_opt="fedadam", server_lr=1e-4, client_lr=0.01, items="unfrozen", item_lr_mult=0.5),
             dict(server_opt="fedadam", server_lr=1e-4, client_lr=0.01, items="unfrozen", item_lr_mult=0.1,
                  personal_head=True),
             dict(server_opt="fedadam", server_lr=1e-4, client_lr=0.01, extra=1),
             dict(server_opt="fedadam", server_lr=True, client_lr=0.01)]
    for r in still:
        assert_raises(F.LaunchRefused, F.check_p2_recipe, r)


def test_wb3_and_wc_dry_runs_and_spec_guards(monkeypatch, tmp_path, pre):
    d, args = _dry(monkeypatch, tmp_path, pre, F.check_p2_recipe(WB3), "--role", "CALIBRATION")
    s = d["arm_spec"]
    assert s["upload_codec"] == F.P2_WB3_CODEC and "frozen_item_tables" not in s and "lora" not in s
    assert s["cfg"]["solver"]["item_lr_mult"] == 0.1 and s["cfg"]["method"] == "FA"
    assert F.default_run_id(args) == "P2_W_FEDADAM_eta0.001_c0.1_UNFROZEN_ilr0.1_DELTA_ROWS_CALIB_s2026"
    dense = F.p2_arm_spec("P2_W", recipe=F.check_p2_recipe(WB1), seed=2026, peak_lr=PEAK, role="CALIBRATION",
                          init_from=pre["ident"])
    assert s["cfg"] == dense["cfg"] and d["arm_spec_sha256"] != F.spec_digest(dense)      # the codec is in the sha
    assert d["fl_config_digest"] == F.fl_run_config(dense, F.default_run_id(args), 64).digest()   # RunConfig blind
    assert F.p2_run_labels(s)["p2_upload"] == "delta_rows" and "p2_upload" not in F.p2_run_labels(dense)
    d, args = _dry(monkeypatch, tmp_path, pre, F.check_p2_recipe(WC1), "--role", "CALIBRATION")
    s = d["arm_spec"]
    assert s["lora"] == {"r": 4, "alpha": 4.0, "targets": ["qkv", "out", "ff1", "ff2"], "seed": 2026}
    assert "frozen_item_tables" not in s and s["upload_codec"] is None
    assert math.isclose(s["cfg"]["solver"]["lr"], 1.0) and s["cfg"]["lr_peak_local"] == s["cfg"]["solver"]["lr"]
    assert F.default_run_id(args) == "P2_W_FEDADAM_eta0.001_c0.1_LORA4x10_CALIB_s2026"
    lab = F.p2_run_labels(s)
    assert lab["p2_lora_rank"] == 4 and lab["p2_lora_lr_factor"] == 10.0 and lab["fl_client_lr"] == 0.1
    assert math.isclose(lab["fl_client_lr_applied"], 1.0) and "fl_frozen_item_tables" not in lab
    m = F.p2_arm_spec("P2_W", recipe=F.check_p2_recipe(dict(WC1, lora_rank=16, lora_lr_factor=1.0)), seed=2027,
                      peak_lr=PEAK, role="MAIN", init_from=pre["ident"])
    assert m["lora"]["seed"] == 2027 and m["lora"]["r"] == 16 and m["cfg"]["solver"]["lr"] == 0.1
    F.check_p2_spec(m)
    wb3 = F.p2_arm_spec("P2_W", recipe=F.check_p2_recipe(WB3), seed=2026, peak_lr=PEAK, role="CALIBRATION",
                        init_from=pre["ident"])
    for t in ({k: v for k, v in m.items() if k != "lora"}, dict(m, lora=dict(m["lora"], r=4)),
              dict(m, lora=dict(m["lora"], seed=2026)), dict(m, frozen_item_tables=list(F.FROZEN_ITEM_KEYS)),
              dict(wb3, upload_codec=None), dict(wb3, upload_codec=dict(F.Q8_CODEC)),
              dict(wb3, upload_codec=dict(F.P2_WB3_CODEC, k_top=100)), dict(wb3, frozen_item_tables=list(F.FROZEN_ITEM_KEYS))):
        assert_raises(F.LaunchRefused, F.check_p2_spec, t)


class _FakeRun:
    pool = None


def test_codec_dispatch_by_name_and_hook_refusals(pre):
    wb3 = F.p2_arm_spec("P2_W", recipe=F.check_p2_recipe(WB3), seed=2026, peak_lr=PEAK, role="CALIBRATION",
                        init_from=pre["ident"])
    for codec in (dict(F.Q8_CODEC), {"name": "OTHER", "k_top": 3}):
        assert_raises(F.LaunchRefused, F.p2_post_run, _FakeRun(), dict(wb3, upload_codec=codec))
        assert_raises(F.LaunchRefused, F.p2_bytes_record, None, None, None, dict(wb3, upload_codec=codec))
    dense = F.p2_arm_spec("P2_W", recipe=F.check_p2_recipe(WB1), seed=2026, peak_lr=PEAK, role="CALIBRATION",
                          init_from=pre["ident"])
    assert_raises(F.LaunchRefused, F.p2_post_run, _FakeRun(), dict(dense, upload_codec=dict(F.P2_WB3_CODEC)))
    assert_raises(F.LaunchRefused, F.p2_post_run, _FakeRun(), dict(wb3, upload_codec=None))   # delta_rows w/o codec
    assert F.p2_post_run(_FakeRun(), dense) == (None, None)
    lora = F.p2_arm_spec("P2_W", recipe=F.check_p2_recipe(WC1), seed=2026, peak_lr=PEAK, role="CALIBRATION",
                         init_from=pre["ident"])
    assert_raises(F.LaunchRefused, F.p2_pre_server, {k: v for k, v in lora.items() if k != "lora"}, None, [], "x")
    assert_raises(F.LaunchRefused, F.p2_pre_server, dict(dense, lora=lora["lora"]), None, [], "x")
    assert_raises(F.LaunchRefused, F.p2_pre_server, dict(lora, frozen_item_tables=list(F.FROZEN_ITEM_KEYS)), None, [],
                  "x")
    assert F.p2_pre_server(dense, None, [], "abc") == (None, "abc")


# ================================================================================================ WB3 end to end
def _wb3_bytes_ok(summary) -> None:
    """The WB3 bytes block (assertions only, so the negative control fails on AssertionError)."""
    b = summary["bytes"]
    pv, fl = b.get("per_visit", {}), b.get("fleet", {})
    assert pv.get("upload_codec") == "SPARSE_ROWS_WB3" and pv.get("k_top") == 3170
    assert b.get("source") == "checkpointed ledger + counters" and fl.get("visits", 0) > 0
    assert isinstance(pv.get("upload_bytes"), int) and pv["upload_bytes"] == -(-fl["bytes_up"] // fl["visits"])
    assert pv.get("upload_bytes_mean") == fl["bytes_up"] / fl["visits"] and isinstance(pv.get("shared_numel"), int)
    assert b.get("privacy_disclosure") == F.P2_WB3_DISCLOSURE
    assert str((b.get("sparse_codec_this_leg_monitoring") or {}).get("scope", "")).startswith("THIS PROCESS LEG ONLY")


def test_wb3_tiny_run_end_to_end_lossless_and_selection_adapter(monkeypatch, tmp_path, pre):
    import v4int.p2_select as P
    _tiny(monkeypatch)
    sp = F.arm_spec("P2_W", seed=2026, peak_lr=PEAK, role="CALIBRATION", init_from=pre["ident"],
                    p2_recipe=F.check_p2_recipe(WB3))
    sd = F.arm_spec("P2_W", seed=2026, peak_lr=PEAK, role="CALIBRATION", init_from=pre["ident"],
                    p2_recipe=F.check_p2_recipe(WB1))
    a, b = _run(tmp_path, pre, sp, "wb3"), _run(tmp_path, pre, sd, "wb1")
    assert a["status"] == b["status"] == "DONE_MAIN"
    sa, sb = a["summary"], b["summary"]
    assert sa["server_digest"] == sb["server_digest"] and sa["evals"] == sb["evals"]        # k_top >= K, SGD_M: lossless
    assert sa["ledger"] != sb["ledger"]                                                     # ... but its own bytes
    _wb3_bytes_ok(sa)
    assert "sparse_codec_this_leg_monitoring" not in sb["bytes"] and "upload_codec" not in sb["bytes"]["per_visit"]
    post = sa["p2"]["post_run"]["wb3"]
    assert post["codec"] == "SPARSE_ROWS_WB3" and post["k_top"] == 3170 and post["channel"] == "SparseChannel"
    assert sa["p2"]["pre_server"] is None and sb["p2"]["post_run"] is None
    side = _json(tmp_path / "wb3" / "ARM_SPEC.json")
    ad = P.p2_run_adapter(side["arm_spec"], _json(tmp_path / "wb3" / "RUN_SUMMARY.json"), tmp_path / "wb3")
    assert ad["bytes_per_visit"] == sa["bytes"]["per_visit"]["upload_bytes"] and ad["recipe"]["upload"] == "delta_rows"
    rows = _json(tmp_path / "wb3" / "RESULT_ROWS.json")["rows"]
    assert rows and all(r["p2_upload"] == "delta_rows" and r["p2_recipe_label"] == F.p2_label(sp["p2_recipe"])
                        for r in rows)
    assert set(F.FROZEN_ITEM_KEYS) <= set(sa["update_norm_log"]["trained_keys"])


def test_wb3_ledger_bytes_identical_after_resume(monkeypatch, tmp_path, pre):
    _tiny(monkeypatch, n=60)
    spec = F.arm_spec("P2_W", seed=2026, peak_lr=PEAK, role="MAIN", init_from=pre["ident"],
                      p2_recipe=F.check_p2_recipe(dict(WB3, server_opt="fedavgm", server_lr=0.1, client_lr=0.03)))
    straight = _run(tmp_path, pre, spec, "s", smoke_rounds=8)
    part = _run(tmp_path, pre, spec, "p", smoke_rounds=8, interrupt_at=5)
    assert part["status"] == "INTERRUPTED"
    res = _run(tmp_path, pre, spec, "p", smoke_rounds=8, resume=True)
    assert res["status"] == straight["status"] == "SMOKE_DONE_NOT_EVIDENCE"
    ss, sr = straight["summary"], res["summary"]
    assert sr["server_digest"] == ss["server_digest"]
    strip = (lambda b: {k: v for k, v in b.items() if k != "sparse_codec_this_leg_monitoring"})  # noqa: E731
    assert strip(sr["bytes"]) == strip(ss["bytes"])                                      # checkpointed ledger
    _wb3_bytes_ok(sr)
    ms, mr = ss["bytes"]["sparse_codec_this_leg_monitoring"], sr["bytes"]["sparse_codec_this_leg_monitoring"]
    assert 0 < mr["visits"] < ms["visits"] == ss["bytes"]["fleet"]["visits"]            # monitoring = this leg only


# ================================================================================================ WC (LoRA) end to end
def sasrec_world(n=8, K=64, seed=2026):
    """A tiny real SASREC_D64_B2 world (synthetic clients), shaped like test_fl12_launch.untied_world."""
    from fl.client import ClientData, valid_rows
    from fl.families import synthetic_examples_v2
    from v4models import build, load_schema
    sch = load_schema()
    srv = build("SASREC", sch, seed, K=K, variant="SASREC_D64_B2").adapter
    wk = build("SASREC", sch, seed, K=K, variant="SASREC_D64_B2").adapter
    clients = {f"user-{i:03d}": ClientData(f"user-{i:03d}",
                                           synthetic_examples_v2(3 + i % 5, K, torch.Generator().manual_seed(100 + i)))
               for i in range(n)}
    nv = {k: int(valid_rows(c.examples).numel()) for k, c in clients.items()}

    def evaluate(mark):
        with torch.no_grad():
            v = float(abs(sum(p.double().sum() for p in srv.module.parameters())) % 1.0)
        rows = [{"aggregation": agg, "target_denominator": den, "metric": met, "k": k,
                 "value": v if (agg, den, met) == ("macro", "E2E", "mrr") else v / 3, "value_status": "OK",
                 "split": "VALIDATION", "eval_manifest_id": "CALIB_VALIDATION_5_100_32768", "method": "C",
                 "training_paradigm": "CENTRAL", "row_status": "CENTRAL_CONTEST", "run_label": "PRODUCTION"}
                for agg in ("micro", "macro") for den in ("E2E", "RANKABLE") for met, k in (("mrr", 20), ("hr", 10))]
        return v, rows, np.arange(5), {"E2E": {"n": 5}}
    return dict(server_adapter=srv, init_sha256=state_digest(srv.broadcast_state()), workers=[wk],
                keys=sorted(clients), n_valid_by_key=nv, get_client=clients.__getitem__,
                n_decisions=sum(nv.values()), manifest_hash="tiny-sasrec", evaluate=evaluate)


def _srun(tmp_path, spec, name, n=8, **kw):
    return F.execute(spec=spec, run_id=name, run_dir=tmp_path / name, say=lambda *_: None, **sasrec_world(n=n), **kw)


def test_wc_lora_tiny_run_end_to_end_payload_and_resume(monkeypatch, tmp_path):
    import fl.peft as PE
    import v4int.p2_select as P
    from fl.checkpoint import load_checkpoint
    _tiny(monkeypatch, n=8)
    w0 = sasrec_world()
    ident = dict(FAKE_IDENT, state_digest=w0["init_sha256"])
    spec = F.arm_spec("P2_W", seed=2026, peak_lr=PEAK, role="CALIBRATION", init_from=ident,
                      p2_recipe=F.check_p2_recipe(WC1))
    out = _srun(tmp_path, spec, "wc1")
    assert out["status"] == "DONE_MAIN"
    s = out["summary"]
    pre_rec = s["p2"]["pre_server"]
    assert pre_rec["lora"]["r"] == 4 and pre_rec["theta_pre_digest"] == w0["init_sha256"]
    assert pre_rec["lora_init_sha256"] != w0["init_sha256"]
    keys = pre_rec["lora"]["lora_keys"]
    assert sorted(s["update_norm_log"]["trained_keys"]) == sorted(keys) and len(keys) == 2 * 4 * 2   # A, B x 4 x 2
    assert s["update_norm_log"]["trained_numel"] == 2048 * 4
    pv = s["bytes"]["per_visit"]
    assert pv["shared_numel"] == 2048 * 4 and pv["upload_bytes"] == 4 * 2048 * 4 + 8
    base_bytes = w0["server_adapter"].manifest.shared_bytes
    assert pv["frozen_bytes_once"] == base_bytes and "lora_frozen_note" in s["bytes"]
    assert s["p2"]["post_run"]["lora"]["shared_numel"] == 2048 * 4 and "frozen_item_tables" not in s
    ck = tmp_path / "wc1" / "ckpt" / "wc1"
    for name in ("best.pt", "endpoint_6.0.pt"):
        ep = load_checkpoint(ck / name)
        assert sorted(ep["theta"]) == sorted(w0["server_adapter"].broadcast_state())   # the merged base architecture
        rec = ep["p2_lora"]
        assert rec["format"] == PE.P2_LORA_FORMAT and rec["meta"]["r"] == 4 and rec["merged_theta_digest"] == \
            state_digest(ep["theta"])
        assert rec["lineage"] == {"theta_pre_digest": w0["init_sha256"], "lora_init_sha256": pre_rec["lora_init_sha256"]}
    side = _json(tmp_path / "wc1" / "ARM_SPEC.json")
    ad = P.p2_run_adapter(side["arm_spec"], _json(tmp_path / "wc1" / "RUN_SUMMARY.json"), tmp_path / "wc1")
    assert ad["trained_params"] == [2048 * 4, 0] and ad["bytes_per_visit"] == 4 * 2048 * 4 + 8
    rows = _json(tmp_path / "wc1" / "RESULT_ROWS.json")["rows"]
    assert rows and all(r["p2_lora_rank"] == 4 and r["method_code"] == "FA" for r in rows)
    # resume: a fresh world per leg (run() rebuilds the adapters every leg), interrupt at 3 of 6 smoke rounds
    _tiny(monkeypatch, n=60)
    w60 = sasrec_world(n=60)
    ident = dict(FAKE_IDENT, state_digest=w60["init_sha256"])
    ms = F.arm_spec("P2_W", seed=2026, peak_lr=PEAK, role="MAIN", init_from=ident,
                    p2_recipe=F.check_p2_recipe(dict(WC1, lora_rank=16, lora_lr_factor=1.0)))
    straight = _srun(tmp_path, ms, "ls", n=60, smoke_rounds=6)
    part = _srun(tmp_path, ms, "lp", n=60, smoke_rounds=6, interrupt_at=3)
    assert part["status"] == "INTERRUPTED"
    res = _srun(tmp_path, ms, "lp", n=60, smoke_rounds=6, resume=True)
    assert res["status"] == straight["status"] == "SMOKE_DONE_NOT_EVIDENCE"
    assert res["summary"]["server_digest"] == straight["summary"]["server_digest"]
    assert res["summary"]["bytes"] == straight["summary"]["bytes"]


# ================================================================================================ firewall (archive)
ARCHIVE = F.CODE / "_archive_p2_wire" / "v4int" / "launch_fl.py"


def _old_module():
    if not ARCHIVE.is_file():
        pytest.skip("code/_archive_p2_wire is not present on this host")
    if hashlib.sha256(ARCHIVE.read_bytes()).hexdigest() != F_ARCHIVE_SHA:
        pytest.skip("the archive is not the pre-wire launch_fl")
    sp = importlib.util.spec_from_file_location("launch_fl_prewire", ARCHIVE)
    m = importlib.util.module_from_spec(sp)
    saved = list(sys.path)
    try:
        sp.loader.exec_module(m)
    finally:
        sys.path[:] = saved
    m.CODE, m.V4 = F.CODE, F.V4
    return m


F_ARCHIVE_SHA = "f4e56b8008e7f4a66d5a6a07cbc0b9d978c28d90d9f9596a3b166f5b01475734"
TK = ({}, dict(optimizer="fedadam", server_lr=0.01, lr_shape="fl_const", client_opt="sgd_m", client_lr=0.1, local_passes=1),
      dict(optimizer="fedavgm", lr_mult=2.0), dict(optimizer="fedadam", server_lr=0.001, client_opt="sgd_m",
                                                   client_lr=0.1, lr_shape="fl_const", local_passes=4))


def _table(M):
    out = {}
    for arm in M.ALL_ARMS:
        ca, v = M.base_arm(arm), M.warm_variant(arm)
        for i, tk in enumerate(TK):
            eff, prule, _ = M.poisson_tuning(arm, M.tuning_config(**tk))
            for wf in ((None, M.WARM_FACTORS[v][1]) if v else (None,)):
                for sf in (None, 0.1, 0.25):
                    for role in ("MAIN", "CALIBRATION"):
                        kw = ({"clip_norm": 0.2843} if ca in ("FA_1024", "DP") else {}) | ({"z": 0.5642} if ca == "DP"
                                                                                          else {})
                        key = f"{arm}|{i}|{wf}|{sf}|{role}"
                        try:
                            s = M.arm_spec(arm, seed=2026, peak_lr=PEAK, fused=True, poisson_local_rule=prule, role=role,
                                           init_from=FAKE_IDENT if v else None, warm_lr_factor=wf,
                                           warm_server_lr_factor=sf, **kw, **M.tuning_kwargs(eff))
                            out[key] = [M.spec_digest(s), M.fl_run_config(s, "x", 64).digest()]
                        except M.LaunchRefused as e:
                            out[key] = str(e)
    for r in list(M.p2_wa_recipes()) + [M.check_p2_recipe(dict(INC, items="unfrozen", item_lr_mult=0.01)),
                                        M.check_p2_recipe(dict(INC, personal_head=True))]:
        for role in ("MAIN", "CALIBRATION"):
            s = M.p2_arm_spec("P2_W", recipe=r, seed=2026, peak_lr=PEAK, role=role, init_from=FAKE_IDENT)
            out[f"P2|{M.p2_label(r)}|{role}"] = [M.spec_digest(s), M.fl_run_config(s, "x", 64).digest(),
                                                 json.dumps(M.p2_run_labels(s), sort_keys=True)]
    return out


def test_firewall_prewire_archive_tables_and_trajectories_identical(monkeypatch, tmp_path, pre):
    from test_fl05_launch import fl05_world
    from test_fl12_launch import warm_world
    old = _old_module()
    assert _table(old) == _table(F)
    assert all(F.semantic_arm(a) == F.base_arm(a) for a in F.ALL_ARMS + F.P2_ARMS)

    def tiny(M, arm, **kw):
        s = M.arm_spec(arm, seed=2026, peak_lr=PEAK, **kw)
        s["cfg"]["endpoint_rounds"] = 24
        s["max_rounds"] = min(24, s["max_rounds"])
        s["plan"]["group_size"] = 4
        if s["cfg"]["dp"] is not None:
            s["cfg"]["dp"]["denominator"] = 4
        return s

    def both(name, spec_fn, world_fn, legs=({},)):
        got = []
        for M in (old, F):
            rd = tmp_path / ("old" if M is old else "new") / name
            outs = []
            for kw in legs:
                random.seed(0)
                np.random.seed(0)
                torch.manual_seed(0)
                o = M.execute(spec=copy.deepcopy(spec_fn(M)), run_id=name, run_dir=rd, say=lambda *_: None,
                              **world_fn(), **kw)
                outs.append(o["status"])
            summ = o["summary"]
            got.append((outs, summ["server_digest"], summ["ledger"], summ["evals"], summ["bytes"],
                        sorted(summ), [(p.name, p.read_bytes()) for p in sorted(rd.glob("*.jsonl"))],
                        _json(rd / "RESULT_ROWS.json")["rows"] if (rd / "RESULT_ROWS.json").is_file() else None))
        assert got[0] == got[1], name

    both("FA_Q8_resume", lambda M: tiny(M, "FA_Q8"), fl05_world, ({"interrupt_at": 5}, {"resume": True}))
    both("DP_WARM_FROZEN", lambda M: tiny(M, "DP_WARM_FROZEN", init_from=pre["ident"], clip_norm=0.2843, z=0.5642),
         lambda: warm_world(pre))
    both("S_CAL", lambda M: tiny(M, "S_CAL"), fl05_world)
    for M in (old, F):
        monkeypatch.setitem(M.CALIB, "n_clients", 12)
        monkeypatch.setattr(M, "MAIN_N_CLIENTS", 12)
        monkeypatch.setitem(M.KNOBS, "group", 4)
    inc = F.p2_wa_recipes()[-1]
    both("P2_W_incumbent", lambda M: M.arm_spec("P2_W", seed=2026, peak_lr=PEAK, role="CALIBRATION",
                                                init_from=pre["ident"], p2_recipe=inc), lambda: warm_world(pre))


# ================================================================================================ negative controls
@nc("HOOK 2 skipped for a WB3 spec: the run would silently go dense; the WB3 bytes check must bite")
def test_nc_wb3_without_hook2(monkeypatch, tmp_path, pre):
    _tiny(monkeypatch)
    monkeypatch.setattr(F, "p2_post_run", lambda run, spec: (None, None))                  # MUTANT
    monkeypatch.setattr(F, "p2_bytes_record", lambda m, c, l, spec, frozen_keys=(): F.bytes_record(m, c, l))   # MUTANT
    sp = F.arm_spec("P2_W", seed=2026, peak_lr=PEAK, role="CALIBRATION", init_from=pre["ident"],
                    p2_recipe=F.check_p2_recipe(WB3))
    o = _run(tmp_path, pre, sp, "nc")
    _wb3_bytes_ok(o["summary"])


@nc("a LoRA spec whose HOOK 1 is skipped would train the full model; the trained-set check must bite")
def test_nc_lora_without_hook1(monkeypatch, tmp_path):
    _tiny(monkeypatch, n=8)
    monkeypatch.setattr(F, "p2_pre_server", lambda spec, a, w, init: (None, init))       # MUTANT
    monkeypatch.setattr(F, "p2_post_run", lambda run, spec: (None, None))                  # MUTANT
    w0 = sasrec_world()
    spec = F.arm_spec("P2_W", seed=2026, peak_lr=PEAK, role="CALIBRATION",
                      init_from=dict(FAKE_IDENT, state_digest=w0["init_sha256"]), p2_recipe=F.check_p2_recipe(WC1))
    o = _srun(tmp_path, spec, "nc", smoke_rounds=1)
    assert o["summary"]["bytes"]["per_visit"]["shared_numel"] == 2048 * 4
