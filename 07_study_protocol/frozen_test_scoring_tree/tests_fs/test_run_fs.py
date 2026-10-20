"""v4fs.run_fs: preflight refusals, labels, retention and the --dry-run pipeline (CPU, synthetic providers).

  * refusals before anything heavy: a TEST manifest in the FS registry (by role or listed), a seed outside
    {2026, 2027}, an unregistered schema, a tampered registry, a reused run directory, the disk floor, and (non-dry)
    no GPU id, an inherited CUDA_VISIBLE_DEVICES mismatch, a busy GPU, an nvidia-smi failure, the memory guard;
  * labels: write_rows refuses an unlabelled payload, an unlabelled row and a row carrying a selection rule or
    eligibility flag; a production run refuses a trainer that cannot label its own rows;
  * dry run, L3 GRU: the CentralTrainer trains the derived schema; the CONTROLLED_BUDGET statistic is rebuilt from
    the 6.0 ranks; the importance (14 groups x 3 seeds) runs on the endpoint_6.0 weights and its baseline equals the
    trainer's value; every weight file is deleted with its sha256 logged; every row is FEATURE_SEARCH_SCREEN;
  * dry run, L0 SASRec: no importance (auto), the endpoint statistics are saved, no weight is left.
Negative control (strict xfail): a writer without the label check accepts an unlabelled row.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from fskit import BIG_DISK, assert_raises, disk, nc, smi_runner
from v4fs import run_fs as R
from v4fs.importance import load_stats


def _args(tmp, *extra, schema="L3S_NEAR_DUPLICATE_SESSION", family="GRU", seed=2026, dry=True):
    a = ["--family", family, "--schema", schema, "--seed", str(seed), "--out", str(tmp / "run"),
         "--fs-registry", str(tmp / "no_registry.json"), "--log-every", "5"]
    return R.parse_args(a + (["--dry-run"] if dry else []) + list(extra))


def _registry(tmp, train="FS_TRAIN_5_100_32768", ev="FS_VALIDATION_5_100_32768", extra_ids=()):
    obj = {"data_version": "FS_SCREEN_5_100_v1", "roles": {R.ROLE: {"train": train, "eval": ev}},
           "manifests": {m: {"path": f"data_fs/{m}"} for m in (train, ev) + tuple(extra_ids)}}
    obj["self_sha256"] = hashlib.sha256(R._canon(obj)).hexdigest()
    p = tmp / "REGISTRY_FS.json"
    p.write_text(json.dumps(obj), encoding="utf-8")
    return p


def test_preflight_refusals(tmp_path):
    ok = R.preflight(_args(tmp_path), disk_usage_fn=BIG_DISK)
    assert ok["run_label"] == R.RUN_LABEL and ok["importance"] and ok["schema"]["self_sha256"]
    assert_raises(R.FSRefused, R.preflight, _args(tmp_path, seed=2028), disk_usage_fn=BIG_DISK)
    assert_raises(R.FSRefused, R.preflight, _args(tmp_path, schema="L9_ID_ONLY_ANCHOR"), disk_usage_fn=BIG_DISK)
    assert_raises(R.FSRefused, R.preflight, _args(tmp_path), disk_usage_fn=disk(3.0))
    reg = _registry(tmp_path)
    a = _args(tmp_path, "--fs-registry", str(reg))
    assert R.preflight(a, disk_usage_fn=BIG_DISK)["manifests"] == {"train": "FS_TRAIN_5_100_32768",
                                                                    "eval": "FS_VALIDATION_5_100_32768"}
    for kw in ({"ev": "M08_RETRO_TEST"}, {"train": "FS_TEST_TRAIN"}, {"extra_ids": ("M07_TEST_WARM",)}):
        bad = _registry(tmp_path, **kw)
        assert_raises(R.FSRefused, R.preflight, _args(tmp_path, "--fs-registry", str(bad)), disk_usage_fn=BIG_DISK)
    obj = json.loads(reg.read_text(encoding="utf-8"))
    obj["roles"][R.ROLE]["eval"] = "FS_VALIDATION_OTHER"                          # tampered, stale self-hash
    reg.write_text(json.dumps(obj), encoding="utf-8")
    assert_raises(R.FSRefused, R.preflight, _args(tmp_path, "--fs-registry", str(reg)), disk_usage_fn=BIG_DISK)
    run = tmp_path / "run"
    run.mkdir()
    (run / R.SUMMARY_FILE).write_text("{}", encoding="utf-8")
    assert_raises(R.FSRefused, R.preflight, _args(tmp_path), disk_usage_fn=BIG_DISK)
    assert_raises(R.FSRefused, R.refuse_test, "M09_TEST_COLD")
    R.refuse_test("FS_VALIDATION_5_100_32768")


def test_preflight_gpu_checks(tmp_path):
    reg = _registry(tmp_path)

    def pre(runner, env=None, gpu="1"):
        extra = ["--fs-registry", str(reg)] + (["--gpu", gpu] if gpu is not None else [])
        return R.preflight(_args(tmp_path, *extra, dry=False), runner=runner, environ=env or {},
                           disk_usage_fn=BIG_DISK)
    rec = pre(smi_runner())
    assert rec["gpu_check"]["gpu_ready"] == "ok"
    assert_raises(R.FSRefused, pre, smi_runner(), gpu=None)
    assert_raises(R.FSRefused, pre, smi_runner(), env={"CUDA_VISIBLE_DEVICES": "0"})
    assert_raises(R.FSRefused, pre, smi_runner(apps="00000000:02:00.0, 4242\n"))
    assert_raises(R.FSRefused, pre, smi_runner(fail=True))
    assert_raises(R.FSRefused, pre, smi_runner(mem_used=900.0))
    pre(smi_runner(apps="00000000:01:00.0, 4242\n"))                              # a process on GPU 0 only: fine
    env = {}
    R.export_environment(1, False, env)
    assert env == {"CUBLAS_WORKSPACE_CONFIG": ":4096:8", "PYTHONDONTWRITEBYTECODE": "1", "CUDA_VISIBLE_DEVICES": "1"}
    assert_raises(R.FSRefused, R.export_environment, 1, False, {"CUBLAS_WORKSPACE_CONFIG": ":16:8"})


def test_write_rows_refuses_unlabelled_output(tmp_path):
    good = R.label_row({"metric": "mrr", "selection_rule": "CONTROLLED_BUDGET", "c_matched_eligible": True})
    assert good["run_label"] == R.RUN_LABEL and good["selection_rule"] is None
    assert good["fs_selection_rule"] == "CONTROLLED_BUDGET" and good["c_matched_eligible"] is False
    R.write_rows(tmp_path / "ok.json", {"run_label": R.RUN_LABEL, "rows": [good]})
    assert_raises(R.FSRefused, R.write_rows, tmp_path / "a.json", {"run_label": "PRODUCTION", "rows": [good]})
    assert_raises(R.FSRefused, R.write_rows, tmp_path / "b.json", {"run_label": R.RUN_LABEL,
                                                                   "rows": [good, {"metric": "mrr"}]})
    assert_raises(R.FSRefused, R.write_rows, tmp_path / "c.json", {"run_label": R.RUN_LABEL,
                                                                   "rows": [dict(good, selection_rule="PRACTICAL_BEST")]})
    assert_raises(R.FSRefused, R.write_rows, tmp_path / "d.json", {"run_label": R.RUN_LABEL,
                                                                   "rows": [dict(good, downstream_seed_allowed=True)]})
    assert not (tmp_path / "b.json").exists()


def test_production_config_is_the_fs_role_and_labelled_by_the_trainer(tmp_path):
    """TRAIN-01R2 present: the production FS RunConfig (no test_overrides) validates with the locked settings and the
    trainer derives FEATURE_SEARCH_SCREEN for every row. Absent: a production run refuses (unlabelled rows)."""
    from v4train.recipe import RunConfig
    sup = R.trainer_label_support(RunConfig)
    if not sup["labels_rows"]:
        a = _args(tmp_path, "--gpu", "1", dry=False)
        pre = {"manifests": {"train": "FS_TRAIN_5_100_32768", "eval": "FS_VALIDATION_5_100_32768"}}
        assert_raises(R.FSRefused, R.run, a, pre, {}, say=lambda *_: None, disk_usage_fn=BIG_DISK)
        return
    key = {"data_protocol_hash": "D", "feature_contract_hash": "F", "split": "VALIDATION", "label_policy_hash": "L",
           "population_hash": "P", "candidate_order_hash": "C"}
    for seed in R.SEEDS:
        cfg = R.make_config(RunConfig, run_id="FS__x", family="GRU", seed=seed, cuda_visible="1", key=key,
                            expected_init="a" * 64, dry_run=False, manifest_registry=str(tmp_path / "REG.json"),
                            data_version="FS_SCREEN_5_100_v1", schema_id="L3_NEAR_DUPLICATE",
                            schema_registry=str(R.SCHEMA_REGISTRY), log_every=50)
        cfg.validate()
        assert cfg.run_label == R.RUN_LABEL and cfg.role == R.ROLE and not cfg.test_overrides
        assert (cfg.effective_batch, cfg.peak_lr, cfg.exposure_point, cfg.determinism_mode, cfg.phase) ==             (256, 1e-3, "end", "STRICT", "MAIN")
    assert_raises(Exception, R.make_config(RunConfig, run_id="FS__x", family="GRU", seed=2028, cuda_visible="1",
                                           key=key, expected_init=None, dry_run=False,
                                           manifest_registry="r", data_version="v", schema_id="L3_NEAR_DUPLICATE",
                                           schema_registry=str(R.SCHEMA_REGISTRY), log_every=50).validate)


def _dry(tmp_path, **kw):
    a = _args(tmp_path, **kw)
    pre = R.preflight(a, disk_usage_fn=BIG_DISK)
    return a, R.run(a, pre, {"note": "in-process test"}, say=lambda *_: None, disk_usage_fn=BIG_DISK)


def test_dry_run_l3_gru_end_to_end(tmp_path):
    a, s = _dry(tmp_path)
    run = Path(a.out)
    assert s["run_label"] == R.RUN_LABEL and s["weight_files_left"] == [] and not list(run.rglob("*.pt"))
    assert s["statistic"] == "CONTROLLED_BUDGET" and s["controlled_budget"]["mark"] == 6.0
    assert s["importance_baseline_equals_controlled_budget"] and len(s["importance"]) == 15      # L3S: G1 ... G13
    kinds = {r["kind"]: r for r in s["retention"] if "sha256" in r}
    assert "endpoint_6.0" in kinds and kinds["endpoint_6.0"]["stage"] == "final"
    assert s["endpoint_checkpoint"]["loaded_state_digest"] == s["endpoint_checkpoint"]["state_digest"]
    log = [json.loads(x) for x in (run / "RETENTION_LOG.jsonl").read_text(encoding="utf-8").splitlines()]
    assert {x["sha256"] for x in log} >= {kinds["endpoint_6.0"]["sha256"]} and all(len(x["sha256"]) == 64 for x in log)
    res = json.loads((run / R.RESULT_FILE).read_text(encoding="utf-8"))
    all_rows = res["rows"] + res["importance_baseline"] + res["importance"]
    assert all_rows and all(r["run_label"] == R.RUN_LABEL and r["selection_rule"] is None for r in all_rows)
    assert all(not r["c_matched_eligible"] and not r["downstream_seed_allowed"] for r in all_rows)
    assert {r.get("fs_selection_rule") for r in res["rows"]} == {"PRACTICAL_BEST", "CONTROLLED_BUDGET"}
    assert res["importance_baseline"][0]["value"] == s["controlled_budget"]["value"]
    st, meta = load_stats(run / R.USERSTATS_FILE)
    assert meta["statistic"] == "CONTROLLED_BUDGET" and len(st) == 2 + 15 * 3
    assert np.array_equal(st["endpoint"].S, st["base"].S) and np.array_equal(st["endpoint"].N, st["base"].N)
    launch = json.loads((run / "FS_LAUNCH.json").read_text(encoding="utf-8"))
    assert launch["schema"]["schema_id"] == "L3S_NEAR_DUPLICATE_SESSION" and launch["shared_numel"] > 0
    imp = {r["group"]: r for r in res["importance"]}
    assert imp["G1"]["descriptive_only"] and not imp["G1"]["droppable"] and imp["G13"]["droppable"]
    assert launch["history_policy"] == "USER_HISTORY_L50"
    assert_raises(R.FSRefused, R.preflight, a, disk_usage_fn=BIG_DISK)             # never reused


def test_dry_run_l0_sasrec_without_importance(tmp_path):
    a, s = _dry(tmp_path, schema="L0S_CLEAN_V2_SESSION", family="SASREC", seed=2027)      # the FS reference
    run = Path(a.out)
    assert s["importance"] is None and s["weight_files_left"] == [] and not list(run.rglob("*.pt"))
    assert {r["kind"] for r in s["retention"] if "sha256" in r} >= {"endpoint_6.0", "best"}
    st, _ = load_stats(run / R.USERSTATS_FILE)
    assert sorted(st) == ["endpoint"]
    res = json.loads((run / R.RESULT_FILE).read_text(encoding="utf-8"))
    assert res["importance"] == [] and all(r["run_label"] == R.RUN_LABEL for r in res["rows"])


def test_session_key_contract_is_checked():
    """SCHEMA-01R1: a provider without DATA-05's keys (or breaking their contract) is refused for an S schema; the L3
    dry run (no extension kept) still works with the synthetic keys present in the batch."""
    import torch
    from fskit import L3S, eval_provider
    good = next(iter(eval_provider(session=True).batches(1024)))
    R.check_session_offsets(good, "eval", schema=L3S)
    for k in ("session_offset_tokens", "session_mode_main_category_tokens", "session_mode_share",
              "current_matches_session_mode_available"):
        bad = {x: v for x, v in good.items() if x != k}
        assert_raises(R.FSRefused, R.check_session_offsets, bad, "eval", schema=L3S)
    bad = dict(good, session_offset_tokens=good["session_offset_tokens"] + 1)          # PAD no longer 0
    assert_raises(R.FSRefused, R.check_session_offsets, bad, "eval", schema=L3S)
    mk = good["current_matches_session_mode_available"].clone()
    mk[0] = 0.5
    assert_raises(R.FSRefused, R.check_session_offsets, dict(good, current_matches_session_mode_available=mk),
                  "eval", schema=L3S)
    tok = good["session_mode_main_category_tokens"].clone()
    tok[good["attention_mask"].bool()] = 17                                                # outside the vocab
    assert_raises(R.FSRefused, R.check_session_offsets, dict(good, session_mode_main_category_tokens=tok),
                  "eval", schema=L3S)
    assert isinstance(good["session_offset_tokens"], torch.Tensor)


@nc("a writer without the label check lets an unlabelled row through")
def test_nc_unlabelled_writer(tmp_path):
    def naive_write(path, payload):
        Path(path).write_text(json.dumps(payload), encoding="utf-8")
        return True
    payload = {"run_label": R.RUN_LABEL, "rows": [{"metric": "mrr"}]}
    wrote = True
    try:
        naive_write(tmp_path / "x.json", payload)
    except R.FSRefused:
        wrote = False
    assert not wrote, "an unlabelled row was written"
