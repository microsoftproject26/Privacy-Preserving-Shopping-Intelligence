#!/bin/bash
# G_lock.sh <freeze_sha256> --i-confirm — review §1 Phase G / MASTER_PLAN §10: builds the CANDIDATE
# company/gates/ALL_METHODS_LOCK.json on the server + the register entry-5 JSON. FINAL r7 form (OPS-ENDGAME-2), NOT RUN.
# Reads hashes / paths / statuses only (rows and rank files are HASHED, never parsed). The candidate becomes the lock only by
# the LOCAL steps of README §3 G (O_EXCL copy, test_exposure_register.py append = entry 5, DL row, sha-checked upload).
# The lock carries the whole TEST plan (every locked checkpoint + sha, device commands, baselines) that H_test.sh executes.
# Predecessors: $R/ANA/out_val_<tag>/RETENTION_RESULT.json + ok/F_<tag>.ok + the registered freeze.
# Output: $R/ANA/lock_<tag>/ALL_METHODS_LOCK.candidate.json, REGISTER_ENTRY_5.json, $Q/endgame/FL_CKPT_RECORD_G.json, ok/G_<tag>.ok.
LOG=/home/jovyan/ahmedabdelhamed01/Privacy-Preserving-Shopping-Intelligence/.local_ai/ppsi_v4/runs_ds3/queues/endgame_G.log
source /home/jovyan/ahmedabdelhamed01/Privacy-Preserving-Shopping-Intelligence/.local_ai/ppsi_v4/runs_ds3/queues/_common_code_ds3r7.sh
source $Q/endgame/_endgame_lib.sh
confirm "$2"
source $Q/endgame/_endgame_ana_args.sh
source $Q/endgame/_endgame_freeze_check.sh
need $ANA_OUT_VAL/RETENTION_RESULT.json; need $OK/F_$ANA_TAG.ok; once $OK/G_$ANA_TAG.ok
[ ! -e $LOCK ] || eg_refuse "$LOCK already exists on the server (the lock is O_EXCL)"
freeze_check "$1"
LD=$ANA/lock_$ANA_TAG; once $LD; mkdir -p $LD
step "PHASE_START G lock freeze=${1:0:16}"
# every FL MAIN run of the lock: best + endpoint, sealed against the runs' own events.jsonl (README Q1, RESOLVED in the lib)
FLDIRS="FA_s2026 FA_s2027 N33_FP_s2026 N33_FP_s2027 N33_PF_s2026 N33_PF_s2027 N33_FA_Q8_s2026 N33_FA_Q8_s2027
 N33_FA_1024_s2026 N33_FA_1024_s2027 N33_DP8_s2026 N33_DP8_s2027 N33_DP1_s2026 FA_WARM_s2026 FA_WARM_s2027
 FA_WARM_FROZEN_s2026 FA_WARM_FROZEN_s2027 FP_WARM_FROZEN_s2026 FP_WARM_FROZEN_s2027 PF_WARM_FROZEN_s2026 PF_WARM_FROZEN_s2027
 FA_Q8_WARM_FROZEN_s2026 FA_Q8_WARM_FROZEN_s2027 FA_1024_WARM_FROZEN_s2026 FA_1024_WARM_FROZEN_s2027 DP8_FROZEN_s2026
 DP8_FROZEN_s2027 DP1_WARM_FROZEN_s2026"
CKS=(); for d in $FLDIRS; do need_done $R/FL_T/$d; k=$(fl_ckdir $R/FL_T/$d); CKS+=($k/best.pt $k/endpoint_6.0.pt); done
FLREC=$EG/FL_CKPT_RECORD_G.json
seal_fl_ckpts $FLREC "${CKS[@]}" >> "$LOG" 2>&1 || { step "LANE_STOP FL checkpoint record REFUSED"; exit 96; }
export V R CR FREEZE CONTRASTS PIN LD FLREC FLDIRS FSHA="$1" BLK="$ANA_BLOCK_VAL" VRES="$ANA_OUT_VAL/RETENTION_RESULT.json" TCA="$EG/B_TRUE_COLD_ADMISSION.tsv"
$PY -B - <<'PY' >> "$LOG" 2>&1 || { step "LANE_STOP lock candidate REFUSED (see $LOG)"; exit 99; }
import hashlib, json, os, re, sys
from pathlib import Path
E = os.environ; V, R, CR, LD = Path(E["V"]), Path(E["R"]), Path(E["CR"]), Path(E["LD"])
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def j(p): return json.loads(Path(p).read_text(encoding="utf-8"))
def refuse(m): print("REFUSED:", m); sys.exit(1)
fz, c, flr = j(E["FREEZE"]), j(E["CONTRASTS"]), j(E["FLREC"])["files"]
blk = Path(E["BLK"]); idx = j(blk / "BLOCK_INDEX.json")
arms = []
for e in idx["entries"]:                                                  # hashes only; nothing parsed
    a = {k: e.get(k) for k in ("arm", "seed", "rule", "role", "comparison_block", "rows", "ranks")}
    for k in ("rows", "ranks", "row_ties"):
        if e.get(k): a[k + "_sha256"] = sha(blk / e[k])
    arms.append(a)
D6, D2 = ("SASREC_D64_B2", "SMALL_SASREC_D64_B2"), ("SASREC_D256_B3", "02B_SASREC")
TAGS = (("PB", "best", "PRACTICAL_BEST"), ("CB", "endpoint_6.0", "CONTROLLED_BUDGET"))
src = {}                                                                  # (arm, seed) -> {tag: checkpoint spec}
def central(arm, seed, run, rid, var):
    ck = j(run / "sealed/CHECKPOINTS.json")
    for tag, f, rule in TAGS:
        d = ck[f]; h = d["sha256"] if isinstance(d, dict) else d
        src.setdefault((arm, seed), {})[tag] = dict(kind="central", ckpt=str(run / f"sealed/checkpoints/{f}.pt"), sha=h,
            variant=var[0], recipe=var[1], rule=rule, ckpt_id=f"{rid}:{f}")
for s in (2026, 2027):
    central("C_BIG", s, V / f"runs_p3/SASREC/02B_SASREC_s{s}", f"02B_SASREC_MAIN_s{s}", D2)
    central("C_SMALL", s, V / f"runs_ds/C_SMALL_s{s}", f"SMALL_SASREC_D64_B2_MAIN_s{s}", D6)
    central("C_FULL", s, R / f"CFULL/C_FULL_02B_SASREC_s{s}", f"C_FULL_02B_SASREC_s{s}", D2)   # s2027 DESCRIPTIVE
pre = V / "runs_pretrain/PRETRAIN_FOR_FL_SMALL_SASREC_D64_B2_s2026"
pinit = j(pre / "PRETRAIN_FOR_FL_INIT.json"); psha = (pinit.get("fl_init") or {}).get("sha256")  # Q10 RESOLVED: the key
#   reeval.check_pre_init_record reads (fl_init.sha256 / .path / .kind endpoint_6.0; reeval.py L535-539)
if not re.fullmatch(r"[0-9a-f]{64}", psha or "") or not psha.startswith("cf08588c"): refuse("PRE fl_init sha256 not cf08588c…")
for tag, _f, rule in TAGS:                                                # ONE theta_pre endpoint under either rule label
    src.setdefault(("PRE", 2026), {})[tag] = dict(kind="central", ckpt=str(pre / "sealed/checkpoints/endpoint_6.0.pt"), sha=psha,
        variant=D6[0], recipe=D6[1], rule=rule, ckpt_id="PRETRAIN_FOR_FL_SMALL_SASREC_D64_B2_s2026:endpoint_6.0")
for dn in E["FLDIRS"].split():
    m = re.fullmatch(r"(?:N33_)?(?P<arm>.+)_s(?P<seed>\d{4})", dn); arm, seed = m.group("arm"), int(m.group("seed"))
    run = R / "FL_T" / dn; ckd = [p for p in (run / "ckpt").iterdir() if p.is_dir()][0]
    for tag, f, rule in TAGS:
        p = str((ckd / f"{f}.pt").resolve()); kind = "pf" if arm in ("PF", "PF_WARM_FROZEN") else "fl"
        spec = dict(kind=kind, ckpt=p, sha=flr[p]["file_sha256"], variant=D6[0], recipe=D6[1], rule=rule,
                    ckpt_id=f"{ckd.name}:{f}", run_dir=str(run))
        if kind == "pf":            # Q11 RESOLVED: launch_fl records each store's sha256 in RUN_SUMMARY (L1631-1637):
            st = run / ("PF_STORE_best.pt" if tag == "PB" else "PF_STORE.pt")      # pf_store_best (PB) / pf_store.record (CB)
            if not st.is_file(): refuse(f"{st} missing")
            sm = j(run / "RUN_SUMMARY.json")                                  # identity keys only (file / sha256)
            want = (sm.get("pf_store_best") or {}) if tag == "PB" else ((sm.get("pf_store") or {}).get("record") or {})
            if want.get("file") != st.name or want.get("sha256") != sha(st):
                refuse(f"{st}: sha256 differs from the run's RUN_SUMMARY store record")
            spec.update(pf_store=str(st), pf_sha=want["sha256"])
        src.setdefault((arm, seed), {})[tag] = spec
reevals = []
def plan(arm, seed, tag, role):
    s = src[(arm, seed)][tag]; base_role = role.replace("_TEST", "")
    name = f"{arm}_s{seed}_{tag}" if role == "MAIN_TEST" else f"{arm}_{base_role}_s{seed}_{tag}"   # Q4: index L359-363
    reevals.append(dict(s, arm_label=arm, seed=seed, role=role, tag=tag, out=str(R / "TEST/REEVAL" / name)))
for (arm, seed), tags in sorted(src.items()):
    for tag in ("PB", "CB"):
        if tag in tags: plan(arm, seed, tag, "MAIN_TEST")
admitted = set()
if Path(E["TCA"]).is_file():
    admitted = {l.split("\t")[0] for l in Path(E["TCA"]).read_text().splitlines() if l.endswith("\tADMITTED")}
PAN = re.compile(r"(?P<arm>.+)_(?P<role>UNSEEN_HOLDOUT|SPARSE_PANEL|HEAVY_PANEL|TRUE_COLD)_s(?P<seed>\d{4})_(?P<tag>PB|CB)")
panels = []
for d in sorted((R / "REEVAL_A40").iterdir()):                          # TEST panels only where VALIDATION panel rows exist
    m = PAN.fullmatch(d.name)
    if not m or not (d / "REEVAL_RECORD.json").is_file(): continue
    arm, seed = m.group("arm"), int(m.group("seed"))
    if (arm, seed) not in src: refuse(f"panel dir {d.name}: no locked source for {arm} s{seed}")
    plan(arm, seed, m.group("tag"), m.group("role") + "_TEST"); panels.append(d.name)
# device arms on MAIN_TEST with the LOCKED selections / bases (same command forms as the VALIDATION lanes) + K9b panel
SEL = lambda n: str(V / "contracts/device" / n)
INIT = str(pre / "PRETRAIN_FOR_FL_INIT.json")
dev = []
BASEARM = {"FT_C": "C_SMALL", "FT_FA": "FA", "FT_FA_WARM_FROZEN": "FA_WARM_FROZEN", "FT_FA_WARM": "FA_WARM"}
def base_of(arm, seed):           # launch_device LAUNCH "start": absolute "path" + "file_sha256" (MEASURED, DEV/FT_FA_s2026)
    L = sorted((R / "DEV" / f"{arm}_s{seed}").glob("LAUNCH_*.json"), key=lambda q: int(re.findall(r"\d+", q.name)[0]))
    sb = (j(L[-1]).get("start") or {}) if L else {}
    return sb.get("path"), sb.get("file_sha256")
for seed in (2026, 2027):                                                 # order: seed 2026 all arms, then 2027
    for arm, extra in (("LO", ["--selection", SEL("SELECTION_LO.json")]),
                       ("FT_C", ["--base", src[("C_SMALL", seed)]["PB"]["ckpt"], "--base-kind", "central", "--selection", SEL("SELECTION_FT_C.json")]),
                       ("FT_PRE", ["--init-from", INIT]),
                       ("FT_FA", ["--base", src[("FA", seed)]["PB"]["ckpt"], "--base-kind", "fl", "--selection", SEL("SELECTION_FT_FA.json")]),
                       ("FT_FA_WARM_FROZEN", ["--base", src[("FA_WARM_FROZEN", seed)]["PB"]["ckpt"], "--base-kind", "fa_warm"]),
                       ("FT_FA_WARM", ["--base", src[("FA_WARM", seed)]["PB"]["ckpt"], "--base-kind", "fa_warm"])):
        bsha = src[(BASEARM[arm], seed)]["PB"]["sha"] if "--base" in extra else None
        if bsha:                                                          # the VALIDATION run used exactly this base
            b, h = base_of(arm, seed); want = extra[extra.index("--base") + 1]
            if b is not None and (str(Path(b).resolve()) != str(Path(want).resolve()) or (h and h != bsha)):
                refuse(f"DEV/{arm}_s{seed}: VALIDATION base {b} / {h} differs from the locked base {want} / {bsha}")
        dev.append(dict(arm=arm, seed=seed, role="MAIN_TEST", base_sha=bsha, args=["--arm", arm, "--seed", str(seed),
                   "--role", "MAIN_TEST", "--run-dir", str(R / f"TEST/DEV/{arm}_MAIN_TEST_s{seed}"), "--n-shards", "64"] + extra))
for arm, extra in (("FT_C", ["--base", src[("C_SMALL", 2026)]["PB"]["ckpt"], "--base-kind", "central", "--selection", SEL("SELECTION_FT_C.json")]),
                   ("LO", ["--selection", SEL("SELECTION_LO.json")])):    # K9b: UNSEEN_HOLDOUT_TEST s2026
    dev.append(dict(arm=arm, seed=2026, role="UNSEEN_HOLDOUT_TEST", args=["--arm", arm, "--seed", "2026", "--role", "UNSEEN_HOLDOUT_TEST",
               "--run-dir", str(R / f"TEST/DEV/{arm}_UNSEEN_HOLDOUT_TEST_s2026"), "--n-shards", "64"] + extra,
               base_sha=src[("C_SMALL", 2026)]["PB"]["sha"] if arm == "FT_C" else None))
bases = [["--baseline", "POP", "--out", str(R / "TEST/BASE/POP")], ["--baseline", "LAST", "--out", str(R / "TEST/BASE/LAST")],
         ["--baseline", "SKNN", "--sknn-k", "500", "--sknn-sample-size", "5000", "--out", str(R / "TEST/BASE/SKNN")],
         ["--baseline", "RULE", "--out", str(R / "TEST/BASE/RULE")],
         ["--baseline", "KNN_EMB", "--item-table", src[("C_BIG", 2026)]["PB"]["ckpt"], "--family", "SASREC", "--table-role", "BIG", "--out", str(R / "TEST/BASE/KNN_EMB_BIG")],
         ["--baseline", "KNN_EMB", "--item-table", src[("C_SMALL", 2026)]["PB"]["ckpt"], "--family", "SASREC", "--table-role", "SMALL", "--out", str(R / "TEST/BASE/KNN_EMB_SMALL")]]
recs = fz["records"]
# FIX-1 (Fable gate 10-03): the K16 TEST dump + bench run from code_ds3r7b (= r7 + DEV-05 + P2 latency-only; DL 358 / 359 /
# 378 / 380; deployed beside r7). The lock pins its manifest aggregate + the two file shas; the deployed tree must be it.
m7b = j(R / "deploy_r7b/SYNC_MANIFEST.json")
for rel, h in m7b["files"].items():
    if sha(V / "code_ds3r7b" / rel[len("code/"):]) != h: refuse(f"code_ds3r7b/{rel} differs from deploy_r7b/SYNC_MANIFEST")
if m7b["base"]["aggregate_sha256"] != fz["code_trees"]["code_ds3r7"]["aggregate_sha256"]: refuse("r7b base is not the frozen r7")
R7B = {"tree": "code_ds3r7b", "path": str(V / "code_ds3r7b"), "aggregate_sha256": m7b["aggregate_sha256"],
       "tar_sha256": m7b["tar_sha256"], "base_r7_aggregate_sha256": m7b["base"]["aggregate_sha256"],
       "dump_rows_retro_sha256": m7b["files"]["code/v4int/dump_rows_retro.py"], "bench_sha256": m7b["files"]["code/v4dev/bench.py"],
       "used_by": "H onnx only (dump_rows_retro MAIN_TEST --ckpt binding + v4dev.bench); every other H step runs on code_ds3r7"}
# FIX-3 (DL 368): K16 pass-1 artifacts as frozen; pass-2 latency file sha if it exists by now (hashed, never parsed)
k16art = fz.get("k16_pass1_artifacts")
lp2 = Path(fz["k16_onnx_dir"]) / "LATENCY_PASS2.json" if fz.get("k16_onnx_dir") else None
if k16art:
    for rel, h in k16art.items():
        if sha(R / rel) != h: refuse(f"K16 pass-1 artifact {rel} changed after the freeze")
lock_part = {
 "document": "ALL_METHODS_LOCK", "status": "CANDIDATE (server-built; becomes the lock only by the local O_EXCL copy + register entry 5 + DL row)",
 "authority": "MASTER_PLAN_V5 §10; FABLE_ENDGAME_REVIEW_20261001 §1 G1 / D4; PREG1_AMENDMENT_06P; DL 315-321",
 "ana_script_freeze_sha256": E["FSHA"], "retention_report_sha256": fz["retention_report_sha256"],
 "validation_result_sha256": sha(E["VRES"]), "validation_block_index_sha256": sha(blk / "BLOCK_INDEX.json"),
 "contrasts_v4_file_sha256": sha(E["CONTRASTS"]), "contrasts_v4_self_sha256": c.get("self_sha256"),
 "comparison_key": {"eval_device_class": "NVIDIA_A40-FP32-DET", "pin_file_sha256": sha(E["PIN"]), "pin_self_sha256": j(E["PIN"]).get("self_sha256")},
 "records": recs, "fl_tuning_selection": "9bca186e… (S4)", "fa_warm_star": fz["fa_warm_star"],
 "warm_selection_files": {k: v for k, v in recs.items() if "FL_WARM_SELECTION" in k},
 "warm_s_record": {k: v for k, v in recs.items() if "S_RECORD_WARM_FROZEN" in k},
 "n33_mapping": fz["n33_mapping"], "exclusions_by_name": fz["exclusions_by_name"],
 "code_trees": dict(fz["code_trees"], code_ds3r7b=R7B), "launch_code_identity": fz["launch_code_identity"],
 "declared_not_run": fz["declared_not_run"], "parked_smoke": fz.get("parked_smoke"),
 "descriptive_metric_table": fz.get("descriptive_metric_table"),
 "k16_pass1_artifacts": k16art, "k16_latency": fz.get("k16_latency"),
 "k16_latency_pass2_sha256": (sha(lp2) if lp2 is not None and lp2.is_file() else "PASS2_PENDING (DL 368)"),
 "arms": arms, "checkpoints": {f"{a}_s{s}": t for (a, s), t in sorted(src.items())},
 "test_manifests_to_build": ["MAIN_RETRO_TEST_5_100_131072", "UNEW_HOLDOUT_RETRO_TEST_32768", "TRUE_COLD_RETRO_TEST",
                             "SPARSE_PANEL_16384_RETRO_TEST", "HEAVY_PANEL_16384_RETRO_TEST"],
 "device_statement": "device arms re-run their LOCKED procedure on TEST (LO, FT_C, FT_PRE, FT_FA, FT_FA_WARM_FROZEN, FT_FA_WARM; "
                     "SELECTION_FT_C e47965d2… / SELECTION_LO / SELECTION_FT_FA and the locked bases); --select is refused on TEST",
 "contest_singles": ("06L-C D1: 02B_SASREC = the C_BIG s2026 / s2027 MAIN_TEST reevals of this plan; 02B_GRU declared not run "
                     "(r7 reeval admits SASREC variants only; see declared_not_run)"),
 "test_plan": {"reevals": reevals, "panels_from_validation": panels, "true_cold_admitted": sorted(admitted), "device": dev,
               "baselines": bases, "onnx": bool(fz.get("k16_included")),
               "onnx_validation_dir": fz.get("k16_onnx_dir"), "onnx_first_users": fz.get("k16_first_users"),
               "onnx_tree": R7B["tree"], "onnx_tree_aggregate_sha256": R7B["aggregate_sha256"],
               "onnx_dump_rows_retro_sha256": R7B["dump_rows_retro_sha256"], "onnx_bench_sha256": R7B["bench_sha256"],
               "onnx_latency_rule": ("the TEST bench's in-run latency is loaded-host (not the device number) unless a TEST "
                                     "pass 2 (--latency-only on the identical TEST artifact) is run after Wave 1; the "
                                     "retention / accuracy rows are unaffected (DL 368)")},
}
# Q7 RESOLVED: load_contrasts_v4 compares every v4_registry_content() key and ignores other keys (retention_report
# L1072-1081); content_sha256 is built from the registered keys only, so the lock = v4 contrasts verbatim + one key
# "all_methods_lock" loads on TEST and matches VALIDATION_RESULT's contrasts_content_sha256 (L5107). self_sha256 is
# recomputed over the new body with the utf8-compact rule company/gates/*.json use (_self_ok, L4639-4647 -> VERIFIED).
lock = dict(c); lock["all_methods_lock"] = lock_part
lock["contrasts_v4_self_sha256_before_lock"] = c.get("self_sha256")
lock["self_sha256"] = hashlib.sha256(json.dumps({k: v for k, v in lock.items() if k != "self_sha256"}, sort_keys=True,
                                                separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")).hexdigest()
out = LD / "ALL_METHODS_LOCK.candidate.json"
fd = os.open(out, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
with os.fdopen(fd, "w", encoding="utf-8") as f: json.dump(lock, f, indent=1, sort_keys=True, ensure_ascii=False)
ls = sha(out)
entry = {"actor_role": "Operations Director", "source": "company/gates/ALL_METHODS_LOCK.json", "split": "RETRO_TEST",
         "access_purpose": "MANIFEST_BUILD", "observations": "NONE",
         "observations_description": "ALL_METHODS_LOCK written before any TEST manifest exists; no TEST row read",
         "decision": "NO_FEEDBACK", "all_methods_lock_sha256": ls, "manifest_header_sha256": None, "model_checkpoint_sha256": None,
         "code_sha256": fz["code_trees"]["code_ds3r7"]["aggregate_sha256"], "receipt_path": "company/gates/ALL_METHODS_LOCK.json"}
fd = os.open(LD / "REGISTER_ENTRY_5.json", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
with os.fdopen(fd, "w", encoding="utf-8") as f: json.dump(entry, f, indent=1, sort_keys=True)
print("LOCK_CANDIDATE_WRITTEN", ls, "reevals", len(reevals), "device", len(dev), "panels", len(panels))
PY
date -u +%FT%TZ > $OK/G_$ANA_TAG.ok
step "QUEUE_DONE G rc=0 lock candidate $(sha $LD/ALL_METHODS_LOCK.candidate.json | cut -c1-16) — fetch, O_EXCL copy, register entry 5, DL row, upload"
