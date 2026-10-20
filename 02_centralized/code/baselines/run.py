"""v4base.run — the BASE-01/BASE-02 CLI.

  python -B -m v4base.run --baseline {POP,LAST,SKNN,KNN_EMB,RULE} --phase {CALIBRATION,MAIN,MAIN_TEST} --out <dir>
      [--grid] [--allow-main-validation] [--sknn-k K --sknn-sample-size S]
      [--item-table <ckpt path> --family {GRU,SASREC} --table-role {SMALL,BIG}]
      [--history-policy USER_HISTORY_L50] [--registry <REGISTRY_V2.json path>]
      [--train-manifest ID --eval-manifest ID] [--n-threads N] [--fit-chunk N]

BASE-02 (contracts/PREG1_AMENDMENT_04.md §2): KNN_EMB (`v4base.knn_emb`, needs `--item-table`/`--family`/
`--table-role`; the SAME flags run BIG or SMALL — "both are just different checkpoint paths" per the task packet,
code path is identical either way) and RULE (`v4base.rule`, category/brand affinity, needs a real user identity for
its window-dedup — resolved here via `v4base.identity`, never a raw batch field). Both still go through the ONE
frozen T3 path (`scoring.run_baseline_evaluation`) and get BASELINE-labelled rows exactly like POP/LAST/SKNN.

**KNN_EMB row label (review fix, MED)**: a SMALL-table run and a BIG-table run must never share one row label —
`--table-role {SMALL,BIG}` is REQUIRED for KNN_EMB and produces `method`/`family` = `KNN_EMB_SMALL` / `KNN_EMB_BIG`
(so the frozen R4 rows are `BASELINE_KNN_EMB_SMALL` / `BASELINE_KNN_EMB_BIG`, never a bare, ambiguous `KNN_EMB`);
the checkpoint's own sha256 is also recorded in `RECEIPT.json["config"]["item_table_sha256"]`.

Order: `configure_low_priority()` first (os.nice(10), torch.set_num_threads(<= 4)) — before any data is touched,
per the Amendment 02 §4 resource rule; MAIN is refused unless `--allow-main-validation` is passed (Ops passes it
only from P3 on); the manifest ids default to REGISTRY_V2's own CALIBRATION/MAIN role entries, read DIRECTLY from
`contracts/manifest_registry/REGISTRY_V2.json` (its `roles` block) — NOT through
`v4train.registry.load_manifest_registry`, whose `ROLES` whitelist (MAIN, CALIBRATION, CONTINUATION, FULL_TRAIN,
FEATURE_SEARCH_SCREEN) refuses REGISTRY_V2 as a whole because the file also carries DATA-06's other population-band
roles (UNSEEN_VAL, SPARSE_PANEL, ...) — the exact finding INT-02's handoff records for the trainer itself. This CLI
sidesteps it by reading the two ids it needs straight out of the registry JSON and handing them to
`v4int.providers.StoreTrainProvider`/`StoreEvalProvider`, whose OWN verifier
(`v4int.verify.verify_manifest_by_registry`) never applies that role whitelist — REGISTRY_V2 passed to the providers
unfiltered, exactly as INT-02 already established is safe.

`execute()` is the CLI-independent core (providers/config passed in, never re-derived from argv): `tests_base`
drives it directly with synthetic providers so the whole pipeline (fit -> score -> evaluate -> rows -> receipt) is
exercised locally without server data — the real `StoreTrainProvider`/`StoreEvalProvider` wiring in `main()` is a
server-only case (no `data_v3` manifests on this host; see company/handoffs/BASE-01.md).

`--phase MAIN_TEST` (INT-07; PREG1_AMENDMENT_06D §2, the single TEST look): refused before any data unless the DATA-09
TEST gate is open (v4int.test_roles.test_gate_ok); `--grid` and `--train-manifest` / `--eval-manifest` are refused
(the SKNN point comes from the CALIBRATION grid; the pair is the registered MAIN_TEST pair). The role is resolved under
the gate from `--registry` (default the DATA-09 REGISTRY_V2_TEST_EXT.json), the DATA-09 loader wrappers are installed
(role=None), the scorer is fitted on MAIN_TRAIN exactly as for MAIN (the byte-identical REGISTRY_V2 entry, verified
through the extension), and the RETRO_TEST rows are scored through the gated store (v4int.retro_store); the rows' key
comes from comparison_key_base (split RETRO_TEST) and RECEIPT.json records the gate.

ANA-04 (PREG1_AMENDMENT_06L_C §C3, "baseline runs also write per-row rank files"): the scored evaluation of every
baseline also writes the frozen evaluator's per-row prediction artifact next to rows.json — `ranks_int32.npy` (the
int32 rank of the target per manifest row, -1 where the row is not RANKABLE; the same file form as v4int.reeval's
evals/<role>.ranks_int32.npy) and `n_tied_int32.npy` (the T3 per-row tie count, -1 where not ranked) — and
`RANKS_RECORD.json` (identity + file sha256 only: method, manifest ids, split, n_rows, the sha256 of rows.json and of
both arrays; no metric value), so tools/ana_block_index.py can bind the rank file to its rows without parsing a value.
`evaluate_with_ranks` is `scoring.run_baseline_evaluation` step for step (frozen config, runtime flags,
predict_scores over the same score stream, the same evaluate call) that also returns the prediction artifact; the
result dict — and therefore rows.json — is unchanged. The SKNN grid (CALIBRATION selection) writes no rank file.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Optional, Tuple

HERE = Path(__file__).resolve().parent
CODE = HERE.parent
V4 = CODE.parent
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))
sys.dont_write_bytecode = True

from v4eval import frozen_config as fc  # noqa: E402

from . import grid as base_grid  # noqa: E402
from . import knn_emb as base_knn_emb  # noqa: E402
from . import last as base_last  # noqa: E402
from . import pop as base_pop  # noqa: E402
from . import rule as base_rule  # noqa: E402
from . import scoring as base_scoring  # noqa: E402
from . import sknn as base_sknn  # noqa: E402
from .guard import RssGuard, configure_low_priority  # noqa: E402

REGISTRY_V2_PATH = V4 / "contracts" / "manifest_registry" / "REGISTRY_V2.json"
BASELINES = ("POP", "LAST", "SKNN", "KNN_EMB", "RULE")
RANKS_FILE = "ranks_int32.npy"                 # ANA-04 / 06L-C §C3: per-row target ranks (-1 = not RANKABLE)
TIES_FILE = "n_tied_int32.npy"                 # the T3 per-row tie count of the same artifact (-1 = not ranked)
RANKS_RECORD = "RANKS_RECORD.json"             # identity + sha256 only (no metric value)
RANKS_RECORD_FORMAT = "PPSI_V4_BASELINE_RANKS_1"
PHASES = ("CALIBRATION", "MAIN", "MAIN_TEST")
TEST_PHASE = "MAIN_TEST"
TABLE_ROLES = ("SMALL", "BIG")


class MainValidationRefused(SystemExit):
    """Raised (a SystemExit subclass) when --phase MAIN is requested without --allow-main-validation."""


def check_main_gate(phase: str, allow_main_validation: bool) -> None:
    """Amendment 02 §4 / the task packet: refuse phase MAIN unless --allow-main-validation is passed (Ops passes it
    only from P3 on). Called BEFORE any provider/data is touched."""
    if phase == "MAIN" and not allow_main_validation:
        raise MainValidationRefused("MAIN refused: pass --allow-main-validation (Ops passes it only from P3 on, "
                                    "per contracts/PREG1_AMENDMENT_02.md §4)")


class TestPhaseRefused(SystemExit):
    """Raised (a SystemExit subclass) when --phase MAIN_TEST is requested with the TEST gate closed or with a selection
    / manifest-override option (INT-07)."""


def check_test_phase(phase: str, *, grid: bool, train_manifest=None, eval_manifest=None) -> None:
    """INT-07: --phase MAIN_TEST only under an open DATA-09 TEST gate, never with --grid (selection) or manifest
    overrides. Called BEFORE any provider / data is touched."""
    if phase != TEST_PHASE:
        return
    if grid:
        raise TestPhaseRefused("MAIN_TEST refused with --grid: the SKNN point is selected on CALIBRATION, never on TEST")
    if train_manifest is not None or eval_manifest is not None:
        raise TestPhaseRefused("MAIN_TEST refused with --train-manifest / --eval-manifest: the registered pair only")
    from v4int import test_roles as TR
    if not TR.test_gate_ok(TEST_PHASE):
        raise TestPhaseRefused("MAIN_TEST refused: the DATA-09 TEST gate is closed (ALL_METHODS_LOCK + its "
                               "MANIFEST_BUILD TEST_EXPOSURE_REGISTER entry)")


def resolve_phase_manifests(phase: str, registry_path: Path) -> Tuple[str, str]:
    """(train_manifest_id, eval_manifest_id) of `phase`, read directly from the registry's own `roles` block (see
    the module docstring for why `v4train.registry.load_manifest_registry` is not used here)."""
    doc = json.loads(Path(registry_path).read_text(encoding="utf-8"))
    entry = (doc.get("roles") or {}).get(phase)
    if not entry or "train" not in entry or "eval" not in entry:
        raise ValueError(f"{registry_path}: role {phase!r} has no train/eval manifest ids")
    return str(entry["train"]), str(entry["eval"])


def fit_scorer(baseline: str, train_provider, K: int, *, sknn_k: Optional[int] = None,
              sknn_sample_size: Optional[int] = None, fit_chunk: Optional[int] = None,
              guard: Optional[RssGuard] = None, item_table_path: Optional[str] = None,
              family: Optional[str] = None, user_keys=None):
    kwargs = {} if fit_chunk is None else {"chunk": fit_chunk}
    if baseline == "POP":
        return base_pop.fit(train_provider, K, guard=guard, **kwargs)
    if baseline == "LAST":
        return base_last.LastItemScorer(K)                # stateless; train_provider is never scanned
    if baseline == "SKNN":
        if sknn_k is None or sknn_sample_size is None:
            raise ValueError("SKNN needs sknn_k and sknn_sample_size (direct, or via --grid)")
        pool = base_sknn.fit(train_provider, K, guard=guard, **kwargs)
        return base_sknn.SessionKNNScorer(pool, k=sknn_k, sample_size=sknn_sample_size)
    if baseline == "KNN_EMB":
        if item_table_path is None or family is None:
            raise ValueError("KNN_EMB needs item_table_path and family (--item-table / --family)")
        fallback = base_pop.fit(train_provider, K, guard=guard, **kwargs)     # the registered empty-history fallback
        table = base_knn_emb.load_item_table(item_table_path, family, K)
        return base_knn_emb.KnnEmbScorer(table, fallback.counts)
    if baseline == "RULE":
        if user_keys is None:
            raise ValueError("RULE needs user_keys (a within-this-scan user grouping key for window de-dup; "
                             "v4base.identity.train_client_ord_by_canonical_position for a real provider)")
        pop_scorer = base_pop.fit(train_provider, K, guard=guard, **kwargs)
        cat_of, brand_of = base_rule.fit_category_brand_map(train_provider, K, user_keys=user_keys, guard=guard,
                                                            **kwargs)
        tables = base_rule.RuleTables(pop_scorer.counts, cat_of, brand_of)
        return base_rule.RuleScorer(tables)
    raise ValueError(f"unknown baseline {baseline!r} (expected {BASELINES})")


def run_grid(train_provider, eval_provider, K: int, cfg: fc.FrozenEvalConfig, *, fit_chunk: Optional[int] = None,
            guard: Optional[RssGuard] = None) -> dict:
    """The registered SKNN grid (Amendment 02 §4): fit the past-session pool ONCE on CALIB_TRAIN, re-evaluate on
    CALIB_VALIDATION per (k, sample_size) point, select by macro MRR@20 E2E."""
    kwargs = {} if fit_chunk is None else {"chunk": fit_chunk}
    pool = base_sknn.fit(train_provider, K, guard=guard, **kwargs)
    points = []
    for k, s in base_grid.grid_points():
        scorer = base_sknn.SessionKNNScorer(pool, k=k, sample_size=s)
        result = base_scoring.run_baseline_evaluation(scorer, eval_provider.manifest, eval_provider.batches(cfg.eval_batch),
                                                      model_sha256=base_scoring.v4base_code_sha256(), config=cfg)
        points.append({"k": k, "sample_size": s, "value": base_grid.metric_value(result)})
        if guard is not None:
            guard.check()
    return {"points": points, "selected": base_grid.select_best(points)}


def evaluate_with_ranks(scorer, manifest, batches, *, model_sha256: str,
                        config: Optional[fc.FrozenEvalConfig] = None) -> Tuple[dict, object]:
    """`scoring.run_baseline_evaluation` step for step (frozen T3 config, its runtime flags, predict_scores over the
    scorer's stream, the same evaluate call and evaluator sha), returning (result, prediction artifact). The artifact
    carries the per-row ranks / n_tied the frozen evaluator computed; the result is the one run_baseline_evaluation
    returns for the same inputs (tests_ana/test_ana04.py checks the equality)."""
    from v4eval.evaluate import evaluate
    from v4eval.predict import predict_scores
    cfg = fc.require_frozen(config) if config is not None else fc.load()
    fc.apply_runtime_flags(cfg)
    art = predict_scores(base_scoring.score_batches(scorer, batches), manifest, model_sha256=model_sha256, config=cfg)
    result = evaluate(manifest, art, config=cfg, evaluator_code_sha256=base_scoring.evaluator_code_sha256())
    return result, art


def _sha256_file(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write_rank_files(out_dir: Path, art, manifest, *, method: str, baseline: str, key_base: dict) -> dict:
    """06L-C §C3: ranks_int32.npy + n_tied_int32.npy (one entry per manifest row, manifest order) and RANKS_RECORD.json
    (identity and sha256 only; the rows.json sha256 binds the rank file to its rows). Refuses a non-aligned artifact."""
    import numpy as np
    out_dir = Path(out_dir)
    ranks = np.asarray(art.ranks, dtype=np.int64)
    ties = np.asarray(art.n_tied, dtype=np.int64)
    if ranks.shape != (int(manifest.n),) or ties.shape != (int(manifest.n),):
        raise ValueError(f"prediction artifact has {ranks.shape} ranks / {ties.shape} ties, the manifest {manifest.n} rows")
    if not np.array_equal(np.asarray(art.decision_id), np.asarray(manifest.decision_id)):
        raise ValueError("prediction artifact rows are not the manifest rows in manifest order")
    if int(ranks.max(initial=-1)) >= 2 ** 31 or int(ties.max(initial=-1)) >= 2 ** 31:
        raise ValueError("a rank / tie count does not fit int32")
    np.save(out_dir / RANKS_FILE, ranks.astype(np.int32))
    np.save(out_dir / TIES_FILE, ties.astype(np.int32))
    rec = {"format": RANKS_RECORD_FORMAT, "baseline": baseline, "method": method,
           "eval_manifest_id": str(manifest.manifest_id),
           "train_manifest_id": key_base.get("train_manifest_id"), "split": key_base.get("split"),
           "n_rows": int(manifest.n), "n_rankable": int(np.count_nonzero(ranks > 0)),
           "eval_ordered_decision_id_sha256": (manifest.hashes or {}).get("eval_ordered_decision_id_sha256"),
           "ranks_file": RANKS_FILE, "ranks_sha256": _sha256_file(out_dir / RANKS_FILE), "ranks_dtype": "int32",
           "n_tied_file": TIES_FILE, "n_tied_sha256": _sha256_file(out_dir / TIES_FILE),
           "rows_file": "rows.json", "rows_sha256": _sha256_file(out_dir / "rows.json"),
           "v4base_code_sha256": base_scoring.v4base_code_sha256(),
           "evaluator_code_sha256": base_scoring.evaluator_code_sha256(),
           "authority": "PREG1_AMENDMENT_06L_C §C3 (baseline runs also write per-row rank files)"}
    body = json.dumps(rec, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    rec["self_sha256"] = hashlib.sha256(body).hexdigest()           # the launchers' compact self-sha rule
    (out_dir / RANKS_RECORD).write_text(json.dumps(rec, indent=2, sort_keys=True), encoding="utf-8")
    return rec


def execute(baseline: str, *, train_provider, eval_provider, key_base: dict, out_dir: Path, K: int,
           grid: bool = False, sknn_k: Optional[int] = None, sknn_sample_size: Optional[int] = None,
           item_table_path: Optional[str] = None, family: Optional[str] = None, table_role: Optional[str] = None,
           user_keys=None, fit_chunk: Optional[int] = None, guard: Optional[RssGuard] = None,
           low_priority: Optional[dict] = None, config: Optional[fc.FrozenEvalConfig] = None,
           receipt_extra: Optional[dict] = None) -> dict:
    """The CLI-independent core: `train_provider` (a BatchProvider) / `eval_provider` (an EvalProvider) and
    `key_base` (the R4 comparison key, e.g. `v4int.providers.comparison_key_base(paths, train, eval)`) are supplied
    by the caller — `main()` builds them from the real store, `tests_base` from synthetic stand-ins.

    `table_role` (`{SMALL,BIG}`, REQUIRED for KNN_EMB): a SMALL-table and a BIG-table run must never share one row
    label (review fix, MED) — the row/family method becomes `KNN_EMB_<table_role>`, never a bare `KNN_EMB`.
    `user_keys` (REQUIRED for RULE): a within-this-scan user grouping key for `rule.fit_category_brand_map`'s
    window de-duplication (`v4base.identity.train_client_ord_by_canonical_position` for a real provider)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = fc.require_frozen(config) if config is not None else fc.load()
    t0 = time.time()
    grid_result = None
    if grid:
        if baseline != "SKNN":
            raise ValueError("--grid only applies to SKNN")
        grid_result = run_grid(train_provider, eval_provider, K, cfg, fit_chunk=fit_chunk, guard=guard)
        sknn_k = grid_result["selected"]["k"]
        sknn_sample_size = grid_result["selected"]["sample_size"]
    method = baseline
    item_table_sha256 = None
    if baseline == "KNN_EMB":
        if table_role not in TABLE_ROLES:
            raise ValueError(f"KNN_EMB needs table_role in {TABLE_ROLES} (--table-role)")
        method = f"KNN_EMB_{table_role}"
        item_table_sha256 = hashlib.sha256(Path(item_table_path).read_bytes()).hexdigest()
    scorer = fit_scorer(baseline, train_provider, K, sknn_k=sknn_k, sknn_sample_size=sknn_sample_size,
                       item_table_path=item_table_path, family=family, user_keys=user_keys, fit_chunk=fit_chunk,
                       guard=guard)
    result, art = evaluate_with_ranks(scorer, eval_provider.manifest, eval_provider.batches(cfg.eval_batch),
                                      model_sha256=base_scoring.v4base_code_sha256(), config=cfg)
    if guard is not None:
        guard.check()
    rows = base_scoring.build_rows(result, key_base, method=method)
    wall_s = time.time() - t0
    receipt = {"baseline": baseline, "method": method,
              "config": {"sknn_k": sknn_k, "sknn_sample_size": sknn_sample_size, "grid": grid,
                        "item_table_path": item_table_path, "item_table_sha256": item_table_sha256,
                        "family": family, "table_role": table_role, "fit_chunk": fit_chunk,
                        "eval_batch": cfg.eval_batch,
                        "eval_config": cfg.eval_config("precomputed_scores", cfg.device_class)},
              "v4base_code_sha256": base_scoring.v4base_code_sha256(),
              "evaluator_code_sha256": base_scoring.evaluator_code_sha256(), "wall_s": round(wall_s, 3),
              "n_rows": len(rows), "grid": grid_result, "resource": (guard.report() if guard is not None else None),
              "low_priority": low_priority, **dict(receipt_extra or {})}
    (out_dir / "rows.json").write_text(json.dumps(rows, indent=2, default=str), encoding="utf-8")
    ranks_rec = write_rank_files(out_dir, art, eval_provider.manifest, method=method, baseline=baseline,
                                 key_base=key_base)                                    # 06L-C §C3 (ANA-04)
    receipt["ranks"] = {k: ranks_rec[k] for k in ("ranks_file", "ranks_sha256", "n_tied_file", "n_tied_sha256",
                                                  "rows_sha256", "self_sha256")}
    (out_dir / "RECEIPT.json").write_text(json.dumps(receipt, indent=2, default=str), encoding="utf-8")
    return {"rows": rows, "receipt": receipt, "result": result, "ranks_record": ranks_rec}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--baseline", required=True, choices=BASELINES)
    ap.add_argument("--phase", required=True, choices=PHASES)
    ap.add_argument("--out", required=True)
    ap.add_argument("--grid", action="store_true")
    ap.add_argument("--allow-main-validation", action="store_true")
    ap.add_argument("--sknn-k", type=int, default=None)
    ap.add_argument("--sknn-sample-size", type=int, default=None)
    ap.add_argument("--item-table", default=None, help="KNN_EMB: checkpoint path (SMALL primary, BIG reference)")
    ap.add_argument("--family", default=None, choices=base_knn_emb.FAMILIES, help="KNN_EMB: GRU or SASREC")
    ap.add_argument("--table-role", default=None, choices=TABLE_ROLES,
                    help="KNN_EMB: labels the row/family KNN_EMB_<role> (required for KNN_EMB)")
    ap.add_argument("--history-policy", default="USER_HISTORY_L50")
    ap.add_argument("--registry", default=None, help="default REGISTRY_V2 (MAIN_TEST: REGISTRY_V2_TEST_EXT)")
    ap.add_argument("--train-manifest", default=None)
    ap.add_argument("--eval-manifest", default=None)
    ap.add_argument("--n-threads", type=int, default=4)
    ap.add_argument("--fit-chunk", type=int, default=None)
    args = ap.parse_args(argv)

    check_main_gate(args.phase, args.allow_main_validation)
    check_test_phase(args.phase, grid=args.grid, train_manifest=args.train_manifest, eval_manifest=args.eval_manifest)

    low_priority = configure_low_priority(args.n_threads)     # before any data is touched (resource rule)
    guard = RssGuard()

    import v4train  # noqa: F401  (ppsi_env first, before v4int's torch/store imports)
    from v4int.providers import StoreEvalProvider, StoreTrainProvider, comparison_key_base
    from v4int.verify import DataPaths

    receipt_extra, test_gate = None, None
    if args.phase == TEST_PHASE:                                # INT-07: resolved and served under the TEST gate
        from v4int import test_roles as TR
        registry_path = Path(args.registry) if args.registry is not None else Path(TR.DEFAULT_EXT)
        TR.install(registry_path, role=None, eager=True)
        res = TR.resolve_role(TEST_PHASE, ext_path=registry_path)
        train_id, eval_id, test_gate = res["train"], res["eval"], res["gate"]
        receipt_extra = {"phase": TEST_PHASE, "test_gate": test_gate.binding(),
                         "registry_ext": {"path": str(registry_path), "self_sha256": res["registry_self_sha256"]}}
    else:
        registry_path = Path(args.registry) if args.registry is not None else REGISTRY_V2_PATH
        train_id, eval_id = args.train_manifest, args.eval_manifest
        if train_id is None or eval_id is None:
            rt, re_ = resolve_phase_manifests(args.phase, registry_path)
            train_id, eval_id = train_id or rt, eval_id or re_

    paths = DataPaths.default()
    train_provider = StoreTrainProvider(train_id, history_policy=args.history_policy, paths=paths,
                                        manifest_registry=registry_path)
    eval_provider = StoreEvalProvider(eval_id, history_policy=args.history_policy, paths=paths,
                                      manifest_registry=registry_path,
                                      **({} if test_gate is None else {"test_gate": test_gate}))
    key_base = comparison_key_base(paths, train_provider, eval_provider)
    if test_gate is not None and key_base["split"] != "RETRO_TEST":
        raise TestPhaseRefused("MAIN_TEST key is not a RETRO_TEST key")
    K = int(eval_provider.manifest.catalogue_K)

    user_keys = None
    if args.baseline == "RULE":
        from . import identity
        user_keys = identity.train_client_ord_by_canonical_position(train_provider, manifest_dir=paths.manifest_dir)

    execute(args.baseline, train_provider=train_provider, eval_provider=eval_provider, key_base=key_base,
           out_dir=Path(args.out), K=K, grid=args.grid, sknn_k=args.sknn_k, sknn_sample_size=args.sknn_sample_size,
           item_table_path=args.item_table, family=args.family, table_role=args.table_role, user_keys=user_keys,
           fit_chunk=args.fit_chunk, guard=guard, low_priority=low_priority, receipt_extra=receipt_extra)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
