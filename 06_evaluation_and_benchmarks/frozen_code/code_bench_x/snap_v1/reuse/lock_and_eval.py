"""Model lock + locked VC/TEST evaluation runner -- Wave-1 Sonnet C (owner).

Implements FULL_RUN_PROTOCOL.md sec. 7 (model lock), sec. 8 (locked
evaluation), sec. 9 (pre-registered decision rules). Subcommands:

    lock     --f0-run DIR --f4-run DIR [--locks-dir DIR]
    confirm  [--locks-dir DIR] [--out PATH] [--rerun-after-crash REASON]
             [--dry-run-split SPLIT]
    test     [--locks-dir DIR] [--out PATH] [--rerun-after-crash REASON]
             [--dry-run-split SPLIT]

`--dry-run-split VALIDATION_SELECT` (with `--locks-dir`/`--out` pointed at a
scratch directory) exercises the WHOLE confirm/test mechanics -- lock
refusal, claim exclusivity + crash-recovery, checkpoint sha256 mismatch
refusal, the tie -> F0 rule -- on VALIDATION_SELECT data instead of
VALIDATION_CONFIRM/TEST, so the machinery can be verified without ever
loading a sealed split. It substitutes the pilot's 10k-user VS cohort
(`cache/vs` + `validation_select_cohort.parquet`) as a stand-in store/
DecisionSet wherever the real VC/TEST split would go, and refits POP/TRANS
on the pilot's TRAIN manifest (`cache/train`) rather than the full
40,180,608-row TRAIN population -- both purely for speed; this is a
mechanics smoke test, not a scientific VS evaluation (baselines_full.py's
`full-vs` CLI mode is the one that reports real full-scale numbers).

TEST and VALIDATION_CONFIRM decisions are NEVER loaded except by a real
(non-dry-run) `confirm`/`test` invocation, and only once each (the claim/done
files below refuse a second run).

Real VC/TEST runs happen only later, by Opus, once Sonnet A's `full_store.py`
(`CompactStore`) exists and `train_full.py` has produced both arms' completed
runs. The non-dry-run code paths here are written directly against the
documented `FULL_INTERFACES.md` contract (`store.decisions`,
`store.decision_item_tokens`) but are NOT exercised by this module's own
verification (see the Sonnet C report for exactly what was and was not
run, and the `summary.json` "completed" schema assumption pending
`train_full.py`).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

CODE_DIR = Path(__file__).resolve().parent
VENDOR_DIR = CODE_DIR / "vendor"
for _p in (str(CODE_DIR), str(VENDOR_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402
import torch  # noqa: E402

import train as train_mod  # noqa: E402
import model as model_mod  # noqa: E402
from window_cache import CacheReader  # noqa: E402
from eval_full import evaluate_model, per_user_metric, paired_bootstrap, DEFAULT_K  # noqa: E402
import baselines_full  # noqa: E402

WORKSPACE = Path(
    "/home/jovyan/ahmedabdelhamed01/Privacy-Preserving-Shopping-Intelligence/.local_ai/gru_v1"
)
DATASET_ROOT = Path("/home/jovyan/shared_data/rees46_nextitem_v1")
PROTOCOL_PATH = WORKSPACE / "FULL_RUN_PROTOCOL.md"
VERSION_HASH = "860e5a1c3a81b8e4bbfeff4f245fa37638a43e4f260df179585964d2d4da8dbd"
EVAL_K = DEFAULT_K

LOCKS_DIR_DEFAULT = WORKSPACE / "locks"
FINAL_EVAL_DIR_DEFAULT = WORKSPACE / "final_eval"

ARM_NAMES = ("F0_item_only_h256", "F4_rich_side_information_h256")
ARM_TO_GROUP = {
    "F0_item_only_h256": "F0_item_only",
    "F4_rich_side_information_h256": "F4_rich_side_information",
}
# Files hashed into MODEL_LOCK.json's "code_sha256" -- whichever of these
# Wave-1 deliverables exist at lock time (full_store.py / train_full.py may
# not exist yet if `lock` is invoked early against completed pilot runs for
# a mechanics smoke test).
OWNED_CODE_FILES = (
    "full_store.py", "train_full.py", "eval_full.py", "baselines_full.py",
    "lock_and_eval.py", "model.py", "objective.py", "evaluator.py",
    "window_cache.py", "train.py",
)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_json_exclusive(path: Path, obj: dict) -> None:
    """Atomic, exclusive create -- fails (FileExistsError) if `path` already
    exists. Used for MODEL_LOCK.json / *_CLAIM.json / *_DONE.json so a second
    invocation can never silently overwrite a first."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(json.dumps(obj, indent=2, default=str))
    except Exception:
        try:
            os.unlink(path)
        except OSError:
            pass
        raise


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# lock
# ---------------------------------------------------------------------------

def _is_completed(summary: dict) -> bool:
    """Whether a run's summary.json reports a completed run. `train_full.py`
    (Sonnet B) had not been written at the time this module was implemented,
    so its exact summary.json schema is unknown; this checks an explicit
    "status" field if present (accepting anything mentioning "complete" or
    "stop", since the protocol's stopping conditions are all a form of
    "stopped on purpose"), and otherwise falls back to the pilot's own
    schema (`train.py:run_training`'s summary.json has no "status" key at
    all -- `best_metric`/`update_number` presence is treated as "completed"
    for that schema). Flagged in the Sonnet C report as an assumption Opus
    should confirm once train_full.py's real schema exists."""
    # Opus integration (2026-09-23): train_full.py's summary.json carries
    # `final_status`, and ONLY "COMPLETE" means the arm stopped by its
    # pre-registered rule (patience or max epochs, protocol sec. 6).
    # STOPPED_NONFINITE / STOPPED_DISK / STOPPED_TEST_HOOK are not locks-eligible.
    # The pilot-schema fallback and the loose substring match were removed:
    # they would have let a NaN- or disk-stopped arm through to the lock.
    if summary.get("final_status") != "COMPLETE":
        return False
    last = (summary.get("selection_history") or [{}])[-1]
    return bool(last.get("stop_now")) and last.get("stop_reason") in ("patience", "max_epochs")


def _best_metric_and_epoch(summary: dict) -> tuple[float, Optional[int]]:
    metric = summary.get("best_metric")
    if metric is None:
        raise KeyError("summary.json has no 'best_metric'")
    epoch = summary.get("best_epoch", summary.get("pass_index"))
    return float(metric), epoch


def cmd_lock(args) -> int:
    locks_dir = Path(args.locks_dir) if args.locks_dir else LOCKS_DIR_DEFAULT
    run_dirs = {"F0_item_only_h256": Path(args.f0_run), "F4_rich_side_information_h256": Path(args.f4_run)}

    arm_info: dict = {}
    for arm_name, run_dir in run_dirs.items():
        summary_path = run_dir / "summary.json"
        if not summary_path.exists():
            raise FileNotFoundError(f"{arm_name}: no summary.json in {run_dir}")
        summary = _read_json(summary_path)
        if not _is_completed(summary):
            raise RuntimeError(f"{arm_name}: run at {run_dir} does not report a completed status "
                                f"(summary={summary})")
        best_ckpt = run_dir / "checkpoints" / "best.pt"
        if not best_ckpt.exists():
            raise FileNotFoundError(f"{arm_name}: no best.pt at {best_ckpt}")
        metric, epoch = _best_metric_and_epoch(summary)
        config_path = run_dir / "config.yaml"
        arm_info[arm_name] = {
            "run_dir": str(run_dir),
            "best_checkpoint_path": str(best_ckpt),
            "best_checkpoint_sha256": _sha256_file(best_ckpt),
            "epoch": epoch,
            "metric": metric,
            "config_path": str(config_path) if config_path.exists() else None,
        }
        log(f"{arm_name}: metric={metric:.6f} epoch={epoch} sha256={arm_info[arm_name]['best_checkpoint_sha256'][:12]}...")

    m0 = arm_info["F0_item_only_h256"]["metric"]
    m4 = arm_info["F4_rich_side_information_h256"]["metric"]
    # protocol sec. 7.1: higher best full-VS primary metric wins; exact tie -> F0.
    selected = "F4_rich_side_information_h256" if m4 > m0 else "F0_item_only_h256"
    log(f"selection: F0={m0:.6f} F4={m4:.6f} -> selected={selected}")

    code_sha256 = {}
    for name in OWNED_CODE_FILES:
        p = CODE_DIR / name
        code_sha256[name] = _sha256_file(p) if p.exists() else None

    manifest = {
        "selected_arm": selected,
        "arms": arm_info,
        "protocol_sha256": _sha256_file(PROTOCOL_PATH) if PROTOCOL_PATH.exists() else None,
        "protocol_path": str(PROTOCOL_PATH),
        "dataset_version_hash": VERSION_HASH,
        "code_sha256": code_sha256,
        "utc": _utcnow(),
    }
    lock_path = locks_dir / "MODEL_LOCK.json"
    _write_json_exclusive(lock_path, manifest)
    log(f"wrote {lock_path}")
    return 0


# ---------------------------------------------------------------------------
# confirm / test -- shared machinery
# ---------------------------------------------------------------------------

def _verify_lock_checksums(lock: dict) -> None:
    for arm_name, info in lock["arms"].items():
        path = Path(info["best_checkpoint_path"])
        if not path.exists():
            raise FileNotFoundError(f"{arm_name}: checkpoint missing at {path} (was present at lock time)")
        actual = _sha256_file(path)
        if actual != info["best_checkpoint_sha256"]:
            raise RuntimeError(
                f"{arm_name}: checkpoint sha256 mismatch -- locked={info['best_checkpoint_sha256']} "
                f"actual={actual} at {path}; refusing to evaluate a checkpoint that changed since lock"
            )


def _load_model_from_lock_info(info: dict, group: str, device) -> torch.nn.Module:
    ckpt = torch.load(info["best_checkpoint_path"], map_location="cpu", weights_only=False)
    poc = train_mod.load_product_idx_of_class()
    model = model_mod.build_model(group, train_mod.VOCAB_SIZES, product_idx_of_class=poc)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def _claim_or_refuse(claim_path: Path, done_path: Path, rerun_after_crash: Optional[str]) -> None:
    if done_path.exists():
        raise RuntimeError(f"{done_path} already exists -- this split has already been evaluated once; refusing")
    if claim_path.exists():
        if not rerun_after_crash:
            raise RuntimeError(
                f"{claim_path} already exists (a previous run claimed this split and did not finish) -- "
                "pass --rerun-after-crash '<reason>' to acknowledge the incident and continue"
            )
        record = _read_json(claim_path)
        incidents = record.setdefault("incidents", [])
        incidents.append({"utc": _utcnow(), "reason": rerun_after_crash, "pid": os.getpid()})
        claim_path.write_text(json.dumps(record, indent=2, default=str), encoding="utf-8")
        log(f"recovered claim at {claim_path} after crash: {rerun_after_crash!r}")
    else:
        _write_json_exclusive(claim_path, {
            "claimed_at_utc": _utcnow(), "pid": os.getpid(), "host": socket.gethostname(), "incidents": [],
        })


# -- dry-run stand-in store (pilot 10k VS cohort) ----------------------------

class _DryRunStore:
    """Stand-in for CompactStore, backed by the pilot's CacheReader over
    `cache/vs` (the 10k-user VALIDATION_SELECT cohort) -- see the module
    docstring: dry-run only, never used for a real VC/TEST evaluation."""

    def __init__(self, cache_dir: Path):
        self.reader = CacheReader(cache_dir)
        self.channels = "F4"

    def batch(self, ds, positions):
        rows = ds.end_row[np.asarray(positions)]
        return self.reader.read(rows)

    def decision_item_tokens(self, ds):
        raw = self.reader.read(ds.end_row)
        item_tokens = raw["item_tokens"].numpy()
        lengths = raw["lengths"].numpy()
        return item_tokens[np.arange(len(ds)), lengths - 1]


class _DryRunDecisionSet:
    """Minimal DecisionSet stand-in (see `_DryRunStore`); duck-types against
    `eval_full.aggregate` / `evaluate_model`."""

    def __init__(self, decision_id, end_row, length, target_class, loss_mask, eval_mask,
                 client_idx, eval_user_key, prior_view_count, prefix_position, last_item):
        self.decision_id = decision_id
        self.end_row = end_row
        self.length = length
        self.target_class = target_class
        self.loss_mask = loss_mask
        self.eval_mask = eval_mask
        self.client_idx = client_idx
        self.eval_user_key = eval_user_key
        self.prior_view_count = prior_view_count
        self.prefix_position = prefix_position
        self.last_item = last_item  # not part of the real DecisionSet contract; baselines_full's
        # run_pop_baseline/run_trans_baseline read it directly off `eval_rows` (see
        # baselines_full.EvalRows) rather than via store.decision_item_tokens(ds), so the dry-run
        # stand-in carries it here too for a uniform call site.

    def __len__(self):
        return int(self.decision_id.shape[0])


def _dry_run_ds(cache_dir: Path, cohort_parquet: Path) -> _DryRunDecisionSet:
    reader = CacheReader(cache_dir)
    n = reader.header["row_count"]
    raw = reader.read(np.arange(n))
    cohort = pq.read_table(cohort_parquet, columns=["eval_user_key"])
    eval_user_key = np.asarray(cohort["eval_user_key"].to_numpy(zero_copy_only=False), dtype=np.int64)
    item_tokens = raw["item_tokens"].numpy()
    lengths = raw["lengths"].numpy()
    last_item = item_tokens[np.arange(n), lengths - 1]
    return _DryRunDecisionSet(
        decision_id=raw["decision_id"].numpy(), end_row=np.arange(n, dtype=np.int64),
        length=lengths, target_class=raw["target_class"].numpy(),
        loss_mask=raw["loss_mask"].numpy(), eval_mask=raw["eval_mask"].numpy(),
        client_idx=raw["client_id"].numpy(), eval_user_key=eval_user_key,
        prior_view_count=np.zeros(n, dtype=np.int64), prefix_position=np.ones(n, dtype=np.int64),
        last_item=last_item,
    )


def _build_real_store(splits: tuple):
    try:
        import full_store  # Sonnet A's module -- not present at the time this was written.
    except ImportError as exc:
        raise RuntimeError(
            "the real (non-dry-run) confirm/test path requires Sonnet A's full_store.py "
            "(CompactStore), which is not in code/ yet -- use --dry-run-split for mechanics "
            "verification until Opus integrates it"
        ) from exc
    return full_store.CompactStore.build(DATASET_ROOT, splits=splits, channels="F4")


def _evaluate_four_models(lock: dict, device, *, split_name: str, dry_run: bool,
                           k: int = EVAL_K, return_ranks: bool = False) -> dict:
    selected_arm = lock["selected_arm"]
    other_arm = [a for a in ARM_NAMES if a != selected_arm][0]

    eval_last_item = None
    if dry_run:
        store = _DryRunStore(WORKSPACE / "cache" / "vs")
        ds = _dry_run_ds(WORKSPACE / "cache" / "vs", WORKSPACE / "artifacts" / "validation_select_cohort.parquet")
        fit = baselines_full.fit_from_cache(WORKSPACE / "cache" / "train", baselines_full.EXPECTED_K)
    else:
        train_splits = ("TRAIN", "VALIDATION_SELECT", "VALIDATION_CONFIRM")
        splits = train_splits if split_name == "VALIDATION_CONFIRM" else train_splits + ("TEST",)
        store = _build_real_store(splits)
        ds = store.decisions(split_name, "eval")
        ds_loss = store.decisions("TRAIN", "loss")
        fit = baselines_full.fit_from_store(store, ds_loss)
        # a real DecisionSet has no `last_item` field (that's an EvalRows-only convenience for
        # readers b/c); reader (a)'s last-item token comes from the store itself.
        eval_last_item = store.decision_item_tokens(ds)

    results: dict = {"selected_arm": selected_arm, "other_arm": other_arm, "split": split_name,
                      "dry_run": dry_run}
    for arm_name in (selected_arm, other_arm):
        info = lock["arms"][arm_name]
        model = _load_model_from_lock_info(info, ARM_TO_GROUP[arm_name], device)
        results[arm_name] = evaluate_model(model, store, ds, device, k=k, return_ranks=True)

    results["POP_full"] = baselines_full.run_pop_baseline(fit, ds, device, k=k, return_ranks=True)
    results["TRANS_full"] = baselines_full.run_trans_baseline(
        fit, ds, device, k=k, return_ranks=True, last_item=eval_last_item)
    # FULL_RUN_PROTOCOL.md sec. 14 (2026-09-23): descriptive Recall/MRR/NDCG at k in {1,2,5,10}
    # from the SAME exact ranks, via the frozen evaluator helpers. Primary metric stays @20.
    for model_key in (selected_arm, other_arm, "POP_full", "TRANS_full"):
        results[model_key]["descriptive_at_k"] = _descriptive_at_k(results[model_key]["ranks"], ds)
        if not return_ranks:
            results[model_key]["_ranks_for_save"] = results[model_key].pop("ranks")
    if return_ranks:
        results["_ds"] = ds  # in-process only; popped before any JSON serialisation

    target_class = np.asarray(ds.target_class)
    eval_mask = np.asarray(ds.eval_mask).astype(bool)
    oov_count = int((eval_mask & (target_class < 0)).sum())
    n_eval = int(eval_mask.sum())
    results["oov"] = {"count": oov_count, "rate": (oov_count / n_eval) if n_eval else None, "n_eval": n_eval}
    return results


def _descriptive_at_k(ranks, ds, ks=(1, 2, 5, 10)) -> dict:
    from evaluator import _credits_from_ranks, _population_block
    ranks_t = torch.as_tensor(np.asarray(ranks), dtype=torch.long)
    tgt = torch.as_tensor(np.asarray(ds.target_class).astype(np.int64))
    em = torch.as_tensor(np.asarray(ds.eval_mask).astype(bool))
    keys = torch.as_tensor(np.asarray(ds.eval_user_key).astype(np.int64))
    out = {}
    for kk in ks:
        credits = _credits_from_ranks(ranks_t, kk)
        out[f"@{kk}"] = {"end_to_end": _population_block(credits, em, keys),
                         "closed": _population_block(credits, em & (tgt >= 0), keys)}
    return out


def _save_vc_ranks(results: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for model_key in (results["selected_arm"], results["other_arm"], "POP_full", "TRANS_full"):
        r = results[model_key].pop("_ranks_for_save", None)
        if r is not None:
            np.save(out_dir / f"vc_ranks_{model_key}.npy", r)


def _device() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available -- lock_and_eval.py requires the GPU-1 slot")
    return torch.device("cuda:0")


# ---------------------------------------------------------------------------
# confirm
# ---------------------------------------------------------------------------

def cmd_confirm(args) -> int:
    locks_dir = Path(args.locks_dir) if args.locks_dir else LOCKS_DIR_DEFAULT
    lock_path = locks_dir / "MODEL_LOCK.json"
    if not lock_path.exists():
        raise FileNotFoundError(f"no MODEL_LOCK.json at {lock_path} -- run `lock` first")
    lock = _read_json(lock_path)
    _verify_lock_checksums(lock)

    claim_path = locks_dir / "VC_CLAIM.json"
    done_path = locks_dir / "VC_DONE.json"
    _claim_or_refuse(claim_path, done_path, args.rerun_after_crash)

    dry_run = bool(args.dry_run_split)
    split_name = args.dry_run_split if dry_run else "VALIDATION_CONFIRM"
    device = _device()

    results = _evaluate_four_models(lock, device, split_name=split_name, dry_run=dry_run, k=EVAL_K)

    out_path = Path(args.out) if args.out else (FINAL_EVAL_DIR_DEFAULT / "VC_RESULTS.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _save_vc_ranks(results, out_path.parent)
    out_path.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    log(f"wrote {out_path}")

    _write_json_exclusive(done_path, {"completed_at_utc": _utcnow(), "out": str(out_path)})
    log(f"wrote {done_path}")
    return 0


# ---------------------------------------------------------------------------
# test
# ---------------------------------------------------------------------------

def cmd_test(args) -> int:
    locks_dir = Path(args.locks_dir) if args.locks_dir else LOCKS_DIR_DEFAULT
    lock_path = locks_dir / "MODEL_LOCK.json"
    if not lock_path.exists():
        raise FileNotFoundError(f"no MODEL_LOCK.json at {lock_path} -- run `lock` first")
    lock = _read_json(lock_path)
    _verify_lock_checksums(lock)

    vc_done_path = locks_dir / "VC_DONE.json"
    if not vc_done_path.exists():
        raise RuntimeError(f"no {vc_done_path} -- run `confirm` first (test requires VC_DONE)")

    claim_path = locks_dir / "TEST_CLAIM.json"
    done_path = locks_dir / "TEST_DONE.json"
    _claim_or_refuse(claim_path, done_path, args.rerun_after_crash)

    dry_run = bool(args.dry_run_split)
    split_name = args.dry_run_split if dry_run else "TEST"
    device = _device()

    results = _evaluate_four_models(lock, device, split_name=split_name, dry_run=dry_run, k=EVAL_K,
                                     return_ranks=True)

    selected_arm = results["selected_arm"]
    other_arm = results["other_arm"]
    ds = results.pop("_ds")

    def _per_user(model_key: str):
        return per_user_metric(results[model_key]["ranks"], ds, metric="mrr", k=EVAL_K,
                                denominator="end_to_end")

    # primary: selected GRU vs TRANS-full, macro MRR@20 end-to-end (protocol sec. 9).
    keys_gru, vals_gru = _per_user(selected_arm)
    keys_trans, vals_trans = _per_user("TRANS_full")
    primary = paired_bootstrap(keys_gru, vals_gru, keys_trans, vals_trans, n=1000, seed=2026)

    # secondary: F4 - F0 (whichever arm was/wasn't selected), same metric.
    f0_key = selected_arm if selected_arm.startswith("F0") else other_arm
    f4_key = selected_arm if selected_arm.startswith("F4") else other_arm
    keys_f4, vals_f4 = _per_user(f4_key)
    keys_f0, vals_f0 = _per_user(f0_key)
    secondary = paired_bootstrap(keys_f4, vals_f4, keys_f0, vals_f0, n=1000, seed=2026)

    results["bootstraps"] = {
        "primary_GRU_minus_TRANS": primary,
        "secondary_F4_minus_F0": secondary,
    }

    out_dir = Path(args.out).parent if args.out else FINAL_EVAL_DIR_DEFAULT
    out_dir.mkdir(parents=True, exist_ok=True)
    ranked_keys = (selected_arm, other_arm, "POP_full", "TRANS_full")
    for model_key in ranked_keys:
        if "ranks" in results.get(model_key, {}):
            np.save(out_dir / f"ranks_{model_key}.npy", results[model_key]["ranks"])

    serializable = {
        k: ({kk: vv for kk, vv in v.items() if kk != "ranks"} if k in ranked_keys else v)
        for k, v in results.items()
    }
    out_path = Path(args.out) if args.out else (FINAL_EVAL_DIR_DEFAULT / "TEST_RESULTS.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(serializable, indent=2, default=str), encoding="utf-8")
    log(f"wrote {out_path}")

    _write_json_exclusive(done_path, {"completed_at_utc": _utcnow(), "out": str(out_path)})
    log(f"wrote {done_path}")
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _configure_threads() -> None:
    os.environ.setdefault("OMP_NUM_THREADS", "3")
    os.environ.setdefault("MKL_NUM_THREADS", "3")
    torch.set_num_threads(3)
    # FULL_RUN_PROTOCOL.md sec. 4: strict FP32 (TF32 off for matmul and cuDNN),
    # cuDNN deterministic. lock_and_eval.py is a standalone entrypoint (unlike
    # `evaluate_model`, which assumes its caller already set these process-global
    # flags, e.g. train_full.py's set_determinism()) -- it must set them itself,
    # or its own GRU evaluations pick up TF32's ~1e-5 relative noise. Measured
    # concretely during this module's own --dry-run-split mechanics check: without
    # this, the locked GRU's macro MRR@20 on the 10k-cohort stand-in came out
    # 0.10115673393011093 against the recorded pilot figure 0.10115985572338104
    # (delta ~3e-6) -- small, but exactly the kind of silent drift the locked
    # VC/TEST numbers must never have.
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    lk = sub.add_parser("lock")
    lk.add_argument("--f0-run", required=True)
    lk.add_argument("--f4-run", required=True)
    lk.add_argument("--locks-dir", default=None)
    lk.set_defaults(func=cmd_lock)

    cf = sub.add_parser("confirm")
    cf.add_argument("--locks-dir", default=None)
    cf.add_argument("--out", default=None)
    cf.add_argument("--rerun-after-crash", default=None)
    cf.add_argument("--dry-run-split", default=None, choices=[None, "VALIDATION_SELECT"])
    cf.set_defaults(func=cmd_confirm)

    ts = sub.add_parser("test")
    ts.add_argument("--locks-dir", default=None)
    ts.add_argument("--out", default=None)
    ts.add_argument("--rerun-after-crash", default=None)
    ts.add_argument("--dry-run-split", default=None, choices=[None, "VALIDATION_SELECT"])
    ts.set_defaults(func=cmd_test)

    return ap


def main() -> int:
    _configure_threads()
    ap = build_arg_parser()
    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
