"""Bounded preprocessing for the scoped T1 MVP: one controlled pass, no second slice.

What this script does, once, for run family ``mvp-t1-001``:

1. Recovers the producer's complete 97,279-user example slice from **all** clients in the
   frozen T2 TRAIN file, censored rows included. Deriving it from T1 alone, or reapplying
   a modulo, silently shifts every global decision position.
2. Loads the TRAIN and VALIDATION raw events for exactly those users through the existing
   ``ppsi.data.rees46`` loader, and asserts the producer's event counts.
3. Binds all 3,113,814 T1 TRAIN and 438,185 T1 VALIDATION frozen decisions to those events
   on the frozen keys, before any pilot subset exists. ``decision_order`` is a position in
   the complete split; a reduced frame would reinterpret it as a different row.
4. Fits the #32 count tables on every valid TRAIN decision and cross-checks the Markov
   table against the existing ``ppsi.models.evaluation.transition_table``.
5. Selects the 1,000-client pilot population and precomputes the 20-round schedule with
   the existing ``sample_clients``, then builds windows for the pilot's TRAIN rows and for
   every VALIDATION decision.

It is deliberately not PEP-723 executable: it imports project modules. Run it with the
pinned per-invocation pandas/numpy override documented in the execution runbook.

Nothing here reads, lists, hashes or reconstructs sealed TEST.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
import pyarrow
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ppsi.baselines.t1_simple import T1CountBaselines
from ppsi.data.rees46 import load_events, locate_decisions, read_json
from ppsi.data.sequences import build_windows
from ppsi.federated.clients import client_id_from_user
from ppsi.federated.mvp_data import MANIFEST_NAME, WINDOW_FIELDS
from ppsi.federated.mvp_support import raw_file_sha256, require_unique_clients
from ppsi.federated.sampling import sample_clients
from ppsi.models.evaluation import transition_table

# The producer's counts. These are assertions, never targets to filter towards.
COUNTS = {
    "TRAIN": {"events": 4376137, "t1": 3113814},
    "VALIDATION": {"events": 622013, "t1": 438185},
}
WINDOW_CHUNK = 100_000

FROZEN_INPUTS = {
    "cohort": "data/protocol/INTERNAL_DO_NOT_UPLOAD_cohort_manifest_v1.proposed.parquet",
    "excluded": "data/protocol/INTERNAL_DO_NOT_UPLOAD_excluded_sessions_v1.proposed.parquet",
    "vocabulary": "docs/evidence/s1-d1-ds-07/vocabulary_v1.proposed.json",
    "catalog": "fixtures/reference/item_catalog_v1.proposed.parquet",
    "protocol": "docs/evidence/s1-ds-05-06/data_protocol_v1.proposed.json",
    "t1_train": "data/examples/INTERNAL_DO_NOT_UPLOAD_task_examples_t1_train_v1.proposed.parquet",
    "t1_validation": "data/examples/INTERNAL_DO_NOT_UPLOAD_task_examples_t1_v1.proposed.parquet",
    "t2_train": "data/examples/INTERNAL_DO_NOT_UPLOAD_task_examples_t2_train_v1.proposed.parquet",
}


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def verify_frozen_inputs(root: Path) -> dict:
    """Hash every frozen input and compare it with the G1 gate's recorded prefix."""
    gate = json.loads(
        (root / "docs/evidence/s1-ds-09/g1_gate_v1.frozen.json").read_text(encoding="utf-8")
    )
    if gate["verdict"] != "GO":
        raise ValueError("G1 is not GO; preparation may not proceed")
    expected = {row["file"]: row["sha256"] for row in gate["artifacts"]}
    recorded = {}
    for label, rel in FROZEN_INPUTS.items():
        path = root / rel
        if not path.is_file():
            raise FileNotFoundError(f"frozen input missing: {rel}")
        digest = raw_file_sha256(path)
        prefix = expected.get(path.name)
        if prefix is None:
            raise ValueError(f"{path.name} is not a G1 artifact")
        if not digest.startswith(prefix):
            raise ValueError(f"frozen input digest mismatch: {path.name}")
        recorded[label] = {
            "path": rel,
            "sha256": digest,
            "g1_recorded_prefix": prefix,
            "prefix_length": len(prefix),
            "hash_convention": "raw_bytes",
        }
    return recorded


TEXT_COLUMNS = ("status", "cohort", "split")
NUMERIC_COLUMNS = (
    "client",
    "session",
    "decision_order",
    "task_mask",
    "label_value",
    "current_category",
    "category_changed",
)


def validate_t1_text_fields(examples_path: Path, split: str) -> None:
    """Check the three string contract fields without holding 3.1M Python strings.

    Each column is read on its own and reduced to its distinct values immediately, so the
    check costs one column at a time instead of a second full copy of the file.
    """
    available = set(pq.ParquetFile(examples_path).schema_arrow.names)
    missing = [name for name in (*TEXT_COLUMNS, *NUMERIC_COLUMNS) if name not in available]
    if missing:
        raise ValueError(f"missing TaskExample contract column: {missing}")
    expected = {"status": {"OBSERVED"}, "cohort": {"C1"}, "split": {split}}
    for column in TEXT_COLUMNS:
        table = pq.read_table(examples_path, columns=[column])
        values = table.column(column).unique().to_pylist()
        del table
        if None in values:
            raise ValueError(f"null in required TaskExample field: {column}")
        if set(values) != expected[column]:
            raise ValueError(f"unexpected frozen TaskExample {column}: {sorted(values)}")


def validate_t1_examples(examples: pd.DataFrame, split: str) -> None:
    """The frozen TaskExample contract checks, reused unchanged from the baseline lane."""
    if examples[list(NUMERIC_COLUMNS)].isna().any().any():
        raise ValueError("null in required TaskExample field")
    if not pd.api.types.is_bool_dtype(examples["task_mask"]):
        raise ValueError("task_mask must be boolean")
    if examples[["client", "session", "decision_order"]].duplicated().any():
        raise ValueError("duplicate frozen decision key")
    if not examples["task_mask"].all():
        raise ValueError("frozen T1 has no censored rows")
    labels = examples["label_value"].to_numpy(dtype=np.float64)
    if (
        not np.isfinite(labels).all()
        or not np.equal(labels, np.floor(labels)).all()
        or ((labels < 0) | (labels >= 588)).any()
    ):
        raise ValueError("invalid observed T1 label")
    if not pd.api.types.is_bool_dtype(examples["category_changed"]):
        raise ValueError("category_changed must be a boolean evaluation flag")
    if not np.array_equal(
        examples["category_changed"].to_numpy(),
        examples["current_category"].to_numpy() != labels,
    ):
        raise ValueError("frozen T1 category-change flag disagrees with its targets")


def bind_split(frame: pd.DataFrame, examples_path: Path, split: str) -> dict:
    """Locate every frozen decision globally and prove the binding row by row."""
    validate_t1_text_fields(examples_path, split)
    examples = pd.read_parquet(examples_path, columns=list(NUMERIC_COLUMNS))
    validate_t1_examples(examples, split)
    if len(examples) != COUNTS[split]["t1"]:
        raise ValueError(f"{split} T1 example count differs from the frozen producer count")

    positions, labels = locate_decisions(frame, examples_path)
    if len(positions) != len(examples):
        raise ValueError("frozen decisions were not all located in the loaded events")

    ordered = examples.sort_values("decision_order")
    declared = ordered["decision_order"].to_numpy()
    if declared.dtype.kind not in "iu" or declared.min() < 0 or declared.max() >= len(frame):
        raise ValueError("decision_order must identify a global canonical event row")
    if not np.array_equal(positions, declared):
        raise ValueError("located positions disagree with the frozen global decision order")

    rows = frame.iloc[positions]
    if not np.array_equal(rows["user"].to_numpy(), ordered["client"].to_numpy().astype("int64")):
        raise ValueError("frozen-to-raw client binding failed")
    if not np.array_equal(rows["session"].to_numpy(), ordered["session"].to_numpy()):
        raise ValueError("frozen-to-raw session binding failed")
    current = ordered["current_category"].to_numpy().astype("int64")
    if not np.array_equal(rows["category"].to_numpy().astype("int64"), current):
        raise ValueError("frozen-to-raw current category binding failed")
    if not np.array_equal(labels, ordered["label_value"].to_numpy().astype("int64")):
        raise ValueError("located labels disagree with the frozen targets")

    return {
        "positions": positions,
        "labels": labels,
        "current": current.astype("int32"),
        "category_changed": ordered["category_changed"].to_numpy().astype(bool),
        "clients": rows["user"].to_numpy().astype("int64"),
    }


class MissingTrainHistoryError(RuntimeError):
    """A VALIDATION client whose TRAIN history was never measured stops preparation."""


def require_measured_history(
    history_counts: pd.Series,
    validation_clients: list[str],
    *,
    evidence_dir: Path,
) -> dict[str, int]:
    """Map each VALIDATION client to its genuinely measured TRAIN event count.

    The frozen evaluator stratifies by TRAIN history, so a missing measurement is not a
    client with no history: it is a client whose history nobody looked up. Substituting a
    zero would move that client into the lowest bucket and quietly change a published
    slice, which is why this fails closed instead.

    A genuine measured zero is still accepted; only an absent measurement is fatal.
    """
    measured = {
        client_id_from_user(str(int(user))): int(value) for user, value in history_counts.items()
    }
    missing = sorted({client for client in validation_clients if client not in measured})
    if missing:
        # Opaque ids stay in the private run directory; the public error carries a count.
        evidence = evidence_dir / "FAILED_missing_train_history.json"
        evidence.write_text(
            json.dumps(
                {
                    "schema": "mvp_missing_train_history_v1",
                    "version": "1",
                    "validation_clients": len(set(validation_clients)),
                    "clients_without_measured_train_history": len(missing),
                    "client_ids": missing,
                    "basis": "TRUE_RAW_TRAIN_EVENT_ROWS_PER_CLIENT",
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        raise MissingTrainHistoryError(
            f"{len(missing)} VALIDATION clients have no measured TRAIN history; a missing "
            "measurement is never recorded as a zero. Evidence preserved at "
            f"{evidence.name} inside the private run directory."
        )
    return {client: measured[client] for client in set(validation_clients)}


def build_split_windows(
    frame: pd.DataFrame,
    positions: np.ndarray,
    labels: np.ndarray,
    *,
    out_dir: Path,
    split: str,
    history_length: int,
) -> dict[str, str]:
    """Write one ``.npy`` per channel in bounded chunks; peak memory is chunk-sized."""
    total = len(positions)
    shapes = {
        "category": ("int32", (history_length,)),
        "product": ("int32", (history_length,)),
        "event": ("int8", (history_length,)),
        "brand": ("int32", (history_length,)),
        "price_band": ("int8", (history_length,)),
        "gap": ("float32", (history_length,)),
        "lengths": ("int64", ()),
        "target": ("int64", ()),
        "query_category": ("int32", ()),
        "query_product": ("int32", ()),
        "query_brand": ("int32", ()),
        "query_price_band": ("int8", ()),
        "client": ("int64", ()),
    }
    if set(shapes) != set(WINDOW_FIELDS):
        raise ValueError("window channel set drifted from the prepared reader")
    handles = {}
    paths = {}
    for name, (dtype, shape) in shapes.items():
        path = out_dir / f"{split.lower()}_{name}.npy"
        paths[name] = path
        handles[name] = np.lib.format.open_memmap(
            path, mode="w+", dtype=dtype, shape=(total, *shape)
        )
    for start in range(0, total, WINDOW_CHUNK):
        stop = min(start + WINDOW_CHUNK, total)
        built = build_windows(
            frame, positions[start:stop], labels[start:stop], history_length=history_length
        )
        for name in shapes:
            handles[name][start:stop] = getattr(built, name)
        del built
        log(f"  {split} windows {stop:,} / {total:,}")
    for handle in handles.values():
        handle.flush()

    lengths = np.asarray(handles["lengths"])
    sample = min(total, 100_000)
    last = np.asarray(handles["category"])[np.arange(sample), lengths[:sample] - 1]
    if not (last >= 0).all():
        raise ValueError("a decision's own category is missing from its window")
    live = np.asarray(handles["gap"])[:sample][
        np.arange(history_length)[None, :] < lengths[:sample, None]
    ]
    if not live.max() > 0:
        raise ValueError("every time gap is zero; the timestamp unit is wrong")
    del handles
    return {name: raw_file_sha256(path) for name, path in paths.items()}


def logical_digest(payload: dict) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare the scoped T1 MVP inputs once.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run", required=True)
    parser.add_argument("--raw", type=Path, required=True)
    args = parser.parse_args()

    policy = json.loads((ROOT / args.config).read_text(encoding="utf-8"))
    if policy.get("schema") != "mvp_execution_policy_v1":
        raise ValueError("unexpected execution policy schema")
    if args.run != policy["run_family"]:
        raise ValueError("run id does not match the declared run family")

    branch = subprocess.run(
        ["git", "-C", str(ROOT), "branch", "--show-current"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    head = subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    log(f"branch={branch} head={head[:12]}")

    raw = (ROOT / args.raw).resolve() if not args.raw.is_absolute() else args.raw.resolve()
    if "sealed_test" in {part.lower() for part in raw.parts} or not raw.is_file():
        raise FileNotFoundError("canonical raw parquet is required; sealed TEST is prohibited")

    resources = policy["resources"]
    available = psutil.virtual_memory().available
    if available < resources["min_available_ram_gib"] * 1024**3:
        raise RuntimeError(
            f"BLOCKED_CAPACITY: {available / 1024**3:.2f} GiB available, "
            f"{resources['min_available_ram_gib']} GiB required before preparation"
        )
    for label, path in (("repository", ROOT), ("temp", Path(tempfile.gettempdir()))):
        free = psutil.disk_usage(str(path)).free
        if free < resources["min_free_disk_gib"] * 1024**3:
            raise RuntimeError(f"BLOCKED_CAPACITY: {label} volume below the free-disk floor")

    prepared = ROOT / policy["outputs"]["private"] / "prepared"
    if prepared.exists() and any(prepared.iterdir()):
        raise FileExistsError(
            "prepared directory already exists; verify and reuse it, or start a new run id. "
            "Overwriting an executed input in place destroys the run's identity."
        )
    prepared.mkdir(parents=True, exist_ok=True)

    inputs = verify_frozen_inputs(ROOT)
    log("frozen input digests verified against G1")

    vocabulary = read_json(ROOT / FROZEN_INPUTS["vocabulary"])
    mapping = vocabulary["categories"]["code_of_category_id"]
    if vocabulary["categories"]["count"] != 588 or set(mapping.values()) != set(range(588)):
        raise ValueError("frozen category vocabulary mismatch")
    category_code = {int(k): int(v) for k, v in mapping.items()}

    catalog = pd.read_parquet(ROOT / FROZEN_INPUTS["catalog"], columns=["item", "price_band"])
    price_band = catalog.set_index("item")["price_band"]

    cohort = pd.read_parquet(ROOT / FROZEN_INPUTS["cohort"])
    c1 = set(cohort.loc[cohort["cohort"].eq("C1"), "user_id"].astype("int64"))
    # Every client value in frozen T2 TRAIN, censored rows included: this is the producer's
    # complete example slice. T1-only or observed-only sets omit users and shift positions.
    users = {
        int(value)
        for value in pq.read_table(ROOT / FROZEN_INPUTS["t2_train"], columns=["client"])["client"]
        .unique()
        .to_pylist()
    }
    if (
        len(c1) != policy["data"]["base_c1_users"]
        or len(users) != policy["data"]["example_slice_users"]
        or not users.issubset(c1)
    ):
        raise ValueError("frozen C1 example-slice membership mismatch; never reapply a modulo")
    log(f"C1={len(c1):,} example slice={len(users):,}")

    excluded = set(pd.read_parquet(ROOT / FROZEN_INPUTS["excluded"])["session_key"])
    protocol = read_json(ROOT / FROZEN_INPUTS["protocol"])
    upstream = read_json(ROOT / "config/s1-ds-05-06.v1.json")
    null_prefix = upstream["session_policy"]["null_fallback_prefix"]
    history_length = int(policy["data"]["history_length"])

    manifest: dict = {
        "schema": "mvp_prepare_manifest_v1",
        "version": "1",
        "run": args.run,
        "scope": policy["scope"],
        "git": {"branch": branch, "head": head, "uncommitted_execution_source": True},
        "inputs": inputs,
        "raw": {
            "path": str(raw).replace("\\", "/"),
            "sha256": None,
            "rows": int(pq.ParquetFile(raw).metadata.num_rows),
            "note": "full-file SHA-256 is measured below; the file is local, not distributed",
        },
        "history_length": history_length,
        "splits": {},
        "sealed_test": {
            "files_opened": 0,
            "example_files_read": 0,
            "physical_no_read_claim": "NOT_CLAIMED_SEE_row_groups_decoded",
        },
        "environment": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "pyarrow": pyarrow.__version__,
        },
    }
    manifest["raw"]["sha256"] = raw_file_sha256(raw)
    log(f"raw parquet hashed: {manifest['raw']['sha256'][:16]}…")

    pilot_cfg = policy["pilot"]
    seed = int(pilot_cfg["seed"])
    bindings: dict[str, dict] = {}
    history_counts: pd.Series | None = None
    population: list[str] = []
    client_rows: dict[str, list[int]] = {}

    for split in ("TRAIN", "VALIDATION"):
        boundary = protocol["temporal_split"][split]
        start = pd.Timestamp(boundary["start"])
        end = pd.Timestamp(boundary["end_exclusive"])
        log(f"loading {split} events {start} .. {end}")
        frame = load_events(
            raw,
            start=start,
            end=end,
            users=users,
            excluded=excluded,
            null_prefix=null_prefix,
            category_code=category_code,
            price_band=price_band,
        )
        provenance = dict(frame.attrs["provenance"])
        if len(frame) != COUNTS[split]["events"]:
            raise ValueError(
                f"{split} raw membership/order count {len(frame):,} differs from the producer's "
                f"{COUNTS[split]['events']:,}"
            )
        log(f"  {split} events={len(frame):,} sessions={frame['session'].nunique():,}")

        examples_path = ROOT / (
            FROZEN_INPUTS["t1_train"] if split == "TRAIN" else FROZEN_INPUTS["t1_validation"]
        )
        bound = bind_split(frame, examples_path, split)
        log(f"  {split} decisions bound: {len(bound['positions']):,}")

        if split == "TRAIN":
            # TRUE raw TRAIN event counts per client, not task-example counts.
            history_counts = frame.groupby("user", sort=True).size()
            baselines = T1CountBaselines.fit(
                bound["current"], bound["labels"], split="TRAIN", category_count=588
            )
            reference = transition_table(bound["current"].astype("int64"), bound["labels"], 588)
            if not np.array_equal(baselines.transition_counts.astype("float64"), reference):
                raise ValueError("count Markov table disagrees with the existing comparator")
            np.savez(
                prepared / "count_tables_full.npz",
                target_counts=baselines.target_counts,
                transition_counts=baselines.transition_counts,
                train_decisions=np.int64(baselines.train_decisions),
            )
            manifest["count_tables_full"] = {
                "content_sha256": baselines.content_sha256(),
                "train_decisions": int(baselines.train_decisions),
                "scope": policy["baselines"]["scope_full"],
                "markov_cross_checked_against": "ppsi.models.evaluation.transition_table",
            }
            log(f"  full count tables fitted on {baselines.train_decisions:,} decisions")

            # Pilot pool: observed T1 TRAIN presence only. VALIDATION never influences it.
            decision_users = np.unique(bound["clients"])
            opaque = {int(u): client_id_from_user(str(int(u))) for u in decision_users}
            pool = require_unique_clients([opaque[int(u)] for u in decision_users])
            wanted = int(pilot_cfg["population_clients"])
            if len(pool) < wanted:
                raise RuntimeError(
                    f"BLOCKED_POPULATION: {len(pool)} eligible clients, {wanted} required; "
                    "the pilot size is fixed before metrics and is never shrunk to fit"
                )
            selection = sample_clients(
                pool,
                seed,
                0,
                wanted,
                sampler_version="mvp_population_v1",
            )
            population = list(selection.selected_client_ids)
            manifest["population"] = {
                "eligible_pool": len(pool),
                "selected": len(population),
                "sampler_version": "mvp_population_v1",
                "experiment_seed": seed,
                "round_index": 0,
                "selected_digest": selection.selected_digest,
                "eligibility": pilot_cfg["eligible"],
            }
            log(f"  pilot population selected from {len(pool):,} eligible clients")

            chosen = set(population)
            row_client = np.array([opaque[int(u)] for u in bound["clients"]], dtype=object)
            keep = np.array([cid in chosen for cid in row_client])
            pilot_positions = bound["positions"][keep]
            pilot_labels = bound["labels"][keep]
            pilot_current = bound["current"][keep]
            pilot_clients = row_client[keep]
            if len(pilot_positions) == 0:
                raise RuntimeError("the pilot population has no TRAIN decisions")

            order = np.argsort(pilot_positions, kind="stable")
            pilot_positions = pilot_positions[order]
            pilot_labels = pilot_labels[order]
            pilot_current = pilot_current[order]
            pilot_clients = pilot_clients[order]
            for row, client in enumerate(pilot_clients):
                client_rows.setdefault(client, []).append(int(row))
            missing = [c for c in population if c not in client_rows]
            if missing:
                raise RuntimeError("a selected pilot client has no TRAIN rows")

            pilot_counts = T1CountBaselines.fit(
                pilot_current, pilot_labels, split="TRAIN", category_count=588
            )
            np.savez(
                prepared / "count_tables_pilot.npz",
                target_counts=pilot_counts.target_counts,
                transition_counts=pilot_counts.transition_counts,
                train_decisions=np.int64(pilot_counts.train_decisions),
            )
            manifest["count_tables_pilot"] = {
                "content_sha256": pilot_counts.content_sha256(),
                "train_decisions": int(pilot_counts.train_decisions),
                "scope": policy["baselines"]["scope_pilot"],
            }
            np.save(prepared / "train_pilot_current_category.npy", pilot_current)
            digests = build_split_windows(
                frame,
                pilot_positions,
                pilot_labels,
                out_dir=prepared,
                split=split,
                history_length=history_length,
            )
            manifest["splits"]["TRAIN"] = {
                "rows": len(pilot_positions),
                "scope": "PILOT_CLIENTS_ONLY_ALL_THEIR_FROZEN_T1_TRAIN_DECISIONS",
                "split_decisions_total": len(bound["positions"]),
                "split_events": len(frame),
                "raw_provenance": provenance,
                "array_sha256": digests,
            }
        else:
            np.save(prepared / "validation_current_category.npy", bound["current"])
            np.save(prepared / "validation_category_changed.npy", bound["category_changed"])
            validation_clients = [client_id_from_user(str(int(u))) for u in bound["clients"]]
            if history_counts is None:
                raise RuntimeError(
                    "TRAIN history counts were never measured; VALIDATION cannot be prepared "
                    "before its TRAIN split"
                )
            per_client = require_measured_history(
                history_counts, validation_clients, evidence_dir=prepared
            )
            (prepared / "validation_clients.json").write_text(
                json.dumps(
                    {
                        "client_ids": validation_clients,
                        "train_history_counts": {
                            client: per_client[client] for client in sorted(set(validation_clients))
                        },
                        "history_count_basis": "TRUE_RAW_TRAIN_EVENT_ROWS_PER_CLIENT",
                        "missing_history_policy": "FAIL_CLOSED_NEVER_DEFAULT_TO_ZERO",
                    },
                    separators=(",", ":"),
                ),
                encoding="utf-8",
            )
            digests = build_split_windows(
                frame,
                bound["positions"],
                bound["labels"],
                out_dir=prepared,
                split=split,
                history_length=history_length,
            )
            manifest["splits"]["VALIDATION"] = {
                "rows": len(bound["positions"]),
                "scope": "ALL_FROZEN_T1_VALIDATION_DECISIONS",
                "split_decisions_total": len(bound["positions"]),
                "split_events": len(frame),
                "raw_provenance": provenance,
                "array_sha256": digests,
                "distinct_clients": len(set(validation_clients)),
            }
        bindings[split] = {"decisions": len(bound["positions"])}
        del frame, bound

    rounds = []
    for server_round in range(1, int(pilot_cfg["rounds"]) + 1):
        result = sample_clients(
            population,
            seed,
            server_round - 1,
            int(pilot_cfg["clients_per_round"]),
        )
        rounds.append(
            {
                "server_round": server_round,
                "round_index": server_round - 1,
                "selected_client_ids": list(result.selected_client_ids),
                "selected_digest": result.selected_digest,
            }
        )
    participations = sum(len(entry["selected_client_ids"]) for entry in rounds)
    unique_participants = len({c for entry in rounds for c in entry["selected_client_ids"]})
    (prepared / "round_schedule.json").write_text(
        json.dumps(
            {
                "sampler_version": "client_sampler_v1",
                "experiment_seed": seed,
                "round_index_basis": "ZERO_BASED_round_index = server_round - 1",
                "rounds": rounds,
            },
            separators=(",", ":"),
        ),
        encoding="utf-8",
    )
    (prepared / "pilot_client_rows.json").write_text(
        json.dumps({"population": population, "client_rows": client_rows}, separators=(",", ":")),
        encoding="utf-8",
    )
    manifest["schedule"] = {
        "rounds": len(rounds),
        "clients_per_round": int(pilot_cfg["clients_per_round"]),
        "scheduled_participations": participations,
        "unique_participating_clients": unique_participants,
        "schedule_digest": logical_digest([entry["selected_digest"] for entry in rounds]),
    }
    manifest["pilot_rows"] = {
        "train_rows": manifest["splits"]["TRAIN"]["rows"],
        "min_client_rows": min(len(v) for v in client_rows.values()),
        "max_client_rows": max(len(v) for v in client_rows.values()),
        "per_client_truncation": "NONE",
    }
    manifest["data_manifest_sha256"] = logical_digest(
        {
            "run": args.run,
            "inputs": {k: v["sha256"] for k, v in inputs.items()},
            "raw": manifest["raw"]["sha256"],
            "splits": {
                name: {"rows": body["rows"], "arrays": body["array_sha256"]}
                for name, body in manifest["splits"].items()
            },
            "population": manifest["population"]["selected_digest"],
            "schedule": manifest["schedule"]["schedule_digest"],
            "count_tables": [
                manifest["count_tables_full"]["content_sha256"],
                manifest["count_tables_pilot"]["content_sha256"],
            ],
        }
    )
    (prepared / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    public = ROOT / policy["outputs"]["public"]
    public.mkdir(parents=True, exist_ok=True)
    summary = {
        key: manifest[key]
        for key in (
            "schema",
            "version",
            "run",
            "scope",
            "inputs",
            "history_length",
            "splits",
            "population",
            "schedule",
            "pilot_rows",
            "count_tables_full",
            "count_tables_pilot",
            "sealed_test",
            "environment",
            "data_manifest_sha256",
        )
    }
    summary["schema"] = "mvp_prepare_summary_v1"
    summary["note"] = (
        "Counts, digests and provenance only. No client identity, session key or row "
        "content is published here."
    )
    summary["raw_rows"] = manifest["raw"]["rows"]
    (public / "prepare_summary.v1.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log("preparation complete")
    print(
        json.dumps(
            {
                "prepared": str(prepared.relative_to(ROOT)).replace("\\", "/"),
                "train_rows": manifest["splits"]["TRAIN"]["rows"],
                "validation_rows": manifest["splits"]["VALIDATION"]["rows"],
                "data_manifest_sha256": manifest["data_manifest_sha256"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
