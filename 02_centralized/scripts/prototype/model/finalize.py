"""Close S2-DS-01: train the selected configuration, save what the other lanes need.

The sweep decides *which* configuration. This produces the artifacts that decision implies:

* a checkpoint per project seed, so `S2-SE-01` has something to export and `S2-PR-09` has
  the centralized reference to reproduce
* the common initialisation and its digest for seeds 13, 42 and 2026, so `S2-PR-06` can
  prove R2a started exactly where the matching R1 did
* a CI parity fixture - a tiny model with fixed weights and a fixed input - because
  `S2-SE-04` asks for one by name
* the frozen config, and the full result with every diagnostic

The strata here are bucketed by **TRAIN events**, the definition `ADR-001` specifies.
Bucketing by decisions instead puts 3,272 clients in the 10-19 bucket where the protocol
says 2,924, and the whole stratified table shifts.
"""

from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd
import torch
from ladder import SEED_SPREAD
from train import (
    CATEGORIES,
    OUTPUT,
    SPEC,
    Split,
    score,
    train,
    train_events_per_client,
)

from ppsi.models.evaluation import by_client_history, coverage_of_top5, divergence_of_top1
from ppsi.models.session_gru import (
    ALLOWED_SEEDS,
    SessionGRUConfig,
    build_model,
    common_initialization,
    parameter_count,
)

BASELINE = {"slice_macro": 0.3143, "slice_micro": 0.3085, "overall_micro": 0.8560}
BASELINE_BY_BUCKET = {"10-19": 0.3302, "20-49": 0.3111, "50-99": 0.3088, "100+": 0.3128}


# Settled by probe_schedule.py: a cosine schedule from 0.001 over 30 epochs. A constant
# 0.0003 and a constant 0.0001 landed within 0.0008 of it, well inside the 0.0033
# run-to-run spread, so the three are indistinguishable and the schedule is chosen for
# being principled rather than hand-tuned - and for converging in 30 epochs rather than 40.
SCHEDULE = "cosine"
SCHEDULE_RATE = 0.001
EPOCHS = 30


def selected_config() -> tuple:
    sweep = json.loads((OUTPUT / "sweep.json").read_text(encoding="utf-8"))
    chosen = sweep["selected"]
    config = SessionGRUConfig(
        channels=tuple(chosen["channels"]),
        use_gap=chosen["use_gap"],
        widths=dict(chosen["widths"]),
        hidden=chosen["hidden"],
        layers=chosen["layers"],
        dropout=chosen["dropout"],
    )
    return config, SCHEDULE_RATE, sweep


def parity_fixture(config: SessionGRUConfig) -> dict:
    """A tiny deterministic model and input for S2-SE-04's ONNX parity test in CI.

    Small enough to run in seconds, real enough to exercise the paths that break an export:
    variable history lengths, a zero-length row, and every categorical channel.
    """
    from ppsi.data.batching import windows_to_batch
    from ppsi.data.sequences import build_windows

    frame = pd.DataFrame(
        {
            "session": [1, 1, 1, 2, 2],
            "order": [0, 1, 2, 3, 4],
            "user": [1, 1, 1, 2, 2],
            "category": np.array([3, 11, 42, 7, 7], dtype="int32"),
            "product_bucket": np.array([31, 77, 88, 12, 19], dtype="int32"),
            "event_code": np.array([1, 2, 3, 1, 1], dtype="int8"),
            "brand_bucket": np.array([4, 6, 6, 2, 2], dtype="int32"),
            "price_band": np.array([1, 3, 2, 4, 0], dtype="int8"),
            "event_time": pd.to_datetime(
                [
                    "2019-10-01 00:00:00",
                    "2019-10-01 00:00:10",
                    "2019-10-01 00:02:00",
                    "2019-10-02 08:00:00",
                    "2019-10-02 08:00:30",
                ],
                utc=True,
            ),
        }
    )
    windows = build_windows(frame, np.array([1, 2, 4]), np.array([42, 7, 7]), history_length=4)
    batch = windows_to_batch(windows, np.arange(3), SPEC)
    model = build_model(13, batch_spec=SPEC, config=config).eval()
    with torch.no_grad():
        output = model(batch)
    return {
        "history_length": 4,
        "batch_size": 3,
        "lengths": batch.lengths.tolist(),
        "inputs": {name: tensor.tolist() for name, tensor in batch.history_categorical_ids.items()},
        "gap": batch.history_continuous_features.squeeze(-1).tolist(),
        "query": {name: tensor.tolist() for name, tensor in batch.query_categorical_ids.items()},
        "expected_t1_logits_row0_first8": [round(v, 6) for v in output.t1_logits[0, :8].tolist()],
        "expected_t2_logit": [round(v, 6) for v in output.t2_logit.squeeze(-1).tolist()],
        "tolerance": 1e-4,
        "note": "S2-SE-04: export this model to ONNX, run these inputs through both "
        "runtimes, and fail if any output differs by more than the tolerance",
    }


def main() -> None:
    started = time.time()
    config, learning_rate, sweep = selected_config()
    print(
        f"selected: {config.channels}  hidden {config.hidden}  layers {config.layers}  "
        f"dropout {config.dropout}  lr {learning_rate}"
    )
    print(f"parameters: {parameter_count(build_model(13, batch_spec=SPEC, config=config)):,}\n")

    train_split, validation = Split.load("train"), Split.load("validation")
    rows = np.arange(len(train_split))
    events = train_events_per_client()

    checkpoints, runs, strata_rows, diagnostics = {}, [], {}, {}
    for seed in ALLOWED_SEEDS:
        model, curve, best_epoch = train(
            train_split,
            validation,
            config=config,
            seed=seed,
            rows=rows,
            batch_size=512,
            learning_rate=learning_rate,
            max_epochs=EPOCHS,
            patience=EPOCHS,
            label=f"final | seed {seed}",
            schedule=SCHEDULE,
        )
        measured = score(model, validation, legacy_clamp=False, popularity=np.zeros(CATEGORIES))

        path = OUTPUT / f"s2_ds_01_gru_t1_seed{seed}.pt"
        torch.save(
            {
                "seed": seed,
                "config": {
                    "channels": list(config.channels),
                    "use_gap": config.use_gap,
                    "widths": config.widths,
                    "hidden": config.hidden,
                    "layers": config.layers,
                    "dropout": config.dropout,
                },
                "state_dict": model.state_dict(),
            },
            path,
        )
        checkpoints[seed] = {"file": path.name, "MB": round(path.stat().st_size / 1e6, 2)}

        strata = by_client_history(
            measured["_slice_values"], validation.data["client"], events, measured["_changed"]
        )
        strata["baseline"] = pd.Series(BASELINE_BY_BUCKET)
        strata["gain"] = (strata["macro"] - strata["baseline"]).round(4)
        strata_rows[seed] = strata.round(4).to_dict()

        runs.append(
            {
                "seed": seed,
                "best_epoch": best_epoch,
                "slice_macro": measured["slice_macro"],
                "slice_micro": measured["slice_micro"],
                "overall_micro": measured["overall_micro"],
                "final_train_val_gap": curve[-1]["gap"],
            }
        )
        diagnostics[seed] = {
            "coverage": coverage_of_top5(measured["_top5"]),
            "per_client": divergence_of_top1(measured["_top1"], validation.data["client"]),
        }
        print(f"    slice macro {measured['slice_macro']}   saved {path.name}")
        print(strata.round(4).to_string())
        print()
        del model

    frame = pd.DataFrame(runs)
    mean = float(frame["slice_macro"].mean())
    spread = float(frame["slice_macro"].max() - frame["slice_macro"].min())
    gain = mean - BASELINE["slice_macro"]

    print("=" * 78)
    print(frame.to_string(index=False))
    print(f"\n  mean slice macro   : {mean:.4f}")
    print(f"  spread over seeds  : {spread:.4f}")
    print(f"  baseline           : {BASELINE['slice_macro']}")
    print(f"  gain               : {gain:+.4f}  =  {gain / max(spread, 1e-9):.1f}x the spread")

    # The common initialisation the federated lane must start R2a from.
    initialisations = {}
    for seed in ALLOWED_SEEDS:
        state, digest = common_initialization(seed, batch_spec=SPEC, config=config)
        path = OUTPUT / f"common_initialization_seed{seed}.pt"
        torch.save(state, path)
        initialisations[seed] = {"file": path.name, "sha256": digest}
        print(f"  common init seed {seed}: {digest[:16]}...")

    fixture = parity_fixture(config)
    (OUTPUT / "ci_parity_fixture.json").write_text(json.dumps(fixture, indent=2), encoding="utf-8")

    result = {
        "task": "S2-DS-01",
        "merges": ["S2-DS-04"],
        "status": "PROPOSED",
        "scope": {
            "cohort": "C1",
            "user_slice": "user_id % 4 == 1",
            "measured_users": 97279,
            "train_events": 4376137,
            "train_decisions": len(train_split),
            "validation_decisions": len(validation),
            "test_rows_read": 0,
        },
        "selected_config": {
            **sweep["selected"],
            "learning_rate": SCHEDULE_RATE,
            "schedule": SCHEDULE,
            "epochs": EPOCHS,
        },
        "baseline": BASELINE,
        "runs": runs,
        "mean_slice_macro": round(mean, 4),
        "seed_spread": round(spread, 4),
        "gain_over_baseline": round(gain, 4),
        "comparison_threshold": SEED_SPREAD,
        "beats_threshold": bool(gain > SEED_SPREAD),
        "by_client_history": strata_rows,
        "diagnostics": diagnostics,
        "checkpoints": checkpoints,
        "common_initialization": initialisations,
        "ci_parity_fixture": "ci_parity_fixture.json",
        "environment": {
            "torch": torch.__version__,
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        },
    }
    (OUTPUT / "s2_ds_01_result.json").write_text(
        json.dumps(result, indent=2, default=str), encoding="utf-8"
    )
    print(f"\ntotal {(time.time() - started) / 60:.0f} min")


if __name__ == "__main__":
    main()
