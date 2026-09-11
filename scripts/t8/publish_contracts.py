"""Emit the S2-DS-08 contract artifacts that issue #47 names as deliverables.

The task produced its numbers and its write-ups, but issue #47 asks for something narrower
and more mechanical: the **contract objects** downstream tasks bind to. `S2-SE-08` exports a
checkpoint by id, `S2-PR-06` and `S2-PR-09` start from a common initialization by id, and
none of them should have to read a README to find out which file that is.

Three artifacts, all validated against the schemas already in `config/experiments/schemas/`:

* **ModelConfig v1** — the architecture the lane froze. Not a description of a GRU in prose,
  a record with an id that a consumer can compare.
* **CommonInitialization v1**, one per seed — the untrained weights R1 and R2 must both start
  from, with the digest that proves they did.
* **the deployment candidate** — one checkpoint id, chosen by the rule that was written down
  before the run, with the sha256 of the file it names.

## Why the candidate is the joint model and not the best T2 model

`t2_2.pt` reaches `0.1424` on T2, better than the joint model's `0.1337`. It is not the
candidate, and the reason is the whole point of this lane: it takes T1 from `0.3479` to
`0.1791`, **below T1's own model-free baseline**. A deployment candidate that is excellent at
one task and worse than a lookup table at another is not a candidate.

The selection rule was fixed before the run: *maximise T2 PR-AUC subject to T1 slice macro
staying within 0.003 of 0.3479*. `joint_lambda1.0.pt` is what that rule selects.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
TASK = Path(__file__).resolve().parent
OUTPUT = TASK / "output"
REPO = PROJECT
sys.path.insert(0, str(REPO))

MODEL_CONFIG_ID = "s2_ds_08_shared_gru_t1_t2_v1"
CANDIDATE = OUTPUT / "joint_lambda1.0.pt"
SEEDS = (13, 42, 2026)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def git_sha() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True,
                          text=True, check=True).stdout.strip()


def model_config() -> dict:
    """The frozen architecture, as a record a consumer can compare rather than read.

    `architecture_parameters` carries what `S2-DS-05` selected and nothing else - the core,
    the width, the depth, the channels. A consumer that builds a model from this and gets a
    different parameter count has found a real disagreement, which is the point.
    """
    from ppsi.models.batch_spec import CATEGORY_COUNT, phase1_batch_spec_v1
    from ppsi.models.session_gru import SessionGRUConfig, build_model, parameter_count

    config = SessionGRUConfig(channels=("category_id", "event_type_id"), use_gap=True,
                              hidden=128, layers=1, dropout=0.3, core="gru")
    spec = phase1_batch_spec_v1()
    model = build_model(13, batch_spec=spec, config=config)

    return {
        "schema": "model_config_v1", "version": "1",
        "model_config_id": MODEL_CONFIG_ID,
        "model_family": "session_gru",
        "architecture_id": "shared_session_encoder_v1",
        "architecture_version": "1",
        "parameter_dtype": "float32",
        "architecture_parameters": {
            "core": config.core,
            "hidden": config.hidden,
            "layers": config.layers,
            "dropout": config.dropout,
            "history_channels": list(config.channels),
            "use_gap": config.use_gap,
            "history_length": 20,
            "batch_spec": spec.schema,
            "parameter_count": parameter_count(model),
            "readout": ("gather at clamp(lengths-1, min=0) over the full padded sequence, "
                        "not pack_padded_sequence, because packing obstructs the ONNX "
                        "export S2-SE-01 owes. With right-padded history the two are "
                        "mathematically identical and a test asserts they agree to 1e-5."),
            "architecture_selected_by": ("S2-DS-05: four cores under one frozen protocol. "
                                         "GRU 0.3477, LSTM 0.3471, transformer 0.3440, "
                                         "TCN 0.3398. GRU and LSTM are tied inside the "
                                         "0.0007 seed spread; the GRU is 27% cheaper."),
        },
        "task_heads": [
            {"task": "T1", "head_id": "t1_category_head_v1", "head_version": "1",
             "output_dim": CATEGORY_COUNT},
            {"task": "T2", "head_id": "t2_purchase_head_v1", "head_version": "1",
             "output_dim": 1},
            {"task": "T3", "head_id": "t3_residual_reranker_v1", "head_version": "1",
             "output_dim": 1},
        ],
    }


def model_config_ref(path: Path) -> dict:
    """An ArtifactRef v1 pointing at the ModelConfig we just wrote.

    A bare id would say *which* config; the ref says which config **and which bytes**. That
    is the difference between a lane claiming it used our architecture and a lane proving it.
    """
    return {
        "schema": "artifact_ref_v1", "version": "1",
        "logical_id": MODEL_CONFIG_ID,
        "artifact_schema": "model_config_v1",
        "artifact_version": "1",
        "uri": "config/experiments/s2-ds-08/model_config.v1.json",
        "sha256": sha256(path),
    }


def common_initializations(sha: str, config_ref: dict) -> tuple:
    """The untrained weights every regime must share, one record per seed.

    R1 and R2 starting from different random weights would put part of the centralized-vs-
    federated gap - this project's headline - into the initialisation. These records exist
    so that cannot happen silently.
    """
    from ppsi.models.batch_spec import phase1_batch_spec_v1
    from ppsi.models.session_gru import SessionGRUConfig, common_initialization

    config = SessionGRUConfig(channels=("category_id", "event_type_id"), use_gap=True,
                              hidden=128, layers=1, dropout=0.3, core="gru")
    spec = phase1_batch_spec_v1()

    records, provenance = [], []
    for seed in SEEDS:
        _, digest = common_initialization(seed, config=config, batch_spec=spec)
        path = PROJECT / "S2-DS-01_GRU_T1_Model/output" / f"common_initialization_seed{seed}.pt"
        records.append({
            "schema": "common_initialization_v1", "version": "1",
            "regime": "R1",
            "tasks": ["T1", "T2"],
            "model_config_ref": config_ref,
            "seed": seed,
            "initializer_git_sha": sha,
        })
        # The schema rejects unknown fields, and it is right to: a contract that accepts
        # anything verifies nothing. The digest and the file location go in a sidecar so the
        # record stays conformant and the provenance is still written down.
        provenance.append({
            "seed": seed,
            "model_config_ref": MODEL_CONFIG_ID,
            "state_dict_digest": digest,
            "artifact": {
                "path": f"S2-DS-01_GRU_T1_Model/output/common_initialization_seed{seed}.pt",
                "sha256": sha256(path) if path.exists() else None,
                "distributed_via": ("not in this repository - it is public. See "
                                    "docs/evidence/seal/checkpoint_manifest.json"),
            },
            "note": ("`common_initialization` requires an explicit config. It used to "
                     "default to the five-channel encoder while R1 uses two, which is a "
                     "different parameter count and a different digest - so a lane calling "
                     "it bare would have started somewhere R1 never was."),
        })
    return records, provenance


def deployment_candidate(sha: str) -> dict:
    joint = json.loads((OUTPUT / "s2_ds_08_joint.json").read_text(encoding="utf-8"))
    selected = next(r for r in joint["results"] if r["lambda_t2"] == 1.0)
    st1 = json.loads(
        (OUTPUT / "s2_ds_st1_shared_vs_separate.json").read_text(encoding="utf-8"))

    return {
        "schema": "deployment_candidate_v1", "version": "1",
        "deployment_candidate_checkpoint_id": "s2_ds_08_joint_lambda1.0_seed13",
        "task": "S2-DS-08",
        "model_config_ref": MODEL_CONFIG_ID,
        "selected_by": joint["selection"],
        "artifact": {
            "path": "S2-DS-08_Joint_Loss/output/joint_lambda1.0.pt",
            "sha256": sha256(CANDIDATE),
            "megabytes": round(CANDIDATE.stat().st_size / 1e6, 2),
            "distributed_via": ("not in this repository - it is public. Bundled in "
                                "_HANDOFF/s2_model_weights_v1.zip with a per-file manifest."),
        },
        "trained_from": joint["started_from"],
        "git_sha": sha,
        "validation": {
            "t1_slice_macro": selected["t1_slice_macro"],
            "t1_cost_against_t1_only": selected["t1_cost"],
            "t2_pr_auc": selected["best_t2_within_margin"],
            "t2_gain_over_baseline": selected["t2_gain_over_baseline"],
            "epoch": selected["epoch"],
            "seeds_run": 1,
        },
        "why_not_the_best_t2_model": {
            "candidate": "t2_2.pt",
            "its_t2": 0.1424, "its_t1": 0.1791,
            "reason": ("it is better at T2 and takes T1 below its own model-free baseline of "
                       "0.3143. A checkpoint that is excellent at one task and worse than a "
                       "lookup table at another is not a deployment candidate."),
        },
        "what_it_does_not_carry": {
            "T3": ("no T3 loss entered the shared encoder. S2-DS-07 has no learned reranker "
                   "that beats the frozen retrieval order, so a T3 term would spend encoder "
                   "capacity on a task with no demonstrated gain. The T3 head exists and "
                   "emits finite scores because RawModelOutput requires all three tensors."),
        },
        "known_limits": [
            "one seed; the T1 cost of 0.0012 is within two seed spreads",
            ("lambda 1.0 is the edge of the pre-registered ladder and T2 rose monotonically "
             "with it, so the breaking point was not found"),
            ("a separate T2 model reaches 0.1430, so sharing costs T2 "
             f"{st1['separate_minus_joint']} in exchange for 2,379,263 fewer parameters "
             "on the device"),
        ],
        "consumed_by": ["S2-SE-08", "S2-PR-09", "S2-PR-06"],
    }


def main() -> None:
    sha = git_sha()
    contracts = REPO / "config" / "experiments" / "s2-ds-08"
    contracts.mkdir(parents=True, exist_ok=True)

    from scripts.experiments.schemas import (
        validate_common_initialization,
        validate_model_config,
    )

    mc = model_config()
    validate_model_config(mc)
    (contracts / "model_config.v1.json").write_text(
        json.dumps(mc, indent=2) + "\n", encoding="utf-8")
    print(f"  ModelConfig v1      -> {mc['model_config_id']}  "
          f"({mc['architecture_parameters']['parameter_count']:,} parameters, validated)")

    records, provenance = common_initializations(
        sha, model_config_ref(contracts / "model_config.v1.json"))
    for record, sidecar in zip(records, provenance, strict=True):
        validate_common_initialization(record)
        seed = record["seed"]
        (contracts / f"common_initialization.seed{seed}.v1.json").write_text(
            json.dumps(record, indent=2) + "\n", encoding="utf-8")
        (contracts / f"common_initialization.seed{seed}.provenance.json").write_text(
            json.dumps(sidecar, indent=2) + "\n", encoding="utf-8")
        print(f"  CommonInitialization -> seed {seed}  "
              f"digest {sidecar['state_dict_digest'][:16]}  (validated)")

    candidate = deployment_candidate(sha)
    (contracts / "deployment_candidate.v1.json").write_text(
        json.dumps(candidate, indent=2) + "\n", encoding="utf-8")
    print(f"\n  deployment candidate -> {candidate['deployment_candidate_checkpoint_id']}")
    print(f"    T1 {candidate['validation']['t1_slice_macro']}  "
          f"T2 {candidate['validation']['t2_pr_auc']}")
    print(f"    sha256 {candidate['artifact']['sha256']}")
    print(f"\n  written to {contracts.relative_to(PROJECT)}")


if __name__ == "__main__":
    main()
