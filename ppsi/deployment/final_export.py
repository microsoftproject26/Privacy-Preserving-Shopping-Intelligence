"""Export the deployment candidate named by S2-DS-08, and nothing else.

The final export must be of one specific checkpoint. The failure this guards against is
exporting something that merely looks right: an untrained initialisation, a search
checkpoint, or a model built from defaults rather than from the frozen configuration.

So the architecture is read from `config/experiments/s2-ds-08/model_config.v1.json` rather
than assumed, the parameter count is checked against what that file declares, and the
weights are verified by SHA-256 against `deployment_candidate.v1.json` before they are
loaded. A mismatch on any of those stops the export.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from ppsi.models.batch_spec import phase1_batch_spec_v1
from ppsi.models.session_gru import SessionGRU, SessionGRUConfig, build_model, parameter_count
from ppsi.training.batch import Phase1BatchSpec

REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_CONFIG_PATH = REPO_ROOT / "config" / "experiments" / "s2-ds-08" / "model_config.v1.json"
CANDIDATE_PATH = REPO_ROOT / "config" / "experiments" / "s2-ds-08" / "deployment_candidate.v1.json"


class DeploymentContractError(RuntimeError):
    """Raised when the thing about to be exported is not the approved candidate."""


@dataclass(frozen=True, slots=True)
class DeploymentCandidate:
    """The frozen identity of what Phase 1 ships."""

    checkpoint_id: str
    model_config_id: str
    weights_sha256: str
    weights_path: str
    declared_parameter_count: int
    history_length: int
    seed: int

    @property
    def architecture_summary(self) -> str:
        return f"{self.model_config_id} ({self.declared_parameter_count:,} parameters)"


def load_deployment_candidate(
    model_config_path: Path | str = MODEL_CONFIG_PATH,
    candidate_path: Path | str = CANDIDATE_PATH,
) -> tuple[DeploymentCandidate, SessionGRUConfig]:
    """Read the two frozen files and turn them into a model configuration.

    Building the configuration here rather than in the caller is the point: a caller that
    constructs `SessionGRUConfig` itself can drift from the frozen file without anything
    noticing, and the export would still succeed.
    """

    model_config = json.loads(Path(model_config_path).read_text(encoding="utf-8"))
    candidate = json.loads(Path(candidate_path).read_text(encoding="utf-8"))

    if candidate["model_config_ref"] != model_config["model_config_id"]:
        raise DeploymentContractError(
            f"the candidate names model config {candidate['model_config_ref']!r} but the "
            f"config file declares {model_config['model_config_id']!r}"
        )

    parameters = model_config["architecture_parameters"]
    seed = int(candidate["deployment_candidate_checkpoint_id"].rsplit("seed", 1)[-1])

    return (
        DeploymentCandidate(
            checkpoint_id=candidate["deployment_candidate_checkpoint_id"],
            model_config_id=model_config["model_config_id"],
            weights_sha256=candidate["artifact"]["sha256"],
            weights_path=candidate["artifact"]["path"],
            declared_parameter_count=int(parameters["parameter_count"]),
            history_length=int(parameters["history_length"]),
            seed=seed,
        ),
        SessionGRUConfig(
            channels=tuple(parameters["history_channels"]),
            use_gap=bool(parameters["use_gap"]),
            hidden=int(parameters["hidden"]),
            layers=int(parameters["layers"]),
            dropout=float(parameters["dropout"]),
            core=str(parameters["core"]),
        ),
    )


def file_sha256(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_deployment_model(
    candidate: DeploymentCandidate,
    config: SessionGRUConfig,
    *,
    weights: Path | str | None = None,
    batch_spec: Phase1BatchSpec | None = None,
) -> SessionGRU:
    """Build the approved architecture, and load the approved weights if they are present.

    The weights file is not in this repository, because the repository is public and the
    checkpoint is 9.53 MB of trained parameters. When it is supplied its SHA-256 must match
    the frozen contract; when it is absent the model carries the approved architecture with
    seed initialisation, which is enough to measure size, latency and parity but is
    explicitly not a deployment artifact.
    """

    spec = batch_spec or phase1_batch_spec_v1()
    model = build_model(candidate.seed, batch_spec=spec, config=config)

    counted = parameter_count(model)
    if counted != candidate.declared_parameter_count:
        raise DeploymentContractError(
            f"built {counted:,} parameters but {candidate.model_config_id} declares "
            f"{candidate.declared_parameter_count:,}; the frozen configuration and the "
            f"code have diverged"
        )

    if weights is not None:
        actual = file_sha256(weights)
        if actual != candidate.weights_sha256:
            raise DeploymentContractError(
                f"checkpoint SHA-256 is {actual} but {candidate.checkpoint_id} requires "
                f"{candidate.weights_sha256}; this is not the approved candidate"
            )
        state = torch.load(weights, map_location="cpu", weights_only=True)
        model.load_state_dict(state.get("model", state))

    return model.eval()


def candidate_identity(candidate: DeploymentCandidate, *, weights_loaded: bool) -> dict[str, Any]:
    """What the evidence file records about which model was measured."""

    return {
        "deployment_candidate_checkpoint_id": candidate.checkpoint_id,
        "model_config_id": candidate.model_config_id,
        "declared_parameter_count": candidate.declared_parameter_count,
        "history_length": candidate.history_length,
        "seed": candidate.seed,
        "weights_sha256_required": candidate.weights_sha256,
        "weights_loaded": weights_loaded,
        "weights_source": candidate.weights_path,
        "status": (
            "deployment artifact"
            if weights_loaded
            else "architecture only; trained weights were not supplied, so this measures the "
            "shipped architecture and not the shipped model"
        ),
    }
