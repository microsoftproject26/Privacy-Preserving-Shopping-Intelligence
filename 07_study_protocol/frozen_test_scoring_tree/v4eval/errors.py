"""v4eval.errors — every refusal the V4 evaluator can raise (EVALUATION_CONTRACT_V4 R2)."""
from __future__ import annotations


class V4EvalError(Exception):
    """Base class: every V4 evaluator refusal."""


class NonFiniteScoreError(V4EvalError, ValueError):
    """A NaN or +-inf score, query or head value (contract G1). The legacy evaluator silently ranks a NaN target 1."""


class NondeterministicScoreError(V4EvalError):
    """The same score block recomputed to different bits (the target score did not reproduce in pass 2)."""


class SampledEvaluationRefused(V4EvalError):
    """Sampled negatives, candidate subsets, top-k or a catalogue other than the full K (contract E6.2, G2; T37)."""


class ManifestError(V4EvalError, ValueError):
    """A manifest violates MANIFEST_SPEC_V4 (forbidden column, order, mask/censor/OOV consistency, hash mismatch)."""


class LockGateError(V4EvalError, PermissionError):
    """A RETRO_TEST manifest without a matching ALL_METHODS_LOCK (contract E10.1; T48)."""


class AlignmentError(V4EvalError):
    """Predictions do not align with the manifest by decision_id and ordered manifest hash (contract G4, E6.3)."""


class StrataError(V4EvalError):
    """A stratum requested from anything but an allowed manifest field (contract E3.5; client_idx is refused)."""


class PairingError(V4EvalError):
    """Paired statistics over different user sets or different resample matrices (contract E7.2; T38)."""


class IdentityError(V4EvalError):
    """The micro or macro coverage identity failed on a result (contract E4.6; T36)."""


class RowKeyError(V4EvalError, ValueError):
    """A result row lacks a comparison-key field (contract §9)."""


class SchemaKeyError(V4EvalError, KeyError):
    """A report looked up a key the result schema does not define (T49)."""
