"""Every refusal the full-catalogue evaluator can raise."""
from __future__ import annotations


class EvaluationError(Exception):
    """Base class of every evaluator refusal."""


class NonFiniteScoreError(EvaluationError, ValueError):
    """A NaN or +-inf score, query or head value. A naive comparison-based ranker silently ranks a NaN target 1."""


class NondeterministicScoreError(EvaluationError):
    """The same score block recomputed to different bits (the target score did not reproduce in the second pass)."""


class SampledEvaluationRefused(EvaluationError):
    """Sampled negatives, candidate subsets, top-k or a catalogue other than the full one."""


class EvalConfigError(EvaluationError):
    """A caller override of the evaluation configuration, an artifact produced under another configuration, or
    runtime flags that differ from the configured ones."""


class ManifestError(EvaluationError, ValueError):
    """A manifest is invalid (forbidden column, order, mask / censor / OOV consistency, hash mismatch)."""


class AlignmentError(EvaluationError):
    """Predictions do not align with the manifest by decision_id and ordered manifest hash."""


class StrataError(EvaluationError):
    """A stratum requested from anything but an allowed manifest field (a client index is refused)."""


class PairingError(EvaluationError):
    """Paired statistics over different user sets or different resample matrices."""


class IdentityError(EvaluationError):
    """The micro or macro coverage identity failed on a result."""


class RowKeyError(EvaluationError, ValueError):
    """A result row lacks a comparison-key field."""


class SchemaKeyError(EvaluationError, KeyError):
    """A report looked up a key the result schema does not define."""
