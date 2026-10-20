"""Loading an `S2-DS-01` encoder into a head task - one implementation, not three.

`S2-DS-06` and `S2-DS-07` each grew their own version of this, and they diverged. The T3
one learned to tolerate a reshaped T3 head when the retrieval-rank channel widened
`candidate_projection` from `[128, 40]` to `[128, 41]`; the T2 one did not, and it stopped
loading the very checkpoint the task is built on. Two copies of one rule is the defect that
has already cost this project three wrong numbers, and here it cost a failed run.

The rule, in one place:

* a tensor whose shape matches is loaded;
* a tensor whose shape does not match is dropped **only** if it belongs to the T3 head,
  which no head task inherits anything from - it existed in `S2-DS-01` solely because
  `RawModelOutput` requires all three tensors, so those weights were random there too;
* a reshaped **encoder** tensor is an error, never a drop. Silently replacing the encoder
  with random weights would still produce a number, and that number would be reported as a
  modelling result.
"""

from __future__ import annotations

from pathlib import Path

import torch

T3_HEAD_PREFIXES = ("candidate_", "t3")


class CheckpointContractError(RuntimeError):
    """A checkpoint does not match the contract the current model was built under.

    An exception and not an `assert`. Every guard in this module used to be an assertion,
    and `python -O` removes assertions entirely - so the one safeguard against loading a
    checkpoint whose category codes mean something different would have vanished under the
    flag most likely to be used for a long federated run.
    """


def load_encoder(model, checkpoint: Path, *, new_prefixes: tuple = (),
                 spec=None, allow_unstamped: bool = False) -> dict:
    """Load `checkpoint` into `model`, and say exactly what did and did not come across.

    `new_prefixes` names the parameters this task is entitled to start fresh - `t2_head`
    and `query_` for T2, for instance. Anything else missing is an error.

    `spec` is the batch spec the caller intends to run under. When given, the checkpoint's
    stamped spec must match it. **This comparison is the point of stamping**, and for a
    while the stamp was written and then never read: a checkpoint whose category codes were
    fitted on different data has the same tensor shapes as one fitted on ours, so it loads
    silently, passes every shape check, and produces numbers that mean nothing. Shape
    equality is not contract equality.

    `allow_unstamped` admits a checkpoint written before stamping existed. It returns
    `stamp_checked: False` so a caller can record that the guarantee was not available,
    rather than leaving the absence looking like a pass.
    """
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    saved = payload["state_dict"]
    stored_spec = payload.get("batch_spec")

    if spec is not None:
        expected = spec_fingerprint(spec)
        if stored_spec is None:
            if not allow_unstamped:
                raise CheckpointContractError(
                    f"{checkpoint.name} carries no batch spec, so there is no way to tell "
                    "whether its category and item codes mean what this model expects. "
                    "Pass allow_unstamped=True to accept that risk deliberately, and "
                    "record it.")
        elif stored_spec != expected:
            differing = sorted(
                key for key in set(stored_spec) | set(expected)
                if stored_spec.get(key) != expected.get(key))
            raise CheckpointContractError(
                f"{checkpoint.name} was built under a different batch spec; these differ: "
                f"{differing}. Loading it would give tensors the right shapes and the "
                "wrong meaning.")

    shapes = {key: value.shape for key, value in model.state_dict().items()}
    usable = {k: v for k, v in saved.items() if k in shapes and v.shape == shapes[k]}
    dropped = sorted(set(saved) - set(usable))

    reshaped_encoder = [k for k in dropped if not k.startswith(T3_HEAD_PREFIXES)]
    if reshaped_encoder:
        raise CheckpointContractError(
            f"these checkpoint parameters do not fit the current model: "
            f"{reshaped_encoder[:5]}. They are encoder parameters, so loading around them "
            "would train on a randomly initialised encoder while reporting the number as "
            "if it came from S2-DS-01.")

    missing, unexpected = model.load_state_dict(usable, strict=False)
    if unexpected:
        raise CheckpointContractError(f"unexpected keys: {unexpected[:5]}")

    unloaded = [k for k in missing
                if not k.startswith(new_prefixes + T3_HEAD_PREFIXES)]
    if unloaded:
        raise CheckpointContractError(
            f"the encoder did not load cleanly: {unloaded[:5]}")

    return {"checkpoint": checkpoint.name, "seed": payload.get("seed"),
            "loaded": len(usable), "dropped": dropped,
            "new_parameters": sorted(missing),
            # None means the checkpoint predates spec stamping. A checkpoint that does not
            # record the spec it was built under is why the widened candidate projection
            # surfaced as a size mismatch mid-run rather than as a clear message.
            "batch_spec": stored_spec,
            "artifact_digests": payload.get("artifact_digests"),
            "stamp_checked": bool(spec is not None and stored_spec is not None)}


def spec_fingerprint(spec) -> dict:
    """The batch-spec facts a checkpoint must record to be loadable later without guessing.

    Every widened channel is a silent break: the tensors still load with `strict=False`,
    the shapes still validate, and the only symptom is a mid-run size mismatch - or worse,
    no symptom at all. Stamping the spec turns that into a comparison anyone can make.
    """
    channels = lambda group: {c.name: [c.pad_id, c.vocab_size]
                              for c in getattr(spec, group, ())}
    return {
        "schema": spec.schema, "version": spec.version,
        "history_continuous_dim": spec.history_continuous_dim,
        "query_continuous_dim": spec.query_continuous_dim,
        "candidate_continuous_dim": spec.candidate_continuous_dim,
        "candidate_id_vocab_size": spec.candidate_id_vocab_size,
        "history_categorical": channels("history_categorical"),
        "query_categorical": channels("query_categorical"),
        "candidate_categorical": channels("candidate_categorical"),
    }


def save(path, *, model, spec, artifacts=None, **facts) -> dict:
    """Write a checkpoint that carries the spec and the artifacts it was built against.

    `artifacts` is the directory holding the frozen S1 outputs. Passing it stores a digest
    per artifact, which is what lets a later load prove the codes still mean the same thing
    rather than merely still having the same shape.
    """
    payload = {"state_dict": model.state_dict(),
               "batch_spec": spec_fingerprint(spec), **facts}
    if artifacts is not None:
        payload["artifact_digests"] = artifact_digests(artifacts)
    torch.save(payload, path)
    return {"batch_spec": payload["batch_spec"],
            "artifact_digests": payload.get("artifact_digests")}


# The frozen S1 artifacts every trained model's numbers are relative to. A checkpoint that
# does not name these cannot be shown to mean the same thing as another one.
FROZEN_ARTIFACTS = (
    "vocabulary_v1.proposed.json",
    "item_catalog_v1.proposed.parquet",
    "price_transform_v1.proposed.json",
    "t3_protocol_v1.proposed.json",
)


def artifact_digests(directory) -> dict:
    """sha256 of each frozen artifact the model's codes are defined by.

    `spec_fingerprint` records that the category vocabulary has 590 slots. It cannot record
    that slot 417 means the same category it meant last week. Refitting the vocabulary on
    different data produces a mapping of identical *shape* and different *meaning*, so every
    dimension check passes and every number silently changes its subject.

    A digest catches that, and nothing else does. It matters most in exactly the place it is
    hardest to notice: R1 and R2 averaging weights that were fitted against two different
    category orderings would still produce a model, still produce a metric, and still
    produce a regime gap - one that measured the vocabulary rather than federation.

    Missing files are recorded as `None` rather than skipped, so a digest set can never look
    complete when it is not.
    """
    import hashlib
    from pathlib import Path

    directory = Path(directory)
    digests = {}
    for name in FROZEN_ARTIFACTS:
        path = directory / name
        if not path.exists():
            digests[name] = None
            continue
        blake = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                blake.update(block)
        digests[name] = blake.hexdigest()
    return digests


def compare_digests(stored: dict | None, expected: dict) -> list:
    """Which frozen artifacts disagree. Empty means the two checkpoints share a meaning."""
    if stored is None:
        return ["<the checkpoint records no artifact digests>"]
    return sorted(name for name in set(stored) | set(expected)
                  if stored.get(name) != expected.get(name))
