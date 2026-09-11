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


def load_encoder(model, checkpoint: Path, *, new_prefixes: tuple = ()) -> dict:
    """Load `checkpoint` into `model`, and say exactly what did and did not come across.

    `new_prefixes` names the parameters this task is entitled to start fresh - `t2_head`
    and `query_` for T2, for instance. Anything else missing is an error.
    """
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    saved = payload["state_dict"]
    shapes = {key: value.shape for key, value in model.state_dict().items()}

    usable = {k: v for k, v in saved.items() if k in shapes and v.shape == shapes[k]}
    dropped = sorted(set(saved) - set(usable))

    reshaped_encoder = [k for k in dropped if not k.startswith(T3_HEAD_PREFIXES)]
    assert not reshaped_encoder, (
        f"these checkpoint parameters do not fit the current model: {reshaped_encoder[:5]}. "
        "They are encoder parameters, so loading around them would train on a randomly "
        "initialised encoder while reporting the number as if it came from S2-DS-01.")

    missing, unexpected = model.load_state_dict(usable, strict=False)
    assert not unexpected, f"unexpected keys: {unexpected[:5]}"

    unloaded = [k for k in missing
                if not k.startswith(new_prefixes + T3_HEAD_PREFIXES)]
    assert not unloaded, f"the encoder did not load cleanly: {unloaded[:5]}"

    spec = payload.get("batch_spec")
    return {"checkpoint": checkpoint.name, "seed": payload.get("seed"),
            "loaded": len(usable), "dropped": dropped,
            "new_parameters": sorted(missing),
            # None means the checkpoint predates spec stamping. A checkpoint that does not
            # record the spec it was built under is why the widened candidate projection
            # surfaced as a size mismatch mid-run rather than as a clear message.
            "batch_spec": spec}


def spec_fingerprint(spec) -> dict:
    """The batch-spec facts a checkpoint must record to be loadable later without guessing.

    Every widened channel is a silent break: the tensors still load with `strict=False`,
    the shapes still validate, and the only symptom is a mid-run size mismatch - or worse,
    no symptom at all. Stamping the spec turns that into a comparison anyone can make.
    """
    channels = lambda group: {c.name: [c.pad_id, c.vocab_size]  # noqa: E731
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


def save(path, *, model, spec, **facts) -> dict:
    """Write a checkpoint that carries the spec it was built under."""
    payload = {"state_dict": model.state_dict(),
               "batch_spec": spec_fingerprint(spec), **facts}
    torch.save(payload, path)
    return payload["batch_spec"]
