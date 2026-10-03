"""File helpers and the scoring-module factory used by the export, quantise and bench commands."""
from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import torch

from ppsi.seqrec.build import build as build_model
from ppsi.seqrec.features import InputWidths

from .scoring import ScoringModule

DEFAULT_WIDTHS = (15, 15, 12, 4)


class CheckpointIOError(ValueError):
    pass


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json_atomic(path, obj) -> str:
    """Write strict JSON (sorted keys, no NaN) to `path` through a temp file and an atomic rename; returns the
    file's sha256."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(obj, indent=1, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as fh:
        fh.write(data)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)
    return hashlib.sha256(data).hexdigest()


def parse_widths(text: str) -> InputWidths:
    """"numeric,flags,user_context,user_mask" -> InputWidths."""
    parts = [int(x) for x in str(text).split(",")]
    if len(parts) != 4:
        raise ValueError("widths must be four integers: numeric,flags,user_context,user_mask")
    return InputWidths(*parts)


def load_state_dict(ckpt_path) -> tuple:
    """(state_dict, meta) from a checkpoint file: a bare state dict, or a record {"model": state_dict, "meta": {...}}.
    `meta` is `{}` for a bare state dict."""
    obj = torch.load(str(ckpt_path), map_location="cpu", weights_only=True)
    if isinstance(obj, Mapping) and "model" in obj and isinstance(obj["model"], Mapping):
        return dict(obj["model"]), dict(obj.get("meta") or {})
    if isinstance(obj, Mapping):
        return dict(obj), {}
    raise CheckpointIOError(f"{ckpt_path}: not a checkpoint record or a state dict (got {type(obj).__name__})")


def build_scoring_module(family: str, *, widths: InputWidths | None = None, variant: str | None = None,
                         ckpt_path=None, seed: int = 2026, k: int | None = None, poc: np.ndarray | None = None,
                         catalogue: str | None = None) -> dict:
    """Build the family/variant module (fresh init if `ckpt_path` is None, else the checkpoint's weights loaded
    strictly on top of it) and wrap it as a `ScoringModule` in eval mode.

    Class map (ppsi.seqrec.build's rule): exactly one of `k` (arange(3, 3+k)), `poc` or `catalogue`.
    SASRec's `impl` is set to "padded_reference": `impl` picks one of two numerically equivalent forward paths over
    the SAME parameters, so switching it after loading is always safe; the padded, mask-based path is the
    ONNX-traceable one (the default "unpadded" path gathers valid tokens with a data-dependent index).

    Returns {"scoring", "built", "checkpoint_meta", "state_dict_keys"}."""
    widths = widths if widths is not None else InputWidths(*DEFAULT_WIDTHS)
    built = build_model(family, widths, seed, "cpu", K=k, poc=poc, catalogue=catalogue, variant=variant)
    meta: dict = {}
    if ckpt_path is not None:
        state, meta = load_state_dict(ckpt_path)
        try:
            built.module.load_state_dict(state, strict=True)
        except RuntimeError as e:
            raise CheckpointIOError(f"checkpoint state dict does not match the {built.family} module "
                                    f"(variant {built.variant}): {e}") from e
    if str(built.family).upper() == "SASREC":
        built.module.impl = "padded_reference"
    built.module.eval()
    return {"scoring": ScoringModule(built), "built": built, "checkpoint_meta": meta,
            "state_dict_keys": len(list(built.module.state_dict()))}


__all__ = ["DEFAULT_WIDTHS", "CheckpointIOError", "build_scoring_module", "load_state_dict", "parse_widths",
           "sha256_file", "write_json_atomic"]
