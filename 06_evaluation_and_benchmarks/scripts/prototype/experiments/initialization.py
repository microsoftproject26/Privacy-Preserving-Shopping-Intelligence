"""Common Initialization generation and serialization (S1-PR-03).

Implements the frozen serialization contract for R1-R4 shared initializations.
Generates deterministic tensors using `torch.random.fork_rng`.
"""

from __future__ import annotations

import contextlib
import struct
from collections.abc import Iterator
from typing import Any

import torch

from scripts.experiments.contracts import (
    ALLOWED_SEEDS,
    ContractValidationError,
    canonical_json_bytes,
)

MAGIC_BYTES = b"PPSI_COMMON_INIT_V1\0"
FORMAT_VERSION = 1


@contextlib.contextmanager
def deterministic_rng(seed: int) -> Iterator[None]:
    """Context manager to deterministically generate random numbers without
    affecting the caller's RNG state. Uses fork_rng across all devices."""
    if seed not in ALLOWED_SEEDS:
        raise ContractValidationError(f"Seed {seed} not in {ALLOWED_SEEDS}")

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        yield


def generate_fixture_linear_v1_state(config: dict[str, Any], seed: int) -> dict[str, torch.Tensor]:
    """Generate state_dict for fixture_linear_v1 architecture."""
    if config["architecture_id"] != "fixture_linear_v1":
        raise ContractValidationError("Only fixture_linear_v1 is supported")
    if config["parameter_dtype"] != "float32":
        raise ContractValidationError("Only float32 is supported")

    params = config["architecture_parameters"]
    in_dim = params["input_dim"]
    out_dim = params["output_dim"]
    use_bias = params["bias"]

    state_dict = {}

    with deterministic_rng(seed):
        # Shared backbone
        state_dict["backbone.weight"] = torch.randn(out_dim, in_dim, dtype=torch.float32)
        if use_bias:
            state_dict["backbone.bias"] = torch.randn(out_dim, dtype=torch.float32)

        # Task heads
        for head in config["task_heads"]:
            h_out = head["output_dim"]
            task = head["task"]
            state_dict[f"heads.{task}.weight"] = torch.randn(h_out, out_dim, dtype=torch.float32)
            if use_bias:
                state_dict[f"heads.{task}.bias"] = torch.randn(h_out, dtype=torch.float32)

    return state_dict


def serialize_state_dict(state_dict: dict[str, torch.Tensor]) -> bytes:
    """Serialize a state_dict according to the frozen PPSI_COMMON_INIT_V1 specification."""
    if not state_dict:
        raise ContractValidationError("State dict is empty")

    out = bytearray()
    out.extend(MAGIC_BYTES)
    out.extend(struct.pack(">H", FORMAT_VERSION))
    out.extend(struct.pack(">I", len(state_dict)))

    for key in sorted(state_dict.keys()):
        if not isinstance(key, str):
            raise ContractValidationError("String keys only")

        tensor = state_dict[key]
        if tensor.is_sparse or tensor.is_quantized or tensor.is_meta:
            raise ContractValidationError("Reject sparse/quantized/meta tensors")

        tensor = tensor.detach().cpu().contiguous()
        if not tensor.is_floating_point():
            raise ContractValidationError("Floating tensors only")
        if not torch.all(torch.isfinite(tensor)):
            raise ContractValidationError("Floating tensors must be finite")

        if tensor.dtype != torch.float32:
            raise ContractValidationError(
                "Explicit supported dtype registry (only float32 currently)"
            )

        payload = tensor.numpy().tobytes("C")

        # We need little-endian payload for floats
        # NumPy default byteorder depends on system, but .tobytes("C") gives contiguous bytes.
        # Ensure it's little-endian:
        if tensor.numpy().dtype.byteorder == ">" or (
            tensor.numpy().dtype.byteorder == "=" and sys.byteorder == "big"
        ):
            payload = tensor.numpy().byteswap().tobytes("C")

        metadata = {
            "key": key,
            "dtype": "float32",
            "shape": list(tensor.shape),
            "numel": tensor.numel(),
            "nbytes": len(payload),
            "byte_order": "little",
            "layout": "dense_strided_c_contiguous",
        }
        meta_bytes = canonical_json_bytes(metadata)

        out.extend(struct.pack(">I", len(meta_bytes)))
        out.extend(meta_bytes)
        out.extend(struct.pack(">Q", len(payload)))
        out.extend(payload)

    return bytes(out)


def deserialize_state_dict(data: bytes) -> dict[str, torch.Tensor]:
    """Deserialize a state_dict from the frozen PPSI_COMMON_INIT_V1 specification."""
    if not data.startswith(MAGIC_BYTES):
        raise ContractValidationError("Invalid magic bytes")

    offset = len(MAGIC_BYTES)
    fmt_ver = struct.unpack_from(">H", data, offset)[0]
    offset += 2
    if fmt_ver != FORMAT_VERSION:
        raise ContractValidationError(f"Unsupported format version {fmt_ver}")

    tensor_count = struct.unpack_from(">I", data, offset)[0]
    offset += 4

    import json

    state_dict = {}
    for _ in range(tensor_count):
        meta_len = struct.unpack_from(">I", data, offset)[0]
        offset += 4

        meta_bytes = data[offset : offset + meta_len]
        offset += meta_len

        payload_len = struct.unpack_from(">Q", data, offset)[0]
        offset += 8

        payload = data[offset : offset + payload_len]
        offset += payload_len

        meta = json.loads(meta_bytes)
        key = meta["key"]

        # Reconstruct tensor (little-endian float32)
        import numpy as np

        arr = np.frombuffer(payload, dtype=np.dtype("<f4")).reshape(meta["shape"])
        tensor = torch.from_numpy(arr.copy())  # copy to make it writable and own memory
        state_dict[key] = tensor

    if offset != len(data):
        raise ContractValidationError("Trailing bytes found after parsing all tensors")

    return state_dict


import sys
