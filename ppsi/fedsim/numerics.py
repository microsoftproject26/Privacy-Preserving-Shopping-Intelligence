"""Strict FP32, determinism flags, stable seed derivation and state digests.

Strict FP32: torch.backends.cuda.matmul.allow_tf32 = False and torch.backends.cudnn.allow_tf32 = False, float32
matmul precision "highest". Every client visit and every local-only run calls
`assert_strict_fp32()`, so a caller that flips TF32 on is refused rather than silently changing the numerics.

Seeds are derived with BLAKE2b from explicit parts (never Python's salted `hash`), so a visit's RNG depends only on
(base seed, stream name, round, logical client key, ...) and not on the worker, the process, the arrival order or a
retry count. The derivation deliberately excludes the method name, so FA and FP(mu = 0) draw identical streams.
"""
from __future__ import annotations

import contextlib
import hashlib
from collections.abc import Iterator, Mapping

import torch
from torch import Tensor


class NumericsError(RuntimeError):
    pass


def enforce_strict_fp32(deterministic: bool = True, warn_only: bool = True) -> dict:
    """Set the strict-FP32 flags (both TF32 switches off) and, optionally, deterministic algorithms.

    warn_only=False is the strict mode used for GPU determinism checks: an op without a deterministic implementation
    then raises instead of warning (CUBLAS_WORKSPACE_CONFIG must be set before CUDA initializes)."""
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=warn_only)
    return numerics_record()


def assert_strict_fp32() -> None:
    bad = []
    if torch.backends.cuda.matmul.allow_tf32:
        bad.append("torch.backends.cuda.matmul.allow_tf32=True")
    if torch.backends.cudnn.allow_tf32:
        bad.append("torch.backends.cudnn.allow_tf32=True")
    if torch.get_float32_matmul_precision() != "highest":
        bad.append(f"float32_matmul_precision={torch.get_float32_matmul_precision()!r}")
    if bad:
        raise NumericsError("strict FP32 required: " + ", ".join(bad))


def numerics_record() -> dict:
    return {"torch": torch.__version__,
            "cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
            "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
            "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
            "deterministic_warn_only": bool(torch.is_deterministic_algorithms_warn_only_enabled()),
            "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
            "intra_op_threads": int(torch.get_num_threads())}


def derive_seed(*parts) -> int:
    """A 63-bit seed from explicit, typed parts (stable across processes, platforms and Python hash salts)."""
    enc = "\x1f".join(f"{type(p).__name__}:{p}" for p in parts).encode("utf-8")
    return int.from_bytes(hashlib.blake2b(enc, digest_size=8).digest(), "big") & ((1 << 63) - 1)


def generator(seed: int, device: str | torch.device = "cpu") -> torch.Generator:
    return torch.Generator(device=device).manual_seed(int(seed))


@contextlib.contextmanager
def seeded_global_rng(seed: int, device: torch.device) -> Iterator[None]:
    """Fork the global RNG (CPU, plus the CUDA device if used), seed it, restore on exit.

    Dropout inside the models draws from the global RNG, so this is how a visit gets an RNG that depends only
    on its derived seed. The global CPU RNG is process-wide: two visits must never run concurrently in threads of one
    process (the pool uses processes for that reason)."""
    devices = [device] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        # Seed exactly the generators in use. (torch.manual_seed would also queue lazy CUDA/XPU seeding, which on a
        # CPU-only build formats a Python stack trace on every call: ~0.3 ms per visit in a profile.)
        torch.default_generator.manual_seed(int(seed))
        if device.type == "cuda":
            torch.cuda.default_generators[device.index if device.index is not None else torch.cuda.current_device()] \
                .manual_seed(int(seed))
        yield


def tensor_bytes(t: Tensor) -> bytes:
    return t.detach().to("cpu").contiguous().reshape(-1).view(torch.uint8).numpy().tobytes() if t.numel() else b""


def state_digest(state: Mapping[str, Tensor]) -> str:
    """sha256 over (key, dtype, shape, raw bytes) in sorted key order: a bit-exact checksum of a state."""
    h = hashlib.sha256()
    for k in sorted(state):
        t = state[k]
        h.update(k.encode())
        h.update(str(t.dtype).encode())
        h.update(str(tuple(t.shape)).encode())
        h.update(tensor_bytes(t))
    return h.hexdigest()


def max_abs_diff(a: Mapping[str, Tensor], b: Mapping[str, Tensor]) -> float:
    if set(a) != set(b):
        raise KeyError(f"key sets differ: {sorted(set(a) ^ set(b))}")
    m = 0.0
    for k in a:
        if a[k].is_floating_point():
            m = max(m, float((a[k].double() - b[k].double()).abs().max()) if a[k].numel() else 0.0)
        elif not torch.equal(a[k], b[k]):
            return float("inf")
    return m


def states_equal(a: Mapping[str, Tensor], b: Mapping[str, Tensor]) -> bool:
    """Bitwise equality (same keys, dtypes, shapes and bytes)."""
    if set(a) != set(b):
        return False
    return all(a[k].dtype == b[k].dtype and a[k].shape == b[k].shape and tensor_bytes(a[k]) == tensor_bytes(b[k])
               for k in a)


_FP_CHUNK = 1 << 23


@torch.no_grad()
def fingerprint(state: Mapping[str, Tensor]) -> str:
    """A bit-exact, on-device state fingerprint (for GPU runs, where a CPU sha256 of a large state per client is slow).

    Each element's raw bit pattern is split into two 16-bit halves (lo, hi); per tensor the exact int64 sums
    sum(lo), sum(hi), sum(lo * w1), sum(hi * w2) with position weights w in [1, 65521] are computed in chunks
    (no overflow: < 2^59 for 2^27 elements), then sha256-hashed on the host with key, dtype and shape. Integer
    addition is associative, so the result does not depend on reduction order. Any single-bit change alters sum(lo)
    or sum(hi); it is a determinism check, not a cryptographic hash."""
    h = hashlib.sha256()
    for k in sorted(state):
        t = state[k].detach().contiguous().reshape(-1)
        if t.dtype in (torch.float32, torch.int32):
            bits = t.view(torch.int32)
        elif t.dtype == torch.int64:
            bits = None
        else:
            bits = t.to(torch.int32)
        sums = []
        n = t.numel()
        for c0 in range(0, n, _FP_CHUNK):
            if bits is None:                              # int64: fingerprint the two 32-bit words
                seg = t[c0:c0 + _FP_CHUNK]
                parts = (seg & 0xFFFF, (seg >> 16) & 0xFFFF, (seg >> 32) & 0xFFFF, (seg >> 48) & 0xFFFF)
            else:
                seg = bits[c0:c0 + _FP_CHUNK].to(torch.int64)
                parts = (seg & 0xFFFF, (seg >> 16) & 0xFFFF)
            idx = torch.arange(c0, c0 + seg.numel(), device=seg.device, dtype=torch.int64)
            for j, part in enumerate(parts):
                w = (idx % (65521 - 2 * j)) + 1
                sums.append(part.sum())
                sums.append((part * w).sum())
        vals = [int(v) for v in torch.stack(sums).cpu().tolist()] if sums else []
        h.update(k.encode())
        h.update(str(t.dtype).encode())
        h.update(str(tuple(state[k].shape)).encode())
        h.update(json_ints(vals))
    return h.hexdigest()


def json_ints(vals) -> bytes:
    return (",".join(str(v) for v in vals)).encode()
