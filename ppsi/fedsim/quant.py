"""The FA_Q8 upload codec: FedAvg with 8-bit stochastically rounded uploads.

FA exactly, except each surviving client's UPLOAD is its model delta quantised per tensor to int8:
  delta_k = upload_k - theta_r,k                    (every SHARED canonical tensor k, manifest order; FP32)
  scale_k = max|delta_k| / 127                      (FP32, sent as 4 bytes; an all-zero tensor has scale 0 -> q = 0)
  q_k     = clamp(floor(delta_k / scale_k + U), -127, 127) as int8,  U ~ Uniform[0, 1) elementwise
            (stochastic rounding: E[q_k] = delta_k / scale_k, so the dequantised delta is unbiased; |q*s - delta| < s)
  U is drawn from ONE torch.Generator per visit on the delta's device, seeded
  numerics.derive_seed(seed, "q8", round, client_key), tensors consumed in manifest shared-key order.
The server dequantises (theta_r,k + q_k * scale_k) and aggregates EXACTLY as FA (aggregate.py: n_consumed-weighted
FedAvg, the same shard order). Download stays dense FP32. Bytes per visit: up = the codec payload sum(numel) +
4 x n_tensors + the same 8-byte n_consumed header that FA's FP32 upload counts (comm.UPLOAD_HEADER_BYTES: the
server needs n_consumed for the FedAvg weight in both methods, so both counts carry it); down = the FP32 shared state.

Integration without touching server.py / pool.py / runtime.py: `Q8Channel` is a VirtualChannel whose `upload()` is
called by server.run_round with (round, client key, the client's upload dict) BEFORE the same dict is added to the
shard accumulator; it quantises the delta, replaces every entry of that dict by its dequantised value (the server-side
decode) and records the quantised byte count on the ledger. The driver contract this relies on (upload() before
acc.add() on the same object, in-process driver only) is verified by `Q8Channel.check_round` after every round and
by the tests (the aggregate equals the FedAvg of the dequantised uploads, bitwise). The process pool keeps the
payload inside the worker (`upload_counted`), so Q8Channel refuses it.
"""
from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable, Mapping

import torch
from torch import Tensor

from .adapter import BufferRule, ParamManifest
from .comm import LEDGER, UPLOAD_HEADER_BYTES, Ledger, VirtualChannel
from .numerics import derive_seed

CODEC = "Q8_SR_PER_TENSOR"
QMAX = 127
SCALE_BYTES = 4


class Q8Error(RuntimeError):
    pass


def visit_seed_q8(seed: int, round_idx: int, key: str) -> int:
    return derive_seed(int(seed), "q8", int(round_idx), str(key))


def quantize_tensor(delta: Tensor, gen: torch.Generator | None, *, stochastic: bool = True) -> tuple:
    """(q int8, scale 0-dim FP32) of one delta tensor. `stochastic=False` (round to nearest) exists ONLY for the
    negative control that shows deterministic rounding is biased."""
    if delta.dtype != torch.float32:
        raise Q8Error(f"delta is {delta.dtype}, expected float32")
    scale = delta.abs().max() / QMAX if delta.numel() else torch.zeros((), dtype=torch.float32, device=delta.device)
    if delta.numel() == 0 or float(scale) == 0.0:
        return (torch.zeros(delta.shape, dtype=torch.int8, device=delta.device),
                torch.zeros((), dtype=torch.float32, device=delta.device))
    x = delta / scale
    if stochastic:
        u = torch.rand(delta.shape, generator=gen, dtype=torch.float32, device=delta.device)
        q = torch.floor(x + u)
    else:
        q = torch.round(x)
    return q.clamp_(-QMAX, QMAX).to(torch.int8), scale.to(torch.float32)


def dequantize_tensor(q: Tensor, scale: Tensor) -> Tensor:
    return q.to(torch.float32) * scale


def encode(upload: Mapping[str, Tensor], theta: Mapping[str, Tensor], keys, *, seed: int, round_idx: int,
           key: str, stochastic: bool = True) -> OrderedDict[str, tuple]:
    """The client-side Q8 payload {k: (q int8, scale)} of one visit (keys in manifest order)."""
    dev = theta[keys[0]].device if keys else torch.device("cpu")
    gen = torch.Generator(device=dev).manual_seed(visit_seed_q8(seed, round_idx, key)) if stochastic else None
    out = OrderedDict()
    for k in keys:
        out[k] = quantize_tensor(upload[k].detach().to(torch.float32) - theta[k], gen, stochastic=stochastic)
    return out


def decode(payload: Mapping[str, tuple], theta: Mapping[str, Tensor]) -> OrderedDict[str, Tensor]:
    """The server-side dequantised upload theta_r + q * scale (FP32), same key order."""
    return OrderedDict((k, theta[k] + dequantize_tensor(q, s)) for k, (q, s) in payload.items())


def payload_bytes_q8(payload: Mapping[str, tuple]) -> int:
    return sum(int(q.numel()) * q.element_size() for q, _ in payload.values()) + SCALE_BYTES * len(payload)


def q8_payload_bytes(manifest: ParamManifest) -> int:
    """The codec payload per visit: sum(numel) + 4 x n_tensors over the shared canonical tensors."""
    return int(manifest.shared_numel) + SCALE_BYTES * len(manifest.shared_keys)


def q8_upload_bytes(manifest: ParamManifest) -> int:
    """The per-visit upload count: the codec payload + the 8-byte n_consumed header (as FA's FP32 upload)."""
    return q8_payload_bytes(manifest) + UPLOAD_HEADER_BYTES


def q8_visit_bytes(manifest: ParamManifest) -> dict:
    """comm.visit_bytes with the Q8 upload (download unchanged: dense FP32)."""
    shared = manifest.shared_bytes
    up = q8_upload_bytes(manifest)
    return {"download_bytes": shared,
            "upload_bytes": up,
            "send_plus_receive_bytes": shared + up,
            "private_bytes_sent": 0,
            "private_resident_bytes": manifest.private_resident_bytes,
            "bootstrap_bytes_once": manifest.buffer_bytes(BufferRule.FIXED) + manifest.buffer_bytes(BufferRule.SERVER_COPY),
            "alias_bytes_not_sent": sum(manifest.entries[k].nbytes for k in manifest.alias_keys),
            "shared_numel": manifest.shared_numel,
            "upload_codec": CODEC, "n_tensors": len(manifest.shared_keys),
            "codec_payload_bytes": q8_payload_bytes(manifest), "header_bytes": UPLOAD_HEADER_BYTES,
            "fp32_upload_bytes_fa": shared + UPLOAD_HEADER_BYTES,
            "rule": ("up = sum(numel) int8 + 4 x n_tensors FP32 scales + the 8-byte n_consumed "
                     "header (as FA); down = dense FP32")}


class Q8Channel(VirtualChannel):
    """A VirtualChannel for FA_Q8 (see the module docstring). `theta_fn()` returns the server's current broadcast
    state; it is snapshotted (cloned) at the first download of every round, i.e. theta_r of that round."""

    def __init__(self, manifest: ParamManifest, *, seed: int, theta_fn: Callable[[], Mapping[str, Tensor]],
                 ledger: Ledger = LEDGER):
        super().__init__(manifest, ledger=ledger)
        self.seed = int(seed)
        self.theta_fn = theta_fn
        self._round: int | None = None
        self._theta: dict | None = None
        self.uploads_by_round: dict = {}          # round -> [client keys decoded, in upload order]

    def download(self, round_idx: int, key: str, retry: bool = False) -> int:
        if self._round != int(round_idx):
            self._round = int(round_idx)
            self._theta = {k: v.detach().clone() for k, v in self.theta_fn().items() if k in self.manifest.shared_keys}
            self.uploads_by_round = {int(round_idx): []}
        return super().download(round_idx, key, retry)

    def upload(self, round_idx: int, key: str, upload: Mapping[str, Tensor]) -> int:
        if set(upload) != set(self.manifest.shared_keys):
            raise ValueError(f"upload of {key} is not exactly the shared key set")
        if self._round != int(round_idx) or self._theta is None:
            raise Q8Error(f"Q8 upload of {key} in round {round_idx} without that round's download (theta_r unknown)")
        if not isinstance(upload, dict):
            raise Q8Error("the upload must be a mutable dict (the server-side decode replaces its entries)")
        keys = self.manifest.shared_keys
        payload = encode(upload, self._theta, keys, seed=self.seed, round_idx=round_idx, key=key)
        deq = decode(payload, self._theta)
        for k in keys:                            # the server-side dequantised value is what gets aggregated
            upload[k] = deq[k]
        n = payload_bytes_q8(payload)
        if n != q8_payload_bytes(self.manifest):
            raise Q8Error(f"Q8 payload bytes {n} != the formula {q8_payload_bytes(self.manifest)}")
        n += UPLOAD_HEADER_BYTES                  # the n_consumed header, counted as for FA
        self._rec(round_idx, "up", n)
        self.uploads_by_round.setdefault(int(round_idx), []).append(str(key))
        return n

    def upload_counted(self, round_idx: int, key: str) -> int:
        raise Q8Error("FA_Q8 runs on the in-process driver only (the process pool keeps the payload in the worker)")

    def check_round(self, round_idx: int, survivors) -> None:
        """After a round: every survivor was decoded exactly once, in logical order (0 survivors: nothing)."""
        got = self.uploads_by_round.get(int(round_idx), [])
        if list(got) != [str(k) for k in survivors]:
            raise Q8Error(f"round {round_idx}: Q8 decoded {len(got)} upload(s) {got[:3]}..., expected the "
                          f"{len(list(survivors))} survivor(s) in logical order")


def install_q8(run, *, seed: int) -> Q8Channel:
    """Attach a Q8Channel to a runtime.FLRun (after construction AND after resume, which rebuilds the channel);
    it shares the run's ledger so comm accounting and the checkpointed ledger carry the Q8 byte counts."""
    if getattr(run, "pool", None) is not None:
        raise Q8Error("FA_Q8 runs on the in-process driver only")
    if run.cfg.method != "FA" or run.cfg.dp is not None or run.cfg.pf is not None or run.cfg.mu != 0:
        raise Q8Error("FA_Q8 is FA exactly (no DP, no PF, mu = 0)")
    ch = Q8Channel(run.server.manifest, seed=seed, theta_fn=lambda: run.server.adapter.broadcast_state(),
                   ledger=run.ledger)
    run.channel = ch
    return ch
