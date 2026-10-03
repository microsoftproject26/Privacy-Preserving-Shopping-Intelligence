"""Virtual communication accounting.

Tensors never leave the process in the simulator; the bytes a deployment would move are counted anyway.

Per client visit (theoretical payload bytes, FP32):
  download  = theta_r: every SHARED tensor once (a tied alias is the same storage and is not sent twice)
  upload    = the client's shared state (same key set) + an 8-byte n_consumed header (int64)
  private   = 0 bytes sent: p_u and its optimizer moments stay on the client (their resident size is reported)
  bootstrap = one-time, per client device: FIXED / SERVER_COPY buffers (integer class maps etc.), not per visit
A retried visit re-downloads theta_r; its bytes are counted separately as retry bytes. The serializer's actual byte
count (torch.save of the payload) is reported separately from the theoretical count (pickle framing overhead).

`LEDGER` is the process-wide record every `VirtualChannel` reports to; LO code never touches a channel, so an LO run
must leave the ledger unchanged.

Frozen item tables: with runtime.freeze_item_tables the frozen tensors are FIXED
buffers of the manifest, so download / upload per visit = the UPLOADED (still shared) parameters only, exactly as
above; `visit_bytes(manifest, frozen_keys=...)` then also reports the frozen tables' one-time download per client
(`frozen_bytes_once`, disclosed: a client downloads them once, on its first visit) and the uploaded parameter count.
Without frozen_keys the record is unchanged.
"""
from __future__ import annotations

import io
from collections.abc import Mapping
from dataclasses import dataclass, field

import torch
from torch import Tensor

from .adapter import BufferRule, ParamManifest

UPLOAD_HEADER_BYTES = 8     # n_consumed as int64


def payload_bytes(payload: Mapping[str, Tensor]) -> int:
    """Theoretical bytes of a tensor payload; tensors sharing one storage (tied aliases) are counted once."""
    seen, n = set(), 0
    for t in payload.values():
        key = (t.untyped_storage().data_ptr(), t.storage_offset(), tuple(t.shape), t.dtype) if t.numel() else None
        if key is not None and key in seen:
            continue
        if key is not None:
            seen.add(key)
        n += t.numel() * t.element_size()
    return n


def serialized_bytes(payload: Mapping[str, Tensor]) -> int:
    buf = io.BytesIO()
    torch.save({k: v.detach().cpu() for k, v in payload.items()}, buf)
    return buf.getbuffer().nbytes


def visit_bytes(manifest: ParamManifest, frozen_keys=()) -> dict:
    shared = manifest.shared_bytes
    out = _visit_bytes(manifest, shared)
    if frozen_keys:                                      # frozen tables, once per client
        fk = tuple(frozen_keys)
        bad = [k for k in fk if k not in manifest.entries or manifest.entries[k].rule != BufferRule.FIXED]
        if bad:
            raise ValueError(f"frozen keys {bad} are not FIXED buffers of the manifest (freeze_item_tables first)")
        out.update(frozen_keys=list(fk), frozen_bytes_once=sum(manifest.entries[k].nbytes for k in fk),
                   frozen_numel=sum(manifest.entries[k].numel for k in fk), uploaded_numel=manifest.shared_numel,
                   frozen_disclosure="the frozen item tables are downloaded ONCE per client (first visit) and never "
                                     "uploaded; per-visit bytes count the uploaded parameters only")
    return out


def _visit_bytes(manifest: ParamManifest, shared: int) -> dict:
    return {"download_bytes": shared,
            "upload_bytes": shared + UPLOAD_HEADER_BYTES,
            "send_plus_receive_bytes": 2 * shared + UPLOAD_HEADER_BYTES,
            "private_bytes_sent": 0,
            "private_resident_bytes": manifest.private_resident_bytes,
            "bootstrap_bytes_once": manifest.buffer_bytes(BufferRule.FIXED) + manifest.buffer_bytes(BufferRule.SERVER_COPY),
            "alias_bytes_not_sent": sum(manifest.entries[k].nbytes for k in manifest.alias_keys),
            "shared_numel": manifest.shared_numel}


@dataclass
class Ledger:
    messages: int = 0
    bytes_down: int = 0
    bytes_up: int = 0
    retry_bytes_down: int = 0
    by_round: dict = field(default_factory=dict)

    def record(self, round_idx: int, direction: str, nbytes: int, retry: bool = False) -> None:
        self.messages += 1
        r = self.by_round.setdefault(int(round_idx), {"down": 0, "up": 0, "retry_down": 0, "messages": 0})
        r["messages"] += 1
        if direction == "down":
            if retry:
                self.retry_bytes_down += nbytes
                r["retry_down"] += nbytes
            else:
                self.bytes_down += nbytes
                r["down"] += nbytes
        elif direction == "up":
            self.bytes_up += nbytes
            r["up"] += nbytes
        else:
            raise ValueError(direction)

    def snapshot(self) -> tuple:
        return (self.messages, self.bytes_down, self.bytes_up, self.retry_bytes_down)

    def state_dict(self) -> dict:
        return {"messages": self.messages, "bytes_down": self.bytes_down, "bytes_up": self.bytes_up,
                "retry_bytes_down": self.retry_bytes_down,
                "by_round": {str(k): dict(v) for k, v in self.by_round.items()}}

    @classmethod
    def from_state_dict(cls, d: dict) -> Ledger:
        return cls(int(d["messages"]), int(d["bytes_down"]), int(d["bytes_up"]), int(d["retry_bytes_down"]),
                   {int(k): {kk: int(vv) for kk, vv in v.items()} for k, v in d["by_round"].items()})


LEDGER = Ledger()


class VirtualChannel:
    def __init__(self, manifest: ParamManifest, ledger: Ledger = LEDGER):
        self.manifest = manifest
        self.ledger = ledger
        self.local = Ledger()

    def _rec(self, *a, **kw) -> None:
        self.ledger.record(*a, **kw)
        self.local.record(*a, **kw)

    def download(self, round_idx: int, key: str, retry: bool = False) -> int:
        n = self.manifest.shared_bytes
        self._rec(round_idx, "down", n, retry)
        return n

    def upload(self, round_idx: int, key: str, upload: Mapping[str, Tensor]) -> int:
        if set(upload) != set(self.manifest.shared_keys):
            raise ValueError(f"upload of {key} is not exactly the shared key set")
        n = payload_bytes(upload) + UPLOAD_HEADER_BYTES
        self._rec(round_idx, "up", n)
        return n

    def upload_counted(self, round_idx: int, key: str) -> int:
        """Upload accounting when the payload stays in a worker process: exactly the shared set + header."""
        n = self.manifest.shared_bytes + UPLOAD_HEADER_BYTES
        self._rec(round_idx, "up", n)
        return n
