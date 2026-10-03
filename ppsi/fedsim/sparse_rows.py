"""A row-sparse ("delta-compressed") item upload codec: only the item rows a client touched, plus the top-k rows of
the output layer, are uploaded.

FA exactly, except the three item tensors of each surviving client's UPLOAD are sent row-sparse:
  item_embed.weight [V, d]          the rows the client TOUCHED: the item tokens of its own trained history (the valid
                                    decisions' real events, attention_mask True) and its target tokens (class j <-> item
                                    token j + 3) — `touched_rows(examples)`; sorted, unique;
  output_embed [K, d] + output_bias [K]
                                    ONE joint row set for the output layer [W_j | b_j]: the top-k classes by the delta
                                    norm sqrt(||W_j - W_r,j||^2 + (b_j - b_r,j)^2) (default k = 3,170 = 2 % of K;
                                    ties -> the lower class id: a stable descending sort; k >= K sends every row);
  every other shared key            dense FP32, exactly as FA.
A sent row carries the client's NEW FP32 row value (not the difference), so the server-side decode is exact on every
sent row: decoded = theta_r with the sent rows replaced (index_copy). A row that is not sent decodes to theta_r's row,
i.e. a ZERO delta for that client (no error feedback). `decode` reads ONLY the payload and theta_r — never the
client's dense upload — so the server cannot see an unsent row. aggregate.py is unchanged: the decoded dicts are what
the n_consumed-weighted FedAvg (same shard order) aggregates.

Wire format and bytes per visit (the codec reports its exact payload; the ledger records it):
  payload = sum over dense keys of numel x 4                          (FP32)
          + per sparse group: 4 (int32 row count) + 4 x n_rows (int32 row ids) + 4 x n_rows x row_width (FP32 rows)
            (row_width = d for item_embed; d + 1 for the joint output layer)
  up      = payload + the 8-byte n_consumed header (comm.UPLOAD_HEADER_BYTES, counted as for FA / FA_Q8)
  down    = the dense FP32 shared state (unchanged).
`payload_bytes_sparse(payload)` counts an actual payload; `sparse_payload_bytes(manifest, n_item_rows, n_output_rows)`
is the formula; the channel checks they agree on every visit.

Losslessness: with client SGD with momentum (client.SGD_M: momentum 0.9, weight decay 0) an untouched
item_embed row receives an exactly-zero gradient and is NOT moved, so the item_embed part is lossless; the output layer
gets a dense softmax gradient, so its top-k truncation is the only lossy part. (AdamW with weight decay would move
untouched rows: the channel's audit reports that dropped mass instead of hiding it.)

Integration without touching server.py / pool.py / runtime.py (the quant.py pattern): `SparseChannel` is a
VirtualChannel whose `upload()` is called by server.run_round with (round, client key, the client's upload dict)
BEFORE the same dict is added to the shard accumulator; it encodes, decodes against theta_r (snapshotted at the round's
first download), replaces the dict's entries by the decoded values and records the exact byte count.
`SparseChannel.check_round` verifies after every round that every survivor was decoded exactly once, in logical order.
`install_sparse(run, k_top=...)` attaches it to an FLRun (after construction AND after resume, which rebuilds the
channel). The process pool keeps the payload inside the worker (`upload_counted`), so the channel refuses it.
"""
from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field

import torch
from torch import Tensor

from .adapter import ParamManifest
from .client import valid_rows
from .comm import LEDGER, UPLOAD_HEADER_BYTES, Ledger, VirtualChannel

CODEC = "SPARSE_ROWS_TOPK"
ITEM_KEY = "item_embed.weight"
OUTPUT_KEYS = ("output_embed", "output_bias")          # one joint row set (class j: W_j and b_j)
K_TOP_REGISTERED = 3170                                 # 2 % of K = 158,486
TARGET_TOKEN_OFFSET = 3                                 # class j <-> item token j + 3 (SASRecCE arange class map)
ROW_ID_BYTES = 4                                        # int32 row ids
ROW_COUNT_BYTES = 4                                     # int32 row count per sparse group (variable-length framing)
VALUE_BYTES = 4                                         # FP32


class SparseError(RuntimeError):
    pass


# ------------------------------------------------------------------------------------------------ the touched rows
def touched_rows(examples: Mapping[str, Tensor], *, offset: int = TARGET_TOKEN_OFFSET,
                 target_key: str = "target_class", loss_mask_key: str = "loss_mask") -> Tensor:
    """The item_embed rows a client's visit can move: the item tokens of the real events (attention_mask True) of its
    VALID decisions (client.valid_rows: the rows it trains on) and their target tokens (target_class + offset).
    Sorted, unique, int64 on the CPU. A client without valid decisions touches nothing (empty)."""
    idx = valid_rows(examples, target_key, loss_mask_key)
    if idx.numel() == 0:
        return torch.zeros(0, dtype=torch.long)
    tok = examples["item_tokens"].index_select(0, idx.to(examples["item_tokens"].device))
    am = examples["attention_mask"].index_select(0, idx.to(examples["attention_mask"].device)).to(torch.bool)
    hist = tok[am.to(tok.device)].long().cpu()
    tgt = examples[target_key].index_select(0, idx.to(examples[target_key].device)).long().cpu() + int(offset)
    return torch.unique(torch.cat([hist, tgt]))         # sorted ascending


# ------------------------------------------------------------------------------------------------ the codec
@dataclass
class SparseGroup:
    keys: tuple                                         # the tensors sharing this row set (dim 0 = row)
    ids: Tensor                                         # int32 [n], sorted ascending, unique
    rows: OrderedDict[str, Tensor]                    # key -> FP32 [n, *row_shape] (the client's new row values)

    @property
    def n(self) -> int:
        return int(self.ids.numel())


@dataclass
class SparsePayload:
    keys: tuple                                         # every encoded key, manifest order
    dense: OrderedDict[str, Tensor] = field(default_factory=OrderedDict)
    groups: list = field(default_factory=list)          # [SparseGroup]

    def group_of(self, key: str) -> SparseGroup | None:
        for g in self.groups:
            if key in g.keys:
                return g
        return None


def _check_fp32(name: str, t: Tensor) -> None:
    if t.dtype != torch.float32:
        raise SparseError(f"{name} is {t.dtype}, expected float32")


def top_k_rows(upload: Mapping[str, Tensor], theta: Mapping[str, Tensor], keys: Sequence[str], k_top: int) -> Tensor:
    """Client side: the top-k row ids (int64, sorted ascending) of the joint delta norm over `keys` (dim 0 = row);
    ties -> the lower id (stable descending sort); k_top >= n_rows -> every row."""
    k_top = int(k_top)
    if k_top < 1:
        raise SparseError(f"k_top must be >= 1, got {k_top}")
    n_rows = int(theta[keys[0]].shape[0])
    dev = theta[keys[0]].device
    if k_top >= n_rows:
        return torch.arange(n_rows, dtype=torch.long, device=dev)
    nsq = torch.zeros(n_rows, dtype=torch.float32, device=dev)
    for k in keys:
        d = upload[k].detach().to(dev) - theta[k]
        nsq = nsq + (d * d).reshape(n_rows, -1).sum(1)
    order = torch.argsort(nsq, descending=True, stable=True)[:k_top]
    return torch.sort(order).values


def encode(upload: Mapping[str, Tensor], theta: Mapping[str, Tensor], keys: Sequence[str], *, touched_rows: Tensor,
           k_top: int) -> SparsePayload:
    """The client-side sparse payload of one visit (see the module docstring). `keys` = the shared keys in manifest order;
    item_embed.weight is sent on `touched_rows` (int ids), the output layer on its top-k joint rows; an item key that
    is not in `keys` (frozen) is simply not encoded; output_embed / output_bias are both present or both absent."""
    keys = tuple(keys)
    have_out = [k in keys for k in OUTPUT_KEYS]
    if any(have_out) and not all(have_out):
        raise SparseError(f"the output layer {OUTPUT_KEYS} is encoded together (one row set) or not at all")
    pay = SparsePayload(keys)
    for k in keys:
        _check_fp32(f"upload[{k}]", upload[k])
        _check_fp32(f"theta[{k}]", theta[k])
        if tuple(upload[k].shape) != tuple(theta[k].shape):
            raise SparseError(f"upload[{k}] shape {tuple(upload[k].shape)} != theta {tuple(theta[k].shape)}")
    if ITEM_KEY in keys:
        V = int(theta[ITEM_KEY].shape[0])
        ids = torch.as_tensor(touched_rows).long().reshape(-1).cpu()
        if ids.numel():
            if int(ids.min()) < 0 or int(ids.max()) >= V:
                raise SparseError(f"touched rows outside [0, {V})")
            if ids.numel() > 1 and not bool((ids[1:] > ids[:-1]).all()):
                raise SparseError("touched rows must be sorted and unique")
        dev = theta[ITEM_KEY].device
        ids_d = ids.to(dev)
        pay.groups.append(SparseGroup((ITEM_KEY,), ids_d.to(torch.int32),
                                      OrderedDict([(ITEM_KEY, upload[ITEM_KEY].detach().to(dev).index_select(0, ids_d))])))
    if all(have_out):
        ids_d = top_k_rows(upload, theta, OUTPUT_KEYS, k_top)
        dev = ids_d.device
        pay.groups.append(SparseGroup(OUTPUT_KEYS, ids_d.to(torch.int32),
                                      OrderedDict((k, upload[k].detach().to(dev).index_select(0, ids_d))
                                                  for k in OUTPUT_KEYS)))
    sparse = {k for g in pay.groups for k in g.keys}
    for k in keys:
        if k not in sparse:
            pay.dense[k] = upload[k].detach()
    return pay


def decode(payload: SparsePayload, theta: Mapping[str, Tensor]) -> OrderedDict[str, Tensor]:
    """Server side: the dense FP32 upload (same key order). Dense keys pass through; a sparse key is theta_r with the
    sent rows replaced. Reads ONLY the payload and theta_r."""
    out = OrderedDict()
    for k in payload.keys:
        if k in payload.dense:
            out[k] = payload.dense[k]
            continue
        g = payload.group_of(k)
        if g is None:
            raise SparseError(f"key {k} is neither dense nor in a sparse group")
        t = theta[k].detach().clone()
        if g.n:
            t.index_copy_(0, g.ids.to(t.device).long(), g.rows[k].to(t.device))
        out[k] = t
    return out


def payload_bytes_sparse(payload: SparsePayload) -> int:
    """The exact bytes of one payload (n_consumed header excluded; the channel adds it)."""
    n = sum(int(t.numel()) * t.element_size() for t in payload.dense.values())
    for g in payload.groups:
        n += ROW_COUNT_BYTES + ROW_ID_BYTES * g.n + sum(int(t.numel()) * t.element_size() for t in g.rows.values())
    return n


def _groups_of(manifest: ParamManifest) -> list:
    keys = manifest.shared_keys
    out = []
    if ITEM_KEY in keys:
        out.append((ITEM_KEY,))
    if all(k in keys for k in OUTPUT_KEYS):
        out.append(OUTPUT_KEYS)
    return out


def _width(manifest: ParamManifest, keys: tuple) -> int:
    w = 0
    for k in keys:
        e = manifest.entries[k]
        w += e.numel // int(e.shape[0])
    return w


def sparse_payload_bytes(manifest: ParamManifest, n_item_rows: int, n_output_rows: int) -> int:
    """The formula: dense FP32 non-sparse keys + per group (4 + 4 n + 4 n width)."""
    groups = _groups_of(manifest)
    sparse = {k for g in groups for k in g}
    n = sum(manifest.entries[k].nbytes for k in manifest.shared_keys if k not in sparse)
    for g in groups:
        rows = int(n_item_rows) if g == (ITEM_KEY,) else int(n_output_rows)
        n += ROW_COUNT_BYTES + ROW_ID_BYTES * rows + VALUE_BYTES * rows * _width(manifest, g)
    return n


def sparse_upload_bytes(manifest: ParamManifest, n_item_rows: int, n_output_rows: int) -> int:
    return sparse_payload_bytes(manifest, n_item_rows, n_output_rows) + UPLOAD_HEADER_BYTES


def output_rows_sent(manifest: ParamManifest, k_top: int) -> int:
    if not all(k in manifest.shared_keys for k in OUTPUT_KEYS):
        return 0
    return min(int(k_top), int(manifest.entries[OUTPUT_KEYS[0]].shape[0]))


def sparse_visit_bytes(manifest: ParamManifest, k_top: int, n_item_rows: int) -> dict:
    """The per-visit record for a visit with `n_item_rows` touched rows (the upload is client-dependent; the ledger
    carries the exact totals)."""
    shared = manifest.shared_bytes
    n_out = output_rows_sent(manifest, k_top)
    up = sparse_upload_bytes(manifest, n_item_rows, n_out)
    return {"download_bytes": shared, "upload_bytes": up, "send_plus_receive_bytes": shared + up,
            "upload_codec": CODEC, "k_top": int(k_top), "n_item_rows": int(n_item_rows), "n_output_rows": n_out,
            "codec_payload_bytes": up - UPLOAD_HEADER_BYTES, "header_bytes": UPLOAD_HEADER_BYTES,
            "fp32_upload_bytes_fa": shared + UPLOAD_HEADER_BYTES,
            "rule": ("up = dense FP32 non-item keys + per sparse group (4-byte int32 row count + 4 x n int32 row ids "
                     "+ 4 x n x row width FP32 rows; item_embed: touched rows, width d; output layer: top-k joint rows "
                     "[W_j | b_j], width d + 1) + the 8-byte n_consumed header; down = dense FP32")}


# ------------------------------------------------------------------------------------------------ the channel
class SparseChannel(VirtualChannel):
    """A VirtualChannel for the sparse-rows codec (see the module docstring). `theta_fn()` returns the server's current broadcast state;
    its sparse-group tensors are snapshotted (cloned) at the first download of every round (= theta_r of that round).
    `touched_fn(client_key)` returns the client's touched item rows. `audit=True` records, per round, the largest
    |delta| of an UNSENT item_embed row (0 under SGD_M) and the fraction of the output layer's squared delta norm that
    was not sent (monitoring only; no tensor is changed). The audit is a SIMULATOR ORACLE: it reads the client's dense
    upload, which a real server never receives. `totals` / `round_stats` / `summary()` live in this process only
    (they restart after a resume); bytes of record come from `sparse_bytes_record` (the checkpointed ledger)."""

    def __init__(self, manifest: ParamManifest, *, k_top: int, theta_fn: Callable[[], Mapping[str, Tensor]],
                 touched_fn: Callable[[str], Tensor], ledger: Ledger = LEDGER, audit: bool = True):
        super().__init__(manifest, ledger=ledger)
        if int(k_top) < 1:
            raise SparseError(f"k_top must be >= 1, got {k_top}")
        self.k_top = int(k_top)
        self.theta_fn = theta_fn
        self.touched_fn = touched_fn
        self.audit = bool(audit)
        self.sparse_keys = tuple(k for g in _groups_of(manifest) for k in g)
        if not self.sparse_keys:
            raise SparseError("no item tensor is a shared (uploaded) key: nothing to encode row-sparse")
        self._round: int | None = None
        self._theta: dict | None = None
        self.uploads_by_round: dict = {}          # round -> [client keys decoded, in upload order] (current round)
        self.round_stats: dict = {}               # round -> stats (current round)
        self.totals = {"visits": 0, "bytes_up": 0, "item_rows": 0, "output_rows": 0,
                       "item_dropped_max_abs": 0.0, "output_dropped_frac_max": 0.0}

    def download(self, round_idx: int, key: str, retry: bool = False) -> int:
        if self._round != int(round_idx):
            self._round = int(round_idx)
            st = self.theta_fn()
            self._theta = {k: st[k].detach().clone() for k in self.sparse_keys}
            self.uploads_by_round = {int(round_idx): []}
            self.round_stats = {int(round_idx): {"visits": 0, "bytes_up": 0, "item_rows": 0, "output_rows": 0,
                                                 "item_dropped_max_abs": 0.0, "output_dropped_frac_max": 0.0}}
        return super().download(round_idx, key, retry)

    def _audit(self, upload: Mapping[str, Tensor], pay: SparsePayload) -> tuple:
        item_drop, out_frac = 0.0, 0.0
        th = self._theta
        g = pay.group_of(ITEM_KEY)
        if g is not None:
            d = (upload[ITEM_KEY].detach() - th[ITEM_KEY]).abs()
            if g.n:
                d.index_fill_(0, g.ids.to(d.device).long(), 0.0)
            item_drop = float(d.max()) if d.numel() else 0.0
        g = pay.group_of(OUTPUT_KEYS[0])
        if g is not None:
            n_rows = int(th[OUTPUT_KEYS[0]].shape[0])
            nsq = torch.zeros(n_rows, dtype=torch.float32, device=th[OUTPUT_KEYS[0]].device)
            for k in OUTPUT_KEYS:
                d = upload[k].detach() - th[k]
                nsq = nsq + (d * d).reshape(n_rows, -1).sum(1)
            tot = float(nsq.sum())
            sent = float(nsq.index_select(0, g.ids.to(nsq.device).long()).sum()) if g.n else 0.0
            out_frac = (tot - sent) / tot if tot > 0 else 0.0
        return item_drop, out_frac

    def upload(self, round_idx: int, key: str, upload: Mapping[str, Tensor]) -> int:
        if set(upload) != set(self.manifest.shared_keys):
            raise ValueError(f"upload of {key} is not exactly the shared key set")
        if self._round != int(round_idx) or self._theta is None:
            raise SparseError(f"sparse upload of {key} in round {round_idx} without that round's download "
                              f"(theta_r unknown)")
        if not isinstance(upload, dict):
            raise SparseError("the upload must be a mutable dict (the server-side decode replaces its entries)")
        keys = self.manifest.shared_keys
        th = self._theta
        rows = self.touched_fn(key) if ITEM_KEY in self.sparse_keys else torch.zeros(0, dtype=torch.long)
        theta_view = {k: th[k] if k in th else upload[k] for k in keys}     # dense keys: theta is never read
        pay = encode(upload, theta_view, keys, touched_rows=rows, k_top=self.k_top)
        stats = self._audit(upload, pay) if self.audit else (0.0, 0.0)
        dec = decode(pay, th)                     # the server side: payload + theta_r only
        for k in keys:
            upload[k] = dec[k]
        n = payload_bytes_sparse(pay)
        gi, go = pay.group_of(ITEM_KEY), pay.group_of(OUTPUT_KEYS[0])
        n_item, n_out = (gi.n if gi is not None else 0), (go.n if go is not None else 0)
        want = sparse_payload_bytes(self.manifest, n_item, n_out)
        if n != want:
            raise SparseError(f"sparse payload bytes {n} != the formula {want}")
        n += UPLOAD_HEADER_BYTES                  # the n_consumed header, counted as for FA
        self._rec(round_idx, "up", n)
        self.uploads_by_round.setdefault(int(round_idx), []).append(str(key))
        for rec in (self.round_stats[int(round_idx)], self.totals):
            rec["visits"] += 1
            rec["bytes_up"] += n
            rec["item_rows"] += n_item
            rec["output_rows"] += n_out
            rec["item_dropped_max_abs"] = max(rec["item_dropped_max_abs"], stats[0])
            rec["output_dropped_frac_max"] = max(rec["output_dropped_frac_max"], stats[1])
        return n

    def upload_counted(self, round_idx: int, key: str) -> int:
        raise SparseError("sparse uploads run on the in-process driver only (the process pool keeps the payload "
                          "in the worker)")

    def check_round(self, round_idx: int, survivors) -> None:
        """After a round: every survivor was decoded exactly once, in logical order (0 survivors: nothing)."""
        got = self.uploads_by_round.get(int(round_idx), [])
        if list(got) != [str(k) for k in survivors]:
            raise SparseError(f"round {round_idx}: sparse codec decoded {len(got)} upload(s) {got[:3]}..., expected "
                              f"the {len(list(survivors))} survivor(s) in logical order")

    def summary(self) -> dict:
        t = dict(self.totals)
        v = max(1, t["visits"])
        t.update(codec=CODEC, k_top=self.k_top, mean_upload_bytes=t["bytes_up"] / v,
                 mean_item_rows=t["item_rows"] / v, mean_output_rows=t["output_rows"] / v, audit=self.audit,
                 scope="THIS PROCESS LEG ONLY (monitoring; restarts after a resume) - bytes of record come from "
                       "sparse_bytes_record (checkpointed ledger)",
                 audit_note="item_dropped / output_dropped are a SIMULATOR ORACLE (they read the client's dense "
                            "upload); a real server never sees unsent rows")
        return t


def is_sparse_codec(codec) -> bool:
    """Launcher dispatch: True only for a spec `upload_codec` mapping whose name is this codec's.
    None, the FA_Q8 codec record (name Q8_SR_PER_TENSOR) or any other name -> False."""
    return isinstance(codec, Mapping) and codec.get("name") == CODEC


def sparse_codec_record(k_top: int = K_TOP_REGISTERED) -> dict:
    """The spec `upload_codec` record of this codec (name + k_top + the wire rule), for a run-spec digest."""
    return {"name": CODEC, "k_top": int(k_top), "item_rows": "touched (valid decisions' history + target tokens)",
            "output_rows": "top-k joint [W_j | b_j] by delta L2 norm, ties -> lower class id",
            "upload_bytes": "dense FP32 non-item + per group (4 + 4 n + 4 n width) + 8-byte n_consumed header",
            "download": "dense FP32"}


def sparse_bytes_record(manifest: ParamManifest, counters, ledger: Ledger, *, k_top: int) -> dict:
    """Sparse-codec bytes from CHECKPOINTED state only: the run's ledger (restored by FLRun.resume) and the
    exposure counters' visit counts. The upload is client-dependent, so the per-visit upload is the fleet MEAN
    ledger.bytes_up / sum(visit_counts); the download per visit is the dense FP32 shared state. SparseChannel.summary()
    covers only the current process leg (monitoring) and is not used here."""
    import numpy as np
    visits = np.asarray(counters.visit_counts, dtype=np.int64)
    n_vis = int(visits.sum())
    msgs, down, up, retry = ledger.snapshot()
    if n_vis and down != n_vis * manifest.shared_bytes:
        raise SparseError(f"ledger download bytes {down} != visits {n_vis} x dense shared {manifest.shared_bytes}")
    up_mean = up / n_vis if n_vis else 0.0
    shared = manifest.shared_bytes
    per = visits * shared + visits * up_mean
    return {"per_visit": {"download_bytes": shared, "upload_bytes_mean": up_mean,
                          "send_plus_receive_bytes_mean": shared + up_mean, "upload_codec": CODEC, "k_top": int(k_top),
                          "fp32_upload_bytes_fa": shared + UPLOAD_HEADER_BYTES,
                          "upload_bytes_if_no_item_rows": sparse_upload_bytes(manifest, 0,
                                                                              output_rows_sent(manifest, k_top)),
                          "rule": "upload_bytes_mean = ledger.bytes_up / sum(counters.visit_counts) (checkpointed "
                                  "state; survives resume); download = dense FP32 shared state"},
            "per_client_approx": {"mean": float(per.mean()) if per.size else 0.0,
                                  "median": float(np.median(per)) if per.size else 0.0,
                                  "rule": "visits x (down + fleet-mean up); the ledger has no per-client upload"},
            "fleet": {"messages": msgs, "bytes_down": down, "bytes_up": up, "retry_bytes_down": retry,
                      "total": down + up + retry, "visits": n_vis},
            "source": "checkpointed ledger + counters"}


def install_sparse(run, *, k_top: int = K_TOP_REGISTERED, touched_fn: Callable[[str], Tensor] | None = None,
                   audit: bool = True) -> SparseChannel:
    """Attach a SparseChannel to a runtime.FLRun (after construction AND after resume, which rebuilds the
    channel); it shares the run's ledger, so comm accounting and the checkpointed ledger carry the exact sparse
    byte counts. `touched_fn` defaults to touched_rows(run.get_client(key).examples)."""
    if getattr(run, "pool", None) is not None:
        raise SparseError("sparse uploads run on the in-process driver only")
    if run.cfg.method != "FA" or run.cfg.dp is not None or run.cfg.pf is not None or run.cfg.mu != 0:
        raise SparseError("the sparse-rows run is FA exactly (no DP, no PF, mu = 0) apart from the upload codec")
    man = run.server.manifest
    missing = [k for k in (ITEM_KEY,) + OUTPUT_KEYS if k not in man.shared_keys]
    if missing:
        raise SparseError(f"the sparse-rows codec needs unfrozen (shared) item tensors; not shared: {missing}")
    if touched_fn is None:
        def touched_fn(key: str, _get=run.get_client) -> Tensor:
            return touched_rows(_get(key).examples)
    ch = SparseChannel(man, k_top=k_top, theta_fn=lambda: run.server.adapter.broadcast_state(),
                       touched_fn=touched_fn, ledger=run.ledger, audit=audit)
    run.channel = ch
    return ch
