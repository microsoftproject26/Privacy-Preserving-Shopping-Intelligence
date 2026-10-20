"""v4trunc.provider — Study A evaluation batches and Study B support batches over the E11.4 builder.

TruncEvalProvider(builder, panel, panel_type, h | FULL)
  The evaluation stream of one sub-panel manifest (v4trunc.panel.subpanel_manifest: one query per user, ascending
  decision id, every row eval_mask) at budget h (or the full available prior history: the C_ref view, E11.10).
  batches(bs) yields exactly `bs` rows per batch except an unpadded tail (the frozen T3 batching of predict_adapter);
  batch_at(pos) serves arbitrary stream positions (the per-client path of launch_device.execute). Every batch is
  checked against the manifest (decision_id and target_class, ProviderIntegrityError otherwise) and records the
  budget. The same builder serves every h: nothing is refit per h (E11.5).
support_batch(builder, q_row, h)
  Study B (E11.9): the support examples of query q under budget h — one example per E11.3 support label e_j in
  H_h(q); the example's input is the builder's block of e_j over the permitted rows up to e_j (the virtual history
  H_h(q) cut at e_j, so it never sees e_q or anything after e_j); its target is e_j's own target; `target_ts` carries
  the label time for the causality check (fl.finetune.check_support_causal, cutoff = decision_ts(q)); `loss_mask` is
  the E11.3 label indicator (True on every support row — a VALIDATION support label has eval_mask, not loss_mask, in
  the ledger, and fl.client.valid_rows keys on loss_mask).
"""
from __future__ import annotations

from typing import Iterator, Optional

import numpy as np

from .builder import FULL, TruncatedHistoryBuilder
from .panel import PanelManifest, subpanel_manifest, support_rows

TS_KEY = "target_ts"
OOV_FLAGS = ("brand_oov_collision", "main_oov_collision")      # the builder audit flags of golden.V2_UCTX_PARTIAL


class ProviderIntegrityError(AssertionError):
    pass


class TruncEvalProvider:
    PROVIDER_ID = "v4trunc.TruncEvalProvider/1"

    def __init__(self, builder: TruncatedHistoryBuilder, pm: PanelManifest, panel_type: str, h: int, *,
                 full_history: bool = False):
        self.builder = builder
        self.panel_type, self.query_h, self.full_history = panel_type, int(h), bool(full_history)
        self.h = FULL if full_history else int(h)
        self.manifest, self.panel_rows = subpanel_manifest(pm, panel_type, h, full_history=full_history)
        t = builder.table
        self.event_rows = np.array([t.row_of(int(d)) for d in self.manifest.decision_id.tolist()], dtype=np.int64)
        if not np.array_equal(t.meta["target_class"][self.event_rows], self.manifest.target_class):
            raise ProviderIntegrityError(f"{self.manifest.manifest_id}: decisions_index target_class != panel "
                                         "query_target_class")
        self._ids = self.manifest.decision_id
        self._target = self.manifest.target_class
        n = int(self._ids.size)                               # REV-ANA-01 B-F7: per-query builder OOV audit flags
        self._served = np.zeros(n, dtype=bool)
        self._oov = {k: np.zeros(n, dtype=bool) for k in OOV_FLAGS}

    @property
    def n_stream_rows(self) -> int:
        return int(self._ids.size)

    def batch_at(self, pos) -> dict:
        pos = np.asarray(pos, dtype=np.int64)
        blocks = [self.builder.block(int(r), self.h) for r in self.event_rows[pos].tolist()]
        b = self.builder.batch_of_blocks(blocks)
        self._served[pos] = True
        for k in OOV_FLAGS:
            self._oov[k][pos] = [bool(blk.audit.get(k)) for blk in blocks]
        if not (np.array_equal(b["decision_id"].numpy(), self._ids[pos])
                and np.array_equal(b["target_class"].numpy(), self._target[pos])):
            raise ProviderIntegrityError(f"{self.manifest.manifest_id}: a batch disagrees with the manifest")
        return b

    def batches(self, batch_size: int) -> Iterator[dict]:
        bs = int(batch_size)
        for s in range(0, self.n_stream_rows, bs):
            yield self.batch_at(np.arange(s, min(self.n_stream_rows, s + bs)))

    def oov_flag_counts(self) -> dict:
        """Queries of this cell the builder flagged as PARTIAL_OOV_IDENTITY (golden.V2_UCTX_PARTIAL), among the
        stream positions served so far (`n_served`; a complete cell has n_served == n_stream_rows)."""
        anyf = np.zeros(self._served.size, dtype=bool)
        for k in OOV_FLAGS:
            anyf |= self._oov[k]
        out = {k: int(self._oov[k].sum()) for k in OOV_FLAGS}
        out.update(any=int(anyf.sum()), n_served=int(self._served.sum()), n_stream_rows=self.n_stream_rows)
        return out

    def describe(self) -> dict:
        return {"provider": self.PROVIDER_ID, "manifest_id": self.manifest.manifest_id,
                "panel_type": self.panel_type, "query_h": self.query_h,
                "history": "FULL" if self.full_history else "H_h", "n_stream_rows": self.n_stream_rows,
                "transform_state": self.builder.transform_state}


def support_blocks(builder: TruncatedHistoryBuilder, q_row: int, h: Optional[int]) -> tuple:
    """(blocks, support rows): one builder block per E11.3 support label of (q, h)."""
    t = builder.table
    rows = support_rows(t, int(q_row), h)
    u = int(t.row_user[q_row])
    first = int(t.user_start[u]) if h is FULL else int(q_row) - int(h)
    return [builder.block(int(j), int(j) - first) for j in rows.tolist()], rows


def support_batch(builder: TruncatedHistoryBuilder, q_row: int, h: Optional[int]) -> Optional[dict]:
    import torch
    blocks, rows = support_blocks(builder, q_row, h)
    if not blocks:
        return None
    b = builder.batch_of_blocks(blocks)
    t = builder.table
    b["loss_mask"] = torch.ones(len(blocks), dtype=torch.bool)
    b[TS_KEY] = torch.as_tensor(t.meta["target_ts_us"][rows], dtype=torch.long)
    if not bool(np.all(t.meta["target_class"][rows] >= 0)):
        raise ProviderIntegrityError("a support row without a catalogue target")
    return b


__all__ = ["TruncEvalProvider", "support_batch", "support_blocks", "TS_KEY", "OOV_FLAGS", "ProviderIntegrityError"]
