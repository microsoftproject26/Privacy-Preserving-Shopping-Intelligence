"""synthetic_store.py (owner: Sonnet-data). SASREC_INTERFACES.md section 10.

Real-data-free fixture store with the SAME interface as the reused
`full_store.CompactStore`: `.decisions(split, kind) -> DecisionSet` and
`.batch(ds, positions) -> dict` with identical keys/dtypes/layout. Used by the
trainer/runtime tests (Sonnet-runtime, Opus) so nothing updates parameters on
real data before Gate C. Never touches the frozen dataset, never imports
`paths`/`full_store`'s dataset-reading code paths -- only the `DecisionSet`
dataclass is reused (a plain container, no I/O).

Data-only. No model code, no training loop.
"""
from __future__ import annotations

import itertools
from typing import Optional

import numpy as np
import torch

import paths  # noqa: F401  -- sys.path side effect only (code/reuse + vendor)
from full_store import DecisionSet  # noqa: E402  -- reused container, no I/O

__all__ = ["SyntheticStore"]

L = 50  # context_length, matches the real store's contract constant

SIDE_VOCAB = {
    "category_tokens": 12,
    "brand_tokens": 12,
    "main_category_tokens": 8,
    "daypart_tokens": 8,
    "weekday_tokens": 10,
}
NUMERIC_DIM = 19
FLAGS_DIM = 15
USER_CTX_DIM = 12
USER_MASK_DIM = 6
# USER_CONTEXT_FEATURES.index("has_history") / USER_CONTEXT_MASKS.index("has_history_mask")
# in the vendored rees46_eda.config -- kept as local constants so this module
# never needs to import the vendored config for a purely synthetic store.
HAS_HISTORY_FEATURE_IDX = 3
HAS_HISTORY_MASK_IDX = 0

OOV_CLIENT_IDX = 2  # matches vendor/rees46_eda/config.py OOV


class SyntheticStore:
    channels: str
    K: int
    item_vocab: int
    vocab_sizes: dict
    product_idx_of_class: np.ndarray
    version_hash: str = "SYNTHETIC"
    splits: tuple = ("TRAIN", "VALIDATION_SELECT")

    @classmethod
    def build(cls, *, channels: str, n_train: int = 4096, n_vs: int = 1024, K: int = 997,
              seed: int = 0, task: str = "mapping") -> "SyntheticStore":
        if channels not in ("F0", "F4"):
            raise ValueError(f"channels must be 'F0' or 'F4', got {channels!r}")
        if task not in ("mapping", "random"):
            raise ValueError(f"task must be 'mapping' or 'random', got {task!r}")
        if K < 4:
            raise ValueError("K must be >= 4 (need room for a non-trivial mapping)")
        if n_train < 2 or n_vs < 8:
            raise ValueError("n_train must be >= 2 and n_vs >= 8 for the required edge cases")

        self = cls()
        self.channels = channels
        self.K = int(K)
        self.item_vocab = int(K) + 3
        self.vocab_sizes = {"item_tokens": self.item_vocab, **SIDE_VOCAB}
        self.product_idx_of_class = np.arange(K, dtype=np.int64) + 3
        self.version_hash = "SYNTHETIC"
        self.splits = ("TRAIN", "VALIDATION_SELECT")
        self.task = task
        self.seed = int(seed)

        rng = np.random.default_rng(seed)
        self._mapping_perm = rng.permutation(K)

        # ONE global monotonic decision_id counter shared across BOTH splits, so
        # TRAIN and VALIDATION_SELECT ids can never collide (each split's ids
        # stay strictly ascending on their own, and VS ids all start strictly
        # after the last TRAIN id). A per-split counter previously let both
        # splits restart at 0, colliding on ~37/64 ids in a tiny build and
        # tripping train_sasrec.py's cross-split decision_id leakage guard.
        id_counter = itertools.count(0)
        self._raw: dict = {
            "TRAIN": self._build_train(rng, n_train, id_counter),
            "VALIDATION_SELECT": self._build_vs(rng, n_vs, id_counter),
        }
        self._decisions_cache: dict = {}
        return self

    # ---------------------------------------------------------- generation --
    def _target_for(self, rng, last_class: int) -> int:
        if self.task == "mapping":
            return int(self._mapping_perm[last_class])
        return int(rng.integers(0, self.K))

    def _make_window(self, rng, length: int):
        """One decision's own materialised [L] window (item classes + side
        tokens + numeric/flags), right-padded with zeros past `length`."""
        item = np.zeros(L, dtype=np.int64)
        cat = np.zeros(L, dtype=np.int64)
        brand = np.zeros(L, dtype=np.int64)
        main = np.zeros(L, dtype=np.int64)
        daypart = np.zeros(L, dtype=np.int64)
        weekday = np.zeros(L, dtype=np.int64)
        num = np.zeros((L, NUMERIC_DIM), dtype=np.float32)
        flags = np.zeros((L, FLAGS_DIM), dtype=np.int8)

        classes = rng.integers(0, self.K, size=length)
        item[:length] = classes + 3
        for name, vocab in SIDE_VOCAB.items():
            # tokens 3..vocab-1 are "known"; leave PAD(0) reserved, occasionally
            # emit MISSING(1) for realism -- never used past `length`.
            toks = rng.integers(3, vocab, size=length)
            {"category_tokens": cat, "brand_tokens": brand, "main_category_tokens": main,
             "daypart_tokens": daypart, "weekday_tokens": weekday}[name][:length] = toks
        num[:length] = rng.normal(size=(length, NUMERIC_DIM)).astype(np.float32)
        flags[:length] = rng.integers(0, 2, size=(length, FLAGS_DIM)).astype(np.int8)

        last_class = int(classes[-1]) if length else 0
        return dict(item=item, cat=cat, brand=brand, main=main, daypart=daypart,
                    weekday=weekday, num=num, flags=flags, last_class=last_class)

    def _append_decision(self, buf: dict, *, decision_id, win, target_class, loss_mask,
                          eval_mask, client_idx, eval_user_key, prior_view_count,
                          prefix_position, length, has_history: bool = False) -> None:
        buf["decision_id"].append(decision_id)
        buf["item"].append(win["item"]); buf["cat"].append(win["cat"]); buf["brand"].append(win["brand"])
        buf["main"].append(win["main"]); buf["daypart"].append(win["daypart"]); buf["weekday"].append(win["weekday"])
        buf["num"].append(win["num"]); buf["flags"].append(win["flags"])
        uctx = np.zeros(USER_CTX_DIM, dtype=np.float32)
        umask = np.zeros(USER_MASK_DIM, dtype=np.float32)
        # OOV(client_idx==2) users only "have history" from their 2nd own
        # decision onward within this VS-only synthetic population -- callers
        # (`_build_vs`) pass `has_history=True` starting at that user's k>0
        # decision, which is what A9's "client_idx==2 rows with has_history>0
        # exist" checks for.
        uctx[HAS_HISTORY_FEATURE_IDX] = 1.0 if has_history else 0.0
        umask[HAS_HISTORY_MASK_IDX] = 1.0
        buf["user_context"].append(uctx)
        buf["user_context_masks"].append(umask)
        buf["target_class"].append(int(target_class))
        buf["loss_mask"].append(bool(loss_mask))
        buf["eval_mask"].append(bool(eval_mask))
        buf["client_idx"].append(int(client_idx))
        buf["eval_user_key"].append(int(eval_user_key))
        buf["prior_view_count"].append(int(prior_view_count))
        buf["prefix_position"].append(int(prefix_position))
        buf["length"].append(int(length))

    @staticmethod
    def _empty_buf() -> dict:
        return {k: [] for k in ("decision_id", "item", "cat", "brand", "main", "daypart", "weekday",
                                 "num", "flags", "user_context", "user_context_masks", "target_class",
                                 "loss_mask", "eval_mask", "client_idx", "eval_user_key",
                                 "prior_view_count", "prefix_position", "length")}

    def _build_train(self, rng, n_train: int, id_counter) -> dict:
        buf = self._empty_buf()
        for i in range(n_train):
            did = next(id_counter)
            length = 1 if i == 0 else (L if i == 1 else int(rng.integers(1, L + 1)))
            win = self._make_window(rng, length)
            target = self._target_for(rng, win["last_class"])
            self._append_decision(buf, decision_id=did, win=win, target_class=target,
                                   loss_mask=True, eval_mask=False, client_idx=3 + i,
                                   eval_user_key=-1, prior_view_count=max(length - 1, 0),
                                   prefix_position=length, length=length)
        return self._finalize(buf)

    def _build_vs(self, rng, n_vs: int, id_counter) -> dict:
        buf = self._empty_buf()
        user_counter = 0

        def next_user():
            nonlocal user_counter
            user_counter += 1
            return user_counter

        n_multi = max(4, min(24, n_vs // 40))
        n_oov_target = max(2, min(16, n_vs // 80))
        n_loss_only = max(2, min(16, n_vs // 80))
        n_single = max(1, n_vs - 2 * n_multi - n_oov_target - n_loss_only)

        # --- group 1: multi-decision users, half OOV(client_idx=2), half warm ---
        for m in range(n_multi):
            euk = next_user()
            is_oov = (m % 2 == 0)
            client_idx = OOV_CLIENT_IDX if is_oov else (1000 + euk)
            n_dec = int(rng.integers(2, 5))
            positions = np.sort(rng.choice(np.arange(1, L + 1), size=n_dec, replace=False))
            for k, p in enumerate(positions):
                did = next(id_counter)
                length = int(p)
                win = self._make_window(rng, length)
                target = self._target_for(rng, win["last_class"])
                self._append_decision(buf, decision_id=did, win=win, target_class=target,
                                       loss_mask=False, eval_mask=True, client_idx=client_idx,
                                       eval_user_key=euk, prior_view_count=max(length - 1, 0),
                                       prefix_position=length, length=length, has_history=(k > 0))

        # --- group 2: OOV targets (unrankable), eval_mask True ---
        for _ in range(n_oov_target):
            did = next(id_counter)
            euk = next_user()
            length = int(rng.integers(1, L + 1))
            win = self._make_window(rng, length)
            self._append_decision(buf, decision_id=did, win=win, target_class=-1,
                                   loss_mask=False, eval_mask=True, client_idx=2000 + euk,
                                   eval_user_key=euk, prior_view_count=max(length - 1, 0),
                                   prefix_position=length, length=length)

        # --- group 3: eval_mask False rows (loss_mask True, visible via kind="loss") ---
        for _ in range(n_loss_only):
            did = next(id_counter)
            euk = next_user()
            length = int(rng.integers(1, L + 1))
            win = self._make_window(rng, length)
            target = self._target_for(rng, win["last_class"])
            self._append_decision(buf, decision_id=did, win=win, target_class=target,
                                   loss_mask=True, eval_mask=False, client_idx=3000 + euk,
                                   eval_user_key=-1, prior_view_count=max(length - 1, 0),
                                   prefix_position=length, length=length)

        # --- group 4: ordinary single-decision distinct warm eval users ---
        for _ in range(n_single):
            did = next(id_counter)
            euk = next_user()
            length = int(rng.integers(1, L + 1))
            win = self._make_window(rng, length)
            target = self._target_for(rng, win["last_class"])
            self._append_decision(buf, decision_id=did, win=win, target_class=target,
                                   loss_mask=False, eval_mask=True, client_idx=4000 + euk,
                                   eval_user_key=euk, prior_view_count=max(length - 1, 0),
                                   prefix_position=length, length=length)

        return self._finalize(buf)

    @staticmethod
    def _finalize(buf: dict) -> dict:
        n = len(buf["decision_id"])
        out = {
            "decision_id": np.asarray(buf["decision_id"], dtype=np.int64),
            "item": np.stack(buf["item"]) if n else np.zeros((0, L), dtype=np.int64),
            "cat": np.stack(buf["cat"]) if n else np.zeros((0, L), dtype=np.int64),
            "brand": np.stack(buf["brand"]) if n else np.zeros((0, L), dtype=np.int64),
            "main": np.stack(buf["main"]) if n else np.zeros((0, L), dtype=np.int64),
            "daypart": np.stack(buf["daypart"]) if n else np.zeros((0, L), dtype=np.int64),
            "weekday": np.stack(buf["weekday"]) if n else np.zeros((0, L), dtype=np.int64),
            "num": np.stack(buf["num"]) if n else np.zeros((0, L, NUMERIC_DIM), dtype=np.float32),
            "flags": np.stack(buf["flags"]) if n else np.zeros((0, L, FLAGS_DIM), dtype=np.int8),
            "user_context": np.stack(buf["user_context"]) if n else np.zeros((0, USER_CTX_DIM), dtype=np.float32),
            "user_context_masks": (np.stack(buf["user_context_masks"]) if n
                                    else np.zeros((0, USER_MASK_DIM), dtype=np.float32)),
            "target_class": np.asarray(buf["target_class"], dtype=np.int64),
            "loss_mask": np.asarray(buf["loss_mask"], dtype=bool),
            "eval_mask": np.asarray(buf["eval_mask"], dtype=bool),
            "client_idx": np.asarray(buf["client_idx"], dtype=np.int64),
            "eval_user_key": np.asarray(buf["eval_user_key"], dtype=np.int64),
            "prior_view_count": np.asarray(buf["prior_view_count"], dtype=np.int32),
            "prefix_position": np.asarray(buf["prefix_position"], dtype=np.int32),
            "length": np.asarray(buf["length"], dtype=np.int16),
        }
        return out

    # -------------------------------------------------------------- access --
    def decisions(self, split: str, kind: str) -> DecisionSet:
        if split not in self.splits:
            raise ValueError(f"split {split!r} not loaded by this synthetic store (loaded: {self.splits})")
        if kind not in ("loss", "eval"):
            raise ValueError(f"kind must be 'loss' or 'eval', got {kind!r}")
        key = (split, kind)
        if key in self._decisions_cache:
            return self._decisions_cache[key]

        arr = self._raw[split]
        mask = arr["loss_mask"] if kind == "loss" else arr["eval_mask"]
        idx = np.flatnonzero(mask)
        decision_id = arr["decision_id"][idx]
        order = np.argsort(decision_id, kind="stable")
        decision_id = decision_id[order]
        if decision_id.shape[0] > 1:
            assert np.all(np.diff(decision_id) > 0), (
                f"decision_id not strictly ascending for split={split} kind={kind} (duplicate?)")
        sel = idx[order]

        eval_user_key = arr["eval_user_key"][sel].astype(np.int64) if kind == "eval" else None
        ds = DecisionSet(
            split=split, kind=kind, decision_id=decision_id, end_row=sel.astype(np.int64),
            length=arr["length"][sel], target_class=arr["target_class"][sel].astype(np.int64),
            loss_mask=arr["loss_mask"][sel].astype(bool), eval_mask=arr["eval_mask"][sel].astype(bool),
            client_idx=arr["client_idx"][sel].astype(np.int64),
            user_bucket=np.zeros(sel.shape[0], dtype=np.int16),
            eval_user_key=eval_user_key,
            prior_view_count=arr["prior_view_count"][sel], prefix_position=arr["prefix_position"][sel],
        )
        self._decisions_cache[key] = ds
        return ds

    def batch(self, ds: DecisionSet, positions) -> dict:
        positions = np.asarray(positions, dtype=np.int64)
        arr = self._raw[ds.split]
        rows = ds.end_row[positions]
        length = ds.length[positions].astype(np.int64)
        ar = np.arange(L, dtype=np.int64)
        valid = ar[None, :] < length[:, None]

        out: dict = {}
        out["item_tokens"] = torch.from_numpy(arr["item"][rows])
        out["attention_mask"] = torch.from_numpy(np.ascontiguousarray(valid))
        pos_ids = np.where(valid, ar[None, :] + 1, 0).astype(np.int64)
        out["position_ids"] = torch.from_numpy(pos_ids)
        out["lengths"] = torch.from_numpy(length)
        out["last_valid_index"] = torch.from_numpy((length - 1).astype(np.int64))
        out["target_class"] = torch.from_numpy(ds.target_class[positions].astype(np.int64))
        out["loss_mask"] = torch.from_numpy(ds.loss_mask[positions].astype(bool))
        out["eval_mask"] = torch.from_numpy(ds.eval_mask[positions].astype(bool))
        out["decision_id"] = torch.from_numpy(ds.decision_id[positions].astype(np.int64))
        out["client_id"] = torch.from_numpy(ds.client_idx[positions].astype(np.int64))

        if self.channels == "F4":
            out["category_tokens"] = torch.from_numpy(arr["cat"][rows])
            out["brand_tokens"] = torch.from_numpy(arr["brand"][rows])
            out["main_category_tokens"] = torch.from_numpy(arr["main"][rows])
            out["daypart_tokens"] = torch.from_numpy(arr["daypart"][rows])
            out["weekday_tokens"] = torch.from_numpy(arr["weekday"][rows])
            out["event_numeric_features"] = torch.from_numpy(np.ascontiguousarray(arr["num"][rows]))
            out["event_quality_flags"] = torch.from_numpy(np.ascontiguousarray(arr["flags"][rows]))
            out["user_context"] = torch.from_numpy(np.ascontiguousarray(arr["user_context"][rows]))
            umask = arr["user_context_masks"][rows]
            masks_binary = bool(np.all((umask == 0.0) | (umask == 1.0)))
            umask = umask.astype(np.uint8) if masks_binary else umask.astype(np.float32)
            out["user_context_masks"] = torch.from_numpy(np.ascontiguousarray(umask))
        return out
