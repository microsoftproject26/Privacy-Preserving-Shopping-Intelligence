"""v4fs.importance — grouped permutation importance under the frozen T3 evaluator (SCHEMA-01; binding: Fable R2 + R4,
POPULATION_5_100_SCIENTIFIC_SYNTHESIS §D3 items 4-5).

Groups: the 14 FS groups G1 … G12c of the L3 / L3S schema (FS_GROUPS; `check_groups` refuses any name that is not an
input, any overlap, and any channel left out). item_tokens and session_offset_tokens (NEVER_PERMUTED) are in no group;
a value and its availability flag / mask are one group and move jointly. SCHEMA-01R1 (DECISION_LOG 47): the REQUIRED
channels are item_tokens, category_tokens and session_offset_tokens; G1 (category_tokens) is still permuted and its
importance reported, descriptively (`descriptive_only`, `droppable` False), and `noise_groups` never returns it. DL 48
adds the candidate group G13 SESSION_INTENT (L3S only; `groups_for` picks the groups a schema holds): its per-event
token is swapped as a whole column and its decision-level values with their mask, under the one donor map of the group;
L3S has 15 groups, 14 droppable (G2 … G12c, G13).

For ONE trained model (an adapter of v4models.build on the derived schema; eval mode, no gradient) and an EvalProvider
(full Clean V2 batches of the frozen eval_batch, in manifest order, the E2E stream):
  1. baseline: the frozen prediction path (v4eval.predict.predict_adapter + v4eval.evaluate.evaluate); while it streams,
     the group channels are collected per stream row;
  2. for every group g and permutation seed p in PERM_SEEDS (3): a permutation pi_{g,p} of the stream rows, seeded by
     sha256("PPSI_V4_FS_PERM|p|g") (the same permutations for every model, family and training seed: a paired design);
     recipient row i receives donor pi(i)'s values of EVERY channel of g: per-event channels ([B, L] tokens, a column
     of the [B, L, C] numeric / flag blocks) as the WHOLE [L] column (all L positions, the donor's PAD included), and
     decision-level channels ([B, C] user context / masks) as one vector. The structural keys (lengths,
     attention_mask, position_ids) are never touched. The same batches (same order, sizes, other channels) go through
     the same frozen path. BINDING alignment (Fable D3 ruling, DECISION_LOG 45): `LENGTH_MATCHED` — the permutation
     is drawn inside each exact window-length class (ascending decision_id within the class; a class of size < 2
     stays unpermuted), because WHOLE_COLUMN moves PAD onto real events of right-padded windows (an artifact).
     WHOLE_COLUMN is kept only as a named diagnostic.)
  3. importance(g) = MRR@20(unpermuted) - mean_p MRR@20(g permuted, seed p) (client-macro, E2E); its paired-user
     bootstrap 95% CI on the ONE resample matrix W of the E2E user set (v4eval.bootstrap: 1,000 resamples, seed
     20260926; per resample b: base_b - mean_p perm_{p,b}); "within noise" iff the CI includes 0. Per-permutation-seed
     values are recorded.
`combine_seeds` gives the family-level importance over the two L3 training seeds (seed mean on the same W:
mean_s(base_{s,b} - mean_p perm_{s,p,b})) with the per-seed values; `noise_groups` returns the groups within noise for
BOTH families (the S* drops). The per-user sufficient statistics (S, N per user, E2E) of every evaluation are kept in
`ModelImportance.stats` so the combination and the acceptance rule (v4fs.decide) can be recomputed from the saved
FS_USERSTATS.npz files. Nothing here computes a gradient or changes a weight. Every row carries run_label
FEATURE_SEARCH_SCREEN.
"""
from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterable, Iterator, Mapping, Optional, Sequence

import numpy as np
import torch

import v4models  # noqa: F401  (ppsi_env first)
from fl.numerics import state_digest
from v4eval import evaluator_code_sha256
from v4eval import frozen_config as fc
from v4eval.bootstrap import UserStats, ci95, make_plan, point, resampled, user_stats
from v4eval.evaluate import evaluate
from v4eval.metrics import credits_from_ranks
from v4eval.predict import predict_adapter
from v4eval.values import Undefined, to_jsonable

RUN_LABEL = "FEATURE_SEARCH_SCREEN"
PRIMARY = "mrr@20"
STATISTIC = "macro_e2e"
PERM_TAG = "PPSI_V4_FS_PERM"
PERM_SEEDS = (0, 1, 2)
ALIGNMENTS = ("WHOLE_COLUMN", "LENGTH_MATCHED")
EVENT_BLOCKS = ("event_numeric_features", "event_quality_flags")
USER_BLOCKS = ("user_context", "user_context_masks")
TOKEN_KEYS = ("item_tokens", "category_tokens", "brand_tokens", "main_category_tokens", "daypart_tokens",
              "weekday_tokens", "session_offset_tokens", "session_mode_main_category_tokens")   # + SCHEMA-01R1 keys
DECISION_EXTENSION_KEYS = ("current_matches_session_mode", "current_matches_session_mode_available",
                           "session_mode_share")          # SCHEMA-01R1 (G13): [B] float keys of the batch
NEVER_PERMUTED = ("item_tokens", "session_offset_tokens")      # in no group (SCHEMA-01R1, DECISION_LOG 47)
from v4fs.schemas import FS_GROUPS, REQUIRED_CHANNELS  # noqa: E402  (the 14 binding groups; defined with the schemas)


def required_groups(groups: Mapping = FS_GROUPS) -> tuple:
    """Groups holding a REQUIRED channel (DECISION_LOG 47: G1 = category_tokens): their importance is computed and
    reported descriptively, but they are never dropped (noise_groups never returns them)."""
    return tuple(g for g, m in groups.items() if set(m) & set(REQUIRED_CHANNELS))


class ImportanceError(RuntimeError):
    pass


def groups_for(schema, groups: Mapping = FS_GROUPS) -> dict:
    """The FS groups of a schema: every group whose channels are all inputs of it (L3: G1 … G12c; L3S: + G13). A group
    that is only partly present is refused."""
    kept = set(schema.channel_names)
    part = [g for g, m in groups.items() if 0 < len(set(m) & kept) < len(m)]
    if part:
        raise ImportanceError(f"groups {part} are only partly inputs of {schema.schema_id}")
    return {g: tuple(m) for g, m in groups.items() if set(m) <= kept}


def check_groups(schema, groups: Mapping = FS_GROUPS, *, complete: bool = True) -> dict:
    """Refuse a group set that is not a partition of the schema's inputs minus item_tokens (names checked exactly)."""
    kept = list(schema.channel_names)
    seen: dict = {}
    problems = []
    for g, mem in groups.items():
        if not mem:
            problems.append(f"{g}: empty")
        for c in mem:
            if c in NEVER_PERMUTED:
                problems.append(f"{g}: {c} is never permuted (REQUIRED, in no group)")
            elif c not in kept:
                problems.append(f"{g}: {c!r} is not an input of {schema.schema_id}")
            if c in seen:
                problems.append(f"{c} is in both {seen[c]} and {g}")
            seen[c] = g
    if complete:
        left = [c for c in kept if c not in seen and c not in NEVER_PERMUTED]
        if left:
            problems.append(f"{schema.schema_id} channels in no group: {left}")
    if problems:
        raise ImportanceError("permutation groups do not match the schema: " + "; ".join(problems))
    return {g: tuple(m) for g, m in groups.items()}


# ------------------------------------------------------------------------------------------------ channel locations
@dataclass(frozen=True)
class Loc:
    """Where a channel lives in the FULL Clean V2 batch."""
    name: str
    key: str               # batch key
    col: Optional[int]     # column in a float block (None for a token channel)

    @property
    def per_event(self) -> bool:
        return self.key in TOKEN_KEYS or self.key in EVENT_BLOCKS


def locate(schema, names: Iterable[str]) -> list:
    """Batch locations of channel names, from the BASE (full Clean V2) name arrays of a derived schema."""
    bw = schema.base.widths
    blocks = {"event_numeric_features": bw.numeric_names, "event_quality_flags": bw.flag_names,
              "user_context": bw.user_context_names, "user_context_masks": bw.user_mask_names}
    out = []
    for n in names:
        if n in TOKEN_KEYS or n in DECISION_EXTENSION_KEYS:      # a whole batch key ([B, L] token or [B] value)
            out.append(Loc(n, n, None))
            continue
        hit = [(k, list(v).index(n)) for k, v in blocks.items() if n in v]
        if len(hit) != 1:
            raise ImportanceError(f"channel {n!r} is not a unique Clean V2 channel")
        out.append(Loc(n, hit[0][0], hit[0][1]))
    return out


# ------------------------------------------------------------------------------------------------ permutations
def perm_seed(p: int, group: str) -> int:
    h = hashlib.sha256(f"{PERM_TAG}|{int(p)}|{group}".encode("utf-8")).digest()
    return int.from_bytes(h[:8], "big")


def donor_map(lengths: np.ndarray, p: int, group: str, *, alignment: str = "LENGTH_MATCHED") -> np.ndarray:
    """int64 [S]: donor stream row of every recipient row (a permutation of the stream rows)."""
    if alignment not in ALIGNMENTS:
        raise ImportanceError(f"alignment must be one of {ALIGNMENTS}")
    lengths = np.asarray(lengths, dtype=np.int64)
    rng = np.random.default_rng(perm_seed(p, group))
    if alignment == "WHOLE_COLUMN":
        return rng.permutation(lengths.size).astype(np.int64)
    donor = np.empty(lengths.size, dtype=np.int64)
    for L in np.unique(lengths):                           # ascending lengths: a fixed draw order
        idx = np.flatnonzero(lengths == L)
        donor[idx] = idx[rng.permutation(idx.size)]
    return donor


def map_sha256(donor: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(donor, dtype="<i8").tobytes()).hexdigest()


def swap_diagnostics(lengths: np.ndarray, donor: np.ndarray) -> dict:
    """How often a whole-column swap pairs windows of different length (the donor's PAD reaching real positions)."""
    lr = np.asarray(lengths, dtype=np.int64)
    ld = lr[donor]
    real = int(lr.sum())
    _, inv, cnt = np.unique(lr, return_inverse=True, return_counts=True) if lr.size else (None, np.zeros(0, np.int64), np.zeros(0, np.int64))
    bin_size = cnt[inv] if lr.size else np.zeros(0, np.int64)
    return {"rows_length_mismatch_share": float(np.mean(lr != ld)) if lr.size else 0.0,
            "real_cells_receiving_donor_pad_share": float(np.maximum(lr - ld, 0).sum() / real) if real else 0.0,
            "fixed_points": int(np.count_nonzero(donor == np.arange(donor.size))),
            # Fable D3 alignment ruling (DECISION_LOG 45): share of rows in length bins of size < 10; bins < 2 stay unpermuted
            "rows_in_bins_lt10_share": float(np.mean(bin_size < 10)) if lr.size else 0.0,
            "rows_unpermuted_bin_lt2": int(np.count_nonzero(bin_size < 2))}


# ------------------------------------------------------------------------------------------------ collection
class ChannelStore:
    """Per stream row, the channels of every group (full-batch values; per-event channels at a common width)."""

    def __init__(self, locs: Iterable[Loc]):
        self.locs = list({(l.key, l.col): l for l in locs}.values())
        self._parts = {(l.key, l.col): [] for l in self.locs}
        self._len, self._did, self._widths = [], [], []
        self.arrays: dict = {}
        self.lengths: Optional[np.ndarray] = None
        self.decision_id: Optional[np.ndarray] = None
        self.width = 0

    def add(self, batch: Mapping) -> None:
        self._len.append(torch.as_tensor(batch["lengths"]).cpu().numpy().astype(np.int64))
        self._did.append(torch.as_tensor(batch["decision_id"]).cpu().numpy().astype(np.int64))
        self._widths.append(int(batch["attention_mask"].shape[1]))
        for l in self.locs:
            t = batch[l.key]
            v = (t if l.col is None else t[..., l.col]).detach().cpu()
            if v.dtype == torch.int64:
                if v.numel() and (int(v.max()) >= 2 ** 31 or int(v.min()) < -2 ** 31):
                    raise ImportanceError(f"{l.name}: token ids exceed int32")
                v = v.to(torch.int32)
            self._parts[(l.key, l.col)].append(v.numpy().copy())

    def finish(self) -> None:
        self.lengths = np.concatenate(self._len) if self._len else np.zeros(0, np.int64)
        self.decision_id = np.concatenate(self._did) if self._did else np.zeros(0, np.int64)
        self.width = W = max(self._widths, default=0)
        for k, parts in self._parts.items():
            if parts and parts[0].ndim == 2:                  # per-event: pad every batch to the common width W
                out = np.zeros((sum(p.shape[0] for p in parts), W), dtype=parts[0].dtype)
                a = 0
                for p in parts:
                    out[a:a + p.shape[0], :p.shape[1]] = p
                    a += p.shape[0]
            else:
                out = np.concatenate(parts) if parts else np.zeros(0)
            self.arrays[k] = out
        self._parts = {k: [] for k in self._parts}

    @property
    def nbytes(self) -> int:
        return int(sum(a.nbytes for a in self.arrays.values()))


def permuted_batches(batches: Iterable[Mapping], store: ChannelStore, locs: Iterable[Loc], donor: np.ndarray,
                     counters: Optional[dict] = None) -> Iterator[dict]:
    """The same batches with the group's channels replaced by the donor rows' values (stream order)."""
    locs = list(locs)
    s = 0
    trunc = 0
    for b in batches:
        n = int(torch.as_tensor(b["decision_id"]).shape[0])
        if not np.array_equal(torch.as_tensor(b["decision_id"]).cpu().numpy(), store.decision_id[s:s + n]):
            raise ImportanceError("a permuted pass sees a different stream than the collection pass")
        d = donor[s:s + n]
        out = dict(b)
        for key in {l.key for l in locs}:
            out[key] = b[key].clone()
        for l in locs:
            arr = store.arrays[(l.key, l.col)]
            t = out[l.key]
            if l.per_event:
                W = int(t.shape[1])
                src = arr[d]
                if src.shape[1] > W:
                    trunc += int(np.count_nonzero(src[:, W:]))
                    src = src[:, :W]
                elif src.shape[1] < W:
                    src = np.pad(src, ((0, 0), (0, W - src.shape[1])))
                val = torch.from_numpy(np.ascontiguousarray(src)).to(t.dtype)
            else:
                val = torch.from_numpy(np.ascontiguousarray(arr[d])).to(t.dtype).reshape(
                    t.shape if l.col is None else t[..., l.col].shape)
            if l.col is None:
                t.copy_(val)
            else:
                t[..., l.col] = val
        s += n
        yield out
    if s != store.decision_id.size:
        raise ImportanceError(f"a permuted pass covered {s} stream rows, the collection pass {store.decision_id.size}")
    if counters is not None:
        counters["truncated_nonzero_donor_cells"] = trunc


def window_consistency(orig: Mapping, perm: Mapping, locs: Iterable[Loc], donor_rows: np.ndarray,
                       donor_batch: Mapping) -> None:
    """Assert (AssertionError) the whole-column swap: for every recipient row, EVERY channel of the group equals its
    ONE donor's channel at every position [0, L) (per-event) or as a vector (decision-level); every other channel and
    the structural keys are unchanged."""
    locs = list(locs)
    idx = torch.as_tensor(np.asarray(donor_rows, dtype=np.int64))
    for l in locs:
        t = perm[l.key] if l.col is None else perm[l.key][..., l.col]
        src = donor_batch[l.key] if l.col is None else donor_batch[l.key][..., l.col]
        assert torch.equal(t, src.index_select(0, idx)), f"{l.name}: a recipient row is not its donor's whole column"
    moved = {(l.key, l.col) for l in locs}
    for k, v in orig.items():
        if not torch.is_tensor(v):
            continue
        if (k, None) in moved:
            continue
        if k in EVENT_BLOCKS + USER_BLOCKS:
            keep = [c for c in range(v.shape[-1]) if (k, c) not in moved]
            assert torch.equal(perm[k][..., keep], v[..., keep]), f"{k}: a channel outside the group changed"
            continue
        assert torch.equal(perm[k], v), f"{k}: a channel outside the group (or a structural key) changed"


# ------------------------------------------------------------------------------------------------ evaluation
def _tap(batches: Iterable[Mapping], store: Optional[ChannelStore]) -> Iterator[Mapping]:
    for b in batches:
        if store is not None:
            store.add(b)
        yield b


def evaluate_stream(adapter, batches, manifest, frozen, model_sha: str, device) -> tuple:
    """(macro E2E MRR@20, UserStats of MRR@20, the evaluate() result, the artifact) under the frozen T3 path."""
    fc.apply_runtime_flags(frozen)
    art = predict_adapter(adapter, batches, manifest, model_sha256=model_sha, config=frozen, device=device)
    res = evaluate(manifest, art, config=frozen, evaluator_code_sha256=evaluator_code_sha256())
    v = res["E2E"]["macro"][PRIMARY]
    if isinstance(v, Undefined):
        raise ImportanceError(f"{PRIMARY} macro E2E is UNDEFINED on {manifest.manifest_id}")
    return float(v), stats_from_ranks(art.ranks, manifest), res, art


def stats_from_ranks(ranks: np.ndarray, manifest) -> UserStats:
    credit = credits_from_ranks(np.asarray(ranks, dtype=np.int64), (20,))[PRIMARY]
    return user_stats(credit, manifest.eval_mask, manifest.rankable, manifest.eval_user_key)


def importance_block(plan, base: UserStats, perms: Sequence[UserStats]) -> dict:
    """importance = base - mean_p perm_p, point and paired-user bootstrap CI (per resample b on the same W)."""
    pb = point(base, STATISTIC)
    pp = [point(s, STATISTIC) for s in perms]
    imp = float(np.mean([pb - v for v in pp]))            # = base - mean_p perm_p (a mean of differences: exact 0
    rb = resampled(plan, base, STATISTIC)                 #   when every permuted value equals the baseline)
    ci = ci95(np.mean([rb - resampled(plan, s, STATISTIC) for s in perms], axis=0))
    return {"importance": imp, "ci95": ci, "within_noise": bool(ci[0] <= 0.0 <= ci[1]),
            "per_perm_seed_delta": [v - pb for v in pp], "relative_importance": imp / pb if pb else None}


@dataclass
class ModelImportance:
    """One model's importance: rows (JSON) + the per-user statistics (for the seed combination and v4fs.decide)."""
    rows: list
    baseline: dict
    stats: Dict[str, UserStats] = field(default_factory=dict)     # "base" and f"{group}|{p}"
    meta: dict = field(default_factory=dict)

    def as_json(self) -> dict:
        return to_jsonable({"run_label": RUN_LABEL, "baseline": self.baseline, "groups": self.rows,
                            "meta": self.meta})


def grouped_permutation_importance(adapter, evalp, schema, *, groups: Optional[Mapping] = None,
                                   perm_seeds: Sequence[int] = PERM_SEEDS, alignment: str = "LENGTH_MATCHED",
                                   frozen=None, device=None, model_sha256: Optional[str] = None, run_id: str = "",
                                   complete: bool = True, log: Callable = print) -> ModelImportance:
    """Grouped permutation importance of `adapter` (a trained model of derived schema `schema`) on `evalp`."""
    frozen = fc.require_frozen(frozen if frozen is not None else fc.load())
    if groups is None:
        groups = groups_for(schema)                                  # L3: 14 groups; L3S: 15 (G13 added)
    groups = check_groups(schema, groups, complete=complete)
    if len(set(perm_seeds)) != len(perm_seeds) or not perm_seeds:
        raise ImportanceError("permutation seeds must be distinct and non-empty")
    manifest = evalp.manifest
    model_sha = model_sha256 or state_digest(adapter.broadcast_state())
    was_training = adapter.module.training
    adapter.module.eval()
    locs = {g: locate(schema, mem) for g, mem in groups.items()}
    store = ChannelStore([l for ls in locs.values() for l in ls])
    t0 = time.time()
    stats: Dict[str, UserStats] = {}
    try:
        with torch.no_grad():
            base_v, base_st, base_res, base_art = evaluate_stream(
                adapter, _tap(evalp.batches(frozen.eval_batch), store), manifest, frozen, model_sha, device)
        store.finish()
        stats["base"] = base_st
        users = np.unique(manifest.eval_user_key[manifest.eval_mask])
        plan = make_plan(users)
        log(f"[importance] {schema.schema_id} {run_id}: baseline {PRIMARY} macro E2E {base_v:.6f}; "
            f"{store.decision_id.size} stream rows, {plan.n_users} users, store {store.nbytes / 2 ** 20:.1f} MiB, "
            f"{time.time() - t0:.1f} s")
        rows = []
        for g, gl in locs.items():
            t1 = time.time()
            per = []
            for p in perm_seeds:
                donor = donor_map(store.lengths, p, g, alignment=alignment)
                cnt: dict = {}
                with torch.no_grad():
                    v, st, res, _ = evaluate_stream(adapter, permuted_batches(evalp.batches(frozen.eval_batch), store,
                                                                              gl, donor, cnt),
                                                    manifest, frozen, model_sha, device)
                stats[f"{g}|{p}"] = st
                per.append({"perm_seed": int(p), "perm_rng_seed": perm_seed(p, g), "donor_map_sha256":
                            map_sha256(donor), "value_permuted": v, "delta": v - base_v,
                            "ndcg@10_permuted": res["E2E"]["macro"]["ndcg@10"], **swap_diagnostics(store.lengths, donor),
                            **cnt})
            blk = importance_block(plan, base_st, [stats[f"{g}|{p}"] for p in perm_seeds])
            rows.append({"run_label": RUN_LABEL, "run_id": run_id, "schema_id": schema.schema_id,
                         "schema_self_sha256": schema.self_sha256, "group": g, "channels": [l.name for l in gl],
                         "droppable": g not in required_groups(groups),
                         "descriptive_only": g in required_groups(groups),      # DECISION_LOG 47: G1 is REQUIRED
                         "alignment": alignment, "metric": PRIMARY, "aggregation": "macro",
                         "target_denominator": "E2E", "importance": blk["importance"], "importance_ci95": blk["ci95"],
                         "within_noise": blk["within_noise"], "relative_importance": blk["relative_importance"],
                         "per_perm_seed": per, "value_unpermuted": base_v, "W_sha256": plan.W_sha256,
                         "n_users": plan.n_users, "n_decisions": base_st.n_decisions,
                         "wall_s": round(time.time() - t1, 2)})
            log(f"[importance] {g}: importance {blk['importance']:+.6f} CI {blk['ci95']} "
                f"{'within noise' if blk['within_noise'] else 'IMPORTANT'} ({time.time() - t1:.1f} s)")
    finally:
        adapter.module.train(was_training)
    baseline = {"run_label": RUN_LABEL, "run_id": run_id, "schema_id": schema.schema_id,
                "schema_self_sha256": schema.self_sha256, "metric": PRIMARY, "aggregation": "macro",
                "target_denominator": "E2E", "value": base_v, "ndcg@10": base_res["E2E"]["macro"]["ndcg@10"],
                "hr@10": base_res["E2E"]["macro"]["hr@10"], "n_decisions": base_st.n_decisions,
                "n_users": int(users.size), "manifest_id": manifest.manifest_id, "manifest_hashes": manifest.hashes,
                "model_sha256": model_sha, "eval_config": base_art.header.get("eval_config"),
                "evaluator_code_sha256": evaluator_code_sha256(), "W_sha256": plan.W_sha256}
    meta = {"perm_seeds": list(perm_seeds), "perm_tag": PERM_TAG, "alignment": alignment,
            "groups": {g: list(m) for g, m in groups.items()}, "store_bytes": store.nbytes,
            "stream_rows": int(store.decision_id.size), "store_width": store.width,
            "n_evaluations": 1 + len(groups) * len(perm_seeds), "wall_s": round(time.time() - t0, 2),
            "no_gradient": True, "frozen_eval_sha256": frozen.sha256}
    return ModelImportance(rows=rows, baseline=baseline, stats=stats, meta=meta)


# ------------------------------------------------------------------------------------------------ persistence
def save_stats(path: Path, stats: Mapping[str, UserStats], meta: Mapping) -> str:
    """FS_USERSTATS.npz: the per-user E2E sufficient statistics (S, N) of every evaluation of a run."""
    keys = sorted(stats)
    users = stats[keys[0]].user_keys
    arrays = {"user_keys": users}
    for k in keys:
        st = stats[k]
        if not np.array_equal(st.user_keys, users):
            raise ImportanceError("all statistics of one run must cover the same user set")
        arrays[f"S::{k}"] = st.S
        arrays[f"N::{k}"] = st.N
        arrays[f"SR::{k}"] = st.SR
        arrays[f"NR::{k}"] = st.NR
    arrays["meta_json"] = np.frombuffer(__import__("json").dumps(to_jsonable(dict(meta)), sort_keys=True)
                                        .encode("utf-8"), dtype=np.uint8)
    p = Path(path)
    tmp = p.with_name(p.name + ".tmp.npz")
    np.savez(tmp, **arrays)
    tmp.replace(p)
    return hashlib.sha256(p.read_bytes()).hexdigest()


def load_stats(path: Path) -> tuple:
    import json
    z = np.load(Path(path), allow_pickle=False)
    users = z["user_keys"]
    stats = {}
    for name in z.files:
        if name.startswith("S::"):
            k = name[3:]
            stats[k] = UserStats(users, z[f"S::{k}"], z[f"N::{k}"], z[f"SR::{k}"], z[f"NR::{k}"], k)
    meta = json.loads(bytes(z["meta_json"]).decode("utf-8"))
    return stats, meta


# ------------------------------------------------------------------------------------------------ family level
def combine_seeds(per_seed: Mapping[int, Mapping[str, UserStats]], groups: Iterable[str],
                  perm_seeds: Sequence[int] = PERM_SEEDS) -> dict:
    """Family-level importance over the L3 training seeds: point = mean_s importance_s; CI on the same W:
    per resample b, mean_s(base_{s,b} - mean_p perm_{s,p,b}); per-seed points and CIs recorded."""
    seeds = sorted(per_seed)
    if not seeds:
        raise ImportanceError("no seed")
    users = per_seed[seeds[0]]["base"].user_keys
    for s in seeds:
        if not np.array_equal(per_seed[s]["base"].user_keys, users):
            raise ImportanceError("the seeds cover different user sets (the paired bootstrap needs one W)")
    plan = make_plan(users)
    out = {}
    for g in groups:
        pts, res, per = [], [], {}
        for s in seeds:
            st = per_seed[s]
            perms = [st[f"{g}|{p}"] for p in perm_seeds]
            blk = importance_block(plan, st["base"], perms)
            per[str(s)] = {"importance": blk["importance"], "ci95": blk["ci95"], "within_noise": blk["within_noise"]}
            pts.append(blk["importance"])
            rb = resampled(plan, st["base"], STATISTIC)
            res.append(np.mean([rb - resampled(plan, x, STATISTIC) for x in perms], axis=0))
        ci = ci95(np.mean(res, axis=0))
        out[g] = {"importance": float(np.mean(pts)), "ci95": ci, "within_noise": bool(ci[0] <= 0.0 <= ci[1]),
                  "per_seed": per, "seeds": seeds, "W_sha256": plan.W_sha256, "n_users": plan.n_users}
    return out


def noise_groups(by_family: Mapping[str, Mapping[str, dict]]) -> list:
    """Groups within noise for EVERY family (the S* drops; DECISION: dropped only if within noise for BOTH). A group
    holding a REQUIRED channel (G1; DECISION_LOG 47) is never returned, whatever its importance."""
    fams = list(by_family)
    if len(fams) < 2:
        raise ImportanceError("the drop rule needs both families")
    groups = set.intersection(*(set(v) for v in by_family.values())) - set(required_groups())
    return [g for g in FS_GROUPS if g in groups and all(by_family[f][g]["within_noise"] for f in fams)] + \
        sorted(g for g in groups if g not in FS_GROUPS and all(by_family[f][g]["within_noise"] for f in fams))


__all__ = ["RUN_LABEL", "FS_GROUPS", "groups_for", "NEVER_PERMUTED", "REQUIRED_CHANNELS", "required_groups", "PERM_SEEDS", "ALIGNMENTS", "ImportanceError", "check_groups", "Loc", "locate",
           "perm_seed", "donor_map", "map_sha256", "swap_diagnostics", "ChannelStore", "permuted_batches",
           "window_consistency", "evaluate_stream", "stats_from_ranks", "importance_block", "ModelImportance",
           "grouped_permutation_importance", "save_stats", "load_stats", "combine_seeds", "noise_groups"]
