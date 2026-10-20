"""census_v2.core (owner: Data Lead, CENSUS-02) - pure, data-free helpers shared by the census job and its unit tests.

Strata (OWNER_AMENDMENT_POP_5_100 section 1/3; CENSUS-02 "Definitions (binding)"), n = n_eligible_TRAIN_examples:
    A0  TRUE COLD, no TRAIN event at all (the user appears only in VALIDATION)
    A1  TRUE COLD, has TRAIN events but n = 0
    B   SPARSE / FEW-SHOT, 1 <= n <= 4
    C   MAIN, 5 <= n <= 100 (inclusive)
    D   HEAVY, n > 100
Selection key v2 (CENSUS-02 deliverable 2): big-endian uint64 of the first 8 bytes of
    sha256(b"PPSI_V4_POP_5_100_SEL_20260926|" + digest32),
where digest32 is the CLIENT_CONTRACT C1.2 pseudonym digest sha256(SALT + b"|" + user_id) (v4data.common, SALT
unchanged). The cohort of size N is the N smallest keys of the stratum-C pool, so every cohort is a prefix of one
fixed ascending order (nested by construction; verified by the caller).
"""
from __future__ import annotations

import hashlib
import math
from typing import Iterable, Sequence

import numpy as np

STRATA = ("A0", "A1", "B", "C", "D")
S_A0, S_A1, S_B, S_C, S_D = 0, 1, 2, 3, 4
B_LO, B_HI = 1, 4
C_LO, C_HI = 5, 100

SEL_PREFIX_V2 = b"PPSI_V4_POP_5_100_SEL_20260926|"

QUANTILE_POINTS = [0, 10, 25, 50, 75, 90, 95, 99, 100]
QUANTILE_LABELS = ["min", "p10", "p25", "p50", "p75", "p90", "p95", "p99", "max"]

CANDIDATE_N = (32768, 65536, 131072, 262144)          # plus "ALL_C"
CALIB_FIXED = (8192, 16384, 32768, 65536)
CALIB_FRACTIONS = ((1, 16), (1, 8), (1, 4))


def stratum_codes(n: np.ndarray, ntrain: np.ndarray) -> np.ndarray:
    """Stratum code per user from its eligible-example count n and its TRAIN decision count."""
    n = np.asarray(n, dtype=np.int64)
    ntrain = np.asarray(ntrain, dtype=np.int64)
    if n.shape != ntrain.shape:
        raise ValueError("n and ntrain must have the same shape")
    if np.any(n < 0) or np.any(ntrain < 0):
        raise ValueError("negative count")
    if np.any((n > 0) & (ntrain == 0)):
        raise ValueError("a user with eligible TRAIN examples but no TRAIN decision")
    if np.any(n > ntrain):
        raise ValueError("n_eligible exceeds the TRAIN decision count")
    out = np.full(n.shape, -1, dtype=np.int8)
    out[(n == 0) & (ntrain == 0)] = S_A0
    out[(n == 0) & (ntrain > 0)] = S_A1
    out[(n >= B_LO) & (n <= B_HI)] = S_B
    out[(n >= C_LO) & (n <= C_HI)] = S_C
    out[n > C_HI] = S_D
    if np.any(out < 0):
        raise AssertionError("a user matched no stratum")
    return out


def selection_key_v2(digest32: bytes) -> int:
    if not isinstance(digest32, (bytes, bytearray)) or len(digest32) != 32:
        raise ValueError("digest32 must be exactly 32 bytes")
    return int.from_bytes(hashlib.sha256(SEL_PREFIX_V2 + bytes(digest32)).digest()[:8], "big")


def selection_keys_v2(digests: Iterable[bytes]) -> np.ndarray:
    rows = list(digests)
    out = np.empty(len(rows), dtype=np.uint64)
    sha = hashlib.sha256
    for i, d in enumerate(rows):
        if len(d) != 32:
            raise ValueError("digest32 must be exactly 32 bytes")
        out[i] = int.from_bytes(sha(SEL_PREFIX_V2 + d).digest()[:8], "big")
    return out


def quantiles(arr) -> dict:
    a = np.asarray(arr, dtype=np.float64)
    if a.size == 0:
        return {lbl: None for lbl in QUANTILE_LABELS} | {"mean": None}
    vals = np.percentile(a, QUANTILE_POINTS, method="linear")
    out = {lbl: float(v) for lbl, v in zip(QUANTILE_LABELS, vals)}
    out["mean"] = float(a.mean())
    return out


def share(num, den):
    return (float(num) / float(den)) if den else None


def gini(x) -> float | None:
    """Gini coefficient of a non-negative vector: (2 * sum_i i * x_(i)) / (n * sum x) - (n + 1) / n, x ascending."""
    a = np.sort(np.asarray(x, dtype=np.float64))
    n = a.size
    if n == 0 or a.sum() == 0:
        return None
    if np.any(a < 0):
        raise ValueError("gini needs non-negative values")
    i = np.arange(1, n + 1, dtype=np.float64)
    return float((2.0 * np.sum(i * a)) / (n * a.sum()) - (n + 1.0) / n)


def top_share(x, frac: float) -> dict:
    """Share of the total held by the top ceil(frac * n) clients (largest first)."""
    a = np.sort(np.asarray(x, dtype=np.int64))[::-1]
    n = a.size
    k = int(math.ceil(frac * n)) if n else 0
    tot = int(a.sum())
    return {"top_fraction": frac, "top_k_clients": k, "examples_held": int(a[:k].sum()), "share": share(a[:k].sum(), tot)}


def calib_sizes(N: int) -> list[tuple[str, int]]:
    """Calibration-prefix candidates for a cohort of N users: N/16, N/8, N/4 (floor), and the fixed sizes
    8192, 16384, 32768, 65536 where they are <= N/2. Returned as (label, size), size ascending, de-duplicated by
    label (a fraction and a fixed size may coincide in value; both labels are kept)."""
    out = []
    for num, den in CALIB_FRACTIONS:
        out.append((f"N/{den // num}", (N * num) // den))
    for s in CALIB_FIXED:
        if s <= N / 2:
            out.append((f"fixed_{s}", s))
    return sorted(out, key=lambda t: (t[1], t[0]))


def fl_local_work(n: np.ndarray, batch: int, passes: int, clients_per_round: int = 64) -> dict:
    """Analytic FL local work for one client population (package 06: every visit runs `passes` passes over the
    client's own examples at local batch `batch`, the last partial batch at its own size, so a pass costs ceil(n/B)
    optimizer steps). EFE = exposures / N_decisions with exposures = passes * n per visit, so one full sweep (every
    client visited once, SMALLER_FINAL_GROUP) is exactly `passes` EFE."""
    n = np.asarray(n, dtype=np.int64)
    if n.size == 0 or np.any(n < 1):
        raise ValueError("every client needs >= 1 example")
    per_pass = -(-n // batch)
    steps = passes * per_pass
    N = int(n.size)
    tot_ex = int(n.sum())
    sweep_steps = int(steps.sum())
    rounds_per_sweep = -(-N // clients_per_round)
    vals, cnts = np.unique(steps, return_counts=True)
    cdf = np.cumsum(cnts) / N
    prev = np.concatenate([[0.0], cdf[:-1]])
    m = clients_per_round
    exp_max = float(np.sum(vals * (cdf ** m - prev ** m)))
    return {
        "local_batch": int(batch), "passes_per_visit": int(passes), "clients": N,
        "examples_per_EFE": tot_ex,
        "per_client_steps_per_visit_quantiles": quantiles(steps),
        "total_client_steps_per_sweep": sweep_steps,
        "EFE_per_sweep": int(passes),
        "total_client_steps_per_EFE": sweep_steps / passes,
        "total_client_steps_to_6_EFE": 6 * sweep_steps / passes,
        "clients_per_round": m,
        "mean_client_steps_per_round": float(m * steps.mean()),
        "expected_max_client_steps_in_a_round": exp_max,
        "expected_max_definition": "E[max of the round's per-client steps] for 64 clients drawn uniformly WITH "
                                   "replacement from the empirical distribution (analytic approximation of the "
                                   "straggler length; the plan draws without replacement)",
        "rounds_per_sweep": int(rounds_per_sweep),
        "rounds_per_EFE_at_64_clients_per_round": rounds_per_sweep / passes,
        "rounds_to_6_EFE_at_64_clients_per_round": 6 * rounds_per_sweep / passes,
        "rounds_per_EFE_if_all_clients_every_round": 1.0 / passes,
        "fraction_clients_data_fits_one_batch": share(int((n <= batch).sum()), N),
        "fraction_clients_with_partial_last_batch": share(int((n % batch != 0).sum()), N),
        "mean_examples_per_optimizer_step": tot_ex * passes / sweep_steps,
    }


def weight_concentration(n: np.ndarray) -> dict:
    n = np.asarray(n, dtype=np.int64)
    return {"gini_of_n": gini(n), "top_1pct": top_share(n, 0.01), "top_10pct": top_share(n, 0.10),
            "note": "FedAvg / FedProx weights by consumed examples are passes * n, so these shares apply unchanged"}


def head_sizes(K_classes: int, inv: dict) -> dict:
    """Model-head size at K output classes from a MODEL-01 inventory record (tied GRU / untied SASRec)."""
    fam = inv["family"]
    pi = inv["parameter_inventory"]
    K0 = int(inv["K"])
    if fam == "GRU":
        d = int(inv["topology"]["item_dim"])
        if pi["item_embed"] != (K0 + 3) * d or pi["output_bias"] != K0:
            raise ValueError("GRU inventory does not have the tied (K+3) x d item table + K bias layout")
        fixed = int(pi["total"]) - (K0 + 3) * d - K0
        head = K_classes * d + K_classes
        total_head_only = fixed + (K0 + 3) * d + K_classes            # tied: the input table keeps K0 + 3 rows
        total_head_and_input = fixed + (K_classes + 3) * d + K_classes
        w_out = {"rows": K_classes, "width": d, "tied_to_input_item_table": True}
    elif fam == "SASREC":
        d = int(inv["topology"]["registry_entry"]["d_model"])
        if pi["output_embed"] != K0 * d or pi["output_bias"] != K0 or pi["item_embed"] != (K0 + 3) * d:
            raise ValueError("SASRec inventory does not have the untied K x d head + (K+3) x d input layout")
        fixed = int(pi["total"]) - pi["item_embed"] - pi["output_embed"] - K0
        head = K_classes * d + K_classes
        total_head_only = fixed + (K0 + 3) * d + K_classes * d + K_classes
        total_head_and_input = fixed + (K_classes + 3) * d + K_classes * d + K_classes
        w_out = {"rows": K_classes, "width": d, "tied_to_input_item_table": False}
    else:
        raise ValueError(f"unknown family {fam}")
    mb = lambda p: {"params": int(p), "bytes_fp32": int(4 * p), "MB_fp32": 4 * p / 1e6, "MiB_fp32": 4 * p / 2 ** 20}  # noqa: E731
    return {"family": fam, "K_classes": int(K_classes), "W_out": w_out, "head_params_W_out_plus_bias": mb(head),
            "total_shared_params_head_only_change": mb(total_head_only),
            "total_shared_params_head_and_input_vocab_change": mb(total_head_and_input),
            "fl_visit_bytes_send_plus_receive_head_only_change": int(2 * 4 * total_head_only),
            "fl_visit_bytes_send_plus_receive_head_and_input_change": int(2 * 4 * total_head_and_input)}


def assert_nested(keys_sorted: np.ndarray, sizes: Sequence[int]) -> bool:
    """Every prefix is a prefix of every larger one and the key order is strictly ascending (unique)."""
    k = np.asarray(keys_sorted, dtype=np.uint64)
    if k.size > 1 and not np.all(k[1:] > k[:-1]):
        return False
    ss = sorted(int(s) for s in sizes)
    prev = None
    for s in ss:
        cur = k[:s]
        if prev is not None and not np.array_equal(cur[:prev.size], prev):
            return False
        prev = cur
    return True
