"""Secure-aggregation simulation for the federated simulator.

Protocol: Bonawitz et al. 2017 (semi-honest, four communication rounds) with the SecAgg+ k-regular graph option
(Bell et al. 2020). The cryptography is SIMULATED: key agreement, the pairwise seeds and the PRG masks are seeded
deterministic functions (sha256 / PCG64), nothing here is secure. Every size is COUNTED at its real value: 32-byte
public keys and secrets, Shamir shares over a 256-bit prime field (4-byte x + 32-byte y), AES-GCM framing (12-byte nonce
+ 16-byte tag) on the share ciphertexts, and mask vectors at the full update length.

Rounds (a client that drops stops after round 1; dropout happens before the masked input is sent):
  0 AdvertiseKeys   up: c_pk, s_pk (2 x 32 B)            down: the neighbours' key pairs (+ 4-byte id each)
  1 ShareKeys       up: one ciphertext per neighbour     down: the ciphertexts addressed to the client
                    (shares of s_sk and of the self-mask seed b; holders = neighbours + self)
  2 MaskedInput     up: 4 x d bytes (uint32 words) + 8-byte n_consumed header (as the plain FP32 upload)
                    down: the neighbour liveness list (4-byte ids)
  3 Unmasking       up: per holder entry, an s_sk share (dropped) or a b share (survivor): 4 + 36 bytes
Masked input: y_i = x_i + PRG(b_i) + sum_{j in N(i)} sign(i, j) PRG(s_ij)  (mod 2^32), sign = + iff i < j.
Server: sum_i y_i minus the survivors' self masks minus, for each dropped j, the surviving neighbours' pair masks
(s_j reconstructed from t shares; s_ij from the public keys). Failure iff survivors < t (t = ceil(2n/3)), or, in the
k-regular graph, some needed secret has fewer than t_share live shares (t_share = (k + 1) // 2 + 1). The consistency-check
round and the signatures of the active-adversary variant are not simulated (semi-honest model).

Fixed point: x -> round(clamp(x, -clip, clip) * scale) mod 2^32 (two's complement); scale = the largest power of two with
n * clip * scale < 2^31 (an L2 clip S implies |x_k| <= S), so the sum of n encoded updates never leaves the signed range.
Correctness criterion: the decoded sum is BIT-IDENTICAL to the plain fixed-point sum of the surviving clients' updates.
"""
from __future__ import annotations

import hashlib
import math
import random
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np

from .comm import UPLOAD_HEADER_BYTES

MOD32 = 1 << 32
PRIME = (1 << 256) - 189                  # the field of the 32-byte secrets (largest prime below 2^256)
KEY_BYTES = 32
SHARE_BYTES = 4 + 32                      # x-coordinate + y
ID_BYTES = 4
AEAD_BYTES = 12 + 16                      # nonce + tag
CT_BYTES = 2 * ID_BYTES + 2 * SHARE_BYTES + AEAD_BYTES     # ids, the s_sk and b shares, framing
UNMASK_ENTRY_BYTES = ID_BYTES + SHARE_BYTES
WORD_BYTES = 4
HEADER_BYTES = UPLOAD_HEADER_BYTES        # n_consumed, as in a plain upload
ROUNDS = 4
PROTOCOL = "BONAWITZ2017_SEMIHONEST_4R+SECAGG_PLUS_KREGULAR"


class SecAggError(RuntimeError):
    pass


def _h(*parts) -> bytes:
    m = hashlib.sha256()
    for p in parts:
        m.update(p if isinstance(p, bytes) else str(p).encode())
        m.update(b"|")
    return m.digest()


def threshold(n: int) -> int:
    return -(-2 * int(n) // 3)


# ---------------------------------------------------------------- fixed point

def fixed_point_scale(clip: float, n: int) -> float:
    if not (clip > 0 and math.isfinite(clip)) or n < 1:
        raise ValueError("clip must be finite positive and n >= 1")
    e = math.floor(math.log2((2 ** 31 - 1) / (n * clip)))
    while n * clip * 2.0 ** e >= 2 ** 31 - 1:
        e -= 1
    return 2.0 ** e


def encode_fixed(x: np.ndarray, scale: float, clip: float) -> np.ndarray:
    """float -> uint32 two's complement of round(clamp(x, -clip, clip) * scale)."""
    q = np.rint(np.clip(np.asarray(x, dtype=np.float64), -clip, clip) * scale).astype(np.int64)
    return (q % MOD32).astype(np.uint32)


def decode_fixed(total: np.ndarray, scale: float) -> np.ndarray:
    return total.astype(np.int32).astype(np.float64) / scale        # uint32 read as signed int32 (mod 2^32)


# ---------------------------------------------------------------- simulated primitives

def prg_mask(seed: bytes, d: int) -> np.ndarray:
    return np.random.Generator(np.random.PCG64(int.from_bytes(seed, "big"))).integers(0, MOD32, size=d, dtype=np.uint32)


def shamir_split(secret: int, xs: Sequence[int], t: int, rng: random.Random) -> list:
    coef = [secret % PRIME] + [rng.getrandbits(256) % PRIME for _ in range(t - 1)]
    out = []
    for x in xs:
        y = 0
        for c in reversed(coef):
            y = (y * x + c) % PRIME
        out.append((x, y))
    return out


def shamir_reconstruct(shares: Sequence, t: int) -> int:
    if len(shares) < t:
        raise SecAggError(f"{len(shares)} shares < threshold {t}")
    pts = list(shares)[:t]
    s = 0
    for i, (xi, yi) in enumerate(pts):
        num = den = 1
        for j, (xj, _) in enumerate(pts):
            if i != j:
                num = num * (-xj) % PRIME
                den = den * (xi - xj) % PRIME
        s = (s + yi * num * pow(den, -1, PRIME)) % PRIME
    return s


def build_graph(n: int, k: int | None, seed: int) -> list:
    """Neighbour lists (sorted). k None / >= n-1: the complete graph; else a seeded k-regular circulant graph on a
    random permutation of the clients (k even, k < n)."""
    if k is None or k >= n - 1:
        return [[j for j in range(n) if j != i] for i in range(n)]
    if k < 2 or k % 2:
        raise ValueError("k must be even and >= 2")
    perm = list(range(n))
    random.Random(int.from_bytes(_h("graph", seed, n, k), "big")).shuffle(perm)
    nb = [set() for _ in range(n)]
    for pos, v in enumerate(perm):
        for m in range(1, k // 2 + 1):
            u = perm[(pos + m) % n]
            nb[v].add(u)
            nb[u].add(v)
    return [sorted(s) for s in nb]


# ---------------------------------------------------------------- the round

@dataclass(frozen=True)
class SecAggConfig:
    n: int = 64
    d: int = 1024
    k: int | None = None               # None: Bonawitz (full graph); even k: SecAgg+ k-regular
    clip: float = 1.0
    seed: int = 0
    t: int | None = None               # global threshold; None -> ceil(2n/3)

    @property
    def t_global(self) -> int:
        return threshold(self.n) if self.t is None else int(self.t)

    @property
    def full(self) -> bool:
        return self.k is None or self.k >= self.n - 1

    @property
    def variant(self) -> str:
        return "full" if self.full else f"kreg{self.k}"

    @property
    def t_share(self) -> int:
        """Shamir threshold within one holder set (neighbours + self)."""
        return self.t_global if self.full else (int(self.k) + 1) // 2 + 1


def plain_upload_bytes(d: int) -> int:
    return WORD_BYTES * d + HEADER_BYTES


def sample_dropped(n: int, p: float, seed: int) -> list:
    rng = random.Random(int.from_bytes(_h("drop", seed, n, repr(float(p))), "big"))
    return sorted(i for i in range(n) if rng.random() < p)


@dataclass
class SecAggResult:
    success: bool
    failure_reason: str | None
    survivors: list
    dropped: list
    total: np.ndarray | None = None            # unmasked uint32 sum (None on failure)
    plain: np.ndarray | None = None            # plain fixed-point sum of the survivors (check value)
    bit_identical: bool | None = None
    rounds: int = ROUNDS
    unmask_cpu_s: float = 0.0                     # server: share reconstruction + mask regeneration + subtraction
    client_cpu_s: float = 0.0                     # all simulated clients together (setup + masking)
    bytes_up: dict = field(default_factory=dict)  # round -> mean bytes over the clients that took part in the round
    bytes_down: dict = field(default_factory=dict)
    prg_words_server: int = 0


def per_client_bytes(cfg: SecAggConfig, nbrs: list, survivors: set, i: int) -> dict:
    """Real-size bytes client i sends / receives per round (a dropped client stops after round 1)."""
    deg = len(nbrs[i])
    up = {0: 2 * KEY_BYTES, 1: deg * CT_BYTES}
    dn = {0: deg * (2 * KEY_BYTES + ID_BYTES), 1: deg * CT_BYTES}
    if i in survivors:
        up[2] = WORD_BYTES * cfg.d + HEADER_BYTES
        dn[2] = ID_BYTES * deg
        up[3] = (deg + 1) * UNMASK_ENTRY_BYTES
        dn[3] = 0
    return {"up": up, "down": dn}


def run_round(cfg: SecAggConfig, enc_fn: Callable[[int], np.ndarray], dropped: Sequence[int],
              check_plain: bool = True) -> SecAggResult:
    """One secure-aggregation round. enc_fn(i) -> client i's fixed-point update (uint32, length d). `dropped` clients
    send keys and shares, then vanish before the masked input."""
    n, d = cfg.n, cfg.d
    drop = {int(i) for i in dropped}
    surv = [i for i in range(n) if i not in drop]
    alive = set(surv)
    nbrs = build_graph(n, cfg.k, cfg.seed)
    t_sh = cfg.t_share
    t0 = time.process_time()
    sk, bsec, pk = [], [], []
    for i in range(n):
        s = int.from_bytes(_h("sk", cfg.seed, i), "big") % PRIME
        b = int.from_bytes(_h("b", cfg.seed, i), "big") % PRIME
        sk.append(s)
        bsec.append(b)
        pk.append(_h("pk", s.to_bytes(32, "big")))
    rng = random.Random(int.from_bytes(_h("shamir", cfg.seed), "big"))
    holders = [sorted(nbrs[i] + [i]) for i in range(n)]
    sh_s, sh_b = [], []                          # per client: {holder: (x, y)}
    for i in range(n):
        xs = [h + 1 for h in holders[i]]
        sh_s.append(dict(zip(holders[i], shamir_split(sk[i], xs, t_sh, rng))))
        sh_b.append(dict(zip(holders[i], shamir_split(bsec[i], xs, t_sh, rng))))

    def pair_seed(a: int, b: int) -> bytes:
        lo, hi = (a, b) if a < b else (b, a)
        return _h("ka", pk[lo], pk[hi])

    y_sum = np.zeros(d, dtype=np.uint32)
    plain = np.zeros(d, dtype=np.uint32) if check_plain else None
    for i in surv:
        x = enc_fn(i)
        if x.dtype != np.uint32 or x.shape != (d,):
            raise SecAggError("enc_fn must return uint32 of length d")
        y = x + prg_mask(_h("self", bsec[i].to_bytes(32, "big")), d)
        for j in nbrs[i]:
            m = prg_mask(pair_seed(i, j), d)
            y = y + m if i < j else y - m
        y_sum += y
        if plain is not None:
            plain += x
    res = SecAggResult(False, None, surv, sorted(drop), client_cpu_s=time.process_time() - t0)
    acc = [per_client_bytes(cfg, nbrs, alive, i) for i in range(n)]
    for r in range(ROUNDS):
        for dirn, store in (("up", res.bytes_up), ("down", res.bytes_down)):
            vals = [a[dirn][r] for a in acc if r in a[dirn]]
            store[r] = (sum(vals) / len(vals)) if vals else 0.0
    if len(surv) < cfg.t_global:
        res.failure_reason = f"survivors {len(surv)} < t {cfg.t_global}"
        return res
    c0 = time.process_time()
    total = y_sum.copy()
    words = 0
    try:
        for i in surv:                           # self masks: b_i from the live holders' shares
            b = shamir_reconstruct([sh_b[i][h] for h in holders[i] if h in alive], t_sh)
            total -= prg_mask(_h("self", b.to_bytes(32, "big")), d)
            words += d
        for j in sorted(drop):                   # dropped clients: s_j, then the survivors' pair masks
            s = shamir_reconstruct([sh_s[j][h] for h in holders[j] if h in alive], t_sh)
            if _h("pk", s.to_bytes(32, "big")) != pk[j]:
                raise SecAggError("reconstructed key does not match the advertised key")
            for i in nbrs[j]:
                if i in alive:
                    m = prg_mask(pair_seed(i, j), d)
                    total += m if i > j else -m   # undo sign(i, j) = + iff i < j
                    words += d
    except SecAggError as e:
        res.failure_reason = f"unmasking: {e}"
        res.unmask_cpu_s = time.process_time() - c0
        return res
    res.unmask_cpu_s = time.process_time() - c0
    res.prg_words_server = words
    res.success = True
    res.total = total
    res.plain = plain
    if plain is not None:
        res.bit_identical = bool(np.array_equal(total, plain))
    return res


def cost_summary(cfg: SecAggConfig, res: SecAggResult) -> dict:
    """Bytes of a client that survives all rounds, and the ratio to a plain upload (up only; up + down)."""
    plain = plain_upload_bytes(cfg.d)
    up = sum(res.bytes_up.values())
    dn = sum(res.bytes_down.values())
    return {"plain_upload_bytes": plain,
            "bytes_up_by_round": {str(r): res.bytes_up[r] for r in res.bytes_up},
            "bytes_down_by_round": {str(r): res.bytes_down[r] for r in res.bytes_down},
            "secagg_up_bytes_survivor": up, "secagg_down_bytes_survivor": dn,
            "overhead_ratio_up": up / plain, "overhead_ratio_up_down": (up + dn) / plain, "rounds": res.rounds}
