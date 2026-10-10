# Secure aggregation (`ppsi.fedsim.secagg`)

In a federated round the server only needs the *sum* of the client updates. Secure aggregation lets it learn that
sum without seeing any single update. `ppsi.fedsim.secagg` simulates the protocol of Bonawitz et al. (2017) with
the sparse-graph option of Bell et al. (2020, "SecAgg+"), so its cost and its behaviour under dropout can be measured
at the real update size of the model.

## What is simulated, and what is real

* **Simulated:** key agreement, pairwise seeds and pseudo-random masks are deterministic functions of a seed
  (SHA-256 and PCG64). Nothing here is cryptographically secure, so do not use it to protect real data.
* **Real:** every message is counted at its real size. That covers 32-byte keys, Shamir shares over a 256-bit prime
  field, AES-GCM framing on the share ciphertexts, and the masked vector at the full update length.
* **Threat model:** semi-honest. The consistency check and the signatures of the active-adversary variant are not
  simulated.

## Protocol

Four rounds. A client that drops out stops after round 1.

| Round | Client sends | Client receives |
|---|---|---|
| 0 Advertise keys | two public keys | its neighbours' public keys |
| 1 Share keys | Shamir shares of its secret key and its self-mask seed, one ciphertext per neighbour | the ciphertexts addressed to it |
| 2 Masked input | the update plus its self mask and the pairwise masks (mod 2^32), the same size as a plain upload | the list of live neighbours |
| 3 Unmasking | shares that let the server remove the masks (self-mask shares of survivors, key shares of dropped clients) | - |

* **Encoding.** Updates are encoded in fixed point: clamped to the clip bound, scaled by the largest power of two
  that keeps the sum of `n` updates inside the signed 32-bit range, and summed modulo 2^32.
* **Complete graph.** The round succeeds if at least `t = ceil(2n / 3)` clients survive.
* **k-regular graph.** Each secret is shared only among a client's `k` neighbours and itself, so the round also
  needs `(k + 1) // 2 + 1` live holders for every secret it must reconstruct.
* **Failure.** A failed round is reported with its reason; it never returns a wrong sum.

## Usage

```python
from ppsi.fedsim import secagg as sa

cfg = sa.SecAggConfig(n=64, d=263822, k=16, clip=0.28, seed=2026)   # k=None: complete graph
scale = sa.fixed_point_scale(cfg.clip, cfg.n)
result = sa.run_round(cfg, lambda i: sa.encode_fixed(update_of(i), scale, cfg.clip),
                      dropped=sa.sample_dropped(cfg.n, 0.1, cfg.seed))
print(result.success, result.bit_identical, sa.cost_summary(cfg, result)["overhead_ratio_up"])
```

The grid command runs the protocol over graphs, dropout rates, seeds and update lengths with synthetic clipped
updates. It writes one JSON report and exits 1 if any decoded sum differs from the plain sum:

```powershell
uv run python -m ppsi.fedsim.secagg_grid --out secagg.json --update-dim 263822 20708708
```

## Results for the device model

For 64 clients, dropout 0 / 10 / 30 %, a complete graph and k = 16 / 32, and three seeds:

* Every decoded sum is bit-identical to the plain sum.
* The upload overhead is under 1 % for the 263,822-value encoder update and about 0.01 % for the full 20.7 M-value
  model.
* One cell per update length fails to decode, reported as a failure: k = 16 at 30 % dropout, where a dropped
  client's key has 8 live shares against a threshold of 9.

The aggregated table is `results/rees46/secure_aggregation.csv`.

## Tests

```powershell
uv run python -m pytest tests/fedsim/test_fedsim_secagg.py tests/fedsim/test_fedsim_secagg_grid.py -q
```

They cover:

* bit-identical sums under dropout, with edge values (exactly at the clip bound, rounding ties);
* threshold failures;
* Shamir round trips;
* graph regularity;
* determinism;
* the byte accounting of every round.
