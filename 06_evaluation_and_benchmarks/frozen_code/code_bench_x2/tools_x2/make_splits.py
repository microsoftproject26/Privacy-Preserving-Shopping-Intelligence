"""X2 A1 (DL 553; POST_HOC_EXPLORATORY): the disjoint user splits S (server band) / F (FL clients). Nothing here trains.

  python -B tools_x2/make_splits.py [--out <runs_bench_x/x2_A1/splits>] [--datasets "..."] [--fracs 0.15 0.5]

Rule (deterministic, seeded, recorded): the permutation of the registered 15 % band, i.e. PCG64(SeedSequence([BAND_TAG,
2026])).permutation(U) over the SORTED external user ids (data.band_users); S_f = the first floor(f U + 0.5) users of
that permutation, F_f = the rest. Hence S_0.15 == the registered pretraining band exactly (checked) and S_0.15 is a
subset of S_0.5 (nested; checked). One split per (dataset, f), shared by both training seeds (2026 / 2027).
Each file is write-once (O_EXCL) and carries self_sha256 = sha256 of its canonical JSON without that field.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

CODE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, CODE)

from v4bench import common, data as D  # noqa: E402

DATASETS = ("s3_beauty", "s3_sports", "s3_toys", "ml_1m")
FRACS = (0.15, 0.5)


def split_id(frac: float) -> str:
    return f"DJ{int(round(frac * 100)):02d}"


def build(ds: str, frac: float) -> dict:
    d = D.load(ds, common.default_data_root(ds))
    u = d.user_ids
    perm = np.random.Generator(np.random.PCG64(np.random.SeedSequence([common.BAND_TAG, common.BAND_SEED]))
                               ).permutation(u.size)
    n = int(np.floor(frac * u.size + 0.5))
    S = np.sort(u[perm[:n]])
    F = np.sort(u[perm[n:]])
    assert np.intersect1d(S, F).size == 0 and S.size + F.size == u.size
    reg = np.sort(u[d.band])
    obj = {"format": D.SPLIT_FORMAT, "tag": common.X2_TAG, "study": "A1_DISJOINT_PRE", "authority": "DECISION_LOG 553",
           "dataset": ds, "train_sha256": d.train_sha256, "split_id": split_id(frac), "frac_S": float(frac),
           "rule": "PCG64(SeedSequence([BAND_TAG=0x42414E44, 2026])).permutation(U) over the sorted external user ids "
                   "(= v4bench.data.band_users); S = first floor(frac U + 0.5), F = the rest",
           "seed": common.BAND_SEED, "n_users": int(u.size), "n_S": int(S.size), "n_F": int(F.size),
           "S_sha256": common.canon_sha256(S.tolist()), "F_sha256": common.canon_sha256(F.tolist()),
           "registered_band_sha256": d.band_sha256(),
           "S_equals_registered_band": bool(np.array_equal(S, reg)),
           "registered_band_subset_of_S": bool(np.isin(reg, S).all()),
           "S_train_decisions": int(sum(d.n_valid(i) for i in np.flatnonzero(np.isin(u, S)))),
           "F_train_decisions": int(sum(d.n_valid(i) for i in np.flatnonzero(np.isin(u, F)))),
           "S_users": S.tolist(), "F_users": F.tolist(), "utc": common.utc()}
    if abs(frac - common.BAND_FRAC) < 1e-12 and not obj["S_equals_registered_band"]:
        raise SystemExit(f"{ds}: S_0.15 differs from the registered band")
    if frac > common.BAND_FRAC and not obj["registered_band_subset_of_S"]:
        raise SystemExit(f"{ds}: the registered band is not inside S_{frac}")
    obj["self_sha256"] = common.canon_sha256(D.split_users_body(obj))
    return obj


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(common.V4 / "runs_bench_x" / "x2_A1" / "splits"))
    ap.add_argument("--datasets", default=" ".join(DATASETS))
    ap.add_argument("--fracs", nargs="+", type=float, default=list(FRACS))
    a = ap.parse_args()
    if "runs_bench_x" not in os.path.abspath(a.out).replace(os.sep, "/").split("/"):
        print("REFUSED: --out must be under runs_bench_x", file=sys.stderr)
        return 2
    for ds in a.datasets.split():
        for f in a.fracs:
            p = os.path.join(a.out, f"{ds}__{split_id(f)}.json")
            if os.path.exists(p):
                print(f"exists (kept): {p}")
                continue
            obj = build(ds, f)
            common.write_exclusive(p, obj)
            print(f"{p}: n_S {obj['n_S']} n_F {obj['n_F']} S_dec {obj['S_train_decisions']} "
                  f"F_dec {obj['F_train_decisions']} self {obj['self_sha256'][:12]} "
                  f"eq_band {obj['S_equals_registered_band']} file {common.sha256_file(p)[:12]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
