"""BM-02a: standard benchmark preprocessing (the published protocols), independent of the PPSI pipeline.

Per dataset → bench_proc/<Dataset>/:
- sequences.parquet: user_idx, item_idx, ts, pos (chronological; ties keep file order = stable sort);
- split.json: protocol;
- items.parquet: item_idx, raw id, side channels;
- STATS.json: users / items / interactions / avg length, plus the published stats and match flags;
- SHA256SUMS.

Protocols:
- MovieLens-1M: all ratings as implicit, iterative 5-core → SASRec / SASRec+ stats 6,040 / 3,416 / 999,611; leave-one-out.
- Amazon-Beauty-2014: reviews_Beauty_5 (already 5-core) → S3-Rec stats 22,363 / 12,101 / 198,502; leave-one-out; side
  channels: categories, brand.
- Amazon2023-<cat>: the official 5-core benchmark files, kept as given (last_out_w_his and timestamp_w_his); side channels:
  main_category, categories, store.
"""
import gzip, hashlib, json, sys, zipfile, ast
from pathlib import Path
import pandas as pd

RAW = Path(sys.argv[1]); OUT = Path(sys.argv[2]); OUT.mkdir(exist_ok=True)


def kcore(df, k=5):
    while True:
        n = len(df)
        df = df[df.groupby("item")["user"].transform("size") >= k]
        df = df[df.groupby("user")["item"].transform("size") >= k]
        if len(df) == n:
            return df


def finish(name, df, side, published, protocol):
    d = OUT / name; d.mkdir(exist_ok=True)
    df = df.sort_values(["user", "ts"], kind="stable")
    umap = {u: i + 1 for i, u in enumerate(pd.unique(df.user))}   # 0 = PAD
    imap = {x: i + 1 for i, x in enumerate(sorted(pd.unique(df.item)))}
    seq = pd.DataFrame({"user_idx": df.user.map(umap).astype("int64"), "item_idx": df.item.map(imap).astype("int64"),
                        "ts": df.ts.astype("int64")})
    seq["pos"] = seq.groupby("user_idx").cumcount()
    seq.to_parquet(d / "sequences.parquet", index=False)
    it = pd.DataFrame({"item_idx": list(imap.values()), "raw_id": list(imap.keys())})
    if side is not None:
        it = it.merge(side, how="left", left_on="raw_id", right_index=True)
    it.to_parquet(d / "items.parquet", index=False)
    (d / "user_map.json").write_text(json.dumps({str(k): v for k, v in umap.items()}))
    st = {"users": len(umap), "items": len(imap), "interactions": int(len(seq)),
          "avg_len": round(len(seq) / len(umap), 2), "published": published,
          "matches_published": {k: st_v == published.get(k) for k, st_v in
                                (("users", len(umap)), ("items", len(imap)), ("interactions", int(len(seq))))
                                if k in published},
          "side_coverage": {c: float(it[c].notna().mean()) for c in it.columns if c not in ("item_idx", "raw_id")}}
    (d / "split.json").write_text(json.dumps(protocol, indent=1))
    (d / "STATS.json").write_text(json.dumps(st, indent=1))
    with open(d / "SHA256SUMS", "w") as f:
        for p in sorted(d.iterdir()):
            if p.name != "SHA256SUMS":
                f.write(f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.name}\n")
    print(name, json.dumps({k: st[k] for k in ("users", "items", "interactions", "avg_len", "matches_published")}))


LOO = {"split": "leave-one-out", "test": "last item per user", "valid": "second-last item per user",
       "train": "all earlier items", "ranking": "full catalogue (main); sampled-100 secondary per compared paper"}

# MovieLens-1M
with zipfile.ZipFile(RAW / "MovieLens-1M/ml-1m.zip") as z:
    r = pd.read_csv(z.open("ml-1m/ratings.dat"), sep="::", engine="python", names=["user", "item", "rating", "ts"])
    mv = pd.read_csv(z.open("ml-1m/movies.dat"), sep="::", engine="python", names=["item", "title", "genres"],
                     encoding="latin-1")
ml = kcore(r[["user", "item", "ts"]], 5)
finish("MovieLens-1M", ml, mv.set_index("item")[["genres"]],
       {"users": 6040, "items": 3416, "interactions": 999611, "source": "SASRec (Kang & McAuley 2018); SASRec+ (2023)"},
       dict(LOO, filter="all ratings implicit; iterative 5-core", max_len_standard=200))

# Amazon Beauty 2014 (S3-Rec version)
rows = [json.loads(l) for l in gzip.open(RAW / "Amazon-Beauty-2014/reviews_Beauty_5.json.gz")]
be = pd.DataFrame({"user": [x["reviewerID"] for x in rows], "item": [x["asin"] for x in rows],
                   "ts": [x["unixReviewTime"] for x in rows]})
meta = {}
for l in gzip.open(RAW / "Amazon-Beauty-2014/meta_Beauty.json.gz"):
    m = ast.literal_eval(l.decode())                                   # the 2014 meta file is python-literal lines
    cats = m.get("categories") or [[]]
    meta[m["asin"]] = {"categories": "|".join(cats[0]) if cats and cats[0] else None, "brand": m.get("brand")}
finish("Amazon-Beauty-2014", be, pd.DataFrame.from_dict(meta, orient="index"),
       {"users": 22363, "items": 12101, "interactions": 198502, "source": "S3-Rec (Zhou et al. 2020); CL4SRec, DuoRec, ICLRec, DIF-SR"},
       dict(LOO, filter="5-core (as distributed)", max_len_standard=50))

# Amazon Reviews 2023 official benchmark files (kept as given)
for c in ("Video_Games", "Baby_Products"):
    base = RAW / f"Amazon2023-{c}"
    side = {}
    for l in gzip.open(base / f"meta_{c}.jsonl.gz"):
        m = json.loads(l)
        side[m["parent_asin"]] = {"main_category": m.get("main_category"),
                                  "categories": "|".join(m.get("categories") or []) or None, "store": m.get("store")}
    side = pd.DataFrame.from_dict(side, orient="index")
    for s in ("last_out_w_his", "timestamp_w_his"):
        parts = {p: pd.read_csv(base / s / f"{c}.{p}.csv.gz") for p in ("train", "valid", "test")}
        allr = pd.concat([v.assign(part=p) for p, v in parts.items()])
        df = allr.rename(columns={"user_id": "user", "parent_asin": "item", "timestamp": "ts"})[["user", "item", "ts"]]
        name = f"Amazon2023-{c}-{s.split('_w_')[0]}"
        finish(name, df, side, {"source": "Amazon Reviews 2023 official benchmark (Hou et al. 2024)"},
               {"split": s, "files": "official train / valid / test kept as given", "ranking": LOO["ranking"],
                "max_len_standard": 50})
        op = OUT / name / "official_rows.parquet"; allr.to_parquet(op, index=False)
        with open(OUT / name / "SHA256SUMS", "a") as f:
            f.write(hashlib.sha256(op.read_bytes()).hexdigest() + "  " + op.name + "\n")
