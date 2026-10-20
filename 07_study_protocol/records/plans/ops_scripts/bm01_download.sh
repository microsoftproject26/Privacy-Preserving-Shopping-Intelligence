#!/bin/bash
# BM-01: download benchmark raw files (no GPU); write-once dir; sha256 manifest at the end
B=/home/jovyan/ahmedabdelhamed01/Privacy-Preserving-Shopping-Intelligence/.local_ai/ppsi_v4/bench_raw; mkdir -p $B; cd $B
LOG=$B/DOWNLOAD.log; ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }
get() { mkdir -p "$(dirname "$1")"; [ -s "$1" ] && { echo "$(ts) SKIP $1" >> $LOG; return; }
  curl -sSL --retry 5 --fail -o "$1.part" "$2" && mv "$1.part" "$1" && echo "$(ts) OK $1 $(stat -c %s "$1")" >> $LOG || echo "$(ts) FAIL $1 $2" >> $LOG; }
get ml-1m/ml-1m.zip https://files.grouplens.org/datasets/movielens/ml-1m.zip
get amazon2014/reviews_Beauty_5.json.gz https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Beauty_5.json.gz
get amazon2014/meta_Beauty.json.gz https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/meta_Beauty.json.gz
A=https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023
for c in All_Beauty Video_Games Baby_Products; do
  for s in last_out_w_his timestamp_w_his; do for p in train valid test; do
    get amazon2023/benchmark/5core/$s/$c.$p.csv.gz $A/benchmark/5core/$s/$c.$p.csv.gz; done; done
  get amazon2023/meta/meta_$c.jsonl.gz $A/raw/meta_categories/meta_$c.jsonl.gz
done
find . -type f ! -name "*.part" ! -name DOWNLOAD.log ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS
echo "$(ts) DONE $(wc -l < SHA256SUMS) files" >> $LOG
