#!/bin/bash
# BM-01b: Taobao UserBehavior from the Kaggle copy (marwa80/userbehavior); accepted only if it matches the official counts
B=/home/jovyan/ahmedabdelhamed01/Privacy-Preserving-Shopping-Intelligence/.local_ai/ppsi_v4/bench_raw/taobao; mkdir -p $B; cd $B
LOG=$B/TAOBAO.log; ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }
[ -s archive.zip ] || { curl -sSL --retry 5 --fail -o archive.zip.part "https://www.kaggle.com/api/v1/datasets/download/marwa80/userbehavior" && mv archive.zip.part archive.zip; }
echo "$(ts) ZIP $(stat -c %s archive.zip) $(sha256sum archive.zip | cut -c1-16)" >> $LOG
unzip -l archive.zip >> $LOG; unzip -o -q archive.zip -d unzipped && echo "$(ts) UNZIPPED" >> $LOG
F=$(find unzipped -iname "UserBehavior.csv" | head -1); echo "$(ts) FILE $F $(stat -c %s "$F")" >> $LOG
awk -F, '{n++; u[$1]; i[$2]; c[$3]; b[$4]} END {printf "COUNTS rows=%d users=%d items=%d cats=%d behaviors=%d\n", n, length(u), length(i), length(c), length(b)}' "$F" >> $LOG
sha256sum "$F" >> $LOG
grep -q "rows=100150807 users=987994 items=4162024" $LOG && echo "$(ts) MATCHES_OFFICIAL" >> $LOG || echo "$(ts) MISMATCH_OFFICIAL" >> $LOG
