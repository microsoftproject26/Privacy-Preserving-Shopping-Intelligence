#!/bin/bash
# fetch pure / cp311 wheels for missing modules (lock versions) into a separate site dir; never touches ppsi-gpu
T=/home/jovyan/venvs/bench-probes-site; W=$T/_wheels; mkdir -p $W; P=/home/jovyan/venvs/ppsi-gpu/bin/python
REPO=/home/jovyan/ahmedabdelhamed01/Privacy-Preserving-Shopping-Intelligence/.local_ai/ppsi_v4/bench_raw/spotify-sequential-capacity-probes
LOCK=$REPO/requirements-lock.txt
pkg() { case $1 in attr) echo attrs;; yaml) echo PyYAML;; pydantic_core) echo pydantic-core;; typing_extensions) echo typing-extensions;; dateutil) echo python-dateutil;; *) echo $1;; esac; }
getw() { u=$(timeout 30 curl -s https://pypi.org/pypi/$1/$2/json | python3 -c "
import json,sys; d=json.load(sys.stdin); c=[f for f in d.get('urls',[]) if f['packagetype']=='bdist_wheel' and ('py3-none-any' in f['filename'] or ('cp311-cp311-manylinux' in f['filename'] and 'x86_64' in f['filename']))]
print(c[0]['url'] if c else '')" 2>/dev/null); [ -n "$u" ] || { echo "NO_WHEEL $1 $2"; return 1; }
  (cd $W && timeout 60 curl -sSLO "$u") && $P -c "import zipfile,sys; zipfile.ZipFile(sys.argv[1]).extractall(sys.argv[2])" "$W/$(basename $u)" $T && echo "GOT $(basename $u)"; }
for i in $(seq 1 15); do
  m=$(cd $REPO/.external/transformer_benchmark && PYTHONPATH=$T:. $P -c "import rectools, rectools.dataset, rectools.dataset.interactions; import src.datasets.common" 2>&1 | grep -o "No module named [^ ]*" | head -1 | sed "s/No module named //; s/[\x27\"]//g; s/\..*//")
  [ -z "$m" ] && { echo IMPORT_CHAIN_OK; exit 0; }
  p=$(pkg $m); v=$(grep -iE "^${p//-/[-_]}==" $LOCK | head -1 | cut -d= -f3); echo "missing $m -> $p ${v:-?}"
  [ -n "$v" ] || v=$(timeout 30 curl -s https://pypi.org/pypi/$p/json | python3 -c "import json,sys;print(json.load(sys.stdin)['info']['version'])")
  getw $p $v || exit 3
done; echo GAVE_UP
