set -euo pipefail
V=/home/jovyan/ahmedabdelhamed01/Privacy-Preserving-Shopping-Intelligence/.local_ai/ppsi_v4
for S in 2026 2027; do
  for ARM in BIG SMALL; do
    if [ "$ARM" = BIG ]; then RUN=$V/runs_p3/SASREC/02B_SASREC_s$S; else RUN=$V/runs_ds/C_SMALL_s$S; fi
    for KIND in best endpoint_6.0; do
      P=$RUN/sealed/checkpoints/$KIND.pt
      test -f "$P"
      printf '%s s%s %s ' "$ARM" "$S" "$KIND"
      sha256sum "$P" | cut -d' ' -f1
    done
  done
done
