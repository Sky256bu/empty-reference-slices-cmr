#!/usr/bin/env bash
# stepE05_acdc_checkpoint_inventory.sh
# ====================================
# Purpose: take inventory of whether the checkpoints corresponding to the 63 formal ACDC runs (arm x N x seed) exist,#       and compute and archive hashes for the files that are found.#
# Warning, wording restriction (**must be preserved verbatim when recording results**):#   The hash this script produces is an anchor for "this file has not changed since this run",#   it is **not** proof that "these are the weights that produced the frozen result".#   The frozen ACDC record (stack_dice_acdc_clean.csv) contains no checkpoint column and no hash,#   so that proof **does not exist** in this project and cannot be manufactured after the fact.#
# Read-only: does not train, does not run inference, moves nothing, deletes nothing. Pure CPU.#
# Usage:#   bash stepE05_acdc_checkpoint_inventory.sh /root/autodl-tmp /root/autodl-tmp/design_phase_quarantine
set -uo pipefail

ROOT="${1:-/root/autodl-tmp}"
OUT="${2:-/root/autodl-tmp/design_phase_quarantine}"
mkdir -p "$OUT"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
LIST="$OUT/stepE05_ckpt_inventory_${STAMP}.tsv"
SUM="$OUT/stepE05_ckpt_summary_${STAMP}.txt"

echo "== stepE05 ACDC checkpoint inventory ==  root=$ROOT  utc=$STAMP" | tee "$SUM"
# ---- 1. all checkpoint filesmapfile -t CKPTS < <(find "$ROOT" -type f \
  \( -name '*.pt' -o -name '*.pth' -o -name '*.ckpt' -o -name '*.safetensors' \) 2>/dev/null | sort)
echo "total checkpoint files: ${#CKPTS[@]}" | tee -a "$SUM"
# ---- 2. bucket by path (mnms / acdc / other)printf 'path\tbucket\tsize_bytes\tmtime_utc\tarm\tN\tseed\tsha256\n' > "$LIST"
n_mnms=0; n_acdc=0; n_other=0
for f in "${CKPTS[@]}"; do
  low="${f,,}"
  if   [[ "$low" == *"/mnms"* ]]; then b="mnms2"; ((n_mnms++))
  elif [[ "$low" == *"acdc"* || "$low" == *"seg_students"* || "$low" == *"stack_dice"* ]]; then b="acdc"; ((n_acdc++))
  else b="other"; ((n_other++)); fi

    # try to parse arm / N / seed from the path (leave blank if it cannot be parsed; do not guess)  arm=$(sed -nE 's|.*[_/-](arm)?[_-]?([ABCI]\|Cneg)([_/.-].*)?$|\2|Ip' <<<"$f" | head -n1)
  Nv=$( sed -nE 's|.*[_/-]N[_-]?([0-9]+)([_/.-].*)?$|\1|Ip'            <<<"$f" | head -n1)
  sd=$( sed -nE 's|.*[_/-]s(eed)?[_-]?([0-9]+)([_/.-].*)?$|\2|Ip'      <<<"$f" | head -n1)

  sz=$(stat -c %s "$f" 2>/dev/null || echo "")
  mt=$(date -u -d "@$(stat -c %Y "$f" 2>/dev/null || echo 0)" +%Y-%m-%dT%H:%M:%SZ 2>/dev/null || echo "")

    # hash the acdc bucket only (the rest are merely registered, to avoid hashing all 692 files)  if [[ "$b" == "acdc" ]]; then h=$(sha256sum "$f" | awk '{print $1}'); else h=""; fi
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$f" "$b" "$sz" "$mt" "$arm" "$Nv" "$sd" "$h" >> "$LIST"
done

{
    echo "buckets: mnms2=$n_mnms  acdc=$n_acdc  other=$n_other"  echo
    echo "-- number of distinct (arm,N,seed) triples parsable from the acdc bucket --"  awk -F'\t' 'NR>1 && $2=="acdc" && $5!="" && $6!="" && $7!="" {print $5"_"$6"_"$7}' "$LIST" \
    | sort -u | tee "$OUT/stepE05_acdc_cells_${STAMP}.txt" | wc -l
  echo
    echo "-- reading (fixed in advance) --"    echo "  63/63 found -> the ACDC weights are complete on disk (still with no freeze-time hash anchor)"    echo "  fewer than 63 -> the missing part is permanently unregenerable and must be recorded in the ledger"    echo "  0 found     -> the ACDC-side inference route is technically impossible, adding one more ground to the Fig. 1 ruling"  echo
    echo "inventory: $LIST"} | tee -a "$SUM"
