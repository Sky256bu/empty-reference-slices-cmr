#!/usr/bin/env bash
# stepE05_acdc_checkpoint_inventory.sh
# ====================================
# 目的：清点 ACDC 63 个正式 run (arm × N × seed) 对应的 checkpoint 是否存在，
#       并对命中的文件**计算哈希存档**。
#
# ⚠️ 措辞限定（**结果记录时必须原样保留**）：
#   本脚本产生的哈希是「**自本次运行起该文件未再变动**」的锚点，
#   **不是**「这就是产出冻结结果的那个权重」的证明。
#   ACDC 的冻结记录（stack_dice_acdc_clean.csv）不含 checkpoint 列与哈希，
#   该证明在本项目中**不存在**，且不可事后补造。
#
# 只读：不训练、不推理、不移动、不删除任何文件。纯 CPU。
#
# 用法：
#   bash stepE05_acdc_checkpoint_inventory.sh /root/autodl-tmp /root/autodl-tmp/design_phase_quarantine
set -uo pipefail

ROOT="${1:-/root/autodl-tmp}"
OUT="${2:-/root/autodl-tmp/design_phase_quarantine}"
mkdir -p "$OUT"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
LIST="$OUT/stepE05_ckpt_inventory_${STAMP}.tsv"
SUM="$OUT/stepE05_ckpt_summary_${STAMP}.txt"

echo "== stepE05 ACDC checkpoint 清点 ==  root=$ROOT  utc=$STAMP" | tee "$SUM"

# ---- 1. 全部 checkpoint 文件
mapfile -t CKPTS < <(find "$ROOT" -type f \
  \( -name '*.pt' -o -name '*.pth' -o -name '*.ckpt' -o -name '*.safetensors' \) 2>/dev/null | sort)
echo "checkpoint 文件总数: ${#CKPTS[@]}" | tee -a "$SUM"

# ---- 2. 按路径归属分桶（mnms / acdc / 其它）
printf 'path\tbucket\tsize_bytes\tmtime_utc\tarm\tN\tseed\tsha256\n' > "$LIST"
n_mnms=0; n_acdc=0; n_other=0
for f in "${CKPTS[@]}"; do
  low="${f,,}"
  if   [[ "$low" == *"/mnms"* ]]; then b="mnms2"; ((n_mnms++))
  elif [[ "$low" == *"acdc"* || "$low" == *"seg_students"* || "$low" == *"stack_dice"* ]]; then b="acdc"; ((n_acdc++))
  else b="other"; ((n_other++)); fi

  # 从路径尝试解析 arm / N / seed（解析不出留空，不猜）
  arm=$(sed -nE 's|.*[_/-](arm)?[_-]?([ABCI]\|Cneg)([_/.-].*)?$|\2|Ip' <<<"$f" | head -n1)
  Nv=$( sed -nE 's|.*[_/-]N[_-]?([0-9]+)([_/.-].*)?$|\1|Ip'            <<<"$f" | head -n1)
  sd=$( sed -nE 's|.*[_/-]s(eed)?[_-]?([0-9]+)([_/.-].*)?$|\2|Ip'      <<<"$f" | head -n1)

  sz=$(stat -c %s "$f" 2>/dev/null || echo "")
  mt=$(date -u -d "@$(stat -c %Y "$f" 2>/dev/null || echo 0)" +%Y-%m-%dT%H:%M:%SZ 2>/dev/null || echo "")

  # 只对 acdc 桶算哈希（其余仅登记，避免对 692 个文件全量哈希耗时）
  if [[ "$b" == "acdc" ]]; then h=$(sha256sum "$f" | awk '{print $1}'); else h=""; fi
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$f" "$b" "$sz" "$mt" "$arm" "$Nv" "$sd" "$h" >> "$LIST"
done

{
  echo "分桶: mnms2=$n_mnms  acdc=$n_acdc  other=$n_other"
  echo
  echo "-- acdc 桶中可解析出 (arm,N,seed) 三元组的唯一组合数 --"
  awk -F'\t' 'NR>1 && $2=="acdc" && $5!="" && $6!="" && $7!="" {print $5"_"$6"_"$7}' "$LIST" \
    | sort -u | tee "$OUT/stepE05_acdc_cells_${STAMP}.txt" | wc -l
  echo
  echo "-- 判读（预先写死）--"
  echo "  命中 63/63 → ACDC 权重在磁盘上齐全（仍无冻结期哈希锚点）"
  echo "  命中 < 63  → 缺失部分为永久不可重生成，须记入台账"
  echo "  命中 0     → ACDC 侧重推理路线在技术上即不成立，⟦Fig.1 决议⟧ 的否决再加一条"
  echo
  echo "清单: $LIST"
} | tee -a "$SUM"
