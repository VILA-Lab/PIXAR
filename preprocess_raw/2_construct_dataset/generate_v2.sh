#!/usr/bin/env bash
set -u  # 不用 -e，避免中途一个失败直接把整批停掉（更适合跑大批量）
set -o pipefail

cd /home/jiacheng/Omni_detection/PIXAR/utils_preprocess/construct_dataset || exit 1

# -------------------------
# Config
# -------------------------
DATASET_DIR="/data/thor/jiacheng/omni_backup/raw_outputs"
OUT_DIR="/data/thor/jiacheng/omni_backup/train/ours"
DESCRIPTIONS_CSV="/home/jiacheng/Omni_detection/PIXAR/utils_preprocess/descriptions_train.csv"

TAOS=(0.1)

# 控制是否处理 validation mock (注入到 trainset 目录)
PROCESS_VAL_MOCK=true  # 设置为 false 可以跳过 validation mock 处理

# ===== Train 数据集分组 =====
train_w_anno_ids=(
  coco_train_inter_replacement_1
  coco_train_inter_replacement_2
  coco_train_replacement_1
  coco_train_replacement_2
)

train_w_anno_bg_ids=(
  coco_train_removal_1
)

train_wo_anno_ids=(
  coco_train_addition
  coco_train_color
  coco_train_motion
  coco_train_material
)

train_wo_anno_bg_ids=(
  coco_train_background
)

# ===== Validation Mock 数据集分组 =====
# val_w_anno_ids=(
#   coco_val_inter_replacement_1
#   coco_val_inter_replacement_2
#   coco_val_replacement_1
#   coco_val_replacement_2
# )

# val_w_anno_bg_ids=(
#   coco_val_removal_1
# )

# val_wo_anno_ids=(
#   coco_val_addition
#   coco_val_color
#   coco_val_motion
#   coco_val_material
# )

val_wo_anno_bg_ids=(
  seedream_coco_val_background  # 如果需要可以取消注释
)

# -------------------------
# Logging helpers
# -------------------------
RUN_ID="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="./logs"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/construct_unified_text_${RUN_ID}.log"

ts() { date +"%F %T"; }

h1() {
  echo -e "\n========================================" | tee -a "$LOG_FILE"
  echo -e "🚀 [$(ts)] $1" | tee -a "$LOG_FILE"
  echo -e "========================================" | tee -a "$LOG_FILE"
}

h2() {
  echo -e "\n----------------------------------------" | tee -a "$LOG_FILE"
  echo -e "📌 [$(ts)] $1" | tee -a "$LOG_FILE"
  echo -e "----------------------------------------" | tee -a "$LOG_FILE"
}

log() {
  # usage: log "message"
  echo "[$(ts)] $1" | tee -a "$LOG_FILE"
}

# -------------------------
# Runner (使用统一脚本 2_construct_dataset_text.py)
# -------------------------
OK=0
FAIL=0

run_one () {
  local id="$1"
  local tao="$2"
  local anno_flag="$3"  # "true" or "false"
  local bg_flag="$4"    # "true" or "false"
  local dest_type="$5"  # "train" or "validation"

  # 构建参数 - 使用 2_construct_dataset_text.py
  local cmd="python 2_construct_dataset_text.py --id \"$id\" --tao \"$tao\" --dataset-dir \"$DATASET_DIR\" --output-dir \"$OUT_DIR\" --dest-type \"$dest_type\" --descriptions-csv \"$DESCRIPTIONS_CSV\""

  if [[ "$anno_flag" == "true" ]]; then
    cmd="$cmd --anno"
  fi

  if [[ "$bg_flag" == "true" ]]; then
    cmd="$cmd --bg"
  fi

  log "▶️  Start: id=${id} | tao=${tao} | anno=${anno_flag} | bg=${bg_flag} | dest=${dest_type}"
  local start_ts end_ts dur

  start_ts=$(date +%s)

  # 执行命令：stdout+stderr 全进 log，同时在终端显示
  eval "$cmd" 2>&1 | tee -a "$LOG_FILE"

  local rc=${PIPESTATUS[0]}
  end_ts=$(date +%s)
  dur=$((end_ts - start_ts))

  if [[ $rc -eq 0 ]]; then
    OK=$((OK + 1))
    log "✅ Done: id=${id} tao=${tao} dest=${dest_type} (${dur}s)"
  else
    FAIL=$((FAIL + 1))
    log "❌ Failed(rc=${rc}): id=${id} tao=${tao} dest=${dest_type} (${dur}s)"
    # 打印最后 30 行，方便快速定位（只输出到终端 & log）
    log "🧾 Tail(30) for failure: id=${id} tao=${tao}"
    tail -n 30 "$LOG_FILE" | sed 's/^/    /' | tee -a "$LOG_FILE"
  fi
}

# -------------------------
# Main
# -------------------------
h1 "Construct Dataset Batch (run_id=${RUN_ID}) - Using Unified Script with Text Descriptions"
log "📂 workdir=$(pwd)"
log "📥 dataset_dir=${DATASET_DIR}"
log "📦 output_dir=${OUT_DIR}"
log "📄 descriptions_csv=${DESCRIPTIONS_CSV}"
log "🧪 taos=${TAOS[*]}"
log "📝 log_file=${LOG_FILE}"
log "🔧 Using unified script: 2_construct_dataset_text.py"
log "🔧 Process validation mock: ${PROCESS_VAL_MOCK}"

for tao in "${TAOS[@]}"; do
  # ========== Process Train Data ==========
  h1 "TAO=${tao} | Processing TRAIN data"

  h2 "TAO=${tao} | Train w/ anno (--anno)"
  for id in "${train_w_anno_ids[@]}"; do
    run_one "$id" "$tao" "true" "false" "train"
  done

  h2 "TAO=${tao} | Train w/o anno"
  for id in "${train_wo_anno_ids[@]}"; do
    run_one "$id" "$tao" "false" "false" "train"
  done

  h2 "TAO=${tao} | Train w/ anno bg (--anno --bg)"
  for id in "${train_w_anno_bg_ids[@]}"; do
    run_one "$id" "$tao" "true" "true" "train"
  done

  h2 "TAO=${tao} | Train w/o anno bg (--bg)"
  for id in "${train_wo_anno_bg_ids[@]}"; do
    run_one "$id" "$tao" "false" "true" "train"
  done

  # ========== Process Validation Mock Data (注入到同一个输出目录) ==========
  if [[ "$PROCESS_VAL_MOCK" == "true" ]]; then
    h1 "TAO=${tao} | Processing VALIDATION MOCK data (injected into trainset)"

    h2 "TAO=${tao} | Validation w/o anno bg (--bg)"
    for id in "${val_wo_anno_bg_ids[@]}"; do
      run_one "$id" "$tao" "false" "true" "validation"
    done
  else
    log "⏭️  Skipping validation mock processing (PROCESS_VAL_MOCK=false)"
  fi
done

h1 "Summary"
log "📊 Total: $((OK + FAIL)) | ✅ OK=${OK} | ❌ FAIL=${FAIL}"
log "✅ Done. Log saved to: ${LOG_FILE}"

# 如果你想：失败就让脚本最后返回非 0（用于 CI 或上层监控）
if [[ $FAIL -ne 0 ]]; then
  exit 1
fi