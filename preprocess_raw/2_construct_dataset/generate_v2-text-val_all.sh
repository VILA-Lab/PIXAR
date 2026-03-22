#!/usr/bin/env bash
set -u  # 不用 -e，避免中途一个失败直接把整批停掉
set -o pipefail

cd /home/jiacheng/Omni_detection/PIXAR/utils_preprocess/construct_dataset || exit 1

# -------------------------
# Config
# -------------------------
ALL_TYPES=(gemini3 gpt flux2 gemini seedream qwen)

TAOS=(0.1)

DATASET_DIR="/data/thor/jiacheng/omni_backup/raw_outputs"
OUT_DIR="/data/ironman/jiacheng/final_Omni_Data/test/full"
DESCRIPTIONS_CSV="/home/jiacheng/Omni_detection/PIXAR/utils_preprocess/descriptions.csv"
MISSING_CSV_DIR="./logs/missing_text_logs"

# -------------------------
# Logging helpers
# -------------------------
RUN_ID="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="./logs"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/construct_unified_text_all_${RUN_ID}.log"

ts() { date +"%F %T"; }

h1() {
  echo -e "\n========================================" | tee -a "$LOG_FILE"
  echo -e "🚀 $1" | tee -a "$LOG_FILE"
  echo -e "========================================" | tee -a "$LOG_FILE"
}

h2() {
  echo -e "\n----------------------------------------" | tee -a "$LOG_FILE"
  echo -e "📌 $1" | tee -a "$LOG_FILE"
  echo -e "----------------------------------------" | tee -a "$LOG_FILE"
}

log() {
  echo "[$(ts)] $1" | tee -a "$LOG_FILE"
}

# -------------------------
# Runner
# -------------------------
OK=0
FAIL=0

run_one () {
  local id="$1"
  local tao="$2"
  local anno_flag="$3"  # "true" or "false"
  local bg_flag="$4"    # "true" or "false"
  local dest_type="$5"  # "train" or "validation"

  local cmd="python 2_construct_dataset_text.py --id \"$id\" --tao \"$tao\" --dataset-dir \"$DATASET_DIR\" --output-dir \"$OUT_DIR\" --dest-type \"$dest_type\" --descriptions-csv \"$DESCRIPTIONS_CSV\" --missing-csv-dir \"$MISSING_CSV_DIR\""

  [[ "$anno_flag" == "true" ]] && cmd="$cmd --anno"
  [[ "$bg_flag"  == "true" ]] && cmd="$cmd --bg"

  log "▶️  Start: id=${id} | tao=${tao} | anno=${anno_flag} | bg=${bg_flag} | dest=${dest_type}"
  local start_ts end_ts dur
  start_ts=$(date +%s)

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
    log "🧾 Tail(30) for failure: id=${id} tao=${tao}"
    tail -n 30 "$LOG_FILE" | sed 's/^/    /' | tee -a "$LOG_FILE"
  fi
}

# -------------------------
# Main
# -------------------------
h1 "Construct ALL Types (run_id=${RUN_ID})"
log "📂 workdir=$(pwd)"
log "📥 dataset_dir=${DATASET_DIR}"
log "📦 output_dir=${OUT_DIR}"
log "📄 descriptions_csv=${DESCRIPTIONS_CSV}"
log "📋 missing_csv_dir=${MISSING_CSV_DIR}"
log "🏷️  types=${ALL_TYPES[*]}"
log "🧪 taos=${TAOS[*]}"
log "📝 log_file=${LOG_FILE}"

for TYPE in "${ALL_TYPES[@]}"; do

  val_w_anno_ids=(
    "${TYPE}_coco_val_inter_replacement_1"
    "${TYPE}_coco_val_replacement_1"
  )
  [[ "$TYPE" == "qwen" ]] && val_w_anno_ids+=(
    "${TYPE}_coco_val_inter_replacement_2"
    "${TYPE}_coco_val_replacement_2"
  )
  val_w_anno_bg_ids=(
    "${TYPE}_coco_val_removal_1"
  )
  val_wo_anno_ids=(
    "${TYPE}_coco_val_addition"
    "${TYPE}_coco_val_color"
    "${TYPE}_coco_val_material"
    "${TYPE}_coco_val_motion"
  )
  val_wo_anno_bg_ids=(
    "${TYPE}_coco_val_background"
  )

  for tao in "${TAOS[@]}"; do
    h1 "TYPE=${TYPE} | TAO=${tao}"

    h2 "TYPE=${TYPE} | TAO=${tao} | w/ anno (--anno)"
    for id in "${val_w_anno_ids[@]}"; do
      run_one "$id" "$tao" "true" "false" "validation"
    done

    h2 "TYPE=${TYPE} | TAO=${tao} | w/o anno"
    for id in "${val_wo_anno_ids[@]}"; do
      run_one "$id" "$tao" "false" "false" "validation"
    done

    h2 "TYPE=${TYPE} | TAO=${tao} | w/ anno bg (--anno --bg)"
    for id in "${val_w_anno_bg_ids[@]}"; do
      run_one "$id" "$tao" "true" "true" "validation"
    done

    h2 "TYPE=${TYPE} | TAO=${tao} | w/o anno bg (--bg)"
    for id in "${val_wo_anno_bg_ids[@]}"; do
      run_one "$id" "$tao" "false" "true" "validation"
    done
  done

done

h1 "Summary"
log "📊 Total: $((OK + FAIL)) | ✅ OK=${OK} | ❌ FAIL=${FAIL}"
log "✅ Done. Log saved to: ${LOG_FILE}"

if [[ $FAIL -ne 0 ]]; then
  exit 1
fij