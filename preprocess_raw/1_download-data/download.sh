#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'

#####################################
#            用户配置区
#####################################

# files.txt 路径（每行一个 zip 文件名）
FILES_TXT="./files.txt"

# rclone 配置名
REMOTE="gdrive"

# Google Drive 上的目录
REMOTE_BASE="ImageDetection_Benchmark/final_bench_raw"

# 解压后的根目录
OUT_DIR="/data/thor/jiacheng/omni_backup/raw_outputs"

# 本地临时存放 zip 的目录
DOWNLOAD_DIR="/data/thor/jiacheng/omni_backup/temp"

#####################################
#            初始化检查
#####################################

mkdir -p "$OUT_DIR"
mkdir -p "$DOWNLOAD_DIR"

command -v rclone >/dev/null 2>&1 || { echo "❌ 未找到 rclone"; exit 1; }
command -v unzip  >/dev/null 2>&1 || { echo "❌ 未找到 unzip"; exit 1; }

echo "📄 文件列表: $FILES_TXT"
echo "☁️  远端路径: ${REMOTE}:${REMOTE_BASE}"
echo "📦 zip 临时目录: $DOWNLOAD_DIR"
echo "📂 解压输出目录: $OUT_DIR"
echo "========================================"

#####################################
#            主下载 + 解压逻辑
#####################################

while IFS= read -r zip_file || [[ -n "$zip_file" ]]; do
  # 去空白
  zip_file="$(echo "$zip_file" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')"
  [[ -z "$zip_file" ]] && continue

  # 只处理 .zip
  if [[ "$zip_file" != *.zip ]]; then
    echo "⚠️ 跳过非 zip 行: $zip_file"
    continue
  fi

  base_name="${zip_file%.zip}"
  dest_dir="${OUT_DIR}"
  local_zip="${DOWNLOAD_DIR}/${zip_file}"
  remote_path="${REMOTE}:${REMOTE_BASE}/${zip_file}"

  echo "🚀 下载: $remote_path"
  rclone copy "$remote_path" "$DOWNLOAD_DIR" -P

  if [[ ! -f "$local_zip" ]]; then
    echo "❌ 下载失败，未找到文件: $local_zip"
    exit 1
  fi

  echo "📦 解压: $zip_file -> $dest_dir"
  mkdir -p "$dest_dir"

  if unzip -oq "$local_zip" -d "$dest_dir"; then
    echo "✅ 解压完成: $zip_file"
    rm -f "$local_zip"
  else
    echo "❌ 解压失败（保留 zip）：$local_zip"
    exit 1
  fi

  echo "----------------------------------------"
done < "$FILES_TXT"

echo "🎉 所有文件下载并解压完成"
