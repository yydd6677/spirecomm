#!/usr/bin/env bash
set -euo pipefail

cd /home/yydd/spirecomm

BAIDUPCS_BIN="${BAIDUPCS_BIN:-/home/yydd/.local/bin/BaiduPCS-Go}"
REMOTE_DIR="${REMOTE_DIR:-/spirecomm_backup_20260614/data/v3_combat_teacher/v5_best_teacher_v12_norandom_200k/}"
TEACHER_DIR="${TEACHER_DIR:-data/v3_combat_teacher/v5_best_teacher_v12_norandom_200k}"
JOBS="${JOBS:-8}"
LOG_DIR="${LOG_DIR:-_cache/teacher_parallel_upload_logs_20260614}"

mkdir -p "$LOG_DIR"

run_one() {
  local file="$1"
  local name
  name="$(basename "$file")"
  env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \
      -u http_proxy -u https_proxy -u all_proxy \
      "$BAIDUPCS_BIN" upload "$file" "$REMOTE_DIR" \
      >"$LOG_DIR/${name}.log" 2>&1
}

export BAIDUPCS_BIN REMOTE_DIR LOG_DIR
export -f run_one

find "$TEACHER_DIR" -maxdepth 1 -type f -name 'shard_*.pt' -print0 \
  | sort -z \
  | xargs -0 -n 1 -P "$JOBS" bash -c 'run_one "$0"'
