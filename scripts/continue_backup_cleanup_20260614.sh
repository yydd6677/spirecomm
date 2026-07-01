#!/usr/bin/env bash
set -euo pipefail

cd /home/yydd/spirecomm

BAIDUPCS_BIN="${BAIDUPCS_BIN:-/home/yydd/.local/bin/BaiduPCS-Go}"
REMOTE_ROOT="${REMOTE_ROOT:-/spirecomm_backup_20260614}"
LOG="${LOG:-_cache/continue_backup_cleanup_20260614.log}"
TEACHER_DIR="data/v3_combat_teacher/v5_best_teacher_v12_norandom_200k"
REMOTE_TEACHER="$REMOTE_ROOT/data/v3_combat_teacher/v5_best_teacher_v12_norandom_200k"

mkdir -p "$(dirname "$LOG")"

log() {
  printf '[%(%Y-%m-%d %H:%M:%S)T] %s\n' -1 "$*" | tee -a "$LOG"
}

run_pcs() {
  env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \
      -u http_proxy -u https_proxy -u all_proxy \
      "$BAIDUPCS_BIN" "$@"
}

local_size_bytes() {
  stat -c '%s' "$1"
}

remote_size_bytes() {
  local remote_path="$1"
  run_pcs meta "$remote_path" | awk '
    $1 == "文件大小" {
      value = $2
      unit = $3
      gsub(/,/, "", value)
      if (unit == "B") mult = 1
      else if (unit == "KB") mult = 1024
      else if (unit == "MB") mult = 1024 * 1024
      else if (unit == "GB") mult = 1024 * 1024 * 1024
      else mult = 1
      printf "%.0f\n", value * mult
      found = 1
    }
    END { if (!found) exit 1 }
  '
}

wait_for_teacher_upload() {
  if [[ ! -d "$TEACHER_DIR" ]]; then
    log "skip waiting for teacher upload: local dir already removed: $TEACHER_DIR"
    return 0
  fi
  log "waiting for teacher upload process to finish"
  while pgrep -f "[B]aiduPCS-Go upload ${TEACHER_DIR}|[B]aiduPCS-Go upload ${TEACHER_DIR}/shard_|[u]pload_teacher_shards_parallel_20260614" >/dev/null; do
    sleep 120
    run_pcs ls "$REMOTE_TEACHER" | tail -n 4 | tee -a "$LOG" || true
  done
  log "teacher upload process is no longer running"
}

verify_teacher_remote() {
  local local_list remote_list
  if [[ ! -d "$TEACHER_DIR" ]]; then
    log "skip teacher verification: local dir already removed: $TEACHER_DIR"
    return 0
  fi
  local_list="$(mktemp)"
  remote_list="$(mktemp)"
  find "$TEACHER_DIR" -maxdepth 1 -type f -printf '%f\n' | sort > "$local_list"
  run_pcs ls "$REMOTE_TEACHER" | awk '/^[[:space:]]*[0-9]+[[:space:]]/ {print $NF}' | sort > "$remote_list"
  log "teacher counts: local=$(wc -l < "$local_list"), remote=$(wc -l < "$remote_list")"
  if ! diff -u "$local_list" "$remote_list" >> "$LOG"; then
    rm -f "$local_list" "$remote_list"
    return 1
  fi
  rm -f "$local_list" "$remote_list"
}

delete_teacher_local() {
  if [[ ! -d "$TEACHER_DIR" ]]; then
    log "skip deleting teacher dir: already removed: $TEACHER_DIR"
    return 0
  fi
  log "deleting verified local teacher dir: $TEACHER_DIR"
  rm -rf "$TEACHER_DIR"
  df -h /home/yydd/spirecomm | tee -a "$LOG"
}

upload_verify_delete_archive() {
  local archive="$1"
  local remote_dir="$REMOTE_ROOT/data/v3_combat_teacher/"
  local remote_path="$remote_dir$(basename "$archive")"
  local local_bytes remote_bytes

  [[ -f "$archive" ]] || return 0
  log "upload archive: $archive"
  run_pcs upload "$archive" "$remote_dir"
  local_bytes="$(local_size_bytes "$archive")"
  remote_bytes="$(remote_size_bytes "$remote_path")"
  log "archive size check $(basename "$archive"): local=${local_bytes}, remote=${remote_bytes}"
  [[ "$local_bytes" == "$remote_bytes" ]]
  log "deleting verified archive: $archive"
  rm -f "$archive"
  df -h /home/yydd/spirecomm | tee -a "$LOG"
}

run_safe_git_gc() {
  local free_kib
  free_kib="$(df -Pk /home/yydd/spirecomm | awk 'NR==2 {print $4}')"
  log "free space before git gc: ${free_kib} KiB"
  if (( free_kib < 62914560 )); then
    log "skip git gc: less than 60GiB free"
    return 0
  fi
  log "git count before gc"
  git count-objects -vH | tee -a "$LOG"
  log "running git gc --prune=now"
  git gc --prune=now
  log "git count after gc"
  git count-objects -vH | tee -a "$LOG"
  du -sh .git | tee -a "$LOG"
  df -h /home/yydd/spirecomm | tee -a "$LOG"
}

resume_remaining_workspace_upload() {
  log "resuming full workspace upload script"
  bash scripts/upload_spirecomm_baidupcs_batches.sh
}

main() {
  log "continue backup cleanup started"
  wait_for_teacher_upload
  verify_teacher_remote
  delete_teacher_local

  shopt -s nullglob
  for archive in data/v3_combat_teacher/*.zip data/v3_combat_teacher/*.zst; do
    upload_verify_delete_archive "$archive"
  done

  run_safe_git_gc
  resume_remaining_workspace_upload
  log "continue backup cleanup finished"
}

main "$@"
