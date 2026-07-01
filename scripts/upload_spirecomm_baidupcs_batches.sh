#!/usr/bin/env bash
set -euo pipefail

BAIDUPCS_BIN="${BAIDUPCS_BIN:-/home/yydd/.local/bin/BaiduPCS-Go}"
REMOTE_ROOT="${REMOTE_ROOT:-/spirecomm_backup_20260614}"

cd /home/yydd/spirecomm

run_pcs() {
  env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \
      -u http_proxy -u https_proxy -u all_proxy \
      "$BAIDUPCS_BIN" "$@"
}

upload_path() {
  local local_path="$1"
  local remote_path="$2"
  if [[ ! -e "$local_path" ]]; then
    echo "skip missing: $local_path" >&2
    return 0
  fi
  echo "upload: $local_path -> $remote_path"
  run_pcs upload "$local_path" "$remote_path"
}

run_pcs mkdir "$REMOTE_ROOT" || true

upload_path "_cache/eval_v3_vs_real_20260614" "$REMOTE_ROOT/_cache/"

find . -maxdepth 1 -type f -print0 | sort -z | while IFS= read -r -d '' file; do
  upload_path "${file#./}" "$REMOTE_ROOT/"
done

upload_path "spirecomm" "$REMOTE_ROOT/"
upload_path "tests" "$REMOTE_ROOT/"
upload_path "scripts" "$REMOTE_ROOT/"
upload_path "tools" "$REMOTE_ROOT/"
upload_path "configs" "$REMOTE_ROOT/"
upload_path "docs" "$REMOTE_ROOT/"
upload_path "models" "$REMOTE_ROOT/"
upload_path "data/v3_combat_tensor" "$REMOTE_ROOT/data/"
upload_path "data/v3_combat_teacher/v5_best_teacher_v12_norandom_200k" "$REMOTE_ROOT/data/v3_combat_teacher/"

shopt -s nullglob
for archive in data/v3_combat_teacher/*.zip data/v3_combat_teacher/*.zst; do
  upload_path "$archive" "$REMOTE_ROOT/data/v3_combat_teacher/"
done
