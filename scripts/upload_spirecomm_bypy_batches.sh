#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/home/yydd/miniforge3/envs/spirecomm-rl/bin/python}"
REMOTE_ROOT="${REMOTE_ROOT:-spirecomm-backup-20260614}"

cd /home/yydd/spirecomm

upload_path() {
  local local_path="$1"
  local remote_path="$2"
  if [[ ! -e "$local_path" ]]; then
    echo "skip missing: $local_path" >&2
    return 0
  fi
  echo "upload: $local_path -> $remote_path"
  "$PYTHON_BIN" -m bypy upload "$local_path" "$remote_path"
}

upload_path "_cache/eval_v3_vs_real_20260614" "$REMOTE_ROOT/_cache/eval_v3_vs_real_20260614"

find . -maxdepth 1 -type f -print0 | sort -z | while IFS= read -r -d '' file; do
  upload_path "${file#./}" "$REMOTE_ROOT/${file#./}"
done

upload_path "spirecomm" "$REMOTE_ROOT/spirecomm"
upload_path "tests" "$REMOTE_ROOT/tests"
upload_path "scripts" "$REMOTE_ROOT/scripts"
upload_path "tools" "$REMOTE_ROOT/tools"
upload_path "configs" "$REMOTE_ROOT/configs"
upload_path "docs" "$REMOTE_ROOT/docs"
upload_path "models" "$REMOTE_ROOT/models"
upload_path "data/v3_combat_tensor" "$REMOTE_ROOT/data/v3_combat_tensor"
upload_path "data/v3_combat_teacher/v5_best_teacher_v12_norandom_200k" "$REMOTE_ROOT/data/v3_combat_teacher/v5_best_teacher_v12_norandom_200k"

shopt -s nullglob
for archive in data/v3_combat_teacher/*.zip data/v3_combat_teacher/*.zst; do
  upload_path "$archive" "$REMOTE_ROOT/data/v3_combat_teacher/$(basename "$archive")"
done
